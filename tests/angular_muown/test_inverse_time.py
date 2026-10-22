# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small CPU checks, outside the unit suite's distributed/GPU fixtures.

Run with: python -m pytest tests/angular_muown/test_inverse_time.py -q
"""

import argparse
import io
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelAngularMuown,
    TensorParallelAngularMuownSimplified,
    _create_emerging_optimizer,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training import training
from megatron.training.arguments import add_megatron_arguments, validate_args


@pytest.fixture(autouse=True)
def eager_cpu():
    """Exercise the update equations without compiling kernels on CPU."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _spec(**overrides):
    return dict(c=2.0, global_batch_size=8, warmup_samples=8, total_samples=80) | overrides


def _scheduler(spec=None, optimizer=None, **overrides):
    kwargs = dict(
        optimizer=optimizer or SimpleNamespace(param_groups=[{"u_lr": None}, {}]),
        init_lr=0,
        max_lr=0.03,
        min_lr=0.003,
        lr_warmup_steps=8,
        lr_decay_steps=80,
        lr_decay_style="linear",
        start_wd=0,
        end_wd=0,
        wd_incr_steps=80,
        wd_incr_style="constant",
        angular_muown_schedule=spec,
    )
    kwargs.update(overrides)
    return OptimizerParamScheduler(**kwargs)


def _optimizer(params=None, *, original=False, **kwargs):
    weight = torch.tensor([[1.0, 0.2, -0.3, 0.4], [-0.1, 2.0, 0.3, 0.2], [0.2, -0.5, 1.0, 0.3]])
    optimizer_cls = TensorParallelAngularMuown if original else TensorParallelAngularMuownSimplified
    if not original:
        kwargs.setdefault("u_lr", 0.0)
    return optimizer_cls(
        [torch.nn.Parameter(weight * scale) for scale in (1.0, 1.1)] if params is None else params,
        split_qkv=False,
        fp32_matmul_prec="highest",
        **kwargs,
    )


def _gradients(optimizer, step):
    for index, param in enumerate(optimizer.param_groups[0]["params"]):
        param.grad = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param) * 0.01
        param.grad.add_(0.1 * (step + 1) + index * 0.03)


def _assert_equal(left, right, *, exact=True):
    tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=2e-5, atol=2e-6)
    for p, q in zip(left.param_groups[0]["params"], right.param_groups[0]["params"]):
        torch.testing.assert_close(p, q, **tolerance)
        assert left.state[p].keys() == right.state[q].keys()
        for key, value in left.state[p].items():
            torch.testing.assert_close(value, right.state[q][key], **tolerance)


@pytest.mark.parametrize("cooldown", [0, 0.2])
def test_warmup_tail_and_cooldown_boundaries(cooldown):
    """Check schedule boundaries and leave the ordinary LR schedule unchanged."""
    scheduler = _scheduler(_spec(cooldown_fraction=cooldown))
    original = _scheduler()
    points = [(0, 0), (4, 1), (8, 2), (16, 1), (64, 0.25)]
    points += (
        [(72, 0.25 * 0.525), (80, 0.0125), (160, 0.0125)] if cooldown else [(80, 0.2), (160, 0.1)]
    )
    for samples, expected in points:
        for current in (scheduler, original):
            current.step(samples - current.num_steps)
        assert scheduler.get_angular_lr() == pytest.approx(expected)
        assert scheduler.optimizer.param_groups[0]["u_lr"] == pytest.approx(expected)
        assert "u_lr" not in scheduler.optimizer.param_groups[1]
        assert scheduler.optimizer.param_groups[0]["lr"] == original.optimizer.param_groups[0]["lr"]
    assert original.get_angular_lr() is None
    assert "angular_muown_schedule" not in original.state_dict()


@pytest.mark.parametrize("coefficient", [0, 2])
def test_zero_coefficient_and_no_warmup(coefficient):
    """Allow zero angular rate and initialize the reciprocal tail without warmup."""
    scheduler = _scheduler(_spec(c=coefficient, warmup_samples=0, cooldown_fraction=0))
    assert scheduler.get_angular_lr() == coefficient
    scheduler.step(8)
    assert scheduler.get_angular_lr() == coefficient / 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("c", float("nan")),
        ("c", -1),
        ("global_batch_size", 0),
        ("warmup_samples", 64),
        ("cooldown_fraction", 1),
        ("cooldown_end_ratio", 1.1),
    ],
)
def test_invalid_schedule(field, value):
    """Reject invalid schedule inputs before updating optimizer groups."""
    spec = _spec()
    spec[field] = value
    with pytest.raises(ValueError):
        _scheduler(spec)


def test_checkpoint_policy_and_fixed_batch_guards():
    """Restore the saved clock once and enforce checkpoint and batch policies."""
    original = _scheduler(_spec())
    original.step(24)
    saved = original.state_dict()
    adopted = _scheduler(_spec(c=3))
    for _ in range(2):
        adopted.load_state_dict(saved)
        assert adopted.state_dict() == saved
        assert adopted.get_angular_lr() == pytest.approx(2 / 3)
    with pytest.raises(AssertionError):
        _scheduler(_spec(c=3), use_checkpoint_opt_param_scheduler=False).load_state_dict(saved)
    overridden = _scheduler(
        _spec(c=3), use_checkpoint_opt_param_scheduler=False, override_opt_param_scheduler=True
    )
    overridden.load_state_dict(saved)
    assert overridden.get_angular_lr() == 1.0
    for override in (False, True):
        for spec in (None, _spec(global_batch_size=4)):
            with pytest.raises(ValueError):
                _scheduler(
                    spec,
                    override_opt_param_scheduler=override,
                    use_checkpoint_opt_param_scheduler=not override,
                ).load_state_dict(saved)
        with pytest.raises(ValueError, match="Cannot switch"):
            _scheduler(
                _spec(),
                override_opt_param_scheduler=override,
                use_checkpoint_opt_param_scheduler=not override,
            ).load_state_dict(_scheduler().state_dict())


@pytest.mark.parametrize("batched", [False, True])
def test_absolute_angular_rate_keeps_adam_independent(batched):
    """Compare directions with the update equations and row norms with Adam."""
    optimizer = _optimizer(u_lr=0.15, batched_step=batched)
    group = optimizer.param_groups[0]
    params = group["params"]
    radii = [torch.nn.Parameter(p.detach().norm(dim=-1, keepdim=True)) for p in params]
    adam = torch.optim.Adam(radii, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    for step, (lr, u_lr) in enumerate(((0.0, 0.15), (0.02, 0.0), (0.003, 0.08))):
        group["lr"] = adam.param_groups[0]["lr"] = lr
        group["u_lr"] = u_lr
        _gradients(optimizer, step)
        optimizer._init_group(group)
        directions = []
        with torch.no_grad():
            for p, g in zip(params, radii):
                u = p / g
                g.grad = (p.grad * u).sum(dim=-1, keepdim=True)
                grad_u = g * (p.grad - u * g.grad)
                momentum = group["momentum"] * optimizer.state[p]["m_u"] + grad_u
                proposal = u - u_lr * optimizer.orthogonalize(
                    p, grad_u + group["momentum"] * momentum
                )
                directions.append(proposal / proposal.norm(dim=-1, keepdim=True))
        adam.step()
        optimizer.step()
        for p, g, direction in zip(params, radii, directions):
            torch.testing.assert_close(optimizer.state[p]["g"], g, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(p, g * direction, rtol=2e-5, atol=2e-6)
            assert optimizer.state[p]["step"] == step + 1


@pytest.mark.parametrize("batched", [False, True])
def test_original_and_batched_update_parity(batched):
    """Match the original optimizer when supplied with its effective angular rate."""
    original = _optimizer(original=True, u_decay_scale=0.7, batched_step=False)
    absolute = _optimizer(u_lr=0.0, batched_step=batched)
    for step, lr in enumerate((0.04, 0.02, 0.01)):
        absolute.param_groups[0]["u_lr"] = lr / (1 + 0.7 * step)
        for optimizer in (original, absolute):
            optimizer.param_groups[0]["lr"] = lr
            _gradients(optimizer, step)
            optimizer.step()
        _assert_equal(original, absolute, exact=not batched)


@pytest.mark.parametrize("batched", [False, True])
def test_serialized_optimizer_and_scheduler_resume(batched):
    """Require exact continuation after serializing optimizer and scheduler state."""
    uninterrupted = _optimizer(u_lr=0.0, batched_step=batched)
    scheduler = _scheduler(_spec(), optimizer=uninterrupted)
    for step in range(5):
        _gradients(uninterrupted, step)
        uninterrupted.step()
        scheduler.step(8)
    checkpoint = io.BytesIO()
    torch.save(
        dict(
            params=[p.detach() for p in uninterrupted.param_groups[0]["params"]],
            optimizer=uninterrupted.state_dict(),
            scheduler=scheduler.state_dict(),
        ),
        checkpoint,
    )
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed = _optimizer([torch.nn.Parameter(p) for p in saved["params"]], batched_step=batched)
    resumed_scheduler = _scheduler(_spec(), optimizer=resumed)
    resumed.load_state_dict(saved["optimizer"])
    resumed_scheduler.load_state_dict(saved["scheduler"])
    _assert_equal(uninterrupted, resumed)
    for step in range(5, 10):
        for optimizer, current in ((uninterrupted, scheduler), (resumed, resumed_scheduler)):
            _gradients(optimizer, step)
            optimizer.step()
            current.step(8)
        _assert_equal(uninterrupted, resumed)
        assert resumed_scheduler.state_dict() == scheduler.state_dict()
        assert resumed.param_groups[0]["u_lr"] == uninterrupted.param_groups[0]["u_lr"]
    assert resumed_scheduler.num_steps == 80
    assert all(state["step"] == 10 for state in resumed.state.values())


@pytest.mark.parametrize("sample_based", [False, True])
@pytest.mark.parametrize("name", ["angular_muown", "angular_muown_simplified"])
@pytest.mark.parametrize("scale_mode", [None, "spectral", "unit_rms_norm"])
def test_cli_config_and_scheduler_wiring(monkeypatch, sample_based, name, scale_mode):
    """Check both CLI variants through the actual optimizer and scheduler factories."""
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    timing = (
        ["--train-samples", "80", "--lr-warmup-samples", "8"]
        if sample_based
        else ["--train-iters", "10", "--lr-warmup-iters", "1"]
    )
    simplified = name == "angular_muown_simplified"
    args = parser.parse_args(
        [
            "--optimizer",
            name,
            "--angular-muown-momentum",
            "0.8",
            "--global-batch-size",
            "8",
            "--micro-batch-size",
            "1",
            "--lr",
            "0.03",
            *timing,
            *(["--angular-muown-c", "2"] if simplified else []),
            *(["--angular-muown-scale-mode", scale_mode] if scale_mode else []),
        ]
    )
    args.start_weight_decay = args.end_weight_decay = args.weight_decay
    for field in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        setattr(args, field, torch.float32)
    config, _ = training.get_megatron_optimizer_config(args)
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=1, num_query_groups=1, kv_channels=4)
    )
    optimizer, init_state = _create_emerging_optimizer(
        config, [torch.nn.Parameter(torch.ones(3, 4))], args.optimizer, [model], None
    )
    assert type(optimizer) is (
        TensorParallelAngularMuownSimplified if simplified else TensorParallelAngularMuown
    )
    assert optimizer.param_groups[0]["momentum"] == 0.8
    assert optimizer.split_qkv  # The training default, for both optimizers.
    assert optimizer._scale_mode == (scale_mode or ("shape_scaling" if simplified else "spectral"))
    assert optimizer._extra_scale_factor == (1.0 if simplified else 0.2)
    if simplified:
        assert optimizer.param_groups[0]["u_lr"] == 0.0
        assert "u_lr_multiplier" not in optimizer.param_groups[0]
    else:
        assert "u_lr" not in optimizer.param_groups[0]
        assert optimizer._u_decay_schedule == "poly" and optimizer._u_decay_scale == 0.001
    init_state(optimizer)
    monkeypatch.setattr(training, "get_args", lambda: args)
    scheduler = training.get_optimizer_param_scheduler(optimizer)
    scheduler.step(16)
    if simplified:
        assert scheduler.get_angular_lr() == 1.0
        assert scheduler.angular_muown_schedule["total_samples"] == 80
        args.decrease_batch_size_if_needed = True
        with pytest.raises(ValueError):
            training.get_optimizer_param_scheduler(optimizer)
    else:
        assert scheduler.get_angular_lr() is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("u_decay_schedule", "poly"),
        ("u_decay_scale", 0.002),
        ("u_decay_p", 2.0),
        ("u_decay_warmup_steps", 1),
        ("u_decay_steps", 10),
        ("u_decay_min_multiplier", 0.1),
        ("extra_scale_factor", 1.0),
        ("extra_scale_factor", 0.2),
    ],
)
def test_simplified_rejects_unused_arguments(option, value):
    """Reject original-only options instead of silently ignoring them."""
    with pytest.raises(TypeError, match=option):
        _optimizer(**{option: value})
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    args = parser.parse_args(
        [
            "--optimizer",
            "angular_muown_simplified",
            "--angular-muown-c",
            "2",
            "--angular-muown-" + option.replace("_", "-"),
            str(value),
        ]
    )
    with pytest.raises(AssertionError, match="apply only to angular_muown"):
        validate_args(args)


@pytest.mark.parametrize(
    "flags,match",
    [
        (["--optimizer", "angular_muown_simplified"], "requires --angular-muown-c"),
        (
            ["--optimizer", "angular_muown", "--angular-muown-c", "2"],
            "requires --optimizer angular_muown_simplified",
        ),
    ],
)
def test_explicit_variant_selection(flags, match):
    """Require the simplified optimizer and its coefficient to be selected together."""
    args = add_megatron_arguments(argparse.ArgumentParser()).parse_args(flags)
    with pytest.raises(AssertionError, match=match):
        validate_args(args)
    with pytest.raises(TypeError, match="u_lr"):
        _optimizer(original=True, u_lr=0.1)
    assert "u_lr" not in _optimizer(original=True).param_groups[0]


@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf")])
def test_simplified_rejects_invalid_directional_rate(rate):
    """Reject negative and non-finite angular rates."""
    with pytest.raises(ValueError, match="u_lr"):
        _optimizer(u_lr=rate)
