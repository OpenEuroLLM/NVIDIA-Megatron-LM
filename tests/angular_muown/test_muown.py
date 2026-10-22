# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Plain Muown versus explicit weight-normalization coordinates."""

import argparse
import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    TensorParallelAngularMuown,
    TensorParallelMuownClassical,
    _create_emerging_optimizer,
    _muown_config_to_kwargs,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.arguments import add_megatron_arguments, validate_args
from megatron.training.training import get_megatron_optimizer_config


def device():
    """Use CPU for these small mathematical and Gloo checks."""
    return "cpu"


@pytest.fixture(autouse=True)
def eager_cpu():
    """Exercise update equations without compiling kernels on CPU."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch._dynamo.config.patch(disable=True):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("shape,split", [((4, 8), False), ((8, 4), False), ((12, 4), True)])
def test_explicit_weight_normalization_reference(batched, nesterov, shape, split):
    """Compare persistent-v/autograd updates with the compact row-norm state."""
    torch.manual_seed(123)
    weights = [torch.nn.Parameter(torch.randn(shape, device=device()) * 0.2) for _ in range(2)]
    vectors = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    gains = [torch.nn.Parameter(v.detach().norm(dim=1, keepdim=True)) for v in vectors]
    moments = [torch.zeros_like(v) for v in vectors]
    optimizer = TensorParallelMuownClassical(
        weights,
        lr=0.008,
        scale_mode="spectral",
        extra_scale_factor=0.2,
        nesterov=nesterov,
        batched_step=batched,
        split_qkv=split,
        is_qkv_fn=lambda p: split,
        qkv_split_shapes=(2, 2, 2),
        fp32_matmul_prec="highest",
    )
    adam = torch.optim.Adam(gains, lr=0.008, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    initial_norms = [v.detach().norm(dim=1, keepdim=True).clone() for v in vectors]

    for lr in (0.001, 0.008, 0.008, 0.004, 0.0008):
        optimizer.param_groups[0]["lr"] = adam.param_groups[0]["lr"] = lr
        for weight, vector, gain, moment in zip(weights, vectors, gains, moments, strict=True):
            grad = torch.randn_like(weight)
            weight.grad = grad
            # Autograd through explicit W(g,v), independent of the optimizer's Jacobian.
            composed = gain * vector / vector.norm(dim=1, keepdim=True)
            gain.grad, grad_v = torch.autograd.grad((composed * grad).sum(), (gain, vector))
            with torch.no_grad():
                moment.mul_(0.95).add_(grad_v)
                update = grad_v + 0.95 * moment if nesterov else moment
                blocks = update.reshape(2, 6, 4).split(2, dim=1) if split else [update]
                directions = []
                for block in blocks:
                    matrix = block.reshape(-1, shape[1])
                    direction = newton_schulz(matrix, 5, "simple")
                    directions.append(direction * (0.2 * max(matrix.shape) ** 0.5))
                if split:
                    direction = torch.cat([d.reshape(2, 2, 4) for d in directions], dim=1)
                    direction = direction.reshape(shape)
                else:
                    direction = directions[0]
                vector.add_(direction, alpha=-lr)
        adam.step()
        optimizer.step()
        for weight, vector, gain, moment in zip(weights, vectors, gains, moments, strict=True):
            norm = vector.detach().norm(dim=1, keepdim=True)
            torch.testing.assert_close(weight, gain * vector / norm, rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(optimizer.state[weight]["v_norm"], norm)
            torch.testing.assert_close(optimizer.state[weight]["m_u"], moment, rtol=2e-4, atol=2e-6)
    assert any(
        not torch.equal(optimizer.state[w]["v_norm"], n)
        for w, n in zip(weights, initial_norms, strict=True)
    )


@pytest.mark.parametrize("optimizer_cls", [TensorParallelMuownClassical, TensorParallelAngularMuown])
def test_checkpoint_resume_and_batched_equivalence(optimizer_cls):
    """Check packed row-state resume and parity between both update paths."""
    torch.manual_seed(12)
    weights = [torch.nn.Parameter(torch.randn(4, 8, device=device())) for _ in range(2)]
    optimizer = optimizer_cls(weights, lr=0.02, split_qkv=False, fp32_matmul_prec="highest")
    for w in weights:
        w.grad = torch.randn_like(w)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    if optimizer_cls is TensorParallelMuownClassical:
        assert all(s["v_norm"].shape == (4, 8) for s in saved["state"].values())
        incompatible = copy.deepcopy(saved)
        for state in incompatible["state"].values():
            del state["v_norm"]
        with pytest.raises(ValueError, match="incompatible"):
            optimizer.load_state_dict(incompatible)
    copies = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    resumed = optimizer_cls(copies, split_qkv=False, batched_step=False, fp32_matmul_prec="highest")
    resumed.load_state_dict(saved)
    for _ in range(3):
        for w, q in zip(weights, copies, strict=True):
            w.grad = torch.randn_like(w)
            q.grad = w.grad.clone()
        optimizer.step()
        resumed.step()
        for w, q in zip(weights, copies, strict=True):
            torch.testing.assert_close(w, q)
            for key in optimizer.state[w]:
                torch.testing.assert_close(optimizer.state[w][key], resumed.state[q][key])


def test_registry_and_config_routing():
    """Reuse AngularMuown settings while retaining auxiliary Adam routing."""
    config = OptimizerConfig(optimizer="muown_classical", weight_decay=0)
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=12, num_query_groups=12, kv_channels=64)
    )
    kwargs = _muown_config_to_kwargs(config, [model], None)
    assert kwargs["betas"] == (config.angular_muown_beta1, config.angular_muown_beta2)
    assert not any(key.startswith("u_") for key in kwargs)
    optimizer, init = (
        _EMERGING_OPTIMIZERS["muown_classical"].optimizer_cls,
        _EMERGING_OPTIMIZERS["muown_classical"].init_state_fn,
    )
    weight = torch.nn.Parameter(torch.ones(4, 8, device=device()))
    instance = optimizer([weight], **kwargs)
    init(instance)
    torch.testing.assert_close(instance.state[weight]["v_norm"], weight.norm(dim=1, keepdim=True))
    # As in AngularMuown, global WD may apply to auxiliary Adam parameters,
    # while direction and magnitude updates have no weight decay.
    config.weight_decay = 0.1
    assert "weight_decay" not in _muown_config_to_kwargs(config, [model], None)
    assert _EMERGING_OPTIMIZERS["muown_classical"].default_param_overrides == (
        _EMERGING_OPTIMIZERS["angular_muown"].default_param_overrides
    )
    with pytest.raises(ValueError, match="u_decay_scale"):
        optimizer([weight], u_decay_scale=0.001)


@pytest.fixture(scope="module")
def tp_group():
    """Keep one Gloo group alive for all tensor-parallel comparisons."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Run with torch.distributed.run --nproc-per-node=2")
    torch.distributed.init_process_group(backend="gloo", timeout=timedelta(seconds=20))
    try:
        yield torch.distributed.group.WORLD
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("partition_dim", [0, 1])
@pytest.mark.parametrize("tp_mode", ["duplicated", "distributed"])
@pytest.mark.parametrize("batched", [False, True])
def test_tensor_parallel_row_states(partition_dim, tp_mode, batched, tp_group):
    """Match an unsharded update with both row and column TP shards on CPU."""
    rank = torch.distributed.get_rank(tp_group)
    size = torch.distributed.get_world_size(tp_group)
    torch.manual_seed(19)
    full = torch.nn.Parameter(torch.randn(8, 8, device=device()))
    shard = torch.nn.Parameter(full.detach().chunk(size, dim=partition_dim)[rank].clone())
    shard.partition_dim = partition_dim
    options = dict(lr=0.008, split_qkv=False, fp32_matmul_prec="highest", batched_step=batched)
    reference = TensorParallelMuownClassical([full], **options)
    parallel = TensorParallelMuownClassical(
        [shard],
        tp_mode=tp_mode,
        pg_collection=SimpleNamespace(tp=tp_group, gtp_remat=None),
        **options,
    )
    for _ in range(4):
        full.grad = torch.randn_like(full)
        shard.grad = full.grad.chunk(size, dim=partition_dim)[rank].clone()
        reference.step()
        parallel.step()
        torch.testing.assert_close(shard, full.chunk(size, dim=partition_dim)[rank])
        expected = reference.state[full]["v_norm"]
        if partition_dim == 0:
            expected = expected.chunk(size, dim=0)[rank]
        torch.testing.assert_close(parallel.state[shard]["v_norm"], expected)


@pytest.mark.parametrize("batched", [False, True])
def test_ordinary_scheduler_and_exact_resume(batched):
    """Use only ordinary lr and resume its clock and optimizer state exactly."""
    torch.manual_seed(14)
    weight = torch.nn.Parameter(torch.randn(4, 8))
    options = dict(lr=0.02, split_qkv=False, batched_step=batched, fp32_matmul_prec="highest")
    optimizer = TensorParallelMuownClassical([weight], **options)

    def schedule(opt):
        return OptimizerParamScheduler(
            opt,
            init_lr=0,
            max_lr=0.02,
            min_lr=0.002,
            lr_warmup_steps=8,
            lr_decay_steps=80,
            lr_decay_style="linear",
            start_wd=0,
            end_wd=0,
            wd_incr_steps=80,
            wd_incr_style="constant",
        )

    scheduler = schedule(optimizer)
    for _ in range(3):
        weight.grad = torch.randn_like(weight)
        optimizer.step()
        scheduler.step(8)
    saved = copy.deepcopy(optimizer.state_dict())
    other = torch.nn.Parameter(weight.detach().clone())
    resumed = TensorParallelMuownClassical([other], **options)
    resumed_scheduler = schedule(resumed)
    resumed.load_state_dict(saved)
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    for _ in range(5):
        weight.grad = torch.randn_like(weight)
        other.grad = weight.grad.clone()
        for opt, sched in ((optimizer, scheduler), (resumed, resumed_scheduler)):
            assert "u_lr" not in opt.param_groups[0]
            assert opt._u_step_size(opt.param_groups[0], 10000) == opt.param_groups[0]["lr"]
            opt.step()
            sched.step(8)
        torch.testing.assert_close(weight, other, rtol=0, atol=0)
        for key in optimizer.state[weight]:
            torch.testing.assert_close(
                optimizer.state[weight][key], resumed.state[other][key], rtol=0, atol=0
            )
        assert scheduler.state_dict() == resumed_scheduler.state_dict()


def test_checkpoint_variants_cannot_be_switched():
    """Reject both directions of a Muown/AngularMuown checkpoint mismatch."""
    weight = torch.nn.Parameter(torch.ones(3, 4))
    weight.grad = torch.ones_like(weight)
    classical = TensorParallelMuownClassical([weight], split_qkv=False, fp32_matmul_prec="highest")
    angular = TensorParallelAngularMuown([weight], split_qkv=False, fp32_matmul_prec="highest")
    classical._init_group(classical.param_groups[0])
    angular._init_group(angular.param_groups[0])
    with pytest.raises(ValueError, match="incompatible"):
        classical.load_state_dict(angular.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        angular.load_state_dict(classical.state_dict())


@pytest.mark.parametrize("scale_mode", [None, "shape_scaling", "unit_rms_norm"])
def test_cli_factory_and_shared_settings(scale_mode):
    """Build Muown through the CLI with shared defaults and explicit overrides."""
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    flags = ["--optimizer", "muown_classical", "--lr", "0.008"]
    if scale_mode:
        flags += [
            "--angular-muown-scale-mode",
            scale_mode,
            "--angular-muown-extra-scale-factor",
            "0.7",
            "--angular-muown-momentum",
            "0.8",
            "--angular-muown-no-nesterov",
            "--angular-muown-no-batched-step",
            "--angular-muown-no-split-qkv",
            "--angular-muown-beta1",
            "0.7",
            "--angular-muown-beta2",
            "0.8",
            "--angular-muown-num-ns-steps",
            "4",
            "--angular-muown-coefficient-type",
            "quintic",
            "--angular-muown-fp32-matmul-prec",
            "high",
            "--angular-muown-tp-mode",
            "distributed",
            "--adam-beta2",
            "0.97",
            "--adam-eps",
            "1e-7",
        ]
    args = parser.parse_args(flags)
    for field in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        setattr(args, field, torch.float32)
    config, _ = get_megatron_optimizer_config(args)
    weight = torch.nn.Parameter(torch.ones(4, 8))
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=1, num_query_groups=1, kv_channels=4)
    )
    optimizer, init = _create_emerging_optimizer(config, [weight], "muown_classical", [model], None)
    init(optimizer)
    assert isinstance(optimizer, TensorParallelMuownClassical)
    assert "v_norm" in optimizer.state[weight]
    assert optimizer._scale_mode == (scale_mode or "spectral")
    assert optimizer._extra_scale_factor == (0.7 if scale_mode else 0.2)
    assert optimizer.param_groups[0]["lr"] == 0.008
    assert optimizer.param_groups[0]["momentum"] == (0.8 if scale_mode else 0.95)
    assert optimizer.param_groups[0]["betas"] == ((0.7, 0.8) if scale_mode else (0.9, 0.95))
    assert optimizer.param_groups[0]["adam_eps"] == (1e-7 if scale_mode else 1e-8)
    assert optimizer.nesterov == optimizer.batched_step == optimizer.split_qkv == (not scale_mode)
    assert optimizer._num_ns_steps == (4 if scale_mode else 5)
    assert optimizer._coefficient_type == ("quintic" if scale_mode else "simple")
    assert optimizer.fp32_matmul_prec == ("high" if scale_mode else "medium")
    assert optimizer.tp_mode == ("distributed" if scale_mode else "duplicated")
    assert "u_lr" not in optimizer.param_groups[0]
    # Row-magnitude betas must not overwrite the auxiliary Adam configuration.
    assert config.adam_beta2 == (0.97 if scale_mode else args.adam_beta2)


@pytest.mark.parametrize(
    "option,value",
    [
        ("c", 2),
        ("u-decay-schedule", "cosine"),
        ("u-decay-scale", 0.002),
        ("u-decay-p", 0),
        ("u-decay-warmup-steps", 5),
        ("u-decay-steps", 10),
        ("u-decay-min-multiplier", 0.1),
        ("cooldown-fraction", 0.3),
        ("cooldown-end-ratio", 0.1),
    ],
)
def test_muown_rejects_angular_schedules(option, value):
    """Reject angular-only flags set away from their defaults (launchers omit default values,
    so an explicit default cannot be told apart from an absent flag)."""
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    args = parser.parse_args(["--optimizer", "muown_classical", "--angular-muown-" + option, str(value)])
    message = "requires --optimizer" if option == "c" else "no angular decay or cooldown"
    with pytest.raises(AssertionError, match=message):
        validate_args(args)


def test_muown_requires_global_row_norms():
    """The shared TP flag and constructor both reject blockwise geometry."""
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "muown_classical", "--angular-muown-tp-mode", "blockwise"])
    with pytest.raises(ValueError, match="blockwise"):
        TensorParallelMuownClassical([torch.nn.Parameter(torch.ones(4, 8))], tp_mode="blockwise")
