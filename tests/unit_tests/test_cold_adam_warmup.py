# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""megatron.core.optimizer.cold_adam_warmup: parameters resumed with an all-zero Adam second moment get their
updates ramped in (k / N), element-wise for partially cold shards; parameters with intact moments step exactly as
without it."""

import copy

import torch

from megatron.core.optimizer.cold_adam_warmup import cold_parameters, install_cold_adam_warmup


class _Wrapper:
    """A Megatron-style optimizer wrapper: the torch optimizer under ``.optimizer``."""

    def __init__(self, opt):
        self.optimizer = opt


def _trained(seed=0, steps=20):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(steps):
        x = torch.randn(16, 8)
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()
    return model, opt


def _grads(model, seed):
    torch.manual_seed(seed)
    model.zero_grad()
    model(torch.randn(16, 8)).pow(2).mean().backward()


def _resume_copy(model, opt):
    m2 = copy.deepcopy(model)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-2, weight_decay=0.0)
    # deep copy: load_state_dict keeps same-device tensors by reference (two copies would share moments)
    o2.load_state_dict(copy.deepcopy(opt.state_dict()))
    return m2, o2


def test_intact_moments_install_nothing_and_step_exactly():
    model, opt = _trained()
    ref, ref_opt = _resume_copy(model, opt)
    m2, o2 = _resume_copy(model, opt)
    summary = install_cold_adam_warmup(_Wrapper(o2), steps=10)
    assert summary["params"] == 0
    for seed in range(3):
        _grads(ref, 100 + seed)
        ref_opt.step()
        _grads(m2, 100 + seed)
        o2.step()
    for a, b in zip(ref.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_cold_moments_are_detected_and_ramped():
    model, opt = _trained()
    cold_model, cold_opt = _resume_copy(model, opt)
    norm = cold_model[1]  # the LayerNorm: its Adam moments "lost" as on a JUPITER checkpoint
    for p in norm.parameters():
        cold_opt.state[p]["exp_avg"].zero_()
        cold_opt.state[p]["exp_avg_sq"].zero_()
    assert [(id(p), m) for p, m in cold_parameters(cold_opt)] == [(id(p), None) for p in norm.parameters()]
    # the same cold resume without the warmup: the reference update of the cold parameters
    raw_model, raw_opt = _resume_copy(cold_model, cold_opt)
    steps = 4
    summary = install_cold_adam_warmup(_Wrapper(cold_opt), steps=steps)
    assert summary["params"] == 2 and summary["partial"] == 0 and summary["elements"] == 16
    for k in range(1, steps + 2):
        before = [p.detach().clone() for p in cold_model[1].parameters()]
        before_raw = [p.detach().clone() for p in raw_model[1].parameters()]
        warm_before = [p.detach().clone() for p in cold_model[0].parameters()]
        warm_before_raw = [p.detach().clone() for p in raw_model[0].parameters()]
        _grads(cold_model, 200 + k)
        cold_opt.step()
        _grads(raw_model, 200 + k)
        raw_opt.step()
        r = min(1.0, k / steps)
        # the warm parameters are untouched by the ramp: their update equals the plain optimizer's (first step,
        # identical states and gradients)
        if k == 1:
            for p, q, pb, qb in zip(cold_model[0].parameters(), raw_model[0].parameters(), warm_before,
                                    warm_before_raw):
                torch.testing.assert_close(p - pb, q - qb)
            # the cold parameters' applied update is r times the plain one
            for p, q, pb, qb in zip(cold_model[1].parameters(), raw_model[1].parameters(), before, before_raw):
                torch.testing.assert_close(p - pb, r * (q - qb))
                # bounded: the plain cold update is ~lr * 0.45 per element, the ramped one r times that
                assert (p - pb).abs().max() <= r * 1e-2 + 1e-7
    # past the ramp the wrapper is gone: the plain step again
    assert "_wrap_step" not in getattr(cold_opt.step, "__qualname__", "")


class _Chained:
    """ChainedOptimizer-like: ``.optimizer`` asserts when more than one optimizer is chained (Muown + Adam)."""

    def __init__(self, *opts):
        self.chained_optimizers = [_Wrapper(o) for o in opts]

    @property
    def optimizer(self):
        assert len(self.chained_optimizers) == 1, "more than one optimizer"
        return self.chained_optimizers[0].optimizer


def test_multi_optimizer_chain_is_walked():
    model, opt = _trained()
    cold_model, cold_opt = _resume_copy(model, opt)
    for p in cold_model[1].parameters():
        cold_opt.state[p]["exp_avg_sq"].zero_()
    other_model, other_opt = _trained(seed=1)
    summary = install_cold_adam_warmup(_Chained(other_opt, cold_opt), steps=4)
    assert summary["params"] == 2 and summary["elements"] == 16


def test_partially_cold_shard_is_ramped_elementwise():
    # the JUPITER checkpoints: a sharded state saved with only rank 0's slice filled (here: the first 3 of 8 rows)
    model, opt = _trained()
    cold_model, cold_opt = _resume_copy(model, opt)
    w = cold_model[0].weight
    cold_opt.state[w]["exp_avg"][3:].zero_()
    cold_opt.state[w]["exp_avg_sq"][3:].zero_()
    raw_model, raw_opt = _resume_copy(cold_model, cold_opt)
    (p, mask), = cold_parameters(cold_opt)
    assert p is w and mask is not None and int(mask.sum()) == 40
    steps = 4
    summary = install_cold_adam_warmup(_Wrapper(cold_opt), steps=steps)
    assert summary["params"] == 1 and summary["partial"] == 1 and summary["elements"] == 40
    before, before_raw = w.detach().clone(), raw_model[0].weight.detach().clone()
    _grads(cold_model, 301)
    cold_opt.step()
    _grads(raw_model, 301)
    raw_opt.step()
    d, d_raw = w - before, raw_model[0].weight - before_raw
    torch.testing.assert_close(d[:3], d_raw[:3])  # warm rows: the plain update
    torch.testing.assert_close(d[3:], d_raw[3:] / steps)  # cold rows: 1/N of it


def test_broken_layout_state_is_zeroed_and_wholly_cold():
    # TE 2.10 FusedAdam under Megatron-FSDP: plain state of the parameter's global shape for a sharded parameter
    from megatron.core.optimizer import cold_adam_warmup as caw

    class P:
        def __init__(self, local, global_shape):
            self.data = local
            self.shape = torch.Size(global_shape)

    local = torch.randn(2, 8)
    p = P(local, (8, 8))
    state = {"exp_avg": torch.randn(8, 8), "exp_avg_sq": torch.rand(8, 8) + 0.1}

    class Opt:
        param_groups = [{"params": [p]}]

    Opt.state = {p: state}
    cold = caw.cold_parameters(Opt)
    assert len(cold) == 1 and cold[0][0] is p and cold[0][1] == "reset"
    assert not state["exp_avg"].any() and not state["exp_avg_sq"].any()
    # a state shaped like the local shard is judged element-wise, and the Engram row-wise moment is skipped
    state2 = {"exp_avg_sq": torch.rand(2, 8) + 0.1}
    Opt.state = {p: state2}
    assert caw.cold_parameters(Opt) == []
    Opt.state = {p: {"exp_avg_sq": torch.zeros(8, 1)}}
    assert caw.cold_parameters(Opt) == []
