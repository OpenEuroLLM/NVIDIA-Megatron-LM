# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""The expert spectral cap measured in the forward pass (expert_forward_cap): sigma_max from a
few batched power iterations on the gathered bf16 matrices against torch's exact value, the scale
min(1, cap / sigma_max), one measurement per armed step (recompute / micro-batches / chunks call
it again), arming / disarming, warm starts, and contiguous stacks measured without a copy."""

import pytest
import torch

from megatron.core.transformer.moe import expert_forward_cap as efc


def _matrices(n, m, k, top, seed=0):
    """n bf16 [m, k] matrices with a planted top singular value top[i] (gap: the rest ~1)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i in range(n):
        a = torch.randn(m, k, generator=g) / (m**0.5 + k**0.5)
        u = torch.randn(m, 1, generator=g)
        v = torch.randn(k, 1, generator=g)
        a += top[i] * (u / u.norm()) @ (v / v.norm()).t()
        out.append(torch.nn.Parameter(a.to(torch.bfloat16).cuda()))
    return out


def _exact(p):
    return float(torch.linalg.matrix_norm(p.detach().float(), ord=2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_measure_scales_against_exact_sigma():
    efc.STATE.reset()
    ps = _matrices(6, 96, 48, top=[0.5, 1.0, 3.0, 8.0, 20.0, 40.0])
    cap = 5.0
    efc.arm(7, {(96, 48): cap})
    assert efc.armed()
    efc.measure([ps])
    scales = efc.take_scales(7)
    assert len(scales) == 6
    for p in ps:
        want = min(1.0, cap / _exact(p))
        got = float(scales[id(p)])
        assert abs(got - want) <= 0.01 * want, (got, want)
    assert efc.take_scales(8) == {}  # scales are for the armed step only


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_measured_once_per_armed_step_and_warm_started():
    efc.STATE.reset()
    ps = _matrices(4, 64, 32, top=[2.0, 4.0, 6.0, 9.0], seed=1)
    efc.arm(3, {(64, 32): 3.0})
    efc.measure([ps])
    first = {k: v.clone() for k, v in efc.take_scales(3).items()}
    with torch.no_grad():
        for p in ps:
            p.mul_(10.0)  # a later call (recompute, next micro-batch) must not re-measure
    efc.measure([ps])
    assert all(torch.equal(first[id(p)], efc.take_scales(3)[id(p)]) for p in ps)
    # the next armed step re-measures, warm-started from the stored vectors
    assert all(id(p) in efc.STATE.vectors for p in ps)
    efc.arm(4, {(64, 32): 3.0})
    efc.measure([ps])
    for p in ps:
        want = min(1.0, 3.0 / _exact(p))
        assert abs(float(efc.take_scales(4)[id(p)]) - want) <= 0.01 * want
    efc.disarm()
    assert not efc.armed()
    efc.measure([ps])  # disarmed: nothing happens
    assert efc.take_scales(4) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_other_shapes_and_contiguous_stacks():
    efc.STATE.reset()
    flat = torch.nn.Parameter(torch.zeros(3, 40, 24, dtype=torch.bfloat16, device="cuda"))
    with torch.no_grad():
        flat.copy_(torch.randn(3, 40, 24, device="cuda") * 2)
    views = [flat[i] for i in range(3)]  # back to back in one storage: a zero-copy stack
    assert efc._stacked(views) is not None
    other = _matrices(2, 16, 16, top=[1.0, 1.0], seed=2)
    efc.arm(1, {(40, 24): 0.5})  # no cap for [16, 16]: not measured
    efc.measure([views, other])
    scales = efc.take_scales(1)
    assert set(scales) == {id(v) for v in views}
    for v in views:
        assert abs(float(scales[id(v)]) - min(1.0, 0.5 / _exact(v))) <= 0.01
