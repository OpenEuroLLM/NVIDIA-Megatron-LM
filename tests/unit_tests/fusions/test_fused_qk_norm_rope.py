# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Fused q/k RMSNorm + RoPE from the QKV projection vs. the unfused torch semantics."""

import pytest
import torch

from megatron.core.fusions.fused_qk_norm_rope import (
    HAVE_TRITON,
    fused_qk_norm_rope,
    qk_norm_rope_reference,
)

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()), reason="needs CUDA and Triton"
)


def _inputs(s, b, ng, hq, d, rot, dtype, seed=0, gate=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    heads = hq * (2 if gate else 1) + 2
    mixed = torch.randn(s, b, ng, heads * d, device="cuda", generator=g).to(dtype)
    qw = (1 + 0.1 * torch.randn(d, device="cuda", generator=g)).to(dtype)
    kw = (1 + 0.1 * torch.randn(d, device="cuda", generator=g)).to(dtype)
    pos = torch.arange(s, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, rot, 2, device="cuda").float() / rot))
    ang = pos[:, None] * inv[None, :]
    freqs = torch.cat([ang, ang], dim=-1).view(s, 1, 1, rot)  # Megatron's layout
    return mixed, qw, kw, freqs


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "s,b,ng,hq,d,rot",
    [(37, 2, 3, 1, 64, 64), (64, 3, 2, 4, 64, 64), (50, 1, 4, 2, 128, 64), (33, 2, 1, 3, 64, 32)],
)
def test_matches_unfused_forward_and_backward(s, b, ng, hq, d, rot, dtype, gate):
    mixed, qw, kw, freqs = _inputs(s, b, ng, hq, d, rot, dtype, gate=gate)
    eps = 1e-6
    outs, grads = [], []
    for fn in (fused_qk_norm_rope, qk_norm_rope_reference):
        m = mixed.clone().requires_grad_()
        w1, w2 = qw.clone().requires_grad_(), kw.clone().requires_grad_()
        outputs = fn(m, w1, w2, freqs, hq, eps, gate=gate)
        q, k, v = outputs[:3]
        assert q.shape == (s, b, ng * hq, d) and k.shape == v.shape == (s, b, ng, d)
        if gate:
            assert outputs[3].shape == q.shape
        g = torch.Generator(device="cuda").manual_seed(1)
        loss = 0
        for t in outputs:
            loss = loss + (t.float() * torch.randn(t.shape, device="cuda", generator=g)).sum()
        loss.backward()
        outs.append(outputs)
        grads.append((m.grad, w1.grad, w2.grad))
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    for a, r in zip(outs[0], outs[1]):
        torch.testing.assert_close(a.float(), r.float(), **tol)
    for a, r in zip(grads[0], grads[1]):
        # weight grads sum over all tokens: compare in norm
        torch.testing.assert_close(a.float(), r.float(), rtol=tol["rtol"] * 5, atol=tol["atol"] * 20)


def _fwd_bwd(fn, mixed, qw, kw, freqs, hq, eps, gate, weight_map=lambda w: w):
    m = mixed.clone().requires_grad_()
    w1, w2 = qw.clone().requires_grad_(), kw.clone().requires_grad_()
    outputs = fn(m, weight_map(w1), weight_map(w2), freqs, hq, eps, gate=gate)
    g = torch.Generator(device="cuda").manual_seed(1)
    loss = 0
    for t in outputs:
        loss = loss + (t.float() * torch.randn(t.shape, device="cuda", generator=g)).sum()
    loss.backward()
    return outputs, (m.grad, w1.grad, w2.grad)


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("s,b,ng,hq,d", [(37, 2, 3, 1, 64), (50, 1, 4, 2, 128)])
def test_nope_norm_only(s, b, ng, hq, d, dtype, gate):
    """freqs=None (NoPE): q / k RMSNorm only, value / gate copied -- as the unfused path."""
    mixed, qw, kw, _ = _inputs(s, b, ng, hq, d, 64, dtype, gate=gate)
    eps = 1e-6
    fused = _fwd_bwd(fused_qk_norm_rope, mixed, qw, kw, None, hq, eps, gate)
    ref = _fwd_bwd(qk_norm_rope_reference, mixed, qw, kw, None, hq, eps, gate)
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    for a, r in zip(fused[0], ref[0]):
        torch.testing.assert_close(a.float(), r.float(), **tol)
    for a, r in zip(fused[1], ref[1]):
        torch.testing.assert_close(a.float(), r.float(), rtol=tol["rtol"] * 5, atol=tol["atol"] * 20)
    # and it is exactly a plain RMSNorm of the head (no rotation)
    heads = hq * (2 if gate else 1) + 2
    x = mixed.view(s, b, ng, heads, d).float()
    k = x[..., heads - 2, :]
    k_ref = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * kw.float()
    torch.testing.assert_close(fused[0][1].float(), k_ref, **tol)


def test_zero_centered_gamma_weights():
    """Zero-centred gamma: the attention layer passes (1 + w); gradients reach w unchanged."""
    s, b, ng, hq, d, rot = 33, 2, 2, 2, 64, 64
    mixed, qw, kw, freqs = _inputs(s, b, ng, hq, d, rot, torch.float32)
    qw, kw = qw - 1, kw - 1  # stored zero-centred
    eps = 1e-6
    fused = _fwd_bwd(fused_qk_norm_rope, mixed, qw, kw, freqs, hq, eps, False, lambda w: w + 1)
    ref = _fwd_bwd(qk_norm_rope_reference, mixed, qw, kw, freqs, hq, eps, False, lambda w: w + 1)
    for a, r in zip(fused[0], ref[0]):
        torch.testing.assert_close(a, r, rtol=1e-5, atol=1e-5)
    for a, r in zip(fused[1], ref[1]):
        torch.testing.assert_close(a, r, rtol=5e-5, atol=2e-4)
