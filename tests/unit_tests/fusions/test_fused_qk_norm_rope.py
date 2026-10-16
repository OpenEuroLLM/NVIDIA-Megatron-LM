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


def _inputs(s, b, ng, hq, d, rot, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mixed = torch.randn(s, b, ng, (hq + 2) * d, device="cuda", generator=g).to(dtype)
    qw = (1 + 0.1 * torch.randn(d, device="cuda", generator=g)).to(dtype)
    kw = (1 + 0.1 * torch.randn(d, device="cuda", generator=g)).to(dtype)
    pos = torch.arange(s, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, rot, 2, device="cuda").float() / rot))
    ang = pos[:, None] * inv[None, :]
    freqs = torch.cat([ang, ang], dim=-1).view(s, 1, 1, rot)  # Megatron's layout
    return mixed, qw, kw, freqs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "s,b,ng,hq,d,rot",
    [(37, 2, 3, 1, 64, 64), (64, 3, 2, 4, 64, 64), (50, 1, 4, 2, 128, 64), (33, 2, 1, 3, 64, 32)],
)
def test_matches_unfused_forward_and_backward(s, b, ng, hq, d, rot, dtype):
    mixed, qw, kw, freqs = _inputs(s, b, ng, hq, d, rot, dtype)
    eps = 1e-6
    outs, grads = [], []
    for fn in (fused_qk_norm_rope, qk_norm_rope_reference):
        m = mixed.clone().requires_grad_()
        w1, w2 = qw.clone().requires_grad_(), kw.clone().requires_grad_()
        q, k, v = fn(m, w1, w2, freqs, hq, eps)
        assert q.shape == (s, b, ng * hq, d) and k.shape == v.shape == (s, b, ng, d)
        g = torch.Generator(device="cuda").manual_seed(1)
        dq, dk, dv = (torch.randn(t.shape, device="cuda", generator=g).to(dtype) for t in (q, k, v))
        (q.float() * dq.float()).sum().add_((k.float() * dk.float()).sum()).add_(
            (v.float() * dv.float()).sum()
        ).backward()
        outs.append((q, k, v))
        grads.append((m.grad, w1.grad, w2.grad))
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    for a, r in zip(outs[0], outs[1]):
        torch.testing.assert_close(a.float(), r.float(), **tol)
    for a, r in zip(grads[0], grads[1]):
        # weight grads sum over all tokens: compare in norm
        torch.testing.assert_close(a.float(), r.float(), rtol=tol["rtol"] * 5, atol=tol["atol"] * 20)
