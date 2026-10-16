# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Fused ShortConv gating + causal depthwise conv vs. torch."""

import pytest
import torch

from megatron.core.fusions.fused_short_conv import (
    HAVE_TRITON,
    fused_short_conv,
    short_conv_reference,
)

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()), reason="needs CUDA and Triton"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("s,b,ch,k", [(37, 2, 96, 4), (128, 3, 200, 3), (64, 1, 64, 2)])
def test_matches_torch(s, b, ch, k, bias, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    bcx = torch.randn(s, b, 3 * ch, device="cuda", generator=g).to(dtype)
    w = (0.3 * torch.randn(ch, 1, k, device="cuda", generator=g)).to(dtype)
    bb = (0.1 * torch.randn(ch, device="cuda", generator=g)).to(dtype) if bias else None
    dy = torch.randn(s, b, ch, device="cuda", generator=g).to(dtype)
    res = []
    for fn in (fused_short_conv, short_conv_reference):
        x = bcx.clone().requires_grad_()
        ww = w.clone().requires_grad_()
        b_ = bb.clone().requires_grad_() if bias else None
        y = fn(x, ww, b_)
        y.float().backward(dy.float())
        res.append(
            [y.float(), x.grad.float(), ww.grad.float()]
            + ([b_.grad.float()] if bias else [])
        )
    tol = (
        dict(rtol=1e-4, atol=1e-4)
        if dtype == torch.float32
        else dict(rtol=2e-2, atol=5e-2)
    )
    for a, r in zip(*res):
        torch.testing.assert_close(a, r, **tol)
