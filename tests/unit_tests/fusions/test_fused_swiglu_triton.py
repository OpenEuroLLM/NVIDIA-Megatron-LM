# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Triton SwiGLU vs. torch."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_swiglu_triton import HAVE_TRITON, swiglu_triton

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()), reason="needs CUDA and Triton"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(37, 2, 2 * 96), (128, 4096), (5, 3, 2 * 1000)])
def test_matches_torch(shape, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    y = torch.randn(shape, device="cuda", generator=g).to(dtype)
    dout = torch.randn(*shape[:-1], shape[-1] // 2, device="cuda", generator=g).to(
        dtype
    )
    results = []
    for fn in (
        swiglu_triton,
        lambda t: F.silu(t[..., : t.shape[-1] // 2].float())
        * t[..., t.shape[-1] // 2 :].float(),
    ):
        x = y.clone().requires_grad_()
        out = fn(x)
        out.float().backward(dout.float())
        results.append((out.float(), x.grad.float()))
    tol = (
        dict(rtol=1e-5, atol=1e-5)
        if dtype == torch.float32
        else dict(rtol=1e-2, atol=1e-2)
    )
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, **tol)
