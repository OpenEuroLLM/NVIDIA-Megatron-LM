# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Triton Canon kernels against their torch references (megatron/core/fusions/fused_canon.py)."""

import pytest
import torch

from megatron.core.fusions.fused_canon import (
    HAVE_TRITON,
    canon_conv,
    canon_conv_reference,
    canon_swiglu,
    canon_swiglu_reference,
)

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()), reason="needs Triton and a GPU"
)

SHAPES = [(1, 1, 8), (7, 3, 33), (64, 2, 256), (130, 1, 300)]


def _inputs(shape, k, dtype, with_prefix, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    s, b, c = shape
    x = torch.randn(shape, device="cuda", generator=gen).to(dtype).requires_grad_()
    w = (0.5 * torch.randn(c, k, device="cuda", generator=gen)).to(dtype).requires_grad_()
    p = None
    if with_prefix:
        p = torch.randn(k - 1, b, c, device="cuda", generator=gen).to(dtype).requires_grad_()
    return x, w, p


def _grads(fn, x, w, p, dy):
    for t in (x, w, p):
        if t is not None:
            t.grad = None
    y = fn(x, w, p)
    y.backward(dy)
    return y.detach(), [t.grad.clone() if t is not None else None for t in (x, w, p)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_canon_conv_matches_reference(shape, k, with_prefix, dtype):
    x, w, p = _inputs(shape, k, dtype, with_prefix)
    dy = torch.randn(shape, device="cuda").to(dtype)
    y, g = _grads(canon_conv, x, w, p, dy)
    y_ref, g_ref = _grads(canon_conv_reference, x, w, p, dy)
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(y, y_ref, **tol)
    for a, r in zip(g, g_ref):
        if r is not None:
            torch.testing.assert_close(
                a.float(),
                r.float(),
                **(
                    tol
                    if a.numel() < 4096 or dtype == torch.float32
                    else dict(rtol=5e-2, atol=2e-1)
                ),
            )


@pytest.mark.parametrize("shape", [(1, 1, 8), (9, 2, 64), (64, 2, 512), (33, 3, 300)])
@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_canon_swiglu_matches_reference(shape, with_prefix, dtype):
    k = 4
    u, w, p = _inputs(shape, k, dtype, with_prefix, seed=1)
    s, b, c = shape
    dy = torch.randn(s, b, c // 2, device="cuda").to(dtype)
    y, g = _grads(canon_swiglu, u, w, p, dy)
    y_ref, g_ref = _grads(canon_swiglu_reference, u, w, p, dy)
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=3e-2, atol=1e-1)
    torch.testing.assert_close(y, y_ref, **tol)
    for a, r in zip(g, g_ref):
        if r is not None:
            torch.testing.assert_close(
                a.float(),
                r.float(),
                **(tol if dtype == torch.float32 else dict(rtol=5e-2, atol=5e-1)),
            )


def test_zero_weight_is_identity():
    x = torch.randn(16, 2, 32, device="cuda")
    w = torch.zeros(32, 4, device="cuda")
    torch.testing.assert_close(canon_conv(x, w), x)
