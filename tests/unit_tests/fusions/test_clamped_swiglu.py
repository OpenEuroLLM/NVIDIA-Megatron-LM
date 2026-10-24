# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""activation_func_clamp_value on the fused SwiGLU kernels matches the unfused clamped GLU."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl


def _ref(x, limit, weights=None):
    a, b = torch.chunk(x, 2, -1)
    out = F.silu(a.clamp(max=limit)) * b.clamp(min=-limit, max=limit)
    return out if weights is None else out * weights


@pytest.mark.parametrize("bias", [False, True])
def test_bias_swiglu_clamped(bias):
    torch.manual_seed(0)
    x = (torch.randn(64, 32, dtype=torch.float64) * 8).requires_grad_()
    b = (torch.randn(32, dtype=torch.float64) * 4).requires_grad_() if bias else None
    out = bias_swiglu_impl(x, b, limit=7.0)
    ref = _ref(x + b if bias else x, 7.0)
    assert (ref.detach().abs() > 0).any() and ((x.detach() > 7) | (x.detach() < -7)).any()
    g = torch.randn_like(out)
    gx, = torch.autograd.grad(out, x, g, retain_graph=True)
    rx, = torch.autograd.grad(ref, x, g, retain_graph=True)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(gx, rx)
    if bias:
        torch.testing.assert_close(torch.autograd.grad(out, b, g)[0], torch.autograd.grad(ref, b, g)[0])


def test_weighted_swiglu_clamped():
    torch.manual_seed(1)
    x = (torch.randn(64, 32, dtype=torch.float64) * 8).requires_grad_()
    w = torch.rand(64, 1, dtype=torch.float64).requires_grad_()
    out = weighted_bias_swiglu_impl(x, None, w, limit=10.0)
    ref = _ref(x, 10.0, w)
    g = torch.randn_like(out)
    torch.testing.assert_close(out, ref)
    for t in (x, w):
        torch.testing.assert_close(torch.autograd.grad(out, t, g, retain_graph=True)[0],
                                   torch.autograd.grad(ref, t, g, retain_graph=True)[0])


def test_unclamped_unchanged():
    torch.manual_seed(2)
    x = torch.randn(16, 8, dtype=torch.float64) * 8
    a, b = torch.chunk(x, 2, -1)
    torch.testing.assert_close(bias_swiglu_impl(x, None), F.silu(a) * b)
