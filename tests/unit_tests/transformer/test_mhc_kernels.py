# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Correctness of the hand-written mHC backward kernels.

These replace terms that autograd would otherwise derive, so a mistake in them
does not raise -- it trains a subtly different model. Every kernel is therefore
checked against the same quantity computed two independent ways: the explicit
PyTorch formulation, and autograd differentiating the forward expression.

The reference is evaluated in float64 so the comparison is about the derivation
rather than accumulation order; the kernels run in float32 and bfloat16 with
tolerances matched to those formats.
"""

import pytest
import torch

from megatron.core.transformer.mhc import _mhc_write
from megatron.core.transformer.mhc_kernels import HAVE_TRITON, grad_h_res, write_backward

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()), reason="requires Triton on a GPU"
)

# Shapes deliberately include a C that is not a multiple of the kernel's block
# width and a token count that is not a multiple of anything, so the masked tail
# of the C loop is exercised rather than assumed.
SHAPES = [(8, 2, 4, 576), (5, 3, 2, 130), (1, 1, 4, 16), (33, 1, 2, 1)]
DTYPES = [(torch.float32, 2e-5), (torch.bfloat16, 6e-2)]


def _make(shape, dtype, seed=0):
    s, b, n, c = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *sh: torch.randn(*sh, device="cuda", dtype=dtype, generator=gen)
    return {
        "g": rnd(s, b, n, c),
        "streams": rnd(s, b, n, c),
        "sublayer_out": rnd(s, b, c),
        "h_post": rnd(s, b, n),
        # h_res is doubly stochastic in the model, so keep it non-negative here.
        "h_res": torch.rand(s, b, n, n, device="cuda", dtype=dtype, generator=gen),
    }


def _autograd_reference(t):
    """The four gradients, from autograd differentiating the forward."""
    leaves = {
        k: t[k].double().detach().requires_grad_(True)
        for k in ("streams", "sublayer_out", "h_post", "h_res")
    }
    _mhc_write(**leaves).backward(t["g"].double())
    return (
        leaves["streams"].grad,
        leaves["sublayer_out"].grad,
        leaves["h_post"].grad,
        leaves["h_res"].grad,
    )


def _rel(got, ref):
    ref = ref.double()
    return ((got.double() - ref).abs().max() / ref.abs().max().clamp(min=1e-12)).item()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype,tol", DTYPES)
def test_grad_h_res_matches_autograd(shape, dtype, tol):
    t = _make(shape, dtype)
    got = grad_h_res(t["g"], t["streams"])
    assert got.shape == (*shape[:3], shape[2])
    assert got.dtype == dtype
    assert _rel(got, _autograd_reference(t)[3]) < tol


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype,tol", DTYPES)
def test_write_backward_matches_autograd(shape, dtype, tol):
    t = _make(shape, dtype)
    got = write_backward(t["g"], t["streams"], t["sublayer_out"], t["h_post"], t["h_res"])
    for name, g, r in zip(
        ("streams", "sublayer_out", "h_post", "h_res"), got, _autograd_reference(t)
    ):
        assert g.shape == r.shape, name
        assert g.dtype == dtype, name
        assert _rel(g, r) < tol, f"{name}: rel {_rel(g, r):.2e}"


@pytest.mark.parametrize("shape", SHAPES)
def test_the_two_kernels_agree_on_h_res(shape):
    """write_backward's h_res term must match the standalone kernel exactly.

    They compute the same accumulation in the same order, so any difference
    means one of the two indexes the token or stream axis differently.
    """
    t = _make(shape, torch.float32)
    standalone = grad_h_res(t["g"], t["streams"])
    fused = write_backward(t["g"], t["streams"], t["sublayer_out"], t["h_post"], t["h_res"])[3]
    torch.testing.assert_close(fused, standalone, rtol=0, atol=0)


def test_non_contiguous_inputs_are_handled():
    """The kernels index as if contiguous, so they must not silently accept views."""
    t = _make((8, 2, 4, 576), torch.float32)
    # A transposed view has the right shape but the wrong memory order.
    g = t["g"].transpose(0, 1).transpose(0, 1)
    assert grad_h_res(g, t["streams"]).shape == (8, 2, 4, 4)
    sliced = _make((8, 2, 4, 1152), torch.float32)["streams"][..., ::2]
    assert not sliced.is_contiguous()
    got = grad_h_res(t["g"], sliced)
    ref = (t["g"].unsqueeze(-2) * sliced.unsqueeze(-3)).sum(dim=-1)
    assert _rel(got, ref) < 2e-5


def test_falls_back_when_stream_count_is_not_a_power_of_two():
    """tl.arange needs a power-of-two extent; n=3 must take the PyTorch path."""
    t = _make((4, 2, 3, 64), torch.float32)
    assert grad_h_res(t["g"], t["streams"], fallback=False) is None
    assert write_backward(
        t["g"], t["streams"], t["sublayer_out"], t["h_post"], t["h_res"], fallback=False
    ) is None
    # ...and still return the right answer when the fallback is allowed.
    got = grad_h_res(t["g"], t["streams"])
    assert _rel(got, _autograd_reference(t)[3]) < 2e-5
