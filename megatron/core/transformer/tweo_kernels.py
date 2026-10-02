"""Fused TWEO elementwise adjoint and FP64 diagnostic reductions.

No coefficient/normalizer precombination: it can flush useful B1 gradients.
CUDA contiguous FP32/BF16 only; the caller retains the reference fallback.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _adjoint(X, G, S, Y, N: tl.constexpr, TAU: tl.constexpr,
             COEFF, NORMALIZER: tl.constexpr,
             SCALE_TENSOR: tl.constexpr, SCALE_VALUE: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    g = tl.load(G + i, i < N, other=0).to(tl.float32)
    x = tl.div_rn(x, TAU)
    extra = (x * x) * x
    extra = extra * COEFF
    extra = extra * (4.0 / NORMALIZER)
    extra = tl.div_rn(extra, TAU)
    if SCALE_TENSOR:
        scale = tl.load(S).to(tl.float32)
    else:
        scale = SCALE_VALUE
    tl.store(Y + i, g + extra * scale, i < N)


@triton.jit
def _fourth_partials(X, P, N: tl.constexpr, TAU: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float64)
    x = x / tl.full((), TAU, tl.float64)
    square = x * x
    tl.store(P + tl.program_id(0), tl.sum(square * square, 0))


@triton.jit
def _record(P, R, NP: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.sum(tl.load(P + i, i < NP, other=0), 0)
    tl.store(R, tl.load(R) + value)
    tl.store(R + 1, tl.load(R + 1) + N)


def adjoint(value: torch.Tensor, gradient: torch.Tensor, scale, coeff: float,
            tau: float, normalizer: int) -> torch.Tensor:
    """Add the quartic derivative in one pass without modifying either input."""
    out = torch.empty_like(value)
    tensor_scale = isinstance(scale, torch.Tensor)
    _adjoint[(triton.cdiv(value.numel(), 1024),)](
        value, gradient, scale if tensor_scale else value, out, value.numel(),
        tau, coeff, normalizer, tensor_scale, 0.0 if tensor_scale else scale,
        1024, enable_fp_fusion=False,
    )
    return out


def accumulate_moments(value: torch.Tensor, tau: float, record: torch.Tensor) -> None:
    """Accumulate every element into the same full FP64 statistic as debug mode."""
    blocks = triton.cdiv(value.numel(), 4096)
    partials = torch.empty((blocks,), device=value.device, dtype=torch.float64)
    _fourth_partials[(blocks,)](value, partials, value.numel(), tau, 4096,
                              enable_fp_fusion=False)
    _record[(1,)](partials, record, blocks, value.numel(), triton.next_power_of_2(blocks),
                  enable_fp_fusion=False)
