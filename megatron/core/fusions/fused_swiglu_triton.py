# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""SwiGLU (no bias) as two Triton kernels: ``silu(y1) * y2`` from y = [y1 | y2].

Megatron's default goes through torch.compile (jit_fuser) and, in the backward, a
concatenation of the two half-gradients; on small models both kernels run at about half
the memory bandwidth. Here the forward reads y once and writes the output, the backward
reads y and the output gradient once and writes both halves of dy in place. fp32 math,
autotuned 2-D tiles. Enabled with config.swiglu_triton (bias-free SwiGLU only).
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:
    _CONFIGS = [
        triton.Config({"BM": bm, "BF": bf}, num_warps=w)
        for bm, bf, w in [
            (16, 256, 4),
            (32, 128, 4),
            (8, 512, 4),
            (16, 512, 8),
            (32, 256, 8),
            (64, 128, 8),
            (4, 1024, 4),
        ]
    ]

    @triton.autotune(configs=_CONFIGS, key=["M", "F"])
    @triton.jit
    def _swiglu_fwd(y, out, M, F, BM: tl.constexpr, BF: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BF + tl.arange(0, BF)
        mask = (rows < M)[:, None] & (cols < F)[None, :]
        r64 = rows.to(tl.int64)[:, None]
        a = tl.load(y + r64 * (2 * F) + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        b = tl.load(y + r64 * (2 * F) + F + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        o = a * tl.sigmoid(a) * b
        tl.store(out + r64 * F + cols[None, :], o.to(out.dtype.element_ty), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["M", "F"])
    @triton.jit
    def _swiglu_bwd(g, y, dy, M, F, BM: tl.constexpr, BF: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BF + tl.arange(0, BF)
        mask = (rows < M)[:, None] & (cols < F)[None, :]
        r64 = rows.to(tl.int64)[:, None]
        a = tl.load(y + r64 * (2 * F) + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        b = tl.load(y + r64 * (2 * F) + F + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        go = tl.load(g + r64 * F + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(a)
        silu = a * sig
        da = go * b * sig * (1.0 + a * (1.0 - sig))
        db = go * silu
        ty = dy.dtype.element_ty
        tl.store(dy + r64 * (2 * F) + cols[None, :], da.to(ty), mask=mask)
        tl.store(dy + r64 * (2 * F) + F + cols[None, :], db.to(ty), mask=mask)


def _grid(M, F):
    return lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(F, meta["BF"]))


class SwiGLUTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y):
        y = y.contiguous()
        m, f2 = y.shape
        f = f2 // 2
        out = y.new_empty(m, f)
        _swiglu_fwd[_grid(m, f)](y, out, m, f)
        ctx.save_for_backward(y)
        return out

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        m, f2 = y.shape
        dy = torch.empty_like(y)
        _swiglu_bwd[_grid(m, f2 // 2)](g.contiguous(), y, dy, m, f2 // 2)
        return dy


def swiglu_triton(y: torch.Tensor) -> torch.Tensor:
    """silu(y[..., :F]) * y[..., F:] for y [..., 2F]."""
    shape = y.shape
    out = SwiGLUTriton.apply(y.reshape(-1, shape[-1]))
    return out.view(*shape[:-1], shape[-1] // 2)
