# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Fused Triton kernels for Canon layers (Allen-Zhu, "Physics of Language Models 4.1").

A Canon layer is a short causal depthwise convolution with a residual,
``y_t = x_t + sum_j w_j * x_{t-j}`` (kernel ``K``, typically 4), over a sequence-first
activation ``[S, B, C]``. It is memory-bound, so the kernels fuse it with what sits
next to it:

* :func:`canon_conv` -- conv + residual in one pass (Canon A/B/C);
* :func:`canon_swiglu` -- conv + residual on the fc1 output followed by SwiGLU
  (Canon D): reads ``[S, B, 2F]`` and writes ``[S, B, F]``, exactly the traffic of the
  plain SwiGLU it replaces, so the Canon layer costs no extra memory pass.

``prefix`` (``[K-1, B, C]`` or None) holds the ``K-1`` rows before the first position
(a sequence-parallel / pipeline halo); None means zeros (the sequence start).
Backward kernels return the input, prefix and weight gradients; the weight gradient is
accumulated in fp32. :func:`canon_conv_reference` / :func:`canon_swiglu_reference` are
the plain-torch definitions the kernels are tested against (and the CPU fallback).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


# ---------------------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------------------
def _extend(x: torch.Tensor, prefix: Optional[torch.Tensor], k: int) -> torch.Tensor:
    if prefix is None:
        prefix = x.new_zeros((k - 1, *x.shape[1:]))
    return torch.cat([prefix, x], dim=0)


def canon_conv_reference(
    x: torch.Tensor, weight: torch.Tensor, prefix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``y_t = x_t + sum_j weight[:, j] * x_{t-j}`` over ``[S, B, C]`` (fp32 math)."""
    k = weight.shape[1]
    ext = _extend(x, prefix, k).float()
    s = x.shape[0]
    y = x.float().clone()
    for j in range(k):
        y = y + weight[:, j].float() * ext[k - 1 - j : k - 1 - j + s]
    return y.to(x.dtype)


def canon_swiglu_reference(
    u: torch.Tensor, weight: torch.Tensor, prefix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``SwiGLU(canon_conv(u))``: ``silu(a) * g`` with ``[a | g]`` the halves of the conv'd u."""
    v = canon_conv_reference(u, weight, prefix).float()
    a, g = v.chunk(2, dim=-1)
    return (F.silu(a) * g).to(u.dtype)


# ---------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------
if HAVE_TRITON:

    @triton.jit
    def _load_ext(x_ptr, p_ptr, t, b, c, B, C, mask, K: tl.constexpr, HAS_PREFIX: tl.constexpr):
        """x at sequence position t (may be negative: read the prefix / zeros)."""
        in_x = t >= 0
        xv = tl.load(x_ptr + (t * B + b) * C + c, mask=mask & in_x, other=0.0).to(tl.float32)
        if HAS_PREFIX:
            pt = t + (K - 1)
            pv = tl.load(
                p_ptr + (pt * B + b) * C + c, mask=mask & (t < 0) & (pt >= 0), other=0.0
            ).to(tl.float32)
            xv = tl.where(in_x, xv, pv)
        return xv

    @triton.jit
    def _canon_conv_fwd(
        x_ptr, p_ptr, w_ptr, y_ptr, S, B, C,
        K: tl.constexpr, HAS_PREFIX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)  # t * B + b
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        t = (rows // B)[:, None]
        b = (rows % B)[:, None]
        c = cols[None, :]
        mask = (rows[:, None] < S * B) & (c < C)
        acc = _load_ext(x_ptr, p_ptr, t, b, c, B, C, mask, K, HAS_PREFIX)  # residual
        for j in tl.static_range(K):
            w = tl.load(w_ptr + cols * K + j, mask=cols < C, other=0.0).to(tl.float32)
            acc += w[None, :] * _load_ext(x_ptr, p_ptr, t - j, b, c, B, C, mask, K, HAS_PREFIX)
        tl.store(y_ptr + (t * B + b) * C + c, acc.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _canon_conv_bwd(
        g_ptr, x_ptr, p_ptr, w_ptr, dx_ptr, dp_ptr, dw_ptr, S, B, C,
        K: tl.constexpr, HAS_PREFIX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        """Over the extended rows t in [-(K-1), S): dx (t >= 0), dprefix (t < 0), dw."""
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)  # extended
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        t = (rows // B)[:, None] - (K - 1)
        b = (rows % B)[:, None]
        c = cols[None, :]
        mask = (rows[:, None] < (S + K - 1) * B) & (c < C)
        g_t = tl.load(g_ptr + (t * B + b) * C + c, mask=mask & (t >= 0), other=0.0).to(
            tl.float32
        )
        acc = g_t  # the residual
        for j in tl.static_range(K):
            w = tl.load(w_ptr + cols * K + j, mask=cols < C, other=0.0).to(tl.float32)
            tj = t + j
            g = tl.load(g_ptr + (tj * B + b) * C + c, mask=mask & (tj >= 0) & (tj < S), other=0.0)
            acc += w[None, :] * g.to(tl.float32)
        tl.store(dx_ptr + (t * B + b) * C + c, acc.to(dx_ptr.dtype.element_ty), mask=mask & (t >= 0))
        if HAS_PREFIX:
            pt = t + (K - 1)
            tl.store(
                dp_ptr + (pt * B + b) * C + c, acc.to(dp_ptr.dtype.element_ty), mask=mask & (t < 0)
            )
        # dw[c, j] = sum_{t, b} g[t] * x[t - j]   (rows t >= 0 only)
        for j in tl.static_range(K):
            xv = _load_ext(x_ptr, p_ptr, t - j, b, c, B, C, mask & (t >= 0), K, HAS_PREFIX)
            part = tl.sum(g_t * xv, axis=0)
            tl.atomic_add(dw_ptr + cols * K + j, part, mask=cols < C)

    @triton.jit
    def _canon_swiglu_fwd(
        u_ptr, p_ptr, w_ptr, o_ptr, S, B, F2, F,
        K: tl.constexpr, HAS_PREFIX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        """o[..., f] = silu(a') * g' with a' = conv'd u[..., f], g' = conv'd u[..., F + f]."""
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)  # f in [0, F)
        t = (rows // B)[:, None]
        b = (rows % B)[:, None]
        fa = cols[None, :]
        fg = fa + F
        mask = (rows[:, None] < S * B) & (fa < F)
        a = _load_ext(u_ptr, p_ptr, t, b, fa, B, F2, mask, K, HAS_PREFIX)
        g = _load_ext(u_ptr, p_ptr, t, b, fg, B, F2, mask, K, HAS_PREFIX)
        for j in tl.static_range(K):
            wa = tl.load(w_ptr + cols * K + j, mask=cols < F, other=0.0).to(tl.float32)
            wg = tl.load(w_ptr + (cols + F) * K + j, mask=cols < F, other=0.0).to(tl.float32)
            a += wa[None, :] * _load_ext(u_ptr, p_ptr, t - j, b, fa, B, F2, mask, K, HAS_PREFIX)
            g += wg[None, :] * _load_ext(u_ptr, p_ptr, t - j, b, fg, B, F2, mask, K, HAS_PREFIX)
        out = a * tl.sigmoid(a) * g
        tl.store(o_ptr + (t * B + b) * F + fa, out.to(o_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _canon_swiglu_dv(
        u_ptr, p_ptr, w_ptr, do_ptr, dv_ptr, S, B, F2, F,
        K: tl.constexpr, HAS_PREFIX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        """Gradient w.r.t. the conv'd fc1 output v = [a' | g'] (recomputed from u)."""
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        t = (rows // B)[:, None]
        b = (rows % B)[:, None]
        fa = cols[None, :]
        fg = fa + F
        mask = (rows[:, None] < S * B) & (fa < F)
        a = _load_ext(u_ptr, p_ptr, t, b, fa, B, F2, mask, K, HAS_PREFIX)
        g = _load_ext(u_ptr, p_ptr, t, b, fg, B, F2, mask, K, HAS_PREFIX)
        for j in tl.static_range(K):
            wa = tl.load(w_ptr + cols * K + j, mask=cols < F, other=0.0).to(tl.float32)
            wg = tl.load(w_ptr + (cols + F) * K + j, mask=cols < F, other=0.0).to(tl.float32)
            a += wa[None, :] * _load_ext(u_ptr, p_ptr, t - j, b, fa, B, F2, mask, K, HAS_PREFIX)
            g += wg[None, :] * _load_ext(u_ptr, p_ptr, t - j, b, fg, B, F2, mask, K, HAS_PREFIX)
        do = tl.load(do_ptr + (t * B + b) * F + fa, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(a)
        da = do * g * sig * (1.0 + a * (1.0 - sig))
        dg = do * a * sig
        base = (t * B + b) * F2
        tl.store(dv_ptr + base + fa, da.to(dv_ptr.dtype.element_ty), mask=mask)
        tl.store(dv_ptr + base + fg, dg.to(dv_ptr.dtype.element_ty), mask=mask)


_BLOCK_R, _BLOCK_C = 32, 128


def _grid(rows: int, cols: int):
    return (triton.cdiv(rows, _BLOCK_R), triton.cdiv(cols, _BLOCK_C))


def _conv_bwd(grad, x, prefix, weight):
    s, b, c = x.shape
    k = weight.shape[1]
    grad = grad.contiguous()
    dx = torch.empty_like(x)
    dprefix = torch.empty_like(prefix) if prefix is not None else None
    dw = torch.zeros(weight.shape, dtype=torch.float32, device=x.device)
    _canon_conv_bwd[_grid((s + k - 1) * b, c)](
        grad, x, prefix if prefix is not None else x, weight, dx,
        dprefix if dprefix is not None else dx, dw, s, b, c,
        K=k, HAS_PREFIX=prefix is not None, BLOCK_R=_BLOCK_R, BLOCK_C=_BLOCK_C,
    )
    return dx, dprefix, dw.to(weight.dtype)


class _CanonConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, prefix):
        x = x.contiguous()
        prefix = prefix.contiguous() if prefix is not None else None
        s, b, c = x.shape
        y = torch.empty_like(x)
        _canon_conv_fwd[_grid(s * b, c)](
            x, prefix if prefix is not None else x, weight.contiguous(), y, s, b, c,
            K=weight.shape[1], HAS_PREFIX=prefix is not None, BLOCK_R=_BLOCK_R, BLOCK_C=_BLOCK_C,
        )
        ctx.save_for_backward(x, weight, prefix)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, weight, prefix = ctx.saved_tensors
        dx, dprefix, dw = _conv_bwd(grad, x, prefix, weight.contiguous())
        return dx, dw, dprefix


class _CanonSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, weight, prefix):
        u = u.contiguous()
        prefix = prefix.contiguous() if prefix is not None else None
        s, b, f2 = u.shape
        f = f2 // 2
        out = u.new_empty((s, b, f))
        _canon_swiglu_fwd[_grid(s * b, f)](
            u, prefix if prefix is not None else u, weight.contiguous(), out, s, b, f2, f,
            K=weight.shape[1], HAS_PREFIX=prefix is not None, BLOCK_R=_BLOCK_R, BLOCK_C=_BLOCK_C,
        )
        ctx.save_for_backward(u, weight, prefix)
        return out

    @staticmethod
    def backward(ctx, grad):
        u, weight, prefix = ctx.saved_tensors
        s, b, f2 = u.shape
        f = f2 // 2
        weight = weight.contiguous()
        dv = torch.empty_like(u)
        _canon_swiglu_dv[_grid(s * b, f)](
            u, prefix if prefix is not None else u, weight, grad.contiguous(), dv, s, b, f2, f,
            K=weight.shape[1], HAS_PREFIX=prefix is not None, BLOCK_R=_BLOCK_R, BLOCK_C=_BLOCK_C,
        )
        du, dprefix, dw = _conv_bwd(dv, u, prefix, weight)
        return du, dw, dprefix


def _use_triton(x: torch.Tensor) -> bool:
    return HAVE_TRITON and x.is_cuda


def canon_conv(
    x: torch.Tensor, weight: torch.Tensor, prefix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Canon conv + residual over ``[S, B, C]``; ``weight`` is ``[C, K]``."""
    if not _use_triton(x):
        return canon_conv_reference(x, weight, prefix)
    return _CanonConv.apply(x, weight, prefix)


def canon_swiglu(
    u: torch.Tensor, weight: torch.Tensor, prefix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Canon conv + residual on the fc1 output ``[S, B, 2F]`` then SwiGLU -> ``[S, B, F]``;
    ``weight`` is ``[2F, K]``."""
    if not _use_triton(u):
        return canon_swiglu_reference(u, weight, prefix)
    return _CanonSwiGLU.apply(u, weight, prefix)


__all__ = [
    "canon_conv",
    "canon_swiglu",
    "canon_conv_reference",
    "canon_swiglu_reference",
    "HAVE_TRITON",
]
