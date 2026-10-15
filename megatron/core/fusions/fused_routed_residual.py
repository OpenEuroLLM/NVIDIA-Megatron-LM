# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Sparse read / in-place write over ``n`` residual streams (routed residual).

The residual state is ``[..., n, C]`` (``n`` streams per token, ``T`` tokens). A
sublayer touches only ``k`` of them per token, given by ``idx`` ``[T, k]``
(distinct per token):

* tap: ``s0[t] = streams[t, main]`` (the router's input)
* read: ``x[t] = sum_j r[t, j] * streams[t, idx[t, j]]``
* write: ``streams[t, idx[t, j]] += w[t, j] * f[t]``, **in place**

so a sublayer moves ``(2k + 1) C`` elements of the stream state instead of the
``>= 2n C`` of a dense mix (mHC). Untouched streams cost nothing.

Autograd. The write modifies its input in place, so neither the tap nor the read
keeps a reference to ``streams`` (the read saves its ``k`` gathered rows). The
three share a :class:`GradStash`: the write's backward -- which runs first, as the
write consumes what the tap and read produce -- passes the stream gradient through
and leaves it in the stash, and the read's and tap's backwards add their rows into
that buffer **in place** (the rows are distinct: no races) and return no gradient
of their own. One ``[T, n, C]`` gradient buffer thus flows down the whole stack
and each sublayer touches ``(2k + 1) C`` of it -- no full-size zero fill and no
full-size gradient sum. All tensors are taken as they are (no reshape views:
an in-place write through a view would make autograd copy the whole gradient).

``*_reference`` are the out-of-place torch definitions the kernels are tested
against.
"""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def routed_read_reference(streams, idx, r):
    """``streams`` [T, n, C], ``idx`` [T, k], ``r`` [T, k] -> ``x`` [T, C]."""
    rows = torch.gather(streams, 1, idx.long().unsqueeze(-1).expand(-1, -1, streams.shape[-1]))
    return (rows.float() * r.float().unsqueeze(-1)).sum(1).to(streams.dtype)


def routed_write_reference(streams, f, idx, w):
    """Out-of-place ``streams[t, idx[t, j]] += w[t, j] * f[t]``."""
    upd = w.float().unsqueeze(-1) * f.float().unsqueeze(1)  # [T, k, C]
    full = torch.zeros(streams.shape, dtype=torch.float32, device=streams.device)
    full = full.scatter_add(1, idx.long().unsqueeze(-1).expand_as(upd), upd)
    return (streams.float() + full).to(streams.dtype)


if HAVE_TRITON:

    @triton.jit
    def _read_fwd(
        s_ptr, idx_ptr, r_ptr, x_ptr, rows_ptr, N, C, K: tl.constexpr, BLOCK: tl.constexpr
    ):
        t = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        c = cb * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(K):
            i = tl.load(idx_ptr + t * K + j).to(tl.int64)
            rj = tl.load(r_ptr + t * K + j).to(tl.float32)
            row = tl.load(s_ptr + (t * N + i) * C + c, mask=m, other=0.0)
            tl.store(rows_ptr + (t * K + j) * C + c, row, mask=m)
            acc += rj * row.to(tl.float32)
        tl.store(x_ptr + t * C + c, acc.to(x_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _read_bwd(
        dx_ptr, rows_ptr, idx_ptr, r_ptr, ds_ptr, dr_ptr, N, C,
        K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # One program per token: dr needs a reduction over all of C, kept in registers.
        t = tl.program_id(0).to(tl.int64)
        for j in tl.static_range(K):
            i = tl.load(idx_ptr + t * K + j).to(tl.int64)
            rj = tl.load(r_ptr + t * K + j).to(tl.float32)
            dot = tl.zeros([BLOCK], dtype=tl.float32)
            for c0 in range(0, C, BLOCK):
                c = c0 + tl.arange(0, BLOCK)
                m = c < C
                dx = tl.load(dx_ptr + t * C + c, mask=m, other=0.0).to(tl.float32)
                row = tl.load(rows_ptr + (t * K + j) * C + c, mask=m, other=0.0).to(tl.float32)
                dot += dx * row
                ds_off = ds_ptr + (t * N + i) * C + c
                ds = tl.load(ds_off, mask=m, other=0.0).to(tl.float32)
                tl.store(ds_off, (ds + rj * dx).to(ds_ptr.dtype.element_ty), mask=m)
            tl.store(dr_ptr + t * K + j, tl.sum(dot, axis=0))

    @triton.jit
    def _write_fwd(s_ptr, f_ptr, idx_ptr, w_ptr, N, C, K: tl.constexpr, BLOCK: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        c = cb * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        f = tl.load(f_ptr + t * C + c, mask=m, other=0.0).to(tl.float32)
        for j in tl.static_range(K):
            i = tl.load(idx_ptr + t * K + j).to(tl.int64)
            wj = tl.load(w_ptr + t * K + j).to(tl.float32)
            off = s_ptr + (t * N + i) * C + c
            s = tl.load(off, mask=m, other=0.0).to(tl.float32)
            tl.store(off, (s + wj * f).to(s_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _write_bwd(
        g_ptr, f_ptr, idx_ptr, w_ptr, df_ptr, dw_ptr, N, C, K: tl.constexpr, BLOCK: tl.constexpr
    ):
        t = tl.program_id(0).to(tl.int64)
        dots = tl.zeros([BLOCK], dtype=tl.float32)
        for c0 in range(0, C, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            m = c < C
            f = tl.load(f_ptr + t * C + c, mask=m, other=0.0).to(tl.float32)
            df = tl.zeros([BLOCK], dtype=tl.float32)
            for j in tl.static_range(K):
                i = tl.load(idx_ptr + t * K + j).to(tl.int64)
                wj = tl.load(w_ptr + t * K + j).to(tl.float32)
                g = tl.load(g_ptr + (t * N + i) * C + c, mask=m, other=0.0).to(tl.float32)
                df += wj * g
                # dw_j partials live in lane j of a register vector (K <= BLOCK).
                part = tl.sum(g * f, axis=0)
                dots += tl.where(tl.arange(0, BLOCK) == j, part, 0.0)
            tl.store(df_ptr + t * C + c, df.to(df_ptr.dtype.element_ty), mask=m)
        lanes = tl.arange(0, BLOCK)
        tl.store(dw_ptr + t * K + lanes, dots, mask=lanes < K)


def _block(c: int) -> int:
    return min(1024, triton.next_power_of_2(c))


def _own(grad: torch.Tensor) -> torch.Tensor:
    """A contiguous gradient buffer the backwards may update in place.

    Gradients arriving from a broadcast (e.g. the mean that collapses the streams)
    are expanded views; those, and anything non-contiguous, are copied once.
    """
    if grad.is_contiguous() and 0 not in grad.stride():
        return grad
    return grad.contiguous().clone()


class GradStash:
    """The stream gradient of one sublayer, shared by its tap, read and write."""

    __slots__ = ("grad",)

    def __init__(self):
        self.grad = None


def _dims(streams):
    n, c = streams.shape[-2], streams.shape[-1]
    return streams.numel() // (n * c), n, c


class _StreamTap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams, main, stash):
        ctx.main, ctx.stash, ctx.shape = main, stash, streams.shape
        return streams.select(-2, main).clone()

    @staticmethod
    def backward(ctx, ds0):
        if ctx.stash.grad is not None:
            # The router's input is computed from the tap, so this backward is the
            # sublayer's last: release the buffer.
            grad, ctx.stash.grad = ctx.stash.grad, None
            grad.select(-2, ctx.main).add_(ds0)
            return None, None, None
        ds = ds0.new_zeros(ctx.shape)
        ds.select(-2, ctx.main).copy_(ds0)
        return ds, None, None


class _RoutedRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams, idx, r, stash):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        x = streams.new_empty(streams.shape[:-2] + (c,))
        rows = streams.new_empty(t, k, c)
        block = _block(c)
        _read_fwd[(t, triton.cdiv(c, block))](streams, idx, r, x, rows, n, c, K=k, BLOCK=block)
        ctx.save_for_backward(rows, idx, r)
        ctx.n, ctx.stash, ctx.shape = n, stash, streams.shape
        return x

    @staticmethod
    def backward(ctx, dx):
        rows, idx, r = ctx.saved_tensors
        t, k, c = rows.shape
        own = ctx.stash.grad is None
        ds = rows.new_zeros(ctx.shape) if own else ctx.stash.grad
        dr = torch.empty(t, k, dtype=torch.float32, device=rows.device)
        _read_bwd[(t,)](dx.contiguous(), rows, idx, r, ds, dr, ctx.n, c, K=k, BLOCK=_block(c))
        return (ds if own else None), None, dr.to(r.dtype), None


class _RoutedWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams, f, idx, w, stash):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        block = _block(c)
        f = f.contiguous()
        _write_fwd[(t, triton.cdiv(c, block))](streams, f, idx, w, n, c, K=k, BLOCK=block)
        ctx.mark_dirty(streams)
        ctx.save_for_backward(f, idx, w)
        ctx.n, ctx.stash = n, stash
        return streams

    @staticmethod
    def backward(ctx, g):
        f, idx, w = ctx.saved_tensors
        c = f.shape[-1]
        t, k = idx.shape
        g = _own(g)
        df = torch.empty_like(f)
        dw = torch.empty(t, k, dtype=torch.float32, device=f.device)
        block = max(_block(c), triton.next_power_of_2(k))
        _write_bwd[(t,)](g, f, idx, w, df, dw, ctx.n, c, K=k, BLOCK=block)
        ctx.stash.grad = g
        return g, df, None, dw.to(w.dtype), None


def _fused(streams):
    return HAVE_TRITON and streams.is_cuda and streams.is_contiguous()


def stream_tap(streams: torch.Tensor, main: int, stash: GradStash) -> torch.Tensor:
    """A copy of stream ``main``, ``[..., C]``."""
    if _fused(streams):
        return _StreamTap.apply(streams, main, stash)
    return streams.select(-2, main)


def routed_read(
    streams: torch.Tensor, idx: torch.Tensor, r: torch.Tensor, stash: GradStash
) -> torch.Tensor:
    """``x`` ``[..., C]`` from ``streams`` ``[..., n, C]``; ``idx``, ``r`` are ``[T, k]``."""
    if _fused(streams):
        return _RoutedRead.apply(streams, idx.int().contiguous(), r.float().contiguous(), stash)
    n, c = streams.shape[-2:]
    x = routed_read_reference(streams.reshape(-1, n, c), idx, r)
    return x.view(streams.shape[:-2] + (c,))


def routed_write(
    streams: torch.Tensor, f: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, stash: GradStash
) -> torch.Tensor:
    """``streams[t, idx[t, j]] += w[t, j] * f[t]`` -- in place when the kernel runs."""
    if _fused(streams):
        return _RoutedWrite.apply(streams, f, idx.int().contiguous(), w.float().contiguous(), stash)
    n, c = streams.shape[-2:]
    out = routed_write_reference(streams.reshape(-1, n, c), f.reshape(-1, c), idx, w)
    return out.view(streams.shape)


def route_streams(
    logits: torch.Tensor, k: int, main: Optional[int] = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stream ``main`` plus the top-(k-1) others by ``logits`` [T, n].

    Returns ``(idx [T, k] int32, read weights [T, k])``, the read weights a softmax
    over the selected logits (a convex combination, as mHC's read; the router learns
    through them, as an MoE gate does).
    """
    t, n = logits.shape
    if not 1 <= k <= n:
        raise ValueError(f"routed top-k must be in [1, {n}], got {k}")
    others = torch.cat([logits[:, :main], logits[:, main + 1 :]], dim=1)
    if k > 1:
        top = others.topk(k - 1, dim=-1).indices
        top = top + (top >= main).long()  # back to stream numbering
        idx = torch.cat([torch.full_like(top[:, :1], main), top], dim=1)
    else:
        idx = torch.full((t, 1), main, dtype=torch.long, device=logits.device)
    r = torch.softmax(torch.gather(logits.float(), 1, idx), dim=-1)
    return idx.int(), r
