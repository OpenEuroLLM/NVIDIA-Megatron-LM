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

Mixing. With a per-token ``[k, k]`` matrix ``M`` (doubly stochastic, as mHC's
``H_res`` but over the routed streams only) the write becomes
``streams[t, idx_j] <- sum_i M[t, j, i] streams[t, idx_i] + w[t, j] f[t]``: still
``2k C`` elements moved, and the backward needs the pre-write rows, which the read
already keeps (shared through the stash, not saved twice). Its backward turns the
stream gradient of the routed rows into ``M^T g`` in place.

Reversible mode. A sublayer only changes ``k`` rows by ``w f``, so its backward can
undo that write in place (``s <- s - w f``) and recover the streams exactly as its
read and router saw them. Then nothing but the change itself (``f``, ``w``, ``idx``)
is kept per sublayer: no gathered rows, no copy of the router's input -- those are
re-read from the restored state. The undo is exact up to the rounding of the
stream dtype; only the router's gradients (the read weights, the routing
projection) read the restored values, the residual path's do not. One stream state
per pipeline stage stays alive until the backward. Mixing is not invertible in
general and does not combine with it.

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


def routed_write_mix_reference(streams, f, idx, w, mix):
    """Out-of-place ``streams[t, idx_j] = sum_i mix[t, j, i] streams[t, idx_i] + w_j f``."""
    index = idx.long().unsqueeze(-1).expand(-1, -1, streams.shape[-1])
    rows = torch.gather(streams, 1, index).float()  # [T, k, C]
    # A broadcast sum, not einsum: a matmul runs in TF32 where it is enabled (NGC
    # containers default to it), which would make the reference the inexact side.
    new = (mix.float().unsqueeze(-1) * rows.unsqueeze(1)).sum(2)
    new = new + w.float().unsqueeze(-1) * f.float().unsqueeze(1)
    return streams.scatter(1, index, new.to(streams.dtype))


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


if HAVE_TRITON:

    @triton.jit
    def _write_mix_fwd(
        s_ptr, f_ptr, idx_ptr, w_ptr, m_ptr, N, C,
        K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # All k rows of this channel block are loaded before any is stored, so the
        # in-place update reads only pre-write values.
        t = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        c = cb * BLOCK + tl.arange(0, BLOCK)
        cm = c < C
        j = tl.arange(0, KP)
        jm = j < K
        idx = tl.load(idx_ptr + t * K + j, mask=jm, other=0).to(tl.int64)
        w = tl.load(w_ptr + t * K + j, mask=jm, other=0.0).to(tl.float32)
        mm = jm[:, None] & jm[None, :]
        m = tl.load(m_ptr + t * K * K + j[:, None] * K + j[None, :], mask=mm, other=0.0)
        m = m.to(tl.float32)
        offs = (t * N + idx)[:, None] * C + c[None, :]
        rm = jm[:, None] & cm[None, :]
        rows = tl.load(s_ptr + offs, mask=rm, other=0.0).to(tl.float32)  # [KP, BLOCK]
        f = tl.load(f_ptr + t * C + c, mask=cm, other=0.0).to(tl.float32)
        new = tl.sum(m[:, :, None] * rows[None, :, :], axis=1) + w[:, None] * f[None, :]
        tl.store(s_ptr + offs, new.to(s_ptr.dtype.element_ty), mask=rm)

    @triton.jit
    def _write_mix_bwd(
        g_ptr, rows_ptr, f_ptr, idx_ptr, w_ptr, m_ptr, df_ptr, dw_ptr, dm_ptr, N, C,
        K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, KP)
        jm = j < K
        idx = tl.load(idx_ptr + t * K + j, mask=jm, other=0).to(tl.int64)
        w = tl.load(w_ptr + t * K + j, mask=jm, other=0.0).to(tl.float32)
        mm = jm[:, None] & jm[None, :]
        m = tl.load(m_ptr + t * K * K + j[:, None] * K + j[None, :], mask=mm, other=0.0)
        m = m.to(tl.float32)
        dm = tl.zeros([KP, KP], dtype=tl.float32)
        dw = tl.zeros([KP], dtype=tl.float32)
        for c0 in range(0, C, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            cm = c < C
            rm = jm[:, None] & cm[None, :]
            goffs = (t * N + idx)[:, None] * C + c[None, :]
            g = tl.load(g_ptr + goffs, mask=rm, other=0.0).to(tl.float32)  # [KP(j), BLOCK]
            roffs = (t * K + j)[:, None] * C + c[None, :]
            rows = tl.load(rows_ptr + roffs, mask=rm, other=0.0).to(tl.float32)  # [KP(i), BLOCK]
            f = tl.load(f_ptr + t * C + c, mask=cm, other=0.0).to(tl.float32)
            dm += tl.sum(g[:, None, :] * rows[None, :, :], axis=2)  # dM[j, i] = <g_j, s_i>
            dw += tl.sum(g * f[None, :], axis=1)
            df = tl.sum(w[:, None] * g, axis=0)
            tl.store(df_ptr + t * C + c, df.to(df_ptr.dtype.element_ty), mask=cm)
            ds = tl.sum(m[:, :, None] * g[:, None, :], axis=0)  # (M^T g)_i
            tl.store(g_ptr + goffs, ds.to(g_ptr.dtype.element_ty), mask=rm)
        tl.store(dw_ptr + t * K + j, dw, mask=jm)
        tl.store(dm_ptr + t * K * K + j[:, None] * K + j[None, :], dm, mask=mm)


def _mix_block(c: int, kp: int) -> int:
    # [KP, KP, BLOCK] register tiles: keep them around 4k elements.
    return max(16, min(triton.next_power_of_2(c), 4096 // (kp * kp)))


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

    __slots__ = ("grad", "rows")

    def __init__(self):
        self.grad = None
        self.rows = None  # the read's gathered pre-write rows, for a mixing write


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
            ctx.stash.rows = None
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
        stash.rows = rows
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


class _RoutedWriteMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams, f, idx, w, mix, stash):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        kp = triton.next_power_of_2(k)
        block = _mix_block(c, kp)
        f = f.contiguous()
        _write_mix_fwd[(t, triton.cdiv(c, block))](
            streams, f, idx, w, mix, n, c, K=k, KP=kp, BLOCK=block
        )
        ctx.mark_dirty(streams)
        ctx.save_for_backward(f, idx, w, mix)
        ctx.n, ctx.stash = n, stash
        return streams

    @staticmethod
    def backward(ctx, g):
        f, idx, w, mix = ctx.saved_tensors
        rows = ctx.stash.rows
        if rows is None:
            raise RuntimeError("a mixing routed write needs the routed read of its sublayer")
        c = f.shape[-1]
        t, k = idx.shape
        kp = triton.next_power_of_2(k)
        g = _own(g)
        df = torch.empty_like(f)
        dw = torch.empty(t, k, dtype=torch.float32, device=f.device)
        dm = torch.empty(t, k, k, dtype=torch.float32, device=f.device)
        _write_mix_bwd[(t,)](
            g, rows, f, idx, w, mix, df, dw, dm, ctx.n, c, K=k, KP=kp, BLOCK=_mix_block(c, kp)
        )
        ctx.stash.grad = g
        return g, df, None, dw.to(w.dtype), dm.to(mix.dtype), None


class _RouterCoeffsReversible(torch.autograd.Function):
    """``coeff_fn(stream main, phi)`` without keeping the stream: the backward
    recomputes it from the restored state."""

    @staticmethod
    def forward(ctx, streams, phi, main, coeff_fn, stash):
        with torch.no_grad():
            out = coeff_fn(streams.select(-2, main), phi)
        ctx.streams, ctx.main, ctx.coeff_fn, ctx.stash = streams, main, coeff_fn, stash
        ctx.save_for_backward(phi)
        return out

    @staticmethod
    def backward(ctx, dout):
        (phi,) = ctx.saved_tensors
        streams = ctx.streams
        with torch.enable_grad():
            s0 = streams.select(-2, ctx.main).detach().clone().requires_grad_()
            p = phi.detach().requires_grad_()
            ds0, dphi = torch.autograd.grad(ctx.coeff_fn(s0, p), (s0, p), dout)
        ctx.streams = None
        stash = ctx.stash
        if stash.grad is not None:
            # This is the sublayer's last backward: release the buffer.
            grad, stash.grad = stash.grad, None
            grad.select(-2, ctx.main).add_(ds0.to(grad.dtype))
            return None, dphi, None, None, None
        ds = ds0.new_zeros(streams.shape)
        ds.select(-2, ctx.main).copy_(ds0)
        return ds, dphi, None, None, None


class _RoutedReadReversible(torch.autograd.Function):
    """The read without saved rows: the backward gathers them from the restored state."""

    @staticmethod
    def forward(ctx, streams, idx, r, stash):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        x = streams.new_empty(streams.shape[:-2] + (c,))
        rows = streams.new_empty(t, k, c)  # scratch, freed after the forward
        block = _block(c)
        _read_fwd[(t, triton.cdiv(c, block))](streams, idx, r, x, rows, n, c, K=k, BLOCK=block)
        ctx.save_for_backward(idx, r)
        ctx.streams, ctx.n, ctx.stash = streams, n, stash
        return x

    @staticmethod
    def backward(ctx, dx):
        idx, r = ctx.saved_tensors
        streams = ctx.streams
        ctx.streams = None
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        flat = streams.view(t, n, c)
        rows = torch.gather(flat, 1, idx.long().unsqueeze(-1).expand(-1, -1, c))
        own = ctx.stash.grad is None
        ds = rows.new_zeros(streams.shape) if own else ctx.stash.grad
        dr = torch.empty(t, k, dtype=torch.float32, device=rows.device)
        _read_bwd[(t,)](dx.contiguous(), rows, idx, r, ds, dr, n, c, K=k, BLOCK=_block(c))
        return (ds if own else None), None, dr.to(r.dtype), None


class _RoutedWriteReversible(torch.autograd.Function):
    """The write; its backward also undoes it in place (``s -= w f``)."""

    @staticmethod
    def forward(ctx, streams, f, idx, w, stash):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        block = _block(c)
        f = f.contiguous()
        _write_fwd[(t, triton.cdiv(c, block))](streams, f, idx, w, n, c, K=k, BLOCK=block)
        ctx.mark_dirty(streams)
        ctx.save_for_backward(f, idx, w)
        ctx.streams, ctx.n, ctx.stash = streams.detach(), n, stash
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
        # Restore the streams as this sublayer's read and router saw them.
        wb = _block(c)
        _write_fwd[(t, triton.cdiv(c, wb))](ctx.streams, f, idx, -w, ctx.n, c, K=k, BLOCK=wb)
        ctx.streams = None
        ctx.stash.grad = g
        return g, df, None, dw.to(w.dtype), None


def router_coeffs_reversible(streams, phi, coeff_fn, stash, main: int = 0):
    """``coeff_fn(streams[..., main, :], phi)``, recomputed in the backward from the
    restored streams (reversible routed residual)."""
    return _RouterCoeffsReversible.apply(streams, phi, main, coeff_fn, stash)


def routed_read_reversible(streams, idx, r, stash):
    """:func:`routed_read` that keeps no rows (reversible routed residual)."""
    return _RoutedReadReversible.apply(streams, idx.int().contiguous(), r.float().contiguous(), stash)


def routed_write_reversible(streams, f, idx, w, stash):
    """:func:`routed_write` whose backward undoes the write (reversible routed residual)."""
    return _RoutedWriteReversible.apply(streams, f, idx.int().contiguous(), w.float().contiguous(), stash)


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


def routed_write_mix(
    streams: torch.Tensor,
    f: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    mix: torch.Tensor,
    stash: GradStash,
) -> torch.Tensor:
    """``streams[t, idx_j] = sum_i mix[t, j, i] streams[t, idx_i] + w[t, j] f[t]`` -- in
    place when the kernel runs (after the sublayer's :func:`routed_read`)."""
    if _fused(streams):
        return _RoutedWriteMix.apply(
            streams, f, idx.int().contiguous(), w.float().contiguous(),
            mix.float().contiguous(), stash,
        )
    n, c = streams.shape[-2:]
    out = routed_write_mix_reference(
        streams.reshape(-1, n, c), f.reshape(-1, c), idx, w, mix
    )
    return out.view(streams.shape)


def route_streams(
    logits: torch.Tensor,
    k: int,
    main: Optional[int] = 0,
    select_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stream ``main`` plus the top-(k-1) others by ``logits`` [T, n].

    Returns ``(idx [T, k] int32, read weights [T, k])``, the read weights a softmax
    over the selected logits (a convex combination, as mHC's read; the router learns
    through them, as an MoE gate does). ``select_bias`` [n] shifts only the selection
    (the aux-loss-free balancing bias), not the read weights.
    """
    t, n = logits.shape
    if not 1 <= k <= n:
        raise ValueError(f"routed top-k must be in [1, {n}], got {k}")
    scores = logits.detach() if select_bias is None else logits.detach() + select_bias
    others = torch.cat([scores[:, :main], scores[:, main + 1 :]], dim=1)
    if k > 1:
        top = others.topk(k - 1, dim=-1).indices
        top = top + (top >= main).long()  # back to stream numbering
        idx = torch.cat([torch.full_like(top[:, :1], main), top], dim=1)
    else:
        idx = torch.full((t, 1), main, dtype=torch.long, device=logits.device)
    r = torch.softmax(torch.gather(logits.float(), 1, idx), dim=-1)
    return idx.int(), r
