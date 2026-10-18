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

LERP write (nGPT). On the unit sphere the write is an nGPT LERP step of each routed
stream toward the normalized sublayer output ``a``, its step the per-channel nGPT
rate ``alpha`` scaled by the row's write weight (``beta = w_j alpha``)::

    streams[t, idx_j] <- m * (s + beta * (a - s)),   s = streams[t, idx_j]

with ``m`` the renormalization of :class:`~megatron.core.transformer.ngpt.NGPTResidual`
(``approx``: from ``beta``; ``analytic``: from ``beta`` and ``cos = <s, a>``;
``direct``: ``1 / |s + beta (a - s)|``). Still ``2k C`` elements moved; ``analytic``
and ``direct`` take a second pass over the routed rows for their row reduction. Per
row the forward keeps one scalar (``cos`` or ``m``), which with ``beta`` and ``a``
makes the step invertible -- ``s = (s' / m - beta a) / (1 - beta)`` -- so the LERP
write also has a reversible form.

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


LERP_WRITE_MODES = ("approx", "analytic", "direct")


def routed_write_lerp_reference(streams, a, idx, w, alpha, mode, eps):
    """Out-of-place LERP write: ``streams[t, idx_j] = m (s + beta (a - s))``,
    ``beta = w[t, j] alpha`` (see the module docstring); ``a`` [T, C], ``alpha`` [C]."""
    index = idx.long().unsqueeze(-1).expand(-1, -1, streams.shape[-1])
    dt = torch.float64 if streams.dtype == torch.float64 else torch.float32
    rows = torch.gather(streams, 1, index).to(dt)  # [T, k, C]
    beta = w.to(dt).unsqueeze(-1) * alpha.to(dt)
    a = a.to(dt).unsqueeze(1)
    h = rows + beta * (a - rows)
    if mode == "approx":
        out = h * torch.rsqrt(torch.clamp(1.0 - 2.0 * beta * (1.0 - beta), min=eps))
    elif mode == "analytic":
        cos = (rows * a).sum(-1, keepdim=True)
        one_minus = 1.0 - beta
        denom = one_minus * one_minus + 2.0 * beta * one_minus * cos + beta * beta
        out = h * torch.rsqrt(torch.clamp(denom, min=eps))
    elif mode == "direct":
        out = h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=eps)
    else:
        raise ValueError(f"routed LERP write: mode must be one of {LERP_WRITE_MODES}, got {mode!r}")
    return streams.scatter(1, index, out.to(streams.dtype))


if HAVE_TRITON:

    @triton.jit
    def _read_fwd(
        s_ptr, idx_ptr, r_ptr, x_ptr, rows_ptr, N, C, K: tl.constexpr, BLOCK: tl.constexpr,
        STORE_ROWS: tl.constexpr = True,
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
            if STORE_ROWS:
                tl.store(rows_ptr + (t * K + j) * C + c, row, mask=m)
            acc += rj * row.to(tl.float32)
        tl.store(x_ptr + t * C + c, acc.to(x_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _read_bwd(
        dx_ptr, rows_ptr, idx_ptr, r_ptr, ds_ptr, dr_ptr, N, C,
        K: tl.constexpr, BLOCK: tl.constexpr, FROM_STREAMS: tl.constexpr = False,
    ):
        # FROM_STREAMS: rows_ptr is the (restored) [T, N, C] stream state, indexed by idx,
        # instead of the [T, K, C] rows the forward gathered
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
                if FROM_STREAMS:
                    row = tl.load(rows_ptr + (t * N + i) * C + c, mask=m, other=0.0)
                else:
                    row = tl.load(rows_ptr + (t * K + j) * C + c, mask=m, other=0.0)
                row = row.to(tl.float32)
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


if HAVE_TRITON:
    # LERP write. MODE 0 approx, 1 analytic, 2 direct (LERP_WRITE_MODES). One program
    # per token; all math in fp32. ``stat`` [T, k] is the per-row scalar the forward
    # keeps: cos = <s, a> (analytic) or m = 1 / max(|h|, eps) (direct).

    @triton.jit
    def _lerp_scale(beta, stat, eps, MODE: tl.constexpr):
        """The renormalization ``m`` [KP, BLOCK] of the step ``h = s + beta (a - s)``."""
        if MODE == 0:
            m = 1.0 / tl.sqrt(tl.maximum(1.0 - 2.0 * beta * (1.0 - beta), eps))
        elif MODE == 1:
            one_minus = 1.0 - beta
            denom = one_minus * one_minus + 2.0 * beta * one_minus * stat[:, None] + beta * beta
            m = 1.0 / tl.sqrt(tl.maximum(denom, eps))
        else:
            m = stat[:, None] + tl.zeros_like(beta)
        return m

    @triton.jit
    def _lerp_fwd(
        s_ptr, a_ptr, idx_ptr, w_ptr, al_ptr, stat_ptr, N, C, eps,
        K: tl.constexpr, KP: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, KP)
        jm = j < K
        idx = tl.load(idx_ptr + t * K + j, mask=jm, other=0).to(tl.int64)
        w = tl.load(w_ptr + t * K + j, mask=jm, other=0.0).to(tl.float32)
        row = (t * N + idx) * C
        stat = tl.zeros([KP], dtype=tl.float32)
        if MODE != 0:
            # The row reduction, over the pre-write rows (nothing is stored yet).
            acc = tl.zeros([KP], dtype=tl.float32)
            for c0 in range(0, C, BLOCK):
                c = c0 + tl.arange(0, BLOCK)
                cm = c < C
                rm = jm[:, None] & cm[None, :]
                s = tl.load(s_ptr + row[:, None] + c[None, :], mask=rm, other=0.0).to(tl.float32)
                a = tl.load(a_ptr + t * C + c, mask=cm, other=0.0).to(tl.float32)
                if MODE == 1:
                    acc += tl.sum(s * a[None, :], axis=1)
                else:
                    al = tl.load(al_ptr + c, mask=cm, other=0.0)
                    beta = w[:, None] * al[None, :]
                    h = s + beta * (a[None, :] - s)
                    acc += tl.sum(h * h, axis=1)
            if MODE == 1:
                stat = acc
            else:
                stat = 1.0 / tl.maximum(tl.sqrt(acc), eps)
            tl.store(stat_ptr + t * K + j, stat, mask=jm)
        for c0 in range(0, C, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            cm = c < C
            rm = jm[:, None] & cm[None, :]
            offs = row[:, None] + c[None, :]
            s = tl.load(s_ptr + offs, mask=rm, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + t * C + c, mask=cm, other=0.0).to(tl.float32)
            al = tl.load(al_ptr + c, mask=cm, other=0.0)
            beta = w[:, None] * al[None, :]
            m = _lerp_scale(beta, stat, eps, MODE)
            out = (s + beta * (a[None, :] - s)) * m
            tl.store(s_ptr + offs, out.to(s_ptr.dtype.element_ty), mask=rm)

    @triton.jit
    def _lerp_pre(src_ptr, srow, rrow, c, rm, a, beta, stat, eps,
                  MODE: tl.constexpr, REV: tl.constexpr):
        """The pre-write rows [KP, BLOCK]: the read's saved rows, or (REV) the inverted
        step of the post-write streams, ``s = (s' / m - beta a) / (1 - beta)``."""
        if REV:
            out = tl.load(src_ptr + srow[:, None] + c[None, :], mask=rm, other=0.0)
            m = _lerp_scale(beta, stat, eps, MODE)
            s = (out.to(tl.float32) / m - beta * a[None, :]) / (1.0 - beta)
        else:
            s = tl.load(src_ptr + rrow[:, None] + c[None, :], mask=rm, other=0.0)
            s = s.to(tl.float32)
        return s

    @triton.jit
    def _lerp_bwd(
        g_ptr, src_ptr, a_ptr, idx_ptr, w_ptr, al_ptr, stat_ptr, da_ptr, dw_ptr, dal_ptr,
        N, C, eps,
        K: tl.constexpr, KP: tl.constexpr, MODE: tl.constexpr, REV: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # g [T, N, C] is the stream gradient, turned into the pre-write rows' gradient
        # in place. src: the read's rows [T, K, C], or (REV) the streams [T, N, C],
        # restored in place.
        t = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, KP)
        jm = j < K
        idx = tl.load(idx_ptr + t * K + j, mask=jm, other=0).to(tl.int64)
        w = tl.load(w_ptr + t * K + j, mask=jm, other=0.0).to(tl.float32)
        srow = (t * N + idx) * C
        rrow = (t * K + j) * C
        stat = tl.zeros([KP], dtype=tl.float32)
        if MODE != 0:
            # 1 on the padding rows: their direct-mode m divides in the inverted step.
            stat = tl.load(stat_ptr + t * K + j, mask=jm, other=1.0)
        # Pass 1, the row reduction: analytic dL/dcos = sum -g h m^3 beta (1 - beta);
        # direct <g, out>.
        red = tl.zeros([KP], dtype=tl.float32)
        if MODE != 0:
            for c0 in range(0, C, BLOCK):
                c = c0 + tl.arange(0, BLOCK)
                cm = c < C
                rm = jm[:, None] & cm[None, :]
                g = tl.load(g_ptr + srow[:, None] + c[None, :], mask=rm, other=0.0).to(tl.float32)
                a = tl.load(a_ptr + t * C + c, mask=cm, other=0.0).to(tl.float32)
                al = tl.load(al_ptr + c, mask=cm, other=0.0)
                beta = w[:, None] * al[None, :]
                s = _lerp_pre(src_ptr, srow, rrow, c, rm, a, beta, stat, eps, MODE, REV)
                m = _lerp_scale(beta, stat, eps, MODE)
                h = s + beta * (a[None, :] - s)
                if MODE == 1:
                    red += tl.sum(-g * h * m * m * m * beta * (1.0 - beta), axis=1)
                else:
                    red += tl.sum(g * h * m, axis=1)
        dw = tl.zeros([KP], dtype=tl.float32)
        for c0 in range(0, C, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            cm = c < C
            rm = jm[:, None] & cm[None, :]
            goffs = srow[:, None] + c[None, :]
            g = tl.load(g_ptr + goffs, mask=rm, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + t * C + c, mask=cm, other=0.0).to(tl.float32)
            al = tl.load(al_ptr + c, mask=cm, other=0.0)
            beta = w[:, None] * al[None, :]
            s = _lerp_pre(src_ptr, srow, rrow, c, rm, a, beta, stat, eps, MODE, REV)
            if REV:
                tl.store(src_ptr + goffs, s.to(src_ptr.dtype.element_ty), mask=rm)
            m = _lerp_scale(beta, stat, eps, MODE)
            h = s + beta * (a[None, :] - s)
            if MODE == 0:
                dh = g * m
                dbeta = dh * (a[None, :] - s) + g * h * m * m * m * (1.0 - 2.0 * beta)
                ds = dh * (1.0 - beta)
                dar = dh * beta
            elif MODE == 1:
                dh = g * m
                u = -0.5 * g * h * m * m * m  # dL/d denom
                dbeta = dh * (a[None, :] - s) + u * 2.0 * (2.0 * beta - 1.0) * (
                    1.0 - stat[:, None]
                )
                ds = dh * (1.0 - beta) + red[:, None] * a[None, :]
                dar = dh * beta + red[:, None] * s
            else:
                dh = m * (g - h * m * red[:, None])
                dbeta = dh * (a[None, :] - s)
                ds = dh * (1.0 - beta)
                dar = dh * beta
            dw += tl.sum(dbeta * al[None, :], axis=1)
            tl.store(da_ptr + t * C + c, tl.sum(dar, axis=0), mask=cm)
            tl.store(dal_ptr + t * C + c, tl.sum(dbeta * w[:, None], axis=0), mask=cm)
            tl.store(g_ptr + goffs, ds.to(g_ptr.dtype.element_ty), mask=rm)
        tl.store(dw_ptr + t * K + j, dw, mask=jm)


def _lerp_block(c: int, kp: int) -> int:
    # A dozen [KP, BLOCK] fp32 temporaries: keep a tile around 2k elements.
    return max(16, min(triton.next_power_of_2(c), 2048 // kp))


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


_LERP_MODE_IDS = {mode: i for i, mode in enumerate(LERP_WRITE_MODES)}


class _RoutedWriteLerp(torch.autograd.Function):
    """The LERP write in place; ``reversible``: its backward also inverts it in place."""

    @staticmethod
    def forward(ctx, streams, a, idx, w, alpha, mode, eps, stash, reversible):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        kp = triton.next_power_of_2(k)
        a = a.contiguous()
        stat = torch.ones(t, k, dtype=torch.float32, device=streams.device)
        _lerp_fwd[(t,)](
            streams, a, idx, w, alpha, stat, n, c, eps,
            K=k, KP=kp, MODE=mode, BLOCK=_lerp_block(c, kp),
        )
        ctx.mark_dirty(streams)
        ctx.save_for_backward(a, idx, w, alpha, stat)
        ctx.n, ctx.mode, ctx.eps, ctx.stash, ctx.reversible = n, mode, eps, stash, reversible
        ctx.streams = streams.detach() if reversible else None
        return streams

    @staticmethod
    def backward(ctx, g):
        a, idx, w, alpha, stat = ctx.saved_tensors
        if ctx.reversible:
            src = ctx.streams
        else:
            src = ctx.stash.rows
            if src is None:
                raise RuntimeError("a LERP routed write needs the routed read of its sublayer")
        c = a.shape[-1]
        t, k = idx.shape
        kp = triton.next_power_of_2(k)
        g = _own(g)
        da = torch.empty(a.shape, dtype=torch.float32, device=a.device)
        dw = torch.empty(t, k, dtype=torch.float32, device=a.device)
        dal = torch.empty(t, c, dtype=torch.float32, device=a.device)
        _lerp_bwd[(t,)](
            g, src, a, idx, w, alpha, stat, da, dw, dal, ctx.n, c, ctx.eps,
            K=k, KP=kp, MODE=ctx.mode, REV=ctx.reversible, BLOCK=_lerp_block(c, kp),
        )
        ctx.streams = None
        ctx.stash.grad = g
        return g, da.to(a.dtype), None, dw, dal.sum(0), None, None, None, None


class _StreamTapTags(torch.autograd.Function):
    """:class:`_StreamTap` plus the first ``tag_dim`` channels of every stream, ``[..., n,
    tag_dim]`` (the content router's keys), in one node: its backward is the sublayer's
    last for both."""

    @staticmethod
    def forward(ctx, streams, main, tag_dim, stash):
        ctx.main, ctx.tag_dim, ctx.stash, ctx.shape = main, tag_dim, stash, streams.shape
        return streams.select(-2, main).clone(), streams[..., :tag_dim].clone()

    @staticmethod
    def backward(ctx, ds0, dtags):
        grad = ctx.stash.grad
        if grad is not None:
            grad, ctx.stash.grad = grad, None
            ctx.stash.rows = None
            own = False
        else:
            grad, own = ds0.new_zeros(ctx.shape), True
        grad.select(-2, ctx.main).add_(ds0)
        grad[..., : ctx.tag_dim].add_(dtags)
        return (grad if own else None), None, None, None


class _RouterCoeffsReversible(torch.autograd.Function):
    """``coeff_fn(stream main, tags, *params)`` without keeping the stream: the backward
    recomputes it from the restored state. ``tags`` are the first ``tag_dim`` channels
    of every stream (None when 0)."""

    @staticmethod
    def forward(ctx, streams, main, tag_dim, coeff_fn, stash, *params):
        with torch.no_grad():
            tags = streams[..., :tag_dim] if tag_dim else None
            out = coeff_fn(streams.select(-2, main), tags, *params)
        ctx.streams, ctx.main, ctx.tag_dim = streams, main, tag_dim
        ctx.coeff_fn, ctx.stash = coeff_fn, stash
        ctx.save_for_backward(*params)
        return out

    @staticmethod
    def backward(ctx, dout):
        params = ctx.saved_tensors
        streams, tag_dim = ctx.streams, ctx.tag_dim
        with torch.enable_grad():
            s0 = streams.select(-2, ctx.main).detach().clone().requires_grad_()
            inputs = [s0]
            tags = None
            if tag_dim:
                tags = streams[..., :tag_dim].detach().clone().requires_grad_()
                inputs.append(tags)
            ps = [p.detach().requires_grad_() for p in params]
            grads = torch.autograd.grad(
                ctx.coeff_fn(s0, tags, *ps), inputs + ps, dout, allow_unused=True
            )
        ctx.streams = None
        ds0, dtags = grads[0], (grads[1] if tag_dim else None)
        dparams = tuple(grads[len(inputs) :])
        stash = ctx.stash
        if stash.grad is not None:
            # This is the sublayer's last backward: release the buffer.
            grad, stash.grad = stash.grad, None
            own = False
        else:
            grad, own = ds0.new_zeros(streams.shape), True
        grad.select(-2, ctx.main).add_(ds0.to(grad.dtype))
        if dtags is not None:
            grad[..., :tag_dim].add_(dtags.to(grad.dtype))
        return ((grad if own else None), None, None, None, None) + dparams


class _RoutedReadReversible(torch.autograd.Function):
    """The read without saved rows: the backward gathers them from the restored state."""

    @staticmethod
    def forward(ctx, streams, idx, r, stash):
        t, n, c = _dims(streams)
        k = idx.shape[-1]
        x = streams.new_empty(streams.shape[:-2] + (c,))
        block = _block(c)
        _read_fwd[(t, triton.cdiv(c, block))](
            streams, idx, r, x, x, n, c, K=k, BLOCK=block, STORE_ROWS=False
        )
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
        own = ctx.stash.grad is None
        ds = streams.new_zeros(streams.shape) if own else ctx.stash.grad
        dr = torch.empty(t, k, dtype=torch.float32, device=streams.device)
        # the rows straight from the restored state (no gather copy)
        _read_bwd[(t,)](
            dx.contiguous(), streams, idx, r, ds, dr, n, c, K=k, BLOCK=_block(c), FROM_STREAMS=True
        )
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


def router_coeffs_reversible(streams, params, coeff_fn, stash, main: int = 0, tag_dim: int = 0):
    """``coeff_fn(streams[..., main, :], tags, *params)``, recomputed in the backward from
    the restored streams (reversible routed residual); ``tags`` is
    ``streams[..., :tag_dim]``, or None when ``tag_dim`` is 0."""
    return _RouterCoeffsReversible.apply(streams, main, tag_dim, coeff_fn, stash, *params)


def routed_read_reversible(streams, idx, r, stash):
    """:func:`routed_read` that keeps no rows (reversible routed residual)."""
    return _RoutedReadReversible.apply(streams, idx.int().contiguous(), r.float().contiguous(), stash)


def routed_write_reversible(streams, f, idx, w, stash):
    """:func:`routed_write` whose backward undoes the write (reversible routed residual)."""
    # _writable: under CUDA-graph recording a layer's input is a view of the record node's
    # output, which may not be written in place -- the write then takes a copy, and the
    # read / router (which reference the untouched input) still see the pre-write state
    return _RoutedWriteReversible.apply(
        _writable(streams), f, idx.int().contiguous(), w.float().contiguous(), stash
    )


if HAVE_TRITON:

    @triton.jit
    def _router_fwd(
        s_ptr, row_stride, off, phi_ptr, out_ptr, inv_ptr, s0_ptr, T, C, M, eps,
        BT: tl.constexpr, BC: tl.constexpr, MP: tl.constexpr, PREC: tl.constexpr,
        WRITE_S0: tl.constexpr,
    ):
        """coeffs = (s0 @ phi) / max(rms(s0), eps) and 1 / that rms, in one pass over s0
        (``WRITE_S0``: also a copy of s0, for a backward that cannot re-read it)."""
        rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
        rmask = rows < T
        cm = tl.arange(0, MP)
        mmask = cm < M
        acc = tl.zeros((BT, MP), dtype=tl.float32)
        ss = tl.zeros((BT,), dtype=tl.float32)
        for c0 in range(0, C, BC):
            cc = c0 + tl.arange(0, BC)
            cmask = cc < C
            x = tl.load(s_ptr + rows[:, None] * row_stride + off + cc[None, :],
                        mask=rmask[:, None] & cmask[None, :], other=0.0)
            w = tl.load(phi_ptr + cc[:, None] * M + cm[None, :],
                        mask=cmask[:, None] & mmask[None, :], other=0.0)
            acc += tl.dot(x, w.to(x.dtype), input_precision=PREC)
            if WRITE_S0:
                tl.store(s0_ptr + rows[:, None] * C + cc[None, :], x,
                         mask=rmask[:, None] & cmask[None, :])
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
        rms = tl.maximum(tl.sqrt(ss / C), eps)
        inv = 1.0 / rms
        tl.store(out_ptr + rows[:, None] * M + cm[None, :], acc * inv[:, None],
                 mask=rmask[:, None] & mmask[None, :])
        tl.store(inv_ptr + rows, inv, mask=rmask)

    @triton.jit
    def _router_bwd(
        s_ptr, row_stride, off, phi_ptr, g_ptr, out_ptr, inv_ptr, ds_ptr, ds_stride, ds_off,
        dphi_ptr, T, C, M, eps,
        BT: tl.constexpr, BC: tl.constexpr, MP: tl.constexpr, PREC: tl.constexpr,
    ):
        """ds0 (added in place into the stream gradient) and dphi (atomically accumulated)
        of coeffs = (s0 @ phi) * inv, inv = 1 / max(rms(s0), eps)."""
        rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
        rmask = rows < T
        cm = tl.arange(0, MP)
        mmask = cm < M
        rm2 = rmask[:, None] & mmask[None, :]
        g = tl.load(g_ptr + rows[:, None] * M + cm[None, :], mask=rm2, other=0.0)
        o = tl.load(out_ptr + rows[:, None] * M + cm[None, :], mask=rm2, other=0.0)
        inv = tl.load(inv_ptr + rows, mask=rmask, other=1.0)
        # d inv = sum_j g_j P_j with P = o / inv; inv = mean(s0^2)^-1/2 unless clamped
        dinv = tl.sum(g * o, axis=1) / inv
        clamped = inv >= (1.0 / eps) * (1.0 - 1e-6)
        scal = tl.where(clamped, 0.0, -dinv * inv * inv * inv / C)
        dp = g * inv[:, None]
        for c0 in range(0, C, BC):
            cc = c0 + tl.arange(0, BC)
            cmask = cc < C
            m2 = rmask[:, None] & cmask[None, :]
            x = tl.load(s_ptr + rows[:, None] * row_stride + off + cc[None, :], mask=m2, other=0.0)
            w = tl.load(phi_ptr + cc[:, None] * M + cm[None, :],
                        mask=cmask[:, None] & mmask[None, :], other=0.0).to(x.dtype)
            ds = tl.dot(dp.to(x.dtype), tl.trans(w), input_precision=PREC)
            ds += x.to(tl.float32) * scal[:, None]
            dst = ds_ptr + rows[:, None] * ds_stride + ds_off + cc[None, :]
            prev = tl.load(dst, mask=m2, other=0.0)
            tl.store(dst, (prev.to(tl.float32) + ds).to(prev.dtype), mask=m2)
            dw = tl.dot(tl.trans(x), dp.to(x.dtype), input_precision=PREC)
            tl.atomic_add(dphi_ptr + cc[:, None] * M + cm[None, :], dw,
                          mask=cmask[:, None] & mmask[None, :])


class _FusedRouterCoeffs(torch.autograd.Function):
    """The router projection ``(s0 @ phi) / rms(s0)`` of stream ``main``, ``[T, M]`` fp32,
    straight from the stream state. Its backward is the sublayer's last: it adds ``ds0``
    into the shared stream gradient in place. ``s0`` for the backward is re-read from the
    restored streams (reversible), else a copy the forward kernel writes in the same pass
    (the streams are written in place after the router)."""

    BT = 32

    @staticmethod
    def _tiles(mp, dtype):
        # shared memory: wide routers (mixing: 2n + n^2 columns) take narrower C tiles
        return (64 if mp <= 32 else 32), ("ieee" if dtype == torch.float32 else "tf32")

    @staticmethod
    def forward(ctx, streams, phi, eps, main, stash, reversible):
        t, n, c = _dims(streams)
        m = phi.shape[1]
        mp = max(16, triton.next_power_of_2(m))
        out = torch.empty(t, m, dtype=torch.float32, device=streams.device)
        inv = torch.empty(t, dtype=torch.float32, device=streams.device)
        s0 = None if reversible else streams.new_empty(t, c)
        bt = _FusedRouterCoeffs.BT
        bc, prec = _FusedRouterCoeffs._tiles(mp, streams.dtype)
        _router_fwd[(triton.cdiv(t, bt),)](
            streams, n * c, main * c, phi.contiguous(), out, inv, inv if s0 is None else s0,
            t, c, m, eps, BT=bt, BC=bc, MP=mp, PREC=prec, WRITE_S0=s0 is not None,
            num_stages=2,
        )
        ctx.save_for_backward(phi, out, inv, *(() if s0 is None else (s0,)))
        ctx.streams = streams.detach() if reversible else None
        ctx.eps, ctx.main, ctx.stash, ctx.shape, ctx.mp = eps, main, stash, streams.shape, mp
        return out

    @staticmethod
    def backward(ctx, g):
        phi, out, inv, *s0 = ctx.saved_tensors
        stash = ctx.stash
        t, m = out.shape
        n, c = ctx.shape[-2], ctx.shape[-1]
        if ctx.streams is not None:
            src, stride, off = ctx.streams, n * c, ctx.main * c
        else:
            src, stride, off = s0[0], c, 0
        if stash.grad is not None:
            grad, stash.grad = stash.grad, None
            own = False
        else:
            grad, own = torch.zeros(ctx.shape, dtype=src.dtype, device=src.device), True
        stash.rows = None
        ctx.streams = None
        dphi = torch.zeros(phi.shape, dtype=torch.float32, device=phi.device)
        _, prec = _FusedRouterCoeffs._tiles(ctx.mp, src.dtype)
        # larger token tiles: fewer programs, so fewer atomic adds into dphi (RTX 3090 Ti,
        # T 8192 x C 2048 x M 16: 290 -> 168 us); the wide mixing router keeps small tiles
        if ctx.mp > 32:
            bt, bc, warps = 32, 32, 4
        elif src.element_size() <= 2:
            bt, bc, warps = 128, 128, 8
        else:  # fp32 tiles: twice the shared memory
            bt, bc, warps = 64, 64, 4
        _router_bwd[(triton.cdiv(t, bt),)](
            src, stride, off, phi.contiguous(), g.contiguous().float(), out, inv,
            grad, n * c, ctx.main * c, dphi, t, c, m, ctx.eps, BT=bt, BC=bc, MP=ctx.mp,
            PREC=prec, num_stages=2, num_warps=warps,
        )
        return (grad if own else None), dphi.to(phi.dtype), None, None, None, None


def fused_router_coeffs(streams, phi, eps, stash, main: int = 0, reversible: bool = False):
    """``(streams[..., main, :] @ phi) / max(rms, eps)`` as ``[T, M]`` fp32, fused (one pass
    over stream ``main`` forward and backward)."""
    return _FusedRouterCoeffs.apply(streams, phi, eps, main, stash, reversible)


def _fused(streams):
    return HAVE_TRITON and streams.is_cuda and streams.is_contiguous()


def stream_tap(streams: torch.Tensor, main: int, stash: GradStash) -> torch.Tensor:
    """A copy of stream ``main``, ``[..., C]``."""
    if _fused(streams):
        return _StreamTap.apply(streams, main, stash)
    return streams.select(-2, main)


def stream_tap_tags(
    streams: torch.Tensor, main: int, tag_dim: int, stash: GradStash
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copies of stream ``main`` ``[..., C]`` and of the first ``tag_dim`` channels of every
    stream ``[..., n, tag_dim]``."""
    if _fused(streams):
        return _StreamTapTags.apply(streams, main, tag_dim, stash)
    return streams.select(-2, main), streams[..., :tag_dim]


def routed_read(
    streams: torch.Tensor, idx: torch.Tensor, r: torch.Tensor, stash: GradStash
) -> torch.Tensor:
    """``x`` ``[..., C]`` from ``streams`` ``[..., n, C]``; ``idx``, ``r`` are ``[T, k]``."""
    if _fused(streams):
        return _RoutedRead.apply(streams, idx.int().contiguous(), r.float().contiguous(), stash)
    n, c = streams.shape[-2:]
    x = routed_read_reference(streams.reshape(-1, n, c), idx, r)
    return x.view(streams.shape[:-2] + (c,))


def _writable(streams: torch.Tensor) -> torch.Tensor:
    """``streams`` itself, or a copy when autograd refuses an in-place write on it: a view
    made inside a custom Function or a leaf that requires grad -- both what a layer's input
    is under Megatron's CUDA graphs (_CudagraphRecordNode's output while recording, the
    static input buffer while capturing)."""
    if streams._is_view() or (streams.is_leaf and streams.requires_grad):
        return streams.clone()
    return streams


def routed_write(
    streams: torch.Tensor, f: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, stash: GradStash
) -> torch.Tensor:
    """``streams[t, idx[t, j]] += w[t, j] * f[t]`` -- in place when the kernel runs."""
    if _fused(streams):
        return _RoutedWrite.apply(
            _writable(streams), f, idx.int().contiguous(), w.float().contiguous(), stash
        )
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
            _writable(streams), f, idx.int().contiguous(), w.float().contiguous(),
            mix.float().contiguous(), stash,
        )
    n, c = streams.shape[-2:]
    out = routed_write_mix_reference(
        streams.reshape(-1, n, c), f.reshape(-1, c), idx, w, mix
    )
    return out.view(streams.shape)


def routed_write_lerp(
    streams: torch.Tensor,
    a: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    alpha: torch.Tensor,
    mode: str,
    eps: float,
    stash: GradStash,
    reversible: bool = False,
) -> torch.Tensor:
    """The LERP write ``streams[t, idx_j] = m (s + w[t, j] alpha (a[t] - s))`` (module
    docstring) -- in place when the kernel runs (after the sublayer's :func:`routed_read`,
    or its :func:`routed_read_reversible` with ``reversible``). ``a`` ``[..., C]`` is the
    (normalized) sublayer output, ``alpha`` ``[C]`` the nGPT step."""
    if mode not in LERP_WRITE_MODES:
        raise ValueError(f"routed LERP write: mode must be one of {LERP_WRITE_MODES}, got {mode!r}")
    if _fused(streams):
        return _RoutedWriteLerp.apply(
            _writable(streams), a, idx.int().contiguous(),
            w.float().contiguous(), alpha.float().contiguous(), _LERP_MODE_IDS[mode], eps,
            stash, reversible,
        )
    n, c = streams.shape[-2:]
    out = routed_write_lerp_reference(
        streams.reshape(-1, n, c), a.reshape(-1, c), idx, w, alpha, mode, eps
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
