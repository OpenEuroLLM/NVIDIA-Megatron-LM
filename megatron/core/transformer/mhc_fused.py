# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

r"""mHC coefficients + read as one autograd Function with Triton kernels (MHC_FUSED_READ=1).

At the production shape (n = 4 streams x C = 2048, 8k tokens, H100 64 GB) the compiled
coefficients + read ran at 28 % of the HBM bound forward and 17 % backward -- 1.6 of the 2.6 ms an
mHC sublayer costs (``scripts/korbi/hybrid_speed/bench_mhc_fuse.py``). Per token the work is
(``flat`` = the ``[n C]`` stream state, ``phi`` = ``[n C, n (n + 2)]``):

.. code-block:: text

    proj = flat @ phi,  sumsq = |flat|^2              -> h_pre, h_post, h_res (tiny, per token)
    x    = sum_i h_pre_i s_i                          the sublayer input

Forward: :func:`_proj_kernel` (tensor-core projection + sum of squares, one read of the streams),
:func:`_coeffs_fwd_kernel` (the coefficients from ``(proj, sumsq)`` per token in registers -- the
math of :func:`~megatron.core.transformer.mhc._mhc_coeffs`, log-space Sinkhorn included; compiled,
that was ~20 small kernels), :func:`_read_kernel` (second read).

Backward, given ``d_x`` and the coefficient grads from the write:

.. code-block:: text

    d_h_pre_i = sum_c d_x s_i                                         _dhpre_kernel
    (d_proj, d_sumsq, d_params) <- _coeffs_bwd_kernel                 per token, partial sums
    d_s_i     = d_s_write_i + h_pre_i d_x + phi_i d_proj + 2 d_sumsq s_i   _dstreams_kernel
    d_phi     = flat^T d_proj                    (same kernel, per-split partials, summed)

``d_s_write`` is the write's gradient for the same stream state: the write hands it over through
the sublayer's :class:`ReadWriteLink` instead of returning it, so autograd's accumulation of the
two -- one more full pass over the widest tensor -- disappears into the kernel.

The previous sublayer's write produced this stream state; its backward needs exactly ``d_s``.
With a :class:`WriteRecord` on the stream state (MHC_FUSED_WRITE_BWD=1, the default), the read's
backward runs that write's backward too (:func:`_fused_bwd_kernel`): ``d_s`` never goes to
memory, and the stream state is recomputed from the write's inputs instead of read -- about 30 C
bytes per token instead of 54 C for the pair. Where the record is missing (the first sublayer of a
recompute chunk or pipeline stage: a detached input) the two run separately.
"""

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the runtime
    HAVE_TRITON = False

ENABLED = os.environ.get("MHC_FUSED_READ", "0") == "1"
_M = 32  # n (n + 2) = 24 coefficients, padded to a tensor-core tile


def can_use(streams, n, phi):
    return (
        ENABLED
        and HAVE_TRITON
        and streams.is_cuda
        and streams.dtype == torch.bfloat16
        and n * (n + 2) <= _M
        and n == 4
        and streams.shape[-1] % 64 == 0
    )


class ReadWriteLink:
    """Carries the write's stream gradient to the read of the same sublayer (see module doc)."""

    __slots__ = ("grad",)

    def __init__(self):
        self.grad = None


class WriteRecord:
    """A write's inputs, attached to its output stream state (``_mhc_write_record``), so that the
    fused read of the next sublayer can run the write's backward inside its own backward kernel
    (:func:`_fused_bwd_kernel`) and leave the results here for the write's backward to return.
    Only set under grad mode; a recompute chunk's (or pipeline stage's) input is a detached copy
    without it, so the fusion falls back at those boundaries by itself."""

    __slots__ = ("inputs", "grads")

    def __init__(self):
        self.inputs = None  # (streams, sublayer_out, h_post, h_res) as the write saved them
        self.grads = None  # (d_streams, d_sublayer_out, d_h_post, d_h_res) from the fused read


FUSE_WRITE_BWD = os.environ.get("MHC_FUSED_WRITE_BWD", "1") == "1"
_DUMMY = {}


def dummy_grad(like):
    """The gradient the fused read returns for the stream state: zeros without memory, which the
    write's backward recognises (``is_dummy``) -- its real gradient was used already."""
    key = (like.device, like.dtype)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros((), dtype=like.dtype, device=like.device)
    return _DUMMY[key].expand(like.shape)


def is_dummy(g):
    base = _DUMMY.get((g.device, g.dtype))
    return base is not None and g.data_ptr() == base.data_ptr() and all(st == 0 for st in g.stride())


if HAVE_TRITON:

    @triton.jit
    def _proj_kernel(s_ptr, phi_ptr, proj_ptr, sq_ptr, T, K: tl.constexpr, M: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr):
        """proj[t] = flat[t] @ phi (bf16 tensor cores, fp32 accumulate), sq[t] = |flat[t]|^2."""
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        mt = t < T
        m = tl.arange(0, M)
        acc = tl.zeros((BT, M), dtype=tl.float32)
        sq = tl.zeros((BT,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(s_ptr + t[:, None].to(tl.int64) * K + k[None, :], mask=mt[:, None], other=0.0)
            w = tl.load(phi_ptr + k[:, None] * M + m[None, :])
            acc = tl.dot(x, w, acc)
            xf = x.to(tl.float32)
            sq += tl.sum(xf * xf, axis=1)
        tl.store(proj_ptr + t[:, None] * M + m[None, :], acc, mask=mt[:, None])
        tl.store(sq_ptr + t, sq, mask=mt)

    @triton.jit
    def _read_kernel(s_ptr, h_ptr, x_ptr, T, C: tl.constexpr, N: tl.constexpr, BT: tl.constexpr,
                     BC: tl.constexpr):
        """x[t, c] = sum_i h_pre[t, i] s[t, i, c] (h_pre rounded to bf16 as the eager read)."""
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        mt = t < T
        base = t[:, None].to(tl.int64) * (N * C) + c[None, :]
        acc = tl.zeros((BT, BC), dtype=tl.float32)
        for i in tl.static_range(N):
            h = tl.load(h_ptr + t * N + i, mask=mt, other=0.0).to(tl.bfloat16).to(tl.float32)
            s = tl.load(s_ptr + base + i * C, mask=mt[:, None], other=0.0).to(tl.float32)
            acc += h[:, None] * s
        tl.store(x_ptr + t[:, None].to(tl.int64) * C + c[None, :], acc.to(tl.bfloat16),
                 mask=mt[:, None])

    @triton.jit
    def _dhpre_kernel(s_ptr, dx_ptr, out_ptr, T, C: tl.constexpr, N: tl.constexpr,
                      BT: tl.constexpr, BC: tl.constexpr):
        """d_h_pre[t, i] = sum_c d_x[t, c] s[t, i, c]."""
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        mt = t < T
        i = tl.arange(0, N)
        acc = tl.zeros((BT, N), dtype=tl.float32)
        for c0 in range(0, C, BC):
            c = c0 + tl.arange(0, BC)
            dx = tl.load(dx_ptr + t[:, None].to(tl.int64) * C + c[None, :], mask=mt[:, None],
                         other=0.0).to(tl.float32)
            s = tl.load(s_ptr + t[:, None, None].to(tl.int64) * (N * C) + i[None, :, None] * C
                        + c[None, None, :], mask=mt[:, None, None], other=0.0).to(tl.float32)
            acc += tl.sum(s * dx[:, None, :], axis=2)
        tl.store(out_ptr + t[:, None] * N + i[None, :], acc, mask=mt[:, None])

    @triton.jit
    def _lse(x, axis: tl.constexpr):
        mx = tl.max(x, axis=axis)
        return mx + tl.log(tl.sum(tl.exp(x - tl.expand_dims(mx, axis)), axis=axis))

    @triton.jit
    def _coeffs_load(proj_ptr, sq_ptr, prm_ptr, t, mt, K, eps, M: tl.constexpr):
        """Per-token inputs of the coefficients: the raw coefficients over r, r itself, and the
        packed parameters [b_pre 4 | b_post 4 | b_res 16 | a_pre | a_post | a_res]."""
        i4 = tl.arange(0, 4)
        sq = tl.load(sq_ptr + t, mask=mt, other=1.0)
        rr = tl.sqrt(tl.maximum(sq, 0.0) / K)
        r = tl.maximum(rr, eps)
        pre = tl.load(proj_ptr + t[:, None] * M + i4[None, :], mask=mt[:, None], other=0.0) / r[:, None]
        post = tl.load(proj_ptr + t[:, None] * M + 4 + i4[None, :], mask=mt[:, None],
                       other=0.0) / r[:, None]
        res = tl.load(proj_ptr + t[:, None, None] * M + 8 + i4[None, :, None] * 4 + i4[None, None, :],
                      mask=mt[:, None, None], other=0.0) / r[:, None, None]
        b_pre = tl.load(prm_ptr + i4)
        b_post = tl.load(prm_ptr + 4 + i4)
        b_res = tl.load(prm_ptr + 8 + i4[:, None] * 4 + i4[None, :])
        a_pre = tl.load(prm_ptr + 24)
        a_post = tl.load(prm_ptr + 25)
        a_res = tl.load(prm_ptr + 26)
        return sq, rr, r, pre, post, res, b_pre, b_post, b_res, a_pre, a_post, a_res

    @triton.jit
    def _coeffs_fwd_kernel(proj_ptr, sq_ptr, prm_ptr, hpre_ptr, hpost_ptr, hres_ptr, T, K, eps,
                           ITERS: tl.constexpr, M: tl.constexpr, BT: tl.constexpr):
        """h_pre, h_post, h_res (log-space Sinkhorn) for BT tokens, all in registers."""
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        mt = t < T
        i4 = tl.arange(0, 4)
        (_, _, _, pre, post, res, b_pre, b_post, b_res, a_pre, a_post,
         a_res) = _coeffs_load(proj_ptr, sq_ptr, prm_ptr, t, mt, K, eps, M)
        h_pre = tl.sigmoid(a_pre * pre + b_pre[None, :])
        h_post = 2.0 * tl.sigmoid(a_post * post + b_post[None, :])
        x = a_res * res + b_res[None, :, :]
        for _ in range(ITERS):
            x = x - tl.expand_dims(_lse(x, 1), 1)  # columns sum to 1
            x = x - tl.expand_dims(_lse(x, 2), 2)  # rows sum to 1
        o4 = t[:, None] * 4 + i4[None, :]
        tl.store(hpre_ptr + o4, h_pre, mask=mt[:, None])
        tl.store(hpost_ptr + o4, h_post, mask=mt[:, None])
        tl.store(hres_ptr + t[:, None, None] * 16 + i4[None, :, None] * 4 + i4[None, None, :],
                 tl.exp(x), mask=mt[:, None, None])

    @triton.jit
    def _coeffs_bwd_kernel(proj_ptr, sq_ptr, prm_ptr, dpre_ptr, dpost_ptr, dres_ptr, scr_ptr,
                           dproj_ptr, dsq_ptr, dprm_ptr, T, K, eps, ITERS: tl.constexpr,
                           M: tl.constexpr, BT: tl.constexpr):
        """Backward of :func:`_coeffs_fwd_kernel`: d_proj, d_sumsq per token and this program's
        partial sums of the parameter grads (packed like the parameters). The Sinkhorn sweeps are
        replayed into ``scr`` (this program's tokens only, L2-resident) and walked in reverse:
        for y = x - lse(x), dx = dy - exp(y) * sum(dy)."""
        pid = tl.program_id(0)
        t = pid * BT + tl.arange(0, BT)
        mt = t < T
        i4 = tl.arange(0, 4)
        (sq, rr, r, pre, post, res, b_pre, b_post, b_res, a_pre, a_post,
         a_res) = _coeffs_load(proj_ptr, sq_ptr, prm_ptr, t, mt, K, eps, M)
        o4 = t[:, None] * 4 + i4[None, :]
        o16 = t[:, None, None] * 16 + i4[None, :, None] * 4 + i4[None, None, :]
        scr = scr_ptr + t[:, None, None].to(tl.int64) * (2 * ITERS * 16) + i4[None, :, None] * 4 \
            + i4[None, None, :]
        x = a_res * res + b_res[None, :, :]
        for k in range(ITERS):
            x = x - tl.expand_dims(_lse(x, 1), 1)
            tl.store(scr + (2 * k) * 16, x, mask=mt[:, None, None])
            x = x - tl.expand_dims(_lse(x, 2), 2)
            tl.store(scr + (2 * k + 1) * 16, x, mask=mt[:, None, None])
        # d h_res -> d (Sinkhorn input)
        dy = tl.load(dres_ptr + o16, mask=mt[:, None, None], other=0.0) * tl.exp(x)
        for kk in range(ITERS):
            k = ITERS - 1 - kk
            y = tl.load(scr + (2 * k + 1) * 16, mask=mt[:, None, None], other=0.0)
            dy = dy - tl.exp(y) * tl.expand_dims(tl.sum(dy, axis=2), 2)
            y = tl.load(scr + (2 * k) * 16, mask=mt[:, None, None], other=0.0)
            dy = dy - tl.exp(y) * tl.expand_dims(tl.sum(dy, axis=1), 1)
        # gates
        sp = tl.sigmoid(a_pre * pre + b_pre[None, :])
        dz_pre = tl.load(dpre_ptr + o4, mask=mt[:, None], other=0.0) * sp * (1.0 - sp)
        sq2 = tl.sigmoid(a_post * post + b_post[None, :])
        dz_post = tl.load(dpost_ptr + o4, mask=mt[:, None], other=0.0) * 2.0 * sq2 * (1.0 - sq2)
        dz_pre = tl.where(mt[:, None], dz_pre, 0.0)
        dz_post = tl.where(mt[:, None], dz_post, 0.0)
        dy = tl.where(mt[:, None, None], dy, 0.0)
        # coeffs = proj / r: d_proj = a d_z / r, d_r = -sum(a d_z coeffs) / r
        dc_pre = a_pre * dz_pre
        dc_post = a_post * dz_post
        dc_res = a_res * dy
        dr = -(tl.sum(dc_pre * pre, axis=1) + tl.sum(dc_post * post, axis=1)
               + tl.sum(tl.sum(dc_res * res, axis=2), axis=1)) / r
        tl.store(dproj_ptr + t[:, None] * M + i4[None, :], dc_pre / r[:, None], mask=mt[:, None])
        tl.store(dproj_ptr + t[:, None] * M + 4 + i4[None, :], dc_post / r[:, None], mask=mt[:, None])
        tl.store(dproj_ptr + t[:, None, None] * M + 8 + i4[None, :, None] * 4 + i4[None, None, :],
                 dc_res / r[:, None, None], mask=mt[:, None, None])
        # r = sqrt(sumsq / K) (clamped at eps: no gradient below)
        dsq = tl.where(rr > eps, dr / (2.0 * K * tl.maximum(rr, eps)), 0.0)
        tl.store(dsq_ptr + t, dsq, mask=mt)
        # parameter partials
        out = dprm_ptr + pid * 32
        tl.store(out + i4, tl.sum(dz_pre, axis=0))
        tl.store(out + 4 + i4, tl.sum(dz_post, axis=0))
        tl.store(out + 8 + i4[:, None] * 4 + i4[None, :], tl.sum(dy, axis=0))
        tl.store(out + 24, tl.sum(tl.sum(dz_pre * pre, axis=1), axis=0))
        tl.store(out + 25, tl.sum(tl.sum(dz_post * post, axis=1), axis=0))
        tl.store(out + 26, tl.sum(tl.sum(tl.sum(dy * res, axis=2), axis=1), axis=0))


    @triton.jit
    def _ds_stream(s_ptr, dsw_ptr, ds_ptr, offs, mt, h, dx, dp, dsq2, phit, acc,
                   HAS_DSW: tl.constexpr):
        """One stream of :func:`_dstreams_kernel`: store d_s_i, return d_phi_i's accumulator."""
        s = tl.load(s_ptr + offs, mask=mt[:, None], other=0.0).to(tl.float32)
        ds = h[:, None] * dx + tl.dot(dp, phit) + dsq2[:, None] * s
        if HAS_DSW:
            ds += tl.load(dsw_ptr + offs, mask=mt[:, None], other=0.0).to(tl.float32)
        tl.store(ds_ptr + offs, ds.to(tl.bfloat16), mask=mt[:, None])
        return tl.dot(tl.trans(s), dp, acc)

    _DS_CONFIGS = [
        triton.Config({"BC": bc, "BT": bt, "SPLITS": sp}, num_warps=w, num_stages=st)
        for bc in (32, 64, 128)
        for bt in (16, 32, 64)
        for sp in (8, 16, 32)
        for w in (4, 8)
        for st in (2, 3)
        if bc * bt <= 4096
    ]

    @triton.autotune(configs=_DS_CONFIGS, key=["C", "T_BUCKET", "HAS_DSW"])
    @triton.jit
    def _dstreams_kernel(s_ptr, dsw_ptr, dx_ptr, hpre_ptr, dp_ptr, dsq_ptr, phi_ptr, ds_ptr,
                         dphi_ptr, T, T_BUCKET, C: tl.constexpr, M: tl.constexpr,
                         HAS_DSW: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                         SPLITS: tl.constexpr):
        """d_s and this program's share of d_phi, for one C-block over one token split (n = 4).

        Owning a C-block keeps its rows of phi in registers for the whole split and lets d_phi
        accumulate there: one [4 BC, M] partial per (C-block, split), summed by the caller. The
        four streams run inside the token loop, so d_x, d_proj and h_pre are read once."""
        cb = tl.program_id(0)
        sp = tl.program_id(1)
        c = cb * BC + tl.arange(0, BC)
        m = tl.arange(0, M)
        p0 = tl.load(phi_ptr + (0 * C + c[None, :]) * M + m[:, None])  # phi_i^T [M, BC]
        p1 = tl.load(phi_ptr + (1 * C + c[None, :]) * M + m[:, None])
        p2 = tl.load(phi_ptr + (2 * C + c[None, :]) * M + m[:, None])
        p3 = tl.load(phi_ptr + (3 * C + c[None, :]) * M + m[:, None])
        a0 = tl.zeros((BC, M), dtype=tl.float32)
        a1 = tl.zeros((BC, M), dtype=tl.float32)
        a2 = tl.zeros((BC, M), dtype=tl.float32)
        a3 = tl.zeros((BC, M), dtype=tl.float32)
        TPS = tl.cdiv(tl.cdiv(T, SPLITS), BT) * BT
        t0 = sp * TPS
        for tt in range(t0, tl.minimum(t0 + TPS, T), BT):
            t = tt + tl.arange(0, BT)
            mt = t < T
            row = t[:, None].to(tl.int64)
            dx = tl.load(dx_ptr + row * C + c[None, :], mask=mt[:, None], other=0.0).to(tl.float32)
            dp = tl.load(dp_ptr + t[:, None] * M + m[None, :], mask=mt[:, None], other=0.0)
            dsq2 = 2.0 * tl.load(dsq_ptr + t, mask=mt, other=0.0)
            h0 = tl.load(hpre_ptr + t * 4 + 0, mask=mt, other=0.0).to(tl.bfloat16).to(tl.float32)
            h1 = tl.load(hpre_ptr + t * 4 + 1, mask=mt, other=0.0).to(tl.bfloat16).to(tl.float32)
            h2 = tl.load(hpre_ptr + t * 4 + 2, mask=mt, other=0.0).to(tl.bfloat16).to(tl.float32)
            h3 = tl.load(hpre_ptr + t * 4 + 3, mask=mt, other=0.0).to(tl.bfloat16).to(tl.float32)
            base = row * (4 * C) + c[None, :]
            a0 = _ds_stream(s_ptr, dsw_ptr, ds_ptr, base, mt, h0, dx, dp, dsq2, p0, a0, HAS_DSW)
            a1 = _ds_stream(s_ptr, dsw_ptr, ds_ptr, base + C, mt, h1, dx, dp, dsq2, p1, a1, HAS_DSW)
            a2 = _ds_stream(s_ptr, dsw_ptr, ds_ptr, base + 2 * C, mt, h2, dx, dp, dsq2, p2, a2,
                            HAS_DSW)
            a3 = _ds_stream(s_ptr, dsw_ptr, ds_ptr, base + 3 * C, mt, h3, dx, dp, dsq2, p3, a3,
                            HAS_DSW)
        out = dphi_ptr + (sp * 4 * C + c[:, None]) * M + m[None, :]
        tl.store(out, a0)
        tl.store(out + C * M, a1)
        tl.store(out + 2 * C * M, a2)
        tl.store(out + 3 * C * M, a3)

    @triton.jit
    def _ld(ptr, offs, mt):
        return tl.load(ptr + offs, mask=mt[:, None], other=0.0).to(tl.float32)

    @triton.jit
    def _ldv(ptr, t, k, stride, mt):
        """Per-token scalar k of a [T, stride] coefficient tensor (bf16 values as the write used)."""
        return tl.load(ptr + t * stride + k, mask=mt, other=0.0).to(tl.float32)

    _FB_CONFIGS = [
        triton.Config({"BC": bc, "BT": bt, "SPLITS": sp}, num_warps=w, num_stages=st)
        for bc in (32, 64, 128)
        for bt in (16, 32)
        for sp in (8, 16, 32)
        for w in (4, 8)
        for st in (1, 2)
        if bc * bt <= 2048
    ]

    @triton.autotune(configs=_FB_CONFIGS, key=["C", "T_BUCKET", "HAS_DSW"])
    @triton.jit
    def _fused_bwd_kernel(sk_ptr, f_ptr, post_ptr, res_ptr, dsw_ptr, dx_ptr, hpre_ptr, dp_ptr,
                          dsq_ptr, phi_ptr, dsk_ptr, df_ptr, dpr_ptr, dphi_ptr, T, T_BUCKET,
                          C: tl.constexpr, M: tl.constexpr, HAS_DSW: tl.constexpr,
                          BT: tl.constexpr, BC: tl.constexpr, SPLITS: tl.constexpr):
        """Read backward of sublayer k+1 and write backward of sublayer k in one pass (n = 4).

        S = S_{k+1} is recomputed from the write's inputs (S_k, f_k, post, res), its gradient
        dS = dS_write + h_pre dx + phi dproj + 2 dsumsq S stays in registers, and the write's
        backward is applied to it at once: dS_k = res^T dS, df = post . dS, and per token the
        C-block's share of d_post = dS . f and d_res = dS . S_k (summed over C-blocks by the
        caller). d_phi accumulates per (C-block, token split) as in :func:`_dstreams_kernel`."""
        cb = tl.program_id(0)
        sp = tl.program_id(1)
        c = cb * BC + tl.arange(0, BC)
        m = tl.arange(0, M)
        P0 = tl.load(phi_ptr + (0 * C + c[None, :]) * M + m[:, None])
        P1 = tl.load(phi_ptr + (1 * C + c[None, :]) * M + m[:, None])
        P2 = tl.load(phi_ptr + (2 * C + c[None, :]) * M + m[:, None])
        P3 = tl.load(phi_ptr + (3 * C + c[None, :]) * M + m[:, None])
        a0 = tl.zeros((BC, M), dtype=tl.float32)
        a1 = tl.zeros((BC, M), dtype=tl.float32)
        a2 = tl.zeros((BC, M), dtype=tl.float32)
        a3 = tl.zeros((BC, M), dtype=tl.float32)
        TPS = tl.cdiv(tl.cdiv(T, SPLITS), BT) * BT
        t0 = sp * TPS
        for tt in range(t0, tl.minimum(t0 + TPS, T), BT):
            t = tt + tl.arange(0, BT)
            mt = t < T
            row = t[:, None].to(tl.int64)
            o = row * (4 * C) + c[None, :]
            of = row * C + c[None, :]
            sk0 = _ld(sk_ptr, o, mt)
            sk1 = _ld(sk_ptr, o + C, mt)
            sk2 = _ld(sk_ptr, o + 2 * C, mt)
            sk3 = _ld(sk_ptr, o + 3 * C, mt)
            f = _ld(f_ptr, of, mt)
            dx = _ld(dx_ptr, of, mt)
            dp = tl.load(dp_ptr + t[:, None] * M + m[None, :], mask=mt[:, None], other=0.0)
            dsq2 = 2.0 * tl.load(dsq_ptr + t, mask=mt, other=0.0)
            # the write's coefficients: res[j, i] (target j, source i), post[j]
            r00 = _ldv(res_ptr, t, 0, 16, mt)[:, None]
            r01 = _ldv(res_ptr, t, 1, 16, mt)[:, None]
            r02 = _ldv(res_ptr, t, 2, 16, mt)[:, None]
            r03 = _ldv(res_ptr, t, 3, 16, mt)[:, None]
            r10 = _ldv(res_ptr, t, 4, 16, mt)[:, None]
            r11 = _ldv(res_ptr, t, 5, 16, mt)[:, None]
            r12 = _ldv(res_ptr, t, 6, 16, mt)[:, None]
            r13 = _ldv(res_ptr, t, 7, 16, mt)[:, None]
            r20 = _ldv(res_ptr, t, 8, 16, mt)[:, None]
            r21 = _ldv(res_ptr, t, 9, 16, mt)[:, None]
            r22 = _ldv(res_ptr, t, 10, 16, mt)[:, None]
            r23 = _ldv(res_ptr, t, 11, 16, mt)[:, None]
            r30 = _ldv(res_ptr, t, 12, 16, mt)[:, None]
            r31 = _ldv(res_ptr, t, 13, 16, mt)[:, None]
            r32 = _ldv(res_ptr, t, 14, 16, mt)[:, None]
            r33 = _ldv(res_ptr, t, 15, 16, mt)[:, None]
            p0 = _ldv(post_ptr, t, 0, 4, mt)[:, None]
            p1 = _ldv(post_ptr, t, 1, 4, mt)[:, None]
            p2 = _ldv(post_ptr, t, 2, 4, mt)[:, None]
            p3 = _ldv(post_ptr, t, 3, 4, mt)[:, None]
            h0 = _ldv(hpre_ptr, t, 0, 4, mt).to(tl.bfloat16).to(tl.float32)[:, None]
            h1 = _ldv(hpre_ptr, t, 1, 4, mt).to(tl.bfloat16).to(tl.float32)[:, None]
            h2 = _ldv(hpre_ptr, t, 2, 4, mt).to(tl.bfloat16).to(tl.float32)[:, None]
            h3 = _ldv(hpre_ptr, t, 3, 4, mt).to(tl.bfloat16).to(tl.float32)[:, None]
            # S_{k+1}, rounded like the forward's bf16 output
            s0 = (r00 * sk0 + r01 * sk1 + r02 * sk2 + r03 * sk3 + p0 * f).to(tl.bfloat16).to(tl.float32)
            s1 = (r10 * sk0 + r11 * sk1 + r12 * sk2 + r13 * sk3 + p1 * f).to(tl.bfloat16).to(tl.float32)
            s2 = (r20 * sk0 + r21 * sk1 + r22 * sk2 + r23 * sk3 + p2 * f).to(tl.bfloat16).to(tl.float32)
            s3 = (r30 * sk0 + r31 * sk1 + r32 * sk2 + r33 * sk3 + p3 * f).to(tl.bfloat16).to(tl.float32)
            d = dsq2[:, None]
            g0 = h0 * dx + tl.dot(dp, P0) + d * s0
            g1 = h1 * dx + tl.dot(dp, P1) + d * s1
            g2 = h2 * dx + tl.dot(dp, P2) + d * s2
            g3 = h3 * dx + tl.dot(dp, P3) + d * s3
            if HAS_DSW:
                g0 += _ld(dsw_ptr, o, mt)
                g1 += _ld(dsw_ptr, o + C, mt)
                g2 += _ld(dsw_ptr, o + 2 * C, mt)
                g3 += _ld(dsw_ptr, o + 3 * C, mt)
            a0 = tl.dot(tl.trans(s0), dp, a0)
            a1 = tl.dot(tl.trans(s1), dp, a1)
            a2 = tl.dot(tl.trans(s2), dp, a2)
            a3 = tl.dot(tl.trans(s3), dp, a3)
            # the write's backward on g = dS_{k+1}
            ms = mt[:, None]
            tl.store(dsk_ptr + o, (r00 * g0 + r10 * g1 + r20 * g2 + r30 * g3).to(tl.bfloat16), mask=ms)
            tl.store(dsk_ptr + o + C, (r01 * g0 + r11 * g1 + r21 * g2 + r31 * g3).to(tl.bfloat16),
                     mask=ms)
            tl.store(dsk_ptr + o + 2 * C, (r02 * g0 + r12 * g1 + r22 * g2 + r32 * g3).to(tl.bfloat16),
                     mask=ms)
            tl.store(dsk_ptr + o + 3 * C, (r03 * g0 + r13 * g1 + r23 * g2 + r33 * g3).to(tl.bfloat16),
                     mask=ms)
            tl.store(df_ptr + of, (p0 * g0 + p1 * g1 + p2 * g2 + p3 * g3).to(tl.bfloat16), mask=ms)
            q = dpr_ptr + (cb * T + t).to(tl.int64) * 20
            tl.store(q + 0, tl.sum(g0 * f, axis=1), mask=mt)
            tl.store(q + 1, tl.sum(g1 * f, axis=1), mask=mt)
            tl.store(q + 2, tl.sum(g2 * f, axis=1), mask=mt)
            tl.store(q + 3, tl.sum(g3 * f, axis=1), mask=mt)
            tl.store(q + 4, tl.sum(g0 * sk0, axis=1), mask=mt)
            tl.store(q + 5, tl.sum(g0 * sk1, axis=1), mask=mt)
            tl.store(q + 6, tl.sum(g0 * sk2, axis=1), mask=mt)
            tl.store(q + 7, tl.sum(g0 * sk3, axis=1), mask=mt)
            tl.store(q + 8, tl.sum(g1 * sk0, axis=1), mask=mt)
            tl.store(q + 9, tl.sum(g1 * sk1, axis=1), mask=mt)
            tl.store(q + 10, tl.sum(g1 * sk2, axis=1), mask=mt)
            tl.store(q + 11, tl.sum(g1 * sk3, axis=1), mask=mt)
            tl.store(q + 12, tl.sum(g2 * sk0, axis=1), mask=mt)
            tl.store(q + 13, tl.sum(g2 * sk1, axis=1), mask=mt)
            tl.store(q + 14, tl.sum(g2 * sk2, axis=1), mask=mt)
            tl.store(q + 15, tl.sum(g2 * sk3, axis=1), mask=mt)
            tl.store(q + 16, tl.sum(g3 * sk0, axis=1), mask=mt)
            tl.store(q + 17, tl.sum(g3 * sk1, axis=1), mask=mt)
            tl.store(q + 18, tl.sum(g3 * sk2, axis=1), mask=mt)
            tl.store(q + 19, tl.sum(g3 * sk3, axis=1), mask=mt)
        out = dphi_ptr + (sp * 4 * C + c[:, None]) * M + m[None, :]
        tl.store(out, a0)
        tl.store(out + C * M, a1)
        tl.store(out + 2 * C * M, a2)
        tl.store(out + 3 * C * M, a3)


_BT_COEFFS = 32


def _pack(params):
    """[b_pre 4 | b_post 4 | b_res 16 | a_pre | a_post | a_res | 0 x 5] fp32 (n = 4)."""
    b_pre, b_post, b_res, a_pre, a_post, a_res = params
    return torch.cat([b_pre.detach().float().reshape(-1), b_post.detach().float().reshape(-1),
                      b_res.detach().float().reshape(-1), a_pre.detach().float().reshape(-1),
                      a_post.detach().float().reshape(-1), a_res.detach().float().reshape(-1),
                      b_pre.new_zeros(5, dtype=torch.float32)])


def _unpack(flat, params):
    out, o = [], 0
    for p, size in zip(params, (4, 4, 16, 1, 1, 1)):
        out.append(flat[o:o + size].view(p.shape).to(p.dtype) if p.requires_grad else None)
        o += size
    return out


def _phi_tile(phi):
    """phi [n C, n (n + 2)] -> [n C, 32] zero-padded fp32 (the backward's dots; also the source of
    the forward's bf16 copy, rounded like the eager ``phi.to(bf16)``)."""
    out = torch.zeros(phi.shape[0], _M, dtype=torch.float32, device=phi.device)
    out[:, : phi.shape[1]] = phi.detach().float()
    return out


class _FusedCoeffsRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams, phi, b_pre, b_post, b_res, a_pre, a_post, a_res, n, eps, iters, link):
        s_, b_, _, c = streams.shape
        T, K = s_ * b_, n * c
        rec = getattr(streams, "_mhc_write_record", None) if FUSE_WRITE_BWD else None
        ctx.record = rec if rec is not None and rec.inputs is not None else None
        streams = streams.contiguous()
        phi32 = _phi_tile(phi)
        phi16 = phi32.to(torch.bfloat16)
        proj = torch.empty(T, _M, dtype=torch.float32, device=streams.device)
        sumsq = torch.empty(T, dtype=torch.float32, device=streams.device)
        _proj_kernel[(triton.cdiv(T, 64),)](streams, phi16, proj, sumsq, T, K=K, M=_M, BT=64,
                                            BK=64, num_warps=4, num_stages=3)
        params = (b_pre, b_post, b_res, a_pre, a_post, a_res)
        prm = _pack(params)
        h_pre = torch.empty(T, n, dtype=torch.float32, device=streams.device)
        h_post = torch.empty(T, n, dtype=torch.float32, device=streams.device)
        h_res = torch.empty(T, n * n, dtype=torch.float32, device=streams.device)
        _coeffs_fwd_kernel[(triton.cdiv(T, _BT_COEFFS),)](
            proj, sumsq, prm, h_pre, h_post, h_res, T, float(K), float(eps), ITERS=iters, M=_M,
            BT=_BT_COEFFS, num_warps=4)
        x = torch.empty(T, c, dtype=torch.bfloat16, device=streams.device)
        _read_kernel[(triton.cdiv(T, 32), c // 128)](streams, h_pre, x, T, C=c, N=n, BT=32,
                                                     BC=128, num_warps=4)
        ctx.save_for_backward(streams, phi, proj, sumsq, h_pre, *params)
        ctx.cfg = (n, K, eps, iters, s_, b_, c)
        ctx.link = link
        return x.view(s_, b_, c), h_post.view(s_, b_, n), h_res.view(s_, b_, n, n)

    @staticmethod
    def backward(ctx, d_x, d_h_post, d_h_res):
        streams, phi, proj, sumsq, h_pre, *params = ctx.saved_tensors
        n, K, eps, iters, s_, b_, c = ctx.cfg
        T = s_ * b_
        dsw, ctx.link.grad = ctx.link.grad, None
        d_x = (torch.zeros(T, c, dtype=torch.bfloat16, device=streams.device) if d_x is None
               else d_x.contiguous().view(T, c))
        d_hpre = torch.empty(T, n, dtype=torch.float32, device=streams.device)
        _dhpre_kernel[(triton.cdiv(T, 16),)](streams, d_x, d_hpre, T, C=c, N=n, BT=16, BC=128,
                                             num_warps=4)
        # coefficient backward: one kernel, Sinkhorn replayed in an L2-resident scratch
        z = lambda shape: torch.zeros(shape, dtype=torch.float32, device=streams.device)  # noqa: E731
        d_hp = z((T, n)) if d_h_post is None else d_h_post.float().contiguous().view(T, n)
        d_hr = z((T, n * n)) if d_h_res is None else d_h_res.float().contiguous().view(T, n * n)
        grid = triton.cdiv(T, _BT_COEFFS)
        scratch = torch.empty(T, 2 * iters * 16, dtype=torch.float32, device=streams.device)
        d_proj = z((T, _M))
        d_sumsq = torch.empty(T, dtype=torch.float32, device=streams.device)
        d_prm = torch.empty(grid, 32, dtype=torch.float32, device=streams.device)
        _coeffs_bwd_kernel[(grid,)](
            proj, sumsq, _pack(params), d_hpre, d_hp, d_hr, scratch, d_proj, d_sumsq, d_prm, T,
            float(K), float(eps), ITERS=iters, M=_M, BT=_BT_COEFFS, num_warps=4)
        del scratch
        d_params = _unpack(d_prm.sum(0), params)
        phi32 = _phi_tile(phi)
        rec = ctx.record
        ctx.record = None
        if rec is not None and rec.inputs is not None:
            return _fused_with_write(rec, streams, dsw, d_x, h_pre, d_proj, d_sumsq, phi32, phi,
                                     d_params, T, K, c)
        d_s = torch.empty_like(streams)
        # one [K, M] partial per token split (autotuned: at most 32)
        d_phi_parts = torch.empty(32, K, _M, dtype=torch.float32, device=streams.device)
        dsw_arg = dsw.contiguous() if dsw is not None else streams
        _dstreams_kernel[lambda meta: (c // meta["BC"], meta["SPLITS"])](
            streams, dsw_arg, d_x, h_pre, d_proj.contiguous(), d_sumsq.contiguous(), phi32, d_s,
            d_phi_parts, T, triton.next_power_of_2(T), C=c, M=_M, HAS_DSW=dsw is not None,
        )
        splits = _dstreams_kernel.best_config.kwargs["SPLITS"]
        d_phi = d_phi_parts[:splits].sum(0)[:, : phi.shape[1]].to(phi.dtype)
        return (d_s, d_phi, *d_params, None, None, None, None)


def _fused_with_write(rec, streams, dsw, d_x, h_pre, d_proj, d_sumsq, phi32, phi, d_params, T, K,
                      c):
    """The fused read backward that also runs the previous write's backward (see WriteRecord)."""
    sk, f, post, res = (t.contiguous() for t in rec.inputs)
    rec.inputs = None
    dev = streams.device
    d_sk = torch.empty_like(sk)
    d_f = torch.empty_like(f)
    d_phi_parts = torch.empty(32, K, _M, dtype=torch.float32, device=dev)
    # per (C-block, token) partial sums of d_post (4) and d_res (16); at most C / 32 blocks
    dpr = torch.empty(c // 32, T, 20, dtype=torch.float32, device=dev)
    dsw_arg = dsw.contiguous() if dsw is not None else streams
    _fused_bwd_kernel[lambda meta: (c // meta["BC"], meta["SPLITS"])](
        sk, f, post, res, dsw_arg, d_x, h_pre, d_proj.contiguous(), d_sumsq.contiguous(), phi32,
        d_sk, d_f, dpr, d_phi_parts, T, triton.next_power_of_2(T), C=c, M=_M,
        HAS_DSW=dsw is not None,
    )
    cfg = _fused_bwd_kernel.best_config.kwargs
    pr = dpr[: c // cfg["BC"]].sum(0)
    d_post = pr[:, :4].reshape(post.shape).to(post.dtype)
    d_res = pr[:, 4:].reshape(res.shape).to(res.dtype)
    rec.grads = (d_sk, d_f, d_post, d_res)
    d_phi = d_phi_parts[: cfg["SPLITS"]].sum(0)[:, : phi.shape[1]].to(phi.dtype)
    return (dummy_grad(streams), d_phi, *d_params, None, None, None, None)


def coeffs_and_read(mixer, streams, link):
    """``(layer_input, h_pre, h_post, h_res)`` like :meth:`MHCMixer.coeffs_and_read` (h_pre None:
    the fused read keeps it)."""
    x, h_post, h_res = _FusedCoeffsRead.apply(
        streams, mixer.phi, mixer.b_pre, mixer.b_post, mixer.b_res, mixer.alpha_pre,
        mixer.alpha_post, mixer.alpha_res, streams.shape[-2], mixer.eps, mixer.num_sinkhorn_iters,
        link,
    )
    return x, None, h_post, h_res
