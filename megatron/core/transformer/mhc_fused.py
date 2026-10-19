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
the coefficients from ``(proj, sumsq)`` (a compiled function of ``[T, 24]`` tensors -- the same
math as :func:`~megatron.core.transformer.mhc._mhc_coeffs`), :func:`_read_kernel` (second read).

Backward, given ``d_x`` and the coefficient grads from the write:

.. code-block:: text

    d_h_pre_i = sum_c d_x s_i                                         _dhpre_kernel
    (d_proj, d_sumsq) <- autograd of the coefficient function          [T, 24] tensors
    d_s_i     = d_s_write_i + h_pre_i d_x + phi_i d_proj + 2 d_sumsq s_i   _dstreams_kernel
    d_phi     = flat^T d_proj                    (same kernel, per-split partials, summed)

``d_s_write`` is the write's gradient for the same stream state: the write hands it over through
the sublayer's :class:`ReadWriteLink` instead of returning it, so autograd's accumulation of the
two -- one more full pass over the widest tensor -- disappears into the kernel. Within one
sublayer only: nothing crosses layers, recompute chunks or pipeline stages.
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
        and n in (2, 4)
        and streams.shape[-1] % 64 == 0
    )


class ReadWriteLink:
    """Carries the write's stream gradient to the read of the same sublayer (see module doc)."""

    __slots__ = ("grad",)

    def __init__(self):
        self.grad = None


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
    def _dstreams_kernel(s_ptr, dsw_ptr, dx_ptr, hpre_ptr, dp_ptr, dsq_ptr, phi_ptr, ds_ptr,
                         dphi_ptr, T, TPS, C: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
                         HAS_DSW: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
        """d_s and this program's share of d_phi, for one C-block over one token split.

        Owning a C-block keeps its rows of phi in registers for the whole split and lets d_phi
        accumulate there: one [n BC, M] partial per (C-block, split), summed by the caller."""
        cb = tl.program_id(0)
        sp = tl.program_id(1)
        c = cb * BC + tl.arange(0, BC)
        m = tl.arange(0, M)
        t0 = sp * TPS
        for i in tl.static_range(N):
            # phi_i^T [M, BC] (fp32, tf32 dots) and the d_phi accumulator [BC, M]
            phit = tl.load(phi_ptr + (i * C + c[None, :]) * M + m[:, None])
            dphi = tl.zeros((BC, M), dtype=tl.float32)
            for tt in range(t0, t0 + TPS, BT):
                t = tt + tl.arange(0, BT)
                mt = t < T
                row = t[:, None].to(tl.int64)
                dx = tl.load(dx_ptr + row * C + c[None, :], mask=mt[:, None], other=0.0).to(tl.float32)
                h = tl.load(hpre_ptr + t * N + i, mask=mt, other=0.0).to(tl.bfloat16).to(tl.float32)
                dp = tl.load(dp_ptr + t[:, None] * M + m[None, :], mask=mt[:, None], other=0.0)
                dsq = tl.load(dsq_ptr + t, mask=mt, other=0.0)
                offs = row * (N * C) + i * C + c[None, :]
                s = tl.load(s_ptr + offs, mask=mt[:, None], other=0.0).to(tl.float32)
                ds = h[:, None] * dx + tl.dot(dp, phit) + (2.0 * dsq)[:, None] * s
                if HAS_DSW:
                    ds += tl.load(dsw_ptr + offs, mask=mt[:, None], other=0.0).to(tl.float32)
                tl.store(ds_ptr + offs, ds.to(tl.bfloat16), mask=mt[:, None])
                dphi = tl.dot(tl.trans(s), dp, dphi)
            tl.store(dphi_ptr + (sp * N * C + i * C + c[:, None]) * M + m[None, :], dphi)


def _coeffs(proj, sumsq, b_pre, b_post, b_res, a_pre, a_post, a_res, n, k, eps, iters):
    """:func:`~megatron.core.transformer.mhc._mhc_coeffs` from the projection and |flat|^2."""
    from megatron.core.transformer.mhc import _sinkhorn_knopp_impl

    r = (sumsq.clamp(min=0) / k).sqrt().clamp(min=eps).unsqueeze(-1)
    coeffs = proj[..., : n * (n + 2)] / r
    pre_raw, post_raw, res_raw = torch.split(coeffs, [n, n, n * n], dim=-1)
    h_pre = torch.sigmoid(a_pre.float() * pre_raw + b_pre.float())
    h_post = 2.0 * torch.sigmoid(a_post.float() * post_raw + b_post.float())
    h_res = a_res.float() * res_raw.view(*res_raw.shape[:-1], n, n) + b_res.float()
    return h_pre, h_post, _sinkhorn_knopp_impl(h_res, iters, 1e-8)


_coeffs_compiled = None


def _coeffs_fn():
    global _coeffs_compiled
    if _coeffs_compiled is None:
        try:
            _coeffs_compiled = torch.compile(_coeffs, dynamic=False)
        except Exception:  # pragma: no cover
            _coeffs_compiled = _coeffs
    return _coeffs_compiled


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
        streams = streams.contiguous()
        phi32 = _phi_tile(phi)
        phi16 = phi32.to(torch.bfloat16)
        proj = torch.empty(T, _M, dtype=torch.float32, device=streams.device)
        sumsq = torch.empty(T, dtype=torch.float32, device=streams.device)
        _proj_kernel[(triton.cdiv(T, 64),)](streams, phi16, proj, sumsq, T, K=K, M=_M, BT=64,
                                            BK=64, num_warps=4, num_stages=3)
        params = (b_pre, b_post, b_res, a_pre, a_post, a_res)
        h_pre, h_post, h_res = _coeffs_fn()(proj, sumsq, *params, n, K, eps, iters)
        h_pre = h_pre.contiguous()
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
        # coefficient backward on [T, 24] tensors
        with torch.enable_grad():
            leaves = [proj.detach().requires_grad_(), sumsq.detach().requires_grad_()]
            pd = [p.detach().requires_grad_(p.requires_grad) for p in params]
            outs = _coeffs_fn()(*leaves, *pd, n, K, eps, iters)
            grads_out = [d_hpre, d_h_post, d_h_res]
            used = [(o, g.reshape(o.shape).float()) for o, g in zip(outs, grads_out) if g is not None]
            wrt = leaves + [p for p in pd if p.requires_grad]
            gr = torch.autograd.grad([o for o, _ in used], wrt, [g for _, g in used],
                                     allow_unused=True)
        d_proj = gr[0] if gr[0] is not None else torch.zeros_like(proj)
        d_sumsq = gr[1] if gr[1] is not None else torch.zeros_like(sumsq)
        it = iter(gr[2:])
        d_params = [next(it) if p.requires_grad else None for p in pd]
        phi32 = _phi_tile(phi)
        d_s = torch.empty_like(streams)
        BC, BT = 64, 32
        splits = max(1, min(16, triton.cdiv(T, 1024)))
        tps = triton.cdiv(triton.cdiv(T, splits), BT) * BT
        splits = triton.cdiv(T, tps)
        d_phi_parts = torch.empty(splits, K, _M, dtype=torch.float32, device=streams.device)
        dsw_arg = dsw.contiguous() if dsw is not None else streams
        _dstreams_kernel[(c // BC, splits)](
            streams, dsw_arg, d_x, h_pre, d_proj.contiguous(), d_sumsq.contiguous(), phi32, d_s,
            d_phi_parts, T, tps, C=c, N=n, M=_M, HAS_DSW=dsw is not None, BT=BT, BC=BC,
            num_warps=4,
        )
        d_phi = d_phi_parts.sum(0)[:, : phi.shape[1]].to(phi.dtype)
        return (d_s, d_phi, *d_params, None, None, None, None)


def coeffs_and_read(mixer, streams, link):
    """``(layer_input, h_pre, h_post, h_res)`` like :meth:`MHCMixer.coeffs_and_read` (h_pre None:
    the fused read keeps it)."""
    x, h_post, h_res = _FusedCoeffsRead.apply(
        streams, mixer.phi, mixer.b_pre, mixer.b_post, mixer.b_res, mixer.alpha_pre,
        mixer.alpha_post, mixer.alpha_res, streams.shape[-2], mixer.eps, mixer.num_sinkhorn_iters,
        link,
    )
    return x, None, h_post, h_res
