# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Power attention (the quadratic form of power retention) as a flash-attention-style kernel.

    p_ts = (scale q_t . k_s)^2 exp(L_t - L_s)   (s <= t; L = cumsum of the log gate)
    o_t  = sum_s p_ts v_s / (sum_s p_ts + eps)

The weights are non-negative and bounded (L_t - L_s <= 0 under the causal mask), so unlike
softmax attention no running maximum / rescaling is needed: one pass accumulates the
unnormalised numerator and the row sum. Backward, with g_ts = (do_t . v_s - do_t . o_t) / l_t:

    dv_s = sum_t p_ts / l_t do_t       ds_ts = 2 g_ts p_ts / s_ts = 2 g_ts s_ts exp(L_t - L_s)
    dq_t = scale sum_s ds_ts k_s       dk_s = scale sum_t ds_ts q_t
    dL_s = -sum_t g_ts p_ts            (dL_t from the query side is sum_s g_ts p_ts = 0)

[B, T, H, D] bf16 in and out, D in {32, 64, 128, 256}; the log gate [B, T, H] fp32 or None.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-6


@triton.jit
def _fwd(Q, K, V, L, O, LSUM, T, H, scale, stride_b, stride_t, stride_lb, stride_lt,
         D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GATED: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    base = b * stride_b + h * D
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    if GATED:
        lq = tl.load(L + b * stride_lb + h + offs_m * stride_lt, mask=offs_m < T, other=0.0)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    lsum = tl.zeros([BM], dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, T)
    for start in range(0, hi, BN):
        offs_n = start + tl.arange(0, BN)
        kv_mask = offs_n[:, None] < T
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        p = s * s
        if GATED:
            lk = tl.load(L + b * stride_lb + h + offs_n * stride_lt, mask=offs_n < T, other=0.0)
            p = p * tl.exp(tl.minimum(lq[:, None] - lk[None, :], 0.0))
        p = tl.where((offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < T), p, 0.0)
        lsum += tl.sum(p, 1)
        acc += tl.dot(p.to(v.dtype), v)
    o = acc / (lsum[:, None] + 1e-6)
    tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :], o.to(O.dtype.element_ty),
             mask=offs_m[:, None] < T)
    tl.store(LSUM + bh * T + offs_m, lsum + 1e-6, mask=offs_m < T)


@triton.jit
def _bwd_dkv(Q, K, V, L, DO, LSUM, DELTA, DK, DV, DLK, T, H, scale, stride_b, stride_t,
             stride_lb, stride_lt, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             GATED: tl.constexpr):
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    base = b * stride_b + h * D
    kv_mask = offs_n[:, None] < T
    k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
    v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
    if GATED:
        lk = tl.load(L + b * stride_lb + h + offs_n * stride_lt, mask=offs_n < T, other=0.0)
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    dlk = tl.zeros([BN], dtype=tl.float32)
    for start in range(pid_n * BN, T, BM):
        offs_m = start + tl.arange(0, BM)
        q_mask = offs_m[:, None] < T
        q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
        lsum = tl.load(LSUM + bh * T + offs_m, mask=offs_m < T, other=1.0)
        delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale                                  # [BM, BN]
        e = tl.full([BM, BN], 1.0, tl.float32)
        if GATED:
            lq = tl.load(L + b * stride_lb + h + offs_m * stride_lt, mask=offs_m < T, other=0.0)
            e = tl.exp(tl.minimum(lq[:, None] - lk[None, :], 0.0))
        causal = (offs_m[:, None] >= offs_n[None, :]) & (offs_m[:, None] < T)
        e = tl.where(causal, e, 0.0)
        p = s * s * e
        pn = p / lsum[:, None]
        dv += tl.dot(tl.trans(pn.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))                                        # do_t . v_s
        g = (dp - delta[:, None]) / lsum[:, None]
        ds = 2.0 * g * s * e
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        if GATED:
            dlk -= tl.sum(g * p, 0)
    tl.store(DK + base + offs_n[:, None] * stride_t + offs_d[None, :],
             (dk * scale).to(DK.dtype.element_ty), mask=kv_mask)
    tl.store(DV + base + offs_n[:, None] * stride_t + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=kv_mask)
    if GATED:
        tl.store(DLK + bh * T + offs_n, dlk, mask=offs_n < T)


@triton.jit
def _bwd_dq(Q, K, V, L, DO, LSUM, DELTA, DQ, T, H, scale, stride_b, stride_t, stride_lb,
            stride_lt, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GATED: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    base = b * stride_b + h * D
    q_mask = offs_m[:, None] < T
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
    lsum = tl.load(LSUM + bh * T + offs_m, mask=offs_m < T, other=1.0)
    delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
    if GATED:
        lq = tl.load(L + b * stride_lb + h + offs_m * stride_lt, mask=offs_m < T, other=0.0)
    dq = tl.zeros([BM, D], dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, T)
    for start in range(0, hi, BN):
        offs_n = start + tl.arange(0, BN)
        kv_mask = offs_n[:, None] < T
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        e = tl.full([BM, BN], 1.0, tl.float32)
        if GATED:
            lk = tl.load(L + b * stride_lb + h + offs_n * stride_lt, mask=offs_n < T, other=0.0)
            e = tl.exp(tl.minimum(lq[:, None] - lk[None, :], 0.0))
        e = tl.where((offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < T), e, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = 2.0 * (dp - delta[:, None]) / lsum[:, None] * s * e
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + base + offs_m[:, None] * stride_t + offs_d[None, :],
             (dq * scale).to(DQ.dtype.element_ty), mask=q_mask)


def _blocks(d: int):
    # (BM, BN, num_warps, num_stages): smaller tiles for wide heads (shared memory)
    if d <= 64:
        return 64, 64, 4, 3
    if d <= 128:
        return 64, 64, 4, 2
    return 32, 32, 4, 1


class _PowerAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, L, scale):
        B, T, H, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        gated = L is not None
        Lc = L.contiguous() if gated else q.new_empty(1, dtype=torch.float32)
        o = torch.empty_like(q)
        lsum = torch.empty(B * H, T, device=q.device, dtype=torch.float32)
        BM, BN, warps, stages = _blocks(D)
        _fwd[(triton.cdiv(T, BM), B * H)](
            q, k, v, Lc, o, lsum, T, H, scale, q.stride(0), q.stride(1),
            Lc.stride(0) if gated else 0, Lc.stride(1) if gated else 0,
            D=D, BM=BM, BN=BN, GATED=gated, num_warps=warps, num_stages=stages)
        ctx.save_for_backward(q, k, v, Lc, o, lsum)
        ctx.scale, ctx.gated = scale, gated
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, Lc, o, lsum = ctx.saved_tensors
        B, T, H, D = q.shape
        do = do.contiguous()
        # delta_t = do_t . o_t, laid out [B*H, T] as lsum
        delta = (do.float() * o.float()).sum(-1).permute(0, 2, 1).reshape(B * H, T).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dlk = torch.zeros(B * H, T, device=q.device, dtype=torch.float32)
        BM, BN, warps, stages = _blocks(D)
        sl = (Lc.stride(0) if ctx.gated else 0, Lc.stride(1) if ctx.gated else 0)
        _bwd_dkv[(triton.cdiv(T, BN), B * H)](
            q, k, v, Lc, do, lsum, delta, dk, dv, dlk, T, H, ctx.scale, q.stride(0), q.stride(1),
            *sl, D=D, BM=BM, BN=BN, GATED=ctx.gated, num_warps=warps, num_stages=stages)
        _bwd_dq[(triton.cdiv(T, BM), B * H)](
            q, k, v, Lc, do, lsum, delta, dq, T, H, ctx.scale, q.stride(0), q.stride(1),
            *sl, D=D, BM=BM, BN=BN, GATED=ctx.gated, num_warps=warps, num_stages=stages)
        dL = dlk.view(B, H, T).permute(0, 2, 1) if ctx.gated else None
        return dq, dk, dv, dL, None


def power_attention(q, k, v, log_g=None, scale=None):
    """Degree-2 power attention, [B, T, H, D] -> [B, T, H, D] (see the module docstring);
    ``log_g`` [B, T, H] (<= 0) is the per-token log gate (cumulated here)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    L = log_g.float().cumsum(1) if log_g is not None else None
    return _PowerAttention.apply(q, k, v, L, scale)
