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

The gate runs in base 2 (L log2(e), exp2); the gradient formulas above are unchanged by it.
Off-diagonal key blocks run unmasked, the diagonal ones masked; tile sizes are autotuned.

[B, T, H, D] bf16 in and out, D in {32, 64, 128, 256}; the log gate [B, T, H] fp32 or None.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-6
LOG2E = 1.4426950408889634


def _configs(stages):
    return [
        triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
        for bm, bn, w in ((64, 64, 4), (128, 64, 8), (64, 32, 4), (128, 32, 4), (32, 32, 4))
        for s in stages
    ]


def _prune(configs, named_args, **kwargs):
    d = named_args["D"] if "D" in named_args else kwargs["D"]
    keep = [c for c in configs if d <= 128 or (c.kwargs["BM"] <= 64 and c.kwargs["BN"] <= 32)]
    return keep or configs[-1:]


@triton.jit
def _fwd_block(acc, lsum, q, lq, K, V, L, base, lbase, offs_m, start, T, scale, stride_t,
               stride_lt, D: tl.constexpr, BN: tl.constexpr, GATED: tl.constexpr,
               MASK: tl.constexpr):
    offs_n = start + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    if MASK:
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)
    else:
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :])
    s = tl.dot(q, tl.trans(k)) * scale
    p = s * s
    if GATED:
        if MASK:
            lk = tl.load(L + lbase + offs_n * stride_lt, mask=offs_n < T, other=0.0)
        else:
            lk = tl.load(L + lbase + offs_n * stride_lt)
        p = p * tl.exp2(tl.minimum(lq[:, None] - lk[None, :], 0.0))
    if MASK:
        p = tl.where((offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < T), p, 0.0)
    lsum += tl.sum(p, 1)
    acc += tl.dot(p.to(v.dtype), v)
    return acc, lsum


@triton.autotune(configs=_configs((2, 3)), key=["T", "D", "GATED"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _fwd(Q, K, V, L, O, LSUM, T, H, scale, stride_b, stride_t, stride_lb, stride_lt,
         D: tl.constexpr, GATED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    base = b * stride_b + h * D
    lbase = b * stride_lb + h
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0)
    lq = tl.zeros([BM], dtype=tl.float32)
    if GATED:
        lq = tl.load(L + lbase + offs_m * stride_lt, mask=offs_m < T, other=0.0)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    lsum = tl.zeros([BM], dtype=tl.float32)
    # key blocks entirely below the diagonal (and inside T): no mask
    full = (pid_m * BM) // BN * BN
    for start in range(0, full, BN):
        acc, lsum = _fwd_block(acc, lsum, q, lq, K, V, L, base, lbase, offs_m, start, T, scale,
                               stride_t, stride_lt, D, BN, GATED, False)
    for start in range(full, tl.minimum((pid_m + 1) * BM, T), BN):
        acc, lsum = _fwd_block(acc, lsum, q, lq, K, V, L, base, lbase, offs_m, start, T, scale,
                               stride_t, stride_lt, D, BN, GATED, True)
    o = acc / (lsum[:, None] + 1e-6)
    tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :], o.to(O.dtype.element_ty),
             mask=offs_m[:, None] < T)
    tl.store(LSUM + bh * T + offs_m, lsum + 1e-6, mask=offs_m < T)


@triton.jit
def _dkv_block(dk, dv, dlk, k, v, lk, Q, DO, L, LSUM, DELTA, base, lbase, bh, offs_n, start, T,
               scale, stride_t, stride_lt, D: tl.constexpr, BM: tl.constexpr,
               GATED: tl.constexpr, MASK: tl.constexpr):
    offs_m = start + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    if MASK:
        q_mask = offs_m[:, None] < T
        q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask,
                     other=0.0)
        lsum = tl.load(LSUM + bh * T + offs_m, mask=offs_m < T, other=1.0)
        delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
    else:
        q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :])
        do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :])
        lsum = tl.load(LSUM + bh * T + offs_m)
        delta = tl.load(DELTA + bh * T + offs_m)
    s = tl.dot(q, tl.trans(k)) * scale                                      # [BM, BN]
    rl = 1.0 / lsum
    if GATED:
        if MASK:
            lq = tl.load(L + lbase + offs_m * stride_lt, mask=offs_m < T, other=0.0)
        else:
            lq = tl.load(L + lbase + offs_m * stride_lt)
        se = s * tl.exp2(tl.minimum(lq[:, None] - lk[None, :], 0.0))
    else:
        se = s
    if MASK:
        se = tl.where((offs_m[:, None] >= offs_n[None, :]) & (offs_m[:, None] < T), se, 0.0)
    p = s * se                                                              # s^2 e
    dv += tl.dot(tl.trans((p * rl[:, None]).to(do.dtype)), do)
    dp = tl.dot(do, tl.trans(v))
    g = (dp - delta[:, None]) * rl[:, None]
    dk += tl.dot(tl.trans((2.0 * g * se).to(q.dtype)), q)
    if GATED:
        dlk -= tl.sum(g * p, 0)
    return dk, dv, dlk


@triton.autotune(configs=_configs((1, 2)), key=["T", "D", "GATED"],
                 prune_configs_by={"early_config_prune": _prune},
                 reset_to_zero=["DLK"])
@triton.jit
def _bwd_dkv(Q, K, V, L, DO, LSUM, DELTA, DK, DV, DLK, T, H, scale, stride_b, stride_t,
             stride_lb, stride_lt, D: tl.constexpr, GATED: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr):
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    base = b * stride_b + h * D
    lbase = b * stride_lb + h
    kv_mask = offs_n[:, None] < T
    k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
    v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
    lk = tl.zeros([BN], dtype=tl.float32)
    if GATED:
        lk = tl.load(L + lbase + offs_n * stride_lt, mask=offs_n < T, other=0.0)
    dk = tl.zeros([BN, D], dtype=tl.float32)
    dv = tl.zeros([BN, D], dtype=tl.float32)
    dlk = tl.zeros([BN], dtype=tl.float32)
    lo = (pid_n * BN) // BM * BM
    # query blocks touching the diagonal: masked; those fully below it (and inside T): not
    diag_end = tl.minimum(((pid_n + 1) * BN + BM - 1) // BM * BM, T)
    for start in range(lo, diag_end, BM):
        dk, dv, dlk = _dkv_block(dk, dv, dlk, k, v, lk, Q, DO, L, LSUM, DELTA, base, lbase, bh,
                                 offs_n, start, T, scale, stride_t, stride_lt, D, BM, GATED, True)
    full_end = T // BM * BM
    for start in range(diag_end, full_end, BM):
        dk, dv, dlk = _dkv_block(dk, dv, dlk, k, v, lk, Q, DO, L, LSUM, DELTA, base, lbase, bh,
                                 offs_n, start, T, scale, stride_t, stride_lt, D, BM, GATED, False)
    for start in range(tl.maximum(full_end, diag_end), T, BM):
        dk, dv, dlk = _dkv_block(dk, dv, dlk, k, v, lk, Q, DO, L, LSUM, DELTA, base, lbase, bh,
                                 offs_n, start, T, scale, stride_t, stride_lt, D, BM, GATED, True)
    tl.store(DK + base + offs_n[:, None] * stride_t + offs_d[None, :],
             (dk * scale).to(DK.dtype.element_ty), mask=kv_mask)
    tl.store(DV + base + offs_n[:, None] * stride_t + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=kv_mask)
    if GATED:
        tl.store(DLK + bh * T + offs_n, dlk, mask=offs_n < T)


@triton.jit
def _dq_block(dq, q, do, lq, rl, delta, K, V, L, base, lbase, offs_m, start, T, scale, stride_t,
              stride_lt, D: tl.constexpr, BN: tl.constexpr, GATED: tl.constexpr,
              MASK: tl.constexpr):
    offs_n = start + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    if MASK:
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < T, other=0.0)
    else:
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :])
    s = tl.dot(q, tl.trans(k)) * scale
    if GATED:
        if MASK:
            lk = tl.load(L + lbase + offs_n * stride_lt, mask=offs_n < T, other=0.0)
        else:
            lk = tl.load(L + lbase + offs_n * stride_lt)
        se = s * tl.exp2(tl.minimum(lq[:, None] - lk[None, :], 0.0))
    else:
        se = s
    if MASK:
        se = tl.where((offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < T), se, 0.0)
    dp = tl.dot(do, tl.trans(v))
    ds = 2.0 * (dp - delta[:, None]) * rl[:, None] * se
    dq += tl.dot(ds.to(k.dtype), k)
    return dq


@triton.autotune(configs=_configs((1, 2)), key=["T", "D", "GATED"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _bwd_dq(Q, K, V, L, DO, LSUM, DELTA, DQ, T, H, scale, stride_b, stride_t, stride_lb,
            stride_lt, D: tl.constexpr, GATED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    base = b * stride_b + h * D
    lbase = b * stride_lb + h
    q_mask = offs_m[:, None] < T
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=q_mask, other=0.0)
    rl = 1.0 / tl.load(LSUM + bh * T + offs_m, mask=offs_m < T, other=1.0)
    delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
    lq = tl.zeros([BM], dtype=tl.float32)
    if GATED:
        lq = tl.load(L + lbase + offs_m * stride_lt, mask=offs_m < T, other=0.0)
    dq = tl.zeros([BM, D], dtype=tl.float32)
    full = (pid_m * BM) // BN * BN
    for start in range(0, full, BN):
        dq = _dq_block(dq, q, do, lq, rl, delta, K, V, L, base, lbase, offs_m, start, T, scale,
                       stride_t, stride_lt, D, BN, GATED, False)
    for start in range(full, tl.minimum((pid_m + 1) * BM, T), BN):
        dq = _dq_block(dq, q, do, lq, rl, delta, K, V, L, base, lbase, offs_m, start, T, scale,
                       stride_t, stride_lt, D, BN, GATED, True)
    tl.store(DQ + base + offs_m[:, None] * stride_t + offs_d[None, :],
             (dq * scale).to(DQ.dtype.element_ty), mask=q_mask)


class _PowerAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, L, scale):
        B, T, H, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        gated = L is not None
        L2 = (L * LOG2E).contiguous() if gated else q.new_zeros(1, 1, dtype=torch.float32)
        o = torch.empty_like(q)
        lsum = torch.empty(B * H, T, device=q.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(T, meta["BM"]), B * H)  # noqa: E731
        _fwd[grid](q, k, v, L2, o, lsum, T, H, scale, q.stride(0), q.stride(1),
                   L2.stride(0), L2.stride(1), D=D, GATED=gated)
        ctx.save_for_backward(q, k, v, L2, o, lsum)
        ctx.scale, ctx.gated = scale, gated
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, L2, o, lsum = ctx.saved_tensors
        B, T, H, D = q.shape
        do = do.contiguous()
        # delta_t = do_t . o_t, laid out [B*H, T] as lsum
        delta = (do.float() * o.float()).sum(-1).permute(0, 2, 1).reshape(B * H, T).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dlk = torch.zeros(B * H, T, device=q.device, dtype=torch.float32)
        args = (T, H, ctx.scale, q.stride(0), q.stride(1), L2.stride(0), L2.stride(1))
        _bwd_dkv[lambda meta: (triton.cdiv(T, meta["BN"]), B * H)](
            q, k, v, L2, do, lsum, delta, dk, dv, dlk, *args, D=D, GATED=ctx.gated)
        _bwd_dq[lambda meta: (triton.cdiv(T, meta["BM"]), B * H)](
            q, k, v, L2, do, lsum, delta, dq, *args, D=D, GATED=ctx.gated)
        dL = dlk.view(B, H, T).permute(0, 2, 1) if ctx.gated else None
        return dq, dk, dv, dL, None


def power_attention(q, k, v, log_g=None, scale=None):
    """Degree-2 power attention, [B, T, H, D] -> [B, T, H, D] (see the module docstring);
    ``log_g`` [B, T, H] (<= 0) is the per-token log gate (cumulated here)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    L = log_g.float().cumsum(1) if log_g is not None else None
    return _PowerAttention.apply(q, k, v, L, scale)
