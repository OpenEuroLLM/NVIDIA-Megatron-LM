# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""ShortConv's gated depthwise causal convolution as one Triton kernel each way.

ShortConv (megatron/core/ssm/short_conv.py) computes, from the input projection
``bcx`` [s, b, 3 * ch] = [B | C | x]::

    y[t] = C[t] * (bias + sum_{j < K} w[K - 1 - j] * (B * x)[t - j])

(a depthwise causal convolution of the gated stream B * x, zero history before t = 0).
Unfused that is chunk, product, permute + pad, cuDNN depthwise conv1d, permute back and a
second product -- about a dozen memory passes with the backward. Here the forward reads
bcx once (plus a K - 1 row halo) and writes y; the backward reads bcx and dy and writes
dbcx, plus per-tile partial sums of the conv weight / bias gradients (reduced in torch).
fp32 math. Unpacked sequences only (no cu_seqlens), any K.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def short_conv_reference(bcx, weight, bias=None):
    """Torch semantics: y [s, b, ch] from bcx [s, b, 3ch], weight [ch, 1, K] / [ch, K]."""
    b_, c_, x_ = bcx.float().chunk(3, dim=-1)
    bx = b_ * x_
    w = weight.reshape(weight.shape[0], -1).float()
    k = w.shape[-1]
    out = torch.nn.functional.conv1d(
        torch.nn.functional.pad(bx.permute(1, 2, 0), (k - 1, 0)),
        w.unsqueeze(1),
        None if bias is None else bias.float(),
        groups=w.shape[0],
    ).permute(2, 0, 1)
    return (c_ * out).to(bcx.dtype)


if HAVE_TRITON:
    _CONFIGS = [
        triton.Config({"BT": bt, "BC": bc}, num_warps=w)
        for bt, bc, w in [
            (16, 128, 4),
            (32, 64, 4),
            (32, 128, 4),
            (64, 64, 4),
            (16, 256, 8),
        ]
    ]

    @triton.autotune(configs=_CONFIGS, key=["S", "NB", "CH", "K"])
    @triton.jit
    def _fwd(
        bcx,
        w,
        bias,
        y,
        S,
        NB,
        CH,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
    ):
        pt, b, pc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        t = pt * BT + tl.arange(0, BT)
        c = pc * BC + tl.arange(0, BC)
        cm = c < CH
        row = 3 * CH * NB  # stride of one time step in bcx
        acc = tl.zeros([BT, BC], dtype=tl.float32)
        for j in tl.static_range(K):
            tj = t - j
            m = ((tj >= 0) & (tj < S))[:, None] & cm[None, :]
            off = tj.to(tl.int64)[:, None] * row + b * 3 * CH + c[None, :]
            bv = tl.load(bcx + off, mask=m, other=0.0).to(tl.float32)
            xv = tl.load(bcx + off + 2 * CH, mask=m, other=0.0).to(tl.float32)
            wj = tl.load(w + c * K + (K - 1 - j), mask=cm, other=0.0).to(tl.float32)
            acc += bv * xv * wj[None, :]
        if HAS_BIAS:
            acc += tl.load(bias + c, mask=cm, other=0.0).to(tl.float32)[None, :]
        m = (t < S)[:, None] & cm[None, :]
        off = t.to(tl.int64)[:, None] * row + b * 3 * CH + c[None, :]
        cv = tl.load(bcx + off + CH, mask=m, other=0.0).to(tl.float32)
        yoff = (t.to(tl.int64)[:, None] * NB + b) * CH + c[None, :]
        tl.store(y + yoff, (cv * acc).to(y.dtype.element_ty), mask=m)

    @triton.autotune(
        configs=_CONFIGS,
        key=["S", "NB", "CH", "K"],
        reset_to_zero=["dw_part", "db_part"],
    )
    @triton.jit
    def _bwd(
        bcx,
        w,
        bias,
        dy,
        dbcx,
        dw_part,
        db_part,
        S,
        NB,
        CH,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
    ):
        pt, b, pc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        t = pt * BT + tl.arange(0, BT)
        c = pc * BC + tl.arange(0, BC)
        cm = c < CH
        row = 3 * CH * NB
        m = (t < S)[:, None] & cm[None, :]
        off = t.to(tl.int64)[:, None] * row + b * 3 * CH + c[None, :]
        yoff = (t.to(tl.int64)[:, None] * NB + b) * CH + c[None, :]
        g = tl.load(dy + yoff, mask=m, other=0.0).to(tl.float32)
        bv = tl.load(bcx + off, mask=m, other=0.0).to(tl.float32)
        cv = tl.load(bcx + off + CH, mask=m, other=0.0).to(tl.float32)
        xv = tl.load(bcx + off + 2 * CH, mask=m, other=0.0).to(tl.float32)
        # conv[t] (for dC) and the weight gradient: both need bx[t - j]
        conv = tl.zeros([BT, BC], dtype=tl.float32)
        gc = g * cv  # d conv[t]
        for j in tl.static_range(K):
            tj = t - j
            mj = ((tj >= 0) & (tj < S))[:, None] & cm[None, :]
            offj = tj.to(tl.int64)[:, None] * row + b * 3 * CH + c[None, :]
            bx = tl.load(bcx + offj, mask=mj, other=0.0).to(tl.float32) * tl.load(
                bcx + offj + 2 * CH, mask=mj, other=0.0
            ).to(tl.float32)
            wj = tl.load(w + c * K + (K - 1 - j), mask=cm, other=0.0).to(tl.float32)
            conv += bx * wj[None, :]
            # dw[K-1-j] = sum_t gc[t] * bx[t - j]
            part = ((pt * NB + b) * CH + c) * K + (K - 1 - j)
            tl.store(dw_part + part, tl.sum(gc * bx, axis=0), mask=cm)
        if HAS_BIAS:
            conv += tl.load(bias + c, mask=cm, other=0.0).to(tl.float32)[None, :]
            tl.store(db_part + (pt * NB + b) * CH + c, tl.sum(gc, axis=0), mask=cm)
        # d(bx)[t] = sum_j w[K-1-j] * gc[t + j]: gc at later times, from dy and C
        dbx = tl.zeros([BT, BC], dtype=tl.float32)
        for j in tl.static_range(K):
            tj = t + j
            mj = (tj < S)[:, None] & cm[None, :]
            offj = tj.to(tl.int64)[:, None] * row + b * 3 * CH + c[None, :]
            yoffj = (tj.to(tl.int64)[:, None] * NB + b) * CH + c[None, :]
            gcj = tl.load(dy + yoffj, mask=mj, other=0.0).to(tl.float32) * tl.load(
                bcx + offj + CH, mask=mj, other=0.0
            ).to(tl.float32)
            wj = tl.load(w + c * K + (K - 1 - j), mask=cm, other=0.0).to(tl.float32)
            dbx += gcj * wj[None, :]
        ty = dbcx.dtype.element_ty
        tl.store(dbcx + off, (dbx * xv).to(ty), mask=m)
        tl.store(dbcx + off + CH, (g * conv).to(ty), mask=m)
        tl.store(dbcx + off + 2 * CH, (dbx * bv).to(ty), mask=m)


_MIN_BT = 16


def _grid(s, nb, ch):
    return lambda meta: (triton.cdiv(s, meta["BT"]), nb, triton.cdiv(ch, meta["BC"]))


class _FusedShortConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bcx, weight, bias):
        bcx = bcx.contiguous()
        s, nb, ch3 = bcx.shape
        ch = ch3 // 3
        w = weight.reshape(ch, -1).contiguous()
        k = w.shape[-1]
        y = bcx.new_empty(s, nb, ch)
        _fwd[_grid(s, nb, ch)](
            bcx,
            w,
            bias if bias is not None else w,
            y,
            s,
            nb,
            ch,
            K=k,
            HAS_BIAS=bias is not None,
        )
        ctx.save_for_backward(bcx, w, bias)
        ctx.wshape = weight.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        bcx, w, bias = ctx.saved_tensors
        s, nb, ch3 = bcx.shape
        ch, k = w.shape
        dbcx = torch.empty_like(bcx)
        n_t = triton.cdiv(s, _MIN_BT)
        dw_part = torch.zeros(n_t * nb, ch, k, device=bcx.device, dtype=torch.float32)
        db_part = torch.zeros(n_t * nb, ch, device=bcx.device, dtype=torch.float32)
        _bwd[_grid(s, nb, ch)](
            bcx,
            w,
            bias if bias is not None else w,
            dy.contiguous(),
            dbcx,
            dw_part,
            db_part,
            s,
            nb,
            ch,
            K=k,
            HAS_BIAS=bias is not None,
        )
        dw = dw_part.sum(0).to(w.dtype).view(ctx.wshape)
        db = db_part.sum(0).to(bias.dtype) if bias is not None else None
        return dbcx, dw, db


def fused_short_conv(bcx, weight, bias=None):
    """C * causal_depthwise_conv(B * x) for bcx [s, b, 3ch] = [B | C | x], weight [ch, 1, K]."""
    return _FusedShortConv.apply(bcx, weight, bias)
