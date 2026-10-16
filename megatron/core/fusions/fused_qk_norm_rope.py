# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Fused q/k RMSNorm + RoPE straight from the fused QKV projection.

The unfused attention path does, per layer and direction: split the QKV projection
output into q / k / v views, RMSNorm q, RMSNorm k (q/k layernorm over the head dim),
RoPE q, RoPE k -- and in the backward, concatenate the three gradients back into the
projection's gradient. That is six to eight memory-bound kernels on [tokens, heads, 64]
tensors, about a tenth of a small model's step. Here one Triton kernel reads the QKV
tensor once and writes contiguous q, k (normed and rotated) and v; the backward reads
the three gradients and writes the QKV gradient directly, plus the norm-weight
gradients (per-block partial sums, reduced in torch).

Layout (Megatron's, per query group): ``mixed`` [s, b, ng, (hq + 2) * d] holds hq query
heads, one key head and one value head of size d; outputs q [s, b, ng * hq, d],
k [s, b, ng, d], v [s, b, ng, d]. RoPE is Megatron's non-interleaved rotate-half on the
first ``rot`` channels (rotary_percent), angles ``freqs`` [s, rot] (the q/k rotary
embedding tensor without its two singleton dims), mscale 1.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def qk_norm_rope_reference(mixed, q_weight, k_weight, freqs, hq, eps):
    """Unfused torch semantics (fp32 math): returns q, k, v as the fused op."""
    s, b, ng, width = mixed.shape
    d = width // (hq + 2)
    rot = freqs.shape[-1]
    x = mixed.view(s, b, ng, hq + 2, d).float()
    q, k, v = x[..., :hq, :], x[..., hq, :], x[..., hq + 1, :]

    def norm(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    def rope(t):
        f = freqs.float().view(s, 1, *([1] * (t.dim() - 3)), rot)
        cos, sin = f.cos(), f.sin()
        r, rest = t[..., :rot], t[..., rot:]
        half = rot // 2
        rotated = torch.cat([-r[..., half:], r[..., :half]], dim=-1)
        return torch.cat([r * cos + rotated * sin, rest], dim=-1)

    qo = rope(norm(q, q_weight)).reshape(s, b, ng * hq, d)
    ko = rope(norm(k, k_weight))
    return qo.to(mixed.dtype), ko.to(mixed.dtype), v.to(mixed.dtype).contiguous()


if HAVE_TRITON:

    @triton.jit
    def _rope_partner(cols, ROT: tl.constexpr, HALF: tl.constexpr):
        # rotate_half partner channel and sign: out[c] = y[c] cos + sign * y[p] sin
        partner = tl.where(cols < HALF, cols + HALF, cols - HALF)
        sign = tl.where(cols < HALF, -1.0, 1.0)
        rotated = cols < ROT
        return tl.where(rotated, partner, cols), sign, rotated

    @triton.jit
    def _fwd_kernel(
        mixed,
        q_out,
        k_out,
        v_out,
        qw,
        kw,
        freqs,
        rstd_out,
        T,
        B,
        eps,
        NG: tl.constexpr,
        HQ: tl.constexpr,
        D: tl.constexpr,
        ROT: tl.constexpr,
        TB: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        g = tl.program_id(1)
        rows = pid_t * TB + tl.arange(0, TB)  # token index t = s * B + b
        rmask = rows < T
        cols = tl.arange(0, D)
        HALF: tl.constexpr = ROT // 2
        partner, sign, rotated = _rope_partner(cols, ROT, HALF)
        m2 = rmask[:, None]
        s_idx = rows // B
        fcols = tl.where(rotated, cols, 0)
        ang = tl.load(
            freqs + s_idx[:, None] * ROT + fcols[None, :],
            mask=m2 & rotated[None, :],
            other=0.0,
        )
        cos = tl.cos(ang)
        sin = tl.sin(ang)
        base = (rows[:, None] * NG + g) * ((HQ + 2) * D)
        for j in tl.static_range(HQ + 2):
            x = tl.load(mixed + base + j * D + cols[None, :], mask=m2, other=0.0).to(
                tl.float32
            )
            if j == HQ + 1:  # value: plain copy
                tl.store(
                    v_out + (rows[:, None] * NG + g) * D + cols[None, :],
                    x.to(v_out.dtype.element_ty),
                    mask=m2,
                )
            else:
                w_ptr = qw if j < HQ else kw
                w = tl.load(w_ptr + cols).to(tl.float32)
                wp = tl.load(w_ptr + partner).to(tl.float32)
                xp = tl.load(
                    mixed + base + j * D + partner[None, :], mask=m2, other=0.0
                ).to(tl.float32)
                rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
                y = x * rstd[:, None] * w[None, :]
                yp = xp * rstd[:, None] * wp[None, :]
                out = tl.where(rotated[None, :], y * cos + sign[None, :] * yp * sin, y)
                if j < HQ:
                    dst = (
                        q_out + ((rows[:, None] * NG + g) * HQ + j) * D + cols[None, :]
                    )
                    tl.store(dst, out.to(q_out.dtype.element_ty), mask=m2)
                else:
                    dst = k_out + (rows[:, None] * NG + g) * D + cols[None, :]
                    tl.store(dst, out.to(k_out.dtype.element_ty), mask=m2)
                tl.store(rstd_out + (rows * NG + g) * (HQ + 1) + j, rstd, mask=rmask)

    @triton.jit
    def _bwd_kernel(
        mixed,
        dq,
        dk,
        dv,
        dmixed,
        qw,
        kw,
        freqs,
        rstd_in,
        dqw_part,
        dkw_part,
        T,
        B,
        NG: tl.constexpr,
        HQ: tl.constexpr,
        D: tl.constexpr,
        ROT: tl.constexpr,
        TB: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        g = tl.program_id(1)
        rows = pid_t * TB + tl.arange(0, TB)
        rmask = rows < T
        cols = tl.arange(0, D)
        HALF: tl.constexpr = ROT // 2
        partner, sign, rotated = _rope_partner(cols, ROT, HALF)
        m2 = rmask[:, None]
        s_idx = rows // B
        fcols = tl.where(rotated, cols, 0)
        pcols = tl.where(rotated, partner, 0)
        ang = tl.load(
            freqs + s_idx[:, None] * ROT + fcols[None, :],
            mask=m2 & rotated[None, :],
            other=0.0,
        )
        ang_p = tl.load(
            freqs + s_idx[:, None] * ROT + pcols[None, :],
            mask=m2 & rotated[None, :],
            other=0.0,
        )
        cos = tl.cos(ang)
        sin_p = tl.sin(ang_p)
        # d/dy of (y cos + sign_c y[p(c)] sin_c): g[c] = do[c] cos[c] - sign[c] do[p(c)] sin[p(c)]
        base = (rows[:, None] * NG + g) * ((HQ + 2) * D)
        dqw_acc = tl.zeros([D], dtype=tl.float32)
        dkw_acc = tl.zeros([D], dtype=tl.float32)
        for j in tl.static_range(HQ + 2):
            if j == HQ + 1:
                gv = tl.load(
                    dv + (rows[:, None] * NG + g) * D + cols[None, :],
                    mask=m2,
                    other=0.0,
                )
                tl.store(
                    dmixed + base + j * D + cols[None, :],
                    gv.to(dmixed.dtype.element_ty),
                    mask=m2,
                )
            else:
                if j < HQ:
                    gptr = dq + ((rows[:, None] * NG + g) * HQ + j) * D
                    w = tl.load(qw + cols).to(tl.float32)
                else:
                    gptr = dk + (rows[:, None] * NG + g) * D
                    w = tl.load(kw + cols).to(tl.float32)
                do = tl.load(gptr + cols[None, :], mask=m2, other=0.0).to(tl.float32)
                do_p = tl.load(gptr + partner[None, :], mask=m2, other=0.0).to(
                    tl.float32
                )
                gy = tl.where(
                    rotated[None, :], do * cos - sign[None, :] * do_p * sin_p, do
                )
                x = tl.load(
                    mixed + base + j * D + cols[None, :], mask=m2, other=0.0
                ).to(tl.float32)
                rstd = tl.load(
                    rstd_in + (rows * NG + g) * (HQ + 1) + j, mask=rmask, other=0.0
                )
                xhat = x * rstd[:, None]
                if j < HQ:
                    dqw_acc += tl.sum(gy * xhat, axis=0)
                else:
                    dkw_acc += tl.sum(gy * xhat, axis=0)
                gx = gy * w[None, :]
                dx = rstd[:, None] * (
                    gx - xhat * (tl.sum(gx * xhat, axis=1) / D)[:, None]
                )
                tl.store(
                    dmixed + base + j * D + cols[None, :],
                    dx.to(dmixed.dtype.element_ty),
                    mask=m2,
                )
        part = pid_t * NG + g
        tl.store(dqw_part + part * D + cols, dqw_acc)
        tl.store(dkw_part + part * D + cols, dkw_acc)


_TB = 32


class _FusedQKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed, q_weight, k_weight, freqs, hq, eps):
        mixed = mixed.contiguous()
        s, b, ng, width = mixed.shape
        d = width // (hq + 2)
        rot = freqs.shape[-1]
        assert width == (hq + 2) * d and d & (d - 1) == 0, (
            "head dim must be a power of two"
        )
        assert rot % 2 == 0 and rot <= d
        freqs = freqs.reshape(s, rot).float().contiguous()
        t = s * b
        q = mixed.new_empty(s, b, ng * hq, d)
        k = mixed.new_empty(s, b, ng, d)
        v = mixed.new_empty(s, b, ng, d)
        rstd = torch.empty(t, ng, hq + 1, device=mixed.device, dtype=torch.float32)
        grid = (triton.cdiv(t, _TB), ng)
        _fwd_kernel[grid](
            mixed,
            q,
            k,
            v,
            q_weight,
            k_weight,
            freqs,
            rstd,
            t,
            b,
            eps,
            NG=ng,
            HQ=hq,
            D=d,
            ROT=rot,
            TB=_TB,
        )
        ctx.save_for_backward(mixed, q_weight, k_weight, freqs, rstd)
        ctx.hq = hq
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        mixed, q_weight, k_weight, freqs, rstd = ctx.saved_tensors
        s, b, ng, width = mixed.shape
        hq = ctx.hq
        d = width // (hq + 2)
        rot = freqs.shape[-1]
        t = s * b
        dmixed = torch.empty_like(mixed)
        n_tb = triton.cdiv(t, _TB)
        dqw_part = torch.empty(n_tb * ng, d, device=mixed.device, dtype=torch.float32)
        dkw_part = torch.empty_like(dqw_part)
        _bwd_kernel[(n_tb, ng)](
            mixed,
            dq.contiguous(),
            dk.contiguous(),
            dv.contiguous(),
            dmixed,
            q_weight,
            k_weight,
            freqs,
            rstd,
            dqw_part,
            dkw_part,
            t,
            b,
            NG=ng,
            HQ=hq,
            D=d,
            ROT=rot,
            TB=_TB,
        )
        dqw = dqw_part.sum(0).to(q_weight.dtype)
        dkw = dkw_part.sum(0).to(k_weight.dtype)
        return dmixed, dqw, dkw, None, None, None


def fused_qk_norm_rope(mixed, q_weight, k_weight, freqs, hq: int, eps: float):
    """q, k, v from the QKV projection ``mixed`` [s, b, ng, (hq + 2) * d]: q / k RMSNormed
    (weights [d]) and rotated with ``freqs`` [s, ..., rot]; see the module docstring."""
    return _FusedQKNormRope.apply(mixed, q_weight, k_weight, freqs, hq, eps)
