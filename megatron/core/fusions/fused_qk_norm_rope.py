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
embedding tensor without its two singleton dims), mscale 1. ``freqs=None`` is NoPE: q / k are
only normed (the kernels then rotate the first two channels by angle 0, the identity).
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def qk_norm_rope_reference(mixed, q_weight, k_weight, freqs, hq, eps, gate=False):
    """Unfused torch semantics (fp32 math): returns q, k, v (, gate) as the fused op."""
    s, b, ng, width = mixed.shape
    heads = hq * (2 if gate else 1) + 2
    d = width // heads
    rot = freqs.shape[-1] if freqs is not None else 0
    x = mixed.view(s, b, ng, heads, d).float()
    q, k, v = x[..., :hq, :], x[..., heads - 2, :], x[..., heads - 1, :]

    def norm(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    def rope(t):
        if freqs is None:  # NoPE: q / k norm only
            return t
        f = freqs.float().view(s, 1, *([1] * (t.dim() - 3)), rot)
        cos, sin = f.cos(), f.sin()
        r, rest = t[..., :rot], t[..., rot:]
        half = rot // 2
        rotated = torch.cat([-r[..., half:], r[..., :half]], dim=-1)
        return torch.cat([r * cos + rotated * sin, rest], dim=-1)

    qo = rope(norm(q, q_weight)).reshape(s, b, ng * hq, d)
    ko = rope(norm(k, k_weight))
    out = (qo.to(mixed.dtype), ko.to(mixed.dtype), v.to(mixed.dtype).contiguous())
    if gate:
        out += (x[..., hq : 2 * hq, :].reshape(s, b, ng * hq, d).to(mixed.dtype),)
    return out


if HAVE_TRITON:

    @triton.jit
    def _load_parts(
        ptr,
        rows2,
        m2,
        HALF: tl.constexpr,
        ROT: tl.constexpr,
        REST: tl.constexpr,
        REST_P: tl.constexpr,
    ):
        """The rotary halves [0, HALF), [HALF, ROT) and the tail [ROT, ROT + REST) of rows."""
        lo = tl.load(ptr + rows2 + tl.arange(0, HALF)[None, :], mask=m2, other=0.0).to(
            tl.float32
        )
        hi = tl.load(
            ptr + rows2 + HALF + tl.arange(0, HALF)[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        rc = tl.arange(0, REST_P)
        rest = tl.load(
            ptr + rows2 + ROT + rc[None, :], mask=m2 & (rc < REST)[None, :], other=0.0
        ).to(tl.float32)
        return lo, hi, rest

    @triton.jit
    def _store_parts(
        ptr,
        rows2,
        m2,
        lo,
        hi,
        rest,
        HALF: tl.constexpr,
        ROT: tl.constexpr,
        REST: tl.constexpr,
        REST_P: tl.constexpr,
    ):
        ty = ptr.dtype.element_ty
        tl.store(ptr + rows2 + tl.arange(0, HALF)[None, :], lo.to(ty), mask=m2)
        tl.store(ptr + rows2 + HALF + tl.arange(0, HALF)[None, :], hi.to(ty), mask=m2)
        rc = tl.arange(0, REST_P)
        tl.store(
            ptr + rows2 + ROT + rc[None, :], rest.to(ty), mask=m2 & (rc < REST)[None, :]
        )

    _CONFIGS = [
        triton.Config({"TB": tb}, num_warps=w) for tb in (16, 32, 64) for w in (2, 4, 8)
    ]

    @triton.autotune(configs=_CONFIGS, key=["T", "NG", "HQ", "D", "ROT"])
    @triton.jit
    def _fwd_kernel(
        mixed,
        q_out,
        k_out,
        v_out,
        g_out,
        qw,
        kw,
        freqs,
        rstd_out,
        T,
        B,
        eps,
        NG: tl.constexpr,
        HQ: tl.constexpr,
        GATE: tl.constexpr,
        D: tl.constexpr,
        ROT: tl.constexpr,
        REST_P: tl.constexpr,
        TB: tl.constexpr,
    ):
        HALF: tl.constexpr = ROT // 2
        REST: tl.constexpr = D - ROT
        pid_t = tl.program_id(0)
        g = tl.program_id(1)
        rows = pid_t * TB + tl.arange(0, TB)  # token index t = s * B + b
        rmask = rows < T
        m2 = rmask[:, None]
        hc = tl.arange(0, HALF)
        ang = tl.load(
            freqs + (rows // B)[:, None] * HALF + hc[None, :], mask=m2, other=0.0
        )
        cos = tl.cos(ang)
        sin = tl.sin(ang)
        rc = tl.arange(0, REST_P)
        base = ((rows * NG + g) * ((HQ * (1 + GATE) + 2) * D))[:, None]
        for j in tl.static_range(HQ + 2):
            # per group: q heads, [gate heads,] k, v
            if j < HQ:
                xptr = mixed + j * D
            elif j == HQ:
                xptr = mixed + HQ * (1 + GATE) * D
            else:
                xptr = mixed + (HQ * (1 + GATE) + 1) * D
            lo, hi, rest = _load_parts(xptr, base, m2, HALF, ROT, REST, REST_P)
            if j == HQ + 1:  # value: plain copy
                _store_parts(
                    v_out,
                    ((rows * NG + g) * D)[:, None],
                    m2,
                    lo,
                    hi,
                    rest,
                    HALF,
                    ROT,
                    REST,
                    REST_P,
                )
            else:
                w_ptr = qw if j < HQ else kw
                wlo = tl.load(w_ptr + hc).to(tl.float32)
                whi = tl.load(w_ptr + HALF + hc).to(tl.float32)
                wr = tl.load(w_ptr + ROT + rc, mask=rc < REST, other=0.0).to(tl.float32)
                ss = (
                    tl.sum(lo * lo, axis=1)
                    + tl.sum(hi * hi, axis=1)
                    + tl.sum(rest * rest, axis=1)
                )
                rstd = 1.0 / tl.sqrt(ss / D + eps)
                ylo = lo * rstd[:, None] * wlo[None, :]
                yhi = hi * rstd[:, None] * whi[None, :]
                yr = rest * rstd[:, None] * wr[None, :]
                olo = ylo * cos - yhi * sin
                ohi = yhi * cos + ylo * sin
                if j < HQ:
                    dst = (((rows * NG + g) * HQ + j) * D)[:, None]
                    _store_parts(q_out, dst, m2, olo, ohi, yr, HALF, ROT, REST, REST_P)
                else:
                    dst = ((rows * NG + g) * D)[:, None]
                    _store_parts(k_out, dst, m2, olo, ohi, yr, HALF, ROT, REST, REST_P)
                tl.store(rstd_out + (rows * NG + g) * (HQ + 1) + j, rstd, mask=rmask)
        if GATE:  # output-gate heads: plain copy
            for j in tl.static_range(HQ):
                lo, hi, rest = _load_parts(
                    mixed + (HQ + j) * D, base, m2, HALF, ROT, REST, REST_P
                )
                dst = (((rows * NG + g) * HQ + j) * D)[:, None]
                _store_parts(g_out, dst, m2, lo, hi, rest, HALF, ROT, REST, REST_P)

    # the backward's per-block weight partials are sized for the smallest TB and zeroed
    @triton.autotune(
        configs=_CONFIGS,
        key=["T", "NG", "HQ", "D", "ROT"],
        reset_to_zero=["dqw_part", "dkw_part"],
    )
    @triton.jit
    def _bwd_kernel(
        mixed,
        dq,
        dk,
        dv,
        dg,
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
        GATE: tl.constexpr,
        D: tl.constexpr,
        ROT: tl.constexpr,
        REST_P: tl.constexpr,
        TB: tl.constexpr,
    ):
        HALF: tl.constexpr = ROT // 2
        REST: tl.constexpr = D - ROT
        pid_t = tl.program_id(0)
        g = tl.program_id(1)
        rows = pid_t * TB + tl.arange(0, TB)
        rmask = rows < T
        m2 = rmask[:, None]
        hc = tl.arange(0, HALF)
        rc = tl.arange(0, REST_P)
        ang = tl.load(
            freqs + (rows // B)[:, None] * HALF + hc[None, :], mask=m2, other=0.0
        )
        cos = tl.cos(ang)
        sin = tl.sin(ang)
        base = ((rows * NG + g) * ((HQ * (1 + GATE) + 2) * D))[:, None]
        wq_lo = tl.zeros([HALF], dtype=tl.float32)
        wq_hi = tl.zeros([HALF], dtype=tl.float32)
        wq_r = tl.zeros([REST_P], dtype=tl.float32)
        wk_lo = tl.zeros([HALF], dtype=tl.float32)
        wk_hi = tl.zeros([HALF], dtype=tl.float32)
        wk_r = tl.zeros([REST_P], dtype=tl.float32)
        for j in tl.static_range(HQ + 2):
            if j == HQ + 1:
                glo, ghi, gr = _load_parts(
                    dv, ((rows * NG + g) * D)[:, None], m2, HALF, ROT, REST, REST_P
                )
                _store_parts(
                    dmixed + (HQ * (1 + GATE) + 1) * D,
                    base,
                    m2,
                    glo,
                    ghi,
                    gr,
                    HALF,
                    ROT,
                    REST,
                    REST_P,
                )
            else:
                if j < HQ:
                    src = dq
                    off = (((rows * NG + g) * HQ + j) * D)[:, None]
                    w_ptr = qw
                    xoff = j * D
                else:
                    src = dk
                    off = ((rows * NG + g) * D)[:, None]
                    w_ptr = kw
                    xoff = HQ * (1 + GATE) * D
                dlo, dhi, dr = _load_parts(src, off, m2, HALF, ROT, REST, REST_P)
                # transpose of the rotation [[c, -s], [s, c]]
                glo = dlo * cos + dhi * sin
                ghi = dhi * cos - dlo * sin
                lo, hi, rest = _load_parts(
                    mixed + xoff, base, m2, HALF, ROT, REST, REST_P
                )
                rstd = tl.load(
                    rstd_in + (rows * NG + g) * (HQ + 1) + j, mask=rmask, other=0.0
                )
                r2 = rstd[:, None]
                xlo, xhi, xr = lo * r2, hi * r2, rest * r2
                if j < HQ:
                    wq_lo += tl.sum(glo * xlo, axis=0)
                    wq_hi += tl.sum(ghi * xhi, axis=0)
                    wq_r += tl.sum(dr * xr, axis=0)
                else:
                    wk_lo += tl.sum(glo * xlo, axis=0)
                    wk_hi += tl.sum(ghi * xhi, axis=0)
                    wk_r += tl.sum(dr * xr, axis=0)
                wlo = tl.load(w_ptr + hc).to(tl.float32)
                whi = tl.load(w_ptr + HALF + hc).to(tl.float32)
                wr = tl.load(w_ptr + ROT + rc, mask=rc < REST, other=0.0).to(tl.float32)
                gxlo = glo * wlo[None, :]
                gxhi = ghi * whi[None, :]
                gxr = dr * wr[None, :]
                dot = (
                    tl.sum(gxlo * xlo, axis=1)
                    + tl.sum(gxhi * xhi, axis=1)
                    + tl.sum(gxr * xr, axis=1)
                ) / D
                d2 = dot[:, None]
                _store_parts(
                    dmixed + xoff,
                    base,
                    m2,
                    r2 * (gxlo - xlo * d2),
                    r2 * (gxhi - xhi * d2),
                    r2 * (gxr - xr * d2),
                    HALF,
                    ROT,
                    REST,
                    REST_P,
                )
        if GATE:  # output-gate heads: plain copy of their gradient
            for j in tl.static_range(HQ):
                src = (((rows * NG + g) * HQ + j) * D)[:, None]
                glo, ghi, gr = _load_parts(dg, src, m2, HALF, ROT, REST, REST_P)
                _store_parts(
                    dmixed + (HQ + j) * D,
                    base,
                    m2,
                    glo,
                    ghi,
                    gr,
                    HALF,
                    ROT,
                    REST,
                    REST_P,
                )
        part = (pid_t * NG + g) * D
        tl.store(dqw_part + part + hc, wq_lo)
        tl.store(dqw_part + part + HALF + hc, wq_hi)
        tl.store(dqw_part + part + ROT + rc, wq_r, mask=rc < REST)
        tl.store(dkw_part + part + hc, wk_lo)
        tl.store(dkw_part + part + HALF + hc, wk_hi)
        tl.store(dkw_part + part + ROT + rc, wk_r, mask=rc < REST)


_MIN_TB = 16  # smallest autotuned token block


def _rest_p(d, rot):
    """Width of the non-rotary tail block (a power of two >= 1; masked to d - rot)."""
    return max(1, triton.next_power_of_2(d - rot)) if d > rot else 1


class _FusedQKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed, q_weight, k_weight, freqs, hq, eps, gate):
        mixed = mixed.contiguous()
        s, b, ng, width = mixed.shape
        heads = hq * (2 if gate else 1) + 2
        d = width // heads
        rot = freqs.shape[-1]
        assert width == heads * d and d & (d - 1) == 0, (
            "head dim must be a power of two"
        )
        assert rot % 2 == 0 and rot <= d and (rot // 2) & (rot // 2 - 1) == 0, (
            "rot/2: power of 2"
        )
        # Megatron's angles are cat(theta, theta): the kernels read the first half
        freqs = freqs.reshape(s, rot)[:, : rot // 2].float().contiguous()
        t = s * b
        q = mixed.new_empty(s, b, ng * hq, d)
        k = mixed.new_empty(s, b, ng, d)
        v = mixed.new_empty(s, b, ng, d)
        g = mixed.new_empty(s, b, ng * hq, d) if gate else v  # (unused without a gate)
        rstd = torch.empty(t, ng, hq + 1, device=mixed.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(t, meta["TB"]), ng)  # noqa: E731
        _fwd_kernel[grid](
            mixed,
            q,
            k,
            v,
            g,
            q_weight,
            k_weight,
            freqs,
            rstd,
            t,
            b,
            eps,
            NG=ng,
            HQ=hq,
            GATE=int(gate),
            D=d,
            ROT=rot,
            REST_P=_rest_p(d, rot),
        )
        ctx.save_for_backward(mixed, q_weight, k_weight, freqs, rstd)
        ctx.hq, ctx.rot, ctx.gate, ctx.d = hq, rot, gate, d
        return (q, k, v, g) if gate else (q, k, v)

    @staticmethod
    def backward(ctx, dq, dk, dv, dg=None):
        mixed, q_weight, k_weight, freqs, rstd = ctx.saved_tensors
        s, b, ng, _ = mixed.shape
        hq, rot, d = ctx.hq, ctx.rot, ctx.d
        t = s * b
        dmixed = torch.empty_like(mixed)
        n_tb = triton.cdiv(t, _MIN_TB)
        dqw_part = torch.zeros(n_tb * ng, d, device=mixed.device, dtype=torch.float32)
        dkw_part = torch.zeros_like(dqw_part)
        dv = dv.contiguous()
        dg = dg.contiguous() if ctx.gate else dv
        grid = lambda meta: (triton.cdiv(t, meta["TB"]), ng)  # noqa: E731
        _bwd_kernel[grid](
            mixed,
            dq.contiguous(),
            dk.contiguous(),
            dv,
            dg,
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
            GATE=int(ctx.gate),
            D=d,
            ROT=rot,
            REST_P=_rest_p(d, rot),
        )
        dqw = dqw_part.sum(0).to(q_weight.dtype)
        dkw = dkw_part.sum(0).to(k_weight.dtype)
        return dmixed, dqw, dkw, None, None, None, None


def fused_qk_norm_rope(
    mixed, q_weight, k_weight, freqs, hq: int, eps: float, gate: bool = False
):
    """q, k, v (and the output gate with ``gate``) from the QKV projection ``mixed``
    [s, b, ng, (hq [+ hq] + 2) * d]: q / k RMSNormed (weights [d]) and rotated with
    ``freqs`` [s, ..., rot], or only normed with ``freqs=None`` (NoPE); see the module
    docstring."""
    if freqs is None:
        # a rotation by angle 0 is the identity: the smallest rotary block (2 channels)
        freqs = torch.zeros(mixed.shape[0], 2, device=mixed.device, dtype=torch.float32)
    return _FusedQKNormRope.apply(mixed, q_weight, k_weight, freqs, hq, eps, gate)
