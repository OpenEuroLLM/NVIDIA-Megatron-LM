# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Product-key top-k (megatron/core/ssm/product_key_memory.py) as one Triton kernel each way.

From the half-key scores ``s1``, ``s2`` ``[T, H, N]`` (fp32): the top ``K`` of each row, the
``K x K`` candidate sums of those, and the top ``K`` of the candidates -> ``scores
[T, H, K]`` and table ``rows = i1 * N + i2``. In torch that is three radix-select top-k
launches over ``[T, H, N]`` / ``[T, H, K*K]`` plus gathers; here one program per (token,
head) reads both rows once. Selection uses ``tl.topk`` on 64-bit keys that pack an
order-preserving encoding of the fp32 value with the (inverted) position, so ties go to the
lower position. The backward writes the dense ``ds1`` / ``ds2`` rows (the scores are sums of
one entry of each), no scatter or zero fill.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = hasattr(tl, "topk")
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _pack(v, pos, P: tl.constexpr):
        """fp32 value + position in [0, P) -> int64, ordered as the value (ties: lower pos)."""
        b = v.to(tl.int32, bitcast=True)
        key = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64)
        return (key << 32) | (P - 1 - pos).to(tl.int64)

    @triton.jit
    def _unpack(packed, P: tl.constexpr):
        b = (packed >> 32).to(tl.int32)
        v = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)
        pos = P - 1 - (packed & 0xFFFFFFFF).to(tl.int32)
        return v, pos

    @triton.jit
    def _pk_topk_fwd(s1_ptr, s2_ptr, scores_ptr, rows_ptr, sel1_ptr, sel2_ptr,
                     N: tl.constexpr, K: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)  # (token, head)
        n = tl.arange(0, N)
        kk = tl.arange(0, K)
        v1, i1 = _unpack(tl.topk(_pack(tl.load(s1_ptr + r * N + n), n, N), K), N)
        v2, i2 = _unpack(tl.topk(_pack(tl.load(s2_ptr + r * N + n), n, N), K), N)
        # candidates: a (index into v1) * K + b (index into v2)
        cand = tl.reshape(v1[:, None] + v2[None, :], (K * K,))
        vs, flat = _unpack(tl.topk(_pack(cand, tl.arange(0, K * K), K * K), K), K * K)
        a, bb = flat // K, flat % K
        # positions of the chosen a / b in the half-key rows (one-hot over the K picks)
        p1 = tl.sum(tl.where(a[:, None] == kk[None, :], i1[None, :], 0), axis=1)
        p2 = tl.sum(tl.where(bb[:, None] == kk[None, :], i2[None, :], 0), axis=1)
        tl.store(scores_ptr + r * K + kk, vs)
        tl.store(rows_ptr + r * K + kk, p1.to(tl.int64) * N + p2)
        tl.store(sel1_ptr + r * K + kk, p1)
        tl.store(sel2_ptr + r * K + kk, p2)

    @triton.jit
    def _pk_topk_bwd(dscores_ptr, sel1_ptr, sel2_ptr, ds1_ptr, ds2_ptr,
                     N: tl.constexpr, K: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        n = tl.arange(0, N)
        kk = tl.arange(0, K)
        g = tl.load(dscores_ptr + r * K + kk)
        p1 = tl.load(sel1_ptr + r * K + kk)
        p2 = tl.load(sel2_ptr + r * K + kk)
        tl.store(ds1_ptr + r * N + n, tl.sum(tl.where(p1[:, None] == n[None, :], g[:, None], 0.0), axis=0))
        tl.store(ds2_ptr + r * N + n, tl.sum(tl.where(p2[:, None] == n[None, :], g[:, None], 0.0), axis=0))


class _ProductKeyTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s1, s2, k):
        T, H, N = s1.shape
        s1, s2 = s1.float().contiguous(), s2.float().contiguous()
        scores = torch.empty((T, H, k), dtype=torch.float32, device=s1.device)
        rows = torch.empty((T, H, k), dtype=torch.int64, device=s1.device)
        sel1 = torch.empty((T, H, k), dtype=torch.int32, device=s1.device)
        sel2 = torch.empty_like(sel1)
        _pk_topk_fwd[(T * H,)](s1, s2, scores, rows, sel1, sel2, N=N, K=k, num_warps=1)
        ctx.save_for_backward(sel1, sel2)
        ctx.N = N
        ctx.mark_non_differentiable(rows)
        return scores, rows

    @staticmethod
    def backward(ctx, dscores, _drows):
        sel1, sel2 = ctx.saved_tensors
        T, H, k = sel1.shape
        ds1 = torch.empty((T, H, ctx.N), dtype=torch.float32, device=sel1.device)
        ds2 = torch.empty_like(ds1)
        _pk_topk_bwd[(T * H,)](dscores.float().contiguous(), sel1, sel2, ds1, ds2, N=ctx.N, K=k,
                               num_warps=1)
        return ds1, ds2, None


def _pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def fused_product_key_topk_applies(s1: torch.Tensor, k: int) -> bool:
    """Triton with tl.topk, CUDA, power-of-two key count and top-k (tl.arange)."""
    return HAVE_TRITON and s1.is_cuda and _pow2(s1.shape[-1]) and _pow2(k) and k <= s1.shape[-1]


def fused_product_key_topk(s1: torch.Tensor, s2: torch.Tensor, k: int):
    """``(scores [T, H, k] fp32, rows [T, H, k] int64)`` from half-key scores ``[T, H, N]``."""
    return _ProductKeyTopK.apply(s1, s2, k)
