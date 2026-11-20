# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Block-sparse attention with a per-row bit mask per (query tile, key block) -- the page selectors' kernel for
per-token choices inside a bounded block union (page_sparse block_candidates) as well as block-wise selection.

Each tile of BM = 64 queries attends a list of key blocks (pages of BN keys); for every listed block one 64-bit word
says which of the tile's queries selected it (bit i = row i). FlexAttention expresses such a mask only through a
per-element mask_mod, which per-token choices make the common case (2.4x the block-wise time); here the mask costs one
word per block. Causality inside a block comes from per-token positions (q_pos / k_pos within the span); the lists
only hold the span's own blocks, so packed documents do not mix.

    o = rowmask_block_attention(q [Hq, Sq, D], k / v [Hk, Sk, D], idx [Gs, nq, NB] int32, bits [Gs, nq, NB] int64,
                                cnt [Gs, nq] int32, q_pos [Sq] int32, k_pos [Sk] int32, scale, per_group)

Gs = Hk with per_group (each KV head's query heads select together), else 1. Forward + backward in Triton: the
forward and dq loop over a tile's list; dk / dv loop over the tiles listing a block (a CSR of the lists).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BM = 64  # queries per tile (= bits per mask word)

_CONFIGS = [triton.Config({}, num_warps=w, num_stages=st) for w in (4, 8) for st in (1, 2, 3)]


@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'PER_GROUP'])
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, IDX, BITS, CNT, QPOS, KPOS, scale,
                Sq, Sk, NQ, NB,
                sqh, sqs, skh, sks, soh, sos,
                REP: tl.constexpr, PER_GROUP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // REP
    sg = hk if PER_GROUP else 0
    rows = r * 64 + tl.arange(0, 64)
    rmask = rows < Sq
    dd = tl.arange(0, D)
    q = tl.load(Q + h * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
    qpos = tl.load(QPOS + rows, mask=rmask, other=-1)
    n = tl.load(CNT + sg * NQ + r)
    m_i = tl.full([64], float("-inf"), tl.float32)
    l_i = tl.zeros([64], tl.float32)
    acc = tl.zeros([64, D], tl.float32)
    shift = tl.arange(0, 64).to(tl.int64)
    base = (sg * NQ + r) * NB
    for j in range(n):
        p = tl.load(IDX + base + j)
        b = tl.load(BITS + base + j)
        cols = p * BN + tl.arange(0, BN)
        cmask = cols < Sk
        kb = tl.load(K + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
        vb = tl.load(V + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
        kpos = tl.load(KPOS + cols, mask=cmask, other=2147483647)
        s = tl.dot(q, tl.trans(kb)) * (scale * 1.4426950408889634)  # log2 units: exp2
        rowbit = ((b >> shift) & 1) != 0
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None])
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        pij = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(pij, 1)
        acc = acc * alpha[:, None] + tl.dot(pij.to(vb.dtype), vb)
        m_i = m_new
    has = l_i > 0
    o = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, (m_i + tl.log2(tl.where(has, l_i, 1.0))) * 0.6931471805599453, float("-inf"))
    tl.store(O + h * soh + rows[:, None] * sos + dd[None, :], o.to(O.dtype.element_ty), mask=rmask[:, None])
    tl.store(LSE + h * Sq + rows, lse, mask=rmask)


@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'PER_GROUP'])
@triton.jit
def _dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, IDX, BITS, CNT, QPOS, KPOS, scale,
               Sq, Sk, NQ, NB,
               sqh, sqs, skh, sks, soh, sos,
               REP: tl.constexpr, PER_GROUP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // REP
    sg = hk if PER_GROUP else 0
    rows = r * 64 + tl.arange(0, 64)
    rmask = rows < Sq
    dd = tl.arange(0, D)
    q = tl.load(Q + h * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
    do = tl.load(DO + h * soh + rows[:, None] * sos + dd[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + h * Sq + rows, mask=rmask, other=float("-inf"))
    lse = tl.where(lse == float("-inf"), float("inf"), lse)  # rows without keys: p = 0
    lse2 = lse * 1.4426950408889634
    delta = tl.load(DELTA + h * Sq + rows, mask=rmask, other=0.0)
    qpos = tl.load(QPOS + rows, mask=rmask, other=-1)
    n = tl.load(CNT + sg * NQ + r)
    dq = tl.zeros([64, D], tl.float32)
    shift = tl.arange(0, 64).to(tl.int64)
    base = (sg * NQ + r) * NB
    for j in range(n):
        p = tl.load(IDX + base + j)
        b = tl.load(BITS + base + j)
        cols = p * BN + tl.arange(0, BN)
        cmask = cols < Sk
        kb = tl.load(K + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
        vb = tl.load(V + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
        kpos = tl.load(KPOS + cols, mask=cmask, other=2147483647)
        s = tl.dot(q, tl.trans(kb)) * (scale * 1.4426950408889634)
        rowbit = ((b >> shift) & 1) != 0
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None])
        pij = tl.where(ok, tl.exp2(s - lse2[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(vb))
        ds = pij * (dp - delta[:, None])
        dq += tl.dot(ds.to(kb.dtype), kb)
    tl.store(DQ + h * sqh + rows[:, None] * sqs + dd[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=rmask[:, None])


@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'PER_GROUP'])
@triton.jit
def _dkv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, CSR_OFF, CSR_R, CSR_B, QPOS, KPOS, scale,
                Sq, Sk, NKB,
                sqh, sqs, skh, sks, soh, sos,
                REP: tl.constexpr, PER_GROUP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    pb = tl.program_id(0)  # key block
    hk = tl.program_id(1)
    sg = hk if PER_GROUP else 0
    cols = pb * BN + tl.arange(0, BN)
    cmask = cols < Sk
    dd = tl.arange(0, D)
    kb = tl.load(K + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
    vb = tl.load(V + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
    kpos = tl.load(KPOS + cols, mask=cmask, other=2147483647)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    shift = tl.arange(0, 64).to(tl.int64)
    e0 = tl.load(CSR_OFF + sg * (NKB + 1) + pb)
    e1 = tl.load(CSR_OFF + sg * (NKB + 1) + pb + 1)
    for e in range(e0, e1):
        r = tl.load(CSR_R + e)
        b = tl.load(CSR_B + e)
        rows = r * 64 + tl.arange(0, 64)
        rmask = rows < Sq
        qpos = tl.load(QPOS + rows, mask=rmask, other=-1)
        rowbit = ((b >> shift) & 1) != 0
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None])
        for hh in range(REP):
            h = hk * REP + hh
            q = tl.load(Q + h * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
            do = tl.load(DO + h * soh + rows[:, None] * sos + dd[None, :], mask=rmask[:, None], other=0.0)
            lse = tl.load(LSE + h * Sq + rows, mask=rmask, other=float("-inf"))
            lse2 = tl.where(lse == float("-inf"), float("inf"), lse) * 1.4426950408889634
            delta = tl.load(DELTA + h * Sq + rows, mask=rmask, other=0.0)
            s = tl.dot(q, tl.trans(kb)) * (scale * 1.4426950408889634)
            pij = tl.where(ok, tl.exp2(s - lse2[:, None]), 0.0)  # [64 q, BN k]
            dv += tl.dot(tl.trans(pij.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(vb))
            ds = pij * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + hk * skh + cols[:, None] * sks + dd[None, :], (dk * scale).to(DK.dtype.element_ty),
             mask=cmask[:, None])
    tl.store(DV + hk * skh + cols[:, None] * sks + dd[None, :], dv.to(DV.dtype.element_ty), mask=cmask[:, None])


def _csr(idx, bits, cnt, nkb):
    """The lists transposed: per selection group, for every key block the (tile, bits) entries listing it."""
    Gs, nq, NB = idx.shape
    slot = torch.arange(NB, device=idx.device)
    valid = slot[None, None, :] < cnt[..., None]
    g_of = torch.arange(Gs, device=idx.device)[:, None, None].expand_as(idx)
    r_of = torch.arange(nq, device=idx.device)[None, :, None].expand_as(idx)
    key = (g_of * nkb + idx.long())[valid]
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=Gs * nkb).view(Gs, nkb)
    off = torch.zeros(Gs, nkb + 1, dtype=torch.long, device=idx.device)
    off[:, 1:] = counts.cumsum(1)
    off = off + (counts.sum(1).cumsum(0) - counts.sum(1))[:, None]  # global offsets
    return off.to(torch.int32).contiguous(), r_of[valid][order].to(torch.int32), bits[valid][order].contiguous()


class _RowMaskAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, idx, bits, cnt, q_pos, k_pos, scale, per_group, bn):
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        nq = -(-Sq // BM)
        NB = idx.shape[2]
        o = torch.empty_like(q)
        lse = torch.empty(Hq, Sq, device=q.device, dtype=torch.float32)
        _fwd_kernel[(nq, Hq)](q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, scale, Sq, Sk, nq, NB,
                              q.stride(0), q.stride(1), k.stride(0), k.stride(1), o.stride(0), o.stride(1),
                              REP=Hq // Hk, PER_GROUP=per_group, D=D, BN=bn)
        ctx.save_for_backward(q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos)
        ctx.scale, ctx.per_group, ctx.bn = scale, per_group, bn
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos = ctx.saved_tensors
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1)  # [Hq, Sq]
        nq, NB = -(-Sq // BM), idx.shape[2]
        rep = Hq // Hk
        dq = torch.empty_like(q)
        _dq_kernel[(nq, Hq)](q, k, v, do, lse, delta, dq, idx, bits, cnt, q_pos, k_pos, ctx.scale, Sq, Sk, nq, NB,
                             q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0), do.stride(1),
                             REP=rep, PER_GROUP=ctx.per_group, D=D, BN=ctx.bn)
        nkb = -(-Sk // ctx.bn)
        off, rr, bb = _csr(idx, bits, cnt, nkb)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        _dkv_kernel[(nkb, Hk)](q, k, v, do, lse, delta, dk, dv, off, rr, bb, q_pos, k_pos, ctx.scale, Sq, Sk, nkb,
                               q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0), do.stride(1),
                               REP=rep, PER_GROUP=ctx.per_group, D=D, BN=ctx.bn)
        return dq, dk, dv, None, None, None, None, None, None, None, None


def rowmask_block_attention(q, k, v, idx, bits, cnt, q_pos, k_pos, scale: float, per_group: bool, bn: int = 64):
    """See the module docstring. idx / bits / cnt: from :func:`lists_from_allowed`."""
    return _RowMaskAttention.apply(q, k, v, idx, bits, cnt, q_pos, k_pos, scale, per_group, bn)


def lists_from_allowed(allowed: torch.Tensor, chunk: int = 64):
    """allowed [Gs, Sq, P] bool (query row -> key block) -> (idx [Gs, nq, NB] int32, bits [Gs, nq, NB] int64,
    cnt [Gs, nq] int32): each tile's blocks some row allows, ascending, with the rows' bit word (bit i = row i).
    In chunks of ``chunk`` tiles, bytes packed directly (8 rows -> one byte, 8 bytes -> the word): an int64 word per
    (tile, page) of the whole batch at once took 8x the mask's memory (OOM at 256k under CP 4, 2026-11-20)."""
    Gs, Sq, P = allowed.shape
    nq = -(-Sq // BM)
    dev = allowed.device
    weights = (2 ** torch.arange(8, device=dev, dtype=torch.int16))  # row j of a byte -> bit j
    parts = []
    for t0 in range(0, nq, chunk):
        t1 = min(nq, t0 + chunk)
        rows = allowed[:, t0 * BM:min(Sq, t1 * BM)]
        rows = torch.nn.functional.pad(rows, (0, 0, 0, (t1 - t0) * BM - rows.shape[1]))
        rows = rows.view(Gs, t1 - t0, BM // 8, 8, P).permute(0, 1, 4, 2, 3)  # [Gs, c, P, 8 bytes, 8 rows]
        byte = (rows.to(torch.int16) * weights).sum(-1).to(torch.uint8)  # [Gs, c, P, 8]
        words = byte.contiguous().view(torch.int64)[..., 0]  # little-endian: byte j holds rows 8j .. 8j + 7
        blk = words != 0
        cnt = blk.sum(-1)
        nb = max(1, int(cnt.max()))
        idx = torch.argsort((~blk).to(torch.int8), dim=-1, stable=True)[..., :nb]
        parts.append((idx, torch.gather(words, -1, idx), cnt))
    NB = max(p[0].shape[-1] for p in parts)
    idx = torch.cat([torch.nn.functional.pad(p[0], (0, NB - p[0].shape[-1])) for p in parts], 1)
    bits = torch.cat([torch.nn.functional.pad(p[1], (0, NB - p[1].shape[-1])) for p in parts], 1)
    cnt = torch.cat([p[2] for p in parts], 1)
    return idx.to(torch.int32).contiguous(), bits.contiguous(), cnt.to(torch.int32).contiguous()
