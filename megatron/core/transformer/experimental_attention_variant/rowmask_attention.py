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
def _fwd_kernel(Q, K, V, O, LSE, IDX, BITS, CNT, QPOS, KPOS, QLO, scale,
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
    qlo = tl.load(QLO + rows, mask=rmask, other=0)
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
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
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
def _dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, IDX, BITS, CNT, QPOS, KPOS, QLO, scale,
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
    qlo = tl.load(QLO + rows, mask=rmask, other=0)
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
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
        pij = tl.where(ok, tl.exp2(s - lse2[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(vb))
        ds = pij * (dp - delta[:, None])
        dq += tl.dot(ds.to(kb.dtype), kb)
    tl.store(DQ + h * sqh + rows[:, None] * sqs + dd[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=rmask[:, None])


@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'PER_GROUP'])
@triton.jit
def _dkv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, CSR_OFF, CSR_R, CSR_B, QPOS, KPOS, QLO, scale,
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
        qlo = tl.load(QLO + rows, mask=rmask, other=0)
        rowbit = ((b >> shift) & 1) != 0
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
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


# ------------------------------------------------------------------------- head-packed tiles (per KV group)
# A tile of TQ queries x the REP query heads of one KV group (REPP = REP padded to a power of 2): M = TQ * REPP rows
# share the group's selection, so every key block is loaded once for all heads, and the tile's page union is that of
# TQ queries instead of 64 -- pure per-token selection (every query its own pages) has a much smaller union per 8
# queries than per 64. Row m = query m // REPP, head m % REPP; a bit word per (tile, key block) has TQ bits.
@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'REPP', 'TQ'])
@triton.jit
def _fwd_hp_kernel(Q, K, V, O, LSE, IDX, BITS, CNT, QPOS, KPOS, QLO, scale,
                   Sq, Sk, NT, NB,
                   sqh, sqs, skh, sks, soh, sos,
                   REP: tl.constexpr, REPP: tl.constexpr, TQ: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    hk = tl.program_id(1)
    m = tl.arange(0, TQ * REPP)
    qi = m // REPP
    hh = m % REPP
    rows = t * TQ + qi
    h = hk * REP + hh
    rin = rows < Sq
    rmask = rin & (hh < REP)
    dd = tl.arange(0, D)
    q = tl.load(Q + h[:, None] * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
    qpos = tl.load(QPOS + rows, mask=rin, other=-1)
    qlo = tl.load(QLO + rows, mask=rin, other=0)
    n = tl.load(CNT + hk * NT + t)
    m_i = tl.full([TQ * REPP], float("-inf"), tl.float32)
    l_i = tl.zeros([TQ * REPP], tl.float32)
    acc = tl.zeros([TQ * REPP, D], tl.float32)
    shift = qi.to(tl.int64)
    base = (hk * NT + t) * NB
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
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
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
    tl.store(O + h[:, None] * soh + rows[:, None] * sos + dd[None, :], o.to(O.dtype.element_ty), mask=rmask[:, None])
    tl.store(LSE + h * Sq + rows, lse, mask=rmask)


@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'REPP', 'TQ'])
@triton.jit
def _dq_hp_kernel(Q, K, V, DO, LSE, DELTA, DQ, IDX, BITS, CNT, QPOS, KPOS, QLO, scale,
                  Sq, Sk, NT, NB,
                  sqh, sqs, skh, sks, soh, sos,
                  REP: tl.constexpr, REPP: tl.constexpr, TQ: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    hk = tl.program_id(1)
    m = tl.arange(0, TQ * REPP)
    qi = m // REPP
    hh = m % REPP
    rows = t * TQ + qi
    h = hk * REP + hh
    rin = rows < Sq
    rmask = rin & (hh < REP)
    dd = tl.arange(0, D)
    q = tl.load(Q + h[:, None] * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
    do = tl.load(DO + h[:, None] * soh + rows[:, None] * sos + dd[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + h * Sq + rows, mask=rmask, other=float("-inf"))
    lse2 = tl.where(lse == float("-inf"), float("inf"), lse) * 1.4426950408889634  # rows without keys: p = 0
    delta = tl.load(DELTA + h * Sq + rows, mask=rmask, other=0.0)
    qpos = tl.load(QPOS + rows, mask=rin, other=-1)
    qlo = tl.load(QLO + rows, mask=rin, other=0)
    n = tl.load(CNT + hk * NT + t)
    dq = tl.zeros([TQ * REPP, D], tl.float32)
    shift = qi.to(tl.int64)
    base = (hk * NT + t) * NB
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
        ok = rowbit[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
        pij = tl.where(ok, tl.exp2(s - lse2[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(vb))
        ds = pij * (dp - delta[:, None])
        dq += tl.dot(ds.to(kb.dtype), kb)
    tl.store(DQ + h[:, None] * sqh + rows[:, None] * sqs + dd[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=rmask[:, None])


@triton.autotune(configs=_CONFIGS, key=['D', 'REP', 'REPP', 'TQ'])
@triton.jit
def _dkv_hp_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, CSR_OFF, CSR_R, CSR_B, QPOS, KPOS, QLO, scale,
                   Sq, Sk, NKB,
                   sqh, sqs, skh, sks, soh, sos,
                   REP: tl.constexpr, REPP: tl.constexpr, TQ: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    pb = tl.program_id(0)  # key block
    hk = tl.program_id(1)
    cols = pb * BN + tl.arange(0, BN)
    cmask = cols < Sk
    dd = tl.arange(0, D)
    kb = tl.load(K + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
    vb = tl.load(V + hk * skh + cols[:, None] * sks + dd[None, :], mask=cmask[:, None], other=0.0)
    kpos = tl.load(KPOS + cols, mask=cmask, other=2147483647)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    m = tl.arange(0, TQ * REPP)
    qi = m // REPP
    hh = m % REPP
    h = hk * REP + hh
    shift = qi.to(tl.int64)
    e0 = tl.load(CSR_OFF + hk * (NKB + 1) + pb)
    e1 = tl.load(CSR_OFF + hk * (NKB + 1) + pb + 1)
    for e in range(e0, e1):
        r = tl.load(CSR_R + e)
        b = tl.load(CSR_B + e)
        rows = r * TQ + qi
        rin = rows < Sq
        rmask = rin & (hh < REP)
        qpos = tl.load(QPOS + rows, mask=rin, other=-1)
        qlo = tl.load(QLO + rows, mask=rin, other=0)
        rowbit = ((b >> shift) & 1) != 0
        ok = (rowbit & rmask)[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
        q = tl.load(Q + h[:, None] * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
        do = tl.load(DO + h[:, None] * soh + rows[:, None] * sos + dd[None, :], mask=rmask[:, None], other=0.0)
        lse = tl.load(LSE + h * Sq + rows, mask=rmask, other=float("-inf"))
        lse2 = tl.where(lse == float("-inf"), float("inf"), lse) * 1.4426950408889634
        delta = tl.load(DELTA + h * Sq + rows, mask=rmask, other=0.0)
        s = tl.dot(q, tl.trans(kb)) * (scale * 1.4426950408889634)
        pij = tl.where(ok, tl.exp2(s - lse2[:, None]), 0.0)  # [M rows, BN keys]
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
    def forward(ctx, q, k, v, idx, bits, cnt, q_pos, k_pos, q_lo, scale, per_group, bn):
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        nq = -(-Sq // BM)
        NB = idx.shape[2]
        o = torch.empty_like(q)
        lse = torch.empty(Hq, Sq, device=q.device, dtype=torch.float32)
        _fwd_kernel[(nq, Hq)](q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo, scale, Sq, Sk, nq, NB,
                              q.stride(0), q.stride(1), k.stride(0), k.stride(1), o.stride(0), o.stride(1),
                              REP=Hq // Hk, PER_GROUP=per_group, D=D, BN=bn)
        ctx.save_for_backward(q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo)
        ctx.scale, ctx.per_group, ctx.bn = scale, per_group, bn
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo = ctx.saved_tensors
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1)  # [Hq, Sq]
        nq, NB = -(-Sq // BM), idx.shape[2]
        rep = Hq // Hk
        dq = torch.empty_like(q)
        _dq_kernel[(nq, Hq)](q, k, v, do, lse, delta, dq, idx, bits, cnt, q_pos, k_pos, q_lo, ctx.scale, Sq, Sk, nq, NB,
                             q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0), do.stride(1),
                             REP=rep, PER_GROUP=ctx.per_group, D=D, BN=ctx.bn)
        nkb = -(-Sk // ctx.bn)
        off, rr, bb = _csr(idx, bits, cnt, nkb)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        _dkv_kernel[(nkb, Hk)](q, k, v, do, lse, delta, dk, dv, off, rr, bb, q_pos, k_pos, q_lo, ctx.scale, Sq, Sk, nkb,
                               q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0), do.stride(1),
                               REP=rep, PER_GROUP=ctx.per_group, D=D, BN=ctx.bn)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class _RowMaskHeadPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, idx, bits, cnt, q_pos, k_pos, q_lo, scale, tq, bn):
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        rep = Hq // Hk
        repp = 1 << (rep - 1).bit_length()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        nt, NB = idx.shape[1], idx.shape[2]
        o = torch.empty_like(q)
        lse = torch.empty(Hq, Sq, device=q.device, dtype=torch.float32)
        _fwd_hp_kernel[(nt, Hk)](q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo, scale, Sq, Sk, nt, NB,
                                 q.stride(0), q.stride(1), k.stride(0), k.stride(1), o.stride(0), o.stride(1),
                                 REP=rep, REPP=repp, TQ=tq, D=D, BN=bn)
        ctx.save_for_backward(q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo)
        ctx.scale, ctx.tq, ctx.bn, ctx.rep, ctx.repp = scale, tq, bn, rep, repp
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo = ctx.saved_tensors
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1)  # [Hq, Sq]
        nt, NB = idx.shape[1], idx.shape[2]
        kw = dict(REP=ctx.rep, REPP=ctx.repp, TQ=ctx.tq, D=D, BN=ctx.bn)
        dq = torch.empty_like(q)
        _dq_hp_kernel[(nt, Hk)](q, k, v, do, lse, delta, dq, idx, bits, cnt, q_pos, k_pos, q_lo, ctx.scale, Sq, Sk,
                                nt, NB, q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0),
                                do.stride(1), **kw)
        nkb = -(-Sk // ctx.bn)
        off, rr, bb = _csr(idx, bits, cnt, nkb)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        _dkv_hp_kernel[(nkb, Hk)](q, k, v, do, lse, delta, dk, dv, off, rr, bb, q_pos, k_pos, q_lo, ctx.scale, Sq,
                                  Sk, nkb, q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0),
                                  do.stride(1), **kw)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def rowmask_head_packed_attention(q, k, v, idx, bits, cnt, q_pos, k_pos, scale: float, tq: int, bn: int = 64,
                                  q_lo=None):
    """Per-KV-group selection with head-packed tiles: q [Hq, Sq, D], k / v [Hk, Sk, D]; idx / bits / cnt
    [Hk, ceil(Sq / tq), NB] from :func:`lists_from_allowed` (allowed [Hk, Sq, P], ``tq``): a tile of ``tq`` queries
    times the group's query heads (padded to a power of 2; tq x heads <= 128 keeps the tiles in registers). Same
    result as :func:`rowmask_block_attention` with per_group."""
    if q_lo is None:
        q_lo = torch.zeros_like(q_pos)
    if tq > 64 or tq & (tq - 1):
        raise ValueError("tq: a power of 2 up to 64")
    return _RowMaskHeadPacked.apply(q, k, v, idx, bits, cnt, q_pos, k_pos, q_lo, scale, tq, bn)


def rowmask_block_attention(q, k, v, idx, bits, cnt, q_pos, k_pos, scale: float, per_group: bool, bn: int = 64,
                            q_lo=None):
    """See the module docstring. idx / bits / cnt: from :func:`lists_from_allowed`. ``q_lo`` [Sq] int32: each
    query's lowest key position (its document's start, for documents packed without padding); default 0."""
    if q_lo is None:
        q_lo = torch.zeros_like(q_pos)
    return _RowMaskAttention.apply(q, k, v, idx, bits, cnt, q_pos, k_pos, q_lo, scale, per_group, bn)


def dense_docs_attention(q, k, v, q_pos, q_lo, scale: float):
    """Causal attention within documents packed WITHOUT padding: q [Hq, Sq, D] (query i at key position q_pos[i],
    its document from key q_lo[i]; rows ascending in both), k / v [Hk, Sk, D] -> [Hq, Sq, D]. Every tile attends the
    contiguous key blocks its rows' documents cover (all bits set; the per-element bound q_lo <= k <= q_pos masks the
    rest) -- one call for any number of short documents."""
    Sq, Sk = q.shape[1], k.shape[1]
    nq = -(-Sq // BM)
    pad = nq * BM - Sq
    first = q_lo[::BM].to(torch.long) // 64  # a tile's first row has the lowest bound
    last_pos = F_pad(q_pos, pad).view(nq, BM).amax(1).clamp(min=0).to(torch.long)
    cnt = (last_pos // 64 - first + 1).clamp(min=0)
    NB = max(1, int(cnt.max()))
    idx = (first[:, None] + torch.arange(NB, device=q.device)[None]).clamp(max=-(-Sk // 64) - 1)
    bits = torch.full((1, nq, NB), -1, dtype=torch.int64, device=q.device)
    k_pos = torch.arange(Sk, device=q.device, dtype=torch.int32)
    return rowmask_block_attention(q, k, v, idx[None].to(torch.int32).contiguous(), bits,
                                   cnt[None].to(torch.int32).contiguous(), q_pos, k_pos, scale, False, 64, q_lo=q_lo)


def F_pad(x, pad):
    return torch.nn.functional.pad(x, (0, pad), value=-1)


def lists_from_allowed(allowed: torch.Tensor, chunk: int = 64, tq: int = BM):
    """allowed [Gs, Sq, P] bool (query row -> key block) -> (idx [Gs, nq, NB] int32, bits [Gs, nq, NB] int64,
    cnt [Gs, nq] int32): each tile's blocks some row allows, ascending, with the rows' bit word (bit i = row i).
    Tiles of ``tq`` queries (default BM = 64; fewer for the head-packed kernel). In chunks of ``chunk`` x 64 queries,
    64-row words packed from bytes (8 rows -> one byte, 8 bytes -> the word): an int64 word per (tile, page) of the
    whole batch at once took 8x the mask's memory (OOM at 256k under CP 4, 2026-11-20)."""
    Gs, Sq, P = allowed.shape
    if tq % 8 or tq > 64:
        raise ValueError("lists_from_allowed: tq a multiple of 8 up to 64")
    nq = -(-Sq // tq)
    dev = allowed.device
    per = chunk * BM // tq  # tiles per chunk
    weights = (2 ** torch.arange(8, device=dev, dtype=torch.int16))  # row j of a byte -> bit j
    nby = tq // 8
    parts = []
    for t0 in range(0, nq, per):
        t1 = min(nq, t0 + per)
        c = t1 - t0
        rows = allowed[:, t0 * tq:min(Sq, t1 * tq)]
        rows = torch.nn.functional.pad(rows, (0, 0, 0, c * tq - rows.shape[1]))
        rows = rows.view(Gs, c, nby, 8, P).permute(0, 1, 4, 2, 3)  # [Gs, c, P, bytes, 8 rows]
        byte = (rows.to(torch.int16) * weights).sum(-1).to(torch.uint8)  # [Gs, c, P, bytes]
        if nby == 8:
            words = byte.contiguous().view(torch.int64)[..., 0]  # little-endian: byte j holds rows 8j .. 8j + 7
        else:
            words = byte.to(torch.int64)[..., 0]
            for j in range(1, nby):
                words = words | (byte[..., j].to(torch.int64) << (8 * j))
        blk = words != 0
        cnt = blk.sum(-1)
        nb = max(1, int(cnt.max()))
        # each tile's blocks in ascending order: the slot of block p is the number of listed blocks before it (a
        # scatter instead of a sort over all P blocks per tile); unlisted blocks go to a dropped slot nb
        slot = torch.where(blk, blk.cumsum(-1) - 1, nb)
        idx = torch.zeros(Gs, c, nb + 1, dtype=torch.int64, device=dev)
        idx.scatter_(-1, slot, torch.arange(P, device=dev).expand(Gs, c, P))
        wsel = torch.zeros(Gs, c, nb + 1, dtype=torch.int64, device=dev)
        wsel.scatter_(-1, slot, words)
        parts.append((idx[..., :nb], wsel[..., :nb], cnt))
    NB = max(p[0].shape[-1] for p in parts)
    idx = torch.cat([torch.nn.functional.pad(p[0], (0, NB - p[0].shape[-1])) for p in parts], 1)
    bits = torch.cat([torch.nn.functional.pad(p[1], (0, NB - p[1].shape[-1])) for p in parts], 1)
    cnt = torch.cat([p[2] for p in parts], 1)
    return idx.to(torch.int32).contiguous(), bits.contiguous(), cnt.to(torch.int32).contiguous()
