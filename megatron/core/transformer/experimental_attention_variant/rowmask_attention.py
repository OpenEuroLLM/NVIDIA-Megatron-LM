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

import os

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
@triton.jit(do_not_specialize=['scale', 'Sq', 'Sk', 'NT', 'NB'])
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


@triton.jit(do_not_specialize=['scale', 'Sq', 'Sk', 'NT', 'NB'])
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


@triton.jit(do_not_specialize=['scale', 'Sq', 'Sk', 'NKB'])
def _dkv_hp_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, CSR_OFF, CSR_R, CSR_B, QPOS, KPOS, QLO, scale,
                   Sq, Sk, NKB,
                   sqh, sqs, skh, sks, soh, sos,
                   REP: tl.constexpr, REPP: tl.constexpr, TQ: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
                   MC: tl.constexpr):
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
    e0 = tl.load(CSR_OFF + hk * (NKB + 1) + pb)
    e1 = tl.load(CSR_OFF + hk * (NKB + 1) + pb + 1)
    for e in range(e0, e1):
        r = tl.load(CSR_R + e)
        b = tl.load(CSR_B + e)
        # the tile's TQ * REPP rows in chunks of MC: [MC, D] operand tiles instead of [TQ * REPP, D] (register
        # spills at head dim 256 grow the context's local memory for good -- non-PyTorch memory, 2026-11-21)
        for c in tl.static_range(TQ * REPP // MC):
            m = c * MC + tl.arange(0, MC)
            qi = m // REPP
            hh = m % REPP
            h = hk * REP + hh
            rows = r * TQ + qi
            rin = rows < Sq
            rmask = rin & (hh < REP)
            qpos = tl.load(QPOS + rows, mask=rin, other=-1)
            qlo = tl.load(QLO + rows, mask=rin, other=0)
            rowbit = ((b >> qi.to(tl.int64)) & 1) != 0
            ok = (rowbit & rmask)[:, None] & (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qlo[:, None])
            q = tl.load(Q + h[:, None] * sqh + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
            do = tl.load(DO + h[:, None] * soh + rows[:, None] * sos + dd[None, :], mask=rmask[:, None], other=0.0)
            lse = tl.load(LSE + h * Sq + rows, mask=rmask, other=float("-inf"))
            lse2 = tl.where(lse == float("-inf"), float("inf"), lse) * 1.4426950408889634
            delta = tl.load(DELTA + h * Sq + rows, mask=rmask, other=0.0)
            s = tl.dot(q, tl.trans(kb)) * (scale * 1.4426950408889634)
            pij = tl.where(ok, tl.exp2(s - lse2[:, None]), 0.0)  # [MC rows, BN keys]
            dv += tl.dot(tl.trans(pij.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(vb))
            ds = pij * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + hk * skh + cols[:, None] * sks + dd[None, :], (dk * scale).to(DK.dtype.element_ty),
             mask=cmask[:, None])
    tl.store(DV + hk * skh + cols[:, None] * sks + dd[None, :], dv.to(DV.dtype.element_ty), mask=cmask[:, None])


# fixed launch configs of the head-packed kernels, (num_warps, num_stages[, MC]) by (kernel, head dim, tile rows):
# no autotune -- autotuning launches every candidate once, and a candidate that spills registers grows the CUDA
# context's local memory for the rest of the process (~+4 GB non-PyTorch memory per GPU at v4b shapes: the 64-node
# per-token run OOMed at step 5, t19g / t19j 2026-11-21), and a rank first meeting a kernel mid-run stalls every
# other rank while it benchmarks. ROWMASK_HP_CONFIG="fwd=8,2;dq=8,1;dkv=8,1,64" overrides (benchmarks).
_HP_CONFIG = {
    "fwd": {"default": (8, 2)},  # H100, d 256, 128 rows: 254 regs, no spills (4 warps: 344 spills, 12 % slower)
    "dq": {"default": (8, 1)},  # 64 spills (4 warps: 496; 2 stages exceed shared memory)
    # MC = the tile's rows (no chunking): at head dim 256 / 128 rows on H100 chunks of 64 / 32 rows were 5x slower
    # (more spills, not fewer); 8 warps: 76 spills, 4 warps: 814 (t20_token_memory, 2026-11-21)
    "dkv": {"default": (8, 1, 128)},
}


def _hp_config(kind: str, D: int, rows: int):
    env = os.environ.get("ROWMASK_HP_CONFIG", "")
    for part in filter(None, env.split(";")):
        name, _, vals = part.partition("=")
        if name.strip() == kind:
            return tuple(int(x) for x in vals.split(","))
    table = _HP_CONFIG[kind]
    return table.get((D, rows), table["default"])


@triton.jit(do_not_specialize=['N'])
def _delta_kernel(DO, O, DELTA, N, D: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < N
    dd = tl.arange(0, D)
    do = tl.load(DO + r[:, None] * D + dd[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    o = tl.load(O + r[:, None] * D + dd[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(DELTA + r, tl.sum(do * o, 1), mask=rm)


def row_delta(do: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """The backward's delta = rowsum(dO * O) [Hq, Sq] fp32 from contiguous [Hq, Sq, D] tensors, without fp32 copies:
    (do.float() * o.float()).sum(-1) made three [Hq, Sq, D] fp32 temporaries -- 3 GB per attention layer at v4b's 64k
    (the 64-node per-token run's OOM, t19g / t19j 2026-11-21)."""
    Hq, Sq, D = o.shape
    if D & (D - 1) or not o.is_cuda:
        return (do.float() * o.float()).sum(-1)
    delta = torch.empty(Hq, Sq, device=o.device, dtype=torch.float32)
    N = Hq * Sq
    BR = max(1, 4096 // D)
    _delta_kernel[(triton.cdiv(N, BR),)](do.contiguous(), o.contiguous(), delta, N, D=D, BR=BR, num_warps=4)
    return delta


def _csr(idx, bits, cnt, nkb):
    """The lists transposed: per selection group, for every key block the (tile, bits) entries listing it."""
    Gs, nq, NB = idx.shape
    slot = torch.arange(NB, device=idx.device)
    valid = slot[None, None, :] < cnt[..., None]
    g_of = torch.arange(Gs, device=idx.device)[:, None, None].expand_as(idx)
    r_of = torch.arange(nq, device=idx.device)[None, :, None].expand_as(idx)
    # static shapes only (boolean indexing and bincount sync the host): unused slots get the key Gs * nkb, sort
    # after every real entry and are never read (the offsets end before them)
    key = torch.where(valid, g_of * nkb + idx.long(), Gs * nkb).reshape(-1)
    order = torch.argsort(key, stable=True)
    counts = torch.zeros(Gs * nkb + 1, dtype=torch.long, device=idx.device)
    counts.scatter_add_(0, key, torch.ones_like(key))
    counts = counts[:-1].view(Gs, nkb)
    off = torch.zeros(Gs, nkb + 1, dtype=torch.long, device=idx.device)
    off[:, 1:] = counts.cumsum(1)
    off = off + (counts.sum(1).cumsum(0) - counts.sum(1))[:, None]  # global offsets
    return (off.to(torch.int32).contiguous(), r_of.reshape(-1)[order].to(torch.int32).contiguous(),
            bits.reshape(-1)[order].contiguous())


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
        delta = row_delta(do, o)  # [Hq, Sq]
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
        w, st = _hp_config("fwd", D, tq * repp)[:2]
        _fwd_hp_kernel[(nt, Hk)](q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo, scale, Sq, Sk, nt, NB,
                                 q.stride(0), q.stride(1), k.stride(0), k.stride(1), o.stride(0), o.stride(1),
                                 REP=rep, REPP=repp, TQ=tq, D=D, BN=bn, num_warps=w, num_stages=st)
        ctx.save_for_backward(q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo)
        ctx.scale, ctx.tq, ctx.bn, ctx.rep, ctx.repp = scale, tq, bn, rep, repp
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, idx, bits, cnt, q_pos, k_pos, q_lo = ctx.saved_tensors
        Hq, Sq, D = q.shape
        Hk, Sk, _ = k.shape
        do = do.contiguous()
        delta = row_delta(do, o)  # [Hq, Sq]
        nt, NB = idx.shape[1], idx.shape[2]
        kw = dict(REP=ctx.rep, REPP=ctx.repp, TQ=ctx.tq, D=D, BN=ctx.bn)
        dq = torch.empty_like(q)
        rows = ctx.tq * ctx.repp
        w, st = _hp_config("dq", D, rows)[:2]
        _dq_hp_kernel[(nt, Hk)](q, k, v, do, lse, delta, dq, idx, bits, cnt, q_pos, k_pos, q_lo, ctx.scale, Sq, Sk,
                                nt, NB, q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0),
                                do.stride(1), **kw, num_warps=w, num_stages=st)
        nkb = -(-Sk // ctx.bn)
        off, rr, bb = _csr(idx, bits, cnt, nkb)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        w, st, mc = _hp_config("dkv", D, rows)
        _dkv_hp_kernel[(nkb, Hk)](q, k, v, do, lse, delta, dk, dv, off, rr, bb, q_pos, k_pos, q_lo, ctx.scale, Sq,
                                  Sk, nkb, q.stride(0), q.stride(1), k.stride(0), k.stride(1), do.stride(0),
                                  do.stride(1), **kw, MC=min(mc, rows), num_warps=w, num_stages=st)
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


def _dense_docs_tiles(q_pos, q_lo):
    """Each 64-query tile's first key block and key-block count (the documents its rows cover)."""
    Sq = q_pos.shape[0]
    nq = -(-Sq // BM)
    first = q_lo[::BM].to(torch.long) // 64  # a tile's first row has the lowest bound
    last_pos = F_pad(q_pos, nq * BM - Sq).view(nq, BM).amax(1).clamp(min=0).to(torch.long)
    return first, (last_pos // 64 - first + 1).clamp(min=0)


def dense_docs_blocks(q_pos, q_lo) -> int:
    """:func:`dense_docs_attention`'s list width (``nb``), from HOST copies of q_pos / q_lo: passing it avoids the
    device-to-host sync of computing it from device tensors."""
    return max(1, int(_dense_docs_tiles(q_pos, q_lo)[1].max()))


_WARMED: set = set()


def warm_dense_docs(hq: int, hk: int, d: int, device, dtype) -> None:
    """Autotune :func:`dense_docs_attention`'s kernels (forward, dq, dk / dv) once per process and head shape. Triton
    autotunes at the first call, and the short-document path runs only when a rank first meets a short document --
    mid-run, one rank benchmarked 6 configs x 3 kernels (~1.4 s) while every other rank waited at the next collective,
    a new rank on almost every step on long-document data (t19i all-rank profile, 2026-11-21). Call it on every rank
    at the first sparse call. No RNG (a recompute must replay the forward's random stream)."""
    key = (str(device), dtype, hq, hk, d)
    if key in _WARMED:
        return
    _WARMED.add(key)
    S = 2 * 1024  # two short documents
    with torch.enable_grad():
        q = torch.zeros(hq, S, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.zeros(hk, S, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.zeros(hk, S, d, device=device, dtype=dtype, requires_grad=True)
        q_pos = torch.arange(S, device=device, dtype=torch.int32)
        q_lo = (q_pos // 1024) * 1024
        dense_docs_attention(q, k, v, q_pos, q_lo, d**-0.5, nb=17).float().sum().backward()


def dense_docs_attention(q, k, v, q_pos, q_lo, scale: float, nb: int | None = None):
    """Causal attention within documents packed WITHOUT padding: q [Hq, Sq, D] (query i at key position q_pos[i],
    its document from key q_lo[i]; rows ascending in both), k / v [Hk, Sk, D] -> [Hq, Sq, D]. Every tile attends the
    contiguous key blocks its rows' documents cover (all bits set; the per-element bound q_lo <= k <= q_pos masks the
    rest) -- one call for any number of short documents. ``nb``: :func:`dense_docs_blocks` (else computed here, which
    syncs the host)."""
    Sq, Sk = q.shape[1], k.shape[1]
    nq = -(-Sq // BM)
    first, cnt = _dense_docs_tiles(q_pos, q_lo)
    NB = max(1, int(cnt.max())) if nb is None else nb
    idx = (first[:, None] + torch.arange(NB, device=q.device)[None]).clamp(max=-(-Sk // 64) - 1)
    bits = torch.full((1, nq, NB), -1, dtype=torch.int64, device=q.device)
    k_pos = torch.arange(Sk, device=q.device, dtype=torch.int32)
    return rowmask_block_attention(q, k, v, idx[None].to(torch.int32).contiguous(), bits,
                                   cnt[None].to(torch.int32).contiguous(), q_pos, k_pos, scale, False, 64, q_lo=q_lo)


def F_pad(x, pad):
    return torch.nn.functional.pad(x, (0, pad), value=-1)


def lists_from_allowed(allowed: torch.Tensor, chunk: int = 64, tq: int = BM, nb: int | None = None):
    """allowed [Gs, Sq, P] bool (query row -> key block) -> (idx [Gs, nq, NB] int32, bits [Gs, nq, NB] int64,
    cnt [Gs, nq] int32): each tile's blocks some row allows, ascending, with the rows' bit word (bit i = row i).
    Tiles of ``tq`` queries (default BM = 64; fewer for the head-packed kernel); ``nb``: a static upper bound of the
    blocks per tile (the list width) -- no host sync; else the widest tile's count. In chunks of ``chunk`` x 64 queries,
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
        nbc = max(1, int(cnt.max())) if nb is None else nb
        # each tile's blocks in ascending order: the slot of block p is the number of listed blocks before it (a
        # scatter instead of a sort over all P blocks per tile); unlisted blocks go to a dropped slot nb
        slot = torch.where(blk, blk.cumsum(-1) - 1, nbc)
        idx = torch.zeros(Gs, c, nbc + 1, dtype=torch.int64, device=dev)
        idx.scatter_(-1, slot, torch.arange(P, device=dev).expand(Gs, c, P))
        wsel = torch.zeros(Gs, c, nbc + 1, dtype=torch.int64, device=dev)
        wsel.scatter_(-1, slot, words)
        parts.append((idx[..., :nbc], wsel[..., :nbc], cnt))
    NB = max(p[0].shape[-1] for p in parts)
    idx = torch.cat([torch.nn.functional.pad(p[0], (0, NB - p[0].shape[-1])) for p in parts], 1)
    bits = torch.cat([torch.nn.functional.pad(p[1], (0, NB - p[1].shape[-1])) for p in parts], 1)
    cnt = torch.cat([p[2] for p in parts], 1)
    return idx.to(torch.int32).contiguous(), bits.contiguous(), cnt.to(torch.int32).contiguous()


# ------------------------------------------------------------------------- per-token page_maxpool scores
@triton.jit
def _maxpool_scores_kernel(QS, WIN, OUT, S, P, NW, q_offset, page, sink, wp,
                           sqs, sws, sos,
                           D: tl.constexpr, BQ: tl.constexpr, BP: tl.constexpr, IEEE: tl.constexpr):
    pq = tl.program_id(0)
    pp = tl.program_id(1)
    rows = pq * BQ + tl.arange(0, BQ)
    rmask = rows < S
    dd = tl.arange(0, D)
    q = tl.load(QS + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
    p = pp * BP + tl.arange(0, BP)
    j = pp * 4 * BP + tl.arange(0, 4 * BP)  # windows 4p .. 4p + 3 of the tile's pages
    jok = j < NW
    w = tl.load(WIN + j[:, None] * sws + dd[None, :], mask=jok[:, None], other=0.0)
    jp = 4 * p - 1  # each page's fifth window (its left neighbour's last): the same dot, the same bits
    jpok = (jp >= 0) & (jp < NW)
    wl = tl.load(WIN + jp[:, None] * sws + dd[None, :], mask=jpok[:, None], other=0.0)
    if IEEE:  # fp32 accuracy from three TF32 products (the FMA "ieee" path was 70x slower)
        sm = tl.dot(q, tl.trans(w), input_precision="tf32x3")
        sl = tl.dot(q, tl.trans(wl), input_precision="tf32x3")
    else:
        sm = tl.dot(q, tl.trans(w), input_precision="tf32")
        sl = tl.dot(q, tl.trans(wl), input_precision="tf32")
    sm = tl.where(jok[None, :], sm, float("-inf"))
    sl = tl.where(jpok[None, :], sl, float("-inf"))
    sc = tl.maximum(tl.max(tl.reshape(sm, (BQ, BP, 4)), axis=2), sl)
    cur = (q_offset + rows) // page
    cand = (p[None, :] >= sink) & (p[None, :] < (cur - wp)[:, None])
    sc = tl.where(cand, sc, float("-inf"))
    tl.store(OUT + rows[:, None] * sos + p[None, :], sc, mask=rmask[:, None] & (p < P)[None, :])


def maxpool_page_scores_triton(qsum: torch.Tensor, win: torch.Tensor, P: int, q_offset: int, page: int, sink: int,
                               wp: int) -> torch.Tensor:
    """page_sparse.maxpool_page_scores for one KV group with the candidate mask applied, fused: qsum [s, d] fp32 (the
    query heads' sum, scaled), win [4 P + 1, d] fp32 -> [s, P] fp32, -inf outside each query's candidates (pages
    sink .. its window's first - 1). One pass instead of the [s, 4 P + 1] window scores, the pooling and the mask
    (most of the per-token selection's memory traffic). TF32 products iff torch allows them for fp32 GEMMs, else
    three TF32 products (fp32 accuracy)."""
    s, d = qsum.shape
    out = torch.empty(s, P, device=qsum.device, dtype=torch.float32)
    BQ, BP = 64, 16
    # torch's fp32 GEMM precision: TF32 when allowed (or forced, as NGC containers do through the cuBLAS override)
    ieee = not (torch.backends.cuda.matmul.allow_tf32 or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1")
    grid = (triton.cdiv(s, BQ), triton.cdiv(P, BP))
    _maxpool_scores_kernel[grid](qsum.contiguous(), win.contiguous(), out, s, P, win.shape[0], q_offset, page, sink,
                                 wp, qsum.stride(0) if qsum.is_contiguous() else d, d, P,
                                 D=d, BQ=BQ, BP=BP, IEEE=ieee, num_warps=4)
    return out


@triton.jit
def _token_lists_kernel(SC, V, IDX, BITS, CNT, S, P, NB, q_offset, page, sink, wp, n,
                        TQ: tl.constexpr, BP: tl.constexpr):
    t = tl.program_id(0)
    rows = t * TQ + tl.arange(0, TQ)
    rmask = rows < S
    v = tl.load(V + rows, mask=rmask, other=float("inf"))
    cur = (q_offset + rows) // page
    ngt = tl.zeros([TQ], tl.int32)
    for p0 in range(0, P, BP):  # each row's count above its n-th largest score
        p = p0 + tl.arange(0, BP)
        s = tl.load(SC + rows[:, None] * P + p[None, :], mask=rmask[:, None] & (p < P)[None, :], other=float("-inf"))
        ngt += tl.sum((s > v[:, None]).to(tl.int32), 1)
    room = n - ngt
    ties = tl.zeros([TQ], tl.int32)
    count = tl.zeros([1], tl.int32)
    shift = tl.arange(0, TQ).to(tl.int64)
    for p0 in range(0, P, BP):
        p = p0 + tl.arange(0, BP)
        pin = p < P
        s = tl.load(SC + rows[:, None] * P + p[None, :], mask=rmask[:, None] & pin[None, :], other=float("-inf"))
        gt = s > v[:, None]
        eq = (s == v[:, None]) & (s > float("-inf"))
        teq = tl.cumsum(eq.to(tl.int32), 1) + ties[:, None]  # the ties of the n-th value: lowest pages first
        sel = gt | (eq & (teq <= room[:, None]))
        ties += tl.sum(eq.to(tl.int32), 1)
        fixed = (p[None, :] < sink) | ((p[None, :] >= (cur - wp)[:, None]) & (p[None, :] <= cur[:, None]))
        allowed = (sel | fixed) & (p[None, :] <= cur[:, None]) & rmask[:, None] & pin[None, :]
        word = tl.sum(allowed.to(tl.int64) << shift[:, None], 0)  # [BP]: bit i = row i
        blk = word != 0
        slot = count + tl.cumsum(blk.to(tl.int32), 0) - 1
        tl.store(IDX + t * NB + slot, p, mask=blk)
        tl.store(BITS + t * NB + slot, word, mask=blk)
        count += tl.sum(blk.to(tl.int32), 0)
    tl.store(CNT + t + tl.arange(0, 1), count)


def token_lists_triton(sc: torch.Tensor, n: int, q_offset: int, page: int, sink: int, wp: int, tq: int, nb: int):
    """One KV group's per-token selection straight into head-packed lists: sc [s, P] fp32 page scores with -inf
    outside the candidates (:func:`maxpool_page_scores_triton`) -> (idx [ceil(s / tq), nb] int32, bits int64 tq-bit
    row words, cnt int32) of each tile's pages, ascending: its rows' top-n pages (page_sparse.topk_mask: ties to the
    lower page), sinks and window pages. Replaces the [s, P] masks and their packing / compaction (the per-token
    selection's remaining passes); ``nb`` a static bound of a tile's pages (no host sync)."""
    s, P = sc.shape
    nt = -(-s // tq)
    n = min(n, P)
    v = sc.topk(n, dim=-1).values[:, -1].contiguous() if n > 0 else torch.full((s,), float("inf"), device=sc.device)
    idx = torch.zeros(nt, nb, dtype=torch.int32, device=sc.device)
    bits = torch.zeros(nt, nb, dtype=torch.int64, device=sc.device)
    cnt = torch.empty(nt, dtype=torch.int32, device=sc.device)
    _token_lists_kernel[(nt,)](sc, v, idx, bits, cnt, s, P, nb, q_offset, page, sink, wp, n, TQ=tq, BP=64,
                               num_warps=4)
    return idx, bits, cnt


# ------------------------------------------------------------------------- per-token selection, packed layout
# Every span of a packed batch at once (the per-span loop's launches made many-document ranks CPU-bound stragglers):
# spans' queries start at multiples of 64 rows and their keys at multiples of the page, so a 64-row tile lies in one
# span; per tile its span's first page (TP0, in the packed layout) and page count (TPN); per row its position in its
# span (QPOS, -1 for padding). Scores are span-local pages [rows, PM] (PM = the longest span's pages, -inf beyond).
@triton.jit(do_not_specialize=['R', 'PM', 'NW'])
def _maxpool_scores_packed_kernel(QS, WIN, OUT, QPOS, TP0, TPN, R, PM, NW, page, sink, wp,
                                  sqg, sqs, swg, sws, sog, sos,
                                  D: tl.constexpr, BQ: tl.constexpr, BP: tl.constexpr, IEEE: tl.constexpr):
    pq = tl.program_id(0)
    pp = tl.program_id(1)
    g = tl.program_id(2)
    rows = pq * BQ + tl.arange(0, BQ)
    rmask = rows < R
    p = pp * BP + tl.arange(0, BP)
    p0 = tl.load(TP0 + pq)
    psp = tl.load(TPN + pq)
    if pp * BP < psp:
        dd = tl.arange(0, D)
        q = tl.load(QS + g * sqg + rows[:, None] * sqs + dd[None, :], mask=rmask[:, None], other=0.0)
        jl = pp * 4 * BP + tl.arange(0, 4 * BP)  # span-local windows 4p .. 4p + 3 of the tile's pages
        jok = jl < 4 * psp
        w = tl.load(WIN + g * swg + (4 * p0 + jl)[:, None] * sws + dd[None, :], mask=jok[:, None], other=0.0)
        jp = 4 * p - 1  # each page's fifth window (its left neighbour's last); page 0's is the span's padding
        jpok = (jp >= 0) & (p < psp)
        wl = tl.load(WIN + g * swg + (4 * p0 + jp)[:, None] * sws + dd[None, :], mask=jpok[:, None], other=0.0)
        if IEEE:
            sm = tl.dot(q, tl.trans(w), input_precision="tf32x3")
            sl = tl.dot(q, tl.trans(wl), input_precision="tf32x3")
        else:
            sm = tl.dot(q, tl.trans(w), input_precision="tf32")
            sl = tl.dot(q, tl.trans(wl), input_precision="tf32")
        sm = tl.where(jok[None, :], sm, float("-inf"))
        sl = tl.where(jpok[None, :], sl, float("-inf"))
        sc = tl.maximum(tl.max(tl.reshape(sm, (BQ, BP, 4)), axis=2), sl)
        qpos = tl.load(QPOS + rows, mask=rmask, other=-1)
        cur = tl.where(qpos >= 0, qpos // page, -1)
        cand = (p[None, :] >= sink) & (p[None, :] < (cur - wp)[:, None]) & (p < psp)[None, :]
        sc = tl.where(cand, sc, float("-inf"))
    else:
        sc = tl.full([BQ, BP], float("-inf"), tl.float32)
    tl.store(OUT + g * sog + rows[:, None] * sos + p[None, :], sc, mask=rmask[:, None] & (p < PM)[None, :])


@triton.jit(do_not_specialize=['R', 'PM', 'NB', 'n'])
def _token_lists_packed_kernel(SC, V, IDX, BITS, CNT, QPOS, TP0, TPN, ANC, R, PM, NB, page, sink, wp, n,
                               scg, svg, sig, scn,
                               TQ: tl.constexpr, BP: tl.constexpr, HAS_ANC: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    rows = t * TQ + tl.arange(0, TQ)
    rmask = rows < R
    tile = (t * TQ) // 64
    p0 = tl.load(TP0 + tile)
    psp = tl.load(TPN + tile)
    if HAS_ANC:  # one selection per block of queries: each row reads its anchor's scores (sinks / window its own)
        srows = tl.load(ANC + rows, mask=rmask, other=0)
    else:
        srows = rows
    v = tl.load(V + g * svg + srows, mask=rmask, other=float("inf"))
    qpos = tl.load(QPOS + rows, mask=rmask, other=-1)
    cur = tl.where(qpos >= 0, qpos // page, -1)
    ngt = tl.zeros([TQ], tl.int32)
    for c0 in range(0, psp, BP):  # each row's count above its n-th largest score
        p = c0 + tl.arange(0, BP)
        s = tl.load(SC + g * scg + srows[:, None] * PM + p[None, :], mask=rmask[:, None] & (p < psp)[None, :],
                    other=float("-inf"))
        ngt += tl.sum((s > v[:, None]).to(tl.int32), 1)
    room = n - ngt
    ties = tl.zeros([TQ], tl.int32)
    count = tl.zeros([1], tl.int32)
    shift = tl.arange(0, TQ).to(tl.int64)
    for c0 in range(0, psp, BP):
        p = c0 + tl.arange(0, BP)
        pin = p < psp
        s = tl.load(SC + g * scg + srows[:, None] * PM + p[None, :], mask=rmask[:, None] & pin[None, :],
                    other=float("-inf"))
        gt = s > v[:, None]
        eq = (s == v[:, None]) & (s > float("-inf"))
        teq = tl.cumsum(eq.to(tl.int32), 1) + ties[:, None]  # the ties of the n-th value: lowest pages first
        sel = gt | (eq & (teq <= room[:, None]))
        ties += tl.sum(eq.to(tl.int32), 1)
        fixed = (p[None, :] < sink) | ((p[None, :] >= (cur - wp)[:, None]) & (p[None, :] <= cur[:, None]))
        allowed = (sel | fixed) & (p[None, :] <= cur[:, None]) & rmask[:, None] & pin[None, :]
        word = tl.sum(allowed.to(tl.int64) << shift[:, None], 0)  # [BP]: bit i = row i
        blk = word != 0
        slot = count + tl.cumsum(blk.to(tl.int32), 0) - 1
        tl.store(IDX + g * sig + t * NB + slot, p0 + p, mask=blk)
        tl.store(BITS + g * sig + t * NB + slot, word, mask=blk)
        count += tl.sum(blk.to(tl.int32), 0)
    tl.store(CNT + g * scn + t + tl.arange(0, 1), count)


def token_lists_packed(qsum: torch.Tensor, win: torch.Tensor, q_pos: torch.Tensor, tile_p0: torch.Tensor,
                       tile_np: torch.Tensor, PM: int, n: int, page: int, sink: int, wp: int, tq: int,
                       idx: torch.Tensor, bits: torch.Tensor, cnt: torch.Tensor, chunk: int) -> None:
    """Pure per-token page_maxpool selection of every span of a packed layout, straight into the head-packed lists:
    qsum [G, R, d] fp32 (each group's query-head sum, scaled; R rows, spans at multiples of 64), win [G, 4 Pk, d]
    (each span's windows at 4 x its first page, a span's last window zero-padded), q_pos [R] (position in the span,
    -1 padding), tile_p0 / tile_np [R / 64] (each 64-row tile's span's first page and page count), PM the longest
    span's pages. Writes idx [G, R / tq, NB] (packed-layout pages), bits, cnt in ``chunk`` rows at a time (the [G,
    chunk, PM] scores and one top-k per chunk: no per-span launches). Same lists as token_lists_triton per span."""
    G, R, d = qsum.shape
    NB = idx.shape[-1]
    nn_ = min(n, PM)
    ieee = not (torch.backends.cuda.matmul.allow_tf32 or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1")
    BQ, BP = 64, 16
    sc = torch.empty(G, min(chunk, R), PM, device=qsum.device, dtype=torch.float32)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        rc = r1 - r0
        scr = sc[:, :rc]
        qs = qsum[:, r0:r1]
        tp0, tpn = tile_p0[r0 // 64:r1 // 64], tile_np[r0 // 64:r1 // 64]
        qp = q_pos[r0:r1]
        _maxpool_scores_packed_kernel[(rc // BQ, triton.cdiv(PM, BP), G)](
            qs, win, scr, qp, tp0, tpn, rc, PM, win.shape[1], page, sink, wp,
            qs.stride(0), qs.stride(1), win.stride(0), win.stride(1), scr.stride(0), scr.stride(1),
            D=d, BQ=BQ, BP=BP, IEEE=ieee, num_warps=4)
        v = scr.topk(nn_, dim=-1).values[..., -1].contiguous()  # [G, rc]
        t0 = r0 // tq
        _token_lists_packed_kernel[(rc // tq, G)](
            scr, v, idx[:, t0:], bits[:, t0:], cnt[:, t0:], qp, tp0, tpn, qp, rc, PM, NB, page, sink, wp, nn_,
            scr.stride(0), v.stride(0), idx.stride(0), cnt.stride(0), TQ=tq, BP=64, HAS_ANC=False, num_warps=4)


def token_lists_packed_anchor(qsum_a: torch.Tensor, win: torch.Tensor, a_pos: torch.Tensor, a_tile_p0: torch.Tensor,
                              a_tile_np: torch.Tensor, q_pos: torch.Tensor, tile_p0: torch.Tensor,
                              tile_np: torch.Tensor, arow: torch.Tensor, chunks, amax: int, PM: int, n: int, page: int,
                              sink: int, wp: int, tq: int, idx: torch.Tensor, bits: torch.Tensor,
                              cnt: torch.Tensor) -> None:
    """:func:`token_lists_packed` with one selection per block of queries (page_sparse select_block B > 1): only the
    anchors are scored -- qsum_a [G, Ra, d] in an anchor layout (each span's anchors, padded to 64 rows; a_pos their
    positions, -1 padding; a_tile_p0 / a_tile_np per 64-row anchor tile) -- with the anchors' candidates and one top-k
    per anchor; every query row then takes its anchor's (arow [R]: its anchor-layout row) top-k pages plus its OWN
    sinks and window. ``chunks``: host (r0, r1, a0, a1) -- query rows [r0, r1) read anchor rows [a0, a1) (64-aligned),
    ``amax`` the widest a1 - a0. Same lists as page_sparse.select_pages(select_block=B) per span."""
    G, Ra, d = qsum_a.shape
    NB = idx.shape[-1]
    nn_ = min(n, PM)
    ieee = not (torch.backends.cuda.matmul.allow_tf32 or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1")
    BQ, BP = 64, 16
    sc = torch.empty(G, amax, PM, device=qsum_a.device, dtype=torch.float32)
    for r0, r1, a0, a1 in chunks:
        ra, rc = a1 - a0, r1 - r0
        scr = sc[:, :ra]
        qs = qsum_a[:, a0:a1]
        _maxpool_scores_packed_kernel[(ra // BQ, triton.cdiv(PM, BP), G)](
            qs, win, scr, a_pos[a0:a1], a_tile_p0[a0 // 64:a1 // 64], a_tile_np[a0 // 64:a1 // 64], ra, PM,
            win.shape[1], page, sink, wp,
            qs.stride(0), qs.stride(1), win.stride(0), win.stride(1), scr.stride(0), scr.stride(1),
            D=d, BQ=BQ, BP=BP, IEEE=ieee, num_warps=4)
        v = scr.topk(nn_, dim=-1).values[..., -1].contiguous()  # [G, ra]
        anc = (arow[r0:r1] - a0).contiguous()
        t0 = r0 // tq
        _token_lists_packed_kernel[(rc // tq, G)](
            scr, v, idx[:, t0:], bits[:, t0:], cnt[:, t0:], q_pos[r0:r1], tile_p0[r0 // 64:r1 // 64],
            tile_np[r0 // 64:r1 // 64], anc, rc, PM, NB, page, sink, wp, nn_,
            scr.stride(0), v.stride(0), idx.stride(0), cnt.stride(0), TQ=tq, BP=64, HAS_ANC=True, num_warps=4)
