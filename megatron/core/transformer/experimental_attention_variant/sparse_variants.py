# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Trained sparse-attention variants on page_sparse.py's primitives (the midtraining campaign's NSA / CSA2 / LSA
arms, docs/campaign_3b_hybrid_midtraining.md in oellm-autoexp). All work on ONE sequence (document): queries
q [1, hq, s, d] at positions q_offset .. q_offset + s - 1 against keys / values [1, g, L, d] at 0 .. L - 1, GQA
(hq % g == 0), ONE selection per token for all heads, causal by construction. Pure torch, differentiable, exact
reference implementations (the selections themselves carry no gradient); fast kernels later for the winners.

:class:`NSA`   Native Sparse Attention (Yuan et al. 2025, arXiv 2502.11089), adapted to a shared selection:
               a compression branch (overlapping blocks of ``block`` keys, stride ``stride``, compressed by a
               learned map initialised to the mean, attended causally), a selection branch (pages ranked by the
               compression attention's mass, summed over heads; + sinks and the local pages), a sliding-window
               branch, combined by learned per-head sigmoid gates on the attention input. The gate bias starts
               on the selection branch (a converted dense model starts as page selection).
:class:`CSA2`  DeepSeek-V4.1's Compressed Sparse Attention 2, adapted to our GQA layers: the main KV compressed
               ``compress``:1 with learned per-token weights (V4.1's Z = H W^Z, softmax within each group; zero
               init = the mean), a lightning indexer whose keys are projected from the compressed main keys
               (V4.1: "indexer K projected from main KV"), the top ``topk_entries`` compressed entries attended
               together with a raw sliding window (V4.1: 128) and the sinks. Indexer warm-up: KL to the dense
               attention's mass per compressed entry (:meth:`CSA2.indexer_loss`).
:class:`LSA`   Latent Sparse Attention (arXiv 2610.42147), N-LSA flat search: L2-normalised keys pooled ``pool``:1,
               clustered within blocks of ``block`` pooled keys into ``buckets`` buckets (spherical k-means, a
               cheap stand-in for the paper's Ward linkage), each bucket a normalised centroid; a query scores
               the centroids (cosine, max over the heads of each KV group, summed over groups) and attends to the
               members of its ``topk`` best buckets + the sinks + the window. No parameters; the model's own q / k
               train through the attention (the selection is stop-gradient, as in the paper).
Hierarchical selection: ``PageSparseConfig.hier_group`` / ``hier_topk`` in page_sparse.py (coarse-to-fine).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:  # inside Megatron
    from . import page_sparse as ps
except ImportError:  # loaded by path (the HF export / the probes register page_sparse first)
    import page_sparse as ps


def _span_mask(q_pos, lo, hi):
    """[s, n]: entries covering [lo, hi) (inclusive lo, exclusive hi) that lie fully before each query's window
    is decided by the caller; this one is just the causal 'ends before or at the query' test."""
    return hi[None, :] - 1 <= q_pos[:, None]


def _attend_entries(q, ke, ve, ok, scale, chunk=512, reduce=None):
    """Dense attention of q [1, hq, s, d] over entries ke / ve [1, g, n, d] with a validity mask ok [s, n] (per
    query); queries without a valid entry get zeros. ``reduce(p)`` maps a query chunk's probabilities [1, hq, c, n]
    (no gradient) to a smaller summary [1, c, ...]; the summaries are concatenated over the chunks. Chunks are
    checkpointed under grad (the probabilities are never stored for all queries). Returns (out, summary or None)."""
    _, hq, s, d = q.shape
    g, n = ke.shape[1], ke.shape[2]
    rep = hq // g

    def block(qc, okc):
        lg = torch.einsum("bgrsd,bgnd->bgrsn", ps._acc(qc).view(1, g, rep, -1, d), ps._acc(ke)) * scale
        p = torch.softmax(lg.masked_fill(~okc[None, None, None], float("-inf")), -1).nan_to_num(0.0)
        return torch.einsum("bgrsn,bgnd->bgrsd", p.to(ve.dtype), ve).reshape(1, hq, -1, d), p

    outs, summ = [], []
    grad = torch.is_grad_enabled() and (q.requires_grad or ke.requires_grad or ve.requires_grad)
    for c0 in range(0, s, chunk):
        c1 = min(s, c0 + chunk)
        if grad:
            outs.append(torch.utils.checkpoint.checkpoint(lambda a, b: block(a, b)[0], q[:, :, c0:c1], ok[c0:c1],
                                                          use_reentrant=False))
        else:
            o, p = block(q[:, :, c0:c1], ok[c0:c1])
            outs.append(o)
        if reduce is not None:
            with torch.no_grad():
                if grad:
                    _, p = block(q[:, :, c0:c1].detach(), ok[c0:c1])
                summ.append(reduce(p.reshape(1, hq, c1 - c0, n)))
    return torch.cat(outs, 2), (torch.cat(summ, 1) if reduce is not None else None)


# ----------------------------------------------------------------------------------------------- NSA
@dataclass
class NSAConfig:
    block: int = 32  # compression block (tokens)
    stride: int = 16
    page: int = 64  # selection block
    topk: int = 16  # selected pages
    sink_pages: int = 1
    select_window: int = 64  # local pages of the selection branch (tokens before the query's page)
    window: int = 512  # sliding-window branch
    gate_init: tuple = (-3.0, 3.0, -3.0)  # (compression, selection, window) gate biases
    select_block: int = 1  # the selection branch's granularity (page_sparse.PageSparseConfig.select_block)


class NSA(nn.Module):
    def __init__(self, hidden: int, num_heads: int, kv_channels: int, cfg: NSAConfig):
        super().__init__()
        if cfg.block > cfg.page or cfg.block % cfg.stride:
            raise ValueError("NSA: block <= page and a multiple of stride")
        self.cfg, self.hq, self.d = cfg, num_heads, kv_channels
        self.flex = False  # FlexAttention for the three branches (CUDA); the exact reference otherwise
        d, b = kv_channels, cfg.block
        self.pos = nn.Parameter(torch.zeros(b, d))  # intra-block position embedding
        # the compressors: [block * d] -> d, initialised to the block mean
        eye = torch.eye(d).repeat(1, b) / b  # [d, block * d]
        self.ck = nn.Parameter(eye.clone())
        self.cv = nn.Parameter(eye.clone())
        self.gate = nn.Linear(hidden, num_heads * 3)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.copy_(torch.tensor(cfg.gate_init).repeat(num_heads))

    def compress(self, k, v):
        """[1, g, L, d] -> entries [1, g, n, d] and each entry's (start, end) token span."""
        b, st = self.cfg.block, self.cfg.stride
        L = k.shape[2]
        if L < b:
            z = k[:, :, :0]
            return z, z, torch.zeros(0, dtype=torch.long, device=k.device), torch.zeros(0, dtype=torch.long,
                                                                                         device=k.device)
        kb = k.unfold(2, b, st).permute(0, 1, 2, 4, 3) + self.pos.to(k.dtype)  # [1, g, n, block, d]
        vb = v.unfold(2, b, st).permute(0, 1, 2, 4, 3) + self.pos.to(v.dtype)
        n = kb.shape[2]
        kc = F.linear(kb.reshape(1, k.shape[1], n, -1), self.ck.to(k.dtype))
        vc = F.linear(vb.reshape(1, v.shape[1], n, -1), self.cv.to(v.dtype))
        start = torch.arange(n, device=k.device) * st
        return kc, vc, start, start + b

    def forward(self, q, k, v, xq, q_offset: int, scale: float):
        cfg = self.cfg
        _, hq, s, d = q.shape
        L = k.shape[2]
        P = -(-L // cfg.page)
        pos = q_offset + torch.arange(s, device=q.device)
        kc, vc, start, end = self.compress(k, v)
        ok = _span_mask(pos, start, end)  # entries that end at or before the query
        # page importance: every entry's mass (summed over heads: one selection per token) to the page(s) it
        # overlaps (block <= page: at most two)
        p0, p1 = start // cfg.page, ((end - 1) // cfg.page).clamp(max=P - 1)
        f0 = torch.where(p0 == p1, torch.ones_like(start, dtype=torch.float64),
                         ((p0 + 1) * cfg.page - start).double() / cfg.block)

        def importance(p):  # [1, hq, c, n] -> [1, c, P]
            mass = p.sum(1)
            imp = torch.zeros(1, mass.shape[1], P, device=q.device, dtype=mass.dtype)
            if start.numel():
                imp.scatter_add_(-1, p0.expand(1, mass.shape[1], -1), mass * f0.to(mass.dtype))
                imp.scatter_add_(-1, p1.expand(1, mass.shape[1], -1), mass * (1 - f0.to(mass.dtype)))
            return imp

        sel_cfg = ps.PageSparseConfig(selector="indexer", page=cfg.page, topk=cfg.topk,
                                      window=cfg.select_window, sink_pages=cfg.sink_pages,
                                      select_block=cfg.select_block)
        win_cfg = ps.PageSparseConfig(selector="window", page=cfg.page, window=cfg.window, sink_pages=0)
        if self.flex and q.is_cuda:
            with torch.no_grad():  # the importances only (no probabilities out of the fused kernel)
                _, imp = _attend_entries(q.detach(), kc.detach(), vc.detach(), ok, scale, reduce=importance)
            o_cmp = self._flex_entries(q, kc, vc, q_offset, scale)
            o_slc = ps.flex_page_sparse_attention(q, k, v, sel_cfg, q_offset, scale, imp, 64)
            o_win = ps.flex_page_sparse_attention(q, k, v, win_cfg, q_offset, scale, None, 64)
        else:
            o_cmp, imp = _attend_entries(q, kc, vc, ok, scale, reduce=importance)
            o_slc = ps.page_sparse_attention(q, k, v, sel_cfg, q_offset, scale, indexer_scores=imp)
            o_win = ps.page_sparse_attention(q, k, v, win_cfg, q_offset, scale)
        gt = torch.sigmoid(ps._acc(self.gate(xq))).view(1, s, hq, 3).permute(0, 2, 1, 3)  # [1, hq, s, 3]
        out = gt[..., 0:1] * ps._acc(o_cmp) + gt[..., 1:2] * ps._acc(o_slc) + gt[..., 2:3] * ps._acc(o_win)
        return out.to(q.dtype)


    def _flex_entries(self, q, kc, vc, q_offset, scale, q_block=64, kv_block=64):
        """The compression branch through FlexAttention: entry e (tokens [e stride, e stride + block)) is visible
        to the queries at or after its last token; queries without a visible entry get zeros."""
        from torch.nn.attention.flex_attention import BlockMask

        cfg = self.cfg
        _, hq, s, d = q.shape
        g, n = kc.shape[1], kc.shape[2]
        if n == 0:
            return torch.zeros_like(q)
        nq, nk = -(-s // q_block), -(-n // kv_block)
        last_q = q_offset + torch.arange(nq, device=q.device).clamp(max=nq) * q_block + q_block - 1
        last_q = last_q.clamp(max=q_offset + s - 1)
        n_vis = ((last_q - cfg.block + 1).div(cfg.stride, rounding_mode="floor") + 1).clamp(0, n)
        kv_num = (-(-n_vis // kv_block)).to(torch.int32)
        kv_idx = torch.arange(nk, device=q.device, dtype=torch.int32).expand(nq, nk).contiguous()
        blk, st = cfg.block, cfg.stride

        def mask_mod(bi, hi, qi, ki):
            return ki * st + blk - 1 <= qi + q_offset

        bm = BlockMask.from_kv_blocks(kv_num[None, None], kv_idx[None, None], BLOCK_SIZE=(q_block, kv_block),
                                      mask_mod=mask_mod, seq_lengths=(s, n))
        out = ps._flex()(q, kc.to(q.dtype), vc.to(q.dtype), block_mask=bm, scale=scale, enable_gqa=hq != g,
                         kernel_options=ps.flex_kernel_options(q, q_block, kv_block))
        # rows without a visible entry: zeros (as the reference), whatever the kernel leaves there
        pos = q_offset + torch.arange(s, device=q.device)
        return out.masked_fill((pos < blk - 1)[None, None, :, None], 0.0)


# ----------------------------------------------------------------------------------------------- CSA2
@dataclass
class CSA2Config:
    compress: int = 2
    topk_entries: int = 512
    window: int = 128  # raw tokens before the query (V4.1's SWA 128; a multiple of page)
    page: int = 64  # the raw window's granularity
    sink_pages: int = 1
    index_heads: int = 4
    index_dim: int = 64


class CSA2(nn.Module):
    def __init__(self, hidden: int, num_query_groups: int, kv_channels: int, cfg: CSA2Config):
        super().__init__()
        if cfg.window % cfg.page or cfg.page % cfg.compress:
            raise ValueError("CSA2: window a multiple of page, page a multiple of compress")
        self.cfg, self.g, self.d = cfg, num_query_groups, kv_channels
        self.z = nn.Linear(hidden, num_query_groups, bias=False)  # the compression weights (zero: the mean)
        nn.init.zeros_(self.z.weight)
        self.idx_q = nn.Linear(hidden, cfg.index_heads * cfg.index_dim, bias=False)
        self.idx_w = nn.Linear(hidden, cfg.index_heads, bias=False)
        self.idx_k = nn.Linear(kv_channels, cfg.index_dim, bias=False)

    def token_features(self, x):
        """Per-token features of the KEY side, gathered with the keys under context parallelism: [t, g]."""
        return self.z(x)

    def compress(self, k, v, z):
        """k / v [1, g, L, d], z [1, L, g] -> entries [1, g, Lc, d] (group e = tokens [e m, (e + 1) m))."""
        m = self.cfg.compress
        g, L, d = k.shape[1], k.shape[2], k.shape[3]
        Lc = L // m
        w = torch.softmax(ps._acc(z[:, : Lc * m]).view(1, Lc, m, g), dim=2).permute(0, 3, 1, 2)  # [1, g, Lc, m]
        kc = (ps._acc(k[:, :, : Lc * m]).view(1, g, Lc, m, d) * w[..., None]).sum(3).to(k.dtype)
        vc = (ps._acc(v[:, :, : Lc * m]).view(1, g, Lc, m, d) * w[..., None]).sum(3).to(v.dtype)
        return kc, vc

    def _candidates(self, s, Lc, q_offset, device):
        cfg = self.cfg
        pos = q_offset + torch.arange(s, device=device)
        first_window = (pos // cfg.page - cfg.window // cfg.page) * cfg.page
        e = torch.arange(Lc, device=device)
        lo = cfg.sink_pages * cfg.page
        return (e[None, :] * cfg.compress >= lo) & ((e[None, :] + 1) * cfg.compress <= first_window[:, None])

    def index_scores(self, xq, kc):
        """xq [1, s, hidden], kc [1, g, Lc, d] -> [1, s, Lc] (lightning-indexer form, keys from the main KV)."""
        cfg = self.cfg
        s = xq.shape[1]
        qi = ps._acc(self.idx_q(xq).view(1, s, cfg.index_heads, cfg.index_dim))
        ki = ps._acc(self.idx_k(ps._acc(kc).mean(1).to(self.idx_k.weight.dtype)))  # [1, Lc, dim]
        w = ps._acc(self.idx_w(xq))
        return torch.einsum("bshn,bsh->bsn", torch.relu(torch.einsum("bshd,bnd->bshn", qi, ki)), w)

    def forward(self, q, k, v, xq, z, q_offset: int, scale: float, chunk: int = 1024):
        cfg = self.cfg
        _, hq, s, d = q.shape
        L = k.shape[2]
        kc, vc = self.compress(k, v, z)
        Lc = kc.shape[2]
        raw_cfg = ps.PageSparseConfig(selector="window", page=cfg.page, window=cfg.window,
                                      sink_pages=cfg.sink_pages)
        tok, tok_ok = ps.select_tokens(q, k, raw_cfg, q_offset, scale)
        n = min(cfg.topk_entries, Lc)
        vals, sels = [], []
        with torch.no_grad():  # the index scores of a query chunk at a time ([c, Lc], not [s, Lc])
            for c0 in range(0, s, chunk):
                c1 = min(s, c0 + chunk)
                cand = self._candidates(c1 - c0, Lc, q_offset + c0, q.device)
                sc = self.index_scores(xq[:, c0:c1], kc).masked_fill(~cand[None], float("-inf"))
                val, sel = sc.topk(n, dim=-1)
                vals.append(val)
                sels.append(sel)
        val, sel = torch.cat(vals, 1), torch.cat(sels, 1)
        kk, vv = torch.cat([k, kc], 2), torch.cat([v, vc], 2)
        idx = torch.cat([tok, L + sel.clamp(min=0)], -1)
        ok = torch.cat([tok_ok, torch.isfinite(val)], -1)
        return ps.attend_gathered(q, kk, vv, idx, ok, scale, None)

    def indexer_loss(self, q, k, xq, z, q_offset: int, scale: float, chunk: int = 256):
        """KL(dense attention mass per compressed entry || softmax(index scores)) over the candidates."""
        cfg = self.cfg
        _, hq, s, d = q.shape
        L = k.shape[2]
        m = cfg.compress
        Lc = L // m
        with torch.no_grad():
            kc, _ = self.compress(k.detach(), k.detach(), z.detach())
        pos = q_offset + torch.arange(s, device=q.device)
        total, count = 0.0, 0
        for c0 in range(0, s, chunk):  # a query chunk at a time; checkpointed: the [c, Lc] scores are not kept
            c1 = min(s, c0 + chunk)
            cand = self._candidates(c1 - c0, Lc, q_offset + c0, q.device)
            if not bool(cand.any()):
                continue
            with torch.no_grad():
                pr = ps._causal_probs(q[:, :, c0:c1].detach(), k.detach(), pos[c0:c1], scale)[..., : Lc * m]
                target = pr.view(1, hq, c1 - c0, Lc, m).sum(-1).mean(1)
            has = int((cand & (target[0] > 0)).any(-1).sum())

            def chunk_loss(x, kc_, target=target, cand=cand):
                return ps.indexer_kl_loss(self.index_scores(x, kc_), target, cand)

            if torch.is_grad_enabled():
                lc = torch.utils.checkpoint.checkpoint(chunk_loss, xq[:, c0:c1], kc, use_reentrant=False)
            else:
                lc = chunk_loss(xq[:, c0:c1], kc)
            total = total + lc * has
            count += has
        return total / max(count, 1) if count else xq.sum() * 0.0


# ----------------------------------------------------------------------------------------------- LSA
@dataclass
class LSAConfig:
    pool: int = 4  # tokens per pooled key
    block: int = 512  # pooled keys per cluster block
    buckets: int = 32  # buckets per block
    iters: int = 4  # spherical k-means iterations
    topk: int = 32  # selected buckets
    window: int = 512
    page: int = 64  # window / sink granularity
    sink_pages: int = 1
    max_members: int = 0  # pooled keys kept per bucket (0: 4x the mean, block / buckets); bounds the gather


class LSA(nn.Module):
    """No parameters; :meth:`index` builds the bucket index of a sequence's keys (cache it per document)."""

    def __init__(self, cfg: LSAConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def index(self, k):
        """k [1, g, L, d] -> (centroids [nb, g, d] (each KV head's part unit-norm), member tokens [nb, M] long,
        member valid [nb, M] bool, each bucket's last token [nb]) over the COMPLETE cluster blocks."""
        cfg = self.cfg
        g, L, d = k.shape[1], k.shape[2], k.shape[3]
        span = cfg.pool * cfg.block  # tokens per cluster block
        nblk = L // span
        if nblk == 0:
            z = torch.zeros(0, g, d, device=k.device)
            return z, torch.zeros(0, 1, dtype=torch.long, device=k.device), torch.zeros(0, 1, dtype=torch.bool,
                                                                                        device=k.device), \
                torch.zeros(0, dtype=torch.long, device=k.device)
        kn = F.normalize(ps._acc(k[0, :, : nblk * span]), dim=-1)  # [g, T, d]
        pooled = F.normalize(kn.view(g, nblk * cfg.block, cfg.pool, d).sum(2), dim=-1)  # [g, Np, d]
        feat = pooled.permute(1, 0, 2).reshape(nblk, cfg.block, g * d) / math.sqrt(g)  # unit norm per pooled key
        C = cfg.buckets
        cent = feat[:, torch.linspace(0, cfg.block - 1, C, device=k.device).long()]  # [nblk, C, g d]
        for _ in range(cfg.iters):
            assign = torch.einsum("bpf,bcf->bpc", feat, cent).argmax(-1)  # [nblk, block]
            oh = F.one_hot(assign, C).to(feat.dtype)  # [nblk, block, C]
            summed = torch.einsum("bpc,bpf->bcf", oh, feat)
            empty = oh.sum(1) == 0
            cent = torch.where(empty[..., None], cent, F.normalize(summed, dim=-1))
        assign = torch.einsum("bpf,bcf->bpc", feat, cent).argmax(-1)
        # members: pooled-key indices of each bucket (padded), then their tokens
        order = torch.argsort(assign * cfg.block + torch.arange(cfg.block, device=k.device), dim=-1)  # by bucket
        counts = F.one_hot(assign, C).sum(1)  # [nblk, C]
        cap = cfg.max_members or 4 * -(-cfg.block // C)
        M = min(int(counts.max()), cap)  # an oversized bucket keeps its first members (by position)
        offs = torch.cumsum(counts, -1) - counts  # [nblk, C]
        slot = torch.arange(M, device=k.device)
        gather = (offs[..., None] + slot).clamp(max=cfg.block - 1)  # [nblk, C, M]
        mem_pool = torch.gather(order[:, None, :].expand(nblk, C, cfg.block), 2, gather)  # pooled idx in block
        mem_ok = slot[None, None, :] < counts[..., None]
        base = (torch.arange(nblk, device=k.device) * cfg.block)[:, None, None]
        pool_glob = (mem_pool + base).view(nblk * C, M)  # global pooled-key index
        tokens = (pool_glob[..., None] * cfg.pool + torch.arange(cfg.pool, device=k.device)).view(nblk * C,
                                                                                                    M * cfg.pool)
        tok_ok = mem_ok.view(nblk * C, M, 1).expand(-1, -1, cfg.pool).reshape(nblk * C, M * cfg.pool)
        cent = cent.view(nblk * C, g, d)  # each KV head's part has norm 1 / sqrt(g) -> renormalise per head
        cent = F.normalize(cent, dim=-1)
        last = (torch.arange(nblk, device=k.device).repeat_interleave(C) + 1) * span - 1
        return cent, tokens, tok_ok, last

    def forward(self, q, k, v, q_offset: int, scale: float, index=None):
        cfg = self.cfg
        _, hq, s, d = q.shape
        g, L = k.shape[1], k.shape[2]
        rep = hq // g
        cent, mem, mem_ok, last = index if index is not None else self.index(k)
        raw_cfg = ps.PageSparseConfig(selector="window", page=cfg.page, window=cfg.window,
                                      sink_pages=cfg.sink_pages)
        tok, tok_ok = ps.select_tokens(q, k, raw_cfg, q_offset, scale)
        nb = cent.shape[0]
        if nb == 0:
            return ps.attend_gathered(q, k, v, tok, tok_ok, scale, None)
        pos = q_offset + torch.arange(s, device=q.device)
        first_window = (pos // cfg.page - cfg.window // cfg.page) * cfg.page
        with torch.no_grad():
            qn = F.normalize(ps._acc(q[0]).view(g, rep, s, d), dim=-1)
            sc = torch.einsum("grsd,ngd->grsn", qn, cent.to(qn.dtype)).amax(1).sum(0)  # [s, nb]
            usable = last[None, :] < first_window[:, None]  # buckets of blocks that end before the window
            sc = sc.masked_fill(~usable, float("-inf"))
            kb = min(cfg.topk, nb)
            val, sel = sc.topk(kb, dim=-1)  # [s, kb]
            stok = mem[sel].view(s, -1)  # [s, kb * M * pool]
            sok = (mem_ok[sel] & torch.isfinite(val)[..., None]).view(s, -1)
            sok = sok & (stok >= cfg.sink_pages * cfg.page)  # the sinks are attended already
        idx = torch.cat([tok, stok[None]], -1)
        ok = torch.cat([tok_ok, sok[None]], -1)
        return ps.attend_gathered(q, k, v, idx, ok, scale, None)
