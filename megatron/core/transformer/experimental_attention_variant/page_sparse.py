# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Page-sparse attention for GQA softmax-attention layers (the midtraining campaign's sparse-attention variants,
docs/campaign_3b_hybrid_midtraining.md in oellm-autoexp).

Every query attends to a bounded set of keys, so the softmax sees a constant number of keys at any context
length (NoPE: no position re-scaling needed) and long-context KV can live in host memory, fetched page by page:

* the sink pages (the first ``sink_pages`` pages of the sequence);
* a local window: the query's own page and the ``window // page`` pages before it;
* ``topk`` pages (or tokens / compressed entries) chosen by a selector among the earlier ones.

ONE selection per query token, shared by all heads (our GQA has 2 KV heads; a shared index keeps the index cache
and the KV fetch per token small). Selectors (``PageSparseConfig.selector``):

``dense``          full causal attention (the reference)
``window``         sinks + window only (the lower bound: what the linear-attention layers carry alone)
``page_mean``      pages scored by q . mean(k of the page), summed over heads (MoBA / InfLLM-v2 style; no params)
``quest``          pages scored by Quest's upper bound sum_d max(q_d kmax_d, q_d kmin_d), summed over heads
``oracle``         pages scored by their true attention mass (dense softmax, averaged over heads): the ceiling of
                   any page selector at the budget, and the warm-up target of a learned indexer
``token_topk``     the ``topk * page`` tokens of highest attention mass (token-level, DSA's ceiling; scattered)
``compress_mean``  CSA2-like: the earlier context as mean-pooled entries of ``compress`` tokens (k and v), the
                   ``topk * page // compress`` entries of highest q . k_c attended with a log(compress) bias
``indexer``        pages scored by a learned :class:`PageIndexer` (lightning-indexer form on page-pooled index keys)

Hierarchical (coarse-to-fine) selection for every page selector (``hier_group`` > 0): groups of ``hier_group``
pages are scored first (the same selector on group-pooled keys; the indexer by its best page), and the pages are
chosen only within each query's ``hier_topk`` best groups -- the per-query cost of a flat score over P pages
becomes P / hier_group + hier_topk * hier_group (10M tokens: the index scan stays small).

The trained variants NSA, CSA2 and LSA live in sparse_variants.py, on these primitives.

Pure torch (no Megatron imports): the HF export loads this file as is. The gathered-key implementation is exact
and differentiable (training at small scale; ``torch.utils.checkpoint`` per query chunk under autograd); fast
kernels replace :func:`attend_gathered` later with the same selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

def _acc(x: torch.Tensor) -> torch.Tensor:
    """The accumulation dtype: fp32, float64 kept (exact tests: the NGC container forces TF32 for fp32 GEMMs)."""
    return x if x.dtype == torch.float64 else x.float()


SELECTORS = ("dense", "window", "page_mean", "quest", "oracle", "token_topk", "compress_mean", "indexer",
             "page_maxpool")
VARIANT_SELECTORS = ("nsa", "csa2", "lsa")  # trained modules of sparse_variants.py (PageSparseCoreAttention)


@dataclass
class PageSparseConfig:
    selector: str = "page_mean"
    page: int = 64
    topk: int = 32  # selected pages; token_topk: topk * page tokens; compress_mean: topk * page // compress entries
    window: int = 512  # tokens before the query's page (rounded down to pages)
    sink_pages: int = 1
    compress: int = 4
    query_chunk: int = 512  # queries per gathered-attention block
    score_chunk: int = 128  # queries per dense-score block (oracle, token_topk)
    hier_group: int = 0  # hierarchical selection: pages per group (0: flat)
    hier_topk: int = 8  # groups kept per query
    # selection granularity: 1 = every query selects its own pages; B > 1 = one selection per block of B queries
    # (absolute positions), made by the block's first query and reused by the rest (V4.1's "Reuse"; decode
    # selects at every B-th token). A page union per block of B: block-sparse kernels attend just the budget.
    # Chunk-invariant when chunks start at multiples of B (a chunk starting mid-block anchors at its first query;
    # PageSparseCoreAttention prepends the block's earlier queries under CP, so its chunks always start aligned).
    select_block: int = 1
    # how a block's selection is scored: "first" = its first query's scores; "prev_max" = the max over the first
    # query and every query of the previous block (causal: all precede the block; decode knows them)
    select_score: str = "first"
    # one selection per KV group (from that group's query heads and keys) instead of one for all heads
    per_group: bool = False
    # per-token selection (select_block 1) within block candidates: each block of ``candidate_block`` queries first
    # keeps the ``block_candidates`` pages most often in its queries' own top-k, then every query selects its top-k
    # among those -- a bounded page union per block (block-sparse kernels stay near the block-wise cost) with
    # per-token choices; 0: plain per-token selection
    block_candidates: int = 0
    candidate_block: int = 64

    def __post_init__(self):
        if self.selector not in SELECTORS + VARIANT_SELECTORS:
            raise ValueError(f"unknown selector {self.selector!r}; one of {SELECTORS + VARIANT_SELECTORS}")
        if self.window % self.page:
            raise ValueError("window must be a multiple of page")
        if self.select_score not in ("first", "prev_max"):
            raise ValueError("select_score must be first or prev_max")

    @property
    def budget(self) -> int:
        """Keys a query attends to at most (sinks + window + selection)."""
        if self.selector == "dense":
            return -1
        sel = 0 if self.selector == "window" else self.topk * self.page
        return self.sink_pages * self.page + self.window + self.page + sel


# --------------------------------------------------------------------------- page statistics
def page_pool(k: torch.Tensor, page: int, kind: str = "mean"):
    """k [b, g, L, d] -> per page [b, g, P, d] (mean, or (max, min) for quest); the last page may be partial."""
    b, g, L, d = k.shape
    P = -(-L // page)
    pad = P * page - L
    kf = _acc(k)
    if kind == "mean":
        x = F.pad(kf, (0, 0, 0, pad)).view(b, g, P, page, d).sum(3)
        cnt = torch.full((P,), float(page), device=k.device)
        cnt[-1] = page - pad
        return x / cnt.view(1, 1, P, 1)
    hi = F.pad(kf, (0, 0, 0, pad), value=float("-inf")).view(b, g, P, page, d).amax(3)
    lo = F.pad(kf, (0, 0, 0, pad), value=float("inf")).view(b, g, P, page, d).amin(3)
    return hi, lo


def _query_pages(s: int, q_offset: int, cfg: PageSparseConfig, device):
    pos = q_offset + torch.arange(s, device=device)
    return pos, pos // cfg.page


def candidate_mask(s: int, P: int, q_offset: int, cfg: PageSparseConfig, device, pos=None) -> torch.Tensor:
    """[s, P]: the pages a selector may choose for each query (after the sinks, before the window); ``pos`` the
    queries' absolute positions (default q_offset .. q_offset + s - 1)."""
    cur = (_query_pages(s, q_offset, cfg, device)[0] if pos is None else pos) // cfg.page
    p = torch.arange(P, device=device)
    first_window = cur - cfg.window // cfg.page
    return (p[None, :] >= cfg.sink_pages) & (p[None, :] < first_window[:, None])


def _causal_probs(q, k, q_pos, scale, k_lo=0):
    """Dense attention probabilities [b, hq, sq, L] of q at absolute positions q_pos over k[..., k_lo:, :]."""
    b, hq, sq, d = q.shape
    g = k.shape[1]
    rep = hq // g
    kk = _acc(k[:, :, k_lo:])
    logits = torch.einsum("bgrsd,bgld->bgrsl", _acc(q).view(b, g, rep, sq, d), kk) * scale
    kpos = k_lo + torch.arange(kk.shape[2], device=q.device)
    logits = logits.masked_fill(kpos[None, None, None, None, :] > q_pos[None, None, None, :, None], float("-inf"))
    return torch.softmax(logits, -1).view(b, hq, sq, -1)


def page_scores(q, k, cfg: PageSparseConfig, q_offset: int, scale: float, indexer_scores=None, pos=None):
    """Selector scores [b, s, P] of every page for every query (higher = more important); ``pos`` the queries'
    absolute positions (default q_offset .. q_offset + s - 1; the oracle's causal mask)."""
    b, hq, s, d = q.shape
    g, L = k.shape[1], k.shape[2]
    rep = hq // g
    P = -(-L // cfg.page)
    qg = _acc(q).view(b, g, rep, s, d)
    if cfg.selector == "indexer":
        if indexer_scores is None:
            raise ValueError("selector 'indexer' needs indexer_scores (PageIndexer)")
        return _acc(indexer_scores)
    if cfg.selector == "page_mean":
        kp = page_pool(k, cfg.page, "mean")
        return torch.einsum("bgsd,bgpd->bsp", qg.sum(2) * scale, kp)  # the heads' sum first: rep x fewer flops
    if cfg.selector == "page_maxpool":
        return maxpool_page_scores(qg.sum(2), maxpool_windows(k, cfg.page), scale, P)
    if cfg.selector == "quest":
        hi, lo = page_pool(k, cfg.page, "minmax")
        hi, lo = hi.nan_to_num(0.0), lo.nan_to_num(0.0)
        return (torch.einsum("bgrsd,bgpd->bsp", qg.clamp(min=0), hi)
                + torch.einsum("bgrsd,bgpd->bsp", qg.clamp(max=0), lo)) * scale
    if cfg.selector == "oracle":
        if pos is None:
            pos, _ = _query_pages(s, q_offset, cfg, q.device)
        out = []
        for c0 in range(0, s, cfg.score_chunk):
            c1 = min(s, c0 + cfg.score_chunk)
            pr = _causal_probs(q[:, :, c0:c1], k, pos[c0:c1], scale)  # [b, hq, sc, L]
            pr = F.pad(pr, (0, P * cfg.page - L)).view(b, hq, c1 - c0, P, cfg.page).sum(-1)
            out.append(pr.mean(1))
        return torch.cat(out, 1)
    raise ValueError(f"page_scores: selector {cfg.selector!r} does not score pages")


def maxpool_windows(k: torch.Tensor, page: int) -> torch.Tensor:
    """page_maxpool's key statistics: k [b, g, L, d] -> windows [b, g, 4 P + 1, d] (fp32), the keys mean-pooled over
    half-page windows at a quarter-page stride (InfLLM-v2), window j = keys [j page / 4, j page / 4 + page / 2)."""
    if page % 4:
        raise ValueError("page_maxpool: page must be a multiple of 4")
    b, g, L, d = k.shape
    P = -(-L // page)
    st, sub = page // 4, page // 2
    # each window the sum of two adjacent quarter-page sums (a cumsum over the keys was 72 % of the selection:
    # torch's outer-dim scan, 17.6 ms per KV group at 64k, 2026-11-20)
    qs = F.pad(_acc(k), (0, 0, 0, (4 * P + 2) * st - L)).view(b, g, 4 * P + 2, st, d).sum(3)
    return (qs[:, :, :-1] + qs[:, :, 1:]) / sub


def maxpool_page_scores(qsum: torch.Tensor, win: torch.Tensor, scale: float, P: int) -> torch.Tensor:
    """page_maxpool scores [b, s, P] from the queries summed over their heads qsum [b, g, s, d] (fp32) and the windows
    of :func:`maxpool_windows`: a page scored by the max over the 5 windows overlapping it (a match straddling a page
    boundary is not diluted by its page)."""
    b, s = qsum.shape[0], qsum.shape[2]
    sc = torch.einsum("bgsd,bgjd->bsj", qsum * scale, win)  # scaled before: one pass less over [s, 4 P + 1]
    pooled = F.max_pool1d(sc.reshape(b * s, 1, -1), kernel_size=5, stride=4, padding=1)  # page p: 4p-1 .. 4p+3
    return pooled.view(b, s, -1)[..., :P]


def topk_mask(scores: torch.Tensor, n: int) -> torch.Tensor:
    """The pages :func:`topk_stable` selects (its n largest, ties to the lower index), finite scores only, as a bool
    mask of ``scores``' shape -- without the full sort: torch.topk's n-th largest value, then every larger score
    and the lowest-index ties of it until n (the stable sort was a third of the per-token selection, 2026-11-21)."""
    n = min(n, scores.shape[-1])
    if n <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    v = scores.topk(n, dim=-1).values[..., -1:]  # the n-th largest
    gt = scores > v
    eq = (scores == v) & torch.isfinite(scores)
    room = n - gt.sum(-1, keepdim=True)
    return gt | (eq & (eq.cumsum(-1) <= room))


def selection_scores(q, k, cfg: PageSparseConfig, q_offset: int, scale: float, indexer_scores=None,
                     per_anchor: bool = False):
    """The selector's page scores [b, s, P] with everything a query may not select at -inf: pages outside its
    candidates, outside its best groups (``hier_group``), and -- with ``select_block`` -- the scores of its
    block's anchor (the block's first query, or the chunk's first if the block started in an earlier chunk).
    ``per_anchor`` (select_block > 1, select_score "first"): (the anchors' scores [b, n_anchor, P], each query's
    anchor row [s]) instead -- what the selection needs, select_block x fewer rows."""
    s, L = q.shape[2], k.shape[2]
    P = -(-L // cfg.page)
    if cfg.select_block > 1 and cfg.select_score == "first":  # score the anchors only (select_block x cheaper)
        B = cfg.select_block
        pos = q_offset + torch.arange(s, device=q.device)
        # the distinct anchors in closed form (torch.unique syncs the host): the chunk's first query, then every block
        # start after it; query i uses anchor row pos_i // B - q_offset // B
        n_anchor = (q_offset + s - 1) // B - q_offset // B + 1
        uniq = (torch.arange(n_anchor, device=q.device) * B + (q_offset // B) * B - q_offset).clamp(min=0)
        inv = pos // B - q_offset // B
        apos = q_offset + uniq
        cand = candidate_mask(len(uniq), P, q_offset, cfg, q.device, pos=apos)
        idx = None if indexer_scores is None else indexer_scores[:, uniq]
        sc = page_scores(q[:, :, uniq], k, cfg, q_offset, scale, idx, pos=apos).masked_fill(~cand[None],
                                                                                         float("-inf"))
        if cfg.hier_group > 0:
            sc = hierarchical_restrict(sc, q[:, :, uniq], k, cfg, q_offset, scale, pos=apos)
        return (sc, inv) if per_anchor else sc[:, inv]
    cand = candidate_mask(s, P, q_offset, cfg, q.device)
    sc = page_scores(q, k, cfg, q_offset, scale, indexer_scores).masked_fill(~cand[None], float("-inf"))
    if cfg.hier_group > 0:
        sc = hierarchical_restrict(sc, q, k, cfg, q_offset, scale)
    if cfg.select_block > 1:
        B = cfg.select_block
        pos = q_offset + torch.arange(s, device=q.device)
        anchor = ((pos // B) * B - q_offset).clamp(min=0)
        if cfg.select_score == "prev_max":
            # per absolute block: the max over its queries in this chunk; block i's selection then uses
            # max(block i - 1, block i's first query)
            blk = pos // B - q_offset // B  # [s] local block index
            nb = int(blk[-1]) + 1
            bmax = sc.new_full((sc.shape[0], nb, sc.shape[2]), float("-inf"))
            bmax = bmax.scatter_reduce(1, blk[None, :, None].expand_as(sc), sc, reduce="amax", include_self=True)
            prev = torch.cat([bmax.new_full(bmax[:, :1].shape, float("-inf")), bmax[:, :-1]], 1)  # [b, nb, P]
            first = sc[:, anchor]
            sc = torch.maximum(first, prev[:, blk])
        else:
            sc = sc[:, anchor]
    return sc


def hierarchical_restrict(sc, q, k, cfg: PageSparseConfig, q_offset: int, scale: float, pos=None):
    """Fine page scores sc [b, s, P] (candidates already masked to -inf) -> the same with every page outside the
    query's ``hier_topk`` best groups of ``hier_group`` pages masked too. Group scores: the selector on
    group-pooled keys (page_mean / quest / oracle), the best page's score otherwise (indexer)."""
    G = cfg.hier_group
    b, s, P = sc.shape
    ng = -(-P // G)
    if cfg.selector in ("page_mean", "quest", "oracle"):
        coarse_cfg = PageSparseConfig(selector=cfg.selector, page=cfg.page * G, topk=cfg.topk, window=0,
                                      sink_pages=0, score_chunk=cfg.score_chunk)
        coarse = page_scores(q, k, coarse_cfg, q_offset, scale, pos=pos)  # [b, s, ng]
        # a group counts only where it has a candidate page
        has = torch.isfinite(F.pad(sc, (0, ng * G - P), value=float("-inf")).view(b, s, ng, G)).any(-1)
        coarse = coarse[..., :ng].masked_fill(~has, float("-inf"))
    else:
        coarse = F.pad(sc, (0, ng * G - P), value=float("-inf")).view(b, s, ng, G).amax(-1)
    val, top = topk_stable(coarse, min(cfg.hier_topk, ng))
    keep = torch.zeros(b, s, ng, dtype=torch.bool, device=sc.device).scatter_(-1, top, torch.isfinite(val))
    keep = keep.repeat_interleave(G, -1)[..., :P]
    return sc.masked_fill(~keep, float("-inf"))


def select_tokens(q, k, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None,
                  indexer_scores=None):
    """The keys of every query: (token indices [b, s, K] long, valid [b, s, K] bool): the sink pages, then the
    selection (pages in page order, or tokens for ``token_topk``), then the window pages (each page's tokens in
    order). Not for ``dense`` / ``compress_mean``."""
    b, hq, s, d = q.shape
    L = k.shape[2]
    scale = d**-0.5 if scale is None else scale
    page = cfg.page
    P = -(-L // page)
    dev = q.device
    pos, cur = _query_pages(s, q_offset, cfg, dev)
    wp = cfg.window // page
    win = (cur[:, None] - torch.arange(wp, -1, -1, device=dev)[None, :]).expand(b, s, wp + 1)
    win_ok = win >= cfg.sink_pages
    sinks = torch.arange(cfg.sink_pages, device=dev).expand(b, s, cfg.sink_pages)
    sink_ok = sinks <= cur[None, :, None]
    pages, pages_ok = [sinks], [sink_ok]
    sel_tok = sel_tok_ok = None
    if cfg.selector == "token_topk":
        sel_tok, sel_tok_ok = _token_topk(q, k, cfg, pos, scale)
    elif cfg.selector != "window" and cfg.topk > 0:
        sc = selection_scores(q, k, cfg, q_offset, scale, indexer_scores)
        val, sel = topk_stable(sc, min(cfg.topk, P))  # ties (page_maxpool's shared boundary window) by page index
        sel, _ = torch.sort(torch.where(torch.isfinite(val), sel, P + 1), dim=-1)  # page order: contiguous fetches
        pages.append(sel)
        # a block's selection may reach into a later query's window (attended there already: no duplicates)
        pages_ok.append((sel < P) & (sel < (cur - wp)[None, :, None]))
    pages.append(win)
    pages_ok.append(win_ok)

    def expand(pg, ok):
        tok = pg.clamp(0, P - 1)[..., None] * page + torch.arange(page, device=dev)
        tok_ok = ok[..., None] & (tok <= pos[None, :, None, None]) & (tok < L)
        return tok.flatten(2), tok_ok.flatten(2)

    parts = [expand(pg, ok) for pg, ok in zip(pages, pages_ok)]
    if sel_tok is not None:
        parts.insert(1, (sel_tok, sel_tok_ok))
    tok = torch.cat([t for t, _ in parts], -1)
    tok_ok = torch.cat([o for _, o in parts], -1)
    return tok.clamp(0, L - 1), tok_ok


def _token_topk(q, k, cfg, pos, scale):
    """The topk * page earlier tokens (after the sinks, before the window) of highest attention mass, summed
    over heads: (indices [b, s, n] in position order, valid [b, s, n])."""
    b, hq, s, d = q.shape
    L = k.shape[2]
    n = min(cfg.topk * cfg.page, L)
    lo = cfg.sink_pages * cfg.page
    kpos = torch.arange(L, device=q.device)
    idx, ok = [], []
    for c0 in range(0, s, cfg.score_chunk):
        c1 = min(s, c0 + cfg.score_chunk)
        pr = _causal_probs(q[:, :, c0:c1], k, pos[c0:c1], scale).sum(1)  # [b, sc, L]
        first_window = (pos[c0:c1] // cfg.page - cfg.window // cfg.page) * cfg.page
        cand = (kpos[None, :] >= lo) & (kpos[None, :] < first_window[:, None])
        val, sel = pr.masked_fill(~cand[None], float("-inf")).topk(n, dim=-1)
        sel, _ = torch.sort(torch.where(torch.isfinite(val), sel, L + 1), dim=-1)
        idx.append(sel.clamp(max=L - 1))
        ok.append(sel < L)
    return torch.cat(idx, 1), torch.cat(ok, 1)


def attend_gathered(q, k, v, tok, tok_ok, scale, bias=None, query_chunk=512):
    """Attention of every query over its own keys: q [b, hq, s, d], k / v [b, g, L, d], tok / tok_ok [b, s, K]
    (bias [b, s, K] added to the logits, e.g. log(compress)). Exact, differentiable; fp32 softmax."""
    b, hq, s, d = q.shape
    g = k.shape[1]
    rep = hq // g
    outs = []

    def block(qc, idx, ok, bc):
        # qc [b, hq, sc, d]; idx / ok [b, sc, K]
        sc_, K = idx.shape[1], idx.shape[2]
        flat = idx.reshape(b, 1, sc_ * K, 1).expand(b, g, sc_ * K, d)
        kg = torch.gather(k, 2, flat).view(b, g, sc_, K, d)
        vg = torch.gather(v, 2, flat).view(b, g, sc_, K, d)
        logits = torch.einsum("bgrsd,bgskd->bgrsk", _acc(qc.view(b, g, rep, sc_, d)), _acc(kg)) * scale
        if bc is not None:
            logits = logits + bc[:, None, None].to(logits.dtype)
        logits = logits.masked_fill(~ok[:, None, None], float("-inf"))
        p = torch.softmax(logits, -1).nan_to_num(0.0)
        o = torch.einsum("bgrsk,bgskd->bgrsd", p.to(vg.dtype), vg)
        return o.reshape(b, hq, sc_, d)

    for c0 in range(0, s, query_chunk):
        c1 = min(s, c0 + query_chunk)
        args = (q[:, :, c0:c1], tok[:, c0:c1], tok_ok[:, c0:c1], None if bias is None else bias[:, c0:c1])
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            outs.append(torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False))
        else:
            outs.append(block(*args))
    return torch.cat(outs, 2)


def _compress_mean_attention(q, k, v, cfg, q_offset, scale):
    """CSA2-like: sinks + window raw; the earlier context as mean-pooled entries of ``compress`` tokens, the
    ``topk * page // compress`` of highest q . k_c (summed over heads) attended with a log(compress) bias."""
    b, hq, s, d = q.shape
    g, L = k.shape[1], k.shape[2]
    m = cfg.compress
    page = cfg.page
    dev = q.device
    # raw part: sinks + window (the page selection without a selection)
    raw_cfg = PageSparseConfig(selector="window", page=page, window=cfg.window, sink_pages=cfg.sink_pages,
                               query_chunk=cfg.query_chunk)
    tok, tok_ok = select_tokens(q, k, raw_cfg, q_offset, scale)
    # compressed entries of the full groups; entry e covers tokens [e m, (e + 1) m)
    Lc = L // m
    kc = _acc(k[:, :, : Lc * m].view(b, g, Lc, m, d)).mean(3).to(k.dtype)
    vc = _acc(v[:, :, : Lc * m].view(b, g, Lc, m, d)).mean(3).to(v.dtype)
    pos, cur = _query_pages(s, q_offset, cfg, dev)
    lo = cfg.sink_pages * page
    first_window = (cur - cfg.window // page) * page
    e = torch.arange(Lc, device=dev)
    cand = (e[None, :] * m >= lo) & ((e[None, :] + 1) * m <= first_window[:, None])
    n = cfg.topk * page // m
    sc = torch.einsum("bgrsd,bged->bse", _acc(q).view(b, g, hq // g, s, d), _acc(kc))
    sc = sc.masked_fill(~cand[None], float("-inf"))
    val, sel = sc.topk(min(n, max(Lc, 1)), dim=-1) if Lc else (sc[..., :0], sc[..., :0].long())
    sel_ok = torch.isfinite(val)
    # one key / value tensor with the raw tokens first, the entries after them
    kk = torch.cat([k, kc], 2)
    vv = torch.cat([v, vc], 2)
    idx = torch.cat([tok, L + sel.clamp(min=0)], -1)
    ok = torch.cat([tok_ok, sel_ok], -1)
    bdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    bias = torch.cat([torch.zeros_like(tok, dtype=bdt), torch.full_like(sel, math.log(m), dtype=bdt)], -1)
    return attend_gathered(q, kk, vv, idx, ok, scale, bias, cfg.query_chunk)


def _per_group(fn, q, k, v, selection, return_selection, **kw):
    """``fn`` once per KV group on that group's query heads and keys (``per_group``): outputs concatenated over
    the heads, selections as a list per group."""
    g = k.shape[1]
    rep = q.shape[1] // g
    outs, sels = [], []
    for gi in range(g):
        r = fn(q[:, gi * rep:(gi + 1) * rep], k[:, gi:gi + 1], v[:, gi:gi + 1],
               selection=None if selection is None else selection[gi], return_selection=True, **kw)
        outs.append(r[0])
        sels.append(r[1])
    if isinstance(outs[0], tuple):  # (out, lse)
        out = (torch.cat([o[0] for o in outs], 1), torch.cat([o[1] for o in outs], 1))
    else:
        out = torch.cat(outs, 1)
    return (out, sels) if return_selection else out


def topk_stable(scores: torch.Tensor, n: int):
    """The n largest of ``scores`` along the last dim, ties to the lower index -- independent of how many trailing
    (unselectable) columns the tensor has. ``torch.topk`` orders tied values arbitrarily, so a score with exact ties
    (CSA2's indexer: a sum of relus, often exactly 0; page_maxpool: two pages sharing their boundary window's max)
    selected differently for different sequence lengths."""
    vals, idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    return vals[..., :n], idx[..., :n]


def page_sparse_attention(q, k, v, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None,
                          indexer_scores=None, selection=None, return_selection: bool = False):
    """Sparse causal attention of queries q [b, hq, s, d] at absolute positions q_offset .. q_offset + s - 1 over
    keys / values k, v [b, g, L, d] at positions 0 .. L - 1 (one sequence; L >= q_offset + s, GQA: hq % g == 0).
    Returns [b, hq, s, d]. ``selection``: the (tokens, valid) of an earlier call to reuse (a shared index across
    layers); ``return_selection``: also return this call's (a list per group with ``per_group``)."""
    d = q.shape[-1]
    scale = d**-0.5 if scale is None else scale
    if cfg.per_group and k.shape[1] > 1 and cfg.selector not in ("dense", "compress_mean"):
        return _per_group(page_sparse_attention, q, k, v, selection, return_selection, cfg=cfg, q_offset=q_offset,
                          scale=scale, indexer_scores=indexer_scores)
    if cfg.selector == "dense":
        return dense_attention(q, k, v, q_offset, scale)
    if cfg.selector == "compress_mean":
        return _compress_mean_attention(q, k, v, cfg, q_offset, scale)
    if cfg.selector in VARIANT_SELECTORS:
        raise ValueError(f"{cfg.selector}: a trained module (sparse_variants.py), not a page selector")
    if selection is None:
        selection = select_tokens(q, k, cfg, q_offset, scale, indexer_scores)
    out = attend_gathered(q, k, v, selection[0], selection[1], scale, None, cfg.query_chunk)
    return (out, selection) if return_selection else out


def dense_attention(q, k, v, q_offset: int = 0, scale: float | None = None):
    """Causal attention, queries at q_offset .. q_offset + s - 1 (bottom-right aligned) over all L keys."""
    from torch.nn.attention.bias import causal_lower_right

    b, hq, s, d = q.shape
    g, L = k.shape[1], k.shape[2]
    rep = hq // g
    scale = d**-0.5 if scale is None else scale
    outs = []
    for j in range(g):  # one KV head at a time: the expanded keys of one group only
        kj = k[:, j: j + 1].expand(b, rep, L, d)
        vj = v[:, j: j + 1].expand(b, rep, L, d)
        qj = q[:, j * rep: (j + 1) * rep]
        mask = causal_lower_right(s, L) if q_offset + s == L else None
        if mask is None:
            raise ValueError("dense_attention: the queries must be the last s positions of the keys")
        outs.append(F.scaled_dot_product_attention(qj, kj, vj, attn_mask=mask, scale=scale))
    return torch.cat(outs, 1)


# --------------------------------------------------------------------------- learned indexer
class PageIndexer(nn.Module):
    """A learned page selector in the lightning-indexer form: index queries (``heads`` x ``dim``) and ONE index
    key per token (``dim``), mean-pooled per page; score(page) = sum_h w_h(x) relu(q_h . k_page). Inference keeps
    only the pooled page keys ([P, dim]: the GPU index). Warm-up: :func:`indexer_kl_loss` against the dense
    attention's page mass (``page_scores`` with selector ``oracle``)."""

    def __init__(self, hidden: int, heads: int = 4, dim: int = 64, page: int = 64):
        super().__init__()
        self.heads, self.dim, self.page = heads, dim, page
        self.wq = nn.Linear(hidden, heads * dim, bias=False)
        self.wk = nn.Linear(hidden, dim, bias=False)
        self.ww = nn.Linear(hidden, heads, bias=False)

    def page_keys(self, x_k: torch.Tensor) -> torch.Tensor:
        """x_k [b, L, hidden] -> pooled index keys [b, P, dim]."""
        kidx = self.wk(x_k)
        return page_pool(kidx.unsqueeze(1), self.page, "mean")[:, 0]

    def forward(self, x_q: torch.Tensor, page_keys: torch.Tensor) -> torch.Tensor:
        """x_q [b, s, hidden], page_keys [b, P, dim] -> scores [b, s, P]."""
        b, s, _ = x_q.shape
        qi = self.wq(x_q).view(b, s, self.heads, self.dim).float()
        w = self.ww(x_q).float()  # [b, s, heads]
        logits = torch.relu(torch.einsum("bshd,bpd->bshp", qi, page_keys.float()))
        return torch.einsum("bshp,bsh->bsp", logits, w) * self.dim**-0.5


def indexer_kl_loss(scores: torch.Tensor, target_mass: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax(scores)) over each query's candidate pages (cand [s, P] or [b, s, P]); queries without
    candidates are skipped. target_mass: page mass (any scale; renormalized over the candidates)."""
    cand = cand.expand_as(scores)
    t = target_mass.float().masked_fill(~cand, 0.0)
    tsum = t.sum(-1, keepdim=True)
    has = tsum[..., 0] > 0
    t = t / tsum.clamp(min=1e-20)
    logp = torch.log_softmax(scores.float().masked_fill(~cand, float("-inf")), -1)
    kl = (t * (torch.log(t.clamp(min=1e-20)) - logp.masked_fill(~cand, 0.0))).sum(-1)
    return (kl * has).sum() / has.sum().clamp(min=1)


# --------------------------------------------------------------------------- fast backend: FlexAttention
PAGE_SELECTORS = ("window", "page_mean", "quest", "oracle", "indexer", "page_maxpool")
_FLEX = {}


def _flex():
    """The compiled FlexAttention, with dynamic shapes: packed documents (and CP chunks, entry / bucket counts) have
    every length, and a static compile recompiles per length until dynamo's recompile limit, then falls back to
    the eager math path that materialises the whole score matrix (the s3 smoke: 48 GiB at 64k)."""
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention

        compiled = torch.compile(flex_attention, dynamic=True)

        def call(q, k, v, **kwargs):
            # contiguous inputs: dynamic shapes do not cover strides, and the callers pass both permuted THD views
            # (batch stride g * d) and contiguous copies (LSA's bucket order) -- each layout a recompile otherwise
            _raise_recompile_limit()
            return compiled(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)

        _FLEX["fn"] = call
    return _FLEX["fn"]


def _raise_recompile_limit(limit: int = 64) -> None:
    """A margin for the per-call-site graphs, re-asserted on every call: the s3 smoke still hit the default 8 with
    a one-time setting (something in the training stack resets dynamo's config)."""
    import torch._dynamo

    cfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name) and getattr(cfg, name) < limit:
            setattr(cfg, name, limit)


def _selects_everything(P: int, cfg: PageSparseConfig) -> bool:
    """A sequence of P pages with at most topk pages outside the sinks and a query's window selects all of them anyway
    (short packed documents: their scoring was CPU-bound overhead, 2026-11-20). Exact only while a select block never
    straddles a page (page a multiple of select_block): a query must have its anchor's window."""
    blocks_in_pages = cfg.select_block <= 1 or cfg.page % cfg.select_block == 0
    return blocks_in_pages and cfg.hier_group <= 0 and P - cfg.sink_pages - (cfg.window // cfg.page + 1) <= cfg.topk


def select_pages(q, k, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None,
                 indexer_scores=None):
    """Every query's pages as a [b, s, P] bool (sinks, selection, window), the page selectors only."""
    if cfg.selector not in PAGE_SELECTORS:
        raise ValueError(f"select_pages: {cfg.selector!r} is not a page selector")
    b, hq, s, d = q.shape
    L = k.shape[2]
    P = -(-L // cfg.page)
    scale = d**-0.5 if scale is None else scale
    dev = q.device
    pos, cur = _query_pages(s, q_offset, cfg, dev)
    p = torch.arange(P, device=dev)
    wp = cfg.window // cfg.page
    allowed = ((p[None, :] < cfg.sink_pages) | ((p[None, :] >= cur[:, None] - wp) & (p[None, :] <= cur[:, None])))
    allowed = allowed & (p[None, :] <= cur[:, None])
    allowed = allowed[None].expand(b, s, P).clone()
    if _selects_everything(P, cfg) and cfg.selector != "window" and cfg.topk > 0:
        p_ok = p[None, :] <= cur[:, None]
        return p_ok[None].expand(b, s, P).clone()
    if cfg.selector != "window" and cfg.topk > 0:
        if cfg.select_block > 1 and cfg.select_score == "first":
            # top-k of the anchors' rows, then each query takes its anchor's (the expanded [s, P] sort was most of
            # the selection's time: s / select_block distinct rows)
            sc, inv = selection_scores(q, k, cfg, q_offset, scale, indexer_scores, per_anchor=True)
            val, sel = topk_stable(sc, min(cfg.topk, P))
            val, sel = val[:, inv], sel[:, inv]
        else:
            sc = selection_scores(q, k, cfg, q_offset, scale, indexer_scores)
            if cfg.block_candidates > 0 and cfg.select_block <= 1:
                sc = restrict_to_block_candidates(sc, cfg, q_offset)
            val, sel = topk_stable(sc, min(cfg.topk, P))
        allowed.scatter_(-1, sel, torch.isfinite(val) | allowed.gather(-1, sel))
    return allowed


def restrict_to_block_candidates(sc, cfg: PageSparseConfig, q_offset: int = 0):
    """Per-query page scores sc [b, s, P] (unselectable at -inf) -> the same with every page outside the query's block
    candidates masked: blocks of ``candidate_block`` queries by absolute position (q_offset + i: the same under any
    chunking / CP split on block boundaries, = the flex rows) keep the ``block_candidates`` pages its FIRST query
    scores highest (that query precedes every query of the block: causal), then every query selects its own top-k
    among them. (Candidates voted by all of a block's queries let later queries decide which pages earlier ones see:
    a causality leak -- those arms trained BELOW dense, 2026-11-20.)"""
    b, s, P = sc.shape
    B, U = cfg.candidate_block, cfg.block_candidates
    if U >= P:
        return sc
    lead = q_offset % B
    qblk = (torch.arange(s, device=sc.device) + lead) // B  # [s] block of each query (0: the call's first)
    # each block's first row, in closed form (searchsorted up to int(qblk[-1]) synced the host)
    first = (torch.arange((s - 1 + lead) // B + 1, device=sc.device) * B - lead).clamp(min=0)
    rv, cand = topk_stable(sc[:, first], min(U, P))  # [b, nb, U]
    keep = torch.zeros(b, len(first), P, dtype=torch.bool, device=sc.device).scatter_(-1, cand, torch.isfinite(rv))
    return sc.masked_fill(~keep[:, qblk], float("-inf"))


def flex_page_sparse_attention(q, k, v, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None,
                               indexer_scores=None, q_block: int = 128, return_lse: bool = False, selection=None,
                               return_selection: bool = False):
    """:func:`page_sparse_attention` for the page selectors through FlexAttention (fused Triton forward and
    backward): the block mask lists, per block of ``q_block`` queries, the union of their pages (built from the
    selection, never by evaluating the dense mask); the per-token mask makes it exact. One batch row (b = 1).
    ``selection``: an earlier call's allowed pages [s, P] to reuse; ``return_selection``: also return this one's."""
    from torch.nn.attention.flex_attention import BlockMask

    b, hq, s, d = q.shape
    g, L = k.shape[1], k.shape[2]
    if b != 1:
        raise ValueError("flex_page_sparse_attention: one sequence (b = 1)")
    scale = d**-0.5 if scale is None else scale
    if cfg.per_group and g > 1 and cfg.selector != "window":
        return _per_group(flex_page_sparse_attention, q, k, v, selection, return_selection, cfg=cfg,
                          q_offset=q_offset, scale=scale, indexer_scores=indexer_scores, q_block=q_block,
                          return_lse=return_lse)
    page = cfg.page
    with torch.no_grad():
        allowed = selection if selection is not None else select_pages(q, k, cfg, q_offset, scale,
                                                                        indexer_scores)[0]  # [s, P]
        P = allowed.shape[1]
        nq = -(-s // q_block)
        rows = F.pad(allowed, (0, 0, 0, nq * q_block - s)).view(nq, q_block, P)
        blk = rows.any(1)  # [nq, P]: pages some query of the block attends
        # full blocks (no mask_mod): pages every query of the block attends entirely -- allowed for all of them
        # (padding rows count as allowed) and wholly before the block's first query
        pad_ok = (torch.arange(nq * q_block, device=q.device) >= s).view(nq, q_block, 1)
        first_q = q_offset + torch.arange(nq, device=q.device) * q_block
        before = (torch.arange(P, device=q.device)[None, :] + 1) * page <= first_q[:, None] + 1
        full = (rows | pad_ok).all(1) & before & blk
        part = blk & ~full
        kv_num = part.sum(-1).to(torch.int32)
        kv_idx = torch.argsort((~part).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        full_num = full.sum(-1).to(torch.int32)
        full_idx = torch.argsort((~full).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        allowed_i = allowed.to(torch.int8)

    def mask_mod(bi, hi, qi, ki):
        return (ki <= qi + q_offset) & (allowed_i[qi, ki // page] > 0)

    bm = BlockMask.from_kv_blocks(kv_num[None, None], kv_idx[None, None], full_num[None, None],
                                  full_idx[None, None], BLOCK_SIZE=(q_block, page), mask_mod=mask_mod,
                                  seq_lengths=(s, L))
    out = _flex()(q, k, v, block_mask=bm, scale=scale, enable_gqa=hq != g, return_lse=return_lse,
                  kernel_options=flex_kernel_options(q, q_block, page))
    return (out, allowed) if return_selection else out


def flex_packed_page_attention(spans, cfg: PageSparseConfig, scale: float, q_block: int = 64, selections=None,
                               return_selection: bool = False, kernel: str = "flex"):
    """:func:`flex_page_sparse_attention` for many sequences at once -- ONE FlexAttention call (forward and
    backward) for every document of a packed batch instead of one per document.

    ``spans``: ``[(q [1, hq, n, d], k [1, g, L, d], v, q_offset, indexer_scores or None)]`` -- each span's queries
    attend its own keys (a document, or a CP chunk of one: L >= q_offset + n). Each span selects its pages as before
    (or reuses ``selections[i]``); the spans are laid out in one padded sequence, every span's queries starting at a
    multiple of ``q_block`` and its keys at a multiple of ``page``, so the kernel's key blocks are exactly the span's
    pages and one block mask covers them all; the per-token mask (same span, causal, selected page) keeps it exact.
    ``per_group``: one selection per KV group, a per-head block mask. Returns the per-span outputs [1, hq, n, d]
    (and the per-span selections)."""
    from torch.nn.attention.flex_attention import BlockMask

    page = cfg.page
    if q_block % page and page % q_block:
        raise ValueError("flex_packed_page_attention: q_block and page must divide one another")
    if selections is None and not return_selection:
        # documents short enough to attend every page (the sparse result is dense causal attention within them): all
        # in ONE call packed without padding -- per-span selection, padding and lists made the many-document
        # sequences (984 documents: 2.1 s vs 0.11 s for one 64k document, 2026-11-20) the step's stragglers
        wp = cfg.window // page

        def everything(sp):
            P = -(-sp[1].shape[2] // page)
            return P <= cfg.sink_pages + wp + 1 or (cfg.selector != "window" and cfg.topk > 0
                                                    and _selects_everything(P, cfg))

        short = [i for i, sp in enumerate(spans) if everything(sp)]
        if short:
            from .rowmask_attention import dense_docs_attention, dense_docs_blocks as _dense_docs_blocks

            outs = [None] * len(spans)
            ss = [spans[i] for i in short]
            qcat = torch.cat([sp[0] for sp in ss], 2)
            kcat = torch.cat([sp[1] for sp in ss], 2)
            vcat = torch.cat([sp[2] for sp in ss], 2)
            dev = qcat.device
            # the per-query bounds on the host (the lengths are host ints: device tensors + repeat_interleave
            # synced the host several times per layer, idling the GPU -- t19d profile, 2026-11-21), one copy over
            n_ = torch.tensor([sp[0].shape[2] for sp in ss])
            L_ = torch.tensor([sp[1].shape[2] for sp in ss])
            off_ = torch.tensor([sp[3] for sp in ss])
            ko_ = torch.cumsum(L_, 0) - L_  # each document's first key in kcat
            qlo = torch.repeat_interleave(ko_, n_)
            first_q = torch.cumsum(n_, 0) - n_
            local = torch.arange(int(n_.sum())) - torch.repeat_interleave(first_q, n_)
            qpos = qlo + torch.repeat_interleave(off_, n_) + local
            nb = _dense_docs_blocks(qpos, qlo)
            bounds = torch.stack([qpos, qlo]).to(torch.int32)
            bounds = bounds.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else bounds.to(dev)
            o = dense_docs_attention(qcat[0], kcat[0], vcat[0], bounds[0], bounds[1], scale, nb=nb)[None]
            for i, part in zip(short, o.split([sp[0].shape[2] for sp in ss], 2)):
                outs[i] = part
            rest = [i for i in range(len(spans)) if outs[i] is None]
            if rest:
                for i, part in zip(rest, flex_packed_page_attention([spans[i] for i in rest], cfg, scale, q_block,
                                                                    kernel=kernel)):
                    outs[i] = part
            return outs
    q0, k0 = spans[0][0], spans[0][1]
    hq, d, g = q0.shape[1], q0.shape[3], k0.shape[1]
    rep = hq // g
    per_group = cfg.per_group and g > 1 and cfg.selector != "window"
    G = g if per_group else 1
    dev = q0.device
    if kernel == "rowmask" and per_group and cfg.select_block <= 1 and cfg.block_candidates <= 0:
        return _token_packed_attention(spans, cfg, scale, q_block, selections, return_selection)
    with torch.no_grad():
        sels, qoff, koff = [], [0], [0]
        for i, (q, k, v, off, idx) in enumerate(spans):
            sel = None if selections is None else selections[i]
            if sel is None:
                if per_group:
                    sel = [select_pages(q[:, gi * rep:(gi + 1) * rep], k[:, gi:gi + 1], cfg, off, scale, idx)[0]
                           for gi in range(g)]
                else:
                    sel = select_pages(q, k, cfg, off, scale, idx)[0]
            sels.append(sel)
            qoff.append(qoff[-1] + -(-q.shape[2] // q_block) * q_block)
            koff.append(koff[-1] + -(-k.shape[2] // page) * page)
        Sq, Sk = qoff[-1], koff[-1]
        Pk, nq = Sk // page, Sq // q_block
        allowed = torch.zeros(G, Sq, Pk, dtype=torch.bool, device=dev)
        q_abs = torch.full((Sq,), -1, dtype=torch.int32, device=dev)  # position within the span's sequence
        k_abs = torch.full((Sk,), 2**30, dtype=torch.int32, device=dev)
        kend = torch.full((Pk,), 2**30, dtype=torch.int64, device=dev)  # last position of a full page
        for i, (q, k, v, off, idx) in enumerate(spans):
            n, L = q.shape[2], k.shape[2]
            P = -(-L // page)
            a, c = qoff[i], koff[i] // page
            sel = sels[i] if per_group else [sels[i]]
            for gi in range(G):
                allowed[gi, a:a + n, c:c + P] = sel[gi]
            q_abs[a:a + n] = torch.arange(off, off + n, device=dev, dtype=torch.int32)
            k_abs[koff[i]:koff[i] + L] = torch.arange(L, device=dev, dtype=torch.int32)
            j = torch.arange(P, device=dev)
            kend[c:c + P] = torch.where((j + 1) * page <= L, (j + 1) * page - 1, 2**30)
        rows = allowed.view(G, nq, q_block, Pk)
        blk = rows.any(2)  # [G, nq, Pk]
        pad_row = (q_abs < 0).view(1, nq, q_block, 1)
        first_q = q_abs.view(nq, q_block)[:, 0].to(torch.int64)  # spans start at block boundaries
        full = (rows | pad_row).all(2) & (kend[None, None, :] <= first_q[None, :, None]) & blk
        part = blk & ~full
        kv_num = part.sum(-1).to(torch.int32)
        kv_idx = torch.argsort((~part).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        full_num = full.sum(-1).to(torch.int32)
        full_idx = torch.argsort((~full).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        if per_group:  # one block mask row per query head: its group's (sorted once per group, not per head)
            kv_num, kv_idx, full_num, full_idx = (t.repeat_interleave(rep, 0)
                                                  for t in (kv_num, kv_idx, full_num, full_idx))
        allowed_i = allowed.to(torch.int8)

    def pad_rows(t, to):  # [1, h, n, d] -> [1, h, to, d]
        return F.pad(t, (0, 0, 0, to - t.shape[2]))

    qp = torch.cat([pad_rows(sp[0], qoff[i + 1] - qoff[i]) for i, sp in enumerate(spans)], 2)
    kp = torch.cat([pad_rows(sp[1], koff[i + 1] - koff[i]) for i, sp in enumerate(spans)], 2)
    vp = torch.cat([pad_rows(sp[2], koff[i + 1] - koff[i]) for i, sp in enumerate(spans)], 2)

    if kernel == "rowmask":  # one bit word per (query tile, page) instead of flex's per-element mask_mod
        from .rowmask_attention import BM, lists_from_allowed, rowmask_block_attention

        if q_block != BM:
            raise ValueError(f"rowmask kernel: q_block must be {BM}")
        # the list width as a static bound (no host sync): a 64-query tile spans at most two selection / candidate
        # blocks and two pages, so it lists at most the sinks, wp + 2 window pages and twice the per-row choice
        per_row = 0 if cfg.selector == "window" else (cfg.block_candidates if cfg.block_candidates > 0 else cfg.topk)
        nb = max(1, min(Pk, cfg.sink_pages + cfg.window // page + 2 + 2 * per_row))
        idx, bits, cnt = lists_from_allowed(allowed, nb=nb)
        out = rowmask_block_attention(qp[0], kp[0], vp[0], idx, bits, cnt, q_abs, k_abs, scale, per_group,
                                      page)[None]
        segs = out.split([qoff[i + 1] - qoff[i] for i in range(len(spans))], dim=2)
        outs = [seg[:, :, :sp[0].shape[2]] for seg, sp in zip(segs, spans)]
        return (outs, sels) if return_selection else outs

    if per_group:
        def mask_mod(bi, hi, qi, ki):
            return (k_abs[ki] <= q_abs[qi]) & (allowed_i[hi // rep, qi, ki // page] > 0)
    else:
        allowed_2d = allowed_i[0]  # a constant index into a 3-D buffer breaks flex's compile (torch 2.10)

        def mask_mod(bi, hi, qi, ki):
            return (k_abs[ki] <= q_abs[qi]) & (allowed_2d[qi, ki // page] > 0)

    H = hq if per_group else 1
    bm = BlockMask.from_kv_blocks(kv_num.view(1, H, nq), kv_idx.view(1, H, nq, Pk), full_num.view(1, H, nq),
                                  full_idx.view(1, H, nq, Pk), BLOCK_SIZE=(q_block, page), mask_mod=mask_mod,
                                  seq_lengths=(Sq, Sk))
    out = _flex()(qp, kp, vp, block_mask=bm, scale=scale, enable_gqa=hq != g,
                  kernel_options=flex_kernel_options(qp, q_block, page))
    # split, then trim each span's padding: a slice per span backpropagates a zero tensor of the whole packed output
    # per span (64 x 1k documents: 3/4 of the step was those adds / fills, 2026-11-20), split's backward is one cat
    segs = out.split([qoff[i + 1] - qoff[i] for i in range(len(spans))], dim=2)
    outs = [seg[:, :, :sp[0].shape[2]] for seg, sp in zip(segs, spans)]
    return (outs, sels) if return_selection else outs


TOKEN_SELECT_CHUNK = 4096  # queries per selection block of the pure per-token path (scores [chunk, 4 P + 1] fp32)


def token_tile(rep: int, device=None) -> int:
    """Queries per head-packed tile for a KV group of ``rep`` query heads: 128 rows on GPUs with an H100's shared
    memory (16 queries x 8 heads at v4b's 16 / 2: 1.3x faster than 8 x 8 on real selections at 64k and 256k despite
    the larger page union, 2026-11-21), 64 on smaller ones; at least 8 queries."""
    smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin if torch.cuda.is_available() else 0
    rows = 128 if smem >= 160 * 1024 else 64
    return max(8, rows // (1 << (rep - 1).bit_length()))


def token_allowed(q, k, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None, kstats=None):
    """:func:`select_pages` (b = 1, [s, P]) for pure per-token selection with page_mean / page_maxpool: the same pages,
    with the key statistics ``kstats`` (:func:`maxpool_windows` / the page means) computed once per span and the
    top-k as a mask (:func:`topk_mask`) instead of a stable sort; other configurations go to select_pages."""
    s, L = q.shape[2], k.shape[2]
    P = -(-L // cfg.page)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    fast = (cfg.selector in ("page_mean", "page_maxpool") and cfg.select_block <= 1 and cfg.hier_group <= 0
            and cfg.block_candidates <= 0 and cfg.topk > 0 and not _selects_everything(P, cfg))
    if not fast:
        return select_pages(q, k, cfg, q_offset, scale)[0]
    b, hq, _, d = q.shape
    g = k.shape[1]
    qsum = q.view(b, g, hq // g, s, d).sum(2, dtype=torch.float32 if q.dtype != torch.float64 else q.dtype)
    pos = q_offset + torch.arange(s, device=q.device)
    cur = pos // cfg.page
    p = torch.arange(P, device=q.device)
    wp = cfg.window // cfg.page
    if cfg.selector == "page_maxpool":
        win = maxpool_windows(k, cfg.page) if kstats is None else kstats
        if q.is_cuda and g == 1 and qsum.dtype == torch.float32 and d in (16, 32, 64, 128, 256):
            from .rowmask_attention import maxpool_page_scores_triton  # scores + candidates fused

            sc = maxpool_page_scores_triton((qsum[0, 0] * scale).contiguous(), win[0, 0].contiguous(), P, q_offset,
                                            cfg.page, cfg.sink_pages, wp)
        else:
            sc = maxpool_page_scores(qsum, win, scale, P)[0]
            sc = sc.masked_fill(~((p[None, :] >= cfg.sink_pages) & (p[None, :] < (cur - wp)[:, None])),
                                float("-inf"))
    else:
        kp = page_pool(k, cfg.page, "mean") if kstats is None else kstats
        sc = torch.einsum("bgsd,bgpd->bsp", qsum * scale, kp)[0]
        sc = sc.masked_fill(~((p[None, :] >= cfg.sink_pages) & (p[None, :] < (cur - wp)[:, None])), float("-inf"))
    fixed = (p[None, :] < cfg.sink_pages) | ((p[None, :] >= (cur - wp)[:, None]) & (p[None, :] <= cur[:, None]))
    return (topk_mask(sc, cfg.topk) | fixed) & (p[None, :] <= cur[:, None])


def token_span_lists(q, k, cfg: PageSparseConfig, q_offset: int, scale: float, indexer_scores=None, tq: int = 8,
                     chunk: int = TOKEN_SELECT_CHUNK):
    """Pure per-token selection of one span (every query its own top-k pages, per KV group) straight into the
    head-packed kernel's lists: (idx [g, ceil(n / tq), NB] span-local pages, bits [.., NB] tq-bit row words, cnt),
    selected ``chunk`` queries at a time -- the [n, P] page mask of a whole 256k span is never built."""
    from .rowmask_attention import lists_from_allowed

    hq, n, g = q.shape[1], q.shape[2], k.shape[1]
    rep = hq // g
    if chunk % 64:
        raise ValueError("token_span_lists: chunk must be a multiple of 64")
    # the list width: a static bound (a query attends at most its sinks, window pages and top-k), no host sync
    P = -(-k.shape[2] // cfg.page)
    nb = min(P, tq * (cfg.topk + cfg.sink_pages + cfg.window // cfg.page + 1))
    kstats = [None] * g
    if indexer_scores is None and cfg.selector in ("page_mean", "page_maxpool"):
        kstats = [maxpool_windows(k[:, gi:gi + 1], cfg.page) if cfg.selector == "page_maxpool"
                  else page_pool(k[:, gi:gi + 1], cfg.page, "mean") for gi in range(g)]
    parts = []
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        if indexer_scores is None:
            allowed = torch.stack([token_allowed(q[:, gi * rep:(gi + 1) * rep, c0:c1], k[:, gi:gi + 1], cfg,
                                                 q_offset + c0, scale, kstats[gi]) for gi in range(g)])
        else:
            allowed = torch.stack([select_pages(q[:, gi * rep:(gi + 1) * rep, c0:c1], k[:, gi:gi + 1], cfg,
                                                q_offset + c0, scale, indexer_scores[:, c0:c1])[0] for gi in range(g)])
        parts.append(lists_from_allowed(allowed, tq=tq, nb=nb))
    NB = max(p[0].shape[-1] for p in parts)
    return tuple(torch.cat([F.pad(p[j], (0, NB - p[j].shape[-1])) if j < 2 else p[j] for p in parts], 1)
                 for j in range(3))


def _token_packed_attention(spans, cfg: PageSparseConfig, scale: float, q_block: int, selections=None,
                            return_selection: bool = False):
    """:func:`flex_packed_page_attention`'s rowmask kernel for pure per-token selection (select_block 1, no block
    candidates, per_group): every query attends its own pages, in head-packed tiles of :func:`token_tile` queries
    (rowmask_attention.rowmask_head_packed_attention) -- the page union of 8 queries instead of 64 bounds the work.
    A span's selection is its lists (:func:`token_span_lists`)."""
    from .rowmask_attention import BM, rowmask_head_packed_attention

    if q_block != BM:
        raise ValueError(f"rowmask kernel: q_block must be {BM}")
    page = cfg.page
    q0, k0 = spans[0][0], spans[0][1]
    hq, g = q0.shape[1], k0.shape[1]
    tq = token_tile(hq // g, q0.device)
    dev = q0.device
    with torch.no_grad():
        sels, qoff, koff = [], [0], [0]
        for i, (q, k, v, off, idx) in enumerate(spans):
            sel = selections[i] if selections is not None else None
            if sel is None:
                sel = token_span_lists(q, k, cfg, off, scale, idx, tq)
            sels.append(sel)
            qoff.append(qoff[-1] + -(-q.shape[2] // q_block) * q_block)
            koff.append(koff[-1] + -(-k.shape[2] // page) * page)
        Sq, Sk = qoff[-1], koff[-1]
        NB = max(s_[0].shape[-1] for s_ in sels)
        idx_l, bits_l, cnt_l = [], [], []
        q_abs = torch.full((Sq,), -1, dtype=torch.int32, device=dev)
        k_abs = torch.full((Sk,), 2**30, dtype=torch.int32, device=dev)
        for i, (q, k, v, off, _) in enumerate(spans):
            n, L = q.shape[2], k.shape[2]
            si, sb, sc = sels[i]
            nt = (qoff[i + 1] - qoff[i]) // tq  # the span's tiles incl. its padding rows (count 0)
            pad_t = nt - si.shape[1]
            idx_l.append(F.pad(si + koff[i] // page, (0, NB - si.shape[-1], 0, pad_t)))
            bits_l.append(F.pad(sb, (0, NB - sb.shape[-1], 0, pad_t)))
            cnt_l.append(F.pad(sc, (0, pad_t)))
            q_abs[qoff[i]:qoff[i] + n] = torch.arange(off, off + n, device=dev, dtype=torch.int32)
            k_abs[koff[i]:koff[i] + L] = torch.arange(L, device=dev, dtype=torch.int32)
        idx_t = torch.cat(idx_l, 1).contiguous()
        bits_t = torch.cat(bits_l, 1).contiguous()
        cnt_t = torch.cat(cnt_l, 1).contiguous()

    def pad_rows(t, to):  # [1, h, n, d] -> [1, h, to, d]
        return F.pad(t, (0, 0, 0, to - t.shape[2]))

    qp = torch.cat([pad_rows(sp[0], qoff[i + 1] - qoff[i]) for i, sp in enumerate(spans)], 2)
    kp = torch.cat([pad_rows(sp[1], koff[i + 1] - koff[i]) for i, sp in enumerate(spans)], 2)
    vp = torch.cat([pad_rows(sp[2], koff[i + 1] - koff[i]) for i, sp in enumerate(spans)], 2)
    out = rowmask_head_packed_attention(qp[0], kp[0], vp[0], idx_t, bits_t, cnt_t, q_abs, k_abs, scale, tq,
                                        page)[None]
    segs = out.split([qoff[i + 1] - qoff[i] for i in range(len(spans))], dim=2)
    outs = [seg[:, :, :sp[0].shape[2]] for seg, sp in zip(segs, spans)]
    return (outs, sels) if return_selection else outs


def merge_attention(parts):
    """Exact merge of attentions over disjoint key sets: [(out [b, hq, s, d], lse [b, hq, s])] -> out. A part
    whose rows saw no key (lse -inf) contributes nothing."""
    lse = torch.stack([p[1].float() for p in parts])  # [n, b, hq, s]
    m = lse.amax(0).clamp(min=-1e30)
    w = torch.exp(lse - m).nan_to_num(0.0)
    w = w / w.sum(0).clamp(min=1e-30)
    return sum(p[0].float() * w[i][..., None] for i, p in enumerate(parts)).to(parts[0][0].dtype)


def flex_kernel_options(q, q_block: int, kv_block: int) -> dict:
    """Kernel tiles that divide the block mask's blocks (else inductor may pick e.g. a decode kernel's 256-row
    tile for a short chunk); smaller ones on consumer GPUs (~100 KB of shared memory)."""
    smem = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
    tile = 32 if smem < 160 * 1024 and q.shape[-1] >= 128 else 64
    # FORCE_USE_FLEX_ATTENTION: inductor picks its decoding kernel for < 128 queries (short documents / CP chunks),
    # which fails to lower the symbolic tiles of the dynamic compile -- and the eager fallback is wrong for masks
    # that rely on the block lists (LSA's bucket part)
    opts = {"BLOCK_M": min(tile, q_block), "BLOCK_N": min(tile, kv_block), "FORCE_USE_FLEX_ATTENTION": True}
    if tile == 32:
        opts["num_stages"] = 1
    elif q.shape[-1] >= 192:
        # H100, head dim 256 (v4b): 8 warps, 2 stages -- the packed page path's fwd + bwd 1.6x faster than inductor's
        # pick (64k: 176 -> 111 ms, 4 x 16k: 133 -> 83 ms), bit-identical outputs / gradients (flex_tune_bench.py,
        # 2026-11-20); head dim 128 not measured yet
        opts.update(num_warps=8, num_stages=2)
    return opts
