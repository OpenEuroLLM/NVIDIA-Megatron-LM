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


SELECTORS = ("dense", "window", "page_mean", "quest", "oracle", "token_topk", "compress_mean", "indexer")
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

    def __post_init__(self):
        if self.selector not in SELECTORS + VARIANT_SELECTORS:
            raise ValueError(f"unknown selector {self.selector!r}; one of {SELECTORS + VARIANT_SELECTORS}")
        if self.window % self.page:
            raise ValueError("window must be a multiple of page")

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


def candidate_mask(s: int, P: int, q_offset: int, cfg: PageSparseConfig, device) -> torch.Tensor:
    """[s, P]: the pages a selector may choose for each query (after the sinks, before the window)."""
    _, cur = _query_pages(s, q_offset, cfg, device)
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


def page_scores(q, k, cfg: PageSparseConfig, q_offset: int, scale: float, indexer_scores=None):
    """Selector scores [b, s, P] of every page for every query (higher = more important)."""
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
        return torch.einsum("bgrsd,bgpd->bsp", qg, kp) * scale
    if cfg.selector == "quest":
        hi, lo = page_pool(k, cfg.page, "minmax")
        hi, lo = hi.nan_to_num(0.0), lo.nan_to_num(0.0)
        return (torch.einsum("bgrsd,bgpd->bsp", qg.clamp(min=0), hi)
                + torch.einsum("bgrsd,bgpd->bsp", qg.clamp(max=0), lo)) * scale
    if cfg.selector == "oracle":
        pos, _ = _query_pages(s, q_offset, cfg, q.device)
        out = []
        for c0 in range(0, s, cfg.score_chunk):
            c1 = min(s, c0 + cfg.score_chunk)
            pr = _causal_probs(q[:, :, c0:c1], k, pos[c0:c1], scale)  # [b, hq, sc, L]
            pr = F.pad(pr, (0, P * cfg.page - L)).view(b, hq, c1 - c0, P, cfg.page).sum(-1)
            out.append(pr.mean(1))
        return torch.cat(out, 1)
    raise ValueError(f"page_scores: selector {cfg.selector!r} does not score pages")


def hierarchical_restrict(sc, q, k, cfg: PageSparseConfig, q_offset: int, scale: float):
    """Fine page scores sc [b, s, P] (candidates already masked to -inf) -> the same with every page outside the
    query's ``hier_topk`` best groups of ``hier_group`` pages masked too. Group scores: the selector on
    group-pooled keys (page_mean / quest / oracle), the best page's score otherwise (indexer)."""
    G = cfg.hier_group
    b, s, P = sc.shape
    ng = -(-P // G)
    if cfg.selector in ("page_mean", "quest", "oracle"):
        coarse_cfg = PageSparseConfig(selector=cfg.selector, page=cfg.page * G, topk=cfg.topk, window=0,
                                      sink_pages=0, score_chunk=cfg.score_chunk)
        coarse = page_scores(q, k, coarse_cfg, q_offset, scale)  # [b, s, ng]
        # a group counts only where it has a candidate page
        has = torch.isfinite(F.pad(sc, (0, ng * G - P), value=float("-inf")).view(b, s, ng, G)).any(-1)
        coarse = coarse[..., :ng].masked_fill(~has, float("-inf"))
    else:
        coarse = F.pad(sc, (0, ng * G - P), value=float("-inf")).view(b, s, ng, G).amax(-1)
    val, top = coarse.topk(min(cfg.hier_topk, ng), dim=-1)
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
        cand = candidate_mask(s, P, q_offset, cfg, dev)
        sc = page_scores(q, k, cfg, q_offset, scale, indexer_scores).masked_fill(~cand[None], float("-inf"))
        if cfg.hier_group > 0:
            sc = hierarchical_restrict(sc, q, k, cfg, q_offset, scale)
        val, sel = sc.topk(min(cfg.topk, P), dim=-1)
        sel, _ = torch.sort(torch.where(torch.isfinite(val), sel, P + 1), dim=-1)  # page order: contiguous fetches
        pages.append(sel)
        pages_ok.append(sel < P)
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


def page_sparse_attention(q, k, v, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None,
                          indexer_scores=None):
    """Sparse causal attention of queries q [b, hq, s, d] at absolute positions q_offset .. q_offset + s - 1 over
    keys / values k, v [b, g, L, d] at positions 0 .. L - 1 (one sequence; L >= q_offset + s, GQA: hq % g == 0).
    Returns [b, hq, s, d]."""
    d = q.shape[-1]
    scale = d**-0.5 if scale is None else scale
    if cfg.selector == "dense":
        return dense_attention(q, k, v, q_offset, scale)
    if cfg.selector == "compress_mean":
        return _compress_mean_attention(q, k, v, cfg, q_offset, scale)
    if cfg.selector in VARIANT_SELECTORS:
        raise ValueError(f"{cfg.selector}: a trained module (sparse_variants.py), not a page selector")
    tok, tok_ok = select_tokens(q, k, cfg, q_offset, scale, indexer_scores)
    return attend_gathered(q, k, v, tok, tok_ok, scale, None, cfg.query_chunk)


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
PAGE_SELECTORS = ("window", "page_mean", "quest", "oracle", "indexer")
_FLEX = {}


def _flex():
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


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
    if cfg.selector != "window" and cfg.topk > 0:
        cand = candidate_mask(s, P, q_offset, cfg, dev)
        sc = page_scores(q, k, cfg, q_offset, scale, indexer_scores).masked_fill(~cand[None], float("-inf"))
        if cfg.hier_group > 0:
            sc = hierarchical_restrict(sc, q, k, cfg, q_offset, scale)
        val, sel = sc.topk(min(cfg.topk, P), dim=-1)
        allowed.scatter_(-1, sel, torch.isfinite(val) | allowed.gather(-1, sel))
    return allowed


def flex_page_sparse_attention(q, k, v, cfg: PageSparseConfig, q_offset: int = 0, scale: float | None = None,
                               indexer_scores=None, q_block: int = 128):
    """:func:`page_sparse_attention` for the page selectors through FlexAttention (fused Triton forward and
    backward): the block mask lists, per block of ``q_block`` queries, the union of their pages (built from the
    selection, never by evaluating the dense mask); the per-token mask makes it exact. One batch row (b = 1)."""
    from torch.nn.attention.flex_attention import BlockMask

    b, hq, s, d = q.shape
    g, L = k.shape[1], k.shape[2]
    if b != 1:
        raise ValueError("flex_page_sparse_attention: one sequence (b = 1)")
    scale = d**-0.5 if scale is None else scale
    page = cfg.page
    with torch.no_grad():
        allowed = select_pages(q, k, cfg, q_offset, scale, indexer_scores)[0]  # [s, P]
        P = allowed.shape[1]
        nq = -(-s // q_block)
        blk = F.pad(allowed, (0, 0, 0, nq * q_block - s)).view(nq, q_block, P).any(1)  # [nq, P]
        kv_num = blk.sum(-1).to(torch.int32)
        kv_idx = torch.argsort((~blk).to(torch.int8), dim=-1, stable=True).to(torch.int32)  # selected first
        allowed_i = allowed.to(torch.int8)

    def mask_mod(bi, hi, qi, ki):
        return (ki <= qi + q_offset) & (allowed_i[qi, ki // page] > 0)

    bm = BlockMask.from_kv_blocks(kv_num[None, None], kv_idx[None, None], BLOCK_SIZE=(q_block, page),
                                  mask_mod=mask_mod, seq_lengths=(s, L))
    # kernel tiles that divide the block mask's blocks (else inductor may pick e.g. a decode kernel's 256-row
    # tile for a short chunk); smaller ones on consumer GPUs (~100 KB of shared memory)
    smem = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
    tile = 32 if smem < 160 * 1024 and d >= 128 else 64
    opts = {"BLOCK_M": min(tile, q_block), "BLOCK_N": min(tile, page)}
    if tile == 32:
        opts["num_stages"] = 1
    return _flex()(q, k, v, block_mask=bm, scale=scale, enable_gqa=hq != g, kernel_options=opts)
