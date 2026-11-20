# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Per-TOKEN page selection for the page selectors (page_mean, page_maxpool, quest, oracle): every query token picks
its own top-k pages (per KV group with ``per_group``), as inference does with OELLM_SPARSE_SELECT=token -- in
training instead of the block-wise selection (one choice per ``select_block`` queries).

The attention runs NSA's selection kernel (flash-linear-attention ``fla.ops.nsa.parallel``): one program per (query
token, KV group) loads exactly the token's pages -- the sink pages, its window's pages and its selected pages --
packed documents as varlen (cu_seqlens). The kernel needs >= 16 query heads per KV group (a power of 2): fewer are
padded with zero heads. Exact: equals ``page_sparse.page_sparse_attention`` with ``select_block=1``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import page_sparse as ps


def token_page_indices(q, k, cfg: ps.PageSparseConfig, q_offset: int, scale: float, chunk: int = 8192):
    """q [1, hq, n, d] (queries at q_offset..), k [1, g, L, d] -> [n, G, S] long page indices per query token (G = g
    with per_group, else 1): the sink pages, the window's pages (the query's page and the wp before it, past the
    sinks), the top-k selected pages; -1 for none. No page twice."""
    _, hq, n, d = q.shape
    g, L = k.shape[1], k.shape[2]
    P = -(-L // cfg.page)
    wp = cfg.window // cfg.page
    dev = q.device
    pos = q_offset + torch.arange(n, device=dev)
    cur = pos // cfg.page  # [n]
    groups = g if (cfg.per_group and g > 1) else 1
    rep = hq // groups
    kb = min(cfg.topk, P)
    sink = torch.arange(cfg.sink_pages, device=dev)[None].expand(n, -1)
    sink = torch.where(sink <= cur[:, None], sink, -1)
    win = cur[:, None] - torch.arange(wp, -1, -1, device=dev)[None]  # [n, wp + 1]: cur - wp .. cur
    win = torch.where(win >= cfg.sink_pages, win, -1)
    out = []
    with torch.no_grad():
        for gi in range(groups):
            qg = q[:, gi * rep:(gi + 1) * rep] if groups > 1 else q
            kg = k[:, gi:gi + 1] if groups > 1 else k
            rows = []
            for c0 in range(0, n, chunk):
                c1 = min(n, c0 + chunk)
                sc = ps.page_scores(qg[:, :, c0:c1], kg, cfg, q_offset + c0, scale, pos=pos[c0:c1])[0]  # [c, P]
                cand = ps.candidate_mask(c1 - c0, P, q_offset + c0, cfg, dev, pos=pos[c0:c1])
                sc = sc.masked_fill(~cand, float("-inf"))
                val, sel = ps.topk_stable(sc, kb)
                rows.append(torch.where(torch.isfinite(val), sel, -1))
            out.append(torch.cat([sink, win, torch.cat(rows, 0)], 1))
    return torch.stack(out, 1)  # [n, G, S]


def _pow2(x: int) -> int:
    return 1 << max(0, x - 1).bit_length()


def token_page_attention_thd(q, k, v, cu_seqlens, cfg: ps.PageSparseConfig, scale: float):
    """Packed documents in THD: q [t, hq, d], k / v [t, g, d], cu_seqlens [N + 1] (each document's queries attend
    its own keys, causal) -> [t, hq, d]. One kernel call for every document."""
    from fla.ops.nsa.parallel import parallel_nsa

    t, hq, d = q.shape
    g = k.shape[1]
    bounds = cu_seqlens.tolist()
    idx = torch.full((t, g, 1), -1, dtype=torch.long, device=q.device)
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b == a:
            continue
        sel = token_page_indices(q[a:b].permute(1, 0, 2)[None], k[a:b].permute(1, 0, 2)[None], cfg, 0, scale)
        if sel.shape[1] == 1 and g > 1:  # one selection for all heads: every KV group gets it
            sel = sel.expand(-1, g, -1)
        pieces.append((a, b, sel))
    S = _pow2(max(p[2].shape[2] for p in pieces))  # the kernel needs a power-of-2 list; -1 entries are skipped
    idx = torch.full((t, g, S), -1, dtype=torch.long, device=q.device)
    for a, b, sel in pieces:
        idx[a:b, :, : sel.shape[2]] = sel
    G = hq // g
    GP = max(16, _pow2(G))
    qp = q.view(t, g, G, d)
    if GP != G:  # zero heads up to the kernel's minimum group size
        qp = F.pad(qp, (0, 0, 0, GP - G))
    o = parallel_nsa(qp.reshape(1, t, g * GP, d), k[None], v[None],
                     g_slc=torch.ones(1, t, g * GP, device=q.device, dtype=q.dtype), block_indices=idx[None],
                     block_counts=S, block_size=cfg.page, scale=scale,
                     cu_seqlens=cu_seqlens.to(torch.long))
    return o[0].view(t, g, GP, -1)[:, :, :G].reshape(t, hq, -1)
