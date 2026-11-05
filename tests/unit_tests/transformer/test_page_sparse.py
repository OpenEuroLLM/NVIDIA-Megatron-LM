# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Page-sparse attention (experimental_attention_variant/page_sparse.py): every selector against dense attention
when the budget covers the whole context, the window against an explicit mask, query chunking, gradients against
a masked dense reference, the oracle on a planted key, the indexer loss. CPU or GPU, no distributed setup:

    pytest --noconftest tests/unit_tests/transformer/test_page_sparse.py
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_P = Path(__file__).resolve().parents[3] / "megatron/core/transformer/experimental_attention_variant/page_sparse.py"
_spec = importlib.util.spec_from_file_location("page_sparse", _P)
ps = importlib.util.module_from_spec(_spec)
sys.modules["page_sparse"] = ps  # loaded by path (it has no Megatron imports): the HF export does the same
_spec.loader.exec_module(ps)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def qkv(b=1, hq=8, g=2, L=640, d=32, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hq, L, d, generator=gen).to(DEV, dtype)
    k = torch.randn(b, g, L, d, generator=gen).to(DEV, dtype)
    v = torch.randn(b, g, L, d, generator=gen).to(DEV, dtype)
    return q, k, v


def masked_reference(q, k, v, allowed, scale, bias=None):
    """allowed [b, s, L] bool; q [b, hq, s, d]."""
    b, hq, s, d = q.shape
    g = k.shape[1]
    rep = hq // g
    kk = k.repeat_interleave(rep, 1).float()
    vv = v.repeat_interleave(rep, 1).float()
    logits = torch.einsum("bhsd,bhld->bhsl", q.float(), kk) * scale
    if bias is not None:
        logits = logits + bias[:, None]
    logits = logits.masked_fill(~allowed[:, None], float("-inf"))
    return torch.einsum("bhsl,bhld->bhsd", torch.softmax(logits, -1), vv)


def dense_reference(q, k, v, q_offset=0):
    s, L = q.shape[2], k.shape[2]
    qpos = q_offset + torch.arange(s, device=q.device)
    allowed = torch.arange(L, device=q.device)[None, :] <= qpos[:, None]
    return masked_reference(q, k, v, allowed[None].expand(q.shape[0], -1, -1), q.shape[-1] ** -0.5)


@pytest.mark.parametrize("selector", ["page_mean", "quest", "oracle", "token_topk"])
def test_full_budget_is_dense(selector):
    q, k, v = qkv()
    cfg = ps.PageSparseConfig(selector=selector, page=32, topk=64, window=64, sink_pages=1, query_chunk=100,
                              score_chunk=50)
    out = ps.page_sparse_attention(q, k, v, cfg)
    torch.testing.assert_close(out.float(), dense_reference(q, k, v), atol=2e-5, rtol=2e-5)


def test_compress_one_full_budget_is_dense():
    q, k, v = qkv()
    cfg = ps.PageSparseConfig(selector="compress_mean", page=32, topk=64, window=64, compress=1)
    out = ps.page_sparse_attention(q, k, v, cfg)
    torch.testing.assert_close(out.float(), dense_reference(q, k, v), atol=2e-5, rtol=2e-5)


def test_dense_selector_bottom_right():
    q, k, v = qkv()
    out = ps.page_sparse_attention(q[:, :, -100:], k, v, ps.PageSparseConfig(selector="dense"), q_offset=540)
    torch.testing.assert_close(out.float(), dense_reference(q[:, :, -100:], k, v, 540), atol=2e-5, rtol=2e-5)


def test_window_matches_mask():
    q, k, v = qkv()
    page, wp = 32, 2
    cfg = ps.PageSparseConfig(selector="window", page=page, window=wp * page, sink_pages=1)
    out = ps.page_sparse_attention(q, k, v, cfg)
    L = k.shape[2]
    pos = torch.arange(L, device=DEV)
    kp, qp = pos // page, pos // page
    allowed = (pos[None, :] <= pos[:, None]) & ((kp[None, :] < 1) | (kp[None, :] >= qp[:, None] - wp))
    torch.testing.assert_close(out.float(), masked_reference(q, k, v, allowed[None], q.shape[-1] ** -0.5),
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("selector", ["page_mean", "quest", "oracle", "token_topk", "compress_mean"])
def test_query_chunks_match_full(selector):
    q, k, v = qkv()
    cfg = ps.PageSparseConfig(selector=selector, page=32, topk=4, window=64, sink_pages=1, query_chunk=77)
    full = ps.page_sparse_attention(q, k, v, cfg)
    cut = 333
    a = ps.page_sparse_attention(q[:, :, :cut], k[:, :, :cut], v[:, :, :cut], cfg, q_offset=0)
    b = ps.page_sparse_attention(q[:, :, cut:], k, v, cfg, q_offset=cut)
    torch.testing.assert_close(torch.cat([a, b], 2), full, atol=2e-5, rtol=2e-5)


def test_selection_is_sparse_and_valid():
    q, k, v = qkv(L=1024)
    cfg = ps.PageSparseConfig(selector="page_mean", page=32, topk=3, window=64, sink_pages=1)
    tok, ok = ps.select_tokens(q, k, cfg)
    assert tok.shape[-1] == (1 + 3 + 64 // 32 + 1) * 32
    pos = torch.arange(1024, device=DEV)
    assert bool((tok[ok] >= 0).all()) and bool(((tok <= pos[None, :, None]) | ~ok).all())
    assert int(ok[0, -1].sum()) == (1 + 3 + 2) * 32 + 32  # the last query: its full page, all valid
    # no key twice for one query
    for t in (0, 500, 1023):
        sel = tok[0, t][ok[0, t]]
        assert sel.unique().numel() == sel.numel()


def test_gradients_match_masked_reference():
    q, k, v = qkv(L=384, d=16)
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    cfg = ps.PageSparseConfig(selector="page_mean", page=32, topk=2, window=32, sink_pages=1, query_chunk=64)
    out = ps.page_sparse_attention(q, k, v, cfg)
    with torch.no_grad():
        tok, ok = ps.select_tokens(q, k, cfg)
    # additive: invalid slots point at real (clamped) positions and must not erase valid ones
    allowed = torch.zeros(1, 384, 384, device=DEV).scatter_add_(2, tok, ok.float()) > 0
    ref = masked_reference(q, k, v, allowed, q.shape[-1] ** -0.5)
    g = torch.randn_like(out)
    ga = torch.autograd.grad((out * g).sum(), (q, k, v))
    gb = torch.autograd.grad((ref * g).sum(), (q, k, v))
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=5e-5, rtol=5e-5)


def test_oracle_finds_planted_key():
    q, k, v = qkv(L=640, d=32)
    page = 32
    target_page = 7
    k = k * 0.01
    k[:, :, target_page * page + 5] = q[:, :, -1].view(1, 2, 4, 32).mean(2) * 20  # the last query's best key
    for selector in ("oracle", "page_mean", "quest"):
        cfg = ps.PageSparseConfig(selector=selector, page=page, topk=1, window=64, sink_pages=1)
        tok, ok = ps.select_tokens(q, k, cfg)
        sel = tok[0, -1, page: 2 * page]  # the selection follows the sink page
        assert int(sel[0]) == target_page * page, selector


def test_indexer_and_kl():
    torch.manual_seed(0)
    idx = ps.PageIndexer(hidden=48, heads=2, dim=16, page=32).to(DEV)
    x = torch.randn(1, 640, 48, device=DEV)
    pk = idx.page_keys(x)
    sc = idx(x, pk)
    assert sc.shape == (1, 640, 20)
    cand = ps.candidate_mask(640, 20, 0, ps.PageSparseConfig(page=32, window=64), DEV)
    loss = ps.indexer_kl_loss(sc, torch.rand_like(sc), cand)
    loss.backward()
    assert torch.isfinite(loss) and idx.wq.weight.grad is not None
    # zero at the target: scores = log(target) on the candidates
    t = torch.rand_like(sc) + 0.1
    assert float(ps.indexer_kl_loss(torch.log(t), t, cand)) < 1e-5
    # an indexer drives the selection
    q, k, v = qkv(L=640)
    cfg = ps.PageSparseConfig(selector="indexer", page=32, topk=2, window=64)
    out = ps.page_sparse_attention(q, k, v, cfg, indexer_scores=sc.detach())
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_budget():
    cfg = ps.PageSparseConfig(selector="page_mean", page=64, topk=32, window=512, sink_pages=1)
    assert cfg.budget == 64 + 512 + 64 + 2048
    assert math.isclose(ps.PageSparseConfig(selector="window").budget, 64 + 512 + 64)
