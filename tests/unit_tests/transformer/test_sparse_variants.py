# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""NSA / CSA2 / LSA (experimental_attention_variant/sparse_variants.py) and hierarchical page selection
(page_sparse.py): each is dense attention when its budget covers the context, is causal, gives the same result
for chunked queries, and trains its parameters. Pure torch, CPU or GPU, float64 (exact):

    pytest --noconftest tests/unit_tests/transformer/test_sparse_variants.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_DIR = Path(__file__).resolve().parents[3] / "megatron/core/transformer/experimental_attention_variant"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load("page_sparse")
sv = _load("sparse_variants")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HQ, G, D, H = 4, 2, 16, 24


def qkvx(L=256, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HQ, L, D, generator=gen, dtype=torch.float64).to(DEV)
    k = torch.randn(1, G, L, D, generator=gen, dtype=torch.float64).to(DEV)
    v = torch.randn(1, G, L, D, generator=gen, dtype=torch.float64).to(DEV)
    x = torch.randn(1, L, H, generator=gen, dtype=torch.float64).to(DEV)
    return q, k, v, x


def dense(q, k, v, q_offset=0):
    return ps.dense_attention(q, k, v, q_offset, D**-0.5)


def _csa2(m=1, topk=10_000):
    cfg = sv.CSA2Config(compress=m, topk_entries=topk, window=16, page=16, sink_pages=1, index_heads=2,
                        index_dim=8)
    return sv.CSA2(H, G, D, cfg).to(DEV, torch.float64)


def _lsa(topk=10_000):
    cfg = sv.LSAConfig(pool=2, block=8, buckets=3, iters=3, topk=topk, window=16, page=16, sink_pages=1)
    return sv.LSA(cfg)


def _nsa(gate=(-40.0, 40.0, -40.0), topk=100):
    cfg = sv.NSAConfig(block=8, stride=4, page=16, topk=topk, sink_pages=1, select_window=16, window=32,
                       gate_init=gate)
    return sv.NSA(H, HQ, D, cfg).to(DEV, torch.float64)


def _run(name, mod, q, k, v, x, off=0):
    if name == "csa2":
        return mod(q, k, v, x[:, off:off + q.shape[2]], mod.token_features(x[:, : k.shape[2]]), off, D**-0.5)
    if name == "lsa":
        return mod(q, k, v, off, D**-0.5)
    return mod(q, k, v, x[:, off:off + q.shape[2]], off, D**-0.5)


@pytest.mark.parametrize("name", ["csa2", "lsa", "nsa"])
def test_full_budget_is_dense(name):
    q, k, v, x = qkvx()
    mod = {"csa2": _csa2, "lsa": _lsa, "nsa": _nsa}[name]()
    out = _run(name, mod, q, k, v, x)
    torch.testing.assert_close(out, dense(q, k, v), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("name", ["csa2", "lsa", "nsa"])
def test_causal_and_chunked(name):
    q, k, v, x = qkvx()
    mod = {"csa2": lambda: _csa2(2, 6), "lsa": lambda: _lsa(2), "nsa": lambda: _nsa((0.0, 0.0, 0.0), 2)}[name]()
    full = _run(name, mod, q, k, v, x)
    k2, v2, x2 = k.clone(), v.clone(), x.clone()
    k2[:, :, 200:] += 1.0
    v2[:, :, 200:] -= 1.0
    x2[:, 200:] += 1.0
    changed = _run(name, mod, q, k2, v2, x2)
    torch.testing.assert_close(changed[:, :, :200], full[:, :, :200], atol=1e-10, rtol=1e-10)
    cut = 133
    a = _run(name, mod, q[:, :, :cut], k[:, :, :cut], v[:, :, :cut], x[:, :cut], 0)
    b = _run(name, mod, q[:, :, cut:], k, v, x, cut)
    torch.testing.assert_close(torch.cat([a, b], 2), full, atol=1e-10, rtol=1e-10)


def test_nsa_and_csa2_train():
    q, k, v, x = qkvx()
    nsa = _nsa((0.0, 0.0, 0.0), 2)
    _run("nsa", nsa, q, k, v, x).sum().backward()
    for n in ("ck", "cv", "pos"):
        assert getattr(nsa, n).grad is not None and getattr(nsa, n).grad.abs().sum() > 0, n
    assert nsa.gate.weight.grad.abs().sum() > 0
    csa = _csa2(2, 6)
    _run("csa2", csa, q, k, v, x).sum().backward()
    assert csa.z.weight.grad.abs().sum() > 0  # the compression weights train through the attention
    loss = csa.indexer_loss(q, k, x, csa.token_features(x), 0, D**-0.5)
    loss.backward()
    assert torch.isfinite(loss) and csa.idx_q.weight.grad.abs().sum() > 0 and csa.idx_k.weight.grad.abs().sum() > 0


def test_csa2_compression_starts_as_the_mean():
    _, k, v, x = qkvx()
    csa = _csa2(4)
    kc, vc = csa.compress(k, v, csa.token_features(x))
    torch.testing.assert_close(kc, k.view(1, G, 64, 4, D).mean(3))


def test_lsa_buckets_partition_the_blocks():
    _, k, _, _ = qkvx(L=256)
    cent, mem, ok, last = _lsa().index(k)
    toks = mem[ok]
    assert toks.unique().numel() == toks.numel() == 256  # every token of the 16 complete blocks, once
    assert bool((last[:3] == 15).all()) and cent.shape == (16 * 3, G, D)


@pytest.mark.parametrize("selector", ["page_mean", "quest", "oracle"])
def test_hierarchical(selector):
    q, k, v, _ = qkvx(L=512)
    flat = ps.PageSparseConfig(selector=selector, page=16, topk=4, window=16, sink_pages=1)
    allg = ps.PageSparseConfig(selector=selector, page=16, topk=4, window=16, sink_pages=1, hier_group=4,
                               hier_topk=100)
    torch.testing.assert_close(ps.page_sparse_attention(q, k, v, allg), ps.page_sparse_attention(q, k, v, flat))
    few = ps.PageSparseConfig(selector=selector, page=16, topk=4, window=16, sink_pages=1, hier_group=4,
                              hier_topk=1)
    tok, ok = ps.select_tokens(q, k, few)
    sel = tok[0, -1, 16:16 + 4 * 16][ok[0, -1, 16:16 + 4 * 16]]  # the selection follows the sink page
    groups = (sel // 16 // 4).unique()
    assert groups.numel() == 1  # one group kept: its pages only


def test_csa2_query_chunks():
    """The chunked selection and the chunked (checkpointed) indexer loss do not depend on the chunk size."""
    q, k, v, x = qkvx()
    csa = _csa2(2, 6)
    z = csa.token_features(x)
    torch.testing.assert_close(csa(q, k, v, x, z, 0, D**-0.5, chunk=48), csa(q, k, v, x, z, 0, D**-0.5),
                               atol=1e-12, rtol=1e-12)
    a = csa.indexer_loss(q, k, x, z, 0, D**-0.5, chunk=40)
    b = csa.indexer_loss(q, k, x, z, 0, D**-0.5, chunk=1024)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)  # indexer_kl_loss is fp32


@pytest.mark.parametrize("selector", ["page_mean", "quest", "oracle"])
def test_block_selection(selector):
    """select_block: every query of a block attends the anchor's selection (+ its own sinks / window), causal,
    chunk-invariant at block-aligned cuts, and block 1 is the per-query selection."""
    q, k, v, _ = qkvx(L=512)
    kw = dict(selector=selector, page=16, topk=3, window=16, sink_pages=1)
    flat = ps.page_sparse_attention(q, k, v, ps.PageSparseConfig(**kw))
    torch.testing.assert_close(ps.page_sparse_attention(q, k, v, ps.PageSparseConfig(**kw, select_block=1)), flat)
    cfg = ps.PageSparseConfig(**kw, select_block=32)
    full = ps.page_sparse_attention(q, k, v, cfg)
    tok, ok = ps.select_tokens(q, k, cfg)
    for t in (100, 300, 511):  # the selected part equals the anchor's minus the query's own window pages
        a = (t // 32) * 32
        sel_t = set(tok[0, t, 16:16 + 48][ok[0, t, 16:16 + 48]].tolist())
        sel_a = set(tok[0, a, 16:16 + 48][ok[0, a, 16:16 + 48]].tolist())
        first_window = (t // 16 - 1) * 16
        assert sel_t == {x for x in sel_a if x < first_window}
        all_t = tok[0, t][ok[0, t]]
        assert all_t.unique().numel() == all_t.numel()  # no key twice
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 300:] += 1.0
    v2[:, :, 300:] -= 1.0
    torch.testing.assert_close(ps.page_sparse_attention(q, k2, v2, cfg)[:, :, :300], full[:, :, :300],
                               atol=1e-10, rtol=1e-10)
    a = ps.page_sparse_attention(q[:, :, :192], k[:, :, :192], v[:, :, :192], cfg, 0)
    b = ps.page_sparse_attention(q[:, :, 192:], k, v, cfg, 192)
    torch.testing.assert_close(torch.cat([a, b], 2), full, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
@pytest.mark.parametrize("select_block", [1, 16])
def test_nsa_flex_matches_reference(select_block):
    """NSA's flex path (three FlexAttention branches) equals the reference, forward and backward (fp32)."""
    q, k, v, x = (t.float().clone().requires_grad_() for t in qkvx(L=512))
    cfg = sv.NSAConfig(block=16, stride=8, page=16, topk=3, sink_pages=1, select_window=16, window=32,
                       gate_init=(0.0, 0.0, 0.0), select_block=select_block)
    nsa = sv.NSA(H, HQ, D, cfg).to(DEV)
    for off in (0, 128):
        ref = nsa(q[:, :, off:], k, v, x[:, off:], off, D**-0.5)
        nsa.flex = True
        out = nsa(q[:, :, off:], k, v, x[:, off:], off, D**-0.5)
        nsa.flex = False
        assert float((out - ref).abs().max() / ref.abs().max()) < 1e-2
    g = torch.randn_like(ref)
    gr = torch.autograd.grad((ref * g).sum(), (q, k, v, nsa.ck, nsa.gate.weight))
    nsa.flex = True
    gf = torch.autograd.grad((nsa(q[:, :, 128:], k, v, x[:, 128:], 128, D**-0.5) * g).sum(),
                             (q, k, v, nsa.ck, nsa.gate.weight))
    for a, b in zip(gf, gr):
        assert float((a - b).abs().max() / b.abs().max()) < 2e-2


def test_csa2_block_selection_reference():
    """select_block: every query uses its block's first query's entries; causal; chunk-invariant at aligned cuts;
    the indexer loss is the anchors' KL."""
    q, k, v, x = qkvx()
    cfg = sv.CSA2Config(compress=2, topk_entries=6, window=16, page=16, sink_pages=1, index_heads=2, index_dim=8,
                        select_block=32)
    csa = sv.CSA2(H, G, D, cfg).to(DEV, torch.float64)
    full = _run("csa2", csa, q, k, v, x)
    a = _run("csa2", csa, q[:, :, :128], k[:, :, :128], v[:, :, :128], x[:, :128], 0)
    b = _run("csa2", csa, q[:, :, 128:], k, v, x, 128)
    torch.testing.assert_close(torch.cat([a, b], 2), full, atol=1e-10, rtol=1e-10)
    k2 = k.clone()
    k2[:, :, 200:] += 1.0
    torch.testing.assert_close(_run("csa2", csa, q, k2, v, x)[:, :, :200], full[:, :, :200], atol=1e-10, rtol=1e-10)
    loss = csa.indexer_loss(q, k, x, csa.token_features(x), 0, D**-0.5)
    loss.backward()
    assert torch.isfinite(loss) and csa.idx_q.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
@pytest.mark.parametrize("off", [0, 128, 144])
def test_csa2_flex_matches_reference(off):
    """CSA2's flex path (raw + entries, LSE-merged) equals the block-selection reference, fwd and bwd (fp32);
    off 144 starts the chunk mid-block (the chunk's first query anchors its partial block in both)."""
    q, k, v, x = (t.float().clone().requires_grad_() for t in qkvx(L=512))
    cfg = sv.CSA2Config(compress=2, topk_entries=6, window=32, page=16, sink_pages=1, index_heads=2, index_dim=8,
                        select_block=32, kv_block=16)
    csa = sv.CSA2(H, G, D, cfg).to(DEV)
    torch.nn.init.normal_(csa.z.weight, std=0.3)

    def run():
        return csa(q[:, :, off:], k, v, x[:, off:], csa.token_features(x), off, D**-0.5)

    ref = run()
    csa.flex = True
    out = run()
    assert float((out - ref).abs().max() / ref.abs().max()) < 1e-2
    g = torch.randn_like(ref)
    gf = torch.autograd.grad((out * g).sum(), (q, k, v, csa.z.weight))
    csa.flex = False
    gr = torch.autograd.grad((run() * g).sum(), (q, k, v, csa.z.weight))
    for a, b in zip(gf, gr):
        assert float((a - b).abs().max() / b.abs().max()) < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
@pytest.mark.parametrize("off", [0, 128, 144])
def test_lsa_flex_matches_reference(off):
    """LSA's flex path (keys in bucket order, LSE-merged with the raw part) equals the block-selection
    reference, forward and backward (fp32); the reference is causal and chunk-invariant at aligned cuts."""
    q, k, v, _ = (t.float().clone().requires_grad_() for t in qkvx(L=512))
    lsa = sv.LSA(sv.LSAConfig(pool=2, block=16, buckets=4, iters=3, topk=2, window=32, page=16, sink_pages=1,
                              select_block=32, kv_block=16))

    def run():
        return lsa(q[:, :, off:], k, v, off, D**-0.5)

    ref = run()
    if off == 0:  # float64: the clustering under forced TF32 rounds differently per chunk size
        q64, k64, v64, _ = qkvx(L=512)
        full = lsa(q64, k64, v64, 0, D**-0.5)
        a = lsa(q64[:, :, :128], k64[:, :, :128], v64[:, :, :128], 0, D**-0.5)
        b = lsa(q64[:, :, 128:], k64, v64, 128, D**-0.5)
        torch.testing.assert_close(torch.cat([a, b], 2), full, atol=1e-10, rtol=1e-10)
    lsa.flex = True
    out = run()
    assert float((out - ref).abs().max() / ref.abs().max()) < 1e-2
    g = torch.randn_like(ref)
    gf = torch.autograd.grad((out * g).sum(), (q, k, v))
    lsa.flex = False
    gr = torch.autograd.grad((run() * g).sum(), (q, k, v))
    for x, y in zip(gf, gr):
        assert float((x - y).abs().max() / y.abs().max()) < 2e-2


@pytest.mark.parametrize("selector", ["page_mean", "quest"])
def test_block_selection_prev_max(selector):
    """select_score prev_max: a block's scores are the max over its first query and the previous block's queries
    (exactly their per-query maximum), and the attention stays causal."""
    q, k, v, _ = qkvx(L=512)
    cfg = ps.PageSparseConfig(selector=selector, page=16, topk=3, window=16, sink_pages=1, select_block=32,
                              select_score="prev_max")
    sc = ps.selection_scores(q, k, cfg, 0, D**-0.5)
    flat = ps.selection_scores(q, k, ps.PageSparseConfig(selector=selector, page=16, topk=3, window=16,
                                                         sink_pages=1), 0, D**-0.5)
    t = 300
    a = (t // 32) * 32
    want = torch.maximum(flat[:, a], flat[:, a - 32:a].amax(1))
    torch.testing.assert_close(sc[:, t], want)
    full = ps.page_sparse_attention(q, k, v, cfg)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 300:] += 1.0
    v2[:, :, 300:] -= 1.0
    torch.testing.assert_close(ps.page_sparse_attention(q, k2, v2, cfg)[:, :, :300], full[:, :, :300],
                               atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
@pytest.mark.parametrize("selector", ["page_mean", "quest", "oracle"])
@pytest.mark.parametrize("off", [0, 144])
def test_block_selection_flex(selector, off):
    """Block-wise selection through flex (full blocks without mask_mod: every query of the block attends the
    page) equals the gathered reference, fwd and bwd (fp32), at aligned and mid-block chunk starts."""
    q, k, v, _ = (t.float().clone().requires_grad_() for t in qkvx(L=512))
    cfg = ps.PageSparseConfig(selector=selector, page=16, topk=3, window=32, sink_pages=1, select_block=32)
    a = ps.flex_page_sparse_attention(q[:, :, off:], k, v, cfg, off, D**-0.5, None, 32)
    ref = ps.page_sparse_attention(q[:, :, off:], k, v, cfg, off, D**-0.5)
    assert float((a - ref).abs().max() / ref.abs().max()) < 1e-2
    g = torch.randn_like(a)
    ga = torch.autograd.grad((a * g).sum(), (q, k, v))
    gb = torch.autograd.grad((ref * g).sum(), (q, k, v))
    for x, y in zip(ga, gb):
        assert float((x - y).abs().max() / y.abs().max()) < 2e-2


@pytest.mark.parametrize("selector", ["page_mean", "quest", "oracle"])
def test_per_group_selection(selector):
    """per_group: each KV group's heads attend the pages that group's own queries / keys select -- the shared
    selection run on that group alone; selections can differ between the groups."""
    q, k, v, _ = qkvx(L=512)
    kw = dict(selector=selector, page=16, topk=3, window=16, sink_pages=1, select_block=32)
    out, sels = ps.page_sparse_attention(q, k, v, ps.PageSparseConfig(**kw, per_group=True), return_selection=True)
    rep = HQ // G
    for gi in range(G):
        ref = ps.page_sparse_attention(q[:, gi * rep:(gi + 1) * rep], k[:, gi:gi + 1], v[:, gi:gi + 1],
                                       ps.PageSparseConfig(**kw))
        torch.testing.assert_close(out[:, gi * rep:(gi + 1) * rep], ref, atol=1e-12, rtol=1e-12)
    assert len(sels) == G
    # reusing the returned selections reproduces the output (the shared-index path)
    again = ps.page_sparse_attention(q, k, v, ps.PageSparseConfig(**kw, per_group=True), selection=sels)
    torch.testing.assert_close(again, out, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
@pytest.mark.parametrize("per_group", [False, True])
def test_flex_selection_reuse(per_group):
    """flex: a returned selection reused on other queries / keys (another layer) equals the gathered reference
    given the same tokens; per_group through flex equals the gathered per-group path."""
    q, k, v, _ = (t.float() for t in qkvx(L=512))
    cfg = ps.PageSparseConfig(selector="page_mean", page=16, topk=3, window=32, sink_pages=1, select_block=32,
                              per_group=per_group)
    a, allowed = ps.flex_page_sparse_attention(q, k, v, cfg, 0, D**-0.5, None, 32, return_selection=True)
    ref = ps.page_sparse_attention(q, k, v, cfg)
    assert float((a - ref).abs().max() / ref.abs().max()) < 1e-2
    q2, k2, v2, _ = (t.float() for t in qkvx(L=512, seed=7))  # "the next layer"
    b = ps.flex_page_sparse_attention(q2, k2, v2, cfg, 0, D**-0.5, None, 32, selection=allowed)
    _, sel = ps.page_sparse_attention(q, k, v, cfg, return_selection=True)
    ref2 = ps.page_sparse_attention(q2, k2, v2, cfg, selection=sel)
    assert float((b - ref2).abs().max() / ref2.abs().max()) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
def test_flex_many_lengths_stay_compiled():
    """Packed documents have every length and the callers pass two memory layouts: flex compiles with dynamic
    shapes on contiguous inputs, so distinct lengths, offsets and layouts neither exhaust dynamo's recompile limit
    (-> the eager math fallback) nor change the results."""
    from torch._dynamo.utils import counters

    counters.clear()  # this test's graphs only
    cfg = ps.PageSparseConfig(selector="page_mean", page=16, topk=3, window=32, sink_pages=1, select_block=32)
    for i, (L, off) in enumerate([(130, 0), (257, 0), (300, 44), (512, 0), (333, 100), (190, 0), (450, 64),
                                  (275, 0), (380, 0), (222, 30)]):
        q, k, v, _ = (t.float() for t in qkvx(L=L, seed=L))
        if i % 2:  # the THD layout: permuted views of [L, heads, d] (batch stride heads * d), as in training
            q, k, v = (t[0].permute(1, 0, 2).contiguous().permute(1, 0, 2).unsqueeze(0) for t in (q, k, v))
        a = ps.flex_page_sparse_attention(q[:, :, off:], k, v, cfg, off, D**-0.5, None, 32)
        ref = ps.page_sparse_attention(q[:, :, off:], k, v, cfg, off, D**-0.5)
        assert float((a - ref).abs().max() / ref.abs().max()) < 1e-2, (L, off)
    assert counters["stats"]["unique_graphs"] <= 4, dict(counters["stats"])  # 10 lengths, few graphs
    # a reset of dynamo's config elsewhere (the training stack hit the default 8 in the s3 smoke): every flex call
    # restores the margin before dynamo checks it
    import torch._dynamo

    torch._dynamo.config.recompile_limit = 8
    ps.flex_page_sparse_attention(q, k, v, cfg, 0, D**-0.5, None, 32)
    assert torch._dynamo.config.recompile_limit >= 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
@pytest.mark.parametrize("per_group", [False, True])
@pytest.mark.parametrize("selector", ["page_mean", "quest", "window"])
def test_packed_flex_equals_per_span(selector, per_group):
    """One flex call for a whole packed batch equals one call per span (documents of different lengths, CP-style
    chunks with offsets), forward and backward; its selections, reused, give the same output again."""
    cfg = ps.PageSparseConfig(selector=selector, page=16, topk=3, window=32, sink_pages=1, select_block=32,
                              per_group=per_group)
    lens = [(200, 0, 200), (77, 0, 77), (64, 96, 160), (40, 0, 40), (300, 0, 300)]  # (n, offset, keys)
    spans, leaves = [], []
    for i, (n, off, L) in enumerate(lens):
        q, k, v, _ = (t.float().clone().requires_grad_() for t in qkvx(L=L, seed=10 + i))
        leaves.append((q, k, v))
        spans.append((q[:, :, off:off + n], k, v, off, None))
    outs, sels = ps.flex_packed_page_attention(spans, cfg, D**-0.5, 32, return_selection=True)
    refs = [ps.flex_page_sparse_attention(q, k, v, cfg, off, D**-0.5, None, 32) for q, k, v, off, _ in spans]
    for o, r in zip(outs, refs):
        assert float((o - r).abs().max() / r.abs().max()) < 1e-2
    gs = [torch.randn_like(r) for r in refs]
    ga = torch.autograd.grad(sum((o * g).sum() for o, g in zip(outs, gs)), [t for lv in leaves for t in lv])
    gb = torch.autograd.grad(sum((r * g).sum() for r, g in zip(refs, gs)), [t for lv in leaves for t in lv])
    for x, y in zip(ga, gb):
        assert float((x - y).abs().max() / y.abs().max().clamp(min=1e-12)) < 2e-2
    again = ps.flex_packed_page_attention(spans, cfg, D**-0.5, 32, selections=sels)
    for o, r in zip(again, outs):  # the same selection (kernel variants round differently under forced TF32)
        assert float((o - r).abs().max() / r.abs().max()) < 1e-2


def _dlsa(topk=3, select_block=1):
    cfg = sv.LSAConfig(pool=2, block=8, buckets=3, iters=3, topk=topk, window=16, page=16, sink_pages=1,
                       select_block=select_block, latent=True)
    return sv.LSA(cfg, G, D).to(DEV, torch.float64)


@pytest.mark.parametrize("select_block", [1, 16])
def test_dlsa_identity_init_is_lsa(select_block):
    """D-LSA's maps start as the identity: the same buckets, selection and output as the parameter-free LSA."""
    q, k, v, _ = qkvx(L=256)
    plain = sv.LSA(sv.LSAConfig(pool=2, block=8, buckets=3, iters=3, topk=3, window=16, page=16, sink_pages=1,
                                select_block=select_block))
    lat = _dlsa(select_block=select_block)
    assert lat.proj_k.dim() == 2 and lat.proj_q.dim() == 2  # matrix optimizers
    torch.testing.assert_close(lat(q, k, v, 0, D**-0.5), plain(q, k, v, 0, D**-0.5))


def test_dlsa_distillation_trains_the_maps():
    """The KL to the dense mass per bucket reaches both maps and the scale, and a few steps lower it."""
    q, k, _, _ = qkvx(L=256)
    lat = _dlsa(select_block=16)
    opt = torch.optim.Adam(lat.parameters(), lr=3e-2)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = lat.indexer_loss(q, k, 0, D**-0.5)
        loss.backward()
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in lat.parameters())
        opt.step()
        losses.append(float(loss))
    assert losses[0] > 0 and losses[-1] < 0.8 * losses[0], losses


def test_selector_losses_reach_all_params_without_candidates():
    """Too short for any candidate: the D-LSA / CSA2 losses are zero but still give every selector parameter a
    (zero) gradient -- a rank without one would hang Megatron-FSDP's gradient reduce-scatter."""
    q, k, v, x = qkvx(L=24)
    lat = _dlsa(select_block=16)
    lat.indexer_loss(q, k, 0, D**-0.5).backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0 for p in lat.parameters())
    csa = sv.CSA2(H, G, D, sv.CSA2Config(compress=2, topk_entries=4, window=16, page=16, sink_pages=1)).to(
        DEV, torch.float64)
    loss = csa.indexer_loss(q, k, x, csa.token_features(x), 0, D**-0.5)
    loss.backward()
    assert all(p.grad is not None for p in csa.parameters())
