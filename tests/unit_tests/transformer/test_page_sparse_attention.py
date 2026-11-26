# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""PageSparseCoreAttention (page_sparse_selector): the TE core-attention replacement on packed (THD) batches --
a full budget is dense attention per document; CP=2 equals CP=1 (outputs, query / key / value gradients);
the indexer gets gradients from its KL warm-up loss, and the warm-up keeps the dense output; the trained variants
(nsa / csa2 / lsa, sparse_variants.py) and hierarchical selection are their per-document reference. 2 GPUs:

    python -m torch.distributed.run --nproc-per-node 2 -m pytest -q tests/unit_tests/transformer/test_page_sparse_attention.py
"""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import fla_cp
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.experimental_attention_variant import page_sparse as ps
from megatron.core.transformer.experimental_attention_variant import sparse_variants as sv
from megatron.core.transformer.experimental_attention_variant.page_sparse_attention import (
    PageSparseCoreAttention,
)
from tests.unit_tests.test_utilities import Utils

HQ, G, D, H = 4, 2, 16, 64
LENGTHS = (120, 260, 132)  # multiples of 4 (2 cp); the middle document spans the stream pieces

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPUs")

# NGC containers force TF32 for every fp32 cuBLAS GEMM (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1): fp32 page scores then differ
# from a float64 reference beyond near-ties, so the per-token selection tests turn the Triton scoring's TF32 off (it
# reads the variable per call) and compare with float64 references; paths scored by cuBLAS fp32 GEMMs are skipped there.
TF32_FORCED = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1"


@pytest.fixture
def exact_scores(monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)



def _config(**kw):
    return TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=HQ, num_query_groups=G,
                             kv_channels=D, **kw)


def _packed(lengths):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                           cu_seqlens_kv_padded=cu, max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))


def _inputs(seed=0, dtype=torch.float64):
    """float64 for the exact comparisons (the NGC container forces TF32 for fp32 GEMMs)."""
    g = torch.Generator().manual_seed(seed)
    t = sum(LENGTHS)
    q, k, v = (torch.randn(t, n, D, generator=g, dtype=torch.float64).cuda().to(dtype) for n in (HQ, G, G))
    x = torch.randn(t, 1, H, generator=g).cuda()
    return q, k, v, x


@pytest.fixture
def single():
    Utils.initialize_model_parallel(1, 1)
    yield ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                 cp=parallel_state.get_context_parallel_group())
    Utils.destroy_model_parallel()


def test_full_budget_is_dense_per_document(single):
    q, k, v, _ = _inputs()
    att = PageSparseCoreAttention(_config(page_sparse_selector="page_mean", page_sparse_page=32,
                                          page_sparse_topk=64, page_sparse_window=64, page_sparse_backend="gathered"),
                                  1, pg_collection=single)
    out = att(q, k, v, packed_seq_params=_packed(LENGTHS))
    ref, a = [], 0
    for n in LENGTHS:
        qd, kd, vd = (x[a:a + n].permute(1, 0, 2).unsqueeze(0) for x in (q, k, v))
        ref.append(ps.dense_attention(qd, kd, vd, 0, D**-0.5)[0].permute(1, 0, 2).reshape(n, HQ * D))
        a += n
    torch.testing.assert_close(out, torch.cat(ref), atol=1e-4, rtol=1e-4)


def test_indexer_warmup_loss_and_dense_forward(single):
    q, k, v, x = _inputs(1, torch.float32)
    cfg = _config(page_sparse_selector="indexer", page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32,
                  page_sparse_indexer_heads=2, page_sparse_indexer_dim=8, page_sparse_indexer_loss_coeff=1.0,
                  page_sparse_dense_forward=True)
    att = PageSparseCoreAttention(cfg, 1, pg_collection=single).cuda()
    att.train()
    att.set_hidden(x)
    out = att(q, k, v, packed_seq_params=_packed(LENGTHS))
    dense = PageSparseCoreAttention(_config(page_sparse_selector="dense"), 1, pg_collection=single)
    torch.testing.assert_close(out, dense(q, k, v, packed_seq_params=_packed(LENGTHS)), atol=1e-4, rtol=1e-4)
    out.sum().backward()  # the KL rides on the output (MoEAuxLossAutoScaler)
    assert att.indexer.wq.weight.grad is not None and att.indexer.wq.weight.grad.abs().sum() > 0


VARIANTS = {  # small budgets: real selections in every document
    "nsa": dict(page_sparse_nsa_block=16, page_sparse_nsa_stride=8, page_sparse_nsa_select_window=32),
    "csa2": dict(page_sparse_compress=2, page_sparse_csa2_topk_entries=12, page_sparse_indexer_heads=2,
                 page_sparse_indexer_dim=8),
    "lsa": dict(page_sparse_lsa_pool=2, page_sparse_lsa_block=16, page_sparse_lsa_buckets=4),
    "hier": dict(page_sparse_hier_group=2, page_sparse_hier_topk=1),
}


def _variant_kw(name):
    sel = "quest" if name == "hier" else name
    return dict(page_sparse_selector=sel, page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32,
                page_sparse_backend="gathered", **VARIANTS.get(name, {}))


def _reference(att, q, k, v, x):
    """The variant / page selection called directly on each document (float64)."""
    outs, a = [], 0
    for n in LENGTHS:
        qd, kd, vd = (t[a:a + n].permute(1, 0, 2).unsqueeze(0) for t in (q, k, v))
        xd = x[a:a + n, 0].unsqueeze(0)
        sel = att.cfg.selector
        if sel == "nsa":
            o = att.variant(qd, kd, vd, xd, 0, D**-0.5)
        elif sel == "csa2":
            o = att.variant(qd, kd, vd, xd, att.variant.token_features(xd), 0, D**-0.5)
        elif sel == "lsa":
            o = att.variant(qd, kd, vd, 0, D**-0.5)
        else:
            o = ps.page_sparse_attention(qd, kd, vd, att.cfg, 0, D**-0.5)
        outs.append(o[0].permute(1, 0, 2).reshape(n, HQ * D))
        a += n
    return torch.cat(outs)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_thd_matches_per_document(single, name):
    q, k, v, x = _inputs(3)
    x = x.double()
    att = PageSparseCoreAttention(_config(**_variant_kw(name)), 1, pg_collection=single).cuda().double()
    att.set_hidden(x)
    out = att(q, k, v, packed_seq_params=_packed(LENGTHS))
    torch.testing.assert_close(out, _reference(att, q, k, v, x), atol=1e-10, rtol=1e-10)


def test_shared_selection_across_layers(single):
    """page_sparse_share_source: the source layer's selection (per document) is what a later layer attends with;
    a consumer before its source on a micro-batch is an error."""
    from megatron.core.transformer.experimental_attention_variant import page_sparse_attention as psa

    kw = dict(page_sparse_selector="page_mean", page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32,
              page_sparse_backend="gathered", page_sparse_share_source=1)
    src = PageSparseCoreAttention(_config(**kw), 1, pg_collection=single)
    dst = PageSparseCoreAttention(_config(**kw), 3, pg_collection=single)
    psa._SHARED_SELECTION.clear()
    q1, k1, v1, _ = _inputs(6)
    q2, k2, v2, _ = _inputs(7)
    with pytest.raises(RuntimeError):
        dst(q2, k2, v2, packed_seq_params=_packed(LENGTHS))
    src(q1, k1, v1, packed_seq_params=_packed(LENGTHS))
    out = dst(q2, k2, v2, packed_seq_params=_packed(LENGTHS))
    ref, a = [], 0
    for n in LENGTHS:
        d1 = [t[a:a + n].permute(1, 0, 2).unsqueeze(0) for t in (q1, k1, v1)]
        d2 = [t[a:a + n].permute(1, 0, 2).unsqueeze(0) for t in (q2, k2, v2)]
        _, sel = ps.page_sparse_attention(*d1, src.cfg, 0, D**-0.5, return_selection=True)
        o = ps.page_sparse_attention(*d2, src.cfg, 0, D**-0.5, selection=sel)
        ref.append(o[0].permute(1, 0, 2).reshape(n, HQ * D))
        a += n
    torch.testing.assert_close(out, torch.cat(ref), atol=1e-10, rtol=1e-10)
    psa._SHARED_SELECTION.clear()


@pytest.mark.parametrize("extra", [{}, {"page_sparse_per_group": True}, {"page_sparse_share_source": 1}],
                         ids=["shared_heads", "per_group", "shared_layers"])
def test_packed_flex_thd_matches_gathered(single, extra):
    """The flex backend on a packed batch (one flex call for every document) equals the gathered reference, forward
    and backward (fp32); with page_sparse_share_source, a later layer reuses the source's selections."""
    from megatron.core.transformer.experimental_attention_variant import page_sparse_attention as psa

    kw = dict(page_sparse_selector="page_mean", page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32,
              page_sparse_select_block=32, page_sparse_flex_q_block=32, page_sparse_packed_flex=True, **extra)
    flex = PageSparseCoreAttention(_config(**kw, page_sparse_backend="flex"), 1, pg_collection=single)
    ref = PageSparseCoreAttention(_config(**kw, page_sparse_backend="gathered"), 1, pg_collection=single)
    psa._SHARED_SELECTION.clear()
    q, k, v, _ = _inputs(8, torch.float32)
    dout = torch.randn(sum(LENGTHS), HQ * D, generator=torch.Generator().manual_seed(9)).cuda()
    leaves_a = [x.clone().requires_grad_() for x in (q, k, v)]
    leaves_b = [x.clone().requires_grad_() for x in (q, k, v)]
    out_a = flex(*leaves_a, packed_seq_params=_packed(LENGTHS))
    psa._SHARED_SELECTION.clear()
    out_b = ref(*leaves_b, packed_seq_params=_packed(LENGTHS))
    assert float((out_a - out_b).abs().max() / out_b.abs().max()) < 1e-2
    (out_a * dout).sum().backward()
    (out_b * dout).sum().backward()
    for a, b in zip(leaves_a, leaves_b):
        assert float((a.grad - b.grad).abs().max() / b.grad.abs().max()) < 2e-2
    if extra.get("page_sparse_share_source"):  # a consumer layer after the flex source
        psa._SHARED_SELECTION.clear()
        flex(q, k, v, packed_seq_params=_packed(LENGTHS))
        dst = PageSparseCoreAttention(_config(**kw, page_sparse_backend="flex"), 3, pg_collection=single)
        q2, k2, v2, _ = _inputs(9, torch.float32)
        out = dst(q2, k2, v2, packed_seq_params=_packed(LENGTHS))
        assert torch.isfinite(out).all()
    psa._SHARED_SELECTION.clear()


def test_csa2_indexer_loss_rides_on_the_output(single):
    q, k, v, x = _inputs(4)
    x = x.double()
    cfg = _config(**_variant_kw("csa2"), page_sparse_indexer_loss_coeff=1.0)
    att = PageSparseCoreAttention(cfg, 1, pg_collection=single).cuda().double()
    att.train()
    att.set_hidden(x)
    att(q, k, v, packed_seq_params=_packed(LENGTHS)).sum().backward()
    assert att.variant.idx_q.weight.grad.abs().sum() > 0 and att.variant.idx_k.weight.grad.abs().sum() > 0


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_cp_matches_single_rank(name):
    """CP=2 vs CP=1 for the trained variants: outputs, q / k / v gradients and the (CP-summed) parameter
    gradients (nsa's gates and compressors, csa2's compression weights whose key-side features are gathered)."""
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        tp = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        solo = [torch.distributed.new_group([r]) for r in range(torch.distributed.get_world_size())]
        kw = dict(_variant_kw(name), context_parallel_size=2)
        torch.manual_seed(7)
        att = PageSparseCoreAttention(_config(**kw), 1, pg_collection=ProcessGroupCollection(tp=tp, cp=cp_group))
        ref = PageSparseCoreAttention(_config(**kw), 1, pg_collection=ProcessGroupCollection(
            tp=tp, cp=solo[torch.distributed.get_rank()]))
        for p in att.parameters():  # random, not the identity init: every parameter matters
            torch.nn.init.normal_(p, std=0.3)
        ref.load_state_dict(att.state_dict())
        att, ref = att.cuda().double(), ref.cuda().double()
        params = _packed(LENGTHS)
        st = fla_cp.thd_stream(params.cu_seqlens_q_padded, cp_group)
        q, k, v, x = _inputs(5)
        x = x.double()
        dout = torch.randn(sum(LENGTHS), HQ * D, generator=torch.Generator().manual_seed(6),
                           dtype=torch.float64).cuda()
        loc = [t[st.local_idx].clone().requires_grad_(True) for t in (q, k, v)]
        att.set_hidden(x[st.local_idx])
        out = att(*loc, packed_seq_params=params)
        (out * dout[st.local_idx]).sum().backward()
        full = [t.clone().requires_grad_(True) for t in (q, k, v)]
        ref.set_hidden(x)
        out_ref = ref(*full, packed_seq_params=params)
        (out_ref * dout).sum().backward()
        torch.testing.assert_close(out, out_ref[st.local_idx], atol=1e-8, rtol=1e-8)
        for i in range(3):
            torch.testing.assert_close(loc[i].grad, full[i].grad[st.local_idx], atol=1e-8, rtol=1e-8)
        for (n, p), pr in zip(att.named_parameters(), ref.parameters()):
            if pr.grad is None:  # e.g. csa2's indexer without its loss (the selection carries no gradient)
                assert p.grad is None, n
                continue
            g = p.grad.clone()
            torch.distributed.all_reduce(g, group=cp_group)
            torch.testing.assert_close(g, pr.grad, atol=1e-8, rtol=1e-8, msg=lambda m: f"{n}: {m}")
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("selector", ["page_mean", "quest", "compress_mean", "dense"])
def test_cp_matches_single_rank(selector):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        tp = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        solo = [torch.distributed.new_group([r]) for r in range(torch.distributed.get_world_size())]
        kw = dict(page_sparse_selector=selector, page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32,
                  context_parallel_size=2, page_sparse_backend="gathered")  # float64: the exact reference
        att = PageSparseCoreAttention(_config(**kw), 1, pg_collection=ProcessGroupCollection(tp=tp, cp=cp_group))
        ref = PageSparseCoreAttention(_config(**kw), 1, pg_collection=ProcessGroupCollection(
            tp=tp, cp=solo[torch.distributed.get_rank()]))
        params = _packed(LENGTHS)
        st = fla_cp.thd_stream(params.cu_seqlens_q_padded, cp_group)
        q, k, v, _ = _inputs(2)
        dout = torch.randn(sum(LENGTHS), HQ * D, generator=torch.Generator().manual_seed(5),
                           dtype=torch.float64).cuda()
        loc = [x[st.local_idx].clone().requires_grad_(True) for x in (q, k, v)]
        out = att(*loc, packed_seq_params=params)
        (out * dout[st.local_idx]).sum().backward()
        full = [x.clone().requires_grad_(True) for x in (q, k, v)]
        out_ref = ref(*full, packed_seq_params=params)
        (out_ref * dout).sum().backward()
        torch.testing.assert_close(out, out_ref[st.local_idx], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(loc[0].grad, full[0].grad[st.local_idx], atol=1e-4, rtol=1e-4)
        # key / value gradients: each rank's local share of the reduce-scattered gradient
        for i in (1, 2):
            torch.testing.assert_close(loc[i].grad, full[i].grad[st.local_idx], atol=1e-4, rtol=1e-4)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("name", ["nsa", "csa2", "lsa_latent", "indexer"])
def test_selector_adam_tags_only_the_selector(single, name):
    """page_sparse_selector_adam: every selector parameter is tagged for Adam (not the matrix optimizer); off by
    default."""
    kw = (dict(page_sparse_selector="lsa", page_sparse_lsa_latent=True) if name == "lsa_latent"
          else dict(page_sparse_selector="indexer") if name == "indexer" else _variant_kw(name))
    off = PageSparseCoreAttention(_config(**kw), 1, pg_collection=single)
    on = PageSparseCoreAttention(_config(**kw, page_sparse_selector_adam=True), 1, pg_collection=single)
    params = list(on.parameters())
    assert params and all(getattr(p, "use_adam_with_emerging_optimizers", False) for p in params)
    assert not any(getattr(p, "use_adam_with_emerging_optimizers", False) for p in off.parameters())


FLEX_CP_ARMS = {  # the t16 arms on the flex backend (the training path), small pages / budgets
    "page_mean_pg": dict(page_sparse_selector="page_mean", page_sparse_per_group=True, page_sparse_packed_flex=True),
    "page_maxpool_pg": dict(page_sparse_selector="page_maxpool", page_sparse_per_group=True,
                            page_sparse_packed_flex=True),
    "nsa": dict(page_sparse_selector="nsa", **VARIANTS["nsa"]),
    "csa2_fix": dict(page_sparse_selector="csa2", page_sparse_csa2_entry_bias=1.3862944,
                     page_sparse_csa2_entry_rms=True, **VARIANTS["csa2"]),
    "lsa_fix": dict(page_sparse_selector="lsa", page_sparse_lsa_visibility="block", **VARIANTS["lsa"]),
}
# "aligned": every CP chunk (length / (2 cp)) a multiple of select_block 32 up to 4 ranks; "straddling": chunks
# start inside select blocks (and at CP 4 are shorter than one), so blocks need their first queries from other
# ranks (PageSparseCoreAttention._cp_block_heads)
CP_LENGTHS = {"aligned": (256, 512, 768), "straddling": (120, 264, 136)}


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("lengths", list(CP_LENGTHS))
@pytest.mark.parametrize("name", list(FLEX_CP_ARMS))
def test_flex_cp_matches_single_rank(name, lengths):
    """CP = world size (2 or 4; intra-node 4 is the 256k layout) vs one rank, both on the flex backend: outputs,
    q / k / v gradients and the CP-summed parameter gradients (fp32; the flex kernels have no float64), also
    when select blocks straddle the CP chunk boundaries."""
    cp_size = Utils.world_size
    Utils.initialize_model_parallel(1, 1, context_parallel_size=cp_size)
    try:
        tp = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        solo = [torch.distributed.new_group([r]) for r in range(torch.distributed.get_world_size())]
        kw = dict(page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32, page_sparse_select_block=32,
                  page_sparse_flex_q_block=32, page_sparse_backend="flex", context_parallel_size=cp_size,
                  **FLEX_CP_ARMS[name])
        torch.manual_seed(7)
        att = PageSparseCoreAttention(_config(**kw), 1, pg_collection=ProcessGroupCollection(tp=tp, cp=cp_group))
        ref = PageSparseCoreAttention(_config(**kw), 1, pg_collection=ProcessGroupCollection(
            tp=tp, cp=solo[torch.distributed.get_rank()]))
        for p in att.parameters():
            torch.nn.init.normal_(p, std=0.3)
        ref.load_state_dict(att.state_dict())
        att, ref = att.cuda(), ref.cuda()
        params = _packed(CP_LENGTHS[lengths])
        st = fla_cp.thd_stream(params.cu_seqlens_q_padded, cp_group)
        g = torch.Generator().manual_seed(11)
        t = sum(CP_LENGTHS[lengths])
        q, k, v = (torch.randn(t, n, D, generator=g).cuda() for n in (HQ, G, G))
        x = torch.randn(t, 1, H, generator=g).cuda()
        dout = torch.randn(t, HQ * D, generator=g).cuda()
        loc = [y[st.local_idx].clone().requires_grad_(True) for y in (q, k, v)]
        att.set_hidden(x[st.local_idx])
        out = att(*loc, packed_seq_params=params)
        (out * dout[st.local_idx]).sum().backward()
        full = [y.clone().requires_grad_(True) for y in (q, k, v)]
        ref.set_hidden(x)
        out_ref = ref(*full, packed_seq_params=params)
        (out_ref * dout).sum().backward()

        def close(a, b, what):
            # 5e-3: the NGC container forces TF32 for fp32 GEMMs and CP sums in another order (lsa_fix straddling
            # dq 2.6e-3 on H100 at CP 4); a selection that differs from one rank's gives 1e-2 .. 1
            err = float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))
            assert err < 5e-3, f"{name} {lengths} cp{cp_size} {what}: rel err {err:.2e}"

        close(out, out_ref[st.local_idx], "output")
        for i, w in enumerate("qkv"):
            close(loc[i].grad, full[i].grad[st.local_idx], f"d{w}")
        for (n, p), pr in zip(att.named_parameters(), ref.parameters()):
            if pr.grad is None:
                assert p.grad is None, n
                continue
            gp = p.grad.clone()
            torch.distributed.all_reduce(gp, group=cp_group)
            close(gp, pr.grad, n)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("L", [256, 300, 61])
def test_page_maxpool_scores_match_cumsum_reference(L):
    """page_maxpool's window means (quarter-page sums, 2026-11-20) equal the original cumsum formulation: half-page
    windows at quarter-page stride, a page scored by the max over the 5 windows overlapping it (float64)."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 37, 16, dtype=torch.float64), torch.randn(1, 2, L, 16, dtype=torch.float64)
    cfg = ps.PageSparseConfig(selector="page_maxpool", page=16, topk=2, window=16)
    P = -(-L // 16)
    st, sub, J = 4, 8, 4 * P + 1
    cs = F.pad(F.pad(k, (0, 0, 0, max(0, (J - 1) * st + sub - L))).cumsum(2), (0, 0, 1, 0))
    starts = torch.arange(J) * st
    win = (cs[:, :, starts + sub] - cs[:, :, starts]) / sub
    sc = torch.einsum("bgrsd,bgjd->bsj", q.view(1, 2, 2, 37, 16), win) * 0.25
    ref = F.max_pool1d(sc.reshape(37, 1, J), kernel_size=5, stride=4, padding=1).view(1, 37, -1)[..., :P]
    torch.testing.assert_close(ps.page_scores(q, k, cfg, 0, 0.25), ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool", "quest"])
@pytest.mark.parametrize("block", [8, 16, 32])
@pytest.mark.parametrize("L,off", [(64, 0), (100, 0), (128, 40), (144, 0), (160, 16), (300, 0)])
def test_short_sequence_shortcut_equals_scored_selection(selector, block, L, off, monkeypatch):
    """select_pages' shortcut for sequences with at most topk selectable pages equals the scored selection (page 16,
    window 16, 1 sink, top 6); with select_block 32 > page it must not apply (anchors' windows differ)."""
    torch.manual_seed(L)
    q, k = torch.randn(1, 4, L - off, 16), torch.randn(1, 2, L, 16)
    cfg = ps.PageSparseConfig(selector=selector, page=16, topk=6, window=16, sink_pages=1, select_block=block)
    got = ps.select_pages(q, k, cfg, off, 0.25)
    monkeypatch.setattr(ps, "_selects_everything", lambda P, cfg: False)
    torch.testing.assert_close(got, ps.select_pages(q, k, cfg, off, 0.25))


@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_token_backend_thd_matches_gathered(single, selector):
    """page_sparse_backend token (per-token selection, NSA kernel over the packed documents) equals the gathered
    reference with select_block 1 on a packed batch, forward and backward."""
    pytest.importorskip("fla.ops.nsa.parallel")
    kw = dict(page_sparse_selector=selector, page_sparse_page=32, page_sparse_topk=2, page_sparse_window=32,
              page_sparse_select_block=1, page_sparse_per_group=True)
    tok = PageSparseCoreAttention(_config(**kw, page_sparse_backend="token"), 1, pg_collection=single)
    ref = PageSparseCoreAttention(_config(**kw, page_sparse_backend="gathered"), 1, pg_collection=single)
    q, k, v, _ = _inputs(8, torch.bfloat16)
    dout = torch.randn(sum(LENGTHS), HQ * D, generator=torch.Generator().manual_seed(9)).cuda()
    leaves_a = [x.clone().requires_grad_() for x in (q, k, v)]
    leaves_b = [x.clone().double().requires_grad_() for x in (q, k, v)]
    out_a = tok(*leaves_a, packed_seq_params=_packed(LENGTHS))
    out_b = ref(*leaves_b, packed_seq_params=_packed(LENGTHS))
    assert float((out_a.double() - out_b).abs().max() / out_b.abs().max()) < 2e-2
    (out_a.float() * dout).sum().backward()
    (out_b * dout.double()).sum().backward()
    for a, b in zip(leaves_a, leaves_b):
        assert float((a.grad.double() - b.grad).abs().max() / b.grad.abs().max()) < 3e-2


@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
@pytest.mark.parametrize("off", [0, 64])
def test_block_candidates_per_token_selection(selector, off):
    """Per-token selection inside block candidates: with candidates >= pages it is plain per-token selection; with U
    candidates every block's selected pages (beyond sinks / windows) are at most U -- its first query's best U --
    and each query's choice lies inside them; invariant to chunks that start on block boundaries."""
    torch.manual_seed(off)
    L = 640
    q, k = torch.randn(1, 4, L - off, 16), torch.randn(1, 2, L, 16)
    base = dict(selector=selector, page=16, topk=4, window=16, sink_pages=1, select_block=1)
    plain = ps.select_pages(q, k, ps.PageSparseConfig(**base), off, 0.25)[0]
    wide = ps.select_pages(q, k, ps.PageSparseConfig(**base, block_candidates=64, candidate_block=32), off, 0.25)[0]
    torch.testing.assert_close(wide, plain)
    cfg = ps.PageSparseConfig(**base, block_candidates=6, candidate_block=32)
    got = ps.select_pages(q, k, cfg, off, 0.25)[0]  # [s, P]
    pos = off + torch.arange(L - off)
    cur = pos // 16
    p = torch.arange(40)
    fixed = (p[None] < 1) | ((p[None] >= cur[:, None] - 1) & (p[None] <= cur[:, None]))
    sel = got & ~fixed
    blk = pos // 32
    for bi in blk.unique():
        assert int(sel[blk == bi].any(0).sum()) <= 6
    h = 320 - off  # chunked on a block boundary (as CP chunks): the second part alone selects as in the whole call
    torch.testing.assert_close(ps.select_pages(q[:, :, h:], k, cfg, off + h, 0.25)[0], got[h:])


@pytest.mark.parametrize("mode", ["block64", "block16", "token", "token_candidates"])
@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_selection_is_causal(mode, selector):
    """No query's pages depend on anything after it: changing every later query and the keys of every later page
    leaves the selections of queries <= t unchanged, in every selection mode (the voted block candidates of
    2026-11-20 failed this: later queries chose the earlier ones' pages)."""
    torch.manual_seed(5)
    L, page = 640, 16
    q, k = torch.randn(1, 4, L, 16), torch.randn(1, 2, L, 16)
    kw = dict(selector=selector, page=page, topk=4, window=16, sink_pages=1, per_group=False)
    cfg = {"block64": ps.PageSparseConfig(**kw, select_block=64), "block16": ps.PageSparseConfig(**kw, select_block=16),
           "token": ps.PageSparseConfig(**kw, select_block=1),
           "token_candidates": ps.PageSparseConfig(**kw, select_block=1, block_candidates=6, candidate_block=64)}[mode]
    base = ps.select_pages(q, k, cfg, 0, 0.25)[0]
    for t in (100, 333, 500):
        q2, k2 = q.clone(), k.clone()
        q2[:, :, t + 1:] = torch.randn_like(q2[:, :, t + 1:])
        k2[:, :, (t // page + 1) * page:] = torch.randn_like(k2[:, :, (t // page + 1) * page:])
        torch.testing.assert_close(ps.select_pages(q2, k2, cfg, 0, 0.25)[0][: t + 1], base[: t + 1])


@pytest.mark.parametrize("hq", [16, 10, 4])
@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_rowmask_pure_token_packed_matches_gathered(hq, selector, exact_scores):
    """Pure per-token selection (select_block 1, no block candidates, per KV group) through the rowmask backend's
    head-packed tiles (8 queries x 8 heads at 16 / 2; padded heads at 10 / 2; 32 queries x 2 heads at 4 / 2) equals
    the gathered reference per span on packed documents and an offset (CP-like) chunk, outputs and q / k / v
    gradients; and it is causal: outputs up to t do not change when every later query / key / value does."""
    if selector == "page_mean" and TF32_FORCED:
        pytest.skip("page_mean scores through cuBLAS fp32 GEMMs, forced to TF32 here")
    torch.manual_seed(2)
    g, d = 2, 32
    cfg = ps.PageSparseConfig(selector=selector, page=64, topk=3, window=128, sink_pages=1, select_block=1,
                              per_group=True)
    spans = []
    for L, off in [(1300, 0), (260, 0), (900, 320)]:
        q = torch.randn(1, hq, L - off, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        spans.append((q, k, v, off, None))
    leaves = [[t.clone().requires_grad_() for t in sp[:3]] for sp in spans]
    outs = ps.flex_packed_page_attention([(*lv, sp[3], None) for lv, sp in zip(leaves, spans)], cfg, d**-0.5, 64,
                                         kernel="rowmask")
    gos = [torch.randn(o.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(i))
           for i, o in enumerate(outs)]
    sum((o.float() * go).sum() for o, go in zip(outs, gos)).backward()
    for (q, k, v, off, _), lv, o, go in zip(spans, leaves, outs, gos):
        qd, kd, vd = (t.detach().double().requires_grad_() for t in (q, k, v))
        ref = ps.page_sparse_attention(qd, kd, vd, cfg, off, d**-0.5)
        assert float((o.double() - ref).abs().max() / ref.abs().max()) < 2e-2
        (ref * go.double()).sum().backward()
        for a, b in zip(lv, (qd, kd, vd)):
            assert float((a.grad.double() - b.grad).abs().max() / b.grad.abs().max()) < 3e-2
    q, k, v, off, _ = spans[0]
    base = ps.flex_packed_page_attention([(q, k, v, 0, None)], cfg, d**-0.5, 64, kernel="rowmask")[0]
    for t in (130, 700):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        q2[:, :, t + 1:] = torch.randn_like(q2[:, :, t + 1:])
        k2[:, :, t + 1:] = torch.randn_like(k2[:, :, t + 1:])
        v2[:, :, t + 1:] = torch.randn_like(v2[:, :, t + 1:])
        got = ps.flex_packed_page_attention([(q2, k2, v2, 0, None)], cfg, d**-0.5, 64, kernel="rowmask")[0]
        torch.testing.assert_close(got[:, :, : t + 1], base[:, :, : t + 1])


@pytest.mark.parametrize("select_block", [4, 8, 16, 32])
@pytest.mark.parametrize("hq", [16, 4])
@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_rowmask_block_token_packed_matches_gathered(select_block, hq, selector, exact_scores):
    """One selection per block of B < 64 queries on the rowmask backend's head-packed tiles (only the anchors scored:
    each block's first query by absolute position, or a chunk's first; every query its anchor's top-k plus its own
    sinks and window) equals the gathered reference -- select_pages' block-wise semantics -- per span on packed
    documents, partial pages and CP-like chunks starting at and between block boundaries, outputs and q / k / v
    gradients; and it is causal."""
    if selector == "page_mean" and TF32_FORCED:
        pytest.skip("page_mean scores through cuBLAS fp32 GEMMs, forced to TF32 here")
    torch.manual_seed(4)
    g, d = 2, 32
    cfg = ps.PageSparseConfig(selector=selector, page=64, topk=3, window=128, sink_pages=1,
                              select_block=select_block, per_group=True)
    spans = []
    for L, off in [(1300, 0), (260, 0), (900, 320), (1111, 333)]:
        q = torch.randn(1, hq, L - off, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        spans.append((q, k, v, off, None))
    leaves = [[t.clone().requires_grad_() for t in sp[:3]] for sp in spans]
    outs = ps.flex_packed_page_attention([(*lv, sp[3], None) for lv, sp in zip(leaves, spans)], cfg, d**-0.5, 64,
                                         kernel="rowmask")
    gos = [torch.randn(o.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(i))
           for i, o in enumerate(outs)]
    sum((o.float() * go).sum() for o, go in zip(outs, gos)).backward()
    for (q, k, v, off, _), lv, o, go in zip(spans, leaves, outs, gos):
        qd, kd, vd = (t.detach().double().requires_grad_() for t in (q, k, v))
        ref = ps.page_sparse_attention(qd, kd, vd, cfg, off, d**-0.5)
        assert float((o.double() - ref).abs().max() / ref.abs().max()) < 2e-2
        (ref * go.double()).sum().backward()
        for a, b in zip(lv, (qd, kd, vd)):
            assert float((a.grad.double() - b.grad).abs().max() / b.grad.abs().max()) < 3e-2
    q, k, v, off, _ = spans[0]
    base = ps.flex_packed_page_attention([(q, k, v, 0, None)], cfg, d**-0.5, 64, kernel="rowmask")[0]
    for t in (130, 700):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        q2[:, :, t + 1:] = torch.randn_like(q2[:, :, t + 1:])
        k2[:, :, t + 1:] = torch.randn_like(k2[:, :, t + 1:])
        v2[:, :, t + 1:] = torch.randn_like(v2[:, :, t + 1:])
        got = ps.flex_packed_page_attention([(q2, k2, v2, 0, None)], cfg, d**-0.5, 64, kernel="rowmask")[0]
        torch.testing.assert_close(got[:, :, : t + 1], base[:, :, : t + 1])


@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_rowmask_pure_token_backend_thd_matches_gathered(single, selector, exact_scores):
    """PageSparseCoreAttention with page_sparse_backend rowmask and per-token selection (select_block 1, per group)
    equals the gathered backend on a packed batch of long and short documents, forward and backward."""
    if selector == "page_mean" and TF32_FORCED:
        pytest.skip("page_mean scores through cuBLAS fp32 GEMMs, forced to TF32 here")
    lengths = (1300, 260, 120, 900)
    kw = dict(page_sparse_selector=selector, page_sparse_page=64, page_sparse_topk=3, page_sparse_window=128,
              page_sparse_select_block=1, page_sparse_per_group=True, page_sparse_flex_q_block=64,
              page_sparse_packed_flex=True)
    tok = PageSparseCoreAttention(_config(**kw, page_sparse_backend="rowmask"), 1, pg_collection=single)
    ref = PageSparseCoreAttention(_config(**kw, page_sparse_backend="gathered"), 1, pg_collection=single)
    gen = torch.Generator().manual_seed(4)
    t = sum(lengths)
    q, k, v = (torch.randn(t, n, D, generator=gen).cuda().to(torch.bfloat16) for n in (HQ, G, G))
    dout = torch.randn(t, HQ * D, generator=gen).cuda()
    leaves_a = [x.clone().requires_grad_() for x in (q, k, v)]
    leaves_b = [x.clone().double().requires_grad_() for x in (q, k, v)]
    out_a = tok(*leaves_a, packed_seq_params=_packed(lengths))
    out_b = ref(*leaves_b, packed_seq_params=_packed(lengths))
    assert float((out_a.double() - out_b).abs().max() / out_b.abs().max()) < 2e-2
    (out_a.float() * dout).sum().backward()
    (out_b * dout.double()).sum().backward()
    for a, b in zip(leaves_a, leaves_b):
        assert float((a.grad.double() - b.grad).abs().max() / b.grad.abs().max()) < 3e-2


def test_topk_mask_equals_stable_topk():
    """topk_mask selects exactly topk_stable's pages (ties to the lower index, -inf never), incl. rows with exact ties
    across the n-th value and rows with fewer finite scores than n."""
    torch.manual_seed(7)
    sc = torch.randint(0, 6, (300, 50)).float()  # many exact ties
    sc[::7, 10:] = float("-inf")  # rows with 10 finite scores
    sc[::11] = float("-inf")
    for n in (1, 4, 12, 50):
        val, sel = ps.topk_stable(sc, n)
        ref = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, sel, torch.isfinite(val))
        torch.testing.assert_close(ps.topk_mask(sc, n), ref)


@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
@pytest.mark.parametrize("off", [0, 192])
def test_token_allowed_equals_select_pages(selector, off, exact_scores):
    """token_allowed (the per-token training path's selection: span-wide key statistics, the top-k mask) selects the
    same pages as select_pages with select_block 1, also with quantized keys (exact score ties between pages)."""
    if selector == "page_mean" and TF32_FORCED:
        pytest.skip("page_mean scores through cuBLAS fp32 GEMMs, forced to TF32 here")
    torch.manual_seed(off)
    L = 2000
    q = torch.randn(1, 8, L - off, 32, device="cuda")
    for k in (torch.randn(1, 1, L, 32, device="cuda"), torch.randint(-1, 2, (1, 1, L, 32), device="cuda").float()):
        cfg = ps.PageSparseConfig(selector=selector, page=64, topk=5, window=128, sink_pages=1, select_block=1)
        kst = ps.maxpool_windows(k, 64) if selector == "page_maxpool" else ps.page_pool(k, 64, "mean")
        got = ps.token_allowed(q, k, cfg, off, 0.2, kst)
        torch.testing.assert_close(got, ps.select_pages(q.double(), k.double(), cfg, off, 0.2)[0])


@pytest.mark.parametrize("tq", [8, 16])
@pytest.mark.parametrize("off", [0, 320])
def test_token_span_lists_fused_equals_select_pages(tq, off, exact_scores):
    """token_span_lists' fused path (Triton scores + top-k / sinks / window -> lists) lists exactly the pages
    select_pages chooses per query (per KV group, ties to the lower page incl. quantized keys), over several selection
    chunks and a CP-like offset."""
    from megatron.core.transformer.experimental_attention_variant import rowmask_attention as ra

    torch.manual_seed(tq + off)
    L, g, rep = 2600, 2, 4
    cfg = ps.PageSparseConfig(selector="page_maxpool", page=64, topk=5, window=128, sink_pages=1, select_block=1)
    q = torch.randn(1, g * rep, L - off, 32, device="cuda", dtype=torch.bfloat16)
    for k in (torch.randn(1, g, L, 32, device="cuda", dtype=torch.bfloat16),
              torch.randint(-1, 2, (1, g, L, 32), device="cuda").to(torch.bfloat16)):
        idx, bits, cnt = ps.token_span_lists(q, k, cfg, off, 0.2, None, tq, chunk=1024)
        ref = torch.stack([ps.select_pages(q[:, gi * rep:(gi + 1) * rep].double(), k[:, gi:gi + 1].double(), cfg, off,
                                           0.2)[0] for gi in range(g)])
        r_idx, r_bits, r_cnt = ra.lists_from_allowed(ref, tq=tq)
        torch.testing.assert_close(cnt, r_cnt)
        nb = r_idx.shape[-1]
        valid = torch.arange(nb, device="cuda")[None, None] < r_cnt[..., None]
        torch.testing.assert_close(torch.where(valid, idx[..., :nb], 0), torch.where(valid, r_idx, 0))
        torch.testing.assert_close(torch.where(valid, bits[..., :nb], 0), torch.where(valid, r_bits, 0))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("per_group", [True, False])
@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
def test_packed_block_selection_equals_per_span(seed, per_group, selector):
    """The block-wise selection of every span at once (flex_packed_page_attention's packed_select: one batched score,
    one top-k, no per-span / per-group loop) equals the per-span select_pages exactly -- the selections, and the
    outputs and q / k / v gradients of both kernels -- on mixed packed layouts: documents with partial pages, short
    ones (the dense fast path), CP-like chunks (q_offset > 0) and random budgets."""
    import random

    if selector == "page_mean" and TF32_FORCED:
        # the batched and per-span scores are differently shaped cuBLAS fp32 GEMMs, forced to TF32 here: near-ties may
        # round apart (both selections valid, not bit-identical)
        pytest.skip("page_mean scores through cuBLAS fp32 GEMMs, forced to TF32 here")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    hq, g, d, page = 8, 2, 32, 64
    cfg = ps.PageSparseConfig(selector=selector, page=page, topk=rng.choice([3, 8]), window=rng.choice([64, 128]),
                              sink_pages=rng.choice([0, 1, 2]), per_group=per_group, select_block=64)
    spans = []
    for _ in range(rng.randint(5, 9)):
        L = rng.choice([rng.randint(700, 2600), rng.randint(100, 600), rng.randint(2600, 4000)])
        off = rng.choice([0, 0, (rng.randint(1, L // 2) // 64) * 64, rng.randint(1, L // 2)])
        q = torch.randn(1, hq, L - off, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
        spans.append((q, k, v, off, None))
    # the selections: packed vs select_pages per span (and per group), on the spans the packed path covers
    sel_spans = [sp for sp in spans if not ps._selects_everything(-(-sp[1].shape[2] // page), cfg)]
    G, rep = (g if per_group else 1), hq // g
    assert ps._packed_select_applies(sel_spans, cfg, None, False)
    layout = tuple((sp[0].shape[2], sp[1].shape[2], sp[3]) for sp in sel_spans)
    tb = ps._packed_tables(layout, cfg, 64, G, torch.device("cuda"))
    qp, kp, _ = ps._packed_qkv(sel_spans, tb["qoff"], tb["koff"])
    got = ps._packed_block_allowed(qp, kp, cfg, d**-0.5, tb, per_group, G, rep)
    ref = torch.zeros_like(got)
    for i, (q, k, v, off, _) in enumerate(sel_spans):
        n, P, a, c = q.shape[2], -(-k.shape[2] // page), tb["qoff"][i], tb["koff"][i] // page
        sel = ([ps.select_pages(q[:, gi * rep:(gi + 1) * rep], k[:, gi:gi + 1], cfg, off, d**-0.5)[0]
                for gi in range(g)] if per_group else [ps.select_pages(q, k, cfg, off, d**-0.5)[0]])
        for gi in range(G):
            ref[gi, a:a + n, c:c + P] = sel[gi]
    assert torch.equal(got, ref)
    torch.testing.assert_close(tb["q_abs"], torch.cat([
        torch.cat([torch.arange(sp[3], sp[3] + sp[0].shape[2]), torch.full((-sp[0].shape[2] % 64,), -1)])
        for sp in sel_spans]).to(torch.int32).cuda())
    # end to end, both kernels: outputs and gradients identical to the per-span path
    for kernel, dtype in (("flex", torch.float32), ("rowmask", torch.bfloat16)):  # (local GPUs: no bf16 flex bwd)
        res = []
        for packed in (True, False):
            leaves = [[t.to(dtype).clone().requires_grad_() for t in sp[:3]] for sp in spans]
            outs = ps.flex_packed_page_attention([(*lv, sp[3], None) for lv, sp in zip(leaves, spans)], cfg,
                                                 d**-0.5, 64, kernel=kernel, packed_select=packed)
            gos = [torch.randn(o.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(i))
                   for i, o in enumerate(outs)]
            sum((o.float() * go).sum() for o, go in zip(outs, gos)).backward()
            res.append((outs, [[t.grad for t in lv] for lv in leaves]))
        for a, b in zip(res[0][0], res[1][0]):
            assert torch.equal(a, b)
        for la, lb in zip(res[0][1], res[1][1]):
            for a, b in zip(la, lb):
                assert torch.equal(a, b)


@pytest.mark.parametrize("select_block", [1, 4, 16])
@pytest.mark.parametrize("hq", [16, 4])
def test_token_packed_select_equals_per_span(hq, select_block):
    """The per-token rowmask path's packed selection (every span in one pass) gives bit-identical outputs and q / k / v
    gradients to the per-span selection on a mixed layout: long documents, short ones (the dense fast path), a span
    whose length is not a page multiple and a CP-like chunk with an offset; quantized keys (score ties) too."""
    torch.manual_seed(11)
    g, d = 2, 32
    cfg = ps.PageSparseConfig(selector="page_maxpool", page=64, topk=3, window=128, sink_pages=1,
                              select_block=select_block, per_group=True)
    for quant in (False, True):
        spans = []
        for L, off in [(1300, 0), (200, 0), (2050, 0), (130, 0), (900, 320), (777, 0), (1111, 333)]:
            q = torch.randn(1, hq, L - off, d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
            if quant:
                k = k.round().clamp(-1, 1)
            v = torch.randn(1, g, L, d, device="cuda", dtype=torch.bfloat16)
            spans.append((q, k, v, off, None))

        def run(packed):
            leaves = [[t.clone().requires_grad_() for t in sp[:3]] for sp in spans]
            outs = ps.flex_packed_page_attention([(*lv, sp[3], None) for lv, sp in zip(leaves, spans)], cfg, d**-0.5,
                                                 64, kernel="rowmask", packed_select=packed)
            gen = torch.Generator(device="cuda").manual_seed(3)
            sum((o.float() * torch.randn(o.shape, device="cuda", generator=gen)).sum() for o in outs).backward()
            return [o.detach() for o in outs] + [t.grad for lv in leaves for t in lv]

        for a, b in zip(run(True), run(False)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_token_fixed_layout_matches_reference(seed, exact_scores):
    """Pure per-token selection on the constant-size packed layout (every span, short ones included, gathered into a
    fixed length per micro-batch shape: _token_fixed_lengths) equals the gathered per-span reference, forward and
    backward; the layout's tensors have the same sizes for every layout of the same token count."""
    import random

    rng = random.Random(seed)
    torch.manual_seed(seed)
    hq, g, d = 8, 2, 64
    total = 6144
    cuts = sorted(rng.sample(range(1, total), rng.randint(0, 8)))
    segs = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    cfg = ps.PageSparseConfig(selector="page_maxpool", page=64, topk=rng.choice([2, 8]), window=128, sink_pages=1,
                              per_group=True, select_block=1)
    q = torch.randn(1, hq, total, d, device="cuda", requires_grad=True)
    k = torch.randn(1, g, total, d, device="cuda", requires_grad=True)
    v = torch.randn(1, g, total, d, device="cuda", requires_grad=True)

    def spans():
        return [(a, b, c, 0, None) for a, b, c in zip(q.split(segs, 2), k.split(segs, 2), v.split(segs, 2))]

    assert ps._token_fixed_applies(spans(), cfg)
    assert ps._token_fixed_lengths(spans(), cfg) == ps._token_fixed_lengths([(q, k, v, 0, None)], cfg)
    out = ps.flex_packed_page_attention(spans(), cfg, d**-0.5, 64, kernel="rowmask")
    ref = [ps.page_sparse_attention(sp[0], sp[1], sp[2], cfg, sp[3], d**-0.5) for sp in spans()]
    w = [torch.randn_like(o) for o in ref]
    g_out = torch.autograd.grad(sum((o * w_).sum() for o, w_ in zip(out, w)), (q, k, v))
    g_ref = torch.autograd.grad(sum((o * w_).sum() for o, w_ in zip(ref, w)), (q, k, v))
    for o, r in zip(out, ref):  # the kernel's dots run in TF32
        torch.testing.assert_close(o, r, atol=2e-2, rtol=2e-2)
    for a, b in zip(g_out, g_ref):
        torch.testing.assert_close(a, b, atol=3e-2 * b.abs().max().item(), rtol=0)


def test_token_path_stress_recompute_deterministic(single):
    """Stress of the per-token THD path as training runs it: several layers per step under activation checkpointing
    (forward, then recompute in the backward), over many random packed layouts (many tiny documents, long ones, the
    constant-size layout and its bucketed fallback), each step run twice: outputs and q / k / v gradients finite and
    bit-identical (a race, an uninitialized read or a buffer reused across layers / recompute shows up here or as an
    illegal address -- the v4b main run's crash on 2026-11-23). PAGE_SPARSE_STRESS_STEPS scales it up."""
    import random

    from torch.utils.checkpoint import checkpoint

    steps = int(os.environ.get("PAGE_SPARSE_STRESS_STEPS", "12"))
    hq, g, d, total = 8, 2, 64, 8192
    kw = dict(page_sparse_selector="page_maxpool", page_sparse_per_group=True, page_sparse_select_block=1,
              page_sparse_backend="rowmask", page_sparse_page=64, page_sparse_topk=8, page_sparse_window=128,
              page_sparse_flex_q_block=64)
    cfg = TransformerConfig(num_layers=3, hidden_size=hq * d, num_attention_heads=hq, num_query_groups=g,
                            kv_channels=d, **kw)
    layers = [PageSparseCoreAttention(cfg, i + 1, pg_collection=single).cuda() for i in range(3)]
    rng = random.Random(0)

    def layout(i):
        kind = i % 4
        if kind == 0:  # many tiny documents (the constant-size layout's worst padding)
            cuts = sorted(rng.sample(range(1, total), 40))
        elif kind == 1:  # one long document
            cuts = []
        elif kind == 2:  # past the fixed padding budget: the bucketed fallback
            cuts = sorted(rng.sample(range(1, total), 200))
        else:
            cuts = sorted(rng.sample(range(1, total), rng.randint(1, 12)))
        return [b - a for a, b in zip([0] + cuts, cuts + [total])]

    def run(segs, seed, ckpt=True):
        cu = torch.tensor([0, *torch.tensor(segs).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                              cu_seqlens_kv_padded=cu, max_seqlen_q=max(segs), max_seqlen_kv=max(segs))
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ins = [tuple(torch.randn(total, n, d, generator=gen, device="cuda").to(torch.bfloat16).requires_grad_()
                     for n in (hq, g, g)) for _ in layers]
        outs = []
        for att, (q, k, v) in zip(layers, ins):
            fn = lambda q_, k_, v_, att=att: att(q_, k_, v_, packed_seq_params=psp)
            outs.append(checkpoint(fn, q, k, v, use_reentrant=False) if ckpt else fn(q, k, v))
        sum(o.float().square().mean() for o in outs).backward()
        return [o.detach() for o in outs] + [t.grad for qkv in ins for t in qkv]

    old_pad = ps.TOKEN_FIXED_PAD
    ps.TOKEN_FIXED_PAD = 2048  # the 200-document layouts exceed it
    try:
        run(layout(1), 0, ckpt=False)  # kernel warm-up outside the checkpoint
        for i in range(steps):
            segs = layout(i)
            a, b = run(segs, i), run(segs, i)
            torch.cuda.synchronize()
            for x, y in zip(a, b):
                assert bool(torch.isfinite(x).all())
                assert torch.equal(x, y), f"step {i}: not deterministic ({len(segs)} documents)"
    finally:
        ps.TOKEN_FIXED_PAD = old_pad


def test_fixed_packed_gradients_are_views():
    """_fixed_packed: the packed tensor and the spans' gradients equal a padded concatenation's, and autograd records
    no CopySlices (whose backward cloned the whole packed gradient once per span)."""
    from megatron.core.transformer.experimental_attention_variant.page_sparse import _fixed_packed

    torch.manual_seed(0)
    lens, offs, total = [5, 64, 1, 30], [0, 64, 128, 192], 256
    base = torch.randn(1, 3, sum(lens), 8, dtype=torch.float64, requires_grad=True)
    xs = list(base.split(lens, dim=2))
    out = _fixed_packed(xs, offs, total)
    names, todo, seen = set(), [out.grad_fn], set()
    while todo:
        fn = todo.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        names.add(type(fn).__name__)
        todo.extend(f for f, _ in fn.next_functions)
    assert "CopySlices" not in names, names
    ref = torch.zeros(1, 3, total, 8, dtype=torch.float64)
    for x, o in zip(xs, offs):
        ref[:, :, o:o + x.shape[2]] = x.detach()
    assert torch.equal(out.detach(), ref)
    w = torch.randn_like(ref)
    (out * w).sum().backward()
    want = torch.cat([w[:, :, o:o + n] for o, n in zip(offs, lens)], dim=2)
    assert torch.equal(base.grad, want)
