# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""PageSparseCoreAttention (page_sparse_selector): the TE core-attention replacement on packed (THD) batches --
a full budget is dense attention per document; CP=2 equals CP=1 (outputs, query / key / value gradients);
the indexer gets gradients from its KL warm-up loss, and the warm-up keeps the dense output; the trained variants
(nsa / csa2 / lsa, sparse_variants.py) and hierarchical selection are their per-document reference. 2 GPUs:

    python -m torch.distributed.run --nproc-per-node 2 -m pytest -q tests/unit_tests/transformer/test_page_sparse_attention.py
"""

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
