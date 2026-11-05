# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""PageSparseCoreAttention (page_sparse_selector): the TE core-attention replacement on packed (THD) batches --
a full budget is dense attention per document; CP=2 equals CP=1 (outputs, query / key / value gradients);
the indexer gets gradients from its KL warm-up loss, and the warm-up keeps the dense output. 2 GPUs:

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
