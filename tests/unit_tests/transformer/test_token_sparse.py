# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Per-token page selection in training (token_sparse.py): the NSA-kernel path over packed documents equals the exact
gathered reference with select_block=1 (every query its own pages), outputs and q / k / v gradients."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import page_sparse as ps
from megatron.core.transformer.experimental_attention_variant import token_sparse as ts

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton kernels)")
pytest.importorskip("fla.ops.nsa.parallel")


@pytest.mark.parametrize("selector", ["page_mean", "page_maxpool"])
@pytest.mark.parametrize("per_group", [True, False])
@pytest.mark.parametrize("lengths", [(256,), (200, 130, 70), (40, 300)])
def test_token_selection_matches_gathered_reference(selector, per_group, lengths):
    torch.manual_seed(0)
    hq, g, d = 4, 2, 64
    t = sum(lengths)
    q = torch.randn(t, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(t, g, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(t, g, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cfg = ps.PageSparseConfig(selector=selector, page=16, topk=3, window=32, sink_pages=1, select_block=1,
                              per_group=per_group)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    scale = d**-0.5
    out = ts.token_page_attention_thd(q, k, v, cu, cfg, scale)
    go = torch.randn_like(out)
    grads = torch.autograd.grad((out.float() * go.float()).sum(), (q, k, v))
    refs, a = [], 0
    qf, kf, vf = (x.detach().double().requires_grad_() for x in (q, k, v))
    for L in lengths:
        qd, kd, vd = (x[a:a + L].permute(1, 0, 2)[None] for x in (qf, kf, vf))
        refs.append(ps.page_sparse_attention(qd, kd, vd, cfg, 0, scale)[0].permute(1, 0, 2))
        a += L
    ref = torch.cat(refs, 0)
    rgrads = torch.autograd.grad((ref * go.double()).sum(), (qf, kf, vf))
    assert float((out.double() - ref).abs().max() / ref.abs().max()) < 2e-2
    for x, y in zip(grads, rgrads):
        assert float((x.double() - y).abs().max() / y.abs().max()) < 3e-2


def test_token_indices_are_distinct_and_causal():
    torch.manual_seed(1)
    q, k = torch.randn(1, 4, 300, 32, device="cuda"), torch.randn(1, 2, 300, 32, device="cuda")
    cfg = ps.PageSparseConfig(selector="page_maxpool", page=16, topk=3, window=32, sink_pages=1, per_group=True)
    idx = ts.token_page_indices(q, k, cfg, 0, 32**-0.5)  # [n, G, S]
    cur = (torch.arange(300, device="cuda") // 16)[:, None, None]
    assert bool(((idx < 0) | (idx <= cur)).all())
    for row in idx.reshape(-1, idx.shape[-1]).tolist():
        valid = [p for p in row if p >= 0]
        assert len(valid) == len(set(valid))
