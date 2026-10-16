# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_product_key_topk import (
    fused_product_key_topk,
    fused_product_key_topk_applies,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(s1, s2, k):
    n = s1.shape[-1]
    v1, i1 = s1.topk(k, dim=-1)
    v2, i2 = s2.topk(k, dim=-1)
    combined = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
    scores, flat = combined.topk(k, dim=-1)
    rows = i1.gather(-1, flat // k) * n + i2.gather(-1, flat % k)
    return scores, rows


@pytest.mark.parametrize("n,k", [(512, 32), (64, 8), (256, 16)])
def test_matches_torch_topk(n, k):
    torch.manual_seed(0)
    T, H = 100, 3
    s1 = torch.randn(T, H, n, device="cuda", requires_grad=True)
    s2 = torch.randn(T, H, n, device="cuda", requires_grad=True)
    assert fused_product_key_topk_applies(s1, k)
    g = torch.randn(T, H, k, device="cuda")

    ref_scores, ref_rows = _reference(s1, s2, k)
    (ref_scores * g).sum().backward()
    ref_grads = (s1.grad.clone(), s2.grad.clone())
    s1.grad = s2.grad = None

    scores, rows = fused_product_key_topk(s1, s2, k)
    (scores * g).sum().backward()
    torch.testing.assert_close(scores, ref_scores)
    # continuous random scores: no ties, so the selection (and its order) is identical
    assert torch.equal(rows, ref_rows)
    torch.testing.assert_close(s1.grad, ref_grads[0])
    torch.testing.assert_close(s2.grad, ref_grads[1])
