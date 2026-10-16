# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_weighted_bag import weighted_bag, weighted_bag_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _case(t, j, u, d, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.randn(u, d, device="cuda", generator=g).to(dtype)
    # every row used at least once (as for the inverse of torch.unique), some heavily
    index = torch.cat([torch.arange(u, device="cuda"),
                       torch.randint(0, u, (t * j - u,), device="cuda", generator=g)])
    index = index[torch.randperm(t * j, device="cuda", generator=g)].view(t, j)
    weights = torch.softmax(torch.randn(t, j, device="cuda", generator=g), -1)
    return table, index, weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("t,j,u,d", [(64, 128, 500, 768), (37, 20, 37, 100), (8, 7, 3, 1030)])
def test_matches_embedding_bag(t, j, u, d, dtype):
    table, index, weights = _case(t, j, u, d, dtype)
    grad = torch.randn(t, d, device="cuda")

    tab_ref = table.detach().float().requires_grad_(True)
    w_ref = weights.detach().clone().requires_grad_(True)
    ref = weighted_bag_reference(tab_ref, index, w_ref)
    ref.backward(grad)

    tab = table.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    out = weighted_bag(tab, index, w)
    out.backward(grad)

    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out, ref, **tol)
    torch.testing.assert_close(w.grad, w_ref.grad, **tol)
    assert tab.grad.dtype == dtype
    torch.testing.assert_close(tab.grad.float(), tab_ref.grad, **tol)


def test_backward_is_deterministic():
    table, index, weights = _case(256, 128, 300, 768, torch.bfloat16, seed=1)
    grads = []
    for _ in range(2):
        tab = table.detach().clone().requires_grad_(True)
        weighted_bag(tab, index, weights).sum().backward()
        grads.append(tab.grad)
    assert torch.equal(grads[0], grads[1])
