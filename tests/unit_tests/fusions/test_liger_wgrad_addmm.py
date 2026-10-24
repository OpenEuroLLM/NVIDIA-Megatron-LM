# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""LIGER_WGRAD_ADDMM: the output-weight gradient accumulated by one in-place GEMM per chunk is the
default path's (and an fp32 reference's) gradient, at least as accurate."""

import pytest
import torch
import torch.nn.functional as F

liger = pytest.importorskip("liger_kernel")

from megatron.core.fusions import liger_fused_linear_cross_entropy as lfce  # noqa: E402


def _grads(monkeypatch, addmm, h, w, labels):
    monkeypatch.setattr(lfce, "_WGRAD_ADDMM", addmm)
    h = h.clone().requires_grad_()
    w = w.clone().requires_grad_()
    loss = lfce.liger_fused_linear_cross_entropy(h, w, labels, chunk_size=64)
    loss.float().sum().backward()
    return loss.detach().float(), h.grad.float(), w.grad.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_wgrad_addmm_matches(monkeypatch):
    torch.manual_seed(0)
    s, b, hidden, vocab = 256, 2, 256, 4096  # 512 tokens: 8 chunks of 64
    h = (torch.randn(s, b, hidden, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(vocab, hidden, device="cuda") / hidden**0.5).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (b, s), device="cuda")
    loss_d, gh_d, gw_d = _grads(monkeypatch, False, h, w, labels)
    loss_a, gh_a, gw_a = _grads(monkeypatch, True, h, w, labels)
    # fp32 reference
    hr = h.float().requires_grad_()
    wr = w.float().requires_grad_()
    logits = hr.transpose(0, 1).reshape(-1, hidden) @ wr.t()
    F.cross_entropy(logits, labels.reshape(-1), reduction="sum").backward()
    torch.testing.assert_close(loss_a, loss_d)
    torch.testing.assert_close(gh_a, gh_d)
    err_d = (gw_d - wr.grad).abs().max().item()
    err_a = (gw_a - wr.grad).abs().max().item()
    scale = wr.grad.abs().max().item()
    assert err_a <= 2e-2 * scale, (err_a, scale)
    assert err_a <= err_d * 1.05 + 1e-6, (err_a, err_d)
