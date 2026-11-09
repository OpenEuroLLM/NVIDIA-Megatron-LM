# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Liger's fused backward scales the gradients by one upstream value; Megatron's loss is sum(loss * loss_mask), so
that value must come from a token that is not masked: with the first token masked the old adapter (which read
grad_output[0]) returned zero gradients for the whole call. Gradients match the unfused reference either way."""

import pytest
import torch
import torch.nn.functional as F

liger = pytest.importorskip("liger_kernel")

from megatron.core.fusions import liger_fused_linear_cross_entropy as lfce  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("first_masked", [False, True])
def test_liger_gradients_with_masked_first_token(monkeypatch, first_masked):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)  # MN5 images enable TF32
    torch.manual_seed(0)
    s, b, hidden, vocab = 128, 2, 64, 1024
    h0 = torch.randn(s, b, hidden, device="cuda")
    w0 = torch.randn(vocab, hidden, device="cuda") / hidden**0.5
    labels = torch.randint(0, vocab, (b, s), device="cuda")
    mask = torch.ones(b, s, device="cuda")
    mask[:, -8:] = 0
    if first_masked:
        mask[0, 0] = 0  # the first token of the flattened [s, b] order

    h = h0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    loss = lfce.liger_fused_linear_cross_entropy(h, w, labels, mask, chunk_size=64)
    (0.37 * loss.float() * mask).sum().backward()

    hr = h0.clone().requires_grad_()
    wr = w0.clone().requires_grad_()
    logits = hr.transpose(0, 1).reshape(-1, hidden) @ wr.t()
    ce = F.cross_entropy(logits, labels.reshape(-1), reduction="none").view(b, s)
    (0.37 * ce * mask).sum().backward()
    assert h.grad.abs().sum() > 0
    torch.testing.assert_close(h.grad, hr.grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(w.grad, wr.grad, atol=2e-5, rtol=2e-4)
