# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""The output z-loss on Liger's fixed-chunk fused head + cross entropy: the returned loss is the pure
cross entropy (as Megatron's native z-loss keeps the logged lm loss), the gradients are those of
CE + coeff * logsumexp^2 per token, the per-token z output is coeff * logsumexp^2 (0 for masked
tokens), and without the coefficient nothing changes."""

import pytest
import torch
import torch.nn.functional as F

liger = pytest.importorskip("liger_kernel")

from megatron.core.fusions import liger_fused_linear_cross_entropy as lfce  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("addmm", [False, True])
def test_liger_z_loss_matches_reference(monkeypatch, addmm):
    monkeypatch.setattr(lfce, "_WGRAD_ADDMM", addmm)
    torch.manual_seed(0)
    s, b, hidden, vocab, coeff = 256, 2, 256, 4096, 1e-2  # 512 tokens: 8 chunks of 64
    h0 = torch.randn(s, b, hidden, device="cuda") * 2.0
    w0 = torch.randn(vocab, hidden, device="cuda") / hidden**0.5
    w0 += 0.05 * torch.randn(1, hidden, device="cuda")  # a shared component: logZ well away from 0
    labels = torch.randint(0, vocab, (b, s), device="cuda")
    mask = torch.ones(b, s, device="cuda")
    mask[:, -16:] = 0  # masked tokens: no CE, no z-loss, no gradient

    h = h0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    loss, z = lfce.liger_fused_linear_cross_entropy(h, w, labels, mask, chunk_size=64,
                                                    lse_square_scale=coeff)
    (loss.float() * mask).sum().backward()

    hr = h0.clone().requires_grad_()
    wr = w0.clone().requires_grad_()
    logits = hr.transpose(0, 1).reshape(-1, hidden) @ wr.t()
    ce = F.cross_entropy(logits, labels.reshape(-1), reduction="none").view(b, s)
    lse = torch.logsumexp(logits, dim=-1).view(b, s)
    ((ce + coeff * lse.square()) * mask).sum().backward()

    torch.testing.assert_close(loss * mask, ce.detach() * mask, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(z, coeff * lse.detach().square() * mask, rtol=1e-4, atol=1e-5)
    assert not z.requires_grad
    torch.testing.assert_close(h.grad, hr.grad, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(w.grad, wr.grad, rtol=2e-3, atol=2e-4)
    # the z-loss really changes the gradient here (the test is not vacuous)
    hn = h0.clone().requires_grad_()
    plain = lfce.liger_fused_linear_cross_entropy(hn, w0, labels, mask, chunk_size=64)
    (plain.float() * mask).sum().backward()
    assert (hn.grad - h.grad).abs().max() > 1e-2 * h.grad.abs().max()
    torch.testing.assert_close(plain * mask, loss * mask, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_liger_z_loss_needs_the_chunk_path():
    h = torch.randn(8, 1, 16, device="cuda")
    w = torch.randn(32, 16, device="cuda")
    labels = torch.randint(0, 32, (1, 8), device="cuda")
    with pytest.raises(NotImplementedError, match="chunk-size"):
        lfce.liger_fused_linear_cross_entropy(h, w, labels, lse_square_scale=1e-4)
