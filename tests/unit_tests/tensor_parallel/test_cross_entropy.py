# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.models.common.language_module import language_module as language_module_module
from megatron.core.tensor_parallel import cross_entropy as cross_entropy_module
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_logsumexp,
)
from tests.unit_tests.test_utilities import Utils


class _FakeTPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


def test_vocab_parallel_cross_entropy_uses_explicit_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    all_reduce_groups = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_groups.append(group)
        return tensor

    def fail_parallel_state_call(*args, **kwargs):
        raise AssertionError("explicit tp_group should avoid parallel_state")

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        cross_entropy_module, "get_tensor_model_parallel_group", fail_parallel_state_call
    )

    vocab_parallel_logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    target = torch.tensor([2, 0])
    expected_output = torch.nn.functional.cross_entropy(
        vocab_parallel_logits.clone(), target, reduction="none"
    )

    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=tp_group)

    torch.testing.assert_close(output, expected_output)
    assert all_reduce_groups == [tp_group, tp_group, tp_group]


def test_language_module_unfused_loss_passes_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    captured = {}

    def fake_vocab_parallel_cross_entropy(
        logits, labels, label_smoothing=0.0, tp_group=None, return_logsumexp=False
    ):
        captured["logits"] = logits
        captured["labels"] = labels
        captured["label_smoothing"] = label_smoothing
        captured["tp_group"] = tp_group
        return torch.zeros_like(labels, dtype=logits.dtype)

    monkeypatch.setattr(
        language_module_module.tensor_parallel,
        "vocab_parallel_cross_entropy",
        fake_vocab_parallel_cross_entropy,
    )

    module = SimpleNamespace(
        config=SimpleNamespace(
            cross_entropy_loss_fusion=False,
            output_z_loss_coeff=None,
            log_output_logsumexp=False,
        ),
        tp_group=tp_group
    )
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.randn(3, 2, 4)

    loss = language_module_module.LanguageModule.compute_language_model_loss(
        module, labels=labels, logits=logits
    )

    assert captured["logits"] is logits
    assert captured["tp_group"] is tp_group
    assert captured["label_smoothing"] == 0.0
    torch.testing.assert_close(captured["labels"], labels.transpose(0, 1).contiguous())
    assert loss.shape == labels.shape


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 2)
    vocab_parallel_logits = torch.range(0, 7).repeat(16, 4).cuda()
    target = torch.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    assert torch.equal(torch.round(expected_output), torch.round(output))
    Utils.destroy_model_parallel()


def _shard_global_logits(global_logits):
    """Slice this TP rank's vocab shard out of a full-vocabulary logits tensor."""
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    vocab = global_logits.size(-1)
    assert vocab % tp_size == 0
    part = vocab // tp_size
    shard = global_logits[..., tp_rank * part : (tp_rank + 1) * part].detach().clone()
    return shard.requires_grad_(True)


@pytest.mark.parametrize("impl", ["native", "fused", "standalone"])
def test_vocab_parallel_output_zloss(impl):
    """logsumexp value + z-loss gradient are TP-correct across all cross-entropy paths.

    A known full-vocabulary logits tensor is sharded across tensor-parallel ranks; the
    per-token logsumexp and the coeff*logZ**2 gradient are checked against a full-vocab
    reference computed with plain PyTorch.
    """
    tp_size = Utils.world_size
    Utils.initialize_model_parallel(tp_size, 1)

    seq, batch, vocab, coeff = 5, 3, 8 * tp_size, 1e-2
    torch.manual_seed(1234)
    global_logits = torch.randn(seq, batch, vocab).cuda()
    target = torch.randint(0, vocab, (seq, batch)).cuda()

    # Full-vocab reference (cross entropy + output z-loss).
    ref_logits = global_logits.detach().clone().requires_grad_(True)
    ref_logZ = torch.logsumexp(ref_logits.reshape(-1, vocab), dim=-1)
    ref_ce = torch.nn.functional.cross_entropy(
        ref_logits.reshape(-1, vocab), target.reshape(-1), reduction="none"
    )
    (ref_ce + coeff * ref_logZ**2).sum().backward()
    ref_logZ = ref_logZ.reshape(seq, batch)

    shard = _shard_global_logits(global_logits)
    tp_group = parallel_state.get_tensor_model_parallel_group()

    if impl == "native":
        loss, logZ = vocab_parallel_cross_entropy(
            shard, target, tp_group=tp_group, return_logsumexp=True
        )
        (loss + coeff * logZ**2).sum().backward()
    elif impl == "fused":
        loss, logZ = fused_vocab_parallel_cross_entropy(
            shard, target, tp_group, return_logsumexp=True
        )
        (loss + coeff * logZ**2).sum().backward()
    else:  # standalone logsumexp used for the TE path
        logZ = vocab_parallel_logsumexp(shard, tp_group)
        (coeff * logZ**2).sum().backward()

    # logsumexp is computed over the full (all-reduced) vocabulary -> matches the reference.
    assert torch.allclose(logZ, ref_logZ, atol=1e-3)

    # Gradient of this rank's shard matches the corresponding slice of the reference grad.
    # The fused path casts gradients to bf16 internally, so use a looser tolerance there.
    part = vocab // tp_size
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    ref_grad_shard = ref_logits.grad[..., tp_rank * part : (tp_rank + 1) * part]
    if impl == "standalone":
        # standalone helper only carries the z-loss gradient, not the CE gradient
        ref_grad_shard = (2 * coeff * ref_logZ).unsqueeze(-1) * torch.softmax(
            global_logits, dim=-1
        )[..., tp_rank * part : (tp_rank + 1) * part]
    atol = 3e-2 if impl == "fused" else 1e-3
    assert torch.allclose(shard.grad.float(), ref_grad_shard, atol=atol)

    Utils.destroy_model_parallel()


def _fused_zloss_grad(logits, target, coeff, fp32_grad_accum, monkeypatch, with_zloss=True):
    """Backward of CE (+ coeff*logZ**2) through the fused kernel at TP=1, on CPU."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, op=None, group=None: t)

    shard = logits.detach().clone().requires_grad_(True)
    loss, logZ = fused_vocab_parallel_cross_entropy(
        shard,
        target,
        _FakeTPGroup(),
        return_logsumexp=True,
        fp32_grad_accum=fp32_grad_accum,
    )
    total = loss.sum() + (coeff * logZ**2).sum() if with_zloss else loss.sum()
    total.backward()
    return shard.grad


def test_fused_zloss_grad_is_annihilated_without_fp32_accum(monkeypatch):
    """The z-loss logit gradient vanishes EXACTLY in bf16, and survives with fp32 accum.

    `calculate_gradients` rounds the cross-entropy gradient to bf16 before the z-loss
    gradient is added. For a non-target vocab entry both are proportional to the same
    softmax probability, so their ratio is the constant r = 2 * coeff * logZ. bf16 keeps 8
    significant bits, so a value already on the bf16 grid cannot move unless perturbed by
    more than half an ulp -- at least 2**-9 in relative terms. When r < 2**-9 the addition
    is a no-op on every single element.

    This is not a rounding nuisance: at the 32B flagship's coeff=1e-4 and logZ~8,
    r = 1.6e-3 sits below that floor and the regularizer contributes nothing at all.
    """
    torch.manual_seed(1234)
    seq, batch, vocab, coeff = 16, 8, 64, 1e-4
    logits = (torch.randn(seq, batch, vocab) * 2.0).bfloat16()
    target = torch.randint(0, vocab, (seq, batch))

    logZ = torch.logsumexp(logits.float(), dim=-1)
    ratio = 2 * coeff * logZ.mean().item()
    assert ratio < 2**-9, f"test setup must land below the bf16 half-ulp floor, got {ratio}"

    ce_only = _fused_zloss_grad(logits, target, coeff, False, monkeypatch, with_zloss=False)
    bf16_accum = _fused_zloss_grad(logits, target, coeff, False, monkeypatch)
    fp32_accum = _fused_zloss_grad(logits, target, coeff, True, monkeypatch)

    assert bf16_accum.dtype == fp32_accum.dtype == torch.bfloat16

    # Adding the z-loss changes NOTHING when the CE gradient is rounded down first.
    assert torch.equal(bf16_accum, ce_only)
    # Summing in fp32 first lets it through on a meaningful fraction of elements.
    moved = (fp32_accum != ce_only).float().mean().item()
    assert moved > 0.01, f"fp32 accumulation delivered nothing either ({moved:.4%} moved)"


def test_fused_zloss_fp32_accum_matches_fp32_reference(monkeypatch):
    """With fp32 accumulation the gradient is closer to the exact fp32 sum of both terms."""
    torch.manual_seed(0)
    seq, batch, vocab, coeff = 16, 8, 64, 1e-4
    logits = (torch.randn(seq, batch, vocab) * 2.0).bfloat16()
    target = torch.randint(0, vocab, (seq, batch))

    ref = logits.float().detach().requires_grad_(True)
    logZ = torch.logsumexp(ref, dim=-1)
    ce = torch.nn.functional.cross_entropy(
        ref.reshape(-1, vocab), target.reshape(-1), reduction="none"
    )
    (ce.sum() + (coeff * logZ**2).sum()).backward()

    bf16_accum = _fused_zloss_grad(logits, target, coeff, False, monkeypatch)
    fp32_accum = _fused_zloss_grad(logits, target, coeff, True, monkeypatch)

    err_bf16 = (bf16_accum.float() - ref.grad).abs().sum()
    err_fp32 = (fp32_accum.float() - ref.grad).abs().sum()
    assert err_fp32 < err_bf16


def test_fp32_grad_accum_makes_fused_agree_with_unfused(monkeypatch):
    """fp32 accumulation makes the fused kernel reproduce the unfused reference.

    The unfused `vocab_parallel_cross_entropy` never narrows its gradient: its softmax is
    fp32 and it adds the logsumexp gradient at that width, so it has always accumulated
    the z-loss correctly and takes no flag. The fused kernel is the one that rounds to
    bf16 first. So the unfused path is the reference the flag is trying to match, and
    "does the flag work" has an exact answer rather than a tolerance.

    Observed bit-for-bit equal with the JIT fuser falling back to eager; asserted here as
    closeness plus a strict ordering so the test does not depend on whether inductor is
    available and whether it reassociates the fused arithmetic.
    """
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, op=None, group=None: t)
    torch.manual_seed(1234)
    seq, batch, vocab, coeff = 32, 8, 128, 1e-4
    logits = (torch.randn(seq, batch, vocab) * 2.0).bfloat16()
    target = torch.randint(0, vocab, (seq, batch))
    tp_group = _FakeTPGroup()

    def run(fused, fp32_grad_accum=False):
        shard = logits.detach().clone().requires_grad_(True)
        if fused:
            loss, logZ = fused_vocab_parallel_cross_entropy(
                shard, target, tp_group, return_logsumexp=True, fp32_grad_accum=fp32_grad_accum
            )
        else:
            loss, logZ = vocab_parallel_cross_entropy(
                shard, target, tp_group=tp_group, return_logsumexp=True
            )
        (loss.sum() + (coeff * logZ**2).sum()).backward()
        return shard.grad.float()

    reference = run(fused=False)
    with_flag = run(fused=True, fp32_grad_accum=True)
    without_flag = run(fused=True, fp32_grad_accum=False)

    err_with = (with_flag - reference).abs().sum()
    err_without = (without_flag - reference).abs().sum()

    assert err_with < err_without, "fp32 accumulation did not move the fused path closer"
    torch.testing.assert_close(with_flag, reference, rtol=1e-3, atol=1e-8)
    # And the default really is wrong, not merely less precise: the z-loss it dropped is
    # large enough to separate it from the reference on a large fraction of elements.
    assert (without_flag != reference).float().mean() > 0.1


@pytest.mark.parametrize("fused", [True, False])
def test_logsumexp_for_logging_only_adds_no_gradient(monkeypatch, fused):
    """log_output_logsumexp must be forward-only: same gradient as not asking for it at all.

    Both cross-entropy Functions call `ctx.set_materialize_grads(False)`, so an unconsumed
    logsumexp output arrives in backward as None rather than as a zero-filled tensor the
    size of the logits shard.
    """
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, op=None, group=None: t)
    torch.manual_seed(7)
    seq, batch, vocab = 16, 8, 64
    logits = (torch.randn(seq, batch, vocab) * 2.0).bfloat16()
    target = torch.randint(0, vocab, (seq, batch))
    tp_group = _FakeTPGroup()

    def run(return_logsumexp):
        shard = logits.detach().clone().requires_grad_(True)
        if fused:
            out = fused_vocab_parallel_cross_entropy(
                shard, target, tp_group, return_logsumexp=return_logsumexp
            )
        else:
            out = vocab_parallel_cross_entropy(
                shard, target, tp_group=tp_group, return_logsumexp=return_logsumexp
            )
        # Mirror compute_language_model_loss: the log-normalizer is detached and only read.
        loss, logZ = out if return_logsumexp else (out, None)
        loss.sum().backward()
        return shard.grad, (logZ.detach() ** 2).mean() if logZ is not None else None

    baseline, _ = run(False)
    logged, stat = run(True)

    assert torch.equal(baseline, logged)
    assert torch.isfinite(stat)


def test_fp32_grad_accum_rejects_te_cross_entropy():
    """The flag must fail loudly on the TE path rather than silently doing nothing."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(num_layers=1, hidden_size=8, num_attention_heads=1)
    # Accepted on the fused native path.
    TransformerConfig(
        **kwargs,
        output_z_loss_coeff=1e-4,
        output_z_loss_fp32_grad_accum=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='native',
    )
    with pytest.raises(ValueError, match="cross_entropy_fusion_impl='te'"):
        TransformerConfig(
            **kwargs,
            output_z_loss_coeff=1e-4,
            output_z_loss_fp32_grad_accum=True,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl='te',
        )
