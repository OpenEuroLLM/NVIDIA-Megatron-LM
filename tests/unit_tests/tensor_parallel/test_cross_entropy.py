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
        config=SimpleNamespace(cross_entropy_loss_fusion=False, output_z_loss_coeff=None),
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
