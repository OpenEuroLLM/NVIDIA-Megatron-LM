import numpy as np
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_logsumexp,
)
from tests.unit_tests.test_utilities import Utils


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
        loss, logZ = vocab_parallel_cross_entropy(shard, target, return_logsumexp=True)
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
