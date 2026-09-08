# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch

from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from megatron.core.tensor_parallel.utils import VocabUtility


@jit_fuser
def calculate_logits_max(vocab_parallel_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the maximum logits of the predicted tokens.
    """

    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits
    )

    return vocab_parallel_logits, logits_max


@jit_fuser
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the predicted logits for the tokens.
    """
    (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
        VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )
    )

    predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))

    return target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits


@jit_fuser
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor, predicted_logits_sum_exp_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the final cross entropy loss for the tokens.
    """
    split_val = predicted_logits_sum_exp_logits.size()[0] // 2
    predicted_logits, sum_exp_logits = torch.split(predicted_logits_sum_exp_logits, split_val)

    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )

    return exp_logits, loss


@jit_fuser
def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the logits gradients scaled based on the CE loss
    """
    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )

    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )

    grad_input = grad_input.to(torch.bfloat16)

    return grad_input


@jit_fuser
def calculate_gradients_fp32(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:
    """Same as `calculate_gradients` but WITHOUT the cast down to bfloat16.

    `softmax` is fp32 (`VocabParallelCrossEntropy.calculate_logits_max` upcasts), so this
    returns the CE logit gradient in fp32. Used when another term -- the output z-loss --
    still has to be added: rounding the CE gradient to bf16 first would round the small
    z-loss contribution away before it is ever summed in.
    """
    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )

    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )

    return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, vocab_parallel_logits, target, tp_group, return_logsumexp=False, fp32_grad_accum=False
    ):
        """
        Forward implementation for the cross entropy loss.
        """
        # Captured before calculate_logits_max, which rebinds the name to an fp32 copy.
        ctx.logits_dtype = vocab_parallel_logits.dtype
        ctx.fp32_grad_accum = fp32_grad_accum

        # When logsumexp is returned for logging only (log_output_logsumexp without a z-loss
        # coefficient) nothing consumes it, and materializing its gradient would allocate a
        # zero-filled fp32 tensor the size of the logits shard -- gigabytes at production
        # vocab -- for an add of zero. Ask autograd for None instead.
        ctx.set_materialize_grads(False)

        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, tp_group.rank(), tp_group.size()
        )

        (target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits) = (
            calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All reduce is needed to get the chunks from other GPUs.
        # In the fused case, tensors are batches to invoke a single
        # AllReduce call
        torch.distributed.all_reduce(
            predicted_logits_sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        # Per-token log-normalizer logZ = log(sum_v exp(logit_v)) = log(sum_exp_logits) + logits_max,
        # where sum_exp_logits is the second half of the (TP all-reduced) concatenated tensor.
        logsumexp = None
        if return_logsumexp:
            split_val = predicted_logits_sum_exp_logits.size()[0] // 2
            sum_exp_logits = predicted_logits_sum_exp_logits[split_val:]
            logsumexp = torch.log(sum_exp_logits) + logits_max

        exp_logits, loss = calculate_cross_entropy_loss(exp_logits, predicted_logits_sum_exp_logits)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        if return_logsumexp:
            return loss, logsumexp
        return loss

    @staticmethod
    def backward(ctx, grad_output, grad_logsumexp=None):
        """
        Backward implementation for the cross entropy loss.
        """
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # Gradient contribution of logsumexp (output z-loss): d logZ / d logit = softmax.
        # Snapshot before calculate_gradients modifies softmax in place.
        logsumexp_grad = (
            softmax * grad_logsumexp.unsqueeze(dim=-1) if grad_logsumexp is not None else None
        )

        if logsumexp_grad is not None and ctx.fp32_grad_accum:
            # Sum both terms at full precision, then round the TOTAL once. The default branch
            # below instead rounds the CE gradient to bf16 first, and the z-loss gradient is
            # 2 * coeff * logZ times its magnitude -- ~1.6e-3 at coeff=1e-4 and logZ~8, i.e.
            # below bf16's 2**-9 unit roundoff, so adding it moves almost no elements.
            grad_input = calculate_gradients_fp32(
                softmax, grad_output, target_mask, masked_target_1d
            )
            # In place: grad_input is this Function's own softmax scratch buffer, and an
            # out-of-place sum would cost another fp32 logits shard -- 2.1 GB at the 32B
            # flagship's [4096, 2, 256000/4].
            grad_input = grad_input.add_(logsumexp_grad).to(ctx.logits_dtype)
        else:
            grad_input = calculate_gradients(softmax, grad_output, target_mask, masked_target_1d)

            if logsumexp_grad is not None:
                grad_input = grad_input + logsumexp_grad.to(grad_input.dtype)

        # One None per forward input after ctx: target, tp_group, return_logsumexp,
        # fp32_grad_accum.
        return grad_input, None, None, None, None


def fused_vocab_parallel_cross_entropy(
    vocab_parallel_logits, target, tp_group, return_logsumexp=False, fp32_grad_accum=False
):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]
        tp_group: the tensor parallel group over which to all reduce
        return_logsumexp: if True, also return the per-token log-normalizer
            logsumexp(logits, dim=vocab) of shape [sequence_length, batch_size], differentiable
            w.r.t. the logits (used by the output z-loss).
        fp32_grad_accum: if True, add the logsumexp gradient to the cross-entropy gradient in
            fp32 and round the sum once, instead of rounding the cross-entropy gradient to
            bf16 first. Only has an effect together with return_logsumexp. Costs one fp32
            buffer the size of the logits shard during the backward pass.

    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, tp_group, return_logsumexp, fp32_grad_accum
    )
