# Copyright (c) 2026, OpenSci / OpenEuroLLM contributors.
#
# Adapter for Liger-Kernel's chunked fused output projection and cross entropy.
# Liger computes logits a token chunk at a time and immediately reuses the
# chunk as dlogits.  This avoids materialising the [sequence, batch, vocab]
# output tensor that the normal vocabulary-parallel loss must retain.

"""Liger fused linear-cross-entropy adapter for non-vocab-parallel GPT heads."""

from __future__ import annotations

import torch

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.ops import fused_linear_cross_entropy as liger_fused_ce_ops
    from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd

    HAVE_LIGER = True
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None  # type: ignore[assignment,misc]
    liger_fused_ce_ops = None  # type: ignore[assignment]
    amp_custom_fwd = lambda fn: fn
    amp_custom_bwd = lambda fn: fn
    HAVE_LIGER = False


class _LigerFixedChunkLinearCrossEntropy(torch.autograd.Function):
    """Liger 0.8.0 LM-head path with an explicit power-of-two token chunk.

    Its automatic chunking preserves a very small temporary-logits budget. For
    our 262k vocabulary/H=512 model this creates 512 chunks at MBS=32 and
    repeats the output-head gradient accumulation 512 times. An opt-in chunk
    trades otherwise available temporary memory for much fewer launches.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, chunk_size: int):
        assert liger_fused_ce_ops is not None
        if chunk_size <= 0 or chunk_size & (chunk_size - 1):
            raise ValueError(f"Liger chunk size must be a positive power of two, got {chunk_size}")

        batch_tokens = input_.shape[0]
        vocab_size = weight.shape[0]
        chunk_size = min(chunk_size, batch_tokens)
        num_chunks = (batch_tokens + chunk_size - 1) // chunk_size
        block_size = min(
            liger_fused_ce_ops.MAX_FUSED_SIZE,
            liger_fused_ce_ops.triton.next_power_of_2(vocab_size),
        )
        num_warps = 32 if not liger_fused_ce_ops.is_hip() else 16
        grad_input = torch.zeros_like(input_)
        grad_weight = torch.zeros_like(weight) if input_.requires_grad and weight.requires_grad else None
        loss_1d = torch.zeros(batch_tokens, dtype=torch.float32, device=input_.device)
        # `reduction="none"` does not use this scalar to scale loss or
        # gradients; ignored targets are handled directly in Liger's Triton
        # kernel. Keeping it a shape-derived Python integer avoids a device
        # `.item()` synchronization, which is illegal during full-iteration
        # CUDA graph capture.
        total_n_non_ignore = batch_tokens

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, batch_tokens)
            input_chunk = input_[start_idx:end_idx]
            logits_chunk = (input_chunk @ weight.t()).contiguous()
            target_chunk = target[start_idx:end_idx].contiguous()
            loss_chunk = loss_1d[start_idx:end_idx]
            n_rows = logits_chunk.shape[0]
            liger_fused_ce_ops.liger_cross_entropy_kernel[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),
                weight_ptr=None,
                loss_ptr=loss_chunk,
                z_loss_ptr=None,
                loss_stride=loss_chunk.stride(-1),
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=vocab_size,
                n_non_ignore=total_n_non_ignore,
                sum_non_ignore_weight=total_n_non_ignore,
                weight_sum=0.0,
                ignore_index=-100,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction="none",
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
                HAS_GRADIENTS=input_.requires_grad,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            if input_.requires_grad:
                grad_input[start_idx:end_idx] = logits_chunk @ weight
                if grad_weight is not None:
                    # Preserve Liger's original-dtype accumulation semantics.
                    grad_weight += torch.mm(logits_chunk.t(), input_chunk).float()

        ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if grad_weight is not None else None)
        return loss_1d

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert liger_fused_ce_ops is not None
        grad_input, grad_weight = ctx.saved_tensors
        # Liger's helper first builds a device scalar and calls ``torch.equal``
        # to skip this when grad_output is one. Both are illegal inside a full
        # CUDA graph capture. Applying the same Triton scaling kernel always is
        # mathematically identical (including the common multiply-by-one case).
        _, hidden_size = grad_input.shape
        block_size = min(
            liger_fused_ce_ops.MAX_FUSED_SIZE,
            liger_fused_ce_ops.triton.next_power_of_2(hidden_size),
        )
        num_warps = 32 if not liger_fused_ce_ops.is_hip() else 16
        liger_fused_ce_ops.element_mul_kernel[(grad_input.shape[0],)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        if grad_weight is not None:
            vocab_size, _ = grad_weight.shape
            liger_fused_ce_ops.element_mul_kernel[(vocab_size,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                hidden_size,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        return grad_input, grad_weight, None, None


def liger_fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Return per-token LM loss without materialising full-vocabulary logits.

    Args:
        hidden_states: Decoder output of shape ``[sequence, batch, hidden]``.
        output_weight: LM-head weight of shape ``[vocab, hidden]``. This may be
            the embedding weight when input/output embeddings are tied.
        labels: Target ids of shape ``[batch, sequence]``.
        loss_mask: Optional binary mask of shape ``[batch, sequence]``. Zero
            entries are converted to Liger's ignore index before the fused
            backward pass. Megatron's pretraining datasets use a binary mask.

    Returns:
        Per-token losses of shape ``[batch, sequence]``, matching Megatron's
        normal ``compute_language_model_loss`` contract.

    Liger's ``reduction='none'`` is intentional: Megatron applies the dataset
    loss mask after the model forward. Released Liger 0.8.0 does not apply a
    non-uniform upstream gradient correctly for this reduction, so masked
    targets are also converted to ``ignore_index`` here. With Megatron's
    binary loss mask this is exactly equivalent and keeps masked tokens out of
    Liger's fused backward pass.

    This adapter is deliberately limited to TP=1 by the model configuration.
    Liger's kernel performs a single-vocabulary softmax and is not a replacement
    for Megatron's vocabulary-parallel cross entropy.
    """
    if not HAVE_LIGER:
        raise ImportError(
            "liger-kernel is required for --liger-fused-linear-cross-entropy. "
            "Rebuild the training container with liger-kernel installed."
        )
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected [sequence, batch, hidden] states, got {hidden_states.shape}")
    if labels.shape != (hidden_states.shape[1], hidden_states.shape[0]):
        raise ValueError(
            "Labels must have shape [batch, sequence] matching hidden states; "
            f"got {labels.shape} for {hidden_states.shape}."
        )
    if loss_mask is not None and loss_mask.shape != labels.shape:
        raise ValueError(
            f"Loss mask must match labels shape {labels.shape}, got {loss_mask.shape}."
        )
    if output_weight.ndim != 2 or output_weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError(
            "LM-head weight must have shape [vocab, hidden]; "
            f"got {output_weight.shape} for hidden size {hidden_states.shape[-1]}."
        )

    sequence_length, micro_batch_size, hidden_size = hidden_states.shape
    inputs_2d = hidden_states.reshape(-1, hidden_size)
    if loss_mask is not None:
        labels = labels.masked_fill(loss_mask == 0, -100)
    targets_1d = labels.transpose(0, 1).contiguous().reshape(-1)
    if chunk_size is None:
        loss_1d = LigerFusedLinearCrossEntropyLoss(reduction="none")(
            output_weight, inputs_2d, targets_1d
        )
    else:
        loss_1d = _LigerFixedChunkLinearCrossEntropy.apply(
            inputs_2d, output_weight, targets_1d, chunk_size
        )
    return loss_1d.view(sequence_length, micro_batch_size).transpose(0, 1).contiguous()
