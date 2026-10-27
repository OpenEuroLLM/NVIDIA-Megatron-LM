# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.tensor_parallel import checkpoint_offload
from megatron.core.transformer.transformer_layer import TransformerLayer

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint
else:
    te_checkpoint = None


def _skip_chunks() -> int:
    """RECOMPUTE_SKIP_CHUNKS for this pipeline rank (0 when unset)."""
    import os

    spec = os.environ.get("RECOMPUTE_SKIP_CHUNKS", "")
    if not spec:
        return 0
    from megatron.core import parallel_state

    counts = [int(c) for c in spec.split(",")]
    rank = parallel_state.get_pipeline_model_parallel_rank()
    return counts[rank] if rank < len(counts) else 0


def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    rotary_pos_emb: Tensor,
    attention_bias: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    padding_mask: Optional[Tensor] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Forward method with activation checkpointing.

    Args:
        extract_layer_indices (Set[int], optional): Global layer
            indices (across all pipeline stages) from which to
            extract features.
        layer_offset (int): The global layer offset for the current
            pipeline stage. Used to convert local layer indices to
            global indices when checking extract_layer_indices.

    Returns:
        If extract_layer_indices is empty: hidden_states tensor
        If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    # Wrap non-dual RoPE to tuple to unify custom_forward interface.
    is_dual_rope = isinstance(rotary_pos_emb, (tuple, list))
    assert not is_dual_rope or len(rotary_pos_emb) == 2, "Dual RoPE input length is not equal to 2"
    rotary_pos_emb = rotary_pos_emb if is_dual_rope else (None, rotary_pos_emb)

    # Execution order, as depth positions. Without looping this is just the layer
    # order, so every index below means what it always did. With looping the same
    # layer index appears at several positions, and recompute has to walk
    # positions rather than layers -- chunking range(num_layers) would silently
    # recompute the block once instead of once per iteration.
    loop_schedule = getattr(self, "_loop_schedule", None)
    order = (
        list(range(self.num_layers_per_pipeline_rank))
        if loop_schedule is None
        else list(loop_schedule)
    )

    def custom(start: int, end: int):
        # Layers that read the micro-batch's token ids from an attribute (Engram `N`, layers
        # with uses_input_ids): the ids they hold NOW belong to this micro-batch, but by the
        # time a pipelined schedule recomputes this chunk (in its backward) later micro-batches
        # have run their forwards and replaced them. Capture them here and hand them back
        # before the recompute, or the recompute runs on another micro-batch's tokens.
        stack_ids = getattr(self, "_input_ids", None)  # HybridStack: this micro-batch's ids
        captured_ids = []
        if stack_ids is not None:
            for position in range(start, end):
                layer = self.layers[order[position]]
                # the layers HybridStack.forward hands the ids to (Engram, uses_input_ids)
                if hasattr(layer, "set_input_ids") and (
                    hasattr(layer, "_input_ids") or getattr(layer, "uses_input_ids", False)
                ):
                    captured_ids.append((layer, stack_ids))

        def custom_forward(
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb_local,
            rotary_pos_emb_global,
            padding_mask=None,
        ):
            for layer, ids in captured_ids:
                layer.set_input_ids(ids)
            rotary_pos_emb = (
                (rotary_pos_emb_local, rotary_pos_emb_global)
                if is_dual_rope
                else rotary_pos_emb_global
            )

            segmented = getattr(self, "ldm_segmented", False)
            if segmented:  # depth_memory_segments: every chunk is a segment (depth_memory_state.py)
                hidden_states = self.ldm_enter(start, end, hidden_states)
            if hidden_states.dim() == 4 and getattr(self.config, "mhc_routed_topk", 0):
                # Routed residual streams are updated in place (routed_residual.py): a chunk
                # must not write into its input, which the checkpoint saves and, in the
                # backward replay, is a detached leaf. Take ownership with one copy.
                from megatron.core.transformer.routed_residual import own_streams

                hidden_states = own_streams(hidden_states, self.config)
            for position in range(start, end):
                # Use self.layers[...] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[order[position]]

                # Get appropriate inner quantization context
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    # TODO: check if fp4 is supported in this case
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                # Build the full TransformerLayer kwarg set; for non-TL
                # layers (currently MambaLayer in HybridStack) pop the kwargs
                # they don't accept and treat the return as a single tensor.
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    else:  # MambaLayer (HybridStack `M` slot)
                        for k in ("context", "context_mask", "attention_bias", "padding_mask"):
                            layer_kwargs.pop(k, None)
                        hidden_states = layer(**layer_kwargs)
                        context = None

                # Some layer paths may still return a tuple (defensive).
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            if segmented:
                hidden_states = self.ldm_exit(start, end, hidden_states)
            return hidden_states, context

        return custom_forward

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        cf = custom(start, end)
        # Unpack the RoPE tuple as torch cannot save tuples for backward pass.
        args = (hidden_states, attention_mask, context, context_mask, *rotary_pos_emb, padding_mask)
        if use_checkpoint:
            # Precision-aware activation checkpoint: TE under FP8/FP4,
            # tensor_parallel under BF16/FP16/FP32.
            if self.config.fp8 or self.config.fp4:
                hidden_states, context = te_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    *args,
                )
            else:
                checkpoint_offload.arm()
                hidden_states, context = tensor_parallel.checkpoint(
                    cf, self.config.distribute_saved_activations, *args
                )
        else:
            # Note: original block-branch no-checkpoint path omitted padding_mask
            # (relied on its default=None); restored here for consistency.
            hidden_states, context = cf(*args)

        if self.config.recompute_method == "uniform":
            if (end - 1 + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)
        else:
            if (start + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

    checkpoint_offload.new_sequence(id(self))
    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of layers and checkpoint
        # the input activation of each divided chunk.
        # RECOMPUTE_SKIP_CHUNKS ("0,0,1,2": per pipeline rank): the last N chunks of this model
        # chunk run without a checkpoint -- their activations are kept, which only the later
        # stages (few micro-batches in flight) can afford.
        skip = _skip_chunks()
        if skip and getattr(self, "ldm_segmented", False):
            raise NotImplementedError("RECOMPUTE_SKIP_CHUNKS with depth_memory_segments")
        if skip and getattr(self, "ldm_records", False):
            # a chunk run without its checkpoint is never recomputed, so the records its input
            # carries never reach the workspace
            raise NotImplementedError("RECOMPUTE_SKIP_CHUNKS with depth_memory_unwind (cute)")
        bounds = []
        layer_idx = 0
        while layer_idx < len(order):
            chunk_end = min(layer_idx + self.config.recompute_num_layers, len(order))
            if getattr(self.config, "mhc_per_layer", False):
                # an mHC block hands its state from its first to its last layer in Python:
                # a chunk must hold whole blocks (mhc_block.py)
                from megatron.core.transformer.mhc_block import is_block_end

                while chunk_end < len(order) and not is_block_end(self.layers[order[chunk_end - 1]]):
                    chunk_end += 1
            bounds.append((layer_idx, chunk_end))
            layer_idx = chunk_end
        for i, (start, end) in enumerate(bounds):
            chunk_runner(start, end, i < len(bounds) - skip)
    elif self.config.recompute_method == 'block':
        if getattr(self.config, "mhc_per_layer", False):
            raise NotImplementedError(
                "mhc_per_layer with recompute_method=block (per-layer checkpoints would split an "
                "mHC block); use uniform"
            )
        # Checkpoint the input activation of only a set number of individual
        # layers and skip the rest. Need at least one input tensor with
        # gradient computation for the re-entrant autograd engine, so under
        # FP8/FP4 we skip checkpointing while hidden_states.requires_grad
        # is False (these slots get pushed past the recompute window).
        recompute_skip_num_layers = 0
        for layer_idx in range(len(order)):
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            use_checkpoint = (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            )
            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
    else:
        raise ValueError("Invalid activation recompute method.")

    # Return intermediate hidden states if feature extraction was requested
    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states
