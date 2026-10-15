# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Depth-wise residual state for Block Attention Residuals.

Block AttnRes replaces the single ``[s, b, h]`` residual stream with a set of
``N`` completed block representations plus the partial sum of the block currently
being filled -- ``N + 1`` tensors of ``[s, b, h]`` that have to be threaded
through every layer of a
:class:`~megatron.core.transformer.transformer_block.TransformerBlock`. The
payload is therefore ``O(N h)`` per token, which is what makes the block count
play the same role as mHC's expansion rate.

The block owns one :class:`DepthResidualState` per forward pass and stashes it on
each layer before calling it.

.. note:: **Why mHC does not use this class.**
   mHC's state is a single fixed-shape ``[s, b, n, h]`` tensor, so it travels as
   the layer's own ``hidden_states`` instead of through an attribute. That
   matters for throughput: CUDA graph capture records tensor *operations*, and
   :class:`~megatron.core.transformer.cuda_graphs.CudaGraphManager` only
   staticizes tensors it can see in a layer's args and kwargs. State reached
   through ``layer._depth_state`` has its address baked into the captured graph,
   so a replay reads whichever buffer was live at capture time. Graphs are worth
   +19-26% here, so mHC passes its streams in and out explicitly.

   Block AttnRes cannot do the same as cheaply. Its sources are a list that grows
   with depth, and flattening it into one tensor per layer costs a full copy of
   the whole payload -- measurably worse than the loop form it replaced. Making
   it capturable needs a preallocated ``[N + 1, s, b, h]`` buffer at a stable
   address with each layer writing its own slice. Shapes are already static per
   layer (layer ``i`` always sees the same source count), so no shape
   specialization is needed -- only the storage and the plumbing. The delicate
   part is autograd: slice-writes into a shared buffer must not bump the version
   counter of a tensor the graph still needs.

Under pipeline parallelism the payload must also cross stage boundaries;
:meth:`DepthResidualState.to_payload` and :meth:`DepthResidualState.load_payload`
provide the flat tensor for that transfer. (mHC needs no such helper -- its
payload is the hidden state itself, so only the pipeline shape negotiation has to
learn the wider shape.)
"""

import torch


def depth_payload_slots(config) -> int:
    """How many ``[s, b, h]`` slices the inter-stage payload occupies.

    Both depth-wise schemes widen what crosses a pipeline stage boundary, and
    the pipeline schedules need that width up front to size their receive
    buffers. Returns 1 when neither scheme is active, i.e. the ordinary residual.

    Args:
        config: A :class:`~megatron.core.transformer.transformer_config.TransformerConfig`.

    Returns:
        The multiplier on ``hidden_size`` for the inter-stage tensor.
    """
    if getattr(config, "mhc", False):
        return config.mhc_expansion_rate
    if getattr(config, "attn_res", False):
        # b_0 (the embedding) plus one slice per completed block. Mid-block the
        # in-progress partial sum takes the slot of the block it will become, so
        # this is the maximum over every boundary, not just the last one.
        return config.attn_res_num_blocks + 1
    return 1


def pack_payload(sources: list[torch.Tensor], slots: int) -> torch.Tensor:
    """Flatten Block AttnRes sources into one ``[s, b, slots * h]`` tensor.

    Padded to a fixed width so every stage boundary ships the same shape, which
    is what lets the schedules compute one tensor shape for the whole pipeline.
    Unused slots are zero and are never read: the receiving stage derives how
    many are live from its own layer offset, which is static.

    The payload is deliberately kept 3-D. The p2p layer exchanges shapes as a
    3-element tensor when ``variable_seq_lengths`` is set, so a rank-4 payload
    would be silently truncated there.

    Args:
        sources: The ``[s, b, h]`` tensors to ship, at most ``slots`` of them.
        slots: Fixed payload width from :func:`depth_payload_slots`.

    Returns:
        ``[s, b, slots * h]``.
    """
    if len(sources) > slots:
        raise ValueError(f"payload has {len(sources)} sources but only {slots} slots")
    s, b, h = sources[0].shape
    payload = sources[0].new_zeros((s, b, slots, h))
    for i, src in enumerate(sources):
        payload[:, :, i] = src
    return payload.reshape(s, b, slots * h)


def unpack_payload(payload: torch.Tensor, slots: int, live: int) -> list[torch.Tensor]:
    """Recover the live ``[s, b, h]`` slices from :func:`pack_payload`.

    Args:
        payload: ``[s, b, slots * h]`` tensor received from the previous stage.
        slots: Fixed payload width.
        live: How many leading slices carry real data.

    Returns:
        A list of ``live`` tensors of shape ``[s, b, h]``.
    """
    s, b, wide = payload.shape
    view = payload.reshape(s, b, slots, wide // slots)
    return [view[:, :, i] for i in range(live)]


class DepthResidualState:
    """Block Attention Residual state for one forward pass.

    Args:
        hidden_states: The block's input hidden states, ``[s, b, h]``.
        layers_per_block: Transformer layers per AttnRes block.
    """

    def __init__(self, hidden_states: torch.Tensor, layers_per_block: int = 1) -> None:
        self.layers_per_block = layers_per_block
        # b_0 is the token embedding, always retained as an attention source.
        self.blocks: list[torch.Tensor] = [hidden_states]
        self.partial: torch.Tensor | None = None

    def sources(self) -> list[torch.Tensor]:
        """Return the value tensors Block AttnRes attends over."""
        if self.partial is None:
            return self.blocks
        return self.blocks + [self.partial]

    def seal_block(self) -> None:
        """Promote the in-progress partial sum to a completed block representation."""
        if self.partial is not None:
            self.blocks.append(self.partial)
            self.partial = None

    def should_seal(self, layer_number: int) -> bool:
        """True when ``layer_number`` closes a block.

        Args:
            layer_number: 1-based transformer layer index within the model.
        """
        if self.layers_per_block <= 0:
            return False
        return layer_number % self.layers_per_block == 0

    def collapse(self) -> torch.Tensor:
        """Reduce the state back to a single ``[s, b, h]`` hidden state."""
        if self.partial is not None:
            return self.partial
        return self.blocks[-1]

    def to_payload(self) -> torch.Tensor:
        """Flatten the state into one ``[k, s, b, h]`` tensor for a stage transfer."""
        return torch.stack(self.sources(), dim=0)

    def load_payload(self, payload: torch.Tensor, has_partial: bool = True) -> None:
        """Restore the state from a payload received across a stage boundary.

        Args:
            payload: The tensor produced by :meth:`to_payload`.
            has_partial: Whether the last slice is an in-progress partial sum
                rather than a completed block.
        """
        tensors = list(torch.unbind(payload, dim=0))
        if has_partial:
            self.blocks, self.partial = tensors[:-1], tensors[-1]
        else:
            self.blocks, self.partial = tensors, None

    def payload_slots(self, num_blocks: int) -> int:
        """Number of ``[s, b, h]`` slices the pipeline payload occupies."""
        return num_blocks + 1
