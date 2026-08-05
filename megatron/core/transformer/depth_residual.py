# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Shared depth-wise residual state for mHC and Block Attention Residuals.

Both schemes replace the single ``[s, b, h]`` residual stream with a wider,
depth-spanning state that has to be threaded through every layer of a
:class:`~megatron.core.transformer.transformer_block.TransformerBlock`:

* **mHC** carries ``n`` parallel streams, ``[s, b, n, h]``.
* **Block AttnRes** carries ``N`` completed block representations plus the partial
  sum of the block currently being filled, i.e. ``N + 1`` tensors of ``[s, b, h]``.

Both payloads are therefore ``O(N h)`` per token, which is what makes the block
count of AttnRes play the same role as mHC's expansion rate.

The block owns one :class:`DepthResidualState` per forward pass and stashes it on
each layer before calling it.

.. note:: **CUDA-graph compatibility (outstanding).**
   Every arm that reaches >200 TFLOP/s at 130M uses CUDA graphs; mHC and Block
   AttnRes cannot, and that is the dominant remaining cost for them. Two things
   block capture, one of which is now fixed:

   1. *(fixed)* mHC cached its connection matrices on this object between the
      layer's read and write halves. The coefficients are now recomputed in the
      write, which is bit-identical because ``streams`` is unchanged in between.
   2. *(outstanding)* The layer reaches this object through
      ``layer._depth_state``, and ``blocks`` is a Python list that grows by
      ``append``. Capture records tensor *operations*, so the tensor identities
      read through the attribute are baked in and a replay would run against
      stale memory.

   The fix for (2) is to allocate the sources once into a fixed
   ``[N + 1, s, b, h]`` buffer at stable addresses and have each layer write its
   slice, so replays hit the same memory. Shapes are already static per layer
   (layer ``i`` always sees the same source count), so no shape specialization is
   needed -- only the storage and the plumbing. The autograd interaction is the
   delicate part: slice-writes into a shared buffer must not bump the version
   counter of a tensor the graph still needs, which is the same failure class as
   the 0-dim scalar bug fixed earlier in this branch. Under pipeline parallelism the payload must also
cross stage boundaries; :meth:`DepthResidualState.to_payload` and
:meth:`DepthResidualState.from_payload` provide the flat tensor for that transfer.
"""

import torch


class DepthResidualState:
    """Depth-spanning residual state for one forward pass.

    Exactly one of the two modes is active, selected by ``mode``.

    Args:
        mode: ``"mhc"`` or ``"attn_res"``.
        hidden_states: The block's input hidden states, ``[s, b, h]``.
        expansion_rate: Number of mHC streams (ignored for ``attn_res``).
        layers_per_block: Transformer layers per AttnRes block (ignored for mHC).
    """

    def __init__(
        self,
        mode: str,
        hidden_states: torch.Tensor,
        expansion_rate: int = 4,
        layers_per_block: int = 1,
    ) -> None:
        if mode not in ("mhc", "attn_res"):
            raise ValueError(f"mode must be 'mhc' or 'attn_res', got {mode!r}")
        self.mode = mode
        self.layers_per_block = layers_per_block
        # mHC state
        self.streams: torch.Tensor | None = None
        self.pending = None
        # Block AttnRes state
        self.blocks: list[torch.Tensor] = []
        self.partial: torch.Tensor | None = None

        if mode == "mhc":
            from megatron.core.transformer.mhc import expand_to_streams

            self.streams = expand_to_streams(hidden_states, expansion_rate)
        else:
            # b_0 is the token embedding, always retained as an attention source.
            self.blocks = [hidden_states]

    # ---------------------------------------------------------------- AttnRes
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
        if self.mode != "attn_res" or self.layers_per_block <= 0:
            return False
        return layer_number % self.layers_per_block == 0

    # ------------------------------------------------------------ collapse/PP
    def collapse(self) -> torch.Tensor:
        """Reduce the state back to a single ``[s, b, h]`` hidden state.

        For mHC this averages the streams (their read map is a convex
        combination, so the mean is the magnitude-preserving collapse). For Block
        AttnRes it returns the final accumulated representation.
        """
        if self.mode == "mhc":
            from megatron.core.transformer.mhc import collapse_streams

            return collapse_streams(self.streams)
        if self.partial is not None:
            return self.partial
        return self.blocks[-1]

    def to_payload(self) -> torch.Tensor:
        """Flatten the state into one tensor for a pipeline-stage transfer.

        Returns:
            ``[n, s, b, h]`` for mHC (streams moved to the front), or
            ``[k, s, b, h]`` for Block AttnRes (blocks then partial).
        """
        if self.mode == "mhc":
            # [s, b, n, h] -> [n, s, b, h]
            return self.streams.permute(2, 0, 1, 3).contiguous()
        return torch.stack(self.sources(), dim=0)

    def load_payload(self, payload: torch.Tensor, has_partial: bool = True) -> None:
        """Restore the state from a payload received across a stage boundary.

        Args:
            payload: The tensor produced by :meth:`to_payload`.
            has_partial: Block AttnRes only -- whether the last slice is an
                in-progress partial sum rather than a completed block.
        """
        if self.mode == "mhc":
            self.streams = payload.permute(1, 2, 0, 3).contiguous()
            return
        tensors = list(torch.unbind(payload, dim=0))
        if has_partial:
            self.blocks, self.partial = tensors[:-1], tensors[-1]
        else:
            self.blocks, self.partial = tensors, None

    def payload_slots(self, num_blocks: int) -> int:
        """Number of ``[s, b, h]`` slices the pipeline payload occupies."""
        if self.mode == "mhc":
            return self.streams.shape[2]
        return num_blocks + 1
