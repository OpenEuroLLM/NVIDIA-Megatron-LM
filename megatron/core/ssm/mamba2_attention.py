# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mamba2 as an experimental attention variant.

Megatron ships a native Mamba2 mixer (``megatron.core.ssm.mamba_mixer.MambaMixer``)
used inside the dedicated ``MambaStack``/``MambaModel``. This thin adapter exposes
that same mixer through the *self-attention* interface used by
``TransformerLayer`` so Mamba2 can be selected as an
``experimental_attention_variant`` and interleaved with softmax-attention layers
via ``linear_attention_freq`` — the identical mechanism used by the mLSTM and
GatedDeltaNet variants. This keeps all hybrid architectures on the single GPT
decoder-block path used by the multilingual architecture comparison.
"""

from typing import Optional

from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class Mamba2Attention(MegatronModule):
    """Adapter that runs a Mamba2 ``MambaMixer`` in the self-attention slot.

    The mixer's dimensions come from the config (``mamba_state_dim``,
    ``mamba_head_dim``, ``mamba_num_groups``, ``mamba_num_heads``); the fused
    input layernorm lives in the mixer's ``in_proj`` (``fuse_input_layernorm`` is
    set on the module spec), mirroring the GatedDeltaNet variant. Mamba2 is a
    causal recurrence, so the attention mask, RoPE and attention-bias arguments
    are accepted (for interface compatibility) but ignored.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaMixerSubmodules,
        layer_number: int = None,
        pg_collection: ProcessGroupCollection = None,
        name: str | None = None,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.mixer = MambaMixer(
            config=config,
            submodules=submodules,
            d_model=config.hidden_size,
            layer_number=layer_number,
            pg_collection=pg_collection,
            name=(name + ".mixer") if name is not None else None,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """Run the Mamba2 mixer. Returns ``(output, bias)`` like an attention layer."""
        return self.mixer(
            hidden_states,
            inference_context=inference_context,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate sharded checkpointing to the wrapped mixer."""
        return self.mixer.sharded_state_dict(
            prefix=f"{prefix}mixer.", sharded_offsets=sharded_offsets, metadata=metadata
        )
