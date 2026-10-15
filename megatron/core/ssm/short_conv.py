# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gated short convolution, the convolution mixer of LFM2-style conv + attention hybrids.

HybridModel places it with the `C` pattern symbol. Per token it computes

    B, C, x = in_proj(norm(h))        # three hidden_size-wide streams
    y = C * causal_depthwise_conv(B * x)
    out = out_proj(y)

which is ``Lfm2ShortConv`` in Hugging Face transformers, with Megatron's tensor-parallel
projections: every operation is per channel, so tensor parallelism shards the channels and
needs no communication beyond the projections' own.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

# The three in_proj streams, in their order within each tensor-parallel shard.
_STREAMS = ("B", "C", "x")


@dataclass
class ShortConvSubmodules:
    """Projections of the gated short convolution."""

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class ShortConv(MegatronModule):
    """Gated short convolution (LFM2) as a sequence mixer.

    Tensor parallelism shards the channels. Each rank's slice of ``in_proj`` holds its own
    ``[B | C | x]`` block, so the computation is local; ``sharded_state_dict`` splits the
    weight into the three streams so checkpoints reshard across TP sizes.

    Context parallelism is not supported: the convolution at the start of a CP chunk needs
    the last ``kernel_size - 1`` tokens of the previous one. Neither are packed sequences,
    where it would mix tokens across document boundaries, nor Megatron inference.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: ShortConvSubmodules,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        *,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for ShortConv"
        self.config = config
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_size = pg_collection.tp.size()
        if pg_collection.cp.size() != 1:
            raise NotImplementedError(
                "ShortConv does not support context parallelism: the convolution at the start "
                "of a context-parallel chunk needs the preceding tokens of the previous chunk."
            )
        assert config.hidden_size % self.tp_size == 0, (
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"tensor_model_parallel_size ({self.tp_size})."
        )
        self.hidden_size = config.hidden_size
        self.hidden_size_local_tp = config.hidden_size // self.tp_size
        self.kernel_size = config.short_conv_kernel_size
        self.conv_bias = config.short_conv_bias

        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            3 * self.hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=pg_collection.tp,
            name=(name + ".in_proj") if name is not None else None,
        )
        # Depthwise, so each channel is its own group; causal padding is applied in forward.
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size_local_tp,
            out_channels=self.hidden_size_local_tp,
            kernel_size=self.kernel_size,
            groups=self.hidden_size_local_tp,
            bias=self.conv_bias,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        for param in self.conv1d.parameters():
            setattr(param, "tensor_model_parallel", True)
            setattr(param, "partition_dim", 0)
        self.out_proj = build_module(
            submodules.out_proj,
            self.hidden_size,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=pg_collection.tp,
            name=(name + ".out_proj") if name is not None else None,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the convolution as nn.Conv1d (and LFM2) does, per tensor-parallel rank."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                self.conv1d.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ):
        """[s, b, h] -> ([s, b, h], bias)."""
        if inference_context is not None:
            raise NotImplementedError("ShortConv does not support Megatron inference.")
        if packed_seq_params is not None:
            raise NotImplementedError(
                "ShortConv does not support packed sequences: its convolution would mix "
                "tokens across document boundaries."
            )
        bcx, _ = self.in_proj(hidden_states)  # [s, b, 3h/tp], sequence gathered under SP
        b, c, x = bcx.chunk(3, dim=-1)
        bx = (b * x).permute(1, 2, 0)  # [b, h/tp, s]
        conv = F.conv1d(
            F.pad(bx, (self.kernel_size - 1, 0)),
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.hidden_size_local_tp,
        )
        y = c * conv.permute(2, 0, 1)  # [s, b, h/tp]
        return self.out_proj(y)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Shard the conv by channel and split in_proj into its B/C/x streams."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        sharded_state_dict = {}
        for name, module in self.named_children():
            if name == "conv1d":
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module.state_dict(prefix="", keep_vars=True),
                    f"{prefix}{name}.",
                    {"weight": 0, "bias": 0},
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        key = f"{prefix}in_proj.weight"
        assert sharded_state_dict[key].data.size(0) == 3 * self.hidden_size_local_tp
        sharded_state_dict[key] = _split_tensor_factory(
            sharded_state_dict[key], [self.hidden_size_local_tp] * 3, list(_STREAMS), 0
        )
        bias_key = f"{prefix}in_proj.bias"
        if bias_key in sharded_state_dict:
            sharded_state_dict[bias_key] = _split_tensor_factory(
                sharded_state_dict[bias_key], [self.hidden_size_local_tp] * 3, list(_STREAMS), 0
            )
        return sharded_state_dict
