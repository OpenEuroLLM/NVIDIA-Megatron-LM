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
from megatron.core.ssm.gated_delta_net.common import (
    _build_thd_cp_a2a_perm,
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
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

    Context parallelism follows the GDN/Mamba scheme: ``B * x`` is all-to-all'd from
    sequence sharding to channel sharding, convolved over the whole sequence on this CP
    rank's channels, and all-to-all'd back; ``C`` gates afterwards, locally.

    Packed sequences (THD) restart the convolution at every document boundary. Megatron
    inference is not supported.
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
        self.cp_size = pg_collection.cp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1
        assert config.hidden_size % (self.tp_size * self.cp_size) == 0, (
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"tensor_model_parallel_size * context_parallel_size "
            f"({self.tp_size} * {self.cp_size})."
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
        thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        seq_len = hidden_states.shape[0] * self.sp_size * self.cp_size  # whole sequence
        cp_group = self.pg_collection.cp

        bcx, _ = self.in_proj(hidden_states)  # [s/cp, b, 3h/tp], sequence gathered under SP
        if getattr(self.config, "fused_short_conv", False) and not thd and self.cp_size == 1:
            from megatron.core.fusions.fused_short_conv import fused_short_conv

            bias = self.conv1d.bias if self.conv_bias else None
            return self.out_proj(fused_short_conv(bcx, self.conv1d.weight, bias))
        b, c, x = bcx.chunk(3, dim=-1)
        bx = b * x

        cu_seqlens, thd_inv = None, None
        if thd:
            assert hidden_states.shape[1] == 1, "Packed sequences expect batch size 1"
            cu_seqlens = self._resolve_cu_seqlens(packed_seq_params, seq_len)
            bx = tensor_a2a_cp2hp(
                bx, seq_dim=0, head_dim=-1, cp_group=cp_group, undo_attention_load_balancing=False
            )
            if self.cp_size > 1:
                thd_idx, thd_inv = _build_thd_cp_a2a_perm(cu_seqlens, self.cp_size, seq_len)
                bx = bx.index_select(0, thd_idx)
        else:
            bx = tensor_a2a_cp2hp(bx, seq_dim=0, head_dim=-1, cp_group=cp_group)
        # bx: [s, b, h/(tp*cp)], the whole sequence on this CP rank's channels.

        weight = get_parameter_local_cp(self.conv1d.weight, dim=0, cp_group=cp_group)
        bias = (
            get_parameter_local_cp(self.conv1d.bias, dim=0, cp_group=cp_group)
            if self.conv_bias
            else None
        )
        conv = self._causal_conv(bx, weight, bias, cu_seqlens)

        if thd:
            if self.cp_size > 1:
                conv = conv.index_select(0, thd_inv)
            conv = tensor_a2a_hp2cp(
                conv, seq_dim=0, head_dim=-1, cp_group=cp_group, redo_attention_load_balancing=False
            )
        else:
            conv = tensor_a2a_hp2cp(conv, seq_dim=0, head_dim=-1, cp_group=cp_group)
        return self.out_proj(c * conv)  # [s/cp, b, h/tp] in, [s/cp(/tp), b, h] out

    def _causal_conv(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Depthwise causal convolution of ``x`` [s, b, ch]; restarts at document starts."""
        k = self.kernel_size
        if cu_seqlens is None:
            out = F.conv1d(F.pad(x.permute(1, 2, 0), (k - 1, 0)), weight, bias, groups=x.shape[-1])
            return out.permute(2, 0, 1)
        # Packed: out[t] = sum_j w[k-1-j] * x[t-j], over the j that stay in t's document.
        s = x.shape[0]
        positions = torch.arange(s, device=x.device)
        doc = torch.bucketize(positions, cu_seqlens[1:].to(positions.dtype), right=True)
        pos_in_doc = positions - cu_seqlens[:-1].to(positions.dtype)[doc]
        out = torch.zeros_like(x)
        for j in range(k):
            shifted = F.pad(x, (0, 0, 0, 0, j, 0))[:s]
            keep = (pos_in_doc >= j).to(x.dtype).view(s, 1, 1)
            out = out + shifted * keep * weight[:, 0, k - 1 - j]
        return out if bias is None else out + bias

    def _resolve_cu_seqlens(self, packed_seq_params: PackedSeqParams, seq_len: int):
        """Document boundaries of the packed sequence, with alignment padding."""
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        if int(cu[-1].item()) != seq_len:
            raise ValueError(
                f"ShortConv: cu_seqlens[-1]={int(cu[-1])} != sequence length {seq_len}"
            )
        if ((cu[1:] - cu[:-1]) % self.cp_size != 0).any():
            raise ValueError(
                f"ShortConv: packed sequence lengths must be divisible by cp_size={self.cp_size}"
            )
        return cu

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
