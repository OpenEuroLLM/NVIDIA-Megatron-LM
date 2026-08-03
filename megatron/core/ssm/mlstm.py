# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from the xLSTM large model reference
# implementation (https://github.com/NX-AI/xlstm) and mirrors the structure of
# megatron/core/ssm/gated_delta_net.py.
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    # FLA's Triton causal_conv1d supports arbitrary kernel width (unlike Tri
    # Dao's CUDA causal_conv1d, which is limited to width 2-4). Shared with the
    # GatedDeltaNet conv branch.
    from fla.modules.convolution import causal_conv1d
except ImportError:
    causal_conv1d = None

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    _split_tensor_factory,
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
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
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from mlstm_kernels.torch import get_mlstm_kernel

    HAVE_MLSTM_KERNELS = True
except ImportError:
    HAVE_MLSTM_KERNELS = False

logger = logging.getLogger(__name__)


@dataclass
class MLSTMSubmodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class MLSTM(MegatronModule):
    """mLSTM (matrix LSTM, xLSTM) layer class.

    Takes input with size [s, b, h] and returns output of the same size.
    Follows the xLSTM-large / xLSTM-7B layer: fused q/k/v/output-gate
    projection, per-head input and forget gates with soft-capped
    preactivations, chunkwise-parallel mLSTM cell (mlstm_kernels TFLA),
    per-head RMSNorm, elementwise sigmoid output gate, output projection.

    Uses the shared linear-attention config fields: linear_num_key_heads must
    equal linear_num_value_heads (the kernel has no GQA), while
    linear_key_head_dim and linear_value_head_dim may differ (e.g. the
    canonical xLSTM qk_dim = v_dim / 2).

    Context parallelism uses the GDN/Mamba all-to-all scheme: sequence
    sharding is converted to head sharding around the recurrence, so each CP
    rank processes the full sequence for num_heads/(tp*cp) heads and no
    recurrent state crosses ranks.

    The first version is training-only: no packed sequences (the
    mlstm_kernels sequence kernels have no cu_seqlens interface) and no
    inference contexts.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLSTMSubmodules,
        layer_number: int = None,
        bias: bool = False,
        gate_soft_cap: Optional[float] = None,
        igate_bias_init: Optional[float] = None,
        fgate_bias_init_range: Tuple[float, float] = (3.0, 6.0),
        pg_collection: ProcessGroupCollection = None,
        name: str | None = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this mLSTM layer.
            bias: Whether to use bias in the linear layers.
            gate_soft_cap: Soft cap for the i/f gate preactivations. Defaults to
                config.mlstm_gate_soft_cap.
            igate_bias_init: Initial value of the input gate bias. Defaults to
                config.mlstm_igate_bias_init.
            fgate_bias_init_range: (low, high) of the linspace initialization of
                the forget gate bias across heads.
            pg_collection: The required process groups to use for tensor model parallel.
            name (str | None): module instance name passed top-down from its parent module.
        """

        if not HAVE_MLSTM_KERNELS:
            raise ImportError(
                "mlstm_kernels is not installed. "
                "Please install it with `pip install mlstm_kernels`."
            )

        super().__init__(config)

        self.layer_number = layer_number
        self.bias = bias
        self.igate_bias_init = (
            igate_bias_init if igate_bias_init is not None else config.mlstm_igate_bias_init
        )
        self.fgate_bias_init_range = fgate_bias_init_range
        assert pg_collection is not None, "pg_collection must be provided for MLSTM"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.tp_rank = self.pg_collection.tp.rank()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_heads = config.linear_num_value_heads
        assert config.linear_num_key_heads == config.linear_num_value_heads, (
            "MLSTM requires linear_num_key_heads == linear_num_value_heads "
            f"(got {config.linear_num_key_heads} vs {config.linear_num_value_heads}); "
            "the mLSTM kernels have no grouped-query mode."
        )
        self.qk_dim = self.key_head_dim * self.num_heads
        self.v_dim = self.value_head_dim * self.num_heads
        # Context parallelism follows the GDN/Mamba approach: an all-to-all
        # converts sequence sharding into head sharding, so each CP rank runs
        # the full-sequence recurrence for num_heads/(tp*cp) heads — no
        # recurrent state has to be transferred between ranks.
        assert self.num_heads % (self.tp_size * self.cp_size) == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"tp_size*cp_size ({self.tp_size}*{self.cp_size})."
        )
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size
        self.num_heads_local_tp = self.num_heads // self.tp_size

        self.gate_soft_cap = (
            gate_soft_cap if gate_soft_cap is not None else config.mlstm_gate_soft_cap
        )
        self.chunk_size = config.mlstm_chunk_size
        backend_name = (
            "chunkwise--native_autograd"
            if self.config.deterministic_mode
            else config.mlstm_backend
        )
        self.mlstm_fn = get_mlstm_kernel(backend_name)

        # Input projection (hidden_states -> q, k, v, output gate, igate, fgate).
        # Per-rank local layout: [q | k | v | o | i | f], each section sharded by heads.
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_heads * 2
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
            name=(name + ".in_proj") if name is not None else None,
        )

        # Per-head i/f gate biases (the fused in_proj carries no bias; the gate
        # biases are essential for the mLSTM gate initialization).
        self.igate_bias = nn.Parameter(
            torch.empty(
                self.num_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.igate_bias, "tensor_model_parallel", True)
        setattr(self.igate_bias, "partition_dim", 0)
        self.fgate_bias = nn.Parameter(
            torch.empty(
                self.num_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.fgate_bias, "tensor_model_parallel", True)
        setattr(self.fgate_bias, "partition_dim", 0)

        # Optional short depthwise causal conv over q,k,v before the mLSTM cell
        # (xLSTM conv branch / GatedDeltaNet-style). Applied in the post-all-to-all
        # full-sequence head-sharded regime, so it is CP-safe: the weight is
        # CP-sliced at runtime exactly like GatedDeltaNet. Width comes from
        # linear_conv_kernel_dim; FLA's causal_conv1d allows width > 4.
        self.use_conv1d = bool(getattr(config, "mlstm_conv1d", False))
        if self.use_conv1d:
            self.conv_kernel_dim = config.linear_conv_kernel_dim
            self.conv_activation = "silu"
            self.conv_dim = self.qk_dim * 2 + self.v_dim
            self.conv_dim_local_tp = self.conv_dim // self.tp_size
            # weight shape: [conv_dim, 1, d_conv]; depthwise (groups=channels).
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim_local_tp,
                out_channels=self.conv_dim_local_tp,
                bias=True,
                kernel_size=self.conv_kernel_dim,
                groups=self.conv_dim_local_tp,
                padding=self.conv_kernel_dim - 1,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            setattr(self.conv1d.weight, "tensor_model_parallel", True)
            setattr(self.conv1d.weight, "partition_dim", 0)
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)
        else:
            self.conv1d = None

        # Per-head RMSNorm weight over the value head dim (xLSTM MultiHeadLayerNorm
        # with use_weight=True, use_bias=False; reductions forced to fp32).
        self.out_norm_weight = nn.Parameter(
            torch.empty(
                self.v_dim_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.out_norm_weight, "tensor_model_parallel", True)
        setattr(self.out_norm_weight, "partition_dim", 0)
        self.norm_eps = config.layernorm_epsilon

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
            name=(name + ".out_proj") if name is not None else None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the mLSTM-specific parameters (gate rows/biases and norm weight)."""
        if not self.config.perform_initialization:
            return
        with get_cuda_rng_tracker().fork(), torch.no_grad():
            # Gate preactivations depend only on their biases at initialization:
            # zero the i/f gate rows of the fused input projection.
            in_proj_weight = getattr(self.in_proj, "weight", None)
            if in_proj_weight is not None:
                gate_rows = 2 * self.num_heads_local_tp
                in_proj_weight.data[-gate_rows:, :].zero_()
            else:
                logger.warning(
                    "MLSTM: in_proj has no .weight attribute; "
                    "skipping gate-row zero initialization."
                )
            self.igate_bias.data.fill_(self.igate_bias_init)
            # Forget gate bias: linspace over the *global* head index, sliced
            # per TP rank so TP does not change the initialization.
            fgate_bias_global = torch.linspace(
                self.fgate_bias_init_range[0],
                self.fgate_bias_init_range[1],
                self.num_heads,
                dtype=self.config.params_dtype,
                device=self.fgate_bias.device,
            )
            rank_slice = slice(
                self.tp_rank * self.num_heads_local_tp,
                (self.tp_rank + 1) * self.num_heads_local_tp,
            )
            self.fgate_bias.data.copy_(fgate_bias_global[rank_slice])
            self.out_norm_weight.data.fill_(1.0)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the mLSTM module.

        Args:
            hidden_states (Tensor): Hidden states [s, b, h].
            attention_mask (Tensor): Unused (the mLSTM is causal by construction).
            inference_context (Optional[BaseInferenceContext]): Not supported yet.
            packed_seq_params (Optional[PackedSeqParams]): Not supported (the
                mlstm_kernels sequence kernels have no cu_seqlens interface).
            sequence_len_offset (Optional[int]): Unused.

        Return:
            (Tuple[Tensor, Tensor]) mLSTM output and bias.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:
            raise NotImplementedError("MLSTM does not support inference for now.")
        assert packed_seq_params is None, "MLSTM does not support packed sequences."

        batch = hidden_states.shape[1]

        # Input projection; with sequence parallelism the sequence dimension is
        # gathered inside.
        nvtx_range_push(suffix="in_proj")
        fused, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # CP all-to-all: sequence sharding -> head sharding (full sequence,
        # num_heads/(tp*cp) heads per rank).
        fused = tensor_a2a_cp2hp(
            fused,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            split_sections=[
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_heads_local_tp,
                self.num_heads_local_tp,
            ],
        )
        seq_len = fused.shape[0]

        # From sbd to bsd, then split into q, k, v, ogate, igate, fgate.
        fused = fused.transpose(0, 1)
        q, k, v, o_preact, i_preact, f_preact = torch.split(
            fused,
            [
                self.qk_dim_local_tp // self.cp_size,
                self.qk_dim_local_tp // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.num_heads_local_tp // self.cp_size,
                self.num_heads_local_tp // self.cp_size,
            ],
            dim=-1,
        )

        # Short depthwise causal conv over q,k,v (xLSTM conv branch), applied on
        # the full sequence in the head-sharded regime; CP-slice the weight per
        # rank exactly like GatedDeltaNet, so no cross-CP-rank conv state is
        # needed.
        if self.conv1d is not None:
            nvtx_range_push(suffix="mlstm_conv1d")
            qkv = torch.cat([q, k, v], dim=-1)  # b, s, (2*qk + v)/cp
            qkv_channels_split_sections = [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
            ]
            conv1d_weight = get_parameter_local_cp(
                self.conv1d.weight,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            conv1d_bias = get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.config.deterministic_mode:
                qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
                conv_out = F.conv1d(
                    input=qkv,
                    weight=conv1d_weight,
                    bias=conv1d_bias,
                    stride=self.conv1d.stride,
                    padding=self.conv1d.padding,
                    dilation=self.conv1d.dilation,
                    groups=self.conv_dim_local_tp // self.cp_size,
                )
                qkv = F.silu(conv_out[..., :seq_len])
                qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
            else:
                assert causal_conv1d is not None, (
                    "mlstm_conv1d=True requires flash-linear-attention's causal_conv1d"
                )
                qkv, _ = causal_conv1d(
                    x=qkv,  # FLA conv1d accepts [b, s, d]
                    weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                    bias=conv1d_bias,
                    activation=self.conv_activation,
                    initial_state=None,
                    output_final_state=False,
                    cu_seqlens=None,
                )
            q, k, v = torch.split(
                qkv,
                [
                    self.qk_dim_local_tp // self.cp_size,
                    self.qk_dim_local_tp // self.cp_size,
                    self.v_dim_local_tp // self.cp_size,
                ],
                dim=-1,
            )
            nvtx_range_pop(suffix="mlstm_conv1d")

        igate_bias_local_cp = get_parameter_local_cp(
            self.igate_bias, dim=0, cp_group=self.pg_collection.cp
        )
        fgate_bias_local_cp = get_parameter_local_cp(
            self.fgate_bias, dim=0, cp_group=self.pg_collection.cp
        )
        q, k, v, i_preact, f_preact = self._prepare_qkvif(
            q, k, v, i_preact, f_preact, igate_bias_local_cp, fgate_bias_local_cp, batch, seq_len
        )

        nvtx_range_push(suffix="mlstm_cell")
        h = self.mlstm_fn(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
            return_last_states=False,
            chunk_size=self.chunk_size,
        )
        nvtx_range_pop(suffix="mlstm_cell")

        # Per-head norm and elementwise output gate.
        nvtx_range_push(suffix="gated_norm")
        out_norm_weight_local_cp = get_parameter_local_cp(
            self.out_norm_weight, dim=0, cp_group=self.pg_collection.cp
        )
        out = self._apply_gated_norm(h, o_preact, out_norm_weight_local_cp, batch, seq_len)
        nvtx_range_pop(suffix="gated_norm")

        # From bsd back to sbd.
        out = out.transpose(0, 1).contiguous()

        # CP all-to-all: head sharding -> sequence sharding.
        out = tensor_a2a_hp2cp(out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _prepare_qkvif(self, q, k, v, i_preact, f_preact, igate_bias, fgate_bias, batch, seq_len):
        """Reshape projections to kernel layout and soft-cap the gate preactivations.

        Returns q/k [b, nh, s, dhqk], v [b, nh, s, dhhv], i/f [b, nh, s] (fp32).
        """
        q = q.reshape(batch, seq_len, -1, self.key_head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.reshape(batch, seq_len, -1, self.key_head_dim)
        k = k.transpose(1, 2).contiguous()
        v = v.reshape(batch, seq_len, -1, self.value_head_dim)
        v = v.transpose(1, 2).contiguous()

        cap = self.gate_soft_cap
        i_preact = i_preact.float() + igate_bias.float()
        f_preact = f_preact.float() + fgate_bias.float()
        if cap is not None:
            i_preact = cap * torch.tanh(i_preact / cap)
            f_preact = cap * torch.tanh(f_preact / cap)
        i_preact = i_preact.transpose(1, 2).contiguous()
        f_preact = f_preact.transpose(1, 2).contiguous()
        return q, k, v, i_preact, f_preact

    @jit_fuser
    def _apply_gated_norm(self, h, o_preact, out_norm_weight, batch, seq_len):
        """Per-head RMSNorm (fp32 reductions) followed by the sigmoid output gate."""
        h_dtype = h.dtype
        # h: [b, nh, s, dhhv] -> [b, s, nh, dhhv]
        h = h.transpose(1, 2)
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        h = h.reshape(batch, seq_len, -1) * out_norm_weight.float()
        h = h * torch.sigmoid(o_preact.float())
        return h.to(h_dtype)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "igate_bias": 0,
                "fgate_bias": 0,
                "out_norm_weight": 0,
            },
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            module_sharded_sd = sharded_state_dict_default(
                module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
            )
            sharded_state_dict.update(module_sharded_sd)

        # Split the fused in_proj into independently reshardable sections.
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_heads_local_tp,
                self.num_heads_local_tp,
            ],
            ["query", "key", "value", "ogate", "igate", "fgate"],
            0,
        )

        return sharded_state_dict

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        self.in_proj.backward_dw()
        self.out_proj.backward_dw()
