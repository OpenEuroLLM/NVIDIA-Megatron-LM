# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import annotate_first_last_layer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import build_loop_schedule
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    log_single_rank,
    make_viewless_tensor,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridStackSubmodules:
    """
    A class for the module specs for the HybridStack.
    """

    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    gdn_layer: Union[ModuleSpec, type] = IdentityOp
    # `L` positions. None derives the layer from config.experimental_attention_variant
    # (get_hybrid_linear_layer_spec), so stack specs need not name a mixer.
    linear_layer: Optional[ModuleSpec] = None
    short_conv_layer: Union[ModuleSpec, type] = IdentityOp
    engram_layer: Union[ModuleSpec, type] = IdentityOp
    pkm_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    dsa_layer: Union[ModuleSpec, type] = IdentityOp
    mla_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None


class HybridStack(MegatronModule):
    """
    Constructor for the HybridStack class.

    Args:
        config (TransformerConfig): the model configuration
        submodules (HybridStackSubmodules): the submodules for the stack
        pre_process (bool, optional): whether to include an embedding layer.
            Defaults to True.
        layer_type_list (list, optional): pre-computed list of layer type symbols for
            this pipeline segment. When provided (by HybridModel), pipeline stage
            selection has already been done via '|' separators in the pattern.
        pp_layer_offset (int, optional): the global layer offset for this pipeline
            segment. Defaults to 0.
        post_layer_norm (bool, optional): whether to include a final layer norm.
            Defaults to True.
        post_process (bool, optional): whether to include an output layer.
            Defaults to True.
        device (optional): the device to use. Defaults to None.
        dtype (optional): the data type to use. Defaults to None.
        pg_collection (ProcessGroupCollection): the required model communication
            process groups to use.
        is_mtp_layer (bool, optional): whether this is an MTP layer. Defaults to False.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: Optional[list[str]] = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)
        self.pre_process = pre_process
        self.pp_layer_offset = pp_layer_offset
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer

        assert pg_collection is not None, "pg_collection must be provided for HybridStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp

        # Required for pipeline parallel schedules
        self.input_tensor = None
        self.pg_collection = pg_collection

        assert layer_type_list is not None, (
            "layer_type_list must be provided. It should be pre-computed from "
            "--hybrid-layer-pattern by HybridModel."
        )
        self.layer_type_list = layer_type_list

        if getattr(self.config, "mla_down_proj_fusion", False):
            submodules = self._fuse_mla_down_proj(submodules)

        # Build layers from the pre-selected segment
        self.layers = nn.ModuleList()
        self._linear_spec = None  # built on the first `L` position, see _linear_layer_spec
        self.per_layer_embeddings = None  # see attach_token_embeddings
        self._input_ids = None
        for i, layer_type in enumerate(self.layer_type_list):
            layer_number = i + 1 + pp_layer_offset
            if self.config.fp8:
                quant_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            elif self.config.fp4:
                quant_init_context = get_fp4_context(self.config, i + pp_layer_offset, is_init=True)
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type in (LayerSymbols.ATTENTION, LayerSymbols.SWA_ATTENTION):
                    # `W` is the same layer; its window comes from config.window_size and
                    # the per-layer window_attn_skip_freq HybridModel derives from the pattern.
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.DS_ATTENTION:
                    layer = build_module(
                        submodules.dsa_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.MLA:
                    layer = build_module(
                        submodules.mla_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.GDN:
                    layer = build_module(
                        submodules.gdn_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.ENGRAM:
                    layer = build_module(
                        submodules.engram_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.PRODUCT_KEY_MEMORY:
                    layer = build_module(
                        submodules.pkm_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.SHORT_CONV:
                    layer = build_module(
                        submodules.short_conv_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.LINEAR:
                    layer = build_module(
                        self._linear_layer_spec(submodules),
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.TRIADIC:
                    layer = build_module(
                        self._triadic_layer_spec(),
                        config=self._triadic_config(),
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                else:
                    raise ValueError("unexpected layer_type")
            self.layers.append(layer)

        if self.config.cuda_graph_impl == "local":
            annotate_first_last_layer(self.layers)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        # Linear Depth Memory: the embedding write and the final read (depth_memory.py)
        from megatron.core.transformer.depth_memory import ldm_streams

        from megatron.core.transformer.attn_res import attn_res_streams

        self.ldm_boundary = None
        self.ldm_stream_boundary = None
        self.attnres_stream_boundary = None
        if attn_res_streams(self.config):
            self._wire_attn_res_streams()
        if ldm_streams(self.config):
            self._wire_depth_memory()
        elif getattr(self.config, "depth_memory", False):
            from megatron.core.transformer.depth_memory import DepthMemoryBoundary

            self.ldm_boundary = DepthMemoryBoundary(self.config)
            for param in self.ldm_boundary.parameters():
                param.sequence_parallel = self.config.sequence_parallel

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        # Looped depth: the middle block is applied loop_iterations times. Shares
        # build_loop_schedule with TransformerBlock so the two stacks cannot drift
        # apart on what a given (prelude, block, iterations, group, coda) means.
        self._loop_schedule = build_loop_schedule(self.config)
        if self._loop_schedule is not None:
            self._validate_loop_schedule()
            # Logged unconditionally: a looped run that silently fell back to a plain
            # stack is indistinguishable from a correct one in the loss curve, and
            # whether the flags reach Megatron at all depends on plumbing outside
            # this repo. This line is the positive confirmation that it is active.
            looped_types = ''.join(self.layer_type_list[i] for i in self._loop_schedule)
            log_single_rank(
                logger,
                logging.INFO,
                f"Looped depth active: {len(self.layers)} unique layers applied as "
                f"{len(self._loop_schedule)} (prelude {self.config.loop_prelude_layers}, "
                f"block {self.config.loop_block_layers} x {self.config.loop_iterations}, "
                f"group {self.config.loop_group_size or self.config.loop_block_layers}, "
                f"coda {self.config.loop_coda_layers}); "
                f"pattern='{''.join(self.layer_type_list)}' -> '{looped_types}'; "
                f"schedule={self._loop_schedule}",
            )

    def _validate_loop_schedule(self) -> None:
        """Reject loop configurations this stack cannot execute correctly.

        Recompute and CUDA graphs both work under a loop schedule and are not
        checked here -- recompute walks the schedule, and CudaGraphManager already
        handles a module called several times per step, which is what gradient
        accumulation does.

        What is checked is specific to a hybrid stack. ``build_loop_schedule`` addresses
        layers by index, and in a hybrid stack the index also selects a layer
        *type*, so repeating a block splices its last layer type directly onto its
        first. A block is not required to be homogeneous -- the natural hybrid
        block is an alternating ``M-M-M-`` and repeating it is exactly the point --
        but a block that begins and ends on the same type produces a doubled type
        at every seam, which is nearly always an off-by-one in the boundaries
        rather than an intent. That is a warning, not an error: it is a design
        smell, and the caller may have meant it.
        """
        prelude = self.config.loop_prelude_layers
        block = self.config.loop_block_layers
        if len(self.layer_type_list) != len(self.layers):
            raise ValueError(
                f"Looped depth expects one layer type per built layer, got "
                f"{len(self.layer_type_list)} types for {len(self.layers)} layers."
            )
        if self.config.loop_iterations > 1 and block > 1:
            first, last = self.layer_type_list[prelude], self.layer_type_list[prelude + block - 1]
            if first == last:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Looped block [{prelude}, {prelude + block}) of pattern "
                    f"'{''.join(self.layer_type_list)}' both starts and ends on "
                    f"'{first}', so every loop seam applies two '{first}' layers in a "
                    "row. If that is not intended, shift the loop boundaries so the "
                    "block covers whole repeats of the pattern period.",
                )

    def _linear_layer_spec(self, submodules: HybridStackSubmodules) -> ModuleSpec:
        """The spec for `L` positions: the stack spec's, else the configured variant's."""
        if submodules.linear_layer is not None:
            return submodules.linear_layer
        # Imported here: the GPT spec module imports the SSM mixers, which import
        # from this package.
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_hybrid_linear_layer_spec,
        )

        if self._linear_spec is None:
            self._linear_spec = get_hybrid_linear_layer_spec(self.config)
        return self._linear_spec

    def _triadic_config(self):
        """The stack config with experimental_attention_variant = "triadic", for `T` layers.

        A shallow copy (no __post_init__ re-run): every other field, including the triadic
        ones (linear_triadic_base / _e / _kernel), is shared with the stack.
        """
        if getattr(self, "_triadic_cfg", None) is None:
            cfg = copy.copy(self.config)
            cfg.experimental_attention_variant = "triadic"
            self._triadic_cfg = cfg
        return self._triadic_cfg

    def _triadic_layer_spec(self) -> ModuleSpec:
        """The spec for `T` positions: the triadic linear mixer."""
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_hybrid_linear_layer_spec,
        )

        if getattr(self, "_triadic_spec", None) is None:
            self._triadic_spec = get_hybrid_linear_layer_spec(self._triadic_config())
        return self._triadic_spec

    def _fuse_mla_down_proj(self, submodules: HybridStackSubmodules) -> HybridStackSubmodules:
        # Avoid modifying the original object so users don't get surprised about their `submodules`
        # being modified underneath them.
        submodules = copy.deepcopy(submodules)
        mla_spec = submodules.mla_layer
        # We always fuse the input layernorm because Hybrid always uses TransformerEngine.
        mla_spec.submodules.input_layernorm = IdentityOp
        mla_spec.submodules.self_attention.module = FusedMLASelfAttention
        mla_spec.submodules.self_attention.submodules.linear_qkv_down_proj = (
            TELayerNormColumnParallelLinear
        )
        mla_spec.submodules.self_attention.submodules.linear_q_down_proj = None
        mla_spec.submodules.self_attention.submodules.linear_kv_down_proj = None
        mla_spec.submodules.sharded_state_dict_keys_map = {
            "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
        }
        return submodules

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """
        Returns the Mamba conv and ssm states shapes per input sequence
        if this block contains Mamba layers (this may not be the case with PP > 1).
        """
        for layer_type, layer in zip(self.layer_type_list, self.layers):
            if layer_type == LayerSymbols.MAMBA:
                return layer.mamba_state_shapes_per_request()
        return None

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask=None,
        input_ids: Optional[Tensor] = None,
    ):
        """
        Forward function of the HybridStack class.

        It either returns the Loss values if labels are given or the
            final hidden units

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): the input tensor.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): the attention mask.
            inference_context (BaseInferenceContext): the inference parameters.
            rotary_pos_emb (Tensor, optional): the rotary positional embeddings.
                Defaults to None.
        Returns:
            Tensor: the output tensor.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Engram (`N`) layers look up hashed n-grams of the token ids, value embeddings
        # and per-layer embeddings plain token tables. Handing the ids over here, rather
        # than through each layer call, keeps activation recompute and the other layer
        # types unaware of them.
        for layer_type, layer in zip(self.layer_type_list, self.layers):
            if layer_type == LayerSymbols.ENGRAM or getattr(layer, "uses_input_ids", False):
                layer.set_input_ids(input_ids)
        self._input_ids = input_ids

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if inference_context and inference_context.is_static_batching():
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (self.config.cuda_graph_impl == "local" or self.config.flash_decode)
            and inference_context
            and inference_context.is_static_batching()
            and InferenceMode.is_active()
        ):
            current_batch_size = hidden_states.shape[1]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device='cuda',
            )
        else:
            sequence_len_offset = None

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        use_fp4_context = self.config.fp4 is not None
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        if use_inner_fp8_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp8_context(config, layer_number)

        elif use_fp4_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp4_context(config, layer_number)

        else:

            def get_inner_quant_context(config, layer_number):
                return nullcontext()

        with outer_fp8_context:
            depth_scheme = getattr(self.config, "mhc", False) or getattr(
                self.config, "attn_res", False
            )
            if (
                getattr(self, "per_layer_embeddings", None) is not None
                and self.config.recompute_granularity == 'full'
                and self.training
            ):
                raise NotImplementedError("per-layer embeddings with full activation recompute")
            full_recompute = self.config.recompute_granularity == 'full' and self.training
            if depth_scheme and full_recompute and getattr(self.config, "attn_res", False):
                # Block AttnRes keeps its block summaries in a Python state the layers mutate
                # (layer._depth_state), which a recomputed chunk would replay against the
                # already-advanced state.
                raise NotImplementedError(
                    "Block AttnRes does not support full activation recompute; use "
                    "selective recompute."
                )
            if depth_scheme and full_recompute and self.config.distribute_saved_activations:
                raise NotImplementedError(
                    "mHC with full recompute does not support distribute_saved_activations"
                )
            # Routed residual streams: recompute per sublayer instead (each TransformerLayer
            # checkpoints its sublayer body between the routed read and the in-place write),
            # so nothing n streams wide is saved per layer -- the layer-chunk checkpoints below
            # would save (and copy) the whole stream state at every chunk.
            routed_sublayer = full_recompute and getattr(self.config, "mhc_routed_topk", 0) > 0
            for layer in self.layers:
                if isinstance(layer, TransformerLayer):
                    layer.routed_sublayer_recompute = routed_sublayer
            if full_recompute and not routed_sublayer:
                # mHC (dense or routed): the widened residual is a plain [s, b, n, h] tensor,
                # so each checkpointed chunk saves and recomputes it like the [s, b, h]
                # residual; enter / exit (widen, or own the incoming stage payload; collapse,
                # or pack for the next stage) stay outside the checkpoints.
                hidden_states, _ = self._depth_enter(hidden_states)
                hidden_states = checkpointed_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=None,
                    context_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                    use_inner_quantization_context=(use_inner_fp8_context or use_fp4_context),
                )
                hidden_states = self._depth_exit(hidden_states, None)
            else:
                if self._loop_schedule is None:
                    order = list(range(len(self.layers)))
                else:
                    # Indexes parameters, not depth: a looped layer is the same
                    # module applied at several positions, so layer.layer_number
                    # repeats. Only the quantization context reads it, and that is
                    # keyed per unique layer by design.
                    order = list(self._loop_schedule)
                layer_order = [self.layers[i] for i in order]
                # mHC widens hidden_states to [s, b, n, h] streams; Block AttnRes keeps
                # its block summaries in a state stashed on every layer.
                hidden_states, depth_state = self._depth_enter(hidden_states)
                for layer_index, layer in zip(order, layer_order):
                    if depth_state is not None:
                        layer._depth_state = depth_state
                    # Layers have 1-indexed layer numbers attribute.
                    inner_quant_context = get_inner_quant_context(
                        self.config, layer.layer_number - 1
                    )
                    with inner_quant_context:
                        if isinstance(layer, TransformerLayer):
                            hidden_states, _ = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                rotary_pos_emb=rotary_pos_emb,
                                sequence_len_offset=sequence_len_offset,
                                packed_seq_params=packed_seq_params,
                                padding_mask=padding_mask,
                            )
                        else:  # MambaLayer, Expert, or MLP
                            hidden_states = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                packed_seq_params=packed_seq_params,
                            )

                    # The attention layer (currently a simplified transformer layer)
                    # outputs a tuple of (hidden_states, context). Context is intended
                    # for cross-attention, and is not needed in our model.
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    if self.ldm_boundary is not None and hidden_states is not depth_state.stream:
                        raise RuntimeError(
                            f"depth_memory: {type(layer).__name__} kept its own residual "
                            "(it has no depth-memory read / write)"
                        )
                    hidden_states = self._apply_per_layer_embedding(layer_index, hidden_states)
                hidden_states = self._depth_exit(hidden_states, depth_state)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        # Ensure that the tensor passed between pipeline parallel stages is
        # viewless. See related notes in TransformerBlock and TransformerLayer
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return hidden_states

    def attach_token_embeddings(self, vocab_size: int) -> None:
        """Build the per-layer embeddings (PLE) of this stage's layers and the value
        embeddings of its attention layers (token_embeddings.py)."""
        from megatron.core.transformer.token_embeddings import PerLayerEmbedding

        self.per_layer_embeddings = None
        dim = getattr(self.config, "per_layer_embedding_dim", 0)
        if dim:
            if getattr(self.config, "mhc", False) or getattr(self.config, "attn_res", False) \
                    or getattr(self.config, "ngpt", False) \
                    or getattr(self.config, "depth_memory", False):
                raise NotImplementedError(
                    "per-layer embeddings with mHC / AttnRes / nGPT / depth memory"
                )
            which = self.config.per_layer_embedding_layers
            if which not in ("mlp", "all"):
                raise ValueError(f"per_layer_embedding_layers must be mlp or all, got {which}")
            mlp = (LayerSymbols.MLP, LayerSymbols.MOE)
            self.per_layer_embeddings = nn.ModuleList(
                [
                    PerLayerEmbedding(self.config, vocab_size, self.tp_group)
                    if which == "all" or t in mlp
                    else None
                    for t in self.layer_type_list
                ]
            )
        if getattr(self.config, "value_embedding", False):
            for layer in self.layers:
                attention = getattr(layer, "self_attention", None)
                if hasattr(attention, "attach_value_embedding"):
                    attention.attach_value_embedding(vocab_size)
                    layer.uses_input_ids = True

    def _apply_per_layer_embedding(self, i: int, hidden_states: Tensor) -> Tensor:
        ple = self.per_layer_embeddings[i] if self.per_layer_embeddings is not None else None
        if ple is None:
            return hidden_states
        return ple(hidden_states, self._input_ids)

    def _wire_depth_memory(self) -> None:
        """Linear Depth Memory, stream form: the boundary, the static state workspace and every
        mixer's step (edge 0 the embedding write, then the sublayers in execution order)."""
        from megatron.core.transformer.depth_memory import (
            DepthMemoryMixer,
            DepthMemoryStreamBoundary,
        )
        from megatron.core.transformer.depth_memory_state import make_workspace

        if self._loop_schedule_requested():
            raise NotImplementedError("depth_memory with looped (weight-tied) depth")
        mixers = []
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                candidates = (layer.mhc_attn_mixer, layer.mhc_mlp_mixer)
            else:
                scheme = getattr(layer, "residual_scheme", None)
                candidates = (scheme.mhc_mixer if scheme is not None else None,)
            mixers += [m for m in candidates if isinstance(m, DepthMemoryMixer)]
        if not mixers:
            raise ValueError("depth_memory: no sublayer has a depth-memory mixer")
        self.ldm_stream_boundary = DepthMemoryStreamBoundary(self.config)
        for param in self.ldm_stream_boundary.parameters():
            param.sequence_parallel = self.config.sequence_parallel
        self.ldm_workspace = make_workspace(self.config, total_edges=len(mixers) + 1)
        self.ldm_stream_boundary.wire(self.ldm_workspace)
        for edge, mixer in enumerate(mixers, start=1):
            mixer.wire(edge, self.ldm_workspace)

    def _wire_attn_res_streams(self) -> None:
        """Block AttnRes, stream form: every mixer's source slice, block slot and seed, from the
        sublayer order and the block size (a block closes after the layer whose number is a
        multiple of layers_per_block, as DepthResidualState.should_seal)."""
        from megatron.core.transformer.attn_res import (
            AttnResStreamBoundary,
            AttnResStreamMixer,
            resolve_layers_per_block,
        )

        if self._loop_schedule_requested():
            raise NotImplementedError("attn_res streams with looped (weight-tied) depth")
        per_block = resolve_layers_per_block(self.config.num_layers, self.config.attn_res_num_blocks)
        completed, partial, slots = 0, False, 1
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                candidates = (layer.mhc_attn_mixer, layer.mhc_mlp_mixer)
            else:
                scheme = getattr(layer, "residual_scheme", None)
                candidates = (scheme.mhc_mixer if scheme is not None else None,)
            for mixer in candidates:
                if not isinstance(mixer, AttnResStreamMixer):
                    continue
                mixer.wire(live=1 + completed + int(partial), slot=1 + completed, seed=not partial)
                partial = True
                slots = max(slots, 2 + completed)
            if partial and layer.layer_number % per_block == 0:
                completed, partial = completed + 1, False
        last = 1 + completed if partial else completed
        self.attnres_stream_boundary = AttnResStreamBoundary(slots, last)

    def _loop_schedule_requested(self) -> bool:
        return build_loop_schedule(self.config) is not None

    def _depth_enter(self, hidden_states: Tensor):
        """Widen the residual for mHC or build the Block AttnRes state.

        The HybridStack counterpart of ``TransformerBlock._depth_enter``: every layer
        type -- TransformerLayer sublayers, Mamba (`M`) and Engram (`N`) through their
        ResidualSchemeAdapter -- reads and writes the same state, so a hybrid pattern
        uses mHC / Block AttnRes across all of its layers. Across a pipeline stage
        boundary both travel as the fixed-width ``[s, b, slots * h]`` payload.

        Returns:
            ``(hidden_states, depth_state)``; ``hidden_states`` is ``[s, b, n, h]``
            under mHC, ``depth_state`` is None unless Block AttnRes is active.
        """
        mhc = getattr(self.config, "mhc", False)
        attn_res = getattr(self.config, "attn_res", False)
        if self.ldm_stream_boundary is not None:
            if not (self.pre_process and self.post_process):
                raise NotImplementedError("depth_memory needs the whole stack on one stage")
            return self.ldm_stream_boundary.expand(hidden_states), None
        if self.attnres_stream_boundary is not None:
            if not (self.pre_process and self.post_process):
                raise NotImplementedError("attn_res streams need the whole stack on one stage")
            return self.attnres_stream_boundary.expand(hidden_states), None
        if self.ldm_boundary is not None:
            if not (self.pre_process and self.post_process):
                raise NotImplementedError("depth_memory needs the whole stack on one stage")
            return hidden_states, self.ldm_boundary.enter(hidden_states)
        if not mhc and not attn_res:
            return hidden_states, None
        if mhc and attn_res:
            raise ValueError("mhc and attn_res are mutually exclusive; enable at most one")

        from megatron.core.transformer.depth_residual import depth_payload_slots, unpack_payload

        slots = depth_payload_slots(self.config)
        if mhc:
            from megatron.core.transformer.mhc import expand_to_streams

            if self.pre_process:
                return expand_to_streams(hidden_states, slots), None
            from megatron.core.transformer.routed_residual import own_streams

            s, b, wide = hidden_states.shape
            return own_streams(hidden_states.reshape(s, b, slots, wide // slots), self.config), None

        from megatron.core.transformer.attn_res import resolve_layers_per_block
        from megatron.core.transformer.depth_residual import DepthResidualState

        layers_per_block = resolve_layers_per_block(
            self.config.num_layers, self.config.attn_res_num_blocks
        )
        if self.pre_process:
            return hidden_states, DepthResidualState(
                hidden_states, layers_per_block=layers_per_block
            )
        done = self.pp_layer_offset
        has_partial = bool(done % layers_per_block)
        live = 1 + done // layers_per_block + int(has_partial)
        sources = unpack_payload(hidden_states, slots, live)
        state = DepthResidualState(sources[0], layers_per_block=layers_per_block)
        if has_partial:
            state.blocks, state.partial = sources[:-1], sources[-1]
        else:
            state.blocks, state.partial = sources, None
        return hidden_states, state

    def _depth_exit(self, hidden_states: Tensor, depth_state) -> Tensor:
        """Collapse the depth state on the last stage, else pack it for the next one."""
        from megatron.core.transformer.depth_residual import depth_payload_slots, pack_payload

        if self.ldm_stream_boundary is not None and hidden_states.dim() == 4:
            return self.ldm_stream_boundary.collapse(hidden_states)
        if self.attnres_stream_boundary is not None and hidden_states.dim() == 4:
            return self.attnres_stream_boundary.collapse(hidden_states)
        if getattr(self.config, "mhc", False) and hidden_states.dim() == 4:
            from megatron.core.transformer.mhc import collapse_streams

            if self.post_process:
                return collapse_streams(hidden_states)
            s, b, n, h = hidden_states.shape
            return make_viewless_tensor(
                inp=hidden_states.reshape(s, b, n * h), requires_grad=True, keep_graph=True
            )
        if depth_state is None:
            return hidden_states
        if self.ldm_boundary is not None:
            hidden_states = self.ldm_boundary.exit(depth_state)
            for layer in self.layers:
                layer._depth_state = None
            return hidden_states
        if self.post_process:
            hidden_states = depth_state.collapse()
        else:
            hidden_states = make_viewless_tensor(
                inp=pack_payload(depth_state.sources(), depth_payload_slots(self.config)),
                requires_grad=True,
                keep_graph=True,
            )
        for layer in self.layers:
            layer._depth_state = None
        return hidden_states

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Optional[tuple] = None,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """
        Returns a sharded state dictionary for the current object.

        This function constructs a sharded state dictionary by iterating over the layers
        in the current object, computing the sharded state dictionary for each layer,
        and combining the results into a single dictionary.

        Parameters:
            prefix (str): The prefix to use for the state dictionary keys.
            sharded_offsets (tuple): The sharded offsets to use for the state dictionary.
            metadata (dict): Additional metadata to use when computing the sharded state dictionary.

        Returns:
            dict: The sharded state dictionary for the current object.
        """

        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'

        for local_layer_idx, layer in enumerate(self.layers):

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = (
                f'{layer_prefix}{local_layer_idx}.'  # module list index in HybridStack
            )

            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

            # A layer's per-layer embedding (PLE) is keyed by the same global layer
            # index, so pipeline layouts can change between save and load.
            ple = (
                self.per_layer_embeddings[local_layer_idx]
                if self.per_layer_embeddings is not None
                else None
            )
            if ple is not None:
                ple_prefix = f'{prefix}per_layer_embeddings.{local_layer_idx}.'
                ple_sd = ple.sharded_state_dict(ple_prefix, sharded_pp_offset, metadata)
                replace_prefix_for_sharding(
                    ple_sd, ple_prefix, f'{prefix}per_layer_embeddings.{global_layer_offset}.'
                )
                sharded_state_dict.update(ple_sd)

        # Add modules other than self.layers (and the per-layer embeddings, above)
        for name, module in self.named_children():
            if not module is self.layers and not module is self.per_layer_embeddings:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        f'{prefix}{name}.',
                        sharded_offsets,
                        metadata,
                        tp_group=self.tp_group,
                    )
                )

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack
