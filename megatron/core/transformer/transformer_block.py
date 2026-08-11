# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Set, Union, cast

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.transformer.enums import InferenceCudaGraphScope, LayerType
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    log_single_rank,
    make_viewless_tensor,
)

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TENorm, get_cpu_offload_context

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm


logger = logging.getLogger(__name__)


def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert (
                layers_to_distribute % pipeline_stages_left == 0
            ), "With uneven pipelineing the left over layers must be divisible by left over stages"
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), f"{num_layers=} should be divisible by {config.pipeline_model_parallel_size=}"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert (
            num_layers_per_pipeline_rank % vp_size == 0
        ), f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert (
                num_layers_to_build >= 0
            ), f"Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, f"Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def build_loop_schedule(config) -> Optional[List[int]]:
    """Return the order in which ``self.layers`` is applied, or None when not looping.

    A looped stack is prelude / block x iterations / coda over *unique* layers, so
    with 3 prelude, a 3-layer block run 4 times and 3 coda the 9 built layers are
    applied in the order::

        [0, 1, 2,  3, 4, 5,  3, 4, 5,  3, 4, 5,  3, 4, 5,  6, 7, 8]

    giving an effective depth of 18. The block's weights are shared across
    iterations, which is only well-posed if the residual keeps a stable magnitude
    with depth -- the property nGPT's LERP residual provides and a plain additive
    residual does not.

    ``loop_group_size`` sets *where* the recurrence sits, at identical parameter
    count and identical FLOPs -- only the application order changes. For a 6-layer
    block run twice::

        group 6 (default):  3 4 5 6 7 8 | 3 4 5 6 7 8    outer recurrence
        group 3:            3 4 5 3 4 5 | 6 7 8 6 7 8
        group 1:            3 3 | 4 4 | 5 5 | 6 6 | ...  each layer repeated

    At group 1 a layer immediately consumes its own output, which is the hardest
    case for magnitude stability; at group 6 it sees the rest of the block first.
    So the LERP's advantage, if the invariance is what makes tying work, should be
    largest at small group sizes.

    Args:
        config: A :class:`TransformerConfig` (or anything exposing the loop fields).

    Returns:
        The layer-index schedule, or None when ``loop_block_layers`` is 0.
    """
    block = getattr(config, "loop_block_layers", 0)
    if not block:
        return None
    prelude = getattr(config, "loop_prelude_layers", 0)
    coda = getattr(config, "loop_coda_layers", 0)
    iterations = getattr(config, "loop_iterations", 1)
    # 0 means "the whole block is one group", i.e. outer recurrence.
    group = getattr(config, "loop_group_size", 0) or block

    schedule = list(range(prelude))
    for start in range(prelude, prelude + block, group):
        for _ in range(iterations):
            schedule.extend(range(start, start + group))
    schedule.extend(range(prelude + block, prelude + block + coda))
    return schedule


@dataclass
class TransformerBlockSubmodules:
    """
    Dataclass for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: Optional[List[ModuleSpec]] = None
    layer_norm: LayerNormBuilder | None = None


def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(GraphableMegatronModule, MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        pp_rank = get_pg_rank(pp_group)

        self.submodules = _get_block_submodules(config, spec, vp_stage, pp_rank)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        if get_cpu_offload_context is not None:
            (self.offload_context, self.group_prefetch_offload_commit_async) = (
                get_cpu_offload_context(
                    self.config.cpu_offloading,
                    self.config.cpu_offloading_num_layers,
                    self.config.num_layers,
                    self.config.cpu_offloading_activations,
                    self.config.cpu_offloading_weights,
                    self.config.cpu_offloading_double_buffering,
                    self.config.cpu_offloading_retain_pinned_cpu_buffers,
                )
            )
            self.config._cpu_offloading_context = (
                self.offload_context if self.config.cpu_offloading else None
            )
        else:
            assert (
                self.config.cpu_offloading is False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            else:
                quantization_context = nullcontext()

            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            return module

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # Looped depth: the middle block is applied loop_iterations times. Stored as
        # an explicit index list rather than nested loops in forward, so the whole
        # schedule is inspectable and testable without running a forward.
        self._loop_schedule = build_loop_schedule(self.config)
        if self._loop_schedule is not None:
            # Logged unconditionally: a looped run that silently fell back to a plain
            # stack is indistinguishable from a correct one in the loss curve, and the
            # flags reaching Megatron at all depends on plumbing outside this repo.
            # This line is the positive confirmation that the loop is active.
            log_single_rank(
                logger,
                logging.INFO,
                f"Looped depth active: {len(self.layers)} unique layers applied as "
                f"{len(self._loop_schedule)} (prelude {self.config.loop_prelude_layers}, "
                f"block {self.config.loop_block_layers} x {self.config.loop_iterations}, "
                f"group {self.config.loop_group_size or self.config.loop_block_layers}, "
                f"coda {self.config.loop_coda_layers}); schedule={self._loop_schedule}",
            )
            # Recompute walks the loop schedule (see megatron/core/recompute.py), and
            # CUDA graphs need no special handling: at PP=1 CudaGraphManager selects
            # the first FWD_READY runner and a forward flips it to BWD_READY, so N
            # calls to one module capture N runners and replay them in capture order.
            # That is the same mechanism gradient accumulation already relies on --
            # a looped layer at mbs<gbs/DP was always being called several times per
            # step. Both paths are covered by tests; neither is guarded here.

        # @TODO: add back account_for_embedding_in_pipeline_split (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if self.has_final_layernorm_in_this_stage():
            self.final_layernorm = not_none(self.submodules.layer_norm)(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

        if self.config.inference_fuse_tp_communication:
            self._setup_fused_tp_communication()

    def has_final_layernorm_in_this_stage(self):
        """
        Check if this vpp stage contains the final layernorm.

        Note:
            Final layernorm now has been moved from the post-process stage to the last decoder
            layer by using this function.
            There will be a small numeric difference because of grad norm reduction when final
            layernorm is placed in different pipeline stages in deterministic mode. It can still
            be bitwise aligned by disabling grad norm clipping.
        """
        if self.config.mtp_num_layers is None:
            # for model without MTPLayer, the final layernorm is set in the stage which does
            # post_process
            return self.submodules.layer_norm and self.post_process and self.post_layer_norm
        else:
            # for model with MTPLayer, the final layernorm is set in the stage which has the
            # last layer of the decoder
            has_final_layernorm_in_this_stage = False
            for layer in self.layers:
                if layer.layer_number == self.config.num_layers:
                    has_final_layernorm_in_this_stage = True
                    break
            return (
                self.submodules.layer_norm
                and has_final_layernorm_in_this_stage
                and self.post_layer_norm
            )

    def _setup_fused_tp_communication(self):
        """Setup fused TP communication for all layers.
        We have a fused reduce-scatter + add + layer-norm + all-gather operation.
        We call this kernel from within row parallel linear layers.
        But layer-norm needs the layer norm weights from the
        successive column parallel linear layer.
        This function is used to pass those weights to the respective layers.
        """

        for i in range(len(self.layers)):
            current_layer = self.layers[i]

            # Get next layer's QKV norm weights (None for last layer)
            if i < len(self.layers) - 1:
                next_qkv_norm_weights = self.layers[i + 1].get_qkv_layer_norm_weights()
            else:
                next_qkv_norm_weights = None

            # Configure all fused TP communication settings in one call
            current_layer.configure_fused_tp_inference(
                skip_qkv_norm_and_all_gather=(i > 0),
                fc2_next_layer_norm_weights=next_qkv_norm_weights,
            )

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        if (
            InferenceMode.is_active()
            and hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                kwargs.get('inference_context') is not None
                or kwargs.get('inference_params') is not None
            )
            and self.config.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()

            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            kwargs['hidden_states'] = (
                kwargs['hidden_states'].unwrap()
                if isinstance(kwargs['hidden_states'], WrappedTensor)
                else kwargs['hidden_states']
            )
            return super().__call__(*args, **kwargs)[0]
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        extract_layer_indices: Optional[Set[int]] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            extract_layer_indices (Set[int], optional): A set of global
                layer indices (0-based across all pipeline stages) from
                which to extract intermediate hidden states. If
                non-empty, the forward pass will collect hidden_states
                after each specified layer.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, List[Tensor]]]:
                - If extract_layer_indices is None or empty: Returns the output hidden states tensor
                  of shape [s, b, h].
                - If extract_layer_indices is non-empty: Returns a tuple
                  of (hidden_states, intermediate_hidden_states) where
                  intermediate_hidden_states is a list of tensors
                  corresponding to hidden states after each layer in
                  extract_layer_indices.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Initialize feature collection (consistent with FastGen's Wan implementation)
        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states: List[Tensor] = []

        # Calculate the global layer offset for this pipeline stage
        # This is needed to convert local layer indices to global indices for feature extraction
        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        layer_offset = get_transformer_layer_offset(
            self.config, self.vp_stage, get_pg_rank(pp_group)
        )

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                checkpointed_result = checkpointed_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    extract_layer_indices=extract_layer_indices,
                    layer_offset=layer_offset,
                )
                # Handle return value from _checkpointed_forward
                if len(extract_layer_indices) > 0:
                    # (hidden_states, intermediate_hidden_states) tuple
                    hidden_states, intermediate_hidden_states = checkpointed_result
                else:
                    # No intermediate_hidden_states requested: just hidden_states
                    hidden_states = checkpointed_result
            else:
                hidden_states, depth_state = self._depth_enter(hidden_states)
                if self._loop_schedule is None:
                    layer_order = list(enumerate(self.layers))
                else:
                    # l_no counts *positions*, not unique layers, so extract_layer_indices
                    # keeps addressing depth rather than parameters.
                    layer_order = [(i, self.layers[li]) for i, li in enumerate(self._loop_schedule)]
                for l_no, layer in layer_order:
                    if depth_state is not None:
                        layer._depth_state = depth_state
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

                    # Extract intermediate embeddings using global layer index
                    if (l_no + layer_offset) in extract_layer_indices:
                        intermediate_hidden_states.append(hidden_states)

        # Collapse the depth-wise residual state (mHC streams / AttnRes blocks)
        # back to a single hidden state before the final norm.
        hidden_states = self._depth_exit(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = apply_module(self.final_layernorm)(cast(Tensor, hidden_states))
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        return hidden_states

    def _depth_enter(self, hidden_states: Tensor):
        """Widen the residual stream for this forward pass, if a scheme is active.

        mHC and Block AttnRes both replace the single residual stream with a
        wider, depth-spanning state that every layer reads from and writes to,
        but they carry it differently within a stage:

        * **mHC** widens ``hidden_states`` itself to ``[s, b, n, h]``, so the
          state travels through the layers as an ordinary argument and return
          value -- which is what allows the layers to be CUDA-graph captured.
        * **Block AttnRes** accumulates a growing list of block summaries, which
          no single fixed-shape tensor expresses, so the block owns a
          :class:`~megatron.core.transformer.depth_residual.DepthResidualState`
          and stashes it on each layer.

        Across a pipeline stage boundary both travel the same way: flattened to
        ``[s, b, slots * h]`` by :func:`~megatron.core.transformer.depth_residual.pack_payload`,
        a fixed width so the schedules can compute one tensor shape for the whole
        pipeline. Deliberately 3-D -- the p2p layer exchanges shapes as a
        3-element tensor under ``variable_seq_lengths``, so a rank-4 payload
        would be silently truncated there.

        Args:
            hidden_states: ``[s, b, h]`` on the first stage, otherwise the
                ``[s, b, slots * h]`` payload from the previous stage.

        Returns:
            ``(hidden_states, depth_state)``. ``hidden_states`` is widened for
            mHC; ``depth_state`` is None unless Block AttnRes is active.
        """
        mhc = getattr(self.config, "mhc", False)
        attn_res = getattr(self.config, "attn_res", False)
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
            # Already widened by an earlier stage; restore the stream axis.
            s, b, wide = hidden_states.shape
            return hidden_states.reshape(s, b, slots, wide // slots), None

        from megatron.core.transformer.attn_res import resolve_layers_per_block
        from megatron.core.transformer.depth_residual import DepthResidualState

        layers_per_block = resolve_layers_per_block(
            self.config.num_layers, self.config.attn_res_num_blocks
        )
        if self.pre_process:
            return hidden_states, DepthResidualState(
                hidden_states, layers_per_block=layers_per_block
            )

        live, has_partial = self._depth_live_slots(layers_per_block)
        sources = unpack_payload(hidden_states, slots, live)
        state = DepthResidualState(sources[0], layers_per_block=layers_per_block)
        # sources() appends the partial last when there is one, so pack_payload
        # put it in the final live slot.
        if has_partial:
            state.blocks, state.partial = sources[:-1], sources[-1]
        else:
            state.blocks, state.partial = sources, None
        return hidden_states, state

    def _depth_live_slots(self, layers_per_block: int) -> tuple[int, bool]:
        """Payload slices carrying real data at this stage's input boundary.

        Derived from the layer offset rather than signalled, because the offset
        is static: layer counts per stage do not vary at run time, so both sides
        of the boundary compute the same answer. ``b_0`` is always present, one
        more slice per block sealed so far, and one for the in-progress partial
        sum when the boundary falls mid-block.

        Returns:
            ``(live_slices, has_partial)``.
        """
        done = get_transformer_layer_offset(self.config, vp_stage=self.vp_stage)
        has_partial = bool(done % layers_per_block)
        return 1 + done // layers_per_block + int(has_partial), has_partial

    def _depth_exit(self, hidden_states: Tensor) -> Tensor:
        """Reduce or repack the depth-wise state at the end of this stage.

        On the last pipeline stage the state collapses to a single ``[s, b, h]``
        hidden state for the final norm. Otherwise it is flattened to the
        fixed-width payload the next stage expects.

        Args:
            hidden_states: The trunk output -- ``[s, b, n, h]`` streams under
                mHC, otherwise ``[s, b, h]``.

        Returns:
            ``[s, b, h]`` on the last stage, else ``[s, b, slots * h]``.
        """
        from megatron.core.transformer.depth_residual import depth_payload_slots, pack_payload

        mhc = getattr(self.config, "mhc", False)
        if mhc and hidden_states.dim() == 4:
            from megatron.core.transformer.mhc import collapse_streams

            if self.post_process:
                return collapse_streams(hidden_states)
            s, b, n, h = hidden_states.shape
            # reshape returns a view, and the pipeline schedule refuses to free a
            # view of another tensor when it deallocates a stage output.
            return make_viewless_tensor(
                inp=hidden_states.reshape(s, b, n * h), requires_grad=True, keep_graph=True
            )

        depth_state = getattr(self.layers[0], "_depth_state", None) if len(self.layers) else None
        if depth_state is None:
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
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        # TODO: remove multiple non_homogeneous_layers=True assignments
        #  once non_homogeneous_layers=False support is dropped
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if self.config.hetereogenous_dist_checkpoint:
            non_homogeneous_layers = True

        if isinstance(self.config.moe_layer_freq, int):
            if self.config.moe_layer_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.moe_layer_freq, list):
            non_homogeneous_layers = True

        if isinstance(self.config.linear_attention_freq, int):
            if self.config.linear_attention_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.linear_attention_freq, list):
            non_homogeneous_layers = True

        if self.config.heterogeneous_block_specs:
            non_homogeneous_layers = True

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        if singleton_local_shards:
            if metadata is not None and metadata.get('non_homogeneous_layers') is False:
                # non_homogeneous_layers=False was set explicitly - emit an override warning
                logger.warning(
                    'non_homogeneous_layers=False is deprecated.'
                    ' Setting non_homogeneous_layers=True.'
                )
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
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
