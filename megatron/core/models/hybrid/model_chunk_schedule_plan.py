# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule-plan classes for HybridModel's EP overlap (combined 1F1B).

Port of upstream's HybridStack schedule plans (#4942): they extend GPTModel's
``TransformerLayerSchedulePlan`` / ``TransformerModelChunkSchedulePlan``. Here one schedule
layer is one implicit group of HybridStack layers (see ``fine_grained_callables``); the
pre/post-process nodes of ``core.models.common.utils`` are reused as they are -- they call
``model._preprocess`` / ``model._postprocess``, which HybridModel implements.
"""

from contextlib import nullcontext

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.common.utils import TransformerLayerNode
from megatron.core.models.hybrid.fine_grained_callables import (
    HybridLayerGroup,
    build_hybrid_group_callables,
    split_hybrid_layer_groups,
)
from megatron.core.pipeline_parallel.utils import NoopScheduleNode


class HybridGroupSchedulePlan(TransformerLayerSchedulePlan):
    """Per-group schedule plan: the slots of one HybridStack layer group."""

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        fwd_callables, bwd_dw_callable_map, is_moe, num_local_experts = (
            build_hybrid_group_callables(self.layer)
        )
        extra_args["config"] = self.layer.config
        extra_args["is_moe"] = is_moe
        extra_args["num_local_experts"] = num_local_experts
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = False

        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            node_extra_args = dict(extra_args)
            if bwd_dw_callables is None:
                node_extra_args["delay_wgrad_compute"] = False
            return TransformerLayerNode(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=node_extra_args,
            )

        pre_dispatch, dispatch, mlp, combine, _ = fwd_callables
        self.pre_dispatch_computation = create_node(
            comp_stream, pre_dispatch, "pre_dispatch_computation"
        )
        self.mlp = create_node(comp_stream, mlp, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, dispatch, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, combine, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()
        # HybridModel's EP overlap rejects MTP: no MTP post-process slot
        self.mtp_post_process = NoopScheduleNode()

    def get_low_precision_context(self):
        """The callables enter each layer's quantization context themselves."""
        return nullcontext()


class HybridModelChunkSchedulePlan(TransformerModelChunkSchedulePlan):
    """Model-chunk schedule plan whose layers are the HybridStack's layer groups."""

    LAYER_SCHEDULE_PLAN_CLASS = HybridGroupSchedulePlan

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream):
        if module is None:
            return
        groups = split_hybrid_layer_groups(module.layer_type_list)
        for group_idx, indices in enumerate(groups):
            extra_args = self._extra_args_for_layer(module, group_idx, len(groups))
            self._transformer_layers.append(
                HybridGroupSchedulePlan(
                    HybridLayerGroup(module, indices),
                    self.event,
                    self.state,
                    comp_stream,
                    comm_stream,
                    extra_args,
                )
            )
