# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer callables for HybridModel's EP-overlap (combined 1F1B) schedule plan.

Port of the HybridStack EP-overlap callables of upstream Megatron-LM (#4798 split as
#4942 / #4944, by @Wohox / Yan Xu) onto this branch's HybridStack, which keeps one flat
list of layers per pattern symbol. Instead of upstream's bracketed groups (``[*E]``) the
schedule groups consecutive layers implicitly: every run of layers that ends at an MLP
(``-``) or MoE (``E``) layer is one schedule "layer" (``*E*E`` schedules as ``[*E][*E]``),
a trailing run without one is a group of its own. Module structure, parameter names and
checkpoints are untouched.

``build_hybrid_group_callables`` splits one group into the schedule plan's slots:

* pre-dispatch compute: the group's mixer layers (attention / MLA / GDN / linear / Mamba
  ...), called exactly as ``HybridStack.forward`` calls them, then the MoE layer's pre-MLP
  norm, shared experts, router and dispatch preprocessing;
* token dispatch;
* the dense MLP layer, or the routed experts;
* token combine, the MoE layer's post-MLP step (residual add) and, on the last group, the
  stack's final norm.

The equivalent callables for GPTModel's ``TransformerLayer`` are in
``megatron/core/models/gpt/fine_grained_callables.py``.
"""

from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.pipeline_parallel.utils import (
    ScheduleNode,
    StageDispatchBwdGrad,
    get_comm_stream,
)
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor

TERMINAL_SYMBOLS = (LayerSymbols.MLP, LayerSymbols.MOE)


def split_hybrid_layer_groups(layer_type_list: Sequence[str]) -> List[List[int]]:
    """Indices of the schedule groups: consecutive layers up to and including each MLP /
    MoE layer; trailing layers without one form the last group."""
    groups, current = [], []
    for i, symbol in enumerate(layer_type_list):
        current.append(i)
        if symbol in TERMINAL_SYMBOLS:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


class HybridLayerGroup:
    """The layers of one schedule group (a stand-in for the single layer the schedule
    plan's base classes expect: they read ``config`` and drop the reference on release)."""

    def __init__(self, stack, indices: Sequence[int]):
        self.stack = stack
        self.layers = [stack.layers[i] for i in indices]
        self.layer_types = [stack.layer_type_list[i] for i in indices]
        self.config = self.layers[0].config

    @property
    def terminal(self):
        """``(symbol, layer)`` of the group's MLP / MoE layer, or ``(None, None)``."""
        if self.layer_types[-1] in TERMINAL_SYMBOLS:
            return self.layer_types[-1], self.layers[-1]
        return None, None

    @property
    def pre_layers(self):
        """``(symbol, layer)`` of the mixer layers before the terminal one."""
        n = len(self.layers) - (1 if self.terminal[0] is not None else 0)
        return list(zip(self.layer_types[:n], self.layers[:n]))


class _MoEBackwardDWWrapper:
    """Run delayed MoE wgrad and expose only the parameters owned by its slot.

    Shared experts and the latent down projection run in pre-dispatch; routed experts and
    the latent up projection run later. Their gradient-accumulation hooks must run after
    their respective wgrad, even if a previous microbatch already populated ``param.grad``.
    """

    def __init__(self, mlp, routed_experts: bool):
        self.dw_callable = partial(
            mlp.backward_dw, routed_experts=routed_experts, shared_experts=not routed_experts
        )
        self.wgrad_on_comm_stream = False
        self.submodules = []
        if routed_experts:
            self.submodules.append(mlp.experts)
        elif mlp.use_shared_expert and not mlp.shared_expert_overlap:
            self.submodules.append(mlp.shared_experts)
        if mlp.config.moe_latent_size and mlp.config.overlap_moe_expert_parallel_comm:
            self.submodules.append(mlp.fc2_latent_proj if routed_experts else mlp.fc1_latent_proj)
            self.wgrad_on_comm_stream = routed_experts

    def backward_dw(self):
        """Run this slot's wgrad after its autograd backward."""
        self.dw_callable()
        if self.wgrad_on_comm_stream:
            # MoELayer runs the latent up-projection wgrad on the communication stream;
            # its accumulation hooks run on this node's compute stream.
            torch.cuda.current_stream().wait_stream(get_comm_stream())
        self.dw_callable = None

    def parameters(self):
        """Parameters whose post-wgrad hooks this slot runs."""
        for module in self.submodules:
            yield from module.parameters()


class _AttentionBackwardDWWrapper:
    """Delayed attention wgrad of a mixer layer in the pre-dispatch slot."""

    def __init__(self, layer: TransformerLayer):
        self.layer = layer

    def backward_dw(self):
        """Run the attention wgrad."""
        self.layer.self_attention.backward_dw()

    def parameters(self):
        """Parameters whose post-wgrad hooks this slot runs."""
        yield from self.layer.self_attention.parameters()


def _inner_quant_context(layer):
    """The per-layer quantization context of ``HybridStack.forward`` (the delayed-recipe
    outer FP8 context is rejected for EP overlap by HybridModel)."""
    config = layer.config
    if config.fp8 and config.fp8_recipe != Fp8Recipe.delayed:
        return get_fp8_context(config, layer.layer_number - 1)
    if config.fp4:
        return get_fp4_context(config, layer.layer_number - 1)
    return nullcontext()


def _flex_backends(config):
    flex = config.moe_token_dispatcher_type == "flex"
    backend = config.moe_flex_dispatcher_backend
    return flex and backend == "deepep", flex and backend == "hybridep", flex and backend == "ncclep"


def _maybe_final_norm(node: ScheduleNode, hidden_states: Tensor) -> Tensor:
    """The stack's final norm after the last group (as at the end of HybridStack.forward)."""
    decoder = node.chunk_state.model.decoder
    final_norm = getattr(decoder, "final_norm", None)
    if node.is_last_layer and decoder.post_process and decoder.post_layer_norm:
        hidden_states = final_norm(hidden_states)
    if node.is_last_layer:
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
    return hidden_states


def _run_pre_layer(node: ScheduleNode, symbol: str, layer, hidden_states: Tensor) -> Tensor:
    """One mixer layer, called exactly as HybridStack.forward calls it."""
    state = node.chunk_state
    with _inner_quant_context(layer):
        if isinstance(layer, TransformerLayer):
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                attention_mask=state.attention_mask,
                inference_context=None,
                rotary_pos_emb=state.rotary_pos_emb,
                sequence_len_offset=state.sequence_len_offset,
                packed_seq_params=state.packed_seq_params,
                padding_mask=state.padding_mask,
            )
        else:  # MambaLayer
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=state.attention_mask,
                inference_context=None,
                packed_seq_params=state.packed_seq_params,
            )
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    return hidden_states


def _run_moe_preprocess(layer, node: ScheduleNode, hidden_states: Tensor):
    """MoE layer up to dispatch: its (identity) attention part, the pre-MLP norm, shared
    experts, router and dispatch preprocess -- the order of TransformerLayer._forward_mlp
    and MoELayer.forward."""
    state = node.chunk_state
    hidden_states, _ = layer._forward_attention(
        hidden_states=hidden_states,
        attention_mask=state.attention_mask,
        rotary_pos_emb=state.rotary_pos_emb,
        sequence_len_offset=state.sequence_len_offset,
        packed_seq_params=state.packed_seq_params,
        padding_mask=state.padding_mask,
    )
    pre_mlp_layernorm_output = layer._forward_pre_mlp_layernorm(hidden_states)
    # A subsequent microbatch can reuse this layer before this node combines.
    node.layer_state.mlp_norm_manager = layer.mlp_norm_manager
    layer.mlp_norm_manager = None
    if isinstance(pre_mlp_layernorm_output, tuple):
        if len(pre_mlp_layernorm_output) != 2:
            raise ValueError(
                f"When the output of pre_mlp_layernorm is a tuple, it is expected to have "
                f"2 elements (output, residual), but got {len(pre_mlp_layernorm_output)}"
            )
        pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
    else:
        residual = hidden_states
    if layer.config.fp32_residual_connection:
        residual = residual.float()

    pre_mlp_layernorm_output, padding_mask, mbs = layer._maybe_unflatten_for_moe(
        pre_mlp_layernorm_output, state.padding_mask, state.packed_seq_params
    )
    # MoELayer.forward hands the router the mask as [s, b] bool
    if padding_mask is not None:
        padding_mask = padding_mask.transpose(0, 1).bool()
    node.layer_state.moe_unflatten_mbs = mbs

    moe = layer.mlp
    shared_expert_output = moe.shared_experts_compute(pre_mlp_layernorm_output)
    probs, routing_map = moe.route(pre_mlp_layernorm_output, padding_mask)
    local_tokens, probs = moe.preprocess(pre_mlp_layernorm_output, probs, routing_map)

    node.layer_state.residual = node.detach(residual)
    node.layer_state.shared_expert_output = None
    if moe.use_shared_expert and not moe.shared_expert_overlap:
        node.layer_state.shared_expert_output = node.detach(shared_expert_output)
    return local_tokens, probs


def _run_moe_experts(layer, node: ScheduleNode, dispatched_tokens: Tensor):
    dispatched_probs = node.layer_state.dispatched_probs
    enable_deepep, enable_hybridep, enable_ncclep = _flex_backends(layer.config)
    token_dispatcher = layer.mlp.token_dispatcher
    if enable_deepep or enable_hybridep or enable_ncclep:
        # the detached probs, so the backward graph does not reach the dispatch node
        token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

    expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

    if enable_hybridep or enable_ncclep:
        # tokens_per_expert is produced on the comm stream: keep a reference
        node.layer_state.tokens_per_expert = (
            token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
        )
    if layer.recompute_pre_mlp_layernorm:
        layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)
    return expert_output


def _run_moe_combine(layer, node: ScheduleNode, output: Tensor):
    from megatron.core.transformer.moe.moe_utils import should_mask_routed_moe_layer

    residual = node.layer_state.residual
    shared_expert_output = node.layer_state.shared_expert_output
    moe = layer.mlp
    output = moe.combine(output)
    if should_mask_routed_moe_layer(moe.layer_number):
        output = torch.zeros_like(output)
    output = moe.postprocess(output, shared_expert_output)
    mbs = node.layer_state.moe_unflatten_mbs
    output = layer._maybe_reflatten_from_moe(output, node.chunk_state.packed_seq_params, mbs)

    # The post-MLP step of TransformerLayer._forward_post_mlp (post-MLP norm, bias-dropout-
    # add, mlp-norm offload). Its recompute hook for the pre-MLP norm is registered on the
    # expert output in _run_moe_experts instead (recompute_pre_mlp_layernorm is rejected
    # with EP overlap by HybridModel, so the hook here never fires twice).
    layer.mlp_norm_manager = node.layer_state.mlp_norm_manager
    hidden_states = layer._forward_post_mlp((output, None), residual)
    node.layer_state.mlp_norm_manager = None

    node.layer_state.residual.record_stream(torch.cuda.current_stream())
    if shared_expert_output is not None:
        shared_expert_output.record_stream(torch.cuda.current_stream())
    node.layer_state.residual = None
    node.layer_state.shared_expert_output = None
    return _maybe_final_norm(node, hidden_states)


def get_hybrid_group_moe_metadata(group: HybridLayerGroup):
    """``(is_moe, num_local_experts)`` of a schedule group."""
    symbol, layer = group.terminal
    is_moe = symbol == LayerSymbols.MOE
    return is_moe, (layer.mlp.num_local_experts if is_moe else None)


def build_hybrid_group_callables(group: HybridLayerGroup):
    """Fine-grained callables of one schedule group.

    Returns ``(forward_funcs, backward_dw, is_moe, num_local_experts)`` in the layout of
    ``build_transformer_layer_callables``: five forward callables (pre-dispatch, dispatch,
    mlp, combine, mtp post-process = None) and the delayed-wgrad callables per slot.
    """
    terminal_type, terminal_layer = group.terminal
    pre_layers = group.pre_layers
    is_moe, num_local_experts = get_hybrid_group_moe_metadata(group)
    if is_moe and not isinstance(terminal_layer.mlp, MoELayer):
        raise ValueError("EP overlap expects the `E` layer's mlp to be a MoELayer.")

    def pre_dispatch_computation(node: ScheduleNode, hidden_states: Tensor):
        for symbol, layer in pre_layers:
            hidden_states = _run_pre_layer(node, symbol, layer, hidden_states)
        if terminal_type == LayerSymbols.MOE:
            with _inner_quant_context(terminal_layer):
                return _run_moe_preprocess(terminal_layer, node, hidden_states)
        if terminal_type is None:
            return _maybe_final_norm(node, hidden_states)
        return hidden_states

    def dispatch(node: ScheduleNode, local_tokens: Tensor, probs: Tensor):
        enable_deepep, enable_hybridep, enable_ncclep = _flex_backends(terminal_layer.config)
        token_dispatcher = terminal_layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep or enable_ncclep:
            # the detached probs, so the backward graph does not reach pre-dispatch
            token_dispatcher._comm_manager.token_probs = probs
        with _inner_quant_context(terminal_layer):
            dispatched_tokens, dispatched_probs = terminal_layer.mlp.dispatch(local_tokens, probs)
        if enable_ncclep and terminal_layer.config.moe_ncclep_zero_copy:
            dispatched_tokens = StageDispatchBwdGrad.apply(dispatched_tokens, token_dispatcher)
        # needed by the experts' backward (swiglu): kept in layer_state, not freed as input
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def mlp(node: ScheduleNode, hidden_states: Tensor):
        if terminal_type == LayerSymbols.MLP:
            hidden_states = _run_pre_layer(node, terminal_type, terminal_layer, hidden_states)
            return _maybe_final_norm(node, hidden_states)
        if terminal_type == LayerSymbols.MOE:
            with _inner_quant_context(terminal_layer):
                return _run_moe_experts(terminal_layer, node, hidden_states)
        return hidden_states

    def combine(node: ScheduleNode, output: Tensor):
        with _inner_quant_context(terminal_layer):
            return _run_moe_combine(terminal_layer, node, output)

    def raise_not_implemented(*args):
        raise NotImplementedError("This callable is not implemented for non-MoE hybrid groups.")

    backward_dw = {}
    pre_bwd_dw = []
    if group.config.delay_wgrad_compute:
        for symbol, layer in pre_layers:
            if not isinstance(layer, TransformerLayer) or not hasattr(
                layer.self_attention, "backward_dw"
            ):
                raise ValueError(
                    "delay_wgrad_compute with EP overlap needs every mixer layer to support "
                    f"a delayed wgrad; layer '{symbol}' ({type(layer).__name__}) does not."
                )
            pre_bwd_dw.append(_AttentionBackwardDWWrapper(layer))
        if is_moe:
            # each slot owns the hooks of exactly the parameters whose wgrad it computes
            pre_dispatch_dw = _MoEBackwardDWWrapper(terminal_layer.mlp, routed_experts=False)
            if pre_dispatch_dw.submodules:
                pre_bwd_dw.append(pre_dispatch_dw)
            backward_dw["mlp"] = _MoEBackwardDWWrapper(terminal_layer.mlp, routed_experts=True)
        elif terminal_type == LayerSymbols.MLP:
            backward_dw["mlp"] = terminal_layer.mlp
        if pre_bwd_dw:
            backward_dw["pre_dispatch_computation"] = pre_bwd_dw

    forward_funcs = [
        pre_dispatch_computation,
        dispatch if is_moe else raise_not_implemented,
        mlp,
        combine if is_moe else raise_not_implemented,
        None,
    ]
    return forward_funcs, backward_dw, is_moe, num_local_experts


def supported_pre_layer(layer) -> bool:
    """Mixer layers the EP-overlap schedule can run in its pre-dispatch slot."""
    return isinstance(layer, (TransformerLayer, MambaLayer))
