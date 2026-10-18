# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MoonEP backend for the MoE layer (``moe_flex_dispatcher_backend='moonep'``).

MoonEP (MoonshotAI) keeps the per-rank token load perfectly balanced with dynamic redundant
experts: each EP rank owns ``epn = E / R`` experts and ``epn`` prefetch slots. An online GPU
planner sends every rank exactly ``S x K`` token slots; tokens of a hot expert land on other
ranks, which receive a copy of that expert's weights into a prefetch slot (NVLink push, before
the expert GEMMs). In the backward pass the prefetched copies' weight gradients are reduced back
to their home ranks (``reduce_grad``). Shapes are static (no host sync per layer) and the
dispatch writes tokens straight into their expert-grouped positions (no permute).

This is not a token dispatcher alone: the experts compute over ``2 * epn`` weight slots (their
own + the prefetched ones), so the backend replaces both the token dispatcher
(:class:`MoEMoonEPTokenDispatcher`) and the experts (:class:`MoonEPGroupedMLP`).

* Weights: ``weight1 [epn, 2F, H]`` (gate rows then up rows, Megatron's SwiGLU layout) and
  ``weight2 [epn, H, F]`` as single contiguous parameters (MoonEP pushes ``[epn, ...]``
  contiguous expert tensors). MoonEP's API carries three projections; the third is a 32 KiB
  per-expert placeholder.
* GEMMs: Transformer Engine's grouped-tensor grouped GEMM (cuBLASLt) with a discrete weight
  list (the local experts followed by this rank's prefetch slots) and device-side group sizes
  from MoonEP's ``cu_seqlens``.
* The prefetch pools are process-global and shared by every MoE layer, so the backward pass
  prefetches its layer's experts again before the dgrad GEMMs.
* Probabilities are applied after the activation (weighted SwiGLU), as in TEGroupedMLP.

Restrictions: NVLink domain only (EP group within one node), expert TP 1, SwiGLU, no bias, bf16
(fp8 not yet), static tokens per rank (S = micro batch x sequence), no token dropping.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu, weighted_swiglu_back
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher

try:
    import moonep  # noqa: F401
    from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment

    HAVE_MOONEP = True
except ImportError:  # pragma: no cover
    HAVE_MOONEP = False

try:
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import (
        GroupedTensorStorage,
    )

    HAVE_TE_GROUPED = True
except ImportError:  # pragma: no cover
    HAVE_TE_GROUPED = False

_DUMMY = 128  # the third projection's per-expert placeholder: [128, 128] bf16


class _MoonEPState:
    """Process-global MoonEP resources of one EP group: the communication Buffer and the
    NVLink prefetch (bf16) / grad-reduce (fp32) pools, shared by all MoE layers."""

    _instances: dict = {}

    def __init__(self, group, S, H, K, E, num_sms, token_padding):
        self.group = group
        self.R = group.size()
        self.rank = group.rank()
        self.S, self.H, self.K, self.E = S, H, K, E
        self.epn = E // self.R
        self.buffer = moonep.Buffer(
            S, H, K, E, self.R, num_sms=num_sms, token_padding=token_padding, group=group
        )
        self.pools = {}

    @classmethod
    def get(cls, group, S, H, K, E, num_sms=None, token_padding=128):
        key = id(group)
        inst = cls._instances.get(key)
        if inst is None or (inst.S, inst.H, inst.K, inst.E) != (S, H, K, E):
            if inst is not None:
                inst.buffer.destroy()
            inst = cls(group, S, H, K, E, num_sms, token_padding)
            cls._instances[key] = inst
        return inst

    def pool(self, name, inner, dtype):
        """[R, epn, *inner] all-rank NVLink view (rank chunks padded to VMM granularity)."""
        key = (name, tuple(inner), dtype)
        if key not in self.pools:
            padded = pad_dim0_for_alignment([self.epn, *inner], dtype)
            t = create_nvl_dist_tensor([padded, *inner], dtype, self.rank, self.R, self.group)
            t = t.view(self.R, padded, *inner)[:, : self.epn]
            t[self.rank].zero_()
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            self.pools[key] = t
        return self.pools[key]

    def weight_pools(self, shapes):
        return [self.pool(f"w{i}", s, torch.bfloat16) for i, s in enumerate(shapes)]

    def grad_pools(self, shapes):
        return [self.pool(f"g{i}", s, torch.float32) for i, s in enumerate(shapes)]

    def prefetch(self, plan, local_weights, shapes):
        pools = self.weight_pools(shapes)
        self.buffer.prefetch_weight(
            plan=plan,
            local_gate_weight=local_weights[0],
            local_up_weight=local_weights[1],
            local_down_weight=local_weights[2],
            gate_prefetch_buffer=pools[0],
            up_prefetch_buffer=pools[1],
            down_prefetch_buffer=pools[2],
        )
        return [p[self.rank] for p in pools]

    def reduce_grad(self, plan, local_grads, shapes):
        pools = self.grad_pools(shapes)
        self.buffer.reduce_grad(
            plan=plan,
            local_gate_grad=local_grads[0],
            local_up_grad=local_grads[1],
            local_down_grad=local_grads[2],
            gate_reduce_buffer=pools[0],
            up_reduce_buffer=pools[1],
            down_reduce_buffer=pools[2],
        )


def _is_shard(state, t):
    """Is ``t`` the NVLink shard itself (written in place: combine without a boundary copy)?"""
    return t.data_ptr() == state.buffer.hidden_nvsh_buffer_view.data_ptr()


class _PlanHolder:
    """Carries the current layer's MoonEP plan from the dispatch to the experts and combine."""

    def __init__(self):
        self.plan = None
        self.cu_seqlens = None


class _MoonEPDispatch(torch.autograd.Function):
    """[S, H] tokens -> [NvS, H] expert-grouped slots (+ [NvS] routing weights)."""

    @staticmethod
    def forward(ctx, hidden, weights_sk, topk, tokens_per_expert, state, holder):
        hidden_nvsh, w_nvs, cu_seqlens, plan = state.buffer.dispatch(
            hidden, weights_sk, topk, tokens_per_expert
        )
        holder.plan, holder.cu_seqlens = plan, cu_seqlens
        ctx.state, ctx.plan = state, plan
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_nvsh, w_nvs, cu_seqlens

    @staticmethod
    def backward(ctx, grad_hidden_nvsh, grad_w_nvs, _grad_cu):
        if grad_w_nvs is None:
            grad_w_nvs = torch.zeros(
                grad_hidden_nvsh.shape[0], dtype=torch.float32, device=grad_hidden_nvsh.device
            )
        grad_hidden_nvsh = grad_hidden_nvsh.contiguous()
        grad_hidden, grad_w_sk, _ = ctx.state.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden_nvsh,
            zero_copy=_is_shard(ctx.state, grad_hidden_nvsh),
            route_weights_nvs=grad_w_nvs.float().contiguous(),
        )
        return grad_hidden, grad_w_sk, None, None, None, None


class _MoonEPCombine(torch.autograd.Function):
    """[NvS, H] expert outputs -> [S, H] (K-sum per token)."""

    @staticmethod
    def forward(ctx, out_nvsh, state, holder):
        plan = holder.plan
        out_nvsh = out_nvsh.contiguous()
        out, _, _ = state.buffer.combine(
            plan=plan, hidden_nvsh=out_nvsh, zero_copy=_is_shard(state, out_nvsh)
        )
        ctx.state, ctx.plan = state, plan
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_nvsh, _, _, _ = ctx.state.buffer.dispatch(grad_out.contiguous(), plan=ctx.plan)
        return grad_nvsh, None, None


_MAX_GROUPS = 64  # TE's cuBLASLt grouped GEMM: at most 64 discrete tensors per call


def _grouped(data, splits, offsets, last_dim):
    """Rows of a packed [rows, last_dim] buffer as a varying-first-dim TE GroupedTensor: group g
    is rows [offsets[g], offsets[g] + splits[g])."""
    return GroupedTensorStorage(
        shape=(data.size(0), last_dim),
        dtype=data.dtype,
        num_tensors=splits.numel(),
        quantizer=None,
        data=data.reshape(-1),
        first_dims=splits,
        tensor_offsets=offsets * last_dim,
    )


def _gemm(weights, a, a_dim, out, out_dim, splits, offsets, layout, accumulate=False):
    """Grouped GEMM over len(weights) row groups of ``a`` / ``out``, in chunks of <= 64 groups.

    layout "TN" (fwd) / "NN" (dgrad): ``weights`` is the discrete A list, ``a`` the grouped B,
    ``out`` the grouped output. layout "NT" (wgrad): ``a`` is the grouped input, ``out`` the
    grouped output grad and ``weights`` the discrete list of weight-grad outputs."""
    n = len(weights)
    for lo in range(0, n, _MAX_GROUPS):
        hi = min(n, lo + _MAX_GROUPS)
        sp, off = splits[lo:hi], offsets[lo : hi + 1]
        if layout == "NT":
            general_grouped_gemm_for_grouped_tensor(
                _grouped(a, sp, off, a_dim), _grouped(out, sp, off, out_dim), weights[lo:hi],
                layout="NT", accumulate=accumulate,
            )
        else:
            general_grouped_gemm_for_grouped_tensor(
                weights[lo:hi], _grouped(a, sp, off, a_dim), _grouped(out, sp, off, out_dim),
                layout=layout, accumulate=accumulate,
            )


class _MoonEPExperts(torch.autograd.Function):
    """Weighted SwiGLU MLP over the 2 * epn expert slots (local experts + prefetched copies)."""

    @staticmethod
    def forward(ctx, x, w_nvs, cu_seqlens, weight1, weight2, module):
        state, holder = module.state, module.holder
        plan = holder.plan
        epn = weight1.shape[0]
        slots = state.prefetch(plan, module.local_weights(), module.pool_shapes)
        w1 = [weight1[i] for i in range(epn)] + [slots[0][i].view_as(weight1[0]) for i in range(epn)]
        w2 = [weight2[i] for i in range(epn)] + [slots[1][i].view_as(weight2[0]) for i in range(epn)]
        ends = cu_seqlens.to(torch.int64)
        offsets = torch.cat([ends.new_zeros(1), ends])
        splits = torch.diff(offsets)
        ffn2, hidden = weight1.shape[1], weight1.shape[2]
        h = torch.empty(x.shape[0], ffn2, dtype=x.dtype, device=x.device)
        _gemm(w1, x, hidden, h, ffn2, splits, offsets, "TN")
        probs = w_nvs.unsqueeze(-1)
        a = weighted_swiglu(h, probs)
        # zero copy: fc2 writes straight into the NVLink shard the combine reads (the shard is
        # free: the dispatch copied the tokens out; y is not saved for backward)
        y = state.buffer.hidden_nvsh_buffer_view
        _gemm(w2, a, ffn2 // 2, y, hidden, splits, offsets, "TN")
        ctx.save_for_backward(x, h, w_nvs, splits, offsets)
        ctx.module, ctx.plan = module, plan
        ctx.weights = (weight1, weight2)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, h, w_nvs, splits, offsets = ctx.saved_tensors
        module, plan = ctx.module, ctx.plan
        state = module.state
        weight1, weight2 = ctx.weights
        epn, ffn2, hidden = weight1.shape
        ffn = ffn2 // 2
        grad_y = grad_y.contiguous()
        # the pools are shared by all layers: fetch this layer's copies again
        slots = state.prefetch(plan, module.local_weights(), module.pool_shapes)
        w1 = [weight1[i] for i in range(epn)] + [slots[0][i].view_as(weight1[0]) for i in range(epn)]
        w2 = [weight2[i] for i in range(epn)] + [slots[1][i].view_as(weight2[0]) for i in range(epn)]
        gpools = [p[state.rank] for p in state.grad_pools(module.pool_shapes)]
        main1, main2 = module.main_grads()

        probs = w_nvs.unsqueeze(-1)
        a = weighted_swiglu(h, probs)
        # fc2: dgrad + wgrad (into the local main grads and this rank's reduce slots)
        grad_a = torch.empty(x.shape[0], ffn, dtype=x.dtype, device=x.device)
        _gemm(w2, grad_y, hidden, grad_a, ffn, splits, offsets, "NN")
        wg2 = [main2[i] for i in range(epn)] + [gpools[1][i].view_as(main2[0]) for i in range(epn)]
        _gemm(wg2, a, ffn, grad_y, hidden, splits, offsets, "NT", accumulate=True)
        grad_h, grad_probs = weighted_swiglu_back(grad_a, h, probs)
        # fc1
        # zero copy: the dispatch backward (a combine) reads fc1's dgrad from the shard
        grad_x = state.buffer.hidden_nvsh_buffer_view
        _gemm(w1, grad_h, ffn2, grad_x, hidden, splits, offsets, "NN")
        wg1 = [main1[i] for i in range(epn)] + [gpools[0][i].view_as(main1[0]) for i in range(epn)]
        _gemm(wg1, x, hidden, grad_h, ffn2, splits, offsets, "NT", accumulate=True)
        # the prefetched copies' grads -> their home ranks' main grads (collective)
        state.reduce_grad(plan, module.local_grads(main1, main2), module.pool_shapes)
        for w in (weight1, weight2):
            w.grad_added_to_main_grad = True
        dummy1 = torch.empty(1, dtype=weight1.dtype, device=weight1.device).expand_as(weight1)
        dummy2 = torch.empty(1, dtype=weight2.dtype, device=weight2.device).expand_as(weight2)
        return grad_x, grad_probs.squeeze(-1).to(w_nvs.dtype), None, dummy1, dummy2, None


class MoonEPGroupedMLP(MegatronModule):
    """The routed experts of the MoonEP backend (see the module docstring)."""

    def __init__(self, num_local_experts, config, pg_collection=None, name=None):
        super().__init__(config=config)
        assert HAVE_MOONEP, "moe_flex_dispatcher_backend='moonep' needs the moonep package"
        assert HAVE_TE_GROUPED, "the MoonEP experts need TE's grouped-tensor grouped GEMM (TE 2.19+)"
        assert config.gated_linear_unit and not config.add_bias_linear, (
            "MoonEP experts: SwiGLU without bias only"
        )
        assert not config.fp8, "MoonEP experts: bf16 only (fp8 not wired yet)"
        self.num_local_experts = epn = num_local_experts
        H, F = config.hidden_size, config.moe_ffn_hidden_size
        dev = torch.cuda.current_device()
        dtype = config.params_dtype
        self.weight1 = torch.nn.Parameter(torch.empty(epn, 2 * F, H, device=dev, dtype=dtype))
        self.weight2 = torch.nn.Parameter(torch.empty(epn, H, F, device=dev, dtype=dtype))
        if config.perform_initialization:
            for i in range(epn):
                config.init_method(self.weight1.data[i])
                config.output_layer_init_method(self.weight2.data[i])
        for p in (self.weight1, self.weight2):
            p.allreduce = False  # expert-data-parallel gradient reduction
        # plain attributes, not buffers: Float16Module would cast an fp32 buffer to bf16, and
        # MoonEP's grad reduce needs the fp32 placeholder grad
        self._dummy_w = torch.zeros(epn, _DUMMY, _DUMMY, device=dev, dtype=torch.bfloat16)
        self._dummy_g = torch.zeros(epn, _DUMMY, _DUMMY, device=dev, dtype=torch.float32)
        self.pool_shapes = [(2 * F, H), (H, F), (_DUMMY, _DUMMY)]
        self.state = None  # set by the dispatcher on its first dispatch
        self.holder = None

    def local_weights(self):
        return [self.weight1.data, self.weight2.data, self._dummy_w]

    def main_grads(self):
        mg = []
        for p in (self.weight1, self.weight2):
            g = getattr(p, "main_grad", None)
            if g is None:  # no Megatron DDP (unit tests): an fp32 grad of our own
                if getattr(p, "_moonep_grad", None) is None:
                    p._moonep_grad = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                g = p._moonep_grad
            assert g.dtype == torch.float32, "MoonEP experts need fp32 main grads"
            mg.append(g.view(p.shape))
        return mg

    def local_grads(self, main1, main2):
        return [main1, main2, self._dummy_g]

    def forward(self, permuted_hidden, tokens_per_expert, permuted_probs, **_):
        # tokens_per_expert carries MoonEP's cu_seqlens [2 * epn]
        out = _MoonEPExperts.apply(
            permuted_hidden, permuted_probs, tokens_per_expert, self.weight1, self.weight2, self
        )
        return out, None

    def backward_dw(self):
        pass


class MoEMoonEPTokenDispatcher(MoETokenDispatcher):
    """Token dispatcher of the MoonEP backend; pairs with :class:`MoonEPGroupedMLP`."""

    def __init__(self, num_local_experts, local_expert_indices, config, pg_collection=None):
        super().__init__(config=config, pg_collection=pg_collection)
        assert HAVE_MOONEP, "moe_flex_dispatcher_backend='moonep' needs the moonep package"
        assert self.tp_size == 1, "MoonEP: expert tensor parallelism is not supported"
        assert config.moe_expert_capacity_factor is None, "MoonEP: no token dropping"
        self.num_local_experts = num_local_experts
        self.holder = _PlanHolder()
        self.experts = None
        self.state = None

    def attach_experts(self, experts: MoonEPGroupedMLP):
        self.experts = experts
        experts.holder = self.holder

    def _state(self, num_tokens, hidden):
        if self.state is None or self.state.S != num_tokens:
            self.state = _MoonEPState.get(
                self.ep_group,
                num_tokens,
                hidden,
                self.config.moe_router_topk,
                self.config.num_moe_experts,
                num_sms=self.config.moe_flex_dispatcher_num_sms,
                # per-expert-slot row padding of the dispatched layout (grouped GEMM tiles)
                token_padding=int(os.environ.get("MOONEP_TOKEN_PADDING", "128")),
            )
            self.experts.state = self.state
        return self.state

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_shape[-1])
        k = self.config.moe_router_topk
        topk = probs.masked_fill(~routing_map, -1.0).topk(k, dim=-1).indices
        self.topk = topk.to(torch.int32).contiguous()
        self.tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int32).contiguous()
        weights_sk = probs.gather(-1, topk).float().contiguous()
        return hidden_states, weights_sk

    def token_dispatch(self, hidden_states, probs):
        state = self._state(hidden_states.shape[0], hidden_states.shape[-1])
        hidden_nvsh, w_nvs, cu_seqlens = _MoonEPDispatch.apply(
            hidden_states.contiguous(), probs, self.topk, self.tokens_per_expert, state,
            self.holder,
        )
        self._cu_seqlens = cu_seqlens
        return hidden_nvsh, w_nvs

    def dispatch_postprocess(self, hidden_states, probs):
        return hidden_states, self._cu_seqlens, probs

    def combine_preprocess(self, hidden_states):
        return hidden_states

    def token_combine(self, hidden_states):
        return _MoonEPCombine.apply(hidden_states, self.state, self.holder)

    def combine_postprocess(self, hidden_states):
        return hidden_states.view(self.hidden_shape)
