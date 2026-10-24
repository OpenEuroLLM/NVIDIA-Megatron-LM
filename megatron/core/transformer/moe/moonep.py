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

* Weights: TEGroupedMLP's ``linear_fc1.weight{i}`` [2F, H] (gate rows then up rows) and
  ``linear_fc2.weight{i}`` [H, F], one 2-D parameter per local expert (so angular MuOwn sees
  each expert as a matrix) and TEGroupedMLP's sharded checkpoint keys. MoonEP pushes
  ``[epn, ...]`` contiguous expert tensors: a zero-copy view when the DDP buffer holds the
  experts back to back, else a mirror refreshed once per optimizer step. MoonEP's API carries
  three projections; the third is a 32 KiB per-expert placeholder.
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
    from moonep.inter_rank_sync import launch_inter_rank_sync

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
            S, H, K, E, self.R, num_sms=num_sms, token_padding=token_padding, group=group,
            # MOONEP_PDL=0: no programmatic dependent launch between MoonEP's kernels
            enable_pdl=os.environ.get("MOONEP_PDL", "1") != "0",
        )
        self.pools = {}
        self._last_prefetch = None  # (experts module, plan) whose weights the pools hold

    @classmethod
    def get(cls, group, S, H, K, E, num_sms=None, token_padding=128, slot=0):
        """``slot``: the token chunk (MOONEP_CHUNKS) -- each chunk has its own Buffer and pools, so
        one chunk's dispatch / combine can run while another chunk's experts compute."""
        key = (id(group), slot)
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

    def prefetch(self, plan, local_weights, shapes, owner=None):
        """Push the plan's redundant experts into every rank's prefetch slots. Skipped when the
        pools still hold exactly this owner's experts for this plan: under full recompute a
        layer's recomputed forward runs right before its backward, so the backward needs no
        second push (all EP ranks see the same sequence, so they skip together)."""
        pools = self.weight_pools(shapes)
        last = self._last_prefetch
        if owner is not None and last is not None and last[0] is owner and last[1] is plan:
            return [p[self.rank] for p in pools]
        self._last_prefetch = (owner, plan)
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
        # reduce_grad PULLS: every rank reads the other ranks' reduce pools, and the caller must
        # make sure those are written (MoonEP's Buffer.reduce_grad contract). Fence all EP ranks
        # on this stream, after this rank's weight-grad GEMMs into its pools; without it a fast
        # rank read a peer's pool before the peer's fc1 wgrad landed (wrong dw1 of the
        # prefetched experts, visible with MOONEP_CHUNKS > 1, latent before).
        launch_inter_rank_sync(self.buffer._require_ctx())
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
    """Carries the current layer's MoonEP plan from the dispatch to the experts and combine.
    ``streams`` (chunked path): (compute, communication) -- the backward of a communication op
    records the tensors it shares with the compute stream (see _record)."""

    def __init__(self):
        self.plan = None
        self.cu_seqlens = None
        self.streams = None


# MOONEP_CHUNK_DEBUG (chunked path): "keepalive" keeps every tensor the path touches alive until the
# next optimizer step (rules out allocator lifetimes); "serial" makes the compute stream wait for each
# communication op right away (same kernels, no concurrency); "fullserial" also makes the
# communication stream wait for the compute stream before each op (two streams, total order).
_CHUNK_DEBUG = os.environ.get("MOONEP_CHUNK_DEBUG", "")
_KEEPALIVE = {"step": -1, "tensors": []}


def _keep(*tensors):
    if _CHUNK_DEBUG != "keepalive":
        return
    if _KEEPALIVE["step"] != _OPTIMIZER_STEPS[0]:
        _KEEPALIVE["step"], _KEEPALIVE["tensors"] = _OPTIMIZER_STEPS[0], []
    _KEEPALIVE["tensors"].extend(t for t in tensors if isinstance(t, torch.Tensor))


def _record(state, stream, *tensors):
    """record_stream for allocator-owned tensors used across the chunked path's two streams (the
    caching allocator otherwise reuses -- or, with expandable segments, unmaps -- a block whose
    Python reference is gone while the other stream still reads it). NVLink shard views are not
    allocator memory and are skipped."""
    if stream is None:
        return
    for t in tensors:
        if t is not None and t.numel() and not _is_shard(state, t):
            t.record_stream(stream)


class _MoonEPDispatch(torch.autograd.Function):
    """[S, H] tokens -> [NvS, H] expert-grouped slots (+ [NvS] routing weights)."""

    @staticmethod
    def forward(ctx, hidden, weights_sk, topk, tokens_per_expert, state, holder):
        hidden_nvsh, w_nvs, cu_seqlens, plan = state.buffer.dispatch(
            hidden, weights_sk, topk, tokens_per_expert
        )
        holder.plan, holder.cu_seqlens = plan, cu_seqlens
        ctx.state, ctx.plan, ctx.streams = state, plan, holder.streams
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_nvsh, w_nvs, cu_seqlens

    @staticmethod
    def backward(ctx, grad_hidden_nvsh, grad_w_nvs, _grad_cu):
        if grad_w_nvs is None:
            grad_w_nvs = torch.zeros(
                grad_hidden_nvsh.shape[0], dtype=torch.float32, device=grad_hidden_nvsh.device
            )
        grad_hidden_nvsh = grad_hidden_nvsh.contiguous()
        grad_w_nvs = grad_w_nvs.float().contiguous()
        if ctx.streams is not None:  # chunked: this runs on the communication stream
            compute, comm = ctx.streams
            _record(ctx.state, comm, grad_hidden_nvsh, grad_w_nvs)
            if _CHUNK_DEBUG == "fullserial":
                comm.wait_stream(compute)
        grad_hidden, grad_w_sk, _ = ctx.state.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden_nvsh,
            zero_copy=_is_shard(ctx.state, grad_hidden_nvsh),
            route_weights_nvs=grad_w_nvs,
        )
        if ctx.streams is not None:
            _record(ctx.state, compute, grad_hidden, grad_w_sk)
            _keep(grad_hidden_nvsh, grad_w_nvs, grad_hidden, grad_w_sk)
            if _CHUNK_DEBUG in ("serial", "fullserial"):
                compute.wait_stream(comm)
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
        ctx.state, ctx.plan, ctx.streams = state, plan, holder.streams
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # zero copy: the experts' backward reads the output grad from the NVLink shard (fc2's
        # dgrad / wgrad consume it before fc1's dgrad overwrites the shard)
        grad_out = grad_out.contiguous()
        if ctx.streams is not None:  # chunked: this runs on the communication stream
            compute, comm = ctx.streams
            _record(ctx.state, comm, grad_out)
            if _CHUNK_DEBUG == "fullserial":
                comm.wait_stream(compute)
        grad_nvsh, _, _, _ = ctx.state.buffer.dispatch(grad_out, plan=ctx.plan, zero_copy=True)
        if ctx.streams is not None:
            _record(ctx.state, compute, grad_nvsh)
            _keep(grad_out, grad_nvsh)
            if _CHUNK_DEBUG in ("serial", "fullserial"):
                compute.wait_stream(comm)
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


def _wgrad(main, gpool, a, a_dim, dy, dy_dim, splits, offsets):
    """Weight grads: the local experts accumulate into their fp32 main grads (a list, one per
    expert); the prefetch slots are written fresh into this rank's reduce slots (cleared after
    each reduce; mostly empty, and an empty group with accumulate=True still reads + writes its
    whole fp32 block)."""
    epn = len(main)
    _gemm(main, a, a_dim, dy, dy_dim, splits[:epn], offsets[: epn + 1], "NT", accumulate=True)
    _gemm([gpool[i].view_as(main[0]) for i in range(epn)], a, a_dim, dy, dy_dim, splits[epn:],
          offsets[epn:], "NT", accumulate=False)


# Optimizer steps so far (every torch optimizer's step, via a global post-step hook): the
# contiguous weight mirrors of the experts are refreshed once per step, lazily, at the first
# forward after one (see MoonEPGroupedMLP.local_weights).
_OPTIMIZER_STEPS = [0]


def _count_optimizer_step(optimizer, args, kwargs):
    _OPTIMIZER_STEPS[0] += 1


from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402

register_optimizer_step_post_hook(_count_optimizer_step)


# Fallback buffers when the weights / main grads are not back to back (no DDP buffer, or a
# layout that ignores contiguous_group): ONE per shape for all layers -- (owner, step) tells
# whether the weight mirror still holds this layer's weights of this optimizer step.
_SHARED = {}


def _shared(name, shape, dtype, device):
    key = (name, tuple(shape), dtype)
    buf = _SHARED.get(key)
    if buf is None:
        buf = _SHARED[key] = {"t": torch.empty(*shape, dtype=dtype, device=device), "tag": None}
    return buf


def _stacked_view(tensors):
    """``[n, *shape]`` zero-copy view of ``n`` same-shape tensors that lie back to back in one
    storage (Megatron's DDP buffers place one module's parameters / main grads that way when
    no bucket boundary falls between them), else None."""
    t0 = tensors[0]
    nbytes = t0.numel() * t0.element_size()
    base = t0.untyped_storage().data_ptr()
    for i, t in enumerate(tensors):
        if (
            t.dtype != t0.dtype
            or t.shape != t0.shape
            or not t.is_contiguous()
            or t.untyped_storage().data_ptr() != base
            or t.data_ptr() != t0.data_ptr() + i * nbytes
        ):
            return None
    return torch.as_strided(
        t0.detach(), (len(tensors), *t0.shape), (t0.numel(), *t0.stride()), t0.storage_offset()
    )


class _MoonEPExperts(torch.autograd.Function):
    """Weighted SwiGLU MLP over the 2 * epn expert slots (local experts + prefetched copies).
    ``weights``: the fc1 then the fc2 expert parameters (one 2-D tensor per expert)."""

    @staticmethod
    def forward(ctx, x, w_nvs, cu_seqlens, module, state, holder, *weights):
        plan = holder.plan
        epn = module.num_local_experts
        fc1, fc2 = [w.detach() for w in weights[:epn]], [w.detach() for w in weights[epn:]]
        slots = state.prefetch(plan, module.local_weights(), module.pool_shapes, owner=module)
        w1 = fc1 + [slots[0][i].view_as(fc1[0]) for i in range(epn)]
        w2 = fc2 + [slots[1][i].view_as(fc2[0]) for i in range(epn)]
        ends = cu_seqlens.to(torch.int64)
        offsets = torch.cat([ends.new_zeros(1), ends])
        splits = torch.diff(offsets)
        ffn2, hidden = fc1[0].shape
        h = torch.empty(x.shape[0], ffn2, dtype=x.dtype, device=x.device)
        _gemm(w1, x, hidden, h, ffn2, splits, offsets, "TN")
        probs = w_nvs.unsqueeze(-1)
        a = weighted_swiglu(h, probs)
        # zero copy: fc2 writes straight into the NVLink shard the combine reads (the shard is
        # free: the dispatch copied the tokens out; y is not saved for backward)
        y = state.buffer.hidden_nvsh_buffer_view
        _gemm(w2, a, ffn2 // 2, y, hidden, splits, offsets, "TN")
        ctx.save_for_backward(x, h, a, w_nvs, splits, offsets)
        ctx.module, ctx.plan, ctx.state, ctx.streams = module, plan, state, holder.streams
        ctx.weights = weights
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, h, a, w_nvs, splits, offsets = ctx.saved_tensors
        module, plan, state = ctx.module, ctx.plan, ctx.state
        weights = ctx.weights
        epn = module.num_local_experts
        fc1, fc2 = [w.detach() for w in weights[:epn]], [w.detach() for w in weights[epn:]]
        ffn2, hidden = fc1[0].shape
        ffn = ffn2 // 2
        grad_y = grad_y.contiguous()
        # the pools are shared by all layers: fetch this layer's copies again
        slots = state.prefetch(plan, module.local_weights(), module.pool_shapes, owner=module)
        w1 = fc1 + [slots[0][i].view_as(fc1[0]) for i in range(epn)]
        w2 = fc2 + [slots[1][i].view_as(fc2[0]) for i in range(epn)]
        gpools = [p[state.rank] for p in state.grad_pools(module.pool_shapes)]
        main1, main2 = module.main_grads()

        probs = w_nvs.unsqueeze(-1)
        # fc2: dgrad + wgrad (into the local main grads and this rank's reduce slots)
        grad_a = torch.empty(x.shape[0], ffn, dtype=x.dtype, device=x.device)
        _gemm(w2, grad_y, hidden, grad_a, ffn, splits, offsets, "NN")
        _wgrad(main2, gpools[1], a, ffn, grad_y, hidden, splits, offsets)
        grad_h, grad_probs = weighted_swiglu_back(grad_a, h, probs)
        # fc1
        # zero copy: the dispatch backward (a combine) reads fc1's dgrad from the shard
        grad_x = state.buffer.hidden_nvsh_buffer_view
        _gemm(w1, grad_h, ffn2, grad_x, hidden, splits, offsets, "NN")
        _wgrad(main1, gpools[0], x, hidden, grad_h, ffn2, splits, offsets)
        # the prefetched copies' grads -> their home ranks' main grads (collective)
        module.reduce_copies_grad(state, plan, main1, main2)
        grad_w = grad_probs.squeeze(-1).to(w_nvs.dtype)
        if ctx.streams is not None:  # chunked: the dispatch backward reads it on the comm stream
            _record(state, ctx.streams[1], grad_w)
            _keep(grad_w, grad_y, grad_a, grad_h)
        dummies = []
        for w in weights:
            w.grad_added_to_main_grad = True
            dummies.append(torch.empty(1, dtype=w.dtype, device=w.device).expand_as(w))
        return (grad_x, grad_w, None, None, None, None, *dummies)


class _ExpertWeights(torch.nn.Module):
    """``weight0 .. weight{n-1}``: one 2-D parameter per local expert, TEGroupedMLP's names and
    shapes (2-D: the emerging optimizers -- angular MuOwn -- orthogonalize each expert; a 3-D
    stack would fall back to Adam). Registered in expert order, as TEGroupedMLP: the overlapped
    param all-gather chains its buckets in registration order, and a reversed registration (which
    would have laid them out back to back in expert order in the DDP buffer) broke that chain."""

    def __init__(self, n, shape, init_method, perform_init, dtype, device):
        super().__init__()
        self.num = n
        for i in range(n):
            w = torch.nn.Parameter(torch.empty(*shape, device=device, dtype=dtype))
            if perform_init:
                init_method(w.data)
            w.allreduce = False  # expert-data-parallel gradient reduction
            # one back-to-back block in the DDP buffers, in expert order (param_layout.
            # group_contiguous): MoonEP reads [n, ...] views of the weights and main grads
            w.contiguous_group = (id(self), i, n)
            self.register_parameter(f"weight{i}", w)

    def weights(self):
        return [getattr(self, f"weight{i}") for i in range(self.num)]

    def forward(self):
        """The weights, through ``__call__``: Megatron's DDP finishes a bucket's parameter
        all-gather (overlap_param_gather) in a forward pre-hook of the module that owns the
        parameters, so they have to be fetched this way before every use."""
        return self.weights()


class MoonEPGroupedMLP(MegatronModule):
    """The routed experts of the MoonEP backend (see the module docstring).

    Parameters as TEGroupedMLP: ``linear_fc1.weight{i}`` [2F, H] (gate rows then up rows) and
    ``linear_fc2.weight{i}`` [H, F], one per local expert, with the same sharded checkpoint keys
    (global expert axis). MoonEP pushes ``[epn, ...]`` contiguous weights to the other ranks and
    reduces into contiguous fp32 grads: zero-copy views when the DDP buffers hold the experts back
    to back, else a weight mirror refreshed once per optimizer step and an fp32 scratch."""

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
        init = config.perform_initialization
        self.linear_fc1 = _ExpertWeights(epn, (2 * F, H), config.init_method, init, dtype, dev)
        self.linear_fc2 = _ExpertWeights(
            epn, (H, F), config.output_layer_init_method, init, dtype, dev
        )
        self.ep_group = getattr(pg_collection, "ep", None) if pg_collection is not None else None
        self.tp_group = getattr(pg_collection, "expt_tp", None) if pg_collection is not None else None
        # the experts' replicate group (checkpoint replica ids), as TEGroupedMLP's
        self.expt_dp_group = (
            getattr(pg_collection, "expt_dp", None) if pg_collection is not None else None
        )
        # plain attributes, not buffers: Float16Module would cast an fp32 buffer to bf16, and
        # MoonEP's grad reduce needs the fp32 placeholder grad
        self._dummy_w = torch.zeros(epn, _DUMMY, _DUMMY, device=dev, dtype=torch.bfloat16)
        self._dummy_g = torch.zeros(epn, _DUMMY, _DUMMY, device=dev, dtype=torch.float32)
        self.pool_shapes = [(2 * F, H), (H, F), (_DUMMY, _DUMMY)]
        self.state = None  # set by the dispatcher on its first dispatch
        self.holder = None

    def weights(self):
        return self.linear_fc1.weights() + self.linear_fc2.weights()

    def local_weights(self):
        """[epn, 2F, H] / [epn, H, F] contiguous local weights for MoonEP's weight push."""
        fc1 = [w.detach() for w in self.linear_fc1.weights()]
        fc2 = [w.detach() for w in self.linear_fc2.weights()]
        out = []
        for name, ws in (("w1", fc1), ("w2", fc2)):
            v = _stacked_view(ws)
            if v is None:
                buf = _shared(name, (len(ws), *ws[0].shape), ws[0].dtype, ws[0].device)
                tag = (id(self), _OPTIMIZER_STEPS[0])
                if buf["tag"] != tag:
                    torch._foreach_copy_(list(buf["t"].unbind(0)), ws)
                    buf["tag"] = tag
                v = buf["t"]
            out.append(v)
        return [*out, self._dummy_w]

    def main_grads(self):
        """Per-expert fp32 main grads ([2F, H] list, [H, F] list)."""
        out = []
        for mod in (self.linear_fc1, self.linear_fc2):
            mg = []
            for p in mod.weights():
                g = getattr(p, "main_grad", None)
                if g is None:  # no Megatron DDP (unit tests): an fp32 grad of our own
                    if getattr(p, "_moonep_grad", None) is None:
                        p._moonep_grad = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                    g = p._moonep_grad
                assert g.dtype == torch.float32, "MoonEP experts need fp32 main grads"
                mg.append(g.view(p.shape))
            out.append(mg)
        return out

    def reduce_copies_grad(self, state, plan, main1, main2):
        """The prefetched copies' weight grads -> their home ranks' main grads (MoonEP's reduce
        accumulates into contiguous [epn, ...] fp32 grads: the main grads themselves when they
        lie back to back, else a zeroed scratch added into them afterwards)."""
        g1, g2 = _stacked_view(main1), _stacked_view(main2)
        if g1 is not None and g2 is not None:
            state.reduce_grad(plan, [g1, g2, self._dummy_g], self.pool_shapes)
            return
        s1 = _shared("g1", (len(main1), *main1[0].shape), torch.float32, main1[0].device)["t"]
        s2 = _shared("g2", (len(main2), *main2[0].shape), torch.float32, main2[0].device)["t"]
        s1.zero_()
        s2.zero_()
        state.reduce_grad(plan, [s1, s2, self._dummy_g], self.pool_shapes)
        torch._foreach_add_(main1, list(s1.unbind(0)))
        torch._foreach_add_(main2, list(s2.unbind(0)))

    def _gathered_weights(self):
        # through the submodules' __call__ (DDP's param-gather pre-hooks, see _ExpertWeights)
        return self.linear_fc1() + self.linear_fc2()

    def forward(self, permuted_hidden, tokens_per_expert, permuted_probs, **_):
        # tokens_per_expert carries MoonEP's cu_seqlens [2 * epn]
        out = _MoonEPExperts.apply(
            permuted_hidden, permuted_probs, tokens_per_expert, self, self.state, self.holder,
            *self._gathered_weights(),
        )
        return out, None

    def chunk_forward(self, x, w_nvs, cu_seqlens, state, holder):
        """The experts over one token chunk's dispatched slots (MOONEP_CHUNKS > 1)."""
        return _MoonEPExperts.apply(
            x, w_nvs, cu_seqlens, self, state, holder, *self._gathered_weights()
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """TEGroupedMLP's sharded keys: ``{prefix}experts.linear_fc{1,2}.weight`` with a global
        expert axis (fc1 through the SwiGLU factory), so TEGroupedMLP checkpoints load into
        MoonEP experts (theirs also carry TE's per-GEMM ``_extra_state``, which this has not)."""
        from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
        from megatron.core.transformer.utils import (
            ensure_metadata_has_dp_cp_group,
            make_sharded_tensors_for_checkpoint,
        )

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        ep = self.ep_group
        ep_size = ep.size() if ep is not None else 1
        ep_rank = ep.rank() if ep is not None else 0
        epn = self.num_local_experts
        num_global, offset = ep_size * epn, ep_rank * epn
        ep_axis = len(sharded_offsets)
        out = {}
        for name, mod in (("linear_fc1", self.linear_fc1), ("linear_fc2", self.linear_fc2)):
            for i, w in enumerate(mod.weights()):
                offs = (*sharded_offsets, (ep_axis, offset + i, num_global))
                sub = make_sharded_tensors_for_checkpoint(
                    {"weight": w}, f"{prefix}experts.{name}.", {}, offs,
                    tp_group=self.tp_group, dp_cp_group=metadata["dp_cp_group"],
                )
                st = sub[f"{prefix}experts.{name}.weight"]
                # The DP replica id over the expert-DP group, as TEGroupedLinear: the dense
                # dp_cp rank made only EP rank 0's experts main replicas (EP > 1: every
                # torch_dist save failed the sharding validation).
                if self.expt_dp_group is not None and not getattr(
                    st, "is_data_parallel_fully_shard", False
                ):
                    st.replica_id = (*st.replica_id[:2], self.expt_dp_group.rank())
                if name == "linear_fc1":
                    st = apply_swiglu_sharded_factory(
                        st, offs, tp_group=self.tp_group, dp_group=metadata["dp_cp_group"]
                    )
                out[f"{prefix}{name}.weight{i}"] = st
        return out

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
        # MOONEP_CHUNKS = n > 1: the layer's tokens are split into n chunks, pipelined so that a
        # chunk's dispatch / combine (on a communication stream) runs under another chunk's expert
        # GEMMs; the shared expert runs under the first dispatch (see chunked_compute).
        self.num_chunks = int(os.environ.get("MOONEP_CHUNKS", "1"))
        # debug: MOONEP_FORCE_CHUNKED=1 sends even 1 chunk through chunked_compute
        self.force_chunked = os.environ.get("MOONEP_FORCE_CHUNKED", "0") == "1"
        self.chunk_holders = [_PlanHolder() for _ in range(self.num_chunks)]
        self.chunk_states = [None] * self.num_chunks
        self._comm_stream = None
        self._chunked_shared_output = None

    def attach_experts(self, experts: MoonEPGroupedMLP):
        self.experts = experts
        experts.holder = self.holder

    def _new_state(self, num_tokens, hidden, slot=0):
        return _MoonEPState.get(
            self.ep_group,
            num_tokens,
            hidden,
            self.config.moe_router_topk,
            self.config.num_moe_experts,
            num_sms=self.config.moe_flex_dispatcher_num_sms,
            # per-expert-slot row padding of the dispatched layout (grouped GEMM tiles)
            token_padding=int(os.environ.get("MOONEP_TOKEN_PADDING", "128")),
            slot=slot,
        )

    def _state(self, num_tokens, hidden):
        if self.state is None or self.state.S != num_tokens:
            self.state = self._new_state(num_tokens, hidden)
            self.experts.state = self.state
        return self.state

    def _chunk_state(self, c, num_tokens, hidden):
        st = self.chunk_states[c]
        if st is None or st.S != num_tokens:
            st = self.chunk_states[c] = self._new_state(num_tokens, hidden, slot=c)
        return st

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_shape[-1])
        k = self.config.moe_router_topk
        topk = probs.masked_fill(~routing_map, -1.0).topk(k, dim=-1).indices
        self.topk = topk.to(torch.int32).contiguous()
        self.routing_map = routing_map
        self.tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int32).contiguous()
        weights_sk = probs.gather(-1, topk).float().contiguous()
        return hidden_states, weights_sk

    def token_dispatch(self, hidden_states, probs):
        state = self._state(hidden_states.shape[0], hidden_states.shape[-1])
        # moe_shared_expert_overlap: the shared expert's fc1 runs on its own stream while the
        # dispatch kernels (on this stream, num_sms SMs) move the tokens; fc2 overlaps the
        # combine. Same call order as the flex dispatcher.
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        hidden_nvsh, w_nvs, cu_seqlens = _MoonEPDispatch.apply(
            hidden_states.contiguous(), probs, self.topk, self.tokens_per_expert, state,
            self.holder,
        )
        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(
                hidden_states.view(self.hidden_shape), wait_current_stream=False
            )
            self.shared_experts.linear_fc1_forward_and_act(hidden_nvsh)
        self._cu_seqlens = cu_seqlens
        return hidden_nvsh, w_nvs

    def dispatch_postprocess(self, hidden_states, probs):
        return hidden_states, self._cu_seqlens, probs

    def combine_preprocess(self, hidden_states):
        return hidden_states

    def token_combine(self, hidden_states):
        # the shared expert's fc2 must not overlap the routed experts' GEMMs
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        return _MoonEPCombine.apply(hidden_states, self.state, self.holder)

    def chunked_compute(self, hidden_states, weights_sk, shared_experts=None):
        """dispatch -> experts -> combine over ``num_chunks`` token chunks, pipelined.

        Communication stream: D0 D1 .. C0 C1 ..; compute stream: (shared expert) E0 E1 ..; chunk c's
        experts wait for D_c, C_c waits for E_c, so D_{c+1} runs under E_c and C_c under E_{c+1}.
        Every op is an autograd Function recorded on its stream, so the backward replays on the
        same streams in reverse (C1' | E1' + C0' | E0' + D1' | D0') with the same overlap. Each
        chunk has its own MoonEP Buffer and pools (_MoonEPState slot), so chunks never share a
        shard. Returns the routed output [S, H]; the shared expert's output (computed under D0
        on the compute stream) is added in combine_postprocess."""
        n = self.num_chunks
        S, H = hidden_states.shape
        assert S % n == 0, f"MOONEP_CHUNKS={n} must divide the {S} tokens per micro-batch"
        Sc = S // n
        cur = torch.cuda.current_stream()
        if self._comm_stream is None:
            # debug: MOONEP_CHUNK_STREAM=same runs the communication on the compute stream
            same = os.environ.get("MOONEP_CHUNK_STREAM", "") == "same"
            self._comm_stream = cur if same else torch.cuda.Stream()
        comm = self._comm_stream
        hidden_states = hidden_states.contiguous()
        weights_sk = weights_sk.contiguous()
        # every input of the communication stream is produced on this stream BEFORE the wait
        # (a chunk histogram computed after it raced the dispatch's planning: illegal address /
        # instruction in training, invisible with CUDA_LAUNCH_BLOCKING or one stream)
        tpes = [
            r.sum(dim=0, dtype=torch.int32).contiguous() for r in self.routing_map.split(S // n)
        ]
        tks = [t.contiguous() for t in self.topk.split(S // n)]
        wss = [w.contiguous() for w in weights_sk.split(S // n)]
        comm.wait_stream(cur)
        # Everything the communication stream reads that this stream allocated: without the
        # record the caching allocator may hand out (or, with expandable segments, unmap) the
        # block once Python drops it while a chunk's dispatch still reads it -- illegal
        # address under memory pressure (8-node runs), invisible with CUDA_LAUNCH_BLOCKING.
        for t in (hidden_states, weights_sk, self.topk, self.routing_map):
            t.record_stream(comm)
        hs = hidden_states.split(Sc)
        states = [self._chunk_state(c, Sc, H) for c in range(n)]
        for holder in self.chunk_holders:
            holder.streams = (cur, comm)
        dispatched = []
        for c in range(n):
            tpe = tpes[c]
            if _CHUNK_DEBUG == "fullserial":
                comm.wait_stream(cur)
            with torch.cuda.stream(comm):
                x, w, cu = _MoonEPDispatch.apply(
                    hs[c], wss[c], tks[c], tpe, states[c],
                    self.chunk_holders[c],
                )
            ev = torch.cuda.Event()
            ev.record(comm)
            if _CHUNK_DEBUG in ("serial", "fullserial"):
                cur.wait_event(ev)
            _keep(hs[c], wss[c], tks[c], tpe, x, w, cu)
            dispatched.append((x, w, cu, ev, tpe))
        self._chunked_shared_output = (
            shared_experts(hidden_states) if shared_experts is not None else None
        )
        outs = []
        for c in range(n):
            x, w, cu, ev, tpe = dispatched[c]
            cur.wait_event(ev)
            for t in (x, w, cu):
                t.record_stream(cur)
            y = self.experts.chunk_forward(x, w, cu, states[c], self.chunk_holders[c])
            ev = torch.cuda.Event()
            ev.record(cur)
            comm.wait_event(ev)
            if _CHUNK_DEBUG == "fullserial":
                comm.wait_stream(cur)
            with torch.cuda.stream(comm):
                outs.append(_MoonEPCombine.apply(y, states[c], self.chunk_holders[c]))
            if _CHUNK_DEBUG in ("serial", "fullserial"):
                cur.wait_stream(comm)
            _keep(y, outs[-1])
            y.record_stream(comm)
            tpe.record_stream(comm)
        cur.wait_stream(comm)
        _keep(hidden_states, weights_sk, self.topk, self.routing_map)
        for o in outs:
            o.record_stream(cur)
        self.holder.plan = self.chunk_holders[-1].plan  # for callers that inspect the plan
        return torch.cat(outs)

    def combine_postprocess(self, hidden_states):
        hidden_states = hidden_states.view(self.hidden_shape)
        if self._chunked_shared_output is not None:
            hidden_states = hidden_states + self._chunked_shared_output.view(self.hidden_shape)
            self._chunked_shared_output = None
            return hidden_states
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(hidden_states)
            self.shared_experts.post_forward_comm()
            hidden_states = hidden_states + self.shared_experts.get_output()
        return hidden_states
