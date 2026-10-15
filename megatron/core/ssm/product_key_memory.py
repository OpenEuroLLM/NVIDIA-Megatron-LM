# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Product-key memory (HybridModel symbol `P`).

Lample et al., "Large Memory Layers with Product Keys" (2019), as scaled up in
Berges et al., "Memory Layers at Scale" (2024). Per token and memory head:

1. a query ``q = W_q RMSNorm(h)`` is split into halves ``q1, q2`` (each RMS-normalized);
2. each half is scored against its own ``n_keys`` sub-keys; the top-k of each half
   are combined into k*k candidates ``(i, j)`` scoring ``s1_i + s2_j`` -- an implicit
   key set of ``n_keys**2`` entries searched with ``2 n_keys`` dot products;
3. the top-k candidates (row ``i * n_keys + j`` of the values table) are read with
   softmax weights; the heads' reads are summed;
4. the result is projected back, optionally gated (``pkm_gated``, Memory+):
   ``y = W_o (read * SiLU(W_g RMSNorm(h)))``; the layer returns ``h + y``.

The values table (``n_keys**2`` rows of ``pkm_value_dim``) holds almost all of the
layer's parameters and each token reads only ``heads x topk`` rows:

* ``pkm_table_parallel="tp"``: rows sharded over tensor parallelism; each rank sums
  the rows it owns and the partial sums ([tokens, value_dim], not the rows) are
  all-reduced. A regular parameter (dense gradient, reduced by DDP), marked
  ``is_memory_table`` (``pkm_lr_mult``, no weight decay).
* ``pkm_table_parallel="dp"``: rows sharded over every rank of the stage, looked up
  with the deduplicating all-to-all of engram_sharded.py and updated from sparse row
  gradients by the chained EngramTableOptimizer (with ``pkm_lr_mult``).

Queries, keys and projections are small and replicated across TP (tp_replicated.py),
so sequence parallelism works. The layer is per-token, so context parallelism and
packed sequences need nothing special.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.tp_replicated import (
    gather_replicated_input,
    replicate_across_tp,
    scatter_replicated_output,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def product_key_topk(
    q: torch.Tensor, keys1: torch.Tensor, keys2: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Product-key search.

    Args:
        q: ``[T, H, k_dim]`` queries.
        keys1, keys2: ``[H, n_keys, k_dim / 2]`` sub-keys.
        topk: candidates kept per half and in the end.

    Returns:
        ``(scores [T, H, topk], rows [T, H, topk])`` with rows in ``[0, n_keys**2)``.
    """
    n_keys = keys1.shape[1]
    q1, q2 = q.chunk(2, dim=-1)
    s1 = torch.einsum("thd,hnd->thn", q1.float(), keys1.float())
    s2 = torch.einsum("thd,hnd->thn", q2.float(), keys2.float())
    v1, i1 = s1.topk(topk, dim=-1)
    v2, i2 = s2.topk(topk, dim=-1)
    combined = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)  # [T, H, k*k]
    scores, flat = combined.topk(topk, dim=-1)
    rows = i1.gather(-1, flat // topk) * n_keys + i2.gather(-1, flat % topk)
    return scores, rows


class ProductKeyMemoryLayer(MegatronModule):
    """``h + PKM(h)``; see the module docstring."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ):
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided"
        self.config = config
        self.layer_number = layer_number
        self.tp_group = pg_collection.tp
        tp_size, tp_rank = self.tp_group.size(), self.tp_group.rank()
        h = config.hidden_size
        self.heads = config.pkm_heads
        self.n_keys = config.pkm_num_keys
        self.k_dim = config.pkm_key_dim
        self.topk = config.pkm_topk
        self.value_dim = config.pkm_value_dim or h
        if self.k_dim % 2:
            raise ValueError("pkm_key_dim must be even (it is split into two halves)")
        if self.topk > self.n_keys:
            raise ValueError("pkm_topk must not exceed pkm_num_keys")
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        half = self.k_dim // 2

        self.norm = _RMSNorm(h, config.layernorm_epsilon).to(device)
        self.query_proj = nn.Linear(
            h, self.heads * self.k_dim, bias=False, device=device, dtype=dtype
        )
        self.query_norm = _RMSNorm(half, config.layernorm_epsilon).to(device)
        self.keys1 = nn.Parameter(torch.empty(self.heads, self.n_keys, half, device=device))
        self.keys2 = nn.Parameter(torch.empty(self.heads, self.n_keys, half, device=device))
        self.gate_proj = (
            nn.Linear(h, self.value_dim, bias=False, device=device, dtype=dtype)
            if config.pkm_gated
            else None
        )
        self.out_proj = nn.Linear(self.value_dim, h, bias=False, device=device, dtype=dtype)

        from megatron.core.ssm.engram_sharded import balanced_row_range

        rows = self.n_keys**2
        self.total_rows = rows
        self.sharded = config.pkm_table_parallel == "dp"
        if self.sharded:
            group = getattr(pg_collection, "tp_dp_cp", None)
            if group is None:  # migration fallback: collections built before tp_dp_cp
                from megatron.core import parallel_state

                group = parallel_state.get_tensor_and_data_parallel_group(
                    with_context_parallel=True
                )
            self.shard_group = group
            self.shard_size, self.shard_rank = group.size(), group.rank()
            self.row_start, row_end = balanced_row_range(rows, self.shard_rank, self.shard_size)
            self.values = nn.Parameter(
                torch.empty(row_end - self.row_start, self.value_dim, device=device),
                requires_grad=False,
            )
            # A plain attribute, not a buffer: Float16Module's cast must not touch it.
            self.values_grad = torch.zeros_like(self.values, dtype=torch.float32)
            self.values.engram_sharded_table = True
            self.values.engram_grad = self.values_grad
            self.values.sharded_table_kind = "pkm"  # its optimizer applies pkm_lr_mult
            self.table_grad = self.values_grad  # sharded_lookup's name (a plain tensor)
        else:
            self.row_start, row_end = balanced_row_range(rows, tp_rank, tp_size)
            self.values = nn.Parameter(
                torch.empty(row_end - self.row_start, self.value_dim, device=device, dtype=dtype)
            )
            self.values.is_memory_table = True
        self.reset_parameters()
        # mHC / Block AttnRes / nGPT replace `h + PKM(h)` (residual_scheme_adapter.py).
        from megatron.core.transformer.residual_scheme_adapter import (
            ResidualSchemeAdapter,
            residual_scheme_active,
        )

        self.residual_scheme = (
            ResidualSchemeAdapter(config) if residual_scheme_active(config) else None
        )
        self._depth_state = None
        for module in (self.norm, self.query_proj, self.query_norm, self.gate_proj, self.out_proj):
            if module is not None:
                replicate_across_tp(module, self.tp_group)
        for p in (self.keys1, self.keys2):
            p.average_gradients_across_tp_domain = True
            if tp_size > 1:
                with torch.no_grad():
                    torch.distributed.broadcast(
                        p, torch.distributed.get_global_rank(self.tp_group, 0), group=self.tp_group
                    )

    @property
    def table(self):
        """The values table under the name engram_sharded.sharded_lookup reads (a property,
        so the parameter is not registered twice)."""
        return self.values

    def _apply(self, fn, recurse=True):
        """Keep a "dp"-sharded fp32 values table out of the model's bf16/fp16 cast."""
        if not self.sharded or "values" not in self._parameters:
            return super()._apply(fn, recurse)
        probe = fn(torch.zeros(1, dtype=torch.float32, device=self.values.device))
        if probe.dtype == torch.float32:
            return super()._apply(fn, recurse)
        values = self._parameters.pop("values")
        try:
            super()._apply(fn, recurse)
        finally:
            if values.device != probe.device:
                values.data = values.data.to(probe.device)
            self._parameters["values"] = values
        return self

    def reset_parameters(self):
        if not self.config.perform_initialization:
            return
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.k_dim // 2)
            self.keys1.uniform_(-bound, bound)
            self.keys2.uniform_(-bound, bound)
            # Values: every rank its own rows (distinct seeds across the shard group).
            rank = self.shard_rank if self.sharded else self.tp_group.rank()
            gen = torch.Generator(device=self.values.device)
            gen.manual_seed(2_000_003 * int(self.layer_number or 0) + 7919 * rank + 29)
            self.values.normal_(mean=0.0, std=self.value_dim**-0.5, generator=gen)
        self.config.init_method(self.query_proj.weight)
        if self.gate_proj is not None:
            self.config.init_method(self.gate_proj.weight)
        self.config.output_layer_init_method(self.out_proj.weight)

    def _read(self, rows: torch.Tensor, weights: torch.Tensor, dtype) -> torch.Tensor:
        """Weighted sum of value rows: rows/weights ``[T, H, k]`` -> ``[T, value_dim]``.

        Queries, keys and weights are computed replicated across TP, but each rank reads
        only part of the values (its rows in "tp" mode, its token slice in "dp" mode);
        the mappings below make every gradient count exactly once on every rank.
        """
        tp = self.tp_group.size()
        if self.sharded:
            from megatron.core.ssm.engram_sharded import sharded_lookup

            if tp > 1:  # each TP rank reads its token slice; weight grads are all-gathered
                if rows.shape[0] % tp:
                    raise ValueError("pkm_table_parallel='dp' needs tokens divisible by TP")
                rows = rows.chunk(tp)[self.tp_group.rank()]
                weights = scatter_to_sequence_parallel_region(weights, group=self.tp_group)
            vals = sharded_lookup(self, rows.reshape(-1), dtype).view(*rows.shape, self.value_dim)
            out = (vals * weights.unsqueeze(-1).to(dtype)).sum(dim=(1, 2))
            if tp > 1:
                out = gather_from_sequence_parallel_region(
                    out, tensor_parallel_output_grad=False, group=self.tp_group
                )
            return out
        if tp > 1:  # every rank's partial read contributes to the weights' gradient
            weights = copy_to_tensor_model_parallel_region(weights, group=self.tp_group)
        local = rows - self.row_start
        outside = (local < 0) | (local >= self.values.shape[0])
        vals = F.embedding(local.masked_fill(outside, 0).reshape(-1), self.values)
        vals = vals.view(*rows.shape, self.value_dim)
        w = weights.masked_fill(outside, 0.0).unsqueeze(-1).to(vals.dtype)
        out = (vals * w).sum(dim=(1, 2))
        if tp > 1:
            out = reduce_from_tensor_model_parallel_region(out, group=self.tp_group)
        return out.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_context is not None:
            raise NotImplementedError("Product-key memory does not support Megatron inference.")
        ctx = None
        if self.residual_scheme is not None:
            hidden_states, ctx = self.residual_scheme.read(hidden_states, self._depth_state)
        h = gather_replicated_input(hidden_states, self.config, self.tp_group)  # [s, b, h]
        s, b, _ = h.shape
        x = self.norm(h).reshape(s * b, -1)
        q = self.query_proj(x).view(s * b, self.heads, 2, self.k_dim // 2)
        q = self.query_norm(q).flatten(-2)  # [T, H, k_dim], halves normalized separately
        scores, rows = product_key_topk(q, self.keys1, self.keys2, self.topk)
        weights = torch.softmax(scores, dim=-1)
        read = self._read(rows, weights, x.dtype)  # [T, value_dim]
        if self.gate_proj is not None:
            read = read * F.silu(self.gate_proj(x))
        y = self.out_proj(read).view(s, b, -1)
        y = scatter_replicated_output(y, self.config, self.tp_group)
        if self.residual_scheme is not None:
            return self.residual_scheme.write(y, ctx, self._depth_state, self.layer_number)
        return hidden_states + y

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The values table is sharded on rows; everything else is replicated."""
        from megatron.core.dist_checkpointing.mapping import ShardedTensor
        from megatron.core.utils import get_pg_rank

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state = self.state_dict(prefix="", keep_vars=True)
        values = state.pop("values")
        out = make_sharded_tensors_for_checkpoint(
            state,
            prefix,
            {},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
        # Disjoint row ranges with the exact global shape: one writer per row, any layout.
        prepend = len(sharded_offsets)
        global_shape = [1] * prepend + [self.total_rows, self.value_dim]
        global_offset = [0] * prepend + [self.row_start, 0]
        for axis, rank_offset, fragments in sharded_offsets:
            global_shape[axis], global_offset[axis] = fragments, rank_offset
        # "dp": one owner per row; "tp": TP-sharded rows, replicated over DP.
        replica_id = (0, 0, 0) if self.sharded else (0, 0, get_pg_rank(metadata["dp_cp_group"]))
        key = f"{prefix}values"
        out[key] = ShardedTensor(
            key=key,
            data=values,
            dtype=values.dtype,
            local_shape=tuple(values.shape),
            global_shape=tuple(global_shape),
            global_offset=tuple(global_offset),
            axis_fragmentations=None,
            replica_id=replica_id,
            prepend_axis_num=prepend,
        )
        return out
