# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Engram: conditional memory via hashed n-gram lookup (HybridModel symbol `N`).

Cheng et al., "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for
Large Language Models" (arXiv:2601.07372). Per token t an Engram layer

1. maps the token ids to canonical ids (optional ``engram_token_map``: NFKC +
   lowercasing collapses surface variants),
2. hashes every suffix n-gram ``(x'_{t-n+1}, ..., x'_t)``, n in ``engram_ngram_orders``,
   with ``engram_num_heads`` multiplicative-XOR hash heads into tables of prime size,
3. concatenates the rows into ``e_t`` and fuses it into the residual stream:

    k = W_K e,  v = W_V e,  alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(d))
    v~ = alpha * v,  Y = SiLU(causal_conv(v~)) + v~,  h <- h + Y

   with a depthwise causal conv (kernel ``engram_conv_kernel_size``, dilation = the
   largest n-gram order) initialised to zero.

The lookup addresses depend on the token ids alone, so the tables can live off the
GPU at inference. Here they are trained on the GPU: under tensor parallelism the
rows of the (concatenated) table are sharded like a vocab-parallel embedding (masked
local lookup, then all-reduce); the small projections, norms and conv are replicated
across TP ranks (tp_replicated.py) and the layer runs on the whole sequence, so it
also works with sequence parallelism. Context parallelism and packed sequences are
not supported: the n-grams and the conv would need tokens across chunk / document
boundaries.

The table parameters carry ``is_engram_table`` so the optimizer can give them their
own learning-rate multiplier and no weight decay (``engram_lr_mult``).
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.tp_replicated import (
    gather_replicated_input,
    replicate_across_tp,
    scatter_replicated_output,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)

# Rows of the concatenated table are padded to a multiple of this, independent of the
# TP size, so a checkpoint's global table shape does not change with the parallelism.
_ROW_ALIGN = 4096
# Fixed seed of the hash multipliers: part of the model definition, not of training.
_HASH_SEED = 0x5EED_E46A
# A token-id offset so that the "before the sequence start" filler (0) is distinct.
_PAD_ID = 0


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def engram_table_sizes(config: TransformerConfig) -> list[int]:
    """Distinct primes >= engram_table_size, one per (order, head), in lookup order."""
    sizes, candidate = [], config.engram_table_size
    for _ in range(len(config.engram_ngram_orders) * config.engram_num_heads):
        while not _is_prime(candidate):
            candidate += 1
        sizes.append(candidate)
        candidate += 1
    return sizes


def engram_hash_multipliers(config: TransformerConfig) -> torch.Tensor:
    """[orders * heads, max_order] odd int64 multipliers (< 2**31), deterministic."""
    rng = np.random.default_rng(_HASH_SEED)
    n_lookups = len(config.engram_ngram_orders) * config.engram_num_heads
    mult = rng.integers(1 << 20, 1 << 31, size=(n_lookups, max(config.engram_ngram_orders)))
    return torch.from_numpy(mult | 1).to(torch.int64)


def engram_hash(
    token_ids: torch.Tensor,
    orders: list[int],
    num_heads: int,
    multipliers: torch.Tensor,
    sizes: list[int],
) -> torch.Tensor:
    """Row index per (order, head) of every suffix n-gram: [b, s] -> [b, s, orders*heads].

    ``h = XOR_i (x_{t-i} + 1) * a_i`` over the last n tokens, then ``h mod M``. Positions
    before the sequence start contribute the filler 0. Products stay below 2**52, so
    the int64 arithmetic never overflows.
    """
    b, s = token_ids.shape
    shifted = token_ids.to(torch.int64) + 1
    # history[i] = x_{t-i} + 1, zero before the start
    history = [shifted] + [
        F.pad(shifted, (i, 0), value=_PAD_ID)[:, :s] for i in range(1, max(orders))
    ]
    out = []
    lookup = 0
    for n in orders:
        for _ in range(num_heads):
            h = torch.zeros_like(shifted)
            for i in range(n):
                h = h ^ (history[i] * multipliers[lookup, i])
            out.append(torch.remainder(h, sizes[lookup]))
            lookup += 1
    return torch.stack(out, dim=-1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class EngramLayer(MegatronModule):
    """One Engram memory layer: ``h + Engram(h, token_ids)``.

    The stack hands it the token ids through ``set_input_ids`` before each forward
    (HybridStack does this for every `N` position).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ):
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for EngramLayer"
        if pg_collection.cp.size() != 1:
            raise NotImplementedError("Engram does not support context parallelism.")
        self.config = config
        self.layer_number = layer_number
        self.tp_group = pg_collection.tp
        tp_size, tp_rank = self.tp_group.size(), self.tp_group.rank()

        self.orders = list(config.engram_ngram_orders)
        self.num_heads = config.engram_num_heads
        self.embed_dim = config.engram_embed_dim
        self.sizes = engram_table_sizes(config)
        offsets = np.cumsum([0] + self.sizes[:-1])
        self.register_buffer(
            "row_offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False
        )
        self.register_buffer("hash_mult", engram_hash_multipliers(config), persistent=False)
        token_map = None
        if config.engram_token_map is not None:
            token_map = torch.from_numpy(np.load(config.engram_token_map).astype(np.int64))
        self.register_buffer("token_map", token_map, persistent=False)

        rows = sum(self.sizes)
        device = torch.cuda.current_device()
        self.sharded = config.engram_table_parallel == "dp"
        if self.sharded:
            # Rows sharded over every rank of the stage, looked up with all-to-all and
            # updated from sparse row gradients (engram_sharded.py).
            if not config.engram_table_fp32:
                raise ValueError("engram_table_parallel='dp' keeps the tables in fp32")
            group = getattr(pg_collection, "tp_dp_cp", None)
            if group is None:  # migration fallback: collections built before tp_dp_cp
                from megatron.core import parallel_state

                group = parallel_state.get_tensor_and_data_parallel_group(
                    with_context_parallel=True
                )
            self.shard_group = group
            self.shard_size, self.shard_rank = group.size(), group.rank()
            self.total_rows = rows  # exact: the global shape never depends on the layout
            from megatron.core.ssm.engram_sharded import balanced_row_range

            self.row_start, row_end = balanced_row_range(rows, self.shard_rank, self.shard_size)
            self.rows_per_rank = row_end - self.row_start
            # Not a DDP parameter: its gradient arrives sparse, on its owner only.
            self.table = nn.Parameter(
                torch.empty(self.rows_per_rank, self.embed_dim, device=device), requires_grad=False
            )
            # A plain attribute, not a buffer: Float16Module's cast must not touch it.
            self.table_grad = torch.zeros_like(self.table, dtype=torch.float32)
            self.table.engram_sharded_table = True
            self.table.engram_grad = self.table_grad
        else:
            # Concatenated table, rows sharded across TP like a vocab-parallel embedding.
            self.total_rows = int(math.ceil(rows / _ROW_ALIGN) * _ROW_ALIGN)
            assert self.total_rows % tp_size == 0, (self.total_rows, tp_size)
            self.rows_per_rank = self.total_rows // tp_size
            self.row_start = tp_rank * self.rows_per_rank
            table_dtype = torch.float32 if config.engram_table_fp32 else config.params_dtype
            self.table = nn.Parameter(
                torch.empty(self.rows_per_rank, self.embed_dim, device=device, dtype=table_dtype)
            )
            setattr(self.table, "tensor_model_parallel", True)
            setattr(self.table, "partition_dim", 0)
            setattr(self.table, "partition_stride", 1)
        setattr(self.table, "is_engram_table", True)

        mem_dim = self.embed_dim * len(self.sizes)
        h = config.hidden_size
        self.key_proj = nn.Linear(mem_dim, h, bias=False, device=device, dtype=config.params_dtype)
        self.value_proj = nn.Linear(
            mem_dim, h, bias=False, device=device, dtype=config.params_dtype
        )
        self.hidden_norm = _RMSNorm(h, config.layernorm_epsilon).to(device)
        self.key_norm = _RMSNorm(h, config.layernorm_epsilon).to(device)
        self.kernel_size = config.engram_conv_kernel_size
        self.dilation = max(self.orders)
        self.conv1d = nn.Conv1d(
            h,
            h,
            kernel_size=self.kernel_size,
            groups=h,
            bias=False,
            dilation=self.dilation,
            device=device,
            dtype=config.params_dtype,
        )
        self.reset_parameters()
        # Everything but the table is one full copy per TP rank.
        for module in (
            self.key_proj,
            self.value_proj,
            self.hidden_norm,
            self.key_norm,
            self.conv1d,
        ):
            replicate_across_tp(module, self.tp_group)
        self._input_ids = None
        # mHC / Block AttnRes / nGPT replace `h + Engram(h)` (residual_scheme_adapter.py).
        from megatron.core.transformer.residual_scheme_adapter import (
            ResidualSchemeAdapter,
            residual_scheme_active,
        )

        self.residual_scheme = (
            ResidualSchemeAdapter(config, layer_number) if residual_scheme_active(config) else None
        )
        self._depth_state = None

    def _apply(self, fn, recurse=True):
        """Keep an fp32 table (``engram_table_fp32``) through the model's bf16/fp16 cast.

        Float16Module converts the whole model with ``module.bfloat16()``; a dtype
        conversion is detected by probing ``fn`` and then applied to everything but
        the table, which only follows device moves.
        """
        if not self.config.engram_table_fp32 or "table" not in self._parameters:
            return super()._apply(fn, recurse)
        probe = fn(torch.zeros(1, dtype=torch.float32, device=self.table.device))
        if probe.dtype == torch.float32:
            return super()._apply(fn, recurse)
        table = self._parameters.pop("table")
        try:
            super()._apply(fn, recurse)
        finally:
            if table.device != probe.device:
                table.data = table.data.to(probe.device)
            self._parameters["table"] = table
        return self

    def reset_parameters(self):
        if not self.config.perform_initialization:
            return
        if self.sharded:
            # Every shard draws its own rows (the model-parallel RNG is shared across DP).
            gen = torch.Generator(device=self.table.device)
            gen.manual_seed(1_000_003 * int(self.layer_number or 0) + 7919 * self.shard_rank + 17)
            with torch.no_grad():
                self.table.normal_(mean=0.0, std=self.config.init_method_std, generator=gen)
        else:
            with get_cuda_rng_tracker().fork():  # different rows on every TP rank
                nn.init.normal_(self.table, mean=0.0, std=self.config.init_method_std)
        self.config.init_method(self.key_proj.weight)
        self.config.output_layer_init_method(self.value_proj.weight)
        nn.init.zeros_(self.conv1d.weight)  # starts as Y = v~

    def set_input_ids(self, input_ids: Optional[torch.Tensor]) -> None:
        """Token ids [b, s] of the current micro-batch (set by HybridStack)."""
        self._input_ids = input_ids

    def _memory(self, input_ids: torch.Tensor) -> torch.Tensor:
        """e_t for every position: [b, s] -> [s, b, lookups * embed_dim], full on every rank."""
        ids = input_ids if self.token_map is None else self.token_map[input_ids]
        rows = engram_hash(ids, self.orders, self.num_heads, self.hash_mult, self.sizes)
        rows = rows + self.row_offsets  # [b, s, L] global row ids
        if self.sharded:
            return self._sharded_memory(rows)
        local = rows - self.row_start
        outside = (local < 0) | (local >= self.rows_per_rank)
        emb = F.embedding(local.masked_fill(outside, 0), self.table)
        emb = emb.masked_fill(outside.unsqueeze(-1), 0.0)
        if self.tp_group.size() > 1:
            emb = reduce_from_tensor_model_parallel_region(emb, group=self.tp_group)
        b, s = input_ids.shape
        return emb.reshape(b, s, -1).transpose(0, 1)

    def _sharded_memory(self, rows: torch.Tensor) -> torch.Tensor:
        """``engram_table_parallel="dp"``: [b, s, L] rows -> [s, b, L * embed_dim].

        TP ranks hold the same tokens, so each looks up only its slice of the sequence
        and the slices are gathered; the replicated Engram compute makes the gradient of
        the gathered rows identical across TP, so the gather's backward just keeps the
        local slice and every row gradient reaches its owner once.
        """
        from megatron.core.ssm.engram_sharded import sharded_lookup
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        b, s, lookups = rows.shape
        tp = self.tp_group.size()
        if s % tp:
            raise ValueError(f"engram_table_parallel='dp' needs seq ({s}) divisible by TP ({tp})")
        lo = self.tp_group.rank() * (s // tp)
        local = rows[:, lo : lo + s // tp].reshape(-1)
        emb = sharded_lookup(self, local, self.config.params_dtype)
        emb = emb.reshape(b, s // tp, lookups * self.embed_dim).transpose(0, 1).contiguous()
        if tp > 1:
            emb = gather_from_sequence_parallel_region(
                emb, tensor_parallel_output_grad=False, group=self.tp_group
            )
        return emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_context is not None:
            raise NotImplementedError("Engram does not support Megatron inference.")
        if packed_seq_params is not None:
            raise NotImplementedError("Engram does not support packed sequences.")
        if self._input_ids is None:
            raise RuntimeError(
                "EngramLayer needs the token ids; HybridStack passes them on the pipeline "
                "stages that hold `N` layers."
            )
        ctx = None
        if self.residual_scheme is not None:
            hidden_states, ctx = self.residual_scheme.read(hidden_states, self._depth_state)
        h = gather_replicated_input(hidden_states, self.config, self.tp_group)  # [s, b, h]
        e = self._memory(self._input_ids).to(h.dtype)
        k = self.key_proj(e)
        v = self.value_proj(e)
        gate = (self.hidden_norm(h).float() * self.key_norm(k).float()).sum(-1, keepdim=True)
        alpha = torch.sigmoid(gate / math.sqrt(h.shape[-1])).to(h.dtype)
        from megatron.core.transformer import arch_metrics

        if arch_metrics.active():
            n = arch_metrics.num_layers_of(self.config)
            arch_metrics.record_mean("engram/gate", self.layer_number, n, alpha)
            arch_metrics.record_mean("engram/gate_closed", self.layer_number, n, alpha < 0.05)
            arch_metrics.record_mean("engram/gate_open", self.layer_number, n, alpha > 0.95)
        v = alpha * v
        pad = (self.kernel_size - 1) * self.dilation
        conv = self.conv1d(F.pad(v.permute(1, 2, 0), (pad, 0))).permute(2, 0, 1)
        y = F.silu(conv) + v
        y = scatter_replicated_output(y, self.config, self.tp_group)
        if self.residual_scheme is not None:
            return self.residual_scheme.write(y, ctx, self._depth_state, self.layer_number)
        return hidden_states + y

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The table is sharded on rows (over TP, or every rank of the stage under
        engram_table_parallel="dp"); everything else is replicated."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state = self.state_dict(prefix="", keep_vars=True)
        if not self.sharded:
            return make_sharded_tensors_for_checkpoint(
                state,
                prefix,
                {"table": 0},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        table = state.pop("table")
        out = make_sharded_tensors_for_checkpoint(
            state,
            prefix,
            {},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
        # Disjoint row ranges with an exact global shape: one writer per row, any layout.
        prepend = len(sharded_offsets)
        global_shape = [1] * prepend + [self.total_rows, self.embed_dim]
        global_offset = [0] * prepend + [self.row_start, 0]
        for axis, rank_offset, fragments in sharded_offsets:
            global_shape[axis], global_offset[axis] = fragments, rank_offset
        key = f"{prefix}table"
        out[key] = ShardedTensor(
            key=key,
            data=table,
            dtype=table.dtype,
            local_shape=tuple(table.shape),
            global_shape=tuple(global_shape),
            global_offset=tuple(global_offset),
            axis_fragmentations=None,
            replica_id=(0, 0, 0),
            prepend_axis_num=prepend,
        )
        return out
