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
also works with sequence parallelism. Under context parallelism Engram gathers the
whole sequence (see ``cp_group``); packed sequences with CP are not supported. Packed
batches (document masking: THD, ``[t, 1, h]``) restart the n-grams and the causal conv at
every document: a token's history never reaches into the previous document, exactly as
at the sequence start.

The table parameters carry ``is_engram_table`` so the optimizer can give them their
own learning-rate multiplier and no weight decay (``engram_lr_mult``).
"""

import math
from typing import Optional

import numpy as np
import os

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
# Fixed seeds of the sign hash (engram_sign_hash): its multipliers and its +-1 table.
_SIGN_HASH_SEED = 0x516E_0BAD
_SIGN_TABLE_SEED = 0x516E_7AB1
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


def engram_hash_multipliers(
    config: TransformerConfig, layer_number: Optional[int] = None, seed: int = _HASH_SEED
) -> torch.Tensor:
    """[orders * heads, max_order] odd int64 multipliers (< 2**31), deterministic; per
    layer under ``engram_hash_per_layer``. ``seed``: the row hash's, or the sign hash's."""
    if config.engram_hash_per_layer:
        seed += 10007 * int(layer_number or 0)
    rng = np.random.default_rng(seed)
    n_lookups = len(config.engram_ngram_orders) * config.engram_num_heads
    mult = rng.integers(1 << 20, 1 << 31, size=(n_lookups, max(config.engram_ngram_orders)))
    return torch.from_numpy(mult | 1).to(torch.int64)


def engram_sign_table(config: TransformerConfig) -> torch.Tensor:
    """The fixed +-1 table of ``engram_sign_hash``: [engram_sign_table_size, embed_dim]."""
    rng = np.random.default_rng(_SIGN_TABLE_SEED)
    bits = rng.integers(0, 2, size=(config.engram_sign_table_size, config.engram_embed_dim))
    return torch.from_numpy(2.0 * bits - 1.0).to(torch.float32)


def document_positions(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Position of every token of a packed sequence within its document: ``[total]``.

    ``cu_seqlens`` are the document boundaries (``[docs + 1]``, starting at 0); tokens past
    the last boundary (padding) continue the last document. On the device, no host sync.
    """
    cu = cu_seqlens.to(torch.int64)
    idx = torch.arange(total, device=cu.device)
    doc = torch.searchsorted(cu[1:], idx, right=True).clamp_(max=cu.numel() - 2)
    return idx - cu[doc]


def engram_hash(
    token_ids: torch.Tensor,
    orders: list[int],
    num_heads: int,
    multipliers: torch.Tensor,
    sizes: list[int],
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Row index per (order, head) of every suffix n-gram: [b, s] -> [b, s, orders*heads].

    ``h = XOR_i (x_{t-i} + 1) * a_i`` over the last n tokens, then ``h mod M``. Positions
    before the sequence start -- or, with ``positions`` (``[b, s]``, the position within the
    document, see :func:`document_positions`), before the document start -- contribute the
    filler 0. Products stay below 2**52, so the int64 arithmetic never overflows.
    """
    b, s = token_ids.shape
    shifted = token_ids.to(torch.int64) + 1
    # history[i] = x_{t-i} + 1, zero before the start
    history = [shifted] + [
        F.pad(shifted, (i, 0), value=_PAD_ID)[:, :s] for i in range(1, max(orders))
    ]
    if positions is not None:
        history = [hist.masked_fill(positions < i, _PAD_ID) for i, hist in enumerate(history)]
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
        # Context parallelism: the n-gram hash and the causal conv need the history across
        # the (load-balanced) CP chunks, so Engram runs on the gathered sequence and keeps
        # this rank's chunks (fla_cp.py; its compute is not split over CP).
        self.cp_group = pg_collection.cp if pg_collection.cp.size() > 1 else None
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
        self.register_buffer(
            "hash_mult", engram_hash_multipliers(config, layer_number), persistent=False
        )
        if config.engram_sign_hash:
            self.register_buffer(
                "sign_mult",
                engram_hash_multipliers(config, layer_number, seed=_SIGN_HASH_SEED),
                persistent=False,
            )
            self.register_buffer("sign_table", engram_sign_table(config), persistent=False)
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
            if os.environ.get("ENGRAM_SPARSE_GRAD", "0") == "1":
                # only the touched rows' gradients (4 B per table parameter less)
                from megatron.core.ssm.engram_sharded import SparseRowGrad

                self.table_grad = SparseRowGrad(self.rows_per_rank, self.embed_dim, device)
            else:
                self.table_grad = torch.zeros_like(self.table, dtype=torch.float32)
            self.table.engram_sharded_table = True
            self.table.engram_grad = self.table_grad
            # the row layout, for the fsdp_dtensor checkpoint of a table hidden from Megatron-FSDP
            self.table._engram_rows = (self.total_rows, self.shard_group)
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
        self.conv_norm = (
            _RMSNorm(h, config.layernorm_epsilon).to(device) if config.engram_conv_prenorm else None
        )
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
            *([self.conv_norm] if self.conv_norm is not None else []),
        ):
            replicate_across_tp(module, self.tp_group)
        self._input_ids = None
        self._step_lookup = None  # see prefetch_step_lookup
        self._step_groups = None  # ENGRAM_STEP_GROUP: see stash_step_groups
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

    def step_lookup_supported(self) -> bool:
        """The step-level lookup (engram_sharded.py): sharded tables, TP 1, no CP."""
        return self.sharded and self.tp_group.size() == 1 and self.cp_group is None

    def _hashed_rows(self, ids, cu):
        ids = ids.reshape(1, -1)
        positions = document_positions(cu.to(ids.device).reshape(-1), ids.shape[1]).unsqueeze(0)
        mapped = ids if self.token_map is None else self.token_map[ids]
        rows = engram_hash(
            mapped, self.orders, self.num_heads, self.hash_mult, self.sizes, positions=positions
        )
        return (rows + self.row_offsets).reshape(-1), (ids.data_ptr(), ids.numel())

    def stash_step_groups(self, tokens_list, cu_seqlens_list, group_size: int) -> None:
        """ENGRAM_STEP_GROUP=G: the step lookup in groups of G micro-batches (the whole-step
        lookup holds every row + fp32 gradient of the step: ~6 GiB at 32 x 32k tokens). A group's
        rows are exchanged at the first forward that needs them; its row gradients go to their
        owners after the last of its micro-batches' backwards. The DP ranks of a stage run the
        same schedule, so the exchanges match."""
        groups = []
        for start in range(0, len(tokens_list), group_size):
            groups.append(
                {
                    "batches": list(
                        zip(
                            tokens_list[start : start + group_size],
                            cu_seqlens_list[start : start + group_size],
                        )
                    ),
                    "lookup": None,
                    "remaining": None,
                }
            )
        key_to_group = {}
        for g in groups:
            for ids, _ in g["batches"]:
                key_to_group[(ids.reshape(1, -1).data_ptr(), ids.numel())] = g
        self._step_groups = key_to_group
        if os.environ.get("ENGRAM_ASYNC_GROUPS", "0") == "1":
            # every group's rows now, all exchange plans in one host sync, group 0 in flight
            from megatron.core.ssm.engram_sharded import plan_group_exchanges

            for g in groups:
                self._group_rows(g)
            plans = plan_group_exchanges(self, [g["uniq"] for g in groups])
            for i, (g, plan) in enumerate(zip(groups, plans)):
                g["plan"], g["next"] = plan, groups[i + 1] if i + 1 < len(groups) else None
            self._start_group_exchange(groups[0])
        from megatron.core.ssm.engram_sharded import PENDING_STEP_LOOKUPS

        PENDING_STEP_LOOKUPS.append(self)

    def _group_rows(self, group):
        """A group's sorted unique rows and each micro-batch's inverse index."""
        rows_all, keys, sizes = [], [], []
        for ids, cu in group["batches"]:
            rows, key = self._hashed_rows(ids, cu)
            rows_all.append(rows)
            sizes.append(rows.numel())
            keys.append(key)
        uniq, inverse = torch.unique(torch.cat(rows_all), return_inverse=True)
        group["uniq"], group["inverse"] = uniq, dict(zip(keys, inverse.split(sizes)))

    def _start_group_exchange(self, group):
        from megatron.core.ssm.engram_sharded import GroupExchange

        if group is not None and group.get("exchange") is None and group["batches"]:
            group["exchange"] = GroupExchange(
                self, group["uniq"], group["plan"], self.config.params_dtype
            )

    def _group_lookup(self, group):
        """Exchange a stashed group's rows (once)."""
        from megatron.core.ssm.engram_sharded import exchange_rows

        if "plan" in group:  # ENGRAM_ASYNC_GROUPS: in flight since the previous group
            self._start_group_exchange(group)
            exchange = group["exchange"]
            values = exchange.wait()
            group["lookup"] = {
                "values": values,
                "grad": torch.zeros(values.shape[0], self.embed_dim, dtype=torch.float32,
                                    device=values.device),
                "exchange": exchange,
                "inverse": group.pop("inverse"),
            }
            group["remaining"] = len(group["batches"])
            self._start_group_exchange(group["next"])  # one group ahead
            return
        rows_all, keys, sizes = [], [], []
        for ids, cu in group["batches"]:
            rows, key = self._hashed_rows(ids, cu)
            rows_all.append(rows)
            sizes.append(rows.numel())
            keys.append(key)
        uniq, inverse = torch.unique(torch.cat(rows_all), return_inverse=True)
        values, state = exchange_rows(self, uniq, self.config.params_dtype)
        grad = torch.zeros(uniq.numel(), self.embed_dim, dtype=torch.float32, device=values.device)
        group["lookup"] = {
            "values": values,
            "grad": grad,
            "state": state,
            "inverse": dict(zip(keys, inverse.split(sizes))),
        }
        group["remaining"] = len(group["batches"])

    def _group_backward_done(self, group):
        group["remaining"] -= 1
        if group["remaining"] == 0:
            self._flush_group(group)

    def _flush_group(self, group):
        from megatron.core.ssm.engram_sharded import exchange_grads

        lookup, group["lookup"] = group["lookup"], None
        if lookup is not None and "exchange" in lookup:
            lookup["exchange"].flush_grads(lookup["grad"], self.config.params_dtype)
            group["exchange"] = None
        elif lookup is not None:
            exchange_grads(self, lookup["state"], lookup["grad"], self.config.params_dtype)
        group["batches"] = []  # done: nothing of it is looked up again this step

    def prefetch_step_lookup(self, tokens_list, cu_seqlens_list) -> None:
        """Fetch every row this step's micro-batches look up (packed ``[1, T]`` token tensors and
        their cu_seqlens), deduplicated, in one exchange; each micro-batch's forward then gathers
        locally (keyed by its token tensor) and the row gradients go to their owners at
        flush_step_lookup (EngramTableOptimizer.prepare_grads)."""
        from megatron.core.ssm.engram_sharded import PENDING_STEP_LOOKUPS, exchange_rows

        rows_all, keys, sizes = [], [], []
        for ids, cu in zip(tokens_list, cu_seqlens_list):
            ids = ids.reshape(1, -1)
            positions = document_positions(cu.to(ids.device).reshape(-1), ids.shape[1]).unsqueeze(0)
            mapped = ids if self.token_map is None else self.token_map[ids]
            rows = engram_hash(
                mapped, self.orders, self.num_heads, self.hash_mult, self.sizes, positions=positions
            )
            rows = (rows + self.row_offsets).reshape(-1)
            rows_all.append(rows)
            sizes.append(rows.numel())
            keys.append((ids.data_ptr(), ids.numel()))
        uniq, inverse = torch.unique(torch.cat(rows_all), return_inverse=True)
        values, state = exchange_rows(self, uniq, self.config.params_dtype)
        grad = torch.zeros(uniq.numel(), self.embed_dim, dtype=torch.float32, device=values.device)
        self._step_lookup = {
            "values": values,
            "grad": grad,
            "state": state,
            "inverse": dict(zip(keys, inverse.split(sizes))),
        }
        PENDING_STEP_LOOKUPS.append(self)

    def flush_step_lookup(self) -> None:
        """The step's row gradients -> their owners' table_grad (one exchange)."""
        from megatron.core.ssm.engram_sharded import exchange_grads

        step, self._step_lookup = self._step_lookup, None
        if step is not None:
            exchange_grads(self, step["state"], step["grad"], self.config.params_dtype)
        groups, self._step_groups = self._step_groups, None
        if groups:  # a group whose backwards did not all run (none expected): flush it now
            for group in {id(g): g for g in groups.values()}.values():
                if group["lookup"] is not None:
                    self._flush_group(group)
                elif group.get("exchange") is not None:  # issued, never looked up
                    group["exchange"].wait()
                    group["exchange"] = None

    def set_input_ids(self, input_ids: Optional[torch.Tensor]) -> None:
        """Token ids [b, s] of the current micro-batch (set by HybridStack)."""
        self._input_ids = input_ids

    def _memory(
        self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """e_t for every position: [b, s] -> [s, b, lookups * embed_dim], full on every rank.
        ``positions`` ([b, s], within the document) restart the n-grams at documents."""
        groups = self._step_groups
        if groups is not None:
            group = groups.get((input_ids.data_ptr(), input_ids.numel()))
            if group is not None and group["batches"]:
                if group["lookup"] is None:
                    self._group_lookup(group)
                from megatron.core.ssm.engram_sharded import StepRowGather

                lookup = group["lookup"]
                grad_on = torch.is_grad_enabled()
                anchor = torch.zeros((), device=input_ids.device, requires_grad=grad_on)
                emb = StepRowGather.apply(
                    anchor,
                    lookup["values"],
                    lookup["grad"],
                    lookup["inverse"][(input_ids.data_ptr(), input_ids.numel())],
                    (lambda g=group: self._group_backward_done(g)) if grad_on else None,
                )
                b, s = input_ids.shape
                return emb.reshape(b, s, -1).transpose(0, 1)
        step = self._step_lookup
        if step is not None:
            inverse = step["inverse"].get((input_ids.data_ptr(), input_ids.numel()))
            if inverse is not None:
                from megatron.core.ssm.engram_sharded import StepRowGather

                anchor = torch.zeros(
                    (), device=input_ids.device, requires_grad=torch.is_grad_enabled()
                )
                emb = StepRowGather.apply(anchor, step["values"], step["grad"], inverse)
                b, s = input_ids.shape
                return emb.reshape(b, s, -1).transpose(0, 1)
        ids = input_ids if self.token_map is None else self.token_map[input_ids]
        rows = engram_hash(
            ids, self.orders, self.num_heads, self.hash_mult, self.sizes, positions=positions
        )
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

    def _signs(
        self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``engram_sign_hash``: the +-1 pattern of every (n-gram, lookup), [s, b, L * dim],
        from an independent hash of the same (canonical) n-grams into the fixed sign table.
        Applied to e_t in every lookup path (direct, step, grouped); constant, so autograd
        carries the sign into the row gradients."""
        ids = input_ids if self.token_map is None else self.token_map[input_ids]
        n_lookups = len(self.orders) * self.num_heads
        rows = engram_hash(
            ids, self.orders, self.num_heads, self.sign_mult,
            [self.config.engram_sign_table_size] * n_lookups, positions=positions,
        )  # [b, s, L]
        signs = F.embedding(rows, self.sign_table)  # [b, s, L, dim]
        b, s = input_ids.shape
        return signs.reshape(b, s, -1).transpose(0, 1)

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
        if self._input_ids is None:
            raise RuntimeError(
                "EngramLayer needs the token ids; HybridStack passes them on the pipeline "
                "stages that hold `N` layers."
            )
        positions = None
        if packed_seq_params is not None:
            positions = self._document_positions(packed_seq_params)
        ctx = None
        if self.residual_scheme is not None:
            hidden_states, ctx = self.residual_scheme.read(hidden_states, self._depth_state)
        h = gather_replicated_input(hidden_states, self.config, self.tp_group)  # [s, b, h]
        input_ids = self._input_ids
        if self.cp_group is not None:
            from megatron.core.ssm import fla_cp

            h = fla_cp.full_sequence(h, self.cp_group)
            input_ids = fla_cp.full_sequence(
                input_ids.transpose(0, 1), self.cp_group, differentiable=False
            ).transpose(0, 1)
        e = self._memory(
            input_ids, None if positions is None else positions.unsqueeze(0)
        ).to(h.dtype)
        if self.config.engram_sign_hash:
            e = e * self._signs(
                input_ids, None if positions is None else positions.unsqueeze(0)
            ).to(e.dtype)
        k = self.key_proj(e)
        v = self.value_proj(e)
        gate = (self.hidden_norm(h).float() * self.key_norm(k).float()).sum(-1, keepdim=True)
        gate = gate / math.sqrt(h.shape[-1])
        if self.config.engram_gate_signed_sqrt:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        alpha = torch.sigmoid(gate).to(h.dtype)
        from megatron.core.transformer import arch_metrics

        if arch_metrics.active():
            n = arch_metrics.num_layers_of(self.config)
            arch_metrics.record_mean("engram/gate", self.layer_number, n, alpha)
            arch_metrics.record_mean("engram/gate_closed", self.layer_number, n, alpha < 0.05)
            arch_metrics.record_mean("engram/gate_open", self.layer_number, n, alpha > 0.95)
        v = alpha * v
        conv_in = v if self.conv_norm is None else self.conv_norm(v)
        if positions is None:
            pad = (self.kernel_size - 1) * self.dilation
            conv = self.conv1d(F.pad(conv_in.permute(1, 2, 0), (pad, 0))).permute(2, 0, 1)
        else:
            conv = self._document_conv(conv_in, positions)
        y = F.silu(conv) + v
        if self.cp_group is not None:
            y = fla_cp.local_sequence(y, self.cp_group)
        y = scatter_replicated_output(y, self.config, self.tp_group)
        if self.residual_scheme is not None:
            return self.residual_scheme.write(y, ctx, self._depth_state, self.layer_number)
        return hidden_states + y

    def _document_positions(self, packed_seq_params) -> torch.Tensor:
        """Position within its document of every token of a packed ``[1, t]`` batch."""
        if self.cp_group is not None:
            raise NotImplementedError("Engram: packed sequences with context parallelism")
        input_ids = self._input_ids
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"Engram: packed sequences need batch 1 ([1, t] ids), got {tuple(input_ids.shape)}"
            )
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        return document_positions(cu, input_ids.shape[1])

    def _document_conv(self, v: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The depthwise causal conv of ``forward`` ([s, b, h], b = 1), tap by tap, without
        the taps that reach before the token's document start."""
        weight = self.conv1d.weight[:, 0, :]  # [h, kernel]
        out = None
        for k in range(self.kernel_size):
            lag = (self.kernel_size - 1 - k) * self.dilation
            if lag == 0:
                tap = v
            else:
                tap = F.pad(v, (0, 0, 0, 0, lag, 0))[: v.shape[0]]
                tap = tap.masked_fill((positions < lag).view(-1, 1, 1), 0.0)
            term = tap * weight[:, k]
            out = term if out is None else out + term
        return out

    def _table_hidden(self) -> bool:
        """The dp-sharded table, unregistered from Megatron-FSDP (hide_sharded_tables)."""
        return getattr(self, "sharded", False) and "table" not in self._parameters and (
            getattr(self, "table", None) is not None
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._table_hidden():  # fsdp_dtensor: the table as a row-sharded DTensor
            from megatron.core.ssm.engram_sharded import row_sharded_dtensor

            destination[prefix + "table"] = row_sharded_dtensor(
                self.table.data, self.total_rows, self.shard_group
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        key = prefix + "table"
        if self._table_hidden() and key in state_dict:
            value = state_dict.pop(key)
            local = value.to_local() if hasattr(value, "to_local") else value
            if local.data_ptr() != self.table.data_ptr():  # DCP loads in place otherwise
                self.table.data.copy_(local)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)

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


def prefetch_engram_step_lookups(model_chunks, packed_doc_attention) -> None:
    """ENGRAM_STEP_LOOKUP=1 (training, document masking): one row exchange per step and Engram
    layer, from the micro-batches packed_doc_attention prefetched for every model chunk."""
    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    for vp_stage, chunk in enumerate(chunks):
        layers = [
            m for m in chunk.modules() if isinstance(m, EngramLayer) and m.step_lookup_supported()
        ]
        if not layers:
            continue
        key = packed_doc_attention._key(vp_stage if len(chunks) > 1 else None)
        batches = packed_doc_attention._BATCH_STASH.get(key)
        metadata = packed_doc_attention._METADATA_STASH.get(key)
        if not batches or batches[0] is None or not metadata:
            continue
        tokens = [b["tokens"] for b in batches]
        cus = [m.cu_seqlens for m in metadata]
        group_size = int(os.environ.get("ENGRAM_STEP_GROUP", "0"))
        for layer in layers:
            if 0 < group_size < len(tokens):
                layer.stash_step_groups(tokens, cus, group_size)
            else:
                layer.prefetch_step_lookup(tokens, cus)
