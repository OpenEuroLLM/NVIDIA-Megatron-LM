# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Engram tables sharded over every rank of a pipeline stage (``engram_table_parallel="dp"``).

The tables are the bulk of an Engram model's parameters but each step touches only the
rows its tokens hash to. Replicating them across data-parallel ranks (the "tp" mode, or
upstream's EP sharding with expert-DP replicas) makes DDP reduce a *dense* table
gradient every step -- table-sized traffic for a sparse update. Here every rank of the
stage's tensor x data x context group owns a disjoint, contiguous range of rows, so:

* forward: each rank sends the (deduplicated) rows its tokens need to their owners
  (all-to-all), owners gather them and send the rows back (all-to-all);
* backward: the row gradients go back to their owners (all-to-all) and are
  accumulated in fp32 into the owner's gradient buffer -- no dense reduction at all;
* the tables are not DDP parameters (``requires_grad=False``); EngramTableOptimizer
  (engram_optimizer.py) steps each rank's shard, chained into the main optimizer so
  the LR schedule, gradient-norm clipping and checkpointing see it.

Traffic per step is ``tokens x lookups x embed_dim`` in the compute dtype, independent
of the table size.
"""

from typing import Tuple

import torch
import torch.distributed as dist


def balanced_row_range(rows: int, rank: int, size: int) -> Tuple[int, int]:
    """Contiguous rows ``[start, end)`` of ``rank``: ``rows`` split as evenly as possible.

    The global table shape is ``rows`` for every layout, so checkpoints reshard freely.
    """
    base, rem = divmod(rows, size)
    start = rank * base + min(rank, rem)
    return start, start + base + int(rank < rem)


def row_owner(rows: torch.Tensor, total_rows: int, size: int) -> torch.Tensor:
    """Owner rank of each global row under :func:`balanced_row_range`."""
    base, rem = divmod(total_rows, size)
    big = (base + 1) * rem  # rows held by the first `rem` ranks (one extra each)
    return torch.where(rows < big, rows // (base + 1), rem + (rows - big) // max(base, 1))


def _exchange(group, tensor: torch.Tensor, out_splits, in_splits) -> torch.Tensor:
    out = tensor.new_empty((sum(out_splits), *tensor.shape[1:]))
    dist.all_to_all_single(
        out,
        tensor.contiguous(),
        output_split_sizes=out_splits,
        input_split_sizes=in_splits,
        group=group,
    )
    return out


class _ShardedRowLookup(torch.autograd.Function):
    """Rows of a table sharded over ``group``; gradients accumulate on the owners.

    ``anchor`` is a scalar requiring grad: the table is not an autograd input (it is not
    a DDP parameter), so the anchor gives the result a backward.
    """

    @staticmethod
    def forward(ctx, anchor, layer, rows, dtype):
        group, size = layer.shard_group, layer.shard_size
        uniq, inverse = torch.unique(rows, return_inverse=True)
        if size == 1:
            ctx.layer, ctx.uniq, ctx.inverse, ctx.single = layer, uniq, inverse, True
            ctx.uniq_count = uniq.numel()
            return layer.table.index_select(0, uniq - layer.row_start)[inverse].to(dtype)
        owners = row_owner(uniq, layer.total_rows, size)  # uniq is sorted, so are owners
        send = torch.bincount(owners, minlength=size)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        send_splits, recv_splits = send.tolist(), recv.tolist()
        requested = _exchange(group, uniq, recv_splits, send_splits)  # rows I own
        values = layer.table.index_select(0, requested - layer.row_start).to(dtype)
        got = _exchange(group, values, send_splits, recv_splits)  # rows I asked for
        ctx.layer, ctx.requested, ctx.single = layer, requested, False
        ctx.splits = (send_splits, recv_splits)
        ctx.inverse, ctx.uniq_count = inverse, uniq.numel()
        return got[inverse]

    @staticmethod
    def backward(ctx, grad):
        layer = ctx.layer
        # Sum the gradients of duplicate requests first (fp32), then send one row each.
        grad_uniq = grad.new_zeros((ctx.uniq_count, grad.shape[-1]), dtype=torch.float32)
        grad_uniq.index_add_(0, ctx.inverse, grad.float())
        if ctx.single:
            layer.table_grad.index_add_(0, ctx.uniq - layer.row_start, grad_uniq)
        else:
            send_splits, recv_splits = ctx.splits
            # Back to the owners in the compute dtype, the reverse of the forward exchange.
            owned = _exchange(layer.shard_group, grad_uniq.to(grad.dtype), recv_splits, send_splits)
            layer.table_grad.index_add_(0, ctx.requested - layer.row_start, owned.float())
        return grad.new_zeros(()), None, None, None


def sharded_lookup(layer, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[n]`` global row ids -> ``[n, embed_dim]`` rows in ``dtype`` (see module doc)."""
    anchor = torch.zeros((), device=rows.device, requires_grad=torch.is_grad_enabled())
    return _ShardedRowLookup.apply(anchor, layer, rows, dtype)


class _ShardedUniqueRows(torch.autograd.Function):
    """:class:`_ShardedRowLookup` for already-unique (sorted) rows, without the expansion
    to every request: ``[u]`` rows -> ``[u, embed_dim]``. The caller combines them (e.g.
    by ``embedding_bag`` over the inverse index), so neither the expanded rows nor their
    gradient are ever materialized."""

    @staticmethod
    def forward(ctx, anchor, layer, uniq, dtype):
        group, size = layer.shard_group, layer.shard_size
        ctx.layer = layer
        if size == 1:
            ctx.requested, ctx.single = uniq, True
            return layer.table.index_select(0, uniq - layer.row_start).to(dtype)
        owners = row_owner(uniq, layer.total_rows, size)  # sorted, so are owners
        send = torch.bincount(owners, minlength=size)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        send_splits, recv_splits = send.tolist(), recv.tolist()
        requested = _exchange(group, uniq, recv_splits, send_splits)  # rows I own
        values = layer.table.index_select(0, requested - layer.row_start).to(dtype)
        ctx.requested, ctx.single, ctx.splits = requested, False, (send_splits, recv_splits)
        return _exchange(group, values, send_splits, recv_splits)  # rows I asked for

    @staticmethod
    def backward(ctx, grad):
        layer = ctx.layer
        if ctx.single:
            owned = grad
        else:
            send_splits, recv_splits = ctx.splits
            owned = _exchange(layer.shard_group, grad.contiguous(), recv_splits, send_splits)
        layer.table_grad.index_add_(0, ctx.requested - layer.row_start, owned.float())
        return grad.new_zeros(()), None, None, None


def sharded_lookup_unique(layer, rows: torch.Tensor, dtype: torch.dtype):
    """``[n]`` global row ids -> ``(values [u, embed_dim] in dtype, inverse [n])`` with
    ``values[inverse]`` the rows of :func:`sharded_lookup`."""
    uniq, inverse = torch.unique(rows, return_inverse=True)
    anchor = torch.zeros((), device=rows.device, requires_grad=torch.is_grad_enabled())
    return _ShardedUniqueRows.apply(anchor, layer, uniq, dtype), inverse


# ---- sparse row gradients (ENGRAM_SPARSE_GRAD=1) ---------------------------------------------
class SparseRowGrad:
    """The fp32 gradient of a table shard, kept for the rows the step touched only.

    A drop-in for the dense ``[rows, dim]`` fp32 buffer (4 B per table parameter): the lookups'
    backwards call ``index_add_(0, local_rows, grads)`` on it as on the dense one; the pairs are
    appended and coalesced (sorted unique rows, summed in fp32) when they pass
    ``coalesce_bytes`` and before the norm / clip / update. A step touches at most its lookups'
    rows (<= ~20 % of a large table), so this costs a fraction of the dense buffer. The update
    stays exact dense Adam: TableAdam scatters each chunk's rows into a zero chunk, so the rows
    without a gradient still move by their momentum, as with the dense buffer.
    """

    def __init__(self, num_rows: int, dim: int, device, coalesce_bytes: int = 256 << 20):
        self.num_rows, self.dim, self.device = num_rows, dim, device
        self.coalesce_bytes = coalesce_bytes
        self.rows = None  # sorted unique local rows (int64) after coalesce()
        self.values = None  # [len(rows), dim] fp32
        self._idx, self._val, self._pending = [], [], 0

    def index_add_(self, dim: int, index: torch.Tensor, source: torch.Tensor):
        if dim != 0:
            raise ValueError("SparseRowGrad accumulates rows (dim 0) only")
        if index.numel():
            self._idx.append(index.to(torch.int64))
            self._val.append(source.float())
            self._pending += source.numel() * 4
            if self._pending >= self.coalesce_bytes:
                self.coalesce()
        return self

    def coalesce(self) -> "SparseRowGrad":
        if not self._idx:
            return self
        idx, val = self._idx, self._val
        if self.rows is not None:
            idx, val = [self.rows, *idx], [self.values, *val]
        idx, val = torch.cat(idx), torch.cat(val)
        self._idx, self._val, self._pending = [], [], 0
        rows, inverse = torch.unique(idx, return_inverse=True)
        values = torch.zeros(rows.numel(), self.dim, dtype=torch.float32, device=val.device)
        values.index_add_(0, inverse, val)
        self.rows, self.values = rows, values
        return self

    def zero_(self) -> "SparseRowGrad":
        self.rows = self.values = None
        self._idx, self._val, self._pending = [], [], 0
        return self

    def coalesced_values(self) -> torch.Tensor:
        """The touched rows' gradients ``[n, dim]`` (empty if none)."""
        self.coalesce()
        if self.values is None:
            return torch.zeros(0, self.dim, dtype=torch.float32, device=self.device)
        return self.values

    def mul_(self, scale) -> "SparseRowGrad":
        self.coalesce()
        if self.values is not None:
            self.values.mul_(scale)
        return self

    def chunk_bounds(self, starts):
        """For row chunks starting at ``starts`` (ascending, Python ints): the [a, b) slices of
        the coalesced rows falling in each chunk (one host sync)."""
        self.coalesce()
        if self.rows is None:
            return [(0, 0)] * len(starts)
        edges = torch.tensor([*starts, self.num_rows], dtype=torch.int64, device=self.rows.device)
        cut = torch.searchsorted(self.rows, edges).tolist()
        return list(zip(cut[:-1], cut[1:]))

    def to_dense(self) -> torch.Tensor:
        """The full ``[rows, dim]`` fp32 gradient (tests, debugging)."""
        self.coalesce()
        out = torch.zeros(self.num_rows, self.dim, dtype=torch.float32, device=self.device)
        if self.rows is not None:
            out.index_copy_(0, self.rows, self.values)
        return out

    def dense_chunk(self, lo: int, hi: int, a: int, b: int) -> torch.Tensor:
        """Rows [lo, hi) as a dense fp32 chunk (zeros where untouched)."""
        g = torch.zeros(hi - lo, self.dim, dtype=torch.float32, device=self.device)
        if b > a:
            g.index_copy_(0, self.rows[a:b] - lo, self.values[a:b])
        return g


# ---- step-level lookup (ENGRAM_STEP_LOOKUP=1) ------------------------------------------------
# The rows a micro-batch looks up depend on its token ids alone, and with document masking the
# whole step's micro-batches are fetched before the schedule runs (packed_doc_attention). So a
# layer can fetch every row the step needs in ONE exchange, serve each micro-batch's forward and
# recompute by a local gather, accumulate the row gradients locally (fp32) and send them to their
# owners in ONE exchange before the optimizer step -- the tables do not change within a step, so
# this is exact. It replaces two host syncs and four all-to-alls per micro-batch and layer (plus
# the recompute's) by the same per step.


def exchange_rows(layer, uniq: torch.Tensor, dtype: torch.dtype):
    """Sorted unique global rows -> (their values ``[u, embed_dim]`` in ``dtype``, the state the
    gradient exchange needs)."""
    group, size = layer.shard_group, layer.shard_size
    if size == 1:
        return layer.table.index_select(0, uniq - layer.row_start).to(dtype), (uniq, None)
    owners = row_owner(uniq, layer.total_rows, size)
    send = torch.bincount(owners, minlength=size)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    send_splits, recv_splits = send.tolist(), recv.tolist()
    requested = _exchange(group, uniq, recv_splits, send_splits)  # rows I own
    values = layer.table.index_select(0, requested - layer.row_start).to(dtype)
    return _exchange(group, values, send_splits, recv_splits), (requested, (send_splits, recv_splits))


def exchange_grads(layer, state, grad_uniq: torch.Tensor, dtype: torch.dtype) -> None:
    """``[u, embed_dim]`` fp32 row gradients (the order of :func:`exchange_rows`) -> accumulated
    into their owners' ``table_grad`` (sent in ``dtype``, as the per-micro-batch path does)."""
    requested, splits = state
    if splits is None:
        layer.table_grad.index_add_(0, requested - layer.row_start, grad_uniq)
        return
    send_splits, recv_splits = splits
    owned = _exchange(layer.shard_group, grad_uniq.to(dtype), recv_splits, send_splits)
    layer.table_grad.index_add_(0, requested - layer.row_start, owned.float())


class StepRowGather(torch.autograd.Function):
    """``values[inverse]`` for one micro-batch; the backward accumulates into the step's fp32 row
    gradient buffer (no exchange). ``anchor`` gives the result a backward (values is no leaf)."""

    @staticmethod
    def forward(ctx, anchor, values, grad_buffer, inverse, on_backward=None):
        ctx.grad_buffer, ctx.on_backward = grad_buffer, on_backward
        ctx.save_for_backward(inverse)
        return values.index_select(0, inverse)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        ctx.grad_buffer.index_add_(0, inverse, grad.float())
        if ctx.on_backward is not None:  # ENGRAM_STEP_GROUP: the group's last backward flushes
            ctx.on_backward()
        return grad.new_zeros(()), None, None, None, None


# ---- asynchronous group exchanges (ENGRAM_ASYNC_GROUPS=1) ---------------------------------------
# The group lookup's exchanges, issued ahead instead of at the group's first forward: every
# group's rows are known at the step start (the micro-batches are prefetched), so the per-owner
# counts of ALL groups are exchanged there in one all-to-all with one host sync; each group's
# row exchange is then a chain of async all-to-alls (requests, then values gathered by the
# owners on a side stream) without host syncs, issued one group ahead and waited for on the GPU
# only. The gradient flush runs the same way on the side stream; prepare_grads joins it. Every
# rank of a shard group issues the same collectives in the same program order, so they match.

_SIDE_STREAM = None
_FLUSH_PENDING = []  # side-stream work the optimizer step must wait for


def _side_stream():
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def plan_group_exchanges(layer, uniqs):
    """Sorted unique rows of each group -> per group ``(send_splits, recv_splits)`` (None for
    an unsharded table): one count all-to-all, one host sync for all groups."""
    group, size = layer.shard_group, layer.shard_size
    if size == 1:
        return [None] * len(uniqs)
    n = len(uniqs)
    counts = torch.stack([torch.bincount(row_owner(u, layer.total_rows, size), minlength=size)
                          for u in uniqs])  # [n, P]
    send = counts.t().contiguous().reshape(-1)  # destination-major: P chunks of n counts
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    both = torch.stack([counts, recv.view(size, n).t()]).tolist()  # the one host sync
    return [(both[0][g], both[1][g]) for g in range(n)]


class GroupExchange:
    """One group's rows, in flight: requests to the owners, their values back (async)."""

    def __init__(self, layer, uniq, splits, dtype):
        self.layer, self.uniq, self.splits = layer, uniq, splits
        if splits is None:
            self.values = layer.table.index_select(0, uniq - layer.row_start).to(dtype)
            self.requested, self.work = uniq, None
            return
        send_splits, recv_splits = splits
        group, side = layer.shard_group, _side_stream()
        self.requested = uniq.new_empty(sum(recv_splits))
        self.values = torch.empty(sum(send_splits), layer.embed_dim, dtype=dtype,
                                  device=uniq.device)
        w1 = dist.all_to_all_single(self.requested, uniq, output_split_sizes=recv_splits,
                                    input_split_sizes=send_splits, group=group, async_op=True)
        with torch.cuda.stream(side):
            w1.wait()  # the side stream waits for the requests, not the host
            owned = layer.table.index_select(0, self.requested - layer.row_start).to(dtype)
            self.work = dist.all_to_all_single(self.values, owned, output_split_sizes=send_splits,
                                               input_split_sizes=recv_splits, group=group,
                                               async_op=True)
        self._owned = owned  # alive until the values arrived

    def wait(self) -> torch.Tensor:
        if self.work is not None:
            self.work.wait()  # the current stream waits (GPU side)
            self.work, self._owned = None, None
        return self.values

    def flush_grads(self, grad_uniq: torch.Tensor, dtype: torch.dtype) -> None:
        """The group's row gradients -> their owners' table_grad, on the side stream."""
        layer = self.layer
        if self.splits is None:
            layer.table_grad.index_add_(0, self.requested - layer.row_start, grad_uniq)
            return
        send_splits, recv_splits = self.splits
        side, current = _side_stream(), torch.cuda.current_stream()
        with torch.cuda.stream(side):
            side.wait_stream(current)  # the gradients are complete
            sent = grad_uniq.to(dtype)
            owned = torch.empty(sum(recv_splits), layer.embed_dim, dtype=dtype,
                                device=grad_uniq.device)
            dist.all_to_all_single(owned, sent, output_split_sizes=recv_splits,
                                   input_split_sizes=send_splits, group=layer.shard_group,
                                   async_op=True).wait()
            owned32 = owned.float()
            layer.table_grad.index_add_(0, self.requested - layer.row_start, owned32)
        for t in (grad_uniq, self.requested):
            t.record_stream(side)  # used on the side stream, freed on the current one
        owned32.record_stream(current)  # SparseRowGrad keeps it and coalesces on the current
        _FLUSH_PENDING.append(side)


def join_async_flushes() -> None:
    """The current stream waits for the side-stream gradient flushes (before the norm / step)."""
    current = torch.cuda.current_stream()
    for stream in {id(s): s for s in _FLUSH_PENDING}.values():
        current.wait_stream(stream)
    _FLUSH_PENDING.clear()


# Layers holding a step lookup whose gradients still have to go to their owners, in the order
# they prefetched (identical on every rank of a shard group, so the exchanges match).
PENDING_STEP_LOOKUPS = []


def flush_step_lookups() -> None:
    """Send every pending step lookup's row gradients to their owners (before the optimizer step:
    EngramTableOptimizer.prepare_grads)."""
    for layer in PENDING_STEP_LOOKUPS:
        layer.flush_step_lookup()
    PENDING_STEP_LOOKUPS.clear()
    join_async_flushes()
