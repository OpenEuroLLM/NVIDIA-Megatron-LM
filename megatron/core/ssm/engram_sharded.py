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
    def forward(ctx, anchor, values, grad_buffer, inverse):
        ctx.grad_buffer = grad_buffer
        ctx.save_for_backward(inverse)
        return values.index_select(0, inverse)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        ctx.grad_buffer.index_add_(0, inverse, grad.float())
        return grad.new_zeros(()), None, None, None


# Layers holding a step lookup whose gradients still have to go to their owners, in the order
# they prefetched (identical on every rank of a shard group, so the exchanges match).
PENDING_STEP_LOOKUPS = []


def flush_step_lookups() -> None:
    """Send every pending step lookup's row gradients to their owners (before the optimizer step:
    EngramTableOptimizer.prepare_grads)."""
    for layer in PENDING_STEP_LOOKUPS:
        layer.flush_step_lookup()
    PENDING_STEP_LOOKUPS.clear()
