# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Context parallelism for the FLA-based token mixers.

Megatron's CP gives rank ``r`` the load-balanced chunks ``r`` and ``2 cp - 1 - r`` (of ``2 cp``)
of every sequence: ``[s / cp, b, h]``. FLA's context parallelism (fla.ops.cp, "KCP": the
delta-rule recurrences pass their state across ranks) instead splits the flattened token stream
``[1, b * s, h]`` (one cu_seqlens entry per sample) into ``cp`` contiguous pieces. The mixer
input and output are converted between the two layouts by an all-gather and a local slice each
-- ``[s, b, h]`` at the model width, cheap next to the mixer; the all-gather's backward is a
reduce-scatter, so the gradients land where the inputs came from.

``to_stream(..., full=True)`` hands every rank the whole stream instead: the fallback for
mixers without CP kernels (correct, but that mixer's compute is not split).

Packed batches (THD, document masking): Megatron shards every document into 2 cp chunks instead;
:class:`ThdStream` converts between that layout and FLA's stream, whose CP context then follows the
document boundaries (the recurrence restarts at every document, also across ranks).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn


def _gather(x: torch.Tensor, group) -> torch.Tensor:
    """``[n, ...]`` per rank -> ``[cp, n, ...]`` (autograd: reduce-scatter backward)."""
    return torch.stack(dist_fn.all_gather(x.contiguous(), group=group))


def full_sequence(x: torch.Tensor, group, differentiable: bool = True) -> torch.Tensor:
    """``[s / cp, b, ...]`` (Megatron's CP layout) -> ``[s, b, ...]`` in token order, on every
    rank. ``differentiable=False`` for tensors without gradient (e.g. token ids)."""
    cp = dist.get_world_size(group)
    if differentiable:
        parts = _gather(x, group)  # [cp, s/cp, b, h]: rank r holds chunks (r, 2cp-1-r)
    else:
        bufs = [torch.empty_like(x) for _ in range(cp)]
        dist.all_gather(bufs, x.contiguous(), group=group)
        parts = torch.stack(bufs)
    half = x.shape[0] // 2
    chunks = parts.view(cp, 2, half, *x.shape[1:])  # [rank, slot]: chunks (r, 2cp-1-r)
    # chunk c sits at rank c, slot 0 (first half) or at rank 2cp-1-c, slot 1 (second half);
    # static slices, no index tensor: no host-to-device copy (CUDA-graph capturable)
    return torch.cat([chunks[c, 0] for c in range(cp)]
                     + [chunks[2 * cp - 1 - c, 1] for c in range(cp, 2 * cp)])


def local_sequence(full: torch.Tensor, group) -> torch.Tensor:
    """``[s, b, h]`` in token order -> this rank's Megatron CP chunks ``[s / cp, b, h]``."""
    cp, r = dist.get_world_size(group), dist.get_rank(group)
    chunks = full.view(2 * cp, full.shape[0] // (2 * cp), *full.shape[1:])
    return torch.cat([chunks[r], chunks[2 * cp - 1 - r]])


def to_stream(x: torch.Tensor, group, full: bool = False) -> torch.Tensor:
    """Megatron ``[s / cp, b, h]`` -> FLA's ``[1, b * s / cp, h]`` (this rank's contiguous piece
    of the flattened stream), or the whole stream ``[1, b * s, h]`` with ``full``."""
    cp, r = dist.get_world_size(group), dist.get_rank(group)
    stream = full_sequence(x, group).transpose(0, 1).flatten(0, 1)  # [b * s, h]
    if not full:
        n = stream.shape[0] // cp
        stream = stream[r * n:(r + 1) * n]
    return stream.unsqueeze(0)


def from_stream(y: torch.Tensor, group, batch: int, full: bool = False) -> torch.Tensor:
    """Inverse of :func:`to_stream`: ``[1, b * s / cp, h]`` (or the whole stream) -> Megatron
    ``[s / cp, b, h]``."""
    y = y.squeeze(0)
    stream = y if full else _gather(y, group).flatten(0, 1)  # [b * s, h]
    full_seq = stream.view(batch, -1, stream.shape[-1]).transpose(0, 1)  # [s, b, h]
    return local_sequence(full_seq, group)


_CONTEXTS: dict = {}


def cp_context(group, batch: int, seq_len: int, conv1d_kernel_size: int | None, device):
    """FLA's CP context for ``batch`` samples of ``seq_len`` tokens (one sequence each),
    cached per shape."""
    key = (id(group), batch, seq_len, conv1d_kernel_size, str(device))
    ctx = _CONTEXTS.get(key)
    if ctx is None:
        from fla.ops.cp import build_cp_context

        cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.long)
        ctx = build_cp_context(cu.to(device), group=group, conv1d_kernel_size=conv1d_kernel_size,
                               cu_seqlens_cpu=cu)
        _CONTEXTS[key] = ctx
    return ctx


# --------------------------------------------------------------------------- packed (THD) batches
def thd_rank_indices(cu_cpu: torch.Tensor, cp: int) -> torch.Tensor:
    """``[cp, total / cp]``: the global token positions each CP rank holds, in its local order, for a packed
    batch with (padded) document boundaries ``cu_cpu``: Megatron's / TE's THD sharding
    (``thd_get_partitioned_indices``) splits every document -- padded to a multiple of 2 cp -- into 2 cp
    chunks, and rank r holds chunks r and 2 cp - 1 - r of every document, document by document."""
    cu = cu_cpu.to(torch.long).tolist()
    out = [[] for _ in range(cp)]
    for a, b in zip(cu[:-1], cu[1:]):
        if (b - a) % (2 * cp):
            raise ValueError(f"packed CP: document [{a}, {b}) is not padded to a multiple of 2 cp = {2 * cp}")
        c = (b - a) // (2 * cp)
        for r in range(cp):
            out[r].append(torch.arange(a + r * c, a + (r + 1) * c))
            out[r].append(torch.arange(a + (2 * cp - 1 - r) * c, a + (2 * cp - r) * c))
    return torch.stack([torch.cat(o) if o else torch.zeros(0, dtype=torch.long) for o in out])


class ThdStream:
    """One packed micro-batch under context parallelism, between Megatron's THD layout (rank r: chunks r and
    2 cp - 1 - r of every document) and FLA's stream layout (rank r: the r-th contiguous piece of the token
    stream, the documents in order; fla.ops.cp passes the recurrent state across the pieces).

    Built once per micro-batch (one host sync for the boundaries) and shared by every layer, forward,
    recompute and backward; see :func:`thd_stream`. The conversions all-gather at the model width (the
    all-gather's backward is a reduce-scatter, so the gradients land where the inputs came from)."""

    def __init__(self, cu: torch.Tensor, group):
        self.group = group
        self.cp, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.cu = cu.to(torch.long)
        self.cu_cpu = self.cu.cpu()
        self.total = int(self.cu_cpu[-1])
        if self.total % self.cp:
            raise ValueError(f"packed CP: {self.total} tokens do not split into {self.cp} pieces")
        self.n = self.total // self.cp
        idx = thd_rank_indices(self.cu_cpu, self.cp).to(cu.device)  # [cp, n]
        self.local_idx = idx[self.rank]
        self.inv = torch.empty(self.total, dtype=torch.long, device=cu.device)
        self.inv[idx.flatten()] = torch.arange(self.total, device=cu.device)
        self._contexts = {}

    def full(self, x: torch.Tensor, differentiable: bool = True) -> torch.Tensor:
        """This rank's THD tokens ``[n, ...]`` -> the whole stream ``[total, ...]`` in token order."""
        if differentiable:
            parts = _gather(x, self.group)
        else:
            bufs = [torch.empty_like(x) for _ in range(self.cp)]
            dist.all_gather(bufs, x.contiguous(), group=self.group)
            parts = torch.stack(bufs)
        return parts.flatten(0, 1).index_select(0, self.inv)

    def piece(self, full: torch.Tensor) -> torch.Tensor:
        """The whole stream -> this rank's contiguous piece ``[n, ...]``."""
        return full[self.rank * self.n:(self.rank + 1) * self.n]

    def to_stream(self, x: torch.Tensor) -> torch.Tensor:
        return self.piece(self.full(x))

    def from_stream(self, y: torch.Tensor) -> torch.Tensor:
        """This rank's stream piece ``[n, ...]`` -> its THD tokens ``[n, ...]``."""
        return _gather(y, self.group).flatten(0, 1).index_select(0, self.local_idx)

    def fla_context(self, conv1d_kernel_size: int | None):
        """FLA's CP context for the documents of this micro-batch (cached per kernel size)."""
        ctx = self._contexts.get(conv1d_kernel_size)
        if ctx is None:
            from fla.ops.cp import build_cp_context

            ctx = build_cp_context(self.cu, group=self.group, conv1d_kernel_size=conv1d_kernel_size,
                                   cu_seqlens_cpu=self.cu_cpu)
            self._contexts[conv1d_kernel_size] = ctx
        return ctx


_STREAMS: dict = {}
_STREAMS_SIZE = 128


def thd_stream(cu: torch.Tensor, group) -> ThdStream:
    """The :class:`ThdStream` of the micro-batch with boundaries ``cu`` (the padded, global cu_seqlens),
    cached by tensor identity: forward, recompute and backward of interleaved micro-batches reuse theirs."""
    key = (id(cu), id(group))
    hit = _STREAMS.get(key)
    if hit is not None and hit[0] is cu:
        return hit[1]
    st = ThdStream(cu, group)
    _STREAMS[key] = (cu, st)
    while len(_STREAMS) > _STREAMS_SIZE:
        _STREAMS.pop(next(iter(_STREAMS)))
    return st
