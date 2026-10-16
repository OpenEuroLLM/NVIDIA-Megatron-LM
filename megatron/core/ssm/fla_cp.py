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
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn


def _gather(x: torch.Tensor, group) -> torch.Tensor:
    """``[n, ...]`` per rank -> ``[cp, n, ...]`` (autograd: reduce-scatter backward)."""
    return torch.stack(dist_fn.all_gather(x.contiguous(), group=group))


def _megatron_order(cp: int) -> list[int]:
    """The chunk ids (of 2 cp) in the order ranks hold them: r, 2 cp - 1 - r per rank."""
    return [c for r in range(cp) for c in (r, 2 * cp - 1 - r)]


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
    chunks = parts.view(cp, 2, half, *x.shape[1:]).flatten(0, 1)  # in _megatron_order
    inverse = torch.empty(2 * cp, dtype=torch.long)
    inverse[torch.tensor(_megatron_order(cp))] = torch.arange(2 * cp)
    return chunks[inverse.to(chunks.device)].flatten(0, 1)


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
