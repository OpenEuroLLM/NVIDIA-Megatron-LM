"""Newton-Schulz on Megatron-FSDP's row-sharded optimizer parameters.

Megatron-FSDP hands the optimizer every parameter as a DTensor with one uneven ``Shard(0)``
placement: each rank holds whole rows (the local shard is viewed as ``[-1, *shape[1:]]``). For
Muon-family optimizers every per-row quantity (momentum, Muown's row magnitudes / norms, Adam on
them) is then local; only the orthogonalization needs the full matrix. :func:`orthogonalize_rows`
gives it one *owner* rank per matrix:

    rows of every matrix  --all_to_all-->  owners (full matrices)  --batched Newton-Schulz-->
    full directions  --all_to_all-->  rows of every matrix

Per rank this moves its own rows out and back (2 x its shard), and the Newton-Schulz work is
spread over the group (each matrix's owner is the rank holding most of its rows). The matrices go in rounds of bounded size, in the same
global order on every rank of a group (the optimizer's parameter order), so every rank issues the
same collectives.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
except ImportError:  # pragma: no cover
    DTensor = None
    Shard = None

# Elements (over all ranks) per exchange round: bounds the owners' full-matrix buffers.
ROUND_NUMEL = int(os.environ.get("MUON_FSDP_ROUND_NUMEL", str(1 << 28)))
# Wire dtype of the exchange ("float32": bit-identical to the unsharded path; "bfloat16": half
# the traffic).
EXCHANGE_DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.environ.get("MUON_FSDP_EXCHANGE_DTYPE", "float32")
]


def _shard_dim(p: torch.Tensor) -> Optional[int]:
    """The mesh dimension sharding ``p`` on rows, if the others replicate it: Megatron-FSDP's
    [Shard(0)], or HSDP's [Replicate (outer), Shard(0) (inner)] (every replica group holds the
    same shards and runs the same exchange within its inner group)."""
    from torch.distributed.tensor.placement_types import Replicate

    dims = [i for i, pl in enumerate(p.placements) if isinstance(pl, Shard)]
    if len(dims) != 1 or p.placements[dims[0]].dim != 0:
        return None
    if not all(isinstance(pl, Replicate) for i, pl in enumerate(p.placements) if i != dims[0]):
        return None
    return dims[0]


def is_row_sharded(p: torch.Tensor) -> bool:
    """A Megatron-FSDP optimizer parameter: a DTensor sharded on dim 0 over one mesh dim (the
    other mesh dims, if any, replicating it)."""
    if DTensor is None or not isinstance(p, DTensor):
        return False
    return _shard_dim(p) is not None


def local(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The local shard of a DTensor (its storage), else the tensor itself."""
    if t is None:
        return None
    return t._local_tensor if DTensor is not None and isinstance(t, DTensor) else t


def shard_group(p: torch.Tensor) -> "torch.distributed.ProcessGroup":
    return p.device_mesh.get_group(_shard_dim(p))


def row_state_like(p: torch.Tensor, rows_local: torch.Tensor) -> torch.Tensor:
    """A per-row ``[rows, 1]`` state of row-sharded ``p`` as a DTensor with p's row sharding."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        update_uneven_dtensor_chunk_metadata,
    )

    out = DTensor.from_local(
        rows_local,
        device_mesh=p.device_mesh,
        placements=p.placements,
        run_check=False,
        shape=torch.Size((p.shape[0], 1)),
        stride=(1, 1),
    )
    update_uneven_dtensor_chunk_metadata(out)
    return out


class _RowCounts:
    """Rows of each matrix on each rank of its group: one all_gather per parameter set, cached."""

    def __init__(self):
        self._cache: Dict[Tuple, List[List[int]]] = {}

    def get(self, params: Sequence[torch.Tensor], group) -> List[List[int]]:
        key = (id(group), tuple(id(p) for p in params))
        counts = self._cache.get(key)
        if counts is None:
            mine = torch.tensor(
                [local(p).shape[0] for p in params], dtype=torch.long, device="cuda"
            )
            world = torch.distributed.get_world_size(group)
            gathered = [torch.empty_like(mine) for _ in range(world)]
            torch.distributed.all_gather(gathered, mine, group=group)
            per_rank = [g.tolist() for g in gathered]  # [rank][param]
            counts = [[per_rank[r][i] for r in range(world)] for i in range(len(params))]
            for i, p in enumerate(params):
                if sum(counts[i]) != p.shape[0]:
                    raise RuntimeError(
                        f"row-sharded parameter {tuple(p.shape)}: shards hold "
                        f"{sum(counts[i])} rows"
                    )
            self._cache[key] = counts
        return counts


_ROW_COUNTS = _RowCounts()


def _owners(rows: Sequence[Sequence[int]]) -> List[int]:
    """Each matrix's owner: the rank holding most of its rows (a Megatron-FSDP shard is a
    contiguous slice of a flat bucket, so a matrix spans few ranks): little of it moves, and the
    Newton-Schulz work follows the (even) shard sizes. Deterministic (ties: lowest rank)."""
    return [max(range(len(r)), key=lambda k: (r[k], -k)) for r in rows]


def _exchange_round(params, updates, metas, group, ns_fn) -> List[torch.Tensor]:
    world = torch.distributed.get_world_size(group)
    me = torch.distributed.get_rank(group)
    rows = _ROW_COUNTS.get(params, group)
    cols = [p.shape[1] for p in params]
    owner = _owners(rows)
    n = len(params)
    device = updates[0].device if updates else torch.device("cuda")
    first = [[0] * (world + 1) for _ in range(n)]  # first[i][r]: rank r's first row of matrix i
    for i in range(n):
        for r in range(world):
            first[i][r + 1] = first[i][r] + rows[i][r]
    mine = [i for i in range(n) if owner[i] == me]
    by_owner: Dict[int, List[int]] = {}
    for i in range(n):
        by_owner.setdefault(owner[i], []).append(i)
    empty = torch.empty(0, dtype=EXCHANGE_DTYPE, device=device)

    # rows -> owners (send order: owner, then matrix)
    send_parts, send_splits = [], [0] * world
    for o in range(world):
        for i in by_owner.get(o, ()):
            if rows[i][me]:
                send_parts.append(updates[i].reshape(-1).to(EXCHANGE_DTYPE))
                send_splits[o] += rows[i][me] * cols[i]
    recv_splits = [sum(rows[i][r] * cols[i] for i in mine) for r in range(world)]
    send = torch.cat(send_parts) if send_parts else empty
    recv = torch.empty(sum(recv_splits), dtype=EXCHANGE_DTYPE, device=device)
    torch.distributed.all_to_all_single(recv, send, recv_splits, send_splits, group=group)

    # owners: full matrices (rank-ordered pieces, one cat each), batched Newton-Schulz
    pieces: Dict[int, List[torch.Tensor]] = {i: [] for i in mine}
    offset = 0
    for r in range(world):
        for i in mine:
            k = rows[i][r] * cols[i]
            if k:
                pieces[i].append(recv[offset : offset + k])
                offset += k
    full = {i: torch.cat(pieces[i]).view(params[i].shape[0], cols[i]).float() for i in mine}
    batches: Dict[Tuple, List[int]] = {}
    for i in mine:
        batches.setdefault(metas[i][0], []).append(i)
    for idx in batches.values():
        out = ns_fn(torch.stack([full[i] for i in idx]), metas[idx[0]][1])
        for j, i in enumerate(idx):
            full[i] = out[j]

    # full directions -> rows (back order: destination, then matrix)
    back_parts, back_splits = [], [0] * world
    for r in range(world):
        for i in mine:
            if rows[i][r]:
                back_parts.append(full[i][first[i][r] : first[i][r + 1]].reshape(-1))
                back_splits[r] += rows[i][r] * cols[i]
    back = torch.cat(back_parts).to(EXCHANGE_DTYPE) if back_parts else empty
    got = torch.empty(sum(send_splits), dtype=EXCHANGE_DTYPE, device=device)
    torch.distributed.all_to_all_single(got, back, send_splits, back_splits, group=group)
    out: List[torch.Tensor] = [updates[i].new_empty((0, cols[i])) for i in range(n)]
    offset = 0
    for o in range(world):
        for i in by_owner.get(o, ()):
            k = rows[i][me] * cols[i]
            if k:
                out[i] = got[offset : offset + k].view(rows[i][me], cols[i]).float()
                offset += k
    return out


def orthogonalize_rows(
    params: Sequence[torch.Tensor],
    updates: Sequence[torch.Tensor],
    metas: Sequence[Tuple[tuple, dict]],
    ns_fn: Callable[[torch.Tensor, dict], torch.Tensor],
) -> List[torch.Tensor]:
    """The local rows of ``ns_fn`` applied to every full matrix.

    ``params`` (row-sharded DTensors, in the optimizer's parameter order, identical on every rank
    of each group), ``updates`` their local rows ``[rows_local, cols]`` (fp32), ``metas`` each
    matrix's (batch key, batch metadata) for ``ns_fn(stack [B, rows, cols], metadata)``.
    Returns the local rows of the orthogonalized matrices, in order.
    """
    out: List[Optional[torch.Tensor]] = [None] * len(params)
    by_group: Dict[int, Tuple[object, List[int]]] = {}
    for i, p in enumerate(params):
        g = shard_group(p)
        by_group.setdefault(id(g), (g, []))[1].append(i)
    for group, idx in by_group.values():
        start = 0
        while start < len(idx):
            end, total = start, 0
            while end < len(idx) and (end == start or total + params[idx[end]].numel()
                                      <= ROUND_NUMEL):
                total += params[idx[end]].numel()
                end += 1
            sel = idx[start:end]
            res = _exchange_round([params[i] for i in sel], [updates[i] for i in sel],
                                  [metas[i] for i in sel], group, ns_fn)
            for i, r in zip(sel, res):
                out[i] = r
            start = end
    return out
