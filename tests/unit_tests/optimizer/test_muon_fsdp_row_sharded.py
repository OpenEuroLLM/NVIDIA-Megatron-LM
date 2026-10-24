# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Muon-family optimizers on Megatron-FSDP's row-sharded (uneven Shard(0)) parameters match the
unsharded update: run with torch.distributed.run --nproc-per-node 4 (NCCL)."""

import os

import pytest
import torch

pytest.importorskip("emerging_optimizers")

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    update_uneven_dtensor_chunk_metadata,
)
from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelMuon,
    TensorParallelMuownClassical,
)
from megatron.core.optimizer.fsdp_row_exchange import local, orthogonalize_rows

# (rows, cols) and each rank's rows -- uneven, with empty shards, as Megatron-FSDP's bucket slices
LAYOUTS = [((12, 8), (5, 7, 0, 0)), ((8, 16), (0, 3, 5, 0)), ((6, 4), (1, 1, 1, 3)),
           ((16, 8), (0, 0, 0, 16)), ((12, 8), (3, 3, 3, 3))]


@pytest.fixture(scope="module")
def mesh():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run with torch.distributed.run --nproc-per-node 4")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return init_device_mesh("cuda", (4,))


def _rows(counts, rank):
    start = sum(counts[:rank])
    return start, start + counts[rank]


def _shard(full, counts, mesh):
    rank = mesh.get_rank()
    lo, hi = _rows(counts, rank)
    dt = DTensor.from_local(full[lo:hi].clone(), device_mesh=mesh, placements=[Shard(0)],
                            run_check=False, shape=full.shape, stride=full.stride())
    update_uneven_dtensor_chunk_metadata(dt)
    return torch.nn.Parameter(dt)


def test_exchange_matches_full_matrix_fn(mesh):
    """Owners see the full matrices: a function of all rows comes back row by row."""
    rank = mesh.get_rank()
    torch.manual_seed(3)
    fulls = [torch.randn(shape, device="cuda") for shape, _ in LAYOUTS]
    params = [_shard(f, counts, mesh) for f, (_, counts) in zip(fulls, LAYOUTS)]
    updates = [local(p).detach().clone() for p in params]
    metas = [((tuple(p.shape),), {}) for p in params]

    def fn(stack, _):  # depends on every row: column sums
        return stack * 2 + stack.sum(dim=1, keepdim=True)

    out = orthogonalize_rows(params, updates, metas, fn)
    for o, f, (_, counts) in zip(out, fulls, LAYOUTS):
        lo, hi = _rows(counts, rank)
        torch.testing.assert_close(o, fn(f[None], None)[0][lo:hi])


def _pair(cls, options, mesh, seed):
    torch.manual_seed(seed)
    fulls = [torch.nn.Parameter(torch.randn(shape, device="cuda") * 0.2) for shape, _ in LAYOUTS]
    shards = [_shard(f.detach(), counts, mesh) for f, (_, counts) in zip(fulls, LAYOUTS)]
    return fulls, shards, cls(fulls, **options), cls(shards, **options)


def _set_grads(fulls, shards, rank):
    for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
        f.grad = torch.randn_like(f)
        lo, hi = _rows(counts, rank)
        s.grad = None if hi == lo else DTensor.from_local(
            f.grad[lo:hi].clone(), device_mesh=s.device_mesh, placements=[Shard(0)],
            run_check=False, shape=f.shape, stride=f.stride())


@pytest.mark.parametrize("nesterov", [False, True])
def test_muown_classical_row_sharded(mesh, nesterov):
    os.environ["ANGULAR_MUOWN_COMPACT_CKPT"] = "1"
    rank = mesh.get_rank()
    options = dict(lr=0.008, nesterov=nesterov, split_qkv=False, fp32_matmul_prec="highest",
                   scale_mode="spectral", extra_scale_factor=0.2)
    fulls, shards, reference, sharded = _pair(TensorParallelMuownClassical, options, mesh, 7)
    for _ in range(4):
        torch.manual_seed(100 + _)
        _set_grads(fulls, shards, rank)
        reference.step()
        sharded.step()
        for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
            lo, hi = _rows(counts, rank)
            torch.testing.assert_close(local(s).detach(), f.detach()[lo:hi])
            for key in ("v_norm", "g", "m_g", "v_g", "m_u"):
                torch.testing.assert_close(local(sharded.state[s][key]),
                                           reference.state[f][key][lo:hi])


def test_muon_row_sharded(mesh):
    rank = mesh.get_rank()
    options = dict(lr=0.02, weight_decay=0.01, split_qkv=False, fp32_matmul_prec="highest",
                   batched_step=True)
    fulls, shards, reference, sharded = _pair(TensorParallelMuon, options, mesh, 11)
    for _ in range(3):
        torch.manual_seed(200 + _)
        _set_grads(fulls, shards, rank)
        reference.step()
        sharded.step()
        for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
            lo, hi = _rows(counts, rank)
            torch.testing.assert_close(local(s).detach(), f.detach()[lo:hi])
