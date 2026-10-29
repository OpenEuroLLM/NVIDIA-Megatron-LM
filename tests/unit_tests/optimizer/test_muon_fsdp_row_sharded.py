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


def test_muown_classical_row_sharded_stable_rank_floor(mesh):
    """The expert stable-rank floor sees the full matrices (not a rank's rows): sharded and
    unsharded steps agree, with the floor active."""
    os.environ["ANGULAR_MUOWN_COMPACT_CKPT"] = "1"
    rank = mesh.get_rank()
    r_min = 4.0
    options = dict(lr=0.05, nesterov=True, split_qkv=False, fp32_matmul_prec="highest",
                   scale_mode="spectral", extra_scale_factor=0.2, stable_rank_floor=r_min)
    fulls, shards, reference, sharded = _pair(TensorParallelMuownClassical, options, mesh, 11)
    for p in fulls + shards:
        p.allreduce = False  # expert parameters
    for f in fulls:  # collapse them: rank-1 dominated, far below the floor
        with torch.no_grad():
            f.add_(torch.outer(torch.randn(f.shape[0], device="cuda"),
                               torch.randn(f.shape[1], device="cuda")))
    for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
        lo, hi = _rows(counts, rank)
        with torch.no_grad():
            local(s).copy_(f[lo:hi])
    for step in range(4):
        torch.manual_seed(200 + step)
        _set_grads(fulls, shards, rank)
        reference.step()
        sharded.step()
        for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
            lo, hi = _rows(counts, rank)
            torch.testing.assert_close(local(s).detach(), f.detach()[lo:hi])
            torch.testing.assert_close(local(sharded.state[s]["g"]),
                                       reference.state[f]["g"][lo:hi])
    for f in fulls:
        if min(f.shape) > r_min:
            srank = f.detach().square().sum() / torch.linalg.matrix_norm(f.detach(), 2) ** 2
            assert srank > 0.8 * r_min, srank  # power-iteration sigma at a small gap


def test_muown_classical_row_sharded_spectral_cap(mesh):
    """The spectral cap on row shards: init references from all-reduced row sums, the cap on
    the full matrices at their owners; sharded and unsharded steps agree."""
    os.environ["ANGULAR_MUOWN_COMPACT_CKPT"] = "1"
    rank = mesh.get_rank()
    options = dict(lr=0.05, nesterov=True, split_qkv=False, fp32_matmul_prec="highest",
                   scale_mode="spectral", extra_scale_factor=0.2)
    torch.manual_seed(13)
    fulls = [torch.nn.Parameter(torch.randn(shape, device="cuda") * 0.2) for shape, _ in LAYOUTS]
    shards = [_shard(f.detach(), counts, mesh) for f, (_, counts) in zip(fulls, LAYOUTS)]
    for p in fulls + shards:
        p.allreduce = False
    reference = TensorParallelMuownClassical(fulls, spectral_cap=1.2, **options)
    sharded = TensorParallelMuownClassical(shards, spectral_cap=1.2, **options)
    for k, v in reference._spectral_ref.items():
        assert abs(sharded._spectral_ref[k] - v) < 1e-5 * v, (k, v, sharded._spectral_ref[k])
    for step in range(6):
        torch.manual_seed(300 + step)
        _set_grads(fulls, shards, rank)
        for f in fulls:  # a shared rank-1 push: grows sigma_max past the cap
            f.grad.add_(-3.0 * torch.outer(torch.ones(f.shape[0], device="cuda"),
                                           torch.ones(f.shape[1], device="cuda")))
        for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
            lo, hi = _rows(counts, rank)
            if hi > lo:
                s.grad = DTensor.from_local(f.grad[lo:hi].clone(), device_mesh=s.device_mesh,
                                            placements=[Shard(0)], run_check=False,
                                            shape=f.shape, stride=f.stride())
        reference.step()
        sharded.step()
        for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
            lo, hi = _rows(counts, rank)
            torch.testing.assert_close(local(s).detach(), f.detach()[lo:hi])
    for f in fulls:
        cap = 1.2 * reference._spectral_ref[tuple(f.shape)]
        assert float(torch.linalg.matrix_norm(f.detach(), ord=2)) < 1.05 * cap


def test_muown_classical_row_sharded_expert_orthogonal(mesh):
    """Muon-like expert weight balancing on row shards: the polar factor of the full matrix at
    its owner; sharded and unsharded steps agree."""
    os.environ["ANGULAR_MUOWN_COMPACT_CKPT"] = "1"
    rank = mesh.get_rank()
    options = dict(lr=0.05, nesterov=True, split_qkv=False, fp32_matmul_prec="highest",
                   scale_mode="spectral", extra_scale_factor=0.2, expert_orthogonal=True)
    fulls, shards, reference, sharded = _pair(TensorParallelMuownClassical, options, mesh, 17)
    for p in fulls + shards:
        p.allreduce = False
    for step in range(4):
        torch.manual_seed(400 + step)
        _set_grads(fulls, shards, rank)
        reference.step()
        sharded.step()
        for f, s, (_, counts) in zip(fulls, shards, LAYOUTS):
            lo, hi = _rows(counts, rank)
            torch.testing.assert_close(local(s).detach(), f.detach()[lo:hi], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(local(sharded.state[s]["g"]),
                                       reference.state[f]["g"][lo:hi], atol=2e-5, rtol=2e-5)


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


def test_muown_classical_row_sharded_headwise(mesh):
    """QKV / head-wise split (qkv_split_shapes, the owner splits the gathered matrix) with shards
    that cut through the head blocks."""
    os.environ["ANGULAR_MUOWN_COMPACT_CKPT"] = "1"
    rank = mesh.get_rank()
    shape, counts, split = (24, 8), (5, 0, 11, 8), [4, 4, 4]  # 2 groups of 3 heads of 4 rows
    torch.manual_seed(5)
    full = torch.nn.Parameter(torch.randn(shape, device="cuda") * 0.2)
    shard = _shard(full.detach(), counts, mesh)
    for p in (full, shard):
        p.is_qkv, p.qkv_split_shapes = True, split
    options = dict(lr=0.008, split_qkv=True, is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
                   fp32_matmul_prec="highest", scale_mode="spectral", extra_scale_factor=0.2)
    reference = TensorParallelMuownClassical([full], **options)
    sharded = TensorParallelMuownClassical([shard], **options)
    lo, hi = _rows(counts, rank)
    for step in range(3):
        torch.manual_seed(300 + step)
        full.grad = torch.randn_like(full)
        shard.grad = None if hi == lo else DTensor.from_local(
            full.grad[lo:hi].clone(), device_mesh=mesh, placements=[Shard(0)], run_check=False,
            shape=full.shape, stride=full.stride())
        reference.step()
        sharded.step()
        torch.testing.assert_close(local(shard).detach(), full.detach()[lo:hi])
        torch.testing.assert_close(local(sharded.state[shard]["v_norm"]),
                                   reference.state[full]["v_norm"][lo:hi])


def test_muown_classical_hsdp(mesh):
    """HSDP: a 2 x 2 mesh, rows sharded over the inner dim (uneven), replicated over the outer."""
    from torch.distributed.tensor.placement_types import Replicate

    os.environ["ANGULAR_MUOWN_COMPACT_CKPT"] = "1"
    mesh2 = init_device_mesh("cuda", (2, 2), mesh_dim_names=("outer", "inner"))
    inner = mesh2.get_local_rank("inner")
    layouts = [((12, 8), (5, 7)), ((8, 16), (0, 8)), ((6, 4), (4, 2))]
    torch.manual_seed(13)
    fulls = [torch.nn.Parameter(torch.randn(shape, device="cuda") * 0.2) for shape, _ in layouts]

    def shard(t, counts):
        lo = sum(counts[:inner])
        dt = DTensor.from_local(t[lo:lo + counts[inner]].clone(), device_mesh=mesh2,
                                placements=[Replicate(), Shard(0)], run_check=False,
                                shape=t.shape, stride=t.stride())
        update_uneven_dtensor_chunk_metadata(dt)
        return dt

    shards = [torch.nn.Parameter(shard(f.detach(), c)) for f, (_, c) in zip(fulls, layouts)]
    options = dict(lr=0.008, split_qkv=False, fp32_matmul_prec="highest", scale_mode="spectral",
                   extra_scale_factor=0.2)
    reference = TensorParallelMuownClassical(fulls, **options)
    sharded = TensorParallelMuownClassical(shards, **options)
    for step in range(3):
        torch.manual_seed(400 + step)
        for f, s_, (_, c) in zip(fulls, shards, layouts):
            f.grad = torch.randn_like(f)
            lo = sum(c[:inner])
            s_.grad = None if c[inner] == 0 else shard(f.grad, c)
        reference.step()
        sharded.step()
        for f, s_, (_, c) in zip(fulls, shards, layouts):
            lo = sum(c[:inner])
            torch.testing.assert_close(local(s_).detach(), f.detach()[lo:lo + c[inner]])
            torch.testing.assert_close(local(sharded.state[s_]["v_norm"]),
                                       reference.state[f]["v_norm"][lo:lo + c[inner]])
