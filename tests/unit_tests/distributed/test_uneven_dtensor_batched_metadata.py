# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The batched chunk-metadata gather of Megatron-FSDP's uneven DTensors matches the per-DTensor
one (uneven and empty shards, 1-D and 2-D meshes, Replicate x Shard as under HSDP). CPU / gloo:
run with torch.distributed.run --nproc-per-node 4."""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    _shard_order,
    gather_and_compute_chunk_metadata_batched,
    preprocess_state_dict_for_uneven_dtensor,
)


@pytest.fixture(scope="module")
def meshes():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run with torch.distributed.run --nproc-per-node 4")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return init_device_mesh("cpu", (4,)), init_device_mesh("cpu", (2, 2), mesh_dim_names=("o", "i"))


def _reference(dtensor):
    """The per-DTensor algorithm (one all_gather_object per sharded mesh dimension)."""
    local_shape = dtensor.to_local().shape
    offsets, cumulative = [0] * len(local_shape), list(local_shape)
    for mesh_dim in reversed(_shard_order(dtensor)):
        p = dtensor.placements[mesh_dim]
        if isinstance(p, Replicate):
            continue
        group = dtensor.device_mesh.get_group(mesh_dim)
        shapes = [None] * dist.get_world_size(group)
        dist.all_gather_object(shapes, cumulative, group=group)
        rank = dist.get_rank(group)
        offsets[p.dim] += sum(s[p.dim] for s in shapes[:rank])
        cumulative[p.dim] = sum(s[p.dim] for s in shapes)
    return tuple(offsets)


def _dtensor(mesh, placements, local_shape, global_shape):
    local = torch.randn(local_shape)
    return DTensor.from_local(local, device_mesh=mesh, placements=placements, run_check=False,
                              shape=torch.Size(global_shape),
                              stride=torch.empty(global_shape).stride())


def _cases(meshes):
    mesh1, mesh2 = meshes
    rank = dist.get_rank()
    rows1 = [5, 7, 0, 4][rank]  # uneven, one empty shard
    o, i = divmod(rank, 2)
    rows2 = [[3, 1], [2, 0]][o][i]
    cols2 = [[2, 6], [3, 1]][o][i]
    return [
        _dtensor(mesh1, [Shard(0)], (rows1, 8), (16, 8)),
        _dtensor(mesh1, [Shard(0)], (rows1 + 1, 3), (20, 3)),
        # HSDP: replicated across the outer dimension, sharded inside
        _dtensor(mesh2, [Replicate(), Shard(0)], ([4, 2][i], 5), (6, 5)),
        _dtensor(mesh2, [Shard(0), Shard(1)], (rows2, cols2), (6, 9)),
        _dtensor(mesh1, [Replicate()], (3, 3), (3, 3)),
    ]


def test_batched_matches_per_dtensor(meshes):
    cases = _cases(meshes)
    expected = [_reference(d) for d in cases]
    batched = gather_and_compute_chunk_metadata_batched(cases)
    assert [tuple(m.offsets) for m in batched] == expected
    assert [tuple(m.sizes) for m in batched] == [tuple(d.to_local().shape) for d in cases]


def test_preprocess_state_dict(meshes):
    cases = _cases(meshes)
    expected = [_reference(d) for d in cases]
    state_dict = {"model": {f"w{k}": d for k, d in enumerate(cases[:3])},
                  "optimizer": {"state": {"a": {"exp_avg": cases[3]}, "b": {"m": cases[4]}}}}
    preprocess_state_dict_for_uneven_dtensor(state_dict)
    got = [d._local_tensor.__create_chunk_list__()[0].offsets for d in cases]
    assert [tuple(g) for g in got] == expected
    write_items = cases[0]._local_tensor.__create_write_items__("model.w0", cases[0])
    if cases[0].to_local().numel():
        assert tuple(write_items[0].index.offset) == expected[0]
    else:
        assert write_items == []
