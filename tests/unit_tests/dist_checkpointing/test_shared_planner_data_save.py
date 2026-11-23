# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The save planner that sends the flattening map from one rank per distinct map writes the same
checkpoint metadata as DCP's DefaultSavePlanner. CPU / gloo: run with
torch.distributed.run --nproc-per-node 4."""

import os
import pickle

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from megatron.training.checkpointing import _SharedPlannerDataSavePlanner


@pytest.fixture(scope="module")
def group():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run with torch.distributed.run --nproc-per-node 4")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist.group.WORLD


def _state_dict():
    # ranks 0/2 and 1/3 are data-parallel replicas of two expert-parallel ranks
    ep = dist.get_rank() % 2
    torch.manual_seed(0)
    shared = torch.randn(4, 3)
    torch.manual_seed(1 + ep)
    return {
        "model": {"shared.weight": shared, f"experts.{ep}.weight": torch.randn(2, 5)},
        "optimizer": {"state": {0: {"step": 7, "exp_avg": shared * 2}}, "param_groups": [{"lr": 1.0}]},
        "iteration": 12,
    }


def _shared_dir(tmp_path_factory, name):
    path = [str(tmp_path_factory.mktemp(name)) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path)
    return path[0]


def _metadata(path):
    with open(os.path.join(path, ".metadata"), "rb") as f:
        return pickle.load(f)


def test_same_metadata_and_data(group, tmp_path_factory):
    ref_dir = _shared_dir(tmp_path_factory, "ref")
    new_dir = _shared_dir(tmp_path_factory, "new")
    dcp.save(_state_dict(), checkpoint_id=ref_dir, planner=dcp.DefaultSavePlanner())
    planner = _SharedPlannerDataSavePlanner()
    dcp.save(_state_dict(), checkpoint_id=new_dir, planner=planner)
    # the map travels from one rank per distinct map: ranks 0 and 1
    assert bool(planner.plan.planner_data) == (dist.get_rank() in (0, 1))
    dist.barrier()
    ref, new = _metadata(ref_dir), _metadata(new_dir)
    assert new.planner_data == ref.planner_data
    assert new.state_dict_metadata == ref.state_dict_metadata
    loaded = _state_dict()
    for v in loaded["model"].values():
        v.zero_()
    loaded["optimizer"]["state"][0]["exp_avg"].zero_()
    dcp.load(loaded, checkpoint_id=new_dir)
    expected = _state_dict()
    for k, v in expected["model"].items():
        assert torch.equal(loaded["model"][k], v)
    assert torch.equal(loaded["optimizer"]["state"][0]["exp_avg"], expected["optimizer"]["state"][0]["exp_avg"])
