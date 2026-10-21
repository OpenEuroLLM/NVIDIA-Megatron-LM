# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""MEGATRON_LAZY_SINGLETON_GROUPS: a single-rank group gets no eager NCCL communicator under a
device-bound default group, and collectives on it still work (megatron/core/parallel_state.py)."""

import pytest
import torch

from megatron.core import parallel_state
from tests.unit_tests.test_utilities import Utils


def _non_torch():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return total - free - torch.cuda.memory_reserved()


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2+ GPUs")
def test_lazy_singleton_group(monkeypatch):
    Utils.initialize_distributed()
    default_pg = torch.distributed.distributed_c10d._get_default_group()
    device = torch.device("cuda", torch.cuda.current_device())
    saved = default_pg.bound_device_id
    default_pg.bound_device_id = device  # as init_process_group(device_id=...) leaves it
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    try:
        monkeypatch.setattr(parallel_state, "_LAZY_SINGLETON_GROUPS", True)
        before = _non_torch()
        mine = None
        for r in range(world):  # every rank creates every singleton group, like Megatron
            g = parallel_state.create_group([r], group_desc="TEST_SINGLETON")
            if r == rank:
                mine = g
        lazy = _non_torch() - before
        assert mine.bound_device_id == device
        x = torch.full((4,), float(rank + 1), device="cuda")
        torch.distributed.all_reduce(x, group=mine)  # lazily initialized, on the bound GPU
        assert torch.equal(x, torch.full((4,), float(rank + 1), device="cuda"))

        monkeypatch.setattr(parallel_state, "_LAZY_SINGLETON_GROUPS", False)
        before = _non_torch()
        for r in range(world):
            parallel_state.create_group([r], group_desc="TEST_SINGLETON_EAGER")
        eager = _non_torch() - before
        # a pair group is unaffected by the flag (and still works)
        monkeypatch.setattr(parallel_state, "_LAZY_SINGLETON_GROUPS", True)
        pair = parallel_state.create_group(list(range(world)), group_desc="TEST_ALL")
        y = torch.ones(2, device="cuda")
        torch.distributed.all_reduce(y, group=pair)
        assert torch.equal(y, torch.full((2,), float(world), device="cuda"))
        print(f"rank {rank}: singleton group eager {eager / 2**20:.0f} MiB, lazy {lazy / 2**20:.0f} MiB")
        assert lazy < eager, (lazy, eager)
    finally:
        default_pg.bound_device_id = saved
        Utils.destroy_model_parallel()
