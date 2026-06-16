# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.training.datasets.data_samplers import MegatronPretrainingRandomSampler


class DummyDataset:
    pass


class PositionDataset:
    def __len__(self):
        return self.num_samples

    def __init__(self, num_samples, sample_length=8):
        self.num_samples = num_samples
        self.sample_length = sample_length

    def __getitem__(self, idx):
        return torch.full((self.sample_length,), idx, dtype=torch.long)


def _collect_global_batches(
    *,
    total_samples=1024,
    consumed_samples=0,
    micro_batch_size=2,
    global_batch_size=64,
    data_parallel_size,
    num_global_batches=4,
    data_sharding=True,
    data_sharding_dp_invariant=False,
    data_sharding_dp_invariant_lanes=None,
):
    microbatches_per_global_batch = global_batch_size // (
        micro_batch_size * data_parallel_size
    )
    rank_microbatches = []
    for data_parallel_rank in range(data_parallel_size):
        sampler = MegatronPretrainingRandomSampler(
            DummyDataset(),
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding=data_sharding,
            data_sharding_dp_invariant=data_sharding_dp_invariant,
            data_sharding_dp_invariant_lanes=data_sharding_dp_invariant_lanes,
        )
        iterator = iter(sampler)
        rank_microbatches.append(
            [
                next(iterator)
                for _ in range(num_global_batches * microbatches_per_global_batch)
            ]
        )

    global_batches = []
    for global_batch_idx in range(num_global_batches):
        batch = []
        start = global_batch_idx * microbatches_per_global_batch
        end = start + microbatches_per_global_batch
        for rank_batches in rank_microbatches:
            for microbatch in rank_batches[start:end]:
                batch.extend(microbatch)
        global_batches.append(frozenset(batch))
    return global_batches


def _collect_global_batches_from_dataloader(
    *,
    total_samples=1024,
    consumed_samples=0,
    micro_batch_size=2,
    global_batch_size=64,
    data_parallel_size,
    num_global_batches=4,
    data_sharding=True,
    data_sharding_dp_invariant=False,
    data_sharding_dp_invariant_lanes=None,
):
    microbatches_per_global_batch = global_batch_size // (
        micro_batch_size * data_parallel_size
    )
    dataset = PositionDataset(total_samples)
    rank_microbatches = []
    for data_parallel_rank in range(data_parallel_size):
        sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding=data_sharding,
            data_sharding_dp_invariant=data_sharding_dp_invariant,
            data_sharding_dp_invariant_lanes=data_sharding_dp_invariant_lanes,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
        )
        iterator = iter(dataloader)
        rank_batches = []
        for _ in range(num_global_batches * microbatches_per_global_batch):
            microbatch = next(iterator)
            assert torch.all(microbatch == microbatch[:, :1])
            rank_batches.append(microbatch[:, 0].tolist())
        rank_microbatches.append(rank_batches)

    global_batches = []
    for global_batch_idx in range(num_global_batches):
        batch = []
        start = global_batch_idx * microbatches_per_global_batch
        end = start + microbatches_per_global_batch
        for rank_batches in rank_microbatches:
            for microbatch in rank_batches[start:end]:
                batch.extend(microbatch)
        global_batches.append(frozenset(batch))
    return global_batches


def test_physical_data_sharding_global_batches_depend_on_data_parallel_size():
    dp4_batches = _collect_global_batches(data_parallel_size=4)
    dp8_batches = _collect_global_batches(data_parallel_size=8)

    assert dp4_batches != dp8_batches


def test_no_data_sharding_global_batches_are_data_parallel_invariant():
    dp4_batches = _collect_global_batches(data_parallel_size=4, data_sharding=False)
    dp8_batches = _collect_global_batches(data_parallel_size=8, data_sharding=False)

    assert dp4_batches == dp8_batches


def test_dp_invariant_data_sharding_defaults_lanes_to_global_batch_size():
    dp4_batches = _collect_global_batches(
        data_parallel_size=4,
        data_sharding_dp_invariant=True,
    )
    dp8_batches = _collect_global_batches(
        data_parallel_size=8,
        data_sharding_dp_invariant=True,
    )

    assert dp4_batches == dp8_batches


def test_dp_invariant_data_sharding_supports_explicit_virtual_lanes():
    dp4_batches = _collect_global_batches(
        data_parallel_size=4,
        data_sharding_dp_invariant=True,
        data_sharding_dp_invariant_lanes=16,
    )
    dp8_batches = _collect_global_batches(
        data_parallel_size=8,
        data_sharding_dp_invariant=True,
        data_sharding_dp_invariant_lanes=16,
    )

    assert dp4_batches == dp8_batches


def test_dp_invariant_lanes_can_preserve_existing_physical_sharding_stream():
    physical_dp8_batches = _collect_global_batches(
        data_parallel_size=8,
        num_global_batches=1024 // 64,
    )
    invariant_dp4_batches = _collect_global_batches(
        data_parallel_size=4,
        num_global_batches=1024 // 64,
        data_sharding_dp_invariant=True,
        data_sharding_dp_invariant_lanes=8,
    )

    assert physical_dp8_batches == invariant_dp4_batches


def test_dp_invariant_lanes_preserve_existing_physical_stream_at_consumed_offset():
    consumed_samples = 3 * 64
    physical_dp8_batches = _collect_global_batches(
        consumed_samples=consumed_samples,
        data_parallel_size=8,
        num_global_batches=(1024 - consumed_samples) // 64,
    )
    invariant_dp4_batches = _collect_global_batches(
        consumed_samples=consumed_samples,
        data_parallel_size=4,
        num_global_batches=(1024 - consumed_samples) // 64,
        data_sharding_dp_invariant=True,
        data_sharding_dp_invariant_lanes=8,
    )

    assert physical_dp8_batches == invariant_dp4_batches


def test_dp_invariant_data_sharding_respects_consumed_samples():
    dp4_batches = _collect_global_batches(
        consumed_samples=3 * 64,
        data_parallel_size=4,
        data_sharding_dp_invariant=True,
    )
    dp8_batches = _collect_global_batches(
        consumed_samples=3 * 64,
        data_parallel_size=8,
        data_sharding_dp_invariant=True,
    )

    assert dp4_batches == dp8_batches


def test_dp_invariant_data_sharding_is_data_parallel_invariant_through_dataloader():
    dp4_batches = _collect_global_batches_from_dataloader(
        data_parallel_size=4,
        data_sharding_dp_invariant=True,
    )
    dp8_batches = _collect_global_batches_from_dataloader(
        data_parallel_size=8,
        data_sharding_dp_invariant=True,
    )

    assert dp4_batches == dp8_batches


def test_dp_invariant_data_sharding_requires_data_sharding():
    with pytest.raises(AssertionError, match='requires data sharding'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding=False,
            data_sharding_dp_invariant=True,
        )


def test_dp_invariant_data_sharding_rejects_zero_lanes():
    with pytest.raises(AssertionError, match='greater than zero'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding_dp_invariant=True,
            data_sharding_dp_invariant_lanes=0,
        )


def test_dp_invariant_data_sharding_requires_lanes_to_divide_global_batch_size():
    with pytest.raises(AssertionError, match='global_batch_size must be divisible'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding_dp_invariant=True,
            data_sharding_dp_invariant_lanes=10,
        )


def test_dp_invariant_data_sharding_requires_lanes_divisible_by_data_parallel_size():
    with pytest.raises(AssertionError, match='must be divisible by data_parallel_size'):
        _collect_global_batches(
            data_parallel_size=8,
            data_sharding_dp_invariant=True,
            data_sharding_dp_invariant_lanes=4,
        )
