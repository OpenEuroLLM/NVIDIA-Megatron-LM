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
    data_sharding_strategy='data_parallel',
    data_sharding_virtual_shards=None,
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
            data_sharding_strategy=data_sharding_strategy,
            data_sharding_virtual_shards=data_sharding_virtual_shards,
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
    data_sharding_strategy='data_parallel',
    data_sharding_virtual_shards=None,
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
            data_sharding_strategy=data_sharding_strategy,
            data_sharding_virtual_shards=data_sharding_virtual_shards,
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


def _collect_rank_samples(
    *,
    total_samples=1024,
    consumed_samples=0,
    micro_batch_size=2,
    global_batch_size=64,
    data_parallel_rank,
    data_parallel_size,
    data_sharding=True,
    data_sharding_strategy='data_parallel',
    data_sharding_virtual_shards=None,
):
    sampler = MegatronPretrainingRandomSampler(
        DummyDataset(),
        total_samples=total_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        data_sharding=data_sharding,
        data_sharding_strategy=data_sharding_strategy,
        data_sharding_virtual_shards=data_sharding_virtual_shards,
    )

    samples = []
    for microbatch in sampler:
        samples.extend(microbatch)
    return samples


def test_data_parallel_data_sharding_global_batches_depend_on_data_parallel_size():
    dp4_batches = _collect_global_batches(data_parallel_size=4)
    dp8_batches = _collect_global_batches(data_parallel_size=8)

    assert dp4_batches != dp8_batches


def test_no_data_sharding_global_batches_are_data_parallel_invariant():
    dp4_batches = _collect_global_batches(data_parallel_size=4, data_sharding=False)
    dp8_batches = _collect_global_batches(data_parallel_size=8, data_sharding=False)

    assert dp4_batches == dp8_batches


def test_virtual_data_sharding_defaults_virtual_shards_to_global_batch_size():
    dp4_batches = _collect_global_batches(
        data_parallel_size=4,
        data_sharding_strategy='virtual',
    )
    dp8_batches = _collect_global_batches(
        data_parallel_size=8,
        data_sharding_strategy='virtual',
    )

    assert dp4_batches == dp8_batches


def test_virtual_data_sharding_supports_explicit_virtual_shards():
    dp4_batches = _collect_global_batches(
        data_parallel_size=4,
        data_sharding_strategy='virtual',
        data_sharding_virtual_shards=16,
    )
    dp8_batches = _collect_global_batches(
        data_parallel_size=8,
        data_sharding_strategy='virtual',
        data_sharding_virtual_shards=16,
    )

    assert dp4_batches == dp8_batches


def test_virtual_shards_can_preserve_existing_physical_sharding_stream():
    physical_dp8_batches = _collect_global_batches(
        data_parallel_size=8,
        num_global_batches=1024 // 64,
    )
    virtual_dp4_batches = _collect_global_batches(
        data_parallel_size=4,
        num_global_batches=1024 // 64,
        data_sharding_strategy='virtual',
        data_sharding_virtual_shards=8,
    )

    assert physical_dp8_batches == virtual_dp4_batches


def test_virtual_shards_preserve_existing_physical_stream_at_consumed_offset():
    consumed_samples = 3 * 64
    physical_dp8_batches = _collect_global_batches(
        consumed_samples=consumed_samples,
        data_parallel_size=8,
        num_global_batches=(1024 - consumed_samples) // 64,
    )
    virtual_dp4_batches = _collect_global_batches(
        consumed_samples=consumed_samples,
        data_parallel_size=4,
        num_global_batches=(1024 - consumed_samples) // 64,
        data_sharding_strategy='virtual',
        data_sharding_virtual_shards=8,
    )

    assert physical_dp8_batches == virtual_dp4_batches


def test_virtual_data_sharding_respects_consumed_samples():
    dp4_batches = _collect_global_batches(
        consumed_samples=3 * 64,
        data_parallel_size=4,
        data_sharding_strategy='virtual',
    )
    dp8_batches = _collect_global_batches(
        consumed_samples=3 * 64,
        data_parallel_size=8,
        data_sharding_strategy='virtual',
    )

    assert dp4_batches == dp8_batches


def test_virtual_data_sharding_drops_incomplete_final_global_batch():
    active_samples = 1024
    total_samples = active_samples + 7
    data_parallel_size = 4
    samples = []

    for data_parallel_rank in range(data_parallel_size):
        samples.extend(
            _collect_rank_samples(
                total_samples=total_samples,
                data_parallel_rank=data_parallel_rank,
                data_parallel_size=data_parallel_size,
                data_sharding_strategy='virtual',
            )
        )

    assert len(samples) == active_samples
    assert set(samples) == set(range(active_samples))


def test_virtual_data_sharding_resumes_inside_global_batch_without_off_by_one():
    consumed_samples = 3 * 2 * 4
    data_parallel_size = 4
    local_consumed_samples = consumed_samples // data_parallel_size

    for data_parallel_rank in range(data_parallel_size):
        full_epoch_samples = _collect_rank_samples(
            total_samples=1024 + 7,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding_strategy='virtual',
        )
        resumed_samples = _collect_rank_samples(
            total_samples=1024 + 7,
            consumed_samples=consumed_samples,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding_strategy='virtual',
        )

        assert resumed_samples == full_epoch_samples[local_consumed_samples:]


def test_virtual_data_sharding_rolls_epoch_after_dropped_remainder():
    total_samples = 1024 + 7
    consumed_samples = 1024
    sampler = MegatronPretrainingRandomSampler(
        DummyDataset(),
        total_samples=total_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=2,
        global_batch_size=64,
        data_parallel_rank=0,
        data_parallel_size=4,
        data_sharding=True,
        data_sharding_strategy='virtual',
    )

    microbatch = next(iter(sampler))

    assert sampler.epoch == 1
    assert all(idx < consumed_samples for idx in microbatch)


def test_virtual_data_sharding_is_data_parallel_invariant_through_dataloader():
    dp4_batches = _collect_global_batches_from_dataloader(
        data_parallel_size=4,
        data_sharding_strategy='virtual',
    )
    dp8_batches = _collect_global_batches_from_dataloader(
        data_parallel_size=8,
        data_sharding_strategy='virtual',
    )

    assert dp4_batches == dp8_batches


def test_virtual_data_sharding_requires_data_sharding():
    with pytest.raises(AssertionError, match='requires data sharding'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding=False,
            data_sharding_strategy='virtual',
        )


def test_virtual_data_sharding_rejects_zero_virtual_shards():
    with pytest.raises(AssertionError, match='greater than zero'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding_strategy='virtual',
            data_sharding_virtual_shards=0,
        )


def test_data_parallel_data_sharding_rejects_virtual_shards():
    with pytest.raises(AssertionError, match='requires virtual data sharding'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding_virtual_shards=16,
        )


def test_virtual_data_sharding_requires_virtual_shards_to_divide_global_batch_size():
    with pytest.raises(AssertionError, match='global_batch_size must be divisible'):
        _collect_global_batches(
            data_parallel_size=4,
            data_sharding_strategy='virtual',
            data_sharding_virtual_shards=10,
        )


def test_virtual_data_sharding_requires_virtual_shards_divisible_by_data_parallel_size():
    with pytest.raises(AssertionError, match='must be divisible by data_parallel_size'):
        _collect_global_batches(
            data_parallel_size=8,
            data_sharding_strategy='virtual',
            data_sharding_virtual_shards=4,
        )
