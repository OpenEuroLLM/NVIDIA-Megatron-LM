# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.core import mpu
from megatron.core.datasets.utils import Split

from megatron.training import get_args
from megatron.training.dist_signal_handler import DistributedSignalHandler


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    if hasattr(dataset, 'split'):
        split = dataset.split
    elif hasattr(dataset, 'index_split'):
        split = dataset.index_split
    else:
        split = None

    if args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset

    # Use eval-specific batch sizes for validation/test splits
    is_eval = split in (Split.valid, Split.test)
    micro_batch_size = getattr(args, 'eval_micro_batch_size', args.micro_batch_size) if is_eval else args.micro_batch_size
    global_batch_size = getattr(args, 'eval_global_batch_size', args.global_batch_size) if is_eval else args.global_batch_size

    if split == Split.valid and args.full_validation:
        batch_sampler = MegatronFullValidationSampler(
            total_samples=len(dataset),
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'single':
        if args.hybrid_context_parallel:
            batch_sampler = HybridCPMegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size())
        else:
            # Megatron sampler
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic' and getattr(
        args, 'data_sampler_reference_dp_size', None
    ):
        # keep the sample order of a reference layout across a node-count change
        batch_sampler = MegatronPretrainingReferenceLayoutSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            reference_dp_size=args.data_sampler_reference_dp_size,
            reference_micro_batch_size=(
                args.data_sampler_reference_micro_batch_size
                if args.data_sampler_reference_micro_batch_size
                else micro_batch_size
            ),
            reference_global_batch_size=getattr(
                args, 'data_sampler_reference_global_batch_size', None
            ),
        )
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )
    else:
        raise Exception('{} dataloader type is not supported.'.format(args.dataloader_type))

    # MEGATRON_SAMPLE_DUMP_DIR: write the first MEGATRON_SAMPLE_DUMP_N (default 200) training
    # micro-batches' sample indices of every rank to <dir>/train_rank<global rank>.jsonl -- to
    # check that two layouts train on the same samples per step
    dump_dir = os.environ.get("MEGATRON_SAMPLE_DUMP_DIR")
    if dump_dir and split not in (Split.valid, Split.test):
        batch_sampler = _SampleDumpSampler(
            batch_sampler, dump_dir, int(os.environ.get("MEGATRON_SAMPLE_DUMP_N", "200"))
        )

    def worker_init_fn(_):
        import os

        # Defensively close GPU device FDs in worker processes so workers do not
        # keep references into NVIDIA memory space. This helps ensure GPU memory
        # can be reclaimed even if a dataloader worker is delayed or fails to exit.
        def close_nvidia_fds():
            for fd in os.listdir("/proc/self/fd"):
                try:
                    path = os.readlink(f"/proc/self/fd/{fd}")
                    if path.startswith("/dev/nvidia"):
                        os.close(int(fd))
                except OSError:
                    pass

        close_nvidia_fds()
        if args.exit_signal_handler:
            DistributedSignalHandler(args.exit_signal).__enter__()

    maybe_worker_init_fn = (
        worker_init_fn if args.num_workers > 0 else None
    )
    # Torch dataloader.
    if args.hybrid_context_parallel:
        extra_kwargs = {"collate_fn": lambda x: x,}
    else:
        extra_kwargs = {}

    # Make the DataLoader prefetch depth configurable. Upstream never passes
    # prefetch_factor, so it silently takes torch's default of 2 batches queued
    # per worker. On a shared parallel filesystem that buffer is the only thing
    # absorbing read-latency spikes: once it drains, every rank in the collective
    # waits for the slowest reader, which shows up as periodic dips in TFLOP/s
    # rather than as an error.
    #
    # prefetch_factor is a better stabiliser than num_workers here, because it
    # deepens the queue WITHOUT adding more concurrent readers -- raising
    # num_workers at 1024 nodes multiplies the processes hammering the same GPFS
    # and can make contention worse, not better.
    #
    # None (the default) preserves upstream behaviour exactly: the kwarg is not
    # passed and torch applies its own default. torch also rejects
    # prefetch_factor when num_workers == 0, hence the guard.
    #
    # MEGATRON_DATALOADER_PREFETCH_FACTOR is the older env-driven form of the
    # same knob (it syncs via PYTHONPATH with no container rebuild); the flag
    # wins when both are set.
    prefetch_factor = getattr(args, "dataloader_prefetch_factor", None)
    if prefetch_factor is None and os.environ.get("MEGATRON_DATALOADER_PREFETCH_FACTOR"):
        prefetch_factor = int(os.environ["MEGATRON_DATALOADER_PREFETCH_FACTOR"])
    if prefetch_factor is not None and args.num_workers > 0:
        extra_kwargs["prefetch_factor"] = prefetch_factor

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=maybe_worker_init_fn,
        **extra_kwargs,
    )

class MegatronPretrainingSampler:
    """
    Sampler for Megatron pretraining dataloaders that divides data samples across
    data parallel workers. Each worker receives a contiguous chunk of data determined by
    its rank and the micro batch size. Supports dropping the last incomplete batch if
    specified, and keeps track of total and consumed samples. Designed to work with
    distributed training using Megatron's data parallelism.
    """

    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert (
            self.consumed_samples < self.total_samples
        ), 'no samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        """
        Calculate the start and end indices for the current data parallel worker's
        chunk within a batch.

        Returns:
            tuple: (start_idx, end_idx) indicating the slice of the batch for this worker.
        """
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

class HybridCPMegatronPretrainingSampler(MegatronPretrainingSampler):
    """
    Data sampler for hybrid context parallel (Hybrid CP) format.
    This data sampler pulls in the entire global batch at once across all data parallel ranks.
    This helps provide the Hybrid CP Dataloader Wrapper to schedule and load balance sub-samples
    of the entire global batch.
    """

    def __init__(self, total_samples, consumed_samples, micro_batch_size, global_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        super().__init__(total_samples, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size, drop_last)
        self.global_batch_size = global_batch_size
        self.data_parallel_size = data_parallel_size
        self.num_micro_batches = self.global_batch_size // self.micro_batch_times_data_parallel_size

    def __len__(self):
        return self.total_samples

    def get_start_end_idx_global_batch(self):
        start_idx = [self.data_parallel_rank * self.micro_batch_size + i * self.micro_batch_size * self.data_parallel_size for i in range(self.num_micro_batches)]
        end_idx = [start_idx[i] + self.micro_batch_size for i in range(self.num_micro_batches)]
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size * self.num_micro_batches:
                start_idx, end_idx = self.get_start_end_idx_global_batch()
                global_batch_idx = []
                for i in range(self.num_micro_batches):
                    global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
                yield global_batch_idx
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx_global_batch()
            global_batch_idx = []
            for i in range(self.num_micro_batches):
                global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
            yield global_batch_idx


class MegatronFullValidationSampler:
    """Sampler for full validation that handles small datasets gracefully.

    This sampler is designed for validation datasets that may be smaller than
    data_parallel_size * micro_batch_size. It uses micro_batch_size=1 to minimize
    the samples needed per batch and properly handles partial batches where some
    ranks may not have data.
    """

    def __init__(self, total_samples, data_parallel_rank, data_parallel_size):
        self.total_samples = total_samples
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = 1  # Always use 1 for small dataset support

        # Sanity checks
        assert self.total_samples > 0, f'no sample to consume: {self.total_samples}'
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            f'data_parallel_rank should be smaller than data size: {self.data_parallel_rank}, {data_parallel_size}'

    def __len__(self):
        """Returns the number of batches this rank will yield."""
        # Each batch takes data_parallel_size samples (1 per rank)
        # This rank gets samples at indices: data_parallel_rank, data_parallel_rank + data_parallel_size, ...
        num_batches = 0
        for batch_idx in range(0, self.total_samples, self.data_parallel_size):
            # Check if this rank has data in this batch
            sample_idx = batch_idx + self.data_parallel_rank
            if sample_idx < self.total_samples:
                num_batches += 1
        return num_batches

    def __iter__(self):
        """Yield batches for this data parallel rank."""
        for batch_idx in range(0, self.total_samples, self.data_parallel_size):
            # Check if this rank has data in this batch
            sample_idx = batch_idx + self.data_parallel_rank
            if sample_idx < self.total_samples:
                # Yield a batch with a single sample index for this rank
                yield [sample_idx]


class RandomSeedDataset(Dataset):
    """
    A dataset wrapper that resets the random seed before each sample.

    This ensures deterministic behavior per sample by setting the RNG state
    for torch, numpy, and random before accessing each underlying data sample.
    The base seed is retrieved from training arguments, and can be varied per epoch
    using the set_epoch method to ensure different shuffling or augmentation each epoch.

    Args:
        dataset: The underlying dataset to wrap.

    Methods:
        set_epoch(epoch): Change the seed offset so each epoch produces different randomization.
        __getitem__(idx): Sets the seed based on the sample index and current epoch.
    """

    def __init__(self, dataset, seed):
        self.base_seed = seed
        self.curr_seed = seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        """
        Change the seed offset so each epoch produces different randomization.

        Args:
            epoch: The epoch number to use as the seed offset.
        """
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:
    """
    Sampler for Megatron pretraining dataloaders that performs random sampling
    across data parallel workers. Supports data sharding to divide the dataset
    into buckets and shuffle within each bucket. Designed to work with distributed
    training using Megatron's data parallelism.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronPretrainingReferenceLayoutSampler:
    """The cyclic sampler's sample order of a REFERENCE layout, served to any layout.

    ``MegatronPretrainingRandomSampler`` orders the samples by the data-parallel layout: with data
    sharding every rank shuffles its own bucket of ``total // (mbs x dp) x mbs`` samples, so a
    different node count (or micro-batch size) reorders the whole stream. This sampler computes,
    for every global step, exactly the samples the reference layout (``reference_dp_size`` ranks x
    ``reference_micro_batch_size``) would have used in that step, and splits them over the actual
    ranks (``data_parallel_size`` x ``micro_batch_size``). Every step trains on the same set of
    samples as in the reference layout (the gradient is the same sum, up to reduction order); with
    the actual layout equal to the reference it yields exactly the reference sampler's batches.

    The global batch size must stay the same; consumed_samples (from the checkpoint) is a step
    boundary. Steps that straddle an epoch end take the next epoch's first micro-batches, as the
    reference run's re-created iterator does.

    ``reference_global_batch_size`` (default: the actual global batch size): the reference run's
    global batch size. The reference steps, each ordered as above (reference rank, reference
    micro-batch, position) and concatenated, form one sample stream; an actual step of a SMALLER
    (or any other) global batch size trains on the next ``global_batch_size`` samples of that stream
    (``consumed_samples`` counts samples of it). So 8 steps at 1/8 of the reference batch train on
    exactly the samples of one reference step, in its order, and a run that switches back to the
    reference batch at a reference-step boundary continues the reference run's data exactly
    (recall surgery: small-batch arms from a main-run checkpoint, docs/recall_surgery_ambitious.md
    §10). At reference_global_batch_size == global_batch_size the batches are unchanged.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        global_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        reference_dp_size,
        reference_micro_batch_size,
        reference_global_batch_size=None,
    ):
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.ref_dp = reference_dp_size
        self.ref_mbs = reference_micro_batch_size
        self.ref_mb_dp = self.ref_mbs * self.ref_dp
        self.ref_gbs = reference_global_batch_size or global_batch_size
        assert total_samples > 0, f'no sample to consume: {total_samples}'
        assert data_parallel_rank < data_parallel_size
        assert self.ref_gbs % self.ref_mb_dp == 0, (
            f'reference global batch {self.ref_gbs} is not a multiple of the reference layout '
            f'{self.ref_dp} x {self.ref_mbs}'
        )
        assert global_batch_size % (micro_batch_size * data_parallel_size) == 0
        self.ref_microbatches = self.ref_gbs // self.ref_mb_dp  # per reference step
        self.per_rank = global_batch_size // data_parallel_size  # samples per rank per step
        # micro-batches per reference rank and epoch (the reference sampler's bucket length)
        self.mb_per_epoch = total_samples // self.ref_mb_dp
        assert self.mb_per_epoch > 0
        self._perm_epoch, self._perm = None, None

    def __len__(self):
        return self.total_samples

    def _perm_for(self, epoch):
        if epoch != self._perm_epoch:
            g = torch.Generator()
            g.manual_seed(epoch)
            if self.data_sharding:
                size = self.mb_per_epoch * self.ref_mbs  # the reference bucket
            else:
                size = (self.total_samples // self.ref_mbs) * self.ref_mbs
            self._perm = torch.randperm(size, generator=g)
            self._perm_epoch = epoch
        return self._perm

    def _reference_samples(self, ref_rank, ref_mb, offsets):
        """Samples ``offsets`` of global reference micro-batch ``ref_mb`` of rank ``ref_rank``."""
        epoch, j = divmod(ref_mb, self.mb_per_epoch)
        perm = self._perm_for(epoch)
        if self.data_sharding:
            bucket = self.mb_per_epoch * self.ref_mbs
            return [ref_rank * bucket + int(perm[j * self.ref_mbs + i]) for i in offsets]
        return [
            int(perm[j * self.ref_mb_dp + i * self.ref_dp + ref_rank]) for i in offsets
        ]

    def __iter__(self):
        assert self.consumed_samples % self.global_batch_size == 0, (
            f'consumed_samples {self.consumed_samples} is not a step boundary '
            f'(global batch {self.global_batch_size})'
        )
        self.epoch = self.consumed_samples // (self.mb_per_epoch * self.ref_mb_dp)
        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)
        per_ref_rank = self.ref_microbatches * self.ref_mbs  # samples per reference rank and step
        lo = self.data_parallel_rank * self.per_rank
        while True:
            # this rank's slice [lo, lo + per_rank) of the step's samples = the positions
            # [consumed + lo, consumed + lo + per_rank) of the reference stream: reference step
            # q // ref_gbs, inside it ordered by (reference rank, reference micro-batch, position)
            samples = []
            q = self.consumed_samples + lo
            end = q + self.per_rank
            while q < end:
                ref_step, pos = divmod(q, self.ref_gbs)
                ref_rank, rem = divmod(pos, per_ref_rank)
                k, i0 = divmod(rem, self.ref_mbs)
                n = min(self.ref_mbs - i0, end - q)
                samples += self._reference_samples(
                    ref_rank, ref_step * self.ref_microbatches + k, range(i0, i0 + n)
                )
                q += n
            for b in range(0, self.per_rank, self.micro_batch_size):
                self.consumed_samples += self.micro_batch_size * self.data_parallel_size
                yield samples[b : b + self.micro_batch_size]


class _SampleDumpSampler:
    """Wraps a batch sampler; writes the first ``limit`` micro-batches it yields (one JSON line
    each: the consumed-samples counter at the start and the sample indices) per rank."""

    def __init__(self, inner, dump_dir, limit):
        self.inner, self.dump_dir, self.limit = inner, dump_dir, limit

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        import json

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"train_rank{rank}.jsonl")
        written = 0
        with open(path, "a") as f:
            for batch in self.inner:
                if written < self.limit:
                    f.write(json.dumps({"ids": [int(x) for x in batch]}) + "\n")
                    f.flush()
                    written += 1
                yield batch
