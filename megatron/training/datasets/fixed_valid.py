# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""--fixed-valid-data-path: fixed held-out validation sets next to a split training blend, shared by the GPT and
hybrid pretraining entry points (pretrain_gpt.py, pretrain_hybrid.py)."""

import dataclasses
import os

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training.utils import print_rank_0


def fixed_valid_dataset_configs(config, args):
    """(training config, [validation config per set]) for --fixed-valid-data-path. The training config differs
    only in fields outside the dataset cache key (multiple_validation_sets, full_validation), so the training
    datasets keep their cache; per set: its one unweighted prefix, at its --fixed-valid-seq-lengths entry
    (default --valid-seq-length, else the training sequence length)."""
    train_config = dataclasses.replace(config, multiple_validation_sets=False, full_validation=False)
    lengths = getattr(args, 'fixed_valid_seq_lengths', None) or \
        [args.valid_seq_length or config.sequence_length] * len(args.fixed_valid_data_path)
    valid_configs = [
        dataclasses.replace(
            config,
            blend=None,
            split=None,
            blend_per_split=[None, ([prefix], None), None],
            sequence_length=length,
            multiple_validation_sets=True,
            full_validation=True,
        )
        for prefix, length in zip(args.fixed_valid_data_path, lengths)
    ]
    return train_config, valid_configs


def build_with_fixed_valid_sets(dataset_type, train_val_test_num_samples, is_dataset_built, config, args):
    """Training / test as without the flag (the split's validation part: 1 sample, unused); validation: the
    fixed sets, each completely (one epoch)."""
    train_config, valid_configs = fixed_valid_dataset_configs(config, args)
    sizes = list(train_val_test_num_samples)
    sizes[1] = 1                       # the split's own validation part: unused (a weighted blend needs a size)
    if sizes[2] is None:
        sizes[2] = 1
    train_ds, _, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, sizes, is_dataset_built, train_config
    ).build()
    valid_ds = []
    for valid_config in valid_configs:      # one set per config, each at its own sequence length
        _, v, _ = BlendedMegatronDatasetBuilder(
            dataset_type, [None, None, None], is_dataset_built, valid_config
        ).build()
        valid_ds.extend(v if isinstance(v, list) else [v])
    print_rank_0("> fixed validation sets: " + ", ".join(
        f"{os.path.basename(c.blend_per_split[1][0][0])} @ {c.sequence_length}" for c in valid_configs))
    return train_ds, valid_ds, test_ds
