# Copyright (c) 2025, Jouni Luoma / AMD.  All rights reserved.
"""PackedSFTDataset — loads pre-packed SFT sequences produced by pack_sft_data.py.

Each JSONL line must have:
    "tokens":    List[int]  — pre-tokenised token IDs, length == seq_len + 1
    "loss_mask": List[int]  — 1 for assistant tokens, 0 for prompt/pad/EOD; length == seq_len + 1

The extra token is needed for the standard LM shift (input=tokens[:-1],
labels=tokens[1:]), producing exactly seq_len tokens — which must be
divisible by TP size for sequence-parallel reduce-scatter.

Position IDs are built here by scanning for EOD tokens and resetting the
counter to 0 after each one.  This is required for two reasons:
  1. RoPE correctness — each conversation must start at position 0, not
     continue from wherever the previous conversation ended.
  2. Attention boundary enforcement — Megatron's varlen flash-attention path
     (--reset-attention-mask) detects sequence boundaries from position-id
     resets, not from an explicit mask passed by the dataloader.
"""
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split


class PackedSFTLowLevelDataset:
    """Low-level dataset that loads pre-packed JSONL produced by pack_sft_data.py."""

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("PackedSFTDataset requires the `datasets` library.")
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        return {
            "tokens":    row["tokens"],
            "loss_mask": row["loss_mask"],
        }


class PackedSFTDataset(MegatronDataset):
    """Mid-level dataset for pre-packed SFT sequences.

    Use this instead of SFTDataset when training data has been preprocessed
    with pack_sft_data.py.  Unlike SFTDataset, this class is compatible with
    --reset-position-ids and --reset-attention-mask so Megatron can handle
    conversation boundaries in packed sequences automatically.
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return PackedSFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[int(self.indices[idx % len(self.indices)])]

        tokens    = torch.tensor(row["tokens"],    dtype=torch.long)
        loss_mask = torch.tensor(row["loss_mask"], dtype=torch.float)

        seq_len = len(tokens)

        # Standard LM shift: predict token[i+1] from token[i].
        # The packed JSONL stores seq_len+1 tokens so that after the shift
        # we get exactly seq_len tokens, which must be divisible by TP size.
        input_tokens = tokens[:-1].contiguous()   # [seq_len]
        labels       = tokens[1:].contiguous()    # [seq_len]
        loss_mask    = loss_mask[1:].contiguous()  # align with labels

        # Build position IDs with EOD-based reset.
        # After each EOD token the counter resets to 0 so that the next
        # conversation starts at position 0.  This is what Megatron's varlen
        # flash-attention path uses to detect conversation boundaries.
        eod_id = self.config.tokenizer.eod
        eod_positions = (input_tokens == eod_id).nonzero(as_tuple=True)[0]
        position_ids = torch.arange(len(input_tokens), dtype=torch.long)
        for eod_pos in eod_positions:
            offset = position_ids[eod_pos] + 1
            position_ids[eod_pos + 1:] -= offset

        # Derive cu_seqlens and max_seqlen from position ID resets.
        starts = (position_ids == 0).nonzero(as_tuple=True)[0]
        cu_seqlens = torch.cat([starts, torch.tensor([len(position_ids)])])
        cu_seqlens = cu_seqlens.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seqlens.max()

        return {
            "tokens":       input_tokens,
            "labels":       labels,
            "loss_mask":    loss_mask,
            "position_ids": position_ids,
            "cu_seqlens":   cu_seqlens,
            "max_seqlen":   max_seqlen,
        }
