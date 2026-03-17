# Copyright (c) 2025, Jouni Luoma / AMD.  All rights reserved.
"""PackedSFTDataset — loads pre-packed SFT sequences produced by pack_sft_data.py.

Each JSONL line must have:
    "tokens":    List[int]  — pre-tokenised token IDs, length == seq_len
    "loss_mask": List[int]  — 1 for assistant tokens, 0 for prompt/pad/EOD

Position IDs and attention mask are built at load time using the standard
get_ltor_masks_and_position_ids() helper, which resets at EOD boundaries when
--reset-position-ids / --reset-attention-mask are set.  This gives each
packed conversation the correct rotary position embeddings and prevents
attention from bleeding across conversation boundaries.
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
        # The packed JSONL already has full-length sequences (seq_len tokens),
        # so we store seq_len+1 and slice here — but since we only have seq_len,
        # we shift by dropping the last input token and first label token.
        input_tokens = tokens[:-1].contiguous()   # [seq_len - 1]
        labels       = tokens[1:].contiguous()    # [seq_len - 1]
        loss_mask    = loss_mask[1:].contiguous()  # align with labels

        # Position IDs — reset_position_ids is handled downstream by
        # get_ltor_masks_and_position_ids() in the pretraining data path.
        # We emit a simple 0..N-1 range here; the pretraining collation
        # code will overwrite it when --reset-position-ids is set.
        position_ids = torch.arange(seq_len - 1, dtype=torch.long)

        ret = {
            "tokens":      input_tokens,
            "labels":      labels,
            "loss_mask":   loss_mask,
            "position_ids": position_ids,
        }

        if self.config.create_attention_mask:
            ret["attention_mask"] = None  # built by collation / model

        return ret
