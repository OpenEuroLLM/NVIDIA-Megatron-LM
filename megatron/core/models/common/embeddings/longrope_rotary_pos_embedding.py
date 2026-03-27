# Based on Microsoft LongRoPE reference (MIT License) - logic adapted for Megatron-LM integration.

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

logger = logging.getLogger(__name__)


def _load_rescale_factors(path: str, device) -> Tensor:
    """Load per-dimension rescale factors from file.

    Supported formats:
      - .pt / .pth : torch.load -> Tensor or list
      - .npy       : numpy array
      - .txt/.csv  : one float per line or comma/space separated

    Returns:
        Float32 tensor on the given device.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LongRoPE rescale factors file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, (list, tuple)):
            data = torch.tensor(data, dtype=torch.float32)
        elif not isinstance(data, torch.Tensor):
            raise ValueError(f"Unexpected object in {path}: {type(data)}")
        rescale = data.to(torch.float32)
    elif ext == ".npy":
        import numpy as np

        arr = np.load(path)
        rescale = torch.from_numpy(arr).to(torch.float32)
    else:
        # text format
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().replace(",", " ").split()
        vals = [float(x) for x in content if x.strip() != ""]
        rescale = torch.tensor(vals, dtype=torch.float32)

    if rescale.dim() != 1:
        raise ValueError(f"Rescale factors must be 1-D. Got shape {tuple(rescale.shape)}")
    return rescale.to(device=device, dtype=torch.float32)


def _calc_mscale(scale: float, policy: str, original_ctx: int) -> float:
    """Compute magnitude scaling factor.

    Policies:
      su    : sqrt(1 + log(scale) / log(original_ctx))
      yarn  : 0.1 * log(scale) + 1
      <float literal> : direct numeric value
    """
    if scale <= 1.0:
        return 1.0
    if policy == "su":
        return math.sqrt(1.0 + math.log(scale) / math.log(original_ctx))
    if policy == "yarn":
        return 0.1 * math.log(scale) + 1.0
    try:
        return float(policy)
    except ValueError:
        raise ValueError(f"Unknown longrope_magnitude_scaling_policy: {policy}")


class LongRoPERotaryEmbedding(RotaryEmbedding):
    """LongRoPE Rotary Embedding integrated with Megatron-LM.

    Differences vs base RotaryEmbedding:
      - Per-dimension rescale_factors loaded from file
      - Modified inverse frequencies: inv_freq[i] = 1 / (rescale_factor[i] * base^(...))
      - Returns (embedding, mscale) tuple like YaRN so attention applies magnitude scaling

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention.
        rotary_percent (float): Percent of rotary dimension to use.
        rescale_factors_path (str): Path to LongRoPE rescale factors file.
        max_position_embeddings (int): Target (extended) context length.
        original_max_position_embeddings (int): Original context length before extension.
        magnitude_scaling_policy (str): Magnitude scaling policy: "su", "yarn", or float.
        rotary_interleaved (bool): If True, interleaved rotary position embeddings.
        rotary_base (float): Base period for rotary position embeddings.
        use_cpu_initialization (bool): If True, initialize on CPU.
        cp_group: Process group for context parallel.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rescale_factors_path: str = None,
        max_position_embeddings: int = 65536,
        original_max_position_embeddings: int = 4096,
        magnitude_scaling_policy: str = "su",
        rotary_interleaved: bool = False,
        rotary_base: float = 10000,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if rotary_interleaved:
            raise ValueError("LongRoPE does not currently support rotary_interleaved=True")

        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = (
            original_max_position_embeddings or max_position_embeddings
        )
        self.scale_ratio = self.max_position_embeddings / float(
            self.original_max_position_embeddings
        )
        self.magnitude_scaling_policy = magnitude_scaling_policy

        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()
        self._rescale_factors = _load_rescale_factors(rescale_factors_path, device=device)

        # Initialize base RotaryEmbedding (builds self.inv_freq)
        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=False,
            rotary_base=rotary_base,
            rope_scaling=False,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
        )

        # After base init, self.inv_freq shape = (effective_dim/2,)
        expected = self.inv_freq.shape[0]
        if self._rescale_factors.shape[0] != expected:
            raise ValueError(
                f"Mismatch: LongRoPE rescale factors length ({self._rescale_factors.shape[0]}) "
                f"!= rotary half-dim ({expected})."
            )

        # Recompute inv_freq with per-dimension rescale factors.
        # Base formula: inv_freq = 1 / (base ** (i/dim))
        # LongRoPE:     inv_freq = 1 / (rescale_factor * base ** (i/dim))
        exponent = torch.arange(0, expected * 2, 2, dtype=torch.float32, device=device) / (
            expected * 2
        )
        self.inv_freq = (
            1.0 / (self._rescale_factors * (rotary_base ** exponent))
        ).to(device=device, dtype=torch.float32)

        # Magnitude scaling factor – returned via forward like YaRN
        self.mscale = float(
            _calc_mscale(
                self.scale_ratio,
                self.magnitude_scaling_policy,
                self.original_max_position_embeddings,
            )
        )

        # Clear the lru_cache for the forward method to prevent memory leaks.
        self.forward.cache_clear()

    @lru_cache(maxsize=32)
    def forward(
        self, max_seq_len: int, offset: int = 0, packed_seq: bool = False
    ) -> Tuple[Tensor, float]:
        """Forward pass of LongRoPE Rotary Embedding.

        Args:
            max_seq_len (int): Maximum size of sequence.
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether using packed sequences. Defaults to False.

        Returns:
            Tuple of (emb, mscale):
                emb: [seq_length, 1, 1, dim] positional embedding tensor
                mscale: float magnitude scaling factor for attention
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())

        seq = (
            torch.arange(
                max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            + offset
        )
        freqs = torch.outer(seq, self.inv_freq)  # [seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
        emb = emb[:, None, None, :]  # [seq, 1, 1, dim]

        if self.cp_group is not None and self.cp_group.size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)

        return emb, self.mscale
