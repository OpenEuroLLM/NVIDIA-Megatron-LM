# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Post-optimizer-step weight normalizer for nGPT-style normalized transformers.

In nGPT (`arXiv:2410.01131 <https://arxiv.org/abs/2410.01131>`_) there is no
LayerNorm/RMSNorm in the trunk: the *only* thing keeping the weight matrices --
and hence the hidden states -- on the unit hypersphere is a renormalization
applied after every optimizer step. This module implements that hook.

Conventions
-----------
Each matrix is normalized along its *embedding* dimension:

* **Input / read matrices** (QKV, MLP ``fc1``, expert ``fc1``, embeddings) map
  the residual stream to a new space. Their embedding dimension is the input
  dimension, so rows are normalized (``dim=1``).
* **Output / write matrices** (attention ``proj``, MLP ``fc2``, expert ``fc2``)
  write back into the residual stream. Under the strict nGPT convention their
  embedding dimension is the *output* dimension, so columns are normalized
  (``dim=0``). Set ``out_norm_dim_0=False`` to row-normalize these instead.

Tensor parallelism
------------------
Megatron shards column-parallel weights along ``dim=0`` and row-parallel weights
along ``dim=1`` (recorded in ``param.partition_dim``). When the reduction axis of
a normalization coincides with the sharded axis, the sum of squares is
all-reduced across the tensor-parallel group so the resulting norm matches the
unsharded matrix. Notably the strict nGPT convention (row-norm for inputs,
column-norm for outputs) never needs that all-reduce -- both are TP-local.
"""

import logging
from typing import Iterable

import torch

from megatron.core import parallel_state
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

# Substrings identifying matrices that read from the residual stream.
_INPUT_PROJECTION_TOKENS = (
    "linear_qkv",
    "linear_fc1",
    "linear_q_proj",
    "linear_kv_proj",
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
)

# Substrings identifying matrices that write into the residual stream.
_OUTPUT_PROJECTION_TOKENS = ("linear_proj", "linear_fc2")

# Substrings identifying token-embedding matrices.
_EMBEDDING_TOKENS = ("word_embeddings", "embedding.word_embeddings")

# Substrings identifying the LM head.
_HEAD_TOKENS = ("output_layer",)


def _is_any(name: str, tokens: Iterable[str]) -> bool:
    """Return True if any token appears in the (dotted) parameter name."""
    return any(tok in name for tok in tokens)


def classify_param(name: str, param: torch.Tensor) -> str | None:
    """Classify a parameter for weight normalization.

    Args:
        name: Fully-qualified parameter name.
        param: The parameter tensor.

    Returns:
        One of ``"input"``, ``"output"``, ``"embedding"``, ``"head"``, or None
        when the parameter must not be normalized (biases, norms, 1-D scalers,
        router weights, MoE expert biases).
    """
    # Never touch 1-D magnitude parameters: nGPT scalers, alphas, norm weights,
    # biases. Their whole purpose is to carry scale.
    if param.ndim < 2:
        return None
    if getattr(param, "is_ngpt_scaler", False):
        return None
    # The MoE router is a scoring function, not a residual-stream map; leaving it
    # unconstrained keeps routing logits expressive.
    if "router" in name:
        return None
    if _is_any(name, _HEAD_TOKENS):
        return "head"
    if _is_any(name, _EMBEDDING_TOKENS):
        return "embedding"
    if _is_any(name, _OUTPUT_PROJECTION_TOKENS):
        return "output"
    if _is_any(name, _INPUT_PROJECTION_TOKENS):
        return "input"
    return None


def _tp_group_for(param: torch.Tensor):
    """Return the tensor-parallel process group a parameter is sharded over."""
    if getattr(param, "expert_tp", False) or getattr(param, "is_expert", False):
        try:
            return parallel_state.get_expert_tensor_parallel_group()
        except (AssertionError, AttributeError):
            return parallel_state.get_tensor_model_parallel_group()
    return parallel_state.get_tensor_model_parallel_group()


@torch.no_grad()
def normalize_weight_(
    param: torch.Tensor,
    norm_dim: int,
    *,
    bounded: bool = False,
    max_norm: float = 1.0,
    eps: float = 1e-8,
) -> None:
    """Normalize ``param`` in place along ``norm_dim``, TP-aware.

    Args:
        param: Weight tensor of rank >= 2. For rank > 2 (grouped MoE experts) the
            normalization is applied independently per leading index.
        norm_dim: Axis to reduce over. ``1`` = row normalization (unit rows),
            ``0`` = column normalization (unit columns).
        bounded: If True only shrink rows whose norm exceeds ``max_norm``
            (projection onto the ball) instead of pinning them to it.
        max_norm: Target norm (``1.0`` reproduces the nGPT unit sphere).
        eps: Numerical floor to avoid division by zero.
    """
    data = param.data
    w = data.float()
    # Reduce over the *last two* dims' selected axis so grouped expert tensors of
    # shape [E, out, in] normalize per expert.
    axis = norm_dim + (w.ndim - 2)
    sq = w.pow(2).sum(dim=axis, keepdim=True)

    # If the reduction axis is sharded across TP, complete the sum of squares.
    partition_dim = getattr(param, "partition_dim", -1)
    if getattr(param, "tensor_model_parallel", False) and partition_dim == norm_dim:
        group = _tp_group_for(param)
        if group is not None and torch.distributed.get_world_size(group=group) > 1:
            torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM, group=group)

    norm = sq.sqrt().clamp(min=eps)
    if bounded:
        scale = (max_norm / norm).clamp(max=1.0)
    else:
        scale = max_norm / norm
    data.copy_((w * scale).to(data.dtype))


class WeightNormalizer:
    """Renormalizes model weights after every optimizer step.

    Args:
        model_chunks: The model chunks whose parameters should be normalized.
        out_norm_dim_0: Column-normalize output projections (strict nGPT
            convention). When False they are row-normalized instead.
        bounded: Project onto the ball (shrink-only) rather than onto the sphere.
        max_norm: Target norm for the constrained rows/columns.
        normalize_embeddings: Also normalize the token-embedding matrix.
        head_unconstrained: Leave the LM head out of normalization (it is then
            typically controlled by weight decay instead).
        normalize_every_n_steps: Throttle: only normalize every N steps.
    """

    def __init__(
        self,
        model_chunks: list[torch.nn.Module],
        out_norm_dim_0: bool = True,
        bounded: bool = False,
        max_norm: float = 1.0,
        normalize_embeddings: bool = True,
        head_unconstrained: bool = True,
        normalize_every_n_steps: int = 1,
    ) -> None:
        self.out_norm_dim_0 = out_norm_dim_0
        self.bounded = bounded
        self.max_norm = max_norm
        self.normalize_embeddings = normalize_embeddings
        self.head_unconstrained = head_unconstrained
        self.normalize_every_n_steps = max(1, int(normalize_every_n_steps))
        self._step_count = 0

        # Resolve the parameter -> (kind, norm_dim) plan once at construction.
        self._plan: list[tuple[str, torch.Tensor, int]] = []
        for chunk in model_chunks:
            module = chunk.module if hasattr(chunk, "module") else chunk
            for name, param in module.named_parameters():
                kind = classify_param(name, param)
                if kind is None:
                    continue
                if kind == "head" and self.head_unconstrained:
                    continue
                if kind == "embedding" and not self.normalize_embeddings:
                    continue
                if kind == "output":
                    norm_dim = 0 if self.out_norm_dim_0 else 1
                else:
                    # input / embedding / (constrained) head all read along dim 1
                    norm_dim = 1
                self._plan.append((name, param, norm_dim))

        log_single_rank(
            logger,
            logging.INFO,
            f"[WeightNormalizer] normalizing {len(self._plan)} tensors "
            f"(out_norm_dim_0={out_norm_dim_0}, bounded={bounded}, "
            f"head_unconstrained={head_unconstrained}, "
            f"every_n_steps={self.normalize_every_n_steps})",
        )

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Apply the normalization plan, honoring the step throttle."""
        self._step_count += 1
        if self._step_count % self.normalize_every_n_steps != 0:
            return
        for _name, param, norm_dim in self._plan:
            normalize_weight_(param, norm_dim, bounded=self.bounded, max_norm=self.max_norm)


def build_weight_normalizer(config, model_chunks: list[torch.nn.Module]):
    """Build a :class:`WeightNormalizer` from an optimizer config, or None.

    Args:
        config: An ``OptimizerConfig`` carrying the ``weight_normalizer_*`` fields.
        model_chunks: Model chunks to normalize.

    Returns:
        A configured :class:`WeightNormalizer`, or None when disabled.
    """
    if not getattr(config, "weight_normalizer", False):
        return None
    return WeightNormalizer(
        model_chunks,
        out_norm_dim_0=getattr(config, "weight_normalizer_out_norm_dim_0", True),
        bounded=getattr(config, "weight_normalizer_bounded", False),
        max_norm=getattr(config, "weight_normalizer_max_norm", 1.0),
        normalize_embeddings=getattr(config, "weight_normalizer_normalize_embeddings", True),
        head_unconstrained=getattr(config, "weight_normalizer_head_unconstrained", True),
        normalize_every_n_steps=getattr(config, "weight_normalizer_every_n_steps", 1),
    )
