# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Token-indexed tables inside the stack: per-layer embeddings and value embeddings.

Per-layer embeddings (PLE; Gemma 3n / Gemma 4). Every chosen layer owns a small token
embedding ``e_l[token]`` (``per_layer_embedding_dim``, e.g. 256). After the layer, the
hidden state gates it and it is added as one more residual update::

    h <- h + Norm(W_up (act(W_gate h) * e_l[token]))

The tables are large but only looked up (vocabulary-parallel over TP, like the word
embeddings); the dense cost is two ``hidden x dim`` projections per layer. Gemma also
adds a projection of the input embedding to ``e_l``; that needs the first stage's
embedding on every pipeline stage and is left out here.

Value embeddings (modded-nanogpt). An attention layer adds a token table to its values,
``v <- lambda_v v + lambda_e ve[token]`` (per key/value head, TP-sharded over heads), a
cheap way to feed token identity straight into what attention moves around.
"""

from typing import Optional

import torch
from torch import nn

from megatron.core.transformer.module import MegatronModule


def _act(name: str):
    return {"gelu": nn.functional.gelu, "silu": nn.functional.silu}[name]


class PerLayerEmbedding(MegatronModule):
    """PLE for one layer: ``h + Norm(W_up(act(W_gate h) * e[token]))``.

    A MegatronModule so the checkpoint walk reaches the vocab-parallel table's own
    sharded_state_dict (a plain nn.Module stores each TP rank's rows as the table)."""

    def __init__(self, config, vocab_size: int, tp_group) -> None:
        super().__init__(config)
        from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
        from megatron.core.transformer.post_norm import build_post_norm

        dim = config.per_layer_embedding_dim
        self.config, self.dim = config, dim
        self.act = _act(config.per_layer_embedding_activation)
        self.table = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            init_method=config.init_method,
            reduce_scatter_embeddings=config.sequence_parallel,
            config=config,
            tp_group=tp_group,
        )
        self.gate = nn.Linear(config.hidden_size, dim, bias=False, dtype=config.params_dtype)
        self.up = nn.Linear(dim, config.hidden_size, bias=False, dtype=config.params_dtype)
        config.init_method(self.gate.weight)
        config.output_layer_init_method(self.up.weight)
        self.norm = build_post_norm(config)
        for p in (self.gate.weight, self.up.weight):
            # Act per token: under SP each TP rank sees its own tokens.
            p.sequence_parallel = config.sequence_parallel
        # A lookup table: Adam under the Muon-family optimizers.
        self.table.weight.use_adam_with_emerging_optimizers = True

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        e = self.table(input_ids)  # [s(/tp), b, dim] under SP, else [b, s, dim]
        if not self.config.sequence_parallel:
            e = e.transpose(0, 1)
        g = self.act(self.gate(hidden_states))
        return hidden_states + self.norm(self.up(g * e.to(g.dtype)))


class ValueEmbedding(nn.Module):
    """A token table of one attention layer's local key/value heads."""

    def __init__(self, config, vocab_size: int, num_kv_heads_local: int, head_dim: int) -> None:
        super().__init__()
        self.shape = (num_kv_heads_local, head_dim)
        self.weight = nn.Parameter(
            torch.empty(
                vocab_size, num_kv_heads_local * head_dim,
                device=torch.cuda.current_device(), dtype=config.params_dtype,
            )
        )
        with torch.no_grad():
            self.weight.normal_(0.0, config.init_method_std)
        # TP-sharded over the heads (columns), like the values themselves.
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 1
        self.weight.partition_stride = 1
        self.weight.use_adam_with_emerging_optimizers = True
        # v <- lambda_v v + lambda_e ve
        self.lambdas = nn.Parameter(torch.tensor([1.0, 1.0], device=torch.cuda.current_device()))
        self.lambdas.is_ngpt_scaler = True  # a scalar: Adam, no weight decay group
        self.tp_group = None

    def forward(self, value: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """``value`` ``[s, b, h_kv, d]`` (or ``[t, h_kv, d]`` packed), ``input_ids``
        ``[b, s]`` (``[1, t]``) of the same tokens."""
        ve = torch.nn.functional.embedding(input_ids.transpose(0, 1), self.weight)
        ve = ve.reshape(value.shape).to(value.dtype)
        lam = self.lambdas.to(value.dtype)
        return lam[0] * value + lam[1] * ve

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 1},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=(metadata or {}).get("dp_cp_group"),
        )


def needs_input_ids(config) -> bool:
    """Whether layers other than Engram look up the token ids."""
    return bool(
        getattr(config, "per_layer_embedding_dim", 0) or getattr(config, "value_embedding", False)
    )
