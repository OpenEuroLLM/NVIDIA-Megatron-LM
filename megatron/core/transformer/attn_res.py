# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

r"""Block Attention Residuals (AttnRes).

Implements Block AttnRes from the Kimi Team technical report
(`arXiv:2603.15031 <https://arxiv.org/abs/2603.15031>`_), which replaces the
fixed, uniformly-weighted residual accumulation with a *learned softmax attention
over depth*.

Standard residuals give every layer the same uniformly-weighted sum of all prior
outputs, :math:`h_l = h_1 + \sum_i f_i(h_i)`, which makes hidden-state magnitude
grow as :math:`O(L)` and dilutes each layer's relative contribution. AttnRes
replaces this with (paper Eq. 1, 4)

.. math::
    h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i, \qquad
    \alpha_{i \to l} = \frac{\phi(q_l, k_i)}{\sum_j \phi(q_l, k_j)}

with :math:`\phi(q, k) = \exp(q^\top \mathrm{RMSNorm}(k))` -- softmax attention
along depth. The query :math:`q_l = w_l \in \mathbb{R}^d` is a *learnable
pseudo-query* per sublayer, decoupled from the layer's own forward computation
(paper Eq. 3); the inner RMSNorm stops large-magnitude layers from dominating.
Because the weights are convex, the hidden-state magnitude stays bounded.

Full AttnRes needs every layer output alive, i.e. :math:`O(Ld)` memory and -- under
pipeline parallelism -- :math:`O(Ld)` cross-stage traffic. **Block AttnRes**
(paper Eq. 5, 6) partitions the ``L`` layers into ``N`` blocks of ``S = L/N``, sums
each block's outputs into a single representation :math:`b_n = \sum_{j \in B_n}
f_j(h_j)`, and attends only over the ``N`` block summaries plus the running
partial sum of the current block. That brings memory and communication down to
:math:`O(Nd)`. ``N = L`` recovers Full AttnRes; ``N = 1`` degenerates to a standard
residual with the embedding isolated. The paper finds ``N ~ 8`` recovers most of
the benefit.

Note the payload that must cross a pipeline stage boundary is ``N + 1`` tensors of
shape ``[s, b, d]`` -- exactly the shape of the ``n``-stream payload in
:mod:`~megatron.core.transformer.mhc`, so the block count plays the same role as
mHC's expansion rate.

Composition with nGPT
---------------------
With ``lerp`` supplied, the intra-block accumulation
:math:`b_n \mathrel{+}= f_j(h_j)` becomes an nGPT LERP step toward the normalized
sublayer output, so block summaries stay on the unit sphere instead of growing
with block size.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


def _maybe_compile(fn):
    """Compile ``fn`` with Inductor, falling back to eager if unavailable.

    The depth-wise aggregation is a chain of small per-source kernels run twice
    per layer, so it is launch-bound rather than compute-bound; fusing it is
    worth far more than the arithmetic it performs.
    """
    state = {"fn": None}

    def wrapper(*args, **kwargs):
        if state["fn"] is None:
            try:
                state["fn"] = torch.compile(fn, dynamic=False)
            except Exception:  # pragma: no cover - depends on the runtime
                state["fn"] = fn
        try:
            return state["fn"](*args, **kwargs)
        except Exception:  # pragma: no cover - Inductor/Triton unavailable
            state["fn"] = fn
            return fn(*args, **kwargs)

    return wrapper


def _attn_res_aggregate_loop(query, gain, eps, *sources):
    """Depth-wise softmax aggregation without stacking the sources.

    The stacked form materializes a [k, s, b, d] tensor whose cost grows with the
    block count -- enough that attnres8 (9 sources) lost a micro-batch step to it.
    This form keeps the per-source loop, so peak memory stays at ~2 [s, b, d]
    buffers, and relies on the compiler to fuse the many small kernels the loop
    would otherwise launch. torch.compile specializes per source count, and there
    are only a handful of distinct counts across the model.
    """
    qg = (query * gain).to(sources[0].dtype)
    logits = []
    for v in sources:
        dot = torch.einsum("sbd,d->sb", v, qg).float()
        inv = torch.rsqrt(v.float().pow(2).mean(dim=-1) + eps)
        logits.append(dot * inv)
    w = torch.softmax(torch.stack(logits, dim=0), dim=0).to(sources[0].dtype)
    out = sources[0] * w[0].unsqueeze(-1)
    for i in range(1, len(sources)):
        out = out + sources[i] * w[i].unsqueeze(-1)
    return out


def _attn_res_aggregate(
    values: torch.Tensor, query: torch.Tensor, gain: torch.Tensor, eps: float
) -> torch.Tensor:
    """Softmax-over-depth aggregation of stacked sources.

    Args:
        values: ``[k, s, b, d]`` stacked attention sources.
        query: ``[d]`` learnable pseudo-query.
        gain: ``[d]`` RMSNorm gain applied to the keys.
        eps: Numerical floor for the RMS scale.

    Returns:
        ``[s, b, d]`` attended hidden state.
    """
    qg = (query * gain).to(values.dtype)
    dot = torch.einsum("ksbd,d->ksb", values, qg).float()
    inv = torch.rsqrt(values.float().pow(2).mean(dim=-1) + eps)
    weights = torch.softmax(dot * inv, dim=0).to(values.dtype)
    return torch.einsum("ksb,ksbd->sbd", weights, values)


_attn_res_aggregate_compiled = _maybe_compile(_attn_res_aggregate)
_attn_res_loop_compiled = _maybe_compile(_attn_res_aggregate_loop)


class BlockAttnResState:
    """Mutable depth-wise state threaded through a transformer block.

    Holds the completed block representations and the partial sum of the block
    currently being accumulated. One instance lives per forward pass.

    Attributes:
        blocks: Completed block representations, each ``[s, b, d]``. ``blocks[0]``
            is the token embedding, which the paper always keeps as a source.
        partial: Running partial sum of the in-progress block, or None if no
            sublayer has contributed to it yet.
    """

    def __init__(self, embedding: torch.Tensor) -> None:
        self.blocks: list[torch.Tensor] = [embedding]
        self.partial: torch.Tensor | None = None

    def sources(self) -> list[torch.Tensor]:
        """Return the value tensors to attend over: completed blocks + partial sum."""
        if self.partial is None:
            return self.blocks
        return self.blocks + [self.partial]

    def seal_block(self) -> None:
        """Close the in-progress block, promoting its partial sum to a block rep."""
        if self.partial is not None:
            self.blocks.append(self.partial)
            self.partial = None

    def to_payload(self) -> torch.Tensor:
        """Stack the state into a single ``[k, s, b, d]`` tensor for PP transfer."""
        return torch.stack(self.sources(), dim=0)

    @classmethod
    def from_payload(cls, payload: torch.Tensor, has_partial: bool) -> "BlockAttnResState":
        """Rebuild the state from a stacked payload received across a PP boundary."""
        tensors = list(torch.unbind(payload, dim=0))
        state = cls.__new__(cls)
        if has_partial:
            state.blocks, state.partial = tensors[:-1], tensors[-1]
        else:
            state.blocks, state.partial = tensors, None
        return state


class BlockAttnRes(nn.Module):
    """Depth-wise softmax attention over block representations.

    One instance per sublayer (attention and MLP each own one), matching the
    paper's treatment of every attention/FFN as an individual layer with its own
    pseudo-query.

    Args:
        hidden_size: Model hidden dimension ``d``.
        eps: Numerical floor for the inner RMSNorm.
        query_init_std: Std of the normal used to initialize the pseudo-query.
            Small values start the depth-attention close to uniform, which
            reproduces the standard residual's averaging behavior at init.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, query_init_std: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # Learnable pseudo-query w_l (paper Eq. 3). Decoupled from the layer's own
        # forward pass, which is what lets all queries in a block be batched.
        self.query = nn.Parameter(torch.zeros(hidden_size))
        # RMSNorm gain applied to the keys.
        self.key_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.query.is_ngpt_scaler = True
        self.key_norm_weight.is_ngpt_scaler = True
        # every parameter of the residual mixer: its own LR group (residual_mixer_lr_mult)
        for p in self.parameters(recurse=False):
            p.is_residual_mixer = True
        self.reset_parameters(query_init_std)

    def reset_parameters(self, query_init_std: float = 0.0) -> None:
        """Initialize the pseudo-query and key-norm gain."""
        with torch.no_grad():
            if query_init_std > 0.0:
                self.query.normal_(mean=0.0, std=query_init_std)
            else:
                # Zero query => uniform attention over depth at init, i.e. exactly
                # the averaging a standard residual performs.
                self.query.zero_()
            self.key_norm_weight.fill_(1.0)

    @torch.no_grad()
    def _record_arch_metrics(self, sources: list[torch.Tensor]) -> None:
        """Depth attention: its normalized entropy over the sources (0 = one source,
        1 = uniform) and the weight on the current partial block (the plain residual)."""
        from megatron.core.transformer import arch_metrics

        tag = arch_metrics.tag_of(self)
        if tag is None or len(sources) < 2:
            return
        kind, layer, n = tag
        qg = (self.query.float() * self.key_norm_weight.float()).to(sources[0].dtype)
        logits = torch.stack(
            [
                torch.einsum("sbd,d->sb", v, qg).float()
                * torch.rsqrt(v.float().pow(2).mean(-1) + self.eps)
                for v in sources
            ]
        )
        w = torch.softmax(logits, dim=0)
        entropy = -(w * w.clamp(min=1e-12).log()).sum(0) / math.log(len(sources))
        arch_metrics.record_mean(f"attnres_{kind}/entropy", layer, n, entropy)
        arch_metrics.record_mean(f"attnres_{kind}/current_weight", layer, n, w[-1])

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate prior representations with learned depth-wise attention.

        Args:
            sources: ``k`` tensors of shape ``[s, b, d]`` -- the completed block
                representations followed by the current partial sum.

        Returns:
            The attended hidden state, ``[s, b, d]``.
        """
        from megatron.core.transformer import arch_metrics

        if arch_metrics.active():
            self._record_arch_metrics(sources)
        # One fused, compiled aggregation instead of ~2k per-source kernels.
        # Stacking is cheap relative to the launch overhead it removes, and
        # Inductor fuses the norm/softmax/weighted-sum chain into a few kernels.
        return _attn_res_loop_compiled(
            self.query.float(), self.key_norm_weight.float(), self.eps, *sources
        )


def accumulate_block(
    state: BlockAttnResState, sublayer_out: torch.Tensor, lerp: nn.Module | None = None
) -> None:
    """Add a sublayer output to the in-progress block representation.

    Implements the intra-block accumulation of paper Eq. 5,
    :math:`b_n = \\sum_{j \\in B_n} f_j(h_j)`. With ``lerp`` supplied the plain sum
    becomes an nGPT LERP step toward the normalized sublayer output, keeping the
    block summary on the unit sphere rather than growing with block size.

    Args:
        state: The depth-wise state to update in place.
        sublayer_out: ``[s, b, d]`` output of the attention or MLP sublayer.
        lerp: Optional nGPT residual module replacing the additive accumulation.
    """
    if state.partial is None:
        # Seed the block. Under the LERP variant the running summary is a point on
        # the unit sphere, so the first contribution must be normalized too --
        # the LERP's closed form assumes a unit-norm base point.
        state.partial = (
            sublayer_out
            if lerp is None
            else F.normalize(sublayer_out.float(), p=2, dim=-1, eps=1e-6).to(sublayer_out.dtype)
        )
    elif lerp is None:
        state.partial = state.partial + sublayer_out
    else:
        state.partial = lerp(sublayer_out, state.partial)


def block_boundary_reached(layer_number: int, layers_per_block: int) -> bool:
    """Return True if ``layer_number`` closes a block.

    Args:
        layer_number: 1-based transformer layer index.
        layers_per_block: Transformer layers per block (``S/2`` in paper terms,
            since each transformer layer contributes an attention and an MLP
            sublayer).
    """
    if layers_per_block <= 0:
        return False
    return layer_number % layers_per_block == 0


def resolve_layers_per_block(num_layers: int, num_blocks: int) -> int:
    """Compute transformer layers per block from a requested block count.

    Args:
        num_layers: Total transformer layers ``L``.
        num_blocks: Requested number of blocks ``N``.

    Returns:
        Layers per block, at least 1. When ``L`` is not divisible by ``N`` the
        final block absorbs the remainder, matching the paper.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    return max(1, num_layers // num_blocks)


# --------------------------------------------------------------------------------------------
# Stream form (attn_res_impl "streams"): CUDA-graph capturable
# --------------------------------------------------------------------------------------------
# The sources travel as one [s, b, slots, h] tensor through the mHC plumbing: slot 0 is the
# embedding, slot 1 + j the j-th block. The block being filled always sits in the next free slot,
# so sealing it moves nothing -- the next block just starts one slot further. Every sublayer reads
# a contiguous slice of slots (its source count is static), and its write replaces one slot. The
# HybridStack wires each mixer's slice / slot / seed from the sublayer order and the block size.


def attn_res_streams(config) -> bool:
    """True when Block AttnRes runs in the CUDA-graph-capturable stream form."""
    return bool(getattr(config, "attn_res", False)) and getattr(
        config, "attn_res_impl", "state"
    ) == "streams"


def _attn_res_aggregate_streams(values, query, gain, eps):
    """Softmax over depth of ``values`` [..., k, h] (the stream slots): [..., h]."""
    qg = (query * gain).to(values.dtype)
    dot = torch.einsum("...kd,d->...k", values, qg).float()
    inv = torch.rsqrt(values.float().pow(2).mean(dim=-1) + eps)
    weights = torch.softmax(dot * inv, dim=-1).to(values.dtype)
    return torch.einsum("...k,...kd->...d", weights, values)


_attn_res_streams_compiled = _maybe_compile(_attn_res_aggregate_streams)


class AttnResStreamMixer(nn.Module):
    """Block AttnRes around one sublayer, mHC mixer API (stream form; see above)."""

    def __init__(self, config):
        super().__init__()
        self.attnres = BlockAttnRes(
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            query_init_std=config.attn_res_query_init_std,
        )
        self.live = self.slot = None
        self.seed = True

    def wire(self, live: int, slot: int, seed: bool) -> None:
        """``live`` sources before the read (slots ``0 .. live-1``), the block slot this write
        goes to, and whether the block starts here (``seed``: no partial sum yet)."""
        self.live, self.slot, self.seed = live, slot, seed

    def forward(self, streams: torch.Tensor):
        return None, None, None

    def read(self, streams: torch.Tensor, h_pre=None) -> torch.Tensor:
        if self.live is None:
            raise RuntimeError("attn_res streams: the mixer was not wired by its HybridStack")
        from megatron.core.transformer import arch_metrics

        sources = streams[..., : self.live, :]
        if arch_metrics.active():
            self.attnres._record_arch_metrics(list(sources.unbind(-2)))
        return _attn_res_streams_compiled(
            sources, self.attnres.query.float(), self.attnres.key_norm_weight.float(),
            self.attnres.eps,
        )

    def write(self, streams, sublayer_out, h_post=None, h_res=None, lerp=None) -> torch.Tensor:
        if lerp is not None:
            raise NotImplementedError("attn_res streams with the nGPT LERP residual")
        partial = sublayer_out if self.seed else streams[..., self.slot, :] + sublayer_out
        return torch.cat(
            (streams[..., : self.slot, :], partial.unsqueeze(-2), streams[..., self.slot + 1 :, :]),
            dim=-2,
        )

    def record_metrics(self, streams, coeffs) -> None:
        pass


class AttnResStreamBoundary:
    """Enter (the embedding in slot 0) and exit (the last written slot) of the stream form."""

    def __init__(self, slots: int, last_slot: int):
        self.slots, self.last_slot = slots, last_slot

    def expand(self, embedding: torch.Tensor) -> torch.Tensor:
        empty = embedding.new_zeros(*embedding.shape[:-1], self.slots - 1, embedding.shape[-1])
        return torch.cat((embedding.unsqueeze(-2), empty), dim=-2)

    def collapse(self, streams: torch.Tensor) -> torch.Tensor:
        return streams[..., self.last_slot, :]
