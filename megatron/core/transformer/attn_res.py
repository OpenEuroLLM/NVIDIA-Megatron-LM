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

import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate prior representations with learned depth-wise attention.

        Args:
            sources: ``k`` tensors of shape ``[s, b, d]`` -- the completed block
                representations followed by the current partial sum.

        Returns:
            The attended hidden state, ``[s, b, d]``.
        """
        # Deliberately never materialize a [k, s, b, d] stack. Doing so copies
        # k * s * b * d elements on every call, and this runs twice per layer --
        # at k=9, s=4096, b=16, d=576 that is ~680 MB per call. Instead compute
        # each source's logit separately (each only [s, b]) and accumulate the
        # weighted sum in place, so peak extra memory is one [s, b, d] buffer.
        query = self.query.float()
        gain = self.key_norm_weight.float()

        # Keys are RMSNormed values (paper Eq. 3): stops large-magnitude blocks
        # from dominating the weights purely by scale. Rather than materializing
        # the normalized key (a full [s, b, d] tensor that backward would retain
        # for every source), factor the RMS scale out of the dot product:
        #   <q, v / rms(v)> == <q, v> / rms(v)
        # so only [s, b] intermediates are ever built.
        qg = (query * gain).to(sources[0].dtype)
        inv_d = 1.0 / self.hidden_size
        logits = []
        for v in sources:
            dot = torch.einsum("sbd,d->sb", v, qg).float()
            sq_norm = torch.linalg.vector_norm(v, dim=-1).float().pow(2)
            logits.append(dot * torch.rsqrt(sq_norm * inv_d + self.eps))

        # Softmax over depth. The stack here is over [s, b] scalars only.
        weights = torch.softmax(torch.stack(logits, dim=0), dim=0)

        # Accumulate in the *source* dtype, not fp32. `mul` saves both operands
        # for backward, so upcasting here would retain one fp32 [s, b, d] copy
        # per source for the whole backward pass -- at k=9 and 36 calls per
        # forward that is tens of GiB and OOMs a 95 GiB GH200. The weights are
        # computed in fp32 and only cast at use; a convex combination needs no
        # extra precision in the values themselves.
        dtype = sources[0].dtype
        w = weights.to(dtype)
        out = sources[0] * w[0].unsqueeze(-1)
        for i in range(1, len(sources)):
            out = out + sources[i] * w[i].unsqueeze(-1)
        return out


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
