# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Routed residual streams: MoE-like sparsity on a widened residual.

mHC (mhc.py) widens the residual into ``n`` streams and every sublayer reads, mixes
and writes all of them -- ``>= 2n C`` elements of memory traffic per token and
sublayer, which is what bounds mHC's throughput. With ``mhc_routed_topk = k`` a
sublayer instead touches ``k`` streams per token: stream 0 (the main stream,
always) and the top ``k - 1`` of the others, picked by a router on stream 0::

    z = phi_z  rms(s_0) + b_z                   (n routing logits)
    S = {0} u top_{k-1}(z_1..z_{n-1})
    x = sum_{j in S} softmax(z_S)_j s_j         (the sublayer input)
    s_j <- s_j + 2 sigmoid(a_w phi_w rms(s_0) + b_w)_j f(x)   for j in S

There is no stream mixing (``H_res = I``): unselected streams are untouched, and
the selected ones are updated in place (fused_routed_residual.py), so a sublayer
moves ``(2k + 1) C`` elements. As in an MoE gate, the router learns through the
read weights. The streams start as copies of the embedding, so at init the read is
the plain residual; the write starts at 1 (``a_w`` small, ``b_w = 0``) and the
routing logits start random (``phi_z`` ~ N(0, 1/C), so logits ~ N(0, 1)), which
spreads the first selections over the streams.

Same interface as :class:`~megatron.core.transformer.mhc.MHCMixer` (so every mHC
call site -- TransformerLayer sublayers, the residual-scheme adapter of Mamba /
Engram / PKM layers, pipeline payloads -- works unchanged); the per-call routing
state travels in the coefficient tuples.
"""

import torch
from torch import nn

from megatron.core.fusions.fused_routed_residual import (
    GradStash,
    route_streams,
    routed_read,
    routed_write,
    stream_tap,
)


class RoutedStreamMixer(nn.Module):
    """Routes one sublayer to ``topk`` of ``expansion_rate`` residual streams."""

    def __init__(
        self, hidden_size: int, expansion_rate: int, topk: int, alpha_init: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 1 <= topk <= expansion_rate:
            raise ValueError(f"mhc_routed_topk must be in [1, {expansion_rate}], got {topk}")
        self.hidden_size, self.n, self.k, self.eps = hidden_size, expansion_rate, topk, eps
        n = expansion_rate
        self.phi = nn.Parameter(torch.empty(hidden_size, 2 * n))
        self.b_route = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        # A router (as an MoE router): Adam, also under the Muon-family optimizers.
        self.phi.use_adam_with_emerging_optimizers = True
        for p in (self.b_route, self.b_post, self.alpha_post):
            p.is_ngpt_scaler = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.phi.zero_()
            self.phi[:, : self.n].normal_(0.0, self.hidden_size**-0.5)
            self.b_route.zero_()
            self.b_post.zero_()

    def forward(self, streams: torch.Tensor):
        """``streams`` ``[s, b, n, C]`` -> ``(pre, post, None)`` routing state."""
        stash = GradStash()
        s0 = stream_tap(streams, 0, stash).reshape(-1, self.hidden_size)
        rms = torch.linalg.vector_norm(s0, dim=-1, keepdim=True).float()
        rms = (rms / self.hidden_size**0.5).clamp(min=self.eps)
        coeffs = torch.matmul(s0, self.phi.to(s0.dtype)).float() / rms
        z, post = coeffs.split(self.n, dim=-1)
        idx, r = route_streams(z + self.b_route.float(), self.k)
        w = 2.0 * torch.sigmoid(self.alpha_post.float() * post + self.b_post.float())
        w = torch.gather(w, 1, idx.long())
        return (idx, r, stash), (idx, w, stash), None


    @torch.no_grad()
    def record_metrics(self, streams: torch.Tensor, coeffs) -> None:
        """Stream selection of this sublayer (arch_metrics): the share of each
        non-main stream among the routed picks (entropy, and min_rel 0 = a stream
        never picked), the read weight left on the main stream, the write strength."""
        from megatron.core.transformer import arch_metrics

        tag = arch_metrics.tag_of(self)
        if tag is None:
            return
        kind, layer, num_layers = tag
        (idx, r, _), (_, w, _), _ = coeffs
        p = f"routed_{kind}"
        if self.k > 1 and self.n > 2:
            picks = idx[:, 1:].long() - 1  # non-main streams, numbered 0..n-2
            onehot = torch.zeros(idx.shape[0], self.n - 1, device=idx.device)
            onehot.scatter_(1, picks, 1.0 / picks.shape[1])
            arch_metrics.record_share(f"{p}/pick_share", layer, num_layers, onehot)
        arch_metrics.record_mean(f"{p}/main_read", layer, num_layers, r[:, 0])
        arch_metrics.record_mean(f"{p}/write", layer, num_layers, w)

    def read(self, streams: torch.Tensor, h_pre) -> torch.Tensor:
        idx, r, stash = h_pre
        return routed_read(streams, idx, r, stash)

    def write(self, streams, sublayer_out, h_post, h_res=None, lerp=None) -> torch.Tensor:
        if lerp is not None:
            raise NotImplementedError("mhc_lerp is not supported with mhc_routed_topk")
        idx, w, stash = h_post
        return routed_write(streams, sublayer_out.to(streams.dtype), idx, w, stash)


def build_stream_mixer(config) -> nn.Module:
    """The mHC mixer of one sublayer: dense mHC, or routed when ``mhc_routed_topk``."""
    topk = getattr(config, "mhc_routed_topk", 0)
    if topk:
        return RoutedStreamMixer(
            hidden_size=config.hidden_size,
            expansion_rate=config.mhc_expansion_rate,
            topk=topk,
            alpha_init=config.mhc_alpha_init,
            eps=config.layernorm_epsilon,
        )
    from megatron.core.transformer.mhc import MHCMixer

    return MHCMixer(
        hidden_size=config.hidden_size,
        expansion_rate=config.mhc_expansion_rate,
        num_sinkhorn_iters=config.mhc_sinkhorn_iters,
        alpha_init=config.mhc_alpha_init,
        res_init=config.mhc_res_init,
        eps=config.layernorm_epsilon,
    )


def own_streams(streams: torch.Tensor, config) -> torch.Tensor:
    """Routed streams are updated in place: take ownership of a received payload
    (a pipeline stage's input is a leaf, and a reshape of it a view)."""
    if getattr(config, "mhc_routed_topk", 0):
        return streams.clone(memory_format=torch.contiguous_format)
    return streams
