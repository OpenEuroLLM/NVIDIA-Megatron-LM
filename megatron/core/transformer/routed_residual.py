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

Unselected streams are untouched and the selected ones are updated in place
(fused_routed_residual.py), so a sublayer moves ``(2k + 1) C`` elements. As in an
MoE gate, the router learns through the read weights.

Mixing (``mhc_routed_mix``). Optionally the routed streams are also mixed, as mHC
does over all streams: ``s_j <- sum_i M_ji s_i + w_j f(x)`` over the selection,
with ``M`` a per-token ``[k, k]`` doubly stochastic matrix (Sinkhorn) from
``a_res phi_res rms(s_0)`` restricted to the selected stream pairs plus a static
``[n, n]`` bias, initialized near the identity (as mHC's ``res_init="identity"``).
The backward needs the pre-write rows, which the read keeps anyway: mixing adds
only the ``[k, k]`` matrices to the activation memory.

Balancing (``mhc_routed_balance_rate``). Like the aux-loss-free MoE router
(DeepSeek-V3), a per-stream bias added to the routing logits for the selection
only (not the read weights) moves after every step toward the under-picked
streams, by ``rate * sign(mean count - count)`` over the global batch. The streams start as copies of the embedding, so at init the read is
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
    routed_read_reversible,
    routed_write_mix,
    routed_write_reversible,
    router_coeffs_reversible,
    stream_tap,
)


class RoutedStreamMixer(nn.Module):
    """Routes one sublayer to ``topk`` of ``expansion_rate`` residual streams."""

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int,
        topk: int,
        alpha_init: float = 0.01,
        eps: float = 1e-6,
        mix: bool = False,
        sinkhorn_iters: int = 8,
        identity_bias: float = 5.0,
        balance: bool = False,
        reversible: bool = False,
    ) -> None:
        super().__init__()
        if reversible and mix:
            raise ValueError("mhc_routed_reversible does not combine with mhc_routed_mix")
        self.reversible = reversible
        if not 1 <= topk <= expansion_rate:
            raise ValueError(f"mhc_routed_topk must be in [1, {expansion_rate}], got {topk}")
        self.hidden_size, self.n, self.k, self.eps = hidden_size, expansion_rate, topk, eps
        self.mix, self.sinkhorn_iters, self.identity_bias = mix and topk > 1, sinkhorn_iters, identity_bias
        n = expansion_rate
        self.phi = nn.Parameter(torch.empty(hidden_size, 2 * n + (n * n if self.mix else 0)))
        self.b_route = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        scalers = [self.b_route, self.b_post, self.alpha_post]
        if self.mix:
            self.b_res = nn.Parameter(torch.zeros(n, n))
            self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))
            scalers += [self.b_res, self.alpha_res]
        # A router (as an MoE router): Adam, also under the Muon-family optimizers.
        self.phi.use_adam_with_emerging_optimizers = True
        for p in scalers:
            p.is_ngpt_scaler = True
        self.balance = balance
        if balance:
            # fp32 like the MoE expert bias (kept so by _apply); updated by
            # finalize_model_grads from the step's pick counts.
            self.register_buffer("route_bias", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(
                "local_picks_per_stream", torch.zeros(n, dtype=torch.float32), persistent=False
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.phi.zero_()
            self.phi[:, : self.n].normal_(0.0, self.hidden_size**-0.5)
            self.b_route.zero_()
            self.b_post.zero_()
            if self.mix:
                self.b_res.copy_(torch.eye(self.n) * self.identity_bias)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if self.balance:  # keep the balancing state fp32 under Float16Module
            self.route_bias.data = self.route_bias.data.float()
            self.local_picks_per_stream.data = self.local_picks_per_stream.data.float()
        return self

    def forward(self, streams: torch.Tensor):
        """``streams`` ``[s, b, n, C]`` -> ``(pre, post, None)`` routing state."""
        stash = GradStash()
        if self._reversible_now(streams):
            coeffs = router_coeffs_reversible(streams, self.phi, self._coeffs, stash)
        else:
            coeffs = self._coeffs(stream_tap(streams, 0, stash), self.phi)
        z, post = coeffs[:, : self.n], coeffs[:, self.n : 2 * self.n]
        select_bias = self.route_bias if self.balance else None
        idx, r = route_streams(z + self.b_route.float(), self.k, select_bias=select_bias)
        if self.balance and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                picks = idx[:, 1:].long()
                self.local_picks_per_stream.scatter_add_(
                    0, picks.reshape(-1), torch.ones(picks.numel(), device=picks.device)
                )
        w = 2.0 * torch.sigmoid(self.alpha_post.float() * post + self.b_post.float())
        w = torch.gather(w, 1, idx.long())
        mix = None
        if self.mix:
            from megatron.core.transformer.mhc import sinkhorn_knopp

            res = coeffs[:, 2 * self.n :].view(-1, self.n, self.n)
            rows = idx.long().unsqueeze(-1).expand(-1, -1, self.n)
            res = torch.gather(res, 1, rows)  # [T, k, n]
            res = torch.gather(res, 2, idx.long().unsqueeze(1).expand(-1, self.k, -1))
            static = self.b_res.float()[idx.long().unsqueeze(-1), idx.long().unsqueeze(1)]
            mix = sinkhorn_knopp(self.alpha_res.float() * res + static, self.sinkhorn_iters)
        return (idx, r, stash), (idx, w, mix, stash), None


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
        (idx, r, _), (_, w, mix, _), _ = coeffs
        p = f"routed_{kind}"
        if self.k > 1 and self.n > 2:
            picks = idx[:, 1:].long() - 1  # non-main streams, numbered 0..n-2
            onehot = torch.zeros(idx.shape[0], self.n - 1, device=idx.device)
            onehot.scatter_(1, picks, 1.0 / picks.shape[1])
            arch_metrics.record_share(f"{p}/pick_share", layer, num_layers, onehot)
        arch_metrics.record_mean(f"{p}/main_read", layer, num_layers, r[:, 0])
        arch_metrics.record_mean(f"{p}/write", layer, num_layers, w)
        if mix is not None:
            arch_metrics.record_mean(
                f"{p}/mix_diag", layer, num_layers, torch.diagonal(mix, dim1=-2, dim2=-1)
            )

    def _coeffs(self, s0: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """The router's projection of the RMS-normalized main stream, ``[T, *]`` fp32."""
        s0 = s0.reshape(-1, self.hidden_size)
        rms = torch.linalg.vector_norm(s0, dim=-1, keepdim=True).float()
        rms = (rms / self.hidden_size**0.5).clamp(min=self.eps)
        return torch.matmul(s0, phi.to(s0.dtype)).float() / rms

    def _reversible_now(self, streams: torch.Tensor) -> bool:
        from megatron.core.fusions.fused_routed_residual import _fused

        return self.reversible and _fused(streams) and torch.is_grad_enabled()

    def read(self, streams: torch.Tensor, h_pre) -> torch.Tensor:
        idx, r, stash = h_pre
        if self._reversible_now(streams):
            return routed_read_reversible(streams, idx, r, stash)
        return routed_read(streams, idx, r, stash)

    def write(self, streams, sublayer_out, h_post, h_res=None, lerp=None) -> torch.Tensor:
        if lerp is not None:
            raise NotImplementedError("mhc_lerp is not supported with mhc_routed_topk")
        idx, w, mix, stash = h_post
        f = sublayer_out.to(streams.dtype)
        if mix is not None:
            return routed_write_mix(streams, f, idx, w, mix, stash)
        if self._reversible_now(streams):
            return routed_write_reversible(streams, f, idx, w, stash)
        return routed_write(streams, f, idx, w, stash)


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
            mix=getattr(config, "mhc_routed_mix", False),
            sinkhorn_iters=config.mhc_sinkhorn_iters,
            balance=bool(getattr(config, "mhc_routed_balance_rate", 0.0)),
            reversible=getattr(config, "mhc_routed_reversible", False),
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


@torch.no_grad()
def update_routed_residual_bias(model_chunks, config, tp_dp_cp_group=None) -> None:
    """End-of-step aux-loss-free balancing of every routed-residual mixer: move each
    stream's selection bias by ``rate * sign(mean - picks)`` over the global batch."""
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias

    rate = getattr(config, "mhc_routed_balance_rate", 0.0)
    mixers = [
        m
        for chunk in model_chunks
        for m in chunk.modules()
        if isinstance(m, RoutedStreamMixer) and m.balance and m.training
    ]
    if not rate or not mixers:
        return
    # The main stream is always picked: balance the others.
    counts = torch.stack([m.local_picks_per_stream[1:] for m in mixers])
    biases = torch.stack([m.route_bias[1:] for m in mixers])
    updated = get_updated_expert_bias(counts, biases, rate, tp_dp_cp_group=tp_dp_cp_group)
    for m, b in zip(mixers, updated):
        m.route_bias[1:].copy_(b)
        m.local_picks_per_stream.zero_()


def own_streams(streams: torch.Tensor, config) -> torch.Tensor:
    """Routed streams are updated in place: take ownership of a received payload
    (a pipeline stage's input is a leaf, and a reshape of it a view)."""
    if getattr(config, "mhc_routed_topk", 0):
        return streams.clone(memory_format=torch.contiguous_format)
    return streams
