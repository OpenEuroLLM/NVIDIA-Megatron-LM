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

Content keys (``mhc_routed_key_dim = d``). The router above sees only stream 0, so it
picks streams by position, not by what they hold. With ``d > 0`` it also reads the
first ``d`` channels of every stream (a tag, ``n d`` elements: a few percent of the
``(2k + 1) C``) and adds an attention-like content score to the routing logits::

    z_j += g <q, k_j> / sqrt(d),   q = phi_q rms(s_0),   k_j = W_k rms(s_j[:d])

``q`` comes from the same projection as the logits (``d`` more columns of ``phi``);
the gain ``g`` starts at 0, so at init the router is the positional one above and
the score is learned in. The model chooses what the tag channels carry.

LERP write (``mhc_lerp`` with ``ngpt``). The write becomes an nGPT LERP step of each
selected stream toward the normalized sublayer output, its step ``alpha`` (the
sublayer's NGPTResidual) scaled per row by the write weight,
``s_j <- Norm(s_j + w_j alpha (a - s_j))`` (fused_routed_residual.py): with
``post_layer_norm`` the output's own scale is normalized away, so the write weight acts
on the step. Every stream stays on the sphere, so the read weights mix vectors of
equal norm, and a stream that is rarely picked keeps its content longer.

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

import os

import torch
from torch import nn

from megatron.core.fusions.fused_routed_residual import (
    GradStash,
    _fused,
    fused_router_coeffs,
    LERP_WRITE_MODES,
    route_streams,
    routed_read,
    routed_write,
    routed_read_reversible,
    routed_write_lerp,
    routed_write_mix,
    routed_write_reversible,
    router_coeffs_reversible,
    stream_tap,
    stream_tap_tags,
)


# MEGATRON_ROUTED_FUSED_ROUTER=0 falls back to the unfused router (tap + torch ops)
_FUSED_ROUTER = os.environ.get("MEGATRON_ROUTED_FUSED_ROUTER", "1") != "0"


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
        key_dim: int = 0,
        free: bool = False,
    ) -> None:
        super().__init__()
        if free and reversible:
            raise ValueError("mhc_routed_free does not combine with mhc_routed_reversible yet")
        self.free = free
        if reversible and mix:
            raise ValueError("mhc_routed_reversible does not combine with mhc_routed_mix")
        if not 0 <= key_dim <= hidden_size:
            raise ValueError(f"mhc_routed_key_dim must be in [0, {hidden_size}], got {key_dim}")
        self.reversible = reversible
        if not 1 <= topk <= expansion_rate:
            raise ValueError(f"mhc_routed_topk must be in [1, {expansion_rate}], got {topk}")
        self.hidden_size, self.n, self.k, self.eps = hidden_size, expansion_rate, topk, eps
        self.mix, self.sinkhorn_iters, self.identity_bias = mix and topk > 1, sinkhorn_iters, identity_bias
        n = expansion_rate
        self.key_dim = key_dim
        # Columns of phi: routing logits [n], write gates [n], mixing [n * n] (mix), query
        # [key_dim].
        self._q0 = 2 * n + (n * n if self.mix else 0)
        self.phi = nn.Parameter(torch.empty(hidden_size, self._q0 + key_dim))
        self.b_route = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        scalers = [self.b_route, self.b_post, self.alpha_post]
        if self.mix:
            self.b_res = nn.Parameter(torch.zeros(n, n))
            self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))
            scalers += [self.b_res, self.alpha_res]
        if key_dim:
            self.w_key = nn.Parameter(torch.empty(key_dim, key_dim))
            self.key_gain = nn.Parameter(torch.zeros(1))
            self.w_key.use_adam_with_emerging_optimizers = True
            scalers.append(self.key_gain)
        # A router (as an MoE router): Adam, also under the Muon-family optimizers.
        self.phi.use_adam_with_emerging_optimizers = True
        for p in scalers:
            p.is_ngpt_scaler = True
        # every parameter of the residual mixer: its own LR group (residual_mixer_lr_mult)
        for p in self.parameters(recurse=False):
            p.is_residual_mixer = True
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
            if self.key_dim:
                # q and k ~ N(0, 1) per entry, so the score <q, k> / sqrt(d) is ~ N(0, 1).
                self.phi[:, self._q0 :].normal_(0.0, self.hidden_size**-0.5)
                self.w_key.normal_(0.0, self.key_dim**-0.5)
                self.key_gain.zero_()
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
        params = (self.phi, self.w_key, self.key_gain) if self.key_dim else (self.phi,)
        # the fused router's backward reads stream 0 from the read's saved rows: it needs
        # stream 0 picked (not so in free mode)
        if not self.key_dim and _FUSED_ROUTER and _fused(streams) and not self.free:
            # one pass over stream 0 forward and backward, no tap copy (its backward reads
            # stream 0 from the restored state or the read's saved rows)
            coeffs = fused_router_coeffs(
                streams, self.phi, self.eps, stash, reversible=self._reversible_now(streams)
            )
        elif self._reversible_now(streams):
            coeffs = router_coeffs_reversible(
                streams, params, self._router, stash, tag_dim=self.key_dim
            )
        elif self.key_dim:
            s0, tags = stream_tap_tags(streams, 0, self.key_dim, stash)
            coeffs = self._router(s0, tags, *params)
        else:
            coeffs = self._router(stream_tap(streams, 0, stash), None, *params)
        z, post = coeffs[:, : self.n], coeffs[:, self.n : 2 * self.n]
        select_bias = self.route_bias if self.balance else None
        logits = z + self.b_route.float()
        idx, r = route_streams(
            logits, self.k, main=None if self.free else 0, select_bias=select_bias
        )
        if self.balance and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                picks = idx.long() if self.free else idx[:, 1:].long()
                self.local_picks_per_stream.scatter_add_(
                    0, picks.reshape(-1), torch.ones(picks.numel(), device=picks.device)
                )
        w = 2.0 * torch.sigmoid(self.alpha_post.float() * post + self.b_post.float())
        w = torch.gather(w, 1, idx.long())
        if self.free:
            # straight-through gate (value 1): the selection learns from the write. In log
            # space: a stream picked through the balancing bias can have a softmax probability
            # that underflows to 0, and p / sg(p) would be 0 / 0
            logp = torch.gather(torch.log_softmax(logits, dim=-1), 1, idx.long())
            w = w * torch.exp(logp - logp.detach())
        mix = None
        if self.mix:
            from megatron.core.transformer.mhc import sinkhorn_knopp

            res = coeffs[:, 2 * self.n :].view(-1, self.n, self.n)
            rows = idx.long().unsqueeze(-1).expand(-1, -1, self.n)
            res = torch.gather(res, 1, rows)  # [T, k, n]
            res = torch.gather(res, 2, idx.long().unsqueeze(1).expand(-1, self.k, -1))
            # b_res[idx_i, idx_j] as a one-hot product: exact, and its backward is a small
            # reduction -- advanced indexing's accumulating backward serializes on the n * n
            # entries (~70 ms a step at T = 32k).
            onehot = torch.nn.functional.one_hot(idx.long(), self.n).float()  # [T, k, n]
            static = onehot @ self.b_res.float() @ onehot.mT
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
        if self.free:
            picks = idx.long()  # every stream can be picked
            onehot = torch.zeros(idx.shape[0], self.n, device=idx.device)
            onehot.scatter_(1, picks, 1.0 / picks.shape[1])
            arch_metrics.record_share(f"{p}/pick_share", layer, num_layers, onehot)
        elif self.k > 1 and self.n > 2:
            picks = idx[:, 1:].long() - 1  # non-main streams, numbered 0..n-2
            onehot = torch.zeros(idx.shape[0], self.n - 1, device=idx.device)
            onehot.scatter_(1, picks, 1.0 / picks.shape[1])
            arch_metrics.record_share(f"{p}/pick_share", layer, num_layers, onehot)
        arch_metrics.record_mean(f"{p}/main_read", layer, num_layers, r[:, 0])
        arch_metrics.record_mean(f"{p}/write", layer, num_layers, w)
        arch_metrics.record_streams(f"{p}/stream_rms", layer, num_layers, streams)
        if self.key_dim:
            arch_metrics.record_mean(f"{p}/key_gain", layer, num_layers, self.key_gain)
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

    def _router(self, s0, tags, phi, w_key=None, key_gain=None) -> torch.Tensor:
        """All coefficients ``[T, 2n (+ n * n)]`` fp32: :meth:`_coeffs`, with the content
        score added to the routing logits when ``tags`` (``[..., n, key_dim]``) are given."""
        coeffs = self._coeffs(s0, phi)
        if tags is None:
            return coeffs
        d = self.key_dim
        q = coeffs[:, self._q0 :]  # [T, d], from the normalized s_0
        tags = tags.reshape(-1, self.n, d)
        rms = torch.linalg.vector_norm(tags, dim=-1, keepdim=True).float()
        rms = (rms / d**0.5).clamp(min=self.eps)
        keys = torch.matmul(tags, w_key.to(tags.dtype)).float() / rms  # [T, n, d]
        score = (keys * q.unsqueeze(1)).sum(-1) * d**-0.5
        z = coeffs[:, : self.n] + key_gain.float() * score
        return torch.cat([z, coeffs[:, self.n : self._q0]], dim=1)

    def _reversible_now(self, streams: torch.Tensor) -> bool:
        from megatron.core.fusions.fused_routed_residual import _fused

        return self.reversible and _fused(streams) and torch.is_grad_enabled()

    def read(self, streams: torch.Tensor, h_pre) -> torch.Tensor:
        idx, r, stash = h_pre
        if self._reversible_now(streams):
            return routed_read_reversible(streams, idx, r, stash)
        return routed_read(streams, idx, r, stash)

    def write(self, streams, sublayer_out, h_post, h_res=None, lerp=None) -> torch.Tensor:
        idx, w, mix, stash = h_post
        f = sublayer_out.to(streams.dtype)
        if lerp is not None:
            return self._write_lerp(streams, f, idx, w, mix, stash, lerp)
        if mix is not None:
            return routed_write_mix(streams, f, idx, w, mix, stash)
        if self._reversible_now(streams):
            return routed_write_reversible(streams, f, idx, w, stash)
        return routed_write(streams, f, idx, w, stash)

    def _write_lerp(self, streams, f, idx, w, mix, stash, lerp) -> torch.Tensor:
        """The nGPT LERP write: each routed stream steps toward the normalized ``f`` by
        ``w_j alpha`` (``lerp``: the sublayer's NGPTResidual)."""
        if mix is not None:
            raise NotImplementedError("mhc_lerp is not supported with mhc_routed_mix yet")
        mode = lerp.lerp_normalization
        if mode not in LERP_WRITE_MODES:
            raise NotImplementedError(
                f"mhc_lerp with mhc_routed_topk supports lerp_normalization in "
                f"{LERP_WRITE_MODES}, got {mode!r}"
            )
        a = f
        if lerp.post_layer_norm:
            a = f / torch.linalg.vector_norm(f, dim=-1, keepdim=True).clamp(min=lerp.eps)
        return routed_write_lerp(
            streams, a, idx, w, lerp.effective_alpha(), mode, lerp.eps, stash,
            reversible=self._reversible_now(streams),
        )


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
            key_dim=getattr(config, "mhc_routed_key_dim", 0),
            free=getattr(config, "mhc_routed_free", False),
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
    # The main stream is always picked: balance the others (free mode: all streams). One
    # stacked update per mode (the stack needs equal widths).
    for free in (False, True):
        group = [m for m in mixers if m.free == free]
        if not group:
            continue
        s0 = 0 if free else 1
        counts = torch.stack([m.local_picks_per_stream[s0:] for m in group])
        biases = torch.stack([m.route_bias[s0:] for m in group])
        updated = get_updated_expert_bias(counts, biases, rate, tp_dp_cp_group=tp_dp_cp_group)
        for m, b in zip(group, updated):
            m.route_bias[s0:].copy_(b)
            m.local_picks_per_stream.zero_()


def own_streams(streams: torch.Tensor, config) -> torch.Tensor:
    """Routed streams are updated in place: take ownership of a received payload
    (a pipeline stage's input is a leaf, and a reshape of it a view)."""
    if getattr(config, "mhc_routed_topk", 0):
        return streams.clone(memory_format=torch.contiguous_format)
    return streams
