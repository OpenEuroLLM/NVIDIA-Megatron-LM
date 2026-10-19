# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

r"""Linear Depth Memory (LDM), González-Martínez, Nazari, Rusch, Liu: "Depth Memory"
(`arXiv:2610.18809 <https://arxiv.org/abs/2610.18809>`_; reference code
github.com/davidgonmar/depth-memory, ``olmo_core/nn/ldm/backends/reference.py``).

Next to the plain residual stream :math:`x`, every token carries a matrix memory
:math:`M \in \mathbb{R}^{G \times K \times D_v}` (``G`` groups, ``D_v = h / G``) and the
inverse key covariances :math:`P \in \mathbb{R}^{G \times K \times K}` of a recursive
least-squares fit of the layer outputs over depth (paper Eq. 6-8). Around every sublayer
:math:`F_l` (attention / mixer / MLP / MoE alike):

.. math::
    \text{read:}\quad r = \gamma \odot M (g \odot q), \qquad \delta_l = F_l(\mathrm{Norm}(x + r))

    \text{write:}\quad x \leftarrow x + \delta_l, \qquad
    \kappa = P\bar k / (\lambda + \bar k^\top P \bar k), \quad u = \bar v - M^\top \bar k,

    M \leftarrow M + \kappa u^\top, \qquad P \leftarrow (P - \kappa \bar k^\top P) / \lambda

with :math:`\bar k = \sqrt{\omega} k`, :math:`\bar v = \sqrt{\omega}\,\delta_l`; the query
comes from the incoming stream, the key from the updated one, both through a small SiLU
address network (RMSNorm, ``h -> H``, SiLU, heads for the L2-normalized address and the
identity-centred gate ``2 sigmoid``). ``lambda`` is a learned per-group forgetting factor per
write (initially 0.99), ``gamma`` a per-group read gain (initially 1). The embedding is
written first (``v = e``), and a final read is added to the stream before the LM head.

The state (``M``, ``P``) is fp32 and per token -- document masking needs nothing. This is
the reference formulation: autograd keeps every write's state (``K x h`` fp32 per token per
sublayer); the paper's kernels unwind the state in the backward instead. The state lives on
the HybridStack's :class:`DepthMemoryState`, stashed on every layer like Block AttnRes', so
the layers are not CUDA-graph capturable and the stack does not split over pipeline stages.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class _AddressNetwork(nn.Module):
    """RMSNorm -> Linear(h, H) -> SiLU -> (address head [G * K], gate head [G])."""

    def __init__(self, hidden_size: int, hidden_dim: int, groups: int, key_dim: int, eps: float,
                 init_std: float):
        super().__init__()
        self.groups, self.key_dim, self.eps = groups, key_dim, eps
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.w1 = nn.Linear(hidden_size, hidden_dim)
        self.address = nn.Linear(hidden_dim, groups * key_dim)
        self.gate = nn.Linear(hidden_dim, groups)
        for linear in (self.w1, self.address):
            nn.init.trunc_normal_(linear.weight, std=init_std, a=-3 * init_std, b=3 * init_std)
            nn.init.zeros_(linear.bias)
        nn.init.zeros_(self.gate.weight)  # gate = 2 sigmoid(0) = 1 at init
        nn.init.zeros_(self.gate.bias)

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[..., h]`` -> (L2-normalized address ``[..., G, K]`` fp32, gate ``[..., G]`` fp32)."""
        x = F.rms_norm(source.float(), (source.shape[-1],), self.norm_weight.float(), self.eps)
        w1 = self.w1.weight
        hidden = F.silu(F.linear(x.to(w1.dtype), w1, self.w1.bias))
        address = self.address(hidden).float().view(*source.shape[:-1], self.groups, self.key_dim)
        address = F.normalize(address, dim=-1, eps=1e-6)
        gate = 2.0 * torch.sigmoid(self.gate(hidden).float())
        return address, gate


class DepthMemoryRead(nn.Module):
    """One read: the gated, gain-scaled query ``gamma * g * q`` and ``x + M q~``."""

    def __init__(self, config):
        super().__init__()
        self.net = _build_network(config)
        self.read_gain = nn.Parameter(torch.ones(config.depth_memory_groups))

    def forward(self, state: "DepthMemoryState", stream: torch.Tensor) -> torch.Tensor:
        query, gate = self.net(stream)
        query = query * (gate * self.read_gain.float()).unsqueeze(-1)
        read = torch.einsum("...gkd,...gk->...gd", state.memory, query)
        from megatron.core.transformer import arch_metrics

        if arch_metrics.active():
            self._record(stream, read, gate)
        return stream + read.reshape(stream.shape).to(stream.dtype)

    def _record(self, stream, read, gate):
        from megatron.core.transformer import arch_metrics

        tag = arch_metrics.tag_of(self)
        if tag is None:
            return
        kind, layer, n = tag
        ratio = read.float().square().sum(dim=(-2, -1)).sqrt() / (
            stream.float().square().sum(-1).sqrt() + 1e-6
        )
        arch_metrics.record_mean(f"ldm_{kind}/read_ratio", layer, n, ratio)
        arch_metrics.record_mean(f"ldm_{kind}/read_gate", layer, n, gate)
        arch_metrics.record_mean(f"ldm_{kind}/read_gain", layer, n, self.read_gain.abs())


class DepthMemoryWrite(nn.Module):
    """One RLS write of ``value`` under the key of ``stream`` (paper Eq. 7-8)."""

    def __init__(self, config):
        super().__init__()
        self.net = _build_network(config)
        self.forget_logit = nn.Parameter(
            torch.full((config.depth_memory_groups,), _logit(config.depth_memory_lambda_init))
        )

    def forward(self, state: "DepthMemoryState", stream: torch.Tensor, value: torch.Tensor):
        key, weight = self.net(stream)
        groups = key.shape[-2]
        value = value.float().reshape(*value.shape[:-1], groups, value.shape[-1] // groups)
        scale = weight.clamp_min(1e-6).sqrt().unsqueeze(-1)
        key, value = key * scale, value * scale  # sqrt(omega) k, sqrt(omega) v
        lam = torch.sigmoid(self.forget_logit.float()).clamp_min(1e-6)
        state.memory, state.covariance = rls_write(
            state.memory, state.covariance, key, value, lam
        )
        from megatron.core.transformer import arch_metrics

        if arch_metrics.active():
            tag = arch_metrics.tag_of(self)
            if tag is not None:
                kind, layer, n = tag
                arch_metrics.record_mean(f"ldm_{kind}/write_weight", layer, n, weight)
                arch_metrics.record_mean(f"ldm_{kind}/forget", layer, n, lam)


def rls_write(memory, covariance, key, value, lam):
    """One recursive least-squares step (paper Eq. 7-8), per token and group.

    ``memory [..., G, K, Dv]``, ``covariance [..., G, K, K]``, ``key [..., G, K]``,
    ``value [..., G, Dv]``, ``lam [G]`` (fp32). Returns the new (memory, covariance).
    """
    pk = torch.einsum("...ij,...j->...i", covariance, key)
    kp = torch.einsum("...i,...ij->...j", key, covariance)
    gain = pk / (lam + (key * pk).sum(-1)).unsqueeze(-1)  # kappa
    covariance = (covariance - gain.unsqueeze(-1) * kp.unsqueeze(-2)) / lam[:, None, None]
    error = value - torch.einsum("...kd,...k->...d", memory, key)  # u = v - M'k
    memory = memory + gain.unsqueeze(-1) * error.unsqueeze(-2)
    return memory, covariance


def _build_network(config) -> _AddressNetwork:
    return _AddressNetwork(
        config.hidden_size,
        config.depth_memory_hidden_dim,
        config.depth_memory_groups,
        config.depth_memory_key_dim,
        config.layernorm_epsilon,
        config.init_method_std,
    )


class DepthMemoryState:
    """The per-micro-batch LDM state of a HybridStack: the stream ``x`` and ``(M, P)``."""

    def __init__(self, stream: torch.Tensor, groups: int, key_dim: int, alpha: float):
        prefix = stream.shape[:-1]
        self.stream = stream
        self.memory = stream.new_zeros(
            (*prefix, groups, key_dim, stream.shape[-1] // groups), dtype=torch.float32
        )
        eye = torch.eye(key_dim, device=stream.device, dtype=torch.float32) * alpha
        self.covariance = eye.expand(*prefix, groups, key_dim, key_dim)


class DepthMemorySite(nn.Module):
    """The read before one sublayer and the write after it."""

    def __init__(self, config):
        super().__init__()
        self.read = DepthMemoryRead(config)
        self.write = DepthMemoryWrite(config)
        for p in self.parameters():
            p.is_residual_mixer = True

    def read_input(self, state: DepthMemoryState) -> torch.Tensor:
        """The sublayer input ``x + r`` (the residual stays ``x``)."""
        return self.read(state, state.stream)

    def write_output(self, state: DepthMemoryState, out: torch.Tensor) -> torch.Tensor:
        """``x <- x + out``, then the RLS write of ``out`` keyed by the new stream."""
        stream = state.stream + out
        self.write(state, stream, out)
        state.stream = stream
        return stream


class DepthMemoryBoundary(nn.Module):
    """The embedding write (enter) and the final read before the LM head (exit)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_write = DepthMemoryWrite(config)
        self.final_read = DepthMemoryRead(config)
        for p in self.parameters():
            p.is_residual_mixer = True

    def enter(self, embedding: torch.Tensor) -> DepthMemoryState:
        c = self.config
        state = DepthMemoryState(
            embedding, c.depth_memory_groups, c.depth_memory_key_dim, c.depth_memory_alpha
        )
        self.embed_write(state, embedding, embedding)
        return state

    def exit(self, state: DepthMemoryState) -> torch.Tensor:
        return self.final_read(state, state.stream)


def depth_memory_site(config) -> Optional[DepthMemorySite]:
    """A site for one sublayer when LDM is on."""
    return DepthMemorySite(config) if getattr(config, "depth_memory", False) else None
