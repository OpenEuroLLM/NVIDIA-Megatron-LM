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


# --------------------------------------------------------------------------------------------
# Stream form (depth_memory_impl "torch" / "cute"): CUDA-graph capturable
# --------------------------------------------------------------------------------------------
# The residual travels as [s, b, 2, h] "streams" through the mHC plumbing: slot 0 is the stream
# x, slot 1 the next sublayer's memory read r. The state (M, P) lives in a static workspace
# (depth_memory_state.py) that every step updates in place; each write site runs the fused step
# of the paper's kernels -- write its output, then read with the NEXT site's query (taken from
# the updated stream, as the next read's query source is). So a sublayer reads x + r from its
# input streams, and its write hands the next sublayer (x', r'), with nothing on the module that
# a CUDA graph would bake in.


def ldm_streams(config) -> bool:
    """True when Linear Depth Memory runs in the CUDA-graph-capturable stream form."""
    return bool(getattr(config, "depth_memory", False)) and getattr(
        config, "depth_memory_impl", "reference"
    ) in ("torch", "cute")


class DepthMemoryMixer(nn.Module):
    """The read before and the fused write (+ next read) after one sublayer, mHC mixer API.

    The mixer owns its write (key, weight, forgetting) and the query network of the read that
    FOLLOWS it -- the next sublayer's, or the final read before the LM head for the last mixer --
    because the read is fused into its write step: a CUDA graph of this layer then touches only
    its own parameters. ``edge`` (this site's step) and the workspace are wired by the HybridStack.
    """

    def __init__(self, config):
        super().__init__()
        self.write_site = DepthMemoryWrite(config)  # this sublayer's key / weight + forgetting
        self.next_query = DepthMemoryRead(config)  # the following read's query network + gain
        for p in self.parameters():
            p.is_residual_mixer = True
        self.edge = None
        self._workspace = None  # a plain list: the workspace is not a module

    def wire(self, edge: int, workspace) -> None:
        self.edge, self._workspace = edge, [workspace]

    # -- mHC mixer API (TransformerLayer._mhc_read / _mhc_write, ResidualSchemeAdapter) --------
    def forward(self, streams: torch.Tensor):
        return None, None, None

    def read(self, streams: torch.Tensor, h_pre=None) -> torch.Tensor:
        return streams[..., 0, :] + streams[..., 1, :]

    def write(self, streams, sublayer_out, h_post=None, h_res=None, lerp=None) -> torch.Tensor:
        if lerp is not None:
            raise NotImplementedError("depth_memory with the nGPT LERP residual")
        if self.edge is None:
            raise RuntimeError("depth_memory: the mixer was not wired by its HybridStack")
        stream = streams[..., 0, :] + sublayer_out
        read = ldm_fused_step(
            self.write_site, self.next_query, self._workspace[0], self.edge, stream, sublayer_out
        )
        return torch.stack((stream, read), dim=-2)

    def record_metrics(self, streams, coeffs) -> None:
        pass


class DepthMemoryPassThrough(nn.Module):
    """A sublayer outside ``depth_memory_sites``: the plain residual on the stream x, the pending
    read r carried to the next site unchanged (that site reads x + r, its query taken at the
    previous site's write). No parameters, no step."""

    def forward(self, streams: torch.Tensor):
        return None, None, None

    def read(self, streams: torch.Tensor, h_pre=None) -> torch.Tensor:
        return streams[..., 0, :]

    def write(self, streams, sublayer_out, h_post=None, h_res=None, lerp=None) -> torch.Tensor:
        if lerp is not None:
            raise NotImplementedError("depth_memory with the nGPT LERP residual")
        return torch.stack((streams[..., 0, :] + sublayer_out, streams[..., 1, :]), dim=-2)

    def record_metrics(self, streams, coeffs) -> None:
        pass


def _site_addresses(stream, eps: float, groups: int, key_dim: int, w_norm, w1, b1, w_addr,
                    b_addr, w_gate, b_gate, logit, q_norm, q1, qb1, q_addr, qb_addr, q_gate,
                    qb_gate, gain):
    """The write's (key, sqrt(weight), forgetting) and the next read's gated query from one
    stream: ONE RMS normalization and ONE first-layer GEMM for both address networks (their
    RMSNorm gains folded into the stacked W1, as the paper's fused address op), then the small
    heads. Compiled: Inductor fuses the norm, SiLU, L2 normalizations and gates around the GEMMs.
    """
    x = stream.float()
    xn = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(w1.dtype)
    weight = torch.cat((w1 * w_norm.to(w1.dtype), q1 * q_norm.to(q1.dtype)), dim=0)
    hidden = F.silu(F.linear(xn, weight, torch.cat((b1, qb1))))
    hw, hq = hidden.split((w1.shape[0], q1.shape[0]), dim=-1)
    shape = (*stream.shape[:-1], groups, key_dim)
    key = F.normalize(F.linear(hw, w_addr, b_addr).float().view(shape), dim=-1, eps=1e-6)
    scale = (2.0 * torch.sigmoid(F.linear(hw, w_gate, b_gate).float())).clamp_min(1e-6).sqrt()
    query = F.normalize(F.linear(hq, q_addr, qb_addr).float().view(shape), dim=-1, eps=1e-6)
    gate = 2.0 * torch.sigmoid(F.linear(hq, q_gate, qb_gate).float())
    query = query * (gate * gain.float()).unsqueeze(-1)
    lam = torch.sigmoid(logit.float()).clamp_min(1e-6)
    return key, scale, query, lam


def _site_addresses_eager(*args):
    return _site_addresses(*args)


_site_addresses_compiled = None


def ldm_fused_step(write, next_read, workspace, edge, stream, value) -> torch.Tensor:
    """Write ``value`` under the key of ``stream``, then read with ``next_read``'s query of
    ``stream``; returns the read as ``[..., h]`` in the stream's dtype."""
    global _site_addresses_compiled
    from megatron.core.transformer.attn_res import _maybe_compile
    from megatron.core.transformer.depth_memory_state import depth_memory_step

    if _site_addresses_compiled is None:
        _site_addresses_compiled = _maybe_compile(_site_addresses_eager)
    shape = stream.shape
    n = stream.numel() // shape[-1]
    workspace.ensure(n, stream.device)
    wn, qn = write.net, next_read.net
    key, scale, query, lam = _site_addresses_compiled(
        stream, wn.eps, wn.groups, wn.key_dim,
        wn.norm_weight, wn.w1.weight, wn.w1.bias, wn.address.weight, wn.address.bias,
        wn.gate.weight, wn.gate.bias, write.forget_logit,
        qn.norm_weight, qn.w1.weight, qn.w1.bias, qn.address.weight, qn.address.bias,
        qn.gate.weight, qn.gate.bias, next_read.read_gain,
    )
    g = key.shape[-2]
    read = depth_memory_step(
        workspace,
        edge,
        key.reshape(n, g, -1),
        query.reshape(n, g, -1),
        value.reshape(n, g, shape[-1] // g),
        scale.reshape(n, g),
        lam,
    )
    return read.reshape(shape).to(stream.dtype)


class DepthMemoryStreamBoundary(nn.Module):
    """Stream form: the embedding write (edge 0) with the first sublayer's query; the final read
    is the last mixer's (its ``next_query``), so ``collapse`` only adds it to the stream."""

    def __init__(self, config):
        super().__init__()
        self.embed_write = DepthMemoryWrite(config)
        self.first_query = DepthMemoryRead(config)
        for p in self.parameters():
            p.is_residual_mixer = True
        self._workspace = None

    def wire(self, workspace) -> None:
        self._workspace = [workspace]

    def expand(self, embedding: torch.Tensor) -> torch.Tensor:
        read = ldm_fused_step(
            self.embed_write, self.first_query, self._workspace[0], 0, embedding, embedding
        )
        return torch.stack((embedding, read), dim=-2)

    def collapse(self, streams: torch.Tensor) -> torch.Tensor:
        return streams[..., 0, :] + streams[..., 1, :]
