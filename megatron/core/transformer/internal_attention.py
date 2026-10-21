# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Internal attention over the residual streams (Internal Attention Transformer, IAT).

"Internal Attention: Growing Token Representations Without Growing the FFN or the KV Cache"
(ICLR 2027 submission, openreview LGFDpZioyZ). A token is ``x = (x_c, x_m)``: the compute part
``x_c`` (width d) that the FFN and the head see, and a memory part ``x_m`` (width m) that only
the attentions touch. The internal attention splits the token into ``k = (d + m) / c`` chunks
and runs a bidirectional attention among them -- per token, so it is causal by construction and
adds nothing to the KV cache -- with shared ``c x c`` Q / K / V / O maps and a learned position
vector per chunk, as a pre-norm residual over all ``d + m`` dims.

Here the chunks are the mHC streams (``[s, b, n, h]``, ``c = h = d``, ``m = (n - 1) d``, the
paper's best ``c = d``), and the mixer below plugs into the mHC stream-mixer API
(``forward`` / ``read`` / ``write`` / ``record_metrics``, see mhc.MHCMixer), so the widened
residual, the pipeline payload and activation recompute are the mHC ones:

* ``mhc_internal_attention = "compute"``: every sublayer (temporal mixer and MLP / MoE) reads and
  writes stream 0 (``x_c``); the memory streams start at zero and the head reads stream 0. The
  mixers keep width d -- the paper's appendix-B variant (temporal attention on ``x_c`` only),
  not its low-rank temporal attention over ``d + m``.
* ``mhc_internal_attention = "mhc"``: dense mHC read / write / stream mixer, plus the internal
  attention after each mixer sublayer's write.

In both, the internal attention follows each mixer sublayer (the paper's TAttn -> IAttn -> FFN
order); MLP / MoE sublayers have none.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def iat_mode(config) -> str:
    """``"none"``, ``"compute"`` or ``"mhc"``."""
    return getattr(config, "mhc_internal_attention", "none") or "none"


def expand_streams(hidden_states: torch.Tensor, n: int, config) -> torch.Tensor:
    """The embedding into ``[s, b, n, h]`` streams: copies (mHC) or ``(x, 0, ..., 0)`` (IAT
    compute mode: the memory part starts at zero)."""
    if iat_mode(config) == "compute":
        empty = hidden_states.new_zeros(*hidden_states.shape[:-1], n - 1, hidden_states.shape[-1])
        return torch.cat((hidden_states.unsqueeze(-2), empty), dim=-2)
    from megatron.core.transformer.mhc import expand_to_streams

    return expand_to_streams(hidden_states, n)


def collapse_streams(streams: torch.Tensor, config) -> torch.Tensor:
    """The streams into the head's ``[s, b, h]``: the mean (mHC) or ``x_c`` (IAT compute)."""
    if iat_mode(config) == "compute":
        return streams[..., 0, :]
    from megatron.core.transformer.mhc import collapse_streams as mean_streams

    return mean_streams(streams)


class InternalAttention(nn.Module):
    """``x + IAttn(Norm_{n h}(x))`` on ``[s, b, n, h]`` streams: per token, the n streams are the
    tokens of a bidirectional multi-head attention with shared ``h x h`` maps."""

    def __init__(self, config, n: int):
        super().__init__()
        h = config.hidden_size
        self.n, self.h = n, h
        self.heads = config.mhc_internal_attention_heads or config.num_attention_heads
        self.eps = config.layernorm_epsilon
        self.layernorm = config.normalization == "LayerNorm"
        self.zero_centered = getattr(config, "layernorm_zero_centered_gamma", False)
        self.norm_weight = nn.Parameter(torch.empty(n * h))
        self.norm_bias = nn.Parameter(torch.zeros(n * h)) if self.layernorm else None
        # learned position of each stream (chunk); content-scale, since it is added to the
        # normalized streams (the memory streams are all-zero at init in compute mode, so their
        # queries / keys are the positions alone)
        self.pos = nn.Parameter(torch.empty(n, h))
        self.pos.use_adam_with_emerging_optimizers = True  # a table, not a linear map
        # separate (not fused) h x h maps: the emerging optimizers normalize per real matrix
        self.w_q = nn.Parameter(torch.empty(h, h))
        self.w_k = nn.Parameter(torch.empty(h, h))
        self.w_v = nn.Parameter(torch.empty(h, h))
        self.w_o = nn.Parameter(torch.empty(h, h))
        self._init_method = config.init_method
        self._output_init_method = config.output_layer_init_method
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.norm_weight.fill_(0.0 if self.zero_centered else 1.0)
            if self.norm_bias is not None:
                self.norm_bias.zero_()
            self.pos.normal_(0.0, 1.0)
        for w in (self.w_q, self.w_k, self.w_v):
            self._init_method(w)
        self._output_init_method(self.w_o)

    def _norm(self, streams: torch.Tensor) -> torch.Tensor:
        s, b, n, h = streams.shape
        flat = streams.reshape(s, b, n * h)
        weight = self.norm_weight + 1 if self.zero_centered else self.norm_weight
        if self.layernorm:
            out = F.layer_norm(flat.float(), (n * h,), weight.float(), self.norm_bias.float(),
                               self.eps)
        else:
            x = flat.float()
            out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight.float()
        return out.to(streams.dtype).view(s, b, n, h)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        s, b, n, h = streams.shape
        heads, dh = self.heads, h // self.heads
        z = self._norm(streams) + self.pos.to(streams.dtype)
        t = s * b

        def split(w):  # [s, b, n, h] -> [t, heads, n, dh]
            return F.linear(z, w).view(t, n, heads, dh).transpose(1, 2)

        q, k, v = split(self.w_q), split(self.w_k), split(self.w_v)
        o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(dh))
        o = o.transpose(1, 2).reshape(s, b, n, h)
        return streams + F.linear(o, self.w_o)


class IATStreamMixer(nn.Module):
    """One sublayer's residual under IAT, mHC mixer API (see the module docstring)."""

    def __init__(self, config, kind: str):
        super().__init__()
        self.mode = iat_mode(config)
        n = config.mhc_expansion_rate
        self.mhc = None
        if self.mode == "mhc":
            from megatron.core.transformer.mhc import MHCMixer

            self.mhc = MHCMixer(
                hidden_size=config.hidden_size,
                expansion_rate=n,
                num_sinkhorn_iters=config.mhc_sinkhorn_iters,
                alpha_init=config.mhc_alpha_init,
                res_init=config.mhc_res_init,
                eps=config.layernorm_epsilon,
            )
        self.iattn = InternalAttention(config, n) if kind == "mixer" else None

    def _tag_inner(self) -> None:
        # the stack tags this module; the inner mHC mixer records its metrics under that tag
        tag = getattr(self, "_arch_metrics_tag", None)
        if self.mhc is not None and tag is not None:
            self.mhc._arch_metrics_tag = tag

    def forward(self, streams: torch.Tensor):
        if self.mhc is not None:
            return self.mhc(streams)
        return None, None, None

    def coeffs_and_read(self, streams: torch.Tensor):
        return None if self.mhc is None else self.mhc.coeffs_and_read(streams)

    def read(self, streams: torch.Tensor, h_pre=None) -> torch.Tensor:
        if self.mhc is not None:
            return self.mhc.read(streams, h_pre)
        return streams[..., 0, :]

    def write(self, streams, sublayer_out, h_post=None, h_res=None, lerp=None, link=None):
        if lerp is not None:
            raise NotImplementedError("mhc_internal_attention with the nGPT LERP residual")
        if self.mhc is not None:
            streams = self.mhc.write(streams, sublayer_out, h_post, h_res)
        else:
            streams = torch.cat(
                ((streams[..., 0, :] + sublayer_out).unsqueeze(-2), streams[..., 1:, :]), dim=-2
            )
        return streams if self.iattn is None else self.iattn(streams)

    def record_metrics(self, streams, coeffs) -> None:
        from megatron.core.transformer import arch_metrics

        if self.mhc is not None:
            self._tag_inner()
            self.mhc.record_metrics(streams, coeffs)
            return
        tag = arch_metrics.tag_of(self)
        if tag is not None:
            kind, layer, num_layers = tag
            arch_metrics.record_streams(f"iat_{kind}/stream_rms", layer, num_layers, streams)
