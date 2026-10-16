# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Power retention (power attention) as a linear mixer: the ``power`` variant.

Power retention (Buckman et al. 2025, "Scaling Context Requires Rethinking
Attention"; github.com/m-a-n-i-f-e-s-t/retention) is gated linear attention with the
symmetric ``deg``-th power of the keys and queries as feature map -- normalized, so
it is attention with ``(q . k)^deg`` similarities -- run chunkwise with a state of
``binom(d + deg - 1, deg) x (d + 1)`` per head::

    o_t = sum_{s<=t} G_{s..t} (q_t . k_s)^deg v_s / sum_{s<=t} G_{s..t} (q_t . k_s)^deg

This module is the baseline: the public kernels (``retention.triton``, Triton, no
build step) behind a layer shaped as their reference model -- q, k, v and a
per-head gate projection (``log G = logsigmoid(6.9 + W x)``, i.e. retention 0.999 at
init), optional RoPE on q / k (off by default: NoPE), output projection. A more
efficient implementation (e.g. TileLang) can replace the kernel call later.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from retention.triton import power_retention

    HAVE_RETENTION = True
except ImportError:  # pragma: no cover
    power_retention = None
    HAVE_RETENTION = False

GATE_INIT = 6.906768  # logsigmoid(6.906768) = log(0.999)

_FLEX = {}


def _flex_attention():
    """torch.compile'd flex_attention (compiled once per process)."""
    if "fn" not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


def _causal_block_mask(t: int, device):
    key = (t, str(device))
    if key not in _FLEX:
        from torch.nn.attention.flex_attention import create_block_mask

        _FLEX[key] = create_block_mask(
            lambda b, h, q, kv: q >= kv, B=None, H=None, Q_LEN=t, KV_LEN=t, device=device
        )
    return _FLEX[key]


def power_attention_flex(q, k, v, log_g=None, deg: int = 2, scale: Optional[float] = None):
    """The quadratic form of power retention on FlexAttention ([B, T, H, D] in and out).

    Its weights ``(scale q_t . k_s)^deg exp(L_t - L_s)`` (L = cumsum log g) normalised over s
    are a softmax over ``deg log|scale q_t . k_s| + L_t - L_s`` -- the retention package's
    attention form (retention/_attention/reference.py: the same 1e-7 inside the log) as a
    score_mod, so the flash-attention-style kernel never forms the [T, T] weights."""
    b, t, h, d = q.shape
    scale = d**-0.5 if scale is None else scale
    Lq = Lk = None
    if log_g is not None:
        L = log_g.float().cumsum(1).transpose(1, 2)  # [B, H, T]
        # one tensor per indexing: flex's score_mod backward takes each captured tensor
        # indexed once (autograd sums the two gradients)
        Lq, Lk = L.contiguous(), L.clone()

    def score_mod(score, bi, hi, qi, ki):
        s = deg * torch.log(torch.abs(score) + 1e-7)
        if Lq is not None:
            s = s + (Lq[bi, hi, qi] - Lk[bi, hi, ki])
        return s

    o = _flex_attention()(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), score_mod=score_mod,
        block_mask=_causal_block_mask(t, q.device), scale=scale,
    )
    return o.transpose(1, 2)


def _rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding over the head dim of ``x`` [B, T, H, D] (positions 0..T-1)."""
    b, t, h, d = x.shape
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)


class PowerRetentionAttention(nn.Module):
    """A power-retention mixer in the FLA layer convention ([B, T, hidden] in,
    ``(o, None, None)`` out), as built by the ``power`` linear variant."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        deg: int = 2,
        chunk_size: int = 128,
        switch_over_seq_len: int = 0,
        attention_impl: str = "retention",
        rope: bool = False,
        gating: bool = True,
        xsa: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if not HAVE_RETENTION:
            raise ImportError(
                "the power variant needs the retention package (github.com/m-a-n-i-f-e-s-t/"
                "retention, with vidrial) on the path"
            )
        if deg != 2:
            # The public Triton kernels implement the chunked algorithm for deg = 2 only
            # (higher degrees: the package's vidrial CUDA kernels, or our own later).
            raise ValueError(f"the power-retention Triton baseline supports deg=2, got {deg}")
        self.h, self.d, self.deg, self.chunk_size = num_heads, head_dim, deg, chunk_size
        self.switch_over_seq_len = switch_over_seq_len
        if attention_impl not in ("flex", "retention"):
            raise ValueError(f"power attention_impl must be flex or retention, got {attention_impl!r}")
        self.attention_impl = attention_impl
        self.rope, self.gating, self.xsa = rope, gating, xsa
        dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads, bias=False) if gating else None
        self.o_proj = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, **kwargs):
        if attention_mask is not None:
            raise NotImplementedError("the power variant takes no padding mask")
        x = hidden_states
        b, t, _ = x.shape
        # all input projections as one GEMM (weights concatenated at run time)
        mods = [self.q_proj, self.k_proj, self.v_proj] + ([self.g_proj] if self.gating else [])
        weights = [m.weight for m in mods]
        pad = -sum(w.shape[0] for w in weights) % 64  # odd widths run on a slow sm80 GEMM
        if pad:
            weights.append(weights[0].new_zeros(pad, weights[0].shape[1]))
        y = F.linear(x, torch.cat(weights, dim=0))
        dim = self.h * self.d
        q, k, v = (y[..., i * dim : (i + 1) * dim].view(b, t, self.h, self.d) for i in range(3))
        if self.rope:
            q, k = _rope(q), _rope(k)
        log_g = None
        if self.gating:
            log_g = F.logsigmoid(GATE_INIT + y[..., 3 * dim : 3 * dim + self.h].float()).contiguous()
        # the quadratic (attention) form up to switch_over_seq_len tokens: faster than any
        # chunk size there (and the only form for t <= chunk_size)
        chunk = self.chunk_size if t > max(self.chunk_size, self.switch_over_seq_len) else None
        if chunk is None and self.attention_impl == "flex":
            o = power_attention_flex(q, k, v, log_g, deg=self.deg, scale=self.d**-0.5)
        else:
            o = power_retention(
                q.contiguous(), k.contiguous(), v.contiguous(), log_g,
                deg=self.deg, scale=self.d**-0.5, chunk_size=chunk,
            )
        if self.xsa:
            from megatron.core.fusions.fused_xsa import exclusive_self_attention_heads

            o = exclusive_self_attention_heads(o, v)
        return self.o_proj(o.reshape(b, t, self.h * self.d)), None, None
