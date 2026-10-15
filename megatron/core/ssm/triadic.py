# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Triadic linear attention over the delta rule: Triadic GDN / KDA / Complex KDA.

Triadic Linear Attention (Sieberling et al. 2026, arXiv:2609.36529) keeps per head a
third-order state of ``E`` slices ``S_e`` (``D x V`` each), written with the outer
product of the key, a second key ``k2`` (``E`` scalars) and the value, and read with
the query and a second query ``q2``::

    S_e <- alpha_{t,e} S_e
    S   <- S + beta_t kappa_t (v_t - S^T kappa_t)^T,   kappa_t = k2_t (x) k_t
    o_t  = S^T (q2_t (x) q_t) * scale

That is the gated delta rule over the joint key ``kappa`` of width ``E * D`` with a
decay that is constant within each slice -- KDA with a Kronecker-structured key. The
natural extensions to KDA and Complex KDA keep KDA's per-channel decay inside each
slice, with a separable gate::

    log |alpha_{t,e,d}| = g2_{t,e} + g_{t,d},    sign(alpha_{t,e,d}) = sign_{t,d}

(``triadic_base``: "gdn" -- no channel gate; "kda" -- softplus channel gate; "ckda" --
the signed channel gate of Complex KDA, with ``beta`` up to 2). ``E = 1`` is the base
mixer itself.

Two backends (``linear_triadic_kernel``):

* "kda": the existing KDA chunk kernel on the expanded ``kappa`` / ``q2 (x) q`` -- exact,
  every base, any GPU the kernel runs on, but the keys are ``E`` times wider in memory
  and FLA's ``chunk_kda`` takes ``E * D <= 256``.
* "cute": the CuTe DSL kernels of cute-triadic-gdn (Hopper / Blackwell), which never
  form ``kappa``: Triadic GDN only (no channel gate), ``D = 128``, ``E`` in
  {1, 2, 4, 8, 12, 16}. "auto" takes them when they apply and are installed.
:func:`triadic_reference` is the naive recurrence over the third-order state the
expansion is tested against.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.layers.complex_kda_layer import compute_gate, init_dt_bias
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda

    HAVE_FLA = True
except ImportError:  # pragma: no cover
    HAVE_FLA = False

BASES = ("gdn", "kda", "ckda")
KERNELS = ("auto", "kda", "cute")
CUTE_E = (1, 2, 4, 8, 12, 16)


def _cute_gdn():
    """cute_triadic_gdn.gdn_joint_call, or None when not installed / not sm90+."""
    try:
        from cute_triadic_gdn import gdn_joint_call
    except Exception:  # pragma: no cover - not installed
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return None
    return gdn_joint_call


def cute_applies(base: str, head_dim: int, e: int) -> bool:
    return base == "gdn" and head_dim == 128 and e in CUTE_E


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def expand_triadic(q, k, q2, k2, g2, g=None, sign=None):
    """The KDA inputs of a triadic step: ``q``, ``k`` [B, T, H, D] (unit norm), ``q2``,
    ``k2`` [B, T, H, E] (unit norm), ``g2`` [B, T, H, E] slice log-decays, ``g``
    [B, T, H, D] channel log-decays or None, ``sign`` [B, T, H, D] or None ->
    ``(q2 (x) q, k2 (x) k, g2 (+) g, sign tiled over E)``, each [B, T, H, E * D]
    (index ``e * D + d``)."""
    b, t, h, d = q.shape
    e = q2.shape[-1]
    qq = (q2.unsqueeze(-1) * q.unsqueeze(-2)).reshape(b, t, h, e * d)
    kk = (k2.unsqueeze(-1) * k.unsqueeze(-2)).reshape(b, t, h, e * d)
    gg = g2.unsqueeze(-1).expand(b, t, h, e, d)
    if g is not None:
        gg = gg + g.unsqueeze(-2)
    gg = gg.reshape(b, t, h, e * d)
    ss = None
    if sign is not None:
        ss = sign.unsqueeze(-2).expand(b, t, h, e, d).reshape(b, t, h, e * d)
    return qq, kk, gg, ss


def triadic_reference(q, k, v, q2, k2, g2, beta, g=None, sign=None, scale=None):
    """Naive recurrence over the third-order state ``S`` [B, H, E, D, V] (fp32).

    ``q``, ``k`` [B, T, H, D] and ``q2``, ``k2`` [B, T, H, E] already unit norm,
    ``beta`` [B, T, H] the final write strength. Differentiable (autograd through the
    loop), for gradient tests on short sequences."""
    b, t, h, d = q.shape
    e, vd = q2.shape[-1], v.shape[-1]
    scale = d**-0.5 if scale is None else scale
    S = torch.zeros(b, h, e, d, vd, device=q.device, dtype=torch.float32)
    out = torch.empty(b, t, h, vd, device=q.device, dtype=torch.float32)
    for i in range(t):
        log_a = g2[:, i].float().unsqueeze(-1)  # [B, H, E, 1]
        if g is not None:
            log_a = log_a + g[:, i].float().unsqueeze(-2)  # [B, H, E, D]
        alpha = log_a.exp().expand(b, h, e, d)
        if sign is not None:
            alpha = alpha * sign[:, i].float().unsqueeze(-2)
        S = S * alpha.unsqueeze(-1)
        kappa = k2[:, i].float().unsqueeze(-1) * k[:, i].float().unsqueeze(-2)  # [B, H, E, D]
        pred = torch.einsum("bhed,bhedv->bhv", kappa, S)
        upd = beta[:, i].float()[..., None] * (v[:, i].float() - pred)  # [B, H, V]
        S = S + kappa.unsqueeze(-1) * upd[:, :, None, None, :]
        qq = q2[:, i].float().unsqueeze(-1) * q[:, i].float().unsqueeze(-2)
        out[:, i] = torch.einsum("bhed,bhedv->bhv", qq, S) * scale
    return out


def triadic_chunk(q, k, v, q2, k2, g2, beta_logits, g=None, sign=None, allow_neg_eigval=False,
                  scale=None, cu_seqlens=None):
    """The triadic delta rule through FLA's ``chunk_kda`` on the expanded keys.
    ``beta_logits`` [B, T, H]: the kernel applies the sigmoid (x2 with
    ``allow_neg_eigval``). Returns ``o`` [B, T, H, V]."""
    d, e = q.shape[-1], q2.shape[-1]
    qq, kk, gg, ss = expand_triadic(q, k, q2, k2, g2, g, sign)
    if e * d > 256:
        return _chunk_kda_wide(qq.to(v.dtype), kk.to(v.dtype), v, gg.float(), beta_logits, ss,
                               d**-0.5 if scale is None else scale, allow_neg_eigval, cu_seqlens)
    o, _ = chunk_kda(
        q=qq.to(v.dtype),
        k=kk.to(v.dtype),
        v=v,
        g=gg.float(),
        beta=beta_logits,
        sign=ss,
        scale=d**-0.5 if scale is None else scale,
        use_qk_l2norm_in_kernel=False,  # q (x) q2 and k (x) k2 are unit norm by construction
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=False,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )
    return o


def _chunk_kda_wide(q, k, v, g, beta, sign, scale, allow_neg_eigval, cu_seqlens):
    """FLA's KDA chunk pipeline for keys wider than its wrapper's ``K <= 256``: the
    state recurrence runs on the wide-key kernels (triadic_wide.py, registered as an
    FLA backend for K > 256, K-major state layout); FLA's other kernels tile the key
    dimension themselves."""
    from fla.ops.kda.chunk import ChunkKDAFunction
    from fla.ops.kda.implementation import resolve_implementation

    from megatron.core.ssm.triadic_wide import register_wide_key_backend

    register_wide_key_backend()
    implementation = resolve_implementation(None, q, v, False, None)
    o, _ = ChunkKDAFunction.apply(
        q, k, v, g, beta, sign,
        None, None,          # A_log, dt_bias (gate computed outside the kernel)
        scale,
        None, False,         # initial_state, output_final_state
        False, False, True,  # use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid
        allow_neg_eigval,
        False,               # state_v_first: the wide kernels use the K-major layout
        cu_seqlens, None,    # cu_seqlens, cu_seqlens_cpu
        False, None, 64,     # safe_gate, lower_bound, chunk_size
        False, False,        # disable_recompute, return_intermediate_states
        None,                # cp_context
        implementation,
    )
    return o


class TriadicDeltaAttention(nn.Module):
    """A triadic delta-rule mixer in the FLA layer convention ([B, T, hidden] in,
    ``(o, None, None)`` out), as built by the ``triadic`` linear variant."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        e: int = 2,
        base: str = "kda",
        gate: str = "signed_sigmoid2",
        allow_neg_eigval: bool = False,
        lower_bound: float = -5.0,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        layer_idx: Optional[int] = None,
        kernel: str = "auto",
    ) -> None:
        super().__init__()
        if not HAVE_FLA:
            raise ImportError("the triadic variant needs flash-linear-attention (ComplexKDA)")
        if base not in BASES:
            raise ValueError(f"triadic base must be one of {BASES}, got {base!r}")
        if kernel not in KERNELS:
            raise ValueError(f"triadic kernel must be one of {KERNELS}, got {kernel!r}")
        self.cute = None
        if kernel != "kda" and cute_applies(base, head_dim, e):
            self.cute = _cute_gdn()
        if kernel == "cute" and self.cute is None:
            raise ValueError(
                "linear_triadic_kernel=cute needs cute-triadic-gdn on an sm90+ GPU and "
                f"base=gdn, head_dim=128, E in {CUTE_E} (got base={base}, head_dim={head_dim}, "
                f"E={e})"
            )
        self.h, self.d, self.e, self.base = num_heads, head_dim, e, base
        self.gate = "softplus" if base == "kda" else gate
        self.allow_neg_eigval = allow_neg_eigval or base == "ckda"
        self.lower_bound = lower_bound
        key_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.q_conv1d = ShortConvolution(key_dim, conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(key_dim, conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(key_dim, conv_size, bias=False, activation="silu")
        if e > 1:
            # second key / query: short conv, softplus, L2-normalised over E (paper)
            self.k2_proj = nn.Linear(hidden_size, num_heads * e, bias=False)
            self.q2_proj = nn.Linear(hidden_size, num_heads * e, bias=False)
            self.k2_conv1d = ShortConvolution(num_heads * e, conv_size, bias=False, activation=None)
            self.q2_conv1d = ShortConvolution(num_heads * e, conv_size, bias=False, activation=None)
        # slice decay, per (head, slice): -exp(A2) softplus(W x + dt2)
        self.a2_proj = nn.Linear(hidden_size, num_heads * e, bias=False)
        self.A2_log = nn.Parameter(torch.log(torch.empty(num_heads * e).uniform_(1, 16)))
        self.dt2_bias = nn.Parameter(init_dt_bias("softplus", num_heads * e, lower_bound, "shipped"))
        for p in (self.A2_log, self.dt2_bias):
            p._no_weight_decay = True
        if base != "gdn":
            # KDA's per-channel gate, low rank as in KDA
            self.f_proj = nn.Sequential(
                nn.Linear(hidden_size, head_dim, bias=False),
                nn.Linear(head_dim, key_dim, bias=False),
            )
            self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(1, 16)))
            self.dt_bias = nn.Parameter(init_dt_bias(self.gate, key_dim, lower_bound, "shipped"))
            for p in (self.A_log, self.dt_bias):
                p._no_weight_decay = True
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, head_dim, bias=False),
            nn.Linear(head_dim, key_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(head_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(key_dim, hidden_size, bias=False)

    def _gates(self, x):
        b, t, _ = x.shape
        u2 = self.a2_proj(x).float() + self.dt2_bias.float()
        g2 = -self.A2_log.float().exp() * F.softplus(u2)
        g2 = g2.view(b, t, self.h, self.e)
        g = sign = None
        if self.base != "gdn":
            z = self.f_proj(x).view(b, t, self.h, self.d)
            sign, g = compute_gate(self.gate, z, self.A_log, self.dt_bias.view(self.h, self.d),
                                   self.lower_bound)
        return g2, g, sign

    def forward(self, hidden_states, attention_mask=None, use_cache=False, **kwargs):
        if attention_mask is not None:
            raise NotImplementedError("the triadic variant takes no padding mask")
        x = hidden_states
        b, t, _ = x.shape
        q, _ = self.q_conv1d(self.q_proj(x))
        k, _ = self.k_conv1d(self.k_proj(x))
        v, _ = self.v_conv1d(self.v_proj(x))
        q = _l2norm(q.view(b, t, self.h, self.d))
        k = _l2norm(k.view(b, t, self.h, self.d))
        v = v.view(b, t, self.h, self.d)
        if self.e > 1:
            k2, _ = self.k2_conv1d(self.k2_proj(x))
            q2, _ = self.q2_conv1d(self.q2_proj(x))
            k2 = _l2norm(F.softplus(k2.float()).view(b, t, self.h, self.e))
            q2 = _l2norm(F.softplus(q2.float()).view(b, t, self.h, self.e))
        else:
            k2 = q2 = torch.ones(b, t, self.h, 1, device=x.device, dtype=torch.float32)
        g2, g, sign = self._gates(x)
        beta = self.b_proj(x).float()
        if self.cute is not None and t % 64 == 0:
            # Triadic GDN without forming kappa (beta in (0, 1): sigmoid here).
            o = self.cute(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                          k2.contiguous(), q2.contiguous(), g2.contiguous(),
                          torch.sigmoid(beta), scale=self.d**-0.5)
        else:
            o = triadic_chunk(q, k, v, q2, k2, g2, beta, g=g, sign=sign,
                              allow_neg_eigval=self.allow_neg_eigval)
        gate = self.g_proj(x).view(b, t, self.h, self.d)
        o = self.o_norm(o, gate)
        return self.o_proj(o.reshape(b, t, self.h * self.d)), None, None
