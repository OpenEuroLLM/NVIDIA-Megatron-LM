"""A second mixer in parallel to a complex_kda mixer, faded in while the complex_kda one fades out.

For the layers in ``config.ckda_branch_layers`` the complex_kda adapter (complex_kda.py) adds

    out = alpha(t) * ckda(x) + m(t) * gamma * branch(x)

- ``branch``: Megatron's own softmax attention, built as the hybrid's ``*`` layers are (fused
  ``linear_qkv`` per query group [q | gate | k | v], output gate, qk norm, XSA, NoPE) but with
  ``ckda_branch_num_query_groups`` KV groups and a plain ``linear_qkv`` (the layer's input norm
  is the complex_kda layer's); or the KronLSTM layer (``ckda_branch="kronlstm"``).
- ``gamma``: one learnable scale per layer, shape (1,) (Adam, no weight decay), 0 at insertion,
  so the model computes exactly what it computed before; the layer uses
  ``ckda_branch_gamma_scale * gamma`` (how fast Adam's ~lr steps move the branch's share).
- ``ckda_branch_crossfade``: the branch's weight is ``(1 - alpha(t)) * (1 + scale * gamma)`` instead
  (the two mixers' weights add up to 1, gamma a learnt correction of the branch's scale; m unused).
- ``alpha(t)``: 1 for ``ckda_branch_alpha_hold`` steps after ``ckda_branch_start_iteration``,
  then linear to ``ckda_branch_alpha_end`` over ``ckda_branch_alpha_ramp`` steps.
- ``m(t)``: an external multiplier, ``ckda_branch_m_start`` falling linearly to 1 over
  ``ckda_branch_m_ramp`` steps: the branch's contribution grows faster than its weights alone
  would move it, and the amplification is gone once the branch carries the layer.

At alpha = 0 an attention branch is a standard attention layer of the model's kind (fold
``m * gamma`` into ``linear_proj``, its input norm is the layer's), so the checkpoint can become
one. The branch's initial weights come from checkpoint surgery
(scripts/korbi/production/add_ckda_branch.py): projections copied from the complex_kda mixer.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

def current_iteration() -> int | None:
    """The training loop's iteration (args.curr_iteration), or None outside training."""
    try:
        from megatron.training import get_args

        it = getattr(get_args(), "curr_iteration", None)
        return int(it) if it is not None else None
    except Exception:  # no args (unit tests, inference): the schedules use their start values
        return None


def schedule(config, iteration: int | None) -> tuple[float, float]:
    """(alpha, m) at ``iteration``; at None (no training loop) the insertion values (1, m_start)."""
    if iteration is None:
        return 1.0, float(config.ckda_branch_m_start)
    t = max(0, iteration - config.ckda_branch_start_iteration)
    hold, ramp, end = config.ckda_branch_alpha_hold, config.ckda_branch_alpha_ramp, config.ckda_branch_alpha_end
    if t <= hold:
        alpha = 1.0
    elif ramp <= 0 or t >= hold + ramp:
        alpha = float(end)
    else:
        alpha = 1.0 + (float(end) - 1.0) * (t - hold) / ramp
    m0, mramp = float(config.ckda_branch_m_start), config.ckda_branch_m_ramp
    m = 1.0 if mramp <= 0 or t >= mramp else m0 + (1.0 - m0) * t / mramp
    return alpha, m


def branch_config(config):
    """The branch's config: a shallow copy (no __post_init__ re-run) with the branch's KV groups."""
    c = copy.copy(config)
    c.ckda_branch_from_ckda = True  # read by the KronLSTM wrapper (conv activation)
    if config.ckda_branch == "attention":
        c.num_query_groups = config.ckda_branch_num_query_groups
        if c.num_attention_heads % c.num_query_groups:
            raise ValueError(
                f"ckda_branch: {c.num_attention_heads} query heads do not split into "
                f"{c.num_query_groups} KV groups"
            )
    return c


def build_branch(config, layer_number: int, pg_collection=None) -> nn.Module:
    if config.ckda_branch == "attention":
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TEDotProductAttention,
            TERowParallelLinear,
        )
        from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.spec_utils import ModuleSpec, build_module

        spec = ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,  # the input norm is the complex_kda layer's
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        )
        cfg = branch_config(config)
        attn = build_module(spec, config=cfg, layer_number=layer_number, pg_collection=pg_collection)
        # head-wise orthogonalization (muon_qkv_headwise): one block per 256-row head; the
        # model-level split shapes assume the stack's own KV groups
        attn.linear_qkv.weight.headwise_dim = cfg.kv_channels
        return attn
    if config.ckda_branch == "kronlstm":
        from megatron.core.ssm.fla_hybrid import KronLSTM

        kron = KronLSTM(branch_config(config), layer_number=layer_number, pg_collection=pg_collection)
        # linear_beta_adam: the per-head erase / write gates on Adam, as complex_kda's b_proj (the
        # branch starts from its states: b_proj and c_proj are copies of complex_kda's b_proj)
        if getattr(config, "linear_beta_adam", False):
            for name in ("b_proj", "c_proj"):
                mod = getattr(kron.layer, name, None)
                if mod is not None:
                    mod.weight.use_adam_with_emerging_optimizers = True
        # muon_qkv_headwise: one block per head, as complex_kda's q / k / v projections (the wrapper's
        # [q; k; v] split tag is for stand-alone layers)
        kron.layer.qkv_proj.weight.headwise_dim = config.linear_key_head_dim
        return kron
    raise ValueError(f"ckda_branch {config.ckda_branch!r}: attention or kronlstm")


def has_branch(config, layer_number: int) -> bool:
    return bool(getattr(config, "ckda_branch", None)) and (layer_number - 1) in set(config.ckda_branch_layers or [])


class BranchMix(nn.Module):
    """gamma and the alpha / m schedules of one layer, plus the ``[ckda_branch]`` log lines."""

    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.gamma = nn.Parameter(torch.zeros(1))
        self._logged = None

    def forward(self, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        it = current_iteration()
        alpha, m = schedule(self.config, it)
        gamma = self.gamma * float(getattr(self.config, "ckda_branch_gamma_scale", 1.0))
        if getattr(self.config, "ckda_branch_crossfade", False):
            m = 1.0 - alpha  # the weights add up to 1; gamma corrects the branch's scale around 1
            gamma = 1.0 + gamma
        out = y * alpha + b * (gamma.to(b.dtype) * m)
        every = getattr(self.config, "ckda_branch_log_interval", 0)
        if every and it is not None and it % every == 0 and self._logged != it and torch.is_grad_enabled():
            self._logged = it
            self._log(it, alpha, m, gamma, y, b)
        return out

    def _log(self, it, alpha, m, gamma, y, b):
        rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not rank0:
            return
        with torch.no_grad():
            g = float(gamma.detach().float().reshape(-1)[0])  # the scaled gamma the layer uses
            ry = float(y.detach().float().pow(2).mean().sqrt())
            rb = float(b.detach().float().pow(2).mean().sqrt())
        print(f"[ckda_branch] iter {it} layer {self.layer_number - 1} alpha {alpha:.4f} m {m:.3f} "
              f"gamma {g:+.5f} rms_ckda {ry:.4e} rms_branch {rb:.4e} "
              f"rms_branch_scaled {abs(g) * m * rb:.4e} rms_ckda_scaled {alpha * ry:.4e}", flush=True)
