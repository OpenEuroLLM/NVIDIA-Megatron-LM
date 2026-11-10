"""A second mixer in parallel to a complex_kda mixer, faded in while the complex_kda one fades out.

For the layers in ``config.ckda_branch_layers`` the complex_kda adapter (complex_kda.py) adds

    out = alpha(t) * ckda(x) + m(t) * gamma * branch(x)

- ``branch``: Megatron's own softmax attention, built as the hybrid's ``*`` layers are (fused
  ``linear_qkv`` per query group [q | gate | k | v], output gate, qk norm, XSA, NoPE) but with
  ``ckda_branch_num_query_groups`` KV groups and a plain ``linear_qkv`` (the layer's input norm
  is the complex_kda layer's); or the KronLSTM layer (``ckda_branch="kronlstm"``); or
  (``ckda_branch="kvshare"``) that attention with a query (+ output gate) projection only, over the
  keys / values of an earlier attention layer (``ckda_branch_kv_source_layers``): no KV of its own.
- ``gamma``: one learnable scale per layer, shape (1,) (Adam, no weight decay), 0 at insertion,
  so the model computes exactly what it computed before; the layer uses
  ``ckda_branch_gamma_scale * gamma`` (how fast Adam's ~lr steps move the branch's share).
- ``ckda_branch_crossfade``: the branch's weight is ``(1 - alpha(t)) * (1 + scale * gamma)`` instead
  (the two mixers' weights add up to 1, gamma a learnt correction of the branch's scale; m unused).
- ``ckda_branch_calibrate``: a buffer ``calib`` = RMS(ckda) / RMS(branch), measured on the first
  training forward (all ranks), multiplies the branch's weight: it starts at the mixer's scale.
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
    """The model's iteration: the training loop's (args.curr_iteration) or, when later, the loaded
    checkpoint's (args.iteration) -- an evaluation-only run (train_iters = the checkpoint's step) never
    enters the loop and leaves curr_iteration at 0, which would evaluate the start of the hand-over
    (alpha 1) instead of the checkpoint's mix. None outside Megatron (unit tests, inference)."""
    try:
        from megatron.training import get_args

        args = get_args()
    except Exception:  # no args (unit tests, inference): the schedules use their start values
        return None
    its = [int(v) for v in (getattr(args, "curr_iteration", None), getattr(args, "iteration", None))
           if v is not None]
    return max(its) if its else None


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


# kvshare: the source attention layers' (key, value) of the current micro-batch, by 0-based layer index.
# A source layer runs before its consumers in the forward and in every recompute (the consumer's
# recompute segment comes earlier in the backward than the source's, so it reads the forward's entry;
# a source in the same segment re-publishes the same values first). Detached: no gradient into the source.
KV_STORE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def kv_source(config, layer_number: int) -> int | None:
    """The kvshare source layer (0-based) of the branch layer ``layer_number`` (1-based)."""
    layers = list(config.ckda_branch_layers or [])
    sources = list(getattr(config, "ckda_branch_kv_source_layers", None) or [])
    if len(sources) != len(layers):
        raise ValueError(
            f"ckda_branch kvshare: ckda_branch_kv_source_layers {sources} needs one entry per "
            f"ckda_branch_layers entry {layers}"
        )
    src = sources[layers.index(layer_number - 1)]
    if src >= layer_number - 1:
        raise ValueError(f"ckda_branch kvshare: source layer {src} is not before layer {layer_number - 1}")
    return src


def publishes_kv(config, layer_number: int) -> bool:
    """Whether the attention layer ``layer_number`` (1-based) is a kvshare source."""
    return getattr(config, "ckda_branch", None) == "kvshare" and (layer_number - 1) in set(
        getattr(config, "ckda_branch_kv_source_layers", None) or []
    )


def publish_kv(layer_number: int, key: torch.Tensor, value: torch.Tensor) -> None:
    KV_STORE[layer_number - 1] = (key.detach(), value.detach())


def fetch_kv(src: int) -> tuple[torch.Tensor, torch.Tensor]:
    if src not in KV_STORE:
        raise RuntimeError(
            f"ckda_branch kvshare: no keys / values of layer {src} (not an attention layer of this "
            f"rank, or it did not run before the branch)"
        )
    return KV_STORE[src]


def _kvshare_attention_class():
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.typed_torch import apply_module

    class KVShareSelfAttention(SelfAttention):
        """SelfAttention whose keys / values are an earlier attention layer's (``fetch_kv``):
        ``linear_qkv`` produces per query group [q | gate] only (the stack's own layout minus k, v),
        q norm, output gate and XSA as the source's kind; no k norm (the keys come normed)."""

        def __init__(self, config, submodules, layer_number, kv_src: int, **kwargs):
            super().__init__(config, submodules, layer_number, **kwargs)
            self.kv_src = kv_src
            if getattr(self, "canon_b", None) is not None:
                raise NotImplementedError("ckda_branch kvshare: Canon B")
            if self.config.num_query_groups < self.world_size:
                raise NotImplementedError("ckda_branch kvshare: fewer KV groups than TP ranks")
            gated = 2 if self.config.attention_output_gate else 1
            self.linear_qkv_out_dim = self.query_projection_size * gated
            self.linear_qkv = submodules.linear_qkv(
                self.config.hidden_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
                tp_group=self.pg_collection.tp,
            )
            self.k_layernorm = None

        def _fused_qk_norm_rope_applies(self, inference_context, packed_seq_params, rotary_pos_emb):
            return False  # no fused k / v: the unfused path below

        def get_query_key_value_tensors(self, hidden_states, key_value_states=None, output_gate=False,
                                        split_qkv=True):
            if not split_qkv:
                raise ValueError("ckda_branch kvshare: split_qkv only")
            mixed, _ = apply_module(self.linear_qkv)(hidden_states)
            ng = self.num_query_groups_per_partition
            per = self.num_attention_heads_per_partition // ng * self.hidden_size_per_attention_head
            mixed = mixed.view(*mixed.shape[:-1], ng, -1)  # [s, b, ng, (1 | 2) * np/ng * hn]
            if output_gate:
                query, gate = torch.split(mixed, [per, per], dim=3)
                gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            else:
                query, gate = mixed, None
            query = query.reshape(*query.shape[:2], -1, self.hidden_size_per_attention_head)
            if self.q_layernorm is not None:
                query = apply_module(self.q_layernorm)(query)
            # key / value: placeholders of the KV shape (views, no memory), replaced by the source's
            # right before the core attention (Attention.forward)
            key = value = query[:, :, :ng]
            return (query, key, value, gate) if output_gate else (query, key, value)

    return KVShareSelfAttention


def build_branch(config, layer_number: int, pg_collection=None) -> nn.Module:
    if config.ckda_branch == "kvshare":
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TEDotProductAttention,
            TERowParallelLinear,
        )
        from megatron.core.transformer.attention import SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType

        if config.context_parallel_size != 1:
            raise NotImplementedError("ckda_branch kvshare: context parallelism")
        attn = _kvshare_attention_class()(
            config,  # the stack's own KV groups: the source's keys / values
            SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
            layer_number,
            kv_source(config, layer_number),
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )
        attn.linear_qkv.weight.headwise_dim = config.kv_channels
        return attn
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
    raise ValueError(f"ckda_branch {config.ckda_branch!r}: attention, kronlstm or kvshare")


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
        self._calibrated = False
        if getattr(config, "ckda_branch_calibrate", False):
            if getattr(config, "pipeline_model_parallel_size", 1) != 1:
                raise NotImplementedError("ckda_branch_calibrate: pipeline parallelism (the all-reduce spans every rank)")
            # RMS(ckda) / RMS(branch), measured once (0: not yet); saved with the model, the surgery writes 0
            self.register_buffer("calib", torch.zeros(1), persistent=True)
        else:
            self.calib = None

    @torch.no_grad()
    def _calibrate(self, y: torch.Tensor, b: torch.Tensor) -> None:
        """calib = RMS(y) / RMS(b) over every rank's first training micro-batch (once; a loaded nonzero
        value is kept). At that step alpha is 1, so the output does not depend on it (recompute-safe)."""
        self._calibrated = True
        if float(self.calib.float().reshape(-1)[0]) != 0.0:
            return
        sq = torch.stack([y.float().pow(2).sum(), b.float().pow(2).sum()])
        if torch.distributed.is_initialized():
            # every rank runs every layer in the same order (pipeline size 1): the default group
            torch.distributed.all_reduce(sq)
        ratio = (sq[0] / sq[1].clamp_min(1e-30)).sqrt()
        self.calib.fill_(float(ratio))
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[ckda_branch] layer {self.layer_number - 1} calibrated: branch scale "
                  f"RMS(ckda) / RMS(branch) = {float(ratio):.4f}", flush=True)

    def forward(self, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        it = current_iteration()
        alpha, m = schedule(self.config, it)
        if self.calib is not None and not self._calibrated and self.training:
            self._calibrate(y, b)
        gamma = self.gamma * float(getattr(self.config, "ckda_branch_gamma_scale", 1.0))
        if getattr(self.config, "ckda_branch_crossfade", False):
            m = 1.0 - alpha  # the weights add up to 1; gamma corrects the branch's scale around 1
            gamma = 1.0 + gamma
        if self.calib is not None:
            gamma = gamma * self.calib.to(gamma.dtype)
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
