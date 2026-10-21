# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron mixers wrapping self-contained flash-linear-attention (FLA) layers.

Unlike ``GatedDeltaNet``/``MLSTM`` -- which take Megatron's own
column/row-parallel projections as submodules and call only the FLA *kernel* --
``SignedKimiDeltaAttention`` and ``PaniniDeltaAttention`` are complete layers
that own their projections, short conv, gates and output norm. So the wrapper's
job is only:

  * layout: Megatron hands mixers ``[s, b, h]``; FLA layers take ``[b, s, h]``.
  * contract: return ``(output, bias)`` with ``bias=None``.
  * config: map ``TransformerConfig`` onto the FLA constructor.

TENSOR PARALLELISM REPLICATES THE LAYER. The FLA layers build plain ``nn.Linear``
projections internally, so there is nothing to shard; under TP>1 every TP rank runs
the whole layer (see megatron/core/ssm/tp_replicated.py: identical start, averaged
gradients, sequence gathered under sequence parallelism). Context parallelism is not
supported and asserts CP==1.

The 0.3.2 FLA in the training container carries none of these layers: prepend
a checkout that does to PYTHONPATH so it shadows the installed package --
ComplexKDA (github.com/OpenEuroLLM/ComplexKDA) for kda and signed_kda, the
legacy flash-linear-attention checkout for panini (which also still serves
kda and signed_kda). Each wrapper says which module it is missing.
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.tp_replicated import (
    gather_replicated_input,
    replicate_across_tp,
    scatter_replicated_output,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule

# Imported ONE BY ONE: which layers exist depends on which `fla` is on the path,
# and a single try block made any one missing module disable all three.
#
#   fla.layers.kda                     every FLA; plain KDA
#   fla.layers.complex_kda_layer       the released signed layer (ComplexKDA,
#                                      github.com/OpenEuroLLM/ComplexKDA)
#   fla.layers.signed_kda_layer        its pre-release name, in the legacy
#                                      flash-linear-attention checkout
#   fla.layers.panini_delta_attention  legacy checkout only
#
# signed_kda prefers the released class. At the arguments SignedKDA passes the
# two build identical parameters, init and outputs (checked bit for bit at both
# output gates), so a checkpoint from either loads into the other.
try:
    from fla.layers.kda import KimiDeltaAttention as _FLAKDA
except ImportError:  # pragma: no cover - exercised only without fla
    _FLAKDA = None
try:
    from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention as _FLASignedKDA
except ImportError:
    try:
        from fla.layers.signed_kda_layer import SignedKimiDeltaAttention as _FLASignedKDA
    except ImportError:
        _FLASignedKDA = None
try:
    from fla.layers.panini_delta_attention import PaniniDeltaAttention as _FLAPanini
except ImportError:
    _FLAPanini = None
try:
    from fla.layers.kronlstm import KronLSTM as _FLAKronLSTM
except ImportError:
    _FLAKronLSTM = None


def megatron_init_fla_layer(layer: torch.nn.Module, config: TransformerConfig) -> None:
    """Re-initialize an FLA layer's projections with Megatron's init methods.

    See ``TransformerConfig.linear_fla_megatron_init``. Runs before the layer is
    replicated across TP, which broadcasts rank 0's values.
    """
    if not getattr(config, "linear_fla_megatron_init", False) or not config.perform_initialization:
        return
    with torch.no_grad():
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name.split(".")[-1] == "o_proj":
                    config.output_layer_init_method(module.weight)
                else:
                    config.init_method(module.weight)


def _with_xsa(config, kwargs):
    """``kwargs`` plus ``xsa=True`` when linear_xsa is set (the ComplexKDA fork's option;
    left out otherwise, so forks without it keep working)."""
    if getattr(config, "linear_xsa", False):
        kwargs = dict(kwargs, xsa=True)
    return kwargs


def _declared_kwargs(cls, kwargs):
    """``kwargs``, refusing any the constructor does not declare.

    Both signed layers take ``**kwargs``, so an argument that was renamed or
    removed upstream would be swallowed and silently change nothing about the
    layer -- the failure mode that moved ``drop_silu``'s default from True
    (legacy) to False (released) without a word.
    """
    import inspect

    declared = inspect.signature(cls.__init__).parameters
    unknown = sorted(k for k in kwargs if k not in declared)
    if unknown:
        raise TypeError(
            f"{cls.__module__}.{cls.__name__} does not declare {unknown}; the installed "
            "fla probably renamed or removed it, and passing it would be ignored."
        )
    return kwargs


_DYNAMO_PATCHED = False


def _exclude_running_sign_from_dynamo():
    """Hide ``running_sign`` from dynamo so the rest of a signed layer compiles.

    ``running_sign`` computes the sign gauge as an integer parity prefix sum
    (``(s < 0).to(int32).cumsum(1)``). Inductor promotes that scan to int64 and
    then fails to generate code for it::

        Error in codegen for ComputedBuffer(... SplitScan, dtype=torch.int64,
                                            combine_fn=cumsum ...)
        InductorError: TypeError: list indices must be integers

    which aborts compilation of the WHOLE layer. Only the signed layers have it,
    which is why plain KDA compiles untouched. Excluding just this function
    leaves it in eager -- it is a cheap elementwise scan carrying no gradient --
    while the gate/projection epilogues around it still fuse.

    Patched in EVERY module that has it: panini imports the symbol by value
    (``from ...signed_kda_layer import running_sign``), so rebinding only the
    defining module would leave panini's copy compiled and still failing. Which
    modules exist depends on the fla on the path (see the imports above).
    """
    global _DYNAMO_PATCHED
    if _DYNAMO_PATCHED:
        return
    import importlib

    for name in (
        "fla.layers.complex_kda_layer",
        "fla.layers.signed_kda_layer",
        "fla.layers.panini_delta_attention",
    ):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "running_sign"):
            mod.running_sign = torch._dynamo.disable(mod.running_sign)
    _DYNAMO_PATCHED = True


class _FLALayerWrapper(MegatronModule):
    """Common plumbing for a self-contained FLA layer used as a Megatron mixer."""

    _NAME = "fla"
    # The layer takes FLA's context-parallel context (cp_context) through its kwargs.
    _FLA_CP = False
    # Where the layer comes from, for the error when it is missing.
    _NEEDS = "fla.layers.*"
    # The layer takes ``cu_seqlens`` (document resets for packed THD batches).
    _PACKED = False

    @staticmethod
    def _fla_class():
        raise NotImplementedError

    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: int = None,
        pg_collection: ProcessGroupCollection = None,
        name: str | None = None,
        **kwargs,
    ):
        if self._fla_class() is None:
            raise ImportError(
                f"{self._NAME} needs {self._NEEDS}, which the fla on the path does not "
                "carry. Prepend a checkout that does to PYTHONPATH (ComplexKDA for "
                "kda/signed_kda, the legacy flash-linear-attention for panini)."
            )
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        self.tp_group = pg_collection.tp if pg_collection is not None else None
        cp = pg_collection.cp.size() if pg_collection is not None else 1
        # Context parallelism (fla_cp.py): layers with FLA's CP kernels (_FLA_CP) run their
        # contiguous piece of the token stream and pass the recurrent state across ranks; the
        # others run the whole gathered sequence on every rank (correct, not split).
        self.cp_group = pg_collection.cp if cp > 1 else None

        head_dim = config.linear_key_head_dim
        num_heads = config.linear_num_key_heads
        if head_dim * num_heads != config.hidden_size:
            raise ValueError(
                f"{self._NAME}: linear_key_head_dim ({head_dim}) * "
                f"linear_num_key_heads ({num_heads}) = {head_dim * num_heads} must "
                f"equal hidden_size ({config.hidden_size}); the FLA layers assume "
                "the head grid tiles the model dim exactly (expand_v scales V only)."
            )

        self.layer = self._build(config, head_dim, num_heads, layer_number)
        megatron_init_fla_layer(self.layer, config)
        self._retune_signed_gate_init(config, layer_number)
        # Match the surrounding model's dtype; the FLA layers build in fp32.
        if config.bf16:
            self.layer = self.layer.to(torch.bfloat16)
        elif config.fp16:
            self.layer = self.layer.to(torch.float16)
        # One full copy per TP rank (no-op at TP=1).
        replicate_across_tp(self.layer, self.tp_group)

        # Compile the BOUND FORWARD, not the module. torch.compile(module)
        # returns an OptimizedModule that prefixes every state_dict key with
        # "_orig_mod.", which would rename every parameter in the checkpoint and
        # break loading against runs captured without it. Compiling the method
        # leaves self.layer -- and therefore the state_dict -- untouched.
        if getattr(config, "linear_fla_torch_compile", False):
            _exclude_running_sign_from_dynamo()
            self._fla_forward = torch.compile(self.layer.forward)
        else:
            self._fla_forward = self.layer.forward

    def _build(self, config, head_dim, num_heads, layer_number):
        raise NotImplementedError

    def _retune_signed_gate_init(self, config, layer_number):
        """Apply neg_frac / fproj_scale on top of FLA's "shipped" init.

        FLA bundles sign diversity and gate input-dependence into one
        ``gate_init_style`` switch: "shipped" is (neg_frac 0.0, scale 1.0) and
        "spread" is (neg_frac 0.5, scale 0.1). Nothing in between is reachable,
        and the two effects push in opposite directions -- diverse signs need a
        damped f_proj to survive, but damping f_proj is exactly what removes the
        gate's input-dependence. ``_build`` therefore requests "shipped"
        whenever either knob is set, and the requested values are applied here.

        Done in the wrapper, not in FLA, so the vendored checkout stays at its
        committed state.

        The sign mask uses its OWN generator seeded from the layer index, not
        the global RNG: every data-parallel rank builds the model independently
        and must land on identical weights, and the mask must also be stable
        across a restart-from-scratch.
        """
        neg_frac = getattr(config, "linear_fla_gate_neg_frac", None)
        scale = getattr(config, "linear_fla_gate_fproj_scale", None)
        if neg_frac is None and scale is None:
            return
        layer = self.layer
        if not hasattr(layer, "dt_bias") or not hasattr(layer, "f_proj"):
            raise ValueError(
                f"{self._NAME}: linear_fla_gate_neg_frac / _fproj_scale need a "
                "layer with dt_bias and f_proj (the signed FLA layers)."
            )
        with torch.no_grad():
            if scale is not None:
                layer.f_proj[-1].weight.mul_(scale)
            if neg_frac is not None:
                if not 0.0 <= neg_frac <= 1.0:
                    raise ValueError(f"neg_frac must be in [0, 1], got {neg_frac}")
                g = torch.Generator(device="cpu").manual_seed(1234 + int(layer_number or 0))
                mask = torch.rand(layer.dt_bias.shape, generator=g) < neg_frac
                sign = torch.where(mask, -1.0, 1.0).to(layer.dt_bias.device)
                # "shipped" dt_bias is all-positive, so |dt_bias| is the intended
                # magnitude and the sign is free to set independently.
                layer.dt_bias.copy_(layer.dt_bias.abs() * sign)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        **kwargs,
    ):
        """[s, b, h] -> [s, b, h]; returns (output, bias=None)."""
        kw = {}
        if packed_seq_params is not None:
            if not self._PACKED:
                raise NotImplementedError(f"{self._NAME} does not support packed sequences yet.")
            from megatron.core.ssm.complex_kda import fla_cu_seqlens

            kw["cu_seqlens"] = fla_cu_seqlens(hidden_states, packed_seq_params, self._NAME)
        hidden_states = gather_replicated_input(hidden_states, self.config, self.tp_group)
        if self.cp_group is not None:
            if kw:
                raise NotImplementedError(
                    f"{self._NAME}: packed sequences with context parallelism are not wired yet"
                )
            out = self._forward_cp(hidden_states)
            return scatter_replicated_output(out, self.config, self.tp_group), None
        # Megatron is sequence-first, FLA is batch-first.
        x = hidden_states.transpose(0, 1).contiguous()
        # attention_mask is Megatron's causal mask; these layers are causal by
        # construction and expect a [B, T] *padding* mask, so pass None.
        out, _, _ = self._fla_forward(hidden_states=x, attention_mask=None, use_cache=False, **kw)
        out = out.transpose(0, 1).contiguous()
        return scatter_replicated_output(out, self.config, self.tp_group), None

    def _forward_cp(self, hidden_states: Tensor) -> Tensor:
        """[s / cp, b, h] in Megatron's CP layout -> the same, through the layer."""
        from megatron.core.ssm import fla_cp

        group, batch = self.cp_group, hidden_states.shape[1]
        if self._FLA_CP:
            x = fla_cp.to_stream(hidden_states, group)
            ctx = fla_cp.cp_context(group, batch, hidden_states.shape[0] * group.size(),
                                    self.config.linear_conv_kernel_dim, x.device)
            out, _, _ = self._fla_forward(hidden_states=x, attention_mask=None, use_cache=False,
                                          cp_context=ctx)
            return fla_cp.from_stream(out, group, batch)
        x = fla_cp.full_sequence(hidden_states, group).transpose(0, 1).contiguous()
        out, _, _ = self._fla_forward(hidden_states=x, attention_mask=None, use_cache=False)
        return fla_cp.local_sequence(out.transpose(0, 1).contiguous(), group)


class KDA(_FLALayerWrapper):
    """FLA KimiDeltaAttention -- plain Kimi Linear KDA, the unsigned baseline.

    The reference point for ``signed_kda``: same per-key-dim vector forget gate
    and the same low-rank ``f_proj``/``g_proj`` bottlenecks, but alpha stays in
    (0, 1] via the shipped ``-exp(A_log) * softplus(u)`` gate instead of being
    widened to [-1, 1]. ``lower_bound`` clamps that gate into [lower_bound, 0),
    which is what makes ``safe_gate`` -- the kernel's M=16 TensorCore path --
    legal; the signed layers reach the same path through ``safe_gate_ok``.
    """

    _NAME = "kda"
    _NEEDS = "fla.layers.kda"
    _FLA_CP = True

    @staticmethod
    def _fla_class():
        return _FLAKDA

    def _build(self, config, head_dim, num_heads, layer_number):
        lower_bound = config.linear_fla_lower_bound
        return _FLAKDA(
            **_declared_kwargs(
                _FLAKDA,
                _with_xsa(config, dict(
                    hidden_size=config.hidden_size,
                    expand_v=config.linear_fla_expand_v,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_v_heads=None,
                    mode="chunk",
                    use_short_conv=True,
                    conv_size=config.linear_conv_kernel_dim,
                    conv_bias=False,
                    allow_neg_eigval=config.linear_fla_allow_neg_eigval,
                    # The M=16 path assumes g in [lower_bound, 0); without a bound the
                    # shipped softplus gate is unbounded below and the kernel is unsafe.
                    safe_gate=lower_bound is not None,
                    lower_bound=lower_bound,
                    layer_idx=layer_number,
                    norm_eps=config.layernorm_epsilon,
                )),
            )
        )


class SignedKDA(_FLALayerWrapper):
    """KDA with a signed decay gate and widened beta, as a Megatron mixer.

    Built from the released ``ComplexKimiDeltaAttention`` when the fla on the
    path has it, else from its pre-release form ``SignedKimiDeltaAttention``;
    see the imports. The same layer as the ``complex_kda`` variant, configured
    through the ``linear_fla_*`` fields and the ``linear_*`` head grid, which is
    what earlier runs used -- so those runs stay reproducible and their
    checkpoints stay loadable.

    ``drop_silu`` is passed EXPLICITLY. The legacy layer defaulted it to True
    and this wrapper relied on that; the released layer defaults it to False.
    Leaving it to the default would have silently changed the model of every
    ``signed_kda`` run once the released fla was on the path.
    """

    _NAME = "signed_kda"
    _FLA_CP = True
    _NEEDS = "fla.layers.complex_kda_layer (or legacy fla.layers.signed_kda_layer)"

    @staticmethod
    def _fla_class():
        return _FLASignedKDA

    def _build(self, config, head_dim, num_heads, layer_number):
        kwargs = dict(
            hidden_size=config.hidden_size,
            expand_v=config.linear_fla_expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=None,
            mode="chunk",
            use_short_conv=True,
            conv_size=config.linear_conv_kernel_dim,
            conv_bias=False,
            allow_neg_eigval=config.linear_fla_allow_neg_eigval,
            gate=config.linear_fla_gate,
            output_gate=config.linear_fla_output_gate,
            lower_bound=config.linear_fla_lower_bound,
            # a retune knob implies the neutral base; see _retune_signed_gate_init
            gate_init_style=(
                "shipped"
                if (
                    config.linear_fla_gate_neg_frac is not None
                    or config.linear_fla_gate_fproj_scale is not None
                )
                else config.linear_fla_gate_init_style
            ),
            drop_silu=config.linear_fla_drop_silu,
            layer_idx=layer_number,
            norm_eps=config.layernorm_epsilon,
        )
        if _FLASignedKDA.__name__ == "SignedKimiDeltaAttention":
            # Legacy only; False is the released layer's fixed behaviour.
            kwargs["tie_sign_across_group"] = False
        return _FLASignedKDA(**_declared_kwargs(_FLASignedKDA, _with_xsa(config, kwargs)))


class PaniniDeltaAttention(_FLALayerWrapper):
    """FLA PaniniDeltaAttention (signed KDA + a signed value-index gate)."""

    _NAME = "panini"
    _NEEDS = "fla.layers.panini_delta_attention (legacy flash-linear-attention only)"

    @staticmethod
    def _fla_class():
        return _FLAPanini

    def _build(self, config, head_dim, num_heads, layer_number):
        return _FLAPanini(
            **_declared_kwargs(
                _FLAPanini,
                dict(
                    hidden_size=config.hidden_size,
                    expand_v=config.linear_fla_expand_v,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_v_heads=None,
                    mode="chunk",
                    use_short_conv=True,
                    conv_size=config.linear_conv_kernel_dim,
                    conv_bias=False,
                    allow_neg_eigval=config.linear_fla_allow_neg_eigval,
                    gate=config.linear_fla_gate,
                    v_gate=config.linear_fla_v_gate,
                    output_gate=config.linear_fla_output_gate,
                    lower_bound=config.linear_fla_lower_bound,
                    lower_bound_v=config.linear_fla_lower_bound_v,
                    gate_init_style=config.linear_fla_gate_init_style,
                    v_gate_init_style=config.linear_fla_v_gate_init_style,
                    # Without this the value gauge is re-based in a PYTHON LOOP around
                    # chunk_kda, one launch per value_segment_length(lower_bound_v)
                    # tokens; at the -2.0 default that is 64 launches per layer per
                    # forward at T=4096 and costs ~3.5x plain kda.
                    v_gate_in_kernel=config.linear_fla_v_gate_in_kernel,
                    tie_sign_across_group=False,
                    layer_idx=layer_number,
                    norm_eps=config.layernorm_epsilon,
                ),
            )
        )


class Triadic(_FLALayerWrapper):
    """Triadic delta rule (Triadic GDN / KDA / Complex KDA) as a Megatron mixer; see
    megatron/core/ssm/triadic.py. ``linear_triadic_e`` slices per head, the base by
    ``linear_triadic_base``."""

    _NAME = "triadic"
    _NEEDS = "fla.ops.kda (ComplexKDA)"

    @staticmethod
    def _fla_class():
        from megatron.core.ssm.triadic import HAVE_FLA, TriadicDeltaAttention

        return TriadicDeltaAttention if HAVE_FLA else None

    def _build(self, config, head_dim, num_heads, layer_number):
        from megatron.core.ssm.triadic import TriadicDeltaAttention

        return TriadicDeltaAttention(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            e=config.linear_triadic_e,
            base=config.linear_triadic_base,
            gate=config.linear_fla_gate,
            allow_neg_eigval=config.linear_fla_allow_neg_eigval,
            lower_bound=config.linear_fla_lower_bound,
            conv_size=config.linear_conv_kernel_dim,
            norm_eps=config.layernorm_epsilon,
            layer_idx=layer_number,
            kernel=config.linear_triadic_kernel,
            xsa=getattr(config, "linear_xsa", False),
        )


class PowerRetention(_FLALayerWrapper):
    """Power retention (power attention) as a Megatron mixer; see
    megatron/core/ssm/power_retention.py. The public Triton kernels are the baseline."""

    _NAME = "power"
    _NEEDS = "retention (github.com/m-a-n-i-f-e-s-t/retention) and vidrial"

    @staticmethod
    def _fla_class():
        from megatron.core.ssm.power_retention import HAVE_RETENTION, PowerRetentionAttention

        return PowerRetentionAttention if HAVE_RETENTION else None

    def _build(self, config, head_dim, num_heads, layer_number):
        from megatron.core.ssm.power_retention import PowerRetentionAttention

        return PowerRetentionAttention(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            deg=config.linear_power_deg,
            chunk_size=config.linear_power_chunk_size,
            switch_over_seq_len=config.linear_power_switch_over_seq_len,
            attention_impl=config.linear_power_attention_impl,
            rope=config.linear_power_rope,
            xsa=getattr(config, "linear_xsa", False),
            layer_idx=layer_number,
        )


class KronLSTM(_FLALayerWrapper):
    """KronLSTM (Kronecker delta rule, github.com/NX-AI/KronLSTM; arXiv 2610.26834) as a
    Megatron mixer, from the layer vendored in our flash-linear-attention fork.

    The forget gate is ``linear_fla_gate``: "softplus" is KronLSTM's own (0, 1] gate, the signed
    gates give f in [-1, 1] with Complex KDA's parameterisation and init, applied in the op as a
    sign gauge on q and k. ``linear_fla_allow_neg_eigval`` gives beta in [0, 2]. Packed
    documents reset the recurrence and the short convolution through ``cu_seqlens``. Context
    parallelism splits the token stream (fla_cp.py): the op relays the recurrent state between
    ranks around its scans (fla/ops/kronlstm/cp.py), the rest of the layer runs in parallel.
    """

    _NAME = "kronlstm"
    _NEEDS = "fla.layers.kronlstm (our flash-linear-attention fork)"
    _PACKED = True
    _FLA_CP = True

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # The layer's q, k and v share ONE projection weight ([q; k; v] stacked). Muon-type
        # optimizers orthogonalize per matrix: tag it as a fused QKV, as Megatron tags attention's
        # linear_qkv, so (angular_)muon_split_qkv orthogonalizes q, k and v separately -- as
        # complex_kda's three projections are -- not as one 3h x h matrix.
        w = self.layer.qkv_proj.weight
        w.is_qkv = True
        w.qkv_split_shapes = [self.layer.key_dim, self.layer.key_dim, self.layer.value_dim]

    @staticmethod
    def _fla_class():
        return _FLAKronLSTM

    def _build(self, config, head_dim, num_heads, layer_number):
        gate = config.linear_fla_gate
        if gate not in ("softplus", "signed_sigmoid2", "signed_tanh"):
            raise ValueError(
                f"kronlstm: linear_fla_gate must be softplus, signed_sigmoid2 or signed_tanh, got {gate!r}"
            )
        v_head_dim = int(head_dim * config.linear_fla_expand_v)
        return _FLAKronLSTM(
            **_declared_kwargs(
                _FLAKronLSTM,
                dict(
                    hidden_size=config.hidden_size,
                    k_head_dim=head_dim,
                    v_head_dim=v_head_dim,
                    num_k_heads=num_heads,
                    num_v_heads=num_heads,
                    mode="chunk",
                    use_short_conv=True,
                    conv_size=config.linear_conv_kernel_dim,
                    conv_bias=False,
                    normalize_v=config.linear_kronlstm_normalize_v,
                    allow_neg_eigval=config.linear_fla_allow_neg_eigval,
                    use_target_gate=config.linear_kronlstm_target_gate,
                    decouple_erase=config.linear_kronlstm_decouple_erase,
                    neg_eigval_init_bias=config.linear_kronlstm_neg_eigval_init_bias,
                    unit_write=config.linear_kronlstm_unit_write,
                    forget_offset=config.linear_kronlstm_forget_offset,
                    forget_angle=config.linear_kronlstm_forget_angle,
                    erase_angle=config.linear_kronlstm_erase_angle,
                    signed_write=config.linear_kronlstm_signed_write,
                    output_gate_rank={"linear": "full", "lowrank": "lowrank"}[
                        config.linear_kronlstm_output_gate
                    ],
                    decay_gate="vector",
                    gate=gate,
                    lower_bound=config.linear_fla_lower_bound,
                    layer_idx=layer_number,
                    norm_eps=config.layernorm_epsilon,
                ),
            )
        )
