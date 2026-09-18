# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Complex KDA: Kimi Delta Attention with signed state transitions.

The transition applied at each step is

    M = (I - beta k k^T) Diag(alpha)

and Complex KDA extends the range of BOTH factors so that M may have negative
and complex eigenvalues:

    alpha in [-1, 1]   the channel-wise decay gate  (``linear_gate_activation``)
    beta  in [0, 2]    the Householder rate         (``linear_beta_max``)

With alpha >= 0 and beta <= 1 -- the defaults of KDA and of GatedDeltaNet -- M
is a product of two positive semi-definite matrices and its spectrum stays on
the non-negative real axis. Extending either factor lets a channel reflect
rather than only contract; extending both admits rotations, which is what a
[0, 1]-constrained linear RNN provably cannot represent (Grazzi et al.).

``linear_beta_max`` is shared with GatedDeltaNet and already carries the beta
half. The gate fields on ``TransformerConfig`` carry the alpha half, which is
what this variant adds.

IMPLEMENTATION. The recurrence and its kernels come from flash-linear-attention
(``fla.layers.complex_kda_layer.ComplexKimiDeltaAttention``), the same way
``gated_delta_net.py`` takes ``chunk_gated_delta_rule`` from fla. This module is
the adapter between that layer and Megatron's self-attention interface:
Megatron passes (sequence, batch, hidden) and expects an ``(output, bias)``
pair; fla takes (batch, sequence, hidden) and returns the output alone.

TENSOR PARALLELISM IS NOT SUPPORTED. The fla layer owns its projections as
plain ``nn.Linear`` and knows nothing about column/row parallel linears, so
there is nothing to shard. ``__init__`` refuses ``tensor_model_parallel_size >
1`` rather than silently training a model whose mixer is replicated while the
rest of the layer is sharded. Supporting TP means reimplementing the
projections against Megatron's parallel linears, at which point the layer is no
longer bit-for-bit the one the reference implementation trains -- which is a
deliberate trade, not an oversight.
"""

from dataclasses import dataclass

try:
    from fla.layers.attn import Attention as FLAAttention
    from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention

    HAVE_FLA = True
except ImportError:
    ComplexKimiDeltaAttention = None
    FLAAttention = None
    HAVE_FLA = False

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class ComplexKDASubmodules:
    """Submodules of :class:`ComplexKDA`.

    Empty, and that is the honest shape: the fla layer constructs its own
    projections, norms and short convolutions. The dataclass exists so the
    spec looks like its neighbours' and so a future tensor-parallel
    implementation has somewhere to put ``in_proj``/``out_proj`` without
    changing the spec's type.
    """


def _head_dim(config: TransformerConfig) -> int:
    """Head dimension for the mixer.

    ``linear_key_head_dim`` when set, as for the other linear variants;
    otherwise the model's own, so a config that never mentions linear
    attention still builds.
    """
    if config.linear_key_head_dim is not None:
        return config.linear_key_head_dim
    if config.kv_channels is not None:
        return config.kv_channels
    return config.hidden_size // config.num_attention_heads


class _MegatronSelfAttentionAdapter(MegatronModule):
    """Shared plumbing: layout, the ``(output, bias)`` contract, TP refusal."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        if not HAVE_FLA:
            raise ImportError(
                "Complex KDA needs flash-linear-attention; install `fla` or "
                "choose another experimental_attention_variant"
            )
        if config.tensor_model_parallel_size != 1:
            raise NotImplementedError(
                "experimental_attention_variant='complex_kda' supports "
                "tensor_model_parallel_size == 1 only; got "
                f"{config.tensor_model_parallel_size}. The fla layer owns its "
                "projections and they are not parallel linears, so a larger TP "
                "would replicate the mixer while sharding the rest."
            )
        self.layer = None  # set by the subclass

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        """Megatron's self-attention signature, over an fla layer.

        ``rotary_pos_emb`` is accepted and DROPPED. Megatron hands every
        self-attention module a rotary embedding; this family carries position
        in the recurrence and does not use one. Taking the argument and
        ignoring it is the honest form of that -- the alternative is letting
        Megatron believe it applied a rotary embedding the layer never saw.
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "complex_kda does not support packed_seq_params: the varlen "
                "metadata would have to reach the recurrence, and silently "
                "ignoring it would run the sequences into each other"
            )
        # (s, b, h) -> (b, s, h)
        x = hidden_states.transpose(0, 1).contiguous()
        out = self.layer(x)
        y = out[0] if isinstance(out, tuple) else out
        # Megatron's TransformerLayer expects (output, bias); the bias is
        # folded into the projections here, so there is none to return.
        return y.transpose(0, 1).contiguous(), None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Replicated tensors: TP is 1, so nothing here is sharded."""
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True), prefix, None, sharded_offsets
        )


class ComplexKDA(_MegatronSelfAttentionAdapter):
    """The Complex-KDA mixer, as a Megatron self-attention module."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: ComplexKDASubmodules | None = None,
        layer_number: int = 1,
        attn_mask_type=None,
        **kwargs,
    ):
        super().__init__(config=config)
        head_dim = _head_dim(config)
        num_heads = config.hidden_size // head_dim
        if num_heads * head_dim != config.hidden_size:
            raise ValueError(
                f"complex_kda: hidden_size {config.hidden_size} is not a whole "
                f"number of heads of width {head_dim}"
            )
        self.layer = ComplexKimiDeltaAttention(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            # beta = sigmoid(x) * linear_beta_max; the layer spells the
            # extended range as a flag rather than a bound.
            allow_neg_eigval=config.linear_beta_max == 2.0,
            gate=config.linear_gate_activation,
            gate_init_style=config.linear_gate_init_style,
            lower_bound=config.linear_gate_lower_bound,
            output_gate=config.linear_output_gate,
            drop_silu=config.linear_drop_qkv_silu,
            conv_size=config.linear_conv_kernel_dim,
            norm_eps=config.layernorm_epsilon,
            layer_idx=layer_number - 1,
        )


class ComplexKDAHybridAttention(_MegatronSelfAttentionAdapter):
    """Output-gated, NoPE full attention, for the attention layers of a hybrid.

    NOT Megatron's own attention, and the difference is not cosmetic. Kimi
    Linear -- the published hybrid this arrangement follows -- interleaves full
    attention WITHOUT a position embedding, because the linear layers already
    carry position; and the attention is output-gated, as in Qwen3-Next. Both
    depart from Megatron's standard attention, and the gate alone is
    ``hidden_size ** 2`` parameters per attention layer, so substituting the
    standard module builds a smaller model under the same name. Measured on the
    47M rung: 46,819,446 parameters against the parameter-matched 47,265,270.

    Selected by ``linear_hybrid_attention='gated_nope'``; the default leaves the
    hybrid's attention layers as Megatron's.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: ComplexKDASubmodules | None = None,
        layer_number: int = 1,
        attn_mask_type=None,
        **kwargs,
    ):
        super().__init__(config=config)
        head_dim = _head_dim(config)
        # No `head_dim` and no `norm_eps`: fla's Attention derives the head
        # width from hidden_size // num_heads and has no norm epsilon of its
        # own. Passing either is a TypeError, which is how the reference
        # implementation's own test suite found it.
        self.layer = FLAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_query_groups or config.num_attention_heads,
            qkv_bias=config.add_qkv_bias,
            qk_norm=config.qk_layernorm,
            output_gate=True,
            use_rope=False,
            layer_idx=layer_number - 1,
        )
