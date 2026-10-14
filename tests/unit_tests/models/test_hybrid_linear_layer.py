# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""HybridModel `L` positions: a linear mixer chosen by experimental_attention_variant.

Everything here is pattern/spec/config/FLOPs logic and runs without a GPU.
"""

import sys

import pytest

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_hybrid_linear_layer_spec,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_layer_counts,
    parse_hybrid_pattern,
    validate_segment_layers,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer

_BASE = dict(
    num_layers=2,
    hidden_size=256,
    num_attention_heads=4,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
)


def _config(variant, **kw):
    return TransformerConfig(
        **{**_BASE, "experimental_attention_variant": variant, "is_hybrid_model": True, **kw}
    )


class TestPattern:
    def test_counts(self):
        counts = get_hybrid_layer_counts("L-L-*-")
        assert counts[Symbols.LINEAR] == 2
        assert counts[Symbols.MLP] == 3
        assert counts[Symbols.ATTENTION] == 1

    def test_linear_and_dsa_rejected(self):
        # Both are configured through experimental_attention_variant.
        with pytest.raises(ValueError, match="linear .L. and DSA .D."):
            validate_segment_layers("LD")
        with pytest.raises(ValueError, match="linear .L. and DSA .D."):
            parse_hybrid_pattern("L-D-")


class TestLayerSpec:
    @pytest.mark.parametrize("variant", ["gated_delta_net", "mlstm"])
    def test_fused_norm_mixer_gets_no_input_norm(self, variant):
        spec = get_hybrid_linear_layer_spec(_config(variant))
        assert spec.module is TransformerLayer
        assert spec.submodules.input_layernorm is IdentityOp
        # The mixer is exactly the one the GPT path builds.
        gpt = get_experimental_attention_variant_module_spec(_config(variant))
        assert spec.submodules.self_attention.module is gpt.module

    def test_self_contained_mixer_gets_input_norm(self):
        spec = get_hybrid_linear_layer_spec(_config("kda"))
        assert spec.submodules.input_layernorm is not IdentityOp

    def test_no_mlp_inside_the_layer(self):
        # HybridModel places MLPs with its own '-'/'E' symbols.
        spec = get_hybrid_linear_layer_spec(_config("mlstm"))
        assert spec.submodules.mlp is IdentityOp
        assert spec.submodules.pre_mlp_layernorm is IdentityOp

    @pytest.mark.parametrize("variant", [None, "dsa"])
    def test_non_linear_variant_rejected(self, variant):
        cfg = TransformerConfig(**_BASE)
        cfg.experimental_attention_variant = variant
        with pytest.raises(ValueError, match="'L' needs experimental_attention_variant"):
            get_hybrid_linear_layer_spec(cfg)


class TestConfig:
    @pytest.mark.parametrize("variant", ["gated_delta_net", "mlstm"])
    def test_hybrid_model_needs_no_linear_attention_freq(self, variant):
        _config(variant)  # the pattern places the layers

    @pytest.mark.parametrize("variant", ["gated_delta_net", "mlstm"])
    def test_gpt_model_still_needs_linear_attention_freq(self, variant):
        with pytest.raises(AssertionError, match="linear_attention_freq must be set"):
            _config(variant, is_hybrid_model=False)


def _args(*flags):
    from megatron.training.arguments import parse_args

    argv = sys.argv
    sys.argv = [
        "pretrain_hybrid.py",
        "--hidden-size",
        "1024",
        "--num-attention-heads",
        "8",
        "--seq-length",
        "4096",
        "--max-position-embeddings",
        "4096",
        "--micro-batch-size",
        "1",
        "--swiglu",
        "--ffn-hidden-size",
        "2816",
        "--linear-key-head-dim",
        "128",
        "--linear-value-head-dim",
        "128",
        "--linear-num-key-heads",
        "4",
        "--linear-num-value-heads",
        "8",
        *flags,
    ]
    try:
        args = parse_args(ignore_unknown_args=True)
    finally:
        sys.argv = argv
    args.padded_vocab_size = 32000
    args.kv_channels = args.kv_channels or 128
    return args


class TestArguments:
    def test_linear_symbol_needs_linear_variant(self):
        from megatron.training.argument_utils import _check_hybrid_linear_layers

        with pytest.raises(ValueError, match="uses 'L'"):
            _check_hybrid_linear_layers(_args("--hybrid-layer-pattern=L-*-"))

    def test_linear_attention_freq_rejected_in_hybrid(self):
        from megatron.training.argument_utils import _check_hybrid_linear_layers

        args = _args(
            "--hybrid-layer-pattern=L-*-",
            "--experimental-attention-variant",
            "mlstm",
            "--linear-attention-freq",
            "2",
        )
        with pytest.raises(ValueError, match="--linear-attention-freq has no effect"):
            _check_hybrid_linear_layers(args)

    def test_valid_linear_pattern_accepted(self):
        from megatron.training.argument_utils import _check_hybrid_linear_layers

        _check_hybrid_linear_layers(
            _args("--hybrid-layer-pattern=L-*-", "--experimental-attention-variant", "mlstm")
        )


class TestFlops:
    def test_linear_layers_priced_like_the_gpt_path(self):
        """An `L` adds 3 (fwd+bwd) x 2 (FMA) x tokens x the variant's per-token FMAs."""
        from megatron.training.training import num_floating_point_operations

        batch = 2
        flags = ("--experimental-attention-variant", "mlstm")
        with_l = num_floating_point_operations(
            _args("--hybrid-layer-pattern=L-L-*-", *flags), batch
        )
        without_l = num_floating_point_operations(
            _args("--hybrid-layer-pattern=--*-", *flags), batch
        )
        # mLSTM, GQA (4 key heads, 8 value heads), no conv, chunk size 128 (default).
        h, dqk, dv, kh, vh, chunk = 1024, 128, 128, 4, 8, 128
        qk_dim, v_dim = dqk * kh, dv * vh
        fmas = (
            h * (2 * qk_dim + 2 * v_dim + 2 * vh)
            + vh * (2 * dqk * dv + chunk * (dqk + dv) // 2)
            + h * v_dim
        )
        tokens = batch * 4096
        assert with_l - without_l == 2 * (3 * 2 * tokens * fmas)
