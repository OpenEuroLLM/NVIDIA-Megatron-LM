# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""HybridModel `T` positions: a triadic linear mixer next to the `L` variant (e.g. complex_kda).

Pattern / config / FLOPs logic only; runs without a GPU.
"""

import types

import pytest

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_layer_counts,
    parse_hybrid_pattern,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.models.test_hybrid_linear_layer import _BASE, _args


def test_counts_and_validity():
    counts = get_hybrid_layer_counts("L-T-L-*-")
    assert counts[Symbols.TRIADIC] == 1
    assert counts[Symbols.LINEAR] == 2
    assert counts[Symbols.ATTENTION] == 1
    parse_hybrid_pattern("T-L-L-L-L-L-")  # a pure-linear stack is valid


def test_triadic_and_dsa_rejected():
    with pytest.raises(ValueError, match="linear .L. and DSA .D."):
        parse_hybrid_pattern("T-D-")


def test_triadic_config_is_a_copy():
    cfg = TransformerConfig(
        **_BASE, experimental_attention_variant="complex_kda", is_hybrid_model=True
    )
    stack = types.SimpleNamespace(config=cfg)
    tri = HybridStack._triadic_config(stack)
    assert tri.experimental_attention_variant == "triadic"
    assert cfg.experimental_attention_variant == "complex_kda"  # the stack's is untouched
    assert tri.hidden_size == cfg.hidden_size and tri.linear_triadic_e == cfg.linear_triadic_e
    assert HybridStack._triadic_config(stack) is tri  # built once


def test_triadic_layers_priced_as_triadic():
    """A `T` costs what an `L` costs under experimental_attention_variant=triadic."""
    from megatron.training.training import num_floating_point_operations

    batch = 2
    ckda = ("--experimental-attention-variant", "complex_kda")
    tri = ("--experimental-attention-variant", "triadic", "--linear-triadic-base", "ckda")
    t_cost = num_floating_point_operations(
        _args("--hybrid-layer-pattern=T-*-", *ckda), batch
    ) - num_floating_point_operations(_args("--hybrid-layer-pattern=-*-", *ckda), batch)
    l_cost = num_floating_point_operations(
        _args("--hybrid-layer-pattern=L-*-", *tri), batch
    ) - num_floating_point_operations(_args("--hybrid-layer-pattern=-*-", *tri), batch)
    assert t_cost == l_cost > 0


def test_mixed_pattern_arguments_accepted():
    from megatron.training.argument_utils import _check_hybrid_linear_layers

    _check_hybrid_linear_layers(
        _args("--hybrid-layer-pattern=*-T-L-L-L-L-", "--experimental-attention-variant", "complex_kda")
    )
