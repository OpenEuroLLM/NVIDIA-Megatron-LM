# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""attention_logit_scale: the factor on the softmax-attention softmax scale (recall surgery)."""

import math

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _cfg(**kw):
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kw)


def test_default_leaves_softmax_scale_unset():
    assert _cfg().softmax_scale is None


def test_scales_the_default():
    c = _cfg(attention_logit_scale=1.25, kv_channels=32)
    assert math.isclose(c.softmax_scale, 1.25 / math.sqrt(32))


def test_scales_an_explicit_softmax_scale():
    assert math.isclose(_cfg(attention_logit_scale=2.0, softmax_scale=0.1).softmax_scale, 0.2)


def test_rejects_non_positive():
    with pytest.raises(ValueError):
        _cfg(attention_logit_scale=0.0)
