# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Sliding-window attention layers (`W`) in a HybridModel; `*` stays global."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

SEQ = 24


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _model(pattern, window=(3, 0), backend=AttnBackend.unfused):
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        attention_backend=backend,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        window_size=window,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _sensitivity_to_first_token(model):
    """Max |logit change| per position when token 0 changes."""
    ids = torch.randint(1, 128, (1, SEQ), device="cuda")
    other = ids.clone()
    other[0, 0] = 0
    with torch.no_grad():
        a = model(ids, position_ids=None, attention_mask=None)
        b = model(other, position_ids=None, attention_mask=None)
    return (a - b).abs().amax(-1)[0]  # [SEQ]


def test_pattern_sets_the_per_layer_window_list():
    model = _model("*-W-W-")
    assert model.config.window_attn_skip_freq == [0, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("backend", [AttnBackend.unfused, AttnBackend.auto])
def test_window_limits_the_receptive_field(backend):
    window = 3  # each query sees itself and the 3 tokens before
    local = _model("W-", window=(window, 0), backend=backend)
    diff = _sensitivity_to_first_token(local)
    assert diff[: window + 1].max() > 0
    torch.testing.assert_close(diff[window + 1 :], torch.zeros_like(diff[window + 1 :]))
    # A global `*` layer in the same model still carries token 0 to the end.
    mixed = _model("W-*-", window=(window, 0), backend=backend)
    assert _sensitivity_to_first_token(mixed)[-1] > 0


def test_w_needs_a_window_size():
    with pytest.raises(ValueError, match="window-size"):
        _model("W-", window=None)


def test_w_trains():
    model = _model("W-*-W-")
    ids = torch.randint(0, 128, (2, SEQ), device="cuda")
    model(ids, position_ids=None, attention_mask=None).float().logsumexp(-1).mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
