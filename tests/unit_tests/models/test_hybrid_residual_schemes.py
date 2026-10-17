# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""mHC / Block AttnRes / nGPT residuals on a HybridModel.

Every HybridStack layer type takes part: the TransformerLayer slots (`*`, `L`, `C`,
`-`) through their own per-sublayer schemes, Engram (`N`) through its
ResidualSchemeAdapter.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.residual_scheme_adapter import ResidualSchemeAdapter
from tests.unit_tests.test_utilities import Utils

PATTERN = "N-C-*-C-*-"
VOCAB = 128

SCHEMES = {
    "mhc": dict(mhc=True, mhc_expansion_rate=2, mhc_sinkhorn_iters=4),
    "mhc_lerp": dict(
        mhc=True, mhc_expansion_rate=2, mhc_sinkhorn_iters=4, mhc_lerp=True, ngpt=True
    ),
    "mhc_routed": dict(mhc=True, mhc_expansion_rate=4, mhc_routed_topk=2),
    "mhc_routed_keys_lerp": dict(
        mhc=True, mhc_expansion_rate=4, mhc_routed_topk=2, mhc_routed_key_dim=8,
        mhc_lerp=True, ngpt=True, ngpt_lerp_normalization="analytic",
    ),
    "attn_res": dict(attn_res=True, attn_res_num_blocks=3),
    "ngpt": dict(ngpt=True),
}


def _model(**scheme):
    config = TransformerConfig(
        num_layers=len(PATTERN),
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=500,
        engram_embed_dim=8,
        **scheme,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=32,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("name", list(SCHEMES))
def test_scheme_trains_through_every_layer_type(name):
    model = _model(**SCHEMES[name])
    ids = torch.randint(0, VOCAB, (2, 24), device="cuda")
    logits = model(ids, position_ids=None, attention_mask=None)
    assert logits.shape == (2, 24, VOCAB) and torch.isfinite(logits).all()
    logits.float().logsumexp(-1).mean().backward()

    # Engram carries the scheme too, and every scheme parameter receives a gradient.
    engram = model.decoder.layers[0]
    assert isinstance(engram.residual_scheme, ResidualSchemeAdapter)
    scheme_params = [
        (n, p)
        for n, p in model.named_parameters()
        if any(k in n for k in ("mhc", "attnres", "ngpt", "residual_scheme"))
    ]
    assert scheme_params
    missing = [n for n, p in scheme_params if p.grad is None]
    assert not missing, missing


def test_single_sublayer_layers_build_only_their_scheme():
    model = _model(**SCHEMES["mhc"])
    layers = model.decoder.layers
    conv, mlp = layers[2], layers[3]  # `C`, `-`
    assert conv.mhc_attn_mixer is not None and conv.mhc_mlp_mixer is None
    assert mlp.mhc_attn_mixer is None and mlp.mhc_mlp_mixer is not None


def test_scheme_params_follow_sequence_parallel_flag():
    model = _model(**SCHEMES["attn_res"])
    for n, p in model.named_parameters():
        if "attnres" in n or "residual_scheme" in n:
            assert getattr(p, "sequence_parallel", None) is False, n
