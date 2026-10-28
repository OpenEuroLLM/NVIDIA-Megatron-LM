# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Sandwich / peri-LN post-sublayer normalization (post_sublayer_norm)."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

PATTERN = "N*-C-P-*E"
EXTRA = dict(
    engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=500, engram_embed_dim=8,
    pkm_heads=2, pkm_num_keys=16, pkm_key_dim=16, pkm_topk=4,
    num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64,
    moe_token_dispatcher_type="alltoall",
)


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _model(**kw):
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
        **EXTRA,
        **kw,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=32,
        hybrid_layer_pattern=PATTERN,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _post_norms(model):
    return {n: m for n, m in model.named_modules() if n.endswith(("post_attn_norm", "post_mlp_norm", "post_norm"))}


def test_every_sublayer_output_is_normalized():
    model = _model(post_sublayer_norm=True)
    norms = _post_norms(model)
    # N, *, -, C, -, P, -, *, E: one norm per sublayer
    assert len(norms) == len(PATTERN), sorted(norms)
    seen = {}
    for name, m in norms.items():
        m.register_forward_hook(
            lambda mod, i, o, name=name: seen.__setitem__(name, (i[0].float(), o.float()))
        )
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    model(ids, position_ids=None, attention_mask=None)
    eps = model.config.layernorm_epsilon
    for name, (x, y) in seen.items():
        # RMSNorm with unit gain: rms(y)^2 = ms(x) / (ms(x) + eps) (Engram's output is
        # tiny at init, so the eps term is visible there)
        ms = x.pow(2).mean(-1)
        expected = (ms / (ms + eps)).sqrt()
        torch.testing.assert_close(y.pow(2).mean(-1).sqrt(), expected, rtol=1e-2, atol=1e-3, msg=name)


def test_off_by_default():
    assert not _post_norms(_model())


@pytest.mark.parametrize(
    "scheme",
    [{}, {"mhc": True, "mhc_expansion_rate": 4, "mhc_routed_topk": 2},
     {"mhc": True, "mhc_expansion_rate": 2, "mhc_sinkhorn_iters": 4},
     {"attn_res": True, "attn_res_num_blocks": 3}, {"ngpt": True},
     {"depth_memory": True}, {"depth_memory": True, "depth_memory_impl": "torch"},
     {"depth_memory": True, "depth_memory_impl": "torch", "ngpt": True, "depth_memory_lerp": True}],
)
def test_composes_with_residual_schemes_and_trains(scheme):
    model = _model(post_sublayer_norm=True, **scheme)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 128, (4, 16), device="cuda")
    losses = []
    for _ in range(5):
        logits = model(ids[:, :-1], position_ids=None, attention_mask=None)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 128), ids[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(p.grad is not None for n, p in model.named_parameters() if "post_" in n)
