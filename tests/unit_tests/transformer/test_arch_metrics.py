# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Architecture diagnostics (megatron/core/transformer/arch_metrics.py)."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig, arch_metrics
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

VOCAB = 128


@pytest.fixture(autouse=True)
def setup():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    arch_metrics.reset()
    arch_metrics.set_active(False)
    yield
    arch_metrics.set_active(False)
    arch_metrics.reset()
    Utils.destroy_model_parallel()


def _model(pattern, **kw):
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        **kw,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=32,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _step(model):
    ids = torch.randint(0, VOCAB, (2, 16), device="cuda")
    logits = model(ids, position_ids=None, attention_mask=None)
    logits.float().logsumexp(-1).mean().backward()


ENGRAM = dict(engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=500, engram_embed_dim=8)
PKM = dict(pkm_heads=2, pkm_num_keys=16, pkm_key_dim=16, pkm_topk=4)
MOE = dict(
    num_moe_experts=4,
    moe_router_topk=2,
    moe_ffn_hidden_size=64,
    moe_token_dispatcher_type="alltoall",
    moe_shared_expert_intermediate_size=64,
    moe_shared_expert_gate=True,
)


def test_tracker_averages_per_layer_and_summarizes_shares():
    # CUDA tensors, as at every call site: collect() all-reduces them (NCCL).
    cuda = dict(device="cuda")
    arch_metrics.record("a", 1, 3, torch.tensor(6.0, **cuda), 3)
    arch_metrics.record("a", 1, 3, torch.tensor(2.0, **cuda), 1)
    arch_metrics.record("a", 3, 3, torch.tensor(1.0, **cuda), 1)
    arch_metrics.record_share("s", 2, 3, torch.tensor([[1.0, 0.0], [1.0, 0.0]], **cuda))
    arch_metrics.record_union("u", 2, 3, 10, torch.tensor([1, 1, 4], **cuda))
    out = arch_metrics.collect()
    assert out["a/layer_1"] == pytest.approx(2.0) and out["a/layer_3"] == pytest.approx(1.0)
    assert out["a"] == pytest.approx(1.5)  # mean over the layers that recorded
    assert out["s/entropy"] == pytest.approx(0.0, abs=1e-6)  # one slot takes everything
    assert out["s/min_rel"] == pytest.approx(0.0)  # ... and the other is dead
    assert out["u"] == pytest.approx(0.2)
    assert arch_metrics.collect() == {}  # cleared


def test_records_only_on_active_training_steps():
    model = _model("*-", attention_output_gate=True)
    _step(model)
    assert arch_metrics.collect() == {}
    arch_metrics.set_active(True)
    with torch.no_grad():  # eval / recompute forwards do not count
        model(torch.zeros(2, 16, dtype=torch.long, device="cuda"), None, None)
    assert arch_metrics.collect() == {}


def test_feature_model_reports_every_feature():
    pattern = "N*EP*-"
    model = _model(
        pattern,
        attention_output_gate=True,
        exclusive_self_attention=True,
        mhc=True,
        mhc_expansion_rate=4,
        mhc_routed_topk=2,
        **ENGRAM,
        **PKM,
        **MOE,
    )
    arch_metrics.set_active(True)
    _step(model)
    out = arch_metrics.collect()
    for key in (
        "attn_gate/mean",
        "attn_gate/closed",
        "xsa/self_cos",
        "engram/gate",
        "engram/gate_closed",
        "pkm/rows_used",
        "pkm/read_entropy",
        "shared_expert/gate",
        "routed_attn/pick_share/entropy",
        "routed_attn/pick_share/min_rel",
        "routed_attn/main_read",
        "routed_mlp/write",
        "routed_mixer/main_read",  # Engram / PKM through the residual-scheme adapter
    ):
        assert key in out, (key, sorted(out))
        assert 0.0 <= out[key] <= 2.0, (key, out[key])
    # Layer attribution: the PKM layer is layer 4, the attention gates layers 2 and 5.
    # the size of every residual stream entering the routed sublayers, per layer
    assert out["routed_attn/stream_rms/main"] > 0
    assert out["routed_mlp/stream_rms/max"] >= out["routed_mlp/stream_rms/main"]
    assert any(k.startswith("routed_attn/stream_rms/others/layer_") for k in out)
    assert "pkm/rows_used/layer_4" in out
    assert {"attn_gate/mean/layer_2", "attn_gate/mean/layer_5"} <= set(out)
    assert 0 < out["pkm/rows_used"] <= 1


def test_dense_mhc_stream_usage():
    model = _model("*-*-", mhc=True, mhc_expansion_rate=4, mhc_sinkhorn_iters=4)
    arch_metrics.set_active(True)
    _step(model)
    out = arch_metrics.collect()
    # At init the read is uniform over the streams and the mixer near the identity.
    assert out["mhc_attn/read_share/entropy"] == pytest.approx(1.0, abs=1e-3)
    assert out["mhc_mlp/res_diag"] > 0.9
    assert out["mhc_attn/write"] == pytest.approx(1.0, abs=1e-2)
    # every stream starts as a copy of the embedding: balanced
    assert out["mhc_attn/stream_rms/main"] > 0
    assert out["mhc_attn/stream_rms/max_over_min"] == pytest.approx(1.0, abs=1e-2)


def test_attn_res_depth_attention():
    model = _model("*-*-*-", attn_res=True, attn_res_num_blocks=2)
    arch_metrics.set_active(True)
    _step(model)
    out = arch_metrics.collect()
    assert 0.0 <= out["attnres_attn/entropy"] <= 1.0
    assert 0.0 <= out["attnres_mlp/current_weight"] <= 1.0


def test_ngpt_lerp_rates():
    model = _model("N*-*-", ngpt=True, **ENGRAM)
    arch_metrics.set_active(True)
    _step(model)
    out = arch_metrics.collect()
    assert out["ngpt_attn/lerp_rate"] > 0 and out["ngpt_mlp/lerp_rate"] > 0
    assert out["ngpt_mixer/lerp_rate"] > 0  # Engram through the residual-scheme adapter


@pytest.mark.parametrize("variant", ["gated_delta_net", "mlstm"])
def test_linear_mixer_gates(variant):
    try:
        model = _model(
            "L-*-",
            experimental_attention_variant=variant,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        )
        arch_metrics.set_active(True)
        _step(model)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"{variant} kernels unavailable: {e}")
    out = arch_metrics.collect()
    prefix = "gdn" if variant == "gated_delta_net" else "mlstm"
    decay = out[f"{prefix}/log_decay" if prefix == "gdn" else "mlstm/log_fgate"]
    assert decay < 0
    assert 0.0 <= out[f"{prefix}/long_memory"] <= 1.0
