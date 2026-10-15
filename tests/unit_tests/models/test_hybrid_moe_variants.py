# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MoE variants on a HybridModel (`E` layers): shared vs no shared expert, and
load balancing by aux loss, by the aux-loss-free expert bias, or by both; plus
which parameters the Muon-family optimizers leave to Adam."""

import fnmatch

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from tests.unit_tests.test_utilities import Utils

VOCAB = 128
EXPERTS, TOPK = 4, 2

SHARED = {
    "routed_only": {},
    "shared": dict(moe_shared_expert_intermediate_size=64),
    "shared_gated": dict(moe_shared_expert_intermediate_size=64, moe_shared_expert_gate=True),
}
BALANCE = {
    "aux_loss": dict(moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=1e-2),
    "seq_aux_loss": dict(moe_router_load_balancing_type="seq_aux_loss", moe_aux_loss_coeff=1e-2),
    "bias": dict(
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    ),
    "aux_loss_and_bias": dict(
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-3,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    ),
}


def _config(pattern, **kw):
    return TransformerConfig(
        num_layers=len(pattern),
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
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_ffn_hidden_size=64,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp32",
        **kw,
    )


def _model(config, pattern):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=32,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda()


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    get_moe_metrics_tracker().clear()
    yield
    get_moe_metrics_tracker().clear()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("shared", list(SHARED))
@pytest.mark.parametrize("balance", list(BALANCE))
def test_moe_variant_trains(shared, balance):
    pattern = "*E*E"
    config = _config(pattern, **SHARED[shared], **BALANCE[balance])
    model = _model(config, pattern)
    ids = torch.randint(0, VOCAB, (2, 16), device="cuda")
    logits = model(ids, position_ids=None, attention_mask=None)
    assert torch.isfinite(logits).all()
    logits.float().logsumexp(-1).mean().backward()

    grads = {n: p.grad for n, p in model.named_parameters()}
    routers = [n for n in grads if n.endswith("router.weight")]
    assert len(routers) == 2 and all(grads[n] is not None for n in routers)
    has_shared = [n for n in grads if "shared_experts" in n]
    assert bool(has_shared) == (shared != "routed_only")
    if shared == "shared_gated":
        assert any(n.endswith("shared_experts.gate_weight") for n in grads)
    assert all(g is not None for n, g in grads.items() if "shared_experts" in n)

    # Aux loss: recorded per MoE layer when its coefficient is set.
    tracked = get_moe_metrics_tracker().metrics
    if config.moe_aux_loss_coeff:
        assert any(v.values.abs().sum() > 0 for v in tracked.values())

    # Expert bias: tokens are counted during the forward and the bias moves toward
    # the under-loaded experts at the end of the step.
    biased = [m for m in model.modules() if getattr(m, "expert_bias", None) is not None]
    assert bool(biased) == config.moe_router_enable_expert_bias
    if biased:
        counts = [m.local_tokens_per_expert.clone() for m in biased]
        assert all(c.sum() == ids.numel() * TOPK for c in counts)
        before = [m.expert_bias.clone() for m in biased]
        _update_router_expert_bias([model], config)
        for m, b, c in zip(biased, before, counts):
            moved = m.expert_bias - b
            under = c < c.float().mean()
            assert (moved[under] >= 0).all() and (moved[~under] <= 0).all()


@pytest.mark.parametrize("optimizer", ["muon", "muown", "angular_muown"])
def test_muon_family_leaves_routers_tables_and_convs_to_adam(optimizer):
    """Muon-family updates only hidden linear maps: MoE routers / shared-expert gates,
    lookup tables (Engram, PKM), per-channel conv kernels (Canon) and the residual
    stream mixers (mHC, routed) stay on Adam."""
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS

    if optimizer not in _EMERGING_OPTIMIZERS:
        pytest.skip(f"{optimizer} not registered")
    pattern = "N*EP-*E"
    config = _config(
        pattern,
        moe_shared_expert_intermediate_size=64,
        moe_shared_expert_gate=True,
        canon_layers="ABCD",
        mhc=True,
        mhc_expansion_rate=4,
        mhc_routed_topk=2,
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=500,
        engram_embed_dim=8,
        pkm_heads=2,
        pkm_num_keys=16,
        pkm_key_dim=16,
        pkm_topk=4,
    )
    model = _model(config, pattern)
    keys = _EMERGING_OPTIMIZERS[optimizer].default_param_overrides

    def on_adam(name, param):
        return any(key.matches(param, name) for key in keys)

    adam = {n for n, p in model.named_parameters() if on_adam(n, p)}
    must_adam = [
        "*router.weight",
        "*gate_weight",
        "*.table",
        "*.values",
        "*canon_*.weight",
        "*mixer.phi",
        "*embedding*",
        "*output_layer.weight",
    ]
    for pat in must_adam:
        hits = [n for n, _ in model.named_parameters() if fnmatch.fnmatch(n, pat)]
        assert hits, pat
        assert set(hits) <= adam, (pat, set(hits) - adam)
    for n, p in model.named_parameters():
        if p.dim() == 2 and any(
            s in n for s in ("linear_qkv.weight", "linear_proj.weight", "linear_fc1.weight")
        ):
            assert n not in adam, n
