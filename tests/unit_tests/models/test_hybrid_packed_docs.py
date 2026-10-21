# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Document masking (packed THD batches, dataloader_inter_document_masking) through the
production hybrid recipe: CKDA 5:1-style mixers, NoPE attention with XSA and output gate, MoE,
routed residual streams with full recompute. Each document of a packed batch must get the
logits it gets alone."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

PATTERN = "LELE*E"
VOCAB = 128
LENGTHS = (40, 64, 24)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _model(pattern=PATTERN, **kw):
    config = TransformerConfig(**{**dict(
        num_layers=len(pattern),
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        experimental_attention_variant="complex_kda",
        linear_beta_max=2.0,
        linear_output_gate="lowrank",
        linear_gate_init_style="shipped",
        linear_drop_qkv_silu=True,
        attention_output_gate=True,
        exclusive_self_attention=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        mhc=True,
        mhc_expansion_rate=4,
        mhc_routed_topk=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        is_hybrid_model=True,
    ), **kw})
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=128,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    ).cuda().bfloat16()  # as Float16Module casts it in training


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "recompute",
    [{}, dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)],
    ids=["no_recompute", "full_recompute"],
)
@pytest.mark.parametrize("fused_qk", [False, True], ids=["qknorm", "fused_qknorm"])
def test_packed_documents_match_separate(recompute, fused_qk):
    model = _model(**recompute, fused_qk_norm_rope=fused_qk)
    model.train()
    docs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in LENGTHS]

    def run(ids, packed=None):
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        return model(ids, position_ids=pos, attention_mask=None, packed_seq_params=packed)

    alone = [run(d).float() for d in docs]
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))
    together = run(torch.cat(docs, dim=1), packed).float()

    # bf16 model: compare in units of the logits' spread
    ref = torch.cat(alone, dim=1)
    scale = ref.std()
    err = (together - ref).abs().max() / scale
    assert err < 0.05, float(err)
    # the check has teeth: without the boundaries later documents see the earlier ones
    leaked = (run(torch.cat(docs, dim=1)).float() - ref)[:, LENGTHS[0]:].abs().max() / scale
    assert leaked > 4 * err, (float(err), float(leaked))

    # gradients flow through every part of the packed forward
    together.logsumexp(-1).mean().backward()
    assert all(p.grad is not None or getattr(p, "main_grad", None) is not None
               for n, p in model.named_parameters() if "expert_bias" not in n and p.requires_grad)


@pytest.mark.parametrize("table_parallel", ["tp", "dp"])
def test_packed_documents_match_separate_with_engram(table_parallel):
    """Engram (`N`) in the packed hybrid: its n-grams and causal conv restart per document.
    In fp32: with a non-zero conv the bf16 model differs by ~0.06 of the logit spread,
    evenly over every document (rounding; packed vs separate leak ~4), fp32 matches exactly."""
    model = _model(
        pattern="NLELE*E",
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=1000,
        engram_embed_dim=16,
        engram_table_parallel=table_parallel,
        mhc=False,  # the sweep's arms: plain residual
        mhc_routed_topk=0,
    ).float()
    model.train()
    engram = model.decoder.layers[0]
    with torch.no_grad():
        engram.conv1d.weight.normal_(std=0.5)  # non-zero, so its dilated taps matter
    docs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in LENGTHS]

    def run(ids, packed=None):
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        return model(ids, position_ids=pos, attention_mask=None, packed_seq_params=packed)

    ref = torch.cat([run(d).float() for d in docs], dim=1)
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))
    together = run(torch.cat(docs, dim=1), packed).float()
    scale = ref.std()
    err = (together - ref).abs().max() / scale
    assert err < 1e-3, float(err)
    leaked = (run(torch.cat(docs, dim=1)).float() - ref)[:, LENGTHS[0]:].abs().max() / scale
    assert leaked > 0.5, float(leaked)  # the check has teeth
    together.logsumexp(-1).mean().backward()
    if table_parallel == "dp":
        assert engram.table_grad.abs().sum() > 0
    else:
        assert engram.table.grad is not None or getattr(engram.table, "main_grad", None) is not None


@pytest.mark.parametrize("pattern", ["LELE*E", "NLELE*E"], ids=["hybrid", "hybrid_engram"])
def test_packed_documents_match_separate_with_depth_memory(pattern):
    """Linear Depth Memory (depth_memory.py) through the packed hybrid: per-token state, so
    documents stay independent; every LDM parameter gets a gradient."""
    model = _model(
        pattern=pattern,
        mhc=False,
        mhc_routed_topk=0,
        depth_memory=True,
        depth_memory_groups=4,
        depth_memory_key_dim=8,
        depth_memory_hidden_dim=16,
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=1000,
        engram_embed_dim=16,
    )
    model.train()
    docs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in LENGTHS]

    def run(ids, packed=None):
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        return model(ids, position_ids=pos, attention_mask=None, packed_seq_params=packed)

    ref = torch.cat([run(d).float() for d in docs], dim=1)
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))
    together = run(torch.cat(docs, dim=1), packed).float()
    err = (together - ref).abs().max() / ref.std()
    assert err < 0.05, float(err)
    together.logsumexp(-1).mean().backward()
    ldm = [(n, p) for n, p in model.named_parameters() if getattr(p, "is_residual_mixer", False)]
    assert ldm and all(p.grad is not None or getattr(p, "main_grad", None) is not None
                       for _, p in ldm), [n for n, p in ldm if p.grad is None]


POLAR = dict(linear_fla_gate="softplus", linear_kronlstm_forget_angle=True,
             linear_kronlstm_erase_angle=True, linear_kronlstm_signed_write=True,
             linear_kronlstm_output_gate="lowrank")


@pytest.mark.parametrize("gate", ["softplus", "signed_sigmoid2", "polar"])
@pytest.mark.parametrize(
    "recompute",
    [{}, dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)],
    ids=["no_recompute", "full_recompute"],
)
def test_packed_documents_match_separate_with_kronlstm(gate, recompute):
    """KronLSTM (`kronlstm`, fla_hybrid.KronLSTM) as the `L` mixer: its recurrence, sign
    parity and short conv restart at every document (polar: the angle gates and the signed
    write, low-rank output gate). In fp32 and with plain mHC (no routed stream picks): in bf16 the
    routed residual's discrete picks flip on rounding between the packed and the separate run
    (max error up to ~0.4 of the logit spread on a few tokens, any mixer)."""
    gates = POLAR if gate == "polar" else dict(linear_fla_gate=gate, linear_fla_allow_neg_eigval=True)
    model = _model(experimental_attention_variant="kronlstm", mhc_routed_topk=0, **gates,
                   **recompute).float()
    model.train()
    if gate == "polar":
        # flipped forget channels and reflecting erase heads, so the signs matter
        with torch.no_grad():
            for layer in model.decoder.layers:
                mixer = getattr(getattr(layer, "self_attention", None), "layer", None)
                if mixer is not None and hasattr(mixer, "f_angle_proj"):
                    b = mixer.f_angle_proj[-1].bias
                    b.copy_(torch.where(torch.rand_like(b) < 0.5, 4.0, -3.0))
                    mixer.b_angle_proj.bias.fill_(2.0)
    if gate.startswith("signed"):
        # a spread of negative forget gates, so the sign gauge matters
        with torch.no_grad():
            for layer in model.decoder.layers:
                mixer = getattr(getattr(layer, "self_attention", None), "layer", None)
                if mixer is not None and hasattr(mixer, "dt_bias"):
                    mixer.dt_bias.mul_(torch.where(torch.rand_like(mixer.dt_bias) < 0.5, -1.0, 1.0))
    docs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in LENGTHS]

    def run(ids, packed=None):
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        return model(ids, position_ids=pos, attention_mask=None, packed_seq_params=packed)

    ref = torch.cat([run(d).float() for d in docs], dim=1)
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))
    together = run(torch.cat(docs, dim=1), packed).float()
    scale = ref.std()
    err = (together - ref).abs().max() / scale
    assert err < 1e-2, float(err)  # fp32; the kernels' fp32 dots run in tf32
    leaked = (run(torch.cat(docs, dim=1)).float() - ref)[:, LENGTHS[0]:].abs().max() / scale
    assert leaked > 4 * err, (float(err), float(leaked))
    together.logsumexp(-1).mean().backward()
    kron = [(n, p) for n, p in model.named_parameters() if ".layer." in n and "self_attention" in n]
    assert kron and all(p.grad is not None or getattr(p, "main_grad", None) is not None
                        for _, p in kron), [n for n, p in kron if p.grad is None]
