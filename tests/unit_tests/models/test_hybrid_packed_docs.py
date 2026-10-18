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


def _model(**kw):
    config = TransformerConfig(
        num_layers=len(PATTERN),
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
        **kw,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB,
        max_sequence_length=128,
        hybrid_layer_pattern=PATTERN,
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
