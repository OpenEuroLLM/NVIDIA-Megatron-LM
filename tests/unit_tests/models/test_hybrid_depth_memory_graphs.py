# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Linear Depth Memory in its stream form (depth_memory_impl torch / cute) through a HybridModel
under per-layer CUDA graphs: the captured-and-replayed model must match the eager one exactly
(the state lives in a static workspace, the residual as [s, b, 2, h] streams), and the stream
form must match the reference implementation."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
    delete_cuda_graphs,
)
from tests.unit_tests.test_utilities import Utils

VOCAB, S, B = 128, 32, 2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# (the hybrid MLP layer "-" is not graph-capturable upstream: MLPLayer reads is_moe_layer
# before it sets it)
SCOPES = {
    "**": ["attn"],
    "*E*E": ["attn", "moe_router", "moe_preprocess"],
}


def _model(pattern, impl, graphs=None, **kw):
    model_parallel_cuda_manual_seed(123)
    moe = dict(
        num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64,
        moe_grouped_gemm=True, moe_permute_fusion=True, moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="none",
    ) if "E" in pattern else {}
    config = TransformerConfig(
        num_layers=len(pattern), hidden_size=64, num_attention_heads=4, ffn_hidden_size=128,
        use_cpu_initialization=True, hidden_dropout=0.0, attention_dropout=0.0,
        normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, is_hybrid_model=True,
        depth_memory=True, depth_memory_impl=impl, depth_memory_groups=4,
        depth_memory_key_dim=8, depth_memory_hidden_dim=16,
        **({"cuda_graph_impl": "local", "cuda_graph_warmup_steps": 2,
            "cuda_graph_modules": graphs} if graphs else {}),
        **moe, **kw,
    )
    return HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=VOCAB,
        max_sequence_length=S, hybrid_layer_pattern=pattern, position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _fwd_bwd(model, ids):
    for p in model.parameters():
        if getattr(p, "main_grad", None) is None:
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        else:
            p.main_grad.zero_()
    pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)
    out = model(ids, position_ids=pos, attention_mask=None)
    model.zero_grad(set_to_none=True)
    out.float().square().mean().backward()
    grads = {}
    for k, p in model.named_parameters():
        grads[k] = (p.grad if p.grad is not None else p.main_grad.to(p.dtype)).clone()
    return out.detach().clone(), grads


class TestDepthMemoryGraphs:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(1, 1)
        CudaGraphManager.global_mempool = None

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("pattern", list(SCOPES))
    def test_graph_replay_matches_eager(self, pattern):
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        eager = _model(pattern, "torch")
        with torch.no_grad():  # off the identity init, so every LDM parameter has a gradient
            for name, p in eager.named_parameters():
                if getattr(p, "is_residual_mixer", False):
                    p.add_(torch.randn_like(p) * 0.2)
        ref_out, ref_grads = _fwd_bwd(eager, ids)
        ldm = [k for k in ref_grads if "next_query" in k or "write_site" in k or "ldm_" in k]
        assert ldm and all(ref_grads[k].abs().sum() > 0 for k in ldm), [
            k for k in ldm if ref_grads[k].abs().sum() == 0]

        graphed = _model(pattern, "torch", graphs=SCOPES[pattern])
        graphed.load_state_dict(eager.state_dict())
        _fwd_bwd(graphed, ids)
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created, "no cudagraphs were created"
        for _ in range(2):  # replay twice: the static workspace is reused per micro-batch
            out, grads = _fwd_bwd(graphed, ids)
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
            for k, g in ref_grads.items():
                torch.testing.assert_close(grads[k], g, rtol=0, atol=0, msg=k)

    def test_stream_form_matches_reference_model(self):
        """The same parameters through the reference (state on the layers) and the stream form
        give the same logits (fp32 state in both)."""
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        stream = _model("*E*E", "torch")
        ref = _model("*E*E", "reference")
        with torch.no_grad():
            for p in stream.parameters():
                if getattr(p, "is_residual_mixer", False):
                    p.add_(torch.randn_like(p) * 0.2)
        # stream form: every writer owns the NEXT read's query; the reference: each site its own
        sd = stream.state_dict()
        writers = ["decoder.ldm_stream_boundary"] + [
            f"decoder.layers.{i}.{kind}" for i, kind in
            enumerate(["mhc_attn_mixer", "mhc_mlp_mixer", "mhc_attn_mixer", "mhc_mlp_mixer"])]
        sites = [f"decoder.layers.{i}.{kind}" for i, kind in
                 enumerate(["ldm_attn", "ldm_mlp", "ldm_attn", "ldm_mlp"])]
        state = {k: v for k, v in sd.items() if "mhc_" not in k and "ldm_stream" not in k}

        def copy(src, dst):
            for k, v in sd.items():
                if k.startswith(src + "."):
                    state[dst + k[len(src):]] = v

        copy("decoder.ldm_stream_boundary.embed_write", "decoder.ldm_boundary.embed_write")
        copy(writers[0] + ".first_query", sites[0] + ".read")
        for i, site in enumerate(sites):
            copy(writers[i + 1] + ".write_site", site + ".write")
            nxt = sites[i + 1] + ".read" if i + 1 < len(sites) else "decoder.ldm_boundary.final_read"
            copy(writers[i + 1] + ".next_query", nxt)
        missing, unexpected = ref.load_state_dict(state, strict=False)
        assert not [m for m in missing if "_extra_state" not in m], missing
        assert not unexpected, unexpected
        out_s, _ = _fwd_bwd(stream, ids)
        out_r, _ = _fwd_bwd(ref, ids)
        torch.testing.assert_close(out_s, out_r, rtol=1e-4, atol=1e-4)
