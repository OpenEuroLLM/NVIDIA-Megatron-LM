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
    base = dict(
        num_layers=len(pattern), hidden_size=64, num_attention_heads=4, ffn_hidden_size=128,
        use_cpu_initialization=True, hidden_dropout=0.0, attention_dropout=0.0,
        normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, is_hybrid_model=True,
        depth_memory=True, depth_memory_impl=impl, depth_memory_groups=4,
        depth_memory_key_dim=8, depth_memory_hidden_dim=16,
        **({"cuda_graph_impl": "local", "cuda_graph_warmup_steps": 2,
            "cuda_graph_modules": graphs} if graphs else {}),
        **moe,
    )
    config = TransformerConfig(**{**base, **kw})
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
        # relative to the logits' spread: the stream form folds the norm gains into one stacked
        # GEMM, which rounds differently under TF32 (the cluster tests' matmul precision)
        err = (out_s - out_r).abs().max() / out_r.std()
        assert err < 2e-3, float(err)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="the CuTe LDM kernels need sm90 / sm100",
)
class TestDepthMemoryCuteGraphs(TestDepthMemoryGraphs):
    """The CuTe kernels: graph replay = eager (same kernels, same buffers), and cute ~ torch."""

    @pytest.mark.parametrize("pattern", list(SCOPES))
    def test_graph_replay_matches_eager(self, pattern):
        pytest.importorskip("cute_ldm.workspace")
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        kw = dict(bf16=True, params_dtype=torch.bfloat16)
        eager = _model(pattern, "cute", **kw).bfloat16()
        ref_out, ref_grads = _fwd_bwd(eager, ids)
        graphed = _model(pattern, "cute", graphs=SCOPES[pattern], **kw).bfloat16()
        graphed.load_state_dict(eager.state_dict())
        _fwd_bwd(graphed, ids)
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created, "no cudagraphs were created"
        for _ in range(2):
            out, grads = _fwd_bwd(graphed, ids)
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
            for k, g in ref_grads.items():
                torch.testing.assert_close(grads[k], g, rtol=0, atol=0, msg=k)

    def test_stream_form_matches_reference_model(self):
        pytest.skip("covered by the torch class; cute vs torch in test_depth_memory_cute.py")


class TestDepthMemoryPerLayer(TestDepthMemoryGraphs):
    """LDM per transformer layer (mhc_per_layer): one read and one fused write per block (mixer
    + MoE), plain residual inside -- the steps halve; graph replay = eager."""

    def test_one_step_per_block(self):
        model = _model("*E*E*E", "torch", mhc_per_layer=True)
        stack = model.decoder
        assert stack.ldm_workspace.total_edges == 3 + 1  # three blocks + the embedding write
        mixers = [l.mhc_block_mixer for l in stack.layers if getattr(l, "mhc_block_mixer", None)]
        assert [m.edge for m in mixers] == [1, 2, 3]
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        out, grads = _fwd_bwd(model, ids)
        assert torch.isfinite(out).all()
        ldm = [k for k in grads if "write_site" in k or "next_query" in k or "ldm_" in k]
        assert ldm and all(grads[k].abs().sum() > 0 for k in ldm if "gate.weight" not in k)

    def test_block_semantics(self):
        """The block reads x + r once; inside, the plain residual; the write gets the block's
        update h_end - (x + r) = d_1 + d_2: the stream after the block is x + d_1 + d_2."""
        from megatron.core.transformer import mhc_block

        model = _model("*E", "torch", mhc_per_layer=True)
        seen = {}
        orig = mhc_block.block_forward

        def spy(layer, inner, args, kwargs):
            out, ctx = orig(layer, inner, args, kwargs)
            seen[layer.layer_number] = out
            return out, ctx

        mhc_block.block_forward = spy
        try:
            ids = torch.randint(0, VOCAB, (B, S), device="cuda")
            _fwd_bwd(model, ids)
        finally:
            mhc_block.block_forward = orig
        # layer 1 (attention) returns the in-block [s, b, h]; layer 2 (MoE, last) the streams
        assert seen[1].dim() == 3 and seen[2].dim() == 4 and seen[2].shape[-2] == 2

    def test_whole_layer_graphs_rejected(self):
        with pytest.raises(ValueError, match="MoE router / preprocess only"):
            _model("*E*E", "torch", graphs=["attn", "moe_router"], mhc_per_layer=True)

    @pytest.mark.parametrize("pattern", ["*E*E"])
    def test_graph_replay_matches_eager(self, pattern):
        """The MoE router / preprocess graphs (round 5's document-masked setting)."""
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        eager = _model(pattern, "torch", mhc_per_layer=True)
        with torch.no_grad():
            for name, p in eager.named_parameters():
                if getattr(p, "is_residual_mixer", False):
                    p.add_(torch.randn_like(p) * 0.2)
        ref_out, ref_grads = _fwd_bwd(eager, ids)
        graphed = _model(pattern, "torch", graphs=["moe_router", "moe_preprocess"],
                         mhc_per_layer=True)
        graphed.load_state_dict(eager.state_dict())
        _fwd_bwd(graphed, ids)
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created, "no cudagraphs were created"
        for _ in range(2):
            out, grads = _fwd_bwd(graphed, ids)
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
            for k, g in ref_grads.items():
                torch.testing.assert_close(grads[k], g, rtol=0, atol=0, msg=k)

    def test_stream_form_matches_reference_model(self):
        pytest.skip("per-layer has no per-sublayer reference")


@pytest.mark.parametrize("sites,mixer_edges", [("mixer", [1, 2]), ("mlp", [1, 2])])
def test_depth_memory_sites(sites, mixer_edges):
    """depth_memory_sites: LDM at the mixer or the MLP / MoE sublayers only; the others pass
    the streams through (plain residual on x, r unchanged) and take no step."""
    from megatron.core.transformer.depth_memory import DepthMemoryMixer, DepthMemoryPassThrough

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(1, 1)
    try:
        model = _model("*E*E", "torch", depth_memory_sites=sites)
        layers = model.decoder.layers
        attn = [l.mhc_attn_mixer for l in layers if l.mhc_attn_mixer is not None]
        mlp = [l.mhc_mlp_mixer for l in layers if l.mhc_mlp_mixer is not None]
        live, passing = (attn, mlp) if sites == "mixer" else (mlp, attn)
        assert all(isinstance(m, DepthMemoryMixer) for m in live)
        assert all(isinstance(m, DepthMemoryPassThrough) for m in passing)
        assert [m.edge for m in live] == mixer_edges
        assert model.decoder.ldm_workspace.total_edges == len(live) + 1
        streams = torch.randn(4, 1, 2, 64, device="cuda")
        out = passing[0].write(streams, torch.ones(4, 1, 64, device="cuda"))
        torch.testing.assert_close(out[..., 1, :], streams[..., 1, :])
        torch.testing.assert_close(out[..., 0, :], streams[..., 0, :] + 1)
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        logits, grads = _fwd_bwd(model, ids)
        assert torch.isfinite(logits).all()
    finally:
        Utils.destroy_model_parallel()
