# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""depth_memory_segments: the LDM state carried in the streams across segments (recompute chunks,
or the stage) gives the outputs and gradients of the one-sequence stream form -- without
recompute, and with full recompute at one or two layers per chunk (every chunk a segment, its
input [x, r, M, P] the checkpoint)."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

VOCAB, S, B = 128, 32, 2
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _model(pattern, impl, **kw):
    torch.manual_seed(0)  # use_cpu_initialization: the CPU generator draws the weights
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=len(pattern), hidden_size=64, num_attention_heads=4, ffn_hidden_size=128,
        use_cpu_initialization=True, hidden_dropout=0.0, attention_dropout=0.0,
        normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, is_hybrid_model=True,
        depth_memory=True, depth_memory_impl=impl, depth_memory_groups=4,
        depth_memory_key_dim=8, depth_memory_hidden_dim=16, **kw,
    )
    return HybridModel(
        config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=VOCAB,
        max_sequence_length=S, hybrid_layer_pattern=pattern, position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    ).cuda()


def _fwd_bwd(model, ids):
    pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)
    model.zero_grad(set_to_none=True)
    out = model(ids, position_ids=pos, attention_mask=None)
    out.float().square().mean().backward()
    return out.detach().float(), {k: p.grad.float().clone() for k, p in model.named_parameters()
                                  if p.grad is not None}


@pytest.mark.parametrize("impl", ["torch", "cute"])
@pytest.mark.parametrize("variant", ["no_recompute", "recompute1", "recompute2"])
def test_segments_match_one_sequence(impl, variant):
    if impl == "cute":
        pytest.importorskip("cute_ldm.workspace")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("cute_ldm needs Hopper / Blackwell")
    Utils.initialize_model_parallel(1, 1)
    try:
        pattern = "*-*-*-"
        ids = torch.randint(0, VOCAB, (B, S), device="cuda")
        ref_out, ref_grads = _fwd_bwd(_model(pattern, impl), ids)
        again, _ = _fwd_bwd(_model(pattern, impl), ids)
        torch.testing.assert_close(again, ref_out, rtol=0, atol=0)  # same weights both builds
        kw = dict(depth_memory_segments=True)
        if variant != "no_recompute":
            kw.update(recompute_granularity="full", recompute_method="uniform",
                      recompute_num_layers=1 if variant == "recompute1" else 2)
        model = _model(pattern, impl, **kw)
        out, grads = _fwd_bwd(model, ids)
        tol = dict(rtol=1e-4, atol=1e-5) if impl == "torch" else dict(rtol=3e-2, atol=3e-3)
        torch.testing.assert_close(out, ref_out, **tol)
        assert set(grads) == set(ref_grads)
        for k in ref_grads:
            torch.testing.assert_close(grads[k], ref_grads[k], **tol, msg=k)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("fused", ["0", "1"])
@pytest.mark.parametrize("pattern", ["*-*-*-", "*E*E-*"])
def test_unwind_matches_one_sequence(pattern, fused, monkeypatch):
    """depth_memory_unwind: full recompute, one layer per chunk, no state in the checkpoints (each
    layer's recompute reads the unwound state); two micro-batches in a row. The torch path does
    the reference's arithmetic (tight); the fused Triton passes compute in true fp32 while the
    reference's einsums run in TF32 under the NGC container's cuBLAS override (~1e-3)."""
    monkeypatch.setenv("DEPTH_MEMORY_UNWIND_TRITON", fused)
    out_tol = dict(rtol=1e-4, atol=1e-5) if fused == "0" else dict(rtol=2e-2, atol=3e-4)
    grad_tol = dict(rtol=1e-3, atol=1e-5) if fused == "0" else dict(rtol=3e-2, atol=3e-4)
    Utils.initialize_model_parallel(1, 1)
    try:
        kw = dict(num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64) if "E" in pattern else {}
        ids = [torch.randint(0, VOCAB, (B, S), device="cuda") for _ in range(2)]
        ref = _model(pattern, "torch", **kw)
        model = _model(pattern, "torch", depth_memory_unwind=True, recompute_granularity="full",
                       recompute_method="uniform", recompute_num_layers=1, **kw)
        for batch in ids:
            ref_out, ref_grads = _fwd_bwd(ref, batch)
            out, grads = _fwd_bwd(model, batch)
            torch.testing.assert_close(out, ref_out, **out_tol)
            assert set(grads) == set(ref_grads)
            for k in ref_grads:
                torch.testing.assert_close(grads[k], ref_grads[k], **grad_tol, msg=k)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])   # 1: windows of 2 edges; 3: odd groups; 4: windows inside a group
@pytest.mark.parametrize("pattern", ["*-*-*-*-", "*E*E-*"])
def test_cute_recompute_matches_one_sequence(pattern, chunk):
    """depth_memory_unwind with the CuTe kernels: full recompute with ``chunk`` layers per chunk,
    the tapes carried in the stream as records, each recomputed layer's read from the running
    state; two micro-batches in a row. Within the CuTe tolerance of the torch oracle
    and of the CuTe one-sequence form without recompute."""
    pytest.importorskip("cute_ldm.workspace")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cute_ldm needs Hopper / Blackwell")
    Utils.initialize_model_parallel(1, 1)
    try:
        kw = dict(num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64) if "E" in pattern else {}
        ids = [torch.randint(0, VOCAB, (B, S), device="cuda") for _ in range(2)]
        oracle = _model(pattern, "torch", **kw)
        ref = _model(pattern, "cute", **kw)
        model = _model(pattern, "cute", depth_memory_unwind=True,
                       recompute_granularity="full", recompute_method="uniform",
                       recompute_num_layers=chunk, **kw)
        tol = dict(rtol=3e-2, atol=3e-3)
        for batch in ids:
            oracle_out, oracle_grads = _fwd_bwd(oracle, batch)
            ref_out, ref_grads = _fwd_bwd(ref, batch)
            out, grads = _fwd_bwd(model, batch)
            assert set(grads) == set(ref_grads) == set(oracle_grads)
            torch.testing.assert_close(out, ref_out, **tol)
            torch.testing.assert_close(out, oracle_out, **tol)
            for k in ref_grads:
                torch.testing.assert_close(grads[k], ref_grads[k], **tol, msg=k)
                torch.testing.assert_close(grads[k], oracle_grads[k], **tol, msg=k)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("chunk", [1, 2])
@pytest.mark.parametrize("storage,interval,prefetch", [
    ("float16", 0, False), ("float16", 4, False), ("float16", 4, True), ("bfloat16", 4, False)])
def test_cute_recompute_fp16_checkpoints_match_oracle(storage, interval, prefetch, chunk):
    """depth_memory_unwind with the CuTe kernels, FP16 storage (the workspace scales its own
    backward: no FP16 training) and M checkpoints every ``interval`` edges: within the CuTe
    tolerance of the torch oracle; two micro-batches in a row (the second scales from the
    first's cotangents)."""
    pytest.importorskip("cute_ldm.workspace")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cute_ldm needs Hopper / Blackwell")
    Utils.initialize_model_parallel(1, 1)
    try:
        pattern = "*-*-*-*-"
        ids = [torch.randint(0, VOCAB, (B, S), device="cuda") for _ in range(2)]
        oracle = _model(pattern, "torch")
        model = _model(pattern, "cute", depth_memory_unwind=True, recompute_granularity="full",
                       recompute_method="uniform", recompute_num_layers=chunk,
                       depth_memory_storage_dtype=storage,
                       depth_memory_checkpoint_interval=interval,
                       depth_memory_checkpoint_prefetch=prefetch)
        ws = model.decoder.ldm_workspace
        assert ws.checkpoint_prefetch == prefetch
        assert ws.storage == getattr(torch, storage)
        assert ws.grad_scaling == (storage == "float16")
        edges = len(pattern) + 1
        assert ws.checkpoints == (list(range(interval, edges, interval)) if interval else [])
        tol = dict(rtol=3e-2, atol=3e-3)
        for batch in ids:
            oracle_out, oracle_grads = _fwd_bwd(oracle, batch)
            out, grads = _fwd_bwd(model, batch)
            assert set(grads) == set(oracle_grads)
            torch.testing.assert_close(out, oracle_out, **tol)
            for k in oracle_grads:
                torch.testing.assert_close(grads[k], oracle_grads[k], **tol, msg=k)
        assert int(ws.grad_nonfinite) == 0
    finally:
        Utils.destroy_model_parallel()


def test_depth_memory_fp16_checkpoint_config():
    base = dict(num_layers=4, hidden_size=64, num_attention_heads=4, is_hybrid_model=True,
                depth_memory=True, depth_memory_groups=4)
    with pytest.raises(ValueError, match="needs depth_memory_impl cute"):
        TransformerConfig(**base, depth_memory_impl="torch", depth_memory_storage_dtype="float16")
    with pytest.raises(ValueError, match="depth_memory_checkpoint_interval needs"):
        TransformerConfig(**base, depth_memory_impl="cute", depth_memory_checkpoint_interval=4)
    with pytest.raises(ValueError, match="multiple of depth_memory_window"):
        TransformerConfig(**base, depth_memory_impl="cute", depth_memory_unwind=True,
                          recompute_granularity="full", recompute_method="uniform",
                          recompute_num_layers=2, depth_memory_checkpoint_interval=6)
