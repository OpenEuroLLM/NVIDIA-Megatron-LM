# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Length-dependent attention temperature (attention_length_scaling: logn / ssmax)."""

import math

import pytest
import torch

import tests.unit_tests.models.test_hybrid_packed_docs as T
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.models.test_hybrid_packed_docs import model_parallel  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _model(**kw):
    m = T._model(pattern="LE*E", mhc=False, mhc_routed_topk=0, **kw).float()
    m.eval()
    return m


def _run(model, ids, packed=None):
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        return model(ids, position_ids=pos, attention_mask=None, packed_seq_params=packed).float()


def test_logn_is_identity_up_to_the_base():
    torch.manual_seed(0)
    ref = _model()
    sd = ref.state_dict()
    torch.manual_seed(0)
    scaled = _model(attention_length_scaling="logn", attention_length_scaling_base=64)
    scaled.load_state_dict(sd)
    ids = torch.randint(0, T.VOCAB, (1, 96), device="cuda")
    a, b = _run(ref, ids), _run(scaled, ids)
    torch.testing.assert_close(a[:, :64], b[:, :64])  # n <= base: factor 1
    assert (a[:, 64:] - b[:, 64:]).abs().max() > 1e-4  # beyond the base: scaled


@pytest.mark.parametrize("mode", ["logn", "ssmax"])
def test_positions_restart_per_document(mode):
    """Packed documents get the logits they get alone: n counts from each document's start."""
    torch.manual_seed(0)
    model = _model(attention_length_scaling=mode, attention_length_scaling_base=16)
    docs = [torch.randint(0, T.VOCAB, (1, n), device="cuda") for n in T.LENGTHS]
    ref = torch.cat([_run(model, d) for d in docs], dim=1)
    cu = torch.tensor([0, *torch.tensor(T.LENGTHS).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(T.LENGTHS), max_seqlen_kv=max(T.LENGTHS))
    together = _run(model, torch.cat(docs, dim=1), packed)
    assert ((together - ref).abs().max() / ref.std()) < 1e-2


def test_ssmax_learns_and_is_scale_one_at_the_base():
    torch.manual_seed(0)
    model = _model(attention_length_scaling="ssmax", attention_length_scaling_base=64)
    s = [p for n, p in model.named_parameters() if n.endswith("ssmax_s")]
    # (the test model is built in bf16, then cast: 1 / ln 64 to bf16 precision)
    assert s and all(torch.allclose(p, torch.full_like(p, 1 / math.log(64)), atol=2e-3) for p in s)
    model.train()
    ids = torch.randint(0, T.VOCAB, (1, 40), device="cuda")
    out = model(ids, position_ids=torch.arange(40, device="cuda")[None], attention_mask=None)
    out.float().logsumexp(-1).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in s)


def test_factor_values():
    """The factor itself: logn = max(1, log n / log base); positions per document in thd."""
    from megatron.core.transformer.attention import Attention

    class Dummy:
        pass

    d = Dummy()
    d.length_scaling = "logn"
    d.config = type("C", (), {"attention_length_scaling_base": 4})()
    d.pg_collection = type("P", (), {"cp": None})()
    q = torch.ones(10, 1, 1, 1)
    out = Attention._length_scale_query(d, q, None, None).flatten()
    exp = torch.clamp(torch.log(torch.arange(1, 11).float()) / math.log(4), min=1.0)
    torch.testing.assert_close(out, exp)
    cu = torch.tensor([0, 3, 10])
    p = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    out = Attention._length_scale_query(d, torch.ones(10, 1, 1), p, None).flatten()
    n = torch.tensor([1, 2, 3, 1, 2, 3, 4, 5, 6, 7]).float()
    torch.testing.assert_close(out, torch.clamp(torch.log(n) / math.log(4), min=1.0))


# --------------------------------------------------------------------------------------------
# Positions under context parallelism and at inference (length_scaling_positions, KV-cache decode)


def _zigzag(full: torch.Tensor, rank: int, cp: int) -> torch.Tensor:
    c = full.shape[0] // (2 * cp)
    return torch.cat([full[rank * c:(rank + 1) * c], full[(2 * cp - 1 - rank) * c:(2 * cp - rank) * c]])


@pytest.mark.parametrize("cp", [1, 2, 4])
def test_positions_follow_cp_sharding(cp):
    """Each rank's positions are the full sequence's positions, sharded like the tokens."""
    from megatron.core.transformer.attention import length_scaling_positions

    for rank in range(cp):
        s = 16 * cp
        exp = _zigzag(torch.arange(s), rank, cp) if cp > 1 else torch.arange(s)
        got = length_scaling_positions(s // cp, "cpu", None, rank, cp)
        torch.testing.assert_close(got, exp)
        # thd: every (padded) document sharded on its own, positions restarting per document
        lens = [8 * cp, 24 * cp, 4 * cp]
        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()])
        exp = torch.cat([_zigzag(torch.arange(n), rank, cp) if cp > 1 else torch.arange(n)
                         for n in lens])
        got = length_scaling_positions(sum(lens) // cp, "cpu", cu, rank, cp)
        torch.testing.assert_close(got, exp)


def test_positions_at_inference():
    """Static batching continues from the cache offset (per batch entry under flash decode);
    dynamic batching uses each token's position in its request."""
    from megatron.core.transformer.attention import Attention

    class D:
        pass

    d = D()
    d.length_scaling = "logn"
    d.config = type("C", (), {"attention_length_scaling_base": 2})()
    d.pg_collection = type("P", (), {"cp": None})()

    class Static:
        sequence_len_offset = 5

        def is_static_batching(self):
            return True

    f = lambda n: torch.clamp(torch.log(n.float()) / math.log(2), min=1.0)  # noqa: E731
    out = Attention._length_scale_query(d, torch.ones(3, 1, 1, 1), None, Static()).flatten()
    torch.testing.assert_close(out, f(torch.tensor([6, 7, 8])))
    off = torch.tensor([3, 9])  # flash decode: one offset per batch entry, one new token
    out = Attention._length_scale_query(d, torch.ones(1, 2, 1, 1), None, Static(),
                                        sequence_len_offset=off).flatten()
    torch.testing.assert_close(out, f(off + 1))

    class Dynamic:
        gpu_view = type("V", (), {"token_to_pos_ids": torch.tensor([0, 1, 2, 40, 7, 0])})()

        def is_static_batching(self):
            return False

    out = Attention._length_scale_query(d, torch.ones(5, 1, 1, 1), None, Dynamic()).flatten()
    torch.testing.assert_close(out, f(torch.tensor([1, 2, 3, 41, 8])))


@pytest.mark.parametrize("mode", ["logn", "ssmax"])
def test_kv_cache_decode_matches_full_forward(mode):
    """Prefill + token-by-token decode through the static KV cache = one full forward."""
    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    torch.manual_seed(0)
    cfg = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4,
                            hidden_dropout=0.0, attention_dropout=0.0,
                            attention_length_scaling=mode, attention_length_scaling_base=4)
    layer = TransformerLayer(cfg, get_gpt_layer_with_transformer_engine_submodules()).cuda().eval()
    if mode == "ssmax":
        with torch.no_grad():
            layer.self_attention.ssmax_s.uniform_(0.2, 0.6)
    L, P = 24, 9
    x = torch.randn(L, 1, 64, device="cuda")
    with torch.no_grad():
        full, _ = layer(hidden_states=x, attention_mask=None)
        ctx = StaticInferenceContext(max_batch_size=1, max_sequence_length=L)
        outs = []
        with InferenceMode.active():
            o, _ = layer(hidden_states=x[:P], attention_mask=None, inference_context=ctx)
            outs.append(o)
            ctx.sequence_len_offset += P
            for t in range(P, L):
                o, _ = layer(hidden_states=x[t:t + 1], attention_mask=None, inference_context=ctx)
                outs.append(o)
                ctx.sequence_len_offset += 1
    torch.testing.assert_close(torch.cat(outs), full, atol=2e-3, rtol=2e-3)
