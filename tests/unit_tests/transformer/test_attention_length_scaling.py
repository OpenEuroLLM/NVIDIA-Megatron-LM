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
