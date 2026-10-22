# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The in-Megatron log-likelihood evaluator (megatron/training/downstream_eval.py)."""

import json

import pytest
import torch

import tests.unit_tests.models.test_hybrid_packed_docs as T
from megatron.training.downstream_eval import (
    PPL_BINS,
    _unwrap,
    maybe_run_downstream_eval,
    score_item,
    score_ppl,
)
from tests.unit_tests.models.test_hybrid_packed_docs import model_parallel  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _tiny():
    model = T._model(pattern="LE*E", mhc=False, mhc_routed_topk=0).float()
    model.eval()
    return model


def test_score_item_matches_full_logits():
    """Slicing the output layer's input to the continuation gives the full forward's log-probs."""
    model = _tiny()
    torch.manual_seed(0)
    ctx = torch.randint(0, T.VOCAB, (37,)).tolist()
    cont = torch.randint(0, T.VOCAB, (5,)).tolist()
    lp, greedy = score_item(model, _unwrap(model), ctx, cont, torch.device("cuda"))
    tokens = torch.tensor([ctx + cont], device="cuda")
    with torch.no_grad():
        logits = model(tokens, torch.arange(tokens.shape[1], device="cuda")[None], None)
    logp = torch.log_softmax(logits[0].float(), -1)
    n = len(cont)
    ref = logp[-n - 1:-1].gather(-1, tokens[0, -n:, None]).sum().item()
    assert lp == pytest.approx(ref, rel=1e-4, abs=1e-4)
    assert greedy == bool((logp[-n - 1:-1].argmax(-1) == tokens[0, -n:]).all())


def test_end_to_end_metrics(tmp_path, monkeypatch):
    """A 'question' whose gold continuation is the model's own greedy continuation scores acc 1."""
    model = _tiny()
    torch.manual_seed(1)
    ctx = torch.randint(0, T.VOCAB, (20,)).tolist()
    with torch.no_grad():
        logits = model(torch.tensor([ctx], device="cuda"), torch.arange(20, device="cuda")[None], None)
    best = int(logits[0, -1].argmax())
    worst = int(logits[0, -1].argmin())
    reqs = [{"task": "toy", "qid": 0, "gold": 1, "kind": "mc",
             "items": [{"ctx": ctx, "cont": [worst], "nchars": 1}, {"ctx": ctx, "cont": [best], "nchars": 1}]},
            {"task": "toy_greedy", "qid": 1, "gold": 0, "kind": "greedy",
             "items": [{"ctx": ctx, "cont": [best], "nchars": 1}]}]
    f = tmp_path / "req.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in reqs))
    out = tmp_path / "out.json"
    monkeypatch.setenv("DOWNSTREAM_EVAL_FILE", str(f))
    monkeypatch.setenv("DOWNSTREAM_EVAL_OUT", str(out))
    maybe_run_downstream_eval([model])
    res = json.loads(out.read_text())
    assert res["toy"]["acc"] == 1.0 and res["toy"]["acc_norm"] == 1.0
    assert res["toy_greedy"]["acc"] == 1.0


def test_score_ppl_matches_full_logits(tmp_path, monkeypatch):
    """The long-document loss per position bin (hidden states kept, logits in chunks) equals the
    full forward's per-token NLL averaged over the same bins; end to end through the evaluator."""
    model = _tiny()
    torch.manual_seed(2)
    doc = torch.randint(0, T.VOCAB, (1500,)).tolist()
    bins = score_ppl(model, _unwrap(model), doc, torch.device("cuda"), chunk=300)
    tokens = torch.tensor([doc], device="cuda")
    with torch.no_grad():
        logits = model(tokens, torch.arange(len(doc), device="cuda")[None], None)
    nll = -torch.log_softmax(logits[0].float(), -1)[:-1].gather(-1, tokens[0, 1:, None]).squeeze(-1)
    assert set(bins) == {"0-1024", "1024-2048"}
    for lo, hi in zip(PPL_BINS[:2], PPL_BINS[1:3]):
        tot, cnt = bins[f"{lo}-{hi}"]
        part = nll[lo:min(hi, len(doc) - 1)]
        assert cnt == part.numel()
        assert tot == pytest.approx(part.sum().item(), rel=1e-4)
    f = tmp_path / "req.jsonl"
    f.write_text(json.dumps({"task": "longppl", "qid": 0, "gold": 0, "kind": "ppl",
                             "items": [{"ctx": doc, "cont": [], "nchars": 0}]}))
    out = tmp_path / "out.json"
    monkeypatch.setenv("DOWNSTREAM_EVAL_FILE", str(f))
    monkeypatch.setenv("DOWNSTREAM_EVAL_OUT", str(out))
    maybe_run_downstream_eval([model])
    res = json.loads(out.read_text())["longppl"]
    assert res["tokens_0-1024"] == 1024 and res["tokens_1024-2048"] == 475
    assert res["nll_0-1024"] == pytest.approx(nll[:1024].mean().item(), rel=1e-4)
