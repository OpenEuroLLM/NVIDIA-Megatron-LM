# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The in-Megatron log-likelihood evaluator (megatron/training/downstream_eval.py)."""

import json

import pytest
import torch

import tests.unit_tests.models.test_hybrid_packed_docs as T
from megatron.training.downstream_eval import _unwrap, maybe_run_downstream_eval, score_item
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
