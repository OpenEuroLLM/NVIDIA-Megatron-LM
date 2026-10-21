# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Weight-only fake quantization for evaluation (megatron/training/quant_eval.py)."""

import torch

from megatron.training.quant_eval import fake_quantize_, maybe_quantize_for_eval


def test_fake_quantize_levels_and_error():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q = fake_quantize_(w.clone(), bits=4, group=32)
    groups = q.reshape(-1, 32)
    assert all(len(torch.unique(g)) <= 16 for g in groups)  # 4 bits per group
    # the error is at most half a step: (max - min) / 15 / 2 per group
    x = w.reshape(-1, 32)
    step = (x.amax(1) - x.amin(1)) / 15
    assert ((q.reshape(-1, 32) - x).abs().amax(1) <= step / 2 + 1e-6).all()


def test_maybe_quantize_skips_embeddings_routers_norms(monkeypatch):
    model = torch.nn.ModuleDict({
        "embedding": torch.nn.Embedding(100, 64),
        "linear_fc1": torch.nn.Linear(64, 128, bias=True),
        "router": torch.nn.Linear(64, 8, bias=False),
        "norm": torch.nn.LayerNorm(64),
        "output_layer": torch.nn.Linear(64, 100, bias=False),
    })
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    monkeypatch.setenv("QUANT_EVAL_BITS", "4")
    maybe_quantize_for_eval(model)
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {"linear_fc1.weight"}


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("QUANT_EVAL_BITS", raising=False)
    lin = torch.nn.Linear(64, 64)
    w = lin.weight.detach().clone()
    maybe_quantize_for_eval(lin)
    assert torch.equal(lin.weight, w)
