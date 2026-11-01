# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""arch_metrics.token_rows: the metric passes look at a strided token subsample, not fp32 copies of
the whole micro-batch (a 64k-token micro-batch ran the v3 main run out of memory)."""

import torch

from megatron.core.transformer import arch_metrics


def test_token_rows_caps_and_keeps_inner_dims():
    x = torch.arange(65536 * 2 * 6, dtype=torch.float32).reshape(65536, 2, 3, 2)
    r = arch_metrics.token_rows(x, 2)
    assert r.shape[1:] == (3, 2) and r.shape[0] <= arch_metrics.MAX_TOKENS
    assert torch.equal(r[0], x.reshape(-1, 3, 2)[0])
    assert r.data_ptr() == x.data_ptr()  # a view, no copy


def test_token_rows_small_input_unchanged():
    x = torch.randn(10, 4)
    assert torch.equal(arch_metrics.token_rows(x), x)


def test_token_rows_mean_unbiased_on_a_constant_pattern():
    x = torch.arange(100000, dtype=torch.float32).remainder(7).reshape(-1, 1)
    full, sub = x.mean(), arch_metrics.token_rows(x).mean()
    assert abs(float(full - sub)) < 0.05
