# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""TransformerConfig.muon_scale_reference_ffn_hidden_size: a widened dense MLP keeps the Muon update scale of its
reference width (the parameter's ``muon_scale_shapes`` = (reference, actual) global shapes; the optimizer multiplies
the orthogonalized update by scale(reference) / scale(actual)); untagged parameters are unchanged."""

import math
from types import SimpleNamespace

import pytest
import torch

eo = pytest.importorskip("megatron.core.optimizer.emerging_optimizers")


def _opt(mode):
    return SimpleNamespace(_scale_mode=mode)


def test_widened_mlp_gets_the_reference_scale():
    fc1, fc2, other = torch.zeros(2), torch.zeros(2), torch.zeros(2)
    fc1.muon_scale_shapes = ((2 * 3584, 1280), (2 * 4608, 1280))
    fc2.muon_scale_shapes = ((1280, 3584), (1280, 4608))
    corr = eo.TensorParallelMuon._scale_correction
    spectral = _opt("spectral")
    assert corr(spectral, fc1) == pytest.approx(math.sqrt(3584 / 4608))  # 1 / 1.134
    assert corr(spectral, fc2) == pytest.approx(math.sqrt(3584 / 4608))
    assert corr(spectral, other) == 1.0
    shape = _opt("shape_scaling")  # sqrt(max(1, m / n)): fc2 (rows < cols) is not shape-scaled at all
    assert corr(shape, fc1) == pytest.approx(math.sqrt(3584 / 4608))
    assert corr(shape, fc2) == pytest.approx(1.0)


def test_orthogonalize_applies_the_correction():
    p = torch.zeros(8, 4)
    p.muon_scale_shapes = ((8, 4), (16, 4))  # reference half as tall: spectral factor sqrt(8) / sqrt(16)
    grad = torch.randn(8, 4)
    stub = SimpleNamespace(
        _scale_mode="spectral", pg_collection=None, tp_mode="blockwise", split_qkv=False,
        scaled_orthogonalize_fn_with_gtp_remat=lambda p, g, tp_group, partition_dim: g * 2.0,
        _scale_correction=lambda q: eo.TensorParallelMuon._scale_correction(stub, q),
        _effective_qkv_split_shapes=lambda q: None,
    )
    out = eo.TensorParallelMuon.orthogonalize(stub, p, grad)
    torch.testing.assert_close(out, grad * 2.0 * math.sqrt(8 / 16))
    q = torch.zeros(8, 4)
    torch.testing.assert_close(eo.TensorParallelMuon.orthogonalize(stub, q, grad), grad * 2.0)
