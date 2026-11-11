# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""engram_output_scale: the factor on the Engram layers' output (recall surgery)."""

from types import SimpleNamespace

import megatron.core.ssm.ckda_branch as cb
from megatron.core.ssm.engram import engram_output_scale


def _cfg(**kw):
    base = dict(engram_output_scale=1.0, engram_output_scale_start=0, engram_output_scale_ramp=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_default_is_identity():
    assert engram_output_scale(_cfg()) == 1.0
    assert engram_output_scale(SimpleNamespace()) == 1.0


def test_ramp(monkeypatch):
    c = _cfg(engram_output_scale=0.0, engram_output_scale_start=100, engram_output_scale_ramp=50)
    for it, want in ((None, 0.0), (90, 1.0), (100, 1.0), (125, 0.5), (150, 0.0), (500, 0.0)):
        monkeypatch.setattr(cb, "current_iteration", lambda it=it: it)
        assert abs(engram_output_scale(c) - want) < 1e-9, (it, engram_output_scale(c))
    c = _cfg(engram_output_scale=0.5, engram_output_scale_start=100)
    monkeypatch.setattr(cb, "current_iteration", lambda: 101)
    assert engram_output_scale(c) == 0.5
