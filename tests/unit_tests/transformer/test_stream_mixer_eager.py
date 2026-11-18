# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""TransformerLayer._mhc_read calls every residual stream mixer as ``mixer(streams, eager=...)`` (the mHC
arch-metrics path takes the eager coefficients, Megatron 1f7bcd072): every mixer must accept the keyword, or an
LDM / IAT / routed / AttnRes model fails at its first forward (the s3 LDM runs did, 2026-11-18)."""

import inspect

import pytest

from megatron.core.transformer.attn_res import AttnResStreamMixer
from megatron.core.transformer.depth_memory import DepthMemoryMixer, DepthMemoryPassThrough
from megatron.core.transformer.internal_attention import IATStreamMixer
from megatron.core.transformer.mhc import MHCMixer
from megatron.core.transformer.routed_residual import RoutedStreamMixer


@pytest.mark.parametrize("cls", [MHCMixer, DepthMemoryMixer, DepthMemoryPassThrough, IATStreamMixer,
                                 RoutedStreamMixer, AttnResStreamMixer])
def test_stream_mixers_accept_eager(cls):
    params = inspect.signature(cls.forward).parameters
    assert "eager" in params and params["eager"].default is False
