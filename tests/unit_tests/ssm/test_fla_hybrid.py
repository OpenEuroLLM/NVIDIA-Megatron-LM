# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The FLA hybrid wrappers (kda / signed_kda / panini) against the fla on the path.

Construction only, on CPU: what is checked is which layer gets built and what
reaches its constructor, which is where these wrappers have gone wrong --
a renamed or re-defaulted fla argument changes the model without an error.
"""

import inspect
from unittest import mock

import pytest
import torch

from megatron.core.ssm import fla_hybrid
from megatron.core.transformer.transformer_config import TransformerConfig

HIDDEN, HEAD_DIM, HEADS = 128, 32, 4


def _config(variant, **overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        linear_key_head_dim=HEAD_DIM,
        linear_num_key_heads=HEADS,
        experimental_attention_variant=variant,
        linear_attention_freq=3,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _spy(cls):
    """A subclass of `cls` that records the kwargs its constructor received."""
    calls = []

    class Spy(cls):
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            super().__init__(*args, **kwargs)

    Spy.__name__ = cls.__name__
    Spy.__init__.__signature__ = inspect.signature(cls.__init__)
    return Spy, calls


needs_signed = pytest.mark.skipif(
    fla_hybrid._FLASignedKDA is None, reason="no signed KDA layer in the installed fla"
)


@needs_signed
@pytest.mark.parametrize("drop_silu", [True, False])
def test_signed_kda_passes_drop_silu_explicitly(drop_silu):
    """The legacy layer defaulted drop_silu to True, the released one to False:
    the wrapper must pass it, or which fla is on the path changes the model."""
    spy, calls = _spy(fla_hybrid._FLASignedKDA)
    with mock.patch.object(fla_hybrid, "_FLASignedKDA", spy):
        fla_hybrid.SignedKDA(_config("signed_kda", linear_fla_drop_silu=drop_silu), layer_number=1)
    assert calls[-1]["drop_silu"] is drop_silu


@needs_signed
def test_signed_kda_default_reproduces_earlier_runs():
    """Defaults = what every signed_kda run so far built (legacy layer defaults)."""
    spy, calls = _spy(fla_hybrid._FLASignedKDA)
    with mock.patch.object(fla_hybrid, "_FLASignedKDA", spy):
        fla_hybrid.SignedKDA(_config("signed_kda"), layer_number=1)
    kw = calls[-1]
    assert kw["drop_silu"] is True
    assert kw["gate_init_style"] == "spread"
    assert kw["output_gate"] == "linear"
    assert kw["allow_neg_eigval"] is True
    assert kw["gate"] == "signed_sigmoid2"


@needs_signed
def test_signed_kda_retune_knob_forces_shipped_base():
    spy, calls = _spy(fla_hybrid._FLASignedKDA)
    with mock.patch.object(fla_hybrid, "_FLASignedKDA", spy):
        wrapper = fla_hybrid.SignedKDA(
            _config("signed_kda", linear_fla_gate_neg_frac=0.25), layer_number=1
        )
    assert calls[-1]["gate_init_style"] == "shipped"
    frac = (wrapper.layer.dt_bias < 0).float().mean().item()
    assert 0.0 < frac < 0.6


@needs_signed
def test_signed_kda_matches_complex_kda_layer():
    """signed_kda and the complex_kda variant are one layer: at the same gate,
    output gate and SiLU setting they build the same parameters."""
    if fla_hybrid._FLASignedKDA.__name__ != "ComplexKimiDeltaAttention":
        pytest.skip("legacy fla: complex_kda needs the released layer")
    from megatron.core.ssm.complex_kda import ComplexKDA

    skda = fla_hybrid.SignedKDA(
        _config(
            "signed_kda", linear_fla_output_gate="lowrank", linear_fla_gate_init_style="shipped"
        ),
        layer_number=1,
    )
    ckda = ComplexKDA(
        _config(
            "complex_kda",
            linear_output_gate="lowrank",
            linear_gate_init_style="shipped",
            linear_drop_qkv_silu=True,
            linear_beta_max=2.0,
        ),
        layer_number=1,
    )
    a, b = skda.layer.state_dict(), ckda.layer.state_dict()
    assert set(a) == set(b)
    assert {k: v.shape for k, v in a.items()} == {k: v.shape for k, v in b.items()}


def test_declared_kwargs_rejects_unknown():
    class Layer(torch.nn.Module):
        def __init__(self, hidden_size, **kwargs):
            super().__init__()

    assert fla_hybrid._declared_kwargs(Layer, {"hidden_size": 4}) == {"hidden_size": 4}
    with pytest.raises(TypeError, match="does not declare"):
        fla_hybrid._declared_kwargs(Layer, {"hidden_size": 4, "tie_sign_across_group": False})


@pytest.mark.skipif(fla_hybrid._FLAKDA is None, reason="no fla.layers.kda")
def test_kda_builds_without_panini():
    """kda must not depend on panini/signed_kda being importable."""
    with mock.patch.object(fla_hybrid, "_FLAPanini", None):
        wrapper = fla_hybrid.KDA(_config("kda"), layer_number=1)
    assert isinstance(wrapper.layer, fla_hybrid._FLAKDA)


def test_missing_layer_names_the_module():
    with mock.patch.object(fla_hybrid, "_FLAPanini", None):
        with pytest.raises(ImportError, match="panini_delta_attention"):
            fla_hybrid.PaniniDeltaAttention(_config("panini"), layer_number=1)
