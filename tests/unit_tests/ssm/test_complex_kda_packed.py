# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Document masking for complex_kda: a packed ``[t, 1, h]`` batch with ``packed_seq_params``
must give each document exactly what it gets alone -- the recurrence and the short
convolutions restart at every boundary -- in outputs and in gradients."""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN, HEAD_DIM, HEADS = 128, 32, 4
LENGTHS = (37, 64, 27)  # not chunk-aligned (fla's chunk is 64)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _layer():
    from megatron.core.ssm.complex_kda import ComplexKDA

    config = TransformerConfig(
        num_layers=2, hidden_size=HIDDEN, num_attention_heads=HEADS, kv_channels=HEAD_DIM,
        linear_key_head_dim=HEAD_DIM, linear_num_key_heads=HEADS,
        experimental_attention_variant="complex_kda", linear_attention_freq=3,
        linear_output_gate="lowrank", linear_gate_init_style="shipped",
        linear_drop_qkv_silu=True, linear_beta_max=2.0,
    )
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    return ComplexKDA(config, layer_number=1).cuda()


def _packed(lengths, pad=0):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                           max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))


def _close(got, ref, frac, msg=""):
    """max |got - ref| within ``frac`` of ref's largest entry: the NGC image forces TF32 for
    cuBLAS, which rounds a packed and a separate batch differently (~1e-3 relative)."""
    err = (got - ref).abs().max() / ref.abs().max().clamp(min=1e-12)
    assert err < frac, f"{msg}: relative error {float(err):.2e}"


def _grads(layer):
    return {n: p.grad.detach().clone() for n, p in layer.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("pad", [0, 5])
def test_packed_matches_separate(pad):
    layer = _layer()
    docs = [torch.randn(n, 1, HIDDEN, device="cuda") for n in LENGTHS]
    # targets: a fixed random projection of each output, so every position matters
    probe = [torch.randn(n, 1, HIDDEN, device="cuda") for n in LENGTHS]

    ref_out, xs = [], []
    layer.zero_grad(set_to_none=True)
    for d, w in zip(docs, probe):
        x = d.clone().requires_grad_(True)
        y, _ = layer(x)
        (y.float() * w).sum().backward()
        ref_out.append(y.detach())
        xs.append(x.grad.detach())
    ref_grads = _grads(layer)

    layer.zero_grad(set_to_none=True)
    tail = torch.randn(pad, 1, HIDDEN, device="cuda")
    x = torch.cat([*docs, tail]).requires_grad_(True)
    y, _ = layer(x, packed_seq_params=_packed(LENGTHS))
    (y[: sum(LENGTHS)].float() * torch.cat(probe)).sum().backward()
    got_grads = _grads(layer)

    _close(y[: sum(LENGTHS)].detach(), torch.cat(ref_out), 1e-2, "output")
    _close(x.grad[: sum(LENGTHS)], torch.cat(xs), 1e-2, "input grad")
    if pad:  # the padding is its own segment: no gradient reaches it from the documents
        assert torch.count_nonzero(x.grad[sum(LENGTHS):]) == 0
    assert ref_grads.keys() == got_grads.keys()
    for n in ref_grads:
        _close(got_grads[n], ref_grads[n], 1e-2, n)


def test_packed_differs_from_unmasked():
    """The check has teeth: without the boundaries the later documents see the earlier ones."""
    layer = _layer()
    x = torch.randn(sum(LENGTHS), 1, HIDDEN, device="cuda")
    with torch.no_grad():
        masked, _ = layer(x, packed_seq_params=_packed(LENGTHS))
        unmasked, _ = layer(x)
    n0 = LENGTHS[0]
    # the first document is the same either way, up to the varlen vs batch kernels' rounding
    first = (masked[:n0] - unmasked[:n0]).abs().max()
    later = (masked[n0:] - unmasked[n0:]).abs().max()
    assert later > 20 * first, (float(first), float(later))


def test_cu_seqlens_converted_once_per_batch():
    """fla memoizes its varlen index preparation by tensor identity: every layer of a
    micro-batch must hand it the same cu_seqlens object (else a host sync per layer)."""
    a, b = _layer(), _layer()
    packed = _packed(LENGTHS)
    x = torch.randn(sum(LENGTHS) + 3, 1, HIDDEN, device="cuda")
    assert a._cu_seqlens(x, packed) is b._cu_seqlens(x, packed)
    assert a._cu_seqlens(x, _packed(LENGTHS)) is not a._cu_seqlens(x, packed)
