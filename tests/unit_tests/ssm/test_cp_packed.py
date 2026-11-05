# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Packed batches (THD, document masking) under context parallelism for the FLA mixers and Engram
(fla_cp.ThdStream): Megatron's THD CP layout <-> FLA's stream; Complex KDA and Engram at CP=2 compute what
they compute at CP=1 on the whole packed batch -- outputs, input gradients and (summed over the CP ranks)
parameter gradients -- with documents that span the rank boundary. Needs 2 GPUs:

    python -m torch.distributed.run --nproc-per-node 2 -m pytest -q tests/unit_tests/ssm/test_cp_packed.py
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import fla_cp
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

CP = 2
# padded to multiples of 2 cp; the middle document spans the boundary between the two stream pieces
LENGTHS = (60, 132, 48)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPUs"),
    pytest.mark.skipif(Utils.world_size < CP, reason=f"needs {CP} GPUs"),
]


@pytest.fixture
def cp_groups():
    Utils.initialize_model_parallel(1, 1, context_parallel_size=CP)
    try:
        model_parallel_cuda_manual_seed(123)
        cp_group = parallel_state.get_context_parallel_group()
        solo = [torch.distributed.new_group([r]) for r in range(torch.distributed.get_world_size())]
        yield parallel_state.get_tensor_model_parallel_group(), cp_group, solo[torch.distributed.get_rank()]
    finally:
        Utils.destroy_model_parallel()


def _packed(lengths):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                           cu_seqlens_kv_padded=cu, max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))


def _close(got, ref, frac, msg=""):
    err = (got - ref).abs().max() / ref.abs().max().clamp(min=1e-12)
    assert err < frac, f"{msg}: relative error {float(err):.2e}"


def test_rank_indices_match_transformer_engine():
    tex = pytest.importorskip("transformer_engine_torch")
    cu = torch.tensor([0, 60, 192, 240], dtype=torch.int32, device="cuda")
    ours = fla_cp.thd_rank_indices(cu.cpu(), CP)
    for r in range(CP):
        te = tex.thd_get_partitioned_indices(cu, 240, CP, r)
        assert torch.equal(ours[r], te.cpu().long()), r


def test_stream_roundtrip(cp_groups):
    _, cp_group, _ = cp_groups
    rank = cp_group.rank()
    params = _packed(LENGTHS)
    st = fla_cp.thd_stream(params.cu_seqlens_q_padded, cp_group)
    torch.manual_seed(0)
    full = torch.randn(sum(LENGTHS), 8, device="cuda")
    local = full[st.local_idx]
    piece = st.to_stream(local)
    torch.testing.assert_close(piece, full[rank * st.n:(rank + 1) * st.n])
    torch.testing.assert_close(st.from_stream(piece), local)
    assert fla_cp.thd_stream(params.cu_seqlens_q_padded, cp_group) is st  # one per micro-batch


def _ckda(pg):
    from megatron.core.ssm.complex_kda import ComplexKDA

    config = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=4, kv_channels=32, linear_key_head_dim=32,
        linear_num_key_heads=4, experimental_attention_variant="complex_kda", linear_attention_freq=3,
        linear_output_gate="lowrank", linear_gate_init_style="shipped", linear_drop_qkv_silu=True,
        linear_beta_max=2.0, context_parallel_size=CP,
    )
    torch.manual_seed(1234)  # the FLA layer initialises on the CPU: the same weights on every rank
    return ComplexKDA(config, layer_number=1, pg_collection=pg).cuda()


def _check_layer(mixer, ref, cp_group, hidden_dim, run):
    """run(layer, x, params) -> output; the CP layer on this rank's THD tokens, the reference on all."""
    params = _packed(LENGTHS)
    st = fla_cp.thd_stream(params.cu_seqlens_q_padded, cp_group)
    torch.manual_seed(0)
    hidden = torch.randn(sum(LENGTHS), 1, hidden_dim, device="cuda")
    dout = torch.randn(sum(LENGTHS), 1, hidden_dim, device="cuda")
    x = hidden[st.local_idx].clone().requires_grad_(True)
    out = run(mixer, x, params, st.local_idx)
    (out.float() * dout[st.local_idx]).sum().backward()
    x_ref = hidden.clone().requires_grad_(True)
    out_ref = run(ref, x_ref, params, None)
    (out_ref.float() * dout).sum().backward()
    _close(out.detach(), out_ref.detach()[st.local_idx], 1e-2, "output")
    _close(x.grad, x_ref.grad[st.local_idx], 1e-2, "input grad")
    for (n, p), (_, p_ref) in zip(mixer.named_parameters(), ref.named_parameters()):
        if p_ref.grad is None:
            continue
        g = p.grad.clone()
        torch.distributed.all_reduce(g, group=cp_group)
        _close(g, p_ref.grad, 1e-2, n)


def test_complex_kda_packed_cp_matches_single_rank(cp_groups):
    tp, cp_group, solo = cp_groups
    mixer = _ckda(ProcessGroupCollection(tp=tp, cp=cp_group))
    ref = _ckda(ProcessGroupCollection(tp=tp, cp=solo))
    ref.load_state_dict(mixer.state_dict())
    with torch.no_grad():  # negative gates on half the channels (the sign gauge across ranks)
        g = torch.Generator(device="cpu").manual_seed(0)
        flip = torch.where(torch.rand(mixer.layer.dt_bias.shape, generator=g) < 0.5, -1.0, 1.0).cuda()
        mixer.layer.dt_bias.mul_(flip)
        ref.layer.dt_bias.mul_(flip)

    def run(layer, x, params, _):
        return layer(x, packed_seq_params=params)[0]

    _check_layer(mixer, ref, cp_group, 128, run)


def test_engram_packed_cp_matches_single_rank(cp_groups):
    from megatron.core.ssm.engram import EngramLayer

    tp, cp_group, solo = cp_groups
    config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4, engram_ngram_orders=[2, 3],
                               engram_num_heads=2, engram_table_size=1000, engram_embed_dim=16,
                               context_parallel_size=CP)
    torch.manual_seed(7)
    mixer = EngramLayer(config, layer_number=1, pg_collection=ProcessGroupCollection(tp=tp, cp=cp_group)).cuda()
    ref = EngramLayer(config, layer_number=1, pg_collection=ProcessGroupCollection(tp=tp, cp=solo)).cuda()
    ref.load_state_dict(mixer.state_dict())
    with torch.no_grad():  # a non-zero conv, so the halo matters
        w = torch.randn_like(mixer.conv1d.weight)
        mixer.conv1d.weight.copy_(w)
        ref.conv1d.weight.copy_(w)
    torch.manual_seed(3)
    ids = torch.randint(0, 50000, (1, sum(LENGTHS)), device="cuda")

    def run(layer, x, params, local_idx):
        layer.set_input_ids(ids if local_idx is None else ids[:, local_idx])
        return layer(x, packed_seq_params=params)

    _check_layer(mixer, ref, cp_group, 64, run)
