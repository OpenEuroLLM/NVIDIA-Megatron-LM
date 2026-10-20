# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""KronLSTM (`kronlstm`, fla_hybrid.KronLSTM) under context parallelism: CP=2 computes what CP=1
computes on the whole sequence (to TF32 noise: the cluster container enables it) -- outputs, input gradients and parameter gradients -- with the
state relayed between ranks (fla/ops/kronlstm/cp.py), for both forget gates."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import fla_hybrid
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN, HEAD = 128, 32


def _cp_chunks(t: torch.Tensor, cp_size: int, rank: int) -> torch.Tensor:
    """This rank's load-balanced sequence chunks (r and 2cp-1-r) of [s, ...]."""
    chunks = t.chunk(2 * cp_size, dim=0)
    return torch.cat([chunks[rank], chunks[2 * cp_size - 1 - rank]], dim=0)


@pytest.mark.skipif(fla_hybrid._FLAKronLSTM is None, reason="no fla.layers.kronlstm")
@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("gate", ["softplus", "signed_sigmoid2"])
def test_kronlstm_cp_matches_full_sequence(gate):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1, hidden_size=HIDDEN, num_attention_heads=4, linear_key_head_dim=HEAD,
            linear_num_key_heads=HIDDEN // HEAD, experimental_attention_variant="kronlstm",
            linear_attention_freq=2, linear_fla_gate=gate, linear_fla_allow_neg_eigval=True,
            context_parallel_size=2,
        )
        tp = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        rank, cp = cp_group.rank(), cp_group.size()
        solo = [torch.distributed.new_group([r]) for r in range(torch.distributed.get_world_size())]
        mixer = fla_hybrid.KronLSTM(config, layer_number=1,
                                    pg_collection=ProcessGroupCollection(tp=tp, cp=cp_group)).cuda()
        ref = fla_hybrid.KronLSTM(config, layer_number=1, pg_collection=ProcessGroupCollection(
            tp=tp, cp=solo[torch.distributed.get_rank()])).cuda()
        with torch.no_grad():
            if gate.startswith("signed"):  # negative gates on half the channels
                g = torch.Generator(device="cpu").manual_seed(0)
                mixer.layer.dt_bias.mul_(torch.where(
                    torch.rand(mixer.layer.dt_bias.shape, generator=g) < 0.5, -1.0, 1.0).cuda())
        ref.load_state_dict(mixer.state_dict())
        torch.manual_seed(0)  # the same full input and cotangent on every rank
        hidden = torch.randn(512, 2, HIDDEN, device="cuda")
        dout = torch.randn(512, 2, HIDDEN, device="cuda")

        x = _cp_chunks(hidden, cp, rank).clone().requires_grad_(True)
        out, _ = mixer(x)
        (out * _cp_chunks(dout, cp, rank)).sum().backward()
        x_ref = hidden.clone().requires_grad_(True)
        out_ref, _ = ref(x_ref)
        (out_ref * dout).sum().backward()

        torch.testing.assert_close(out, _cp_chunks(out_ref, cp, rank), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(x.grad, _cp_chunks(x_ref.grad, cp, rank), rtol=1e-2, atol=1e-2)
        # each CP rank holds its share of a parameter's gradient; their sum is the full one
        for (n, p), (_, p_ref) in zip(mixer.named_parameters(), ref.named_parameters()):
            grad = p.grad.clone()
            torch.distributed.all_reduce(grad, group=cp_group)
            torch.testing.assert_close(grad, p_ref.grad, rtol=1e-2, atol=1e-2, msg=n)
    finally:
        Utils.destroy_model_parallel()
