# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""attention_length_scaling under context parallelism: a layer at CP 2 (load-balanced sharding,
sbhd and packed thd) gives the CP-1 outputs. Needs 2 GPUs:

    torchrun --nproc-per-node 2 -m pytest tests/unit_tests/transformer/test_attention_length_scaling_cp.py
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_thd_correctness import (
    assert_close,
    gather_sbhd,
    gather_thd,
    make_packed_seq_params,
    shard_sbhd,
    shard_thd,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires 2 GPUs"
)
H, SEQLENS = 128, [24, 40, 64]


def _layer(cp: int, mode: str):
    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, ffn_hidden_size=256, num_attention_heads=4,
        num_query_groups=2, bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0, attention_dropout=0.0, context_parallel_size=cp,
        cp_comm_type="p2p" if cp > 1 else None,
        attention_length_scaling=mode, attention_length_scaling_base=8)
    return TransformerLayer(cfg, get_gpt_layer_with_transformer_engine_spec().submodules).cuda()


def _outputs(cp: int, mode: str, state=None):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(42)
    layer = _layer(cp, mode)
    if state is None:
        torch.manual_seed(0)
        if mode == "ssmax":
            with torch.no_grad():
                layer.self_attention.ssmax_s.uniform_(0.2, 0.6)
        state = {k: v.clone() for k, v in layer.state_dict().items()}
    else:
        layer.load_state_dict(state)
    rank = parallel_state.get_context_parallel_rank()
    torch.manual_seed(1)
    seqs = [torch.randn(n, H, dtype=torch.bfloat16, device="cuda") for n in SEQLENS]
    full = torch.zeros(max(SEQLENS), len(SEQLENS), H, dtype=torch.bfloat16, device="cuda")
    for b, (n, d) in enumerate(zip(SEQLENS, seqs)):
        full[:n, b] = d
    with torch.no_grad():
        sbhd, _ = layer(hidden_states=shard_sbhd(full, rank, cp, 0, 1, False))
        sbhd = gather_sbhd(sbhd, cp, 1, False)
        thd, _ = layer(hidden_states=shard_thd(seqs, SEQLENS, rank, cp, 0, 1, False, H),
                       packed_seq_params=make_packed_seq_params(SEQLENS, cp))
        thd = gather_thd(thd, SEQLENS, cp, 1, False, H)
    Utils.destroy_model_parallel()
    return state, sbhd, thd


@pytest.mark.parametrize("mode", ["logn", "ssmax"])
def test_cp2_matches_cp1(mode):
    state, sbhd1, thd1 = _outputs(1, mode)
    _, sbhd2, thd2 = _outputs(2, mode, state)
    assert_close("sbhd", sbhd2, sbhd1, False)
    assert_close("thd", thd2, thd1, False)
    # the check has teeth: without the scaling the outputs differ
    _, sbhd0, _ = _outputs(1, "none", {k: v for k, v in state.items() if "ssmax" not in k})
    assert (sbhd0 - sbhd1).abs().max() > 1e-2
