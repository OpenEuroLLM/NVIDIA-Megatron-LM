# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""ROUTER_TRACE_OUT: the routers' top-k expert ids per token, in token order."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.router_trace import maybe_trace_routers
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def test_trace_records_topk_ids(tmp_path, monkeypatch):
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    cfg = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, num_moe_experts=8,
                            moe_router_topk=3, moe_router_score_function="sigmoid",
                            moe_router_load_balancing_type="none", add_bias_linear=False)
    routers = torch.nn.ModuleList([TopKRouter(cfg, pg_collection=ProcessGroupCollection.use_mpu_process_groups()) for _ in range(2)]).cuda()
    for i, r in enumerate(routers):
        r.set_layer_number(i + 1)
    out = tmp_path / "trace.pt"
    monkeypatch.setenv("ROUTER_TRACE_OUT", str(out))
    save = maybe_trace_routers(routers)
    x = torch.randn(10, 32, device="cuda")
    expected = {}
    for r in routers:
        _, rmap = r(x)
        expected[r.layer_number] = rmap
    save()
    got = torch.load(out)
    assert got["num_experts"] == 8 and got["topk"] == 3
    for ln, rmap in expected.items():
        ids = got["trace"][ln][0].long()
        assert ids.shape == (10, 3)
        rebuilt = torch.zeros_like(rmap).scatter_(1, ids.cuda(), True)
        assert torch.equal(rebuilt, rmap)
    Utils.destroy_model_parallel()
