# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""moe_router_stay_bonus (routing locality for expert offloading): a token's expert selection gets a bonus on the
experts its previous token selects without the bonus. Bonus ~0 routes as before; a bonus of the order of the score
gaps raises the share of experts shared with the previous token; a huge bonus only shifts the plain selection by one
token (the bonus is on the previous token's plain selection, so training stays parallel); the gate weights of the
selected experts stay the router's (selection only), for quantile balancing (bonus on the logits) and the plain top-k
path with expert bias (bonus on the scores). One GPU:

    python -m torch.distributed.run --nproc-per-node 1 -m pytest -q tests/unit_tests/transformer/moe/test_router_stay_bonus.py
"""

from typing import cast

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

E, K = 16, 3
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def mp():
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    yield
    Utils.destroy_model_parallel()


def _router(balancing, bonus):
    kw = dict(moe_router_load_balancing_type="quantile_balancing") if balancing == "qb" else dict(
        moe_router_load_balancing_type="none", moe_router_enable_expert_bias=True)
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, num_moe_experts=E,
                               use_cpu_initialization=True, moe_router_score_function="sigmoid", moe_router_topk=K,
                               moe_aux_loss_coeff=0, bf16=True, params_dtype=torch.bfloat16, add_bias_linear=False,
                               moe_router_stay_bonus=bonus, **kw)
    sub = get_submodules(get_gpt_layer_local_submodules(num_experts=E, moe_grouped_gemm=False).mlp)
    torch.manual_seed(0)
    return cast(Router, MoELayer(config, sub).router).cuda()


def _stay(routing_map, bsz):
    m = routing_map.float()
    return float((m[bsz:] * m[:-bsz]).sum() / (K * (m.shape[0] - bsz)))


@pytest.mark.parametrize("balancing,moderate", [("qb", 0.1), ("topk", 0.03)])
def test_stay_bonus(mp, balancing, moderate):
    seq, bsz = 256, 2
    x = torch.randn(seq, bsz, 32, generator=torch.Generator().manual_seed(1)).cuda().bfloat16()
    base = _router(balancing, 0.0)
    with torch.no_grad():
        p0, m0 = base(x)

    def run(bonus):
        r = _router(balancing, bonus)
        r.load_state_dict(base.state_dict())
        with torch.no_grad():
            return r(x)

    pz, mz = run(1e-30)  # the bonus path at an effectively zero bonus routes the same
    assert torch.equal(m0, mz)
    pm, mm = run(moderate)
    assert mm.sum() == m0.sum() == seq * bsz * K
    assert _stay(mm, bsz) > _stay(m0, bsz) + 0.1, (_stay(mm, bsz), _stay(m0, bsz))
    # selection only: tokens routed identically get identical gate weights
    rows = (m0 == mm).all(-1)
    assert rows.any() and not rows.all()
    torch.testing.assert_close(pm[rows], p0[rows])
    _, mh = run(50.0)  # huge bonus: each token takes its predecessor's plain selection, the first tokens their own
    assert torch.equal(mh[bsz:], m0[:-bsz]) and torch.equal(mh[:bsz], m0[:bsz])


@pytest.mark.parametrize("balancing", ["qb", "topk"])
def test_stay_rate_metric(mp, balancing):
    """moe_router_log_stay_rate records the stay rate of the final selection in a grad-enabled forward (the
    quantile-balancing routing itself runs under no_grad), nothing under no_grad."""
    from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker, get_moe_metrics_tracker

    seq, bsz = 128, 2
    x = torch.randn(seq, bsz, 32, generator=torch.Generator().manual_seed(2)).cuda().bfloat16()
    r = _router(balancing, 0.1)
    r.config.moe_router_log_stay_rate = True
    r.set_layer_number(1)
    destroy_moe_metrics_tracker()
    with torch.no_grad():
        r(x)
    assert "stay_rate" not in get_moe_metrics_tracker().metrics
    _, m = r(x)
    got = float(get_moe_metrics_tracker().metrics["stay_rate"].values[0])
    assert abs(got - _stay(m, bsz)) < 1e-5 and got > 0
    destroy_moe_metrics_tracker()
