# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.moe_health import (
    _aggregate_layer_values,
    _CachingDataIterator,
    clear_expert_utilization_tracker,
    compute_expert_load_metrics,
    compute_normalized_entropy,
    compute_router_score_distribution,
    expert_rms_statistics,
    get_expert_utilization_tracker,
    get_expert_viability_tracker,
    get_router_stats_tracker,
    local_expert_rms,
    mask_routed_moe_layer,
    report_moe_health_metrics,
    save_to_expert_utilization_tracker,
    should_mask_routed_moe_layer,
    update_expert_near_dead_streaks,
)
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker


class FakeWandb:
    def __init__(self):
        self.data = {}

    def log(self, values, _iteration):
        self.data.update(values)


def test_expert_load_metrics_balanced_and_collapsed():
    balanced = compute_expert_load_metrics(torch.tensor([10.0, 10.0, 10.0, 10.0]))
    assert balanced[0].item() == 0
    assert balanced[1].item() == 0
    assert balanced[2].item() == 0.25
    assert math.isclose(balanced[3].item(), 1.0, rel_tol=1e-6)
    assert balanced[4].item() == 0.0
    assert balanced[5].item() == 0.0

    collapsed = compute_expert_load_metrics(torch.tensor([40.0, 0.0, 0.0, 0.0]))
    assert collapsed[0].item() == 3
    assert collapsed[1].item() == 3
    assert collapsed[2].item() == 1.0
    assert collapsed[3].item() == 0.0
    assert collapsed[4].item() == 3.0
    assert math.isclose(collapsed[5].item(), math.sqrt(3.0), rel_tol=1e-6)


def test_expert_load_metrics_detects_near_dead_expert():
    dead, near_dead, _, normalized_entropy, _, _ = compute_expert_load_metrics(
        torch.tensor([1000.0, 1.0, 1000.0, 1000.0])
    )
    assert dead.item() == 0
    assert near_dead.item() == 1
    assert 0.0 < normalized_entropy.item() < 1.0


def test_expert_load_metrics_max_vio_and_cv_match_reference_formulas():
    _, dead_experts, _, _, max_vio, cv = compute_expert_load_metrics(torch.tensor([0.0, 10.0]))
    assert dead_experts.item() == 1
    assert max_vio.item() == 1.0
    assert cv.item() == 1.0


def test_router_score_distribution_matches_configured_score_function():
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    softmax_distribution = compute_router_score_distribution(logits, "softmax")
    torch.testing.assert_close(softmax_distribution, torch.softmax(logits, dim=-1))

    unbiased = compute_router_score_distribution(logits, "sigmoid")
    biased = compute_router_score_distribution(
        logits, "sigmoid", expert_bias=torch.tensor([-0.8, 0.0, 0.8])
    )
    torch.testing.assert_close(biased.sum(dim=-1), torch.ones(1))
    assert unbiased.argmax(dim=-1).item() == 0
    assert biased.argmax(dim=-1).item() == 2


def test_normalized_entropy_spans_zero_to_one():
    distributions = torch.tensor([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        compute_normalized_entropy(distributions), torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0
    )


def test_utilization_tracker_keeps_selected_and_dispatched_counts():
    tracker = get_expert_utilization_tracker()
    tracker.clear()

    save_to_expert_utilization_tracker(
        torch.tensor([4.0, 0.0]), torch.tensor([2.0, 0.0]), layer_number=1, num_layers=1
    )
    save_to_expert_utilization_tracker(
        torch.tensor([0.0, 4.0]), torch.tensor([0.0, 4.0]), layer_number=1, num_layers=1
    )

    torch.testing.assert_close(tracker["selected_values"], torch.tensor([[4.0, 4.0]]))
    torch.testing.assert_close(tracker["dispatched_values"], torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(
        tracker["selected_near_dead_streaks"], torch.tensor([[0, 0]], dtype=torch.int32)
    )

    tracker.clear()


def test_near_dead_streak_reaches_persistence_threshold_and_resets():
    streaks = torch.zeros(1, 3, dtype=torch.int32)
    near_dead = torch.tensor([[True, False, True]])

    for _ in range(100):
        update_expert_near_dead_streaks(streaks, near_dead)

    torch.testing.assert_close(streaks, torch.tensor([[100, 0, 100]], dtype=torch.int32))
    assert (streaks >= 100).sum().item() == 2

    update_expert_near_dead_streaks(streaks, torch.tensor([[False, False, True]]))
    torch.testing.assert_close(streaks, torch.tensor([[0, 0, 101]], dtype=torch.int32))


def test_step_clear_preserves_near_dead_streaks():
    tracker = get_expert_utilization_tracker()
    tracker.clear()
    save_to_expert_utilization_tracker(
        torch.tensor([4.0, 0.0]), torch.tensor([4.0, 0.0]), layer_number=1, num_layers=1
    )
    tracker["selected_near_dead_streaks"].fill_(7)

    clear_expert_utilization_tracker()

    torch.testing.assert_close(
        tracker["selected_near_dead_streaks"], torch.tensor([[7, 7]], dtype=torch.int32)
    )
    assert tracker["selected_values"].sum().item() == 0
    assert tracker["dispatched_values"].sum().item() == 0
    tracker.clear()


def test_near_dead_streak_rejects_mismatched_shapes():
    streaks = torch.zeros(1, 2, dtype=torch.int32)
    try:
        update_expert_near_dead_streaks(streaks, torch.zeros(2, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected mismatched streak and mask shapes to fail")


def test_aggregate_layer_values():
    assert _aggregate_layer_values([]) == {}
    torch.testing.assert_close(
        torch.tensor(list(_aggregate_layer_values([1.0, 3.0, 5.0]).values())),
        torch.tensor([3.0, 1.0, 5.0]),
    )


def test_expert_rms_statistics_supports_grouped_mlp_layout():
    class Config:
        hidden_size = 4

    class FakeGroupedMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.weight1 = torch.nn.Parameter(torch.ones(4, 16))
            self.weight2 = torch.nn.Parameter(torch.full((8, 4), 2.0))

    experts = FakeGroupedMLP()
    sq_sums, counts = expert_rms_statistics(experts, num_local_experts=2)
    torch.testing.assert_close(counts, torch.tensor([48.0, 48.0]))
    torch.testing.assert_close(sq_sums, torch.tensor([96.0, 96.0]))

    rms = local_expert_rms(experts, num_local_experts=2)
    torch.testing.assert_close(rms, torch.full((2,), math.sqrt(2.0)))


def test_mask_routed_moe_layer_context():
    assert not should_mask_routed_moe_layer(3)
    with mask_routed_moe_layer(3):
        assert should_mask_routed_moe_layer(3)
        assert not should_mask_routed_moe_layer(4)
    assert not should_mask_routed_moe_layer(3)


def test_caching_data_iterator_replays_microbatches():
    source = iter([{"batch": 1}, {"batch": 2}])
    caching_iterator = _CachingDataIterator(source)

    assert next(caching_iterator) == {"batch": 1}
    assert next(caching_iterator) == {"batch": 2}

    caching_iterator.rewind()
    assert next(caching_iterator) == {"batch": 1}
    assert next(caching_iterator) == {"batch": 2}


def test_aux_metrics_use_router_wandb_groups():
    tracker = MoEMetricsTracker()
    tracker.record("load_balancing_loss", torch.tensor(1.0), layer_number=1, num_layers=2)
    tracker.record("load_balancing_loss", torch.tensor(2.0), layer_number=2, num_layers=2)
    wandb = FakeWandb()

    tracker._log_scalars(
        {"load_balancing_loss": torch.tensor(1.5)}, iteration=3, writer=None, wandb_writer=wandb
    )
    tracker._log_per_layer(
        loss_scale=1.0,
        metric_names=["load_balancing_loss"],
        iteration=3,
        writer=None,
        wandb_writer=wandb,
    )

    assert set(wandb.data) == {
        "router-aggregates/load_balancing_loss",
        "router-layers/load_balancing_loss_layer_0",
        "router-layers/load_balancing_loss_layer_1",
    }


def test_health_wandb_groups_are_compact(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    groups = SimpleNamespace(pp=object(), dp_cp_gtp_remat=object(), tp_cp=object(), ep=object())

    utilization = get_expert_utilization_tracker()
    utilization.clear()
    utilization.update(
        selected_values=torch.tensor([[6.0, 2.0], [4.0, 4.0]]),
        dispatched_values=torch.tensor([[5.0, 1.0], [4.0, 4.0]]),
        selected_near_dead_streaks=torch.zeros(2, 2, dtype=torch.int32),
        reduce_group=None,
    )

    router = get_router_stats_tracker()
    router.clear()
    router.update(
        sum_max_score=torch.tensor([1.5, 1.0]),
        sum_score_entropy=torch.tensor([1.0, 1.5]),
        sum_logits=torch.tensor([[2.0, 0.0], [1.0, 1.0]]),
        token_count=torch.tensor([2.0, 2.0]),
        reduce_group=None,
    )

    viability = get_expert_viability_tracker()
    viability.clear()
    viability.update(
        routed_sq_sum=torch.tensor([4.0, 9.0]),
        input_sq_sum=torch.tensor([4.0, 4.0]),
        output_sq_sum=torch.tensor([4.0, 9.0]),
        routed_count=torch.tensor([1.0, 1.0]),
        input_count=torch.tensor([1.0, 1.0]),
        output_count=torch.tensor([1.0, 1.0]),
        weight_sq_sum=torch.tensor([[1.0, 4.0], [4.0, 9.0]]),
        weight_count=torch.ones(2, 2),
        grad_sq_sum=torch.ones(2, 2),
        grad_count=torch.ones(2, 2),
        initial_rms=torch.ones(2, 2),
    )

    wandb = FakeWandb()
    report_moe_health_metrics(
        iteration=3,
        wandb_writer=wandb,
        per_layer_logging=True,
        expert_viability_metrics=True,
        pg_collection=groups,
    )

    grouped = {}
    for name in wandb.data:
        prefix = name.split("/", 1)[0]
        grouped[prefix] = grouped.get(prefix, 0) + 1
    assert grouped == {
        "router-layers": 10,
        "router-aggregates": 15,
        "viability-layers": 8,
        "viability-aggregates": 9,
    }
