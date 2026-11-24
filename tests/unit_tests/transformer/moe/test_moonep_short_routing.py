# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MoonEP's dense [S, k] view of a routing map that holds fewer than k experts in a row.

CPU only: ``dense_topk_from_routing`` / ``dispatch_preprocess`` touch no CUDA and need no moonep
package. The planner (MoonEP ``run_c1``) counts the [S, k] indices per expert and compares that
histogram with ``tokens_per_expert``, then places tokens at the offsets those counts imply -- a
mismatch writes past an expert's allocation (cudaErrorIllegalAddress on ec3eb8b, the trap on
e9e8238). The invariant is therefore ``bincount(indices) == tokens_per_expert``; the pre-fix
dispatcher (``probs.masked_fill(~routing_map, -1.0).topk(k)`` + ``routing_map.sum(0)``) breaks it
on every row that holds fewer than k routed experts, and these tests pin both sides: the old
formula is recomputed here to show it disagrees, the dispatcher's output must not.
"""

from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.moonep import (
    MoEMoonEPTokenDispatcher,
    dense_topk_from_routing,
)


def _dispatcher(K, E):
    """A dispatcher without __init__ (it asserts HAVE_MOONEP / CUDA): only dispatch_preprocess."""
    d = MoEMoonEPTokenDispatcher.__new__(MoEMoonEPTokenDispatcher)
    d.config = SimpleNamespace(moe_router_topk=K, num_moe_experts=E)
    return d


def _probs(S, E, routed, seed=0):
    """[S, E] sparse probs of the fused router: 0 outside the routed experts, positive inside."""
    g = torch.Generator().manual_seed(seed)
    probs = torch.zeros(S, E, dtype=torch.float32)
    for s, experts in enumerate(routed):
        w = torch.rand(len(experts), generator=g) + 0.1
        probs[s, torch.tensor(experts)] = w / w.sum()
    return probs


def _map(S, E, routed):
    m = torch.zeros(S, E, dtype=torch.bool)
    for s, experts in enumerate(routed):
        m[s, torch.tensor(experts)] = True
    return m


def _pre_fix(probs, routing_map, k):
    """What dispatch_preprocess computed before the fix: indices, weights, counts."""
    topk = probs.masked_fill(~routing_map, -1.0).topk(k, dim=-1).indices
    return topk, probs.gather(-1, topk).float(), routing_map.sum(dim=0, dtype=torch.int32)


def _check_planner_invariant(indices, weights, counts):
    """The three properties MoonEP's planner needs from the dense [S, k] view."""
    assert torch.equal(
        torch.bincount(indices.reshape(-1), minlength=counts.numel()).to(torch.int32), counts
    ), "planner histogram of the indices must equal tokens_per_expert"
    for row, row_indices in enumerate(indices):
        for slot, e in enumerate(row_indices.tolist()):
            if weights[row, slot] == 0:
                continue  # a padded slot: it must carry no weight (checked below)
            assert counts[e] > 0, "a slot with weight must name an expert the map routes to"
    assert torch.isfinite(weights).all()


def test_well_formed_rows_keep_the_exact_pre_fix_numerics():
    """k routed experts per row: identical indices, weights and counts (bit for bit)."""
    K, E, S = 5, 12, 64
    routed = [sorted(torch.randperm(E)[:K].tolist()) for _ in range(S)]
    probs, routing_map = _probs(S, E, routed), _map(S, E, routed)
    old_topk, old_w, old_counts = _pre_fix(probs, routing_map, K)

    indices, weights, counts = dense_topk_from_routing(probs, routing_map, K)
    assert torch.equal(indices, old_topk)
    assert torch.equal(weights, old_w)
    assert torch.equal(counts, old_counts)
    assert torch.equal(counts, routing_map.sum(dim=0, dtype=torch.int32))

    d = _dispatcher(K, E)
    d.dispatch_preprocess(torch.zeros(S, 4), routing_map, probs)
    assert torch.equal(d.topk, old_topk.to(torch.int32))
    assert torch.equal(d.tokens_per_expert, old_counts)
    _check_planner_invariant(d.topk.long(), weights, d.tokens_per_expert)


def test_one_of_k_row_is_padded_and_the_pre_fix_view_disagrees():
    """A row holding 1 of 5 experts: the pre-fix top-k named 4 empty experts; the fix pads."""
    K, E = 5, 8
    routed = [[3]]
    probs, routing_map = _probs(1, E, routed), _map(1, E, routed)

    # the bug: the -1.0 fill names experts 0, 1, 2, 4 -- which the map says are empty
    old_topk, _, old_counts = _pre_fix(probs, routing_map, K)
    assert old_topk[0].tolist() == [3, 0, 1, 2, 4]
    assert old_counts[0] == 0 and old_counts[3] == 1
    assert torch.bincount(old_topk.reshape(-1), minlength=E)[0] == 1  # disagrees with 0

    d = _dispatcher(K, E)
    _, weights = d.dispatch_preprocess(torch.zeros(1, 4), routing_map, probs)
    assert d.topk[0].tolist() == [3, 3, 3, 3, 3]
    assert weights[0].tolist() == [probs[0, 3].item(), 0.0, 0.0, 0.0, 0.0]
    assert d.tokens_per_expert.tolist() == [0, 0, 0, 5, 0, 0, 0, 0]
    _check_planner_invariant(d.topk.long(), weights, d.tokens_per_expert)


def test_empty_row_contributes_nothing():
    """A row with no routed expert: every slot goes to expert 0 with weight 0."""
    K, E = 5, 8
    probs = torch.zeros(2, E, dtype=torch.float32)
    routing_map = torch.zeros(2, E, dtype=torch.bool)
    routing_map[1, 6] = True  # the second row is well formed, so the counts mix both cases
    probs[1, 6] = 1.0

    d = _dispatcher(K, E)
    _, weights = d.dispatch_preprocess(torch.zeros(2, 4), routing_map, probs)
    assert d.topk[0].tolist() == [0, 0, 0, 0, 0]
    assert weights[0].tolist() == [0.0] * K
    assert d.topk[1].tolist() == [6] * K
    assert weights[1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert d.tokens_per_expert.tolist() == [5, 0, 0, 0, 0, 0, 5, 0]
    _check_planner_invariant(d.topk.long(), weights, d.tokens_per_expert)


def test_mixed_rows_keep_the_planner_invariant():
    """Full, short and empty rows together, the shape the real planner sees."""
    K, E, S = 5, 16, 9
    routed = [sorted(torch.randperm(E)[:K].tolist()) for _ in range(S)]
    routed[2] = [routed[2][0]]
    routed[5] = []
    routed[7] = sorted(torch.randperm(E)[:3].tolist())
    probs, routing_map = _probs(S, E, routed), _map(S, E, routed)

    old_topk, _, old_counts = _pre_fix(probs, routing_map, K)
    assert not torch.equal(
        torch.bincount(old_topk.reshape(-1), minlength=E).to(torch.int32), old_counts
    ), "the pre-fix view must disagree on short rows (this is the illegal-access bug)"

    d = _dispatcher(K, E)
    _, weights = d.dispatch_preprocess(torch.zeros(S, 4), routing_map, probs)
    _check_planner_invariant(d.topk.long(), weights, d.tokens_per_expert)
    # the well-formed rows are untouched
    for s in (0, 1, 3, 4, 6, 8):
        assert d.topk[s].tolist() == old_topk[s].tolist()
        assert weights[s].tolist() == probs[s].gather(0, old_topk[s]).tolist()
