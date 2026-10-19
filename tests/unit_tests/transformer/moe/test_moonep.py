# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""MoonEP backend (megatron/core/transformer/moe/moonep.py) against a dense fp32 reference.

Needs 4 GPUs on one NVLink node and the moonep package:

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/transformer/moe/test_moonep.py

Checks the routed-expert output, the input / routing-probability gradients and the expert
weight gradients (local experts + the prefetched copies reduced to their home ranks), with
skewed routing so the planner duplicates hot experts.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

moonep = pytest.importorskip("moonep")

from megatron.core.transformer.moe.moonep import (  # noqa: E402
    MoEMoonEPTokenDispatcher,
    MoonEPGroupedMLP,
)
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402


def _init():
    if not dist.is_initialized():
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    return dist.get_rank(), dist.get_world_size()


def _config(H, Fh, E, K, R):
    return TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=4,
        ffn_hidden_size=4 * H,
        moe_ffn_hidden_size=Fh,
        num_moe_experts=E,
        moe_router_topk=K,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="moonep",
        expert_model_parallel_size=R,
        moe_grouped_gemm=True,
    )


def _reference(x, probs, routing_map, w1, w2):
    """out[s] = sum over routed e of p[s, e] * W2_e silu-glu(W1_e x_s), fp32."""
    out = torch.zeros_like(x)
    for e in range(w1.shape[0]):
        rows = routing_map[:, e].nonzero().squeeze(-1)
        if rows.numel() == 0:
            continue
        h = x[rows] @ w1[e].t()
        g, u = h.chunk(2, dim=-1)
        a = F.silu(g) * u * probs[rows, e : e + 1]
        out = out.index_add(0, rows, a @ w2[e].t())
    return out


@pytest.mark.parametrize("skew", [0.0, 4.0])
def test_moonep_matches_dense_reference(skew):
    rank, R = _init()
    if R != 4:
        pytest.skip("needs 4 ranks (torchrun --nproc-per-node 4)")
    S, H, Fh, E, K = 1024, 512, 256, 32, 4
    epn = E // R
    group = dist.new_group(list(range(R)))
    singles = [dist.new_group([r]) for r in range(R)]
    cfg = _config(H, Fh, E, K, R)
    pgc = SimpleNamespace(ep=group, expt_tp=singles[rank], tp_ep=group, tp=singles[rank],
                          expt_dp=singles[rank])

    torch.manual_seed(0)
    w1 = (torch.randn(E, 2 * Fh, H, device="cuda") / H**0.5)
    w2 = (torch.randn(E, H, Fh, device="cuda") / Fh**0.5)
    torch.manual_seed(100 + rank)
    x = torch.randn(S, H, device="cuda")
    logits = torch.randn(S, E, device="cuda")
    logits[:, :3] += skew  # hot experts 0-2 (rank 0) -> redundant copies elsewhere
    top = logits.topk(K, dim=-1)
    routing_map = torch.zeros(S, E, dtype=torch.bool, device="cuda").scatter_(1, top.indices, True)
    probs = torch.zeros(S, E, device="cuda").scatter_(1, top.indices, torch.sigmoid(top.values))

    # MoonEP path
    experts = MoonEPGroupedMLP(epn, cfg)
    for i in range(epn):
        getattr(experts.linear_fc1, f"weight{i}").data.copy_(w1[rank * epn + i])
        getattr(experts.linear_fc2, f"weight{i}").data.copy_(w2[rank * epn + i])
    disp = MoEMoonEPTokenDispatcher(epn, list(range(rank * epn, (rank + 1) * epn)), cfg, pgc)
    disp.attach_experts(experts)
    xb = x.to(torch.bfloat16).requires_grad_(True)
    pb = probs.clone().requires_grad_(True)
    h, w_sk = disp.dispatch_preprocess(xb, routing_map, pb)
    h, w_nvs = disp.token_dispatch(h, w_sk)
    h, cu, w_nvs = disp.dispatch_postprocess(h, w_nvs)
    y, _ = experts(h, cu, w_nvs)
    out = disp.combine_postprocess(disp.token_combine(disp.combine_preprocess(y)))
    torch.manual_seed(7 + rank)
    gout = torch.randn_like(x)
    out.float().backward(gout.to(torch.bfloat16).float())
    torch.cuda.synchronize()

    # reference (fp32, all experts local)
    xr = x.clone().requires_grad_(True)
    pr = probs.clone().requires_grad_(True)
    w1r = w1.clone().requires_grad_(True)
    w2r = w2.clone().requires_grad_(True)
    ref = _reference(xr, pr, routing_map, w1r, w2r)
    ref.backward(gout.to(torch.bfloat16).float())
    gw1 = w1r.grad.clone()
    gw2 = w2r.grad.clone()
    dist.all_reduce(gw1, group=group)
    dist.all_reduce(gw2, group=group)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    errs = {
        "out": rel(out, ref),
        "dx": rel(xb.grad, xr.grad),
        "dprobs": rel(pb.grad * routing_map, pr.grad * routing_map),
        "dw1": rel(torch.stack(experts.main_grads()[0]), gw1[rank * epn : (rank + 1) * epn]),
        "dw2": rel(torch.stack(experts.main_grads()[1]), gw2[rank * epn : (rank + 1) * epn]),
    }
    remote = int((disp.holder.plan.experts_to_copy >= 0).sum().item())
    print(f"rank {rank} skew {skew}: prefetched copies {remote}, rel errors {errs}", flush=True)
    for k, v in errs.items():
        assert v < 2e-2, (k, v)
    if skew > 0:
        assert remote > 0, "skewed routing should create redundant experts"


@pytest.mark.parametrize("chunks,skew", [(2, 0.0), (2, 4.0), (4, 4.0)])
def test_moonep_chunked_matches_dense_reference(chunks, skew, monkeypatch):
    """MOONEP_CHUNKS > 1: dispatch / experts / combine pipelined over token chunks on two streams
    (chunked_compute), against the same dense fp32 reference -- forward, input / probability
    gradients and the expert weight gradients (both chunks accumulate into the main grads)."""
    rank, R = _init()
    if R != 4:
        pytest.skip("needs 4 ranks (torchrun --nproc-per-node 4)")
    monkeypatch.setenv("MOONEP_CHUNKS", str(chunks))
    S, H, Fh, E, K = 1024, 512, 256, 32, 4
    epn = E // R
    group = dist.new_group(list(range(R)))
    singles = [dist.new_group([r]) for r in range(R)]
    cfg = _config(H, Fh, E, K, R)
    pgc = SimpleNamespace(ep=group, expt_tp=singles[rank], tp_ep=group, tp=singles[rank],
                          expt_dp=singles[rank])

    torch.manual_seed(0)
    w1 = (torch.randn(E, 2 * Fh, H, device="cuda") / H**0.5)
    w2 = (torch.randn(E, H, Fh, device="cuda") / Fh**0.5)
    torch.manual_seed(100 + rank)
    x = torch.randn(S, H, device="cuda")
    logits = torch.randn(S, E, device="cuda")
    logits[:, :3] += skew
    top = logits.topk(K, dim=-1)
    routing_map = torch.zeros(S, E, dtype=torch.bool, device="cuda").scatter_(1, top.indices, True)
    probs = torch.zeros(S, E, device="cuda").scatter_(1, top.indices, torch.sigmoid(top.values))

    experts = MoonEPGroupedMLP(epn, cfg)
    for i in range(epn):
        getattr(experts.linear_fc1, f"weight{i}").data.copy_(w1[rank * epn + i])
        getattr(experts.linear_fc2, f"weight{i}").data.copy_(w2[rank * epn + i])
    disp = MoEMoonEPTokenDispatcher(epn, list(range(rank * epn, (rank + 1) * epn)), cfg, pgc)
    disp.attach_experts(experts)
    assert disp.num_chunks == chunks
    xb = x.to(torch.bfloat16).requires_grad_(True)
    pb = probs.clone().requires_grad_(True)
    h, w_sk = disp.dispatch_preprocess(xb, routing_map, pb)
    out = disp.combine_postprocess(disp.chunked_compute(h, w_sk))
    torch.manual_seed(7 + rank)
    gout = torch.randn_like(x)
    out.float().backward(gout.to(torch.bfloat16).float())
    torch.cuda.synchronize()

    xr = x.clone().requires_grad_(True)
    pr = probs.clone().requires_grad_(True)
    w1r = w1.clone().requires_grad_(True)
    w2r = w2.clone().requires_grad_(True)
    ref = _reference(xr, pr, routing_map, w1r, w2r)
    ref.backward(gout.to(torch.bfloat16).float())
    gw1 = w1r.grad.clone()
    gw2 = w2r.grad.clone()
    dist.all_reduce(gw1, group=group)
    dist.all_reduce(gw2, group=group)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    errs = {
        "out": rel(out, ref),
        "dx": rel(xb.grad, xr.grad),
        "dprobs": rel(pb.grad * routing_map, pr.grad * routing_map),
        "dw1": rel(torch.stack(experts.main_grads()[0]), gw1[rank * epn : (rank + 1) * epn]),
        "dw2": rel(torch.stack(experts.main_grads()[1]), gw2[rank * epn : (rank + 1) * epn]),
    }
    print(f"rank {rank} chunks {chunks} skew {skew}: rel errors {errs}", flush=True)
    for k, v in errs.items():
        assert v < 2e-2, (k, v)


def test_moonep_two_steps_and_param_layout(monkeypatch):
    """Per-expert 2-D parameters (TEGroupedMLP names, what angular MuOwn needs) and the weight
    mirror: after an optimizer step changes the weights, the next forward must use them (the
    prefetched copies on the other ranks included). Checked against the dense reference after
    an SGD step on every expert."""
    rank, R = _init()
    if R != 4:
        pytest.skip("needs 4 ranks (torchrun --nproc-per-node 4)")
    S, H, Fh, E, K = 1024, 512, 256, 32, 4
    epn = E // R
    group = dist.new_group(list(range(R)))
    singles = [dist.new_group([r]) for r in range(R)]
    cfg = _config(H, Fh, E, K, R)
    pgc = SimpleNamespace(ep=group, expt_tp=singles[rank], tp_ep=group, tp=singles[rank],
                          expt_dp=singles[rank])
    experts = MoonEPGroupedMLP(epn, cfg, pgc)
    names = sorted(n for n, _ in experts.named_parameters())
    assert names == sorted([f"linear_fc1.weight{i}" for i in range(epn)]
                           + [f"linear_fc2.weight{i}" for i in range(epn)])
    assert all(p.ndim == 2 for p in experts.parameters())
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * Fh, H, device="cuda") / H**0.5
    w2 = torch.randn(E, H, Fh, device="cuda") / Fh**0.5
    for i in range(epn):
        getattr(experts.linear_fc1, f"weight{i}").data.copy_(w1[rank * epn + i])
        getattr(experts.linear_fc2, f"weight{i}").data.copy_(w2[rank * epn + i])
    disp = MoEMoonEPTokenDispatcher(epn, list(range(rank * epn, (rank + 1) * epn)), cfg, pgc)
    disp.attach_experts(experts)
    torch.manual_seed(100 + rank)
    x = torch.randn(S, H, device="cuda")
    logits = torch.randn(S, E, device="cuda")
    logits[:, :3] += 4.0
    top = logits.topk(K, dim=-1)
    routing_map = torch.zeros(S, E, dtype=torch.bool, device="cuda").scatter_(1, top.indices, True)
    probs = torch.zeros(S, E, device="cuda").scatter_(1, top.indices, torch.sigmoid(top.values))

    def run():
        h, w_sk = disp.dispatch_preprocess(x.to(torch.bfloat16), routing_map, probs)
        h, w_nvs = disp.token_dispatch(h, w_sk)
        h, cu, w_nvs = disp.dispatch_postprocess(h, w_nvs)
        y, _ = experts(h, cu, w_nvs)
        return disp.combine_postprocess(disp.token_combine(disp.combine_preprocess(y)))

    with torch.no_grad():
        run()
    # an optimizer step on every expert weight (the global post-step hook bumps the step count)
    opt = torch.optim.SGD(list(experts.parameters()), lr=1.0)
    for i in range(epn):
        getattr(experts.linear_fc1, f"weight{i}").grad = torch.full_like(
            getattr(experts.linear_fc1, f"weight{i}"), 0.01)
        getattr(experts.linear_fc2, f"weight{i}").grad = torch.full_like(
            getattr(experts.linear_fc2, f"weight{i}"), -0.01)
    opt.step()
    with torch.no_grad():
        out = run()
        ref = _reference(x, probs, routing_map, w1 - 0.01, w2 + 0.01)
    err = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"rank {rank}: after one step rel error {err}", flush=True)
    assert err < 2e-2, err
