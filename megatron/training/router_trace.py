# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Record the MoE routers' expert choices during evaluation (ROUTER_TRACE_OUT=<file>.pt, or 1
for <save>/router_trace.pt).

After the checkpoint is loaded, every TopKRouter gets a forward hook that keeps the chosen expert
ids of each token, ``[tokens, topk]`` int16 in token order, per router call; data-parallel rank 0
(tensor / pipeline rank 0 of its stage) saves ``{layer_number: [calls...]}`` plus the expert count
and top-k at exit. For offline studies of the routing (expert caching / offloading at inference).
Use with --skip-train and without CUDA graphs on the router (a replayed graph runs no hooks).
"""

import atexit
import os

import torch


def maybe_trace_routers(model):
    """Hook the routers if ROUTER_TRACE_OUT is set; returns the save function (also run at exit)."""
    out = os.environ.get("ROUTER_TRACE_OUT", "")
    if not out:
        return None
    if out == "1":  # default: the run's save directory (one env for every task of an array)
        from megatron.training import get_args

        out = os.path.join(get_args().save, "router_trace.pt")
    from megatron.core import parallel_state as ps
    from megatron.core.transformer.moe.router import TopKRouter

    if ps.get_data_parallel_rank() != 0 or ps.get_tensor_model_parallel_rank() != 0:
        return None
    stage = ps.get_pipeline_model_parallel_rank()
    trace, meta = {}, {}

    def hook(module, _inp, output):
        _, routing_map = output
        k = module.topk
        ids = routing_map.nonzero()[:, 1].view(routing_map.shape[0], k)  # rows in token order
        trace.setdefault(module.layer_number, []).append(ids.to(torch.int16).cpu())
        meta.setdefault("num_experts", routing_map.shape[1])
        meta.setdefault("topk", k)

    chunks = model if isinstance(model, list) else [model]
    n = 0
    for chunk in chunks:
        for m in chunk.modules():
            if isinstance(m, TopKRouter):
                m.register_forward_hook(hook)
                n += 1

    def save():
        path = out if stage == 0 else f"{out}.stage{stage}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"trace": trace, **meta}, path)
        calls = sum(len(v) for v in trace.values())
        print(f"[router_trace] {n} routers, {calls} calls -> {path}", flush=True)

    atexit.register(save)
    print(f"[router_trace] hooked {n} routers (stage {stage})", flush=True)
    return save
