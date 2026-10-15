# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run a self-contained (non tensor-parallel) mixer inside a tensor-parallel model.

The flash-linear-attention layers (KDA, signed KDA, panini, Complex-KDA and its gated
attention) own plain ``nn.Linear`` projections, so there is nothing to shard. They can
still sit in a TP>1 model if every TP rank runs the whole layer on the whole input:

* without sequence parallelism the residual stream is identical on all TP ranks, so the
  replicas compute the same output and the same gradients;
* with sequence parallelism the layer all-gathers the sequence first and keeps only this
  rank's slice of its output. The gather's backward must *split* rather than
  reduce-scatter: every replica already holds the full input gradient, and summing them
  would count it TP times.

The replicas start identical (broadcast from TP rank 0) and stay identical: their
gradients are averaged across the TP group (``average_gradients_across_tp_domain``),
which also absorbs any nondeterminism in the kernels' backward passes. The cost is
redundant compute, TP times the layer's FLOPs; memory and checkpoints are as for TP=1.
"""

import torch

from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)


def replicate_across_tp(module: torch.nn.Module, tp_group) -> None:
    """Make ``module``'s parameters TP replicas: same start, averaged gradients."""
    for param in module.parameters():
        setattr(param, "average_gradients_across_tp_domain", True)
    if tp_group is None or tp_group.size() == 1:
        return
    src = torch.distributed.get_global_rank(tp_group, 0)
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            device_tensor = tensor.data.to(torch.cuda.current_device())
            torch.distributed.broadcast(device_tensor, src=src, group=tp_group)
            tensor.data.copy_(device_tensor)


def gather_replicated_input(hidden_states: torch.Tensor, config, tp_group) -> torch.Tensor:
    """[s/tp, b, h] under sequence parallelism -> [s, b, h]; unchanged otherwise."""
    if config.sequence_parallel and tp_group is not None and tp_group.size() > 1:
        return gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp_group
        )
    return hidden_states


def scatter_replicated_output(output: torch.Tensor, config, tp_group) -> torch.Tensor:
    """[s, b, h] -> this rank's [s/tp, b, h] under sequence parallelism; unchanged otherwise."""
    if config.sequence_parallel and tp_group is not None and tp_group.size() > 1:
        return scatter_to_sequence_parallel_region(output, group=tp_group)
    return output
