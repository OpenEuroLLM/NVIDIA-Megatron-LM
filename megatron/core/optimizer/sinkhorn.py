# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Momentum update with Sinkhorn balancing for vocabulary-sized matrices.

DeepSeek-V4.1 (Algorithm 1) optimizes the token embedding, the prediction head and the Engram
tables this way instead of with Adam: Nesterov momentum, then K alternating row / column
l2-normalizations of the update (odd K: starting and ending with the rows), so every row (a token
or n-gram) and every column (a hidden feature) of the step has an RMS of about 1; rows whose update
norm is at most ``tau`` times the mean row norm (tokens absent from the batch) are masked to zero.
The step is ``lr * lr_ratio * sqrt(n) * U`` (n = hidden size; ``lr_ratio`` matches Adam's update
size, 0.18 in the report). Only a momentum buffer is kept, and no weight decay is applied.

Rows may be sharded across a tensor-parallel group (vocab-parallel embedding / head,
``partition_dim == 0``): the column sums and the mean row norm are then all-reduced over it.
"""

from typing import Callable, Optional

import torch
import torch.distributed as dist


class SinkhornMomentum(torch.optim.Optimizer):
    """Nesterov momentum + Sinkhorn row / column balancing (no second moment, no weight decay).

    Args:
        params: parameters (2-D, rows = vocabulary entries) or param groups.
        lr: base learning rate (the scheduler's; the step uses ``lr * lr_ratio``).
        momentum: momentum coefficient beta.
        nesterov: Nesterov momentum (DeepSeek-V4.1: on).
        iters: number of normalization steps K (odd).
        tau: rows with norm <= tau * mean row norm are masked.
        eps: added to the norms.
        lr_ratio: learning-rate correction gamma.
        row_group_fn: ``param -> ProcessGroup | None``, the group its rows are sharded over.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        iters: int = 11,
        tau: float = 1e-3,
        eps: float = 1e-20,
        lr_ratio: float = 0.18,
        row_group_fn: Optional[Callable[[torch.Tensor], Optional[dist.ProcessGroup]]] = None,
    ):
        if iters < 1 or iters % 2 == 0:
            raise ValueError(f"Sinkhorn iters must be odd and positive, got {iters}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, iters=iters, tau=tau,
                        eps=eps, lr_ratio=lr_ratio, weight_decay=0.0)
        super().__init__(params, defaults)
        self.row_group_fn = row_group_fn

    @staticmethod
    def balance(update: torch.Tensor, iters: int, tau: float, eps: float,
                group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """Sinkhorn-balanced direction of ``update`` [rows, n], scaled to unit row RMS."""
        u = update.float()
        rho = u.norm(dim=1)
        stats = torch.stack([rho.sum(), torch.tensor(float(rho.numel()), device=u.device)])
        if group is not None:
            dist.all_reduce(stats, group=group)
        mean = stats[0] / stats[1].clamp(min=1.0)
        u = u * (rho > tau * mean).unsqueeze(1)
        for k in range(iters):
            if k % 2 == 0:  # rows (k = 1, 3, ... in the report's 1-based numbering)
                u = u / (u.norm(dim=1, keepdim=True) + eps)
            else:  # columns
                col_sq = (u * u).sum(dim=0)
                if group is not None:
                    dist.all_reduce(col_sq, group=group)
                u = u / (col_sq.sqrt().unsqueeze(0) + eps)
        return u * (update.shape[1] ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError("SinkhornMomentum expects 2-D (vocabulary x hidden) parameters")
                g = p.grad.float()
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.float32)
                m = state['momentum_buffer']
                m.mul_(beta).add_(g, alpha=1.0 - beta)
                upd = m * beta + g * (1.0 - beta) if group['nesterov'] else m
                rows_group = self.row_group_fn(p) if self.row_group_fn is not None else None
                delta = self.balance(upd, group['iters'], group['tau'], group['eps'], rows_group)
                p.add_(delta.to(p.dtype), alpha=-group['lr'] * group['lr_ratio'])
        return loss
