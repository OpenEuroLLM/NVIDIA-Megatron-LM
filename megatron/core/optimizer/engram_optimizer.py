# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Optimizer for Engram tables sharded with ``engram_table_parallel="dp"``.

Such a table is not a DDP parameter: every rank owns a disjoint row range (fp32) and its
row gradients arrive sparse, through the lookup's backward all-to-all, into an fp32
buffer (``param.engram_grad``). This optimizer runs Adam on the local shard and is chained
into the main optimizer (:func:`attach_engram_table_optimizer`), so

* the LR scheduler drives its param group (``max_lr`` / ``min_lr`` = engram_lr_mult x the
  base ones, ``wd_mult`` 0, as the "tp" tables get through their param-group override);
* the global gradient norm and clipping include the tables -- each row has one owner,
  so the norm is a sum of squares over all ranks;
* it checkpoints in the param-keyed format: the moments are ShardedTensors of the table
  shard they belong to, so they reshard with it.
"""

import logging
import math
from typing import List, Optional

import torch

from ..dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from .optimizer import ChainedOptimizer, MegatronOptimizer
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


def engram_sharded_tables(model_chunks) -> List[torch.nn.Parameter]:
    """The ``engram_table_parallel="dp"`` table shards of this rank."""
    return [
        p
        for chunk in model_chunks
        for p in chunk.parameters()
        if getattr(p, "engram_sharded_table", False)
    ]


class EngramTableOptimizer(MegatronOptimizer):
    """Adam on this rank's Engram table shards (see the module docstring)."""

    def __init__(self, params: List[torch.nn.Parameter], config: OptimizerConfig):
        lr = config.lr if config.lr is not None else 0.0
        mult = config.engram_lr_mult
        group = dict(
            params=params,
            lr=lr * mult,
            max_lr=lr * mult,
            min_lr=(config.min_lr or 0.0) * mult,
            wd_mult=0.0,
            weight_decay=0.0,
            is_engram_table=True,
            is_expert_parallel=False,
        )
        optimizer = torch.optim.Adam(
            [group], betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps
        )
        super().__init__(optimizer, config, init_state_fn=lambda opt, cfg=None: None)
        self.params = params
        self.is_stub_optimizer = False

    # ---- gradients -------------------------------------------------------------
    def get_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.params)

    def prepare_grads(self) -> bool:
        for p in self.params:
            p.grad = p.engram_grad
        return False

    def get_grad_stats_parallel_group(self):
        # Rows are owned once across the whole job (different stages hold different
        # tables), so the norm is a sum over all ranks.
        return torch.distributed.group.WORLD

    def get_grads_for_grad_norm(self, grad_norm_group: Optional[str] = None):
        return [p.engram_grad for p in self.params] if grad_norm_group is None else []

    @torch.no_grad()
    def get_grad_norm(self):
        device = torch.cuda.current_device()
        sq = torch.zeros((), dtype=torch.float32, device=device)
        for p in self.params:
            sq += p.engram_grad.float().square().sum()
        torch.distributed.all_reduce(sq, group=self.get_grad_stats_parallel_group())
        return math.sqrt(sq.item())

    def count_zeros(self):
        return 0

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            p.engram_grad.zero_()
            p.grad = None

    def get_loss_scale(self):
        return torch.ones((), device=torch.cuda.current_device())

    def reload_model_params(self, state_dict=None):
        pass

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        self.optimizer.step()
        for p in self.params:
            p.grad = None
        return True

    @torch.no_grad()
    def step(self):
        self.prepare_grads()
        grad_norm = self.get_grad_norm()
        if self.config.clip_grad > 0.0 and grad_norm > self.config.clip_grad:
            for p in self.params:
                p.grad.mul_(self.config.clip_grad / (grad_norm + 1e-6))
        return self.step_with_ready_grads(), grad_norm, None

    # ---- checkpointing ---------------------------------------------------------
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if state_dict.get('param_keyed'):
            return self._load_param_keyed(state_dict)
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        from ..dist_checkpointing.dict_utils import nested_values

        by_data = {
            id(sh.data): sh for sh in nested_values(model_sharded_state_dict) if hasattr(sh, "data")
        }
        if is_loading:  # make sure the moments exist so the template has their shapes
            for p in self.params:
                state = self.optimizer.state[p]
                if not state:
                    state['step'] = torch.zeros((), dtype=torch.float32)
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
        param_state = {}
        steps = []
        for i, p in enumerate(self.params):
            sharded = by_data[id(p)]
            state = self.optimizer.state.get(p, {})
            entry = {}
            for name in ('exp_avg', 'exp_avg_sq'):
                if name in state:
                    entry[name] = make_sharded_optimizer_tensor(
                        sharded, state[name], prefix=f'optimizer.state.{name}'
                    )
            if 'step' in state:
                steps.append(float(state['step']))
            param_state[f"{i}:{sharded.key}"] = entry
        group = {k: v for k, v in self.optimizer.param_groups[0].items() if k != 'params'}
        out = {'param_keyed': True, 'param_groups': {'0': group}, 'param_state': param_state}
        if steps:
            out['common_step'] = max(steps)
        self._param_keys = list(param_state)
        return out

    def _load_param_keyed(self, state_dict):
        saved = next(iter(state_dict['param_groups'].values()))
        live = self.optimizer.param_groups[0]
        for k, v in saved.items():
            if k not in ('params', '_has_params'):
                live[k] = v
        step = state_dict.get('common_step', 0.0)
        for p, key in zip(self.params, self._param_keys):
            entry = state_dict['param_state'].get(key, {})
            state = self.optimizer.state[p]
            for name, value in entry.items():
                if name in state:
                    state[name].copy_(value)
                else:
                    state[name] = value.clone()
            state['step'] = torch.tensor(float(step), dtype=torch.float32)


def attach_engram_table_optimizer(optimizer, model_chunks, config: OptimizerConfig):
    """Chain an EngramTableOptimizer for the ``engram_table_parallel="dp"`` tables."""
    params = engram_sharded_tables(model_chunks)
    has_tables = torch.tensor([len(params)], device=torch.cuda.current_device())
    torch.distributed.all_reduce(has_tables)
    if int(has_tables.item()) == 0:
        return optimizer
    engram = EngramTableOptimizer(params, getattr(optimizer, 'config', config))
    if isinstance(optimizer, ChainedOptimizer) and type(optimizer) is ChainedOptimizer:
        optimizer.chained_optimizers.append(engram)
        return optimizer
    return ChainedOptimizer([optimizer, engram])
