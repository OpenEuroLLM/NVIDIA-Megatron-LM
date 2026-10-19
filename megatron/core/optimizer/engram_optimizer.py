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

Memory per table parameter: the fp32 table (4 B), its fp32 gradient buffer (4 B) and the
Adam moments -- 8 B with fp32 moments, 2 B + ~0 with ``engram_table_momentum_dtype="bf16"``
and ``engram_table_rowwise_v`` (one second moment per row, as row-wise Adagrad for
recommender embeddings): 16 -> 10 B. The update runs over row chunks, so its temporaries stay
a few MB instead of table-sized (torch's foreach Adam allocates several table-sized ones).
"""

import logging
import math
from dataclasses import replace
from typing import List, Optional

import torch

from ..dist_checkpointing.mapping import ShardedTensor
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


_CHUNK_ELEMENTS = 1 << 24  # elements per update chunk (64 MB of fp32 temporaries)


class TableAdam(torch.optim.Optimizer):
    """Adam for the sharded tables, updated in row chunks.

    ``rowwise_v``: the second moment is one value per row (the mean square of the row's
    gradient), ``[rows, 1]``. ``momentum_dtype``: storage dtype of the first moment (the
    arithmetic is fp32). With the defaults this is torch.optim.Adam (no weight decay).
    """

    def __init__(self, params, lr, betas, eps, rowwise_v=False, momentum_dtype=torch.float32):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.rowwise_v = rowwise_v
        self.momentum_dtype = momentum_dtype

    def init_state(self, p):
        state = self.state[p]
        if not state:
            state['step'] = torch.zeros((), dtype=torch.float32)
            state['exp_avg'] = torch.zeros_like(p, dtype=self.momentum_dtype)
            v_shape = (*p.shape[:-1], 1) if self.rowwise_v else p.shape
            state['exp_avg_sq'] = torch.zeros(v_shape, dtype=torch.float32, device=p.device)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.init_state(p)
                state['step'] += 1
                t = float(state['step'])
                step_size = group['lr'] / (1 - beta1**t)
                bc2_sqrt = math.sqrt(1 - beta2**t)
                rows = max(1, _CHUNK_ELEMENTS // max(1, p.shape[-1]))
                for lo in range(0, p.shape[0], rows):
                    w, g = p.data[lo : lo + rows], p.grad[lo : lo + rows].float()
                    m, v = state['exp_avg'][lo : lo + rows], state['exp_avg_sq'][lo : lo + rows]
                    m32 = m.float().mul_(beta1).add_(g, alpha=1 - beta1)
                    sq = g.square_()  # g is a chunk copy (float()) or the grad: reused below
                    if self.rowwise_v:
                        sq = sq.mean(-1, keepdim=True)
                    v.mul_(beta2).add_(sq, alpha=1 - beta2)
                    w.addcdiv_(m32, v.sqrt().div_(bc2_sqrt).add_(group['eps']), value=-step_size)
                    m.copy_(m32)
        return None


class EngramTableOptimizer(MegatronOptimizer):
    """Adam on this rank's Engram table shards (see the module docstring)."""

    def __init__(self, params: List[torch.nn.Parameter], config: OptimizerConfig):
        lr = config.lr if config.lr is not None else 0.0
        # One group per LR multiplier: Engram tables (engram_lr_mult) and product-key
        # memory values (pkm_lr_mult, set on the parameter) may share this optimizer.
        by_mult = {}
        for p in params:
            kind = getattr(p, "sharded_table_kind", "engram")
            mult = config.pkm_lr_mult if kind == "pkm" else config.engram_lr_mult
            by_mult.setdefault(float(mult), []).append(p)
        groups = [
            dict(
                params=group_params,
                lr=lr * mult,
                max_lr=lr * mult,
                min_lr=(config.min_lr or 0.0) * mult,
                wd_mult=0.0,
                weight_decay=0.0,
                is_engram_table=True,
                is_expert_parallel=False,
                sharded_table_lr_mult=mult,
            )
            for mult, group_params in sorted(by_mult.items())
        ] or [dict(params=[], lr=0.0, max_lr=0.0, min_lr=0.0, wd_mult=0.0, weight_decay=0.0)]
        momentum_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[
            config.engram_table_momentum_dtype
        ]
        optimizer = TableAdam(
            groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            rowwise_v=config.engram_table_rowwise_v,
            momentum_dtype=momentum_dtype,
        )
        super().__init__(optimizer, config, init_state_fn=lambda opt, cfg=None: None)
        self.params = [p for g in optimizer.param_groups for p in g["params"]]
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
                self.optimizer.init_state(p)
        param_state = {}
        steps = []
        for i, p in enumerate(self.params):
            sharded = by_data[id(p)]
            state = self.optimizer.state.get(p, {})
            entry = {}
            for name in ('exp_avg', 'exp_avg_sq'):
                if name in state:
                    entry[name] = _moment_sharded_tensor(
                        sharded, state[name], prefix=f'optimizer.state.{name}'
                    )
            if 'step' in state:
                steps.append(float(state['step']))
            param_state[f"{i}:{sharded.key}"] = entry
        groups = {
            str(i): {k: v for k, v in g.items() if k != 'params'}
            for i, g in enumerate(self.optimizer.param_groups)
        }
        out = {'param_keyed': True, 'param_groups': groups, 'param_state': param_state}
        if steps:
            out['common_step'] = max(steps)
        self._param_keys = list(param_state)
        return out

    def _load_param_keyed(self, state_dict):
        saved = {g.get('sharded_table_lr_mult'): g for g in state_dict['param_groups'].values()}
        for live in self.optimizer.param_groups:
            match = saved.get(live.get('sharded_table_lr_mult'))
            for k, v in (match or {}).items():
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


def _moment_sharded_tensor(table: ShardedTensor, moment: torch.Tensor, prefix: str):
    """A moment of a table shard: the table's ShardedTensor, or for a row-wise moment
    ``[rows, 1]`` the same rows with the last axis collapsed (the tables are sharded on rows
    only)."""
    if moment.shape == table.local_shape or moment.shape[-1] != 1:
        return make_sharded_optimizer_tensor(table, moment, prefix=prefix)
    sh = replace(
        table,
        key=f'{prefix}.{table.key}',
        data=moment,
        dtype=moment.dtype,
        local_shape=(*table.local_shape[:-1], 1),
        global_shape=(*table.global_shape[:-1], 1),
        global_offset=(*table.global_offset[:-1], 0),
    )
    sh.validate_metadata_integrity()
    return sh


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
