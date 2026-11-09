# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""fsdp_dtensor checkpoints: the optimizer's parameter-group metadata and per-parameter scalars
(``step``) go into two pickled DCP items per (sub-)optimizer instead of one item per leaf, and
load back into the unpacked state dict; checkpoints written before the packing still load."""

import copy

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, default_planner
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from megatron.training.checkpointing import (
    _restore_loaded_scalars,
    fsdp_dtensor_checkpoint_is_packed,
    pack_fsdp_dtensor_optimizer_scalars,
)

EXPERTS = 40


def _meta(lr):
    return {'lr': lr, 'min_lr': 0.0, 'betas': (0.9, 0.95), 'wd_mult': 0.0, 'is_expert_parallel': True}


def _sub(step, lr, val, names):
    """``step`` None: entries without a step counter (an optimizer before its first update)."""
    def entry():
        e = {'m_u': torch.full((3,), val), 'v_norm': torch.full((1,), val)}
        return e if step is None else {'step': step, **e}
    return {
        'state': {n: entry() for n in names},
        'param_to_group_meta': {n: _meta(lr if i % 2 else 2 * lr) for i, n in enumerate(names)},
    }


def _optim(step, lr, val, chained, engram_step=True):
    experts = [f'layers.1.mlp.experts.linear_fc1.weight{e}' for e in range(EXPERTS)]
    if not chained:
        return _sub(step, lr, val, experts)
    engram = _sub(2 * step if engram_step else None, 5 * lr, val, ['engram.table'])
    engram['table_step_group'] = 2 * step  # another non-tensor leaf of a sub-optimizer
    return {0: _sub(step, lr, val, experts), 1: engram}


def _save(path, chained, packed):
    sd = {'model': {'w': torch.full((2,), 7.0)}, 'optimizer': _optim(4, 0.002, 2.0, chained),
          'iteration': 20, 'args': ('cfg', 3), 'opt_param_scheduler': {'num_steps': 2560}}
    if packed:
        sd, _ = pack_fsdp_dtensor_optimizer_scalars(sd)
    dcp.save(sd, checkpoint_id=str(path))
    return FileSystemReader(str(path)).read_metadata().state_dict_metadata


def _load(path, chained, engram_step=True):
    """As megatron.training.checkpointing's fsdp_dtensor load: a raw template, a shallow copy for
    DCP, packed when the checkpoint is, then the scalars copied back into the raw dict.
    ``engram_step=False``: the Engram template has no step counters (packed checkpoints restore
    every saved scalar; the unpacked path loads only the template's keys)."""
    raw = {'model': {'w': torch.zeros(2)},
           'optimizer': _optim(1, 0.0, 0.0, chained, engram_step=engram_step),
           'iteration': 0, 'args': None, 'opt_param_scheduler': {'num_steps': 0}}
    raw_optim = raw['optimizer'].copy()
    sd = dict(raw)
    metadata = FileSystemReader(str(path)).read_metadata().state_dict_metadata
    unpack = None
    if fsdp_dtensor_checkpoint_is_packed(metadata):
        sd, unpack = pack_fsdp_dtensor_optimizer_scalars(sd, checkpoint_keys=metadata)
    dcp.load(sd, checkpoint_id=str(path),
             planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
    if unpack is not None:
        sd = unpack()
    _restore_loaded_scalars(raw_optim, sd['optimizer'])
    return sd, raw_optim


def _check(loaded, optim, chained):
    # the top-level non-tensor values (DCP writes them into the dict it is given)
    assert loaded['iteration'] == 20 and loaded['args'] == ('cfg', 3)
    assert loaded['opt_param_scheduler'] == {'num_steps': 2560}
    torch.testing.assert_close(loaded['model']['w'], torch.full((2,), 7.0))
    if chained:
        assert optim[1]['table_step_group'] == 8
    subs = [optim[0], optim[1]] if chained else [optim]
    steps = [4, 8] if chained else [4]
    lrs = [0.002, 0.01] if chained else [0.002]
    for sub, step, lr in zip(subs, steps, lrs):
        for i, (name, entry) in enumerate(sub['state'].items()):
            assert entry['step'] == step
            torch.testing.assert_close(entry['m_u'], torch.full((3,), 2.0))
            assert sub['param_to_group_meta'][name] == _meta(lr if i % 2 else 2 * lr)


@pytest.mark.parametrize('chained', [False, True])
def test_packed_round_trip(tmp_path, chained):
    metadata = _save(tmp_path, chained, packed=True)
    assert fsdp_dtensor_checkpoint_is_packed(metadata)
    n_bytes = sum(isinstance(v, BytesStorageMetadata) for k, v in metadata.items()
                  if k.startswith('optimizer.'))
    assert n_bytes == (5 if chained else 2)  # vs 6 leaves per expert unpacked
    _check(*_load(tmp_path, chained, engram_step=False), chained)


@pytest.mark.parametrize('chained', [False, True])
def test_unpacked_checkpoint_still_loads(tmp_path, chained):
    metadata = _save(tmp_path, chained, packed=False)
    assert not fsdp_dtensor_checkpoint_is_packed(metadata)
    assert sum(isinstance(v, BytesStorageMetadata) for v in metadata.values()) >= 6 * EXPERTS
    _check(*_load(tmp_path, chained), chained)


def test_pack_does_not_modify_its_input():
    sd = {'model': {}, 'optimizer': _optim(4, 0.002, 2.0, chained=True)}
    before = copy.deepcopy(sd)
    packed, _ = pack_fsdp_dtensor_optimizer_scalars(sd)
    assert 'param_to_group_meta' not in packed['optimizer'][0]
    assert all('step' not in e for e in packed['optimizer'][0]['state'].values())
    assert sd['optimizer'][0]['state'].keys() == before['optimizer'][0]['state'].keys()
    assert sd['optimizer'][0]['param_to_group_meta'] == before['optimizer'][0]['param_to_group_meta']
    assert all(e['step'] == 4 for e in sd['optimizer'][0]['state'].values())


def test_unpack_fills_only_the_template_names(tmp_path):
    """A template with fewer parameters (allow_partial_load) takes only its own entries."""
    metadata = _save(tmp_path, chained=False, packed=True)
    template = {'model': {'w': torch.zeros(2)},
                'optimizer': _sub(1, 0.0, 0.0, ['layers.1.mlp.experts.linear_fc1.weight3'])}
    sd, unpack = pack_fsdp_dtensor_optimizer_scalars(template, checkpoint_keys=metadata)
    dcp.load(sd, checkpoint_id=str(tmp_path),
             planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
    out = unpack()['optimizer']
    assert list(out['state']) == ['layers.1.mlp.experts.linear_fc1.weight3']
    assert out['state']['layers.1.mlp.experts.linear_fc1.weight3']['step'] == 4
    assert out['param_to_group_meta']['layers.1.mlp.experts.linear_fc1.weight3'] == _meta(0.002)


def test_new_parameters_join_their_groups_loaded_metadata(tmp_path):
    """Parameters the checkpoint does not hold (checkpoint surgery, e.g. a ckda_branch) take the loaded
    group metadata of a parameter that shares their group in the template: the groups stay consistent
    (distrib_optimizer._param2group_meta_to_param_groups) and keep the trained values."""
    metadata = _save(tmp_path, chained=False, packed=True)
    experts = [f'layers.1.mlp.experts.linear_fc1.weight{e}' for e in range(EXPERTS)]
    # template lr 1.0: the two groups differ (odd index lr, even 2 lr); new.a joins the even, new.b the odd
    template = {'model': {'w': torch.zeros(2)}, 'optimizer': _sub(1, 1.0, 0.0, experts + ['new.a', 'new.b'])}
    sd, unpack = pack_fsdp_dtensor_optimizer_scalars(template, checkpoint_keys=metadata)
    dcp.load(sd, checkpoint_id=str(tmp_path),
             planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
    out = unpack()['optimizer']
    meta = out['param_to_group_meta']
    assert meta['layers.1.mlp.experts.linear_fc1.weight1'] == _meta(0.002)
    assert meta['new.a'] == _meta(0.004) and meta['new.b'] == _meta(0.002)
    assert out['state']['new.a']['step'] == 1  # a new parameter's optimizer state starts fresh
    assert out['state']['layers.1.mlp.experts.linear_fc1.weight1']['step'] == 4


def _ep_sub(step, lr, experts):
    return _sub(step, lr, 0.0, [f'layers.1.mlp.experts.linear_fc1.weight{e}' for e in experts]
                + ['decoder.final_layernorm.weight'])


def test_expert_parallel_ranks_keep_their_own_items(tmp_path):
    """Expert-parallel ranks hold other experts: their packed items get other keys (DCP keeps one
    copy of a key saved by several ranks), data-parallel replicas share one; a load on either
    rank, or on a layout holding all experts, restores every parameter's scalars."""
    ep0 = pack_fsdp_dtensor_optimizer_scalars({'optimizer': _ep_sub(9, 0.002, range(0, 4))})[0]
    ep1 = pack_fsdp_dtensor_optimizer_scalars({'optimizer': _ep_sub(9, 0.002, range(4, 8))})[0]
    replica = pack_fsdp_dtensor_optimizer_scalars({'optimizer': _ep_sub(9, 0.002, range(0, 4))})[0]
    tags0 = set(ep0['optimizer']['state_scalars_packed'])
    tags1 = set(ep1['optimizer']['state_scalars_packed'])
    assert tags0 == set(replica['optimizer']['state_scalars_packed']) and tags0.isdisjoint(tags1)
    # what DCP writes after deduplication: every distinct key once
    merged = {'optimizer': {'state': {**ep0['optimizer']['state'], **ep1['optimizer']['state']}}}
    for kind in ('param_to_group_meta_packed', 'state_scalars_packed'):
        merged['optimizer'][kind] = {**ep0['optimizer'][kind], **ep1['optimizer'][kind]}
    dcp.save(merged, checkpoint_id=str(tmp_path))
    metadata = FileSystemReader(str(tmp_path)).read_metadata().state_dict_metadata
    assert fsdp_dtensor_checkpoint_is_packed(metadata)
    for experts in (range(0, 4), range(4, 8), range(0, 8)):
        template = {'optimizer': _ep_sub(1, 0.0, experts)}
        sd, unpack = pack_fsdp_dtensor_optimizer_scalars(template, checkpoint_keys=metadata)
        dcp.load(sd, checkpoint_id=str(tmp_path),
                 planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
        out = unpack()['optimizer']
        assert all(e['step'] == 9 for e in out['state'].values()), list(experts)
        assert all(m == _meta(0.002) or m == _meta(0.004)
                   for m in out['param_to_group_meta'].values())
