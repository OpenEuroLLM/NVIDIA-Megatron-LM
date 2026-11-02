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
    return {
        'state': {n: {'step': step, 'm_u': torch.full((3,), val), 'v_norm': torch.full((1,), val)}
                  for n in names},
        'param_to_group_meta': {n: _meta(lr if i % 2 else 2 * lr) for i, n in enumerate(names)},
    }


def _optim(step, lr, val, chained):
    experts = [f'layers.1.mlp.experts.linear_fc1.weight{e}' for e in range(EXPERTS)]
    if not chained:
        return _sub(step, lr, val, experts)
    return {0: _sub(step, lr, val, experts), 1: _sub(2 * step, 5 * lr, val, ['engram.table'])}


def _save(path, chained, packed):
    sd = {'model': {'w': torch.full((2,), 7.0)}, 'optimizer': _optim(4, 0.002, 2.0, chained)}
    if packed:
        sd, _ = pack_fsdp_dtensor_optimizer_scalars(sd)
    dcp.save(sd, checkpoint_id=str(path))
    return FileSystemReader(str(path)).read_metadata().state_dict_metadata


def _load(path, chained):
    """As megatron.training.checkpointing's fsdp_dtensor load: a raw template, a shallow copy for
    DCP, packed when the checkpoint is, then the scalars copied back into the raw dict."""
    raw = {'model': {'w': torch.zeros(2)}, 'optimizer': _optim(1, 0.0, 0.0, chained)}
    raw_optim = raw['optimizer'].copy()
    sd = dict(raw)
    metadata = FileSystemReader(str(path)).read_metadata().state_dict_metadata
    unpack = None
    if fsdp_dtensor_checkpoint_is_packed(metadata):
        sd, unpack = pack_fsdp_dtensor_optimizer_scalars(sd)
    dcp.load(sd, checkpoint_id=str(path),
             planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
    if unpack is not None:
        sd = unpack()
    _restore_loaded_scalars(raw_optim, sd['optimizer'])
    return raw['model'], raw_optim


def _check(model, optim, chained):
    torch.testing.assert_close(model['w'], torch.full((2,), 7.0))
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
    n_bytes = sum(isinstance(v, BytesStorageMetadata) for v in metadata.values())
    assert n_bytes == (4 if chained else 2)  # vs 6 leaves per expert unpacked
    _check(*_load(tmp_path, chained), chained)


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
    _save(tmp_path, chained=False, packed=True)
    template = {'model': {'w': torch.zeros(2)},
                'optimizer': _sub(1, 0.0, 0.0, ['layers.1.mlp.experts.linear_fc1.weight3'])}
    sd, unpack = pack_fsdp_dtensor_optimizer_scalars(template)
    dcp.load(sd, checkpoint_id=str(tmp_path),
             planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
    out = unpack()['optimizer']
    assert list(out['state']) == ['layers.1.mlp.experts.linear_fc1.weight3']
    assert out['state']['layers.1.mlp.experts.linear_fc1.weight3']['step'] == 4
    assert out['param_to_group_meta']['layers.1.mlp.experts.linear_fc1.weight3'] == _meta(0.002)
