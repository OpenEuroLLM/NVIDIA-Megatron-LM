# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""fsdp_dtensor load: optimizer step counters and param-group values survive DCP's load into the
preprocessed (copied, SwiGLU-split) state dict of a ChainedOptimizer with int keys."""

import copy

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import default_planner

from megatron.training.checkpointing import _restore_loaded_scalars


def _optim(step, lr, val, split):
    fc1 = (
        {'fc1_w': {'step': step, 'exp': torch.full((2,), val)},
         'fc1_v': {'step': step, 'exp': torch.full((2,), val)}}
        if split else {'fc1': {'step': step, 'exp': torch.full((4,), val)}}
    )
    return {
        0: {'state': {'a': {'step': step, 'exp': torch.full((3,), val)}, **fc1},
            'param_groups': [{'lr': lr, 'step': step, 'params': ['a', 'fc1']}]},
        1: {'state': {'t': {'step': 2 * step, 'm': torch.full((1,), val)}},
            'param_groups': [{'lr': 5 * lr, 'params': ['t']}]},
    }


def test_scalars_restored(tmp_path):
    dcp.save({'optimizer': _optim(4, 0.002, 2.0, split=True)}, checkpoint_id=str(tmp_path))
    raw = _optim(1, 0.0, 0.0, split=False)
    loaded = copy.deepcopy(raw)
    # the preprocessed copy: fc1 split into _w / _v, a's state shared (tensors load in place)
    loaded[0]['state'] = {**_optim(1, 0.0, 0.0, split=True)[0]['state'], 'a': raw[0]['state']['a']}
    dcp.load({'optimizer': loaded}, checkpoint_id=str(tmp_path),
             planner=default_planner.DefaultLoadPlanner(allow_partial_load=True))
    _restore_loaded_scalars(raw, loaded)
    assert raw[0]['state']['a']['step'] == 4 and raw[0]['state']['fc1']['step'] == 4
    assert raw[0]['param_groups'][0]['lr'] == 0.002 and raw[0]['param_groups'][0]['step'] == 4
    assert raw[1]['state']['t']['step'] == 8 and raw[1]['param_groups'][0]['lr'] == 0.01
    torch.testing.assert_close(raw[0]['state']['a']['exp'], torch.full((3,), 2.0))
