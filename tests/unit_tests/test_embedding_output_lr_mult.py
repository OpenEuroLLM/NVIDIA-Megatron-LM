# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""embedding_lr_mult / output_layer_lr_mult: the word embedding and the untied output layer get
their own LR groups, combined with embedding_wd_mult."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, _get_param_groups, get_standard_config_overrides


class _Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(32, 8)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Emb()
        self.fc = torch.nn.Linear(8, 8, bias=False)
        self.output_layer = torch.nn.Linear(8, 32, bias=False)


@patch('torch.distributed.get_world_size', return_value=1)
@patch('torch.distributed.all_gather_object', lambda out, obj: out.__setitem__(0, obj))
@pytest.mark.parametrize("emb,out,wd", [(1.0, 1.0, 1.0), (5.0, 0.2, 0.0), (1.0, 0.2, 1.0)])
def test_lr_groups(_ws, emb, out, wd):
    net = _Net()
    config = OptimizerConfig(optimizer='adam', lr=2e-3, min_lr=2e-4, weight_decay=0.1,
                             embedding_lr_mult=emb, output_layer_lr_mult=out, embedding_wd_mult=wd)
    groups = _get_param_groups([net], config, get_standard_config_overrides(config=config))
    want = {id(net.embedding.word_embeddings.weight): (emb, wd), id(net.output_layer.weight): (out, 1.0),
            id(net.fc.weight): (1.0, 1.0)}
    seen = set()
    for g in groups:
        for p in g['params']:
            m, w = want[id(p)]
            seen.add(id(p))
            assert (g['max_lr'] or 2e-3) == pytest.approx(2e-3 * m)
            assert (g['min_lr'] if g['max_lr'] is not None else 2e-4) == pytest.approx(2e-4 * m)
            assert g['wd_mult'] == pytest.approx(w)
    assert seen == set(want)


def test_conflicts_with_decoupled_lr():
    config = OptimizerConfig(optimizer='adam', lr=2e-3, decoupled_lr=1e-3, output_layer_lr_mult=0.2)
    with pytest.raises(ValueError):
        get_standard_config_overrides(config=config)
