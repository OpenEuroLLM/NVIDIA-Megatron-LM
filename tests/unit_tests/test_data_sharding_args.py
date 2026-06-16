# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import io

import pytest

from megatron.training import arguments


def _parse_and_validate_data_sharding_args(monkeypatch, extra_args, *, cyclic=True):
    cli_args = [
        'test_data_sharding_args.py',
        '--num-layers',
        '1',
        '--hidden-size',
        '16',
        '--num-attention-heads',
        '2',
        '--max-position-embeddings',
        '16',
        '--seq-length',
        '16',
        '--micro-batch-size',
        '2',
        '--global-batch-size',
        '64',
        '--train-iters',
        '1',
        '--lr',
        '1e-4',
    ]
    if cyclic:
        cli_args.extend(['--dataloader-type', 'cyclic'])
    cli_args.extend(extra_args)

    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setattr(arguments, '_print_args', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(arguments, 'print_rank_0', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(arguments, 'warn_rank_0', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('sys.argv', cli_args)

    args = arguments.parse_args()
    return arguments.validate_args(args)


def test_cli_defaults_to_data_parallel_data_sharding(monkeypatch):
    args = _parse_and_validate_data_sharding_args(monkeypatch, [])

    assert args.data_sharding
    assert args.data_sharding_strategy == 'data_parallel'
    assert args.data_sharding_virtual_shards is None


def test_cli_accepts_virtual_data_sharding(monkeypatch):
    args = _parse_and_validate_data_sharding_args(
        monkeypatch,
        ['--data-sharding-strategy', 'virtual'],
    )

    assert args.data_sharding_strategy == 'virtual'
    assert args.data_sharding_virtual_shards is None


def test_cli_accepts_explicit_virtual_shards(monkeypatch):
    args = _parse_and_validate_data_sharding_args(
        monkeypatch,
        ['--data-sharding-strategy', 'virtual', '--data-sharding-virtual-shards', '16'],
    )

    assert args.data_sharding_strategy == 'virtual'
    assert args.data_sharding_virtual_shards == 16


def test_cli_rejects_virtual_shards_without_virtual_strategy(monkeypatch):
    with pytest.raises(AssertionError, match='requires --data-sharding-strategy virtual'):
        _parse_and_validate_data_sharding_args(
            monkeypatch,
            ['--data-sharding-virtual-shards', '16'],
        )


def test_cli_rejects_virtual_strategy_without_data_sharding(monkeypatch):
    with pytest.raises(AssertionError, match='requires data sharding'):
        _parse_and_validate_data_sharding_args(
            monkeypatch,
            ['--data-sharding-strategy', 'virtual', '--no-data-sharding'],
        )


def test_cli_rejects_virtual_strategy_without_cyclic_dataloader(monkeypatch):
    with pytest.raises(AssertionError, match='only applies to the cyclic dataloader'):
        _parse_and_validate_data_sharding_args(
            monkeypatch,
            ['--data-sharding-strategy', 'virtual'],
            cyclic=False,
        )


def test_cli_rejects_invalid_data_sharding_strategy(monkeypatch):
    monkeypatch.setattr(
        'sys.argv',
        [
            'test_data_sharding_args.py',
            '--data-sharding-strategy',
            'not-a-strategy',
        ],
    )

    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):
            arguments.parse_args()
