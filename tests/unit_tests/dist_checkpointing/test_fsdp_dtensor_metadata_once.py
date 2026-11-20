# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""fsdp_dtensor load: a checkpoint's ``.metadata`` (327 MB for the 84B v4b checkpoint) is read once
per process across the args-only load, the rerun-state check and the main load, not three times."""

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from megatron.training import checkpointing
from megatron.training.checkpointing import _drop_metadata_cache, _MetadataOnceReader


def _save(path, scale):
    state = {'args': 'the args', 'iteration': 42081, 'w': torch.arange(6.0) * scale}
    dcp.save(state, checkpoint_id=str(path), no_dist=True)


def _count_reads(monkeypatch):
    reads = []
    real = FileSystemReader.read_metadata

    def counting(self, *args, **kwargs):
        reads.append(str(self.path))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(FileSystemReader, 'read_metadata', counting)
    return reads


def test_metadata_is_read_once_across_readers(tmp_path, monkeypatch):
    _save(tmp_path / 'iter_0042081', 1.0)
    reads = _count_reads(monkeypatch)
    _drop_metadata_cache()
    first, second = _MetadataOnceReader(tmp_path / 'iter_0042081'), _MetadataOnceReader(
        str(tmp_path / 'iter_0042081') + '/'
    )
    m1 = first.read_metadata()
    first_load_id = m1.storage_meta.load_id
    m2 = second.read_metadata()
    assert len(reads) == 1 and m1 is m2
    # each reader's own load id, as FileSystemReader.read_metadata sets it
    assert m2.storage_meta.load_id == second.load_id and second.load_id != first_load_id


def test_loads_through_the_cache_return_the_checkpoint(tmp_path, monkeypatch):
    _save(tmp_path / 'iter_0042081', 2.0)
    reads = _count_reads(monkeypatch)
    _drop_metadata_cache()
    head = {'args': None, 'iteration': None}
    dcp.load(head, storage_reader=_MetadataOnceReader(tmp_path / 'iter_0042081'), no_dist=True)
    full = {'args': None, 'iteration': None, 'w': torch.zeros(6)}
    dcp.load(full, storage_reader=_MetadataOnceReader(tmp_path / 'iter_0042081'), no_dist=True)
    assert len(reads) == 1
    assert head == {'args': 'the args', 'iteration': 42081}
    assert torch.equal(full['w'], torch.arange(6.0) * 2.0)


def test_another_checkpoint_or_a_dropped_cache_reads_again(tmp_path, monkeypatch):
    _save(tmp_path / 'iter_0042000', 1.0)
    _save(tmp_path / 'iter_0042081', 3.0)
    reads = _count_reads(monkeypatch)
    _drop_metadata_cache()
    _MetadataOnceReader(tmp_path / 'iter_0042000').read_metadata()
    _MetadataOnceReader(tmp_path / 'iter_0042081').read_metadata()
    assert len(reads) == 2 and len(checkpointing._METADATA_CACHE) == 1  # one checkpoint held
    _drop_metadata_cache()
    _MetadataOnceReader(tmp_path / 'iter_0042081').read_metadata()
    assert len(reads) == 3
