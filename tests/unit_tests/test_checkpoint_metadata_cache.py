# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The per-process ``.metadata`` cache of the fsdp_dtensor load (checkpointing._MetadataOnceReader).

One load used to unpickle a checkpoint's ``.metadata`` three times per rank (args, rerun-state
check, state load). Now every _MetadataOnceReader of the same unchanged checkpoint shares one read;
a rewritten ``.metadata`` (new mtime / size) is read again; the args items load locally (no_dist).

    python -m pytest tests/unit_tests/test_checkpoint_metadata_cache.py   (or run the file directly)
"""

import os
import tempfile
import time
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from megatron.training import checkpointing as C


def _save(path, scale):
    sd = {
        "args": SimpleNamespace(kv_channels=256, scale=scale),
        "iteration": 7,
        "checkpoint_version": 3.0,
        "w": torch.arange(12, dtype=torch.float32).reshape(3, 4) * scale,
    }
    dcp.save(sd, storage_writer=FileSystemWriter(path), no_dist=True)


def test_metadata_read_once_and_invalidated():
    calls = []
    orig = FileSystemReader.read_metadata

    def counting(self, *a, **k):
        calls.append(os.fspath(self.path))
        return orig(self, *a, **k)

    FileSystemReader.read_metadata = counting
    try:
        with tempfile.TemporaryDirectory() as d:
            C._drop_metadata_cache()
            _save(d, 1.0)

            # args load as in _load_base_checkpoint(rank0=True): local, through the cache
            sd = {"args": None, "iteration": None, "checkpoint_version": None}
            dcp.load(sd, storage_reader=C._MetadataOnceReader(d), no_dist=True)
            assert sd["iteration"] == 7 and sd["args"].kv_channels == 256
            # rerun-state check + the state load: new readers, no new read
            md = C._MetadataOnceReader(d).read_metadata()
            assert "w" in md.state_dict_metadata
            w = {"w": torch.empty(3, 4)}
            dcp.load(w, storage_reader=C._MetadataOnceReader(d), no_dist=True)
            assert torch.equal(w["w"], torch.arange(12, dtype=torch.float32).reshape(3, 4))
            assert len(calls) == 1, calls

            # a rewritten checkpoint at the same path is read again (mtime / size key)
            time.sleep(0.01)
            _save(d, 2.0)
            w = {"w": torch.empty(3, 4)}
            dcp.load(w, storage_reader=C._MetadataOnceReader(d), no_dist=True)
            assert torch.equal(w["w"], 2.0 * torch.arange(12, dtype=torch.float32).reshape(3, 4))
            assert len(calls) == 2, calls

            # dropped at the end of a load: the next load reads again
            C._drop_metadata_cache()
            C._MetadataOnceReader(d).read_metadata()
            assert len(calls) == 3, calls
    finally:
        FileSystemReader.read_metadata = orig
        C._drop_metadata_cache()


if __name__ == "__main__":
    test_metadata_read_once_and_invalidated()
    print("ok")
