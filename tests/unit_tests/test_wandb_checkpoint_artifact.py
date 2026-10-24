# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The W&B checkpoint artifact never fails a save (the tracker file is rank 0's, written without a
barrier before the last rank logs the artifact)."""

import threading
import time
from types import SimpleNamespace

from megatron.training import wandb_utils


class _Artifact:
    def __init__(self, *a, **k):
        self.files = []

    def add_reference(self, *a, **k):
        pass

    def add_file(self, path):
        from pathlib import Path

        if not Path(path).is_file():
            raise ValueError(f"Path is not a file: {path!r}")
        self.files.append(path)


def _writer(logged):
    run = SimpleNamespace(entity="e", project="p", log_artifact=lambda a, aliases: logged.append(a))
    return SimpleNamespace(Artifact=_Artifact, run=run)


def test_waits_for_a_late_tracker(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(wandb_utils, "get_wandb_writer", lambda: _writer(logged))
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    threading.Timer(1.5, lambda: tracker.write_text("4800")).start()
    t0 = time.monotonic()
    wandb_utils.on_save_checkpoint_success(str(tmp_path / "iter_0004800"), str(tracker), str(tmp_path), 4800)
    assert logged and logged[0].files == [str(tracker)]
    assert 1.0 < time.monotonic() - t0 < 30.0


def test_missing_tracker_does_not_raise(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(wandb_utils, "get_wandb_writer", lambda: _writer(logged))
    monkeypatch.setattr(wandb_utils.time, "sleep", lambda s: None)
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(wandb_utils.time, "monotonic", lambda: next(clock))
    wandb_utils.on_save_checkpoint_success(str(tmp_path / "iter_1"), str(tmp_path / "nope.txt"), str(tmp_path), 1)
    assert logged == []
