# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""--packing-plan-path: serve precomputed sequences in place of some training samples.

A packing plan (built offline, e.g. by the superproject's scripts/korbi/cross_doc_seqs/build_plan.py)
lists, for a set of the training dataset's sample indices, the sequence to serve instead: segments
(dataset, document, offset, length) into the SAME blend's indexed datasets, in Megatron's virtual
document coordinates (with --dataloader-add-bos, offset 0 is the BOS prepended at read time). Each
plan sequence is sequence_length + 1 tokens, like a regular sample.

Nothing else changes: the sampler, consumed samples, checkpoints and resumes are untouched, a step
reads the same sample indices as without the plan, and every index the plan does not list (all
steps outside the planned ranges) is served by the wrapped dataset exactly as before. The offline
builder chooses which indices to replace (the ones the sampler serves in the planned steps) and
keeps the window's tokens (a repacking moves documents between the window's sequences).

Boundaries (--packing-plan-boundaries) decide the cu_seqlens of a plan sequence (with
--dataloader-inter-document-masking; without it every sequence is plain causal anyway):
    doc    one segment per document piece (as GPTDataset: document masking, RoPE restart)
    group  one segment per run of pieces with the same plan group (a semantic cluster): attention
           and recurrent state flow between related documents only
    none   the whole sequence is one segment (cross-document attention everywhere)

Plan directory format (format 1): slot_bidx.npy (sorted int64 dataset indices), slot_seq.npy (the
plan sequence of each), seq_ptr.npy (sequence -> segment range), seg_{ds,doc,off,len,group}.npy,
meta.json {"format": 1, "blend_md5", "seq_length", ...}.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy
import torch

logger = logging.getLogger(__name__)

BOUNDARY_MODES = ("doc", "group", "none")


class PackingPlan:
    """One plan directory (arrays memory-mapped on first use: fork-safe for loader workers)."""

    _KEYS = ("slot_bidx", "slot_seq", "seq_ptr", "seg_ds", "seg_doc", "seg_off", "seg_len", "seg_group")

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        assert self.meta.get("format") == 1, f"{path}: unknown packing plan format {self.meta.get('format')}"
        self._arrays = None

    def __getattr__(self, name):
        if name in PackingPlan._KEYS:
            if self.__dict__.get("_arrays") is None:
                self._arrays = {
                    k: numpy.load(os.path.join(self.path, k + ".npy"), mmap_mode="r") for k in PackingPlan._KEYS
                }
            return self._arrays[name]
        raise AttributeError(name)


class PackingPlanDataset(torch.utils.data.Dataset):
    """The training dataset (a BlendedDataset of GPTDatasets, or one GPTDataset) with the samples
    listed in the plans replaced by the plan sequences."""

    def __init__(self, dataset, plan_paths: List[str], boundaries: str = "doc"):
        assert boundaries in BOUNDARY_MODES, f"--packing-plan-boundaries {boundaries}"
        self.dataset = dataset
        self.boundaries = boundaries
        self.plans = [PackingPlan(p) for p in plan_paths]
        self._gpt = list(dataset.datasets) if hasattr(dataset, "datasets") else [dataset]
        blend_md5 = getattr(dataset, "unique_description_hash", None)
        seq = self._gpt[0].config.sequence_length
        for p in self.plans:
            assert p.meta["seq_length"] == seq, (
                f"{p.path}: plan sequence length {p.meta['seq_length']} != {seq}")
            assert p.meta["blend_md5"] == blend_md5, (
                f"{p.path}: plan built for blend {p.meta['blend_md5']}, the training dataset is {blend_md5}")
        if boundaries != "none" and not self._gpt[0].config.inter_document_masking:
            logger.warning("--packing-plan-boundaries %s has no effect without "
                           "--dataloader-inter-document-masking (plain causal attention)", boundaries)
        self._index = None

    def _lookup_tables(self):
        if self._index is None:
            keys, owner = [], []
            for i, p in enumerate(self.plans):
                keys.append(numpy.asarray(p.slot_bidx))
                owner.append(numpy.full(len(keys[-1]), i, dtype=numpy.int32))
            k = numpy.concatenate(keys) if keys else numpy.zeros(0, numpy.int64)
            o = numpy.concatenate(owner) if owner else numpy.zeros(0, numpy.int32)
            pos = numpy.concatenate([numpy.arange(len(x)) for x in keys]) if keys else numpy.zeros(0, numpy.int64)
            order = numpy.argsort(k, kind="stable")
            k, o, pos = k[order], o[order], pos[order]
            assert len(k) == 0 or (numpy.diff(k) > 0).all(), "two plans replace the same sample"
            self._index = (k, o, pos)
        return self._index

    def plan_slot(self, idx: int):
        """(plan, slot) serving dataset index ``idx``, or None."""
        k, o, pos = self._lookup_tables()
        j = numpy.searchsorted(k, idx)
        if j < len(k) and k[j] == idx:
            return self.plans[int(o[j])], int(pos[j])
        return None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name):
        # everything else (split, config, unique_description_hash, ...) is the wrapped dataset's
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def plan_sequence(self, plan: PackingPlan, slot: int):
        """(text, document lengths, first dataset id) of the plan sequence at ``slot``."""
        s = int(plan.slot_seq[slot])
        a, b = int(plan.seq_ptr[s]), int(plan.seq_ptr[s + 1])
        ds = numpy.asarray(plan.seg_ds[a:b])
        doc = numpy.asarray(plan.seg_doc[a:b])
        off = numpy.asarray(plan.seg_off[a:b])
        ln = numpy.asarray(plan.seg_len[a:b])
        parts = [
            self._gpt[int(d)]._get_document(int(c), offset=int(o), length=int(n))
            for d, c, o, n in zip(ds, doc, off, ln)
        ]
        text = numpy.concatenate(parts).astype(numpy.int64)
        if self.boundaries == "doc":
            lengths = [int(n) for n in ln]
        elif self.boundaries == "group":
            grp = numpy.asarray(plan.seg_group[a:b])
            starts = numpy.nonzero(numpy.r_[True, grp[1:] != grp[:-1]])[0]
            lengths = [int(x) for x in numpy.add.reduceat(ln, starts)]
        else:
            lengths = [int(ln.sum())]
        return text, lengths, int(ds[0])

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        hit = None if idx is None else self.plan_slot(int(idx))
        if hit is None:
            return self.dataset[idx]
        text, lengths, ds = self.plan_sequence(*hit)
        gpt = self._gpt[ds]
        assert len(text) == gpt.config.sequence_length + gpt.config.add_extra_token_to_sequence, (
            f"plan sequence for sample {idx}: {len(text)} tokens")
        sample = gpt._sample_from_text(text, lengths, padding=False)
        if hasattr(self.dataset, "datasets"):
            return {"dataset_id": ds, **sample}
        return sample


def plan_coverage(plans, start: int):
    """End of the contiguous planned range that contains the step taken from checkpoint ``start``
    (plans cover steps from checkpoints [from_ckpt, to_ckpt)), or None if that step is unplanned."""
    ranges = sorted((int(p.meta["from_ckpt"]), int(p.meta["to_ckpt"])) for p in plans)
    end = None
    for a, b in ranges:
        if end is None:
            if a <= start < b:
                end = b
        elif a == end:
            end = b
        elif a > end:
            break
    return end


def check_plan_geometry(plans, args):
    """The plans replace the sample indices the sampler serves in their steps: they are only valid for
    the sampler geometry they were built with (global batch, reference layout, phase, train_iters)."""
    phases = getattr(args, "phase_transition_iterations", None) or []
    want = {
        "global_batch_size": args.global_batch_size,
        "ref_dp": getattr(args, "data_sampler_reference_dp_size", None),
        "ref_mbs": getattr(args, "data_sampler_reference_micro_batch_size", None),
        "ref_gbs": getattr(args, "data_sampler_reference_global_batch_size", None),
        "train_iters": args.train_iters,
    }
    for p in plans:
        geo = p.meta.get("geometry")
        assert geo is not None, f"{p.path}: plan without sampler geometry (built before format checks)"
        for k, v in want.items():
            assert geo.get(k) == v, f"{p.path}: plan built for {k}={geo.get(k)}, this run has {v}"
        assert geo.get("phase_start") in [0, *phases], (
            f"{p.path}: plan built for a phase starting at {geo.get('phase_start')}, this run's phase "
            f"transitions are {phases}")


def wrap_with_packing_plans(train_ds, args):
    """The provider's hook: wrap the training dataset when --packing-plan-path is given. Checks the
    plans' sampler geometry against the run and that the next step is planned; records the end of the
    planned range in args.packing_plan_end_iteration (the training loop saves and exits there unless
    --packing-plan-at-end continue)."""
    paths = getattr(args, "packing_plan_path", None)
    if not paths or train_ds is None:
        return train_ds
    expanded = []
    for p in paths:  # a directory of plan directories is expanded (sorted)
        if os.path.exists(os.path.join(p, "meta.json")):
            expanded.append(p)
        else:
            subs = sorted(os.path.join(p, d) for d in os.listdir(p)
                          if os.path.exists(os.path.join(p, d, "meta.json")))
            assert subs, f"--packing-plan-path {p}: no plan directory"
            expanded.extend(subs)
    ds = PackingPlanDataset(train_ds, expanded, getattr(args, "packing_plan_boundaries", "doc"))
    check_plan_geometry(ds.plans, args)
    at_end = getattr(args, "packing_plan_at_end", "exit")
    start = int(getattr(args, "iteration", 0) or 0)
    end = plan_coverage(ds.plans, start)
    if end is None:
        msg = (f"--packing-plan-path: the step after checkpoint {start} is not planned "
               f"(plans cover {[(p.meta['from_ckpt'], p.meta['to_ckpt']) for p in ds.plans]})")
        assert at_end == "continue", msg + "; pass --packing-plan-at-end continue to train without plans"
        print(f"> WARNING: {msg}; training on the regular samples", flush=True)
    args.packing_plan_end_iteration = end if at_end == "exit" else None
    k, _, _ = ds._lookup_tables()
    ranges = [(p.meta.get("from_ckpt"), p.meta.get("to_ckpt")) for p in ds.plans]
    logger.info("packing plans: %d (%s), %d replaced samples, boundaries=%s",
                len(ds.plans), ranges, len(k), ds.boundaries)
    print(f"> packing plans: {len(ds.plans)} plans covering checkpoints {ranges}, {len(k)} replaced "
          f"samples, boundaries={ds.boundaries}; planned from {start} to {end} "
          f"(at the end: {at_end})", flush=True)
    return ds
