# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Static-length cu_seqlens for packed document masking (PACKED_DOC_STATIC_SEGMENTS).

Inter-document masking hands attention one cu_seqlens row per sample: 0, then the document ends,
then the sequence length. The dataset already pads every row to ``sequence_length + 1`` entries
with trailing copies of ``sequence_length`` (gpt_dataset.py, so that default_collate can stack
samples with different document counts), and ``core.utils._merge_cu_seqlens_across_micro_batch``
strips those copies back to the real document count. The stripped row therefore has a different
length for every microbatch, which is why a CUDA graph would have to recapture every step -- and
why ``arguments.py`` rejects ``--cuda-graph-impl`` while the masking flag is on.

With ``PACKED_DOC_STATIC_SEGMENTS=N`` the rows keep ``N + 1`` entries: the real document ends
come first and the rest stay at ``sequence_length``, i.e. zero-length segments. The merged tensor
then has ``micro_batch_size * N + 1`` entries whatever the documents are, so every shape the
attention sees is static and a graph can be replayed. Equal neighbours are a zero-length segment,
which both the varlen and the page-sparse kernel treat as an empty range -- page-sparse attention
in per-token mode makes one kernel call for all documents, so the padded tail costs only the
cu_seqlens bytes, not extra launches.

N is an upper bound on the documents packed into one sequence; the dataset asserts on it, so a
run that exceeds it fails loudly instead of silently falling back to variable shapes. Raise N (or
set ``document_segment_length``) if that assert fires.
"""

import os

# 0 disables the mode: the padded rows keep sequence_length + 1 entries and are stripped as
# before, so a run without the variable set is byte-identical to the upstream path.
STATIC_SEGMENTS = int(os.environ.get("PACKED_DOC_STATIC_SEGMENTS", "0") or 0)


def enabled() -> bool:
    return STATIC_SEGMENTS > 0


def padded_length(seq_length: int) -> int:
    """Entries in one padded cu_seqlens row: ``N + 1`` in static mode, else ``seq_length + 1``."""
    return STATIC_SEGMENTS + 1 if STATIC_SEGMENTS > 0 else seq_length + 1
