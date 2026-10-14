# OELLM port onto Megatron-Core 0.19

Branch `oellm/v0.19`, based on tag `core_v0.19.0` (package version 0.19.1).

Replaces the previous OELLM fork branch `auto_restart`, which was based on
`3522e9eb9` (2025-12-10, megatron-core 0.16.0.dev). The last commit shared with
NVIDIA/Megatron-LM was `fcc1aaf16` (2025-12-05, "ci: Avoid naming collision
(#2558)"); the gap between the two is 5 commits / 6 files / 53 lines (an AMD
legacy-kernel disable and an HF tokenizer default) and was not carried over.

The port is organised by **feature**, not by original commit. Several of the
old commits are fixups that partly revert each other, so replaying them in
order would mean conflict-resolving the same feature repeatedly against a tree
eight months newer.

## Disposition

| # | Feature | Old commits | Status |
|---|---------|-------------|--------|
| 3 | Reset FT rank-monitor client on in-process restart | `1d1d633ad` | **ported** — `450c31c41` |
| 4 | `_is_usable()` guard in `ft_integration` | `c2a2adb78` | **dropped** — fixed upstream |
| 1 | Energy monitor null-NVML guard | `7231359bc` | **dropped** — fixed upstream |
| 8 | `--packed-doc-attention` | `aa7d6d5f3`, `3192738e1`, `d3f3d5890` | **ported** — as SCATTER on top of upstream's `--dataloader-inter-document-masking`, see below |
| 2 | Skip wandb artifact for non-persistent checkpoints | `6be53e28f` | **ported** — `b503bd8d3` |
| 10 | `--save-extra-steps` | `7c44f6d5e` | **ported** — `8dc8d7f80`, re-expressed as a `CheckpointConfig` field |
| 5 | `--dataloader-prefetch-factor` | `fc7b3b204` | **ported** — `8b2e30175` |
| 9 | Tokens/s/GPU logging (+ wandb memory stats) | `cfae32698` | **ported** — `38898bff0` |
| 6 | norm-gain weight decay | `e7c9d4dd2` | **ported** — `2157fea9d`, rewritten and **split into two knobs** |
| 7 | `--final-logit-softcapping` + `--output-z-loss-coeff` | `67dff69fd`, `19238a240` | **ported** — re-expressed against 0.19's explicit process groups |
| — | `.github/` CI workflows | `d5652c26a`, `f858f45ab`, `055f7defc` | **dropped** — OELLM repo CI, not training |

## Why the dropped ones were dropped

**#4 — `_is_usable()` guard.** The bug was that `train()` closed the rank-monitor
socket via a bare `get_rank_monitor_client().shutdown_workload_monitoring()`
(0.16 `training.py:2642`) without clearing `_GLOBAL_RANK_MONITOR_CLIENT`, so the
later `on_checkpointing_start()` hooks saw a non-None but dead client and raised
`RankMonitorClientError` on the natural-completion path (JUPITER job 1355470).
In 0.19 that early call site is gone: the only `shutdown_workload_monitoring()`
in the tree is inside `ft_integration.shutdown()` (`ft_integration.py:214`),
which clears the global two lines later. The shut-down-but-not-None state can no
longer arise, so the existing `is not None` guards are sufficient.

**#1 — Energy monitor null-NVML guard.** Upstream fixed this independently with
the same guard and the same rationale (`energy_monitor.py:62-65`, "Passing None
to nvmlDeviceGetTotalEnergyConsumption can cause a core dump").

**#8 — REVERSED 2026-08-27, then PORTED the same day.**
Measured on JUPITER, upstream's `--dataloader-inter-document-masking` does not work for plain
GPT pretraining. Two independent blockers, both of which our own patch already solved:

1. **The pipeline p2p buffers are not folded.** `flatten_batch_for_packed_sequences`
   (`core/utils.py:2565`) folds the micro-batch `[b, s]` into one packed sequence `[1, b*s]`,
   because TE's thd kernels want `[t, h, d]` and mcore reaches that via `query.squeeze(1)`
   (`attention.py:1469`). But `train_step` sizes the p2p buffers from
   `(args.seq_length, args.micro_batch_size)` unfolded, so a middle stage receives a
   `[b*s, 1, h]` activation into an `[s, b, h]` buffer. The element count matches, nothing
   errors at the boundary, `squeeze(1)` is then a no-op because dim 1 is `b`, and the 4-D
   query reaches the thd rope kernel: `RuntimeError: expected 3D tensor`, rank 31 (stage 1 of
   4), job 1511160. At `micro_batch_size 1` the fold is the identity, which is why PP>1 with
   mbs=1 never showed it — and why the original diagnosis ("nothing produces thd layout")
   was wrong; upstream produces it correctly, it just does not tell the pipeline.
2. **Its data plumbing is our deleted "local" mode.** `get_batch` widens the read to middle
   pipeline stages (`has_cu_seqlens`), which requires `is_dataset_built_on_rank(...,
   is_packed_sequence=True)` -- the dataset built on EVERY stage. Upstream sets that flag for
   SFT only, so masking at PP>1 first dies with `TypeError: 'NoneType' object is not an
   iterator` (jobs 1511085 / 1511146, ranks 20/24 at PP=4). Setting it correctly fixes that
   crash but reintroduces the memory blow-up local mode was deleted FOR: one index map per
   model chunk, ~412 GB against ~472 GB of node RAM at PP=4/VPP=4 (job 1497412). The 16-node
   test survived only because it runs VPP=2 (~206 GB). Our SCATTER mode avoids this -- publish
   once on the reading stage, broadcast cu_seqlens over the model-parallel group before the
   schedule -- and measured no cost at 512 nodes (-2.8%, inside a +-3.2% noise floor, 1498007).

### What was actually ported, and what was deliberately NOT

The 0.16 module was NOT copied back. Upstream 0.19 does the dataloader half properly and does
more of it than our patch ever did — `cu_seqlens` from the document index rather than from an
EOD scan, per-document position-id reset, per-sample merging across the micro-batch,
varlen-aware FLOPs accounting, `cu_seqlens_padded` / `local_cp_size` / hybrid-CP plumbing, and
the SFT path. Re-importing our EOD derivation on top of that would have been two sources of
truth for the same integers. Only the **distribution** was taken from the fork:

- **new** `megatron/training/packed_doc_attention.py` — per-iteration prefetch, one
  header + one payload broadcast over the model-parallel group, per-chunk queues,
  `share_metadata_from` for VPP, and the CP boundary snapping below.
- `pretrain_gpt.py` — `get_batch` split into `_read_microbatch` (shared with the upstream and
  SFT paths, so the read pipeline has one implementation) and `_get_batch_scattered`;
  `is_packed_sequence` returned to SFT-only; prefetch hook registered at import;
  `--packed-doc-attention-log-cu-seqlens` debug logging.
- `megatron/training/training.py` — `maybe_prefetch_cu_seqlens` before all three
  `forward_backward_func` call sites (train_step, evaluate, the non-loss-data pass), and
  `get_pipeline_tensor_shapes`, which fixes blocker 1 for SFT as well as for masking.
- `megatron/training/arguments.py` — `_validate_packed_doc_attention`, the fork's guard set
  re-expressed against the upstream flag name.
- `tests/unit_tests/test_packed_doc_attention.py` — 68 tests, CPU-only.

**Context parallelism is now supported, which the 0.16 patch asserted against.** TE's
context-parallel thd path requires a non-None `cu_seqlens_padded` (`context_parallel.py:419`,
"cu_seqlens_padded is required for THD format!") and document lengths divisible by
`2 * cp_size` (`thd_get_partitioned_indices`). `GPTDataset` produces neither, and upstream
side-steps that by passing `use_per_sequence_balancing=True` — a zigzag over the whole pack,
which is not the per-document layout TE's thd kernels index with. Instead
`snap_cu_seqlens_to_multiple` moves each document boundary to the nearest multiple of
`2 * cp_size`, which keeps the token stream and every tensor shape untouched and lets the
per-document zigzag run. It costs at most `2*cp_size - 1` tokens per boundary still attending
into the neighbouring document; `validate_args` warns once at rank 0 with the number.

### VALIDATED ON HARDWARE 2026-08-28 — PP=4 masking runs, and costs nothing

`fp8_speed_test_fp32grad_docmask.yaml`, 16 nodes, TP4 x PP4 x DP4, VPP=2, mbs=2, FP8, 20
iterations. Jobs 1516064 (control) / 1516065 (masking on), both COMPLETED 0:0 in 4m30s.
**Zero errors** — past both walls that killed the earlier attempts (no
`TypeError: 'NoneType' object is not an iterator`, no `RuntimeError: expected 3D tensor`).

| steady state, iters 6-20, medians | ms/iter | tok/s/GPU |
|---|---|---|
| control (no masking) | 2333.0 | 1755.7 |
| masking on | 2340.1 | 1750.3 |
| | **+0.3%** | **-0.3%** |

max/median iteration time 1.01x and 1.02x — no dataloader tail, unlike the 0.16-era runs.
Masking was genuinely active: the argument dumps differ only in that flag, and the losses
diverge at ITERATION 1 (1.352874E+01 vs 1.352256E+01, -4.6E-04) where the two arms still hold
identical weights and read identical data, so the attention pattern is the only thing that
can have moved them apart.

### 512 NODES AT PRODUCTION SETTINGS, VPP=4 — PASS (2026-08-28)

Jobs 1516808 (control) / 1516809 (masking), `speed_prodsettings_n512_019.yaml`, 50 iterations,
production settings verbatim: fp8 delayed/1024/max, the 16-segment layout (**VPP=4**),
tp_comm_overlap, GBS 4096, TP4 x PP4, DP=128.

| steady state, iter >= 15 | ms/iter | tok/s/GPU | TFLOP/s |
|---|---|---|---|
| control (0.19, no masking) | 4193.2 | 1953.6 | 406.7 |
| masking on | 4265.8 | 1920.4 | 389.4 |
| | **+1.7%** | **-1.7%** | — |

**Two results in one.** The control's 406.7 TFLOP/s against the live 0.16 flagship's
408.1-408.2 settles the migration question: 0.19 is at parity (-0.3%, inside the +-3.2% noise
floor). And masking costs -1.7% tok/s/GPU at production shape.

Do NOT read the TFLOP/s column across arms: with corrected varlen accounting the packed arm
genuinely reports fewer FLOPs for the same work (TF-per-tok 0.203 vs 0.208), so its lower
TFLOP/s is the accounting being right, not the run being slow.

**This is the first hardware exercise of `share_metadata_from` at VPP=4** — the layout where
the 0.16 scatter broadcast an empty cu_seqlens and killed 1865 ranks in
`fused_rope.cu:477 Assertion failed: cu_seqlens != nullptr` (job 1497767). Clean, 50/50
iterations, and the FLOPs guard stayed silent.

### CONTEXT PARALLELISM — PASS (2026-08-28)

Jobs 1517312 (control) / 1517313 (masking), `cp_docmask_n32.yaml`, 32 nodes,
TP4 x **CP2** x PP4, DP=4 (chosen to match the 16-node gate's DP, so memory and data ordering
are unchanged), 20 iterations.

| steady state, iters 6-20 | ms/iter | tok/s/GPU | mem |
|---|---|---|---|
| CP=2 control | 2147.1 | 953.8 | 0.455 |
| CP=2 masking | 2197.9 | 931.8 | 0.455 |
| | **+2.4%** | **-2.3%** | |

TE accepts the snapped `cu_seqlens_padded` and the per-document zigzag. The cost is higher
than CP=1's +0.3%, which is expected: the CP path adds `thd_get_partitioned_indices` and TE's
context-parallel thd attention. Masking is active — TF-per-tok 0.203 vs the control's 0.208,
the same signature as every other shape.

### Upstream bug #3: context parallelism is broken on any BLENDED dataset

The CP **control** crashed where the masking arm did not — `IndexError: tuple index out of
range`, zero iterations, rank 16 (job 1516834).

`BlendedDataset.__getitem__` returns `{"dataset_id": ..., **sample}` (blended_dataset.py:109);
collated that is a 1-D `[micro_batch_size]` tensor. `get_batch` normalises the BATCH_KEYS
entries but does not strip unknown ones, so it reaches
`_get_batch_on_this_cp_rank_per_sequence_balancing`, which iterates EVERY key and evaluates
`val.shape[seq_dim]`. Upstream guards the *return* against exactly this key — get_batch's own
comment names "provenance fields wrappers like BlendedDataset add (e.g. dataset_id)" — but the
CP split runs before that return.

This breaks **any CP>1 run on a blended dataset, with or without masking**; the production
datamix is a blend. The masking path escaped only because it routes to per-DOCUMENT balancing,
which touches the four sequence tensors by name.

Fixed by skipping entries with no sequence dimension (`not torch.is_tensor(val) or
val.dim() <= seq_dim`) — by RANK rather than by name, so it covers the whole class and cannot
exclude a real sequence tensor (tokens/labels/loss_mask/position_ids are 2-D, attention_mask
is 4-D with seq_dim=2).

Still not exercised: **CP>2**, and CP combined with sequence parallelism at TP>1 beyond this
shape.

### Two upstream metrics bugs found by reading that result

**1. varlen FLOPs overcount by VPP.** The masking arm reported **717.5 TFLOP/s/GPU against
the control's 366.7 — a factor 1.96** — while tokens/s/GPU agreed to 0.3%. Tokens/s comes
from the closed form and was right; the FLOPs number comes from
`consume_seqlen_stats_in_iteration`, whose `dedup = tp_size * cp_size * pp_size` omits the
virtual pipeline size.

The dedup factor is best derived rather than pattern-matched. `update_seqlen_stats_from_cu_seqlens`
runs inside `forward_step`, once per (model chunk, micro-batch) on every rank, and each call
adds the whole micro-batch's `cu_seqlens`. TP replicates it (TP broadcast), PP replicates it
(the scatter), CP replicates it (`cu_seqlens` is in `METADATA_KEYS`, so the split leaves it
untouched and each CP rank reports the FULL pack). VPP is **not a rank axis at all** — it
multiplies the call count per rank — which is exactly why it was missed. DP is the one axis
that must survive. So the world all-reduce yields `TP * CP * PP * VPP * truth`.

**2. evaluation leaks into the next training iteration.** `evaluate()` drives the same
`forward_step`, so it accumulates, but only the training loop calls `consume`. The training
iteration after each validation folded the validation micro-batches into its FLOPs number.
Fixed with `reset_seqlen_stats()` per eval step.

Both are metrics-only — nothing about training depends on either, and the gate's `+0.3%`
conclusion is unaffected because it is read off ms/iter and tok/s/GPU, both closed-form. But
TFLOP/s is the number a throughput comparison gets reported with, so it had to be right
before the 512-node arm runs.

**A guard now makes this class of bug self-reporting.** `check_seqlen_stats_token_count`
compares the recovered `sum(L_i)` against `global_batch_size * seq_length` — an independent
count of the same quantity, exact for pretraining because GPTDataset folds any shortfall into
the last document and boundary snapping preserves the total. It warns once at rank 0 with the
ratio. It would have fired at iteration 1 of a single run; catching this needed an A/B against
an unpacked control instead. Skipped for SFT, whose dataset legitimately pads.

Test coverage for all of it: a (TP, CP, PP, VPP, DP) grid that reconstructs the accumulator
from the per-rank call pattern and checks `consume` recovers the true global totals, plus a
pair that feed our folded shapes to the REAL `schedules.get_tensor_shapes` and compare against
an independently computed activation shape. Verified to FAIL against the pre-fix formulas
(10 and 17 failures respectively), so they are not decoration.

The original (now superseded) rationale follows.

**#8 — original call.** Upstream 0.19 added
`--dataloader-inter-document-masking`, which is a superset: `cu_seqlens` is built
in `gpt_dataset.py` from the document index, per-document position IDs are reset
there, middle pipeline stages are forced to pull the batch when `has_cu_seqlens`
is set (`pretrain_gpt.py:117`), `create_attention_mask_in_dataloader` is
auto-disabled, and unlike ours it also supports context parallelism
(`cu_seqlens_padded`, `local_cp_size`, `hybrid_cp_group`), the SFT path, and
varlen-aware FLOPs accounting.

Two behavioural differences to validate rather than assume:

- Ours folded the whole micro-batch into a single packed sequence; upstream
  keeps per-sample `cu_seqlens` padded to `seq_length + 1`. Same masking
  semantics, different attention shape — throughput numbers in
  `packed_doc_attention_speed_n512.yaml` do not transfer.
- Upstream passes `use_per_sequence_balancing=args.dataloader_inter_document_masking`
  (`pretrain_gpt.py:172`). That is a context-parallel batch-slicing option
  consumed by `get_batch_on_this_cp_rank` (`core/utils.py:2617`, zigzag CP
  balancing), NOT a dataloader or sampler setting — it does not interact with
  feature #5.

## Feature 7 notes (softcapping + LM-head z-loss)

Three places where 0.19 forced a different expression, not a copy:

- **`return_logsumexp` had to move behind `tp_group`.** 0.19 added `tp_group` as the 4th
  positional parameter of `_VocabParallelCrossEntropy.forward`, exactly where the old fork put
  `return_logsumexp`. Appending it as the 5th keeps both call conventions working; `backward`
  correspondingly returns five values, one per forward input after `ctx`.
- **The z-loss tracker uses `self.pg_collection.dp_cp`, not
  `parallel_state.get_data_parallel_group(with_context_parallel=True)`.** `CLAUDE.md` forbids
  new direct global-process-group reads in `megatron/core`, and `dp_cp` is exactly that group.
- **MTP suppression is now a `functools.partial`.** The old fork called
  `self.compute_language_model_loss(..., record_z_loss=False)` directly; 0.19 hands MTP the
  bound method as a callable (`multi_token_prediction.py:855` calls it positionally), so the
  flag is bound at the call site in `gpt_model.py` instead. The `output_processor` hook keeps
  the default `record_z_loss=True` -- it computes the main head's loss, not an auxiliary one.

Soft-capping is applied AFTER 0.19's new MuP `_scale_logits`, so the cap bounds the logits that
actually reach the loss rather than a pre-scaled version.

`tests/unit_tests/tensor_parallel/test_cross_entropy.py` gains
`test_vocab_parallel_output_zloss[native|fused|standalone]`, which checks logZ and the
z-loss gradient against a full-vocabulary PyTorch reference. The pre-existing
`test_language_module_unfused_loss_passes_tp_group` needed two adjustments, because the port
extends the call it fakes: its stub now accepts `return_logsumexp`, and its `SimpleNamespace`
config carries `output_z_loss_coeff=None`.

Verified on 1 GPU: the 3 new tests and both pre-existing `tp_group` tests pass. The rest of
`tests/unit_tests/tensor_parallel/` fails only on `world_size (1) is not divisible by 4/8`
(needs 8 GPUs) and a local `share_storage_ext` C++ build error -- nothing attributable to the
port. **A multi-rank run is still owed**, since TP correctness is exactly what the new test
asserts and at TP=1 every all-reduce is a no-op.

## Config migration (oellm-autoexp)

`packed_doc_attention: true` becomes `dataloader_inter_document_masking: true`,
and `packed_doc_attention_log_cu_seqlens` is removed. Affected:
`config/backend/megatron/base_defaults.yaml`,
`config/backend/megatron/oellm_32b_dense/data/oellm_256k_15TT_jupiter_train.yaml`,
eight `config/experiments/oellm_32b_dense/packed_doc_attention_*.yaml`, plus
`scripts/korbi/check_cu_seqlens_agreement.py`,
`scripts/korbi/compare_doc_attention_arms.py` and
`scripts/tests/test_packed_doc_attention.sh`.

`scaler_wd_mult: X` is **replaced by two settings** (feature #6 was deliberately
changed during the port, not just moved): `qk_layernorm_wd_mult` for the
q/k-layernorm gains and `residual_norm_wd_mult` for the remaining
residual-stream norm gains. Setting both to the old `scaler_wd_mult` value
reproduces the old behaviour exactly. Affected:
`config/backend/megatron/base_defaults.yaml` and
`config/experiments/oellm_32b_dense/stability_check_nodes512_lr3e-4.yaml`.

The tensorboard/wandb metric key `throughput` is now logged as `TFLOPS`, and
`Tokens per second per GPU` is added alongside it (feature #9) — matching what
every OELLM run has recorded so far, so existing dashboards keep working.

Surviving OELLM arguments: `--qk-layernorm-wd-mult`, `--residual-norm-wd-mult`,
`--final-logit-softcapping`, `--output-z-loss-coeff`, `--save-extra-steps`,
`--dataloader-prefetch-factor`.

## Container gate: REOPENED — 26.04 DOES work, after a one-line version relaxation

Superseded 2026-08-26. The earlier verdict ("26.04 is out") was based on a wrong reading of
the blocker; the corrected finding is that **no container move is required**.

| container | torch | TE | nvrx | ships mcore | mcore 0.19 imports? |
|---|---|---|---|---|---|
| `nemo_26.04.sif` | 2.11.0a0 | 2.14.0 | 0.6.0.dev33 | 0.17.0rc0 | **yes, with the patch below** |
| `nemo_26.06.sif` | 2.12.0a0 | 2.16.0 | 0.6.0 | 0.18.2 | yes |
| `nemo_26.08.00.sif` | 2.13.0a0 | 2.17.1 | 0.6.0 | **0.19.0** | yes |

The symptom was real: `import megatron.core` died in
`dist_checkpointing/strategies/nvrx.py` with
`AssertionError: Minimum required nvidia-resiliency-ext package version is 0.6.0.`, because
`has_nvrx_async_support()` runs unconditionally at import and PEP 440 orders
`0.6.0.dev33 < 0.6.0`.

**The diagnosis was wrong.** The earlier note claimed 26.04's nvrx was missing
`filesystem_async._results_queue`, and concluded that bypassing the assert would silently
yield `HAVE_NVRX = False`. Checked directly in-container, all NINE symbols the function
requires are present in `0.6.0.dev33+15a8515` — `AsyncCallsQueue`, `AsyncRequest`,
`CachedMetadataFileSystemReader`, `FileSystemWriterAsync`, `get_write_results_queue`,
`CheckpointMetadataCache`, `save_state_dict_async_finalize`, `save_state_dict_async_plan`
and `_results_queue`. It is a version-STRING gate, not an API gate.

`is_nvrx_min_version()` now also accepts a dev/pre-release of exactly the required release
tuple, so `0.6.0.dev33` satisfies a `0.6.0` minimum while `0.5.0` still does not. The symbol
check below the assert is untouched and remains the real gate: a genuinely incomplete build
still returns False. Measured after the change, with this branch on `PYTHONPATH`:

```
nemo_26.04:    mcore 0.19.1 | nvrx 0.6.0.dev33+15a8515 | HAVE_NVRX=True
nemo_26.08.00: mcore 0.19.1 | nvrx 0.6.0              | HAVE_NVRX=True
```

and `tests/unit_tests/tensor_parallel/test_cross_entropy.py` passes inside **nemo_26.04**
(5 passed; the 6th needs 8 GPUs and the login node exposes 1).

**Still owed before trusting this in production:** `HAVE_NVRX=True` proves the symbols
resolve, not that async checkpointing *behaves* correctly on a dev build. A functional
async-save + resume test on 26.04 is the remaining gate. If that fails, `nemo_26.08.00`
remains the fallback — it ships megatron-core 0.19.0 itself, i.e. the pairing NVIDIA
validated — at the cost of a container move whose throughput is unmeasured (26.06 cost
-1.5% vs 26.04).

Note torch and TE were never blockers: 26.04's torch 2.11 clears 0.19's `torch>=2.6.0`, and
TE 2.14 clears its TE 2.6/2.7 guards.

### What has been verified in-container

With `PYTHONPATH` pointing at this branch (confirmed resolving to it, not the
container's own megatron-core), on **both 26.06 and 26.08**:

- `import megatron.core` -> 0.19.1
- the ported arguments parse: `--qk-layernorm-wd-mult`,
  `--residual-norm-wd-mult`, `--dataloader-prefetch-factor`, `--save-extra-steps`
- `wd_mult` resolves identically to the local run for every parameter class,
  including the unchanged upstream defaults
- 83 unit tests pass (`test_argument_utils.py` — which covers the extended
  `CheckpointConfig` — `optimizer/test_param_group_identifier_keys.py`,
  `test_optimizer_param_scheduler.py`)

All of this is CPU-only. No GPU test, no training step, and no loss-parity check
against the old fork has been run yet.

## Open gates

- **Checkpoint resume.** The 32B flagship has live `torch_dist` checkpoints
  written by the 0.16-based fork. 0.19 must be able to resume them, or the
  switch can only happen at a run boundary.
- **nvrx async checkpointing on 26.04's dev build.** Symbols resolve; behaviour untested.
  This is now the gate that decides whether the container can stay put.
- **Throughput on 26.08.** Unmeasured, and only relevant if 26.04 is ruled out after all.
- **Loss parity.** ~~Feature #7 is now ported, but nothing has been validated by running a
  training step.~~ **CLEARED 2026-08-26.** 0.13B, 2 nodes, 200 iterations, same data and seed,
  `cross_entropy_fusion_impl: native` on both arms so megatron-core is the only moving part
  (jobs 1511166 / 1511165, config
  `config/experiments/korbi/mcore019_lossparity_130M.yaml`):

  | iter | 0.16 | 0.19 | rel |
  |------|------|------|-----|
  | 10 | 1.067719E+01 | 1.067720E+01 | 9E-07 |
  | 100 | 7.170163E+00 | 7.168594E+00 | -2E-04 |
  | 200 | 5.870253E+00 | 5.876484E+00 | 1E-03 |

  Worst 1.5E-03 at iteration 150, and 0.19 sits above 0.16 on 17 of 20 points — i.e. it
  crosses, so the drift has no sign. That is floating-point chaos amplification off a
  bitwise-different kernel mix, which is what a correct port looks like; a bias would have
  been a bug. Re-check with `scripts/korbi/compare_loss_parity.py`.

  NB this ran TP=1, so it does NOT exercise the z-loss all-reduces: `vocab_parallel_logsumexp`
  and the `return_logsumexp` plumbing are no-ops at TP=1. That still needs a TP>1 run.
- **Packed document attention on hardware.** ~~never run a step~~ **PP=4 CLEARED 2026-08-28**
  (jobs 1516064/65, +0.3% ms/iter — see feature #8). Remaining: **VPP=4** at the production
  layout, and **CP>1**, which has never run at all.
- **The config migration below is NOT done**, and must not be done ahead of the switch: the
  live flagship configs point at `submodules/Megatron-LM` (the 0.16 fork), where
  `packed_doc_attention` is still the working flag. Renaming to
  `dataloader_inter_document_masking` before the container and checkpoint gates clear would
  break running configs.
