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
| 8 | `--packed-doc-attention` | `aa7d6d5f3`, `3192738e1`, `d3f3d5890` | **dropped** — superseded upstream |
| 2 | Skip wandb artifact for non-persistent checkpoints | `6be53e28f` | **ported** — `b503bd8d3` |
| 10 | `--save-extra-steps` | `7c44f6d5e` | **ported** — `8dc8d7f80`, re-expressed as a `CheckpointConfig` field |
| 5 | `--dataloader-prefetch-factor` | `fc7b3b204` | **ported** — `8b2e30175` |
| 9 | Tokens/s/GPU logging (+ wandb memory stats) | `cfae32698` | **ported** — `38898bff0` |
| 6 | norm-gain weight decay | `e7c9d4dd2` | **ported** — `2157fea9d`, rewritten and **split into two knobs** |
| 7 | `--final-logit-softcapping` + `--output-z-loss-coeff` | `67dff69fd`, `19238a240` | todo |
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

**#8 — `--packed-doc-attention`.** Upstream 0.19 added
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

## Container gate: RESOLVED — 26.04 is out, 26.06 and 26.08 work

Tested 2026-08-26 on JUPITER from `~/work/Projects/oellm-autoexp-debug`.

| container | torch | TE | nvrx | ships mcore | mcore 0.19 imports? |
|---|---|---|---|---|---|
| `nemo_26.04.sif` | 2.11.0a0 | 2.14.0 | **0.6.0.dev33** | 0.17.0rc0 | **NO** |
| `nemo_26.06.sif` | 2.12.0a0 | 2.16.0 | 0.6.0 | 0.18.2 | yes |
| `nemo_26.08.00.sif` | 2.13.0a0 | 2.17.1 | 0.6.0 | **0.19.0** | yes |

**`nemo_26.04.sif` cannot run megatron-core 0.19 at all**, and this has nothing
to do with the port — plain `import megatron.core` fails. TE and torch are fine
(2.14 and 2.11 clear 0.19's `torch>=2.6.0` and its TE 2.6/2.7 guards). The
blocker is nvidia-resiliency-ext:

```
megatron/core/__init__.py -> ... -> dist_checkpointing/strategies/torch.py:64
    HAVE_NVRX = has_nvrx_async_support()
  -> dist_checkpointing/strategies/nvrx.py:45
    AssertionError: Minimum required nvidia-resiliency-ext package version is 0.6.0.
```

`strategies/nvrx.py` is new in 0.19 and does not exist in the 0.16 fork.
`has_nvrx_async_support()` runs unconditionally at import and *asserts* instead
of returning False, so an older nvrx makes the whole package unimportable
whether or not nvrx checkpointing is used. 26.04 ships `0.6.0.dev33+15a8515`,
and PEP 440 orders `0.6.0.dev33 < 0.6.0`.

**The assert is legitimate, not cosmetic — do not patch it out.** Of the nine
API symbols the check requires, 26.04's nvrx has eight; it is missing
`filesystem_async._results_queue`, which nvrx 0.6.0 final does have (verified on
both 26.06 and 26.08). Bypassing the assert would therefore yield
`HAVE_NVRX = False` anyway and silently disable nvrx async checkpointing, which
the FT/async-save setup depends on.

So adopting 0.19 requires a container move. `nemo_26.08.00.sif` is the natural
target: it ships megatron-core 0.19.0 itself, i.e. it is the pairing NVIDIA
validated 0.19 against. Note the previous 26.06 evaluation measured -1.5%
throughput and +22% checkpoint time against 26.04, so the container change has
its own cost that needs re-measuring for 26.08.

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
- **Throughput on 26.08.** Unmeasured. 26.06 cost -1.5% vs 26.04.
- **Loss parity.** Feature #7 (softcap + LM-head z-loss) is not ported yet, and
  nothing so far has been validated by running a training step.
