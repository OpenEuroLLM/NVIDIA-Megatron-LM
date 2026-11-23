#!/usr/bin/env python
"""Thin wrapper around the SHARED Megatron pretrain_gpt.py that freezes the MoE
router (gating) weights before training. Does NOT modify the shared repo.

It imports the shared pretrain_gpt module (so identical code/version as the
normal runs), then wraps its model_provider to set requires_grad=False on every
parameter whose name contains '.router.' (the token->expert router). Everything
else is passed through to the stock pretrain() call verbatim.

Invoke exactly like pretrain_gpt.py (same CLI args). MEGATRON_ROOT env must point
at the shared Megatron checkout (the sbatch already exports it).
"""
import os, sys, time

MEGATRON_ROOT = os.environ.get(
    "MEGATRON_ROOT", "/shared_silo/scratch/shared/sft/repos/NVIDIA-Megatron-LM"
)
# Make the SHARED repo win for `import pretrain_gpt`, `gpt_builders`, `megatron.*`.
sys.path.insert(0, MEGATRON_ROOT)
sys.argv[0] = os.path.join(MEGATRON_ROOT, "pretrain_gpt.py")

import pretrain_gpt as pg  # noqa: E402


def _freeze_router_builder(*args, **kwargs):
    """Build the model via the stock provider, then freeze MoE router params."""
    model = pg.model_provider(pg.gpt_builder, *args, **kwargs)
    frozen = 0
    sample = []
    for name, p in model.named_parameters():
        if ".router." in name:
            p.requires_grad = False
            frozen += 1
            if len(sample) < 4:
                sample.append(name)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[freeze-router] froze {frozen} router params; examples={sample}",
              flush=True)
    return model


if __name__ == "__main__":
    pg._MAIN_ENTRY_TIME = time.time()
    pg.set_startup_timestamps(program_start=pg._PROGRAM_START_TIME,
                              main_entry=pg._MAIN_ENTRY_TIME)
    pg.train_valid_test_datasets_provider.is_distributed = True
    _pretrain, _store = pg.inprocess_restart.maybe_wrap_for_inprocess_restart(pg.pretrain)
    _ = pg.parse_and_validate_args(
        extra_args_provider=pg.add_modelopt_args if pg.has_nvidia_modelopt else None,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    _pretrain(
        pg.train_valid_test_datasets_provider,
        _freeze_router_builder,
        pg.ModelType.encoder_or_decoder,
        pg.forward_step,
        store=_store,
        get_embedding_ranks=pg.get_embedding_ranks,
    )
