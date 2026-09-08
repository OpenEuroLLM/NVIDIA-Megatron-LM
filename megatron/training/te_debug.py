# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Opt-in wiring of NVIDIA DL Framework Inspect for Transformer Engine layers.

`--te-debug-config <yaml>` hands a `nvdlfw_inspect` feature configuration to
every TE module of the model (LogTensorStats: min/max/mean/std/l1/l2/amax/
dynamic range per GEMM operand; LogFp8TensorStats: underflows%/overflows%/
scale_inv/MSE for the current FP8 recipe AND simulated for other recipes on the
same tensors; DisableFP8GEMM/PerTensorScaling per layer; ...). Everything here
is a no-op unless the flag is set.

WHERE THE NUMBERS GO. nvdlfw_inspect reduces each statistic over the tensor
reduction group (set to the TP group here) and then EVERY rank writes its own
`debug_statistics_logs/*_globalrank-<r>.log` file under --te-debug-log-dir.
That is fine for a 4-16 node probe and NOT fine for 2048 ranks on a shared
filesystem; keep this to small probes and use the reduced `--diag-*`
collectors (megatron/training/diagnostics.py) for production-scale runs.

Call order (see training.py): `init_te_debug(args)` after initialize_megatron
and BEFORE the model is built (TE reads the debug state at module
construction); `attach_te_debug(model, iteration)` once the model exists and
the checkpoint is loaded; `te_debug_step()` after every training step so the
feature buffers are reduced and flushed.
"""

import os

_ENABLED = False


def init_te_debug(args) -> None:
    """Initialize nvdlfw_inspect from --te-debug-config, if given."""
    global _ENABLED
    config = getattr(args, "te_debug_config", None)
    if not config:
        return
    import nvdlfw_inspect.api as debug_api
    import transformer_engine.debug.features as te_features

    log_dir = getattr(args, "te_debug_log_dir", None)
    if not log_dir:
        base = getattr(args, "tensorboard_dir", None) or "."
        log_dir = os.path.join(base, "te_debug")
    os.makedirs(log_dir, exist_ok=True)
    debug_api.initialize(
        config_file=config,
        feature_dirs=[os.path.dirname(te_features.__file__)],
        log_dir=log_dir,
        default_logging_enabled=True,
    )
    _ENABLED = True
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[te_debug] nvdlfw_inspect initialized from {config}; logs under {log_dir}", flush=True)


def attach_te_debug(model, iteration: int = 0) -> None:
    """Name every module (layer numbers are GLOBAL under PP) and set the
    reduction group; align the inspect step counter with the resumed iteration."""
    if not _ENABLED:
        return
    import nvdlfw_inspect.api as debug_api
    from megatron.core import parallel_state as mpu

    debug_api.infer_and_assign_layer_names(model if isinstance(model, list) else [model])
    debug_api.set_tensor_reduction_group(mpu.get_tensor_model_parallel_group())
    debug_api.initialize_training_step(int(iteration))


def te_debug_step() -> None:
    """Flush/reduce the feature buffers; call once per training step."""
    if not _ENABLED:
        return
    import nvdlfw_inspect.api as debug_api

    debug_api.step()
