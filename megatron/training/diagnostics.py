# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Opt-in training-time diagnostics for long runs.

WHY THIS EXISTS
---------------
When a multi-week run degrades, the training log records a loss and a global
gradient norm and nothing else, so the post-mortem has to be done from
checkpoints — which are written every few thousand iterations and therefore
cannot resolve *when* anything started. Everything here is a quantity that is
free at the point where training already computes it, but impossible to
reconstruct afterwards.

The four things it answers:

  1. NON-FINITE VALUES, and WHERE. `--check-for-nan-in-loss-and-grad` already
     aborts on a bad loss or gradient, but it does not say which parameter or
     which layer, and it never looks at activations at all. Here the scan is
     per-layer and reports the offending name.
  2. NORM GAINS, per layer. Megatron exempts every 1-D parameter from weight
     decay, so the RMSNorm gains have no restoring force and can drift for tens
     of thousands of iterations without anything noticing. If they do go
     extreme, the levers are `--residual-norm-wd-mult` / `--qk-layernorm-wd-mult`
     (already on this branch) or zero-centred RMSNorm.
  3. PER-LAYER GRADIENT NORMS. One global norm hides the early-vs-late layer
     asymmetry entirely; a vanishing first layer and an exploding last layer
     average to something unremarkable.
  4. CLIP EVENTS. A single clipped step is noise. Clipping that FIRES ON MANY
     SUCCESSIVE STEPS is an optimizer that has lost the plot, and the streak is
     the signal — which is why the streak counter runs on EVERY step even though
     it is only reported at `--diagnostics-interval`.

COST
----
Everything is gated on `--diagnostics-interval` (0 = off, and off is the
default). On a diagnostic iteration the cost is:

  * gains / grad norms / non-finite: one pass over parameters, all of it on
    device, and exactly ONE all-reduce each of a small fixed-size tensor.
  * activations: two reductions per transformer layer over one microbatch's
    hidden states. That is memory-bandwidth bound and measurably not free —
    roughly 20 GB of reads for a 64-layer 32B model — so it runs on the FIRST
    MICROBATCH ONLY, and at an interval of 100 it costs well under 0.1% of the
    step time amortised.

There is exactly one device-to-host copy per diagnostic iteration: the stats
accumulate into preallocated device tensors and are read back once, in `emit`.
"""

from typing import Dict, List, Tuple

import torch

from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import (
    param_is_not_gtp_duplicate,
    param_is_not_tensor_parallel_duplicate,
)
from megatron.core.transformer.module import param_is_not_shared

# Gain families, in a fixed order that also fixes their slot offsets in the
# packed buffer. `other_norm` is the catch-all so a model with norms this list
# does not know about still reports something rather than silently nothing.
GAIN_FAMILIES = ("input_norm", "pre_mlp_norm", "q_norm", "k_norm", "other_norm")
N_GAIN_STATS = 4  # mean, std, min, max

# Extra (non per-layer) grad-norm buckets, appended after the per-layer slots.
GRAD_EXTRAS = ("embedding", "output_layer", "final_norm", "other")

# Activation slots per layer: the input (qkv) norm and the pre-MLP norm.
ACT_SLOTS_PER_LAYER = 2
N_ACT_STATS = 3  # rms denominator, mean, non-finite count


def _classify_gain(name: str) -> str | None:
    """Map a 1-D norm-gain parameter name onto a family, or None if not a gain.

    TE fuses the input and pre-MLP norms into the following GEMM, so those gains
    are stored as `...linear_qkv.layer_norm_weight` / `...linear_fc1.layer_norm_weight`
    and NOT as a `layernorm.weight` of their own. Matching only on `layernorm`
    would miss both of the two biggest families.
    """
    if name.endswith("linear_qkv.layer_norm_weight"):
        return "input_norm"
    if name.endswith("linear_fc1.layer_norm_weight"):
        return "pre_mlp_norm"
    if name.endswith("q_layernorm.weight") or name.endswith("linear_q_layernorm.weight"):
        return "q_norm"
    if name.endswith("k_layernorm.weight") or name.endswith("linear_kv_layernorm.weight"):
        return "k_norm"
    if "layernorm" in name or "layer_norm_weight" in name:
        return "other_norm"
    return None


def _classify_grad_bucket(name: str) -> str | None:
    """Non-per-layer grad buckets. Returns None for anything that has a layer."""
    if "word_embeddings" in name or "position_embeddings" in name:
        return "embedding"
    if "output_layer" in name:
        return "output_layer"
    if "final_layernorm" in name:
        return "final_norm"
    return "other"


class TrainingDiagnostics:
    """Collects the per-iteration diagnostics and writes them to the loggers.

    Lifetime: constructed once in `setup_diagnostics`, `setup(model, optimizer)`
    once the model exists, then `observe_clip` every step, `collect` after the
    backward pass, and `emit` from `training_log`.
    """

    def __init__(self, args):
        self.args = args
        self.interval = int(getattr(args, "diagnostics_interval", 0) or 0)
        self.enabled = self.interval > 0
        self.num_layers = int(args.num_layers)

        self.want_gains = self.enabled and getattr(args, "diag_norm_gains", False)
        self.want_grads = self.enabled and getattr(args, "diag_layer_grad_norms", False)
        self.want_nonfinite = self.enabled and getattr(args, "diag_nonfinite", False)
        self.want_acts = self.enabled and getattr(args, "diag_activations", False)
        # Clip tracking is independent of `interval`: a STREAK can only be counted
        # by looking at every step. It is also free — it reads a scalar training
        # has already computed and does no collective of its own.
        self.want_clip = getattr(args, "diag_clip_events", False)

        # --- clip / grad-norm accumulators (host-side, every step) ------------
        self.clip_streak = 0
        self._clip_max_streak = 0
        self._clip_fired = 0
        self._clip_steps = 0
        self._clip_coeff_min = 1.0
        self._gn_sum = 0.0
        self._gn_max = 0.0
        self._gn_min = float("inf")

        # --- device buffers ---------------------------------------------------
        self._gain_buf: torch.Tensor | None = None
        self._grad_buf: torch.Tensor | None = None
        self._act_buf: torch.Tensor | None = None
        self._nonfinite_buf: torch.Tensor | None = None

        # --- static maps, built once in setup() -------------------------------
        # (param, gain_slot) for every norm gain this rank owns
        self._gain_params: List[Tuple[torch.nn.Parameter, int]] = []
        # (param, grad_slot, sub_range) for every gradient piece this rank should
        # count. sub_range is None when the whole parameter counts.
        self._grad_entries: List[Tuple[torch.nn.Parameter, int, Tuple[int, int] | None]] = []
        self._all_named_params: List[Tuple[str, torch.nn.Parameter]] = []
        self._hooks: List = []
        self._act_seen: torch.Tensor | None = None

        self._is_diag_iter = False
        self._have_gains = False
        self._have_grads = False
        self._have_acts = False
        self._have_nonfinite = False

        # Set in setup(), once mpu and the optimizer exist. See _build_grad_map.
        self._grad_replica_factor = 1.0

    # ------------------------------------------------------------------ setup
    def setup(self, model, optimizer=None):
        """Build the parameter->slot maps and install the activation hooks."""
        if not self.enabled:
            return
        if not isinstance(model, list):
            model = [model]

        dev = torch.cuda.current_device()
        n_gain_slots = len(GAIN_FAMILIES) * self.num_layers + 1  # +1 = final_layernorm
        n_grad_slots = self.num_layers + len(GRAD_EXTRAS)
        n_act_slots = ACT_SLOTS_PER_LAYER * self.num_layers + 1

        self._gain_buf = torch.zeros(n_gain_slots, N_GAIN_STATS, dtype=torch.float32, device=dev)
        self._grad_buf = torch.zeros(n_grad_slots, dtype=torch.float32, device=dev)
        self._act_buf = torch.zeros(n_act_slots, N_ACT_STATS, dtype=torch.float32, device=dev)
        self._act_seen = torch.zeros(n_act_slots, dtype=torch.float32, device=dev)
        self._nonfinite_buf = torch.zeros(2, dtype=torch.float32, device=dev)

        global_layer = self._build_global_layer_map(model)

        slot_of: Dict[int, int] = {}
        for chunk in model:
            for name, param in chunk.named_parameters():
                self._all_named_params.append((name, param))
                layer = global_layer.get(id(param))
                slot_of[id(param)] = self._grad_slot(name, layer)

                if self.want_gains and param.dim() == 1:
                    family = _classify_gain(name)
                    if family is not None:
                        self._gain_params.append((param, self._gain_slot(family, layer)))

        if self.want_grads:
            self._build_grad_map(model, optimizer, slot_of)

        if self.want_acts:
            self._install_activation_hooks(model)

    def _build_grad_map(self, model, optimizer, slot_of: Dict[int, int]):
        """Decide which piece of which gradient this rank is allowed to count.

        THE TRAP: with the distributed optimizer, DDP REDUCE-SCATTERs into the
        grad buffer, so `param.main_grad` is a full-size view of which only this
        rank's shard holds a reduced value — the rest is a partial result. Summing
        the whole tensor on every rank and all-reducing would add garbage.

        The distributed optimizer has already computed the intersection we need:
        `gbuf_ranges[...]["param_map"][param]["param"]` is the sub-range OF THE
        PARAMETER that this DP rank owns, and the ranges partition the parameter
        across DP. Summing squares over those disjoint pieces and all-reducing is
        therefore exact.

        Without the distributed optimizer DDP all-reduces instead, so every DP
        rank holds an identical full gradient and the same world sum over-counts
        by dp*cp. There we take the whole parameter and scale the result back.
        """
        tp_group = mpu.get_tensor_model_parallel_group()
        expert_tp_group = mpu.get_expert_tensor_parallel_group()

        def keep(param) -> bool:
            """The three dedup filters Megatron's own global grad norm applies.

            All three are load-bearing; dropping any one inflates the result by
            exactly the factor that axis replicates.

            `param_is_not_shared` is the one that is easy to miss and the one the
            smoke test caught: with TIED embeddings
            (`untie_embeddings_and_output_weights: False`) the word embedding
            lives on BOTH the first and last pipeline stage, so at PP>1 it was
            counted twice and `total_check` came out at sqrt(2) x `grad-norm`
            (measured 1.40597 vs sqrt(2)=1.41421 on job 1579619, a 130M model
            where that one tensor is 29M of the parameters). Invisible at PP=1.
            """
            return (
                param_is_not_shared(param)
                and param_is_not_tensor_parallel_duplicate(
                    param, tp_group=tp_group, expert_tp_group=expert_tp_group
                )
                and param_is_not_gtp_duplicate(param)
            )

        optimizers = getattr(optimizer, "chained_optimizers", None) or [optimizer]
        range_maps = []
        for opt in optimizers:
            gbuf_ranges = getattr(opt, "gbuf_ranges", None)
            if gbuf_ranges:
                range_maps.append(gbuf_ranges)

        if range_maps:
            for gbuf_ranges in range_maps:
                for gbuf_range_map in gbuf_ranges:
                    for _dtype, per_bucket in gbuf_range_map.items():
                        for bucket_range_map in per_bucket:
                            for param, rmap in bucket_range_map["param_map"].items():
                                slot = slot_of.get(id(param))
                                if slot is None or not keep(param):
                                    continue
                                rng = rmap["param"]
                                self._grad_entries.append((param, slot, (rng.start, rng.end)))
            self._grad_replica_factor = 1.0
        else:
            for _name, param in self._all_named_params:
                if keep(param):
                    self._grad_entries.append((param, slot_of[id(param)], None))
            self._grad_replica_factor = float(
                mpu.get_data_parallel_world_size(with_context_parallel=True)
            )

    def _build_global_layer_map(self, model) -> Dict[int, int]:
        """id(param) -> GLOBAL 0-based layer index, for every param inside a layer.

        Parsing `decoder.layers.<N>.` out of the parameter name is WRONG under
        pipeline parallelism: `named_parameters()` numbers the layers locally
        within each chunk, so every PP stage would report layers 0..k-1 and the
        per-stage slots would collide the moment they are reduced together.
        `TransformerLayer.layer_number` is assigned as
        `layer_number + get_transformer_layer_offset(...)`, i.e. it is global
        (and 1-based), so the module tree is the only correct source.
        """
        from megatron.core.transformer.transformer_layer import TransformerLayer

        out: Dict[int, int] = {}
        for chunk in model:
            for module in chunk.modules():
                if not isinstance(module, TransformerLayer):
                    continue
                li = min(max(int(module.layer_number) - 1, 0), self.num_layers - 1)
                for param in module.parameters(recurse=True):
                    out[id(param)] = li
        return out

    def _gain_slot(self, family: str, layer: int | None) -> int:
        if layer is None:
            return len(GAIN_FAMILIES) * self.num_layers  # the single final_layernorm slot
        fam = GAIN_FAMILIES.index(family)
        return fam * self.num_layers + min(layer, self.num_layers - 1)

    def _grad_slot(self, name: str, layer: int | None) -> int:
        if layer is not None:
            return min(layer, self.num_layers - 1)
        return self.num_layers + GRAD_EXTRAS.index(_classify_grad_bucket(name))

    def _install_activation_hooks(self, model):
        """Hook the modules that normalise the residual stream.

        The hook is a forward PRE-hook so the tensor it sees is the norm INPUT,
        which is what the RMSNorm denominator is computed from. With TE the norm
        is fused into `linear_qkv` / `linear_fc1`, so those modules are the hook
        sites; a non-fused build exposes `input_layernorm` / `pre_mlp_layernorm`
        instead and both spellings are handled.
        """
        from megatron.core.transformer.transformer_layer import TransformerLayer

        def make_hook(slot: int):
            def hook(module, inputs):
                # Two guards, cheapest first: not a diagnostic iteration at all,
                # or this slot already sampled on this iteration (we take the
                # first microbatch only, not all `get_num_microbatches()` of them).
                if not self._is_diag_iter or not inputs:
                    return
                x = inputs[0]
                if not torch.is_tensor(x) or x.numel() == 0:
                    return
                if bool(self._act_seen[slot].item()):
                    return
                self._act_seen[slot] = 1.0
                xf = x.detach().float()
                finite = torch.isfinite(xf)
                # Store the MEAN SQUARE, not its root. The buffer is averaged
                # across the ranks that share this layer before it is reported,
                # and mean-then-sqrt is the exact global RMS for equal-sized
                # shards, whereas sqrt-then-mean is not.
                self._act_buf[slot, 0] = xf.pow(2).mean()
                self._act_buf[slot, 1] = xf.mean()
                self._act_buf[slot, 2] = (~finite).sum().float()

            return hook

        def pick_norm_module(layer, fused_parent, fused_child, standalone):
            """The module whose INPUT is the thing the norm divides.

            Fused first: with TE the norm lives inside `linear_qkv` / `linear_fc1`
            and the standalone `input_layernorm` / `pre_mlp_layernorm` attribute
            is an IdentityOp. Hooking the identity would sample a tensor that no
            norm consumes, so the fused module has to win when both are present.
            """
            parent = getattr(layer, fused_parent, None)
            if parent is not None:
                inner = getattr(parent, fused_child, None)
                if inner is not None and hasattr(inner, "layer_norm_weight"):
                    return inner
            target = getattr(layer, standalone, None)
            if target is not None and any(p.dim() == 1 for p in target.parameters(recurse=False)):
                return target
            return None

        for chunk in model:
            for module in chunk.modules():
                if not isinstance(module, TransformerLayer):
                    continue
                li = int(module.layer_number) - 1  # layer_number is global and 1-based
                li = min(max(li, 0), self.num_layers - 1)
                for off, (parent, child, standalone) in enumerate(
                    (
                        ("self_attention", "linear_qkv", "input_layernorm"),
                        ("mlp", "linear_fc1", "pre_mlp_layernorm"),
                    )
                ):
                    target = pick_norm_module(module, parent, child, standalone)
                    if target is not None:
                        slot = ACT_SLOTS_PER_LAYER * li + off
                        self._hooks.append(target.register_forward_pre_hook(make_hook(slot)))

        for chunk in model:
            mod = chunk
            for _ in range(3):  # unwrap DDP / Float16Module / ...
                decoder = getattr(mod, "decoder", None)
                if decoder is not None:
                    break
                mod = getattr(mod, "module", None)
                if mod is None:
                    break
            else:
                decoder = None
            final_norm = getattr(decoder, "final_layernorm", None) if decoder is not None else None
            if final_norm is not None:
                slot = ACT_SLOTS_PER_LAYER * self.num_layers
                self._hooks.append(final_norm.register_forward_pre_hook(make_hook(slot)))
                break

    # ------------------------------------------------------------- per-step
    def begin_step(self, iteration: int):
        """Arm or disarm the activation hooks for this iteration."""
        self._is_diag_iter = self.enabled and (iteration % self.interval == 0)
        if self._is_diag_iter and self.want_acts and self._act_seen is not None:
            self._act_seen.zero_()
            self._act_buf.zero_()

    def observe_clip(self, grad_norm):
        """Runs EVERY step. Records whether the clip fired and how long a run of
        clipped steps we are in. `grad_norm` is the pre-clip total norm, i.e.
        exactly the denominator of clip_coeff = clip_grad / (total_norm + 1e-6).
        """
        if not self.want_clip or grad_norm is None:
            return
        gn = float(grad_norm)
        if gn != gn:  # NaN
            return
        clip_grad = float(getattr(self.args, "clip_grad", 0.0) or 0.0)
        coeff = 1.0
        if clip_grad > 0.0:
            coeff = min(1.0, clip_grad / (gn + 1.0e-6))
        fired = coeff < 1.0

        self.clip_streak = self.clip_streak + 1 if fired else 0
        self._clip_max_streak = max(self._clip_max_streak, self.clip_streak)
        self._clip_fired += int(fired)
        self._clip_steps += 1
        self._clip_coeff_min = min(self._clip_coeff_min, coeff)
        self._gn_sum += gn
        self._gn_max = max(self._gn_max, gn)
        self._gn_min = min(self._gn_min, gn)

    def collect(self, iteration: int):
        """Gather gains, per-layer grad norms and the non-finite scan.

        Called AFTER the backward pass and BEFORE `optimizer.step()`, because
        the clip scales the gradients in place and the interesting quantity is
        the gradient the optimizer was handed, not the clipped one.
        """
        if not self._is_diag_iter:
            return
        if self.want_gains:
            self._collect_gains()
        if self.want_grads:
            self._collect_grad_norms()
        if self.want_nonfinite:
            self._collect_nonfinite()

    def _collect_gains(self):
        self._gain_buf.zero_()
        for param, slot in self._gain_params:
            v = param.detach().float()
            self._gain_buf[slot, 0] = v.mean()
            self._gain_buf[slot, 1] = v.std() if v.numel() > 1 else torch.zeros((), device=v.device)
            self._gain_buf[slot, 2] = v.min()
            self._gain_buf[slot, 3] = v.max()
        # Norm gains are REPLICATED across TP, DP and CP and PARTITIONED across
        # PP, so every rank of a pipeline stage already holds identical values
        # for that stage's layers and the stages' slots are disjoint. Reducing
        # over the pipeline group alone is therefore exact and needs no
        # duplicate filtering — reducing over the world would multiply every
        # slot by tp*dp*cp.
        torch.distributed.all_reduce(
            self._gain_buf,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_pipeline_model_parallel_group(),
        )
        self._have_gains = True

    def _grad_of(self, param):
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        return grad

    def _collect_grad_norms(self):
        self._grad_buf.zero_()
        for param, slot, rng in self._grad_entries:
            grad = self._grad_of(param)
            if grad is None:
                continue
            grad = grad.detach()
            if rng is not None:
                grad = grad.reshape(-1)[rng[0] : rng[1]]
            self._grad_buf[slot] += grad.float().pow(2).sum()
        if self._grad_replica_factor != 1.0:
            self._grad_buf.div_(self._grad_replica_factor)
        # Sums of squares from disjoint pieces (TP shards, DP shards under the
        # distributed optimizer, PP stages) add, so ONE world all-reduce gives
        # the true per-layer norm^2 on every rank.
        torch.distributed.all_reduce(self._grad_buf, op=torch.distributed.ReduceOp.SUM)
        self._have_grads = True

    def _collect_nonfinite(self):
        self._nonfinite_buf.zero_()
        offenders = []
        for name, param in self._all_named_params:
            bad_w = (~torch.isfinite(param.detach())).sum()
            self._nonfinite_buf[0] += bad_w
            grad = self._grad_of(param)
            if grad is not None:
                self._nonfinite_buf[1] += (~torch.isfinite(grad.detach())).sum()
        # One sync, and only to decide whether to pay for the second pass that
        # names the offender. On a healthy run this is the only cost.
        if float(self._nonfinite_buf.sum().item()) > 0.0:
            for name, param in self._all_named_params:
                if not torch.isfinite(param.detach()).all():
                    offenders.append(f"weight:{name}")
                grad = self._grad_of(param)
                if grad is not None and not torch.isfinite(grad.detach()).all():
                    offenders.append(f"grad:{name}")
                if len(offenders) >= 8:
                    break
            rank = torch.distributed.get_rank()
            print(f"[diagnostics] rank {rank} NON-FINITE in: {', '.join(offenders)}", flush=True)
        # MAX, not SUM: parameters are replicated across TP/DP/CP, so a sum would
        # report a count that depends on the parallel layout. The worst single
        # rank is the number that means something.
        torch.distributed.all_reduce(self._nonfinite_buf, op=torch.distributed.ReduceOp.MAX)
        self._have_nonfinite = True

    # ------------------------------------------------------------------ emit
    def emit(self, iteration: int, writer, wandb_writer) -> Dict[str, float]:
        """Write everything collected this iteration and reset the accumulators.

        Returns the small dict of headline scalars so the caller can also put
        them on the iteration line.
        """
        # Gate on what was actually COLLECTED, not on `_is_diag_iter`.
        # --diagnostics-interval and --tensorboard-log-interval are independent,
        # so data gathered on iteration N is emitted at the next logging
        # iteration, which may be a later one. Testing _is_diag_iter here would
        # silently drop it whenever the two intervals are not aligned.
        if not (
            self._have_gains
            or self._have_grads
            or self._have_acts
            or self._have_nonfinite
            or (self.want_clip and self._clip_steps > 0)
        ):
            return {}

        logs: Dict[str, float] = {}

        if self.want_clip and self._clip_steps > 0:
            logs["diag/clip/fired_frac"] = self._clip_fired / self._clip_steps
            logs["diag/clip/streak"] = float(self.clip_streak)
            logs["diag/clip/max_streak"] = float(self._clip_max_streak)
            logs["diag/clip/coeff_min"] = self._clip_coeff_min
            # The clip denominator: mean / max / min of the pre-clip total norm
            # over the interval, which a single sampled `grad-norm` cannot show.
            logs["diag/grad_norm/mean"] = self._gn_sum / self._clip_steps
            logs["diag/grad_norm/max"] = self._gn_max
            logs["diag/grad_norm/min"] = self._gn_min
            self._clip_fired = 0
            self._clip_steps = 0
            self._clip_max_streak = 0
            self._clip_coeff_min = 1.0
            self._gn_sum = 0.0
            self._gn_max = 0.0
            self._gn_min = float("inf")

        if self._have_gains:
            g = self._gain_buf.cpu()
            for fi, family in enumerate(GAIN_FAMILIES):
                block = g[fi * self.num_layers : (fi + 1) * self.num_layers]
                if float(block.abs().sum()) == 0.0:
                    continue  # family absent from this model
                for li in range(self.num_layers):
                    p = f"diag/gain/{family}/layer_{li:02d}"
                    logs[f"{p}/mean"] = float(block[li, 0])
                    logs[f"{p}/std"] = float(block[li, 1])
                    logs[f"{p}/min"] = float(block[li, 2])
                    logs[f"{p}/max"] = float(block[li, 3])
                # Alarm-able summaries: the extremes over layers, and WHICH layer.
                logs[f"diag/gain/{family}/min_over_layers"] = float(block[:, 2].min())
                logs[f"diag/gain/{family}/max_over_layers"] = float(block[:, 3].max())
                logs[f"diag/gain/{family}/argmax_layer"] = float(block[:, 3].argmax())
                logs[f"diag/gain/{family}/argmin_layer"] = float(block[:, 2].argmin())
            final = g[len(GAIN_FAMILIES) * self.num_layers]
            if float(final.abs().sum()) != 0.0:
                logs["diag/gain/final_norm/mean"] = float(final[0])
                logs["diag/gain/final_norm/std"] = float(final[1])
                logs["diag/gain/final_norm/min"] = float(final[2])
                logs["diag/gain/final_norm/max"] = float(final[3])
            self._have_gains = False

        if self._have_grads:
            n = self._grad_buf.clamp_min(0).sqrt().cpu()
            for li in range(self.num_layers):
                logs[f"diag/grad_norm/layer_{li:02d}"] = float(n[li])
            for ei, extra in enumerate(GRAD_EXTRAS):
                v = float(n[self.num_layers + ei])
                if v != 0.0:
                    logs[f"diag/grad_norm/{extra}"] = v
            layers = n[: self.num_layers]
            nz = layers[layers > 0]
            if nz.numel() > 0:
                # early-vs-late asymmetry as one number, so it is alarm-able
                logs["diag/grad_norm/layer_max"] = float(nz.max())
                logs["diag/grad_norm/layer_min"] = float(nz.min())
                logs["diag/grad_norm/layer_ratio"] = float(nz.max() / nz.min())
                logs["diag/grad_norm/argmax_layer"] = float(layers.argmax())
            # SELF-CHECK. The per-layer norms are sums of squares over disjoint
            # pieces, so recombining them must reproduce the global norm Megatron
            # computes independently and already logs as `grad-norm`. If the
            # TP-duplicate filter or the distributed-optimizer shard ranges were
            # wrong, this is where it shows: the ratio drifts off 1.0 by exactly
            # the factor that was double-counted or dropped. Cheap, and it turns a
            # silent numerical bug into a visible one.
            logs["diag/grad_norm/total_check"] = float(self._grad_buf.sum().clamp_min(0).sqrt())
            self._have_grads = False

        if self._have_acts:
            a = self._act_buf.cpu()
            for li in range(self.num_layers):
                for off, tag in ((0, "input_norm"), (1, "pre_mlp_norm")):
                    row = a[ACT_SLOTS_PER_LAYER * li + off]
                    if float(row.abs().sum()) == 0.0:
                        continue
                    p = f"diag/act/{tag}/layer_{li:02d}"
                    # col 0 holds the MEAN SQUARE; the RMSNorm denominator is its root.
                    logs[f"{p}/rms_denom"] = float(row[0].clamp_min(0).sqrt())
                    logs[f"{p}/mean"] = float(row[1])
            final = a[ACT_SLOTS_PER_LAYER * self.num_layers]
            if float(final.abs().sum()) != 0.0:
                logs["diag/act/final_norm/rms_denom"] = float(final[0].clamp_min(0).sqrt())
                logs["diag/act/final_norm/mean"] = float(final[1])
            logs["diag/act/nonfinite"] = float(a[:, 2].sum())
            self._have_acts = False

        if self._have_nonfinite:
            logs["diag/nonfinite/weight"] = float(self._nonfinite_buf[0].item())
            logs["diag/nonfinite/grad"] = float(self._nonfinite_buf[1].item())
            self._have_nonfinite = False

        if not logs:
            return {}
        if writer is not None:
            for key, value in logs.items():
                writer.add_scalar(key, value, iteration)
        if wandb_writer is not None:
            wandb_writer.log(logs, iteration)
        return {
            k: logs[k]
            for k in ("diag/clip/max_streak", "diag/nonfinite/weight", "diag/nonfinite/grad")
            if k in logs
        }

    def finish_activation_pass(self):
        """Reduce the activation buffer across ranks and mark it ready.

        WITHOUT THIS REDUCTION THE ACTIVATION STATS COVER ONLY ONE PIPELINE
        STAGE. A hook only fires on the rank that owns the layer, and the
        tensorboard/wandb writers live on the LAST rank (global_vars.py gates
        them on `rank == world_size - 1`), so at PP>1 the writer sees only its
        own stage. Measured on job 1583194 (PP=2, 18 layers): layers 9-17 were
        reported and 0-8 were silently absent. At the flagship's PP=4 that would
        be ~16 of 64 layers, and the surviving numbers look perfectly healthy —
        which is the dangerous part.

        Slots are disjoint across pipeline stages but SHARED by the tp*dp*cp
        ranks within a stage, and those ranks hold genuinely different data (a
        different microbatch under DP, a different sequence slice under
        sequence-parallel). So the mean square and mean are AVERAGED over them,
        while the non-finite count is SUMMED — a count is a total, not an
        average, and halving it would hide a fault on one rank.
        """
        if not (self._is_diag_iter and self.want_acts):
            return
        torch.distributed.all_reduce(self._act_buf, op=torch.distributed.ReduceOp.SUM)
        contributors = max(
            1, torch.distributed.get_world_size() // mpu.get_pipeline_model_parallel_world_size()
        )
        if contributors > 1:
            self._act_buf[:, 0:2] /= contributors
        self._have_acts = True


_GLOBAL_DIAGNOSTICS: TrainingDiagnostics | None = None


def setup_diagnostics(args) -> TrainingDiagnostics:
    """Construct the global diagnostics object (cheap; does nothing when off)."""
    global _GLOBAL_DIAGNOSTICS
    _GLOBAL_DIAGNOSTICS = TrainingDiagnostics(args)
    return _GLOBAL_DIAGNOSTICS


def get_diagnostics() -> TrainingDiagnostics | None:
    """The global diagnostics object, or None before `setup_diagnostics`."""
    return _GLOBAL_DIAGNOSTICS
