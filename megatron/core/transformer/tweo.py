"""Opt-in post-residual TWEO, initially restricted to dense CP1 training.

The custom identity retains BF16 activations, computes the quartic derivative
in FP32 during backward, and avoids retaining a second FP32 activation graph.
"""
import math

import torch


class TWEOState:
    """Per-process schedule scale and detached diagnostics for one training step."""

    scale = None
    moments = {}
    collect = True

    @classmethod
    def reset(cls, collect=True):
        cls.collect = collect
        cls.scale = None
        cls.moments = {}


def coefficient(target: float, step: int, start: int, warmup: int) -> float:
    """Return an absolute-update ramp; resumption never restarts the ramp."""
    if not math.isfinite(target) or target < 0 or min(step, start, warmup) < 0:
        raise ValueError("Invalid TWEO schedule")
    if step < start:
        return 0.0
    return target if warmup == 0 else target * min(1.0, (step - start) / warmup)


def validate(config) -> None:
    """Fail closed for configurations not covered by this candidate's derivation."""
    if config.tweo_loss_coeff == 0:
        return
    if getattr(config, "tweo_implementation", "fused") not in ("reference", "fused") or getattr(config, "tweo_diagnostics_interval", 100) < 0:
        raise ValueError("Invalid TWEO implementation or diagnostics interval")
    if not math.isfinite(config.tweo_loss_coeff) or config.tweo_loss_coeff < 0:
        raise ValueError("TWEO coefficient must be finite and nonnegative")
    if not math.isfinite(config.tweo_tau) or config.tweo_tau <= 0:
        raise ValueError("TWEO tau must be finite and positive")
    if config.tweo_start_step < 0 or config.tweo_warmup_steps < 0:
        raise ValueError("TWEO schedule coordinates must be nonnegative")
    if config.context_parallel_size != 1 or config.calculate_per_token_loss:
        raise ValueError("TWEO candidate currently requires CP1 and microbatch-mean loss")
    if config.num_moe_experts is not None or config.mtp_num_layers:
        raise ValueError("TWEO candidate currently supports dense GPT without MTP")
    if getattr(config, "cuda_graph_impl", "none") not in (None, "none"):
        raise ValueError("TWEO candidate CUDA-graph integration is not validated")


class _TWEOIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coeff, tau, layers, sequence_shards):
        ctx.save_for_backward(value)
        ctx.coeff = coeff
        ctx.tau = tau + 1e-6
        ctx.normalizer = layers * value.numel() * sequence_shards
        return value

    @staticmethod
    def backward(ctx, grad_output):
        if TWEOState.scale is None:
            raise RuntimeError("TWEO backward scale was not set by the pipeline schedule")
        (value,) = ctx.saved_tensors
        # Do not precombine lambda/N: at measured B1 activation scales a
        # useful tiny lambda can make that constant subnormal and flush to zero.
        dtype = torch.float64 if value.dtype == torch.float64 or ctx.coeff < 2**-126 else torch.float32
        extra = (value.to(dtype) / ctx.tau).pow(3)
        extra = extra * ctx.coeff * (4.0 / ctx.normalizer) / ctx.tau
        extra = extra * TWEOState.scale
        return (grad_output.to(dtype) + extra).to(value.dtype), None, None, None, None


class _TWEOFusedIdentity(_TWEOIdentity):
    @staticmethod
    def backward(ctx, grad_output):
        if TWEOState.scale is None:
            raise RuntimeError("TWEO backward scale was not set by the pipeline schedule")
        (value,) = ctx.saved_tensors
        if (value.is_cuda and value.is_contiguous() and grad_output.is_contiguous()
                and value.dtype in (torch.float32, torch.bfloat16)
                and ctx.coeff >= 2**-126):
            from .tweo_kernels import adjoint
            result = adjoint(value, grad_output, TWEOState.scale, ctx.coeff, ctx.tau, ctx.normalizer)
            return result, None, None, None, None
        # Preserve the validated FP64/tiny-coefficient/noncontiguous/CPU paths.
        return _TWEOIdentity.backward(ctx, grad_output)


def attach(value: torch.Tensor, config, layer_number: int) -> torch.Tensor:
    """Attach the paper's block-output penalty, without changing forward values."""
    if config.tweo_loss_coeff == 0 or not torch.is_grad_enabled():
        return value
    if not hasattr(config, "_tweo_current_coeff"):
        raise RuntimeError("TWEO requires an explicit training-step coefficient")
    coeff = config._tweo_current_coeff
    if coeff == 0:
        return value
    shards = config.tensor_model_parallel_size if config.sequence_parallel else 1
    fused = getattr(config, "tweo_implementation", "fused") == "fused"
    if TWEOState.collect:
        if fused and value.is_cuda and value.is_contiguous() and value.numel() > 0:
            from .tweo_kernels import accumulate_moments
            with torch.no_grad():
                if layer_number not in TWEOState.moments:
                    TWEOState.moments[layer_number] = torch.zeros(2, dtype=torch.float64, device=value.device)
                accumulate_moments(value, config.tweo_tau + 1e-6, TWEOState.moments[layer_number])
        else:
            with torch.no_grad():
                # Scalar FP64 accumulation avoids overflow when DP ranks combine the
                # enormous measured residual fourth moments. Chunk the temporary.
                fourth = torch.zeros((), dtype=torch.float64, device=value.device)
                for chunk in value.detach().reshape(-1).split(1024 * 1024):
                    scaled = chunk.double() / (config.tweo_tau + 1e-6)
                    fourth.add_(scaled.square().square().sum())
                record = torch.stack((fourth, fourth.new_tensor(value.numel())))
                if layer_number in TWEOState.moments:
                    TWEOState.moments[layer_number].add_(record)
                else:
                    TWEOState.moments[layer_number] = record
    function = _TWEOFusedIdentity if fused else _TWEOIdentity
    return function.apply(value, coeff, config.tweo_tau, config.num_layers, shards)
