# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""FP8 parameter all-gather for Megatron-FSDP (``fsdp_fp8_param_gather``).

ZeRO-3 all-gathers every FSDP unit's bf16 weights before its forward and again before its
recompute / backward. With many routed experts that traffic dominates the step
(docs/ambitious_size64.md in oellm-autoexp: at 64 nodes, 384 experts, ~17 s per step of exposed
AllGather). Here a rank's bf16 shard is quantized to FP8 E4M3 with one fp32 scale per 128
contiguous elements (amax / 448, as DeepSeek-V3's 1 x 128 tiles), the packed bytes are
all-gathered (~1.03 instead of 2 bytes per element), and every rank dequantizes the gathered
shards into the usual unsharded bf16 bucket.

Only the communicated copy is quantized: the GEMMs run in bf16 on the dequantized weights, and
the master weights, gradients and optimizer stay as they are (fp32). Every rank dequantizes the
same bytes, so the unsharded weights are identical on all ranks (the local shard included).
The gradient is the one of the quantized forward (straight-through for the weight rounding), as
in FP8 training.

Packed layout of one rank's shard of ``n`` elements (``nb = ceil(n / 128)`` blocks): ``n`` fp8
bytes, zero-padded to a multiple of 16, then ``nb`` fp32 scales, the total padded to a multiple
of 16 bytes.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

BLOCK = 128
FP8_MAX = 448.0  # E4M3 (float8_e4m3fn) largest finite value


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def packed_layout(n: int):
    """(data bytes, scale byte offset, total bytes) of one rank's packed shard of n elements."""
    nb = (n + BLOCK - 1) // BLOCK
    data = _align(n, 16)
    total = _align(data + 4 * nb, 16)
    return nb, data, total


if HAVE_TRITON:

    @triton.jit
    def _dequantize_unpack_kernel(
        g_ptr, gs_ptr, out_ptr, n, rank_bytes, scale_off_words, BLOCK_SIZE: tl.constexpr
    ):
        # program (block b of rank r): rank r's packed bytes start at r * rank_bytes, its scales
        # at (r * rank_bytes) / 4 + scale_off_words in the fp32 view. E4M3 is decoded from its
        # bits (no fp8 dtype: works on any architecture and Triton version): sign s, exponent e
        # (bias 7), mantissa m (3 bits); normal (1 + m / 8) * 2^(e - 7) is the fp32 with exponent
        # field e - 7 + 127 and mantissa m << 20; subnormal (e = 0) is m * 2^-9. No NaN codes
        # occur (the quantizer clamps to +-448).
        b = tl.program_id(0)
        r = tl.program_id(1)
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        base = r.to(tl.int64) * rank_bytes
        q = tl.load(g_ptr + base + offs, mask=mask, other=0).to(tl.int32)
        sign = (q >> 7) & 1
        e = (q >> 3) & 15
        m = q & 7
        normal = ((sign << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = m.to(tl.float32) * 0.001953125
        sub = tl.where(sign == 1, -sub, sub)
        v = tl.where(e == 0, sub, normal)
        scale = tl.load(gs_ptr + base // 4 + scale_off_words + b)
        tl.store(out_ptr + r.to(tl.int64) * n + offs, (v * scale).to(tl.bfloat16), mask=mask)


@torch.no_grad()
def quantize_pack(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 shard [n] -> packed uint8 [packed_layout(n)[2]] (fp8 bytes + fp32 scales).

    Plain torch (the shard is small: one rank's 1 / world of a bucket)."""
    assert x.dtype == torch.bfloat16 and x.dim() == 1
    n = x.numel()
    nb, data, total = packed_layout(n)
    if out is None:
        out = torch.empty(total, dtype=torch.uint8, device=x.device)
    assert out.numel() == total and out.dtype == torch.uint8
    if nb == 0:
        return out
    xp = torch.zeros(nb * BLOCK, dtype=torch.float32, device=x.device)
    xp[:n] = x
    xb = xp.view(nb, BLOCK)
    amax = xb.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (xb / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    out[:n].copy_(q.view(-1)[:n])
    if data > n:
        out[n:data].zero_()
    out[data : data + 4 * nb].view(torch.float32).copy_(scale.view(-1))
    if total > data + 4 * nb:
        out[data + 4 * nb :].zero_()
    return out


def dequantize_unpack(gathered: torch.Tensor, n: int, world: int, out: torch.Tensor) -> None:
    """Packed uint8 [world * total] (all ranks' quantize_pack outputs) -> bf16 out [world * n]."""
    nb, data, total = packed_layout(n)
    assert gathered.numel() == world * total and gathered.dtype == torch.uint8
    assert out.dtype == torch.bfloat16 and out.numel() == world * n and out.is_contiguous()
    if nb == 0:
        return
    _dequantize_unpack_kernel[(nb, world)](
        gathered, gathered.view(torch.float32), out, n, total, data // 4, BLOCK_SIZE=BLOCK
    )


def quantize_dequantize_reference(x: torch.Tensor) -> torch.Tensor:
    """What one shard looks like after the FP8 round trip (plain torch, for tests)."""
    n = x.numel()
    nb = (n + BLOCK - 1) // BLOCK
    xp = torch.zeros(nb * BLOCK, dtype=torch.float32, device=x.device)
    xp[:n] = x.float()
    xb = xp.view(nb, BLOCK)
    amax = xb.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).view(-1)[:n].to(torch.bfloat16)


class _StreamEvent:
    """``wait()`` makes the current stream wait for a recorded CUDA event (the interface
    AllGatherPipeline.wait_bucket_ready expects of an all-gather handle)."""

    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def wait(self):
        torch.cuda.current_stream().wait_event(self.event)


@torch.no_grad()
def fp8_all_gather_into_tensor(
    output_tensor: torch.Tensor, input_tensor: torch.Tensor, group
) -> _StreamEvent:
    """``all_gather_into_tensor(output_tensor, input_tensor, group)`` for bf16, through FP8.

    Runs on the current stream: quantize, all-gather the packed shards, wait for them on this
    stream, dequantize into ``output_tensor``. Returns a handle whose ``wait()`` orders the
    caller's stream after the dequantization.
    """
    world = torch.distributed.get_world_size(group)
    n = input_tensor.numel()
    assert output_tensor.numel() == world * n, (output_tensor.numel(), world, n)
    # the input (the persistent local shard) must not alias the output we overwrite
    out_lo, out_hi = output_tensor.data_ptr(), output_tensor.data_ptr() + 2 * output_tensor.numel()
    in_lo = input_tensor.data_ptr()
    assert not (out_lo <= in_lo < out_hi), "fp8 param gather: input shard aliases the bucket"
    _, _, total = packed_layout(n)
    packed = quantize_pack(input_tensor)
    gathered = torch.empty(world * total, dtype=torch.uint8, device=input_tensor.device)
    work = torch.distributed.all_gather_into_tensor(gathered, packed, group=group, async_op=True)
    work.wait()  # orders the current stream after the collective (no host block)
    dequantize_unpack(gathered, n, world, output_tensor.view(-1))
    event = torch.cuda.Event()
    event.record()
    return _StreamEvent(event)


def max_relative_error_bound() -> float:
    """E4M3 has 3 mantissa bits: round-to-nearest relative error <= 2^-4 for values near the
    block maximum (smaller values in a block lose more, down to the subnormal floor)."""
    return 2.0 ** -4 * (1 + 1e-6)


__all__ = [
    "BLOCK",
    "FP8_MAX",
    "packed_layout",
    "quantize_pack",
    "dequantize_unpack",
    "quantize_dequantize_reference",
    "fp8_all_gather_into_tensor",
    "max_relative_error_bound",
    "HAVE_TRITON",
]
