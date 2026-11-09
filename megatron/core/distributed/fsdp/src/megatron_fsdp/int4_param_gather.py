"""INT4 parameter all-gather for Megatron-FSDP (``fsdp_fp8_param_gather`` with
``fsdp_param_gather_format="int4"``): 4-bit quantization-aware training of the gathered weights.

As fp8_param_gather.py, but a rank's bf16 shard becomes symmetric INT4 with one fp32 scale per 128
contiguous elements (scale = amax / 7, values round(x / scale) in [-7, 7]): ~0.53 instead of 2 bytes
per element on the wire (FP8: ~1.03). Every rank dequantizes the same bytes, so the forward of every
rank runs on the same INT4 weights; the GEMMs stay bf16, master weights / gradients / optimizer fp32,
the gradient is straight-through for the rounding -- quantization-aware training of exactly the
format the consumer W4A16 kernels serve (cute_kernels lowbit_experts: int4, groups of 128).

Packed layout of one rank's shard of ``n`` elements (``nb = ceil(n / 128)`` blocks): ``ceil(n / 2)``
bytes of two 4-bit values each (element 2 i in the low nibble, 2 i + 1 in the high one, stored as
q + 8), zero-padded to a multiple of 16, then ``nb`` fp32 scales, the total padded to a multiple of 16.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from .fp8_param_gather import _StreamEvent

BLOCK = 128
QMAX = 7.0


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def packed_layout(n: int):
    """(blocks, scale byte offset, total bytes) of one rank's packed shard of n elements."""
    nb = (n + BLOCK - 1) // BLOCK
    data = _align((n + 1) // 2, 16)
    total = _align(data + 4 * nb, 16)
    return nb, data, total


if HAVE_TRITON:

    @triton.jit
    def _dequantize_unpack_kernel(g_ptr, gs_ptr, out_ptr, n, rank_bytes, scale_off_words, BLOCK_SIZE: tl.constexpr):
        # one program per (128-element block, rank)
        b = tl.program_id(0)
        r = tl.program_id(1)
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        base = r.to(tl.int64) * rank_bytes
        byte = tl.load(g_ptr + base + offs // 2, mask=mask, other=8).to(tl.int32)
        nib = tl.where(offs % 2 == 0, byte & 15, (byte >> 4) & 15)
        scale = tl.load(gs_ptr + base // 4 + scale_off_words + b)
        tl.store(out_ptr + r.to(tl.int64) * n + offs, ((nib - 8).to(tl.float32) * scale).to(tl.bfloat16),
                 mask=mask)


def _quantize(x: torch.Tensor):
    """bf16 / fp32 [n] -> (codes q + 8 in [1, 15] as uint8 [nb * 128], fp32 scales [nb])."""
    n = x.numel()
    nb = (n + BLOCK - 1) // BLOCK
    xp = torch.zeros(nb * BLOCK, dtype=torch.float32, device=x.device)
    xp[:n] = x.float()
    xb = xp.view(nb, BLOCK)
    amax = xb.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.round(xb / scale).clamp_(-QMAX, QMAX).to(torch.int32) + 8
    return q.view(-1).to(torch.uint8), scale.view(-1)


@torch.no_grad()
def quantize_pack(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 shard [n] -> packed uint8 [packed_layout(n)[2]] (4-bit codes + fp32 scales)."""
    assert x.dtype == torch.bfloat16 and x.dim() == 1
    n = x.numel()
    nb, data, total = packed_layout(n)
    if out is None:
        out = torch.empty(total, dtype=torch.uint8, device=x.device)
    assert out.numel() == total and out.dtype == torch.uint8
    if nb == 0:
        return out
    codes, scale = _quantize(x)
    pairs = codes[: _align(n, 2)].view(-1, 2)  # padding codes are 8 (= 0)
    nbytes = pairs.shape[0]
    out[:nbytes].copy_(pairs[:, 0] | (pairs[:, 1] << 4))
    if data > nbytes:
        out[nbytes:data].zero_()
    out[data : data + 4 * nb].view(torch.float32).copy_(scale)
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
    """What one shard looks like after the INT4 round trip (plain torch, for tests and exports)."""
    n = x.numel()
    codes, scale = _quantize(x)
    v = (codes.view(-1, BLOCK).float() - 8) * scale[:, None]
    return v.view(-1)[:n].to(torch.bfloat16)


@torch.no_grad()
def int4_all_gather_into_tensor(output_tensor: torch.Tensor, input_tensor: torch.Tensor, group) -> _StreamEvent:
    """``all_gather_into_tensor(output_tensor, input_tensor, group)`` for bf16, through INT4 (as
    fp8_all_gather_into_tensor)."""
    world = torch.distributed.get_world_size(group)
    n = input_tensor.numel()
    assert output_tensor.numel() == world * n, (output_tensor.numel(), world, n)
    out_lo, out_hi = output_tensor.data_ptr(), output_tensor.data_ptr() + 2 * output_tensor.numel()
    assert not (out_lo <= input_tensor.data_ptr() < out_hi), "int4 param gather: input shard aliases the bucket"
    _, _, total = packed_layout(n)
    packed = quantize_pack(input_tensor)
    gathered = torch.empty(world * total, dtype=torch.uint8, device=input_tensor.device)
    work = torch.distributed.all_gather_into_tensor(gathered, packed, group=group, async_op=True)
    work.wait()
    dequantize_unpack(gathered, n, world, output_tensor.view(-1))
    event = torch.cuda.Event()
    event.record()
    return _StreamEvent(event)


def max_relative_error_bound() -> float:
    """Round to nearest on a grid of amax / 7: |error| <= amax / 14 of the block maximum."""
    return 0.5 / QMAX * (1 + 1e-6)


__all__ = [
    "BLOCK",
    "QMAX",
    "packed_layout",
    "quantize_pack",
    "dequantize_unpack",
    "quantize_dequantize_reference",
    "int4_all_gather_into_tensor",
    "max_relative_error_bound",
    "HAVE_TRITON",
]
