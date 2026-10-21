# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Weight-only quantization of a loaded model, for evaluation (QUANT_EVAL_BITS=4).

Fake quantization in place, after the checkpoint is loaded: each matrix weight is rounded to
``bits`` bits per value, asymmetric (min / max) per group of QUANT_EVAL_GROUP (default 32)
consecutive values along the input dimension -- close to llama.cpp's Q4_K (32-value sub-blocks
with their own scale and minimum). As llama.cpp's Q4_K_M keeps them, the token embeddings, the
output layer, the MoE routers, norms, biases and the small gate / mixing vectors stay unquantized.
Use with --skip-train (evaluation only), never for training.
"""

import os
import re

import torch

# matrices that stay in full precision (llama.cpp keeps routers in f32 and the embeddings / output
# in higher precision); the residual mixers (mHC / LDM) and 1-D tensors are skipped by shape / name
_SKIP = re.compile(r"(embedding|output_layer|router|\.gate\.weight|norm|mhc|residual_mixer|depth_memory|"
                   r"attn_res|A_log|dt_bias|bias$)")


@torch.no_grad()
def fake_quantize_(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Round ``w`` in place to ``bits``-bit asymmetric groups of ``group`` values along its last dim."""
    shape = w.shape
    if shape[-1] % group:
        return w
    x = w.float().reshape(-1, group)
    lo, hi = x.amin(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp_min(1e-12) / levels
    q = ((x - lo) / scale).round().clamp_(0, levels)
    w.copy_((q * scale + lo).reshape(shape).to(w.dtype))
    return w


def maybe_quantize_for_eval(model) -> None:
    """Fake-quantize the model's matrix weights if QUANT_EVAL_BITS is set; logs what it did."""
    bits = int(os.environ.get("QUANT_EVAL_BITS", "0"))
    if bits <= 0:
        return
    group = int(os.environ.get("QUANT_EVAL_GROUP", "32"))
    chunks = model if isinstance(model, list) else [model]
    n_q = n_skip = 0
    numel_q = 0
    for chunk in chunks:
        for name, p in chunk.named_parameters():
            if p.dim() < 2 or _SKIP.search(name) or p.shape[-1] < group:
                n_skip += 1
                continue
            fake_quantize_(p.data, bits, group)
            n_q += 1
            numel_q += p.numel()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[quant_eval] {bits}-bit, group {group}: quantized {n_q} matrices "
              f"({numel_q / 1e6:.1f}M values on rank 0), kept {n_skip} tensors", flush=True)
