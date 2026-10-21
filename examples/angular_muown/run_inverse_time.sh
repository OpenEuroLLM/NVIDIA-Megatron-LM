#!/usr/bin/env bash
# Tiny single-node mock-data run; requires an already provisioned Megatron environment.
# OUT_DIR=/absolute/local/scratch/run PYTHON=/path/to/python bash "$0" 0.04 [Megatron args...]
# C=0.04 is a smoke-test input, not a recommendation for the 40B-token experiment.
# Select angular_muown_simplified; scaling defaults to shape_scaling, with extra factor 1.
# Override the mode with --angular-muown-scale-mode spectral (or unit_rms_norm).
# Angular C/t uses updates since warmup at fixed global batch size, independently
# of Adam LR; it inherits LR warmup/horizon and defaults to a final 20% cooldown
# ending at 5% of its branch rate.
# For a resume check, run with --exit-interval 6, then repeat the same command
# without that flag and add --load "$OUT_DIR/checkpoints" --exit-on-missing-checkpoint.
# Keep the full 12-update horizon and batch size unchanged; use another OUT_DIR
# for the uninterrupted comparison. GPU training and ROCm remain unvalidated.
# GPUS defaults to 1. DRY_RUN=1 prints the command without creating files/running Python.
set -euo pipefail
c=${1:?Usage: OUT_DIR=/absolute/local/scratch/run bash run_inverse_time.sh C [Megatron args...]}
shift
: "${OUT_DIR:?Set OUT_DIR to an absolute local scratch directory outside the checkout}"
[[ $OUT_DIR == /* ]] || { echo "OUT_DIR must be absolute." >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
out=$(realpath -m -- "$OUT_DIR")
case "$out/" in "$repo/"*) echo "OUT_DIR must be outside the checkout." >&2; exit 2 ;; esac

export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$out/cache" TORCH_HOME="$out/cache/torch"
export TORCH_EXTENSIONS_DIR="$out/cache/extensions" TRITON_CACHE_DIR="$out/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$out/cache/inductor" CUDA_CACHE_PATH="$out/cache/cuda"
export UV_CACHE_DIR="$out/cache/uv" TMPDIR="$out/tmp"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

command=(uv run --no-project --no-sync --offline --python "${PYTHON:-python3}" python
    -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${GPUS:-1}"
    "$repo/pretrain_gpt.py"
    --num-layers 2 --hidden-size 128 --num-attention-heads 4 --ffn-hidden-size 256
    --seq-length 64 --max-position-embeddings 64
    --mock-data --tokenizer-type NullTokenizer --vocab-size 256
    --make-vocab-size-divisible-by 128 --split 100,0,0
    --micro-batch-size 1 --global-batch-size 8
    --train-iters 12 --lr-decay-iters 12 --lr-warmup-iters 2
    --lr 0.001 --min-lr 0.0001 --lr-decay-style WSD
    --lr-wsd-decay-iters 2 --lr-wsd-decay-style linear
    --optimizer angular_muown_simplified --angular-muown-c "$c"
    --bf16 --attention-dropout 0 --hidden-dropout 0 --weight-decay 0
    --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1 --seed 1234
    --num-workers 0 --log-interval 1 --eval-iters 0
    --no-gradient-accumulation-fusion --no-masked-softmax-fusion
    --no-bias-gelu-fusion --no-bias-dropout-fusion --no-rope-fusion
    --save "$out/checkpoints" --save-interval 6 --ckpt-format torch_dist
    --data-cache-path "$out/cache/data" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

# Reject NFS before writing caches; the closest existing parent determines the FS.
parent=$out
while [[ ! -d $parent ]]; do parent=$(dirname -- "$parent"); done
case "$(stat -f -c %T -- "$parent")" in
    nfs*) echo "OUT_DIR must use local scratch or /tmp, not NFS." >&2; exit 2 ;;
esac
mkdir -p -- "$out/tmp" "$out/cache"
cd -- "$repo"
"${command[@]}" 2>&1 | tee -a "$out/run.log"
