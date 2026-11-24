#!/usr/bin/env bash
# Poro 3 (Qwen3.5-35B-A3B based) SFT/CPT with a frozen MoE router, 32k sequence length.
#
# Loads from an existing Megatron checkpoint (base or an earlier CPT stage) and
# continues training with the router (token->expert gate) weights frozen, via
# pretrain_gpt_freeze_router.py. See that script for what "frozen router" means.
#
# NOTE: DATA_PATH must already be packed (sequence-packed, drop-long applied).
# Unpacked/unrolled data will not work with --packed-sft below.
#
# Required env vars (export these, or edit the defaults below, before sbatch'ing):
#   MEGATRON_ROOT   Path to this Megatron-LM checkout.
#   EXPERIMENT      Output root for this run (checkpoints/logs/cache go under it).
#   DATA_PATH       Packed SFT jsonl.
#   LOAD_CPT        Megatron checkpoint dir to initialize from (must contain
#                   latest_checkpointed_iteration.txt and an iter_NNNNNNN/ dir).
#   TOKENIZER_MODEL HF tokenizer dir for Qwen3.5-35B-A3B.
#   CONTAINER       Apptainer/Singularity image with the training environment.
#
# Optional env vars:
#   TRAIN_ITERS, WARMUP_ITERS   Defaults below assume global-batch-size 128;
#                               recompute as ceil(epochs * packed_rows / GBS).
#   FLA_OVERLAY                 Apptainer overlay image, if your container needs
#                               one for the gated-delta-net (FLA) kernels.
#   ROCM_FIX                    Path to a patched libamdhip64 shared object, if
#                               your ROCm build needs one bind-mounted over the
#                               container's copy.
#
#SBATCH --job-name=poro3_35b_a3b_frozerouter_32k
#SBATCH --partition=<your-partition>
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=240:00:00
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${MEGATRON_ROOT:?set MEGATRON_ROOT to this Megatron-LM checkout}"
: "${EXPERIMENT:?set EXPERIMENT to an output directory for this run}"
: "${DATA_PATH:?set DATA_PATH to the packed SFT jsonl}"
: "${LOAD_CPT:?set LOAD_CPT to a Megatron checkpoint dir to initialize from}"
: "${TOKENIZER_MODEL:?set TOKENIZER_MODEL to the HF tokenizer dir}"
: "${CONTAINER:?set CONTAINER to an apptainer/singularity image}"

TRAIN_ITERS=${TRAIN_ITERS:-11279}   # 1 epoch = ceil(packed_rows / global_batch_size)
WARMUP_ITERS=${WARMUP_ITERS:-338}   # ~3% of TRAIN_ITERS

RUN_NAME=${RUN_NAME:-poro3_35B_A3B_frozerouter_32k}
OUTPUT_DIR=$EXPERIMENT/output/$RUN_NAME
LOG_DIR=$EXPERIMENT/logs
CACHE_DIR=$EXPERIMENT/cache/$RUN_NAME

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-9981}
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export RCCL_NET_GDR_LEVEL=2 RCCL_ENABLE_DIRECT_COPY=1
export NCCL_BUFFSIZE=8388608 RCCL_MSCCL_ENABLE=0
export HSA_ENABLE_SDMA=1 RCCL_P2P_DISABLE=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_MIN_NCHANNELS=8 RCCL_NCHANNELS=8 NCCL_NTHREADS=512
export NCCL_NET_GDR_READ=1 RCCL_ENABLE_SMI_P2P=1
export NCCL_CROSS_NIC=0
export NCCL_IB_GID_INDEX=3 NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13
export NCCL_IB_QPS_PER_CONNECTION=4 HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1 NVTE_CK_IS_V3_ATOMIC_FP32=1
export USE_ROCM_AITER_ROPE_BACKEND=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=5400
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID

# If your fabric needs explicit interface/HCA names, set these before sbatch'ing:
#   export NCCL_SOCKET_IFNAME=... GLOO_SOCKET_IFNAME=...
#   export NCCL_IB_HCA=...

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/checkpoints" "$CACHE_DIR"
test -s "$DATA_PATH"
test -s "$TOKENIZER_MODEL/tokenizer.json"
test -s "$LOAD_CPT/latest_checkpointed_iteration.txt"
LC_ITER=$(cat "$LOAD_CPT/latest_checkpointed_iteration.txt")
[[ $LC_ITER =~ ^[0-9]+$ ]]
test -f "$LOAD_CPT/iter_$(printf '%07d' "$LC_ITER")/.metadata"
test -f "$MEGATRON_ROOT/pretrain_gpt.py"
test -f "$MEGATRON_ROOT/pretrain_gpt_freeze_router.py"
test -f "$CONTAINER"

# First launch resets the optimizer and step from LOAD_CPT. A later invocation
# (same EXPERIMENT/RUN_NAME) resumes from this run's own checkpoint instead.
if test -s "$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt"; then
    export LOAD_DIR=$OUTPUT_DIR/checkpoints
    export FRESH_START=0
    RESUME_STEP=$(cat "$LOAD_DIR/latest_checkpointed_iteration.txt")
    [[ $RESUME_STEP =~ ^[0-9]+$ ]]
    test -f "$LOAD_DIR/iter_$(printf '%07d' "$RESUME_STEP")/.metadata"
    echo "Resuming from step $RESUME_STEP"
else
    export LOAD_DIR=$LOAD_CPT
    export FRESH_START=1
    echo "Initializing from $LOAD_CPT (iter $LC_ITER)"
fi

echo "Launching $RUN_NAME on $WORLD_SIZE GPUs (train-iters $TRAIN_ITERS, warmup $WARMUP_ITERS)"
srun --output="$LOG_DIR/rank-%j-%t.out" --error="$LOG_DIR/rank-%j-%t.err" bash -c '
    set -euo pipefail
    export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
    export AITER_JIT_DIR=/tmp/$USER/aiter_jit_dir
    export LOCAL_ADDR=$(nslookup "$(hostname)" | awk "/Address: / {addr=\$2} END {print addr}")
    if test "$FRESH_START" = 1; then
        set -- --finetune --no-load-optim --no-load-rng
    else
        set --
    fi
    APPTAINER_ARGS=(--rocm --bind "$MEGATRON_ROOT" --bind "$EXPERIMENT" --bind "$DATA_PATH" --bind "$LOAD_CPT" --bind "$TOKENIZER_MODEL")
    if [[ -n "${FLA_OVERLAY:-}" ]]; then
        APPTAINER_ARGS+=(--overlay "$FLA_OVERLAY:ro")
    fi
    if [[ -n "${ROCM_FIX:-}" ]]; then
        APPTAINER_ARGS+=(--bind "$ROCM_FIX:/opt/venv/lib/python3.10/site-packages/torch/lib/libamdhip64.so")
        APPTAINER_ARGS+=(--bind "$ROCM_FIX:/opt/venv/lib/python3.10/site-packages/torch/lib/libamdhip64.so.7")
    fi
    exec apptainer exec "${APPTAINER_ARGS[@]}" \
        "$CONTAINER" python "$MEGATRON_ROOT/pretrain_gpt_freeze_router.py" \
        --data-path "$DATA_PATH" --packed-sft \
        --data-cache-path "$CACHE_DIR" \
        --dataloader-type cyclic --split 1000,0,0 --eval-iters 0 --eval-interval 5000 \
        --seq-length 32768 --max-position-embeddings 262144 \
        --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
        --vocab-size 248320 --make-vocab-size-divisible-by 256 \
        --num-layers 40 --hidden-size 2048 --ffn-hidden-size 512 \
        --num-attention-heads 16 --num-query-groups 2 --kv-channels 256 \
        --num-experts 256 --moe-router-topk 8 --moe-ffn-hidden-size 512 \
        --moe-shared-expert-intermediate-size 512 --moe-shared-expert-gate \
        --moe-router-dtype fp32 --moe-router-load-balancing-type global_aux_loss \
        --moe-aux-loss-coeff 0.0001 --moe-token-dispatcher-type alltoall \
        --moe-permute-fusion \
        --experimental-attention-variant gated_delta_net --linear-attention-freq 4 \
        --attention-output-gate --attention-softmax-in-fp32 \
        --attention-dropout 0.0 --hidden-dropout 0.0 \
        --group-query-attention --qk-layernorm --normalization RMSNorm \
        --norm-epsilon 1e-6 --apply-layernorm-1p \
        --position-embedding-type rope --use-rotary-position-embeddings \
        --rotary-base 10000000 --rotary-percent 0.25 \
        --swiglu --disable-bias-linear --apply-wd-to-qk-layernorm \
        --untie-embeddings-and-output-weights --no-create-attention-mask-in-dataloader \
        --bf16 --use-flash-attn --te-rng-tracker \
        --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --context-parallel-size 1 \
        --sequence-parallel --micro-batch-size 1 --global-batch-size 128 \
        --use-distributed-optimizer --overlap-grad-reduce \
        --recompute-activations --recompute-granularity full \
        --clip-grad 1.0 \
        --lr 1e-5 --min-lr 1e-6 --lr-decay-style linear \
        --lr-warmup-iters '"$WARMUP_ITERS"' --lr-decay-iters '"$TRAIN_ITERS"' \
        --train-iters '"$TRAIN_ITERS"' --seed 1234 \
        --load "$LOAD_DIR" "$@" --save "$OUTPUT_DIR/checkpoints" \
        --ckpt-format torch_dist --dist-ckpt-strictness log_unexpected \
        --save-interval 1000 \
        --distributed-timeout-minutes 240 --num-workers 7 \
        --manual-gc --manual-gc-interval 100 \
        --log-interval 10 --log-progress --log-throughput \
        --log-timers-to-tensorboard --tensorboard-dir "$OUTPUT_DIR/tensorboard"
'
