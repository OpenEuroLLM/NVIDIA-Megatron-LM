#!/bin/bash
#SBATCH --job-name=moe_qwen3
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --partition=booster
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --account=e-sta-openeurollm
#SBATCH --output logs/latest.log

set -euo pipefail
set -x

module purge

for ARG in "$@"; do
  case "$ARG" in
    *=*)
      KEY="${ARG%%=*}"
      VALUE="${ARG#*=}"
      export "$KEY"="$VALUE"
      ;;
    *)
      echo "Ignoring arg '$ARG' (expected KEY=VALUE)" >&2
      ;;
  esac
done

# ------------------------------
# Defaults (override via KEY=VALUE)
# ------------------------------
: "${MODEL_SIZE:=30B}"

case "$MODEL_SIZE" in
  235B)
    : "${TP:=2}"
    : "${PP:=8}"
    : "${EP:=16}"
    : "${VPP:=4}"
    ;;
  30B)
    : "${TP:=1}"
    : "${PP:=2}"
    : "${EP:=4}"
    : "${VPP:=1}"
    ;;
  *)
    echo "Unknown MODEL_SIZE=$MODEL_SIZE" >&2
    exit 1
    ;;
esac

: "${SEQ_LENGTH:=4096}"
: "${GBS:=2048}"
: "${MBS:=2}"
: "${TRAIN_ITERS:=15}"
: "${NUM_WORKERS:=8}"
: "${CPUS_PER_TASK:=288}"
: "${FP8:=0}"
: "${DISPATCHER:=flex}"               # This needs to be flex for deepep/hybridep
: "${D_BACKEND:=hybridep}"
: "${NUM_SMS:=16}"                    # GPU Streaming multiprocessors for deepep/hybridep
: "${ENABLE_MOE_OVERLAP:=0}"          # --overlap-moe-expert-parallel-comm + --delay-wgrad-compute
: "${CUDA_MAX_CONNECTIONS:=1}"        # set >1 when enabling overlap
: "${GC_INTERVAL:=5}"
: "${NUMA_BIND:=node}"
: "${COMMENT:=""}"
: "${PROFILE:=0}"
: "${CPU_OFFLOAD:=0}"
: "${WANDB:=1}"
: "${WANDB_PROJECT:=qwen3-30B-A2B-scaling-jupiter}"

# SHARP runtime env
export WANDB_MODE="offline"
export SHARP_SMX_SOCK_ADDR_FAMILY=ipv4
export NCCL_SOCKET_FAMILY=AF_INET

# Distributed
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-9999}
export WORLD_SIZE=$((SLURM_GPUS_PER_NODE * SLURM_NNODES))

export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-"logs/nccl/node-%h.log"}

export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_MAX_CONNECTIONS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVTE_CPU_OFFLOAD_V1=1
#export NCCL_NVLS_ENABLE=0
#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=2
#export NVTE_NORM_FWD_USE_CUDNN=0
#export NVTE_NORM_BWD_USE_CUDNN=0

# ------------------------------
# Model config
# ------------------------------
if [[ "$MODEL_SIZE" == "235B" ]]; then
  MODEL=qwen3-235B-A22B
  NLAYERS=94
  NHIDDEN=4096
  NHEADS=64
  FFN_HIDDEN_SIZE=12288
  NUM_QUERY_GROUPS=4
  NUM_EXPERTS=128
  MOE_FFN_SIZE=1536
  KV_CHANNELS=128
elif [[ "$MODEL_SIZE" == "30B" ]]; then
  MODEL=qwen3-30B-A2B
  NLAYERS=48
  NHIDDEN=2048
  NHEADS=32
  FFN_HIDDEN_SIZE=6144
  NUM_QUERY_GROUPS=4
  NUM_EXPERTS=128
  MOE_FFN_SIZE=768
  KV_CHANNELS=128
else
  echo "Unknown MODEL_SIZE=$MODEL_SIZE" >&2
  exit 1
fi

# ------------------------------
# Paths
# ------------------------------
SLURM_JOB_NAME="JUPITER_${MODEL}_TP${TP}_PP${PP}_VPP_${VPP}_EP${EP}_GBS_${GBS}_MBS${MBS}_NN${SLURM_NNODES}_${COMMENT}"
OUTPUT_DIR="/e/project1/e-sta-openeurollm/komulainen1_jupiter/output/MEGLM/distributed-training-benchmarking"
LOG_FILE="${OUTPUT_DIR}/${SLURM_JOB_NAME}.log"
TENSORBOARD_PATH="$OUTPUT_DIR/$SLURM_JOB_NAME/tensorboard/"
DATA_CACHE_PATH="/e/scratch/e-sta-openeurollm/komulainen1_jupiter/data/cache"
DATA_PATH="/e/project1/e-sta-openeurollm/komulainen1_jupiter/data/fineweb-10BT/fineweb-10BT_text_document"
TOKENIZER_MODEL="/e/project1/e-sta-openeurollm/komulainen1_jupiter/hf_cache/hub/models--openeurollm--tokenizer-256k/snapshots/ac1ce776f8a78fc7aa96dfc91ffc1ae6966b2fa1"

# ------------------------------
# Args
# ------------------------------
OPTIMIZER_ARGS=(
  --no-load-optim
  --use-distributed-optimizer
  --optimizer adam
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
  --lr 3e-4
  --min-lr 3e-5
  --lr-decay-style cosine
  --clip-grad 1.0
  --weight-decay 1e-1
)

DATA_ARGS=(
  --data-cache-path "$DATA_CACHE_PATH"
  --tokenizer-type HuggingFaceTokenizer
  --data-path "$DATA_PATH"
  --tokenizer-model "$TOKENIZER_MODEL"
  --num-workers "$NUM_WORKERS"
  --vocab-size 256000
  --no-mmap-bin-files
  --no-create-attention-mask-in-dataloader
  --manual-gc
  --manual-gc-interval $GC_INTERVAL
)

TRAIN_ARGS=(
  --micro-batch-size "$MBS"
  --global-batch-size "$GBS"
  --bf16
  --train-iters "$TRAIN_ITERS"
  --eval-interval 999
  --eval-iters 0
  --distributed-timeout-minutes 30
  --use-flash-attn
  --disable-gloo-process-groups             # Gloo process groups are needed only for saving dist optimizer states. GLOO_SOCKET_IFNAME=ib0 is also needed
  --disable-bias-linear
  --no-check-for-nan-in-loss-and-grad
  --rerun-mode disabled
)

MODEL_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --qk-layernorm
  --normalization RMSNorm
  --norm-epsilon 1e-06
  --position-embedding-type rope
  --swiglu
  --group-query-attention
  --rotary-base 1000000
  --num-query-groups "$NUM_QUERY_GROUPS"
  --kv-channels "$KV_CHANNELS"
  --num-layers "$NLAYERS"
  --ffn-hidden-size "$FFN_HIDDEN_SIZE"
  --hidden-size "$NHIDDEN"
  --num-attention-heads "$NHEADS"
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$SEQ_LENGTH"
  --untie-embeddings-and-output-weights
)

MOE_ARGS=(
  --num-experts "$NUM_EXPERTS"
  --moe-ffn-hidden-size "$MOE_FFN_SIZE"
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff 0.001
  --moe-router-topk 8
  --moe-router-dtype fp32
  --moe-grouped-gemm
  --moe-permute-fusion
  --moe-router-force-load-balancing
  --moe-token-dispatcher-type "$DISPATCHER"
  --moe-flex-dispatcher-backend "$D_BACKEND"
  --moe-hybridep-num-sms "$NUM_SMS"
  --moe-deepep-num-sms  "$NUM_SMS"
)

RECOMPUTE_ARGS=(
  --recompute-granularity full 
  --recompute-method uniform 
  --recompute-num-layers 1
)

FUSED_KERNEL_ARGS=(
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl te
  --moe-router-fusion
  --overlap-grad-reduce
  --overlap-param-gather
)

OUTPUT_ARGS=(
  --log-throughput
  --log-interval 1
  --log-timers-to-tensorboard
  --tensorboard-dir "$TENSORBOARD_PATH"
  --tensorboard-queue-size 100
)

if [ "$WANDB" == "1" ]; then
    OUTPUT_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-save-dir "$OUTPUT_DIR/$SLURM_JOB_NAME"
        --wandb-exp-name "$SLURM_JOB_NAME"
        --wandb-entity openeurollm-project
    )
fi

PARALLEL_ARGS=(
  --tensor-model-parallel-size "$TP"
  --pipeline-model-parallel-size "$PP"
  --expert-model-parallel-size "$EP"
  --expert-tensor-parallel-size 1
)

if [[ "$ENABLE_MOE_OVERLAP" == "1" ]]; then
  MOE_ARGS+=( --overlap-moe-expert-parallel-comm --delay-wgrad-compute )
fi

if (( VPP > 1 )); then
  PARALLEL_ARGS+=( --num-layers-per-virtual-pipeline-stage "$VPP" )
fi

# FP8 args
FP8_ARGS=(
  --fp8-format hybrid
  --fp8-recipe blockwise
  --fp8-param-gather
)

# PYTORCH PROFILER ARGS
PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start 4
    --profile-step-end 5
    --profile-ranks 0 1 2 3
)

OFFLOAD_ARGS=(
  --fine-grained-activation-offloading
  --offload-modules expert_fc1 core_attn
)

# ------------------------------
# Command assembly
# ------------------------------
CMD=(
  pretrain_gpt.py
  "${MODEL_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${PARALLEL_ARGS[@]}"
  "${OUTPUT_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${MOE_ARGS[@]}"
  "${FUSED_KERNEL_ARGS[@]}"
  "${TRAIN_ARGS[@]}"
  "${RECOMPUTE_ARGS[@]}"
)

if [[ "$FP8" == "1" ]]; then
  CMD+=( "${FP8_ARGS[@]}" )
fi
if [[ "$PROFILE" == "1" ]]; then
  CMD+=( "${PROFILE_ARGS[@]}" )
fi
if [[ "$CPU_OFFLOAD" == "1" ]]; then
  CMD+=( "${OFFLOAD_ARGS[@]}" )
fi

export CONTAINER="/e/project1/e-sta-openeurollm/container/nemo_26.04.sif"
export APPTAINER_BINDPATH="/e/project1/e-sta-openeurollm"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"
export APPTAINERENV_TRITON_HOME="/tmp/triton_cache"

# Disable slurm's cpu bind with --cpu-bind="none" probably needed incase we use torchrun --numa-binding - To be confirmed
SRUN_ARGS=(
  --wait=60
  --kill-on-bad-exit=1
  --jobid "$SLURM_JOB_ID"
  --cpus-per-task="$CPUS_PER_TASK"
  --cpu-bind="none"
)

LAUNCHER=(
  python -u -m torch.distributed.run
  --nproc_per_node "$SLURM_GPUS_PER_NODE"
  --nnodes "$SLURM_NNODES"
  --node_rank '${SLURM_PROCID}'
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT"
  --rdzv_backend c10d
  --max_restarts 0
  --tee 3
  --numa-binding "node"
)

srun "${SRUN_ARGS[@]}" apptainer exec --writable-tmpfs --nv "$CONTAINER" bash -lc "${LAUNCHER[*]} ${CMD[*]}" 2>&1 | tee "$LOG_FILE"
