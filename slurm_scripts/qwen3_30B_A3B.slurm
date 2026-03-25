#!/bin/bash
#SBATCH --job-name=test_meglm-qwen3-30B_A3B
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000963
#SBATCH --output logs/%x-%j.log

# Bash "strict mode"
# (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# Compilers in the container
export CC=gcc-12
export CXX=g++-12

# PATHS
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export MEGATRON_PATH="/scratch/${SLURM_JOB_ACCOUNT:-project_462000963}/users/${USER}/frameworks/NVIDIA-Megatron-LM"
OUTPUT_DIR="$BASE_DIR/output"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/${SLURM_JOB_NAME:-local}-${SLURM_JOBID:-0}"

mkdir -p "$CHECKPOINT_PATH"    # This needs to exist

export PYTHONWARNINGS=ignore
export LOGLEVEL=ERROR
export CUDA_DEVICE_MAX_CONNECTIONS=1    # Needed for sequence parallelism
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
    export MASTER_ADDR=localhost
fi
export MASTER_PORT=9999
export WORLD_SIZE=${WORLD_SIZE:-${SLURM_NTASKS:-32}}

# PERFORMANCE FLAGS
export OMP_NUM_THREADS=1
#export HSA_ENABLE_SDMA=0
#export NCCL_NCHANNELS_PER_PEER=32
#export NCCL_MIN_NCHANNELS=$NCCL_NCHANNELS_PER_PEER

# DATA
DATA_PATH="/scratch/project_462000963/users/vitiugin/data_tok/oellm/hplt/fin_Latn/fin_Latn"
DATA_CACHE_PATH="/flash/project_462000963/users/${USER}/megatron_cache/"
TOKENIZER_MODEL="openeurollm/tokenizer-256k"
export TOKENIZERS_PARALLELISM=false

PROFILE=0
RECOMPUTATION=1
SEQ_LENGTH=4096
GLOBAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=1

# Parallelism defaults (override via environment when sweeping).
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
EXPERT_TP_SIZE=${EXPERT_TP_SIZE:-$TP_SIZE}
PARALLEL_ORDER=${PARALLEL_ORDER:-tp-cp-ep-dp-pp}

# Model constants used by parallel sweep safeguards.
NUM_LAYERS=48
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=4
NUM_EXPERTS=128

# Optional candidate lists for safe sweep dry-runs.
TP_CANDIDATES=(${TP_CANDIDATES:-1 2 4 8})
PP_CANDIDATES=(${PP_CANDIDATES:-1 2 4 8})
CP_CANDIDATES=(${CP_CANDIDATES:-1})
EP_CANDIDATES=(${EP_CANDIDATES:-1 2 4 8 16})

validate_parallel_combo() {
    local tp=$1
    local pp=$2
    local cp=$3
    local ep=$4
    local etp=$5

    if (( tp < 1 || pp < 1 || cp < 1 || ep < 1 || etp < 1 )); then
        return 1
    fi

    # Megatron: world_size must be divisible by tp * pp * cp.
    local model_size=$((tp * pp * cp))
    if (( WORLD_SIZE % model_size != 0 )); then
        return 1
    fi

    # Megatron core parallel-state: world_size must be divisible by etp * ep * pp.
    local expert_model_size=$((etp * ep * pp))
    if (( WORLD_SIZE % expert_model_size != 0 )); then
        return 1
    fi

    # Megatron args validation / transformer config checks.
    if (( NUM_LAYERS % pp != 0 )); then
        return 1
    fi
    if (( NUM_ATTENTION_HEADS % tp != 0 )); then
        return 1
    fi
    if (( NUM_QUERY_GROUPS % tp != 0 )); then
        return 1
    fi
    if (( cp > 1 )) && (( SEQ_LENGTH % (2 * cp) != 0 )); then
        return 1
    fi
    if (( ep > 1 )) && (( NUM_EXPERTS % ep != 0 )); then
        return 1
    fi

    # Megatron core rank-ordering constraint when order does not end with pp.
    if [[ "$PARALLEL_ORDER" != *pp ]]; then
        local dp_size=$((WORLD_SIZE / model_size))
        local expert_dp_size=$((WORLD_SIZE / expert_model_size))
        if (( pp > 1 )) && (( dp_size != expert_dp_size )); then
            return 1
        fi
    fi

    return 0
}

print_valid_parallel_combos() {
    local found=0
    echo "Valid parallel tuples (tp pp cp ep etp dp expert_dp):"
    for tp in "${TP_CANDIDATES[@]}"; do
        for pp in "${PP_CANDIDATES[@]}"; do
            for cp in "${CP_CANDIDATES[@]}"; do
                for ep in "${EP_CANDIDATES[@]}"; do
                    local etp=$tp
                    if validate_parallel_combo "$tp" "$pp" "$cp" "$ep" "$etp"; then
                        local model_size=$((tp * pp * cp))
                        local expert_model_size=$((etp * ep * pp))
                        local dp_size=$((WORLD_SIZE / model_size))
                        local expert_dp_size=$((WORLD_SIZE / expert_model_size))
                        echo "  $tp $pp $cp $ep $etp $dp_size $expert_dp_size"
                        found=1
                    fi
                done
            done
        done
    done

    if (( found == 0 )); then
        echo "No valid combinations found for provided candidate lists."
        return 1
    fi
    return 0
}

if [[ "${LIST_VALID_PARALLEL_COMBOS:-0}" == "1" ]]; then
    print_valid_parallel_combos
    exit 0
fi

if ! validate_parallel_combo "$TP_SIZE" "$PP_SIZE" "$CP_SIZE" "$EP_SIZE" "$EXPERT_TP_SIZE"; then
    echo "Invalid parallel configuration: tp=${TP_SIZE} pp=${PP_SIZE} cp=${CP_SIZE} ep=${EP_SIZE} etp=${EXPERT_TP_SIZE}"
    echo "Set LIST_VALID_PARALLEL_COMBOS=1 to print only valid combinations for your candidate lists."
    exit 1
fi

OPTIMIZER_ARGS=(
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
    --tokenizer-model "$TOKENIZER_MODEL"
    --data-path "$DATA_PATH"
    --num-workers 4
    --vocab-size 256000
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --max-position-embeddings 32768
    --num-layers $NUM_LAYERS
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads $NUM_ATTENTION_HEADS
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --qk-layernorm
    --normalization RMSNorm
    --norm-epsilon 1e-06
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --bf16
    --kv-channels 128
    --rerun-mode disabled
    --disable-gloo-process-groups
)

MOE_ARGS=(
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather
    --moe-router-force-load-balancing
    --num-experts $NUM_EXPERTS
    --moe-router-topk 8
    --moe-router-dtype fp64
)

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --seq-length $SEQ_LENGTH
    --train-iters 5
    --eval-iters 1
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
    --overlap-param-gather
)

OUTPUT_ARGS=(
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
    --log-interval 1
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
)

PARALLEL_ARGS=(
    --pipeline-model-parallel-size $PP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $EXPERT_TP_SIZE
    --use-distributed-optimizer
    --context-parallel-size $CP_SIZE
    --tensor-model-parallel-size $TP_SIZE
)

if (( TP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
        --recompute-activations
        --recompute-granularity selective
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
    --profile
        --use-pytorch-profiler
        --profile-ranks 0
        --profile-step-start 5
        --profile-step-end 7
    --record-memory-history
    --memory-snapshot-path "${TENSORBOARD_DIR}/memory_snapshot.pickle"
    )
else
    PROFILE_ARGS=()
fi

CHECKPOINT_ARGS=(
#    --async-save    # requires --ckpt-format torch_dist
#    --load "$CHECKPOINT_PATH"
#    --save "$CHECKPOINT_PATH"
#    --save-interval $SAVE_INTERVAL
)

COMMAND=" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
"

#SINGULARITY ENVIRONMENT
export CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-te-dockerhash-560b76ceab56.sif
export SINGULARITY_BIND=/boot/config-6.4.0-150600.23.73_15.0.14-cray_shasta_c,/scratch/project_462000963/,/projappl/project_462000963/,/flash/project_462000963/,/var/spool/slurmd
echo "CONTAINER" $CONTAINER

export PWD=(`pwd -P`)

# Mask to bind tasks to CPUs for one thread per core
c="fe"
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
# --network=disable_rdzv_get: An alternative rendezvous protocol designed by HPE spesifically for AI workloads

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    --cpu-bind=mask_cpu:$BIND_MASK \
    "

srun $SRUN_ARGS \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launcher.sh \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
