#!/bin/bash
#SBATCH --job-name=meglm-qwen3-dev-235B
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=64
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
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
BASE_DIR="$SLURM_SUBMIT_DIR"
export MEGATRON_PATH="/scratch/${SLURM_JOB_ACCOUNT}/users/${USER}/frameworks/NVIDIA-Megatron-LM"
OUTPUT_DIR="$BASE_DIR/output"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"

mkdir -p "$CHECKPOINT_PATH"    # This needs to exist

export PYTHONWARNINGS=ignore
export CUDA_DEVICE_MAX_CONNECTIONS=1    #Needed for sequence paralellism
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS    # Note: only valid if ntasks==ngpus

# OMP THREADING
export OMP_NUM_THREADS=1    # OMP_NUM_THREADS=1 is the safe option
export HSA_ENABLE_SDMA=0

# This setting is reported to provide a performance improvement
# (https://arxiv.org/pdf/2408.14090v1) but as of April 2025 is causing
# training instability on LUMI with pipeline parallelism.
# (see https://github.com/spyysalo/lumi-fineweb-replication/issues/1)
export NCCL_NCHANNELS_PER_PEER=32
export NCCL_MIN_NCHANNELS=$NCCL_NCHANNELS_PER_PEER

# DATA
DATA_PATH="/scratch/project_462000963/users/vitiugin/data_tok/oellm/hplt/fin_Latn/fin_Latn"
DATA_CACHE_PATH="/flash/project_462000963/users/${USER}/megatron_cache/"
TOKENIZER_MODEL="openeurollm/tokenizer-256k"
export TOKENIZERS_PARALLELISM=false

SEQ_LENGTH=4096
PROFILE=0
# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_ITERS=500
CLIP_GRAD=1.0
WEIGHT_DECAY=1e-1

# TRAINING
GLOBAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=1 # optimize this 
RECOMPUTATION=1

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
    --num-layers 94
    --hidden-size 4096
    --ffn-hidden-size 12288
    --moe-ffn-hidden-size 1536
    --num-attention-heads 64
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --qk-layernorm
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --bf16
    --num-experts 64
    --moe-router-topk 8
    --moe-router-dtype fp64 # fp64 uses native PyTorch implementation and thus avoid faulty TE-kernel on our LUMI environment. TODO: Investigate and fix the environment
    --kv-channels 128
    --rerun-mode disabled
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather
    --moe-router-force-load-balancing # This simulates a fully load-balanced routing scenario. DO NOT USE for real training runs!
    --disable-gloo-process-groups # Not sure about the impact. Helps with communication errors on LUMI.
)

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --seq-length $SEQ_LENGTH
    --train-iters 50
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion    # was buggy on AMD, do not enable without validating
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
    --overlap-param-gather
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
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
    --pipeline-model-parallel-size 16
    --expert-model-parallel-size 8
    --decoder-first-pipeline-num-layers 5
    --decoder-last-pipeline-num-layers 5
    --use-distributed-optimizer
    --sequence-parallel
    --context-parallel-size 1
    --tensor-model-parallel-size 1
)

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
    ${MEGATRON_PATH}/pretrain_gpt.py \
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
    "

srun $SRUN_ARGS \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launcher.sh \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
