#!/bin/bash
#SBATCH --job-name=MoE_qwen_30B
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --partition=booster
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --account=e-sta-openeurollm
#SBATCH --output logs/debug.log

set -eou pipefail
echo "Starting bash script"
module purge
#module load Stages/2026
#module load NVSHMEM/3.5.21-CUDA-13

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

#export CUDA_DEVICE_MAX_CONNECTIONS=1                    #This is needed for sequence paralellism
export WORLD_SIZE=$((SLURM_GPUS_PER_NODE * ${SLURM_NNODES}))
#OMP THREADING
export OMP_NUM_THREADS=1                                #Virtual threads to accompany the HW threads set by slurm

export PYTHONWARNINGS=ignore
#####
#DEBUGGING, INCREASE VERBOSITY IN LOGS

#TORCH
#export TORCH_SHOW_CPP_STACKTRACES=1                    #C++ traceback for pytorch crashes
#export TORCH_DIST_INIT_BARRIER=1                       #Initialize torch distributed new group instance with a barrier to synchronize ranks

export APPTAINERENV_TRITON_HOME=/tmp/triton_cache

##TORCHNCCL##
#TORCH FLIGHTRECORDER, see https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html
#export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
#export TORCH_NCCL_DUMP_ON_TIMEOUT=true
#export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=logs/fr/nccl_trace_rank

#NCCL/RCCL
export NCCL_DEBUG=INFO                                  #Verbose logging from RCCL
export NCCL_DEBUG_SUBSYS=ALL                            #Specify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P(Peer2Peer), COLL(Collecetives) etc.
export NCCL_DEBUG_FILE=logs/nccl.log
export NCCL_ASYNC_ERROR_HANDLING=1

##TRITON##
#export TRITON_ALWAYS_COMPILE=1                         #Always force Triton to compile even in the case of cache hit.

##TransformerEngine##
export NVTE_DEBUG=1                                    #Shows things like what dot product attention implementation is being used
export NVTE_DEBUG_LEVEL=2

#Optional cli arguments
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-0}"
MODEL_SIZE="${MODEL_SIZE:-9}"
TP_SIZE="${TP:-1}"
PP_SIZE="${PP:-2}"
VPP_SIZE="${VPP_SIZE:-1}"
NN="${NN:-$SLURM_NNODES}"
N_tasks="${N_tasks:-$SLURM_NTASKS}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
SEQ_LEN="${SEQ_LEN:-4096}"
ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-"debug"}"
RECOMPUTATION="${RECOMPUTATION:-1}"

export TOKENIZERS_PARALLELISM=false
WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5

# PATHS
BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="$BASE_DIR"/output/NVIDIA_meglm
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_PATH="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"

# DATA
DATA_CACHE_PATH="/e/project1/e-sta-openeurollm/komulainen1_jupiter/data/"
TOKENIZER_MODEL="openeurollm/tokenizer-256k"
export TOKENIZERS_PARALLELISM=false

##BATCH SIZE
GLOBAL_BATCH_SIZE="${GBS:-2048}"
#MBS max 2 with the default model and model parallel parameters
MICRO_BATCH_SIZE="${MBS:-1}"

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

#LLama 34B
if [[ $MODEL_SIZE -eq 34 ]]; then
    MODEL=PORO_34B
    NLAYERS=56
    NHIDDEN=7168
    NHEADS=56
    FFN_HIDDEN_SIZE=20480
    SEQ_LEN=4096
    NUM_KV_HEADS=8
    NUM_QUERY_GROUPS=8
elif [[ $MODEL_SIZE -eq 9 ]]; then
    MODEL=LLAMA3_8B
    NHIDDEN=4096
    FFN_HIDDEN_SIZE=14336
    NLAYERS=32
    NHEADS=32
    SEQ_LEN=2048
    NUM_KV_HEADS=8
    GROUP_SIZE=4
    NUM_QUERY_GROUPS=8
fi

LOG_INTERVAL=1
SAVE_INTERVAL=500
EVAL_INTERVAL=2000
EVAL_ITERS=1
INIT_METHOD_STD=0.00747017

OPTIMIZER_ARGS=(
    --use-distributed-optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-5
    --lr 1.5e-4
    --min-lr 1.5e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.05
)

DATA_ARGS=(
    --data-cache-path "$DATA_CACHE_PATH"
    --tokenizer-type NullTokenizer
    --mock-data
    --num-workers 4
    --vocab-size 256000
)

MODEL_ARGS=(
    --num-layers $NLAYERS
    --hidden-size $NHIDDEN
    --num-attention-heads $NHEADS
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --max-position-embeddings $SEQ_LEN
    --seq-length $SEQ_LEN
    --train-iters $TRAIN_ITERS
    --eval-iters $EVAL_ITERS
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --normalization RMSNorm
    --bf16
    --init-method-std $INIT_METHOD_STD
    --seed 42
    --untie-embeddings-and-output-weights
    --swiglu
    --attention-dropout 0
    --hidden-dropout 0
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --use-rotary-position-embeddings
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --distributed-timeout-minutes 10
    --disable-bias-linear
    --rerun-mode disabled
)

FUSED_KERNEL_ARGS=(
    --no-masked-softmax-fusion
)

OUTPUT_ARGS=(
    --log-throughput
    --log-interval 1
    --log-memory-to-tensorboard
    --tensorboard-dir $TENSORBOARD_PATH
    --tensorboard-queue-size 5
)

if [ "$ENABLE_WANDB" -eq 1 ]; then
    OUTPUT_ARGS+=(
        --wandb-project $WANDB_PROJECT
        --wandb-save-dir $OUTPUT_DIR
    )
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
)

if (( VPP_SIZE > 1)); then
    PARALLEL_ARGS+=(
        --num-layers-per-virtual-pipeline-stage $VPP_SIZE
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
	--recompute-activations
    )
fi

#PYTORCH PROFILER ARGS
PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start 6
    --profile-step-end 10
    --profile-ranks 0 1 2 3 4 5 6 7
)

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
    "${MOE_ARGS[@]}" \
    "${FUSED_KERNEL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
"

if [ "$ENABLE_PROFILING" -eq 1 ]; then
    COMMAND+=" ${PROFILE_ARGS[@]}"
fi

#SINGULARITY ENVIRONMENT
export CONTAINER="/e/project1/e-sta-openeurollm/harmsen1/containers/pytorch_25.08_aarch64.sif"
export APPTAINER_BINDPATH="/e/project1/e-sta-openeurollm"

# Tell Triton exactly where libcuda.so.1 lives INSIDE the container
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

echo "MASTER_ADDR=$MASTER_ADDR"

export PWD=(`pwd -P`)

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
# --network=disable_rdzv_get: An alternative rendezvous protocol designed by HPE spesifically for AI workloads

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    --label \
    "

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "

apptainer exec --nv $CONTAINER bash -c "$LAUNCHER $COMMAND"