#!/bin/bash
#SBATCH --job-name=MoE_qwen_30B_profile
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000963
#SBATCH --output logs/debug.log

set -eou pipefail
echo "Starting bash script"
module purge

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
export CUDA_DEVICE_MAX_CONNECTIONS=1                    #This is needed for sequence paralellism
export WORLD_SIZE=$((SLURM_GPUS_PER_NODE * ${SLURM_NNODES}))
#OMP THREADING
export OMP_NUM_THREADS=1                                #Virtual threads to accompany the HW threads set by slurm

#HSA=Heterogeneous System Architecture
#AMD provides HSA through ROCR https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/index.html#hsa-runtime-api-and-runtime-for-rocm
export HSA_ENABLE_SDMA=0                                #https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/
                                                        #https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-memory.html#system-direct-memory-access
                                                        #HSA_ENABLE_SDMA=0 -- Uses blit kernels for comms through the gpu -- more bandwith between gcd's
                                                        #With HSA_ENABLE_SDMA=1 capped at 50GB/S uni-directional bandwidth
                                                        #Enables overlapping communication with computation
#export HSA_ENABLE_PEER_SDMA=0                          #Enable or disable SDMA engines for device-to-device copies, HSA_ENABLE_SDMA overrides this value
                                                        #https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/api-reference/environment_variables.html
#export HSA_XNACK=1                                     #A fallback for pagefault https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-memory.html#xnack

export PYTHONWARNINGS=ignore
#####
#DEBUGGING, INCREASE VERBOSITY IN LOGS

#TORCH
#export TORCH_SHOW_CPP_STACKTRACES=1                    #C++ traceback for pytorch crashes
#export TORCH_DIST_INIT_BARRIER=1                       #Initialize torch distributed new group instance with a barrier to synchronize ranks

##TORCH COMPILE/TORCHINDUCTOR##
export SINGULARITYENV_TRITON_HOME=/tmp/triton_cache
#export TORCH_COMPILE_DISABLE=1                          #Disable torch.compile
#export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1             #Disable all caching for Torchinductor
#export TORCHINDUCTOR_MAX_AUTOTUNE=1                     #More rigorous autotuning done by Triton for GPU kernels, this might lead to more performant kernels at the cost of increased compilation time
#export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1             #Enabling this decreases the chance of cache hit slightly
#export TORCHINDUCTOR_COMPILE_THREADS=$
export TORCH_LOGS="-inductor,-dynamo"
#export TORCHDYNAMO_VERBOSE=1
export ROCM_PYTORCH_ARCH=gfx90a

##TORCHNCCL##
#TORCH FLIGHTRECORDER, see https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html
#export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
#export TORCH_NCCL_DUMP_ON_TIMEOUT=true
#export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=logs/fr/nccl_trace_rank

##LIBFABRIC##
#export FI_LOG_LEVEL=warn                              #Very verbose Libfabric logging. May cause performance degredation when set to 'debug'
#export FI_LOG_PROV=cxi                                 #Determine which libfabric provider to get logs from. cxi is the only provider on LUMI
#export FI_CXI_ODP=1                                    #On demand paging. This should be an alternative to the CXI_FORK_SAFE for unsafe fork. See more with man intro_mpi
#export FI_HMEM=rocr                                    #Ensure that libfabric uses rocr's hmem implementation. Not sure if this supported for rocm < 6.4
#export FI_HMEM_ROCR_USE_DMABUF=1
#export FI_MR_ROCR_CACHE_MONITOR_ENABLED=1              #Detecting ROCR device memory (FI_HMEM_ROCR) changes made between the device virtual addresses used by an application and the underlying device physical pages.
#export SLINGSHOT_DEVICES=cxi0,cxi1                     #Specify which cxi NIC's get enabled
#export FI_CXI_REQ_BUF_SIZE=
#export FI_CXI_RDZV_GET_MIN=16364                       #Min byte size of message for cxi to use rendezvouz instead of eager messaging
#export FI_CXI_RDZV_THRESHOLD=32728                     #Max byte size
#export FI_CXI_DEFAULT_TX_SIZE=4096                     #Default is 1024
#export FI_CXI_RX_SIZE=4096
#export FI_CXI_DISABLE_HOST_REGISTER=1
#export FI_CXI_RX_MATCH_MODE=software                    #Address matching can be either hardware(cxi) or software
#export FI_CXI_RDZV_PROTO=alt_read
#export FI_MR_CACHE_MONITOR=userfaultfd
#export FI_CXI_DEFAULT_CQ_SIZE=131072

#NCCL/RCCL
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1                 #Print c++ code traceback for nccl collectives
export NCCL_DEBUG=INFO                                  #Verbose logging from RCCL
export NCCL_DEBUG_SUBSYS=ALL                            #Specify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P(Peer2Peer), COLL(Collecetives) etc.
#export NCCL_NCHANNELS_PER_PEER=16                      #Increases the amount of channels available for p2p communication. Should be a slight performance boost. Might lead to NaN's
#export NCCL_MAX_NCHANNELS=16
#export HIP_LAUNCH_BLOCKING=1                           #MAKES NCCL COLLECTIVES SYNCHRONOUS INSTEAD OF ASYNC
#export NCCL_DMABUF_ENABLE=1                            #Enable DMA buffers from the RCCL side
#export GPU_MAX_HW_QUEUES=2
#export NCCL_SET_THREAD_NAME=1                          #Give meaningful names to NCCL cpu threads
#export NCCL_TUNER_PLUGIN=/projappl/project_462000353/villekom/rccl-tuner/librccl-tuner.so

##HIP/AMD##
#export AMD_SERIALIZE_KERNEL=3                          #Verbose logs from HIP
#export AMD_LOG_LEVEL=1                                 #Very verbose logging. In our context seems to concern hipified cuda kernels
#export HIPBLASLT_LOG_LEVEL=4                           #https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/logging-heuristics.html#logging-heuristics
#export HIPBLASLT_LOG_FILE=logs/hipblas_%i.log          #Redirect hipblas logging output

##TRITON##
#export TRITON_ALWAYS_COMPILE=1                         #Always force Triton to compile even in the case of cache hit.

##TransformerEngine##
#export NVTE_DEBUG=1                                    #Shows things like what dot product attention implementation is being used
#export NVTE_DEBUG_LEVEL=2
#export NVTE_USE_RMSNORM_TRITON=1                      #Experimental RMRSNORM triton kernel optimized for AMD/ROCM
#export GEMM_TUNING=1
#export TE_HIPBLASLT_TUNING_RUN_COUNT=10
#export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
#export TE_HIPBLASLT_ALGO_SAVE=logs/algo_tune.csv

#Optional cli arguments
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-0}"
TP="${TP:-1}"
PP="${PP:-2}"
EP="${EP:-8}"
VPP="${VPP:-1}"
NN="${NN:-$SLURM_NNODES}"
N_tasks="${N_tasks:-$SLURM_NTASKS}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-"debug"}"
RECOMPUTATION="${RECOMPUTATION:-1}"
# PATHS
BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="/scratch/project_462000963/users/$USER/output/NVIDIA_meglm"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_PATH="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"

mkdir -p "$CHECKPOINT_PATH"    # This needs to exist

# DATA
DATA_PATH="/scratch/project_462000963/users/vitiugin/data_tok/openeurollm-256k/nemotron-cc-high-actual/merged_batch_001_text_document"
DATA_CACHE_PATH="/flash/project_462000963/users/${USER}/megatron_cache/"
TOKENIZER_MODEL="openeurollm/tokenizer-256k"
export TOKENIZERS_PARALLELISM=false


SEQ_LENGTH=4096
GLOBAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=1

OPTIMIZER_ARGS=(
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
    --tokenizer-model "$TOKENIZER_MODEL"
    --mock-data
    --num-workers 4
    --vocab-size 256000
)

TRAIN_ARGS=(
    --disable-bias-linear
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --bf16
    --rerun-mode disabled
    --disable-gloo-process-groups
    --overlap-grad-reduce
    --overlap-param-gather
    --train-iters 5
    --eval-iters 1
)

MODEL_ARGS=(
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
    --no-position-embedding
    --rotary-base 1000000
    --num-query-groups 4
    --kv-channels 128
    --max-position-embeddings 32768
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32    
    --use-flash-attn
    --attention-softmax-in-fp32
    --seq-length $SEQ_LENGTH
    --distributed-timeout-minutes 30
)

MOE_ARGS=(
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather
    --moe-router-force-load-balancing
    --num-experts 128
    --moe-router-topk 8
    --moe-router-dtype fp64
)

FUSED_KERNEL_ARGS=(
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
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
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
)

if (( VPP > 1)); then
    PARALLEL_ARGS+=(
        --num-layers-per-virtual-pipeline-stage $VPP
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
	--recompute-activations
    )
fi

LAST_RANK=$((SLURM_NTASKS - 1))

#PYTORCH PROFILER ARGS
PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start 4
    --profile-step-end 5
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
export CONTAINER=/scratch/project_462000963/containers/staging/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
export SINGULARITY_BIND=/boot/config-6.4.0-150600.23.73_15.0.14-cray_shasta_c,/scratch/project_462000963/,/projappl/project_462000963/,/flash/project_462000963/,/var/spool/slurmd
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
    --label \
    --cpu-bind=mask_cpu:"${BIND_MASK}" \
    "

srun $SRUN_ARGS \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launcher_laif.sh \
    $COMMAND
