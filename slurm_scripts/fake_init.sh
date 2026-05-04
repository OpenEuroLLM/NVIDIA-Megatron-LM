set -euo pipefail
set -x
export MEGATRON_PATH=.
export PYTHONPATH=$MEGATRON_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=8

MODEL_SIZE=${MODEL_SIZE:-30B}

case "$MODEL_SIZE" in
235B)
TP=${TP:-2}
PP=${PP:-8}
EP=${EP:-16}
VPP=${VPP:-6}
NLAYERS=${NLAYERS:-94}
NHIDDEN=${NHIDDEN:-4096}
NHEADS=${NHEADS:-64}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-12288}
NUM_EXPERTS=${NUM_EXPERTS:-128}
MOE_FFN_SIZE=${MOE_FFN_SIZE:-1536}
#
#--account-for-embedding-in-pipeline-split \
#--account-for-loss-in-pipeline-split \
;;
30B)
TP=${TP:-1}
PP=${PP:-2}
EP=${EP:-4}
VPP=${VPP:-1}
NLAYERS=${NLAYERS:-48}
NHIDDEN=${NHIDDEN:-2048}
NHEADS=${NHEADS:-32}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-6144}
NUM_EXPERTS=${NUM_EXPERTS:-128}
MOE_FFN_SIZE=${MOE_FFN_SIZE:-768}
;;
*)
echo "Unknown MODEL_SIZE=$MODEL_SIZE" >&2
exit 1
;;
esac

MBS=${MBS:-1}
GBS=${GBS:-4096}

VPP_ARG=()
if [[ "${PP}" -gt 1 ]]; then
  VPP_ARG=(--num-layers-per-virtual-pipeline-stage "${VPP}")
fi

export NVTE_CPU_OFFLOAD_V1=1
export WORLD_SIZE=128
export RANK=0
export CONTAINER="/e/project1/e-sta-openeurollm/container/nemo_26.04.sif"
export APPTAINER_BINDPATH="/e/project1/e-sta-openeurollm"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

apptainer exec --nv $CONTAINER python $MEGATRON_PATH/pretrain_gpt.py \
  --use-flash-attn \
  --fp8-format hybrid \
  --fp8-recipe blockwise \
  --fp8-param-gather \
  --use-precision-aware-optimizer \
  --exp-avg-dtype fp8 \
  --exp-avg-sq-dtype fp8 \
  --moe-router-dtype fp32 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --distributed-timeout-minutes 60 \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --expert-model-parallel-size $EP \
  "${VPP_ARG[@]}" \
  --context-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-distributed-optimizer \
  --no-create-attention-mask-in-dataloader \
  --attention-softmax-in-fp32 \
  --disable-bias-linear \
  --micro-batch-size $MBS \
  --global-batch-size $GBS \
  --train-samples 32768 \
  --transformer-impl transformer_engine \
  --data-cache-path /tmp/data-cache \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 256000 \
  --split 100,0,0 \
  --no-mmap-bin-files \
  --num-workers 8 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --normalization RMSNorm \
  --swiglu \
  --norm-epsilon 1e-06 \
  --num-layers $NLAYERS \
  --hidden-size $NHIDDEN \
  --ffn-hidden-size $FFN_HIDDEN_SIZE \
  --num-attention-heads $NHEADS \
  --group-query-attention \
  --num-query-groups 4 \
  --kv-channels 128 \
  --qk-layernorm \
  --seq-length 4096 \
  --no-load-optim \
  --max-position-embeddings 4096 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --clip-grad 1.0 \
  --weight-decay 0.1 \
  --lr-decay-samples 255126953 \
  --lr-warmup-samples 162761 \
  --lr 1.2e-4 \
  --min-lr 1.2e-5 \
  --lr-decay-style cosine \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --num-experts $NUM_EXPERTS \
  --moe-ffn-hidden-size $MOE_FFN_SIZE \
  --moe-router-load-balancing-type aux_loss \
  --moe-router-topk 8 \
  --moe-router-pre-softmax \
  --moe-grouped-gemm \
  --moe-aux-loss-coeff 1e-3 \
  --moe-token-dispatcher-type alltoall \
  --moe-permute-fusion \
  --eval-iters 32 \
  --eval-interval 500 \
  --auto-detect-ckpt-format \
  --no-ckpt-fully-parallel-save \
  --init-method-std 0.02 \
  --log-throughput \
  --log-interval 1 \
  --bf16 \
  --moe-router-force-load-balancing \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl te \
  --exit-interval 5 \
  --fake-process-group \
  --record-memory-history \
  --memory-snapshot-path ./qwen3_${MODEL_SIZE}_TP${TP}_PP${PP}_EP${EP}_VPP${VPP}_MBS${MBS}_.pickle
