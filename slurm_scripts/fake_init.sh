export MEGATRON_PATH=.
export PYTHONPATH=$MEGATRON_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=${TP:-4}
PP=${PP:-8}
EP=${EP:-8}
VPP=${VPP:-2}
MBS=${MBS:-4}
GBS=${GBS:-2048}

export WORLD_SIZE=256
export RANK=0
export CONTAINER="/e/project1/laionize/luukkonen1/container_cachedir/nemo-v26.02-nemotron3-super.sif"
export APPTAINER_BINDPATH="/e/project1/e-sta-openeurollm"
export APPTAINERENV_TRITON_LIBCUDA_PATH="/usr/local/cuda/compat/lib.real"

apptainer exec --nv $CONTAINER python $MEGATRON_PATH/pretrain_gpt.py \
  --fp8-param-gather \
  --fp8-recipe blockwise \
  --fp8-format hybrid \
  --recompute-granularity selective \
  --recompute-modules moe_act layernorm \
  --distributed-timeout-minutes 60 \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --expert-model-parallel-size $EP \
  --num-layers-per-virtual-pipeline-stage $VPP \
  --context-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-distributed-optimizer \
  --no-create-attention-mask-in-dataloader \
  --use-mcore-models \
  --sequence-parallel \
  --use-flash-attn \
  --disable-bias-linear \
  --micro-batch-size $MBS \
  --global-batch-size $GBS \
  --train-samples 32768 \
  --transformer-impl transformer_engine \
  --data-cache-path /tmp/data-cache \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 262000 \
  --split 99,1,0 \
  --no-mmap-bin-files \
  --num-workers 6 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --rotary-seq-len-interpolation-factor 1 \
  --normalization RMSNorm \
  --swiglu \
  --norm-epsilon 1e-06 \
  --num-layers 94 \
  --hidden-size 4096 \
  --ffn-hidden-size 12288 \
  --num-attention-heads 64 \
  --group-query-attention \
  --num-query-groups 4 \
  --qk-layernorm \
  --seq-length 4096 \
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
  --num-experts 128 \
  --moe-ffn-hidden-size 1536 \
  --moe-router-load-balancing-type aux_loss \
  --moe-router-topk 8 \
  --moe-router-pre-softmax \
  --moe-grouped-gemm \
  --moe-aux-loss-coeff 1e-3 \
  --moe-token-dispatcher-type alltoall \
  --moe-permute-fusion \
  --eval-iters 32 \
  --eval-interval 500 \
  --finetune \
  --auto-detect-ckpt-format \
  --no-ckpt-fully-parallel-save \
  --dist-ckpt-strictness log_all \
  --init-method-std 0.02 \
  --log-timers-to-tensorboard \
  --log-memory-to-tensorboard \
  --log-num-zeros-in-grad \
  --log-params-norm \
  --log-validation-ppl-to-tensorboard \
  --log-throughput \
  --log-interval 1 \
  --bf16 \
  --account-for-embedding-in-pipeline-split \
  --account-for-loss-in-pipeline-split \
  --moe-router-force-load-balancing \
  --exit-interval 5 \
  --fake-process-group \
  --record-memory-history \
  --memory-snapshot-path ./qwen3_235b_TP${TP}_PP${PP}_EP${EP}_VPP${VPP}.pickle