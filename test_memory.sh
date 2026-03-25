CUDA_DEVICE_MAX_CONNECTIONS=1 \
WORLD_SIZE=512 \
AITER_JIT_DIR=/tmp/aiter \
CXX=clang++ CC=clang \
PYTHONPATH="." \
srun -A project_462000963 -p dev-g --gpus=1 -c 7 -J estimate_memory --mem=50G -t 00:15:00 singularity exec \
    -B $PWD \
    -B /scratch/project_462000963 \
    /scratch/project_462000394/containers/for-turkunlp-team/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-te-dockerhash-560b76ceab56.sif \
python tools/report_theoretical_memory.py \
    --use-mcore-models \
    --num-layers 94 \
    --hidden-size 4096 \
    --moe-ffn-hidden-size 1536 \
    --num-attention-heads 64 \
    --num-query-groups 4 \
    --group-query-attention \
    --ffn-hidden-size 10880 \
    --kv-channels 128 \
    --seq-length 8192 \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 16 \
    --decoder-first-pipeline-num-layers 5 \
    --decoder-last-pipeline-num-layers 5 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --num-experts 64 \
    --moe-router-topk 8 \
    --moe-router-dtype fp64 \
    --bf16 \
    --use-distributed-optimizer \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --qk-layernorm \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --position-embedding-type rope \
    --max-position-embeddings 32768 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --rotary-base 1000000 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model EleutherAI/gpt-neox-20b \
    --recompute-activations \
    --recompute-granularity selective \
    --use-flash-attn