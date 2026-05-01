#!/bin/bash
salloc --bell \
    --nodes=$1 \
    --ntasks-per-node=$2 \
    --time=$3 \
    -J $4 \
    --gpus-per-node=4 \
    --partition=booster \
    --account=e-sta-openeurollm \
    --mem=0 \
    --exclusive \
