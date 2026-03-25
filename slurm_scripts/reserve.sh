#!/bin/bash
salloc --bell \
    --nodes=$1 \
    --ntasks-per-node=$2 \
    --time=$3 \
    -J $4 \
    --partition=$5 \
    --account=project_462000963 \
    --gpus-per-node=8 \
    --cpus-per-task=7 \
    --mem=0 \
    --exclusive \
