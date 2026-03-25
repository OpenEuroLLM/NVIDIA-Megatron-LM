#!/bin/bash

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export AITER_JIT_DIR=/tmp/aiter_cache/$SLURM_PROCID
if [[ -d $AITER_JIT_DIR ]]; then
    echo "Rank $SLURM_PROCID --> Removing existing AITER_JIT_DIR at $AITER_JIT_DIR"
fi
rm -rf $AITER_JIT_DIR
# Report affinity
#echo "Rank $SLURM_PROCID --> $(taskset -p \$\$)"

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG_FILE=logs/nccl/sam_nccl_info_rank_${SLURM_PROCID}.log
export TORCH_FR_DUMP_TEMP_FILE=logs/fr/nccl_trace_rank_${SLURM_PROCID}.log

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

python3 -u "$@"