#!/bin/bash
# Weak scaling sweep - scales global batch size (GBS) with node count

set -euo pipefail

NODES=(4 8 16 32 64 128 256)
BASE_GBS=64
SLURM_SCRIPT="slurm_scripts/qwen3_30B_A3B.slurm"

for N in "${NODES[@]}"; do  
  GBS=$((BASE_GBS * N)) COMMENT="fp8_weak_scaling_scaling" sbatch --nodes="$N" "$SLURM_SCRIPT"
done