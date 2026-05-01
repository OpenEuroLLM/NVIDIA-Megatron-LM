#!/bin/bash
# Weak scaling sweep - scales global batch size (GBS) with node count

set -euo pipefail

NODES=(2 4 8 16 32 64 128 256)
SLURM_SCRIPT="slurm_scripts/qwen3_30B_A3B.slurm"

for N in "${NODES[@]}"; do
  NUM_SMS=40 VPP=2 FP=8 MBS=4 COMMENT="strong_scaling_FP8" sbatch --nodes="$N" "$SLURM_SCRIPT"
done