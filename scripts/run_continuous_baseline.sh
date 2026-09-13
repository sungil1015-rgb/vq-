#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

# Architecture exploration intentionally uses one fixed seed. Multi-seed configs are
# kept for the final evaluation, but this script does not launch them automatically.
seed=0
output_dir="outputs/continuous_baseline/blocks_2/seed_${seed}"
mkdir -p "$output_dir"

echo "starting continuous blocks=2 seed=$seed on GPU 0"
CUDA_VISIBLE_DEVICES=0 uv run kobeni train \
    --config "config/continuous_baseline/blocks_2_seed_${seed}.toml" \
    2>&1 | tee "$output_dir/train.log"
echo "continuous baseline seed=$seed completed"
