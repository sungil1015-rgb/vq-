#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/head_ablation/gap.toml"
    "config/architecture/head_ablation/pool_2x2.toml"
    "config/architecture/head_ablation/gap_pool_2x2_concat.toml"
    "config/architecture/head_ablation/dwconv3x3.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group head_ablation)" \
    outputs/architecture "${configs[@]}"
