#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/pooling_ablation/max_pool.toml"
    "config/architecture/pooling_ablation/avg_pool.toml"
    "config/architecture/pooling_ablation/max_then_avg_pool.toml"
    "config/architecture/pooling_ablation/avg_then_max_pool.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group pooling_ablation)" \
    outputs/architecture "${configs[@]}"
