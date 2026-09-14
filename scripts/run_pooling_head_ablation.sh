#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/pooling_head_ablation/gap.toml"
    "config/architecture/pooling_head_ablation/dwconv3x3.toml"
    "config/architecture/pooling_head_ablation/learned_weighted_pool.toml"
    "config/architecture/pooling_head_ablation/single_query_attention_pool.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group pooling_head_ablation)" \
    outputs/architecture "${configs[@]}"
