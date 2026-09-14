#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/layer_ablation/no_1x1_conv.toml"
    "config/architecture/layer_ablation/max_pool_downsampling.toml"
    "config/architecture/layer_ablation/batch_norm.toml"
    "config/architecture/layer_ablation/fullres_no_bottleneck.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group layer_ablation)" \
    outputs/architecture "${configs[@]}"
