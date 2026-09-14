#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/advanced_attention_head_ablation/gap_attention_residual.toml"
    "config/architecture/advanced_attention_head_ablation/four_query_attention_mean.toml"
    "config/architecture/advanced_attention_head_ablation/global_self_attention.toml"
    "config/architecture/advanced_attention_head_ablation/dwconv_attention_pool.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group advanced_attention_head_ablation)" \
    outputs/architecture "${configs[@]}"
