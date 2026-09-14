#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

uv run kobeni prepare-data --config config/segmentation/decoder_ablation/a_direct.toml
readonly configs=(
    "config/segmentation/decoder_ablation/a_direct.toml"
    "config/segmentation/decoder_ablation/b_linear.toml"
    "config/segmentation/decoder_ablation/c_relu.toml"
    "config/segmentation/decoder_ablation/d_batchnorm.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group ade20k_decoder_ablation)" \
    outputs/segmentation "${configs[@]}"
