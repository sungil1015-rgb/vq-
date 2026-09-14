#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/codebook_size_ablation/k_256.toml"
    "config/architecture/codebook_size_ablation/k_512.toml"
    "config/architecture/codebook_size_ablation/k_1024.toml"
    "config/architecture/codebook_size_ablation/k_2048.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group codebook_size_ablation)" \
    outputs/architecture "${configs[@]}"
