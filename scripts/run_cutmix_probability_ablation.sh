#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/cutmix_probability_ablation/crop_flip.toml"
    "config/architecture/cutmix_probability_ablation/crop_flip_cutmix_p025.toml"
    "config/architecture/cutmix_probability_ablation/crop_flip_cutmix_p050.toml"
    "config/architecture/cutmix_probability_ablation/crop_flip_cutmix_p075.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group cutmix_probability_ablation)" \
    outputs/architecture "${configs[@]}"
