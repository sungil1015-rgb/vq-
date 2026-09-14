#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/augmentation_ablation/crop.toml"
    "config/architecture/augmentation_ablation/flip.toml"
    "config/architecture/augmentation_ablation/cutmix.toml"
    "config/architecture/augmentation_ablation/crop_flip_cutmix.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group augmentation_ablation)" \
    outputs/architecture "${configs[@]}"
