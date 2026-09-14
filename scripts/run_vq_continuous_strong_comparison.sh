#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/vq_continuous_strong_comparison/vq_seed_0.toml"
    "config/architecture/vq_continuous_strong_comparison/continuous_seed_0.toml"
    "config/architecture/vq_continuous_strong_comparison/vq_seed_1.toml"
    "config/architecture/vq_continuous_strong_comparison/continuous_seed_1.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group vq_continuous_strong)" \
    outputs/architecture "${configs[@]}"
