#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/latent_width_ablation/latent_dim_64.toml"
    "config/architecture/latent_width_ablation/latent_dim_128.toml"
    "config/architecture/latent_width_ablation/latent_dim_256.toml"
    "config/architecture/latent_width_ablation/encoder_width_384.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group latent_width_ablation)" \
    outputs/architecture "${configs[@]}"
