#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

readonly configs=(
    "config/architecture/encoder_ablation/remove_middle_conv.toml"
    "config/architecture/encoder_ablation/pool_after_first_conv.toml"
    "config/architecture/encoder_ablation/silu.toml"
    "config/architecture/encoder_ablation/channels_256.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group encoder_ablation)" \
    outputs/architecture "${configs[@]}"
