#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source scripts/lib/four_gpu_ablation.sh

uv run kobeni prepare-data --config config/architecture/vq_baseline.toml
readonly configs=(
    "config/optuna_followup/conservative_200/lr500_lambda100.toml"
    "config/optuna_followup/conservative_200/lr500_lambda140.toml"
    "config/optuna_followup/conservative_200/lr600_lambda100.toml"
    "config/optuna_followup/conservative_200/lr600_lambda140.toml"
)
run_four_gpu_experiments "$(four_gpu_run_group vq_conservative_200)" \
    outputs/optuna_followup "${configs[@]}"
