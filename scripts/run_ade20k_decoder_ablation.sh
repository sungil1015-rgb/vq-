#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

run_group="$(TZ=Asia/Seoul date +%m%d_%H%M%S)_ade20k_decoder_ablation"
echo "run group=$run_group"
uv run kobeni prepare-data --config config/segmentation/decoder_ablation/a_direct.toml

readonly configs=(
    "config/segmentation/decoder_ablation/a_direct.toml"
    "config/segmentation/decoder_ablation/b_linear.toml"
    "config/segmentation/decoder_ablation/c_relu.toml"
    "config/segmentation/decoder_ablation/d_batchnorm.toml"
)
declare -a pids=()

run_experiment() {
    local gpu="$1"
    local config_path="$2"
    echo "starting gpu=$gpu config=$config_path"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run kobeni train \
        --config "$config_path" \
        --run-group "$run_group"
    echo "completed gpu=$gpu config=$config_path"
}

for gpu in 0 1 2 3; do
    run_experiment "$gpu" "${configs[$gpu]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if ((status != 0)); then
    echo "one or more ADE20K decoder ablations failed" >&2
    exit "$status"
fi

uv run kobeni summarize-experiments --run-dir outputs/segmentation
echo "all ADE20K decoder ablations completed"
