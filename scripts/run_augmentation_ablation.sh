#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

run_group="$(TZ=Asia/Seoul date +%m%d_%H%M%S)_augmentation_ablation"
echo "run group=$run_group"

readonly configs=(
    "config/architecture/augmentation_ablation/crop.toml"
    "config/architecture/augmentation_ablation/flip.toml"
    "config/architecture/augmentation_ablation/cutmix.toml"
    "config/architecture/augmentation_ablation/crop_flip_cutmix.toml"
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
    echo "launched gpu=$gpu config=${configs[$gpu]} pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if ((status != 0)); then
    echo "one or more augmentation ablations failed" >&2
    exit "$status"
fi

uv run kobeni summarize-experiments --run-dir outputs/architecture
echo "all augmentation ablations completed"
