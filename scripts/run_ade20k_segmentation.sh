#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

uv run kobeni prepare-data --config config/segmentation/ade20k_direct.toml

run_group="$(TZ=Asia/Seoul date +%m%d_%H%M%S)_ade20k_vq_segmentation"
echo "run group=$run_group"

run_experiment() {
    local gpu="$1"
    local config_path="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run kobeni train \
        --config "$config_path" \
        --run-group "$run_group"
}

run_experiment 0 config/segmentation/ade20k_direct.toml &
direct_pid=$!
run_experiment 1 config/segmentation/ade20k_decoder.toml &
decoder_pid=$!

status=0
wait "$direct_pid" || status=1
wait "$decoder_pid" || status=1
if ((status != 0)); then
    echo "one or more ADE20K segmentation runs failed" >&2
    exit "$status"
fi

uv run kobeni summarize-experiments --run-dir outputs/segmentation
