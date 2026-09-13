#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

readonly seeds=(0 1 2 3)
declare -a pids=()

run_pair() {
    local seed="$1"
    local gpu="$2"
    local variant
    local config_path
    local output_dir

    for variant in continuous_bottleneck vq; do
        config_path="config/phase1/${variant}.toml"
        output_dir="outputs/comparison/block2_200/${variant}/seed_${seed}"
        mkdir -p "$output_dir"

        echo "starting variant=$variant seed=$seed on GPU $gpu"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" uv run kobeni train \
            --config "$config_path" \
            --seed "$seed" \
            --output-dir "$output_dir" \
            2>&1 | tee "$output_dir/train.log"
        echo "completed variant=$variant seed=$seed on GPU $gpu"
    done
}

for seed in "${seeds[@]}"; do
    run_pair "$seed" "$seed" &
    pids+=("$!")
    echo "launched seed=$seed pair on GPU $seed (pid=${pids[-1]})"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if ((status != 0)); then
    echo "one or more runs failed; inspect each seed's train.log" >&2
    exit "$status"
fi

echo "all 200-epoch continuous/VQ runs completed"
