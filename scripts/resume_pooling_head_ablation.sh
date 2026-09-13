#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
    echo "usage: $0 <run_group>" >&2
    exit 2
fi

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

run_group="$1"
group_dir="outputs/architecture/$run_group"

readonly configs=(
    "config/architecture/pooling_head_ablation/gap.toml"
    "config/architecture/pooling_head_ablation/dwconv3x3.toml"
    "config/architecture/pooling_head_ablation/learned_weighted_pool.toml"
    "config/architecture/pooling_head_ablation/single_query_attention_pool.toml"
)
readonly run_names=(
    "vq_pooling_head_gap"
    "vq_pooling_head_dwconv3x3"
    "vq_pooling_head_learned_weighted_pool"
    "vq_pooling_head_single_query_attention_pool"
)
declare -a pids=()

run_experiment() {
    local gpu="$1"
    local config_path="$2"
    local run_name="$3"
    local checkpoint="$group_dir/$run_name/seed_0/last.pt"
    local command=(uv run kobeni train --config "$config_path" --run-group "$run_group")

    if [[ -f "$checkpoint" ]]; then
        command+=(--resume-from "$checkpoint")
        echo "resuming gpu=$gpu config=$config_path checkpoint=$checkpoint"
    else
        echo "starting fresh gpu=$gpu config=$config_path"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${command[@]}"
    echo "completed gpu=$gpu config=$config_path"
}

for gpu in 0 1 2 3; do
    run_experiment "$gpu" "${configs[$gpu]}" "${run_names[$gpu]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if ((status != 0)); then
    echo "one or more pooling-head resumes failed" >&2
    exit "$status"
fi

uv run kobeni summarize-experiments --run-dir outputs/architecture
echo "all pooling-head runs completed"
