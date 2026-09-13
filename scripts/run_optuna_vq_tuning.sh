#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

study_config="${STUDY_CONFIG:-config/tuning/vq_optuna.toml}"
run_group="${1:-$(TZ=Asia/Seoul date +%m%d_%H%M%S)_vq_optuna}"
n_trials="${N_TRIALS_PER_GPU:-}"
study_dir="outputs/optuna/$run_group"
mkdir -p "$study_dir/logs"

echo "run group=$run_group"
echo "study config=$study_config"
uv run kobeni prepare-data --config config/architecture/vq_baseline.toml

declare -a pids=()
cleanup() {
    if ((${#pids[@]} > 0)); then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

for gpu in 0 1 2 3; do
    command=(
        uv run kobeni tune
        --study-config "$study_config"
        --run-group "$run_group"
        --worker-id "$gpu"
    )
    if [[ -n "$n_trials" ]]; then
        command+=(--n-trials "$n_trials")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${command[@]}" \
        >"$study_dir/logs/gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "launched gpu=$gpu pid=${pids[-1]} log=$study_dir/logs/gpu_${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
trap - INT TERM

uv run kobeni summarize-tuning \
    --study-config "$study_config" \
    --run-group "$run_group"

if ((status != 0)); then
    echo "one or more Optuna workers failed; inspect $study_dir/logs" >&2
    exit "$status"
fi
echo "all Optuna workers completed: $study_dir"
