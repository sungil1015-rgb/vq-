#!/usr/bin/env bash

four_gpu_run_group() {
    local suffix="$1"
    TZ=Asia/Seoul date "+%m%d_%H%M%S_$suffix"
}

run_four_gpu_experiments() {
    if (($# != 6)); then
        echo "run_four_gpu_experiments requires: run_group summary_dir config0 config1 config2 config3" >&2
        return 2
    fi

    local run_group="$1"
    local summary_dir="$2"
    shift 2
    local configs=("$@")
    local pids=()
    local gpu
    local status=0

    echo "run group=$run_group"
    for gpu in 0 1 2 3; do
        (
            echo "starting gpu=$gpu config=${configs[$gpu]}"
            CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run kobeni train \
                --config "${configs[$gpu]}" \
                --run-group "$run_group"
            echo "completed gpu=$gpu config=${configs[$gpu]}"
        ) &
        pids+=("$!")
        echo "launched gpu=$gpu config=${configs[$gpu]} pid=${pids[-1]}"
    done

    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    if ((status != 0)); then
        echo "one or more experiments failed in run group $run_group" >&2
        return "$status"
    fi

    uv run kobeni summarize-experiments --run-dir "$summary_dir"
    echo "all experiments completed: $run_group"
}
