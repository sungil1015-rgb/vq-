from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def summarize_accuracy(run_dir: Path) -> dict[str, Any]:
    records = []
    seed_dirs = sorted(
        run_dir.glob("seed_*"),
        key=lambda path: int(path.name.removeprefix("seed_")),
    )
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        epochs = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if not epochs:
            continue
        best = max(epochs, key=lambda row: row["eval"]["accuracy"])
        records.append(
            {
                "seed": int(seed_dir.name.removeprefix("seed_")),
                "best_epoch": best["epoch"],
                "best_accuracy": best["eval"]["accuracy"],
                "final_accuracy": epochs[-1]["eval"]["accuracy"],
            }
        )
    if not records:
        raise FileNotFoundError(f"No completed metrics found under {run_dir}")

    best_accuracies = [record["best_accuracy"] for record in records]
    final_accuracies = [record["final_accuracy"] for record in records]
    summary = {
        "run_dir": str(run_dir),
        "num_seeds": len(records),
        "runs": records,
        "best_accuracy_mean": statistics.mean(best_accuracies),
        "best_accuracy_sample_std": (
            statistics.stdev(best_accuracies) if len(best_accuracies) > 1 else 0.0
        ),
        "final_accuracy_mean": statistics.mean(final_accuracies),
        "final_accuracy_sample_std": (
            statistics.stdev(final_accuracies) if len(final_accuracies) > 1 else 0.0
        ),
    }
    (run_dir / "accuracy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
