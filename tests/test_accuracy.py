import json
from pathlib import Path

import pytest

from kobeni.analysis.accuracy import summarize_accuracy


def test_summarize_accuracy_aggregates_seeds(tmp_path: Path) -> None:
    for seed, accuracies in ((0, (0.7, 0.8)), (1, (0.75, 0.78))):
        seed_dir = tmp_path / f"seed_{seed}"
        seed_dir.mkdir()
        rows = [
            {"epoch": epoch, "eval": {"accuracy": accuracy}}
            for epoch, accuracy in enumerate(accuracies, start=1)
        ]
        (seed_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    summary = summarize_accuracy(tmp_path)

    assert summary["num_seeds"] == 2
    assert summary["best_accuracy_mean"] == pytest.approx(0.79)
    assert summary["runs"][0]["best_epoch"] == 2
    assert (tmp_path / "accuracy_summary.json").is_file()
