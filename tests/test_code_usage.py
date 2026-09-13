import csv
import json
from pathlib import Path

import pytest
import torch

from kobeni.analysis.code_usage import summarize_counts, write_usage_artifacts


def test_summarize_counts_reports_class_distributions() -> None:
    counts = torch.tensor([[8, 2, 0], [0, 5, 5]])

    summary = summarize_counts(counts, ("class_a", "class_b"))

    assert summary["global"]["token_count"] == 20
    assert summary["global"]["active_codes"] == 3
    assert summary["classes"][0]["active_codes"] == 2
    assert summary["classes"][0]["top_codes"][0] == {
        "code_id": 0,
        "probability": pytest.approx(0.8),
    }


def test_write_usage_artifacts(tmp_path: Path) -> None:
    counts = torch.tensor([[8, 2, 0], [0, 5, 5]])

    write_usage_artifacts(tmp_path, counts, ("class_a", "class_b"), {"split": "test"})

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "code_usage.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert summary["metadata"]["split"] == "test"
    assert len(rows) == 6
    assert (tmp_path / "code_usage_heatmap.svg").read_text().startswith("<svg")
    assert torch.equal(torch.load(tmp_path / "counts.pt"), counts)
