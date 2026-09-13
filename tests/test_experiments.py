import json
from dataclasses import asdict
from pathlib import Path

import pytest

from kobeni.analysis.experiments import summarize_experiments
from kobeni.config import DataConfig, ExperimentConfig, ModelConfig, TrackingConfig, TrainConfig


def test_summarize_experiments_uses_miou_for_segmentation_selection(tmp_path: Path) -> None:
    output_dir = tmp_path / "segmentation_direct" / "seed_0"
    output_dir.mkdir(parents=True)
    config = ExperimentConfig(
        model=ModelConfig(task="segmentation_direct", num_classes=150),  # type: ignore[arg-type]
        data=DataConfig(dataset="ade20k", batch_size=2, num_workers=0),
        train=TrainConfig(epochs=2, output_dir=str(output_dir)),
        tracking=TrackingConfig(enabled=True, run_name="direct"),
    )
    metrics = {
        "loss": 1.0,
        "accuracy": 0.7,
        "pixel_accuracy": 0.7,
        "mean_iou": 0.4,
        "classification_loss": 1.0,
        "vq_loss": 0.1,
        "commitment_loss": 0.05,
        "codebook_loss": 0.05,
        "reconstruction_loss": 0.0,
    }
    summary = {
        "selection_metric": "mean_iou",
        "best_selection_epoch": 2,
        "best_eval_score": 0.44,
        "best_selection_metrics": {**metrics, "mean_iou": 0.44},
        "epochs_completed": 2,
        "elapsed_seconds": 12.0,
        "best_accuracy_epoch": 2,
        "best_eval_accuracy": 0.72,
        "best_accuracy_metrics": metrics,
        "minimum_classification_loss_epoch": 2,
        "minimum_eval_classification_loss": 1.0,
        "final_epoch": 2,
        "final_train_metrics": metrics,
        "final_eval_metrics": metrics,
        "final_accuracy_generalization_gap": 0.0,
        "final_classification_loss_generalization_gap": 0.0,
        "final_timing": {"epoch_seconds": 6.0, "peak_gpu_memory_mb": 100.0},
    }
    (output_dir / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps({"status": "completed", "source_fingerprint": "test"}), encoding="utf-8"
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps({"total": 10, "trainable": 10}), encoding="utf-8"
    )
    (output_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = summarize_experiments(tmp_path)

    row = report["runs"][0]
    group = report["groups"][0]
    assert row["task"] == "segmentation_direct"
    assert row["selection_metric"] == "mean_iou"
    assert row["best_eval_mean_iou"] == pytest.approx(0.44)
    assert row["final_eval_selection_score"] == pytest.approx(0.4)
    assert group["best_selection_score_mean"] == pytest.approx(0.44)
    assert group["best_mean_iou_mean"] == pytest.approx(0.44)
