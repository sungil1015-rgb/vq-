from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(config: dict[str, Any], source_fingerprint: str | None) -> str:
    train = {
        key: value
        for key, value in config["train"].items()
        if key not in {"seed", "output_dir", "resume_from"}
    }
    payload = {
        "model": config["model"],
        "data": config["data"],
        "train": train,
        "source_fingerprint": source_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _run_row(summary_path: Path, root: Path) -> dict[str, Any] | None:
    output_dir = summary_path.parent
    required = [
        output_dir / "config.json",
        output_dir / "run_metadata.json",
        output_dir / "model_summary.json",
    ]
    if not all(path.is_file() for path in required):
        return None

    summary = _load(summary_path)
    config = _load(required[0])
    metadata = _load(required[1])
    model_summary = _load(required[2])
    model = config["model"]
    data = config["data"]
    train = config["train"]
    tracking = config.get("tracking", {})
    final_train = summary["final_train_metrics"]
    final_eval = summary["final_eval_metrics"]
    timing = summary.get("final_timing", {})
    selection_metric = summary.get("selection_metric", "accuracy")
    best_selection_metrics = summary.get("best_selection_metrics", {})
    fingerprint = metadata.get("source_fingerprint")

    return {
        "run_path": str(output_dir.relative_to(root)),
        "run_name": tracking.get("run_name", metadata.get("run_name")),
        "status": metadata.get("status"),
        "started_at": metadata.get("started_at"),
        "source_fingerprint": fingerprint,
        "experiment_signature": _signature(config, fingerprint),
        "task": model.get("task", "classification"),
        "dataset": data["dataset"],
        "random_crop": data.get("random_crop", True),
        "random_horizontal_flip": data.get("random_horizontal_flip", True),
        "cutmix_alpha": data.get("cutmix_alpha", 0.0),
        "cutmix_probability": data.get("cutmix_probability", 1.0),
        "variant": model["variant"],
        "seed": train["seed"],
        "encoder_blocks": model["encoder_blocks"],
        "encoder_channels": model["encoder_channels"],
        "latent_dim": model["latent_dim"],
        "codebook_size": model["codebook_size"],
        "commitment_weight": model["commitment_weight"],
        "codebook_weight": model.get("codebook_weight", 1.0),
        "attention_temperature": model.get("attention_temperature"),
        "attention_residual_alpha_init": model.get("attention_residual_alpha_init"),
        "reconstruction": model["reconstruction"],
        "use_bottleneck": model.get("use_bottleneck", True),
        "normalization": model.get("normalization", "group"),
        "downsampling": model.get("downsampling", "strided_conv"),
        "activation": model.get("activation", "relu"),
        "remove_middle_conv": model.get("remove_middle_conv", False),
        "pool_after_first_conv": model.get("pool_after_first_conv", False),
        "head": model.get("head", "gap"),
        "segmentation_decoder_normalization": model.get("segmentation_decoder_normalization"),
        "segmentation_decoder_activation": model.get("segmentation_decoder_activation"),
        "segmentation_upsample_mode": model.get("segmentation_upsample_mode"),
        "epochs": train["epochs"],
        "learning_rate": train["learning_rate"],
        "scheduler": train["scheduler"],
        "optimizer": train.get("optimizer", "adamw"),
        "weight_decay": train["weight_decay"],
        "lambda_cls": train.get("lambda_cls", 1.0),
        "lambda_vq": train["lambda_vq"],
        "lambda_rec": train["lambda_rec"],
        "label_smoothing": train.get("label_smoothing", 0.0),
        "batch_size": data["batch_size"],
        "gradient_accumulation_steps": train.get("gradient_accumulation_steps", 1),
        "mixed_precision": train.get("mixed_precision", False),
        "tf32": train.get("tf32", False),
        "channels_last": train.get("channels_last", False),
        "effective_batch_size": data["batch_size"]
        * train.get("gradient_accumulation_steps", 1),
        "total_parameters": model_summary["total"],
        "trainable_parameters": model_summary["trainable"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "selection_metric": selection_metric,
        "best_selection_epoch": summary.get("best_selection_epoch"),
        "best_eval_score": summary.get("best_eval_score"),
        "best_eval_mean_iou": best_selection_metrics.get("mean_iou"),
        "best_eval_pixel_accuracy": best_selection_metrics.get("pixel_accuracy"),
        "best_accuracy_epoch": summary["best_accuracy_epoch"],
        "best_eval_accuracy": summary["best_eval_accuracy"],
        "minimum_loss_epoch": summary["minimum_classification_loss_epoch"],
        "minimum_eval_classification_loss": summary["minimum_eval_classification_loss"],
        "final_train_accuracy": final_train["accuracy"],
        "final_eval_accuracy": final_eval["accuracy"],
        "final_eval_top5_accuracy": final_eval.get("top5_accuracy"),
        "final_eval_loss": final_eval["loss"],
        "final_eval_classification_loss": final_eval["classification_loss"],
        "final_eval_mean_iou": final_eval.get("mean_iou"),
        "final_eval_pixel_accuracy": final_eval.get("pixel_accuracy"),
        "final_eval_selection_score": final_eval.get(selection_metric),
        "final_eval_vq_loss": final_eval.get("vq_loss"),
        "final_eval_commitment_loss": final_eval.get("commitment_loss"),
        "final_eval_codebook_loss": final_eval.get("codebook_loss"),
        "final_active_codes": final_eval.get("active_codes"),
        "final_perplexity": final_eval.get("perplexity"),
        "final_dead_code_fraction": final_eval.get("dead_code_fraction"),
        "final_quantization_error": final_eval.get("quantization_error"),
        "accuracy_generalization_gap": summary["final_accuracy_generalization_gap"],
        "classification_loss_generalization_gap": summary[
            "final_classification_loss_generalization_gap"
        ],
        "final_epoch_seconds": timing.get("epoch_seconds"),
        "peak_gpu_memory_mb": timing.get("peak_gpu_memory_mb"),
    }


def _optional_mean(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return statistics.mean(value for value in values if value is not None)


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["experiment_signature"], []).append(row)

    summaries = []
    for signature, members in sorted(grouped.items()):
        best_accuracy = [member["best_eval_accuracy"] for member in members]
        final_accuracy = [member["final_eval_accuracy"] for member in members]
        best_selection = [member["best_eval_score"] for member in members]
        final_selection = [member["final_eval_selection_score"] for member in members]
        best_mean_iou = [member["best_eval_mean_iou"] for member in members]
        summaries.append(
            {
                "experiment_signature": signature,
                "run_name": members[0]["run_name"],
                "task": members[0]["task"],
                "dataset": members[0]["dataset"],
                "variant": members[0]["variant"],
                "selection_metric": members[0]["selection_metric"],
                "num_runs": len(members),
                "seeds": [member["seed"] for member in members],
                "best_selection_score_mean": statistics.mean(best_selection),
                "best_selection_score_sample_std": (
                    statistics.stdev(best_selection) if len(best_selection) > 1 else 0.0
                ),
                "final_selection_score_mean": statistics.mean(final_selection),
                "final_selection_score_sample_std": (
                    statistics.stdev(final_selection) if len(final_selection) > 1 else 0.0
                ),
                "best_mean_iou_mean": _optional_mean(best_mean_iou),
                "best_accuracy_mean": statistics.mean(best_accuracy),
                "best_accuracy_sample_std": (
                    statistics.stdev(best_accuracy) if len(best_accuracy) > 1 else 0.0
                ),
                "final_accuracy_mean": statistics.mean(final_accuracy),
                "final_accuracy_sample_std": (
                    statistics.stdev(final_accuracy) if len(final_accuracy) > 1 else 0.0
                ),
                "elapsed_seconds_mean": statistics.mean(
                    member["elapsed_seconds"] for member in members
                ),
            }
        )
    return summaries


def summarize_experiments(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    rows = []
    for summary_path in sorted(run_dir.rglob("run_summary.json")):
        row = _run_row(summary_path, run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No tracked runs found under {run_dir}")

    report = {
        "run_dir": str(run_dir),
        "num_runs": len(rows),
        "groups": _group_rows(rows),
        "runs": rows,
    }
    (run_dir / "experiment_table.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (run_dir / "experiment_table.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report
