from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
import torchvision
from torch import nn

from kobeni.config import ExperimentConfig

_KST = ZoneInfo("Asia/Seoul")


def _now_kst() -> datetime:
    return datetime.now(_KST)


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def resolve_output_dir(config: ExperimentConfig) -> Path:
    base = Path(config.train.output_dir)
    if not config.tracking.enabled:
        return base

    timestamp = _now_kst().strftime("%m%d_%H%M%S")
    run_name = _safe_path_component(config.tracking.run_name)
    group_name = config.tracking.run_group
    if group_name is not None:
        return base / _safe_path_component(group_name) / run_name / f"seed_{config.train.seed}"
    return base / timestamp / run_name / f"seed_{config.train.seed}"


def _source_manifest() -> tuple[dict[str, str], str]:
    package_root = Path(__file__).resolve().parent
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root.parent).as_posix()
        contents = path.read_bytes()
        file_hash = hashlib.sha256(contents).hexdigest()
        files[relative] = file_hash
        combined.update(relative.encode("utf-8"))
        combined.update(contents)
    return files, combined.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty(project_root: Path) -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _parameter_summary(model: nn.Module) -> dict[str, Any]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "by_component": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in model.named_children()
        },
        "model_repr": str(model),
    }


def _snapshot_source(output_dir: Path) -> None:
    package_root = Path(__file__).resolve().parent
    destination_root = output_dir / "source_snapshot"
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root.parent)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_run_artifacts(
    output_dir: Path,
    config: ExperimentConfig,
    model: nn.Module,
    device: torch.device,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_files, source_fingerprint = _source_manifest()
    parameter_summary = _parameter_summary(model)
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": _now_kst().isoformat(),
        "run_name": config.tracking.run_name,
        "output_dir": str(output_dir),
        "command": sys.argv,
        "device": str(device),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": _git_commit(project_root),
        "git_dirty": _git_dirty(project_root),
        "source_fingerprint": source_fingerprint,
        "source_files": source_files,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "model_summary.json").write_text(
        json.dumps(parameter_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if config.tracking.enabled and config.tracking.save_source_snapshot:
        _snapshot_source(output_dir)


def write_run_summary(
    output_dir: Path,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
    selection_metric: str = "accuracy",
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty training history")

    best_selection = max(records, key=lambda row: row["eval"][selection_metric])
    best_accuracy = max(records, key=lambda row: row["eval"]["accuracy"])
    best_classification_loss = min(records, key=lambda row: row["eval"]["classification_loss"])
    final = records[-1]
    summary = {
        "selection_metric": selection_metric,
        "best_selection_epoch": best_selection["epoch"],
        "best_eval_score": best_selection["eval"][selection_metric],
        "best_selection_metrics": best_selection["eval"],
        "epochs_completed": len(records),
        "elapsed_seconds": elapsed_seconds,
        "best_accuracy_epoch": best_accuracy["epoch"],
        "best_eval_accuracy": best_accuracy["eval"]["accuracy"],
        "best_accuracy_metrics": best_accuracy["eval"],
        "minimum_classification_loss_epoch": best_classification_loss["epoch"],
        "minimum_eval_classification_loss": best_classification_loss["eval"]["classification_loss"],
        "final_epoch": final["epoch"],
        "final_train_metrics": final["train"],
        "final_eval_metrics": final["eval"],
        "final_accuracy_generalization_gap": (
            final["train"]["accuracy"] - final["eval"]["accuracy"]
        ),
        "final_classification_loss_generalization_gap": (
            final["eval"]["classification_loss"] - final["train"]["classification_loss"]
        ),
        "final_timing": final.get("timing", {}),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "completed",
            "completed_at": _now_kst().isoformat(),
            "elapsed_seconds": elapsed_seconds,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def mark_run_status(output_dir: Path, status: str, **details: Any) -> None:
    """Mark an interrupted run without discarding its checkpoints or metrics."""
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": status,
            "completed_at": _now_kst().isoformat(),
            **details,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
