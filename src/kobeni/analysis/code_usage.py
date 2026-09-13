from __future__ import annotations

import csv
import html
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from kobeni.config import ExperimentConfig
from kobeni.data import build_cifar_analysis_loader
from kobeni.models import SpatialVocabularyModel
from kobeni.training import resolve_device


def _perplexity(counts: Tensor) -> float:
    probabilities = counts.float() / counts.sum().clamp_min(1)
    nonzero = probabilities > 0
    if not nonzero.any():
        return 0.0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return float(entropy.exp().item())


def summarize_counts(counts: Tensor, class_names: Sequence[str]) -> dict[str, Any]:
    if counts.ndim != 2 or counts.shape[0] != len(class_names):
        raise ValueError("counts must have shape [num_classes, codebook_size]")
    classes = []
    for class_id, class_name in enumerate(class_names):
        row = counts[class_id]
        total = int(row.sum().item())
        probabilities = row.float() / row.sum().clamp_min(1)
        top_values, top_indices = probabilities.topk(min(10, row.numel()))
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "token_count": total,
                "active_codes": int((row > 0).sum().item()),
                "perplexity": _perplexity(row),
                "top_codes": [
                    {"code_id": int(code_id), "probability": float(probability)}
                    for probability, code_id in zip(
                        top_values.tolist(), top_indices.tolist(), strict=True
                    )
                ],
            }
        )
    return {
        "global": {
            "token_count": int(counts.sum().item()),
            "active_codes": int((counts.sum(dim=0) > 0).sum().item()),
            "perplexity": _perplexity(counts.sum(dim=0)),
        },
        "classes": classes,
    }


def _write_csv(path: Path, counts: Tensor, class_names: Sequence[str]) -> None:
    class_totals = counts.sum(dim=1, keepdim=True).clamp_min(1)
    code_totals = counts.sum(dim=0, keepdim=True).clamp_min(1)
    probability_code_given_class = counts.float() / class_totals
    probability_class_given_code = counts.float() / code_totals
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "code_id",
                "count",
                "p_code_given_class",
                "p_class_given_code",
            ]
        )
        for class_id, class_name in enumerate(class_names):
            for code_id in range(counts.shape[1]):
                writer.writerow(
                    [
                        class_id,
                        class_name,
                        code_id,
                        int(counts[class_id, code_id]),
                        float(probability_code_given_class[class_id, code_id]),
                        float(probability_class_given_code[class_id, code_id]),
                    ]
                )


def _color(value: float) -> str:
    intensity = math.sqrt(min(max(value, 0.0), 1.0))
    start = (247, 251, 255)
    end = (8, 48, 107)
    rgb = tuple(round(a + (b - a) * intensity) for a, b in zip(start, end, strict=True))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _write_svg(path: Path, counts: Tensor, class_names: Sequence[str]) -> None:
    probabilities = counts.float() / counts.sum(dim=1, keepdim=True).clamp_min(1)
    maximum = float(probabilities.max().item()) or 1.0
    cell_width = 6
    row_height = 24
    label_width = 100
    header_height = 40
    width = label_width + counts.shape[1] * cell_width + 24
    height = header_height + len(class_names) * row_height + 36
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text { font-family: sans-serif; fill: #222; }</style>",
        f'<text x="{label_width}" y="16" font-size="12">code id</text>',
    ]
    for code_id in range(0, counts.shape[1], 16):
        x = label_width + code_id * cell_width
        lines.append(f'<text x="{x}" y="34" font-size="9">{code_id}</text>')
    for class_id, class_name in enumerate(class_names):
        y = header_height + class_id * row_height
        safe_name = html.escape(class_name)
        lines.append(f'<text x="4" y="{y + 16}" font-size="11">{safe_name}</text>')
        for code_id in range(counts.shape[1]):
            x = label_width + code_id * cell_width
            normalized = float(probabilities[class_id, code_id]) / maximum
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{row_height}" '
                f'fill="{_color(normalized)}"><title>{safe_name}, code {code_id}: '
                f"{float(probabilities[class_id, code_id]):.6f}</title></rect>"
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_usage_artifacts(
    output_dir: Path,
    counts: Tensor,
    class_names: Sequence[str],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"metadata": metadata, **summarize_counts(counts, class_names)}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output_dir / "code_usage.csv", counts, class_names)
    _write_svg(output_dir / "code_usage_heatmap.svg", counts, class_names)
    torch.save(counts, output_dir / "counts.pt")


def analyze_code_usage(
    config: ExperimentConfig,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
) -> None:
    if config.model.variant not in {"vq", "random_vq"}:
        raise ValueError("Code usage analysis requires a VQ model config")
    device = resolve_device(config.train.device)
    model = SpatialVocabularyModel(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = build_cifar_analysis_loader(config.data, split)
    class_names = tuple(loader.dataset.classes)
    counts = torch.zeros(config.model.num_classes, config.model.codebook_size, dtype=torch.long)
    sample_count = 0
    latent_shape: list[int] | None = None
    with torch.inference_mode():
        for images, targets in loader:
            output = model(images.to(device, non_blocking=True))
            if output.indices is None:
                raise RuntimeError("Model did not produce discrete code indices")
            indices = output.indices.cpu()
            targets = targets.cpu()
            latent_shape = list(indices.shape[1:])
            sample_count += targets.numel()
            for class_id in targets.unique().tolist():
                class_indices = indices[targets == class_id].flatten()
                counts[class_id] += torch.bincount(
                    class_indices, minlength=config.model.codebook_size
                )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_eval_accuracy": checkpoint.get("best_eval_accuracy"),
        "dataset": config.data.dataset,
        "split": split,
        "sample_count": sample_count,
        "latent_grid": latent_shape,
        "codebook_size": config.model.codebook_size,
        "encoder_blocks": config.model.encoder_blocks,
        "seed": config.train.seed,
    }
    write_usage_artifacts(output_dir, counts, class_names, metadata)
    print(
        json.dumps(
            {"output_dir": str(output_dir), **summarize_counts(counts, class_names)["global"]}
        )
    )
