from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader

from kobeni.ade20k import build_ade20k_loaders
from kobeni.config import ExperimentConfig
from kobeni.models import ModelOutput, SpatialVocabularySegmentationModel
from kobeni.tracking import resolve_output_dir, write_run_artifacts, write_run_summary
from kobeni.training import (
    build_optimizer,
    build_scheduler,
    configure_cuda_performance,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
)

EpochCallback = Callable[[dict[str, Any]], None]


def _resize_targets(targets: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
    if targets.shape[-2:] == spatial_shape:
        return targets
    return (
        functional.interpolate(targets.unsqueeze(1).float(), size=spatial_shape, mode="nearest")
        .squeeze(1)
        .long()
    )



def _upsample_logits_for_metrics(logits: Tensor, target_shape: tuple[int, int]) -> Tensor:
    """Evaluate every segmentation model against the original-resolution mask."""
    if logits.shape[-2:] == target_shape:
        return logits
    return functional.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)


def compute_segmentation_loss(
    output: ModelOutput,
    images: Tensor,
    targets: Tensor,
    lambda_cls: float,
    lambda_vq: float,
    lambda_rec: float,
    label_smoothing: float,
    ignore_index: int,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    resized_targets = _resize_targets(targets, output.logits.shape[-2:])
    segmentation = functional.cross_entropy(
        output.logits,
        resized_targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    reconstruction = images.new_zeros(())
    total = lambda_cls * segmentation + lambda_vq * output.vq_loss + lambda_rec * reconstruction
    return (
        total,
        {
            "classification_loss": segmentation.detach(),
            "segmentation_loss": segmentation.detach(),
            "vq_loss": output.vq_loss.detach(),
            "commitment_loss": output.commitment_loss.detach(),
            "codebook_loss": output.codebook_loss.detach(),
            "reconstruction_loss": reconstruction.detach(),
        },
        resized_targets,
    )


def _run_segmentation_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer | None,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    sample_count = 0
    valid_pixel_count = 0
    total_loss = 0.0
    component_totals = {
        "classification_loss": 0.0,
        "segmentation_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "reconstruction_loss": 0.0,
    }
    confusion = torch.zeros(
        config.model.num_classes,
        config.model.num_classes,
        dtype=torch.long,
    )
    code_counts = torch.zeros(config.model.codebook_size, dtype=torch.long)
    quantization_error_total = 0.0
    quantized_batches = 0

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
                output = model(images)
                loss, components, resized_targets = compute_segmentation_loss(
                    output,
                    images,
                    targets,
                    config.train.lambda_cls,
                    config.train.lambda_vq,
                    config.train.lambda_rec,
                    config.train.label_smoothing,
                    config.data.segmentation_ignore_index,
                )
            if optimizer is not None:
                scaled_loss = loss / config.train.gradient_accumulation_steps
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                should_step = (batch_index + 1) % config.train.gradient_accumulation_steps == 0
                if should_step or batch_index + 1 == len(loader):
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            batch_size = images.shape[0]
            sample_count += batch_size
            total_loss += loss.detach().item() * batch_size
            for name, value in components.items():
                component_totals[name] += value.item() * batch_size
            metric_logits = _upsample_logits_for_metrics(output.logits, targets.shape[-2:])
            predictions = metric_logits.argmax(dim=1)
            valid = targets != config.data.segmentation_ignore_index
            valid_pixel_count += int(valid.sum().item())
            if valid.any():
                encoded = (
                    (targets[valid] * config.model.num_classes + predictions[valid])
                    .detach()
                    .cpu()
                )
                confusion += torch.bincount(
                    encoded,
                    minlength=config.model.num_classes**2,
                ).view(config.model.num_classes, config.model.num_classes)
            if output.indices is not None:
                code_counts += torch.bincount(
                    output.indices.detach().cpu().flatten(),
                    minlength=config.model.codebook_size,
                )
                quantization_error_total += output.codebook_metrics["quantization_error"].item()
                quantized_batches += 1

    intersections = confusion.diag().float()
    if sample_count == 0:
        raise RuntimeError("Segmentation loader produced no batches; check dataset and batch_size")
    unions = confusion.sum(dim=0).float() + confusion.sum(dim=1).float() - intersections
    valid_classes = unions > 0
    mean_iou = (intersections[valid_classes] / unions[valid_classes]).mean().item()
    pixel_accuracy = intersections.sum().item() / max(1, valid_pixel_count)
    metrics = {
        "loss": total_loss / sample_count,
        "accuracy": pixel_accuracy,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "valid_pixels": float(valid_pixel_count),
        **{name: value / sample_count for name, value in component_totals.items()},
    }
    if code_counts.sum() > 0:
        probabilities = code_counts.float() / code_counts.sum()
        nonzero = probabilities > 0
        active_codes = int(nonzero.sum().item())
        metrics.update(
            {
                "active_codes": float(active_codes),
                "perplexity": float(
                    (-(probabilities[nonzero] * probabilities[nonzero].log()).sum()).exp().item()
                ),
                "dead_code_fraction": 1.0 - active_codes / config.model.codebook_size,
                "quantization_error": quantization_error_total / quantized_batches,
            }
        )
    return metrics


def train_segmentation(
    config: ExperimentConfig,
    epoch_callback: EpochCallback | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    configure_cuda_performance(device, config.train.tf32)
    mixed_precision = config.train.mixed_precision and device.type == "cuda"
    channels_last = config.train.channels_last and device.type == "cuda"
    output_dir = (
        Path(config.train.resume_from).parent
        if config.train.resume_from is not None
        else resolve_output_dir(config)
    )
    config = replace(config, train=replace(config.train, output_dir=str(output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SpatialVocabularySegmentationModel(config.model).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    seed_everything(config.train.seed)
    train_loader, validation_loader = build_ade20k_loaders(config.data)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_scheduler(optimizer, config.train)
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config.train.amp_dtype]
    scaler = (
        torch.amp.GradScaler(
            "cuda",
            init_scale=config.train.grad_scaler_init_scale,
            growth_factor=config.train.grad_scaler_growth_factor,
            backoff_factor=config.train.grad_scaler_backoff_factor,
            growth_interval=config.train.grad_scaler_growth_interval,
            enabled=True,
        )
        if mixed_precision and amp_dtype == torch.float16
        else None
    )
    start_epoch = 1
    best_eval_score = float("-inf")
    if config.train.resume_from is not None:
        start_epoch, best_eval_score = load_checkpoint(
            config.train.resume_from, model, optimizer, scheduler, device, scaler
        )
    history_path = output_dir / "metrics.jsonl"
    if history_path.exists() and config.train.resume_from is None:
        raise FileExistsError(f"{history_path} already exists; use a new output_dir or resume")
    write_run_artifacts(output_dir, config, model, device)
    print(json.dumps({"event": "run_started", "output_dir": str(output_dir)}), flush=True)

    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_started = time.perf_counter()
        train_metrics = _run_segmentation_epoch(
            model,
            train_loader,
            device,
            config,
            optimizer,
            mixed_precision,
            amp_dtype,
            channels_last,
            scaler,
        )
        train_seconds = time.perf_counter() - train_started
        eval_started = time.perf_counter()
        eval_metrics = _run_segmentation_epoch(
            model,
            validation_loader,
            device,
            config,
            None,
            mixed_precision,
            amp_dtype,
            channels_last,
            None,
        )
        eval_seconds = time.perf_counter() - eval_started
        improved = eval_metrics["mean_iou"] > best_eval_score
        if improved:
            best_eval_score = eval_metrics["mean_iou"]
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "batching": {
                "per_device_batch_size": config.data.batch_size,
                "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
                "effective_batch_size": config.data.batch_size
                * config.train.gradient_accumulation_steps,
            },
            "train": train_metrics,
            "eval": eval_metrics,
            "best_eval_mean_iou": best_eval_score,
            "timing": {
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "peak_gpu_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0
                ),
            },
        }
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            best_eval_score,
            config,
            scaler,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                best_eval_score,
                config,
                scaler,
            )
        if epoch_callback is not None:
            epoch_callback(record)

    records = [
        json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line
    ]
    summary = write_run_summary(
        output_dir,
        records,
        time.perf_counter() - run_started,
        selection_metric="mean_iou",
    )
    print(
        json.dumps({"event": "run_completed", "output_dir": str(output_dir), "summary": summary}),
        flush=True,
    )
    return summary
