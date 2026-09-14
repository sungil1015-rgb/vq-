from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

from kobeni.config import ExperimentConfig, TrainConfig
from kobeni.data import build_cifar_loaders
from kobeni.models import ModelOutput, SpatialVocabularyModel
from kobeni.tracking import resolve_output_dir, write_run_artifacts, write_run_summary


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_performance(device: torch.device, tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def build_optimizer(model: nn.Module, config: TrainConfig) -> Optimizer:
    if config.optimizer != "adamw":
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adamw_beta1, config.adamw_beta2),
        eps=config.adamw_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.adamw_amsgrad,
    )


def build_scheduler(optimizer: Optimizer, config: TrainConfig) -> LRScheduler | None:
    if config.scheduler == "none":
        return None

    minimum_factor = config.min_learning_rate / config.learning_rate

    def learning_rate_factor(step: int) -> float:
        if config.warmup_epochs > 0 and step < config.warmup_epochs:
            return (step + 1) / config.warmup_epochs
        if config.warmup_epochs == 0:
            progress = step / max(1, config.epochs - 1)
        else:
            progress = (step - config.warmup_epochs + 1) / max(
                1, config.epochs - config.warmup_epochs
            )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=learning_rate_factor)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    best_eval_accuracy: float,
    config: ExperimentConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_eval_accuracy": best_eval_accuracy,
        "config": asdict(config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler_state = checkpoint.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_eval_accuracy = float(checkpoint.get("best_eval_accuracy", float("-inf")))
    return start_epoch, best_eval_accuracy


def compute_loss(
    output: ModelOutput,
    images: Tensor,
    targets: Tensor,
    lambda_vq: float,
    lambda_rec: float,
    secondary_targets: Tensor | None = None,
    mixing_lambda: float = 1.0,
    lambda_cls: float = 1.0,
    label_smoothing: float = 0.0,
    reconstruction_loss: str = "l1",
    classification_loss: str = "cross_entropy",
) -> tuple[Tensor, dict[str, Tensor]]:
    if classification_loss != "cross_entropy":
        raise ValueError(f"Unknown classification loss: {classification_loss}")
    classification = functional.cross_entropy(
        output.logits, targets, label_smoothing=label_smoothing
    )
    if secondary_targets is not None:
        classification = mixing_lambda * classification + (
            1.0 - mixing_lambda
        ) * functional.cross_entropy(
            output.logits,
            secondary_targets,
            label_smoothing=label_smoothing,
        )
    reconstruction = images.new_zeros(())
    if output.reconstruction is not None:
        if reconstruction_loss != "l1":
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")
        reconstruction = functional.l1_loss(output.reconstruction, images)
    total = lambda_cls * classification + lambda_vq * output.vq_loss + lambda_rec * reconstruction
    return total, {
        "classification_loss": classification.detach(),
        "vq_loss": output.vq_loss.detach(),
        "commitment_loss": output.commitment_loss.detach(),
        "codebook_loss": output.codebook_loss.detach(),
        "reconstruction_loss": reconstruction.detach(),
    }


def _apply_cutmix(
    images: Tensor,
    targets: Tensor,
    alpha: float,
    probability: float,
) -> tuple[Tensor, Tensor | None, float]:
    if alpha <= 0 or images.shape[0] < 2 or torch.rand(()).item() >= probability:
        return images, None, 1.0

    mixing_lambda = float(torch.distributions.Beta(alpha, alpha).sample().item())
    height, width = images.shape[-2:]
    cut_ratio = math.sqrt(1.0 - mixing_lambda)
    cut_height = int(height * cut_ratio)
    cut_width = int(width * cut_ratio)
    center_y = int(torch.randint(height, (1,), device=images.device).item())
    center_x = int(torch.randint(width, (1,), device=images.device).item())
    y1 = max(0, center_y - cut_height // 2)
    y2 = min(height, center_y + cut_height // 2)
    x1 = max(0, center_x - cut_width // 2)
    x2 = min(width, center_x + cut_width // 2)
    permutation = torch.randperm(images.shape[0], device=images.device)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    adjusted_lambda = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return mixed_images, targets[permutation], adjusted_lambda


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_vq: float,
    lambda_rec: float,
    optimizer: Optimizer | None,
    codebook_size: int,
    lambda_cls: float = 1.0,
    label_smoothing: float = 0.0,
    reconstruction_loss: str = "l1",
    classification_loss: str = "cross_entropy",
    gradient_accumulation_steps: int = 1,
    cutmix_alpha: float = 0.0,
    cutmix_probability: float = 1.0,
    mixed_precision: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    top_k: int = 5,
    scaler: Any | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    sample_count = 0
    correct = 0.0
    top5_correct = 0.0
    total_loss = 0.0
    component_totals = {
        "classification_loss": 0.0,
        "vq_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "reconstruction_loss": 0.0,
    }
    code_counts = torch.zeros(codebook_size, dtype=torch.long)
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
            secondary_targets = None
            mixing_lambda = 1.0
            if is_training:
                images, secondary_targets, mixing_lambda = _apply_cutmix(
                    images, targets, cutmix_alpha, cutmix_probability
                )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=mixed_precision,
            ):
                output = model(images)
                loss, components = compute_loss(
                    output,
                    images,
                    targets,
                    lambda_vq=lambda_vq,
                    lambda_rec=lambda_rec,
                    secondary_targets=secondary_targets,
                    mixing_lambda=mixing_lambda,
                    lambda_cls=lambda_cls,
                    label_smoothing=label_smoothing,
                    reconstruction_loss=reconstruction_loss,
                    classification_loss=classification_loss,
                )
            if optimizer is not None:
                scaled_loss = loss / gradient_accumulation_steps
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                should_step = (batch_index + 1) % gradient_accumulation_steps == 0
                if should_step or batch_index + 1 == len(loader):
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = images.shape[0]
            sample_count += batch_size
            predictions = output.logits.argmax(dim=1)
            primary_correct = (predictions == targets).float()
            effective_top_k = min(top_k, output.logits.shape[1])
            top5_predictions = output.logits.topk(effective_top_k, dim=1).indices
            primary_top5 = top5_predictions.eq(targets[:, None]).any(dim=1).float()
            if secondary_targets is None:
                correct += primary_correct.sum().item()
                top5_correct += primary_top5.sum().item()
            else:
                secondary_correct = (predictions == secondary_targets).float()
                secondary_top5 = top5_predictions.eq(secondary_targets[:, None]).any(dim=1).float()
                correct += (
                    (mixing_lambda * primary_correct + (1.0 - mixing_lambda) * secondary_correct)
                    .sum()
                    .item()
                )
                top5_correct += (
                    (mixing_lambda * primary_top5 + (1.0 - mixing_lambda) * secondary_top5)
                    .sum()
                    .item()
                )
            total_loss += loss.detach().item() * batch_size
            for name, value in components.items():
                component_totals[name] += value.item() * batch_size
            if output.indices is not None:
                code_counts += torch.bincount(
                    output.indices.detach().cpu().flatten(), minlength=codebook_size
                )
                quantization_error_total += output.codebook_metrics["quantization_error"].item()
                quantized_batches += 1

    if sample_count == 0:
        raise RuntimeError("Classification loader produced no batches")

    metrics = {
        "loss": total_loss / sample_count,
        "accuracy": correct / sample_count,
        "top5_accuracy": top5_correct / sample_count,
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
                "dead_code_fraction": 1.0 - active_codes / codebook_size,
                "quantization_error": quantization_error_total / quantized_batches,
            }
        )
    return metrics


EpochCallback = Callable[[dict[str, Any]], None]


def train(
    config: ExperimentConfig,
    epoch_callback: EpochCallback | None = None,
) -> dict[str, Any]:
    if config.model.task != "classification":
        from kobeni.segmentation_training import train_segmentation

        return train_segmentation(config, epoch_callback)
    run_started = time.perf_counter()
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    configure_cuda_performance(device, config.train.tf32)
    mixed_precision = config.train.mixed_precision and device.type == "cuda"
    channels_last = config.train.channels_last and device.type == "cuda"
    if config.train.resume_from is not None:
        output_dir = Path(config.train.resume_from).parent
    else:
        output_dir = resolve_output_dir(config)
    config = replace(
        config,
        train=replace(config.train, output_dir=str(output_dir)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SpatialVocabularyModel(config.model).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    # Keep data order and augmentation RNG paired across model variants. Model
    # construction consumes a different number of random values for VQ.
    seed_everything(config.train.seed)
    train_loader, test_loader = build_cifar_loaders(config.data)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_scheduler(optimizer, config.train)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.train.amp_dtype]
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
    best_eval_accuracy = float("-inf")
    if config.train.resume_from is not None:
        start_epoch, best_eval_accuracy = load_checkpoint(
            config.train.resume_from, model, optimizer, scheduler, device, scaler
        )

    history_path = output_dir / "metrics.jsonl"
    if history_path.exists() and config.train.resume_from is None:
        raise FileExistsError(
            f"{history_path} already exists; use a new output_dir or set resume_from"
        )
    write_run_artifacts(output_dir, config, model, device)
    print(json.dumps({"event": "run_started", "output_dir": str(output_dir)}), flush=True)

    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_started = time.perf_counter()
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            lambda_vq=config.train.lambda_vq,
            lambda_rec=config.train.lambda_rec,
            optimizer=optimizer,
            codebook_size=config.model.codebook_size,
            lambda_cls=config.train.lambda_cls,
            label_smoothing=config.train.label_smoothing,
            reconstruction_loss=config.train.reconstruction_loss,
            classification_loss=config.train.classification_loss,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            cutmix_alpha=config.data.cutmix_alpha,
            cutmix_probability=config.data.cutmix_probability,
            mixed_precision=mixed_precision,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            top_k=config.train.top_k,
            scaler=scaler,
        )
        train_seconds = time.perf_counter() - train_started
        eval_started = time.perf_counter()
        eval_metrics = _run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            lambda_vq=config.train.lambda_vq,
            lambda_rec=config.train.lambda_rec,
            optimizer=None,
            codebook_size=config.model.codebook_size,
            lambda_cls=config.train.lambda_cls,
            label_smoothing=config.train.label_smoothing,
            reconstruction_loss=config.train.reconstruction_loss,
            classification_loss=config.train.classification_loss,
            mixed_precision=mixed_precision,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            top_k=config.train.top_k,
        )
        eval_seconds = time.perf_counter() - eval_started
        epoch_seconds = time.perf_counter() - epoch_started
        peak_gpu_memory_mb = 0.0
        if device.type == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "batching": {
                "per_device_batch_size": config.data.batch_size,
                "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
                "effective_batch_size": (
                    config.data.batch_size * config.train.gradient_accumulation_steps
                ),
            },
            "train": train_metrics,
            "eval": eval_metrics,
            "timing": {
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "epoch_seconds": epoch_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            },
        }
        improved = eval_metrics["accuracy"] > best_eval_accuracy
        if improved:
            best_eval_accuracy = eval_metrics["accuracy"]
        record["best_eval_accuracy"] = best_eval_accuracy
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
            best_eval_accuracy,
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
                best_eval_accuracy,
                config,
                scaler,
            )
        if epoch_callback is not None:
            epoch_callback(record)

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = write_run_summary(output_dir, records, time.perf_counter() - run_started)
    completion = {"event": "run_completed", "output_dir": str(output_dir)}
    completion["summary"] = summary
    print(json.dumps(completion), flush=True)
    return summary
