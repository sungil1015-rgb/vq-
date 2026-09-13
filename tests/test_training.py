import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import kobeni.training as training
from kobeni.analysis.experiments import summarize_experiments
from kobeni.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrackingConfig,
    TrainConfig,
)
from kobeni.models import SpatialVocabularyModel
from kobeni.tracking import resolve_output_dir
from kobeni.training import build_scheduler, load_checkpoint, save_checkpoint


def test_cosine_scheduler_warms_up_and_reaches_minimum() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = AdamW([parameter], lr=0.1)
    config = TrainConfig(
        epochs=10,
        learning_rate=0.1,
        scheduler="cosine",
        warmup_epochs=2,
        min_learning_rate=0.01,
    )
    scheduler = build_scheduler(optimizer, config)

    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_checkpoint_restores_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    config = ExperimentConfig(
        train=TrainConfig(epochs=10, warmup_epochs=2, output_dir=str(tmp_path))
    )
    model = SpatialVocabularyModel(config.model)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
    scheduler = build_scheduler(optimizer, config.train)
    assert scheduler is not None

    images = torch.randn(2, 3, 32, 32)
    model(images).logits.mean().backward()
    optimizer.step()
    scheduler.step()
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(checkpoint_path, 3, model, optimizer, scheduler, 0.75, config)

    restored_model = SpatialVocabularyModel(config.model)
    restored_optimizer = AdamW(restored_model.parameters(), lr=config.train.learning_rate)
    restored_scheduler = build_scheduler(restored_optimizer, config.train)
    start_epoch, best_accuracy = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
    )

    assert start_epoch == 4
    assert best_accuracy == pytest.approx(0.75)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    for expected, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, restored)


def test_train_writes_best_and_last_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    loader = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(training, "build_cifar_loaders", lambda _: (loader, loader))
    config = ExperimentConfig(
        model=ModelConfig(variant="continuous", encoder_channels=16),
        data=DataConfig(batch_size=2, num_workers=0),
        train=TrainConfig(
            epochs=2,
            learning_rate=3e-4,
            warmup_epochs=1,
            gradient_accumulation_steps=2,
            output_dir=str(tmp_path),
        ),
    )

    reported_epochs = []
    summary = training.train(config, epoch_callback=lambda record: reported_epochs.append(record))

    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    records = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    final_record = json.loads(records[-1])
    assert "top5_accuracy" in final_record["eval"]
    assert "commitment_loss" in final_record["eval"]
    assert "timing" in final_record
    assert final_record["batching"]["gradient_accumulation_steps"] == 2
    assert final_record["batching"]["effective_batch_size"] == 4
    assert (tmp_path / "run_metadata.json").is_file()
    assert (tmp_path / "model_summary.json").is_file()
    assert (tmp_path / "run_summary.json").is_file()
    assert [record["epoch"] for record in reported_epochs] == [1, 2]
    assert summary["epochs_completed"] == 2

    report = summarize_experiments(tmp_path)
    assert report["num_runs"] == 1
    assert report["runs"][0]["encoder_blocks"] == 2
    assert (tmp_path / "experiment_table.csv").is_file()

    last_checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu")
    assert last_checkpoint["epoch"] == 2
    assert last_checkpoint["scheduler"] is not None

def test_tracked_train_creates_timestamped_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    loader = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(training, "build_cifar_loaders", lambda _: (loader, loader))
    output_base = tmp_path / "architecture"
    config = ExperimentConfig(
        model=ModelConfig(variant="continuous", encoder_channels=16),
        data=DataConfig(batch_size=2, num_workers=0),
        train=TrainConfig(
            epochs=1,
            learning_rate=3e-4,
            warmup_epochs=0,
            seed=7,
            output_dir=str(output_base),
        ),
        tracking=TrackingConfig(enabled=True, run_name="cosine test"),
    )

    training.train(config)

    run_dirs = list(output_base.glob("????_??????/cosine_test/seed_7"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "source_snapshot/kobeni/training.py").is_file()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["started_at"].endswith("+09:00")
    assert metadata["completed_at"].endswith("+09:00")
    assert len(metadata["source_fingerprint"]) == 64
    assert (run_dir / "run_summary.json").is_file()


def test_explicit_run_group_keeps_parallel_runs_together(tmp_path: Path) -> None:
    config = ExperimentConfig(
        train=TrainConfig(output_dir=str(tmp_path), seed=3),
        tracking=TrackingConfig(
            enabled=True,
            run_name="vq_bn_k_256",
            run_group="0911_180000_codebook_size_ablation",
        ),
    )

    assert resolve_output_dir(config) == (
        tmp_path
        / "0911_180000_codebook_size_ablation"
        / "vq_bn_k_256"
        / "seed_3"
    )


def test_cutmix_returns_mixed_targets_and_valid_lambda() -> None:
    torch.manual_seed(0)
    images = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3])

    mixed_images, secondary_targets, mixing_lambda = training._apply_cutmix(
        images, targets, alpha=1.0, probability=1.0
    )

    assert mixed_images.shape == images.shape
    assert secondary_targets is not None
    assert secondary_targets.shape == targets.shape
    assert sorted(secondary_targets.tolist()) == targets.tolist()
    assert 0.0 <= mixing_lambda <= 1.0


def test_loss_coefficients_are_applied_explicitly() -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="vq"))
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    output = model(images)

    loss, components = training.compute_loss(
        output,
        images,
        targets,
        lambda_cls=0.5,
        lambda_vq=2.0,
        lambda_rec=0.0,
        label_smoothing=0.1,
    )

    expected = 0.5 * components["classification_loss"] + 2.0 * components["vq_loss"]
    assert loss.detach() == pytest.approx(expected)


def test_build_optimizer_uses_adamw_config() -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="continuous"))
    config = TrainConfig(
        learning_rate=0.0123,
        weight_decay=0.0456,
        adamw_beta1=0.8,
        adamw_beta2=0.88,
        adamw_epsilon=1e-7,
        adamw_amsgrad=True,
    )

    optimizer = training.build_optimizer(model, config)
    group = optimizer.param_groups[0]

    assert group["lr"] == pytest.approx(0.0123)
    assert group["weight_decay"] == pytest.approx(0.0456)
    assert group["betas"] == pytest.approx((0.8, 0.88))
    assert group["eps"] == pytest.approx(1e-7)
    assert group["amsgrad"] is True
