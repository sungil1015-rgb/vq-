from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from kobeni.ade20k import ADE20KSegmentationDataset
from kobeni.config import DataConfig, ExperimentConfig, ModelConfig
from kobeni.models import SpatialVocabularySegmentationModel
from kobeni.segmentation_training import (
    _upsample_logits_for_metrics,
    compute_segmentation_loss,
)


def _small_segmentation_config(task: str) -> ModelConfig:
    return ModelConfig(
        task=task,  # type: ignore[arg-type]
        variant="vq",
        num_classes=150,
        encoder_stem_channels=8,
        encoder_intermediate_channels=8,
        encoder_channels=16,
        encoder_blocks=1,
        latent_dim=8,
        codebook_size=16,
        segmentation_decoder_hidden_channels=12,
        segmentation_decoder_output_channels=8,
    )


@pytest.mark.parametrize(
    ("task", "expected_logits_shape", "expected_target_shape"),
    [
        ("segmentation_direct", (2, 150, 8, 8), (2, 8, 8)),
        ("segmentation_decoder", (2, 150, 32, 32), (2, 32, 32)),
    ],
)
def test_vq_segmentation_paths_preserve_their_intended_supervision_scale(
    task: str,
    expected_logits_shape: tuple[int, ...],
    expected_target_shape: tuple[int, ...],
) -> None:
    model = SpatialVocabularySegmentationModel(_small_segmentation_config(task))
    images = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 150, (2, 32, 32))

    output = model(images)
    loss, components, supervision = compute_segmentation_loss(
        output,
        images,
        targets,
        lambda_cls=1.0,
        lambda_vq=1.0,
        lambda_rec=0.0,
        label_smoothing=0.0,
        ignore_index=255,
    )
    loss.backward()

    assert tuple(output.logits.shape) == expected_logits_shape
    assert tuple(supervision.shape) == expected_target_shape
    assert output.indices is not None
    assert tuple(output.indices.shape) == (2, 8, 8)
    assert components["segmentation_loss"].item() > 0
    assert model.quantizer.embedding.weight.grad is not None


def test_full_resolution_metrics_interpolate_logits_before_argmax() -> None:
    logits = torch.tensor(
        [[[[2.0, 0.0], [0.0, 2.0]], [[0.0, 2.0], [2.0, 0.0]]]]
    )

    metric_logits = _upsample_logits_for_metrics(logits, (4, 4))
    expected = torch.nn.functional.interpolate(
        logits,
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )

    assert torch.equal(metric_logits, expected)
    assert tuple(metric_logits.argmax(dim=1).shape) == (1, 4, 4)


def test_ade20k_dataset_maps_labels_and_pairs_spatial_transforms(tmp_path: Path) -> None:
    root = tmp_path / "ADEChallengeData2016"
    image_dir = root / "images" / "training"
    annotation_dir = root / "annotations" / "training"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    Image.new("RGB", (4, 2), color=(10, 20, 30)).save(image_dir / "sample.jpg")
    mask = Image.new("L", (4, 2))
    mask.putdata([0, 1, 150, 151, 1, 150, 0, 151])
    mask.save(annotation_dir / "sample.png")

    dataset = ADE20KSegmentationDataset(
        DataConfig(
            dataset="ade20k",
            root=str(tmp_path),
            image_shape=[2, 4],
            random_horizontal_flip=False,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        ),
        split="training",
        train=False,
    )
    image, target = dataset[0]

    assert tuple(image.shape) == (3, 2, 4)
    assert target.tolist() == [[255, 0, 149, 255], [0, 149, 255, 255]]

@pytest.mark.parametrize(
    ("normalization", "activation", "expected_batch_norm", "expected_relu"),
    [
        ("none", "none", 0, 0),
        ("none", "relu", 0, 2),
        ("batch", "none", 2, 0),
        ("batch", "relu", 2, 2),
    ],
)
def test_decoder_normalization_and_activation_are_independently_configurable(
    normalization: str,
    activation: str,
    expected_batch_norm: int,
    expected_relu: int,
) -> None:
    config = replace(
        _small_segmentation_config("segmentation_decoder"),
        segmentation_decoder_normalization=normalization,  # type: ignore[arg-type]
        segmentation_decoder_activation=activation,  # type: ignore[arg-type]
    )
    model = SpatialVocabularySegmentationModel(config)

    assert model.decoder is not None
    assert sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.decoder.modules()) == (
        expected_batch_norm
    )
    assert sum(isinstance(module, torch.nn.ReLU) for module in model.decoder.modules()) == (
        expected_relu
    )


@pytest.mark.parametrize(
    ("name", "task", "normalization", "activation"),
    [
        ("a_direct", "segmentation_direct", "batch", "relu"),
        ("b_linear", "segmentation_decoder", "none", "none"),
        ("c_relu", "segmentation_decoder", "none", "relu"),
        ("d_batchnorm", "segmentation_decoder", "batch", "none"),
    ],
)
def test_ade20k_decoder_ablation_configs(
    name: str, task: str, normalization: str, activation: str
) -> None:
    config = ExperimentConfig.from_toml(Path(f"config/segmentation/decoder_ablation/{name}.toml"))

    assert config.model.task == task
    assert config.model.segmentation_decoder_normalization == normalization
    assert config.model.segmentation_decoder_activation == activation
    assert config.train.epochs == 100
