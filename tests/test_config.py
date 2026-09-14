import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

from kobeni.config import VQ_SCALES, DataConfig, ExperimentConfig, ModelConfig, TrainConfig


def test_load_vq_config() -> None:
    config = ExperimentConfig.from_toml(Path("config/phase1/vq.toml"))

    assert config.model.variant == "vq"
    assert config.model.codebook_size == 128
    assert config.data.dataset == "cifar10"
    assert config.train.epochs == 200
    assert config.train.scheduler == "cosine"
    assert config.train.warmup_epochs == 5


def test_load_cifar100_comparison_configs() -> None:
    continuous = ExperimentConfig.from_toml(Path("config/cifar100/continuous_bottleneck.toml"))
    vq = ExperimentConfig.from_toml(Path("config/cifar100/vq.toml"))

    assert continuous.model.variant == "continuous_bottleneck"
    assert vq.model.variant == "vq"
    for config in (continuous, vq):
        assert config.model.num_classes == 100
        assert config.model.encoder_blocks == 2
        assert config.data.dataset == "cifar100"
        assert config.train.epochs == 200


def test_architecture_config_enables_timestamp_tracking() -> None:
    config = ExperimentConfig.from_toml(Path("config/architecture/vq_baseline.toml"))

    assert config.tracking.enabled
    assert config.tracking.run_name == "vq_strong_gap_attention_residual"
    assert config.data.dataset == "cifar100"
    assert config.model.num_classes == 100
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.model.activation == "relu"
    assert config.model.head == "gap_attention_residual"
    assert config.data.random_crop
    assert config.data.random_horizontal_flip
    assert config.data.cutmix_alpha == 1.0
    assert config.train.mixed_precision
    assert config.train.tf32
    assert config.train.channels_last
    assert config.train.output_dir == "outputs/architecture"


@pytest.mark.parametrize(
    ("name", "stem", "intermediate", "width", "blocks", "latent_dim", "codebook_size"),
    [
        ("tiny", 64, 96, 384, 2, 64, 128),
        ("small", 64, 128, 512, 3, 64, 128),
        ("base", 96, 192, 768, 4, 96, 256),
        ("large", 128, 256, 1024, 6, 128, 256),
        ("xlarge", 160, 320, 1280, 8, 128, 512),
    ],
)
def test_model_scale_presets_inherit_the_strong_baseline(
    name: str,
    stem: int,
    intermediate: int,
    width: int,
    blocks: int,
    latent_dim: int,
    codebook_size: int,
) -> None:
    config = ExperimentConfig.from_toml(Path(f"config/architecture/model_scale/{name}.toml"))

    assert config.model.encoder_stem_channels == stem
    assert config.model.encoder_intermediate_channels == intermediate
    assert config.model.encoder_channels == width
    assert config.model.encoder_blocks == blocks
    assert config.model.latent_dim == latent_dim
    assert config.model.codebook_size == codebook_size
    assert config.model.head == "gap_attention_residual"
    assert config.model.downsampling == "max_pool"
    assert config.data.dataset == "cifar100"
    assert config.train.epochs == 200
    assert config.train.output_dir == "outputs/model_scale"
    assert config.tracking.run_name == f"vq_scale_{name}"


def test_vq_scale_presets() -> None:
    assert VQ_SCALES == {
        "compact": {"latent_dim": 64, "codebook_size": 128},
        "medium": {"latent_dim": 96, "codebook_size": 256},
        "large": {"latent_dim": 128, "codebook_size": 512},
    }


def test_reconstruction_weight_requires_decoder(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "[model]\nreconstruction = false\n[train]\nlambda_rec = 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires model.reconstruction"):
        ExperimentConfig.from_toml(config_path)


@pytest.mark.parametrize(
    ("name", "expected_shape", "expected_batch", "expected_accumulation"),
    [
        ("no_1x1_conv", (128, "strided_conv", "group"), 128, 1),
        ("max_pool_downsampling", (64, "max_pool", "group"), 128, 1),
        ("batch_norm", (64, "strided_conv", "batch"), 128, 1),
        ("fullres_no_bottleneck", (128, "none", "group"), 32, 4),
    ],
)
def test_layer_ablation_configs(
    name: str,
    expected_shape: tuple[int, str, str],
    expected_batch: int,
    expected_accumulation: int,
) -> None:
    config = ExperimentConfig.from_toml(Path(f"config/architecture/layer_ablation/{name}.toml"))

    assert config.model.latent_dim == expected_shape[0]
    assert config.model.downsampling == expected_shape[1]
    assert config.model.normalization == expected_shape[2]
    assert config.data.batch_size == expected_batch
    assert config.train.gradient_accumulation_steps == expected_accumulation


@pytest.mark.parametrize(
    ("name", "downsampling"),
    [
        ("max_pool", "max_pool"),
        ("avg_pool", "avg_pool"),
        ("max_then_avg_pool", "max_then_avg_pool"),
        ("avg_then_max_pool", "avg_then_max_pool"),
    ],
)
def test_pooling_ablation_configs(name: str, downsampling: str) -> None:
    config = ExperimentConfig.from_toml(Path(f"config/architecture/pooling_ablation/{name}.toml"))

    assert config.data.dataset == "cifar100"
    assert config.model.num_classes == 100
    assert config.model.normalization == "batch"
    assert config.model.downsampling == downsampling
    assert config.data.batch_size == 128
    assert config.train.gradient_accumulation_steps == 1


@pytest.mark.parametrize("codebook_size", [256, 512, 1024, 2048])
def test_codebook_size_ablation_configs(codebook_size: int) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/codebook_size_ablation/k_{codebook_size}.toml")
    )

    assert config.data.dataset == "cifar100"
    assert config.model.num_classes == 100
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.model.codebook_size == codebook_size
    assert config.model.latent_dim == 64


@pytest.mark.parametrize(
    ("name, expected_channels, activation, remove_middle_conv, pool_after_first_conv"),
    [
        ("remove_middle_conv", 128, "relu", True, False),
        ("pool_after_first_conv", 128, "relu", False, True),
        ("silu", 128, "silu", False, False),
        ("channels_256", 256, "relu", False, False),
    ],
)
def test_encoder_ablation_configs(
    name: str,
    expected_channels: int,
    activation: str,
    remove_middle_conv: bool,
    pool_after_first_conv: bool,
) -> None:
    config = ExperimentConfig.from_toml(Path(f"config/architecture/encoder_ablation/{name}.toml"))

    assert config.data.dataset == "cifar100"
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.model.codebook_size == 128
    assert config.model.encoder_channels == expected_channels
    assert config.model.activation == activation
    assert config.model.remove_middle_conv is remove_middle_conv
    assert config.model.pool_after_first_conv is pool_after_first_conv


@pytest.mark.parametrize(
    ("name", "encoder_channels", "latent_dim"),
    [
        ("latent_dim_64", 256, 64),
        ("latent_dim_128", 256, 128),
        ("latent_dim_256", 256, 256),
        ("encoder_width_384", 384, 64),
    ],
)
def test_latent_width_ablation_configs(name: str, encoder_channels: int, latent_dim: int) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/latent_width_ablation/{name}.toml")
    )

    assert config.data.dataset == "cifar100"
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.model.activation == "relu"
    assert config.model.codebook_size == 128
    assert config.model.encoder_channels == encoder_channels
    assert config.model.latent_dim == latent_dim


@pytest.mark.parametrize(
    ("name", "random_crop", "random_horizontal_flip", "cutmix_alpha"),
    [
        ("crop", True, False, 0.0),
        ("flip", False, True, 0.0),
        ("cutmix", False, False, 1.0),
        ("crop_flip_cutmix", True, True, 1.0),
    ],
)
def test_augmentation_ablation_configs(
    name: str,
    random_crop: bool,
    random_horizontal_flip: bool,
    cutmix_alpha: float,
) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/augmentation_ablation/{name}.toml")
    )

    assert config.data.dataset == "cifar100"
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.model.activation == "relu"
    assert config.data.random_crop is random_crop
    assert config.data.random_horizontal_flip is random_horizontal_flip
    assert config.data.cutmix_alpha == cutmix_alpha
    assert config.data.cutmix_probability == 1.0


@pytest.mark.parametrize(
    ("name", "cutmix_alpha", "cutmix_probability"),
    [
        ("crop_flip", 0.0, 1.0),
        ("crop_flip_cutmix_p025", 1.0, 0.25),
        ("crop_flip_cutmix_p050", 1.0, 0.5),
        ("crop_flip_cutmix_p075", 1.0, 0.75),
    ],
)
def test_cutmix_probability_ablation_configs(
    name: str, cutmix_alpha: float, cutmix_probability: float
) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/cutmix_probability_ablation/{name}.toml")
    )

    assert config.data.dataset == "cifar100"
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.data.random_crop
    assert config.data.random_horizontal_flip
    assert config.data.cutmix_alpha == cutmix_alpha
    assert config.data.cutmix_probability == cutmix_probability


@pytest.mark.parametrize(
    ("name", "variant", "seed", "lambda_vq", "run_name"),
    [
        ("vq_seed_0", "vq", 0, 1.0, "vq_strong_recipe"),
        (
            "continuous_seed_0",
            "continuous_bottleneck",
            0,
            0.0,
            "continuous_strong_recipe",
        ),
        ("vq_seed_1", "vq", 1, 1.0, "vq_strong_recipe"),
        (
            "continuous_seed_1",
            "continuous_bottleneck",
            1,
            0.0,
            "continuous_strong_recipe",
        ),
    ],
)
def test_vq_continuous_strong_comparison_configs(
    name: str, variant: str, seed: int, lambda_vq: float, run_name: str
) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/vq_continuous_strong_comparison/{name}.toml")
    )

    assert config.model.variant == variant
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.model.activation == "relu"
    assert config.data.random_crop
    assert config.data.random_horizontal_flip
    assert config.data.cutmix_alpha == 1.0
    assert config.data.cutmix_probability == 0.5
    assert config.train.lambda_vq == lambda_vq
    assert config.train.seed == seed
    assert config.train.mixed_precision
    assert config.train.tf32
    assert config.train.channels_last
    assert config.tracking.run_name == run_name


@pytest.mark.parametrize(
    "name",
    ["gap", "pool_2x2", "gap_pool_2x2_concat", "dwconv3x3"],
)
def test_head_ablation_configs(name: str) -> None:
    config = ExperimentConfig.from_toml(Path(f"config/architecture/head_ablation/{name}.toml"))

    assert config.model.head == name
    assert config.model.variant == "vq"
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.data.random_crop
    assert config.data.random_horizontal_flip
    assert config.data.cutmix_alpha == 1.0
    assert config.data.cutmix_probability == 0.5
    assert config.train.mixed_precision
    assert config.train.tf32
    assert config.train.channels_last


@pytest.mark.parametrize(
    "name",
    [
        "gap",
        "dwconv3x3",
        "learned_weighted_pool",
        "single_query_attention_pool",
    ],
)
def test_pooling_head_ablation_configs(name: str) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/pooling_head_ablation/{name}.toml")
    )

    assert config.model.head == name
    assert config.model.variant == "vq"
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.data.random_crop
    assert config.data.random_horizontal_flip
    assert config.data.cutmix_alpha == 1.0
    assert config.data.cutmix_probability == 0.5
    assert config.train.epochs == 200
    assert config.train.mixed_precision
    assert config.train.tf32
    assert config.train.channels_last


@pytest.mark.parametrize(
    "name",
    [
        "gap_attention_residual",
        "four_query_attention_mean",
        "global_self_attention",
        "dwconv_attention_pool",
    ],
)
def test_advanced_attention_head_ablation_configs(name: str) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/architecture/advanced_attention_head_ablation/{name}.toml")
    )

    assert config.model.head == name
    assert config.model.variant == "vq"
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128
    assert config.model.normalization == "batch"
    assert config.model.downsampling == "max_pool"
    assert config.data.random_crop
    assert config.data.random_horizontal_flip
    assert config.data.cutmix_alpha == 1.0
    assert config.data.cutmix_probability == 0.5
    assert config.train.epochs == 200
    assert config.train.mixed_precision
    assert config.train.tf32
    assert config.train.channels_last


def test_vq_baseline_explicitly_declares_all_hyperparameters() -> None:
    with Path("config/architecture/vq_baseline.toml").open("rb") as file:
        raw = tomllib.load(file)

    assert set(raw["model"]) == {field.name for field in fields(ModelConfig)}
    assert set(raw["data"]) == {field.name for field in fields(DataConfig)}
    assert {field.name for field in fields(TrainConfig)} - set(raw["train"]) == {"resume_from"}


@pytest.mark.parametrize(
    ("name", "learning_rate", "lambda_vq"),
    [
        ("lr500_lambda100", 0.0005, 1.0),
        ("lr500_lambda140", 0.0005, 1.4),
        ("lr600_lambda100", 0.0006, 1.0),
        ("lr600_lambda140", 0.0006, 1.4),
    ],
)
def test_conservative_optuna_200_epoch_followup_configs(
    name: str, learning_rate: float, lambda_vq: float
) -> None:
    config = ExperimentConfig.from_toml(
        Path(f"config/optuna_followup/conservative_200/{name}.toml")
    )

    assert config.train.epochs == 200
    assert config.train.learning_rate == pytest.approx(learning_rate)
    assert config.train.weight_decay == pytest.approx(0.0001)
    assert config.model.commitment_weight == pytest.approx(0.35)
    assert config.train.lambda_vq == pytest.approx(lambda_vq)
    assert config.train.seed == 0
    assert config.model.encoder_channels == 384
    assert config.model.latent_dim == 64
    assert config.model.codebook_size == 128


@pytest.mark.parametrize(
    ("reconstruction", "lambda_rec"),
    [(True, 0.0), (False, 0.1)],
)
def test_segmentation_rejects_unsupported_reconstruction(
    reconstruction: bool,
    lambda_rec: float,
) -> None:
    config = ExperimentConfig(
        model=ModelConfig(
            task="segmentation_direct",
            variant="vq",
            num_classes=150,
            reconstruction=reconstruction,
        ),
        data=DataConfig(dataset="ade20k"),
        train=TrainConfig(lambda_rec=lambda_rec),
    )

    with pytest.raises(ValueError, match="Segmentation does not support reconstruction"):
        config.validate()
