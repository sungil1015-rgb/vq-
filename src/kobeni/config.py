from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModelVariant = Literal["continuous", "continuous_bottleneck", "vq", "random_vq"]
TaskKind = Literal["classification", "segmentation_direct", "segmentation_decoder"]
NormalizationKind = Literal["group", "batch"]
ActivationKind = Literal["relu", "silu"]
ClassifierHeadKind = Literal[
    "gap",
    "pool_2x2",
    "gap_pool_2x2_concat",
    "dwconv3x3",
    "learned_weighted_pool",
    "single_query_attention_pool",
    "gap_attention_residual",
    "four_query_attention_mean",
    "global_self_attention",
    "dwconv_attention_pool",
]
DownsamplingKind = Literal[
    "strided_conv",
    "max_pool",
    "avg_pool",
    "max_then_avg_pool",
    "avg_then_max_pool",
    "none",
]
VQScaleName = Literal["compact", "medium", "large"]

VQ_SCALES: dict[VQScaleName, dict[str, int]] = {
    "compact": {
        "latent_dim": 64,
        "codebook_size": 128,
    },
    "medium": {
        "latent_dim": 96,
        "codebook_size": 256,
    },
    "large": {
        "latent_dim": 128,
        "codebook_size": 512,
    },
}


def _merge_tables(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_tables(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml_with_extends(
    path: Path,
    parents: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in parents:
        chain = " -> ".join(str(item) for item in (*parents, resolved))
        raise ValueError(f"Cyclic TOML extends chain: {chain}")
    with resolved.open("rb") as file:
        raw: dict[str, Any] = tomllib.load(file)
    parent_reference = raw.pop("extends", None)
    if parent_reference is None:
        return raw
    if not isinstance(parent_reference, str):
        raise ValueError("Top-level extends must be a TOML path string")
    parent_path = resolved.parent / parent_reference
    parent = _load_toml_with_extends(parent_path, (*parents, resolved))
    return _merge_tables(parent, raw)


@dataclass(frozen=True)
class ModelConfig:
    task: TaskKind = "classification"
    variant: ModelVariant = "vq"
    num_classes: int = 10
    encoder_channels: int = 384
    encoder_blocks: int = 2
    latent_dim: int = 64
    codebook_size: int = 128
    commitment_weight: float = 0.25
    codebook_weight: float = 1.0
    codebook_init_scale: float = 1.0
    quantizer_force_float32: bool = True
    use_bottleneck: bool = True
    input_channels: int = 3
    encoder_stem_channels: int = 64
    encoder_intermediate_channels: int = 96
    encoder_kernel_size: int = 3
    residual_kernel_size: int = 3
    projection_kernel_size: int = 1
    convolution_bias: bool = False
    normalization: NormalizationKind = "batch"
    normalization_epsilon: float = 1e-5
    batch_norm_momentum: float = 0.1
    group_norm_groups: int = 8
    downsampling: DownsamplingKind = "max_pool"
    downsampling_stride: int = 2
    pool_kernel_size: int = 2
    pool_stride: int = 2
    activation: ActivationKind = "relu"
    remove_middle_conv: bool = False
    pool_after_first_conv: bool = False
    bottleneck_kernel_size: int = 1
    bottleneck_bias: bool = True
    head: ClassifierHeadKind = "gap"
    classifier_bias: bool = True
    head_spatial_pool_size: int = 2
    head_dwconv_kernel_size: int = 3
    head_depthwise_bias: bool = False
    weighted_pool_hidden_dim: int = 16
    attention_query_count: int = 4
    attention_query_init_std: float = 0.125
    attention_temperature: float = 8.0
    attention_residual_alpha_init: float = 0.0
    segmentation_decoder_hidden_channels: int = 128
    segmentation_decoder_output_channels: int = 64
    segmentation_upsample_mode: Literal["nearest", "bilinear"] = "bilinear"
    segmentation_decoder_normalization: Literal["none", "batch"] = "batch"
    segmentation_decoder_activation: Literal["none", "relu"] = "relu"
    self_attention_heads: int = 4
    self_attention_mlp_ratio: float = 2.0
    reconstruction: bool = False


@dataclass(frozen=True)
class DataConfig:
    dataset: Literal["cifar10", "cifar100", "ade20k"] = "cifar10"
    root: str = "data"
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 32
    image_shape: list[int] | None = None
    normalization_mean: list[float] | None = None
    normalization_std: list[float] | None = None
    random_crop: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    horizontal_flip_probability: float = 0.5
    cutmix_alpha: float = 0.0
    cutmix_probability: float = 1.0
    segmentation_ignore_index: int = 255
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 3e-4
    scheduler: Literal["cosine", "none"] = "cosine"
    warmup_epochs: int = 5
    min_learning_rate: float = 1e-6
    optimizer: Literal["adamw"] = "adamw"
    weight_decay: float = 1e-4
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_epsilon: float = 1e-8
    adamw_amsgrad: bool = False
    classification_loss: Literal["cross_entropy"] = "cross_entropy"
    reconstruction_loss: Literal["l1"] = "l1"
    lambda_cls: float = 1.0
    lambda_vq: float = 1.0
    lambda_rec: float = 0.0
    label_smoothing: float = 0.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000
    tf32: bool = False
    channels_last: bool = False
    top_k: int = 5
    seed: int = 0
    device: str = "auto"
    output_dir: str = "outputs/default"
    resume_from: str | None = None


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    run_name: str = "experiment"
    run_group: str | None = None
    save_source_snapshot: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        raw = _load_toml_with_extends(Path(path))
        config = cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.variant not in {
            "continuous",
            "continuous_bottleneck",
            "vq",
            "random_vq",
        }:
            raise ValueError(f"Unknown model variant: {self.model.variant}")
        if self.model.task not in {
            "classification",
            "segmentation_direct",
            "segmentation_decoder",
        }:
            raise ValueError(f"Unknown task: {self.model.task}")
        expected_classes = {"cifar10": 10, "cifar100": 100, "ade20k": 150}[self.data.dataset]
        if self.model.num_classes != expected_classes:
            raise ValueError(
                f"{self.data.dataset} requires num_classes={expected_classes}, "
                f"got {self.model.num_classes}"
            )
        if self.model.task != "classification" and self.model.variant != "vq":
            raise ValueError("Segmentation tasks require model.variant = 'vq'")
        if self.model.codebook_size < 2 or self.model.latent_dim < 1:
            raise ValueError("codebook_size must be >= 2 and latent_dim must be positive")
        positive_model_values = {
            "input_channels": self.model.input_channels,
            "encoder_stem_channels": self.model.encoder_stem_channels,
            "encoder_intermediate_channels": self.model.encoder_intermediate_channels,
            "encoder_kernel_size": self.model.encoder_kernel_size,
            "residual_kernel_size": self.model.residual_kernel_size,
            "projection_kernel_size": self.model.projection_kernel_size,
            "normalization_epsilon": self.model.normalization_epsilon,
            "group_norm_groups": self.model.group_norm_groups,
            "downsampling_stride": self.model.downsampling_stride,
            "segmentation_decoder_hidden_channels": self.model.segmentation_decoder_hidden_channels,
            "segmentation_decoder_output_channels": self.model.segmentation_decoder_output_channels,
            "pool_kernel_size": self.model.pool_kernel_size,
            "pool_stride": self.model.pool_stride,
            "bottleneck_kernel_size": self.model.bottleneck_kernel_size,
            "head_spatial_pool_size": self.model.head_spatial_pool_size,
            "head_dwconv_kernel_size": self.model.head_dwconv_kernel_size,
            "weighted_pool_hidden_dim": self.model.weighted_pool_hidden_dim,
            "attention_query_count": self.model.attention_query_count,
            "attention_query_init_std": self.model.attention_query_init_std,
            "attention_temperature": self.model.attention_temperature,
            "self_attention_heads": self.model.self_attention_heads,
            "self_attention_mlp_ratio": self.model.self_attention_mlp_ratio,
            "codebook_init_scale": self.model.codebook_init_scale,
        }
        for name, value in positive_model_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "encoder_kernel_size": self.model.encoder_kernel_size,
            "residual_kernel_size": self.model.residual_kernel_size,
            "projection_kernel_size": self.model.projection_kernel_size,
            "bottleneck_kernel_size": self.model.bottleneck_kernel_size,
            "head_dwconv_kernel_size": self.model.head_dwconv_kernel_size,
        }.items():
            if value % 2 == 0:
                raise ValueError(f"{name} must be odd to preserve spatial shape")
        if self.model.codebook_weight < 0 or self.model.commitment_weight < 0:
            raise ValueError("VQ loss weights must be non-negative")
        if not 0 < self.model.batch_norm_momentum <= 1:
            raise ValueError("batch_norm_momentum must be in (0, 1]")
        if (
            self.model.head == "global_self_attention"
            and self.model.latent_dim % self.model.self_attention_heads != 0
        ):
            raise ValueError("latent_dim must be divisible by self_attention_heads")
        if not 1 <= self.model.encoder_blocks <= 8:
            raise ValueError("encoder_blocks must be between 1 and 8")
        if self.model.normalization not in {"group", "batch"}:
            raise ValueError(f"Unknown normalization: {self.model.normalization}")
        if self.model.activation not in {"relu", "silu"}:
            raise ValueError(f"Unknown activation: {self.model.activation}")
        if self.model.head not in {
            "gap",
            "pool_2x2",
            "gap_pool_2x2_concat",
            "dwconv3x3",
            "learned_weighted_pool",
            "single_query_attention_pool",
            "gap_attention_residual",
            "four_query_attention_mean",
            "global_self_attention",
            "dwconv_attention_pool",
        }:
            raise ValueError(f"Unknown classifier head: {self.model.head}")
        if self.model.downsampling not in {
            "strided_conv",
            "max_pool",
            "avg_pool",
            "max_then_avg_pool",
            "avg_then_max_pool",
            "none",
        }:
            raise ValueError(f"Unknown downsampling: {self.model.downsampling}")
        if self.model.input_channels != 3:
            raise ValueError("Configured datasets require input_channels=3")
        if self.data.image_size < 1 or self.data.random_crop_padding < 0:
            raise ValueError("image_size must be positive and crop padding non-negative")
        if self.data.image_shape is not None and (
            len(self.data.image_shape) != 2 or any(value < 1 for value in self.data.image_shape)
        ):
            raise ValueError("image_shape must contain two positive values")
        if self.data.segmentation_ignore_index < 0:
            raise ValueError("segmentation_ignore_index must be non-negative")
        if self.model.task != "classification" and self.data.cutmix_alpha != 0:
            raise ValueError("CutMix is not supported for segmentation tasks")
        if self.model.segmentation_decoder_normalization not in {"none", "batch"}:
            raise ValueError("Unknown segmentation_decoder_normalization")
        if self.model.segmentation_decoder_activation not in {"none", "relu"}:
            raise ValueError("Unknown segmentation_decoder_activation")
        if self.model.segmentation_upsample_mode not in {"nearest", "bilinear"}:
            raise ValueError("Unknown segmentation_upsample_mode")
        if not 0 <= self.data.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        for name, values in {
            "normalization_mean": self.data.normalization_mean,
            "normalization_std": self.data.normalization_std,
        }.items():
            if values is not None and len(values) != self.model.input_channels:
                raise ValueError(f"{name} must have one value per input channel")
        if self.data.normalization_std is not None and any(
            value <= 0 for value in self.data.normalization_std
        ):
            raise ValueError("normalization_std values must be positive")
        if self.model.task != "classification" and (
            self.model.reconstruction or self.train.lambda_rec != 0
        ):
            raise ValueError("Segmentation does not support reconstruction")
        if self.train.lambda_rec > 0 and not self.model.reconstruction:
            raise ValueError("lambda_rec > 0 requires model.reconstruction = true")
        if min(self.train.lambda_cls, self.train.lambda_vq, self.train.lambda_rec) < 0:
            raise ValueError("loss coefficients must be non-negative")
        if not 0 <= self.train.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.train.optimizer != "adamw":
            raise ValueError(f"Unknown optimizer: {self.train.optimizer}")
        if self.train.classification_loss != "cross_entropy":
            raise ValueError(f"Unknown classification loss: {self.train.classification_loss}")
        if self.train.reconstruction_loss != "l1":
            raise ValueError(f"Unknown reconstruction loss: {self.train.reconstruction_loss}")
        if self.train.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(f"Unknown amp_dtype: {self.train.amp_dtype}")
        if self.train.scheduler not in {"cosine", "none"}:
            raise ValueError(f"Unknown scheduler: {self.train.scheduler}")
        if self.train.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.train.adamw_beta1 < 1 or not 0 <= self.train.adamw_beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.train.adamw_epsilon <= 0:
            raise ValueError("adamw_epsilon must be positive")
        if self.train.grad_scaler_init_scale <= 0:
            raise ValueError("grad_scaler_init_scale must be positive")
        if self.train.grad_scaler_growth_factor <= 1:
            raise ValueError("grad_scaler_growth_factor must be > 1")
        if not 0 < self.train.grad_scaler_backoff_factor < 1:
            raise ValueError("grad_scaler_backoff_factor must be in (0, 1)")
        if self.train.grad_scaler_growth_interval < 1:
            raise ValueError("grad_scaler_growth_interval must be positive")
        if self.train.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.train.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.data.cutmix_alpha < 0:
            raise ValueError("cutmix_alpha must be non-negative")
        if not 0 <= self.data.cutmix_probability <= 1:
            raise ValueError("cutmix_probability must be in [0, 1]")
        if not 0 <= self.train.warmup_epochs < self.train.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if not 0 <= self.train.min_learning_rate <= self.train.learning_rate:
            raise ValueError("min_learning_rate must be between 0 and learning_rate")
        if not self.tracking.run_name.strip():
            raise ValueError("tracking.run_name must not be empty")
        if self.tracking.run_group is not None and not self.tracking.run_group.strip():
            raise ValueError("tracking.run_group must not be empty when provided")
