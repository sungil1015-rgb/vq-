import pytest
import torch

from kobeni.config import ModelConfig, ModelVariant
from kobeni.models import SpatialVocabularyModel


@pytest.mark.parametrize(
    ("variant", "expected_channels", "has_indices"),
    [
        ("continuous", 384, False),
        ("continuous_bottleneck", 64, False),
        ("vq", 64, True),
        ("random_vq", 64, True),
    ],
)
def test_phase1_model_shapes(
    variant: ModelVariant, expected_channels: int, has_indices: bool
) -> None:
    model = SpatialVocabularyModel(ModelConfig(variant=variant))
    output = model(torch.randn(2, 3, 32, 32))

    assert output.logits.shape == (2, 10)
    assert output.latent.shape == (2, expected_channels, 8, 8)
    assert (output.indices is not None) is has_indices
    if output.indices is not None:
        assert output.indices.shape == (2, 8, 8)


def test_vq_straight_through_and_codebook_receive_gradients() -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="vq"))
    output = model(torch.randn(2, 3, 32, 32))
    (output.logits.mean() + output.vq_loss).backward()

    assert model.encoder.stem[0].weight.grad is not None
    assert model.quantizer is not None
    assert model.quantizer.embedding.weight.grad is not None


@pytest.mark.parametrize(
    ("height", "width", "expected_grid"),
    [
        (28, 28, (7, 7)),
        (40, 28, (10, 7)),
    ],
)
def test_vq_grid_follows_runtime_image_shape(
    height: int, width: int, expected_grid: tuple[int, int]
) -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="vq"))
    output = model(torch.randn(2, 3, height, width))

    assert output.latent.shape == (2, 64, *expected_grid)
    assert output.indices is not None
    assert output.indices.shape == (2, *expected_grid)


def test_random_codebook_is_frozen() -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="random_vq"))

    assert model.quantizer is not None
    assert not model.quantizer.embedding.weight.requires_grad


def test_small_decoder_restores_image_shape() -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="vq", reconstruction=True))
    output = model(torch.randn(2, 3, 32, 32))

    assert output.reconstruction is not None
    assert output.reconstruction.shape == (2, 3, 32, 32)


def test_encoder_depth_keeps_shape_and_increases_capacity() -> None:
    parameter_counts = []
    for blocks in range(1, 5):
        model = SpatialVocabularyModel(ModelConfig(variant="vq", encoder_blocks=blocks))
        output = model(torch.randn(1, 3, 32, 32))
        assert output.latent.shape == (1, 64, 8, 8)
        parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))

    assert parameter_counts == sorted(parameter_counts)
    assert len(set(parameter_counts)) == 4


@pytest.mark.parametrize(
    ("config", "expected_shape", "uses_batch_norm"),
    [
        (
            ModelConfig(
                variant="vq", encoder_channels=128, use_bottleneck=False, normalization="group"
            ),
            (2, 128, 8, 8),
            False,
        ),
        (
            ModelConfig(variant="vq", downsampling="max_pool", normalization="group"),
            (2, 64, 8, 8),
            False,
        ),
        (
            ModelConfig(variant="vq", normalization="batch"),
            (2, 64, 8, 8),
            True,
        ),
        (
            ModelConfig(
                variant="vq",
                encoder_channels=128,
                use_bottleneck=False,
                normalization="group",
                downsampling="none",
            ),
            (2, 128, 32, 32),
            False,
        ),
    ],
)
def test_encoder_ablation_shapes(
    config: ModelConfig, expected_shape: tuple[int, int, int, int], uses_batch_norm: bool
) -> None:
    model = SpatialVocabularyModel(config)
    output = model(torch.randn(2, 3, 32, 32))

    assert tuple(output.latent.shape) == expected_shape
    assert output.indices is not None
    assert output.indices.shape == expected_shape[:1] + expected_shape[2:]
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_batch_norm is uses_batch_norm


@pytest.mark.parametrize(
    ("downsampling", "expected_pool_names"),
    [
        ("max_pool", ["MaxPool2d", "MaxPool2d"]),
        ("avg_pool", ["AvgPool2d", "AvgPool2d"]),
        ("max_then_avg_pool", ["MaxPool2d", "AvgPool2d"]),
        ("avg_then_max_pool", ["AvgPool2d", "MaxPool2d"]),
    ],
)
def test_pooling_order_preserves_spatial_shape(
    downsampling: str, expected_pool_names: list[str]
) -> None:
    model = SpatialVocabularyModel(
        ModelConfig(variant="vq", normalization="batch", downsampling=downsampling)
    )
    output = model(torch.randn(2, 3, 32, 32))
    pool_names = [
        type(module).__name__
        for module in model.encoder.stem
        if isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d))
    ]

    assert tuple(output.latent.shape) == (2, 64, 8, 8)
    assert pool_names == expected_pool_names


def test_default_model_uses_batchnorm_and_max_pooling() -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="vq"))
    pool_names = [
        type(module).__name__
        for module in model.encoder.stem
        if isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d))
    ]

    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert pool_names == ["MaxPool2d", "MaxPool2d"]


@pytest.mark.parametrize(
    ("config", "expected_shape", "expected_pool_count", "expected_stem_convs"),
    [
        (
            ModelConfig(variant="vq", remove_middle_conv=True),
            (2, 64, 8, 8),
            2,
            2,
        ),
        (
            ModelConfig(variant="vq", pool_after_first_conv=True),
            (2, 64, 4, 4),
            3,
            3,
        ),
        (
            ModelConfig(variant="vq", activation="silu"),
            (2, 64, 8, 8),
            2,
            3,
        ),
        (
            ModelConfig(variant="vq", encoder_channels=256),
            (2, 64, 8, 8),
            2,
            3,
        ),
    ],
)
def test_encoder_architecture_ablations(
    config: ModelConfig,
    expected_shape: tuple[int, int, int, int],
    expected_pool_count: int,
    expected_stem_convs: int,
) -> None:
    model = SpatialVocabularyModel(config)
    output = model(torch.randn(2, 3, 32, 32))
    pool_count = sum(isinstance(module, torch.nn.MaxPool2d) for module in model.encoder.stem)
    stem_conv_count = sum(isinstance(module, torch.nn.Conv2d) for module in model.encoder.stem)

    assert tuple(output.latent.shape) == expected_shape
    assert pool_count == expected_pool_count
    assert stem_conv_count == expected_stem_convs


def test_silu_replaces_relu_and_wider_encoder_reaches_bottleneck() -> None:
    silu_model = SpatialVocabularyModel(ModelConfig(variant="vq", activation="silu"))
    wide_model = SpatialVocabularyModel(ModelConfig(variant="vq", encoder_channels=256))

    assert any(isinstance(module, torch.nn.SiLU) for module in silu_model.modules())
    assert not any(isinstance(module, torch.nn.ReLU) for module in silu_model.modules())
    assert isinstance(wide_model.bottleneck, torch.nn.Conv2d)
    assert wide_model.bottleneck.in_channels == 256


@pytest.mark.parametrize(
    ("encoder_channels", "latent_dim"),
    [(256, 64), (256, 128), (256, 256), (384, 64)],
)
def test_strong_baseline_latent_and_width_variants(encoder_channels: int, latent_dim: int) -> None:
    model = SpatialVocabularyModel(
        ModelConfig(
            variant="vq",
            encoder_channels=encoder_channels,
            latent_dim=latent_dim,
        )
    )
    output = model(torch.randn(2, 3, 32, 32))

    assert tuple(output.latent.shape) == (2, latent_dim, 8, 8)
    assert model.quantizer is not None
    assert model.quantizer.embedding.weight.shape == (128, latent_dim)
    assert isinstance(model.bottleneck, torch.nn.Conv2d)
    assert model.bottleneck.in_channels == encoder_channels


@pytest.mark.parametrize(
    ("head", "classifier_parameters"),
    [
        ("gap", 650),
        ("pool_2x2", 2570),
        ("gap_pool_2x2_concat", 3210),
        ("dwconv3x3", 1226),
        ("learned_weighted_pool", 1707),
        ("single_query_attention_pool", 714),
        ("gap_attention_residual", 715),
        ("four_query_attention_mean", 906),
        ("global_self_attention", 34122),
        ("dwconv_attention_pool", 1290),
    ],
)
def test_classifier_heads_preserve_logits_shape(head: str, classifier_parameters: int) -> None:
    model = SpatialVocabularyModel(ModelConfig(variant="vq", head=head))
    output = model(torch.randn(2, 3, 32, 32))

    assert output.logits.shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.classifier.parameters()) == (
        classifier_parameters
    )


def test_baseline_architecture_hyperparameters_reach_modules() -> None:
    config = ModelConfig(
        variant="vq",
        encoder_stem_channels=32,
        encoder_intermediate_channels=48,
        encoder_channels=80,
        encoder_kernel_size=5,
        residual_kernel_size=5,
        pool_kernel_size=2,
        pool_stride=2,
        bottleneck_kernel_size=3,
        attention_query_init_std=0.02,
        attention_temperature=3.5,
        attention_residual_alpha_init=0.25,
        head="gap_attention_residual",
    )
    model = SpatialVocabularyModel(config)
    stem_convs = [module for module in model.encoder.stem if isinstance(module, torch.nn.Conv2d)]

    assert stem_convs[0].out_channels == 32
    assert stem_convs[1].out_channels == 48
    assert stem_convs[0].kernel_size == (5, 5)
    assert model.encoder.blocks[0].body[0].kernel_size == (5, 5)
    assert model.bottleneck.kernel_size == (3, 3)
    assert model.classifier.temperature == pytest.approx(3.5)
    assert model.classifier.alpha.item() == pytest.approx(0.25)
