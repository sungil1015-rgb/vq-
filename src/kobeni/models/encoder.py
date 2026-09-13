from __future__ import annotations

from torch import Tensor, nn

from kobeni.config import ActivationKind, DownsamplingKind, NormalizationKind


def _normalization(
    channels: int,
    kind: NormalizationKind,
    epsilon: float,
    batch_norm_momentum: float,
    group_norm_groups: int,
) -> nn.Module:
    if kind == "group":
        groups = min(group_norm_groups, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=epsilon)
    if kind == "batch":
        return nn.BatchNorm2d(channels, eps=epsilon, momentum=batch_norm_momentum)
    raise ValueError(f"Unknown normalization: {kind}")


def _activation(kind: ActivationKind) -> nn.Module:
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {kind}")


def _pooling_layer(
    downsampling: DownsamplingKind,
    stage_index: int,
    kernel_size: int,
    stride: int,
) -> nn.Module:
    if downsampling == "max_pool":
        return nn.MaxPool2d(kernel_size, stride=stride)
    if downsampling == "avg_pool":
        return nn.AvgPool2d(kernel_size, stride=stride)
    if downsampling == "max_then_avg_pool":
        pool = nn.MaxPool2d if stage_index == 0 else nn.AvgPool2d
        return pool(kernel_size, stride=stride)
    if downsampling == "avg_then_max_pool":
        pool = nn.AvgPool2d if stage_index == 0 else nn.MaxPool2d
        return pool(kernel_size, stride=stride)
    raise ValueError(f"Unknown pooling downsampling: {downsampling}")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: NormalizationKind,
        activation: ActivationKind,
        kernel_size: int,
        projection_kernel_size: int,
        convolution_bias: bool,
        normalization_epsilon: float,
        batch_norm_momentum: float,
        group_norm_groups: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        projection_padding = projection_kernel_size // 2

        def norm(channels: int) -> nn.Module:
            return _normalization(
                channels,
                normalization,
                normalization_epsilon,
                batch_norm_momentum,
                group_norm_groups,
            )

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=convolution_bias,
            ),
            norm(out_channels),
            _activation(activation),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=convolution_bias,
            ),
            norm(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    projection_kernel_size,
                    stride=stride,
                    padding=projection_padding,
                    bias=convolution_bias,
                ),
                norm(out_channels),
            )
        )
        self.activation = _activation(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class CifarEncoder(nn.Module):
    """CIFAR encoder whose experiment-relevant constants are configurable."""

    def __init__(
        self,
        output_channels: int = 384,
        num_blocks: int = 2,
        normalization: NormalizationKind = "batch",
        downsampling: DownsamplingKind = "max_pool",
        activation: ActivationKind = "relu",
        remove_middle_conv: bool = False,
        pool_after_first_conv: bool = False,
        input_channels: int = 3,
        stem_channels: int = 64,
        intermediate_channels: int = 96,
        encoder_kernel_size: int = 3,
        residual_kernel_size: int = 3,
        projection_kernel_size: int = 1,
        convolution_bias: bool = False,
        normalization_epsilon: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        group_norm_groups: int = 8,
        downsampling_stride: int = 2,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if downsampling not in {
            "strided_conv",
            "max_pool",
            "avg_pool",
            "max_then_avg_pool",
            "avg_then_max_pool",
            "none",
        }:
            raise ValueError(f"Unknown downsampling: {downsampling}")
        if activation not in {"relu", "silu"}:
            raise ValueError(f"Unknown activation: {activation}")

        def norm(channels: int) -> nn.Module:
            return _normalization(
                channels,
                normalization,
                normalization_epsilon,
                batch_norm_momentum,
                group_norm_groups,
            )

        pooling_downsampling = downsampling in {
            "max_pool",
            "avg_pool",
            "max_then_avg_pool",
            "avg_then_max_pool",
        }
        first_stride = (
            downsampling_stride
            if remove_middle_conv and downsampling == "strided_conv"
            else 1
        )
        encoder_padding = encoder_kernel_size // 2
        stem_layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                stem_channels,
                encoder_kernel_size,
                stride=first_stride,
                padding=encoder_padding,
                bias=convolution_bias,
            ),
            norm(stem_channels),
            _activation(activation),
        ]
        if remove_middle_conv and pooling_downsampling:
            stem_layers.append(
                _pooling_layer(
                    downsampling, 0, pool_kernel_size, pool_stride
                )
            )
        if pool_after_first_conv:
            stem_layers.append(nn.MaxPool2d(pool_kernel_size, stride=pool_stride))

        stages = (
            ((stem_channels, output_channels, 1),)
            if remove_middle_conv
            else (
                (stem_channels, intermediate_channels, 0),
                (intermediate_channels, output_channels, 1),
            )
        )
        for in_channels, out_channels, stage_index in stages:
            stride = downsampling_stride if downsampling == "strided_conv" else 1
            stem_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        encoder_kernel_size,
                        stride=stride,
                        padding=encoder_padding,
                        bias=convolution_bias,
                    ),
                    norm(out_channels),
                    _activation(activation),
                ]
            )
            if pooling_downsampling:
                stem_layers.append(
                    _pooling_layer(
                        downsampling,
                        stage_index,
                        pool_kernel_size,
                        pool_stride,
                    )
                )

        self.stem = nn.Sequential(*stem_layers)
        self.blocks = nn.Sequential(
            *(
                ResidualBlock(
                    output_channels,
                    output_channels,
                    normalization,
                    activation,
                    residual_kernel_size,
                    projection_kernel_size,
                    convolution_bias,
                    normalization_epsilon,
                    batch_norm_momentum,
                    group_norm_groups,
                )
                for _ in range(num_blocks)
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(self.stem(inputs))
