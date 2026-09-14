from __future__ import annotations

from kobeni.config import ModelConfig
from kobeni.models.encoder import CifarEncoder
from kobeni.models.quantizer import VectorQuantizer


def build_encoder(config: ModelConfig) -> CifarEncoder:
    return CifarEncoder(
        config.encoder_channels,
        config.encoder_blocks,
        normalization=config.normalization,
        downsampling=config.downsampling,
        activation=config.activation,
        remove_middle_conv=config.remove_middle_conv,
        pool_after_first_conv=config.pool_after_first_conv,
        input_channels=config.input_channels,
        stem_channels=config.encoder_stem_channels,
        intermediate_channels=config.encoder_intermediate_channels,
        encoder_kernel_size=config.encoder_kernel_size,
        residual_kernel_size=config.residual_kernel_size,
        projection_kernel_size=config.projection_kernel_size,
        convolution_bias=config.convolution_bias,
        normalization_epsilon=config.normalization_epsilon,
        batch_norm_momentum=config.batch_norm_momentum,
        group_norm_groups=config.group_norm_groups,
        downsampling_stride=config.downsampling_stride,
        pool_kernel_size=config.pool_kernel_size,
        pool_stride=config.pool_stride,
    )


def build_quantizer(
    config: ModelConfig,
    embedding_dim: int,
    *,
    frozen: bool = False,
) -> VectorQuantizer:
    return VectorQuantizer(
        codebook_size=config.codebook_size,
        embedding_dim=embedding_dim,
        commitment_weight=config.commitment_weight,
        codebook_weight=config.codebook_weight,
        codebook_init_scale=config.codebook_init_scale,
        force_float32=config.quantizer_force_float32,
        frozen=frozen,
    )
