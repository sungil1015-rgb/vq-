from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from kobeni.config import ModelConfig
from kobeni.models.encoder import CifarEncoder
from kobeni.models.heads import SmallDecoder, build_classifier
from kobeni.models.quantizer import VectorQuantizer


@dataclass
class ModelOutput:
    logits: Tensor
    latent: Tensor
    indices: Tensor | None = None
    reconstruction: Tensor | None = None
    vq_loss: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    commitment_loss: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    codebook_loss: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    codebook_metrics: dict[str, Tensor] = field(default_factory=dict)


class SpatialVocabularyModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CifarEncoder(
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
        uses_bottleneck = config.variant != "continuous" and config.use_bottleneck
        classifier_dim = config.latent_dim if uses_bottleneck else config.encoder_channels
        self.bottleneck = (
            nn.Conv2d(
                config.encoder_channels,
                config.latent_dim,
                config.bottleneck_kernel_size,
                padding=config.bottleneck_kernel_size // 2,
                bias=config.bottleneck_bias,
            )
            if uses_bottleneck
            else nn.Identity()
        )
        self.quantizer = (
            VectorQuantizer(
                codebook_size=config.codebook_size,
                embedding_dim=classifier_dim,
                commitment_weight=config.commitment_weight,
                codebook_weight=config.codebook_weight,
                codebook_init_scale=config.codebook_init_scale,
                force_float32=config.quantizer_force_float32,
                frozen=config.variant == "random_vq",
            )
            if config.variant in {"vq", "random_vq"}
            else None
        )
        self.classifier = build_classifier(config, classifier_dim)
        self.decoder = SmallDecoder(classifier_dim) if config.reconstruction else None

    def forward(self, inputs: Tensor) -> ModelOutput:
        latent = self.bottleneck(self.encoder(inputs))
        indices = None
        vq_loss = latent.new_zeros(())
        commitment_loss = latent.new_zeros(())
        codebook_loss = latent.new_zeros(())
        metrics: dict[str, Tensor] = {}

        if self.quantizer is not None:
            quantizer_output = self.quantizer(latent)
            latent = quantizer_output.quantized
            indices = quantizer_output.indices
            vq_loss = quantizer_output.loss
            commitment_loss = quantizer_output.commitment_loss
            codebook_loss = quantizer_output.codebook_loss
            metrics = {
                "perplexity": quantizer_output.perplexity,
                "active_codes": quantizer_output.active_codes,
                "dead_code_fraction": quantizer_output.dead_code_fraction,
                "quantization_error": quantizer_output.quantization_error,
            }

        return ModelOutput(
            logits=self.classifier(latent),
            latent=latent,
            indices=indices,
            reconstruction=self.decoder(latent) if self.decoder is not None else None,
            vq_loss=vq_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            codebook_metrics=metrics,
        )
