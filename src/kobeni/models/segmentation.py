from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from kobeni.config import ModelConfig
from kobeni.models.components import build_encoder, build_quantizer
from kobeni.models.system import ModelOutput


class LightweightSegmentationDecoder(nn.Module):
    """Two-stage, skip-free decoder used only with quantized VQ features."""

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        output_channels: int,
        num_classes: int,
        upsample_mode: str,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.block1 = self._build_block(latent_dim, hidden_channels, normalization, activation)
        self.block2 = self._build_block(hidden_channels, output_channels, normalization, activation)
        self.classifier = nn.Conv2d(output_channels, num_classes, kernel_size=1)

    @staticmethod
    def _build_block(
        input_channels: int,
        output_channels: int,
        normalization: str,
        activation: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        ]
        if normalization == "batch":
            layers.append(nn.BatchNorm2d(output_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _upsample(self, inputs: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        arguments: dict[str, object] = {"mode": self.upsample_mode}
        if self.upsample_mode == "bilinear":
            arguments["align_corners"] = False
        if size is None:
            arguments["scale_factor"] = 2.0
        else:
            arguments["size"] = size
        return functional.interpolate(inputs, **arguments)

    def forward(self, inputs: Tensor, output_size: tuple[int, int]) -> Tensor:
        decoded = self.block1(inputs)
        decoded = self._upsample(decoded)
        decoded = self.block2(decoded)
        decoded = self._upsample(decoded)
        logits = self.classifier(decoded)
        if logits.shape[-2:] != output_size:
            logits = self._upsample(logits, output_size)
        return logits


class SpatialVocabularySegmentationModel(nn.Module):
    """VQ segmentation without encoder-to-decoder skip connections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.task not in {"segmentation_direct", "segmentation_decoder"}:
            raise ValueError(f"Expected a segmentation task, got {config.task}")
        self.encoder = build_encoder(config)
        self.bottleneck = nn.Conv2d(
            config.encoder_channels,
            config.latent_dim,
            config.bottleneck_kernel_size,
            padding=config.bottleneck_kernel_size // 2,
            bias=config.bottleneck_bias,
        )
        self.quantizer = build_quantizer(config, config.latent_dim)
        self.direct_classifier = (
            nn.Conv2d(config.latent_dim, config.num_classes, kernel_size=1)
            if config.task == "segmentation_direct"
            else None
        )
        self.decoder = (
            LightweightSegmentationDecoder(
                config.latent_dim,
                config.segmentation_decoder_hidden_channels,
                config.segmentation_decoder_output_channels,
                config.num_classes,
                config.segmentation_upsample_mode,
                config.segmentation_decoder_normalization,
                config.segmentation_decoder_activation,
            )
            if config.task == "segmentation_decoder"
            else None
        )

    def forward(self, inputs: Tensor) -> ModelOutput:
        latent = self.bottleneck(self.encoder(inputs))
        quantizer_output = self.quantizer(latent)
        quantized = quantizer_output.quantized
        if self.direct_classifier is not None:
            logits = self.direct_classifier(quantized)
        else:
            assert self.decoder is not None
            logits = self.decoder(quantized, inputs.shape[-2:])
        return ModelOutput(
            logits=logits,
            latent=quantized,
            indices=quantizer_output.indices,
            vq_loss=quantizer_output.loss,
            commitment_loss=quantizer_output.commitment_loss,
            codebook_loss=quantizer_output.codebook_loss,
            codebook_metrics={
                "perplexity": quantizer_output.perplexity,
                "active_codes": quantizer_output.active_codes,
                "dead_code_fraction": quantizer_output.dead_code_fraction,
                "quantization_error": quantizer_output.quantization_error,
            },
        )
