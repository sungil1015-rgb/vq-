from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass
class QuantizerOutput:
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    perplexity: Tensor
    active_codes: Tensor
    dead_code_fraction: Tensor
    quantization_error: Tensor


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        codebook_init_scale: float = 1.0,
        force_float32: bool = True,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.force_float32 = force_float32
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        bound = codebook_init_scale / math.sqrt(embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bound, bound)
        self.embedding.weight.requires_grad_(not frozen)

    def forward(self, inputs: Tensor) -> QuantizerOutput:
        if inputs.ndim != 4 or inputs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected [B, {self.embedding_dim}, H, W], got {tuple(inputs.shape)}"
            )

        precision_context = (
            torch.autocast(device_type=inputs.device.type, enabled=False)
            if self.force_float32
            else nullcontext()
        )
        with precision_context:
            quantizer_inputs = inputs.float() if self.force_float32 else inputs
            flat = quantizer_inputs.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
            codebook = (
                self.embedding.weight.float()
                if self.force_float32
                else self.embedding.weight
            )
            distances = (
                flat.square().sum(dim=1, keepdim=True)
                + codebook.square().sum(dim=1)
                - 2.0 * flat @ codebook.t()
            )
            flat_indices = distances.argmin(dim=1)
            raw_quantized = functional.embedding(flat_indices, codebook)
            raw_quantized = raw_quantized.view(
                inputs.shape[0], inputs.shape[2], inputs.shape[3], self.embedding_dim
            ).permute(0, 3, 1, 2)

            commitment_loss = functional.mse_loss(quantizer_inputs, raw_quantized.detach())
            if codebook.requires_grad:
                codebook_loss = functional.mse_loss(raw_quantized, quantizer_inputs.detach())
            else:
                codebook_loss = quantizer_inputs.new_zeros(())
            loss = self.codebook_weight * codebook_loss + self.commitment_weight * commitment_loss
            quantized = quantizer_inputs + (raw_quantized - quantizer_inputs).detach()

            counts = torch.bincount(flat_indices, minlength=self.codebook_size)
            probabilities = counts.float() / counts.sum().clamp_min(1)
            nonzero = probabilities > 0
            entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            active_codes = nonzero.sum()

        return QuantizerOutput(
            quantized=quantized,
            indices=flat_indices.view(inputs.shape[0], inputs.shape[2], inputs.shape[3]),
            loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=entropy.exp(),
            active_codes=active_codes,
            dead_code_fraction=1.0 - active_codes.float() / self.codebook_size,
            quantization_error=functional.mse_loss(
                quantizer_inputs.detach(), raw_quantized.detach()
            ),
        )
