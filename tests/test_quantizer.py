import torch

from kobeni.models.quantizer import VectorQuantizer


def test_quantizer_reports_usage_metrics() -> None:
    quantizer = VectorQuantizer(codebook_size=16, embedding_dim=8)
    output = quantizer(torch.randn(2, 8, 4, 4))

    assert output.quantized.shape == (2, 8, 4, 4)
    assert output.indices.shape == (2, 4, 4)
    assert 1 <= output.active_codes.item() <= 16
    assert 1 <= output.perplexity.item() <= 16
    assert 0 <= output.dead_code_fraction.item() < 1
    assert output.quantization_error.item() >= 0



def test_quantizer_keeps_vq_calculation_in_float32() -> None:
    quantizer = VectorQuantizer(codebook_size=16, embedding_dim=8)
    inputs = torch.randn(2, 8, 4, 4, dtype=torch.float16, requires_grad=True)

    output = quantizer(inputs)
    output.loss.backward()

    assert output.quantized.dtype == torch.float32
    assert output.loss.dtype == torch.float32
    assert output.quantization_error.dtype == torch.float32
    assert inputs.grad is not None


def test_quantizer_applies_explicit_inner_loss_weights() -> None:
    quantizer = VectorQuantizer(
        codebook_size=16,
        embedding_dim=8,
        commitment_weight=0.5,
        codebook_weight=2.0,
        codebook_init_scale=0.25,
    )
    output = quantizer(torch.randn(2, 8, 4, 4))

    expected = 2.0 * output.codebook_loss + 0.5 * output.commitment_loss
    torch.testing.assert_close(output.loss, expected)
    assert quantizer.embedding.weight.abs().max() <= 0.25 / (8**0.5)
