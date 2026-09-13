from __future__ import annotations

import torch
from torch import Tensor, nn

from kobeni.config import ModelConfig


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(self.pool(inputs).flatten(1))


class Pool2x2Classifier(nn.Module):
    def __init__(
        self, input_dim: int, num_classes: int, pool_size: int = 2, bias: bool = True
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.linear = nn.Linear(pool_size**2 * input_dim, num_classes, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(self.pool(inputs).flatten(1))


class GlobalSpatialConcatClassifier(nn.Module):
    def __init__(
        self, input_dim: int, num_classes: int, pool_size: int = 2, bias: bool = True
    ) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.linear = nn.Linear((1 + pool_size**2) * input_dim, num_classes, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        global_features = self.global_pool(inputs).flatten(1)
        spatial_features = self.spatial_pool(inputs).flatten(1)
        return self.linear(torch.cat((global_features, spatial_features), dim=1))


class DepthwiseConvClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        kernel_size: int = 3,
        depthwise_bias: bool = False,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_dim,
            bias=depthwise_bias,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(self.pool(self.depthwise(inputs)).flatten(1))


class LearnedWeightedPoolingClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 16,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        token_scores = self.score(inputs).flatten(1)
        token_weights = token_scores.softmax(dim=1).unsqueeze(1)
        pooled = (inputs.flatten(2) * token_weights).sum(dim=2)
        return self.linear(pooled)


def _single_query_pool(inputs: Tensor, query: Tensor, temperature: float) -> Tensor:
    tokens = inputs.flatten(2)
    token_scores = (tokens * query.view(1, -1, 1)).sum(dim=1)
    token_weights = (token_scores / temperature).softmax(dim=1)
    return (tokens * token_weights.unsqueeze(1)).sum(dim=2)


class SingleQueryAttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        query_init_std: float = 0.125,
        temperature: float = 8.0,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(input_dim))
        nn.init.normal_(self.query, std=query_init_std)
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(_single_query_pool(inputs, self.query, self.temperature))


class GapAttentionResidualClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        query_init_std: float = 0.125,
        temperature: float = 8.0,
        alpha_init: float = 0.0,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(input_dim))
        nn.init.normal_(self.query, std=query_init_std)
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        gap_features = inputs.mean(dim=(2, 3))
        attention_features = _single_query_pool(inputs, self.query, self.temperature)
        return self.linear(gap_features + self.alpha * attention_features)


class FourQueryAttentionMeanClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        query_count: int = 4,
        query_init_std: float = 0.125,
        temperature: float = 8.0,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.query_count = query_count
        self.queries = nn.Parameter(torch.empty(query_count, input_dim))
        nn.init.normal_(self.queries, std=query_init_std)
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = inputs.flatten(2)
        queries = self.queries.view(1, self.query_count, -1, 1)
        token_scores = (tokens.unsqueeze(1) * queries).sum(dim=2)
        token_weights = (token_scores / self.temperature).softmax(dim=2)
        query_features = (tokens.unsqueeze(1) * token_weights.unsqueeze(2)).sum(dim=3)
        return self.linear(query_features.mean(dim=1))


class GlobalSelfAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = round(input_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = inputs.flatten(2).transpose(1, 2)
        normalized_tokens = self.norm1(tokens)
        attention_features, _ = self.attention(
            normalized_tokens, normalized_tokens, normalized_tokens, need_weights=False
        )
        tokens = tokens + attention_features
        tokens = tokens + self.mlp(self.norm2(tokens))
        return self.linear(tokens.mean(dim=1))


class DepthwiseAttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        kernel_size: int = 3,
        depthwise_bias: bool = False,
        query_init_std: float = 0.125,
        temperature: float = 8.0,
        classifier_bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_dim,
            bias=depthwise_bias,
        )
        self.query = nn.Parameter(torch.empty(input_dim))
        nn.init.normal_(self.query, std=query_init_std)
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, num_classes, bias=classifier_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        mixed = self.depthwise(inputs)
        return self.linear(_single_query_pool(mixed, self.query, self.temperature))


def build_classifier(config: ModelConfig, input_dim: int) -> nn.Module:
    common = (input_dim, config.num_classes)
    if config.head == "gap":
        return LinearClassifier(*common, bias=config.classifier_bias)
    if config.head == "pool_2x2":
        return Pool2x2Classifier(
            *common, config.head_spatial_pool_size, config.classifier_bias
        )
    if config.head == "gap_pool_2x2_concat":
        return GlobalSpatialConcatClassifier(
            *common, config.head_spatial_pool_size, config.classifier_bias
        )
    if config.head == "dwconv3x3":
        return DepthwiseConvClassifier(
            *common,
            config.head_dwconv_kernel_size,
            config.head_depthwise_bias,
            config.classifier_bias,
        )
    if config.head == "learned_weighted_pool":
        return LearnedWeightedPoolingClassifier(
            *common, config.weighted_pool_hidden_dim, config.classifier_bias
        )
    if config.head == "single_query_attention_pool":
        return SingleQueryAttentionPoolingClassifier(
            *common,
            config.attention_query_init_std,
            config.attention_temperature,
            config.classifier_bias,
        )
    if config.head == "gap_attention_residual":
        return GapAttentionResidualClassifier(
            *common,
            config.attention_query_init_std,
            config.attention_temperature,
            config.attention_residual_alpha_init,
            config.classifier_bias,
        )
    if config.head == "four_query_attention_mean":
        return FourQueryAttentionMeanClassifier(
            *common,
            config.attention_query_count,
            config.attention_query_init_std,
            config.attention_temperature,
            config.classifier_bias,
        )
    if config.head == "global_self_attention":
        return GlobalSelfAttentionClassifier(
            *common,
            config.self_attention_heads,
            config.self_attention_mlp_ratio,
            config.classifier_bias,
        )
    if config.head == "dwconv_attention_pool":
        return DepthwiseAttentionPoolingClassifier(
            *common,
            config.head_dwconv_kernel_size,
            config.head_depthwise_bias,
            config.attention_query_init_std,
            config.attention_temperature,
            config.classifier_bias,
        )
    raise ValueError(f"Unknown classifier head: {config.head}")


class SmallDecoder(nn.Module):
    """Deliberately small 8x8 -> 32x32 decoder for Phase 2."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(latent_dim, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
