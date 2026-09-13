# VQ encoder depth sweep: 1–4 residual blocks

## 실험 목적

VQ 직전 encoder 깊이가 CIFAR-10 분류 성능, codebook 사용량과 일반화에 미치는 영향을
확인한다. 블록 수 외의 latent shape, codebook과 optimization 조건은 고정했다.

## 공통 구조

```text
32×32×3
→ 3×3 Conv, 64 channels, stride 1       (32×32×64)
→ 3×3 Conv, 96 channels, stride 2       (16×16×96)
→ 3×3 Conv, 128 channels, stride 2      (8×8×128)
→ N × ResidualBlock(128 → 128)          (8×8×128)
→ 1×1 Conv bottleneck                   (8×8×64)
→ Vector Quantizer, K=128               (8×8×64 + 8×8 indices)
→ Global Average Pool + Linear          (10 classes)
```

모든 convolution 뒤 normalization은 GroupNorm이다. Residual block 하나는 두 개의 3×3
convolution으로 구성된다. 고정 stem의 receptive field는 9×9이고 block 하나마다 theoretical
receptive field가 16씩 증가한다.

| Blocks | Latent shape | Receptive field | Parameters |
|---:|---:|---:|---:|
| 1 | 64×8×8 | 25×25 | 480,714 |
| 2 | 64×8×8 | 41×41 | 776,138 |
| 3 | 64×8×8 | 57×57 | 1,071,562 |
| 4 | 64×8×8 | 73×73 | 1,366,986 |

## 학습 조건

- CIFAR-10, seed 0
- 100 epochs, batch size 128
- AdamW, peak learning rate 3e-4, weight decay 1e-4
- 5-epoch linear warmup + cosine decay to 1e-6
- Classification loss + VQ loss, reconstruction 없음
- GPU 0/1/2/3에 blocks 1/2/3/4를 각각 배정
- GPU 0은 기존 별도 학습과 compute를 공유했으므로 runtime 비교에는 사용하지 않음

## 결과

아래 codebook 지표는 각 모델의 best-accuracy epoch에서 측정했다.

| Blocks | Best epoch | Best eval acc. | Final acc. | Last-10 acc. | Train–eval gap | Perplexity | Active | Quant. error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98 | 85.05% | 84.82% | 84.96 ± 0.07% | 7.36%p | 114.00 | 128 | 0.1684 |
| 2 | 90 | 87.56% | 87.23% | 87.30 ± 0.06% | 9.11%p | 122.49 | 128 | 0.1035 |
| 3 | 87 | 87.28% | 87.15% | 87.13 ± 0.07% | 10.38%p | 102.51 | 128 | 0.0645 |
| 4 | 94 | **87.89%** | 87.63% | **87.69 ± 0.09%** | 10.68%p | 75.61 | 128 | **0.0481** |

## 해석

1. 한 block은 분류 성능이 부족하다. 두 번째 block 추가로 best accuracy가 2.51%p 상승했다.
2. 두 block 이후 정확도 증가는 포화된다. 네 block은 두 block보다 0.33%p 높지만 파라미터가
   590,848개, 약 76% 더 많다.
3. 깊이가 증가할수록 quantization error는 단조 감소한다. 더 깊은 feature가 code vector에
   가까운 compact cluster를 형성한다.
4. 낮은 quantization error가 더 좋은 vocabulary를 뜻하지는 않는다. Block 4는 모든 code가
   한 번 이상 사용됐지만 perplexity가 75.61로 내려가 사용 분포가 상당히 편향됐다.
5. train–eval gap은 깊이와 함께 증가했다. Block 2 이후 추가 깊이는 주로 학습 데이터 적합도를
   높였고 일반화 이득은 작았다.

## Blocks 2 vs 4: three-seed validation

Blocks 2와 4를 seed 0, 1, 2에서 비교했다. 평균의 `±` 값은 세 seed 사이의 sample standard
deviation이다.

| Blocks | Seed 0 | Seed 1 | Seed 2 | Best acc. mean | Final acc. mean | Train–eval gap | Perplexity | Quant. error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 87.56% | 87.01% | 87.44% | 87.34 ± 0.29% | 87.10 ± 0.23% | 9.37%p | 123.38 | 0.1066 |
| 4 | 87.89% | 87.16% | 87.53% | **87.53 ± 0.37%** | **87.34 ± 0.32%** | 10.80%p | 71.12 | **0.0515** |

동일 seed에서 `blocks 4 - blocks 2`의 best accuracy 차이는 각각 `+0.33%p`, `+0.15%p`,
`+0.09%p`이며 평균은 `+0.19 ± 0.12%p`다. Block 4가 세 seed 모두 근소하게 높지만,
효과 크기는 작다. Block 4의 seed 2에서는 best epoch에 121/128 code만 활성 상태였다.

## 결정

기본 encoder는 **2 blocks**로 결정한다. Block 4는 평균 accuracy가 0.19%p 높지만 파라미터가
76% 더 많고, train–eval gap은 1.43%p 크며, 평균 perplexity는 52.26 낮다. 순수 분류 정확도만
최대화하면 block 4를 고를 수 있지만, 이 프로젝트의 목표인 compact하고 고르게 사용되는 spatial
vocabulary에는 block 2가 더 적합하다.

Spatial locality가 주 목표이므로 reconstruction과 token intervention 단계에서는 receptive
field가 작은 block 2의 선택을 accuracy만으로 확정하지 않고 token deletion의 영향 반경으로
다시 검증한다.

## 산출물

각 실행의 config, 전체 epoch metrics, 최고/마지막 checkpoint는 다음 위치에 있다.

```text
outputs/depth_sweep/blocks_{1,3}/seed_0/
outputs/depth_sweep/blocks_{2,4}/seed_{0,1,2}/
├── config.json
├── metrics.jsonl
├── best.pt
├── last.pt
└── train.log
```
