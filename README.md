# Kobeni

Kobeni는 이미지에서 **discrete spatial latent map**을 학습하고, 이 표현이 분류 정보와
공간 정보를 함께 보존하는지 검증하기 위한 연구 코드베이스입니다.

첫 구현 단계는 CIFAR-10에서 아래 네 조건을 동일한 인코더와 학습 설정으로 비교합니다.

- `continuous`: 인코더 특징을 그대로 GAP + Linear에 입력
- `continuous_bottleneck`: VQ와 같은 차원의 연속 bottleneck 사용
- `vq`: 학습 가능한 codebook을 거친 뒤 GAP + Linear에 입력
- `random_vq`: 고정된 random codebook을 사용하는 대조군

## 시작하기

```bash
uv sync --dev
uv run kobeni check --config config/phase1/vq.toml
uv run pytest
```

실제 CIFAR-10 학습은 다음과 같이 시작합니다. 데이터는 기본적으로 `data/`에 내려받고,
결과와 체크포인트는 `outputs/`에 저장합니다.

```bash
uv run kobeni train --config config/phase1/continuous.toml
uv run kobeni train --config config/phase1/vq.toml
```

정식 설정은 5 epoch linear warmup 이후 cosine decay를 적용해 총 200 epoch를 학습합니다.
각 실행 폴더에는 다음 파일이 만들어집니다.

- `config.json`: 실행에 사용된 전체 설정
- `metrics.jsonl`: epoch별 learning rate, train/eval metric, best accuracy
- `last.pt`: 마지막 epoch의 model, optimizer, scheduler 상태
- `best.pt`: 가장 높은 eval accuracy를 기록한 checkpoint

중단된 실행은 설정의 `[train]`에 같은 출력 폴더와 checkpoint를 지정해 재개할 수 있습니다.

```toml
output_dir = "outputs/formal/phase1/vq/seed_0"
resume_from = "outputs/formal/phase1/vq/seed_0/last.pt"
```

기존 `metrics.jsonl`이 있는 폴더에 `resume_from` 없이 새 결과를 덧붙이는 것은 오류로
처리하므로, 새로운 실험은 별도의 `output_dir`을 사용해야 합니다.

`check`는 데이터 다운로드 없이 임의의 32×32 입력으로 모델의 출력 shape, 손실 및
codebook 통계를 확인합니다.

## 구조

```text
config/                 재현 가능한 실험별 TOML 설정
src/kobeni/config.py    설정 로딩과 검증
src/kobeni/data.py      CIFAR-10/100 데이터 로더
src/kobeni/models/      encoder, quantizer, heads, 전체 모델
src/kobeni/training.py  최소 학습/평가 루프와 체크포인트
tests/                  shape, gradient, 설정 스모크 테스트
PLAN.md                 구현 순서와 단계별 완료 조건
```

설계의 기준과 연구 질문은
[`discrete_spatial_visual_vocabulary_project_plan.md`](discrete_spatial_visual_vocabulary_project_plan.md),
구현 순서는 [`PLAN.md`](PLAN.md)를 참고하세요.

1~4개 residual block 비교 결과는
[`reports/depth_sweep_blocks_1_4.md`](reports/depth_sweep_blocks_1_4.md)에 정리되어 있습니다.

## Block-2 continuous baseline

VQ와 직접 비교하는 continuous baseline은 같은 encoder와 64×8×8 bottleneck을 사용하고
quantization만 제거한 `continuous_bottleneck`입니다. 아키텍처 탐색 단계에서는 seed 0 하나만
실행합니다.

```bash
bash scripts/run_continuous_baseline.sh
```

결과는 `outputs/continuous_baseline/blocks_2/seed_0/`에 저장됩니다. Seed 1, 2 설정 파일은
최종 설정이 확정된 뒤 다중-seed 평가에 사용할 수 있도록 보존되어 있지만 위 스크립트에서는
자동 실행하지 않습니다. 여러 seed 실행이 끝난 경우 다음 명령으로 best/final accuracy의
평균과 표준편차를 집계할 수 있습니다.

```bash
uv run kobeni summarize-accuracy \
  --run-dir outputs/continuous_baseline/blocks_2
```

## Class별 code usage

학습된 block-2 VQ의 best checkpoint에서 test split의 `P(code|class)`를 계산합니다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni analyze-code-usage \
  --config config/depth_sweep/blocks_2.toml \
  --checkpoint outputs/depth_sweep/blocks_2/seed_0/best.pt \
  --split test \
  --output-dir outputs/code_usage/blocks_2/seed_0/test
```

출력 폴더에는 다음 파일이 생성됩니다.

- `summary.json`: global/class별 active code, perplexity와 top-10 code
- `code_usage.csv`: count, `P(code|class)`, `P(class|code)` 전체 값
- `code_usage_heatmap.svg`: 행=class, 열=code인 분포 heatmap
- `counts.pt`: 후속 분석을 위한 원본 `[class, code]` count tensor

Train split도 보고 싶다면 `--split train`과 별도의 output directory를 사용합니다. 서로 다른
seed의 code ID는 직접 대응하지 않으므로 seed별 heatmap과 통계는 독립적으로 해석해야 합니다.

## Block-2 200-epoch paired comparison

확정 비교는 seed 0, 1, 2, 3을 GPU 0, 1, 2, 3에 각각 배정합니다. 각 GPU에서는 같은 seed의
`continuous_bottleneck`과 `vq`를 같은 200-epoch optimizer/scheduler 설정으로 순차 실행합니다.
모델 종류가 달라도 같은 seed 쌍은 data shuffle과 augmentation 난수열을 공유합니다.

```bash
bash scripts/run_vq_vs_continuous_200.sh
```

결과와 로그는 `outputs/comparison/block2_200/{continuous_bottleneck,vq}/seed_{0,1,2,3}/`에
저장됩니다. 완료 후 두 모델을 각각 집계합니다.

```bash
uv run kobeni summarize-accuracy \
  --run-dir outputs/comparison/block2_200/continuous_bottleneck
uv run kobeni summarize-accuracy \
  --run-dir outputs/comparison/block2_200/vq
```

개별 실험이 필요하면 기본 TOML을 수정하지 않고 CLI에서 seed와 출력 폴더를 덮어쓸 수 있습니다.

```bash
uv run kobeni train \
  --config config/phase1/vq.toml \
  --seed 3 \
  --output-dir outputs/comparison/block2_200/vq/seed_3
```

## CIFAR-100 dataset comparison

CIFAR-10 비교가 끝난 뒤 같은 paired protocol을 CIFAR-100에서 반복합니다. 두 모델 모두
`num_classes = 100`과 CIFAR-100 전용 normalization을 사용하며 나머지 구조와 학습 조건은
CIFAR-10 비교와 같습니다.

```bash
bash scripts/run_cifar100_vq_vs_continuous_200.sh
```

스크립트는 병렬 실행 전에 CIFAR-100 train/test 데이터를 한 번 준비합니다. 결과와 로그는
`outputs/comparison/cifar100/block2_200/{continuous_bottleneck,vq}/seed_{0,1,2,3}/`에
저장됩니다. 완료 후 다음 명령으로 집계합니다.

```bash
uv run kobeni summarize-accuracy \
  --run-dir outputs/comparison/cifar100/block2_200/continuous_bottleneck
uv run kobeni summarize-accuracy \
  --run-dir outputs/comparison/cifar100/block2_200/vq
```
## Timestamped architecture experiments

새 VQ 구조 실험은 `config/architecture/vq_baseline.toml`을 복사한 뒤 `[model]`과
`tracking.run_name`만 변경하는 방식을 권장합니다. Tracking이 활성화된 설정은 실행할 때마다
월일과 시간이 포함된 새 폴더를 만들므로 기존 결과를 덮어쓰지 않습니다.

```bash
uv run kobeni train --config config/architecture/vq_baseline.toml
```

TOML을 복사하지 않고 실행 이름만 바꾸려면 `--run-name`을 사용할 수 있습니다.

```bash
uv run kobeni train \
  --config config/architecture/vq_baseline.toml \
  --run-name vq_cosine_ema \
  --seed 0
```

단일 실행의 출력 경로는 KST(Asia/Seoul) 기준 timestamp를 사용합니다. 4-GPU 스크립트는 하나의 KST run group을 생성하므로, 같은 실행의 결과가 한 폴더 아래에 묶입니다.

```text
outputs/architecture/<KST_MMDD_HHMMSS_or_run_group>/<run_name>/seed_<N>/
```

각 실행 폴더에는 다음 자료가 저장됩니다.

- `config.json`: 실제 적용된 model/data/train/tracking 설정
- `run_metadata.json`: 시작·종료 시각, GPU, PyTorch/CUDA, Git 상태와 source fingerprint
- `model_summary.json`: 전체·컴포넌트별 파라미터 수와 model 구조
- `source_snapshot/`: 해당 실행 시점의 `src/kobeni/**/*.py`
- `metrics.jsonl`: top-1/top-5, 분리된 loss, codebook 지표, epoch 시간과 peak GPU memory
- `run_summary.json`: best/final/min-loss epoch, 일반화 간격과 최종 지표
- `best.pt`, `last.pt`: checkpoint

여러 실험 결과는 다음 명령으로 한 표에 합칩니다.

```bash
uv run kobeni summarize-experiments --run-dir outputs/architecture
```

이 명령은 `experiment_table.csv`와 `experiment_table.json`을 생성합니다. CSV의 각 행에는
구조 설정, seed, source fingerprint, 파라미터 수, 정확도, loss 구성, code usage, 속도와
메모리가 함께 기록됩니다. 같은 source/config 조합의 여러 seed는 JSON의 `groups`에서
평균과 표준편차로 추가 집계됩니다.
## Four-GPU BatchNorm pooling ablation

CIFAR-100 기본 VQ 구조는 `128 → 1×1 Conv → 64 → VQ`와 두 stage의 MaxPool2d(2)를
사용합니다. 아래 네 설정은 같은 BatchNorm VQ 조건에서 pooling 종류와 순서를 비교한 기록입니다.
모든 실험은 100 epoch, seed 0, effective batch size 128이며 latent map도 모두 `64×8×8`로 같습니다.

| GPU | 설정 | 32→16 stage | 16→8 stage |
|---:|---|---|---|
| 0 | `max_pool` | MaxPool2d(2) | MaxPool2d(2) |
| 1 | `avg_pool` | AvgPool2d(2) | AvgPool2d(2) |
| 2 | `max_then_avg_pool` | MaxPool2d(2) | AvgPool2d(2) |
| 3 | `avg_then_max_pool` | AvgPool2d(2) | MaxPool2d(2) |

각 pool은 해당 stage의 stride-2 convolution을 대체합니다. 따라서 `max→average`와
`average→max`는 pool을 한 stage에 겹쳐 적용하는 것이 아니라, 앞·뒤 downsampling stage의
연산 순서를 뜻합니다.

```bash
bash scripts/run_pooling_ablation.sh
```

완료되면 스크립트가 `outputs/architecture/experiment_table.csv`와 JSON을 갱신합니다.
BatchNorm baseline을 새로 100 epoch 학습하려면 GPU가 비어진 뒤 다음을 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni train 
  --config config/architecture/vq_baseline.toml
```


## Four-GPU codebook-size ablation

BatchNorm + MaxPool-only CIFAR-100 baseline에서 codebook size `K`만 바꿉니다. encoder, 1×1 bottleneck,
latent dimension(64), optimizer, schedule, epoch(100), seed(0)는 모두 동일합니다. 기존 `K=128`
결과와 아래 네 실행을 함께 비교합니다.

| GPU | 설정 | K |
|---:|---|---:|
| 0 | `k_256` | 256 |
| 1 | `k_512` | 512 |
| 2 | `k_1024` | 1024 |
| 3 | `k_2048` | 2048 |

```bash
bash scripts/run_codebook_size_ablation.sh
```

큰 K는 VQ distance matrix 계산량과 메모리가 증가하므로, 실행 결과의 epoch 시간·peak GPU memory도 summary 표에서 함께 비교합니다.


## Four-GPU encoder ablation

기본 조건은 CIFAR-100, BatchNorm, MaxPool-only, ReLU, encoder channel 128, 1×1 bottleneck
`128→64`, VQ `K=128`, 100 epoch, seed 0입니다. 아래 네 run은 한 가지 encoder 변경만 적용합니다.

| GPU | 설정 | 변경 | VQ map |
|---:|---|---|---|
| 0 | `remove_middle_conv` | `3→64→96→128` stem에서 `64→96` 3×3 Conv 제거; `3→64→128`로 단순화 | 64×8×8 |
| 1 | `pool_after_first_conv` | 첫 `3→64` Conv 직후 MaxPool 추가 | 64×4×4 |
| 2 | `silu` | encoder와 residual block의 ReLU를 SiLU로 교체 | 64×8×8 |
| 3 | `channels_256` | encoder 최대 channel `128→256`; bottleneck은 `256→64` | 64×8×8 |

두 번째 실험은 MaxPool이 세 번 적용되므로 token 수가 64에서 16으로 줄어듭니다. 따라서
성능 차이는 activation/pooling 위치뿐 아니라 VQ spatial capacity 변화도 포함해 해석해야 합니다.

```bash
bash scripts/run_encoder_ablation.sh
```

결과는 KST 기준의 하나의 `..._encoder_ablation/` 폴더 아래에 GPU별 run이 함께 저장됩니다.


## Latent dimension and encoder width ablation

이전 latent-width ablation은 CIFAR-100에서 `encoder_channels=256`, `K=128`, `D=64`, BatchNorm,
MaxPool-only, ReLU, 1×1 bottleneck, 100 epoch, seed 0을 사용합니다. 다음 네 run은 이 baseline
대비 latent dimension 또는 encoder width 하나만 변경합니다.

| GPU | 설정 | encoder width | latent D | 변경 |
|---:|---|---:|---:|---|
| 0 | `latent_dim_64` | 256 | 64 | strong baseline 재현 control |
| 1 | `latent_dim_128` | 256 | 128 | bottleneck/VQ embedding `64→128` |
| 2 | `latent_dim_256` | 256 | 256 | bottleneck/VQ embedding `64→256` |
| 3 | `encoder_width_384` | 384 | 64 | encoder 최대 channel `256→384` |

모든 run의 VQ codebook size는 `K=128`, spatial map은 `8×8`로 고정됩니다. D=256은
`256→256` 1×1 transform 뒤 256-dim VQ를 사용합니다.

```bash
bash scripts/run_latent_width_ablation.sh
```

결과는 KST 기준 공통 `..._latent_width_ablation/` 폴더 아래에 저장됩니다.


## Four-GPU augmentation ablation

현재 strong baseline은 CIFAR-100에서 `encoder_channels=384`, `K=128`, `D=64`, BatchNorm,
MaxPool-only, ReLU, 1×1 bottleneck을 사용합니다. 네 run은 이 architecture와 optimizer/schedule을
고정하고 train augmentation만 바꿉니다. 평가는 모든 run에서 augmentation 없이 동일하게 수행합니다.

| GPU | 설정 | RandomCrop | HorizontalFlip | CutMix |
|---:|---|---:|---:|---:|
| 0 | `crop` | Yes | No | No |
| 1 | `flip` | No | Yes | No |
| 2 | `cutmix` | No | No | `alpha=1.0`, p=1.0 |
| 3 | `crop_flip_cutmix` | Yes | Yes | `alpha=1.0`, p=1.0 |

CutMix는 patch를 섞은 비율로 두 label의 cross-entropy를 가중 합산합니다. 따라서 CutMix run의
train accuracy는 mixed label에 대한 가중 정확도이며, 모델 간 비교는 unaugmented eval accuracy를
기준으로 합니다.

```bash
bash scripts/run_augmentation_ablation.sh
```

결과는 KST 공통 `..._augmentation_ablation/` 폴더 아래에 저장되며 summary 표에는 Crop/Flip/
CutMix 설정도 함께 기록됩니다.


## Four-GPU CutMix probability ablation

Crop+HorizontalFlip을 표준 control로 고정하고, CutMix 적용 확률만 비교합니다. 모든 설정은
strong baseline(`encoder=384`, `D=64`, `K=128`, BatchNorm, MaxPool-only, ReLU)과 100 epoch,
seed 0을 공유하며 CutMix의 α는 1.0입니다.

| GPU | 설정 | Crop + Flip | CutMix α | CutMix p |
|---:|---|---:|---:|---:|
| 0 | `crop_flip` | Yes | — | 0.0 (off) |
| 1 | `crop_flip_cutmix_p025` | Yes | 1.0 | 0.25 |
| 2 | `crop_flip_cutmix_p050` | Yes | 1.0 | 0.50 |
| 3 | `crop_flip_cutmix_p075` | Yes | 1.0 | 0.75 |

```bash
bash scripts/run_cutmix_probability_ablation.sh
```

결과는 KST 공통 `..._cutmix_probability_ablation/` 폴더 아래에 저장됩니다.


## Strong-recipe VQ versus continuous comparison

현재 strong recipe는 `W=384`, `D=64`, `K=128`, BatchNorm, MaxPool-only, ReLU,
RandomCrop+HorizontalFlip+CutMix(`α=1.0`, `p=0.5`)입니다. 이 recipe에서 seed 0과 1을
사용해 VQ와 continuous를 paired comparison합니다.

continuous는 raw encoder output이 아니라 VQ와 동일한 `1×1 Conv(384→64)` bottleneck을 쓰는
`continuous_bottleneck`입니다. 따라서 quantization 유무만 비교합니다.

| GPU | variant | seed | VQ loss |
|---:|---|---:|---:|
| 0 | `vq` | 0 | 1.0 |
| 1 | `continuous_bottleneck` | 0 | 0.0 |
| 2 | `vq` | 1 | 1.0 |
| 3 | `continuous_bottleneck` | 1 | 0.0 |

```bash
bash scripts/run_vq_continuous_strong_comparison.sh
```

각 variant의 두 seed는 동일 run name 아래 `seed_0`, `seed_1`로 저장되어 summary JSON에서
평균과 표준편차로 집계됩니다.


## RTX 3090 학습 속도 설정

현재 strong recipe는 batch size 128을 유지하며, CUDA에서만 AMP(FP16), TF32, `channels_last`를
활성화합니다. VQ의 거리 계산·codebook 손실·quantization error는 AMP와 무관하게 FP32로
계산됩니다. 따라서 속도 이득은 CNN과 classifier 쪽에서 얻고 code assignment의 수치 안정성은
유지합니다. `torch.compile`은 아직 기본값으로 켜지지 않으며, 별도 재현성 검증 후 선택적으로
도입합니다.


## Four-GPU classifier-head ablation

strong VQ recipe를 고정하고 classifier head만 비교합니다. 공통 조건은 CIFAR-100,
`encoder=384`, `D=64`, `K=128`, BatchNorm, MaxPool-only, ReLU, Crop+HorizontalFlip+
CutMix(`alpha=1`, `p=0.5`), seed 0, 100 epoch입니다. 모든 run은 AMP, TF32,
`channels_last`를 사용하며 VQ 계산은 FP32로 유지합니다.

| GPU | 설정 | head 구조 | classifier parameter (C=64, CIFAR-100) |
|---:|---|---|---:|
| 0 | `gap` | GAP → Linear | 6,500 |
| 1 | `pool_2x2` | AdaptiveAvgPool(2×2) → Flatten → Linear | 25,700 |
| 2 | `gap_pool_2x2_concat` | GAP + AdaptiveAvgPool(2×2) concat → Linear | 32,100 |
| 3 | `dwconv3x3` | 3×3 depthwise Conv → GAP → Linear | 7,076 |

DWConv에는 activation이나 pointwise Conv를 추가하지 않았고, depthwise kernel만으로 local spatial
interaction의 효과를 측정합니다. `pool_2x2`와 concat head는 8×8 VQ map을 adaptive pooling하므로
coarse 2×2 위치 정보를 보존합니다.

```bash
bash scripts/run_head_ablation.sh
```

결과는 하나의 KST run group 아래에 저장되며 `experiment_table.csv`의 `head` 열로 직접 비교할 수
있습니다.


## Four-GPU pooling-head ablation (200 epochs)

VQ encoder와 augmentation은 strong recipe로 고정하고, latent `64×8×8` map을 classifier로 요약하는
방식만 비교합니다. 공통 조건은 CIFAR-100, `encoder=384`, `D=64`, `K=128`, BatchNorm,
MaxPool-only, Crop+HorizontalFlip+CutMix(`alpha=1`, `p=0.5`), seed 0, 200 epoch입니다.

| GPU | 설정 | head | classifier parameter (C=64, CIFAR-100) |
|---:|---|---|---:|
| 0 | `gap` | GAP → Linear | 6,500 |
| 1 | `dwconv3x3` | 3×3 DWConv → GAP → Linear | 7,076 |
| 2 | `learned_weighted_pool` | token-scoring MLP → softmax weighted sum → Linear | 7,557 |
| 3 | `single_query_attention_pool` | learned query-token similarity → softmax weighted sum → Linear | 6,564 |

`learned_weighted_pool`의 `f(z_i)`는 `64→16→1` 1×1 MLP(GELU)입니다. 이 비선형 scorer를 써서
단순한 query dot-product와 구분됩니다. `single_query_attention_pool`은 64-dim learned query 하나만
추가해 각 이미지에서 content-dependent하게 token 중요도를 계산합니다.

```bash
bash scripts/run_pooling_head_ablation.sh
```

결과는 KST 공통 run group에 저장되며 `experiment_table.csv`의 `head`, parameter, epoch time,
best/final accuracy를 함께 비교할 수 있습니다.


## Four-GPU advanced attention-head ablation (200 epochs)

앞선 pooling-head 비교를 확장해 GAP 안정성, multi-query coverage, token 간 global interaction,
local+selective pooling 조합을 테스트합니다. 공통 조건은 CIFAR-100 strong VQ recipe(`encoder=384`,
`D=64`, `K=128`, BN, MaxPool-only, Crop+Flip+CutMix p=0.5, seed 0)와 200 epoch입니다.

| GPU | 설정 | 핵심 head | classifier parameter (C=64, CIFAR-100) |
|---:|---|---|---:|
| 0 | `gap_attention_residual` | GAP + α·single-query attention | 6,565 |
| 1 | `four_query_attention_mean` | 4 query attention summaries의 평균 | 6,756 |
| 2 | `global_self_attention` | 4-head MHSA + 2× MLP + GAP | 39,972 |
| 3 | `dwconv_attention_pool` | 3×3 DWConv + single-query attention | 7,140 |

Residual head의 α는 0으로 초기화되므로 첫 step은 정확히 GAP이고, 이후 attention residual을 학습합니다.
MHSA는 pre-norm residual Transformer block 1개이며 64개 VQ tokens 사이의 전역 상호작용을 직접
학습합니다.

```bash
bash scripts/run_advanced_attention_head_ablation.sh
```


## Self-contained final VQ baseline

`config/architecture/vq_baseline.toml` is the canonical final VQ classification recipe. It explicitly
declares every experiment-relevant value on the active path: encoder widths/kernels, normalization, pooling,
bottleneck, quantizer precision and initialization, inner/outer loss coefficients, GAP-attention residual
parameters, CIFAR normalization and augmentation, AdamW, scheduler, AMP/GradScaler, TF32, and runtime
batching. Python defaults remain only for compatibility with historical TOMLs. `resume_from` is intentionally
omitted because it is runtime state rather than a hyperparameter.

## ADE20K spatial VQ segmentation

ADE20K은 자동 내려받기를 하지 않습니다. 공식 사이트에서 등록·다운로드한 뒤, 다음처럼
압축을 풀어야 합니다.

```text
data/ade20k/ADEChallengeData2016/
  images/{training,validation}/
  annotations/{training,validation}/
```

두 비교 설정은 모두 Tiny VQ (`64 → K=128`)와 `224×224` 입력을 사용합니다.
`ade20k_direct.toml`은 `56×56` VQ token마다 150-class logit을 만들고 학습 loss용 GT mask만 nearest
resize하여 그 해상도에서 학습합니다. mIoU/pixel accuracy는 logits를 bilinear upsample해 원본 해상도 GT와 계산합니다. `ade20k_decoder.toml`은 같은 quantized token만 받아
`Conv → ×2 upsample → Conv → ×2 upsample → 1×1 Conv`로 `224×224` logit을 만들며 encoder
skip connection은 없습니다. ADE label `1…150`은 training label `0…149`로 변환하며 0과
범위 밖 값은 ignore index 255로 처리됩니다.

```bash
# 다운로드/압축 해제 후 경로만 검증
uv run kobeni prepare-data --config config/segmentation/ade20k_direct.toml

# 모델·loss·shape만 확인; 데이터가 없어도 가능
uv run kobeni check --config config/segmentation/ade20k_direct.toml
uv run kobeni check --config config/segmentation/ade20k_decoder.toml

# GPU 0 direct, GPU 1 decoder를 병렬로 학습
bash scripts/run_ade20k_segmentation.sh
```

각 run은 `outputs/segmentation/<KST_run_group>/<architecture>/seed_0/`에 저장됩니다.
`best.pt`는 validation **mIoU** 기준 checkpoint이고, `metrics.jsonl`에는 pixel accuracy,
mIoU, VQ loss, active code/perplexity, epoch 시간과 peak GPU memory가 기록됩니다.
