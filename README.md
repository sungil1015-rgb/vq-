# Kobeni

Kobeni는 이미지의 각 공간 위치를 연속 feature가 아닌 **학습 가능한 discrete code**로
표현하고, 이 visual vocabulary가 분류와 semantic segmentation에 얼마나 유용한지 비교하는
PyTorch 연구 코드베이스입니다.

현재 작업은 두 축으로 구성됩니다.

- CIFAR-100: VQ 분류 모델의 encoder, codebook, augmentation, pooling head와 학습 설정 비교
- ADE20K: 같은 spatial VQ encoder를 direct probe 또는 lightweight decoder에 연결한 segmentation

## 현재 기준 모델

분류 strong baseline은
[config/architecture/vq_baseline.toml](config/architecture/vq_baseline.toml)입니다. 실험에 영향을
주는 구조·loss·optimizer·runtime 값이 모두 명시되어 있으며, 새 실험 설정은 이 파일을
extends한 뒤 바뀌는 값만 덮어쓰는 방식을 사용합니다.

~~~text
CIFAR-100 32×32 RGB
  → Conv 3×3, 3→64 + BN + ReLU
  → Conv 3×3, 64→96 + BN + ReLU + MaxPool
  → Conv 3×3, 96→384 + BN + ReLU + MaxPool
  → ResidualBlock(384) × 2
  → 1×1 Conv, 384→64
  → Vector Quantizer, D=64, K=128
  → GAP + α·single-query attention pooling
  → Linear, 64→100
~~~

32×32 입력은 두 번의 2×2 MaxPool을 거쳐 8×8 VQ grid, 즉 이미지당 64개 token이 됩니다.
입력 크기를 H×W로 바꾸면 VQ grid도 대략 floor(H/4)×floor(W/4)로 변합니다.

현재 주요 학습 설정은 다음과 같습니다.

| 항목 | 값 |
| --- | --- |
| Dataset | CIFAR-100 |
| Epoch / batch | 200 / 128 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Scheduler | 5 epoch warmup + cosine decay |
| Augmentation | RandomCrop + HorizontalFlip + CutMix(alpha=1, p=0.5) |
| VQ commitment beta | 0.35 |
| Outer VQ coefficient | lambda_vq=1.0 |
| Runtime | AMP FP16 + TF32 + channels-last, VQ 계산은 FP32 |

전체 loss는 다음 구조입니다.

~~~text
VQ loss    = codebook_weight × codebook_loss
           + commitment_weight × commitment_loss

Total loss = lambda_cls × classification_or_segmentation_loss
           + lambda_vq  × VQ loss
           + lambda_rec × reconstruction_loss
~~~

현재 baseline은 reconstruction을 사용하지 않으므로 lambda_rec=0입니다. GAP-attention residual의
alpha는 0으로 초기화되어 학습 시작 시에는 정확히 GAP처럼 동작하고, 이후 필요한 attention
성분만 학습합니다.

## 설치와 빠른 확인

Python 3.12와 uv를 사용합니다.

~~~bash
cd /home/shizuku/HXH/kobeni
uv sync --dev
uv run pytest
uv run ruff check .
~~~

데이터를 받거나 긴 학습을 시작하기 전에 설정, forward, loss, backward와 tensor shape를
확인할 수 있습니다.

~~~bash
uv run kobeni check --config config/architecture/vq_baseline.toml
uv run kobeni check --config config/segmentation/ade20k_direct.toml
~~~

| 명령 | 역할 |
| --- | --- |
| kobeni check | 임의 입력으로 모델·loss·shape 검사 |
| kobeni prepare-data | 데이터 다운로드 또는 경로 검증 |
| kobeni train | 단일 설정 학습 |
| kobeni analyze-code-usage | class별 code 사용 분포 계산 |
| kobeni summarize-accuracy | seed별 분류 정확도 집계 |
| kobeni summarize-experiments | 분류/segmentation 실행 결과 통합 |
| kobeni tune | Optuna worker 실행 |
| kobeni summarize-tuning | Optuna 전체 trial과 상위 4개 설정 저장 |

## 단일 CIFAR-100 학습

CIFAR-10/100은 torchvision을 통해 처음 실행할 때 자동으로 data/에 다운로드됩니다.

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni train \
  --config config/architecture/vq_baseline.toml
~~~

seed와 실행 이름은 원본 TOML을 수정하지 않고 CLI에서 바꿀 수 있습니다.

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni train \
  --config config/architecture/vq_baseline.toml \
  --seed 1 \
  --run-name baseline_seed_1
~~~

device=auto는 CUDA가 정상적으로 초기화되면 CUDA를, 그렇지 않으면 CPU를 선택합니다. 다만
NVIDIA 드라이버 자체가 깨진 상태에서는 CPU 전환 전에 CUDA 초기화 오류가 날 수 있으므로
학습 전 nvidia-smi 확인을 권장합니다.

## 모델 variant

| Variant | 설명 |
| --- | --- |
| continuous | encoder의 384-channel feature를 바로 classifier에 입력 |
| continuous_bottleneck | VQ와 같은 1×1 Conv 384→64를 쓰되 quantization만 제거 |
| vq | 학습되는 K개 code vector 중 가장 가까운 code로 치환 |
| random_vq | 고정 random codebook을 사용하는 control |

VQ 효과만 비교하려면 encoder와 bottleneck이 같은 vq와 continuous_bottleneck을 비교해야 합니다.

~~~bash
bash scripts/run_vq_continuous_strong_comparison.sh
~~~

이 스크립트는 seed 0/1의 VQ와 continuous bottleneck을 GPU 0~3에 하나씩 배정합니다.

## 설정 상속과 모델 크기

TOML 최상단의 extends는 부모 설정을 재귀적으로 병합합니다.

~~~toml
extends = "../vq_baseline.toml"

[model]
encoder_channels = 512
encoder_blocks = 3

[tracking]
run_name = "my_small_variant"
~~~

| Preset | Stem | Intermediate | Width | Blocks | Latent D | Codebook K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 64 | 96 | 384 | 2 | 64 | 128 |
| Small | 64 | 128 | 512 | 3 | 64 | 128 |
| Base | 96 | 192 | 768 | 4 | 96 | 256 |
| Large | 128 | 256 | 1024 | 6 | 128 | 256 |
| XLarge | 160 | 320 | 1280 | 8 | 128 | 512 |

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni train \
  --config config/architecture/model_scale/small.toml
~~~

## 4-GPU architecture 실험

scripts/의 launcher는 하나의 한국시간 run group을 만든 뒤 GPU별 설정을 병렬 실행합니다.

| 스크립트 | 비교 대상 |
| --- | --- |
| run_layer_ablation.sh | 1×1 Conv, downsampling, normalization |
| run_pooling_ablation.sh | Max/Average pooling과 순서 |
| run_codebook_size_ablation.sh | K=256/512/1024/2048 |
| run_encoder_ablation.sh | encoder 단순화, 추가 pooling, SiLU, width |
| run_latent_width_ablation.sh | latent D와 encoder width |
| run_augmentation_ablation.sh | Crop, Flip, CutMix |
| run_cutmix_probability_ablation.sh | CutMix p=0/0.25/0.5/0.75 |
| run_head_ablation.sh | GAP, 2×2 spatial pool, concat, DWConv |
| run_pooling_head_ablation.sh | GAP, DWConv, weighted, query attention |
| run_advanced_attention_head_ablation.sh | 최종 attention head 후보 |

~~~bash
bash scripts/run_advanced_attention_head_ablation.sh
~~~

서버 접속이 끊겨도 계속 실행하려면 tmux 안에서 시작합니다.

~~~bash
tmux new -s kobeni
bash scripts/run_advanced_attention_head_ablation.sh
~~~

분리는 Ctrl-b, d이고 다시 접속할 때는 tmux attach -t kobeni를 사용합니다.

## 체크포인트와 결과

tracking이 활성화된 실행은 한국시간 기준으로 다음 경로에 저장됩니다.

~~~text
outputs/<category>/<MMDD_HHMMSS_run_group>/<run_name>/seed_<N>/
~~~

| 파일 | 내용 |
| --- | --- |
| config.json | 상속과 CLI override가 모두 적용된 최종 설정 |
| metrics.jsonl | epoch별 train/eval metric, learning rate, 시간, GPU memory |
| best.pt | 분류는 최고 validation accuracy, segmentation은 최고 mIoU |
| last.pt | 마지막 정상 epoch의 model/optimizer/scheduler/scaler 상태 |
| model_summary.json | 모델 구조와 component별 parameter 수 |
| run_metadata.json | 실행 시각, 환경, Git/source fingerprint, 실행 상태 |
| run_summary.json | best/final epoch와 일반화 차이 |
| source_snapshot/ | 실행 당시 Python source 사본 |

중단된 학습은 last.pt의 부모 폴더에 그대로 이어 씁니다.

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni train \
  --config config/segmentation/decoder_ablation/a_direct.toml \
  --resume-from outputs/segmentation/RUN_GROUP/RUN_NAME/seed_0/last.pt
~~~

완료된 실험을 CSV와 JSON으로 합치려면:

~~~bash
uv run kobeni summarize-experiments --run-dir outputs/architecture
uv run kobeni summarize-experiments --run-dir outputs/segmentation
~~~

분류는 accuracy, segmentation은 mIoU를 선택 지표로 사용합니다. 같은 설정의 여러 seed는
평균과 sample standard deviation으로 함께 집계됩니다.

## Class별 code usage

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run kobeni analyze-code-usage \
  --config config/architecture/vq_baseline.toml \
  --checkpoint outputs/architecture/RUN_GROUP/RUN_NAME/seed_0/best.pt \
  --split test \
  --output-dir outputs/code_usage/baseline/seed_0/test
~~~

summary.json, code_usage.csv, code_usage_heatmap.svg, counts.pt가 생성됩니다. 서로 다른 seed에서는
code ID permutation이 달라질 수 있으므로 code 번호를 직접 일대일 대응시키면 안 됩니다.

## Optuna 4-GPU tuning

탐색 범위는 [config/tuning/vq_optuna.toml](config/tuning/vq_optuna.toml)의 parameters 아래에서
관리합니다. 현재 learning rate, weight decay, commitment weight, lambda_vq를 탐색합니다.

~~~bash
bash scripts/run_optuna_vq_tuning.sh
~~~

스크립트가 출력한 run group으로 진행 로그를 확인합니다.

~~~bash
tail -f outputs/optuna/RUN_GROUP/logs/gpu_0.log
~~~

중간에 종료했거나 별도로 집계하려면:

~~~bash
uv run kobeni summarize-tuning \
  --study-config config/tuning/vq_optuna.toml \
  --run-group RUN_GROUP
~~~

study_summary.json, trials.csv/json과 완료 trial 기준 상위 4개 정보가 top_configs/에 저장됩니다.
실행 중인 trial은 running으로 표시되며 완성된 checkpoint로 취급하지 않습니다.

## ADE20K spatial VQ segmentation

ADE20K은 자동 다운로드하지 않습니다. 공식 데이터 이용 조건에 동의해 내려받은 뒤 다음 구조로
압축을 풉니다.

~~~text
data/ade20k/ADEChallengeData2016/
  images/training/
  images/validation/
  annotations/training/
  annotations/validation/
~~~

~~~bash
uv run kobeni prepare-data --config config/segmentation/ade20k_direct.toml
~~~

공통 흐름은 224×224 입력을 Tiny encoder와 VQ에 통과시켜 56×56×64 token map을 만드는 것입니다.
ADE label 1…150은 학습 label 0…149로 변환하고, 0과 범위 밖 값은 ignore index 255로 처리합니다.

| 설정 | 구조와 평가 |
| --- | --- |
| A direct | 1×1 Conv 64→150. 56×56에서 loss, logits를 bilinear로 224×224에 올려 mIoU 계산 |
| B linear | Conv 64→128 → Up → Conv 128→64 → Up → 1×1 Conv, BN/ReLU 없음 |
| C ReLU | B의 두 3×3 Conv 뒤에 ReLU |
| D BatchNorm | B의 두 3×3 Conv 뒤에 BN, ReLU 없음 |

~~~bash
bash scripts/run_ade20k_decoder_ablation.sh
~~~

현재 segmentation 설정은 실험 초기 단계입니다. CIFAR에서 정한 VQ loss 계수를 상속하므로,
segmentation loss와 VQ loss의 절대 크기, perplexity, dead-code fraction을 함께 확인한 뒤
ADE 전용 loss scale을 결정해야 합니다.

## CUDA 장애 확인

~~~bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 uv run python -c "import torch; x=torch.randn(8,device='cuda'); print(torch.cuda.get_device_name(0),x.sum())"
~~~

CUDA 719, sticky error, device handle Unknown Error가 발생하면 해당 Python 프로세스는 복구할 수
없습니다. nvidia-smi에서도 GPU가 사라졌다면 코드 재실행보다 서버/드라이버 복구가 먼저입니다.
CUDA_LAUNCH_BLOCKING=1은 재부팅 후에도 정상 GPU에서 같은 연산이 반복 실패할 때 stack trace를
정확히 찾는 짧은 디버그 실행에만 사용합니다.

## 저장소 구조

~~~text
config/
  architecture/          CIFAR architecture와 training ablation
  segmentation/          ADE20K direct/decoder 설정
  tuning/                Optuna study와 탐색 범위
scripts/                 multi-GPU launcher
src/kobeni/
  models/                encoder, quantizer, classifier head, segmentation decoder
  analysis/              accuracy, code usage, 전체 실험 집계
  config.py              TOML 상속과 설정 검증
  training.py            분류 학습·평가·checkpoint
  segmentation_training.py
  tuning.py              Optuna worker와 상위 trial 저장
tests/                   CPU 기반 shape, gradient, training regression test
outputs/                 checkpoint와 결과; Git에서 제외
data/                    dataset; Git에서 제외
~~~

연구 목표와 초기 설계는
[discrete_spatial_visual_vocabulary_project_plan.md](discrete_spatial_visual_vocabulary_project_plan.md),
구현 단계는 [PLAN.md](PLAN.md), 초기 depth 비교는
[reports/depth_sweep_blocks_1_4.md](reports/depth_sweep_blocks_1_4.md)를 참고하세요.
