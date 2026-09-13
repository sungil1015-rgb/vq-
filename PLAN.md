# Kobeni implementation plan

## 구현 원칙

핵심 비교는 `continuous`와 `discrete`이며, 비교할 때 인코더, 증강, optimizer,
epoch, latent spatial resolution과 seed를 고정한다. 처음부터 segmentation이나 detection을
섞지 않고, 각 단계의 실패 원인을 앞 단계에서 분리한다.

## 0. 기반 구축 (현재)

- `src/kobeni` 패키지와 `uv` 기반 환경
- CIFAR용 8×8 latent CNN encoder
- 학습 가능한 VQ와 frozen random VQ
- GAP + Linear 분류기와 작은 reconstruction decoder
- 네 가지 Phase 1 설정 및 reconstruction 설정
- 분류, VQ, reconstruction 손실을 조합하는 학습 루프
- warmup + cosine learning-rate schedule
- best/last checkpoint와 중단 지점 resume
- active codes, perplexity, dead-code fraction, quantization error 기록
- shape와 gradient를 검증하는 단위 테스트

완료 조건: 모든 설정이 `kobeni check`를 통과하고 테스트가 성공한다.

## 1. Classification viability

1. CIFAR-10에서 `continuous`, `continuous_bottleneck`, `vq`, `random_vq`를 각각 3개 seed로 학습한다.
2. Top-1 accuracy, parameter 수, 학습 시간과 VQ 통계를 JSONL로 보존한다.
3. VQ와 continuous 사이의 정확도 차이 및 seed 평균/표준편차를 보고한다.
4. 학습 안정성이 확인된 뒤 CIFAR-100에서 같은 비교를 반복한다.

완료 조건: VQ가 continuous 대비 잃는 정확도와 유효 vocabulary 크기를 재현 가능하게 설명한다.

## 2. Reconstruction trade-off

1. 같은 VQ 모델에 작은 decoder만 연결한다.
2. `lambda_rec = 0, 0.01, 0.1, 1.0`을 sweep한다.
3. accuracy와 함께 L1/MSE, PSNR을 기록한다.
4. decoder 용량은 고정하여 latent 품질 비교를 흐리지 않는다.

완료 조건: semantic 보존과 spatial detail 보존 사이의 Pareto 구간을 확인한다.

## 3. Codebook analysis와 intervention

1. dataset 전체의 code histogram, perplexity, dead-code 비율과 quantization error를 집계한다.
2. code별 nearest image patch와 class 조건부 사용량을 시각화한다.
3. 단일 token 삭제/교체 후 reconstruction 변화의 공간적 범위를 측정한다.
4. leave-one-token-out 또는 linear contribution으로 classification importance map을 만든다.
5. 실제 중복 collapse가 관찰될 때에만 active code 대상 separation loss를 추가한다.

완료 조건: code가 재사용되는지, token 효과가 국소적인지 정량·정성 근거를 함께 확보한다.

## 4. Codebook size

`K = 32, 64, 128, 256, 512, 1024`를 같은 조건에서 비교한다. 명목상 K가 아니라
active count와 perplexity가 어디서 포화되는지를 주 결과로 본다. Adaptive K는 이 단계가 끝난 뒤에만 검토한다.

## 5. Segmentation, 이후 detection

분류 경쟁력, codebook 안정성, reconstruction, token locality가 모두 확인된 뒤 작은
`1x1 Conv + bilinear upsampling` head로 segmentation을 시작한다. Detection은 segmentation
결과가 나온 이후 별도 단계로 둔다.

## 당장 다음 실행 순서

1. `uv run pytest`
2. 네 Phase 1 설정에 대한 `kobeni check`
3. 정식 출력 폴더에서 continuous, matched bottleneck, VQ, random VQ를 각각 실행
4. seed 3개를 사용한 baseline 반복
5. 결과 집계 스크립트와 code usage 시각화 추가
