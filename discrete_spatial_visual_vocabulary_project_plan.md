# Discrete Spatial Visual Vocabulary: Research Plan

## 1. Project goal

Build an end-to-end vision model that converts an image into a **discrete spatial latent map** using a learned codebook, and test whether that representation can simultaneously support:

- lightweight classification,
- image/feature reconstruction,
- segmentation,
- later, detection.

Core hypothesis:

> A compact discrete spatial vocabulary can preserve both semantic identity and spatial structure well enough to become a shared intermediate representation for multiple vision tasks.

---

## 2. Core architecture

```text
Image
  |
Encoder
  |
Continuous feature map
  |
Vector Quantization
  |
Discrete spatial latent map
  |----------------------|
  |                      |
GAP + Linear          Small Decoder
  |                      |
Classification      Reconstruction / Segmentation
```

Let

\[
F = E_	heta(x), \qquad F \in \mathbb{R}^{H' 	imes W' 	imes D}
\]

and codebook

\[
\mathcal{E}=\{e_1,\dots,e_K\}.
\]

Each spatial feature is quantized by

\[
k^*=rg\min_k \|F(i,j)-e_k\|_2
\]

and

\[
Z_q(i,j)=e_{k^*}.
\]

The important point is that the latent representation remains spatial rather than collapsing immediately into one global vector.

---

## 3. What the project should prove

The project should answer these questions in order:

1. **Does quantization preserve classification information?**
2. **Can the quantized map support classification with a very small head?**
3. **Does reconstruction supervision help the latent codes become more spatially meaningful?**
4. **Do individual latent locations retain correspondence with local image regions?**
5. **Do codebook entries become reusable visual primitives rather than arbitrary clusters?**
6. **Can the same representation later support segmentation without a large task-specific encoder?**

Do not begin by trying to solve classification, reconstruction, segmentation, and detection at once.

---

# Phase 1 — Classification viability

## 4. Dataset

Start with:

- CIFAR-10

Then validate with:

- CIFAR-100

Only after the concept works:

- ImageNet-1K
- ImageNet-S
- optional Pascal VOC for fast segmentation experiments

---

## 5. Encoder

Use a simple, standard encoder first.

Recommended:

- small CNN for debugging,
- CIFAR-style ResNet-18 truncated after Stage 2 or Stage 3 for the main experiment.

Example:

```text
32x32 RGB
 |
CNN encoder
 |
8x8xD feature map
 |
VQ
 |
8x8xD quantized map
 |
Global Average Pool
 |
Linear
 |
Class
```

The classification head should intentionally be weak.

Preferred:

```text
Global Average Pool + Linear
```

Do not initially use a Transformer or deep MLP head.

If classification only works with a powerful head, the latent representation itself is not doing enough work.

---

## 6. Mandatory baselines

### A. Continuous representation

```text
Encoder -> continuous feature -> GAP -> Linear
```

### B. Discrete representation

```text
Encoder -> VQ -> GAP -> Linear
```

### C. Random/frozen codebook

Useful control:

```text
Encoder -> frozen random codebook -> GAP -> Linear
```

### D. Continuous bottleneck with matched capacity

```text
Encoder -> lower-dimensional continuous latent -> GAP -> Linear
```

This baseline is important because otherwise any improvement may simply come from compression rather than discreteness.

---

## 7. First success criterion

A promising result would look roughly like:

```text
Continuous latent: 94.5%
Quantized latent:  93.8-94.5%
```

The exact numbers are not important.

The important result is:

> The codebook loses little semantic information while making the representation discrete and structured.

Measure across at least three random seeds.

---

# Phase 2 — Reconstruction

## 8. Add a small decoder

```text
Image
 |
Encoder
 |
VQ
 |---------> GAP + Linear -> Classification
 |
Small decoder
 |
Reconstruction
```

Start with a deliberately small decoder.

Avoid initially:

- diffusion,
- adversarial losses,
- large U-Net decoders,
- strong perceptual networks.

Otherwise the decoder may compensate for a weak latent representation.

---

## 9. Loss

Start with:

\[
L =
L_{cls}
+
\lambda_{vq}L_{vq}
+
\lambda_{rec}L_{rec}.
\]

Use a simple reconstruction loss first:

\[
L_{rec}=\|x-\hat{x}\|_1
\]

or MSE.

Sweep the reconstruction weight.

Example:

```text
lambda_rec = 0
             0.01
             0.1
             1.0
```

The objective is not perfect image reconstruction.

The objective is to preserve enough spatial information without sacrificing semantic compression.

---

## 10. Critical trade-off

Classification encourages:

\[
	ext{invariance}
\]

while reconstruction encourages:

\[
	ext{spatial/detail preservation}.
\]

The core scientific question is whether a useful middle point exists.

Possible outcomes:

```text
Too classification-heavy:
semantic but spatial detail disappears

Too reconstruction-heavy:
good pixels but weak semantic organization

Balanced:
semantic + spatial latent representation
```

---

# Phase 3 — Analyze the codebook

## 11. Code utilization

For each code:

\[
p_k =
rac{	ext{assignments to code }k}
{	ext{all assignments}}.
\]

Measure:

- number of active codes,
- dead-code fraction,
- entropy,
- codebook perplexity.

\[
P=
\exp\left(-\sum_kp_k\log p_kight).
\]

Nominal codebook size \(K\) is less important than effective usage.

Example:

```text
K = 512
Perplexity = 45
```

means most capacity is not actually being used.

---

## 12. Quantization quality

Measure:

\[
D_q=
rac{1}{N}\sum_i \min_k \|z_i-e_k\|^2.
\]

Desired state:

```text
low quantization error
+ good utilization
+ low redundancy
```

---

## 13. Code separation

Do not add a separation loss immediately.

First determine whether learned codes actually collapse near one another.

If necessary:

\[
L_{sep}
=
\sum_{i<j}
\max(0,m-\|e_i-e_j\|_2)^2.
\]

Important:

> The goal is only to prevent near-duplicate codes, not to force all concepts to be maximally far apart.

If used, preferably apply this to active codes rather than dead entries.

---

# Phase 4 — Test whether spatial meaning is real

This phase is essential.

Good reconstruction alone does **not** prove that individual code locations correspond to meaningful image regions.

---

## 14. Token deletion experiment

Replace one spatial code:

\[
Z_q(i,j)
\]

with zero, the mean code, or another code.

Decode again.

Compute:

\[
\Delta x =
|\hat{x}-\hat{x}_{modified}|.
\]

Desired behavior:

```text
one latent token removed
        |
        v
mostly local image region changes
```

If the entire image changes, the latent representation is strongly entangled.

---

## 15. Token replacement experiment

Replace:

\[
e_a ightarrow e_b
\]

at one or several locations.

Observe whether the effect is:

- local texture change,
- boundary change,
- color/appearance change,
- object-part change,
- global structural change.

This helps determine what the codebook actually represents.

---

## 16. Classification importance map

Because the classifier is lightweight, measure the contribution of individual latent locations.

Possible methods:

- gradient magnitude,
- leave-one-token-out,
- linear classifier contribution.

Then map the important latent locations back onto the original image.

Desired chain:

```text
class decision
-> important latent code
-> latent position
-> corresponding image region
```

This is one of the strongest possible interpretability results.

---

# Phase 5 — Codebook-size experiments

## 17. Fixed K first

Test:

\[
K=
32,\;64,\;128,\;256,\;512,\;1024.
\]

For each setting report:

- classification accuracy,
- reconstruction quality,
- quantization error,
- active code count,
- perplexity,
- dead-code percentage.

Key question:

> Does performance depend on nominal \(K\), or does effective vocabulary size naturally saturate?

This may become an interesting result by itself.

---

## 18. Adaptive K only later

Do **not** begin with learnable codebook size.

After the fixed-\(K\) behavior is understood, possible extensions include:

```text
large K_max
+ usage-based pruning
+ code splitting
+ code merging
+ minimum-distance constraint
```

The effective vocabulary size could then emerge from the data.

But introducing this too early will make failures difficult to diagnose.

---

# Phase 6 — Segmentation

## 19. Only move to segmentation if the previous phases work

Requirements before proceeding:

- quantized classification is competitive,
- codebook utilization is stable,
- reconstruction works reasonably,
- latent-token interventions show spatial locality.

Then test whether the representation transfers to dense prediction.

---

## 20. Dataset

Recommended:

- ImageNet-S if ImageNet-aligned classification and segmentation are desired,
- optionally Pascal VOC for faster experiments.

---

## 21. Minimal segmentation setup

```text
Image
 |
Encoder
 |
VQ latent map
 |
small segmentation head
 |
mask
```

Keep the segmentation head weak initially.

Example:

```text
1x1 Conv
-> bilinear upsampling
-> segmentation logits
```

Compare against:

```text
same encoder
same decoder
continuous latent
```

The only intended variable should be quantization.

Measure:

- mIoU,
- pixel accuracy,
- optional boundary IoU.

---

# Detection comes last

Do not add detection until the representation works for segmentation.

Detection introduces:

- box regression,
- object assignment,
- multi-scale representation,
- scale variation,
- more complex losses.

Adding it too early obscures the central research question.

---

# 22. Main failure modes

## Failure A — Quantization destroys classification

Possible causes:

- K too small,
- latent dimension too small,
- quantization too early,
- unstable codebook optimization.

First response:

- increase K,
- increase latent dimension,
- quantize at a later encoder stage.

Do not redesign the entire architecture immediately.

---

## Failure B — Good reconstruction, weak classification

Interpretation:

The codebook stores pixel-level information but not enough semantics.

Response:

- increase classification weight,
- reduce reconstruction weight,
- inspect code usage and class separability.

---

## Failure C — Good classification, poor reconstruction/segmentation

Interpretation:

The bottleneck has become too invariant and lost spatial detail.

Response:

- retain a higher-resolution latent map,
- move quantization earlier/later,
- moderately increase reconstruction supervision.

---

## Failure D — Codebook collapse

Example:

```text
K = 512
active codes = 20
```

Possible solutions:

- EMA code updates,
- code reset,
- better initialization,
- weak usage regularization,
- separation loss if actual duplication exists.

---

## Failure E — Decoder hides a bad latent representation

A strong decoder may reconstruct plausible images using learned priors even when latent information is poor.

Controls:

- keep decoder small,
- perform token deletion/replacement,
- compare reconstruction sensitivity,
- measure code information directly.

---

## Failure F — Classification and segmentation objectives conflict

Classification may discard exact position while segmentation needs it.

This conflict is not merely an implementation bug; it may be a fundamental result.

If strong conflict persists, later extensions could include:

- hierarchical codebooks,
- shared + task-specific codes,
- separate coarse/fine vocabularies.

Do not assume one codebook must succeed.

---

# 23. Essential experimental controls

Keep these fixed wherever possible:

- encoder architecture,
- decoder architecture,
- optimizer,
- learning-rate schedule,
- training epochs,
- augmentation,
- latent spatial resolution,
- random seeds.

The most important comparison is:

```text
continuous latent
vs.
discrete latent
```

with everything else matched.

---

# 24. Metrics

### Classification
- Top-1 accuracy

### Reconstruction
- L1/MSE
- PSNR
- optional SSIM

### Codebook
- utilization
- perplexity
- dead-code fraction
- quantization error
- pairwise code distance distribution

### Representation
- linear probe
- k-NN probe
- class separability
- CKA
- intrinsic dimension

### Segmentation
- mIoU
- pixel accuracy
- optional boundary IoU

### Efficiency
- parameter count
- MACs/FLOPs
- GPU latency
- memory use

---

# 25. Minimum viable experiment

Start only with:

```text
CIFAR-10

Image
 |
small CNN / truncated ResNet
 |
8x8 spatial feature map
 |
VQ
 |--------------|
 |              |
GAP + Linear    small decoder
 |              |
class           reconstruction
```

Train four variants:

### Model A
Continuous latent + classifier

### Model B
VQ latent + classifier

### Model C
VQ latent + classifier + reconstruction

### Model D
VQ latent + classifier + reconstruction + separation loss, only if needed

Do not add segmentation before these results are understood.

---

# 26. First experimental questions

Answer in this order:

1. How much accuracy does VQ lose versus continuous features?
2. Does reconstruction supervision improve or hurt classification?
3. How many codes are actually used?
4. Does increasing K continue to help?
5. Are individual latent locations spatially local?
6. Can a simple linear classifier use the quantized representation effectively?
7. Do the learned codes correlate with recurring visual patterns?

If the answers are encouraging, proceed to segmentation.

---

# 27. Strong outcomes

### Outcome 1 — Minimal accuracy loss

```text
Continuous: 94.5%
VQ:         94.2%
```

This shows that discrete representation can preserve semantics.

### Outcome 2 — Reconstruction improves semantics

```text
VQ + classification:                  92.8%
VQ + classification + reconstruction: 94.0%
```

This would be especially interesting because it suggests spatial reconstruction regularizes the visual vocabulary.

### Outcome 3 — Lightweight classifier works

If

```text
VQ latent -> GAP -> Linear
```

matches a much deeper classifier, the encoder/codebook has already performed most of the semantic abstraction.

### Outcome 4 — Codes align with regions

If code maps correlate with segmentation regions or token intervention effects remain local, the representation gains strong spatial meaning.

### Outcome 5 — Effective vocabulary saturates

If \(K\) becomes large while effective usage stabilizes at a much smaller value, this may reveal a natural task-dependent visual vocabulary size.

---

# 28. Negative results are still valuable

Useful findings include:

- discrete representation works for classification but not segmentation,
- reconstruction and classification strongly conflict,
- large codebooks provide no benefit,
- one shared codebook is insufficient,
- continuous bottlenecks consistently outperform VQ,
- code IDs are reusable but not human-interpretable.

Do not force the original hypothesis if the evidence disagrees.

---

# 29. What not to add early

Avoid initially:

- detection,
- dynamic K,
- Transformer after VQ,
- DINO teacher,
- hierarchical VQ,
- multi-scale decoder,
- diffusion,
- GAN loss,
- complicated routing,
- complex codebook regularizers.

Each additional mechanism makes causal interpretation harder.

---

# 30. Literature areas to review before claiming novelty

Search terms:

```text
VQ-VAE
vector quantized representation learning
discrete visual representation
semantic vector quantization
VQ classification
VQ segmentation
discrete semantic bottleneck
visual tokenizer
vector quantization dense prediction
semantic codebook vision
multi-task vector quantization
BEiT tokenizer
dVAE visual tokenizer
VQGAN
information bottleneck vision
```

The most important novelty check is:

> Has a shared discrete spatial bottleneck already been systematically evaluated for both lightweight classification and dense prediction under matched continuous baselines?

---

# 31. Recommended paper framing

Do not frame the work primarily as:

> “We created another end-to-end multi-task model.”

A stronger framing is:

> “We investigate whether a compact discrete spatial vocabulary can serve as a common intermediate representation that preserves both semantic and spatial information.”

The paper should emphasize:

1. representation properties,
2. controlled continuous-vs-discrete comparisons,
3. codebook geometry/utilization,
4. spatial intervention experiments,
5. classification-to-segmentation transfer.

---

# 32. Final project objective

The central proposition to test is:

\[
oxed{
	ext{A discrete spatial visual vocabulary can compress semantics without destroying spatial correspondence.}
}
\]

If this is true, classification can operate cheaply on code-level representations while dense tasks can decode the same representation back into spatial predictions.

That is the main research target.
