# VQ model-scale presets

All presets inherit the complete CIFAR-100 training recipe from
`../vq_baseline.toml`. Only encoder scale, output directory, and run name are overridden.

| Preset | Stem | Intermediate | Main width | Residual blocks | Latent D | Codebook K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 64 | 96 | 384 | 2 | 64 | 128 |
| Small | 64 | 128 | 512 | 3 | 64 | 128 |
| Base | 96 | 192 | 768 | 4 | 96 | 256 |
| Large | 128 | 256 | 1024 | 6 | 128 | 256 |
| XLarge | 160 | 320 | 1280 | 8 | 128 | 512 |

The VQ grid is not a preset field: the quantizer preserves the encoder's
runtime spatial shape. With the current two 2×2, stride-2 MaxPools, an input
of H×W produces floor(H/4)×floor(W/4) VQ tokens (32×32 → 8×8).

