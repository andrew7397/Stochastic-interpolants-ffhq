# Stochastic Interpolants for Face Image Generation

Implementation of a **face image generation** system based on the **Stochastic Interpolants** framework, using a **Diffusion Transformer (DiT)** as the backbone. Trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset on an HPC cluster.

> Master's thesis — M.Sc. in Applied Computer Science, University of Naples Parthenope, 2026  
> Grade: 110/110 cum laude

## Overview

Stochastic Interpolants define a family of probability paths between two distributions:

```
x_t = α(t) · x₀ + β(t) · x₁ + γ(t) · z
```

where `x₀ ~ N(0, I)` is noise, `x₁` is a real image, and `z` is an additional stochastic perturbation. The model learns the **velocity field** `v(x, t)` that transports noise to data along this path. At inference, a standard ODE solver integrates the learned velocity from `t=0` to `t=1` to generate new images.

## Architecture

- **Backbone**: DiT-S/4 (Diffusion Transformer, patch size 4)
- **Input**: 128×128 RGB images
- **Path**: Linear interpolant (`α(t) = 1-t`, `β(t) = t`)
- **Learning target**: Velocity field
- **ODE solver**: `dopri5` (adaptive step, via `torchdiffeq`)
- **Training**: AdamW, lr=1e-4, cosine scheduler, EMA (β=0.995), gradient clipping

## Project structure

```
satflow/
├── common/
│   ├── interflow/
│   │   ├── stochastic_interpolant.py   # Core interpolant, velocity/score models
│   │   ├── fabrics.py                  # Interpolant path definitions (α, β, γ)
│   │   ├── prior.py                    # Base distribution (Gaussian, GMM)
│   │   └── util.py
│   ├── config.py                       # Config dataclasses
│   ├── checkpoints.py                  # Save/load logic
│   ├── metrics.py                      # FID and other metrics
│   └── logging.py
├── latent_to_image/
│   ├── models/DiT.py                   # Diffusion Transformer implementation
│   └── training/
│       ├── main.py                     # Training entry point
│       ├── base.py                     # Training loop
│       ├── single_model.py             # Single-model trainer (velocity)
│       └── metrics.py
├── data/datasets/ffhq.py               # FFHQ dataset loader
├── json_configs/
│   └── ffhq.json                       # Main training config
├── evaluate.py                         # Generation and evaluation script
└── training.py                         # Top-level training script
```

## Configuration

Training is controlled via JSON config (`json_configs/ffhq.json`):

```json
{
  "model":       { "DiT_type": "DiT-S/4", "input_size": 128 },
  "interpolant": { "path": "linear", "type_of_learning": "velocity", "sampler_method": "dopri5" },
  "training":    { "learning_rate": 1e-4, "ema": true, "max_epochs": 500 }
}
```

## Requirements

```bash
pip install torch torchvision torchdiffeq wandb einops timm
```

## Training

```bash
python training.py --config json_configs/ffhq.json
```

Experiment tracking is done via [Weights & Biases](https://wandb.ai). Training was run on an HPC cluster with GPU nodes.

## References

- Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants* (2022)
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (2023)
- Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks* — FFHQ dataset (2019)
