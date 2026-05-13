# Beyond Uniform Eikonal: Confidence-Weighted Regularization for Neural Implicit Surfaces

This repository implements the proposed confidence-weighted regularization framework on top of [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl). The method augments NeuS with a lightweight confidence MLP that predicts per-point geometric confidence from hash-grid features, supervised by online multi-view photometric variance. Eikonal supervision points are importance-sampled toward high-confidence regions, reducing regularization in uncertain areas through sampling frequency rather than explicit loss weighting.

## Environment

- Python 3.11
- Google Colab L4 GPU
- CUDA 12.4

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Dataset

Download the NeRF-Synthetic dataset and place it under your data directory. The file structure should be:

```
nerf_synthetic/
    lego/
    hotdog/
    chair/
    ...
```

## Method Components

All proposed components are CLI-toggleable via OmegaConf dot-notation under the `model.conf` block:

| Flag | Default | Description |
|---|---|---|
| `model.conf.enabled` | `true` | Master switch for all confidence components |
| `model.conf.use_eikonal_importance` | `true` | §3.4 importance-sampled eikonal |
| `model.conf.use_ray_sampling` | `false` | §3.5 confidence-weighted ray sampling |
| `model.conf.start_step` | `15000` | Step at which EMA and confidence head activate |
| `model.conf.ramp_steps` | `5000` | Linear ramp from uniform to confidence sampling |
| `model.conf.ema_alpha` | `0.99` | EMA decay for per-pixel variance buffer |
| `model.conf.eikonal_sample_ratio` | `0.5` | Fraction of eikonal candidate points to keep |
| `model.conf.sample_temperature` | `3.0` | Softens sampling distribution to avoid collapse |

## Running Experiments

### Vanilla NeuS Baseline

```bash
python instant-nsr-pl/launch.py \
  --config instant-nsr-pl/configs/neus-blender.yaml \
  --gpu 0 --train \
  dataset.scene=lego \
  dataset.root_dir=/content/archive/nerf_synthetic/lego \
  tag=neus_lego_baseline \
  model.geometry.mlp_network_config.otype=VanillaMLP \
  model.texture.mlp_network_config.otype=VanillaMLP \
  model.conf.enabled=false
```

### Full Method (Confidence + Eikonal Importance Sampling)

```bash
python instant-nsr-pl/launch.py \
  --config instant-nsr-pl/configs/neus-blender.yaml \
  --gpu 0 --train \
  dataset.scene=lego \
  dataset.root_dir=/content/archive/nerf_synthetic/lego \
  tag=neus_lego \
  model.geometry.mlp_network_config.otype=VanillaMLP \
  model.texture.mlp_network_config.otype=VanillaMLP \
  model.conf.enabled=true \
  model.conf.use_eikonal_importance=true \
  model.conf.use_ray_sampling=false \
  model.conf.start_step=15000 \
  model.grid_prune_occ_thre=0.01 \
  system.loss.lambda_sparsity=0.01 \
  system.loss.sparsity_scale=10.
```

### Full Method + Confidence-Weighted Ray Sampling

```bash
python instant-nsr-pl/launch.py \
  --config instant-nsr-pl/configs/neus-blender.yaml \
  --gpu 0 --train \
  dataset.scene=lego \
  dataset.root_dir=/content/archive/nerf_synthetic/lego \
  tag=neus_lego_raysamp \
  model.geometry.mlp_network_config.otype=VanillaMLP \
  model.texture.mlp_network_config.otype=VanillaMLP \
  model.conf.enabled=true \
  model.conf.use_eikonal_importance=true \
  model.conf.use_ray_sampling=true \
  model.conf.start_step=15000 \
  model.grid_prune_occ_thre=0.01 \
  system.loss.lambda_sparsity=0.01 \
  system.loss.sparsity_scale=10.
```

## Ablation Commands

### Ablation 1 — Confidence without ray sampling

```bash
python instant-nsr-pl/launch.py \
  --config instant-nsr-pl/configs/neus-blender.yaml \
  --gpu 0 --train \
  dataset.scene=lego \
  dataset.root_dir=/content/archive/nerf_synthetic/lego \
  tag=neus_lego_ablation1 \
  model.geometry.mlp_network_config.otype=VanillaMLP \
  model.texture.mlp_network_config.otype=VanillaMLP \
  model.conf.enabled=true \
  model.conf.use_eikonal_importance=true \
  model.conf.use_ray_sampling=false \
  model.conf.start_step=15000 \
  model.grid_prune_occ_thre=0.01 \
  system.loss.lambda_sparsity=0.01 \
  system.loss.sparsity_scale=10.
```

### Ablation 2 — Confidence + curvature loss

```bash
python instant-nsr-pl/launch.py \
  --config instant-nsr-pl/configs/neus-blender.yaml \
  --gpu 0 --train \
  dataset.scene=lego \
  dataset.root_dir=/content/archive/nerf_synthetic/lego \
  tag=neus_lego_ablation2 \
  model.geometry.mlp_network_config.otype=VanillaMLP \
  model.texture.mlp_network_config.otype=VanillaMLP \
  model.conf.enabled=true \
  model.conf.use_eikonal_importance=true \
  model.conf.use_ray_sampling=true \
  model.conf.start_step=15000 \
  model.grid_prune_occ_thre=0.01 \
  system.loss.lambda_sparsity=0.01 \
  system.loss.sparsity_scale=10. \
  model.geometry.grad_type=finite_difference \
  system.loss.lambda_curvature=0.001
```

### Ablation 3 — Confidence + higher eikonal weight

```bash
python instant-nsr-pl/launch.py \
  --config instant-nsr-pl/configs/neus-blender.yaml \
  --gpu 0 --train \
  dataset.scene=lego \
  dataset.root_dir=/content/archive/nerf_synthetic/lego \
  tag=neus_lego_ablation3 \
  model.geometry.mlp_network_config.otype=VanillaMLP \
  model.texture.mlp_network_config.otype=VanillaMLP \
  model.conf.enabled=true \
  model.conf.use_eikonal_importance=true \
  model.conf.use_ray_sampling=true \
  model.conf.start_step=15000 \
  model.grid_prune_occ_thre=0.01 \
  system.loss.lambda_sparsity=0.01 \
  system.loss.sparsity_scale=10. \
  system.loss.lambda_eikonal=0.2
```

## Testing a Trained Checkpoint

```bash
python instant-nsr-pl/launch.py \
  --config path/to/exp/config/parsed.yaml \
  --resume path/to/exp/ckpt/epoch=0-step=20000.ckpt \
  --gpu 0 --test
```

## Metrics

PSNR, SSIM, and LPIPS are computed at validation and test time and logged to TensorBoard under `val/` and `test/`. Checkpoints and experiment outputs are saved to `exp/[name]/[tag]@[timestamp]`. TensorBoard logs are at `runs/[name]/[tag]@[timestamp]`.

## Key Implementation Files

| File | Description |
|---|---|
| `models/neus.py` | `ConfidenceMLP`, `get_confidence()`, phase gating, `eikonal_points` |
| `systems/neus.py` | EMA buffer, importance-sampled eikonal, `Lconf`, SSIM/LPIPS metrics |
| `systems/criterions.py` | `LPIPS` wrapper |
| `configs/neus-blender.yaml` | `model.conf` block, `lambda_conf`, `confidence_mlp` optimizer entry |

## Acknowledgements

Built on top of [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl) by Yuan-Chen Guo.
