# EoMT vs ViT-Adapter+Mask2Former on MSD Lung

**Research question:** Does EoMT's encoder-only design generalize to a medical imaging domain, or does the heavier two-component pipeline of ViT-Adapter+Mask2Former hold an advantage when the input distribution shifts from natural images to CT scans?

Both models are implemented paper-faithfully and trained under identical conditions. The only divergences from the original paper are forced by the dataset (binary classes, CT input, limited training size). See [METHODOLOGY.md](METHODOLOGY.md) for full details.

| Model | Backbone | Params (backbone) | Paper variant |
|---|---|---|---|
| **EoMT** | DINOv2 ViT-B/14 reg4 | ~86 M | ViT-B (this repo) / ViT-L (paper) |
| **ViT-Adapter + Mask2Former** | DINOv2 ViT-B/14 reg4 | ~86 M | ViT-B (this repo) / ViT-L (paper) |
| **Mask2Former** | Swin-Base (ADE20k pretrained) | ~88 M | Reference baseline (non-DINOv2) |

> The EoMT paper (CVPR 2025) reports main results with ViT-Large (~307 M). To reproduce at paper scale, set `backbone_name` to `vit_large_patch14_reg4_dinov2.lvd142m` in the relevant config. See [Model Size Variants](#model-size-variants).

---

## Repository layout

```
medical-seg/
├── prepare_data.py             # One-time: NIfTI → 2D numpy slices
├── train.py                    # Unified training entry point
├── evaluate.py                 # Side-by-side DICE / IoU comparison
│
├── data/
│   └── msd_lung.py             # PyTorch Dataset + collate_fn
│
├── models/                     # EoMT architecture (from eomt-master)
│   ├── vit.py                  # ViT wrapper (timm / HuggingFace)
│   ├── eomt.py                 # EoMT encoder with masked attention
│   └── scale_block.py          # Transposed-conv upscale block
│
├── training/
│   ├── mask_classification_loss.py          # Hungarian-matched BCE+Dice+CE loss
│   └── two_stage_warmup_poly_schedule.py   # Paper LR schedule (from eomt-master)
│
├── architectures/
│   └── vit_adapter_mask2former.py    # ViT-Adapter + Mask2Former (standalone)
│
├── runners/
│   ├── eomt_runner.py                # Lightning module: EoMT
│   ├── vit_adapter_m2f_runner.py     # Lightning module: ViT-Adapter+M2F
│   ├── mask2former_runner.py         # Lightning module: Mask2Former (HF)
│   ├── optim_utils.py                # LLRD parameter group builder
│   └── dice_metric.py                # DICE and IoU helpers
│
└── configs/
    ├── eomt_lung.yaml
    ├── vit_adapter_m2f_lung.yaml
    └── mask2former_lung.yaml
```

---

## Installation

```bash
pip install torch>=2.1.0 torchvision>=0.16.0 lightning>=2.1.0 \
            timm>=0.9.12 transformers>=4.38.0 nibabel>=5.1.0 \
            numpy>=1.26.0 tqdm>=4.66.0
```

---

## Data Preparation

Download the MSD Lung dataset (Task06_Lung) from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). Extract the tar archive, then run:

```bash
python prepare_data.py \
    --data_dir /path/to/Task06_Lung \
    --output_dir ./data/lung \
    --val_fraction 0.2
```

This extracts axial 2D slices, applies lung CT windowing (W=1500, L=−600), stacks adjacent slices as multi-slice pseudo-RGB `[z-1, z, z+1]`, remaps labels, and saves numpy arrays. Only slices containing lung/body content are kept.

**Output layout:**
```
data/lung/
    metadata.json
    train/images/<case>_z<z>.npy   [3, H, W]  uint8   (multi-slice)
    train/labels/<case>_z<z>.npy   [H, W]     uint8
    val/  ...
```

**Label convention:**

| Value | Class |
|---|---|
| `0` | Cancer |
| `255` | Background / ignore |

---

## Training

Both primary models share the same hyperparameter interface and training schedule. Config files are in `configs/`.

### EoMT (DINOv2 ViT-Base)
```bash
python train.py \
    --model eomt \
    --data_dir ./data/lung \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints
```

### ViT-Adapter + Mask2Former (DINOv2 ViT-Base)
```bash
python train.py \
    --model vit_adapter_m2f \
    --data_dir ./data/lung \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints
```

### Mask2Former (Swin-Base, reference baseline)
```bash
python train.py \
    --model mask2former \
    --data_dir ./data/lung \
    --m2f_model facebook/mask2former-swin-base-ade-semantic \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints
```

All models save the top-3 checkpoints by `val/dice_mean` plus `last.ckpt`. Training stops early if `val/dice_mean` does not improve for 10 consecutive epochs.

### Multi-GPU Training

Pass `--devices N` to use N GPUs with DDP. The per-GPU batch size stays fixed; effective batch size scales linearly.

```bash
# 2-GPU example (e.g. CUDA_VISIBLE_DEVICES=0,2)
CUDA_VISIBLE_DEVICES=0,2 python train.py \
    --model eomt \
    --data_dir ./data/lung \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --devices 2 --max_epochs 50 \
    --output_dir ./checkpoints
```

| `--devices` | Per-GPU batch | Effective batch | ~Epoch time (relative) |
|---|---|---|---|
| 1 | 2 | 2 | 1× (baseline) |
| 2 | 2 | 4 | ~0.55× |
| 4 | 2 | 8 | ~0.30× |

The LR schedule's `max_steps` is computed as per-GPU step count (`len(train_dl) × epochs // devices`), keeping warmup and polynomial decay proportionally correct at any device count.

---

## Evaluation

```bash
python evaluate.py \
    --data_dir ./data/lung \
    --eomt_ckpt        ./checkpoints/eomt/ckpts/last.ckpt \
    --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \
    --m2f_ckpt         ./checkpoints/mask2former/ckpts/last.ckpt \
    --img_size 512
```

Prints a side-by-side table and writes `eval_results.json`:

```
                          EoMT  ViT-Adapter+M2F  Mask2Former
──────────────────────────────────────────────────────────────
dice_cancer             0.XXXX           0.XXXX       0.XXXX
iou_cancer              0.XXXX           0.XXXX       0.XXXX
dice_mean               0.XXXX           0.XXXX       0.XXXX
miou                    0.XXXX           0.XXXX       0.XXXX
──────────────────────────────────────────────────────────────
```

---

## Model Size Variants

The configs default to **Base** variants for compute feasibility. To reproduce paper-scale results, swap the backbone:

| Variant | `backbone_name` | Backbone params |
|---|---|---|
| ViT-Base (default) | `vit_base_patch14_reg4_dinov2.lvd142m` | ~86 M |
| ViT-Large (paper) | `vit_large_patch14_reg4_dinov2.lvd142m` | ~307 M |

For Mask2Former, the Swin backbone size can be changed via `--m2f_model`:

| Variant | HuggingFace ID | Backbone params |
|---|---|---|
| Swin-Tiny | `facebook/mask2former-swin-tiny-ade-semantic` | ~28 M |
| Swin-Base (default) | `facebook/mask2former-swin-base-ade-semantic` | ~88 M |
| Swin-Large | `facebook/mask2former-swin-large-ade-semantic` | ~197 M |

---

## Methodology and Evaluation

See [METHODOLOGY.md](METHODOLOGY.md) for detailed descriptions of:
- Research goal and experimental design
- Dataset preprocessing and CT windowing
- Multi-slice pseudo-RGB construction
- Model architectures (paper-faithful)
- Training hyperparameters and schedule
- Global DICE and IoU evaluation
- Differences from the paper setup
