# EoMT vs ViT-Adapter+Mask2Former on Medical CT Segmentation

**Research question:** Does EoMT's encoder-only design generalize to medical imaging domains, or does the heavier two-component pipeline of ViT-Adapter+Mask2Former hold an advantage when the input distribution shifts from natural images to CT scans?

The question is tested across **two MSD datasets** with different characteristics — a single-class binary task (lung cancer) and a two-class multi-structure task (hepatic vessels + tumours). Both models are implemented paper-faithfully and trained under identical conditions. The only divergences from the original paper are forced by the datasets. See [METHODOLOGY.md](METHODOLOGY.md) for full details.

| Model | Backbone | Params (backbone) | Paper variant |
|---|---|---|---|
| **EoMT** | DINOv2 ViT-B/14 reg4 | ~86 M | ViT-B (this repo) / ViT-L (paper) |
| **ViT-Adapter + Mask2Former** | DINOv2 ViT-B/14 reg4 | ~86 M | ViT-B (this repo) / ViT-L (paper) |
| **Mask2Former** | Swin-Base (ADE20k pretrained) | ~88 M | Reference baseline (non-DINOv2) |

> The EoMT paper (CVPR 2025) reports main results with ViT-Large (~307 M). To reproduce at paper scale, set `backbone_name` to `vit_large_patch14_reg4_dinov2.lvd142m` in the relevant config. See [Model Size Variants](#model-size-variants).

---

## Datasets

| Dataset | MSD task | Classes | CT window | Volumes |
|---|---|---|---|---|
| **Lung** | Task06_Lung | 1 (cancer) | Lung (L=−600, W=1500 HU) | 63 labelled |
| **Hepatic Vessel** | Task08_HepaticVessel | 2 (vessel, tumour) | Soft-tissue (L=40, W=400 HU) | 303 labelled |

---

## Repository layout

```
medical-seg/
├── prepare_data.py             # One-time: MSD Lung NIfTI → 2D slices
├── prepare_hepatic.py          # One-time: MSD HepaticVessel NIfTI → 2D slices
├── train.py                    # Unified training entry point (--dataset lung|hepatic)
├── evaluate.py                 # Side-by-side DICE / IoU comparison
├── visualize.py                # Segmentation overlay panels
│
├── data/
│   ├── msd_lung.py             # Lung dataset (1 class)
│   └── msd_hepatic.py          # Hepatic dataset (2 classes)
│
├── models/                     # EoMT architecture (from eomt-master)
│   ├── vit.py
│   ├── eomt.py
│   └── scale_block.py
│
├── training/
│   ├── mask_classification_loss.py
│   └── two_stage_warmup_poly_schedule.py
│
├── architectures/
│   └── vit_adapter_mask2former.py
│
├── runners/
│   ├── eomt_runner.py               # Supports any num_classes
│   ├── vit_adapter_m2f_runner.py    # Supports any num_classes
│   ├── mask2former_runner.py
│   ├── optim_utils.py
│   └── dice_metric.py
│
└── configs/
    ├── eomt_lung.yaml
    ├── eomt_hepatic.yaml
    ├── vit_adapter_m2f_lung.yaml
    ├── vit_adapter_m2f_hepatic.yaml
    └── mask2former_lung.yaml
```

---

## Installation

```bash
pip install torch>=2.1.0 torchvision>=0.16.0 lightning>=2.1.0 \
            timm>=0.9.12 transformers>=4.38.0 nibabel>=5.1.0 \
            numpy>=1.26.0 tqdm>=4.66.0 matplotlib>=3.8.0
```

---

## Data Preparation

Download datasets from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).

### MSD Lung (Task06)

```bash
python prepare_data.py \
    --data_dir /path/to/Task06_Lung \
    --output_dir ./data/lung \
    --val_fraction 0.2
```

| Value | Class |
|---|---|
| `0` | Cancer |
| `255` | Background / ignore |

### MSD HepaticVessel (Task08)

```bash
python prepare_hepatic.py \
    --data_dir /path/to/Task08_HepaticVessel \
    --output_dir ./data/hepatic \
    --val_fraction 0.2
```

| Value | Class |
|---|---|
| `0` | Vessel |
| `1` | Tumour |
| `255` | Background / ignore |

Both scripts extract axial 2D slices, apply dataset-appropriate CT windowing, stack adjacent slices as multi-slice pseudo-RGB `[z−1, z, z+1]`, and save numpy arrays. Output layout is identical:

```
data/<name>/
    metadata.json
    train/images/<case>_z<z>.npy   [3, H, W]  uint8
    train/labels/<case>_z<z>.npy   [H, W]     uint8
    val/  ...
```

---

## Training

Both primary models share the same interface. Select the dataset with `--dataset`.

### MSD Lung (Task06) — 1 class

```bash
# EoMT
python train.py --model eomt --dataset lung \
    --data_dir ./data/lung \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints

# ViT-Adapter + Mask2Former
python train.py --model vit_adapter_m2f --dataset lung \
    --data_dir ./data/lung \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints

# Mask2Former (Swin-Base, reference baseline)
python train.py --model mask2former --dataset lung \
    --data_dir ./data/lung \
    --m2f_model facebook/mask2former-swin-base-ade-semantic \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints
```

### MSD HepaticVessel (Task08) — 2 classes

```bash
# EoMT
python train.py --model eomt --dataset hepatic \
    --data_dir ./data/hepatic \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints

# ViT-Adapter + Mask2Former
python train.py --model vit_adapter_m2f --dataset hepatic \
    --data_dir ./data/hepatic \
    --backbone_name vit_base_patch14_reg4_dinov2.lvd142m \
    --img_size 512 --batch_size 2 --max_epochs 50 \
    --output_dir ./checkpoints
```

`num_classes` is read automatically from `metadata.json` — no manual flag needed.

All models save the top-3 checkpoints by `val/dice_mean` plus `last.ckpt`. Training stops early if `val/dice_mean` does not improve for 10 consecutive epochs.

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,2 python train.py \
    --model eomt --dataset lung \
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

`max_steps` is computed as the per-GPU step count, keeping warmup and polynomial decay proportionally correct at any device count.

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

Prints a side-by-side table and writes `eval_results.json`. For multi-class datasets, per-class DICE and IoU are reported alongside the mean.

---

## Visualisation

Generate segmentation overlay panels. Each row shows one CT slice with:
**green** = true positive · **red** = false positive · **yellow** = false negative

```bash
# Lung — foreground-only slices
python visualize.py --dataset lung \
    --data_dir ./data/lung \
    --eomt_ckpt        ./checkpoints/eomt/ckpts/last.ckpt \
    --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \
    --n_samples 12 --cancer_only --vis_dir ./vis_output/lung

# Hepatic — foreground-only slices
python visualize.py --dataset hepatic \
    --data_dir ./data/hepatic \
    --eomt_ckpt        ./checkpoints/eomt/ckpts/last.ckpt \
    --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \
    --n_samples 12 --cancer_only --vis_dir ./vis_output/hepatic
```

Output PNGs are written to `--vis_dir`, with `--samples_per_figure` rows per file (default 6). Each prediction panel shows per-slice DICE below it.

---

## Model Size Variants

| Variant | `backbone_name` | Backbone params |
|---|---|---|
| ViT-Base (default) | `vit_base_patch14_reg4_dinov2.lvd142m` | ~86 M |
| ViT-Large (paper) | `vit_large_patch14_reg4_dinov2.lvd142m` | ~307 M |

For Mask2Former (Swin backbone):

| Variant | HuggingFace ID | Backbone params |
|---|---|---|
| Swin-Tiny | `facebook/mask2former-swin-tiny-ade-semantic` | ~28 M |
| Swin-Base (default) | `facebook/mask2former-swin-base-ade-semantic` | ~88 M |
| Swin-Large | `facebook/mask2former-swin-large-ade-semantic` | ~197 M |

---

## Methodology and Evaluation

See [METHODOLOGY.md](METHODOLOGY.md) for detailed descriptions of:
- Research goal and experimental design across both datasets
- CT windowing and preprocessing per dataset
- Multi-slice pseudo-RGB construction
- Model architectures (paper-faithful)
- Training hyperparameters and schedule
- Global DICE and IoU evaluation (per-class for multi-class tasks)
- Differences from the paper setup
