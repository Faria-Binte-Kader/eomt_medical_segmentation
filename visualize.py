"""
Visualise segmentation predictions for EoMT and ViT-Adapter+Mask2Former on MSD Lung.

For each sample the figure shows one row:
    CT slice | Ground truth | EoMT | ViT-Adapter+M2F | Mask2Former (if provided)

Cancer overlays use a traffic-light scheme:
    Green  = true positive  (predicted AND in GT)
    Red    = false positive (predicted, not in GT)
    Yellow = false negative (in GT, not predicted)
    Grey   = CT background (correct negatives, unlabelled)

Usage:
    python visualize.py \\
        --data_dir ./data/lung \\
        --eomt_ckpt       ./checkpoints/eomt/ckpts/last.ckpt \\
        --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \\
        --n_samples 16 \\
        --vis_dir   ./vis_output

    # Cancer-only samples (skip slices with no tumour)
    python visualize.py ... --cancer_only

    # Include Mask2Former baseline
    python visualize.py ... --m2f_ckpt ./checkpoints/mask2former/ckpts/last.ckpt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, Subset

import lightning as L
from runners.dice_metric import IGNORE_IDX


# ── model loaders ────────────────────────────────────────────────────────────

def _load_eomt(ckpt, device):
    from runners.eomt_runner import EoMTMedicalModule
    m = EoMTMedicalModule.load_from_checkpoint(ckpt, map_location=device)
    m.network.masked_attn_enabled = False
    return m.eval().to(device)


def _load_vit_adapter(ckpt, device):
    from runners.vit_adapter_m2f_runner import ViTAdapterM2FModule
    m = ViTAdapterM2FModule.load_from_checkpoint(ckpt, map_location=device)
    return m.eval().to(device)


def _load_m2f(ckpt, device):
    from runners.mask2former_runner import Mask2FormerMedicalModule
    m = Mask2FormerMedicalModule.load_from_checkpoint(ckpt, map_location=device)
    return m.eval().to(device)


# ── inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(module, imgs, img_size, model_name):
    """Returns [B, H, W] bool cancer prediction."""
    imgs = imgs.to(next(module.parameters()).device)
    if model_name == "eomt":
        ml_list, cl_list = module(imgs)
        ml_up = F.interpolate(ml_list[-1], img_size, mode="bilinear", align_corners=False)
        logits = module._to_per_pixel_logits(ml_up, cl_list[-1])
    elif model_name == "vit_adapter_m2f":
        ml_list, cl_list = module(imgs)
        logits = module._to_per_pixel_logits(ml_list[-1], cl_list[-1], img_size)
    else:
        outputs = module(imgs)
        logits = module._semseg_from_outputs(outputs, img_size)
    return (logits[:, 0] > 0.5).cpu()


# ── overlay renderer ──────────────────────────────────────────────────────────

def _make_overlay(ct_slice: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    ct_slice: [H, W] uint8 grayscale
    gt:       [H, W] bool  (True = cancer in GT)
    pred:     [H, W] bool  (True = cancer predicted)

    Returns [H, W, 3] uint8 RGB overlay.
    """
    rgb = np.stack([ct_slice] * 3, axis=-1).astype(np.float32)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    alpha = 0.55
    # TP → green
    rgb[tp] = rgb[tp] * (1 - alpha) + np.array([0, 210, 80],  dtype=np.float32) * alpha
    # FP → red
    rgb[fp] = rgb[fp] * (1 - alpha) + np.array([220, 40, 40], dtype=np.float32) * alpha
    # FN → yellow
    rgb[fn] = rgb[fn] * (1 - alpha) + np.array([255, 200, 0], dtype=np.float32) * alpha

    return rgb.clip(0, 255).astype(np.uint8)


# ── figure builder ────────────────────────────────────────────────────────────

def render_samples(
    samples,            # list of (img_tensor [3,H,W], seg_tensor [H,W], predictions dict)
    model_names,        # ordered list of model names to show
    out_path: Path,
    cols_per_row: int = 4,
):
    """
    Each sample gets one figure with columns: CT | GT | model_1 | model_2 ...
    All samples are tiled into a single grid PNG.
    """
    n_models = len(model_names)
    n_cols   = 2 + n_models          # CT + GT + one per model
    n_rows   = len(samples)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a1a")

    col_titles = ["CT (z)", "Ground Truth"] + model_names

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="white", fontsize=11, pad=6)

    for row_idx, (img, seg, preds) in enumerate(samples):
        ct = img[1].numpy()   # middle channel = current slice z
        gt = (seg == 0).numpy()

        # ── CT ─────────────────────────────────────────────────────────
        ax = axes[row_idx, 0]
        ax.imshow(ct, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

        # ── Ground truth ────────────────────────────────────────────────
        ax = axes[row_idx, 1]
        gt_overlay = _make_overlay(ct, gt, gt)   # TP = whole GT region → all green
        ax.imshow(gt_overlay)
        ax.axis("off")

        # ── Model predictions ────────────────────────────────────────────
        for col_offset, name in enumerate(model_names, start=2):
            pred = preds[name].numpy()
            dice = _slice_dice(pred, gt)
            overlay = _make_overlay(ct, gt, pred)
            ax = axes[row_idx, col_offset]
            ax.imshow(overlay)
            ax.set_xlabel(f"DICE = {dice:.3f}", color="#cccccc", fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    # legend
    legend_patches = [
        mpatches.Patch(color=(0/255, 210/255, 80/255),  label="True positive"),
        mpatches.Patch(color=(220/255, 40/255, 40/255), label="False positive"),
        mpatches.Patch(color=(255/255, 200/255, 0/255), label="False negative"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        framealpha=0.3,
        facecolor="#333333",
        labelcolor="white",
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _slice_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp + 1e-6) / float(pred.sum() + gt.sum() + 1e-6)


# ── sample selection ──────────────────────────────────────────────────────────

def select_samples(dataset, n: int, cancer_only: bool, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = list(range(len(dataset)))
    if cancer_only:
        indices = [i for i in indices if dataset[i][1]["seg_map"].min() == 0]
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    return sorted(chosen.tolist())


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",           required=True)
    p.add_argument("--eomt_ckpt",          default=None)
    p.add_argument("--vit_adapter_ckpt",   default=None)
    p.add_argument("--m2f_ckpt",           default=None)
    p.add_argument("--img_size",           type=int, default=512)
    p.add_argument("--n_samples",          type=int, default=12,
                   help="Number of validation samples to visualise")
    p.add_argument("--dataset",            choices=["lung", "hepatic"], default="lung",
                   help="Which dataset to load (lung=Task06, hepatic=Task08)")
    p.add_argument("--cancer_only",        action="store_true",
                   help="Only show slices that contain a foreground class in the GT")
    p.add_argument("--vis_dir",            default="./vis_output")
    p.add_argument("--samples_per_figure", type=int, default=6,
                   help="Rows per output PNG (creates multiple files if n_samples > this)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.seed_everything(42)

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_size, args.img_size)
    if args.dataset == "hepatic":
        from data.msd_hepatic import MSDHepaticDataset
        val_ds = MSDHepaticDataset(args.data_dir, split="val", img_size=img_size, augment=False)
    else:
        from data.msd_lung import MSDLungDataset
        val_ds = MSDLungDataset(args.data_dir, split="val", img_size=img_size, augment=False)

    chosen_idx = select_samples(val_ds, args.n_samples, args.cancer_only)
    print(f"Selected {len(chosen_idx)} samples (cancer_only={args.cancer_only})")

    # ── load models ──────────────────────────────────────────────────────────
    models = {}
    if args.eomt_ckpt:
        print("Loading EoMT …")
        models["EoMT"] = (_load_eomt(args.eomt_ckpt, device), "eomt")
    if args.vit_adapter_ckpt:
        print("Loading ViT-Adapter+M2F …")
        models["ViT-Adapter+M2F"] = (_load_vit_adapter(args.vit_adapter_ckpt, device), "vit_adapter_m2f")
    if args.m2f_ckpt:
        print("Loading Mask2Former …")
        models["Mask2Former"] = (_load_m2f(args.m2f_ckpt, device), "mask2former")

    if not models:
        print("No checkpoints provided. Pass --eomt_ckpt and/or --vit_adapter_ckpt.")
        return

    model_names = list(models.keys())

    # ── collect samples ───────────────────────────────────────────────────────
    print("Running inference …")
    samples = []
    for idx in chosen_idx:
        img, target = val_ds[idx]           # img: [3, H, W] uint8 tensor
        seg = target["seg_map"]             # [H, W]

        imgs_batch = img.unsqueeze(0)       # [1, 3, H, W]
        preds = {}
        for name, (module, model_key) in models.items():
            preds[name] = _predict(module, imgs_batch, img_size, model_key)[0]  # [H, W]

        samples.append((img.cpu(), seg.cpu(), preds))

    # ── render figures ────────────────────────────────────────────────────────
    spf = args.samples_per_figure
    for fig_idx, start in enumerate(range(0, len(samples), spf)):
        chunk = samples[start : start + spf]
        out   = vis_dir / f"predictions_{fig_idx + 1:02d}.png"
        render_samples(chunk, model_names, out)

    print(f"\nDone. {len(chosen_idx)} samples → {vis_dir}/")


if __name__ == "__main__":
    main()
