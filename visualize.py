"""
Visualise segmentation predictions for EoMT and ViT-Adapter+Mask2Former.

Two independent modes (both can run in the same invocation):

  SLICE mode (default):
      Shows individual 2D slices. Each row = one val sample.
      Columns: CT | Ground Truth | EoMT | ViT-Adapter+M2F [| Mask2Former]

  MIP mode (--mip):
      Maximum Intensity Projection across all axial slices of a volume.
      CT MIP = max pixel intensity along z.
      Mask MIP = OR of per-slice predictions along z.
      Overlay colours: green=TP, red=FP, yellow=FN.
      Caption = volumetric 3-D DICE for that case.

Usage:
    # Slice panels (lung)
    python visualize.py --dataset lung --data_dir ./data/lung \\
        --eomt_ckpt ./checkpoints/eomt/ckpts/last.ckpt \\
        --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \\
        --n_samples 12 --cancer_only --vis_dir ./vis_output

    # MIP panels
    python visualize.py --dataset hepatic --data_dir ./data/hepatic \\
        --eomt_ckpt ./checkpoints/eomt/ckpts/last.ckpt \\
        --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \\
        --mip --n_cases 6 --vis_dir ./vis_output

    # Both together
    python visualize.py --dataset lung --data_dir ./data/lung \\
        --eomt_ckpt ./checkpoints/eomt/ckpts/last.ckpt \\
        --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \\
        --n_samples 8 --mip --n_cases 4 --vis_dir ./vis_output
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import lightning as L
from runners.dice_metric import IGNORE_IDX

SMOOTH = 1e-6


# ── model loaders ────────────────────────────────────────────────────────────

def _load_eomt(ckpt, device):
    from runners.eomt_runner import EoMTMedicalModule
    m = EoMTMedicalModule.load_from_checkpoint(ckpt, map_location=device)
    m.network.masked_attn_enabled = False
    return m.eval().to(device)


def _load_vit_adapter(ckpt, device):
    from runners.vit_adapter_m2f_runner import ViTAdapterM2FModule
    return ViTAdapterM2FModule.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


def _load_m2f(ckpt, device):
    from runners.mask2former_runner import Mask2FormerMedicalModule
    return Mask2FormerMedicalModule.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


# ── inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def _logits(module, imgs, img_size, model_name):
    """Returns per-pixel logits [B, C, H, W] for all foreground classes."""
    imgs = imgs.to(next(module.parameters()).device)
    if model_name == "eomt":
        ml_list, cl_list = module(imgs)
        ml_up = F.interpolate(ml_list[-1], img_size, mode="bilinear", align_corners=False)
        return module._to_per_pixel_logits(ml_up, cl_list[-1])
    elif model_name == "vit_adapter_m2f":
        ml_list, cl_list = module(imgs)
        return module._to_per_pixel_logits(ml_list[-1], cl_list[-1], img_size)
    else:
        outputs = module(imgs)
        return module._semseg_from_outputs(outputs, img_size)


@torch.no_grad()
def _predict(module, imgs, img_size, model_name):
    """Returns [B, H, W] bool prediction for class 0 (used by slice mode)."""
    return (_logits(module, imgs, img_size, model_name)[:, 0] > 0.5).cpu()


@torch.no_grad()
def _predict_mc(module, imgs, img_size, model_name):
    """Returns [B, C, H, W] bool predictions for all classes (used by MIP mode)."""
    return (_logits(module, imgs, img_size, model_name) > 0.5).cpu()


# ── overlay renderer ──────────────────────────────────────────────────────────

def _make_overlay(ct: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    ct:   [H, W] uint8 grayscale
    gt:   [H, W] bool
    pred: [H, W] bool
    Returns [H, W, 3] uint8 RGB with TP=green, FP=red, FN=yellow.
    """
    rgb   = np.stack([ct] * 3, axis=-1).astype(np.float32)
    alpha = 0.55
    tp    = pred & gt
    fp    = pred & ~gt
    fn    = ~pred & gt
    rgb[tp] = rgb[tp] * (1 - alpha) + np.array([0,   210, 80],  dtype=np.float32) * alpha
    rgb[fp] = rgb[fp] * (1 - alpha) + np.array([220, 40,  40],  dtype=np.float32) * alpha
    rgb[fn] = rgb[fn] * (1 - alpha) + np.array([255, 200, 0],   dtype=np.float32) * alpha
    return rgb.clip(0, 255).astype(np.uint8)


def _legend_patches():
    return [
        mpatches.Patch(color=(0/255,   210/255, 80/255),  label="True positive"),
        mpatches.Patch(color=(220/255,  40/255, 40/255),  label="False positive"),
        mpatches.Patch(color=(255/255, 200/255,  0/255),  label="False negative"),
    ]


# ── SLICE MODE ────────────────────────────────────────────────────────────────

def _slice_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp + SMOOTH) / float(pred.sum() + gt.sum() + SMOOTH)


def select_samples(dataset, n: int, cancer_only: bool, seed: int = 42):
    rng     = np.random.default_rng(seed)
    indices = list(range(len(dataset)))
    if cancer_only:
        indices = [i for i in indices if dataset[i][1]["seg_map"].min() == 0]
    chosen  = rng.choice(indices, size=min(n, len(indices)), replace=False)
    return sorted(chosen.tolist())


def render_slices(samples, model_names, out_path: Path):
    """
    samples: list of (img [3,H,W], seg [H,W], preds dict name→[H,W] bool)
    Columns: CT (z) | Ground Truth | model_1 | model_2 ...
    """
    n_cols = 2 + len(model_names)
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")

    col_titles = ["CT (z)", "Ground Truth"] + model_names
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="white", fontsize=11, pad=6)

    for row_idx, (img, seg, preds) in enumerate(samples):
        ct = img[1].numpy()
        gt = (seg == 0).numpy()

        axes[row_idx, 0].imshow(ct, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(_make_overlay(ct, gt, gt))
        axes[row_idx, 1].axis("off")

        for col_offset, name in enumerate(model_names, start=2):
            pred    = preds[name].numpy()
            dice    = _slice_dice(pred, gt)
            ax      = axes[row_idx, col_offset]
            ax.imshow(_make_overlay(ct, gt, pred))
            ax.set_xlabel(f"DICE = {dice:.3f}", color="#cccccc", fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
               framealpha=0.3, facecolor="#333333", labelcolor="white",
               fontsize=10, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── MIP MODE ──────────────────────────────────────────────────────────────────

def build_volume_index(data_dir: str, split: str, slice_names: list) -> dict:
    """
    Groups slice names (from metadata.json) by case ID.
    Names follow the pattern "<case_id>_z<z>" — split on the last "_z".

    Returns: dict  case_id → sorted list of (z, img_path, lbl_path)
    """
    data_dir = Path(data_dir)
    img_root = data_dir / split / "images"
    lbl_root = data_dir / split / "labels"

    volumes: dict[str, list] = {}
    for name in slice_names:
        parts   = name.rsplit("_z", 1)
        case_id = parts[0]
        z       = int(parts[1]) if len(parts) > 1 else 0
        volumes.setdefault(case_id, []).append(
            (z, img_root / f"{name}.npy", lbl_root / f"{name}.npy")
        )

    for c in volumes:
        volumes[c].sort(key=lambda x: x[0])

    return volumes


@torch.no_grad()
def compute_case_mip(case_slices, models, img_size, num_classes, device, batch_size=8):
    """
    Processes every slice of one volume and returns the MIP projections.

    Args:
        case_slices: list of (z, img_path, lbl_path) sorted by z
        models:      dict  display_name → (module, model_key)

    Returns:
        ct_mip    [H, W] uint8   — max of channel-1 (current slice) across all z
        gt_mip    [C, H, W] bool — OR of GT masks across z
        pred_mips dict  name → [C, H, W] bool — OR of predicted masks across z
        vol_dice  dict  name → list of per-class {"tp", "pred", "tgt"} accumulators
    """
    H, W = img_size
    ct_mip    = np.zeros((H, W), dtype=np.float32)
    gt_mip    = np.zeros((num_classes, H, W), dtype=bool)
    pred_mips = {name: np.zeros((num_classes, H, W), dtype=bool) for name in models}
    vol_dice  = {
        name: [{"tp": 0.0, "pred": 0.0, "tgt": 0.0} for _ in range(num_classes)]
        for name in models
    }

    for b_start in range(0, len(case_slices), batch_size):
        chunk = case_slices[b_start : b_start + batch_size]

        imgs_list, lbls_list = [], []
        for _, img_path, lbl_path in chunk:
            img_t = torch.from_numpy(np.load(img_path).astype(np.float32))
            lbl_t = torch.from_numpy(np.load(lbl_path).astype(np.int64))
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=img_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            lbl_t = F.interpolate(
                lbl_t.unsqueeze(0).unsqueeze(0).float(), size=img_size, mode="nearest"
            ).squeeze().long()
            imgs_list.append(img_t)
            lbls_list.append(lbl_t)

        imgs_batch = torch.stack(imgs_list)   # [B, 3, H, W]

        # CT MIP: max of the middle channel across the batch then running max
        ct_mip = np.maximum(ct_mip, imgs_batch[:, 1].numpy().max(axis=0))

        # GT MIP: OR across z
        for lbl_t in lbls_list:
            lbl_np = lbl_t.numpy()
            for c in range(num_classes):
                gt_mip[c] |= (lbl_np == c)

        # Prediction MIP + volumetric DICE accumulation
        for name, (module, model_key) in models.items():
            pred_all = _predict_mc(module, imgs_batch, img_size, model_key)  # [B, C, H, W]
            for b_idx, lbl_t in enumerate(lbls_list):
                lbl_np = lbl_t.numpy()
                for c in range(num_classes):
                    pc = pred_all[b_idx, c].numpy()           # [H, W] bool
                    sc = (lbl_np == c)                        # [H, W] bool
                    pred_mips[name][c] |= pc
                    vol_dice[name][c]["tp"]   += float((pc & sc).sum())
                    vol_dice[name][c]["pred"] += float(pc.sum())
                    vol_dice[name][c]["tgt"]  += float(sc.sum())

    return ct_mip.clip(0, 255).astype(np.uint8), gt_mip, pred_mips, vol_dice


def render_mip_figure(cases_data, class_names, model_names, out_path: Path):
    """
    Renders a grid of MIP panels.

    Rows: one per volume (case)
    Columns: CT MIP | [GT (cls) | model_1 (cls) | model_2 (cls) ...] × num_classes

    Caption under each prediction cell: volumetric 3-D DICE for that case × class.
    """
    num_classes = len(class_names)
    n_models    = len(model_names)
    n_rows      = len(cases_data)
    # 1 CT MIP column + (1 GT + n_models) columns per class
    n_cols      = 1 + num_classes * (1 + n_models)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a1a")

    # Column headers
    col_titles = ["CT MIP"]
    for cname in class_names:
        col_titles.append(f"GT\n({cname})")
        for mname in model_names:
            col_titles.append(f"{mname}\n({cname})")
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="white", fontsize=8, pad=6)

    for row_idx, (case_id, ct_mip, gt_mip, pred_mips, vol_dice) in enumerate(cases_data):

        # ── CT MIP column ─────────────────────────────────────────────────
        ax = axes[row_idx, 0]
        ax.imshow(ct_mip, cmap="gray", vmin=0, vmax=255)
        ax.set_ylabel(case_id, color="#aaaaaa", fontsize=7, rotation=90, va="center")
        ax.set_yticks([])
        ax.set_xticks([])

        col_cursor = 1
        for c_idx, cname in enumerate(class_names):
            gt_c = gt_mip[c_idx]    # [H, W] bool

            # ── GT projection ─────────────────────────────────────────────
            ax = axes[row_idx, col_cursor]
            ax.imshow(_make_overlay(ct_mip, gt_c, gt_c))
            ax.axis("off")
            col_cursor += 1

            # ── Model predictions ─────────────────────────────────────────
            for mname in model_names:
                pred_c  = pred_mips[mname][c_idx]   # [H, W] bool
                d       = vol_dice[mname][c_idx]
                v_dice  = (2 * d["tp"] + SMOOTH) / (d["pred"] + d["tgt"] + SMOOTH)

                ax = axes[row_idx, col_cursor]
                ax.imshow(_make_overlay(ct_mip, gt_c, pred_c))
                ax.set_xlabel(f"vol.DICE = {v_dice:.3f}", color="#cccccc", fontsize=8)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                col_cursor += 1

    fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
               framealpha=0.3, facecolor="#333333", labelcolor="white",
               fontsize=10, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved MIP: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           choices=["lung", "hepatic"], default="lung")
    p.add_argument("--data_dir",          required=True)
    p.add_argument("--eomt_ckpt",         default=None)
    p.add_argument("--vit_adapter_ckpt",  default=None)
    p.add_argument("--m2f_ckpt",          default=None)
    p.add_argument("--img_size",          type=int, default=512)
    p.add_argument("--vis_dir",           default="./vis_output")

    # ── slice mode ──────────────────────────────────────────────────────────
    p.add_argument("--n_samples",         type=int, default=12,
                   help="Number of individual slices to visualise (0 to skip slice panels)")
    p.add_argument("--cancer_only",       action="store_true",
                   help="Only select slices that contain a foreground class")
    p.add_argument("--samples_per_figure", type=int, default=6,
                   help="Rows per output PNG")

    # ── MIP mode ────────────────────────────────────────────────────────────
    p.add_argument("--mip",              action="store_true",
                   help="Generate Maximum Intensity Projection panels")
    p.add_argument("--n_cases",          type=int, default=4,
                   help="Number of volumes (cases) to project for MIP panels")
    p.add_argument("--mip_batch",        type=int, default=8,
                   help="Slices per inference batch during MIP computation")
    p.add_argument("--cases_per_figure", type=int, default=4,
                   help="Rows (cases) per MIP output PNG")

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

    class_names = val_ds.class_names

    # ── load models ───────────────────────────────────────────────────────────
    models: dict[str, tuple] = {}
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

    # ══ SLICE MODE ═══════════════════════════════════════════════════════════

    if args.n_samples > 0:
        print(f"\n── Slice panels ({args.n_samples} samples) ──")
        chosen_idx = select_samples(val_ds, args.n_samples, args.cancer_only)
        print(f"Selected {len(chosen_idx)} slices (cancer_only={args.cancer_only})")

        samples = []
        for idx in chosen_idx:
            img, target = val_ds[idx]
            seg         = target["seg_map"]
            imgs_batch  = img.unsqueeze(0)
            preds       = {name: _predict(module, imgs_batch, img_size, mkey)[0]
                           for name, (module, mkey) in models.items()}
            samples.append((img.cpu(), seg.cpu(), preds))

        spf = args.samples_per_figure
        for fig_idx, start in enumerate(range(0, len(samples), spf)):
            render_slices(samples[start:start + spf], model_names,
                          vis_dir / f"predictions_{fig_idx + 1:02d}.png")

    # ══ MIP MODE ═════════════════════════════════════════════════════════════

    if args.mip:
        print(f"\n── MIP panels ({args.n_cases} volumes) ──")

        vol_index = build_volume_index(args.data_dir, "val", val_ds.slice_names)
        all_case_ids = list(vol_index.keys())

        rng          = np.random.default_rng(42)
        chosen_cases = rng.choice(
            all_case_ids, size=min(args.n_cases, len(all_case_ids)), replace=False
        ).tolist()

        cases_data = []
        for case_id in chosen_cases:
            slices = vol_index[case_id]
            print(f"  {case_id}: {len(slices)} slices …")
            ct_mip, gt_mip, pred_mips, vol_dice = compute_case_mip(
                slices, models, img_size, val_ds.num_classes, device,
                batch_size=args.mip_batch,
            )
            cases_data.append((case_id, ct_mip, gt_mip, pred_mips, vol_dice))

        cpf = args.cases_per_figure
        for fig_idx, start in enumerate(range(0, len(cases_data), cpf)):
            render_mip_figure(
                cases_data[start:start + cpf], class_names, model_names,
                vis_dir / f"mip_{fig_idx + 1:02d}.png",
            )

    print(f"\nDone → {vis_dir}/")


if __name__ == "__main__":
    main()
