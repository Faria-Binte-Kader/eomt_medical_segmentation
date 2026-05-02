"""
Evaluation script — run all three models on the validation set and print a
side-by-side DICE / IoU comparison table.

Usage:
    python evaluate.py \\
        --data_dir  ./data/lung \\
        --eomt_ckpt ./checkpoints/eomt/ckpts/last.ckpt \\
        --m2f_ckpt  ./checkpoints/mask2former/ckpts/last.ckpt \\
        --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \\
        --img_size  512

Pass any subset of checkpoint flags to evaluate just those models.
Results are also written to eval_results.json.
"""

import argparse
import json

import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader

from data.msd_lung import MSDLungDataset, collate_fn
from runners.dice_metric import IGNORE_IDX


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_eomt(ckpt_path: str, device):
    from runners.eomt_runner import EoMTMedicalModule
    module = EoMTMedicalModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    return module.to(device)


def _load_m2f(ckpt_path: str, device):
    from runners.mask2former_runner import Mask2FormerMedicalModule
    module = Mask2FormerMedicalModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    return module.to(device)


def _load_vit_adapter(ckpt_path: str, device):
    from runners.vit_adapter_m2f_runner import ViTAdapterM2FModule
    module = ViTAdapterM2FModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    return module.to(device)


@torch.no_grad()
def evaluate_model(module, dataloader, img_size, model_name, device) -> dict:
    """Run inference and return cancer DICE and IoU."""
    dice_tp   = 0.0
    dice_pred = 0.0
    dice_tgt  = 0.0
    iou_inter = 0.0
    iou_union = 0.0
    smooth    = 1e-6

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)

        # ── forward pass ──────────────────────────────────────────────
        if model_name == "eomt":
            was = module.network.masked_attn_enabled
            module.network.masked_attn_enabled = False
            ml_per_block, cl_per_block = module(imgs)
            module.network.masked_attn_enabled = was
            ml_up  = F.interpolate(ml_per_block[-1], img_size, mode="bilinear", align_corners=False)
            logits = module._to_per_pixel_logits(ml_up, cl_per_block[-1])
        elif model_name == "vit_adapter_m2f":
            ml_list, cl_list = module(imgs)
            logits = module._to_per_pixel_logits(ml_list[-1], cl_list[-1], img_size)
        else:  # mask2former
            outputs = module(imgs)
            logits  = module._semseg_from_outputs(outputs, img_size)

        # logits: [B, 1, H, W]  — threshold channel 0 for cancer prediction
        pred_cancer = logits[:, 0, ...] > 0.5   # [B, H, W] bool

        for j, target in enumerate(targets):
            seg   = target["seg_map"].to(device)   # [H, W]: 0=cancer, 255=ignore
            pc    = pred_cancer[j].float()
            sc    = (seg == 0).float()

            dice_tp   += (pc * sc).sum().item()
            dice_pred += pc.sum().item()
            dice_tgt  += sc.sum().item()

            valid  = seg != IGNORE_IDX
            pi     = pred_cancer[j] & valid
            ti     = (seg == 0) & valid
            iou_inter += (pi & ti).float().sum().item()
            iou_union += (pi | ti).float().sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  [{model_name}] processed {batch_idx + 1} batches …", end="\r")

    dice = (2.0 * dice_tp + smooth) / (dice_pred + dice_tgt + smooth)
    iou  = (iou_inter + smooth) / (iou_union + smooth)

    return {
        "dice_cancer": round(float(dice), 4),
        "iou_cancer":  round(float(iou),  4),
        "dice_mean":   round(float(dice), 4),   # only 1 class, mean == cancer DICE
        "miou":        round(float(iou),  4),
    }


def print_table(results: dict[str, dict]):
    """Pretty-print a comparison table."""
    header_models = list(results.keys())
    if not header_models:
        return
    all_keys = list(next(iter(results.values())).keys())

    col_w = 18
    sep   = "─" * (14 + col_w * len(header_models))

    print(f"\n{'':14s}", end="")
    for m in header_models:
        print(f"{m:>{col_w}s}", end="")
    print()
    print(sep)

    for key in all_keys:
        print(f"{key:<14s}", end="")
        for m in header_models:
            val  = results[m].get(key, "—")
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"{cell:>{col_w}s}", end="")
        print()

    print(sep)


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate models on MSD Lung val set")
    p.add_argument("--data_dir",          required=True)
    p.add_argument("--eomt_ckpt",         default=None)
    p.add_argument("--m2f_ckpt",          default=None)
    p.add_argument("--vit_adapter_ckpt",  default=None)
    p.add_argument("--img_size",          type=int, default=512)
    p.add_argument("--batch_size",        type=int, default=4)
    p.add_argument("--num_workers",       type=int, default=4)
    p.add_argument("--output_json",       default="eval_results.json")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.seed_everything(42)

    img_size = (args.img_size, args.img_size)
    val_ds   = MSDLungDataset(args.data_dir, split="val", img_size=img_size, augment=False)
    val_dl   = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )

    all_results: dict[str, dict] = {}

    if args.eomt_ckpt:
        print(f"\nEvaluating EoMT …  (checkpoint: {args.eomt_ckpt})")
        module = _load_eomt(args.eomt_ckpt, device)
        all_results["EoMT"] = evaluate_model(module, val_dl, img_size, "eomt", device)
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.m2f_ckpt:
        print(f"\nEvaluating Mask2Former …  (checkpoint: {args.m2f_ckpt})")
        module = _load_m2f(args.m2f_ckpt, device)
        all_results["Mask2Former"] = evaluate_model(module, val_dl, img_size, "mask2former", device)
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.vit_adapter_ckpt:
        print(f"\nEvaluating ViT-Adapter+M2F …  (checkpoint: {args.vit_adapter_ckpt})")
        module = _load_vit_adapter(args.vit_adapter_ckpt, device)
        all_results["ViT-Adapter+M2F"] = evaluate_model(module, val_dl, img_size, "vit_adapter_m2f", device)
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("No checkpoints provided. Pass --eomt_ckpt, --m2f_ckpt, and/or --vit_adapter_ckpt.")
        return

    print_table(all_results)

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
