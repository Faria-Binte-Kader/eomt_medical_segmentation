"""
Evaluation script — run all three models on the validation set and print a
side-by-side DICE / IoU comparison table, plus efficiency metrics
(# params, MACs, FPS, peak GPU memory).

Usage:
    python evaluate.py \\
        --dataset lung \\
        --data_dir  ./data/lung \\
        --eomt_ckpt        ./checkpoints/eomt/ckpts/last.ckpt \\
        --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \\
        --m2f_ckpt         ./checkpoints/mask2former/ckpts/last.ckpt \\
        --img_size 512

MACs require:  pip install fvcore
If fvcore is absent, MACs are reported as N/A.
Results are written to eval_results.json (or --output_json).
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader

from runners.dice_metric import IGNORE_IDX

SMOOTH = 1e-6


# ── model loaders ────────────────────────────────────────────────────────────

def _load_eomt(ckpt, device):
    from runners.eomt_runner import EoMTMedicalModule
    return EoMTMedicalModule.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


def _load_m2f(ckpt, device):
    from runners.mask2former_runner import Mask2FormerMedicalModule
    return Mask2FormerMedicalModule.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


def _load_vit_adapter(ckpt, device):
    from runners.vit_adapter_m2f_runner import ViTAdapterM2FModule
    return ViTAdapterM2FModule.load_from_checkpoint(ckpt, map_location=device).eval().to(device)


# ── inference helper ─────────────────────────────────────────────────────────

def _forward(module, model_name, imgs, img_size):
    """Single forward pass → per-pixel logits [B, C, H, W]."""
    if model_name == "eomt":
        module.network.masked_attn_enabled = False
        ml_per_block, cl_per_block = module(imgs)
        ml_up = F.interpolate(ml_per_block[-1], img_size, mode="bilinear", align_corners=False)
        return module._to_per_pixel_logits(ml_up, cl_per_block[-1])
    elif model_name == "vit_adapter_m2f":
        ml_list, cl_list = module(imgs)
        return module._to_per_pixel_logits(ml_list[-1], cl_list[-1], img_size)
    else:
        outputs = module(imgs)
        return module._semseg_from_outputs(outputs, img_size)


# ── profiling ─────────────────────────────────────────────────────────────────

class _FlopWrapper(torch.nn.Module):
    """Thin nn.Module wrapper so fvcore can trace through the full forward."""
    def __init__(self, module, model_name, img_size):
        super().__init__()
        self._m  = module
        self._mn = model_name
        self._sz = img_size

    def forward(self, x):
        return _forward(self._m, self._mn, x, self._sz)


@torch.no_grad()
def profile_model(module, model_name, img_size, device, n_warmup=10, n_runs=50):
    """
    Returns a dict with:
      params_M    — total parameters in millions
      macs_G      — multiply-accumulate ops in billions (requires fvcore)
      fps         — images per second (batch size 1, single GPU)
      peak_mem_mb — peak GPU memory for one forward pass in MB
    """
    dummy = torch.randn(1, 3, *img_size, device=device)

    # 1. parameter count
    n_params = sum(p.numel() for p in module.parameters())

    # 2. MACs — fvcore traces the graph and sums registered flop counters
    macs_val = None
    macs_str = "N/A"
    try:
        from fvcore.nn import FlopCountAnalysis
        wrapper = _FlopWrapper(module, model_name, img_size).to(device)
        fa = FlopCountAnalysis(wrapper, dummy)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        total_flops = fa.total()
        macs_val = total_flops / 2          # 1 MAC = 2 FLOPs (multiply + add)
        macs_str = round(macs_val / 1e9, 2)
    except ImportError:
        macs_str = "N/A (pip install fvcore)"
    except Exception:
        macs_str = "N/A (trace failed)"

    # 3. FPS — warmup, then timed run of n_runs single-image forwards
    for _ in range(n_warmup):
        _forward(module, model_name, dummy, img_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _forward(module, model_name, dummy, img_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fps = round(n_runs / (time.perf_counter() - t0), 1)

    # 4. Peak GPU memory during a single forward pass
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _forward(module, model_name, dummy, img_size)
        torch.cuda.synchronize()
        peak_mb = round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
    else:
        peak_mb = "N/A"

    return {
        "params_M":    round(n_params / 1e6, 2),
        "macs_G":      macs_str,
        "fps":         fps,
        "peak_mem_mb": peak_mb,
    }


# ── segmentation evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(module, dataloader, img_size, model_name, device, class_names) -> dict:
    """Accumulate global DICE and IoU over the full validation set."""
    num_classes = len(class_names)
    dice_store  = [{"tp": 0.0, "pred": 0.0, "tgt": 0.0} for _ in range(num_classes)]
    iou_store   = [{"inter": 0.0, "union": 0.0}          for _ in range(num_classes)]

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs   = imgs.to(device)
        logits = _forward(module, model_name, imgs, img_size)   # [B, C, H, W]
        pred   = logits > 0.5                                   # bool

        for j, target in enumerate(targets):
            seg   = target["seg_map"].to(device)
            valid = seg != IGNORE_IDX
            for c in range(num_classes):
                pc = pred[j, c].float()
                sc = (seg == c).float()
                dice_store[c]["tp"]   += (pc * sc).sum().item()
                dice_store[c]["pred"] += pc.sum().item()
                dice_store[c]["tgt"]  += sc.sum().item()
                pi = pred[j, c] & valid
                ti = (seg == c) & valid
                iou_store[c]["inter"] += (pi & ti).float().sum().item()
                iou_store[c]["union"] += (pi | ti).float().sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  [{model_name}] {batch_idx + 1} batches …", end="\r")

    results = {}
    dice_vals, iou_vals = [], []
    for c, name in enumerate(class_names):
        tp, pred_s, tgt = dice_store[c]["tp"], dice_store[c]["pred"], dice_store[c]["tgt"]
        inter, union    = iou_store[c]["inter"], iou_store[c]["union"]
        dice = (2.0 * tp + SMOOTH) / (pred_s + tgt + SMOOTH)
        iou  = (inter + SMOOTH) / (union + SMOOTH)
        dice_vals.append(dice)
        iou_vals.append(iou)
        results[f"dice_{name}"] = round(float(dice), 4)
        results[f"iou_{name}"]  = round(float(iou),  4)

    results["dice_mean"] = round(sum(dice_vals) / len(dice_vals), 4)
    results["miou"]      = round(sum(iou_vals)  / len(iou_vals),  4)
    return results


# ── display ──────────────────────────────────────────────────────────────────

_PROFILE_KEYS = {"params_M", "macs_G", "fps", "peak_mem_mb"}
_PROFILE_LABELS = {
    "params_M":    "params (M)",
    "macs_G":      "MACs (G)",
    "fps":         "FPS",
    "peak_mem_mb": "mem (MB)",
}


def print_table(results: dict):
    models = list(results.keys())
    if not models:
        return

    all_keys = list(next(iter(results.values())).keys())
    seg_keys  = [k for k in all_keys if k not in _PROFILE_KEYS]

    col_w = 22
    n_col = len(models)
    width = 18 + col_w * n_col

    def _header():
        print(f"  {'':16s}", end="")
        for m in models:
            print(f"{m:>{col_w}s}", end="")
        print()

    def _sep(char="─"):
        print("  " + char * (width - 2))

    def _row(label, key):
        print(f"  {label:<16s}", end="")
        for m in models:
            val  = results[m].get(key, "—")
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"{cell:>{col_w}s}", end="")
        print()

    print()
    _header()
    _sep()

    print("  Segmentation")
    _sep("·")
    for k in seg_keys:
        _row(k, k)

    print()
    print("  Efficiency (single image, GPU)")
    _sep("·")
    for k in ["params_M", "macs_G", "fps", "peak_mem_mb"]:
        _row(_PROFILE_LABELS[k], k)

    _sep()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          choices=["lung", "hepatic"], default="lung")
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--eomt_ckpt",        default=None)
    p.add_argument("--m2f_ckpt",         default=None)
    p.add_argument("--vit_adapter_ckpt", default=None)
    p.add_argument("--img_size",         type=int, default=512)
    p.add_argument("--batch_size",       type=int, default=4)
    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--profile_runs",     type=int, default=50,
                   help="Number of timed forward passes for FPS measurement")
    p.add_argument("--output_json",      default="eval_results.json")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.seed_everything(42)

    img_size = (args.img_size, args.img_size)

    if args.dataset == "hepatic":
        from data.msd_hepatic import MSDHepaticDataset, collate_fn as ds_collate
        val_ds = MSDHepaticDataset(args.data_dir, split="val", img_size=img_size, augment=False)
    else:
        from data.msd_lung import MSDLungDataset, collate_fn as ds_collate
        val_ds = MSDLungDataset(args.data_dir, split="val", img_size=img_size, augment=False)

    class_names = val_ds.class_names
    print(f"Dataset: {args.dataset}  |  classes: {class_names}  |  val slices: {len(val_ds)}")

    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=ds_collate,
    )

    loaders = []
    if args.eomt_ckpt:
        loaders.append(("EoMT",           "eomt",           _load_eomt,        args.eomt_ckpt))
    if args.vit_adapter_ckpt:
        loaders.append(("ViT-Adapter+M2F","vit_adapter_m2f",_load_vit_adapter, args.vit_adapter_ckpt))
    if args.m2f_ckpt:
        loaders.append(("Mask2Former",    "mask2former",    _load_m2f,         args.m2f_ckpt))

    if not loaders:
        print("No checkpoints provided. Pass --eomt_ckpt, --vit_adapter_ckpt, and/or --m2f_ckpt.")
        return

    all_results: dict[str, dict] = {}

    for display_name, model_name, load_fn, ckpt in loaders:
        print(f"\n{'─'*60}")
        print(f"  {display_name}  —  {ckpt}")
        print(f"{'─'*60}")

        module = load_fn(ckpt, device)

        print("  Profiling …")
        prof = profile_model(module, model_name, img_size, device, n_runs=args.profile_runs)
        print(f"  params={prof['params_M']} M  |  MACs={prof['macs_G']} G  "
              f"|  FPS={prof['fps']}  |  mem={prof['peak_mem_mb']} MB")

        print("  Evaluating segmentation …")
        seg = evaluate_model(module, val_dl, img_size, model_name, device, class_names)
        print(f"  dice_mean={seg['dice_mean']:.4f}  |  miou={seg['miou']:.4f}")

        all_results[display_name] = {**seg, **prof}

        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(all_results)

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
