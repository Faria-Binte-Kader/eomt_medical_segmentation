"""
Prepare Medical Segmentation Decathlon (MSD) Lung dataset (Task06).

Extracts 2D axial slices from 3D NIfTI volumes, applies lung CT windowing,
stacks adjacent slices as multi-slice pseudo-RGB [z-1, z, z+1], and saves
as numpy arrays for fast loading during training.

Usage:
    python prepare_data.py --data_dir /path/to/Task06_Lung --output_dir ./data/lung

MSD Lung label convention:
    0 = background
    1 = cancer

Output label convention (remapped, 0-indexed):
    0   = cancer
    255 = background / ignore

Multi-slice RGB:
    Channel 0 = slice z-1   (previous axial slice)
    Channel 1 = slice z     (current axial slice)
    Channel 2 = slice z+1   (next axial slice)
    At volume boundaries, edge slices are replicated.
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


# Lung CT windowing: captures lung parenchyma (-1350 to 150 HU)
WINDOW_CENTER = -600   # HU
WINDOW_WIDTH  = 1500   # HU  →  range [-1350, 150] HU


def apply_window(vol: np.ndarray) -> np.ndarray:
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 255.0
    return vol.astype(np.uint8)


def remap_labels(lbl: np.ndarray) -> np.ndarray:
    """Remap MSD Lung labels to 0-indexed foreground + 255 background."""
    out = np.full_like(lbl, fill_value=255, dtype=np.uint8)
    out[lbl == 1] = 0   # cancer → 0
    return out


def process_volume(
    img_path: Path,
    lbl_path: Path,
    output_dir: Path,
    split: str,
    case_id: str,
) -> list[str]:
    img_nib = nib.load(img_path)
    lbl_nib = nib.load(lbl_path)

    img_vol = img_nib.get_fdata()                       # [H, W, D]  float64
    lbl_vol = lbl_nib.get_fdata().astype(np.int16)      # [H, W, D]

    img_win = apply_window(img_vol)                     # [H, W, D]  uint8

    n_slices = img_vol.shape[2]
    saved    = []

    img_dir = output_dir / split / "images"
    lbl_dir = output_dir / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for z in range(n_slices):
        lbl_slice = lbl_vol[:, :, z]

        # Keep slices that contain cancer OR show meaningful lung/body content.
        # Windowed value > 100 corresponds to lung parenchyma and denser tissue
        # (at -600 HU, lung parenchyma maps to ~127; pure air maps to ~60).
        has_cancer  = bool((lbl_slice == 1).any())
        has_content = float((img_win[:, :, z] > 100).mean()) > 0.05

        if not (has_cancer or has_content):
            continue

        # Multi-slice pseudo-RGB: stack [z-1, z, z+1], clamping at boundaries
        z_prev = max(0, z - 1)
        z_next = min(n_slices - 1, z + 1)
        img_rgb = np.stack([
            img_win[:, :, z_prev],
            img_win[:, :, z],
            img_win[:, :, z_next],
        ], axis=0)  # [3, H, W] uint8

        lbl_remapped = remap_labels(lbl_slice)   # [H, W] uint8  (0=cancer, 255=bg)

        name = f"{case_id}_z{z:04d}"
        np.save(img_dir / f"{name}.npy", img_rgb)
        np.save(lbl_dir / f"{name}.npy", lbl_remapped)
        saved.append(name)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Prepare MSD Lung 2D slices")
    parser.add_argument("--data_dir",     required=True,
                        help="Path to Task06_Lung/ directory")
    parser.add_argument("--output_dir",   default="./data/lung",
                        help="Where to write processed slices")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "dataset.json") as f:
        info = json.load(f)

    cases = info["training"]
    random.shuffle(cases)
    n_val       = max(1, int(len(cases) * args.val_fraction))
    val_cases   = cases[:n_val]
    train_cases = cases[n_val:]

    print(f"Dataset: {len(cases)} volumes  |  train={len(train_cases)}  val={len(val_cases)}")

    metadata = {
        "train": [],
        "val":   [],
        "num_classes": 1,
        "class_names": ["cancer"],
        "window_center": WINDOW_CENTER,
        "window_width":  WINDOW_WIDTH,
    }

    for split, split_cases in [("train", train_cases), ("val", val_cases)]:
        print(f"\nProcessing {split} …")
        for case in tqdm(split_cases):
            img_path = data_dir / case["image"].lstrip("./")
            lbl_path = data_dir / case["label"].lstrip("./")
            case_id  = Path(case["image"]).stem.split(".")[0]   # strip .nii from stem

            slices = process_volume(img_path, lbl_path, output_dir, split, case_id)
            metadata[split].extend(slices)
            print(f"  {case_id}: {len(slices)} valid slices", end="\r")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n\nDone!")
    print(f"  Train slices : {len(metadata['train'])}")
    print(f"  Val   slices : {len(metadata['val'])}")
    print(f"  Saved to     : {output_dir}")


if __name__ == "__main__":
    main()
