"""
Prepare Medical Segmentation Decathlon (MSD) Hepatic Vessel dataset (Task08).

Extracts 2D axial slices from 3D NIfTI volumes, applies soft-tissue CT windowing,
stacks adjacent slices as multi-slice pseudo-RGB [z-1, z, z+1], and saves as numpy
arrays for fast loading during training.

Usage:
    python prepare_hepatic.py \
        --data_dir /path/to/Task08_HepaticVessel \
        --output_dir ./data/hepatic

MSD HepaticVessel label convention:
    0 = background
    1 = vessel
    2 = tumour

Output label convention (remapped, 0-indexed):
    0   = vessel
    1   = tumour
    255 = background / ignore
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


# Soft-tissue / abdominal window: captures liver parenchyma and hepatic vessels.
# Liver ~40–80 HU, hepatic vessels ~50–70 HU (portal venous phase), tumours variable.
WINDOW_CENTER = 40    # HU
WINDOW_WIDTH  = 400   # HU  →  range [-160, 240] HU


def apply_window(vol: np.ndarray) -> np.ndarray:
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 255.0
    return vol.astype(np.uint8)


def remap_labels(lbl: np.ndarray) -> np.ndarray:
    """Remap MSD HepaticVessel labels to 0-indexed foreground + 255 background."""
    out = np.full_like(lbl, fill_value=255, dtype=np.uint8)
    out[lbl == 1] = 0   # vessel → 0
    out[lbl == 2] = 1   # tumour → 1
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

        has_vessel  = bool((lbl_slice == 1).any())
        has_tumour  = bool((lbl_slice == 2).any())
        # Windowed value > 100 corresponds to HU > -60, capturing liver and soft tissue.
        # This keeps slices with visible abdominal anatomy and excludes pure-air slices.
        has_content = float((img_win[:, :, z] > 100).mean()) > 0.05

        if not (has_vessel or has_tumour or has_content):
            continue

        z_prev = max(0, z - 1)
        z_next = min(n_slices - 1, z + 1)
        img_rgb = np.stack([
            img_win[:, :, z_prev],
            img_win[:, :, z],
            img_win[:, :, z_next],
        ], axis=0)  # [3, H, W] uint8

        lbl_remapped = remap_labels(lbl_slice)   # [H, W] uint8  (0=vessel, 1=tumour, 255=bg)

        name = f"{case_id}_z{z:04d}"
        np.save(img_dir / f"{name}.npy", img_rgb)
        np.save(lbl_dir / f"{name}.npy", lbl_remapped)
        saved.append(name)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Prepare MSD HepaticVessel 2D slices")
    parser.add_argument("--data_dir",     required=True,
                        help="Path to Task08_HepaticVessel/ directory")
    parser.add_argument("--output_dir",   default="./data/hepatic",
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
        "num_classes": 2,
        "class_names": ["vessel", "tumour"],
        "window_center": WINDOW_CENTER,
        "window_width":  WINDOW_WIDTH,
    }

    for split, split_cases in [("train", train_cases), ("val", val_cases)]:
        print(f"\nProcessing {split} …")
        for case in tqdm(split_cases):
            img_path = data_dir / case["image"].lstrip("./")
            lbl_path = data_dir / case["label"].lstrip("./")
            case_id  = Path(case["image"]).stem.split(".")[0]

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
