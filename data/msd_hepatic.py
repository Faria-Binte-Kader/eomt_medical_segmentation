"""
MSD HepaticVessel (Task08) 2D-slice dataset.

Expects the directory layout produced by prepare_hepatic.py:
    <data_dir>/
        metadata.json
        train/
            images/<case_id>_z<z>.npy   [3, H, W] uint8   (multi-slice [z-1,z,z+1])
            labels/<case_id>_z<z>.npy   [H, W]    uint8
        val/
            images/ …
            labels/ …

Label convention (remapped, 0-indexed):
    0   = vessel
    1   = tumour
    255 = background / ignore

The __getitem__ return value matches MSDLungDataset exactly:
    img    : FloatTensor [3, img_size, img_size]  values in [0, 255]
    target : dict with
        "masks"   : BoolTensor  [N, img_size, img_size]  one mask per present class
        "labels"  : LongTensor  [N]                      class indices (0=vessel, 1=tumour)
        "seg_map" : LongTensor  [img_size, img_size]     per-pixel ground truth
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MSDHepaticDataset(Dataset):
    IGNORE_IDX = 255

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: tuple[int, int] = (512, 512),
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split    = split
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        with open(self.data_dir / "metadata.json") as f:
            meta = json.load(f)

        self.slice_names = meta[split]
        self.num_classes = meta["num_classes"]      # 2
        self.class_names = meta["class_names"]      # ["vessel", "tumour"]

    def __len__(self) -> int:
        return len(self.slice_names)

    def __getitem__(self, idx: int):
        name = self.slice_names[idx]

        img_np = np.load(self.data_dir / self.split / "images" / f"{name}.npy")  # [3, H, W]
        lbl_np = np.load(self.data_dir / self.split / "labels" / f"{name}.npy")  # [H, W]

        img = torch.from_numpy(img_np.astype(np.float32))   # [3, H, W]
        lbl = torch.from_numpy(lbl_np.astype(np.int64))     # [H, W]

        img = F.interpolate(
            img.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        lbl = F.interpolate(
            lbl.unsqueeze(0).unsqueeze(0).float(),
            size=self.img_size, mode="nearest",
        ).squeeze().long()

        if self.augment:
            img, lbl = self._augment(img, lbl)

        seg_map = lbl.clone()

        fg_classes = torch.unique(lbl)
        fg_classes = fg_classes[(fg_classes != self.IGNORE_IDX) & (fg_classes >= 0)]

        if len(fg_classes) == 0:
            masks  = torch.zeros(0, *self.img_size, dtype=torch.bool)
            labels = torch.zeros(0, dtype=torch.long)
        else:
            masks  = torch.stack([lbl == c for c in fg_classes])
            labels = fg_classes

        return img, {
            "masks":   masks,
            "labels":  labels,
            "seg_map": seg_map,
        }

    def _augment(self, img: torch.Tensor, lbl: torch.Tensor):
        if random.random() > 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl.unsqueeze(0)).squeeze(0)

        if random.random() > 0.5:
            img = TF.vflip(img)
            lbl = TF.vflip(lbl.unsqueeze(0)).squeeze(0)

        if random.random() > 0.5:
            angle = random.uniform(-20.0, 20.0)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            lbl = TF.rotate(
                lbl.unsqueeze(0), angle,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=self.IGNORE_IDX,
            ).squeeze(0)

        if random.random() > 0.5:
            brightness = random.uniform(0.85, 1.15)
            img = (img * brightness).clamp(0, 255)

        return img, lbl


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)
