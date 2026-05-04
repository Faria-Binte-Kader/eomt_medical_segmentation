"""
Lightning module for ViT-Adapter + Mask2Former with DINOv2 backbone.

Uses the same MaskClassificationLoss as EoMT for a fair training comparison.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from architectures.vit_adapter_mask2former import ViTAdapterMask2Former
from training.mask_classification_loss import MaskClassificationLoss
from runners.dice_metric import IGNORE_IDX
from runners.optim_utils import build_backbone_llrd_groups

class ViTAdapterM2FModule(L.LightningModule):
    """
    ViT-Adapter + Mask2Former Lightning module.

    Training: multi-layer Hungarian-matched loss (BCE + Dice + CE).
    Validation: per-class threshold (score > 0.5), global DICE and IoU.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        img_size: tuple = (512, 512),
        num_classes: int = 1,
        num_queries: int = 100,
        num_decoder_layers: int = 9,
        adapter_interval: int = 6,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        llrd_decay: float = 0.8,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        max_steps: int = 10_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.network = ViTAdapterMask2Former(
            backbone_name=backbone_name,
            num_classes=num_classes,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            adapter_interval=adapter_interval,
            img_size=img_size,
        )

        self.criterion = MaskClassificationLoss(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=num_classes,
            no_object_coefficient=0.1,
        )

        self.num_classes     = num_classes
        self._val_dice_store = None
        self._val_iou_store  = None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _reset_stores(self):
        self._val_dice_store = [{"tp": 0.0, "pred": 0.0, "tgt": 0.0} for _ in range(self.num_classes)]
        self._val_iou_store  = [{"inter": 0.0, "union": 0.0}          for _ in range(self.num_classes)]

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, imgs: torch.Tensor) -> tuple:
        return self.network(imgs / 255.0)

    # ── training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        mask_logits_list, class_logits_list = self(imgs)

        n = len(mask_logits_list)
        all_losses = {}

        for i, (ml, cl) in enumerate(zip(mask_logits_list, class_logits_list)):
            suffix = f"_layer_{i - n + 1}"
            losses = self.criterion(
                masks_queries_logits=ml,
                class_queries_logits=cl,
                targets=targets,
            )
            for k, v in losses.items():
                all_losses[k + suffix] = v

        loss = self.criterion.loss_total(all_losses, self.log)
        if not torch.isfinite(loss):
            print(f"[WARNING] Non-finite loss at step {self.global_step} — skipping batch")
            # The forward pass already updated BatchNorm running stats with NaN inputs.
            # Reset them now so the next batch normalises from a clean state.
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    # ── validation ───────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._reset_stores()

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        img_size = (imgs.shape[2], imgs.shape[3])

        with torch.no_grad():
            mask_logits_list, class_logits_list = self(imgs)

        # Use final decoder layer output
        ml, cl = mask_logits_list[-1], class_logits_list[-1]
        logits = self._to_per_pixel_logits(ml, cl, img_size)  # [B, C, H, W]

        pred = logits > 0.5  # [B, C, H, W] bool

        for j, target in enumerate(targets):
            seg   = target["seg_map"].to(self.device)   # [H, W]
            valid = seg != IGNORE_IDX
            for c in range(self.num_classes):
                pc = pred[j, c].float()
                sc = (seg == c).float()
                self._val_dice_store[c]["tp"]   += (pc * sc).sum().item()
                self._val_dice_store[c]["pred"] += pc.sum().item()
                self._val_dice_store[c]["tgt"]  += sc.sum().item()
                pi = pred[j, c] & valid
                ti = (seg == c) & valid
                self._val_iou_store[c]["inter"] += (pi & ti).float().sum().item()
                self._val_iou_store[c]["union"] += (pi | ti).float().sum().item()

    def on_validation_epoch_end(self):
        smooth      = 1e-6
        class_names = getattr(self, "class_names", [f"class{c}" for c in range(self.num_classes)])
        dice_vals, iou_vals = [], []

        for c in range(self.num_classes):
            name = class_names[c] if c < len(class_names) else f"class{c}"
            tp   = self._val_dice_store[c]["tp"]
            pred = self._val_dice_store[c]["pred"]
            tgt  = self._val_dice_store[c]["tgt"]
            dice = (2.0 * tp + smooth) / (pred + tgt + smooth)
            dice_vals.append(dice)
            self.log(f"val/dice_{name}", dice, prog_bar=(self.num_classes == 1), sync_dist=True)

            inter = self._val_iou_store[c]["inter"]
            union = self._val_iou_store[c]["union"]
            iou   = (inter + smooth) / (union + smooth)
            iou_vals.append(iou)
            self.log(f"val/iou_{name}", iou, sync_dist=True)

        dice_mean = sum(dice_vals) / len(dice_vals)
        iou_mean  = sum(iou_vals)  / len(iou_vals)
        self.log("val/dice_mean", dice_mean, prog_bar=True,  sync_dist=True)
        self.log("val/miou",      iou_mean,  prog_bar=True,  sync_dist=True)

        parts = "  |  ".join(f"DICE {class_names[c] if c < len(class_names) else c}: {dice_vals[c]:.4f}"
                              for c in range(self.num_classes))
        print(f"\n[ViT-Adapter+M2F]  {parts}  |  mean: {dice_mean:.4f}")

    # ── per-pixel logits helper ───────────────────────────────────────────────

    @staticmethod
    def _to_per_pixel_logits(
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        img_size: tuple,
    ) -> torch.Tensor:
        """Combines mask and class logits into [B, C, H, W] per-pixel scores."""
        ml = F.interpolate(mask_logits, img_size, mode="bilinear", align_corners=False)
        cl_soft = class_logits.softmax(dim=-1)[..., :-1]    # [B, Q, C]  softmax then drop no-object
        ml_sig  = ml.sigmoid()                              # [B, Q, H, W]
        return torch.einsum("bqc,bqhw->bchw", cl_soft, ml_sig)

    # ── optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        backbone_groups, bb_ids = build_backbone_llrd_groups(
            self.network.backbone,
            self.hparams.backbone_lr,
            self.hparams.llrd_decay,
        )
        head_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in bb_ids
        ]
        optimizer = torch.optim.AdamW(
            backbone_groups + [{"params": head_params, "lr": self.hparams.lr}],
            weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.warmup_steps
        total  = self.hparams.max_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
