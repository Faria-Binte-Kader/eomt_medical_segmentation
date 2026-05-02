"""
EoMT fine-tuning Lightning module for MSD Lung segmentation.

Label convention (0-indexed):
    0 = cancer   |   255 = background / ignore
"""

import math

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_loss import MaskClassificationLoss
from runners.dice_metric import IGNORE_IDX

NUM_CLASSES = 1   # cancer only


class EoMTMedicalModule(lightning.LightningModule):
    """
    Fine-tune EoMT (with a DINOv2 ViT backbone) on lung cancer segmentation.

    Architecture
    ────────────
    ViT backbone  →  EoMT queries injected in the last `num_blocks` layers
    →  mask_head + class_head  →  [B, 1, H, W] per-pixel cancer scores

    Inference
    ─────────
    Score > 0.5 → cancer predicted (single-class threshold, not argmax).
    Masked attention is disabled at inference for a clean forward pass.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = NUM_CLASSES,
        num_q: int = 20,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        max_steps: int = 10_000,
        mask_coef: float = 5.0,
        dice_coef: float = 5.0,
        class_coef: float = 2.0,
        no_object_coef: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        encoder = ViT(img_size=img_size, backbone_name=backbone_name)
        self.network = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
        )

        self.criterion = MaskClassificationLoss(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=mask_coef,
            dice_coefficient=dice_coef,
            class_coefficient=class_coef,
            num_labels=num_classes,
            no_object_coefficient=no_object_coef,
        )

        self.img_size    = img_size
        self.num_classes = num_classes
        self._val_dice_store = None
        self._val_iou_store  = None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _reset_stores(self):
        self._val_dice_store = {"tp": 0.0, "pred": 0.0, "tgt": 0.0}
        self._val_iou_store  = {"inter": 0.0, "union": 0.0}

    @staticmethod
    def _to_per_pixel_logits(
        mask_logits:  torch.Tensor,
        class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """[B,Q,H,W] × [B,Q,C+1]  →  [B,C,H,W] per-pixel class scores."""
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],   # drop no-object
        )

    # ── forward / training ──────────────────────────────────────────────────

    def forward(self, imgs: torch.Tensor):
        """imgs: [B,3,H,W] float32 in [0, 255]."""
        return self.network(imgs / 255.0)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        mask_logits_per_block, class_logits_per_block = self(imgs)

        n = len(mask_logits_per_block)
        all_losses: dict[str, torch.Tensor] = {}

        for i, (ml, cl) in enumerate(zip(mask_logits_per_block, class_logits_per_block)):
            losses = self.criterion(
                masks_queries_logits=ml,
                class_queries_logits=cl,
                targets=targets,
            )
            suffix = "" if i == n - 1 else f"_block_{i - n + 1}"
            all_losses.update({f"{k}{suffix}": v for k, v in losses.items()})

        return self.criterion.loss_total(all_losses, self.log)

    # ── validation ──────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._reset_stores()

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        was_enabled = self.network.masked_attn_enabled
        self.network.masked_attn_enabled = False
        with torch.no_grad():
            mask_logits_per_block, class_logits_per_block = self(imgs)
        self.network.masked_attn_enabled = was_enabled

        # Use final block output only
        ml_up  = F.interpolate(mask_logits_per_block[-1], self.img_size,
                               mode="bilinear", align_corners=False)
        logits = self._to_per_pixel_logits(ml_up, class_logits_per_block[-1])  # [B, 1, H, W]

        # Binary cancer prediction via threshold (argmax is undefined for 1 class)
        pred_cancer = logits[:, 0, ...] > 0.5  # [B, H, W] bool

        for j, target in enumerate(targets):
            seg = target["seg_map"].to(self.device)   # [H, W]: 0=cancer, 255=ignore
            pc = pred_cancer[j].float()
            sc = (seg == 0).float()   # cancer label = 0

            self._val_dice_store["tp"]   += (pc * sc).sum().item()
            self._val_dice_store["pred"] += pc.sum().item()
            self._val_dice_store["tgt"]  += sc.sum().item()

            valid = seg != IGNORE_IDX
            pi = pred_cancer[j] & valid
            ti = (seg == 0) & valid
            self._val_iou_store["inter"] += (pi & ti).float().sum().item()
            self._val_iou_store["union"] += (pi | ti).float().sum().item()

    def on_validation_epoch_end(self):
        smooth = 1e-6

        tp   = self._val_dice_store["tp"]
        pred = self._val_dice_store["pred"]
        tgt  = self._val_dice_store["tgt"]
        dice = (2.0 * tp + smooth) / (pred + tgt + smooth)
        self.log("val/dice_cancer", dice, prog_bar=True,  sync_dist=True)
        self.log("val/dice_mean",   dice, prog_bar=False, sync_dist=True)

        inter = self._val_iou_store["inter"]
        union = self._val_iou_store["union"]
        iou = (inter + smooth) / (union + smooth)
        self.log("val/iou_cancer", iou, sync_dist=True)
        self.log("val/miou",       iou, prog_bar=True, sync_dist=True)

        print(f"\n[EoMT]  DICE cancer: {dice:.4f}  |  IoU cancer: {iou:.4f}")

    # ── optimiser ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        backbone_params = list(self.network.encoder.parameters())
        head_params = (
            list(self.network.q.parameters())
            + list(self.network.class_head.parameters())
            + list(self.network.mask_head.parameters())
            + list(self.network.upscale.parameters())
        )

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": head_params,     "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        ws  = self.hparams.warmup_steps
        tot = self.hparams.max_steps

        def lr_lambda(step: int) -> float:
            if step < ws:
                return step / max(ws, 1)
            progress = (step - ws) / max(tot - ws, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
