"""
Mask2Former (Swin backbone) fine-tuning Lightning module for MSD Lung segmentation.

Uses HuggingFace Transformers Mask2FormerForUniversalSegmentation.
The class head is replaced to support num_labels=1 (cancer only).

Label convention (0-indexed):
    0 = cancer   |   255 = background / ignore
"""

import math

import lightning
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import Mask2FormerForUniversalSegmentation

from runners.dice_metric import IGNORE_IDX

# ImageNet normalisation expected by the Swin backbone
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


class Mask2FormerMedicalModule(lightning.LightningModule):
    """
    Fine-tune a pretrained Mask2Former model (Swin-Base/Large backbone)
    on lung cancer segmentation.

    Architecture
    ────────────
    Swin encoder  →  FPN pixel decoder  →  Transformer decoder (cross-attention)
    →  class + mask prediction heads  →  [B, 1, H, W] per-pixel cancer scores

    Training
    ────────
    mask_labels / class_labels passed directly to the model; loss computed
    internally via Hungarian matching.

    Inference
    ─────────
    Score > 0.5 on the single cancer channel → cancer predicted.
    """

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-base-ade-semantic",
        num_classes: int = 1,               # cancer only
        img_size: tuple[int, int] = (512, 512),
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        max_steps: int = 10_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        self.img_size    = img_size
        self.num_classes = num_classes
        self._val_dice_store = None
        self._val_iou_store  = None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _reset_stores(self):
        self._val_dice_store = {"tp": 0.0, "pred": 0.0, "tgt": 0.0}
        self._val_iou_store  = {"inter": 0.0, "union": 0.0}

    def _normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert [0, 255] float tensors to ImageNet-normalised [B,3,H,W]."""
        mean = _IMAGENET_MEAN.to(imgs.device, imgs.dtype)
        std  = _IMAGENET_STD .to(imgs.device, imgs.dtype)
        return (imgs / 255.0 - mean) / std

    @staticmethod
    def _prepare_labels(targets: list[dict]):
        """
        Extract mask_labels and class_labels lists expected by HF Mask2Former.

        mask_labels  : list of FloatTensor  [N_i, H, W]
        class_labels : list of LongTensor   [N_i]
        """
        mask_labels  = [t["masks"].float() for t in targets]
        class_labels = [t["labels"].long() for t in targets]
        return mask_labels, class_labels

    @staticmethod
    def _semseg_from_outputs(outputs, img_size: tuple[int, int]) -> torch.Tensor:
        """
        Combine mask / class logits from HF Mask2Former into [B, C, H, W].
        With num_labels=1, C=1; threshold channel 0 at 0.5 for cancer prediction.
        """
        masks_q = F.interpolate(
            outputs.masks_queries_logits, img_size,
            mode="bilinear", align_corners=False,
        )
        class_q = outputs.class_queries_logits
        return torch.einsum(
            "bqhw, bqc -> bchw",
            masks_q.sigmoid(),
            class_q.softmax(dim=-1)[..., :-1],   # drop no-object
        )

    # ── forward / training ──────────────────────────────────────────────────

    def forward(self, imgs: torch.Tensor, mask_labels=None, class_labels=None):
        pixel_values = self._normalize(imgs)
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        mask_labels, class_labels = self._prepare_labels(targets)
        outputs = self(imgs, mask_labels=mask_labels, class_labels=class_labels)
        loss = outputs.loss
        self.log("losses/train_total", loss, prog_bar=True, sync_dist=True)
        return loss

    # ── validation ──────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._reset_stores()

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        with torch.no_grad():
            outputs = self(imgs)

        logits = self._semseg_from_outputs(outputs, self.img_size)  # [B, 1, H, W]
        pred_cancer = logits[:, 0, ...] > 0.5                       # [B, H, W] bool

        for j in range(len(imgs)):
            seg = targets[j]["seg_map"].to(pred_cancer.device)   # [H, W]: 0=cancer, 255=ignore
            pc = pred_cancer[j].float()
            sc = (seg == 0).float()

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

        print(f"\n[Mask2Former]  DICE cancer: {dice:.4f}  |  IoU cancer: {iou:.4f}")

    # ── optimiser ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        try:
            backbone_params = list(
                self.model.model.pixel_level_module.encoder.parameters()
            )
            backbone_ids = {id(p) for p in backbone_params}
            other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        except AttributeError:
            backbone_params = []
            other_params = list(self.model.parameters())

        param_groups = [{"params": other_params, "lr": self.hparams.lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.hparams.backbone_lr})

        optimizer = AdamW(param_groups, weight_decay=self.hparams.weight_decay)

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
