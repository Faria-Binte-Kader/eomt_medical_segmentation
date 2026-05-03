"""
Unified training script for EoMT and Mask2Former on MSD Lung (Task06).

Usage examples:
    # EoMT with DINOv2-Base backbone
    python train.py --model eomt \\
        --data_dir ./data/lung \\
        --output_dir ./checkpoints/eomt \\
        --img_size 512 --batch_size 2 --devices 1

    # Mask2Former with Swin-Base backbone
    python train.py --model mask2former \\
        --data_dir ./data/lung \\
        --output_dir ./checkpoints/mask2former \\
        --img_size 512 --batch_size 2 --devices 1

    # Multi-GPU (4 GPUs, effective batch = 8)
    python train.py --model eomt --batch_size 2 --devices 4

After training, evaluate with evaluate.py.
"""

import argparse
import os
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from data.msd_lung import MSDLungDataset, collate_fn


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train EoMT or Mask2Former on MSD Lung")

    p.add_argument("--model", choices=["eomt", "mask2former", "vit_adapter_m2f"], required=True)
    p.add_argument("--data_dir",   required=True, help="Path to processed data/ dir")
    p.add_argument("--output_dir", default="./checkpoints", help="Where to save checkpoints")

    # Data
    p.add_argument("--img_size",   type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    p.add_argument("--num_workers",type=int, default=4)

    # Training schedule
    p.add_argument("--max_epochs",   type=int, default=50)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--backbone_lr",  type=float, default=1e-5)
    p.add_argument("--llrd_decay",   type=float, default=0.8,
                   help="Layer-wise LR decay per ViT block (paper=0.8; 1.0=disabled)")
    p.add_argument("--weight_decay",       type=float, default=0.05)
    p.add_argument("--warmup_steps",       type=int,   default=500,
                   help="Head warmup steps (backbone frozen during this phase)")
    p.add_argument("--vit_warmup_steps",   type=int,   default=500,
                   help="Backbone warmup steps after head warmup (EoMT)")
    p.add_argument("--poly_power",         type=float, default=0.9,
                   help="Polynomial decay power for LR schedule (paper=0.9)")
    p.add_argument("--no_mask_annealing",  action="store_true",
                   help="Disable attention-mask annealing (EoMT)")
    p.add_argument("--val_check_interval", type=float, default=1.0,
                   help="Fraction of training epoch between validations (1.0=every epoch)")

    # Hardware
    p.add_argument("--devices",    type=int, default=1,
                   help="Number of GPUs to use (1 = single-GPU, 2+ = DDP multi-GPU)")
    p.add_argument("--precision",  default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--compile",    action="store_true", help="torch.compile the model")

    # EoMT-specific
    p.add_argument("--backbone_name",
                   default="vit_base_patch14_reg4_dinov2.lvd142m",
                   help="timm backbone name for EoMT")
    p.add_argument("--num_q",      type=int, default=20,
                   help="Number of segmentation queries (EoMT)")
    p.add_argument("--num_blocks", type=int, default=4,
                   help="Number of last ViT blocks to inject queries (EoMT)")
    p.add_argument("--eomt_ckpt",  default=None,
                   help="Optional pretrained EoMT checkpoint (.bin / .pt)")

    # Mask2Former-specific
    p.add_argument("--m2f_model",
                   default="facebook/mask2former-swin-base-ade-semantic",
                   help="HuggingFace model ID for Mask2Former")

    # ViT-Adapter + Mask2Former specific
    p.add_argument("--num_queries",        type=int, default=100,
                   help="Number of mask queries (vit_adapter_m2f)")
    p.add_argument("--num_decoder_layers", type=int, default=9,
                   help="Transformer decoder depth (vit_adapter_m2f)")
    p.add_argument("--adapter_interval",   type=int, default=6,
                   help="Inject/extract every N ViT blocks (vit_adapter_m2f)")

    # Logging
    p.add_argument("--wandb",    action="store_true", help="Enable W&B logging")
    p.add_argument("--run_name", default=None)

    return p.parse_args()


# ── main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    img_size = (args.img_size, args.img_size)
    out_dir  = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = MSDLungDataset(args.data_dir, split="train", img_size=img_size, augment=True)
    val_ds   = MSDLungDataset(args.data_dir, split="val",   img_size=img_size, augment=False)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )

    # ── Estimate max_steps ───────────────────────────────────────────────
    steps_per_epoch = len(train_dl)
    max_steps       = steps_per_epoch * args.max_epochs // args.devices

    # ── Model ────────────────────────────────────────────────────────────
    if args.model == "eomt":
        from runners.eomt_runner import EoMTMedicalModule
        module = EoMTMedicalModule(
            backbone_name=args.backbone_name,
            img_size=img_size,
            num_q=args.num_q,
            num_blocks=args.num_blocks,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            llrd_decay=args.llrd_decay,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            vit_warmup_steps=args.vit_warmup_steps,
            max_steps=max_steps,
            poly_power=args.poly_power,
            attn_mask_annealing_enabled=not args.no_mask_annealing,
        )
        if args.eomt_ckpt:
            ckpt = torch.load(args.eomt_ckpt, map_location="cpu", weights_only=True)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            module.load_state_dict(ckpt, strict=False)
            print(f"[EoMT] Loaded checkpoint: {args.eomt_ckpt}")
    elif args.model == "mask2former":
        from runners.mask2former_runner import Mask2FormerMedicalModule
        module = Mask2FormerMedicalModule(
            model_name=args.m2f_model,
            img_size=img_size,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=max_steps,
        )
    else:  # vit_adapter_m2f
        from runners.vit_adapter_m2f_runner import ViTAdapterM2FModule
        module = ViTAdapterM2FModule(
            backbone_name=args.backbone_name,
            img_size=img_size,
            num_queries=args.num_queries,
            num_decoder_layers=args.num_decoder_layers,
            adapter_interval=args.adapter_interval,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            llrd_decay=args.llrd_decay,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=max_steps,
        )

    if args.compile:
        module = torch.compile(module)

    # ── Logger ───────────────────────────────────────────────────────────
    run_name = args.run_name or f"{args.model}_lung"
    if args.wandb:
        logger = WandbLogger(project="medical-seg", name=run_name)
    else:
        logger = CSVLogger(save_dir=str(out_dir), name=run_name)

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "ckpts"),
        filename=f"{args.model}_{{epoch:02d}}_dice{{val/dice_mean:.4f}}",
        monitor="val/dice_mean",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    early_stop_cb = EarlyStopping(
        monitor="val/dice_mean",
        mode="max",
        patience=10,
        verbose=True,
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb, early_stop_cb],
        gradient_clip_val=0.01,
        gradient_clip_algorithm="norm",
        val_check_interval=args.val_check_interval,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best DICE      : {ckpt_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
