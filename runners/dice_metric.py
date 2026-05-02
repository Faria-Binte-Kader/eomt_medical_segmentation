"""
DICE score utilities for binary and multi-class segmentation.

All functions operate on CPU or GPU tensors.
"""

import torch


CLASS_NAMES = {0: "cancer"}
IGNORE_IDX  = 255


def binary_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """
    Compute binary DICE between two boolean (or 0/1) tensors.

    Args:
        pred   : [H, W] bool or 0/1 tensor – predicted foreground
        target : [H, W] bool or 0/1 tensor – ground-truth foreground
        smooth : Laplace smoothing to avoid division by zero

    Returns:
        DICE in [0, 1] as a Python float
    """
    pred   = pred.bool().float()
    target = target.bool().float()
    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def compute_slice_dice(
    pred_map: torch.Tensor,
    seg_map: torch.Tensor,
    num_classes: int = 2,
    ignore_idx: int = IGNORE_IDX,
) -> dict[str, float]:
    """
    Compute per-class DICE for a single 2-D slice.

    Args:
        pred_map : [H, W] long tensor – predicted class indices (argmax output)
        seg_map  : [H, W] long tensor – ground-truth class indices (255 = ignore)
        num_classes : number of foreground classes (not counting ignore)

    Returns:
        dict with "dice_<name>" for each class and "dice_mean"
    """
    scores: dict[str, float] = {}
    for cls_idx in range(num_classes):
        name = CLASS_NAMES.get(cls_idx, f"class_{cls_idx}")
        pred_mask   = (pred_map == cls_idx)
        target_mask = (seg_map  == cls_idx)
        scores[f"dice_{name}"] = binary_dice(pred_mask, target_mask)

    scores["dice_mean"] = sum(scores.values()) / len(scores)
    return scores


def accumulate_dice(
    dice_store: dict[str, list],
    pred_map: torch.Tensor,
    seg_map: torch.Tensor,
    num_classes: int = 2,
) -> None:
    """In-place: append per-class DICE values into dice_store lists."""
    for cls_idx in range(num_classes):
        name = CLASS_NAMES.get(cls_idx, f"class_{cls_idx}")
        pred_mask   = (pred_map == cls_idx)
        target_mask = (seg_map  == cls_idx)
        dice_store[name].append(binary_dice(pred_mask, target_mask))


def summarise_dice(dice_store: dict[str, list]) -> dict[str, float]:
    """Average accumulated per-class DICE values; add 'dice_mean'."""
    summary = {}
    for name, scores in dice_store.items():
        summary[f"dice_{name}"] = sum(scores) / len(scores) if scores else 0.0
    if summary:
        summary["dice_mean"] = sum(summary.values()) / len(summary)
    return summary
