"""Layer-wise learning rate decay (LLRD) for timm ViT backbones.

Paper reference: EoMT (CVPR 2025) uses LLRD with decay factor 0.8.

Each ViT block i (0-indexed from the input) gets:
    lr_i = base_lr × decay^(n_blocks − 1 − i)

The deepest block (most task-relevant) gets base_lr × decay^0 = base_lr.
The shallowest block gets base_lr × decay^(n−1).
All non-block backbone params (patch_embed, norm, cls_token, pos_embed, registers)
share the lowest rate: base_lr × decay^n_blocks.

With decay=1.0 this reduces exactly to the flat two-group scheme.
"""

import torch.nn as nn


def build_backbone_llrd_groups(
    backbone: nn.Module,
    base_lr: float,
    decay: float,
) -> tuple[list[dict], set[int]]:
    """
    Returns (param_groups, seen_ids).

    param_groups: list of {params, lr} dicts ready for AdamW.
    seen_ids:     set of id() for every backbone parameter, so the caller can
                  collect the remaining (head) params with a simple exclusion.
    """
    n = len(backbone.blocks)
    groups: list[dict] = []
    seen: set[int] = set()

    # Per-block groups — deeper blocks get higher lr
    for i, block in enumerate(backbone.blocks):
        params = [p for p in block.parameters() if p.requires_grad]
        groups.append({"params": params, "lr": base_lr * (decay ** (n - 1 - i))})
        seen.update(id(p) for p in params)

    # Remaining backbone params (patch_embed, norm, cls/reg tokens, pos_embed)
    other = [p for p in backbone.parameters() if p.requires_grad and id(p) not in seen]
    if other:
        groups.insert(0, {"params": other, "lr": base_lr * (decay ** n)})
        seen.update(id(p) for p in other)

    return groups, seen
