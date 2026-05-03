"""
ViT-Adapter + Mask2Former with DINOv2 backbone.

Architecture matches the baseline used in the EoMT paper (Table 1):
  - DINOv2 ViT-Base or ViT-Large backbone (timm)
  - Spatial Prior Module (CNN) generates multi-scale features at strides 4/8/16/32
  - Adapter layers (Inject + Extract) every `adapter_interval` ViT blocks
  - FPN pixel decoder merges ViT + CNN features into mask features + multi-scale
  - Mask2Former transformer decoder (9 layers) with masked cross-attention
  - Hungarian-matched BCE + Dice + CE loss (from MaskClassificationLoss)

Deformable attention is replaced by standard multi-head cross-attention, which
is equivalent for DINOv2 backbones (ablation in EoMT paper shows adapter
contributes only 0.4 PQ with DINOv2 — Table 1 row comparison).

Attribute layout (mirrors EoMT's encoder/backbone split):
  self.backbone  = timm model (embed_dim, blocks, patch_embed, norm, etc.)
  self.pixel_mean / self.pixel_std = ImageNet normalisation buffers (on self)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit import ViT


# ── Layer norm for 2-D feature maps ─────────────────────────────────────────

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ── Spatial Prior Module (CNN stem) ─────────────────────────────────────────

class SpatialPriorModule(nn.Module):
    """
    Lightweight CNN producing multi-scale feature maps at strides 4, 8, 16, 32.
    Input:  normalised image [B, 3, H, W].
    Output: dict {4: [B,D,H/4,W/4], 8: ..., 16: ..., 32: ...}
    """

    def __init__(self, embed_dim: int = 768, in_channels: int = 3):
        super().__init__()
        inner = embed_dim // 4  # 192 for ViT-B (768), 256 for ViT-L (1024)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, inner, 7, stride=4, padding=3, bias=False),
            nn.GroupNorm(32, inner), nn.GELU(),
            nn.Conv2d(inner, inner, 3, padding=1, bias=False),
            nn.GroupNorm(32, inner), nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(inner, inner, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, inner), nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inner, inner, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, inner), nn.GELU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inner, inner, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, inner), nn.GELU(),
        )

        self.proj4  = nn.Sequential(nn.Conv2d(inner, embed_dim, 1), LayerNorm2d(embed_dim))
        self.proj8  = nn.Sequential(nn.Conv2d(inner, embed_dim, 1), LayerNorm2d(embed_dim))
        self.proj16 = nn.Sequential(nn.Conv2d(inner, embed_dim, 1), LayerNorm2d(embed_dim))
        self.proj32 = nn.Sequential(nn.Conv2d(inner, embed_dim, 1), LayerNorm2d(embed_dim))

    def forward(self, x: torch.Tensor) -> dict:
        f4  = self.stem(x)
        f8  = self.down1(f4)
        f16 = self.down2(f8)
        f32 = self.down3(f16)
        return {
            4:  self.proj4(f4),
            8:  self.proj8(f8),
            16: self.proj16(f16),
            32: self.proj32(f32),
        }


# ── Injector: CNN features → ViT tokens ──────────────────────────────────────

class Injector(nn.Module):
    """ViT patch tokens (Q) attend to flattened CNN multi-scale features (K/V)."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gamma   = nn.Parameter(torch.zeros(1))

    def forward(self, vit_patches: torch.Tensor, spm_flat: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(self.norm_q(vit_patches), self.norm_kv(spm_flat), self.norm_kv(spm_flat))
        return vit_patches + self.gamma * out


# ── Extractor: ViT tokens → CNN features ─────────────────────────────────────

class Extractor(nn.Module):
    """Flattened CNN features (Q) attend to ViT patch tokens (K/V)."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gamma   = nn.Parameter(torch.zeros(1))

    def forward(self, spm_flat: torch.Tensor, vit_patches: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(self.norm_q(spm_flat), self.norm_kv(vit_patches), self.norm_kv(vit_patches))
        return spm_flat + self.gamma * out


# ── FPN Pixel Decoder ────────────────────────────────────────────────────────

class FPNPixelDecoder(nn.Module):
    """
    Top-down FPN over {32, 16, 8, 4} strides.
    Returns (mask_features [B,D,H/4,W/4], [p32, p16, p8]) for the transformer decoder.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256):
        super().__init__()

        self.lat32 = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), LayerNorm2d(hidden))
        self.lat16 = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), LayerNorm2d(hidden))
        self.lat8  = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), LayerNorm2d(hidden))
        self.lat4  = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), LayerNorm2d(hidden))

        self.out32 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), LayerNorm2d(hidden), nn.GELU())
        self.out16 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), LayerNorm2d(hidden), nn.GELU())
        self.out8  = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), LayerNorm2d(hidden), nn.GELU())

        self.mask_proj = nn.Conv2d(hidden, hidden, 3, padding=1)

    def forward(self, feats: dict) -> tuple:
        p32 = self.out32(self.lat32(feats[32]))
        p16 = self.out16(self.lat16(feats[16]) + F.interpolate(p32, scale_factor=2, mode="nearest"))
        p8  = self.out8 (self.lat8 (feats[8])  + F.interpolate(p16, scale_factor=2, mode="nearest"))
        p4  = self.lat4(feats[4]) + F.interpolate(p8, scale_factor=2, mode="nearest")

        mask_features = self.mask_proj(p4)   # [B, hidden, H/4, W/4]
        return mask_features, [p32, p16, p8]


# ── Mask2Former Transformer Decoder ─────────────────────────────────────────

class _DecoderLayer(nn.Module):
    """
    One Mask2Former transformer decoder layer.
    Order (from Cheng et al. 2022): masked cross-attn → self-attn → FFN.
    Uses pre-norm (LayerNorm applied before each sub-layer).
    """

    def __init__(self, hidden: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden), nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,                          # [B, Q, hidden]
        kv: torch.Tensor,                          # [B, HW, hidden]  (already level-embedded)
        attn_mask: Optional[torch.Tensor] = None,  # [B*heads, Q, HW]
    ) -> torch.Tensor:
        # masked cross-attention (pre-norm on Q only; K/V from pixel decoder)
        q2, _ = self.cross_attn(self.norm1(q), kv, kv, attn_mask=attn_mask)
        q = q + q2
        # self-attention (pre-norm)
        q_n = self.norm2(q)
        q2, _ = self.self_attn(q_n, q_n, q_n)
        q = q + q2
        # FFN (pre-norm)
        q = q + self.ffn(self.norm3(q))
        return q


class Mask2FormerTransformerDecoder(nn.Module):
    """
    9-layer Mask2Former transformer decoder.
    Cycles over 3 scales (p32 → p16 → p8) with masked cross-attention.
    Produces per-layer (mask_logits, class_logits) for auxiliary loss.
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        hidden: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 9,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers  = num_layers
        self.hidden      = hidden

        self.query_feat  = nn.Embedding(num_queries, hidden)
        self.query_pos   = nn.Embedding(num_queries, hidden)
        self.level_embed = nn.Embedding(3, hidden)

        self.layers = nn.ModuleList([
            _DecoderLayer(hidden, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.class_heads = nn.ModuleList([
            nn.Linear(hidden, num_classes + 1) for _ in range(num_layers)
        ])
        self.mask_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(num_layers)
        ])

    def _compute_mask_logits(self, q: torch.Tensor, mask_features: torch.Tensor, layer_i: int) -> torch.Tensor:
        """[B,Q,D] × [B,D,H,W] → [B,Q,H,W]"""
        return torch.einsum("bqd,bdhw->bqhw", self.mask_embeds[layer_i](q), mask_features)

    def _attn_mask(self, mask_logits: torch.Tensor, kv_len: int, num_heads: int) -> torch.Tensor:
        B, Q, H, W = mask_logits.shape
        # True = block this position (sigmoid < 0.5 means mask not predicted here)
        mask = (mask_logits.detach().sigmoid() < 0.5).view(B, Q, H * W)
        # if every position is blocked for a query, unblock all (avoids NaN gradient)
        mask = mask & ~mask.all(dim=-1, keepdim=True)
        if H * W != kv_len:
            kv_h = int(math.isqrt(kv_len))
            mask = F.interpolate(
                mask.float().view(B, Q, H, W), size=(kv_h, kv_h), mode="nearest"
            ).bool().view(B, Q, kv_len)
        # [B*heads, Q, kv_len]
        return mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(B * num_heads, Q, kv_len)

    def forward(self, mask_features: torch.Tensor, multi_scale: list) -> tuple:
        """
        mask_features : [B, D, H/4, W/4]
        multi_scale   : [p32, p16, p8]  each [B, D, h, w]

        Returns lists of length num_layers:
          mask_logits_all  : [B, Q, H/4, W/4]
          class_logits_all : [B, Q, C+1]
        """
        B = mask_features.shape[0]
        num_heads = self.layers[0].cross_attn.num_heads

        q = (
            self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
            + self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        )   # [B, Q, D]

        # initial mask prediction (before any layer) for layer-0 attention mask
        prev_mask = self._compute_mask_logits(q, mask_features, layer_i=0)

        mask_logits_all, class_logits_all = [], []

        for i, layer in enumerate(self.layers):
            scale_idx = i % 3
            feat = multi_scale[scale_idx]              # [B, D, h, w]
            _, _, h_, w_ = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1) + self.level_embed.weight[scale_idx]

            attn_mask = self._attn_mask(prev_mask, h_ * w_, num_heads)

            q = layer(q, feat_flat, attn_mask=attn_mask)

            mask_logits  = self._compute_mask_logits(q, mask_features, layer_i=i)
            class_logits = self.class_heads[i](q)

            mask_logits_all.append(mask_logits)
            class_logits_all.append(class_logits)
            prev_mask = mask_logits

        return mask_logits_all, class_logits_all


# ── Full ViT-Adapter + Mask2Former model ─────────────────────────────────────

class ViTAdapterMask2Former(nn.Module):
    """
    DINOv2 ViT backbone + Spatial Prior Module + Adapter (Inject/Extract) +
    FPN pixel decoder + Mask2Former transformer decoder.

    Attribute layout (mirrors EoMT):
      self.backbone  = timm ViT model (all timm attrs: .embed_dim, .blocks, etc.)
      self.pixel_mean / self.pixel_std = ImageNet normalisation buffers

    The runner's parameter grouping uses self.backbone.parameters() for backbone_lr
    and everything else for head_lr, same as EoMT.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        num_classes: int = 2,
        num_queries: int = 100,
        hidden: int = 256,
        num_decoder_layers: int = 9,
        adapter_interval: int = 6,
        img_size: tuple = (512, 512),
    ):
        super().__init__()

        # ── backbone ────────────────────────────────────────────────────────
        # Use the ViT wrapper only to build + get pixel stats, then store the
        # inner timm model directly so attribute access matches EoMT's pattern
        # (self.backbone.embed_dim / .blocks / .patch_embed / .norm / etc.)
        _vit_wrapper = ViT(img_size=img_size, backbone_name=backbone_name)
        self.backbone = _vit_wrapper.backbone          # timm model
        self.register_buffer("pixel_mean", _vit_wrapper.pixel_mean.clone().detach())
        self.register_buffer("pixel_std",  _vit_wrapper.pixel_std.clone().detach())
        del _vit_wrapper

        embed_dim = self.backbone.embed_dim            # 768 (ViT-B) / 1024 (ViT-L)

        # ── spatial prior module ────────────────────────────────────────────
        self.spm = SpatialPriorModule(embed_dim)

        # ── adapter layers ──────────────────────────────────────────────────
        n_blocks = len(self.backbone.blocks)
        adapter_positions = list(range(adapter_interval - 1, n_blocks, adapter_interval))
        self.adapter_positions = set(adapter_positions)
        n_adapters = len(adapter_positions)

        num_heads = embed_dim // 64   # 12 for ViT-B, 16 for ViT-L
        self.injectors = nn.ModuleList([Injector(embed_dim, num_heads) for _ in range(n_adapters)])
        self.extractors = nn.ModuleList([Extractor(embed_dim, num_heads) for _ in range(n_adapters)])

        # ── pixel decoder ───────────────────────────────────────────────────
        self.pixel_decoder = FPNPixelDecoder(embed_dim, hidden)

        # ── transformer decoder ─────────────────────────────────────────────
        self.decoder = Mask2FormerTransformerDecoder(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden=hidden,
            num_heads=8,
            ffn_dim=hidden * 8,
            num_layers=num_decoder_layers,
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: [0,1] float → ImageNet-normalised."""
        return (x - self.pixel_mean.view(1, 3, 1, 1)) / self.pixel_std.view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: [B, 3, H, W]  float32 in [0, 1]  (caller divides by 255)

        Returns (mask_logits_per_layer, class_logits_per_layer):
          each a list of length num_decoder_layers.
        """
        x_norm = self._normalize(x)

        # 1. CNN multi-scale features  {4, 8, 16, 32}
        spm_feats = self.spm(x_norm)

        # 2. ViT forward with adapter injection/extraction
        bb = self.backbone                          # timm model
        n_prefix = bb.num_prefix_tokens             # 5 for DINOv2-reg4 (1 CLS + 4 regs)

        vit_x = bb.patch_embed(x_norm)
        vit_x = bb._pos_embed(vit_x)                # adds CLS + register tokens

        # flatten all SPM scales for cross-attention: [B, M, D]
        # record spatial shapes so we can unpack spm_flat back after adapters
        spm_shapes = {s: (spm_feats[s].shape[2], spm_feats[s].shape[3]) for s in (4, 8, 16, 32)}
        spm_flat = torch.cat(
            [spm_feats[s].flatten(2).permute(0, 2, 1) for s in (4, 8, 16, 32)],
            dim=1,
        )

        adapter_idx = 0
        for block_idx, block in enumerate(bb.blocks):
            vit_x = block(vit_x)

            if block_idx in self.adapter_positions:
                patch_tokens = vit_x[:, n_prefix:]   # [B, N, D]  (exclude CLS/regs)
                patch_tokens = self.injectors[adapter_idx](patch_tokens, spm_flat)
                spm_flat     = self.extractors[adapter_idx](spm_flat, patch_tokens)
                vit_x = torch.cat([vit_x[:, :n_prefix], patch_tokens], dim=1)
                adapter_idx += 1

        vit_x = bb.norm(vit_x)
        patch_tokens = vit_x[:, n_prefix:]           # [B, N_patches, D]

        # 3a. Write ViT-enriched spm_flat back to spatial feature maps so FPN
        #     sees adapter-updated features at all four scales (not just stride-16).
        B, N, D = patch_tokens.shape
        offset = 0
        for s in (4, 8, 16, 32):
            h, w = spm_shapes[s]
            n_tok = h * w
            spm_feats[s] = spm_flat[:, offset:offset + n_tok].permute(0, 2, 1).view(B, D, h, w)
            offset += n_tok

        # 3b. Additionally merge final-layer ViT patch tokens into stride-16 map
        gh, gw  = bb.patch_embed.grid_size            # e.g. (36, 36) for 512px / patch14
        f_vit   = patch_tokens.permute(0, 2, 1).view(B, D, gh, gw)

        h16, w16 = spm_shapes[16]
        spm_feats[16] = spm_feats[16] + F.interpolate(
            f_vit, size=(h16, w16), mode="bilinear", align_corners=False
        )

        # 4. Pixel decoder → mask features + multi-scale for transformer
        mask_features, multi_scale = self.pixel_decoder(spm_feats)

        # 5. Mask2Former transformer decoder
        return self.decoder(mask_features, multi_scale)
