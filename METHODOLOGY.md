# Methodology and Evaluation

## 1. Dataset

### MSD Lung (Task06)

The Medical Segmentation Decathlon (MSD) Lung dataset contains 3D chest CT volumes from The Cancer Imaging Archive. Each volume has two labels: background (0) and lung cancer (1). The task is **binary segmentation**: identify cancerous lesions within the lung.

**Key statistics:**
- 63 labelled training volumes (3D), with 20% held out for validation (~50 train / ~13 val)
- No labelled test set (32 test volumes exist but have no labels)
- Cancer prevalence: small fraction of slices per volume contain visible tumour
- Class imbalance is severe — cancer occupies a tiny fraction of each CT volume

---

## 2. Data Preprocessing

### 2.1 CT Windowing

CT images store Hounsfield Unit (HU) values ranging from approximately −1000 (air) to +3000 (dense bone). Lung tissue and pulmonary nodules occupy a specific band of this range. We apply a **lung window** before normalisation:

```
Window centre (L) = −600 HU
Window width  (W) = 1500 HU
Effective range   = [L − W/2, L + W/2] = [−1350, 150] HU
```

Pixels outside this range are clipped, then the window is linearly mapped to `[0, 255]` as `uint8`. This suppresses dense bone and maximally differentiates air-filled alveoli (~−700 HU → ~110 in windowed), soft tissue lung cancer (~40 HU → ~236 in windowed), and adjacent structures.

### 2.2 Multi-Slice Pseudo-RGB

For each target slice `z`, the three input channels are stacked from neighbouring axial slices:

```
Channel 0 = slice z−1  (previous slice)
Channel 1 = slice z    (current slice)
Channel 2 = slice z+1  (next slice)
```

At volume boundaries, the edge slice is replicated (e.g., slice 0 uses `[0, 0, 1]`). This gives the pretrained RGB backbones (DINOv2 ViT, Swin) their expected 3-channel input while providing spatial context across the z-axis — useful for detecting small lesions that may be visible in adjacent slices.

This is preferable to replicating a single slice across all three channels (`[z, z, z]`), which would waste channel capacity and provide no cross-slice context.

### 2.3 Label Remapping

Original MSD label values are remapped to a 0-indexed foreground convention:

| MSD label | Tissue | Remapped value |
|---|---|---|
| 0 | Background | **255** (ignore index) |
| 1 | Cancer | **0** |

Setting background to 255 allows the loss and metrics to exclude it via `ignore_index=255` without treating it as a foreground class. The model is never directly supervised to predict "background" — background is implicitly the "not cancer" state, captured by the no-object class in the Hungarian matcher.

### 2.4 Slice Filtering

Axial slices are retained if they satisfy either of the following criteria:

1. **Cancer present**: at least one pixel has label = 1 (cancer). All tumour-containing slices are always kept.
2. **Lung content present**: more than 5% of pixels in the windowed image have a value > 100. This corresponds to pixels at HU > approximately −700, capturing lung parenchyma and denser structures. Pure-air slices at the superior/inferior extremes of the scan are excluded.

This balances positive samples (cancer slices) with negative samples (lung-visible, cancer-free slices) for training.

### 2.5 Augmentation (Training Only)

| Transform | Probability | Details |
|---|---|---|
| Random horizontal flip | 50% | Applied jointly to image and label |
| Random vertical flip | 50% | Applied jointly to image and label |
| Random rotation | 50% | ±20°; label uses `NEAREST` interpolation with `fill=255` |
| Brightness jitter | 50% | Scale factor in [0.85, 1.15], clamped to [0, 255] |

Labels are always interpolated with `NEAREST` mode. Rotated border pixels are filled with 255 (ignore index) so the loss does not supervise on out-of-bounds regions.

---

## 3. Model Architectures

All three models treat lung cancer detection as **binary semantic segmentation**: each pixel is assigned either cancer or background, with background pixels excluded from the loss and metrics.

### 3.1 EoMT — Encoder-only Mask Transformer

EoMT (Cheng et al., CVPR 2025) removes the separate decoder entirely. Learnable segmentation queries are **prepended** to the patch token sequence and processed by the last `num_blocks=4` ViT blocks alongside image patches.

**Architecture summary:**
- Backbone: DINOv2 ViT-B/14 with 4 register tokens (`vit_base_patch14_reg4_dinov2.lvd142m`, ~86 M params)
- `num_q = 20` learnable query embeddings injected at the start of the last 4 blocks
- Mask head: 3-layer MLP producing per-query mask embeddings → einsum with upscaled patch features → `[B, Q, H', W']` mask logits
- Class head: linear layer → `[B, Q, 2]` logits (1 cancer class + 1 no-object)
- Upscaling: `num_upscale = 2` transposed-conv blocks (×2 each) to recover spatial resolution
- **Masked attention** during training: queries only attend to patch tokens within their predicted mask region, annealed to full attention at inference

Per-layer predictions are produced at all `num_blocks+1` intermediate states, enabling deep auxiliary supervision.

**To switch to ViT-Large (paper variant):**
```bash
--backbone_name vit_large_patch14_reg4_dinov2.lvd142m  # ~307 M params
```

### 3.2 ViT-Adapter + Mask2Former

This is the primary baseline from the EoMT paper. It couples a DINOv2 ViT backbone with:

1. **Spatial Prior Module (SPM):** a lightweight CNN stem producing multi-scale feature maps at strides 4, 8, 16, 32. Each scale is projected to `embed_dim` via a 1×1 convolution + LayerNorm.

2. **Adapter layers (Injector + Extractor):** inserted every `adapter_interval=6` ViT blocks.
   - *Injector*: patch tokens cross-attend to SPM features, importing CNN-derived spatial context.
   - *Extractor*: SPM features cross-attend to patch tokens, importing ViT-derived semantic context.
   - Standard multi-head cross-attention with `gamma`-scaled residuals (gamma initialised to 0).
   - The original paper uses multi-scale deformable attention (MSDA); we use standard cross-attention because MSDA requires custom CUDA operators, and the EoMT paper's Table 1 ablation shows the adapter contributes only 0.4 PQ with DINOv2, making this a minor approximation.

3. **FPN Pixel Decoder:** top-down feature pyramid → `mask_features [B, 256, H/4, W/4]` and three multi-scale maps `[p32, p16, p8]`.

4. **Mask2Former Transformer Decoder:** 9 layers cycling over `p32 → p16 → p8 → repeat`. Each layer: masked cross-attention → self-attention → FFN (pre-norm). Per-layer predictions enable auxiliary loss at all 9 decoder steps.

**Backbone:** DINOv2 ViT-B/14 reg4 (~86 M), same as EoMT for a fair comparison.

### 3.3 Mask2Former (Swin-Base)

The HuggingFace `Mask2FormerForUniversalSegmentation` pretrained on ADE20k, with the class head replaced to output 1 foreground class (`num_labels=1, ignore_mismatched_sizes=True`). Backbone: Swin-Base (~88 M). This model is used as the standard non-DINOv2 baseline to isolate the effect of the backbone.

Training uses the model's internal loss (Hungarian matching applied inside HF's forward pass).

---

## 4. Training

### 4.1 Loss Function

EoMT and ViT-Adapter+Mask2Former use `MaskClassificationLoss`, which wraps HuggingFace's `Mask2FormerLoss` with explicit coefficient control:

| Loss component | Weight |
|---|---|
| Binary cross-entropy per matched mask | λ_bce = 5.0 |
| Dice loss per matched mask | λ_dice = 5.0 |
| Cross-entropy over class predictions | λ_ce = 2.0 |

The **Hungarian matcher** finds the optimal bijection between predicted queries and ground-truth cancer masks. Unmatched queries are assigned to the no-object class.

For **multi-layer supervision**, losses are computed at every intermediate decoder state and summed. EoMT computes losses at all `num_blocks+1 = 5` intermediate predictions; ViT-Adapter+Mask2Former computes them at all 9 decoder layers.

Mask2Former (Swin-Base) uses HuggingFace's internal loss with the same matching-based formulation.

### 4.2 Optimiser and Learning Rate Schedule

All models use **AdamW** with:
- Weight decay: 0.05
- Head / decoder learning rate: 1×10⁻⁴
- Backbone learning rate: 1×10⁻⁵ (10× lower for pretrained backbone)
- Gradient clipping: max norm = 0.01 (EoMT/ViT-Adapter)

**Learning rate schedule:** linear warmup for `warmup_steps=500` steps, followed by cosine decay to 0 over the remaining training steps:

```
lr(step) = step / warmup_steps                     if step < warmup_steps
         = 0.5 * (1 + cos(π * progress))           otherwise
```

where `progress = (step − warmup_steps) / (max_steps − warmup_steps)`.

### 4.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 512 × 512 |
| Batch size | 2 per GPU |
| Max epochs | 50 (early stopping with patience 10) |
| Mixed precision | 16-bit (`bfloat16`) |
| Checkpoint criterion | best `val/dice_mean` (= cancer DICE) |
| Top-k checkpoints saved | 3 + last |
| Early stopping | halts if `val/dice_mean` stagnates for 10 epochs |

### 4.4 Multi-GPU Training

Training supports data-parallel scaling via PyTorch DDP (controlled by `--devices N`). The per-GPU batch size is kept fixed at 2, so the **effective batch size scales linearly**: 2 GPUs → batch 4, 4 GPUs → batch 8.

The LR scheduler's `max_steps` is computed as:

```
max_steps = (len(train_dataloader) × max_epochs) // num_gpus
```

This gives the per-process step count, which is what the per-GPU optimizer scheduler should track. Warmup and cosine decay therefore remain proportionally correct regardless of the number of GPUs.

All validation metrics use `sync_dist=True`, which all-reduces TP/pred/tgt accumulators across GPUs before computing the final DICE and IoU, ensuring correct global metrics in multi-GPU runs.

---

## 5. Evaluation

### 5.1 Inference

**EoMT:** Masked attention is disabled at inference (`masked_attn_enabled = False`) for a single clean forward pass. The final block's outputs are used.

**ViT-Adapter+Mask2Former:** Final decoder layer output.

**Mask2Former:** Standard single forward pass.

Per-pixel cancer scores are obtained by combining mask and class logits:

```
score[b, h, w] = Σ_q  sigmoid(mask[b, q, h, w]) × P(cancer | query_q)
```

where `P(cancer | query_q) = softmax(class[b, q])[0]` (the no-object logit is excluded from the softmax normalisation).

The final prediction is:
```
pred[b, h, w] = (score[b, h, w] > 0.5)   →  True = cancer, False = background
```

Note: argmax is not used here because with a single foreground class the output is `[B, 1, H, W]`, and argmax on a 1-channel tensor always returns 0. Thresholding at 0.5 gives proper binary predictions.

### 5.2 DICE Score

We use **global DICE** (macro volumetric DICE), which is the standard metric for MSD challenge submissions:

```
DICE = (2 × TP + ε) / (|pred_cancer| + |gt_cancer| + ε)
```

where TP, pred, and gt counts are accumulated across **all validation slices** before the final division. `ε = 1e-6` prevents division by zero.

This differs from **per-slice DICE** (compute per image then average). Per-slice DICE is misleading for cancer: most lung slices contain no cancer, and `binary_dice(pred, empty_target)` returns ≈ 1 when the model correctly predicts nothing — artificially inflating the average. Global DICE treats all validation pixels uniformly.

### 5.3 IoU

Intersection-over-Union for the cancer class:

```
IoU = (TP + ε) / (TP + FP + FN + ε)
```

Background pixels (label = 255) are excluded from all IoU computations. IoU is accumulated globally across validation slices (same rationale as global DICE).

### 5.4 Running Evaluation

During training, `val/dice_mean` (= cancer DICE) and `val/miou` (= cancer IoU) are logged at every epoch end. To produce the final comparison table:

```bash
python evaluate.py \
    --data_dir ./data/lung \
    --eomt_ckpt       ./checkpoints/eomt/ckpts/last.ckpt \
    --vit_adapter_ckpt ./checkpoints/vit_adapter_m2f/ckpts/last.ckpt \
    --m2f_ckpt        ./checkpoints/mask2former/ckpts/last.ckpt \
    --output_json eval_results.json
```

The script runs each model independently on the same validation DataLoader, accumulates global TP/pred/tgt, and prints:

```
                      EoMT   ViT-Adapter+M2F   Mask2Former
────────────────────────────────────────────────────────────
dice_cancer         0.XXXX            0.XXXX        0.XXXX
iou_cancer          0.XXXX            0.XXXX        0.XXXX
dice_mean           0.XXXX            0.XXXX        0.XXXX
miou                0.XXXX            0.XXXX        0.XXXX
────────────────────────────────────────────────────────────
```

### 5.5 Expected Performance Context

The models are fine-tuned from strong ImageNet and scene-segmentation priors and adapted to a medical domain with limited data. Typical performance on MSD Lung Task06 with similar methods:

| Metric | Expected range |
|---|---|
| Cancer DICE | 0.40 – 0.65 (high variance; small, irregular lesions) |
| Cancer IoU | 0.30 – 0.50 |

Lung cancer segmentation is substantially harder than liver segmentation because tumours are small, irregularly shaped, and occupy a tiny fraction of the image. Results vary significantly with training set size, augmentation, and model capacity.

EoMT is expected to be competitive with or outperform ViT-Adapter+Mask2Former with significantly fewer FLOPs, which is the central claim of the paper. Mask2Former (Swin-Base) serves as the non-DINOv2 baseline.

---

## 6. Differences from the EoMT Paper Setup

| Aspect | EoMT Paper | This Project |
|---|---|---|
| Task | ADE20k / COCO panoptic (natural images) | MSD Lung Task06 (medical CT) |
| Backbone size | ViT-Large (default) | ViT-Base (default; Large available) |
| Input resolution | 640 / 896 px | 512 × 512 px |
| Training data | Millions of labelled images | ~50 CT volumes → thousands of 2D slices |
| Evaluation metric | PQ / mIoU | Global DICE + IoU |
| ViT-Adapter attention | Multi-scale deformable attention | Standard multi-head cross-attention |
| Number of queries | 200 (EoMT-L) | 20 (EoMT) / 100 (ViT-Adapter+M2F) |
| Input channels | Single-slice pseudo-RGB [z,z,z] | Multi-slice [z-1, z, z+1] |
| Number of classes | Many (panoptic) | 1 (cancer only) |
