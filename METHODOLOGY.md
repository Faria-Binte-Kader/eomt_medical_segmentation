# Methodology and Evaluation

## 1. Research Goal

The EoMT paper (CVPR 2025) introduces an encoder-only mask transformer that injects learnable queries directly into the last few ViT blocks, eliminating the need for a separate decoder. On natural-image benchmarks (ADE20k, COCO), EoMT matches or exceeds ViT-Adapter+Mask2Former with fewer parameters and FLOPs.

**This project tests whether that advantage holds under domain shift.** Both models are evaluated on chest CT and abdominal CT — fundamentally different input distributions from the natural images both backbones were pretrained on. Two datasets are used deliberately:

- **MSD Lung (Task06):** single-class binary segmentation — detect small, irregular lung cancer lesions. Severe class imbalance, limited data.
- **MSD HepaticVessel (Task08):** two-class segmentation — delineate hepatic vessels and hepatic tumours simultaneously. Vessels form a dense tree structure; tumours are irregular and often touching vessels.

Together these probe whether EoMT's in-network query mechanism generalises across CT modality shift (both datasets) and across task complexity (single-class vs multi-class). ViT-Adapter+Mask2Former's explicit multi-scale spatial prior module (SPM) was designed to compensate for the ViT's lack of inductive spatial bias; it may provide an advantage when localising fine structures in CT.

Both models are implemented as faithfully as possible to the EoMT paper and trained under identical conditions. All departures from the paper setup are documented in Section 7.

---

## 2. Datasets

### 2.1 MSD Lung (Task06)

3D chest CT volumes from The Cancer Imaging Archive. Binary labels: background and lung cancer.

| Statistic | Value |
|---|---|
| Labelled volumes | 63 (3D) |
| Train / val split | ~50 / ~13 (80/20 random) |
| Foreground classes | 1 (cancer) |
| Class imbalance | Severe — tumours occupy a tiny fraction of each volume |

### 2.2 MSD HepaticVessel (Task08)

3D abdominal CT volumes. Two foreground labels: hepatic vessels and hepatic tumours.

| Statistic | Value |
|---|---|
| Labelled volumes | 303 (3D) |
| Train / val split | ~242 / ~61 (80/20 random) |
| Foreground classes | 2 (vessel, tumour) |
| Challenge | Vessels form a connected tree; tumours are irregularly shaped and often adjacent to vessels |

---

## 3. Data Preprocessing

All preprocessing steps are identical between datasets except CT windowing and label remapping, which are dataset-specific.

### 3.1 CT Windowing

CT images store Hounsfield Unit (HU) values. A window is applied per dataset to suppress irrelevant intensity ranges before normalisation:

| Dataset | Window centre | Window width | Effective HU range | Rationale |
|---|---|---|---|---|
| Lung (Task06) | −600 HU | 1500 HU | [−1350, 150] | Differentiates air-filled alveoli, lung parenchyma, and soft-tissue tumours |
| Hepatic (Task08) | 40 HU | 400 HU | [−160, 240] | Captures liver parenchyma (~40–80 HU), hepatic vessels (~50–70 HU), and tumours |

Pixels outside the window are clipped; the window is then linearly mapped to `[0, 255]` as `uint8`.

### 3.2 Multi-Slice Pseudo-RGB

For each target axial slice `z`, three channels are stacked from neighbouring slices:

```
Channel 0 = slice z−1  (previous)
Channel 1 = slice z    (current)
Channel 2 = slice z+1  (next)
```

At volume boundaries the edge slice is replicated. This gives the pretrained RGB backbones their expected 3-channel input while providing cross-slice context useful for detecting thin vessels and small lesions. This 2.5D approach is used identically for both datasets.

### 3.3 Label Remapping

Labels are remapped to a 0-indexed foreground convention; background becomes 255 (ignore index):

**Lung (Task06):**

| MSD label | Tissue | Remapped |
|---|---|---|
| 0 | Background | 255 |
| 1 | Cancer | 0 |

**HepaticVessel (Task08):**

| MSD label | Tissue | Remapped |
|---|---|---|
| 0 | Background | 255 |
| 1 | Vessel | 0 |
| 2 | Tumour | 1 |

Background = 255 excludes those pixels from the loss and metrics via `ignore_index=255`. The model is never directly supervised to predict "background" — it is the implicit state handled by the no-object class in the Hungarian matcher.

### 3.4 Slice Filtering

Slices are retained if they satisfy at least one criterion:

**Lung:** has cancer (label 1 present) **or** has lung content (>5% of windowed pixels > 100, ~HU > −700).

**Hepatic:** has vessel (label 1) **or** has tumour (label 2) **or** has abdominal content (>5% of windowed pixels > 100, ~HU > −60, capturing liver and soft tissue).

This keeps informative slices (foreground present) plus negative examples (anatomy visible, no foreground) for balanced training.

### 3.5 Augmentation (Training Only)

Identical for both datasets:

| Transform | Probability | Details |
|---|---|---|
| Random horizontal flip | 50% | Joint image and label |
| Random vertical flip | 50% | Joint image and label |
| Random rotation | 50% | ±20°; label uses `NEAREST` interpolation, border fill = 255 |
| Brightness jitter | 50% | Scale in [0.85, 1.15], clamped to [0, 255] |

---

## 4. Model Architectures

Both primary models use an **identical backbone** (DINOv2 ViT-B/14 reg4, ~86 M params), the **same loss function**, and the **same training schedule**, differing only in how they process backbone features into mask predictions. Both support any number of foreground classes — `num_classes` is read automatically from the dataset's `metadata.json`.

### 4.1 EoMT — Encoder-only Mask Transformer

EoMT removes the decoder entirely. Learnable segmentation queries are **prepended to the patch token sequence** and co-processed by the last `num_blocks = 4` ViT blocks alongside image patches.

**Architecture (matching eomt-master):**

| Component | Configuration |
|---|---|
| Backbone | DINOv2 ViT-B/14 reg4 (`vit_base_patch14_reg4_dinov2.lvd142m`) |
| Queries | `num_q = 20` learnable embeddings injected at the start of the last 4 blocks |
| Mask head | 3-layer MLP → per-query mask embedding; einsum with upscaled patch features → `[B, Q, H', W']` |
| Class head | Linear → `[B, Q, num_classes + 1]` (foreground classes + no-object) |
| Upscaling | 2 × transposed-conv blocks (ScaleBlock, ×2 each) recovering spatial resolution |
| Multi-layer supervision | Predictions at all `num_blocks + 1 = 5` intermediate states |

**Masked attention:** During training, each query attends only to patch tokens within its predicted mask region. The constraint is progressively relaxed via a staggered per-block annealing schedule. See Section 5.3.

### 4.2 ViT-Adapter + Mask2Former

ViT-Adapter augments the ViT backbone with an explicit multi-scale spatial prior, then decodes with a full Mask2Former transformer decoder.

**Architecture:**

| Component | Configuration |
|---|---|
| Backbone | DINOv2 ViT-B/14 reg4 (same as EoMT) |
| Spatial Prior Module (SPM) | Lightweight CNN stem → feature maps at strides 4, 8, 16, 32 |
| Adapter layers | Injector + Extractor every `adapter_interval = 6` ViT blocks (at blocks 5 and 11) |
| — Injector | Patch tokens cross-attend to SPM features (imports CNN spatial context) |
| — Extractor | SPM features cross-attend to patch tokens (imports ViT semantic context) |
| FPN Pixel Decoder | Top-down pyramid → `mask_features [B, 256, H/4, W/4]` + `[p32, p16, p8]` |
| Mask2Former Decoder | 9 layers cycling `p32 → p16 → p8`; each: masked cross-attention → self-attention → FFN |
| Class head | Linear → `[B, Q, num_classes + 1]` per decoder layer |
| Queries | `num_queries = 100` |
| Multi-layer supervision | Predictions at all 9 decoder layers |

**Note on deformable attention:** The original paper uses multi-scale deformable attention (MSDA) in the adapters; we use standard multi-head cross-attention. The EoMT paper's ablation shows the adapter contributes ~0.4 PQ with DINOv2, making this a minor approximation.

### 4.3 Mask2Former (Swin-Base, reference baseline)

HuggingFace `Mask2FormerForUniversalSegmentation` pretrained on ADE20k, class head replaced for the target number of foreground classes. Backbone: Swin-Base (~88 M). Used as the non-DINOv2 reference on the lung dataset.

---

## 5. Training

### 5.1 Loss Function

EoMT and ViT-Adapter+Mask2Former use `MaskClassificationLoss` — a Hungarian-matched multi-component loss:

| Component | Weight |
|---|---|
| Binary cross-entropy per matched mask (BCE) | λ_bce = 5.0 |
| Dice loss per matched mask | λ_dice = 5.0 |
| Cross-entropy over class predictions (CE) | λ_ce = 2.0 |
| No-object class weight | 0.1 |

The **Hungarian matcher** finds the optimal bijection between predicted queries and ground-truth masks at each decoder step. For multi-class datasets, the matcher operates over all foreground classes jointly. Losses are summed across all intermediate decoder states (5 for EoMT, 9 for ViT-Adapter+M2F).

### 5.2 Optimiser and Learning Rate Schedule

Both models use **AdamW** with identical hyperparameters across both datasets:

| Hyperparameter | Value | Paper |
|---|---|---|
| Head / decoder LR | 1×10⁻⁴ | 1×10⁻⁴ |
| Backbone base LR | 1×10⁻⁵ | 1×10⁻⁵ |
| LLRD decay factor | 0.8 per ViT block | 0.8 |
| Weight decay | 0.05 | 0.05 |
| Gradient clip (max norm) | 0.01 | 0.01 |

**Layer-wise LR decay (LLRD):** Each ViT block receives a distinct learning rate. Deeper blocks (closer to the output) get higher rates:

```
lr_block_i = backbone_lr × 0.8^(n_blocks − 1 − i)
```

Non-block backbone parameters (patch embedding, norm, positional embedding) use `backbone_lr × 0.8^n`.

**LR schedule:** Two-stage warmup followed by polynomial decay (matching the paper's `TwoStageWarmupPolySchedule`):

1. **Stage 1 (steps 0–500):** Head / decoder params warm up linearly 0 → `lr`. Backbone frozen at LR = 0.
2. **Stage 2 (steps 500–1000):** Backbone warms up linearly 0 → `backbone_lr`. Head continues.
3. **Decay (steps 1000–max_steps):** All groups decay polynomially:

```
lr(step) = lr_base × (1 − progress)^0.9
```

where `progress = (step − 1000) / (max_steps − 1000)`.

### 5.3 Attention Mask Annealing (EoMT only)

During training each query's attention is restricted to patch tokens within its predicted mask region. The constraint is relaxed staggered across blocks:

- Training is divided into `num_blocks + 1 = 5` equal windows.
- Block `i` (0 = first injected, 3 = deepest) anneals during window `[i+1, i+2]`.
- Within the window, the enforcement probability decays polynomially from 1.0 → 0.0 (power = 0.9).
- At inference, masked attention is fully disabled (`masked_attn_enabled = False`).

This matches eomt-master exactly and applies to both datasets identically.

### 5.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 512 × 512 |
| Batch size | 2 per GPU |
| Max epochs | 50 |
| Early stopping | patience = 10 epochs on `val/dice_mean` |
| Mixed precision | bf16 |
| Checkpoint criterion | best `val/dice_mean` (mean DICE across all classes) |
| Top-k checkpoints saved | 3 + last |

**Multi-GPU:** `--devices N` enables DDP. Per-GPU batch size is fixed; `max_steps` is the per-GPU step count, keeping the LR schedule correct at any device count. All validation metrics use `sync_dist=True`.

---

## 6. Evaluation

### 6.1 Inference

**EoMT:** Masked attention disabled at inference; final block output used.  
**ViT-Adapter+Mask2Former:** Final (9th) decoder layer output used.

Per-pixel scores per class `c`:

```
score[b, c, h, w] = Σ_q  sigmoid(mask[b, q, h, w]) × P(c | query_q)
```

where `P(c | query_q) = softmax(class[b, q, :])[c]` (no-object logit excluded).

Each class is thresholded independently:
```
pred_c[b, h, w] = (score[b, c, h, w] > 0.5)
```

Argmax is not used because queries can be matched to different classes and each class channel should be evaluated independently.

### 6.2 Global DICE Score

**Global DICE** is computed per class and accumulated across all validation slices before division:

```
DICE_c = (2 × TP_c + ε) / (|pred_c| + |gt_c| + ε)
```

The reported `val/dice_mean` is the mean across all foreground classes. `ε = 1e-6`.

Per-slice DICE is not used: slices without any foreground would return DICE ≈ 1 when the model correctly predicts nothing, artificially inflating the average.

**Logged metrics per dataset:**

| Dataset | Metrics logged |
|---|---|
| Lung (1 class) | `val/dice_cancer`, `val/iou_cancer`, `val/dice_mean`, `val/miou` |
| Hepatic (2 classes) | `val/dice_vessel`, `val/dice_tumour`, `val/iou_vessel`, `val/iou_tumour`, `val/dice_mean`, `val/miou` |

### 6.3 IoU

```
IoU_c = (TP_c + ε) / (TP_c + FP_c + FN_c + ε)
```

Background pixels (label = 255) excluded. Accumulated globally across all validation slices.

---

## 7. Differences from the EoMT Paper Setup

Dataset-driven adjustments are unavoidable; the one architectural approximation is noted separately.

| Aspect | EoMT Paper | This Project | Reason |
|---|---|---|---|
| **Task** | ADE20k / COCO (natural images) | MSD Lung + Hepatic (CT) | *Domain shift study* |
| **Number of classes** | Up to 150 (semantic) / panoptic | 1 (lung) or 2 (hepatic) | *Dataset* |
| **Input channels** | RGB (natural photos) | Multi-slice pseudo-RGB [z−1, z, z+1] | *CT has no colour* |
| **Evaluation metric** | PQ / mIoU | Global DICE + IoU per class | *Standard medical metric* |
| **Training data size** | Millions of images | 50–242 volumes → thousands of 2D slices | *Dataset* |
| **Backbone size** | ViT-Large (default) | ViT-Base (default; Large available) | *Computational constraint* |
| **Input resolution** | 640 / 896 px | 512 × 512 px | *Computational constraint* |
| **Number of queries** | 200 (EoMT-L) | EoMT: 20 / ViT-Adapter+M2F: 100 | *Scaled to task* |
| **Adapter attention** | Multi-scale deformable (MSDA) | Standard multi-head cross-attention | *MSDA requires custom CUDA ops* |

All other hyperparameters — AdamW, LLRD (0.8), two-stage warmup + polynomial decay (0.9), loss weights, masked attention annealing, gradient clipping (0.01) — match the paper exactly.
