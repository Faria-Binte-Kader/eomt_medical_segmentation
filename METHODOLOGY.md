# Methodology and Evaluation

## 1. Research Goal

The EoMT paper (CVPR 2025) introduces an encoder-only mask transformer that injects learnable queries directly into the last few ViT blocks, eliminating the need for a separate decoder. On natural-image benchmarks (ADE20k, COCO), EoMT matches or exceeds ViT-Adapter+Mask2Former with fewer parameters and FLOPs.

**This project tests whether that advantage holds under domain shift.** MSD Lung (Task06) is chest CT data — a fundamentally different input distribution from the natural images both backbones were pretrained on. ViT-Adapter+Mask2Former's explicit multi-scale spatial prior module (SPM) was designed to compensate for the ViT's lack of inductive spatial bias; it may provide a practical advantage when localising small, irregularly-shaped lung tumours in CT.

Both models are implemented as faithfully as possible to the EoMT paper, trained under identical conditions, and evaluated on the same held-out validation set. The only departures from the paper setup are those forced by the dataset and computational constraints — documented in Section 6.

---

## 2. Dataset

### MSD Lung (Task06)

The Medical Segmentation Decathlon Lung dataset contains 3D chest CT volumes from The Cancer Imaging Archive. Each volume has two labels: background (0) and lung cancer (1). The task is **binary segmentation**: identify cancerous lesions within the lung.

**Key statistics:**
- 63 labelled training volumes (3D), with 20% held out for validation (~50 train / ~13 val)
- No labelled test set (32 test volumes exist but have no labels)
- Class imbalance is severe — cancer occupies a tiny fraction of each CT volume

---

## 3. Data Preprocessing

### 3.1 CT Windowing

CT images store Hounsfield Unit (HU) values ranging from approximately −1000 (air) to +3000 (dense bone). We apply a **lung window** before normalisation:

```
Window centre (L) = −600 HU
Window width  (W) = 1500 HU
Effective range   = [L − W/2, L + W/2] = [−1350, 150] HU
```

Pixels outside this range are clipped, then the window is linearly mapped to `[0, 255]` as `uint8`. This differentiates air-filled alveoli (~−700 HU → ~110 windowed), soft tissue lung cancer (~40 HU → ~236 windowed), and surrounding structures.

### 3.2 Multi-Slice Pseudo-RGB

For each target slice `z`, three channels are stacked from neighbouring axial slices:

```
Channel 0 = slice z−1  (previous)
Channel 1 = slice z    (current)
Channel 2 = slice z+1  (next)
```

At volume boundaries, the edge slice is replicated. This gives the pretrained RGB backbones their expected 3-channel input while providing cross-slice context for detecting small lesions. Replicating a single slice across all three channels `[z, z, z]` would waste channel capacity; using adjacent slices is a common convention in 2.5D medical segmentation.

### 3.3 Label Remapping

| MSD label | Tissue | Remapped value |
|---|---|---|
| 0 | Background | **255** (ignore index) |
| 1 | Cancer | **0** |

Background is set to 255 so it is excluded from loss and metrics via `ignore_index=255`. The model is never directly supervised to predict "background" — it is the implicit "not predicted by any query" state, handled by the no-object class in the Hungarian matcher.

### 3.4 Slice Filtering

Axial slices are retained if they satisfy either:
1. **Cancer present**: at least one pixel has label = 1.
2. **Lung content present**: more than 5% of windowed pixels have value > 100 (~HU > −700, capturing lung parenchyma). Pure-air slices at the extremes of the scan are excluded.

### 3.5 Augmentation (Training Only)

| Transform | Probability | Details |
|---|---|---|
| Random horizontal flip | 50% | Joint image and label |
| Random vertical flip | 50% | Joint image and label |
| Random rotation | 50% | ±20°; label uses `NEAREST` interpolation, border fill = 255 |
| Brightness jitter | 50% | Scale in [0.85, 1.15], clamped to [0, 255] |

---

## 4. Model Architectures

Both primary models use an **identical backbone** (DINOv2 ViT-B/14 reg4, ~86 M params), the **same loss function**, and the **same training schedule**, differing only in how they process backbone features into mask predictions.

### 4.1 EoMT — Encoder-only Mask Transformer

EoMT removes the decoder entirely. Learnable segmentation queries are **prepended to the patch token sequence** and co-processed by the last `num_blocks = 4` ViT blocks.

**Architecture (matching eomt-master):**

| Component | Configuration |
|---|---|
| Backbone | DINOv2 ViT-B/14 reg4 (`vit_base_patch14_reg4_dinov2.lvd142m`) |
| Queries | `num_q = 20` learnable embeddings injected at the start of the last 4 blocks |
| Mask head | 3-layer MLP → per-query mask embedding; einsum with upscaled patch features → `[B, Q, H', W']` |
| Class head | Linear layer → `[B, Q, 2]` (1 cancer + 1 no-object) |
| Upscaling | 2 × transposed-conv blocks (ScaleBlock, ×2 each) recovering spatial resolution from patch grid |
| Multi-layer supervision | Predictions at all 5 intermediate states (`num_blocks + 1`) |

**Masked attention:** During training, each query attends only to patch tokens within its predicted mask region. This is implemented via a boolean attention mask (`attn_mask_probs`) and gradually annealed to full attention. See Section 5.3.

### 4.2 ViT-Adapter + Mask2Former

ViT-Adapter augments the ViT backbone with an explicit multi-scale spatial prior, then decodes with a full Mask2Former transformer decoder.

**Architecture:**

| Component | Configuration |
|---|---|
| Backbone | DINOv2 ViT-B/14 reg4 (same as EoMT) |
| Spatial Prior Module (SPM) | Lightweight CNN stem → feature maps at strides 4, 8, 16, 32; each projected to `embed_dim` via 1×1 conv + LayerNorm |
| Adapter layers | Injector + Extractor inserted every `adapter_interval = 6` ViT blocks (at blocks 5 and 11 for ViT-B with 12 blocks) |
| — Injector | Patch tokens cross-attend to SPM features (imports CNN spatial context) |
| — Extractor | SPM features cross-attend to patch tokens (imports ViT semantic context) |
| FPN Pixel Decoder | Top-down feature pyramid → `mask_features [B, 256, H/4, W/4]` + multi-scale maps `[p32, p16, p8]` |
| Mask2Former Decoder | 9 layers cycling `p32 → p16 → p8`; each: masked cross-attention → self-attention → FFN |
| Queries | `num_queries = 100` |
| Multi-layer supervision | Predictions at all 9 decoder layers |

**Note on deformable attention:** The original paper uses multi-scale deformable attention (MSDA) in the adapters, which requires custom CUDA operators. We use standard multi-head cross-attention as an approximation. The EoMT paper's Table 1 ablation shows the adapter contributes ~0.4 PQ with DINOv2, so this is a minor deviation.

### 4.3 Mask2Former (Swin-Base, reference baseline)

HuggingFace `Mask2FormerForUniversalSegmentation` pretrained on ADE20k, class head replaced for 1 foreground class (`num_labels=1, ignore_mismatched_sizes=True`). Backbone: Swin-Base (~88 M). Serves as the non-DINOv2, non-DINOv2-adapter reference point.

---

## 5. Training

### 5.1 Loss Function

EoMT and ViT-Adapter+Mask2Former use `MaskClassificationLoss` — a Hungarian-matched multi-component loss matching the paper:

| Component | Weight |
|---|---|
| Binary cross-entropy per matched mask (BCE) | λ_bce = 5.0 |
| Dice loss per matched mask | λ_dice = 5.0 |
| Cross-entropy over class predictions (CE) | λ_ce = 2.0 |
| No-object class weight | 0.1 |

The **Hungarian matcher** finds the optimal bijection between predicted queries and ground-truth cancer masks at each decoder step. Unmatched queries are supervised toward the no-object class. Losses are summed across all intermediate decoder states (5 for EoMT, 9 for ViT-Adapter+M2F).

### 5.2 Optimiser and Learning Rate Schedule

Both models use **AdamW** with identical hyperparameters:

| Hyperparameter | Value | Paper |
|---|---|---|
| Head / decoder LR | 1×10⁻⁴ | 1×10⁻⁴ |
| Backbone base LR | 1×10⁻⁵ | 1×10⁻⁵ |
| LLRD decay factor | 0.8 per ViT block | 0.8 |
| Weight decay | 0.05 | 0.05 |
| Gradient clip (max norm) | 0.01 | 0.01 |

**Layer-wise LR decay (LLRD):** Each ViT block receives a distinct learning rate. Deeper blocks (closer to the output) get higher rates; shallower blocks get lower rates:

```
lr_block_i = backbone_lr × decay^(n_blocks − 1 − i)
```

The deepest block gets `backbone_lr × 0.8^0 = backbone_lr`; the shallowest gets `backbone_lr × 0.8^(n−1)`. Non-block backbone parameters (patch embedding, norm, positional embedding) get `backbone_lr × 0.8^n`.

**LR schedule:** Two-stage warmup followed by polynomial decay (matching the paper's `TwoStageWarmupPolySchedule`):

1. **Stage 1 (head warmup, steps 0–500):** Head / decoder params warm up linearly from 0 → `lr`. Backbone LR is held at 0 (frozen).
2. **Stage 2 (backbone warmup, steps 500–1000):** Backbone params warm up linearly from 0 → `backbone_lr`. Head params continue from stage 1.
3. **Decay (steps 1000–max_steps):** All param groups decay polynomially to 0:

```
lr(step) = lr_base × (1 − progress)^0.9
```

where `progress = (step − warmup_end) / (max_steps − warmup_end)`.

### 5.3 Attention Mask Annealing (EoMT only)

During EoMT training, each query's attention is restricted to patch tokens within its predicted mask region (masked attention). To prevent queries from getting stuck in degenerate configurations, the mask constraint is progressively relaxed over training using a staggered per-block annealing schedule:

- Training is divided into `num_blocks + 1 = 5` equal windows.
- Block `i` (0 = first injected, 3 = last) anneals during window `[i+1, i+2]`, i.e., the earliest-injected block anneals first.
- Within each window, the probability that a query's mask constraint is enforced decays polynomially from 1.0 → 0.0 (power = 0.9).
- At inference, masked attention is disabled entirely (`masked_attn_enabled = False`).

This matches the annealing implementation in eomt-master exactly. The `attn_mask_prob_0..3` values are logged each training step.

### 5.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 512 × 512 |
| Batch size | 2 per GPU |
| Max epochs | 50 |
| Early stopping | patience = 10 epochs on `val/dice_mean` |
| Mixed precision | bf16 |
| Checkpoint criterion | best `val/dice_mean` |
| Top-k checkpoints saved | 3 + last |

**Multi-GPU:** Pass `--devices N` for DDP. Per-GPU batch size is fixed; effective batch scales linearly. `max_steps` is computed as the per-GPU step count, keeping the LR schedule proportionally correct. All validation metrics use `sync_dist=True` for correct global aggregation.

---

## 6. Evaluation

### 6.1 Inference

**EoMT:** Masked attention disabled at inference; final block output used.
**ViT-Adapter+Mask2Former:** Final (9th) decoder layer output used.

Per-pixel cancer scores are obtained by combining mask and class logits:

```
score[b, h, w] = Σ_q  sigmoid(mask[b, q, h, w]) × P(cancer | query_q)
```

where `P(cancer | query_q) = softmax(class[b, q, :])[cancer_idx]` (no-object logit excluded before softmax).

Final binary prediction:
```
pred[b, h, w] = (score[b, h, w] > 0.5)   →  True = cancer
```

Argmax is not used — with a single foreground class the output is `[B, 1, H, W]`, and argmax on a 1-channel tensor always returns 0 regardless of confidence. Thresholding at 0.5 gives proper binary predictions.

### 6.2 Global DICE Score

**Global DICE** (macro volumetric DICE) is the standard MSD challenge metric and is used throughout:

```
DICE = (2 × TP + ε) / (|pred_cancer| + |gt_cancer| + ε)
```

TP, `|pred_cancer|`, and `|gt_cancer|` are accumulated across **all validation slices** before division. `ε = 1e-6`.

Per-slice DICE is not used: most lung slices contain no cancer, and `dice(pred=∅, gt=∅) ≈ 1` artificially inflates the average. Global DICE treats all validation pixels uniformly.

### 6.3 IoU

```
IoU = (TP + ε) / (TP + FP + FN + ε)
```

Background pixels (label = 255) are excluded. Accumulated globally across all validation slices.

---

## 7. Differences from the EoMT Paper Setup

The table below separates dataset-driven adjustments (unavoidable) from the one architectural approximation made for ViT-Adapter:

| Aspect | EoMT Paper | This Project | Reason |
|---|---|---|---|
| **Task** | ADE20k / COCO (natural images) | MSD Lung Task06 (CT) | *Dataset* |
| **Number of classes** | Up to 150 (semantic) / panoptic | 1 (cancer only) | *Dataset* |
| **Input channels** | RGB (natural photos) | Multi-slice pseudo-RGB [z−1, z, z+1] | *Dataset — CT has no colour* |
| **Evaluation metric** | PQ / mIoU | Global DICE + IoU | *Standard medical metric* |
| **Training data size** | Millions of images | ~50 volumes → thousands of 2D slices | *Dataset* |
| **Backbone size** | ViT-Large (default) | ViT-Base (default; Large available) | *Computational constraint* |
| **Input resolution** | 640 / 896 px | 512 × 512 px | *Computational constraint* |
| **Number of queries** | 200 (EoMT-L) | EoMT: 20 / ViT-Adapter+M2F: 100 | *Scaled to 1-class task* |
| **Adapter attention** | Multi-scale deformable (MSDA) | Standard multi-head cross-attention | *MSDA requires custom CUDA ops* |

All other hyperparameters — AdamW, LLRD (0.8), two-stage warmup + polynomial decay (0.9), loss weights, masked attention annealing, gradient clipping (0.01) — match the paper exactly.
