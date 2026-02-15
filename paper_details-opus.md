# Sustained Boosting for Multimodal Medical Image Segmentation: Method Details

> This document transfers the Sustained Boosting (AUG) method from multimodal classification to the BraTS2018 brain tumor segmentation task.  
> Reference: Jiang et al., "Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion" (NeurIPS 2025)  
> Same-dataset reference: Li et al., "SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality" (CVPR 2025)

---

## 1. Problem Definition and Notation

### 1.1 Task Definition

Given the BraTS2018 dataset with $N$ samples, each sample has $M = 4$ MRI modalities: T1, T1ce, T2, and FLAIR. The input data is denoted as:

$$
\mathcal{X} = \{(\mathbf{x}_i^{(1)}, \mathbf{x}_i^{(2)}, \mathbf{x}_i^{(3)}, \mathbf{x}_i^{(4)})\}_{i=1}^N, \quad \mathbf{x}_i^{(m)} \in \mathbb{R}^{1 \times D \times H \times W}
$$

The corresponding labels are three-channel binary segmentation masks:

$$
\mathcal{Y} = \{\mathbf{y}_i\}_{i=1}^N, \quad \mathbf{y}_i \in \{0, 1\}^{3 \times D \times H \times W}
$$

The three channels correspond to:
- **WT (Whole Tumor)**: labels $\in \{1, 2, 3\}$
- **TC (Tumor Core)**: labels $\in \{2, 3\}$
- **ET (Enhancing Tumor)**: label $= 3$

### 1.2 Core Symbol Definitions

| Symbol | Description |
|--------|-------------|
| $M = 4$ | Number of modalities (T1, T1ce, T2, FLAIR) |
| $K = 3$ | Number of segmentation sub-tasks (WT, TC, ET) |
| $\phi^{(m)}(\cdot; \theta^{(m)})$ | Feature extractor for modality $m$ (UNet Encoder + Decoder), parameters $\theta^{(m)}$ |
| $\mathbf{f}_i^{(m)}$ | Decoder feature map of modality $m$ for sample $i$, $\in \mathbb{R}^{C_f \times D \times H \times W}$ |
| $n_m$ | Number of segmentation heads currently assigned to modality $m$ |
| $\psi_t^{(m)}(\cdot; \Theta_t^{(m)})$ | Private part of head $t$ for modality $m$, parameters $\Theta_t^{(m)}$ |
| $\omega(\cdot; \mathbf{W}_s)$ | Shared projection layer (Shared Head), parameters $\mathbf{W}_s$, shared by all modalities and heads |
| $\mathbf{p}_{it}^{(m)}$ | Predicted logits of head $t$ for modality $m$ on sample $i$, $\in \mathbb{R}^{K \times D \times H \times W}$ |
| $\sigma(\cdot)$ | Sigmoid activation |
| $\lambda$ | Residual label smoothing coefficient |
| $\lambda_b$ | Boosting loss weight |
| $\sigma_{aca}$ | ACA confidence comparison coefficient |
| $\tau$ | ACA dead-zone tolerance threshold |
| $t_N$ | ACA check interval (in epochs) |

---

## 2. Overall Architecture

### 2.1 Architecture Overview

The method uses a **Late Fusion** architecture with three core components:

1. **Modality-specific feature extractors**: $M = 4$ independent 3D UNets (nnUNet-style), one per modality.
2. **Configurable segmentation head set**: Each modality has $n_m \geq 1$ lightweight 1×1×1 convolutional heads; the number is adjusted dynamically during training based on confidence.
3. **Shared projection layer**: All modalities and all heads share a final projection convolution mapping to $K$ segmentation outputs.

Forward pass:

$$
\text{Input: } \mathbf{X} \in \mathbb{R}^{B \times 4 \times D \times H \times W} \xrightarrow{\text{split}} \{\mathbf{x}^{(m)}\}_{m=1}^4 \xrightarrow[\text{per modality}]{\phi^{(m)}} \{\mathbf{f}^{(m)}\}_{m=1}^4 \xrightarrow[\text{multi-head}]{\psi_t^{(m)}, \omega} \text{Output}
$$

### 2.2 Modality-Specific Feature Extractors

Each modality $m$ has an independent 3D UNet with an encoder (6 stages) and decoder (5 stages). The encoder extracts multi-scale features with downsampling; the decoder restores spatial resolution via skip connections.

For input $\mathbf{x}^{(m)} \in \mathbb{R}^{B \times 1 \times D \times H \times W}$, the feature extraction is:

$$
\mathbf{f}^{(m)} = \phi^{(m)}(\mathbf{x}^{(m)}; \theta^{(m)}) \in \mathbb{R}^{B \times C_f \times D \times H \times W}
$$

where $C_f$ is the number of channels in the final decoder stage. The encoder uses standard blocks `Conv3d → InstanceNorm3d → LeakyReLU`; the decoder uses `ConvTranspose3d` for upsampling and concatenates skip features from the encoder.

**Design note**: In the Boosted method, the UNet's built-in `seg_layer` (final 1×1 segmentation conv) is not used. The UNet acts only as a feature extraction pipeline and outputs the final decoder feature map $\mathbf{f}^{(m)}$. All segmentation predictions are produced by external configurable heads.

### 2.3 Configurable Segmentation Head

Each head has two parts:

**Private part** (per head):
$$
\mathbf{h}_{it}^{(m)} = \text{ReLU}\left(\text{Conv3d}_{1 \times 1 \times 1}(\mathbf{f}_i^{(m)}; \Theta_t^{(m)})\right) \in \mathbb{R}^{B \times C_h \times D \times H \times W}
$$

**Shared projection** (shared by all modalities and heads):
$$
\mathbf{p}_{it}^{(m)} = \text{Conv3d}_{1 \times 1 \times 1}(\mathbf{h}_{it}^{(m)}; \mathbf{W}_s) \in \mathbb{R}^{B \times K \times D \times H \times W}
$$

where $C_h$ is the head's intermediate channel dimension.

**Initial head**: At initialization, each modality has exactly one head; that head's private part is the identity (no nonlinearity), so the initial behavior matches the baseline. Newly added heads use the full Conv3d + ReLU.

**Why 1×1×1 instead of 3×3×3?**
- For 3D volumes, 3×3×3 conv has 27× the memory of 1×1×1.
- Spatial context is already captured by the UNet decoder; heads only need to map features to class logits.
- This matches the role of FC layers in the original paper—FC is effectively 1×1 with no spatial extent.

**Role of the shared projection**:
- Pushes all modalities and heads to predict in the same semantic space.
- Enables cross-modal interaction: gradients from both weak and strong modalities flow through the shared layer.
- Common in multimodal learning (e.g. MLA, DI-MML).

### 2.4 Fusion Strategy

**Intra-modality fusion** (Boosting): sum logits over all heads of the same modality:

$$
\hat{\mathbf{p}}_i^{(m)} = \sum_{t=1}^{n_m} \mathbf{p}_{it}^{(m)} \in \mathbb{R}^{K \times D \times H \times W}
$$

**Inter-modality fusion** (Late Fusion): equal-weight average over modalities:

$$
\mathbf{o}_i = \frac{1}{M} \sum_{m=1}^{M} \hat{\mathbf{p}}_i^{(m)} \in \mathbb{R}^{K \times D \times H \times W}
$$

Final segmentation is obtained by Sigmoid and thresholding:

$$
\hat{\mathbf{y}}_i = \mathbb{1}[\sigma(\mathbf{o}_i) > 0.5]
$$

---

## 3. Sustained Boosting Loss

### 3.1 Main Idea

Classical gradient boosting trains classifiers sequentially: train the first, fix it, then the second, etc. **Sustained Boosting** instead **optimizes all classifiers jointly end-to-end** via three complementary loss terms.

### 3.2 Residual Labels

For modality $m$ with $n_m$ heads, the residual label for the $n_m$-th (newest) head is:

$$
\hat{\mathbf{y}}_{i,n_m}^{(m)} = \text{clamp}\left(\mathbf{y}_i - \lambda \sum_{j=1}^{n_m - 1} \mathbf{y}_i \odot \sigma(\mathbf{p}_{ij}^{(m)}\text{.detach}), \; \min=0\right)
$$

where:
- $\lambda \in [0, 1]$ is the smoothing coefficient controlling residual decay.
- $\odot$ is element-wise product.
- $\mathbf{y}_i$ masks so that residuals are computed only on foreground voxels.
- `.detach` means earlier heads' predictions do not receive gradients through the residual label.
- $\text{clamp}(\cdot, \min=0)$ keeps the residual label non-negative.

**Interpretation**: The residual label is "what the previous heads have not yet predicted well." On foreground voxels, if earlier heads already predict correctly with high confidence ($\sigma \approx 1$), the residual is near 0 and the new head need not focus there; otherwise the residual is near $\mathbf{y}$ and the new head focuses there.

**When $n_m = 1$**: There are no previous heads, so the residual label reduces to the original label $\hat{\mathbf{y}} = \mathbf{y}$.

### 3.3 Three Loss Terms

#### 3.3.1 Residual loss $\varepsilon$

Loss of the newest head's prediction on the residual label:

$$
\varepsilon^{(m)} = \text{BCE}\left(\sigma(\mathbf{p}_{i,n_m}^{(m)}), \; \hat{\mathbf{y}}_{i,n_m}^{(m)}\right)
$$

**BCE only (no Dice)** is used because the residual label $\hat{\mathbf{y}}$ is a continuous soft label in $[0, 1]$; Dice on soft labels can be unstable, while BCE naturally supports soft labels.

#### 3.3.2 Joint prediction loss $\varepsilon_{all}$

Loss of the sum of all $n_m$ heads' logits on the original label:

$$
\varepsilon_{all}^{(m)} = \mathcal{L}_{\text{DiceBCE}}\left(\sum_{j=1}^{n_m} \mathbf{p}_{ij}^{(m)}, \; \mathbf{y}_i\right)
$$

This term keeps the ensemble of all heads well-calibrated on the original target.

#### 3.3.3 Previous prediction loss $\varepsilon_{pre}$

Loss of the sum of the first $n_m - 1$ heads' logits on the original label:

$$
\varepsilon_{pre}^{(m)} = \begin{cases} \mathcal{L}_{\text{DiceBCE}}\left(\sum_{j=1}^{n_m-1} \mathbf{p}_{ij}^{(m)}, \; \mathbf{y}_i\right), & \text{if } n_m > 1 \\ 0, & \text{if } n_m = 1 \end{cases}
$$

This term prevents the shared encoder from degrading the performance of existing heads when it is updated.

**Roles of the three terms**:
- $\varepsilon$: Teaches the new head to fit the residual (core boosting idea).
- $\varepsilon_{all}$: Maintains quality of the full ensemble.
- $\varepsilon_{pre}$: Protects existing heads from degradation (encoder is shared, so updates affect all heads).

### 3.4 Boosting Modality Loss

For modality $m$ with more than one head ($n_m > 1$), the Boosting loss is:

$$
\mathcal{L}_{\text{boost}}^{(m)} = \varepsilon^{(m)} + \varepsilon_{all}^{(m)} + \varepsilon_{pre}^{(m)}
$$

### 3.5 Total Training Loss

The total loss has two parts:

1. **Fused loss**: Standard Dice + BCE on the fused output $\mathbf{o}$:

$$
\mathcal{L}_{\text{fused}} = \mathcal{L}_{\text{DiceBCE}}(\mathbf{o}, \mathbf{y})
$$

2. **Boosting loss**: Sum over modalities with multiple heads, normalized:

$$
\mathcal{L}_{\text{boost}} = \frac{1}{|\mathcal{M}_b|} \sum_{m \in \mathcal{M}_b} \mathcal{L}_{\text{boost}}^{(m)}
$$

where $\mathcal{M}_b = \{m \mid n_m > 1\}$ and $|\mathcal{M}_b|$ is its size.

**Total loss**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}} + \lambda_b \cdot \mathcal{L}_{\text{boost}}
$$

with $\lambda_b$ the Boosting loss weight.

**When all modalities have $n_m = 1$**: $\mathcal{L}_{\text{boost}} = 0$, so the total loss reduces to $\mathcal{L}_{\text{fused}}$, matching the baseline.

### 3.6 Base Loss $\mathcal{L}_{\text{DiceBCE}}$

$$
\mathcal{L}_{\text{DiceBCE}}(\mathbf{p}, \mathbf{y}) = \mathcal{L}_{\text{Dice}}(\sigma(\mathbf{p}), \mathbf{y}) + \mathcal{L}_{\text{BCE}}(\sigma(\mathbf{p}), \mathbf{y})
$$

Dice loss:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{v} \hat{y}_v \cdot y_v}{\sum_{v} \hat{y}_v + \sum_{v} y_v + \epsilon}
$$

---

## 4. Adaptive Classifier Assignment (ACA)

### 4.1 Motivation

During training, modalities learn at different speeds and have different segmentation strengths, and these differences change over time. ACA monitors per-modality confidence and dynamically adds new heads to the weakest modality to strengthen it.

### 4.2 Confidence Score

For modality $m$, confidence is the mean Sigmoid probability over foreground voxels of the joint prediction:

$$
s^{(m)} = \frac{1}{N} \sum_{i=1}^{N} \text{mean}_{\text{spatial}}\left(\mathbf{y}_i \odot \sigma\left(\sum_{j=1}^{n_m} \mathbf{p}_{ij}^{(m)}\right)\right)
$$

In practice: for each sample, average the predicted probability in the foreground ($\mathbf{y} > 0.5$), then average over spatial and class dimensions, then over samples.

### 4.3 Assignment Rule (4-Modality Extension)

The original paper compares two modalities; here it is extended to 4 modalities.

Every $t_N$ epochs:

1. Compute confidences for the 4 modalities: $s^{(1)}, s^{(2)}, s^{(3)}, s^{(4)}$.
2. Set $s_{\max} = \max_m s^{(m)}$, $s_{\min} = \min_m s^{(m)}$.
3. Identify the weakest modality $m^* = \arg\min_m s^{(m)}$.
4. **If** $s_{\max} - \sigma_{aca} \cdot s_{\min} > \tau$, **then** add a new head to $m^*$: $n_{m^*} \leftarrow n_{m^*} + 1$.

$$
\text{if } s_{\max} - \sigma_{aca} \cdot s_{\min} > \tau: \quad n_{m^*} \leftarrow n_{m^*} + 1
$$

### 4.4 New Head Initialization and Optimizer Registration

When ACA adds a head:
1. The new head's private part is initialized with default PyTorch (Kaiming uniform).
2. Its parameters are added to the optimizer's param groups with the current learning rate.
3. The shared projection is not re-initialized.

---

## 5. Full Model Data Flow

```
Input: X ∈ R^{B×4×D×H×W}
  │
  ├── x^(1) = X[:, 0:1, ...]     (T1)
  ├── x^(2) = X[:, 1:2, ...]     (T1ce)
  ├── x^(3) = X[:, 2:3, ...]     (T2)
  └── x^(4) = X[:, 3:4, ...]     (FLAIR)
        │           │           │           │
        ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ UNet_1  │ │ UNet_2  │ │ UNet_3  │ │ UNet_4  │
   │Enc+Dec  │ │Enc+Dec  │ │Enc+Dec  │ │Enc+Dec  │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │
   f^(1)∈R^{B,Cf,D,H,W}  ...          ...          ...
        │           │           │           │
   ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
   │Head_1_1│ │ Head_2_1│ │ Head_3_1│ │ Head_4_1│
   │ Head_1_2│ │   ...   │ │   ...   │ │   ...   │
   │   ...   │ │(n_2 个) │ │(n_3 个) │ │(n_4 个) │
   │(n_1 个) │ │         │ │         │ │         │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │
    Σ logits    Σ logits    Σ logits    Σ logits
   p̂^(1)       p̂^(2)       p̂^(3)       p̂^(4)
        │           │           │           │
        └───────────┴─────┬─────┴───────────┘
                          │
                   o = mean(p̂^(1), ..., p̂^(4))
                          │
                          ▼
               Final output ∈ R^{B×3×D×H×W}
```

### Gradient Flow

Gradients of the total loss flow along two paths:

1. **$\mathcal{L}_{\text{fused}}$**: fused output $\mathbf{o}$ → equal-weight average → per-modality aggregated predictions → each head → shared layer → UNet decoder → UNet encoder.
2. **$\mathcal{L}_{\text{boost}}$**: only for modalities with $n_m > 1$, via $\varepsilon$, $\varepsilon_{all}$, $\varepsilon_{pre}$ → corresponding heads → shared layer → UNet.

The shared projection $\omega$ receives gradients from all modalities and all heads; PyTorch accumulates them automatically.

---

## 6. Training Procedure

### 6.1 Training Algorithm

```
Initialize: n_1 = n_2 = n_3 = n_4 = 1
Initialize all UNet params {θ^(m)}, all head params {Θ_1^(m)}, shared layer W_s

for epoch = 1 → T do:
    // ========== Training ==========
    for each mini-batch (X, y) do:
        1. Split input by modality
        2. Extract features f^(m) per modality via UNet
        3. Get {p_t^(m)} from all heads per modality
        4. Compute L_total = L_fused + λ_b · L_boost
        5. Backprop and update all parameters
    end for

    // ========== ACA check ==========
    if epoch mod t_N == 0:
        6. Compute confidences s^(1), ..., s^(4) on training set
        7. Find s_max, s_min, m* = argmin(s)
        8. if s_max - σ_aca · s_min > τ:
               Add new head to modality m*
               n_{m*} ← n_{m*} + 1
               Add new head params to optimizer
    end if

    // ========== Validation ==========
    if epoch mod val_freq == 0:
        Compute Dice (WT, TC, ET) on validation set
        Save best model
    end if
end for
```

### 6.2 Implementation Details

1. **Detach in residual labels**: When computing the residual label, earlier heads' predictions must be `.detach()` so the residual label does not backprop through them.

2. **Boost loss normalization**: The Boost loss is normalized by the number of boosted modalities $|\mathcal{M}_b|$ so its scale stays stable across training.

3. **When $n_m = 1$**: If every modality has exactly one head, $\mathcal{L}_{\text{boost}} = 0$ and the model equals the baseline. Boosting is introduced gradually—until the first ACA trigger, training is baseline-only.

4. **Gradient monitoring**: Forward hooks on the first encoder stage of each UNet record gradient norms and ratios per modality to monitor imbalance.

---

## 7. Evaluation Metrics

Dice coefficient is used for segmentation evaluation, reported for the three sub-tasks:

$$
\text{Dice}(P, G) = \frac{2|P \cap G|}{|P| + |G|}
$$

where $P$ is the predicted segmentation and $G$ the ground truth. Implemented with MONAI's `DiceHelper`; logits are passed through Sigmoid before comparison with labels.

---

## 8. Data Processing

### 8.1 Data Format

- **Source**: BraTS2018, nnUNet-preprocessed.
- **Images**: `.npy`, shape `(4, D, H, W)` for T1, T1ce, T2, FLAIR.
- **Labels**: `.npy`, shape `(1, D, H, W)`.
- **Split**: 5-fold cross-validation.

### 8.2 Label Conversion

Original BraTS labels → three-channel binary masks:

$$
\text{WT}: y \in \{1, 2, 3\} \to 1, \quad \text{TC}: y \in \{2, 3\} \to 1, \quad \text{ET}: y = 3 \to 1
$$

### 8.3 Data Augmentation

**Training**:
1. `SpatialPadd` → pad to 128³
2. `RandSpatialCropd` → random crop 128³
3. `RandFlipd` × 3 → random flip on three axes (prob=0.5)
4. `RandGaussianNoised` (prob=0.15)
5. `RandGaussianSmoothd` (prob=0.15)
6. `RandAdjustContrastd` (prob=0.15)
7. `RandScaleIntensityd` (prob=0.15)

**Validation**:
1. `CenterSpatialCropd` → center crop 128³
2. `SpatialPadd` → pad to 128³

---

## 9. Hyperparameters

This section defines the hyperparameter categories and their roles; specific values are not listed.

### 9.1 UNet Architecture

| Hyperparameter | Role |
|----------------|------|
| Input channels | Channels per expert (single modality) |
| Number of classes $K$ | Output channels (segmentation sub-tasks) |
| Encoder stages | Number of encoder stages |
| Feature channels per stage | Channel sequence for encoder/decoder |
| Kernel sizes | 3D conv kernel size per stage |
| Strides | Downsampling stride per stage |
| Deep supervision | Whether to use deep supervision |

### 9.2 Sustained Boosting

| Hyperparameter | Symbol | Role |
|----------------|--------|------|
| Residual label smoothing | $\lambda$ | Controls decay of residual labels; larger → larger residuals, new head does more |
| Boost loss weight | $\lambda_b$ | Balance between fused and Boost loss |
| Head intermediate channels | $C_h$ | Output channels of head's private conv |
| Max heads per modality | $n_{\max}$ | Upper bound on heads per modality (regularization and memory) |

### 9.3 ACA

| Hyperparameter | Symbol | Role |
|----------------|--------|------|
| Confidence comparison coefficient | $\sigma_{aca}$ | How strict the modality gap must be to add a head; larger → harder to trigger |
| Dead-zone threshold | $\tau$ | No assignment when gap is within this range |
| Check interval | $t_N$ | ACA check every $t_N$ epochs |

### 9.4 Training

| Hyperparameter | Role |
|----------------|------|
| Random seed | Reproducibility |
| Total epochs | Training length |
| Learning rate | Adam initial learning rate |
| Train batch size | Samples per batch (train) |
| Val batch size | Samples per batch (val) |
| Validation frequency | Validation every N epochs |
| Cross-validation fold | Which K-fold index to use |

---

## 10. Innovation Summary

### 10.1 Main Idea: From Balancing the Learning Process to Balancing Segmentation Ability

Many multimodal methods address imbalance by **balancing the learning process** (e.g. gradient modulation OGM, learning rate MSLR). This method instead **balances segmentation ability**: it directly strengthens weak modalities by dynamically adding lightweight heads.

### 10.2 When New Heads Are Added

Heads are added by **ACA** during training. Conditions:

1. **When**: at epochs $k \cdot t_N$ ($k = 1, 2, 3, ...$).
2. **Condition**: confidence gap above threshold, $s_{\max} - \sigma_{aca} \cdot s_{\min} > \tau$.
3. **Where**: add a new head to the modality with lowest confidence (weakest).

From training logs, for example:
- **Epoch 60**: T1ce (modality 1) gets a second head → n_heads `[1, 2, 1, 1]`.
- **Epoch 120**: T1 (modality 0) gets a second head → `[2, 2, 1, 1]`.
- **Epoch 180**: FLAIR (modality 3) gets a second head → `[2, 2, 1, 2]`.
- **Epoch 240**: T1 gets a third head → `[3, 2, 1, 2]`.

T2 (modality 2) may keep a single head throughout, indicating consistently higher confidence.

### 10.3 How New Heads Are Added

1. **Create head**: New `ConfigurableSegHead` with private Conv3d($C_f \to C_h$, k=1) + ReLU and reference to shared layer.
2. **Initialize**: Private part with PyTorch default (Kaiming uniform).
3. **Device**: Move to same device as model.
4. **Register in model**: Append to the modality's `ModuleList`.
5. **Register in optimizer**: `optimizer.add_param_group()` with current learning rate.
6. **Training**: New head participates in forward and backward; its logits are summed in the modality's aggregate.

### 10.4 Differences from the Original (AUG) Paper

| Aspect | Original (AUG, classification) | This project (segmentation) |
|--------|--------------------------------|-----------------------------|
| Encoder | ResNet18, 1D feature vector | 3D UNet Enc+Dec, 3D feature map |
| Classifier / head | FC: D→256→K | Conv3d 1×1: $C_f$→$C_h$→K |
| Output | Softmax probability vector | Sigmoid probability volume |
| Loss | Cross-Entropy | Dice + BCE (joint/previous), BCE (residual) |
| Confidence | $\mathbf{y}^\top \cdot \mathbf{p}$ | Mean Sigmoid on foreground voxels |
| Modalities | 2 | 4 |
| ACA | Two-modality comparison | Strongest vs weakest |
| Total loss | $\sum_m \mathcal{L}_{\text{SUB}}^{(m)}$ | $\mathcal{L}_{\text{fused}} + \lambda_b \cdot \mathcal{L}_{\text{boost}}$ |

---

## 11. Ablation Design

### 11.1 Common Setup

Following SimMLM and AUG:
- **Dataset**: BraTS2018, 285 annotated samples, 4:1 train/val (K-fold).
- **Metrics**: Dice (WT, TC, ET); report mean ± std.
- **Training**: Same optimizer (Adam), learning rate, batch size, augmentation as baseline.
- **Statistics**: At least 3 seeds per setting; report mean ± std.
- **Control**: Only the ablated component varies; all other hyperparameters and data pipeline fixed.

### 11.2 Ablation Overview

| ID | Experiment | Ablated | Setup | Goal |
|----|------------|---------|-------|------|
| A0 | **Baseline** | — | 4 UNets, static equal-weight fusion, Dice+BCE | Reference |
| A1 | **Full Method** | — | Sustained Boosting + ACA + Shared Head | Upper bound |
| A2 | **w/o ACA (Fixed Heads)** | ACA | ACA off; all modalities fixed to same number of heads (e.g. $n_{\max}$) | ACA vs fixed assignment |
| A3 | **w/o $\varepsilon$ (Residual Loss)** | Residual loss | Remove $\varepsilon$; keep $\varepsilon_{all} + \varepsilon_{pre}$ | Necessity of residual learning |
| A4 | **w/o $\varepsilon_{all}$ (Joint Loss)** | Joint loss | Remove $\varepsilon_{all}$; keep $\varepsilon + \varepsilon_{pre}$ | Necessity of ensemble constraint |
| A5 | **w/o $\varepsilon_{pre}$ (Previous Loss)** | Previous loss | Remove $\varepsilon_{pre}$; keep $\varepsilon + \varepsilon_{all}$ | Necessity of protecting existing heads |
| A6 | **w/o Shared Head** | Shared projection | Each head has its own projection (no shared Layer2) | Necessity of sharing |
| A7 | **w/o Boost Loss ($\lambda_b = 0$)** | Boosting loss | Keep ACA and head addition; only $\mathcal{L}_{\text{fused}}$ | Contribution of Boost loss |

### 11.3 Hyperparameter Sensitivity

| ID | Experiment | Variable | Range | Goal |
|----|------------|----------|-------|------|
| S1 | **$\lambda$** | Residual smoothing | $\{0.1, 0.2, 0.33, 0.5, 1.0\}$ | Robustness to $\lambda$ |
| S2 | **$\lambda_b$** | Boost weight | $\{0.1, 0.5, 1.0, 2.0, 5.0\}$ | Fused vs Boost balance |
| S3 | **$\sigma_{aca}$** | ACA coefficient | $\{1.0, 1.25, 1.5, 1.75\}$ | Robustness to $\sigma_{aca}$ |
| S4 | **$t_N$** | ACA interval | $\{10, 30, 60, 90\}$ epochs | Effect of check frequency |
| S5 | **$C_h$** | Head channels | $\{4, 8, 16, 32\}$ | Head capacity |
| S6 | **$n_{\max}$** | Max heads | $\{2, 4, 6, 8, 10\}$ | Heads vs performance/cost |

### 11.4 Additional Analyses

| ID | Experiment | Goal | Setup |
|----|------------|------|-------|
| E1 | **Gradient norms** | Modality imbalance over time | Plot encoder gradient norms and ratios per modality |
| E2 | **Confidence over time** | ACA decisions | Plot $s^{(m)}$ vs epoch, mark ACA triggers |
| E3 | **Head count over time** | Assignment dynamics | Plot $n_m$ vs epoch per modality |
| E4 | **$t-1$ heads vs all heads** | Residual learning | Compare Dice of first $n_m-1$ heads vs all $n_m$ (cf. original Fig. 3) |
| E5 | **Params and compute** | Efficiency | Params, memory, train/inference time vs baseline and head count |
| E6 | **Per sub-task** | WT/TC/ET | Dice gains per sub-task |

### 11.5 Result Table Templates

#### Table 1: Component ablation

| Method | WT Dice | TC Dice | ET Dice | Avg Dice |
|--------|---------|---------|---------|----------|
| A0: Baseline | | | | |
| A7: w/o Boost Loss | | | | |
| A3: w/o ε | | | | |
| A4: w/o ε_all | | | | |
| A5: w/o ε_pre | | | | |
| A6: w/o Shared Head | | | | |
| A2: Fixed Heads | | | | |
| A1: Full Method | | | | |

#### Table 2: Hyperparameter sensitivity (e.g. $\lambda$)

| $\lambda$ | WT Dice | TC Dice | ET Dice | Avg Dice |
|-----------|---------|---------|---------|----------|
| 0.1 | | | | |
| 0.2 | | | | |
| 0.33 | | | | |
| 0.5 | | | | |
| 1.0 | | | | |

#### Table 3: Compute (cf. AUG Table VI)

| Setting | #Heads (final) | #Params | Train time | Inference time | Avg Dice |
|---------|----------------|---------|------------|----------------|----------|
| Baseline (A0) | [1,1,1,1] | | | | |
| Max 2 heads | | | | | |
| Max 4 heads | | | | | |
| Max 6 heads | | | | | |
| Max 8 heads | | | | | |

---

## 12. Project Structure

```
reb-UNet/
├── configs.py                          # Baseline (UNet, data, training)
├── configs_boosted.py                  # Boosted hyperparameters
├── pipeline.py                         # Baseline training entry
├── pipeline_boosted.py                 # Boosted training entry
├── models/
│   ├── nnunet.py                       # 3D UNet (return_features)
│   ├── baseline.py                     # BaselineLateFusion (4 UNets, static average)
│   └── boosted_fusion.py              # BoostedLateFusion (UNet + configurable heads + shared layer)
├── loss/
│   ├── dice_bce_loss.py                # Dice + BCE
│   └── boosted_loss.py                 # SustainedBoostingLoss (ε + ε_all + ε_pre)
├── train/
│   ├── trainer.py                      # Baseline training loop
│   └── trainer_boosted.py              # Boosted loop + ACA
├── dataset/
│   ├── processors.py                   # SingleStreamDataset, BratsEvalSet
│   └── utils.py                        # Data utilities
└── assets/
    └── kfold_splits.json               # Data splits
```
