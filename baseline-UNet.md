# Baseline UNet 设计文档（Late Fusion 版）

> 本文档基于 SimMLM 项目，剥离所有创新点后，提取出最基础的 BraTS2018 分割 baseline 模型设计方案。
> 采用 **Late Fusion** 架构：4 个独立的单通道 3D UNet Expert，各自处理一个 MRI 模态，输出通过**静态等权平均**融合。
> 此设计保留了逐模态独立处理的结构，便于后续监控各模态的梯度范数、置信度等指标。

---

## 一、SimMLM 创新点识别（需要删除的部分）

| 创新点 | 说明 | 涉及文件 |
|--------|------|----------|
| **动态 Router/Gating 网络** | 5 层 CNN + Linear，学习各模态的动态权重（softmax 归一化） | `models/dmome.py` 中 `self.router` |
| **MoFe 排序损失** | 强制"更多模态→更好性能"的排序约束损失 | `loss/blended_loss.py`（`compute_ranking_loss` 部分） |
| **PairedDataset** | 为 MoFe 损失构造的配对数据集（同一样本不同模态组合配对） | `dataset/processors.py` 中 `PairedDataset` 类 |
| **两阶段训练** | 第一阶段：单模态 Expert 预训练；第二阶段：DMoME + MoFe 联合训练 | `pipeline_expert_pretraining.py`, `pipeline_joint_training.py` |
| **缺失模态处理** | 运行时随机丢弃模态、Router 掩码缺失模态（权重设为 -inf）、15 种缺失组合评估 | `models/dmome.py` 中 `modality_mask`、`dataset/processors.py` 中 `drop_mode` |
| **MoMKE 对比模型** | 另一种混合专家对比模型 | `models/momke.py` 整个文件 |

**保留的核心骨架（本 Baseline 的基础）：**
- 4 个独立的单通道 UNet Expert（与 SimMLM 相同的 nnUNet 架构）
- 各 Expert 独立处理对应模态 → 输出在 logit 层融合

---

## 二、Baseline 模型定义

### 2.1 核心思路

Baseline 采用 **Late Fusion**（后期融合）架构：
- **4 个独立的单通道 3D UNet**（nnUNet 风格），分别处理 T1、T1ce、T2、FLAIR 四个 MRI 模态
- 每个 Expert 输入 `(B, 1, 128, 128, 128)`，输出 logits `(B, 3, 128, 128, 128)`（WT, TC, ET）
- 4 个 Expert 的输出通过 **静态等权平均** 融合为最终预测

**与 SimMLM 的关键区别：**

| 维度 | Baseline (本文档) | SimMLM |
|------|-------------------|--------|
| Expert 结构 | 相同（4 个单通道 UNet） | 相同 |
| 融合方式 | **静态等权平均** (`output = mean(o1, o2, o3, o4)`) | **动态加权** (Router 学习权重) |
| 损失函数 | Dice + BCE | Dice + BCE **+ MoFe 排序损失** |
| 训练策略 | **单阶段**端到端训练 | **两阶段**（Expert 预训练 + 联合训练） |
| 缺失模态 | 不处理（始终使用全部 4 模态） | 动态掩码 + 权重重分配 |

> **论文参照**: 本 Baseline 对应论文 Supplementary Table A7 中的 **"DMoME w/ static averaging"** 消融实验设置。

### 2.2 单个 UNet Expert 架构参数

```python
# 每个 Expert 使用相同的 nnUNet 配置
input_channels = 1           # 单模态输入
n_classes = 3                # WT, TC, ET 三个分割子任务
n_stages = 6                 # 编码器 6 个阶段
n_features_per_stage = [8, 16, 32, 64, 80, 80]  # 轻量版（SimMLM 使用此配置）
# n_features_per_stage = [32, 64, 128, 256, 320, 320]  # 原始 nnUNet 配置（可选）
kernel_sizes = [[3, 3, 3]] * 6
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
apply_deep_supervision = False
```

---

## 三、模型结构详解

### 3.1 整体架构（Baseline Late Fusion Model）

```
输入: (B, 4, 128, 128, 128)   [4个MRI模态]
      按通道拆分为 4 个单模态输入
      │
      ├──── x_T1    = x[:, 0:1, ...]   (B, 1, 128, 128, 128)
      ├──── x_T1ce  = x[:, 1:2, ...]   (B, 1, 128, 128, 128)
      ├──── x_T2    = x[:, 2:3, ...]   (B, 1, 128, 128, 128)
      └──── x_FLAIR = x[:, 3:4, ...]   (B, 1, 128, 128, 128)
              │           │           │           │
              ▼           ▼           ▼           ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ Expert_0│ │ Expert_1│ │ Expert_2│ │ Expert_3│
         │ (UNet)  │ │ (UNet)  │ │ (UNet)  │ │ (UNet)  │
         │ T1      │ │ T1ce    │ │ T2      │ │ FLAIR   │
         └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
              │           │           │           │
              ▼           ▼           ▼           ▼
            o_T1        o_T1ce      o_T2       o_FLAIR
         (B,3,128³)  (B,3,128³)  (B,3,128³)  (B,3,128³)
              │           │           │           │
              └─────────┬─┴───────────┴─┬─────────┘
                        │               │
                        ▼               │
              ┌────────────────────┐    │
              │  静态等权平均融合   │◄───┘
              │  output = (o_T1 + o_T1ce + o_T2 + o_FLAIR) / 4
              └─────────┬──────────┘
                        │
                        ▼
              最终输出 (B, 3, 128, 128, 128)   [WT, TC, ET logits]
                        │
                        ▼
              sigmoid → 阈值 0.5 → 二值分割图
```

### 3.2 单个 Expert（UNet）内部结构

来源：`models/nnunet.py` 中的 `UNet` 类。4 个 Expert 结构完全相同，参数独立。

#### 编码器（Encoder）— 6 个阶段

```
输入: (B, 1, 128, 128, 128)   # 单模态 MRI

Stage 0: Conv3d(1→8, k=3, s=1, p=1) → IN → LeakyReLU → Conv3d(8→8, k=3, s=1, p=1) → IN → LeakyReLU
         输出: (B, 8, 128, 128, 128)   [空间尺寸不变]

Stage 1: Conv3d(8→16, k=3, s=2, p=1) → IN → LeakyReLU → Conv3d(16→16, k=3, s=1, p=1) → IN → LeakyReLU
         输出: (B, 16, 64, 64, 64)     [空间尺寸减半]

Stage 2: Conv3d(16→32, k=3, s=2, p=1) → IN → LeakyReLU → Conv3d(32→32, k=3, s=1, p=1) → IN → LeakyReLU
         输出: (B, 32, 32, 32, 32)

Stage 3: Conv3d(32→64, k=3, s=2, p=1) → IN → LeakyReLU → Conv3d(64→64, k=3, s=1, p=1) → IN → LeakyReLU
         输出: (B, 64, 16, 16, 16)

Stage 4: Conv3d(64→80, k=3, s=2, p=1) → IN → LeakyReLU → Conv3d(80→80, k=3, s=1, p=1) → IN → LeakyReLU
         输出: (B, 80, 8, 8, 8)

Stage 5: Conv3d(80→80, k=3, s=2, p=1) → IN → LeakyReLU → Conv3d(80→80, k=3, s=1, p=1) → IN → LeakyReLU
         输出: (B, 80, 4, 4, 4)        [瓶颈层/Bottleneck]
```

> 注：IN = InstanceNorm3d (affine=True)，LeakyReLU (inplace=True, negative_slope=0.01)

#### 解码器（Decoder）— 5 个阶段（含跳跃连接）

```
解码器从 Bottleneck (Stage 5) 开始，逐步上采样并与编码器的跳跃连接特征拼接：

Decoder Stage 0 (对应 Encoder Stage 5→4):
  上采样:   ConvTranspose3d(80→80, k=2, s=2)  → (B, 80, 8, 8, 8)
  跳跃连接: cat(上采样结果, Encoder_Stage4) → (B, 160, 8, 8, 8)
  解码块:   ConvTranspose3d(160→80, k=3, s=1, p=1) → IN → LeakyReLU
            → ConvTranspose3d(80→80, k=3, s=1, p=1) → IN → LeakyReLU
  分割头:   Conv3d(80→3, k=1, s=1)  → (B, 3, 8, 8, 8)  [低分辨率分割预测]

Decoder Stage 1 (对应 Encoder Stage 4→3):
  上采样:   ConvTranspose3d(80→64, k=2, s=2)  → (B, 64, 16, 16, 16)
  跳跃连接: cat(上采样结果, Encoder_Stage3)  → (B, 128, 16, 16, 16)
  解码块:   ConvTranspose3d(128→64, k=3, s=1, p=1) → IN → LeakyReLU
            → ConvTranspose3d(64→64, k=3, s=1, p=1) → IN → LeakyReLU
  分割头:   Conv3d(64→3, k=1, s=1)

Decoder Stage 2 (对应 Encoder Stage 3→2):
  上采样:   ConvTranspose3d(64→32, k=2, s=2)  → (B, 32, 32, 32, 32)
  跳跃连接: cat(上采样结果, Encoder_Stage2)  → (B, 64, 32, 32, 32)
  解码块:   ConvTranspose3d(64→32, k=3, s=1, p=1) → IN → LeakyReLU
            → ConvTranspose3d(32→32, k=3, s=1, p=1) → IN → LeakyReLU
  分割头:   Conv3d(32→3, k=1, s=1)

Decoder Stage 3 (对应 Encoder Stage 2→1):
  上采样:   ConvTranspose3d(32→16, k=2, s=2)  → (B, 16, 64, 64, 64)
  跳跃连接: cat(上采样结果, Encoder_Stage1)  → (B, 32, 64, 64, 64)
  解码块:   ConvTranspose3d(32→16, k=3, s=1, p=1) → IN → LeakyReLU
            → ConvTranspose3d(16→16, k=3, s=1, p=1) → IN → LeakyReLU
  分割头:   Conv3d(16→3, k=1, s=1)

Decoder Stage 4 (对应 Encoder Stage 1→0):  ← 最终输出层
  上采样:   ConvTranspose3d(16→8, k=2, s=2)   → (B, 8, 128, 128, 128)
  跳跃连接: cat(上采样结果, Encoder_Stage0)   → (B, 16, 128, 128, 128)
  解码块:   ConvTranspose3d(16→8, k=3, s=1, p=1) → IN → LeakyReLU
            → ConvTranspose3d(8→8, k=3, s=1, p=1) → IN → LeakyReLU
  分割头:   Conv3d(8→3, k=1, s=1)  → (B, 3, 128, 128, 128)  [最终分割预测 logits]
```

> **注意**: `apply_deep_supervision=False` 时，只取最后一层解码器 (Dec4) 的输出作为最终预测。

### 3.3 单个 Expert 数据流总结图

```
单模态输入 (B, 1, 128, 128, 128)
     │
     ▼
┌─ Encoder ────────────────────────────────────────────────────────┐
│  Stage0: (B,1,128³) → (B,8,128³)   ──────────────── skip_0 ─┐  │
│  Stage1: (B,8,128³) → (B,16,64³)   ──────────────── skip_1 ─┤  │
│  Stage2: (B,16,64³) → (B,32,32³)   ──────────────── skip_2 ─┤  │
│  Stage3: (B,32,32³) → (B,64,16³)   ──────────────── skip_3 ─┤  │
│  Stage4: (B,64,16³) → (B,80,8³)    ──────────────── skip_4 ─┤  │
│  Stage5: (B,80,8³)  → (B,80,4³)    [Bottleneck]             │  │
└──────────────────────────────────────────────────────────────────┘
     │                                                          │
     ▼                                                          │
┌─ Decoder ────────────────────────────────────────────────────────┐
│  Dec0: Upsample(80→80) + cat(skip_4) → decode → seg_head_0     │
│  Dec1: Upsample(80→64) + cat(skip_3) → decode → seg_head_1     │
│  Dec2: Upsample(64→32) + cat(skip_2) → decode → seg_head_2     │
│  Dec3: Upsample(32→16) + cat(skip_1) → decode → seg_head_3     │
│  Dec4: Upsample(16→8)  + cat(skip_0) → decode → seg_head_4     │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
Expert 输出 (B, 3, 128, 128, 128)   [该模态的 WT, TC, ET logits]
```

---

## 四、融合策略（Baseline 静态等权平均）

### 4.1 融合公式

```python
# 4 个 Expert 各自输出 logits
o_0 = expert_0(x[:, 0:1, ...])  # T1    → (B, 3, 128, 128, 128)
o_1 = expert_1(x[:, 1:2, ...])  # T1ce  → (B, 3, 128, 128, 128)
o_2 = expert_2(x[:, 2:3, ...])  # T2    → (B, 3, 128, 128, 128)
o_3 = expert_3(x[:, 3:4, ...])  # FLAIR → (B, 3, 128, 128, 128)

# 静态等权平均融合（在 logit 空间，sigmoid 之前）
output = (o_0 + o_1 + o_2 + o_3) / 4.0
```

### 4.2 与 SimMLM 动态 Router 的对比

| 维度 | Baseline 静态平均 | SimMLM 动态 Router |
|------|-------------------|-------------------|
| 权重 | 固定 `[0.25, 0.25, 0.25, 0.25]` | 由 Router 网络动态生成，逐样本、逐任务变化 |
| 参数 | 无额外参数 | Router 增加约 0.005M 参数（5层CNN + Linear） |
| 可学习性 | 不可学习 | 可学习（softmax 归一化后的权重） |
| 缺失模态适应 | 不支持 | 支持（缺失模态权重设为 -inf → softmax 后为 0） |

### 4.3 Baseline Forward 伪代码

```python
class BaselineLateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 个独立的单通道 UNet Expert
        self.expert_ls = nn.ModuleList([
            UNet(input_channels=1, n_classes=3, n_stages=6,
                 n_features_per_stage=[8, 16, 32, 64, 80, 80],
                 kernel_size=[[3,3,3]]*6,
                 strides=[[1,1,1], *[[2,2,2]]*5])
            for _ in range(4)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, 4, 128, 128, 128) — 4 个 MRI 模态
        Returns:
            output: (B, 3, 128, 128, 128) — 融合后的分割 logits
        """
        expert_outputs = []
        for modality_idx in range(4):
            # 提取单模态: (B, 1, D, H, W)
            x_m = x[:, modality_idx:modality_idx+1, ...]
            # Expert 独立前向传播
            o_m = self.expert_ls[modality_idx](x_m)  # (B, 3, D, H, W)
            expert_outputs.append(o_m)

        # 静态等权平均融合（logit 空间）
        output = torch.stack(expert_outputs, dim=0).mean(dim=0)  # (B, 3, D, H, W)
        return output
```

### 4.4 为什么在 logit 空间融合

论文 Supplementary Figure A5 和 Table A8 对比了三种融合位置：

| 融合位置 | 说明 | ET Dice | TC Dice | WT Dice |
|----------|------|---------|---------|---------|
| Feature-level | 解码器特征图加权融合 → 共享分割头 | 62.76 | 78.19 | 87.27 |
| **Logit-level** | 各 Expert 输出 logits 加权融合 | **63.22** | **78.54** | 87.21 |
| Probability-level | 各 Expert sigmoid 概率加权融合 | 62.72 | 78.31 | **87.33** |

Logit-level 融合在 ET 和 TC 两个更难的子任务上表现最优，且最简洁。Baseline 沿用此设计。

---

## 五、损失函数

### 5.1 损失计算

损失施加在**融合后的输出**上：

```python
Loss = Dice Loss + Binary Cross Entropy Loss

# 具体实现:
output = model(x)                              # 融合后的 logits (B, 3, D, H, W)
probs = sigmoid(output)                        # logits → 概率
dice_loss = DiceLoss(sigmoid=False)(probs, labels)   # Dice 损失 (monai)
bce_loss  = BCELoss()(probs, labels.float())         # BCE 损失

total_loss = dice_loss.mean() + bce_loss.mean()
```

### 5.2 梯度回传路径

由于是静态平均融合，梯度会**自然地**回传到每个 Expert：

```
total_loss
    │
    ▼ backward
output = (o_0 + o_1 + o_2 + o_3) / 4
    │
    ├── ∂loss/∂o_0 = ∂loss/∂output × (1/4)  ──→ Expert_0 (T1)    参数更新
    ├── ∂loss/∂o_1 = ∂loss/∂output × (1/4)  ──→ Expert_1 (T1ce)  参数更新
    ├── ∂loss/∂o_2 = ∂loss/∂output × (1/4)  ──→ Expert_2 (T2)    参数更新
    └── ∂loss/∂o_3 = ∂loss/∂output × (1/4)  ──→ Expert_3 (FLAIR) 参数更新
```

这意味着你可以在每个 Expert 的编码器/解码器上注册 hook，**独立监控各模态的梯度范数**。

**来源**: `loss/dice_bce_loss.py` 中的 `DiceBCEWithLogitsLoss` 类（可直接复用）

---

## 六、数据集处理

### 6.1 数据格式
- 数据路径: `/root/autodl-tmp/datasets/nnUNet_preprocessed_BraTS2018/`
- 数据格式: `.npy` 文件（nnUNet 预处理后）
  - 图像: `{sample_id}.npy` → shape: `(4, D, H, W)`，4 通道对应 T1, T1ce, T2, FLAIR
  - 标签: `{sample_id}_seg.npy` → shape: `(1, D, H, W)`
- 数据划分: `assets/kfold_splits.json`（5折交叉验证，使用 fold 0）
  - 训练集/验证集比例 4:1

### 6.2 标签转换
原始 BraTS 标签 → 三通道二值掩码：
```python
# 原始标签: 0=背景, 1=坏死核, 2=水肿, 3=增强肿瘤
# 转换为三个子任务的二值掩码:
WT (whole tumor):     label ∈ {1, 2, 3}   → 1, 否则 → 0
TC (tumor core):      label ∈ {2, 3}       → 1, 否则 → 0
ET (enhancing tumor): label == 3           → 1, 否则 → 0
```
**注意**: 代码中有 label 值修正逻辑（`label[label==0]=-2; label[label==-1]=0`），这是因为 nnUNet 预处理将背景存为 -1，需要转换回来。

### 6.3 数据增强
- **训练时**:
  1. `SpatialPadd` → 空间填充至 128³（对称模式）
  2. `RandSpatialCropd` → 随机裁剪 128³
  3. `RandFlipd` × 3 → 三轴随机翻转 (prob=0.5)
  4. `RandGaussianNoised` → 高斯噪声 (prob=0.15)
  5. `RandGaussianSmoothd` → 高斯模糊 (prob=0.15)
  6. `RandAdjustContrastd` → 随机对比度 (prob=0.15)
  7. `RandScaleIntensityd` → 随机强度缩放 (prob=0.15)

- **验证时**:
  1. `CenterSpatialCropd` → 中心裁剪 128³
  2. `SpatialPadd` → 填充至 128³

### 6.4 Baseline 数据集配置
```python
# Baseline 使用 SingleStreamDataset，关键配置：
drop_mode = None        # 不丢弃任何模态（始终使用全部4个模态）
unimodality = False     # 不使用单模态模式（保留 4 通道输出，在模型内部拆分）
# 输出: sample['img'] shape = (4, 128, 128, 128)
```

---

## 七、训练配置

```python
# 优化器: 所有4个Expert的参数统一优化
optimizer = Adam(params=model.parameters(), lr=0.01)

# 训练参数
batch_size = 4          # 训练
val_batch_size = 8      # 验证
num_epochs = 300        # 训练轮数
val_freq = 5            # 每 5 个 epoch 验证一次
random_seed = 12345

# 评估指标
metric = DiceHelper(include_background=True, sigmoid=True, activate=True)
# 保存最优模型（按 val_loss 最小）
```

---

## 八、可选的梯度/指标监控扩展点

Late Fusion 架构天然支持逐模态独立监控。以下是可以扩展的监控点（不属于 baseline 本身，但 baseline 架构已为此预留了结构基础）：

### 8.1 各模态梯度范数监控

```python
# 通过注册 hook 监控各 Expert 编码器的梯度范数
for k, expert in enumerate(model.expert_ls):
    def make_hook(idx):
        def hook(module, grad_input, grad_output):
            grad_norm = grad_output[0].norm().item()
            # 记录 Expert_{idx} 的梯度范数
        return hook
    expert.encoder_stages[0].register_backward_hook(make_hook(k))
```

### 8.2 各模态输出置信度监控

```python
# 在 forward 中，可分别获取各 Expert 的输出
expert_outputs = []
for modality_idx in range(4):
    o_m = model.expert_ls[modality_idx](x[:, modality_idx:modality_idx+1, ...])
    expert_outputs.append(o_m)
    # 计算该模态的置信度: sigmoid(o_m).max() 或 entropy
    confidence_m = torch.sigmoid(o_m).mean()
```

### 8.3 各模态梯度范数比值

```python
# 参考 SimMLM 中的 grad_encoder_min_over_max 指标
# 比值越接近 1，说明各模态的梯度越均衡
grad_norms = [grad_norm_expert_0, grad_norm_expert_1, grad_norm_expert_2, grad_norm_expert_3]
balance_ratio = min(grad_norms) / (max(grad_norms) + 1e-8)
```

---

## 九、需要从 SimMLM 项目复制的文件

### 9.1 可直接复用的文件

| 文件 | 用途 | 修改说明 |
|------|------|----------|
| `models/nnunet.py` | 单个 UNet Expert 的模型定义 | **直接复用**，`input_channels=1` |
| `loss/dice_bce_loss.py` | Dice+BCE 损失函数 | **直接复用**，无需修改 |
| `dataset/processors.py` | 数据集类 | **仅复用 `SingleStreamDataset` 和 `BratsEvalSet`**，删除 `PairedDataset` |
| `dataset/utils.py` | 数据预处理工具 | **直接复用** |
| `assets/kfold_splits.json` | 数据划分 | **直接复用** |
| `train/trainer_expert_pretraining.py` | 基础训练循环 | **参考复用**，需修改 step 中的模型调用方式 |

### 9.2 需要新建/重写的文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `models/baseline.py` | Baseline Late Fusion 模型 | 封装 4 个 UNet Expert + 静态平均融合（参考第四节伪代码） |
| `configs.py` | Baseline 配置 | UNet 参数、数据集路径、训练超参 |
| `pipeline.py` | 训练入口 | 单阶段端到端训练 |
| `train/trainer.py` | 训练循环 | 基于 `trainer_expert_pretraining.py` 修改，适配 Late Fusion 模型 |

### 9.3 完全不需要的文件（SimMLM 创新点）

| 文件 | 原因 |
|------|------|
| `models/dmome.py` | 包含 Router/Gating 网络、缺失模态处理、梯度监控 hook |
| `models/momke.py` | MoMKE 对比模型 |
| `loss/blended_loss.py` | 包含 MoFe 排序损失 |
| `pipeline_joint_training.py` | DMoME 两阶段联合训练流程 |
| `pipeline_expert_pretraining.py` | 单模态 Expert 预训练流程（baseline 不预训练） |
| `train/trainer_joint_training.py` | DMoME 联合训练器 |
| `train/log_utils.py` | DMoME 专用日志工具（可选择性复用梯度绘图功能） |
| `configs_joint_training.py` | DMoME 联合训练配置 |
| `configs_expert_pretraining.py` | 单模态 Expert 预训练配置 |

---

## 十、Baseline 项目文件结构

```
baseline-unet/
├── configs.py                          # 配置文件
├── pipeline.py                         # 训练入口
├── models/
│   ├── nnunet.py                       # 单个 UNet Expert 定义（从 SimMLM 复制，input_channels=1）
│   └── baseline.py                     # BaselineLateFusion 模型（新建：4 个 Expert + 静态平均）
├── dataset/
│   ├── processors.py                   # SingleStreamDataset（从 SimMLM 复制，删除 PairedDataset）
│   └── utils.py                        # 数据预处理工具（从 SimMLM 复制）
├── loss/
│   └── dice_bce_loss.py                # Dice+BCE 损失（从 SimMLM 复制）
├── train/
│   └── trainer.py                      # 训练循环（基于 trainer_expert_pretraining.py 修改）
└── assets/
    └── kfold_splits.json               # 数据划分文件（从 SimMLM 复制）
```

---

## 十一、Baseline 与 SimMLM 的关键差异总结

| 维度 | Baseline (Late Fusion) | SimMLM |
|------|------------------------|--------|
| **模型架构** | 4 个独立单通道 UNet Expert | 4 个独立单通道 UNet Expert + **Router 网络** |
| **输入方式** | 逐模态拆分 (B,1,D,H,W) × 4 | 逐模态拆分 (B,1,D,H,W) × 4 |
| **融合方式** | **静态等权平均**（无可学习参数） | **动态加权**（Router 网络输出 softmax 权重） |
| **损失函数** | Dice + BCE | Dice + BCE **+ MoFe 排序损失** |
| **训练策略** | **单阶段**端到端训练（从头训练） | **两阶段**（Expert 预训练 + 联合训练） |
| **缺失模态** | 不处理（始终使用全部 4 模态） | 动态适应（Router 掩码 + 权重重分配） |
| **训练数据** | SingleStreamDataset（无模态丢弃） | PairedDataset（配对 + 随机丢弃） |
| **参数量** | ~7.8M（4 × ~1.95M UNet） | ~7.8M（4 × ~1.95M UNet + ~0.005M Router） |
| **逐模态监控** | 天然支持（各 Expert 独立） | 支持（但与 Router 耦合） |

---

## 十二、参考：SimMLM 代码中各功能定义的精确位置

便于后续复制代码时快速定位：

| 功能 | 文件路径 | 类/函数 | 行号 |
|------|----------|---------|------|
| UNet 模型定义 | `models/nnunet.py` | `UNet` 类 | 7-157 |
| UNet 编码器构建 | `models/nnunet.py` | `UNet._build_encoder()` | 24-56 |
| UNet 解码器构建 | `models/nnunet.py` | `UNet._build_decoder()` | 58-115 |
| UNet 前向传播 | `models/nnunet.py` | `UNet.forward()` | 117-143 |
| Dice+BCE 损失 | `loss/dice_bce_loss.py` | `DiceBCEWithLogitsLoss` 类 | 7-30 |
| 数据集定义 | `dataset/processors.py` | `SingleStreamDataset` 类 | 20-132 |
| 标签转换 | `dataset/processors.py` | `SingleStreamDataset._transform_label()` | 84-91 |
| 数据增强 | `dataset/processors.py` | `SingleStreamDataset._get_sample_transforms()` | 48-82 |
| BraTS 评估集 | `dataset/processors.py` | `BratsEvalSet` 类 | 136-186 |
| 归一化工具 | `dataset/utils.py` | `zero_mean_unit_variance_normalization()` | - |
| 训练循环（参考） | `train/trainer_expert_pretraining.py` | `StepRunner`, `EpochRunner`, `train_model()` | 21-197 |
| 数据划分 | `assets/kfold_splits.json` | fold 0 的 train/val 列表 | - |
