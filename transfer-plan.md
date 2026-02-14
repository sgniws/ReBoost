# 迁移方案：将 Sustained Boosting (AUG) 从分类任务迁移到医学图像分割

> 参考论文：《Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion》（NeurIPS 2025）
> 目标任务：BraTS2018 四模态（T1, T1ce, T2, FLAIR）脑肿瘤分割
> 基础架构：Baseline Late Fusion（4 个独立 3D UNet Expert，静态等权平均融合）

---

## 一、论文核心方法理解

### 1.1 问题定义

多模态学习中存在**模态不平衡（Modality Imbalance）**问题：不同模态在联合训练中收敛速度不同，强模态（如音频在音频-视频任务中）趋于更快收敛、获得更强的分类器，而弱模态（如视频）则被欠优化。现有方法主要从**平衡学习过程**（如梯度调制 OGM、学习率调整 MSLR）的角度解决，但未直接增强弱模态的分类能力。

### 1.2 核心思想

本文另辟蹊径，从**平衡分类能力（Classification Ability）**的角度出发：

- 不是减慢强模态或加速弱模态的学习，而是**直接为弱模态增加分类能力**
- 借鉴 **Boosting（提升）** 的核心思想：通过组合多个弱分类器构建强分类器
- 为弱模态动态添加多个轻量级分类器，每个新分类器专注于前序分类器的残差（即"学不好的部分"）

### 1.3 关键组件

#### 1.3.1 Sustained Boosting（持续提升算法）

**与传统 Gradient Boosting 的区别**：传统 GB 顺序训练分类器（先训练第 1 个，固定后训练第 2 个...）。本文的"持续提升"则是**同时端到端优化所有分类器**，通过三个损失项实现：

**损失设计**（对于模态 o，当前有 t 个分类器）：

1. **残差损失 ε**：第 t 个（最新）分类器在残差标签上的损失
   ```
   ε(x, y, t) = ℓ(p_t, ŷ_t)
   ```
   其中残差标签：`ŷ_t = y - λ · Σ_{j=1}^{t-1} (y ⊙ p_j)`
   - λ ∈ [0,1] 是平滑系数
   - ⊙ 表示逐元素乘积
   - y 用于掩码，确保残差标签非负

2. **全体分类器联合损失 ε_all**：所有 t 个分类器输出之和在原始标签上的损失
   ```
   ε_all(x, y, t) = ℓ(Σ_{j=1}^{t} p_j, y)
   ```
   确保所有分类器的集成预测整体性能良好。

3. **前序分类器损失 ε_pre**：前 t-1 个分类器输出之和在原始标签上的损失
   ```
   ε_pre(x, y, t) = ℓ(Σ_{j=1}^{t-1} p_j, y)
   ```
   防止共享编码器更新时导致已有分类器性能退化。

**总损失**：

```
L_SUB(o) = (1/N) Σ_i [ε(x_i, y_i, n_o) + ε_all(x_i, y_i, n_o) + ε_pre(x_i, y_i, n_o)]
```

训练时，所有模态的 L_SUB 求和作为最终优化目标：

```
Total_Loss = Σ_{o} L_SUB(o)
```

**为什么需要三个损失？**
- ε 教导新分类器学习残差（梯度提升的核心）
- ε_all 保证集成预测质量
- ε_pre 保护已有分类器不退化（因为编码器是共享的，更新会影响所有分类器）

#### 1.3.2 可配置分类器（Configurable Classifier）

**架构**：
```
Layer1(D × 256) → ReLU → Layer2(256 × K)
```
- D 是编码器输出特征维度，K 是类别数
- **Layer1 是每个分类器头私有的**（各有各的参数）
- **Layer2 是所有模态、所有分类器头共享的（Shared Head）**

**Shared Head 的作用**：
- 所有模态和分类器共享最后的投影层，强制它们在同一语义空间中产生预测
- 促进跨模态交互：弱模态和强模态的梯度都流经同一个 Shared Head
- 这是 MML 中常见的设计（MLA、DI-MML 等也使用）

**每个模态的分类器数量 n_o 是独立的**：弱模态有更多分类器，强模态可能只有 1 个。

#### 1.3.3 自适应分类器分配策略（Adaptive Classifier Assignment, ACA）

**动机**：训练过程中，模态的强弱关系可能动态变化，因此需要动态决定为哪个模态添加分类器。

**置信度分数（Confidence Score）**：
```
s^o_t = (1/N) Σ_i y_i^T · (Σ_{j=1}^{n_o} p^o_ij)
```
即：对于每个样本，计算真实类别上的联合预测概率，再对所有样本取平均。置信度越高，说明该模态的分类能力越强。

**分配规则**（以双模态 a, v 为例）：
每 t_N 个 epoch 检查一次：
- 若 `s^a - σ · s^v > τ`：音频模态更强 → 为视频模态添加一个分类器，`n_v += 1`
- 若 `s^a - σ · s^v < -τ`：视频模态更强 → 为音频模态添加一个分类器，`n_a += 1`
- 否则：不操作（在死区 τ 范围内）

**超参数**：
- σ ≥ 1：系数，控制判定的严格程度（默认 1.0）
- τ：容忍阈值/死区（默认 0.01）
- t_N：检查频率（以 epoch 为单位）

### 1.4 完整训练流程（Algorithm 1）

```
初始化：n_a = 1, n_v = 1，初始化所有 DNN 参数

for t = 1 → T (总迭代数) do:
    1. 采样 mini-batch
    2. 提取各模态特征：u^a = φ^a(x^a), u^v = φ^v(x^v)
    3. 各模态通过各自的所有分类器头得到预测 {p^a_j}, {p^v_j}
    4. 计算 L_SUB 损失（公式 6）
    5. SGD 更新所有参数
    
    if mod(t, t_N) == 0:  // ACA 检查
        6. 计算置信度 s^a, s^v
        7. 根据置信度差距决定是否添加分类器
end for
```

### 1.5 理论保证

论文证明了 Gap 函数 G(Φ) = L^a(Φ^a) - L^v(Φ^v) 在 Sustained Boosting 下以 **O(1/T)** 速率收敛，即模态间的损失差距随训练进行逐渐缩小。

### 1.6 关键实验发现

- 在 CREMAD 数据集上，方法将多模态准确率从 65.07%（Naive MML）提升至 85.15%
- 自适应策略优于固定分类器数量（85.15% vs 81.18%）
- 额外参数量仅增加约 1M（原模型 23.6M），约 4%
- 训练和推理时间增加有限（训练 ~1.98hrs vs 1.68hrs，推理 5.62s vs 5.29s）
- 方法对超参数 σ 和 λ 不敏感

---

## 二、从分类到分割的迁移分析

### 2.1 核心映射关系

| 维度 | 分类任务（论文原始） | 分割任务（本项目） |
|------|---------------------|-------------------|
| **编码器 φ^o(·)** | ResNet18 特征提取器 | UNet Encoder + Decoder（输出 8 通道特征图） |
| **分类器 ψ^o(·)** | FC 层: Layer1(D→256)→ReLU→Layer2(256→K) | 分割头: Conv3d(8→C_h, k=1)→ReLU→Conv3d(C_h→3, k=1) |
| **输出 p^o** | softmax 概率向量 (K,) | sigmoid 概率体 (3, 128, 128, 128) |
| **标签 y** | one-hot 向量 y ∈ {0,1}^K | 二值掩码 y ∈ {0,1}^{3×128×128×128} |
| **损失函数** | Cross-Entropy | Dice + BCE（分割标准损失） |
| **置信度** | y^T · p（真实类上的预测概率） | mean(y ⊙ σ(logit)) 前景体素上的平均预测概率 |
| **模态数** | 2（音频 + 视频） | 4（T1, T1ce, T2, FLAIR） |
| **类别数 K** | 数据集类别数（如 6） | 3（WT, TC, ET）逐体素二值分割 |

### 2.2 迁移设计详述

#### 2.2.1 特征提取器（对应论文中的"编码器"）

在本项目的 Baseline 中，每个模态对应一个完整的 UNet（Encoder + Decoder + seg_head）。在迁移中：

- **"编码器"= UNet Encoder（6 个阶段）+ UNet Decoder（5 个阶段，不含最终 seg_layer）**
- 输出：Decoder 最后一个阶段的特征图，shape = `(B, 8, 128, 128, 128)`
- **模态内共享**：同一模态的所有分类器头共享同一个 Encoder+Decoder
- **模态间独立**：4 个模态各有独立的 Encoder+Decoder

**为什么把 Decoder 也算作"编码器"？**
- 在分类中，编码器输出是 1D 特征向量，分类器只需 FC 层
- 在分割中，需要产生与输入同分辨率的预测图，UNet 的 Decoder 负责上采样和空间恢复
- 将 Encoder+Decoder 视为"特征提取管线"，其输出的 8 通道特征图等价于分类任务中的编码器特征向量

#### 2.2.2 可配置分类器（Segmentation Head）

**设计**：

```
私有部分（每个分类器头独有）:
    Conv3d(8 → C_head, kernel_size=1) → ReLU

共享部分（所有模态、所有头共享，对应 Layer2）:
    Conv3d(C_head → 3, kernel_size=1)
```

- `C_head`：中间通道数，推荐 8（轻量，与论文的 Layer1 对应）
- **私有 Conv3d + ReLU** 相当于论文的 `Layer1(D×256) → ReLU`
- **共享 Conv3d** 相当于论文的 `Layer2(256×K)`
- 使用 1×1 卷积（而非 3×3）以控制显存和计算量

**为什么使用 1×1 而非 3×3 卷积？**
- 3D 体积数据中，3×3×3 卷积的激活值显存是 1×1×1 的 27 倍
- 空间上下文已由 UNet Decoder 充分处理，分割头只需做特征到类别的映射
- 论文中的 FC 层本质上就是"没有空间卷积"的 1×1 操作

#### 2.2.3 残差标签的分割适配

**分类原始公式**：
```
ŷ_it = y_i - λ · Σ_{j=1}^{t-1} (y_i ⊙ p_ij)
```

**分割适配**：
```
ŷ_it = clamp(y_i - λ · Σ_{j=1}^{t-1} (y_i ⊙ σ(logit^o_ij)), min=0)
```

- `y_i ∈ {0,1}^{3×D×H×W}`：三通道二值分割掩码
- `σ(logit^o_ij) ∈ (0,1)^{3×D×H×W}`：第 j 个分割头的 sigmoid 输出
- `y_i ⊙ σ(logit^o_ij)`：只保留前景体素处的预测概率，背景处为 0
- `clamp(..., min=0)`：确保残差标签非负（当多个分类器的累积预测较大时）

**物理意义**：
- 残差标签表示"前序分类器尚未充分预测的部分"
- 在前景体素中，若前序分类器已经高置信度预测正确（σ 接近 1），则残差接近 0 → 新分类器无需在此处用力
- 在前景体素中，若前序分类器预测不好（σ 接近 0），则残差接近 y → 新分类器需要在此处努力
- 背景体素的残差始终为 0

#### 2.2.4 损失函数适配

对于模态 o，当前有 n_o 个分割头：

**1. 残差损失 ε（使用残差标签，仅 BCE）**：
```
ε = BCE(σ(logit_{n_o}), ŷ_{n_o})
```
- 只使用 BCE，不使用 Dice Loss
- 原因：残差标签 ŷ 是连续软标签 ∈ [0,1]，Dice Loss 对软标签的行为不稳定
- BCE 天然支持软标签，数学意义明确

**2. 联合预测损失 ε_all（使用原始标签，Dice + BCE）**：
```
combined_logit = Σ_{j=1}^{n_o} logit_j
ε_all = DiceBCE(combined_logit, y)
```
- 所有分类器头的 logits 求和作为联合预测
- 在 logit 空间求和（而非 sigmoid 后），与 Baseline 融合方式一致
- 使用标准 DiceBCE 损失（原始硬标签）

**3. 前序预测损失 ε_pre（使用原始标签，Dice + BCE）**：
```
pre_logit = Σ_{j=1}^{n_o - 1} logit_j
ε_pre = DiceBCE(pre_logit, y)
```
- 当 n_o = 1 时，ε_pre = 0（没有前序分类器）

**模态总损失**：
```
L_SUB(o) = ε + ε_all + ε_pre
```

**所有模态的最终训练损失**：
```
Total_Loss = Σ_{o ∈ {T1, T1ce, T2, FLAIR}} L_SUB(o)
```

#### 2.2.5 置信度分数的分割适配

**分类原始公式**：
```
s^o = (1/N) Σ_i y_i^T · (Σ_j p^o_j)
```

**分割适配**：
```
combined_prob = σ(Σ_j logit^o_j)               # (B, 3, D, H, W)
s^o = (1/N) Σ_i mean_spatial(y_i ⊙ combined_prob_i)
```

即：计算联合预测在前景体素处的平均 sigmoid 概率。置信度越高 → 该模态的分割能力越强。

**注意**：为避免零除，当某个样本的前景体素数为 0 时，该样本对置信度的贡献为 0。

#### 2.2.6 ACA 策略的 4 模态扩展

论文原始方案是双模态比较。扩展到 4 模态：

```
每 t_N 个 epoch：
    1. 计算 4 个模态的置信度：s_T1, s_T1ce, s_T2, s_FLAIR
    2. s_max = max(s_T1, s_T1ce, s_T2, s_FLAIR)
    3. s_min = min(s_T1, s_T1ce, s_T2, s_FLAIR)
    4. weak_modality = argmin(s_T1, s_T1ce, s_T2, s_FLAIR)
    5. if s_max - σ · s_min > τ:
           n_{weak_modality} += 1  // 为最弱模态增加一个分割头
```

**扩展说明**：
- 原论文每次只比较两个模态，此处比较所有 4 个模态的最强与最弱
- 每次最多只为一个模态（最弱的）添加分类器，保持方法的渐进性
- 若多个模态同样弱，选择置信度最低的那个

#### 2.2.7 融合策略

**模态内融合（Boosting 预测聚合）**：
```
pred_o = Σ_{j=1}^{n_o} logit^o_j      # 同一模态所有头的 logits 求和
```

**模态间融合（Late Fusion）**：
```
output = (pred_T1 + pred_T1ce + pred_T2 + pred_FLAIR) / 4
```

保持 Baseline 的等权平均融合方式不变，仅在模态内增加 Boosting 机制。

### 2.3 医学图像分割领域的潜在问题

#### 问题 1：3D 体积数据的显存压力

**描述**：分类任务的特征向量通常是 1D 的（如 512 维），而分割任务的特征图是 3D 的（8×128³ = 16M 元素/头）。每增加一个分割头，训练时需要额外存储其前向激活值和反向梯度。

**估算**：
- 每个分割头的中间激活：`(B=4, C_head=8, 128, 128, 128)` ≈ 268 MB
- 加上梯度 ≈ 536 MB/头
- 若 4 个模态各有 5 个头 = 20 头 → 额外 ~10.7 GB
- Baseline 显存 18 GB + 额外 10.7 GB = ~29 GB（48 GB 显存可承受）

**结论**：在 C_head=8、最大 10 头/模态的设定下，显存可控。

#### 问题 2：严重的类别不平衡

**描述**：BraTS 数据集中，肿瘤区域远小于背景区域（ET 尤其小）。Boosting 的残差标签机制可能受到类别不平衡的影响——背景体素的残差始终为 0，前景体素的残差才有意义。

**影响**：残差损失 ε 的 BCE 可能被大量的"0 标签"背景体素主导，使新分类器无法有效学习前景的残差。

#### 问题 3：残差标签与 Dice Loss 的不兼容

**描述**：Dice Loss 设计用于硬标签（0/1）。残差标签 ŷ 是连续软标签 ∈ [0,1]，将其直接用于 Dice Loss 可能导致不稳定。

**当前方案**：残差损失 ε 仅使用 BCE（已在 2.2.4 中说明）。但若希望也使用 Dice，需要特殊处理。

#### 问题 4：置信度分数的稳定性

**描述**：分割任务中，前景区域大小在不同样本间差异极大（有的样本肿瘤很大，有的很小）。基于前景概率的置信度分数可能波动较大，导致 ACA 策略不稳定（频繁、错误地添加分类器）。

#### 问题 5：多分割头预测的空间一致性

**描述**：多个轻量级分割头独立预测后求和，可能在空间上产生不一致的分割边界。分类任务不存在这个问题，因为输出是全局标签。

### 2.4 解决思路

#### 解决问题 1（显存）：
- 使用 1×1 卷积而非 3×3 卷积
- 设置最大分类器数量上限（如 max_heads=10）
- 可选：对分割头使用梯度检查点（gradient checkpointing）

#### 解决问题 2（类别不平衡）：
- 在残差损失 BCE 中引入前景掩码权重
- 或使用 Focal Loss 替代 BCE 作为残差损失
- 计算残差损失时，仅在前景区域（y=1）计算 BCE

#### 解决问题 3（残差 + Dice）：
- 保持当前设计：残差损失仅用 BCE
- 如需混合，可对残差标签进行硬化（阈值化）后再用 Dice

#### 解决问题 4（置信度稳定性）：
- 使用指数移动平均（EMA）平滑置信度分数
- 或使用 Dice 系数替代前景概率作为置信度指标
- 增大死区 τ 以减少误触发

#### 解决问题 5（空间一致性）：
- 在 ε_all 损失中，联合预测（logits 求和后）自然会产生空间一致的预测
- ε_all 和 ε_pre 的梯度会引导各头产生互补且一致的预测

---

## 三、代码实践需求说明书

> **说明**：本节仅包含论文方法的核心迁移方案，不包含第二节中提到的医学图像分割领域特有问题的解决方案。这些解决方案将在后续实验中根据需要逐步引入。

### 3.1 整体架构

```
输入: (B, 4, 128, 128, 128)   [4个MRI模态]
      按通道拆分为 4 个单模态输入
      │
      ├──── x_T1    = x[:, 0:1, ...]
      ├──── x_T1ce  = x[:, 1:2, ...]
      ├──── x_T2    = x[:, 2:3, ...]
      └──── x_FLAIR = x[:, 3:4, ...]
              │           │           │           │
              ▼           ▼           ▼           ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
       │ Encoder_0 │ │ Encoder_1 │ │ Encoder_2 │ │ Encoder_3 │
       │ Decoder_0 │ │ Decoder_1 │ │ Decoder_2 │ │ Decoder_3 │
       │ (共享UNet)│ │ (共享UNet)│ │ (共享UNet)│ │ (共享UNet)│
       └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
             │              │              │              │
        feat (B,8,128³) feat (B,8,128³) feat (B,8,128³) feat (B,8,128³)
             │              │              │              │
        ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
        │ Head_0_1│    │ Head_1_1│    │ Head_2_1│    │ Head_3_1│
        │ Head_0_2│    │ (n_1个) │    │ (n_2个) │    │ (n_3个) │
        │ ...     │    │ ...     │    │ ...     │    │ ...     │
        │(n_0个)  │    │         │    │         │    │         │
        └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │              │
             ▼              ▼              ▼              ▼
        Σ logits_0     Σ logits_1     Σ logits_2     Σ logits_3
        (B,3,128³)     (B,3,128³)     (B,3,128³)     (B,3,128³)
             │              │              │              │
             └──────┬───────┴──────────────┴──────┬──────┘
                    │        等权平均融合           │
                    ▼                              │
             output = mean(Σ logits_0, ..., Σ logits_3)
                    │
                    ▼
          最终输出 (B, 3, 128, 128, 128)
```

### 3.2 需要修改/新建的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `models/nnunet.py` | **修改** | UNet 增加返回解码器特征的方法（不含 seg_layer） |
| `models/boosted_fusion.py` | **新建** | BoostedLateFusion 模型：包含 UNet 骨干 + 可配置分割头 + Shared Head |
| `loss/boosted_loss.py` | **新建** | SustainedBoostingLoss：计算 ε + ε_all + ε_pre |
| `train/trainer_boosted.py` | **新建** | 训练器：支持 ACA 策略、置信度监控、动态添加分割头 |
| `configs_boosted.py` | **新建** | Boosted 方法的超参数配置 |
| `pipeline_boosted.py` | **新建** | 训练入口脚本 |

### 3.3 UNet 修改：返回解码器特征

在 `models/nnunet.py` 的 `UNet.forward()` 方法中，增加一个选项，使其可以返回解码器最终阶段的特征图（不经过 seg_layer）：

```python
def forward(self, x, return_features=False):
    """
    Args:
        x: (B, 1, D, H, W)
        return_features: 若为 True，同时返回解码器最终特征图
    Returns:
        若 return_features=False: logits (B, 3, D, H, W)
        若 return_features=True:  (logits, decoder_features)
            decoder_features: (B, 8, D, H, W) — 最后一个解码阶段的输出（未经 seg_layer）
    """
    # ... encoder ...
    # ... decoder loop ...
    # 在最后一个 decoder stage 后，output 是 (B, 8, D, H, W)
    # seg_output = self.seg_layers[i](output)  # (B, 3, D, H, W)
    
    if return_features:
        return seg_output, decoder_feature_before_seg
    else:
        return seg_output
```

### 3.4 BoostedLateFusion 模型定义

```python
class ConfigurableSegHead(nn.Module):
    """可配置分割头 - 对应论文的 Configurable Classifier"""
    
    def __init__(self, in_channels=8, hidden_channels=8):
        """
        Args:
            in_channels: 解码器特征的通道数 (默认 8)
            hidden_channels: 中间通道数 (对应论文 Layer1 的 256)
        """
        super().__init__()
        # 私有部分（对应 Layer1 → ReLU）
        self.private = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, features, shared_head):
        """
        Args:
            features: (B, 8, D, H, W) 解码器特征
            shared_head: 共享的最终投影层
        Returns:
            logits: (B, 3, D, H, W)
        """
        h = self.private(features)        # (B, C_head, D, H, W)
        logits = shared_head(h)           # (B, 3, D, H, W)
        return logits


class BoostedLateFusion(nn.Module):
    """
    Sustained Boosting + Adaptive Classifier Assignment 的分割模型。
    每个模态有一个共享的 UNet (encoder+decoder) 和多个可配置分割头。
    """
    
    def __init__(self, unet_params, n_experts=4, head_hidden_channels=8, max_heads_per_modality=10):
        """
        Args:
            unet_params: UNet 构造参数的 dict
            n_experts: 模态数量 (默认 4)
            head_hidden_channels: 分割头中间通道数 (默认 8)
            max_heads_per_modality: 每个模态最大分割头数量 (默认 10)
        """
        super().__init__()
        self.n_experts = n_experts
        self.head_hidden_channels = head_hidden_channels
        self.max_heads = max_heads_per_modality
        
        # 4 个独立的 UNet（encoder + decoder，不含最终 seg_layer）
        # 或者复用 UNet 但不使用其 seg_layer
        self.backbones = nn.ModuleList([
            UNet(**unet_params) for _ in range(n_experts)
        ])
        
        # 共享分割头（对应论文 Layer2，所有模态、所有头共享）
        self.shared_head = nn.Conv3d(head_hidden_channels, 3, kernel_size=1, bias=True)
        
        # 每个模态的可配置分割头列表
        # 初始化：每个模态 1 个头
        feat_channels = unet_params.get('n_features_per_stage', [8, 16, 32, 64, 80, 80])[0]  # = 8
        self.modality_heads = nn.ModuleList([
            nn.ModuleList([
                ConfigurableSegHead(in_channels=feat_channels, hidden_channels=head_hidden_channels)
            ])
            for _ in range(n_experts)
        ])
    
    def add_head(self, modality_idx):
        """为指定模态动态添加一个新的分割头"""
        if len(self.modality_heads[modality_idx]) >= self.max_heads:
            return False  # 已达上限
        feat_channels = self.backbones[0].n_features_per_stage[0]  # = 8
        new_head = ConfigurableSegHead(
            in_channels=feat_channels,
            hidden_channels=self.head_hidden_channels,
        ).to(next(self.parameters()).device)
        self.modality_heads[modality_idx].append(new_head)
        return True
    
    def get_num_heads(self):
        """返回各模态当前的分割头数量"""
        return [len(heads) for heads in self.modality_heads]
    
    def forward(self, x):
        """
        Args:
            x: (B, 4, D, H, W)
        Returns:
            dict:
                'output': (B, 3, D, H, W) — 最终融合预测（用于推理）
                'modality_all_logits': list of 4 个 (B, 3, D, H, W) — 各模态的联合预测
                'modality_head_logits': list of 4 个 list — 各模态各头的独立 logits
        """
        modality_all_logits = []    # 各模态的联合预测 Σ logits
        modality_head_logits = []   # 各模态各头的独立 logits
        
        for m_idx in range(self.n_experts):
            x_m = x[:, m_idx:m_idx+1, ...]  # (B, 1, D, H, W)
            
            # 通过 UNet backbone 提取特征
            _, decoder_features = self.backbones[m_idx](x_m, return_features=True)
            # decoder_features: (B, 8, D, H, W)
            
            # 通过该模态的所有分割头
            head_logits_list = []
            for head in self.modality_heads[m_idx]:
                logits = head(decoder_features, self.shared_head)  # (B, 3, D, H, W)
                head_logits_list.append(logits)
            
            # 模态内聚合：所有头的 logits 求和
            combined = torch.stack(head_logits_list, dim=0).sum(dim=0)  # (B, 3, D, H, W)
            
            modality_all_logits.append(combined)
            modality_head_logits.append(head_logits_list)
        
        # 模态间融合：等权平均
        output = torch.stack(modality_all_logits, dim=0).mean(dim=0)  # (B, 3, D, H, W)
        
        return {
            'output': output,
            'modality_all_logits': modality_all_logits,
            'modality_head_logits': modality_head_logits,
        }
```

### 3.5 损失函数定义

```python
class SustainedBoostingLoss(nn.Module):
    """
    Sustained Boosting 损失函数。
    对每个模态计算三个损失项：ε (残差) + ε_all (联合) + ε_pre (前序)。
    """
    
    def __init__(self, lambda_smooth=0.33):
        """
        Args:
            lambda_smooth: 残差标签平滑系数 λ (论文默认搜索 {0.1, 0.2, 0.33, 0.5, 1.0})
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.dice_bce = DiceBCEWithLogitsLoss()     # 用于 ε_all 和 ε_pre（硬标签）
        self.bce = nn.BCELoss(reduction='mean')     # 用于 ε（软残差标签）
    
    def compute_residual_labels(self, labels, head_logits_list):
        """
        计算最新分类器的残差标签。
        
        Args:
            labels: (B, 3, D, H, W) 原始二值标签
            head_logits_list: 该模态所有头的 logits 列表
        Returns:
            residual_labels: (B, 3, D, H, W) 残差标签 ∈ [0, 1]
        """
        n_heads = len(head_logits_list)
        if n_heads <= 1:
            return labels.float()   # 只有 1 个头时，残差标签 = 原始标签
        
        # 累积前 t-1 个头的预测概率
        prev_probs_sum = torch.zeros_like(labels, dtype=torch.float32)
        for j in range(n_heads - 1):
            prev_probs_sum += labels.float() * torch.sigmoid(head_logits_list[j].detach())
            # 注意：使用 detach()，残差标签不对前序分类器产生梯度
            # 这与论文一致：残差标签是"给定前序预测"下计算的
        
        residual = labels.float() - self.lambda_smooth * prev_probs_sum
        residual = torch.clamp(residual, min=0.0)
        return residual
    
    def forward(self, model_output, labels):
        """
        Args:
            model_output: BoostedLateFusion.forward() 的输出 dict
            labels: (B, 3, D, H, W) 原始二值标签
        Returns:
            total_loss: 标量
            loss_details: dict，包含各分项损失，用于日志记录
        """
        modality_head_logits = model_output['modality_head_logits']
        n_modalities = len(modality_head_logits)
        
        total_loss = 0.0
        loss_details = {}
        
        for m_idx in range(n_modalities):
            head_logits_list = modality_head_logits[m_idx]
            n_heads = len(head_logits_list)
            
            # ---------- ε: 残差损失（最新头 vs 残差标签）----------
            residual_labels = self.compute_residual_labels(labels, head_logits_list)
            newest_probs = torch.sigmoid(head_logits_list[-1])
            epsilon = self.bce(newest_probs, residual_labels)
            
            # ---------- ε_all: 全体联合预测损失 ----------
            combined_logits = torch.stack(head_logits_list, dim=0).sum(dim=0)
            epsilon_all = self.dice_bce(combined_logits, labels)
            
            # ---------- ε_pre: 前序联合预测损失 ----------
            if n_heads > 1:
                pre_logits = torch.stack(head_logits_list[:-1], dim=0).sum(dim=0)
                epsilon_pre = self.dice_bce(pre_logits, labels)
            else:
                epsilon_pre = 0.0
            
            modality_loss = epsilon + epsilon_all + epsilon_pre
            total_loss += modality_loss
            
            loss_details[f'modality_{m_idx}_epsilon'] = epsilon.item() if isinstance(epsilon, torch.Tensor) else epsilon
            loss_details[f'modality_{m_idx}_epsilon_all'] = epsilon_all.item()
            loss_details[f'modality_{m_idx}_epsilon_pre'] = epsilon_pre.item() if isinstance(epsilon_pre, torch.Tensor) else epsilon_pre
            loss_details[f'modality_{m_idx}_n_heads'] = n_heads
        
        loss_details['total_loss'] = total_loss.item()
        return total_loss, loss_details
```

### 3.6 置信度计算与 ACA 策略

```python
class AdaptiveClassifierAssignment:
    """
    自适应分类器分配策略。
    每 t_N 个 epoch 检查各模态的置信度，为最弱模态添加分割头。
    """
    
    def __init__(self, sigma=1.0, tau=0.01, check_interval_epochs=10, max_heads=10):
        """
        Args:
            sigma: 系数 σ（默认 1.0）
            tau: 容忍阈值 τ（默认 0.01）
            check_interval_epochs: 检查频率 t_N（默认 10 个 epoch）
            max_heads: 单模态最大分割头数
        """
        self.sigma = sigma
        self.tau = tau
        self.check_interval = check_interval_epochs
        self.max_heads = max_heads
    
    @torch.no_grad()
    def compute_confidence_scores(self, model, dataloader, device):
        """
        计算各模态的置信度分数。
        
        Returns:
            scores: list of 4 个 float，各模态的置信度
        """
        model.eval()
        n_modalities = model.n_experts
        score_sums = [0.0] * n_modalities
        count = 0
        
        for batch in dataloader:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            output = model(x)
            
            for m_idx in range(n_modalities):
                combined_logits = output['modality_all_logits'][m_idx]
                combined_probs = torch.sigmoid(combined_logits)     # (B, 3, D, H, W)
                
                # 前景体素上的平均预测概率
                foreground_mask = (y > 0.5).float()                 # (B, 3, D, H, W)
                masked_probs = combined_probs * foreground_mask     # 只保留前景
                
                # 每个样本的置信度 = 前景区域的平均预测概率
                fg_count = foreground_mask.sum(dim=[2, 3, 4]).clamp(min=1)  # (B, 3)
                sample_confidence = (masked_probs.sum(dim=[2, 3, 4]) / fg_count).mean(dim=1)  # (B,)
                score_sums[m_idx] += sample_confidence.sum().item()
            
            count += x.shape[0]
        
        scores = [s / max(count, 1) for s in score_sums]
        model.train()
        return scores
    
    def should_add_head(self, epoch):
        """检查当前 epoch 是否需要执行 ACA"""
        return epoch > 0 and epoch % self.check_interval == 0
    
    def assign(self, model, scores):
        """
        根据置信度分数执行分配。
        
        Args:
            model: BoostedLateFusion 实例
            scores: 各模态置信度列表
        Returns:
            added_modality: 被添加分割头的模态索引，若未添加则为 None
        """
        s_max = max(scores)
        s_min = min(scores)
        weak_idx = scores.index(s_min)
        
        if s_max - self.sigma * s_min > self.tau:
            current_heads = model.get_num_heads()
            if current_heads[weak_idx] < self.max_heads:
                success = model.add_head(weak_idx)
                if success:
                    return weak_idx
        return None
```

### 3.7 训练流程

训练器需要在 Baseline 训练器基础上做以下修改：

1. **模型输出为 dict**：forward 返回包含 `output`、`modality_all_logits`、`modality_head_logits` 的字典
2. **损失函数**：使用 `SustainedBoostingLoss` 替代 `DiceBCEWithLogitsLoss`
3. **ACA 逻辑**：每 `t_N` 个 epoch 计算置信度并决定是否添加分割头
4. **优化器动态更新**：当新增分割头时，需要将其参数添加到优化器中
5. **日志记录**：记录各模态的分割头数量、置信度、各分项损失

**关键训练循环伪代码**：

```python
def train_boosted_model(model, optimizer, loss_fn, aca, train_dl, val_dl, ...):
    for epoch in range(1, num_epochs + 1):
        # === 训练阶段 ===
        model.train()
        for batch in train_dl:
            x, y = batch['img'].to(device), batch['label'].to(device)
            
            model_output = model(x)
            total_loss, loss_details = loss_fn(model_output, y)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # === ACA 检查 ===
        if aca.should_add_head(epoch):
            scores = aca.compute_confidence_scores(model, train_dl, device)
            added = aca.assign(model, scores)
            if added is not None:
                # 将新增头的参数加入优化器
                new_head = model.modality_heads[added][-1]
                optimizer.add_param_group({
                    'params': new_head.parameters(),
                    'lr': current_lr,
                })
                print(f'Epoch {epoch}: Added head to modality {added}, '
                      f'now has {model.get_num_heads()} heads')
        
        # === 验证阶段 ===
        if epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                for batch in val_dl:
                    x, y = batch['img'].to(device), batch['label'].to(device)
                    model_output = model(x)
                    # 使用 output['output'] 计算 Dice 指标
                    ...
```

### 3.8 超参数配置

```python
class BoostedConfig:
    """Sustained Boosting 方法的超参数"""
    
    # ---- Sustained Boosting 参数 ----
    LAMBDA_SMOOTH = 0.33        # 残差标签平滑系数 λ (论文搜索范围: {0.1, 0.2, 0.33, 0.5, 1.0})
    HEAD_HIDDEN_CHANNELS = 8    # 可配置分割头的中间通道数 (对应论文 Layer1 的隐藏维度)
    MAX_HEADS_PER_MODALITY = 10 # 每个模态最大分割头数量
    
    # ---- ACA 参数 ----
    ACA_SIGMA = 1.0             # 系数 σ (论文默认 1.0)
    ACA_TAU = 0.01              # 容忍阈值 τ (论文默认 0.01)
    ACA_CHECK_INTERVAL = 10     # 检查频率 t_N (以 epoch 为单位)
    
    # ---- 训练参数（与 Baseline 保持一致）----
    RANDOM_SEED = 12345
    N_EPOCHS = 300
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 8
    VAL_FREQ = 5
    RESULTS_DIR = 'saved_models/boosted'
```

### 3.9 显存估算

| 组件 | 估算显存 |
|------|----------|
| Baseline 4×UNet (encoder+decoder) | ~18 GB（实测） |
| 每个分割头激活 (B=4, C=8, 128³) | ~268 MB（前向）+ ~268 MB（梯度） ≈ 0.5 GB |
| 假设最终：4个模态 × 平均5个头 = 20个头 | 20 × 0.5 = ~10 GB |
| Shared Head（极小） | < 0.01 GB |
| **估计总显存** | **~28 GB（48 GB 可承受）** |

### 3.10 推理流程

推理时，使用模型的 `output` 字段（所有模态、所有头的联合融合预测）：

```python
model.eval()
with torch.no_grad():
    model_output = model(x)
    final_logits = model_output['output']       # (B, 3, D, H, W)
    predictions = (torch.sigmoid(final_logits) > 0.5).float()
```

### 3.11 需要特别注意的实现细节

1. **残差标签中的 `detach()`**：计算残差标签时，前序分类器的预测需要 `.detach()`，残差标签不应对前序分类器产生梯度（残差标签是"给定前序预测"下的固定量）。

2. **Shared Head 的梯度来源**：Shared Head 接收来自所有模态、所有头的梯度。在反向传播时，PyTorch 会自动累积这些梯度，无需额外处理。

3. **新增头的优化器注册**：当 ACA 添加新的分割头时，必须将新头的参数显式添加到优化器的参数组中。否则新头的参数不会被更新。

4. **新增头的权重初始化**：新头的私有部分使用默认 PyTorch 初始化（Kaiming uniform）。Shared Head 在添加新头时不需要重新初始化（它是共享的、已有的）。

5. **UNet backbone 不使用原有 seg_layer**：BoostedLateFusion 中，UNet 的 `seg_layers` 不再被使用（或仅用于兼容）。所有分割预测都通过可配置分割头产生。实际实现时，可以修改 UNet.forward 返回解码器特征，或者在 BoostedLateFusion 中 hook 解码器最后一个阶段的输出。

6. **n_o = 1 时的损失退化**：当某模态只有 1 个头时，ε_pre = 0，残差标签 = 原始标签。此时 L_SUB = ε + ε_all = 2 × DiceBCE，等价于 Baseline 损失的 2 倍（常数因子对优化无影响）。

---

## 附录：论文关键公式速查

| 公式 | 编号 | 说明 |
|------|------|------|
| `p^o = ψ^o(ϕ^o(x^o))` | — | 模态预测 |
| `ŷ_it = y_i - λ Σ_{j<t} (y_i ⊙ p_ij)` | Sec 3.2 | 残差标签 |
| `ε = ℓ(p_t, ŷ_t)` | (2) | 残差损失 |
| `ε_all = ℓ(Σ_{j≤t} p_j, y)` | (3) | 联合预测损失 |
| `ε_pre = ℓ(Σ_{j<t} p_j, y)` | (4) | 前序预测损失 |
| `L = ε + ε_all + ε_pre` | (5) | 单样本总损失 |
| `L_SUB = (1/N) Σ_i L(x_i, y_i, n_o)` | (6) | 模态总损失 |
| `s^o = (1/N) Σ_i y_i^T Σ_j p^o_j` | Sec 3.3 | 置信度分数 |
| `G(Φ) = L^a - L^v` | (7) | Gap 函数 |
| `G(Φ(T)) ≤ G(Φ(0)) / (1 + dT·G(Φ(0)))` | (8) | 收敛速率 O(1/T) |
