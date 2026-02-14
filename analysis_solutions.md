# Boost 方法实现分析与问题诊断

> 参考论文：Jiang 等 - Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion (NeurIPS 2025)
> 
> 官方实现：/root/autodl-tmp/codes/NeurIPS25-AUG
> 
> 本项目实现：/root/autodl-tmp/codes/reb-UNet (boost 分支)

---

## 一、两种代码实现的对比分析

### 1.1 训练策略：分离优化 vs 联合优化（关键差异）

**AUG 官方实现**：对每个模态使用**独立的 forward-backward-step 循环**：

```python
# 官方：audio 独立优化
o_a = model.audio_encoder(spectrogram)
out_a, o_fea, add_fea = model.classfier(o_a, is_a=True)
loss_a = ...
loss_a.backward()
optimizer.step()
optimizer.zero_grad()

# 官方：video 独立优化
o_v = model.video_encoder(image)
out_v, o_fea, add_fea = model.classfier(o_v, is_a=False)
loss_v = ...
loss_v.backward()
optimizer.step()
optimizer.zero_grad()
```

每个模态的 encoder 在各自的 backward 中独立接收梯度，optimizer.step() 独立执行。这意味着：
- Audio 的 backward 只更新 audio encoder + shared head
- Video 的 backward 只更新 video encoder + shared head
- 两次 step 之间，shared head 被更新两次（分别接收来自 audio 和 video 的梯度）

**本项目 Boost 实现**：所有 4 个模态进行**联合 forward，然后单次 backward + step**：

```python
# Boost 实现：联合优化
model_output = self.net(features)  # 同时前向 4 个模态
total_loss, loss_details = self.loss_fn(model_output, labels)  # 4 个模态 loss 求和
total_loss.backward()  # 单次 backward，所有 encoder 同时接收梯度
self.optimizer.step()
self.optimizer.zero_grad()
```

**影响**：
- 联合 backward 导致 4 个模态的梯度在 shared head 上被同时累积，梯度可能相互干扰
- 论文的理论分析（Theorem 1）假设 "the parameters for each modality are independent"（见论文 Proof 2 中的关键步骤：`Lv(Φ(t+1)) = Lv(Φ(t))`），这要求模态间独立更新
- 联合优化破坏了这一理论前提，无法保证 Gap 函数的 O(1/T) 收敛速率

### 1.2 残差标签计算方式的差异

**AUG 官方**：

```python
kl = y * o_fea.detach().softmax(1)
# 其中 o_fea = 所有前序 head 的 logits 之和（combined logits）
# kl = y ⊙ softmax(Σ_{j<t} logit_j)
```

残差标签的实质是：`ŷ = y - λ · y ⊙ softmax(combined_previous_logits)`

这里 `softmax` 作用于**所有前序 head 的联合 logits**。在分类任务中，softmax 将联合 logits 归一化为概率分布。

**Boost 实现**：

```python
prev_sum = torch.zeros_like(labels, ...)
for j in range(n_heads - 1):
    prev_sum += labels.float() * torch.sigmoid(head_logits_list[j].detach())
residual = labels.float() - self.lambda_smooth * prev_sum
```

残差标签的实质是：`ŷ = y - λ · Σ_{j<t} (y ⊙ sigmoid(logit_j))`

**关键区别**：
| 维度 | AUG 官方 | Boost 实现 |
|------|----------|-----------|
| 激活函数 | softmax（对联合 logits） | sigmoid（对每个 head 独立） |
| 前序预测聚合 | 先 sum logits，再 softmax | 分别 sigmoid，再 sum |
| 值域 | `y ⊙ softmax(·)` 的值在 [0,1]，且各类之和 ≤ 1 | `Σ sigmoid(·)` 的值可能 > 1（多个 head 都高置信时） |

在分割任务中，由于使用 sigmoid（每个通道独立二分类）而非 softmax，Boost 实现的适配思路是合理的。但当 `n_heads` 较大时，`Σ sigmoid(·)` 可能远大于 1，导致 `residual = y - λ * Σ sigmoid(·)` 的许多值被 clamp 到 0，新 head 几乎没有可学习的残差信号。

### 1.3 损失函数的差异

**AUG 官方**（当 `add_fea` 不为 None 时）：

```python
loss_a = criterion(out_a, y).mean()           # ε_all: 全体联合预测
       + criterion(o_fea, y).mean()           # ε_pre: 前序联合预测
       + criterion(add_fea, y).mean()         # \
       - 0.5 * criterion(add_fea, kl).mean()  # / ε: 残差损失（隐式）
```

官方利用交叉熵损失对 target 的线性性质：`CE(p, y - λ·kl) = CE(p, y) - λ·CE(p, kl)`，将残差损失分解为两项。这里 λ = 0.5（硬编码）。

**Boost 实现**：

```python
# ε: 残差损失
newest_probs = torch.sigmoid(head_logits_list[-1])
epsilon = self.bce(newest_probs, residual_labels)  # BCE(sigmoid(newest), residual)

# ε_all: 联合预测损失
combined_logits = torch.stack(head_logits_list, dim=0).sum(dim=0)
epsilon_all = self.dice_bce(combined_logits, labels)  # DiceBCE

# ε_pre: 前序预测损失
pre_logits = torch.stack(head_logits_list[:-1], dim=0).sum(dim=0)
epsilon_pre = self.dice_bce(pre_logits, labels)  # DiceBCE
```

**区别**：
1. 残差损失 ε 使用 BCE（合理，因为残差标签是软标签）
2. ε_all 和 ε_pre 使用 DiceBCE（比 baseline 多了 Dice 损失项）
3. 损失量级不同：Boost 的总损失 = 4个模态 × (ε + ε_all + ε_pre)，远大于 baseline

### 1.4 ACA 策略的差异

**AUG 官方**（双模态）：基于 **ratio（比值）** 判断

```python
ratio_a = score_a / score_v
if ratio_a > args.lam + 0.01:  # audio 强于 video
    model.add_layer(is_a=False)  # 给弱模态 video 加头
elif ratio_a < args.lam - 0.01:  # video 强于 audio
    model.add_layer(is_a=True)   # 给弱模态 audio 加头
```

每个 epoch 都会累积 `score_a` 和 `score_v`，然后取比值。这是一种**相对比较**。

**Boost 实现**（四模态）：基于 **max-min 差值** 判断

```python
s_max = max(scores)
s_min = min(scores)
weak_idx = scores.index(s_min)
if s_max - self.sigma * s_min > self.tau:
    model.add_head(weak_idx)
```

这是一种**绝对差值比较**。当 τ = 0.01 且 sigma = 1.0 时，只要 max - min > 0.01 就会触发，条件非常宽松。

**关键问题**：在训练初期，模态间置信度差距几乎总是 > 0.01，导致每次 ACA 检查都会触发 head 添加。从日志中可以看到：
- Epoch 30：添加 head 到模态 0（n_heads: [2,1,1,1]）
- Epoch 60：再次添加 head 到模态 0（n_heads: [3,1,1,1]）
- Epoch 90：添加 head 到模态 2（n_heads: [3,1,2,1]）

### 1.5 新增 Head 的优化器注册

**AUG 官方**：添加新层后，**没有**显式将新层参数加入优化器。

```python
def add_layer(self, is_a=True):
    new_layer = nn.Linear(self.hidden_dim, 256).cuda()
    nn.init.xavier_normal_(new_layer.weight)
    nn.init.constant_(new_layer.bias, 0)
    if is_a:
        self.additional_layers_a.append(new_layer)
```

由于 PyTorch 的优化器在创建时捕获参数引用，新添加到 `ModuleList` 的层的参数**不在**优化器的 param_groups 中，因此新层的 `private` 部分参数实际上**不会被优化器更新**（但 shared head `fc_out` 仍然会通过新层的梯度路径接收梯度）。

**Boost 实现**：正确地将新 head 的参数加入优化器：

```python
if added is not None:
    current_lr = optimizer.param_groups[0]['lr']
    new_head = net.modality_heads[added][-1]
    optimizer.add_param_group({'params': new_head.parameters(), 'lr': current_lr})
```

虽然 Boost 在这一点上更加"正确"，但这恰恰可能导致了更大的问题：新初始化的随机权重 head 立即以全学习率参与优化，导致训练不稳定。

### 1.6 ACA 置信度检查频率

**AUG 官方**：在 CREMAD 上使用 tN = 20 epochs 作为检查频率，总共训练 200 epochs（约 10 次检查）。

**Boost 实现**：使用 30 epochs 作为检查间隔，300 epochs 训练（约 10 次检查）。比例相近，但考虑到分割任务的复杂性，间隔可能仍然不够。

---

## 二、训练日志现象分析

### 2.1 现象一：训练初期 WT/TC/ET 为 0

**对比数据**：
| 指标 | Baseline Epoch 5 | Boost Epoch 5 |
|------|-----------------|---------------|
| task_loss | 1.0240 | 3.8525 |
| val_dice | 0.6316 | 0.0000 |
| WT/TC/ET | 0.8207/0.5699/0.4439 | 0.0000/0.0000/0.0000 |

**根因分析**：

1. **损失量级差距**：Boost 的总 loss ≈ 3.85（4 个模态，每个模态 ε + ε_all），而 baseline 仅 1.02。这是因为：
   - 当 `n_heads=1` 时，ε 的 residual_label = labels（原始标签），所以 ε ≈ ε_all ≈ 单模态的 DiceBCE
   - Total = 4 × (ε + ε_all) = 4 × 2 × baseline_per_modality ≈ 8× baseline_total

2. **梯度过大**：8 倍的 loss 意味着 8 倍的初始梯度，在初始训练阶段造成参数更新过激，模型无法有效收敛。

3. **预测全为 0 或全为 1**：过大的梯度可能导致 sigmoid 输出被推向极端值（全 0 或全 1），使得 Dice Score 为 0。

### 2.2 现象二：梯度范数比率失衡，T1 成为最大梯度模态

**对比数据**（稳定训练阶段）：

| 阶段 | T1 | T1ce | T2 | FLAIR |
|------|-----|-------|-----|-------|
| Baseline Epoch 50 | 17.1% | **49.2%** | 20.2% | 13.5% |
| Boost Epoch 25 (添加头前) | **52.1%** | 23.8% | 13.6% | 10.5% |
| Boost Epoch 35 (添加头后) | **64.6%** | 18.5% | 8.9% | 8.1% |

**根因分析**：

1. **梯度主导模态反转**：在 baseline 中，T1ce（模态 1）是梯度最大的模态（~50%），这符合 BraTS 数据的特征（T1ce 对增强肿瘤最敏感）。但在 Boost 中，T1（模态 0）反转成了最大的，这是不正常的。

2. **ACA 的恶性循环**：
   - T1 是 BraTS 中信息量最弱的模态（对肿瘤分割贡献最小）
   - ACA 检测到 T1 置信度最低 → 为 T1 添加 head
   - 添加 head → T1 的 loss 项增多（多了 ε_pre 和新的 ε）→ T1 的梯度范数增大
   - T1 的梯度增大 → 通过 shared head 影响其他模态
   - 其他模态的有效学习被干扰 → 整体性能下降

3. **添加 head 后的梯度激增**：
   - Epoch 29（添加前）：T1 梯度 = 1.30e-02
   - Epoch 31（添加后）：T1 梯度 = 2.45e-02（增幅 88%）
   - 新 head 的随机权重产生大梯度，对 shared head 和 backbone encoder 都造成冲击

### 2.3 现象三：ET Dice 严重下跌

**对比数据**：

| 时间点 | Baseline | Boost |
|--------|----------|-------|
| Epoch 80 ET | **0.7446** | 0.5632 |
| Epoch 85 ET | 0.7433 | 0.5751 |
| 最终 ET（~90 epoch） | 0.7368 | 0.5302 |

**根因分析**：

1. **ET 是最难的子任务**：增强肿瘤（ET）体积最小、边界最模糊，对模型精度要求最高。

2. **Boost 引入的额外噪声**：
   - 多个 head 的 logits 求和可能在空间上产生不一致的预测
   - 残差标签的计算在小目标区域（ET）可能不稳定
   - 当前景体素很少时，BCE 残差损失被大量背景体素的 0 标签主导

3. **Head 添加导致的 loss spike**：
   - Epoch 30 添加 head：val_loss 从 2.40 → 3.24，val_dice 从 0.643 → 0.665（看似回升但 ET 受损）
   - Epoch 60 添加 head：val_loss 从 2.55 → 2.82
   - Epoch 90 添加 head：val_loss 从 2.42 → 3.56
   - 每次添加 head 都造成训练的剧烈震荡，ET 作为最脆弱的指标受影响最大

4. **Shared head 的容量瓶颈**：所有模态所有 head 共享一个 Conv3d(8→3, k=1)。当 head 数量增多时，shared head 需要同时满足多个不同 head 的映射需求，容易在 ET 这种精细任务上产生冲突。

### 2.4 综合诊断：为什么 Boost 没有发挥预期作用

**核心原因**：从分类任务到分割任务的迁移存在多个根本性差异，当前实现未能充分处理这些差异：

1. **任务复杂度差异**：分类输出是一个 K 维向量，分割输出是一个 K×D×H×W 的体积。Boosting 在低维空间中很有效，但在高维空间中残差学习更困难。

2. **损失量级问题**：没有对总损失进行合理的归一化/缩放。

3. **优化策略不匹配**：联合优化 vs 分离优化的选择对方法效果至关重要。

4. **ACA 策略过于敏感**：在分割任务中，置信度分数波动更大，需要更保守的策略。

---

## 三、可能的解决方案

### 方案 1：改为分离优化（模态独立 backward）⭐ 最重要

**描述**：遵循 AUG 官方实现，对每个模态执行独立的 forward-backward-step 循环。

**实施要点**：
```python
for m_idx in range(4):
    optimizer.zero_grad()
    x_m = x[:, m_idx:m_idx+1, ...]
    # 只前向当前模态的 backbone 和 heads
    _, decoder_feat = model.backbones[m_idx](x_m, return_features=True)
    head_logits_list = [head(decoder_feat, model.shared_head) for head in model.modality_heads[m_idx]]
    # 计算当前模态的 loss
    modality_loss = compute_modality_loss(head_logits_list, labels)
    modality_loss.backward()
    optimizer.step()
```

**预期效果**：
- 满足论文理论分析的独立性假设
- 减少模态间梯度干扰
- shared head 在每次 step 中只接收一个模态的梯度，更新更稳定

**注意事项**：
- 这会增加训练时间（4 次 forward-backward vs 1 次）
- 需要注意 shared head 的梯度累积行为
- 需要在每个模态的 backward 后清理其他模态参数的梯度

### 方案 2：损失归一化/缩放

**描述**：对 Boost 的总损失进行合理缩放，使其与 baseline 在同一量级。

**选项 A**：按模态数和 head 数归一化
```python
total_loss = total_loss / n_modalities  # 除以模态数
```

**选项 B**：使用加权系数
```python
modality_loss = alpha * epsilon + beta * epsilon_all + gamma * epsilon_pre
# 例如 alpha=0.5, beta=1.0, gamma=0.5
```

**选项 C**：n_heads=1 时简化损失
```python
if n_heads == 1:
    modality_loss = epsilon_all  # 只使用联合预测损失，等效于 baseline
else:
    modality_loss = epsilon + epsilon_all + epsilon_pre
```

### 方案 3：修改残差标签计算方式

**描述**：更接近论文原始意图，使用组合预测而非独立预测之和。

```python
def compute_residual_labels(self, labels, head_logits_list):
    n_heads = len(head_logits_list)
    if n_heads <= 1:
        return labels.float()
    
    # 方式 A：使用组合 logits 的 sigmoid（推荐）
    pre_logits = torch.stack(head_logits_list[:-1], dim=0).sum(dim=0)
    prev_probs = torch.sigmoid(pre_logits.detach())
    residual = labels.float() - self.lambda_smooth * (labels.float() * prev_probs)
    residual = torch.clamp(residual, min=0.0)
    return residual
```

**原理**：先 sum logits 再 sigmoid，避免多个 head 的 sigmoid 值累加超过 1，使残差标签更加稳定。

### 方案 4：更保守的 ACA 策略

**4a. 增大阈值和检查间隔**：
```python
ACA_TAU = 0.05        # 从 0.01 增大到 0.05
ACA_CHECK_INTERVAL = 50  # 从 30 增大到 50
```

**4b. 添加 warm-up 期**：
```python
def should_check(self, epoch):
    return epoch >= 50 and epoch % self.check_interval == 0
```
在前 50 个 epoch 不进行 ACA，让模型先充分收敛。

**4c. 使用 EMA 平滑置信度**：
```python
ema_scores = ema_alpha * current_scores + (1 - ema_alpha) * ema_scores
# 使用平滑后的分数做决策
```

**4d. 每次只在差距真正显著时才添加**：
```python
# 使用比值而非差值，更符合原论文
ratio = max_score / (min_score + 1e-8)
if ratio > 1.5:  # 只有强模态是弱模态的 1.5 倍以上才添加
    model.add_head(weak_idx)
```

### 方案 5：新 Head 的渐进式引入

**描述**：新添加的 head 不应立即以全强度参与训练。

**5a. 新 head 使用较小的学习率**：
```python
optimizer.add_param_group({
    'params': new_head.parameters(), 
    'lr': current_lr * 0.1  # 初始学习率只有主体的 1/10
})
```

**5b. 新 head 输出使用渐进式权重**：
```python
# 在 forward 中，新 head 的输出乘以一个从 0 渐增到 1 的系数
head_age = epoch - head_creation_epoch
warmup_weight = min(1.0, head_age / warmup_epochs)
logits = head(decoder_feat, self.shared_head) * warmup_weight
```

**5c. 初始化策略**：将新 head 的 private 层初始化为接近零输出
```python
nn.init.zeros_(new_head.private[0].weight)
nn.init.zeros_(new_head.private[0].bias)
```

### 方案 6：解耦 Shared Head 或增大其容量

**描述**：当前的 shared head 只有一个 Conv3d(8→3, k=1)，参数量极少（27 个参数），可能成为瓶颈。

**选项 A**：增大 hidden_channels
```python
HEAD_HIDDEN_CHANNELS = 16  # 从 8 增大到 16
```

**选项 B**：增大 shared head 容量
```python
self.shared_head = nn.Sequential(
    nn.Conv3d(head_hidden_channels, head_hidden_channels, 1),
    nn.ReLU(),
    nn.Conv3d(head_hidden_channels, 3, 1),
)
```

**选项 C**：取消 shared head，每个 head 独立输出
```python
# 每个 head 自带完整的映射
class ConfigurableSegHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 8, 1), nn.ReLU(),
            nn.Conv3d(8, n_classes, 1),
        )
```
不过这会丧失论文中 shared head 促进跨模态交互的设计意图。

### 方案 7：先训练 Baseline，再 Fine-tune 添加 Boost

**描述**：两阶段训练方案。

- **阶段 1**（Epoch 1-100）：使用 baseline 的 Late Fusion 方式训练，不使用 Boost。此时每个模态只有 1 个 head，loss = DiceBCE（与 baseline 完全一致）。
- **阶段 2**（Epoch 101-300）：开启 ACA + Sustained Boosting。此时模型已有较好的初始化，添加 head 和残差学习不会造成过大震荡。

```python
if epoch <= warmup_epochs:
    # Phase 1: Standard training
    total_loss = sum(dice_bce(modality_logits[i], labels) for i in range(4))
else:
    # Phase 2: Sustained Boosting
    total_loss = sustained_boosting_loss(model_output, labels)
```

### 方案 8：针对 ET 的特殊处理

**描述**：ET 是最脆弱的指标，可以针对性保护。

**8a. 在残差损失中对 ET 通道加权**：
```python
# 对 ET channel (index=2) 使用更小的平滑系数
lambda_per_channel = [0.33, 0.33, 0.1]  # ET 使用更小的 lambda，保留更多原始标签信息
```

**8b. 在 ε_all 中对 ET 使用更大权重**：
```python
channel_weights = torch.tensor([1.0, 1.0, 2.0])  # ET 权重更大
epsilon_all = dice_bce(combined_logits, labels, weight=channel_weights)
```

---

## 补充分析：禁用 ACA 后 Boost 仍不等于 Baseline

> 背景：将 `ACA_CHECK_INTERVAL` 设为 300（即整个训练过程不触发 ACA），此时 `n_heads=[1,1,1,1]` 始终不变。理论上 Boost 应退化为 Baseline，但实验表明二者表现差距显著。
> 
> 对比日志：
> - Baseline: `saved_models/baseline/training_2026-02-15_00-11-54.log`
> - Boost (无ACA): `saved_models/boosted/training_2026-02-15_06-21-54.log`

### S1. 关键指标对比

| 指标 | Baseline Epoch 5 | Boost(无ACA) Epoch 5 |
|------|-----------------|---------------------|
| task_loss | 1.0240 | **3.8802** |
| val_dice | 0.6316 | **0.0000** |
| WT/TC/ET | 0.8207/0.5699/0.4439 | **0.0000/0.0000/0.0000** |

| 指标 | Baseline Epoch 50 | Boost(无ACA) Epoch 50 |
|------|------------------|----------------------|
| task_loss | 0.2473 | **2.1224** |
| val_dice | **0.8273** | 0.7018 |
| WT/TC/ET | **0.9040/0.8248/0.7294** | 0.8586/0.6914/0.5048 |

| 指标 | Baseline 梯度主导模态 | Boost(无ACA) 梯度主导模态 |
|------|---------------------|--------------------------|
| Epoch 1 | T2 (45.3%) | **T1 (27.8%)** → 快速升至 T1 |
| Epoch 15 | T1ce (37.8%) | **T1 (60.3%)** |
| Epoch 50 | T1ce (49.2%) | **T1 (53.1%)** |

即使没有 ACA 添加任何 head，Boost 仍然存在三大差异：**损失量级偏大**、**训练收敛显著变慢**、**梯度主导模态反转**。

### S2. 差异根因分析

#### 根因 1：损失函数结构不同（最关键）

当 `n_heads=1` 时，`SustainedBoostingLoss` 对每个模态计算：

| 损失项 | 计算方式 | 值 |
|--------|---------|-----|
| ε (残差损失) | `BCE(sigmoid(logits_m), labels)` | ≈ BCE per modality |
| ε_all (联合预测损失) | `DiceBCE(logits_m, labels) = Dice + BCE` | ≈ Dice + BCE per modality |
| ε_pre (前序损失) | `0` (n_heads=1 时不存在) | 0 |
| **模态小计** | **ε + ε_all = Dice + 2×BCE** | |

而 Baseline 的损失是对**融合后的输出**计算一次：

| 损失项 | 计算方式 |
|--------|---------|
| 总损失 | `DiceBCE(mean(4 modality logits), labels) = Dice + BCE` |

**差异汇总**：

```
Boost 总损失 = Σ_{m=0}^{3} (Dice_m + 2 × BCE_m) = 4 × Dice + 8 × BCE    [对每个模态独立计算]
Baseline 总损失 = Dice_fused + BCE_fused                                     [只对融合输出计算一次]
```

这导致了两个层面的差异：

**(a) 量级差异（~6-8 倍）**：Boost 对 4 个模态分别计算损失再求和，加上每个模态多一个 BCE 项，总量级约为 Baseline 的 6-8 倍。这与日志中 Epoch 1 的 loss 比值（8.79 / 1.53 ≈ 5.7）吻合。

**(b) 优化目标差异**：Baseline 优化的是**融合预测的质量**（mean 后的 logits vs labels），4 个模态的梯度通过 mean 操作相互耦合。Boost 优化的是**每个模态独立的预测质量**，4 个模态各自为战。这改变了梯度在模态间的分配方式，是梯度主导模态反转的根本原因。

#### 根因 2：分割头架构不同

| 特征 | Baseline | Boost |
|------|----------|-------|
| 分割头路径 | decoder_feat → `seg_layer` (Conv3d(8→3, k=1)) | decoder_feat → `private` (Conv3d(8→8, k=1) + ReLU) → `shared_head` (Conv3d(8→3, k=1)) |
| 层数 | 1 层 | 2 层 + 1 个 ReLU |
| 参数量 | 27 (8×3 + 3) | 75 (8×8+8) + 27 (8×3+3) = 102 |
| 非线性 | 无（线性投影） | ReLU（引入非线性） |

即使逻辑上 ConfigurableSegHead 应该与 seg_layer 等效，实际上：

**(a) 多了 ReLU 非线性**：Baseline 的 seg_layer 是纯线性映射（Conv3d(8→3)），而 Boost 的路径经过了一个 ReLU。ReLU 会截断负值特征，改变信息流。

**(b) 初始化不同**：两条路径的初始权重不同。Baseline 使用 UNet 构建时的默认 PyTorch 初始化，而 ConfigurableSegHead 有独立的初始化。

**(c) UNet 的 seg_layers 成为死参数**：Boost 模型中每个 UNet 的 `seg_layers`（5 层 Conv3d）仍然存在于模型参数中，但在 forward 中其输出被丢弃（`_, decoder_feat = backbone(x_m, return_features=True)`），因此这些参数：
- 不接收任何梯度（grad=None）
- 但仍占用优化器的状态内存（Adam 的 m/v buffers）
- 不影响训练正确性，但是多余的参数开销

#### 根因 3：梯度主导模态反转的机制

Baseline 中梯度从融合点（mean）向 4 个模态回传，梯度大小取决于**每个模态对融合预测误差的边际贡献**。T1ce 在 BraTS 中信息量最大，对融合误差的边际贡献最大，因此梯度最大。

Boost 中每个模态独立计算 loss，梯度大小取决于**每个模态独立预测的误差**。T1 信息量最弱，独立预测误差最大，因此梯度最大。

这正是"弱模态梯度大"的核心矛盾——在独立优化范式下，弱模态因为预测差而接收更大梯度，但更大梯度并不一定能帮助弱模态学得更好（可能过拟合噪声），反而通过 shared head 干扰强模态。

### S3. 解决方案

#### 方案 9：Boost 退化时等效于 Baseline（基础等效性修复）⭐ 最高优先级

**目标**：确保 `n_heads=1` 且无 ACA 时，Boost 的行为与 Baseline 完全一致。这是正确实现 Boost 的前提。

**修复内容**：

**(9a) 损失函数退化**：当 `n_heads=1` 时，不计算额外的 ε 残差损失，只保留 ε_all：

```python
def forward(self, model_output, labels):
    ...
    for m_idx in range(n_modalities):
        head_logits_list = modality_head_logits[m_idx]
        n_heads = len(head_logits_list)
        
        combined_logits = torch.stack(head_logits_list, dim=0).sum(dim=0)
        epsilon_all = self.dice_bce(combined_logits, labels)
        
        if n_heads > 1:
            # 只有多个 head 时才计算残差损失和前序损失
            residual_labels = self.compute_residual_labels(labels, head_logits_list)
            newest_probs = torch.sigmoid(head_logits_list[-1])
            epsilon = self.bce(newest_probs, residual_labels)
            pre_logits = torch.stack(head_logits_list[:-1], dim=0).sum(dim=0)
            epsilon_pre = self.dice_bce(pre_logits, labels)
            modality_loss = epsilon + epsilon_all + epsilon_pre
        else:
            modality_loss = epsilon_all  # 退化为标准 DiceBCE
        
        total_loss = total_loss + modality_loss
    
    # 除以模态数，使总量级与 baseline 一致
    total_loss = total_loss / n_modalities
    ...
```

**(9b) 使用融合输出计算损失（替代方案）**：直接使用融合后的 output 计算 baseline loss，加上可选的 per-modality boost loss：

```python
# 始终计算融合损失（与 baseline 等效）
fused_loss = self.dice_bce(model_output['output'], labels)

# 只有当 n_heads > 1 的模态存在时，才添加 boost 辅助损失
boost_loss = 0
for m_idx in range(n_modalities):
    if len(modality_head_logits[m_idx]) > 1:
        boost_loss += compute_boost_loss(modality_head_logits[m_idx], labels)

total_loss = fused_loss + lambda_boost * boost_loss
```

**(9c) 分割头架构对齐**：移除 ConfigurableSegHead 中的 ReLU 和中间层，使其与 Baseline 的 seg_layer 等效（仅在 `n_heads=1` 时）：

方案一：初始 head 直接使用线性映射（与 baseline seg_layer 等效）
```python
class ConfigurableSegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_nonlinearity=True):
        super().__init__()
        if use_nonlinearity:
            self.private = nn.Sequential(
                nn.Conv3d(in_channels, hidden_channels, 1, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.private = nn.Identity()  # 直接传递到 shared_head
```

方案二：保持当前架构但不影响等效性（如果 9a 的损失修复已足够，架构差异可接受）

#### 方案 10：修正梯度主导模态反转

**10a. 在 per-modality loss 前加上融合 loss 作为主损失**：

```python
total_loss = dice_bce(fused_output, labels)  # 主损失：与 baseline 一致
for m_idx in range(n_modalities):
    if n_heads[m_idx] > 1:
        total_loss += boost_loss_m / n_modalities  # 辅助损失：仅对有多个 head 的模态
```

这样梯度仍然主要由融合 loss 驱动（T1ce 主导），boost 部分只是辅助。

**10b. 对 per-modality loss 取平均而非求和**：

```python
total_loss = total_loss / n_modalities  # 已在方案 9a 中包含
```

---

## 四、建议的优先级排序

| 优先级 | 方案 | 预期收益 | 实现难度 | 说明 |
|--------|------|---------|---------|------|
| **P0** | **方案 9：基础等效性修复** | **极高** | **低** | **最优先：确保 n_heads=1 时 Boost ≡ Baseline** |
| P0 | 方案 2：损失归一化（已包含在方案 9 中） | 高 | 低 | 除以模态数 + n_heads=1 时去掉 ε |
| P1 | 方案 10：融合 loss 为主 + boost 为辅 | 高 | 低 | 修正梯度主导模态反转 |
| P1 | 方案 1：分离优化 | 高 | 中 | 遵循 AUG 官方的分离 backward |
| P1 | 方案 4：保守 ACA | 中-高 | 低 | 增大阈值 + warm-up |
| P2 | 方案 5：新 Head 渐进引入 | 中 | 低 | 零初始化或小学习率 |
| P2 | 方案 3：修改残差标签 | 中 | 低 | sum-then-sigmoid |
| P2 | 方案 7：两阶段训练 | 中 | 低 | 先 baseline 再 boost |
| P3 | 方案 6：增大 Head 容量 | 低-中 | 低 | 增大 hidden_channels |
| P3 | 方案 8：ET 特殊处理 | 低-中 | 低 | 通道级加权 |

**实施路径建议**：

1. **第一步（P0）**：实施方案 9，修复基础等效性。验证 `n_heads=1, ACA_CHECK_INTERVAL=300` 时 Boost 的训练曲线与 Baseline 一致。这是所有后续工作的基础。

2. **第二步（P1）**：在等效性修复的基础上，开启 ACA 和多 head 训练，同时实施方案 10（融合 loss 为主）和方案 4（保守 ACA）。

3. **第三步（P2-P3）**：根据第二步结果调整残差计算、head 引入策略等细节。

---

## 五、总结

当前 Boost 实现存在**两个层次**的问题：

### 基础层（必须首先解决）
即使完全禁用 ACA（`ACA_CHECK_INTERVAL=300`），Boost 也无法等效于 Baseline。根因是：
1. **损失结构不同**：`n_heads=1` 时多了一个冗余的 ε 残差损失项（此时 ε ≈ BCE(logits, labels)），加上 4 模态独立计算并求和，导致总 loss 约为 Baseline 的 6-8 倍
2. **优化目标不同**：Baseline 优化融合后的预测质量，Boost 优化每个模态独立的预测质量，导致梯度分配完全不同（T1ce 主导 → T1 主导）
3. **分割头架构不同**：多了一层 Conv3d + ReLU 非线性

### 方法层（在基础层修复后再处理）
Boost 方法本身的迁移问题：
1. **联合优化 vs 分离优化**与论文假设不匹配
2. **ACA 策略过于敏感**导致频繁添加 head 引发震荡
3. **新 Head 引入方式过于激进**

**建议实施路径**：先修复基础等效性（方案 9），验证 Boost 在 `n_heads=1` 时能复现 Baseline 表现；再在此基础上逐步引入 ACA 和 Sustained Boosting 的功能。
