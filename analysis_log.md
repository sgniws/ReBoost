# Boost 训练分析记录（方案10实现性 + 强弱模态判定）

## 1. 分析目标

1) 判断 `analysis_solutions.md` 中的**方案10**是否已在当前工程实现。  
2) 依据以下两篇论文对“强势/弱势模态”的定义，结合日志 `saved_models/boosted/training_2026-02-15_09-07-23-间隔30-修复版.log`，判断当前实验中的强弱模态：
- `references/Jiang 等 - Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproporti.pdf`
- `references/Peng 等 - 2022 - Balanced Multimodal Learning via On-the-fly Gradient Modulation.pdf`

---

## 2. 方案10是否已实现

## 结论：**已实现核心思想（10a），10b为“等价改写/部分差异实现”**

### 2.1 方案10原文要点（来自 `analysis_solutions.md`）

- **10a**：总损失以融合输出 loss 为主，再叠加 boost 辅助项。  
  形式：`total_loss = dice_bce(fused_output, labels) + boost_aux`
- **10b**：per-modality boost 项建议“取平均而非求和”（文中示例是除以 `n_modalities`）。

### 2.2 当前实现核对（来自 `loss/boosted_loss.py`）

- 存在明确的主损失：
  - `fused_loss = dice_bce(model_output["output"], labels)`
- 仅当某模态 `n_heads > 1` 时，才计算该模态 boost 项：
  - `epsilon + epsilon_all + epsilon_pre`
- 最终总损失：
  - `total_loss = fused_loss + lambda_boost * (boost_loss_raw / n_boosted)`

这与方案10的核心“**融合loss主导，boost作为辅助**”完全一致（10a实现成立）。

### 2.3 与10b的差异说明

- 文档10b建议是“按 `n_modalities` 平均”；
- 当前代码是“按 `n_boosted`（当前实际被boost的模态数）平均”。

这属于归一化策略的实现差异，但不改变10a的核心机制。  
因此判定：**方案10核心已落地，且实现更偏向只惩罚已进入多头阶段的模态。**

---

## 3. 强弱模态定义对齐

### 3.1 Jiang (NeurIPS 2025) 视角

- 论文定义：  
  - **强势模态（strong modality）**：分类能力更强、置信分更高、收敛更快；  
  - **弱势模态（weak modality）**：分类能力较弱、置信分更低。  
- 在其 ACA 机制中，会对**弱势模态**分配新分类器（新 head）。

### 3.2 Peng (CVPR 2022 OGM-GE) 视角

- 论文定义：  
  - **强势/主导模态**：对联合目标贡献更大、在优化中占主导；  
  - **弱势模态**：受压制、优化不足（under-optimized）。
- 在可观测统计上，常体现为主导模态占据更大优化“份额”（如梯度占比更高）。

---

## 4. 日志证据

基于日志解析（Epoch 1~261）：

### 4.1 ACA 加头记录

- Epoch 30: `ACA_added_modality: 1`，`n_heads -> [1,2,1,1]`
- Epoch 60: `ACA_added_modality: 0`，`n_heads -> [2,2,1,1]`
- Epoch 90: `ACA_added_modality: 0`，`n_heads -> [3,2,1,1]`
- Epoch 120/150/180/210/240: 均为 `ACA_added_modality: 0`
- 最终头数演化：`[1,1,1,1] -> [1,2,1,1] -> [2,2,1,1] -> ... -> [8,2,1,1]`

### 4.2 梯度占比统计（encoder_ratio）

- 全程均值（%）：
  - 模态0: **45.33**
  - 模态1: **47.16**
  - 模态2: **3.71**
  - 模态3: **3.80**
- 最大占比出现次数（261个epoch）：
  - 模态0: **208次**
  - 模态1: **53次**
  - 模态2: 0次
  - 模态3: 0次
- 分阶段均值（%）：
  - Epoch 1-29: `[31.32, 32.98, 18.67, 17.02]`
  - Epoch 31-59: `[3.69, 88.54, 2.94, 4.83]`
  - Epoch 61-89: `[50.09, 46.13, 1.65, 2.16]`
  - Epoch 91-260: `[54.30, 42.46, 1.64, 1.61]`

---

## 5. 强弱模态判定（按论文定义分别给出）

### 5.1 按 Jiang 的“分类能力/ACA”定义

- **弱势模态（主弱势）**：`模态0 (T1)`  
  - 原因：在 epoch60 之后几乎每个 ACA 检查点都被判为弱势并持续加头（共7次）。
- **阶段性弱势**：`模态1 (T1ce)`  
  - 仅在 epoch30 被加头一次，随后不再是主要弱势。
- **强势模态**：无法仅凭现有日志精确唯一定位到 1/2/3 中哪一个是长期 `s_max`  
  - 因为日志未直接输出每次 ACA 的 `confidence scores`，仅输出了“被加头的弱势模态”。

> 可确定的是：Jiang-ACA 角度下，“当前被系统持续判弱”的核心对象是 **模态0(T1)**。

### 5.2 按 Peng 的“优化主导/被抑制”定义

- **强势（优化主导）模态**：`模态0(T1)` 与 `模态1(T1ce)`  
  - 原因：两者几乎占据全部梯度份额（合计约 92%），且最大梯度占比只在0/1之间切换。
- **弱势（优化受抑）模态**：`模态2(T2)`、`模态3(FLAIR)`  
  - 原因：长期梯度占比仅约 1%~4%，明显处于被压制状态。

---

## 6. 最终结论（直接回答）

1) **方案10是否实现？**  
**是。** 当前 `loss/boosted_loss.py` 已实现“融合损失为主、boost为辅”（方案10a核心）。  
10b 的“平均”策略采用了 `n_boosted` 归一化而非 `n_modalities`，属于实现细节差异。

2) **哪个模态是强势/弱势？**  
- **按 Jiang(分类能力/ACA) 语义**：  
  - 主要弱势模态：**模态0 (T1)**（被反复加头）  
  - 阶段性弱势：模态1 (T1ce)（仅早期一次）  
- **按 Peng(优化主导) 语义**：  
  - 强势模态：**模态0 (T1)、模态1 (T1ce)**  
  - 弱势模态：**模态2 (T2)、模态3 (FLAIR)**

> 两种定义会出现“看似矛盾”的结果：  
> ACA 视角判定 T1 为弱，是“分类能力/置信度”语义；  
> 梯度份额视角又显示 T1/T1ce 主导优化，是“优化贡献/主导”语义。  
> 在当前实验里，这种分离现象确实存在，并且日志证据充分。

