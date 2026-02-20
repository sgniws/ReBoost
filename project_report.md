# reb-UNet 项目说明书

## 一、项目概述

### 1.1 任务与背景

本项目面向 **BraTS2018** 脑肿瘤 MRI 分割任务，使用四模态（T1、T1ce、T2、FLAIR）输入，输出三个子任务的分割图：**WT（全肿瘤）**、**TC（肿瘤核心）**、**ET（增强肿瘤）**。

核心创新来自论文《Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion》（NeurIPS 2025）的迁移：

- **问题**：多模态学习中不同模态收敛速度不一致，强模态主导、弱模态被欠优化。
- **思路**：通过监控各模态的**置信度**，对置信度较低的模态动态添加轻量级 **1×1 卷积分割头**，并配合 **Sustained Boosting** 的辅助损失（残差损失 ε、联合损失 ε_all、前序损失 ε_pre）强化弱模态、平衡模态学习速度。
- **实现**：在 Late Fusion（4 个独立 UNet Expert + 静态等权平均）基础上，将每个 Expert 的“最终分割层”替换为**可配置多分割头 + 共享头**，训练中按 **ACA（自适应分类器分配）** 策略为最弱模态增加新头。

### 1.2 技术要点摘要

| 维度 | 说明 |
|------|------|
| 基础架构 | 4 个独立单通道 3D UNet（nnUNet 风格），logit 空间等权平均融合 |
| 置信度 | 各模态“联合预测”在前景体素上的平均 sigmoid 概率 |
| 动态头 | 根据置信度差距（s_max - σ·s_min > τ）为最弱模态添加新 1×1 分割头 |
| 损失 | 融合损失（Dice+BCE）+ λ_boost ×（ε + ε_all + ε_pre）的归一化项 |
| 数据 | nnUNet 预处理 BraTS2018，128³ 体素，5 折划分 |

---

## 二、项目结构

```
reb-UNet/
├── configs.py              # Baseline/UNet/数据集/训练通用配置
├── configs_boosted.py      # Boosted 方法超参（λ、ACA、最大头数等）
├── train.py                # 命令行入口（argparse → run_baseline/run_boosted）
├── train/entry.py          # 统一入口：run_baseline()、run_boosted()，数据/模型/训练组装
├── pipeline.py             # Baseline 训练脚本（调用 run_baseline）
├── pipeline_boosted.py     # Boosted 训练脚本（调用 run_boosted）
├── sweep_train.py         # W&B 超参扫描（lr、lambda_smooth、lambda_boost）
├── models/
│   ├── nnunet.py          # 3D UNet 骨干（Encoder+Decoder+seg_layers；支持 return_features）
│   ├── baseline.py        # BaselineLateFusion：4×UNet，静态等权平均
│   └── boosted_fusion.py  # BoostedLateFusion：4×UNet 骨干 + 可配置多分割头 + 共享头
├── loss/
│   ├── dice_bce_loss.py   # Dice + BCE（融合损失与 ε_all/ε_pre）
│   └── boosted_loss.py    # SustainedBoostingLoss：ε、ε_all、ε_pre 与总损失
├── train/
│   ├── trainer.py         # Baseline 训练循环（StepRunner/EpochRunner/保存最佳）
│   └── trainer_boosted.py # Boosted 训练循环 + ACA、置信度计算、动态 add_head
├── dataset/
│   ├── processors.py      # SingleStreamDataset：加载 .npy、标签转三通道、增强
│   └── utils.py           # 归一化等工具
├── assets/
│   ├── kfold_splits.json  # 5 折划分（train/val）
│   └── data/              # 数据目录（可符号链接到 nnUNet 预处理路径）
└── saved_models/          # 按 mode/exp_name/run_name 保存 ckpt、日志、eval_res
```

---

## 三、各模块工作原理

### 3.1 配置模块

- **configs.py**  
  - `UNetConfig`：单通道、3 类、6 阶段、轻量特征 [8,16,32,64,80,80]、无深度监督。  
  - `DatasetConfig`：数据目录、kfold 路径、DROP_MODE（本项目用 None，即全 4 模态）、FOLD。  
  - `TrainingConfig`：随机种子、epoch、学习率、batch、验证频率、结果目录等。

- **configs_boosted.py**  
  - `BoostedConfig`：`LAMBDA_SMOOTH`（残差标签平滑）、`LAMBDA_BOOST`（boost 损失权重）、`HEAD_HIDDEN_CHANNELS`、`MAX_HEADS_PER_MODALITY`；ACA 的 `ACA_SIGMA`、`ACA_TAU`、`ACA_CHECK_INTERVAL`（每 N 个 epoch 检查一次）；训练参数与 baseline 对齐。

### 3.2 数据模块（dataset/）

- **SingleStreamDataset**（processors.py）  
  - 从 `splits_file_path` 的 `fold` 中读取 train/val 列表，从 `dataset_dir` 加载 `{id}.npy`（4×D×H×W）和 `{id}_seg.npy`。  
  - 标签预处理：将 nnUNet 的 0/-1 转为背景，再按 BraTS 规则转为三通道二值掩码（WT/TC/ET）。  
  - 训练：SpatialPadd → RandSpatialCropd(128³) → 三轴 RandFlipd → RandGaussianNoised/Smooth、RandAdjustContrastd、RandScaleIntensityd。  
  - 验证：CenterSpatialCropd + SpatialPadd 到 128³。  
  - `__getitem__` 返回 `img`、`label`（三通道）、`background_mask`、`mask_code` 等，便于后续 loss 与多模态逻辑使用。

### 3.3 模型模块（models/）

#### 3.3.1 UNet（nnunet.py）

- 6 阶段 Encoder（stride 1,2,2,2,2,2），5 阶段 Decoder（上采样 + 与 Encoder 跳跃连接），每解码阶段带 `seg_layers`。  
- `forward(x, return_features=False)`：  
  - `return_features=False`：仅返回最终分割 logits (B, 3, D, H, W)。  
  - `return_features=True`：返回 `(logits, low_res_input)`，其中 `low_res_input` 为**最后一层解码器输出**（即 seg_layer 前的特征），形状 (B, 8, D, H, W)，供 Boosted 模型接多分割头。

#### 3.3.2 BaselineLateFusion（baseline.py）

- 4 个独立 `UNet` Expert，每个输入 (B,1,D,H,W)，输出 (B,3,D,H,W)。  
- 融合：`output = stack(expert_outputs).mean(dim=0)`，即 **logit 空间静态等权平均**。  
- 用于对比实验与梯度监控（各 Expert 梯度可单独统计）。

#### 3.3.3 BoostedLateFusion（boosted_fusion.py）

- **骨干**：4 个独立 UNet，但前向时用 `return_features=True`，只取解码器最后一层特征 (B, 8, D, H, W)，**不再使用 UNet 自带的 seg_layers**。  
- **可配置分割头 ConfigurableSegHead**：  
  - 每个头：私有部分 `Conv3d(8 → head_hidden_channels, 1×1) + ReLU`（或 Identity，首个头为线性），再通过**共享**的 `shared_head`：`Conv3d(head_hidden_channels → 3, 1×1)` 得到 logits。  
  - 对应论文中 Layer1 私有、Layer2 共享的 Configurable Classifier。  
- **每模态头数**：`modality_heads[m]` 为 ModuleList，初始每模态 1 个头；训练中通过 `add_head(modality_idx)` 动态追加（受 `max_heads_per_modality` 限制）。  
- **前向**：  
  - 对每个模态得到 `decoder_feat`，经该模态所有头得到 `head_logits_list`；  
  - 模态内：`combined_m = sum(head_logits_list)`；  
  - 模态间：`output = mean(combined_0, ..., combined_3)`。  
  - 返回 `output`、`modality_all_logits`、`modality_head_logits`，供损失与 ACA 使用。

### 3.4 损失模块（loss/）

#### 3.4.1 DiceBCEWithLogitsLoss（dice_bce_loss.py）

- 对预测 logits 做 sigmoid 后，计算 Dice Loss（monai）与 BCE，相加后 mean。  
- 用于：Baseline 的融合损失；Boosted 的融合损失以及 ε_all、ε_pre（原始硬标签）。

#### 3.4.2 SustainedBoostingLoss（boosted_loss.py）

- **输入**：`model_output`（含 `output`、`modality_head_logits`）、`labels` (B,3,D,H,W)。  
- **融合损失**：始终计算 `dice_bce(output, labels)`，与 Baseline 对齐。  
- **对每个模态**：  
  - **ε_all**：该模态所有头 logits 求和后的 `dice_bce(combined_logits, labels)`，所有模态都算（用于日志）。  
  - 若该模态 **n_heads > 1**：  
    - **残差标签**：`ŷ = clamp(y - λ_smooth * Σ_{j<n} (y ⊙ σ(logit_j)))`（仅用前 n-1 个头，detach），得到软标签。  
    - **ε**：最新头的 sigmoid 与残差标签的 BCE。  
    - **ε_pre**：前 n-1 个头 logits 之和的 `dice_bce(pre_logits, labels)`。  
  - 该模态 boost 项 = ε + ε_all + ε_pre。  
- **归一化**：`boost_loss_normalized = sum(各模态 boost) / max(1, 有多个头的模态数)`。  
- **总损失**：`total_loss = fused_loss + lambda_boost * boost_loss_normalized`。  
- 返回 `total_loss` 与 `loss_details`（便于日志与监控）。

### 3.5 训练模块（train/）

#### 3.5.1 Baseline：trainer.py

- **StepRunner**：前向 → DiceBCE 损失 → 反向；可选在 4 个 Expert 的 encoder 首层注册 hook，统计各模态梯度范数/比例/平衡度。  
- **EpochRunner**：遍历 DataLoader，累积 loss 与 extra 指标，返回 epoch 日志。  
- **train_model**：按 epoch 循环，按 `val_freq` 做验证，根据 `monitor`（如 val_loss）保存最佳 `ckpt_bst.pt`，可选早停；历史写入 `hist.json`。

#### 3.5.2 Boosted：trainer_boosted.py

- **AdaptiveClassifierAssignment (ACA)**：  
  - `compute_confidence_scores(model, dataloader, device)`：模型 eval，遍历 dataloader，对每个模态用 `modality_all_logits` 的 sigmoid 在前景体素上做平均，再对样本求平均，得到 4 维置信度列表。  
  - `should_check(epoch)`：是否到达检查点（每 `check_interval_epochs`）。  
  - `assign(model, scores)`：若 `s_max - sigma*s_min > tau` 且最弱模态头数未达上限，则 `model.add_head(weak_idx)`，返回被加头的模态索引。  
- **StepRunner**：前向得到 dict → `loss_fn(model_output, labels)` 得到 `(total_loss, loss_details)` → backward → step；可选与 baseline 类似的梯度监控（针对 backbones 首层）。  
- **train_model**：每 epoch 训练后，若 ACA 需要检查则计算置信度、执行 assign；若某模态新增头，则把新头的参数加入 optimizer 的 param_groups。验证与最佳模型保存逻辑同 baseline；日志中可打印 n_heads、ACA_added_modality 等。

### 3.6 入口与流水线

- **train/entry.py**：  
  - `parse_args()`：解析 mode（baseline/boosted）、fold、lr、batch、epochs、Boosted 的 λ、ACA 参数等。  
  - `run_baseline(args)`：构建 BaselineLateFusion、SingleStreamDataset、DiceBCEWithLogitsLoss、Adam；调用 `train_model_baseline`；训练结束后加载 `ckpt_bst.pt` 在 val 上评估，写 `eval_res.json`。  
  - `run_boosted(args)`：构建 BoostedLateFusion、SustainedBoostingLoss、ACA；调用 `train_model_boosted`；若启用 wandb 则注册 log_callback 记录 val_dice 等；同样在结束后用最佳 ckpt 做 val 评估并写 `eval_res.json`。  

- **pipeline.py / pipeline_boosted.py**：用 config 拼好 `args`（含 `results_dir`），直接调用 `run_baseline` 或 `run_boosted`。  

- **sweep_train.py**：W&B sweep，对 lr、lambda_smooth、lambda_boost 做 grid，每个 run 调用 `run_boosted`。

---

## 四、数据流

### 4.1 从数据到 logits（Baseline）

```
磁盘 .npy (4, D, H, W) + _seg.npy
  → SingleStreamDataset（增强、标签转三通道）
  → batch['img'] (B, 4, 128, 128, 128), batch['label'] (B, 3, 128, 128, 128)
  → 按通道拆成 4 个 (B, 1, 128³)
  → Expert_i = UNet(x_i) → (B, 3, 128³)
  → output = mean(Expert_0, ..., Expert_3)  (B, 3, 128³)
  → Loss = DiceBCE(sigmoid(output), label)
```

### 4.2 从数据到 logits（Boosted）

```
同上得到 batch['img'], batch['label']
  → 按通道拆成 4 个 (B, 1, 128³)
  → backbone_m(x_m) 且 return_features=True
      → logits_unused, decoder_feat (B, 8, 128³)
  → 对 modality m：head_1(decoder_feat), head_2(decoder_feat), ... → list of (B, 3, 128³)
  → combined_m = sum(head_logits_list)
  → output = mean(combined_0, ..., combined_3)  (B, 3, 128³)
  → 同时得到 modality_all_logits、modality_head_logits 供 loss 与 ACA
```

### 4.3 Boosted 损失与梯度流

```
total_loss = fused_loss + lambda_boost * boost_loss_normalized
  ├─ fused_loss → 对 output 求导 → 等权回传到 4 个 combined_m → 再回传到各头与 backbone
  └─ boost_loss_normalized
       ├─ 每模态 (ε + ε_all + ε_pre)
       │    ├─ ε：仅最新头 + 残差标签，梯度只更新该头与对应 backbone
       │    ├─ ε_all：combined_logits，梯度更新该模态所有头与 backbone
       │    └─ ε_pre：前序头之和，梯度更新前序头与 backbone，保护已有头不退化
       └─ 共享头被所有模态、所有头共用，梯度来自多处，促进语义空间一致
```

### 4.4 ACA 与置信度数据流

```
每 ACA_CHECK_INTERVAL 个 epoch：
  → model.eval()，遍历 train_dl
  → 对每个 batch：model(x) → modality_all_logits
  → 每模态：combined_probs = sigmoid(modality_all_logits[m])
  → 前景掩码 (y>0.5)，confidence_m = mean_over_samples( mean_over_voxels(masked_probs) )
  → 得到 [s_0, s_1, s_2, s_3]
  → s_max = max(s), s_min = min(s), weak_idx = argmin(s)
  → 若 s_max - sigma*s_min > tau 且 n_heads[weak_idx] < max：add_head(weak_idx)，新头加入 optimizer
```

---

## 五、训练与推理流程

### 5.1 Baseline 训练

1. 读取 configs + 可选命令行，确定 `results_dir`、fold、lr、batch、epochs 等。  
2. 构建 BaselineLateFusion、SingleStreamDataset（train/val）、DiceBCEWithLogitsLoss、Adam。  
3. 每 epoch：StepRunner 前向→损失→反向→step；可选统计 4 个 Expert 首层梯度。  
4. 每 `val_freq` 个 epoch 在 val 上评估，按 val_loss 保存 `ckpt_bst.pt`。  
5. 训练结束后加载 `ckpt_bst.pt`，在 val 上算每样本 Dice，写 `eval_res.json`。

### 5.2 Boosted 训练

1. 同 Baseline 构建数据与设备；构建 BoostedLateFusion、SustainedBoostingLoss、ACA（若未 `--no_aca`）。  
2. 每 epoch：StepRunner 前向（返回 dict）→ SustainedBoostingLoss → 反向 → step；可选梯度监控。  
3. 若 `aca.should_check(epoch)`：在 train_dl 上算置信度 → `aca.assign`，必要时 `add_head` 并把新参数加入 optimizer。  
4. 验证与最佳模型保存同 Baseline；日志含 n_heads、ACA_added_modality、各 modality 的 ε/ε_all/ε_pre 等。  
5. 结束后用最佳 ckpt 做 val 评估；Boosted 推理时使用 `model(x)['output']`。

### 5.3 推理（两模式统一）

- **Baseline**：`pred = model(x)`，即 (B, 3, D, H, W) logits；通常再 `sigmoid(pred) > 0.5` 得二值图。  
- **Boosted**：`out = model(x); pred = out['output']`，同样 (B, 3, D, H, W)，后处理相同。  
- 评估脚本（entry 中）：用 MONAI DiceHelper（sigmoid=True）在 val 上逐样本计算 Dice，写 `eval_res.json`。

---

## 六、配置与运行

### 6.1 关键配置项

| 类型 | 项 | 含义 | 典型值 |
|------|----|------|--------|
| 数据 | DATASET_DIR, FOLD | 数据路径与折 | assets/data, 1 |
| 训练 | N_EPOCHS, BATCH_SIZE, VAL_FREQ | 轮数、批大小、验证间隔 | 400, 4, 5 |
| Boosted | LAMBDA_SMOOTH, LAMBDA_BOOST | 残差平滑、boost 权重 | 0.33, 1.0 |
| Boosted | MAX_HEADS_PER_MODALITY | 每模态最大头数 | 6 |
| ACA | ACA_SIGMA, ACA_TAU, ACA_CHECK_INTERVAL | 置信度判定与检查周期 | 1.0, 0.01, 60 |

### 6.2 常用命令

```bash
# Baseline（config 驱动）
python pipeline.py

# Boosted（config 驱动）
python pipeline_boosted.py

# 命令行覆盖部分参数
python train.py --mode baseline --fold 0 --epochs 200
python train.py --mode boosted --fold 1 --no_aca   # 禁用 ACA，仅多损失

# 快速试跑 2 epoch
QUICK_TEST=1 python pipeline_boosted.py

# W&B 超参扫描（需先配置 WANDB_PROJECT 等）
python sweep_train.py --create_only   # 仅创建 sweep
python sweep_train.py --sweep_id <id> --count 27
```

### 6.3 输出与检查点

- **目录**：`saved_models/{mode}/{exp_name}/{run_name}/`。  
- **文件**：`config_snapshot.json`、`training_*.log`、`hist.json`、`ckpt_bst.pt`（最佳）、`ckpt_final.pt`（最后一轮）、`eval_res.json`（最佳 ckpt 在 val 上的 Dice 汇总）。  
- Boosted 的 `ckpt_bst.pt` 可能包含动态增加的头，加载时需用相同 `head_hidden_channels`、`max_heads_per_modality` 的模型，并 `strict=False` 以兼容不同头数。

---

## 七、小结

本项目在 BraTS2018 四模态分割上实现了“**按置信度监控模态、为弱模态增加 1×1 卷积头 + 辅助损失**”的完整流程：数据与增强、Baseline/Boosted 双模型、DiceBCE 与 Sustained Boosting 损失、ACA 策略与训练/验证/评估入口均打通。理解各模块后，可在此基础上调整 λ、ACA 间隔与阈值、最大头数或损失权重，或扩展为其他数据集与骨干。
