# W&B Sweeps: Boost 超参敏感性

使用 Weights & Biases Sweeps 对 Boosted 模型做网格搜索，固定 fold=1、epochs=400，搜索 `lr`、`lambda_smooth`、`lambda_boost`，以 **val_dice_et**（ET）为优化目标。

## 依赖

- `pip install wandb`
- 已执行 `wandb login`（或设置 `WANDB_API_KEY`）

## 搜索空间（网格，共 27 组）

- **lr**: [1e-4, 5e-4, 1e-3]
- **lambda_smooth**: [0.25, 0.33, 0.5]
- **lambda_boost**: [0.5, 1.0, 1.5]

固定：fold=1，epochs=400，HEAD_HIDDEN_CHANNELS=8，其余与 `configs_boosted.py` 一致。

## 创建 Sweep 并运行 Agent

**方式一：脚本自动创建 sweep，再启动 agent**

```bash
# 创建 sweep 并立即跑满 27 组（单卡）
python sweep_train.py --project reb-unet --count 27
```

**方式二：先创建 sweep，得到 sweep_id 后再跑 agent（适合多卡并行）**

```bash
# 1. 只创建 sweep，打印 sweep_id
python sweep_train.py --project reb-unet --create_only

# 2. 单卡跑满 27 组
python sweep_train.py --sweep_id <上一步打印的id> --project reb-unet --count 27

# 3. 5 卡并行：每张卡起一个 agent，共享同一 sweep_id
CUDA_VISIBLE_DEVICES=0 python sweep_train.py --sweep_id <id> --project reb-unet &
CUDA_VISIBLE_DEVICES=1 python sweep_train.py --sweep_id <id> --project reb-unet &
CUDA_VISIBLE_DEVICES=2 python sweep_train.py --sweep_id <id> --project reb-unet &
CUDA_VISIBLE_DEVICES=3 python sweep_train.py --sweep_id <id> --project reb-unet &
CUDA_VISIBLE_DEVICES=4 python sweep_train.py --sweep_id <id> --project reb-unet &
```

每个 agent 会从 W&B 拉取下一组未跑的 (lr, lambda_smooth, lambda_boost)，训练时使用本进程可见的那张卡（每台机器上 `device_id=0` 即当前可见的第一张卡）。

## 结果

- **本地**：`saved_models/boosted/sweep_boosted/<wandb_run_id>/`，内含 ckpt、log、config_snapshot.json。
- **W&B**：每个 run 记录 `val_dice_wt`、`val_dice_tc`、`val_dice_et` 及 train/val loss；Sweep 按 **val_dice_et** 排序，ET 为最重要指标。

## 配置文件

- `sweep_boosted_grid.yaml`：与脚本内 `DEFAULT_SWEEP_CONFIG` 一致，供查阅或复制到 W&B UI 手动创建 sweep。
