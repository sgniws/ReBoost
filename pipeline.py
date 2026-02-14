"""Baseline Late Fusion: single-stage end-to-end training entry point."""
import os
import shutil
import json
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceHelper
from pytorch_lightning import seed_everything
from datetime import datetime

from configs import UNetConfig, DatasetConfig, TrainingConfig
from dataset.processors import SingleStreamDataset
from models.baseline import BaselineLateFusion
from loss.dice_bce_loss import DiceBCEWithLogitsLoss
from train.trainer import train_model


def main():
    print('########################################################################')
    print('Baseline Late Fusion — configs:')
    with open('configs.py', 'r') as f:
        print(f.read())
    print('########################################################################')

    if not os.path.exists(TrainingConfig.RESULTS_DIR):
        os.makedirs(TrainingConfig.RESULTS_DIR)
    shutil.copyfile(
        'configs.py',
        os.path.join(TrainingConfig.RESULTS_DIR, f'current_configs_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.py')
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed=TrainingConfig.RANDOM_SEED)

    model = BaselineLateFusion(
        input_channels=UNetConfig.INPUT_CHANNELS,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION,
        n_experts=4,
    ).to(device)

    train_ds = SingleStreamDataset(
        sample_type='train',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=DatasetConfig.FOLD,
        unimodality=False,
    )
    val_ds = SingleStreamDataset(
        sample_type='val',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=DatasetConfig.FOLD,
        unimodality=False,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=TrainingConfig.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    loss_fn = DiceBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)

    def make_dice_channel(channel_index):
        """Return a callable that computes mean Dice for one channel (WT/TC/ET)."""
        dice_fn = DiceHelper(
            include_background=True,
            sigmoid=True,
            activate=True,
            get_not_nans=False,
            reduction='mean',
        )

        def fn(preds, labels):
            return dice_fn(
                preds[:, channel_index : channel_index + 1],
                labels[:, channel_index : channel_index + 1],
            )

        return fn

    metrics_dict = {
        'dice': DiceHelper(
            include_background=True,
            sigmoid=True,
            activate=True,
            get_not_nans=False,
            reduction='mean',
        ),
        'dice_wt': make_dice_channel(0),
        'dice_tc': make_dice_channel(1),
        'dice_et': make_dice_channel(2),
    }

    num_epoch = TrainingConfig.N_EPOCHS
    if os.environ.get('QUICK_TEST') == '1':
        num_epoch = 2
        print('QUICK_TEST=1: running only 2 epochs.')
    if 'NUM_EPOCHS' in os.environ:
        num_epoch = int(os.environ['NUM_EPOCHS'])
        print(f'NUM_EPOCHS={num_epoch}: running {num_epoch} epochs.')

    train_model(
        model,
        optimizer,
        loss_fn,
        metrics_dict,
        train_data=train_dl,
        val_data=val_dl,
        val_freq=TrainingConfig.VAL_FREQ,
        num_epoch=num_epoch,
        results_dir=TrainingConfig.RESULTS_DIR,
        device=device,
        apply_early_stopping=TrainingConfig.APPLY_EARLY_STOPPING,
        monitor='val_loss',
        eval_mode='min',
        monitor_modality_grad=True,
    )

    # Load best checkpoint and run final validation evaluation
    model = BaselineLateFusion(
        input_channels=UNetConfig.INPUT_CHANNELS,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION,
        n_experts=4,
    ).to(device)
    ckpt_path = os.path.join(TrainingConfig.RESULTS_DIR, 'ckpt_bst.pt')
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dice_fn = DiceHelper(include_background=True, sigmoid=True, activate=True, get_not_nans=False, reduction='none')
    test_ds = SingleStreamDataset(
        sample_type='val',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=DatasetConfig.FOLD,
        unimodality=False,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    res = []
    with torch.no_grad():
        for batch in test_dl:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            pred = model(x)
            res.append(dice_fn(pred, y).cpu().numpy())

    import numpy as np
    res = np.array(res)
    eval_res = {'val_dice_mean': str(np.nanmean(res, axis=0))}
    with open(os.path.join(TrainingConfig.RESULTS_DIR, 'eval_res.json'), 'w') as f:
        json.dump(eval_res, f)
    print('Eval result (best ckpt on val):', eval_res)


if __name__ == '__main__':
    main()
