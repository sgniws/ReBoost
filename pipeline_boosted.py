"""Boosted Late Fusion: Sustained Boosting + ACA training entry point."""
import os
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceHelper
from pytorch_lightning import seed_everything
from datetime import datetime

from configs import UNetConfig, DatasetConfig
from configs_boosted import BoostedConfig
from dataset.processors import SingleStreamDataset
from models.boosted_fusion import BoostedLateFusion
from loss.boosted_loss import SustainedBoostingLoss
from train.trainer_boosted import train_model, AdaptiveClassifierAssignment


def main():
    print('########################################################################')
    print('Boosted Late Fusion (Sustained Boosting + ACA) — configs:')
    with open('configs_boosted.py', 'r') as f:
        print(f.read())
    print('########################################################################')

    if not os.path.exists(BoostedConfig.RESULTS_DIR):
        os.makedirs(BoostedConfig.RESULTS_DIR)
    shutil.copyfile(
        'configs_boosted.py',
        os.path.join(
            BoostedConfig.RESULTS_DIR,
            f'current_configs_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.py',
        ),
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed=BoostedConfig.RANDOM_SEED)

    model = BoostedLateFusion(
        input_channels=UNetConfig.INPUT_CHANNELS,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION,
        n_experts=4,
        head_hidden_channels=BoostedConfig.HEAD_HIDDEN_CHANNELS,
        max_heads_per_modality=BoostedConfig.MAX_HEADS_PER_MODALITY,
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
        batch_size=BoostedConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BoostedConfig.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    loss_fn = SustainedBoostingLoss(
        lambda_smooth=BoostedConfig.LAMBDA_SMOOTH,
        lambda_boost=BoostedConfig.LAMBDA_BOOST,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=BoostedConfig.LEARNING_RATE)

    aca = AdaptiveClassifierAssignment(
        sigma=BoostedConfig.ACA_SIGMA,
        tau=BoostedConfig.ACA_TAU,
        check_interval_epochs=BoostedConfig.ACA_CHECK_INTERVAL,
        max_heads=BoostedConfig.MAX_HEADS_PER_MODALITY,
    )

    def make_dice_channel(channel_index):
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

    num_epoch = BoostedConfig.N_EPOCHS
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
        val_freq=BoostedConfig.VAL_FREQ,
        num_epoch=num_epoch,
        results_dir=BoostedConfig.RESULTS_DIR,
        device=device,
        apply_early_stopping=BoostedConfig.APPLY_EARLY_STOPPING,
        monitor='val_loss',
        eval_mode='min',
        monitor_modality_grad=True,
        aca=aca,
    )

    # Load best checkpoint and run final validation evaluation
    model = BoostedLateFusion(
        input_channels=UNetConfig.INPUT_CHANNELS,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION,
        n_experts=4,
        head_hidden_channels=BoostedConfig.HEAD_HIDDEN_CHANNELS,
        max_heads_per_modality=BoostedConfig.MAX_HEADS_PER_MODALITY,
    ).to(device)
    ckpt_path = os.path.join(BoostedConfig.RESULTS_DIR, 'ckpt_bst.pt')
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.eval()

    dice_fn = DiceHelper(
        include_background=True,
        sigmoid=True,
        activate=True,
        get_not_nans=False,
        reduction='none',
    )
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
            out = model(x)
            pred = out['output']
            res.append(dice_fn(pred, y).cpu().numpy())

    res = np.array(res)
    eval_res = {'val_dice_mean': str(np.nanmean(res, axis=0))}
    with open(os.path.join(BoostedConfig.RESULTS_DIR, 'eval_res.json'), 'w') as f:
        json.dump(eval_res, f)
    print('Eval result (best ckpt on val):', eval_res)


if __name__ == '__main__':
    main()
