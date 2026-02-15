"""Shared training entry logic: run_baseline, run_boosted, argparse and result-dir helpers."""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceHelper
from pytorch_lightning import seed_everything

from configs import UNetConfig, DatasetConfig, TrainingConfig
from configs_boosted import BoostedConfig
from dataset.processors import SingleStreamDataset
from models.baseline import BaselineLateFusion
from models.boosted_fusion import BoostedLateFusion
from loss.dice_bce_loss import DiceBCEWithLogitsLoss
from loss.boosted_loss import SustainedBoostingLoss
from train.trainer import train_model as train_model_baseline
from train.trainer_boosted import train_model as train_model_boosted, AdaptiveClassifierAssignment

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline or boosted late fusion.')
    parser.add_argument('--mode', type=str, choices=['baseline', 'boosted'], default='boosted',
                        help='Training mode.')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Experiment purpose, e.g. hpo_20260216, 5fold_final.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run subdir name; if omitted, auto-generated from fold and key hyperparams.')
    parser.add_argument('--fold', type=int, default=DatasetConfig.FOLD,
                        help='K-fold index (0-4).')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device index.')
    parser.add_argument('--lr', type=float, default=TrainingConfig.LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=TrainingConfig.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=TrainingConfig.N_EPOCHS)
    parser.add_argument('--val_freq', type=int, default=TrainingConfig.VAL_FREQ)
    parser.add_argument('--seed', type=int, default=TrainingConfig.RANDOM_SEED)
    parser.add_argument('--apply_early_stopping', action='store_true', default=TrainingConfig.APPLY_EARLY_STOPPING)
    parser.add_argument('--quick_test', action='store_true', help='Run only 2 epochs.')
    parser.add_argument('--lambda_smooth', type=float, default=BoostedConfig.LAMBDA_SMOOTH)
    parser.add_argument('--lambda_boost', type=float, default=BoostedConfig.LAMBDA_BOOST)
    parser.add_argument('--head_hidden_channels', type=int, default=BoostedConfig.HEAD_HIDDEN_CHANNELS)
    parser.add_argument('--max_heads_per_modality', type=int, default=BoostedConfig.MAX_HEADS_PER_MODALITY)
    parser.add_argument('--aca_sigma', type=float, default=BoostedConfig.ACA_SIGMA)
    parser.add_argument('--aca_tau', type=float, default=BoostedConfig.ACA_TAU)
    parser.add_argument('--aca_check_interval', type=int, default=BoostedConfig.ACA_CHECK_INTERVAL)
    args = parser.parse_args()
    if args.quick_test:
        args.epochs = 2
    return args


def _default_run_name(args):
    if getattr(args, 'run_name', None) is not None and args.run_name:
        return args.run_name
    if args.mode == 'baseline':
        return f'fold{args.fold}'
    lr_s = f'{args.lr:.4f}'.rstrip('0').rstrip('.')
    ls_s = f'{args.lambda_smooth:.2f}'.rstrip('0').rstrip('.')
    lb_s = f'{args.lambda_boost:.2f}'.rstrip('0').rstrip('.')
    return f'fold{args.fold}_lr{lr_s}_ls{ls_s}_lb{lb_s}'


def _results_dir(args):
    run_name = _default_run_name(args)
    return os.path.join('saved_models', args.mode, args.exp_name, run_name)


def _write_config_snapshot(results_dir, args):
    d = vars(args).copy()
    for k, v in list(d.items()):
        if not isinstance(v, (int, float, str, bool, type(None))):
            d[k] = str(v)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'config_snapshot.json'), 'w') as f:
        json.dump(d, f, indent=2)


def run_baseline(args):
    """Baseline Late Fusion: build model/data, train, then eval best ckpt on val. Uses args.results_dir."""
    results_dir = getattr(args, 'results_dir', None)
    if results_dir is None:
        results_dir = _results_dir(args)
    args.results_dir = results_dir
    _write_config_snapshot(results_dir, args)

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed=args.seed)

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
        fold=args.fold,
        unimodality=False,
    )
    val_ds = SingleStreamDataset(
        sample_type='val',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=args.fold,
        unimodality=False,
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=TrainingConfig.VAL_BATCH_SIZE, shuffle=False, num_workers=4)

    loss_fn = DiceBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def make_dice_channel(channel_index):
        dice_fn = DiceHelper(
            include_background=True, sigmoid=True, activate=True,
            get_not_nans=False, reduction='mean',
        )
        def fn(preds, labels):
            return dice_fn(
                preds[:, channel_index : channel_index + 1],
                labels[:, channel_index : channel_index + 1],
            )
        return fn

    metrics_dict = {
        'dice': DiceHelper(
            include_background=True, sigmoid=True, activate=True,
            get_not_nans=False, reduction='mean',
        ),
        'dice_wt': make_dice_channel(0),
        'dice_tc': make_dice_channel(1),
        'dice_et': make_dice_channel(2),
    }

    train_model_baseline(
        model, optimizer, loss_fn, metrics_dict,
        train_data=train_dl, val_data=val_dl,
        val_freq=args.val_freq, num_epoch=args.epochs,
        results_dir=results_dir, device=device,
        apply_early_stopping=args.apply_early_stopping,
        monitor='val_loss', eval_mode='min',
        monitor_modality_grad=True,
    )

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
    ckpt_path = os.path.join(results_dir, 'ckpt_bst.pt')
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dice_fn = DiceHelper(
        include_background=True, sigmoid=True, activate=True,
        get_not_nans=False, reduction='none',
    )
    test_ds = SingleStreamDataset(
        sample_type='val',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=args.fold,
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
    res = np.array(res)
    eval_res = {'val_dice_mean': str(np.nanmean(res, axis=0))}
    with open(os.path.join(results_dir, 'eval_res.json'), 'w') as f:
        json.dump(eval_res, f)
    print('Eval result (best ckpt on val):', eval_res)


def run_boosted(args):
    """Boosted Late Fusion: build model/data, train with ACA, then eval best ckpt on val. Uses args.results_dir."""
    results_dir = getattr(args, 'results_dir', None)
    if results_dir is None:
        results_dir = _results_dir(args)
    args.results_dir = results_dir
    _write_config_snapshot(results_dir, args)

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed=args.seed)

    model = BoostedLateFusion(
        input_channels=UNetConfig.INPUT_CHANNELS,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION,
        n_experts=4,
        head_hidden_channels=args.head_hidden_channels,
        max_heads_per_modality=args.max_heads_per_modality,
    ).to(device)

    train_ds = SingleStreamDataset(
        sample_type='train',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=args.fold,
        unimodality=False,
    )
    val_ds = SingleStreamDataset(
        sample_type='val',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=args.fold,
        unimodality=False,
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BoostedConfig.VAL_BATCH_SIZE, shuffle=False, num_workers=4)

    loss_fn = SustainedBoostingLoss(
        lambda_smooth=args.lambda_smooth,
        lambda_boost=args.lambda_boost,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    aca = AdaptiveClassifierAssignment(
        sigma=args.aca_sigma,
        tau=args.aca_tau,
        check_interval_epochs=args.aca_check_interval,
        max_heads=args.max_heads_per_modality,
    )

    def make_dice_channel(channel_index):
        dice_fn = DiceHelper(
            include_background=True, sigmoid=True, activate=True,
            get_not_nans=False, reduction='mean',
        )
        def fn(preds, labels):
            return dice_fn(
                preds[:, channel_index : channel_index + 1],
                labels[:, channel_index : channel_index + 1],
            )
        return fn

    metrics_dict = {
        'dice': DiceHelper(
            include_background=True, sigmoid=True, activate=True,
            get_not_nans=False, reduction='mean',
        ),
        'dice_wt': make_dice_channel(0),
        'dice_tc': make_dice_channel(1),
        'dice_et': make_dice_channel(2),
    }

    best_val_metrics = {'val_dice_wt': [], 'val_dice_tc': [], 'val_dice_et': []}

    def _wandb_log_callback(epoch, train_log, val_log):
        if wandb is None or not wandb.run or not val_log:
            return
        d = {'epoch': epoch}
        v_train = train_log.get('train_loss')
        if v_train is not None and isinstance(v_train, (int, float)):
            d['train_loss'] = v_train
        for k in ('val_loss', 'val_dice', 'val_dice_wt', 'val_dice_tc', 'val_dice_et'):
            v = val_log.get(k)
            if v is not None and isinstance(v, (int, float)):
                d[k] = v
                if k in best_val_metrics:
                    best_val_metrics[k].append(v)
        wandb.log(d)

    log_callback = _wandb_log_callback if (wandb is not None and wandb.run) else None

    train_model_boosted(
        model, optimizer, loss_fn, metrics_dict,
        train_data=train_dl, val_data=val_dl,
        val_freq=args.val_freq, num_epoch=args.epochs,
        results_dir=results_dir, device=device,
        apply_early_stopping=args.apply_early_stopping,
        monitor='val_loss', eval_mode='min',
        monitor_modality_grad=True, aca=aca,
        log_callback=log_callback,
    )

    if wandb is not None and wandb.run:
        for k in ('val_dice_wt', 'val_dice_tc', 'val_dice_et'):
            if best_val_metrics[k]:
                wandb.run.summary[f'best_{k}'] = max(best_val_metrics[k])
        if best_val_metrics['val_dice_et']:
            wandb.run.summary['val_dice_et'] = max(best_val_metrics['val_dice_et'])

    model = BoostedLateFusion(
        input_channels=UNetConfig.INPUT_CHANNELS,
        n_classes=UNetConfig.N_CLASSES,
        n_stages=UNetConfig.N_STAGES,
        n_features_per_stage=UNetConfig.N_FEATURES_PER_STAGE,
        kernel_size=UNetConfig.KERNEL_SIZES,
        strides=UNetConfig.STRIDES,
        apply_deep_supervision=UNetConfig.APPLY_DEEP_SUPERVISION,
        n_experts=4,
        head_hidden_channels=args.head_hidden_channels,
        max_heads_per_modality=args.max_heads_per_modality,
    ).to(device)
    ckpt_path = os.path.join(results_dir, 'ckpt_bst.pt')
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.eval()

    dice_fn = DiceHelper(
        include_background=True, sigmoid=True, activate=True,
        get_not_nans=False, reduction='none',
    )
    test_ds = SingleStreamDataset(
        sample_type='val',
        dataset_dir=DatasetConfig.DATASET_DIR,
        splits_file_path=DatasetConfig.SPLITS_FILE_PATH,
        drop_mode=DatasetConfig.DROP_MODE,
        possible_dropped_modality_combinations=DatasetConfig.POSSIBLE_DROPPED_MODALITY_COMBINATIONS,
        fold=args.fold,
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
    with open(os.path.join(results_dir, 'eval_res.json'), 'w') as f:
        json.dump(eval_res, f)
    print('Eval result (best ckpt on val):', eval_res)
