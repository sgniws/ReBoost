"""Baseline Late Fusion: single-stage end-to-end training entry point (config-based, calls run_baseline)."""
import os
import argparse

from configs import DatasetConfig, TrainingConfig
from train.entry import _default_run_name, run_baseline


def main():
    # Build args from config (same behavior as before; results go to saved_models/baseline/default/fold{N})
    exp_name = 'default'
    run_name = _default_run_name(argparse.Namespace(
        mode='baseline', run_name=None, fold=DatasetConfig.FOLD,
        lr=TrainingConfig.LEARNING_RATE, lambda_smooth=0.33, lambda_boost=1.0,
    ))
    results_dir = os.path.join('saved_models', 'baseline', exp_name, run_name)

    args = argparse.Namespace(
        mode='baseline',
        exp_name=exp_name,
        run_name=run_name,
        results_dir=results_dir,
        fold=DatasetConfig.FOLD,
        device_id=0,
        lr=TrainingConfig.LEARNING_RATE,
        batch_size=TrainingConfig.BATCH_SIZE,
        epochs=TrainingConfig.N_EPOCHS,
        val_freq=TrainingConfig.VAL_FREQ,
        seed=TrainingConfig.RANDOM_SEED,
        apply_early_stopping=TrainingConfig.APPLY_EARLY_STOPPING,
        quick_test=os.environ.get('QUICK_TEST') == '1',
        # Boosted-only (ignored by run_baseline)
        lambda_smooth=0.33,
        lambda_boost=1.0,
        head_hidden_channels=8,
        max_heads_per_modality=8,
        aca_sigma=1.0,
        aca_tau=0.01,
        aca_check_interval=60,
    )
    if args.quick_test:
        args.epochs = 2
        print('QUICK_TEST=1: running only 2 epochs.')
    if 'NUM_EPOCHS' in os.environ:
        args.epochs = int(os.environ['NUM_EPOCHS'])
        print(f'NUM_EPOCHS={args.epochs}: running {args.epochs} epochs.')

    run_baseline(args)


if __name__ == '__main__':
    main()
