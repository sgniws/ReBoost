"""Boosted Late Fusion: Sustained Boosting + ACA training entry point (config-based, calls run_boosted)."""
import os
import argparse

from configs import DatasetConfig
from configs_boosted import BoostedConfig
from train.entry import _default_run_name, run_boosted


def main():
    # Build args from config (same behavior as before; results go to saved_models/boosted/default/fold{N}_lr...)
    exp_name = 'default'
    args_for_run_name = argparse.Namespace(
        mode='boosted', run_name=None, fold=DatasetConfig.FOLD,
        lr=BoostedConfig.LEARNING_RATE, lambda_smooth=BoostedConfig.LAMBDA_SMOOTH,
        lambda_boost=BoostedConfig.LAMBDA_BOOST,
    )
    run_name = _default_run_name(args_for_run_name)
    results_dir = os.path.join('saved_models', 'boosted', exp_name, run_name)

    args = argparse.Namespace(
        mode='boosted',
        exp_name=exp_name,
        run_name=run_name,
        results_dir=results_dir,
        fold=DatasetConfig.FOLD,
        device_id=0,
        lr=BoostedConfig.LEARNING_RATE,
        batch_size=BoostedConfig.BATCH_SIZE,
        epochs=BoostedConfig.N_EPOCHS,
        val_freq=BoostedConfig.VAL_FREQ,
        seed=BoostedConfig.RANDOM_SEED,
        apply_early_stopping=BoostedConfig.APPLY_EARLY_STOPPING,
        quick_test=os.environ.get('QUICK_TEST') == '1',
        lambda_smooth=BoostedConfig.LAMBDA_SMOOTH,
        lambda_boost=BoostedConfig.LAMBDA_BOOST,
        head_hidden_channels=BoostedConfig.HEAD_HIDDEN_CHANNELS,
        max_heads_per_modality=BoostedConfig.MAX_HEADS_PER_MODALITY,
        aca_sigma=BoostedConfig.ACA_SIGMA,
        aca_tau=BoostedConfig.ACA_TAU,
        aca_check_interval=BoostedConfig.ACA_CHECK_INTERVAL,
    )
    if args.quick_test:
        args.epochs = 2
        print('QUICK_TEST=1: running only 2 epochs.')
    if 'NUM_EPOCHS' in os.environ:
        args.epochs = int(os.environ['NUM_EPOCHS'])
        print(f'NUM_EPOCHS={args.epochs}: running {args.epochs} epochs.')

    run_boosted(args)


if __name__ == '__main__':
    main()
