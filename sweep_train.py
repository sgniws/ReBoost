"""W&B Sweep agent for Boosted hyperparameter search (grid: lr, lambda_smooth, lambda_boost)."""
import argparse
import os

import wandb

from configs_boosted import BoostedConfig
from train.entry import run_boosted, _default_run_name


# Default sweep config (same as sweep_boosted_grid.yaml)
DEFAULT_SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {'name': 'val_dice_et', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [0.0001, 0.0005, 0.001]},
        'lambda_smooth': {'values': [0.25, 0.33, 0.5]},
        'lambda_boost': {'values': [0.5, 1.0, 1.5]},
    },
}


def train_fn():
    """Called by wandb.agent; wandb.config is injected with lr, lambda_smooth, lambda_boost."""
    wandb.init(project=os.environ.get('WANDB_PROJECT', 'reb-unet'), config=wandb.config)
    cfg = wandb.config

    args = argparse.Namespace(
        mode='boosted',
        exp_name='sweep_boosted',
        run_name=None,
        results_dir=None,
        fold=1,
        device_id=0,
        lr=float(cfg.lr),
        batch_size=BoostedConfig.BATCH_SIZE,
        epochs=400,
        val_freq=BoostedConfig.VAL_FREQ,
        seed=BoostedConfig.RANDOM_SEED,
        apply_early_stopping=BoostedConfig.APPLY_EARLY_STOPPING,
        quick_test=False,
        lambda_smooth=float(cfg.lambda_smooth),
        lambda_boost=float(cfg.lambda_boost),
        head_hidden_channels=8,
        max_heads_per_modality=BoostedConfig.MAX_HEADS_PER_MODALITY,
        aca_sigma=BoostedConfig.ACA_SIGMA,
        aca_tau=BoostedConfig.ACA_TAU,
        aca_check_interval=BoostedConfig.ACA_CHECK_INTERVAL,
    )
    args.run_name = wandb.run.id if wandb.run else _default_run_name(args)
    args.results_dir = os.path.join('saved_models', args.mode, args.exp_name, args.run_name)

    run_boosted(args)


def main():
    parser = argparse.ArgumentParser(description='Run W&B Sweep agent for Boosted grid search.')
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='Existing sweep ID (from W&B UI or from a previous --create_only run).')
    parser.add_argument('--project', type=str, default='reb-unet', help='W&B project name.')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (team/user).')
    parser.add_argument('--count', type=int, default=27,
                        help='Max number of runs for this agent (grid size = 27).')
    parser.add_argument('--create_only', action='store_true',
                        help='Only create sweep from config and print sweep_id; do not run agent.')
    args = parser.parse_args()

    if args.sweep_id is None:
        sweep_id = wandb.sweep(DEFAULT_SWEEP_CONFIG, project=args.project, entity=args.entity)
        print('Created sweep:', sweep_id)
        if args.create_only:
            print('Run agent with: python sweep_train.py --sweep_id', sweep_id, '--project', args.project)
            return
    else:
        sweep_id = args.sweep_id

    wandb.agent(
        sweep_id,
        function=train_fn,
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


if __name__ == '__main__':
    main()
