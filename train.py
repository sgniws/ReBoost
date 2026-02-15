"""Unified training entry: argparse-based CLI, results under saved_models/{mode}/{exp_name}/{run_name}/."""
import os
from train.entry import parse_args, _default_run_name, run_baseline, run_boosted


def main():
    args = parse_args()
    run_name = _default_run_name(args)
    results_dir = os.path.join('saved_models', args.mode, args.exp_name, run_name)
    args.run_name = run_name
    args.results_dir = results_dir
    print('Results dir:', results_dir)
    if args.mode == 'baseline':
        run_baseline(args)
    else:
        run_boosted(args)


if __name__ == '__main__':
    main()
