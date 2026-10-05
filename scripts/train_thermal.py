#!/usr/bin/env python3
"""Train a thermal model with train-only scaling and recursive validation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/private/l14_merged_data_with_rain.csv')
    parser.add_argument('--mode', choices=['ac', 'nv'], required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--chronological', action='store_true', help='Use all pre-validation segments for another formatted dataset.')
    parser.add_argument('--validation-start', default='2024-10-01')
    parser.add_argument('--test-start', default='2024-10-10')
    parser.add_argument('--study-end', default='2024-10-31')
    args = parser.parse_args()
    from mmv4ef.output import require_ignored_output
    from mmv4ef.thermal_training import train_mode
    require_ignored_output(args.output_dir)
    selection = train_mode(args.data, args.mode, args.output_dir, seeds=args.seeds, epochs=args.epochs,
                           validation_start=args.validation_start, test_start=args.test_start,
                           study_end=args.study_end, retained_training=not args.chronological)
    print(selection)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
