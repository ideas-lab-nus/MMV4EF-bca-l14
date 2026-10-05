#!/usr/bin/env python3
"""Run the clean main-study or sensitivity notebook without a Jupyter server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/private/l14_merged_data_with_rain.csv')
    parser.add_argument('--study', choices=['main', 'sweep'], default='main')
    parser.add_argument('--future', type=Path, default=ROOT / 'outputs/lstm64/mpc_future_disturbances.csv.gz')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-days', type=int)
    parser.add_argument('--window', choices=['full', 'smoke'], default='full')
    parser.add_argument('--check', action='store_true', help='Validate data and model loading without starting Gurobi.')
    args = parser.parse_args()
    if args.max_days is not None and args.max_days < 1:
        parser.error('--max-days must be positive')
    if args.window == 'smoke' and args.max_days is None:
        args.max_days = 1

    from mmv4ef.controller import make_default_hooks
    from mmv4ef.lstm64_checkpoint import load_lstm64_checkpoint
    hooks = make_default_hooks(data_path=args.data, segment_split='test+val', test_date_policy='union')
    segments = hooks._segments_cached()
    if not segments:
        raise ValueError('No complete controller-comparison days. Check dates and fields in docs/DATA_REQUIREMENTS.md.')
    # A retained day must cover every minute of the occupied window.
    import pandas as pd
    for segment in segments:
        date = segment.date.iloc[0].date().isoformat()
        expected = pd.date_range(date + ' 07:30', date + ' 19:00', freq='min')
        if not pd.DatetimeIndex(segment.date).equals(expected):
            raise ValueError(f'Incomplete occupied window for {date}; require 07:30 through 19:00 inclusive.')
    for target in ['temperature', 'relative_humidity', 'wind_speed', 'wind_direction', 'solar']:
        for seed in [17, 29, 43]:
            load_lstm64_checkpoint(ROOT / f'models/lstm64/checkpoints/univariate/{target}/seed_{seed}.pt', target, expected_seed=seed)
    if args.check:
        print(f'Input schema and all 17 neural checkpoints passed; {len(segments)} complete comparison days.')
        return 0

    if args.study == 'main' and not args.future.is_file():
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_lstm64_forecasts.py'),
                        '--data', str(args.data.resolve()), '--output-dir', str(args.future.parent.resolve())],
                       cwd=ROOT, check=True)
        generated = args.future.parent / 'mpc_future_disturbances.csv.gz'
        if args.future.name != generated.name:
            raise ValueError('Automatic forecast generation uses mpc_future_disturbances.csv.gz; supply an existing --future for a different filename.')
    output = args.output_dir or ROOT / ('outputs/simulations/main' if args.study == 'main' else 'outputs/simulations/weight_sweep')
    # Generated tables stay local and are ignored. Refuse a committable destination.
    from mmv4ef.output import require_ignored_output
    require_ignored_output(output)
    os.environ['MMV4EF_DATA'] = str(args.data.resolve())
    os.environ['MMV4EF_LSTM64_FUTURE'] = str(args.future.resolve())
    os.environ['MMV4EF_OUTPUT_DIR'] = str(output.resolve())
    os.environ['MMV4EF_WINDOW'] = args.window
    if args.max_days is None:
        os.environ.pop('MMV4EF_MAX_SEGMENTS', None)
    else:
        os.environ['MMV4EF_MAX_SEGMENTS'] = str(args.max_days)
    os.chdir(ROOT)
    name = 'main_study_observed_lstm64' if args.study == 'main' else 'temperature_slack_sensitivity'
    notebook = json.loads((ROOT / f'notebooks/{name}.ipynb').read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'{name}:cell{index}', 'exec'), namespace)
    print(f'{args.study} completed and validated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
