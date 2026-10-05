"""Training-only scaling and recursive-validation checkpoint selection."""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .controller import CNNLSTM
from .data import STATES, INPUTS, MODE, load_formatted_data

VALIDATION_EPOCHS = (1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 165, 200)
RETAINED_TRAINING_SEGMENTS = {'ac': 75, 'nv': 79}


def mode_segments(frame, mode, validation_start='2024-10-01', test_start='2024-10-10',
                  study_end='2024-10-31', retained_training=True):
    if not pd.Timestamp(validation_start) < pd.Timestamp(test_start) < pd.Timestamp(study_end):
        raise ValueError('Require validation_start < test_start < study_end.')
    frame = frame[frame.date < pd.Timestamp(study_end)].copy()
    frame['partition'] = np.where(frame.date < pd.Timestamp(validation_start), 'train',
                                  np.where(frame.date < pd.Timestamp(test_start), 'validation', 'test'))
    chosen = frame[frame[MODE].eq(0 if mode == 'ac' else 1)]
    boundary = (chosen.date.diff() != pd.Timedelta(minutes=1)) | (
        chosen.date.dt.normalize() != chosen.date.shift().dt.normalize())
    cols = STATES + INPUTS[mode]
    complete = [g.reset_index(drop=True) for _, g in chosen.groupby(boundary.cumsum())
                if len(g) >= 10 and np.isfinite(g[cols].to_numpy(dtype=float)).all()]
    result = {p: [] for p in ['train', 'validation', 'test']}
    retained = complete[:RETAINED_TRAINING_SEGMENTS[mode]] if retained_training else None
    if retained is not None:
        if len(retained) != RETAINED_TRAINING_SEGMENTS[mode] or any(g.partition.ne('train').any() for g in retained):
            raise ValueError('Data do not cover the retained study training segments. Use chronological mode with explicit split dates for another dataset.')
        result['train'] = retained
    for g in complete:
        for part in result:
            if part == 'train' and retained is not None:
                continue
            selected = g[g.partition.eq(part)].reset_index(drop=True)
            if len(selected) >= 10:
                result[part].append(selected)
    return result


def fit_scalers(train, mode):
    if not train:
        raise ValueError('No eligible training segments.')
    data = pd.concat(train, ignore_index=True)
    if not data.partition.eq('train').all():
        raise ValueError('Scalers may only be fitted on the training partition.')
    return (MinMaxScaler(clip=False).fit(data[STATES]),
            MinMaxScaler(clip=False).fit(data[INPUTS[mode]]))


def pairs(segment, mode, sx, su):
    x = sx.transform(segment[STATES]).astype(np.float32)
    u = su.transform(segment[INPUTS[mode]]).astype(np.float32)
    return (torch.from_numpy(np.concatenate([x[:-1], u[:-1]], axis=1)[:, None, :]),
            torch.from_numpy(x[1:]))


def physical_prediction(model, sx, su, x, u):
    x, u = np.atleast_2d(x), np.atleast_2d(u)
    z = np.concatenate([x * sx.scale_ + sx.min_, u * su.scale_ + su.min_], axis=1)
    with torch.inference_mode():
        y = model(torch.tensor(z[:, None, :], dtype=torch.float32)).numpy()
    return (y - sx.min_) / sx.scale_


def horizon_predictions(segs, mode, bundle, horizon=60):
    starts, truths, inputs = [], [], []
    for seg in segs:
        x, u = seg[STATES].to_numpy(), seg[INPUTS[mode]].to_numpy()
        for origin in range(0, len(seg) - horizon, 5):
            starts.append(x[origin]); truths.append(x[origin + horizon]); inputs.append(u[origin:origin + horizon])
    if not starts:
        raise ValueError(f'Validation requires at least one segment longer than {horizon} minutes.')
    state, inputs = np.array(starts), np.array(inputs)
    for step in range(horizon):
        state = physical_prediction(*bundle, state, inputs[:, step, :])
    return np.array(truths), state


def full_segment_predictions(segs, mode, bundle):
    states = np.array([s[STATES].iloc[0].to_numpy(dtype=float) for s in segs])
    x, u = [s[STATES].to_numpy() for s in segs], [s[INPUTS[mode]].to_numpy() for s in segs]
    truth, pred = [], []
    for step in range(max(len(s) for s in segs) - 1):
        active = [i for i, s in enumerate(segs) if step + 1 < len(s)]
        states[active] = physical_prediction(*bundle, states[active], np.array([u[i][step] for i in active]))
        truth.extend([x[i][step + 1] for i in active]); pred.extend(states[active].copy())
    return np.array(truth), np.array(pred)


def train_mode(data_path, mode, output_dir, *, seeds=(17, 29, 43), epochs=200,
               validation_start='2024-10-01', test_start='2024-10-10',
               study_end='2024-10-31', retained_training=True):
    if mode not in INPUTS or epochs < 1 or not seeds:
        raise ValueError('Choose ac/nv, positive epochs and at least one seed.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_formatted_data(data_path, STATES + INPUTS[mode] + [MODE])
    partitions = mode_segments(data, mode, validation_start, test_start, study_end, retained_training)
    train, validation = partitions['train'], partitions['validation']
    if not validation:
        raise ValueError('No eligible recursive-validation segments.')
    sx, su = fit_scalers(train, mode)
    batches = [pairs(s, mode, sx, su) for s in train]
    from .output import require_ignored_output
    output = require_ignored_output(output_dir); output.mkdir(parents=True, exist_ok=True)
    candidates = output / 'candidates' / mode; candidates.mkdir(parents=True, exist_ok=True)
    records, best = [], None
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = CNNLSTM(5 + len(INPUTS[mode]), 64, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.SmoothL1Loss()
        for epoch in range(1, epochs + 1):
            model.train()
            for a, b in batches:
                optimizer.zero_grad(); loss = criterion(model(a), b); loss.backward(); optimizer.step()
            if epoch not in VALIDATION_EPOCHS and epoch != epochs:
                continue
            model.eval()
            truth60, pred60 = horizon_predictions(validation, mode, (model, sx, su))
            truthfull, predfull = full_segment_predictions(validation, mode, (model, sx, su))
            rmse60 = float(np.sqrt(np.mean((pred60 - truth60)**2)))
            rmsefull = float(np.sqrt(np.mean((predfull - truthfull)**2)))
            score = float(np.sqrt((rmse60**2 + rmsefull**2) / 2))
            if not np.isfinite(score):
                raise RuntimeError('Recursive-validation score is nonfinite.')
            row = dict(mode=mode, seed=int(seed), epoch=epoch, validation_score_c=score,
                       validation_60min_rmse_c=rmse60, validation_full_segment_rmse_c=rmsefull)
            path = candidates / f'seed_{seed}_epoch_{epoch}.pth'
            payload = dict(model_state_dict=model.state_dict(), input_dim=5 + len(INPUTS[mode]),
                           hidden_dim=64, output_dim=5, scaler_X=sx, scaler_U=su,
                           state_columns=STATES, input_columns=INPUTS[mode], **row)
            torch.save(payload, path)
            records.append(row)
            if best is None or (score, epoch, seed) < best[0]:
                best = ((score, epoch, seed), row)
                torch.save(payload, output / f'{mode}_model.pth')
    pd.DataFrame(records).to_csv(output / f'{mode}_training_history.csv', index=False)
    # Test values never enter optimizer, scaling or selection.
    return best[1]
