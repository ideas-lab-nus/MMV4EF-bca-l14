"""Offline input validation and causal preparation of formatted observations."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import VALIDATION_DATES, TEST_DATES

STATES = [f'Zone {i} Temperature' for i in range(1, 6)]
MODE = 'Z1 Windows Open Close Status'
INPUTS = {
    'ac': ['OutdoorTemperatureWindow', 'FCU-01 Supply Air Temp',
           'FCU-02 Supply Air Temp - 1 min', 'FCU-03 Supply Air Temp',
           'FCU-04 Supply Air Temp', 'FCU-05 Supply Air Temp',
           'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp',
           'rain_status', 'Solar Radiation'],
    'nv': ['OutdoorTemperatureWindow', 'Wind Speed', 'Wind Direction',
           'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp',
           'rain_status', 'Solar Radiation'],
}
CONTROL_COLUMNS = list(dict.fromkeys(STATES + INPUTS['ac'] + INPUTS['nv']
                                    + [MODE, 'OutdoorHumidityWindow']))


def causal_solar_fill(frame, limit=5):
    """Use earlier samples only, within a continuous calendar-day block."""
    frame = frame.sort_values('date').copy()
    boundary = (frame.date.diff() != pd.Timedelta(minutes=1)) | (
        frame.date.dt.normalize() != frame.date.shift().dt.normalize())
    frame['Solar Radiation'] = frame.groupby(boundary.cumsum())['Solar Radiation'].ffill(limit=limit)
    return frame


def load_formatted_data(path, required_columns=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Formatted input is missing. Supply --data or MMV4EF_DATA; see docs/DATA_REQUIREMENTS.md.')
    frame = pd.read_csv(path)
    required = list(required_columns or CONTROL_COLUMNS)
    missing = sorted(set(['date', *required]) - set(frame.columns))
    if missing:
        raise ValueError(f'Formatted input is missing columns: {missing}')
    frame['date'] = pd.to_datetime(frame['date'], format='mixed', errors='raise')
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError('Input timestamps must be present and unique.')
    if frame.date.dt.tz is not None:
        raise ValueError('Input timestamps must be timezone-naive Singapore local time.')
    if (frame.date.dt.second.ne(0) | frame.date.dt.microsecond.ne(0)).any():
        raise ValueError('Input timestamps must lie on exact minute boundaries.')
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    for name in [MODE, 'rain_status']:
        if name in required and not frame[name].dropna().isin([0, 1]).all():
            raise ValueError(f'{name} must use binary values 0 and 1.')
    frame = causal_solar_fill(frame)
    minute = frame.date.dt.hour * 60 + frame.date.dt.minute
    return frame[(minute >= 450) & (minute <= 1140) & (frame.date.dt.weekday < 5)].reset_index(drop=True)


def get_split_dates(data_path, policy='intersection'):
    """Published controller-comparison dates, separate from thermal training splits."""
    if policy not in {'intersection', 'union', 'ac', 'nv'}:
        raise ValueError('Unknown date policy.')
    frame = load_formatted_data(data_path)
    present = set(frame.date.dt.date)
    val = set(VALIDATION_DATES) & present
    test = set(TEST_DATES) & present
    return {'val': sorted(d.isoformat() for d in val),
            'test': sorted(d.isoformat() for d in test),
            'train': sorted(d.isoformat() for d in present if d < min(VALIDATION_DATES))}


def prepare_segments(data_path, max_segments=40, split='test',
                     test_policy='intersection', min_len=600):
    aliases = {'validation':'val', 'validate':'val', 'valid':'val',
               'val+test':'test+val', 'test_val':'test+val'}
    split = aliases.get(split, split)
    if split not in {'train', 'val', 'test', 'test+val', 'all'}:
        raise ValueError('split must be train, val, test, test+val or all.')
    frame = load_formatted_data(data_path)
    dates = None
    if split != 'all':
        parts = get_split_dates(data_path, test_policy)
        dates = set(parts['val'] + parts['test'] if split == 'test+val' else parts[split])
    boundary = (frame.date.diff() != pd.Timedelta(minutes=1)) | (
        frame.date.dt.normalize() != frame.date.shift().dt.normalize())
    result = []
    for _, segment in frame.groupby(boundary.cumsum()):
        if len(segment) < min_len or not np.isfinite(segment[CONTROL_COLUMNS].to_numpy(dtype=float)).all():
            continue
        if dates is None or segment.date.iloc[0].date().isoformat() in dates:
            result.append(segment.reset_index(drop=True))
    return result[:max_segments]
