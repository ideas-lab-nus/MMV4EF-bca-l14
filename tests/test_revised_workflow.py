"""Causality, split isolation, rain safety and model release contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from mmv4ef.data import STATES, INPUTS, MODE, causal_solar_fill, load_formatted_data
from mmv4ef.thermal_training import fit_scalers, mode_segments
from mmv4ef.controller import (
    build_simple_pmv_cfg, observed_rain_forces_ac,
    update_observed_rain_lock_steps, neutralize_rain_for_mpc_prediction,
)
from mmv4ef.output import require_ignored_output

ROOT = Path(__file__).resolve().parents[1]


class CausalPreparationTests(unittest.TestCase):
    def test_future_solar_and_other_days_cannot_fill_missing_past(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2024-10-01 07:30', '2024-10-01 07:31',
                                                      '2024-10-01 07:32', '2024-10-02 07:30']),
                              'Solar Radiation': [np.nan, 10.0, np.nan, np.nan]})
        filled = causal_solar_fill(frame)
        self.assertTrue(np.isnan(filled['Solar Radiation'].iloc[0]))
        self.assertEqual(filled['Solar Radiation'].iloc[2], 10.0)
        self.assertTrue(np.isnan(filled['Solar Radiation'].iloc[3]))
        changed = frame.copy(); changed.loc[2, 'Solar Radiation'] = 999.0
        self.assertTrue(np.isnan(causal_solar_fill(changed)['Solar Radiation'].iloc[0]))

    def test_scalers_exclude_validation_and_test_extremes(self):
        rows = []
        for date, temperature in [('2024-09-30', 25.0), ('2024-10-01', 40.0), ('2024-10-10', 80.0)]:
            block = pd.DataFrame({'date': pd.date_range(date + ' 07:30', periods=70, freq='min')})
            for col in STATES + INPUTS['ac']: block[col] = temperature
            block[MODE] = 0
            rows.append(block)
        parts = mode_segments(pd.concat(rows), 'ac', retained_training=False)
        sx, _ = fit_scalers(parts['train'], 'ac')
        np.testing.assert_equal(sx.data_max_, np.full(5, 25.0))
        self.assertGreater(sx.transform(parts['test'][0][STATES]).min(), 1.0)
        with self.assertRaisesRegex(ValueError, 'training partition'):
            fit_scalers(parts['validation'], 'ac')

    def test_a_mode_segment_is_cut_at_partition_boundary(self):
        frame = pd.DataFrame({'date': pd.date_range('2024-09-30 23:45', periods=30, freq='min')})
        for col in STATES + INPUTS['ac']: frame[col] = 25.0
        frame[MODE] = 0
        parts = mode_segments(frame, 'ac', retained_training=False)
        self.assertTrue(all(g.date.lt(pd.Timestamp('2024-10-01')).all() for g in parts['train']))
        self.assertTrue(all(g.date.ge(pd.Timestamp('2024-10-01')).all() for g in parts['validation']))

    def test_missing_schema_is_reported_before_model_or_solver_use(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.csv'
            pd.DataFrame({'date': ['2024-10-01 07:30']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'missing columns'):
                load_formatted_data(path)


class ControllerContractsTests(unittest.TestCase):
    def test_observed_rain_holds_ac_and_future_rain_is_neutralized(self):
        cfg = build_simple_pmv_cfg()
        wet, dry = pd.Series({'rain_status': 1}), pd.Series({'rain_status': 0})
        self.assertTrue(observed_rain_forces_ac(cfg, wet, 0))
        remaining = update_observed_rain_lock_steps(cfg, wet, 0)
        self.assertEqual(remaining, 6)
        for _ in range(6):
            self.assertTrue(observed_rain_forces_ac(cfg, dry, remaining))
            remaining = update_observed_rain_lock_steps(cfg, dry, remaining)
        self.assertFalse(observed_rain_forces_ac(cfg, dry, remaining))
        future = pd.DataFrame({'rain_status': [0, 1], 'OutdoorTemperatureWindow': [28, 29]})
        result = neutralize_rain_for_mpc_prediction(cfg, future)
        self.assertTrue(result.rain_status.eq(0).all())
        self.assertEqual(future.rain_status.iloc[1], 1)

    def test_private_outputs_cannot_enter_source_or_models(self):
        for path in [ROOT / 'models/new', ROOT / 'scripts/new', ROOT.parent / 'outputs']:
            with self.assertRaises(ValueError): require_ignored_output(path)
        self.assertEqual(require_ignored_output(ROOT / 'outputs/run'), ROOT / 'outputs/run')


class ModelPrivacyTests(unittest.TestCase):
    def test_weather_payloads_contain_weights_and_minimal_metadata_only(self):
        for path in ROOT.glob('models/lstm64/checkpoints/univariate/*/seed_*.pt'):
            payload = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(set(payload), {'model_config', 'state_dict', 'training_metadata'})
            self.assertEqual(set(payload['training_metadata']), {'family', 'target_name', 'one_step', 'seed', 'best_epoch'})
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in payload['state_dict'].values()))

    def test_thermal_checkpoints_exclude_observation_arrays_and_histories(self):
        for path in ROOT.glob('models/thermal/*.pth'):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(set(payload), {'model_state_dict', 'input_dim', 'hidden_dim', 'output_dim', 'scaler_X', 'scaler_U'})
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in payload['model_state_dict'].values()))
            for name in ['scaler_X', 'scaler_U']:
                scaler = payload[name]
                self.assertTrue(all(not isinstance(v, (pd.DataFrame, pd.Series)) for v in vars(scaler).values()))
                self.assertTrue(all(v.ndim <= 1 for v in vars(scaler).values() if isinstance(v, np.ndarray)))


if __name__ == '__main__':
    unittest.main()
