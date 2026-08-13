"""Small executable checks for the released inference boundary."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mmv4ef.mpc_future_data import (
    BASE_WEATHER_COLUMNS,
    FutureForecast,
    apply_future_data_source,
)


class FutureDataTests(unittest.TestCase):
    def test_observed_provider_returns_an_isolated_typed_result(self) -> None:
        index = pd.date_range("2024-10-01 08:30", periods=12, freq="5min")
        frame = pd.DataFrame(
            {
                "OutdoorTemperatureWindow": np.linspace(28.0, 29.0, len(index)),
                "OutdoorHumidityWindow": 70.0,
                "Wind Speed": 100.0,
                "Wind Direction": 900.0,
                "Solar Radiation": 500.0,
                "rain_status": 0.0,
            },
            index=index,
        )

        result = apply_future_data_source(frame, "observed")

        self.assertIsInstance(result, FutureForecast)
        self.assertTrue(result.frame.equals(frame))
        self.assertIsNot(result.frame, frame)
        self.assertEqual(tuple(result.frame.columns), BASE_WEATHER_COLUMNS)
        self.assertFalse(result.boundary_completed.any())
        self.assertTrue(result.provenance.index.equals(index))

        result.frame.iloc[0, 0] = -999.0
        self.assertNotEqual(result.frame.iloc[0, 0], frame.iloc[0, 0])


if __name__ == "__main__":
    unittest.main()

