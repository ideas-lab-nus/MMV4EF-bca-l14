"""Exact future-disturbance artifact and MPC overlay for LSTM64 weather."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ForecastConfig


MPC_DISTURBANCE_COLUMNS = (
    "split",
    "date",
    "decision_time_sgt",
    "forecast_time_sgt",
    "horizon_step",
    "lead_minutes",
    "forecast_source_segment",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "solar_wm2",
    "OutdoorTemperatureWindow",
    "OutdoorHumidityWindow",
    "Wind Speed",
    "Wind Direction",
    "Solar Radiation",
    "T_out",
    "I_solar",
    "current_observed_rain_status",
)
RAW_COLUMNS = (
    "date",
    "OutdoorTemperatureWindow",
    "OutdoorHumidityWindow",
    "Wind Speed",
    "Wind Direction",
    "Solar Radiation",
    "rain_status",
)
MPC_WEATHER_COLUMNS = (
    "OutdoorTemperatureWindow",
    "OutdoorHumidityWindow",
    "Wind Speed",
    "Wind Direction",
    "Solar Radiation",
)


def _load_raw_local(source: str | Path | pd.DataFrame, timezone: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = pd.read_csv(Path(source), usecols=list(RAW_COLUMNS))
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw local MPC data is missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(raw.pop("date"), format="%m/%d/%Y %H:%M", errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Raw local MPC timestamps must be unique and monotonic")
    raw.index = timestamps.dt.tz_localize(timezone)
    for column in RAW_COLUMNS[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def _startup_state(raw: pd.DataFrame, stamp: pd.Timestamp) -> np.ndarray:
    index = pd.date_range(stamp, periods=5, freq="1min")
    block = raw.reindex(index)
    if block.loc[:, list(MPC_WEATHER_COLUMNS)].isna().any().any():
        raise ValueError(f"Observed startup horizon is incomplete at {stamp.isoformat()}")
    return np.asarray([
        block["OutdoorTemperatureWindow"].mean(),
        block["OutdoorHumidityWindow"].mean(),
        block["Wind Speed"].mean() / 100.0,
        np.mod(block["Wind Direction"].mean() / 10.0, 360.0),
        block["Solar Radiation"].mean(),
    ], dtype=float)


def _row(
    split: str,
    decision: pd.Timestamp,
    step: int,
    source: str,
    state: np.ndarray,
    rain: float,
) -> dict[str, object]:
    temperature, rh, speed, direction, solar = map(float, state)
    return {
        "split": split,
        "date": decision.date().isoformat(),
        "decision_time_sgt": decision,
        "forecast_time_sgt": decision + pd.Timedelta(minutes=5 * step),
        "horizon_step": step,
        "lead_minutes": 5 * (step + 1),
        "forecast_source_segment": source,
        "temperature_c": temperature,
        "relative_humidity_pct": rh,
        "wind_speed_ms": speed,
        "wind_direction_deg": direction,
        "solar_wm2": solar,
        "OutdoorTemperatureWindow": temperature,
        "OutdoorHumidityWindow": rh,
        "Wind Speed": 100.0 * speed,
        "Wind Direction": 10.0 * direction,
        "Solar Radiation": solar,
        "T_out": temperature,
        "I_solar": solar,
        "current_observed_rain_status": rain,
    }


def build_mpc_future_disturbances(
    raw_local_source: str | Path | pd.DataFrame,
    forecast_inputs,
    ensemble_trajectory: np.ndarray,
    config: ForecastConfig,
) -> pd.DataFrame:
    raw = _load_raw_local(raw_local_source, config.timezone)
    trajectory = np.asarray(ensemble_trajectory, dtype=float)
    if trajectory.shape != (len(forecast_inputs), 12, 5):
        raise ValueError("LSTM64 ensemble trajectory does not align with forecast inputs")
    if not np.isfinite(trajectory).all():
        raise ValueError("LSTM64 ensemble trajectory must be finite")
    metadata = forecast_inputs.metadata.copy()
    metadata["decision_time_sgt"] = pd.to_datetime(
        metadata["decision_time_sgt"], utc=True, errors="raise"
    ).dt.tz_convert(config.timezone)
    if metadata["decision_time_sgt"].duplicated().any():
        raise ValueError("LSTM64 forecast inputs contain duplicate decisions")
    path_by_decision = {
        stamp: trajectory[index]
        for index, stamp in enumerate(metadata["decision_time_sgt"])
    }
    rows: list[dict[str, object]] = []
    selected_days = tuple((*config.validation_dates, *config.test_dates))
    for day in selected_days:
        split = "validation" if day in config.validation_dates else "test"
        first = pd.Timestamp.combine(day, time(7, 30)).tz_localize(config.timezone)
        last = pd.Timestamp.combine(day, time(18, 55)).tz_localize(config.timezone)
        for decision in pd.date_range(first, last, freq="5min"):
            rain = raw.reindex(pd.DatetimeIndex([decision]))["rain_status"].iloc[0]
            if not np.isfinite(rain) or rain not in (0.0, 1.0):
                raise ValueError(f"Decision-time rain is invalid at {decision.isoformat()}")
            if decision.time() < time(8, 30):
                source = "observed_startup"
                states = np.stack([
                    _startup_state(raw, decision + pd.Timedelta(minutes=5 * step))
                    for step in range(12)
                ])
            else:
                source = "lstm64_recursive_ensemble"
                if decision not in path_by_decision:
                    raise ValueError(f"LSTM64 path is missing for {decision.isoformat()}")
                states = path_by_decision[decision]
            rows.extend(
                _row(split, decision, step, source, states[step], float(rain))
                for step in range(12)
            )
    result = pd.DataFrame(rows, columns=MPC_DISTURBANCE_COLUMNS)
    validate_mpc_future_disturbances(result)
    return result


def validate_mpc_future_disturbances(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != MPC_DISTURBANCE_COLUMNS:
        raise AssertionError("MPC LSTM64 artifact schema/order does not match")
    if len(frame) != 36_432:
        raise AssertionError(f"MPC LSTM64 artifact must have 36,432 rows, got {len(frame)}")
    if frame.isna().any().any():
        raise AssertionError("MPC LSTM64 artifact contains missing values")
    values = frame.loc[:, [
        "temperature_c", "relative_humidity_pct", "wind_speed_ms",
        "wind_direction_deg", "solar_wm2", *MPC_WEATHER_COLUMNS, "T_out", "I_solar",
    ]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("MPC LSTM64 artifact contains non-finite values")
    decision = pd.to_datetime(frame["decision_time_sgt"], utc=True, errors="raise").dt.tz_convert("Asia/Singapore")
    forecast = pd.to_datetime(frame["forecast_time_sgt"], utc=True, errors="raise").dt.tz_convert("Asia/Singapore")
    keys = pd.DataFrame({"decision": decision, "step": frame["horizon_step"]})
    if keys.duplicated().any():
        raise AssertionError("MPC LSTM64 artifact contains duplicate decision/stage keys")
    if decision.dt.date.nunique() != 22:
        raise AssertionError("MPC LSTM64 artifact must contain 22 dates")
    counts = pd.DataFrame({"date": decision.dt.date, "decision": decision}).groupby("date")["decision"].nunique()
    if not counts.eq(138).all():
        raise AssertionError("Every MPC LSTM64 date must contain 138 decisions")
    expected_steps = list(range(12))
    if not keys.groupby("decision")["step"].apply(list).map(lambda x: x == expected_steps).all():
        raise AssertionError("Every MPC LSTM64 decision must contain ordered stages 0--11")
    step = pd.to_numeric(frame["horizon_step"], errors="raise").astype(int)
    lead = pd.to_numeric(frame["lead_minutes"], errors="raise").astype(int)
    if not (lead == 5 * (step + 1)).all():
        raise AssertionError("MPC LSTM64 lead mapping is invalid")
    if not (forecast == decision + pd.to_timedelta(5 * step, unit="min")).all():
        raise AssertionError("MPC LSTM64 forecast timestamps are invalid")
    before = decision.dt.time < time(8, 30)
    source = frame["forecast_source_segment"].astype(str)
    if not source[before].eq("observed_startup").all() or not source[~before].eq("lstm64_recursive_ensemble").all():
        raise AssertionError("MPC LSTM64 source boundary is invalid")
    if int(before.sum()) != 3_168 or int((~before).sum()) != 33_264:
        raise AssertionError("MPC LSTM64 source segment counts are invalid")
    if not frame["temperature_c"].between(10.0, 45.0).all():
        raise AssertionError("MPC LSTM64 temperature is outside [10,45]")
    if not frame["relative_humidity_pct"].between(0.0, 100.0).all():
        raise AssertionError("MPC LSTM64 RH is outside [0,100]")
    if not frame["wind_speed_ms"].ge(0.0).all() or not frame["solar_wm2"].ge(0.0).all():
        raise AssertionError("MPC LSTM64 speed/solar must be nonnegative")
    if not frame["wind_direction_deg"].ge(0.0).all() or not frame["wind_direction_deg"].lt(360.0).all():
        raise AssertionError("MPC LSTM64 direction must be within [0,360)")
    np.testing.assert_allclose(frame["Wind Speed"], 100.0 * frame["wind_speed_ms"])
    np.testing.assert_allclose(frame["Wind Direction"], 10.0 * frame["wind_direction_deg"])
    np.testing.assert_allclose(frame["OutdoorTemperatureWindow"], frame["temperature_c"])
    np.testing.assert_allclose(frame["OutdoorHumidityWindow"], frame["relative_humidity_pct"])
    np.testing.assert_allclose(frame["Solar Radiation"], frame["solar_wm2"])
    np.testing.assert_allclose(frame["T_out"], frame["OutdoorTemperatureWindow"])
    np.testing.assert_allclose(frame["I_solar"], frame["Solar Radiation"])
    rain = pd.to_numeric(frame["current_observed_rain_status"], errors="coerce")
    if not rain.isin([0.0, 1.0]).all() or not rain.groupby(decision).nunique().eq(1).all():
        raise AssertionError("MPC LSTM64 current-rain audit is invalid")


def write_mpc_future_disturbances(
    frame: pd.DataFrame, destination: str | Path
) -> Path:
    validate_mpc_future_disturbances(frame)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path


def load_mpc_lstm64_future_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in ("decision_time_sgt", "forecast_time_sgt"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise").dt.tz_convert("Asia/Singapore")
    frame = frame.loc[:, list(MPC_DISTURBANCE_COLUMNS)]
    validate_mpc_future_disturbances(frame)
    return frame


def _sgt_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("Asia/Singapore")
    return index.tz_convert("Asia/Singapore")


def apply_mpc_lstm64_future(
    base_forecast: pd.DataFrame,
    decision_time: pd.Timestamp,
    future_data: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(base_forecast.index, pd.DatetimeIndex):
        raise ValueError("Base MPC forecast must use a DatetimeIndex")
    missing = set(MPC_WEATHER_COLUMNS) - set(base_forecast.columns)
    if missing:
        raise ValueError(f"Base MPC forecast is missing weather fields: {sorted(missing)}")
    decision = pd.Timestamp(decision_time)
    decision = decision.tz_localize("Asia/Singapore") if decision.tzinfo is None else decision.tz_convert("Asia/Singapore")
    source_decision = pd.to_datetime(future_data["decision_time_sgt"], utc=True).dt.tz_convert("Asia/Singapore")
    selected = future_data.loc[source_decision.eq(decision)].sort_values("horizon_step")
    if len(selected) != 12 or selected["horizon_step"].tolist() != list(range(12)):
        raise ValueError("LSTM64 future data lacks the exact twelve-stage decision")
    expected = pd.DatetimeIndex(pd.to_datetime(selected["forecast_time_sgt"], utc=True)).tz_convert("Asia/Singapore")
    if not _sgt_index(base_forecast.index).equals(expected):
        raise ValueError("LSTM64 future timestamps do not match the MPC horizon")
    result = base_forecast.copy(deep=True)
    for column in MPC_WEATHER_COLUMNS:
        result[column] = selected[column].to_numpy()
    if "T_out" in result:
        result["T_out"] = result["OutdoorTemperatureWindow"]
    if "I_solar" in result:
        result["I_solar"] = result["Solar Radiation"]
    return result
