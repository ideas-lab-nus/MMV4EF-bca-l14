"""Load and transform BCA-campus local weather observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOCAL_HISTORY_COLUMNS


SOURCE_COLUMNS = {
    "OutdoorTemperatureWindow": "temperature_c",
    "OutdoorHumidityWindow": "rh_pct",
    "Solar Radiation": "solar_wm2",
    "rain_status": "rain_observed",
}
REQUIRED_SOURCE_COLUMNS = ("date", *SOURCE_COLUMNS, "Wind Speed", "Wind Direction")


def wind_speed_direction_to_uv(
    speed_ms: np.ndarray | pd.Series,
    direction_deg: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.asarray(speed_ms, dtype=float)
    radians = np.deg2rad(np.asarray(direction_deg, dtype=float))
    return -speed * np.sin(radians), -speed * np.cos(radians)


def wind_uv_to_speed_direction(
    wind_u_ms: np.ndarray | pd.Series,
    wind_v_ms: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(wind_u_ms, dtype=float)
    v = np.asarray(wind_v_ms, dtype=float)
    speed = np.hypot(u, v)
    direction = (np.rad2deg(np.arctan2(-u, -v)) + 360.0) % 360.0
    direction = np.where(speed <= 1e-12, 0.0, direction)
    return speed, direction


def _invalidate_out_of_range(frame: pd.DataFrame, column: str, valid: pd.Series) -> int:
    invalid = frame[column].notna() & ~valid
    count = int(invalid.sum())
    if count:
        frame.loc[invalid, column] = np.nan
    return count


def load_local_weather(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, usecols=list(REQUIRED_SOURCE_COLUMNS))
    timestamps = pd.to_datetime(frame.pop("date"), format="%m/%d/%Y %H:%M", errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Local weather contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Local weather timestamps are not monotonic")
    frame.index = timestamps.dt.tz_localize("Asia/Singapore")
    frame = frame.rename(columns=SOURCE_COLUMNS)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    wind_speed = frame.pop("Wind Speed") / 100.0
    wind_direction = frame.pop("Wind Direction") / 10.0
    quality_counts = {
        "temperature_out_of_range": _invalidate_out_of_range(
            frame, "temperature_c", frame["temperature_c"].between(10.0, 45.0)
        ),
        "rh_out_of_range": _invalidate_out_of_range(
            frame, "rh_pct", frame["rh_pct"].between(0.0, 100.0)
        ),
        "solar_out_of_range": _invalidate_out_of_range(
            frame, "solar_wm2", frame["solar_wm2"].ge(0.0)
        ),
    }
    bad_speed = wind_speed.notna() & wind_speed.lt(0.0)
    bad_direction = wind_direction.notna() & ~wind_direction.between(0.0, 360.0)
    quality_counts["wind_speed_out_of_range"] = int(bad_speed.sum())
    quality_counts["wind_direction_out_of_range"] = int(bad_direction.sum())
    wind_speed = wind_speed.mask(bad_speed)
    wind_direction = wind_direction.mask(bad_direction)
    frame["wind_u_ms"], frame["wind_v_ms"] = wind_speed_direction_to_uv(
        wind_speed, wind_direction
    )
    frame["rain_observed"] = frame["rain_observed"].where(
        frame["rain_observed"].isin([0, 1])
    )
    result = frame.loc[:, list(LOCAL_HISTORY_COLUMNS)].astype(float)
    result.attrs["quality_counts"] = quality_counts
    result.attrs["unit_transformations"] = {
        "Wind Speed": "divide by 100 to obtain m/s",
        "Wind Direction": "divide by 10 to obtain degrees",
    }
    return result


def aggregate_local_five_minute(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(LOCAL_HISTORY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical local columns: {sorted(missing)}")
    continuous = list(LOCAL_HISTORY_COLUMNS[:-1])
    means = frame[continuous].resample("5min", label="left", closed="left").mean()
    rain = frame["rain_observed"].resample("5min", label="left", closed="left").max()
    return means.assign(rain_observed=rain).loc[:, list(LOCAL_HISTORY_COLUMNS)]
