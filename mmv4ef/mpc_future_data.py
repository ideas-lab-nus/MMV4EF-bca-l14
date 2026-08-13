"""Validated observed/LSTM64 future-weather boundary for the MPC study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LSTM64_PHYSICAL_COLUMNS = (
    "temperature_c", "relative_humidity_pct", "wind_speed_ms",
    "wind_direction_deg", "solar_wm2",
)
LSTM64_OVERLAY_COLUMNS = (
    "OutdoorTemperatureWindow", "OutdoorHumidityWindow", "Wind Speed",
    "Wind Direction", "Solar Radiation", "T_out", "I_solar",
)
LSTM64_PROVENANCE_COLUMNS = (
    "forecast_source_segment", "lead_minutes", "current_observed_rain_status",
)
LSTM64_REQUIRED_COLUMNS = (
    "split", "date", "decision_time_sgt", "forecast_time_sgt",
    "horizon_step", "lead_minutes", "forecast_source_segment",
    *LSTM64_PHYSICAL_COLUMNS, *LSTM64_OVERLAY_COLUMNS,
    "current_observed_rain_status",
)
BASE_WEATHER_COLUMNS = (
    "OutdoorTemperatureWindow", "OutdoorHumidityWindow", "Wind Speed",
    "Wind Direction", "Solar Radiation", "rain_status",
)
AUDIT_WEATHER_COLUMNS = (*BASE_WEATHER_COLUMNS, "T_out", "I_solar")
SOURCE_NAMES = {"observed", "lstm64"}
AUDIT_COLUMNS = (
    "case", "controller_objective", "forecast_source", "decision_time",
    "forecast_time", "horizon_step", "stage_offset_min", "lead_minutes",
    "forecast_source_segment", "boundary_completed", *AUDIT_WEATHER_COLUMNS,
    "base_forecast_rain_status", "mpc_forecast_rain_status",
    "current_observed_rain_status",
)

@dataclass(frozen=True)
class FutureForecast:
    """Provider-transformed MPC frame plus row-aligned audit provenance."""

    frame: pd.DataFrame
    boundary_completed: pd.Series
    provenance: pd.DataFrame

def _require_datetime_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are not monotonic increasing")

def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

def _require_finite(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")

def _normalize_sgt_timestamps(values: pd.Series, label: str) -> pd.DatetimeIndex:
    parsed: list[pd.Timestamp] = []
    expected_offset = pd.Timedelta(hours=8)
    for value in values:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None or stamp.utcoffset() != expected_offset:
            raise ValueError(f"{label} must use the Singapore offset +08:00")
        parsed.append(stamp.tz_convert("Asia/Singapore").tz_localize(None))
    return pd.DatetimeIndex(parsed, name=label)

def _require_close(left: pd.Series, right: pd.Series, label: str) -> None:
    if not np.allclose(
        left.to_numpy(dtype=float),
        right.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(f"LSTM64 {label} contract does not match")

def _validate_complete_lstm64_study(frame: pd.DataFrame) -> None:
    if len(frame) != 36_432:
        raise ValueError(
            f"LSTM64 study contract requires 36,432 rows; found {len(frame):,}"
        )
    decisions = frame.index.get_level_values("decision_time_sgt")
    dates = pd.Index(decisions.date).unique()
    if len(dates) != 22:
        raise ValueError(f"LSTM64 study contract requires 22 dates; found {len(dates)}")

    expected_times = pd.date_range("07:30", "18:55", freq="5min").time
    grouped = frame.groupby(decisions.date, sort=False)
    for day, day_frame in grouped:
        day_decisions = day_frame.index.get_level_values("decision_time_sgt").unique()
        if len(day_decisions) != 138 or not np.array_equal(
            day_decisions.time, expected_times
        ):
            raise ValueError(f"LSTM64 decision schedule is incomplete for {day}")
        counts = day_frame.groupby(level="decision_time_sgt").size()
        if not counts.eq(12).all():
            raise ValueError(f"LSTM64 horizon coverage is incomplete for {day}")

    steps = frame.index.get_level_values("horizon_step")
    if set(steps.unique()) != set(range(12)):
        raise ValueError("LSTM64 study contract requires horizon steps 0..11")

    clock_minutes = decisions.hour * 60 + decisions.minute
    expected_segment = np.where(
        clock_minutes <= 8 * 60 + 25,
        "observed_startup",
        "lstm64_recursive_ensemble",
    )
    actual_segment = frame["forecast_source_segment"].to_numpy(dtype=object)
    if not np.array_equal(actual_segment, expected_segment):
        raise ValueError("LSTM64 forecast_source_segment schedule does not match")
    segment_counts = frame["forecast_source_segment"].value_counts().to_dict()
    if segment_counts != {
        "lstm64_recursive_ensemble": 33_264,
        "observed_startup": 3_168,
    }:
        raise ValueError(f"LSTM64 segment counts do not match: {segment_counts}")

def load_lstm64_future(
    path: str | Path,
    *,
    enforce_study_contract: bool = True,
) -> pd.DataFrame:
    """Load and validate the keyed LSTM64 MPC future-weather artifact."""

    source = Path(path)
    frame = pd.read_csv(source)
    _require_columns(frame, LSTM64_REQUIRED_COLUMNS, "LSTM64 input")
    frame = frame.loc[:, list(LSTM64_REQUIRED_COLUMNS)].copy()

    decision_time = _normalize_sgt_timestamps(
        frame["decision_time_sgt"], "decision_time_sgt"
    )
    forecast_time = _normalize_sgt_timestamps(
        frame["forecast_time_sgt"], "forecast_time_sgt"
    )
    frame["decision_time_sgt"] = decision_time
    frame["forecast_time_sgt"] = forecast_time

    for column in ("horizon_step", "lead_minutes"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"LSTM64 {column} must contain integers")
        frame[column] = numeric.astype(int)

    if frame.duplicated(["decision_time_sgt", "horizon_step"]).any():
        raise ValueError("LSTM64 input contains duplicate primary keys")
    if frame["horizon_step"].lt(0).any():
        raise ValueError("LSTM64 horizon_step must be non-negative")

    expected_forecast = frame["decision_time_sgt"] + pd.to_timedelta(
        frame["horizon_step"] * 5, unit="min"
    )
    if not frame["forecast_time_sgt"].equals(expected_forecast):
        raise ValueError("LSTM64 forecast timestamp does not match decision and step")
    expected_lead = 5 * (frame["horizon_step"] + 1)
    if not frame["lead_minutes"].equals(expected_lead):
        raise ValueError("LSTM64 lead_minutes do not match horizon steps")
    expected_dates = frame["decision_time_sgt"].dt.date.astype(str)
    if not frame["date"].astype(str).equals(expected_dates):
        raise ValueError("LSTM64 date does not match decision_time_sgt")

    numeric_columns = (
        *LSTM64_PHYSICAL_COLUMNS,
        *LSTM64_OVERLAY_COLUMNS,
        "current_observed_rain_status",
    )
    _require_finite(frame, numeric_columns, "LSTM64 input")
    _require_close(
        frame["OutdoorTemperatureWindow"], frame["temperature_c"], "temperature"
    )
    _require_close(
        frame["OutdoorHumidityWindow"],
        frame["relative_humidity_pct"],
        "relative humidity",
    )
    _require_close(frame["Wind Speed"], 100.0 * frame["wind_speed_ms"], "wind speed")
    _require_close(
        frame["Wind Direction"],
        10.0 * frame["wind_direction_deg"],
        "wind direction",
    )
    _require_close(frame["Solar Radiation"], frame["solar_wm2"], "solar")
    _require_close(frame["T_out"], frame["temperature_c"], "T_out alias")
    _require_close(frame["I_solar"], frame["solar_wm2"], "I_solar alias")

    frame = frame.set_index(
        ["decision_time_sgt", "horizon_step"], drop=True
    ).sort_index()
    if enforce_study_contract:
        _validate_complete_lstm64_study(frame)
    return frame

def _empty_provenance(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_source_segment": pd.Series(pd.NA, index=index, dtype="string"),
            "lead_minutes": pd.Series(pd.NA, index=index, dtype="Int64"),
            "current_observed_rain_status": pd.Series(
                pd.NA, index=index, dtype="Float64"
            ),
        },
        index=index,
    )

def _normalize_local_decision_time(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Singapore").tz_localize(None)
    return stamp

def _select_lstm64_rows(
    requested: pd.DatetimeIndex,
    decision_time: pd.Timestamp,
    lstm64_future: pd.DataFrame,
) -> pd.DataFrame:
    if len(requested) == 0:
        raise ValueError("MPC forecast horizon is empty")
    if len(requested) > 12:
        raise ValueError("LSTM64 MPC forecast horizon cannot exceed 12 stages")
    decision = _normalize_local_decision_time(decision_time)
    keys = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([decision] * len(requested)),
            np.arange(len(requested), dtype=int),
        ],
        names=["decision_time_sgt", "horizon_step"],
    )
    missing = keys.difference(lstm64_future.index)
    if len(missing):
        raise ValueError(
            f"LSTM64 input is missing exact keys for decision {decision}: "
            f"{missing.tolist()[:3]}"
        )
    rows = lstm64_future.loc[keys].copy()
    artifact_times = pd.DatetimeIndex(rows["forecast_time_sgt"])
    if not artifact_times.equals(requested):
        raise ValueError(
            "LSTM64 forecast timestamps do not match the MPC horizon index"
        )
    rows.index = requested
    return rows

def apply_future_data_source(
    base_forecast: pd.DataFrame,
    source: str,
    *,
    decision_time: pd.Timestamp | None = None,
    lstm64_future: pd.DataFrame | None = None,
) -> FutureForecast:
    """Return an isolated forecast frame for observed or LSTM64 weather."""
    if source not in SOURCE_NAMES:
        raise ValueError(
            f"Unknown future-data source {source!r}; expected observed or lstm64"
        )
    _require_datetime_index(base_forecast, "MPC forecast")
    _require_columns(base_forecast, BASE_WEATHER_COLUMNS, "MPC forecast")
    _require_finite(base_forecast, BASE_WEATHER_COLUMNS, "MPC forecast")

    result = base_forecast.copy(deep=True)
    flags = pd.Series(False, index=result.index, name="boundary_completed", dtype=bool)
    provenance = _empty_provenance(result.index)
    if source == "observed":
        return FutureForecast(result, flags, provenance)

    if decision_time is None:
        raise ValueError("LSTM64 source requires the MPC decision_time")
    if lstm64_future is None:
        raise ValueError("LSTM64 source requires the keyed future table")
    if not isinstance(lstm64_future.index, pd.MultiIndex) or (
        lstm64_future.index.names != ["decision_time_sgt", "horizon_step"]
    ):
        raise ValueError(
            "LSTM64 input must use the decision_time_sgt/horizon_step index"
        )
    _require_columns(
        result, LSTM64_OVERLAY_COLUMNS[:5], "MPC forecast for LSTM64 overlay"
    )
    rows = _select_lstm64_rows(result.index, decision_time, lstm64_future)
    untouched = [
        column for column in result.columns if column not in LSTM64_OVERLAY_COLUMNS
    ]
    untouched_before = result.loc[:, untouched].copy(deep=True)
    for column in LSTM64_OVERLAY_COLUMNS:
        result[column] = rows[column].to_numpy()
    if not result.loc[:, untouched].equals(untouched_before):
        raise RuntimeError("LSTM64 overlay changed non-weather MPC columns")
    provenance = rows.loc[:, list(LSTM64_PROVENANCE_COLUMNS)].copy()
    provenance.index = result.index
    _require_finite(result, BASE_WEATHER_COLUMNS, "Mapped MPC weather")
    return FutureForecast(result, flags, provenance)

def build_forecast_audit(
    forecast: pd.DataFrame,
    boundary_completed: pd.Series,
    *,
    source: str,
    case_name: str,
    decision_time: pd.Timestamp,
    ctrl_period_min: int,
    controller_objective: str = "not_applicable",
    provenance: pd.DataFrame | None = None,
    base_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one immutable, row-per-stage record of weather passed to MPC."""

    if source not in SOURCE_NAMES:
        raise ValueError(f"Unknown future-data source {source!r}")
    _require_datetime_index(forecast, "MPC forecast")
    _require_columns(forecast, BASE_WEATHER_COLUMNS, "MPC forecast")
    if not forecast.index.equals(boundary_completed.index):
        raise ValueError(
            "Forecast and boundary-completion timestamps do not match"
        )
    if int(ctrl_period_min) <= 0:
        raise ValueError("ctrl_period_min must be positive")
    if provenance is None:
        provenance = _empty_provenance(forecast.index)
    if not forecast.index.equals(provenance.index):
        raise ValueError("Forecast and provider-provenance timestamps do not match")
    _require_columns(
        provenance,
        LSTM64_PROVENANCE_COLUMNS,
        "Provider provenance",
    )
    if base_forecast is None:
        base_forecast = forecast
    if not forecast.index.equals(base_forecast.index):
        raise ValueError("Forecast and base-forecast timestamps do not match")
    _require_columns(base_forecast, ("rain_status",), "Base forecast")

    weather = forecast.loc[:, list(BASE_WEATHER_COLUMNS)].copy()
    weather["T_out"] = (
        forecast["T_out"]
        if "T_out" in forecast
        else forecast["OutdoorTemperatureWindow"]
    )
    weather["I_solar"] = (
        forecast["I_solar"]
        if "I_solar" in forecast
        else forecast["Solar Radiation"]
    )
    audit = pd.DataFrame(
        {
            "case": case_name,
            "controller_objective": controller_objective,
            "forecast_source": source,
            "decision_time": pd.Timestamp(decision_time),
            "forecast_time": forecast.index,
            "horizon_step": np.arange(len(forecast), dtype=int),
            "stage_offset_min": (
                np.arange(len(forecast), dtype=int) * int(ctrl_period_min)
            ),
            "lead_minutes": provenance["lead_minutes"].array,
            "forecast_source_segment": provenance[
                "forecast_source_segment"
            ].array,
            "boundary_completed": boundary_completed.to_numpy(dtype=bool),
        }
    )
    result = pd.concat(
        [audit.reset_index(drop=True), weather.reset_index(drop=True)],
        axis=1,
    )
    result["base_forecast_rain_status"] = base_forecast[
        "rain_status"
    ].to_numpy()
    result["mpc_forecast_rain_status"] = forecast["rain_status"].to_numpy()
    result["current_observed_rain_status"] = provenance[
        "current_observed_rain_status"
    ].array
    return result.loc[:, list(AUDIT_COLUMNS)]
