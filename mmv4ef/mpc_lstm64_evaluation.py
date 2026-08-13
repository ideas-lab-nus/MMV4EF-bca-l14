"""Long-form predictions, persistence comparison, and metrics for LSTM64."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .mpc_lstm64_dataset import PHYSICAL_COLUMNS, TARGET_ORDER, LSTM64Examples


TARGET_PHYSICAL_INDICES = {name: index for index, name in enumerate(TARGET_ORDER)}
TARGET_UNITS = {
    "temperature": "deg C",
    "relative_humidity": "%",
    "wind_speed": "m/s",
    "wind_direction": "deg",
    "solar": "W/m^2",
}
MAPE_THRESHOLDS = {
    "temperature": 0.0,
    "relative_humidity": 0.0,
    "wind_speed": 0.5,
    "solar": 20.0,
}
PREDICTION_COLUMNS = (
    "split",
    "date",
    "decision_time_sgt",
    "valid_bin_start_sgt",
    "lead_minutes",
    "target",
    "model",
    "seed",
    "observed_value",
    "predicted_value",
    "unit",
    "observed_wind_speed_ms",
)


def _prediction_rows(
    examples: LSTM64Examples,
    path: np.ndarray,
    model: str,
    seed: str,
) -> list[dict[str, object]]:
    predicted = np.asarray(path, dtype=float)
    if predicted.shape != examples.target_physical.shape:
        raise ValueError("LSTM64 path and target shapes do not match")
    if not np.isfinite(predicted).all():
        raise ValueError("LSTM64 prediction path must be finite")
    rows: list[dict[str, object]] = []
    for example_index, metadata in examples.metadata.iterrows():
        decision = pd.Timestamp(metadata["decision_time_sgt"])
        for lead_index in range(12):
            for target_index, target in enumerate(TARGET_ORDER):
                rows.append({
                    "split": metadata["split"],
                    "date": metadata["date"],
                    "decision_time_sgt": decision,
                    "valid_bin_start_sgt": decision + pd.Timedelta(minutes=5 * lead_index),
                    "lead_minutes": 5 * (lead_index + 1),
                    "target": target,
                    "model": model,
                    "seed": seed,
                    "observed_value": float(
                        examples.target_physical[example_index, lead_index, target_index]
                    ),
                    "predicted_value": float(predicted[example_index, lead_index, target_index]),
                    "unit": TARGET_UNITS[target],
                    "observed_wind_speed_ms": float(
                        examples.target_physical[example_index, lead_index, 2]
                    ),
                })
    return rows


def trajectories_to_long_predictions(
    examples: LSTM64Examples,
    paths_by_seed: dict[int, np.ndarray],
    ensemble_path: np.ndarray,
) -> pd.DataFrame:
    if set(paths_by_seed) != {17, 29, 43}:
        raise ValueError("LSTM64 predictions require seeds 17, 29, and 43")
    rows: list[dict[str, object]] = []
    for seed in (17, 29, 43):
        rows.extend(_prediction_rows(examples, paths_by_seed[seed], "univariate_lstm64", str(seed)))
    rows.extend(_prediction_rows(examples, ensemble_path, "univariate_lstm64", "ensemble"))
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def persistence_to_long_predictions(examples: LSTM64Examples) -> pd.DataFrame:
    last = np.column_stack([
        examples.history[:, -1, 0],
        examples.history[:, -1, 1],
        examples.history[:, -1, 2],
        np.mod(
            np.rad2deg(np.arctan2(examples.history[:, -1, 3], examples.history[:, -1, 4])),
            360.0,
        ),
        examples.history[:, -1, 5],
    ])
    path = np.repeat(last[:, None, :], 12, axis=1)
    return pd.DataFrame(
        _prediction_rows(examples, path, "persistence", "baseline"),
        columns=PREDICTION_COLUMNS,
    )


def _metric_row(keys, target, metric, value, eligible, total):
    return {
        **keys,
        "target": target,
        "metric": metric,
        "value": float(value),
        "eligible_count": int(eligible),
        "total_count": int(total),
        "coverage": eligible / total if total else 0.0,
    }


def _continuous_rows(group: pd.DataFrame, keys: dict[str, object], target: str):
    observed = pd.to_numeric(group["observed_value"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(group["predicted_value"], errors="coerce").to_numpy()
    finite = np.isfinite(observed) & np.isfinite(predicted)
    count = int(finite.sum())
    total = len(group)
    error = predicted[finite] - observed[finite]
    rows = [
        _metric_row(keys, target, "mae", np.mean(np.abs(error)) if count else np.nan, count, total),
        _metric_row(keys, target, "rmse", np.sqrt(np.mean(error**2)) if count else np.nan, count, total),
        _metric_row(keys, target, "bias", np.mean(error) if count else np.nan, count, total),
    ]
    eligible_mask = finite & (observed > MAPE_THRESHOLDS[target])
    eligible = int(eligible_mask.sum())
    mape = (
        100.0 * np.mean(
            np.abs(predicted[eligible_mask] - observed[eligible_mask])
            / np.abs(observed[eligible_mask])
        )
        if eligible else np.nan
    )
    rows.append(_metric_row(keys, target, "mape_pct", mape, eligible, total))
    return rows


def _direction_rows(group: pd.DataFrame, keys: dict[str, object]):
    observed = pd.to_numeric(group["observed_value"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(group["predicted_value"], errors="coerce").to_numpy()
    speed = pd.to_numeric(group["observed_wind_speed_ms"], errors="coerce").to_numpy()
    eligible_mask = (
        np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(speed) & (speed >= 0.5)
    )
    count = int(eligible_mask.sum())
    error = np.abs(
        ((predicted[eligible_mask] - observed[eligible_mask] + 180.0) % 360.0) - 180.0
    )
    return [_metric_row(
        keys, "wind_direction", "circular_mae",
        np.mean(error) if count else np.nan, count, len(group)
    )]


def compute_lstm64_metrics(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...] | Iterable[str],
) -> pd.DataFrame:
    groups = tuple(group_columns)
    required = {*groups, "target", "observed_value", "predicted_value", "observed_wind_speed_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LSTM64 metric frame is missing columns: {sorted(missing)}")
    if "target" in groups:
        raise ValueError("target is added automatically")
    rows = []
    for key, group in frame.groupby([*groups, "target"], dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        target = str(values[-1])
        keys = dict(zip(groups, values[:-1]))
        if target == "wind_direction":
            rows.extend(_direction_rows(group, keys))
        elif target in MAPE_THRESHOLDS:
            rows.extend(_continuous_rows(group, keys, target))
        else:
            raise ValueError(f"Unknown LSTM64 metric target: {target}")
    return pd.DataFrame(rows, columns=[
        *groups, "target", "metric", "value", "eligible_count", "total_count", "coverage"
    ])


def reconcile_lstm64_metrics(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"LSTM64 predictions are missing columns: {sorted(missing)}")
    if predictions.duplicated([
        "split", "date", "decision_time_sgt", "valid_bin_start_sgt",
        "lead_minutes", "target", "model", "seed",
    ]).any():
        raise AssertionError("LSTM64 predictions contain duplicate keys")
    if set(pd.to_numeric(predictions["lead_minutes"], errors="raise")) != set(range(5, 61, 5)):
        raise AssertionError("LSTM64 predictions require exact 5--60 minute leads")
    for column in ("observed_value", "predicted_value"):
        if not np.isfinite(pd.to_numeric(predictions[column], errors="coerce")).all():
            raise AssertionError(f"LSTM64 {column} must be finite")
    standard = {"target", "metric", "value", "eligible_count", "total_count", "coverage"}
    groups = tuple(column for column in metrics.columns if column not in standard)
    actual = compute_lstm64_metrics(predictions, groups)
    keys = [*groups, "target", "metric"]
    expected = metrics.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if expected[keys].astype(str).to_dict("records") != actual[keys].astype(str).to_dict("records"):
        raise AssertionError("LSTM64 metric keys do not reconcile")
    for column in ("value", "eligible_count", "total_count", "coverage"):
        np.testing.assert_allclose(
            pd.to_numeric(expected[column], errors="coerce"),
            pd.to_numeric(actual[column], errors="coerce"),
            rtol=1e-10, atol=1e-12, equal_nan=True,
        )
