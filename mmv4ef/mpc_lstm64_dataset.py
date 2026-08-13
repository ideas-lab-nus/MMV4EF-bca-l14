"""RH-aware local-only examples for five univariate one-step LSTM64 models."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ForecastConfig, split_for_date
from .local import wind_uv_to_speed_direction


TARGET_ORDER = (
    "temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "solar",
)
PHYSICAL_COLUMNS = (
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "solar_wm2",
)
REQUIRED_LOCAL_COLUMNS = (
    "temperature_c",
    "rh_pct",
    "wind_u_ms",
    "wind_v_ms",
    "solar_wm2",
)
HISTORY_INDICES = {
    "temperature": (0,),
    "relative_humidity": (1,),
    "wind_speed": (2,),
    "wind_direction": (3, 4),
    "solar": (5,),
}
CONTINUOUS_TARGETS = (
    "temperature",
    "relative_humidity",
    "wind_speed",
    "solar",
)
CONTINUOUS_HISTORY_INDICES = np.asarray([0, 1, 2, 5], dtype=int)
CONTINUOUS_PHYSICAL_INDICES = np.asarray([0, 1, 2, 4], dtype=int)
APPROVED_FULL_COUNTS = {"train": 5398, "validation": 1380, "test": 1150}


def _finite_array(value: np.ndarray, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise ValueError(f"{label} must have shape [N, {', '.join(map(str, shape_tail))}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


@dataclass(frozen=True)
class LSTM64Examples:
    history: np.ndarray
    future_clock: np.ndarray
    target_physical: np.ndarray
    metadata: pd.DataFrame
    exclusion_counts: dict[str, int]

    def __post_init__(self) -> None:
        history = _finite_array(self.history, (12, 6), "LSTM64 history")
        clock = _finite_array(self.future_clock, (12, 2), "LSTM64 future clock")
        target = _finite_array(
            self.target_physical, (12, 5), "LSTM64 physical target"
        )
        if not (len(history) == len(clock) == len(target) == len(self.metadata)):
            raise ValueError("LSTM64 example arrays and metadata must align")

    def __len__(self) -> int:
        return len(self.metadata)

    def subset(self, mask: np.ndarray) -> "LSTM64Examples":
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (len(self),):
            raise ValueError("LSTM64 example mask has the wrong shape")
        return LSTM64Examples(
            history=self.history[selected],
            future_clock=self.future_clock[selected],
            target_physical=self.target_physical[selected],
            metadata=self.metadata.loc[selected].reset_index(drop=True),
            exclusion_counts=dict(self.exclusion_counts),
        )

    def for_split(self, split: str) -> "LSTM64Examples":
        return self.subset(self.metadata["split"].eq(split).to_numpy())


@dataclass(frozen=True)
class LSTM64ForecastInputs:
    history: np.ndarray
    future_clock: np.ndarray
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        history = _finite_array(self.history, (12, 6), "LSTM64 inference history")
        clock = _finite_array(self.future_clock, (12, 2), "LSTM64 inference clock")
        if not (len(history) == len(clock) == len(self.metadata)):
            raise ValueError("LSTM64 inference arrays and metadata must align")

    def __len__(self) -> int:
        return len(self.metadata)


@dataclass(frozen=True)
class LSTM64Scalers:
    history_mean: np.ndarray
    history_scale: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray

    def __post_init__(self) -> None:
        for name in ("history_mean", "history_scale", "delta_mean", "delta_scale"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (4,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite four-vector")
            if "scale" in name and np.any(value <= 0.0):
                raise ValueError(f"{name} must be positive")

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: np.asarray(getattr(self, name), dtype=float).tolist()
            for name in ("history_mean", "history_scale", "delta_mean", "delta_scale")
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "LSTM64Scalers":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{name: np.asarray(payload[name], dtype=float) for name in (
            "history_mean", "history_scale", "delta_mean", "delta_scale"
        )})


def _complete(frame: pd.DataFrame) -> bool:
    return len(frame) > 0 and not frame.loc[:, list(REQUIRED_LOCAL_COLUMNS)].isna().any().any()


def _aggregate_five_minute(frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0 or len(frame) % 5:
        raise ValueError("LSTM64 aggregation requires complete five-minute blocks")
    rows: list[list[float]] = []
    for start in range(0, len(frame), 5):
        block = frame.iloc[start : start + 5]
        speed, direction = wind_uv_to_speed_direction(
            np.asarray([block["wind_u_ms"].mean()]),
            np.asarray([block["wind_v_ms"].mean()]),
        )
        rows.append([
            float(block["temperature_c"].mean()),
            float(block["rh_pct"].mean()),
            float(speed[0]),
            float(direction[0]),
            float(block["solar_wm2"].mean()),
        ])
    result = np.asarray(rows, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("LSTM64 aggregation produced non-finite values")
    return result


def physical_state_to_lstm64_history_row(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("LSTM64 physical state must have finite shape [N, 5]")
    angle = np.deg2rad(np.mod(values[:, 3], 360.0))
    return np.column_stack([
        values[:, 0], values[:, 1], values[:, 2],
        np.sin(angle), np.cos(angle), values[:, 4],
    ])


def _history_channels(physical: np.ndarray) -> np.ndarray:
    return physical_state_to_lstm64_history_row(np.asarray(physical, dtype=float))


def _future_clock(decision: pd.Timestamp, cfg: ForecastConfig) -> np.ndarray:
    starts = pd.date_range(
        decision, periods=cfg.horizon_steps, freq=f"{cfg.step_minutes}min"
    )
    minute = starts.hour.to_numpy() * 60 + starts.minute.to_numpy()
    angle = 2.0 * np.pi * minute / (24.0 * 60.0)
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _validate_local(local: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    missing = set(REQUIRED_LOCAL_COLUMNS) - set(local.columns)
    if missing:
        raise ValueError(f"Local data is missing LSTM64 columns: {sorted(missing)}")
    result = local.loc[:, list(REQUIRED_LOCAL_COLUMNS)].copy()
    if not isinstance(result.index, pd.DatetimeIndex) or result.index.tz is None:
        raise ValueError("Local timestamps must be timezone-aware")
    result.index = result.index.tz_convert(cfg.timezone)
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("Local timestamps must be unique and monotonic")
    return result


def _history_index(decision: pd.Timestamp, cfg: ForecastConfig) -> pd.DatetimeIndex:
    return pd.date_range(
        decision - pd.Timedelta(minutes=cfg.history_minutes),
        periods=cfg.history_minutes,
        freq="1min",
    )


def build_lstm64_examples(
    local_one_minute: pd.DataFrame,
    config: ForecastConfig,
    expected_counts: dict[str, int] | None = None,
) -> LSTM64Examples:
    """Build scored 08:30--18:00 examples with complete history and targets."""

    local = _validate_local(local_one_minute, config)
    histories: list[np.ndarray] = []
    clocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    excluded: Counter[str] = Counter()
    for day in sorted(set(local.index.date)):
        split = split_for_date(day, config)
        if split is None:
            continue
        first = pd.Timestamp.combine(day, config.first_decision).tz_localize(config.timezone)
        last = pd.Timestamp.combine(day, config.last_decision).tz_localize(config.timezone)
        for decision in pd.date_range(first, last, freq=f"{config.step_minutes}min"):
            history = local.reindex(_history_index(decision, config))
            target_index = pd.date_range(
                decision, periods=config.horizon_minutes, freq="1min"
            )
            target = local.reindex(target_index)
            if not _complete(history):
                excluded["missing_or_invalid_history"] += 1
                continue
            if not _complete(target):
                excluded["missing_or_invalid_target"] += 1
                continue
            histories.append(_history_channels(_aggregate_five_minute(history)))
            clocks.append(_future_clock(decision, config))
            targets.append(_aggregate_five_minute(target))
            metadata.append({
                "date": day.isoformat(),
                "split": split,
                "decision_time_sgt": decision,
            })
    if not metadata:
        raise ValueError("No eligible LSTM64 examples were constructed")
    result = LSTM64Examples(
        history=np.stack(histories),
        future_clock=np.stack(clocks),
        target_physical=np.stack(targets),
        metadata=pd.DataFrame(metadata),
        exclusion_counts=dict(excluded),
    )
    if expected_counts is not None:
        actual = {name: len(result.for_split(name)) for name in ("train", "validation", "test")}
        expected = {name: int(expected_counts.get(name, -1)) for name in actual}
        if actual != expected:
            raise AssertionError(
                f"LSTM64 window counts differ: expected={expected}, actual={actual}"
            )
    return result


def build_lstm64_forecast_inputs(
    local_one_minute: pd.DataFrame,
    config: ForecastConfig,
    first_decision: time = time(8, 30),
    last_decision: time = time(18, 55),
) -> LSTM64ForecastInputs:
    """Build selected-date causal histories without indexing future targets."""

    local = _validate_local(local_one_minute, config)
    selected_days = tuple((*config.validation_dates, *config.test_dates))
    histories: list[np.ndarray] = []
    clocks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for day in selected_days:
        split = split_for_date(day, config)
        first = pd.Timestamp.combine(day, first_decision).tz_localize(config.timezone)
        last = pd.Timestamp.combine(day, last_decision).tz_localize(config.timezone)
        for decision in pd.date_range(first, last, freq=f"{config.step_minutes}min"):
            history = local.reindex(_history_index(decision, config))
            if not _complete(history):
                raise ValueError(
                    f"Incomplete observed LSTM64 history for decision {decision.isoformat()}"
                )
            histories.append(_history_channels(_aggregate_five_minute(history)))
            clocks.append(_future_clock(decision, config))
            metadata.append({
                "date": day.isoformat(),
                "split": split,
                "decision_time_sgt": decision,
            })
    result = LSTM64ForecastInputs(
        history=np.stack(histories),
        future_clock=np.stack(clocks),
        metadata=pd.DataFrame(metadata),
    )
    representative = pd.date_range(
        pd.Timestamp.combine(selected_days[0], first_decision),
        pd.Timestamp.combine(selected_days[0], last_decision),
        freq=f"{config.step_minutes}min",
    )
    expected = len(selected_days) * len(representative)
    if len(result) != expected:
        raise AssertionError(f"Expected {expected} LSTM64 decisions, got {len(result)}")
    return result


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("LSTM64 scaler values must be a nonempty finite matrix")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    return mean, np.where(scale < 1e-8, 1.0, scale)


def fit_lstm64_scalers(examples: LSTM64Examples) -> LSTM64Scalers:
    if len(examples) == 0 or not examples.metadata["split"].eq("train").all():
        raise ValueError("LSTM64 scalers require training-only examples")
    history_values = examples.history[..., CONTINUOUS_HISTORY_INDICES].reshape(-1, 4)
    history_mean, history_scale = _mean_scale(history_values)
    last_values = examples.history[:, -1, CONTINUOUS_HISTORY_INDICES]
    next_values = examples.target_physical[:, 0, CONTINUOUS_PHYSICAL_INDICES]
    delta_mean, delta_scale = _mean_scale(next_values - last_values)
    return LSTM64Scalers(history_mean, history_scale, delta_mean, delta_scale)


def normalize_lstm64_history(
    history: np.ndarray, scalers: LSTM64Scalers
) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("LSTM64 history must have finite shape [N, steps, 6]")
    normalized = values.copy()
    normalized[..., CONTINUOUS_HISTORY_INDICES] = (
        normalized[..., CONTINUOUS_HISTORY_INDICES] - scalers.history_mean
    ) / scalers.history_scale
    return normalized


class LSTM64UnivariateDataset(Dataset):
    def __init__(
        self,
        examples: LSTM64Examples,
        scalers: LSTM64Scalers,
        target_name: str,
    ) -> None:
        if target_name not in TARGET_ORDER:
            raise ValueError(f"Unknown LSTM64 target: {target_name}")
        normalized = normalize_lstm64_history(examples.history, scalers)
        self.target_name = target_name
        self.history_indices = HISTORY_INDICES[target_name]
        self.history = torch.as_tensor(
            normalized[..., list(self.history_indices)], dtype=torch.float32
        )
        self.clock = torch.as_tensor(
            examples.future_clock[:, :1], dtype=torch.float32
        )
        if target_name in CONTINUOUS_TARGETS:
            slot = CONTINUOUS_TARGETS.index(target_name)
            history_index = int(CONTINUOUS_HISTORY_INDICES[slot])
            physical_index = int(CONTINUOUS_PHYSICAL_INDICES[slot])
            delta = (
                examples.target_physical[:, 0, physical_index]
                - examples.history[:, -1, history_index]
            )
            target = ((delta - scalers.delta_mean[slot]) / scalers.delta_scale[slot])[:, None, None]
        else:
            angle = np.deg2rad(examples.target_physical[:, 0, 3])
            target = np.stack([np.sin(angle), np.cos(angle)], axis=-1)[:, None, :]
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.history[index], self.clock[index], self.target[index]
