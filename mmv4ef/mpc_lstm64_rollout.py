"""Leakage-safe recursive physical rollouts for five univariate LSTM64 models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .models import WeatherLSTM
from .mpc_lstm64_dataset import (
    CONTINUOUS_TARGETS,
    HISTORY_INDICES,
    TARGET_ORDER,
    LSTM64Scalers,
    normalize_lstm64_history,
    physical_state_to_lstm64_history_row,
)


StepPredictor = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
HISTORY_BASELINE_INDICES = {
    "temperature": 0,
    "relative_humidity": 1,
    "wind_speed": 2,
    "solar": 5,
}


def constrain_lstm64_state(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 5:
        raise ValueError("LSTM64 state must end with five physical variables")
    if not np.isfinite(values).all():
        raise RuntimeError("LSTM64 predictor returned non-finite state")
    constrained = values.copy()
    if np.any((constrained[..., 0] < 10.0) | (constrained[..., 0] > 45.0)):
        raise RuntimeError("LSTM64 temperature is outside [10, 45] degC")
    constrained[..., 1] = np.clip(constrained[..., 1], 0.0, 100.0)
    constrained[..., 2] = np.maximum(constrained[..., 2], 0.0)
    constrained[..., 3] = np.mod(constrained[..., 3], 360.0)
    constrained[..., 4] = np.maximum(constrained[..., 4], 0.0)
    return constrained


def recursive_lstm64_rollout(
    initial_history: np.ndarray,
    future_clock: np.ndarray,
    predict_step: StepPredictor,
) -> np.ndarray:
    history = np.asarray(initial_history, dtype=float).copy()
    clock = np.asarray(future_clock, dtype=float).copy()
    if history.ndim != 3 or history.shape[1:] != (12, 6):
        raise ValueError("Initial LSTM64 history must have shape [N, 12, 6]")
    if clock.shape != (history.shape[0], 12, 2):
        raise ValueError("LSTM64 future clock must have shape [N, 12, 2]")
    if not np.isfinite(history).all() or not np.isfinite(clock).all():
        raise ValueError("LSTM64 rollout inputs must be finite")
    states: list[np.ndarray] = []
    for step in range(12):
        predicted = predict_step(
            history.copy(), clock[:, step : step + 1].copy(), step
        )
        state = constrain_lstm64_state(predicted)
        if state.shape != (history.shape[0], 5):
            raise ValueError("LSTM64 predictor must return shape [N, 5]")
        states.append(state)
        next_row = physical_state_to_lstm64_history_row(state)
        history = np.concatenate([history[:, 1:], next_row[:, None]], axis=1)
    result = np.stack(states, axis=1)
    if result.shape != (len(history), 12, 5):
        raise AssertionError("LSTM64 rollout produced the wrong shape")
    return result


def ensemble_lstm64_trajectories(
    paths_by_seed: dict[int, np.ndarray],
) -> np.ndarray:
    if set(paths_by_seed) != {17, 29, 43}:
        raise ValueError("LSTM64 ensemble requires seeds 17, 29, and 43")
    stacked = np.stack([
        np.asarray(paths_by_seed[seed], dtype=float) for seed in (17, 29, 43)
    ])
    if stacked.ndim != 4 or stacked.shape[-2:] != (12, 5):
        raise ValueError("LSTM64 trajectories must have shape [N, 12, 5]")
    if not np.isfinite(stacked).all():
        raise ValueError("LSTM64 seed trajectories must be finite")
    ensemble = stacked.mean(axis=0)
    angles = np.deg2rad(stacked[..., 3])
    sin_mean = np.sin(angles).mean(axis=0)
    cos_mean = np.cos(angles).mean(axis=0)
    norm = np.hypot(sin_mean, cos_mean)
    direction = np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0)
    direction = np.where(norm >= 1e-8, direction, stacked[0, ..., 3])
    ensemble[..., 3] = direction
    return constrain_lstm64_state(ensemble)


def _model_output(
    model: WeatherLSTM,
    history: np.ndarray,
    clock: np.ndarray,
    device: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.as_tensor(history, dtype=torch.float32, device=device),
            torch.as_tensor(clock, dtype=torch.float32, device=device),
        )
    values = output.detach().cpu().numpy()
    if values.ndim != 3 or values.shape[1] != 1 or not np.isfinite(values).all():
        raise RuntimeError("One-step LSTM64 output must be finite [N, 1, channels]")
    return values[:, 0]


def _decode_direction(output: np.ndarray, raw_history: np.ndarray) -> np.ndarray:
    if output.shape != (len(raw_history), 2):
        raise ValueError("Direction LSTM64 output must have two channels")
    norm = np.linalg.norm(output, axis=1)
    direction = np.mod(np.rad2deg(np.arctan2(output[:, 0], output[:, 1])), 360.0)
    fallback = np.mod(
        np.rad2deg(
            np.arctan2(raw_history[:, -1, 3], raw_history[:, -1, 4])
        ),
        360.0,
    )
    return np.where(norm >= 1e-8, direction, fallback)


def _decode_continuous(
    output: np.ndarray,
    target_name: str,
    raw_history: np.ndarray,
    scalers: LSTM64Scalers,
) -> np.ndarray:
    if output.shape != (len(raw_history), 1):
        raise ValueError("Continuous LSTM64 output must have one channel")
    slot = CONTINUOUS_TARGETS.index(target_name)
    delta = output[:, 0] * scalers.delta_scale[slot] + scalers.delta_mean[slot]
    return raw_history[:, -1, HISTORY_BASELINE_INDICES[target_name]] + delta


def rollout_lstm64_seed(
    examples,
    scalers: LSTM64Scalers,
    models_by_target: dict[str, WeatherLSTM],
    device: str = "cpu",
) -> np.ndarray:
    if set(models_by_target) != set(TARGET_ORDER):
        raise ValueError("LSTM64 rollout requires all five target models")

    def predict_step(
        raw_history: np.ndarray, clock: np.ndarray, step: int
    ) -> np.ndarray:
        del step
        normalized = normalize_lstm64_history(raw_history, scalers)
        predicted: dict[str, np.ndarray] = {}
        for target_name in TARGET_ORDER:
            output = _model_output(
                models_by_target[target_name],
                normalized[..., list(HISTORY_INDICES[target_name])],
                clock,
                device,
            )
            if target_name in CONTINUOUS_TARGETS:
                predicted[target_name] = _decode_continuous(
                    output, target_name, raw_history, scalers
                )
            else:
                predicted[target_name] = _decode_direction(output, raw_history)
        return np.column_stack([predicted[name] for name in TARGET_ORDER])

    return recursive_lstm64_rollout(
        examples.history, examples.future_clock, predict_step
    )
