"""Strict, inference-only loading for the released LSTM64 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import WeatherLSTM
from .mpc_lstm64_dataset import HISTORY_INDICES, TARGET_ORDER


FAMILY_NAME = "mpc_univariate_lstm64"
APPROVED_SEEDS = (17, 29, 43)
OUTPUT_WIDTHS = {
    "temperature": 1,
    "relative_humidity": 1,
    "wind_speed": 1,
    "wind_direction": 2,
    "solar": 1,
}


def load_lstm64_checkpoint(
    path: str | Path,
    expected_target: str,
    *,
    expected_seed: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[WeatherLSTM, dict[str, Any]]:
    """Load and validate one trusted checkpoint shipped with this repository."""

    if expected_target not in TARGET_ORDER:
        raise ValueError(f"Unknown LSTM64 target: {expected_target}")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # These release checkpoints contain tensors plus primitive dictionaries only.
    # weights_only=True avoids Python object unpickling during inference.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {"model_config", "state_dict", "training_metadata"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"Unexpected checkpoint payload in {checkpoint_path}")

    model_config = dict(payload["model_config"])
    expected_config = {
        "local_features": len(HISTORY_INDICES[expected_target]),
        "future_features": 2,
        "hidden_dim": 64,
        "targets": OUTPUT_WIDTHS[expected_target],
    }
    if model_config != expected_config:
        raise ValueError(
            f"Checkpoint architecture mismatch for {expected_target}: {model_config}"
        )

    metadata = dict(payload["training_metadata"])
    if metadata.get("family") != FAMILY_NAME:
        raise ValueError("Checkpoint family does not match LSTM64")
    if metadata.get("target_name") != expected_target:
        raise ValueError("Checkpoint target does not match requested target")
    if metadata.get("one_step") is not True:
        raise ValueError("Checkpoint is not marked as a one-step model")
    seed = int(metadata.get("seed", -1))
    if seed not in APPROVED_SEEDS:
        raise ValueError(f"Unapproved LSTM64 seed: {seed}")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"Expected seed {expected_seed}, found {seed}")

    model = WeatherLSTM(**model_config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, metadata

