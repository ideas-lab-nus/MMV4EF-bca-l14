#!/usr/bin/env python3
"""Run inference with the released LSTM64 ensemble; no training is performed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmv4ef.config import ForecastConfig
from mmv4ef.local import load_local_weather
from mmv4ef.lstm64_checkpoint import APPROVED_SEEDS, load_lstm64_checkpoint
from mmv4ef.mpc_lstm64_dataset import (
    LSTM64Scalers,
    TARGET_ORDER,
    build_lstm64_examples,
    build_lstm64_forecast_inputs,
)
from mmv4ef.mpc_lstm64_evaluation import (
    compute_lstm64_metrics,
    persistence_to_long_predictions,
    trajectories_to_long_predictions,
)
from mmv4ef.mpc_lstm64_mpc import (
    build_mpc_future_disturbances,
    write_mpc_future_disturbances,
)
from mmv4ef.mpc_lstm64_rollout import (
    ensemble_lstm64_trajectories,
    rollout_lstm64_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/private/l14_merged_data_with_rain.csv",
        help="Private one-minute L14 CSV (never committed).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "models/lstm64",
        help="Directory containing scalers.json and checkpoints/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/lstm64",
        help="Ignored directory for locally generated forecasts and metrics.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Build only the MPC future-disturbance artifact.",
    )
    return parser.parse_args()


def load_seed_models(model_root: Path, seed: int, device: str):
    models = {}
    metadata = {}
    for target in TARGET_ORDER:
        path = (
            model_root
            / "checkpoints/univariate"
            / target
            / f"seed_{seed}.pt"
        )
        models[target], metadata[target] = load_lstm64_checkpoint(
            path,
            target,
            expected_seed=seed,
            device=device,
        )
    return models, metadata


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Private input is missing: {args.data}. See docs/DATA_REQUIREMENTS.md."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ForecastConfig()
    local = load_local_weather(args.data)
    scalers = LSTM64Scalers.load(args.model_root / "scalers.json")

    forecast_inputs = build_lstm64_forecast_inputs(local, cfg)
    forecast_paths = {}
    checkpoint_metadata = {}
    for seed in APPROVED_SEEDS:
        models, metadata = load_seed_models(args.model_root, seed, args.device)
        forecast_paths[seed] = rollout_lstm64_seed(
            forecast_inputs, scalers, models, device=args.device
        )
        checkpoint_metadata[str(seed)] = {
            target: {
                "family": values["family"],
                "target_name": values["target_name"],
                "seed": int(values["seed"]),
            }
            for target, values in metadata.items()
        }
    forecast_ensemble = ensemble_lstm64_trajectories(forecast_paths)
    artifact = build_mpc_future_disturbances(
        args.data, forecast_inputs, forecast_ensemble, cfg
    )
    artifact_path = write_mpc_future_disturbances(
        artifact, args.output_dir / "mpc_future_disturbances.csv.gz"
    )

    metrics_path: Path | None = None
    if not args.skip_metrics:
        examples = build_lstm64_examples(local, cfg)
        scored = examples.subset(
            examples.metadata["split"].isin(("validation", "test")).to_numpy()
        )
        scored_paths = {}
        for seed in APPROVED_SEEDS:
            models, _ = load_seed_models(args.model_root, seed, args.device)
            scored_paths[seed] = rollout_lstm64_seed(
                scored, scalers, models, device=args.device
            )
        ensemble = ensemble_lstm64_trajectories(scored_paths)
        predictions = trajectories_to_long_predictions(
            scored, scored_paths, ensemble
        )
        predictions = pd.concat(
            [predictions, persistence_to_long_predictions(scored)],
            ignore_index=True,
        )
        metrics = compute_lstm64_metrics(
            predictions, ("split", "model", "seed")
        )
        metrics_path = args.output_dir / "metrics_aggregate.csv"
        metrics.to_csv(metrics_path, index=False)

    run_summary = {
        "training_performed": False,
        "device": args.device,
        "forecast_rows": len(artifact),
        "forecast_artifact": artifact_path.name,
        "metrics_artifact": metrics_path.name if metrics_path else None,
        "checkpoint_metadata": checkpoint_metadata,
    }
    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

