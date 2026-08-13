"""Rebuild the manuscript PV holdout figure from private inputs and the fitted model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/private/pv_appendix_b.csv"),
    )
    parser.add_argument(
        "--best-model",
        type=Path,
        default=Path("models/pv/pv_appendix_b_best_model.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/pv_appendix_b_holdout_plots.png"),
    )
    return parser.parse_args()


def build_figure(dataset_path: Path, best_model_path: Path, output_path: Path) -> None:
    dataset = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    best_model = pd.read_csv(best_model_path).iloc[0]
    required = {"timestamp", "pv_power_kw", "ghi_wm2", "outdoor_air_temp_c"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"missing Appendix B plot columns: {sorted(missing)}")

    plot_data = dataset.dropna(subset=list(required)).sort_values("timestamp")
    ghi = pd.to_numeric(plot_data["ghi_wm2"], errors="raise").to_numpy(float)
    outdoor = pd.to_numeric(
        plot_data["outdoor_air_temp_c"], errors="raise"
    ).to_numpy(float)
    plot_data["appendix_b_prediction_kw"] = np.maximum(
        float(best_model["a1"]) * ghi**2
        + float(best_model["a2"]) * ghi * outdoor
        + float(best_model["a3"]) * ghi,
        0.0,
    )
    test_count = int(best_model["test_count"])
    if test_count <= 0 or test_count > len(plot_data):
        raise ValueError("invalid holdout row count in best-model summary")
    holdout = plot_data.tail(test_count).copy()
    first_week = holdout.iloc[: min(len(holdout), 24 * 60 * 7)].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16.0,
            "axes.labelsize": 16.0,
            "legend.fontsize": 14.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(
        first_week["timestamp"],
        first_week["pv_power_kw"],
        label="Actual PV power",
        linewidth=1.8,
    )
    axes[0].plot(
        first_week["timestamp"],
        first_week["appendix_b_prediction_kw"],
        label="Appendix B prediction",
        linewidth=1.8,
    )
    axes[0].set_ylabel("PV power (kW)")
    axes[1].scatter(
        holdout["pv_power_kw"],
        holdout["appendix_b_prediction_kw"],
        s=12,
        alpha=0.25,
    )
    axes[1].set_xlabel("Actual PV power (kW)")
    axes[1].set_ylabel("Predicted PV power (kW)")
    for panel_label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.text(
            0.01,
            0.97,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
        )
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    build_figure(args.dataset, args.best_model, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
