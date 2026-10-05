#!/usr/bin/env python
"""Generate daily rollout trajectory figures.

The script mirrors the clean weekday segment filtering and chronological
date partitions used by the final thermal-validation protocol.

It produces one daily rollout plot per validation/test day. The nonlinear
CNN-LSTM dynamic model and 5-minute frozen linearization use measured
minute-by-minute inputs. A second 5-minute diagnostic freezes both the
linearization and the first measured input over each control period.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpc_miqp_simulation_fast import (
    AC_INPUT_COLS,
    AC_STATE_COLS,
    COLUMNS_TO_CHECK,
    NV_INPUT_COLS,
    forward_surrogate,
    linearize_surrogate,
    load_surrogate,
    make_continuous_segments,
)

from mmv4ef.data import causal_solar_fill

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14.0,
        "axes.titlesize": 16.0,
        "axes.labelsize": 15.0,
        "legend.fontsize": 12.0,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
    }
)

WINDOW_COL = "Z1 Windows Open Close Status"
SET2 = {
    "teal": "#66c2a5",
    "orange": "#fc8d62",
    "lavender": "#8da0cb",
    "pink": "#e78ac3",
    "lime": "#a6d854",
    "yellow": "#ffd92f",
    "tan": "#e5c494",
    "gray": "#b3b3b3",
}
GROUND_TRUTH_COLOR = "#1f1f1f"
TEXT_COLOR = "#2b2b2b"
GRID_COLOR = "#b3b3b3"
MODEL_STYLES = {
    "cnn": {
        "label": "Deep learning model rollout",
        "color": SET2["teal"],
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 0.9,
        "marker": None,
    },
    "linear_5min": {
        "label": "5-min linearization rollout varied",
        "color": SET2["pink"],
        "linestyle": (0, (5, 2.6)),
        "linewidth": 1.5,
        "alpha": 0.42,
        "marker": "^",
        "markersize": 8.0,
        "markeredgewidth": 0.0,
    },
    "linear_5min_hold_u": {
        "label": "Zero-order-hold affine surrogate",
        "color": SET2["orange"],
        "linestyle": "-.",
        "linewidth": 1.5,
        "alpha": 0.58,
        "marker": "^",
        "markersize": 4.8,
        "markeredgewidth": 0.0,
    },
}
FIVE_MIN_ROLLOUT_KEYS = ("linear_5min", "linear_5min_hold_u")
FIVE_MIN_ROLLOUT_ARG_TO_KEY = {
    "linearized": "linear_5min",
    "held-input": "linear_5min_hold_u",
}


@dataclass(frozen=True)
class ControlBlock:
    start: int
    end: int
    mode: str
    u_anchor: np.ndarray

    @property
    def steps(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fair daily rollout trajectory plots."
    )
    parser.add_argument("--data", default="data/private/l14_merged_data_with_rain.csv")
    parser.add_argument("--ac-model", default="models/thermal/ac_model.pth")
    parser.add_argument("--nv-model", default="models/thermal/nv_model.pth")
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/linearization_validity",
        help="Directory where plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "validation", "test"],
    )
    parser.add_argument("--min-len", type=int, default=600)
    parser.add_argument("--ctrl-period-min", type=int, default=5)
    parser.add_argument(
        "--exclude-date",
        action="append",
        default=["2024-10-31"],
        help="Date to exclude. Can be repeated. Defaults to 2024-10-31.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--max-days-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit for each requested split.",
    )
    parser.add_argument(
        "--five-min-rollouts",
        nargs="+",
        default=["both"],
        choices=["linearized", "held-input", "both", "none"],
        help=(
            "Which 5-minute trajectories to plot next to ground truth and the "
            "deep learning rollout. Use 'linearized', 'held-input', 'both', or 'none'."
        ),
    )
    parser.add_argument(
        "--split-on-mode-switch",
        action="store_true",
        help=(
            "Split a 5-minute control period at measured AC/NV mode switches. "
            "By default the block-start mode is held for the full control "
            "period, matching a controller that updates f_m every 5 minutes."
        ),
    )
    parser.add_argument(
        "--hold-mode-through-switch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def mode_from_row(row: pd.Series) -> str:
    return "nv" if int(row[WINDOW_COL]) == 1 else "ac"


def build_measured_input(row: pd.Series, mode: str) -> np.ndarray:
    if mode == "ac":
        return row[AC_INPUT_COLS].astype(float).to_numpy(copy=True)
    return row[NV_INPUT_COLS].astype(float).to_numpy(copy=True)


def measured_input_at(day_df: pd.DataFrame, minute_idx: int, mode: str) -> np.ndarray:
    return build_measured_input(day_df.iloc[minute_idx], mode)


def clean_working_hour_segments(
    data_path: str | Path,
    min_len: int,
    exclude_dates: Iterable[str],
) -> list[pd.DataFrame]:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["date"].dt.time >= pd.to_datetime("07:30").time())
        & (df["date"].dt.time <= pd.to_datetime("19:00").time())
    ]
    df = df[df["date"].dt.weekday < 5]

    excluded = pd.to_datetime(list(exclude_dates)).date
    if len(excluded):
        df = df[~df["date"].dt.date.isin(excluded)]

    df = df.sort_values("date").reset_index(drop=True)
    df = causal_solar_fill(df)

    segments = [
        seg.reset_index(drop=True)
        for seg in make_continuous_segments(df)
        if len(seg) >= min_len
    ]
    return [
        seg.dropna(subset=COLUMNS_TO_CHECK)
        for seg in segments
        if not seg[COLUMNS_TO_CHECK].isna().any().any()
    ]


def split_segments(segments: list[pd.DataFrame]) -> dict[str, list[int]]:
    dates = [seg.date.iloc[0] for seg in segments]
    return {
        'train': [i for i, d in enumerate(dates) if d < pd.Timestamp('2024-10-01')],
        'val': [i for i, d in enumerate(dates) if pd.Timestamp('2024-10-01') <= d < pd.Timestamp('2024-10-10')],
        'test': [i for i, d in enumerate(dates) if pd.Timestamp('2024-10-10') <= d < pd.Timestamp('2024-10-31')],
    }


def select_model(mode: str, models: dict[str, tuple]) -> tuple:
    return models[mode]


def control_blocks_for_period(
    day_df: pd.DataFrame,
    period_start: int,
    period_end: int,
    split_on_mode_switch: bool,
) -> list[ControlBlock]:
    blocks: list[ControlBlock] = []
    status = day_df[WINDOW_COL].astype(int).to_numpy()
    start = period_start
    while start < period_end:
        mode_code = status[start]
        end = period_end
        if split_on_mode_switch:
            switch_locs = np.flatnonzero(status[start + 1 : period_end] != mode_code)
            if len(switch_locs):
                end = start + 1 + int(switch_locs[0])

        mode = "nv" if mode_code == 1 else "ac"
        blocks.append(
            ControlBlock(
                start=start,
                end=end,
                mode=mode,
                u_anchor=measured_input_at(day_df, start, mode),
            )
        )
        start = end
    return blocks


def iter_control_blocks(
    day_df: pd.DataFrame,
    ctrl_period_min: int,
    split_on_mode_switch: bool,
) -> Iterable[ControlBlock]:
    n_transitions = len(day_df) - 1
    for period_start in range(0, n_transitions, ctrl_period_min):
        period_end = min(period_start + ctrl_period_min, n_transitions)
        yield from control_blocks_for_period(
            day_df,
            period_start=period_start,
            period_end=period_end,
            split_on_mode_switch=split_on_mode_switch,
        )


def forward_steps(
    x0: np.ndarray,
    block: ControlBlock,
    day_df: pd.DataFrame,
    models: dict[str, tuple],
    device: torch.device,
    keep_all_minutes: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    x = np.asarray(x0, dtype=float).copy()
    minute_values: list[np.ndarray] = []
    model, scaler_x, scaler_u = select_model(block.mode, models)
    for minute_idx in range(block.start, block.end):
        u = measured_input_at(day_df, minute_idx, block.mode)
        x = forward_surrogate(model, scaler_x, scaler_u, x, u, device)
        if keep_all_minutes:
            minute_values.append(x.copy())
    return x, minute_values


def linear_1min_steps(
    x0: np.ndarray,
    block: ControlBlock,
    day_df: pd.DataFrame,
    models: dict[str, tuple],
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    x = np.asarray(x0, dtype=float).copy()
    minute_values: list[np.ndarray] = []
    model, scaler_x, scaler_u = select_model(block.mode, models)
    for minute_idx in range(block.start, block.end):
        u = measured_input_at(day_df, minute_idx, block.mode)
        lin = linearize_surrogate(model, scaler_x, scaler_u, x, u, device)
        x = lin.A @ x + lin.B @ u + lin.c
        minute_values.append(x.copy())
    return x, minute_values


def linear_5min_frozen_steps(
    x0: np.ndarray,
    block: ControlBlock,
    day_df: pd.DataFrame,
    models: dict[str, tuple],
    device: torch.device,
    keep_all_minutes: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Freeze A, B, c at the block start, then apply measured minute inputs."""
    x = np.asarray(x0, dtype=float).copy()
    minute_values: list[np.ndarray] = []
    model, scaler_x, scaler_u = select_model(block.mode, models)
    lin = linearize_surrogate(model, scaler_x, scaler_u, x, block.u_anchor, device)
    for minute_idx in range(block.start, block.end):
        u = measured_input_at(day_df, minute_idx, block.mode)
        x = lin.A @ x + lin.B @ u + lin.c
        if keep_all_minutes:
            minute_values.append(x.copy())
    return x, minute_values


def linear_5min_frozen_input_steps(
    x0: np.ndarray,
    block: ControlBlock,
    models: dict[str, tuple],
    device: torch.device,
    keep_all_minutes: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Freeze A, B, c and the block-start input through the 5-minute block."""
    x = np.asarray(x0, dtype=float).copy()
    minute_values: list[np.ndarray] = []
    model, scaler_x, scaler_u = select_model(block.mode, models)
    lin = linearize_surrogate(model, scaler_x, scaler_u, x, block.u_anchor, device)
    for _ in range(block.start, block.end):
        x = lin.A @ x + lin.B @ block.u_anchor + lin.c
        if keep_all_minutes:
            minute_values.append(x.copy())
    return x, minute_values


def rollout_fair_daily(
    day_df: pd.DataFrame,
    models: dict[str, tuple],
    device: torch.device,
    ctrl_period_min: int,
    split_on_mode_switch: bool,
    five_min_rollouts: Sequence[str],
) -> dict[str, object]:
    timestamps = pd.to_datetime(day_df["date"]).reset_index(drop=True)
    true_zones = day_df[AC_STATE_COLS].astype(float).to_numpy()
    x_cnn = true_zones[0].copy()
    selected_rollouts = set(five_min_rollouts)
    x_lin_5min = true_zones[0].copy() if "linear_5min" in selected_rollouts else None
    x_lin_5min_hold_u = (
        true_zones[0].copy() if "linear_5min_hold_u" in selected_rollouts else None
    )

    cnn_preds = [x_cnn.copy()]
    lin_5min_preds = [x_lin_5min.copy()] if x_lin_5min is not None else None
    lin_5min_hold_u_preds = (
        [x_lin_5min_hold_u.copy()] if x_lin_5min_hold_u is not None else None
    )
    control_horizon_anchor_indices = np.arange(
        0,
        len(day_df) - 1,
        ctrl_period_min,
        dtype=int,
    )

    for block in iter_control_blocks(day_df, ctrl_period_min, split_on_mode_switch):
        x_cnn, cnn_minutes = forward_steps(
            x_cnn,
            block,
            day_df,
            models,
            device,
            keep_all_minutes=True,
        )
        cnn_preds.extend(cnn_minutes)

        if x_lin_5min is not None and lin_5min_preds is not None:
            x_lin_5min, lin_5min_minutes = linear_5min_frozen_steps(
                x_lin_5min,
                block,
                day_df,
                models,
                device,
                keep_all_minutes=True,
            )
            lin_5min_preds.extend(lin_5min_minutes)

        if x_lin_5min_hold_u is not None and lin_5min_hold_u_preds is not None:
            x_lin_5min_hold_u, hold_u_minutes = linear_5min_frozen_input_steps(
                x_lin_5min_hold_u,
                block,
                models,
                device,
                keep_all_minutes=True,
            )
            lin_5min_hold_u_preds.extend(hold_u_minutes)

    rollout = {
        "timestamps": timestamps,
        "true_zones": true_zones,
        "cnn": np.vstack(cnn_preds),
        "control_horizon_anchor_indices": control_horizon_anchor_indices,
    }
    if lin_5min_preds is not None:
        rollout["linear_5min"] = np.vstack(lin_5min_preds)
    if lin_5min_hold_u_preds is not None:
        rollout["linear_5min_hold_u"] = np.vstack(lin_5min_hold_u_preds)
    return rollout


def zone_average_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred.mean(axis=1) - y_true.mean(axis=1))))


def n_window_switches(day_df: pd.DataFrame) -> int:
    return int(np.count_nonzero(np.diff(day_df[WINDOW_COL].astype(int).to_numpy())))


def write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.new{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"Could not overwrite locked CSV {path}; wrote {fallback} instead.")
        return fallback


def plot_model_line(
    ax: plt.Axes,
    timestamps: pd.Series,
    values: np.ndarray,
    style_key: str,
    *,
    markevery: int | list[int] | None = None,
) -> None:
    style = MODEL_STYLES[style_key]
    line_kwargs = {k: v for k, v in style.items() if k != "label"}
    if markevery is not None:
        line_kwargs["markevery"] = markevery
    ax.plot(
        timestamps,
        values,
        label=style["label"],
        dash_capstyle="round",
        solid_capstyle="round",
        **line_kwargs,
    )


def plot_fair_daily_rollout(
    day_df: pd.DataFrame,
    split_name: str,
    segment_id: int,
    rollout: dict[str, object],
    output_dir: Path,
    dpi: int,
) -> dict[str, object]:
    timestamps = rollout["timestamps"]
    true_zones = rollout["true_zones"]
    true_avg = true_zones.mean(axis=1)
    cnn = rollout["cnn"]
    five_min_keys = [key for key in FIVE_MIN_ROLLOUT_KEYS if key in rollout]
    horizon_anchor_indices = rollout["control_horizon_anchor_indices"]
    cnn_avg = cnn.mean(axis=1)
    five_min_avgs = {
        key: rollout[key].mean(axis=1)
        for key in five_min_keys
    }
    window_open = day_df[WINDOW_COL].astype(int).to_numpy() == 1

    all_y = np.concatenate(
        [
            true_avg,
            cnn_avg,
            *five_min_avgs.values(),
        ]
    )
    y_min = float(np.nanmin(all_y) - 0.25)
    y_max = float(np.nanmax(all_y) + 0.25)

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.fill_between(
        timestamps,
        y_min,
        y_max,
        where=window_open,
        color=SET2["lime"],
        alpha=0.16,
        step="post",
        linewidth=0,
    )
    ax.plot(
        timestamps,
        true_avg,
        color=GROUND_TRUTH_COLOR,
        linewidth=1.25,
        label="Ground truth",
        zorder=6,
    )
    plot_model_line(ax, timestamps, cnn_avg, "cnn")
    for key in five_min_keys:
        plot_model_line(
            ax,
            timestamps,
            five_min_avgs[key],
            key,
            markevery=horizon_anchor_indices.tolist(),
        )

    date_label = timestamps.iloc[0].date().isoformat()
    fig.suptitle(date_label, y=0.995, color=TEXT_COLOR, fontsize=16.0)
    ax.set_ylabel(r"Zone average temperature ($^\circ$C)")
    ax.set_xlabel("Time of day")
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color=GROUND_TRUTH_COLOR,
                linewidth=1.25,
                label="Ground truth",
            ),
            plt.Line2D(
                [0],
                [0],
                color=MODEL_STYLES["cnn"]["color"],
                linewidth=MODEL_STYLES["cnn"]["linewidth"],
                linestyle=MODEL_STYLES["cnn"]["linestyle"],
                label=MODEL_STYLES["cnn"]["label"],
                alpha=MODEL_STYLES["cnn"]["alpha"],
            ),
            *[
                plt.Line2D(
                    [0],
                    [0],
                    color=MODEL_STYLES[key]["color"],
                    linewidth=MODEL_STYLES[key]["linewidth"],
                    linestyle=MODEL_STYLES[key]["linestyle"],
                    marker=MODEL_STYLES[key]["marker"],
                    markersize=MODEL_STYLES[key]["markersize"],
                    markeredgewidth=MODEL_STYLES[key]["markeredgewidth"],
                    label=MODEL_STYLES[key]["label"],
                    alpha=MODEL_STYLES[key]["alpha"],
                )
                for key in five_min_keys
            ],
            Patch(facecolor=SET2["lime"], alpha=0.16, label="Windows open"),
        ]
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=12.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))

    split_dir = output_dir / "daily_rollout" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{split_name}_segment_{segment_id:03d}_{date_label}_fair_rollout.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "split": split_name,
        "segment": segment_id,
        "date": date_label,
        "n_points": len(day_df),
        "n_window_switches": n_window_switches(day_df),
        "nv_fraction": float(day_df[WINDOW_COL].astype(int).mean()),
        "cnn_lstm_zone_avg_mae_deg_c": zone_average_mae(true_zones, cnn),
        "plot_path": str(out_path),
    }
    for key in five_min_keys:
        summary[f"{key}_zone_avg_mae_deg_c"] = zone_average_mae(true_zones, rollout[key])
    return summary


def local_5min_records(
    day_df: pd.DataFrame,
    split_name: str,
    segment_id: int,
    models: dict[str, tuple],
    device: torch.device,
    ctrl_period_min: int,
    mode_aware: bool,
) -> list[dict[str, object]]:
    timestamps = pd.to_datetime(day_df["date"]).reset_index(drop=True)
    true_zones = day_df[AC_STATE_COLS].astype(float).to_numpy()
    records: list[dict[str, object]] = []
    n_transitions = len(day_df) - 1

    for period_start in range(0, n_transitions, ctrl_period_min):
        period_end = min(period_start + ctrl_period_min, n_transitions)
        if period_end - period_start < ctrl_period_min:
            continue

        status_window = day_df.iloc[period_start:period_end][WINDOW_COL].astype(int).to_numpy()
        has_mode_switch = bool(np.any(status_window != status_window[0]))
        if has_mode_switch and not mode_aware:
            continue

        blocks = control_blocks_for_period(
            day_df,
            period_start=period_start,
            period_end=period_end,
            split_on_mode_switch=mode_aware,
        )

        x0 = true_zones[period_start].copy()
        x_cnn = x0.copy()
        x_lin = x0.copy()
        for block in blocks:
            x_cnn, _ = forward_steps(
                x_cnn,
                block,
                day_df,
                models,
                device,
                keep_all_minutes=False,
            )
            x_lin, _ = linear_5min_frozen_steps(
                x_lin,
                block,
                day_df,
                models,
                device,
                keep_all_minutes=False,
            )

        cnn_error = x_cnn - true_zones[period_end]
        lin_vs_cnn_error = x_lin - x_cnn
        lin_vs_truth_error = x_lin - true_zones[period_end]

        base = {
            "split": split_name,
            "segment": segment_id,
            "date": timestamps.iloc[0].date().isoformat(),
            "t_start": timestamps.iloc[period_start],
            "t_end": timestamps.iloc[period_end],
            "start_index": period_start,
            "end_index": period_end,
            "mode_aware": mode_aware,
            "has_mode_switch": has_mode_switch,
            "mode_sequence": "".join("N" if int(v) == 1 else "A" for v in status_window),
        }
        for zone_idx, zone_name in enumerate(AC_STATE_COLS, start=1):
            records.append(
                {
                    **base,
                    "zone": f"Zone {zone_idx}",
                    "zone_name": zone_name,
                    "cnn_minus_truth_deg_c": float(cnn_error[zone_idx - 1]),
                    "linear_minus_cnn_deg_c": float(lin_vs_cnn_error[zone_idx - 1]),
                    "linear_minus_truth_deg_c": float(lin_vs_truth_error[zone_idx - 1]),
                }
            )
    return records


def plot_local_error_by_zone(
    records: pd.DataFrame,
    output_dir: Path,
    filename: str,
    title: str,
    dpi: int,
) -> Path | None:
    if records.empty:
        print(f"No records for {title}")
        return None

    zone_order = [f"Zone {i}" for i in range(1, 6)]
    abs_errors = records.assign(abs_linear_minus_cnn=records["linear_minus_cnn_deg_c"].abs())
    stats = (
        abs_errors.groupby("zone")["abs_linear_minus_cnn"]
        .agg(["mean", "std", "count"])
        .reindex(zone_order)
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    x = np.arange(len(zone_order))

    for zone_idx, zone in enumerate(zone_order):
        color = list(SET2.values())[zone_idx % len(SET2)]
        vals = abs_errors.loc[abs_errors["zone"] == zone, "abs_linear_minus_cnn"].to_numpy()
        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), zone_idx) + jitter,
                vals,
                s=14,
                alpha=0.28,
                color=color,
                linewidths=0,
                label="5-min anchors" if zone_idx == 0 else None,
            )
        row = stats.loc[stats["zone"] == zone]
        if not row.empty and not np.isnan(float(row["mean"].iloc[0])):
            ax.errorbar(
                [zone_idx],
                [float(row["mean"].iloc[0])],
                yerr=[float(row["std"].iloc[0])],
                fmt="D",
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=5,
                markersize=7,
                markeredgecolor=GROUND_TRUTH_COLOR,
                markeredgewidth=0.8,
                label="Mean abs error +/- std" if zone_idx == 0 else None,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(zone_order)
    ax.set_ylabel("|linearized 5-min - CNN-LSTM 5-min| (deg C)")
    ax.set_title(title, loc="left", color=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_local_errors(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    return (
        records.assign(abs_linear_minus_cnn=records["linear_minus_cnn_deg_c"].abs())
        .groupby(["mode_aware", "has_mode_switch", "zone"])["abs_linear_minus_cnn"]
        .agg(["count", "mean", "std", "max"])
        .reset_index()
    )


def normalize_split_names(splits: Sequence[str]) -> list[str]:
    return ["val" if split == "validation" else split for split in splits]


def normalize_five_min_rollouts(requested: Sequence[str]) -> list[str]:
    if "none" in requested:
        return []
    if "both" in requested:
        return list(FIVE_MIN_ROLLOUT_KEYS)

    selected: list[str] = []
    for name in requested:
        key = FIVE_MIN_ROLLOUT_ARG_TO_KEY[name]
        if key not in selected:
            selected.append(key)
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    models = {
        "ac": load_surrogate(args.ac_model, device),
        "nv": load_surrogate(args.nv_model, device),
    }

    segments = clean_working_hour_segments(args.data, args.min_len, args.exclude_date)
    split_ids = split_segments(segments)
    requested_splits = normalize_split_names(args.splits)
    five_min_rollouts = normalize_five_min_rollouts(args.five_min_rollouts)

    print(f"Prepared {len(segments)} clean segments.")
    for split_name in ["train", "val", "test"]:
        ids = split_ids[split_name]
        span = f"{ids[0]}-{ids[-1]}" if ids else "none"
        print(f"{split_name}: {len(ids)} segments ({span})")

    daily_summary_rows: list[dict[str, object]] = []
    split_on_mode_switch = bool(args.split_on_mode_switch and not args.hold_mode_through_switch)
    if split_on_mode_switch:
        print("Mode switches inside a control period will split the 5-minute block.")
    else:
        print("Holding the block-start mode for each 5-minute control period.")
    if five_min_rollouts:
        names = ", ".join(MODEL_STYLES[key]["label"] for key in five_min_rollouts)
        print(f"Plotting 5-minute rollouts: {names}")
    else:
        print("Plotting no 5-minute rollout trajectories.")
    for split_name in requested_splits:
        ids = split_ids[split_name]
        if args.max_days_per_split is not None:
            ids = ids[: args.max_days_per_split]

        for segment_id in ids:
            day_df = segments[segment_id].copy().reset_index(drop=True)
            date_label = pd.to_datetime(day_df["date"]).iloc[0].date().isoformat()
            print(f"Processing {split_name} segment {segment_id} ({date_label})")

            rollout = rollout_fair_daily(
                day_df,
                models,
                device,
                ctrl_period_min=args.ctrl_period_min,
                split_on_mode_switch=split_on_mode_switch,
                five_min_rollouts=five_min_rollouts,
            )
            daily_summary_rows.append(
                plot_fair_daily_rollout(
                    day_df,
                    split_name,
                    segment_id,
                    rollout,
                    output_dir,
                    args.dpi,
                )
            )

    daily_summary = pd.DataFrame(daily_summary_rows)
    daily_summary_path = output_dir / "daily_fair_rollout_summary.csv"
    daily_summary_path = write_csv_with_fallback(daily_summary, daily_summary_path)

    print(f"Wrote {len(daily_summary)} daily rollout plots under {output_dir / 'daily_rollout'}")
    print(f"Wrote daily summary CSV: {daily_summary_path}")


if __name__ == "__main__":
    main()
