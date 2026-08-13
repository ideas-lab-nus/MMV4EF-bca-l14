#!/usr/bin/env python3
"""Create a 2024-08-23 MMV/PV motivation figure with window status.

The figure uses the corrected L14 dataset, resamples all plotted signals to
15-minute means, and shows:
- total FCU/PFCU cooling power
- global solar radiation (GHI)
- aggregate window open/close status
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13.0,
        "axes.titlesize": 15.0,
        "axes.labelsize": 14.0,
        "legend.fontsize": 11.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
    }
)


COOLING_COLS = [
    "FCU-1 Cooling Load_kW",
    "FCU-2 Cooling Load_kW",
    "FCU-3 Cooling Load_kW",
    "FCU-4 Cooling Load_kW",
    "FCU-5 Cooling Load_kW",
    "PFCU-1 Cooling Load_kW",
    "PFCU-2 Cooling Load_kW",
]

FAN_WATT_COLS = [
    "FCU-01 Watt",
    "FCU-02 Watt",
    "FCU-03 Watt",
    "FCU-04 Watt",
    "FCU-05 Watt",
    "PFCU-01 Watt",
    "PFCU-02 Watt",
]

WINDOW_COLS = [
    "Z7 Windows Open Close Status",
    "Z6 Windows Open Close Status",
    "Z5 Windows Open Close Status",
    "Z1 Windows Open Close Status",
    "Z2 Windows Open Close Status",
    "Z3 Windows Open Close Status",
]

SOLAR_COL = "Solar Radiation"
RAIN_COL = "rain_status"
OUTDOOR_TEMP_COL = "OutdoorTemperatureWindow"
SOLAR_DISPLAY_NAME = "GHI"
SOLAR_AXIS_LABEL = r"global solar radiation (GHI) ($\mathrm{W\,m^{-2}}$)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 15-minute resampled MMV/PV motivation figure for one day."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/private/l14_merged_data_with_rain.csv"),
        help="Input L14 merged CSV.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2024-08-23",
        help="Date to plot in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for the figure.",
    )
    parser.add_argument(
        "--resample-minutes",
        type=int,
        default=15,
        help="Resampling interval in minutes.",
    )
    parser.add_argument(
        "--power-scope",
        choices=["cooling", "cooling_plus_fan"],
        default="cooling",
        help="Power series to show. Default is cooling load only.",
    )
    parser.add_argument(
        "--save-resampled-data",
        action="store_true",
        help="Also write the derived plotting table under outputs/ (off by default).",
    )
    return parser.parse_args()


def parse_datetime_col(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %H:%M", errors="coerce")
    if parsed.isna().any():
        parsed_alt = pd.to_datetime(series, format="%m/%d/%Y %H:%M", errors="coerce")
        parsed = parsed.fillna(parsed_alt)
    if parsed.isna().any():
        parsed_fallback = pd.to_datetime(series, errors="coerce")
        parsed = parsed.fillna(parsed_fallback)
    return parsed


def load_day_data(path: Path, target_date: str, power_scope: str, resample_minutes: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    required = ["date", SOLAR_COL, RAIN_COL, OUTDOOR_TEMP_COL] + COOLING_COLS + WINDOW_COLS
    if power_scope == "cooling_plus_fan":
        required += FAN_WATT_COLS

    df = pd.read_csv(path, usecols=required)
    df["date"] = parse_datetime_col(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    day = pd.Timestamp(target_date).date()
    df = df.loc[df["date"].dt.date == day].copy()
    if df.empty:
        raise ValueError(f"No data found for {target_date}")

    numeric_cols = [c for c in required if c != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[COOLING_COLS] = df[COOLING_COLS].clip(lower=0.0).fillna(0.0)
    df[SOLAR_COL] = df[SOLAR_COL].clip(lower=0.0).ffill().fillna(0.0)
    df[WINDOW_COLS] = df[WINDOW_COLS].clip(lower=0.0, upper=1.0).fillna(0.0)
    df[RAIN_COL] = df[RAIN_COL].fillna(0.0)
    df[OUTDOOR_TEMP_COL] = df[OUTDOOR_TEMP_COL].ffill().bfill()

    df["cooling_power_kw"] = df[COOLING_COLS].sum(axis=1)
    if power_scope == "cooling_plus_fan":
        df[FAN_WATT_COLS] = df[FAN_WATT_COLS].clip(lower=0.0).fillna(0.0)
        df["fan_power_kw"] = df[FAN_WATT_COLS].sum(axis=1) / 1000.0
        df["power_kw"] = df["cooling_power_kw"] + df["fan_power_kw"]
        df["power_label"] = "cooling + fan power"
    else:
        df["fan_power_kw"] = 0.0
        df["power_kw"] = df["cooling_power_kw"]
        df["power_label"] = "cooling power"
    df["window_open_status"] = (df[WINDOW_COLS].median(axis=1) >= 0.5).astype(float)
    df["is_raining"] = (df[RAIN_COL] > 0.0).astype(float)

    numeric_mean_cols = ["power_kw", "cooling_power_kw", "fan_power_kw", SOLAR_COL, OUTDOOR_TEMP_COL]
    mean_part = df.set_index("date")[numeric_mean_cols].resample(f"{resample_minutes}min").mean()
    state_part = df.set_index("date")[["window_open_status", "is_raining"]].resample(
        f"{resample_minutes}min"
    ).last()
    out = pd.concat([mean_part, state_part], axis=1).dropna(how="all").reset_index()
    out["window_open_status"] = out["window_open_status"].ffill().bfill().round().clip(0, 1)
    out["is_raining"] = out["is_raining"].fillna(0.0).round().clip(0, 1)
    out["power_label"] = df["power_label"].iloc[0]
    return out


def shade_rain_periods(ax: plt.Axes, plot_df: pd.DataFrame, step: pd.Timedelta) -> None:
    if "is_raining" not in plot_df.columns or not (plot_df["is_raining"] > 0).any():
        return

    start = None
    raining = (plot_df["is_raining"] > 0.0).to_numpy()
    timestamps = plot_df["date"].reset_index(drop=True)
    for idx, is_raining in enumerate(raining):
        if is_raining and start is None:
            start = timestamps.iloc[idx]
        is_last = idx == len(raining) - 1
        if start is not None and (not is_raining or is_last):
            end_idx = idx if is_raining and is_last else idx - 1
            end = timestamps.iloc[end_idx] + step
            ax.axvspan(start, end, color="0.85", alpha=0.22, linewidth=0, zorder=0)
            start = None


def make_figure(plot_df: pd.DataFrame, output_path: Path, target_date: str, resample_minutes: int) -> None:
    set2 = plt.get_cmap("Set2")
    power_color = set2(0)
    solar_color = set2(1)
    window_color = "#4b5563"
    temp_color = set2(2)

    step = pd.Timedelta(minutes=resample_minutes)
    fig, (ax, ax_window) = plt.subplots(
        2,
        1,
        figsize=(9.4, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.9], "hspace": 0.12},
        constrained_layout=True,
    )

    shade_rain_periods(ax, plot_df, step)
    shade_rain_periods(ax_window, plot_df, step)

    power_label = str(plot_df["power_label"].iloc[0])
    power_line = ax.plot(
        plot_df["date"],
        plot_df["power_kw"],
        color=power_color,
        linewidth=2.6,
        label=power_label,
    )[0]
    ax.fill_between(
        plot_df["date"],
        0.0,
        plot_df["power_kw"].to_numpy(dtype=float),
        color=power_color,
        alpha=0.25,
        linewidth=0,
    )
    ax.set_ylabel("cooling power (kW)" if power_label == "cooling power" else "HVAC power (kW)")
    fig.suptitle(target_date, y=0.995, fontsize=15.0)
    ax.grid(False)

    ax_solar = ax.twinx()
    solar_line = ax_solar.plot(
        plot_df["date"],
        plot_df[SOLAR_COL],
        color=solar_color,
        linewidth=2.3,
        label=SOLAR_DISPLAY_NAME,
    )[0]
    ax_solar.set_ylabel(SOLAR_AXIS_LABEL)
    ax_solar.grid(False)

    ax_window.step(
        plot_df["date"],
        plot_df["window_open_status"],
        where="post",
        color=window_color,
        linewidth=2.0,
        label="window status",
    )
    ax_window.fill_between(
        plot_df["date"],
        0.0,
        plot_df["window_open_status"].to_numpy(dtype=float),
        step="post",
        color=window_color,
        alpha=0.12,
        linewidth=0,
    )
    ax_window.set_ylim(-0.08, 1.08)
    ax_window.set_yticks([0, 1])
    ax_window.set_yticklabels(["closed", "open"])
    ax_window.set_ylabel("window\nstatus")
    ax_window.set_xlabel("time of day")
    ax_window.grid(False)
    ax_window.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax_temp = ax_window.twinx()
    temp_line = ax_temp.plot(
        plot_df["date"],
        plot_df[OUTDOOR_TEMP_COL],
        color=temp_color,
        linewidth=2.0,
        label=r"$T_{OA}$",
    )[0]
    ax_temp.set_ylabel(r"$T_{OA}$ ($^\circ$C)")
    ax_temp.grid(False)

    handles = [power_line, solar_line, ax_window.lines[0], temp_line]
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=11.5,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.84))

    for axis in [ax, ax_solar, ax_window, ax_temp]:
        axis.spines["top"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = load_day_data(args.input, args.date, args.power_scope, args.resample_minutes)

    stem = f"mmv_motivation_{args.date.replace('-', '')}_{args.resample_minutes}min_{args.power_scope}"
    output_path = args.output_dir / f"{stem}.png"
    make_figure(plot_df, output_path, args.date, args.resample_minutes)

    print(output_path)
    print(output_path.with_suffix(".svg"))
    if args.save_resampled_data:
        data_path = Path("outputs/analysis/motivation") / f"{stem}_resampled.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(data_path, index=False)
        print(data_path)


if __name__ == "__main__":
    main()
