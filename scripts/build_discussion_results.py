"""Build validated evidence and figures for Discussion Sections 4.1--4.3.

The calculation boundary is intentionally narrow: forecast robustness compares
only observed future disturbances with LSTM64 disturbances, while weather and
PV interpretation reuse the accepted observed-future result tables.  HVAC
power is independently reconstructed with the heat balance already defined in
the manuscript; this module introduces no empirical energy model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from build_observed_future_results import (
    FCU_COLUMNS,
    PFCU_COLUMNS,
    POWER_ACCOUNTING_ATOL_KW,
    STEP_HOURS,
    ZONE_COLUMNS,
    attach_observed_context,
    attach_pmv,
    audit_hvac_power,
    load_observed_context,
    recompute_hvac_power_kw,
)

FORECAST_CASES = (
    ("no_pv", "observed"),
    ("no_pv", "lstm64"),
    ("onsite_pv", "observed"),
    ("onsite_pv", "lstm64"),
)
CASE_BY_KEY = {
    ("no_pv", "observed"): "MIQP no PV | observed future",
    ("no_pv", "lstm64"): "MIQP no PV | LSTM64",
    ("onsite_pv", "observed"): "MIQP onsite PV | observed future",
    ("onsite_pv", "lstm64"): "MIQP onsite PV | LSTM64",
}
KEY_BY_CASE = {case: key for key, case in CASE_BY_KEY.items()}
OBJECTIVE_LABELS = {"no_pv": "MPC-CA", "onsite_pv": "MPC-PV"}
SOURCE_LABELS = {"observed": "Perfect-forecast", "lstm64": "LSTM64"}
from plot_styles import FORECAST_COLORS, GRID_COLORS, SELF_COLORS, focus_temperature_axes

OBJECTIVE_COLORS = {"no_pv": "#8da0cb", "onsite_pv": "#66c2a5"}
SOURCE_STYLES = {"observed": "-", "lstm64": "--"}

OBSERVED_CASE_LABELS = {
    "AC baseline": "RBC-AC",
    "RBC baseline": "RBC-MM",
    "MIQP no PV | observed future": "MPC-CA",
    "MIQP onsite PV | observed future": "MPC-PV",
}
OBSERVED_COLORS = {
    "RBC-AC": "#4D4D4D",
    "RBC-MM": "#E69F00",
    "MPC-CA": "#8da0cb",
    "MPC-PV": "#66c2a5",
}

FULL_DATE_COUNT = 22
FULL_SAMPLES_PER_DATE_CASE = 690
MORNING_START_MINUTE = 9 * 60
MORNING_END_MINUTE = 11 * 60 + 30
EVENING_START_MINUTE = 16 * 60
EVENING_END_MINUTE = 18 * 60 + 30

DEFAULT_SOURCE_ROOT = Path("outputs/simulations/main")
DEFAULT_STEM = "future_data_source_comparison_test_val_union_all_full"
DEFAULT_TIMESERIES_PATH = DEFAULT_SOURCE_ROOT / f"{DEFAULT_STEM}_timeseries.csv"
DEFAULT_VALIDATION_PATH = DEFAULT_SOURCE_ROOT / f"{DEFAULT_STEM}_validation.csv"
DEFAULT_OBSERVED_CONTEXT_PATH = Path("data/private/l14_merged_data_with_rain.csv")
DEFAULT_OBSERVED_DAILY_PATH = Path(
    "outputs/analysis/observed_future_results/observed_future_daily_metrics.csv"
)
DEFAULT_OBSERVED_POOLED_PATH = Path(
    "outputs/analysis/observed_future_results/observed_future_pooled_metrics.csv"
)
DEFAULT_FORECAST_METRICS_PATH = Path(
    "outputs/lstm64/metrics_aggregate.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/discussion_results")
DEFAULT_PAPER_FIG_DIR = Path("figures")


@dataclass(frozen=True)
class FigureResult:
    png_path: Path
    pdf_path: Path
    axes_count: int


@dataclass(frozen=True)
class PipelineResult:
    validated: bool
    date_count: int
    minute_rows: int
    output_paths: dict[str, Path]
    selected_dates: dict[str, str]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16.0,
            "axes.titlesize": 17.0,
            "axes.labelsize": 16.0,
            "legend.fontsize": 14.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> FigureResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    png_path = output / f"{stem}.png"
    pdf_path = output / f"{stem}.pdf"
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "MMV4EF discussion analysis"},
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "MMV4EF discussion analysis",
            "CreationDate": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "ModDate": datetime(2024, 1, 1, tzinfo=timezone.utc),
        },
    )
    result = FigureResult(png_path, pdf_path, len(figure.axes))
    plt.close(figure)
    return result


def load_forecast_cases(path: Path, *, validate_full: bool = True) -> pd.DataFrame:
    """Load the four approved MPC/source combinations and audit their energy."""

    required = [
        "ts",
        "case",
        "controller_objective",
        "forecast_source",
        "z",
        "T_out",
        "T_mean",
        "pv_kw",
        "hvac_kw",
        "grid_kw",
        "export_kw",
        "self_kw",
        *ZONE_COLUMNS,
        *FCU_COLUMNS,
        *PFCU_COLUMNS,
    ]
    source = pd.read_csv(path)
    _require_columns(source, required, "future trajectory")
    approved_labels = set(CASE_BY_KEY.values())
    frame = source.loc[source["case"].isin(approved_labels), required].copy()
    if set(frame["case"].unique()) != approved_labels:
        missing = sorted(approved_labels - set(frame["case"].unique()))
        raise ValueError(f"missing forecast comparison cases: {missing}")

    expected = frame["case"].map(KEY_BY_CASE)
    expected_objective = expected.map(lambda key: key[0])
    expected_source = expected.map(lambda key: key[1])
    if not frame["controller_objective"].eq(expected_objective).all():
        raise ValueError("case/controller-objective metadata contradiction")
    if not frame["forecast_source"].eq(expected_source).all():
        raise ValueError("case/forecast-source metadata contradiction")

    frame["ts"] = pd.to_datetime(frame["ts"], errors="raise")
    if frame["ts"].dt.tz is not None:
        raise ValueError("trajectory timestamps must be timezone-naive local time")
    if frame.duplicated(["ts", "case"]).any():
        raise ValueError("duplicate timestamp/case keys in forecast trajectories")

    numeric = [
        "z",
        "T_out",
        "T_mean",
        "pv_kw",
        "hvac_kw",
        "grid_kw",
        "export_kw",
        "self_kw",
        *ZONE_COLUMNS,
        *FCU_COLUMNS,
        *PFCU_COLUMNS,
    ]
    frame.loc[:, numeric] = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    active_fcu = frame["z"].eq(1)
    active_pfcu = frame["z"].eq(0)
    always_finite = ["z", "T_out", "T_mean", "pv_kw", "hvac_kw", "grid_kw", "export_kw", "self_kw", *ZONE_COLUMNS]
    if not np.isfinite(frame.loc[:, always_finite].to_numpy(float)).all():
        raise ValueError("non-finite required trajectory values")
    if active_fcu.any() and not np.isfinite(frame.loc[active_fcu, FCU_COLUMNS].to_numpy(float)).all():
        raise ValueError("non-finite active FCU supply temperatures")
    if active_pfcu.any() and not np.isfinite(frame.loc[active_pfcu, PFCU_COLUMNS].to_numpy(float)).all():
        raise ValueError("non-finite active PFCU supply temperatures")
    if not frame["z"].isin((0.0, 1.0)).all():
        raise ValueError("mode state z must be binary")

    stored_hvac = frame["hvac_kw"].to_numpy(float)
    expected_grid = np.maximum(stored_hvac - frame["pv_kw"].to_numpy(float), 0.0)
    expected_export = np.maximum(frame["pv_kw"].to_numpy(float) - stored_hvac, 0.0)
    expected_self = np.minimum(stored_hvac, frame["pv_kw"].to_numpy(float))
    for column, expected_values in (
        ("grid_kw", expected_grid),
        ("export_kw", expected_export),
        ("self_kw", expected_self),
    ):
        if not np.allclose(frame[column], expected_values, rtol=0.0, atol=POWER_ACCOUNTING_ATOL_KW):
            raise ValueError(f"stored PV accounting failed for {column}")

    audit = audit_hvac_power(frame)
    frame["hvac_kw"] = recompute_hvac_power_kw(frame)
    frame["grid_kw"] = np.maximum(frame["hvac_kw"] - frame["pv_kw"], 0.0)
    frame["export_kw"] = np.maximum(frame["pv_kw"] - frame["hvac_kw"], 0.0)
    frame["self_kw"] = np.minimum(frame["hvac_kw"], frame["pv_kw"])

    pv_spread = frame.groupby("ts", sort=False)["pv_kw"].agg(lambda x: float(x.max() - x.min()))
    if (pv_spread > POWER_ACCOUNTING_ATOL_KW).any():
        raise ValueError("PV availability differs by forecast case")

    if validate_full:
        frame["date"] = frame["ts"].dt.strftime("%Y-%m-%d")
        counts = frame.groupby(["date", "case"], observed=True).size()
        if frame["date"].nunique() != FULL_DATE_COUNT:
            raise ValueError(f"expected {FULL_DATE_COUNT} matched dates")
        if len(counts) != FULL_DATE_COUNT * len(FORECAST_CASES) or not counts.eq(FULL_SAMPLES_PER_DATE_CASE).all():
            raise ValueError("expected 690 one-minute samples per date and forecast case")
        for (_, _), group in frame.groupby(["date", "case"], sort=False, observed=True):
            stamps = group["ts"].sort_values()
            if stamps.iloc[0].strftime("%H:%M") != "07:30" or stamps.iloc[-1].strftime("%H:%M") != "18:59":
                raise ValueError("every trajectory must span 07:30 through 18:59")
            if not stamps.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
                raise ValueError("trajectory timestamps must be one minute apart")
    else:
        frame["date"] = frame["ts"].dt.strftime("%Y-%m-%d")

    frame = frame.sort_values(["ts", "case"], kind="stable").reset_index(drop=True)
    frame.attrs["energy_audit"] = audit
    return frame


def validate_safety_table(path: Path, dates: Sequence[str]) -> dict[str, Any]:
    validation = pd.read_csv(path)
    _require_columns(
        validation,
        ["date", "case", "expected_rows", "actual_rows", "exact_index", "rain_lockout_violations"],
        "safety validation",
    )
    selected = validation.loc[validation["case"].isin(CASE_BY_KEY.values())].copy()
    if set(selected["case"].unique()) != set(CASE_BY_KEY.values()):
        raise ValueError("safety validation is missing an approved forecast case")
    selected["date"] = pd.to_datetime(selected["date"]).dt.strftime("%Y-%m-%d")
    if set(selected["date"]) != set(dates):
        raise ValueError("safety-validation dates do not match trajectory dates")
    if not selected["actual_rows"].eq(selected["expected_rows"]).all():
        raise ValueError("safety validation contains incomplete trajectories")
    exact = selected["exact_index"].astype(str).str.lower().isin(("true", "1"))
    if not exact.all():
        raise ValueError("safety validation contains non-exact indices")
    if not pd.to_numeric(selected["rain_lockout_violations"], errors="coerce").eq(0).all():
        raise ValueError("rain safety violation in forecast comparison")
    return {
        "rows": int(len(selected)),
        "rain_lockout_violations": 0,
        "all_indices_exact": True,
    }


def _load_ac27_reference(path: Path, dates: Sequence[str]) -> pd.DataFrame:
    daily = pd.read_csv(path)
    _require_columns(
        daily,
        ["date", "case", "hvac_kwh", "morning_hvac_kwh", "evening_hvac_kwh"],
        "observed daily metrics",
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    reference = daily.loc[daily["case"].eq("AC baseline")].copy()
    if reference.duplicated("date").any() or set(reference["date"]) != set(dates):
        raise ValueError("AC27 reference dates do not match forecast trajectories")
    return reference.set_index("date")


def compute_lstm64_daily_metrics(
    minute: pd.DataFrame, ac27_reference: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-date metrics with AC27 as the sole DR denominator."""

    required = [
        "ts",
        "date",
        "case",
        "controller_objective",
        "forecast_source",
        "z",
        "T_mean",
        "hvac_kw",
        "grid_kw",
        "pv_kw",
        "self_kw",
        "export_kw",
        "pmv_mean",
        *ZONE_COLUMNS,
        *[f"pmv_zone_{zone}" for zone in range(1, 6)],
    ]
    _require_columns(minute, required, "enriched forecast trajectories")
    records: list[dict[str, object]] = []
    pmv_columns = [f"pmv_zone_{zone}" for zone in range(1, 6)]
    for (date, case), group in minute.groupby(["date", "case"], sort=True, observed=True):
        group = group.sort_values("ts", kind="stable")
        objective, source = KEY_BY_CASE[str(case)]
        minute_of_day = group["ts"].dt.hour * 60 + group["ts"].dt.minute
        morning = minute_of_day.ge(MORNING_START_MINUTE) & minute_of_day.lt(MORNING_END_MINUTE)
        evening = minute_of_day.ge(EVENING_START_MINUTE) & minute_of_day.lt(EVENING_END_MINUTE)
        energy = {
            f"{stem}_kwh": float(group[column].sum() * STEP_HOURS)
            for stem, column in (
                ("hvac", "hvac_kw"),
                ("grid", "grid_kw"),
                ("pv", "pv_kw"),
                ("self", "self_kw"),
                ("export", "export_kw"),
            )
        }
        pmv_values = group.loc[:, pmv_columns].to_numpy(float)
        zone_values = group.loc[:, ZONE_COLUMNS].to_numpy(float)
        z = group["z"].round().astype(int).to_numpy()
        record: dict[str, object] = {
            "date": str(date),
            "case": str(case),
            "controller_objective": objective,
            "controller": OBJECTIVE_LABELS[objective],
            "forecast_source": source,
            "source_label": SOURCE_LABELS[source],
            "sample_count": int(len(group)),
            "zone_sample_count": int(pmv_values.size),
            **energy,
            "sc_pct": _safe_pct(energy["self_kwh"], energy["pv_kwh"]),
            "ss_pct": _safe_pct(energy["self_kwh"], energy["hvac_kwh"]),
            "mean_temp_c": float(zone_values.mean()),
            "max_temp_c": float(zone_values.max()),
            "mean_pmv": float(pmv_values.mean()),
            "pmv_within_05_pct": 100.0 * float(np.mean(np.abs(pmv_values) <= 0.5)),
            "pmv_within_10_pct": 100.0 * float(np.mean(np.abs(pmv_values) <= 1.0)),
            "window_open_fraction": float(np.mean(1 - z)),
            "switch_count": int(np.count_nonzero(z[1:] != z[:-1])),
            "morning_hvac_kwh": float(group.loc[morning, "hvac_kw"].sum() * STEP_HOURS),
            "evening_hvac_kwh": float(group.loc[evening, "hvac_kw"].sum() * STEP_HOURS),
        }
        ac = ac27_reference.loc[str(date)]
        record.update(
            {
                "ac27_hvac_kwh": float(ac["hvac_kwh"]),
                "ac27_morning_hvac_kwh": float(ac["morning_hvac_kwh"]),
                "ac27_evening_hvac_kwh": float(ac["evening_hvac_kwh"]),
            }
        )
        record["dr_e_pct"] = _safe_pct(record["ac27_hvac_kwh"] - energy["hvac_kwh"], record["ac27_hvac_kwh"])
        record["dr_p_morning_pct"] = _safe_pct(record["ac27_morning_hvac_kwh"] - record["morning_hvac_kwh"], record["ac27_morning_hvac_kwh"])
        record["dr_p_evening_pct"] = _safe_pct(record["ac27_evening_hvac_kwh"] - record["evening_hvac_kwh"], record["ac27_evening_hvac_kwh"])
        records.append(record)
    result = pd.DataFrame(records)
    expected = len(result["date"].unique()) * len(FORECAST_CASES)
    if len(result) != expected:
        raise ValueError("daily forecast metrics are not complete")
    return result.sort_values(["date", "controller_objective", "forecast_source"], kind="stable").reset_index(drop=True)


def pool_lstm64_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (objective, source), group in daily.groupby(
        ["controller_objective", "forecast_source"], sort=False, observed=True
    ):
        sums = {
            column: float(group[column].sum())
            for column in (
                "hvac_kwh",
                "grid_kwh",
                "pv_kwh",
                "self_kwh",
                "export_kwh",
                "morning_hvac_kwh",
                "evening_hvac_kwh",
            )
        }
        weights = group.get("zone_sample_count", pd.Series(1.0, index=group.index)).to_numpy(float)
        record: dict[str, object] = {
            "controller_objective": objective,
            "controller": OBJECTIVE_LABELS.get(str(objective), str(objective)),
            "forecast_source": source,
            "source_label": SOURCE_LABELS.get(str(source), str(source)),
            "date_count": int(group["date"].nunique()),
            **sums,
            "sc_pct": _safe_pct(sums["self_kwh"], sums["pv_kwh"]),
            "ss_pct": _safe_pct(sums["self_kwh"], sums["hvac_kwh"]),
        }
        for column in ("mean_temp_c", "mean_pmv", "pmv_within_05_pct", "pmv_within_10_pct", "window_open_fraction"):
            if column in group:
                record[column] = float(np.average(group[column], weights=weights))
        if {"ac27_hvac_kwh", "ac27_morning_hvac_kwh", "ac27_evening_hvac_kwh"}.issubset(group.columns):
            ac_hvac = float(group["ac27_hvac_kwh"].sum())
            ac_morning = float(group["ac27_morning_hvac_kwh"].sum())
            ac_evening = float(group["ac27_evening_hvac_kwh"].sum())
            record["dr_e_pct"] = _safe_pct(ac_hvac - sums["hvac_kwh"], ac_hvac)
            record["dr_p_morning_pct"] = _safe_pct(ac_morning - sums["morning_hvac_kwh"], ac_morning)
            record["dr_p_evening_pct"] = _safe_pct(ac_evening - sums["evening_hvac_kwh"], ac_evening)
        else:
            for column in ("dr_e_pct", "dr_p_morning_pct", "dr_p_evening_pct"):
                record[column] = float(group[column].mean())
        records.append(record)
    return pd.DataFrame(records).sort_values(["controller_objective", "forecast_source"], kind="stable").reset_index(drop=True)


def build_paired_lstm64_deltas(daily: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "hvac_kwh",
        "grid_kwh",
        "dr_e_pct",
        "dr_p_morning_pct",
        "dr_p_evening_pct",
        "sc_pct",
        "ss_pct",
        "mean_temp_c",
        "pmv_within_05_pct",
        "window_open_fraction",
    ]
    observed = daily.loc[daily["forecast_source"].eq("observed"), ["date", "controller_objective", *value_columns]].set_index(["date", "controller_objective"])
    lstm = daily.loc[daily["forecast_source"].eq("lstm64"), ["date", "controller_objective", *value_columns]].set_index(["date", "controller_objective"])
    if not observed.index.equals(lstm.index):
        raise ValueError("observed and LSTM64 daily indices do not pair exactly")
    out = pd.DataFrame(index=observed.index).reset_index()
    for column in value_columns:
        out[f"observed_{column}"] = observed[column].to_numpy(float)
        out[f"lstm64_{column}"] = lstm[column].to_numpy(float)
        out[f"delta_lstm64_minus_observed_{column}"] = lstm[column].to_numpy(float) - observed[column].to_numpy(float)
    return out


def select_lstm64_profile_dates(minute):
    """Exploratory examples with the largest matched window-state differences."""
    records = []
    for objective in ['no_pv', 'onsite_pv']:
        candidates = []
        subset = minute[minute.controller_objective.eq(objective)]
        for date, group in subset.groupby('date', sort=True):
            paired = group.pivot(index='ts', columns='forecast_source', values='z')
            if paired.isna().any().any() or set(paired.columns) != {'observed', 'lstm64'}:
                raise ValueError('Profile selection requires complete matched forecast trajectories.')
            candidates.append({'date': str(date), 'mode_difference_minutes': int(paired.observed.ne(paired.lstm64).sum())})
        winner = sorted(candidates, key=lambda row: (-row['mode_difference_minutes'], row['date']))[0]
        records.append({'controller_objective': objective, 'controller': OBJECTIVE_LABELS[objective],
                        'date': winner['date'], 'selection_metric': 'largest_mode_difference_minutes',
                        'absolute_selection_value': winner['mode_difference_minutes'],
                        'exploratory_selection': True})
    return pd.DataFrame(records)


def load_weather_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path)
    required = ["date", "case", "mean_t_out_c", "rain_minutes", "pv_kwh", "dr_e_pct", "window_open_fraction", "ss_pct"]
    _require_columns(daily, required, "observed daily metrics")
    cases = ("RBC baseline", "MIQP no PV | observed future", "MIQP onsite PV | observed future")
    frame = daily.loc[daily["case"].isin(cases), required].copy()
    if set(frame["case"].unique()) != set(cases):
        raise ValueError("weather analysis is missing an observed-future controller")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["controller"] = frame["case"].map(OBSERVED_CASE_LABELS)
    counts = frame.groupby("controller", observed=True)["date"].nunique()
    if len(counts) != 3 or not counts.eq(FULL_DATE_COUNT).all():
        raise ValueError("weather analysis requires 22 dates for RBC, MPC, and MPC-PV")
    return frame.sort_values(["date", "controller"], kind="stable").reset_index(drop=True)


def compute_weather_associations(weather_daily: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        weather_daily,
        ["controller", "mean_t_out_c", "rain_minutes", "pv_kwh", "dr_e_pct", "window_open_fraction", "ss_pct"],
        "weather association data",
    )
    specifications = [
        ("temperature_dr", "mean_t_out_c", "dr_e_pct", ("RBC-MM", "MPC-CA", "MPC-PV")),
        ("rain_window", "rain_minutes", "window_open_fraction", ("RBC-MM", "MPC-CA", "MPC-PV")),
        ("pv_dr", "pv_kwh", "dr_e_pct", ("MPC-PV",)),
        ("pv_ss", "pv_kwh", "ss_pct", ("MPC-PV",)),
    ]
    records: list[dict[str, object]] = []
    for panel, x_column, y_column, controllers in specifications:
        for controller in controllers:
            selected = weather_daily.loc[weather_daily["controller"].eq(controller), [x_column, y_column]].dropna()
            rho, p_value = spearmanr(selected[x_column].to_numpy(float), selected[y_column].to_numpy(float))
            records.append(
                {
                    "panel": panel,
                    "controller": controller,
                    "x_column": x_column,
                    "y_column": y_column,
                    "n": int(len(selected)),
                    "spearman_rho": float(rho),
                    "p_value_two_sided": float(p_value),
                }
            )
    return pd.DataFrame(records)


def compute_weather_groups(weather_daily: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for controller, group in weather_daily.groupby("controller", sort=False, observed=True):
        rain_group = np.where(group["rain_minutes"].gt(0), "rainy", "dry")
        for label in ("dry", "rainy"):
            selected = group.loc[rain_group == label]
            records.append(
                {
                    "controller": controller,
                    "group_variable": "rain_presence",
                    "group": label,
                    "n": int(len(selected)),
                    "mean_dr_e_pct": float(selected["dr_e_pct"].mean()),
                    "mean_window_open_fraction": float(selected["window_open_fraction"].mean()),
                    "mean_ss_pct": float(selected["ss_pct"].mean()),
                }
            )
    pv_case = weather_daily.loc[weather_daily["controller"].eq("MPC-PV")].copy()
    median_pv = float(pv_case["pv_kwh"].median())
    for label, selected in (
        ("low", pv_case.loc[pv_case["pv_kwh"].le(median_pv)]),
        ("high", pv_case.loc[pv_case["pv_kwh"].gt(median_pv)]),
    ):
        records.append(
            {
                "controller": "MPC-PV",
                "group_variable": "pv_median_split",
                "group": label,
                "n": int(len(selected)),
                "mean_dr_e_pct": float(selected["dr_e_pct"].mean()),
                "mean_window_open_fraction": float(selected["window_open_fraction"].mean()),
                "mean_ss_pct": float(selected["ss_pct"].mean()),
                "pv_median_kwh": median_pv,
            }
        )
    return pd.DataFrame(records)


def load_observed_pooled(path: Path) -> pd.DataFrame:
    pooled = pd.read_csv(path)
    required = ["case", "hvac_kwh", "grid_kwh", "self_kwh", "pv_kwh", "export_kwh", "sc_pct", "ss_pct"]
    _require_columns(pooled, required, "observed pooled metrics")
    selected = pooled.loc[pooled["case"].isin(OBSERVED_CASE_LABELS), required].copy()
    if set(selected["case"]) != set(OBSERVED_CASE_LABELS):
        raise ValueError("pooled energy-flow data is missing a controller")
    return selected


def build_energy_flows(pooled: pd.DataFrame) -> pd.DataFrame:
    required = ["case", "hvac_kwh", "grid_kwh", "self_kwh", "pv_kwh", "export_kwh", "sc_pct", "ss_pct"]
    _require_columns(pooled, required, "pooled energy flows")
    out = pooled.loc[:, required].copy()
    out["controller"] = out["case"].map(OBSERVED_CASE_LABELS).fillna(out["case"])
    out["hvac_closure_error_kwh"] = out["hvac_kwh"] - out["grid_kwh"] - out["self_kwh"]
    out["pv_closure_error_kwh"] = out["pv_kwh"] - out["self_kwh"] - out["export_kwh"]
    if out[["hvac_closure_error_kwh", "pv_closure_error_kwh"]].abs().to_numpy(float).max() > 1e-8:
        raise ValueError("pooled HVAC or PV energy-flow balance does not close")
    order = {label: index for index, label in enumerate(("RBC-AC", "RBC-MM", "MPC-CA", "MPC-PV"))}
    out["_order"] = out["controller"].map(order).fillna(len(order))
    return out.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def load_lstm64_forecast_errors(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    _require_columns(metrics, ["split", "model", "seed", "target", "metric", "value", "eligible_count", "coverage"], "forecast metrics")
    selected = metrics.loc[
        metrics["model"].eq("univariate_lstm64")
        & metrics["seed"].astype(str).eq("ensemble")
        & metrics["split"].isin(("test", "validation"))
        & (
            metrics["metric"].eq("mae")
            | (metrics["target"].eq("wind_direction") & metrics["metric"].eq("circular_mae"))
        )
    ].copy()
    targets = {"temperature", "relative_humidity", "solar", "wind_speed", "wind_direction"}
    if set(selected["target"]) != targets or len(selected) != len(targets) * 2:
        raise ValueError("LSTM64 forecast-error table is incomplete")
    selected["units"] = selected["target"].map(
        {
            "temperature": "degC",
            "relative_humidity": "percentage points",
            "solar": "W m-2",
            "wind_speed": "m s-1",
            "wind_direction": "degrees",
        }
    )
    selected["rain_forecast_scope"] = "current observed rain; not forecast"
    return selected.sort_values(["split", "target"], kind="stable").reset_index(drop=True)


def render_lstm64_daily_profiles(
    minute: pd.DataFrame, selected_dates: pd.DataFrame, output_dir: Path,
    *, stem: str = "discussion_lstm64_daily_profiles",
) -> FigureResult:
    _configure_plot_style()
    figure, axes = plt.subplots(4, 2, figsize=(13.2, 10.4), sharex="col", constrained_layout=True)
    panel_letters = "abcdefgh"
    for column, selection in selected_dates.sort_values("controller_objective").reset_index(drop=True).iterrows():
        objective = str(selection["controller_objective"])
        date = str(selection["date"])
        subset = minute.loc[
            minute["controller_objective"].eq(objective) & minute["date"].eq(date)
        ].copy()
        if set(subset["forecast_source"]) != {"observed", "lstm64"}:
            raise ValueError(f"daily profile is incomplete for {objective} on {date}")
        for source in ("observed", "lstm64"):
            color = FORECAST_COLORS[(objective, source)]
            group = subset.loc[subset["forecast_source"].eq(source)].sort_values("ts")
            style = SOURCE_STYLES[source]
            label = SOURCE_LABELS[source]
            axes[0, column].plot(group["ts"], group["T_mean"], color=color, linestyle=style, linewidth=1.8, label=f"{OBJECTIVE_LABELS[objective]} - {label}")
            axes[1, column].step(group["ts"], 1 - group["z"], where="post", color=color, linestyle=style, linewidth=1.5, label=label)
            axes[2, column].plot(group["ts"], group["pmv_mean"], color=color, linestyle=style, linewidth=1.6, label=label)
            axes[3, column].plot(group["ts"], group["hvac_kw"], color=color, linestyle=style, linewidth=1.6, label="_nolegend_")
            axes[3, column].plot(group["ts"], group["grid_kw"], color=GRID_COLORS[source], linestyle=style, linewidth=1.15, label=f"Grid, {label}")
            if objective == "onsite_pv":
                axes[3, column].plot(group["ts"], group["self_kw"], color=SELF_COLORS[source], linestyle=style, linewidth=1.15, label=f"Self-used PV, {label}")
        common = subset.loc[subset["forecast_source"].eq("observed")].sort_values("ts")
        axes[3, column].plot(common["ts"], common["pv_kw"], color="#444444", linestyle=":", linewidth=1.4, label="Available PV")
        axes[0, column].axhline(30.0, color="#555555", linestyle=":", linewidth=1.0)
        axes[1, column].set_ylim(-0.08, 1.08)
        axes[1, column].set_yticks((0, 1), labels=("Closed", "Open"))
        axes[2, column].axhspan(-0.5, 0.5, color="#56B4E9", alpha=0.10)
        axes[2, column].axhline(-1.0, color="#777777", linestyle=":", linewidth=0.8)
        axes[2, column].axhline(1.0, color="#777777", linestyle=":", linewidth=0.8)
        axes[0, column].set_title(pd.Timestamp(date).strftime("%Y-%m-%d"))
        axes[3, column].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[3, column].set_xlabel("Local time")

    focus_temperature_axes(list(axes[0, :]), label="forecast daily temperature " + stem)
    row_labels = ("Mean zone\ntemperature (°C)", "Window\nstatus", "Mean PMV", "Power (kW)")
    for row in range(4):
        for column in range(2):
            axes[row, column].grid(True, axis="y")
            axes[row, column].text(0.01, 0.96, f"({panel_letters[row * 2 + column]})", transform=axes[row, column].transAxes, ha="left", va="top", fontsize=14.0, fontweight="bold")
        axes[row, 0].set_ylabel(row_labels[row])
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for axis in (axes[0, 0], axes[0, 1], axes[3, 0], axes[3, 1]):
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            compact_label = "HVAC power" if label.startswith("HVAC") else label
            if compact_label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(compact_label)
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=12.5,
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.84))
    return _save_figure(figure, output_dir, stem)


def _paired_metric_axis(
    axis: plt.Axes,
    pooled: pd.DataFrame,
    metrics: Sequence[tuple[str, str, str]],
    ylabel: str,
) -> None:
    x = np.arange(len(metrics), dtype=float)
    for index, (objective, column, label) in enumerate(metrics):
        values = []
        for source in ("observed", "lstm64"):
            row = pooled.loc[
                pooled["controller_objective"].eq(objective)
                & pooled["forecast_source"].eq(source)
            ]
            if len(row) != 1:
                raise ValueError(f"missing pooled {objective}/{source} row")
            values.append(float(row.iloc[0][column]))
        axis.plot([x[index] - 0.12, x[index] + 0.12], values, color="#888888", linewidth=1.0, zorder=1)
        axis.scatter(x[index] - 0.12, values[0], s=38, color=OBJECTIVE_COLORS[objective], marker="o", zorder=2)
        axis.scatter(x[index] + 0.12, values[1], s=42, facecolors="white", edgecolors=OBJECTIVE_COLORS[objective], marker="s", linewidths=1.4, zorder=2)
        axis.text(x[index] - 0.12, values[0], f" {values[0]:.1f}", va="bottom", ha="center", fontsize=12.0)
        axis.text(x[index] + 0.12, values[1], f" {values[1]:.1f}", va="top", ha="center", fontsize=12.0)
    axis.set_xticks(x, [label for _, _, label in metrics])
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y")


def render_lstm64_aggregate(daily: pd.DataFrame, output_dir: Path) -> FigureResult:
    _configure_plot_style()
    pooled = pool_lstm64_metrics(daily)
    figure, axes = plt.subplots(3, 1, figsize=(11.2, 9.0), constrained_layout=True)
    _paired_metric_axis(
        axes[0],
        pooled,
        (
            ("no_pv", "hvac_kwh", "MPC\nHVAC"),
            ("no_pv", "grid_kwh", "MPC\ngrid"),
            ("onsite_pv", "hvac_kwh", "MPC-PV\nHVAC"),
            ("onsite_pv", "grid_kwh", "MPC-PV\ngrid"),
        ),
        "Pooled energy (kWh)",
    )
    _paired_metric_axis(
        axes[1],
        pooled,
        tuple(
            (objective, column, f"{OBJECTIVE_LABELS[objective]}\n{label}")
            for objective in ("no_pv", "onsite_pv")
            for column, label in (
                ("dr_e_pct", "full day"),
                ("dr_p_morning_pct", "morning"),
                ("dr_p_evening_pct", "evening"),
            )
        ),
        "Demand reduction vs RBC-AC (%)",
    )
    _paired_metric_axis(
        axes[2],
        pooled,
        (("onsite_pv", "sc_pct", "Self-consumption"), ("onsite_pv", "ss_pct", "Self-sufficiency")),
        "PV utilization (%)",
    )
    axes[0].scatter([], [], color="#555555", marker="o", label="Perfect-forecast")
    axes[0].scatter([], [], facecolors="white", edgecolors="#555555", marker="s", label="LSTM64")
    figure.legend(
        axes[0].get_legend_handles_labels()[0],
        axes[0].get_legend_handles_labels()[1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    for index, axis in enumerate(axes):
        axis.text(0.005, 0.97, f"({chr(ord('a') + index)})", transform=axis.transAxes, ha="left", va="top", fontsize=14.0, fontweight="bold")
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.92))
    return _save_figure(figure, output_dir, "discussion_lstm64_aggregate")


def render_weather_flexibility(
    weather_daily: pd.DataFrame, associations: pd.DataFrame, output_dir: Path
) -> FigureResult:
    _configure_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    panels = (
        ("temperature_dr", "mean_t_out_c", "dr_e_pct", "Mean outdoor temperature (°C)", "Full-day reduction\nvs RBC-AC (%)", ("RBC-MM", "MPC-CA", "MPC-PV")),
        ("rain_window", "rain_minutes", "window_open_fraction", "Rain minutes", "Window-open\nfraction", ("RBC-MM", "MPC-CA", "MPC-PV")),
        ("pv_dr", "pv_kwh", "dr_e_pct", "Available PV energy (kWh)", "MPC-PV reduction\nvs RBC-AC (%)", ("MPC-PV",)),
        ("pv_ss", "pv_kwh", "ss_pct", "Available PV energy (kWh)", "MPC-PV\nself-sufficiency (%)", ("MPC-PV",)),
    )
    markers = {"RBC-MM": "^", "MPC-CA": "o", "MPC-PV": "s"}
    for axis, (panel, x_column, y_column, xlabel, ylabel, controllers) in zip(axes.flat, panels):
        annotation_lines = []
        for controller in controllers:
            group = weather_daily.loc[weather_daily["controller"].eq(controller)].sort_values(x_column)
            color = OBSERVED_COLORS[controller]
            axis.scatter(group[x_column], group[y_column], s=28, alpha=0.78, color=color, marker=markers[controller], label=controller)
            if group[x_column].nunique() > 1:
                coefficient = np.polyfit(group[x_column].to_numpy(float), group[y_column].to_numpy(float), deg=1)
                grid = np.linspace(group[x_column].min(), group[x_column].max(), 100)
                axis.plot(grid, np.polyval(coefficient, grid), color=color, alpha=0.45, linewidth=1.0)
            stat = associations.loc[associations["panel"].eq(panel) & associations["controller"].eq(controller)].iloc[0]
            annotation_lines.append(f"{controller}: ρ={stat['spearman_rho']:.2f}, n={int(stat['n'])}")
        axis.text(0.98, 0.97, "\n".join(annotation_lines), transform=axis.transAxes, ha="right", va="top", fontsize=12.0, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2})
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    for index, axis in enumerate(axes.flat):
        axis.text(0.01, 0.97, f"({chr(ord('a') + index)})", transform=axis.transAxes, ha="left", va="top", fontsize=14.0, fontweight="bold")
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.92))
    return _save_figure(figure, output_dir, "discussion_weather_flexibility")


def render_pv_energy_flow(flows: pd.DataFrame, output_dir: Path) -> FigureResult:
    _configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), constrained_layout=True)
    x = np.arange(len(flows))
    controllers = flows["controller"].tolist()
    colors = [OBSERVED_COLORS.get(controller, "#777777") for controller in controllers]

    axes[0].bar(x, flows["self_kwh"], color="#F0C36E", edgecolor="#5A5A5A", linewidth=0.6, label="Self-consumed PV")
    axes[0].bar(x, flows["grid_kwh"], bottom=flows["self_kwh"], color="#B8C4CE", edgecolor="#5A5A5A", linewidth=0.6, label="Grid supply")
    for index, row in flows.iterrows():
        axes[0].text(index, row["hvac_kwh"] + 12, f"{row['hvac_kwh']:.0f} kWh\nSS {row['ss_pct']:.1f}%", ha="center", va="bottom", fontsize=12.0, color=colors[index])
    axes[0].set_ylabel("Pooled HVAC energy (kWh)")
    axes[0].set_xticks(x, controllers)
    axes[0].grid(True, axis="y")

    axes[1].bar(x, flows["self_kwh"], color="#F0C36E", edgecolor="#5A5A5A", linewidth=0.6, label="Self-consumed PV")
    axes[1].bar(x, flows["export_kwh"], bottom=flows["self_kwh"], color="#E4E4E4", edgecolor="#5A5A5A", linewidth=0.6, label="Exported PV")
    for index, row in flows.iterrows():
        axes[1].text(index, row["pv_kwh"] + 8, f"SC {row['sc_pct']:.1f}%", ha="center", va="bottom", fontsize=12.0, color=colors[index])
    axes[1].set_ylabel("Pooled available PV energy (kWh)")
    axes[1].set_xticks(x, controllers)
    axes[1].grid(True, axis="y")
    for index, axis in enumerate(axes):
        axis.text(0.01, 0.97, f"({chr(ord('a') + index)})", transform=axis.transAxes, ha="left", va="top", fontsize=14.0, fontweight="bold")
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
    figure.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
    return _save_figure(figure, output_dir, "discussion_pv_energy_flow")


def _copy_figure_pair(result: FigureResult, paper_fig_dir: Path) -> tuple[Path, Path]:
    destination = Path(paper_fig_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / result.png_path.name
    pdf = destination / result.pdf_path.name
    shutil.copy2(result.png_path, png)
    shutil.copy2(result.pdf_path, pdf)
    return png, pdf


def reconcile_observed_daily(daily: pd.DataFrame, observed_daily_path: Path) -> dict[str, Any]:
    """Check shared definitions without equating mean-zone and zone-sample PMV."""

    columns = ["hvac_kwh", "grid_kwh", "pv_kwh", "self_kwh", "export_kwh",
               "morning_hvac_kwh", "evening_hvac_kwh", "dr_e_pct",
               "dr_p_morning_pct", "dr_p_evening_pct", "switch_count",
               "window_open_fraction", "mean_temp_c", "mean_pmv"]
    reference = pd.read_csv(observed_daily_path)
    _require_columns(reference, ["date", "case", *columns], "isolated observed daily metrics")
    reference = reference.loc[reference["case"].isin(CASE_BY_KEY[key] for key in FORECAST_CASES if key[1] == "observed")].copy()
    reference["date"] = pd.to_datetime(reference["date"]).dt.strftime("%Y-%m-%d")
    selected = daily.loc[daily["forecast_source"].eq("observed")].copy()
    selected["date"] = pd.to_datetime(selected["date"]).dt.strftime("%Y-%m-%d")
    if reference.duplicated(["date", "case"]).any() or selected.duplicated(["date", "case"]).any():
        raise ValueError("observed reconciliation keys must be unique")
    reference = reference.set_index(["date", "case"]).sort_index()
    selected = selected.set_index(["date", "case"]).sort_index()
    if not reference.index.equals(selected.index):
        raise ValueError("observed discussion and primary tables do not share identical date/case keys")
    errors = {}
    for column in columns:
        actual = selected[column].to_numpy(float)
        expected = reference[column].to_numpy(float)
        # Historical and surrogate mean temperatures may be stored as float32.
        tolerance = 5e-6 if column == "mean_temp_c" else 1e-8
        if not np.allclose(actual, expected, rtol=1e-9, atol=tolerance, equal_nan=True):
            raise ValueError(f"isolated observed reconciliation failed for {column}")
        difference = np.abs(actual - expected)
        errors[column] = float(np.nanmax(difference)) if np.isfinite(difference).any() else None
    return {"passed": True, "rows": len(selected), "max_absolute_errors": errors,
            "comfort_definition_note": "Section 3 mean-zone PMV coverage and discussion zone-sample coverage remain distinct"}


def run_pipeline(
    *,
    timeseries_path: Path = DEFAULT_TIMESERIES_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    observed_context_path: Path = DEFAULT_OBSERVED_CONTEXT_PATH,
    observed_daily_path: Path = DEFAULT_OBSERVED_DAILY_PATH,
    observed_pooled_path: Path = DEFAULT_OBSERVED_POOLED_PATH,
    forecast_metrics_path: Path = DEFAULT_FORECAST_METRICS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    paper_fig_dir: Path = DEFAULT_PAPER_FIG_DIR,
    validate_only: bool = False,
    validate_full: bool = True,
) -> PipelineResult:
    output_dir = Path(output_dir)
    paper_fig_dir = Path(paper_fig_dir)
    minute = load_forecast_cases(timeseries_path, validate_full=validate_full)
    audit = minute.attrs["energy_audit"]
    context = load_observed_context(observed_context_path)
    minute = attach_pmv(attach_observed_context(minute, context))
    minute["date"] = minute["ts"].dt.strftime("%Y-%m-%d")
    dates = sorted(minute["date"].unique())
    ac27 = _load_ac27_reference(observed_daily_path, dates)
    daily = compute_lstm64_daily_metrics(minute, ac27)
    pooled = pool_lstm64_metrics(daily)
    deltas = build_paired_lstm64_deltas(daily)
    selected = select_lstm64_profile_dates(minute)
    selections = dict(zip(selected["controller_objective"], selected["date"]))
    safety = validate_safety_table(validation_path, dates)
    weather_daily = load_weather_daily(observed_daily_path)
    associations = compute_weather_associations(weather_daily)
    groups = compute_weather_groups(weather_daily)
    flows = build_energy_flows(load_observed_pooled(observed_pooled_path))
    forecast_errors = load_lstm64_forecast_errors(forecast_metrics_path)

    # Reconcile the independently recomputed observed trajectories to this
    # experiment's observed tables, rather than to obsolete historical totals.
    observed_reconciliation = reconcile_observed_daily(daily, observed_daily_path)

    if validate_only:
        return PipelineResult(True, len(dates), len(minute), {}, selections)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tables = {
        "lstm64_minute_data": (minute, "lstm64_minute_data.csv.gz"),
        "lstm64_daily_metrics": (daily, "lstm64_daily_metrics.csv"),
        "lstm64_pooled_metrics": (pooled, "lstm64_pooled_metrics.csv"),
        "lstm64_paired_deltas": (deltas, "lstm64_paired_deltas.csv"),
        "lstm64_selected_dates": (selected, "lstm64_selected_dates.csv"),
        "lstm64_forecast_errors": (forecast_errors, "lstm64_forecast_errors.csv"),
        "weather_flexibility_daily": (weather_daily, "weather_flexibility_daily.csv"),
        "weather_flexibility_associations": (associations, "weather_flexibility_associations.csv"),
        "weather_flexibility_groups": (groups, "weather_flexibility_groups.csv"),
        "pv_energy_flows": (flows, "pv_energy_flows.csv"),
    }
    for key, (table, filename) in tables.items():
        destination = output / filename
        table.to_csv(destination, index=False, float_format="%.12g")
        paths[key] = destination

    figures = (
        render_lstm64_daily_profiles(minute, selected, output),
        render_lstm64_aggregate(daily, output),
        render_weather_flexibility(weather_daily, associations, output),
        render_pv_energy_flow(flows, output),
    )
    for result in figures:
        paths[result.png_path.stem + "_png"] = result.png_path
        paths[result.pdf_path.stem + "_pdf"] = result.pdf_path
        paper_png, paper_pdf = _copy_figure_pair(result, paper_fig_dir)
        paths[paper_png.stem + "_paper_png"] = paper_png
        paths[paper_pdf.stem + "_paper_pdf"] = paper_pdf

    manifest = {
        "scope": {
            "forecast_sources": ["observed", "lstm64"],
            "forecast_cases": [CASE_BY_KEY[key] for key in FORECAST_CASES],
            "dates": dates,
            "samples_per_date_case": FULL_SAMPLES_PER_DATE_CASE if validate_full else None,
            "ac27_is_demand_reduction_reference": True,
            "rain_scope": "current observed rain; not forecast",
        },
        "energy": {
            "calculation": "manuscript heat balance",
            "audit": asdict(audit),
            "hvac_flow_max_abs_closure_error_kwh": float(flows["hvac_closure_error_kwh"].abs().max()),
            "pv_flow_max_abs_closure_error_kwh": float(flows["pv_closure_error_kwh"].abs().max()),
        },
        "pmv": {
            "ac_relative_humidity_pct": 65.0,
            "window_open_relative_humidity": "observed outdoor relative humidity",
        },
        "peak_windows": {
            "morning": "09:00--11:30, start-inclusive and end-exclusive",
            "evening": "16:00--18:30, start-inclusive and end-exclusive",
        },
        "selected_profiles": selections,
        "observed_reconciliation": observed_reconciliation,
        "historical_selected_profiles": {"no_pv": "2024-10-08", "onsite_pv": "2024-10-09"},
        "historical_profile_dates_are_not_acceptance_gates": True,
        "safety": safety,
        "sources": {
            "trajectory": {"file": Path(timeseries_path).name, "sha256": _sha256(timeseries_path)},
            "validation": {"file": Path(validation_path).name, "sha256": _sha256(validation_path)},
            "observed_context": {"file": Path(observed_context_path).name, "sha256": _sha256(observed_context_path)},
            "observed_daily": {"file": Path(observed_daily_path).name, "sha256": _sha256(observed_daily_path)},
            "observed_pooled": {"file": Path(observed_pooled_path).name, "sha256": _sha256(observed_pooled_path)},
            "forecast_metrics": {"file": Path(forecast_metrics_path).name, "sha256": _sha256(forecast_metrics_path)},
        },
        "outputs": {key: {"file": path.name, "sha256": _sha256(path)} for key, path in paths.items() if path.is_file() and path.parent == output},
    }
    manifest_path = output / "discussion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"] = manifest_path
    return PipelineResult(True, len(dates), len(minute), paths, selections)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", type=Path, default=DEFAULT_TIMESERIES_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--observed-context", type=Path, default=DEFAULT_OBSERVED_CONTEXT_PATH)
    parser.add_argument("--observed-daily", type=Path, default=DEFAULT_OBSERVED_DAILY_PATH)
    parser.add_argument("--observed-pooled", type=Path, default=DEFAULT_OBSERVED_POOLED_PATH)
    parser.add_argument("--forecast-metrics", type=Path, default=DEFAULT_FORECAST_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_pipeline(
        timeseries_path=arguments.timeseries,
        validation_path=arguments.validation,
        observed_context_path=arguments.observed_context,
        observed_daily_path=arguments.observed_daily,
        observed_pooled_path=arguments.observed_pooled,
        forecast_metrics_path=arguments.forecast_metrics,
        output_dir=arguments.output_dir,
        paper_fig_dir=arguments.paper_fig_dir,
        validate_only=arguments.validate_only,
    )
    print(
        json.dumps(
            {
                "validated": result.validated,
                "date_count": result.date_count,
                "minute_rows": result.minute_rows,
                "selected_dates": result.selected_dates,
                "outputs": {key: str(path) for key, path in result.output_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
