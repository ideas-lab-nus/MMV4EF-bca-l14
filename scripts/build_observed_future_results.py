"""Build observed-future closed-loop result artifacts.

This module starts from a deliberately narrow calculation boundary.  It accepts
only the four approved observed-future comparison cases, reconstructs HVAC
power from the manuscript heat balance, audits the stored power column, and
then replaces that column with the independently reconstructed values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd


APPROVED_CASES = (
    "AC baseline",
    "RBC baseline",
    "MIQP no PV | observed future",
    "MIQP onsite PV | observed future",
)
DR_REFERENCE_CASE = "AC baseline"
DR_COMPARISON_CASES = tuple(
    case for case in APPROVED_CASES if case != DR_REFERENCE_CASE
)
EXPECTED_FORECAST_SOURCE = {
    "AC baseline": "not_applicable",
    "RBC baseline": "not_applicable",
    "MIQP no PV | observed future": "observed",
    "MIQP onsite PV | observed future": "observed",
}

ZONE_COLUMNS = tuple(f"Zone {zone} Temperature" for zone in range(1, 6))
FCU_COLUMNS = tuple(f"fcu{unit:02d}_supply" for unit in range(1, 6))
PFCU_COLUMNS = tuple(f"pfc{unit:02d}_supply" for unit in range(1, 3))

SPECIFIC_HEAT_AIR = 1005.0
AIR_DENSITY = 1.225
CFM_TO_M3S = 0.0283168 / 60.0
COP_FCU = 5.0
COP_PFCU = 5.0
KAPPA_F_KW_PER_K = (
    SPECIFIC_HEAT_AIR
    * AIR_DENSITY
    * (800.0 * CFM_TO_M3S)
    / 1000.0
    / COP_FCU
)
KAPPA_P_KW_PER_K = (
    SPECIFIC_HEAT_AIR
    * AIR_DENSITY
    * (1000.0 * CFM_TO_M3S)
    / 1000.0
    / COP_PFCU
)

POWER_ACCOUNTING_ATOL_KW = 1e-9
FULL_DATE_COUNT = 22
FULL_SAMPLES_PER_CASE_DATE = 690
STEP_HOURS = 1.0 / 60.0
AC_RELATIVE_HUMIDITY_PCT = 65.0
PMV_LINEAR_COEFFS = {
    1: (-7.673842, 0.249984, 0.011084),
    2: (-7.539536, 0.244956, 0.011203),
    3: (-7.595545, 0.245508, 0.011841),
    4: (-7.625768, 0.246251, 0.011946),
    5: (-7.599742, 0.246740, 0.011352),
}
MPC_CASES = (
    "MIQP no PV | observed future",
    "MIQP onsite PV | observed future",
)
from plot_styles import focus_temperature_axes

CASE_COLORS = {
    "AC baseline": "#fc8d62",
    "RBC baseline": "#e78ac3",
    "MIQP no PV | observed future": "#8da0cb",
    "MIQP onsite PV | observed future": "#66c2a5",
}
CASE_LINESTYLES = {
    "AC baseline": "-",
    "RBC baseline": "--",
    "MIQP no PV | observed future": "-.",
    "MIQP onsite PV | observed future": ":",
}
CASE_DISPLAY_LABELS = {
    "AC baseline": "RBC-AC",
    "RBC baseline": "RBC-MM",
    "MIQP no PV | observed future": "MPC-CA",
    "MIQP onsite PV | observed future": "MPC-PV",
}
WINDOW_PLOT_OFFSETS = {
    "RBC baseline": -0.035,
    "MIQP no PV | observed future": 0.0,
    "MIQP onsite PV | observed future": 0.035,
}
DAILY_PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")

DEFAULT_SOURCE_ROOT = Path("outputs/simulations/main")
DEFAULT_RESULT_STEM = "future_data_source_comparison_test_val_union_all_full"
DEFAULT_TIMESERIES_PATH = DEFAULT_SOURCE_ROOT / f"{DEFAULT_RESULT_STEM}_timeseries.csv"
DEFAULT_SOURCE_DAILY_SUMMARY_PATH = (
    DEFAULT_SOURCE_ROOT / f"{DEFAULT_RESULT_STEM}_daily_summary.csv"
)
DEFAULT_SOURCE_AGGREGATE_SUMMARY_PATH = (
    DEFAULT_SOURCE_ROOT / f"{DEFAULT_RESULT_STEM}_aggregate_summary.csv"
)
DEFAULT_SOURCE_VALIDATION_PATH = (
    DEFAULT_SOURCE_ROOT / f"{DEFAULT_RESULT_STEM}_validation.csv"
)
DEFAULT_OBSERVED_CONTEXT_PATH = Path("data/private/l14_merged_data_with_rain.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/observed_future_results")
DEFAULT_PAPER_FIG_DIR = Path("figures")


@dataclass(frozen=True)
class EnergyAudit:
    """Summary of stored versus independently reconstructed HVAC power."""

    rows: int
    max_abs_error_kw: float
    mean_abs_error_kw: float
    mismatches_above_tolerance: int
    tolerance_kw: float


@dataclass(frozen=True)
class GlobalPlotLimits:
    """Shared daily-figure limits calculated once from all selected dates."""

    temperature_c: tuple[float, float]
    pmv: tuple[float, float]
    pv_self_kw: tuple[float, float]


@dataclass(frozen=True)
class DailyFigureResult:
    """Paths and structural metadata for one rendered daily figure."""

    png_path: Path
    pdf_path: Path
    date_iso: str
    case_labels: tuple[str, ...]
    display_labels: tuple[str, ...]
    plot_limits: GlobalPlotLimits
    window_offsets: dict[str, float]
    rain_interval_count: int
    panel_labels: tuple[str, ...]


@dataclass(frozen=True)
class ContactSheetResult:
    """Output path and deterministic ordering for the daily contact sheet."""

    output_path: Path
    ordered_dates: tuple[str, ...]
    ordered_input_paths: tuple[Path, ...]
    grid_shape: tuple[int, int]


@dataclass(frozen=True)
class AggregateFigureResult:
    """Paths and transparent point-accounting metadata for the KPI figure."""

    png_path: Path
    pdf_path: Path
    dr_reference_label: str
    dr_panel_case_labels: tuple[str, ...]
    finite_counts: dict[str, int]
    undefined_counts: dict[str, int]
    extrema: dict[str, tuple[float, float]]
    outlier_annotations: tuple[str, ...]
    figure_size_px: tuple[int, int]
    count_label_boxes_px: dict[str, tuple[float, float, float, float]]
    peak_inset_box_px: tuple[float, float, float, float] | None
    outlier_text_boxes_px: dict[str, tuple[float, float, float, float]]
    outlier_arrow_boxes_px: dict[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class PipelineResult:
    """High-level result returned by validation-only and production runs."""

    validated: bool
    row_count: int
    date_count: int
    output_paths: dict[str, Path]
    reconciliation: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _require_finite(
    frame: pd.DataFrame, columns: tuple[str, ...] | list[str], *, label: str
) -> None:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {label}")


def recompute_hvac_power_kw(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct HVAC electrical power from the manuscript heat balance.

    AC power is based only on the five zone-to-FCU supply temperature
    differences.  Window-open power is based only on the two
    outdoor-to-PFCU supply temperature differences.  Negative temperature
    differences contribute zero load.
    """

    required = ["z", "T_out", *ZONE_COLUMNS, *FCU_COLUMNS, *PFCU_COLUMNS]
    _require_columns(frame, required)

    zones = frame.loc[:, ZONE_COLUMNS].to_numpy(dtype=float)
    fcu = frame.loc[:, FCU_COLUMNS].to_numpy(dtype=float)
    pfc = frame.loc[:, PFCU_COLUMNS].to_numpy(dtype=float)
    outdoor = frame.loc[:, ["T_out"]].to_numpy(dtype=float)

    ac_kw = KAPPA_F_KW_PER_K * np.maximum(zones - fcu, 0.0).sum(axis=1)
    nv_kw = KAPPA_P_KW_PER_K * np.maximum(outdoor - pfc, 0.0).sum(axis=1)
    ac_mode = frame["z"].round().astype(int).eq(1).to_numpy()
    values = np.where(ac_mode, ac_kw, nv_kw)
    return pd.Series(values, index=frame.index, name="hvac_kw_recomputed")


def audit_hvac_power(
    frame: pd.DataFrame, atol_kw: float = POWER_ACCOUNTING_ATOL_KW
) -> EnergyAudit:
    """Audit stored HVAC power against the independent heat-balance series."""

    if atol_kw < 0:
        raise ValueError("heat-balance audit tolerance must be nonnegative")
    _require_columns(frame, ["hvac_kw"])
    if frame.empty:
        raise ValueError("heat-balance audit requires at least one row")

    stored = pd.to_numeric(frame["hvac_kw"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(stored).all():
        raise ValueError("heat-balance audit found non-finite stored HVAC power")
    recomputed = recompute_hvac_power_kw(frame).to_numpy(dtype=float)
    if not np.isfinite(recomputed).all():
        raise ValueError("heat-balance audit produced non-finite HVAC power")

    absolute_error = np.abs(stored - recomputed)
    mismatch_count = int(np.count_nonzero(absolute_error > atol_kw))
    result = EnergyAudit(
        rows=len(frame),
        max_abs_error_kw=float(absolute_error.max()),
        mean_abs_error_kw=float(absolute_error.mean()),
        mismatches_above_tolerance=mismatch_count,
        tolerance_kw=float(atol_kw),
    )
    if mismatch_count:
        raise ValueError(
            "heat-balance audit failed: "
            f"{mismatch_count} of {len(frame)} rows exceed {atol_kw:g} kW "
            f"(maximum error {result.max_abs_error_kw:.12g} kW)"
        )
    return result


def _validate_numeric_inputs(frame: pd.DataFrame) -> None:
    core_columns = [
        "z",
        "T_out",
        "hvac_kw",
        "pv_kw",
        "grid_kw",
        "export_kw",
        "self_kw",
        *ZONE_COLUMNS,
    ]
    numeric_columns = [*core_columns, *FCU_COLUMNS, *PFCU_COLUMNS]
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    _require_finite(frame, core_columns, label="required numeric fields")

    z_values = frame["z"].to_numpy(dtype=float)
    if not np.isin(z_values, (0.0, 1.0)).all():
        raise ValueError("z must contain only binary AC/window-open states")

    ac_rows = frame["z"].eq(1)
    nv_rows = frame["z"].eq(0)
    if ac_rows.any():
        _require_finite(frame.loc[ac_rows], list(FCU_COLUMNS), label="active FCU supply")
    if nv_rows.any():
        _require_finite(
            frame.loc[nv_rows], list(PFCU_COLUMNS), label="active PFCU supply"
        )

    power_columns = ["hvac_kw", "pv_kw", "grid_kw", "export_kw", "self_kw"]
    if (frame.loc[:, power_columns].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("power fields must be nonnegative")


def _validate_pv_accounting(frame: pd.DataFrame) -> None:
    hvac = frame["hvac_kw"].to_numpy(dtype=float)
    pv = frame["pv_kw"].to_numpy(dtype=float)
    expected = {
        "grid_kw": np.maximum(hvac - pv, 0.0),
        "export_kw": np.maximum(pv - hvac, 0.0),
        "self_kw": np.minimum(hvac, pv),
    }
    failures = [
        column
        for column, values in expected.items()
        if not np.allclose(
            frame[column].to_numpy(dtype=float),
            values,
            rtol=0.0,
            atol=POWER_ACCOUNTING_ATOL_KW,
        )
    ]
    if failures:
        raise ValueError(
            "grid/export/self accounting failed for: " + ", ".join(failures)
        )


def _validate_identical_pv_availability(frame: pd.DataFrame) -> None:
    pv_spread = frame.groupby("ts", sort=False)["pv_kw"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (pv_spread > POWER_ACCOUNTING_ATOL_KW).any():
        raise ValueError("cases do not have identical PV availability by timestamp")


def _validate_full_shape(frame: pd.DataFrame) -> None:
    normalized_dates = frame["ts"].dt.normalize()
    if normalized_dates.nunique() != FULL_DATE_COUNT:
        raise ValueError(f"expected exactly {FULL_DATE_COUNT} dates")

    counts = frame.assign(_date=normalized_dates).groupby(
        ["_date", "case"], observed=True
    )["ts"].size()
    expected_group_count = FULL_DATE_COUNT * len(APPROVED_CASES)
    if len(counts) != expected_group_count or not counts.eq(
        FULL_SAMPLES_PER_CASE_DATE
    ).all():
        raise ValueError(
            "expected exactly 690 rows for every approved date/case combination"
        )

    for (_, _), group in frame.assign(_date=normalized_dates).groupby(
        ["_date", "case"], observed=True, sort=False
    ):
        timestamps = group["ts"].sort_values().reset_index(drop=True)
        if timestamps.iloc[0].time().isoformat() != "07:30:00":
            raise ValueError("each full date/case must start at 07:30")
        if timestamps.iloc[-1].time().isoformat() != "18:59:00":
            raise ValueError("each full date/case must end at 18:59")
        differences = timestamps.diff().dropna()
        if not differences.eq(pd.Timedelta(minutes=1)).all():
            raise ValueError("full date/case timestamps must be one minute apart")


def load_observed_cases(path: Path, *, validate_full: bool = True) -> pd.DataFrame:
    """Load, audit, and return only the four approved observed-future cases.

    The returned ``hvac_kw`` column is never the source column: after the audit
    succeeds, it is overwritten with the independently reconstructed
    heat-balance values.  The audit summary is retained in ``DataFrame.attrs``.
    """

    source_path = Path(path)
    frame = pd.read_csv(source_path)
    required = [
        "ts",
        "case",
        "forecast_source",
        "z",
        "T_out",
        "hvac_kw",
        "pv_kw",
        "grid_kw",
        "export_kw",
        "self_kw",
        *ZONE_COLUMNS,
        *FCU_COLUMNS,
        *PFCU_COLUMNS,
    ]
    _require_columns(frame, required)

    selected = frame.loc[frame["case"].isin(APPROVED_CASES)].copy()
    if selected.empty:
        raise ValueError("no approved observed-future cases found")
    present_cases = set(selected["case"])
    missing_cases = set(APPROVED_CASES) - present_cases
    if missing_cases:
        raise ValueError(
            "missing approved cases: " + ", ".join(sorted(missing_cases))
        )

    expected_sources = selected["case"].map(EXPECTED_FORECAST_SOURCE)
    source_matches = selected["forecast_source"].eq(expected_sources)
    if not source_matches.all():
        contradictions = (
            selected.loc[~source_matches, ["case", "forecast_source"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            "case/forecast-source metadata contradiction: "
            f"{contradictions}"
        )

    selected["ts"] = pd.to_datetime(selected["ts"], errors="raise")
    if selected["ts"].isna().any():
        raise ValueError("timestamps must not be missing")
    if selected.duplicated(["ts", "case"]).any():
        raise ValueError("duplicate timestamp/case keys")

    _validate_numeric_inputs(selected)
    audit = audit_hvac_power(selected)
    selected.loc[:, "hvac_kw"] = recompute_hvac_power_kw(selected)
    _validate_pv_accounting(selected)
    _validate_identical_pv_availability(selected)
    if validate_full:
        _validate_full_shape(selected)

    selected = selected.sort_values(["ts", "case"], kind="stable").reset_index(
        drop=True
    )
    selected.attrs["energy_audit"] = audit
    return selected


def load_observed_context(path: Path) -> pd.DataFrame:
    """Load timestamped observed window humidity and raw rain status.

    The source uses month-first ``M/D/YYYY H:MM`` strings.  Parsing with an
    explicit format prevents ambiguous dates such as 7/8 from silently being
    interpreted as 8 July in one environment and 7 August in another.
    """

    source_path = Path(path)
    context = pd.read_csv(
        source_path,
        usecols=["date", "OutdoorHumidityWindow", "rain_status"],
    )
    context["ts"] = pd.to_datetime(
        context["date"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M",
        errors="raise",
        exact=True,
    )
    if context["ts"].dt.tz is not None:
        raise ValueError("observed context timestamps must be timezone-naive local time")
    context.loc[:, ["OutdoorHumidityWindow", "rain_status"]] = context.loc[
        :, ["OutdoorHumidityWindow", "rain_status"]
    ].apply(pd.to_numeric, errors="coerce")
    _require_finite(
        context,
        ["OutdoorHumidityWindow", "rain_status"],
        label="observed humidity/rain context",
    )
    if context.duplicated("ts").any():
        raise ValueError("observed context contains duplicate timestamps")
    return context.loc[
        :, ["ts", "OutdoorHumidityWindow", "rain_status"]
    ].sort_values("ts", kind="stable").reset_index(drop=True)


def attach_observed_context(
    frame: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    """Attach exact observed humidity and raw rain to every case trajectory."""

    _require_columns(frame, ["ts"])
    _require_columns(context, ["ts", "OutdoorHumidityWindow", "rain_status"])
    source_attrs = dict(frame.attrs)
    selected = frame.copy()
    observed = context.copy()
    selected["ts"] = pd.to_datetime(selected["ts"], errors="raise")
    observed["ts"] = pd.to_datetime(observed["ts"], errors="raise")
    if selected["ts"].dt.tz is not None or observed["ts"].dt.tz is not None:
        raise ValueError("context join requires timezone-naive local timestamps")
    if observed.duplicated("ts").any():
        raise ValueError("observed context contains duplicate timestamps")

    # The context file is authoritative.  In particular, do not preserve the
    # controller's rain_force_ac signal or an input rain_status surrogate.
    selected = selected.drop(
        columns=["OutdoorHumidityWindow", "rain_status"], errors="ignore"
    )
    joined = selected.merge(
        observed.loc[:, ["ts", "OutdoorHumidityWindow", "rain_status"]],
        on="ts",
        how="left",
        validate="many_to_one",
        indicator="_context_join",
        sort=False,
    )
    missing = joined["_context_join"].ne("both")
    if missing.any():
        examples = joined.loc[missing, "ts"].drop_duplicates().head(5).tolist()
        raise ValueError(
            "observed context exact timestamp join failed for "
            f"{joined.loc[missing, 'ts'].nunique()} timestamps; examples={examples}"
        )
    joined = joined.drop(columns="_context_join")
    _require_finite(
        joined,
        ["OutdoorHumidityWindow", "rain_status"],
        label="joined observed humidity/rain context",
    )
    joined.attrs.update(source_attrs)
    return joined


def attach_pmv(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct five zone PMV values using mode-specific humidity."""

    required = ["z", "OutdoorHumidityWindow", *ZONE_COLUMNS]
    _require_columns(frame, required)
    out = frame.copy()
    _require_finite(out, ["z", "OutdoorHumidityWindow", *ZONE_COLUMNS], label="PMV inputs")
    z = pd.to_numeric(out["z"], errors="coerce")
    if not z.isin((0.0, 1.0)).all():
        raise ValueError("PMV reconstruction requires binary z states")
    observed_rh = pd.to_numeric(out["OutdoorHumidityWindow"], errors="coerce")
    out["pmv_rh_pct"] = np.where(
        z.eq(1), AC_RELATIVE_HUMIDITY_PCT, observed_rh
    )
    for zone, (intercept, beta_t, beta_rh) in PMV_LINEAR_COEFFS.items():
        out[f"pmv_zone_{zone}"] = (
            intercept
            + beta_t * pd.to_numeric(out[f"Zone {zone} Temperature"], errors="coerce")
            + beta_rh * out["pmv_rh_pct"]
        )
    pmv_columns = [f"pmv_zone_{zone}" for zone in PMV_LINEAR_COEFFS]
    out["pmv_mean"] = out.loc[:, pmv_columns].mean(axis=1)
    return out


def _safe_percent(numerator: float, denominator: float) -> float:
    """Return ``100*numerator/denominator`` or NaN for an undefined ratio."""

    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def _validate_paired_dr_dates(daily: pd.DataFrame) -> None:
    """Require unique rows and exact AC27/comparison date pairing."""

    _require_columns(daily, ["date", "case"])
    if daily.duplicated(["date", "case"]).any():
        raise ValueError("daily metrics must have unique date/case rows")
    reference_dates = set(
        daily.loc[daily["case"].eq(DR_REFERENCE_CASE), "date"]
    )
    for case in DR_COMPARISON_CASES:
        case_dates = set(daily.loc[daily["case"].eq(case), "date"])
        if case_dates and case_dates != reference_dates:
            missing_reference = sorted(case_dates - reference_dates)
            missing_case = sorted(reference_dates - case_dates)
            raise ValueError(
                "paired AC27/comparison date sets are required before demand-"
                f"reduction calculation for {case}; dates without AC27="
                f"{missing_reference}, dates without case={missing_case}"
            )


def _validate_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "ts",
        "case",
        "z",
        "T_out",
        "T_mean",
        "pmv_mean",
        "hvac_kw",
        "pv_kw",
        "grid_kw",
        "export_kw",
        "self_kw",
        "rain_status",
    ]
    _require_columns(frame, required)
    if frame.empty:
        raise ValueError("daily metrics require at least one row")
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="raise")
    if out["ts"].dt.tz is not None:
        raise ValueError("daily metrics require timezone-naive local timestamps")
    if out.duplicated(["ts", "case"]).any():
        raise ValueError("daily metrics found duplicate timestamp/case keys")
    numeric = [column for column in required if column not in ("ts", "case")]
    out.loc[:, numeric] = out.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    _require_finite(out, numeric, label="daily metric inputs")
    if not out["z"].isin((0.0, 1.0)).all():
        raise ValueError("daily metrics require binary z states")
    if (out.loc[:, ["hvac_kw", "pv_kw", "grid_kw", "export_kw", "self_kw"]] < 0).any().any():
        raise ValueError("daily metric power values must be nonnegative")
    out["date"] = out["ts"].dt.normalize()
    return out


def compute_daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute per-date, per-case values and paired AC27 flexibility KPIs."""

    source = _validate_metric_frame(frame)
    records: list[dict[str, object]] = []
    for (date_value, case), group in source.groupby(
        ["date", "case"], sort=True, observed=True
    ):
        group = group.sort_values("ts", kind="stable")
        minute_of_day = group["ts"].dt.hour * 60 + group["ts"].dt.minute
        morning = minute_of_day.ge(time(9, 0).hour * 60) & minute_of_day.lt(
            time(11, 30).hour * 60 + time(11, 30).minute
        )
        evening = minute_of_day.ge(time(16, 0).hour * 60) & minute_of_day.lt(
            time(18, 30).hour * 60 + time(18, 30).minute
        )
        energy = {
            f"{stem}_kwh": float(group[column].sum() * STEP_HOURS)
            for stem, column in (
                ("hvac", "hvac_kw"),
                ("pv", "pv_kw"),
                ("grid", "grid_kw"),
                ("export", "export_kw"),
                ("self", "self_kw"),
            )
        }
        z_values = group["z"].round().astype(int).to_numpy()
        window_open = 1 - z_values
        pmv_abs = group["pmv_mean"].abs()
        record: dict[str, object] = {
            "date": pd.Timestamp(date_value),
            "case": str(case),
            "sample_count": int(len(group)),
            **energy,
            "sc_pct": _safe_percent(energy["self_kwh"], energy["pv_kwh"]),
            "ss_pct": _safe_percent(energy["self_kwh"], energy["hvac_kwh"]),
            "mean_temp_c": float(group["T_mean"].mean()),
            "mean_t_out_c": float(group["T_out"].mean()),
            "mean_pmv": float(group["pmv_mean"].mean()),
            "pmv_within_05_pct": 100.0 * float(pmv_abs.le(0.5).mean()),
            "pmv_within_10_pct": 100.0 * float(pmv_abs.le(1.0).mean()),
            "window_open_fraction": float(window_open.mean()),
            "window_open_minutes": float(window_open.sum() * STEP_HOURS * 60.0),
            "switch_count": int(np.count_nonzero(z_values[1:] != z_values[:-1])),
            "rain_minutes": int(group["rain_status"].ge(0.5).sum()),
            "morning_samples": int(morning.sum()),
            "evening_samples": int(evening.sum()),
            "morning_hvac_kwh": float(group.loc[morning, "hvac_kw"].sum() * STEP_HOURS),
            "evening_hvac_kwh": float(group.loc[evening, "hvac_kw"].sum() * STEP_HOURS),
        }
        records.append(record)

    daily = pd.DataFrame.from_records(records)
    _validate_paired_dr_dates(daily)
    for column in ("dr_e_pct", "dr_p_morning_pct", "dr_p_evening_pct"):
        daily[column] = np.nan

    reference = daily.loc[daily["case"].eq(DR_REFERENCE_CASE)].set_index("date")
    for index in daily.index[daily["case"].isin(APPROVED_CASES)]:
        date_value = daily.at[index, "date"]
        if date_value not in reference.index:
            continue
        ac27 = reference.loc[date_value]
        daily.at[index, "dr_e_pct"] = _safe_percent(
            ac27["hvac_kwh"] - daily.at[index, "hvac_kwh"],
            ac27["hvac_kwh"],
        )
        daily.at[index, "dr_p_morning_pct"] = _safe_percent(
            ac27["morning_hvac_kwh"] - daily.at[index, "morning_hvac_kwh"],
            ac27["morning_hvac_kwh"],
        )
        daily.at[index, "dr_p_evening_pct"] = _safe_percent(
            ac27["evening_hvac_kwh"] - daily.at[index, "evening_hvac_kwh"],
            ac27["evening_hvac_kwh"],
        )

    case_order = {case: index for index, case in enumerate(APPROVED_CASES)}
    daily["_case_order"] = daily["case"].map(case_order).fillna(len(case_order))
    return daily.sort_values(["date", "_case_order", "case"], kind="stable").drop(
        columns="_case_order"
    ).reset_index(drop=True)


def _weighted_daily_mean(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(group["sample_count"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def compute_pooled_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Pool energy first, then calculate AC27-referenced ratios."""

    required = [
        "date",
        "case",
        "sample_count",
        "hvac_kwh",
        "pv_kwh",
        "grid_kwh",
        "export_kwh",
        "self_kwh",
        "morning_hvac_kwh",
        "evening_hvac_kwh",
        "morning_samples",
        "evening_samples",
        "mean_temp_c",
        "mean_t_out_c",
        "mean_pmv",
        "pmv_within_05_pct",
        "pmv_within_10_pct",
        "window_open_fraction",
        "window_open_minutes",
        "switch_count",
        "rain_minutes",
    ]
    _require_columns(daily, required)
    if daily.empty:
        raise ValueError("pooled metrics require at least one daily row")
    _validate_paired_dr_dates(daily)

    energy_columns = [
        "hvac_kwh",
        "pv_kwh",
        "grid_kwh",
        "export_kwh",
        "self_kwh",
        "morning_hvac_kwh",
        "evening_hvac_kwh",
    ]
    records: list[dict[str, object]] = []
    for case, group in daily.groupby("case", sort=False, observed=True):
        sums = {column: float(group[column].sum()) for column in energy_columns}
        record: dict[str, object] = {
            "case": str(case),
            "date_count": int(group["date"].nunique()),
            "sample_count": int(group["sample_count"].sum()),
            **sums,
            "sc_pct": _safe_percent(sums["self_kwh"], sums["pv_kwh"]),
            "ss_pct": _safe_percent(sums["self_kwh"], sums["hvac_kwh"]),
            "mean_temp_c": _weighted_daily_mean(group, "mean_temp_c"),
            "mean_t_out_c": _weighted_daily_mean(group, "mean_t_out_c"),
            "mean_pmv": _weighted_daily_mean(group, "mean_pmv"),
            "pmv_within_05_pct": _weighted_daily_mean(group, "pmv_within_05_pct"),
            "pmv_within_10_pct": _weighted_daily_mean(group, "pmv_within_10_pct"),
            "window_open_fraction": _weighted_daily_mean(group, "window_open_fraction"),
            "window_open_minutes": float(group["window_open_minutes"].sum()),
            "switch_count": int(group["switch_count"].sum()),
            "rain_minutes": int(group["rain_minutes"].sum()),
            "morning_samples": int(group["morning_samples"].sum()),
            "evening_samples": int(group["evening_samples"].sum()),
            "dr_e_pct": float("nan"),
            "dr_p_morning_pct": float("nan"),
            "dr_p_evening_pct": float("nan"),
        }
        records.append(record)

    pooled = pd.DataFrame.from_records(records)
    reference_rows = pooled.loc[pooled["case"].eq(DR_REFERENCE_CASE)]
    if not reference_rows.empty:
        ac27 = reference_rows.iloc[0]
        for index in pooled.index[pooled["case"].isin(APPROVED_CASES)]:
            pooled.at[index, "dr_e_pct"] = _safe_percent(
                ac27["hvac_kwh"] - pooled.at[index, "hvac_kwh"],
                ac27["hvac_kwh"],
            )
            pooled.at[index, "dr_p_morning_pct"] = _safe_percent(
                ac27["morning_hvac_kwh"]
                - pooled.at[index, "morning_hvac_kwh"],
                ac27["morning_hvac_kwh"],
            )
            pooled.at[index, "dr_p_evening_pct"] = _safe_percent(
                ac27["evening_hvac_kwh"]
                - pooled.at[index, "evening_hvac_kwh"],
                ac27["evening_hvac_kwh"],
            )

    case_order = {case: index for index, case in enumerate(APPROVED_CASES)}
    pooled["_case_order"] = pooled["case"].map(case_order).fillna(len(case_order))
    return pooled.sort_values(["_case_order", "case"], kind="stable").drop(
        columns="_case_order"
    ).reset_index(drop=True)


def _stable_rank(series: pd.Series, *, ascending: bool) -> pd.Series:
    return series.rank(method="first", ascending=ascending, na_option="keep").astype(
        "Int64"
    )


def build_selection_aids(daily: pd.DataFrame) -> pd.DataFrame:
    """Build transparent all-date descriptors for representative-date choice."""

    required = [
        "date",
        "case",
        "dr_e_pct",
        "sc_pct",
        "pv_kwh",
        "rain_minutes",
        "mean_t_out_c",
        "switch_count",
    ]
    _require_columns(daily, required)
    if daily.empty:
        raise ValueError("selection aids require daily metrics")
    _validate_paired_dr_dates(daily)

    selected = daily.loc[daily["case"].isin(MPC_CASES)].copy()
    counts = selected.groupby("date", observed=True)["case"].nunique()
    if counts.empty or not counts.eq(len(MPC_CASES)).all():
        raise ValueError("selection aids require both observed-future MPC cases per date")
    no_pv = selected.loc[selected["case"].eq(MPC_CASES[0])].set_index("date")
    pv_aware = selected.loc[selected["case"].eq(MPC_CASES[1])].set_index("date")
    dates = no_pv.index.sort_values()
    pv_aware = pv_aware.reindex(dates)
    no_pv = no_pv.reindex(dates)

    if not np.allclose(
        no_pv["pv_kwh"], pv_aware["pv_kwh"], rtol=0.0, atol=1e-9
    ):
        raise ValueError("selection aids found case-dependent PV availability")
    if not np.allclose(
        no_pv["rain_minutes"], pv_aware["rain_minutes"], rtol=0.0, atol=0.0
    ):
        raise ValueError("selection aids found case-dependent raw rain minutes")

    aids = pd.DataFrame(
        {
            "date": dates,
            "mpc_dr_e_pct": no_pv["dr_e_pct"].to_numpy(dtype=float),
            "mpc_pv_dr_e_pct": pv_aware["dr_e_pct"].to_numpy(dtype=float),
            "mpc_pv_sc_gain_pct_points": (
                pv_aware["sc_pct"] - no_pv["sc_pct"]
            ).to_numpy(dtype=float),
            "pv_available_kwh": no_pv["pv_kwh"].to_numpy(dtype=float),
            "rain_minutes": no_pv["rain_minutes"].to_numpy(dtype=int),
            "mean_t_out_c": no_pv["mean_t_out_c"].to_numpy(dtype=float),
            "mpc_switch_count": no_pv["switch_count"].to_numpy(dtype=int),
            "mpc_pv_switch_count": pv_aware["switch_count"].to_numpy(dtype=int),
        }
    )
    aids["mpc_dr_e_deviation_pp"] = aids["mpc_dr_e_pct"] - aids[
        "mpc_dr_e_pct"
    ].median(skipna=True)
    aids["mpc_pv_dr_e_deviation_pp"] = aids["mpc_pv_dr_e_pct"] - aids[
        "mpc_pv_dr_e_pct"
    ].median(skipna=True)
    aids["mpc_pv_sc_gain_deviation_pp"] = aids[
        "mpc_pv_sc_gain_pct_points"
    ] - aids["mpc_pv_sc_gain_pct_points"].median(skipna=True)
    aids["mpc_dr_e_undefined"] = aids["mpc_dr_e_pct"].isna()
    aids["mpc_pv_dr_e_undefined"] = aids["mpc_pv_dr_e_pct"].isna()

    solar_percentile = aids["pv_available_kwh"].rank(
        method="first", ascending=True, pct=True
    )
    aids["solar_quantile"] = np.select(
        [solar_percentile.le(1.0 / 3.0), solar_percentile.le(2.0 / 3.0)],
        ["low", "middle"],
        default="high",
    )
    aids["rain_group"] = np.where(aids["rain_minutes"].gt(0), "wet", "dry")
    aids["solar_rank_low_to_high"] = _stable_rank(
        aids["pv_available_kwh"], ascending=True
    )
    aids["rain_rank_high_to_low"] = _stable_rank(
        aids["rain_minutes"], ascending=False
    )
    aids["mpc_dr_e_rank_high_to_low"] = _stable_rank(
        aids["mpc_dr_e_pct"], ascending=False
    )
    aids["mpc_pv_dr_e_rank_high_to_low"] = _stable_rank(
        aids["mpc_pv_dr_e_pct"], ascending=False
    )
    aids["mpc_pv_sc_gain_rank_high_to_low"] = _stable_rank(
        aids["mpc_pv_sc_gain_pct_points"], ascending=False
    )
    aids["control_switch_total"] = (
        aids["mpc_switch_count"] + aids["mpc_pv_switch_count"]
    )
    aids["control_variation_rank_high_to_low"] = _stable_rank(
        aids["control_switch_total"], ascending=False
    )
    return aids.sort_values("date", kind="stable").reset_index(drop=True)


def _prepare_plot_frame(
    frame: pd.DataFrame, *, require_single_date: bool
) -> pd.DataFrame:
    """Validate the observed-future plotting boundary and normalize ordering."""

    required = [
        "ts",
        "case",
        "z",
        "T_out",
        "T_mean",
        "pmv_mean",
        "pv_kw",
        "self_kw",
        "rain_status",
    ]
    _require_columns(frame, required)
    if frame.empty:
        raise ValueError("daily plotting requires at least one row")

    source = frame.copy()
    source["ts"] = pd.to_datetime(source["ts"], errors="raise")
    if source["ts"].dt.tz is not None:
        raise ValueError("daily plotting requires timezone-naive local timestamps")
    if source.duplicated(["ts", "case"]).any():
        raise ValueError("daily plotting found duplicate timestamp/case keys")

    present_cases = set(source["case"])
    if present_cases != set(APPROVED_CASES):
        missing = sorted(set(APPROVED_CASES) - present_cases)
        unexpected = sorted(present_cases - set(APPROVED_CASES))
        raise ValueError(
            "daily plotting requires exactly the four approved cases; "
            f"missing={missing}, unexpected={unexpected}"
        )

    numeric = [
        "z",
        "T_out",
        "T_mean",
        "pmv_mean",
        "pv_kw",
        "self_kw",
        "rain_status",
    ]
    source.loc[:, numeric] = source.loc[:, numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(source["rain_status"].to_numpy(dtype=float)).all():
        raise ValueError("daily plotting requires raw joined rain_status values")
    _require_finite(
        source,
        ["z", "T_mean", "pmv_mean", "pv_kw", "self_kw"],
        label="daily plotting inputs",
    )
    if not source["z"].isin((0.0, 1.0)).all():
        raise ValueError("daily plotting requires binary z states")
    if (source.loc[:, ["pv_kw", "self_kw"]] < 0.0).any().any():
        raise ValueError("daily plotting requires nonnegative PV/self-use power")
    if (source["self_kw"] - source["pv_kw"] > POWER_ACCOUNTING_ATOL_KW).any():
        raise ValueError("daily plotting found self-use above available PV power")

    source["_plot_date"] = source["ts"].dt.normalize()
    date_count = int(source["_plot_date"].nunique())
    if require_single_date and date_count != 1:
        raise ValueError("daily figure input must contain exactly one date")

    for date_value, group in source.groupby("_plot_date", sort=True, observed=True):
        if set(group["case"]) != set(APPROVED_CASES):
            raise ValueError(
                f"date {pd.Timestamp(date_value).date()} is missing approved cases"
            )
        reference = (
            group.loc[group["case"].eq(APPROVED_CASES[0]), "ts"]
            .sort_values(kind="stable")
            .reset_index(drop=True)
        )
        for case in APPROVED_CASES[1:]:
            timestamps = (
                group.loc[group["case"].eq(case), "ts"]
                .sort_values(kind="stable")
                .reset_index(drop=True)
            )
            if not timestamps.equals(reference):
                raise ValueError(
                    "daily plotting requires identical timestamps across cases; "
                    f"date={pd.Timestamp(date_value).date()}, case={case}"
                )

        if len(reference) != FULL_SAMPLES_PER_CASE_DATE:
            raise ValueError(
                "daily plotting requires exactly 690 one-minute timestamps "
                "for every date/case"
            )
        expected_start = pd.Timestamp(date_value) + pd.Timedelta(hours=7, minutes=30)
        expected_end = pd.Timestamp(date_value) + pd.Timedelta(hours=18, minutes=59)
        if reference.iloc[0] != expected_start or reference.iloc[-1] != expected_end:
            raise ValueError(
                "daily plotting timestamps must span 07:30 through 18:59"
            )
        if not reference.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
            raise ValueError("daily plotting requires a complete one-minute cadence")

        pv_spread = group.groupby("ts", sort=False)["pv_kw"].agg(
            lambda values: float(values.max() - values.min())
        )
        if (pv_spread > POWER_ACCOUNTING_ATOL_KW).any():
            raise ValueError(
                "daily plotting requires identical PV availability across cases"
            )
        outdoor_temperature_spread = group.groupby("ts", sort=False)["T_out"].agg(
            lambda values: float(values.max() - values.min())
        )
        if (outdoor_temperature_spread > 1e-9).any():
            raise ValueError(
                "daily plotting requires identical outdoor temperature across cases"
            )
        rain_spread = group.groupby("ts", sort=False)["rain_status"].agg(
            lambda values: float(values.max() - values.min())
        )
        if (rain_spread > 0.0).any():
            raise ValueError(
                "daily plotting requires identical raw joined rain_status across cases"
            )

    case_order = {case: index for index, case in enumerate(APPROVED_CASES)}
    source["_case_order"] = source["case"].map(case_order)
    return source.sort_values(
        ["_plot_date", "_case_order", "ts"], kind="stable"
    ).reset_index(drop=True)


def _padded_limits(
    values: np.ndarray,
    *,
    anchors: tuple[float, ...] = (),
    minimum_padding: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if anchors:
        finite = np.concatenate([finite, np.asarray(anchors, dtype=float)])
    if finite.size == 0:
        raise ValueError("cannot compute plot limits without finite data")
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    padding = max(minimum_padding, 0.05 * span)
    return (lower - padding, upper + padding)


def compute_global_plot_limits(frame: pd.DataFrame) -> GlobalPlotLimits:
    """Compute pooled fallback limits; daily temperature panels use focused per-day ranges."""

    source = _prepare_plot_frame(frame, require_single_date=False)
    temperature = _padded_limits(
        source.loc[:, ["T_mean"]].to_numpy(dtype=float),
        anchors=(30.0,),
        minimum_padding=0.25,
    )
    pmv = _padded_limits(
        source["pmv_mean"].to_numpy(dtype=float),
        anchors=(-0.5, 0.5),
        minimum_padding=0.10,
    )
    maximum_power = float(
        source.loc[:, ["pv_kw", "self_kw"]].to_numpy(dtype=float).max()
    )
    power_upper = max(0.5, maximum_power * 1.06)
    return GlobalPlotLimits(
        temperature_c=temperature,
        pmv=pmv,
        pv_self_kw=(0.0, power_upper),
    )


def _observed_rain_intervals(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    reference = frame.loc[frame["case"].eq(APPROVED_CASES[0]), ["ts", "rain_status"]]
    reference = reference.sort_values("ts", kind="stable").reset_index(drop=True)
    active = reference["rain_status"].ge(0.5).to_numpy()
    timestamps = reference["ts"].tolist()
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    previous: pd.Timestamp | None = None
    one_minute = pd.Timedelta(minutes=1)
    for timestamp, is_active in zip(timestamps, active, strict=True):
        timestamp = pd.Timestamp(timestamp)
        contiguous = previous is not None and timestamp - previous == one_minute
        if is_active and (start is None or not contiguous):
            if start is not None and previous is not None:
                intervals.append((start, previous + one_minute))
            start = timestamp
        elif not is_active and start is not None:
            if previous is None:
                raise RuntimeError("rain interval state is inconsistent")
            intervals.append((start, previous + one_minute))
            start = None
        previous = timestamp
    if start is not None and previous is not None:
        intervals.append((start, previous + one_minute))
    return intervals


def _load_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _configure_plot_style(plt: Any) -> None:
    """Use manuscript-scale typography before LaTeX resizes the figure."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16.0,
            "axes.labelsize": 16.0,
            "axes.titlesize": 17.0,
            "legend.fontsize": 14.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
        }
    )


def render_daily_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    plot_limits: GlobalPlotLimits | None = None,
) -> DailyFigureResult:
    """Render one four-panel observed-future daily trajectory figure."""

    source = _prepare_plot_frame(frame, require_single_date=True)
    if plot_limits is None:
        plot_limits = compute_global_plot_limits(source.drop(columns="_plot_date"))
    if not isinstance(plot_limits, GlobalPlotLimits):
        raise TypeError("plot_limits must be a GlobalPlotLimits instance")

    date_value = pd.Timestamp(source["_plot_date"].iloc[0])
    date_iso = date_value.strftime("%Y-%m-%d")
    rain_intervals = _observed_rain_intervals(source)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / f"observed_future_daily_{date_iso}.png"
    pdf_path = output_root / f"observed_future_daily_{date_iso}.pdf"

    plt = _load_matplotlib_pyplot()
    _configure_plot_style(plt)
    import matplotlib.dates as mdates

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(10.5, 12.5),
        sharex=True,
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    figure.text(
        0.5,
        0.988,
        date_iso,
        ha="center",
        va="top",
        fontsize=17,
        fontweight="semibold",
    )

    for case in APPROVED_CASES:
        group = source.loc[source["case"].eq(case)].sort_values("ts", kind="stable")
        axes[0].plot(
            group["ts"],
            group["T_mean"],
            color=CASE_COLORS[case],
            linestyle=CASE_LINESTYLES[case],
            linewidth=1.55,
            label=CASE_DISPLAY_LABELS[case],
        )
    outdoor_reference = source.loc[
        source["case"].eq(APPROVED_CASES[0])
    ].sort_values("ts", kind="stable")
    outdoor_axis = axes[0].twinx()
    outdoor_axis.plot(
        outdoor_reference["ts"],
        outdoor_reference["T_out"],
        color="#555555",
        linestyle=(0, (4, 2)),
        linewidth=1.45,
        label="Outdoor temperature (right axis)",
    )
    axes[0].axhline(
        30.0,
        color="#6B6B6B",
        linestyle=(0, (2, 2)),
        linewidth=1.0,
        label="30 °C reference",
    )
    axes[0].plot([], [], color="#555555", linestyle=(0, (4, 2)),
                 linewidth=1.45, label="Outdoor temperature (right axis)")
    axes[0].set_ylabel("Mean indoor\ntemperature (°C)")
    outdoor_axis.set_ylabel("Outdoor\ntemperature (°C)", color="#555555")
    outdoor_axis.tick_params(axis="y", colors="#555555")
    focus_temperature_axes([axes[0]], label="observed indoor " + date_iso)
    focus_temperature_axes([outdoor_axis], label="observed outdoor " + date_iso)

    for case, offset in WINDOW_PLOT_OFFSETS.items():
        group = source.loc[source["case"].eq(case)].sort_values("ts", kind="stable")
        axes[1].step(
            group["ts"],
            1.0 - group["z"].to_numpy(dtype=float) + offset,
            where="post",
            color=CASE_COLORS[case],
            linestyle=CASE_LINESTYLES[case],
            linewidth=1.45,
            label=CASE_DISPLAY_LABELS[case],
        )
    axes[1].set_ylabel("Window open\nstatus (0/1)")
    axes[1].set_yticks([0.0, 1.0], labels=["Closed", "Open"])
    axes[1].set_ylim(-0.10, 1.10)
    axes[2].axhspan(
        -0.5,
        0.5,
        color="#BDBDBD",
        alpha=0.18,
        linewidth=0.0,
        label="PMV comfort band (±0.5)",
    )
    for case in APPROVED_CASES:
        group = source.loc[source["case"].eq(case)].sort_values("ts", kind="stable")
        axes[2].plot(
            group["ts"],
            group["pmv_mean"],
            color=CASE_COLORS[case],
            linestyle=CASE_LINESTYLES[case],
            linewidth=1.55,
            label=CASE_DISPLAY_LABELS[case],
        )
    axes[2].axhline(0.0, color="#6B6B6B", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("Mean PMV")
    axes[2].set_ylim(*plot_limits.pmv)

    pv_reference = source.loc[source["case"].eq(APPROVED_CASES[0])].sort_values(
        "ts", kind="stable"
    )
    axes[3].fill_between(
        pv_reference["ts"],
        0.0,
        pv_reference["pv_kw"],
        color="#F0C541",
        alpha=0.35,
        linewidth=0.8,
        edgecolor="#B58B00",
        label="Available PV",
    )
    for case in APPROVED_CASES:
        group = source.loc[source["case"].eq(case)].sort_values("ts", kind="stable")
        axes[3].plot(
            group["ts"],
            group["self_kw"],
            color=CASE_COLORS[case],
            linestyle=CASE_LINESTYLES[case],
            linewidth=1.55,
            label=CASE_DISPLAY_LABELS[case],
        )
    axes[3].set_ylabel("Power (kW)")
    axes[3].set_ylim(*plot_limits.pv_self_kw)
    axes[3].set_xlabel("Local time")

    for panel_label, axis in zip(DAILY_PANEL_LABELS, axes, strict=True):
        for interval_index, (start, end) in enumerate(rain_intervals):
            axis.axvspan(
                start,
                end,
                color="#9ECAE1",
                alpha=0.24,
                linewidth=0.0,
                label=(
                    "Observed rain"
                    if axis is axes[0] and interval_index == 0
                    else "_nolegend_"
                ),
            )
        axis.text(
            0.008,
            0.94,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=14.0,
            fontweight="semibold",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    first_timestamp = pd.Timestamp(source["ts"].min())
    last_timestamp = pd.Timestamp(source["ts"].max()) + pd.Timedelta(minutes=1)
    axes[-1].set_xlim(first_timestamp, last_timestamp)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            if label.startswith("_") or label in legend_labels:
                continue
            legend_handles.append(handle)
            legend_labels.append(label)
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
        fontsize=14.0,
        handlelength=3.0,
        columnspacing=1.4,
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.86))

    figure.savefig(
        png_path,
        dpi=300,
        facecolor="white",
        metadata={"Software": "MMV4EF observed-future results"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Creator": "MMV4EF observed-future results",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)

    return DailyFigureResult(
        png_path=png_path,
        pdf_path=pdf_path,
        date_iso=date_iso,
        case_labels=APPROVED_CASES,
        display_labels=tuple(CASE_DISPLAY_LABELS[case] for case in APPROVED_CASES),
        plot_limits=plot_limits,
        window_offsets=dict(WINDOW_PLOT_OFFSETS),
        rain_interval_count=len(rain_intervals),
        panel_labels=DAILY_PANEL_LABELS,
    )


_ISO_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def _date_from_thumbnail_path(path: Path) -> str:
    match = _ISO_DATE_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"daily thumbnail filename has no ISO date: {path.name}")
    date_iso = match.group(1)
    try:
        parsed = pd.Timestamp(date_iso)
    except ValueError as error:
        raise ValueError(f"invalid ISO date in thumbnail filename: {path.name}") from error
    if parsed.strftime("%Y-%m-%d") != date_iso:
        raise ValueError(f"invalid ISO date in thumbnail filename: {path.name}")
    return date_iso


def render_contact_sheet(
    daily_png_paths: Sequence[Path],
    output_path: Path,
    *,
    validate_full: bool = True,
) -> ContactSheetResult:
    """Render a chronological 4x6 white-background daily contact sheet."""

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    supplied = [Path(path) for path in daily_png_paths]
    if validate_full and len(supplied) != FULL_DATE_COUNT:
        raise ValueError(
            f"full contact sheet requires exactly {FULL_DATE_COUNT} daily images"
        )
    if not supplied:
        raise ValueError("contact sheet requires at least one daily image")
    rows, columns = 6, 4
    if len(supplied) > rows * columns:
        raise ValueError("4x6 contact sheet accepts at most 24 daily images")
    missing = [path for path in supplied if not path.is_file()]
    if missing:
        raise ValueError(f"contact sheet input does not exist: {missing[0]}")

    dated_paths = [(_date_from_thumbnail_path(path), path) for path in supplied]
    dates = [date_iso for date_iso, _ in dated_paths]
    if len(set(dates)) != len(dates):
        raise ValueError("contact sheet requires one daily image per ISO date")
    dated_paths.sort(key=lambda item: (item[0], item[1].name))

    cell_width, cell_height = 720, 700
    caption_height = 46
    margin = 16
    canvas = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), color="white"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default(size=30)

    for index, (date_iso, path) in enumerate(dated_paths):
        row, column = divmod(index, columns)
        left = column * cell_width
        top = row * cell_height
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline="#E0E0E0",
            width=1,
        )
        text_box = draw.textbbox((0, 0), date_iso, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (left + (cell_width - text_width) / 2.0, top + 7),
            date_iso,
            fill="#262626",
            font=font,
        )
        with Image.open(path) as daily_image:
            thumbnail = ImageOps.contain(
                daily_image.convert("RGB"),
                (
                    cell_width - 2 * margin,
                    cell_height - caption_height - 2 * margin,
                ),
                method=Image.Resampling.LANCZOS,
            )
        image_left = left + (cell_width - thumbnail.width) // 2
        image_top = top + caption_height + (
            cell_height - caption_height - thumbnail.height
        ) // 2
        canvas.paste(thumbnail, (image_left, image_top))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=False)
    ordered_dates = tuple(date_iso for date_iso, _ in dated_paths)
    ordered_paths = tuple(path for _, path in dated_paths)
    return ContactSheetResult(
        output_path=destination,
        ordered_dates=ordered_dates,
        ordered_input_paths=ordered_paths,
        grid_shape=(rows, columns),
    )


def _deterministic_jitter(count: int, width: float = 0.12) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=float)
    if count == 1:
        return np.zeros(1, dtype=float)
    return np.linspace(-width, width, count, dtype=float)


def _aggregate_group(
    daily: pd.DataFrame, metric: str, case: str
) -> tuple[pd.DataFrame, np.ndarray, int]:
    group = daily.loc[daily["case"].eq(case), ["date", metric]].copy()
    group["date"] = pd.to_datetime(group["date"], errors="raise").dt.normalize()
    group[metric] = pd.to_numeric(group[metric], errors="coerce")
    group = group.sort_values("date", kind="stable").reset_index(drop=True)
    values = group[metric].to_numpy(dtype=float)
    finite = np.isfinite(values)
    return group.loc[finite].reset_index(drop=True), values[finite], int((~finite).sum())


def _draw_boxplot_groups(
    axis: Any,
    groups: list[dict[str, Any]],
    *,
    record_metadata: bool,
    finite_counts: dict[str, int] | None = None,
    undefined_counts: dict[str, int] | None = None,
    extrema: dict[str, tuple[float, float]] | None = None,
) -> None:
    positions: list[float] = []
    tick_labels: list[str] = []
    for group in groups:
        position = float(group["position"])
        values = np.asarray(group["values"], dtype=float)
        color = str(group["color"])
        positions.append(position)
        # The manuscript caption defines the common 22-day sample. Repeating
        # counts below every category makes the plot denser without adding
        # information, so category names alone are shown here.
        tick_labels.append(str(group["label"]))
        if values.size:
            artists = axis.boxplot(
                [values],
                positions=[position],
                widths=0.48,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={"facecolor": "none", "edgecolor": color, "linewidth": 1.45},
                whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2},
                medianprops={"color": color, "linewidth": 1.8},
            )
            for patch in artists["boxes"]:
                patch.set_alpha(1.0)
            axis.scatter(
                position + _deterministic_jitter(len(values)),
                values,
                s=18,
                marker=group.get("marker", "o"),
                facecolors="white",
                edgecolors=color,
                linewidths=0.85,
                alpha=0.88,
                zorder=3,
            )
        if record_metadata:
            if finite_counts is None or undefined_counts is None or extrema is None:
                raise RuntimeError("aggregate metadata stores are required")
            key = str(group["key"])
            finite_counts[key] = int(values.size)
            undefined_counts[key] = int(group["undefined"])
            extrema[key] = (
                (float(values.min()), float(values.max()))
                if values.size
                else (float("nan"), float("nan"))
            )
    axis.set_xticks(positions, labels=tick_labels)


def _validate_aggregate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = [
        "date",
        "case",
        "dr_e_pct",
        "dr_p_morning_pct",
        "dr_p_evening_pct",
        "sc_pct",
        "ss_pct",
    ]
    _require_columns(daily, required)
    source = daily.loc[daily["case"].isin(APPROVED_CASES), required].copy()
    if source.empty or set(source["case"]) != set(APPROVED_CASES):
        raise ValueError("aggregate figure requires all four approved cases")
    source["date"] = pd.to_datetime(source["date"], errors="raise").dt.normalize()
    if source.duplicated(["date", "case"]).any():
        raise ValueError("aggregate figure requires unique date/case rows")
    expected_dates = set(source.loc[source["case"].eq(APPROVED_CASES[0]), "date"])
    for case in APPROVED_CASES[1:]:
        if set(source.loc[source["case"].eq(case), "date"]) != expected_dates:
            raise ValueError("aggregate figure requires identical date sets across cases")
    for column in required[2:]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
        if np.isinf(source[column].to_numpy(dtype=float)).any():
            raise ValueError(f"aggregate figure found infinite {column}")
    comparison = source["case"].isin(DR_COMPARISON_CASES)
    for column in ("dr_e_pct", "dr_p_morning_pct", "dr_p_evening_pct"):
        if not np.isfinite(source.loc[comparison, column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"aggregate figure requires finite AC27-referenced {column} "
                "for RBC, MPC, and MPC-PV"
            )
    return source.sort_values(["date", "case"], kind="stable").reset_index(drop=True)


def render_aggregate_kpi_figure(
    daily: pd.DataFrame, output_dir: Path
) -> AggregateFigureResult:
    """Render audited daily KPI distributions while retaining every finite point."""

    source = _validate_aggregate_daily(daily)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "observed_future_energy_flexibility.png"
    pdf_path = destination / "observed_future_energy_flexibility.pdf"
    finite_counts: dict[str, int] = {}
    undefined_counts: dict[str, int] = {}
    extrema: dict[str, tuple[float, float]] = {}
    outlier_annotations: list[str] = []
    outlier_artists: dict[str, Any] = {}
    peak_inset: Any | None = None

    def make_group(
        metric: str,
        case: str,
        label: str,
        position: float,
        marker: str = "o",
        split_count_label: bool = False,
    ) -> dict[str, Any]:
        finite_rows, values, undefined = _aggregate_group(source, metric, case)
        return {
            "key": f"{metric}|{CASE_DISPLAY_LABELS[case]}",
            "metric": metric,
            "case": case,
            "label": label,
            "position": position,
            "color": CASE_COLORS[case],
            "marker": marker,
            "split_count_label": split_count_label,
            "finite_rows": finite_rows,
            "values": values,
            "undefined": undefined,
        }

    panel_a = [
        make_group("dr_e_pct", case, CASE_DISPLAY_LABELS[case], index + 1.0)
        for index, case in enumerate(DR_COMPARISON_CASES)
    ]
    panel_b = [
        *[
            make_group(
                "dr_p_morning_pct",
                case,
                f"AM\n{CASE_DISPLAY_LABELS[case]}",
                1.15 * index + 1.0,
                "o",
                True,
            )
            for index, case in enumerate(DR_COMPARISON_CASES)
        ],
        *[
            make_group(
                "dr_p_evening_pct",
                case,
                f"PM\n{CASE_DISPLAY_LABELS[case]}",
                1.15 * index + 4.7,
                "s",
                True,
            )
            for index, case in enumerate(DR_COMPARISON_CASES)
        ],
    ]
    panel_c = [
        make_group("sc_pct", case, CASE_DISPLAY_LABELS[case], index + 1.0)
        for index, case in enumerate(APPROVED_CASES)
    ]
    panel_d = [
        make_group("ss_pct", case, CASE_DISPLAY_LABELS[case], index + 1.0)
        for index, case in enumerate(APPROVED_CASES)
    ]

    plt = _load_matplotlib_pyplot()
    _configure_plot_style(plt)
    from matplotlib.ticker import PercentFormatter

    output_dpi = 300.0
    figure, axes_grid = plt.subplots(
        2, 2, figsize=(12.0, 8.8), constrained_layout=True
    )
    figure.set_dpi(output_dpi)
    figure.patch.set_facecolor("white")
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.045, 1.0, 0.955))
    axes = axes_grid.ravel()
    for axis, groups in zip(
        axes, (panel_a, panel_b, panel_c, panel_d), strict=True
    ):
        _draw_boxplot_groups(
            axis,
            groups,
            record_metadata=True,
            finite_counts=finite_counts,
            undefined_counts=undefined_counts,
            extrema=extrema,
        )
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=12.5)

    dr_reference_label = "RBC-AC reference (0%)"
    for axis in axes[:2]:
        axis.axhline(
            0.0,
            color=CASE_COLORS[DR_REFERENCE_CASE],
            linewidth=0.9,
            zorder=1,
        )
    for panel_label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes, strict=True):
        axis.text(
            0.01,
            0.98,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=14.0,
            fontweight="semibold",
        )
    axes[0].set_ylabel("$DR^E$ (%)")
    axes[1].set_ylabel("$DR^P$ (%)")
    axes[2].set_ylabel("SC (%)")
    axes[3].set_ylabel("SS (%)")

    previous_signature: tuple[tuple[float, ...], ...] | None = None
    for _ in range(12):
        figure.canvas.draw()
        layout_axes = [*axes, *([peak_inset] if peak_inset is not None else [])]
        signature = tuple(
            tuple(float(value) for value in axis.get_position().bounds)
            for axis in layout_axes
        )
        if previous_signature is not None and np.allclose(
            np.asarray(signature),
            np.asarray(previous_signature),
            rtol=0.0,
            atol=1e-12,
        ):
            break
        previous_signature = signature
    else:
        raise RuntimeError("aggregate constrained layout did not converge")

    # Quantize the converged geometry before the final draw. Sub-pixel layout
    # noise can otherwise produce different antialiasing in byte-identical
    # cold-process renders even when the visible layout is unchanged.
    frozen_positions = [
        tuple(round(float(value), 6) for value in axis.get_position().bounds)
        for axis in axes
    ]
    frozen_inset_position = (
        tuple(
            round(float(value), 6)
            for value in peak_inset.get_position().bounds
        )
        if peak_inset is not None
        else None
    )
    figure.set_layout_engine(None)
    for axis, position in zip(axes, frozen_positions, strict=True):
        axis.set_position(position)
    if peak_inset is not None and frozen_inset_position is not None:
        peak_inset.set_axes_locator(None)
        peak_inset.set_position(frozen_inset_position)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bbox_scale = output_dpi / float(figure.dpi)

    def scaled_box(artist: Any) -> tuple[float, float, float, float]:
        box = artist.get_window_extent(renderer=renderer)
        return tuple(float(value) * bbox_scale for value in box.extents)

    count_label_boxes: dict[str, tuple[float, float, float, float]] = {}
    for panel_index, axis in enumerate(axes):
        for label_index, label in enumerate(axis.get_xticklabels()):
            count_label_boxes[f"panel_{panel_index}_label_{label_index}"] = scaled_box(
                label
            )
    peak_inset_box = scaled_box(peak_inset) if peak_inset is not None else None
    outlier_text_boxes = {
        date_iso: scaled_box(annotation)
        for date_iso, annotation in sorted(outlier_artists.items())
    }
    outlier_arrow_boxes = {
        date_iso: scaled_box(annotation.arrow_patch)
        for date_iso, annotation in sorted(outlier_artists.items())
        if annotation.arrow_patch is not None
    }
    figure_size_px = tuple(
        int(round(value * output_dpi)) for value in figure.get_size_inches()
    )

    figure.savefig(
        png_path,
        dpi=output_dpi,
        facecolor="white",
        metadata={"Software": "MMV4EF observed-future results"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Creator": "MMV4EF observed-future results",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return AggregateFigureResult(
        png_path=png_path,
        pdf_path=pdf_path,
        dr_reference_label=dr_reference_label,
        dr_panel_case_labels=tuple(
            CASE_DISPLAY_LABELS[case] for case in DR_COMPARISON_CASES
        ),
        finite_counts=finite_counts,
        undefined_counts=undefined_counts,
        extrema=extrema,
        outlier_annotations=tuple(sorted(set(outlier_annotations))),
        figure_size_px=figure_size_px,
        count_label_boxes_px=count_label_boxes,
        peak_inset_box_px=peak_inset_box,
        outlier_text_boxes_px=outlier_text_boxes,
        outlier_arrow_boxes_px=outlier_arrow_boxes,
    )


_SUMMARY_METRIC_MAP = {
    "hvac_kwh": "hvac_kwh",
    "pv_kwh": "pv_available_kwh",
    "grid_kwh": "grid_kwh",
    "export_kwh": "export_kwh",
    "self_kwh": "self_consumed_kwh",
    "sc_pct": "self_consumption_ratio",
    "ss_pct": "self_sufficiency_ratio",
}


def _validate_summary_sources(
    source: pd.DataFrame, *, label: str, include_date: bool
) -> pd.DataFrame:
    keys = ["case"] if not include_date else ["date", "case"]
    required = [*keys, *_SUMMARY_METRIC_MAP.values()]
    if include_date:
        required.append("switch_count")
    _require_columns(source, required)
    out = source.loc[source["case"].isin(APPROVED_CASES)].copy()
    if include_date:
        out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if out.duplicated(keys).any():
        raise ValueError(f"source {label} contains duplicate summary keys")
    if set(out["case"]) != set(APPROVED_CASES):
        raise ValueError(f"source {label} is missing approved cases")
    if "forecast_source" in out.columns:
        expected = out["case"].map(EXPECTED_FORECAST_SOURCE)
        if not out["forecast_source"].eq(expected).all():
            raise ValueError(f"source {label} forecast-source metadata mismatch")
    return out


def _assert_semantic_close(
    actual: pd.Series,
    expected: pd.Series,
    *,
    label: str,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> None:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    expected_values = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    both_nan = np.isnan(actual_values) & np.isnan(expected_values)
    close = np.isclose(
        actual_values, expected_values, atol=atol, rtol=rtol, equal_nan=False
    )
    valid = close | both_nan
    if not valid.all():
        index = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            f"{label} reconciliation failed at row {index}: "
            f"derived={actual_values[index]:.12g}, source={expected_values[index]:.12g}"
        )


def _truthy_exact_index(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value is True
        or (isinstance(value, (int, np.integer)) and int(value) == 1)
        or str(value).strip().lower() in {"true", "1", "1.0"}
    )


def reconcile_source_summaries(
    daily: pd.DataFrame,
    pooled: pd.DataFrame,
    source_daily: pd.DataFrame,
    source_aggregate: pd.DataFrame,
    source_validation: pd.DataFrame,
    *,
    validate_full: bool = True,
) -> dict[str, Any]:
    """Reconcile independent KPIs to source summaries without trusting them as input."""

    _require_columns(daily, ["date", "case", "switch_count", *_SUMMARY_METRIC_MAP])
    _require_columns(pooled, ["case", "switch_count", *_SUMMARY_METRIC_MAP])
    derived_daily = daily.loc[daily["case"].isin(APPROVED_CASES)].copy()
    derived_daily["date"] = pd.to_datetime(
        derived_daily["date"], errors="raise"
    ).dt.normalize()
    derived_pooled = pooled.loc[pooled["case"].isin(APPROVED_CASES)].copy()
    if derived_daily.duplicated(["date", "case"]).any() or derived_pooled.duplicated(
        "case"
    ).any():
        raise ValueError("derived summary keys must be unique")
    if set(derived_daily["case"]) != set(APPROVED_CASES) or set(
        derived_pooled["case"]
    ) != set(APPROVED_CASES):
        raise ValueError("derived summaries require all four approved cases")

    source_day = _validate_summary_sources(
        source_daily, label="daily", include_date=True
    )
    source_pool = _validate_summary_sources(
        source_aggregate, label="aggregate", include_date=False
    )
    daily_keys = set(map(tuple, derived_daily[["date", "case"]].to_records(index=False)))
    source_keys = set(map(tuple, source_day[["date", "case"]].to_records(index=False)))
    if daily_keys != source_keys:
        raise ValueError("source daily summary does not match derived date/case keys")
    derived_daily = derived_daily.sort_values(["date", "case"], kind="stable")
    source_day = source_day.sort_values(["date", "case"], kind="stable")
    derived_pooled = derived_pooled.sort_values("case", kind="stable")
    source_pool = source_pool.sort_values("case", kind="stable")

    comparison_count = 0
    for derived_column, source_column in _SUMMARY_METRIC_MAP.items():
        daily_expected = pd.to_numeric(source_day[source_column], errors="coerce")
        pooled_expected = pd.to_numeric(source_pool[source_column], errors="coerce")
        if derived_column in {"sc_pct", "ss_pct"}:
            daily_expected = 100.0 * daily_expected
            pooled_expected = 100.0 * pooled_expected
        _assert_semantic_close(
            derived_daily[derived_column],
            daily_expected,
            label=f"daily {derived_column}",
        )
        _assert_semantic_close(
            derived_pooled[derived_column],
            pooled_expected,
            label=f"aggregate {derived_column}",
        )
        comparison_count += len(derived_daily) + len(derived_pooled)

    source_switch_totals = source_day.groupby("case", observed=True)[
        "switch_count"
    ].sum()
    derived_switch_totals = derived_pooled.set_index("case")["switch_count"]
    _assert_semantic_close(
        derived_switch_totals.reindex(APPROVED_CASES),
        source_switch_totals.reindex(APPROVED_CASES),
        label="daily-summed switch_count",
    )

    validation_required = [
        "date",
        "case",
        "expected_rows",
        "actual_rows",
        "exact_index",
        "missing_required_columns",
        "required_missing_values",
        "rain_lockout_violations",
    ]
    _require_columns(source_validation, validation_required)
    validation = source_validation.loc[
        source_validation["case"].isin(APPROVED_CASES)
    ].copy()
    validation["date"] = pd.to_datetime(validation["date"], errors="raise").dt.normalize()
    if validation.duplicated(["date", "case"]).any():
        raise ValueError("source validation contains duplicate date/case keys")
    validation_keys = set(
        map(tuple, validation[["date", "case"]].to_records(index=False))
    )
    if validation_keys != daily_keys:
        raise ValueError("source validation does not match derived date/case keys")
    if validate_full:
        if derived_daily["date"].nunique() != FULL_DATE_COUNT or len(
            derived_daily
        ) != FULL_DATE_COUNT * len(APPROVED_CASES):
            raise ValueError("source reconciliation requires 22 dates and four cases")
    expected_rows = pd.to_numeric(validation["expected_rows"], errors="coerce")
    actual_rows = pd.to_numeric(validation["actual_rows"], errors="coerce")
    if not expected_rows.eq(FULL_SAMPLES_PER_CASE_DATE).all() or not actual_rows.eq(
        FULL_SAMPLES_PER_CASE_DATE
    ).all():
        raise ValueError("source validation must report exactly 690 expected/actual rows")
    if not _truthy_exact_index(validation["exact_index"]).all():
        raise ValueError("source validation exact_index must be true")
    missing_columns = validation["missing_required_columns"].fillna("").astype(str).str.strip()
    if not missing_columns.isin({"", "0", "0.0", "[]"}).all():
        raise ValueError("source validation reports missing required columns")
    missing_values = pd.to_numeric(
        validation["required_missing_values"], errors="coerce"
    )
    if not missing_values.eq(0).all():
        raise ValueError("source validation reports missing required values")
    mpc_validation = validation.loc[validation["case"].isin(MPC_CASES)]
    rain_violations = pd.to_numeric(
        mpc_validation["rain_lockout_violations"], errors="coerce"
    )
    if not rain_violations.eq(0).all():
        raise ValueError("observed-future rain lockout violations must be zero")

    switch_totals = {
        CASE_DISPLAY_LABELS[case]: int(derived_switch_totals.loc[case])
        for case in APPROVED_CASES
    }
    return {
        "status": "passed",
        "daily_rows": int(len(derived_daily)),
        "aggregate_rows": int(len(derived_pooled)),
        "validation_rows": int(len(validation)),
        "numeric_comparisons": int(comparison_count),
        "tolerance_absolute": 1e-9,
        "tolerance_relative": 1e-9,
        "switch_totals": switch_totals,
        "switch_source": "summed source daily summaries",
        "aggregate_switch_count_ignored": True,
        "observed_mpc_rain_lockout_violations": int(rain_violations.sum()),
    }


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _ordered_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ts" in out.columns:
        out = out.sort_values(["ts", "case"], kind="stable")
        out["ts"] = pd.to_datetime(out["ts"], errors="raise").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    elif "date" in out.columns and "case" in out.columns:
        out = out.sort_values(["date", "case"], kind="stable")
        out["date"] = pd.to_datetime(out["date"], errors="raise").dt.strftime(
            "%Y-%m-%d"
        )
    elif "case" in out.columns:
        order = {case: index for index, case in enumerate(APPROVED_CASES)}
        out["_case_order"] = out["case"].map(order).fillna(len(order))
        out = out.sort_values(["_case_order", "case"], kind="stable").drop(
            columns="_case_order"
        )
    return out.reset_index(drop=True)


def write_result_tables(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    pooled: pd.DataFrame,
    selection_aids: pd.DataFrame,
    audit: EnergyAudit,
    output_dir: Path,
) -> dict[str, Path]:
    """Write stable tabular artifacts and the independent energy audit."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "minute_data": root / "observed_future_minute_data.csv.gz",
        "daily_metrics": root / "observed_future_daily_metrics.csv",
        "pooled_metrics": root / "observed_future_pooled_metrics.csv",
        "selection_aids": root / "observed_future_selection_aids.csv",
        "energy_audit": root / "observed_future_energy_audit.json",
    }
    _ordered_for_csv(minute).to_csv(
        paths["minute_data"],
        index=False,
        float_format="%.15g",
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    for key, frame in (
        ("daily_metrics", daily),
        ("pooled_metrics", pooled),
        ("selection_aids", selection_aids),
    ):
        _ordered_for_csv(frame).to_csv(
            paths[key],
            index=False,
            float_format="%.15g",
            lineterminator="\n",
        )
    paths["energy_audit"].write_text(
        _json_text(asdict(audit)), encoding="utf-8", newline="\n"
    )
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    source_paths: dict[str, Path],
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    pooled: pd.DataFrame,
    audit: EnergyAudit,
    reconciliation: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build the deterministic provenance and validation manifest."""

    dates = sorted(
        pd.to_datetime(daily["date"], errors="raise").dt.strftime("%Y-%m-%d").unique()
    )
    counts = minute.assign(
        _date=pd.to_datetime(minute["ts"], errors="raise").dt.strftime("%Y-%m-%d")
    ).groupby(["_date", "case"], observed=True).size()
    samples = sorted({int(value) for value in counts.to_numpy(dtype=int)})
    if samples != [FULL_SAMPLES_PER_CASE_DATE]:
        raise ValueError("manifest requires exactly 690 samples per date/case")
    return {
        "schema_version": 1,
        "source_files": {
            key: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path)),
            }
            for key, path in sorted(source_paths.items())
        },
        "dataset": {
            "cases": list(APPROVED_CASES),
            "display_aliases": {
                case: CASE_DISPLAY_LABELS[case] for case in APPROVED_CASES
            },
            "dates": dates,
            "date_count": len(dates),
            "row_count": int(len(minute)),
            "daily_row_count": int(len(daily)),
            "pooled_row_count": int(len(pooled)),
            "samples_per_date_case": FULL_SAMPLES_PER_CASE_DATE,
            "sample_boundary_local": "07:30 through 18:59, one-minute, inclusive",
        },
        "heat_balance": {
            "specific_heat_air_j_kg_k": SPECIFIC_HEAT_AIR,
            "air_density_kg_m3": AIR_DENSITY,
            "cfm_to_m3s": CFM_TO_M3S,
            "fcu_flow_cfm_each": 800.0,
            "pfcu_flow_cfm_each": 1000.0,
            "cop_fcu": COP_FCU,
            "cop_pfcu": COP_PFCU,
            "kappa_f_kw_per_k": KAPPA_F_KW_PER_K,
            "kappa_p_kw_per_k": KAPPA_P_KW_PER_K,
            "ac_expression": "kappa_f * sum(max(T_zone - T_FCU_supply, 0))",
            "window_open_expression": (
                "kappa_p * sum(max(T_out - T_PFCU_supply, 0))"
            ),
        },
        "energy_audit": asdict(audit),
        "pmv": {
            "ac_relative_humidity_pct": AC_RELATIVE_HUMIDITY_PCT,
            "ac_humidity_source": "fixed manuscript assumption",
            "window_open_humidity_source": "OutdoorHumidityWindow",
            "zone_linear_coefficients": {
                str(zone): list(coefficients)
                for zone, coefficients in sorted(PMV_LINEAR_COEFFS.items())
            },
        },
        "rain": {
            "source_file": "observed_context",
            "source_column": "rain_status",
            "controller_lockout_signal_used_as_raw_rain": False,
        },
        "reconciliation": reconciliation,
        "outputs": {
            key: str(Path(path).resolve())
            for key, path in sorted(output_paths.items())
        },
    }


def _copy_artifact_atomically(source: Path, destination: Path) -> None:
    """Replace a watched paper artifact without truncating it in place."""

    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_pipeline(
    *,
    timeseries_path: Path = DEFAULT_TIMESERIES_PATH,
    observed_context_path: Path = DEFAULT_OBSERVED_CONTEXT_PATH,
    source_daily_summary_path: Path = DEFAULT_SOURCE_DAILY_SUMMARY_PATH,
    source_aggregate_summary_path: Path = DEFAULT_SOURCE_AGGREGATE_SUMMARY_PATH,
    source_validation_path: Path = DEFAULT_SOURCE_VALIDATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    paper_fig_dir: Path = DEFAULT_PAPER_FIG_DIR,
    validate_only: bool = False,
    validate_full: bool = True,
) -> PipelineResult:
    """Validate inputs and optionally generate the complete observed-future bundle."""

    output_dir = Path(output_dir)
    paper_fig_dir = Path(paper_fig_dir)
    selected = load_observed_cases(Path(timeseries_path), validate_full=validate_full)
    audit = selected.attrs.get("energy_audit")
    if not isinstance(audit, EnergyAudit):
        raise RuntimeError("audited observed-case loader did not retain EnergyAudit")
    context = load_observed_context(Path(observed_context_path))
    minute = attach_pmv(attach_observed_context(selected, context))
    daily = compute_daily_metrics(minute)
    pooled = compute_pooled_metrics(daily)
    selection_aids = build_selection_aids(daily)
    source_daily = pd.read_csv(Path(source_daily_summary_path))
    source_aggregate = pd.read_csv(Path(source_aggregate_summary_path))
    source_validation = pd.read_csv(Path(source_validation_path))
    reconciliation = reconcile_source_summaries(
        daily,
        pooled,
        source_daily,
        source_aggregate,
        source_validation,
        validate_full=validate_full,
    )
    date_count = int(daily["date"].nunique())
    if validate_only:
        return PipelineResult(
            validated=True,
            row_count=int(len(minute)),
            date_count=date_count,
            output_paths={},
            reconciliation=reconciliation,
        )

    result_root = Path(output_dir)
    output_paths = write_result_tables(
        minute, daily, pooled, selection_aids, audit, result_root
    )
    aggregate = render_aggregate_kpi_figure(daily, result_root)
    output_paths["aggregate_png"] = aggregate.png_path
    output_paths["aggregate_pdf"] = aggregate.pdf_path

    plot_limits = compute_global_plot_limits(minute)
    daily_root = result_root / "daily"
    daily_png_paths: list[Path] = []
    for date_value in sorted(pd.to_datetime(daily["date"]).dt.normalize().unique()):
        date_mask = pd.to_datetime(minute["ts"]).dt.normalize().eq(
            pd.Timestamp(date_value)
        )
        rendered = render_daily_figure(
            minute.loc[date_mask].copy(), daily_root, plot_limits=plot_limits
        )
        daily_png_paths.append(rendered.png_path)
        output_paths[f"daily_png_{rendered.date_iso}"] = rendered.png_path
        output_paths[f"daily_pdf_{rendered.date_iso}"] = rendered.pdf_path
    contact_path = result_root / "observed_future_daily_contact_sheet.png"
    render_contact_sheet(
        daily_png_paths, contact_path, validate_full=validate_full
    )
    output_paths["daily_contact_sheet"] = contact_path

    paper_root = Path(paper_fig_dir)
    paper_root.mkdir(parents=True, exist_ok=True)
    paper_png = paper_root / aggregate.png_path.name
    paper_pdf = paper_root / aggregate.pdf_path.name
    _copy_artifact_atomically(aggregate.png_path, paper_png)
    _copy_artifact_atomically(aggregate.pdf_path, paper_pdf)
    output_paths["paper_aggregate_png"] = paper_png
    output_paths["paper_aggregate_pdf"] = paper_pdf

    manifest_path = result_root / "observed_future_manifest.json"
    output_paths["manifest"] = manifest_path
    manifest = build_manifest(
        source_paths={
            "timeseries": Path(timeseries_path),
            "observed_context": Path(observed_context_path),
            "source_daily_summary": Path(source_daily_summary_path),
            "source_aggregate_summary": Path(source_aggregate_summary_path),
            "source_validation": Path(source_validation_path),
        },
        minute=minute,
        daily=daily,
        pooled=pooled,
        audit=audit,
        reconciliation=reconciliation,
        output_paths=output_paths,
    )
    manifest_path.write_text(
        _json_text(manifest), encoding="utf-8", newline="\n"
    )
    return PipelineResult(
        validated=True,
        row_count=int(len(minute)),
        date_count=date_count,
        output_paths=output_paths,
        reconciliation=reconciliation,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build audited observed-future closed-loop result artifacts."
    )
    parser.add_argument("--timeseries", type=Path, default=DEFAULT_TIMESERIES_PATH)
    parser.add_argument(
        "--observed-context", type=Path, default=DEFAULT_OBSERVED_CONTEXT_PATH
    )
    parser.add_argument(
        "--source-daily-summary",
        type=Path,
        default=DEFAULT_SOURCE_DAILY_SUMMARY_PATH,
    )
    parser.add_argument(
        "--source-aggregate-summary",
        type=Path,
        default=DEFAULT_SOURCE_AGGREGATE_SUMMARY_PATH,
    )
    parser.add_argument(
        "--source-validation", type=Path, default=DEFAULT_SOURCE_VALIDATION_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every data, heat-balance, PMV, KPI, and reconciliation check without writing outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_pipeline(
        timeseries_path=args.timeseries,
        observed_context_path=args.observed_context,
        source_daily_summary_path=args.source_daily_summary,
        source_aggregate_summary_path=args.source_aggregate_summary,
        source_validation_path=args.source_validation,
        output_dir=args.output_dir,
        paper_fig_dir=args.paper_fig_dir,
        validate_only=args.validate_only,
        validate_full=True,
    )
    print(
        json.dumps(
            {
                "validated": result.validated,
                "rows": result.row_count,
                "dates": result.date_count,
                "outputs": {
                    key: str(path) for key, path in sorted(result.output_paths.items())
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
