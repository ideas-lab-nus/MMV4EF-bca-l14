"""Build the gated three-point MPC temperature-slack sensitivity evidence.

Only ``w_x_slack`` varies.  The accepted run must be CPU-only and its
``w_x_slack=100`` energy totals must reproduce this experiment's observed-future baselines within
0.05 kWh.  All energy, peak-window, comfort, and PV metrics are recalculated
from the minute trajectories instead of trusting notebook summary columns.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from build_discussion_results import (
    FigureResult,
    _configure_plot_style,
    _save_figure,
    _sha256,
    _safe_pct,
)
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

SWEEP_WEIGHTS = (100.0, 1000.0, 10000.0)
OBJECTIVES = ("no_pv", "onsite_pv")
OBJECTIVE_LABELS = {"no_pv": "MPC-CA", "onsite_pv": "MPC-PV"}
from plot_styles import WEIGHT_STYLES, WEIGHT_OFFSETS, focus_temperature_axes

OBJECTIVE_COLORS = {"no_pv": "#8da0cb", "onsite_pv": "#66c2a5"}
WEIGHT_COLORS = {100.0: "#66c2a5", 1000.0: "#fc8d62", 10000.0: "#8da0cb"}
BASELINE_TOLERANCE_KWH = 0.05

FULL_DATE_COUNT = 22
FULL_SAMPLES_PER_DATE_CASE = 690
MORNING_START_MINUTE = 9 * 60
MORNING_END_MINUTE = 11 * 60 + 30
EVENING_START_MINUTE = 16 * 60
EVENING_END_MINUTE = 18 * 60 + 30

REQUIRED_FILES = (
    "wx_sweep_timeseries.csv.gz",
    "wx_sweep_daily_metrics.csv",
    "wx_sweep_pooled_metrics.csv",
    "wx_sweep_validation.csv",
    "wx_sweep_run_metadata.json",
)
DEFAULT_SWEEP_DIR = Path("outputs/simulations/weight_sweep")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/discussion_results")
DEFAULT_PAPER_FIG_DIR = Path("figures")
DEFAULT_OBSERVED_POOLED_PATH = Path(__file__).resolve().parents[1] / "outputs/analysis/observed_future_results/observed_future_pooled_metrics.csv"


@dataclass
class SweepInputs:
    sweep_dir: Path
    minute: pd.DataFrame
    metadata: dict[str, Any]
    energy_audit: Any
    baseline_gate: dict[str, Any]
    safety: dict[str, Any]
    source_paths: dict[str, Path]
    observed_context_path: Path
    observed_pooled_path: Path


@dataclass(frozen=True)
class PipelineResult:
    validated: bool
    date_count: int
    minute_rows: int
    baseline_gate: dict[str, Any]
    selected_dates: dict[str, str]
    output_paths: dict[str, Path]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def validate_runtime_metadata(metadata: dict[str, Any]) -> None:
    runtime = metadata.get("runtime", {})
    if str(runtime.get("device", "")).lower() != "cpu":
        raise ValueError("sensitivity study must use the matched CPU runtime")
    if bool(runtime.get("cuda_available", True)):
        raise ValueError("sensitivity study CPU gate requires CUDA to be unavailable")
    if "+cpu" not in str(runtime.get("torch", "")):
        raise ValueError("sensitivity study must record a CPU PyTorch build")
    weights = tuple(float(value) for value in metadata.get("varied_configuration", {}).get("weights", ()))
    if weights != SWEEP_WEIGHTS:
        raise ValueError(f"expected exactly the three weights {SWEEP_WEIGHTS}")
    if metadata.get("validation_passed") is not True:
        raise ValueError("sweep metadata does not report validation_passed=true")
    if metadata.get("window_mode") != "full":
        raise ValueError("weight sensitivity requires the full occupied-period run")


def validate_baseline_gate(
    pooled: pd.DataFrame, tolerance_kwh: float = BASELINE_TOLERANCE_KWH,
    *, observed_pooled_path: Path = DEFAULT_OBSERVED_POOLED_PATH,
) -> dict[str, Any]:
    _require_columns(pooled, ["controller_objective", "w_x_slack", "hvac_kwh"], "baseline-gate table")
    reference = pd.read_csv(observed_pooled_path)
    _require_columns(reference, ["case", "hvac_kwh"], "isolated observed pooled reference")
    expected_by_objective = {}
    for objective in OBJECTIVES:
        case = f"MIQP {'no PV' if objective == 'no_pv' else 'onsite PV'} | observed future"
        selected_reference = reference.loc[reference["case"].eq(case)]
        if len(selected_reference) != 1:
            raise ValueError(f"isolated reference requires exactly one row for {case}")
        expected_by_objective[objective] = float(selected_reference.iloc[0]["hvac_kwh"])
    records: dict[str, Any] = {"reference": str(Path(observed_pooled_path).resolve())}
    all_passed = True
    for objective, expected in expected_by_objective.items():
        selected = pooled.loc[
            pooled["controller_objective"].eq(objective)
            & pd.to_numeric(pooled["w_x_slack"], errors="coerce").eq(100.0)
        ]
        if len(selected) != 1:
            raise ValueError(f"baseline gate requires one weight-100 row for {objective}")
        actual = float(selected.iloc[0]["hvac_kwh"])
        difference = actual - expected
        passed = abs(difference) <= tolerance_kwh
        all_passed = all_passed and passed
        records[objective] = {
            "expected_kwh": expected,
            "actual_kwh": actual,
            "difference_kwh": difference,
            "absolute_tolerance_kwh": tolerance_kwh,
            "passed": passed,
        }
    records["passed"] = all_passed
    if not all_passed:
        raise ValueError(f"isolated observed baseline gate failed: {records}")
    return records


def _read_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sweep metadata must be a JSON object")
    return value


def _resolve_observed_context(
    metadata: dict[str, Any], override: Path | None
) -> Path:
    if override is not None:
        return Path(override)
    manifest = metadata.get("copied_input_manifest", {})
    entry = manifest.get("l14_merged_data_with_rain.csv", {})
    copied = entry.get("copy")
    if not copied:
        raise ValueError("run metadata does not identify its frozen observed-context copy")
    path = Path(copied)
    expected_hash = metadata.get("copied_input_hashes", {}).get("l14_merged_data_with_rain.csv")
    if expected_hash and _sha256(path) != expected_hash:
        raise ValueError("frozen observed-context hash does not match run metadata")
    return path


def _case_contract(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    baseline_cases = {"AC baseline", "RBC baseline"}
    mpc_cases = {
        f"MIQP {'no PV' if objective == 'no_pv' else 'onsite PV'} | observed future | wx={int(weight)}"
        for objective in OBJECTIVES
        for weight in SWEEP_WEIGHTS
    }
    present = set(frame["case"].astype(str).unique())
    if present != baseline_cases | mpc_cases:
        raise ValueError(
            "sweep case contract mismatch; missing="
            f"{sorted((baseline_cases | mpc_cases) - present)}, extra={sorted(present - (baseline_cases | mpc_cases))}"
        )
    return baseline_cases, mpc_cases


def _prepare_minute_table(path: Path, *, validate_full: bool) -> tuple[pd.DataFrame, Any]:
    frame = pd.read_csv(path)
    required = [
        "ts",
        "case",
        "controller_objective",
        "forecast_source",
        "w_x_slack",
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
    _require_columns(frame, required, "sweep minute table")
    baseline_cases, mpc_cases = _case_contract(frame)
    frame["ts"] = pd.to_datetime(frame["ts"], errors="raise")
    if frame["ts"].dt.tz is not None:
        raise ValueError("sweep timestamps must be timezone-naive local time")
    if frame.duplicated(["ts", "case"]).any():
        raise ValueError("sweep contains duplicate timestamp/case keys")
    numeric = [
        "z",
        "T_out",
        "T_mean",
        "w_x_slack",
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
    core = ["z", "T_out", "T_mean", "pv_kw", "hvac_kw", "grid_kw", "export_kw", "self_kw", *ZONE_COLUMNS]
    if not np.isfinite(frame.loc[:, core].to_numpy(float)).all():
        raise ValueError("sweep has non-finite required minute values")
    if not frame["z"].isin((0.0, 1.0)).all():
        raise ValueError("sweep mode state z must be binary")
    if not np.isfinite(frame.loc[frame["z"].eq(1), FCU_COLUMNS].to_numpy(float)).all():
        raise ValueError("sweep has non-finite active FCU supply temperatures")
    if not np.isfinite(frame.loc[frame["z"].eq(0), PFCU_COLUMNS].to_numpy(float)).all():
        raise ValueError("sweep has non-finite active PFCU supply temperatures")
    mpc = frame["case"].isin(mpc_cases)
    if set(frame.loc[mpc, "controller_objective"]) != set(OBJECTIVES):
        raise ValueError("sweep MPC objectives are incomplete")
    if set(frame.loc[mpc, "forecast_source"]) != {"observed"}:
        raise ValueError("weight sweep must use observed future disturbances")
    if set(frame.loc[mpc, "w_x_slack"].dropna().astype(float)) != set(SWEEP_WEIGHTS):
        raise ValueError("weight sweep minute table does not contain the approved weights")
    if frame.loc[frame["case"].isin(baseline_cases), "w_x_slack"].notna().any():
        raise ValueError("fixed baselines must not carry a sweep weight")

    stored_hvac = frame["hvac_kw"].to_numpy(float)
    pv = frame["pv_kw"].to_numpy(float)
    expected_accounting = {
        "grid_kw": np.maximum(stored_hvac - pv, 0.0),
        "export_kw": np.maximum(pv - stored_hvac, 0.0),
        "self_kw": np.minimum(stored_hvac, pv),
    }
    for column, expected in expected_accounting.items():
        if not np.allclose(frame[column], expected, rtol=0.0, atol=POWER_ACCOUNTING_ATOL_KW):
            raise ValueError(f"stored sweep PV accounting failed for {column}")
    audit = audit_hvac_power(frame)
    frame["hvac_kw"] = recompute_hvac_power_kw(frame)
    frame["grid_kw"] = np.maximum(frame["hvac_kw"] - frame["pv_kw"], 0.0)
    frame["export_kw"] = np.maximum(frame["pv_kw"] - frame["hvac_kw"], 0.0)
    frame["self_kw"] = np.minimum(frame["hvac_kw"], frame["pv_kw"])
    frame["date"] = frame["ts"].dt.strftime("%Y-%m-%d")

    if validate_full:
        counts = frame.groupby(["date", "case"], observed=True).size()
        if frame["date"].nunique() != FULL_DATE_COUNT:
            raise ValueError(f"expected {FULL_DATE_COUNT} sweep dates")
        if len(counts) != FULL_DATE_COUNT * 8 or not counts.eq(FULL_SAMPLES_PER_DATE_CASE).all():
            raise ValueError("expected 690 minute rows for each of eight cases on every date")
        for (_, _), group in frame.groupby(["date", "case"], sort=False, observed=True):
            stamps = group["ts"].sort_values()
            if stamps.iloc[0].strftime("%H:%M") != "07:30" or stamps.iloc[-1].strftime("%H:%M") != "18:59":
                raise ValueError("full sweep trajectories must span 07:30 through 18:59")
            if not stamps.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
                raise ValueError("sweep timestamps must be one minute apart")
    return frame.sort_values(["ts", "case"], kind="stable").reset_index(drop=True), audit


def _quick_mpc_pooled(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["controller_objective"].isin(OBJECTIVES)].copy()
    return (
        selected.groupby(["controller_objective", "w_x_slack"], observed=True, sort=False)["hvac_kw"]
        .sum()
        .mul(STEP_HOURS)
        .rename("hvac_kwh")
        .reset_index()
    )


def validate_sweep_run(
    sweep_dir: Path,
    *,
    observed_context_path: Path | None = None,
    observed_pooled_path: Path = DEFAULT_OBSERVED_POOLED_PATH,
    validate_full: bool = True,
) -> SweepInputs:
    root = Path(sweep_dir)
    paths = {name: root / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"sweep directory is missing required files: {missing}")
    metadata = _read_metadata(paths["wx_sweep_run_metadata.json"])
    validate_runtime_metadata(metadata)
    validation = pd.read_csv(paths["wx_sweep_validation.csv"])
    _require_columns(validation, ["check", "passed"], "sweep validation")
    passed = validation["passed"].astype(str).str.lower().isin(("true", "1"))
    if validation.empty or not passed.all():
        failed = validation.loc[~passed, "check"].tolist()
        raise ValueError(f"sweep validation contains failed checks: {failed}")
    errors_path = root / "wx_sweep_errors.csv"
    if errors_path.is_file() and not pd.read_csv(errors_path).empty:
        raise ValueError("sweep contains error rows")

    minute, audit = _prepare_minute_table(paths["wx_sweep_timeseries.csv.gz"], validate_full=validate_full)
    gate = validate_baseline_gate(_quick_mpc_pooled(minute), observed_pooled_path=observed_pooled_path)
    context_path = _resolve_observed_context(metadata, observed_context_path)
    context = load_observed_context(context_path)
    minute = attach_pmv(attach_observed_context(minute, context))
    minute["date"] = minute["ts"].dt.strftime("%Y-%m-%d")
    safety_checks = validation.loc[
        validation["check"].isin(("rain_lockout_safety", "dwell_lock_safety", "pfcu_disabled_in_ac_mode"))
    ]
    safety = {
        "validation_rows": int(len(validation)),
        "rain_lockout_safety_passed": bool(safety_checks["check"].eq("rain_lockout_safety").any()),
        "dwell_lock_safety_passed": bool(safety_checks["check"].eq("dwell_lock_safety").any()),
        "pfcu_mode_safety_passed": bool(safety_checks["check"].eq("pfcu_disabled_in_ac_mode").any()),
        "all_checks_passed": True,
    }
    return SweepInputs(root, minute, metadata, audit, gate, safety, paths, context_path, Path(observed_pooled_path))


def _controller_label(objective: str) -> str:
    return {
        "baseline_ac": "RBC-AC",
        "baseline_rbc": "RBC-MM",
        "no_pv": "MPC-CA",
        "onsite_pv": "MPC-PV",
    }.get(objective, objective)


def compute_weight_metrics(inputs: SweepInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    minute = inputs.minute.copy()
    pmv_columns = [f"pmv_zone_{zone}" for zone in range(1, 6)]
    records: list[dict[str, object]] = []
    for (date, case), group in minute.groupby(["date", "case"], sort=True, observed=True):
        group = group.sort_values("ts", kind="stable")
        objective = str(group["controller_objective"].iloc[0])
        weight_value = pd.to_numeric(group["w_x_slack"], errors="coerce").dropna()
        weight = float(weight_value.iloc[0]) if not weight_value.empty else float("nan")
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
        zone = group.loc[:, ZONE_COLUMNS].to_numpy(float)
        pmv = group.loc[:, pmv_columns].to_numpy(float)
        z = group["z"].round().astype(int).to_numpy()
        exceedance = np.maximum(zone - 30.0, 0.0)
        record: dict[str, object] = {
            "date": str(date),
            "case": str(case),
            "controller_objective": objective,
            "controller": _controller_label(objective),
            "w_x_slack": weight,
            "sample_count": int(len(group)),
            "zone_sample_count": int(zone.size),
            **energy,
            "sc_pct": _safe_pct(energy["self_kwh"], energy["pv_kwh"]),
            "ss_pct": _safe_pct(energy["self_kwh"], energy["hvac_kwh"]),
            "mean_temp_c": float(zone.mean()),
            "max_temp_c": float(zone.max()),
            "any_zone_minutes_above_30": int(np.count_nonzero(np.any(zone > 30.0, axis=1))),
            "zone_degree_minutes_above_30": float(exceedance.sum()),
            "mean_pmv": float(pmv.mean()),
            "pmv_within_05_pct": 100.0 * float(np.mean(np.abs(pmv) <= 0.5)),
            "pmv_within_10_pct": 100.0 * float(np.mean(np.abs(pmv) <= 1.0)),
            "mean_abs_pmv_deviation_from_minus_05": float(np.mean(np.abs(pmv + 0.5))),
            "window_open_fraction": float(np.mean(1 - z)),
            "switch_count": int(np.count_nonzero(z[1:] != z[:-1])),
            "rain_minutes": int(group["rain_status"].ge(0.5).sum()),
            "morning_hvac_kwh": float(group.loc[morning, "hvac_kw"].sum() * STEP_HOURS),
            "evening_hvac_kwh": float(group.loc[evening, "hvac_kw"].sum() * STEP_HOURS),
        }
        records.append(record)

    daily = pd.DataFrame(records)
    ac = daily.loc[daily["controller_objective"].eq("baseline_ac")].set_index("date")
    if ac.index.nunique() != daily["date"].nunique() or ac.index.has_duplicates:
        raise ValueError("weight metrics require one AC27 reference row per date")
    for column in ("ac27_hvac_kwh", "ac27_morning_hvac_kwh", "ac27_evening_hvac_kwh", "dr_e_pct", "dr_p_morning_pct", "dr_p_evening_pct"):
        daily[column] = np.nan
    for index in daily.index:
        reference = ac.loc[daily.at[index, "date"]]
        daily.at[index, "ac27_hvac_kwh"] = reference["hvac_kwh"]
        daily.at[index, "ac27_morning_hvac_kwh"] = reference["morning_hvac_kwh"]
        daily.at[index, "ac27_evening_hvac_kwh"] = reference["evening_hvac_kwh"]
        daily.at[index, "dr_e_pct"] = _safe_pct(reference["hvac_kwh"] - daily.at[index, "hvac_kwh"], reference["hvac_kwh"])
        daily.at[index, "dr_p_morning_pct"] = _safe_pct(reference["morning_hvac_kwh"] - daily.at[index, "morning_hvac_kwh"], reference["morning_hvac_kwh"])
        daily.at[index, "dr_p_evening_pct"] = _safe_pct(reference["evening_hvac_kwh"] - daily.at[index, "evening_hvac_kwh"], reference["evening_hvac_kwh"])

    pooled = pool_weight_metrics(daily)
    inputs.baseline_gate = validate_baseline_gate(pooled, observed_pooled_path=inputs.observed_pooled_path)
    return (
        daily.sort_values(["date", "controller_objective", "w_x_slack"], kind="stable").reset_index(drop=True),
        pooled,
    )


def pool_weight_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["_weight_key"] = pd.to_numeric(frame["w_x_slack"], errors="coerce").fillna(-1.0)
    records: list[dict[str, object]] = []
    energy_columns = ("hvac_kwh", "grid_kwh", "pv_kwh", "self_kwh", "export_kwh", "morning_hvac_kwh", "evening_hvac_kwh")
    for (objective, weight_key), group in frame.groupby(["controller_objective", "_weight_key"], sort=False, observed=True):
        sums = {column: float(group[column].sum()) for column in energy_columns}
        weight = float(weight_key) if float(weight_key) >= 0 else float("nan")
        weights = group.get("zone_sample_count", pd.Series(1.0, index=group.index)).to_numpy(float)
        record: dict[str, object] = {
            "controller_objective": objective,
            "controller": _controller_label(str(objective)),
            "w_x_slack": weight,
            "date_count": int(group["date"].nunique()),
            **sums,
            "sc_pct": _safe_pct(sums["self_kwh"], sums["pv_kwh"]),
            "ss_pct": _safe_pct(sums["self_kwh"], sums["hvac_kwh"]),
            "zone_degree_minutes_above_30": float(group["zone_degree_minutes_above_30"].sum()),
            "any_zone_minutes_above_30": int(group.get("any_zone_minutes_above_30", pd.Series(0, index=group.index)).sum()),
            "max_temp_c": float(group["max_temp_c"].max()),
            "switch_count": int(group.get("switch_count", pd.Series(0, index=group.index)).sum()),
        }
        for column in ("mean_temp_c", "mean_pmv", "pmv_within_05_pct", "pmv_within_10_pct", "mean_abs_pmv_deviation_from_minus_05", "window_open_fraction"):
            if column in group:
                record[column] = float(np.average(group[column], weights=weights))
        ac_hvac = float(group["ac27_hvac_kwh"].sum())
        ac_morning = float(group["ac27_morning_hvac_kwh"].sum())
        ac_evening = float(group["ac27_evening_hvac_kwh"].sum())
        record["dr_e_pct"] = _safe_pct(ac_hvac - sums["hvac_kwh"], ac_hvac)
        record["dr_p_morning_pct"] = _safe_pct(ac_morning - sums["morning_hvac_kwh"], ac_morning)
        record["dr_p_evening_pct"] = _safe_pct(ac_evening - sums["evening_hvac_kwh"], ac_evening)
        records.append(record)
    return pd.DataFrame(records).sort_values(["controller_objective", "w_x_slack"], kind="stable", na_position="first").reset_index(drop=True)


def select_weight_profile_dates(daily: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "controller_objective", "w_x_slack", "zone_degree_minutes_above_30", "hvac_kwh"]
    _require_columns(daily, required, "weight daily metrics")
    mpc = daily.loc[daily["controller_objective"].isin(OBJECTIVES)].copy()
    if set(pd.to_numeric(mpc["w_x_slack"], errors="coerce").dropna()) != set(SWEEP_WEIGHTS):
        raise ValueError("date selection requires all three approved weights")
    burden = mpc.pivot(index="date", columns=["controller_objective", "w_x_slack"], values="zone_degree_minutes_above_30")
    hvac = mpc.pivot(index="date", columns=["controller_objective", "w_x_slack"], values="hvac_kwh")
    records: list[dict[str, object]] = []
    no_pv_reduction = burden[("no_pv", 100.0)] - burden[("no_pv", 10000.0)]
    no_pv_candidates = pd.DataFrame({"date": no_pv_reduction.index.astype(str), "burden_reduction": no_pv_reduction.to_numpy(float)})
    no_pv_winner = no_pv_candidates.sort_values(["burden_reduction", "date"], ascending=[False, True], kind="stable").iloc[0]
    records.append(
        {
            "controller_objective": "no_pv",
            "controller": "MPC-CA",
            "date": no_pv_winner["date"],
            "selection_metric": "largest_degree_minute_reduction",
            "burden_reduction_degree_minutes": float(no_pv_winner["burden_reduction"]),
            "absolute_hvac_change_kwh": float(abs(hvac.loc[no_pv_winner["date"], ("no_pv", 10000.0)] - hvac.loc[no_pv_winner["date"], ("no_pv", 100.0)])),
        }
    )
    pv_reduction = burden[("onsite_pv", 100.0)] - burden[("onsite_pv", 10000.0)]
    pv_change = (hvac[("onsite_pv", 10000.0)] - hvac[("onsite_pv", 100.0)]).abs()
    pv_candidates = pd.DataFrame({"date": pv_reduction.index.astype(str), "burden_reduction": pv_reduction.to_numpy(float), "absolute_hvac_change": pv_change.to_numpy(float)})
    pv_candidates = pv_candidates.loc[pv_candidates["burden_reduction"].gt(0)]
    if pv_candidates.empty:
        raise ValueError("no MPC-PV date improves high-temperature burden at weight 10000")
    pv_winner = pv_candidates.sort_values(["absolute_hvac_change", "date"], ascending=[False, True], kind="stable").iloc[0]
    records.append(
        {
            "controller_objective": "onsite_pv",
            "controller": "MPC-PV",
            "date": pv_winner["date"],
            "selection_metric": "largest_abs_hvac_change_with_improved_burden",
            "burden_reduction_degree_minutes": float(pv_winner["burden_reduction"]),
            "absolute_hvac_change_kwh": float(pv_winner["absolute_hvac_change"]),
        }
    )
    return pd.DataFrame(records)


def render_weight_daily_profiles(
    minute: pd.DataFrame, selected_dates: pd.DataFrame, output_dir: Path,
    *, stem: str = "discussion_weight_daily_profiles",
) -> FigureResult:
    _configure_plot_style()
    figure, axes = plt.subplots(4, 2, figsize=(13.2, 10.4), sharex="col", constrained_layout=True)
    ordered = selected_dates.set_index("controller_objective").loc[list(OBJECTIVES)].reset_index()
    for column, selection in ordered.iterrows():
        objective = str(selection["controller_objective"])
        date = str(selection["date"])
        subset = minute.loc[minute["controller_objective"].eq(objective) & minute["date"].eq(date)].copy()
        if set(subset["w_x_slack"].dropna().astype(float)) != set(SWEEP_WEIGHTS):
            raise ValueError(f"daily weight profile is incomplete for {objective} on {date}")
        for weight in SWEEP_WEIGHTS:
            group = subset.loc[pd.to_numeric(subset["w_x_slack"], errors="coerce").eq(weight)].sort_values("ts")
            color = WEIGHT_COLORS[weight]
            style = WEIGHT_STYLES[weight]
            label = f"$w_x^{{slack}}={int(weight):,}$"
            axes[0, column].plot(group["ts"], group["T_mean"], color=color, linestyle=style, linewidth=1.8, label=label)
            axes[1, column].plot(group["ts"], group["pmv_mean"], color=color, linestyle=style, linewidth=1.8, label=label)
            axes[2, column].step(group["ts"], 1 - group["z"] + WEIGHT_OFFSETS[weight], where="post", color=color, linestyle=style, linewidth=1.7, label=label)
            axes[3, column].plot(group["ts"], group["hvac_kw"], color=color, linestyle=style, linewidth=1.8, label=label)
        common = subset.loc[pd.to_numeric(subset["w_x_slack"], errors="coerce").eq(100.0)].sort_values("ts")
        # Solar availability is common to both objectives, even though only
        # onsite_pv includes the export penalty in optimization.
        axes[3, column].plot(common["ts"], common["pv_kw"], color="#444444", linestyle=":", linewidth=1.4, label="Available PV")
        axes[0, column].axhline(30.0, color="#555555", linestyle=":", linewidth=1.0)
        axes[1, column].axhspan(-0.5, 0.5, color="#56B4E9", alpha=0.10)
        axes[1, column].axhline(-1.0, color="#777777", linestyle=":", linewidth=0.8)
        axes[1, column].axhline(1.0, color="#777777", linestyle=":", linewidth=0.8)
        axes[2, column].set_ylim(-0.08, 1.08)
        axes[2, column].set_yticks((0, 1), labels=("Closed", "Open"))
        axes[0, column].set_title(pd.Timestamp(date).strftime("%Y-%m-%d"))
        axes[3, column].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[3, column].set_xlabel("Local time")
    focus_temperature_axes(list(axes[0, :]), label="weight daily temperature " + stem)
    ylabels = ("Mean zone\ntemperature (°C)", "Mean PMV", "Window\nstatus", "Power (kW)")
    for row in range(4):
        for column in range(2):
            axes[row, column].grid(True, axis="y")
            axes[row, column].text(0.01, 0.96, f"({chr(ord('a') + row * 2 + column)})", transform=axes[row, column].transAxes, ha="left", va="top", fontsize=14.0, fontweight="bold")
        axes[row, 0].set_ylabel(ylabels[row])
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for axis in (axes[0, 0], axes[0, 1], axes[3, 0], axes[3, 1]):
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
    figure.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=13.5)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.89))
    return _save_figure(figure, output_dir, stem)


def _plot_weight_response(
    axis: plt.Axes,
    pooled: pd.DataFrame,
    y_column: str,
    ylabel: str,
) -> None:
    for objective, marker, linestyle in (("no_pv", "o", "-"), ("onsite_pv", "s", "--")):
        group = pooled.loc[pooled["controller_objective"].eq(objective) & pooled["w_x_slack"].notna()].sort_values("w_x_slack")
        axis.plot(group["w_x_slack"], group[y_column], color=OBJECTIVE_COLORS[objective], marker=marker, linestyle=linestyle, linewidth=1.3, label=OBJECTIVE_LABELS[objective])
    axis.set_xscale("log")
    axis.set_xticks(SWEEP_WEIGHTS, ["100", "1,000", "10,000"])
    axis.set_xlabel("Zone-temperature slack weight $w_x^{slack}$")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major")


def render_weight_sensitivity(daily: pd.DataFrame, output_dir: Path) -> FigureResult:
    _configure_plot_style()
    pooled = pool_weight_metrics(daily)
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)
    _plot_weight_response(axes[0, 0], pooled, "zone_degree_minutes_above_30", "Zone-degree-minutes\nabove 30 °C")
    _plot_weight_response(axes[0, 1], pooled, "pmv_within_05_pct", "Samples within\n|PMV| ≤ 0.5 (%)")
    line_styles = {"dr_e_pct": "-", "dr_p_morning_pct": "--", "dr_p_evening_pct": ":"}
    line_labels = {"dr_e_pct": "Full day", "dr_p_morning_pct": "Morning", "dr_p_evening_pct": "Evening"}
    for objective, marker in (("no_pv", "o"), ("onsite_pv", "s")):
        group = pooled.loc[pooled["controller_objective"].eq(objective) & pooled["w_x_slack"].notna()].sort_values("w_x_slack")
        for metric in line_styles:
            axes[1, 0].plot(group["w_x_slack"], group[metric], color=OBJECTIVE_COLORS[objective], marker=marker, linestyle=line_styles[metric], linewidth=1.2, label="_nolegend_")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xticks(SWEEP_WEIGHTS, ["100", "1,000", "10,000"])
    axes[1, 0].set_xlabel("Zone-temperature slack weight $w_x^{slack}$")
    axes[1, 0].set_ylabel("Demand reduction\nvs RBC-AC (%)")
    axes[1, 0].grid(True, which="major")
    pv = pooled.loc[pooled["controller_objective"].eq("onsite_pv") & pooled["w_x_slack"].notna()].sort_values("w_x_slack")
    axes[1, 1].plot(pv["w_x_slack"], pv["sc_pct"], color="#D55E00", marker="o", linewidth=1.3, label="Self-consumption")
    axes[1, 1].plot(pv["w_x_slack"], pv["ss_pct"], color="#8da0cb", marker="s", linestyle="--", linewidth=1.3, label="Self-sufficiency")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xticks(SWEEP_WEIGHTS, ["100", "1,000", "10,000"])
    axes[1, 1].set_xlabel("Zone-temperature slack weight $w_x^{slack}$")
    axes[1, 1].set_ylabel("MPC-PV\nutilization (%)")
    axes[1, 1].grid(True, which="major")
    for index, axis in enumerate(axes.flat):
        axis.text(0.01, 0.97, f"({chr(ord('a') + index)})", transform=axis.transAxes, ha="left", va="top", fontsize=14.0, fontweight="bold")
    legend_handles = [
        Line2D([], [], color=OBJECTIVE_COLORS["no_pv"], marker="o", linestyle="-", label="MPC-CA"),
        Line2D([], [], color=OBJECTIVE_COLORS["onsite_pv"], marker="s", linestyle="--", label="MPC-PV"),
        Line2D([], [], color="#555555", linestyle="-", label="Full day"),
        Line2D([], [], color="#555555", linestyle="--", label="Morning"),
        Line2D([], [], color="#555555", linestyle=":", label="Evening"),
        Line2D([], [], color="#D55E00", marker="o", linestyle="-", label="Self-consumption"),
        Line2D([], [], color="#8da0cb", marker="s", linestyle="--", label="Self-sufficiency"),
    ]
    figure.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=13.5)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
    return _save_figure(figure, output_dir, "discussion_weight_sensitivity")


def _copy_figure_pair(result: FigureResult, paper_fig_dir: Path) -> tuple[Path, Path]:
    destination = Path(paper_fig_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / result.png_path.name
    pdf = destination / result.pdf_path.name
    shutil.copy2(result.png_path, png)
    shutil.copy2(result.pdf_path, pdf)
    return png, pdf


def run_pipeline(
    *,
    sweep_dir: Path = DEFAULT_SWEEP_DIR,
    observed_context_path: Path | None = None,
    observed_pooled_path: Path = DEFAULT_OBSERVED_POOLED_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    paper_fig_dir: Path = DEFAULT_PAPER_FIG_DIR,
    validate_only: bool = False,
) -> PipelineResult:
    output_dir = Path(output_dir)
    paper_fig_dir = Path(paper_fig_dir)
    inputs = validate_sweep_run(sweep_dir, observed_context_path=observed_context_path, observed_pooled_path=observed_pooled_path)
    daily, pooled = compute_weight_metrics(inputs)
    selected = select_weight_profile_dates(daily)
    selected_dates = dict(zip(selected["controller_objective"], selected["date"]))
    if validate_only:
        return PipelineResult(True, int(daily["date"].nunique()), len(inputs.minute), inputs.baseline_gate, selected_dates, {})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, table, filename in (
        ("weight_minute_data", inputs.minute, "weight_minute_data.csv.gz"),
        ("weight_daily_metrics", daily, "weight_daily_metrics.csv"),
        ("weight_pooled_metrics", pooled, "weight_pooled_metrics.csv"),
        ("weight_selected_dates", selected, "weight_selected_dates.csv"),
    ):
        path = output / filename
        table.to_csv(path, index=False, float_format="%.12g")
        paths[key] = path
    figures = (
        render_weight_daily_profiles(inputs.minute, selected, output),
        render_weight_sensitivity(daily, output),
    )
    for result in figures:
        paths[result.png_path.stem + "_png"] = result.png_path
        paths[result.pdf_path.stem + "_pdf"] = result.pdf_path
        paper_png, paper_pdf = _copy_figure_pair(result, paper_fig_dir)
        paths[paper_png.stem + "_paper_png"] = paper_png
        paths[paper_pdf.stem + "_paper_pdf"] = paper_pdf
    metadata = inputs.metadata
    manifest = {
        "run_id": metadata.get("run_id"),
        "weights": list(SWEEP_WEIGHTS),
        "varied_parameter": "w_x_slack",
        "fixed_interpretation": "zone-temperature state-slack penalty; PMV stage weight unchanged",
        "runtime": {
            "device": metadata.get("runtime", {}).get("device"),
            "cuda_available": metadata.get("runtime", {}).get("cuda_available"),
            "torch": metadata.get("runtime", {}).get("torch"),
            "python": metadata.get("runtime", {}).get("python"),
        },
        "baseline_gate": inputs.baseline_gate,
        "energy": {
            "calculation": "manuscript heat balance",
            "audit": asdict(inputs.energy_audit),
            "ac27_is_demand_reduction_reference": True,
        },
        "pmv": {
            "ac_relative_humidity_pct": 65.0,
            "window_open_relative_humidity": "observed outdoor relative humidity",
        },
        "peak_windows": {
            "morning": "09:00--11:30, start-inclusive and end-exclusive",
            "evening": "16:00--18:30, start-inclusive and end-exclusive",
        },
        "selected_profiles": selected.to_dict("records"),
        "safety": inputs.safety,
        "sources": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in inputs.source_paths.items()
        }
        | {
            "observed_context": {
                "file": inputs.observed_context_path.name,
                "sha256": _sha256(inputs.observed_context_path),
            }
        },
        "outputs": {
            key: {"file": path.name, "sha256": _sha256(path)}
            for key, path in paths.items()
            if path.is_file() and path.parent == output
        },
    }
    manifest_path = output / "weight_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["weight_sensitivity_manifest"] = manifest_path
    return PipelineResult(True, int(daily["date"].nunique()), len(inputs.minute), inputs.baseline_gate, selected_dates, paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--observed-context", type=Path, default=Path("data/private/l14_merged_data_with_rain.csv"))
    parser.add_argument("--observed-pooled", type=Path, default=DEFAULT_OBSERVED_POOLED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_pipeline(
        sweep_dir=arguments.sweep_dir,
        observed_context_path=arguments.observed_context,
        observed_pooled_path=arguments.observed_pooled,
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
                "baseline_gate": result.baseline_gate,
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
