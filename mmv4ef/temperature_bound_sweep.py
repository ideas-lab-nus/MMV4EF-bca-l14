"""Numerical validation and Pareto helpers for the temperature sweep."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

import numpy as np
import pandas as pd


SWEEP_WEIGHTS = (100.0, 1000.0, 10000.0)
PARETO_THERMAL_ABS_TOL_DEGREE_MIN = 1e-6
PARETO_ENERGY_ABS_TOL_KWH = 1e-6
PARETO_REL_TOL = 1e-9

VALIDATION_FAILURE_COLUMNS = (
    "table",
    "column",
    "date",
    "case",
    "reason",
    "actual",
    "expected",
    "denominator",
)


@dataclass(frozen=True)
class DenominatorAwareRatioContract:
    ratio_column: str
    numerator_column: str
    denominator_column: str
    denominator_guard_gt: float = 0.0
    scale: float = 1.0


def _empty_validation_failures() -> pd.DataFrame:
    return pd.DataFrame(columns=VALIDATION_FAILURE_COLUMNS)


def _validation_failure(
    *,
    table_name: str,
    column: str,
    row: pd.Series | None,
    reason: str,
    actual=None,
    expected=None,
    denominator=None,
) -> dict:
    return {
        "table": table_name,
        "column": column,
        "date": None if row is None else row.get("date"),
        "case": None if row is None else row.get("case"),
        "reason": reason,
        "actual": actual,
        "expected": expected,
        "denominator": denominator,
    }


def find_nonfinite_numeric_failures(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> pd.DataFrame:
    """Return one diagnostic record per required non-finite numeric cell."""
    failures = []
    for column in columns:
        if column not in frame.columns:
            failures.append(
                _validation_failure(
                    table_name=table_name,
                    column=column,
                    row=None,
                    reason="missing_column",
                    expected="finite",
                )
            )
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        for position in np.flatnonzero(invalid):
            row = frame.iloc[int(position)]
            failures.append(
                _validation_failure(
                    table_name=table_name,
                    column=column,
                    row=row,
                    reason="nonfinite_numeric",
                    actual=row[column],
                    expected="finite",
                )
            )
    if not failures:
        return _empty_validation_failures()
    return pd.DataFrame(failures, columns=VALIDATION_FAILURE_COLUMNS)


def validate_denominator_aware_ratios(
    frame: pd.DataFrame,
    contracts: Sequence[DenominatorAwareRatioContract],
    *,
    table_name: str,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Validate guarded ratios while preserving intentional NaN semantics.

    A ratio must be finite and agree with ``scale * numerator / denominator``
    when the production denominator guard passes. Otherwise, its value must be
    missing; finite placeholders and infinities are contract violations.
    """
    failures = []
    for contract in contracts:
        required = (
            contract.ratio_column,
            contract.numerator_column,
            contract.denominator_column,
        )
        missing = [column for column in required if column not in frame.columns]
        if missing:
            failures.append(
                _validation_failure(
                    table_name=table_name,
                    column=contract.ratio_column,
                    row=None,
                    reason="missing_contract_columns",
                    actual=missing,
                    expected="all contract columns present",
                )
            )
            continue

        actual = pd.to_numeric(
            frame[contract.ratio_column], errors="coerce"
        ).to_numpy(dtype=float)
        numerator = pd.to_numeric(
            frame[contract.numerator_column], errors="coerce"
        ).to_numpy(dtype=float)
        denominator = pd.to_numeric(
            frame[contract.denominator_column], errors="coerce"
        ).to_numpy(dtype=float)
        available = denominator > float(contract.denominator_guard_gt)
        expected = np.full(len(frame), np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            expected[available] = (
                float(contract.scale)
                * numerator[available]
                / denominator[available]
            )

        for position in range(len(frame)):
            row = frame.iloc[position]
            if available[position]:
                if not np.isfinite(actual[position]):
                    reason = "available_ratio_nonfinite"
                elif not np.isfinite(expected[position]):
                    reason = "available_expected_nonfinite"
                elif not np.isclose(
                    actual[position],
                    expected[position],
                    rtol=float(relative_tolerance),
                    atol=float(absolute_tolerance),
                ):
                    reason = "available_ratio_mismatch"
                else:
                    continue
            else:
                raw_actual = row[contract.ratio_column]
                if bool(pd.isna(raw_actual)):
                    continue
                reason = "unavailable_ratio_not_nan"

            failures.append(
                _validation_failure(
                    table_name=table_name,
                    column=contract.ratio_column,
                    row=row,
                    reason=reason,
                    actual=row[contract.ratio_column],
                    expected=(
                        expected[position] if available[position] else np.nan
                    ),
                    denominator=denominator[position],
                )
            )

    if not failures:
        return _empty_validation_failures()
    return pd.DataFrame(failures, columns=VALIDATION_FAILURE_COLUMNS)


def _json_safe_validation_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_validation_value(item) for item in value]
    return str(value)


def format_validation_failures(
    failures: pd.DataFrame,
    *,
    sample_limit: int = 3,
) -> str:
    """Format compact, deterministic validation diagnostics for CSV output."""
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    if failures.empty:
        return "[]"
    required = set(VALIDATION_FAILURE_COLUMNS)
    missing = sorted(required - set(failures.columns))
    if missing:
        raise ValueError(f"Validation failures are missing columns: {missing}")

    payload = []
    for (table_name, column), group in failures.groupby(
        ["table", "column"], sort=False, dropna=False
    ):
        sample = []
        for _, row in group.head(sample_limit).iterrows():
            sample.append(
                {
                    key: _json_safe_validation_value(row[key])
                    for key in (
                        "date",
                        "case",
                        "reason",
                        "actual",
                        "expected",
                        "denominator",
                    )
                }
            )
        payload.append(
            {
                "table": _json_safe_validation_value(table_name),
                "column": _json_safe_validation_value(column),
                "count": int(len(group)),
                "sample": sample,
            }
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class SolverRetryDiagnostics:
    initial_status: int
    final_status: int
    numeric_retry_used: bool
    retry_count: int
    initial_runtime_s: float
    retry_runtime_s: float
    inf_or_unbd_disambiguation_used: bool
    inf_or_unbd_disambiguation_count: int
    inf_or_unbd_disambiguation_runtime_s: float
    total_runtime_s: float
    retry_numeric_focus: int | None


class SolverStatusError(RuntimeError):
    def __init__(
        self,
        message: str,
        diagnostics: SolverRetryDiagnostics,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _solver_status_label(grb, status: int) -> str:
    labels = {
        int(grb.OPTIMAL): "OPTIMAL",
        int(grb.INFEASIBLE): "INFEASIBLE",
        int(grb.INF_OR_UNBD): "INF_OR_UNBD",
        int(grb.UNBOUNDED): "UNBOUNDED",
        int(grb.TIME_LIMIT): "TIME_LIMIT",
        int(grb.INTERRUPTED): "INTERRUPTED",
        int(grb.NUMERIC): "NUMERIC",
        int(grb.SUBOPTIMAL): "SUBOPTIMAL",
    }
    return labels.get(int(status), str(int(status)))


def optimize_miqp_with_numeric_retry(
    model, grb, *, verbose: bool = False
) -> SolverRetryDiagnostics:
    """Optimize and apply one mutually exclusive status-specific recovery."""
    model.optimize()
    initial_status = int(model.Status)
    initial_runtime_s = float(model.Runtime)
    numeric_retry_used = initial_status == int(grb.NUMERIC)
    inf_or_unbd_disambiguation_used = initial_status == int(grb.INF_OR_UNBD)
    retry_runtime_s = 0.0
    inf_or_unbd_disambiguation_runtime_s = 0.0
    retry_numeric_focus = None

    if numeric_retry_used:
        retry_numeric_focus = 3
        model.Params.NumericFocus = retry_numeric_focus
        model.reset()
        model.optimize()
        final_status = int(model.Status)
        retry_runtime_s = float(model.Runtime)
    elif inf_or_unbd_disambiguation_used:
        if verbose:
            print(
                "[MPC] Solver returned INF_OR_UNBD; retrying with "
                "DualReductions=0 to disambiguate."
            )
        model.Params.DualReductions = 0
        model.reset()
        model.optimize()
        final_status = int(model.Status)
        inf_or_unbd_disambiguation_runtime_s = float(model.Runtime)
    else:
        final_status = initial_status

    diagnostics = SolverRetryDiagnostics(
        initial_status=initial_status,
        final_status=final_status,
        numeric_retry_used=numeric_retry_used,
        retry_count=int(numeric_retry_used),
        initial_runtime_s=initial_runtime_s,
        retry_runtime_s=retry_runtime_s,
        inf_or_unbd_disambiguation_used=inf_or_unbd_disambiguation_used,
        inf_or_unbd_disambiguation_count=int(
            inf_or_unbd_disambiguation_used
        ),
        inf_or_unbd_disambiguation_runtime_s=(
            inf_or_unbd_disambiguation_runtime_s
        ),
        total_runtime_s=(
            initial_runtime_s
            + retry_runtime_s
            + inf_or_unbd_disambiguation_runtime_s
        ),
        retry_numeric_focus=retry_numeric_focus,
    )
    final_solution_count = int(model.SolCount)
    accepted_status = final_status in (int(grb.OPTIMAL), int(grb.SUBOPTIMAL))
    accepted_time_limit_incumbent = (
        final_status == int(grb.TIME_LIMIT) and final_solution_count > 0
    )
    if not (accepted_status or accepted_time_limit_incumbent):
        raise SolverStatusError(
            "MPC solve failed: "
            f"initial status {initial_status} "
            f"({_solver_status_label(grb, initial_status)}); "
            f"final status {final_status} "
            f"({_solver_status_label(grb, final_status)}); "
            f"final solution count {final_solution_count}; "
            f"numeric retry count {diagnostics.retry_count}; "
            "INF_OR_UNBD disambiguation count "
            f"{diagnostics.inf_or_unbd_disambiguation_count}",
            diagnostics,
        )
    return diagnostics


def temperature_metrics(
    frame: pd.DataFrame,
    zone_columns: Sequence[str],
    bound_c: float = 30.0,
    exclude_first_minutes: int = 0,
) -> dict[str, int | float]:
    excluded = int(exclude_first_minutes)
    if excluded < 0:
        raise ValueError("exclude_first_minutes must be non-negative")
    selected = frame.iloc[excluded:]
    values = selected.loc[:, list(zone_columns)].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("temperature metrics require finite non-empty zone data")
    above = np.maximum(values - float(bound_c), 0.0)
    any_above = np.any(above > 0.0, axis=1)
    z_mode = selected["z"].astype(int).to_numpy()
    return {
        "any_zone_minutes_above_30": int(any_above.sum()),
        "zone_degree_minutes_above_30": float(above.sum()),
        "mean_zone_minutes_above_30": int(
            (selected["T_mean"] > bound_c).sum()
        ),
        "max_zone_temp_c": float(values.max()),
        "max_excursion_above_30_c": float(above.max()),
        "closed_window_violation_minutes": int(
            (any_above & (z_mode == 1)).sum()
        ),
        "window_open_violation_minutes": int(
            (any_above & (z_mode == 0)).sum()
        ),
    }


def _numeric_axis(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Pareto axis must contain only finite values: {column}")
    return values


def _axis_tolerance(
    values: np.ndarray, abs_tol: float, rel_tol: float = PARETO_REL_TOL
) -> float:
    if values.size == 0:
        return float(abs_tol)
    scale = float(np.max(np.abs(values)))
    return max(float(abs_tol), float(rel_tol) * scale)


def _axis_response(
    values: np.ndarray, abs_tol: float, rel_tol: float = PARETO_REL_TOL
) -> dict[str, float | bool]:
    tolerance = _axis_tolerance(values, abs_tol, rel_tol)
    spread = 0.0 if values.size == 0 else float(values.max() - values.min())
    return {
        "spread": spread,
        "tolerance": tolerance,
        "no_material_response": bool(spread <= tolerance),
    }


def _normalize_axis(
    values: np.ndarray,
    abs_tol: float,
    rel_tol: float = PARETO_REL_TOL,
) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    lower = float(values.min())
    span = float(values.max()) - lower
    if span <= _axis_tolerance(values, abs_tol, rel_tol):
        return np.zeros(values.shape, dtype=float)
    return (values - lower) / span


def _display_axis(
    values: np.ndarray,
    abs_tol: float,
    rel_tol: float = PARETO_REL_TOL,
) -> np.ndarray:
    """Collapse a no-response axis without changing its raw metric column."""
    response = _axis_response(values, abs_tol, rel_tol)
    if values.size and response["no_material_response"]:
        return np.full(values.shape, float(np.mean(values)), dtype=float)
    return values.astype(float, copy=True)


def _distinct_points(
    frame: pd.DataFrame,
    thermal_col: str,
    energy_col: str,
    thermal_abs_tol: float,
    energy_abs_tol: float,
    rel_tol: float,
) -> pd.DataFrame:
    """Return deterministic representatives of tolerance-equivalent points."""
    if frame.empty:
        return frame.copy()
    work = frame.loc[:, ["w_x_slack", thermal_col, energy_col]].copy()
    thermal = _numeric_axis(work, thermal_col)
    energy = _numeric_axis(work, energy_col)
    thermal_tolerance = _axis_tolerance(
        thermal, thermal_abs_tol, rel_tol
    )
    energy_tolerance = _axis_tolerance(energy, energy_abs_tol, rel_tol)
    work["_thermal_normalized"] = _normalize_axis(
        thermal, thermal_abs_tol, rel_tol
    )
    work["_energy_normalized"] = _normalize_axis(
        energy, energy_abs_tol, rel_tol
    )
    work = work.sort_values(
        ["_thermal_normalized", "_energy_normalized", "w_x_slack"],
        kind="stable",
    ).reset_index(drop=True)
    retained_positions: list[int] = []
    for position, row in work.iterrows():
        equivalent = any(
            abs(float(row[thermal_col]) - float(work.iloc[other][thermal_col]))
            <= thermal_tolerance
            and abs(
                float(row[energy_col]) - float(work.iloc[other][energy_col])
            )
            <= energy_tolerance
            for other in retained_positions
        )
        if not equivalent:
            retained_positions.append(int(position))
    return work.iloc[retained_positions].reset_index(drop=True)


def pareto_frontier(
    frame: pd.DataFrame,
    thermal_col: str,
    energy_col: str,
    *,
    thermal_abs_tol: float = PARETO_THERMAL_ABS_TOL_DEGREE_MIN,
    energy_abs_tol: float = PARETO_ENERGY_ABS_TOL_KWH,
    rel_tol: float = PARETO_REL_TOL,
) -> pd.DataFrame:
    """Return tolerance-aware non-dominated rows, ordered by weight."""
    if frame.empty:
        return frame.copy().sort_values("w_x_slack").reset_index(drop=True)
    thermal = _numeric_axis(frame, thermal_col)
    energy = _numeric_axis(frame, energy_col)
    thermal_tolerance = _axis_tolerance(
        thermal, thermal_abs_tol, rel_tol
    )
    energy_tolerance = _axis_tolerance(energy, energy_abs_tol, rel_tol)
    retained = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        no_worse = (
            (thermal <= thermal[index] + thermal_tolerance)
            & (energy <= energy[index] + energy_tolerance)
        )
        strictly_better = (
            (thermal < thermal[index] - thermal_tolerance)
            | (energy < energy[index] - energy_tolerance)
        )
        retained[index] = not bool(np.any(no_worse & strictly_better))
    return (
        frame.loc[retained]
        .copy()
        .sort_values("w_x_slack", kind="stable")
        .reset_index(drop=True)
    )


def geometric_knee(
    frontier: pd.DataFrame,
    thermal_col: str,
    energy_col: str,
    *,
    thermal_abs_tol: float = PARETO_THERMAL_ABS_TOL_DEGREE_MIN,
    energy_abs_tol: float = PARETO_ENERGY_ABS_TOL_KWH,
    rel_tol: float = PARETO_REL_TOL,
) -> float | None:
    """Select the point furthest from the normalized endpoint chord."""
    if frontier.empty:
        return None
    work = _distinct_points(
        frontier,
        thermal_col,
        energy_col,
        thermal_abs_tol,
        energy_abs_tol,
        rel_tol,
    )
    if len(work) < 3:
        return None
    points = work.loc[
        :, ["_thermal_normalized", "_energy_normalized"]
    ].to_numpy(dtype=float)
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length == 0.0:
        return None
    offsets = points[1:-1] - points[0]
    distances = np.abs(
        chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    ) / chord_length
    knee_index = int(np.argmax(distances)) + 1
    return float(work.iloc[knee_index]["w_x_slack"])


def _validate_complete_pairing(pooled: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "controller_objective",
        "w_x_slack",
        "zone_degree_minutes_above_30",
        "hvac_kwh",
        "self_consumption_ratio",
        "self_sufficiency_ratio",
    }
    missing_columns = sorted(required_columns - set(pooled.columns))
    if missing_columns:
        raise ValueError(
            "complete 2 x 3 controller-weight table requires columns: "
            + ", ".join(missing_columns)
        )
    annotated = pooled.copy()
    annotated["w_x_slack"] = pd.to_numeric(
        annotated["w_x_slack"], errors="coerce"
    ).astype(float)
    expected_pairs = {
        (objective, weight)
        for objective in ("no_pv", "onsite_pv")
        for weight in SWEEP_WEIGHTS
    }
    actual_pairs = list(
        zip(
            annotated["controller_objective"],
            annotated["w_x_slack"],
            strict=True,
        )
    )
    if (
        len(actual_pairs) != len(expected_pairs)
        or len(set(actual_pairs)) != len(actual_pairs)
        or set(actual_pairs) != expected_pairs
    ):
        raise ValueError(
            "balanced selection requires a complete 2 x 3 "
            "controller-weight pairing"
        )
    for column in ("zone_degree_minutes_above_30", "hvac_kwh"):
        annotated[column] = pd.to_numeric(annotated[column], errors="coerce")
        if not np.isfinite(annotated[column].to_numpy(dtype=float)).all():
            raise ValueError(f"balanced selection requires finite {column}")
    objective_order = pd.Categorical(
        annotated["controller_objective"],
        categories=["no_pv", "onsite_pv"],
        ordered=True,
    )
    return (
        annotated.assign(_objective_order=objective_order)
        .sort_values(["_objective_order", "w_x_slack"], kind="stable")
        .drop(columns="_objective_order")
        .reset_index(drop=True)
    )


def build_balanced_selection(
    pooled: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Annotate the complete sweep and select controller/common Pareto knees."""
    thermal_col = "zone_degree_minutes_above_30"
    energy_col = "hvac_kwh"
    annotated = _validate_complete_pairing(pooled)
    annotated["thermal_normalized"] = np.nan
    annotated["energy_normalized"] = np.nan
    annotated["thermal_pareto_display"] = np.nan
    annotated["energy_pareto_display"] = np.nan
    annotated["controller_pareto"] = False
    annotated["controller_knee"] = False

    individual: dict[str, dict[str, object]] = {}
    controller_frontier_weights: dict[str, set[float]] = {}
    for objective in ("no_pv", "onsite_pv"):
        mask = annotated["controller_objective"] == objective
        controller = annotated.loc[mask].copy()
        thermal_values = controller[thermal_col].to_numpy(dtype=float)
        energy_values = controller[energy_col].to_numpy(dtype=float)
        thermal_response = _axis_response(
            thermal_values, PARETO_THERMAL_ABS_TOL_DEGREE_MIN
        )
        energy_response = _axis_response(
            energy_values, PARETO_ENERGY_ABS_TOL_KWH
        )
        annotated.loc[mask, "thermal_normalized"] = _normalize_axis(
            thermal_values, PARETO_THERMAL_ABS_TOL_DEGREE_MIN
        )
        annotated.loc[mask, "energy_normalized"] = _normalize_axis(
            energy_values, PARETO_ENERGY_ABS_TOL_KWH
        )
        annotated.loc[mask, "thermal_pareto_display"] = _display_axis(
            thermal_values, PARETO_THERMAL_ABS_TOL_DEGREE_MIN
        )
        annotated.loc[mask, "energy_pareto_display"] = _display_axis(
            energy_values, PARETO_ENERGY_ABS_TOL_KWH
        )
        frontier = pareto_frontier(controller, thermal_col, energy_col)
        frontier_weights = {
            float(weight) for weight in frontier["w_x_slack"].tolist()
        }
        knee = geometric_knee(frontier, thermal_col, energy_col)
        distinct_frontier_point_count = len(
            _distinct_points(
                frontier,
                thermal_col,
                energy_col,
                PARETO_THERMAL_ABS_TOL_DEGREE_MIN,
                PARETO_ENERGY_ABS_TOL_KWH,
                PARETO_REL_TOL,
            )
        )
        controller_frontier_weights[objective] = frontier_weights
        annotated.loc[
            mask & annotated["w_x_slack"].isin(frontier_weights),
            "controller_pareto",
        ] = True
        if knee is not None:
            annotated.loc[
                mask & (annotated["w_x_slack"] == knee), "controller_knee"
            ] = True
        individual[objective] = {
            "frontier_weights": sorted(frontier_weights),
            "knee_weight": knee,
            "distinct_frontier_point_count": distinct_frontier_point_count,
            "thermal_axis": thermal_response,
            "energy_axis": energy_response,
            "no_material_response": bool(
                thermal_response["no_material_response"]
                and energy_response["no_material_response"]
            ),
        }

    combined = (
        annotated.groupby("w_x_slack", as_index=False, sort=True)[
            ["thermal_normalized", "energy_normalized"]
        ]
        .mean()
        .rename(
            columns={
                "thermal_normalized": "combined_thermal_normalized",
                "energy_normalized": "combined_energy_normalized",
            }
        )
    )
    combined_frontier = pareto_frontier(
        combined,
        "combined_thermal_normalized",
        "combined_energy_normalized",
        thermal_abs_tol=0.0,
        energy_abs_tol=0.0,
    )
    combined_frontier_weights = {
        float(weight) for weight in combined_frontier["w_x_slack"].tolist()
    }
    provisional_knee = geometric_knee(
        combined_frontier,
        "combined_thermal_normalized",
        "combined_energy_normalized",
        thermal_abs_tol=0.0,
        energy_abs_tol=0.0,
    )
    combined_thermal = combined["combined_thermal_normalized"].to_numpy(
        dtype=float
    )
    combined_energy = combined["combined_energy_normalized"].to_numpy(
        dtype=float
    )
    combined_thermal_response = _axis_response(combined_thermal, 0.0)
    combined_energy_response = _axis_response(combined_energy, 0.0)
    combined_distinct_point_count = len(
        _distinct_points(
            combined_frontier,
            "combined_thermal_normalized",
            "combined_energy_normalized",
            0.0,
            0.0,
            PARETO_REL_TOL,
        )
    )
    annotated = annotated.merge(
        combined, on="w_x_slack", how="left", validate="many_to_one"
    )
    annotated["combined_pareto"] = annotated["w_x_slack"].isin(
        combined_frontier_weights
    )
    annotated["provisional_common_knee"] = (
        False
        if provisional_knee is None
        else annotated["w_x_slack"] == provisional_knee
    )
    dominated_controllers = (
        []
        if provisional_knee is None
        else [
            objective
            for objective in ("no_pv", "onsite_pv")
            if provisional_knee not in controller_frontier_weights[objective]
        ]
    )
    annotated["common_knee_dominated_for_controller"] = False
    if provisional_knee is not None:
        annotated.loc[
            (annotated["w_x_slack"] == provisional_knee)
            & annotated["controller_objective"].isin(dominated_controllers),
            "common_knee_dominated_for_controller",
        ] = True
    common_knee = None if dominated_controllers else provisional_knee

    annotated["sc_change_pp"] = np.nan
    annotated["ss_change_pp"] = np.nan
    annotated["sc_warning"] = False
    annotated["ss_warning"] = False
    pv_mask = annotated["controller_objective"] == "onsite_pv"
    pv_ratios = annotated.loc[
        pv_mask, ["self_consumption_ratio", "self_sufficiency_ratio"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(pv_ratios.to_numpy(dtype=float)).all():
        raise ValueError("balanced selection requires finite MPC-PV SC/SS ratios")
    baseline = annotated.loc[
        pv_mask & (annotated["w_x_slack"] == SWEEP_WEIGHTS[0]),
        ["self_consumption_ratio", "self_sufficiency_ratio"],
    ].iloc[0]
    sc_change = np.round(
        (pv_ratios["self_consumption_ratio"] - float(baseline.iloc[0]))
        * 100.0,
        10,
    )
    ss_change = np.round(
        (pv_ratios["self_sufficiency_ratio"] - float(baseline.iloc[1]))
        * 100.0,
        10,
    )
    annotated.loc[pv_mask, "sc_change_pp"] = sc_change.to_numpy()
    annotated.loc[pv_mask, "ss_change_pp"] = ss_change.to_numpy()
    annotated.loc[pv_mask, "sc_warning"] = sc_change.to_numpy() < -2.0
    annotated.loc[pv_mask, "ss_warning"] = ss_change.to_numpy() < -2.0
    annotated["pv_metric_warning"] = (
        annotated["sc_warning"] | annotated["ss_warning"]
    )
    warning_weights = sorted(
        float(weight)
        for weight in annotated.loc[
            pv_mask & annotated["pv_metric_warning"], "w_x_slack"
        ].tolist()
    )

    summary: dict[str, object] = {
        "numerical_tolerances": {
            "thermal_abs_degree_min": (
                PARETO_THERMAL_ABS_TOL_DEGREE_MIN
            ),
            "energy_abs_kwh": PARETO_ENERGY_ABS_TOL_KWH,
            "relative": PARETO_REL_TOL,
            "relative_scale": "maximum_absolute_axis_value",
            "axis_tolerance_rule": "max(abs_tol, relative * scale)",
        },
        "individual": individual,
        "combined_frontier_weights": sorted(combined_frontier_weights),
        "combined_response": {
            "distinct_frontier_point_count": combined_distinct_point_count,
            "thermal_axis": combined_thermal_response,
            "energy_axis": combined_energy_response,
            "no_material_response": bool(
                combined_thermal_response["no_material_response"]
                and combined_energy_response["no_material_response"]
            ),
        },
        "no_material_response": all(
            individual[objective]["no_material_response"]
            for objective in ("no_pv", "onsite_pv")
        ),
        "provisional_common_knee_weight": provisional_knee,
        "common_knee_weight": common_knee,
        "provisional_common_knee_dominated": bool(dominated_controllers),
        "common_knee_dominated_controllers": dominated_controllers,
        "pv_metric_warning": bool(warning_weights),
        "pv_metric_warning_weights": warning_weights,
    }
    return annotated, summary
