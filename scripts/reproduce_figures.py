#!/usr/bin/env python3
"""Check or rebuild the 17 numbered manuscript figures.

This entry point intentionally contains no study data.  It coordinates the
small figure-producing programs in this repository after the user supplies
the private observational data and, for Figures 6--16, the derived simulation
exports described in ``docs/DATA_REQUIREMENTS.md``.

``--list`` and ``--check`` are read-only and do not import the scientific
Python stack.  A normal run writes temporary products below
``outputs/figure_work`` and copies only the selected manuscript assets to
``figures``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "figures"
DEFAULT_WORK_DIR = ROOT / "outputs" / "figure_work"

PRIVATE_L14 = ROOT / "data" / "private" / "l14_merged_data_with_rain.csv"
PRIVATE_PV = ROOT / "data" / "private" / "pv_appendix_b.csv"
AC_MODEL = ROOT / "models" / "thermal" / "ac_model.pth"
NV_MODEL = ROOT / "models" / "thermal" / "nv_model.pth"
PV_MODEL = ROOT / "models" / "pv" / "pv_appendix_b_best_model.csv"
STATIC_FAN = ROOT / "assets" / "ceiling_fan.png"

MAIN_DIR = ROOT / "outputs" / "simulations" / "main"
MAIN_STEM = "future_data_source_comparison_test_val_union_all_full"
MAIN_TIMESERIES = MAIN_DIR / f"{MAIN_STEM}_timeseries.csv"
MAIN_DAILY = MAIN_DIR / f"{MAIN_STEM}_daily_summary.csv"
MAIN_AGGREGATE = MAIN_DIR / f"{MAIN_STEM}_aggregate_summary.csv"
MAIN_VALIDATION = MAIN_DIR / f"{MAIN_STEM}_validation.csv"
LSTM_METRICS = ROOT / "outputs" / "lstm64" / "metrics_aggregate.csv"
SWEEP_DIR = ROOT / "outputs" / "simulations" / "weight_sweep"

SCRIPT_MOTIVATION = ROOT / "scripts" / "make_aug23_mmv_motivation_figure.py"
SCRIPT_EMULATOR = ROOT / "scripts" / "build_emulator_diagram.py"
SCRIPT_ROLLOUT = ROOT / "scripts" / "compare_daily_rollout_methods.py"
SCRIPT_PMV = ROOT / "fit_all_pmv_regression.py"
SCRIPT_OBSERVED = ROOT / "scripts" / "build_observed_future_results.py"
SCRIPT_DISCUSSION = ROOT / "scripts" / "build_discussion_results.py"
SCRIPT_WEIGHT = ROOT / "scripts" / "build_weight_sensitivity_results.py"
SCRIPT_PV = ROOT / "scripts" / "build_pv_appendix_figure.py"


@dataclass(frozen=True)
class InputSpec:
    path: Path
    label: str
    columns: tuple[str, ...] = ()
    one_of: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class FigureSpec:
    number: int
    title: str
    producer: str
    outputs: tuple[str, ...]
    inputs: tuple[InputSpec, ...]
    stage: str


ZONE_COLUMNS = tuple(f"Zone {zone} Temperature" for zone in range(1, 6))
FCU_COLUMNS = tuple(f"fcu{unit:02d}_supply" for unit in range(1, 6))
PFCU_COLUMNS = tuple(f"pfc{unit:02d}_supply" for unit in range(1, 3))
TRAJECTORY_COLUMNS = (
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
)
MOTIVATION_COLUMNS = (
    "date",
    "Solar Radiation",
    "rain_status",
    "OutdoorTemperatureWindow",
    *(f"FCU-{unit} Cooling Load_kW" for unit in range(1, 6)),
    "PFCU-1 Cooling Load_kW",
    "PFCU-2 Cooling Load_kW",
    "Z7 Windows Open Close Status",
    "Z6 Windows Open Close Status",
    "Z5 Windows Open Close Status",
    "Z1 Windows Open Close Status",
    "Z2 Windows Open Close Status",
    "Z3 Windows Open Close Status",
)
ROLLOUT_COLUMNS = (
    "date",
    *ZONE_COLUMNS,
    "OutdoorTemperatureWindow",
    "FCU-01 Supply Air Temp",
    "FCU-02 Supply Air Temp - 1 min",
    "FCU-03 Supply Air Temp",
    "FCU-04 Supply Air Temp",
    "FCU-05 Supply Air Temp",
    "PFCU-01 Supply Air Temp",
    "PFCU-02 Supply Air Temp",
    "Wind Speed",
    "Wind Direction",
    "rain_status",
    "Solar Radiation",
    "Z1 Windows Open Close Status",
)
RH_ALTERNATIVES = tuple(
    (
        f"Zone {zone} RH",
        f"Zone {zone} Humidity",
        f"Zone {zone} Relative Humidity",
        f"FCU-{zone:02d} Return Air Humi",
    )
    for zone in range(1, 6)
)

L14_MOTIVATION = InputSpec(PRIVATE_L14, "private merged observations", MOTIVATION_COLUMNS)
L14_ROLLOUT = InputSpec(PRIVATE_L14, "private merged observations", ROLLOUT_COLUMNS)
L14_PMV = InputSpec(
    PRIVATE_L14,
    "private merged observations",
    ("date", *ZONE_COLUMNS),
    RH_ALTERNATIVES,
)
L14_CONTEXT = InputSpec(
    PRIVATE_L14,
    "private merged observations",
    ("date", "OutdoorHumidityWindow", "rain_status"),
)
MAIN_TRAJECTORY = InputSpec(MAIN_TIMESERIES, "derived main-study trajectories", TRAJECTORY_COLUMNS)
MAIN_DAILY_SPEC = InputSpec(
    MAIN_DAILY,
    "derived main-study daily summary",
    (
        "date",
        "case",
        "hvac_kwh",
        "pv_available_kwh",
        "grid_kwh",
        "export_kwh",
        "self_consumed_kwh",
        "self_consumption_ratio",
        "self_sufficiency_ratio",
        "switch_count",
    ),
)
MAIN_AGGREGATE_SPEC = InputSpec(
    MAIN_AGGREGATE,
    "derived main-study aggregate summary",
    (
        "case",
        "hvac_kwh",
        "pv_available_kwh",
        "grid_kwh",
        "export_kwh",
        "self_consumed_kwh",
        "self_consumption_ratio",
        "self_sufficiency_ratio",
    ),
)
MAIN_VALIDATION_SPEC = InputSpec(
    MAIN_VALIDATION,
    "derived main-study validation table",
    (
        "date",
        "case",
        "expected_rows",
        "actual_rows",
        "exact_index",
        "missing_required_columns",
        "required_missing_values",
        "rain_lockout_violations",
    ),
)
LSTM_METRICS_SPEC = InputSpec(
    LSTM_METRICS,
    "derived LSTM64 evaluation metrics",
    ("split", "model", "seed", "target", "metric", "value", "eligible_count", "coverage"),
)
SWEEP_SPECS = tuple(
    InputSpec(SWEEP_DIR / name, "derived temperature-slack sweep")
    for name in (
        "wx_sweep_timeseries.csv.gz",
        "wx_sweep_daily_metrics.csv",
        "wx_sweep_pooled_metrics.csv",
        "wx_sweep_validation.csv",
        "wx_sweep_run_metadata.json",
    )
)
PV_DATA_SPEC = InputSpec(
    PRIVATE_PV,
    "private aligned PV holdout inputs",
    ("timestamp", "pv_power_kw", "ghi_wm2", "outdoor_air_temp_c"),
)


FIGURES = (
    FigureSpec(1, "Measured MMV/PV motivation", str(SCRIPT_MOTIVATION.relative_to(ROOT)),
               ("mmv_motivation_20240823_15min_cooling.png",), (L14_MOTIVATION,), "motivation"),
    FigureSpec(2, "Mode-specific thermal emulator", str(SCRIPT_EMULATOR.relative_to(ROOT)),
               ("emulator_diagram.pdf",), (), "emulator"),
    FigureSpec(3, "Ceiling-fan source image (retained static asset)", "assets/ceiling_fan.png",
               ("ceiling_fan.png",), (InputSpec(STATIC_FAN, "retained cited static source"),), "static"),
    FigureSpec(4, "Nonlinear and affine thermal rollouts", str(SCRIPT_ROLLOUT.relative_to(ROOT)),
               ("2024-10-02_rollout.png", "2024-10-09_rollout.png", "2024-10-10_rollout.png"),
               (L14_ROLLOUT, InputSpec(AC_MODEL, "AC thermal model"), InputSpec(NV_MODEL, "NV thermal model")),
               "rollout"),
    FigureSpec(5, "Linear PMV approximation", str(SCRIPT_PMV.relative_to(ROOT)),
               ("all_temp_rh_pmv_regression.png",), (L14_PMV,), "pmv"),
    *(
        FigureSpec(number, f"Closed-loop trajectories on {date}", str(SCRIPT_OBSERVED.relative_to(ROOT)),
                   (f"observed_future_daily_{iso_date}.pdf",),
                   (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC, MAIN_VALIDATION_SPEC),
                   "observed")
        for number, date, iso_date in (
            (6, "24 September 2024", "2024-09-24"),
            (7, "3 October 2024", "2024-10-03"),
            (8, "4 October 2024", "2024-10-04"),
            (9, "10 October 2024", "2024-10-10"),
            (10, "15 October 2024", "2024-10-15"),
        )
    ),
    FigureSpec(11, "Daily energy-flexibility KPI distributions", str(SCRIPT_OBSERVED.relative_to(ROOT)),
               ("observed_future_energy_flexibility.pdf",),
               (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC, MAIN_VALIDATION_SPEC),
               "observed"),
    FigureSpec(12, "Perfect versus LSTM64 forecast trajectories", str(SCRIPT_DISCUSSION.relative_to(ROOT)),
               ("discussion_lstm64_daily_profiles.pdf",),
               (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC,
                MAIN_VALIDATION_SPEC, LSTM_METRICS_SPEC), "discussion"),
    FigureSpec(13, "Weather and flexibility associations", str(SCRIPT_DISCUSSION.relative_to(ROOT)),
               ("discussion_weather_flexibility.pdf",),
               (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC,
                MAIN_VALIDATION_SPEC, LSTM_METRICS_SPEC), "discussion"),
    FigureSpec(14, "Aggregate PV energy flows", str(SCRIPT_DISCUSSION.relative_to(ROOT)),
               ("discussion_pv_energy_flow.pdf",),
               (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC,
                MAIN_VALIDATION_SPEC, LSTM_METRICS_SPEC), "discussion"),
    FigureSpec(15, "Daily state-slack-weight response", str(SCRIPT_WEIGHT.relative_to(ROOT)),
               ("discussion_weight_daily_profiles.pdf",), (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC, MAIN_VALIDATION_SPEC, *SWEEP_SPECS), "weight"),
    FigureSpec(16, "Aggregate state-slack-weight sensitivity", str(SCRIPT_WEIGHT.relative_to(ROOT)),
               ("discussion_weight_sensitivity.pdf",), (L14_CONTEXT, MAIN_TRAJECTORY, MAIN_DAILY_SPEC, MAIN_AGGREGATE_SPEC, MAIN_VALIDATION_SPEC, *SWEEP_SPECS), "weight"),
    FigureSpec(17, "PV-model chronological holdout", str(SCRIPT_PV.relative_to(ROOT)),
               ("pv_appendix_b_holdout_plots.png",),
               (PV_DATA_SPEC, InputSpec(PV_MODEL, "PV coefficient model")), "pv"),
)

FIGURE_BY_NUMBER = {figure.number: figure for figure in FIGURES}
STAGE_SCRIPTS = {
    "motivation": SCRIPT_MOTIVATION,
    "emulator": SCRIPT_EMULATOR,
    "rollout": SCRIPT_ROLLOUT,
    "pmv": SCRIPT_PMV,
    "observed": SCRIPT_OBSERVED,
    "discussion": SCRIPT_DISCUSSION,
    "weight": SCRIPT_WEIGHT,
    "pv": SCRIPT_PV,
}


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_header(path: Path) -> set[str]:
    opener: Callable[..., object] = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:  # type: ignore[arg-type]
        row = next(csv.reader(handle), None)
    return set(row or ())


def _input_problems(spec: InputSpec) -> list[str]:
    if not spec.path.is_file():
        return [f"missing {_relative(spec.path)} ({spec.label})"]
    if not spec.columns and not spec.one_of:
        return []
    try:
        header = _read_header(spec.path)
    except (OSError, UnicodeError, csv.Error) as error:
        return [f"cannot read header of {_relative(spec.path)}: {error}"]
    problems: list[str] = []
    missing = [column for column in spec.columns if column not in header]
    if missing:
        problems.append(f"{_relative(spec.path)} missing columns: {', '.join(missing)}")
    for alternatives in spec.one_of:
        if not any(column in header for column in alternatives):
            problems.append(
                f"{_relative(spec.path)} needs one of: {', '.join(alternatives)}"
            )
    return problems


def _unique_inputs(figures: Iterable[FigureSpec]) -> list[InputSpec]:
    found: dict[Path, InputSpec] = {}
    for figure in figures:
        for spec in figure.inputs:
            previous = found.get(spec.path)
            if previous is None:
                found[spec.path] = spec
                continue
            columns = tuple(dict.fromkeys((*previous.columns, *spec.columns)))
            one_of = tuple(dict.fromkeys((*previous.one_of, *spec.one_of)))
            found[spec.path] = InputSpec(spec.path, previous.label, columns, one_of)
    return list(found.values())


def _producer_problems(figures: Iterable[FigureSpec]) -> list[str]:
    stages = {figure.stage for figure in figures}
    if "discussion" in stages:
        # The discussion producer consumes daily and pooled tables rebuilt by
        # the observed-future producer during the same orchestration run.
        stages.add("observed")
    problems = []
    for stage in sorted(stages):
        script = STAGE_SCRIPTS.get(stage)
        if script is not None and not script.is_file():
            problems.append(f"missing producer {_relative(script)}")
    if "rollout" in stages and not (ROOT / "mpc_miqp_simulation_fast.py").is_file():
        problems.append("missing producer support mpc_miqp_simulation_fast.py")
    if "pmv" in stages and not (ROOT / "fit_zone_pmv_regression.py").is_file():
        problems.append("missing producer support fit_zone_pmv_regression.py")
    return problems


def list_figures() -> None:
    print("No.  Stage       Release asset group -> output")
    for figure in FIGURES:
        output_text = ", ".join(f"figures/{name}" for name in figure.outputs)
        print(f"{figure.number:>2}   {figure.stage:<11} {figure.title} -> {output_text}")


def check_figures(figures: Sequence[FigureSpec]) -> None:
    print("Release structure:")
    producer_problems = _producer_problems(figures)
    if producer_problems:
        for problem in producer_problems:
            print(f"  [MISSING] {problem}")
    else:
        print("  [OK] selected figure producers and support modules are present")

    print("Inputs:")
    for spec in _unique_inputs(figures):
        problems = _input_problems(spec)
        if problems:
            for problem in problems:
                label = "PRIVATE/DERIVED" if "producer" not in problem else "MISSING"
                print(f"  [{label}] {problem}")
        else:
            print(f"  [OK] {_relative(spec.path)} ({spec.label})")

    print(
        "Check complete. Missing private or derived data is expected in a clean public clone; "
        "see docs/DATA_REQUIREMENTS.md. No files were written."
    )


def _run(command: Sequence[object]) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"\n> {printable}")
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def _copy(source: Path, output_dir: Path, name: str | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"expected producer output is missing: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (name or source.name)
    shutil.copy2(source, destination)
    print(f"  wrote {_relative(destination)}")


def _run_motivation(work_dir: Path, output_dir: Path) -> None:
    work = work_dir / "figure_01"
    _run(
        (
            sys.executable,
            SCRIPT_MOTIVATION,
            "--input",
            PRIVATE_L14,
            "--date",
            "2024-08-23",
            "--output-dir",
            work,
            "--resample-minutes",
            "15",
            "--power-scope",
            "cooling",
        )
    )
    _copy(work / "mmv_motivation_20240823_15min_cooling.png", output_dir)


def _run_emulator(work_dir: Path, output_dir: Path) -> None:
    work = work_dir / "figure_02"
    work.mkdir(parents=True, exist_ok=True)
    output = work / "emulator_diagram.pdf"
    module_spec = importlib.util.spec_from_file_location("mmv4ef_emulator_figure", SCRIPT_EMULATOR)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_EMULATOR}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.build(output)
    _copy(output, output_dir)


def _run_static(output_dir: Path) -> None:
    _copy(STATIC_FAN, output_dir, "ceiling_fan.png")


def _run_rollout(work_dir: Path, output_dir: Path) -> None:
    work = work_dir / "figure_04"
    _run(
        (
            sys.executable,
            SCRIPT_ROLLOUT,
            "--data",
            PRIVATE_L14,
            "--ac-model",
            AC_MODEL,
            "--nv-model",
            NV_MODEL,
            "--output-dir",
            work,
            "--splits",
            "val",
            "--five-min-rollouts",
            "held-input",
            "--device",
            "cpu",
        )
    )
    for date in ("2024-10-02", "2024-10-09", "2024-10-10"):
        matches = sorted((work / "daily_rollout" / "val").glob(f"*_{date}_fair_rollout.png"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one validation rollout for {date}, found {len(matches)}")
        _copy(matches[0], output_dir, f"{date}_rollout.png")


def _run_pmv(work_dir: Path, output_dir: Path) -> None:
    work = work_dir / "figure_05"
    _run(
        (
            sys.executable,
            SCRIPT_PMV,
            "--input",
            PRIVATE_L14,
            "--output-plot",
            work / "all_temp_rh_pmv_regression.png",
            "--output-3d-plot",
            work / "all_temp_rh_pmv_regression_3d.png",
            "--output-summary",
            work / "all_temp_rh_pmv_regression_summary.csv",
            "--output-zone-breakdown",
            work / "all_temp_rh_pmv_regression_zone_breakdown.csv",
            "--output-diagnostics-summary",
            work / "all_temp_rh_pmv_linear_diagnostics_summary.csv",
            "--output-diagnostics-plot",
            work / "all_temp_rh_pmv_linear_diagnostics.png",
        )
    )
    _copy(work / "all_temp_rh_pmv_regression.png", output_dir)


def _run_observed(work_dir: Path) -> Path:
    work = work_dir / "observed_future"
    _run(
        (
            sys.executable,
            SCRIPT_OBSERVED,
            "--timeseries",
            MAIN_TIMESERIES,
            "--observed-context",
            PRIVATE_L14,
            "--source-daily-summary",
            MAIN_DAILY,
            "--source-aggregate-summary",
            MAIN_AGGREGATE,
            "--source-validation",
            MAIN_VALIDATION,
            "--output-dir",
            work,
            "--paper-fig-dir",
            work / "paper_copy",
        )
    )
    return work


def _copy_observed(selected: set[int], work: Path, output_dir: Path) -> None:
    dates = {
        6: "2024-09-24",
        7: "2024-10-03",
        8: "2024-10-04",
        9: "2024-10-10",
        10: "2024-10-15",
    }
    for number, date in dates.items():
        if number in selected:
            _copy(work / "daily" / f"observed_future_daily_{date}.pdf", output_dir)
    if 11 in selected:
        _copy(work / "observed_future_energy_flexibility.pdf", output_dir)


def _run_discussion(work_dir: Path, observed_work: Path, output_dir: Path, selected: set[int]) -> None:
    work = work_dir / "discussion"
    _run(
        (
            sys.executable,
            SCRIPT_DISCUSSION,
            "--timeseries",
            MAIN_TIMESERIES,
            "--validation",
            MAIN_VALIDATION,
            "--observed-context",
            PRIVATE_L14,
            "--observed-daily",
            observed_work / "observed_future_daily_metrics.csv",
            "--observed-pooled",
            observed_work / "observed_future_pooled_metrics.csv",
            "--forecast-metrics",
            LSTM_METRICS,
            "--output-dir",
            work,
            "--paper-fig-dir",
            work / "paper_copy",
        )
    )
    names = {
        12: "discussion_lstm64_daily_profiles.pdf",
        13: "discussion_weather_flexibility.pdf",
        14: "discussion_pv_energy_flow.pdf",
    }
    for number, name in names.items():
        if number in selected:
            _copy(work / name, output_dir)


def _run_weight(work_dir: Path, output_dir: Path, selected: set[int]) -> None:
    work = work_dir / "weight_sensitivity"
    _run(
        (
            sys.executable,
            SCRIPT_WEIGHT,
            "--sweep-dir",
            SWEEP_DIR,
            "--observed-pooled",
            work_dir / "observed_future" / "observed_future_pooled_metrics.csv",
            "--observed-context",
            PRIVATE_L14,
            "--output-dir",
            work,
            "--paper-fig-dir",
            work / "paper_copy",
        )
    )
    names = {
        15: "discussion_weight_daily_profiles.pdf",
        16: "discussion_weight_sensitivity.pdf",
    }
    for number, name in names.items():
        if number in selected:
            _copy(work / name, output_dir)


def _run_pv(work_dir: Path, output_dir: Path) -> None:
    output = work_dir / "figure_17" / "pv_appendix_b_holdout_plots.png"
    _run(
        (
            sys.executable,
            SCRIPT_PV,
            "--dataset",
            PRIVATE_PV,
            "--best-model",
            PV_MODEL,
            "--output",
            output,
        )
    )
    _copy(output, output_dir)


def _parse_numbers(values: Sequence[str] | None) -> set[int]:
    if not values:
        return set(FIGURE_BY_NUMBER)
    numbers: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                numbers.update(range(start, end + 1))
            else:
                numbers.add(int(token))
    invalid = sorted(numbers - set(FIGURE_BY_NUMBER))
    if invalid:
        raise ValueError(f"figure numbers must be 1--17; invalid: {invalid}")
    return numbers


def reproduce(selected: set[int], work_dir: Path, output_dir: Path) -> None:
    chosen = [FIGURE_BY_NUMBER[number] for number in sorted(selected)]
    problems = _producer_problems(chosen)
    for spec in _unique_inputs(chosen):
        problems.extend(_input_problems(spec))
    if problems:
        detail = "\n  - ".join(problems)
        raise SystemExit(
            "Cannot reproduce the selected figures. Resolve these requirements first:\n"
            f"  - {detail}\nSee docs/DATA_REQUIREMENTS.md."
        )

    if 1 in selected:
        _run_motivation(work_dir, output_dir)
    if 2 in selected:
        _run_emulator(work_dir, output_dir)
    if 3 in selected:
        _run_static(output_dir)
    if 4 in selected:
        _run_rollout(work_dir, output_dir)
    if 5 in selected:
        _run_pmv(work_dir, output_dir)

    observed_work: Path | None = None
    if selected.intersection(range(6, 17)):
        observed_work = _run_observed(work_dir)
    if selected.intersection(range(6, 12)) and observed_work is not None:
        _copy_observed(selected, observed_work, output_dir)
    if selected.intersection({12, 13, 14}):
        if observed_work is None:
            raise RuntimeError("discussion figures require the observed-future stage")
        _run_discussion(work_dir, observed_work, output_dir, selected)
    if selected.intersection({15, 16}):
        _run_weight(work_dir, output_dir, selected)
    if 17 in selected:
        _run_pv(work_dir, output_dir)

    print(f"\nCompleted figures: {', '.join(map(str, sorted(selected)))}")
    print(f"Manuscript assets: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List all 17 figures and exit.")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Read-only check of producers, input paths, and available CSV headers.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        metavar="N",
        help="Figures to check/build (for example: 1 2 6-11 or 1,3,17). Default: all.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected = _parse_numbers(args.figures)
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    chosen = [FIGURE_BY_NUMBER[number] for number in sorted(selected)]
    if args.list:
        list_figures()
        return 0
    if args.check:
        check_figures(chosen)
        return 0
    reproduce(selected, args.work_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
