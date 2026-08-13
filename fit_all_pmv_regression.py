#!/usr/bin/env python3
"""Fit PMV regression models using all zones pooled into one dataset.

This is the pooled counterpart to fit_zone_pmv_regression.py. It stacks
Zone 1-5 temperature/RH/PMV samples into one long table, fits one shared:
1) Multiple linear regression: PMV ~ temp + rh
2) Second-order regression: PMV ~ temp + rh + temp^2 + rh^2 + temp*rh

The per-zone script is intentionally left untouched so its existing outputs
remain comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fit_zone_pmv_regression import (
    ZONE_IDS,
    add_local_air_speed,
    breusch_pagan_test,
    build_design_matrix,
    build_points,
    compute_pmv,
    compute_vif_table,
    fit_ols,
    ols_standardized_residuals,
    read_dataset,
    resolve_rh_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one pooled PMV regression across all zone samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/private/l14_merged_data_with_rain.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("figures/all_temp_rh_pmv_regression.png"),
        help="Output regression plot path.",
    )
    parser.add_argument(
        "--output-3d-plot",
        type=Path,
        default=Path("figures/all_temp_rh_pmv_regression_3d.png"),
        help="Output 3D plot path with ground-truth PMV scatter and fitted PMV surface.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/analysis/pmv/all_temp_rh_pmv_regression_summary.csv"),
        help="Output CSV summary path for the pooled fit.",
    )
    parser.add_argument(
        "--output-zone-breakdown",
        type=Path,
        default=Path("outputs/analysis/pmv/all_temp_rh_pmv_regression_zone_breakdown.csv"),
        help="Output CSV with pooled-model metrics evaluated separately by zone.",
    )
    parser.add_argument(
        "--output-diagnostics-summary",
        type=Path,
        default=Path("outputs/analysis/pmv/all_temp_rh_pmv_linear_diagnostics_summary.csv"),
        help="Output CSV path for pooled linear-model OLS diagnostics.",
    )
    parser.add_argument(
        "--output-diagnostics-plot",
        type=Path,
        default=Path("outputs/analysis/pmv/all_temp_rh_pmv_linear_diagnostics.png"),
        help="Output plot path for pooled linear-model OLS diagnostics.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--met",
        type=float,
        default=1.2,
        help="Metabolic rate in met.",
    )
    parser.add_argument(
        "--clo",
        type=float,
        default=0.5,
        help="Clothing insulation in clo.",
    )
    parser.add_argument(
        "--min-rh",
        type=float,
        default=20.0,
        help="Minimum valid RH percentage retained before fitting/plotting.",
    )
    parser.add_argument(
        "--max-rh",
        type=float,
        default=100.0,
        help="Maximum valid RH percentage retained before fitting/plotting.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["adj_r2", "rmse"],
        default="adj_r2",
        help="Metric for choosing linear vs second-order.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=80000,
        help="Maximum pooled points drawn in the regression scatter plot.",
    )
    parser.add_argument(
        "--max-3d-scatter-points",
        type=int,
        default=12000,
        help="Maximum ground-truth points drawn in the 3D regression plot.",
    )
    parser.add_argument(
        "--max-diagnostic-points",
        type=int,
        default=15000,
        help="Maximum pooled points drawn in residual diagnostic plots.",
    )
    parser.add_argument(
        "--shapiro-max-n",
        type=int,
        default=5000,
        help="Maximum residual sample size used for Shapiro-Wilk normality test.",
    )
    return parser.parse_args()


def apply_date_filter(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    out = df
    if start_date:
        start = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
        out = out[out["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[out["date"] <= end]
    if out.empty:
        raise ValueError("No records left after date filtering.")
    return out


def apply_rh_filter(points: pd.DataFrame, min_rh: float, max_rh: float) -> pd.DataFrame:
    if min_rh > max_rh:
        raise ValueError("--min-rh must be less than or equal to --max-rh.")
    out = points[points["rh_pct"].between(min_rh, max_rh)].copy()
    if out.empty:
        raise ValueError("No PMV points left after RH filtering.")
    return out.reset_index(drop=True)


def predict_from_beta(temp: np.ndarray, rh: np.ndarray, beta: Sequence[float]) -> np.ndarray:
    beta_arr = np.asarray(beta, dtype=float)
    if len(beta_arr) == 3:
        return beta_arr[0] + beta_arr[1] * temp + beta_arr[2] * rh
    if len(beta_arr) == 6:
        return (
            beta_arr[0]
            + beta_arr[1] * temp
            + beta_arr[2] * rh
            + beta_arr[3] * temp * temp
            + beta_arr[4] * rh * rh
            + beta_arr[5] * temp * rh
        )
    raise ValueError(f"Expected 3 or 6 coefficients, got {len(beta_arr)}.")


def metric_dict(y: np.ndarray, y_hat: np.ndarray, n_features: int) -> dict[str, float | int]:
    mask = np.isfinite(y) & np.isfinite(y_hat)
    if not np.any(mask):
        raise ValueError("No finite predictions to evaluate.")
    y_valid = y[mask]
    y_hat_valid = y_hat[mask]
    residual = y_valid - y_hat_valid
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_valid - y_valid.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    n = int(mask.sum())
    adj_r2 = (
        1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
        if n > n_features + 1
        else np.nan
    )
    abs_resid = np.abs(residual)
    return {
        "n_eval": n,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(abs_resid)),
        "bias": float(np.mean(y_hat_valid - y_valid)),
        "p95_abs_error": float(np.quantile(abs_resid, 0.95)),
        "max_abs_error": float(np.max(abs_resid)),
    }


def fit_all(points: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    t = points["temp_c"].to_numpy(dtype=float)
    rh = points["rh_pct"].to_numpy(dtype=float)
    y = points["pmv"].to_numpy(dtype=float)

    X_lin = build_design_matrix(t, rh, second_order=False)
    lin_beta, _, lin_r2, lin_adj_r2, lin_rmse = fit_ols(X_lin, y)

    X_quad = build_design_matrix(t, rh, second_order=True)
    quad_beta, _, quad_r2, quad_adj_r2, quad_rmse = fit_ols(X_quad, y)

    if selection_metric == "adj_r2":
        best_model = "second_order" if quad_adj_r2 > lin_adj_r2 else "linear"
    else:
        best_model = "second_order" if quad_rmse < lin_rmse else "linear"

    row = {
        "scope": "all_zones",
        "zones": ",".join(str(z) for z in ZONE_IDS),
        "n_points": int(len(points)),
        "zone_count": int(points["zone"].nunique()),
        "temp_mean_c": float(points["temp_c"].mean()),
        "temp_median_c": float(points["temp_c"].median()),
        "rh_mean_pct": float(points["rh_pct"].mean()),
        "rh_median_pct": float(points["rh_pct"].median()),
        "v_mean_mps": float(points["v_local_mps"].mean()),
        "v_median_mps": float(points["v_local_mps"].median()),
        "pmv_mean": float(points["pmv"].mean()),
        "pmv_median": float(points["pmv"].median()),
        "linear_r2": lin_r2,
        "linear_adj_r2": lin_adj_r2,
        "linear_rmse": lin_rmse,
        "linear_beta_0": lin_beta[0],
        "linear_beta_temp": lin_beta[1],
        "linear_beta_rh": lin_beta[2],
        "quad_r2": quad_r2,
        "quad_adj_r2": quad_adj_r2,
        "quad_rmse": quad_rmse,
        "quad_beta_0": quad_beta[0],
        "quad_beta_temp": quad_beta[1],
        "quad_beta_rh": quad_beta[2],
        "quad_beta_temp2": quad_beta[3],
        "quad_beta_rh2": quad_beta[4],
        "quad_beta_temp_rh": quad_beta[5],
        "best_model": best_model,
        "best_adj_r2": quad_adj_r2 if best_model == "second_order" else lin_adj_r2,
        "best_rmse": quad_rmse if best_model == "second_order" else lin_rmse,
    }
    return pd.DataFrame([row])


def beta_from_summary(summary_row: pd.Series, family: str) -> tuple[list[float], str]:
    if family == "linear":
        return (
            [
                summary_row["linear_beta_0"],
                summary_row["linear_beta_temp"],
                summary_row["linear_beta_rh"],
            ],
            "linear",
        )
    if family == "second_order":
        return (
            [
                summary_row["quad_beta_0"],
                summary_row["quad_beta_temp"],
                summary_row["quad_beta_rh"],
                summary_row["quad_beta_temp2"],
                summary_row["quad_beta_rh2"],
                summary_row["quad_beta_temp_rh"],
            ],
            "second_order",
        )
    if family == "best":
        return beta_from_summary(summary_row, str(summary_row["best_model"]))
    raise ValueError(f"Unknown model family: {family}")


def evaluate_by_zone(points: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    summary_row = summary.iloc[0]
    rows = []
    for family in ["linear", "second_order", "best"]:
        beta, concrete_model = beta_from_summary(summary_row, family)
        for zone in ZONE_IDS:
            subset = points[points["zone"] == zone]
            t = subset["temp_c"].to_numpy(dtype=float)
            rh = subset["rh_pct"].to_numpy(dtype=float)
            y = subset["pmv"].to_numpy(dtype=float)
            y_hat = predict_from_beta(t, rh, beta)
            metrics = metric_dict(y, y_hat, n_features=len(beta) - 1)
            rows.append(
                {
                    "requested_family": family,
                    "concrete_model": concrete_model,
                    "zone": zone,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def compute_linear_diagnostics(
    points: pd.DataFrame,
    shapiro_max_n: int,
) -> pd.DataFrame:
    t = points["temp_c"].to_numpy(dtype=float)
    rh = points["rh_pct"].to_numpy(dtype=float)
    y = points["pmv"].to_numpy(dtype=float)
    X_lin = build_design_matrix(t, rh, second_order=False)
    beta, fitted, r2, adj_r2, rmse = fit_ols(X_lin, y)
    residuals = y - fitted
    std_resid = ols_standardized_residuals(
        residuals,
        n_obs=len(y),
        n_params=X_lin.shape[1],
    )

    rng = np.random.default_rng(42)
    shapiro_sample_n = min(len(residuals), shapiro_max_n)
    if len(residuals) > shapiro_sample_n:
        sample_idx = rng.choice(len(residuals), size=shapiro_sample_n, replace=False)
        shapiro_sample = residuals[sample_idx]
    else:
        shapiro_sample = residuals
    shapiro_w, shapiro_p = stats.shapiro(shapiro_sample)

    bp_lm, bp_lm_p, bp_f, bp_f_p = breusch_pagan_test(residuals, X_lin)
    vif = compute_vif_table({"temp_c": t, "rh_pct": rh})

    return pd.DataFrame(
        [
            {
                "scope": "all_zones",
                "n_points": len(points),
                "linear_beta_0": beta[0],
                "linear_beta_temp": beta[1],
                "linear_beta_rh": beta[2],
                "linear_r2": r2,
                "linear_adj_r2": adj_r2,
                "linear_rmse": rmse,
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals, ddof=1)),
                "std_residual_mean": float(np.mean(std_resid)),
                "std_residual_std": float(np.std(std_resid, ddof=1)),
                "shapiro_sample_n": shapiro_sample_n,
                "shapiro_w": float(shapiro_w),
                "shapiro_pvalue": float(shapiro_p),
                "breusch_pagan_lm": bp_lm,
                "breusch_pagan_lm_pvalue": bp_lm_p,
                "breusch_pagan_f": bp_f,
                "breusch_pagan_f_pvalue": bp_f_p,
                "vif_temp": vif["temp_c"],
                "vif_rh": vif["rh_pct"],
            }
        ]
    )


def sample_points(points: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if max_points <= 0:
        raise ValueError("Maximum plot points must be positive.")
    if len(points) <= max_points:
        return points
    return points.sample(max_points, random_state=seed)


def plot_all_regression(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    output_plot: Path,
    met: float,
    clo: float,
    max_scatter_points: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
        }
    )
    plot_points = sample_points(points, max_scatter_points, seed=42)
    row = summary.iloc[0]
    beta, concrete_model = beta_from_summary(row, "best")

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    norm = plt.Normalize(vmin=-2.0, vmax=2.0)
    scatter = ax.scatter(
        plot_points["temp_c"],
        plot_points["rh_pct"],
        c=plot_points["pmv"],
        cmap="coolwarm",
        norm=norm,
        s=7,
        alpha=0.28,
        edgecolors="none",
    )

    tt, rr = np.meshgrid(
        np.linspace(points["temp_c"].min(), points["temp_c"].max(), 160),
        np.linspace(points["rh_pct"].min(), points["rh_pct"].max(), 160),
    )
    zz = predict_from_beta(tt, rr, beta)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    contours = ax.contour(tt, rr, zz, levels=levels, colors="black", linewidths=0.9)
    ax.clabel(contours, inline=True, fmt="%0.1f", fontsize=12)

    zone_counts = points["zone"].value_counts().sort_index()
    # count_text = "\n".join(f"Zone {int(z)}: {int(n):,}" for z, n in zone_counts.items())
    # metric_text = "\n".join(
    #     [
    #         f"Best model: {concrete_model}",
    #         f"Rows: {len(points):,}",
    #         f"Linear R2: {row['linear_r2']:.6f}",
    #         f"Linear RMSE: {row['linear_rmse']:.6f}",
    #         f"Second-order R2: {row['quad_r2']:.6f}",
    #         f"Second-order RMSE: {row['quad_rmse']:.6f}",
    #         "",
    #         count_text,
    #     ]
    # )
    # ax.text(
    #     0.02,
    #     0.98,
    #     metric_text,
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=10,
    #     bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9},
    # )

    cb = fig.colorbar(scatter, ax=ax, pad=0.02)
    cb.set_label("PMV")
    ax.set_xlabel(r"Indoor Temperature ($^\circ$C)")
    ax.set_ylabel("Relative Humidity (%)")
    ax.grid(alpha=0.25)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def plot_all_regression_3d(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    output_plot: Path,
    met: float,
    clo: float,
    max_scatter_points: int,
) -> None:
    plot_points = sample_points(points, max_scatter_points, seed=314)
    row = summary.iloc[0]
    beta, concrete_model = beta_from_summary(row, "best")

    temp_grid, rh_grid = np.meshgrid(
        np.linspace(points["temp_c"].min(), points["temp_c"].max(), 80),
        np.linspace(points["rh_pct"].min(), points["rh_pct"].max(), 80),
    )
    pmv_fit_grid = predict_from_beta(temp_grid, rh_grid, beta)

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        temp_grid,
        rh_grid,
        pmv_fit_grid,
        cmap="viridis",
        alpha=0.52,
        linewidth=0,
        antialiased=True,
        rcount=80,
        ccount=80,
    )
    scatter = ax.scatter(
        plot_points["temp_c"],
        plot_points["rh_pct"],
        plot_points["pmv"],
        c=plot_points["pmv"],
        cmap="coolwarm",
        s=7,
        alpha=0.28,
        depthshade=False,
        label="Ground-truth PMV samples",
    )

    ax.plot([], [], [], color="#4c9f70", linewidth=6, alpha=0.65, label=f"Fitted PMV surface ({concrete_model})")
    ax.set_xlabel(r"Indoor Temperature ($^\circ$C)", labelpad=10)
    ax.set_ylabel("Relative Humidity (%)", labelpad=10)
    ax.set_zlabel("PMV", labelpad=10)
    ax.view_init(elev=26, azim=-132)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_title(
        "All-Zones PMV Regression Surface\n"
        f"Ground truth scatter + fitted {concrete_model} PMV, "
        f"met={met:.2f}, clo={clo:.2f}"
    )

    cb_scatter = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.64)
    cb_scatter.set_label("Ground-truth PMV")
    cb_surface = fig.colorbar(surface, ax=ax, pad=0.11, shrink=0.64)
    cb_surface.set_label("Fitted PMV surface")

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def _binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(bins, np.nan, dtype=float)
    for i in range(bins):
        if i == bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(mask):
            values[i] = float(np.mean(y[mask]))
    mask = np.isfinite(values)
    return centers[mask], values[mask]


def plot_linear_diagnostics(
    points: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_plot: Path,
    max_points: int,
) -> None:
    t = points["temp_c"].to_numpy(dtype=float)
    rh = points["rh_pct"].to_numpy(dtype=float)
    y = points["pmv"].to_numpy(dtype=float)
    X_lin = build_design_matrix(t, rh, second_order=False)
    _, fitted, _, _, _ = fit_ols(X_lin, y)
    residuals = y - fitted
    std_resid = ols_standardized_residuals(
        residuals,
        n_obs=len(y),
        n_params=X_lin.shape[1],
    )
    sqrt_abs_std = np.sqrt(np.abs(std_resid))

    rng = np.random.default_rng(7)
    if len(points) > max_points:
        choice = rng.choice(len(points), size=max_points, replace=False)
        fitted_plot = fitted[choice]
        resid_plot = residuals[choice]
        sqrt_plot = sqrt_abs_std[choice]
    else:
        fitted_plot = fitted
        resid_plot = residuals
        sqrt_plot = sqrt_abs_std

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)
    ax_resid, ax_qq, ax_scale, ax_text = axes

    ax_resid.scatter(fitted_plot, resid_plot, s=8, alpha=0.25, edgecolors="none", color="#3d85c6")
    ax_resid.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    bx, by = _binned_curve(fitted_plot, resid_plot)
    if len(bx):
        ax_resid.plot(bx, by, color="#cc0000", linewidth=1.4)
    ax_resid.set_title("Residuals vs Fitted")
    ax_resid.set_xlabel("Fitted PMV")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(alpha=0.2)

    qq_sample_n = min(len(residuals), 5000)
    if len(residuals) > qq_sample_n:
        qq_idx = rng.choice(len(residuals), size=qq_sample_n, replace=False)
        qq_resid = residuals[qq_idx]
    else:
        qq_resid = residuals
    (osm, osr), (slope, intercept, _) = stats.probplot(qq_resid, dist="norm")
    ax_qq.scatter(osm, osr, s=8, alpha=0.35, edgecolors="none", color="#6a329f")
    xline = np.array([np.min(osm), np.max(osm)])
    ax_qq.plot(xline, slope * xline + intercept, color="black", linewidth=1.0, linestyle="--")
    ax_qq.set_title("Normal Q-Q")
    ax_qq.set_xlabel("Theoretical Quantiles")
    ax_qq.set_ylabel("Sample Quantiles")
    ax_qq.grid(alpha=0.2)

    ax_scale.scatter(fitted_plot, sqrt_plot, s=8, alpha=0.25, edgecolors="none", color="#e69138")
    sx, sy = _binned_curve(fitted_plot, sqrt_plot)
    if len(sx):
        ax_scale.plot(sx, sy, color="#cc0000", linewidth=1.4)
    ax_scale.set_title("Scale-Location")
    ax_scale.set_xlabel("Fitted PMV")
    ax_scale.set_ylabel(r"$\sqrt{|standardized\ residual|}$")
    ax_scale.grid(alpha=0.2)

    row = diagnostics.iloc[0]
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(
            [
                "All-zones linear OLS",
                f"Rows = {int(row['n_points']):,}",
                f"R^2 = {row['linear_r2']:.6f}",
                f"Adj R^2 = {row['linear_adj_r2']:.6f}",
                f"RMSE = {row['linear_rmse']:.6f}",
                f"Shapiro-Wilk W = {row['shapiro_w']:.4f}",
                f"Shapiro p = {row['shapiro_pvalue']:.3g}",
                f"BP LM = {row['breusch_pagan_lm']:.3f}",
                f"BP p = {row['breusch_pagan_lm_pvalue']:.3g}",
                f"VIF(temp) = {row['vif_temp']:.3f}",
                f"VIF(RH) = {row['vif_rh']:.3f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4f4f4", "edgecolor": "#aaaaaa"},
    )

    fig.suptitle("OLS Diagnostics for the All-Zones Linear PMV Surrogate", fontsize=15)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = read_dataset(args.input)
    df = apply_date_filter(df, args.start_date, args.end_date)

    rh_cols = resolve_rh_columns(df)
    points = build_points(df, rh_cols)
    raw_point_count = len(points)
    points = apply_rh_filter(points, min_rh=args.min_rh, max_rh=args.max_rh)
    rh_filtered_count = raw_point_count - len(points)
    points = add_local_air_speed(points)
    pmv_points = compute_pmv(points, met=args.met, clo=args.clo)

    summary = fit_all(pmv_points, selection_metric=args.selection_metric)
    diagnostics = compute_linear_diagnostics(pmv_points, shapiro_max_n=args.shapiro_max_n)
    zone_breakdown = evaluate_by_zone(pmv_points, summary)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    args.output_diagnostics_summary.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output_diagnostics_summary, index=False)
    args.output_zone_breakdown.parent.mkdir(parents=True, exist_ok=True)
    zone_breakdown.to_csv(args.output_zone_breakdown, index=False)

    plot_all_regression(
        pmv_points,
        summary=summary,
        output_plot=args.output_plot,
        met=args.met,
        clo=args.clo,
        max_scatter_points=args.max_scatter_points,
    )
    plot_all_regression_3d(
        pmv_points,
        summary=summary,
        output_plot=args.output_3d_plot,
        met=args.met,
        clo=args.clo,
        max_scatter_points=args.max_3d_scatter_points,
    )
    plot_linear_diagnostics(
        pmv_points,
        diagnostics=diagnostics,
        output_plot=args.output_diagnostics_plot,
        max_points=args.max_diagnostic_points,
    )

    row = summary.iloc[0]
    diag = diagnostics.iloc[0]
    print("All-zones PMV regression fit complete.")
    print(f"Input:          {args.input.resolve()}")
    print(f"Rows used:      {len(pmv_points):,}")
    print(f"RH filter:      {args.min_rh:g}% to {args.max_rh:g}% ({rh_filtered_count:,} rows removed)")
    print(f"Zones pooled:   {row['zones']}")
    print(f"Summary:        {args.output_summary.resolve()}")
    print(f"Zone breakdown: {args.output_zone_breakdown.resolve()}")
    print(f"Plot:           {args.output_plot.resolve()}")
    print(f"3D plot:        {args.output_3d_plot.resolve()}")
    print(f"Diag CSV:       {args.output_diagnostics_summary.resolve()}")
    print(f"Diag plot:      {args.output_diagnostics_plot.resolve()}")
    print("")
    print(
        "Fit metrics: "
        f"linear R2={row['linear_r2']:.6f}, linear RMSE={row['linear_rmse']:.6f}; "
        f"second-order R2={row['quad_r2']:.6f}, second-order RMSE={row['quad_rmse']:.6f}; "
        f"best={row['best_model']}"
    )
    print(
        "Linear diagnostics: "
        f"Shapiro p={diag['shapiro_pvalue']:.3g}, "
        f"BP p={diag['breusch_pagan_lm_pvalue']:.3g}, "
        f"VIF(temp)={diag['vif_temp']:.3f}, "
        f"VIF(RH)={diag['vif_rh']:.3f}"
    )


if __name__ == "__main__":
    main()
