#!/usr/bin/env python3
"""Fit PMV regression models (linear vs second-order) by zone.

For each zone, fits:
1) Multiple linear regression: PMV ~ temp + rh
2) Second-order regression: PMV ~ temp + rh + temp^2 + rh^2 + temp*rh

Then plots five zone scatter panels (Temp-RH colored by PMV) with fitted PMV contours
from the selected best model (by adjusted R^2 by default).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache_pythermalcomfort")
)

from pythermalcomfort.models import pmv_ppd_ashrae


ZONE_IDS = [1, 2, 3, 4, 5]

TEMP_COLS: Dict[int, str] = {
    1: "Zone 1 Temperature",
    2: "Zone 2 Temperature",
    3: "Zone 3 Temperature",
    4: "Zone 4 Temperature",
    5: "Zone 5 Temperature",
}

RH_FALLBACK_COLS: Dict[int, str] = {
    1: "FCU-01 Return Air Humi",
    2: "FCU-02 Return Air Humi",
    3: "FCU-03 Return Air Humi",
    4: "FCU-04 Return Air Humi",
    5: "FCU-05 Return Air Humi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and compare PMV regressions (linear vs second-order) by zone."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/private/l14_merged_data_with_rain.csv"),
        help="Private L14 input CSV path.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("outputs/analysis/pmv/temp_rh_pmv_regression_by_zone.png"),
        help="Output regression plot path.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/analysis/pmv/temp_rh_pmv_regression_summary.csv"),
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--output-diagnostics-summary",
        type=Path,
        default=Path("outputs/analysis/pmv/temp_rh_pmv_linear_diagnostics_summary.csv"),
        help="Output CSV path for linear-model OLS diagnostics.",
    )
    parser.add_argument(
        "--output-diagnostics-plot",
        type=Path,
        default=Path("outputs/analysis/pmv/temp_rh_pmv_linear_diagnostics.png"),
        help="Output plot path for linear-model OLS diagnostics.",
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
        help="Metabolic rate in met (default: 1.2).",
    )
    parser.add_argument(
        "--clo",
        type=float,
        default=0.5,
        help="Clothing insulation in clo (default: 0.5).",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["adj_r2", "rmse"],
        default="adj_r2",
        help="Model selection metric between linear and second-order.",
    )
    parser.add_argument(
        "--max-scatter-points-per-zone",
        type=int,
        default=25000,
        help="Maximum points per zone drawn in scatter for readability.",
    )
    parser.add_argument(
        "--max-diagnostic-points-per-zone",
        type=int,
        default=5000,
        help="Maximum points per zone drawn in residual diagnostic plots.",
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


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column in the input CSV.")
    df["date"] = parse_datetime_col(df["date"])
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def resolve_rh_columns(df: pd.DataFrame) -> Dict[int, str]:
    rh_map: Dict[int, str] = {}
    for zone in ZONE_IDS:
        candidates = [
            f"Zone {zone} RH",
            f"Zone {zone} Humidity",
            f"Zone {zone} Relative Humidity",
            RH_FALLBACK_COLS[zone],
        ]
        chosen = next((c for c in candidates if c in df.columns), None)
        if chosen is None:
            raise KeyError(f"Missing RH source for Zone {zone}. Tried: {candidates}")
        rh_map[zone] = chosen
    return rh_map


def build_points(df: pd.DataFrame, rh_cols: Dict[int, str]) -> pd.DataFrame:
    missing_temps = [c for c in TEMP_COLS.values() if c not in df.columns]
    if missing_temps:
        raise KeyError(f"Missing temperature columns: {missing_temps}")

    frames = []
    for zone in ZONE_IDS:
        part = df[["date", TEMP_COLS[zone], rh_cols[zone]]].copy()
        part.columns = ["date", "temp_c", "rh_pct"]
        part["zone"] = zone
        frames.append(part)

    points = pd.concat(frames, ignore_index=True)
    points["temp_c"] = pd.to_numeric(points["temp_c"], errors="coerce")
    points["rh_pct"] = pd.to_numeric(points["rh_pct"], errors="coerce")
    points = points.dropna(subset=["temp_c", "rh_pct"])
    points = points[(points["rh_pct"] >= 0.0) & (points["rh_pct"] <= 100.0)]
    return points.reset_index(drop=True)


def add_local_air_speed(points: pd.DataFrame) -> pd.DataFrame:
    a = -2.6290730
    b = 7.5244548
    t = points["temp_c"].to_numpy(dtype=float)
    c = 4.218295023 - 0.3078496 * t
    disc = b * b - 4.0 * a * c

    v = np.full_like(t, fill_value=np.nan, dtype=float)
    ok = disc >= 0.0
    if np.any(ok):
        root = np.sqrt(disc[ok])
        v1 = (-b + root) / (2.0 * a)
        v2 = (-b - root) / (2.0 * a)
        v_ok = np.where(v1 >= 0.0, v1, v2)
        v_ok = np.where(v_ok >= 0.0, v_ok, np.nan)
        v[ok] = v_ok

    out = points.copy()
    out["v_local_mps"] = v
    return out


def compute_pmv(points: pd.DataFrame, met: float, clo: float) -> pd.DataFrame:
    valid = points.dropna(subset=["temp_c", "rh_pct", "v_local_mps"]).copy()
    valid = valid[
        np.isfinite(valid["temp_c"])
        & np.isfinite(valid["rh_pct"])
        & np.isfinite(valid["v_local_mps"])
    ].copy()
    if valid.empty:
        raise ValueError("No valid points left after local-air-speed computation.")

    result = pmv_ppd_ashrae(
        tdb=valid["temp_c"].to_numpy(dtype=float),
        tr=valid["temp_c"].to_numpy(dtype=float),
        vr=valid["v_local_mps"].to_numpy(dtype=float),
        rh=valid["rh_pct"].to_numpy(dtype=float),
        met=met,
        clo=clo,
        model="55-2023",
    )
    valid["pmv"] = np.asarray(result.pmv, dtype=float)
    valid = valid[np.isfinite(valid["pmv"])].copy()
    valid = valid.dropna(subset=["pmv"]).reset_index(drop=True)
    return valid


def build_design_matrix(temp: np.ndarray, rh: np.ndarray, second_order: bool) -> np.ndarray:
    if second_order:
        return np.column_stack(
            [np.ones_like(temp), temp, rh, temp * temp, rh * rh, temp * rh]
        )
    return np.column_stack([np.ones_like(temp), temp, rh])


def fit_ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        y_hat = X @ beta
    mask = np.isfinite(y_hat) & np.isfinite(y)
    if not np.any(mask):
        raise ValueError("Regression produced no finite predictions.")
    y_valid = y[mask]
    y_hat_valid = y_hat[mask]
    resid = y_valid - y_hat_valid
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_valid - y_valid.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    n = int(mask.sum())
    p = X.shape[1] - 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    return beta, y_hat, r2, adj_r2, rmse


def ols_standardized_residuals(
    residuals: np.ndarray,
    n_obs: int,
    n_params: int,
) -> np.ndarray:
    dof = max(n_obs - n_params, 1)
    sigma = np.sqrt(np.sum(residuals**2) / dof)
    if sigma <= 0.0 or not np.isfinite(sigma):
        return np.zeros_like(residuals)
    return residuals / sigma


def breusch_pagan_test(residuals: np.ndarray, X: np.ndarray) -> tuple[float, float, float, float]:
    y_aux = np.asarray(residuals, dtype=float) ** 2
    _, y_aux_hat, _, _, _ = fit_ols(X, y_aux)

    ss_res = float(np.sum((y_aux - y_aux_hat) ** 2))
    ss_tot = float(np.sum((y_aux - y_aux.mean()) ** 2))
    r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    n = len(y_aux)
    k = X.shape[1] - 1
    lm_stat = n * r2_aux
    lm_pvalue = float(stats.chi2.sf(lm_stat, k)) if k > 0 else np.nan

    denom_df = n - X.shape[1]
    if k > 0 and denom_df > 0 and r2_aux < 1.0:
        f_stat = (r2_aux / k) / ((1.0 - r2_aux) / denom_df)
        f_pvalue = float(stats.f.sf(f_stat, k, denom_df))
    else:
        f_stat = np.nan
        f_pvalue = np.nan
    return float(lm_stat), lm_pvalue, float(f_stat), f_pvalue


def compute_vif_table(feature_map: Dict[str, np.ndarray]) -> Dict[str, float]:
    vif: Dict[str, float] = {}
    names = list(feature_map.keys())
    for idx, name in enumerate(names):
        y = np.asarray(feature_map[name], dtype=float)
        other_names = [n for j, n in enumerate(names) if j != idx]
        if not other_names:
            vif[name] = 1.0
            continue
        X_other = np.column_stack(
            [np.ones_like(y)] + [np.asarray(feature_map[n], dtype=float) for n in other_names]
        )
        _, _, r2, _, _ = fit_ols(X_other, y)
        vif[name] = float(np.inf) if r2 >= 1.0 else float(1.0 / max(1.0 - r2, 1e-12))
    return vif


def compute_linear_diagnostics(
    points: pd.DataFrame,
    shapiro_max_n: int = 5000,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)

    for zone in ZONE_IDS:
        z = points[points["zone"] == zone]
        t = z["temp_c"].to_numpy(dtype=float)
        rh = z["rh_pct"].to_numpy(dtype=float)
        y = z["pmv"].to_numpy(dtype=float)
        X_lin = build_design_matrix(t, rh, second_order=False)
        beta, fitted, r2, adj_r2, rmse = fit_ols(X_lin, y)
        residuals = y - fitted
        std_resid = ols_standardized_residuals(residuals, n_obs=len(y), n_params=X_lin.shape[1])

        shapiro_sample_n = min(len(residuals), shapiro_max_n)
        if len(residuals) > shapiro_sample_n:
            sample_idx = rng.choice(len(residuals), size=shapiro_sample_n, replace=False)
            shapiro_sample = residuals[sample_idx]
        else:
            shapiro_sample = residuals
        shapiro_w, shapiro_p = stats.shapiro(shapiro_sample)

        bp_lm, bp_lm_p, bp_f, bp_f_p = breusch_pagan_test(residuals, X_lin)
        vif = compute_vif_table({"temp_c": t, "rh_pct": rh})

        rows.append(
            {
                "zone": zone,
                "n_points": len(z),
                "linear_beta_0": beta[0],
                "linear_beta_temp": beta[1],
                "linear_beta_rh": beta[2],
                "linear_r2": r2,
                "linear_adj_r2": adj_r2,
                "linear_rmse": rmse,
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals, ddof=1)),
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
        )

    return pd.DataFrame(rows)


def _binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
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
    max_points_per_zone: int,
) -> None:
    if max_points_per_zone <= 0:
        raise ValueError("--max-diagnostic-points-per-zone must be positive.")

    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(len(ZONE_IDS), 4, figsize=(20, 4.6 * len(ZONE_IDS)), constrained_layout=True)
    if len(ZONE_IDS) == 1:
        axes = np.array([axes])

    for row_idx, zone in enumerate(ZONE_IDS):
        ax_resid, ax_qq, ax_scale, ax_text = axes[row_idx]
        z = points[points["zone"] == zone]
        t = z["temp_c"].to_numpy(dtype=float)
        rh = z["rh_pct"].to_numpy(dtype=float)
        y = z["pmv"].to_numpy(dtype=float)
        X_lin = build_design_matrix(t, rh, second_order=False)
        _, fitted, _, _, _ = fit_ols(X_lin, y)
        residuals = y - fitted
        std_resid = ols_standardized_residuals(residuals, n_obs=len(y), n_params=X_lin.shape[1])
        sqrt_abs_std = np.sqrt(np.abs(std_resid))

        if len(z) > max_points_per_zone:
            choice = rng.choice(len(z), size=max_points_per_zone, replace=False)
            fitted_plot = fitted[choice]
            resid_plot = residuals[choice]
            sqrt_plot = sqrt_abs_std[choice]
        else:
            fitted_plot = fitted
            resid_plot = residuals
            sqrt_plot = sqrt_abs_std

        ax_resid.scatter(fitted_plot, resid_plot, s=9, alpha=0.3, edgecolors="none", color="#3d85c6")
        ax_resid.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        bx, by = _binned_curve(fitted_plot, resid_plot)
        if len(bx):
            ax_resid.plot(bx, by, color="#cc0000", linewidth=1.4)
        ax_resid.set_title(f"Zone {zone}: Residuals vs Fitted")
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
        ax_qq.set_title(f"Zone {zone}: Normal Q-Q")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")
        ax_qq.grid(alpha=0.2)

        ax_scale.scatter(fitted_plot, sqrt_plot, s=9, alpha=0.3, edgecolors="none", color="#e69138")
        sx, sy = _binned_curve(fitted_plot, sqrt_plot)
        if len(sx):
            ax_scale.plot(sx, sy, color="#cc0000", linewidth=1.4)
        ax_scale.set_title(f"Zone {zone}: Scale-Location")
        ax_scale.set_xlabel("Fitted PMV")
        ax_scale.set_ylabel(r"$\sqrt{|standardized\ residual|}$")
        ax_scale.grid(alpha=0.2)

        row = diagnostics[diagnostics["zone"] == zone].iloc[0]
        ax_text.axis("off")
        ax_text.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"Zone {zone} summary",
                    f"R^2 = {row['linear_r2']:.4f}",
                    f"Adj R^2 = {row['linear_adj_r2']:.4f}",
                    f"RMSE = {row['linear_rmse']:.4f}",
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

    fig.suptitle(
        "OLS Diagnostics for the Linear PMV Surrogate Used in Control",
        fontsize=15,
    )
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def predict_from_beta(temp: np.ndarray, rh: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if len(beta) == 3:
        return beta[0] + beta[1] * temp + beta[2] * rh
    return (
        beta[0]
        + beta[1] * temp
        + beta[2] * rh
        + beta[3] * temp * temp
        + beta[4] * rh * rh
        + beta[5] * temp * rh
    )


def fit_by_zone(points: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    rows = []
    for zone in ZONE_IDS:
        z = points[points["zone"] == zone]
        t = z["temp_c"].to_numpy(dtype=float)
        rh = z["rh_pct"].to_numpy(dtype=float)
        y = z["pmv"].to_numpy(dtype=float)

        X_lin = build_design_matrix(t, rh, second_order=False)
        lin_beta, _, lin_r2, lin_adj_r2, lin_rmse = fit_ols(X_lin, y)

        X_quad = build_design_matrix(t, rh, second_order=True)
        quad_beta, _, quad_r2, quad_adj_r2, quad_rmse = fit_ols(X_quad, y)

        if selection_metric == "adj_r2":
            best_model = "second_order" if quad_adj_r2 > lin_adj_r2 else "linear"
        else:
            best_model = "second_order" if quad_rmse < lin_rmse else "linear"

        rows.append(
            {
                "zone": zone,
                "n_points": len(z),
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
        )

    return pd.DataFrame(rows)


def plot_zone_regression(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    output_plot: Path,
    met: float,
    clo: float,
    max_scatter_points_per_zone: int,
) -> None:
    if max_scatter_points_per_zone <= 0:
        raise ValueError("--max-scatter-points-per-zone must be positive.")

    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    axes_flat = axes.flatten()
    rng = np.random.default_rng(42)
    norm = plt.Normalize(vmin=-2.0, vmax=2.0)

    scatter_ref = None
    for idx, zone in enumerate(ZONE_IDS):
        ax = axes_flat[idx]
        z = points[points["zone"] == zone].copy()

        if len(z) > max_scatter_points_per_zone:
            choice = rng.choice(len(z), size=max_scatter_points_per_zone, replace=False)
            z_plot = z.iloc[choice].copy()
        else:
            z_plot = z

        sc = ax.scatter(
            z_plot["temp_c"],
            z_plot["rh_pct"],
            c=z_plot["pmv"],
            cmap="coolwarm",
            norm=norm,
            s=8,
            alpha=0.35,
            edgecolors="none",
        )
        if scatter_ref is None:
            scatter_ref = sc

        row = summary[summary["zone"] == zone].iloc[0]
        t_min, t_max = z["temp_c"].min(), z["temp_c"].max()
        rh_min, rh_max = z["rh_pct"].min(), z["rh_pct"].max()

        tt, rr = np.meshgrid(
            np.linspace(t_min, t_max, 120),
            np.linspace(rh_min, rh_max, 120),
        )

        if row["best_model"] == "second_order":
            beta = np.array(
                [
                    row["quad_beta_0"],
                    row["quad_beta_temp"],
                    row["quad_beta_rh"],
                    row["quad_beta_temp2"],
                    row["quad_beta_rh2"],
                    row["quad_beta_temp_rh"],
                ],
                dtype=float,
            )
        else:
            beta = np.array(
                [row["linear_beta_0"], row["linear_beta_temp"], row["linear_beta_rh"]],
                dtype=float,
            )

        zz = predict_from_beta(tt, rr, beta)
        levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        cs = ax.contour(tt, rr, zz, levels=levels, colors="k", linewidths=0.8, alpha=0.85)
        ax.clabel(cs, inline=True, fmt="%0.1f", fontsize=7)

        ax.set_xlabel("Indoor Temperature (C)")
        ax.set_ylabel("Relative Humidity (%)")
        ax.grid(alpha=0.25)
        ax.set_title(
            f"Zone {zone} | best={row['best_model']}\n"
            f"lin R2={row['linear_r2']:.4f}, quad R2={row['quad_r2']:.4f}"
        )

    axes_flat[-1].axis("off")
    if scatter_ref is not None:
        cb = fig.colorbar(scatter_ref, ax=axes_flat[:-1], shrink=0.75, pad=0.02)
        cb.set_label("PMV (point value)")

    fig.suptitle(
        "Zone-wise PMV Regression Fit (Linear vs Second-order)\n"
        f"Assumptions: met={met:.2f}, clo={clo:.2f}, tr=tdb, local v from quadratic fan model",
        fontsize=13,
    )
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = read_dataset(args.input)

    if args.start_date:
        start = pd.to_datetime(args.start_date, format="%Y-%m-%d", errors="raise")
        df = df[df["date"] >= start]
    if args.end_date:
        end = pd.to_datetime(args.end_date, format="%Y-%m-%d", errors="raise")
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df["date"] <= end]
    if df.empty:
        raise ValueError("No records left after date filtering.")

    rh_cols = resolve_rh_columns(df)
    points = build_points(df, rh_cols)
    points = add_local_air_speed(points)
    pmv_points = compute_pmv(points, met=args.met, clo=args.clo)

    summary = fit_by_zone(pmv_points, selection_metric=args.selection_metric)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    diagnostics = compute_linear_diagnostics(pmv_points)
    args.output_diagnostics_summary.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output_diagnostics_summary, index=False)

    plot_zone_regression(
        pmv_points,
        summary=summary,
        output_plot=args.output_plot,
        met=args.met,
        clo=args.clo,
        max_scatter_points_per_zone=args.max_scatter_points_per_zone,
    )
    plot_linear_diagnostics(
        pmv_points,
        diagnostics=diagnostics,
        output_plot=args.output_diagnostics_plot,
        max_points_per_zone=args.max_diagnostic_points_per_zone,
    )

    n_second = int((summary["best_model"] == "second_order").sum())
    n_linear = int((summary["best_model"] == "linear").sum())

    print("Zone regression fit complete.")
    print(f"Input:   {args.input.resolve()}")
    print(f"Plot:    {args.output_plot.resolve()}")
    print(f"Summary: {args.output_summary.resolve()}")
    print(f"Diag plot: {args.output_diagnostics_plot.resolve()}")
    print(f"Diag CSV:  {args.output_diagnostics_summary.resolve()}")
    print(f"Rows used for PMV: {len(pmv_points):,}")
    print(f"Best model counts -> second_order: {n_second}, linear: {n_linear}")
    print("Per-zone metrics (R2):")
    for _, r in summary.iterrows():
        print(
            f"  Zone {int(r['zone'])}: "
            f"linear={r['linear_r2']:.5f}, second_order={r['quad_r2']:.5f}, "
            f"best={r['best_model']}"
        )
    print("Per-zone linear-model diagnostics:")
    for _, r in diagnostics.iterrows():
        print(
            f"  Zone {int(r['zone'])}: "
            f"Shapiro p={r['shapiro_pvalue']:.3g}, "
            f"BP p={r['breusch_pagan_lm_pvalue']:.3g}, "
            f"VIF(temp)={r['vif_temp']:.3f}, "
            f"VIF(RH)={r['vif_rh']:.3f}"
        )


if __name__ == "__main__":
    main()
