"""Shared final-study MPC and baseline implementation.

One predicted mode change, 60-minute ordinary dwell, observed-rain safety,
minute-by-minute nonlinear feedback and validated MIQP first-mode branches.
Importing this module does not read data, start a solver or discover licenses.
"""
from __future__ import annotations
from .mpc_future_data import apply_future_data_source, build_forecast_audit
import atexit

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_OK = True
except ImportError:
    GUROBI_OK = False

_GUROBI_ENV = None

def _gurobi_environment():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        _GUROBI_ENV = env
        atexit.register(env.dispose)
    return _GUROBI_ENV

import time

import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import matplotlib.pyplot as plt

from dataclasses import dataclass

from typing import Dict, Tuple, Optional, List, Callable

from collections import OrderedDict

from pathlib import Path

class CNNLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_surrogate(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CNNLSTM(ckpt['input_dim'], ckpt['hidden_dim'], ckpt['output_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['scaler_X'], ckpt['scaler_U']

def _scaler_params_to_torch(scaler, device: torch.device):
    scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
    offset = torch.as_tensor(scaler.min_, dtype=torch.float32, device=device)
    return scale, offset

@torch.inference_mode()
def forward_surrogate(model: CNNLSTM, scaler_X, scaler_U, x, u, device: torch.device):
    x0 = torch.as_tensor(np.array(x, dtype=np.float32, copy=True), dtype=torch.float32, device=device).view(1, -1)
    u0 = torch.as_tensor(np.array(u, dtype=np.float32, copy=True), dtype=torch.float32, device=device).view(1, -1)
    x_scale, x_min = _scaler_params_to_torch(scaler_X, device)
    u_scale, u_min = _scaler_params_to_torch(scaler_U, device)

    x_norm = x0 * x_scale + x_min
    u_norm = u0 * u_scale + u_min
    model_in = torch.cat([x_norm, u_norm], dim=1).unsqueeze(1)
    x_next_norm = model(model_in).squeeze(0)
    x_next = (x_next_norm - x_min) / x_scale
    return x_next.detach().cpu().numpy()

@dataclass
class Linearization:
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray

def linearize_surrogate(model: CNNLSTM, scaler_X, scaler_U, x_bar, u_bar, device: torch.device) -> Linearization:
    """Vectorized Jacobian-based linearization to reduce autograd overhead."""
    x0 = torch.as_tensor(np.array(x_bar, dtype=np.float32, copy=True), dtype=torch.float32, device=device).view(-1)
    u0 = torch.as_tensor(np.array(u_bar, dtype=np.float32, copy=True), dtype=torch.float32, device=device).view(-1)

    x_scale, x_min = _scaler_params_to_torch(scaler_X, device)
    u_scale, u_min = _scaler_params_to_torch(scaler_U, device)

    nx = x0.shape[0]
    nu = u0.shape[0]

    def model_fn(z: torch.Tensor) -> torch.Tensor:
        x_vec = z[:nx].unsqueeze(0)
        u_vec = z[nx:].unsqueeze(0)
        x_norm = x_vec * x_scale + x_min
        u_norm = u_vec * u_scale + u_min
        model_in = torch.cat([x_norm, u_norm], dim=1).unsqueeze(1)
        x_next_norm = model(model_in).squeeze(0)
        return (x_next_norm - x_min) / x_scale

    z0 = torch.cat([x0, u0]).requires_grad_(True)
    use_cuda_jacobian = device.type == 'cuda'
    use_vectorized_jacobian = not use_cuda_jacobian
    # CUDA + cuDNN RNN backward cannot use eval-mode cuDNN graphs or batched VJP.
    with torch.backends.cudnn.flags(enabled=not use_cuda_jacobian):
        x_next = model_fn(z0)
        J = torch.autograd.functional.jacobian(model_fn, z0, vectorize=use_vectorized_jacobian)

    A = J[:, :nx].detach()
    B = J[:, nx:].detach()
    c = (x_next.detach() - A @ x0.detach() - B @ u0.detach()).detach()
    return Linearization(A=A.cpu().numpy(), B=B.cpu().numpy(), c=c.cpu().numpy())

AC_STATE_COLS = [
    'Zone 1 Temperature', 'Zone 2 Temperature', 'Zone 3 Temperature', 'Zone 4 Temperature', 'Zone 5 Temperature'
]

AC_INPUT_COLS = [
    'OutdoorTemperatureWindow',
    'FCU-01 Supply Air Temp',
    'FCU-02 Supply Air Temp - 1 min',
    'FCU-03 Supply Air Temp',
    'FCU-04 Supply Air Temp',
    'FCU-05 Supply Air Temp',
    'PFCU-01 Supply Air Temp',
    'PFCU-02 Supply Air Temp',
    'rain_status',
    'Solar Radiation',
]

NV_INPUT_COLS_BASE = [
    'OutdoorTemperatureWindow',
    'Wind Speed',
    'Wind Direction',
    'rain_status',
    'Solar Radiation',
]

NV_INPUT_COLS = [
    'OutdoorTemperatureWindow', 'Wind Speed', 'Wind Direction', 'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp', 'rain_status', 'Solar Radiation'
]

COLUMNS_TO_CHECK = list(set(AC_STATE_COLS + AC_INPUT_COLS + NV_INPUT_COLS + ['Z1 Windows Open Close Status']))

AC_CONTROL_IDX = [1, 2, 3, 4, 5, 6, 7]

NV_CONTROL_IDX = [3, 4]

AC_TRAIN_COLUMNS = [
    'Zone 1 Temperature', 'Zone 2 Temperature', 'Zone 3 Temperature', 'Zone 4 Temperature', 'Zone 5 Temperature',
    'OutdoorTemperatureWindow', 'FCU-01 Supply Air Temp', 'FCU-02 Supply Air Temp - 1 min', 'FCU-03 Supply Air Temp',
    'FCU-04 Supply Air Temp', 'FCU-05 Supply Air Temp', 'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp',
    'rain_status', 'Solar Radiation'
]

NV_TRAIN_COLUMNS = [
    'Zone 1 Temperature', 'Zone 2 Temperature', 'Zone 3 Temperature', 'Zone 4 Temperature', 'Zone 5 Temperature',
    'OutdoorTemperatureWindow', 'Wind Speed', 'Wind Direction', 'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp',
    'rain_status', 'Solar Radiation'
]

DATES_TO_REMOVE = pd.to_datetime(['2024-10-31', '2024-11-04'])

SPLIT_RATIOS = (0.7, 0.15, 0.15)

_SPLIT_DATES_CACHE = {}

def _pfcus_for_nv(row: pd.Series, supply: np.ndarray) -> tuple[float, float]:
    # Extract PFCU-01/02 supply temps for NV/PFC mode from a control vector.
    supply_arr = np.asarray(supply, dtype=float).reshape(-1)
    if supply_arr.shape[0] != 2:
        raise ValueError(f'NV/PFC supply must have 2 entries (PFCU-01, PFCU-02), got {supply_arr.shape[0]}')
    p1 = float(supply_arr[0])
    p2 = float(supply_arr[1])
    return p1, p2

def build_u_vector(row: pd.Series, mode: str, supply) -> np.ndarray:
    # Map control vector into surrogate input (AC: 6 entries, NV/PFC: 2 entries).
    if mode == 'ac':
        supply_arr = np.asarray(supply, dtype=float).reshape(-1)
        if supply_arr.shape[0] != len(AC_CONTROL_IDX):
            raise ValueError(f'AC supply must have {len(AC_CONTROL_IDX)} entries, got {supply_arr.shape[0]}')
        u_vec = row[AC_INPUT_COLS].astype(float).to_numpy(copy=True)
        u_vec[AC_CONTROL_IDX] = supply_arr
    else:
        p1, p2 = _pfcus_for_nv(row, supply)
        u_vec = row[NV_INPUT_COLS_BASE].astype(float).to_list()
        u_vec.insert(3, p1)
        u_vec.insert(4, p2)
        u_vec = np.asarray(u_vec, dtype=float)
    return u_vec

def lift_linearization(A1: np.ndarray, B1: np.ndarray, c1: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    'Accumulate 1-min linear model over `steps` minutes with hold control.'
    A_acc = np.linalg.matrix_power(A1, steps)
    B_acc = np.zeros_like(B1)
    c_acc = np.zeros_like(c1)
    A_power = np.eye(A1.shape[0])
    for _ in range(steps):
        B_acc += A_power @ B1
        c_acc += A_power @ c1
        A_power = A_power @ A1
    return A_acc, B_acc, c_acc

def make_continuous_segments(df: pd.DataFrame) -> List[pd.DataFrame]:
    list_of_dfs = []
    current_df = [df.iloc[0]]
    for i in range(1, len(df)):
        if (df.iloc[i]['date'] - df.iloc[i-1]['date']).seconds == 60:
            current_df.append(df.iloc[i])
        else:
            list_of_dfs.append(pd.DataFrame(current_df))
            current_df = [df.iloc[i]]
    list_of_dfs.append(pd.DataFrame(current_df))
    return list_of_dfs

def _normalize_segment_split(split: Optional[str]) -> str:
    key = str(split or 'all').strip().lower().replace(' ', '')
    aliases = {
        'train': 'train',
        'val': 'val',
        'valid': 'val',
        'validate': 'val',
        'validation': 'val',
        'test': 'test',
        'test+val': 'test+val',
        'val+test': 'test+val',
        'test_val': 'test+val',
        'val_test': 'test+val',
        'test,val': 'test+val',
        'val,test': 'test+val',
        'all': 'all',
        '*': 'all',
        'none': 'all',
        '': 'all',
    }
    if key not in aliases:
        raise ValueError(
            "Unknown split. Use one of: 'test', 'val'/'validate', 'test+val', 'train', or 'all'. "
            f"Got: {split!r}"
        )
    return aliases[key]

SPECIFIC_HEAT_AIR = 1005.0  # J/(kg*K)

AIR_DENSITY = 1.225         # kg/m^3

FCU_AIRFLOW_CFM = 800.0     # ft^3/min per FCU terminal, constant-volume assumption

LOCAL_COOLING_CFM = 1000.0  # ft^3/min per PFCU terminal, constant-volume assumption

CFM_TO_M3S = 0.0283168 / 60.0

JOULES_PER_KWH = 3.6e6

DEFAULT_COP_FCU = 5.0

DEFAULT_COP_PFC = 5.0

PMV_LINEAR_COEFFS = {
    1: (-7.673842, 0.249984, 0.011084),
    2: (-7.539536, 0.244956, 0.011203),
    3: (-7.595545, 0.245508, 0.011841),
    4: (-7.625768, 0.246251, 0.011946),
    5: (-7.599742, 0.246740, 0.011352),
}

PMV_QUAD_COEFFS = {
    1: (-2.668230, 0.027159, -0.043495, 0.001987, 0.000063, 0.001618),
    2: (-2.952313, 0.044535, -0.042467, 0.001691, 0.000056, 0.001615),
    3: (-3.600209, 0.097232, -0.046012, 0.000616, 0.000051, 0.001761),
    4: (-2.617966, 0.025267, -0.044229, 0.001944, 0.000056, 0.001682),
    5: (-2.581557, 0.022383, -0.044050, 0.001940, 0.000046, 0.001725),
}

def _extract_outdoor_rh(exog_row: pd.Series, rh_nv_source_col: str = "OutdoorHumidityWindow") -> float:
    return float(
        exog_row.get(
            rh_nv_source_col,
            exog_row.get("OutdoorHumidityWindow", exog_row.get("Outdoor Humidity", exog_row.get("OutdoorHumidity", 65.0))),
        )
    )

def _rain_active(value, threshold: float) -> bool:
    try:
        rain_value = float(value)
    except (TypeError, ValueError):
        rain_value = 0.0
    if not np.isfinite(rain_value):
        rain_value = 0.0
    return rain_value >= float(threshold)

def current_observed_rain_active(cfg, current_exog_row: pd.Series) -> bool:
    if not bool(getattr(cfg, "enforce_current_rain_safety", False)):
        return False
    threshold = float(getattr(cfg, "rain_active_threshold", 0.5))
    return _rain_active(current_exog_row.get("rain_status", 0.0), threshold)

def rain_lockout_steps(cfg) -> int:
    lockout_min = max(int(getattr(cfg, "rain_lockout_min", 0)), 0)
    ctrl_period_min = max(int(getattr(cfg, "ctrl_period_min", 5)), 1)
    return int(np.ceil(lockout_min / ctrl_period_min)) if lockout_min > 0 else 0

def observed_rain_forces_ac(cfg, current_exog_row: pd.Series, rain_lock_steps: int = 0) -> bool:
    """Return whether the current receding-horizon move must be AC because of observed rain or lockout."""
    if not bool(getattr(cfg, "enforce_current_rain_safety", False)):
        return False
    return current_observed_rain_active(cfg, current_exog_row) or int(rain_lock_steps) > 0

def update_observed_rain_lock_steps(cfg, current_exog_row: pd.Series, previous_lock_steps: int) -> int:
    """Advance the 30-minute observed-rain lockout at one MPC control horizon."""
    if not bool(getattr(cfg, "enforce_current_rain_safety", False)):
        return 0
    if current_observed_rain_active(cfg, current_exog_row):
        return rain_lockout_steps(cfg)
    return max(int(previous_lock_steps) - 1, 0)

def neutralize_rain_for_mpc_prediction(cfg, exog_5min_forecast: pd.DataFrame) -> pd.DataFrame:
    """Return a forecast copy with rain_status set to no-rain for MIQP prediction only."""
    if not bool(getattr(cfg, "ignore_rain_in_mpc_prediction", False)):
        return exog_5min_forecast
    if "rain_status" not in exog_5min_forecast.columns:
        return exog_5min_forecast
    forecast = exog_5min_forecast.copy()
    forecast["rain_status"] = 0.0
    return forecast

def _pmv_expr(zone_id: int, temp_expr, rh_value: float, model: str):
    model = str(model).lower()
    if model == "linear":
        b0, bt, brh = PMV_LINEAR_COEFFS[int(zone_id)]
        return b0 + bt * temp_expr + brh * float(rh_value)
    if model == "second_order":
        b0, bt, brh, bt2, brh2, btrh = PMV_QUAD_COEFFS[int(zone_id)]
        rh = float(rh_value)
        return b0 + bt * temp_expr + brh * rh + bt2 * temp_expr * temp_expr + brh2 * rh * rh + btrh * temp_expr * rh
    raise ValueError(f"Unsupported pmv_model={model!r}; expected 'linear' or 'second_order'.")

class RepoHooks:
    def load_day_dataframe(self, day_idx: int):
        raise NotImplementedError

    def plant_step_1min(self, mode: int, x, u, exog_row: pd.Series):
        raise NotImplementedError

    def linearize_5min(self, mode: int, x_oper, u_oper, exog_fore_5min: pd.Series):
        raise NotImplementedError

@dataclass
class MPCConfig:
    # Comfort targets
    T_ref_ac: float = 27.0
    T_nv_max: float = 31.0
    T_nv_soft: float = 28.5
    dt_min: int = 1
    ctrl_period_min: int = 5
    horizon_min: int = 60
    x_min: float = 22.0
    x_max: float = 30.0
    u_min: float = 12.0  # legacy lower bound (kept for compatibility)
    u_min_ac: float = 12.5
    u_min_pfc: float = 18.5
    u_max_ac: float = 21.0
    u_max_pfc: float = 18.5
    w_du: float = 1e1
    w_switch: float = 1e1
    w_energy: float = 1e4
    pmv_target: float = -0.5
    pmv_model: str = "linear"  # options: "linear", "second_order"
    rh_ac_fixed: float = 65.0
    rh_nv_source_col: str = "OutdoorHumidityWindow"
    enforce_current_rain_safety: bool = False
    rain_active_threshold: float = 0.5
    rain_lockout_min: int = 0
    ignore_rain_in_mpc_prediction: bool = False
    pmv_big_m: float = 8.0  # used by second_order PMV gating
    energy_norm_kwh: float = 1.0
    pfc_big_m: float = 50.0
    min_dwell_steps: int = 12  # 60 minutes at the 5-minute update
    max_predicted_switches: int = 1
    energy_coeff_scale: float = 1.0
    cop_fcu: float = DEFAULT_COP_FCU
    cop_pfc: float = DEFAULT_COP_PFC
    linearize_energy: bool = False
    penalize_undercool: bool = False
    use_pv_objective: bool = False
    pv_model: str = "simple_linear"  # options: "simple_linear", "appendix_b"
    pv_alpha: float = 0.002
    pv_pmax: Optional[float] = 3.0
    pv_a1: float = 0.0
    pv_a2: float = 0.0
    pv_a3: float = 0.0
    pv_source_area_sqft: float = 5000.0
    building_area_sqft: float = 75.0
    pv_scale_factor: Optional[float] = None
    w_pv_export: float = 1.0
    price_low: float = 1.0
    price_high: float = 4.0
    T_out_nv_ref: float = 30.0
    w_terminal: float = 1.0
    soften_x_bounds: bool = True
    w_x_slack: float = 1e2
    mip_time_limit: Optional[float] = 600
    gurobi_output: int = 0
    gurobi_numeric_focus: int = 2
    validate_first_mode_branches: bool = True
    gurobi_presolve: Optional[int] = 1
    gurobi_mipfocus: Optional[int] = 2
    gurobi_heuristics: Optional[float] = 0.1
    gurobi_nonconvex: Optional[int] = None
    write_iis_on_infeasible: bool = False

def time_in_windows(ts: pd.Timestamp) -> bool:
    h, m = ts.hour, ts.minute
    minutes = h * 60 + m
    w1 = (7*60 + 30) <= minutes < (10*60)
    w2 = (16*60 + 30) <= minutes < (19*60)
    return w1 or w2

def _pv_scale(cfg: MPCConfig) -> float:
    explicit_scale = getattr(cfg, "pv_scale_factor", None)
    if explicit_scale is not None:
        return float(explicit_scale)

    source_area = float(getattr(cfg, "pv_source_area_sqft", 0.0) or 0.0)
    building_area = float(getattr(cfg, "building_area_sqft", 0.0) or 0.0)
    if source_area > 0 and building_area > 0:
        return building_area / source_area
    return 1.0

def pv_available_power_kw(cfg: MPCConfig, I_solar: float, T_out: float) -> float:
    ghi = max(float(I_solar), 0.0)
    t_out = float(T_out)
    pv_model = str(getattr(cfg, "pv_model", "simple_linear")).lower()
    scale = _pv_scale(cfg)

    if pv_model == "appendix_b":
        pv_kw = scale * (cfg.pv_a1 * ghi * ghi + cfg.pv_a2 * ghi * t_out + cfg.pv_a3 * ghi)
    elif pv_model == "simple_linear":
        pv_kw = scale * (cfg.pv_alpha * ghi)
    else:
        raise ValueError(f"Unsupported pv_model={cfg.pv_model!r}; expected 'simple_linear' or 'appendix_b'.")

    pv_kw = max(pv_kw, 0.0)
    pv_pmax = getattr(cfg, "pv_pmax", None)
    if pv_pmax is not None:
        pv_kw = min(pv_kw, float(pv_pmax))
    return pv_kw

def _validate_cop(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return value

def hvac_power_heat_exchange_kw(
    cfg: MPCConfig,
    z_mode: int,
    zone_temps,
    fcu_supplies,
    pfc_supplies,
    T_out: float,
) -> float:
    """Electrical cooling power from constant-volume heat exchange; fan power excluded."""
    cop_fcu = _validate_cop("cop_fcu", getattr(cfg, "cop_fcu", DEFAULT_COP_FCU))
    cop_pfc = _validate_cop("cop_pfc", getattr(cfg, "cop_pfc", DEFAULT_COP_PFC))
    fcu_coeff_kw_per_k = SPECIFIC_HEAT_AIR * AIR_DENSITY * (FCU_AIRFLOW_CFM * CFM_TO_M3S) / 1000.0 / cop_fcu
    pfc_coeff_kw_per_k = SPECIFIC_HEAT_AIR * AIR_DENSITY * (LOCAL_COOLING_CFM * CFM_TO_M3S) / 1000.0 / cop_pfc

    if int(round(float(z_mode))) == 1:
        x = np.asarray(zone_temps, dtype=float)
        u = np.asarray(fcu_supplies, dtype=float)
        delta = np.maximum(x - u, 0.0)
        return float(fcu_coeff_kw_per_k * np.nan_to_num(delta, nan=0.0).sum())

    u_pfc = np.asarray(pfc_supplies, dtype=float)
    delta = np.maximum(float(T_out) - u_pfc, 0.0)
    return float(pfc_coeff_kw_per_k * np.nan_to_num(delta, nan=0.0).sum())

def _explain_mpc_decision(step_idx: int, total_steps: int, t: pd.Timestamp, debug: Dict[str, object]):
    if debug is None:
        return

    z0 = int(debug.get('z0', 0))
    mode = 'AC' if z0 == 1 else 'NV'
    k0 = debug.get('k0', {})
    breakdown = debug.get('breakdown', {})

    print(f"\n[MPC-Explain] step {step_idx + 1}/{total_steps} @ {t}")
    print(f"  chosen mode: {mode} (z0={z0}), obj={debug.get('obj', float('nan')):.3f}")

    cf = debug.get('counterfactual_z0', None)
    if isinstance(cf, dict):
        ac = cf.get('ac', {})
        nv = cf.get('nv', {})

        def _fmt3(v):
            try:
                return f"{float(v):.3f}"
            except Exception:
                return "nan"

        def _fmt_breakdown(bd):
            if not isinstance(bd, dict):
                return "n/a"
            return (
                f"comfort={_fmt3(bd.get('comfort'))}, "
                f"du={_fmt3(bd.get('du'))}, "
                f"switch={_fmt3(bd.get('switch'))}, "
                f"energy={_fmt3(bd.get('energy'))}, "
                f"x_slack={_fmt3(bd.get('x_slack'))}, "
                f"pv={_fmt3(bd.get('pv'))}"
            )

        print(
            "  forced z0 counterfactual: "
            f"obj(z0=AC)={_fmt3(ac.get('obj'))}, "
            f"obj(z0=NV)={_fmt3(nv.get('obj'))}, "
            f"x1_mean(z0=AC)={_fmt3(ac.get('x1_mean'))}, "
            f"x1_mean(z0=NV)={_fmt3(nv.get('x1_mean'))}"
        )
        print(f"    breakdown(z0=AC): {_fmt_breakdown(ac.get('breakdown'))}")
        print(f"    breakdown(z0=NV): {_fmt_breakdown(nv.get('breakdown'))}")
    elif debug.get('counterfactual_z0_error'):
        print(f"  forced z0 counterfactual unavailable: {debug.get('counterfactual_z0_error')}")

    print(
        "  k=0 signals: "
        f"T_out={k0.get('T_out', float('nan')):.2f}, "
        f"rh_nv={k0.get('rh_nv', float('nan')):.2f}, "
        f"sw0={k0.get('sw0', float('nan')):.3f}, "
        f"pmv_target={k0.get('pmv_target', float('nan')):.2f}, "
        f"pmv_ac_dev_sq_sum={k0.get('pmv_ac_dev_sq_sum', k0.get('pmv_ac_sq_sum', float('nan'))):.4f}, "
        f"pmv_nv_dev_sq_sum={k0.get('pmv_nv_dev_sq_sum', k0.get('pmv_nv_sq_sum', float('nan'))):.4f}, "
        f"x_high_sum={k0.get('x_high_sum', float('nan')):.4f}"
    )
    print(
        "  k=0 energy proxies (kWh per control stage): "
        f"AC_like={k0.get('energy_fcu0_kwh', float('nan')):.5f}, "
        f"NV_like={k0.get('energy_pfc0_kwh', float('nan')):.5f}, "
        f"used={k0.get('energy_used0_kwh', float('nan')):.5f}"
    )

    t_ac = k0.get('x_next_mean_ac0')
    t_nv = k0.get('x_next_mean_nv0')
    if t_ac is not None and t_nv is not None:
        print(f"  one-step mean temp forecast: AC={t_ac:.3f}, NV={t_nv:.3f}")

    pmv_ac_metric = float(k0.get('pmv_ac_dev_sq_sum', k0.get('pmv_ac_sq_sum', 1e9)))
    pmv_nv_metric = float(k0.get('pmv_nv_dev_sq_sum', k0.get('pmv_nv_sq_sum', 1e9)))

    reasons = []
    if z0 == 0:
        if pmv_nv_metric <= pmv_ac_metric:
            reasons.append('NV PMV deviation^2 from target at k=0 is no worse than AC.')
        if k0.get('energy_pfc0_kwh', 1e9) < k0.get('energy_fcu0_kwh', 1e9):
            reasons.append('NV-like stage energy is lower than AC-like stage energy.')
    else:
        if pmv_ac_metric < pmv_nv_metric:
            reasons.append('AC PMV deviation^2 from target at k=0 is better than NV.')
        if k0.get('x_high_sum', 0.0) > 1e-6:
            reasons.append('Global x_max soft-bound pressure is active at k=0.')
        if k0.get('energy_fcu0_kwh', 0.0) <= k0.get('energy_pfc0_kwh', 0.0):
            reasons.append('AC-like stage energy is not worse than NV-like at k=0.')

    if reasons:
        print('  inferred drivers:')
        for r in reasons:
            print('   -', r)

    print(
        "  objective breakdown: "
        f"comfort={breakdown.get('comfort', float('nan')):.3f}, "
        f"du={breakdown.get('du', float('nan')):.3f}, "
        f"switch={breakdown.get('switch', float('nan')):.3f}, "
        f"energy={breakdown.get('energy', float('nan')):.3f}, "
        f"x_slack={breakdown.get('x_slack', float('nan')):.3f}, "
        f"pv={breakdown.get('pv', float('nan')):.3f}"
    )

def solve_miqp(cfg: MPCConfig, x0, u_prev: float, z_prev: int, t0: pd.Timestamp, exog_5min_forecast: pd.DataFrame, hooks: RepoHooks, lock_steps: int = 0, verbose: bool = False, force_z0: Optional[int] = None, rain_force_ac_now: bool = False, known_rain_forced_stages: int = 0, collect_diagnostics: bool = False) -> Dict[str, float]:
    if not GUROBI_OK:
        raise RuntimeError("Gurobi not available. Install gurobipy to run the MIQP.")

    # Negative comfort/energy/slack penalties can create pathological or unbounded objectives.
    nonneg_weight_fields = (
        "w_du",
        "w_switch",
        "w_energy",
        "w_x_slack",
        "w_terminal",
        "w_pv_export",
    )
    for name in nonneg_weight_fields:
        value = float(getattr(cfg, name, 0.0))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0; got {value}")

    x0_vec = np.asarray(x0, dtype=float).reshape(-1)
    nx = x0_vec.shape[0]
    H = len(exog_5min_forecast)
    u_prev_vec = np.asarray(u_prev, dtype=float).reshape(-1)
    rain_force_ac_now = bool(rain_force_ac_now) and bool(getattr(cfg, "enforce_current_rain_safety", False))

    # Keep linearization horizon aligned with control period if hooks expose steps_per_ctrl.
    if hasattr(hooks, "steps_per_ctrl"):
        steps_per_ctrl = int(getattr(hooks, "steps_per_ctrl"))
        if steps_per_ctrl != cfg.ctrl_period_min:
            hooks.steps_per_ctrl = int(cfg.ctrl_period_min)
            if hasattr(hooks, "reset_cache"):
                hooks.reset_cache()

    linearize_energy = bool(getattr(cfg, "linearize_energy", False))
    pmv_model = str(getattr(cfg, "pmv_model", "linear")).lower()
    if pmv_model not in ("linear", "second_order"):
        raise ValueError(f"Unsupported pmv_model={pmv_model!r}; expected 'linear' or 'second_order'.")

    m = gp.Model("mmv_mpc", env=_gurobi_environment())
    m.Params.Threads = 2
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = int(cfg.gurobi_numeric_focus)
    m.Params.OutputFlag = int(getattr(cfg, "gurobi_output", 0))

    nonconvex = getattr(cfg, "gurobi_nonconvex", None)
    if (not linearize_energy or pmv_model == "second_order") and nonconvex is None:
        nonconvex = 2
    if nonconvex is not None:
        m.Params.NonConvex = int(nonconvex)

    presolve = getattr(cfg, "gurobi_presolve", 1)
    if presolve is not None:
        m.Params.Presolve = int(presolve)
    mipfocus = getattr(cfg, "gurobi_mipfocus", 1)
    if mipfocus is not None:
        m.Params.MIPFocus = int(mipfocus)
    heuristics = getattr(cfg, "gurobi_heuristics", 0.1)
    if heuristics is not None:
        m.Params.Heuristics = float(heuristics)

    if cfg.mip_time_limit:
        m.Params.TimeLimit = float(cfg.mip_time_limit)

    soften_x_bounds = bool(getattr(cfg, "soften_x_bounds", True))
    w_x_slack = float(getattr(cfg, "w_x_slack", 1e4))
    write_iis = bool(getattr(cfg, "write_iis_on_infeasible", True))

    ctrl_count = len(AC_CONTROL_IDX)
    fcu_indices = [0, 1, 2, 3, 4]
    pfc_indices = [5, 6]
    pfc_dataset_cols = ["PFCU-01 Supply Air Temp", "PFCU-02 Supply Air Temp"]
    t_out_fore = [float(exog_5min_forecast.iloc[k].get("T_out", exog_5min_forecast.iloc[k].get("OutdoorTemperatureWindow", 0.0))) for k in range(H)]
    rh_nv_fore = [_extract_outdoor_rh(exog_5min_forecast.iloc[k], getattr(cfg, "rh_nv_source_col", "OutdoorHumidityWindow")) for k in range(H)]
    rh_nv_fore_h = rh_nv_fore + ([rh_nv_fore[-1]] if H > 0 else [float(getattr(cfg, "rh_ac_fixed", 65.0))])
    pfc_data_fore = {
        p_idx: [float(exog_5min_forecast.iloc[k].get(col, t_out_fore[k])) for k in range(H)]
        for p_idx, col in enumerate(pfc_dataset_cols)
    }
    u_lower_bound = {}
    u_upper_bound = {}
    for k in range(H):
        for j in fcu_indices:
            u_lower_bound[(k, j)] = cfg.u_min_ac
            u_upper_bound[(k, j)] = cfg.u_max_ac
        for p_idx, j in enumerate(pfc_indices):
            t_out_k = t_out_fore[k]
            t_pfc_data_k = pfc_data_fore[p_idx][k]
            u_lower_bound[(k, j)] = min(cfg.u_min_pfc, t_out_k, t_pfc_data_k)
            u_upper_bound[(k, j)] = max(cfg.u_max_pfc, t_out_k, t_pfc_data_k)

    x = m.addVars(H + 1, nx, name="x")
    u = m.addVars(H, ctrl_count, lb=u_lower_bound, ub=u_upper_bound, name="u")
    z_mode = m.addVars(H, vtype=GRB.BINARY, name="z_mode")
    if force_z0 is not None:
        force_z0 = int(force_z0)
        if force_z0 not in (0, 1):
            raise ValueError(f"force_z0 must be 0, 1, or None; got {force_z0!r}")
        if H <= 0:
            raise ValueError("force_z0 requires a non-empty horizon")
        if rain_force_ac_now and force_z0 != 1:
            raise ValueError("rain_force_ac_now conflicts with force_z0=0")
        m.addConstr(z_mode[0] == force_z0, name="force_z0")
    if rain_force_ac_now:
        if H <= 0:
            raise ValueError("rain_force_ac_now requires a non-empty horizon")
        m.addConstr(z_mode[0] == 1, name="current_rain_force_ac")
    for k in range(min(int(known_rain_forced_stages), H)):
        m.addConstr(z_mode[k] == 1, name=f"known_rain_lock_{k}")
    pfc_on = m.addVars(H, vtype=GRB.BINARY, name="pfc_on")
    e_pos_ac = m.addVars(H + 1, nx, lb=0.0, name="e_pos_ac")
    e_neg_ac = m.addVars(H + 1, nx, lb=0.0, name="e_neg_ac")
    s_nv_high = m.addVars(H + 1, nx, lb=0.0, name="s_nv_high")
    s_nv_soft = m.addVars(H + 1, nx, lb=0.0, name="s_nv_soft")
    pmv_ac = m.addVars(H + 1, nx, lb=-GRB.INFINITY, name="pmv_ac")
    pmv_nv = m.addVars(H + 1, nx, lb=-GRB.INFINITY, name="pmv_nv")
    sw = m.addVars(H, lb=0.0, name="sw")
    lift_fcu_raw = m.addVars(H, nx, lb=-GRB.INFINITY, name="lift_fcu_raw")
    lift_pfc_raw = m.addVars(H, len(pfc_indices), lb=-GRB.INFINITY, name="lift_pfc_raw")
    delta_cool_fcu = m.addVars(H, nx, lb=0.0, name="delta_cool_fcu")
    delta_cool_pfc = m.addVars(H, len(pfc_indices), lb=0.0, name="delta_cool_pfc")
    energy_fcu_used = m.addVars(H, lb=0.0, name="energy_fcu_used") if linearize_energy else None
    energy_pfc_used = m.addVars(H, lb=0.0, name="energy_pfc_used") if linearize_energy else None

    s_x_low = m.addVars(H + 1, nx, lb=0.0, name="s_x_low")
    s_x_high = m.addVars(H + 1, nx, lb=0.0, name="s_x_high")

    step_seconds = cfg.ctrl_period_min * 60
    energy_coeff_scale = float(getattr(cfg, "energy_coeff_scale", 1.0))
    if energy_coeff_scale <= 0:
        raise ValueError("energy_coeff_scale must be > 0")
    cop_fcu = _validate_cop("cop_fcu", getattr(cfg, "cop_fcu", DEFAULT_COP_FCU))
    cop_pfc = _validate_cop("cop_pfc", getattr(cfg, "cop_pfc", DEFAULT_COP_PFC))
    dt_h = step_seconds / 3600.0
    # Constant-volume heat exchange. Flow constants are per terminal unit, so the
    # power/energy expressions multiply by the sum of each unit's temperature lift.
    # Fan power is excluded by design for this constant-volume formulation.
    fcu_power_coeff = SPECIFIC_HEAT_AIR * AIR_DENSITY * (FCU_AIRFLOW_CFM * CFM_TO_M3S) / 1000.0 / cop_fcu
    local_power_coeff = SPECIFIC_HEAT_AIR * AIR_DENSITY * (LOCAL_COOLING_CFM * CFM_TO_M3S) / 1000.0 / cop_pfc
    fcu_energy_coeff = fcu_power_coeff * dt_h * energy_coeff_scale
    local_energy_coeff = local_power_coeff * dt_h * energy_coeff_scale
    w_energy = float(cfg.w_energy) / energy_coeff_scale
    energy_norm_kwh = float(getattr(cfg, "energy_norm_kwh", 1.0))
    if energy_norm_kwh <= 0:
        raise ValueError("energy_norm_kwh must be > 0")
    pfc_big_m = float(getattr(cfg, "pfc_big_m", 50.0))

    if cfg.use_pv_objective:
        p_hvac = m.addVars(H, lb=0.0, name="p_hvac")
        p_grid = m.addVars(H, lb=0.0, name="p_grid")
        p_export = m.addVars(H, lb=0.0, name="p_export")

    for j in range(nx):
        m.addConstr(x[0, j] == float(x0_vec[j]))

    for k in range(H + 1):
        for j in range(nx):
            if soften_x_bounds:
                m.addConstr(x[k, j] + s_x_low[k, j] >= cfg.x_min)
                m.addConstr(x[k, j] - s_x_high[k, j] <= cfg.x_max)
            else:
                m.addConstr(x[k, j] >= cfg.x_min)
                m.addConstr(x[k, j] <= cfg.x_max)

    for k in range(H + 1):
        z_k = z_mode[k] if k < H else z_mode[H - 1]
        rh_nv_k = float(rh_nv_fore_h[k])
        for j in range(nx):
            # Legacy temperature comfort terms retained for compatibility (not used in objective).
            m.addGenConstrIndicator(z_k, True, e_pos_ac[k, j] >= x[k, j] - cfg.T_ref_ac)
            m.addGenConstrIndicator(z_k, True, e_neg_ac[k, j] >= cfg.T_ref_ac - x[k, j])
            m.addGenConstrIndicator(z_k, False, e_pos_ac[k, j] == 0.0)
            m.addGenConstrIndicator(z_k, False, e_neg_ac[k, j] == 0.0)
            m.addGenConstrIndicator(z_k, False, x[k, j] <= cfg.T_nv_max + s_nv_high[k, j])
            m.addGenConstrIndicator(z_k, False, s_nv_soft[k, j] >= x[k, j] - cfg.T_nv_soft)
            m.addGenConstrIndicator(z_k, True, s_nv_soft[k, j] == 0.0)

            # PMV definitions by mode:
            #   AC mode uses fixed RH (cfg.rh_ac_fixed)
            #   NV mode uses outdoor RH (cfg.rh_nv_source_col from exogenous forecast)
            pmv_expr_ac = _pmv_expr(j + 1, x[k, j], float(getattr(cfg, "rh_ac_fixed", 65.0)), pmv_model)
            pmv_expr_nv = _pmv_expr(j + 1, x[k, j], rh_nv_k, pmv_model)

            if pmv_model == "linear":
                # Exact mode gating in linear PMV mode (avoids big-M induced infeasibility under dwell locks).
                m.addGenConstrIndicator(z_k, True, pmv_ac[k, j] - pmv_expr_ac == 0.0)
                m.addGenConstrIndicator(z_k, False, pmv_ac[k, j] == 0.0)
                m.addGenConstrIndicator(z_k, False, pmv_nv[k, j] - pmv_expr_nv == 0.0)
                m.addGenConstrIndicator(z_k, True, pmv_nv[k, j] == 0.0)
            else:
                pmv_big_m = float(getattr(cfg, "pmv_big_m", 8.0))

                # AC active when z_k == 1
                m.addConstr(pmv_ac[k, j] - pmv_expr_ac <= pmv_big_m * (1.0 - z_k))
                m.addConstr(pmv_expr_ac - pmv_ac[k, j] <= pmv_big_m * (1.0 - z_k))
                m.addConstr(pmv_ac[k, j] <= pmv_big_m * z_k)
                m.addConstr(pmv_ac[k, j] >= -pmv_big_m * z_k)

                # NV active when z_k == 0
                m.addConstr(pmv_nv[k, j] - pmv_expr_nv <= pmv_big_m * z_k)
                m.addConstr(pmv_expr_nv - pmv_nv[k, j] <= pmv_big_m * z_k)
                m.addConstr(pmv_nv[k, j] <= pmv_big_m * (1.0 - z_k))
                m.addConstr(pmv_nv[k, j] >= -pmv_big_m * (1.0 - z_k))

    for k in range(H):
        if k == 0:
            m.addConstr(sw[k] >= z_mode[k] - z_prev)
            m.addConstr(sw[k] >= z_prev - z_mode[k])
        else:
            m.addConstr(sw[k] >= z_mode[k] - z_mode[k - 1])
            m.addConstr(sw[k] >= z_mode[k - 1] - z_mode[k])

    for k in range(H):
        m.addConstr(pfc_on[k] <= 1 - z_mode[k])

    # Optional dwell-time lock (disable with min_dwell_steps=0). Observed rain can override
    # the current move so a prior NV dwell lock cannot keep windows open during rain.
    if lock_steps > 0:
        for k in range(min(lock_steps, H)):
            if rain_force_ac_now and k == 0 and int(z_prev) == 0:
                continue
            m.addConstr(z_mode[k] == z_prev)

    min_dwell_steps = int(getattr(cfg, "min_dwell_steps", 0))
    # With nonnegative sw and a global cap of one, every rolling dwell
    # inequality is implied. Inherited execution locks remain enforced above.
    if min_dwell_steps > 1 and cfg.max_predicted_switches > 1:
        for k in range(1, H):
            # A rain-forced switch now clears ordinary dwell in the original execution policy.
            exempt_current_switch = rain_force_ac_now and int(z_prev) == 0
            lo = max(1 if exempt_current_switch else 0, k - min_dwell_steps + 1)
            m.addConstr(gp.quicksum(sw[i] for i in range(lo, k + 1)) <= 1)

    A0_ac = B0_ac = g0_ac = None
    A0_nv = B0_nv = g0_nv = None

    for k in range(H):
        exog_k = exog_5min_forecast.iloc[k]
        T_out = float(exog_k.get("T_out", exog_k.get("OutdoorTemperatureWindow", 0.0)))
        for j in range(nx):
            m.addConstr(lift_fcu_raw[k, j] == x[k, j] - u[k, fcu_indices[j]])
            m.addGenConstrMax(delta_cool_fcu[k, j], [lift_fcu_raw[k, j]], constant=0.0)
        for p_idx, u_idx in enumerate(pfc_indices):
            m.addConstr(lift_pfc_raw[k, p_idx] == T_out - u[k, u_idx])
            m.addGenConstrMax(delta_cool_pfc[k, p_idx], [lift_pfc_raw[k, p_idx]], constant=0.0)

        # Optional PFCU on/off during NV mode (z_mode=0)
        for p_idx, u_idx in enumerate(pfc_indices):
            T_pfc_data = float(exog_k.get(pfc_dataset_cols[p_idx], T_out))

            # If NV and PFCU is ON, pin supply to dataset real-time value
            m.addConstr(u[k, u_idx] - T_pfc_data <= pfc_big_m * (z_mode[k] + (1 - pfc_on[k])))
            m.addConstr(T_pfc_data - u[k, u_idx] <= pfc_big_m * (z_mode[k] + (1 - pfc_on[k])))
            # If NV and PFCU is OFF, pin supply to outdoor temperature
            m.addConstr(u[k, u_idx] - T_out <= pfc_big_m * (z_mode[k] + pfc_on[k]))
            m.addConstr(T_out - u[k, u_idx] <= pfc_big_m * (z_mode[k] + pfc_on[k]))

            # If AC, pin PFCU supply to fixed setpoint (e.g., 21°C)
            m.addGenConstrIndicator(z_mode[k], True, u[k, u_idx] == cfg.u_min_pfc)

        A_ac, B_ac, g_ac = hooks.linearize_5min(1, x0_vec, u_prev_vec, exog_k)
        A_nv, B_nv, g_nv = hooks.linearize_5min(0, x0_vec, u_prev_vec[-2:], exog_k)
        if k == 0:
            A0_ac, B0_ac, g0_ac = A_ac.copy(), B_ac.copy(), g_ac.copy()
            A0_nv, B0_nv, g0_nv = A_nv.copy(), B_nv.copy(), g_nv.copy()

        for j in range(nx):
            lhs_ac = gp.quicksum(A_ac[j, i] * x[k, i] for i in range(nx)) + gp.quicksum(B_ac[j, l] * u[k, l] for l in range(ctrl_count)) + float(g_ac[j])
            lhs_nv = gp.quicksum(A_nv[j, i] * x[k, i] for i in range(nx)) + gp.quicksum(B_nv[j, idx] * u[k, pfc_indices[idx]] for idx in range(len(pfc_indices))) + float(g_nv[j])
            m.addGenConstrIndicator(z_mode[k], True, x[k+1, j] - lhs_ac == 0.0)
            m.addGenConstrIndicator(z_mode[k], False, x[k+1, j] - lhs_nv == 0.0)

        if cfg.use_pv_objective:
            T_out = float(exog_k.get("T_out", exog_k.get("OutdoorTemperatureWindow", 0.0)))
            fcu_power = fcu_power_coeff * gp.quicksum(delta_cool_fcu[k, j] for j in range(nx))
            pfc_power = local_power_coeff * gp.quicksum(delta_cool_pfc[k, p] for p in range(len(pfc_indices)))
            # Mode-gated electrical cooling power for PV/grid balance.
            # z_mode=1: FCU cooling; z_mode=0: PFCU/local cooling.
            m.addGenConstrIndicator(z_mode[k], True, p_hvac[k] - fcu_power == 0.0)
            m.addGenConstrIndicator(z_mode[k], False, p_hvac[k] - pfc_power == 0.0)

            I_solar = float(exog_k.get("I_solar", exog_k.get("Solar Radiation", 0.0)))
            p_pv = pv_available_power_kw(cfg, I_solar=I_solar, T_out=T_out)
            m.addConstr(p_grid[k] >= p_hvac[k] - p_pv)
            m.addConstr(p_export[k] >= p_pv - p_hvac[k])
            m.addConstr(p_grid[k] >= 0.0)
            m.addConstr(p_export[k] >= 0.0)

    du_cost = 0.0
    for k in range(H):
        if k == 0:
            du_cost += gp.quicksum((u[k, j] - u_prev_vec[j]) * (u[k, j] - u_prev_vec[j]) for j in range(ctrl_count))
        else:
            du_cost += gp.quicksum((u[k, j] - u[k-1, j]) * (u[k, j] - u[k-1, j]) for j in range(ctrl_count))

    obj = 0.0
    terminal_weight = float(getattr(cfg, "w_terminal", 1.0))
    pmv_target = float(getattr(cfg, "pmv_target", -0.5))
    for k in range(H + 1):
        stage_w = terminal_weight if k == H else 1.0
        for j in range(nx):
            pmv_ac_err = pmv_ac[k, j] - pmv_target
            pmv_nv_err = pmv_nv[k, j] - pmv_target
            obj += stage_w * (pmv_ac_err * pmv_ac_err + pmv_nv_err * pmv_nv_err)
    obj += cfg.w_du * du_cost
    obj += cfg.w_switch * gp.quicksum(sw[k] for k in range(H))
    m.addConstr(gp.quicksum(sw[k] for k in range(H)) <= float(cfg.max_predicted_switches))

    energy_cost = 0.0
    for k in range(H):
        energy_fcu = fcu_energy_coeff * gp.quicksum(delta_cool_fcu[k, j] for j in range(nx))
        energy_pfc = local_energy_coeff * gp.quicksum(delta_cool_pfc[k, p] for p in range(len(pfc_indices)))
        if linearize_energy:
            m.addGenConstrIndicator(z_mode[k], True, energy_fcu_used[k] - energy_fcu == 0.0)
            m.addGenConstrIndicator(z_mode[k], False, energy_fcu_used[k] == 0.0)
            m.addGenConstrIndicator(z_mode[k], False, energy_pfc_used[k] - energy_pfc == 0.0)
            m.addGenConstrIndicator(z_mode[k], True, energy_pfc_used[k] == 0.0)
            energy_cost += energy_fcu_used[k] + energy_pfc_used[k]
        else:
            # energy_cost += z_mode * energy_fcu + (1.0 - z_mode) * energy_pfc
            energy_cost += z_mode[k] * energy_fcu + (1.0 - z_mode[k]) * energy_pfc
    obj += w_energy * energy_cost / energy_norm_kwh

    if cfg.use_pv_objective:
        for k in range(H):
            obj += cfg.w_pv_export * p_export[k]

    if soften_x_bounds:
        obj += w_x_slack * gp.quicksum(s_x_low[k, j] + s_x_high[k, j] for k in range(H + 1) for j in range(nx))

    forced_sequence = getattr(cfg, "test_forced_sequence", None)
    if forced_sequence is not None:
        for k, value in enumerate(forced_sequence):
            m.addConstr(z_mode[k] == int(value), name=f"test_force_{k}")
    m.setObjective(obj, GRB.MINIMIZE)
    solver_started = time.perf_counter()
    from .numerical_solver import validated_optimize
    numerical_report = validated_optimize(m, check_first_mode=(
        cfg.validate_first_mode_branches and force_z0 is None and lock_steps == 0
        and not rain_force_ac_now and known_rain_forced_stages == 0
        and forced_sequence is None))

    if m.Status == GRB.INF_OR_UNBD:
        if verbose:
            print("[MPC] Solver returned INF_OR_UNBD; retrying with DualReductions=0 to disambiguate.")
        m.Params.DualReductions = 0
        m.reset()
        m.optimize()

    if m.Status == GRB.INFEASIBLE and write_iis:
        m.computeIIS()
        m.write("mpc_miqp_iis.ilp")
        print("[MPC] Infeasible: wrote IIS to mpc_miqp_iis.ilp")

    if m.Status == GRB.UNBOUNDED:
        try:
            m.write("mpc_miqp_unbounded.lp")
            print("[MPC] Unbounded: wrote model snapshot to mpc_miqp_unbounded.lp")
        except Exception:
            pass

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        status_label = {
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
        }.get(m.Status, str(m.Status))
        raise RuntimeError(f"MPC solve failed with status {m.Status} ({status_label})")

    solver_elapsed = time.perf_counter() - solver_started
    predicted_modes = [int(round(z_mode[k].X)) for k in range(H)]
    diagnostic = {"t": str(t0), "numerical_validation": numerical_report,
                  "solver_runtime_s": float(m.Runtime),
                  "solver_wall_s": solver_elapsed, "solver_status": int(m.Status),
                  "mip_gap": float(m.MIPGap), "nodes": float(m.NodeCount),
                  "objective": float(m.ObjVal), "predicted_modes": predicted_modes,
                  "z_prev": int(z_prev), "inherited_lock_steps": int(lock_steps),
                  "rain_force_ac_now": bool(rain_force_ac_now),
                  "known_rain_forced_stages": int(known_rain_forced_stages)}
    u0_vec = np.array([float(u[0, j].X) for j in range(ctrl_count)])
    z0_val = int(round(z_mode[0].X))
    pfc_on0_val = int(round(pfc_on[0].X))

    debug = None
    if verbose or collect_diagnostics:
        comfort_term = 0.0
        for k in range(H + 1):
            stage_w = terminal_weight if k == H else 1.0
            for j in range(nx):
                comfort_term += stage_w * ((float(pmv_ac[k, j].X) - pmv_target) ** 2 + (float(pmv_nv[k, j].X) - pmv_target) ** 2)

        du_term = cfg.w_du * float(sum((float(u[k, j].X) - (float(u_prev_vec[j]) if k == 0 else float(u[k-1, j].X))) ** 2 for k in range(H) for j in range(ctrl_count)))
        switch_term = cfg.w_switch * float(sum(float(sw[k].X) for k in range(H)))
        energy_raw_fcu = [fcu_energy_coeff * float(sum(float(delta_cool_fcu[k, j].X) for j in range(nx))) for k in range(H)]
        energy_raw_pfc = [local_energy_coeff * float(sum(float(delta_cool_pfc[k, p].X) for p in range(len(pfc_indices)))) for k in range(H)]

        if linearize_energy:
            energy_cost_sum = float(sum(float(energy_fcu_used[k].X) + float(energy_pfc_used[k].X) for k in range(H)))
            energy_used0 = float(energy_fcu_used[0].X) + float(energy_pfc_used[0].X)
        else:
            energy_cost_sum = float(sum(float(z_mode[k].X) * energy_raw_fcu[k] + (1.0 - float(z_mode[k].X)) * energy_raw_pfc[k] for k in range(H)))
            energy_used0 = float(z_mode[0].X) * energy_raw_fcu[0] + (1.0 - float(z_mode[0].X)) * energy_raw_pfc[0]
        energy_term = w_energy * energy_cost_sum / energy_norm_kwh

        pv_term = 0.0
        if cfg.use_pv_objective:
            for k in range(H):
                pv_term += cfg.w_pv_export * float(p_export[k].X)

        x_slack_term = 0.0
        if soften_x_bounds:
            x_slack_term = w_x_slack * float(sum(float(s_x_low[k, j].X) + float(s_x_high[k, j].X) for k in range(H + 1) for j in range(nx)))

        x_next_mean_ac0 = None
        x_next_mean_nv0 = None
        if A0_ac is not None and A0_nv is not None:
            try:
                x_next_ac0 = A0_ac @ x0_vec + B0_ac @ u0_vec + g0_ac
                u0_pfc = np.asarray([u0_vec[pfc_indices[0]], u0_vec[pfc_indices[1]]], dtype=float)
                x_next_nv0 = A0_nv @ x0_vec + B0_nv @ u0_pfc + g0_nv
                x_next_mean_ac0 = float(np.mean(x_next_ac0))
                x_next_mean_nv0 = float(np.mean(x_next_nv0))
            except Exception:
                x_next_mean_ac0 = None
                x_next_mean_nv0 = None

        debug = {
            "obj": float(m.ObjVal),
            "z0": z0_val,
            "rain_force_ac_now": bool(rain_force_ac_now),
            "breakdown": {
                "comfort": float(comfort_term),
                "du": float(du_term),
                "switch": float(switch_term),
                "energy": float(energy_term),
                "x_slack": float(x_slack_term),
                "pv": float(pv_term),
            },
            "k0": {
                "T_out": float(t_out_fore[0]),
                "sw0": float(sw[0].X),
                "rh_nv": float(rh_nv_fore_h[0]),
                "pmv_target": float(pmv_target),
                "pmv_ac_abs_sum": float(sum(abs(float(pmv_ac[0, j].X)) for j in range(nx))),
                "pmv_nv_abs_sum": float(sum(abs(float(pmv_nv[0, j].X)) for j in range(nx))),
                "pmv_ac_sq_sum": float(sum(float(pmv_ac[0, j].X) ** 2 for j in range(nx))),
                "pmv_nv_sq_sum": float(sum(float(pmv_nv[0, j].X) ** 2 for j in range(nx))),
                "pmv_ac_dev_sq_sum": float(sum((float(pmv_ac[0, j].X) - pmv_target) ** 2 for j in range(nx))),
                "pmv_nv_dev_sq_sum": float(sum((float(pmv_nv[0, j].X) - pmv_target) ** 2 for j in range(nx))),
                "x_high_sum": float(sum(float(s_x_high[0, j].X) for j in range(nx))),
                "energy_fcu0_kwh": float(energy_raw_fcu[0]),
                "energy_pfc0_kwh": float(energy_raw_pfc[0]),
                "energy_used0_kwh": float(energy_used0),
                "x_next_mean_ac0": x_next_mean_ac0,
                "x_next_mean_nv0": x_next_mean_nv0,
            }
        }

    if debug is not None:
        attempts = numerical_report['initial_attempts']
        recovery = numerical_report.get('recovery_attempts', [])
        retries = attempts[1:] + recovery
        branch_attempts = (numerical_report.get('alternative') or {}).get('attempts', [])
        total_solver_runtime = sum(a['runtime_s'] for a in attempts + recovery + branch_attempts)
        debug.update({
            'predicted_max_temp_c': max(float(x[k, j].X) for k in range(H + 1) for j in range(nx)),
            'predicted_upper_slack_sum': sum(float(s_x_high[k, j].X) for k in range(H + 1) for j in range(nx)) if soften_x_bounds else 0.0,
            'predicted_upper_slack_positive': int(soften_x_bounds and any(s_x_high[k, j].X > 1e-6 for k in range(H + 1) for j in range(nx))),
            'solver_status': int(m.Status), 'solver_initial_status': attempts[0]['status'],
            'solver_final_status': int(m.Status), 'numeric_retry_used': int(bool(retries)),
            'solver_retry_count': len(retries), 'inf_or_unbd_disambiguation_used': 0,
            'solver_inf_or_unbd_disambiguation_count': 0,
            'solver_initial_runtime_s': attempts[0]['runtime_s'],
            'solver_retry_runtime_s': sum(a['runtime_s'] for a in retries),
            'solver_inf_or_unbd_disambiguation_runtime_s': 0.0,
            'solver_total_runtime_s': total_solver_runtime,
            'solver_branch_runtime_s': sum(a['runtime_s'] for a in branch_attempts),
            'solver_retry_numeric_focus': max([a['numeric_focus'] for a in retries], default=float('nan')),
            'solver_runtime_s': total_solver_runtime, 'solver_mip_gap': float(m.MIPGap),
            'solver_mip_gap_available': 1,
            'numerical_validation': numerical_report,
        })
    result = {"u0": u0_vec, "z0": z0_val, "pfc_on0": pfc_on0_val, "obj": float(m.ObjVal), "debug": debug, "diagnostic": diagnostic}
    m.dispose()
    return result

class MMVRepoHooks(RepoHooks):
    def __init__(self, data_path: str = None, ac_model_path=None, nv_model_path=None, device: str = 'cpu', control_mask_ac=None, control_mask_nv=None, steps_per_ctrl: int = 1, lin_cache_size: int = 512, cache_round_decimals: int = 4, segment_split: str = 'test', test_date_policy: str = 'intersection', min_seg_len: int = 600):
        self.data_path = str(data_path or RELEASE_ROOT / 'data/private/l14_merged_data_with_rain.csv')
        self.device = torch.device(device)
        self.ac_model, self.ac_scaler_X, self.ac_scaler_U = load_surrogate(str(ac_model_path or RELEASE_ROOT / 'models/thermal/ac_model.pth'), self.device)
        self.nv_model, self.nv_scaler_X, self.nv_scaler_U = load_surrogate(str(nv_model_path or RELEASE_ROOT / 'models/thermal/nv_model.pth'), self.device)
        self.state_cols = AC_STATE_COLS
        self.steps_per_ctrl = steps_per_ctrl
        self.segment_split = segment_split
        self.test_date_policy = test_date_policy
        self.min_seg_len = min_seg_len
        self.control_mask_ac = control_mask_ac if control_mask_ac is not None else AC_CONTROL_IDX
        self.control_mask_nv = control_mask_nv if control_mask_nv is not None else NV_CONTROL_IDX
        self._segments = None
        self._lin_cache = OrderedDict()
        self.lin_cache_size = lin_cache_size
        self.cache_round_decimals = cache_round_decimals
        self.lin_cache_hits = 0
        self.lin_cache_misses = 0

    def _segments_cached(self):
        if self._segments is None:
            self._segments = prepare_segments(self.data_path, max_segments=50, split=self.segment_split, test_policy=self.test_date_policy, min_len=self.min_seg_len)
        return self._segments

    def load_day_dataframe(self, day_idx: int = 0) -> pd.DataFrame:
        segments = self._segments_cached()
        if not segments:
            raise ValueError('No clean segments found in data.')

        if not (0 <= int(day_idx) < len(segments)):
            raise ValueError(f'day_idx must be in [0, {len(segments)-1}], got {day_idx}')
        target = segments[int(day_idx)]

        df = target.copy().set_index('date')
        df['T_out'] = df['OutdoorTemperatureWindow']
        df['I_solar'] = df['Solar Radiation']
        return df

    def _select_model(self, mode: str):
        if mode == 'ac':
            return self.ac_model, self.ac_scaler_X, self.ac_scaler_U
        else:
            return self.nv_model, self.nv_scaler_X, self.nv_scaler_U

    def _control_mask(self, mode: str):
        return self.control_mask_ac if mode == "ac" else self.control_mask_nv

    def _linearization_cache_key(self, mode_name: str, x_oper, u_oper, exog_fore_5min: pd.Series):
        x_key = tuple(np.round(np.asarray(x_oper, dtype=float).reshape(-1), self.cache_round_decimals))
        if mode_name == 'ac':
            u_vec = np.asarray(u_oper, dtype=float).reshape(-1)
            if u_vec.size == 1:
                u_vec = np.repeat(u_vec, len(AC_CONTROL_IDX))
            u_key = tuple(np.round(u_vec, self.cache_round_decimals))
            exog_cols = AC_INPUT_COLS
        else:
            u_vec = np.asarray(u_oper, dtype=float).reshape(-1)[-2:]
            u_key = tuple(np.round(u_vec, self.cache_round_decimals))
            exog_cols = NV_INPUT_COLS_BASE
        exog_vals = tuple(np.round([float(exog_fore_5min.get(col, 0.0)) for col in exog_cols], self.cache_round_decimals))
        return (mode_name, x_key, u_key, exog_vals)

    def _cache_lookup(self, key):
        cached = self._lin_cache.get(key)
        if cached is None:
            self.lin_cache_misses += 1
            return None
        self.lin_cache_hits += 1
        self._lin_cache.move_to_end(key)
        return tuple(np.array(arr, copy=True) for arr in cached)

    def _cache_store(self, key, A5, B5, c5):
        self._lin_cache[key] = (A5, B5, c5)
        self._lin_cache.move_to_end(key)
        if len(self._lin_cache) > self.lin_cache_size:
            self._lin_cache.popitem(last=False)

    def reset_cache(self):
        self._lin_cache.clear()
        self.lin_cache_hits = 0
        self.lin_cache_misses = 0

    def cache_stats(self):
        return {"hits": self.lin_cache_hits, "misses": self.lin_cache_misses, "entries": len(self._lin_cache)}

    def plant_step_1min(self, mode: int, x, u, exog_row: pd.Series):
        mode_name = 'ac' if mode == 1 else 'nv'
        model, sx, su = self._select_model(mode_name)
        u_vec = build_u_vector(exog_row, mode_name, u if mode_name == "ac" else np.asarray(u)[-2:])
        return forward_surrogate(model, sx, su, x, u_vec, self.device)

    def linearize_5min(self, mode: int, x_oper, u_oper, exog_fore_5min: pd.Series):
        mode_name = 'ac' if mode == 1 else 'nv'
        key = self._linearization_cache_key(mode_name, x_oper, u_oper, exog_fore_5min)
        cached = self._cache_lookup(key)
        if cached is not None:
            return cached

        model, sx, su = self._select_model(mode_name)
        u_vec = build_u_vector(exog_fore_5min, mode_name, u_oper if mode_name == "ac" else np.asarray(u_oper)[-2:])
        lin = linearize_surrogate(model, sx, su, x_oper, u_vec, self.device)
        mask = np.asarray(self._control_mask(mode_name), dtype=int)

        # Reduce full-input linearization to control-only form while preserving
        # the linearization point contribution from non-control (exogenous) inputs.
        omit_mask = np.ones(lin.B.shape[1], dtype=bool)
        omit_mask[mask] = False
        B_sel = lin.B[:, mask]
        c_reduced = lin.c + lin.B[:, omit_mask] @ u_vec[omit_mask]

        A5, B5, c5 = lift_linearization(lin.A, B_sel, c_reduced, steps=self.steps_per_ctrl)
        self._cache_store(key, A5, B5, c5)
        return tuple(np.array(arr, copy=True) for arr in (A5, B5, c5))

def make_default_hooks(**kwargs):
    return MMVRepoHooks(**kwargs)

DEADBAND = 0.3

NV_CC_BASE = 29.0

CC_AC_BASE = 30.0

NV_CC_HIGH = NV_CC_BASE + DEADBAND

NV_CC_LOW = NV_CC_BASE - DEADBAND

CC_AC_HIGH = CC_AC_BASE + DEADBAND

CC_AC_LOW = CC_AC_BASE - DEADBAND

AC_ZONE_SETPOINT_C = 27.0
AC_THERMOSTAT_GAIN = 3.0
MIN_MODE_LOCK = 5

@dataclass
class ControllerState:
    mode: str
    last_switch_idx: Optional[int] = None
    rain_hold: int = 0

def pick_initial_rbc_mode(row: pd.Series) -> str:
    oa = float(row['OutdoorTemperatureWindow'])
    rain = float(row['rain_status'])
    if rain >= 1:
        return 'ac'
    if oa >= CC_AC_HIGH:
        return 'ac'
    if oa >= NV_CC_HIGH:
        return 'cc'
    return 'nv'

def update_rbc_mode(row: pd.Series, idx: int, state: ControllerState) -> str:
    oa = float(row['OutdoorTemperatureWindow'])
    rain = float(row['rain_status'])

    if rain >= 1:
        state.rain_hold = RAIN_LOCKOUT
        if state.mode != 'ac':
            state.mode = 'ac'
            state.last_switch_idx = idx
        return state.mode

    if state.rain_hold > 0:
        state.rain_hold -= 1
        if state.mode != 'ac':
            state.mode = 'ac'
            state.last_switch_idx = idx
        return state.mode

    locked = state.last_switch_idx is not None and (idx - state.last_switch_idx) < MIN_MODE_LOCK
    if locked:
        return state.mode

    prev_mode = state.mode
    if state.mode == 'nv':
        if oa >= NV_CC_HIGH:
            state.mode = 'cc'
    elif state.mode == 'cc':
        if oa >= CC_AC_HIGH:
            state.mode = 'ac'
        elif oa <= NV_CC_LOW:
            state.mode = 'nv'
    elif state.mode == 'ac':
        if oa <= CC_AC_LOW:
            state.mode = 'cc'
    else:
        state.mode = 'nv'

    if state.mode != prev_mode:
        state.last_switch_idx = idx
    return state.mode

def ac_supply_for_setpoint(x, cfg: MPCConfig, setpoint_c: float = AC_ZONE_SETPOINT_C) -> np.ndarray:
    mean_temp = float(np.mean(np.asarray(x, dtype=float)))
    cooling_error = max(mean_temp - float(setpoint_c), 0.0)
    fcu_supply = np.clip(cfg.u_max_ac - AC_THERMOSTAT_GAIN * cooling_error, cfg.u_min_ac, cfg.u_max_ac)
    return np.array([fcu_supply] * 5 + [cfg.u_min_pfc, cfg.u_min_pfc], dtype=float)

def control_vector_for_baseline(mode_label: str, x, exog_row: pd.Series, cfg: MPCConfig) -> tuple[int, np.ndarray, int]:
    if mode_label == 'ac':
        return 1, ac_supply_for_setpoint(x, cfg), 0

    u_vec = np.array([np.nan] * 5 + [cfg.u_min_pfc, cfg.u_min_pfc], dtype=float)
    if mode_label == 'nv':
        oa = float(exog_row['OutdoorTemperatureWindow'])
        u_vec[-2:] = [oa, oa]
        return 0, u_vec, 0

    if mode_label == 'cc':
        u_vec[-2:] = [cfg.u_min_pfc, cfg.u_min_pfc]
        return 0, u_vec, 1

    raise ValueError(f'Unknown mode_label={mode_label!r}')

def simulate_policy_day(
    case_name: str,
    hooks: RepoHooks,
    cfg: MPCConfig,
    day_idx: int,
    policy: str,
    start_hhmm: str = '07:30',
    end_hhmm: str = '19:00',
) -> pd.DataFrame:
    df_1min = hooks.load_day_dataframe(day_idx).copy().sort_index()
    date0 = df_1min.index[0].date()
    t_start = pd.Timestamp(f'{date0} {start_hhmm}')
    t_end = pd.Timestamp(f'{date0} {end_hhmm}')
    df_sim = df_1min.loc[t_start:t_end - pd.Timedelta(minutes=1)]
    if df_sim.empty:
        raise ValueError('No data in the requested baseline window.')

    x = df_sim.iloc[0][hooks.state_cols].astype(float).to_numpy(copy=True)
    rbc_state = None
    rows = []

    for i, (ts, exog_row) in enumerate(df_sim.iterrows()):
        if policy == 'ac':
            mode_label = 'ac'
        elif policy == 'rbc':
            if rbc_state is None:
                rbc_state = ControllerState(mode=pick_initial_rbc_mode(exog_row))
            mode_label = update_rbc_mode(exog_row, i, rbc_state)
        else:
            raise ValueError(f'Unknown baseline policy={policy!r}')

        z_mode, u_vec, pfc_on = control_vector_for_baseline(mode_label, x, exog_row, cfg)
        rec = {
            'ts': ts,
            'case': case_name,
            'z': z_mode,
            'mode_label': mode_label.upper(),
            'pfc_on': int(pfc_on),
            'T_out': float(exog_row.get('T_out', exog_row.get('OutdoorTemperatureWindow', np.nan))),
            'I_solar': float(exog_row.get('I_solar', exog_row.get('Solar Radiation', 0.0))),
            'fcu01_supply': float(u_vec[0]) if z_mode == 1 else np.nan,
            'fcu02_supply': float(u_vec[1]) if z_mode == 1 else np.nan,
            'fcu03_supply': float(u_vec[2]) if z_mode == 1 else np.nan,
            'fcu04_supply': float(u_vec[3]) if z_mode == 1 else np.nan,
            'fcu05_supply': float(u_vec[4]) if z_mode == 1 else np.nan,
            'pfc01_supply': float(u_vec[5]),
            'pfc02_supply': float(u_vec[6]),
        }
        for val, name in zip(x, hooks.state_cols):
            rec[name] = float(val)
        rec['T_mean'] = float(np.mean(x))
        rows.append(rec)

        x = np.asarray(hooks.plant_step_1min(mode=z_mode, x=x, u=u_vec, exog_row=exog_row), dtype=float)

    return pd.DataFrame(rows).set_index('ts')

def enrich_results(sim_df: pd.DataFrame, cfg: MPCConfig, case_name: Optional[str] = None, segment_date: Optional[str] = None) -> pd.DataFrame:
    out = sim_df.copy()
    if case_name is not None:
        out['case'] = case_name
    if segment_date is not None:
        out['date'] = segment_date
    out['pv_kw'] = [
        pv_available_power_kw(cfg, I_solar=i_solar, T_out=t_out)
        for i_solar, t_out in zip(out['I_solar'], out['T_out'])
    ]
    out['hvac_kw'] = [
        hvac_power_heat_exchange_kw(
            cfg,
            z_mode=row['z'],
            zone_temps=row[AC_STATE_COLS].to_numpy(dtype=float),
            fcu_supplies=row[FCU_SUPPLY_COLS].fillna(row['T_out']).to_numpy(dtype=float),
            pfc_supplies=row[PFC_SUPPLY_COLS].to_numpy(dtype=float),
            T_out=row['T_out'],
        )
        for _, row in out.iterrows()
    ]
    out['grid_kw'] = np.maximum(out['hvac_kw'] - out['pv_kw'], 0.0)
    out['export_kw'] = np.maximum(out['pv_kw'] - out['hvac_kw'], 0.0)
    out['self_kw'] = np.minimum(out['hvac_kw'], out['pv_kw'])
    return out

def sparse_progress(label: str, segment_date: str):
    def _progress(step_idx: int, total_steps: int, ts: pd.Timestamp) -> None:
        if step_idx == 0 or (step_idx + 1) % 20 == 0 or step_idx + 1 == total_steps:
            print(f'{segment_date} {label}: step {step_idx + 1}/{total_steps} at {ts}')
    return _progress
def simulate_day(
    day_idx: int,
    hooks: RepoHooks,
    cfg: MPCConfig,
    start_hhmm: str = "07:30",
    end_hhmm: str = "19:00",
    init_u: float = 24.0,
    init_z: int = 1,
    init_state_cols: Optional[List[str]] = None,
    state_labels: Optional[List[str]] = None,
    progress_fn: Optional[Callable[[int, int, pd.Timestamp], None]] = None,
    verbose: bool = False,
    forecast_source: str = "observed",
    lstm64_future: Optional[pd.DataFrame] = None,
    forecast_case_name: str = "MIQP",
    forecast_controller_objective: str = "not_applicable",
    forecast_audit_records: Optional[List[pd.DataFrame]] = None,
    decision_diagnostic_records: Optional[list[dict]] = None,
) -> pd.DataFrame:
    df_1min = hooks.load_day_dataframe(day_idx).copy().sort_index()

    date0 = df_1min.index[0].date()
    t_start = pd.Timestamp(f"{date0} {start_hhmm}")
    t_end = pd.Timestamp(f"{date0} {end_hhmm}")
    df_sim = df_1min.loc[t_start:t_end - pd.Timedelta(minutes=1)]
    if df_sim.empty:
        raise ValueError("No data in the requested window.")

    if init_state_cols is None:
        init_state_cols = getattr(hooks, 'state_cols', None) or [df_sim.columns[0]]
    if isinstance(init_state_cols, str):
        init_state_cols = [init_state_cols]

    x = df_sim.iloc[0][init_state_cols].astype(float).to_numpy(copy=True)
    u_prev = np.asarray(init_u, dtype=float).reshape(-1)
    if u_prev.size == 1:
        u_prev = np.full(len(AC_CONTROL_IDX), float(u_prev.item()))
    if u_prev.size != len(AC_CONTROL_IDX):
        raise ValueError(f"init_u must be scalar or length {len(AC_CONTROL_IDX)} vector")
    z_prev = int(init_z)

    if cfg.dt_min != 1 or cfg.ctrl_period_min < 1 or cfg.horizon_min % cfg.ctrl_period_min:
        raise ValueError('Use dt_min=1 and a horizon divisible by the control period.')
    expected = pd.date_range(t_start, t_end - pd.Timedelta(minutes=1), freq='min')
    if not df_sim.index.equals(expected):
        raise ValueError('Requested simulation window must contain every one-minute timestamp.')
    ctrl_step = cfg.ctrl_period_min
    H = cfg.horizon_min // cfg.ctrl_period_min
    times = df_sim.index
    rows = []

    labels = state_labels or getattr(hooks, 'state_cols', None)
    if labels is None:
        labels = [f"x{j}" for j in range(len(x))]

    ctrl_times = list(times[::ctrl_step])
    total_steps = len(ctrl_times)
    lock_steps = 0
    rain_lock_steps = 0

    for step_idx, t in enumerate(ctrl_times):
        if progress_fn is not None:
            progress_fn(step_idx, total_steps, t)
        else:
            print(f"[simulate_day] step {step_idx + 1}/{total_steps} at {t}")



        t_fore_end = t + pd.Timedelta(minutes=cfg.horizon_min)
        df_fore_1min = df_1min.loc[t:t_fore_end - pd.Timedelta(minutes=1)]
        if len(df_fore_1min) < cfg.horizon_min:
            df_fore_1min = df_fore_1min.reindex(
                pd.date_range(t, periods=cfg.horizon_min, freq="1min"),
                method="nearest"
            )

        df_fore_5min = df_fore_1min.resample(f"{cfg.ctrl_period_min}min").mean().iloc[:H]
        future_forecast = apply_future_data_source(
            df_fore_5min,
            forecast_source,
            decision_time=t,
            lstm64_future=lstm64_future,
        )
        df_fore_5min = future_forecast.frame
        current_exog_row = df_1min.loc[t]
        rain_lock_steps_before = rain_lock_steps
        rain_force_ac_now = observed_rain_forces_ac(cfg, current_exog_row, rain_lock_steps_before)
        rain_lock_steps_after = update_observed_rain_lock_steps(cfg, current_exog_row, rain_lock_steps_before)
        known_rain_forced_stages = (1 + rain_lock_steps_after) if current_observed_rain_active(cfg, current_exog_row) else rain_lock_steps_before
        df_fore_5min_mpc = neutralize_rain_for_mpc_prediction(cfg, df_fore_5min)
        if forecast_audit_records is not None:
            forecast_audit_records.append(
                build_forecast_audit(
                    df_fore_5min_mpc,
                    future_forecast.boundary_completed,
                    source=forecast_source,
                    case_name=forecast_case_name,
                    decision_time=t,
                    ctrl_period_min=cfg.ctrl_period_min,
                    controller_objective=forecast_controller_objective,
                    provenance=future_forecast.provenance,
                    base_forecast=future_forecast.frame,
                )
            )
        pre_lock_steps = lock_steps
        effective_lock_steps = 0 if rain_force_ac_now else lock_steps
        if verbose and rain_force_ac_now:
            print(f"[MPC-Explain] observed rain/lockout active; forcing current move to AC (rain lock steps before solve: {rain_lock_steps_before}).")
        elif verbose and pre_lock_steps > 0:
            print(f"[MPC-Explain] dwell lock active for this step (remaining lock steps before solve: {pre_lock_steps}); mode is constrained by dwell policy.")

        try:
            sol = solve_miqp(cfg, x0=x, u_prev=u_prev, z_prev=z_prev, t0=t, exog_5min_forecast=df_fore_5min_mpc, hooks=hooks, lock_steps=effective_lock_steps, verbose=verbose, rain_force_ac_now=rain_force_ac_now,
                collect_diagnostics=decision_diagnostic_records is not None, known_rain_forced_stages=known_rain_forced_stages)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}; step={step_idx + 1}/{total_steps}, t={t}, z_prev={z_prev}, lock_steps={effective_lock_steps}, rain_force_ac_now={rain_force_ac_now}, T_mean={float(np.mean(x)):.3f}"
            ) from exc
        u0_vec, z0 = sol['u0'], sol['z0']

        if verbose and pre_lock_steps == 0 and not rain_force_ac_now:
            # Debug-only true counterfactual: force z0 and re-solve from the same state/forecast.
            try:
                sol_cf_ac = solve_miqp(cfg, x0=x, u_prev=u_prev, z_prev=z_prev, t0=t, exog_5min_forecast=df_fore_5min_mpc, hooks=hooks, lock_steps=0, verbose=True, force_z0=1)
                sol_cf_nv = solve_miqp(cfg, x0=x, u_prev=u_prev, z_prev=z_prev, t0=t, exog_5min_forecast=df_fore_5min_mpc, hooks=hooks, lock_steps=0, verbose=True, force_z0=0)
                dbg = sol.get('debug') or {}
                dbg_ac = sol_cf_ac.get('debug') or {}
                dbg_nv = sol_cf_nv.get('debug') or {}
                dbg['counterfactual_z0'] = {
                    'ac': {
                        'obj': float(sol_cf_ac.get('obj', np.nan)),
                        'x1_mean': float((dbg_ac.get('k0') or {}).get('x_next_mean_ac0', np.nan)),
                        'breakdown': dict(dbg_ac.get('breakdown') or {}),
                    },
                    'nv': {
                        'obj': float(sol_cf_nv.get('obj', np.nan)),
                        'x1_mean': float((dbg_nv.get('k0') or {}).get('x_next_mean_nv0', np.nan)),
                        'breakdown': dict(dbg_nv.get('breakdown') or {}),
                    },
                }
                sol['debug'] = dbg
            except Exception as exc:
                dbg = sol.get('debug') or {}
                dbg['counterfactual_z0_error'] = str(exc)
                sol['debug'] = dbg

            _explain_mpc_decision(step_idx, total_steps, t, sol.get('debug'))
        pfc_on0 = sol.get('pfc_on0', 0)
        if decision_diagnostic_records is not None:
            debug = sol.get('debug') or {}
            breakdown = debug.get('breakdown') or {}
            decision_diagnostic_records.append({
                'numerical_validation': debug.get('numerical_validation'),
                'solver_branch_runtime_s': debug.get('solver_branch_runtime_s', 0.0),
                'date': t.date().isoformat(),
                'case': forecast_case_name,
                'decision_time': t.isoformat(),
                'controller_objective': forecast_controller_objective,
                'forecast_source': forecast_source,
                'w_x_slack': float(getattr(cfg, 'w_x_slack')),
                'chosen_mode_z': int(z0),
                'previous_mode_z': int(z_prev),
                'pfcu_on': int(pfc_on0),
                'dwell_lock_steps_before': int(pre_lock_steps),
                'effective_dwell_lock_steps': int(effective_lock_steps),
                'rain_observed_active': int(
                    current_observed_rain_active(cfg, current_exog_row)
                ),
                'rain_force_ac': int(rain_force_ac_now),
                'rain_lock_steps_before': int(rain_lock_steps_before),
                'rain_lock_steps_after': int(rain_lock_steps_after),
                'objective_total': float(sol.get('obj', np.nan)),
                'objective_comfort': float(breakdown.get('comfort', np.nan)),
                'objective_du': float(breakdown.get('du', np.nan)),
                'objective_switch': float(breakdown.get('switch', np.nan)),
                'objective_energy': float(breakdown.get('energy', np.nan)),
                'objective_x_slack': float(breakdown.get('x_slack', np.nan)),
                'objective_pv': float(breakdown.get('pv', np.nan)),
                'predicted_max_temp_c': float(
                    debug.get('predicted_max_temp_c', np.nan)
                ),
                'predicted_upper_slack_sum': float(
                    debug.get('predicted_upper_slack_sum', np.nan)
                ),
                'predicted_upper_slack_positive': int(
                    debug.get('predicted_upper_slack_positive', 0)
                ),
                'solver_status': int(debug.get('solver_status', -1)),
                'solver_initial_status': int(
                    debug.get('solver_initial_status', -1)
                ),
                'solver_final_status': int(
                    debug.get('solver_final_status', -1)
                ),
                'numeric_retry_used': int(
                    debug.get('numeric_retry_used', 0)
                ),
                'solver_retry_count': int(
                    debug.get('solver_retry_count', 0)
                ),
                'inf_or_unbd_disambiguation_used': int(
                    debug.get('inf_or_unbd_disambiguation_used', 0)
                ),
                'solver_inf_or_unbd_disambiguation_count': int(
                    debug.get(
                        'solver_inf_or_unbd_disambiguation_count', 0
                    )
                ),
                'solver_initial_runtime_s': float(
                    debug.get('solver_initial_runtime_s', np.nan)
                ),
                'solver_retry_runtime_s': float(
                    debug.get('solver_retry_runtime_s', np.nan)
                ),
                'solver_inf_or_unbd_disambiguation_runtime_s': float(
                    debug.get(
                        'solver_inf_or_unbd_disambiguation_runtime_s', np.nan
                    )
                ),
                'solver_total_runtime_s': float(
                    debug.get('solver_total_runtime_s', np.nan)
                ),
                'solver_retry_numeric_focus': float(
                    debug.get('solver_retry_numeric_focus', np.nan)
                ),
                'solver_runtime_s': float(
                    debug.get('solver_runtime_s', np.nan)
                ),
                'solver_mip_gap': float(
                    debug.get('solver_mip_gap', np.nan)
                ),
                'solver_mip_gap_available': int(
                    debug.get('solver_mip_gap_available', 0)
                ),
            })
        min_dwell = int(getattr(cfg, "min_dwell_steps", 0))
        if min_dwell > 0:
            if rain_force_ac_now:
                lock_steps = 0
            elif z0 != z_prev:
                lock_steps = max(min_dwell - 1, 0)
            else:
                lock_steps = max(lock_steps - 1, 0)
        rain_lock_steps = rain_lock_steps_after

        for i in range(min(cfg.ctrl_period_min, int((t_end - t).total_seconds() // 60))):
            exog_row = df_1min.loc[t + pd.Timedelta(minutes=i)]
            x_next = hooks.plant_step_1min(mode=z0, x=x, u=u0_vec, exog_row=exog_row)
            rec = {"ts": t + pd.Timedelta(minutes=i), "forecast_source": forecast_source, "z": z0, "pfc_on": int(pfc_on0), "rain_status": float(exog_row.get('rain_status', 0.0)), "rain_force_ac": int(rain_force_ac_now), "rain_lock_steps_before": int(rain_lock_steps_before), "rain_lock_steps_after": int(rain_lock_steps_after), "T_out": float(exog_row.get('T_out', exog_row.get('OutdoorTemperatureWindow', np.nan))), "I_solar": float(exog_row.get('I_solar', exog_row.get('Solar Radiation', 0.0)))}
            # per-unit supply temps for plotting
            if z0 == 1:
                rec['fcu01_supply'], rec['fcu02_supply'], rec['fcu03_supply'], rec['fcu04_supply'], rec['fcu05_supply'], rec['pfc01_supply'], rec['pfc02_supply'] = [float(v) for v in u0_vec]
            else:
                rec['fcu01_supply'] = rec['fcu02_supply'] = rec['fcu03_supply'] = rec['fcu04_supply'] = rec['fcu05_supply'] = np.nan
                rec['pfc01_supply'] = float(u0_vec[5])
                rec['pfc02_supply'] = float(u0_vec[6])
            for val, name in zip(x, labels):
                rec[name] = float(val)
            rec['T_mean'] = float(np.mean(x))
            rows.append(rec)
            x = np.asarray(x_next, dtype=float)

        u_prev, z_prev = u0_vec, z0

    out = pd.DataFrame(rows).set_index('ts')
    return out
FIXED_MPC_KWARGS = dict(
    T_ref_ac=27.0,
    T_nv_max=30.5,
    T_nv_soft=30.0,
    pmv_model='linear',
    rh_ac_fixed=65.0,
    rh_nv_source_col='OutdoorHumidityWindow',
    pmv_big_m=5.0,
    energy_norm_kwh=10.0,
    cop_fcu=5.0,
    cop_pfc=5.0,
    T_out_nv_ref=30.0,
    min_dwell_steps=12,
    enforce_current_rain_safety=True,
    rain_active_threshold=0.5,
    rain_lockout_min=30,
    ignore_rain_in_mpc_prediction=True,
    linearize_energy=True,
    w_terminal=2e1,
    w_switch=1e1,
    w_x_slack=1e2,
)
RELEASE_ROOT = Path(__file__).resolve().parents[1]
FCU_SUPPLY_COLS = [f'fcu{i:02d}_supply' for i in range(1, 6)]
PFC_SUPPLY_COLS = ['pfc01_supply', 'pfc02_supply']

def build_simple_pmv_cfg(energy_weight=1667.0, smoothness_weight=0.1,
                         pv_self_consumption_weight=20.0, use_pv_objective=False,
                         pv_coefficient_path=None, **overrides):
    settings = dict(FIXED_MPC_KWARGS, w_energy=energy_weight,
                    w_du=smoothness_weight, use_pv_objective=use_pv_objective)
    if use_pv_objective:
        path = pv_coefficient_path or RELEASE_ROOT / 'models/pv/pv_appendix_b_best_model.csv'
        row = pd.read_csv(path).iloc[0]
        settings.update(pv_model='appendix_b', pv_a1=float(row.a1), pv_a2=float(row.a2),
                        pv_a3=float(row.a3), pv_scale_factor=0.015, pv_pmax=None,
                        w_pv_export=pv_self_consumption_weight)
    settings.update(overrides)
    return MPCConfig(**settings)

from .data import load_formatted_data, prepare_segments, get_split_dates
