#!/usr/bin/env python
# coding: utf-8

# ## MIQP MPC (AC/NV)
# Hook the tentative MPC/MIQP framework into the MMV surrogates. Implements the RepoHooks contract (data load, plant step, linearization) using the existing AC/NV CNN-LSTM models and the daily data loader from `rule_based_control_simulation.ipynb`.

# **Usage notes**
# - Control period 5 minutes, 1-minute plant dynamics, 1-hour horizon.
# - Binary mode variable (1=AC, 0=NV) with indicator constraints in Gurobi.
# - Data loader mirrors the clean weekday segments from `l14_merged_data_with_rain.csv` (07:30–19:00, no missing cols).
# - A single scalar control `u` drives all FCU/PFCU supply temperatures; other exogenous signals remain measured.
# - State tracked as the 5 zone temperatures; outputs also include a mean for quick plotting.

# ### Fast variant
# This copy keeps the original MIQP setup but adds two speedups: (1) linearizations use a vectorized Jacobian call instead of repeated backward passes, and (2) 5-minute linear models are cached across solver calls with configurable rounding so repeated operating points avoid autograd work.

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Callable
from collections import OrderedDict
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 4)


# In[2]:


# Surrogate utilities (same architecture as training notebooks)
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


# scaler helpers reused for forward + linearization
def _scaler_params_to_torch(scaler, device: torch.device):
    scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
    offset = torch.as_tensor(scaler.min_, dtype=torch.float32, device=device)
    return scale, offset


@torch.inference_mode()
def forward_surrogate(model: CNNLSTM, scaler_X, scaler_U, x, u, device: torch.device):
    x0 = torch.as_tensor(np.array(x, dtype=np.float32, copy=True), device=device).view(1, -1)
    u0 = torch.as_tensor(np.array(u, dtype=np.float32, copy=True), device=device).view(1, -1)
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
    x0 = torch.as_tensor(np.array(x_bar, dtype=np.float32, copy=True), device=device).view(-1)
    u0 = torch.as_tensor(np.array(u_bar, dtype=np.float32, copy=True), device=device).view(-1)

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
    x_next = model_fn(z0)
    # CUDA + cuDNN LSTM backward does not support the batched Jacobian path.
    use_vectorized_jacobian = device.type != 'cuda'
    J = torch.autograd.functional.jacobian(model_fn, z0, vectorize=use_vectorized_jacobian)

    A = J[:, :nx].detach()
    B = J[:, nx:].detach()
    c = (x_next.detach() - A @ x0.detach() - B @ u0.detach()).detach()
    return Linearization(A=A.cpu().numpy(), B=B.cpu().numpy(), c=c.cpu().numpy())


# In[3]:


# Column definitions + helper utilities
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

# Scalar control drives all FCU/PFCU supply temperatures in each regime
AC_CONTROL_IDX = [1, 2, 3, 4, 5, 6, 7]
NV_CONTROL_IDX = [3, 4]

# Training split definitions aligned with the AC/NV surrogate training notebooks.
AC_TRAIN_COLUMNS = AC_STATE_COLS + AC_INPUT_COLS
NV_TRAIN_COLUMNS = AC_STATE_COLS + [
    'OutdoorTemperatureWindow',
    'Wind Speed',
    'Wind Direction',
    'PFCU-01 Supply Air Temp',
    'PFCU-02 Supply Air Temp',
    'rain_status',
    'Solar Radiation',
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


def _load_filtered_data(data_path: str, mode_value: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if mode_value is not None:
        df = df[df['Z1 Windows Open Close Status'] == mode_value]
    df['date'] = pd.to_datetime(df['date'])
    df = df[
        (df['date'].dt.time >= pd.to_datetime('07:30').time()) &
        (df['date'].dt.time <= pd.to_datetime('19:00').time())
    ]
    df = df[df['date'].dt.weekday < 5]
    df = df[~df['date'].dt.normalize().isin(DATES_TO_REMOVE)]
    df = df.sort_values(by='date').reset_index(drop=True)
    if 'Solar Radiation' in df.columns:
        df.loc[:, 'Solar Radiation'] = df['Solar Radiation'].interpolate(method='nearest', limit_direction='both')
    return df


def _split_segment_dates(df: pd.DataFrame, columns_to_check: List[str], min_len: int = 10) -> Dict[str, List[str]]:
    segments = [seg.reset_index(drop=True) for seg in make_continuous_segments(df) if len(seg) >= min_len]
    clean_segments = [
        seg.dropna(subset=columns_to_check)
        for seg in segments
        if not seg[columns_to_check].isna().any().any()
    ]
    dates = [seg['date'].iloc[0].date().isoformat() for seg in clean_segments]
    split_ratio, val_ratio, _ = SPLIT_RATIOS
    split_index = int(len(clean_segments) * split_ratio)
    val_index = int(len(clean_segments) * (split_ratio + val_ratio))
    return {
        'train': dates[:split_index],
        'val': dates[split_index:val_index],
        'test': dates[val_index:],
    }


def get_split_dates(data_path: str, policy: str = 'intersection') -> Dict[str, List[str]]:
    cache_key = (data_path, policy)
    if cache_key in _SPLIT_DATES_CACHE:
        return _SPLIT_DATES_CACHE[cache_key]

    ac_df = _load_filtered_data(data_path, mode_value=0)
    nv_df = _load_filtered_data(data_path, mode_value=1)
    ac_dates = _split_segment_dates(ac_df, AC_TRAIN_COLUMNS)
    nv_dates = _split_segment_dates(nv_df, NV_TRAIN_COLUMNS)

    if policy == 'intersection':
        combined = {k: sorted(set(ac_dates[k]) & set(nv_dates[k])) for k in ac_dates}
    elif policy == 'union':
        combined = {k: sorted(set(ac_dates[k]) | set(nv_dates[k])) for k in ac_dates}
    elif policy == 'ac':
        combined = {k: sorted(set(ac_dates[k])) for k in ac_dates}
    elif policy == 'nv':
        combined = {k: sorted(set(nv_dates[k])) for k in ac_dates}
    else:
        raise ValueError(f'Unknown test date policy: {policy}')

    _SPLIT_DATES_CACHE[cache_key] = combined
    return combined


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


def prepare_segments(
    data_path: str,
    max_segments: int = 20,
    split: str = 'all',
    test_policy: str = 'intersection',
    min_len: int = 600,
) -> List[pd.DataFrame]:
    df = _load_filtered_data(data_path)
    segments = [seg.reset_index(drop=True) for seg in make_continuous_segments(df) if len(seg) >= min_len]
    clean_segments = [
        seg.dropna(subset=COLUMNS_TO_CHECK)
        for seg in segments
        if not seg[COLUMNS_TO_CHECK].isna().any().any()
    ]

    split_key = _normalize_segment_split(split)
    if split_key != 'all':
        split_dates = get_split_dates(data_path, policy=test_policy)
        if split_key == 'test+val':
            split_set = set(split_dates.get('test', [])) | set(split_dates.get('val', []))
        else:
            split_set = set(split_dates.get(split_key, []))
        clean_segments = [
            seg for seg in clean_segments
            if seg['date'].iloc[0].date().isoformat() in split_set
        ]

    return clean_segments[:max_segments]


# In[ ]:


# MPC core (supports multi-state)
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_OK = True
except Exception:
    GUROBI_OK = False


SPECIFIC_HEAT_AIR = 1005.0  # J/(kg·K)
AIR_DENSITY = 1.225         # kg/m³
FCU_AIRFLOW_CPH = 800.0     # CFH
LOCAL_COOLING_CPH = 1000.0  # CFH
CFH_TO_M3S = 0.0283168 / 3600.0
JOULES_PER_KWH = 3.6e6


class RepoHooks:
    def load_day_dataframe(self, day_idx: int):
        raise NotImplementedError

    def plant_step_1min(self, mode: int, x, u, exog_row: pd.Series):
        raise NotImplementedError

    def linearize_5min(self, mode: int, x_oper, u_oper, exog_fore_5min: pd.Series):
        raise NotImplementedError


@dataclass
class MPCConfig:
    T_ref: float = 29.0
    dt_min: int = 1
    ctrl_period_min: int = 5
    horizon_min: int = 30
    x_min: float = 22.0
    x_max: float = 31.0
    u_min: float = 12.0  # legacy lower bound (kept for compatibility)
    u_min_ac: float = 12.5
    u_min_pfc: float = 21.0
    u_max_ac: float = 27.0
    u_max_pfc: float = 22.0
    w_overheat: float = 100.0
    w_undercool: float = 0.0
    w_du: float = 0.2   
    w_switch: float = 0.02
    w_energy: float = 0.001
    penalize_undercool: bool = False
    use_pv_objective: bool = False
    pv_model: str = "simple_linear"  # options: "simple_linear", "appendix_b"
    pv_alpha: float = 0.002
    pv_pmax: Optional[float] = None
    pv_a1: float = 0.0
    pv_a2: float = 0.0
    pv_a3: float = 0.0
    pv_source_area_sqft: float = 5000.0
    building_area_sqft: float = 750.0
    pv_scale_factor: Optional[float] = None
    w_pv_import: float = 1.0
    w_pv_export: float = 1.0
    p_a0: float = 0.0
    p_aT: float = 0.0
    p_aTout: float = 0.2
    p_aU: float = -0.3
    p_aZ: float = 0.8
    price_low: float = 1.0
    price_high: float = 4.0
    soften_x_bounds: bool = True
    w_x_slack: float = 1e4
    mip_time_limit: Optional[float] = 60
    gurobi_output: int = 1


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


def hvac_power_proxy_kw(cfg: MPCConfig, T_zone_mean: float, T_out: float, u_fcu_mean: float, z_mode: int) -> float:
    power_kw = (
        cfg.p_a0
        + cfg.p_aT * float(T_zone_mean)
        + cfg.p_aTout * float(T_out)
        + cfg.p_aU * float(u_fcu_mean)
        + cfg.p_aZ * float(z_mode)
    )
    return max(power_kw, 0.0)


def solve_miqp(cfg: MPCConfig, x0, u_prev: float, z_prev: int, t0: pd.Timestamp, exog_5min_forecast: pd.DataFrame, hooks: RepoHooks) -> Dict[str, float]:
    if not GUROBI_OK:
        raise RuntimeError("Gurobi not available. Install gurobipy to run the MIQP.")

    x0_vec = np.asarray(x0, dtype=float).reshape(-1)
    nx = x0_vec.shape[0]
    H = len(exog_5min_forecast)
    u_prev_vec = np.asarray(u_prev, dtype=float).reshape(-1)

    m = gp.Model("mmv_mpc")
    m.Params.OutputFlag = int(getattr(cfg, "gurobi_output", 1))
    m.Params.NonConvex = 2
    m.Params.Presolve = 1
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.1
    if cfg.mip_time_limit:
        m.Params.TimeLimit = float(cfg.mip_time_limit)

    soften_x_bounds = bool(getattr(cfg, "soften_x_bounds", True))
    w_x_slack = float(getattr(cfg, "w_x_slack", 1e4))
    write_iis = bool(getattr(cfg, "write_iis_on_infeasible", True))

    ctrl_count = len(AC_CONTROL_IDX)
    fcu_indices = [0, 1, 2, 3, 4]
    pfc_indices = [4, 5]
    u_lower_bound = {(k, j): (cfg.u_min_ac if j in fcu_indices else cfg.u_min_pfc) for k in range(H) for j in range(ctrl_count)}
    u_upper_bound = {(k, j): (cfg.u_max_ac if j in fcu_indices else cfg.u_max_pfc) for k in range(H) for j in range(ctrl_count)}

    x = m.addVars(H + 1, nx, name="x")
    u = m.addVars(H, ctrl_count, lb=u_lower_bound, ub=u_upper_bound, name="u")
    z_mode = m.addVars(H, vtype=GRB.BINARY, name="z_mode")
    e_pos = m.addVars(H + 1, nx, lb=0.0, name="e_pos")
    e_neg = m.addVars(H + 1, nx, lb=0.0, name="e_neg")
    sw = m.addVars(H, lb=0.0, name="sw")
    delta_cool = m.addVars(H, lb=0.0, name="delta_cool")

    s_x_low = m.addVars(H + 1, nx, lb=0.0, name="s_x_low")
    s_x_high = m.addVars(H + 1, nx, lb=0.0, name="s_x_high")

    step_seconds = cfg.ctrl_period_min * 60
    fcu_energy_coeff = 5 * SPECIFIC_HEAT_AIR * AIR_DENSITY * (FCU_AIRFLOW_CPH * CFH_TO_M3S) * step_seconds / JOULES_PER_KWH
    local_energy_coeff = 2 * SPECIFIC_HEAT_AIR * AIR_DENSITY * (LOCAL_COOLING_CPH * CFH_TO_M3S) * step_seconds / JOULES_PER_KWH

    if cfg.use_pv_objective:
        p_hvac = m.addVars(H, lb=-GRB.INFINITY, name="p_hvac")
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
        for j in range(nx):
            m.addConstr(e_pos[k, j] >= x[k, j] - cfg.T_ref)
            m.addConstr(e_neg[k, j] >= cfg.T_ref - x[k, j])

    for k in range(H):
        if k == 0:
            m.addConstr(sw[k] >= z_mode[k] - z_prev)
            m.addConstr(sw[k] >= z_prev - z_mode[k])
        else:
            m.addConstr(sw[k] >= z_mode[k] - z_mode[k - 1])
            m.addConstr(sw[k] >= z_mode[k - 1] - z_mode[k])

    for k in range(H):
        exog_k = exog_5min_forecast.iloc[k]
        u_fcu_mean = gp.quicksum(u[k, j] for j in fcu_indices) / float(len(fcu_indices))
        for j in range(nx):
            m.addConstr(delta_cool[k] >= x[k, j] - u_fcu_mean)
        m.addConstr(delta_cool[k] >= 0.0)

        A_ac, B_ac, g_ac = hooks.linearize_5min(1, x0_vec, u_prev_vec, exog_k)
        A_nv, B_nv, g_nv = hooks.linearize_5min(0, x0_vec, u_prev_vec[-2:], exog_k)

        for j in range(nx):
            lhs_ac = gp.quicksum(A_ac[j, i] * x[k, i] for i in range(nx)) + gp.quicksum(B_ac[j, l] * u[k, l] for l in range(ctrl_count)) + float(g_ac[j])
            lhs_nv = gp.quicksum(A_nv[j, i] * x[k, i] for i in range(nx)) + gp.quicksum(B_nv[j, idx] * u[k, pfc_indices[idx]] for idx in range(len(pfc_indices))) + float(g_nv[j])
            m.addGenConstrIndicator(z_mode[k], True, x[k+1, j] - lhs_ac == 0.0)
            m.addGenConstrIndicator(z_mode[k], False, x[k+1, j] - lhs_nv == 0.0)

        if cfg.use_pv_objective:
            T_out = float(exog_k.get("T_out", exog_k.get("OutdoorTemperatureWindow", 0.0)))
            p_expr = cfg.p_a0 + cfg.p_aTout * T_out + cfg.p_aZ * z_mode[k]
            for j in range(nx):
                p_expr += cfg.p_aT * x[k, j]
            p_expr += cfg.p_aU * (gp.quicksum(u[k, j] for j in fcu_indices) / float(len(fcu_indices)))
            m.addConstr(p_hvac[k] == p_expr)

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
    for k in range(H + 1):
        for j in range(nx):
            obj += cfg.w_overheat * e_pos[k, j] * e_pos[k, j]
            obj += cfg.w_overheat * s_x_high[k, j] * s_x_high[k, j]
            if getattr(cfg, 'penalize_undercool', False) and cfg.w_undercool > 0:
                obj += cfg.w_undercool * e_neg[k, j] * e_neg[k, j]
    obj += cfg.w_du * du_cost
    obj += cfg.w_switch * gp.quicksum(sw[k] for k in range(H))
    m.addConstr(gp.quicksum(sw[k] for k in range(H)) <= 1.0)

    energy_cost = 0.0
    for k in range(H):
        energy_fcu = fcu_energy_coeff * delta_cool[k]
        energy_local = local_energy_coeff * delta_cool[k]
        # energy_cost += z_mode[k] * energy_fcu + (1.0 - z_mode[k]) * energy_local
        energy_cost += z_mode[k] * energy_fcu + energy_local
    obj += cfg.w_energy * energy_cost

    if cfg.use_pv_objective:
        for k in range(H):
            ts_k = t0 + pd.Timedelta(minutes=cfg.ctrl_period_min * k)
            price = cfg.price_high if time_in_windows(ts_k) else cfg.price_low
            obj += cfg.w_pv_import * price * p_grid[k]
            obj += cfg.w_pv_export * p_export[k]

    if soften_x_bounds:
        obj += w_x_slack * gp.quicksum(s_x_low[k, j] + s_x_high[k, j] for k in range(H + 1) for j in range(nx))

    m.setObjective(obj, GRB.MINIMIZE)
    # m.Params.DualReductions = 0
    m.optimize()

    if m.Status == GRB.INFEASIBLE and write_iis:
        m.computeIIS()
        m.write("mpc_miqp_iis.ilp")
        print("[MPC] Infeasible: wrote IIS to mpc_miqp_iis.ilp")

    has_usable_incumbent = m.Status == GRB.TIME_LIMIT and m.SolCount > 0
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and not has_usable_incumbent:
        raise RuntimeError(f"MPC solve failed with status {m.Status}")

    return {"u0": np.array([float(u[0, j].X) for j in range(ctrl_count)]), "z0": int(round(z_mode[0].X)), "obj": float(m.ObjVal)}


def simulate_day(day_idx: int, hooks: RepoHooks, cfg: MPCConfig, start_hhmm: str = "07:30", end_hhmm: str = "19:00", init_u: float = 24.0, init_z: int = 1, init_state_cols: Optional[List[str]] = None, state_labels: Optional[List[str]] = None, progress_fn: Optional[Callable[[int, int, pd.Timestamp], None]] = None) -> pd.DataFrame:
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

    x = df_sim.iloc[0][init_state_cols].astype(float).to_numpy()
    u_prev = np.asarray(init_u, dtype=float).reshape(-1)
    if u_prev.size == 1:
        u_prev = np.full(len(AC_CONTROL_IDX), float(u_prev.item()))
    if u_prev.size != len(AC_CONTROL_IDX):
        raise ValueError(f"init_u must be scalar or length {len(AC_CONTROL_IDX)} vector")
    z_prev = int(init_z)

    ctrl_step = cfg.ctrl_period_min
    H = cfg.horizon_min // cfg.ctrl_period_min
    times = df_sim.index
    rows = []

    labels = state_labels or getattr(hooks, 'state_cols', None)
    if labels is None:
        labels = [f"x{j}" for j in range(len(x))]

    ctrl_times = list(times[::ctrl_step])
    total_steps = len(ctrl_times)

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
        sol = solve_miqp(cfg, x0=x, u_prev=u_prev, z_prev=z_prev, t0=t, exog_5min_forecast=df_fore_5min, hooks=hooks)
        u0_vec, z0 = sol['u0'], sol['z0']

        for i in range(cfg.ctrl_period_min):
            exog_row = df_1min.loc[t + pd.Timedelta(minutes=i)]
            x_next = hooks.plant_step_1min(mode=z0, x=x, u=u0_vec, exog_row=exog_row)
            rec = {"ts": t + pd.Timedelta(minutes=i), "z": z0, "T_out": float(exog_row.get('T_out', exog_row.get('OutdoorTemperatureWindow', np.nan))), "I_solar": float(exog_row.get('I_solar', exog_row.get('Solar Radiation', 0.0)))}
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


# In[5]:


# Repo-specific hooks wired to existing surrogates and data loader
class MMVRepoHooks(RepoHooks):
    def __init__(self, data_path: str = 'data/private/l14_merged_data_with_rain.csv', control_mask_ac=None, control_mask_nv=None, steps_per_ctrl: int = 5, lin_cache_size: int = 512, cache_round_decimals: int = 2, device=None, segment_split: str = 'all', test_date_policy: str = 'intersection', min_seg_len: int = 600, ac_model_path: str | Path | None = None, nv_model_path: str | Path | None = None):
        self.data_path = data_path
        self.device = torch.device('cpu' if device is None else device)
        model_root = Path(__file__).resolve().parent / 'models' / 'thermal'
        ac_model_path = model_root / 'ac_model.pth' if ac_model_path is None else Path(ac_model_path)
        nv_model_path = model_root / 'nv_model.pth' if nv_model_path is None else Path(nv_model_path)
        self.ac_model, self.ac_scaler_X, self.ac_scaler_U = load_surrogate(str(ac_model_path), self.device)
        self.nv_model, self.nv_scaler_X, self.nv_scaler_U = load_surrogate(str(nv_model_path), self.device)
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
            self._segments = prepare_segments(
                self.data_path,
                max_segments=50,
                split=self.segment_split,
                test_policy=self.test_date_policy,
                min_len=self.min_seg_len,
            )
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
        mask = self._control_mask(mode_name)
        B_sel = lin.B[:, mask]
        A5, B5, c5 = lift_linearization(lin.A, B_sel, lin.c, steps=self.steps_per_ctrl)
        self._cache_store(key, A5, B5, c5)
        return tuple(np.array(arr, copy=True) for arr in (A5, B5, c5))


# Convenience factory
def make_default_hooks(**kwargs):
    return MMVRepoHooks(**kwargs)


# In[6]:


# Optional quick-start example (guarded if Gurobi is available)
if __name__ == "__main__":
    hooks = make_default_hooks()
    cfg = MPCConfig(use_pv_objective=False)
    print(f"Prepared {len(hooks._segments_cached())} clean day segments. Gurobi available: {GUROBI_OK}")

    run_example = True  # set to False to skip the example run
    if run_example and GUROBI_OK:
        sim_df = simulate_day(day_idx=10, hooks=hooks, cfg=cfg, init_state_cols=hooks.state_cols)
        sim_df[['T_mean']].plot(title='Zone mean temperature')
        plt.figure()
        plt.step(sim_df.index, sim_df['z'], where='post')
        plt.title('Mode (1=AC, 0=NV)')
        plt.figure()
        supply_cols = ['fcu01_supply', 'fcu02_supply', 'fcu03_supply', 'fcu04_supply', 'fcu05_supply', 'pfc01_supply', 'pfc02_supply']
        sim_df[supply_cols].plot(ax=plt.gca(), title='Supply temperatures (FCU/PFCU)')
        plt.ylabel('Supply Temp (C)')
        plt.show()
