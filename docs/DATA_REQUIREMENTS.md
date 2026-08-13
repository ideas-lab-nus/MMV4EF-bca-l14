# Data requirements

No study data is distributed with this repository. Do not commit files placed under `data/private/` or generated under `outputs/`. The paths below are the portable interface between the public code and data held by the study team.

Column names are case-sensitive. Tables may contain additional columns; the lists below are the minimum required schema or, where noted, the recommended raw-data superset. No sample rows are included here.

## Expected placement

| Path | Status | Purpose |
|---|---|---|
| `data/private/l14_merged_data_with_rain.csv` | Private, user supplied | Minute-scale building, weather, mode, humidity, and equipment observations used by Figs. 1, 4, 5, and as context for Figs. 6-16. |
| `data/private/pv_appendix_b.csv` | Private, user supplied | Aligned source-system PV measurements and weather inputs used only to render Fig. 17. |
| `models/thermal/ac_model.pth` | Released artifact | AC CNN-LSTM model and embedded normalization state. |
| `models/thermal/nv_model.pth` | Released artifact | NV CNN-LSTM model and embedded normalization state. |
| `models/pv/pv_appendix_b_best_model.csv` | Released artifact | Selected PV coefficient model and holdout metadata. |
| `models/lstm64/checkpoints/.../seed_*.pt` and `models/lstm64/scalers.json` | Released artifacts | Three-seed LSTM64 ensembles for temperature, RH, solar, wind speed, and wind direction. |
| `outputs/simulations/main/` | Derived locally | Main observed-future and LSTM64 closed-loop trajectory bundle. |
| `outputs/simulations/weight_sweep/` | Derived locally | Accepted three-point state-slack sweep bundle. |
| `outputs/lstm64/metrics_aggregate.csv` | Derived locally | LSTM64 validation/test error table consumed by the discussion renderer. |

Timestamps are local Singapore civil time and must be timezone-naive in the CSVs. The full evaluation contract uses 22 dates, one-minute samples from 07:30 through 18:59 inclusive, and 690 rows per date/case.

## Private merged observations

The direct figure scripts consume the following union schema from `data/private/l14_merged_data_with_rain.csv`.

| Column(s) | Unit/domain | Used for |
|---|---|---|
| `date` | Local timestamp; accepted raw forms include `M/D/YY H:MM` and `M/D/YYYY H:MM`; observed-context validation requires the latter exactly | Alignment and daily selection |
| `Zone 1 Temperature` ... `Zone 5 Temperature` | °C | Thermal states, rollouts, PMV |
| One RH column per zone: `Zone N RH`, `Zone N Humidity`, `Zone N Relative Humidity`, or fallback `FCU-0N Return Air Humi` | % RH | Fig. 5 PMV reference calculation |
| `OutdoorTemperatureWindow` | °C | Weather boundary and Fig. 1 |
| `OutdoorHumidityWindow` | % RH | Window-open PMV reconstruction for Figs. 6-16 |
| `Wind Speed` | Source units are cm/s; LSTM64 weather inference divides by 100 to obtain m/s | NV thermal-model input and LSTM64 history |
| `Wind Direction` | Source units are 0.1 degrees; LSTM64 weather inference divides by 10 to obtain degrees | NV thermal-model input and LSTM64 history |
| `Solar Radiation` | W/m² global horizontal irradiance | AC/NV model input, PV model, Fig. 1 |
| `rain_status` | Binary 0/1; values >= 0.5 are treated as rain in result plots | Rain constraint/context |
| `FCU-01 Supply Air Temp`, `FCU-02 Supply Air Temp - 1 min`, `FCU-03 Supply Air Temp`, `FCU-04 Supply Air Temp`, `FCU-05 Supply Air Temp` | °C | AC thermal-model inputs |
| `PFCU-01 Supply Air Temp`, `PFCU-02 Supply Air Temp` | °C | AC/NV thermal-model inputs |
| `Z1 Windows Open Close Status` | Binary; source convention 1 = open | Mode segmentation and thermal rollout |
| `Z2`, `Z3`, `Z5`, `Z6`, `Z7 Windows Open Close Status` | Binary; 1 = open | Median aggregate window state in Fig. 1 |
| `FCU-1` ... `FCU-5 Cooling Load_kW`; `PFCU-1`, `PFCU-2 Cooling Load_kW` | kW, clipped at zero by the renderer | Fig. 1 cooling power |

If Fig. 1 is deliberately changed to `cooling_plus_fan`, the following optional columns become mandatory and are interpreted as watts: `FCU-01 Watt` ... `FCU-05 Watt`, `PFCU-01 Watt`, and `PFCU-02 Watt`. The manuscript reproduction command uses cooling only.

This is a figure-workflow superset, not a public data dictionary for every raw point in the original building-management export.

## Main-study trajectory bundle

Place the following files directly in `outputs/simulations/main/`, retaining the exact stem `future_data_source_comparison_test_val_union_all_full`.

### `_timeseries.csv`

Minimum shared schema:

| Column(s) | Unit/domain |
|---|---|
| `ts` | Timezone-naive local timestamp, one-minute spacing |
| `case` | Case label. Required observed cases: `AC baseline`, `RBC baseline`, `MIQP no PV \| observed future`, `MIQP onsite PV \| observed future`. Required forecast-comparison cases add `MIQP no PV \| LSTM64` and `MIQP onsite PV \| LSTM64`. |
| `controller_objective` | `baseline_ac`, `baseline_rbc`, `no_pv`, or `onsite_pv`, consistent with `case` |
| `forecast_source` | `not_applicable`, `observed`, or `lstm64`, consistent with `case` |
| `z` | Binary mode state: `1` = closed-window AC; `0` = window-open NV/PFCU |
| `T_out`, `T_mean` | °C |
| `Zone 1 Temperature` ... `Zone 5 Temperature` | °C |
| `fcu01_supply` ... `fcu05_supply` | °C; finite whenever `z = 1` |
| `pfc01_supply`, `pfc02_supply` | °C; finite whenever `z = 0` |
| `pv_kw`, `hvac_kw`, `grid_kw`, `export_kw`, `self_kw` | kW, nonnegative. Accounting must satisfy `grid=max(hvac-pv,0)`, `export=max(pv-hvac,0)`, and `self=min(hvac,pv)`. |

The result scripts independently reconstruct `hvac_kw` using the manuscript sensible heat balance and require agreement within `1e-9 kW`. Available PV must be identical across cases at a given timestamp.

### `_daily_summary.csv`

Minimum columns: `date`, `case`, `hvac_kwh`, `pv_available_kwh`, `grid_kwh`, `export_kwh`, `self_consumed_kwh`, `self_consumption_ratio`, `self_sufficiency_ratio`, and `switch_count`.

Energy columns are kWh. The two `*_ratio` fields are fractions on `[0,1]`, not percentages. One row is required per date/case.

### `_aggregate_summary.csv`

Minimum columns: `case`, `hvac_kwh`, `pv_available_kwh`, `grid_kwh`, `export_kwh`, `self_consumed_kwh`, `self_consumption_ratio`, and `self_sufficiency_ratio`. Units match the daily summary.

### `_validation.csv`

Minimum columns: `date`, `case`, `expected_rows`, `actual_rows`, `exact_index`, `missing_required_columns`, `required_missing_values`, and `rain_lockout_violations`. Discussion rendering uses the subset `date`, `case`, `expected_rows`, `actual_rows`, `exact_index`, and `rain_lockout_violations`. All selected rows must be complete, index-exact, and have zero rain-lockout violations.

## LSTM64 evaluation metrics

`outputs/lstm64/metrics_aggregate.csv` must contain:

| Column | Domain/unit |
|---|---|
| `split` | `validation` or `test` |
| `model` | `univariate_lstm64` |
| `seed` | `ensemble` for the rows used in the manuscript |
| `target` | `temperature`, `relative_humidity`, `solar`, `wind_speed`, or `wind_direction` |
| `metric` | `mae` for the first four targets; `circular_mae` for wind direction |
| `value` | °C, percentage points RH, W/m², m/s, or degrees, according to target |
| `eligible_count` | Count of evaluated forecast points |
| `coverage` | Fraction on `[0,1]` |

The released checkpoints and scalers do not contain these evaluation rows. They must be recomputed from authorized observations or supplied as a derived result.

## Temperature-slack sweep bundle

Place these five exact filenames directly under `outputs/simulations/weight_sweep/`:

- `wx_sweep_timeseries.csv.gz`
- `wx_sweep_daily_metrics.csv`
- `wx_sweep_pooled_metrics.csv`
- `wx_sweep_validation.csv`
- `wx_sweep_run_metadata.json`

The renderer requires all five for provenance, although it recomputes the manuscript metrics from `wx_sweep_timeseries.csv.gz`.

The compressed minute table uses the main-study trajectory schema plus `w_x_slack`. It must contain exactly the fixed `AC baseline` and `RBC baseline` cases plus both MPC objectives at weights `100`, `1000`, and `10000`. Baselines have a missing sweep weight; MPC rows use `forecast_source=observed`. `w_x_slack` is dimensionless. Full-study shape is 22 dates x 8 cases x 690 rows.

`wx_sweep_validation.csv` requires `check` and `passed`; every row must pass, including named checks for rain lockout, dwell lock, and PFCU-disabled-in-AC safety. `wx_sweep_run_metadata.json` must record a CPU PyTorch runtime, `cuda_available=false`, the three weights in order, `validation_passed=true`, and `window_mode=full`. The runner supplies the private observed-context path explicitly, so stale absolute paths inside historical metadata are not used.

## PV holdout input and model

`data/private/pv_appendix_b.csv` has the exact minimum schema:

| Column | Unit/domain |
|---|---|
| `timestamp` | Chronologically sortable timestamp |
| `pv_power_kw` | Actual source-system PV power, kW |
| `ghi_wm2` | Global horizontal irradiance, W/m² |
| `outdoor_air_temp_c` | Outdoor-air temperature, °C |

The released `models/pv/pv_appendix_b_best_model.csv` must retain these fields: `target_name`, `target_col`, `ghi_name`, `ghi_col`, `temp_name`, `temp_col`, `sample_count`, `train_count`, `test_count`, `train_rmse_kw`, `test_rmse_kw`, `train_mae_kw`, `test_mae_kw`, `train_r2`, `test_r2`, `a1`, `a2`, and `a3`. The renderer computes `a1 * ghi_wm2^2 + a2 * ghi_wm2 * outdoor_air_temp_c + a3 * ghi_wm2`, clips predictions at zero, and uses `test_count` to select the chronological tail. The plotted values are for the source PV system before the manuscript's `0.015` testbed-area scaling.

The underlying PV training/holdout records are not released. The coefficient row alone cannot recreate Fig. 17.

## Credentials and sensitive material

- Do not place a `gurobi.lic`, token, API key, session cookie, or credential-bearing configuration in this repository.
- Configure a solver through its normal per-user or environment mechanism before executing controller notebooks.
- Keep raw exports and all generated trajectory/result tables outside version control.
- Before sharing any derived table, review whether its minute-resolution timestamps or operational signals can disclose site behavior.
