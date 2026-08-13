# Figure reproduction

This repository contains the minimum code and released model artifacts needed to reconstruct the manuscript workflow. It deliberately contains no observational data, no simulation result tables, and no credentials. Consequently, `python scripts/reproduce_figures.py --list` works in a clean clone, while numerical figures become runnable only after the private and derived inputs in [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) have been supplied.

The orchestration command copies only final manuscript assets into `figures/`; intermediate plots and tables remain under `outputs/figure_work/`.

## Commands

```text
python scripts/reproduce_figures.py --list
python scripts/reproduce_figures.py --check
python scripts/reproduce_figures.py --check --figures 1 4 17
python scripts/reproduce_figures.py --figures 1-5
python scripts/reproduce_figures.py                 # all 17 figures
```

`--list` and `--check` are read-only. A missing private or derived input is reported as expected release state rather than silently replaced with synthetic data. A normal build stops before running if a selected input or producer is absent.

## Numbered manuscript figures

| Fig. | Manuscript content and subfigures | Producer | Input boundary | Final file(s) in `figures/` |
|---:|---|---|---|---|
| 1 | Measured outdoor conditions and MMV operation: cooling demand/GHI and window status/outdoor temperature. | `scripts/make_aug23_mmv_motivation_figure.py` | Private merged observations, 2024-08-23; 15-minute means. | `mmv_motivation_20240823_15min_cooling.png` |
| 2 | AC/NV CNN-LSTM thermal-model architecture and its nonlinear-propagation/instantaneous-linearization roles. | `scripts/build_emulator_diagram.py` | Data-free diagram. | `emulator_diagram.pdf` |
| 3 | Retained ceiling-fan source image. The supplied bitmap visibly plots probability of acceptability against air speed; this does not match the current manuscript description of fan speed versus measured air speed. | No generator is claimed. The cited source image is deliberately retained as `assets/ceiling_fan.png` and copied verbatim. | Static cited source; resolve the caption/content mismatch and see the rights note below. | `ceiling_fan.png` |
| 4 | Three validation-day rollouts: 2 Oct (AC), 9 Oct (two mode changes), and 10 Oct 2024 (three mode changes). Each constituent plot shows ground truth, nonlinear CNN-LSTM rollout, zero-order-hold affine surrogate, and window-open shading. | `scripts/compare_daily_rollout_methods.py` with the `held-input` five-minute rollout | Private merged observations plus released AC/NV thermal models. | `2024-10-02_rollout.png`; `2024-10-09_rollout.png`; `2024-10-10_rollout.png` |
| 5 | Reference PMV and the selected linear PMV approximation over temperature and relative humidity. | Root `fit_all_pmv_regression.py`, supported by root `fit_zone_pmv_regression.py` | Private merged observations; one accepted RH source for each zone. | `all_temp_rh_pmv_regression.png` |
| 6 | Four-panel closed-loop trajectories for dry, low-PV 24 Sep 2024: (a) temperature, (b) window status, (c) PMV, (d) power/PV. | `scripts/build_observed_future_results.py` | Main-study trajectory bundle plus observed humidity/rain context. | `observed_future_daily_2024-09-24.pdf` |
| 7 | Same four panels for 3 Oct 2024. | Same as Fig. 6. | Same as Fig. 6. | `observed_future_daily_2024-10-03.pdf` |
| 8 | Same four panels for rainy 4 Oct 2024; light-blue spans indicate 283 minutes of raw observed rain. | Same as Fig. 6. | Same as Fig. 6. | `observed_future_daily_2024-10-04.pdf` |
| 9 | Same four panels for the high-switching 10 Oct 2024 case. | Same as Fig. 6. | Same as Fig. 6. | `observed_future_daily_2024-10-10.pdf` |
| 10 | Same four panels for the greatest-PV-energy day, 15 Oct 2024. | Same as Fig. 6. | Same as Fig. 6. | `observed_future_daily_2024-10-15.pdf` |
| 11 | Daily KPI distributions: (a) occupied energy reduction, (b) morning/evening peak shedding, (c) PV self-consumption, (d) self-sufficiency. | `scripts/build_observed_future_results.py` | Same audited 22-day main-study bundle as Figs. 6-10. | `observed_future_energy_flexibility.pdf` |
| 12 | Perfect-versus-LSTM64 trajectories: MPC on 8 Oct and MPC-PV on 9 Oct; eight panels pair temperature, window status, PMV, and power. | `scripts/build_discussion_results.py` | Main-study observed/LSTM64 trajectories, validation table, observed results rebuilt for Figs. 6-11, and LSTM64 evaluation metrics. | `discussion_lstm64_daily_profiles.pdf` |
| 13 | Weather-flexibility associations: (a) energy reduction/temperature, (b) window fraction/rain, (c) MPC-PV energy reduction/PV, (d) MPC-PV self-sufficiency/PV. | `scripts/build_discussion_results.py` | Same bundle as Fig. 12. | `discussion_weather_flexibility.pdf` |
| 14 | Aggregate energy flow: (a) HVAC supplied by PV/grid and self-sufficiency, (b) available PV self-consumed/exported and self-consumption. | `scripts/build_discussion_results.py` | Same bundle as Fig. 12. | `discussion_pv_energy_flow.pdf` |
| 15 | State-slack-weight daily response: MPC on 30 Oct and MPC-PV on 10 Oct; eight panels pair temperature, PMV, window status, and power. | `scripts/build_weight_sensitivity_results.py` | Accepted three-weight sweep (`100`, `1,000`, `10,000`) plus observed humidity/rain context. | `discussion_weight_daily_profiles.pdf` |
| 16 | Aggregate state-slack sensitivity: (a) degree-minutes above 30 °C, (b) samples within \|PMV\| <= 0.5, (c) full-day/morning/evening demand reduction, (d) MPC-PV self-consumption/self-sufficiency. | `scripts/build_weight_sensitivity_results.py` | Same bundle as Fig. 15. | `discussion_weight_sensitivity.pdf` |
| 17 | PV-model chronological holdout: (a) first holdout week actual/predicted time series, (b) full-holdout actual-versus-predicted scatter. | `scripts/build_pv_appendix_figure.py` | Released PV coefficient row plus privately supplied timestamps, measured PV power, GHI, and outdoor-air temperature. The renderer calculates predictions from `a1`, `a2`, and `a3`. | `pv_appendix_b_holdout_plots.png` |

## Workflow boundary

The long controller runs are intentionally separated from figure rendering:

1. Place the private inputs as documented in `docs/DATA_REQUIREMENTS.md`.
2. Use `notebooks/main_study_observed_lstm64.ipynb` to create the main trajectory/summary/validation bundle under `outputs/simulations/main/`.
3. Use `notebooks/temperature_slack_sensitivity.ipynb` to create the accepted sweep bundle under `outputs/simulations/weight_sweep/` when reproducing Figs. 15-16.
4. Place or generate the LSTM64 evaluation table at `outputs/lstm64/metrics_aggregate.csv`.
5. Run `python scripts/reproduce_figures.py --check`, then build the desired figures.

These notebooks may require a locally configured mathematical-programming solver. No solver license or token is stored in the repository.

## Figure 3 provenance and rights

`assets/ceiling_fan.png` is retained solely because no generating code or underlying experimental table was found in the project. It is not represented as a reproducible output. The manuscript attributes the relationship to:

> Lei, Y., Tekler, Z. D., Zhan, S., Miller, C., and Chong, A. (2023), “Experimental evaluation of thermal adaptation and transient thermal comfort in a tropical mixed-mode ventilation context,” *Building and Environment*.

The retained bitmap's axes read "Probability of Acceptable" and "Air Speed (m/s)", while the current manuscript caption describes a fan-speed/air-speed relationship. Resolve that mismatch before submission.

The image's reuse rights were not established from the code repository. Before public redistribution, the maintainer must verify ownership, permission, journal license terms, and whether a newly redrawn figure from authorized source data is required. Retention here documents manuscript provenance; it does not grant downstream reuse rights.

## No-data limitation

The released `.pth` and coefficient files are models, not replacements for measurements. Figs. 1, 4, 5, and 17 require private observational/holdout records. Figs. 6-16 require closed-loop trajectory exports derived from those records. The code therefore reproduces the calculations and rendering once authorized inputs are supplied, but a data-free clone cannot recreate the numerical plots byte-for-byte on its own.
