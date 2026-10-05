# Final-study source update

The release now uses the corrected one-switch controller from the final study. The controller-comparison configuration uses a one-minute nonlinear plant, five-minute decisions, a 60-minute prediction horizon, at most one predicted mode change, and a 60-minute ordinary dwell. Current observed rain overrides an NV dwell and starts a 30-minute lockout. Known remaining rain-hold stages are enforced in the horizon; future rain is neutralized in the predictor.

The shared `mmv4ef/controller.py` implementation serves both study notebooks and `scripts/run_study.py`. It retains the final comfort, heat-balance and PV-allocation calculations, energy weight 1667 with a 10 kWh normalization, smoothness weight 0.1, terminal weight 20, and state-slack weights 100/1000/10000. Numerical validation checks optimality gap and primal/integer violations, retries questionable solves, and checks an alternative first-mode branch when unlocked. Solver timing includes those additional solves. CPU execution uses one PyTorch thread and two Gurobi threads.

Thermal retraining uses past-only solar filling, training-only normalization and recursive-validation selection. The original retained training sets and chronological partition boundaries are documented in `DATA_REQUIREMENTS.md`. Training outputs are local and do not overwrite the released final-study thermal checkpoints.

The 15 weather checkpoints missing from the original repository are included. They contain the final weights and minimal inference metadata; ancillary training histories/losses have been removed. Their weights match the study checkpoints exactly. The existing thermal checkpoints, PV model coefficients and weather scalers are unchanged. `models/SHA256SUMS.txt` covers all 19 fitted parameter artifacts.

Figure builders use the revised display names, common PV availability, consistent forecast/weight encodings and temperature limits that cover the plotted data. Forecast examples use the largest matched window-state differences with deterministic date tie-breaking; this exploratory rule is recorded in generated selection tables. Sensitivity results are reconciled against the corresponding current observed-future baseline, rather than an obsolete fixed energy total. The weight-only figure command now builds its observed-future reference first.

## Release validation

- Input/schema/model preflight passed for the 22 formatted comparison days.
- Weather inference regenerated all 36,432 horizon rows and evaluation metrics.
- The main-study smoke run completed all six cases with the full 60-minute horizon; its 540 minute rows matched the corresponding final-study trajectories to floating-point precision, with identical mode choices.
- The sensitivity smoke run completed all eight cases and passed its rain, dwell, PFCU and numerical-provenance checks.
- Both thermal modes completed a one-epoch training/recursive-validation check. This checks execution, not a replacement selection of the final model.
- Revised observed-future, forecast/discussion and sensitivity figure pipelines validated/rendered the saved full 22-day results locally. The full 22-day controller optimization was not repeated for this source-only release.
- Unit tests, model checksums, clean-notebook checks and the release privacy audit pass.

No measured data, minute trajectories, training histories, forecast tables, generated figures, solver licenses, institutional acquisition scripts, login details or local-user paths are included in this update. Reproduction requires authorized formatted inputs and an independently configured Gurobi license. The Python requirements select the CPU build from the [official PyTorch wheel index](https://download.pytorch.org/whl/cpu/torch/).
