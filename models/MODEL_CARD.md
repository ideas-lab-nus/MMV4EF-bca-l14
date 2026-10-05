# Released model artifacts

This directory contains fitted models required by the manuscript workflow. It contains no training rows, validation predictions, or credentials.

## Main thermal emulators

`thermal/ac_model.pth` and `thermal/nv_model.pth` are the mode-specific CNN-LSTM checkpoints used by the closed-loop controller. Each checkpoint contains a PyTorch state dictionary, architecture dimensions, and fitted scikit-learn `MinMaxScaler` objects. Clean training notebooks are included because these are main-study models.

The scaler objects require the compatible dependency versions in `requirements.txt` and Python pickle loading. Treat the bundled, checksummed files as trusted release artifacts; never load a replacement checkpoint from an untrusted source.

## PV model

`pv/pv_appendix_b_best_model.csv` is one fitted coefficient row for

```text
P_pv = max(a1 * GHI^2 + a2 * GHI * T_out + a3 * GHI, 0)
```

The PV study is ancillary. Its training data, candidate fits, and training notebook are intentionally excluded. The coefficient row and private aligned holdout inputs are sufficient for the released Figure 17 renderer.

## LSTM64 weather ensemble

`lstm64/checkpoints/univariate/` contains five univariate one-step target models—temperature, relative humidity, wind speed, wind direction, and solar irradiance—for seeds 17, 29, and 43. `lstm64/scalers.json` is required for inference. The inference code validates the family, target, seed, architecture, and payload shape before use.

LSTM64 training code, training histories/losses, cached predictions, metrics, and future-disturbance tables are intentionally excluded. Released checkpoint metadata contains only family, target, one-step flag, seed and selected epoch. The weights and scaler values are unchanged from the final study; checksums reflect removal of ancillary training histories. `scripts/build_lstm64_forecasts.py` recreates the required local outputs from authorized observations without retraining.

The models were developed for the study building and date range represented in the manuscript. They are research artifacts, not safety-certified operational controllers, and should not be assumed to generalize to other buildings or climates.

