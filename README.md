# MMV4EF: data-free open-source release

MMV4EF is an open-source toolkit for data-driven model predictive control and energy-flexibility analysis in a tropical mixed-mode office building. This repository provides a compact workflow for training the primary AC/NV thermal emulators, running observed- and forecast-driven control studies, and generating the project figures from user-supplied inputs.

It intentionally contains **no building observations, PV training records, simulation result tables, or credentials**. Numerical figures therefore require the authorized private inputs described in [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md). The public clone never substitutes synthetic data.

## What is included

- AC and NV CNN-LSTM checkpoints plus clean training notebooks for the main study.
- The fitted three-coefficient PV model. PV training data and training notebook are omitted.
- Fifteen LSTM64 checkpoints (five weather targets, three seeds), the required scaler, and inference/evaluation code. The LSTM64 model-training pipeline and prediction tables are omitted.
- The observed/LSTM64 main-study and three-weight sensitivity notebooks.
- Figure builders and one entry point covering all 17 released figures.
- A pinned Python environment and checksums for every released model artifact.

`assets/ceiling_fan.png` is the only non-generated figure source. No generating data or code was found for Figure 3; its provenance and redistribution caveat are documented in [`assets/README.md`](assets/README.md).

## Quick start

Python 3.11 is the supported version.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/reproduce_figures.py --list
python scripts/reproduce_figures.py --check
```

The final command is read-only. In a clean clone it will identify the deliberately absent private and derived inputs while confirming that the released producers and models are present.

The accepted Figures 15-16 sensitivity run used a CPU-only PyTorch build, and its reporter enforces that provenance gate. If your PyTorch package reports CUDA support, install the matching CPU-only `torch==2.12.0` wheel before running `notebooks/temperature_slack_sensitivity.ipynb`.

## Private inputs

Keep private files at these ignored paths:

```text
data/private/l14_merged_data_with_rain.csv
data/private/pv_appendix_b.csv
```

The exact column contracts and units are in [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md). Do not force-add anything under `data/`, `outputs/`, or `figures/` to version control.

## Reproduction order

1. Generate LSTM64 forecasts and validation metrics from the authorized L14 observations:

   ```bash
   python scripts/build_lstm64_forecasts.py
   ```

2. Run `notebooks/main_study_observed_lstm64.ipynb`. It evaluates AC27, RBC, MPC, and MPC-PV with observed future disturbances, plus the two MPC variants with the released LSTM64 forecasts. Its result bundle is written under `outputs/simulations/main/`.

3. For Figures 15-16, run `notebooks/temperature_slack_sensitivity.ipynb`. It varies only the state-slack weight over `100`, `1000`, and `10000`, writing to `outputs/simulations/weight_sweep/`.

4. Rebuild every available figure, or a selected range:

   ```bash
   python scripts/reproduce_figures.py
   python scripts/reproduce_figures.py --figures 1-5
   ```

The detailed figure-to-code map, output names, and input boundary are in [`docs/FIGURE_REPRODUCTION.md`](docs/FIGURE_REPRODUCTION.md).

The controller notebooks solve mixed-integer quadratic programs and can take substantial time for all 22 dates. Use their documented smoke-run environment variables before a full run.

## Gurobi license

`gurobipy` is the Python interface to the proprietary Gurobi Optimizer and requires your own valid license. Configure that license through Gurobi's normal per-user installation or an external `GRB_LICENSE_FILE`. Never place `gurobi.lic` or any WLS credential inside this repository. See [`SECURITY.md`](SECURITY.md).

## Checkpoint trust

The LSTM64 loader uses PyTorch's tensor-only loading mode. The legacy AC/NV checkpoints also contain scikit-learn scaler objects, so their compatible loader must use Python pickle semantics. Load only the two checksummed thermal checkpoints shipped here; do not use that loader on an untrusted `.pth` file. Versions are pinned to keep those scaler objects compatible.

Verify all released model files with:

```bash
python scripts/verify_model_artifacts.py
```

## Release checks

Before sharing changes, run:

```bash
python scripts/verify_model_artifacts.py
python scripts/reproduce_figures.py --check
```

The model verifier checks the released fitted artifacts, while the figure preflight confirms that required producers are present and reports any deliberately absent private inputs.

## Repository layout

```text
assets/       retained non-generated figure source and rights note
docs/         data contract and figure reproduction map
mmv4ef/       LSTM64 inference/evaluation modules
models/       released AC/NV, PV, and LSTM64 fitted artifacts only
notebooks/    main study, sensitivity study, and AC/NV training
scripts/      inference and figure-building entry points
```

## License

The code is released under the [`MIT License`](LICENSE). That license does not provide a Gurobi license, rights to private study data, or permission to redistribute third-party material. Verify the Figure 3 asset rights before making the repository public.
