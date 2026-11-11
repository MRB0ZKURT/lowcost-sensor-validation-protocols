# Validation protocols for machine-learning calibration of low-cost gas sensor arrays

Analysis code for the manuscript *"The validation protocol determines both the reported
accuracy and the selected model in machine-learning calibration of low-cost gas sensor
arrays"*, submitted to *Atmospheric Measurement Techniques*.

The code reproduces every number, table and figure in the manuscript from the two public
datasets, which are downloaded automatically by stage 1.

## What the study does

Seven regression models are evaluated under three validation protocols — random
cross-validation, monthly forward-chaining cross-validation, and a deployment split in
which six months of calibration are followed by seven months of unaltered use — on a
one-year hourly co-location record of a metal-oxide sensor array. The comparison is
repeated with protocol-matched nested hyperparameter search, decomposed into
time-since-calibration and seasonal components with a rolling-origin design, expressed
in terms of the EU data-quality objectives, and replicated on an independent 16-sensor
array recorded over 36 months.

## Data

Both datasets come from the UCI Machine Learning Repository and are downloaded by
stage 1; nothing needs to be obtained manually.

| Dataset | URL | SHA-256 prefix of the archive |
|---|---|---|
| Air Quality (De Vito et al.) | https://archive.ics.uci.edu/dataset/360 | `d4a64013fb385288` |
| Gas Sensor Array Drift (Vergara et al.) | https://archive.ics.uci.edu/dataset/224 | `91e8f466f202e7a0` |

Two columns of the first dataset are dropped before analysis: `NMHC(GT)`, which is
90.2 % missing, and `C6H6(GT)`, which is a near-deterministic log transform of the
`PT08.S2` sensor response (log–log *r* = 0.990) and would therefore act as a leak.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.13 was used; 3.10 or newer should work. No GPU is required.

## Running

```bash
python run_all.py            # everything, roughly one hour
python run_all.py --list     # stages and approximate runtimes
python run_all.py 4 10 11    # selected stages (dependencies must already have run)
```

Stages write into `data/`, `results/` and `figures/` inside this repository.

| Stage | Script | Produces |
|---|---|---|
| 1 | `s01_data.py` | `airquality_clean.parquet`, `vergara_drift.parquet`, descriptive tables |
| 2 | `s02_protocols_fixed.py` | `model_results_raw.csv`, `deployment_predictions.parquet` |
| 3 | `s03_protocols_tuned.py` | `model_results_tuned.csv` (with the selected grid point per fold) |
| 4 | `s04_rolling_origin.py` | `rolling_origin.csv` |
| 5 | `s05_recalibration.py` | `recalibration_results.csv`, `table_recalibration.csv`, `table_drift_by_month.csv` |
| 6 | `s06_drift_corrections.py` | `drift_corrections.csv` |
| 7 | `s07_importance.py` | `permutation_importance.csv` |
| 8 | `s08_uncertainty.py` | `predictions_all_protocols.parquet`, `table_reu.csv` |
| 9 | `s09_vergara.py` | `vergara_protocols.csv` |
| 10 | `s10_statistics.py` | `table_horizon_effect.csv`, `table_protocol_inflation.csv`, `table_tuned_summary.csv`, `table_bootstrap_ci.csv`, `statistics_report.txt` |
| 11 | `s11_figures.py` | `fig1` … `fig6` at 300 dpi |
| 12 | `s12_uncertainty_diagnostics.py` | `table_reu_sensitivity.csv`, `table_residual_scaling.csv`, `figA1` |
| 13 | `s13_inner_validation.py` | `table_inner_validation.csv` |

Dependencies between stages: 5 and 10 read the output of 2; 10 also reads 3 and 4;
11 reads 2–9. Stages 1–9 are otherwise independent of one another.

`results/statistics_report.txt` collects every inferential number quoted in the
manuscript, which is the quickest way to check a reproduction run.

## Reproducibility

All random seeds are fixed at 0, and `src/common.py` pins every thread count to 1 and
forces the sequential joblib backend before numpy or scikit-learn is imported. The run
is therefore deterministic and bitwise reproducible on the pinned versions. The
single-threaded setting is also why the pipeline takes about an hour; it is a
deliberate trade of speed for reproducibility.

One deliberate asymmetry is worth noting: the random forest uses 500 trees in stage 2
(the fixed-configuration experiment) and 300 trees in stages 3–9, which refit the model
many times. Both settings are stated in the manuscript. Section 4.1 of the manuscript
reports results from both experiments, so the 500- and 300-tree settings both appear
there.

## Layout

```
code/
  run_all.py            pipeline driver
  requirements.txt      pinned versions
  LICENSE
  src/
    common.py           paths, variables, model factory, grids, metrics
    s01_data.py … s13_inner_validation.py
```

## Citation

If you use this code, please cite both the software archive and the paper:

> Bozkurt, M. R., Bozkurt, F., and Bozkurt, A. S.: Analysis code for "The validation
> protocol determines both the reported accuracy and the selected model in
> machine-learning calibration of low-cost gas sensor arrays", v1.0.0, Zenodo,
> https://doi.org/10.5281/zenodo.24021644, 2026.

> Bozkurt, M. R., Bozkurt, F., and Bozkurt, A. S.: The validation protocol determines
> both the reported accuracy and the selected model in machine-learning calibration of
> low-cost gas sensor arrays, *Atmospheric Measurement Techniques*, **[under review]**.

Machine-readable metadata is in `CITATION.cff`.

## Licence

BSD 3-Clause (see `LICENSE`).
