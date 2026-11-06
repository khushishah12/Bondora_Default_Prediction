# Bondora default prediction: final presentation package

Predict whether a Bondora peer-to-peer loan defaults within 12 months of issue, using only information known at issuance, and compare against Bondora's own risk score.

**Headline (test year 2020, 12-month window, 18,651 loans):** CatBoost reaches ROC-AUC **0.762** against **0.663** for Bondora's own score, PR-AUC 0.455 against 0.341 (lift 2.2), and accuracy 80.0% against a 79.0% "never predict default" baseline. Read accuracy next to its baseline; ranking quality is the main result. Details, caveats and the 12 pipeline stages are in `REPORT.docx` / `REPORT.md`.

This package is built around the approach that produced the strongest results. A second package, `Bondora_Supporting_Analysis.zip`, holds our independently built pipeline and the evidence behind the data-quality decisions.

## Contents

| Path | What it is |
|---|---|
| `REPORT.docx`, `REPORT.md` | The written report (same content, two formats) |
| `notebook/Bondora_Default_Prediction.ipynb` | The full pipeline as a notebook, section by section (stages 1 to 12). **Delivered unexecuted**, see "Running" below |
| `code/zip_approach_window_compare.py` | Runs the protocol for chosen window widths and a chosen test year; produced the tables in the report |
| `code/zip_approach_test2019.py` | One full run with 2019 as the test year (includes the tuning grid and feature importance) |
| `results/window_compare_test2019.csv`, `window_compare_test2020.csv` | Window sweep results (all metrics, per-country ROC-AUC, Bondora reference columns) |
| `results/test2019_n12_full_run/` | Outputs of the full 2019 run: validation comparison, tuning grid, test results, per-country results, predictions, funding deciles |
| `figures/` | Charts used in the report |
| `original_submission/` | The earlier submission for reference (its report, its notebook with outputs, its 2020 result files) |
| `data/` | Empty. Put the raw file here (see below) |
| `model/` | Empty until the notebook is run (it saves the final model and metadata here) |
| `requirements.txt` | Pinned library versions (Python 3.12) |

## Running

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

1. Put **`LoanData_Bondora.csv`** (or the `.zip`) in the `data/` folder. The raw file is not bundled: it is about 150 MB, and Bondora's usage terms should be checked before redistributing it. Alternatively set the environment variable `BONDORA_DATA` to its path.
2. **Notebook:** open `notebook/Bondora_Default_Prediction.ipynb` and run all cells (about 30 to 40 minutes, mostly the boosted models and the tuning grid). It writes tables to `results/notebook_run/`, charts to `figures/` and the final model to `model/`.
3. **Scripts** (run from the `code/` folder):

```bash
cd code
python zip_approach_window_compare.py --test-year 2020 --windows 6,12,18,24
python zip_approach_window_compare.py --test-year 2019 --windows 6,12,18,24
python zip_approach_test2019.py
```

Seed 42 is fixed everywhere. Small numeric differences are possible with other library versions.

## What has and has not been run

* The result tables, figures and report numbers come from the scripts in `code/`.
* The notebook's first 12 code cells (load, target, split, EDA, preprocessing, class balance, baselines) ran without errors while preparing this package. The remaining 8 cells (models, tuning, comparison, final test, diagnostics, funding simulation, window summary) have **not** been run in full; run the notebook once before presenting. Its logic follows the scripts.
* The 2020 test year is scored once per protocol and window; the sweep repeats the protocol for other windows and for 2019, as stated in the report.

## Honest limitations (short version)

Absolute accuracy at a 12-month window is only about one point above the baseline; ranking is where the model is clearly better than Bondora's score. About a third of "no default" labels are loans that default after month 12. Default rates drift by year and country, and several borrower fields stopped being collected in June 2017. Full list in the report.
