# CSE382 — City Hotel Booking Demand (H2)

End-to-end machine-learning project for the **City Hotel H2 dataset only**: ADR regression, cancellation classification, and booking clustering. The solution is modular Python code runnable from `main.py`; notebooks are not the main submission.

## Submission information — complete before uploading

- **Team:** Team 2
- **Team members (4–5 students):** add names, IDs, and GitHub usernames in the progress report and project records.
- **GitHub repository:** <https://github.com/Kerolos-eng/team-2-hotel-booking-ml>
- **Milestone 1 submitted commit SHA:** `[add output of git rev-parse HEAD after final review]`
- **Milestone 1 report:** [`docs/progress_report_milestone1.md`](docs/progress_report_milestone1.md)

Do not claim a member's contribution unless it is supported by the real work and remote commit/PR history. Each member must use their own GitHub account.

## Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download `H2.csv` from the public mirror listed in the project description and place it at `data/H2.csv`:

<https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv>

The loader asserts that the file is the expected **79,330 × 31** City Hotel H2 data. Do not substitute or merge H1 (Resort Hotel). Raw and generated CSV data are ignored by Git; the report tables and figures are retained.

## Milestone 1 commands

Run from the project root:

```bash
python main.py analyze                 # verify H2; write EDA profile, tables and plots
python main.py clean                   # clean -> data/H2_clean.csv
python main.py split                   # stratified 80/20 -> data/train.csv, data/test.csv
python main.py baseline --task all     # preliminary CV baselines + clustering scan on TRAIN
python main.py all                     # analyze, clean, split, and run all baselines
python -m pytest -q tests              # unit tests
```

The random seed is **42**. The stratified split is **80/20** by `IsCanceled`; supervised baselines use 5-fold CV on TRAIN. TEST has not been used for model selection or final evaluation in Milestone 1. The split command prints the partition cancellation rates only to verify stratification.

## Milestone 1 contents

| Area | Code / output |
|---|---|
| Data loading and H2 shape check | `src/data/loader.py` |
| EDA and per-column data profile | `src/data/eda.py`; `outputs/tables/column_profile.csv`, `eda_summary.csv`; `outputs/eda/*.png` |
| Cleaning and cleaning report | `src/data/cleaning.py`; `outputs/tables/cleaning_report.csv` |
| Prediction points and leakage map | `src/features/feature_sets.py`; `outputs/tables/leakage_map.csv` |
| Train/test split | `src/data/split.py` |
| Reusable impute/scale/one-hot pipelines | `src/preprocessing/pipelines.py` |
| Regression baseline | `src/regression/baseline.py`; `outputs/tables/baseline_regression.csv` |
| Classification baseline and labelled leakage demos | `src/classification/baseline.py`; `outputs/tables/baseline_classification.csv` |
| K-Means baseline and k scan | `src/clustering/baseline.py`; `outputs/tables/clustering_*.csv`, `outputs/figures/` |
| Milestone 1 progress report | `docs/progress_report_milestone1.md` |
| Team commit workflow and per-role message examples | `docs/git_workflow.md` |

`ReservationStatus` is used only for an explicitly labelled leakage demonstration and is never a candidate deployed feature. Cluster outcome rates are post-hoc descriptions and are not used to fit or select clusters.

## Current scope and next milestone

Milestone 1 has been run locally and the included tests pass. The project is **not yet the final project**. Milestone 2 still requires three or more baseline-and-tuned models per supervised task, three optimized clustering algorithms, PCA/LDA comparisons, final test-set evaluation, saved inference artifacts, a usable prediction interface, and final model selection/reporting. See the progress report for the exact preliminary results and open items.
