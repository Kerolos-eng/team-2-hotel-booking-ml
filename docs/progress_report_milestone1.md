# CSE382 Project Progress Report — Milestone 1

**Project:** City Hotel Booking Demand (H2): Regression, Classification, and Clustering  
**Team:** Team 2  
**Dataset:** City Hotel H2, obtained from the source URL specified in the project description  
**Report status:** Draft for team review; member names/accounts, repository URL, and submitted commit SHA must be completed by the team before submission.

## 1. Executive summary

The repository now has a modular Python CLI, a single YAML configuration for paths and random seed, data loading and verification, exploratory data analysis, cleaning, a stratified train/test split, reusable scikit-learn preprocessing pipelines, initial regression/classification/clustering experiments, output tables and figures, and unit tests. The code was run on the specified H2 dataset (79,330 rows and 31 source variables). All five current tests pass.

Milestone 1 is **substantially implemented and verified locally, but not submission-ready yet**. The major remaining Milestone 1 actions are administrative/reproducibility work: create or identify the team GitHub repository, add the actual 4–5 students and their GitHub accounts, record authentic member contributions and commit history, add the repository URL and submitted SHA, and have the team review the unusually high number of exact duplicate rows removed during cleaning. This report does not invent team identities or contributions.

## 2. Reproduction environment and commands

The raw CSV is intentionally excluded from the repository/ZIP by `.gitignore`; download H2.csv from the source URL in the project description and save it as `data/H2.csv`. From the project root:

```bash
python -m pip install -r requirements.txt
python main.py analyze
python main.py clean
python main.py split
python main.py baseline --task all
python -m pytest -q tests
```

`python main.py analyze` verifies the expected raw shape and writes EDA tables and plots. `clean` and `split` create local CSVs under `data/`; those generated files are ignored by Git. `baseline --task all` runs cross-validation on the training partition and computes unsupervised clustering diagnostics on the training partition. The final supervised test-set scores have **not** been used for model selection or reported at this milestone; the split command prints the two partition sizes and cancellation rates only as a stratification sanity check.

The project uses `seed: 42`, 80/20 stratified train/test split by `IsCanceled`, and 5-fold shuffled CV on TRAIN (`random_state=42`). Median imputation, scaling, and one-hot encoding are inside the scikit-learn pipelines so fold-specific learned preprocessing is fitted within each training fold. Rare categorical values below 1% are grouped by the encoder's infrequent-category behavior.

## 3. Data understanding and preparation

The downloaded source was verified as **79,330 rows × 31 columns**, matching H2 (City Hotel) in the specification. The EDA profile (`outputs/tables/column_profile.csv`) records every source variable's dtype, missing count/percentage, distinct-value count, numeric range/summary values or top category, and the count of the literal `NULL` token. The other EDA summary and cleaning tables are `outputs/tables/eda_summary.csv` and `outputs/tables/cleaning_report.csv`.

Key raw-data observations from the executed run:

- Raw cancellation rate: **41.73%**.
- Missing values: `Children` has **4** NaNs and `Country` has **24** NaNs.
- `Agent` contains **8,131** literal `NULL` values and `Company` contains **75,641**; these are treated as “no agent/company” indicators, not ordinary observed IDs.
- `ADR` has **1,208** zero values, no negative values, a maximum of **5,400**, and one value at or above the configured 1,000 threshold.
- There are **167** zero-guest records and **331** zero-night records before cleaning.
- The EDA includes the target balance, missing/NULL patterns, ADR distribution, ADR by month, lead-time distribution and cancellation rate by lead-time bin, cancellation rates by selected categories, numerical correlation heatmap, and bookings over time.

Cleaning removed **25,902 exact duplicate rows** and **386 additional invalid rows** (invalid-rule counts overlap). Four remaining missing `Children` values were filled with 0 and three remaining missing `Country` values were filled with `Unknown`. The cleaned dataset has **53,042 rows** and a cancellation rate of **30.20%**. The magnitude of exact-duplicate removal is material: the team should verify whether identical rows represent duplicate data errors or separate bookings with identical recorded attributes. Milestone 2 should include a documented sensitivity check or a justified alternative if this review changes the cleaning policy.

The 80/20 split produced **42,433 training rows** and **10,609 test rows**; the cancellation rates are **30.21%** and **30.20%**, respectively.

## 4. Initial baseline experiments

All supervised metrics below are means across 5-fold CV on TRAIN, not final test scores.

### Regression — target `ADR`

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| DummyRegressor (mean) | 31.549 | 41.525 | -0.000 |
| Ridge (alpha=1) | 18.353 | 25.524 | 0.622 |

The preliminary Ridge baseline improves substantially over the mean-prediction floor. It is an initial benchmark only; model-family comparison and tuning are Milestone 2 work.

### Classification — target `IsCanceled`

| Model / experiment | Accuracy | Precision | Recall | F1 | PR-AUC (average precision) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (prior) | 0.698 | 0.000 | 0.000 | 0.000 | 0.302 | 0.500 |
| Logistic Regression, booking-time features | 0.726 | 0.592 | 0.301 | 0.399 | 0.549 | 0.747 |
| Logistic Regression + `DepositType` (doubtful-feature ablation) | 0.733 | 0.619 | 0.299 | 0.403 | 0.569 | 0.751 |
| Logistic Regression + `ReservationStatus` (intentional leakage demonstration; never deploy) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The deployable preliminary result is the booking-time Logistic Regression row, not either ablation. The perfect score from `ReservationStatus` is included only to demonstrate outcome leakage. The normalized confusion matrix is an out-of-fold training visualization.

### Clustering — K-Means

K-Means was fitted to TRAIN booking attributes without using `IsCanceled` to select `k`. The scan covered **k = 2…8**. The best observed silhouette was **0.177 at k=4**; the lowest Davies–Bouldin index was **1.534 at k=5**. The configured **k=3** profile is retained as a preliminary baseline rather than claimed as the final optimum.

| k | Silhouette (higher is better) | Davies–Bouldin (lower is better) | Inertia |
|---:|---:|---:|---:|
| 2 | 0.128 | 2.383 | 274,780.249 |
| 3 | 0.166 | 1.759 | 230,133.334 |
| 4 | 0.177 | 1.604 | 201,261.557 |
| 5 | 0.173 | 1.534 | 186,849.871 |
| 6 | 0.153 | 1.757 | 174,557.802 |
| 7 | 0.131 | 1.779 | 166,570.440 |
| 8 | 0.145 | 1.718 | 158,787.910 |

The preliminary k=3 cluster profiles are in `outputs/tables/clustering_profile_k3.csv`. They show different booking lead times, lengths of stay, guest counts, ADR, and post-hoc cancellation rates. Cancellation rate is used only to describe the discovered clusters after fitting, never as a clustering input or selection criterion.

## 5. Preliminary leakage audit and prediction points

The feature decisions are recorded in `src/features/feature_sets.py` and exported to `outputs/tables/leakage_map.csv`.

- **ADR regression:** the stated point is when the booking is quoted/created. Reservation status and date, assigned room, and the ADR target itself are excluded. Only features treated as known then are used in the baseline; uncertain later-life-cycle fields are excluded. Milestone 2 should revisit the prediction point and justify each included feature against the actual booking lifecycle.
- **Cancellation classification:** the stated point is when the booking enters the PMS. `ReservationStatus` and `ReservationStatusDate` are excluded as post-outcome fields; `AssignedRoomType` and `BookingChanges` are also excluded due to their later lifecycle timing. `DepositType` is treated as doubtful and appears only in an explicitly labelled ablation. The all-ones status experiment is a leakage demonstration and must never be treated as a valid model.
- **Clustering:** the main input is a booking profile, without `IsCanceled`, `ReservationStatus`, `ReservationStatusDate`, or assigned room. `IsCanceled` is used only for post-hoc cluster interpretation. The use of ADR in clustering should be explicitly justified as a booking attribute and not a supplied outcome for the clustering question.
- **Preprocessing and evaluation:** learned preprocessing is in the estimator pipeline and therefore refitted inside each CV fold. The final test partition has not been used for hyperparameter/model selection; final test evaluation is deferred to Milestone 2.

## 6. Milestone 1 completion checklist

| Requirement | Status | Evidence / next action |
|---|---|---|
| Modular project structure, configuration, dependencies, CLI, setup instructions | Implemented | `src/`, `config.yaml`, `requirements.txt`, `main.py`, README |
| Load and verify H2 only | Verified locally | `src/data/loader.py`; shape assertion passed on H2 |
| EDA, missingness, duplicates, types/ranges, unusual values, plots | Implemented and run | `src/data/eda.py`; `outputs/tables/column_profile.csv`, EDA summary, and figures |
| Cleaning with documented rules | Implemented and run; duplicate policy needs review | `src/data/cleaning.py`, cleaning report; verify interpretation of 25,902 exact repeats |
| Prediction points, task features, preliminary leakage audit | Implemented | `src/features/feature_sets.py`, leakage map, this report |
| Reproducible split and reusable preprocessing | Implemented and run | fixed seed, stratified split, 5-fold CV, sklearn pipelines |
| Initial regression, classification, and clustering baselines | Implemented and run | baseline CSV tables and logs summarized above |
| Unit tests | Passed | `python -m pytest -q tests`: **5 passed** |
| Progress report | Drafted | This file; team identities/contribution history still need team confirmation |
| GitHub repository, member accounts, meaningful commits, URL and submitted SHA | **Not verified / incomplete** | Current extracted working directory was not a Git repository. Team must publish through actual members' accounts and fill README. |

## 7. Team contributions and GitHub record

The project specification requires 4–5 students to be identified and each student to contribute through their own GitHub account. The supplied archive has role slots A–E but no names, student IDs, GitHub usernames, repository URL, commit history, or reliable attribution of who wrote each existing file. Therefore this report does not assign work to named members.

Before submission, replace the rows below with the actual team records and verify each person's commits in the remote repository. Do not create commits impersonating another student.

| Team member | Name / student ID | GitHub account | Verified contribution / PR / commits |
|---|---|---|---|
| A | [fill in] | [fill in] | [fill in from actual history] |
| B | [fill in] | [fill in] | [fill in from actual history] |
| C | [fill in] | [fill in] | [fill in from actual history] |
| D | [fill in] | [fill in] | [fill in from actual history] |
| E (only if team has 5 members) | [fill in] | [fill in] | [fill in from actual history] |

**GitHub repository:** <https://github.com/Kerolos-eng/team-2-hotel-booking-ml>  
**Milestone 1 submitted commit SHA:** [add `git rev-parse HEAD` after the final reviewed commit]

## 8. Figures, tables, and source map

| Report item | File / generating source |
|---|---|
| Raw dataset and per-column profile | `outputs/tables/column_profile.csv` — `src/data/eda.py` (`column_profile`) |
| Data/cleaning summary | `outputs/tables/eda_summary.csv`, `outputs/tables/cleaning_report.csv` — `src/data/eda.py`, `src/data/cleaning.py` |
| [FIGURE: target balance] | `outputs/eda/01_target_balance.png` — `src/data/eda.py` |
| [FIGURE: missing and `NULL` values] | `outputs/eda/02_missing_values.png` — `src/data/eda.py` |
| [FIGURE: ADR distribution and cleaned range] | `outputs/eda/03_adr_distribution.png` — `src/data/eda.py` |
| [FIGURE: ADR by arrival month] | `outputs/eda/04_adr_by_month.png` — `src/data/eda.py` |
| [FIGURE: lead-time distribution and cancellation rate by lead-time bin] | `outputs/eda/05_leadtime_cancel.png` — `src/data/eda.py` |
| [FIGURE: cancellation rates by deposit, market segment, and channel] | `outputs/eda/06_cancel_by_category.png` — `src/data/eda.py` |
| [FIGURE: numeric correlation heatmap] | `outputs/eda/07_correlation_heatmap.png` — `src/data/eda.py` |
| [FIGURE: booking volume by arrival month] | `outputs/eda/08_bookings_per_month.png` — `src/data/eda.py` |
| Regression baseline metrics | `outputs/tables/baseline_regression.csv` — `src/regression/baseline.py` |
| Classification baseline and leakage ablations | `outputs/tables/baseline_classification.csv` — `src/classification/baseline.py` |
| [FIGURE: normalized confusion matrix] | `outputs/figures/clf_confusion_matrix.png` — `src/classification/baseline.py` |
| Clustering scan and profile | `outputs/tables/clustering_k_scan.csv`, `outputs/tables/clustering_profile_k3.csv` — `src/clustering/baseline.py` |
| [FIGURE: clustering k selection and PCA scatter] | `outputs/figures/clu_k_selection.png`, `outputs/figures/clu_pca_scatter.png` — `src/clustering/baseline.py` |
| [FIGURE: regression predicted vs actual and residuals] | `outputs/figures/reg_pred_vs_actual.png` — `src/regression/baseline.py` |

## 9. Remaining work plan for Milestone 2

1. Complete and verify the team roster, repository link, contributor access, individual commit history, and final SHA in README.
2. Review the large exact-duplicate count and finalize a defensible row-retention/splitting policy.
3. Expand regression and classification to at least three model families each, with baseline and tuned results for every selected model; retain an untouched test set for the final evaluation.
4. Optimize at least three clustering algorithms with unsupervised criteria; document search spaces, metrics, sampling, and the selection rationale.
5. Complete PCA explained-variance analysis, supervised comparisons with/without PCA, and classification comparisons using original/PCA/LDA representations.
6. Produce final model comparison tables, selected models, final test metrics, clustering visualizations and interpretation, and a complete leakage audit.
7. Save fitted inference pipelines locally and add reliable prediction commands or a GUI; verify a new ADR prediction and cancellation prediction without retraining at startup.
8. Finalize the report and reproducible application demonstration; keep serialized fitted model artifacts and the large raw dataset out of the submission ZIP as instructed.
