# Notebook 52: MEDDx-Scale Offline Resolver Calibration

Notebook `52` analyzes the completed Notebook `51` MEDDx-scale live artifact without making any API calls.

## Purpose

The control question was whether the new `900`-workup live corpus could calibrate a deployable candidate-pool resolver that repairs the disappointing Notebook `51` final predictions.

Inputs:

- scale artifact: `artifacts/universal_meddx/meddx_scale_hypothesis_branching_confirmation_v1_meddx100/`
- transfer artifact: `artifacts/universal_meddx/meddx_aligned_dataset_native_driver_v1_eval30/`
- `900` scale workups: `100` cases per dataset at budgets `5`, `10`, and `15`
- `90` old transfer workups from Notebook `46`

## Implementation

Notebook `52` builds candidate-level features from the saved resolver exports:

- candidate rank, resolver score, support count, and source indicators
- base LLM rank/confidence and ranked differential membership
- dataset-native signals from DDXPlus MLP, RareBench phenotype graph, and casebase prior
- branch-source indicators and branch counts
- workup-level budget, question count, pool size, and resolver margin

It trains an L2-regularized logistic candidate scorer on a case-group split of the `900` scale workups:

- all budgets for the same dataset/case remain in the same split
- train/validate/test split is `60/20/20`
- threshold is selected on validation only
- old Notebook `46` `90`-workup artifacts are used only as transfer/regression evidence

Implementation note: the first clean-room execution exposed a leakage bug in the feature collector because `is_truth_candidate` matched the generic `is_` prefix. The script now excludes label-derived columns and raises if any feature contains truth/correct/top-k label fragments. The reported artifacts below are from the corrected rerun.

## Scale Run Results

Notebook `51` live result:

| Slice | Top-1 | Top-3 | Top-5 | Candidate-pool recall | Mean questions |
|---|---:|---:|---:|---:|---:|
| Overall | 715/900 | 773/900 | 791/900 | 809/900 | 5.47 |
| DDXPlus B5 | 72/100 | 80/100 | 85/100 | 87/100 | 4.42 |
| DDXPlus B10 | 78/100 | 84/100 | 87/100 | 90/100 | 7.63 |
| DDXPlus B15 | 79/100 | 85/100 | 89/100 | 90/100 | 9.69 |
| iCraft-MD B5 | 93/100 | 100/100 | 100/100 | 100/100 | 3.10 |
| iCraft-MD B10 | 94/100 | 100/100 | 100/100 | 100/100 | 4.18 |
| iCraft-MD B15 | 93/100 | 99/100 | 100/100 | 100/100 | 5.24 |
| RareBench B5 | 70/100 | 75/100 | 78/100 | 82/100 | 3.52 |
| RareBench B10 | 69/100 | 75/100 | 75/100 | 79/100 | 5.18 |
| RareBench B15 | 67/100 | 75/100 | 77/100 | 81/100 | 6.23 |

The main ceiling is now visible: even an oracle resolver over the saved candidate pools can only reach `809/900` (`0.899`) because `91/900` workups do not contain the true diagnosis in the pool.

## Resolver Calibration

Validation-selected logistic resolver:

| Evaluation | Current | Selected resolver | Wins | Regressions | Changed |
|---|---:|---:|---:|---:|---:|
| Validation split | 147/180 | 148/180 | 1 | 0 | 12 |
| Internal test split | 144/180 | 143/180 | 1 | 2 | 12 |
| Old Notebook 46 transfer | 73/90 | 72/90 | 2 | 3 | 7 |

Case-blocked out-of-fold diagnostics on the full `900` workups:

| Resolver | Top-1 | Wins | Regressions | Changed |
|---|---:|---:|---:|---:|
| Logistic balanced | 729/900 | 16 | 2 | 53 |
| Histogram GBM diagnostic | 721/900 | 12 | 6 | 45 |
| ExtraTrees diagnostic | 721/900 | 7 | 1 | 28 |

Full-scale label-fit diagnostics are not promoted. A logistic resolver fit and evaluated on the same `900` workups reaches `730/900`, but still falls far below the `809/900` candidate-pool oracle and has `2` regressions.

## Failure Decomposition

Notebook `51` final misses split into two types:

- candidate-generation misses: the true disease is absent from the pool
- resolver misses: the true disease is present, but the selected top-1 is wrong

Important patterns:

- DDXPlus candidate-pool recall plateaus near `90/100` by budgets `10` and `15`.
- RareBench candidate-pool recall stays lower: `82/100`, `79/100`, and `81/100`.
- iCraft-MD candidate-pool recall is `100/100` at all budgets, so its remaining errors are resolver discrimination only.
- Increasing budget does not monotonically help RareBench top-1; it drops from `70/100` at budget `5` to `67/100` at budget `15`.

Branching also remained sparse:

| Dataset | Budget | Branch rate | Mean branch questions |
|---|---:|---:|---:|
| DDXPlus | 5 | 0.17 | 0.34 |
| DDXPlus | 10 | 0.32 | 1.21 |
| DDXPlus | 15 | 0.35 | 1.39 |
| iCraft-MD | 5/10/15 | 0.00 | 0.00 |
| RareBench | 5 | 0.05 | 0.10 |
| RareBench | 10 | 0.09 | 0.35 |
| RareBench | 15 | 0.13 | 0.52 |

This means the hypothesis-branching machinery did not materially expand candidate coverage on the hardest MEDDx-scale slices, especially RareBench.

## Decision

The selected offline resolver is **not promoted**.

Why:

- validation gain is only `+1/180`
- internal test regresses from `144/180` to `143/180`
- transfer regresses from `73/90` to `72/90`
- the scale corpus has a real candidate-generation ceiling of `809/900`

Notebook `52` should be treated as a calibration and failure-mapping lab, not as a new final model.

## Artifacts

- notebook: `notebooks/52_meddx_scale_offline_resolver_calibration.ipynb`
- script mirror: `scripts/meddx_scale_offline_resolver_calibration_nb52.py`
- artifact root: `artifacts/universal_meddx/meddx_scale_offline_resolver_calibration_v1/`

Key outputs:

- `resolved_run_config.json`
- `baseline_and_oracle_summary.csv`
- `candidate_level_scale_features.csv`
- `candidate_level_transfer_features.csv`
- `validation_threshold_sweep.csv`
- `internal_split_policy_summary.csv`
- `case_blocked_oof_resolver_summary.csv`
- `transfer_policy_summary.csv`
- `failure_decomposition_summary.csv`
- `selected_offline_resolver_policy.json`
- figures under `figures/`

Verification:

- script executed top-to-bottom with no API calls
- artifact contract passed
- label-leakage feature guard passed on rerun
- selected policy JSON records `claim_status = "not_promoted"`

## Interpretation

The disappointing live result is not fixable by a simple post-hoc resolver layer. The candidate pool is strong for iCraft-MD, acceptable but imperfect for DDXPlus, and too weak for RareBench. The next credible improvement should target candidate generation and dataset-native evidence representation, not just a stronger classifier over the existing pool.
