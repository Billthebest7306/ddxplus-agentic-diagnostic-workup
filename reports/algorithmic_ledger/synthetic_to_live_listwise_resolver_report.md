# Synthetic-to-Live Listwise Resolver

Notebook 40 tests whether a resolver trained on DDXPlus-derived synthetic partial evidence states transfers to the saved live candidate pools from Notebooks 33, 37, and 38. It makes no API calls.

## Main Pooled Results

| policy_name                                    | claim_type                               |   correct |   n_cases |   accuracy |   candidate_pool_recall_correct |
|:-----------------------------------------------|:-----------------------------------------|----------:|----------:|-----------:|--------------------------------:|
| current_final_pipeline                         | saved_artifact_reference                 |       320 |       343 |   0.932945 |                             335 |
| candidate_pool_oracle_non_deployable           | oracle_non_deployable                    |       335 |       343 |   0.976676 |                             335 |
| synthetic_logistic_group_softmax               | synthetic_only_transfer                  |       315 |       343 |   0.918367 |                             335 |
| synthetic_hist_gradient_boosting_group_softmax | synthetic_only_transfer                  |       315 |       343 |   0.918367 |                             335 |
| synthetic_listwise_mlp                         | synthetic_only_transfer                  |       307 |       343 |   0.895044 |                             335 |
| synthetic_pairwise_bradley_terry               | synthetic_only_transfer                  |       314 |       343 |   0.915452 |                             335 |
| artifact_loco_logistic                         | synthetic_plus_loco_artifact_calibration |       317 |       343 |   0.924198 |                             335 |
| artifact_loco_gbm                              | synthetic_plus_loco_artifact_calibration |       315 |       343 |   0.918367 |                             335 |
| artifact_fit_logistic                          | diagnostic_artifact_label_fit            |       316 |       343 |   0.921283 |                             335 |
| artifact_fit_gbm                               | diagnostic_artifact_label_fit            |       319 |       343 |   0.930029 |                             335 |

## Synthetic Validation

| policy_name                                    |   correct |   n_cases |   accuracy |   candidate_pool_recall_correct |   row_average_precision |   row_auc |
|:-----------------------------------------------|----------:|----------:|-----------:|--------------------------------:|------------------------:|----------:|
| synthetic_logistic_group_softmax               |      1225 |      2000 |     0.6125 |                            1810 |                0.806458 |  0.95394  |
| synthetic_hist_gradient_boosting_group_softmax |      1223 |      2000 |     0.6115 |                            1810 |                0.807862 |  0.95422  |
| synthetic_listwise_mlp                         |      1222 |      2000 |     0.611  |                            1810 |                0.797274 |  0.950708 |
| synthetic_pairwise_bradley_terry               |      1211 |      2000 |     0.6055 |                            1810 |                0.737338 |  0.924671 |

## Leave-One-Cohort-Out Artifact Calibration

| cohort     | policy_name            |   correct |   n_cases |   accuracy |   candidate_pool_recall_correct |
|:-----------|:-----------------------|----------:|----------:|-----------:|--------------------------------:|
| nb33_49    | artifact_loco_logistic |        45 |        49 |   0.918367 |                              49 |
| nb37_98    | artifact_loco_logistic |        88 |        98 |   0.897959 |                              92 |
| nb38_196   | artifact_loco_logistic |       184 |       196 |   0.938776 |                             194 |
| pooled_343 | artifact_loco_logistic |       317 |       343 |   0.924198 |                             335 |
| nb33_49    | artifact_loco_gbm      |        44 |        49 |   0.897959 |                              49 |
| nb37_98    | artifact_loco_gbm      |        87 |        98 |   0.887755 |                              92 |
| nb38_196   | artifact_loco_gbm      |       184 |       196 |   0.938776 |                             194 |
| pooled_343 | artifact_loco_gbm      |       315 |       343 |   0.918367 |                             335 |

## Interpretation

The current saved final pipeline is `320/343`. The selected leave-one-cohort-out resolver `artifact_loco_logistic` is `317/343` with `1` wins and `4` regressions versus the current final pipeline.

This means synthetic DDXPlus states are useful for building and testing resolver families, but the current synthetic-to-live transfer does not by itself produce the desired near-oracle resolver. The candidate-pool oracle remains the upper bound, not a final result.

## Remaining Failure Modes

| cohort   | failure_pair                                   |   cases |   candidate_pool_has_true |
|:---------|:-----------------------------------------------|--------:|--------------------------:|
| nb38_196 | Acute rhinosinusitis -> Chronic rhinosinusitis |       4 |                         4 |
| nb33_49  | Acute rhinosinusitis -> Chronic rhinosinusitis |       1 |                         1 |
| nb33_49  | Bronchitis -> URTI                             |       1 |                         1 |
| nb33_49  | Croup -> Acute otitis media                    |       1 |                         1 |
| nb33_49  | Pericarditis -> Panic attack                   |       1 |                         1 |
| nb37_98  | Acute laryngitis -> Acute otitis media         |       1 |                         0 |
| nb37_98  | Acute rhinosinusitis -> Pneumonia              |       1 |                         0 |
| nb37_98  | Atrial fibrillation -> PSVT                    |       1 |                         1 |
| nb37_98  | Bronchiolitis -> Bronchitis                    |       1 |                         1 |
| nb37_98  | Croup -> Bronchitis                            |       1 |                         0 |
| nb37_98  | Inguinal hernia -> Viral pharyngitis           |       1 |                         0 |
| nb37_98  | Myasthenia gravis -> Acute dystonic reactions  |       1 |                         1 |
| nb37_98  | Pericarditis -> Guillain-Barré syndrome        |       1 |                         0 |
| nb37_98  | Pulmonary embolism -> Acute dystonic reactions |       1 |                         1 |
| nb37_98  | Stable angina -> Possible NSTEMI / STEMI       |       1 |                         0 |
| nb38_196 | Acute laryngitis -> Acute otitis media         |       1 |                         1 |
| nb38_196 | Allergic sinusitis -> Acute otitis media       |       1 |                         0 |
| nb38_196 | Bronchitis -> URTI                             |       1 |                         1 |
| nb38_196 | Croup -> Larygospasm                           |       1 |                         1 |
| nb38_196 | Ebola -> Acute dystonic reactions              |       1 |                         0 |
| nb38_196 | Ebola -> URTI                                  |       1 |                         1 |
| nb38_196 | Epiglottitis -> Larygospasm                    |       1 |                         1 |
| nb38_196 | Pneumonia -> Bronchiectasis                    |       1 |                         1 |

## Artifact Contract

- `resolved_run_config.json`
- `synthetic_state_generation_summary.csv`
- `synthetic_model_validation_summary.csv`
- `live_candidate_pool_summary.csv`
- `synthetic_only_live_transfer_summary.csv`
- `synthetic_only_live_transfer_case_results.csv`
- `leave_one_cohort_out_summary.csv`
- `leave_one_cohort_out_case_predictions.csv`
- `diagnostic_artifact_fit_summary.csv`
- `diagnostic_artifact_fit_case_predictions.csv`
- `live_transfer_policy_summary.csv`
- `live_transfer_candidate_scores.csv`
- `selected_loco_resolver_case_results.csv`
- `selected_loco_failure_modes.csv`
- `hard_case_listwise_resolver_audits.json`
- `selected_listwise_resolver_policy.json`
- figures under `figures/`
