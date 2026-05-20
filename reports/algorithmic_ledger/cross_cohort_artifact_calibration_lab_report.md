# Cross-Cohort Artifact Calibration Lab

Notebook 39 is an offline calibration analysis over saved Notebooks 33, 37, and 38 artifacts. It makes no API calls.

## Main Results

| policy_name                                 | claim_type                                    |   correct |   n_cases |   accuracy |   candidate_pool_recall_correct |   top3_correct |   top5_correct |   mean_total_branch_requests |
|:--------------------------------------------|:----------------------------------------------|----------:|----------:|-----------:|--------------------------------:|---------------:|---------------:|-----------------------------:|
| current_final_pipeline                      | deployable_saved_artifact                     |       320 |       343 |   0.932945 |                             335 |            335 |            335 |                      9.18027 |
| raw_gbm_candidate_argmax                    | diagnostic                                    |       317 |       343 |   0.924198 |                             335 |            335 |            335 |                      9.18027 |
| calibration196_rule_layer_v1                | calibration_selected_needs_fresh_confirmation |       323 |       343 |   0.941691 |                             335 |            335 |            335 |                      9.18027 |
| pooled_label_fit_no_regret_rules_diagnostic | diagnostic_label_fit                          |       330 |       343 |   0.962099 |                             335 |            335 |            335 |                      9.18027 |
| candidate_pool_oracle_non_deployable        | oracle_non_deployable                         |       335 |       343 |   0.976676 |                             335 |            335 |            335 |                      9.18027 |

## Selected Calibration Rule Layer

Rules were selected only from the Notebook 38 196-case calibration cohort with a zero-regression constraint inside that cohort. They were then checked on the older 49/98-case artifacts.

| anchor_prediction      | challenger_prediction   | root_id   | status   |   trigger_count |   wins |   regressions |   net_gain |   incremental_wins |   incremental_regressions |   incremental_triggers |   incremental_net_gain |
|:-----------------------|:------------------------|:----------|:---------|----------------:|-------:|--------------:|-----------:|-------------------:|--------------------------:|-----------------------:|-----------------------:|
| Chronic rhinosinusitis | Acute rhinosinusitis    | E_103     | present  |               3 |      3 |             0 |          3 |                  3 |                         0 |                      3 |                      3 |

The selected layer changes pooled accuracy from 320/343 to 323/343, with 3 wins and 0 regressions on the pooled artifacts.

## Interpretation

The artifacts support a modest calibration improvement, not a solved universal resolver. Candidate-pool recall remains the ceiling driver: the pooled candidate-pool oracle is 335/343. The selected rule layer mainly repairs repeated acute-vs-chronic rhinosinusitis decisions in the 196-case calibration cohort, but this signal should be treated as calibration-only until a fresh confirmation run.

## Remaining Failure Modes

| cohort   | failure_pair_calibration196                     |   cases |   candidate_pool_has_true |
|:---------|:------------------------------------------------|--------:|--------------------------:|
| nb33_49  | Acute rhinosinusitis -> Chronic rhinosinusitis  |       1 |                         1 |
| nb37_98  | Acute laryngitis -> Viral pharyngitis           |       1 |                         0 |
| nb37_98  | Acute rhinosinusitis -> Pneumonia               |       1 |                         0 |
| nb37_98  | Atrial fibrillation -> Myocarditis              |       1 |                         1 |
| nb37_98  | Bronchiolitis -> Bronchitis                     |       1 |                         1 |
| nb37_98  | Croup -> Bronchitis                             |       1 |                         0 |
| nb37_98  | Inguinal hernia -> Viral pharyngitis            |       1 |                         0 |
| nb37_98  | Myasthenia gravis -> Acute dystonic reactions   |       1 |                         1 |
| nb37_98  | Pericarditis -> Bronchitis                      |       1 |                         0 |
| nb37_98  | Pulmonary embolism -> Acute dystonic reactions  |       1 |                         1 |
| nb37_98  | Stable angina -> Possible NSTEMI / STEMI        |       1 |                         0 |
| nb38_196 | Acute laryngitis -> Acute otitis media          |       1 |                         1 |
| nb38_196 | Acute rhinosinusitis -> Chronic rhinosinusitis  |       1 |                         1 |
| nb38_196 | Allergic sinusitis -> Acute otitis media        |       1 |                         0 |
| nb38_196 | Atrial fibrillation -> Spontaneous pneumothorax |       1 |                         1 |
| nb38_196 | Croup -> Larygospasm                            |       1 |                         1 |
| nb38_196 | Ebola -> Acute dystonic reactions               |       1 |                         0 |
| nb38_196 | Ebola -> URTI                                   |       1 |                         1 |
| nb38_196 | Epiglottitis -> Larygospasm                     |       1 |                         1 |
| nb38_196 | Pneumonia -> Bronchiectasis                     |       1 |                         1 |

## Request Cost

| cohort   |   cases |   mean_selected_requests |   mean_total_branch_requests |   median_total_branch_requests |   p90_total_branch_requests |   max_total_branch_requests |
|:---------|--------:|-------------------------:|-----------------------------:|-------------------------------:|----------------------------:|----------------------------:|
| nb33_49  |      49 |                nan       |                    nan       |                            nan |                       nan   |                         nan |
| nb37_98  |      98 |                  8.36735 |                      8.42857 |                              6 |                        22.6 |                          24 |
| nb38_196 |     196 |                  6.77041 |                      9.55612 |                              5 |                        22   |                          85 |

## Artifact Contract

- `resolved_run_config.json`
- `cross_cohort_case_pool.csv`
- `cross_cohort_candidate_scores.csv`
- `cross_cohort_policy_summary.csv`
- `cross_cohort_leave_one_cohort_out.csv`
- `cross_cohort_rule_mining_summary.csv`
- `cross_cohort_pooled_rule_mining_summary.csv`
- `selected_calibration_rules.csv`
- `pooled_label_fit_rules_diagnostic.csv`
- `cross_cohort_failure_modes.csv`
- `cross_cohort_request_cost_summary.csv`
- `selected_rule_train_validate_stats.csv`
- `hard_case_calibration_audits.json`
- `selected_cross_cohort_calibration_policy.json`
- figures under `figures/`
