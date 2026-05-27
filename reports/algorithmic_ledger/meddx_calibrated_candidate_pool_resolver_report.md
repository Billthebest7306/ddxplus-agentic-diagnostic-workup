# Notebook 49: MEDDx Calibrated Candidate-Pool Resolver

Notebook `49` is an offline learned resolver lab over the Notebook `48` candidate-level feature table. It makes no API calls.

## Control Question

Can we solve the Notebook `46` resolver-discrimination bottleneck with one system-wide calibrated candidate-pool resolver, rather than case-specific rules?

## Inputs

- `artifacts/universal_meddx/meddx_aligned_dataset_native_driver_v1_eval30/`
- `artifacts/universal_meddx/meddx_candidate_pool_adjudicator_lab_v1/candidate_level_educator_features.csv`

## Resolver

Selected resolver:

```text
calibrated_logistic_pool_resolver_v1
```

The model is an L2-regularized logistic candidate scorer:

```text
candidate_score = logistic(features(candidate, workup))
q(candidate | workup) = candidate_score / sum(candidate_score over pool)
```

The selected policy accepts the learned top candidate when it differs from the current answer and does not fall more than one independent support signal below the current answer. Otherwise it keeps the Notebook `46` current final answer.

The model is system-wide: one resolver is trained across DDXPlus, iCraft-MD, and RareBench, with dataset name as a feature. It is not a case-by-case patch.

## Main Results

| Policy | Top-1 | Top-3 | Top-5 | Wins | Regressions | Notes |
|---|---:|---:|---:|---:|---:|---|
| Notebook 46 current | 73/90 | 77/90 | 77/90 | 0 | 0 | live baseline |
| Notebook 48 conservative educator | 75/90 | 77/90 | 77/90 | 2 | 0 | hand-rule label-free repair |
| Notebook 48 case-blocked HGB | 77/90 | 80/90 | 83/90 | 4 | 0 | diagnostic |
| Notebook 49 calibrated logistic resolver | 78/90 | 80/90 | 81/90 | 5 | 0 | selected offline calibration candidate |
| Notebook 49 strict nested threshold diagnostic | 77/90 | 80/90 | 81/90 | 4 | 0 | stricter threshold stress test |
| Notebook 48 label-fit HGB | 86/90 | 88/90 | 88/90 | 13 | 0 | non-deployable label-fit diagnostic |
| Candidate-pool oracle | 88/90 | 88/90 | 88/90 | 15 | 0 | non-deployable oracle |

The selected resolver fixes five DDXPlus workups:

| Case | Budget | Truth | Original | Resolver |
|---|---:|---|---|---|
| `test:100196` | 5 | Pancreatic neoplasm | Acute laryngitis | Pancreatic neoplasm |
| `test:122530` | 10 | URTI | Bronchitis | URTI |
| `test:122530` | 15 | URTI | Bronchitis | URTI |
| `test:126130` | 5 | Bronchitis | Pneumonia | Bronchitis |
| `test:127667` | 10 | Acute otitis media | Tuberculosis | Acute otitis media |

It changes two additional already-wrong DDXPlus workups without causing a regression.

## Interpretation

This is the first system-wide learned resolver result that materially improves the multi-dataset Notebook `46` run while keeping zero regressions under case-blocked out-of-fold evaluation.

The result is not the same as the label-fit `86/90` diagnostic. The selected `78/90` result is case-blocked by patient/case, so all budgets for a held-out case are predicted by a model trained without that case. That is the right direction for a defensible resolver.

The strict nested threshold diagnostic drops to `77/90`, which is a useful warning: the threshold layer is still data-limited with only `30` unique case groups. The frozen resolver should be treated as an offline calibration candidate that needs fresh held-out live confirmation.

Follow-up Notebook `50` tested stronger candidate-signal augmentation. It did not beat this Notebook `49` top-1 result under case-blocked evaluation (`77/90` versus `78/90`), although it improved top-3/top-5 to `83/90`. Notebook `49` therefore remains the strongest current MEDDx top-1 resolver candidate.

## Remaining Failure Modes

After the selected resolver:

- DDXPlus has four remaining misses.
- iCraft-MD has two remaining misses.
- RareBench has six remaining misses.

The RareBench failures do not improve because the current features still lock onto the wrong rare-disease neighbor. The DDXPlus `test:51945` series remains problematic because the truth is weak or absent in the strongest candidate signals.

## Artifacts

- notebook: `notebooks/49_meddx_calibrated_candidate_pool_resolver.ipynb`
- script mirror: `scripts/meddx_calibrated_candidate_pool_resolver_nb49.py`
- artifact root: `artifacts/universal_meddx/meddx_calibrated_candidate_pool_resolver_v1/`

Required outputs:

- `resolved_run_config.json`
- `candidate_resolver_feature_contract.json`
- `case_blocked_candidate_scores.csv`
- `case_blocked_workup_decision_features.csv`
- `resolver_threshold_sweep.csv`
- `case_level_calibrated_resolver_results.csv`
- `calibrated_resolver_policy_summary.csv`
- `strict_nested_threshold_results.csv`
- `strict_nested_threshold_summary.csv`
- `calibrated_logistic_pool_resolver_v1.joblib`
- `logistic_feature_coefficients.csv`
- `final_fit_candidate_scores_diagnostic.csv`
- `final_fit_policy_summary_diagnostic.csv`
- `selected_resolver_failure_audit.csv`
- `selected_calibrated_resolver.json`
- figures under `figures/`

## Verification

- `python3 -m py_compile scripts/meddx_calibrated_candidate_pool_resolver_nb49.py` passed
- Notebook `49` code cells parsed with `ast.parse`
- script executed top-to-bottom with no API calls
- artifact contract passed

`jupyter-nbconvert` is not installed in the current environment, so execution verification used the script mirror that generated the notebook.
