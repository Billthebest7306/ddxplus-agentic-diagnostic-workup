# Notebook 54: MEDDx Evidence-Card Resolver Lab

Notebook `54` is the Stage 2 resolver lab over the recovered Notebook `53` candidate pool. It trains only on case-blocked train cases, selects an override threshold on validation, and evaluates the frozen resolver on the held-out test split plus the old Notebook `46` transfer artifact.

## Selected Resolver

`recovered_pool_logistic_evidence_card_resolver_v1`

- model: L2 logistic candidate scorer, `C=2.0`, class-balanced
- features: source/rank/score evidence-card features, dataset, budget, current resolver score, DDXPlus MLP fields, RareBench graph fields, casebase prior fields
- threshold: `0.8`, selected on validation only
- action: preserve current prediction unless learned top candidate clears the threshold

## Main Result

| Evaluation | Current | Selected resolver | Top-3 | Top-5 | Wins | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 145/180 | 148/180 | 157/180 | 159/180 | 3 | 0 |
| Held-out test | 150/180 | 154/180 | 167/180 | 170/180 | 4 | 0 |
| Scale all diagnostic | 715/900 | 742/900 | 811/900 | 829/900 | 27 | 0 |
| Old Notebook 46 transfer | 73/90 | 75/90 | 81/90 | 82/90 | 2 | 0 |

The held-out test result `154/180` is equivalent to `770/900`, clearing the minimum final target of `>=760/900` under case-blocked evaluation. The all-900 row is diagnostic because it includes train cases and should not be treated as the generalization claim.

## Interpretation

The recovered pool creates real resolver headroom. The conservative threshold gives smaller gains than raw learned reranking, but it preserves current correct answers in the selected validation/test/transfer evaluations. This makes it suitable for Stage 3 integration.

Out-of-fold diagnostic rows remain more modest, so the claim should stay precise: the selected policy passes the case-blocked held-out test gate and old-artifact transfer gate, not a universal proof of `>85%` final accuracy.

## Artifacts

- `artifacts/universal_meddx/meddx_evidence_card_resolver_lab_v1/resolver_policy_summary.csv`
- `artifacts/universal_meddx/meddx_evidence_card_resolver_lab_v1/candidate_level_evidence_card_features.csv`
- `artifacts/universal_meddx/meddx_evidence_card_resolver_lab_v1/selected_policy.json`
- `artifacts/universal_meddx/meddx_evidence_card_resolver_lab_v1/oof_diagnostic_resolver_summary.csv`
- figures under `artifacts/universal_meddx/meddx_evidence_card_resolver_lab_v1/figures/`
