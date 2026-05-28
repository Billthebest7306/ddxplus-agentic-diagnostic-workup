# Notebook 56: Prospective Integrated MEDDx Live Confirmation

Notebook `56` is the prospective confirmation runner for the frozen Notebook `55` policy.

- notebook: `notebooks/56_meddx_prospective_integrated_live_confirmation.ipynb`
- script mirror: `scripts/meddx_prospective_integrated_live_confirmation_nb56.py`
- live artifact root: `artifacts/universal_meddx/meddx_prospective_integrated_live_confirmation_v1_prospective90/`
- live pool artifact root: `artifacts/universal_meddx/meddx_prospective_integrated_live_confirmation_v1_prospective90_candidate_pool/`
- live final artifact root: `artifacts/universal_meddx/meddx_prospective_integrated_live_confirmation_v1_prospective90_frozen_policy/`

## Purpose

The notebook runs a fresh `90`-workup confirmation cohort:

| Dataset | Unique cases | Budgets | Workups |
|---|---:|---|---:|
| DDXPlus | 10 | 5, 10, 15 | 30 |
| iCraft-MD | 10 | 5, 10, 15 | 30 |
| RareBench | 10 | 5, 10, 15 | 30 |
| Total | 30 | 5, 10, 15 | 90 |

It uses the Notebook `51` MEDDx-aligned live driver, but changes the sampling seed to `560` and excludes prior artifact case IDs where possible. It then applies the frozen Notebook `55` pool and resolver stack without using prospective labels for threshold selection.

## Frozen Policy

The confirmation policy is:

`integrated_recovered_pool_evidence_card_policy_v1`

The post-processing layer uses:

- pool: `saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1`
- resolver: `recovered_pool_logistic_evidence_card_resolver_v1`
- model: L2 logistic candidate scorer, `C=2.0`, class-balanced
- threshold: `0.8`, selected previously on Notebook `54` validation only

The notebook reports `prospective_current_live`, frozen integrated policy, and the non-deployable candidate-pool oracle side by side.

## Dry-Run Verification

No-API dry-run smoke completed over `9` workups and wrote the expected artifact contract.

Dry-run artifact roots:

- `artifacts/universal_meddx/meddx_prospective_integrated_live_confirmation_dryrun_smoke_v1_prospective90/`
- `artifacts/universal_meddx/meddx_prospective_integrated_live_confirmation_dryrun_smoke_v1_prospective90_candidate_pool/`
- `artifacts/universal_meddx/meddx_prospective_integrated_live_confirmation_dryrun_smoke_v1_prospective90_frozen_policy/`

The smoke run is a plumbing check only. Its tiny three-case sample should not be interpreted as performance evidence.

## Artifact Contract

The final policy root writes:

- `prospective_candidate_pool_results.csv`
- `prospective_expanded_candidate_pool_long.csv`
- `prospective_candidate_level_evidence_card_features.csv`
- `prospective_candidate_level_resolver_scores.csv`
- `prospective_current_case_results.csv`
- `prospective_integrated_policy_case_results.csv`
- `prospective_integrated_policy_summary.csv`
- `paired_live_current_vs_integrated_policy.csv`
- `hard_case_audits.json`
- `selected_live_confirmation_policy.json`
- figures under `figures/`

## Interpretation Plan

After the live run, analyze:

- top-1/top-3/top-5 for current live versus frozen integrated policy
- candidate-pool recall
- paired wins and regressions
- per-dataset/per-budget behavior
- request cost and branch cost
- whether the frozen resolver generalizes beyond the Notebook `51`/old90 artifacts

This notebook should be treated as a confirmation run. Do not recalibrate thresholds on the new labels before reporting the prospective result.
