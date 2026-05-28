# Notebook 55: MEDDx Integrated Candidate-Pool Resolver Policy

Notebook `55` freezes the Stage 1 pool recovery and Stage 2 evidence-card resolver into one integrated offline policy. It does not fit a new model and makes no API calls.

## Frozen Policy

`integrated_recovered_pool_evidence_card_policy_v1`

Components:

- pool policy: `saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1`
- resolver policy: `recovered_pool_logistic_evidence_card_resolver_v1`

## Gate Audit

| Gate | Target | Observed | Passed |
|---|---:|---:|---:|
| Stage 1 scale pool recall | >=850/900 | 866/900 | yes |
| Stage 1 old90 transfer pool recall | >= current | 89/90 vs 81/90 | yes |
| Stage 2 case-blocked test top-1 | >=152/180 | 154/180 | yes |
| Stage 2 old90 transfer top-1 | >= current | 75/90 vs 73/90 | yes |
| Held-out test regressions | 0 | 0 | yes |
| Pool size | mean <=15, p90 <=25 | mean 13.02, p90 22.0 | yes |

## Integrated Summary

| Evaluation | Top-1 | Top-3 | Top-5 | Pool recall | Mean pool size | Mean questions | Claim status |
|---|---:|---:|---:|---:|---:|---:|---|
| Scale all | 742/900 | 811/900 | 829/900 | 866/900 | 13.02 | 5.47 | diagnostic, contains train cases |
| Scale held-out test | 154/180 | 167/180 | 170/180 | 180/180 | 13.27 | 5.46 | primary case-blocked gate |
| Old Notebook 46 transfer | 75/90 | 81/90 | 82/90 | 89/90 | 12.62 | 6.08 | transfer check |

## Interpretation

The project now has a coherent offline MEDDx-scale improvement path:

1. Candidate-pool recovery solves the `809/900` oracle ceiling by raising pool recall to `866/900`.
2. The evidence-card resolver turns that larger pool into a held-out improvement from `150/180` to `154/180`, with no held-out regressions.
3. The same frozen policy transfers positively to the old Notebook `46` artifact, improving final top-1 from `73/90` to `75/90`.

The integrated policy is ready for a small live pilot, but the report should not overclaim the all-900 final row as held-out performance. The strongest defensible final claim is that the offline gates pass and justify prospective confirmation.

## Artifacts

- `artifacts/universal_meddx/meddx_integrated_candidate_pool_resolver_policy_v1/integrated_policy_summary.csv`
- `artifacts/universal_meddx/meddx_integrated_candidate_pool_resolver_policy_v1/integrated_policy_gate_audit.csv`
- `artifacts/universal_meddx/meddx_integrated_candidate_pool_resolver_policy_v1/selected_policy.json`
- figures under `artifacts/universal_meddx/meddx_integrated_candidate_pool_resolver_policy_v1/figures/`
