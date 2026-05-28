# Notebook 53: MEDDx Candidate-Pool Recovery Lab

Notebook `53` is an offline Stage 1 recovery lab over the completed Notebook `51` MEDDx-scale live artifact. It makes no API calls and keeps the live workups frozen.

## Control Question

Can a label-free final candidate-pool expansion layer raise candidate-pool recall from the Notebook `51` ceiling of `809/900` to at least `850/900` without breaking old-artifact transfer?

## Selected Pool Policy

`saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1`

Candidate pool:

- current resolver pool
- saved final ranked differential top-10
- saved raw LLM ranked differential top-10
- DDXPlus MLP top-5 when available
- branch ranked differential top-10
- casebase prior top label
- RareBench graph top label
- DDXPlus visible-evidence Bayes top-10
- RareBench visible-HPO exemplar top-10

## Main Result

| Evaluation | Current pool recall | Selected pool recall | Recoveries | Mean pool size | P90 pool size |
|---|---:|---:|---:|---:|---:|
| Scale all | 809/900 | 866/900 | 57 | 13.02 | 22.0 |
| Scale held-out test | 161/180 | 180/180 | 19 | 13.27 | 23.0 |
| Old Notebook 46 transfer | 81/90 | 89/90 | 8 | 12.62 | 20.2 |

The Stage 1 minimum target (`>=850/900`) and strong target (`>=865/900`) both pass.

## Interpretation

The failure in Notebook `52` was not just a weak resolver. The system was leaving useful candidates outside the formal resolver pool. The largest single saved-source gain came from the raw LLM ranked differential top-10, followed by DDXPlus Bayes and RareBench visible-HPO expansion.

This is promoted to Stage 2 as a candidate-pool recovery layer. It is not a final diagnostic system by itself.

## Artifacts

- `artifacts/universal_meddx/meddx_candidate_pool_recovery_lab_v1/candidate_pool_recovery_summary.csv`
- `artifacts/universal_meddx/meddx_candidate_pool_recovery_lab_v1/expanded_candidate_pool_long.csv`
- `artifacts/universal_meddx/meddx_candidate_pool_recovery_lab_v1/selected_policy.json`
- figures under `artifacts/universal_meddx/meddx_candidate_pool_recovery_lab_v1/figures/`
