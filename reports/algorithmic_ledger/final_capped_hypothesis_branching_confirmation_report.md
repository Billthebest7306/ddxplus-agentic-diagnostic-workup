# Notebook 41 Final Capped Hypothesis-Branching Confirmation

Last updated: 2026-05-20

Notebook `41` is the final lean live confirmation runner for the hypothesis-branching candidate-pool architecture.

## Purpose

The project is no longer trying to buy a larger calibration set. Notebook `41` freezes a practical final policy and prepares a fresh 100-case live confirmation:

- keep the Notebook `13`-style base LLM evidence-acquisition backbone
- launch hypothesis-forced branches only when the learned branch gate fires
- cap spawned branches so request cost cannot explode
- resolve the final candidate pool with graph/Bayes/MLP candidate features
- report top-1, top-3, top-5, candidate-pool recall, selected-request cost, and total branch-request cost
- remove the Notebook `33`/`38` close-confounder extra-root rescue layer

## Files

- notebook: `notebooks/41_final_capped_hypothesis_branching_confirmation.ipynb`
- script mirror: `scripts/final_capped_hypothesis_branching_confirmation_nb41.py`
- live artifact root: `artifacts/trajectory_replicates/final_capped_hypothesis_branching_confirmation100_v1/`
- dry-run smoke artifact root: `artifacts/trajectory_replicates/final_capped_hypothesis_branching_confirmation100_dryrun_smoke_v1/`

## Frozen Policy

| Setting | Value |
|---|---:|
| LLM model | `gpt-4.1-mini` |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Balanced cases per pathology | `2` |
| Final target cases | `100` |
| Branch trigger threshold | `0.20` |
| Adaptive max branches | `2` |
| Continuation-value threshold | `0.20` |
| Base request cap | `24` |
| Per-branch request cap | `8` |
| Hard total request cap per case | `24` |
| Close-confounder extra-root layer | excluded |

The live cohort excludes the original Notebook `13` 49-case reference set and, when possible, excludes prior live benchmark cohorts. The sampler takes two held-out test cases per pathology and then adds extra held-out cases to reach 100 rows.

## Artifact Contract

Notebook `41` writes:

- `benchmark_cases.csv`
- `balanced_pathology_case_counts.csv`
- `reference_case_exclusion_summary.csv`
- `reference_summary.csv`
- `resolved_run_config.json`
- `branch_mlp_train_validate_features.csv`
- `candidate_resolver_train_validate_features.csv`
- `branch_mlp_validation_summary.csv`
- `branch_mlp_threshold_sweep.csv`
- `candidate_resolver_validation_summary.csv`
- `pairwise_evidence_separation_graph.csv`
- `predictions.csv`
- `paired_notebook13_vs_notebook41.csv`
- `branch_case_results.csv`
- `candidate_level_live_scores.csv`
- `candidate_level_live_resolver_scores.csv`
- `candidate_pool_topk_rankings.csv`
- `adaptive_live_final_predictions.csv`
- `final_resolver_trace.csv`
- `adaptive_branch_decision_trace.csv`
- `topk_summary.csv`
- `metrics.json`
- `metrics_final.json`
- `final_confirmation_paired_outcomes.csv`
- `final_confirmation_truth_rank_analysis.csv`
- `final_confirmation_failure_modes.csv`
- `final_confirmation_branch_trigger_threshold_diagnostics.csv`
- `final_confirmation_resolver_margin_diagnostics.csv`
- `final_confirmation_candidate_source_recall.csv`
- `final_confirmation_summary.json`
- `summary_metrics.csv`
- `selected_hypothesis_branch_policy.json`
- `selected_final_capped_policy.json`
- `hypothesis_branch_assignments.csv`
- `traces.jsonl`
- `branch_traces.jsonl`
- `raw_api_responses.jsonl`
- `hard_case_branch_audits.json`

Figures are written under `figures/`.

## Verification

Dry-run smoke execution completed with no API key:

- dry-run benchmark size: `2`
- base correct: `1/2`
- final capped branch-selected correct: `1/2`
- wins/regressions versus base: `0/0`
- mean total branch requests: `16.5`
- final resolver candidate-pool recall: `2/2`
- artifact contract: passed

Static checks:

- `python3 -m py_compile scripts/final_capped_hypothesis_branching_confirmation_nb41.py`
- all Notebook `41` code cells parsed with `ast.parse`

## Interpretation

Notebook `41` is a confirmation runner, not a tuning notebook. It intentionally avoids the fragile calibration-only rules from Notebook `39` and the close-confounder extra-root layer from Notebook `33`/`38`. Its role is to give the project one clean final live run with controlled branch cost and restored differential-diagnosis reporting.

The result should be reported after the live 100-case run completes. Until then, Notebook `41` is prepared and smoke-tested but not yet a performance claim.
