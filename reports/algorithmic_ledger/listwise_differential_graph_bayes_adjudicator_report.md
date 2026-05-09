# Notebook 29 Listwise Differential Graph-Bayes-MLP Adjudicator

- notebook: `notebooks/29_listwise_differential_graph_bayes_adjudicator.ipynb`
- artifact root: `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`
- status: executed offline, no API calls

## Question

Notebook `28` showed that the live branch system was no longer failing only because branches were missing. Several remaining true diagnoses were already present inside the base or branch ranked differentials, but the resolver scored only branch top-1 predictions and graph/Bayes/MLP pseudo top-1 candidates.

Notebook `29` asks whether a train/validate-calibrated listwise adjudicator can promote those ranked differential entries without using the 49-case labels for training or threshold selection.

## Method

Notebook `29` keeps Notebook `28`'s live evidence acquisition, branches, and revealed evidence fixed. It then explodes every branch state into a candidate set:

- source top-1 prediction
- ranked differential entries
- graph top-5
- Bayes top-5
- MLP top-5

For each candidate diagnosis it recomputes:

- graph score, posterior, rank, positive support, and contradiction
- Bayesian log score, posterior, and rank
- partial-evidence MLP posterior and rank
- branch/request features and ranked-differential rank

The selected resolver is an L2 logistic candidate scorer trained on Notebook `28` train/validate synthetic candidate features. The fixed selected policy is:

```text
policy = listwise_differential_graph_bayes_mlp_v1
score = trained logistic candidate probability
eligible candidates = source top-1 plus ranked differential top-3
override margin over Notebook 28 selected answer = 0.02
```

The margin is recorded as a fixed validation-derived setting. Diagnostic variants are written to artifacts but not selected by looking at the 49-case labels.

## Validation

The candidate scorer recovers the Notebook `28` resolver calibration:

| Metric | Value |
|---|---:|
| Train candidate rows | 33,232 |
| Validate candidate rows | 16,670 |
| Validate positive rate | 0.109 |
| Candidate AUC | 0.955 |
| Candidate average precision | 0.809 |

The validation margin sweep is saved in:

- `listwise_validation_margin_sweep.csv`

## 49-Case Result

| System | Correct | Accuracy | Mean selected requests | Mean total branch requests |
|---|---:|---:|---:|---:|
| Notebook `28` selected | 44/49 | 0.898 | 6.63 | 9.96 |
| Notebook `29` listwise selected | 45/49 | 0.918 | 6.63 | 9.96 |

Paired result versus Notebook `28`:

| Metric | Value |
|---|---:|
| Wins | 1 |
| Regressions | 0 |
| Changed predictions | 1 |

The single win is `test:81691` Croup. Notebook `28` selected Acute otitis media; Notebook `29` promotes Croup from the `graph_bayes_scout` branch differential at rank 3. That candidate is graph rank 1, Bayes rank 1, and MLP rank 3 in the branch state.

Promotion decision:

```text
not_promoted_diagnostic_offline_listwise_candidate
```

Notebook `29` improves Notebook `28`, but it does not reach the `47/49` promotion target.

## Oracle Ceiling

The candidate-pool ceiling is now very high:

| Candidate pool | Oracle correct |
|---|---:|
| Source top-1 plus ranked top-1 | 44/49 |
| Source top-1 plus ranked top-2 | 47/49 |
| Source top-1 plus ranked top-3 | 48/49 |
| Source top-1 plus ranked top-5 | 48/49 |
| All exploded graph/Bayes/MLP/ranked candidates | 49/49 |

This is the key finding: after Notebook `29`, the candidate pool is no longer the obvious bottleneck. The hard problem is calibrated selection among close confounders.

## Remaining Misses

| Case | True pathology | Notebook `29` prediction | Observation |
|---|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | True label is rank 2, but graph/Bayes/MLP all strongly favor Chronic rhinosinusitis. |
| `test:11655` | Bronchitis | URTI | True label is rank 2, but the evidence ledger supports URTI as graph/Bayes/MLP rank 1. |
| `test:8666` | Influenza | HIV (initial infection) | True label is rank 2, but the final revealed evidence supports the wrong anchor. |
| `test:76022` | Panic attack | Myocarditis | The full exploded pool contains Panic attack diagnostically, but selected support still favors Myocarditis. |

## Interpretation

Notebook `29` is a useful improvement over Notebook `28`, but it is not the killer result by itself. It confirms that the ranked-differential route is real: top-2/top-3 differential candidates contain enough information for `47/49` to `48/49`. But a straightforward train/validate logistic scorer is still too loyal to graph/Bayes/MLP rank-1 support in silent near-neighbor confusions.

The next credible step is not more broad branching. It is a pairwise or abstaining adjudicator trained specifically for top-1 versus rank-2 confounder swaps, with a calibration objective that penalizes confident wrong-anchor consensus. That model should learn when a rank-2 LLM differential is a legitimate unresolved confounder rather than a weak alternative.

## Artifact Contract

Required artifacts were written:

- `resolved_run_config.json`
- `listwise_candidate_train_validate_features.csv`
- `listwise_validation_summary.csv`
- `listwise_validation_margin_sweep.csv`
- `listwise_live_candidate_scores.csv`
- `listwise_policy_summary.csv`
- `case_level_listwise_results.csv`
- `paired_notebook28_vs_listwise.csv`
- `candidate_pool_oracle_summary.csv`
- `hard_case_listwise_audits.json`
- `selected_listwise_policy.json`
- `hard_case_rank_movement.csv`
- figures under `figures/`
