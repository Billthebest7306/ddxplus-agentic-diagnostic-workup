# Notebook 31 Neural Candidate Pool Resolver

Notebook `31` is an offline final-head experiment over the completed Notebook `30` live candidate pool.

The motivating observation from Notebook `30` is unusually sharp: the selected hand-built resolver reached `44/49`, but the small resolver candidate pool contained the true diagnosis in all `49/49` cases. Notebook `31` therefore asks whether a train/validate-derived neural resolver can choose better from the already generated candidates without new API calls and without using the 49-case labels for training.

## Policy

Selected policy:

```text
compact_neural_candidate_resolver_v1
```

Model:

- `sklearn.neural_network.MLPClassifier`
- hidden layers: `(64, 32)`
- selected features: graph, Bayes, MLP, candidate-role, request-state, and candidate-set context features
- excluded from selected model: disease-name one-hot features and 49-case labels for training or threshold selection

The model scores each Notebook `30` candidate independently and selects the highest-scoring candidate within each case.

## Inputs

Notebook `31` uses existing artifacts only:

- Notebook `30` live candidate pool:
  - `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1/candidate_level_live_scores.csv`
- Notebook `30` train/validate synthetic resolver features:
  - `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1/candidate_resolver_train_validate_features.csv`
- Notebook `30` predictions and metrics:
  - `predictions.csv`
  - `metrics.json`

It makes no API calls.

## Results

Artifact root:

```text
artifacts/trajectory_replicates/neural_candidate_pool_resolver_49case_v1/
```

| System | Correct | Accuracy | Mean selected requests | Mean total branch requests |
|---|---:|---:|---:|---:|
| Notebook `30` base branch | 42/49 | 0.857 | 6.80 | 6.80 |
| Notebook `30` hand resolver | 44/49 | 0.898 | 6.78 | 12.10 |
| Notebook `31` compact neural resolver | 46/49 | 0.939 | 6.78 | 12.10 |
| Candidate-pool oracle, diagnostic only | 49/49 | 1.000 | 6.78 | 12.10 |

Paired against Notebook `30`:

| Metric | Value |
|---|---:|
| Wins | 2 |
| Regressions | 0 |
| Changed predictions | 3 |
| Wins vs base branch | 4 |
| Regressions vs base branch | 0 |

The two Notebook `31` wins over Notebook `30` are:

| Case | True diagnosis | Notebook `30` | Notebook `31` |
|---|---|---|---|
| `test:81691` | Croup | Acute otitis media | Croup |
| `test:35039` | Myocarditis | Pericarditis | Myocarditis |

Remaining misses:

| Case | True diagnosis | Notebook `31` prediction | Main interpretation |
|---|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | high-confidence near-neighbor anchor |
| `test:11655` | Bronchitis | URTI | high-confidence near-neighbor anchor |
| `test:62878` | Pericarditis | Anemia | branch/candidate evidence state remains poorly adjudicated |

## Candidate-Pool Finding

Notebook `31` confirms the Notebook `30` candidate-pool observation:

| Pool statistic | Value |
|---|---:|
| Mean candidate rows per case | 4.08 |
| Mean unique diagnoses per case | 3.98 |
| Minimum unique diagnoses | 3 |
| Maximum unique diagnoses | 8 |
| True diagnosis present in candidate pool | 49/49 |

For comparison, Notebook `30`'s final ranked differential contains:

| Ranked differential | Correct |
|---|---:|
| Top-3 | 46/49 |
| Top-5 | 47/49 |

The important distinction is that the broader resolver candidate pool includes base, pseudo graph/Bayes/MLP candidates, and live branch candidates. It is not the same object as the final displayed ranked differential.

## Interpretation

Notebook `31` is the strongest learned final-head result over the Notebook `30` live candidate pool so far. It improves from `44/49` to `46/49` with zero regressions against the Notebook `30` selected answer.

It still does not achieve the `47/49+` target as an actual selected policy. The `49/49` number is an oracle ceiling that uses the true label to pick the correct candidate and must be reported only as diagnostic candidate-pool recall.

The result reframes the next bottleneck:

- candidate generation is now very strong
- full LLM branch-to-completion is expensive
- final selection among close confounders remains unresolved

The next credible direction is not more broad branching. It is a stronger resolver for close candidate sets, possibly with abstention or one targeted discriminator question when the learned resolver sees a high-confidence near-neighbor anchor.

## Artifacts

Required artifacts written:

- `resolved_run_config.json`
- `candidate_pool_oracle_summary.csv`
- `candidate_pool_oracle_summary.json`
- `neural_resolver_validation_summary.csv`
- `candidate_level_neural_scores.csv`
- `case_level_neural_resolver_results.csv`
- `paired_notebook30_vs_neural_resolver.csv`
- `hard_case_neural_resolver_audits.json`
- `selected_neural_resolver.json`
- `summary_metrics.csv`
- figures under `figures/`

## Status

Notebook `31` is an offline candidate for follow-up confirmation, not a replacement for the defended Notebook `13` method. It is a useful research result because it shows that the multi-agent/hypothesis-forced candidate pool can support substantially better final accuracy if selection improves.
