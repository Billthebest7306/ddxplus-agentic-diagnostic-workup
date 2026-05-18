# Notebook 30 Hypothesis-Forced Differential Branching

Notebook `30` implements the next prospective live branching candidate after Notebooks `27` and `28`.

The main change is that branches are no longer generic alternate attempts. The base Notebook `13`-style branch runs first, then the coordinator computes candidate challenger diagnoses from the base terminal visible evidence using graph, Bayes, MLP, and ranked-differential signals. If the learned branch gate fires, each fresh LLM branch receives a specific assigned target hypothesis and discriminator roots before it starts.

## Policy

Selected policy:

```text
trigger = hypothesis_branch_trigger_mlp_v1
branch budget = 3
judge = hypothesis_forced_graph_bayes_mlp_resolver_v1
```

Branch assignment:

- rank candidate hypotheses from graph top-5, Bayes top-5, MLP top-5, LLM ranked differential, and hybrid ranked differential
- score each challenger using source rank support, graph/Bayes support gaps, and missing pairwise discriminator utility
- assign each branch a target hypothesis, role kind, preferred discriminator roots, and support summary
- run each branch from a fresh message list, without base free-text reasoning

Branch role templates:

- graph/Bayes hypothesis scout
- pairwise discriminator scout
- counter-anchor stress-test scout

## Dry-Run Verification

Artifact root:

```text
artifacts/trajectory_replicates/hypothesis_forced_differential_branching_dryrun_smoke_v1/
```

The dry-run is no-spend and uses scripted responses. It also forces the first smoke case through the branch path so the hypothesis-assignment machinery is actually exercised.

| Metric | Value |
|---|---:|
| Dry-run cases | 2 |
| Base correct | 2/2 |
| Selected correct | 2/2 |
| Branch trigger rate | 1/2 |
| Branches spawned | 3 |
| Mean selected requests | 6.0 |
| Mean total branch requests | 15.0 |
| API calls | 0 live calls |

The forced smoke case generated three assigned branch targets:

| Target hypothesis | Role kind |
|---|---|
| Pericarditis | graph-bayes challenger |
| Myasthenia gravis | pairwise discriminator |
| Myocarditis | counter-anchor stress test |

## Artifacts

Important artifacts:

- `resolved_run_config.json`
- `branch_mlp_train_validate_features.csv`
- `branch_mlp_validation_summary.csv`
- `candidate_resolver_train_validate_features.csv`
- `candidate_resolver_validation_summary.csv`
- `hypothesis_branch_assignments.csv`
- `predictions.csv`
- `branch_case_results.csv`
- `candidate_level_live_scores.csv`
- `traces.jsonl`
- `branch_traces.jsonl`
- `selected_hypothesis_branch_policy.json`
- `summary_metrics.csv`
- `metrics.json`
- figures under `figures/`

## Interpretation

Notebook `30` now has a completed 49-case live run.

Live artifact root:

```text
artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1/
```

Live result:

| Metric | Value |
|---|---:|
| Base branch accuracy | 42/49 = 0.857 |
| Selected policy accuracy | 44/49 = 0.898 |
| Wins vs base | 2 |
| Regressions vs base | 0 |
| Branch trigger rate | 6/49 |
| Branches spawned | 18 |
| Mean selected requests | 6.78 |
| Mean total branch requests | 12.10 |

The live run supports the hypothesis-forced branching direction only partially. It safely improves its own base trajectory, but the gain is modest relative to the branch cost. The candidate source distribution is also revealing:

| Selected source | Cases |
|---|---:|
| Base | 45 |
| Real LLM branch | 1 |
| Pseudo graph candidate | 3 |

The strongest finding is therefore not that full branch-to-completion agents solve the problem. The stronger finding is that Notebook `30` creates a small candidate pool with excellent recall: mean `3.98` unique diagnoses per case, with the true diagnosis present in all `49/49` cases. The hand resolver selected only `44/49`, motivating Notebook `31`'s neural candidate-pool resolver.

The original live test question was:

> Does assigning explicit graph/Bayes/MLP challenger hypotheses before branch execution produce more useful branch diversity than generic fresh-context branches?

The answer is mixed. It improves over the same-run base and avoids regressions, but most value appears to come from candidate generation and graph/Bayes pseudo-candidates rather than from expensive full LLM branch trajectories. Notebook `13` remains the defended proposed method unless a later candidate is prospectively confirmed.

## Live-Run Reliability Patch

During the first live attempt, the notebook failed in the live execution cell because `requests.post(...)` raised a transient TLS/API transport error before an OpenAI response was returned. This was not a hypothesis-branching logic error.

The API caller now uses bounded retry/backoff:

- `LLM_MAX_RETRIES = 5`
- `LLM_REQUEST_TIMEOUT_SECONDS = 180`
- retryable HTTP statuses: `408`, `409`, `425`, `429`, `500`, `502`, `503`, `504`
- non-retryable HTTP errors still fail immediately

The stale error output was cleared from the notebook after the patch.
