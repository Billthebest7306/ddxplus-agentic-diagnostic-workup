# Live Targeted Branching Confirmation Report

Notebook `27` implemented the prospective live multi-agent branching experiment suggested by Notebook `26`.

- notebook: `notebooks/27_live_targeted_branching_confirmation.ipynb`
- live artifact root: `artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/`
- dry-run artifact root: `artifacts/trajectory_replicates/live_targeted_branching_confirmation_dryrun_smoke_v1/`
- status: full 49-case live run completed; not promoted

## Fixed Policy

The policy is frozen before the live 49-case result:

```text
trigger=hybrid_suspicion_v1
branch_budget=2
judge=highest_raw_bayes_posterior
```

The base branch is the Notebook `13` selected-stop live workup loop. The branching layer does not change the first-pass workup policy.

For each case:

1. Run the base branch to completion.
2. Compute label-free terminal suspicion signals from the visible ledger, MLP state, train-derived graph support, and train-derived Bayesian posterior.
3. If `hybrid_suspicion_v1` fires, launch at most two fresh-context full workup branches from the original initial evidence.
4. Give each branch a different role directive:
   - `graph_bayes_scout`
   - `counteranchor_scout`
5. Select among the completed base/branch candidates by raw Bayesian posterior of each branch's own final prediction, with graph support, MLP confidence, and base order used only as tie-breakers.

## Why This Notebook Exists

Notebook `26` showed that broad voting is not the opportunity. Majority vote over five observed live-style trajectories stayed at `43/49`, while oracle best-of-five reached `47/49`.

The strongest diagnostic Notebook `13`-base policy in Notebook `26` used sparse suspicious-state branching and a Bayes judge:

| Metric | Value |
|---|---:|
| Accuracy | 47/49 |
| Base accuracy | 43/49 |
| Wins versus base | 4 |
| Regressions versus base | 0 |
| Branch trigger rate | 0.184 |
| Mean selected requests | 6.86 |
| Mean total branch requests | 9.18 |

Notebook `27` turned that diagnostic feasibility result into a live prospective test.

## Live 49-Case Result

The live run completed under:

```text
artifact_root=artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/
model=gpt-4.1-mini
temperature=0.0
top_p=1.0
```

Headline result:

| Metric | Value |
|---|---:|
| Cases | 49 |
| Base branch accuracy | 42/49 = 0.857 |
| Targeted branching accuracy | 43/49 = 0.878 |
| Top-3 accuracy | 0.918 |
| Top-5 accuracy | 0.939 |
| Wins versus base | 2 |
| Regressions versus base | 1 |
| Branch trigger rate | 9/49 = 0.184 |
| Branches spawned | 18 |
| Mean base requests | 6.92 |
| Mean selected requests | 6.82 |
| Mean total branch requests | 11.45 |

Promotion decision:

```text
not_promoted_keep_notebook13_or_prior_confirmed_method
```

The branching layer produced a net `+1` over its own live base branch, but it did not approach the hoped-for `47/49` target and introduced one regression.

## Paired Outcome Analysis

The two true wins were exactly the kind of lock-in escapes the notebook was built to test:

| Case | True pathology | Base prediction | Branch-selected prediction |
|---|---|---|---|
| `test:35039` | Myocarditis | Sarcoidosis | Myocarditis |
| `test:76022` | Panic attack | Anaphylaxis | Panic attack |

The regression was:

| Case | True pathology | Base prediction | Branch-selected prediction |
|---|---|---|---|
| `test:38475` | Acute COPD exacerbation / infection | Acute COPD exacerbation / infection | Bronchospasm / acute asthma exacerbation |

This regression was a resolver failure, not a trigger failure. The base branch was graph rank `1` and Bayes rank `1` for COPD, but the raw Bayes judge selected the bronchospasm/asthma branch because its posterior was slightly higher (`0.609` versus `0.532`).

## Branch-Candidate Ceiling

The actual live branch pool had a limited ceiling:

| System | Correct |
|---|---:|
| Base branch only | 42/49 |
| Notebook `27` selected judge | 43/49 |
| Oracle over actual base + spawned branch predictions | 44/49 |

The branch pool contained correct alternatives for Myocarditis and Panic attack, and the base branch was already correct for COPD. It did not contain correct final branch predictions for:

- `test:111176`: Acute rhinosinusitis
- `test:11655`: Bronchitis
- `test:81691`: Croup
- `test:19160`: Possible NSTEMI / STEMI
- `test:62878`: Pericarditis

So even a perfect chooser over the actual completed LLM branch predictions would only reach `44/49`.

## Raw Graph/Bayes Diagnostic Signal

A stronger signal remains in the mathematical ledger:

| Diagnostic head over Notebook `27` base final state | Correct |
|---|---:|
| Base branch prediction | 42/49 |
| Raw graph top-1 | 45/49 |
| Raw Bayes top-1 | 45/49 |

The raw graph and raw Bayes top-1 heads both fixed:

- `test:35039`: Myocarditis
- `test:76022`: Panic attack
- `test:19160`: Possible NSTEMI / STEMI

with no base-correct regressions in this diagnostic replay.

This reinforces the earlier Notebook `25` quickcheck: graph/Bayes should be part of the final candidate pool, not merely a score assigned to LLM branch final predictions.

## Failure Modes

There were two major failure classes.

### Silent Confident Wrong Cases

Three selected misses never triggered branching:

| Case | True pathology | Prediction | Why no branch fired |
|---|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | graph, Bayes, MLP, and LLM all agreed with the wrong chronic diagnosis |
| `test:11655` | Bronchitis | URTI | graph, Bayes, MLP, and LLM all agreed with URTI |
| `test:81691` | Croup | Acute otitis media | graph, Bayes, MLP, and LLM all agreed with otitis |

For all three, `suspicion_signal_count = 0`. The current trigger detects uncertainty and contradiction, but these are consensus wrong-answer failures. A confidence-based branch trigger will not catch them.

### Triggered But No Correct LLM Branch

Two selected misses did trigger, but none of the completed LLM branches chose the right diagnosis:

| Case | True pathology | Base prediction | Branch predictions |
|---|---|---|---|
| `test:19160` | Possible NSTEMI / STEMI | Epiglottitis | Acute laryngitis, Acute laryngitis |
| `test:62878` | Pericarditis | Panic attack | Chagas, Spontaneous pneumothorax |

For `test:19160`, graph and Bayes both strongly preferred `Possible NSTEMI / STEMI`, but the resolver could not select it because graph/Bayes top-1 was not included as a pseudo-candidate. This is the clearest argument for pseudo-candidates.

## Diagnostic Resolver Checks

Over the actual Notebook `27` branch candidates:

| Diagnostic chooser over actual candidates | Correct | Wins | Regressions |
|---|---:|---:|---:|
| Highest raw Bayes posterior, selected policy | 43/49 | 2 | 1 |
| Highest graph support | 44/49 | 2 | 0 |
| Cautious Bayes with base protection | 44/49 | 2 | 0 |

These are post-hoc diagnostics, not promoted policies. They show that the live branch trajectories were useful, but the selected raw-Bayes-only resolver was too eager to override a graph/Bayes-supported base answer.

## Leakage Controls

- No 49-case labels are used in the trigger, branch prompts, or judge.
- Branches receive structured ledger state and branch-role directives, not the base branch's free-text reasoning.
- Graph edges come from Notebook `16`.
- Bayesian likelihoods and priors come from Notebook `19`.
- The MLP sees only currently revealed evidence.
- Labels are used only after prediction for metrics.

## Dry-Run Validation

A no-spend two-case smoke run was executed with:

```text
NOTEBOOK27_RUN_LIVE_API=0
NOTEBOOK27_ALLOW_DRY_RUN=1
```

Smoke result:

| Metric | Value |
|---|---:|
| Cases | 2 |
| Base correct | 2/2 |
| Branch-selected correct | 2/2 |
| Branch triggers | 0 |
| Mean total branch requests | 6.0 |

The two top-to-bottom smoke cases were stable, so no branch was naturally triggered. An additional no-spend forced branch-path smoke was run on one case to confirm that a fresh branch can execute, apply the early divergent-root guard, and be scored by the raw Bayes judge.

This validates notebook order, graph/Bayes loading, artifact writing, figures, the artifact contract, and the branch execution path. It is not a method result.

## Artifact Contract

The live artifact root contains:

- `benchmark_cases.csv`
- `reference_summary.csv`
- `resolved_run_config.json`
- `predictions.csv`
- `paired_notebook13_vs_live_branching.csv`
- `branch_case_results.csv`
- `candidate_branch_scores.csv`
- `traces.jsonl`
- `branch_traces.jsonl`
- `raw_api_responses.jsonl`
- `metrics.json`
- `summary_metrics.csv`
- `hard_case_live_branch_audits.json`
- `selected_live_branching_policy.json`
- figures under `figures/`

## Current Interpretation

Notebook `27` partially validates the branching hypothesis: fresh branches can escape wrong-answer lock-in on some cases. However, the selected policy is not promoted because the live run reached only `43/49`, added substantial total branch request cost, and introduced one regression.

The next credible branching notebook should not be confidence-triggered alone. It should add:

- graph/Bayes pseudo-candidates to the final resolver
- base protection when the base answer is graph rank `1` and Bayes rank `1`
- a confounder-coverage branch trigger that detects under-adjudicated nearest-neighbor disease families even when MLP/graph/Bayes are confidently wrong

Notebook `13` remains the defended evidence-acquisition backbone. Notebook `23` remains the strongest offline graph/Bayes enhancement candidate. Notebook `27` is best reported as a partial live confirmation of branch diversity and a negative confirmation of raw-Bayes-only branch resolution.
