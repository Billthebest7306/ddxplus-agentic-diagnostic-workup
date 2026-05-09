# Offline Branching Trajectory Lab

Last updated: 2026-05-09

## Summary

Notebook `26` is an offline multi-trajectory branching lab for the Notebook `13`-style live diagnostic workup.

- notebook: `notebooks/26_offline_branching_trajectory_lab.ipynb`
- artifact root: `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/`
- API usage: none

The notebook combines five already-collected base trajectories over the same 49-case benchmark:

| Run | Correct | Accuracy | Mean requests |
|---|---:|---:|---:|
| Notebook `13` frozen artifact | 43/49 | 0.878 | 6.59 |
| Notebook `24` fresh base | 45/49 | 0.918 | 6.20 |
| Notebook `25` replicate r01 | 44/49 | 0.898 | 7.00 |
| Notebook `25` replicate r02 | 42/49 | 0.857 | 7.02 |
| Notebook `25` replicate r03 | 42/49 | 0.857 | 6.82 |

The goal is to test whether a future multi-agent system should branch broadly, branch only on suspicious states, or avoid branching.

## Divergence Result

Across the five observed trajectories:

| Metric | Value |
|---|---:|
| Cases | 49 |
| Same final prediction across all five runs | 41/49 |
| Prediction instability cases | 8/49 |
| Correctness instability cases | 7/49 |
| Majority-vote accuracy | 43/49 = 0.878 |
| Oracle best-of-five accuracy | 47/49 = 0.959 |
| Notebook `13` misses with at least one alternate correct run | 4 |

The important result is that majority voting does not solve the problem. The opportunity comes from targeted branch selection and adjudication.

Case-level first divergence types:

| First divergence type | Cases |
|---|---:|
| Request choice | 32 |
| Stop-vs-request | 6 |
| No action-sequence divergence | 11 |

State-level same-prefix divergence:

| State divergence type | States |
|---|---:|
| Request choice | 49 |
| Stop-vs-request | 15 |
| Impactful downstream correctness instability | 10 |

This supports the multi-agent hypothesis: the model is not globally random, but some visible states are fragile enough that a different request or a continue-vs-stop decision changes the final diagnosis.

## Branch Policy Simulation

Notebook `26` simulates branch policies over the observed runs. A policy can:

1. take one run as the base trajectory,
2. decide whether to spawn alternate observed trajectories using only terminal features from the base run,
3. adjudicate among the base and spawned branches using graph, Bayesian, MLP, and consensus features.

The label-free suspicion signals include:

- LLM/MLP disagreement
- cap hit
- uncertain MLP margin or entropy
- early uncertain stop
- graph conflict with the final prediction
- Bayesian conflict with the final prediction
- graph and Bayes both preferring a different top-1

The pre-registered next-live candidate in the notebook is:

```text
trigger=hybrid_suspicion_v1
branch_budget=1
judge=cautious_fused_judge
```

It is conservative: it branches on about one fifth of pooled base decisions and requires a fused graph/Bayes/MLP advantage before overriding the base.

| Scope | Accuracy | Base accuracy | Wins | Regressions | Branch rate | Mean selected requests | Mean total branch requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| All five base runs pooled | 224/245 = 0.914 | 0.882 | 8 | 0 | 0.192 | 6.62 | 8.34 |
| Notebook `13` base only | 46/49 = 0.939 | 0.878 | 3 | 0 | 0.184 | 6.63 | 7.84 |

The strongest diagnostic Notebook `13` branch policy was:

```text
trigger=hybrid_suspicion_v1
branch_budget=2
judge=highest_bayes_posterior
```

On the Notebook `13` base scope, this reaches:

| Metric | Value |
|---|---:|
| Accuracy | 47/49 = 0.959 |
| Base accuracy | 43/49 = 0.878 |
| Wins versus base | 4 |
| Regressions versus base | 0 |
| Branch trigger rate | 0.184 |
| Mean branches spawned | 0.367 |
| Mean selected requests | 6.86 |
| Mean total branch requests | 9.18 |

Fixed Notebook `13` misses:

| Case | True pathology | Notebook `13` | Branch-selected prediction | Selected branch |
|---|---|---|---|---|
| `test:125508` | Unstable angina | Anemia | Unstable angina | Notebook `24` base |
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | Acute COPD exacerbation / infection | Notebook `25` r01 |
| `test:81691` | Croup | Anemia | Croup | Notebook `24` base |
| `test:8666` | Influenza | HIV initial infection | Influenza | Notebook `24` base |

Remaining misses:

| Case | True pathology | Prediction |
|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis |
| `test:62878` | Pericarditis | Anemia |

## Replicate Final-Layer Graph/Bayes Quickcheck

After Notebook `26`, a quick offline check replayed final graph/Bayes scoring over the three Notebook `25` replicate traces only. This uses the same final visible evidence states and makes no API calls.

Output:

- `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/replicate_graph_bayes_final_layer_quickcheck.csv`

Result:

| Replicate | Base | Conservative graph/Bayes final layer | Raw graph top-1 | Raw Bayes top-1 |
|---|---:|---:|---:|---:|
| `r01` | 44/49 | 44/49 | 45/49 | 45/49 |
| `r02` | 42/49 | 42/49 | 44/49 | 44/49 |
| `r03` | 42/49 | 42/49 | 44/49 | 44/49 |

The strict conservative override did not fire on any replicate, so it made no changes. However, the raw graph and Bayes posterior top-1 heads improved all three replicates with zero regressions in this diagnostic check:

| Replicate | Raw graph/Bayes wins over base | Raw graph/Bayes regressions |
|---|---:|---:|
| `r01` | 1 | 0 |
| `r02` | 2 | 0 |
| `r03` | 2 | 0 |

Cases where raw graph/Bayes recovered a base miss included Panic attack in `r01-r03` and Myocarditis in `r02-r03`. The takeaway is not that raw graph/Bayes should become the final head; it is that graph/Bayes remains valuable as a branch judge, while the older conservative final override is too cautious for these fresh trajectories.

## Interpretation

Notebook `26` gives the clearest evidence so far for the multi-agent direction:

- running more trajectories and taking a majority vote is not enough
- the reachable ceiling from existing live-style trajectories is `47/49`
- the useful branch region is small, about 18-28% of cases depending on trigger strictness
- graph/Bayes support is more useful as an adjudicator over completed branches than as a question controller
- the replicate quickcheck reinforces this: raw graph/Bayes heads improved all three Notebook `25` replicates diagnostically, but the strict final override did not fire
- a two-branch suspicious-state policy can recover the same `47/49` target that Notebook `23` reached offline, but through trajectory diversity rather than deterministic rescue questions

This is not promoted as a final method because it reuses observed branches and evaluates policy curves with 49-case labels. The correct next step is a live confirmation notebook that implements the fixed trigger and judge prospectively.

## Artifact Contract

Notebook `26` writes:

- `resolved_run_config.json`
- `analysis_summary.json`
- `run_level_summary.csv`
- `case_run_outcomes.csv`
- `turn_level_branch_features.csv`
- `state_divergence_points.csv`
- `case_divergence_summary.csv`
- `branch_policy_summary.csv`
- `branch_policy_pareto.csv`
- `branch_oracle_summary.csv`
- `branch_policy_case_results.csv`
- `candidate_branch_scores.csv`
- `replicate_graph_bayes_final_layer_quickcheck.csv`
- `hard_case_branch_audits.json`
- `recommended_branching_policy.json`
- figures under `figures/`

Validation:

- static parsing passed
- top-to-bottom execution completed using a local notebook runner
- no live API key was required
- no live API calls were made
