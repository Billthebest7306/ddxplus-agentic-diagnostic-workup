# MLP-Gated Confounder Graph-Bayes Branching

Notebook `28` implements the learned-gate live multi-agent candidate after Notebook `27`.

- notebook: `notebooks/28_mlp_gated_confounder_graph_bayes_branching.ipynb`
- dry-run artifact root: `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1/`
- live artifact root: `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1/`
- status: live 49-case run completed; not promoted as the defended method

## Method

Notebook `27` used fixed suspicion rules. Notebook `28` replaces that trigger with a learned branch gate:

```text
branch_trigger_mlp_v1_probability >= 0.375
```

The branch gate is trained only on train/validate synthetic partial evidence states. The input features combine diagnostic MLP confidence/margin/entropy, graph support/ranks, Bayesian posterior/ranks, graph-Bayes agreement, request count, visible-root count, and pairwise confounder coverage.

If the gate fires, the notebook can spawn up to three fresh-context LLM workup branches:

- `graph_bayes_scout`
- `confounder_pair_scout`
- `counteranchor_scout`

The final resolver scores real branch outputs plus graph/Bayes/MLP pseudo-candidates. It also protects a base answer when the base is graph rank `1` and Bayes rank `1` unless a challenger clears a stronger resolver margin.

## Live 49-Case Result

| Metric | Value |
|---|---:|
| Cases | 49 |
| Base branch accuracy | 42/49 = 0.857 |
| Notebook `28` selected accuracy | 44/49 = 0.898 |
| Top-3 accuracy | 0.959 |
| Top-5 accuracy | 0.959 |
| Wins versus base | 2 |
| Regressions versus base | 0 |
| Changed predictions | 2 |
| Branch trigger rate | 4/49 = 0.082 |
| Branches spawned | 12 |
| Mean base requests | 6.92 |
| Mean selected requests | 6.63 |
| Mean total branch requests | 9.96 |

Selection sources:

| Source | Selected cases |
|---|---:|
| Base | 47 |
| `counteranchor_scout` | 1 |
| `pseudo_graph_top1` | 1 |

Promotion decision:

```text
not_promoted_keep_notebook13_or_prior_confirmed_method
```

Notebook `28` improved its own live base by two cases with no paired regressions, but it did not reach the `47/49` target and did not beat the strongest live Notebook `24` base result of `45/49`.

## Paired Changes

| Case | True pathology | Base prediction | Notebook `28` prediction | Source | Outcome |
|---|---|---|---|---|---|
| `test:35039` | Myocarditis | Sarcoidosis | Myocarditis | `pseudo_graph_top1` | win |
| `test:62878` | Pericarditis | Panic attack | Pericarditis | `counteranchor_scout` | win |

The resolver's base protection avoided the COPD-style regression seen in Notebook `27`. On triggered COPD and Acute otitis media cases, branch candidates scored high but the base answer remained graph/Bayes-supported and was preserved.

## Remaining Misses

| Case | True pathology | Prediction | Branch trigger | Notes |
|---|---|---|---:|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | no | Correct disease was rank 2 in the final differential. |
| `test:11655` | Bronchitis | URTI | no | Correct disease was rank 2 in the final differential and also the challenger. |
| `test:81691` | Croup | Acute otitis media | yes | No scored candidate was Croup, but a branch differential contained Croup at rank 3. |
| `test:8666` | Influenza | HIV (initial infection) | no | Correct disease was rank 2 in the final differential and also the challenger. |
| `test:76022` | Panic attack | Myocarditis | no | Correct disease did not appear in the scored pool or top-5 differential. |

## Post-Hoc Candidate-Pool Analysis

The key post-run finding is that the judge was not the main bottleneck. The scored candidate pool itself had only a `44/49` oracle ceiling:

| Candidate pool | Oracle correct |
|---|---:|
| Current scored candidates | 44/49 |
| Current candidates plus all candidate ranked top-1 | 44/49 |
| Current candidates plus all candidate ranked top-2 | 47/49 |
| Current candidates plus all candidate ranked top-3 | 48/49 |
| Current candidates plus all candidate ranked top-5 | 48/49 |
| Selected final ranked differential top-2 only | 47/49 |

This means Notebook `28` was too narrow about what counts as a candidate. It scored final top-1 predictions from the base/branches plus graph, Bayes, and MLP top-1 pseudo-candidates. It did not promote the LLM's second or third differential entries into resolver candidates, even when the correct disease was already visible in that differential.

The post-hoc analysis artifacts are:

- `posthoc_candidate_pool_oracle_summary.csv`
- `posthoc_miss_candidate_coverage.csv`
- `posthoc_notebook28_analysis.json`

## Calibration Diagnostics

Branch trigger validation:

| Metric | Value |
|---|---:|
| Synthetic train states | 4000 |
| Synthetic validate states | 2000 |
| Validation wrong-anchor rate | 0.436 |
| Selected threshold | 0.375 |
| Precision at threshold | 0.823 |
| Recall at threshold | 0.940 |
| Branch rate at threshold | 0.499 |
| AUC | 0.954 |
| Average precision | 0.931 |
| Brier score | 0.083 |

Candidate resolver validation:

| Metric | Value |
|---|---:|
| Validate candidate rows | 16670 |
| Candidate positive rate | 0.109 |
| Candidate AUC | 0.955 |
| Candidate average precision | 0.809 |

The branch MLP generalized imperfectly from synthetic validation states to live trajectories. Validation branch rate was about `0.50`, while live branch rate was only `0.082`. In live misses, graph/Bayes/MLP often agreed with the wrong anchor, causing the gate to treat silent consensus-wrong cases as safe.

## Evaluation Figures

Notebook `28` writes the richer evaluation suite requested for this iteration:

- accuracy comparison
- paired outcomes
- branch trigger probability histogram
- trigger probability by correctness
- branch rate vs threshold
- validation gain vs threshold
- request-cost distribution
- total branch requests by case
- pseudo-candidate source counts
- resolver score diagnostics
- graph/Bayes/MLP agreement heatmap
- confounder coverage by case
- hard-case rank movement
- branch MLP calibration
- branch MLP ROC/PR curves
- permutation-style feature importance

## Current Interpretation

Notebook `28` is a better live branching system than Notebook `27`: it produced `2` wins, `0` regressions, and kept selected request cost near Notebook `13`. But it did not produce enough correct alternatives to reach `47/49`.

The next credible iteration should be a listwise differential adjudicator, not simply more branching. The candidate pool should include:

- base ranked top-5
- branch ranked top-5
- graph top-5
- Bayes top-5
- MLP top-5
- confounder challenger candidates

Then a calibrated resolver can choose among those candidates using graph/Bayes/MLP support, rank position, contradiction, coverage, and request-state features. Notebook `28` shows that the mathematical resolver can avoid regressions; the missing piece is giving it the full differential menu.

## Follow-Up Notebook 29

Notebook `29` implemented this listwise follow-up offline over the frozen Notebook `28` live traces:

- report: `reports/algorithmic_ledger/listwise_differential_graph_bayes_adjudicator_report.md`
- artifact root: `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`

It improves Notebook `28` from `44/49` to `45/49` with one win and zero regressions by promoting Croup from the `graph_bayes_scout` ranked differential at rank 3. It is still not promoted because it does not reach `47/49`.

The follow-up changes the bottleneck diagnosis: the expanded candidate pool now has a `47/49` oracle at ranked top-2, `48/49` at ranked top-3, and `49/49` over all exploded graph/Bayes/MLP/ranked candidates. The remaining problem is calibrated selection among close confounders, not candidate availability alone.
