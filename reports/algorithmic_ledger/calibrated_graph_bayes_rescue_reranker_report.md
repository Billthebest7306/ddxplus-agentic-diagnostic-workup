# Calibrated Graph-Bayes Rescue Reranker

Last updated: 2026-05-08

## Summary

Notebook `23` is an offline graph/Bayes/MLP rescue layer over the saved Notebook `13` live workup trace.

- notebook: `notebooks/23_calibrated_graph_bayes_rescue_reranker.ipynb`
- artifact root: `artifacts/graph_algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_49case_v1/`
- API usage: none

The method keeps Notebook `13` as the first-pass live evidence-acquisition system, then applies calibrated mathematical certificates and a tiny deterministic rescue continuation. It uses train/validate-derived synthetic partial evidence states for candidate scoring and does not use the 49-case labels for training or threshold selection.

Live confirmation note: Notebook `24` later tested this frozen policy in a fresh 49-case live run. The rescue layer was **not promoted**: the fresh live base reached `45/49`, and the live rescue also reached `45/49` with nine extra rescue requests. Notebook `23` should therefore be described as the strongest offline graph/Bayes enhancement candidate, not as a live-confirmed replacement.

## Method

Notebook `23` combines four signals:

- Notebook `13` final prediction, LLM top-5, MLP top-5, and initial prior
- Notebook `16` train-derived graph support/contradiction
- Notebook `19` Bayesian likelihood tables
- the existing partial-evidence MLP checkpoint

The selected policy is:

```text
calibrated_graph_bayes_rescue_v1
```

Decision certificates:

```text
1. prior recovery:
   use prior_top1 if graph rank <= 5,
   Notebook 13 graph support is negative,
   and prior graph support is better than Notebook 13.

2. conservative graph critic:
   reuse the Notebook 22 graph top-1 override certificate.

3. rescue continuation:
   if stop_reason == agent_stop,
   num_requests <= 3,
   and graph_top1_posterior < 0.80,
   ask up to 3 graph/Bayes discriminator roots.

4. post-rescue rerank:
   score the post-rescue candidate pool with the
   train/validate-calibrated L2 reranker,
   use graph rank only as a near-tie breaker,
   and accept only when the candidate passes the
   validation-selected score/margin thresholds and graph support guard.
```

The rescue question utility is:

```text
utility(root) =
  expected entropy reduction over the active graph candidate set
  + pairwise confounder resolution bonus
  - request cost
```

## Result

| System | Correct | Accuracy | Mean requests | Extra requests | Improvements | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 43/49 | 0.878 | 6.59 | 0 | 0 | 0 |
| Notebook `22` conservative graph critic | 44/49 | 0.898 | 6.59 | 0 | 1 | 0 |
| Notebook `23` calibrated graph-Bayes rescue | 47/49 | 0.959 | 6.96 | 18 total | 4 | 0 |

Promotion decision:

```text
offline_candidate_promoted
```

Notebook `23` reaches the target `47/49` with zero regressions against Notebook `13`. It adds `18` total rescue requests across the `49` cases, increasing mean requests from `6.59` to `6.96`.

## Paired Error Analysis

Fixed Notebook `13` misses:

| Case | True pathology | Notebook `13` | Notebook `23` | Source |
|---|---|---|---|---|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | Acute COPD exacerbation / infection | prior recovery |
| `test:81691` | Croup | Anemia | Croup | conservative graph critic |
| `test:8666` | Influenza | HIV initial infection | Influenza | rescue rerank |
| `test:125508` | Unstable angina | Anemia | Unstable angina | rescue rerank |

Remaining misses:

| Case | True pathology | Notebook `23` prediction | Interpretation |
|---|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | graph and model signals still prefer the chronic neighbor |
| `test:62878` | Pericarditis | Anemia | revealed evidence does not support Pericarditis strongly enough |

Paired counts:

| Outcome | Cases |
|---|---:|
| Both correct | 43 |
| Notebook `23` only correct | 4 |
| Notebook `13` only correct | 0 |
| Both wrong | 2 |

## Interpretation

This is the first algorithmic-ledger result that changes the project materially. Notebook `22` proved graph support could fix one final answer. Notebook `23` shows the stronger path: use the graph as a calibrated critic, add Bayesian discriminator requests only for suspicious early stops, and preserve Notebook `13` as the live first-pass workup.

The result should be presented as an offline enhancement candidate, not as a live-confirmed replacement. Its scientific value is that it reaches `47/49` without new API calls, without changing the saved LLM workup trace, and without tuning directly on the six Notebook `13` misses.

## Limitations

- This is still an offline saved-trace result.
- Notebook `24` did not validate the rescue as a live improvement on a fresh trajectory.
- The rescue continuation reveals DDXPlus evidence roots offline; live trajectory variation can change which cases need rescue.
- The candidate scorer is calibrated on synthetic train/validate partial states, which are useful but not identical to real LLM trajectories.
- The method does not fix acute-vs-chronic rhinosinusitis or Pericarditis.

## Artifact Contract

Notebook `23` writes:

- `resolved_run_config.json`
- `synthetic_state_generation_summary.csv`
- `candidate_level_train_validate_features.csv`
- `reranker_validation_summary.csv`
- `notebook13_rescue_case_results.csv`
- `candidate_level_49case_scores.csv`
- `rescue_trace.jsonl`
- `paired_notebook13_vs_rescue_reranker.csv`
- `hard_case_rescue_audits.json`
- `selected_rescue_policy.json`
- `rescue_policy_summary.csv`
- figures under `figures/`

Static parsing and top-to-bottom execution completed successfully with no API key and no live API path.
