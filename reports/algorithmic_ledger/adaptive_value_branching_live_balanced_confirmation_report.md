# Notebook 37: Adaptive Value Branching Live Balanced Confirmation

Notebook `37` is the larger live confirmation experiment for the latest candidate-pool architecture.

## Purpose

The previous 49-case replay showed that the adaptive controller from Notebook `35` can preserve the Notebook `33` result:

- `49/49` candidate-pool recall
- `48/49` final accuracy
- `8.98` mean total requests
- branch 2/3 allowed, but not used on the saved replay

Notebook `36` then showed that the saved 49-case branch pool cannot prove natural branch-2/3 recovery. Notebook `37` therefore creates the larger live confirmation run needed to test that behavior prospectively.

## Cohort

The notebook samples a balanced fresh test cohort:

```text
cases_per_pathology = 2
exclude original Notebook 13/30 49-case confirmation set = true
split = test
```

DDXPlus has 49 pathologies in this release, so this produced a 98-case balanced confirmation rather than an exactly random 100-case sample.

## Policy

The live workup keeps the Notebook `13` selected-stop branch as the base and uses:

```text
LLM model = gpt-4.1-mini
temperature = 0.0
top_p = 1.0
branch_trigger_threshold = 0.80
max_branches = 3
continuation_value_threshold = 0.40
```

The first branch launches only when the learned branch-trigger MLP fires. Branch 2/3 launch only if the label-free continuation value remains high after the previous branch completes.

## Final Resolver And Top-K Metrics

Notebook `37` restores ranked differential evaluation:

- base branch top-1/top-3/top-5
- branch-judge selected top-1/top-3/top-5
- candidate-pool GBM plus close-confounder final top-1/top-3/top-5

The final resolver layer uses the Notebook `32` `gradient_boosting_name_family` candidate scorer, then applies a Notebook `33` style close-confounder discriminator:

```text
extra_root_budget = 2
flag if close top pair and score margin <= 0.70 and missing pair utility >= 0.08
override only if extra-root log Bayes factor >= 2.0
```

## Artifact Contract

Artifacts were written under:

```text
artifacts/trajectory_replicates/adaptive_value_branching_live_balanced2_v1/
```

Key expected outputs:

- `benchmark_cases.csv`
- `balanced_pathology_case_counts.csv`
- `reference_case_exclusion_summary.csv`
- `predictions.csv`
- `candidate_level_live_scores.csv`
- `candidate_level_live_resolver_scores.csv`
- `candidate_pool_topk_rankings.csv`
- `adaptive_live_final_predictions.csv`
- `close_confounder_discriminator_trace.csv`
- `adaptive_branch_decision_trace.csv`
- `topk_summary.csv`
- `metrics_final.json`
- `selected_adaptive_live_policy.json`
- `notebook37_paired_outcome_analysis.csv`
- `notebook37_failure_modes.csv`
- `notebook37_truth_rank_analysis.csv`
- `notebook37_branch_trigger_threshold_counterfactual.csv`
- `notebook37_analysis_summary.json`

## Live Result

| System | Correct | Accuracy | Top-3 | Top-5 | Mean selected requests | Mean total requests |
|---|---:|---:|---:|---:|---:|---:|
| Base Notebook `13`-style branch | 83/98 | 0.847 | 0.888 | 0.908 | 8.20 | 8.20 |
| Notebook `37` branch-judge selected | 86/98 | 0.878 | 0.908 | 0.929 | 8.20 | 8.31 |
| Notebook `37` GBM + close-confounder final | 88/98 | 0.898 | 0.939 | 0.939 | 8.37 | 8.43 |

Paired outcomes versus the base branch:

| Output | Wins | Regressions | Net |
|---|---:|---:|---:|
| Branch-judge selected | 5 | 2 | +3 |
| Final GBM + close-confounder | 5 | 0 | +5 |

The user's observed `5` wins and `2` regressions are the intermediate branch-judge result. The final resolver/discriminator layer removes both regressions:

- `test:2255`, Croup: branch judge regressed to Epiglottitis; close-confounder extra roots gave a strong Bayes factor back to Croup.
- `test:83391`, SLE: branch judge regressed to Pancreatic neoplasm; the final GBM resolver kept SLE first.

## Main Failure Modes

The disappointing result is not primarily a final-resolver collapse. It is a candidate-generation and branch-trigger failure.

| Failure mode | Cases | Meaning |
|---|---:|---|
| Candidate-pool miss | 6 | The true diagnosis was absent from the final GBM candidate pool. No resolver could recover these. |
| Resolver miss with truth in pool | 4 | The true diagnosis was present, but ranked below a confounder. |

Candidate-pool recall dropped from the Notebook `30` 49-case diagnostic ceiling of `49/49` to `92/98 = 0.939`. This capped final top-3/top-5 at `92/98`, because all true diagnoses that entered the final pool were ranked in the top three:

| True diagnosis rank in final pool | Cases |
|---|---:|
| 1 | 87 |
| 2 | 4 |
| 3 | 1 |
| Missing | 6 |

Candidate-pool misses:

| Case | True diagnosis | Final prediction | Trigger probability | Notes |
|---|---|---|---:|---|
| `test:54031` | Acute laryngitis | Viral pharyngitis | 0.011 | Cap hit; truth absent from final pool. |
| `test:127556` | Acute rhinosinusitis | Pneumonia | 0.027 | Graph/Bayes/MLP all favored pneumonia/allergic/chronic sinus variants. |
| `test:85739` | Croup | Bronchitis | 0.842 | Only triggered branch; branch target was Cluster headache and did not recover Croup. |
| `test:39464` | Inguinal hernia | Viral pharyngitis | 0.796 | Near trigger threshold but did not branch; truth absent from final pool. |
| `test:20922` | Pericarditis | Bronchitis | 0.021 | Truth absent from final pool. |
| `test:63258` | Stable angina | Possible NSTEMI / STEMI | 0.521 | Moderate trigger score but below threshold; stable angina absent. |

Resolver misses where the true diagnosis was present:

| Case | True diagnosis | Final prediction | True rank |
|---|---|---|---:|
| `test:92249` | Atrial fibrillation | Myocarditis | 3 |
| `test:37106` | Bronchiolitis | Bronchitis | 2 |
| `test:108410` | Myasthenia gravis | Acute dystonic reactions | 2 |
| `test:130885` | Pulmonary embolism | Acute dystonic reactions | 2 |

## Branching Behavior

The adaptive controller almost never actually branched:

```text
branch_trigger_threshold = 0.80
triggered cases = 1/98
branches spawned = 1 total
branch 2 launches = 0
branch 3 launches = 0
```

The only triggered case was `test:85739` Croup. The branch was assigned to Cluster headache and ended at Bronchitis, so it did not help.

This explains why the result did not validate the multi-agent hypothesis as strongly as hoped. Most of the gain came from graph/Bayes/MLP pseudo-candidates, not from fresh LLM branches:

```text
branch-judge selected sources:
base = 83 cases
pseudo_graph_topk = 13 cases
pseudo_bayes_topk = 1 case
pseudo_mlp_top1 = 1 case
real branch = 0 selected cases
```

The final GBM layer selected one real branch candidate, but it was still wrong for Croup.

## Why The 49-Case Percentage Did Not Transfer

The old `48/49` replay result did not behave like a stable percentage for three reasons:

1. The 49-case replay was no longer an independent confirmation set. Several design choices after Notebook `30` were motivated by that artifact, so the old result was an optimistic development estimate even though the model features themselves were train/validate-derived.
2. Candidate-pool recall changed. Notebook `30`/`35` had `49/49` pool recall on the original cases. Notebook `37` had only `92/98`. That alone caps any resolver at `93.9%`.
3. The live branch trigger distribution shifted. The threshold `0.80` fired on `6/49` old cases, but only `1/98` fresh balanced cases. The synthetic validation table estimated a much higher branch rate at this threshold, so terminal live states are not calibrated like the synthetic states.

Counterfactual trigger counts on this live run:

| Threshold | Cases that would trigger |
|---:|---:|
| 0.80 | 1 |
| 0.70 | 2 |
| 0.60 | 5 |
| 0.50 | 8 |
| 0.30 | 9 |
| 0.20 | 10 |

Lowering the threshold would have branched more cases, including Inguinal hernia and Stable angina, but it would not catch all misses: Acute laryngitis, Acute rhinosinusitis, and Pericarditis had trigger probabilities below `0.03`.

## Verification Completed

- Notebook `37` code cells static-parse successfully.
- Notebook-equivalent script was generated at `scripts/adaptive_value_branching_live_balanced_confirmation_nb37.py`.
- `python3 -m py_compile scripts/adaptive_value_branching_live_balanced_confirmation_nb37.py` passed.
- Live artifact contract completed for 98 cases.
- Additional failure-analysis artifacts were generated after the run.

## Interpretation

Notebook `37` is a useful but sobering confirmation. It improves the base branch from `83/98` to `88/98` with no final regressions and only a small request-cost increase, but it does not reproduce the `48/49` replay accuracy rate.

The current conclusion:

- the architecture helps, but the old 49-case result was optimistic
- candidate generation is not universally solved; the fresh pool missed `6/98`
- adaptive branching did not meaningfully activate at the selected threshold
- top-3/top-5 are still valuable: final top-3/top-5 reached `92/98`, which exactly matches candidate-pool recall
- the next fix should target candidate-pool expansion and branch-trigger calibration, not just stronger final resolution
