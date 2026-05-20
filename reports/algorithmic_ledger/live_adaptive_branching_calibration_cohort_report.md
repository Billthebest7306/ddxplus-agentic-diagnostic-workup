# Notebook 38: Live Adaptive Branching Calibration Cohort

Notebook `38` is the 196-case live calibration cohort for the adaptive candidate-pool architecture.

## Purpose

Notebook `37` improved a fresh balanced base branch from `83/98` to `88/98`, but it exposed the generalization problem:

- candidate-pool recall fell to `92/98`
- the `0.80` branch trigger fired on only `1/98` cases
- branch 2/3 never launched

Notebook `38` therefore lowered the trigger thresholds and ran a larger calibration cohort. This run is not a final held-out confirmation. Its labels may be used to choose thresholds for the next frozen run, so its accuracy should be reported as calibration/development evidence.

## Cohort

```text
cases_per_pathology = 4
cases = 196
split = test
exclude prior live benchmark cases = true
```

The notebook excludes prior benchmark cohorts by reading existing `benchmark_cases.csv` artifacts, including the original 49-case confirmation and Notebook `37`'s 98-case balanced confirmation.

## Exploratory Live Policy

```text
LLM model = gpt-4.1-mini
temperature = 0.0
top_p = 1.0
branch_trigger_threshold = 0.20
max_branches = 3
continuation_value_threshold = 0.20
target candidate-pool recall for calibration = 0.98
```

These values were intentionally exploratory. The goal was to collect live branch/candidate-pool behavior, not to freeze a final policy.

## Headline Result

Artifacts:

```text
artifacts/trajectory_replicates/adaptive_value_branching_live_calibration196_v1/
```

| System | Cases | Correct | Accuracy | Top-3 | Top-5 | Mean selected requests | Mean total requests | P90 total | Max total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Notebook `13`-style branch | 196 | 172 | 0.878 | 0.944 | 0.964 | 6.77 | 6.77 | 17 | 24 |
| Notebook `38` branch-judge selected | 196 | 183 | 0.934 | 0.974 | 0.985 | 6.79 | 9.52 | 22 | 85 |
| Notebook `38` GBM + close-confounder final | 196 | 184 | 0.939 | 0.990 | 0.990 | 6.77 | 9.56 | 22 | 85 |

Paired final result versus the base branch:

| Outcome | Cases |
|---|---:|
| Both correct | 171 |
| Final only correct | 13 |
| Base only correct | 1 |
| Both wrong | 11 |

The final layer improves the same-run base by `+12` net cases: `13` wins and `1` regression.

## Candidate-Pool Recall

| Candidate subset | Recall | Count | Mean pool size | Max pool size |
|---|---:|---:|---:|---:|
| Base only | 0.847 | 166/196 | 0.96 | 1 |
| Base + pseudo graph/Bayes/MLP | 0.923 | 181/196 | 3.62 | 6 |
| Base + real branches | 0.913 | 179/196 | 1.13 | 4 |
| All candidates | 0.990 | 194/196 | 3.79 | 8 |

This is the most important positive result. Notebook `37` had only `92/98` candidate-pool recall; Notebook `38` restores near-complete candidate coverage on a larger cohort. Candidate generation is not perfect, but it is much closer to the desired operating point.

## Branching Behavior

| Metric | Value |
|---|---:|
| Branch trigger rate | 23/196 = 0.117 |
| Branches spawned total | 42 |
| Mean branches spawned | 0.214 |
| Cases with 0 branches | 173 |
| Cases with 1 branch | 12 |
| Cases with 2 branches | 3 |
| Cases with 3 branches | 8 |

The adaptive controller did naturally launch branch 2/3 on a small subset of hard cases. This directly addresses the Notebook `36` limitation: the previous 49-case replay could not prove natural branch 2/3 behavior because the saved pool lacked examples. Notebook `38` does show branch 2/3 firing live.

The cost issue is the long tail. The selected path remains efficient on average (`6.77` requests), but total branch requests reach `85` in the hardest cases because several full branches can each run near the cap. The next frozen policy should preserve candidate recall while reducing this tail.

## Calibration Sweeps

The branch-trigger sweep is a flagging analysis over observed live terminal states. It says which cases would be flagged at different trigger thresholds; it does not prove those additional cases would be fixed, because branches were only actually spawned under the executed threshold.

| Trigger threshold | Cases flagged | Base incorrect flagged | Candidate-pool misses flagged | Final errors flagged |
|---:|---:|---:|---:|---:|
| 0.05 | 39 | 20 | 2/2 | 9/12 |
| 0.10 | 27 | 18 | 2/2 | 8/12 |
| 0.20 | 23 | 16 | 2/2 | 6/12 |
| 0.50 | 16 | 12 | 2/2 | 4/12 |
| 0.80 | 7 | 7 | 2/2 | 4/12 |

The resolver-margin sweep points to the other bottleneck:

| Resolver score-margin threshold | Cases flagged | Final errors flagged | Resolver misses flagged |
|---:|---:|---:|---:|
| 0.50 | 17 | 7/12 | 5 |
| 0.70 | 29 | 9/12 | 7 |
| 0.75 | 30 | 10/12 | 8 |
| 0.85 | 45 | 12/12 | 10 |

Interpretation: a lower branch threshold can catch more suspicious cases, but many remaining errors are not candidate-generation failures. A resolver-margin guard plus targeted close-confounder evidence may be more cost-effective than simply launching more full LLM branches.

## Failure Modes

Final misses: `12/196`.

| Failure mode | Count |
|---|---:|
| Resolver miss with truth in candidate pool | 10 |
| Candidate-pool miss | 2 |

The two candidate-pool misses were:

| Case | True pathology | Final prediction | Notes |
|---|---|---|---|
| `test:127067` | Allergic sinusitis | Acute otitis media | Branch fired, but the true diagnosis never entered the pool. |
| `test:30358` | Ebola | Acute dystonic reactions | Branch fired, but graph/Bayes/pseudo candidates still missed Ebola. |

The dominant resolver failure is acute versus chronic rhinosinusitis:

| True pathology | Cases | Final behavior |
|---|---:|---|
| Acute rhinosinusitis | 4 | All four had the true label in the pool, but all four resolved as Chronic rhinosinusitis. |

Other resolver misses include Acute laryngitis, Atrial fibrillation, Croup, Epiglottitis, Ebola, and Pneumonia. These are mostly close-family or high-plausibility near-neighbor choices where the correct diagnosis is present but scored below the wrong diagnosis.

The single final regression versus the base branch was:

| Case | True pathology | Base prediction | Final prediction | Failure type |
|---|---|---|---|---|
| `test:113762` | Ebola | Ebola | URTI | Pseudo-candidate resolver override despite correct base answer. |

This suggests the next frozen resolver needs stronger base protection for rare/high-risk base answers unless graph/Bayes/MLP evidence strongly contradicts them.

## What This Run Shows

Notebook `38` is much more encouraging than Notebook `37`.

- Accuracy rises from the same-run base `172/196` to final `184/196`.
- Top-3/top-5 reach `194/196`, matching candidate-pool recall.
- Candidate-pool recall recovers from Notebook `37`'s `92/98` to `194/196`.
- Branch 2/3 do occur naturally under a more sensitive trigger.
- The main remaining bottleneck is final resolver discrimination, not broad candidate discovery.

What went wrong:

- The resolver is still overconfident on some close confounders, especially Acute rhinosinusitis versus Chronic rhinosinusitis.
- The branch trigger does not flag several resolver failures because the base terminal state looks confident.
- Extra branches can be expensive: the mean total request cost is acceptable, but the p90 and max are too high for a final efficiency claim.
- One base-correct Ebola case was overwritten by a pseudo-candidate, so base protection is still not conservative enough.

## Recommended Next Step

Do not claim Notebook `38` as the final result. Use it to freeze Notebook `39`.

Recommended frozen-confirmation direction:

1. Keep the Notebook `38` candidate-pool recipe because it restores `194/196` recall.
2. Use a more sensitive branch trigger than Notebook `37`; the calibration table suggests testing around `0.05` to `0.10`, but the exact value must be frozen before the next run.
3. Add a resolver-margin guard so low-margin final decisions receive targeted discriminator evidence rather than full additional branches.
4. Add stronger base protection for rare/high-risk base answers, especially when the final resolver wants to replace a correct-looking base with a pseudo-only candidate.
5. Add a specific acute/chronic rhinosinusitis discriminator rule based on train-derived duration/onset roots, because all four acute rhinosinusitis misses are truth-in-pool resolver failures.
6. Run a separate held-out confirmation cohort, ideally another balanced `98` or `196` cases not used in this calibration.

## Artifact Contract

Key result files:

- `metrics.json`
- `metrics_final.json`
- `topk_summary.csv`
- `summary_metrics.csv`
- `predictions.csv`
- `adaptive_live_final_predictions.csv`
- `candidate_level_live_scores.csv`
- `candidate_level_live_resolver_scores.csv`
- `candidate_pool_topk_rankings.csv`
- `adaptive_branch_decision_trace.csv`
- `close_confounder_discriminator_trace.csv`
- `live_calibration_paired_outcomes.csv`
- `live_calibration_truth_rank_analysis.csv`
- `live_calibration_failure_modes.csv`
- `live_calibration_branch_trigger_threshold_sweep.csv`
- `live_calibration_resolver_margin_sweep.csv`
- `live_calibration_candidate_source_recall.csv`
- `live_calibration_post_run_analysis_summary.json`
- `live_calibration_post_run_case_outcomes.csv`
- `selected_live_calibration_policy.json`

Figures are under:

```text
artifacts/trajectory_replicates/adaptive_value_branching_live_calibration196_v1/figures/
```

## Verification Completed

- Notebook `38` code cells static-parsed before the live run.
- Notebook-equivalent script generated at `scripts/live_adaptive_branching_calibration_cohort_nb38.py`.
- `python3 -m py_compile scripts/live_adaptive_branching_calibration_cohort_nb38.py` passed.
- Post-run artifacts were inspected and summarized in `live_calibration_post_run_analysis_summary.json`.

