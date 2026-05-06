# Integrated Evidence-Budget Comparison Report

Last updated: 2026-05-06

## Purpose

This report explains the updated notebook `09` comparison after adding evidence-budget views.

The earlier plots were mostly indexed by lambda, which is useful for understanding the policy setting but less direct for the scientific question:

> How much diagnostic performance do we get for the actual number of evidence fields acquired?

Notebook `09` now keeps the original lambda-based plots and adds request/evidence-count views.

## Updated Notebook

Notebook:

- `notebooks/09_matched_evidence_integrated_comparison.ipynb`

Integrated comparison artifact:

- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/`

New/updated outputs:

- `integrated_summary.csv`
- `evidence_budget_summary.csv`
- `evidence_efficiency_frontier.csv`
- `case_outcome_matrix.csv`
- `figures/integrated_accuracy_comparison.png`
- `figures/integrated_top3_comparison.png`
- `figures/integrated_ranking_comparison.png`
- `figures/accuracy_vs_mean_requests_integrated.png`
- `figures/top5_vs_mean_requests_integrated.png`
- `figures/accuracy_vs_revealed_roots_integrated.png`
- `figures/top5_vs_revealed_roots_integrated.png`
- `figures/evidence_usage_by_policy_setting.png`
- `figures/full_evidence_gain_recovered.png`
- `figures/hybrid_mlp_belief_signals.png`

## Main Evidence-Budget Result

Hybrid v1 has three useful settings:

| Lambda | Mean requests | Mean visible roots incl. initial | Hybrid acc | LLM-final acc | Online MLP acc | Offline matched MLP acc | Full-evidence acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 9.67 | 10.67 | 0.833 | 0.875 | 0.833 | 0.833 | 1.000 |
| 0.22 | 7.46 | 8.46 | 0.875 | 0.875 | 0.875 | 0.875 | 1.000 |
| 0.35 | 5.88 | 6.88 | 0.833 | 0.833 | 0.833 | 0.833 | 1.000 |

Correct-count interpretation on 24 cases:

| Lambda | Hybrid correct | LLM-final correct | Online MLP correct | Offline matched MLP correct |
|---:|---:|---:|---:|---:|
| 0.10 | 20/24 | 21/24 | 20/24 | 20/24 |
| 0.22 | 21/24 | 21/24 | 21/24 | 21/24 |
| 0.35 | 20/24 | 20/24 | 20/24 | 20/24 |

Because the slice has only 24 cases, one case changes accuracy by `0.0417`. Small visual differences can therefore look flat or step-like.

## Ranking Quality

Top-5 ranking quality is strong for the MLP-based heads:

| Lambda | LLM top-5 | Online MLP top-5 | Hybrid top-5 | Offline matched MLP top-5 | Full-evidence top-5 |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.22 | 0.917 | 0.958 | 0.958 | 0.958 | 1.000 |
| 0.35 | 0.875 | 0.917 | 0.917 | 0.917 | 1.000 |

This supports the user’s observation: for top-5, the online MLP final, hybrid final, and offline matched MLP are essentially moving together. The MLP-based heads preserve ranking quality while the policy acquires fewer fields at larger lambdas.

## Comparison To The Earlier Simple Hybrid

The earlier simple hybrid was:

```text
notebook 08 LLM policy gathers evidence
offline partial-evidence MLP diagnoses from that exact evidence
```

The new hybrid v1 is:

```text
MLP runs during the episode
MLP feedback influences stopping
the final heads diagnose from the evidence acquired by this earlier-stopping policy
```

This distinction matters because the MLP is not always receiving the same evidence.

Comparison against the previous notebook `08` matched-MLP result:

| Lambda | Old matched MLP acc | Old mean requests | New hybrid matched/MLP acc | New mean requests | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.10 | 0.875 | 13.04 | 0.833 | 9.67 | Lost one case after stopping earlier |
| 0.22 | 0.875 | 10.67 | 0.875 | 7.46 | Same accuracy with about 30% fewer requests |
| 0.35 | 0.833 | 8.33 | 0.833 | 5.88 | Same accuracy with fewer requests |

So the best interpretation is:

- yes, at `lambda = 0.22` and `lambda = 0.35`, the hybrid run maintains the previous matched-MLP accuracy with fewer evidence requests
- no, it does not preserve the previous matched-MLP accuracy at every setting
- the loss at `lambda = 0.10` is one case, caused by earlier/different evidence acquisition

## Why The New Graphs Matter

Lambda is a controller setting. It does not directly tell us how much evidence was used.

The new request/evidence-root graphs answer a clearer question:

> For a given number of requested evidence fields, which final head gives the best diagnostic performance?

The evidence-budget plots are now more useful for the research claim because they show the practical tradeoff:

- lower lambda usually asks more questions
- higher lambda asks fewer questions
- performance remains high until the policy stops too early
- hybrid v1’s current useful point is `lambda = 0.22`

## Interpretation Of Current Hybrid V1

Hybrid v1 is currently best framed as an evidence-efficiency improvement, not a final-diagnosis improvement.

Supported:

- hybrid v1 can preserve useful accuracy with fewer evidence requests
- `lambda = 0.22` is the strongest evidence-efficiency point so far
- the MLP-based heads maintain strong top-5 ranking quality
- actual evidence-count plots make the tradeoff clearer than lambda-only plots

Not yet supported:

- hybrid adjudication is not better than the individual LLM/MLP heads
- MLP confidence is not calibrated enough to safely override the LLM in disagreements
- the current result is not large enough for a final statistical claim

## Practical Next Step

The next change should not be another broad lambda sweep.

Recommended patch:

- keep online MLP feedback in the prompt and stop policy
- keep `lambda = 0.22` as the main useful setting
- make final adjudication more conservative
- do not allow high-confidence MLP to override a correct-looking LLM disagreement by default
- rerun the same 24-case slice at `lambda = 0.22`

If that preserves `21/24` correct with around `7.5` requests, then the next reasonable live run is a 49-case balanced pilot.

## Bottom Line

The updated notebook `09` now supports the clearer claim:

> Hybrid v1 can maintain the same diagnostic accuracy as the earlier matched-MLP setup at selected lambdas while using fewer requested evidence fields. The benefit is currently evidence efficiency, not better final diagnosis.

## Subsequent Notebook 13 Confirmation

Notebook `13` was run after this integrated comparison report. It is not a lambda sweep and is not yet folded into notebook `09`, but it directly confirms the selected stop-policy direction:

| System | Accuracy | Mean requests |
|---|---:|---:|
| Notebook `08`, lambda `0.10` | 0.917 | 13.04 |
| Notebook `11`, lambda `0.22` | 0.875 | 7.46 |
| Notebook `12`, offline selected stop | 0.917 | 6.875 |
| Notebook `13`, live selected stop | 0.917 | 6.58 |

This makes the evidence-budget story stronger: the selected online MLP stop signal preserves the high-accuracy notebook `08` result while using fewer requested fields than both notebook `08` and notebook `11`.

If notebook `09` is updated again, it should include notebook `13` as the current live selected-stop point rather than treating notebook `11` lambda `0.22` as the best hybrid evidence-efficiency result.
