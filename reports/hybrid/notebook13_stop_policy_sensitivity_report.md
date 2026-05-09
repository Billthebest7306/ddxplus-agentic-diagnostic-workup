# Notebook 15: Notebook 13 Stop-Policy Sensitivity And Evidence-Trajectory Diagnostics

Last updated: 2026-05-06

## Purpose

Notebook `15` was added to answer a specific concern: whether Notebook `13`'s selected MLP-guided stopping rule is actually meaningful, or whether different MLP confidence/margin/entropy thresholds could produce a better accuracy/request tradeoff.

This is an offline analysis. It makes no API calls. It parses the final Notebook `13` 49-case live traces and replays possible stopping thresholds over the already-observed turn states.

Notebook:

- `notebooks/15_notebook13_stop_policy_sensitivity.ipynb`

Artifact root:

- `artifacts/stop_policy_sensitivity/notebook13_49case_v1/`

## Inputs

Source run:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`

Parsed data:

| Item | Count |
|---|---:|
| Cases | 49 |
| Turn states | 372 |
| Requested evidence rows | 323 |

The notebook uses `predictions.csv` as the authoritative final output for observed final states, because a small number of final no-reveal trace rows differ from the saved final ranked differential. Earlier turn states are still taken from the trace and used for early-stop counterfactuals.

## Current Notebook 13 Rule

Notebook `13` uses:

| Threshold | Value |
|---|---:|
| minimum requests | 1 |
| MLP confidence | `>= 0.70` |
| MLP margin | `>= 0.20` |
| MLP entropy | `<= 0.10` |

Recovered current-rule metrics from Notebook `15`:

| Metric | Value |
|---|---:|
| Accuracy | 43/49 = 0.878 |
| Top-3 | 0.918 |
| Top-5 | 0.939 |
| Macro-F1 | 0.845 |
| Mean requests | 6.59 |
| Median requests | 5.0 |
| Threshold-fired rate | 36/49 = 0.735 |
| Observed-final fallback cases | 13 |

This matches the official Notebook `13` final result.

## Threshold Sweep Finding

Notebook `15` swept confidence, margin, entropy, minimum-request, and final-head settings.

Best offline tie under the current request budget:

| Policy | Final head | Accuracy | Top-5 | Mean requests |
|---|---|---:|---:|---:|
| Current Notebook `13` rule | hybrid | 0.878 | 0.939 | 6.59 |
| Best offline tie | LLM/hybrid | 0.878 | 0.939 | 6.55 |

The selected offline tie was:

```text
min_requests = 0
confidence >= 0.55
margin >= 0.10
entropy <= 0.05
```

This is not a meaningful improvement. It saves only about `0.04` mean requests and does not fix any incorrect case. It mainly stops two already-easy correct cases one turn earlier.

Conclusion:

> Notebook `13`'s stop rule is already near the offline frontier on the observed 49-case traces. Threshold tweaking alone is unlikely to materially improve the result.

## Evidence Request Findings

Most requested evidence fields:

| Evidence | Count | Interpretation |
|---|---:|---|
| `E_129`, skin lesions/redness/problems | 21 | broad systemic/infectious/allergic discriminator |
| `E_151`, swelling | 20 | broad edema/inflammation discriminator |
| `E_201`, cough | 14 | respiratory discriminator |
| `E_79`, smoking | 13 | respiratory/cardiac risk discriminator |
| `E_91`, fever | 11 | infection/inflammation discriminator |
| `E_181`, nasal congestion/runny nose | 10 | URI/sinus discriminator |
| `E_214`, wheezing | 10 | asthma/bronchospasm/respiratory discriminator |
| `E_41`, similar-symptom contact | 10 | infectious exposure discriminator |

This supports the interpretation that the LLM is often asking broadly plausible clinical discriminators, but some requests are still generic rather than tightly disease-separating.

## Request Allocation

Request usage by final correctness:

| Final correctness | Cases | Mean requests | Median requests | Request range |
|---|---:|---:|---:|---|
| Incorrect | 6 | 11.83 | 11.5 | 2-24 |
| Correct | 43 | 5.86 | 5.0 | 0-23 |

This is important: incorrect cases generally used **more** evidence, not less. Therefore, most remaining errors are not simply caused by too few requests.

The six final errors split into types:

| Case | True pathology | Prediction | Requests | Interpretation |
|---|---|---|---:|---|
| `test:125508` | Unstable angina | Anemia | 2 | early confident wrong stop |
| `test:8666` | Influenza | HIV initial infection | 3 | early/mid wrong stop; true label still in top-5 |
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | 8 | close-label confusion |
| `test:62878` | Pericarditis | Anemia | 15 | long wrong trajectory |
| `test:81691` | Croup | Anemia | 19 | long wrong trajectory; true label recovered in final top-5 |
| `test:38475` | Acute COPD exacerbation/infection | Myocarditis | 24 | cap hit; persistent uncertainty/wrong trajectory |

## Prediction Transition Findings

Turn-level prediction transitions:

| Head | Wrong-to-correct | Correct-to-wrong | Stable correct | Stable wrong |
|---|---:|---:|---:|---:|
| LLM | 40 | 11 | 79 | 193 |
| Hybrid | 40 | 11 | 79 | 193 |
| MLP | 38 | 10 | 66 | 209 |

Interpretation:

- Additional evidence often helps: wrong-to-correct transitions are much more common than correct-to-wrong transitions.
- Drift exists: there are still 11 LLM/hybrid correct-to-wrong transitions.
- The larger issue is stable wrong belief: many turn transitions remain wrong-to-wrong.

This explains why mean request count alone is not enough. More evidence can help, but if the trajectory is already focused on the wrong disease family, extra requests do not necessarily recover the correct diagnosis.

## What This Means For The Project

Notebook `15` makes the current position clearer:

1. The selected Notebook `13` stop policy is not arbitrary. It sits near the observed offline accuracy/request frontier.
2. Simple threshold tuning is not the next high-value improvement.
3. The remaining bottleneck is not primarily stopping. It is wrong belief convergence and evidence trajectory quality.
4. A more conservative stop policy cannot be evaluated fully offline because once the original live trace stopped, there is no future evidence to replay.

Recommended next technical direction:

- keep Notebook `13` as the frozen proposed method
- do not replace the stop rule based on this offline sweep
- if improving further, focus on contradiction handling, hard-case trajectory repair, or better question selection for persistent failures
- only run a new live policy if it changes the evidence trajectory, not merely because thresholds were adjusted

## Key Artifacts

- `turn_level_states.csv`
- `requested_evidence_long.csv`
- `requested_evidence_frequency.csv`
- `case_request_outcomes.csv`
- `prediction_transition_summary.csv`
- `threshold_sweep_summary.csv`
- `same_or_better_than_current_thresholds.csv`
- `threshold_policy_case_results.csv.gz`
- `selected_vs_current_case_comparison.csv`
- `analysis_summary.json`

Key figures:

- `figures/requested_evidence_frequency.png`
- `figures/request_count_by_correctness.png`
- `figures/per_case_request_allocation.png`
- `figures/prediction_transition_totals.png`
- `figures/hybrid_turn_correctness_raster.png`
- `figures/threshold_sweep_accuracy_vs_requests.png`
- `figures/threshold_sweep_top5_vs_requests.png`
- `figures/hard_case_mlp_confidence_progression.png`
- `figures/hard_case_mlp_entropy_progression.png`
