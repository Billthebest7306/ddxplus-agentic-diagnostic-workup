# Live MedKGI Graph Shortlist Pilot

Last updated: 2026-05-07

## Summary

Notebook `17` implements the first live algorithmic-ledger pilot:

- notebook: `notebooks/17_live_medkgi_graph_shortlist_pilot.ipynb`
- dry-run artifact: `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_dryrun_smoke_v1/`
- live pilot artifact: `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/`
- deferred 49-case artifact path: `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_final49_v1/`

The method keeps Notebook `13`'s proven hybrid loop and MLP stop rule, but replaces the evidence shortlist with a MedKGI-style graph information-gain shortlist.

The 24-case live pilot is complete. Result: Notebook `17` is not promoted. It reduced mean requests slightly, but accuracy fell from Notebook `13`'s `22/24` to `20/24`.

## What Changed From Notebook 13

Unchanged:

- single-agent LLM loop
- `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- max request cap `24`
- partial-evidence MLP feedback
- selected MLP stop rule: confidence `>=0.70`, margin `>=0.20`, entropy `<=0.10`, min requests `>=1`
- final heads: LLM, MLP, agreement hybrid, conservative hybrid

Changed:

- Notebook `13` deterministic shortlist is replaced by a graph shortlist.
- The graph shortlist uses train-derived DDXPlus outcome statistics from Notebook `16`.
- Candidate evidence roots are scored by:

```text
score = penalty * (0.80 * information_gain + 0.15 * split_balance + 0.05 * global_mi)
```

The LLM still chooses the final request, but only from the graph top-10 legal fields.

## Fairness Boundaries

Allowed inside live policy:

- train-derived graph statistics
- currently visible ledger evidence
- current partial-evidence MLP belief
- previous LLM differential from the same episode
- deterministic diagnosis state and initial one-shot prior

Forbidden inside live policy:

- current test label
- current test differential diagnosis
- unrevealed test evidence
- full-evidence predictions
- `release_conditions.json` symptom metadata for graph scoring

## Dry-Run Validation

The notebook was executed in safe dry-run mode:

```python
RUN_LIVE_API = False
ALLOW_DRY_RUN_BENCHMARK = True
RUN_SCOPE = "pilot24"
DRY_RUN_MAX_CASES = 2
```

The dry run wrote all expected core artifacts:

- `metrics.json`
- `predictions.csv`
- `traces.jsonl`
- `raw_api_responses.jsonl`
- `resolved_run_config.json`
- `reference_comparison.csv`
- `graph_request_quality.csv`
- `requested_evidence_frequency.csv`
- `hard_case_trace_summary.csv`
- `promotion_decision.json`
- figures under `figures/`

The dry-run metrics are not scientific results. The promotion decision is explicitly marked:

```text
dry_run_smoke_not_for_promotion
```

## Live Pilot Result

Live configuration:

- `RUN_LIVE_API = True`
- `ALLOW_DRY_RUN_BENCHMARK = False`
- `RUN_SCOPE = "pilot24"`
- `LLM_MODEL = "gpt-4.1-mini"`
- `temperature = 0.0`
- `top_p = 1.0`
- max request cap `24`

Artifact:

- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/`

Core comparison:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Median requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 | 4.50 |
| Notebook `08` lambda `0.10` LLM-only | 22/24 | 0.917 | 0.917 | 0.917 | 0.846 | 13.04 | n/a |
| Notebook `12` offline selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 | n/a |
| Notebook `17` graph shortlist pilot | 20/24 | 0.833 | 0.833 | 0.875 | 0.744 | 6.21 | 5.00 |

Notebook `17` graph-specific metrics:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 1.76 |
| Mean requested information gain | 0.373 |
| Requests outside graph top-10 | 0 |
| Mean graph shortlist size | 10.0 |
| Stop-before-cap rate | 0.958 |
| Cap-hit count | 1 |
| LLM/MLP agreement rate | 1.000 |

Stop reasons:

| Stop reason | Cases |
|---|---:|
| Agent stop | 13 |
| Selected MLP stop | 10 |
| Max requests reached | 1 |

Promotion decision:

```text
reject_keep_notebook13_v1
```

Reason:

- accuracy dropped by two cases versus Notebook `13`
- mean requests improved only slightly, from `6.58` to `6.21`
- top-5 also fell from `0.917` to `0.875`
- no case was fixed that Notebook `13` got wrong

## Paired Case Analysis

Paired artifact:

- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/notebook13_vs_graph_paired_case_results.csv`

Paired outcomes:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| Notebook `13` only correct | 2 |
| Notebook `17` only correct | 0 |
| Both wrong | 2 |

Notebook `17` introduced two additional errors:

| Case | True pathology | Notebook `13` | Notebook `17` | N13 requests | N17 requests |
|---|---|---|---|---:|---:|
| `test:51421` | Chagas | Chagas | Sarcoidosis | 23 | 17 |
| `test:77908` | Ebola | Ebola | HIV (initial infection) | 6 | 2 |

Persistent failures:

| Case | True pathology | Notebook `13` | Notebook `17` | N13 requests | N17 requests |
|---|---|---|---|---:|---:|
| `test:81691` | Croup | Chagas | Acute otitis media | 23 | 23 |
| `test:62878` | Pericarditis | Anemia | Anemia | 16 | 24 |

Largest request savings that stayed correct:

| Case | True pathology | N13 requests | N17 requests | Delta |
|---|---|---:|---:|---:|
| `test:88250` | Allergic sinusitis | 18 | 7 | -11 |
| `test:95923` | Pneumonia | 9 | 5 | -4 |
| `test:34242` | Guillain-Barre syndrome | 3 | 1 | -2 |

Largest request increases:

| Case | True pathology | N13 requests | N17 requests | Delta | Outcome |
|---|---|---:|---:|---:|---|
| `test:62878` | Pericarditis | 16 | 24 | +8 | both wrong |
| `test:8666` | Influenza | 6 | 13 | +7 | both correct |

## Interpretation

Notebook `17` is an informative negative result. The graph shortlist is mechanically doing what it was designed to do: requested fields have strong graph ranks, no request falls outside the graph top-10, and the requested evidence has high estimated information gain. However, that did not translate into better diagnosis.

The likely reason is that hard graph replacement is too restrictive. The graph scores are derived from the current active differential, which itself depends on the partial-evidence MLP and prior LLM state. If that active differential is already missing the correct disease or over-weighting a wrong disease family, a pure graph top-10 shortlist can efficiently ask high-scoring questions for the wrong diagnostic neighborhood.

This is why Notebook `17` can look graph-efficient while still losing accuracy. The graph controller improved local evidence-score quality, but it did not improve global trajectory quality.

The result supports a more cautious graph role:

- do not replace Notebook `13`'s shortlist with pure graph top-10
- use graph scores as an advisory or blending signal
- preserve Notebook `13`'s broader deterministic/LLM shortlist diversity
- add rare/critical-disease preservation guards before graph pruning

## Next Recommendation

Do not run `final49` for Notebook `17` v1. The 24-case pilot is already clearly worse than Notebook `13` on the promotion rule.

Keep Notebook `13` as the frozen proposed method.

If graph work continues, the next version should be a hybrid graph-advisory shortlist rather than a graph-replacement shortlist:

```text
Notebook 13 shortlist diversity
+ graph top-ranked discriminative fields
+ rare/critical disease safety roots
+ guard against pruning diseases absent from the current active top-k
```

This would let the graph ledger improve question quality without allowing it to narrow the action space too aggressively.

## How To Run Live

For the 24-case pilot:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
RUN_SCOPE = "pilot24"
```

For the 49-case confirmation, only after the pilot is not clearly worse:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
RUN_SCOPE = "final49"
```

Current status: do not run `final49` for this v1 graph shortlist, because `pilot24` was clearly worse than Notebook `13`.

API key options:

- set `LLM_API_KEY` in the environment before launching Jupyter
- or set `INTERACTIVE_API_KEY_BOOTSTRAP = True` in the first notebook cell and paste the key when prompted

## What To Look For In Any Future Graph Run

Primary comparison:

- Notebook `17` vs matching Notebook `13` scope

For `pilot24`, compare against:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

For `final49`, compare against:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`

Promotion rule:

- promote if accuracy is at least Notebook `13` and mean requests are not higher
- conditionally review if accuracy drops by only one case but hard-case trajectory quality improves
- reject if accuracy drops by more than one case or requests rise without hard-case benefit

Important graph-specific outputs:

- `mean_requested_graph_rank`
- `mean_requested_information_gain`
- `requests_outside_graph_top10`
- `top_graph_score_at_stop`
- `graph_request_quality.csv`

These answer whether graph shortlisting actually improves evidence trajectory quality.
