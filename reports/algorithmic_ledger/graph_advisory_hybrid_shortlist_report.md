# Graph-Advisory Hybrid Shortlist

Last updated: 2026-05-08

## Summary

Notebook `18` implements the successor to Notebook `17`:

- notebook: `notebooks/18_graph_advisory_hybrid_shortlist.ipynb`
- dry-run artifact: `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_dryrun_smoke_v1/`
- live pilot artifact: `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_v1/`

Notebook `17` used a hard MedKGI-style graph top-10 replacement shortlist and was rejected: `20/24` accuracy versus Notebook `13`'s `22/24`. Notebook `18` keeps Notebook `13`'s broader shortlist diversity and adds graph evidence as an advisory signal.

## Method

Unchanged from Notebook `13`:

- single-agent LLM controller
- `gpt-4.1-mini`
- deterministic settings: `temperature = 0.0`, `top_p = 1.0`
- selected MLP stop rule from Notebook `12`
- deterministic evidence ledger
- final heads: LLM, MLP, agreement hybrid, conservative hybrid

Changed from Notebook `17`:

- graph no longer hard-replaces the shortlist
- candidate pool is a union of Notebook `13` base candidates, graph information-gain candidates, rare disease-specific support candidates, and safety-recovery candidates
- exposed shortlist size is `12`
- agent stop can be conservatively overridden when the MLP is uncertain and high-value or rare evidence remains

Advisory score:

```text
score =
  0.45 * notebook13_score_norm
+ 0.25 * graph_information_gain_norm
+ 0.20 * disease_specific_support_norm
+ 0.10 * split_balance
```

Rare-support rule:

```text
rare_support_candidate if:
  active disease contains pathology with root outcome log_odds_support >= 5.0
  and root global_present_rate <= 0.05
  and candidate root is legal and unrevealed
```

This is intended to recover Notebook `17` failure modes like Ebola and Chagas, where decisive evidence fields had low global MI but very high disease-specific support.

## Dry-Run Validation

Dry-run configuration:

```python
RUN_LIVE_API = False
ALLOW_DRY_RUN_BENCHMARK = True
RUN_SCOPE = "pilot24"
DRY_RUN_MAX_CASES = 2
```

Validation result:

- static code-cell parse passed
- notebook executed top-to-bottom with `nbclient`
- no live API calls were made
- interactive API bootstrap defaults to `False`
- deterministic API settings are logged in `resolved_run_config.json`

Expected artifacts were written:

- `metrics.json`
- `predictions.csv`
- `traces.jsonl`
- `raw_api_responses.jsonl`
- `resolved_run_config.json`
- `reference_comparison.csv`
- `notebook13_vs_notebook18_paired_case_results.csv`
- `notebook17_vs_notebook18_paired_case_results.csv`
- `advisory_shortlist_components.csv`
- `rare_evidence_coverage.csv`
- `agent_stop_safety_checks.csv`
- `agent_stop_safety_overrides.csv`
- `requested_evidence_frequency.csv`
- `hard_case_trace_summary.csv`
- `promotion_decision.json`
- figures under `figures/`

Dry-run metrics are not scientific results. They only validate execution and artifact contracts.

## How To Run Live

For the 24-case pilot:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
RUN_SCOPE = "pilot24"
RESUME_IF_AVAILABLE = False
```

Use the same API key setup as the other live notebooks:

- set `LLM_API_KEY` before launching Jupyter, or
- set `INTERACTIVE_API_KEY_BOOTSTRAP = True` in the first notebook cell and paste the key when prompted

Do not run `final49` until `pilot24` meets the promotion rule.

## Promotion Rule

Promote Notebook `18` only if one of these holds:

- `correct_count >= 22` and `mean_requests <= 6.58`
- `correct_count >= 23` and `mean_requests <= 8.0`
- `correct_count == 22`, requests are slightly higher, and it fixes a persistent hard case without new failures

Reject if:

- `correct_count <= 21`
- it repeats Notebook `17`'s Chagas or Ebola failures
- it increases request count without clear hard-case benefit

## Live Pilot Result

Notebook `18` has now been run live on the same 24-case pilot slice used by Notebooks `13` and `17`.

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Median requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 | 4.5 |
| Notebook `17` hard graph shortlist | 20/24 | 0.833 | 0.833 | 0.875 | 0.744 | 6.21 | 5.0 |
| Notebook `18` graph-advisory shortlist | 21/24 | 0.875 | 0.875 | 0.917 | 0.795 | 7.67 | 5.0 |

Notebook `18` improved over Notebook `17`, but it did not meet the promotion rule against Notebook `13`.

## Notebook 18 Metrics

| Metric | Value |
|---|---:|
| Accuracy | 21/24 = 0.875 |
| Top-3 accuracy | 0.875 |
| Top-5 accuracy | 0.917 |
| Macro-F1 | 0.795 |
| Mean requests | 7.67 |
| Median requests | 5.0 |
| Stop-before-cap rate | 0.875 |
| Cap-hit count | 3 |
| Selected MLP stop-rule fired | 17/24 = 0.708 |
| Agent-stop count | 4 |
| LLM/MLP top-1 agreement | 20/24 = 0.833 |
| Safety-stop overrides | 8 |

Final-head comparison:

| Final head | Correct | Accuracy |
|---|---:|---:|
| LLM final | 21/24 | 0.875 |
| Online MLP final | 20/24 | 0.833 |
| Agreement hybrid final | 21/24 | 0.875 |
| Conservative hybrid final | 21/24 | 0.875 |

Graph/advisory diagnostics:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 5.12 |
| Mean requested information gain | 0.242 |
| Mean top graph score at stop | 0.796 |
| Mean shortlist size | 12.0 |
| Requests outside pure graph top-10 | 17 |

The `requests outside pure graph top-10` count is expected for Notebook `18`: the whole point of the advisory design is that Notebook `13` base candidates and rare-support candidates can remain available even when they are not in the hard graph top-10.

## Paired Results

Against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 21 |
| Notebook `13` only correct | 1 |
| Notebook `18` only correct | 0 |
| Both wrong | 2 |

Case changes against Notebook `13`:

| Case | True pathology | Notebook 13 | Notebook 18 | Request delta |
|---|---|---|---|---:|
| `test:81691` | Croup | Chagas, wrong | Spontaneous pneumothorax, wrong | +1 |
| `test:62878` | Pericarditis | Anemia, wrong | Whooping cough, wrong | +8 |
| `test:16097` | Stable angina | Stable angina, correct | Boerhaave, wrong | +15 |

Against Notebook `17`:

| Outcome | Cases |
|---|---:|
| Both correct | 19 |
| Notebook `18` only correct | 2 |
| Notebook `17` only correct | 1 |
| Both wrong | 2 |

Case changes against Notebook `17`:

| Case | True pathology | Notebook 17 | Notebook 18 | Request delta |
|---|---|---|---|---:|
| `test:51421` | Chagas | Sarcoidosis, wrong | Chagas, correct | -8 |
| `test:77908` | Ebola | HIV initial infection, wrong | Ebola, correct | +6 |
| `test:16097` | Stable angina | Stable angina, correct | Boerhaave, wrong | +14 |
| `test:81691` | Croup | Acute otitis media, wrong | Spontaneous pneumothorax, wrong | +1 |
| `test:62878` | Pericarditis | Anemia, wrong | Whooping cough, wrong | 0 |

## Hard-Case Interpretation

Notebook `18` did what it was designed to do in one important way: it recovered the two Notebook `17` graph-replacement failures on Chagas and Ebola. This supports the hypothesis that graph information should not be used as a hard replacement shortlist. Adding rare disease-specific support and restoring Notebook `13` shortlist diversity prevented the graph from pruning away useful rare evidence.

However, the advisory version still failed the promotion test because it introduced a new Stable angina failure and increased the mean request count. The three wrong cases all hit the `24` request cap, and none fired the selected MLP stop rule:

| Case | True pathology | Prediction | Requests | Stop reason | Final MLP confidence | Final MLP entropy |
|---|---|---|---:|---|---:|---:|
| `test:81691` | Croup | Spontaneous pneumothorax | 24 | max requests reached | 0.137 | 0.698 |
| `test:62878` | Pericarditis | Whooping cough | 24 | max requests reached | 0.639 | 0.238 |
| `test:16097` | Stable angina | Boerhaave | 24 | max requests reached | 0.557 | 0.460 |

This is a different failure mode from unsafe early stopping. The system did not stop too early; it kept asking until the cap and still failed to recover the right belief. The issue is wrong trajectory steering and noisy later evidence, not just stop threshold sensitivity.

The safety wrapper fired eight overrides across five cases. It prevented some early stops, but it also made the system more conservative and increased request usage. The Pericarditis case received a late safety override at turn `23`, but by then the trajectory was already wrong and the extra request did not recover the diagnosis.

## Promotion Decision

Notebook `18` should not be promoted.

Reason:

- it scored `21/24`, below the `22/24` minimum promotion threshold
- it used more evidence than Notebook `13`: `7.67` vs `6.58` mean requests
- it fixed Notebook `17`'s Chagas and Ebola failures, but introduced a new Stable angina failure
- it did not fix the persistent Croup or Pericarditis hard cases

The correct research decision is:

```text
reject_keep_notebook13_v1
```

Notebook `13` remains the frozen proposed method.

## Scientific Takeaway

Notebook `18` is not a failed experiment in the sense of being useless. It gives a sharper result:

> Graph information is useful as a diagnostic and recovery signal, but even advisory graph blending can increase trajectory complexity and does not yet improve the live policy over Notebook `13`.

Current graph conclusion:

- hard graph replacement is too restrictive
- advisory graph support recovers rare-disease failures better than hard graph replacement
- graph support still does not outperform the simpler Notebook `13` policy
- graph analysis is valuable for auditing missed evidence and hard cases
- graph control should not replace the current policy without a better mechanism for belief correction and contradiction handling

Recommended next action:

- keep Notebook `13` as the main proposed method
- present Notebook `17` and Notebook `18` as negative/diagnostic graph ablations
- do not spend API budget on a Notebook `18` `final49` run
- if graph work continues, make it an offline explanation/audit layer or develop a more principled contradiction-recovery mechanism before another live test
