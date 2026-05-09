# Graph-Context Policy Lab Report

## Purpose

Notebook `21_graph_context_policy_lab.ipynb` is an offline experimental lab for the graph-ledger direction.

It was created after Notebook `20` showed a mixed result: lower top-1 accuracy than Notebook `13`, but better top-3/top-5 ranking quality. The lab tests whether graph context can act as a critic, guardrail, or adjudicator rather than as a question-selection controller.

## Inputs

Runs compared:

| Run | Artifact root |
|---|---|
| Notebook `13` selected-stop hybrid v1 | `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/` |
| Notebook `17` hard graph shortlist | `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/` |
| Notebook `18` graph-advisory shortlist | `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_v1/` |
| Notebook `20` LLM-led graph context | `artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1/` |

Notebook `21` makes no API calls and trains no model. It replays traces and uses existing graph statistics.

## Reference Metrics

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook `17` | 0.833 | 0.833 | 0.875 | 6.21 |
| Notebook `18` | 0.875 | 0.875 | 0.917 | 7.67 |
| Notebook `20` | 0.833 | 0.958 | 0.958 | 6.13 |

The important signal remains Notebook `20`'s ranking quality: it reaches `23/24` top-3 and top-5 while using slightly fewer requests than Notebook `13`.

## Main Experiments

Notebook `21` tested:

- graph feature diagnostics
- top-5 graph adjudicators
- LLM/MLP consensus adjudicators
- stop-guard replay/flagging
- drift guards based on stable graph-supported diagnoses
- oracle top-3/top-5 upper bounds

The non-oracle variants did **not** beat Notebook `13`.

Best experimental non-oracle result:

| Variant | Source | Accuracy | Top-5 | Mean requests | Notes |
|---|---|---:|---:|---:|---|
| `drift_guard_notebook13_tc1.0_delta0.5` | Notebook `13` trace | 0.917 | 0.917 | 6.58 | Changed one wrong case to another wrong case; no real improvement |

Best Notebook `20` graph-context lead:

| Variant | Accuracy | Top-5 | Mean requests | Notes |
|---|---:|---:|---:|---|
| `drift_guard_notebook20_tc1.0_delta0.5` | 0.875 | 0.958 | 6.13 | Fixes Chagas vs Notebook `20`, but still below Notebook `13` and has one regression vs Notebook `13` |

Oracle upper bound:

| Variant | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `20` oracle top-3 | 0.958 | 0.958 | 0.958 | 6.13 |
| Notebook `20` oracle top-5 | 0.958 | 0.958 | 0.958 | 6.13 |

This means the Notebook `20` trace contains enough ranking information to reach `23/24`, but the current non-oracle graph rules cannot reliably select the right top-1.

## Diagnostic Findings

For Notebook `20`, wrong cases had much higher graph contradiction on the final top diagnosis:

| Feature | Correct mean | Wrong mean | Wrong - correct |
|---|---:|---:|---:|
| Top contradiction | 0.377 | 4.015 | +3.638 |
| Top contradiction minus support | -6.964 | 0.906 | +7.870 |
| Top net support | 6.964 | -0.906 | -7.870 |

Interpretation:

- Graph contradiction is a strong warning signal.
- The graph can often tell that the final top-1 is suspect.
- However, choosing the correct alternative from the top-5 remains hard.

Hard-case interpretation:

- `Croup`: Notebook `20` recovered Croup into rank 3 with fewer requests, but graph/drift rules could not safely promote it to top-1.
- `Chagas`: drift guard can recover Chagas from Notebook `20`'s wrong Sarcoidosis prediction.
- `Influenza`: Notebook `20` ranked Influenza second, but the graph rules did not confidently promote it.
- `Pericarditis`: remains the hardest case; graph variants continue to choose wrong alternatives.

## Stop-Guard Result

The stop-guard replay flagged many stops as unsafe, but most were not evaluable because the recorded trace ended at the actual stop. This means the lab can identify likely unsafe stops, but cannot prove what would happen after continuing without a live run.

This is important: stop guards are plausible, but offline replay cannot fully validate them when there is no future trace after the stop.

## Selection Decision

`selected_live_candidate.json` status:

```text
no_promotable_candidate
```

Reason:

- no non-oracle variant beat Notebook `13`
- no non-oracle variant improved a Notebook `13` miss without introducing comparable risk
- Notebook `20` top-5 signal is promising, but current graph rules are not enough to convert it into top-1

Recommendation:

- do not promote Notebook `21` variants directly into a live Notebook `22`
- keep Notebook `13` as the frozen proposed method
- if continuing graph work, focus on a learned or calibrated adjudicator rather than hand-threshold graph rules

## Artifacts

Artifact root:

```text
artifacts/graph_algorithmic_ledger/graph_context_policy_lab_24case_v1/
```

Important outputs:

- `reference_metric_check.csv`
- `turn_level_trace_features.csv`
- `graph_feature_diagnostics.csv`
- `policy_variant_summary.csv`
- `case_level_variant_results.csv`
- `variant_frontier.csv`
- `hard_case_rank_trajectories.csv`
- `hard_case_audits.json`
- `selected_live_candidate.json`
- figures under `figures/`

## Scientific Meaning

Notebook `21` strengthens the research story even though it does not produce a promoted method.

It shows:

- graph context has real diagnostic signal
- Notebook `20`'s top-5 improvement is not random noise
- graph contradiction is strongly associated with wrong final predictions
- hand-written graph adjudication rules are too brittle to safely replace or override the current final answer

This supports the next research direction: graph ledger as a calibrated critic or learned adjudication feature, not a simple threshold-based final decision rule.
