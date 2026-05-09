# Notebook 20 Plan: LLM-Led Workup With Graph Ledger Context

Created: 2026-05-08

## Summary

Create:

```text
notebooks/20_llm_led_graph_ledger_context.ipynb
```

This notebook should adapt Notebook `13`, not Notebook `17`, `18`, or `19`.

Core change:

```text
Notebook 13 LLM-led loop
+ graph-derived evidence-state context in the prompt
```

No controller replacement. No hard graph shortlist. No Bayesian VOI replacement policy.

## Primary Question

Can graph-structured ledger context help the LLM make better question choices and maintain better diagnostic coherence while preserving Notebook `13`'s strong stop policy?

## Fixed Baseline To Beat

Notebook `13`, 49-case confirmation:

| Correct | Accuracy | Top-5 | Mean requests |
|---:|---:|---:|---:|
| 43/49 | 0.878 | 0.939 | 6.59 |

Pilot reference:

| Correct | Accuracy | Top-5 | Mean requests |
|---:|---:|---:|---:|
| 22/24 | 0.917 | 0.917 | 6.58 |

## What To Reuse

From Notebook `13`:

- DDXPlus loading/parsing
- case episode compiler
- deterministic evidence ledger
- decoded evidence display
- legal request validation
- base action menu/shortlist
- OpenAI-compatible API adapter
- interactive API key bootstrap
- strict JSON schema
- partial-evidence MLP feedback
- selected MLP stop rule
- artifact layout conventions

From Notebook `16`/`19`:

- train-derived disease/evidence statistics
- root log-odds support
- root mutual information / reliability
- support/contradiction calculations
- rare evidence flags

But do not reuse their controller logic.

## Graph Ledger Context Compiler

Implement a local class:

```python
class GraphLedgerContextCompiler:
    def update(observed_evidence, llm_differential, mlp_probs, prior_probs) -> GraphLedgerContext:
        ...
```

Outputs:

```python
{
    "active_differential": [...],
    "diagnosis_support": [...],
    "diagnosis_contradictions": [...],
    "unresolved_pairs": [...],
    "missing_discriminator_advice": [...],
    "consistency_warnings": [...],
}
```

Keep this compact. The prompt should not become a graph dump.

## Prompt Additions

Add one section to the Notebook `13` prompt:

```text
GRAPH LEDGER CONTEXT

This is advisory context from the structured evidence ledger.
It summarizes support, contradiction, and unresolved diagnosis pairs.
Use it to reason, but you remain responsible for choosing the next evidence request.

[compact JSON block]
```

The LLM should still output the same schema:

```json
{
  "decision": "request" | "stop",
  "requested_evidence_id": "...",
  "predicted_pathology": "...",
  "ranked_differential": ["...", "..."],
  "confidence": 0.0,
  "brief_reasoning": "..."
}
```

## Artifact Root

Use:

```text
artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_<scope>_v1/
```

Scopes:

- `pilot24`
- `final49`

Safe default:

```python
RUN_LIVE_API = False
RUN_SCOPE = "pilot24"
DRY_RUN_MAX_CASES = 2
```

## Outputs

Required:

- `metrics.json`
- `predictions.csv`
- `traces.jsonl`
- `raw_api_responses.jsonl`
- `resolved_run_config.json`
- `reference_comparison.csv`
- `graph_context_by_turn.csv`
- `warning_resolution_summary.csv`
- `hard_case_graph_context_audit.json`
- `promotion_decision.json`
- figures under `figures/`

Additional columns in predictions:

- `num_graph_warnings_final`
- `num_unresolved_pairs_final`
- `final_top_diagnosis_support`
- `final_top_diagnosis_contradiction`
- `warnings_resolved_count`
- `graph_context_tokens_estimate`

## Promotion Rule

For `pilot24`, promote to final49 if:

- `correct_count >= 22` and `mean_requests <= 7.0`, or
- `correct_count >= 23` and `mean_requests <= 8.0`, or
- same accuracy as Notebook `13` while improving hard-case trace quality without new obvious failure modes

For `final49`, promote over Notebook `13` if:

- `correct_count >= 44`, or
- `correct_count == 43` and mean requests decrease, or
- `correct_count == 43` and graph-context traces are materially more coherent on hard cases

Do not promote if:

- accuracy drops below Notebook `13` by more than one case
- mean requests increase without hard-case improvement
- the graph context causes the LLM to over-focus on generic evidence

## Key Difference From Failed Notebooks

Failed replacement pattern:

```text
graph/Bayes computes top evidence
-> LLM forced into graph/Bayes-shaped action space
```

Notebook `20` pattern:

```text
graph computes structured understanding
-> LLM sees better case state
-> LLM still chooses the question
```

This directly addresses the user's intended architecture.

## Expected Cost

Pilot24 should cost roughly similar to Notebook `13` 24-case run, possibly slightly higher due to prompt context.

Cost control:

- keep graph context compact
- cap graph block length
- do not include all evidence roots
- do not include raw matrices
- summarize only active differential and top unresolved pairs

## Recommended First Run

Run dry smoke first:

```python
RUN_LIVE_API = False
ALLOW_DRY_RUN_BENCHMARK = True
RUN_SCOPE = "pilot24"
DRY_RUN_MAX_CASES = 2
```

Then live pilot:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
RUN_SCOPE = "pilot24"
SEQUENTIAL_MAX_CASES = 24
```

Only run `final49` if the 24-case pilot is not worse than Notebook `13`.
