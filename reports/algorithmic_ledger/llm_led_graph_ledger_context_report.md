# LLM-Led Graph-Ledger Context Report

## Purpose

Notebook `20_llm_led_graph_ledger_context.ipynb` implements the corrected algorithmic-ledger experiment.

The key correction is architectural: the graph ledger is no longer used as a replacement controller. The LLM remains the evidence-requesting agent, Notebook `13`'s legal action menu remains active, and Notebook `13`'s partial-evidence MLP stop rule remains unchanged. The graph ledger only adds structured context to the prompt.

## Why This Notebook Exists

Notebooks `17`, `18`, and `19` tested algorithmic graph/Bayesian ideas mostly by changing the evidence controller:

| Notebook | What changed | Result |
|---|---|---|
| `17` | Replaced Notebook `13` shortlist with hard graph top-10 | Rejected: `20/24`, worse than Notebook `13` |
| `18` | Blended graph advisory scoring into shortlist | Rejected: `21/24`, still worse than Notebook `13` |
| `19` | Offline Bayesian VOI replacement policy | Rejected: best fused `33/49`, far below Notebook `13` |

These were useful negative ablations, but they were not the intended algorithmic-ledger design. The intended design is:

```text
LLM chooses evidence
-> deterministic ledger reveals legal evidence
-> graph ledger interprets revealed evidence
-> graph context helps the LLM reason about support, contradiction, and unresolved pairs
-> partial-evidence MLP decides stop readiness
```

## Implementation

Notebook `20` was adapted from Notebook `13`, not from the rejected graph-controller notebooks.

Unchanged from Notebook `13`:

- `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- deterministic DDXPlus evidence ledger
- legal evidence-action menu
- existing deterministic shortlist
- selected MLP stop policy:
  - `min_requests >= 1`
  - `mlp_confidence >= 0.70`
  - `mlp_margin >= 0.20`
  - `mlp_entropy <= 0.10`
  - `mlp_stability >= 0`
- final heads:
  - LLM final
  - MLP final
  - agreement hybrid final
  - conservative hybrid final

New in Notebook `20`:

- `GraphLedgerContextCompiler`
- `GRAPH LEDGER CONTEXT` prompt block
- turn-level graph context artifact
- warning-resolution artifact
- hard-case graph audit artifact

The graph context contains:

- active differential from MLP, previous LLM differential, deterministic state, and one-shot prior
- per-diagnosis support scores from revealed evidence
- per-diagnosis contradiction scores from revealed evidence
- unresolved competing diagnosis pairs
- advisory discriminator fields drawn from the same current shortlist the LLM is allowed to request from
- consistency warnings when LLM, MLP, and graph support disagree

## Fairness Boundaries

Notebook `20` does not use:

- hidden test labels inside the policy
- hidden test differentials inside the policy
- full-evidence predictions inside the policy
- unrevealed patient evidence except when the environment reveals a requested root
- graph top-k replacement control
- Bayesian VOI replacement control

Train-derived evidence/pathology statistics are allowed because they are learned from the DDXPlus training split and used only as support/contradiction context.

## Dry-Run Validation

Default dry-run configuration:

| Setting | Value |
|---|---|
| `RUN_LIVE_API` | `False` |
| `ALLOW_DRY_RUN_BENCHMARK` | `True` |
| `RUN_SCOPE` | `pilot24` |
| `DRY_RUN_MAX_CASES` | `2` |
| `MAX_REQUEST_CAP` | `24` |

Dry-run artifact root:

```text
artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_dryrun_smoke_v1/
```

Validation status:

- static parse passed for all Notebook `20` code cells
- dry-run executed top-to-bottom without API credentials
- prompt preview includes the `GRAPH LEDGER CONTEXT` block
- graph context rows were written
- prediction/traces/raw-response artifacts were written
- promotion decision correctly marks dry-run as non-promotional evidence

Dry-run metrics are only a smoke test and should not be interpreted scientifically:

| Metric | Value |
|---|---:|
| Cases | 2 |
| Accuracy | 1.000 |
| Top-5 | 1.000 |
| Mean requests | 6.500 |
| Stop-before-cap rate | 1.000 |
| Mean graph context token estimate | about `2.49k` |

## Artifact Contract

Notebook `20` writes:

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

New prediction columns:

- `num_graph_warnings_final`
- `num_unresolved_pairs_final`
- `final_top_diagnosis_support`
- `final_top_diagnosis_contradiction`
- `warnings_resolved_count`
- `graph_context_tokens_estimate`

## Live Pilot Result

The live `pilot24` run has now completed.

Live artifact root:

```text
artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1/
```

Matched comparison:

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook `20` LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

Interpretation:

- Notebook `20` is not promoted because top-1 accuracy dropped from `22/24` to `20/24`.
- Notebook `20` is still important because top-3/top-5 improved to `23/24`, the strongest ranking quality observed in the graph-ledger line.
- The result suggests graph context may help keep the correct diagnosis in the differential, but does not yet help the system choose the final top-1.

## How To Run The Live Pilot

Open `notebooks/20_llm_led_graph_ledger_context.ipynb`.

In the first cell, either paste the API key through the notebook config path or enable interactive bootstrap:

```python
INTERACTIVE_API_KEY_BOOTSTRAP = True
```

In the config cell, set:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
RUN_SCOPE = "pilot24"
```

Keep:

```python
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_REQUEST_CAP = 24
```

Then run top-to-bottom.

The live pilot will write:

```text
artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1/
```

## Promotion Decision

The original promotion criteria were:

- at least `22/24` correct with mean requests `<= 7.0`
- at least `23/24` correct with mean requests `<= 8.0`
- same accuracy as Notebook `13` with materially better hard-case trace coherence and no new obvious failure pattern

Notebook `20` does not meet these criteria.

Decision:

- do not promote Notebook `20` to `final49`
- keep Notebook `13` frozen as the proposed method
- use the Notebook `20` top-5 signal for offline graph-critic analysis in Notebook `21`

## Current Interpretation

Notebook `20` is a mixed but useful result.

It fails as a replacement method because top-1 accuracy drops. It succeeds as an analysis signal because it improves the ranked differential. This is why Notebook `21` was created: to test whether graph context can serve as a critic or adjudicator.

The current conclusion is narrow:

> Graph-ledger prompt context improves ranking signal in this pilot, but not final-answer selection. The next graph-ledger step should be critic/adjudication analysis, not another live graph-context run.
