# Cost-Sensitive Sequential Policy

## Purpose

Notebook `08_cost_sensitive_sequential_lambda_sweep.ipynb` turns the refined single-agent sequential policy into a cost-sensitive evidence-acquisition experiment.

The previous notebooks swept fixed request budgets. That is useful, but it makes the budget itself the main experimental control. The new notebook instead asks:

> Can the agent stop when additional evidence is unlikely to be worth its cost?

The request cap is now a safety ceiling. The experimental variable is `lambda`, the cost of acquiring one more evidence item.

## Fixed Backbone And Determinism

The notebook fixes the LLM backbone to:

- `gpt-4.1-mini`

All OpenAI-compatible API calls use:

- `temperature = 0.0`
- `top_p = 1.0`

The resolved run config logs:

- model name
- base URL
- temperature
- top-p
- case sampling seed
- shortlist tie-breaking rules
- case order convention
- lambda values
- request cap

The deterministic settings are meant to make reruns and critiques cleaner. They do not guarantee bit-perfect vendor-side reproducibility, but they remove the avoidable randomness under our control.

## Policy Change

The notebook keeps the notebook 05 architecture:

- single agent
- deterministic evidence ledger
- decoded DDXPlus evidence
- parent-child legality gating
- one-shot prior anchoring
- shortlist tied to competing diagnoses
- drift and stability controls

The new piece is a practical marginal-value estimate for another evidence request.

The estimate uses existing deterministic state signals:

- top shortlist score
- deterministic margin between leading diagnoses
- unresolved probability mass
- recent margin gain
- prediction stability
- remaining cap

The policy compares:

- estimated marginal value of another question
- evidence cost `lambda`

If the marginal value is lower than lambda and the current differential is sufficiently stable, the controller can force a stop. If the marginal value is clearly higher than lambda and the model tries to stop too early, the controller can force one more request.

This is intentionally not a probabilistic belief-state model, RL policy, graph inference method, or learned question selector. It is still a lightweight deterministic policy refinement.

## Artifact Layout

Notebook path:

- `notebooks/08_cost_sensitive_sequential_lambda_sweep.ipynb`

Artifact root:

- `artifacts/sequential_single_agent_cost_sensitive/`

Default dry-run artifact created during validation:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_dryrun_test_1perclass_cap24_6lambdas_lambda_cost_v1/`

Expected files:

- `resolved_run_config.json`
- `benchmark_cases.csv`
- `lambda_sweep_summary.csv`
- per-lambda `metrics.json`
- per-lambda `predictions.csv`
- per-lambda `traces.jsonl`
- per-lambda `raw_api_responses.jsonl`
- per-lambda `paired_vs_one_shot.csv`
- per-lambda `comparison_summary.json`
- plots under `figures/`

## Plots

The notebook generates:

- performance vs lambda
- macro-F1 vs lambda
- mean requests and stop rate vs lambda
- utility vs lambda
- accuracy vs mean requests
- token usage vs lambda
- request usage heatmap
- top requested evidence heatmap

## Validation Run

The notebook was executed end-to-end in dry-run mode.

This checked:

- deterministic config path
- dataset loading
- one-shot prior loading
- ledger construction
- shortlist construction
- lambda-aware stop signal construction
- artifact writing
- plot generation

Dry-run results should not be interpreted as live LLM performance. The dry-run policy uses the deterministic diagnosis state directly and is a contract check, not a scientific result.

## Final Live Pilot Results

The cost-sensitive notebook has now been run live on a 10-case balanced pilot.

Run artifact:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1/`

Live settings:

- model: `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- request cap: `24`
- cases: `10`
- lambda values: `[0.00, 0.03, 0.06, 0.10, 0.15, 0.22]`

| Lambda | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Stop before cap | Input tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.900 | 0.900 | 0.900 | 0.867 | 18.4 | 0.70 | 429,339 |
| 0.03 | 0.900 | 0.900 | 0.900 | 0.867 | 16.6 | 0.70 | 382,887 |
| 0.06 | 0.900 | 0.900 | 0.900 | 0.867 | 15.7 | 0.80 | 363,486 |
| 0.10 | 0.900 | 0.900 | 0.900 | 0.818 | 14.1 | 0.80 | 323,388 |
| 0.15 | 0.900 | 0.900 | 0.900 | 0.818 | 12.8 | 0.80 | 293,985 |
| 0.22 | 0.900 | 0.900 | 0.900 | 0.818 | 11.8 | 0.80 | 269,060 |

The main result is evidence efficiency. Accuracy stayed at `0.900` across all lambda values, while mean requests fell from `18.4` to `11.8`. That is roughly a 36% reduction in requested evidence. Input tokens fell by roughly 37%.

The strongest practical setting from this pilot is likely in the `0.10` to `0.22` range. It preserves accuracy while cutting a meaningful amount of evidence acquisition and API context usage.

Important caveat:

- this is a 10-case pilot, not a final statistical estimate
- the next validation should use the 49-case balanced sample with only a few selected lambda values

## Wider 24-Case Live Sweep

A second live run was executed to find the lambda cutoff more clearly.

Run artifact:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/`

Live settings:

- model: `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- request cap: `24`
- cases: `24`
- lambda values: `[0.10, 0.22, 0.35, 0.50, 0.75]`

| Lambda | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Cap hits | Stop before cap | Input tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.917 | 0.917 | 0.846 | 13.0 | 4/24 | 0.833 | 710,832 |
| 0.22 | 0.875 | 0.875 | 0.917 | 0.795 | 10.7 | 2/24 | 0.917 | 585,943 |
| 0.35 | 0.875 | 0.875 | 0.875 | 0.813 | 8.3 | 1/24 | 0.958 | 456,292 |
| 0.50 | 0.417 | 0.625 | 0.750 | 0.274 | 2.2 | 0/24 | 1.000 | 140,435 |
| 0.75 | 0.375 | 0.583 | 0.708 | 0.288 | 1.0 | 0/24 | 1.000 | 84,875 |

This run shows the cutoff that the 10-case pilot could not reveal.

Interpretation:

- `lambda = 0.10` is the highest-accuracy setting: `22/24` correct.
- `lambda = 0.22` and `lambda = 0.35` trade one additional error for fewer requests.
- `lambda = 0.35` is the strongest evidence-efficiency setting: `21/24` correct with about `8.3` requests per case.
- `lambda = 0.50` and `lambda = 0.75` are too aggressive; they stop after roughly one to two requests and collapse close to the initial-evidence baseline.
- cap hits decrease as lambda increases, which confirms that the cost-sensitive stopping rule is controlling evidence use.

Recommended lambda range after this run:

- use `0.10` if maximizing accuracy is the priority
- use `0.22` or `0.35` if evidence efficiency is the priority
- do not use `0.50+` for this policy without changing the stop rule

## Live Run Instructions

For a real experiment, set:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
TOP_P = 1.0
```

Use a fresh `RUN_VERSION` before each live run.

The recommended next live configuration is now:

```python
EVIDENCE_COST_LAMBDAS = [0.10, 0.22, 0.35]
MAX_REQUEST_CAP = 24
SEQUENTIAL_SAMPLE_PER_CLASS = 1
SEQUENTIAL_MAX_CASES = None
RUN_VERSION = "lambda_cost_49case_cutoff_v1"
```

That runs the 49-case balanced pilot while avoiding lambda values that the 24-case sweep already showed to be too aggressive.

## Scientific Meaning

This notebook moves the project toward evidence efficiency.

The key metrics are not just accuracy. The important tradeoff is:

- how much accuracy is gained
- how many evidence fields are requested
- how quickly the system stops
- whether utility improves under evidence cost

If lambda can reduce mean requests while preserving most of the higher-budget accuracy, that supports the claim that controlled sequential evidence acquisition is useful.
