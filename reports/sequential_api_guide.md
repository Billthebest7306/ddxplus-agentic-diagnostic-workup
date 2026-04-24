# Sequential API Guide

This guide explains how to run [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb) with a real API-backed model.

## What The Notebook Expects

The notebook uses an OpenAI-compatible `POST /chat/completions` interface by default.

You need to provide:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

The notebook reads those from the environment first and also exposes matching placeholders in the config cell.

## Recommended Setup

Before launching Jupyter, export the variables in your shell:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="YOUR_API_KEY_HERE"
export LLM_MODEL="gpt-4.1-mini"
```

Then open the notebook:

```bash
jupyter lab notebooks/02_single_agent_sequential_baseline.ipynb
```

If you prefer, you can leave the environment empty and paste values directly into the notebook config cell instead.

## Notebook Placeholders To Review

In the top config cell, check these values:

- `RUN_LIVE_API = True`
- `ALLOW_DRY_RUN_BENCHMARK = False`
- `MAX_REQUESTS = 3`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 5`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `USE_JSON_MODE = False`
- `INPUT_COST_PER_1K_TOKENS`
- `OUTPUT_COST_PER_1K_TOKENS`

Notes:

- Keep `USE_JSON_MODE = False` unless you know your provider supports OpenAI JSON mode.
- Fill the token pricing fields only if you want cost estimates in the saved metrics.

## Dry-Run Vs Live Mode

`RUN_LIVE_API = False`
- no external API calls are made
- the notebook previews the prompt, action space, and a schema-valid mock response
- useful for checking formatting and environment behavior
- not a fair benchmark result

`RUN_LIVE_API = True`
- the notebook calls the configured chat-completions endpoint
- each case can request up to `MAX_REQUESTS` evidence fields
- outputs are saved incrementally so interrupted runs can resume

## Response Format Required From The Model

The model must return one JSON object with these keys:

```json
{
  "decision": "request",
  "requested_evidence_id": "some_root_id",
  "predicted_pathology": "Bronchitis",
  "ranked_differential": ["Bronchitis", "URTI", "Pneumonia", "Asthma", "Bronchiolitis"],
  "confidence": 0.62,
  "brief_reasoning": "The current evidence favors a respiratory infection, but more history is needed."
}
```

Rules:

- `decision` must be `request` or `stop`
- `requested_evidence_id` must be null when `decision` is `stop`
- `requested_evidence_id` must be one of the currently available DDXPlus root evidence ids when `decision` is `request`
- `predicted_pathology` must be one of the 49 DDXPlus labels
- `ranked_differential` must contain up to 5 valid labels

## What Happens If The Model Responds Badly

The notebook has one automatic repair pass.

If the first answer is malformed or invalid:

- the notebook sends one repair message asking for corrected JSON
- if the repaired answer is still invalid, the case is forced to stop
- the run logs the failure in `error_flags`

This applies to:

- malformed JSON
- invalid pathology labels
- missing ranked differential
- invalid or repeated evidence requests

## Resume Behavior

The live run writes incrementally to:

- `predictions.csv`
- `traces.jsonl`
- `raw_api_responses.jsonl`

If the notebook is interrupted and `RESUME_IF_AVAILABLE = True`, rerunning the execution cell will skip any `case_id`s already present in `predictions.csv`.

That means you do not need to restart the 245-case benchmark from zero after a crash or network issue.

## Saved Outputs

The sequential run writes to:

- `artifacts/sequential_single_agent/<run_name>/metrics.json`
- `artifacts/sequential_single_agent/<run_name>/predictions.csv`
- `artifacts/sequential_single_agent/<run_name>/traces.jsonl`
- `artifacts/sequential_single_agent/<run_name>/raw_api_responses.jsonl`
- `artifacts/sequential_single_agent/<run_name>/resolved_run_config.json`
- `artifacts/sequential_single_agent/<run_name>/benchmark_cases.csv`

Key fields in `predictions.csv`:

- `case_id`
- `source_row_index`
- `true_pathology`
- `predicted_pathology`
- `correct`
- `ranked_differential`
- `num_requests`
- `stop_reason`
- `api_calls`
- `input_tokens`
- `output_tokens`
- `estimated_cost`

## Cost Reporting

If you set:

- `INPUT_COST_PER_1K_TOKENS`
- `OUTPUT_COST_PER_1K_TOKENS`

the notebook will estimate cost per case and total run cost.

If those fields stay `None`, cost values remain empty and the rest of the evaluation still works.

## Practical Run Order

1. Run the dry-run preview cells first.
2. Confirm the prompt and action list look sensible.
3. Switch `RUN_LIVE_API = True`.
4. Fill the API placeholders or export the env vars.
5. Run the benchmark execution cell.
6. After it finishes, open [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb) to compare against the selected one-shot model.
