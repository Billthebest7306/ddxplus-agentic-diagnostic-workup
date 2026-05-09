# Live Graph-Bayes Rescue Confirmation

Last updated: 2026-05-08

## Summary

Notebook `24` is the live-confirmation wrapper for the Notebook `23` graph/Bayes rescue breakthrough.

- notebook: `notebooks/24_live_graph_bayes_rescue_confirmation.ipynb`
- dry-run artifact root: `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_dryrun_smoke_v1/`
- live artifact root: `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1/`

The notebook keeps the Notebook `13` live LLM workup loop unchanged, then applies the frozen Notebook `23` rescue layer after the base stop.

## Method

Execution order:

1. Run the Notebook `13` LLM-led evidence acquisition loop with MLP-guided stopping.
2. Save the base `notebook13_live_prediction`.
3. Reconstruct the visible evidence roots from the live trace.
4. Apply the frozen Notebook `23` rescue policy:
   - prior recovery certificate
   - conservative graph critic certificate
   - graph/Bayes rescue trigger for suspicious early `agent_stop` cases
   - up to three deterministic rescue evidence requests
   - post-rescue partial-evidence MLP update
   - train-calibrated L2 candidate reranker
5. Save the final `graph_bayes_rescue_prediction`.

No full-evidence predictions, labels, hidden differentials, or correctness flags are used inside the live decision policy.

## Dry-Run Validation

Default configuration:

| Setting | Value |
|---|---|
| `RUN_LIVE_API` | `False` |
| `ALLOW_DRY_RUN_BENCHMARK` | `True` |
| `DRY_RUN_MAX_CASES` | `2` |
| `LLM_MODEL` | `gpt-4.1-mini` |
| `TEMPERATURE` | `0.0` |
| `TOP_P` | `1.0` |
| `MAX_REQUEST_CAP` | `24` |

Dry-run result:

| Metric | Value |
|---|---:|
| Cases | 2 |
| Notebook `13` base accuracy | 1.000 |
| Graph/Bayes rescue accuracy | 1.000 |
| Mean base requests | 6.00 |
| Mean total requests | 6.00 |
| Extra rescue requests | 0 |
| Regressions | 0 |

This dry-run is not scientific evidence. It validates the notebook mechanics: prompt construction, base workup execution, trace reconstruction, rescue policy loading, candidate reranker scoring, artifact writing, and figure generation.

## Live 49-Case Result

The live confirmation run completed on the same 49-case benchmark scope.

| System | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|
| Original Notebook `13` artifact | `43/49 = 0.878` | `0.918` | `0.939` | `0.845` | `6.59` |
| Notebook `23` offline rescue candidate | `47/49 = 0.959` | n/a | n/a | n/a | `6.96` |
| Notebook `24` fresh live base workup | `45/49 = 0.918` | `0.939` | `0.939` | `0.895` | `6.20` |
| Notebook `24` live graph/Bayes rescue | `45/49 = 0.918` | `0.939` | `0.939` | `0.895` | `6.39` |

The rescue layer was **not promoted**. It made no net improvement over the fresh live base trajectory:

| Metric | Value |
|---|---:|
| Cases | `49` |
| Live base correct | `45` |
| Rescue correct | `45` |
| Improvements over live base | `0` |
| Regressions against live base | `0` |
| Changed predictions | `1` |
| Extra rescue requests | `9` |

Decision source distribution:

| Decision source | Cases |
|---|---:|
| Notebook `13` live reference retained | `45` |
| Rescue abstained after extra evidence | `3` |
| Prior recovery certificate | `1` |

The one changed prediction was `test:76022` Panic attack: the live base predicted Anaphylaxis, and the prior-recovery certificate changed it to PSVT. Both were incorrect, so this was neither an improvement nor a regression.

The rescue continuation requested extra evidence on three already-correct cases and then abstained:

| Case | True pathology | Extra roots | Outcome |
|---|---|---|---|
| `test:58986` | Acute laryngitis | `E_194`, `E_66`, `E_190` | remained correct |
| `test:90978` | Bronchiolitis | `E_181`, `E_66`, `E_4` | remained correct |
| `test:38202` | Inguinal hernia | `E_53`, `E_220`, `E_166` | remained correct |

## Live Base Drift Versus The Original Notebook `13` Artifact

The most important result is that the fresh Notebook `13`-style live base trajectory inside Notebook `24` improved from the original `43/49` artifact to `45/49`, while using fewer requests (`6.20` versus `6.59`). This appears to be live-run trajectory variation, despite deterministic notebook settings.

Changed outcomes relative to the original Notebook `13` artifact:

| Case | True pathology | Original Notebook `13` | Notebook `24` live base | Result |
|---|---|---|---|---|
| `test:125508` | Unstable angina | Anemia | Unstable angina | fixed |
| `test:81691` | Croup | Anemia | Croup | fixed |
| `test:8666` | Influenza | HIV (initial infection) | Influenza | fixed |
| `test:76022` | Panic attack | Panic attack | Anaphylaxis | regression |
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | Anemia | still wrong |

Remaining live rescue errors:

| Case | True pathology | Final rescue prediction |
|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis |
| `test:38475` | Acute COPD exacerbation / infection | Anemia |
| `test:62878` | Pericarditis | Anemia |
| `test:76022` | Panic attack | PSVT |

## Artifact Contract

Notebook `24` writes:

- `metrics.json`
- `predictions.csv`
- `traces.jsonl`
- `raw_api_responses.jsonl`
- `rescue_trace.jsonl`
- `candidate_scores.csv`
- `paired_notebook13_vs_live_rescue.csv`
- `selected_live_rescue_policy.json`
- `resolved_run_config.json`
- `qualitative_examples.json`
- figures under `figures/`

`predictions.csv` includes both the base and rescue predictions:

- `notebook13_live_prediction`
- `graph_bayes_rescue_prediction`
- `notebook13_correct`
- `rescue_correct`
- `num_requests_notebook13`
- `extra_rescue_requests`
- `num_requests_total`
- `decision_source`
- `extra_requested_roots`

## How To Run The Live Confirmation

Open `notebooks/24_live_graph_bayes_rescue_confirmation.ipynb`.

In the config cell, set:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
```

Keep:

```python
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_REQUEST_CAP = 24
SEQUENTIAL_MAX_CASES = 49
```

The notebook uses interactive API key bootstrap when `RUN_LIVE_API=True` and no `LLM_API_KEY` is already present.

The live run will write to:

```text
artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1/
```

## Promotion Criteria And Decision

The live rescue layer should be promoted only if:

- strong confirmation: at least `46/49` correct and zero regressions versus the live Notebook `13` base prediction
- ideal confirmation: `47/49` correct with mean total requests `<= 7.25`

If the rescue layer falls below the Notebook `13` base result or introduces a regression without at least two compensating improvements, keep Notebook `13` as the final defended method and treat Notebook `23` as an offline candidate only.

Decision:

```text
not_promoted
```

Notebook `24` did not confirm the Notebook `23` offline rescue gain in a fresh live trajectory. It does, however, strengthen the live evidence for the base architecture: the LLM-led workup plus MLP stopping reached `45/49 = 0.918` with only `6.20` mean requests before rescue.

## Current Status

Status:

```text
live_completed_not_promoted
```

Notebook `24` should be kept as a live confirmation and robustness result. The frozen Notebook `13`-style live backbone remains the defensible acquisition method; Notebook `23` remains an offline graph/Bayes rescue candidate that did not transfer cleanly to this fresh live run.
