# Notebook 43 Unified MEDDx-Style Hybrid Driver

Notebook `43` is the first notebook that follows MEDDxAgent's multi-dataset architecture more directly while preserving our project’s deterministic ledger and hybrid resolver ideas.

## Motivation

Notebook `42` proved that the adapters can load DDXPlus, iCraft-MD, and RareBench into a shared schema, but its live `v5_pilot3` run showed that a single generic prompt was too brittle. MEDDxAgent does not actually rely on one generic prompt. It uses a shared driver with dataset adapters, exact diagnosis options, a history-taking phase, a patient simulator, dynamic similar-case examples, and a separate diagnosis strategy phase.

Notebook `43` adopts that shape.

## Architecture

```text
dataset adapter
  -> universal patient schema
  -> MEDDx-style history-taking agent
  -> deterministic guarded patient simulator
  -> visible patient-profile ledger
  -> MEDDx-style diagnosis strategy agent
  -> dynamic similar-case examples
  -> margin-gated reference-case prior rerank
  -> MEDDx-style top-k/rank metrics
```

The key change from Notebook `42` is separation of responsibilities:

- history-taking asks natural-language clinical questions
- the simulator answers from hidden-profile spans only
- diagnosis is performed by a separate final agent prompt
- dynamic reference examples are shown in the diagnosis phase, similar to MEDDxAgent few-shot prompting
- the reference-case prior can rerank the final differential only under a margin gate

## MEDDxAgent Alignment

| MEDDxAgent component | Notebook 43 adaptation |
|---|---|
| Benchmark adapters | DDXPlus, iCraft-MD, and RareBench adapters from Notebook `42` |
| Patient schema | `initial_patient_info`, hidden profile, ground truth, candidate list |
| History-taking agent | one question per budget step |
| Patient simulator | deterministic retrieval simulator instead of an LLM patient |
| Diagnosis strategy agent | separate final diagnosis prompt with exact options |
| Dynamic few-shot | visible-evidence Jaccard retrieval from reference cases |
| Metrics | GTPA@1, GTPA@3, GTPA@5, capped true rank, progress, questions |

## Active Config

Active pilot:

- notebook: `notebooks/43_unified_meddxstyle_hybrid_driver.ipynb`
- script mirror: `scripts/unified_meddxstyle_hybrid_driver_nb43.py`
- active suffix: `v1_pilot3`
- live case cap: `3` total cases across all enabled datasets
- budgets: `5`, `10`, `15`
- expected live workups: `9`
- model: `gpt-4.1-mini`
- temperature: `0.0`
- top_p: `1.0`

Full confirmation after pilot validation:

- set `RUN_VERSION_SUFFIX = "v1"`
- set `LIVE_TOTAL_MAX_CASES = 49`
- keep budgets `[5, 10, 15]`

## Dry-Run Smoke

No API calls were made.

Artifact root:

`artifacts/universal_meddx/unified_meddxstyle_hybrid_driver_dryrun_smoke_v1_pilot3/`

Reference-case prior counts:

| Dataset | Reference cases |
|---|---:|
| DDXPlus | 1500 |
| iCraft-MD | 140 |
| RareBench | 1120 |

Dry-run selected one case per enabled dataset and wrote the full artifact contract:

- `resolved_run_config.json`
- `adapter_preflight.csv`
- `universal_cases.csv`
- `casebase_prior_reference_summary.csv`
- `predictions.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `patient_simulator_retrieval_audit.csv`
- `meddx_style_metrics_summary.csv`
- figures under `figures/`

The dry-run metrics are not performance claims because the no-API history and diagnosis agents are scripted.

## Interpretation

Notebook `43` is the cleaner cross-dataset direction. It avoids the central Notebook `42` design mistake: treating universal diagnosis as a single generic prompt. It instead makes the framework universal while allowing dataset-aware diagnosis options, dynamic examples, and mathematical priors underneath.

The next step is to run the `v1_pilot3` live pilot. If it succeeds on the three-case smoke, restore the full `49` selected-case run and compare against MEDDxAgent’s 5/10/15-budget framing.
