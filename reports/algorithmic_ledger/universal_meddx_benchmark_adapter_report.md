# Notebook 42 Universal MEDDx Benchmark Adapter

Last updated: 2026-05-21

Notebook `42` starts the cross-dataset generalization phase of the project.

## Purpose

The DDXPlus notebooks use a dataset-native evidence ledger: the model requests legal DDXPlus evidence roots and receives exact structured values. That is powerful, but it is not universal enough for the broader MEDDxAgent benchmark family.

Notebook `42` introduces a universal interaction harness:

```text
dataset adapter
  -> universal patient case schema
  -> LLM natural-language question
  -> guarded patient simulator answers from hidden profile only
  -> question-answer ledger
  -> LLM final ranked differential
  -> MEDDx-style metrics
```

This lets the same workup loop run on:

- DDXPlus, by converting structured evidence rows into hidden patient profiles
- iCraft-MD, through the MEDDxAgent benchmark JSONL file
- RareBench, through MEDDxAgent mapping files plus the public HuggingFace RareBench data zip

## Files

- notebook: `notebooks/42_universal_meddx_benchmark_adapter.ipynb`
- script mirror: `scripts/universal_meddx_benchmark_adapter_nb42.py`
- active pilot dry-run artifact root: `artifacts/universal_meddx/universal_meddx_benchmark_adapter_dryrun_smoke_v6_pilot3/`
- next pilot live artifact root: `artifacts/universal_meddx/universal_meddx_benchmark_adapter_v6_pilot3/`
- full live artifact root after pilot validation: `artifacts/universal_meddx/universal_meddx_benchmark_adapter_v6/`

## Universal Schema

Every adapter emits:

| Field | Meaning |
|---|---|
| `case_id` | stable case identifier |
| `dataset_name` | `ddxplus`, `icraft_md`, or `rarebench` |
| `initial_patient_info` | visible starting state |
| `hidden_full_profile` | complete patient profile used only by the simulator |
| `ground_truth_diagnosis` | evaluation label |
| `candidate_disease_list` | possible diagnoses shown to the agent |
| `metadata` | adapter-specific provenance |

## Patient Simulator

Version 1 uses a deterministic retrieval simulator. It retrieves profile spans relevant to the agent's natural-language question and answers only from those spans. If the profile does not mention the requested fact, it answers that the available patient profile does not mention it.

This avoids using dataset-specific evidence-root IDs while also avoiding an unconstrained LLM-patient simulator that could hallucinate.

## Metrics

Notebook `42` reports MEDDx-style metrics:

- `GTPA@1`
- `GTPA@3`
- `GTPA@5`
- capped mean rank of the true diagnosis, with missing rank `11`
- progress rate from first ranked differential to final ranked differential
- mean questions asked
- stop-before-budget rate
- token counts

The notebook keeps the MEDDxAgent reference budgets `5`, `10`, and `15`, and the live comparison run now evaluates `49` selected cases at each budget.

Live evaluation uses a global unique-case cap: `LIVE_TOTAL_MAX_CASES = 49` across all loaded datasets, balanced over the available adapters. With three active budgets, the intended live progress-bar denominator is therefore `147` workups, not `300`.

`REQUIRE_ALL_ENABLED_DATASETS = True` is now enabled. If DDXPlus, iCraft-MD, or RareBench fails preflight, the notebook raises before spending live API budget.

The expected MEDDxAgent benchmark files live under `external/meddxagent/ddxdriver/benchmarks/data/`. That external checkout is intentionally not tracked in this repo; recreate it with `git clone --depth 1 https://github.com/nec-research/meddxagent.git external/meddxagent` if needed.

## Failed v4 Live Diagnostic

Notebook `42` live `v4` was a useful failure, not a valid benchmark result. It ran one combined cross-dataset cohort of `49` selected cases across DDXPlus, iCraft-MD, and RareBench at budgets `5`, `10`, and `15`.

| Budget | Cases | GTPA@1 | GTPA@3 | GTPA@5 | Mean questions |
|---:|---:|---:|---:|---:|---:|
| 5 | 49 | 0.184 | 0.286 | 0.327 | 4.16 |
| 10 | 49 | 0.204 | 0.245 | 0.286 | 7.82 |
| 15 | 49 | 0.265 | 0.327 | 0.388 | 11.12 |

Root causes:

- candidate lists were truncated alphabetically at `5000` characters; the true diagnosis was visible for only `50%` of selected iCraft-MD cases and `25%` of selected RareBench cases
- RareBench predictions collapsed toward early alphabetic diagnoses, especially `3-Hydroxy-3-methylglutaryl-CoA lyase deficiency`
- DDXPlus question-answer spans were split at question marks, so answers often returned only the question text without `Answer: yes/no/value`
- RareBench phenotype-only profiles produced too many `not mentioned` responses under the generic lexical retriever
- exact-label evaluation exposed spelling/alias issues in iCraft-MD and RareBench

Because several selected cases were impossible under the prompt actually shown to the model, `v4` should be treated as a harness failure analysis, not as a system-performance claim.

## v5 Repairs

Notebook `42` `v5` keeps the same combined cross-dataset design but repairs the harness:

- raises `CANDIDATE_TEXT_MAX_CHARS` to `50000`, so selected iCraft-MD/RareBench true labels are not truncated out of the prompt
- uses iCraft-MD case-level answer options instead of the full dermatology diagnosis universe
- uses RareBench subset-level diagnosis options instead of a combined cross-subset list
- keeps DDXPlus `Question? Answer: value` spans intact during simulator retrieval
- increases simulator answer capacity to `5` spans and skips previously revealed spans
- adds dataset-specific prompting for DDXPlus, iCraft-MD, and RareBench
- canonicalizes near-exact output labels back to allowed candidate names before scoring

The `v5` no-API smoke loaded all three adapters and selected one case per dataset. All three selected cases had the true diagnosis in the candidate list:

| Dataset | Selected smoke cases | Candidate count | True diagnosis visible |
|---|---:|---:|---:|
| DDXPlus | 1 | 49 | yes |
| iCraft-MD | 1 | 4 | yes |
| RareBench | 1 | 216 | yes |

## v5_pilot3 Live Diagnostic

The cost-guarded `v5_pilot3` live run completed `9` workups: one selected case from each enabled dataset at budgets `5`, `10`, and `15`.

| Dataset | Case | Truth | Budget 15 Prediction | Budget 15 GTPA@1 | Budget 15 GTPA@5 |
|---|---|---|---|---:|---:|
| DDXPlus | `test:18312` | Influenza | Pneumonia | 0 | 0 |
| iCraft-MD | `icraft_md:55` | Levamisole-induced antineutrophil cytoplasmic antibody vasculitis | Levamisole-induced antineutrophil cytoplasmic antibody vasculitis | 1 | 1 |
| RareBench | `rarebench:LIRICAL:289` | Cockayne syndrome | Aarskog-Scott syndrome | 0 | 0 |

This was not the old candidate-truncation failure. All adapters loaded and the selected true labels were visible. The remaining failures were behavioral:

- DDXPlus: the agent narrowed too early to bronchiolitis/pneumonia-style respiratory questions. It asked fever, cough, dyspnea/wheeze, rhinorrhea, sore throat, and repeat airway questions, but missed the broad positive syndrome inventory: fatigue/bedridden state, diffuse myalgia, appetite loss, headache/neck pain, sweating, and immunosuppression.
- iCraft-MD: the repaired case-level options worked. The agent asked about cocaine/warfarin, rash morphology, and coagulopathy, then selected the correct option.
- RareBench: the agent retrieved useful phenotypes but lacked a disease-phenotype ranking prior over a `216`-diagnosis LIRICAL candidate list. It over-ranked common dysmorphology/neurodevelopmental alternatives while never bringing Cockayne syndrome into the top five.

Interpretation: `v5_pilot3` moved Notebook `42` from a broken harness to a real universal-workup problem. The next fix should not be another candidate-list capacity change; it should add a broad positive-finding acquisition step and a mathematical prior/resolver signal for large candidate spaces.

## v6 Repair

Notebook `42` now uses active suffix `v6_pilot3`. The repair adds two universal mechanisms:

- Broad first-turn inventory: the prompt now prefers asking for additional positive symptoms/findings/phenotypes before committing to a narrow path, and the guarded simulator ranks broad-inventory answers toward high-yield positive spans.
- Margin-gated reference-case prior: the notebook builds a visible-evidence-only Jaccard casebase prior from reference patients, using DDXPlus train cases plus iCraft-MD/RareBench reference profiles. The prior is only used for candidate lists with at least `10` options, and final reranking is margin-gated so low-margin priors do not force a diagnosis.

The no-API `v6_pilot3` smoke passed under:

`artifacts/universal_meddx/universal_meddx_benchmark_adapter_dryrun_smoke_v6_pilot3/`

Reference-case prior counts:

| Dataset | Reference cases |
|---|---:|
| DDXPlus | 1500 |
| iCraft-MD | 140 |
| RareBench | 1120 |

The dry-run is still not a performance claim because the no-API agent path is scripted. It verifies the new artifact contract, including `casebase_prior_reference_summary.csv`, and confirms that the next live pilot should be `v6_pilot3`, not `v5_pilot3`.

## Dry-Run Smoke Result

No API calls were made.

| Dataset | Status | Cases |
|---|---|---:|
| DDXPlus | loaded | 3 |
| iCraft-MD | loaded | 3 |
| RareBench | loaded | 3 |

Dry-run metrics are not performance claims because the agent path is scripted. They only validate the adapter and artifact contract.

| Dataset | Budget | Cases | GTPA@1 | GTPA@3 | GTPA@5 | Mean Questions |
|---|---:|---:|---:|---:|---:|---:|
| DDXPlus | 5 | 1 | 0.000 | 0.000 | 0.000 | 2.0 |
| iCraft-MD | 5 | 1 | 0.000 | 0.000 | 0.000 | 2.0 |
| RareBench | 5 | 1 | 0.000 | 0.000 | 0.000 | 2.0 |

## Artifact Contract

The dry-run writes:

- `resolved_run_config.json`
- `adapter_preflight.csv`
- `universal_cases.csv`
- `predictions.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `patient_simulator_retrieval_audit.csv`
- `meddx_style_metrics_summary.csv`
- figures under `figures/`

## Interpretation

Notebook `42` is the first universal version of the system. It does not yet prove cross-dataset performance, but it creates the necessary abstraction boundary:

- dataset-specific loading is isolated in adapters
- evidence acquisition is natural-language and LLM-led
- the patient answer channel is guarded by hidden-profile retrieval
- evaluation uses MEDDx-style top-k/rank/progress metrics

The next step is a cost-guarded `v6_pilot3` live run: `3` selected cases across DDXPlus/iCraft-MD/RareBench at each of budgets `5`, `10`, and `15`, for `9` live workups. If that pilot passes, restore `RUN_VERSION_SUFFIX = "v6"` and `LIVE_TOTAL_MAX_CASES = 49` for the full `147`-workup run.
