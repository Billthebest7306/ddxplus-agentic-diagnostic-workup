# Notebook 51: MEDDx-Scale Hypothesis-Branching Confirmation

Notebook `51` is the frozen live-run scaffold for the next MEDDx phase. It is not an offline optimizer.

## Purpose

Run the current MEDDx-aligned architecture at the same scale as the MEDDxAgent interactive benchmark:

- `100` DDXPlus cases
- `100` iCraft-MD cases
- `100` RareBench cases
- budgets `5`, `10`, and `15`
- `900` total budgeted workups in live mode

The output will become the calibration corpus for a later offline resolver layer. The old `90`-workup Notebook `46` artifact should then be used as a retrospective transfer/regression check, not as the only proof set.

## Architecture

Notebook `51` preserves the working Notebook `46` design:

- one shared MEDDx-style budget/evaluation loop
- dataset-native evidence adapters
- DDXPlus structured evidence-root ledger with exact present/absent reveals
- DDXPlus partial-evidence MLP monitor and stopping rule
- iCraft-MD profile question-answer adapter
- RareBench phenotype graph adapter and conservative graph/discriminator gate
- hypothesis branching under unused budget
- candidate-pool resolver artifact export
- top-1/top-3/top-5/request-cost reporting

The run uses `gpt-4.1-mini`, `temperature = 0.0`, and `top_p = 1.0`.

Sampling follows the bundled MEDDxAgent driver pattern: shuffle patients with seed `42`, then take the first `N`.

## Live Configuration

| Field | Value |
|---|---:|
| Live cases per dataset | `100` |
| Enabled datasets | `ddxplus`, `icraft_md`, `rarebench` |
| Active budgets | `5`, `10`, `15` |
| Sample mode | `meddxagent_seed42_shuffle_first_n` |
| Unique live cases | `300` |
| Live workups | `900` |
| Max branches | `2` |
| Max branch questions | `2` |
| Total cap policy | base plus branch questions may not exceed active MEDDx budget |

## Artifacts

- notebook: `notebooks/51_meddx_scale_hypothesis_branching_confirmation.ipynb`
- script mirror: `scripts/meddx_scale_hypothesis_branching_confirmation_nb51.py`
- live artifact root: `artifacts/universal_meddx/meddx_scale_hypothesis_branching_confirmation_v1_meddx100/`
- dry-run artifact root: `artifacts/universal_meddx/meddx_scale_hypothesis_branching_confirmation_dryrun_smoke_v1_meddx100/`

Expected outputs:

- `resolved_run_config.json`
- `adapter_preflight.csv`
- `universal_cases.csv`
- `predictions.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `patient_simulator_retrieval_audit.csv`
- `candidate_level_resolver_scores.csv`
- `branch_case_results.csv` when branches fire
- `meddx_style_metrics_summary.csv`
- figures under `figures/`

## Dry-Run Verification

The no-API smoke run completed successfully:

- loaded `1` case per dataset
- ran all three MEDDx budgets
- queued `9` dry-run workups
- wrote the required artifact contract
- confirmed `resolved_run_config.json` records `live_cases_per_dataset = 100`

This is only a wiring check. It is not a performance claim.

## Use After Live Run

After the live run finishes:

1. Analyze Notebook `51` metrics directly against MEDDxAgent by dataset and budget.
2. Train/calibrate the next offline candidate-pool resolver on the new large cohort.
3. Freeze that resolver.
4. Apply the frozen resolver back to the old Notebook `46` `90`-workup artifact as a cross-cohort transfer check.

This separates live data generation, offline calibration, and transfer evidence cleanly.
