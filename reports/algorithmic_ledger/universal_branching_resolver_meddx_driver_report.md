# Notebook 45 Universal Branching Resolver MEDDx Driver

Notebook `45` ports the original DDXPlus architecture ideas into the multi-dataset MEDDx-style harness from Notebooks `42`-`44`.

Files:

- notebook: `notebooks/45_universal_branching_resolver_meddx_driver.ipynb`
- script mirror: `scripts/universal_branching_resolver_meddx_driver_nb45.py`
- latest dry-run artifact root: `artifacts/universal_meddx/universal_branching_resolver_meddx_driver_dryrun_smoke_v1_pilot4/`
- diagnosed live pilot root: `artifacts/universal_meddx/universal_branching_resolver_meddx_driver_v1_pilot3/`

## Control Question

Can the cross-dataset MEDDx-style driver use the DDXPlus project’s strongest components while keeping the MEDDx budget comparison fair?

The active policy keeps each case under the selected MEDDx question cap:

```text
budget in {5, 10, 15}
base questions + branch questions <= budget
```

Early stopping can use fewer questions than the cap, and hypothesis branches can spend only unused budget.

## Architecture

Notebook `45` adds these components on top of Notebook `44`:

- cap-aware stop probes after partial evidence collection
- actual DDXPlus partial-evidence MLP monitor when structured DDXPlus roots can be reconstructed
- universal confidence fallback for iCraft-MD and RareBench
- hypothesis-forced branches over challenger diagnoses
- candidate-pool resolver over base LLM rank, branch ranks, casebase prior, and RareBench graph/HPO support
- conservative RareBench graph/discriminator gate
- improved patient-simulator retrieval with semantic-topic alignment and RareBench phenotype rarity weighting

The DDXPlus MLP is deliberately not applied to iCraft-MD or RareBench. Those datasets do not share the DDXPlus evidence-root feature schema.

## RareBench Regression Fix

Notebook `44` showed that the RareBench graph/discriminator can help, but it can also regress correct LLM answers at high budget. Notebook `45` makes the graph/discriminator conservative:

- if LLM top-1 and graph top-1 agree, the answer is locked
- a weak graph margin cannot override the LLM top-1
- a discriminator-selected third option is blocked unless graph support is strong
- final resolution still keeps graph support in the candidate-pool score

This is a system-level no-regression gate, not a case-specific patch.

## Dry-Run Smoke

Verification completed with no API calls:

- `python3 -m py_compile scripts/universal_branching_resolver_meddx_driver_nb45.py`
- all Notebook `45` code cells parsed with `ast.parse`
- all three adapters loaded
- DDXPlus partial-evidence MLP monitor loaded successfully
- dry-run artifact contract passed

Dry-run selected one case per dataset at budget `5`. These numbers are not live performance claims because dry-run agents are scripted:

| Dataset | Case | Top-1 | Top-3 | Top-5 | Questions | Branches |
|---|---|---:|---:|---:|---:|---:|
| DDXPlus | `test:18312` | 1 | 1 | 1 | 2 | 0 |
| iCraft-MD | `icraft_md:55` | 0 | 0 | 1 | 5 | 2 |
| RareBench | `rarebench:LIRICAL:289` | 1 | 1 | 1 | 5 | 2 |

The original `v1_pilot3` smoke run mainly confirmed wiring:

- DDXPlus MLP confidence suppressed unnecessary branches on the DDXPlus smoke case
- RareBench graph support recovered `Cockayne syndrome`
- branch and resolver artifacts were written

## Live Pilot `v1_pilot3` Diagnosis

The first live pilot was intentionally tiny: one selected case per dataset, evaluated at budgets `5`, `10`, and `15`. It is too small for performance claims, but it exposed a real DDXPlus integration bug.

| Dataset | Top-1 Rows | Top-3 Rows | Top-5 Rows | Main Finding |
|---|---:|---:|---:|---|
| DDXPlus | `0/3` | `0/3` | `1/3` | `test:18312` true `Influenza` was missed at all budgets |
| iCraft-MD | `3/3` | `3/3` | `3/3` | levamisole-induced ANCA vasculitis case solved at all budgets |
| RareBench | `3/3` | `3/3` | `3/3` | Cockayne syndrome case solved at all budgets |

The DDXPlus miss was not evidence that the original DDXPlus architecture stopped working. Notebooks `43` and `44` had already run the same `test:18312` case; Notebook `44` solved it at budgets `5`, `10`, and `15`. Notebook `45` regressed it for two system-level reasons:

- the new semantic-topic patient-simulator penalty filtered out useful DDXPlus systemic evidence, especially appetite/fatigue/myalgia-style fields that were previously revealed with broad infectious questions
- the DDXPlus MLP monitor was treated as a high-confidence stop signal without checking whether the MLP diagnosis agreed with the LLM diagnosis

On the failed case, the MLP was highly confident in `URTI`, the LLM selected `Bronchiolitis` or `Viral pharyngitis`, and the ground truth was `Influenza`. That is a disagreement signal, not a safe stop signal. Early stopping then prevented the budget-10 and budget-15 runs from reaching the extra systemic evidence that Notebook `44` used to recover `Influenza`.

## `v1_pilot4` Patch

Notebook `45` now applies a system-level correction:

- DDXPlus retrieval no longer applies the RareBench-oriented topic-mismatch penalty
- DDXPlus early stopping requires agreement between the LLM stop-probe top-1 and the DDXPlus MLP top-1
- a high-confidence DDXPlus MLP can suppress branching only when it agrees with the base top-1
- DDXPlus MLP/LLM disagreement forces branch eligibility when budget remains
- MLP top-1/top-5 are now written to `predictions.csv` and `interaction_traces.jsonl`

No-API smoke passed after the patch under:

`artifacts/universal_meddx/universal_branching_resolver_meddx_driver_dryrun_smoke_v1_pilot4/`

The active live config is now `RUN_VERSION_SUFFIX = "v1_pilot4"`, still with `LIVE_TOTAL_MAX_CASES = 3` and budgets `[5, 10, 15]`.

## Artifact Contract

Notebook `45` writes:

- `resolved_run_config.json`
- `adapter_preflight.csv`
- `universal_cases.csv`
- `casebase_prior_reference_summary.csv`
- `rarebench_graph_phenotype_reference_summary.csv`
- `predictions.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `patient_simulator_retrieval_audit.csv`
- `branch_case_results.csv` when branches fire
- `candidate_level_resolver_scores.csv`
- figures under `figures/`

## Interpretation

Notebook `45` is the first complete cross-dataset port of the DDXPlus architecture. Notebook `44` proved the universal shell and RareBench graph idea; Notebook `45` adds the original project’s stop/branch/resolve logic inside that shell.

The next step is a small live pilot, not a large run. The active live config is intentionally conservative:

- `RUN_VERSION_SUFFIX = "v1_pilot4"`
- `LIVE_TOTAL_MAX_CASES = 3`
- budgets `[5, 10, 15]`

If that pilot behaves well, the natural follow-up is a scaled balanced run similar to Notebook `44`'s `v1_eval30`.
