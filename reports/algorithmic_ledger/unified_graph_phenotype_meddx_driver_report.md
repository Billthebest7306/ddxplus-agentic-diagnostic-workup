# Notebook 44: Unified Graph-Phenotype MEDDx Driver

Notebook: `notebooks/44_unified_graph_phenotype_meddx_driver.ipynb`  
Script mirror: `scripts/unified_graph_phenotype_meddx_driver_nb44.py`  
Dry-run artifact root: `artifacts/universal_meddx/unified_graph_phenotype_meddx_driver_dryrun_smoke_v1_pilot3/`

## Purpose

Notebook 44 addresses the first live Notebook 43 pilot failure. The goal is still a MEDDxAgent-style unified benchmark driver across DDXPlus, iCraft-MD, and RareBench, but the RareBench resolver is no longer treated as ordinary prose ranking over a long candidate list.

The new correction is a RareBench graph-phenotype resolver:

```text
visible RareBench phenotype names
  -> exact HPO phenotype-node set
  -> leave-one-case-out disease exemplar support within the same RareBench subset
  -> graph-prior rank fusion
  -> optional rare-disease discriminator prompt
```

## Notebook 43 Live Pilot Diagnosis

The latest live Notebook 43 pilot ran one selected case per dataset at budgets 5, 10, and 15:

| Dataset | Case | Budget 5 | Budget 10 | Budget 15 | Main finding |
|---|---:|---:|---:|---:|---|
| DDXPlus | Influenza | wrong | correct | correct | extra evidence recovered early respiratory anchoring |
| iCraft-MD | Levamisole-induced ANCA vasculitis | correct | correct | correct | case-level options plus clear exposure/morphology worked |
| RareBench | Cockayne syndrome | wrong | wrong | wrong | large rare-disease candidate resolution failed |

The overall row-level result was `5/9` top-1 across the three budgets, but that is only three unique cases and should be interpreted as a pilot diagnostic, not a benchmark estimate.

The important failure was RareBench. The agent collected useful phenotypes for `rarebench:LIRICAL:289`, but selected `Neurodevelopmental disorder with or without anomalies of the brain, eye, or heart` instead of `Cockayne syndrome`. Increasing the budget did not fix this; Cockayne moved from rank 8 at budget 5 to rank 10 at budget 10 and outside the top 10 at budget 15.

## Root Cause

Notebook 43's casebase prior used lexical-token overlap over free-text patient profiles. On RareBench, this is the wrong representation. Long broad phenotype profiles can share many words with the visible case and beat a more specific named disease.

For the failed Cockayne case, exact phenotype matching shows the opposite signal:

| Method | Top support for failed RareBench case |
|---|---|
| Notebook 43 prose-token casebase | broad neurodevelopmental phenotype category |
| Notebook 44 exact phenotype-node graph | `Cockayne syndrome` |

The fix is therefore mathematical rather than just prompt wording: use HPO phenotype names as atomic graph nodes.

Applying the Notebook 44 graph score to the saved Notebook 43 RareBench pilot trace gives the desired direction without using the case label:

| Budget | Notebook 43 LLM top-1 | Graph top-1 from exact phenotype nodes | Graph-fused top-1 |
|---:|---|---|---|
| 5 | Neurodevelopmental disorder with or without anomalies of the brain, eye, or heart | Cockayne syndrome | Cockayne syndrome |
| 10 | Neurodevelopmental disorder with or without anomalies of the brain, eye, or heart | Cockayne syndrome | Cockayne syndrome |
| 15 | Neurodevelopmental disorder with or without anomalies of the brain, eye, or heart | Cockayne syndrome | Cockayne syndrome |

This is diagnostic evidence, not a live Notebook 44 result, because it reuses the saved Notebook 43 trace. It justifies the next live pilot because the correction addresses the observed mechanism of failure directly.

## Implementation

Notebook 44 keeps the Notebook 43 MEDDxAgent-style skeleton:

- universal patient schema
- dataset adapters for DDXPlus, iCraft-MD, and RareBench
- separate history-taking and diagnosis phases
- deterministic hidden-profile patient simulator
- dynamic similar-patient examples
- MEDDx-style GTPA@1/@3/@5, rank, progress, and question-budget metrics

It adds:

- `RarebenchPhenotypeReference` records with exact phenotype sets
- `extract_rarebench_visible_phenotypes(...)`
- `rarebench_graph_phenotype_prior_for_case(...)`
- `apply_rarebench_graph_rerank(...)`
- `rarebench_graph_phenotype_text(...)`
- optional `rarebench_graph_phenotype_discriminator`

The graph score is based on same-subset, leave-one-case-out reference support:

```text
score(candidate disease) =
  max over reference cases with that disease:
    Jaccard(visible phenotype nodes, reference phenotype nodes)
    + 0.05 * visible_recall
    + 0.05 * reference_precision
```

The final RareBench resolver uses graph support as an audit/reranking signal and applies a small penalty to broad descriptive category labels.

## Dry-Run Verification

No-API smoke execution passed:

- script compiles
- notebook code cells parse
- all three adapters load
- dry-run selects one case per dataset
- artifact contract passes

Dry-run artifacts include:

- `resolved_run_config.json`
- `adapter_preflight.csv`
- `universal_cases.csv`
- `casebase_prior_reference_summary.csv`
- `rarebench_graph_phenotype_reference_summary.csv`
- `predictions.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `patient_simulator_retrieval_audit.csv`
- figures under `figures/`

The dry run is not a performance claim because the LLM agents are scripted. It is still useful because the failed RareBench case is present in the smoke cohort: the graph-phenotype layer recovers `Cockayne syndrome` as the top graph-supported diagnosis.

## Interpretation

Notebook 44 is the correct next live pilot after Notebook 43. The issue was not that MEDDx-style unification is impossible; the issue was that RareBench requires a different internal representation than DDXPlus or iCraft-MD.

For a publishable three-dataset claim, the next live run should use Notebook 44, not Notebook 43. A successful result would need to show:

- DDXPlus remains competitive at the 5/10/15 evidence budgets
- iCraft-MD remains stable with case-level options
- RareBench improves meaningfully from the Notebook 43 pilot
- graph-phenotype support improves RareBench without simply memorizing the evaluation case

## Next Run

The `v1_pilot3` live pilot passed and has now been replaced by the first scaled evaluation config:

```python
RUN_LIVE_API = True
RUN_VERSION_SUFFIX = "v1_eval30"
LIVE_TOTAL_MAX_CASES = 30
LIVE_BUDGETS_TO_RUN = [5, 10, 15]
```

This run should produce `90` live workups: `30` unique cases x `3` MEDDx-style budgets. With all adapters loaded, the sampler should select about `10` cases per dataset.

No-API smoke passed for this scaled configuration under:

`artifacts/universal_meddx/unified_graph_phenotype_meddx_driver_dryrun_smoke_v1_eval30/`

Original pilot-scale config:

```python
RUN_LIVE_API = True
RUN_VERSION_SUFFIX = "v1_pilot3"
LIVE_TOTAL_MAX_CASES = 3
LIVE_BUDGETS_TO_RUN = [5, 10, 15]
```

Keep the same frozen graph-phenotype resolver settings when scaling; otherwise the next result becomes another method-development run rather than an evaluation run.
