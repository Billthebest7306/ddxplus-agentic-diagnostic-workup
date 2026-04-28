# DDXPlus Agentic Diagnostic Workup

This repository contains a notebook-first research workflow for a DDXPlus-based diagnostic workup project. The project started with a strong one-shot baseline, then moved into single-agent sequential workup, and is now focused on making that sequential policy more structured, more stateful, and more diagnostically efficient before moving to a true multi-agent system.

The current research direction is:

- dataset: `DDXPlus`
- baseline ladder:
  - strong one-shot classifier
  - earlier single-agent sequential baseline
  - structured single-agent sequential baseline
  - refined single-agent sequential baseline with anchored diagnosis state
- best current direction:
  - use the sequential policy as a controlled evidence-acquisition system
  - use either the LLM, the partial-evidence classifier, or a hybrid/adjudicated head for final diagnosis
  - evaluate whether targeted evidence acquisition can approach full-evidence performance with far fewer revealed fields
- future target:
  - explainable evidence-gated multi-agent diagnostic workup system

## What Is In This Repo

Main folders:

- [notebooks](notebooks)
- [artifacts](artifacts)
- [reports](reports)
- [scripts](scripts)
- [PROJECT_WORKLOG.md](PROJECT_WORKLOG.md)

Supporting docs:

- [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md)
- [PROJECT_WORKLOG.md](PROJECT_WORKLOG.md)

Dataset helper:

- [download_ddxplus.py](scripts/download_ddxplus.py)

## Current Notebook Map

Notebook progression:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)
  - trains and compares one-shot BASD-style classifiers
  - produces the selected one-shot comparator

- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)
  - earlier sequential baseline
  - useful mainly as historical reference now

- [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb)
  - compares one-shot and sequential outputs on aligned cases

- [04_single_agent_structured_policy_improvement.ipynb](notebooks/04_single_agent_structured_policy_improvement.ipynb)
  - first structured-policy sequential improvement
  - cleaner ledger, legality, and shortlisting
  - important intermediate step, but empirically weaker than the current refined version

- [05_single_agent_structured_policy_refinement.ipynb](notebooks/05_single_agent_structured_policy_refinement.ipynb)
  - current main refined sequential notebook
  - adds anchored deterministic diagnosis state, stronger shortlist scoring, and improved stop/request behavior

- [06_single_agent_budget_scaling.ipynb](notebooks/06_single_agent_budget_scaling.ipynb)
  - same policy as notebook 05
  - only changes default request budgets to study scaling and plateau behavior

- [07_full_evidence_one_shot_comparator.ipynb](notebooks/07_full_evidence_one_shot_comparator.ipynb)
  - trains the full-evidence direct diagnosis comparator
  - evaluation ceiling only; do not use full evidence inside live sequential policy

- [08_cost_sensitive_sequential_lambda_sweep.ipynb](notebooks/08_cost_sensitive_sequential_lambda_sweep.ipynb)
  - keeps the refined single-agent policy but replaces arbitrary budget sweeps with lambda/cost-sensitive stopping
  - fixed backbone for this phase: `gpt-4.1-mini`
  - deterministic API settings: `temperature=0.0`, `top_p=1.0`

- [09_matched_evidence_integrated_comparison.ipynb](notebooks/09_matched_evidence_integrated_comparison.ipynb)
  - compares initial one-shot, sequential, matched-evidence one-shot, and full-evidence one-shot on aligned cases
  - separates evidence acquisition value from final reasoning value

- [10_partial_evidence_one_shot_comparator.ipynb](notebooks/10_partial_evidence_one_shot_comparator.ipynb)
  - trains a direct classifier on partial-evidence states
  - uses sequential trace request patterns to create policy-shaped evidence masks
  - gives notebook `09` a fairer matched-information comparator than the full-evidence-model fallback

If you are continuing the main sequential-policy line, start from notebook `08` for cost-sensitive stopping, notebook `10` for the stronger matched comparator, or notebook `09` for integrated evaluation. Use notebooks `05` and `06` as the refined-policy history.

## Current State Of The Project

The current best live small-sample sequential result comes from the cost-sensitive notebook line.

Relevant artifact roots:

- [basd_pathology_full](artifacts/one_shot/basd_pathology_full)
- [full_evidence_pathology_full](artifacts/one_shot_full_evidence/full_evidence_pathology_full)
- [single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1](artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1)
- [single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1](artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1)

Current headline results:

- initial-evidence one-shot full test accuracy: about `0.378`
- full-evidence one-shot full test accuracy: about `0.996`
- cost-sensitive sequential live 10-case pilot accuracy: `0.900`
- cost-sensitive sequential mean requests dropped from `18.4` to `11.8` across the lambda sweep without reducing pilot accuracy
- cost-sensitive sequential 24-case wide sweep found the cutoff: `0.917` accuracy at lambda `0.10`, `0.875` at lambdas `0.22` and `0.35`, then collapse to `0.417` at lambda `0.50`
- old matched-evidence fallback on the same 24-case live slice reached `0.625` to `0.708` at useful lambdas
- new partial-evidence matched comparator reached `0.875` at lambdas `0.10` and `0.22`, and `0.833` at lambda `0.35`
- current interpretation: evidence acquisition is clearly useful; the final diagnostic head may be LLM, direct classifier, or a hybrid

## Best Current Research Direction

The strongest defensible direction is not "LLM reasoning beats every direct classifier." The current evidence says something more specific and more useful: a structured sequential policy can choose a small, targeted subset of DDXPlus evidence that makes diagnosis much easier than initial evidence alone. The partial-evidence classifier then shows that much of the value may come from acquiring the right evidence, not necessarily from the LLM being the best final diagnostic head.

The next research version should therefore frame the system as a diagnostic workup controller:

- the sequential LLM/policy decides what evidence to request, using the ledger, legality rules, one-shot prior, and cost-sensitive stopping
- the final diagnosis can be produced by the LLM, the partial-evidence neural classifier, or a hybrid rule that adjudicates disagreements
- the main scientific question becomes whether targeted sequential evidence acquisition can approach the full-evidence ceiling while revealing only a limited subset of fields

This is the cleanest path into the later multi-agent system. Multi-agent work should add specialized evidence-gathering roles and coordination on top of this evidence-acquisition framing, not replace it with unconstrained debate.

Important caution:

- the cost-sensitive sequential result is now stronger than the 10-case pilot, but the 24-case run is still not a final statistical claim
- the next recommended run is the 49-case balanced pilot with fixed `gpt-4.1-mini` and selected lambdas `[0.10, 0.22, 0.35]`
- older notebook `05` artifacts used `gpt-5.4-mini`; current rigorous comparison phase fixes the sequential backbone to `gpt-4.1-mini`

Latest report:

- [final_results_summary.md](reports/final_results_summary.md)
- [partial_evidence_matched_comparator.md](reports/partial_evidence_matched_comparator.md)

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download the DDXPlus dataset into the repo-local `dataset/` folder:

```bash
python3 scripts/download_ddxplus.py
```

The dataset is not committed to git. The repo now uses this path resolution order:

1. `DDXPLUS_DATASET_DIR` if set
2. `dataset/`
3. legacy fallback: `.data/ddxplus/22687585/`

Examples:

```bash
export DDXPLUS_DATASET_DIR="/absolute/path/to/your/local/ddxplus"
python3 scripts/download_ddxplus.py --output-dir "$DDXPLUS_DATASET_DIR"
```

If you and a collaborator keep the dataset in different places on different machines, set `DDXPLUS_DATASET_DIR` locally and do not commit the dataset directory.

## Running The Project

### One-shot baseline

Run notebook `01` for:

- one-shot training
- one-shot model selection
- one-shot artifacts under `artifacts/one_shot/`

### Sequential notebooks

For sequential runs:

1. open notebook `05` or `06`
2. set the experiment variables in the main config cell
3. use environment variables or the safe bootstrap cell for API credentials
4. rerun top-to-bottom

### API variables

Typical variables:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

Do not hardcode real keys into notebook cells or outputs.

## Artifact Layout

Main artifact families:

- `artifacts/one_shot/`
- `artifacts/one_shot_full_evidence/`
- `artifacts/one_shot_partial_evidence/`
- `artifacts/sequential_single_agent/`
- `artifacts/sequential_single_agent_improved/`
- `artifacts/sequential_single_agent_refined/`
- `artifacts/sequential_single_agent_cost_sensitive/`
- `artifacts/comparisons/`
- `artifacts/integrated_comparisons/`
- `artifacts/_legacy/`

General rule:

- keep each experiment in its own artifact directory
- use a fresh `RUN_VERSION` whenever behavior or model settings change
- do not overwrite prior runs unless there is a specific cleanup reason

## Collaboration Rules

This repo is intentionally notebook-first, but it still needs discipline.

### 1. New notebook rule

Do not keep rewriting the same notebook once the method meaningfully changes.

Use the next numbered notebook when:

- the policy logic changes materially
- the experiment framing changes materially
- the notebook becomes a new methodological stage

Examples:

- `04` -> first structured policy improvement
- `05` -> refined anchored diagnosis-state policy
- `06` -> budget-scaling experiment using the same policy

Use the same notebook only when:

- the change is tiny
- the notebook is clearly the same experimental stage
- you are only fixing a bug or improving clarity

### 2. Worklog rule

After every meaningful change, update:

- [PROJECT_WORKLOG.md](PROJECT_WORKLOG.md)

This is mandatory for continuity.

The worklog should record:

- what changed
- why it changed
- what notebook or report was added/updated
- what was tested
- what the result means
- what the next likely step is

Treat the worklog as the persistent research memory for the repo.

### 3. Report rule

If a result matters, write a report in:

- [reports](reports)

Do this when:

- a notebook produced an important new result
- a debugging pass revealed a major failure mode
- a policy change materially improved results

### 4. Secrets rule

Never commit:

- real API keys
- notebook outputs containing real API keys
- `dataset/`
- `.data/`

Before pushing:

- search notebooks for `sk-`
- clear or scrub any unsafe outputs

### 5. Experiment hygiene rule

When running a new experiment:

- change `RUN_VERSION`
- keep the artifact root distinct
- preserve old results for comparison

This matters because the comparison story in this repo depends heavily on exact run lineage.

## Recommended Handoff Starting Point

If Hassan is continuing immediately, the best starting point is:

- read [README.md](README.md)
- read [PROJECT_WORKLOG.md](PROJECT_WORKLOG.md)
- inspect [sequential_policy_refinement_report.md](reports/sequential_policy_refinement_report.md)
- inspect [ledger_implementation_explained.md](reports/ledger_implementation_explained.md)
- continue from notebook `05` or `06`

## Current Practical Recommendation

For the next clean experiment:

- run notebook `07` to create full-evidence one-shot artifacts
- run notebook `08` live with `gpt-4.1-mini` on the 10-case pilot
- if coherent, scale notebook `08` to the balanced 49-case pilot
- rerun notebook `09` to compare sequential, matched-evidence one-shot, initial one-shot, and full-evidence one-shot

Model ablations and multi-agent work are intentionally postponed until this single-agent evaluation phase is cleaner.
