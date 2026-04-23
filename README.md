# DDXPlus Agentic Diagnostic Workup

This repository contains a notebook-first research workflow for a DDXPlus-based diagnostic workup project. The project started with a strong one-shot baseline, then moved into single-agent sequential workup, and is now focused on making that sequential policy more structured, more stateful, and more diagnostically efficient before moving to a true multi-agent system.

The current research direction is:

- dataset: `DDXPlus`
- baseline ladder:
  - strong one-shot classifier
  - earlier single-agent sequential baseline
  - structured single-agent sequential baseline
  - refined single-agent sequential baseline with anchored diagnosis state
- future target:
  - explainable evidence-gated multi-agent diagnostic workup system

## What Is In This Repo

Main folders:

- [notebooks](/Users/bilalawan/claw/assignments/baseline_model/notebooks)
- [artifacts](/Users/bilalawan/claw/assignments/baseline_model/artifacts)
- [reports](/Users/bilalawan/claw/assignments/baseline_model/reports)
- [scripts](/Users/bilalawan/claw/assignments/baseline_model/scripts)
- [PROJECT_WORKLOG.md](/Users/bilalawan/claw/assignments/baseline_model/PROJECT_WORKLOG.md)

Supporting docs:

- [INSTRUCTOR_GUIDE.md](/Users/bilalawan/claw/assignments/baseline_model/INSTRUCTOR_GUIDE.md)
- [PROJECT_WORKLOG.md](/Users/bilalawan/claw/assignments/baseline_model/PROJECT_WORKLOG.md)

Dataset helper:

- [download_ddxplus.py](/Users/bilalawan/claw/assignments/baseline_model/scripts/download_ddxplus.py)

## Current Notebook Map

Notebook progression:

- [01_one_shot_classifier_baselines.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/01_one_shot_classifier_baselines.ipynb)
  - trains and compares one-shot BASD-style classifiers
  - produces the selected one-shot comparator

- [02_single_agent_sequential_baseline.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/02_single_agent_sequential_baseline.ipynb)
  - earlier sequential baseline
  - useful mainly as historical reference now

- [03_compare_baselines.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/03_compare_baselines.ipynb)
  - compares one-shot and sequential outputs on aligned cases

- [04_single_agent_structured_policy_improvement.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/04_single_agent_structured_policy_improvement.ipynb)
  - first structured-policy sequential improvement
  - cleaner ledger, legality, and shortlisting
  - important intermediate step, but empirically weaker than the current refined version

- [05_single_agent_structured_policy_refinement.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/05_single_agent_structured_policy_refinement.ipynb)
  - current main refined sequential notebook
  - adds anchored deterministic diagnosis state, stronger shortlist scoring, and improved stop/request behavior

- [06_single_agent_budget_scaling.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/06_single_agent_budget_scaling.ipynb)
  - same policy as notebook 05
  - only changes default request budgets to study scaling and plateau behavior

If you are continuing the main sequential-policy line, start from notebook `05` or `06`, not `02`.

## Current State Of The Project

The current best live small-sample sequential result comes from the refined notebook line.

Relevant artifact roots:

- [single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1)
- [single_agent_refined_live_test_1perclass_4budgets_anchor_guard_budget_scaling_v1](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_budget_scaling_v1)

Important caution:

- these later live sequential results were run with `gpt-5.4-mini`
- earlier structured-policy notebook 04 live runs used `gpt-4.1-mini`
- so backbone and policy changed together unless an ablation is run explicitly

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download the DDXPlus dataset into `.data/`:

```bash
python3 scripts/download_ddxplus.py
```

The dataset is not committed to git. The repo assumes:

- `.data/ddxplus/22687585/`

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
- `artifacts/sequential_single_agent/`
- `artifacts/sequential_single_agent_improved/`
- `artifacts/sequential_single_agent_refined/`
- `artifacts/comparisons/`
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

- [PROJECT_WORKLOG.md](/Users/bilalawan/claw/assignments/baseline_model/PROJECT_WORKLOG.md)

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

- [reports](/Users/bilalawan/claw/assignments/baseline_model/reports)

Do this when:

- a notebook produced an important new result
- a debugging pass revealed a major failure mode
- a policy change materially improved results

### 4. Secrets rule

Never commit:

- real API keys
- notebook outputs containing real API keys
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

- read [README.md](/Users/bilalawan/claw/assignments/baseline_model/README.md)
- read [PROJECT_WORKLOG.md](/Users/bilalawan/claw/assignments/baseline_model/PROJECT_WORKLOG.md)
- inspect [sequential_policy_refinement_report.md](/Users/bilalawan/claw/assignments/baseline_model/reports/sequential_policy_refinement_report.md)
- inspect [ledger_implementation_explained.md](/Users/bilalawan/claw/assignments/baseline_model/reports/ledger_implementation_explained.md)
- continue from notebook `05` or `06`

## Current Practical Recommendation

For the next clean experiment, do one of:

- run notebook `05` again with a controlled backbone ablation
- run notebook `06` on a larger sample to test whether the apparent `24 -> 32` performance plateau is real

Those two directions are currently the highest-value next steps.
