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

- [11_online_hybrid_mlp_feedback.ipynb](notebooks/11_online_hybrid_mlp_feedback.ipynb)
  - online hybrid v1 system
  - keeps the LLM as the evidence-acquisition controller
  - runs the partial-evidence MLP after every ledger update
  - uses MLP confidence/margin/entropy/agreement signals for stopping and final-head analysis

- [12_stopping_policy_ablation.ipynb](notebooks/12_stopping_policy_ablation.ipynb)
  - offline replay ablation
  - tests whether MLP-guided stopping beats aggressive LLM-only stopping at matched request budgets
  - uses existing notebook `08` and `11` traces only
  - makes no API calls

- [13_live_selected_hybrid_stopping_confirmation.ipynb](notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb)
  - live confirmation of notebook `12`'s selected MLP-guided stop rule
  - keeps the LLM as evidence-acquisition controller
  - tests one selected stop policy instead of another lambda sweep
  - default is safe: no live API calls unless `RUN_LIVE_API = True`

- [14_hybrid_v2_mlp_discriminative_shortlist.ipynb](notebooks/14_hybrid_v2_mlp_discriminative_shortlist.ipynb)
  - candidate hybrid v2 method
  - keeps notebook `13`'s stop policy and final heads
  - changes only the question shortlist to use MLP competing diagnoses and counterfactual entropy reduction
  - includes v1-v2 paired comparison and a promotion decision for whether v2 deserves a 49-case confirmation

If you are continuing the main sequential-policy line, start from notebook `08` for the LLM-only cost-sensitive baseline, notebook `10` for the partial-evidence matched classifier, notebook `11` for hybrid v1, notebook `12` for stopping-policy ablation, notebook `13` for live selected-stop confirmation, notebook `14` for the candidate v2 shortlist experiment, or notebook `09` for integrated evaluation. Use notebooks `05` and `06` as the refined-policy history.

## Current State Of The Project

The current proposed method is notebook `13`: a single-agent LLM workup controller with deterministic ledger state and MLP-guided stopping. Its final 49-case confirmation reaches `43/49 = 0.878` accuracy with `6.59` mean requests. The earlier 24-case pilot reached `22/24 = 0.917`, but the 49-case run is the stronger final number to report.

Relevant artifact roots:

- [basd_pathology_full](artifacts/one_shot/basd_pathology_full)
- [full_evidence_pathology_full](artifacts/one_shot_full_evidence/full_evidence_pathology_full)
- [single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1](artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1)
- [single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1](artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1)
- [hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1](artifacts/sequential_hybrid_mlp_feedback/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1)
- [hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1](artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1)
- [stopping_policy_ablation_24case_v1](artifacts/stopping_policy_ablation/stopping_policy_ablation_24case_v1)
- [selected_stop_live_confirmation_dryrun_smoke_v1](artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_dryrun_smoke_v1)
- [selected_stop_live_confirmation_24case_v1](artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1)
- [selected_stop_live_confirmation_49case_v1](artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1)
- [hybrid_v2_mlp_discriminative_shortlist_24case_v1](artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_24case_v1)

Current headline results:

- initial-evidence one-shot full test accuracy: about `0.378`
- full-evidence one-shot full test accuracy: about `0.996`
- cost-sensitive sequential live 10-case pilot accuracy: `0.900`
- cost-sensitive sequential mean requests dropped from `18.4` to `11.8` across the lambda sweep without reducing pilot accuracy
- cost-sensitive sequential 24-case wide sweep found the cutoff: `0.917` accuracy at lambda `0.10`, `0.875` at lambdas `0.22` and `0.35`, then collapse to `0.417` at lambda `0.50`
- old matched-evidence fallback on the same 24-case live slice reached `0.625` to `0.708` at useful lambdas
- new partial-evidence matched comparator reached `0.875` at lambdas `0.10` and `0.22`, and `0.833` at lambda `0.35`
- hybrid v1 reached `0.875` accuracy at lambda `0.22` with `7.5` mean requests, matching notebook `08` accuracy at that lambda while reducing requests from `10.7`
- notebook `09` now includes evidence-budget views, showing accuracy/top-k against actual mean requested fields and visible evidence roots rather than lambda alone
- hybrid v1 did not improve final diagnosis over the individual LLM/MLP heads; its useful contribution so far is evidence-efficiency/stopping, not better adjudication
- notebook `12` shows MLP-guided stopping is stronger than the best tested pure LLM-only stop rule at a matched request budget: `22/24` correct at `6.9` requests versus `20/24` correct at `6.3` requests
- notebook `12` also shows the high-accuracy notebook `08` result can be preserved offline with roughly half the requests: `22/24` at `6.9` requests versus `22/24` at `13.0` requests
- notebook `12` also found a higher-budget offline MLP-final point: `23/24` correct at about `9.8` requests, suggesting a possible accuracy-biased hybrid operating mode
- notebook `13` live confirmation preserved `22/24` accuracy with `6.58` mean requests, about `49.5%` fewer requests than notebook `08`
- notebook `13` 49-case confirmation reached `43/49 = 0.878` accuracy, `0.939` top-5, and `6.59` mean requests
- notebook `14` tested MLP-discriminative question shortlisting and was rejected by the promotion rule: `21/24` accuracy with `7.38` mean requests, despite improved top-5 and fixing Pericarditis
- current interpretation: evidence acquisition is clearly useful; final diagnosis should be treated as a separate design choice between LLM, partial-evidence classifier, and conservative hybrid adjudication

## Best Current Research Direction

The strongest defensible direction is not "LLM reasoning beats every direct classifier." The current evidence says something more specific and more useful: a structured sequential policy can choose a small, targeted subset of DDXPlus evidence that makes diagnosis much easier than initial evidence alone. The partial-evidence classifier then shows that much of the value may come from acquiring the right evidence, not necessarily from the LLM being the best final diagnostic head.

The newest stopping-policy ablation strengthens the hybrid direction. It suggests the partial-evidence MLP is useful as an online stopping signal: on fixed replay trajectories, MLP-guided stopping preserved `22/24` accuracy at about `6.9` requests, while the best pure LLM-only stop rule at the same target budget preserved only `20/24`. Notebook `13` then confirmed this live on 24 cases and was later rerun on 49 cases, reaching `43/49` with nearly the same mean request count.

The next research version should therefore frame the system as a diagnostic workup controller:

- the sequential LLM/policy decides what evidence to request, using the ledger, legality rules, one-shot prior, and cost-sensitive stopping
- the final diagnosis can be produced by the LLM, the partial-evidence neural classifier, or a hybrid rule that adjudicates disagreements
- the main scientific question becomes whether targeted sequential evidence acquisition can approach the full-evidence ceiling while revealing only a limited subset of fields

This is the cleanest path into the later multi-agent system. Multi-agent work should add specialized evidence-gathering roles and coordination on top of this evidence-acquisition framing, not replace it with unconstrained debate.

Important caution:

- the cost-sensitive sequential result is now stronger than the 10-case pilot, but the 24-case run is still not a final statistical claim
- the selected MLP-guided stop rule is confirmed live on both 24-case and 49-case slices
- notebook `14` shows direct MLP-driven shortlisting is not automatically better; notebook `13` remains the frozen proposed method
- the current unresolved issue is hard-case evidence trajectory quality, especially `Croup` and `Pericarditis`
- older notebook `05` artifacts used `gpt-5.4-mini`; current rigorous comparison phase fixes the sequential backbone to `gpt-4.1-mini`

Latest report:

- [final_report.md](reports/final_report.md)
- [final_results_summary.md](reports/final_results_summary.md)
- [Reports Index](reports/README.md)
- [partial_evidence_matched_comparator.md](reports/baselines/partial_evidence_matched_comparator.md)
- [hybrid_mlp_feedback_report.md](reports/hybrid/hybrid_mlp_feedback_report.md)
- [integrated_evidence_budget_comparison_report.md](reports/baselines/integrated_evidence_budget_comparison_report.md)
- [stopping_policy_ablation_report.md](reports/hybrid/stopping_policy_ablation_report.md)
- [live_selected_hybrid_stopping_confirmation.md](reports/hybrid/live_selected_hybrid_stopping_confirmation.md)

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
- `artifacts/stopping_policy_ablation/`
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
- inspect [final_results_summary.md](reports/final_results_summary.md)
- inspect [final_report.md](reports/final_report.md)
- inspect [stopping_policy_ablation_report.md](reports/hybrid/stopping_policy_ablation_report.md)
- inspect [live_selected_hybrid_stopping_confirmation.md](reports/hybrid/live_selected_hybrid_stopping_confirmation.md)
- inspect [integrated_evidence_budget_comparison_report.md](reports/baselines/integrated_evidence_budget_comparison_report.md)
- continue from notebook `13` if running live selected-stop confirmation, notebook `12` if working on stop-policy evidence, notebook `11` if revisiting hybrid v1, or notebook `09` if updating integrated comparisons

## Current Practical Recommendation

For the next clean experiment:

- treat notebook `13` 49-case confirmation as the current frozen proposed-method result
- if improving the method, focus on hard-case trajectory failures rather than broad new notebooks
- if running more confirmation, keep `gpt-4.1-mini`, `temperature = 0.0`, and `top_p = 1.0`
- after any new live run, rerun/update the integrated comparison reports

Model ablations and multi-agent work are intentionally postponed until this single-agent evaluation phase is cleaner.
