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

- [15_notebook13_stop_policy_sensitivity.ipynb](notebooks/15_notebook13_stop_policy_sensitivity.ipynb)
  - offline analysis of the final notebook `13` 49-case traces
  - inspects requested evidence fields, hard-case trajectories, prediction transitions, and stop-threshold sensitivity
  - makes no API calls

- [16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb](notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb)
  - first MedKGI-style algorithmic graph-ledger analysis
  - builds train-derived DDXPlus evidence/outcome graph statistics
  - replays Notebook `13` traces offline to score evidence questions and graph stop certificates
  - makes no API calls

- [17_live_medkgi_graph_shortlist_pilot.ipynb](notebooks/17_live_medkgi_graph_shortlist_pilot.ipynb)
  - first live MedKGI-style graph-shortlist pilot
  - keeps Notebook `13`'s MLP-guided stop rule fixed
  - replaces only the question shortlist with train-derived graph information-gain top-10 fields
  - default is safe two-case dry-run; set `RUN_LIVE_API = True` for live evaluation

- [18_graph_advisory_hybrid_shortlist.ipynb](notebooks/18_graph_advisory_hybrid_shortlist.ipynb)
  - successor to the rejected Notebook `17` graph-replacement shortlist
  - keeps Notebook `13`'s stop rule and shortlist diversity
  - adds graph information gain, rare disease-specific support, and conservative unsafe-stop overrides as advisory controls
  - default is safe two-case dry-run; set `RUN_LIVE_API = True` for live evaluation

- [19_bayesian_voi_algorithmic_ledger_offline.ipynb](notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb)
  - offline-only Bayesian value-of-information ledger
  - builds train-derived posterior/likelihood tables and fuses them with the partial-evidence MLP
  - selects evidence by expected posterior improvement and stops by confidence plus remaining VOI
  - makes no API calls; full 49-case run completed as a negative ablation and was not promoted

- [20_llm_led_graph_ledger_context.ipynb](notebooks/20_llm_led_graph_ledger_context.ipynb)
  - corrected algorithmic-ledger experiment
  - adapts notebook `13`, not the rejected graph-controller notebooks
  - keeps the LLM as the evidence-request chooser and keeps notebook `13`'s MLP stop rule unchanged
  - adds a compact graph-ledger context block with support, contradiction, unresolved diagnosis pairs, advisory discriminators, and consistency warnings
  - default is a safe two-case dry-run; set `RUN_LIVE_API = True`, `ALLOW_DRY_RUN_BENCHMARK = False`, and `RUN_SCOPE = "pilot24"` for live evaluation

- [21_graph_context_policy_lab.ipynb](notebooks/21_graph_context_policy_lab.ipynb)
  - offline graph-context policy lab
  - replays existing notebook `13`, `17`, `18`, and `20` artifacts
  - tests graph context as a critic, guardrail, adjudicator, and drift detector
  - makes no API calls and trains no model
  - result: no non-oracle variant beats notebook `13`, but graph contradiction strongly flags wrong Notebook `20` top-1 predictions

- [22_graph_posterior_final_adjudicator.ipynb](notebooks/22_graph_posterior_final_adjudicator.ipynb)
  - offline graph-posterior final adjudicator
  - keeps notebook `13` evidence acquisition unchanged
  - computes train-derived signed graph support over the final revealed evidence state
  - conservative graph critic improves the saved 49-case trace from `43/49` to `44/49` with no additional evidence requests
  - makes no API calls

- [23_calibrated_graph_bayes_rescue_reranker.ipynb](notebooks/23_calibrated_graph_bayes_rescue_reranker.ipynb)
  - offline calibrated graph/Bayes/MLP rescue reranker
  - keeps notebook `13` as the first-pass live workup trace
  - trains candidate scoring on train/validate-derived synthetic partial evidence states
  - uses prior recovery, the Notebook `22` graph critic, and up to three graph-Bayes rescue questions for suspicious early stops
  - improves the saved 49-case trace from `43/49` to `47/49` with `6.96` mean requests and zero regressions
  - makes no API calls

- [24_live_graph_bayes_rescue_confirmation.ipynb](notebooks/24_live_graph_bayes_rescue_confirmation.ipynb)
  - live confirmation wrapper for notebook `23`
  - keeps the Notebook `13` live workup loop unchanged
  - applies the frozen graph/Bayes rescue layer after the base stop
  - completed the 49-case live confirmation
  - result: the rescue layer was not promoted, but the fresh live base reached `45/49 = 0.918` with `6.20` mean requests

- [25_live_base_trajectory_replicates.ipynb](notebooks/25_live_base_trajectory_replicates.ipynb)
  - three-replicate runner for the Notebook `13`-style live base workup
  - disables graph/Bayes rescue so the natural live trajectory distribution can be measured cleanly
  - writes `replicate_r01`, `replicate_r02`, and `replicate_r03` artifact roots from one notebook
  - dry-run validated with two cases per replicate

- [26_offline_branching_trajectory_lab.ipynb](notebooks/26_offline_branching_trajectory_lab.ipynb)
  - offline multi-trajectory branching lab over Notebook `13`, Notebook `24`, and Notebook `25` base trajectories
  - measures same-prefix divergence, stop-vs-request fragility, branch oracle ceilings, and label-free branch-trigger/judge variants
  - result: majority vote over five trajectories stays at `43/49`, oracle best-of-five reaches `47/49`, and a diagnostic Notebook `13` base plus sparse two-branch Bayes judge reaches `47/49` with zero regressions
  - makes no API calls and motivated the Notebook `27` prospective live branching confirmation

- [27_live_targeted_branching_confirmation.ipynb](notebooks/27_live_targeted_branching_confirmation.ipynb)
  - prospective live multi-agent branching confirmation, completed
  - runs one Notebook `13`-style base workup, applies the fixed `hybrid_suspicion_v1` terminal trigger, spawns at most two fresh-context full branches only when suspicious, and adjudicates with raw Bayesian posterior plus graph/MLP tie signals
  - result: base branch `42/49`, targeted branching `43/49`, wins `2`, regressions `1`, mean selected requests `6.82`, mean total branch requests `11.45`; not promoted
  - key finding: branches recovered Myocarditis and Panic attack, but the raw-Bayes-only resolver regressed COPD and confidence/contradiction triggers missed consensus wrong-answer cases
  - writes live artifacts under `artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/`

- [28_mlp_gated_confounder_graph_bayes_branching.ipynb](notebooks/28_mlp_gated_confounder_graph_bayes_branching.ipynb)
  - prospective multi-agent candidate, live run completed
  - trains a branch-trigger MLP on train/validate synthetic partial states; branching is now a learned classifier decision rather than a hand-written suspicion rule
  - if the gate fires, spawns up to three fresh-context LLM branches: graph/Bayes scout, confounder-pair scout, and counter-anchor scout
  - final resolver scores base/branch predictions plus graph, Bayes, and MLP pseudo-candidates with base protection for graph/Bayes-supported base answers
  - live result: base branch `42/49`, selected policy `44/49`, wins `2`, regressions `0`, mean selected requests `6.63`, mean total branch requests `9.96`; not promoted
  - post-hoc finding: current scored candidate oracle was only `44/49`, but adding ranked-differential pseudo-candidates lifts the oracle to `47/49` at top-2 and `48/49` at top-3
  - writes dry-run artifacts under `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1/`
  - writes live artifacts under `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1/`

- [29_listwise_differential_graph_bayes_adjudicator.ipynb](notebooks/29_listwise_differential_graph_bayes_adjudicator.ipynb)
  - offline final-head continuation over frozen Notebook `28` live traces
  - explodes base/branch ranked differentials, graph top-5, Bayes top-5, and MLP top-5 into a listwise candidate pool
  - trains an L2 logistic candidate scorer on Notebook `28` train/validate synthetic candidate features and uses a fixed `0.02` validation-derived override margin
  - result: improves Notebook `28` from `44/49` to `45/49` with one win, zero regressions, unchanged selected request cost, and no API calls; not promoted because it does not reach `47/49`
  - key finding: source top-1 plus ranked top-2 has a `47/49` oracle, ranked top-3 has `48/49`, and the full exploded graph/Bayes/MLP/ranked pool contains the true label in all `49/49`
  - writes artifacts under `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`

- [30_hypothesis_forced_differential_branching.ipynb](notebooks/30_hypothesis_forced_differential_branching.ipynb)
  - prospective live branching candidate, live run completed
  - computes explicit challenger hypotheses from the base terminal graph/Bayes/MLP/ranked-differential state
  - if the learned branch gate fires, spawns up to three fresh-context LLM branches, each with an assigned target hypothesis and discriminator roots
  - live result: base branch `42/49`, selected policy `44/49`, wins `2`, regressions `0`, mean selected requests `6.78`, mean total branch requests `12.10`
  - key finding: the broader resolver candidate pool contains the true diagnosis in `49/49` with only `3.98` unique diagnoses per case on average
  - writes dry-run artifacts under `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_dryrun_smoke_v1/`
  - writes live artifacts under `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1/`

- [31_neural_candidate_pool_resolver.ipynb](notebooks/31_neural_candidate_pool_resolver.ipynb)
  - offline neural final-head lab over the completed Notebook `30` candidate pool
  - trains a compact MLP candidate scorer on Notebook `30` train/validate synthetic resolver features
  - selected result: `46/49 = 0.939`, improving Notebook `30` by two cases with zero regressions
  - diagnostic candidate-pool oracle is `49/49`, but this is an upper bound, not a deployable policy
  - writes artifacts under `artifacts/trajectory_replicates/neural_candidate_pool_resolver_49case_v1/`

- [32_resolver_ablation_lab.ipynb](notebooks/32_resolver_ablation_lab.ipynb)
  - offline resolver-ablation lab over the completed Notebook `30` candidate pool and Notebook `31` resolver artifact
  - compares heuristic rank fusion, supervised row models, tree ensembles, disease-name/family features, pairwise ranking, branch-differential voting, and diagnostic non-deployable ceilings
  - best strict validation-selected resolver reaches only `45/49`, so it is not promoted
  - best deployable live-diagnostic ablation is `gradient_boosting_name_family` at `47/49`, but it was identified by inspecting the 49-case ablation outcomes and requires independent confirmation
  - diagnostic candidate-pool oracle remains `49/49`; full-evidence non-deployable rows reach `48/49`
  - writes artifacts under `artifacts/trajectory_replicates/resolver_ablation_lab_49case_v1/`

If you are continuing the main sequential-policy line, start from notebook `08` for the LLM-only cost-sensitive baseline, notebook `10` for the partial-evidence matched classifier, notebook `11` for hybrid v1, notebook `12` for stopping-policy ablation, notebook `13` for live selected-stop confirmation, notebook `14` for the candidate v2 shortlist experiment, notebook `15` for offline stop-threshold/evidence-trajectory analysis, notebook `16` for algorithmic graph-ledger analysis, notebook `17` for rejected hard graph-shortlist testing, notebook `18` for graph-advisory shortlist testing, notebook `19` for Bayesian VOI ledger analysis, notebook `20` for the corrected LLM-led graph-ledger context experiment, notebook `21` for offline graph-context adjudication analysis, notebook `22` for graph-posterior final adjudication over Notebook `13` traces, notebook `23` for calibrated graph/Bayes rescue reranking, notebook `24` for live rescue confirmation, notebook `25` for base trajectory replicate collection, notebook `26` for offline branching trajectory analysis, notebook `27` for prospective live targeted branching confirmation, notebook `28` for MLP-gated confounder branching, notebook `29` for ranked-differential listwise adjudication, notebook `30` for hypothesis-forced branching, notebook `31` for neural candidate-pool resolution, notebook `32` for resolver ablation, or notebook `09` for integrated evaluation. Use notebooks `05` and `06` as the refined-policy history.

## Current State Of The Project

The current live proposed workup method is notebook `13`: a single-agent LLM workup controller with deterministic ledger state and MLP-guided stopping. Its frozen 49-case artifact reaches `43/49 = 0.878` accuracy with `6.59` mean requests. A fresh Notebook `13`-style live base run inside notebook `24` reached `45/49 = 0.918` with `6.20` mean requests. Notebook `23` remains the strongest offline deterministic graph/Bayes rescue candidate, reaching `47/49 = 0.959` with `6.96` mean requests on the saved 49-case trace, but notebook `24` did not confirm the rescue layer as a live improvement. Notebook `26` showed that trajectory branching was a credible next direction, and Notebook `27` partially confirmed it live. Notebook `28` improved its same-run base from `42/49` to `44/49` with zero regressions. Notebook `29` then expanded the final candidate pool to ranked differentials and improved Notebook `28` to `45/49` with zero regressions. Notebook `30` completed hypothesis-forced branching and improved its same-run base from `42/49` to `44/49`, but the larger discovery is that its small candidate pool contains the true diagnosis in `49/49`. Notebook `31` trained a compact neural resolver over that candidate pool and reached `46/49` with zero regressions versus Notebook `30`. Notebook `32` ablated alternative resolvers: the strict validation-selected policy did not improve (`45/49`), while the best live-diagnostic deployable candidate reached `47/49` and needs independent confirmation; the `49/49` candidate-pool oracle remains diagnostic only.

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
- [medkgi_style_offline_notebook13_49case_v1](artifacts/graph_algorithmic_ledger/medkgi_style_offline_notebook13_49case_v1)
- [live_medkgi_graph_shortlist_pilot24_dryrun_smoke_v1](artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_dryrun_smoke_v1)
- [live_medkgi_graph_shortlist_pilot24_v1](artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1)
- [live_graph_advisory_hybrid_shortlist_pilot24_dryrun_smoke_v1](artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_dryrun_smoke_v1)
- [live_graph_advisory_hybrid_shortlist_pilot24_v1](artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_v1)
- [bayesian_voi_offline_notebook13_49case_v1_smoke](artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1_smoke)
- [bayesian_voi_offline_notebook13_49case_v1](artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1)
- [llm_led_graph_ledger_context_pilot24_dryrun_smoke_v1](artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_dryrun_smoke_v1)
- [llm_led_graph_ledger_context_pilot24_v1](artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1)
- [graph_context_policy_lab_24case_v1](artifacts/graph_algorithmic_ledger/graph_context_policy_lab_24case_v1)
- [graph_posterior_final_adjudicator_49case_v1](artifacts/graph_algorithmic_ledger/graph_posterior_final_adjudicator_49case_v1)
- [calibrated_graph_bayes_rescue_reranker_49case_v1](artifacts/graph_algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_49case_v1)
- [live_graph_bayes_rescue_confirmation_49case_dryrun_smoke_v1](artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_dryrun_smoke_v1)
- [live_graph_bayes_rescue_confirmation_49case_v1](artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1)
- [notebook13_style_live_base_replicates_dryrun_smoke_v1](artifacts/trajectory_replicates/notebook13_style_live_base_replicates_dryrun_smoke_v1)
- [notebook13_style_live_base_replicates_49case_v1](artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1)
- [offline_branching_trajectory_lab_49case_v1](artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1)
- [live_targeted_branching_confirmation_dryrun_smoke_v1](artifacts/trajectory_replicates/live_targeted_branching_confirmation_dryrun_smoke_v1)
- [live_targeted_branching_confirmation_49case_v1](artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1)
- [mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1](artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1)
- [mlp_gated_confounder_graph_bayes_branching_49case_v1](artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1)
- [listwise_differential_graph_bayes_adjudicator_49case_v1](artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1)
- [hypothesis_forced_differential_branching_dryrun_smoke_v1](artifacts/trajectory_replicates/hypothesis_forced_differential_branching_dryrun_smoke_v1)
- [hypothesis_forced_differential_branching_49case_v1](artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1)
- [neural_candidate_pool_resolver_49case_v1](artifacts/trajectory_replicates/neural_candidate_pool_resolver_49case_v1)
- [resolver_ablation_lab_49case_v1](artifacts/trajectory_replicates/resolver_ablation_lab_49case_v1)

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
- notebook `15` shows simple MLP stop-threshold tuning does not materially improve notebook `13`; the best offline tie keeps `43/49` accuracy and only reduces mean requests from `6.59` to `6.55`
- notebook `16` shows MedKGI-style graph evidence scores are informative: incorrect trajectories had worse mean graph rank (`9.73`) and lower mean information gain (`0.194`) than correct trajectories (`6.61`, `0.247`)
- notebook `16` does not yet prove a better graph stop policy; its main value is justifying a controlled live graph-shortlist pilot while keeping notebook `13`'s stop rule fixed
- notebook `17` live graph-shortlist pilot was rejected: `20/24 = 0.833` accuracy with `6.21` mean requests versus notebook `13`'s `22/24 = 0.917` with `6.58` mean requests
- notebook `17` still produced useful graph diagnostics: requested fields had mean graph rank `1.76`, mean information gain `0.373`, and zero requests outside the graph top-10, but hard graph replacement over-constrained the action space
- notebook `18` graph-advisory pilot improved over notebook `17` but was still rejected: `21/24 = 0.875` accuracy with `7.67` mean requests versus notebook `13`'s `22/24 = 0.917` with `6.58` mean requests
- notebook `18` recovered notebook `17`'s Chagas and Ebola failures, but introduced a new Stable angina failure and did not fix Croup or Pericarditis
- notebook `19` tested an offline Bayesian VOI ledger and was rejected: best fused result was `33/49 = 0.673` with `22.37` mean requests, far below notebook `13`'s `43/49 = 0.878` with `6.59` mean requests
- notebook `20` implements the corrected algorithmic-ledger direction: graph context is advisory prompt context while the LLM remains the question chooser; the live 24-case result was mixed, with lower top-1 than Notebook `13` but improved top-3/top-5 ranking to `23/24`
- notebook `21` tested whether graph context can act as a critic or adjudicator; it found no promotable non-oracle rule, but confirmed that graph contradiction strongly flags suspect Notebook `20` top-1 predictions
- notebook `22` tested a different graph role: final-state graph-posterior adjudication over Notebook `13` traces. The fixed conservative graph critic improved the saved 49-case result from `43/49 = 0.878` to `44/49 = 0.898` with the same `6.59` mean requests and no regressions.
- post-run Notebook `22` analysis shows the selected critic fires exactly once on the 49-case trace, fixing Croup; remaining failures are COPD/Myocarditis, acute-vs-chronic rhinosinusitis, Influenza/HIV initial infection, Pericarditis/Anemia, and Unstable angina/Anemia
- the larger opportunity is a learned/calibrated reranker: Notebook `13` top-1 plus graph top-3 has a `47/49` oracle ceiling, and Notebook `13` top-1 plus graph top-5 has a `48/49` oracle ceiling, so the graph signal is strong enough for a meaningful improvement if the selector can be learned without tuning on the six misses
- notebook `23` realizes that direction offline: calibrated graph/Bayes rescue reaches `47/49 = 0.959`, adds only `18` total rescue requests across `49` cases, raises mean requests from `6.59` to `6.96`, and has zero regressions against Notebook `13`
- notebook `24` completed the 49-case live confirmation; the fresh live base reached `45/49 = 0.918` with `6.20` mean requests, while the graph/Bayes rescue layer stayed at `45/49 = 0.918` with `6.39` mean requests and was not promoted
- notebook `25` completed three rescue-disabled live base replicates: `44/49`, `42/49`, and `42/49`, showing concentrated but meaningful trajectory variability
- notebook `26` analyzed five observed base trajectories: majority vote stayed at `43/49`, oracle best-of-five reached `47/49`, and a diagnostic sparse two-branch Bayes-judged policy over the Notebook `13` base reached `47/49` with zero regressions
- notebook `27` live targeted branching reached `43/49`, improving its own live base by one case but introducing one regression
- notebook `28` live MLP-gated branching reached `44/49`, improving its own live base by two cases with zero regressions; post-hoc ranked-differential analysis shows a `47/49` top-2 and `48/49` top-3 candidate-pool oracle
- notebook `30` live hypothesis-forced branching reached `44/49`, improving its same-run base by two cases with zero regressions, but mean total branch requests rose to `12.10`
- notebook `30`'s more important finding is candidate-pool recall: the selected ranked differential top-5 contains `47/49`, and the broader resolver candidate pool contains all `49/49` true diagnoses with about `4` unique diagnoses per case
- notebook `31` trains a compact neural resolver over that Notebook `30` candidate pool and reaches `46/49` with zero regressions versus Notebook `30`; it still does not achieve the `47/49+` target as an actual selected policy
- notebook `32` turns the resolver into an ablation lab: strict validation selection picks a `45/49` resolver, but the best deployable live-diagnostic row reaches `47/49`; treat that row as a confirmation candidate, not a promoted policy
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
- notebook `17` shows pure MedKGI-style graph replacement shortlisting is also not automatically better; graph scores should be used as advisory/blended signals rather than a hard top-10 action-space replacement
- notebook `18` shows graph-advisory blending is safer than hard graph replacement, but still does not beat notebook `13`; graph information is currently strongest as an audit/explanation layer
- notebook `19` shows posterior-level Bayesian VOI is useful as an audit idea but not as a replacement controller in this version; it over-selected generic evidence and pushed the partial-evidence MLP into overconfident out-of-distribution states
- notebook `20` is the corrected graph-ledger test: it does not replace the LLM controller, and its live `pilot24` result shows graph context is useful for ranking but not yet top-1 selection
- notebook `21` says not to proceed to a live graph-context hand-threshold adjudicator from the 24-case pilot alone
- notebook `22` shows a stronger graph direction: use train-derived signed graph support as a final-state posterior critic over Notebook `13`'s acquired evidence. This is an offline final-head enhancement candidate, not a replacement evidence controller.
- notebook `23` realizes the larger graph/Bayes rescue direction: use Notebook `13` as the first-pass workup, then apply calibrated graph/Bayes/MLP rescue only when the saved trace is suspicious. It is the strongest current algorithmic-ledger enhancement but still needs live or held-out confirmation.
- notebook `24` shows that Notebook `23`'s offline rescue gain did not transfer cleanly to a fresh live trajectory; treat it as a strong offline candidate, not the final live method
- notebook `26` shows the next credible multi-agent direction: targeted suspicious-state branching, not broad majority voting
- notebooks `30` and `31` narrow the current unresolved issue further: candidate generation can reach a diagnostic `49/49` oracle on the completed live pool, but selected accuracy is `46/49`, so close-confounder resolution is the remaining bottleneck
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
- [graph_advisory_hybrid_shortlist_report.md](reports/algorithmic_ledger/graph_advisory_hybrid_shortlist_report.md)
- [bayesian_voi_ledger_offline_report.md](reports/algorithmic_ledger/bayesian_voi_ledger_offline_report.md)
- [llm_led_graph_ledger_context_report.md](reports/algorithmic_ledger/llm_led_graph_ledger_context_report.md)
- [graph_context_policy_lab_report.md](reports/algorithmic_ledger/graph_context_policy_lab_report.md)
- [graph_posterior_final_adjudicator_report.md](reports/algorithmic_ledger/graph_posterior_final_adjudicator_report.md)
- [calibrated_graph_bayes_rescue_reranker_report.md](reports/algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md)
- [live_graph_bayes_rescue_confirmation_report.md](reports/algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md)
- [hypothesis_forced_differential_branching_report.md](reports/algorithmic_ledger/hypothesis_forced_differential_branching_report.md)
- [neural_candidate_pool_resolver_report.md](reports/algorithmic_ledger/neural_candidate_pool_resolver_report.md)
- [resolver_ablation_lab_report.md](reports/algorithmic_ledger/resolver_ablation_lab_report.md)

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
- inspect [notebook13_stop_policy_sensitivity_report.md](reports/hybrid/notebook13_stop_policy_sensitivity_report.md)
- inspect [integrated_evidence_budget_comparison_report.md](reports/baselines/integrated_evidence_budget_comparison_report.md)
- inspect [graph_context_policy_lab_report.md](reports/algorithmic_ledger/graph_context_policy_lab_report.md)
- inspect [graph_posterior_final_adjudicator_report.md](reports/algorithmic_ledger/graph_posterior_final_adjudicator_report.md)
- inspect [calibrated_graph_bayes_rescue_reranker_report.md](reports/algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md)
- inspect [live_graph_bayes_rescue_confirmation_report.md](reports/algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md)
- inspect [live_base_trajectory_replicates_report.md](reports/algorithmic_ledger/live_base_trajectory_replicates_report.md)
- inspect [offline_branching_trajectory_lab_report.md](reports/algorithmic_ledger/offline_branching_trajectory_lab_report.md)
- inspect [live_targeted_branching_confirmation_report.md](reports/algorithmic_ledger/live_targeted_branching_confirmation_report.md)
- inspect [mlp_gated_confounder_graph_bayes_branching_report.md](reports/algorithmic_ledger/mlp_gated_confounder_graph_bayes_branching_report.md)
- inspect [listwise_differential_graph_bayes_adjudicator_report.md](reports/algorithmic_ledger/listwise_differential_graph_bayes_adjudicator_report.md)
- inspect [hypothesis_forced_differential_branching_report.md](reports/algorithmic_ledger/hypothesis_forced_differential_branching_report.md)
- inspect [neural_candidate_pool_resolver_report.md](reports/algorithmic_ledger/neural_candidate_pool_resolver_report.md)
- inspect [resolver_ablation_lab_report.md](reports/algorithmic_ledger/resolver_ablation_lab_report.md)
- continue from notebook `32` for resolver ablation and close-confounder adjudication, notebook `31` for neural candidate-pool resolution, notebook `30` for hypothesis-forced live branching, notebook `29` for ranked-differential adjudication, notebook `28` for the latest completed live branching result, notebook `27` for earlier live targeted-branching analysis, notebook `26` for branching trajectory analysis, notebook `25` for base trajectory replicate collection, notebook `13` for the frozen live acquisition method, notebook `24` for live rescue-confirmation analysis, notebook `23` for the strongest offline graph/Bayes rescue candidate, notebook `22` for graph-posterior final adjudication, notebook `15` for offline evidence-trajectory/threshold analysis, notebook `21` for graph-context critic/adjudication analysis, notebook `12` if working on stop-policy evidence, notebook `19` if studying the rejected Bayesian VOI ablation, notebook `11` if revisiting hybrid v1, or notebook `09` if updating integrated comparisons

## Current Practical Recommendation

For the next clean experiment:

- treat notebook `13` 49-case confirmation as the current frozen proposed-method result
- treat notebook `23` as the current strongest offline graph-ledger enhancement candidate: it improves the saved Notebook `13` 49-case trace to `47/49` with `6.96` mean requests and zero regressions
- treat notebook `24` as a completed live confirmation that did not promote the rescue layer, but did show a stronger fresh live base result of `45/49` with `6.20` mean requests
- use notebook `26` as the current branching feasibility analysis: it shows majority vote is insufficient, but sparse suspicious-state branching plus graph/Bayes adjudication can reach the `47/49` diagnostic ceiling on observed trajectories
- note the Notebook `25` replicate quickcheck: strict graph/Bayes final overrides did not fire, but raw graph and Bayes top-1 heads improved the three replicates diagnostically from `44/42/42` to `45/44/44` with zero regressions, supporting graph/Bayes as a branch judge
- treat notebook `27` as a completed partial live confirmation of targeted branching, not a promoted result: branches recovered Myocarditis and Panic attack, but the selected raw-Bayes-only resolver regressed COPD and reached only `43/49`
- treat notebook `28` as a completed learned-gate branching live test, not a promoted result: it improved its own base from `42/49` to `44/49` with zero regressions, but the scored candidate pool itself had only a `44/49` oracle
- treat notebook `29` as a completed offline ranked-differential adjudicator, not a promoted result: it improves Notebook `28` to `45/49` with zero regressions, while showing that ranked top-2/top-3 candidate availability has a `47/49` to `48/49` oracle
- the Notebook `29` rerun/post-hoc analysis sharpens this: graph top-2 and Bayes top-2 each have `49/49` oracle coverage, while graph top-1 and Bayes top-1 reach only `45/49`
- treat notebook `30` as a completed hypothesis-forced branching live test: it improves its own base from `42/49` to `44/49` with zero regressions, but most selected answers still come from the base or graph/Bayes pseudo-candidates rather than real LLM branches
- treat notebook `31` as the current strongest final-head result over the Notebook `30` pool: it reaches `46/49` with zero regressions against Notebook `30`, while confirming a diagnostic `49/49` candidate-pool oracle
- treat notebook `32` as the current resolver workflow: the validation-selected policy does not beat Notebook `31`, but the `gradient_boosting_name_family` ablation reaches `47/49` diagnostically and should be independently confirmed before promotion
- the next clean experiment should focus on close-confounder adjudication or cheap targeted discriminator questions rather than candidate-pool expansion alone
- use notebook `25` if collecting more live base trajectories; it runs rescue-disabled base replicates for the branching/divergence lab
- still present notebook `13` as the live evidence-acquisition backbone; present Notebook `23` as an offline enhancement candidate rather than a live-confirmed replacement
- if improving the method after Notebook `32`, focus on independent confirmation of the `47/49` gradient-boosting resolver candidate and then pairwise/abstaining adjudication for silent consensus-wrong anchors rather than broad new controller replacements or more indiscriminate branching
- do not launch a live graph-context hand-threshold adjudicator from Notebook `21`; the successful Notebook `22` direction is final-state graph posterior scoring over Notebook `13` traces, not another graph-context controller
- do not train or report a calibrated graph adjudicator from only the current Notebook `21` 24-case artifacts; the Notebook `20` graph-context trace covers the same 24-case pilot slice and all 24 cases are already contained in the 49-case Notebook `13` confirmation, so there is no clean held-out graph-context development/test split
- if continuing graph work, use graph context as a calibrated/learned critic or adjudicator rather than as a replacement controller, but first create or reserve separate development traces for validation
- for the course deliverable, defend Notebook `13` as the live evidence-acquisition method and present Notebook `23` as the strongest offline mathematical graph/Bayes rescue enhancement, with Notebook `22` as the simpler graph-posterior critic ablation
- if running more confirmation, keep `gpt-4.1-mini`, `temperature = 0.0`, and `top_p = 1.0`
- after any new live run, rerun/update the integrated comparison reports
- do not promote Notebook `19`; if using Bayesian VOI again, use it as advisory scoring or retrain the partial-evidence MLP on VOI-generated trajectories first

Model ablations and multi-agent work are intentionally postponed until this single-agent evaluation phase is cleaner.
