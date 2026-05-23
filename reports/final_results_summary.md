# Final Results Summary

## Scope

This report summarizes the final comparison artifacts currently available for the DDXPlus diagnostic workup project.

Final artifact roots used:

- `artifacts/one_shot/basd_pathology_full/`
- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/`
- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1/`
- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`
- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/`
- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_24case_wide_sweep_v1/`
- `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`
- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_partial_policy_v1/`
- `artifacts/sequential_hybrid_mlp_feedback/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1/`
- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/`
- `artifacts/stopping_policy_ablation/stopping_policy_ablation_24case_v1/`
- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_dryrun_smoke_v1/`
- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`
- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`
- `artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_24case_v1/`
- `artifacts/stop_policy_sensitivity/notebook13_49case_v1/`
- `artifacts/graph_algorithmic_ledger/medkgi_style_offline_notebook13_49case_v1/`
- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/`
- `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_v1/`
- `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1/`
- `artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1/`
- `artifacts/graph_algorithmic_ledger/graph_context_policy_lab_24case_v1/`
- `artifacts/graph_algorithmic_ledger/graph_posterior_final_adjudicator_49case_v1/`
- `artifacts/graph_algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_49case_v1/`
- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_dryrun_smoke_v1/`
- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1/`
- `artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1/`
- `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/`
- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_dryrun_smoke_v1/`
- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/`
- `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1/`
- `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1/`
- `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`
- `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_dryrun_smoke_v1/`
- `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1/`
- `artifacts/trajectory_replicates/neural_candidate_pool_resolver_49case_v1/`
- `artifacts/trajectory_replicates/resolver_ablation_lab_49case_v1/`
- `artifacts/trajectory_replicates/close_confounder_discriminator_49case_v1/`
- `artifacts/trajectory_replicates/candidate_recall_gated_branching_efficiency_49case_v1/`
- `artifacts/trajectory_replicates/adaptive_value_branching_controller_49case_v1/`
- `artifacts/trajectory_replicates/adaptive_branching_stress_test_49case_v1/`
- `artifacts/trajectory_replicates/adaptive_value_branching_live_balanced2_v1/`
- `artifacts/trajectory_replicates/adaptive_value_branching_live_calibration196_v1/`

The strongest frozen live first-pass workup artifact remains notebook `13`: it uses the LLM as the evidence-acquisition controller and the partial-evidence MLP as an online stopping signal. The original 49-case confirmation reaches `43/49 = 0.878` accuracy with `6.59` mean evidence requests.

Current graph-ledger update on 2026-05-08: Notebook `23` is the strongest offline algorithmic-ledger enhancement. It keeps Notebook `13` as the first-pass workup, uses train/validate-calibrated graph/Bayes/MLP rescue logic, and improves the saved 49-case trace from `43/49` to `47/49` with `6.96` mean requests and zero regressions. Notebook `24` completed the live confirmation run, but the rescue layer did not reproduce the offline gain: the fresh live base reached `45/49 = 0.918`, and the live rescue also ended at `45/49 = 0.918` with `6.39` mean total requests.

Trajectory-branching update on 2026-05-20: Notebook `25` collected three rescue-disabled Notebook `13`-style base replicates, and Notebook `26` used those trajectories plus the original Notebook `13` and Notebook `24` base runs for an offline branching lab. Across five observed trajectories, majority vote remains `43/49`, but oracle best-of-five reaches `47/49`. Notebook `27` completed the prospective live confirmation: targeted branching improved its own live base from `42/49` to `43/49`, but introduced one regression, had an actual branch-candidate oracle ceiling of `44/49`, and was not promoted. Notebook `28` then tested a learned branch-trigger MLP, up to three fresh LLM branches, graph/Bayes/MLP pseudo-candidates, and a calibrated resolver with base protection. It improved its own live base from `42/49` to `44/49` with zero regressions, but did not reach the `47/49` promotion target. Notebook `29` expanded the final candidate pool to ranked differentials and improved Notebook `28` to `45/49` with one win and zero regressions. Notebook `30` tested hypothesis-forced branching and improved its same-run base from `42/49` to `44/49` with zero regressions. Its most important finding is that the broader resolver candidate pool contains all `49/49` true labels with only about four unique diagnoses per case. Notebook `31` trained a compact neural resolver over that pool and reached `46/49` with zero regressions against Notebook `30`. Notebook `32` ablated resolver alternatives: strict validation selection chose a `45/49` policy, while the best deployable live-diagnostic ablation reached `47/49` and requires independent confirmation. Notebook `33` adds targeted close-confounder evidence over the fixed Notebook `32` GBM row and reaches `48/49` with zero regressions and `12` total extra evidence requests. Notebook `34` then prunes the saved branch pool: one highest-priority hypothesis branch at trigger threshold `0.80` preserves `49/49` candidate-pool recall and `48/49` final accuracy while reducing mean total branch requests from `12.35` to `8.98`. Notebook `35` reframes this as an adaptive value controller: max three branches are allowed, but branch 2/3 launch only when a label-free continuation-value score crosses `0.40`. On the saved replay it chooses one branch for each high-trigger case and keeps the same `49/49` candidate-pool recall, `48/49` final accuracy, and `8.98` mean total requests. Notebook `36` stress-tests this design and shows the saved 49-case pool cannot prove natural branch-2/3 rescue. Notebook `37` then live-confirms the architecture on a fresh balanced 98-case cohort, improving the base from `83/98` to `88/98` but with only `92/98` candidate-pool recall. Notebook `38` runs a larger 196-case live calibration cohort with lower exploratory trigger thresholds: final accuracy improves from `172/196` to `184/196`, candidate-pool recall reaches `194/196`, and branch 2/3 launch naturally on a small subset of hard cases. Because Notebook `38` is a calibration run, it should be used to freeze the next confirmation policy rather than reported as final held-out performance.

## Headline Results

| System | Cases | Evidence visible | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---|---:|---:|---:|---:|---:|
| Initial-evidence one-shot, full test split | 134,529 | Age, sex, initial evidence only | 0.378 | 0.615 | 0.730 | 0.373 | 0 |
| Full-evidence one-shot, full test split | 134,529 | Age, sex, all evidence | 0.996 | 1.000 | 1.000 | 0.995 | all fields |
| Initial-evidence one-shot, live 10-case slice | 10 | Age, sex, initial evidence only | 0.300 | 0.400 | 0.400 | not primary | 0 |
| Cost-sensitive sequential, lambda 0.00 | 10 | Sequentially requested evidence | 0.900 | 0.900 | 0.900 | 0.867 | 18.4 |
| Cost-sensitive sequential, lambda 0.22 | 10 | Sequentially requested evidence | 0.900 | 0.900 | 0.900 | 0.818 | 11.8 |
| Initial-evidence one-shot, live 24-case slice | 24 | Age, sex, initial evidence only | 0.333 | 0.542 | 0.625 | not primary | 0 |
| Cost-sensitive sequential, lambda 0.10 | 24 | Sequentially requested evidence | 0.917 | 0.917 | 0.917 | 0.846 | 13.0 |
| Cost-sensitive sequential, lambda 0.35 | 24 | Sequentially requested evidence | 0.875 | 0.875 | 0.875 | 0.813 | 8.3 |
| Cost-sensitive sequential, lambda 0.50 | 24 | Sequentially requested evidence | 0.417 | 0.625 | 0.750 | 0.274 | 2.2 |
| Matched-evidence one-shot, best live lambda slice | 10 | Same evidence acquired by sequential policy | 0.700 | 0.900 | 1.000 | 0.538 | same as sequential |
| Matched-evidence one-shot, best 24-case lambda slice | 24 | Same evidence acquired by sequential policy | 0.708 | 0.792 | 0.917 | 0.575 | same as sequential |
| Partial-evidence one-shot, policy masks | 39,998 | Initial evidence plus policy-shaped sampled evidence | 0.515 | 0.741 | 0.827 | 0.519 | sampled |
| Partial matched one-shot, lambda 0.10 slice | 24 | Same evidence acquired by sequential policy | 0.875 | 1.000 | 1.000 | 0.778 | same as sequential |
| Partial matched one-shot, lambda 0.22 slice | 24 | Same evidence acquired by sequential policy | 0.875 | 1.000 | 1.000 | 0.778 | same as sequential |
| Hybrid v1, lambda 0.10 | 24 | Sequentially requested evidence plus online MLP feedback | 0.833 | 1.000 | 1.000 | 0.756 | 9.7 |
| Hybrid v1, lambda 0.22 | 24 | Sequentially requested evidence plus online MLP feedback | 0.875 | 0.958 | 0.958 | 0.813 | 7.5 |
| Hybrid v1, lambda 0.35 | 24 | Sequentially requested evidence plus online MLP feedback | 0.833 | 0.917 | 0.917 | 0.744 | 5.9 |
| Offline ablation, best pure LLM-only stop | 24 | Same notebook 08 replay trajectory | 0.833 | 0.917 | 0.958 | 0.767 | 6.3 |
| Offline ablation, selected MLP-guided stop | 24 | Same notebook 08 replay trajectory | 0.917 | 0.917 | 0.917 | 0.867 | 6.9 |
| Offline ablation, best higher-budget MLP-final | 24 | Same notebook 08 replay trajectory | 0.958 | 1.000 | 1.000 | 0.920 | 9.8 |
| Live selected MLP-stop confirmation, notebook 13 | 24 | Sequentially requested evidence plus online MLP stop signal | 0.917 | 0.917 | 0.917 | 0.867 | 6.6 |
| Live selected MLP-stop confirmation, notebook 13 | 49 | Sequentially requested evidence plus online MLP stop signal | 0.878 | 0.918 | 0.939 | 0.845 | 6.6 |
| Hybrid v2 MLP-discriminative shortlist, notebook 14 | 24 | MLP-guided shortlist plus notebook 13 stop signal | 0.875 | 0.958 | 0.958 | 0.840 | 7.4 |
| MedKGI hard graph shortlist, notebook 17 | 24 | Graph top-10 shortlist plus notebook 13 stop signal | 0.833 | 0.833 | 0.875 | 0.744 | 6.2 |
| Graph-advisory shortlist, notebook 18 | 24 | Notebook 13 shortlist plus graph advisory scoring | 0.875 | 0.875 | 0.917 | 0.795 | 7.7 |
| Bayesian VOI offline ledger, notebook 19 best fused | 49 | Offline train-derived posterior/VOI policy | 0.673 | 0.837 | 0.878 | 0.620 | 22.4 |
| LLM-led graph context, notebook 20 | 24 | Notebook 13 controller plus graph prompt context | 0.833 | 0.958 | 0.958 | 0.744 | 6.1 |
| Graph-context policy lab, notebook 21 best non-oracle | 24 | Offline graph critic/adjudication replay | 0.917 | 0.917 | 0.917 | 0.889 | 6.6 |
| Graph posterior final critic, notebook 22 | 49 | Same Notebook 13 evidence plus graph final adjudication | 0.898 | 0.939 | 0.939 | 0.867 | 6.6 |
| Calibrated graph-Bayes rescue, notebook 23 | 49 | Same Notebook 13 evidence plus offline rescue requests | 0.959 | n/a | n/a | n/a | 7.0 |
| Live graph-Bayes rescue confirmation, notebook 24 dry-run | 2 | Dry-run smoke only | 1.000 | 1.000 | 1.000 | 1.000 | 6.0 |
| Live Notebook 13-style base inside notebook 24 | 49 | Fresh live sequential workup before rescue | 0.918 | 0.939 | 0.939 | 0.895 | 6.2 |
| Live graph-Bayes rescue confirmation, notebook 24 | 49 | Fresh live workup plus deterministic rescue layer | 0.918 | 0.939 | 0.939 | 0.895 | 6.4 |
| Live Notebook 13-style base replicate r01, notebook 25 | 49 | Fresh live base replicate, rescue disabled | 0.898 | 0.939 | 0.939 | 0.871 | 7.0 |
| Live Notebook 13-style base replicate r02, notebook 25 | 49 | Fresh live base replicate, rescue disabled | 0.857 | 0.918 | 0.918 | 0.816 | 7.0 |
| Live Notebook 13-style base replicate r03, notebook 25 | 49 | Fresh live base replicate, rescue disabled | 0.857 | 0.939 | 0.939 | 0.810 | 6.8 |
| Offline branching lab oracle best-of-five, notebook 26 | 49 | Diagnostic upper bound over observed base trajectories | 0.959 | n/a | n/a | n/a | n/a |
| Offline branching lab, Notebook 13 base + sparse 2-branch Bayes judge | 49 | Diagnostic observed-branch policy, not live-promoted | 0.959 | n/a | n/a | n/a | 6.9 selected / 9.2 total branch requests |
| Live targeted branching confirmation, notebook 27 | 49 | Prospective fixed live multi-agent branching; not promoted | 0.878 | 0.918 | 0.939 | 0.837 | 6.8 selected / 11.4 total branch requests |
| MLP-gated confounder branching, notebook 28 dry-run | 2 | Learned branch gate and graph/Bayes/MLP resolver; smoke only | 1.000 | 1.000 | 1.000 | 1.000 | 6.0 selected / 6.0 total branch requests |
| MLP-gated confounder branching, notebook 28 live | 49 | Learned branch gate, three fresh LLM branches, pseudo-candidates, calibrated resolver; not promoted | 0.898 | 0.959 | 0.959 | 0.864 | 6.6 selected / 10.0 total branch requests |
| Listwise differential adjudicator, notebook 29 offline | 49 | Frozen Notebook 28 live traces plus ranked-differential graph/Bayes/MLP candidate scoring; not promoted | 0.918 | n/a | n/a | n/a | 6.6 selected / 10.0 total branch requests |
| Hypothesis-forced differential branching, notebook 30 live | 49 | Explicit challenger-hypothesis LLM branches plus graph/Bayes/MLP pseudo-candidates; not promoted | 0.898 | 0.939 | 0.959 | 0.871 | 6.8 selected / 12.1 total branch requests |
| Neural candidate-pool resolver, notebook 31 offline | 49 | Frozen Notebook 30 candidate pool plus compact neural resolver; offline follow-up candidate | 0.939 | n/a | n/a | n/a | 6.8 selected / 12.1 total branch requests |
| Resolver ablation lab, notebook 32 live-diagnostic best | 49 | Frozen Notebook 30 candidate pool plus deployable gradient-boosting resolver candidate; needs independent confirmation | 0.959 | n/a | n/a | n/a | 6.8 selected / 12.1 total branch requests |
| Resolver ablation lab, notebook 32 validation-selected | 49 | Strict validation-selected resolver over the same pool; not promoted | 0.918 | n/a | n/a | n/a | 6.8 selected / 12.1 total branch requests |
| Close-confounder discriminator, notebook 33 offline | 49 | Frozen Notebook 30/32 candidate pool plus two-root targeted confounder discriminator; needs independent confirmation | 0.980 | n/a | n/a | n/a | 7.0 selected / 12.3 total branch requests |
| Candidate-recall-gated branch pruning, notebook 34 offline | 49 | Notebook 30/33 replay with one highest-priority branch and threshold 0.80; live confirmation candidate | 0.980 | n/a | n/a | n/a | 7.2 selected / 9.0 total branch requests |
| Adaptive value branch controller, notebook 35 offline | 49 | Notebook 30/33 replay with max 3 branches but continuation-value gating; live confirmation candidate | 0.980 | n/a | n/a | n/a | 7.2 selected / 9.0 total branch requests |
| Adaptive live balanced confirmation, notebook 37 base | 98 | Fresh two-per-pathology live Notebook 13-style base branch | 0.847 | 0.888 | 0.908 | n/a | 8.2 selected / 8.2 total branch requests |
| Adaptive live balanced confirmation, notebook 37 final | 98 | Fresh live adaptive candidate pool plus GBM resolver and close-confounder discriminator | 0.898 | 0.939 | 0.939 | n/a | 8.4 selected / 8.4 total branch requests |
| Adaptive live calibration, notebook 38 base | 196 | Four-per-pathology live calibration base branch | 0.878 | 0.944 | 0.964 | n/a | 6.8 selected / 6.8 total branch requests |
| Adaptive live calibration, notebook 38 final | 196 | Calibration-only adaptive candidate pool plus GBM resolver and close-confounder discriminator | 0.939 | 0.990 | 0.990 | n/a | 6.8 selected / 9.6 total branch requests |
| Full-evidence one-shot, live 10-case slice | 10 | All evidence | 1.000 | 1.000 | 1.000 | 1.000 | all fields |
| Full-evidence one-shot, live 24-case slice | 24 | All evidence | 1.000 | 1.000 | 1.000 | 1.000 | all fields |

## Full-Evidence One-Shot Comparator

The full-evidence direct model reached near-ceiling performance:

- validation accuracy: `0.9954`
- validation macro-F1: `0.9943`
- test accuracy: `0.9958`
- test top-3 accuracy: `1.0000`
- test top-5 accuracy: `1.0000`
- test macro-F1: `0.9948`

This establishes that DDXPlus contains enough structured evidence for highly accurate diagnosis when the relevant evidence is visible.

A duplicate robustness check was also run because exact duplicate rows exist across the official train/validate/test files. The official metrics are kept for comparability, but validation/test rows whose raw-row or feature signatures appeared in training were filtered and rescored.

Deduplicated test results remained essentially unchanged:

| Dedup type | Test rows removed | Duplicate fraction | Dedup accuracy | Dedup top-3 | Dedup top-5 | Dedup macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Raw row signature | 1,823 | 0.0136 | 0.9958 | 1.0000 | 1.0000 | 0.9948 |
| Feature signature | 1,989 | 0.0148 | 0.9958 | 1.0000 | 1.0000 | 0.9948 |

Interpretation: cross-split duplicate contamination exists, but it does not explain the near-ceiling full-evidence score. The full-evidence result is best treated as a ceiling-style comparator showing that the dataset is highly diagnosable from complete structured evidence.

## Cost-Sensitive Sequential Policy

The first live cost-sensitive sequential run used 10 balanced test cases and swept lambda values from `0.00` to `0.22`.

| Lambda | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Stop before cap | Input tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.900 | 0.900 | 0.900 | 0.867 | 18.4 | 0.70 | 429,339 |
| 0.03 | 0.900 | 0.900 | 0.900 | 0.867 | 16.6 | 0.70 | 382,887 |
| 0.06 | 0.900 | 0.900 | 0.900 | 0.867 | 15.7 | 0.80 | 363,486 |
| 0.10 | 0.900 | 0.900 | 0.900 | 0.818 | 14.1 | 0.80 | 323,388 |
| 0.15 | 0.900 | 0.900 | 0.900 | 0.818 | 12.8 | 0.80 | 293,985 |
| 0.22 | 0.900 | 0.900 | 0.900 | 0.818 | 11.8 | 0.80 | 269,060 |

This is the most useful sequential result so far. Accuracy stayed flat at 90% while mean requests dropped from 18.4 to 11.8. That is about a 36% reduction in requested evidence and about a 37% reduction in input tokens, without losing top-1 accuracy on this pilot slice.

The utility column in the raw artifact is `accuracy - lambda * mean_requests`, so it naturally becomes negative at larger lambda values. It is useful as a controller diagnostic, but the clearer scientific result is the accuracy-vs-request curve.

A second live sweep used 24 balanced cases and wider lambda values. This run is more informative because accuracy moves in increments of about `0.042` rather than `0.100`, and the larger lambda values expose the failure point.

| Lambda | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Stop before cap | Input tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.917 | 0.917 | 0.846 | 13.0 | 0.833 | 710,832 |
| 0.22 | 0.875 | 0.875 | 0.917 | 0.795 | 10.7 | 0.917 | 585,943 |
| 0.35 | 0.875 | 0.875 | 0.875 | 0.813 | 8.3 | 0.958 | 456,292 |
| 0.50 | 0.417 | 0.625 | 0.750 | 0.274 | 2.2 | 1.000 | 140,435 |
| 0.75 | 0.375 | 0.583 | 0.708 | 0.288 | 1.0 | 1.000 | 84,875 |

This is now a meaningful cutoff curve. Lambda values `0.10` to `0.35` preserve strong performance while reducing evidence usage. Lambda `0.50` and above stop too early and collapse toward the initial-evidence baseline.

The best accuracy setting is `lambda = 0.10`, with `22/24` correct and about 13 requests per case. The best efficiency-preserving setting is likely `lambda = 0.35`, with `21/24` correct and about 8.3 requests per case. Compared with lambda `0.10`, lambda `0.35` uses about 36% fewer requests for one additional error.

## Integrated Matched-Evidence Comparison

The integrated comparison asks whether the sequential LLM adds value beyond evidence acquisition by comparing it to a direct classifier using the same acquired evidence.

| Lambda | Initial one-shot acc | Sequential acc | Matched-evidence one-shot acc | Full-evidence acc | Mean requests | Sequential full-gain recovered | Matched full-gain recovered |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.300 | 0.900 | 0.600 | 1.000 | 18.4 | 0.857 | 0.429 |
| 0.03 | 0.300 | 0.900 | 0.600 | 1.000 | 16.6 | 0.857 | 0.429 |
| 0.06 | 0.300 | 0.900 | 0.600 | 1.000 | 15.7 | 0.857 | 0.429 |
| 0.10 | 0.300 | 0.900 | 0.700 | 1.000 | 14.1 | 0.857 | 0.571 |
| 0.15 | 0.300 | 0.900 | 0.700 | 1.000 | 12.8 | 0.857 | 0.571 |
| 0.22 | 0.300 | 0.900 | 0.700 | 1.000 | 11.8 | 0.857 | 0.571 |

On this live slice, sequential reasoning outperformed the matched-evidence one-shot classifier by 20 to 30 accuracy points. That suggests the LLM is not merely acquiring evidence; it is also using the acquired evidence more effectively than the current direct matched-evidence comparator.

Caveat: the matched-evidence comparator used for these results reuses the full-evidence one-shot model on partial evidence states. That is a fair first comparator, but it may be distribution-shifted because the model was trained with all evidence visible. Notebook `10` now implements the stronger partial-evidence matched comparator; rerun notebook `09` after notebook `10` to update this comparison.

The 24-case integrated comparison shows the same pattern with a clearer efficiency frontier:

| Lambda | Initial one-shot acc | Sequential acc | Matched-evidence one-shot acc | Full-evidence acc | Mean requests | Sequential full-gain recovered | Matched full-gain recovered |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.917 | 0.625 | 1.000 | 13.0 | 0.875 | 0.438 |
| 0.22 | 0.333 | 0.875 | 0.708 | 1.000 | 10.7 | 0.813 | 0.563 |
| 0.35 | 0.333 | 0.875 | 0.667 | 1.000 | 8.3 | 0.813 | 0.500 |
| 0.50 | 0.333 | 0.417 | 0.333 | 1.000 | 2.2 | 0.125 | 0.000 |
| 0.75 | 0.333 | 0.375 | 0.250 | 1.000 | 1.0 | 0.063 | -0.125 |

Sequential beats the current matched-evidence one-shot fallback at every lambda in this run. At useful lambdas, the sequential advantage is 17 to 29 accuracy points. At very high lambdas, both systems degrade because too little evidence is acquired. This should be rechecked after training the partial-evidence matched comparator in notebook `10`.

Notebook `10` has now trained that stronger partial-evidence comparator.

Standalone partial-mask test performance:

| Model | Test rows | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Partial-evidence one-shot, policy masks | 39,998 | 0.515 | 0.741 | 0.827 | 0.519 |

Integrated 24-case comparison using the partial-evidence matched comparator:

| Lambda | Initial one-shot acc | Sequential acc | Partial matched acc | Partial matched top-3 | Partial matched top-5 | Full-evidence acc | Mean requests |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.917 | 0.875 | 1.000 | 1.000 | 1.000 | 13.0 |
| 0.22 | 0.333 | 0.875 | 0.875 | 1.000 | 1.000 | 1.000 | 10.7 |
| 0.35 | 0.333 | 0.875 | 0.833 | 0.958 | 0.958 | 1.000 | 8.3 |
| 0.50 | 0.333 | 0.417 | 0.458 | 0.708 | 0.792 | 1.000 | 2.2 |
| 0.75 | 0.333 | 0.375 | 0.375 | 0.583 | 0.708 | 1.000 | 1.0 |

This is the most important interpretive update. The old matched fallback understated direct-classifier performance under partial evidence. The partial-evidence comparator nearly matches the sequential LLM at useful lambdas: sequential is ahead by one case at `0.10` and `0.35`, tied at `0.22`, and the matched classifier has stronger top-3/top-5 ranking quality at `0.10` and `0.22`.

## Error Pattern

In the 10-case run, across all lambda settings, the sequential system missed the same case:

- `test:81691`, true pathology `Croup`

The predicted wrong class changed with lambda, but the persistent failure suggests that either:

- the shortlist did not surface the right discriminating pediatric respiratory evidence early enough
- the LLM misinterpreted the revealed evidence pattern
- the stop policy accepted a plausible but incorrect competing diagnosis

This case should be used as the first targeted qualitative debugging example.

In the 24-case run, `Croup` and `Pericarditis` remain the most persistent hard cases. At `lambda = 0.10`, the only two misses are:

- `test:81691`, true pathology `Croup`
- `test:62878`, true pathology `Pericarditis`

At `lambda = 0.35`, the model still misses those two and additionally misses `Allergic sinusitis`. At `lambda = 0.50+`, many cases stop after only one or two requests, so the errors become broad rather than clinically specific.

## Scientific Interpretation

The project is not failing based on these final artifacts. The evidence now supports a cleaner story:

1. DDXPlus has strong diagnostic signal when complete evidence is available.
2. Initial-evidence-only diagnosis is much harder, with the full-test one-shot baseline around 38% accuracy.
3. Controlled sequential evidence acquisition can recover a large fraction of the full-evidence gain on a small live slice.
4. The lambda policy improves evidence efficiency by reducing requests while preserving accuracy in the pilot.
5. The stronger partial-evidence matched classifier nearly matches the sequential LLM on the same acquired evidence, so the clearest value is targeted evidence acquisition rather than an unambiguous LLM final-reasoning advantage.

The main limitation is still sample size, but the 24-case run is much more conclusive than the 10-case pilot. It shows a real tradeoff curve and identifies the cutoff region: `lambda = 0.10` is strongest, `0.22-0.35` are efficient high-performance settings, and `0.50+` is too aggressive. The partial-evidence matched result motivated hybrid v1; the live hybrid run below shows that hybrid feedback currently helps evidence efficiency more than final-head accuracy.

## Hybrid V1 Live Results

Notebook `11` has now run the online MLP feedback system on the same 24-case balanced live slice with `gpt-4.1-mini`, `temperature=0.0`, `top_p=1.0`, request cap 24, and lambdas `[0.10, 0.22, 0.35]`.

| Lambda | Hybrid acc | LLM-final acc | MLP-final acc | Matched MLP acc | Full-evidence acc | Mean requests | Stop before cap | LLM/MLP agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.833 | 0.875 | 0.833 | 0.833 | 1.000 | 9.7 | 0.875 | 0.917 |
| 0.22 | 0.875 | 0.875 | 0.875 | 0.875 | 1.000 | 7.5 | 0.917 | 1.000 |
| 0.35 | 0.833 | 0.833 | 0.833 | 0.833 | 1.000 | 5.9 | 0.958 | 1.000 |

Compared with notebook `08` on the same lambda values:

| Lambda | Notebook 08 acc | Notebook 08 requests | Hybrid acc | Hybrid requests | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.10 | 0.917 | 13.0 | 0.833 | 9.7 | Fewer requests, worse accuracy |
| 0.22 | 0.875 | 10.7 | 0.875 | 7.5 | Same accuracy, about 30% fewer requests |
| 0.35 | 0.875 | 8.3 | 0.833 | 5.9 | Fewer requests, one extra error |

The best hybrid result is `lambda = 0.22`: `21/24` correct with about `7.5` requests per case. This preserves the notebook `08` accuracy at the same lambda while reducing mean requests from `10.7` to `7.5`.

The hybrid final head did not improve over the individual heads. At `lambda = 0.10`, the LLM final alone scored `0.875`, while the hybrid final scored `0.833` because one correct LLM answer was overwritten by a high-confidence but wrong MLP prediction. At `lambda = 0.22` and `0.35`, LLM-final, online MLP-final, hybrid-final, and matched MLP all agree in top-1 accuracy.

Notebook `09` has also been updated to evaluate systems against the actual evidence acquired, not only against lambda. The new evidence-budget artifacts show:

| Lambda | Mean requests | Mean visible roots incl. initial | Hybrid acc | Online MLP acc | Offline matched MLP acc | Hybrid top-5 | Offline matched top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 9.7 | 10.7 | 0.833 | 0.833 | 0.833 | 1.000 | 1.000 |
| 0.22 | 7.5 | 8.5 | 0.875 | 0.875 | 0.875 | 0.958 | 0.958 |
| 0.35 | 5.9 | 6.9 | 0.833 | 0.833 | 0.833 | 0.917 | 0.917 |

Compared with the earlier notebook `08` matched-MLP result, hybrid v1 preserves matched-MLP accuracy at `lambda = 0.22` and `0.35` while using fewer requests. At `lambda = 0.10`, it loses one case because the online hybrid policy stops earlier and gives the MLP less/different evidence.

Current interpretation:

- hybrid v1 is useful as an evidence-efficiency/stopping improvement
- hybrid v1 is not yet a better final-diagnosis adjudicator
- MLP confidence is not calibrated enough to safely override the LLM in disagreements
- the next targeted improvement should make final adjudication more conservative while preserving the `lambda = 0.22` evidence-efficiency gain

Persistent hybrid errors are `Croup`, `Influenza`, and `Pericarditis`. At `lambda = 0.35`, `Chagas` also fails. These cases should be the targeted qualitative debug set for the next policy patch.

## Stopping-Policy Ablation

Notebook `12` was added to answer whether hybrid v1's efficiency gain is really from the partial-evidence MLP providing a better stop signal, or whether any aggressive LLM-only stop rule could do the same.

The notebook is offline only. It replays notebook `08` lambda `0.10` traces, reconstructs turn-level ledger states, runs the partial-evidence MLP locally at each turn, and sweeps stopping policies without making new API calls.

Validation:

- replay rows: `333`
- aligned cases: `24`
- stopping policies: `309`
- policy summary rows: `1,236`
- MLP reconstruction check against notebook `11`: `24/24` matched, `1.000` match rate

Matched-budget result at approximately `7.5` mean requests:

| Stopping family | Final head | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---|---:|---:|---:|---:|---:|---:|
| Best pure LLM-only stop | LLM | 20/24 | 0.833 | 0.917 | 0.958 | 0.767 | 6.33 |
| Best MLP-guided stop | LLM | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |
| Best MLP-guided stop | MLP | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.25 |
| Best MLP-guided stop | conservative hybrid | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |
| Best MLP-guided stop | agreement hybrid | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |

Selected policy:

- `mlp_conf_ge_0.70_margin_ge_0.20_entropy_le_0.10_stab_0`
- selected final head: `agreement_hybrid_final`
- accuracy: `0.9167`
- mean requests: `6.875`

Interpretation:

This is the strongest evidence so far that the MLP contributes a real stopping signal. On fixed evidence trajectories, the best MLP-guided stop rule preserves the notebook `08` high-accuracy result (`22/24`) while using roughly half the requests of the original notebook `08` lambda `0.10` run (`6.9` vs `13.0`). The best pure LLM-only stop rule at the same budget reaches only `20/24`.

The full policy sweep also found a higher-budget offline operating point: an LLM-confidence or deterministic-state stop rule followed by the MLP final head reaches `23/24` at about `9.8-10.0` requests. This is not the main matched-budget answer, but it suggests the partial-evidence MLP may be a strong final diagnostic head once enough targeted evidence has accumulated.

Selected-policy error pattern:

- `test:81691`, true `Croup`, predicted `Acute otitis media` after 7 requests; this was a high-confidence false agreement between LLM and MLP.
- `test:62878`, true `Pericarditis`, predicted `Panic attack` after forced end-of-trace at 23 requests; this looks more like a trajectory/question-selection failure.

Caveat:

This is still offline replay. It proves the stopping signal is useful on already-recorded trajectories, not that a live run with this exact rule will necessarily follow the same question path. The next clean experiment is a live confirmation run using only the selected stop rule.

## Live Selected-Stop Confirmation

Notebook `13` has now run live with notebook `12`'s selected MLP-guided stopping rule.

Notebook:

- `notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb`

Default live artifact root:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

Dry-run smoke artifact:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_dryrun_smoke_v1/`

The notebook is intentionally not a lambda sweep. It tests one selected rule:

- `mlp_conf_ge_0.70_margin_ge_0.20_entropy_le_0.10_stab_0`

Live metrics:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `08`, lambda `0.10` | 22/24 | 0.917 | 0.917 | 0.917 | 0.846 | 13.04 |
| Notebook `11`, lambda `0.22` | 21/24 | 0.875 | 0.958 | 0.958 | 0.813 | 7.46 |
| Notebook `12`, offline selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.875 |
| Notebook `13`, live selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 |

Additional notebook `13` details:

- median requests: `4.5`
- mean visible roots including initial: `7.58`
- stop-before-cap rate: `1.000`
- cap hits: `0`
- selected stop rule fired: `20/24`
- LLM/MLP top-1 agreement: `23/24`
- input tokens: `410,536`
- output tokens: `20,979`

Efficiency:

- versus notebook `08`, notebook `13` uses `49.5%` fewer requests and `42.2%` fewer input tokens at the same accuracy
- versus notebook `11`, notebook `13` uses `11.7%` fewer requests and `9.3%` fewer input tokens while improving accuracy by one case

Final heads:

| Final head | Accuracy | Top-5 |
|---|---:|---:|
| Agreement hybrid | 0.917 | 0.917 |
| Conservative hybrid | 0.917 | 0.917 |
| LLM final | 0.917 | 0.917 |
| MLP final | 0.917 | 0.958 |

The live confirmation meets the preferred acceptance target: `22/24` correct with fewer than `7.5` requests. The main claim is now stronger than offline replay alone: the selected MLP stop signal worked inside the live LLM loop.

Notebook `13` errors:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:81691` | `Croup` | `Chagas` | 23 | selected MLP stop |
| `test:62878` | `Pericarditis` | `Anemia` | 16 | agent stop |

Both failures were wrong for all final heads. This suggests the next bottleneck is not final-head arbitration; it is the evidence trajectory and disease-specific confusion for the hard cases.

## Notebook 13 49-Case Confirmation

After selecting notebook `13` as the frozen proposed method, the same policy was rerun on a broader 49-case balanced test slice.

Artifact:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`

Run settings:

| Setting | Value |
|---|---|
| LLM | `gpt-4.1-mini` |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Cases | 49 |
| Request cap | 24 |
| Stop rule | MLP confidence `>=0.70`, margin `>=0.20`, entropy `<=0.10`, min requests `>=1` |

Main metrics:

| Metric | Value |
|---|---:|
| Agreement-hybrid accuracy | 43/49 = 0.878 |
| LLM-final accuracy | 43/49 = 0.878 |
| MLP-final accuracy | 41/49 = 0.837 |
| Conservative-hybrid accuracy | 43/49 = 0.878 |
| Agreement-hybrid top-3 | 0.918 |
| Agreement-hybrid top-5 | 0.939 |
| Macro-F1 | 0.845 |
| Mean requests | 6.59 |
| Median requests | 5.0 |
| Mean visible roots including initial | 7.59 |
| Stop-before-cap rate | 0.980 |
| Cap hits | 1 |
| Selected stop-rule fired | 36/49 = 0.735 |
| LLM/MLP top-1 agreement | 46/49 = 0.939 |
| Input tokens | 823,478 |
| Output tokens | 42,721 |

Comparison to the original 24-case notebook `13` pilot:

| Run | Correct | Accuracy | Top-5 | Macro-F1 | Mean requests | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13`, 24 cases | 22/24 | 0.917 | 0.917 | 0.867 | 6.58 | 0 |
| Notebook `13`, 49 cases | 43/49 | 0.878 | 0.939 | 0.845 | 6.59 | 1 |

Same-case 49-case framing:

| Comparator | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Initial-evidence one-shot on same 49 cases | 0.286 | 0.673 | 0 |
| Notebook `13` selected hybrid stop on same 49 cases | 0.878 | 0.939 | 6.59 |
| Full-evidence one-shot ceiling on same 49 cases | 0.980 | 1.000 | all fields |

Interpretation:

- accuracy dropped from the very strong 24-case pilot, which is expected when moving to a broader case slice
- mean requests remained essentially unchanged, so the evidence-efficiency result held up
- top-5 improved relative to the 24-case run, which means the correct answer was often still near the top even when top-1 failed
- the system remains far above the initial-evidence one-shot full-test baseline, but it still has a non-trivial gap to the full-evidence ceiling

49-case error cases:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:38475` | `Acute COPD exacerbation / infection` | `Myocarditis` | 24 | max requests reached |
| `test:111176` | `Acute rhinosinusitis` | `Chronic rhinosinusitis` | 8 | selected MLP stop |
| `test:81691` | `Croup` | `Anemia` | 19 | agent stop |
| `test:8666` | `Influenza` | `HIV (initial infection)` | 3 | agent stop |
| `test:62878` | `Pericarditis` | `Anemia` | 15 | agent stop |
| `test:125508` | `Unstable angina` | `Anemia` | 2 | agent stop |

The errors show the remaining bottleneck more clearly. The method is efficient, but it can still stop or converge incorrectly when the LLM and MLP agree on a wrong diagnosis. The hardest failures are not simply request-budget failures; several wrong cases stopped well before the cap with confident but incorrect agreement.

## Hybrid V2 MLP-Discriminative Shortlist

Notebook `14` tested whether the MLP should guide question selection directly, not only stopping. It kept notebook `13`'s selected stop rule and final heads fixed, but replaced the action shortlist with an MLP-discriminative shortlister using MLP top competing diagnoses, train/validate-derived pathology evidence rates, and counterfactual MLP entropy reduction.

Live artifact:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_24case_v1/`

Report:

- `reports/hybrid/hybrid_v2_mlp_discriminative_shortlist_report.md`

Main result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Notebook `13` hybrid v1 selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 | 410,536 |
| Notebook `14` hybrid v2 MLP shortlist | 21/24 | 0.875 | 0.958 | 0.958 | 0.840 | 7.38 | 509,158 |

Promotion decision:

- `reject_keep_notebook13_v1`

Paired outcomes:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| V1 only correct | 2 |
| V2 only correct | 1 |
| Both wrong | 1 |

Case-level result:

- v2 fixed `Pericarditis`, but required all `24` requests
- v2 still failed `Croup` after `23` requests
- v2 introduced new errors on `Chagas` and `Influenza`
- v2 improved top-5 but reduced top-1 and used more evidence

Interpretation:

V2 is an informative negative result. MLP-discriminative shortlisting works mechanically and produces high-separation questions, but direct MLP control of the shortlist can over-focus on unstable or wrong MLP competitors. The current best method remains notebook `13`: LLM-led evidence acquisition with MLP-guided stopping.

## Notebook 15 Stop-Policy Sensitivity

Notebook `15` was added after the 49-case confirmation to test whether simple MLP-threshold tuning could improve Notebook `13` without new API calls.

Notebook:

- `notebooks/15_notebook13_stop_policy_sensitivity.ipynb`

Artifact:

- `artifacts/stop_policy_sensitivity/notebook13_49case_v1/`

Report:

- `reports/hybrid/notebook13_stop_policy_sensitivity_report.md`

Current Notebook `13` rule recovered by offline replay:

| Rule | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| confidence `>=0.70`, margin `>=0.20`, entropy `<=0.10`, min requests `>=1` | 0.878 | 0.939 | 6.59 |

Best offline threshold tie:

| Rule | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| confidence `>=0.55`, margin `>=0.10`, entropy `<=0.05`, min requests `>=0` | 0.878 | 0.939 | 6.55 |

Interpretation:

- no threshold variant improved beyond `43/49`
- the best tie saved only about `0.04` mean requests
- the selected Notebook `13` thresholds are already near the observed offline frontier
- remaining errors are not mainly due to the stop threshold

Evidence-trajectory diagnostics:

| Final correctness | Cases | Mean requests | Median requests |
|---|---:|---:|---:|
| Incorrect | 6 | 11.83 | 11.5 |
| Correct | 43 | 5.86 | 5.0 |

This is important because incorrect cases generally used more requests, not fewer. The remaining bottleneck is therefore wrong belief convergence and evidence trajectory quality, not simply under-questioning.

## Notebook 16 MedKGI-Style Graph Ledger V1

Notebook `16` was added as the first algorithmic graph-ledger version. It is offline-only and replays Notebook `13` traces; it does not call the LLM.

Notebook:

- `notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb`

Artifact:

- `artifacts/graph_algorithmic_ledger/medkgi_style_offline_notebook13_49case_v1/`

Report:

- `reports/algorithmic_ledger/medkgi_style_graph_ledger_v1_report.md`

What it built:

- train-derived DDXPlus evidence graph over `223` root fields and `49` pathologies
- outcome-state probabilities from `1,025,602` training rows
- MedKGI-style information-gain scoring for legal evidence questions
- actual request graph-rank analysis over Notebook `13`
- graph stop-certificate replay
- hard-case graph audits

Evidence-selection result:

| Final outcome | Requests | Mean graph rank | Mean graph score | Mean information gain | Top-10 rate |
|---|---:|---:|---:|---:|---:|
| Incorrect final diagnosis | 71 | 9.73 | 0.233 | 0.194 | 0.676 |
| Correct final diagnosis | 252 | 6.61 | 0.283 | 0.247 | 0.794 |

Interpretation:

- graph scores are meaningful because correct trajectories generally requested more graph-informative evidence
- failed trajectories often retained useful graph evidence value or requested lower-ranked fields
- this supports the idea that the next bottleneck is evidence-selection quality, not simply stop-threshold tuning

Stop-certificate result:

| Threshold | Accuracy | Correct | Top-5 | Mean requests | Graph stops | Terminal fallbacks |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03-0.20 | 0.878 | 43 | 0.939 | 6.59 | 0 | 49 |
| 0.30 | 0.878 | 43 | 0.939 | 6.59 | 6 | 43 |
| 1.00 | 0.878 | 43 | 0.939 | 6.59 | 36 | 13 |

The `1.00` threshold is effectively an MLP-only reference, not a meaningful graph gate. Notebook `16` therefore does not prove that graph stopping improves Notebook `13`. Its stronger result is that graph-guided question selection is worth testing live.

Recommended next step:

- build Notebook `17` as a live MedKGI-style graph-shortlist pilot
- keep Notebook `13`'s MLP-guided stop rule fixed
- replace only the evidence shortlist with graph top-10 candidates
- compare against Notebook `13` on the same cases and hard-case trajectories

## Notebook 17 Live MedKGI Graph Shortlist Pilot

Notebook `17` implemented the live version suggested by Notebook `16`.

Notebook:

- `notebooks/17_live_medkgi_graph_shortlist_pilot.ipynb`

Artifacts:

- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_dryrun_smoke_v1/`
- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/`

Report:

- `reports/algorithmic_ledger/live_medkgi_graph_shortlist_pilot.md`

Important implementation point:

- Notebook `17` does not change the stop rule.
- It keeps Notebook `13`'s selected MLP-guided stop rule fixed.
- It changes only the evidence shortlist to MedKGI-style graph top-10 legal fields.

Dry-run validation:

- default settings ran two cases without API calls
- all expected artifact files were written
- graph request-quality fields are present in `predictions.csv`
- `promotion_decision.json` is correctly marked `dry_run_smoke_not_for_promotion`

Live pilot result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 |
| Notebook `08` lambda `0.10` LLM-only | 22/24 | 0.917 | 0.917 | 0.917 | 0.846 | 13.04 |
| Notebook `12` offline selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |
| Notebook `17` graph shortlist pilot | 20/24 | 0.833 | 0.833 | 0.875 | 0.744 | 6.21 |

Graph-request quality:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 1.76 |
| Mean requested information gain | 0.373 |
| Requests outside graph top-10 | 0 |
| Mean graph shortlist size | 10.0 |
| Stop-before-cap rate | 0.958 |
| Cap-hit count | 1 |

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| Notebook `13` only correct | 2 |
| Notebook `17` only correct | 0 |
| Both wrong | 2 |

Notebook `17` introduced two extra errors: `Chagas` became `Sarcoidosis`, and `Ebola` became `HIV (initial infection)`. It did not fix either persistent Notebook `13` failure on `Croup` or `Pericarditis`.

Promotion decision:

- `reject_keep_notebook13_v1`

Interpretation:

Notebook `17` is an informative negative result. It shows that train-derived graph scores are useful diagnostics, but a hard graph top-10 replacement shortlist is too restrictive as the live controller. The graph mechanism selected locally high-information evidence, but if the active differential was already biased, the shortlist efficiently explored the wrong neighborhood.

Current conclusion:

- Notebook `13` remains the frozen proposed method.
- Notebook `17` v1 should not be run on `final49`.
- Future graph work should use graph scores as an advisory/blending signal, not as a hard replacement for the broader Notebook `13` shortlist.

## Notebook 18 Graph-Advisory Hybrid Shortlist

Notebook `18` implements the next graph version after the rejected Notebook `17` pilot.

Notebook:

- `notebooks/18_graph_advisory_hybrid_shortlist.ipynb`

Artifacts:

- `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_dryrun_smoke_v1/`
- `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_v1/`

Report:

- `reports/algorithmic_ledger/graph_advisory_hybrid_shortlist_report.md`

Implementation change:

- Notebook `13`'s selected MLP stop rule is unchanged.
- Notebook `13` shortlist diversity is preserved.
- MedKGI graph information gain is used as an advisory score.
- rare disease-specific train-derived log-odds support can force decisive evidence fields into the shortlist.
- unsafe early agent stops can be overridden when the MLP is uncertain and high-value or rare evidence remains.

Advisory score:

| Component | Weight |
|---|---:|
| Notebook `13` base shortlist score | 0.45 |
| Graph information gain | 0.25 |
| Disease-specific rare support | 0.20 |
| Split balance | 0.10 |

Dry-run validation:

- static code-cell parse passed
- notebook executed top-to-bottom with `nbclient`
- no live API calls were made
- dry-run artifacts and figures were written
- `INTERACTIVE_API_KEY_BOOTSTRAP = False` by default, so dry runs do not prompt for a key

Live pilot result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 |
| Notebook `17` hard graph shortlist | 20/24 | 0.833 | 0.833 | 0.875 | 0.744 | 6.21 |
| Notebook `18` graph-advisory shortlist | 21/24 | 0.875 | 0.875 | 0.917 | 0.795 | 7.67 |

Notebook `18` recovered the two Notebook `17` failures it was specifically designed to address:

- `test:51421` Chagas changed from Sarcoidosis/wrong in Notebook `17` to Chagas/correct in Notebook `18`
- `test:77908` Ebola changed from HIV initial infection/wrong in Notebook `17` to Ebola/correct in Notebook `18`

However, it introduced a new failure:

- `test:16097` Stable angina changed from correct in Notebooks `13` and `17` to Boerhaave/wrong in Notebook `18`

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 21 |
| Notebook `13` only correct | 1 |
| Notebook `18` only correct | 0 |
| Both wrong | 2 |

Paired result against Notebook `17`:

| Outcome | Cases |
|---|---:|
| Both correct | 19 |
| Notebook `18` only correct | 2 |
| Notebook `17` only correct | 1 |
| Both wrong | 2 |

Graph/advisory diagnostics:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 5.12 |
| Mean requested information gain | 0.242 |
| Mean top graph score at stop | 0.796 |
| Requests outside pure graph top-10 | 17 |
| Agent-stop safety overrides | 8 |
| Cap-hit count | 3 |

Interpretation:

Notebook `18` is better than the rejected hard-graph Notebook `17`, but it still does not beat Notebook `13`. It increased request usage and dropped one case below Notebook `13`. The three wrong cases all hit the request cap, so the main failure is not premature stopping. The failure is wrong trajectory steering and inability to recover the correct belief even after many requests.

Promotion decision:

- reject Notebook `18`
- keep Notebook `13` as the frozen proposed method
- treat graph evidence as a useful audit/advisory signal, not as a promoted live controller yet

## Notebook 19 Bayesian VOI Algorithmic Ledger Offline

Notebook `19` is the next algorithmic/mathematical successor after the graph experiments.

Notebook:

- `notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb`

Artifacts:

- smoke: `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1_smoke/`
- full: `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1/`

Report:

- `reports/algorithmic_ledger/bayesian_voi_ledger_offline_report.md`

Implementation:

- builds train-only Bayesian likelihood tables over DDXPlus root evidence outcomes
- maintains an explicit posterior over all `49` pathologies
- fuses Bayesian posterior with the selected partial-evidence MLP posterior
- chooses evidence by one-step value of information
- stops only when confidence, margin, entropy, remaining VOI, contradiction, and Bayes/MLP agreement are acceptable
- makes no API calls

Posterior fusion:

```text
fused = softmax(
  0.60 * log(MLP posterior)
+ 0.40 * log(Bayesian posterior)
)
```

VOI utility:

```text
utility =
  0.55 * expected_fused_entropy_reduction
+ 0.20 * expected_margin_gain
+ 0.15 * contradiction_resolution_gain
+ 0.10 * rare_recovery_bonus
- lambda_cost
- redundancy_penalty
```

Smoke validation:

| Metric | Value |
|---|---:|
| Cases | 3 |
| Fused correct | 2/3 |
| Fused accuracy | 0.667 |
| Mean requests | 2.67 |
| Stop-before-cap rate | 0.667 |
| Cap-hit count | 1 |

The smoke result is not a scientific benchmark. It only confirms that the notebook runs, writes the artifact contract, loads the partial-evidence MLP, updates the Bayesian ledger, and generates figures.

Full 49-case result:

| Lambda | Fused correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Cap hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 33/49 | 0.673 | 0.837 | 0.878 | 0.620 | 22.37 | 28 |
| 0.02 | 26/49 | 0.531 | 0.735 | 0.837 | 0.461 | 7.98 | 4 |
| 0.05 | 26/49 | 0.531 | 0.633 | 0.776 | 0.447 | 5.65 | 1 |
| 0.10 | 25/49 | 0.510 | 0.653 | 0.776 | 0.424 | 4.43 | 1 |
| 0.15 | 24/49 | 0.490 | 0.633 | 0.776 | 0.396 | 4.33 | 1 |

Reference:

| System | Correct | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 43/49 | 0.878 | 0.939 | 6.59 |
| Notebook `19` best fused, lambda `0.00` | 33/49 | 0.673 | 0.878 | 22.37 |
| Notebook `19` near-budget, lambda `0.02` | 26/49 | 0.531 | 0.837 | 7.98 |

Paired against Notebook `13` at lambda `0.00`:

| Outcome | Cases |
|---|---:|
| Both correct | 32 |
| Notebook `19` only correct | 1 |
| Notebook `13` only correct | 11 |
| Both wrong | 5 |

The one Notebook `19` fix was `test:125508` Unstable angina. It was outweighed by eleven regressions.

Promotion rule:

- promote only if Notebook `19` reaches at least Notebook `13`'s `43/49` accuracy with mean requests `<=6.59`, or reaches `44/49` with mean requests `<=9.0`, or fixes at least two persistent Notebook `13` hard cases without more than one new regression

Promotion decision:

- `do_not_promote_yet`
- keep Notebook `13` as the frozen proposed method
- do not create a live Notebook `20` from this version

Interpretation:

Notebook `19` is an important negative result. Posterior-level Bayesian VOI is mathematically cleaner than the graph-only shortlist, but this implementation does not beat the LLM-led Notebook `13` trajectory. Lambda `0.00` asks much more evidence than Notebook `13` and still performs much worse, so the main failure is not early stopping. The failure is trajectory quality and posterior calibration: the VOI controller selects many generic high-information roots, and the partial-evidence MLP becomes overconfident on evidence subsets outside the trace distribution it was trained on.

The Bayesian signal remains useful for audit and possible advisory scoring, but it should not replace the Notebook `13` evidence-acquisition policy.

## Notebook 20 LLM-Led Graph-Ledger Context

Notebook `20` implements the corrected algorithmic-ledger experiment after the Notebook `17-19` negative ablations.

Notebook:

- `notebooks/20_llm_led_graph_ledger_context.ipynb`

Dry-run artifacts:

- `artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_dryrun_smoke_v1/`

Live artifacts:

- `artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1/`

Report:

- `reports/algorithmic_ledger/llm_led_graph_ledger_context_report.md`

Core correction:

- the graph ledger is no longer a replacement controller
- the LLM remains the evidence-request chooser
- Notebook `13`'s legal action menu remains active
- Notebook `13`'s MLP-guided stop rule remains unchanged
- graph information is provided only as compact prompt context

Graph context fields:

- active differential
- diagnosis support from revealed evidence
- diagnosis contradiction from revealed evidence
- unresolved diagnosis pairs
- advisory discriminator hints
- consistency warnings

Dry-run validation:

| Metric | Value |
|---|---:|
| Cases | 2 |
| Accuracy | 1.000 |
| Top-5 | 1.000 |
| Mean requests | 6.500 |
| Stop-before-cap rate | 1.000 |

Interpretation:

The dry-run result is not scientific evidence. It only confirms that the notebook runs top-to-bottom without API credentials and writes the required graph-context artifacts.

Live `pilot24` result:

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook `20` LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

Notebook `20` did not meet the top-1 promotion rule. However, it produced the strongest ranking result in the graph-ledger line: `23/24` top-3 and top-5.

Promotion decision:

- do not promote Notebook `20` to `final49`
- keep Notebook `13` as the frozen proposed method
- use Notebook `20` as motivation for Notebook `21`, which tests whether graph context can act as a non-oracle critic/adjudicator

## Notebook 21 Graph-Context Policy Lab

Notebook `21` is an offline-only experimental lab created after the live Notebook `20` result.

Notebook:

- `notebooks/21_graph_context_policy_lab.ipynb`

Artifacts:

- `artifacts/graph_algorithmic_ledger/graph_context_policy_lab_24case_v1/`

Report:

- `reports/algorithmic_ledger/graph_context_policy_lab_report.md`

Purpose:

- test whether graph-ledger context can act as a critic, guardrail, adjudicator, or drift detector
- explain why Notebook `20` improved top-3/top-5 to `23/24` but dropped top-1 to `20/24`
- identify whether a small non-oracle rule is strong enough to justify a future live Notebook `22`

Notebook `21` makes no API calls and trains no model. It replays existing traces from Notebooks `13`, `17`, `18`, and `20`.

Reference check:

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook `17` hard graph shortlist | 0.833 | 0.833 | 0.875 | 6.21 |
| Notebook `18` graph-advisory shortlist | 0.875 | 0.875 | 0.917 | 7.67 |
| Notebook `20` LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

The important positive signal is Notebook `20`'s ranking quality: it reaches `23/24` top-3/top-5 while using slightly fewer requests than Notebook `13`.

Policy variants tested:

- graph top-3/top-5 adjudicators
- LLM/MLP consensus adjudicators
- drift guards for stable graph-supported diagnoses
- stop guard flagging and replay where future trace turns existed
- combined graph guard/adjudication variants
- oracle top-3/top-5 upper bounds

Best non-oracle outcome:

| Variant | Source | Accuracy | Top-5 | Mean requests | Interpretation |
|---|---|---:|---:|---:|---|
| `drift_guard_notebook13_tc1.0_delta0.5` | Notebook `13` | 0.917 | 0.917 | 6.58 | Matches Notebook `13`, but changes one wrong case to another wrong case |

This is not a real improvement.

Oracle upper bound:

| Variant | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `20` oracle top-3 | 0.958 | 0.958 | 0.958 | 6.13 |
| Notebook `20` oracle top-5 | 0.958 | 0.958 | 0.958 | 6.13 |

Interpretation:

- Notebook `20` contains enough ranking information to reach `23/24`.
- The graph context signal is real.
- Current hand-written graph adjudication rules are not reliable enough to choose the correct top-1 without labels.

Graph diagnostic result for Notebook `20` final states:

| Feature | Correct mean | Wrong mean | Wrong - correct |
|---|---:|---:|---:|
| Top contradiction | 0.377 | 4.015 | +3.638 |
| Top contradiction minus support | -6.964 | 0.906 | +7.870 |
| Top net support | 6.964 | -0.906 | -7.870 |

This means the graph ledger is good at flagging suspect final diagnoses. The remaining difficulty is selecting the correct alternative from the ranked differential.

Selection decision:

```text
no_promotable_candidate
```

Project interpretation:

- keep Notebook `13` as the frozen proposed method
- do not create a live graph-context controller from the current hand-threshold graph rules
- treat Notebook `21` as a strong diagnostic result showing that graph-ledger context is useful as a critic/audit feature
- if continuing graph work, use calibrated or learned adjudication instead of more manually tuned thresholds

## Notebook 22 Graph Posterior Final Adjudicator

Notebook `22` is an offline-only graph-ledger final adjudicator added after the Notebook `21` policy lab.

Notebook:

- `notebooks/22_graph_posterior_final_adjudicator.ipynb`

Artifacts:

- `artifacts/graph_algorithmic_ledger/graph_posterior_final_adjudicator_49case_v1/`

Report:

- `reports/algorithmic_ledger/graph_posterior_final_adjudicator_report.md`

Purpose:

- preserve Notebook `13` evidence acquisition unchanged
- reconstruct the final visible evidence state from Notebook `13` traces
- compute train-derived signed graph support for every pathology
- use the graph posterior as a conservative final critic, not a question controller
- make no API calls

Primary score:

```text
graph_score(disease) =
  sum clip(log_odds_support(revealed outcome -> disease), -3, 3)
```

Selected conservative override:

```text
override Notebook 13 top-1 only if:
  graph_top1 differs from Notebook 13 top-1
  graph_margin >= 1.0
  graph_score(Notebook 13 top-1) < 0
  graph_score(graph_top1) > 0
```

Result:

| System | Cases | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 49 | 43/49 = 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Graph-only final head | 49 | 44/49 = 0.898 | 0.959 | 0.980 | n/a | 6.59 |
| Notebook `22` conservative graph critic | 49 | 44/49 = 0.898 | 0.939 | 0.939 | 0.867 | 6.59 |

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 43 |
| Notebook `22` only correct | 1 |
| Notebook `13` only correct | 0 |
| Both wrong | 5 |

The only changed prediction is:

| Case | True pathology | Notebook `13` | Notebook `22` | Result |
|---|---|---|---|---|
| `test:81691` | Croup | Anemia | Croup | fixed |

Promotion decision:

```text
offline_candidate_promoted
```

Interpretation:

- Notebook `22` is the first graph-ledger variant to improve over Notebook `13` on the 49-case artifact.
- It succeeds because graph information is used as a final-state mathematical critic rather than as a replacement evidence controller.
- It should be presented as an offline final-head enhancement candidate that needs held-out or live confirmation before replacing Notebook `13` as the main defended method.

Post-run analysis:

- The selected critic fires exactly once on the 49-case confirmation.
- The single override fixes `test:81691` Croup: Notebook `13` predicts Anemia with graph score `-2.359`; the graph top-1 is Croup with score `1.177` and margin `2.073`.
- The 24-case sanity slice shows the same behavior, improving from `22/24` to `23/24` by fixing Croup while leaving Pericarditis wrong. This is a consistency check, not independent validation, because the 24 cases are part of the 49-case confirmation.
- The graph-only head reaches `44/49` top-1, `0.959` top-3, and `0.980` top-5, which confirms strong ranking signal, but graph-only is not safe enough to replace Notebook `13` as the final head.

Remaining Notebook `22` errors:

| Case | True pathology | Final prediction | Requests | Graph true rank | Main interpretation |
|---|---|---|---:|---:|---|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | 24 | 4 | graph distrusts Notebook `13` but lacks a confident correct alternative |
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | 8 | 2 | graph and Notebook `13` both prefer the chronic neighbor |
| `test:8666` | Influenza | HIV initial infection | 3 | 2 | graph and Notebook `13` both prefer HIV initial infection |
| `test:62878` | Pericarditis | Anemia | 15 | 15 | revealed evidence does not support the true disease strongly enough |
| `test:125508` | Unstable angina | Anemia | 2 | 3 | true disease is plausible but below Anemia on the final evidence state |

Bottom line: Notebook `22` gives the project the mathematical graph-ledger hook we wanted, but the result is a final-state critic result. The next defensible improvement was to train or calibrate a selector on development data that combines graph support, graph contradiction, LLM ranking, and MLP ranking features; Notebook `23` below implements that larger rescue direction offline.

Ambition check for a larger improvement:

The Croup fix is not enough to make Notebook `22` a new center-of-gravity result by itself. The stronger finding is that the graph posterior top-k contains enough signal for a much larger rescue if a selector can be learned without label-peeking:

| Candidate pool | Oracle accuracy |
|---|---:|
| Notebook `13` top-1 or graph top-1 | 44/49 |
| Notebook `13` top-1 or graph top-2 | 46/49 |
| Notebook `13` top-1 or graph top-3 | 47/49 |
| Notebook `13` top-1 or graph top-5 | 48/49 |
| Notebook `13` top-1 or union of LLM/MLP/graph top-5 | 48/49 |

This suggests the next serious algorithmic direction is not another hand threshold. It is a calibrated final reranker over a candidate set containing Notebook `13` top-1, graph top-3/top-5, MLP top-5, and LLM top-5. The evaluation target should be at least `47/49`, but the rule must be trained or calibrated from train/validate-derived partial evidence states, or from fresh held-out traces. Directly tuning a selector on the six 49-case misses would make the number look better without making the claim stronger.

## Notebook 23 Calibrated Graph-Bayes Rescue Reranker

Notebook `23` implements the larger graph-ledger rescue direction identified after Notebook `22`.

Notebook:

- `notebooks/23_calibrated_graph_bayes_rescue_reranker.ipynb`

Artifacts:

- `artifacts/graph_algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_49case_v1/`

Report:

- `reports/algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md`

Purpose:

- keep Notebook `13` as the first-pass live workup trace
- train/calibrate candidate scoring from DDXPlus train/validate synthetic partial evidence states
- combine graph support, Bayesian likelihoods, MLP ranks, LLM ranks, and the initial prior
- ask up to three additional graph/Bayes discriminator roots only for suspicious early stops
- choose post-rescue candidates with the trained L2 reranker, plus a conservative graph-support accept guard
- make no API calls

Selected policy:

```text
calibrated_graph_bayes_rescue_v1
```

Result:

| System | Cases | Accuracy | Mean requests | Extra requests | Improvements vs Notebook `13` | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 49 | 43/49 = 0.878 | 6.59 | 0 | 0 | 0 |
| Notebook `22` graph critic | 49 | 44/49 = 0.898 | 6.59 | 0 | 1 | 0 |
| Notebook `23` graph-Bayes rescue | 49 | 47/49 = 0.959 | 6.96 | 18 total | 4 | 0 |

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 43 |
| Notebook `23` only correct | 4 |
| Notebook `13` only correct | 0 |
| Both wrong | 2 |

Fixed Notebook `13` misses:

| Case | True pathology | Notebook `13` | Notebook `23` | Source |
|---|---|---|---|---|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | Acute COPD exacerbation / infection | prior recovery |
| `test:81691` | Croup | Anemia | Croup | graph critic |
| `test:8666` | Influenza | HIV initial infection | Influenza | rescue rerank |
| `test:125508` | Unstable angina | Anemia | Unstable angina | rescue rerank |

Remaining misses:

| Case | True pathology | Notebook `23` prediction |
|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis |
| `test:62878` | Pericarditis | Anemia |

Promotion decision:

```text
offline_candidate_promoted
```

Interpretation:

- Notebook `23` is the first graph-ledger enhancement that materially changes the result, reaching the `47/49` target suggested by the graph top-3 rescue ceiling.
- It does not replace Notebook `13` as a live acquisition method; it is an offline rescue layer over Notebook `13` traces.
- The result is strong enough to present as the main algorithmic-ledger contribution, with live confirmation left as future work.

## Notebook 24 Live Graph-Bayes Rescue Confirmation

Notebook `24` implements the live-confirmation wrapper for Notebook `23`.

Notebook:

- `notebooks/24_live_graph_bayes_rescue_confirmation.ipynb`

Dry-run artifacts:

- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_dryrun_smoke_v1/`

Live artifacts:

- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1/`

Report:

- `reports/algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md`

Implementation:

- keeps the Notebook `13` live workup loop unchanged
- applies the frozen Notebook `23` graph/Bayes rescue layer after the base stop
- saves both `notebook13_live_prediction` and `graph_bayes_rescue_prediction`
- uses the Notebook `23` train/validate-calibrated L2 reranker
- uses Notebook `16` graph statistics and Notebook `19` Bayesian likelihoods
- makes no live API calls by default

Dry-run validation:

| Metric | Value |
|---|---:|
| Cases | 2 |
| Notebook `13` base accuracy | 1.000 |
| Graph/Bayes rescue accuracy | 1.000 |
| Mean base requests | 6.00 |
| Mean total requests | 6.00 |
| Extra rescue requests | 0 |
| Regressions | 0 |

The dry-run is only a smoke test. It confirms that prompt construction, base workup execution, trace reconstruction, frozen rescue policy loading, reranker scoring, and artifact writing work in one notebook.

Live result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Original Notebook `13` artifact | 43/49 | 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Notebook `23` offline rescue candidate | 47/49 | 0.959 | n/a | n/a | n/a | 6.96 |
| Notebook `24` fresh live base workup | 45/49 | 0.918 | 0.939 | 0.939 | 0.895 | 6.20 |
| Notebook `24` live graph/Bayes rescue | 45/49 | 0.918 | 0.939 | 0.939 | 0.895 | 6.39 |

Live rescue decision summary:

| Metric | Value |
|---|---:|
| Improvements versus live base | 0 |
| Regressions versus live base | 0 |
| Changed predictions | 1 |
| Extra rescue requests | 9 |

The rescue layer did not meet the promotion criteria. It changed `test:76022` Panic attack from Anaphylaxis to PSVT, but both predictions were wrong. It also requested three extra roots on three already-correct cases and then abstained, so the final diagnosis stayed correct but request count increased.

The fresh live base trajectory is itself important: it improved over the original Notebook `13` artifact from `43/49` to `45/49`, while reducing mean requests from `6.59` to `6.20`. That means Notebook `24` strengthens the evidence for the base LLM-led + MLP-stopped architecture, but it does not validate the graph/Bayes rescue as a live improvement.

Promotion criteria:

- strong confirmation: at least `46/49` correct and zero regressions versus the live Notebook `13` base prediction
- ideal confirmation: `47/49` correct with mean total requests `<= 7.25`

Current status:

```text
live_completed_not_promoted
```

Interpretation:

- Notebook `13` remains the core live evidence-acquisition method.
- Notebook `23` remains a strong offline graph/Bayes rescue candidate, not a confirmed live replacement.
- Notebook `24` shows that live trajectories can vary even with deterministic settings; future claims should report the original frozen artifact and the fresh live confirmation separately.

## Notebook 25 Live Base Trajectory Replicates

Notebook `25` collected three rescue-disabled Notebook `13`-style live base replicates on the same 49-case benchmark.

Artifact root:

- `artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1/`

| Run | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| `r01` | 44/49 | 0.898 | 0.939 | 0.939 | 0.871 | 7.00 |
| `r02` | 42/49 | 0.857 | 0.918 | 0.918 | 0.816 | 7.02 |
| `r03` | 42/49 | 0.857 | 0.939 | 0.939 | 0.810 | 6.82 |

Within the three Notebook `25` replicates:

- same final prediction: `45/49`
- prediction instability: `4/49`
- correctness instability: `3/49`

Combined with the original Notebook `13` and Notebook `24` base trajectories, these replicates provide enough live-style variation to study branching without using new API calls.

## Notebook 26 Offline Branching Trajectory Lab

Notebook `26` is the offline branching lab over five observed Notebook `13`-style base trajectories:

- Notebook `13` frozen artifact
- Notebook `24` fresh live base
- Notebook `25` replicates `r01`, `r02`, `r03`

Artifact root:

- `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/`

Key trajectory statistics:

| Metric | Value |
|---|---:|
| Same prediction across all five runs | 41/49 |
| Prediction instability cases | 8/49 |
| Correctness instability cases | 7/49 |
| Majority-vote accuracy | 43/49 = 0.878 |
| Oracle best-of-five accuracy | 47/49 = 0.959 |
| Same-prefix divergent states | 64 |
| Impactful divergent states | 10 |

The central conclusion is that broad voting is not enough, but targeted branching is promising.

Best diagnostic Notebook `13`-base policy:

```text
trigger=hybrid_suspicion_v1
branch_budget=2
judge=highest_bayes_posterior
```

| Metric | Value |
|---|---:|
| Accuracy | 47/49 = 0.959 |
| Base accuracy | 43/49 = 0.878 |
| Wins versus base | 4 |
| Regressions versus base | 0 |
| Branch trigger rate | 0.184 |
| Mean branches spawned | 0.367 |
| Mean selected requests | 6.86 |
| Mean total branch requests | 9.18 |

Fixed Notebook `13` misses:

- `test:125508`: Unstable angina
- `test:38475`: Acute COPD exacerbation / infection
- `test:81691`: Croup
- `test:8666`: Influenza

Remaining misses:

- `test:111176`: Acute rhinosinusitis predicted as Chronic rhinosinusitis
- `test:62878`: Pericarditis predicted as Anemia

Status:

```text
diagnostic_only_not_promoted
```

Notebook `26` guided the next live experiment, Notebook `27`. It should still be treated as diagnostic because it reused observed trajectories and evaluated policy curves on the 49-case labels.

Post-hoc replicate final-layer quickcheck:

| Replicate | Base | Strict graph/Bayes final layer | Raw graph top-1 | Raw Bayes top-1 |
|---|---:|---:|---:|---:|
| `r01` | 44/49 | 44/49 | 45/49 | 45/49 |
| `r02` | 42/49 | 42/49 | 44/49 | 44/49 |
| `r03` | 42/49 | 42/49 | 44/49 | 44/49 |

The strict conservative final layer did not fire. The raw graph/Bayes heads improved all three replicates diagnostically with zero regressions, which reinforces the decision to use graph/Bayes as a branch adjudication signal rather than as a hard question controller. The quickcheck artifact is:

- `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/replicate_graph_bayes_final_layer_quickcheck.csv`

## Notebook 27 Live Targeted Branching Confirmation

Notebook `27` implemented the next live experiment implied by Notebook `26`.

Notebook:

- `notebooks/27_live_targeted_branching_confirmation.ipynb`

Report:

- `reports/algorithmic_ledger/live_targeted_branching_confirmation_report.md`

Live artifact root:

- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/`

Dry-run smoke artifact root:

- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_dryrun_smoke_v1/`

Frozen prospective policy:

```text
trigger=hybrid_suspicion_v1
branch_budget=2
judge=highest_raw_bayes_posterior
```

The notebook runs the Notebook `13` selected-stop workup as a base branch. If the terminal state is suspicious by label-free graph/Bayes/MLP ledger signals, it launches up to two fresh-context full branches:

- `graph_bayes_scout`
- `counteranchor_scout`

The final judge selects among the completed base/branch candidates by raw Bayesian posterior of each branch's own final prediction, using graph support, MLP confidence, and base order only as deterministic tie-breakers.

Implementation validation:

| Check | Result |
|---|---:|
| Static parse of code cells | passed |
| No-spend two-case dry run | passed |
| Dry-run cases | 2 |
| Dry-run branch-selected accuracy | 2/2 |
| Dry-run artifact contract | passed |
| Forced no-spend branch-path smoke | passed |

The dry-run is only an implementation smoke test. The live 49-case run completed afterward.

Live result:

| Metric | Value |
|---|---:|
| Base branch accuracy | 42/49 = 0.857 |
| Targeted branching accuracy | 43/49 = 0.878 |
| Top-3 accuracy | 0.918 |
| Top-5 accuracy | 0.939 |
| Wins versus base | 2 |
| Regressions versus base | 1 |
| Branch trigger rate | 9/49 = 0.184 |
| Branches spawned | 18 |
| Mean base requests | 6.92 |
| Mean selected requests | 6.82 |
| Mean total branch requests | 11.45 |

Paired changes:

| Case | True pathology | Base prediction | Branch-selected prediction | Outcome |
|---|---|---|---|---|
| `test:35039` | Myocarditis | Sarcoidosis | Myocarditis | win |
| `test:76022` | Panic attack | Anaphylaxis | Panic attack | win |
| `test:38475` | Acute COPD exacerbation / infection | Acute COPD exacerbation / infection | Bronchospasm / acute asthma exacerbation | regression |

The actual branch pool ceiling was limited:

| System | Correct |
|---|---:|
| Base branch only | 42/49 |
| Notebook `27` selected judge | 43/49 |
| Oracle over actual base + spawned branch predictions | 44/49 |

The raw mathematical ledger still showed stronger candidate-generation signal:

| Diagnostic head over Notebook `27` base final state | Correct |
|---|---:|
| Base branch prediction | 42/49 |
| Raw graph top-1 | 45/49 |
| Raw Bayes top-1 | 45/49 |

Raw graph and Bayes fixed Myocarditis, Panic attack, and Possible NSTEMI / STEMI with no base-correct regressions in this diagnostic replay.

Interpretation:

- Notebook `27` partially validates the multi-agent branching hypothesis because fresh branches recovered Myocarditis and Panic attack.
- The selected raw-Bayes-only resolver is not promoted because it regressed a correct COPD base answer and reached only `43/49`.
- Confidence/contradiction-triggered branching misses consensus wrong-answer cases: Acute rhinosinusitis, Bronchitis, and Croup all had `suspicion_signal_count = 0`.
- The next credible direction is a confounder-coverage branch classifier plus a cautious graph/Bayes/MLP resolver with graph/Bayes pseudo-candidates and base protection.

## Notebook 28 MLP-Gated Confounder Graph-Bayes Branching

Notebook `28` implements and tests that next direction:

- `notebooks/28_mlp_gated_confounder_graph_bayes_branching.ipynb`
- `reports/algorithmic_ledger/mlp_gated_confounder_graph_bayes_branching_report.md`
- dry-run root: `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1/`
- live root: `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1/`

Selected policy:

```text
branch_trigger=branch_trigger_mlp_v1
branch_threshold=0.375
branch_budget=3
resolver=calibrated_graph_bayes_mlp_resolver_v1
```

Notebook `28` differs from Notebook `27` in three important ways:

- branching is decided by a learned MLP classifier trained on train/validate synthetic partial states
- the final candidate pool includes graph, Bayes, and MLP pseudo-candidates, not only completed LLM branch predictions
- the resolver includes base protection when the base answer is graph rank `1` and Bayes rank `1`

Dry-run result:

| Metric | Value |
|---|---:|
| Dry-run cases | 2 |
| Base accuracy | 2/2 |
| Notebook `28` selected accuracy | 2/2 |
| Branch trigger rate | 0/2 |
| Mean selected requests | 6.0 |
| Mean total branch requests | 6.0 |

Calibration diagnostics:

| Model | AUC | Average precision | Notes |
|---|---:|---:|---|
| Branch-trigger MLP | 0.954 | 0.931 | selected threshold `0.375`, recall `0.940` |
| Candidate resolver | 0.955 | 0.811 | trained on synthetic candidate rows |

Live 49-case result:

| Metric | Value |
|---|---:|
| Base branch accuracy | 42/49 = 0.857 |
| Notebook `28` selected accuracy | 44/49 = 0.898 |
| Top-3 accuracy | 0.959 |
| Top-5 accuracy | 0.959 |
| Macro-F1 | 0.864 |
| Wins versus base | 2 |
| Regressions versus base | 0 |
| Branch trigger rate | 4/49 = 0.082 |
| Branches spawned | 12 |
| Mean selected requests | 6.63 |
| Mean total branch requests | 9.96 |

Paired wins:

| Case | True pathology | Base prediction | Notebook `28` prediction | Source |
|---|---|---|---|---|
| `test:35039` | Myocarditis | Sarcoidosis | Myocarditis | `pseudo_graph_top1` |
| `test:62878` | Pericarditis | Panic attack | Pericarditis | `counteranchor_scout` |

Notebook `28` is not promoted because it did not reach the `47/49` promotion target and did not beat the fresh Notebook `24` live base of `45/49`.

The post-hoc analysis is still important. The actual scored candidate pool had only a `44/49` oracle, so the final judge was not the main bottleneck. When ranked differential entries are promoted into the candidate pool, the oracle changes sharply:

| Candidate pool | Oracle correct |
|---|---:|
| Current scored candidates | 44/49 |
| Current candidates plus all candidate ranked top-2 | 47/49 |
| Current candidates plus all candidate ranked top-3 | 48/49 |
| Selected final ranked differential top-2 only | 47/49 |

The remaining misses show why:

- `test:111176` Acute rhinosinusitis was rank 2 behind Chronic rhinosinusitis.
- `test:11655` Bronchitis was rank 2 behind URTI.
- `test:8666` Influenza was rank 2 behind HIV initial infection.
- `test:81691` Croup appeared in the `graph_bayes_scout` branch differential at rank 3, but was never scored as a standalone candidate.
- `test:76022` Panic attack did not appear in the scored pool or top-5 differential.

This motivated Notebook `29`: a listwise differential adjudicator that scores base ranked top-5, branch ranked top-5, graph top-5, Bayes top-5, MLP top-5, and confounder challengers rather than only final top-1 predictions.

## Notebook 29 Listwise Differential Graph-Bayes-MLP Adjudicator

Notebook `29` implements that offline final-head continuation over frozen Notebook `28` traces.

- `notebooks/29_listwise_differential_graph_bayes_adjudicator.ipynb`
- `reports/algorithmic_ledger/listwise_differential_graph_bayes_adjudicator_report.md`
- artifact root: `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`

Selected policy:

```text
candidate scorer = L2 logistic model trained on Notebook 28 train/validate synthetic candidate features
candidate pool = source top-1 + ranked differential top-3 + graph/Bayes/MLP top candidates
override margin = 0.02 over Notebook 28 selected answer
```

Result:

| Metric | Value |
|---|---:|
| Notebook `28` selected accuracy | 44/49 = 0.898 |
| Notebook `29` selected accuracy | 45/49 = 0.918 |
| Wins vs Notebook `28` | 1 |
| Regressions vs Notebook `28` | 0 |
| Changed predictions | 1 |
| Mean selected requests | 6.63 |
| Mean total branch requests | 9.96 |

The single win is `test:81691` Croup, promoted from the `graph_bayes_scout` branch differential at rank 3. The candidate is graph rank 1, Bayes rank 1, and MLP rank 3 in that branch evidence state.

Candidate-pool oracle:

| Candidate pool | Oracle correct |
|---|---:|
| Source top-1 plus ranked top-1 | 44/49 |
| Source top-1 plus ranked top-2 | 47/49 |
| Source top-1 plus ranked top-3 | 48/49 |
| Source top-1 plus ranked top-5 | 48/49 |
| All exploded graph/Bayes/MLP/ranked candidates | 49/49 |

Notebook `29` is not promoted because it does not reach `47/49`. It does, however, sharpen the next research target: candidate availability is strong enough; calibrated selection among top-1 and rank-2/rank-3 confounders is the bottleneck.

Post-run deep analysis after the rerun sharpened this further:

| Candidate signal family | Oracle correct |
|---|---:|
| Source top-1 only | 44/49 |
| Ranked differential top-2 | 47/49 |
| Ranked differential top-3 | 48/49 |
| Graph top-1 | 45/49 |
| Graph top-2 | 49/49 |
| Bayes top-1 | 45/49 |
| Bayes top-2 | 49/49 |
| MLP top-3 | 48/49 |

The remaining misses are not missing from the mathematical ledger: Acute rhinosinusitis, Bronchitis, and Influenza are all rank 2 under the LLM differential, graph, Bayes, and usually MLP. Panic attack is not in the LLM ranked differential, but graph and Bayes still place it rank 2. The next useful controller is therefore a pairwise or abstaining confounder adjudicator, not another broad candidate-pool expansion.

## Notebook 30 Hypothesis-Forced Differential Branching

Notebook `30` implements hypothesis-forced live branching. It keeps the Notebook `13` selected-stop base branch, but replaces generic branch roles with explicit target hypotheses computed from the base terminal visible evidence.

- `notebooks/30_hypothesis_forced_differential_branching.ipynb`
- `reports/algorithmic_ledger/hypothesis_forced_differential_branching_report.md`
- dry-run artifact root: `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_dryrun_smoke_v1/`
- live artifact root: `artifacts/trajectory_replicates/hypothesis_forced_differential_branching_49case_v1/`

Policy:

```text
trigger = hypothesis_branch_trigger_mlp_v1
branch budget = 3
judge = hypothesis_forced_graph_bayes_mlp_resolver_v1
```

The target hypothesis table ranks challengers using graph top-5, Bayes top-5, MLP top-5, LLM ranked differential, hybrid ranked differential, graph/Bayes support gaps, and missing pairwise discriminator utility. If the learned gate fires, each branch receives:

- assigned target hypothesis
- base prediction to challenge
- preferred discriminator roots
- graph/Bayes/MLP support summary
- branch role template

Live result:

| Metric | Value |
|---|---:|
| Base branch accuracy | 42/49 = 0.857 |
| Selected policy accuracy | 44/49 = 0.898 |
| Wins vs base | 2 |
| Regressions vs base | 0 |
| Changed predictions | 4 |
| Branch trigger rate | 6/49 |
| Branches spawned | 18 |
| Mean selected requests | 6.78 |
| Mean total branch requests | 12.10 |

The live run supports the hypothesis-forced branching idea only partially. It safely improves the same-run base, but the cost increase is substantial and only one selected final answer comes from a real LLM branch. The more important result is candidate-pool recall:

| Candidate object | Correct label coverage |
|---|---:|
| Final ranked differential top-3 | 46/49 |
| Final ranked differential top-5 | 47/49 |
| Broader resolver candidate pool | 49/49 |

The broader resolver pool averages only `4.08` candidate rows and `3.98` unique diagnoses per case. This means Notebook `30` shifts the bottleneck from candidate generation to candidate selection.

## Notebook 31 Neural Candidate Pool Resolver

Notebook `31` is an offline final-head experiment over the completed Notebook `30` live candidate pool.

- `notebooks/31_neural_candidate_pool_resolver.ipynb`
- `reports/algorithmic_ledger/neural_candidate_pool_resolver_report.md`
- artifact root: `artifacts/trajectory_replicates/neural_candidate_pool_resolver_49case_v1/`

Selected policy:

```text
compact_neural_candidate_resolver_v1
model = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4)
selection = argmax neural score within the Notebook 30 candidate pool
```

Result:

| System | Correct | Accuracy | Mean selected requests | Mean total branch requests |
|---|---:|---:|---:|---:|
| Notebook `30` base branch | 42/49 | 0.857 | 6.80 | 6.80 |
| Notebook `30` hand resolver | 44/49 | 0.898 | 6.78 | 12.10 |
| Notebook `31` compact neural resolver | 46/49 | 0.939 | 6.78 | 12.10 |
| Candidate-pool oracle, diagnostic only | 49/49 | 1.000 | 6.78 | 12.10 |

Paired against Notebook `30`, Notebook `31` has two wins and zero regressions. It fixes:

| Case | True diagnosis | Notebook `30` | Notebook `31` |
|---|---|---|---|
| `test:81691` | Croup | Acute otitis media | Croup |
| `test:35039` | Myocarditis | Pericarditis | Myocarditis |

The remaining Notebook `31` misses are Acute rhinosinusitis vs Chronic rhinosinusitis, Bronchitis vs URTI, and Pericarditis vs Anemia.

Notebook `31` is the strongest pre-ablation learned final-head result over the Notebook `30` candidate pool, but it still does not achieve the `47/49+` target as a selected policy. The `49/49` result is only a label-using oracle ceiling.

## Notebook 32 Resolver Ablation Lab

Notebook `32` is an offline resolver-ablation workflow over the Notebook `30` candidate pool and Notebook `31` neural resolver artifact.

- `notebooks/32_resolver_ablation_lab.ipynb`
- `reports/algorithmic_ledger/resolver_ablation_lab_report.md`
- artifact root: `artifacts/trajectory_replicates/resolver_ablation_lab_49case_v1/`

It evaluates heuristic rank fusion, graph/Bayes/MLP posterior heads, supervised row scorers, tree ensembles, disease-name/family features, pairwise ranking, branch-differential voting, prior one-shot signals, and explicitly marked non-deployable diagnostic ceilings.

Result:

| System | Correct | Status |
|---|---:|---|
| Notebook `30` hand resolver | 44/49 | reference |
| Notebook `31` compact neural resolver | 46/49 | reference |
| Notebook `32` strict validation-selected resolver | 45/49 | not promoted |
| Notebook `32` best deployable live-diagnostic resolver, `gradient_boosting_name_family` | 47/49 | confirmation candidate |
| Notebook `32` full-evidence diagnostic blend | 48/49 | non-deployable because it uses unobserved evidence |
| Candidate-pool label oracle | 49/49 | non-deployable label oracle |

The `47/49` row fixes the Pericarditis/Anemia miss but still misses acute-vs-chronic rhinosinusitis and Bronchitis-vs-URTI. Because it was selected by inspecting the 49-case ablation table rather than by strict validation selection, it should be independently confirmed before being treated as a promoted resolver.

## Notebook 33 Close-Confounder Discriminator

Notebook `33` is an offline targeted evidence experiment over the Notebook `30` candidate pool and the fixed Notebook `32` `gradient_boosting_name_family` resolver candidate.

- `notebooks/33_close_confounder_discriminator.ipynb`
- `reports/algorithmic_ledger/close_confounder_discriminator_report.md`
- artifact root: `artifacts/trajectory_replicates/close_confounder_discriminator_49case_v1/`

Policy:

```text
flag = same-family or near-name top-2 candidate pair with enough missing pair utility
extra evidence = up to 2 roots ranked by train-derived JS separation * root MI
override = challenger only if extra-root log Bayes factor >= 2.0
```

Result:

| System | Correct | Mean selected requests | Mean total branch requests |
|---|---:|---:|---:|
| Notebook `31` compact neural resolver | 46/49 | 6.78 | 12.10 |
| Notebook `32` GBM diagnostic row | 47/49 | 6.78 | 12.10 |
| Notebook `33` close-confounder discriminator | 48/49 | 7.02 | 12.35 |

Paired result against Notebook `32` GBM:

| Metric | Value |
|---|---:|
| Wins | 1 |
| Regressions | 0 |
| Flagged cases | 6/49 |
| Overrides | 1/49 |
| Extra evidence requests | 12 total |

The single win is `test:11655`, where the base resolver selected URTI, the challenger was Bronchitis, and roots `E_55` and `E_54` produced a decisive extra-root Bayes factor for Bronchitis. The Pericarditis case remains stable: it is flagged against PSVT, but the extra evidence strongly rejects the challenger. The remaining miss is `test:111176`, Acute rhinosinusitis predicted as Chronic rhinosinusitis; the selected discriminator roots do not support an override.

Notebook `33` is the first saved non-oracle candidate-pool follow-up at `48/49`, but it should still be reported as an offline confirmation candidate. The fixed GBM base row came from Notebook `32` ablation inspection, so a fresh trace or held-out candidate-pool confirmation is needed before promotion.

## Notebook 34 Candidate-Recall-Gated Branching Efficiency

Notebook `34` is an offline pruning replay over the saved Notebook `30` hypothesis-forced branch pool and the Notebook `33` final discriminator.

- `notebooks/34_candidate_recall_gated_branching_efficiency_lab.ipynb`
- `reports/algorithmic_ledger/candidate_recall_gated_branching_efficiency_report.md`
- artifact root: `artifacts/trajectory_replicates/candidate_recall_gated_branching_efficiency_49case_v1/`

The control question is whether we can keep the useful Notebook `30` discovery, `49/49` candidate-pool recall, while paying for fewer live branches.

Replay policy:

```text
keep base + graph/Bayes/MLP pseudo-candidates
include LLM branch candidates only when branch_trigger_probability >= threshold
limit branch candidates to the first k highest-priority hypothesis branches
resolve with Notebook 32 gradient_boosting_name_family
apply Notebook 33 close-confounder discriminator when the replayed top pair matches
```

Selected efficiency candidate:

```text
candidate_recall_gated_branching_v1
branch_trigger_threshold = 0.80
branch_budget = 1
```

Result:

| System | Candidate-pool recall | Correct | Mean selected requests | Mean total requests | P90 total | Max total |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `33` native replay | 49/49 | 48/49 | 7.02 | 12.35 | 26.8 | 85 |
| Notebook `34` selected pruning replay | 49/49 | 48/49 | 7.16 | 8.98 | 19.6 | 48 |

The reduction is `27.3%` in mean total branch requests while preserving both `49/49` candidate-pool recall and `48/49` final accuracy.

The important finding is that threshold alone is not the main efficiency lever. The useful Croup branch and one unnecessary branch case have nearly identical branch-trigger probabilities. The stronger lever is branch budget: one highest-priority hypothesis branch is enough to preserve the candidate-pool oracle and final result on this replay.

The absolute lowest-cost same-accuracy diagnostic row uses a threshold near `0.8058`, but that is a knife-edge between two nearly tied cases. The selected live-confirmation candidate uses `0.80` for robustness.

Notebook `34` is not a new live result. It is an offline pruning replay over already observed branches. Notebook `35` keeps the efficiency result while replacing the hard branch cap with an adaptive continuation controller; that adaptive policy should be live-confirmed before promotion or MEDDxAgent-style hard-cap comparisons.

## Notebook 35 Adaptive Value Branching Controller

Notebook `35` answers the design concern raised by Notebook `34`: a fixed one-branch budget is efficient, but it is not an intelligent general controller. Notebook `35` therefore allows up to three branches and decides after each completed branch whether another branch is worth launching.

- `notebooks/35_adaptive_value_branching_controller.ipynb`
- `reports/algorithmic_ledger/adaptive_value_branching_controller_report.md`
- artifact root: `artifacts/trajectory_replicates/adaptive_value_branching_controller_49case_v1/`

Adaptive policy:

```text
branch_trigger_threshold = 0.80
max_branches = 3
continuation_value_threshold = 0.40
resolver = Notebook 32 gradient_boosting_name_family
final discriminator = Notebook 33 close-confounder Bayes-factor override
```

Continuation value:

```text
continuation_value = unresolved_mass * next_priority_ratio * suppression
unresolved_mass = 0.45 * ledger_disagreement
                + 0.30 * resolver_entropy
                + 0.25 * margin_uncertainty
```

Suppression terms reduce the value of another full LLM branch when the current decision has enough graph/Bayes/MLP support, or when the cheaper close-confounder discriminator can adjudicate the top pair.

Result:

| System | Candidate-pool recall | Correct | Mean selected requests | Mean total requests | P90 total | Max total |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `33` native replay | 49/49 | 48/49 | 7.02 | 12.35 | 26.8 | 85 |
| Notebook `34` fixed one-branch replay | 49/49 | 48/49 | 7.16 | 8.98 | 19.6 | 48 |
| Notebook `35` adaptive controller replay | 49/49 | 48/49 | 7.16 | 8.98 | 19.6 | 48 |

The adaptive controller is allowed to launch branch 2/3, but on this replay it does not. After branch 1, every high-trigger case either has sufficient current graph/Bayes/MLP decision support, has a cheaper close-confounder discriminator available, or has continuation value below the selected threshold. This preserves the Notebook `34` efficiency result while avoiding the conceptual weakness of a hard one-branch cap.

Notebook `35` is still offline replay over saved branches. The next clean step is a live confirmation using the adaptive controller, then a hard-cap and mean-cost comparison against MEDDxAgent-style evidence budgets.

## Notebook 36 Adaptive Branching Stress Test

Notebook `36` is an artificial offline stress test for Notebook `35`. It asks whether we can prove, without new API calls, that branch 2/3 would fire when branch 1 fails.

- `notebooks/36_adaptive_branching_stress_test.ipynb`
- `reports/algorithmic_ledger/adaptive_branching_stress_test_report.md`
- artifact root: `artifacts/trajectory_replicates/adaptive_branching_stress_test_49case_v1/`

The stress test uses saved Notebook `30` branches and runs three adversarial probes:

- remove branch 1 and let branch 2 become the first available branch
- spend branch 1 but hide its candidate contribution, simulating a no-signal first branch
- sweep continuation thresholds under the no-signal branch-1 condition

Prefix diagnostic:

| Replay prefix | Candidate-pool recall | Correct |
|---|---:|---:|
| No branch | 46/49 | 45/49 |
| Branch 1 only | 49/49 | 48/49 |
| Branch 2 only | 47/49 | 46/49 |
| Branch 3 only | 47/49 | 46/49 |
| Branch 1 + 2 + 3 | 49/49 | 48/49 |

Stress result:

| Scenario | Candidate-pool recall | Correct | Mean total requests | Branch 2 launches | Branch 3 launches |
|---|---:|---:|---:|---:|---:|
| Native Notebook `35` adaptive replay | 49/49 | 48/49 | 8.98 | 0 | 0 |
| Branch 1 removed, branch 2 remapped first | 47/49 | 46/49 | 8.33 | 6 | 0 |
| Branch 1 no-signal, selected controller | 46/49 | 45/49 | 8.94 | 0 | 0 |
| Branch 1 no-signal, threshold `0.05` | 47/49 | 46/49 | 11.18 | 6 | 3 |

The key result is not that Notebook `35` is wrong. The key result is that this 49-case saved pool cannot prove branch-2/3 recovery. For Croup and Myocarditis, branch 1 is the only saved branch that contains the correct diagnosis; branch 2 and branch 3 do not recover them even when forced. For Panic attack, branch 2/3 can recover the true candidate, but the selected continuation threshold does not fire after a no-signal branch 1.

Interpretation:

- Notebook `35` proves efficient non-overbranching on the saved replay.
- Notebook `36` does not prove adaptive recovery under natural branch-2/3 need because this 49-case pool has no natural positive examples.
- The main artificial failure mode is false stability: graph/Bayes/MLP can agree on the wrong current top diagnosis, suppressing continuation.
- To prove branch 2/3 behavior scientifically, the next step must collect a larger branch pool or run a live confirmation where branch 2/3 opportunities occur naturally.

## Notebook 37 Adaptive Live Balanced Confirmation

Notebook `37` is the larger live confirmation runner.

- `notebooks/37_adaptive_value_branching_live_balanced_confirmation.ipynb`
- `reports/algorithmic_ledger/adaptive_value_branching_live_balanced_confirmation_report.md`
- artifact root: `artifacts/trajectory_replicates/adaptive_value_branching_live_balanced2_v1/`

Design:

```text
cohort = 2 held-out test cases per pathology
exclude original 49-case confirmation set = true
expected n = 98 because DDXPlus has 49 pathologies
LLM = gpt-4.1-mini
temperature = 0.0
top_p = 1.0
branch_trigger_threshold = 0.80
max_branches = 3
continuation_value_threshold = 0.40
```

The notebook restores ranked differential reporting for:

- base Notebook `13`-style branch top-1/top-3/top-5
- live adaptive branch-judge top-1/top-3/top-5
- final candidate-pool GBM plus close-confounder discriminator top-1/top-3/top-5

Result:

| System | Correct | Accuracy | Top-3 | Top-5 | Mean selected requests | Mean total requests |
|---|---:|---:|---:|---:|---:|---:|
| Base Notebook `13`-style branch | 83/98 | 0.847 | 0.888 | 0.908 | 8.20 | 8.20 |
| Notebook `37` branch-judge selected | 86/98 | 0.878 | 0.908 | 0.929 | 8.20 | 8.31 |
| Notebook `37` GBM + close-confounder final | 88/98 | 0.898 | 0.939 | 0.939 | 8.37 | 8.43 |

The intermediate branch-judge output has `5` wins and `2` regressions versus the base branch. The final GBM plus close-confounder output removes those regressions, ending with `5` wins and `0` regressions versus base.

Important failure analysis:

- candidate-pool recall is `92/98`, not the `49/49` seen in the original 49-case Notebook `30`/`35` pool
- final top-3/top-5 are also `92/98`, exactly matching candidate-pool recall
- only `1/98` cases crossed the `0.80` branch-trigger threshold
- total real LLM branches spawned: `1`
- branch 2/3 launches: `0`
- most gains came from graph/Bayes/MLP pseudo-candidate resolution, not fresh LLM branches

Final misses split cleanly:

| Failure mode | Cases |
|---|---:|
| True diagnosis absent from final candidate pool | 6 |
| True diagnosis present but ranked below confounder | 4 |

Interpretation:

Notebook `37` improves the base branch with modest added cost and no final regressions, but it does not reproduce the optimistic `48/49` replay rate. The old 49-case result was not a stable percentage estimate for a fresh balanced cohort. Candidate-pool generation and branch-trigger calibration are now the main bottlenecks.

## Notebook 38 Live Calibration Cohort

Notebook `38` is the completed 196-case live development run.

- `notebooks/38_live_adaptive_branching_calibration_cohort.ipynb`
- `reports/algorithmic_ledger/live_adaptive_branching_calibration_cohort_report.md`
- artifact root: `artifacts/trajectory_replicates/adaptive_value_branching_live_calibration196_v1/`

Purpose:

Notebook `37` showed that the selected `0.80` branch trigger was too conservative on fresh live terminal states and that candidate-pool recall was no longer perfect. Notebook `38` is therefore a calibration cohort, not a final confirmation cohort.

Design:

```text
cohort = 4 held-out test cases per pathology
expected n = 196
exclude prior live benchmark cases = true
LLM = gpt-4.1-mini
temperature = 0.0
top_p = 1.0
exploratory branch_trigger_threshold = 0.20
max_branches = 3
exploratory continuation_value_threshold = 0.20
target candidate-pool recall for calibration = 0.98
```

The lower thresholds are intentional because the calibration run needs to collect enough live branch/candidate-pool behavior to estimate future thresholds. The notebook writes calibration-specific artifacts:

- `live_calibration_branch_trigger_threshold_sweep.csv`
- `live_calibration_resolver_margin_sweep.csv`
- `live_calibration_candidate_source_recall.csv`
- `live_calibration_failure_modes.csv`
- `live_calibration_post_run_analysis_summary.json`
- `live_calibration_post_run_case_outcomes.csv`
- `selected_live_calibration_policy.json`

Result:

| System | Correct | Accuracy | Top-3 | Top-5 | Mean selected requests | Mean total requests | P90 total requests | Max total requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Notebook `13`-style branch | 172/196 | 0.878 | 0.944 | 0.964 | 6.77 | 6.77 | 17 | 24 |
| Notebook `38` branch-judge selected | 183/196 | 0.934 | 0.974 | 0.985 | 6.79 | 9.52 | 22 | 85 |
| Notebook `38` GBM + close-confounder final | 184/196 | 0.939 | 0.990 | 0.990 | 6.77 | 9.56 | 22 | 85 |

Candidate-pool recall:

| Candidate subset | Recall | Mean pool size |
|---|---:|---:|
| Base only | 166/196 = 0.847 | 0.96 |
| Base + pseudo graph/Bayes/MLP | 181/196 = 0.923 | 3.62 |
| Base + real branches | 179/196 = 0.913 | 1.13 |
| All candidates | 194/196 = 0.990 | 3.79 |

Paired final result versus the base branch:

| Outcome | Cases |
|---|---:|
| Both correct | 171 |
| Notebook `38` final only correct | 13 |
| Base only correct | 1 |
| Both wrong | 11 |

Branching behavior:

- branch trigger rate: `23/196 = 0.117`
- total branches spawned: `42`
- branch count distribution: `173` cases with no branch, `12` with one branch, `3` with two branches, `8` with three branches
- branch 2/3 therefore occur naturally on this larger live calibration cohort

Failure pattern:

- final misses: `12/196`
- candidate-pool misses: `2/196`
- resolver misses with the truth in the pool: `10/196`
- all four Acute rhinosinusitis cases were in the candidate pool but resolved as Chronic rhinosinusitis
- the only final regression was `test:113762`, Ebola, where the base branch was correct but the resolver selected URTI

Interpretation:

Notebook `38` is much more encouraging than Notebook `37`: candidate-pool recall recovers from `92/98` to `194/196`, and final top-3/top-5 also reach `194/196`. The main problem has shifted from broad candidate discovery to final resolver discrimination among close confounders. The cost issue is a long tail: average total branch cost is reasonable, but hard cases can still spend up to `85` total branch requests.

The labels from Notebook `38` may be used to tune thresholds. Therefore Notebook `38` accuracy must not be reported as held-out final performance. The correct next step is to freeze a separate confirmation policy using these calibration artifacts, then run a new held-out 98-case or 196-case confirmation cohort.

## Notebook 39 Cross-Cohort Artifact Calibration Lab

Notebook `39` is an offline calibration analysis over saved Notebook `33`, `37`, and `38` artifacts.

- `notebooks/39_cross_cohort_artifact_calibration_lab.ipynb`
- `reports/algorithmic_ledger/cross_cohort_artifact_calibration_lab_report.md`
- artifact root: `artifacts/trajectory_replicates/cross_cohort_artifact_calibration_lab_v1/`

Purpose:

Notebook `38` showed that the candidate pool is nearly complete on a larger live calibration cohort, but the final resolver still misses close confounders. Notebook `39` asks whether the accumulated artifacts can calibrate that resolver layer without another live run.

Pooled artifact result:

| Policy | Claim type | Correct | Accuracy | Candidate-pool recall |
|---|---|---:|---:|---:|
| Current saved final pipeline | deployable saved artifact | 320/343 | 0.933 | 335/343 |
| Raw GBM candidate argmax | diagnostic | 317/343 | 0.924 | 335/343 |
| Notebook `38`-selected calibration rule layer | calibration-only candidate | 323/343 | 0.942 | 335/343 |
| Pooled no-regret label-fit rules | diagnostic label-fit | 330/343 | 0.962 | 335/343 |
| Candidate-pool oracle | non-deployable oracle | 335/343 | 0.977 | 335/343 |

The selected calibration rule is:

```text
if current final = Chronic rhinosinusitis
and Acute rhinosinusitis is in the candidate pool
and visible evidence has E_103 present
then promote Acute rhinosinusitis
```

Effect:

- fixes three Notebook `38` Acute rhinosinusitis -> Chronic rhinosinusitis misses
- causes zero pooled regressions across the saved 343-case artifacts
- does not trigger on the older 49/98-case artifacts

Interpretation:

Notebook `39` does not prove a final new method. It shows that a modest calibration layer can improve the pooled saved artifacts from `320/343` to `323/343`, while label-fit diagnostic rules can reach `330/343`. The oracle ceiling is `335/343`, so the remaining problem is mostly resolver discrimination among close confounders, plus an `8/343` candidate-pool recall gap.

The selected `E_103` rule is useful but fragile: train/validate disease statistics show E_103 present in `0.742` of Acute rhinosinusitis and `0.843` of Chronic rhinosinusitis, so it should not be treated as a robust medical discriminator. It is a calibration-only rescue candidate that needs fresh frozen confirmation before promotion.

## Notebook 40 Synthetic-To-Live Listwise Resolver

Notebook `40` is an offline resolver-transfer lab over the saved Notebook `33`, `37`, and `38` candidate pools.

- `notebooks/40_synthetic_to_live_listwise_resolver.ipynb`
- `scripts/synthetic_to_live_listwise_resolver_nb40.py`
- `reports/algorithmic_ledger/synthetic_to_live_listwise_resolver_report.md`
- artifact root: `artifacts/trajectory_replicates/synthetic_to_live_listwise_resolver_v1/`

Purpose:

Notebook `39` showed that the pooled candidate-pool oracle is `335/343`, while the current saved final pipeline is `320/343`. Notebook `40` asks whether a resolver trained on DDXPlus train/validate synthetic partial evidence states can transfer into these live candidate pools and close that gap without hand-written one-case rules.

Training and evaluation setup:

- synthetic train states: `4000`
- synthetic validation states: `2000`
- synthetic validation candidate-pool recall: `1810/2000 = 0.905`
- live artifact cohorts: Notebook `33` 49-case, Notebook `37` 98-case, Notebook `38` 196-case
- live pooled candidate-pool recall: `335/343 = 0.977`
- no live API calls

Pooled result:

| Policy | Claim type | Correct | Accuracy | Candidate-pool recall |
|---|---|---:|---:|---:|
| Current saved final pipeline | saved artifact reference | 320/343 | 0.933 | 335/343 |
| Candidate-pool oracle | non-deployable oracle | 335/343 | 0.977 | 335/343 |
| Synthetic logistic | synthetic-only transfer | 315/343 | 0.918 | 335/343 |
| Synthetic hist-GBM | synthetic-only transfer | 315/343 | 0.918 | 335/343 |
| Synthetic listwise MLP | synthetic-only transfer | 307/343 | 0.895 | 335/343 |
| Synthetic pairwise Bradley-Terry | synthetic-only transfer | 314/343 | 0.915 | 335/343 |
| Artifact LOCO logistic | synthetic plus leave-one-cohort calibration | 317/343 | 0.924 | 335/343 |
| Artifact LOCO hist-GBM | synthetic plus leave-one-cohort calibration | 315/343 | 0.918 | 335/343 |
| Artifact-fit GBM | diagnostic label-fit | 319/343 | 0.930 | 335/343 |

Selected policy:

- policy name: `artifact_loco_logistic`
- pooled result: `317/343`
- wins versus current final pipeline: `1`
- regressions versus current final pipeline: `4`
- status: not promoted

Interpretation:

Notebook `40` gives a clean negative result. Synthetic partial evidence states are useful for creating large resolver training sets, but they do not match the live candidate-pool distribution well enough to beat the current saved final pipeline. Even leave-one-cohort-out artifact calibration underperforms the current final result. This means the near-oracle candidate-pool gap is not solved by a generic synthetic-to-live listwise or pairwise resolver.

The practical conclusion is that the project should not claim a solved resolver. Candidate generation is strong, but final discrimination among close confounders remains the bottleneck. The next defensible improvement would need either a frozen held-out confirmation of a pre-registered calibration rule layer, or a resolver that can request missing discriminator evidence before committing to the final top-1.

## Notebook 41 Final Capped Hypothesis-Branching Confirmation

Notebook `41` is the final prepared live confirmation runner for the candidate-pool architecture.

- `notebooks/41_final_capped_hypothesis_branching_confirmation.ipynb`
- `scripts/final_capped_hypothesis_branching_confirmation_nb41.py`
- `reports/algorithmic_ledger/final_capped_hypothesis_branching_confirmation_report.md`
- live artifact root: `artifacts/trajectory_replicates/final_capped_hypothesis_branching_confirmation100_v1/`
- dry-run smoke artifact root: `artifacts/trajectory_replicates/final_capped_hypothesis_branching_confirmation100_dryrun_smoke_v1/`

Purpose:

The project is no longer expanding to a 700+ case live calibration set. Notebook `41` freezes a clean, budget-controlled final live run over 100 held-out cases.

Frozen setup:

| Control | Value |
|---|---:|
| Cohort | 100 held-out test cases |
| Sampling | two per pathology plus two extra held-out rows |
| LLM | `gpt-4.1-mini`, temperature `0.0`, top-p `1.0` |
| Base evidence cap | `24` |
| Spawned branch cap | `8` |
| Hard total request cap per case | `24` |
| Max spawned branches | `2` |
| Branch trigger threshold | `0.20` |
| Continuation threshold | `0.20` |
| Close-confounder extra-root layer | excluded |

Outputs:

- top-1, top-3, and top-5 accuracy
- candidate-pool recall
- selected-request cost
- total branch-request cost
- branch trigger usage
- paired base-vs-final outcomes
- final failure modes
- request-cost figures

Dry-run smoke verification:

| Check | Result |
|---|---:|
| Dry-run cases | 2 |
| Base correct | 1/2 |
| Final capped branch-selected correct | 1/2 |
| Wins/regressions | 0/0 |
| Mean total branch requests | 16.5 |
| Final resolver candidate-pool recall | 2/2 |
| Artifact contract | passed |

Interpretation:

Notebook `41` is prepared and verified, but it is not yet a live performance result. It is the appropriate final run because it removes the fragile close-confounder rescue layer, enforces a hard total request cap, restores differential-diagnosis metrics, and keeps the architecture clean enough to compare against fixed-budget diagnostic baselines.

## Notebook 42 Universal MEDDx Benchmark Adapter

Notebook `42` starts the cross-dataset MEDDx-style generalization phase.

- `notebooks/42_universal_meddx_benchmark_adapter.ipynb`
- `scripts/universal_meddx_benchmark_adapter_nb42.py`
- `reports/algorithmic_ledger/universal_meddx_benchmark_adapter_report.md`
- active pilot dry-run artifact root: `artifacts/universal_meddx/universal_meddx_benchmark_adapter_dryrun_smoke_v6_pilot3/`
- failed live diagnostic root: `artifacts/universal_meddx/universal_meddx_benchmark_adapter_v4/`

Purpose:

DDXPlus uses legal structured evidence roots, but the broader MEDDxAgent benchmark family uses patient-profile style datasets. Notebook `42` creates a universal harness that lets the LLM acquire information through natural-language questions instead of dataset-specific fields.

Architecture:

```text
dataset adapter
  -> universal patient case schema
  -> LLM natural-language question
  -> guarded hidden-profile retrieval simulator
  -> question-answer ledger
  -> LLM final ranked differential
  -> MEDDx-style evaluation
```

Supported adapters:

| Dataset | Notebook 42 status |
|---|---|
| DDXPlus | implemented immediately by converting structured evidence into hidden profile text |
| iCraft-MD | native MEDDxAgent JSONL adapter loaded from `external/meddxagent` |
| RareBench | native adapter using MEDDxAgent mappings plus the public HuggingFace RareBench data zip |

Live runs now use `49` total unique cases across the loaded adapters at all three reference budgets `5`, `10`, and `15`, for `147` intended workups.

The notebook now requires all enabled adapters to load before live evaluation. The dry-run smoke selected one case each from DDXPlus, iCraft-MD, and RareBench.

The first live `v4` run failed badly, but the failure was diagnostic rather than a valid benchmark result. At budget `15`, overall GTPA@1 was only `13/49`, with GTPA@5 `19/49`. The main cause was the harness: iCraft-MD/RareBench candidate lists were truncated alphabetically, hiding the true label in half of selected iCraft-MD cases and three quarters of selected RareBench cases; DDXPlus question-answer spans were also split so the simulator often returned a question without its `Answer:` value.

Notebook `42` `v5` repaired those issues by raising candidate text capacity, using iCraft case-level options, using RareBench subset-level options, keeping DDXPlus `Question? Answer: value` spans intact, skipping previously revealed spans, and adding dataset-specific prompting.

The `v5_pilot3` live run then completed the intended `9` workups. It was a real diagnostic failure rather than a broken-harness failure: DDXPlus `Influenza` was missed after the agent narrowed too early to bronchiolitis/pneumonia-style respiratory questions; iCraft-MD solved the levamisole-induced ANCA vasculitis option case; RareBench `Cockayne syndrome` was missed because the final resolver could not rank the correct rare disease inside a `216`-diagnosis LIRICAL candidate list.

Notebook `42` `v6_pilot3` is now the active pilot. It adds broad first-turn positive-finding inventory guidance, a simulator mode that returns high-yield positive spans for broad inventory questions, and a margin-gated visible-evidence reference-case prior. The prior uses Jaccard overlap with reference profiles, only applies to candidate lists with at least `10` diagnoses, and cannot force a final rerank unless its margin is at least `0.12`.

Metrics:

- `GTPA@1`
- `GTPA@3`
- `GTPA@5`
- capped true-diagnosis rank, missing rank `11`
- progress rate
- mean questions
- stop-before-budget rate

Dry-run smoke result:

| Dataset | Budget | Cases | GTPA@1 | GTPA@3 | GTPA@5 | Mean questions |
|---|---:|---:|---:|---:|---:|---:|
| DDXPlus | 5 | 1 | 0.000 | 0.000 | 0.000 | 2.0 |
| iCraft-MD | 5 | 1 | 0.000 | 0.000 | 0.000 | 2.0 |
| RareBench | 5 | 1 | 0.000 | 0.000 | 0.000 | 2.0 |

Interpretation:

The dry-run is not a performance claim because the no-API agent is scripted. The result means the repaired DDXPlus/iCraft-MD/RareBench adapter boundary, guarded patient simulator, question-answer ledger, MEDDx-style metrics, figures, casebase-prior summary, and artifact contract work. The next real step is the `v6_pilot3` live cross-dataset pilot at budgets `5`, `10`, and `15`.

## Notebook 43 Unified MEDDx-Style Hybrid Driver

Notebook `43` supersedes Notebook `42` as the preferred MEDDxAgent-style method pilot.

- `notebooks/43_unified_meddxstyle_hybrid_driver.ipynb`
- `scripts/unified_meddxstyle_hybrid_driver_nb43.py`
- `reports/algorithmic_ledger/unified_meddxstyle_hybrid_driver_report.md`
- dry-run artifact root: `artifacts/universal_meddx/unified_meddxstyle_hybrid_driver_dryrun_smoke_v1_pilot3/`

The key correction is architectural. MEDDxAgent does not ask one generic LLM prompt to solve every dataset. It uses a shared patient schema and driver, but separates history-taking, patient simulation, retrieval/few-shot support, and final diagnosis.

Notebook `43` adapts that pattern:

```text
dataset adapter
  -> universal patient schema
  -> MEDDx-style history-taking agent
  -> deterministic guarded patient simulator
  -> visible patient-profile ledger
  -> MEDDx-style diagnosis strategy agent
  -> dynamic similar-patient examples
  -> margin-gated reference-case prior rerank
  -> MEDDx-style evaluation
```

Dry-run smoke completed with no API calls:

| Dataset | Adapter status | Reference cases |
|---|---|---:|
| DDXPlus | loaded | 1500 |
| iCraft-MD | loaded | 140 |
| RareBench | loaded | 1120 |

The dry-run is not a performance claim because the no-API agents are scripted. It verifies that Notebook `43` writes the expected artifacts and is ready for the `v1_pilot3` live run: three selected cases across DDXPlus, iCraft-MD, and RareBench at budgets `5`, `10`, and `15`.

## Notebook 44 Graph-Phenotype MEDDx Driver

Notebook `44` is the successor to Notebook `43` for the MEDDxAgent-style multi-dataset line.

- `notebooks/44_unified_graph_phenotype_meddx_driver.ipynb`
- `scripts/unified_graph_phenotype_meddx_driver_nb44.py`
- `reports/algorithmic_ledger/unified_graph_phenotype_meddx_driver_report.md`
- dry-run artifact root: `artifacts/universal_meddx/unified_graph_phenotype_meddx_driver_dryrun_smoke_v1_pilot3/`

The Notebook `43` live `v1_pilot3` run diagnosed the failure:

| Dataset | Case | Budget 5 | Budget 10 | Budget 15 |
|---|---:|---:|---:|---:|
| DDXPlus | Influenza | wrong | correct | correct |
| iCraft-MD | Levamisole-induced ANCA vasculitis | correct | correct | correct |
| RareBench | Cockayne syndrome | wrong | wrong | wrong |

The RareBench issue was not candidate visibility. The true `Cockayne syndrome` label was present in the `216`-diagnosis LIRICAL option list. The problem was the representation: Notebook `43` used prose-token overlap and LLM ranking over long phenotype profiles, which favored a broad neurodevelopmental phenotype-category label.

Notebook `44` changes RareBench to an exact graph representation:

```text
visible phenotype names
  -> exact HPO phenotype-node set
  -> leave-one-case-out same-subset disease exemplar support
  -> graph-prior rerank/discriminator
```

No-API smoke passed. The dry run is not a performance claim, but it verifies the artifact contract and shows that the new graph-phenotype layer ranks `Cockayne syndrome` as the top graph-supported diagnosis on the previously failed RareBench smoke case.

Follow-up live pilot:

- Notebook `44` live `v1_pilot3` completed the same three-case, three-budget pilot shape as Notebook `43`
- row-level result improved from Notebook `43`'s `5/9` top-1/top-3/top-5 to `9/9` top-1/top-3/top-5
- DDXPlus, iCraft-MD, and RareBench were all correct at budgets `5`, `10`, and `15`
- the RareBench base diagnosis still initially selected the broad neurodevelopmental category, but the graph-phenotype layer and discriminator promoted `Cockayne syndrome`

Scaled evaluation config:

- active Notebook `44` config is now `RUN_VERSION_SUFFIX = "v1_eval30"`
- `LIVE_TOTAL_MAX_CASES = 30`
- budgets remain `[5, 10, 15]`
- intended scaled run is `90` live workups, balanced across the three datasets when all adapters load
- no-API smoke passed under `artifacts/universal_meddx/unified_graph_phenotype_meddx_driver_dryrun_smoke_v1_eval30/`
