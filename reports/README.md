# Reports Index

This folder now keeps the final reports at the top level and older/supporting reports in subfolders.

## Final Reports

- [final_report.md](final_report.md): main project walkthrough and presentation outline.
- [final_results_summary.md](final_results_summary.md): detailed metric summary across artifacts.

## Baselines

- [baseline_summary.md](baselines/baseline_summary.md)
- [baseline_results_and_next_steps.md](baselines/baseline_results_and_next_steps.md)
- [full_evidence_one_shot_comparator.md](baselines/full_evidence_one_shot_comparator.md)
- [partial_evidence_matched_comparator.md](baselines/partial_evidence_matched_comparator.md)
- [matched_evidence_integrated_comparison_report.md](baselines/matched_evidence_integrated_comparison_report.md)
- [integrated_evidence_budget_comparison_report.md](baselines/integrated_evidence_budget_comparison_report.md)
- [lambda_cost_sensitive_policy_report.md](baselines/lambda_cost_sensitive_policy_report.md)
- [results_assessment.md](baselines/results_assessment.md)
- [proposed_improvement_1_results.md](baselines/proposed_improvement_1_results.md)
- [sequential_api_guide.md](baselines/sequential_api_guide.md)

## Hybrid And Sequential Policy

- [sequential_policy_refinement_report.md](hybrid/sequential_policy_refinement_report.md)
- [hybrid_mlp_feedback_report.md](hybrid/hybrid_mlp_feedback_report.md)
- [stopping_policy_ablation_report.md](hybrid/stopping_policy_ablation_report.md)
- [live_selected_hybrid_stopping_confirmation.md](hybrid/live_selected_hybrid_stopping_confirmation.md)
- [hybrid_v2_mlp_discriminative_shortlist_report.md](hybrid/hybrid_v2_mlp_discriminative_shortlist_report.md)
- [notebook13_stop_policy_sensitivity_report.md](hybrid/notebook13_stop_policy_sensitivity_report.md)

## Architecture And Ledger

- [ledger_implementation_explained.md](architecture/ledger_implementation_explained.md)
- [evidence_ledger_algorithm_and_improvements.md](architecture/evidence_ledger_algorithm_and_improvements.md)
- [architecture_v1_freeze_and_experimental_scope.md](architecture/architecture_v1_freeze_and_experimental_scope.md)
- [proposed_multi_agent_architecture.md](architecture/proposed_multi_agent_architecture.md)
- [multi_agent_architecture_simple_diagram.md](architecture/multi_agent_architecture_simple_diagram.md)

## Algorithmic Ledger

- [medkgi_style_graph_ledger_v1_report.md](algorithmic_ledger/medkgi_style_graph_ledger_v1_report.md)
- [live_medkgi_graph_shortlist_pilot.md](algorithmic_ledger/live_medkgi_graph_shortlist_pilot.md)
- [graph_advisory_hybrid_shortlist_report.md](algorithmic_ledger/graph_advisory_hybrid_shortlist_report.md): Notebook 18 graph-advisory pilot; rejected but useful for rare-disease recovery analysis.
- [bayesian_voi_ledger_offline_report.md](algorithmic_ledger/bayesian_voi_ledger_offline_report.md): Notebook 19 offline Bayesian VOI ledger; full 49-case run completed and rejected as a replacement controller.
- [llm_led_graph_ledger_context_report.md](algorithmic_ledger/llm_led_graph_ledger_context_report.md): Notebook 20 corrected graph-ledger experiment; graph context is advisory prompt state while the LLM remains the question chooser.
- [graph_context_policy_lab_report.md](algorithmic_ledger/graph_context_policy_lab_report.md): Notebook 21 offline policy lab; graph context is useful as a critic/ranking signal, but no non-oracle hand-rule variant beats Notebook 13.
- [graph_posterior_final_adjudicator_report.md](algorithmic_ledger/graph_posterior_final_adjudicator_report.md): Notebook 22 offline graph-posterior final critic; improves the saved Notebook 13 49-case trace to 44/49 with no new requests.
- [calibrated_graph_bayes_rescue_reranker_report.md](algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md): Notebook 23 calibrated graph/Bayes rescue reranker; improves the saved Notebook 13 49-case trace to 47/49 with 6.96 mean requests and zero regressions.
- [live_graph_bayes_rescue_confirmation_report.md](algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md): Notebook 24 live confirmation of Notebook 23; rescue was not promoted, but the fresh live base reached 45/49 with 6.20 mean requests.
- [live_base_trajectory_replicates_report.md](algorithmic_ledger/live_base_trajectory_replicates_report.md): Notebook 25 three-replicate runner for Notebook 13-style base trajectory collection; rescue disabled, dry-run validated.
- [offline_branching_trajectory_lab_report.md](algorithmic_ledger/offline_branching_trajectory_lab_report.md): Notebook 26 offline branching lab over Notebook 13/24/25 trajectories; majority vote stays at 43/49, oracle best-of-five reaches 47/49, and sparse graph/Bayes-adjudicated branching motivated Notebook 27.
- [live_targeted_branching_confirmation_report.md](algorithmic_ledger/live_targeted_branching_confirmation_report.md): Notebook 27 live targeted-branching confirmation; improved its own live base from 42/49 to 43/49 but introduced one regression, so it is a partial confirmation of branch diversity and not a promoted method.
- [mlp_gated_confounder_graph_bayes_branching_report.md](algorithmic_ledger/mlp_gated_confounder_graph_bayes_branching_report.md): Notebook 28 learned branch-trigger MLP plus graph/Bayes/MLP resolver over fresh LLM branches and pseudo-candidates; live run reached 44/49 with zero regressions versus its base, and post-hoc analysis shows the next opportunity is ranked-differential listwise adjudication.
- [listwise_differential_graph_bayes_adjudicator_report.md](algorithmic_ledger/listwise_differential_graph_bayes_adjudicator_report.md): Notebook 29 offline listwise adjudicator over Notebook 28 ranked differentials; improves Notebook 28 to 45/49 with zero regressions, but remains diagnostic because it does not reach the 47/49 promotion target.
- [hypothesis_forced_differential_branching_report.md](algorithmic_ledger/hypothesis_forced_differential_branching_report.md): Notebook 30 prospective branching candidate; assigns explicit graph/Bayes/MLP challenger hypotheses and discriminator roots to fresh LLM branches, with no-spend dry-run verification.
- [neural_candidate_pool_resolver_report.md](algorithmic_ledger/neural_candidate_pool_resolver_report.md): Notebook 31 offline neural resolver over the completed Notebook 30 candidate pool; improves Notebook 30 from 44/49 to 46/49 with zero regressions and confirms a 49/49 diagnostic candidate-pool oracle.
- [resolver_ablation_lab_report.md](algorithmic_ledger/resolver_ablation_lab_report.md): Notebook 32 offline resolver ablation lab over the Notebook 30/31 candidate pool; strict validation selection does not improve on Notebook 31, while the best deployable live-diagnostic row reaches 47/49 and requires independent confirmation.
- [close_confounder_discriminator_report.md](algorithmic_ledger/close_confounder_discriminator_report.md): Notebook 33 close-confounder discriminator over the Notebook 30/32 candidate pool; asks two targeted train-statistic-ranked roots for six flagged cases and reaches 48/49 as an offline confirmation candidate.
- [candidate_recall_gated_branching_efficiency_report.md](algorithmic_ledger/candidate_recall_gated_branching_efficiency_report.md): Notebook 34 offline branch-pruning replay; preserves 49/49 candidate-pool recall and 48/49 final accuracy with one highest-priority branch and 8.98 mean total requests.
- [adaptive_value_branching_controller_report.md](algorithmic_ledger/adaptive_value_branching_controller_report.md): Notebook 35 adaptive branch-continuation replay; allows up to three branches but uses a label-free continuation-value controller, preserving 49/49 candidate-pool recall and 48/49 final accuracy at 8.98 mean total requests.
- [adaptive_branching_stress_test_report.md](algorithmic_ledger/adaptive_branching_stress_test_report.md): Notebook 36 artificial branch stress test; shows the saved 49-case pool has no natural branch-2/3 rescue cases and that branch 2/3 do not reliably recover when branch 1 is artificially removed.
- [adaptive_value_branching_live_balanced_confirmation_report.md](algorithmic_ledger/adaptive_value_branching_live_balanced_confirmation_report.md): Notebook 37 fresh balanced two-per-pathology live confirmation; final GBM + close-confounder output improves the base from 83/98 to 88/98 with zero final regressions, but candidate-pool recall drops to 92/98 and the adaptive branch trigger fires on only 1/98 cases.
- [live_adaptive_branching_calibration_cohort_report.md](algorithmic_ledger/live_adaptive_branching_calibration_cohort_report.md): Notebook 38 completed 196-case live calibration cohort; final GBM + close-confounder output improves the base from 172/196 to 184/196, candidate-pool recall reaches 194/196, and the remaining work is resolver calibration plus branch-cost tail control before a frozen held-out confirmation.
- [cross_cohort_artifact_calibration_lab_report.md](algorithmic_ledger/cross_cohort_artifact_calibration_lab_report.md): Notebook 39 offline cross-cohort calibration lab over Notebooks 33/37/38; pooled saved final result is 320/343, calibration-selected rule layer reaches 323/343, diagnostic label-fit rules reach 330/343, and the candidate-pool oracle is 335/343.
- [synthetic_to_live_listwise_resolver_report.md](algorithmic_ledger/synthetic_to_live_listwise_resolver_report.md): Notebook 40 offline synthetic-to-live resolver lab; synthetic-only and leave-one-cohort artifact-calibrated listwise/pairwise resolvers do not beat the current saved final pipeline, so the selected LOCO resolver is not promoted.
- [final_capped_hypothesis_branching_confirmation_report.md](algorithmic_ledger/final_capped_hypothesis_branching_confirmation_report.md): Notebook 41 final capped 100-case live confirmation runner; removes the close-confounder extra-root layer, enforces base/branch/total request caps, restores top-3/top-5 reporting, and has passed no-API dry-run smoke verification.

## Project Direction

- [project_direction_and_claims_assessment.md](project/project_direction_and_claims_assessment.md)
- [phase_next_rigorous_evaluation_plan.md](project/phase_next_rigorous_evaluation_plan.md)
