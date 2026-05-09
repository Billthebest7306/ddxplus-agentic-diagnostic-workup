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

## Project Direction

- [project_direction_and_claims_assessment.md](project/project_direction_and_claims_assessment.md)
- [phase_next_rigorous_evaluation_plan.md](project/phase_next_rigorous_evaluation_plan.md)
