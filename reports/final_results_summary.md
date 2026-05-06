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

The strongest final proposed method remains notebook `13`: it uses the LLM as the evidence-acquisition controller and the partial-evidence MLP as an online stopping signal. The 49-case confirmation reaches `43/49 = 0.878` accuracy with `6.59` mean evidence requests.

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
