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

The strongest current sequential results are from a 24-case balanced live sweep with `gpt-4.1-mini`, `temperature=0.0`, `top_p=1.0`, request cap 24, and five evidence-cost lambda values.

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

The main limitation is still sample size, but the 24-case run is much more conclusive than the 10-case pilot. It shows a real tradeoff curve and identifies the cutoff region: `lambda = 0.10` is strongest, `0.22-0.35` are efficient high-performance settings, and `0.50+` is too aggressive. The partial-evidence matched result also suggests a hybrid system may be stronger than either the LLM or direct classifier alone.

## Recommended Next Step

Run a larger but still cost-controlled live validation with:

- fixed model: `gpt-4.1-mini`
- `temperature=0.0`
- `top_p=1.0`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = None`
- lambdas: `[0.10, 0.22, 0.35]`
- request cap: `24`

This should test whether the 24-case cutoff trend survives on the full 49-case balanced sample without wasting budget on lambda values that are now known to be too aggressive.
