# Matched-Evidence And Integrated Comparison

## Purpose

Notebook `09_matched_evidence_integrated_comparison.ipynb` evaluates whether the sequential system helps because it reasons better, because it acquires useful evidence, or both.

The core question is:

> Given exactly the same evidence acquired by the sequential policy, does the sequential LLM final answer outperform a direct one-shot classifier?

This matters because a sequential system can add value in two different ways:

- choosing useful questions
- making the final diagnosis from the acquired evidence

Those should be evaluated separately.

## Compared Systems

For each sequential condition, the notebook compares:

- initial-evidence one-shot
- sequential final prediction
- matched-evidence one-shot
- full-evidence one-shot

The matched-evidence one-shot input contains:

- demographics
- initial evidence
- exactly the evidence roots revealed by the sequential policy

It does not contain:

- turn order
- unrevealed evidence
- hidden pathology labels
- hidden DDXPlus differential
- full-evidence predictions

## How Matched Evidence Is Built

The notebook reads the sequential `traces.jsonl` file.

For each case, it reconstructs:

- initial evidence root
- every requested evidence root
- whether each requested root was present or absent
- categorical or multi-choice revealed values when present

This produces a bag-of-evidence state for direct classification.

The current implementation now prefers a partial-evidence direct model from notebook `10` when that artifact exists. If notebook `10` has not been run, it falls back to the older full-evidence direct model on partial states.

The partial-evidence comparator is the stronger matched baseline because it is trained to handle incomplete evidence states rather than being trained only on full-evidence inputs.

## Artifact Layout

Notebook path:

- `notebooks/09_matched_evidence_integrated_comparison.ipynb`

Artifact root:

- `artifacts/integrated_comparisons/`

Dry-run validation artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_dryrun_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`

Expected files:

- `paired_case_results.csv`
- `integrated_summary.csv`
- `resolved_comparison_config.json`
- figures under `figures/`

When the partial-evidence comparator is available, `paired_case_results.csv` includes:

- `matched_model_source`

Expected values:

- `partial_evidence_one_shot`
- `full_evidence_model_fallback`

## Plots

The notebook generates:

- integrated accuracy comparison
- ranking quality comparison
- accuracy vs mean requests
- fraction of full-evidence gain recovered
- remaining gap to full-evidence comparator

The last two plots become available after notebook 07 has produced a full-evidence model artifact.

## Current Validation Status

The notebook executed successfully against the dry-run cost-sensitive artifact.

The original dry-run validation occurred before notebook 07 had produced a full-evidence checkpoint, so those validation-only matched/full-evidence columns were expectedly unavailable. The final live comparison below supersedes that dry-run-only status.

## Final Integrated Live Results

Notebook 09 has now been rerun after the full-evidence one-shot model and live cost-sensitive sequential run were available.

Integrated artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`

The comparison uses the same 10 live pilot cases for all systems.

| Lambda | Initial one-shot acc | Sequential acc | Matched-evidence one-shot acc | Full-evidence acc | Mean requests | Sequential full-gain recovered | Matched full-gain recovered |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.300 | 0.900 | 0.600 | 1.000 | 18.4 | 0.857 | 0.429 |
| 0.03 | 0.300 | 0.900 | 0.600 | 1.000 | 16.6 | 0.857 | 0.429 |
| 0.06 | 0.300 | 0.900 | 0.600 | 1.000 | 15.7 | 0.857 | 0.429 |
| 0.10 | 0.300 | 0.900 | 0.700 | 1.000 | 14.1 | 0.857 | 0.571 |
| 0.15 | 0.300 | 0.900 | 0.700 | 1.000 | 12.8 | 0.857 | 0.571 |
| 0.22 | 0.300 | 0.900 | 0.700 | 1.000 | 11.8 | 0.857 | 0.571 |

Interpretation:

- sequential improves strongly over the initial-evidence one-shot baseline on this slice
- sequential recovers about 86% of the gap between initial-evidence and full-evidence diagnosis
- matched-evidence one-shot recovers about 43% to 57% of that gap
- sequential beats matched-evidence one-shot by 20 to 30 accuracy points on the same acquired evidence
- full evidence still has a remaining 10-point advantage over sequential on this 10-case pilot

The matched-evidence result is important, but it should be interpreted carefully. The current matched comparator reuses the full-evidence one-shot model on partial-evidence states. That is a useful first check, but a better future matched comparator would train directly on partial-evidence inputs so it is not distribution-shifted.

The one persistent sequential miss across lambda settings is:

- `test:81691`, true pathology `Croup`

That case should be the first target for qualitative trace inspection.

## Wider 24-Case Integrated Comparison

Notebook 09 was rerun against the wider 24-case lambda sweep.

Integrated artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_24case_wide_sweep_v1/`

| Lambda | Initial one-shot acc | Sequential acc | Matched-evidence one-shot acc | Full-evidence acc | Mean requests | Sequential full-gain recovered | Matched full-gain recovered |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.917 | 0.625 | 1.000 | 13.0 | 0.875 | 0.438 |
| 0.22 | 0.333 | 0.875 | 0.708 | 1.000 | 10.7 | 0.813 | 0.563 |
| 0.35 | 0.333 | 0.875 | 0.667 | 1.000 | 8.3 | 0.813 | 0.500 |
| 0.50 | 0.333 | 0.417 | 0.333 | 1.000 | 2.2 | 0.125 | 0.000 |
| 0.75 | 0.333 | 0.375 | 0.250 | 1.000 | 1.0 | 0.063 | -0.125 |

Win/loss against matched-evidence one-shot:

| Lambda | Both correct | Sequential only correct | Matched only correct | Both wrong |
|---:|---:|---:|---:|---:|
| 0.10 | 15 | 7 | 0 | 2 |
| 0.22 | 17 | 4 | 0 | 3 |
| 0.35 | 16 | 5 | 0 | 3 |
| 0.50 | 6 | 4 | 2 | 12 |
| 0.75 | 5 | 4 | 1 | 14 |

Scientific interpretation:

- the sequential policy beats the matched-evidence one-shot comparator at every lambda
- at useful lambdas, there are no cases where matched-evidence one-shot is correct while sequential is wrong
- the sequential system is therefore adding value beyond just choosing evidence, at least against the current matched comparator
- the matched comparator still remains imperfect because it uses a full-evidence-trained model on partial-evidence states
- the full-evidence comparator remains at `1.000` on this 24-case slice, so there is still a measurable gap to full-information diagnosis

The 24-case run also shows that the sequential policy has a real evidence-efficiency frontier:

- `lambda = 0.10`: best accuracy, more evidence
- `lambda = 0.22` or `0.35`: slightly lower accuracy, meaningfully fewer requests
- `lambda >= 0.50`: too little evidence, large performance collapse

## Partial-Evidence Matched Comparator Results

Notebook 09 was rerun again after notebook 10 trained the partial-evidence matched classifier.

Integrated artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_partial_policy_v1/`

Matched model source:

- `partial_evidence_one_shot`

| Lambda | Initial acc | Sequential acc | Partial matched acc | Partial matched top-3 | Partial matched top-5 | Full-evidence acc | Mean requests |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.917 | 0.875 | 1.000 | 1.000 | 1.000 | 13.0 |
| 0.22 | 0.333 | 0.875 | 0.875 | 1.000 | 1.000 | 1.000 | 10.7 |
| 0.35 | 0.333 | 0.875 | 0.833 | 0.958 | 0.958 | 1.000 | 8.3 |
| 0.50 | 0.333 | 0.417 | 0.458 | 0.708 | 0.792 | 1.000 | 2.2 |
| 0.75 | 0.333 | 0.375 | 0.375 | 0.583 | 0.708 | 1.000 | 1.0 |

Win/loss against sequential:

| Lambda | Both correct | Sequential only correct | Matched only correct | Both wrong |
|---:|---:|---:|---:|---:|
| 0.10 | 21 | 1 | 0 | 2 |
| 0.22 | 20 | 1 | 1 | 2 |
| 0.35 | 20 | 1 | 0 | 3 |
| 0.50 | 9 | 1 | 2 | 12 |
| 0.75 | 8 | 1 | 1 | 14 |

This changes the interpretation.

The old fallback matched comparator made the sequential LLM look clearly better. The stronger partial-evidence comparator nearly catches it:

- at `lambda = 0.10`, sequential is ahead by one case
- at `lambda = 0.22`, they tie on top-1 accuracy
- at `lambda = 0.35`, sequential is ahead by one case
- at very high lambdas, the direct matched comparator can be slightly better, but both systems are poor because too little evidence is acquired

The strongest conclusion is therefore not that LLM final reasoning dominates direct classification. The stronger conclusion is that targeted evidence acquisition is valuable, and that final diagnosis may be best handled by either:

- the sequential LLM,
- the partial-evidence classifier,
- or a hybrid adjudication step when they disagree.

## Scientific Interpretation

This notebook is important because it can support a more nuanced conclusion.

If sequential beats matched-evidence one-shot:

- the LLM reasoning over the acquired evidence is adding value

If matched-evidence one-shot beats sequential:

- the sequential policy may still be useful as an evidence acquisition controller
- the final diagnosis should perhaps be delegated to a direct classifier

If both remain far below full-evidence one-shot:

- the policy is not acquiring enough of the right evidence

If both approach full-evidence one-shot with far fewer fields:

- the project has strong evidence that targeted sequential workup is efficient and useful
