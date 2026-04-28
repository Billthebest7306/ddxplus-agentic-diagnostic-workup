# Rigorous Evaluation Phase Summary

## Current Research Question

The project is now framed around this question:

> Can a sequential diagnostic policy approach full-evidence diagnostic performance while acquiring only a limited, targeted subset of evidence?

A second question is:

> Given the same acquired evidence, does the sequential LLM reasoning outperform a direct one-shot classifier, or is its main value evidence acquisition?

## What Was Added

Three successor notebooks were added:

- `notebooks/07_full_evidence_one_shot_comparator.ipynb`
- `notebooks/08_cost_sensitive_sequential_lambda_sweep.ipynb`
- `notebooks/09_matched_evidence_integrated_comparison.ipynb`

Small deterministic fixes were also applied to notebooks `05` and `06`:

- default LLM fixed to `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- deterministic settings logged in new run configs
- safe API bootstrap no longer prompts during non-live execution

## Why This Matters

Before this phase, the refined sequential system looked promising on a small slice but was still easy to criticize:

- it consumed most of the request budget
- budget sweeps were arbitrary
- there was no matched-information comparator
- full-information ceiling was missing
- model/backbone settings were not consistently deterministic

This phase addresses those criticisms without moving into the deeper algorithmic ledger work.

## Completed Empirical Runs

The first empirical pass for this phase is now complete.

Completed runs:

1. Notebook 07 trained the full-evidence one-shot comparator on the full official train split.
2. Notebook 07 recomputed validation/test metrics after removing validation/test rows duplicated in the training split.
3. Notebook 08 ran live with `gpt-4.1-mini` on a 10-case balanced pilot.
4. Notebook 09 reran the integrated comparison after the full-evidence and live sequential artifacts existed.

## Current Validation

Notebook 08 was executed in dry-run mode and produced artifacts under:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_dryrun_test_1perclass_cap24_6lambdas_lambda_cost_v1/`

Notebook 09 was executed against that dry-run artifact and produced:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_dryrun_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`

These dry-run artifacts verify the pipeline and artifact contracts. They are not live scientific results.

Live result artifacts:

- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/`
- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1/`
- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`

## Main Result So Far

The 10-case live pilot supported the phase goal:

- initial-evidence one-shot on the same 10 cases: `0.300` accuracy
- cost-sensitive sequential: `0.900` accuracy across all lambda values
- matched-evidence one-shot: `0.600` to `0.700` accuracy
- full-evidence one-shot: `1.000` accuracy

The lambda sweep reduced mean requests from `18.4` at lambda `0.00` to `11.8` at lambda `0.22` without reducing top-1 accuracy on this pilot. This suggests the cost-sensitive controller is improving evidence efficiency.

The full-evidence one-shot model reached `0.9958` official test accuracy and remained at `0.9958` after train-overlap dedup robustness filtering, so the full-evidence ceiling is not explained by duplicate leakage.

The later 24-case wide lambda sweep is now the stronger sequential result:

- initial-evidence one-shot on the same 24 cases: `0.333` accuracy
- sequential at lambda `0.10`: `0.917` accuracy with `13.0` mean requests
- sequential at lambda `0.22`: `0.875` accuracy with `10.7` mean requests
- sequential at lambda `0.35`: `0.875` accuracy with `8.3` mean requests
- sequential at lambda `0.50`: `0.417` accuracy with `2.2` mean requests
- sequential at lambda `0.75`: `0.375` accuracy with `1.0` mean requests

This identifies the approximate cutoff region. Lambda values up to `0.35` preserve strong performance; lambda values of `0.50` or higher stop too early.

Notebook 10 added a stronger partial-evidence matched comparator, and notebook 9 was rerun with it:

- partial-evidence matched at lambda `0.10`: `0.875` accuracy, `1.000` top-3/top-5
- partial-evidence matched at lambda `0.22`: `0.875` accuracy, `1.000` top-3/top-5
- partial-evidence matched at lambda `0.35`: `0.833` accuracy, `0.958` top-3/top-5

This changes the interpretation. Sequential evidence acquisition is clearly valuable, but the final LLM answer is not clearly dominant over a direct classifier trained for partial evidence. A hybrid system is now a serious candidate.

## Recommended Next Live Run

The broad 10-case lambda sweep and the wider 24-case cutoff sweep have already been run. The next run should be narrower and larger:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
TOP_P = 1.0
EVIDENCE_COST_LAMBDAS = [0.10, 0.22, 0.35]
MAX_REQUEST_CAP = 24
SEQUENTIAL_SAMPLE_PER_CLASS = 1
SEQUENTIAL_MAX_CASES = None
RUN_VERSION = "lambda_cost_49case_cutoff_v1"
```

This runs the 49-case balanced pilot while avoiding unnecessary API spend on lambda values that are now known to be too aggressive.

After that run, rerun notebook 9 with the partial-evidence matched comparator and report both:

- sequential final diagnosis
- partial-evidence classifier final diagnosis

If possible, also add a simple hybrid analysis that marks cases where the two systems disagree.

## Interpretation Standard

The project should not claim success only from higher raw accuracy.

The stronger claim requires:

- sequential accuracy improves over initial-evidence one-shot
- mean requests remain substantially below full evidence
- lambda controls request usage in the expected direction
- matched-evidence comparison clarifies whether the final diagnosis should be made by the LLM or by a direct classifier
- full-evidence comparison shows how much of the available DDXPlus signal is being recovered

That is the standard for a scientifically cleaner next result.
