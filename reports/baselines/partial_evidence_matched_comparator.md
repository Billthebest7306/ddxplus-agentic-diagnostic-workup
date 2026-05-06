# Partial-Evidence Matched Comparator

## Purpose

Notebook `10_partial_evidence_one_shot_comparator.ipynb` adds a stronger direct matched-information comparator.

The earlier notebook `09` matched comparator used the full-evidence one-shot model on partial evidence states. That was useful, but imperfect because the model was trained with all evidence visible and then tested with only the evidence collected by the sequential policy.

The new comparator trains a direct classifier specifically for partial-evidence diagnosis.

## What It Uses

The sequential traces already record the evidence fields requested by the single-agent policy.

Each trace step contains:

- `policy_response.requested_evidence_id`
- `reveal_payload.root_evidence_id`
- `reveal_payload.status`
- `reveal_payload.revealed_values`
- `reveal_payload.summary`

Notebook `10` uses those traces to estimate the policy-shaped evidence mask distribution:

- how many evidence roots are usually requested
- which root evidence fields are requested frequently
- what partial-evidence states a matched one-shot comparator should be robust to

It does not use sequential test labels as training data.

## Training Design

The model is trained on official DDXPlus train/validation rows.

For each training row:

- demographics are visible
- `INITIAL_EVIDENCE` is visible
- additional evidence roots are sampled from the sequential policy's observed request distribution
- requested evidence roots are encoded as present/value if they exist in that patient row
- requested evidence roots are encoded as absent if they do not exist in that patient row
- all unrequested evidence remains unknown

The architecture stays aligned with the one-shot baseline family:

- BASD-style slot encoding
- MLP with hidden sizes `[2048, 2048, 2048]`
- pathology cross-entropy objective

## Why This Is Fairer

This is not a new model per test case.

It is one partial-evidence model trained to handle incomplete evidence states. At evaluation time, notebook `09` feeds it exactly the evidence acquired by the sequential policy for each case.

This is fairer than the previous fallback because the model is trained on partial evidence instead of being trained only for full evidence.

## Fairness Boundaries

Allowed:

- using sequential trace root IDs to define the kind of evidence masks the comparator should handle
- training on official train/validation labels
- evaluating on exact sequentially acquired evidence fields

Not allowed:

- training on test labels from the sequential benchmark
- training a separate model on the exact test cases being compared
- using hidden full evidence in the matched input
- using turn order or LLM reasoning traces as model input

## Artifact Layout

Notebook path:

- `notebooks/10_partial_evidence_one_shot_comparator.ipynb`

Artifact root:

- `artifacts/one_shot_partial_evidence/`

Current selected run artifact:

- `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

Expected files:

- `best_model.pt`
- `metrics.json`
- `resolved_run_config.json`
- `training_history.json`
- figures under `figures/`

The notebook also writes:

- `artifacts/one_shot_partial_evidence/selected_model.json`

Notebook `09` now checks for this selected partial-evidence model. If it exists, `matched_*` predictions use it. If not, notebook `09` falls back to the older full-evidence model on partial states.

The updated notebook `09` writes the new comparison under:

- `artifacts/integrated_comparisons/<sequential_run_name>__matched_integrated_partial_policy_v1/`

## Current Status

Notebook `10` has been rebuilt from the ground up after repeated loader/path issues.

The current notebook now has:

- parent-walking project-root discovery
- robust DDXPlus zip loading for release files whose internal members do not end in `.csv`
- robust parsing for `INITIAL_EVIDENCE` when stored as a bare token
- direct discovery of `lambda_*/traces.jsonl` files
- explicit dataset and trace preflight summaries
- `smoke`, `quick`, `final`, and `full` run modes
- a notebook-09-compatible checkpoint format

Validation performed:

- all notebook code cells parse cleanly
- a direct smoke harness validated local split loading, trace-mask extraction, partial-state encoding, and a tiny train/evaluate pass

Notebook `09` has been updated to prefer the partial-evidence matched model when available.

## Results

Notebook `10` has now been run in `final` mode.

Selected model:

- `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

Training setup:

- train rows: `300,000`
- validation rows: `40,000`
- test rows: `39,998`
- feature size: `922`
- classes: `49`
- architecture: `[2048, 2048, 2048]` MLP with dropout `0.10`
- device: `mps`
- best epoch: `5`
- runtime: about `198` seconds

Standalone partial-mask performance:

| Split | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Validation | 0.513 | 0.739 | 0.827 | 0.507 |
| Test | 0.515 | 0.741 | 0.827 | 0.519 |

This is much stronger than the initial-evidence one-shot baseline but far below the full-evidence comparator, which is expected. The model is solving a harder partial-information problem.

## Integrated Comparison Result

Notebook `09` was rerun using this partial-evidence comparator.

Integrated artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_partial_policy_v1/`

On the 24-case sequential slice:

| Lambda | Sequential acc | Partial matched acc | Partial matched top-3 | Partial matched top-5 | Mean requests |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.875 | 1.000 | 1.000 | 13.0 |
| 0.22 | 0.875 | 0.875 | 1.000 | 1.000 | 10.7 |
| 0.35 | 0.875 | 0.833 | 0.958 | 0.958 | 8.3 |
| 0.50 | 0.417 | 0.458 | 0.708 | 0.792 | 2.2 |
| 0.75 | 0.375 | 0.375 | 0.583 | 0.708 | 1.0 |

Win/loss against the sequential final answer:

| Lambda | Both correct | Sequential only correct | Matched only correct | Both wrong |
|---:|---:|---:|---:|---:|
| 0.10 | 21 | 1 | 0 | 2 |
| 0.22 | 20 | 1 | 1 | 2 |
| 0.35 | 20 | 1 | 0 | 3 |
| 0.50 | 9 | 1 | 2 | 12 |
| 0.75 | 8 | 1 | 1 | 14 |

Interpretation:

- the stronger matched comparator closes most of the gap to the sequential LLM
- sequential is still best at `lambda = 0.10` and `lambda = 0.35`, but only by one case
- at `lambda = 0.22`, sequential and matched one-shot tie on top-1 accuracy
- the matched comparator has very strong ranking quality at useful lambdas, with top-3/top-5 reaching `1.000` at lambdas `0.10` and `0.22`
- this weakens the claim that the LLM final reasoning is clearly superior to direct classification
- it strengthens the claim that the sequential system's main value is targeted evidence acquisition

## Next Step

The most promising next architecture is hybrid:

- use the sequential policy to acquire evidence efficiently
- use the partial-evidence direct classifier as a final diagnostic head
- optionally compare or reconcile the classifier and LLM final answers when they disagree

The current sample is still only 24 cases, so this should be validated on the 49-case balanced slice before making a final claim.
