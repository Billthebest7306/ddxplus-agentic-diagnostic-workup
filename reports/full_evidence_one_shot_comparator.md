# Full-Evidence One-Shot Comparator

## Purpose

Notebook `07_full_evidence_one_shot_comparator.ipynb` adds an oracle-style direct diagnosis comparator for DDXPlus.

The existing one-shot baseline sees only:

- age
- sex
- `INITIAL_EVIDENCE`

The full-evidence comparator sees:

- age
- sex
- every DDXPlus root evidence field as known present/value or absent

This answers a different scientific question from the initial-evidence baseline:

> If the full structured case is visible, how much diagnostic signal does DDXPlus contain?

It is a ceiling-style comparator, not a live workup policy.

## Fairness Rule

Full-evidence predictions, probabilities, labels, and derived features must not be used inside the live sequential policy.

The full-evidence model is allowed only for:

- post-hoc evaluation
- estimating the full-information ceiling
- measuring the remaining gap after sequential evidence acquisition

It is not allowed for:

- action shortlisting
- stop decisions
- live diagnosis updates
- prompting the LLM

## Implementation

Notebook path:

- `notebooks/07_full_evidence_one_shot_comparator.ipynb`

Artifact root:

- `artifacts/one_shot_full_evidence/`

The notebook reuses the same model family as notebook 01:

- BASD-style DDXPlus slot encoding
- age and sex features
- MLP with hidden sizes `[2048, 2048, 2048]`
- ReLU activations
- pathology cross-entropy objective

The key difference is visibility:

- initial-evidence baseline marks unrevealed evidence as unknown
- full-evidence comparator marks every root evidence field as observed
- roots present in `EVIDENCES` are encoded as present/value
- roots absent from `EVIDENCES` are encoded as absent

## Efficiency Fix

The full-evidence encoder was implemented with an all-absent evidence template.

Instead of applying 223 absent observations for every patient row, the notebook:

- precomputes one full-evidence absent template
- copies it per patient
- sets demographics
- applies only the positive/value roots from that patient

This keeps the full-evidence comparator practical on the full DDXPlus split.

## Expected Outputs

After running notebook 07, the expected files are:

- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/metrics.json`
- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/predictions.csv`
- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/best_model.pt`
- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/training_history.json`
- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/confusion_summary.csv`
- `artifacts/one_shot_full_evidence/selected_model.json`

The deduplication robustness section additionally writes:

- `dedup_robustness_summary.csv`
- `dedup_robustness_summary.json`
- `dedup_metrics_validate_raw_row_signature.json`
- `dedup_metrics_validate_feature_signature.json`
- `dedup_metrics_test_raw_row_signature.json`
- `dedup_metrics_test_feature_signature.json`
- matching `dedup_predictions_*` CSV files when rows remain after filtering

These artifacts live inside the selected full-evidence run directory.

## Deduplicated Robustness Check

The audit found no obvious code-level target leakage in the full-evidence feature construction, but the official DDXPlus train/validate/test files contain exact duplicate rows across splits.

Notebook 07 therefore keeps the official split metrics unchanged for comparability, then adds a non-destructive robustness check:

- `raw_row_signature`: removes validation/test rows whose full raw patient row appears in training
- `feature_signature`: removes validation/test rows whose `AGE`, `SEX`, `EVIDENCES`, and `INITIAL_EVIDENCE` signature appears in training

The second check is stricter for feature leakage because it removes rows that present the same model-visible patient state even if target or differential fields differ.

This robustness check does not retrain the model. It reloads the selected trained checkpoint and recomputes validation/test metrics on filtered validation/test subsets only.

Interpretation rule:

- official metrics remain the headline numbers for comparability with the released dataset
- deduplicated metrics are reported as robustness checks against cross-split duplicate contamination
- if deduplicated metrics remain high, the full-evidence result is better explained by highly diagnostic evidence rather than duplicate leakage
- if deduplicated metrics drop materially, the writeup should quantify that drop and frame the official metric as potentially inflated

## Scientific Meaning

This comparator is important because it separates two limitations:

- the dataset may not contain enough evidence
- the sequential policy may not be acquiring or using the evidence well

If the full-evidence one-shot model performs much higher than the initial-evidence one-shot model, then DDXPlus has strong diagnostic signal. In that case, sequential failures should be interpreted mainly as policy, stopping, or reasoning failures rather than dataset ambiguity.

## Final Results

The full-evidence notebook has now been run in `full` mode.

Run artifact:

- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/`

Official full-split metrics:

| Split/result | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Best validation checkpoint | 0.9954 | not primary | not primary | 0.9943 |
| Test | 0.9958 | 1.0000 | 1.0000 | 0.9948 |

The final model trained on:

- train rows: `1,025,602`
- validation rows: `132,448`
- test rows: `134,529`
- device: `mps`
- runtime: about `1,557` seconds

Most remaining errors are clinically adjacent confusions. The largest test confusion is `Acute rhinosinusitis` predicted as `Chronic rhinosinusitis`, followed by smaller confusions such as `Acute laryngitis` vs `Viral pharyngitis` and `Unstable angina` vs `Stable angina`.

## Deduplication Robustness Results

The deduplication robustness check was run after training.

| Split | Dedup type | Rows removed | Duplicate fraction | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Validate | Raw row signature | 1,643 | 0.0124 | 0.9953 | 1.0000 | 1.0000 | 0.9943 |
| Validate | Feature signature | 1,808 | 0.0137 | 0.9953 | 1.0000 | 1.0000 | 0.9943 |
| Test | Raw row signature | 1,823 | 0.0136 | 0.9958 | 1.0000 | 1.0000 | 0.9948 |
| Test | Feature signature | 1,989 | 0.0148 | 0.9958 | 1.0000 | 1.0000 | 0.9948 |

This confirms that exact or feature-level duplicate overlap with the training split exists, but it does not materially explain the full-evidence result. Official metrics remain the headline numbers for comparability; deduplicated metrics are the robustness check.

The full-evidence model should still be interpreted as a ceiling-style comparator, not as a live clinical workup system and not as a source of information for the sequential policy.
