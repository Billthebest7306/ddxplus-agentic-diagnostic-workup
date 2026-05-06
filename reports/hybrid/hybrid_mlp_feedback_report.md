# Hybrid V1 MLP Feedback Report

Last updated: 2026-05-05

## Purpose

Hybrid v1 tests whether the partial-evidence MLP can improve the current single-agent sequential workup by acting as an online diagnostic belief monitor.

The system remains single-agent:

- the LLM chooses evidence requests
- the deterministic ledger controls legal evidence access
- the MLP sees only the current visible ledger state
- the MLP provides confidence, margin, entropy, stability, and ranked differential feedback
- the final trace saves `llm_final`, `mlp_final`, and `hybrid_final`
- no multi-agent system, graph ledger, RL policy, or hidden/full-evidence oracle is used

## Artifacts

Notebook:

- `notebooks/11_online_hybrid_mlp_feedback.ipynb`

Integrated comparison notebook:

- `notebooks/09_matched_evidence_integrated_comparison.ipynb`

Live hybrid run:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1/`

Integrated comparison:

- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/`

Important files:

- `lambda_sweep_summary.csv`
- per-lambda `metrics.json`
- per-lambda `predictions.csv`
- per-lambda `traces.jsonl`
- `integrated_summary.csv`
- `paired_case_results.csv`
- figures under each artifact folder

## Run Configuration

The live run used:

- model: `gpt-4.1-mini`
- temperature: `0.0`
- top_p: `1.0`
- split: `test`
- sample: 24 balanced cases
- lambdas: `[0.10, 0.22, 0.35]`
- max request cap: `24`
- one-shot/partial MLP source: `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

The loaded partial-evidence MLP had standalone policy-mask test performance:

| Model | Test rows | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Partial-evidence MLP, policy masks | 39,998 | 0.515 | 0.741 | 0.827 | 0.519 |

## Hybrid Results

| Lambda | Hybrid acc | Hybrid top-3 | Hybrid top-5 | Macro-F1 | Mean requests | Stop before cap | Cap hits | LLM/MLP agreement | Input tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.833 | 1.000 | 1.000 | 0.756 | 9.67 | 0.875 | 3 | 0.917 | 585,925 |
| 0.22 | 0.875 | 0.958 | 0.958 | 0.813 | 7.46 | 0.917 | 2 | 1.000 | 452,676 |
| 0.35 | 0.833 | 0.917 | 0.917 | 0.744 | 5.88 | 0.958 | 1 | 1.000 | 359,948 |

The strongest hybrid setting is `lambda = 0.22`: `21/24` correct with about `7.46` requests per case. This is the best evidence-efficiency point in this run.

## Comparison Against Notebook 08

Notebook 08 is the previous cost-sensitive LLM-only sequential baseline on the same 24-case sample.

| Lambda | Notebook 08 acc | Notebook 08 requests | Hybrid acc | Hybrid requests | Accuracy change | Request change |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 13.04 | 0.833 | 9.67 | -0.083 | -3.38 |
| 0.22 | 0.875 | 10.67 | 0.875 | 7.46 | 0.000 | -3.21 |
| 0.35 | 0.875 | 8.33 | 0.833 | 5.88 | -0.042 | -2.46 |

Interpretation:

- at `lambda = 0.22`, hybrid v1 preserves the prior sequential accuracy while reducing mean requests by about 30%
- at `lambda = 0.10`, hybrid v1 asks fewer questions but loses two more cases than notebook 08
- at `lambda = 0.35`, hybrid v1 asks fewer questions but loses one more case than notebook 08
- hybrid v1 improves evidence efficiency, but not raw accuracy

Input-token usage also dropped relative to notebook 08:

| Lambda | Notebook 08 input tokens | Hybrid input tokens | Token reduction |
|---:|---:|---:|---:|
| 0.10 | 710,832 | 585,925 | 17.6% |
| 0.22 | 585,943 | 452,676 | 22.7% |
| 0.35 | 456,292 | 359,948 | 21.1% |

## Evidence-Budget View

Notebook `09` now reports performance against actual evidence used, not only against lambda.

| Lambda | Mean requests | Mean visible roots incl. initial | Hybrid acc | Online MLP acc | Offline matched MLP acc | Hybrid top-5 | Offline matched top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 9.67 | 10.67 | 0.833 | 0.833 | 0.833 | 1.000 | 1.000 |
| 0.22 | 7.46 | 8.46 | 0.875 | 0.875 | 0.875 | 0.958 | 0.958 |
| 0.35 | 5.88 | 6.88 | 0.833 | 0.833 | 0.833 | 0.917 | 0.917 |

The evidence-budget view is the cleanest way to read the result. At `lambda = 0.22`, hybrid v1 reaches the same top-1 accuracy as the earlier matched-MLP setup but with fewer requested fields. At `lambda = 0.35`, it also preserves matched-MLP accuracy while using fewer fields. At `lambda = 0.10`, it loses one case because it stops earlier than the previous notebook `08` policy.

New detailed report:

- `reports/baselines/integrated_evidence_budget_comparison_report.md`

## Integrated Comparator Results

Notebook 09 compared the hybrid run against initial one-shot, full-evidence ceiling, and matched-evidence MLP on the same `case_id`s.

| Lambda | Initial one-shot | Hybrid final | LLM final | Online MLP final | Matched MLP | Full evidence | Mean requests |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.833 | 0.875 | 0.833 | 0.833 | 1.000 | 9.67 |
| 0.22 | 0.333 | 0.875 | 0.875 | 0.875 | 0.875 | 1.000 | 7.46 |
| 0.35 | 0.333 | 0.833 | 0.833 | 0.833 | 0.833 | 1.000 | 5.88 |

The integrated result is important: the hybrid final, online MLP final, and offline matched MLP are identical on top-1 accuracy at all three lambdas. That means hybrid v1 did not yet prove that adjudication is better than either individual head. Its main contribution is more efficient stopping/evidence use.

At `lambda = 0.10`, the LLM final alone scored `0.875`, while the hybrid final scored `0.833`. The hybrid rule lost one case by trusting a high-confidence wrong MLP over a correct LLM.

## Error Pattern

Hybrid final errors:

| Lambda | Errors |
|---:|---|
| 0.10 | `Bronchiolitis`, `Croup`, `Influenza`, `Pericarditis` |
| 0.22 | `Croup`, `Influenza`, `Pericarditis` |
| 0.35 | `Chagas`, `Croup`, `Influenza`, `Pericarditis` |

Persistent hard cases:

- `test:81691`, true `Croup`
- `test:8666`, true `Influenza`
- `test:62878`, true `Pericarditis`

Notable failure:

- at `lambda = 0.10`, `test:90978` true `Bronchiolitis` was correctly predicted by the LLM but incorrectly changed to `Croup` by the high-confidence MLP override

This suggests that MLP confidence is not calibrated enough to safely override the LLM in disagreements.

## Scientific Interpretation

Hybrid v1 is useful, but not in the way originally hoped.

Supported claim:

- online MLP feedback can reduce evidence requests while preserving accuracy at a useful lambda
- the best point is `lambda = 0.22`, where hybrid matches notebook 08 accuracy with about 30% fewer requests

Unsupported claim:

- hybrid adjudication does not yet beat LLM-final or matched-MLP-final diagnosis
- high-confidence MLP override is not safe enough in disagreements

Current best framing:

> Hybrid v1 improves evidence efficiency more than final diagnostic accuracy. The MLP is useful as a stopping and stability signal, but final-head adjudication needs to be more conservative.

## Recommended Next Step

Do not run a larger hybrid benchmark yet without a small policy patch.

Recommended targeted change:

- keep MLP feedback in the prompt and stop policy
- remove or weaken the high-confidence MLP override in final adjudication
- in disagreements, prefer the LLM final unless there is additional evidence that the LLM answer is unstable or outside the MLP top-k
- rerun the same 24-case slice at `lambda = 0.22` only to check whether the request-efficiency gain remains

If that patch holds:

- run the 49-case balanced pilot with `lambda = 0.22`
- optionally include `lambda = 0.10` as a high-accuracy comparison point

## Bottom Line

Hybrid v1 is not a failure. It is a narrower result:

- it improves request efficiency
- it does not improve top-1 diagnosis beyond the existing heads
- it reveals that final-head trust/adjudication is the next bottleneck
- it strengthens the project’s scientific story because we can now separate evidence acquisition, stopping, and final diagnosis quality

## Subsequent Notebook 13 Confirmation

Notebook `13` has now tested the selected MLP-guided stop rule live, without running another lambda sweep.

| System | Accuracy | Mean requests |
|---|---:|---:|
| Notebook `08`, lambda `0.10` | 0.917 | 13.04 |
| Notebook `11`, lambda `0.22` | 0.875 | 7.46 |
| Notebook `12`, offline selected stop | 0.917 | 6.875 |
| Notebook `13`, live selected stop | 0.917 | 6.58 |

This confirms the best interpretation of hybrid v1: the MLP is most useful as a stopping signal. The selected stop rule preserved notebook `08` accuracy while cutting requests by about half.

The two live failures were `Croup` and `Pericarditis`; both were wrong for all final heads. That shifts the next bottleneck from final-head adjudication toward evidence trajectory and hard-case discrimination.
