# Stopping-Policy Ablation Report

## Purpose

Notebook `12_stopping_policy_ablation.ipynb` tests whether hybrid v1's evidence-efficiency gain is actually coming from the partial-evidence MLP providing a better stopping signal, or whether a more aggressive LLM-only stopping rule could achieve the same accuracy/request tradeoff.

This is an offline replay ablation. It does not make new API calls. It replays already-recorded notebook `08` trajectories and asks: if we had stopped at different turns along the same evidence path, which stopping rule would have preserved diagnostic accuracy best?

Primary question:

> At the same mean number of requested evidence fields, does MLP-guided stopping preserve accuracy better than LLM-only stopping?

## Inputs

Main replay source:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/lambda_0p100/`

Reference hybrid source:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1/`

Partial-evidence MLP source:

- `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

Output artifact root:

- `artifacts/stopping_policy_ablation/stopping_policy_ablation_24case_v1/`

## Validation

The notebook generated:

- `333` turn-level replay rows
- `24` aligned cases
- `309` stopping policy specifications
- `7,416` case-policy rows
- `1,236` policy-summary rows

The partial-evidence MLP reconstruction check against notebook `11` passed:

| Check | Result |
|---|---:|
| Cases checked | 24 |
| Matched reconstructed MLP predictions | 24 |
| Match rate | 1.000 |

This means notebook `12` is reconstructing the online MLP state consistently with the hybrid notebook.

## Observed Live References

These are not simulated policies. They are the original live runs used as references.

| System | Lambda | Final head | Correct | Accuracy | Top-5 | Macro-F1 | Mean requests |
|---|---:|---|---:|---:|---:|---:|---:|
| Notebook 08 LLM-only | 0.10 | LLM | 22/24 | 0.917 | 0.917 | 0.846 | 13.04 |
| Notebook 08 LLM-only | 0.22 | LLM | 21/24 | 0.875 | 0.917 | 0.795 | 10.67 |
| Notebook 08 LLM-only | 0.35 | LLM | 21/24 | 0.875 | 0.875 | 0.813 | 8.33 |
| Notebook 11 hybrid v1 | 0.22 | Hybrid | 21/24 | 0.875 | 0.958 | 0.813 | 7.46 |

The observed hybrid result at `lambda = 0.22` already suggested that MLP feedback improved efficiency: same accuracy as notebook `08` at `lambda = 0.22`, but with fewer requests.

## Matched-Budget Ablation Result

The ablation target was approximately `7.5` mean requested evidence fields. The selected policy rule was:

- first, maximize accuracy at or below the target request budget
- then prefer fewer requests
- then prefer higher top-5, fewer forced end-of-trace stops, and simpler rules

Matched-budget comparison:

| Comparison | Policy family | Final head | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Best pure LLM-only stop | deterministic-state | LLM | 20/24 | 0.833 | 0.917 | 0.958 | 0.767 | 6.33 |
| Best MLP-guided stop | MLP confidence | LLM | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |
| Best MLP-guided stop | MLP confidence | MLP | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.25 |
| Best MLP-guided stop | MLP confidence | conservative hybrid | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |
| Best MLP-guided stop | MLP confidence | agreement hybrid | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |

Selected policy:

- policy: `mlp_conf_ge_0.70_margin_ge_0.20_entropy_le_0.10_stab_0`
- final head: `agreement_hybrid_final`
- correct: `22/24`
- accuracy: `0.9167`
- mean requests: `6.875`
- median requests: `4.5`
- stop by rule: `22/24`
- forced end-of-trace: `2/24`

Selected-policy errors:

| Case | True pathology | Predicted pathology | Requests before stop | Stop mode | Key signal |
|---|---|---|---:|---|---|
| `test:81691` | `Croup` | `Acute otitis media` | 7 | rule stop | LLM and MLP agreed with very high MLP confidence |
| `test:62878` | `Pericarditis` | `Panic attack` | 23 | forced end-of-trace | LLM and MLP still agreed on the wrong diagnosis late in the trace |

The `Croup` error is the more important stopping-policy failure: both heads converged confidently on the wrong diagnosis early enough for the stop rule to fire. The `Pericarditis` error is more of a trajectory/question-selection failure because it remained wrong even after nearly the full recorded evidence path.

## Higher-Budget Sweep Finding

The best overall offline policies were not the selected matched-budget policies. At a higher request budget, the replay sweep found MLP-final policies reaching `23/24`.

| Policy family | Stop signal uses MLP? | Final head | Correct | Accuracy | Top-5 | Mean requests | Forced end rate |
|---|---:|---|---:|---:|---:|---:|---:|
| LLM-confidence stop, `llm_conf_ge_0.85_stab_2` | no | MLP | 23/24 | 0.958 | 1.000 | 9.83 | 0.458 |
| Deterministic-state stop, `det_margin_ge_3.00_unres_le_0.05` | no | MLP | 23/24 | 0.958 | 1.000 | 9.96 | 0.250 |

This does not overturn the matched-budget conclusion because these policies use about `3` more requests than the selected policy and rely on the MLP final head. It does suggest a useful second operating point:

- efficiency point: `22/24` at about `6.9` requests using MLP-guided stopping
- accuracy-biased point: `23/24` at about `9.8-10.0` requests using a later stop and MLP final diagnosis

The only remaining error for the high-budget MLP-final policies is `test:62878`, true `Pericarditis`.

## Interpretation

This ablation supports the claim that the MLP is not merely making the system stop earlier. Under the same approximate evidence budget, the best MLP-guided stopping rules preserve more accuracy than the best pure LLM-only stopping rule found in the sweep.

The practical result is:

- pure LLM-only stopping at about this budget: `20/24`
- MLP-guided stopping at about this budget: `22/24`
- observed notebook `08` accuracy at much higher budget: `22/24` with `13.04` requests

So the offline replay suggests that MLP-guided stopping can recover the high-accuracy notebook `08` result while using roughly half as many requested evidence fields.

This is stronger than the original hybrid v1 live result because it isolates the stopping decision. The live hybrid notebook changed the interaction online, so it mixed stopping effects with changed future evidence paths. Notebook `12` holds the evidence path fixed and tests stopping quality directly.

The higher-budget finding also suggests that the MLP is useful not only as a stopping signal but also as a final diagnostic head once enough evidence has accumulated. That should be treated carefully because it is still offline replay, but it is a strong reason to test a live hybrid policy with a conservative MLP-final option.

## What This Does Not Prove

This is not a full live-policy proof.

Offline replay can test different stop points along an existing trajectory. It cannot test how the LLM would choose future questions if a different stopping rule changed earlier turns, prompt state, or confidence framing.

Therefore, the correct claim is:

> On fixed 24-case notebook `08` evidence trajectories, MLP-guided stopping dominated the best tested pure LLM-only stop rule at the target request budget.

The next live experiment should confirm whether this holds when the selected stop rule is used online.

## Recommended Next Step

Implement the selected stopping rule in a small live successor run:

- keep `gpt-4.1-mini`
- keep `temperature = 0.0`, `top_p = 1.0`
- use the same 24-case slice first
- run only the selected MLP-guided stopping rule, not another wide lambda sweep
- compare against notebook `08` lambda `0.10` and notebook `11` lambda `0.22`

If live confirmation stays near `22/24` while using about `6-7` requests, then the project has a much stronger evidence-efficiency claim:

> MLP-guided stopping lets a sequential diagnostic workup retain high diagnostic accuracy while acquiring substantially less evidence.

Optional second live check after that:

- test an accuracy-biased policy that stops later and uses the MLP final head
- target about `9-10` requests
- compare whether it can approach the offline `23/24` result without returning to the `13`-request notebook `08` budget

## Live Confirmation Notebook Status

Notebook `13_live_selected_hybrid_stopping_confirmation.ipynb` has now run the live confirmation.

Status:

- dry-run smoke validation passed
- live API confirmation completed

Default live artifact root:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

Report:

- `reports/hybrid/live_selected_hybrid_stopping_confirmation.md`

Live confirmation result:

| System | Correct | Accuracy | Mean requests |
|---|---:|---:|---:|
| Notebook `08`, lambda `0.10` | 22/24 | 0.917 | 13.04 |
| Notebook `12`, offline selected stop | 22/24 | 0.917 | 6.875 |
| Notebook `13`, live selected stop | 22/24 | 0.917 | 6.58 |

The live run confirms the replay result: the selected MLP-guided stop rule preserved notebook `08`'s accuracy while using about half as many requested evidence fields. The remaining failures were `Croup` and `Pericarditis`, which points toward evidence trajectory and disease-specific confusion rather than pure stop-policy failure.
