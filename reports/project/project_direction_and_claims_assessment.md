# Project Direction And Claims Assessment

Last updated: 2026-05-06

## Executive Assessment

The project should stop expanding sideways for now. We have enough work to say the **simple sequential baseline line is mature**:

- initial one-shot baseline exists
- full-evidence ceiling exists
- LLM-only sequential policy exists
- matched-evidence classifier exists
- hybrid MLP-guided stopping exists
- offline ablation and live confirmation exist

The current evidence is not enough to make the broad claim that an “agentic diagnostic system is better than direct neural diagnosis.” That claim is too vague and too easy to attack because the full-evidence neural classifier is near-perfect and the partial-evidence neural classifier is competitive once it receives useful evidence.

The stronger, cleaner, and currently defensible claim is narrower:

> On a balanced DDXPlus live confirmation, a structured sequential workup policy with online partial-evidence MLP stopping reached `43/49 = 0.878` accuracy and `0.939` top-5 while using only `6.59` requested evidence fields per case. Earlier 24-case controls showed that this MLP stop rule could preserve the best LLM-only sequential accuracy while reducing requested evidence by about half.

That is a real result. It is not just a demo. It should still be presented as a course-project live confirmation rather than a definitive benchmark against official DDXPlus RL methods.

## What The Project Is Really Becoming

The project started as a **Multi-Agent Diagnostic Workup Copilot** idea. The implemented work has clarified the real core:

> Diagnostic workup is primarily an evidence-acquisition problem. The system should acquire a small, targeted subset of evidence, then diagnose from that evidence using the best available diagnostic head.

That means the project should be framed less as:

```text
LLM agent reasons better than a classifier.
```

and more as:

```text
A structured workup controller can efficiently reveal high-value evidence under incomplete information.
```

The strongest future architecture is therefore not “LLM-only” and not “MLP-only.” It is a controlled hybrid:

```text
DDXPlus patient episode
  -> deterministic evidence ledger
  -> LLM-led evidence acquisition
  -> partial-evidence MLP belief monitor
  -> MLP-guided stopping
  -> final diagnosis by LLM, MLP, or conservative adjudication
```

This framing is scientifically stronger because it separates three questions:

- What evidence should be acquired?
- When should the workup stop?
- Which diagnostic head should make the final prediction?

## Current Evidence Ladder

### 1. Initial-Evidence One-Shot

Artifact:

- `artifacts/one_shot/basd_pathology_full/`

Full-test result:

| System | Cases | Evidence | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| Initial-evidence MLP | 134,529 | age, sex, initial evidence only | 0.378 | 0.615 | 0.730 | 0.373 |

Interpretation:

- This is the official non-agentic baseline for incomplete evidence.
- It proves the starting state is diagnostically incomplete.
- It is a strong course baseline because it uses BASD-style DDXPlus encoding and a deep MLP, not a toy bag-of-words model.

### 2. Full-Evidence One-Shot Ceiling

Artifact:

- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/`

Full-test result:

| System | Cases | Evidence | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| Full-evidence MLP | 134,529 | all DDXPlus evidence | 0.996 | 1.000 | 1.000 | 0.995 |

Interpretation:

- DDXPlus contains enough structured information for near-perfect diagnosis when all evidence is visible.
- This is not the main baseline to beat. It is a ceiling comparator.
- Deduplication robustness checks found cross-split duplicates, but deduplicated performance stayed essentially unchanged, so the full-evidence result is not explained by duplicate leakage.

### 3. LLM-Only Cost-Sensitive Sequential Policy

Artifact:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/`

24-case live result:

| Lambda | Accuracy | Top-5 | Macro-F1 | Mean requests |
|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.917 | 0.846 | 13.04 |
| 0.22 | 0.875 | 0.917 | 0.795 | 10.67 |
| 0.35 | 0.875 | 0.875 | 0.813 | 8.33 |
| 0.50 | 0.417 | 0.750 | 0.274 | 2.21 |
| 0.75 | 0.375 | 0.708 | 0.288 | 1.04 |

Interpretation:

- Evidence acquisition clearly helps compared with the initial-evidence one-shot baseline on the same 24-case slice.
- The lambda sweep found a real cutoff: lambda `0.50+` stops too early and collapses.
- The best LLM-only policy reaches `22/24` but uses about `13` evidence requests per case.

### 4. Partial-Evidence Matched Classifier

Artifact:

- `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

Standalone policy-mask test result:

| System | Rows | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Partial-evidence MLP | 39,998 | 0.515 | 0.741 | 0.827 | 0.519 |

Matched 24-case integrated results:

| Lambda | Sequential acc | Partial matched MLP acc | Partial matched top-5 | Mean requests |
|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.875 | 1.000 | 13.04 |
| 0.22 | 0.875 | 0.875 | 1.000 | 10.67 |
| 0.35 | 0.875 | 0.833 | 0.958 | 8.33 |

Interpretation:

- This is one of the most important results because it prevents overclaiming.
- Once useful evidence is acquired, a direct partial-evidence MLP can almost match the LLM final answer.
- The project should not claim that LLM final reasoning is clearly superior.
- The more defensible point is that evidence acquisition itself is valuable, and neural belief estimates are useful for monitoring when enough evidence has been acquired.

### 5. Hybrid MLP-Guided Stopping

Artifacts:

- `artifacts/stopping_policy_ablation/stopping_policy_ablation_24case_v1/`
- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

Key results:

| System | Accuracy | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook 08 LLM-only, lambda 0.10 | 0.917 | 0.917 | 0.846 | 13.04 |
| Notebook 12 offline selected MLP stop | 0.917 | 0.917 | 0.867 | 6.875 |
| Notebook 13 live selected MLP stop, 24 cases | 0.917 | 0.917 | 0.867 | 6.58 |
| Notebook 13 live selected MLP stop, 49 cases | 0.878 | 0.939 | 0.845 | 6.59 |

Notebook 13 reduced requested evidence by:

- `49.5%` versus notebook 08 lambda `0.10`
- `42.2%` in input tokens versus notebook 08 lambda `0.10`

Interpretation:

- This is the strongest current result.
- The MLP is not only an offline comparator. It gives a useful online stopping signal.
- The hybrid advantage currently appears mainly in **when to stop**, not in final diagnosis adjudication.
- Final heads tied at top-1 in notebook 13: LLM final, MLP final, agreement hybrid, and conservative hybrid all reached `22/24`.
- MLP final had better top-5 in notebook 13: `0.958` versus `0.917` for the agreement hybrid.
- On the 49-case confirmation, agreement-hybrid/LLM/conservative final heads reached `43/49`, while MLP-final reached `41/49`.

## Are We Done With The Simple Sequential Agent?

Yes, as a baseline family.

The simple sequential agent line has done its job:

- it exposed the failure of naive prompting
- it showed that decoded evidence and ledger state matter
- it produced a cost-sensitive LLM-only policy
- it gave a meaningful live curve over evidence budget
- it gave a comparator for the hybrid stopping policy

We should not keep tuning the simple LLM-only sequential agent indefinitely. At this point it is a baseline, not the main research contribution.

The current main method should be the **hybrid evidence-efficiency system**:

- LLM chooses evidence requests
- deterministic ledger enforces legal state
- partial-evidence MLP monitors diagnostic certainty
- MLP-guided stop rule controls evidence efficiency

## What Claims Are Currently Defensible?

### Strongly Defensible Now

These claims are backed by full-scale training or direct live experiments:

1. **Initial DDXPlus evidence is insufficient for high diagnostic accuracy.**
   - Full-test initial-evidence MLP accuracy is about `0.378`.

2. **Complete DDXPlus evidence is highly diagnostic.**
   - Full-test full-evidence MLP accuracy is about `0.996`.

3. **Sequentially acquired evidence can recover a large part of the full-evidence performance gap on live balanced slices.**
   - Initial one-shot on the 24-case slice is `0.333`.
   - Best 24-case sequential result is `0.917`.
   - The broader 49-case hybrid confirmation reaches `0.878` top-1 and `0.939` top-5 with about `6.6` requests.

4. **MLP-guided stopping improves evidence efficiency on the 24-case live slice and remains efficient on 49 cases.**
   - Notebook 13 matches notebook 08's `22/24` accuracy with about half the requests on the 24-case pilot.
   - Notebook 13 keeps nearly the same request count on the 49-case confirmation: `6.59` mean requests.

### Defensible Only As Pilot Claims

These are promising but need a larger run before being presented strongly:

1. **The hybrid stop policy is generally better than LLM-only stopping.**
   - Notebook 12 supports this offline.
   - Notebook 13 confirms the selected rule live.
   - But a live matched-budget LLM-only control has not yet been run.

2. **The hybrid system approaches the full-evidence ceiling efficiently.**
   - On 24 cases, it recovers most of the gap.
   - On 49 cases, it remains high but visibly below the full-evidence ceiling.
   - This should be framed as evidence-efficient diagnosis under incomplete information, not near-ceiling diagnosis.

3. **The MLP diagnostic head improves differential ranking.**
   - Notebook 13 MLP final has better top-5.
   - But the slice is too small for a broad claim.

### Not Supported Yet

These claims should not be made:

1. **The LLM final answer is better than a direct neural classifier.**
   - The partial-evidence MLP is competitive under matched evidence.

2. **The current hybrid system improves final diagnosis beyond both LLM and MLP heads.**
   - Final heads tie in top-1 on notebook 13.

3. **A multi-agent architecture is better.**
   - No multi-agent experiment has been run yet.

4. **The method beats DDXPlus RL baselines.**
   - We have not implemented or directly reproduced the official RL baseline.

5. **The result is statistically conclusive.**
   - The strongest final live confirmation is `49` cases, which is stronger than the pilot but still not a definitive full benchmark.

## The Main Weakness In The Current Evidence

The current project has a strong pilot plus a broader 49-case confirmation, but not a final proof.

The issue is not that the method is weak. The issue is that live LLM experiments are still measured on relatively small balanced slices:

- `22/24` accuracy has a wide 95% Wilson interval of roughly `0.742` to `0.977`.
- `21/24` accuracy has a wide interval of roughly `0.690` to `0.957`.
- On 24 cases, one case changes accuracy by `0.0417`.

The 49-case confirmation reduces this problem but does not remove it. One case is still about `2.04` percentage points.

The second weakness is the missing live matched-budget LLM-only control:

- Notebook 12 offline replay found the best pure LLM-only stop near the same budget at `20/24`.
- Notebook 13 live selected MLP stop reached `22/24`.
- But we have not yet run a live LLM-only policy explicitly targeting `6-7` requests on the same slice.

This does not invalidate notebook 13. It only limits the strength of the claim.

## What Would Prove The Strong Claim?

The strong claim should be:

> A structured sequential DDXPlus workup controller with MLP-guided stopping preserves high diagnostic accuracy while acquiring substantially fewer evidence fields than an LLM-only sequential controller.

To prove that strongly, the project needs three things.

### 1. A Matched-Budget Live LLM-Only Control

Run one live control, not another broad sweep:

- same 24 cases first
- same `gpt-4.1-mini`
- same deterministic API settings
- same evidence shortlist and ledger
- no MLP stop signal
- target about `6-7` mean requests

If this control reaches `22/24`, then the hybrid stop advantage weakens. If it reaches around the notebook 12 offline LLM-only result (`20/24`) or notebook 11 level (`21/24`), then the hybrid stop claim becomes much stronger.

This is the cleanest missing experiment if the goal is to defend the stop-policy claim.

### 2. A Larger Live Confirmation

The next scale should not be a new method. It should be the same selected method on more cases.

Reasonable options:

| Scale | Purpose | Cost profile | Interpretation strength |
|---:|---|---|---|
| 49 cases | one per pathology | moderate | good course-level confirmation |
| 98 cases | two per pathology | higher | stronger stability check |
| 245 cases | five per pathology | expensive | closest to a serious benchmark slice |

The 49-case confirmation has now been run for notebook `13`. A larger run is optional, not required for the course project unless the team wants a stronger research-style benchmark. The cleanest remaining control is a matched-budget LLM-only live run, if budget allows.

### 3. A Fixed Claim And Frozen Method

Before scaling, freeze the method:

- fixed model: `gpt-4.1-mini`
- fixed temperature: `0.0`
- fixed top-p: `1.0`
- fixed selected stop rule
- fixed shortlist logic
- fixed final-head reporting
- no manual case-specific prompt edits

Otherwise the benchmark keeps moving, and the project becomes hard to defend.

## Recommended Project Direction

The next phase should be **confirmation and consolidation**, not invention.

### Freeze These As Historical/Baseline Notebooks

- Notebook 01: initial one-shot
- Notebook 07: full-evidence ceiling
- Notebook 08: LLM-only cost-sensitive sequential baseline
- Notebook 10: partial-evidence MLP comparator
- Notebook 12: offline stop-policy ablation
- Notebook 13: live selected hybrid stop confirmation

### Treat These As Development History

- Notebook 02: early sequential baseline
- Notebook 03: early comparison
- Notebook 04: first structured improvement
- Notebook 05: refined policy
- Notebook 06: budget scaling history
- Notebook 11: hybrid v1 lambda sweep

### Current Main Method

Notebook 13 is the current main method:

> live single-agent sequential workup with deterministic ledger, LLM evidence acquisition, partial-evidence MLP monitoring, and MLP-guided stopping.

Its final confirmation result is `43/49 = 0.878` accuracy, `0.939` top-5, and `6.59` mean requested evidence fields.

Notebook 14 tested a stronger-looking v2 idea where the MLP also guided the evidence shortlist. It was not promoted:

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Notebook 13 hybrid v1 selected stop | 0.917 | 0.917 | 6.58 |
| Notebook 14 MLP-discriminative shortlist | 0.875 | 0.958 | 7.38 |

This means the project should not claim that direct MLP-driven question selection is currently better. The evidence supports MLP-guided stopping more strongly than MLP-guided shortlisting.

### Next Experiment If More Work Is Needed

Only do a new notebook if it answers a specific control question:

> Can an LLM-only stop policy match notebook 13 at the same evidence budget?

If yes, create one control notebook. If not, do not add more notebooks. Move to report writing and instructor presentation.

## How To Present This To The Instructor

The project should be presented as an evidence-acquisition study:

1. DDXPlus diagnosis from only initial evidence is hard.
2. Diagnosis from full evidence is nearly solved by a neural classifier.
3. Therefore, the real problem is deciding which missing evidence to acquire.
4. A naive LLM sequential agent initially failed.
5. Adding a deterministic ledger, legal evidence actions, decoded fields, one-shot priors, and cost-sensitive stopping made the sequential policy useful.
6. A partial-evidence MLP trained on policy-shaped evidence masks showed that a neural head can use acquired evidence effectively.
7. The best current hybrid uses the LLM for question selection and the MLP for stopping.
8. On a 24-case live pilot, this hybrid retained `22/24` accuracy while cutting evidence requests from `13.04` to `6.58`.
9. On the 49-case confirmation, it reached `43/49` accuracy with `6.59` requests and `0.939` top-5.
10. Notebook 14 tested direct MLP-guided question selection and was rejected, so notebook 13 is the frozen proposed method.

## Bottom Line

The work is not rootless. The frame of reference is now clear:

- initial-evidence one-shot: lower bound for incomplete information
- full-evidence one-shot: ceiling for complete information
- LLM-only sequential: baseline evidence-acquisition controller
- partial-evidence MLP: matched-information diagnostic head
- hybrid MLP-guided stopping: current proposed improvement

The current work is enough to claim:

> We built a rigorous DDXPlus baseline ladder and found that online MLP-guided stopping supports evidence-efficient sequential diagnosis: the final 49-case confirmation reached `43/49` accuracy with about `6.6` requested evidence fields per case, after the 24-case pilot showed that this stop signal could match the best LLM-only result with about half the requests.

The current work is not enough to claim:

> Our agentic architecture is generally superior to all direct neural baselines or official DDXPlus sequential methods.

Notebook 14 reinforces this direction: it was useful to test, but the result makes notebook 13 the cleaner current method. The right next move is to stop broad experimentation, freeze notebook 13 as the current method, and either:

- run one matched-budget LLM-only live control, or
- move into final writeup and presentation,

then write the final project around evidence-efficient diagnostic workup rather than around unconstrained agentic superiority.
