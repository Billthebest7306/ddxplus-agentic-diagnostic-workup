# Project Direction And Claims Assessment

Last updated: 2026-05-08

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

> On a balanced DDXPlus live confirmation, a structured sequential workup policy with online partial-evidence MLP stopping reached `43/49 = 0.878` accuracy and `0.939` top-5 while using only `6.59` requested evidence fields per case. A fresh live run of the same backbone inside Notebook `24` reached `45/49 = 0.918` with `6.20` requests. Earlier 24-case controls showed that this MLP stop rule could preserve the best LLM-only sequential accuracy while reducing requested evidence by about half.

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
| Notebook 17 MedKGI graph shortlist, 24 cases | 0.833 | 0.875 | 0.744 | 6.21 |
| Notebook 20 LLM-led graph context, 24 cases | 0.833 | 0.958 | 0.744 | 6.13 |
| Notebook 22 graph posterior final critic, 49 cases | 0.898 | 0.939 | 0.867 | 6.59 |
| Notebook 23 calibrated graph-Bayes rescue, 49 cases | 0.959 | n/a | n/a | 6.96 |
| Notebook 28 MLP-gated branching, 49 cases | 0.898 | 0.959 | 0.864 | 6.63 selected / 9.96 total branch |
| Notebook 29 listwise differential adjudicator, 49 cases | 0.918 | n/a | n/a | 6.63 selected / 9.96 total branch |

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
- Notebook 17 reduced requests slightly versus Notebook 13 on the 24-case slice, but lost two additional cases; it is evidence that graph scores should be advisory, not a hard replacement shortlist.
- Notebook 22 is the first graph-ledger enhancement to improve the 49-case artifact: it keeps Notebook 13 acquisition unchanged and uses a conservative graph-posterior final critic to reach `44/49` with no extra requests.
- Notebook 23 is the first graph-ledger enhancement to materially change the saved-trace result: it keeps Notebook 13 as the first-pass workup and reaches `47/49` with `6.96` mean requests and zero regressions.
- Notebook 24 tested that rescue layer live. It did not promote the rescue layer, but the fresh live base reached `45/49` with `6.20` mean requests.
- Notebook 28 tested learned-gate live branching and reached `44/49` with zero regressions versus its own base.
- Notebook 29 tested ranked-differential listwise adjudication over frozen Notebook 28 traces and reached `45/49` with zero regressions, but did not reach the `47/49` promotion target.

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
- Notebook 14: rejected MLP-discriminative shortlist
- Notebook 15: offline stop-threshold/evidence-trajectory sensitivity
- Notebook 16: offline MedKGI-style graph evidence analysis
- Notebook 17: rejected live MedKGI graph shortlist pilot

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

Notebook 17 tested a MedKGI-style graph top-10 evidence shortlist while keeping Notebook 13's stop rule fixed. It was also not promoted:

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Notebook 13 hybrid v1 selected stop | 0.917 | 0.917 | 6.58 |
| Notebook 17 MedKGI graph shortlist | 0.833 | 0.875 | 6.21 |

The graph request metrics were strong locally, with mean requested graph rank `1.76`, mean information gain `0.373`, and no requests outside the graph top-10. The failure is therefore not that the graph cannot rank evidence. The failure is that a hard graph replacement shortlist can over-constrain the action space around an already-biased active differential.

Notebook 18 then tested the natural follow-up: use graph evidence as an advisory/blended signal instead of a hard replacement shortlist. It also was not promoted:

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Notebook 13 hybrid v1 selected stop | 0.917 | 0.917 | 6.58 |
| Notebook 17 MedKGI hard graph shortlist | 0.833 | 0.875 | 6.21 |
| Notebook 18 graph-advisory shortlist | 0.875 | 0.917 | 7.67 |

Notebook 18 recovered the Chagas and Ebola failures introduced by Notebook 17, which means the rare-support/advisory idea has some value. But it introduced a new Stable angina failure, did not fix Croup or Pericarditis, and used more requests than Notebook 13. The graph-advisory method is therefore useful as a diagnostic ablation, not as the new main method.

Notebook 20 corrected the graph-ledger framing by keeping the LLM as the question chooser and adding graph support/contradiction context only as prompt state:

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook 13 hybrid v1 selected stop | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook 20 LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

Notebook 20 is not a promoted method because top-1 accuracy dropped. However, its `23/24` top-3/top-5 result is the strongest ranking signal seen in the graph-ledger line.

Notebook 21 then tested whether graph context could convert that ranking signal into top-1 through non-oracle critic/adjudication rules:

| Result | Value |
|---|---:|
| Best non-oracle variant accuracy | 0.917 |
| Best non-oracle variant mean requests | 6.58 |
| Notebook 20 oracle top-3/top-5 upper bound | 0.958 |
| Selected live candidate | none |

The best non-oracle variant only matched Notebook 13 and did not fix any Notebook 13 miss. The graph context does strongly flag suspect answers: in Notebook 20, wrong top-1 predictions had much higher contradiction than correct predictions. But the hand-written rules still cannot reliably choose the correct alternative from the top-5.

Therefore the graph-ledger direction is not dead, but graph information should not be used as a blunt replacement controller. Notebook 22 shows the stronger path: use the train-derived evidence graph as a final-state mathematical critic over the evidence Notebook 13 already acquired.

The Notebook 22 post-run analysis sharpens that claim. The conservative critic fires exactly once on the 49-case trace and fixes Croup with no regressions. It also improves the overlapping 24-case sanity slice from `22/24` to `23/24` through the same Croup correction. The remaining errors show why this should stay framed as a critic rather than a controller: in COPD, the graph correctly weakens the Notebook 13 diagnosis but prefers another wrong alternative; in Pericarditis, the final evidence state does not support the true pathology strongly enough. The next credible graph step is therefore validation or calibration, not another hand-threshold controller.

Notebook 23 realizes the larger opportunity offline. It uses train/validate-derived synthetic partial evidence states, graph/Bayes candidate scoring, prior recovery, the Notebook 22 graph critic, and a tiny rescue continuation for suspicious early stops. It fixes COPD, Croup, Influenza, and Unstable angina without regressing any Notebook 13 correct case. The remaining misses are acute-vs-chronic rhinosinusitis and Pericarditis.

Notebook 24 completed the live-confirmation wrapper for Notebook 23. The live rescue did not reproduce the offline `47/49` gain: the fresh live base reached `45/49 = 0.918`, and the live rescue also reached `45/49 = 0.918` with nine extra rescue requests. This means the graph/Bayes rescue layer is not promoted, but the base LLM-led + MLP-stopped method is stronger than the original frozen artifact suggested.

Notebook 28 and Notebook 29 extend this line into prospective branching and ranked-differential adjudication. Notebook 28 improves its own live base from `42/49` to `44/49` with zero regressions. Notebook 29 then keeps those live traces fixed, explodes ranked differentials plus graph/Bayes/MLP candidates, and improves the selected result to `45/49` with one additional no-regression win. Its oracle analysis is the important signal: ranked top-2 contains `47/49`, ranked top-3 contains `48/49`, and the full exploded candidate pool contains all `49/49` true labels.

### Next Experiment If More Work Is Needed

Only do a new notebook if it answers a specific control question that is not already answered by Notebooks 14, 17, 18, 20, 21, 22, 23, 24, 28, and 29:

> Can the Notebook 23 calibrated graph-Bayes rescue layer hold up on held-out or live traces while keeping Notebook 13 as the first-pass workup?

Notebook 24 was the implementation of that question, and its live result is now complete. It did not promote the rescue layer. Notebook 13 is already a defensible live acquisition endpoint, and Notebook 23 remains the current offline graph-ledger rescue enhancement rather than a live replacement.

The post-Notebook-29 control question is narrower:

> Can a pairwise or abstaining confounder adjudicator decide when to promote rank-2/rank-3 differentials over a graph/Bayes/MLP consensus anchor without using the 49-case labels?

That is the next credible research direction if more work is needed.

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
11. Notebook 17 tested a MedKGI-style graph shortlist and was also rejected, showing that graph information is useful for analysis but should not hard-prune the action space yet.
12. Notebook 18 tested graph-advisory blending and was also rejected: it recovered Chagas and Ebola relative to Notebook 17 but dropped to `21/24` and used `7.67` requests.
13. Notebook 20 kept the LLM as question chooser and added graph context; it improved top-3/top-5 to `23/24` but dropped top-1.
14. Notebook 21 showed graph contradiction is a useful critic signal, but no non-oracle rule-based adjudicator beat Notebook 13.
15. Notebook 22 used train-derived graph log-odds as a final-state posterior critic over Notebook 13 evidence and improved the saved 49-case result to `44/49` with the same `6.59` requests.
16. Notebook 23 used calibrated graph/Bayes rescue to improve the saved 49-case trace to `47/49` with `6.96` mean requests.
17. Notebook 24 tested that rescue live; it was not promoted, but the fresh live base reached `45/49` with `6.20` mean requests.
18. Notebook 28 tested learned-gate live branching and graph/Bayes/MLP branch adjudication; it improved its own base to `44/49` with zero regressions.
19. Notebook 29 tested ranked-differential listwise adjudication over Notebook 28 traces; it reached `45/49` with zero regressions and showed a `47/49` to `48/49` oracle in the ranked differential pool.

## Bottom Line

The work is not rootless. The frame of reference is now clear:

- initial-evidence one-shot: lower bound for incomplete information
- full-evidence one-shot: ceiling for complete information
- LLM-only sequential: baseline evidence-acquisition controller
- partial-evidence MLP: matched-information diagnostic head
- hybrid MLP-guided stopping: current live proposed improvement
- calibrated graph/Bayes rescue reranker: current offline mathematical enhancement
- ranked-differential listwise adjudication: diagnostic path showing the `47/49+` candidate ceiling

The current work is enough to claim:

> We built a rigorous DDXPlus baseline ladder and found that online MLP-guided stopping supports evidence-efficient sequential diagnosis: the frozen live 49-case confirmation reached `43/49` accuracy with about `6.6` requested evidence fields per case, and a fresh live run of the same backbone reached `45/49` with `6.2` requests. Offline train-derived graph-ledger enhancements improved the saved 49-case trace to `44/49` with a simple graph critic and to `47/49` with a calibrated graph/Bayes rescue layer, but the live rescue confirmation did not promote that layer. Prospective branching and ranked-differential adjudication now show that `47/49+` is available in the candidate pool, while safe selection among close confounders remains unresolved.

The current work is not enough to claim:

> Our agentic architecture is generally superior to all direct neural baselines or official DDXPlus sequential methods.

Notebooks 14, 17, 18, 19, 24, 28, and 29 reinforce this direction: they were useful to test, but the results make notebook 13 the cleaner current live method. The right next move is to stop broad method-chasing, freeze notebook 13 as the current method, and treat any further work as targeted confounder adjudication rather than a new controller family.
