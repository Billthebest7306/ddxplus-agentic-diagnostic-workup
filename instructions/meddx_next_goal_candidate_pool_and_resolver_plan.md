# MEDDx Next Goal: Candidate-Pool Recovery And Generalizable Resolver Plan

## Goal Statement

Build the next offline research phase for the MEDDx-style multi-dataset diagnostic workup system.

The primary goal is to improve the system in three linked stages:

```text
 Stage 1: Candidate-pool recovery
 Stage 2: Evidence-card resolver improvement
 Stage 3: Integrated frozen policy and confirmation gate
```

Stage 1 addresses the current hard ceiling: **candidate-pool formation**. The current Notebook `51` scale artifact has:

| Metric | Current |
|---|---:|
| Workups | `900` |
| Unique patients | `300` |
| Final top-1 | `715/900` |
| Top-3 | `773/900` |
| Top-5 | `791/900` |
| Candidate-pool oracle | `809/900` |
| Candidate-pool misses | `91/900` |

Stage 1 should try to raise candidate-pool recall to at least:

```text
 Minimum target: 850/900
 Strong target: 865/900+
 Stretch target: 880/900+
```

A resolver-only improvement is not enough. If the correct diagnosis is absent from the pool, no final resolver can recover it. But pool recovery alone is also not enough: once the pool gets larger and stronger, the final resolver must learn to choose correctly among more close candidates.

The core research target is therefore:

> Can a dataset-native, tri-state, candidate-pool-aware evidence acquisition and expansion layer raise candidate-pool recall across DDXPlus, iCraft-MD, and RareBench without overfitting to the latest artifacts?

Stage 2 should improve final selection over the recovered pool:

```text
 Minimum final top-1 target: >= 760/900 on case-blocked evaluation.
 Strong final top-1 target: >= 780/900 on case-blocked evaluation.
 Stretch final top-1 target: >= 800/900 if candidate-pool recall rises enough.
 Transfer target: non-negative or positive transfer on the old Notebook 46 90-workup artifact.
 Regression target: no unacceptable regressions versus the current final answer.
```

Stage 3 should freeze one integrated policy only if Stages 1 and 2 both work under strict evaluation. The integrated policy should define:

```text
 when to expand the pool
 when to ask more targeted questions
 how candidates are scored
 how the resolver chooses or preserves the base answer
 when the system abstains from overriding
```

Do not claim a new final method unless it survives case-blocked and transfer evaluation. Treat label-fit/oracle results as diagnostic only.

## Non-Negotiable Generalization Rules

Do not tune directly on all `900` workups and report the same `900` performance as if it is generalization.

Use strict splits:

- group by unique patient/case, not by workup
- all budgets for the same patient stay in the same split
- split within each dataset
- evaluate DDXPlus, iCraft-MD, and RareBench separately
- keep the older Notebook `46` `90`-workup artifact as a transfer/regression check

Suggested split:

```text
 Train:      60% of unique cases per dataset
 Validation:20% of unique cases per dataset
 Test:      20% of unique cases per dataset
```

Also run diagnostics with:

- leave-one-budget-out checks
- leave-one-dataset-out checks where feasible
- old Notebook `46` transfer
- comparison against current Notebook `51` final predictions
- comparison against current candidate-pool oracle

Decision-time features must be label-free. It is allowed to use labels for training and validation, but not for case-specific rules, handwritten fixes, or threshold selection on test cases.

Avoid hidden leakage:

- do not include columns containing `truth`, `correct`, `gtpa`, `true_rank`, or direct label-match indicators as decision-time features
- do not select rules because they fix named recent failures
- do not use hidden full evidence in deployable scoring unless that evidence would have been requested or is part of a legal dataset-native adapter
- distinguish diagnostic label-fit/oracle results from deployable results in every artifact and report

## Core Diagnosis Of Current Failure

The current architecture is not failing only at the final resolver.

Notebook `52` showed:

```text
 Current final pipeline: 715/900
 Candidate-pool oracle: 809/900
 Missing true diagnosis: 91/900
```

This means:

- iCraft-MD is mostly a resolver-discrimination problem because candidate-pool recall is `100/100` at all budgets.
- DDXPlus has both resolver misses and candidate-pool misses.
- RareBench has the weakest candidate-pool behavior and does not benefit reliably from larger budgets.

The key failure categories to investigate:

1. **Questioning failure**
   The system asked evidence that did not bring the true diagnosis into the candidate pool.

2. **Candidate expansion failure**
   The right diagnosis was near a disease family/phenotype neighborhood but was not added to the candidate list.

3. **Negative evidence failure**
   Some graph/Bayes/offline layers may treat unrevealed evidence and explicitly absent evidence too similarly. This weakens rule-out reasoning.

4. **Dataset-native mismatch**
   DDXPlus, iCraft-MD, and RareBench need different evidence representations, even inside one shared MEDDx-style shell.

5. **Resolver semantic weakness**
   When the true diagnosis is already in the pool, the current numeric resolver sometimes lacks enough medical semantic reasoning to choose correctly.

## Three-Stage Research Program

### Stage 1: Candidate-Pool Recovery

Goal:

```text
 Raise candidate-pool recall from 809/900 to >= 850/900, ideally 865-880/900.
```

Main tools:

- failure observatory
- tri-state evidence scoring
- candidate-pool expansion
- pool-miss risk detector
- offline counterfactual questioning

Reason:

The current system cannot exceed `809/900` final top-1 on the saved workups because the correct diagnosis is absent from the pool in `91/900` cases.

### Stage 2: Resolver Upgrade

Goal:

```text
 Convert a larger, better candidate pool into higher top-1 accuracy.
```

Main tools:

- evidence-card features
- pairwise candidate comparison
- listwise reranking
- base-protected override rules
- optional constrained LLM adjudicator
- abstention / preserve-current logic

Reason:

Current resolver failures are also substantial. If candidate-pool recall improves but the resolver remains weak, the final system may add the right diagnosis to the pool without choosing it.

### Stage 3: Integrated Frozen Policy

Goal:

```text
 Combine pool recovery and resolver improvement into one deployable policy that survives transfer and is worth a small live pilot.
```

Main tools:

- fixed thresholds selected on validation only
- case-blocked test evaluation
- old Notebook 46 transfer
- request-cost accounting
- paired wins/regressions
- small live pilot only after offline gates pass

Reason:

The project needs a coherent final architecture, not isolated diagnostics.

## Phase 1: Build A Failure Observatory

Create the next notebook as an offline lab, tentatively:

```text
 notebooks/53_meddx_candidate_pool_recovery_lab.ipynb
 scripts/meddx_candidate_pool_recovery_lab_nb53.py
 artifacts/universal_meddx/meddx_candidate_pool_recovery_lab_v1/
 reports/algorithmic_ledger/meddx_candidate_pool_recovery_lab_report.md
```

Primary inputs:

- Notebook `51` live scale artifact:
  `artifacts/universal_meddx/meddx_scale_hypothesis_branching_confirmation_v1_meddx100/`
- Notebook `52` offline calibration artifact:
  `artifacts/universal_meddx/meddx_scale_offline_resolver_calibration_v1/`
- Notebook `46` old transfer artifact:
  `artifacts/universal_meddx/meddx_aligned_dataset_native_driver_v1_eval30/`
- DDXPlus train/validate/test data and evidence metadata
- existing DDXPlus partial-evidence MLP checkpoint
- RareBench mapping/HPO data from the MEDDxAgent external repo
- iCraft-MD data from the MEDDxAgent external repo

Required first outputs:

- `case_split_assignment.csv`
- `baseline_failure_map.csv`
- `candidate_pool_miss_audit.csv`
- `resolver_miss_audit.csv`
- `questioning_audit.csv`
- `dataset_budget_failure_summary.csv`
- `candidate_pool_recall_summary.csv`

The first notebook section should answer:

- Which datasets and budgets produce candidate-pool misses?
- Which true diagnoses are repeatedly missing?
- Were branches triggered on those cases?
- Did extra budget help, hurt, or do nothing?
- Did the true diagnosis appear in top-3/top-5 even when absent from the broader candidate pool?
- Which evidence sources contributed each candidate?
- Which cases are pure resolver failures versus acquisition/pool failures?

## Phase 2: Fix Evidence Semantics With Tri-State Scoring

Implement a tri-state evidence representation for all offline graph/Bayes candidate scoring:

```text
 unknown / not requested: 0 contribution
 present / positive:      log-likelihood contribution from presence
 absent / negative:       log-likelihood contribution from absence
```

For a binary root:

```text
 score_present(disease, root)
   = log P(root present | disease) / P(root present | not disease)

 score_absent(disease, root)
   = log P(root absent | disease) / P(root absent | not disease)
```

Use smoothing:

```text
 p = (count + alpha) / (n + 2 * alpha)
```

Clip contributions conservatively:

```text
 clip(log_odds, -3, +3)
```

Critical rule:

```text
 Do not treat absent as unknown.
 Do not treat unknown as negative.
```

DDXPlus is the cleanest place to do this because exact evidence roots and absence states are available once requested.

Artifacts:

- `tri_state_evidence_features.csv`
- `tri_state_graph_bayes_candidate_scores.csv`
- `tri_state_vs_old_pool_recall_summary.csv`
- `tri_state_failure_delta.csv`

Success condition:

Tri-state scoring should improve candidate-pool recall or candidate ranking on validation/test splits without hurting transfer. If it only improves label-fit diagnostics, do not promote it.

## Phase 3: Candidate-Pool Expansion Before Final Resolution

Implement a candidate-pool maximizer. The pool should become a union of independent sources:

```text
 candidate_pool =
   base LLM ranked differential
   + DDXPlus MLP top-k
   + tri-state Bayes top-k
   + graph/ontology top-k
   + RareBench HPO phenotype-neighbor top-k
   + iCraft exemplar/text-neighbor top-k
   + branch hypotheses
   + disease-family / synonym / neighbor expansions
```

This stage is about recall, not final top-1. The goal is to make sure the correct diagnosis is present before asking the final resolver to choose.

Evaluation:

- pool size distribution
- candidate-pool recall
- recall by dataset/budget
- recall by source family
- marginal contribution of each source
- precision-like diagnostics: how many extra candidates are added per recovered truth?
- transfer performance on old `90`

Important:

Do not let the pool explode without cost. Track:

```text
 mean pool size
 median pool size
 p90 pool size
 recoveries per added candidate
```

Reasonable target:

```text
 candidate-pool recall >= 850/900
 mean pool size <= 10 if possible
 p90 pool size <= 15 if possible
```

## Phase 4: Pool-Miss Risk Detector

Train a label-free-at-decision-time model to estimate:

```text
 P(true diagnosis is missing from current pool)
```

Training labels can be derived from artifacts:

```text
 y = 1 if true diagnosis absent from current candidate pool
```

But features must be available at runtime:

- dataset name
- budget
- current pool size
- top-score margin
- source disagreement
- entropy/margin from DDXPlus MLP where available
- number of independent sources supporting top candidate
- graph/Bayes/MLP rank disagreement
- branch trigger information
- candidate family diversity
- RareBench phenotype coverage
- DDXPlus observed-root count and absent/present balance

Use simple, interpretable models first:

- logistic regression
- calibrated random forest / ExtraTrees as diagnostic
- isotonic or Platt calibration if needed

Evaluate:

- AUROC/AUPRC for pool-miss detection
- calibration curve
- recall of pool-miss cases at fixed trigger rates
- dataset-specific performance
- false-positive rate and expected added request cost

Artifacts:

- `pool_miss_risk_features.csv`
- `pool_miss_risk_validation_summary.csv`
- `pool_miss_risk_calibration_curve.png`
- `pool_miss_trigger_policy.json`

Promotion condition:

The trigger must identify many pool misses without firing constantly. If it only works on the latest artifact and fails transfer or case-blocked test, do not use it.

## Phase 5: Offline Counterfactual Questioning Lab

This is the most important part.

Do not spend API yet. Use the hidden dataset rows only as an offline simulator, while ensuring policy selection uses only runtime-visible state.

For DDXPlus:

- candidate roots are legal evidence roots
- requested roots reveal exact present/absent/value states from the row
- score possible next roots using train-derived tri-state likelihood and current pool-miss risk

For RareBench:

- candidate questions correspond to phenotype/HPO features or phenotype groups
- use known phenotype profile as offline reveal
- rank HPO features by ability to separate current candidate neighborhoods
- include HPO ancestors/synonyms/semantic similarity

For iCraft-MD:

- candidate recall is already strong, so focus on minimal changes
- test exemplar/text-neighbor candidate expansion and resolver support

Question utility should be candidate-pool-oriented:

```text
 utility(question) =
   expected candidate-pool recall gain
   + disease-family separation
   + value of negative evidence
   + source-disagreement reduction
   - redundancy penalty
   - request cost
```

Do not optimize directly for known test truth. Use train/validate statistics to rank evidence. In offline replay, the hidden row is used only to reveal the outcome of a selected question, just like the real simulator.

Outputs:

- `counterfactual_question_policy_summary.csv`
- `counterfactual_question_trace.jsonl`
- `pool_recall_after_counterfactual_questions.csv`
- `added_request_cost_summary.csv`
- `candidate_recovery_examples.json`
- `question_utility_ablation_summary.csv`

Candidate policies to compare:

1. Current Notebook `51` behavior
2. Tri-state Bayes expansion only
3. HPO/ontology expansion only
4. Pool-miss-risk-triggered expansion
5. Pool-miss-risk-triggered extra questioning
6. Combined candidate-pool maximizer

Success condition before any live run:

```text
 candidate-pool recall >= 850/900
 no catastrophic pool-size explosion
 improvement appears on held-out case-blocked test split
 transfer to old 90 does not regress badly
```

## Phase 6: Resolver Redesign After Pool Recovery

Once candidate-pool recall improves, revisit the final resolver. This is Stage 2 of the overall plan, not optional cleanup.

Current numeric resolver is weak because it mostly sees ranks, source counts, margins, and shallow support. Improve it with richer evidence cards.

For each candidate, build an evidence card:

```text
 candidate disease
 supporting present evidence
 contradicting absent evidence
 missing high-value evidence
 graph / Bayes / MLP / HPO / exemplar ranks
 source agreement
 close confounders
 dataset-specific notes
```

Resolver variants:

1. Conservative numeric resolver
   - L2 logistic or calibrated gradient boosting
   - case-blocked validation
   - base-protection rules

2. Pairwise candidate resolver
   - compare candidate A vs candidate B using evidence-card differences
   - aggregate pairwise wins into a final ranking

3. Listwise resolver
   - rank all candidates together
   - optimize top-1 and top-3/top-5

4. Constrained LLM adjudicator
   - only over candidate pool, not freeform diagnosis
   - sees revealed evidence cards
   - must cite revealed evidence
   - cannot invent unavailable evidence
   - can abstain or preserve base answer

The LLM adjudicator should not be the first fix, because it costs API and cannot recover missing candidates unless allowed to propose outside-pool diagnoses. Treat it as a later low-cost pilot after offline evidence-card generation works.

Potential LLM adjudicator prompt shape:

```text
 You are adjudicating among candidate diagnoses.
 Use only the revealed evidence listed below.
 For each candidate, state:
   - evidence supporting it
   - evidence contradicting it
   - key missing discriminator evidence
 Rank the candidates.
 If evidence is insufficient to override the base diagnosis, say so.
 Do not introduce a new diagnosis unless none of the candidates fit.
```

Stage 2 promotion condition:

- improves final top-1 on case-blocked test
- no unacceptable regressions
- transfers to old `90`
- top-3/top-5 do not degrade
- request cost remains controlled

Expected outcome range:

```text
 If Stage 1 raises pool recall to 850/900,
 a strong resolver should aim for 760-780/900 final top-1.

 If Stage 1 raises pool recall to 865-880/900,
 a strong resolver can reasonably target 780-800+/900 final top-1.
```

Do not report resolver performance only as label-fit accuracy. The useful result is case-blocked and transfer-tested top-1/top-3/top-5 with paired wins/regressions.

## Phase 7: Integrated Frozen Policy And Live-Run Gate

This is Stage 3 of the plan. Do not run a large live experiment until the offline gates pass.

Minimum live gate:

```text
 candidate-pool recall offline >= 850/900
 final deployable offline/test result improves current baseline
 resolver upgrade improves or preserves transfer
 old 90 transfer is non-negative or positive
 pool size remains manageable
 request-cost estimate remains below budget
```

Then run only a small pilot:

```text
 10-15 unique cases per dataset
 budgets: 5, 10, 15
 total: 90-135 workups
```

Only after the pilot works should a larger confirmation be considered.

## Reporting Requirements

Every notebook in this phase must separate:

- current live baseline
- deployable offline policy
- validation-selected policy
- test result
- transfer result
- label-fit diagnostic
- candidate-pool oracle
- full-evidence/non-deployable oracle

Do not mix these categories.

Required figures:

- candidate-pool recall by dataset and budget
- final top-1/top-3/top-5 by dataset and budget
- candidate-pool miss decomposition
- pool size versus recall
- request-cost delta
- risk-detector calibration curve
- source contribution waterfall
- old-versus-new paired wins/regressions

Required summary files:

- `resolved_run_config.json`
- `case_split_assignment.csv`
- `baseline_policy_summary.csv`
- `candidate_pool_recovery_summary.csv`
- `resolver_policy_summary.csv`
- `paired_current_vs_candidate.csv`
- `failure_decomposition_summary.csv`
- `selected_policy.json`
- `hard_case_audits.json`

Update:

- `PROJECT_WORKLOG.md`
- `README.md`
- `reports/README.md`
- `reports/final_results_summary.md`
- `reports/final_report.md`
- add a dedicated report under `reports/algorithmic_ledger/`

## Suggested Notebook Sequence

### Notebook 53: Candidate-Pool Recovery Lab

Main purpose:

- failure observatory
- tri-state graph/Bayes replay
- candidate-pool expansion
- pool-miss risk detector
- offline counterfactual question policy

Primary success criterion:

```text
 candidate-pool recall >= 850/900 on deployable, case-blocked evaluation
```

### Notebook 54: Evidence-Card Resolver Lab

Only if Notebook `53` improves candidate-pool recall.

Main purpose:

- build evidence cards
- evaluate numeric, pairwise, listwise, and optionally constrained LLM adjudicator resolvers

Primary success criterion:

```text
 final top-1 improves materially without unacceptable regressions
```

### Notebook 55: Frozen Small Live Pilot

Only if Notebook `53` and `54` pass.

Main purpose:

- small live validation before any large spend
- 10-15 cases per dataset
- budgets 5/10/15

Primary success criterion:

```text
 candidate-pool recall and final top-1 improve versus Notebook 51-style baseline
 without cost explosion
```

## Final Strategic Position

The next phase should not be "make a better resolver" in isolation.

The correct architecture is:

```text
 dataset-native evidence adapter
   -> tri-state evidence ledger
   -> candidate-pool completeness/risk estimation
   -> targeted pool-expanding questioning
   -> candidate-pool maximizer
   -> evidence-card resolver
   -> base-protected final decision
```

The main claim to pursue:

> A MEDDx-style diagnostic agent improves when evidence acquisition is optimized for candidate-pool completeness, not merely confidence or final top-1 prediction.

This is the cleanest research story because it explains why the 900-workup run failed, uses the existing artifacts productively, and creates a principled path to stronger results without burning more API blindly.
