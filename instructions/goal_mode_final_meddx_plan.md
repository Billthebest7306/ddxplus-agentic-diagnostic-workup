# Goal Mode Final Plan: MEDDx Candidate-Pool Recovery, Resolver Upgrade, And Frozen Policy

## Start Here

Before doing any implementation, read the detailed plan in:

```text
instructions/meddx_next_goal_candidate_pool_and_resolver_plan.md
```

That file is the full technical planning document. This file is the copy-ready final goal-mode brief.

## Objective

Pursue a serious offline research phase to improve the MEDDx-style multi-dataset diagnostic workup system using the existing artifacts, with full freedom to try principled offline approaches.

The core target is to move beyond the current Notebook `51` / Notebook `52` result:

```text
Current final top-1:       715/900
Current top-3:             773/900
Current top-5:             791/900
Candidate-pool oracle:     809/900
Candidate-pool misses:      91/900
```

Primary target:

```text
Raise candidate-pool recall from 809/900 to at least 850/900.
Strong target: 865/900+
Stretch target: 880/900+
```

Secondary target:

```text
Improve final deployable top-1 after candidate-pool recovery.
Minimum final target: >= 760/900 on case-blocked evaluation.
Strong final target: >= 780/900.
Stretch final target: >= 800/900 if pool recall rises enough.
```

Transfer target:

```text
The selected policy should transfer non-negatively or positively to the old Notebook 46 90-workup artifact.
```

Do not claim success from label-fit diagnostics or oracle rows. Those are useful for analysis only.

## Three-Stage Plan

### Stage 1: Candidate-Pool Recovery

Main question:

```text
Can we get the correct diagnosis into the candidate pool much more often?
```

The current final system is capped because `91/900` workups do not contain the true diagnosis in the candidate pool. This must be addressed first.

Try all reasonable offline approaches, including but not limited to:

- tri-state evidence scoring, where unknown, present, and absent evidence are distinct
- DDXPlus exact-root Bayes/log-likelihood scoring
- DDXPlus graph evidence support and contradiction
- DDXPlus MLP top-k candidate expansion
- DDXPlus counterfactual root-question replay
- RareBench HPO phenotype expansion
- RareBench ontology ancestor/synonym/neighbor expansion
- RareBench phenotype semantic-similarity candidate expansion
- iCraft-MD exemplar/text-similarity support
- casebase retrieval and dynamic neighbor expansion
- candidate source-union/rank-fusion strategies
- pool-miss risk prediction
- pool-miss-triggered additional questioning
- source-disagreement-triggered branching
- negative-evidence-aware candidate scoring
- budget-specific candidate-pool policies

Stage 1 success condition:

```text
Candidate-pool recall >= 850/900 under deployable, case-blocked evaluation.
Pool size must remain controlled enough that the resolver is still meaningful.
```

Track:

- pool recall by dataset
- pool recall by budget
- pool size mean/median/p90
- recoveries per added candidate
- candidate-source contribution
- request-cost impact
- paired recoveries/misses versus current Notebook 51 pool

### Stage 2: Resolver Upgrade

Main question:

```text
Once the right diagnosis is in the pool, can we choose it more reliably?
```

Do not stop after improving candidate-pool recall. The final result depends on the resolver.

Try all reasonable offline resolver approaches, including but not limited to:

- evidence-card candidate features
- pairwise candidate comparison
- listwise candidate reranking
- calibrated logistic candidate scoring
- tree/boosting diagnostics with strict validation selection
- base-protected override rules
- graph/Bayes/MLP/HPO/exemplar score fusion
- source-consensus and source-disagreement features
- candidate-family and close-confounder features
- abstention / preserve-current policies
- constrained LLM adjudicator as a diagnostic or later low-cost pilot

The LLM adjudicator, if used, must be constrained:

```text
It should see candidate evidence cards.
It must rank existing candidates first.
It must cite revealed evidence.
It must not invent unavailable evidence.
It may propose one outside-pool diagnosis only under a clearly marked diagnostic escape hatch.
```

Stage 2 success condition:

```text
Final top-1 improves materially on case-blocked test evaluation.
Top-3/top-5 should not degrade.
Regressions must be tracked.
Transfer to the old Notebook 46 90-workup artifact must be non-negative or positive before promotion.
```

### Stage 3: Integrated Frozen Policy

Main question:

```text
Can candidate-pool recovery and resolver improvement be combined into one deployable policy?
```

The final output should not be a scattered collection of diagnostics. It should be one frozen candidate policy with a clear artifact contract.

The integrated policy must specify:

- when to expand the candidate pool
- when to ask additional targeted evidence questions
- how unknown/present/absent evidence is represented
- how graph/Bayes/MLP/HPO/exemplar scores are combined
- how the resolver chooses among candidates
- when the resolver preserves the base answer
- when overrides are forbidden
- how request cost is capped
- which thresholds were selected on validation only

Stage 3 success condition:

```text
The integrated policy passes offline gates and is worth a small live pilot.
```

Minimum live-run gate:

```text
Candidate-pool recall >= 850/900 offline.
Final deployable result improves current baseline on case-blocked test.
Old 90-workup transfer is non-negative or positive.
Pool size remains manageable.
Request-cost estimate is acceptable.
No label leakage or test-label threshold selection.
```

Only after those gates should a small live pilot be considered.

## Generalization And Anti-Overfit Rules

Do not overfit to the newest artifact.

Required evaluation discipline:

- split by unique patient/case, not by workup row
- keep all budgets for a patient in the same split
- stratify by dataset
- evaluate DDXPlus, iCraft-MD, and RareBench separately
- keep old Notebook `46` as transfer/regression evidence
- clearly separate train, validation, test, transfer, diagnostic label-fit, and oracle results

Forbidden:

- case-by-case hardcoded fixes
- selecting thresholds on test labels
- reporting label-fit rows as deployable
- using `truth`, `correct`, `gtpa`, `true_rank`, or equivalent label-derived fields as decision-time features
- treating unrevealed evidence as absent
- treating absent evidence as unknown
- hiding regressions behind aggregate accuracy

## Expected Notebook Sequence

Start with:

```text
notebooks/53_meddx_candidate_pool_recovery_lab.ipynb
scripts/meddx_candidate_pool_recovery_lab_nb53.py
artifacts/universal_meddx/meddx_candidate_pool_recovery_lab_v1/
reports/algorithmic_ledger/meddx_candidate_pool_recovery_lab_report.md
```

If Stage 1 succeeds, continue with:

```text
notebooks/54_meddx_evidence_card_resolver_lab.ipynb
scripts/meddx_evidence_card_resolver_lab_nb54.py
artifacts/universal_meddx/meddx_evidence_card_resolver_lab_v1/
reports/algorithmic_ledger/meddx_evidence_card_resolver_lab_report.md
```

If Stage 2 succeeds, create:

```text
notebooks/55_meddx_integrated_candidate_pool_resolver_policy.ipynb
scripts/meddx_integrated_candidate_pool_resolver_policy_nb55.py
artifacts/universal_meddx/meddx_integrated_candidate_pool_resolver_policy_v1/
reports/algorithmic_ledger/meddx_integrated_candidate_pool_resolver_policy_report.md
```

Do not create more notebooks blindly. Each notebook must answer a specific control question.

## Required Artifacts

Every notebook should write a complete artifact contract, including:

- `resolved_run_config.json`
- `case_split_assignment.csv`
- baseline/current policy summaries
- candidate-level features
- case-level results
- paired current-vs-candidate comparisons
- failure decomposition tables
- selected policy JSON
- hard-case audits
- figures under `figures/`

The selected policy JSON must clearly record:

- inputs used
- split strategy
- selected thresholds
- feature list
- promotion decision
- wins/regressions
- request-cost delta
- transfer result
- whether the result is deployable, diagnostic, or oracle-only

## Documentation Requirements

After each meaningful notebook:

- update `PROJECT_WORKLOG.md`
- update `README.md`
- update `reports/README.md`
- update `reports/final_results_summary.md`
- update `reports/final_report.md`
- add/update the dedicated report under `reports/algorithmic_ledger/`

Keep the claims honest:

```text
Live result
Offline deployable result
Calibration-only result
Label-fit diagnostic
Candidate-pool oracle
Full-evidence oracle
```

must always be separated.

## Final Research Claim To Aim For

The ideal final claim is:

> A MEDDx-style diagnostic workup system improves when evidence acquisition is optimized for candidate-pool completeness and tri-state evidence semantics, then paired with an evidence-card resolver, rather than relying only on confidence-based stopping or a final scalar resolver.

This is the strongest path because it directly addresses what went wrong in Notebook `51` and Notebook `52`, uses the existing `900` workups productively, and avoids burning API before offline gates show the new policy is worth testing.

