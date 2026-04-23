# Sequential Policy Refinement Report

## What Was Going Wrong

The earlier structured sequential notebook (`04_single_agent_structured_policy_improvement.ipynb`) was no longer failing because of opaque evidence tokens or obviously broken legality. The main problem had shifted to **policy drift**:

- the LLM's latest ranked differential was steering the shortlist too strongly
- once the model drifted to a weak diagnosis, the shortlist started asking questions that reinforced that drift
- stop behavior looked efficient, but it was not strongly tied to whether the top competing diagnoses had actually been separated
- the one-shot prior was present, but it was too weakly coupled to the evolving sequential state

In short: the ledger was recording the state correctly, but the policy was still too eager to follow the model's last guess instead of maintaining an anchored differential.

## What Was Changed

A successor notebook was created:

- [05_single_agent_structured_policy_refinement.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/05_single_agent_structured_policy_refinement.ipynb)

The core refinements stay within the same single-agent architecture:

- deterministic diagnosis-state manager
  - builds a stable differential from the one-shot prior plus revealed evidence
  - tracks margin, unresolved mass, and prior strength
- shortlist tied to competing diagnoses
  - scores actions by how well they separate the current top competing diagnoses
  - penalizes generic high-frequency questions more aggressively
  - limits repeated overexposure to the same parent-question family
- policy controller / drift guard
  - can veto premature stop decisions when the differential is still unresolved
  - can veto drift-heavy diagnosis jumps when the deterministic state is clearly anchored elsewhere
- replay diagnostics
  - can replay the older live run on the same revealed evidence and compare the old predictions against the refined diagnosis-state predictions without spending more API budget

## What Was Tested

Two cheap validation passes were run.

### 1. Offline replay against the earlier live structured-policy run

Artifact root:

- [single_agent_refined_dryrun_test_1perclass_4budgets_anchor_guard_v1](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_dryrun_test_1perclass_4budgets_anchor_guard_v1)

Replay summary:

- [replay_source_run_summary.csv](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_dryrun_test_1perclass_4budgets_anchor_guard_v1/replay_source_run_summary.csv)

Results on the exact same revealed evidence as notebook 04:

- budget `1`: source `0.30` -> refined state `0.20`
- budget `3`: source `0.20` -> refined state `0.40`
- budget `5`: source `0.10` -> refined state `0.50`
- budget `8`: source `0.20` -> refined state `0.60`

Interpretation:

- budget `1` gets slightly worse because the refined state manager is intentionally less eager to overtrust a weak first clue
- budgets `3+` improve strongly, which supports the diagnosis that the main failure mode was **bad evidence revision and drift**, not lack of usable information

### 2. End-to-end dry-run sweep of the refined policy logic

Budget sweep summary:

- [budget_sweep_summary.csv](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_dryrun_test_1perclass_4budgets_anchor_guard_v1/budget_sweep_summary.csv)

Dry-run results on the 10-case benchmark:

- budget `1`: accuracy `0.20`
- budget `3`: accuracy `0.40`
- budget `5`: accuracy `0.70`
- budget `8`: accuracy `0.60`

Interpretation:

- the refined policy mechanics now extract real value from additional evidence
- unlike notebook 04, higher request budgets are no longer uniformly harmful
- the peak around budget `5` suggests the new policy is much better at using extra evidence, even if it can still over-ask beyond that point

Important caveat:

- this dry-run sweep validates the **policy logic**, not final live LLM behavior
- the live API-backed result still needs to be rerun with notebook 05

## What This Means

This is a meaningful troubleshooting result.

It does **not** prove the sequential system now beats the one-shot baseline in live mode. But it does show:

- the earlier poor sequential performance was not just “LLMs are bad at diagnosis”
- the refined diagnosis-state and action-scoring logic materially improves how extra evidence is interpreted
- the remaining bottleneck is now more clearly the interaction between the LLM and the improved controller, not a broken notebook interface

## Remaining Weaknesses

Even with the refined policy logic, some cases remain difficult:

- weak priors can still send the state manager into noisy respiratory/infectious clusters
- some diseases with overlapping evidence still remain hard to separate without richer value-level statistics
- after about `5` requests, there is still some evidence that extra questioning can become noisy again on hard cases

## Recommended Next Step

Run notebook 05 live on a small benchmark first:

- `RUN_LIVE_API = True`
- keep `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- keep `SEQUENTIAL_MAX_CASES = 10`
- use the same budget sweep `[1, 3, 5, 8]`

That will tell us whether the improved controller is strong enough to pull the LLM into the same healthier regime seen in the replay and dry-run diagnostics.

## Live Notebook 05 Result

The live run has now been completed.

Live artifact root:

- [single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1)

Live budget sweep summary:

- [budget_sweep_summary.csv](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1/budget_sweep_summary.csv)

Live results on the 10-case benchmark:

- budget `1`: accuracy `0.20`, top-3 `0.30`, top-5 `0.40`
- budget `3`: accuracy `0.20`, top-3 `0.40`, top-5 `0.80`
- budget `5`: accuracy `0.50`, top-3 `0.60`, top-5 `0.70`
- budget `8`: accuracy `0.80`, top-3 `0.80`, top-5 `0.80`

One-shot on the same 10 cases remained:

- accuracy `0.30`

So the refined sequential policy now beats the one-shot baseline decisively at higher budgets:

- budget `5`: sequential `0.50` vs one-shot `0.30`
- budget `8`: sequential `0.80` vs one-shot `0.30`

The comparison is especially important because notebook 04 had shown the opposite pattern:

- notebook 04 budget `5`: `0.10`
- notebook 04 budget `8`: `0.20`

That means notebook 05 is not a cosmetic cleanup. It materially changed the live policy behavior.

### What Improved

Compared with notebook 04 on the same 10-case benchmark:

- budget `5`: accuracy improved by `+0.40`
- budget `8`: accuracy improved by `+0.60`
- budget `8`: sequential-only wins increased from `1` to `5`
- budget `8`: both-wrong cases fell from `6` to `2`

Sequential-only wins at budget `8`:

- `Chagas`
- `Ebola`
- `Pulmonary embolism`
- `Stable angina`
- `Tuberculosis`

These are meaningful gains, not just label swaps around already-easy cases.

### Important Interpretation

The result now makes clinical sense:

- extra evidence is no longer uniformly harmful
- higher request budgets are actually paying off
- the system is recovering diagnoses that the one-shot model misses when enough targeted workup is allowed

There is still one important caveat:

- this is still only a `10`-case live sample, so the result is encouraging rather than conclusive

But the project is no longer in the “sequential policy looks doomed” state. The refined policy has crossed into a genuinely competitive regime on this slice.

### Remaining Failure Cases

Budget `8` still misses:

- `Croup` -> `Viral pharyngitis`
- `Pneumonia` -> `Myasthenia gravis`

This suggests the remaining bottlenecks are now narrower:

- some respiratory / upper-airway trajectories still drift into nearby infectious patterns
- some weakly anchored respiratory cases can still get pulled into unrelated high-margin states after several negative findings

### Notable Observation

In this live run:

- `drift_override_rate = 0.0`

So the gain did **not** come from hard overrides firing constantly. It mainly came from:

- the stronger deterministic diagnosis state
- the improved shortlist quality
- the improved prompt contract around anchored revision and stop behavior

That is a good sign because it means the agent itself is behaving better, not just being post-processed into correctness.
