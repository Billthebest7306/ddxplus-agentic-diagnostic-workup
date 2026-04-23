# Proposed Improvement 1 Results

## Scope

This report summarizes the first live run of:

- [04_single_agent_structured_policy_improvement.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/04_single_agent_structured_policy_improvement.ipynb)

Artifact root:

- [single_agent_improved_live_test_1perclass_4budgets_ledger_shortlist_budget_sweep_v1](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_improved/single_agent_improved_live_test_1perclass_4budgets_ledger_shortlist_budget_sweep_v1)

This notebook was designed to test a more structured single-agent policy with:

- a deterministic evidence ledger
- decoded evidence only
- legal action gating
- deterministic action shortlisting
- optional one-shot priors
- budget sweeps across multiple request limits

The run used a small live benchmark:

- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = 10`
- request budgets: `[1, 3, 5, 8]`

## Main Result

The new notebook is a **clear engineering improvement**, but on this first live evaluation it is **not yet a meaningful performance improvement** over the one-shot baseline, and it does not show a stable performance gain as the request budget increases.

The strongest result in this sweep was actually the smallest budget:

- budget `1`: sequential accuracy `0.30`

That exactly matched the one-shot baseline on the same 10 cases:

- one-shot accuracy `0.30`

As the budget increased, the improved sequential system did **not** improve:

- budget `3`: sequential accuracy `0.20`
- budget `5`: sequential accuracy `0.10`
- budget `8`: sequential accuracy `0.20`

So the current evidence does **not** support the claim that this first structured-policy version is already outperforming either:

- the one-shot baseline
- or the earlier best one-shot reference on these cases

## Budget Sweep Summary

From [budget_sweep_summary.csv](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_improved/single_agent_improved_live_test_1perclass_4budgets_ledger_shortlist_budget_sweep_v1/budget_sweep_summary.csv):

- budget `1`
  - sequential accuracy: `0.30`
  - top-3: `0.40`
  - top-5: `0.40`
  - macro-F1: `0.1875`
  - mean requests: `1.0`
  - one-shot accuracy on same cases: `0.30`
- budget `3`
  - sequential accuracy: `0.20`
  - top-3: `0.40`
  - top-5: `0.40`
  - macro-F1: `0.125`
  - mean requests: `2.7`
  - one-shot accuracy on same cases: `0.30`
- budget `5`
  - sequential accuracy: `0.10`
  - top-3: `0.40`
  - top-5: `0.40`
  - macro-F1: `0.0625`
  - mean requests: `4.4`
  - one-shot accuracy on same cases: `0.30`
- budget `8`
  - sequential accuracy: `0.20`
  - top-3: `0.40`
  - top-5: `0.40`
  - macro-F1: `0.125`
  - mean requests: `7.0`
  - one-shot accuracy on same cases: `0.30`

## Does This Make Sense?

Yes, the result is disappointing, but it does make sense.

The behavior suggests the following:

### 1. The structured notebook is not broken

The run appears operationally clean:

- no invalid-request contamination
- no schema failures dominating the run
- legality and gating are working
- the agent stops legally instead of looping indefinitely
- budget-specific artifacts and comparisons were all written correctly

So this does **not** look like the earlier “broken interface” problem from the first sequential notebook.

### 2. More budget is hurting because the policy is drifting

This is the main finding.

Per-case behavior shows that extra turns often make the answer worse rather than better.

Examples:

- `Acute dystonic reactions`
  - budget `1`: correct
  - budgets `3`, `5`, `8`: drifted to `Myasthenia gravis`
- `Guillain-Barré syndrome`
  - budget `1` and `3`: correct
  - budgets `5` and `8`: drifted to `Myasthenia gravis`
- `Pulmonary embolism`
  - only became correct by budget `8`
- `Whooping cough`
  - stayed correct across all budgets

So the pattern is not “the model cannot use more evidence.” The pattern is:

> more evidence often causes the current policy to overfit to the wrong hypothesis.

### 3. The shortlist is still too generic

From the traces, the most common requested evidence under larger budgets included:

- `E_201` cough
- `E_53` pain somewhere
- `E_91` fever
- `E_41`, `E_124`, `E_129`, `E_181`

This is better than exposing the entire action space directly, but it still looks too generic.

The policy is narrowing the space, but not narrowly enough around the real competing diagnoses for each case.

### 4. The stop guidance is not protecting against drift

At larger budgets, the run increasingly stops early:

- budget `3`: stop-before-cap `0.3`
- budget `5`: stop-before-cap `0.6`
- budget `8`: stop-before-cap `0.9`

That looks efficient on paper, but accuracy did not improve with it.

This implies the stop logic is not firing after the model has genuinely converged. Instead, it is often stopping after the model has stabilized on the **wrong** diagnosis.

### 5. The top-k ranking did not improve either

Across all budgets, top-3 and top-5 remained flat:

- top-3: `0.40`
- top-5: `0.40`

That means the extra questioning is not even improving ranking quality on this small sample. It is mostly moving the top prediction around without improving the candidate set.

## What Improved In A Meaningful Way

Even though the accuracy result is weak, there are still meaningful improvements in the system itself:

- the episode state is now more structured and inspectable
- the ledger is a real source of truth, not just ad hoc prompt text
- legality is explicit and deterministic
- action selection is reproducible and controllable
- budget sweeps are now easy to run and analyze
- the notebook generates strong artifacts and plots for diagnosis-policy analysis

So this notebook is still valuable.

It is just currently more successful as a **better experimental platform** than as a better-performing sequential policy.

## Comparison To Earlier Sequential Baselines

This live run cannot be treated as a clean apples-to-apples comparison to the earlier sequential runs, because the benchmark slice is different.

So the right conclusion is **not**:

- “the improved notebook is strictly worse than the previous sequential notebook”

The right conclusion is:

- on this 10-case live slice, the new structured policy did **not** produce a meaningful empirical gain
- and increasing request budget did **not** translate into better diagnosis

## Interpretation

The results strongly suggest that the current bottleneck is now:

- **shortlist quality**
- **diagnosis-update stability**
- **stop policy quality**

and not:

- raw access to evidence
- legality handling
- decoded evidence presentation

This is actually a useful research outcome.

It means the next improvement should not be:

- “just give the agent more requests”

It should be:

- better case-specific shortlist generation
- better control over how the current diagnosis is revised after each reveal
- stronger safeguards against drift away from an initially plausible diagnosis

## Most Likely Reasons For Underperformance

Based on the traces, the likely issues are:

### 1. The deterministic shortlist heuristic is too coarse

It uses pathology evidence-presence statistics and optional one-shot priors, but it is still broad enough to keep surfacing generic questions.

### 2. The model is over-updating on later evidence

Instead of using later evidence to refine a good early hypothesis, it often jumps to a different wrong label with rising confidence.

### 3. The stop signal is more “efficiency-aware” than “correctness-aware”

The stop logic is able to stop the episode, but not necessarily at the right time.

### 4. The one-shot prior is not yet being used strongly enough as an anchor

The one-shot prior is present, but the current blend may be too weak to keep the sequential agent grounded when later evidence is noisy or generic.

## Bottom Line

This first run of Proposed Improvement 1 **does make sense**, but it is **not yet a meaningful empirical improvement**.

The new notebook is clearly better as:

- a structured, reproducible sequential evaluation framework
- a cleaner ledger-based experimental platform
- a better tool for analyzing budget/performance tradeoffs

But it is not yet better as:

- a stronger diagnostic policy

In short:

> the structured-policy direction is technically sound, but the current shortlist and stop-control logic are not yet strong enough to improve diagnostic performance.

## Recommended Next Step

The next improvement should target policy quality directly:

1. strengthen shortlist generation using the one-shot differential more aggressively
2. penalize generic high-frequency questions more strongly
3. make stopping depend on both confidence and diagnostic consistency under new evidence
4. add a stronger “do not abandon a high-quality prior without discriminative evidence” rule

That would still stay within the current single-agent scope while addressing the exact failure mode seen here.
