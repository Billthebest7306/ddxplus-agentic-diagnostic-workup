# Results Assessment

## Bottom Line

The one-shot side worked well. The sequential single-agent side did not.

Right now the evidence supports this claim:

> A strong one-shot DDXPlus baseline is reproducible and competitive, but the current naive single-agent sequential setup collapses under the large evidence-request action space and does not beat the one-shot model.

It does **not** support this claim yet:

> The agentic system improves diagnosis over the one-shot baseline.

There is also one important integrity note: the first sequential run was slightly contaminated by a repair fallback bug that leaked hidden differential information on some error cases. I patched the notebook, but the saved live sequential artifacts and comparison artifacts were produced **before** that patch, so they should be treated as stale until notebook 02 and notebook 03 are rerun.

## 1. One-Shot Results

Artifacts:

- [selected_model.json](/Users/bilalawan/claw/assignments/baseline_model/artifacts/one_shot/selected_model.json)
- [run_registry_full.csv](/Users/bilalawan/claw/assignments/baseline_model/artifacts/one_shot/run_registry_full.csv)
- [basd_pathology_full metrics](/Users/bilalawan/claw/assignments/baseline_model/artifacts/one_shot/basd_pathology_full/metrics.json)
- [basd_joint_full metrics](/Users/bilalawan/claw/assignments/baseline_model/artifacts/one_shot/basd_joint_full/metrics.json)
- [basd_differential_full metrics](/Users/bilalawan/claw/assignments/baseline_model/artifacts/one_shot/basd_differential_full/metrics.json)

Full-split results:

- `basd_pathology`:
  - accuracy: `37.82%`
  - top-3: `61.55%`
  - top-5: `73.03%`
  - macro-F1: `37.30%`
- `basd_joint`:
  - accuracy: `37.72%`
  - top-3: `60.88%`
  - top-5: `72.60%`
  - macro-F1: `37.01%`
- `basd_differential`:
  - accuracy: `31.95%`
  - top-3: `51.79%`
  - top-5: `62.77%`
  - macro-F1: `29.39%` 

Takeaway:

- The stronger one-shot benchmark suite was worth building.
- `basd_pathology` won the official selection on validation accuracy and macro-F1.
- `basd_joint` was very close, so the one-shot story is stable: the improvement came from moving away from differential-only training, not from random noise.

## 2. Sequential Single-Agent Results

Artifacts:

- [sequential metrics](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent/single_agent_live_test_5perclass_max3/metrics.json)
- [sequential predictions](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent/single_agent_live_test_5perclass_max3/predictions.csv)
- [sequential traces](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent/single_agent_live_test_5perclass_max3/traces.jsonl)
- [sequential config](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent/single_agent_live_test_5perclass_max3/resolved_run_config.json)

Reported live-run numbers on the 245-case benchmark:

- model: `gpt-4.1-mini`
- accuracy: `4.49%`
- top-3: `8.98%`
- top-5: `14.69%`
- macro-F1: `3.48%`
- mean requests: `2.96 / 3`
- stop-before-cap rate: `3.67%`
- total input tokens: `3,264,910`
- total output tokens: `63,926`
- runtime: about `28.0` minutes

Behaviorally, the run collapsed:

- `236 / 245` cases consumed the full request budget
- only `9 / 245` stopped before the cap
- the model overpredicted a few labels:
  - `Viral pharyngitis`: `66`
  - `Stable angina`: `63`
  - `GERD`: `38`
- request selection also collapsed:
  - turn 1: `81.2%` of requests were one of `E_55`, `E_53`, `E_54`
  - turn 2: `75.9%`
  - turn 3: `63.3%`

Those evidence ids are generic pain-related questions, so the agent was not meaningfully adapting its workup to the case.

## 3. Sequential Integrity Problem

This matters.

In the pre-patch version of notebook 02, the repair fallback after a malformed or invalid second response used:

- `episode.differential[0][0]`

That is hidden dataset information. It should never have influenced the model prediction path.

This affected `9` sequential cases with the error:

- `requested_evidence_id is not currently available`

Of the `11` total sequentially correct cases:

- `4` came from those contaminated fallback cases
- only `7` were clean

So the current saved sequential artifact is actually a little optimistic. If those error cases are conservatively treated as wrong, the sequential accuracy drops from:

- reported: `4.49%`
- conservative decontaminated estimate: `2.86%`

On the non-error subset only, accuracy is:

- `2.97%`

I patched [02_single_agent_sequential_baseline.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/02_single_agent_sequential_baseline.ipynb) so that fallback no longer uses hidden differential information. The live sequential run and the comparison notebook should be rerun before using final numbers in a writeup.

## 4. Paired Comparison

Artifacts:

- [comparison summary](/Users/bilalawan/claw/assignments/baseline_model/artifacts/comparisons/basd_pathology_full__vs__single_agent_live_test_5perclass_max3/summary_metrics.json)
- [paired case results](/Users/bilalawan/claw/assignments/baseline_model/artifacts/comparisons/basd_pathology_full__vs__single_agent_live_test_5perclass_max3/paired_case_results.csv)
- [pathology delta](/Users/bilalawan/claw/assignments/baseline_model/artifacts/comparisons/basd_pathology_full__vs__single_agent_live_test_5perclass_max3/pathology_delta.csv)

Current paired comparison on the same 245 cases:

- one-shot `basd_pathology`:
  - accuracy: `33.06%`
  - top-3: `55.51%`
  - top-5: `71.43%`
  - macro-F1: `31.79%`
- sequential single-agent:
  - accuracy: `4.49%` raw
  - top-3: `8.98%`
  - top-5: `14.69%`
  - macro-F1: `3.48%`

Win/loss table:

- both correct: `4`
- sequential only correct: `7`
- one-shot only correct: `77`
- both wrong: `157`

Important qualifier:

- `4` of the `7` sequential-only wins came from the contaminated fallback cases
- so the sequential side is even weaker than the saved comparison suggests

One extra nuance:

- on this specific 245-case paired sample, `basd_joint` actually edges out `basd_pathology` slightly in top-1 accuracy (`34.29%` vs `33.06%`)
- that does not change the main conclusion because the sequential run is still dramatically worse than either strong one-shot candidate

## 5. What Failed

The current failure is not subtle. It is not a “maybe the numbers are noisy” situation.

The sequential baseline failed for structural reasons:

- the action space is too large and too unstructured
  - the agent sees essentially the whole remaining DDXPlus root evidence catalog each turn
- the model learned a generic request pattern instead of case-specific workup
  - mostly cycling through generic pain questions
- the stop behavior is weak
  - it almost never decides it has enough evidence
- the prompt is asking a small general-purpose model to do planning, diagnosis, tool use, and schema control all at once
- there are no demonstrations, no learned policy, no action shortlist, and no scoring signal for “informative next question”

So the result is not “agentic medicine is bad.”

The result is:

> This particular naive single-agent prompting setup is bad.

## 6. Is The Project Failing?

If the intended claim is:

> our current agentic baseline beats the strong one-shot classifier

then yes, the current evidence does not support that. On the current run, it fails clearly.

If the intended claim becomes:

> naive sequential LLM workup performs badly under a large medical action space, which motivates structured coordination and evidence-gated control

then the project is still coherent.

That is not just spin. The current results actually support that story.

The present state of the project is:

- strong one-shot baseline: successful
- naive sequential baseline: failed
- evidence-ledger / structured agentic method: not tested yet

So the project is not dead, but the claim has to be disciplined.

## 7. What I Would Do Next

Minimum responsible next step:

1. Rerun notebook 02 and notebook 03 after the fallback patch.

If you want one last serious attempt to rescue the agentic side:

1. Keep the one-shot baseline exactly as is.
2. Constrain the sequential action space before each turn.
3. Give the agent a shortlist of maybe `10-20` candidate evidence requests instead of the full remaining catalog.
4. Prefer a stronger model for the sequential baseline if you have budget.
5. Only then decide whether the agentic direction is worth continuing.

If you do **not** want to keep pushing the agentic side, the safest project narrative is:

> We built and evaluated a strong DDXPlus one-shot baseline and a naive single-agent sequential workup baseline. The sequential baseline underperformed badly, revealing that medical evidence-gathering agents need stronger action control than free-form prompting over a large diagnostic action space.

That is a negative result, but it is a coherent one.
