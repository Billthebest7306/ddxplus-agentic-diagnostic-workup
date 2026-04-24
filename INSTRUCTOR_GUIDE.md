# Instructor Guide: DDXPlus Baselines And Agentic Diagnostic Workup Plan

## 1. Project Summary

This project is about building a **medical diagnostic workup copilot** that starts from incomplete patient evidence, gathers additional information over time, and improves diagnostic reasoning in a structured, auditable way.

The long-term project direction is:

- **main idea**: Multi-Agent Diagnostic Workup Copilot
- **dataset**: DDXPlus
- **core mechanism**: iterative evidence gathering with a structured evidence ledger
- **research question**: can structured, agentic evidence gathering outperform strong non-agentic baselines on diagnostic prediction?

The current work has focused on building the correct baseline ladder before making any claim about the final agentic system.

That ladder is:

1. a strong **one-shot** diagnostic baseline
2. a **single-agent sequential** workup baseline
3. later, a true **multi-agent** evidence-ledger system

This order matters because if we eventually claim that an agentic method is better, we need to compare it against serious baselines rather than weak strawman models.

## 2. Why This Problem Matters

Real diagnostic work is not one-shot. A clinician rarely sees one symptom and immediately commits to a diagnosis. Instead, they:

- start with partial information
- ask follow-up questions
- gather discriminative evidence
- narrow the differential diagnosis
- then finalize a conclusion

That makes this a good setting for an agentic system. If we can model diagnostic workup as a sequence of evidence requests and updates, then the system can become:

- more **interactive**
- more **clinically interpretable**
- more **auditable**
- more **extendable** than a single static classifier

The value of the project is not only prediction accuracy. It is also the structure:

- what evidence was known at each step
- what evidence was requested
- what changed the diagnosis
- how the final conclusion was reached

That is why the project is built around an **evidence ledger**, not just a label predictor.

## 3. Why DDXPlus Is The Right Dataset

DDXPlus is a strong fit because it naturally supports staged workup.

It provides:

- large-scale structured synthetic patient cases
- `49` pathologies
- evidence metadata with question text and value meanings
- an explicit `INITIAL_EVIDENCE` field
- structured evidence fields that can be hidden and revealed

This makes DDXPlus useful for both:

- standard one-shot diagnosis
- sequential diagnostic workup with controlled evidence reveal

It is therefore a much better fit for this project than a static question-answer benchmark alone.

## 4. What We Have Planned

The project is being developed in phases.

### Phase 1. Strong One-Shot Baseline

Question:

> How well can a serious static model diagnose the pathology from only the initially visible evidence?

This gives us the strongest non-agentic reference point.

### Phase 2. Single-Agent Sequential Baseline

Question:

> If the system is allowed to request additional evidence over a few turns, can a single agent outperform the one-shot baseline?

This is the fair pre-agentic comparison point. It tests whether sequential evidence gathering helps even before introducing multiple agents.

### Phase 3. Multi-Agent Diagnostic Workup

Question:

> If multiple specialized agents coordinate through an evidence ledger, can they outperform a single sequential agent?

This is the actual target of the project. It is not implemented yet, but the environment and evaluation structure are being built to support it.

## 5. Single-Agent Vs Multi-Agent

### Single-Agent

A single-agent setup means:

- one LLM sees the current case state
- it decides what question to ask next or when to stop
- it updates its own differential diagnosis over time

This is useful as a baseline because it already tests iterative workup without the added complexity of coordination.

### Multi-Agent

A multi-agent setup would mean:

- multiple specialized agents with different responsibilities
- examples could include:
  - a **triage/workup planner**
  - a **diagnostic synthesizer**
  - a **skeptic or auditor**
  - a **ledger manager** or evidence verifier
- all agents would work against the same evolving evidence ledger

The multi-agent system is the real research contribution we want to build toward. But it only becomes meaningful if we first understand how well a strong one-shot model and a single sequential agent perform.

## 6. What Technical Work We Bring To The Table

This project is not just a prompt demo. The work so far includes substantial technical infrastructure:

- a notebook-first experimental suite
- a reproducible DDXPlus downloader and file validation script
- a **case-episode compiler** that turns each patient into an episode with:
  - visible evidence
  - hidden evidence
  - requestable evidence fields
  - an evidence ledger
- BASD-style structured observation encoding for one-shot modeling
- a train/evaluate pipeline for multiple one-shot objectives
- a sequential diagnostic environment with controlled reveal of hidden evidence
- artifact writing for metrics, predictions, traces, and comparisons
- paired evaluation so one-shot and sequential methods can be compared on the exact same cases

So even though the multi-agent method is not built yet, the project already has a real experimental and engineering backbone.

## 7. Why The Baseline Model Is Necessary

The baseline model is necessary for both the course requirements and the research logic.

### Course Requirement

The assignment requires at least one faithful, reproducible baseline on the chosen dataset. That means we must have:

- a working model
- clean implementation
- reproducible runs
- saved outputs
- evaluation metrics

### Research Requirement

If the final paper or presentation wants to claim that an agentic method helps, then we must answer:

> Better than what?

That “what” cannot be a weak toy baseline. It should be a serious comparator.

That is why we first built a strong one-shot baseline instead of jumping straight to agents.

## 8. What “One-Shot” Means

In this project, **one-shot diagnosis** means:

- the model gets only the initially visible patient clues
- it does **not** ask follow-up questions
- it makes a single direct prediction of the pathology

This is intentionally hard. It reflects the setting:

- no iterative evidence gathering
- no tool use
- no coordination
- just one pass from sparse initial evidence to final diagnosis

That makes it the cleanest non-agentic benchmark.

## 9. What We Built For The One-Shot Baseline

The one-shot notebook is:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)

It evaluates three related one-shot models:

- `basd_pathology`
- `basd_differential`
- `basd_joint`

### Input Representation

The model uses:

- age encoded into `8` bins
- sex encoded as `2` dimensions
- official DDXPlus/BASD-style structured evidence slots

Evidence is represented using the structured observation scheme from DDXPlus metadata:

- binary evidence: `1` slot
- categorical evidence: one or more slots depending on type
- multi-choice evidence: one slot per possible value

State values follow the official structured-style convention:

- unknown: `0`
- present: `1`
- absent: `-1`

Only the root group corresponding to `INITIAL_EVIDENCE` is made visible in the one-shot setup.

### Architecture

The one-shot model is a multilayer perceptron with:

- hidden sizes: `[2048, 2048, 2048]`
- ReLU activations
- configurable dropout
- final output over `49` pathologies

This is technically meaningful for the course because it is a real neural model using a structured medical evidence representation, not a trivial classical baseline.

### Objective Variants

- `basd_pathology`: standard cross-entropy on the gold pathology
- `basd_differential`: soft cross-entropy using the DDXPlus differential distribution
- `basd_joint`: pathology loss plus weighted differential supervision

### Purpose

This one-shot setup tells us:

- how far we can get with only the initial evidence
- what a strong static baseline looks like
- what level the future sequential and agentic systems must beat

## 10. One-Shot Results

The strongest full-split one-shot results are:

- `basd_pathology`
  - accuracy: `37.82%`
  - top-3: `61.55%`
  - top-5: `73.03%`
  - macro-F1: `37.30%`
- `basd_joint`
  - accuracy: `37.72%`
  - top-3: `60.88%`
  - top-5: `72.60%`
  - macro-F1: `37.01%`
- `basd_differential`
  - accuracy: `31.95%`
  - top-3: `51.79%`
  - top-5: `62.77%`
  - macro-F1: `29.39%`

The selected official comparator is:

- `basd_pathology_full`

These are solid results for a `49`-class diagnosis task using only sparse initial evidence.

The one-shot baseline is therefore:

- valid
- reproducible
- meaningful
- strong enough to serve as a serious comparator

## 11. Why The One-Shot Baseline Matters To The Agentic Story

The one-shot baseline gives us a clear lower-level question:

> What can a strong static model do before we give the system any ability to gather more evidence?

That matters because if a future agentic system cannot beat this, then the extra complexity may not be justified.

So the one-shot model is not a side exercise. It is the anchor point for the entire project.

## 12. The One-Shot Limitations

The one-shot model has an obvious limitation:

- it cannot ask for more evidence

So even a strong one-shot result does not solve the broader problem we care about. It cannot:

- reduce uncertainty by targeted questioning
- adapt its workup to the case
- record why a later piece of evidence changed the diagnosis

That limitation is exactly why the sequential and later multi-agent stages matter.

## 13. What The Single-Agent Sequential Setup Is

The sequential notebook is:

- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)

Its setup is:

1. start with:
   - age
   - sex
   - the initial visible evidence only
2. show a decoded evidence ledger in English
3. allow the agent to either:
   - request one more evidence field
   - or stop and finalize the diagnosis
4. reveal the requested evidence through the environment
5. repeat for up to `3` requests

The agent returns structured JSON each turn containing:

- a decision: `request` or `stop`
- the requested evidence id if applicable
- the predicted pathology
- a ranked differential
- a confidence score
- a short explanation

This makes the system behave like a simple diagnostic workup agent rather than a static classifier.

## 14. Why The Single-Agent Baseline Matters

This baseline is important because it is the fairest bridge between one-shot and multi-agent systems.

It asks:

> If the model is allowed to gather more evidence, does that help enough even without multi-agent coordination?

If the answer is yes, then a later multi-agent setup has a strong foundation.

If the answer is no, then we need to understand whether the problem is:

- the model
- the action space
- the prompt design
- the environment interface
- or the overall idea itself

## 15. The Sequential Problems We Initially Hit

The first sequential implementation performed extremely badly, and the result was not trustworthy.

The main issues were:

- the LLM was initially seeing token-like codes such as `E_66` and `V_20` instead of readable clinical findings
- dependent follow-up questions were available even when the parent finding was absent
- there was also a repair fallback bug that could leak hidden differential information on some error cases

That meant the initial poor result was partly caused by a bad interface, not just a weak idea.

## 16. How We Improved The Sequential Setup

We patched the sequential notebook so that:

- visible evidence is decoded into readable English
- revealed values are decoded into human-readable labels
- dependent follow-up questions are gated using DDXPlus metadata
- the agent is told to reason over the decoded clinical ledger, not raw token ids
- the repair fallback no longer uses hidden label information

This was an important methodological fix because it made the environment much more faithful to how an LLM should interact with medical evidence.

## 17. Current Sequential Results

The newer `gpt-5.4-mini` pilot on a balanced `49`-case test sample produced:

- accuracy: `24.49%`
- top-3: `36.73%`
- top-5: `48.98%`
- macro-F1: `15.95%`
- mean requests: `1.92`
- stop-before-cap rate: `69.39%`

This is much more believable than the earlier broken setup.

It also shows some genuine sequential-only wins on cases such as:

- Panic attack
- Epiglottitis
- Bronchitis
- Pulmonary embolism
- Influenza

So the sequential setup is now behaving like a real diagnostic workup system rather than a broken prompt loop.

## 18. Current Sequential Limitation

Even after the improvements, the sequential single-agent baseline still trails the one-shot baseline on the same 49-case benchmark:

- one-shot top-1 on the paired sample: `38.78%`
- sequential top-1 on the paired sample: `24.49%`

This means the project is **not** yet at the point where we can claim:

> sequential agentic-style workup is better than the strong one-shot model

But it also does **not** mean the project has failed.

What it means is:

- the sequential baseline is no longer broken
- the model can use iterative evidence gathering in some cases
- the current policy is still weaker than the static one-shot model

The main bottleneck now appears to be policy quality rather than data access.

## 19. Why The Sequential Baseline Still Underperforms

The current sequential policy is still fairly naive.

Right now it does:

- prompt-only planning
- over a still fairly large action space
- with no learned question policy
- no shortlist of the most informative next questions
- no one-shot prior to focus the workup

So the agent still has to do too much:

- manage diagnosis
- choose good follow-up questions
- obey schema rules
- decide when to stop

all inside one prompt loop.

That is likely why it still struggles on the harder cases.

## 20. How We Plan To Improve The Single-Agent Level

The next planned improvement is to make the sequential policy more focused.

The most promising direction is:

### Use The One-Shot Model As A Prior

Idea:

1. run the one-shot model on the current visible evidence
2. get a top-k candidate differential
3. score which remaining evidence questions are most informative for separating those candidates
4. only expose a short candidate list of next questions to the LLM

This would help because the LLM would no longer choose from the full remaining catalog. Instead, it would choose from a clinically plausible shortlist.

That should improve:

- action quality
- sample efficiency
- API cost
- stop behavior
- final accuracy

Other possible improvements include:

- better demonstrations or exemplars in the prompt
- stronger stop-control prompting
- a stronger or larger reasoning model if needed
- value-of-information scoring for candidate questions

## 21. What We Will Do For The Actual Agentic Work

Once the single-agent sequential baseline is stronger, the actual agentic work will focus on multi-agent coordination through the evidence ledger.

The likely design direction is:

- a planner agent proposes which evidence is most valuable next
- a diagnostic agent updates the differential diagnosis
- a skeptic or auditor challenges weak conclusions or missing evidence
- all agents share a structured evidence ledger

The ledger would track:

- what evidence is visible
- what was requested
- what was absent
- what supports each diagnosis
- what contradicts each diagnosis

This is the real novelty we want to study.

The point is not just to have many agents. The point is to test whether **structured coordination** improves diagnostic workup over:

- one-shot classification
- and a single sequential agent

## 22. What We Have Already Accomplished

At this stage, the project already has:

- a strong one-shot baseline suite
- a sequential diagnostic environment
- a case-episode compiler
- an evidence-ledger scaffold
- saved artifacts and evaluation outputs
- a comparison notebook for aligned evaluation

So we are not starting from zero on the agentic side. The scaffolding is already in place.

## 23. What Still Needs To Be Done

The important remaining work is:

1. improve the single-agent sequential policy
2. rerun the sequential benchmark after that improvement
3. scale beyond the 49-case pilot if the new policy looks promising
4. implement a vanilla multi-agent baseline
5. implement the evidence-ledger / evidence-gated multi-agent method
6. compare:
   - one-shot
   - single-agent sequential
   - multi-agent baseline
   - final evidence-ledger variant

## 24. Why The Project Is Still Worth Pursuing

The current results are not a failure of the whole idea. They show something more useful:

- a strong one-shot model is hard to beat
- naive sequential prompting is not enough
- environment design matters enormously
- structured control is likely necessary for medical evidence gathering

That actually supports the motivation for the multi-agent evidence-ledger direction.

In other words, the current results suggest:

> if we want agentic diagnosis to work well, it cannot just be “ask an LLM to pick questions.” It likely needs structured control, better evidence prioritization, and better coordination.

That is a defensible and interesting project direction.

## 25. The End Goal

The end goal is to build and evaluate a system that:

- starts from limited initial evidence
- gathers additional evidence in an informed way
- keeps a structured ledger of what is known and why it matters
- produces a more transparent and hopefully more accurate diagnostic process

The hope is that the final multi-agent system will:

- outperform the one-shot baseline
- outperform a single sequential agent
- provide a more interpretable diagnostic workflow

Even if the gains are modest, the value of the work is in showing whether structured agentic coordination can meaningfully improve diagnostic workup under controlled conditions.

## 26. Current Honest Position For Discussion

The most honest summary to give right now is:

> We have already built a strong and credible one-shot baseline. We also built a sequential single-agent workup environment and learned that naive prompting is not enough. After fixing major interface issues, the sequential results improved meaningfully but still do not beat the one-shot model. The next step is to improve the sequential policy using one-shot priors and better action control, then build the true multi-agent evidence-ledger system on top of that scaffold.

## 27. Key Files

Main notebooks:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)
- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)
- [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb)

Useful persistent context:

- [PROJECT_WORKLOG.md](PROJECT_WORKLOG.md)

Best one-shot artifacts:

- [selected_model.json](artifacts/one_shot/selected_model.json)
- [basd_pathology_full metrics](artifacts/one_shot/basd_pathology_full/metrics.json)

Current sequential pilot:

- [gpt-5.4-mini metrics](artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v3_gpt54mini/metrics.json)
- [gpt-5.4-mini predictions](artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v3_gpt54mini/predictions.csv)
- [gpt-5.4-mini traces](artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v3_gpt54mini/traces.jsonl)

Paired comparison:

- [comparison summary](artifacts/comparisons/basd_pathology_full__vs__single_agent_live_test_1perclass_max3_decoded_gated_v3_gpt54mini/summary_metrics.json)
