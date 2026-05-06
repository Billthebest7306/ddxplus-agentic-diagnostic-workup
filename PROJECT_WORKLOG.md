# Project Worklog And Persistent Context

This file is a persistent working memory for the DDXPlus medical AI project in this folder. It is meant to preserve project intent, implementation choices, experiment history, results, failure analysis, and next-step guidance so future work does not depend on chat context.

## 1. Project Goal

The project direction is:

- **Main concept**: Multi-Agent Diagnostic Workup Copilot
- **Primary dataset**: DDXPlus
- **Core idea**: iterative diagnostic evidence gathering with a structured evidence ledger
- **Course framing**: Deep Learning project with a reproducible baseline pipeline

The project should not just be a static classifier. The intended final story is:

1. build a strong one-shot baseline
2. build a single-agent sequential workup baseline
3. compare them fairly
4. later build the true multi-agent evidence-ledger system

## 2. Assignment And Scope Constraints

What the assignment clearly required:

- at least one faithful baseline model on the chosen dataset
- reproducible notebook and implementation pipeline
- modular enough to extend
- saved outputs and metrics
- technically meaningful for a deep learning course

What we explicitly decided:

- DDXPlus is the main dataset
- MedQA is not used in the current implementation pass
- the one-shot baseline should be serious, not just a trivial classical model
- the project should be notebook-first for submission
- only the DDXPlus download helper remains as a standalone script

## 3. External Project Docs Used As Guidance

The implementation was aligned to these project-review documents:

- `/Users/bilalawan/claw/output/critical-thinker/medical-multi-agent-project-review/00-executive-summary.md`
- `/Users/bilalawan/claw/output/critical-thinker/medical-multi-agent-project-review/06-refined-best-current-direction.md`
- `/Users/bilalawan/claw/output/critical-thinker/medical-multi-agent-project-review/10-detailed-project-overview.md`
- `/Users/bilalawan/claw/output/critical-thinker/medical-multi-agent-project-review/11-dataset-strategy.md`

Important ideas taken from them:

- DDXPlus is the right environment for staged evidence reveal and iterative workup
- the baseline ladder should be:
  1. one-shot direct diagnosis
  2. single-agent sequential workup
  3. vanilla multi-agent workup
  4. evidence-ledger / evidence-gated variant
- the eventual novelty claim should be about structured evidence-ledger coordination, not just “we used multiple agents”

## 4. Current Project Structure

Main notebooks:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)
- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)
- [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb)

Helper script:

- [download_ddxplus.py](scripts/download_ddxplus.py)

Reports:

- [baseline_summary.md](reports/baselines/baseline_summary.md)
- [baseline_results_and_next_steps.md](reports/baselines/baseline_results_and_next_steps.md)
- [results_assessment.md](reports/baselines/results_assessment.md)
- [sequential_api_guide.md](reports/baselines/sequential_api_guide.md)

Artifact layout:

- `artifacts/one_shot/`
- `artifacts/sequential_single_agent/`
- `artifacts/comparisons/`
- `artifacts/_legacy/`

Dataset location:

- default local repo path: `dataset/`
- machine-specific override: `DDXPLUS_DATASET_DIR`
- legacy fallback still supported: `.data/ddxplus/22687585/`

## 5. DDXPlus Dataset Facts Used In The Project

From the official English DDXPlus release:

- about `1.3M` patients
- `49` pathologies
- `223` root evidence items
- official train / validate / test patient files
- structured evidence metadata in `release_evidences.json`
- pathology metadata in `release_conditions.json`

Important implementation fact:

- the patient zip archives do **not** contain filenames ending in `.csv`
- each zip has a single member like `release_train_patients`
- the notebook loader was patched to use the first non-directory archive member when no `.csv` suffix exists

## 6. High-Level Implementation History

### Phase A. Initial one-shot baseline

We first built a self-contained one-shot direct-diagnosis notebook using:

- age + sex
- only the `INITIAL_EVIDENCE` root evidence group
- BASD-style slot encoding
- MLP classifier
- saved metrics and predictions

This established that the project had a working faithful baseline.

### Phase B. Notebook-first restructure

The project was restructured so the **notebook itself** contains the core pipeline logic instead of hiding it in a `src/` package. That was done because the submission artifact should be readable and runnable top-to-bottom.

### Phase C. Stronger baseline suite

The project was then reorganized into three notebooks:

1. stronger one-shot classifier benchmark suite
2. single-agent sequential baseline
3. comparison notebook

This was done because a single weak baseline would not be a fair comparator for the future agentic system.

## 7. Notebook 01 Design

Notebook 01 implements the stronger one-shot suite.

Input representation:

- 8-bin age encoding
- 2-dim sex encoding
- BASD-style slot state from official DDXPlus evidence metadata
- only the root group corresponding to `INITIAL_EVIDENCE` is visible

Candidate objectives:

- `basd_pathology`
- `basd_differential`
- `basd_joint`

Architecture:

- MLP with hidden sizes `[2048, 2048, 2048]`
- ReLU
- configurable dropout

Selection rule:

- validation top-1 accuracy is the primary selector
- validation macro-F1 is the tie-breaker

Artifacts written per run:

- `metrics.json`
- `predictions.csv`
- `qualitative_examples.json`
- `confusion_summary.csv`
- `training_history.json`
- `best_model.pt`
- `resolved_run_config.json`

Important output contract:

- `case_id` is preserved as `"<split>:<source_row_index>"`
- this is used to align notebook 01 and notebook 02 outputs in notebook 03

## 8. Notebook 01 Results

Full-run one-shot results under `artifacts/one_shot/`:

### `basd_pathology_full`

- accuracy: `0.3782232827122777`
- top-3: `0.6154583770042147`
- top-5: `0.7303258033583837`
- macro-F1: `0.37300761300571456`

### `basd_joint_full`

- accuracy: `0.37723464829144643`
- top-3: `0.6087758029867166`
- top-5: `0.7260293319655985`
- macro-F1: `0.37013660066372966`

### `basd_differential_full`

- accuracy: `0.3194924514416966`
- top-3: `0.5179329363928967`
- top-5: `0.6277308238372395`
- macro-F1: `0.29389682871462686`

Selection result:

- official selected one-shot comparator: `basd_pathology_full`

Important nuance:

- on some paired subsets, `basd_joint_full` can slightly edge out `basd_pathology_full`
- but `basd_pathology_full` won the official full validation-based selection

## 9. Notebook 02 Original Design

Notebook 02 was intended to implement a single-agent sequential diagnostic workup baseline.

Original intended behavior:

- start with demographics + initial evidence only
- allow up to `MAX_REQUESTS = 3`
- expose DDXPlus evidence questions as actions
- return structured JSON:
  - `decision`
  - `requested_evidence_id`
  - `predicted_pathology`
  - `ranked_differential`
  - `confidence`
  - `brief_reasoning`
- dry-run preview mode
- live API mode
- resume-safe run behavior

This notebook uses an OpenAI-compatible chat-completions interface by default, but the provider can be changed as long as the API contract is compatible.

## 10. First Sequential Failure

The first live sequential run used:

- model: `gpt-4.1-mini`
- benchmark: `5` cases per class = `245` test cases
- max requests: `3`

Saved artifact:

- `artifacts/sequential_single_agent/single_agent_live_test_5perclass_max3/`

Reported metrics:

- accuracy: `0.044897959183673466`
- top-3: `0.08979591836734693`
- top-5: `0.1469387755102041`
- macro-F1: `0.03482077511538625`
- mean requests: `2.963265306122449`
- stop-before-cap rate: `0.036734693877551024`

This result was clearly too bad to be taken at face value.

## 11. Root Cause Analysis Of The Sequential Failure

The bad sequential result was **not** interpreted as “LLMs are useless for diagnosis.”

The main issues found were:

### A. Opaque evidence representation

The LLM was seeing:

- `Initial evidence token: E_66`
- revealed values like `V_20`, `V_21`, `V_148`

instead of readable clinical content like:

- `shortness of breath -> yes`
- `pain location -> pharynx`

This meant the model was effectively reasoning over codes rather than medicine.

### B. Ungated dependent actions

Notebook 02 originally exposed follow-up questions even when the parent finding was unknown or absent.

Example:

- pain-detail questions such as `E_54`, `E_56`, `E_57`, `E_58`, `E_59`
- could still appear after pain was absent

This contributed to repetitive, low-value request sequences.

### C. Repair fallback contamination bug

On invalid response cases, the notebook used hidden differential information as a fallback prediction source.

This was incorrect because it leaked unavailable label information into the evaluation path.

The bug was removed later.

### D. Notebook 03 auto-discovery bug

Notebook 03 originally picked the lexicographically last sequential artifact folder rather than the most recent one.

This caused it to compare against the stale old sequential run instead of the new patched run.

## 12. Specific Fixes Applied To Notebook 02

Notebook 02 was patched heavily.

### Fix 1. Decode visible evidence into English

The prompt now shows:

- readable initial finding summaries
- readable revealed findings
- readable evidence ledger entries

instead of raw token ids alone.

### Fix 2. Decode revealed values using DDXPlus `value_meaning`

For evidence with categorical or multivalue content, revealed values are decoded from DDXPlus metadata into English descriptions.

### Fix 3. Gate dependent questions using `code_question`

The notebook now uses the DDXPlus parent-question relationship to hide child questions until the parent finding is satisfied.

Example:

- pain detail questions do not show up before the parent pain question is present
- and they stay hidden if the parent finding is absent

### Fix 4. Remove hidden-differential fallback leak

The repair fallback no longer uses hidden label information.

### Fix 5. Scrubbed notebook secrets

At one point, notebook 02 had been saved with a real API key embedded in the config cell and printed in outputs.

This was scrubbed:

- config cell reset to env-based placeholders
- outputs cleared from the notebook
- no secret should be reintroduced into the file

### Fix 6. Fresh run versioning

A run version suffix was added so patched reruns do not resume stale broken artifacts.

Example:

- `single_agent_live_test_1perclass_max3_decoded_gated_v2`

### Fix 7. Prompt guidance improved

The system prompt and user prompt now explicitly say:

- the ledger is decoded into English
- use decoded findings, not raw token ids, for reasoning
- prefer stopping over low-value generic questioning once evidence is informative

## 13. Notebook 02 After Patch: 49-Case Pilot

Patched pilot run artifact:

- `artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v2/`

Settings:

- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- total cases = `49`
- model = `gpt-4.1-mini`

Metrics:

- accuracy: `0.14285714285714285`
- top-3: `0.3469387755102041`
- top-5: `0.4489795918367347`
- macro-F1: `0.09562682215743439`
- mean requests: `2.877551020408163`
- stop-before-cap rate: `0.061224489795918366`

This is still weak, but it is **far more credible** than the earlier broken run.

Interpretation:

- the patched interface improved behavior materially
- the model is no longer acting completely blind
- some traces now look medically sensible
- the agent is still too weak / too inefficient to beat the strong one-shot baseline

## 14. Examples Of Improved Sequential Behavior

The patched sequential run produced some genuine sequential-only wins on the 49-case pilot.

Examples:

- `Pulmonary embolism`
- `Boerhaave`
- `Acute rhinosinusitis`
- `Panic attack`

Example sensible trace:

- ask about fever -> no
- ask about pleuritic pain -> yes
- stop with `Pulmonary embolism`

This kind of trace is much more aligned with project goals than the original token-chasing behavior.

## 15. Comparison Against One-Shot On The Same 49 Cases

On the exact same 49 pilot cases:

### One-shot

- `basd_pathology_full`: `0.3878` accuracy
- `basd_joint_full`: `0.4286` accuracy
- `basd_differential_full`: `0.3265` accuracy

### Patched sequential

- `gpt-4.1-mini` patched pilot: `0.14285714285714285` accuracy

So the sequential baseline improved a lot but is still clearly behind the one-shot baseline.

## 16. Win/Loss View On The 49-Case Pilot

Using `basd_pathology_full` as the one-shot comparator on the same 49 pilot cases:

- both correct: `3`
- sequential-only correct: `4`
- one-shot-only correct: `16`
- both wrong: `26`

This means:

- the sequential baseline is no longer dead-on-arrival
- but it is still not competitive

## 17. Notebook 03 Fix

Notebook 03 originally auto-selected the wrong sequential artifact because it sorted folder names lexicographically.

This was patched so it now:

- prefers the most recently modified sequential artifact by default
- still allows explicit override through `SEQUENTIAL_RUN_NAME`

Important practical tip:

If there is any ambiguity, set this manually in notebook 03:

```python
SEQUENTIAL_RUN_NAME = "single_agent_live_test_1perclass_max3_decoded_gated_v2"
```

That guarantees the comparison notebook uses the intended sequential run.

## 18. Current Interpretation Of Project Health

### What is working

- the one-shot benchmark suite is strong and credible
- notebook 01 is in good shape
- the DDXPlus case-episode representation is useful
- the patched sequential notebook now produces intelligible LLM behavior
- the project is no longer “obviously broken”

### What is not working yet

- the sequential single-agent baseline is still too weak
- stop behavior is poor
- the model still uses most of the available request budget
- action selection is improved but still not efficient enough
- the strong one-shot model still wins clearly

### Honest state of the project

The project is **not doomed**, but the true agentic claim is **not earned yet**.

Right now the evidence supports:

> a strong one-shot baseline works well, and a naive or lightly structured sequential agent still struggles even after interface fixes

It does **not** yet support:

> the current agentic / sequential system beats the one-shot baseline

## 19. Important Lessons Learned

### Lesson 1

The LLM-facing environment representation matters enormously. Giving an LLM token ids instead of readable clinical findings can make a reasonable method look catastrophic.

### Lesson 2

Large open action spaces are hard. Exposing the full remaining DDXPlus catalog is still probably too much, even after dependency gating.

### Lesson 3

A strong one-shot baseline can be surprisingly hard to beat. The selected MLP baseline is not a toy.

### Lesson 4

Evaluation plumbing bugs can meaningfully distort conclusions. We already saw this with:

- hidden-differential fallback contamination
- wrong comparison artifact selection

## 20. What To Do Next

If continuing the sequential / agentic direction:

1. test a stronger small model on the **same 49-case pilot**
2. keep the patched notebook 02 interface
3. use a fresh `RUN_VERSION`
4. rerun notebook 03 after each sequential pilot

The current planned next test is:

- same patched notebook 02
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- switch from `gpt-4.1-mini` to `gpt-5.4-mini`
- new run version string

Why this test is reasonable:

- the interface is now much more faithful to the intended task
- remaining weakness may now be backbone capability plus still-too-large action space

If the stronger model still fails badly:

- next step should be action shortlisting before each turn
- do **not** just keep burning API budget on the same setup

If the stronger model improves meaningfully:

- then the agentic side is still viable and worth scaling to a larger sampled benchmark

## 21. Practical Rerun Guidance

### Notebook 02

Safe pattern:

- keep `SEQUENTIAL_SAMPLE_PER_CLASS = 1` for pilot runs
- use a unique `RUN_VERSION`
- keep `RESUME_IF_AVAILABLE = True`

Reason:

- changing `RUN_VERSION` creates a fresh artifact directory
- otherwise the resume logic may skip already completed `case_id`s

### Notebook 03

Either:

- let it auto-pick the most recent sequential artifact

or explicitly set:

```python
SEQUENTIAL_RUN_NAME = "your_exact_run_name_here"
```

## 22. Security / Hygiene Note

Never hardcode real API keys into notebook cells.

Notebook 02 should stay on:

- `LLM_BASE_URL = os.environ.get(...)`
- `LLM_API_KEY = os.environ.get(...)`
- `LLM_MODEL = os.environ.get(...)`

and notebook outputs should not be saved with secrets printed.

## 23. Files That Matter Most Right Now

Main work files:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)
- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)
- [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb)

Best one-shot artifact:

- [selected_model.json](artifacts/one_shot/selected_model.json)
- [basd_pathology_full metrics](artifacts/one_shot/basd_pathology_full/metrics.json)

Patched sequential pilot artifact:

- [decoded_gated_v2 metrics](artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v2/metrics.json)
- [decoded_gated_v2 predictions](artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v2/predictions.csv)
- [decoded_gated_v2 traces](artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v2/traces.jsonl)

Diagnostic reports:

- [results_assessment.md](reports/baselines/results_assessment.md)
- [baseline_results_and_next_steps.md](reports/baselines/baseline_results_and_next_steps.md)

## 24. One-Sentence Summary For Future Context Recovery

We built a strong DDXPlus one-shot baseline suite and a sequential LLM baseline; the first sequential attempt failed because the LLM saw opaque token codes and ungated actions, notebook 02 was patched to decode evidence and gate dependent questions, the patched 49-case sequential pilot improved from about `4.5%` to about `14.3%` accuracy but still trails the strong one-shot baseline, and the next rational test is a stronger model on the same patched pilot before spending more budget.

## 25. GPT-5.4-Mini Sequential Pilot

After the patched `gpt-4.1-mini` pilot, we ran the same 49-case benchmark with `gpt-5.4-mini` using a fresh run version:

- sequential artifact: `artifacts/sequential_single_agent/single_agent_live_test_1perclass_max3_decoded_gated_v3_gpt54mini/`
- comparison artifact: `artifacts/comparisons/basd_pathology_full__vs__single_agent_live_test_1perclass_max3_decoded_gated_v3_gpt54mini/`

### Sequential Metrics

- accuracy: `0.24489795918367346`
- top-3: `0.3673469387755102`
- top-5: `0.4897959183673469`
- macro-F1: `0.1595238095238095`
- mean requests: `1.9183673469387754`
- stop-before-cap rate: `0.6938775510204082`
- mean API calls: `2.673469387755102`
- runtime: `234.5s`
- token usage: `563,729` input / `13,966` output

### Comparison Against Patched GPT-4.1-Mini Pilot

The move from `gpt-4.1-mini` to `gpt-5.4-mini` was a real improvement, not noise:

- top-1 improved from `0.1429` to `0.2449`
- top-3 improved from `0.3469` to `0.3673`
- top-5 improved from `0.4490` to `0.4898`
- macro-F1 improved from `0.0956` to `0.1595`
- mean requests dropped from `2.88` to `1.92`
- stop-before-cap rate jumped from `0.0612` to `0.6939`

Accuracy by request count in the `gpt-5.4-mini` pilot:

- `0` requests: `0.50`
- `1` request: `0.40`
- `2` requests: `0.2941`
- `3` requests: `0.00`

This suggests the agent is now much better at stopping early when it already has enough signal, but still performs poorly on cases that remain unresolved after three turns.

### Comparison Against One-Shot Baseline On The Same 49 Cases

Using `basd_pathology_full` as the official one-shot comparator:

- one-shot top-1: `0.3877551020408163`
- sequential top-1: `0.24489795918367346`
- one-shot top-3: `0.6530612244897959`
- sequential top-3: `0.3673469387755102`
- one-shot top-5: `0.7755102040816326`
- sequential top-5: `0.4897959183673469`
- one-shot macro-F1: `0.3294460641399417`
- sequential macro-F1: `0.1595238095238095`

Win/loss on the paired 49 cases:

- both correct: `7`
- sequential only correct: `5`
- one-shot only correct: `12`
- both wrong: `25`

Interpretation:

- `gpt-5.4-mini` closed part of the gap, but the sequential baseline still clearly trails the strong one-shot model.
- The project is not dead, but the current single-agent sequential baseline is still not competitive enough to claim superiority.

### Sequential-Only Wins

The `gpt-5.4-mini` sequential run got these right while the one-shot comparator missed them:

- Panic attack
- Epiglottitis
- Bronchitis
- Pulmonary embolism
- Influenza

These are useful because they show that iterative questioning can help on some clinically meaningful cases.

### Cases Improved Vs GPT-4.1-Mini

The `gpt-5.4-mini` run corrected several cases the `gpt-4.1-mini` pilot missed:

- Acute dystonic reactions
- Anemia
- Bronchitis
- Epiglottitis
- Influenza
- Larygospasm
- Spontaneous rib fracture

It regressed on:

- Acute rhinosinusitis
- Boerhaave

### Behavior Changes

The top requested evidence ids were:

- `E_54`: pain characterization
- `E_66`: significant shortness of breath
- `E_91`: fever
- `E_14`: chest pain at rest
- `E_53`: pain somewhere
- `E_75`: choking or suffocating
- `E_65`: difficulty swallowing
- `E_155`: palpitations
- `E_220`: pleuritic pain
- `E_201`: cough

These are at least clinically interpretable and much better than the old degenerate pattern where the model kept fixating on opaque token-coded pain questions.

However, there is still a clear limitation:

- when the model uses all 3 requests, accuracy on this pilot is `0.0`
- so the hard cases are not being resolved well by the current policy
- the current action space is still likely too broad, even after dependency gating

### Current Bottom Line

The patched sequential notebook is now producing believable and actionable results:

- it is no longer obviously broken
- the stronger model helps materially
- the sequential baseline now shows genuine sequential-only wins

But the present state is still:

- strong one-shot baseline
- partially working sequential baseline
- no evidence yet that the sequential baseline beats the one-shot baseline

### Best Next Moves

If continuing the sequential / agentic direction, the next high-value improvements are:

1. add action shortlisting so the agent chooses from a smaller clinically plausible set each turn
2. optionally seed the sequential agent with the one-shot top-k differential as a prior
3. only then consider scaling the sequential evaluation beyond the 49-case pilot

Avoid jumping straight to expensive larger API runs without changing the policy, because the current traces already show where the remaining bottleneck is.

### Meeting Guide Added

A meeting-ready project summary was added at:

- [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md)

Purpose:

- explain the overall project idea
- justify the need for one-shot and sequential baselines
- summarize the current results
- explain why the multi-agent stage has not been built yet
- provide a clear roadmap for the next technical steps

### Temporary Unrestricted Pilot Config

Notebook 02 was temporarily reconfigured for a cheap unrestricted-capability test:

- `MAX_REQUESTS = 222`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = 10`
- `RUN_VERSION = "unrestricted_v5_gpt41mini_10casepilot"`

Purpose:

- let the sequential agent request essentially all reachable evidence
- but only on a small deterministic 10-case pilot to control API cost

Implementation detail:

- the notebook first builds the standard 49-case `1-per-class` sample
- then deterministically downsamples it to 10 cases
- the run name now includes the case cap so it does not resume the aborted unrestricted run artifacts

### Unrestricted 10-Case GPT-4.1-Mini Pilot Result

Artifact:

- `artifacts/sequential_single_agent/single_agent_live_test_1perclass_max222_10cases_unrestricted_v5_gpt41mini_10casepilot/`

Metrics:

- accuracy: `0.40`
- top-3: `0.50`
- top-5: `0.50`
- macro-F1: `0.0748`
- mean requests: `8.6`
- stop-before-cap rate: `1.0`
- mean API calls: `9.7`
- total tokens: `422,371` input / `10,136` output

Interpretation:

- letting `gpt-4.1-mini` request much more evidence materially improved it versus the capped 3-turn version
- on the same 10 cases:
  - unrestricted `gpt-4.1-mini`: `0.40`
  - capped `gpt-4.1-mini` (`v2`) on same 10 cases: `0.20`
  - capped `gpt-5.4-mini` (`v3`) on same 10 cases: `0.30`
- this suggests the 3-turn budget was a real bottleneck for `gpt-4.1-mini`

But:

- the unrestricted sequential result still did not beat the one-shot baseline on the same 10 cases
- one-shot `basd_pathology_full` reached `0.50` on that slice
- win/loss on the 10 paired cases was:
  - both correct: `2`
  - sequential only correct: `2`
  - one-shot only correct: `3`
  - both wrong: `3`

Behavioral note:

- the model now shows real value from extended questioning on some cases, e.g. `Pneumonia`
- but it is still inefficient, sometimes using `9+` requests on cases that a strong policy should resolve much faster
- some obvious failures remain, e.g. `Croup -> Whooping cough`, `Tuberculosis -> Pulmonary embolism`

Current conclusion:

- unrestricted evidence access helps the LLM
- so diagnosis quality is not purely capped by model ability; turn budget matters
- however, the current sequential policy is still not efficient or accurate enough to outperform the strong one-shot baseline

## 26. Proposed Improvement 1 Notebook

A new notebook was added:

- [04_single_agent_structured_policy_improvement.ipynb](notebooks/04_single_agent_structured_policy_improvement.ipynb)

Purpose:

- keep the system **single-agent**
- improve the sequential policy without jumping yet to deeper algorithmic ledger methods
- make the workup more structured, stateful, and controlled than notebook 02

Main additions:

- deterministic evidence ledger / state manager as the episode source of truth
- decoded evidence and value rendering only; no token-id-only reasoning
- legal-action handling with parent-child gating preserved
- deterministic **action shortlisting** each turn
- optional **one-shot prior differential** merged in as advisory context
- deterministic **stop guidance**
- evaluation across **multiple request budgets** in one run
- saved artifacts and plots per budget

Artifact root used by the default dry-run proof of concept:

- `artifacts/sequential_single_agent_improved/single_agent_improved_dryrun_test_1perclass_4budgets_ledger_shortlist_budget_sweep_v1/`

Proof-of-concept validation status:

- the notebook JSON is valid
- all code cells compile
- the notebook was executed end-to-end in dry-run mode
- it produced:
  - per-budget `predictions.csv`, `traces.jsonl`, `metrics.json`
  - paired one-shot comparison files
  - cached shortlist stats
  - six budget-analysis plots

Default dry-run configuration:

- `RUN_LIVE_API = False`
- `ALLOW_DRY_RUN_BENCHMARK = True`
- `REQUEST_BUDGETS = [1, 3, 5, 8]`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = 10`
- `SHORTLIST_SIZE = 12`
- `SHORTLIST_STATS_SOURCE = "validate"`
- `SHORTLIST_STATS_MAX_ROWS = 30000`

Intended next live use:

- keep the same notebook structure
- enable `RUN_LIVE_API = True`
- choose a small deterministic benchmark first
- inspect performance, stop behavior, request usage, and one-shot gap across budgets

### First Live Result For Proposed Improvement 1

Live artifact root:

- `artifacts/sequential_single_agent_improved/single_agent_improved_live_test_1perclass_4budgets_ledger_shortlist_budget_sweep_v1/`

Budget sweep on the 10-case live sample:

- budget `1`: accuracy `0.30`
- budget `3`: accuracy `0.20`
- budget `5`: accuracy `0.10`
- budget `8`: accuracy `0.20`

On the same 10 cases, one-shot accuracy stayed:

- `0.30`

Interpretation:

- the notebook is operationally sound
- the structured ledger / shortlist pipeline works
- but this first structured-policy version did **not** deliver a meaningful empirical improvement
- larger budgets actually introduced drift on several cases rather than helping consistently

Main observed failure mode:

- extra requests often caused the model to move away from an initially plausible diagnosis
- stop behavior became more efficient, but not more correct
- the deterministic shortlist still appears too generic

Report written at:

- [proposed_improvement_1_results.md](reports/baselines/proposed_improvement_1_results.md)

### Sequential Policy Refinement (Notebook 05)

New successor notebook created:

- [05_single_agent_structured_policy_refinement.ipynb](notebooks/05_single_agent_structured_policy_refinement.ipynb)

Reason for creating notebook 05:

- notebook 04 was no longer failing because of opaque evidence access
- the main failure mode had become **diagnostic drift**
- the shortlist and stop logic were still following the model's last differential too closely
- higher budgets often hurt because extra evidence was not being revised into the differential well

Main policy changes in notebook 05:

- deterministic diagnosis-state manager
  - anchors the evolving differential using one-shot priors plus revealed evidence
  - computes top candidates, margin, unresolved mass, and prior strength
- stronger shortlist logic
  - scores questions by how well they separate the current competing diagnoses
  - penalizes generic high-frequency questions more aggressively
  - limits repeated overexposure to the same parent-question family
- policy controller
  - can force a request when the agent tries to stop while the deterministic differential is still unresolved
  - can force a stop when the deterministic state is stable and the remaining shortlist is weak
  - can override drift-heavy diagnosis jumps when the deterministic state is clearly anchored elsewhere
- replay diagnostics
  - replays the earlier live notebook 04 run on the same revealed evidence
  - measures how much the refined diagnosis-state logic would have improved the final predictions without spending more API budget

Notebook 05 default configuration:

- `RUN_LIVE_API = False`
- `ALLOW_DRY_RUN_BENCHMARK = True`
- `RUN_REPLAY_DIAGNOSTICS = True`
- `REQUEST_BUDGETS = [1, 3, 5, 8]`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = 10`
- artifact root family: `artifacts/sequential_single_agent_refined/`

Validation status:

- notebook 05 JSON is valid
- all updated code cells executed end-to-end via `nbconvert`
- the safe env bootstrap cell was patched so headless execution no longer fails on `getpass()`

Notebook 05 artifact root from the executed dry-run:

- `artifacts/sequential_single_agent_refined/single_agent_refined_dryrun_test_1perclass_4budgets_anchor_guard_v1/`

Dry-run sweep results for notebook 05:

- budget `1`: accuracy `0.20`
- budget `3`: accuracy `0.40`
- budget `5`: accuracy `0.70`
- budget `8`: accuracy `0.60`

Interpretation:

- unlike notebook 04, extra evidence is now being used productively by the refined policy logic
- the best offline policy point on this 10-case slice is around budget `5`
- the refined controller is not just cleaner; it materially changes the direction of the budget/performance curve

Replay diagnostics against notebook 04 live traces:

- source budget `1`: `0.30` -> refined state on same revealed evidence: `0.20`
- source budget `3`: `0.20` -> refined state: `0.40`
- source budget `5`: `0.10` -> refined state: `0.50`
- source budget `8`: `0.20` -> refined state: `0.60`

Interpretation of replay:

- the refined state manager is weaker on 1-turn diagnosis because it is less eager to overtrust a thin first clue
- once several turns of evidence are available, it clearly outperforms the earlier live sequential policy on the **same revealed evidence**
- this strongly supports the diagnosis that the main remaining bottleneck was belief revision / drift control, not lack of information access

Current honest project state after notebook 05:

- the sequential system is still not proven live-better than the one-shot baseline
- but we now have strong evidence that the notebook 04 failure was not the final verdict on the idea
- notebook 05 gives a materially stronger controller and a much sharper explanation of what was going wrong

Report written at:

- [sequential_policy_refinement_report.md](reports/hybrid/sequential_policy_refinement_report.md)

### Live Notebook 05 Outcome

The live refined run has now completed:

- [single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1](artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1)

Live budget sweep results on the 10-case sample:

- budget `1`: accuracy `0.20`
- budget `3`: accuracy `0.20`, top-5 `0.80`
- budget `5`: accuracy `0.50`
- budget `8`: accuracy `0.80`

Paired one-shot accuracy on the same 10 cases remained:

- `0.30`

Meaning:

- notebook 05 now clearly beats the one-shot baseline at higher budgets
- the project is no longer in the “sequential looks empirically hopeless” state

Most important comparison against notebook 04:

- notebook 04 budget `5`: `0.10` -> notebook 05 budget `5`: `0.50`
- notebook 04 budget `8`: `0.20` -> notebook 05 budget `8`: `0.80`

This is the strongest evidence so far that the main issue really was sequential policy quality, not the impossibility of the idea.

Sequential-only wins at budget `8`:

- `Chagas`
- `Ebola`
- `Pulmonary embolism`
- `Stable angina`
- `Tuberculosis`

Remaining failures at budget `8`:

- `Croup -> Viral pharyngitis`
- `Pneumonia -> Myasthenia gravis`

Important interpretation:

- gains came mainly from the refined diagnosis-state, shortlist, and prompt behavior
- `drift_override_rate` stayed `0.0`, so the improvement was not just brute-force postprocessing

Updated report with the live notebook 05 results:

- [sequential_policy_refinement_report.md](reports/hybrid/sequential_policy_refinement_report.md)

### Budget Scaling Successor Notebook

Created a minimal successor notebook for the plateau / saturation question:

- [06_single_agent_budget_scaling.ipynb](notebooks/06_single_agent_budget_scaling.ipynb)

Purpose:

- keep notebook 05 policy logic unchanged
- change only the default budget sweep to larger values
- test whether sequential gains continue to improve, plateau, or regress as the request budget increases substantially

Default experiment changes in notebook 06:

- `REQUEST_BUDGETS = [8, 16, 24, 32]`
- `RUN_VERSION = "anchor_guard_budget_scaling_v1"`

Everything else is intentionally kept aligned with notebook 05 so the scaling experiment is a clean continuation rather than a new method.

### Git Handoff / README

Added root repo documentation:

- [README.md](README.md)

Purpose of the README:

- explain the project structure and notebook progression
- identify which notebooks are historical versus current
- document how to run the project
- document artifact layout and experiment hygiene
- make it explicit that every meaningful notebook/result change must also update the worklog

This was added so collaborator handoff is less dependent on chat context.

## 27. Repo Cleanup And Path Standardization

A repo cleanup pass was applied so collaborators on different devices do not need the dataset in the same absolute location.

What changed:

- the default dataset fallback was standardized to `dataset/`
- a machine-local override is now supported via `DDXPLUS_DATASET_DIR`
- the downloader now accepts both `--output-dir` and the older `--dataset-dir` alias, but `--output-dir` is the canonical flag
- notebooks `01`, `02`, `04`, `05`, and `06` were updated to resolve dataset paths in this order:
  1. `DDXPLUS_DATASET_DIR`
  2. `dataset/`
  3. legacy `.data/ddxplus/22687585/`
- `scikit-learn` was added to `requirements.txt` because notebooks `04` to `06` import `sklearn.metrics`
- `dataset/` was added to `.gitignore` so local dataset copies are not accidentally committed

Why this matters:

- the repo now has a single documented default local dataset location
- collaborators can still keep the data elsewhere without editing notebooks
- old notebooks remain backward-compatible with legacy local setups

What did **not** change:

- experiment logic
- prompts
- budgets
- saved artifacts

So this cleanup should not require rerunning historical experiments. It is an environment and reproducibility fix, not a methodological change.

## 28. Ledger Novelty And Multi-Agent Architecture Notes

Two new design reports were added for the instructor-facing discussion of project novelty and next-stage architecture:

- [evidence_ledger_algorithm_and_improvements.md](reports/architecture/evidence_ledger_algorithm_and_improvements.md)
- [proposed_multi_agent_architecture.md](reports/architecture/proposed_multi_agent_architecture.md)

Purpose:

- clarify that the evidence ledger is not strong enough as a novelty claim if presented only as shared memory
- reframe the real method as an evidence-gated differential ledger that constrains requests, diagnosis updates, critique, and stopping
- provide a concrete multi-agent architecture centered on the ledger rather than free-form inter-agent chat

Current recommendation captured in those reports:

- do **not** claim novelty from “multiple agents share memory” alone
- claim novelty from a diagnosis-specific ledger protocol with support/contradiction tracking, validation, and stop control
- use a controller-constrained multi-agent design with planner, synthesizer, critic, optional retriever, and stop agent

An additional design-freeze note was added at:

- [architecture_v1_freeze_and_experimental_scope.md](reports/architecture/architecture_v1_freeze_and_experimental_scope.md)

Purpose:

- define what is fixed in the next implementation cycle
- prevent unnecessary churn in top-level architecture
- make the ledger/control protocol the main experimental variable

## 29. Rigorous Single-Agent Evaluation Phase

The next phase was implemented without moving into deeper algorithmic ledger methods, multi-agent systems, model ablations, probabilistic belief-state modeling, graph inference, RL, or learned policies.

The user explicitly fixed the scope:

- keep the LLM backbone fixed to `gpt-4.1-mini`
- do not run model ablations yet
- make the current refined single-agent system more rigorous, reproducible, fairly evaluated, and evidence-efficient
- work in successor notebooks rather than destructively rewriting historical notebooks

### Deterministic API Fixes

Small reproducibility fixes were applied to notebooks `05`, `06`, and the new `08`:

- default LLM model is now `gpt-4.1-mini`
- `TEMPERATURE = 0.0`
- `TOP_P = 1.0`
- OpenAI-compatible request bodies include both `temperature` and `top_p`
- resolved run configs log deterministic controls
- the secure API bootstrap cell no longer prompts for `getpass()` during non-live execution

This matters because notebook execution should not block during dry runs or CI-style validation, and live runs should not silently use stochastic API settings.

### New Notebook 07: Full-Evidence One-Shot Comparator

Added:

- [07_full_evidence_one_shot_comparator.ipynb](notebooks/07_full_evidence_one_shot_comparator.ipynb)

Purpose:

- train a full-evidence direct diagnosis comparator
- reveal every DDXPlus root evidence field as present/value or absent
- estimate the full-information ceiling available in DDXPlus
- compare against the initial-evidence one-shot baseline

Important fairness rule:

- full-evidence predictions and probabilities are evaluation-only
- they must not be used inside live sequential policy, action selection, stop logic, prompting, or diagnosis updates

Implementation note:

- the full-evidence encoder uses a precomputed all-absent evidence template
- per patient, it copies that template, sets demographics, and applies only present/value roots
- this avoids repeatedly applying all 223 absent root observations per row

Status:

- notebook created
- code cells parse cleanly
- full training has not been executed in this pass

Report:

- [full_evidence_one_shot_comparator.md](reports/baselines/full_evidence_one_shot_comparator.md)

### New Notebook 08: Cost-Sensitive Sequential Lambda Sweep

Added:

- [08_cost_sensitive_sequential_lambda_sweep.ipynb](notebooks/08_cost_sensitive_sequential_lambda_sweep.ipynb)

Purpose:

- replace arbitrary request-budget sweeps with a cost-sensitive evidence acquisition experiment
- use a generous request cap as a safety ceiling
- sweep evidence cost `lambda`
- test whether the policy can stop earlier when more evidence is not worth the cost

Default validation settings:

- `RUN_LIVE_API = False`
- `ALLOW_DRY_RUN_BENCHMARK = True`
- `LLM_MODEL = "gpt-4.1-mini"`
- `TEMPERATURE = 0.0`
- `TOP_P = 1.0`
- `EVIDENCE_COST_LAMBDAS = [0.00, 0.03, 0.06, 0.10, 0.15, 0.22]`
- `MAX_REQUEST_CAP = 24`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = 10`

Policy addition:

- a lightweight deterministic marginal-value estimate for another evidence request
- uses current shortlist score, diagnostic margin, unresolved mass, recent margin gain, stability, and remaining cap
- compares marginal evidence value against `lambda`
- can force stop when value is below cost
- can force request when value clearly exceeds cost and the diagnosis is unresolved

This remains within the current architecture:

- single agent
- deterministic ledger
- decoded evidence
- legal action gating
- one-shot prior anchoring
- shortlist tied to competing diagnoses
- drift/stability controls

Validation:

- notebook 08 executed end-to-end in dry-run mode
- artifact root:
  - `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_dryrun_test_1perclass_cap24_6lambdas_lambda_cost_v1/`
- plots and per-lambda artifacts were written successfully

Important caveat:

- dry-run metrics are contract checks only
- they are not live LLM scientific results

Report:

- [lambda_cost_sensitive_policy_report.md](reports/baselines/lambda_cost_sensitive_policy_report.md)

### New Notebook 09: Matched-Evidence Integrated Comparison

Added:

- [09_matched_evidence_integrated_comparison.ipynb](notebooks/09_matched_evidence_integrated_comparison.ipynb)

Purpose:

- compare initial-evidence one-shot, sequential prediction, matched-evidence one-shot, and full-evidence one-shot on the same cases
- separate the value of evidence acquisition from the value of LLM sequential reasoning

Matched-evidence comparator:

- reads sequential `traces.jsonl`
- reconstructs exactly the evidence roots revealed by the policy
- encodes demographics + initial evidence + revealed fields as a bag-of-evidence state
- does not include turn order, unrevealed evidence, hidden labels, hidden differentials, or full-evidence predictions

Scientific interpretation:

- if sequential beats matched-evidence one-shot, LLM reasoning over acquired evidence is adding value
- if matched-evidence one-shot beats sequential, the sequential system may still be useful as an evidence acquisition controller while a direct classifier handles final diagnosis
- if both trail full-evidence one-shot, the policy is not acquiring enough of the right evidence

Validation:

- notebook 09 executed successfully against the notebook 08 dry-run artifact
- artifact root:
  - `artifacts/integrated_comparisons/single_agent_cost_sensitive_dryrun_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`
- at that time, notebook 07 had not yet been run, so matched/full-evidence columns were `NaN` in the dry-run validation output
- this was later superseded by the final live integrated comparison recorded in section 31

Report:

- [matched_evidence_integrated_comparison_report.md](reports/baselines/matched_evidence_integrated_comparison_report.md)

### Phase Report

Added:

- [phase_next_rigorous_evaluation_plan.md](reports/project/phase_next_rigorous_evaluation_plan.md)

This report summarizes the new phase, current status, dry-run validation, and recommended next live runs.

### Recommended Next Steps

1. Run notebook 07 to create full-evidence one-shot artifacts.
2. Run notebook 08 live on the 10-case pilot with `gpt-4.1-mini`.
3. If results are coherent, run notebook 08 on the 49-case balanced pilot.
4. Rerun notebook 09 after full-evidence and live lambda artifacts exist.
5. Use notebook 09 to decide whether final diagnosis should remain with the LLM or be delegated to a direct classifier after sequential evidence gathering.

## 30. Full-Evidence Deduplication Robustness Check

Notebook 07 was updated with a non-destructive deduplication robustness section.

Reason:

- the full-evidence one-shot model showed extremely high validation accuracy very early
- a code audit found no obvious direct target leakage into the feature vector
- however, exact duplicate rows exist across the official DDXPlus splits

What was added:

- a post-training section named `Deduplicated Robustness Check`
- official train/validate/test metrics remain unchanged
- validation/test rows can be filtered if their signatures appear in training
- two duplicate definitions are reported:
  - `raw_row_signature`: full raw patient-row signature
  - `feature_signature`: `AGE`, `SEX`, `EVIDENCES`, and `INITIAL_EVIDENCE`
- the selected trained checkpoint is reloaded
- metrics are recomputed on filtered validation/test subsets

Artifacts written inside the selected full-evidence run directory:

- `dedup_robustness_summary.csv`
- `dedup_robustness_summary.json`
- `dedup_metrics_validate_raw_row_signature.json`
- `dedup_metrics_validate_feature_signature.json`
- `dedup_metrics_test_raw_row_signature.json`
- `dedup_metrics_test_feature_signature.json`
- matching `dedup_predictions_*` CSV files

Important interpretation:

- official metrics are still reported for comparability with the released split
- deduplicated metrics are a robustness check against cross-split duplicate contamination
- the full-evidence baseline remains a ceiling-style comparator
- full evidence must still not be used inside the live sequential policy

Updated report:

- [full_evidence_one_shot_comparator.md](reports/baselines/full_evidence_one_shot_comparator.md)

## 31. Final Comparator Results Integrated Into Reports

The final available artifacts for the rigorous evaluation phase were inspected and summarized.

New summary report:

- [final_results_summary.md](reports/final_results_summary.md)

Updated reports:

- [full_evidence_one_shot_comparator.md](reports/baselines/full_evidence_one_shot_comparator.md)
- [lambda_cost_sensitive_policy_report.md](reports/baselines/lambda_cost_sensitive_policy_report.md)
- [matched_evidence_integrated_comparison_report.md](reports/baselines/matched_evidence_integrated_comparison_report.md)
- [phase_next_rigorous_evaluation_plan.md](reports/project/phase_next_rigorous_evaluation_plan.md)
- [README.md](README.md)

### Full-Evidence One-Shot Result

Artifact:

- `artifacts/one_shot_full_evidence/full_evidence_pathology_full/`

Official full-split result:

- best validation accuracy: `0.9954`
- best validation macro-F1: `0.9943`
- test accuracy: `0.9958`
- test top-3 accuracy: `1.0000`
- test top-5 accuracy: `1.0000`
- test macro-F1: `0.9948`

Dedup robustness result:

- train-overlap duplicate rows exist in validation/test
- validation duplicates removed: about `1.2%` to `1.4%`
- test duplicates removed: about `1.4%` to `1.5%`
- deduplicated validation/test accuracy remained effectively unchanged
- deduplicated test accuracy stayed about `0.9958`

Interpretation:

- duplicate contamination exists but does not explain the near-ceiling full-evidence result
- DDXPlus contains enough structured evidence for highly accurate diagnosis when full evidence is visible
- this remains a ceiling comparator and must not be used inside the live sequential policy

### Cost-Sensitive Sequential Live Pilot

Artifact:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1/`

Live settings:

- model: `gpt-4.1-mini`
- temperature: `0.0`
- top-p: `1.0`
- request cap: `24`
- cases: `10`
- lambdas: `[0.00, 0.03, 0.06, 0.10, 0.15, 0.22]`

Main result:

- accuracy stayed at `0.900` across all lambda values
- mean requests dropped from `18.4` to `11.8`
- input tokens dropped from `429,339` to `269,060`
- stop-before-cap rate improved from `0.70` to `0.80`

Interpretation:

- the lambda controller materially improved evidence efficiency on the pilot slice
- the best practical lambda range is currently `0.10` to `0.22`
- the result is promising but still small-sample

### Integrated Matched-Evidence Comparison

Artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_6lambdas_lambda_cost_v1__matched_integrated_v1/`

On the same 10 live pilot cases:

- initial-evidence one-shot accuracy: `0.300`
- sequential accuracy: `0.900`
- matched-evidence one-shot accuracy: `0.600` to `0.700`
- full-evidence one-shot accuracy: `1.000`
- sequential recovered about `0.857` of the full-evidence gain
- matched-evidence one-shot recovered about `0.429` to `0.571` of the full-evidence gain

Interpretation:

- sequential is adding value beyond the initial evidence baseline
- sequential also beats the current matched-evidence direct comparator on the same acquired evidence
- this suggests the LLM is not merely selecting useful evidence; it is also using that evidence effectively in this pilot
- caveat: the matched comparator reuses the full-evidence model on partial evidence, so a future partial-evidence-trained comparator would be stronger

Persistent miss:

- `test:81691`, true pathology `Croup`

This case should be used for the next qualitative trace debugging pass.

### Current Recommendation

Do not run another broad exploratory sweep immediately.

Run a cost-controlled 49-case balanced validation with:

- notebook: `08_cost_sensitive_sequential_lambda_sweep.ipynb`
- model: `gpt-4.1-mini`
- temperature: `0.0`
- top-p: `1.0`
- lambdas: `[0.10, 0.15, 0.22]`
- request cap: `24`
- `SEQUENTIAL_SAMPLE_PER_CLASS = 1`
- `SEQUENTIAL_MAX_CASES = None`

Then rerun notebook 09 against that 49-case artifact.

## 32. Wide Lambda Sweep Configuration For Cost-Controlled Follow-Up

Notebook 08 and notebook 09 were reconfigured for the next live experiment.

Reason:

- the previous 10-case run showed flat `0.900` accuracy across all lambda values
- with only 10 cases, accuracy moves in increments of `0.10`, so the run cannot reveal smaller performance differences
- the previous lambda range did not push evidence cost high enough to find the cutoff where accuracy begins to drop
- API cost from the prior 10-case, 6-lambda run was about `$0.93`, so the next run needs to stay near a `$2` budget

Notebook 08 changes:

- `EVIDENCE_COST_LAMBDAS = [0.10, 0.22, 0.35, 0.50, 0.75]`
- `SEQUENTIAL_MAX_CASES = 24`
- `MAX_REQUEST_CAP = 24`
- `ALLOW_DRY_RUN_BENCHMARK = False`
- `RUN_VERSION = "lambda_cost_24case_wide_sweep_v1"`

Expected notebook 08 artifact:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/`

Notebook 09 changes:

- `SEQUENTIAL_RUN_NAME = "single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1"`
- `OUTPUT_VERSION = "matched_integrated_24case_wide_sweep_v1"`

Expected notebook 09 artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_24case_wide_sweep_v1/`

Interpretation goal:

- identify whether lambda values above `0.22` begin to reduce accuracy
- keep enough cases to improve accuracy resolution from `0.10` steps to about `0.042` steps
- keep the experiment likely near the user's API budget

Stop-policy note:

- `MAX_REQUEST_CAP` is a hard upper bound because the sequential loop runs for at most `MAX_REQUEST_CAP` turns
- it is also visible to the policy through `remaining_budget`, prompt construction, and the final forced-stop path
- the cap is not the main experimental variable in notebook 08; lambda is the main variable
- the cap should be interpreted as a safety ceiling, not as unlimited evidence access

## 33. Wide Lambda Sweep Results

The 24-case wide lambda sweep completed and produced the cutoff behavior that the 10-case pilot could not show.

Sequential artifact:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/`

Integrated comparison artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_24case_wide_sweep_v1/`

Settings:

- model: `gpt-4.1-mini`
- temperature: `0.0`
- top-p: `1.0`
- request cap: `24`
- cases: `24`
- lambdas: `[0.10, 0.22, 0.35, 0.50, 0.75]`

Sequential results:

| Lambda | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Cap hits |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.917 | 0.917 | 0.846 | 13.0 | 4/24 |
| 0.22 | 0.875 | 0.875 | 0.917 | 0.795 | 10.7 | 2/24 |
| 0.35 | 0.875 | 0.875 | 0.875 | 0.813 | 8.3 | 1/24 |
| 0.50 | 0.417 | 0.625 | 0.750 | 0.274 | 2.2 | 0/24 |
| 0.75 | 0.375 | 0.583 | 0.708 | 0.288 | 1.0 | 0/24 |

Integrated matched-evidence results:

| Lambda | Initial one-shot acc | Sequential acc | Matched-evidence acc | Full-evidence acc |
|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.917 | 0.625 | 1.000 |
| 0.22 | 0.333 | 0.875 | 0.708 | 1.000 |
| 0.35 | 0.333 | 0.875 | 0.667 | 1.000 |
| 0.50 | 0.333 | 0.417 | 0.333 | 1.000 |
| 0.75 | 0.333 | 0.375 | 0.250 | 1.000 |

Interpretation:

- the run now shows a clear accuracy-efficiency frontier
- `lambda = 0.10` is the strongest accuracy setting, with `22/24` correct
- `lambda = 0.22` and `lambda = 0.35` preserve high accuracy while reducing evidence use
- `lambda = 0.35` gives `21/24` correct with about `8.3` requests per case
- `lambda = 0.50` and `lambda = 0.75` are too aggressive and stop after too little evidence
- sequential beats matched-evidence one-shot at every lambda in this run
- at useful lambdas, there were no cases where matched-evidence one-shot was correct while sequential was wrong

Persistent hard cases:

- `test:81691`, true `Croup`
- `test:62878`, true `Pericarditis`

Updated recommendation:

- next run should be a 49-case balanced validation with lambdas `[0.10, 0.22, 0.35]`
- avoid spending API budget on `0.50+` unless the stop policy is changed

## 34. Partial-Evidence Matched Comparator Added

A stronger matched-information comparator was added.

New notebook:

- [10_partial_evidence_one_shot_comparator.ipynb](notebooks/10_partial_evidence_one_shot_comparator.ipynb)

Updated notebook:

- [09_matched_evidence_integrated_comparison.ipynb](notebooks/09_matched_evidence_integrated_comparison.ipynb)

New report:

- [partial_evidence_matched_comparator.md](reports/baselines/partial_evidence_matched_comparator.md)

Reason:

- notebook 09 previously used the full-evidence one-shot model on partial evidence states for the matched comparator
- this was fair as a first check, but imperfect because the model was trained with all evidence visible
- the stronger comparator should be trained to diagnose from incomplete evidence states

What the sequential traces provide:

- requested evidence root IDs
- reveal payloads
- present/absent/value summaries
- request counts per case
- evidence-root request frequencies

Notebook 10 uses this to train a policy-shaped partial-evidence direct classifier:

- demographics are always visible
- initial evidence is always visible
- additional requested roots are sampled from the sequential policy's observed request distribution
- if a sampled root is present in the training patient row, its true value is encoded
- if a sampled root is absent, it is encoded as absent
- all unrequested fields remain unknown

Fairness rule:

- notebook 10 trains only on official train/validation rows
- it does not train on sequential test labels
- it does not use hidden full evidence at matched-evaluation time
- it does not train a separate model per test case

Notebook 09 update:

- added discovery of `artifacts/one_shot_partial_evidence/selected_model.json`
- if a partial-evidence model exists, matched predictions use it
- if no partial-evidence model exists, notebook 09 falls back to the old full-evidence-model-on-partial-state comparator
- changed notebook 09 `OUTPUT_VERSION` to `matched_integrated_partial_policy_v1` so the new comparison does not overwrite the previous fallback comparison
- `paired_case_results.csv` now includes `matched_model_source`
- `resolved_comparison_config.json` records the partial-evidence model path and matched model source

Expected workflow:

1. Run notebook 10 to train the partial-evidence matched model.
2. Rerun notebook 09.
3. Compare sequential vs the stronger matched-evidence one-shot.

Interpretation:

- if partial-evidence matched one-shot beats sequential, the sequential policy may be best used as an evidence acquisition controller with a direct classifier for final diagnosis
- if sequential still beats partial-evidence matched one-shot, the claim that LLM reasoning adds value over the acquired evidence becomes stronger

### Notebook 10 Path/Loader Fix

Notebook 10 hit a local loader issue when reading the official DDXPlus patient zips.

Problem:

- the release zip members are named `release_train_patients`, `release_validate_patients`, and `release_test_patients`
- they do not necessarily end in `.csv`
- notebook 10 originally searched only for members ending in `.csv`, which raised `FileNotFoundError`

Fix:

- updated notebook 10 `load_patient_split(...)` to match the robust loader used in the earlier notebooks
- it now uses a `.csv` member if present, otherwise falls back to the first non-directory member in the zip
- also made project-root discovery walk upward through all parent directories instead of checking only `cwd` and `cwd.parent`
- cleared stale notebook error outputs after patching

Validation:

- notebook 10 code cells parse cleanly
- the patched loader successfully reads the local train/validate/test release zips

### Notebook 10 Trace Discovery Fix

Notebook 10 hit a second issue while loading the sequential policy mask distribution.

Problem:

- `lambda_dirs(...)` returned lambda artifact directories such as `lambda_0p100`
- `load_policy_mask_distribution(...)` then tried to open those directories as files
- this raised `IsADirectoryError`

Fix:

- replaced `lambda_dirs(...)` with `lambda_trace_files(...)`
- the function now returns each `lambda_*/traces.jsonl` file directly
- cleared stale notebook outputs after patching

Validation:

- notebook 10 code cells parse cleanly
- trace discovery now finds five trace files for the 24-case wide sweep
- mask loading finds `120` policy masks with mean request count about `7.06`

### Notebook 10 Ground-Up Rebuild

Notebook 10 was rebuilt from scratch after repeated incremental issues.

Reason:

- the previous notebook had accumulated brittle fixes
- failures came from multiple assumptions:
  - DDXPlus zip members do not necessarily end in `.csv`
  - sequential lambda artifacts are directories, not trace files
  - `INITIAL_EVIDENCE` can be a bare token like `E_172`, not a Python list string
- rebuilding was cleaner than continuing to patch the old execution path

New implementation properties:

- explicit project-root discovery through parent traversal
- robust DDXPlus split loader matching earlier notebooks
- robust evidence-list parser that handles both list strings and bare tokens
- single trace-loading path that returns `lambda_*/traces.jsonl` files
- explicit preflight summary of dataset zip members and trace masks
- empirical trace-mask sampling from the sequential policy's observed requested root sets
- BASD-compatible partial-evidence encoding
- checkpoint format compatible with notebook 09
- `smoke`, `quick`, `final`, and `full` run modes
- default mode is `final`

Validation performed:

- notebook code cells parse cleanly
- direct smoke harness validated:
  - local DDXPlus split loading
  - evidence parsing
  - trace mask extraction
  - partial-state encoding
  - one small train/evaluate pass
- smoke harness found `feature_size = 922`
- smoke harness found `120` sequential trace masks

Run guidance:

- use notebook 10 in default `final` mode for the stronger matched comparator
- if only checking execution, set environment variable `PARTIAL_EVIDENCE_RUN_MODE=smoke`
- after notebook 10 completes, rerun notebook 09 so it uses the new partial-evidence selected model

## 35. Partial-Evidence Comparator Results And Updated Interpretation

Notebook 10 was run successfully in `final` mode.

Selected partial-evidence model:

- `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

Notebook 9 was rerun using the selected partial-evidence model.

Integrated comparison artifact:

- `artifacts/integrated_comparisons/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1__matched_integrated_partial_policy_v1/`

Notebook 10 standalone metrics:

| Split | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Validation | 0.513 | 0.739 | 0.827 | 0.507 |
| Test | 0.515 | 0.741 | 0.827 | 0.519 |

Training setup:

- train rows: `300,000`
- validation rows: `40,000`
- test rows: `39,998`
- feature size: `922`
- model: `[2048, 2048, 2048]` MLP with dropout `0.10`
- best epoch: `5`
- runtime: about `198` seconds on `mps`

Integrated 24-case partial-matched comparison:

| Lambda | Sequential acc | Partial matched acc | Partial matched top-3 | Partial matched top-5 | Mean requests |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.917 | 0.875 | 1.000 | 1.000 | 13.0 |
| 0.22 | 0.875 | 0.875 | 1.000 | 1.000 | 10.7 |
| 0.35 | 0.875 | 0.833 | 0.958 | 0.958 | 8.3 |
| 0.50 | 0.417 | 0.458 | 0.708 | 0.792 | 2.2 |
| 0.75 | 0.375 | 0.375 | 0.583 | 0.708 | 1.0 |

Win/loss against sequential:

| Lambda | Both correct | Sequential only correct | Matched only correct | Both wrong |
|---:|---:|---:|---:|---:|
| 0.10 | 21 | 1 | 0 | 2 |
| 0.22 | 20 | 1 | 1 | 2 |
| 0.35 | 20 | 1 | 0 | 3 |
| 0.50 | 9 | 1 | 2 | 12 |
| 0.75 | 8 | 1 | 1 | 14 |

Interpretation update:

- the old full-evidence-model fallback was too weak as a matched comparator
- the new partial-evidence direct classifier closes most of the gap to the sequential LLM
- at `lambda = 0.10`, sequential beats the partial matched comparator by one case
- at `lambda = 0.22`, sequential and partial matched tie on top-1 accuracy
- at `lambda = 0.35`, sequential beats partial matched by one case
- partial matched has excellent top-3/top-5 ranking quality at useful lambdas
- therefore, the strongest current claim is about targeted evidence acquisition, not about LLM final diagnosis being clearly superior

Current research implication:

- the sequential policy is useful because it chooses evidence that makes cases highly diagnosable
- the final diagnostic head could be the LLM, the partial-evidence classifier, or a hybrid/adjudicated combination
- the next rigorous evaluation should compare this hybrid idea on the 49-case balanced slice

## 36. Best Current Research Direction To Carry Forward

Based on the latest notebook 08, 09, and 10 results, the strongest current research direction is:

**A controlled sequential evidence-acquisition system with a flexible final diagnostic head.**

This is more defensible than claiming that the LLM alone is the main source of diagnostic performance. The evidence now suggests:

- initial-evidence one-shot diagnosis is weak compared with informed workup
- full-evidence one-shot diagnosis is near-ceiling, which means DDXPlus contains enough information for high diagnostic performance when the right evidence is visible
- cost-sensitive sequential workup can acquire a targeted subset of evidence and reach strong small-sample accuracy
- the partial-evidence direct classifier can use the sequentially acquired evidence almost as well as the LLM on the current 24-case slice
- therefore, the project should emphasize evidence selection, evidence efficiency, and final-head comparison rather than only LLM diagnostic reasoning

Practical architecture direction:

- use the sequential LLM/policy as the workup controller
- keep the deterministic ledger as the source of truth for visible evidence, requested evidence, legality, and revealed values
- use the partial-evidence classifier as a serious matched-information diagnostic comparator
- test a hybrid final decision strategy where the LLM and partial-evidence classifier can agree, disagree, or trigger adjudication
- later multi-agent work should improve evidence acquisition and coordination, not simply add debate around the same weak information state

Next rigorous experiment:

- run the 49-case balanced evaluation slice
- keep `gpt-4.1-mini` fixed
- use deterministic API settings: `temperature = 0.0`, `top_p = 1.0`
- focus on useful lambdas: `0.10`, `0.22`, and `0.35`
- compare sequential LLM final answers, partial-evidence classifier final answers, and a simple hybrid/adjudicated rule
- report accuracy, top-3/top-5, macro-F1, mean requests, stop behavior, and disagreement cases

Scientific framing:

The best current research question is:

Can a structured sequential diagnostic workup controller approach full-evidence diagnostic performance while acquiring only a limited targeted subset of evidence, and should final diagnosis be made by the LLM, a neural classifier, or a hybrid of both?

## 37. Future Direction: Online LLM-MLP Hybrid Workup Controller

A future idea worth preserving is an **online hybrid system** that is stronger than the current matched-evidence comparator.

Important distinction:

- current matched-evidence comparator is offline/retrospective
- the sequential LLM has already acquired evidence
- the partial-evidence MLP is then evaluated afterward on exactly that acquired evidence
- this proves that LLM-acquired evidence can be useful for a neural diagnostic head, but the neural head does not influence the workup while it is happening

The proposed hybrid should be online:

```text
current ledger state
    -> partial-evidence MLP updates diagnosis distribution
    -> LLM sees selected MLP belief signals
    -> stop/request logic uses MLP confidence, LLM diagnosis, and disagreement
    -> LLM requests next legal evidence if uncertainty/disagreement remains
    -> ledger reveals only that requested evidence
    -> loop repeats
```

This is **not GAN-like** and should not be framed as adversarial training. The MLP and LLM are not opponents. They have different roles:

- `LLM`: flexible evidence-acquisition planner and clinical-language reasoner
- `MLP`: stable structured diagnostic belief estimator over the BASD-style evidence state
- `ledger`: deterministic source of truth for visible evidence, hidden evidence, legal actions, revealed values, and traceability
- `stop/request policy`: uses confidence, margin, stability, disagreement, and evidence cost

Possible hybrid levels:

1. One-way hybrid:
   - LLM gathers evidence
   - MLP makes final diagnosis
   - this is closest to the current matched-comparator result

2. Confidence-gated hybrid:
   - MLP runs after each reveal
   - stop if MLP confidence/margin is high enough
   - otherwise LLM asks another evidence question

3. Disagreement-aware hybrid:
   - LLM and MLP both maintain top-k differentials
   - if they agree, stop
   - if they disagree, ask a discriminative question targeted at the competing hypotheses

4. MLP-guided question selection:
   - MLP identifies uncertain top competing pathologies
   - LLM chooses a legal evidence question that separates those pathologies
   - this keeps the LLM in control of language/action selection while grounding its choices in a neural belief state

Why this may matter:

- our latest results suggest the partial-evidence MLP is competitive with the LLM once evidence has been acquired
- this means the best architecture may be to use the LLM for evidence control and the MLP for calibrated final diagnosis or confidence signals
- online MLP feedback could improve lambda behavior by making stopping depend on diagnostic margin rather than only hand-designed heuristic value
- disagreement between LLM and MLP can be used as a trigger for more evidence instead of forcing an early final answer

How to evaluate it later:

- compare against current notebook 08 cost-sensitive LLM-only sequential policy
- compare against notebook 09 offline matched-evidence MLP
- use the same balanced 49-case slice first
- keep `gpt-4.1-mini` fixed for continuity
- report accuracy, top-3/top-5, macro-F1, mean requests, disagreement rate, stop-before-cap rate, and evidence-efficiency utility

Claim discipline:

- do not claim this is a new paradigm like GANs
- do not claim hybrid superiority until tested
- frame it as a practical division of labor between evidence acquisition and structured diagnosis
- cite active feature acquisition as older related work, and position the novelty around LLM-led evidence control plus deterministic ledger plus matched/online diagnostic-head analysis

## 38. Current Roadmap Before Next Implementation Phase

This section records the current project direction before starting the next round of implementation.

The main clarification:

- the broad ideas of evidence acquisition, diagnostic agents, RL workup policies, LLM medical agents, and classifier-assisted diagnosis already exist
- our novelty should be framed around the specific combination and application:
  - DDXPlus structured diagnostic workup
  - deterministic evidence ledger
  - LLM-controlled legal evidence acquisition
  - BASD-style partial-evidence MLP diagnostic head
  - matched-evidence decomposition
  - cost-sensitive stopping
  - future online LLM-MLP feedback
  - future architecture-agnostic algorithmic ledger signals

Working novelty statement:

> Prior DDXPlus work studies trained RL/supervised diagnostic agents, and recent medical LLM-agent work studies interactive diagnosis. Our project studies a ledger-gated LLM evidence-acquisition controller on DDXPlus, evaluates it against BASD-style neural diagnostic heads under matched evidence, and develops a path toward online hybrid feedback and algorithmic ledger-guided evidence selection.

### Roadmap Stage 1: Freeze The Evaluation Frame

Before adding more architecture, the comparator story should stay stable.

Keep these as named comparator families:

- initial-evidence one-shot
- full-evidence one-shot ceiling
- offline matched-evidence partial MLP
- current LLM-only cost-sensitive sequential policy
- future online hybrid LLM-MLP policy
- later heuristic or simple non-LLM evidence-acquisition baseline
- published DDXPlus AARLC/BASD numbers as external reference points

Rules:

- do not casually replace old comparators after interpreting results
- if a comparator improves, version it as a new comparator
- report what each comparator is meant to answer
- avoid the vague claim "agentic beats one-shot"
- use the better question: where does value come from, evidence acquisition or final diagnosis?

### Roadmap Stage 2: Implement Online Hybrid Before Multi-Agent

The next implementation target should be the online hybrid.

Reason:

- it is the smallest meaningful step beyond the current matched comparator
- it directly tests whether MLP feedback improves evidence efficiency and stopping
- it gives the future algorithmic ledger a useful consumer

Planned online hybrid loop:

```text
ledger state
    -> partial-evidence MLP predicts top-k diagnosis distribution
    -> compute confidence, margin, entropy, stability, and LLM-MLP agreement
    -> LLM receives selected belief signals plus legal/shortlisted evidence actions
    -> stop if confidence/agreement is sufficient under current lambda
    -> otherwise request one legal evidence field
    -> ledger reveals present/absent/value information
    -> repeat
```

Initial hybrid variants to test:

- MLP-final:
  - LLM chooses evidence
  - MLP makes final diagnosis

- confidence-gated:
  - MLP confidence/margin controls stopping
  - LLM chooses questions when uncertainty remains

- disagreement-aware:
  - LLM and MLP top-k predictions are compared
  - disagreement triggers more targeted evidence requests

Expected value:

- may reduce unnecessary evidence requests
- may stabilize final diagnosis
- may give cleaner lambda behavior
- may show whether the LLM is better used as question selector than final classifier

### Roadmap Stage 3: Build Architecture-Agnostic Algorithmic Ledger

The algorithmic ledger should not be dependent on any one controller.

It should sit between the deterministic environment ledger and the policy/controller:

```text
DDXPlus environment
    -> deterministic evidence ledger
    -> algorithmic ledger signals
    -> controller/policy
    -> final diagnostic head
```

The algorithmic ledger should expose reusable signals:

- visible evidence state
- requested evidence history
- legal actions
- top competing diagnoses
- belief distribution from MLP or other diagnostic head
- confidence/margin/entropy
- diagnosis stability across turns
- discriminative evidence candidates
- severity-aware risk flags
- contradiction or consistency warnings
- expected-value proxy for asking another question
- cost-sensitive stop score

Consumers:

- single-agent LLM can receive ledger signals in the prompt
- online hybrid can use MLP belief and disagreement signals
- multi-agent system can share one ledger as the common source of truth
- heuristic baseline can use ledger scores directly without LLM
- future evolutionary search can tune ledger weights/rules

Design principle:

- the ledger should be a reusable state-and-signal layer, not hardwired to the online hybrid

### Roadmap Stage 4: Add Multi-Agent Only After Ledger Signals Are Stable

Do not jump to multi-agent next.

Reason:

- multi-agent without shared state can become prompt choreography
- it increases cost and variance
- it may not improve evidence acquisition unless roles are grounded in ledger signals

Possible later multi-agent roles:

- evidence planner:
  - proposes the next most discriminative evidence request

- diagnosis tracker:
  - maintains top competing diagnoses

- severity/safety reviewer:
  - checks whether severe alternatives have been ruled out

- adjudicator:
  - resolves LLM/MLP disagreement or asks for more evidence

Multi-agent should be justified only if each role consumes a different view of the ledger and improves a measurable metric.

### Roadmap Stage 5: Consider Evolutionary Optimization As A Later Strategy Tuner

Evolutionary algorithms are relevant, but they should not replace the agentic system.

Best use:

- optimize weights or rules inside the algorithmic ledger
- tune stopping thresholds
- tune evidence-score weights
- tune disagreement triggers
- tune severity-aware tradeoffs

Example score:

```text
score(question) =
    w1 * uncertainty_reduction
  + w2 * diagnosis_discrimination
  + w3 * severity_ruleout
  + w4 * evidence_frequency
  - w5 * evidence_cost
```

Evolutionary search could optimize the `w` values on validation episodes.

Why this is useful:

- keeps the policy interpretable
- avoids immediately moving into opaque RL
- can improve evidence strategy while preserving LLM/ledger architecture

Why it is later, not next:

- it needs stable ledger signals first
- it needs a clear validation protocol
- it should optimize an existing strategy layer, not redefine the whole project

### Roadmap Stage 6: External Benchmark Alignment

To make the project harder to criticize, future reports should align more directly with the original DDXPlus paper.

Add official or paper-aligned metrics where feasible:

- interaction length / mean requests
- ground-truth pathology top-1 accuracy
- ground-truth pathology included in differential
- positive evidence recall
- differential diagnosis recall
- differential diagnosis precision
- differential diagnosis F1

Use published DDXPlus AARLC/BASD results as external anchors.

Important caveat:

- if our live API experiments use small balanced slices, do not present them as full benchmark replacements
- label them as controlled pilots until a larger run is performed

### Current Recommended Order

1. Keep the current evaluation frame fixed.
2. Implement online hybrid LLM-MLP controller.
3. Compare online hybrid to LLM-only sequential and offline matched MLP.
4. Add architecture-agnostic algorithmic ledger signals.
5. Test ledger signals with single-agent and hybrid controllers.
6. Add heuristic evidence-acquisition baseline.
7. Only then evaluate whether multi-agent roles are useful.
8. Consider evolutionary optimization of ledger weights/rules after the ledger is stable.

Current guiding question:

> Can a ledger-controlled LLM/hybrid diagnostic workup system select a small, targeted subset of DDXPlus evidence that approaches full-evidence performance, while remaining more interpretable and controllable than trained black-box acquisition policies?

## 39. Hybrid V1 Implementation Started

Hybrid v1 was implemented as the next successor notebook.

New notebook:

- `notebooks/11_online_hybrid_mlp_feedback.ipynb`

Updated notebook:

- `notebooks/09_matched_evidence_integrated_comparison.ipynb`

New report scaffold:

- `reports/hybrid/hybrid_mlp_feedback_report.md`

Purpose:

- keep the current single-agent evidence-acquisition setup
- load the selected partial-evidence MLP from notebook 10
- run the MLP after each ledger update
- expose compact MLP belief signals to the LLM prompt
- use MLP confidence, margin, entropy, stability, and LLM/MLP agreement for stopping/final-head adjudication
- save LLM-final, MLP-final, and hybrid-final outputs from the same trace

Artifact root:

- `artifacts/sequential_hybrid_mlp_feedback/<run_name>/`

Default settings:

- `LLM_MODEL = gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- `MAX_REQUEST_CAP = 24`
- lambdas: `[0.10, 0.22, 0.35]`
- default live API mode is controlled directly in the notebook with `RUN_LIVE_API`
- dry-run benchmark artifacts are controlled directly in the notebook with `ALLOW_DRY_RUN_BENCHMARK`

Important scope boundary:

- hybrid v1 is not multi-agent
- hybrid v1 does not implement graph inference or algorithmic ledger reasoning
- LLM still chooses evidence
- MLP acts as a diagnostic belief monitor and final-head candidate

Evaluation update:

- notebook 09 can now discover `artifacts/sequential_hybrid_mlp_feedback`
- comparison outputs include:
  - initial one-shot
  - full-evidence ceiling
  - offline matched MLP
  - hybrid LLM-final
  - hybrid online MLP-final
  - hybrid adjudicated final

Next action:

- run notebook 11 in dry-run smoke mode
- then run a live 24-case pilot with lambdas `[0.10, 0.22, 0.35]`
- rerun notebook 09 against the hybrid artifact

Status: completed and superseded by section 40.

## 40. Hybrid V1 Live Results And Interpretation

Notebook `11` has now been run live and notebook `09` has been rerun against the hybrid artifact.

Hybrid artifact:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1/`

Integrated comparison artifact:

- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/`

Reports updated:

- `reports/hybrid/hybrid_mlp_feedback_report.md`
- `reports/final_results_summary.md`
- `README.md`

Run settings:

- model: `gpt-4.1-mini`
- temperature: `0.0`
- top_p: `1.0`
- sample: 24 balanced test cases
- request cap: `24`
- lambdas: `[0.10, 0.22, 0.35]`
- partial MLP source: `artifacts/one_shot_partial_evidence/partial_evidence_one_shot_final_policy_masked_v2/`

Hybrid results:

| Lambda | Hybrid acc | Top-3 | Top-5 | Macro-F1 | Mean requests | Stop before cap | Cap hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.833 | 1.000 | 1.000 | 0.756 | 9.7 | 0.875 | 3 |
| 0.22 | 0.875 | 0.958 | 0.958 | 0.813 | 7.5 | 0.917 | 2 |
| 0.35 | 0.833 | 0.917 | 0.917 | 0.744 | 5.9 | 0.958 | 1 |

Comparison against notebook `08` on the same useful lambdas:

| Lambda | Notebook 08 acc | Notebook 08 requests | Hybrid acc | Hybrid requests | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.10 | 0.917 | 13.0 | 0.833 | 9.7 | fewer requests, worse accuracy |
| 0.22 | 0.875 | 10.7 | 0.875 | 7.5 | same accuracy, about 30% fewer requests |
| 0.35 | 0.875 | 8.3 | 0.833 | 5.9 | fewer requests, one extra error |

Integrated comparison results:

| Lambda | Initial one-shot | Hybrid final | LLM final | Online MLP final | Matched MLP | Full evidence |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.333 | 0.833 | 0.875 | 0.833 | 0.833 | 1.000 |
| 0.22 | 0.333 | 0.875 | 0.875 | 0.875 | 0.875 | 1.000 |
| 0.35 | 0.333 | 0.833 | 0.833 | 0.833 | 0.833 | 1.000 |

Interpretation:

- hybrid v1 did not improve raw accuracy over notebook `08`
- hybrid v1 did improve evidence efficiency at the useful `lambda = 0.22` point
- at `lambda = 0.22`, hybrid matches notebook `08` accuracy while reducing mean requests from `10.7` to `7.5`
- hybrid v1 did not prove that final adjudication is better than LLM-final or MLP-final
- at `lambda = 0.10`, the hybrid final rule hurt one case by trusting a high-confidence wrong MLP over a correct LLM
- MLP confidence is not calibrated enough to safely override the LLM in disagreements

Persistent hybrid errors:

- `test:81691`, true `Croup`
- `test:8666`, true `Influenza`
- `test:62878`, true `Pericarditis`
- at lambda `0.35`, `test:51421`, true `Chagas`, also fails

Scientific takeaway:

Hybrid v1 is a useful evidence-efficiency result, not a final-diagnosis breakthrough. The partial-evidence MLP appears helpful for stopping and stability, but final-head arbitration should be more conservative.

Recommended next step:

1. Do not run a larger hybrid benchmark yet.
2. Patch the hybrid final rule so high-confidence MLP does not override the LLM by default in disagreements.
3. Keep MLP feedback in the prompt and stopping logic.
4. Rerun the same 24-case slice at `lambda = 0.22` only.
5. If the same `0.875` accuracy with roughly `7.5` requests holds, then run a 49-case balanced pilot.

Current best claim:

> Online MLP feedback can make the ledger-controlled sequential workup more evidence-efficient, but the current hybrid final-adjudication rule is not yet better than the individual LLM or MLP heads.

## 41. Notebook 09 Evidence-Budget Comparison Upgrade

Notebook `09` was updated after reviewing the integrated comparison graphs.

Reason:

- lambda is a controller setting, not the actual amount of evidence acquired
- the previous plots were useful but too lambda-centric
- the project needs clearer evidence-efficiency graphs showing accuracy/ranking quality against actual request counts and visible evidence roots

Notebook updated:

- `notebooks/09_matched_evidence_integrated_comparison.ipynb`

New/updated artifacts:

- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/evidence_budget_summary.csv`
- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/evidence_efficiency_frontier.csv`
- `artifacts/integrated_comparisons/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1__hybrid_v1_integrated_v1/case_outcome_matrix.csv`
- `figures/accuracy_vs_mean_requests_integrated.png`
- `figures/top5_vs_mean_requests_integrated.png`
- `figures/accuracy_vs_revealed_roots_integrated.png`
- `figures/top5_vs_revealed_roots_integrated.png`
- `figures/evidence_usage_by_policy_setting.png`
- lambda-based figures are preserved

New report:

- `reports/baselines/integrated_evidence_budget_comparison_report.md`

Key evidence-budget table:

| Lambda | Mean requests | Mean visible roots incl. initial | Hybrid acc | Online MLP acc | Offline matched MLP acc | Hybrid top-5 | Offline matched top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 9.7 | 10.7 | 0.833 | 0.833 | 0.833 | 1.000 | 1.000 |
| 0.22 | 7.5 | 8.5 | 0.875 | 0.875 | 0.875 | 0.958 | 0.958 |
| 0.35 | 5.9 | 6.9 | 0.833 | 0.833 | 0.833 | 0.917 | 0.917 |

Interpretation:

- yes, the hybrid run maintains matched-MLP accuracy with less evidence at `lambda = 0.22` and `lambda = 0.35`
- at `lambda = 0.10`, it loses one case versus the previous notebook `08` matched-MLP result because it stops earlier
- top-5 ranking quality for online MLP, hybrid final, and offline matched MLP moves together
- the current benefit is evidence efficiency, not better final diagnosis
- the actual evidence-count plots are now more scientifically useful than lambda-only plots

Current best result:

- `lambda = 0.22`
- `21/24` correct
- `7.46` mean requested evidence fields
- `8.46` mean visible roots including initial evidence
- same top-1 accuracy as the old matched-MLP setup at that lambda, with about 30% fewer requests

## 42. Notebook 12 Stopping-Policy Ablation

Implemented a new offline ablation notebook:

- `notebooks/12_stopping_policy_ablation.ipynb`

Purpose:

- determine whether hybrid v1's evidence-efficiency gain comes from the partial-evidence MLP providing a genuinely better stopping signal
- compare MLP-guided stopping against LLM-only stopping at matched evidence budgets
- avoid extra API usage by replaying existing notebook `08` and notebook `11` traces

Main replay source:

- `artifacts/sequential_single_agent_cost_sensitive/single_agent_cost_sensitive_live_test_1perclass_cap24_5lambdas_lambda_cost_24case_wide_sweep_v1/lambda_0p100/`

Hybrid reference source:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_mlp_feedback_live_test_1perclass_cap24_3lambdas_hybrid_mlp_feedback_v1/`

New artifact root:

- `artifacts/stopping_policy_ablation/stopping_policy_ablation_24case_v1/`

New report:

- `reports/hybrid/stopping_policy_ablation_report.md`

Notebook behavior:

- no API calls
- reconstructs turn-level ledger states from saved traces
- runs the notebook `10` partial-evidence MLP at every saved turn
- extracts LLM confidence, differential, stability, deterministic margin, marginal value, MLP confidence, MLP margin, MLP entropy, and LLM/MLP agreement signals
- sweeps LLM-only, deterministic-state, marginal-value, MLP-confidence, LLM/MLP-agreement, and conservative hybrid stopping rules
- evaluates LLM-final, MLP-final, conservative hybrid final, and agreement hybrid final heads from the same simulated stop turn

Validation:

| Item | Result |
|---|---:|
| Replay cases | 24 |
| Turn-level rows | 333 |
| Policy specs | 309 |
| Case-policy rows | 7,416 |
| Policy summary rows | 1,236 |
| MLP reconstruction match against notebook 11 | 24/24 |

Observed references:

| System | Lambda | Correct | Accuracy | Mean requests |
|---|---:|---:|---:|---:|
| Notebook 08 LLM-only | 0.10 | 22/24 | 0.917 | 13.04 |
| Notebook 08 LLM-only | 0.22 | 21/24 | 0.875 | 10.67 |
| Notebook 08 LLM-only | 0.35 | 21/24 | 0.875 | 8.33 |
| Notebook 11 hybrid v1 | 0.22 | 21/24 | 0.875 | 7.46 |

Matched-budget ablation at about `7.5` requests:

| Policy group | Final head | Correct | Accuracy | Mean requests |
|---|---|---:|---:|---:|
| Best pure LLM-only stop | LLM | 20/24 | 0.833 | 6.33 |
| Best MLP-guided stop | LLM | 22/24 | 0.917 | 6.88 |
| Best MLP-guided stop | MLP | 22/24 | 0.917 | 6.25 |
| Best MLP-guided stop | conservative hybrid | 22/24 | 0.917 | 6.88 |
| Best MLP-guided stop | agreement hybrid | 22/24 | 0.917 | 6.88 |

Selected policy:

- `mlp_conf_ge_0.70_margin_ge_0.20_entropy_le_0.10_stab_0`
- selected final head: `agreement_hybrid_final`
- accuracy: `0.9167`
- mean requests: `6.875`
- median requests: `4.5`

Selected-policy errors:

| Case | True pathology | Predicted pathology | Requests | Interpretation |
|---|---|---|---:|---|
| `test:81691` | `Croup` | `Acute otitis media` | 7 | high-confidence false agreement between LLM and MLP |
| `test:62878` | `Pericarditis` | `Panic attack` | 23 | remained wrong at end of trace; likely question-selection/trajectory failure |

Additional sweep finding:

| Policy | Final head | Correct | Accuracy | Mean requests | Note |
|---|---|---:|---:|---:|---|
| `llm_conf_ge_0.85_stab_2` | MLP | 23/24 | 0.958 | 9.83 | highest offline accuracy; stop signal itself does not use MLP |
| `det_margin_ge_3.00_unres_le_0.05` | MLP | 23/24 | 0.958 | 9.96 | similar high-accuracy point |

Interpretation:

- this is the clearest evidence so far that MLP feedback helps stopping, not merely final diagnosis
- on fixed notebook `08` evidence trajectories, MLP-guided stopping preserves the high-accuracy `22/24` result with roughly half the requests of notebook `08` lambda `0.10`
- the best pure LLM-only stopping rule at the same approximate budget reaches only `20/24`
- this supports a stronger hybrid framing: the MLP should act as an online diagnostic confidence/stopping monitor, while final diagnosis can remain conservative
- the higher-budget `23/24` replay result suggests the MLP final head may become the best diagnostic head once enough targeted evidence is acquired

Important limitation:

- this is offline replay
- it tests when to stop along an already-recorded evidence trajectory
- it does not prove that the LLM would ask the same future questions in a live run under the selected stop policy

Next recommended step:

- implement the selected MLP-guided stopping rule in a live successor run
- keep `gpt-4.1-mini`, `temperature = 0.0`, and `top_p = 1.0`
- run the same 24-case slice first
- avoid another wide lambda sweep
- compare against notebook `08` lambda `0.10` and notebook `11` lambda `0.22`
- if that works, test a second accuracy-biased live policy targeting about `9-10` requests with MLP final diagnosis

## 43. Notebook 13 Live Selected-Stop Confirmation

Implemented the live confirmation notebook:

- `notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb`

Purpose:

- take notebook `12`'s selected MLP-guided stop rule out of offline replay
- use it inside the actual sequential LLM loop
- verify whether the offline `22/24` at about `6.875` requests can survive live interaction

Default artifact root:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

Dry-run smoke artifact:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_dryrun_smoke_v1/`

New report:

- `reports/hybrid/live_selected_hybrid_stopping_confirmation.md`

Selected stop rule:

| Parameter | Value |
|---|---:|
| minimum requested fields | 1 |
| MLP confidence minimum | 0.70 |
| MLP margin minimum | 0.20 |
| MLP entropy maximum | 0.10 |
| MLP stability minimum | 0 |

Notebook behavior:

- single selected stop policy, not a lambda sweep
- `RUN_LIVE_API = False` by default
- `gpt-4.1-mini`, `temperature = 0.0`, `top_p = 1.0`
- same 24-case slice as notebooks `08`, `11`, and `12`
- LLM still chooses evidence requests
- partial-evidence MLP supplies online confidence, margin, entropy, top-k predictions, and stop signal
- final heads saved from the same trace:
  - LLM final
  - MLP final
  - agreement hybrid final
  - conservative hybrid final

Dry-run validation completed:

| Check | Result |
|---|---|
| Notebook code parse | passed |
| Dry-run smoke cases | 2 |
| Live API calls | none |
| Predictions written | yes |
| Traces written | yes |
| Raw response log written | yes |
| Metrics written | yes |
| Figures written | yes |
| Qualitative examples written | yes |

Dry-run smoke result:

- `2/2` correct
- mean requests: `6.0`
- selected stop rule fired for both smoke cases

This is not a scientific result. It only validates the notebook wiring.

Live run instructions:

- use the notebook `08`-style interactive `getpass` key bootstrap or set `LLM_API_KEY`
- set `RUN_LIVE_API = True`
- keep `ALLOW_DRY_RUN_BENCHMARK = False`
- keep `SEQUENTIAL_MAX_CASES = 24`
- restart kernel and run all cells

Acceptance target:

- preferred: `22/24` correct with `6-7.5` mean requests
- acceptable: `21/24` correct with clear request reduction versus notebook `08`
- failure: below notebook `11` accuracy without meaningful request reduction

Live result completed:

| System | Correct | Accuracy | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|
| Notebook `08`, lambda `0.10` | 22/24 | 0.917 | 0.917 | 0.846 | 13.04 |
| Notebook `11`, lambda `0.22` | 21/24 | 0.875 | 0.958 | 0.813 | 7.46 |
| Notebook `12`, offline selected stop | 22/24 | 0.917 | 0.917 | 0.867 | 6.875 |
| Notebook `13`, live selected stop | 22/24 | 0.917 | 0.917 | 0.867 | 6.58 |

Notebook `13` live details:

- median requests: `4.5`
- stop-before-cap rate: `1.0`
- cap hits: `0`
- selected stop rule fired: `20/24`
- LLM/MLP top-1 agreement: `23/24`
- input tokens: `410,536`
- output tokens: `20,979`
- runtime: about `717` seconds

Efficiency interpretation:

- versus notebook `08`, notebook `13` used `49.5%` fewer requests and `42.2%` fewer input tokens with the same `22/24` accuracy
- versus notebook `11`, notebook `13` used `11.7%` fewer requests and improved accuracy by one case
- live result matches the offline notebook `12` accuracy and slightly improves mean requests (`6.58` vs `6.875`)

Final-head result:

- agreement hybrid, conservative hybrid, LLM final, and MLP final all reached `22/24`
- MLP final had better top-5 (`0.958`) than the other heads (`0.917`)
- final arbitration is not the bottleneck in this run

Live errors:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:81691` | `Croup` | `Chagas` | 23 | selected MLP stop |
| `test:62878` | `Pericarditis` | `Anemia` | 16 | agent stop |

Current interpretation:

- this is the strongest evidence-efficiency result so far
- the MLP stop signal is now confirmed both offline and live
- the next bottleneck is not whether to stop, but how to avoid bad evidence trajectories for hard diseases like `Croup` and `Pericarditis`
- next possible experiment is either a larger 49-case live confirmation or targeted hard-case policy work before scaling

## 44. Evaluation Scale Policy Going Forward

Use the 24-case balanced slice for method development and candidate selection, not for final statistical claims.

Reason:

- on 24 cases, one case changes accuracy by `4.17` percentage points, so accuracy differences are coarse and sometimes visually confusing
- 24 cases are still useful for comparing request efficiency, stop behavior, token usage, hard-case traces, and whether a new policy is obviously worse
- repeated 49-case live runs are more expensive, so they should be reserved for confirmation after a method is frozen

Current policy:

- develop v2-style ideas on the same 24-case slice first
- promote only if the new method preserves about `22/24` accuracy while reducing requests, improves hard-case traces, or clearly improves ranking quality
- use 49 cases near the end as a broader confirmation run, ideally after freezing the selected method and avoiding further prompt/policy tuning

## 45. Notebook 14 Hybrid V2 Candidate: MLP-Discriminative Shortlist

Implemented a new candidate notebook:

- `notebooks/14_hybrid_v2_mlp_discriminative_shortlist.ipynb`

Purpose:

- keep notebook `13`'s proven hybrid v1 stop policy and final-head setup
- change only the question-selection layer
- test whether MLP-guided evidence shortlisting can preserve v1 accuracy while reducing mean requests or improving hard-case trajectories

Key method change:

```text
current ledger-visible evidence
  -> partial-evidence MLP top competing diagnoses
  -> score legal unrevealed DDXPlus root evidence fields
  -> shortlist fields with high diagnostic separation and MLP entropy reduction
  -> LLM chooses from that shortlist
  -> notebook 12 selected MLP stop rule decides when to stop
```

V2 shortlisting score:

```text
score = penalty * (
  0.35 * mlp_pair_gap
  + 0.25 * top1_vs_rest_gap
  + 0.25 * entropy_gain
  + 0.10 * split_balance
  + 0.05 * disagreement_gap
)
```

Where:

- `mlp_pair_gap` separates the MLP top competing diagnoses using train/validate-derived pathology evidence rates
- `top1_vs_rest_gap` separates the MLP top diagnosis from the remaining top competitors
- `entropy_gain` estimates how much the MLP's uncertainty would drop under counterfactual reveal states for that evidence field
- `split_balance` favors questions that plausibly divide the competing diagnoses rather than being almost always present/absent
- `disagreement_gap` boosts fields that separate the deterministic anchor from the MLP top diagnosis when they disagree
- `penalty` preserves v1's generic/global/child-action penalties

Fairness rule:

- no hidden test labels are used
- no full-evidence predictions are used
- no unrevealed test evidence is used
- aggregate evidence rates are allowed because they are derived from non-test policy statistics, not from the current case label

Default notebook state:

- `RUN_LIVE_API = False`
- `ALLOW_DRY_RUN_BENCHMARK = False`
- `SEQUENTIAL_MAX_CASES = 24`
- fixed LLM model: `gpt-4.1-mini`
- fixed API determinism: `temperature = 0.0`, `top_p = 1.0`

Live artifact target:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_24case_v1/`

Dry-run smoke validation completed without live API calls:

- smoke artifact: `artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_dryrun_smoke_v1/`
- 2 dry-run cases executed
- predictions, traces, raw response logs, metrics, v1-v2 paired comparison, promotion decision, shortlist component diagnostics, request-frequency tables, and figures were written
- code-cell static parse passed

Promotion rule:

- promote v2 to a 49-case confirmation only if it preserves about `22/24` accuracy while lowering requests, reaches `23/24` with reasonable requests, or preserves `22/24` while improving top-5/hard-case behavior
- if v2 fails those criteria, notebook `13` remains the frozen proposed method

Live v2 result completed:

| System | Correct | Accuracy | Top-5 | Macro-F1 | Mean requests | Input tokens |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` hybrid v1 selected stop | 22/24 | 0.917 | 0.917 | 0.867 | 6.58 | 410,536 |
| Notebook `14` hybrid v2 MLP shortlist | 21/24 | 0.875 | 0.958 | 0.840 | 7.38 | 509,158 |

V2 promotion decision:

- `reject_keep_notebook13_v1`

Paired v1-v2 outcomes:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| V1 only correct | 2 |
| V2 only correct | 1 |
| Both wrong | 1 |

Case-level interpretation:

- v2 fixed `test:62878` / `Pericarditis`, but only after hitting the full 24-request cap
- v2 still failed `test:81691` / `Croup`, despite briefly considering Croup early and spending 23 requests
- v2 introduced new failures on `Chagas` and `Influenza`
- v2 improved top-5 from `0.917` to `0.958`, but lost top-1 accuracy and used more evidence

Scientific interpretation:

- v2 is an informative negative result, not a project failure
- MLP-driven shortlisting mechanically works and produces discriminative questions, but direct MLP control of the shortlist can amplify unstable/wrong MLP competitors
- the strongest current claim remains notebook `13`: MLP as an online stopping signal is useful; MLP as direct question-selection controller is not yet proven

New report:

- `reports/hybrid/hybrid_v2_mlp_discriminative_shortlist_report.md`

## 46. Notebook 13 Final 49-Case Confirmation

After rejecting notebook `14` as the main method, notebook `13` was frozen as the proposed hybrid v1 system and rerun on a broader 49-case balanced live slice.

Notebook:

- `notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb`

Artifact:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`

Fixed settings:

- LLM: `gpt-4.1-mini`
- temperature: `0.0`
- top-p: `1.0`
- request cap: `24`
- cases: `49`
- stop rule: notebook `12` selected MLP-guided stop rule

Final 49-case result:

| Metric | Value |
|---|---:|
| Agreement-hybrid accuracy | 43/49 = 0.878 |
| LLM-final accuracy | 43/49 = 0.878 |
| MLP-final accuracy | 41/49 = 0.837 |
| Conservative-hybrid accuracy | 43/49 = 0.878 |
| Top-3 accuracy | 0.918 |
| Top-5 accuracy | 0.939 |
| Macro-F1 | 0.845 |
| Mean requests | 6.59 |
| Median requests | 5.0 |
| Mean visible roots including initial | 7.59 |
| Stop-before-cap rate | 0.980 |
| Cap hits | 1 |
| Selected stop-rule fired | 36/49 = 0.735 |
| LLM/MLP agreement | 46/49 = 0.939 |
| Input tokens | 823,478 |
| Output tokens | 42,721 |

Comparison to 24-case pilot:

| Run | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Notebook `13`, 24 cases | 0.917 | 0.917 | 6.58 |
| Notebook `13`, 49 cases | 0.878 | 0.939 | 6.59 |

Interpretation:

- the 24-case result was optimistic but not misleading about evidence efficiency
- top-1 accuracy dropped on the broader slice, but top-5 improved and mean requests stayed almost identical
- the final claim should use the 49-case result as the main proposed-method result
- notebook `13` remains the strongest current method
- notebook `14` remains a negative ablation showing that direct MLP-driven question shortlisting is not automatically better

49-case error cases:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:38475` | `Acute COPD exacerbation / infection` | `Myocarditis` | 24 | max requests reached |
| `test:111176` | `Acute rhinosinusitis` | `Chronic rhinosinusitis` | 8 | selected MLP stop |
| `test:81691` | `Croup` | `Anemia` | 19 | agent stop |
| `test:8666` | `Influenza` | `HIV (initial infection)` | 3 | agent stop |
| `test:62878` | `Pericarditis` | `Anemia` | 15 | agent stop |
| `test:125508` | `Unstable angina` | `Anemia` | 2 | agent stop |

Final project framing:

- lower bound: initial-evidence MLP, `0.378` full-test accuracy
- ceiling: full-evidence MLP, `0.996` full-test accuracy
- sequential baseline: notebook `08`, `22/24` at `13.04` requests on the 24-case slice
- proposed method: notebook `13`, `43/49` at `6.59` requests on the final confirmation
- negative ablation: notebook `14`, `21/24` at `7.38` requests, not promoted

Final report added:

- `reports/final_report.md`

Report organization added:

- current final reports remain in `reports/`
- baseline reports moved under `reports/baselines/`
- hybrid/sequential policy reports moved under `reports/hybrid/`
- architecture and ledger reports moved under `reports/architecture/`
- planning/claims reports moved under `reports/project/`
- index added at `reports/README.md`
