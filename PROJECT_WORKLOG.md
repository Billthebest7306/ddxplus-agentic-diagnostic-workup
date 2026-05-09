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

## 47. Notebook 15 Stop-Policy Sensitivity And Evidence-Trajectory Diagnostics

Notebook `15` was added as an offline analysis notebook for the final Notebook `13` 49-case live run.

Notebook:

- `notebooks/15_notebook13_stop_policy_sensitivity.ipynb`

Artifact root:

- `artifacts/stop_policy_sensitivity/notebook13_49case_v1/`

Report:

- `reports/hybrid/notebook13_stop_policy_sensitivity_report.md`

Purpose:

- inspect which evidence fields were requested
- compare request counts for correct vs incorrect cases
- inspect hard-case trajectories
- quantify wrong-to-correct and correct-to-wrong prediction transitions over turns
- run an offline sweep over MLP stop thresholds without any API calls

Important implementation detail:

- the notebook parses Notebook `13` traces for turn-level states
- for observed final states, it treats Notebook `13` `predictions.csv` as authoritative so recovered metrics match the final saved run
- the max-cap case is given a synthetic final state after the 24th reveal because its trace has no explicit no-reveal stop turn

Recovered current Notebook `13` rule:

| Threshold | Value |
|---|---:|
| minimum requests | 1 |
| MLP confidence | `>= 0.70` |
| MLP margin | `>= 0.20` |
| MLP entropy | `<= 0.10` |

Recovered current-rule metrics:

| Metric | Value |
|---|---:|
| Accuracy | 43/49 = 0.878 |
| Top-3 | 0.918 |
| Top-5 | 0.939 |
| Macro-F1 | 0.845 |
| Mean requests | 6.59 |
| Median requests | 5.0 |
| Threshold-fired rate | 36/49 = 0.735 |

Threshold-sweep finding:

- no offline threshold variant improves above `43/49`
- the best tie reaches the same `43/49` and `0.939` top-5 at `6.55` mean requests
- this only saves about `0.04` mean requests relative to Notebook `13`
- the apparent saving comes from stopping two already-easy correct cases one turn earlier
- therefore threshold tuning alone is not a meaningful improvement

Evidence-request findings:

- Notebook `13` requested `323` evidence fields across `49` cases
- top requested evidence fields were broad clinical discriminators:
  - `E_129`: skin lesions/redness/problems, `21` requests
  - `E_151`: swelling, `20` requests
  - `E_201`: cough, `14` requests
  - `E_79`: smoking, `13` requests
  - `E_91`: fever, `11` requests

Request allocation:

| Final correctness | Cases | Mean requests | Median requests | Range |
|---|---:|---:|---:|---|
| Incorrect | 6 | 11.83 | 11.5 | 2-24 |
| Correct | 43 | 5.86 | 5.0 | 0-23 |

Interpretation:

- incorrect cases generally used more evidence, not less
- this means the remaining failures are not mostly caused by too few requests
- some failures are early wrong stops, but several are long wrong trajectories

Prediction-transition finding:

| Head | Wrong-to-correct | Correct-to-wrong | Stable correct | Stable wrong |
|---|---:|---:|---:|---:|
| LLM | 40 | 11 | 79 | 193 |
| Hybrid | 40 | 11 | 79 | 193 |
| MLP | 38 | 10 | 66 | 209 |

Interpretation:

- extra evidence often helps because wrong-to-correct transitions exceed correct-to-wrong transitions
- extra evidence can still hurt because correct-to-wrong transitions exist
- the biggest remaining issue is stable wrong belief, not simply under-questioning

Final takeaway:

- Notebook `13` remains the frozen proposed method
- Notebook `15` supports keeping the current stop thresholds
- the next high-value improvement is not threshold tweaking
- further progress should focus on contradiction handling, hard-case trajectory repair, or better question selection for persistent wrong-belief cases

Report organization added:

- current final reports remain in `reports/`
- baseline reports moved under `reports/baselines/`
- hybrid/sequential policy reports moved under `reports/hybrid/`
- architecture and ledger reports moved under `reports/architecture/`
- planning/claims reports moved under `reports/project/`
- index added at `reports/README.md`

## 48. Algorithmic Evidence Ledger Research Phase

After reviewing MEDDxAgent and the Notebook `15` stop-policy diagnostics, the next planned method improvement is an **algorithmic evidence ledger**.

New research/design notes:

- `research/algorithmic_evidence_ledger_research.md`
- `research/algorithmic_evidence_ledger_design_plan.md`

Core motivation:

- Notebook `13` already has a strong MLP-guided stop rule.
- Notebook `15` showed threshold tuning alone does not improve beyond `43/49`.
- Incorrect cases usually requested more evidence than correct cases, so the remaining bottleneck is not simply "ask more."
- Remaining failures are better described as wrong-belief trajectories, unresolved competing diagnoses, contradiction handling, and diagnostic drift.

Research grounding:

- DDXPlus and AARLC justify evidence acquisition with classifier uncertainty/entropy.
- Active feature acquisition and cost-sensitive classification justify asking only high-value evidence.
- MediQ and ALFA show that LLM question-asking needs structured guidance.
- MEDDxAgent establishes interactive LLM DDx as prior work, so our novelty should be narrower.
- TriMediQ supports the idea that structured intermediate representations improve multi-turn medical reasoning.

Planned ledger concept:

- name: **Evidence-Gated Differential Ledger** (`EGDL`)
- role: deterministic control layer, not just memory
- inputs: visible DDXPlus evidence, partial-evidence MLP belief, LLM differential, train-derived pathology/evidence statistics
- outputs: support/contradiction signals, unresolved-pair flags, drift warnings, candidate action values, and stop certificates

Recommended implementation sequence:

1. build Notebook `16`, an offline algorithmic-ledger analysis over Notebook `13` 49-case traces
2. validate that ledger signals explain or flag current hard cases without API calls
3. only after offline validation, build Notebook `17` as a live confirmation run

Notebook `16` should answer:

- can the ledger flag wrong final stops before they happen?
- do unresolved diagnosis pairs predict errors?
- do contradiction/drift flags explain hard cases?
- can a stop certificate preserve `43/49` while reducing or better justifying requests?
- which high-value evidence fields were missed or delayed?

Notebook `17` should only be run if Notebook `16` produces actionable evidence.

## 49. Graph Algorithmic Ledger Research Update

After further discussion, the ledger direction was refined from a deterministic scoring ledger into a more explicitly **graph-structured algorithmic evidence ledger**.

New research/design files:

- `research/graph_algorithmic_evidence_ledger_prior_work.md`
- `research/graph_algorithmic_evidence_ledger_design.md`

Reason for revision:

- The original "algorithmic ledger" design was useful but too close to tables of deterministic scores.
- The stronger and more defensible version is a patient-specific graph where evidence outcomes, pathologies, candidate requests, and unresolved differential pairs are nodes connected by signed support/contradiction/discriminator edges.
- This better matches the original project ambition around graphs and a real algorithmic ledger.

Important prior-work finding:

- MedKGI is very close to the graph direction: medical KG alignment, information-gain inquiry, structured diagnostic records, and hypothesis-driven termination.
- Therefore the project should not claim generic novelty for "KG + interactive diagnosis."
- The project distinction should be DDXPlus-native graph construction over official evidence root slots, online partial-evidence MLP belief, exact legal evidence reveals, and matched-evidence evaluation.

Planned graph method:

- name: **Graph Evidence Ledger Controller** (`GEL-C`)
- build a global DDXPlus evidence graph from train-derived `P(outcome | pathology)` statistics
- build a dynamic patient episode graph from visible evidence only
- compute signed support and contradiction edges from observed outcomes to active diseases
- construct unresolved disease-pair nodes for top competing diagnoses
- score candidate evidence roots by information gain, pair separation, contradiction probing, graph centrality, split balance, redundancy penalty, and generic penalty
- require a graph stop certificate before final diagnosis

Next notebook should be:

- `notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb`

This notebook should be offline only and should answer whether graph signals explain or flag the current Notebook `13` failures before any new live API run.

## 50. Notebook 16 MedKGI-Style Graph Ledger V1

Notebook `16` was implemented as the first algorithmic graph-ledger version.

Notebook:

- `notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb`

Artifacts:

- `artifacts/graph_algorithmic_ledger/medkgi_style_offline_notebook13_49case_v1/`

Report:

- `reports/algorithmic_ledger/medkgi_style_graph_ledger_v1_report.md`

What was built:

- a DDXPlus-native global evidence graph from the training split only
- pathology nodes, root evidence nodes, and outcome-state nodes represented through exported edge tables
- train-derived outcome probabilities, log-odds support, mutual information, root entropy, and present-rate statistics
- replay of Notebook `13` 49-case traces with visible evidence only
- MedKGI-style candidate scoring using expected information gain, split balance, global MI, and generic/redundancy penalties
- actual-request graph-rank analysis
- offline graph stop-certificate replay
- hard-case graph audits

Key numbers:

| Item | Result |
|---|---:|
| Train rows used for graph stats | `1,025,602` |
| Root evidence fields | `223` |
| Pathologies | `49` |
| Notebook `13` reference accuracy | `43/49 = 0.878` |
| Notebook `13` reference mean requests | `6.59` |

Request-quality result:

| Final outcome | Requests | Mean graph rank | Mean graph score | Mean information gain | Top-10 rate |
|---|---:|---:|---:|---:|---:|
| Incorrect final diagnosis | `71` | `9.73` | `0.233` | `0.194` | `0.676` |
| Correct final diagnosis | `252` | `6.61` | `0.283` | `0.247` | `0.794` |

Interpretation:

- graph scores are meaningful: correct trajectories requested more graph-informative evidence
- failed trajectories often requested lower-ranked evidence and retained high unresolved graph value
- hard cases such as COPD, Croup, Influenza, Pericarditis, and Unstable angina show evidence-trajectory problems, not just stop-threshold problems

Stop-certificate result:

- strict graph thresholds `0.03` through `0.20` did not fire on recorded trajectories
- threshold `0.30` fired on only `6/49` cases
- threshold `1.00` behaves like an MLP-only reference and should not be treated as a meaningful graph gate
- offline replay does not prove that a graph stop certificate improves Notebook `13`

Current conclusion:

Notebook `16` supports a conditional live graph-shortlist pilot, not a stop-policy replacement claim.

Recommended next step:

- build Notebook `17` as a live MedKGI-style graph-shortlist pilot
- keep the Notebook `13` MLP-guided stop rule fixed
- replace only the evidence shortlist with graph top-10 candidates
- compare against Notebook `13` on the same cases, request count, hard-case outcomes, and request graph value

## 51. Notebook 17 Live MedKGI Graph Shortlist Pilot

Notebook `17` was implemented as the first live algorithmic-ledger pilot.

Notebook:

- `notebooks/17_live_medkgi_graph_shortlist_pilot.ipynb`

Report/run guide:

- `reports/algorithmic_ledger/live_medkgi_graph_shortlist_pilot.md`

Dry-run artifact:

- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_dryrun_smoke_v1/`

What changed from Notebook `13`:

- the LLM loop is still single-agent
- the LLM model remains fixed to `gpt-4.1-mini`
- deterministic API settings remain `temperature = 0.0`, `top_p = 1.0`
- the selected MLP stop rule is unchanged
- final heads are unchanged
- only the evidence shortlist changed

New shortlist method:

- loads or rebuilds train-derived graph stats from Notebook `16`
- builds the active differential from MLP top-5, previous LLM differential, deterministic candidates, and the prior anchor
- scores legal unrevealed roots by:

```text
score = penalty * (0.80 * information_gain + 0.15 * split_balance + 0.05 * global_mi)
```

- exposes graph top-10 legal roots to the LLM
- includes graph score, information gain, split balance, weighted present rate, and active diseases in the prompt

Run scopes:

- `RUN_SCOPE = "pilot24"` compares against Notebook `13` 24-case selected-stop run
- `RUN_SCOPE = "final49"` compares against Notebook `13` 49-case selected-stop run
- default is safe: `RUN_LIVE_API = False`, `ALLOW_DRY_RUN_BENCHMARK = True`, `DRY_RUN_MAX_CASES = 2`

Dry-run validation:

- Notebook `17` executed successfully without API calls
- all expected artifact files were written
- `promotion_decision.json` correctly marks the smoke result as `dry_run_smoke_not_for_promotion`

Next action:

- run `pilot24` live first
- only run `final49` if the 24-case live pilot is not clearly worse than Notebook `13`
- evaluate primarily against accuracy, mean requests, hard-case outcomes, and graph request quality

## 52. Notebook 17 Live Pilot Result

Notebook `17` has now been run live on the 24-case pilot scope.

Artifact:

- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/`

Updated report:

- `reports/algorithmic_ledger/live_medkgi_graph_shortlist_pilot.md`

Additional paired artifact:

- `artifacts/graph_algorithmic_ledger/live_medkgi_graph_shortlist_pilot24_v1/notebook13_vs_graph_paired_case_results.csv`

Live result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 |
| Notebook `08` lambda `0.10` LLM-only | 22/24 | 0.917 | 0.917 | 0.917 | 0.846 | 13.04 |
| Notebook `12` offline selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.88 |
| Notebook `17` graph shortlist pilot | 20/24 | 0.833 | 0.833 | 0.875 | 0.744 | 6.21 |

Graph-specific metrics from Notebook `17`:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 1.76 |
| Mean requested information gain | 0.373 |
| Requests outside graph top-10 | 0 |
| Mean graph shortlist size | 10.0 |
| Stop-before-cap rate | 0.958 |
| Cap-hit count | 1 |

Paired outcome against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| Notebook `13` only correct | 2 |
| Notebook `17` only correct | 0 |
| Both wrong | 2 |

Case-level notes:

- `test:51421` Chagas was correct in Notebook `13` but became Sarcoidosis in Notebook `17`
- `test:77908` Ebola was correct in Notebook `13` but became HIV initial infection in Notebook `17`
- `test:81691` Croup remained wrong
- `test:62878` Pericarditis remained wrong and used more requests than Notebook `13`
- Notebook `17` fixed no cases that Notebook `13` got wrong

Promotion decision:

- `reject_keep_notebook13_v1`

Interpretation:

Notebook `17` is a useful negative result. The graph shortlist is mechanically high quality by its own scoring system: requested fields are high-ranked, information-gain values are strong, and no request falls outside the graph top-10. But diagnosis got worse. That means the problem is not whether the graph can rank evidence under an active differential; it is whether the active differential and hard graph pruning are safe enough to control the LLM action space.

The likely failure mode is over-constrained question selection. If the active differential is already biased or missing the true disease family, the graph top-10 efficiently asks questions that separate the wrong candidates. This can reduce request count slightly while harming diagnostic accuracy.

Current project decision:

- Notebook `13` remains the frozen proposed method.
- Do not run Notebook `17` `final49` for this graph-replacement v1.
- If graph work continues, use graph scores as advisory/blended shortlist features rather than a hard replacement shortlist.

Next graph direction, if needed:

```text
Notebook 13 shortlist diversity
+ graph top-ranked discriminative fields
+ rare/critical disease safety roots
+ guard against pruning diseases absent from current active top-k
```

This keeps the algorithmic ledger useful without letting it collapse the action space around a potentially wrong active belief.

## 53. Notebook 18 Graph-Advisory Hybrid Shortlist

Notebook `18` was implemented as the successor to the rejected Notebook `17` graph-replacement shortlist.

Notebook:

- `notebooks/18_graph_advisory_hybrid_shortlist.ipynb`

Report:

- `reports/algorithmic_ledger/graph_advisory_hybrid_shortlist_report.md`

Dry-run artifact:

- `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_dryrun_smoke_v1/`

What changed from Notebook `17`:

- graph no longer hard-replaces the action shortlist
- Notebook `13` shortlist diversity is preserved
- graph information gain becomes an advisory/reranking feature
- rare disease-specific train-derived support can force decisive evidence fields into the candidate pool
- agent stop can be conservatively overridden when the MLP is uncertain and high-value or rare evidence remains

Advisory score:

```text
score =
  0.45 * notebook13_score_norm
+ 0.25 * graph_information_gain_norm
+ 0.20 * disease_specific_support_norm
+ 0.10 * split_balance
```

Rare-support rule:

```text
rare_support_candidate if:
  active disease contains pathology with root outcome log_odds_support >= 5.0
  and root global_present_rate <= 0.05
  and candidate root is legal and unrevealed
```

Why this matters:

- Notebook `17` failed partly because decisive rare fields such as Ebola contact/bleeding and Chagas weight-loss evidence had low global MI
- Notebook `18` explicitly adds disease-specific log-odds support so rare high-specificity evidence is not pruned out
- this tests whether the graph ledger is better as an advisory layer than as a hard controller

Validation completed:

- all code cells parse successfully
- safe dry-run executed top-to-bottom with `nbclient`
- no live API calls were made
- `INTERACTIVE_API_KEY_BOOTSTRAP = False` by default, so dry runs do not prompt
- expected artifacts were written, including:
  - `metrics.json`
  - `predictions.csv`
  - `traces.jsonl`
  - `reference_comparison.csv`
  - `notebook13_vs_notebook18_paired_case_results.csv`
  - `notebook17_vs_notebook18_paired_case_results.csv`
  - `advisory_shortlist_components.csv`
  - `rare_evidence_coverage.csv`
  - `agent_stop_safety_checks.csv`
  - `agent_stop_safety_overrides.csv`
  - `promotion_decision.json`
  - figures under `figures/`

Dry-run metrics are not scientific results. They only prove that the notebook executes and writes the expected artifacts.

Next action:

- run Notebook `18` live on `pilot24`
- do not rerun Notebooks `13` or `17`; Notebook `18` compares against their existing artifacts
- do not run `final49` unless `pilot24` meets the promotion rule

Promotion rule:

- promote if `correct_count >= 22` and mean requests `<= 6.58`
- promote if `correct_count >= 23` and mean requests `<= 8.0`
- reject if `correct_count <= 21`
- reject if it repeats Notebook `17`'s Chagas/Ebola failures

## 54. Notebook 18 Live Pilot Result

Notebook `18` has now been run live on the 24-case pilot slice.

Artifact:

- `artifacts/graph_algorithmic_ledger/live_graph_advisory_hybrid_shortlist_pilot24_v1/`

Updated report:

- `reports/algorithmic_ledger/graph_advisory_hybrid_shortlist_report.md`

Live result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 |
| Notebook `17` hard graph shortlist | 20/24 | 0.833 | 0.833 | 0.875 | 0.744 | 6.21 |
| Notebook `18` graph-advisory shortlist | 21/24 | 0.875 | 0.875 | 0.917 | 0.795 | 7.67 |

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 21 |
| Notebook `13` only correct | 1 |
| Notebook `18` only correct | 0 |
| Both wrong | 2 |

Paired result against Notebook `17`:

| Outcome | Cases |
|---|---:|
| Both correct | 19 |
| Notebook `18` only correct | 2 |
| Notebook `17` only correct | 1 |
| Both wrong | 2 |

Case-level interpretation:

- Notebook `18` recovered Notebook `17`'s Chagas failure: `test:51421` went from Sarcoidosis/wrong to Chagas/correct.
- Notebook `18` recovered Notebook `17`'s Ebola failure: `test:77908` went from HIV initial infection/wrong to Ebola/correct.
- Notebook `18` introduced a new Stable angina failure: `test:16097` went from Stable angina/correct in Notebook `13` and Notebook `17` to Boerhaave/wrong.
- Croup (`test:81691`) and Pericarditis (`test:62878`) remain persistent hard cases.

Graph/advisory diagnostics:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 5.12 |
| Mean requested information gain | 0.242 |
| Mean top graph score at stop | 0.796 |
| Requests outside pure graph top-10 | 17 |
| Agent-stop safety overrides | 8 |
| Cap-hit count | 3 |

The `requests outside pure graph top-10` value is expected because Notebook `18` deliberately restores Notebook `13` diversity and rare-support fields. Unlike Notebook `17`, graph scores no longer hard-control the action space.

Research decision:

- Notebook `18` is rejected under the written promotion rule because it scored only `21/24` and used more requests than Notebook `13`.
- The generated artifact labels the result as conditional review, but the research interpretation should follow the predeclared rule: `correct_count <= 21` means reject.
- Notebook `13` remains the frozen proposed method.
- Notebook `18` is still useful as a diagnostic ablation: graph advisory support can fix some rare-disease failures, but it also adds trajectory complexity and does not yet improve the main method.

Current graph conclusion:

- hard graph replacement is too restrictive
- graph-advisory blending is safer than hard replacement
- neither graph variant beats Notebook `13`
- graph information is currently strongest as an audit/explanation layer, not as the promoted live controller

Recommended next step:

- do not run Notebook `18` on `final49`
- keep Notebook `13` as the main proposed system for final presentation
- use Notebooks `16`, `17`, and `18` as rigorous algorithmic-ledger ablations showing why graph control needs stronger belief-correction logic before promotion

## 55. Bayesian VOI Algorithmic Ledger Research Direction

After Notebook `17` and Notebook `18`, the next proposed algorithmic direction is a Bayesian value-of-information evidence ledger.

New research note:

- `research/bayesian_voi_algorithmic_ledger_research.md`

Updated literature map:

- `research/project_literature_map.md`

Why this direction:

- Notebook `13` is already strong at stopping.
- Notebook `17` and Notebook `18` show that graph ranking alone does not reliably improve question selection.
- The remaining failure mode is wrong belief recovery, not simply early stopping.
- A Bayesian VOI ledger directly maintains a posterior over all `49` diagnoses and asks evidence questions based on expected posterior improvement.

Core idea:

```text
visible evidence
-> Bayesian posterior over all DDXPlus pathologies
-> support and contradiction accounting
-> expected value of each legal unrevealed evidence field
-> request field if value exceeds evidence cost
-> stop only when posterior, MLP, and remaining VOI agree
```

Research basis:

- Bayesian medical inquiry and QMR-style disease inference
- active diagnosis via entropy/information-gain test selection
- cost-sensitive feature acquisition and classification
- AARLC-style entropy alignment between inquiry and classifier
- MedKGI-style information-guided inquiry, but with posterior-level VOI instead of graph-edge-only ranking

What we have to work with:

- DDXPlus train split for likelihood tables:
  - `P(evidence_outcome | pathology)`
  - disease priors
  - log-likelihood ratios
  - root mutual information
  - rare disease-specific support
- DDXPlus test environment:
  - hidden evidence can be revealed only when requested
  - this allows an offline deterministic VOI agent with no API calls
- existing MLP artifacts:
  - initial-evidence one-shot
  - full-evidence ceiling
  - partial-evidence policy-shaped MLP
- existing traces:
  - Notebook `13` 49-case traces
  - Notebook `17`/`18` graph failure traces

Recommended next notebook:

```text
notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb
```

Notebook `19` should be offline-only:

- build train-derived Bayesian likelihood tables
- calibrate posterior on validation
- replay Notebook `13` traces with Bayesian posterior/VOI diagnostics
- run a deterministic Bayesian VOI agent that requests evidence from the DDXPlus environment without using test labels
- compare Bayes-only, MLP-only, and fused final heads
- audit hard cases: Croup, Pericarditis, Stable angina, Chagas, Ebola

Promotion condition:

- do not make a live Notebook `20` until Notebook `19` shows a real signal
- a real signal means matching or beating Notebook `13`, improving persistent hard cases, or reducing requests at matched accuracy

Interpretation:

This is the strongest next mathematical direction. It does not claim Bayesian/VOI diagnosis is new. The project-specific contribution would be a DDXPlus-native Bayesian VOI ledger fused with the BASD-style partial-evidence MLP and optionally used to constrain the LLM evidence-acquisition agent.

## 56. Notebook 19 Bayesian VOI Algorithmic Ledger Offline

Notebook `19` has been implemented as the next offline algorithmic-ledger candidate.

New notebook:

- `notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb`

New report:

- `reports/algorithmic_ledger/bayesian_voi_ledger_offline_report.md`

Artifact roots:

- smoke: `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1_smoke/`
- full: `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1/`

What changed:

- built train-only Bayesian disease/evidence likelihood tables
- added disease priors, age/sex likelihoods, root outcome likelihoods, root mutual information, and root log-odds support
- implemented a `BayesianEvidenceLedger` that tracks visible evidence, requested evidence, legal roots, posterior state, contradiction score, and request history
- fused Bayesian posterior with the selected partial-evidence MLP posterior using fixed log-linear pooling
- selected evidence by one-step value of information over legal unrevealed DDXPlus evidence roots
- added a stop certificate requiring confidence, margin, entropy, low remaining VOI, low contradiction, and Bayes/MLP/fused agreement
- saved posterior calibration, policy sweep, traces, candidate scores, hard-case audits, comparison rows, and figures

VOI utility:

```text
utility =
  0.55 * expected_fused_entropy_reduction
+ 0.20 * expected_margin_gain
+ 0.15 * contradiction_resolution_gain
+ 0.10 * rare_recovery_bonus
- lambda_cost
- redundancy_penalty
```

Stop certificate:

```text
stop if:
  fused_confidence >= 0.70
  fused_margin >= 0.20
  fused_entropy <= 0.35
  max_remaining_utility <= 0
  contradiction_score <= 1.5
  min_requests >= 1
  Bayes/MLP/fused agreement is acceptable
```

Validation completed:

- static code-cell parse passed
- source notebook has no outputs
- no API adapter, `LLM_API_KEY`, or live HTTP call path exists
- smoke execution completed top-to-bottom with `BAYESIAN_VOI_SMOKE_MODE=1`
- smoke artifacts and figures were written successfully

Smoke metrics:

| Metric | Value |
|---|---:|
| Cases | 3 |
| Fused correct | 2/3 |
| Fused accuracy | 0.667 |
| Mean requests | 2.67 |
| Stop-before-cap rate | 0.667 |
| Cap-hit count | 1 |

Full 49-case result:

| Lambda | Fused correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Cap hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 33/49 | 0.673 | 0.837 | 0.878 | 0.620 | 22.37 | 28 |
| 0.02 | 26/49 | 0.531 | 0.735 | 0.837 | 0.461 | 7.98 | 4 |
| 0.05 | 26/49 | 0.531 | 0.633 | 0.776 | 0.447 | 5.65 | 1 |
| 0.10 | 25/49 | 0.510 | 0.653 | 0.776 | 0.424 | 4.43 | 1 |
| 0.15 | 24/49 | 0.490 | 0.633 | 0.776 | 0.396 | 4.33 | 1 |

Reference:

| System | Correct | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 43/49 | 0.878 | 0.939 | 6.59 |
| Notebook `19` best fused, lambda `0.00` | 33/49 | 0.673 | 0.878 | 22.37 |
| Notebook `19` near-budget, lambda `0.02` | 26/49 | 0.531 | 0.837 | 7.98 |

Paired result against Notebook `13` at lambda `0.00`:

| Outcome | Cases |
|---|---:|
| Both correct | 32 |
| Notebook `19` only correct | 1 |
| Notebook `13` only correct | 11 |
| Both wrong | 5 |

The one Notebook `19` fix over Notebook `13` was `test:125508` Unstable angina, where Notebook `13` predicted Anemia and Notebook `19` predicted Unstable angina. This did not offset the eleven regressions.

Main failure interpretation:

- Notebook `19` is not a close miss; it is a negative algorithmic ablation.
- Lambda `0.00` asked far more evidence than Notebook `13` but still performed much worse, so the main issue is not early stopping.
- Higher lambdas reduced requests but collapsed accuracy further.
- The VOI policy over-requested generic/global roots such as palpitations, cough, sweating, fatigue, asthma history, obesity, and GERD.
- The partial-evidence MLP became overconfident on VOI-generated evidence subsets that differ from the trace distribution it was trained on.
- Bayes-only at lambda `0.00` reached `0.714`, better than fused/MLP, but still far below Notebook `13`.
- The Bayesian naive-independence model is useful as an audit signal, but not strong enough as a replacement controller.

Promotion decision:

- `do_not_promote_yet`
- do not create a live Notebook `20` from this version
- keep Notebook `13` as the frozen proposed method
- use Notebook `19` as evidence that posterior-level VOI is not automatically enough unless it is calibrated to the same evidence-trajectory distribution as the partial-evidence classifier

Next possible direction:

- if continuing algorithmic work, use Bayesian VOI as an advisory feature inside the Notebook `13` shortlist rather than a replacement controller
- alternatively, train a partial-evidence MLP on VOI-generated trajectories before trusting MLP feedback on those trajectories

## 57. Algorithmic Ledger Reset: LLM-Led Graph Ledger Context

After reviewing the intent of the project, the algorithmic-ledger direction has been reset.

New research folder:

- `research/llm_led_algorithmic_ledger_v2/`

New files:

- `research/llm_led_algorithmic_ledger_v2/README.md`
- `research/llm_led_algorithmic_ledger_v2/01_prior_work.md`
- `research/llm_led_algorithmic_ledger_v2/02_design_plan.md`
- `research/llm_led_algorithmic_ledger_v2/03_notebook20_plan.md`

Reason for reset:

- Notebooks `17`, `18`, and `19` tested graph/Bayesian/VOI logic mostly as replacement controllers or heavy shortlist controllers.
- That was not the intended architecture.
- The intended architecture is that the LLM remains the question chooser while the graph/algorithmic ledger provides structured case understanding.

Corrected architecture:

```text
DDXPlus environment
-> deterministic evidence ledger
-> graph evidence ledger updates support/contradiction/unresolved pairs
-> prompt compiler gives compact graph state to LLM
-> LLM chooses next evidence request
-> DDXPlus reveals answer
-> partial-evidence MLP evaluates stop readiness
-> repeat
```

What the graph ledger should do:

- track which revealed evidence supports each active diagnosis
- track which revealed evidence contradicts each active diagnosis
- identify unresolved competing diagnosis pairs
- suggest missing discriminators as advisory context
- warn when LLM, MLP, and graph support disagree
- improve state understanding and trace explainability

What it should not do:

- replace the LLM as the question chooser
- hard-filter the action space to graph top-k fields
- force Bayesian/VOI actions
- override the LLM except for legality or schema repair

Online research consulted:

- DDXPlus NeurIPS 2022
- MedKGI
- Think-on-Graph
- Dr.Knows / KG-augmented diagnosis prediction
- MEDDxAgent
- KoSEL
- MedKA
- MDAgents
- KG4Diagnosis

Main research conclusion:

The corrected direction is closer to knowledge-graph-augmented LLM reasoning than active feature acquisition. The graph ledger should be an intelligent memory/reasoning substrate, not a replacement policy.

Recommended next notebook:

```text
notebooks/20_llm_led_graph_ledger_context.ipynb
```

Notebook `20` should adapt Notebook `13`, not Notebook `17`, `18`, or `19`.

Fixed first implementation:

- keep `gpt-4.1-mini`
- keep Notebook `13` LLM-led question loop
- keep Notebook `13` base action menu
- keep Notebook `13` partial-evidence MLP stop rule
- add one compact `GRAPH LEDGER CONTEXT` prompt section
- evaluate first on the same 24-case pilot slice

Promotion logic:

- promote to 49 cases only if pilot24 does not regress against Notebook `13`
- success can be better accuracy, same accuracy with fewer requests, or same accuracy with materially better hard-case trace coherence

Interpretation:

This resolves the mismatch between the user's intended algorithmic ledger and the last few algorithmic experiments. The failed notebooks remain useful as negative ablations, but they should not be treated as tests of the intended graph-ledger support architecture.

## 58. Notebook 20 Implemented: LLM-Led Graph Ledger Context

Implemented:

- `notebooks/20_llm_led_graph_ledger_context.ipynb`
- `reports/algorithmic_ledger/llm_led_graph_ledger_context_report.md`

Artifact root for default dry-run:

- `artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_dryrun_smoke_v1/`

Artifact root for live pilot:

- `artifacts/graph_algorithmic_ledger/llm_led_graph_ledger_context_pilot24_v1/`

Core implementation decision:

- Notebook `20` was cloned from Notebook `13`, not from Notebooks `17`, `18`, or `19`.
- The LLM remains the evidence-request chooser.
- The Notebook `13` deterministic action menu remains in place.
- The Notebook `13` MLP-guided stop rule remains unchanged.
- The graph ledger is used only as prompt context.

New graph context components:

- active differential from MLP, previous LLM differential, deterministic state, and prior
- support score for each active diagnosis from revealed evidence
- contradiction score for each active diagnosis from revealed evidence
- unresolved diagnosis pairs
- advisory discriminator fields
- consistency warnings when LLM, MLP, and graph support disagree

New output files:

- `graph_context_by_turn.csv`
- `warning_resolution_summary.csv`
- `hard_case_graph_context_audit.json`
- `promotion_decision.json`

New prediction columns:

- `num_graph_warnings_final`
- `num_unresolved_pairs_final`
- `final_top_diagnosis_support`
- `final_top_diagnosis_contradiction`
- `warnings_resolved_count`
- `graph_context_tokens_estimate`

Validation completed:

- static parse passed for all Notebook `20` code cells
- dry-run executed top-to-bottom with no API key
- default notebook state is safe:
  - `RUN_LIVE_API = False`
  - `ALLOW_DRY_RUN_BENCHMARK = True`
  - `RUN_SCOPE = "pilot24"`
  - `DRY_RUN_MAX_CASES = 2`
- all required dry-run artifacts were written

Dry-run smoke metrics:

| Metric | Value |
|---|---:|
| Cases | 2 |
| Accuracy | 1.000 |
| Top-5 | 1.000 |
| Mean requests | 6.500 |
| Stop-before-cap rate | 1.000 |

Important interpretation:

- These dry-run numbers are not scientific evidence.
- They only prove the notebook path works: prompt construction, graph context, traces, predictions, and artifact writing.

Live `pilot24` result:

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook `20` LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

Promotion decision:

- Notebook `20` is not promoted to `final49` because top-1 accuracy dropped.
- The `23/24` top-3/top-5 result is still useful because it shows graph context can improve ranking quality.

Current recommendation:

- Use Notebook `21` to test whether graph context can act as a critic/adjudicator.
- Keep Notebook `13` frozen as the proposed method unless a graph-context variant beats it live.

## 59. Notebook 21 Implemented: Graph-Context Policy Lab

Implemented:

- `notebooks/21_graph_context_policy_lab.ipynb`
- `reports/algorithmic_ledger/graph_context_policy_lab_report.md`

Artifact root:

- `artifacts/graph_algorithmic_ledger/graph_context_policy_lab_24case_v1/`

Purpose:

Notebook `21` was added after live Notebook `20` produced a mixed result. Notebook `20` degraded top-1 accuracy versus Notebook `13`, but improved ranking quality to `23/24` top-3 and top-5 on the 24-case pilot. Notebook `21` asks whether graph-ledger information can be used as a critic, guardrail, or adjudicator to convert that top-5 signal into better top-1 accuracy without new API calls.

Important implementation boundaries:

- offline-only
- no API calls
- no model training
- no changes to Notebook `13` or Notebook `20`
- labels are used only for evaluation and oracle upper-bound analysis
- non-oracle variants use only visible trace evidence, ranked differentials, MLP feedback, and reconstructed graph context

Inputs compared:

| Run | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 0.917 | 0.917 | 0.917 | 6.58 |
| Notebook `17` hard graph shortlist | 0.833 | 0.833 | 0.875 | 6.21 |
| Notebook `18` graph-advisory shortlist | 0.875 | 0.875 | 0.917 | 7.67 |
| Notebook `20` LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

Experiments run:

- graph feature diagnostics
- top-3/top-5 graph adjudication rules
- LLM/MLP consensus adjudicators
- stop guard flagging and replay where future turns existed
- drift guards for stable graph-supported diagnoses
- combined graph guard/adjudication variants
- oracle top-3/top-5 upper bounds for Notebook `20`

Main result:

- No non-oracle policy variant beat Notebook `13`.
- The best selected experimental variant was `drift_guard_notebook13_tc1.0_delta0.5`.
- It matched Notebook `13` at `22/24 = 0.917` accuracy and `6.58` mean requests, but changed one wrong case into another wrong case.
- Therefore it is not a real improvement and should not be promoted.

Oracle result:

| Variant | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `20` oracle top-3 | 0.958 | 0.958 | 0.958 | 6.13 |
| Notebook `20` oracle top-5 | 0.958 | 0.958 | 0.958 | 6.13 |

Interpretation:

- Notebook `20`'s top-5 improvement is a real signal, not just noise.
- The graph context plus LLM trace often contains the correct answer in the ranked differential.
- The current problem is adjudication: choosing the right top-1 from the ranked set without using labels.

Graph diagnostic signal:

For Notebook `20`, wrong final top-1 predictions had much higher graph contradiction than correct predictions.

| Feature | Correct mean | Wrong mean | Wrong - correct |
|---|---:|---:|---:|
| Top contradiction | 0.377 | 4.015 | +3.638 |
| Top contradiction minus support | -6.964 | 0.906 | +7.870 |
| Top net support | 6.964 | -0.906 | -7.870 |

This is useful because it means the graph ledger can identify suspect answers. It is not enough by itself because the graph threshold rules still cannot reliably choose the correct alternative.

Selection decision:

```text
no_promotable_candidate
```

Recommendation:

- Do not create a live Notebook `22` from the current hand-threshold graph adjudicators.
- Keep Notebook `13` as the frozen proposed method.
- Treat Notebook `21` as evidence that graph context is useful as an audit/critic signal.
- If continuing graph-ledger work, the next credible direction is a calibrated or learned adjudicator trained on development traces, not more hand-picked graph thresholds.

Current interpretation:

Notebook `21` strengthens the project scientifically even though it does not produce a new method. It shows that the graph ledger has diagnostic signal, but also prevents overclaiming. The defensible claim is now:

> Graph-ledger context improves ranking signal and flags suspect predictions, but simple rule-based graph adjudication is not yet enough to outperform the frozen Notebook `13` hybrid sequential method.

## 60. Handoff Decision After Notebook 21 Artifact Inspection

Inspected:

- `README.md`
- `PROJECT_WORKLOG.md`
- `reports/final_results_summary.md`
- `reports/final_report.md`
- latest reports under `reports/algorithmic_ledger/`
- `artifacts/graph_algorithmic_ledger/graph_context_policy_lab_24case_v1/`

Key artifact checks:

- Notebook `13` 24-case pilot cases are all contained in the Notebook `13` 49-case confirmation: `24/24` overlap.
- Notebook `20` graph-context pilot uses the same 24 cases as Notebook `13` pilot.
- Notebook `21` has `96` final states across Notebooks `13`, `17`, `18`, and `20`, but these are four correlated runs over the same `24` cases, not independent train/dev examples.
- Notebook `21` tested `614` variant summaries and selected no promotable live candidate.
- The best non-oracle variant only matched Notebook `13`; it did not produce a real improvement.

Decision:

- Do not create Notebook `22` for a learned or calibrated graph adjudicator from the current traces alone.
- The current data are useful for critic/audit analysis, but they are not sufficient for a defensible held-out learned adjudicator because the only graph-context ranking trace is the same 24-case pilot slice.
- Keep Notebook `13` as the frozen proposed method for the course deliverable.
- Treat Notebooks `16-21` as diagnostic and negative algorithmic-ledger ablations that sharpen the future-work story.

Report hygiene completed:

- Updated `README.md` current practical recommendation to make the train/dev limitation explicit.
- Updated `reports/final_report.md` with the 2026-05-08 handoff decision.
- Updated `reports/final_results_summary.md` so its artifact list and headline table include the latest graph/VOI ledger runs.

Next likely step:

- Finalize the written report/presentation around Notebook `13`.
- Only revisit graph adjudication if separate graph-context development traces are created or reserved for strict validation.

## 61. Notebook 22 Implemented: Graph Posterior Final Adjudicator

Implemented:

- `notebooks/22_graph_posterior_final_adjudicator.ipynb`
- `reports/algorithmic_ledger/graph_posterior_final_adjudicator_report.md`

Artifact root:

- `artifacts/graph_algorithmic_ledger/graph_posterior_final_adjudicator_49case_v1/`

Purpose:

Notebook `22` tests the graph ledger in a different role from Notebooks `17-21`. It does not change the Notebook `13` evidence-acquisition trajectory, does not call an API, and does not use graph information as a live question controller. Instead, it reconstructs the final visible evidence state from Notebook `13` traces and computes a train-derived graph posterior over all `49` pathologies.

Primary graph score:

```text
graph_score(disease) =
  sum over revealed evidence outcomes:
    clip(log_odds_support(outcome -> disease), -3, 3)
```

Selected conservative critic:

```text
override Notebook 13 top-1 only if:
  graph_top1 differs from Notebook 13 top-1
  graph_margin >= 1.0
  graph_score(Notebook 13 top-1) < 0
  graph_score(graph_top1) > 0
```

Validation completed:

- static parse passed for all Notebook `22` code cells
- notebook code cells executed top-to-bottom with no API key and no live API path
- Notebook `13` reference metrics were recovered exactly:
  - `43/49 = 0.878` accuracy
  - `0.918` top-3
  - `0.939` top-5
  - `6.59` mean requests
- all expected artifacts were written

Main 49-case result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 43/49 | 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Graph-only final head | 44/49 | 0.898 | 0.959 | 0.980 | n/a | 6.59 |
| Notebook `22` conservative graph critic | 44/49 | 0.898 | 0.939 | 0.939 | 0.867 | 6.59 |

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 43 |
| Notebook `22` only correct | 1 |
| Notebook `13` only correct | 0 |
| Both wrong | 5 |

The only changed prediction:

| Case | True pathology | Notebook `13` | Notebook `22` | Result |
|---|---|---|---|---|
| `test:81691` | Croup | Anemia | Croup | fixed |

Promotion decision:

```text
offline_candidate_promoted
```

Interpretation:

- Notebook `22` is the first graph-ledger variant to improve over Notebook `13` on the 49-case artifact.
- The improvement comes from using graph information as a final-state mathematical critic, not as a replacement evidence controller.
- Notebook `13` remains the live evidence-acquisition method.
- Notebook `22` should be described as an offline final-head enhancement candidate until confirmed on held-out or live traces.

Post-run analysis:

- The selected critic fired exactly once on the 49-case run.
- The single override fixed `test:81691` Croup:
  - Notebook `13`: Anemia, graph score `-2.359`
  - graph top-1: Croup, graph score `1.177`
  - graph margin: `2.073`
- The 24-case sanity slice showed the same pattern, improving from `22/24` to `23/24` by fixing Croup while leaving Pericarditis wrong.
- The remaining 49-case errors after Notebook `22` are COPD/Myocarditis, acute-vs-chronic rhinosinusitis, Influenza/HIV initial infection, Pericarditis/Anemia, and Unstable angina/Anemia.
- The most useful technical lesson is that graph support is credible as a final-state critic but still not sufficient as a standalone final head or controller. In COPD, for example, the graph correctly distrusted Notebook `13`'s Myocarditis answer but preferred the wrong alternative, so the conservative abstention rule prevented a regression.

Report/doc updates completed:

- added the Notebook `22` report under `reports/algorithmic_ledger/`
- updated `README.md`
- updated `reports/README.md`
- updated `reports/final_results_summary.md`
- updated `reports/final_report.md`

Next likely step:

- The Croup-only gain is useful but not ambitious enough by itself. A result that changes the project should aim for at least `47/49`.
- The post-run rescue ceiling supports that target:
  - Notebook `13` top-1 or graph top-1 oracle: `44/49`
  - Notebook `13` top-1 or graph top-2 oracle: `46/49`
  - Notebook `13` top-1 or graph top-3 oracle: `47/49`
  - Notebook `13` top-1 or graph top-5 oracle: `48/49`
  - Notebook `13` top-1 or union of LLM/MLP/graph top-5 oracle: `48/49`
- The next credible notebook should therefore be a calibrated graph/LLM/MLP final reranker or a graph-triggered rescue continuation, not another hand threshold over the same six errors.
- The rule must be trained/calibrated from train/validate-derived partial evidence states or fresh held-out traces. Directly tuning on the 49-case errors would not be defensible.

## 62. Notebook 23 Implemented: Calibrated Graph-Bayes Rescue Reranker

Implemented:

- `notebooks/23_calibrated_graph_bayes_rescue_reranker.ipynb`
- `reports/algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md`

Artifact root:

- `artifacts/graph_algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_49case_v1/`

Purpose:

Notebook `23` tests the larger graph-ledger opportunity identified after Notebook `22`: can graph top-2/top-3 rescue signal be converted into a defensible result near `47/49` without tuning directly on the six Notebook `13` misses?

Method:

- keep Notebook `13` as the first-pass live workup trace
- generate `8,000` train and `4,000` validate synthetic partial evidence states from DDXPlus train/validate splits
- train an L2 logistic candidate reranker over graph, Bayes, MLP, prior, and rank features
- select a fixed policy named `calibrated_graph_bayes_rescue_v1`
- apply three certificates:
  - prior recovery
  - Notebook `22` conservative graph critic
  - graph/Bayes rescue continuation for suspicious early `agent_stop` cases
- after rescue continuation, select the final candidate with the trained L2 reranker, using graph rank only as a near-tie breaker and graph support as an accept guard

Validation completed:

- static parse passed for all Notebook `23` code cells
- notebook code cells executed top-to-bottom with no API key and no live API path
- Notebook `13` reference metrics were recovered exactly:
  - `43/49 = 0.878` accuracy
  - `0.918` top-3
  - `0.939` top-5
  - `6.59` mean requests
- Notebook `22` reference metrics were recovered:
  - `44/49 = 0.898` accuracy
  - `0` regressions against Notebook `13`
- all expected artifacts were written

Main 49-case result:

| System | Correct | Accuracy | Mean requests | Extra requests | Improvements | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 43/49 | 0.878 | 6.59 | 0 | 0 | 0 |
| Notebook `22` graph critic | 44/49 | 0.898 | 6.59 | 0 | 1 | 0 |
| Notebook `23` graph-Bayes rescue | 47/49 | 0.959 | 6.96 | 18 total | 4 | 0 |

Fixed Notebook `13` misses:

| Case | True pathology | Notebook `13` | Notebook `23` | Source |
|---|---|---|---|---|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | Acute COPD exacerbation / infection | prior recovery |
| `test:81691` | Croup | Anemia | Croup | graph critic |
| `test:8666` | Influenza | HIV initial infection | Influenza | rescue rerank |
| `test:125508` | Unstable angina | Anemia | Unstable angina | rescue rerank |

Remaining errors:

- `test:111176`: Acute rhinosinusitis predicted as Chronic rhinosinusitis
- `test:62878`: Pericarditis predicted as Anemia

Promotion decision:

```text
offline_candidate_promoted
```

Interpretation:

- Notebook `23` is now the strongest algorithmic-ledger result.
- It reaches the `47/49` target with only a small evidence-cost increase, from `6.59` to `6.96` mean requests.
- Notebook `13` remains the live-confirmed acquisition backbone.
- Notebook `23` should be presented as the offline graph/Bayes rescue enhancement candidate; later Notebook `24` tested it live and did not promote it.

## 63. Notebook 24 Implemented: Live Graph-Bayes Rescue Confirmation

Implemented:

- `notebooks/24_live_graph_bayes_rescue_confirmation.ipynb`
- `reports/algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md`

Dry-run artifact root:

- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_dryrun_smoke_v1/`

Live artifact root:

- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1/`

Purpose:

Notebook `24` is the live-confirmation wrapper for the Notebook `23` graph/Bayes rescue candidate. It keeps the Notebook `13` LLM-led evidence acquisition loop unchanged, then applies the frozen Notebook `23` rescue policy after the base stop.

Implementation:

- cloned the Notebook `13` live workup backbone
- changed the artifact family to `artifacts/graph_algorithmic_ledger/`
- set safe defaults:
  - `RUN_LIVE_API = False`
  - `ALLOW_DRY_RUN_BENCHMARK = True`
  - `DRY_RUN_MAX_CASES = 2`
- kept fixed live settings:
  - `gpt-4.1-mini`
  - `temperature = 0.0`
  - `top_p = 1.0`
  - `MAX_REQUEST_CAP = 24`
- rebuilt the Notebook `23` calibrated L2 reranker from its train/validate synthetic feature artifact
- loaded Notebook `16` graph statistics and Notebook `19` Bayesian likelihoods
- applied the frozen Notebook `23` certificates:
  - prior recovery
  - conservative graph critic
  - graph/Bayes rescue continuation for suspicious early `agent_stop`
  - post-rescue reranker accept guard

Dry-run validation:

| Check | Result |
|---|---|
| Static parse | passed |
| Dry-run execution | passed |
| Cases | 2 |
| Base Notebook `13` dry-run accuracy | 2/2 |
| Rescue dry-run accuracy | 2/2 |
| Extra rescue requests | 0 |
| Required artifacts | written |
| Source notebook outputs | stripped |

Interpretation:

- The dry-run is not scientific evidence.
- It confirms that the live Notebook `13` backbone, trace reconstruction, frozen Notebook `23` rescue policy, reranker scoring, and artifact writing all work in one notebook.
- Notebook `24` was ready for the paid 49-case live confirmation run; section 64 records the completed live result.

How to run live:

1. Open `notebooks/24_live_graph_bayes_rescue_confirmation.ipynb`.
2. Set `RUN_LIVE_API = True`.
3. Set `ALLOW_DRY_RUN_BENCHMARK = False`.
4. Keep `SEQUENTIAL_MAX_CASES = 49`.
5. Run top-to-bottom with the API key provided through the interactive prompt or `LLM_API_KEY`.

Promotion criteria:

- strong confirmation: at least `46/49` correct with zero regressions versus the live Notebook `13` base prediction
- ideal confirmation: `47/49` correct with mean total requests `<= 7.25`

Current status:

```text
implemented_dryrun_validated_then_live_completed_not_promoted
```

## 64. Notebook 24 Live Result: Rescue Not Promoted, Base Workup Stronger Than Expected

Live artifact root:

- `artifacts/graph_algorithmic_ledger/live_graph_bayes_rescue_confirmation_49case_v1/`

Reports updated:

- `reports/algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md`
- `reports/algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md`
- `reports/final_results_summary.md`
- `reports/final_report.md`
- `reports/project/project_direction_and_claims_assessment.md`
- `README.md`

Live result:

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Original Notebook `13` artifact | 43/49 | 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Notebook `23` offline rescue candidate | 47/49 | 0.959 | n/a | n/a | n/a | 6.96 |
| Notebook `24` fresh live base workup | 45/49 | 0.918 | 0.939 | 0.939 | 0.895 | 6.20 |
| Notebook `24` live graph/Bayes rescue | 45/49 | 0.918 | 0.939 | 0.939 | 0.895 | 6.39 |

Notebook `24` did not promote the graph/Bayes rescue layer:

- improvements versus the fresh live base: `0`
- regressions versus the fresh live base: `0`
- changed predictions: `1`
- extra rescue requests: `9`
- promotion status: `not_promoted`

The one changed prediction was `test:76022` Panic attack:

- fresh live base predicted Anaphylaxis
- rescue changed it to PSVT through the prior-recovery certificate
- both were wrong, so it was neither an improvement nor a regression

The rescue continuation spent extra evidence on three already-correct cases and then abstained:

| Case | True pathology | Extra requested roots | Outcome |
|---|---|---|---|
| `test:58986` | Acute laryngitis | `E_194`, `E_66`, `E_190` | stayed correct |
| `test:90978` | Bronchiolitis | `E_181`, `E_66`, `E_4` | stayed correct |
| `test:38202` | Inguinal hernia | `E_53`, `E_220`, `E_166` | stayed correct |

The important positive result is the fresh live base trajectory:

- original Notebook `13` artifact: `43/49`, `6.59` mean requests
- fresh Notebook `13`-style base inside Notebook `24`: `45/49`, `6.20` mean requests

Changed outcomes between the original Notebook `13` artifact and the Notebook `24` live base:

| Case | True pathology | Original Notebook `13` | Notebook `24` live base | Result |
|---|---|---|---|---|
| `test:125508` | Unstable angina | Anemia | Unstable angina | fixed |
| `test:81691` | Croup | Anemia | Croup | fixed |
| `test:8666` | Influenza | HIV (initial infection) | Influenza | fixed |
| `test:76022` | Panic attack | Panic attack | Anaphylaxis | regression |
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | Anemia | still wrong |

Remaining Notebook `24` live rescue errors:

- `test:111176`: Acute rhinosinusitis predicted as Chronic rhinosinusitis
- `test:38475`: Acute COPD exacerbation / infection predicted as Anemia
- `test:62878`: Pericarditis predicted as Anemia
- `test:76022`: Panic attack predicted as PSVT

Interpretation:

- Notebook `23` remains the strongest offline graph/Bayes enhancement candidate, but it is not live-confirmed.
- Notebook `24` strengthens the base architecture claim: LLM-led evidence acquisition plus MLP-guided stopping appears robust in the `43-45/49` range with roughly `6-6.6` requests.
- The offline rescue did not transfer because the fresh live trajectory changed the failure pattern. Cases that Notebook `23` fixed offline, such as Croup, Influenza, and Unstable angina, were already fixed by the fresh live base; a new Panic attack regression appeared instead.
- Future graph/Bayes work should focus on live-trajectory calibration and trigger selectivity, not more broad controller replacements.

Current project position:

- defend Notebook `13` / Notebook `24` base as the live evidence-efficient workup method
- present Notebook `23` as a strong offline mathematical enhancement
- present Notebook `24` as a useful live confirmation attempt that did not promote the rescue layer
- do not claim the graph/Bayes rescue is a live-confirmed improvement

## 65. Notebook 25 Implemented: Live Base Trajectory Replicates

Implemented:

- `notebooks/25_live_base_trajectory_replicates.ipynb`
- `reports/algorithmic_ledger/live_base_trajectory_replicates_report.md`

Purpose:

Notebook `25` collects repeated Notebook `13`-style base trajectories so a later offline branching lab can study live LLM divergence without graph/Bayes rescue confounding the measurement.

Design:

- cloned the Notebook `24` live base workup shell
- removed the Notebook `23`/`24` graph-Bayes rescue intervention
- keeps the Notebook `13` selected-stop evidence acquisition loop and MLP-guided stopping
- runs three replicate roots from one notebook:
  - `replicate_r01`
  - `replicate_r02`
  - `replicate_r03`
- writes parent-level aggregate files:
  - `replicate_summary.csv`
  - `replicate_case_predictions.csv`
  - `replicate_case_stability.csv`
  - `replicate_stability_summary.json`

Dry-run validation:

| Check | Result |
|---|---|
| Static parse | passed |
| Dry-run execution | passed |
| Replicates | 3 |
| Dry-run cases per replicate | 2 |
| Artifact contract | passed |
| Live API use | none |

Dry-run artifact root:

- `artifacts/trajectory_replicates/notebook13_style_live_base_replicates_dryrun_smoke_v1/`

Live target artifact root:

- `artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1/`

How to run live:

1. Open `notebooks/25_live_base_trajectory_replicates.ipynb`.
2. Set `RUN_LIVE_API = True`.
3. Set `ALLOW_DRY_RUN_BENCHMARK = False`.
4. Keep `REPLICATE_IDS = ["r01", "r02", "r03"]`.
5. Run top-to-bottom.

Interpretation:

This is not a new method result yet. It is data collection for the proposed branching trajectory lab. The key constraint is that all future branch-trigger features must be derived from current visible state, not from knowing that a specific test case diverged historically.

## 66. Notebook 26 Implemented: Offline Branching Trajectory Lab

Implemented:

- `notebooks/26_offline_branching_trajectory_lab.ipynb`
- `reports/algorithmic_ledger/offline_branching_trajectory_lab_report.md`

Artifact root:

- `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/`

Inputs:

- Notebook `13` frozen 49-case artifact
- Notebook `24` fresh live base trajectory
- Notebook `25` live base replicates `r01`, `r02`, and `r03`
- Notebook `16` train-derived graph edges
- Notebook `19` Bayesian likelihood tables

Validation:

| Check | Result |
|---|---|
| Static parse | passed |
| Top-to-bottom execution | passed through a local notebook runner |
| Live API calls | none |
| Required artifacts | written |

Observed base trajectory accuracy:

| Run | Correct | Accuracy | Mean requests |
|---|---:|---:|---:|
| Notebook `13` frozen | 43/49 | 0.878 | 6.59 |
| Notebook `24` base | 45/49 | 0.918 | 6.20 |
| Notebook `25` r01 | 44/49 | 0.898 | 7.00 |
| Notebook `25` r02 | 42/49 | 0.857 | 7.02 |
| Notebook `25` r03 | 42/49 | 0.857 | 6.82 |

Divergence findings:

- same final prediction across all five trajectories: `41/49`
- prediction instability cases: `8/49`
- correctness instability cases: `7/49`
- majority-vote accuracy across the five trajectories: `43/49`
- oracle best-of-five accuracy: `47/49`
- Notebook `13` misses with at least one alternate correct trajectory: `4`
- same-prefix divergent states: `64`
- same-prefix divergent states with downstream correctness instability: `10`

The result strongly supports the user's branching hypothesis but rejects naive voting. The useful signal is not that more agents automatically help; it is that a small number of fragile states create alternate trajectories, and a graph/Bayes/MLP judge can sometimes select the right branch.

Best diagnostic Notebook `13` branch policy:

```text
trigger=hybrid_suspicion_v1
branch_budget=2
judge=highest_bayes_posterior
```

Notebook `13` scope result:

| Metric | Value |
|---|---:|
| Accuracy | 47/49 = 0.959 |
| Base accuracy | 43/49 = 0.878 |
| Wins versus base | 4 |
| Regressions versus base | 0 |
| Branch trigger rate | 0.184 |
| Mean branches spawned | 0.367 |
| Mean selected requests | 6.86 |
| Mean total branch requests | 9.18 |

Fixed Notebook `13` misses:

- `test:125508` Unstable angina
- `test:38475` Acute COPD exacerbation / infection
- `test:81691` Croup
- `test:8666` Influenza

Remaining misses:

- `test:111176` Acute rhinosinusitis predicted as Chronic rhinosinusitis
- `test:62878` Pericarditis predicted as Anemia

Current interpretation:

- Notebook `26` is not a promoted live method because it reuses observed trajectories and evaluates policies on the 49-case labels.
- It is the strongest evidence so far that multi-agent branching is worth a prospective live confirmation.
- The next credible notebook should be a fixed live branching confirmation: run one base branch, apply the `hybrid_suspicion_v1` trigger, spawn at most two alternate base-style branches only when triggered, and adjudicate with a pre-fixed graph/Bayes/MLP judge.
- Do not use majority vote as the main method; the offline lab shows majority vote stays at `43/49`.

## 67. Replicate Final-Layer Graph/Bayes Quickcheck

Added a quick offline final-layer graph/Bayes replay over the three Notebook `25` live base replicate traces.

Output:

- `artifacts/trajectory_replicates/offline_branching_trajectory_lab_49case_v1/replicate_graph_bayes_final_layer_quickcheck.csv`

Design:

- reconstruct final visible evidence from each replicate trace
- compute train-derived graph support from Notebook `16`
- compute Bayesian posterior from Notebook `19`
- evaluate:
  - base replicate prediction
  - strict conservative graph/Bayes final layer
  - raw graph top-1 diagnostic head
  - raw Bayes top-1 diagnostic head

Result:

| Replicate | Base | Strict final layer | Raw graph top-1 | Raw Bayes top-1 |
|---|---:|---:|---:|---:|
| `r01` | 44/49 | 44/49 | 45/49 | 45/49 |
| `r02` | 42/49 | 42/49 | 44/49 | 44/49 |
| `r03` | 42/49 | 42/49 | 44/49 | 44/49 |

The strict conservative final layer did not fire on any replicate. The raw graph and Bayes top-1 heads improved all three replicates diagnostically with zero regressions:

- `r01`: 1 win, 0 regressions
- `r02`: 2 wins, 0 regressions
- `r03`: 2 wins, 0 regressions

Interpretation:

- Graph/Bayes remains useful on fresh replicate trajectories.
- The existing conservative override is too cautious for these trajectories.
- This supports using graph/Bayes as the branch judge in the multi-agent policy, not necessarily as a blunt single final-head override.
- This quickcheck is diagnostic only and should not be reported as a promoted method.

## 68. Notebook 27 Live Targeted Branching Confirmation

Implemented the prospective live multi-agent branching notebook:

- `notebooks/27_live_targeted_branching_confirmation.ipynb`
- `reports/algorithmic_ledger/live_targeted_branching_confirmation_report.md`

Live artifact root:

- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/`

Dry-run smoke artifact root:

- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_dryrun_smoke_v1/`

Frozen prospective policy:

```text
trigger=hybrid_suspicion_v1
branch_budget=2
judge=highest_raw_bayes_posterior
```

Design:

- run the Notebook `13` selected-stop workup once as the base branch
- compute terminal label-free suspicion signals from the visible ledger, MLP state, raw graph support, and raw Bayesian posterior
- if the trigger fires, launch at most two fresh-context full workup branches from the original initial evidence
- branch prompts receive structured ledger state and branch-role directives, not base free-text reasoning
- branch roles are `graph_bayes_scout` and `counteranchor_scout`
- final adjudication selects the completed branch whose own final prediction has the highest raw Bayesian posterior, using graph support, MLP confidence, and base order only as tie-breakers

Validation completed:

- static parsed all Notebook `27` code cells
- executed a two-case no-spend dry run with `NOTEBOOK27_RUN_LIVE_API=0 NOTEBOOK27_ALLOW_DRY_RUN=1`
- dry-run artifact contract passed
- dry-run result was `2/2`, with no branch triggers because both smoke cases were stable
- executed an additional no-spend forced branch-path smoke on one case to confirm a fresh branch can run, apply the early divergent-root guard, and be scored by the raw Bayes judge

Interpretation:

- This was the clean live confirmation notebook for the Notebook `26` branching hypothesis.
- The full live result later completed; section 69 records the outcome.
- Notebook `13` remains the defended live backbone because Notebook `27` did not reach a promotable result.

## 69. Notebook 27 Live Result And Branching Findings

The full 49-case Notebook `27` live targeted-branching confirmation completed.

Artifact root:

- `artifacts/trajectory_replicates/live_targeted_branching_confirmation_49case_v1/`

Headline result:

| Metric | Value |
|---|---:|
| Base branch accuracy | 42/49 = 0.857 |
| Targeted branching accuracy | 43/49 = 0.878 |
| Top-3 accuracy | 0.918 |
| Top-5 accuracy | 0.939 |
| Wins versus base | 2 |
| Regressions versus base | 1 |
| Triggered cases | 9/49 |
| Branches spawned | 18 |
| Mean base requests | 6.92 |
| Mean selected requests | 6.82 |
| Mean total branch requests | 11.45 |

Promotion decision:

```text
not_promoted_keep_notebook13_or_prior_confirmed_method
```

Paired changes:

| Case | True pathology | Base prediction | Branch-selected prediction | Outcome |
|---|---|---|---|---|
| `test:35039` | Myocarditis | Sarcoidosis | Myocarditis | win |
| `test:76022` | Panic attack | Anaphylaxis | Panic attack | win |
| `test:38475` | Acute COPD exacerbation / infection | Acute COPD exacerbation / infection | Bronchospasm / acute asthma exacerbation | regression |

Branch-candidate ceiling:

| System | Correct |
|---|---:|
| Base branch only | 42/49 |
| Notebook `27` selected judge | 43/49 |
| Oracle over actual base + spawned branch predictions | 44/49 |

This means the live branches did contain useful alternate trajectories, but not enough to reach `47/49` by branch selection alone.

Raw graph/Bayes diagnostic replay over the Notebook `27` base final state:

| Diagnostic head | Correct |
|---|---:|
| Base branch prediction | 42/49 |
| Raw graph top-1 | 45/49 |
| Raw Bayes top-1 | 45/49 |

Raw graph and Bayes both fixed:

- `test:35039` Myocarditis
- `test:76022` Panic attack
- `test:19160` Possible NSTEMI / STEMI

with no base-correct regressions in this diagnostic replay.

Main failure modes:

- **Silent confident wrong cases** did not trigger branching because graph, Bayes, MLP, and LLM all agreed with the wrong answer:
  - `test:111176`: Acute rhinosinusitis predicted as Chronic rhinosinusitis
  - `test:11655`: Bronchitis predicted as URTI
  - `test:81691`: Croup predicted as Acute otitis media
- **Triggered but no correct LLM branch**:
  - `test:19160`: graph/Bayes top-1 was Possible NSTEMI / STEMI, but both spawned branches ended Acute laryngitis
  - `test:62878`: Pericarditis remained unresolved; branches ended Chagas and Spontaneous pneumothorax
- **Resolver regression**:
  - `test:38475`: the base COPD answer was graph rank `1` and Bayes rank `1`, but the raw-Bayes-only resolver chose the asthma/bronchospasm branch because its posterior was slightly higher

Interpretation:

- Notebook `27` partially validates the multi-agent hypothesis: branches can escape lock-in on some live cases.
- The selected raw-Bayes branch resolver is not good enough and should not be promoted.
- Confidence/contradiction triggers miss consensus wrong-answer failures.
- The next credible direction is a confounder-coverage branch classifier plus a cautious graph/Bayes/MLP resolver with graph/Bayes pseudo-candidates and base protection.

## 70. Notebook 28 Implemented: MLP-Gated Confounder Graph-Bayes Branching

Implemented `notebooks/28_mlp_gated_confounder_graph_bayes_branching.ipynb`.

Artifact root for the no-spend smoke run:

- `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_dryrun_smoke_v1/`

Planned live root:

- `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1/`

Notebook `28` is the successor to Notebook `27`. The key change is that branching is now a learned decision:

```text
branch_trigger_mlp_v1_probability >= 0.40
```

The branch-trigger MLP is trained only on train/validate synthetic partial evidence states. It uses diagnostic MLP belief features, graph support/rank/contradiction features, Bayesian posterior/rank features, graph-Bayes agreement, request-count features, and pairwise confounder coverage.

The actual branches remain fresh-context LLM workup agents. If the learned gate fires, the notebook can spawn up to three branches:

- `graph_bayes_scout`
- `confounder_pair_scout`
- `counteranchor_scout`

The final resolver scores the base prediction, completed branch predictions, and graph/Bayes/MLP pseudo-candidates. It also includes base protection when the base answer is graph rank `1` and Bayes rank `1`.

No-spend dry-run validation passed top to bottom:

| Check | Result |
|---|---:|
| Dry-run cases | 2 |
| Base accuracy | 2/2 |
| Notebook `28` selected accuracy | 2/2 |
| Branch trigger rate | 0/2 |
| Mean selected requests | 6.0 |
| Mean total branch requests | 6.0 |

Calibration diagnostics:

| Model | Key validation result |
|---|---:|
| Branch-trigger MLP | AUC `0.939`, average precision `0.896`, threshold `0.40`, recall `0.951` in the dry-run implementation; live artifacts later selected threshold `0.375` with AUC `0.954` and average precision `0.931` |
| Candidate resolver | AUC `0.955`, average precision `0.811` |

The notebook writes a richer evaluation suite: trigger probability histograms, threshold curves, request-cost plots, pseudo-candidate counts, resolver score diagnostics, graph/Bayes/MLP agreement heatmap, confounder coverage plots, hard-case rank movement, branch MLP calibration, ROC/PR curves, and feature importance.

Current status:

- implemented and smoke-tested
- live 49-case run completed; section 71 records the result
- not promoted because the live result did not reach the promotion rule
- Notebook `13` remains the defended live backbone for now

## 71. Notebook 28 Live Result And Post-Hoc Candidate-Pool Analysis

The full 49-case Notebook `28` live run completed.

Artifact root:

- `artifacts/trajectory_replicates/mlp_gated_confounder_graph_bayes_branching_49case_v1/`

Post-hoc analysis artifacts added after reading the live outputs:

- `posthoc_candidate_pool_oracle_summary.csv`
- `posthoc_miss_candidate_coverage.csv`
- `posthoc_notebook28_analysis.json`

Headline result:

| Metric | Value |
|---|---:|
| Base branch accuracy | 42/49 = 0.857 |
| Notebook `28` selected accuracy | 44/49 = 0.898 |
| Top-3 accuracy | 0.959 |
| Top-5 accuracy | 0.959 |
| Wins versus base | 2 |
| Regressions versus base | 0 |
| Changed predictions | 2 |
| Branch trigger rate | 4/49 = 0.082 |
| Branches spawned | 12 |
| Mean base requests | 6.92 |
| Mean selected requests | 6.63 |
| Mean total branch requests | 9.96 |

Selection sources:

| Source | Selected cases |
|---|---:|
| Base | 47 |
| `counteranchor_scout` | 1 |
| `pseudo_graph_top1` | 1 |

Paired wins:

| Case | True pathology | Base prediction | Notebook `28` prediction | Source |
|---|---|---|---|---|
| `test:35039` | Myocarditis | Sarcoidosis | Myocarditis | `pseudo_graph_top1` |
| `test:62878` | Pericarditis | Panic attack | Pericarditis | `counteranchor_scout` |

Promotion decision:

```text
not_promoted_keep_notebook13_or_prior_confirmed_method
```

Notebook `28` improved its own live base by two cases with zero regressions, and the base-protection mechanism prevented the COPD-style regression observed in Notebook `27`. However, the selected method reached only `44/49`, below the `47/49` promotion target and below the fresh Notebook `24` base result of `45/49`.

The most important post-hoc finding is that the final judge was not the main bottleneck. The actual scored candidate pool had only a `44/49` oracle ceiling:

| Candidate pool | Oracle correct |
|---|---:|
| Current scored candidates | 44/49 |
| Current candidates plus all candidate ranked top-1 | 44/49 |
| Current candidates plus all candidate ranked top-2 | 47/49 |
| Current candidates plus all candidate ranked top-3 | 48/49 |
| Current candidates plus all candidate ranked top-5 | 48/49 |
| Selected final ranked differential top-2 only | 47/49 |

Remaining misses:

| Case | True pathology | Prediction | Key observation |
|---|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | Correct disease was rank 2 in the final differential. |
| `test:11655` | Bronchitis | URTI | Correct disease was rank 2 in the final differential and was also the challenger. |
| `test:81691` | Croup | Acute otitis media | Branching fired, but no scored candidate was Croup; a branch differential had Croup at rank 3. |
| `test:8666` | Influenza | HIV (initial infection) | Correct disease was rank 2 in the final differential and was also the challenger. |
| `test:76022` | Panic attack | Myocarditis | Correct disease did not appear in the scored pool or top-5 differential. |

Interpretation:

- Notebook `28` is a cleaner live branching system than Notebook `27`: fewer triggers, no regressions, and a better selected score.
- The branch MLP under-fired on live trajectories. It was calibrated to branch at about `50%` on synthetic validation states but fired on only `4/49` live cases.
- Silent consensus-wrong cases remain hard because graph, Bayes, MLP, and LLM can all support the same wrong anchor.
- The next credible iteration should expand the final candidate pool rather than simply spawn more branches.

Next iteration direction:

```text
Notebook 29: listwise differential graph-Bayes-MLP adjudicator
```

The candidate pool should include base ranked top-5, branch ranked top-5, graph top-5, Bayes top-5, MLP top-5, and confounder challengers. A calibrated resolver should then choose among candidates using graph/Bayes/MLP support, rank position, contradiction, confounder coverage, and request-state features. The post-hoc Notebook `28` oracle shows this is the first direction with a direct path back to `47/49+`.

## 72. Notebook 29 Implemented And Executed: Listwise Differential Graph-Bayes-MLP Adjudicator

Implemented and executed `notebooks/29_listwise_differential_graph_bayes_adjudicator.ipynb`.

Artifact root:

- `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`

Report:

- `reports/algorithmic_ledger/listwise_differential_graph_bayes_adjudicator_report.md`

Notebook `29` is offline-only. It does not call the LLM API. It keeps Notebook `28`'s live base branch, spawned branches, and revealed evidence fixed, then adds a mathematical final adjudicator over ranked differential candidates.

Method summary:

- explode each base/branch state into source top-1, ranked differential entries, graph top-5, Bayes top-5, and MLP top-5 candidates
- recompute graph score/rank/posterior, Bayesian posterior/rank, and partial-evidence MLP posterior/rank for each candidate
- train an L2 logistic candidate scorer on Notebook `28` train/validate synthetic candidate features
- use the fixed selected policy `listwise_differential_graph_bayes_mlp_v1`
- admit source top-1 plus ranked differential top-3 candidates
- override Notebook `28` only when the candidate score exceeds the Notebook `28` selected answer by margin `0.02`

Validation summary:

| Metric | Value |
|---|---:|
| Train candidate rows | 33,232 |
| Validate candidate rows | 16,670 |
| Candidate AUC | 0.955 |
| Candidate average precision | 0.809 |

49-case result:

| Metric | Notebook `28` | Notebook `29` |
|---|---:|---:|
| Correct | 44/49 | 45/49 |
| Accuracy | 0.898 | 0.918 |
| Wins vs Notebook `28` | n/a | 1 |
| Regressions vs Notebook `28` | n/a | 0 |
| Changed predictions | n/a | 1 |
| Mean selected requests | 6.63 | 6.63 |
| Mean total branch requests | 9.96 | 9.96 |

The single win is `test:81691` Croup:

- Notebook `28`: Acute otitis media
- Notebook `29`: Croup
- source: `graph_bayes_scout`
- ranked-differential position: rank `3`
- graph rank: `1`
- Bayes rank: `1`
- MLP rank: `3`

Candidate-pool oracle:

| Candidate pool | Oracle correct |
|---|---:|
| Source top-1 plus ranked top-1 | 44/49 |
| Source top-1 plus ranked top-2 | 47/49 |
| Source top-1 plus ranked top-3 | 48/49 |
| Source top-1 plus ranked top-5 | 48/49 |
| All exploded graph/Bayes/MLP/ranked candidates | 49/49 |

Promotion decision:

```text
not_promoted_diagnostic_offline_listwise_candidate
```

Interpretation:

- Notebook `29` is a real offline improvement over Notebook `28`, but not a promoted result.
- The candidate pool is no longer the obvious bottleneck: a non-oracle top-2/top-3 ranked-differential resolver could reach the target.
- The current learned scorer remains too loyal to graph/Bayes/MLP rank-1 consensus in silent near-neighbor confusions.
- Remaining misses are Acute rhinosinusitis vs Chronic rhinosinusitis, Bronchitis vs URTI, Influenza vs HIV initial infection, and Panic attack vs Myocarditis.
- The next iteration should be a pairwise or abstaining adjudicator for top-1-vs-rank-2 confounders, trained/calibrated without using the 49-case labels.

Execution checks:

- static parse of all Notebook `29` code cells passed
- Notebook `29` executed top-to-bottom through a local cell runner because `nbconvert`/`nbclient` are not installed
- required artifacts and figures were written

## 73. Notebook 29 Post-Run Deep Analysis

After the user reran Notebook `29`, the latest artifacts were inspected again. The rerun refreshed Notebook outputs and `resolved_run_config.json`, but the selected metrics remained stable:

| Metric | Value |
|---|---:|
| Notebook `29` selected correct | 45/49 |
| Wins vs Notebook `28` | 1 |
| Regressions vs Notebook `28` | 0 |
| Changed predictions | 1 |

Additional post-hoc artifacts were generated under:

- `artifacts/trajectory_replicates/listwise_differential_graph_bayes_adjudicator_49case_v1/`

New post-hoc files:

- `posthoc_candidate_pool_signal_oracle.csv`
- `posthoc_case_true_candidate_score_gaps.csv`
- `posthoc_remaining_miss_score_gaps.csv`
- `posthoc_ranked_true_candidate_opportunities.csv`
- `posthoc_notebook29_deep_analysis.json`
- `figures/posthoc_candidate_pool_signal_oracle.png`
- `figures/posthoc_remaining_miss_score_gaps.png`

The deeper candidate-availability result:

| Candidate signal family | Oracle correct |
|---|---:|
| Source top-1 only | 44/49 |
| Ranked differential top-2 | 47/49 |
| Ranked differential top-3 | 48/49 |
| Graph top-1 | 45/49 |
| Graph top-2 | 49/49 |
| Graph top-3 | 49/49 |
| Bayes top-1 | 45/49 |
| Bayes top-2 | 49/49 |
| Bayes top-3 | 49/49 |
| MLP top-3 | 48/49 |

This is the key analysis update: graph and Bayes are not failing to surface the right diagnosis. In the remaining hard cases, the true diagnosis is usually graph rank `2` and Bayes rank `2`, but the selected scorer remains over-loyal to the wrong rank-1 anchor.

Remaining misses after Notebook `29`:

| Case | True pathology | Selected prediction | Selected score - true score | True differential rank | True graph rank | True Bayes rank | True MLP rank |
|---|---|---|---:|---:|---:|---:|---:|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | 0.209 | 2 | 2 | 2 | 3 |
| `test:11655` | Bronchitis | URTI | 0.105 | 2 | 2 | 2 | 2 |
| `test:76022` | Panic attack | Myocarditis | 0.082 | not ranked | 2 | 2 | 11 |
| `test:8666` | Influenza | HIV initial infection | 0.230 | 2 | 2 | 2 | 2 |

Interpretation:

- Candidate expansion has now mostly done its job.
- Top-1 graph/Bayes is not enough, because it reaches only `45/49`.
- Top-2 graph/Bayes contains the target answer in all cases, so the mathematical ledger is surfacing the right neighborhood.
- The next credible model should be pairwise or abstaining, focused on top-1-vs-rank-2 confounder resolution.
- A global multicandidate logistic scorer is too blunt for near-neighbor disease pairs where all heads prefer the same wrong anchor.
