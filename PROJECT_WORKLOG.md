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

- [baseline_summary.md](reports/baseline_summary.md)
- [baseline_results_and_next_steps.md](reports/baseline_results_and_next_steps.md)
- [results_assessment.md](reports/results_assessment.md)
- [sequential_api_guide.md](reports/sequential_api_guide.md)

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

- [results_assessment.md](reports/results_assessment.md)
- [baseline_results_and_next_steps.md](reports/baseline_results_and_next_steps.md)

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

- [proposed_improvement_1_results.md](reports/proposed_improvement_1_results.md)

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

- [sequential_policy_refinement_report.md](reports/sequential_policy_refinement_report.md)

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

- [sequential_policy_refinement_report.md](reports/sequential_policy_refinement_report.md)

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

- [evidence_ledger_algorithm_and_improvements.md](reports/evidence_ledger_algorithm_and_improvements.md)
- [proposed_multi_agent_architecture.md](reports/proposed_multi_agent_architecture.md)

Purpose:

- clarify that the evidence ledger is not strong enough as a novelty claim if presented only as shared memory
- reframe the real method as an evidence-gated differential ledger that constrains requests, diagnosis updates, critique, and stopping
- provide a concrete multi-agent architecture centered on the ledger rather than free-form inter-agent chat

Current recommendation captured in those reports:

- do **not** claim novelty from “multiple agents share memory” alone
- claim novelty from a diagnosis-specific ledger protocol with support/contradiction tracking, validation, and stop control
- use a controller-constrained multi-agent design with planner, synthesizer, critic, optional retriever, and stop agent

An additional design-freeze note was added at:

- [architecture_v1_freeze_and_experimental_scope.md](reports/architecture_v1_freeze_and_experimental_scope.md)

Purpose:

- define what is fixed in the next implementation cycle
- prevent unnecessary churn in top-level architecture
- make the ledger/control protocol the main experimental variable
