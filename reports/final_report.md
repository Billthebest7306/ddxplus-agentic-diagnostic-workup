# Final Report: Evidence-Efficient Diagnostic Workup On DDXPlus

Last updated: 2026-05-06

## Executive Summary

This project started as a **Multi-Agent Diagnostic Workup Copilot** for DDXPlus. The implemented work clarified a sharper research direction:

> Can a structured sequential diagnostic workup policy approach full-evidence diagnostic performance while acquiring only a small, targeted subset of evidence?

The final proposed method is notebook `13`: a **single-agent LLM evidence-acquisition controller** with a deterministic evidence ledger and an online **partial-evidence MLP stopping signal**.

Final confirmation result:

| System | Cases | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected hybrid stop | 49 | 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |

Same-case 49-case framing:

| Comparator | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Initial-evidence one-shot | 0.286 | 0.673 | 0 |
| Proposed hybrid workup | 0.878 | 0.939 | 6.59 |
| Full-evidence one-shot ceiling | 0.980 | 1.000 | all fields |

The strongest conclusion is not that “agentic diagnosis beats every classifier.” The strongest conclusion is:

> Online MLP-guided stopping makes sequential LLM workup evidence-efficient. The proposed system substantially improves over initial-evidence diagnosis while using only a small subset of the available DDXPlus evidence fields.

## Research Question

Final research question:

> Can an explainable, ledger-controlled sequential diagnostic workup system improve diagnostic performance and evidence efficiency over one-shot and LLM-only sequential baselines under incomplete clinical evidence?

Secondary question:

> Given acquired evidence, should final diagnosis be made by the LLM, a neural classifier, or a conservative hybrid rule?

## Motivation

Real diagnostic workup is not a single-shot classification problem. Clinicians start with incomplete information and decide which questions, symptoms, history items, or tests are worth acquiring next.

DDXPlus is a good fit because it provides:

- synthetic patient cases with pathologies
- age, sex, initial evidence
- hidden evidence fields
- structured evidence metadata
- official train/validate/test splits
- 49 pathologies and 223 root evidence questions

The project therefore treats DDXPlus as a structured workup environment:

```text
initial patient state
  -> request evidence
  -> update ledger
  -> revise differential
  -> decide ask/stop
  -> final diagnosis
```

## Baseline Ladder

The project built a full baseline ladder rather than comparing against a weak strawman.

| Stage | Notebook | Purpose |
|---|---|---|
| Initial one-shot MLP | `01` | Required assignment baseline using only initial evidence |
| Early sequential agent | `02-06` | Development history for LLM workup and ledger controls |
| Full-evidence MLP | `07` | Ceiling comparator when all evidence is visible |
| LLM-only cost-sensitive sequential | `08` | Main sequential baseline |
| Matched-evidence integrated comparison | `09` | Compare systems on aligned evidence/cases |
| Partial-evidence MLP | `10` | Direct neural head trained for partial evidence states |
| Hybrid v1 lambda sweep | `11` | Online MLP feedback exploration |
| Stopping-policy ablation | `12` | Offline replay test of MLP-guided stopping |
| Proposed method | `13` | Live selected MLP stop confirmation |
| Hybrid v2 ablation | `14` | Negative test of MLP-driven question shortlisting |

## Model And System Design

### One-Shot Neural Baseline

The assignment required at least one faithful baseline. The one-shot baseline is a BASD-style MLP:

- input: age, sex, and `INITIAL_EVIDENCE`
- representation: official DDXPlus/BASD slot-style evidence encoding
- architecture: MLP with `[2048, 2048, 2048]` hidden layers
- output: 49 pathology classes
- training: full official train split

Full-test result:

| Model | Evidence | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Initial-evidence MLP | initial evidence only | 0.378 | 0.615 | 0.730 | 0.373 |

This establishes that initial evidence alone is diagnostically incomplete.

### Full-Evidence Ceiling

Notebook `07` trains the same style of MLP with all DDXPlus evidence visible.

Full-test result:

| Model | Evidence | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Full-evidence MLP | all evidence | 0.996 | 1.000 | 1.000 | 0.995 |

This shows DDXPlus contains enough signal for near-perfect diagnosis when all evidence is visible. The project’s real challenge is therefore **which evidence to acquire**, not whether DDXPlus is diagnosable at all.

### Evidence Ledger

The ledger is the system’s state manager. It tracks:

- visible initial evidence
- requested evidence history
- revealed present/absent/value states
- legal remaining evidence actions
- parent/child evidence gating
- diagnosis and shortlist history

This matters because the LLM is not allowed to invent evidence or inspect all fields. The ledger is the source of truth for what the agent has seen.

### LLM-Only Sequential Baseline

Notebook `08` uses `gpt-4.1-mini` as a single evidence-acquisition agent. It asks for DDXPlus evidence fields and eventually stops with a final diagnosis.

Best 24-case result:

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| LLM-only sequential, lambda 0.10 | 0.917 | 0.917 | 13.04 |

This showed sequential evidence acquisition can work, but the LLM-only policy used many requests.

### Partial-Evidence MLP

Notebook `10` trains a direct classifier on partial evidence states shaped by sequential policy masks.

Standalone test result:

| Model | Rows | Accuracy | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Partial-evidence MLP | 39,998 | 0.515 | 0.741 | 0.827 | 0.519 |

The point was not to make this the final system. It provided a neural belief head that can evaluate how diagnosable the current partial evidence state is.

### Hybrid V1: Proposed Method

Notebook `13` is the final proposed method.

Workflow:

```text
DDXPlus case
  -> deterministic evidence ledger
  -> LLM chooses evidence requests
  -> evidence reveal updates ledger
  -> partial-evidence MLP runs after each update
  -> MLP confidence/margin/entropy decide when to stop
  -> final diagnosis from LLM / MLP / conservative hybrid heads
```

Fixed stop rule from notebook `12`:

| Signal | Threshold |
|---|---:|
| minimum requests | 1 |
| MLP confidence | >= 0.70 |
| MLP margin | >= 0.20 |
| MLP entropy | <= 0.10 |
| MLP stability | >= 0 |

The LLM still chooses the evidence. The MLP primarily tells the system when enough evidence has been acquired.

## Results

### Full-Scale Neural Baselines

| System | Cases | Evidence | Accuracy | Top-5 | Macro-F1 |
|---|---:|---|---:|---:|---:|
| Initial-evidence MLP | 134,529 | age, sex, initial evidence | 0.378 | 0.730 | 0.373 |
| Partial-evidence MLP | 39,998 | sampled partial evidence masks | 0.515 | 0.827 | 0.519 |
| Full-evidence MLP | 134,529 | all evidence | 0.996 | 1.000 | 0.995 |

### Sequential And Hybrid Live Results

| System | Cases | Accuracy | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|
| LLM-only sequential, lambda 0.10 | 24 | 0.917 | 0.917 | 0.846 | 13.04 |
| Offline selected MLP stop | 24 | 0.917 | 0.917 | 0.867 | 6.88 |
| Hybrid v1 selected stop | 24 | 0.917 | 0.917 | 0.867 | 6.58 |
| Hybrid v2 MLP shortlist | 24 | 0.875 | 0.958 | 0.840 | 7.38 |
| Hybrid v1 selected stop | 49 | 0.878 | 0.939 | 0.845 | 6.59 |

### Final 49-Case Confirmation

The 49-case run is the main final result.

| Metric | Value |
|---|---:|
| Accuracy | 43/49 = 0.878 |
| Top-3 accuracy | 0.918 |
| Top-5 accuracy | 0.939 |
| Macro-F1 | 0.845 |
| Mean requests | 6.59 |
| Median requests | 5.0 |
| Stop-before-cap rate | 0.980 |
| Cap hits | 1 |
| Selected stop-rule fired | 36/49 = 0.735 |
| LLM/MLP top-1 agreement | 46/49 = 0.939 |

Same-case 49-case comparison:

| Comparator | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Initial-evidence one-shot | 0.286 | 0.673 | 0 |
| Proposed hybrid workup | 0.878 | 0.939 | 6.59 |
| Full-evidence one-shot ceiling | 0.980 | 1.000 | all fields |

## Ablations And Negative Results

### Stopping-Policy Ablation

Notebook `12` replayed saved trajectories and tested stop rules without new API calls.

Matched-budget result:

| Stop policy | Final head | Accuracy | Mean requests |
|---|---|---:|---:|
| Best pure LLM-only stop | LLM | 0.833 | 6.33 |
| Selected MLP-guided stop | agreement hybrid | 0.917 | 6.88 |

This supports the claim that the MLP provides a useful stop signal.

### Hybrid V2 Rejection

Notebook `14` tested whether the MLP should also guide question selection directly.

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Hybrid v1 selected stop | 0.917 | 0.917 | 6.58 |
| Hybrid v2 MLP shortlist | 0.875 | 0.958 | 7.38 |

V2 fixed `Pericarditis`, but it introduced new failures on `Chagas` and `Influenza`, still failed `Croup`, and used more evidence. It was rejected by the predefined promotion rule.

This is an important negative result:

> The MLP is useful as a stopping monitor, but direct MLP-driven question shortlisting is not automatically better.

## Error Analysis

Final 49-case errors:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | 24 | max requests reached |
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | 8 | selected MLP stop |
| `test:81691` | Croup | Anemia | 19 | agent stop |
| `test:8666` | Influenza | HIV initial infection | 3 | agent stop |
| `test:62878` | Pericarditis | Anemia | 15 | agent stop |
| `test:125508` | Unstable angina | Anemia | 2 | agent stop |

The main failure pattern is wrong belief convergence. In several errors, the LLM and MLP agreed on the wrong diagnosis, so the issue is not simply final-head arbitration. The next technical improvement would need stronger contradiction handling, calibration, or disease-specific safeguards.

## Claims We Can Make

Supported:

- A faithful DDXPlus deep learning baseline was implemented.
- DDXPlus initial evidence is insufficient for strong diagnosis.
- DDXPlus full evidence is near-ceiling for a neural classifier.
- Sequential evidence acquisition substantially improves over initial-evidence diagnosis on balanced live slices.
- MLP-guided stopping improves evidence efficiency relative to the LLM-only sequential baseline on the 24-case control.
- The final proposed system reaches high-80s accuracy with about 6.6 evidence requests on the 49-case confirmation.

Not supported:

- The LLM final answer is generally better than a neural classifier.
- Multi-agent systems are better; no multi-agent experiment has been run yet.
- Direct MLP-guided question selection is better; notebook `14` rejected that.
- The method beats official DDXPlus RL baselines; those were not reproduced.
- The result is a definitive clinical benchmark.

## Final Interpretation

The project should be presented as an **evidence-efficient diagnostic workup system**, not as an unrestricted agentic-diagnosis win.

The strongest final statement is:

> We built a reproducible DDXPlus baseline ladder and proposed a structured sequential workup system where an LLM acquires evidence under a deterministic ledger and a partial-evidence MLP decides when enough information has been gathered. The final 49-case confirmation reached `43/49` accuracy and `0.939` top-5 while requesting about `6.6` evidence fields per case.

This is a credible course-project contribution because it includes:

- full deep learning baseline
- ceiling comparator
- sequential baseline
- matched evidence comparator
- stopping-policy ablation
- negative v2 ablation
- live 49-case confirmation
- clear limitations

## Presentation Outline

### Slide 1. Title

Evidence-Efficient Diagnostic Workup on DDXPlus.

### Slide 2. Problem

Diagnosis under incomplete evidence is not the same as static classification. The system must decide what to ask next.

### Slide 3. Dataset

DDXPlus: 49 pathologies, 223 evidence roots, age, sex, initial evidence, hidden requestable evidence, official train/validate/test splits.

### Slide 4. Research Question

Can a structured sequential workup approach full-evidence performance while acquiring only a limited subset of evidence?

### Slide 5. Baseline Ladder

Initial one-shot MLP, full-evidence ceiling MLP, LLM-only sequential, partial-evidence MLP, hybrid proposed method.

### Slide 6. Evidence Ledger

Ledger tracks visible evidence, hidden evidence, legal requests, parent/child gating, revealed values, and request history.

### Slide 7. One-Shot Baseline

Initial-evidence MLP: 0.378 full-test accuracy. This proves the initial state is hard.

### Slide 8. Full-Evidence Ceiling

Full-evidence MLP: 0.996 full-test accuracy. This proves DDXPlus is highly diagnosable once evidence is visible.

### Slide 9. Sequential Baseline

LLM-only sequential policy reached 22/24 on a pilot but required 13.04 evidence requests.

### Slide 10. Hybrid V1 Proposed Method

LLM asks questions; MLP monitors diagnostic belief; stop when MLP confidence, margin, and entropy pass thresholds.

### Slide 11. Why MLP Stopping

Offline ablation: best pure LLM-only stop at matched budget reached 20/24; MLP-guided stop reached 22/24.

### Slide 12. Final 49-Case Result

Hybrid v1 reached 43/49 accuracy, 0.939 top-5, and 6.59 mean requests.

### Slide 13. Negative V2 Ablation

MLP-guided question shortlisting was tested and rejected: lower top-1 and higher requests, despite better top-5.

### Slide 14. Error Analysis

Errors are mainly wrong-belief convergence, not just insufficient evidence budget.

### Slide 15. Claims And Limitations

Strong claim: evidence-efficient sequential workup with MLP-guided stopping. Limitation: not a proof of multi-agent superiority or RL baseline dominance.

### Slide 16. Future Work

Matched-budget LLM-only control, larger live runs, better calibration, contradiction handling, and later multi-agent ledger-based coordination.

## Key Files

- `notebooks/01_one_shot_classifier_baselines.ipynb`
- `notebooks/07_full_evidence_one_shot_comparator.ipynb`
- `notebooks/08_cost_sensitive_sequential_lambda_sweep.ipynb`
- `notebooks/10_partial_evidence_one_shot_comparator.ipynb`
- `notebooks/12_stopping_policy_ablation.ipynb`
- `notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb`
- `notebooks/14_hybrid_v2_mlp_discriminative_shortlist.ipynb`
- `reports/final_results_summary.md`
- `reports/hybrid/live_selected_hybrid_stopping_confirmation.md`
- `reports/hybrid/hybrid_v2_mlp_discriminative_shortlist_report.md`
