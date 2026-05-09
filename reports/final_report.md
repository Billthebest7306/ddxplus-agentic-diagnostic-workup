# Final Report: Evidence-Efficient Diagnostic Workup On DDXPlus

Last updated: 2026-05-09

## Executive Summary

This project started as a **Multi-Agent Diagnostic Workup Copilot** for DDXPlus. The implemented work clarified a sharper research direction:

> Can a structured sequential diagnostic workup policy approach full-evidence diagnostic performance while acquiring only a small, targeted subset of evidence?

The live first-pass workup method is notebook `13`: a **single-agent LLM evidence-acquisition controller** with a deterministic evidence ledger and an online **partial-evidence MLP stopping signal**.

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

The newest and strongest offline graph-ledger enhancement is notebook `23`: a calibrated graph/Bayes rescue reranker over the saved Notebook `13` trace. It improves the saved 49-case result to `47/49 = 0.959` with `6.96` mean requests and zero regressions, but it should be described as an offline enhancement candidate until confirmed in a live or held-out run.

Notebook `24` completed that live confirmation. The rescue layer did **not** reproduce the offline `47/49` result, so it was not promoted. The important positive result is that the fresh Notebook `13`-style live base inside Notebook `24` reached `45/49 = 0.918` with `6.20` mean requests; the live rescue also finished at `45/49 = 0.918` with `6.39` mean requests.

Notebook `27` completed the prospective live targeted-branching confirmation. It partially confirmed the branching idea by improving its own live base from `42/49` to `43/49`, but it introduced one regression and used `11.45` mean total branch requests, so it was not promoted.

Notebook `28` completed the successor branching design: a learned branch-trigger MLP, up to three fresh-context LLM branches, graph/Bayes/MLP pseudo-candidates, and a calibrated resolver with base protection. It improved its own live base from `42/49` to `44/49` with zero regressions, but it did not reach the `47/49` promotion target.

The strongest conclusion is not that “agentic diagnosis beats every classifier.” The strongest conclusion is:

> Online MLP-guided stopping makes sequential LLM workup evidence-efficient. Graph/Bayes rescue is promising offline, and targeted multi-agent branching can recover some live wrong trajectories. The learned-gate Notebook `28` run improved over its own base without regressions, but its post-hoc analysis shows the next real opportunity is listwise adjudication over ranked differentials, not simply more branch top-1 votes.

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
| Stop-threshold sensitivity | `15` | Offline analysis of Notebook `13` 49-case trajectories |
| Algorithmic graph ledger | `16` | Offline MedKGI-style graph evidence analysis |
| Live graph shortlist | `17` | Negative test of hard graph top-10 question shortlisting |
| Graph advisory / context lab | `18-21` | Negative graph-controller tests plus graph-context critic analysis |
| Graph posterior final critic | `22` | Offline graph-ledger final adjudicator over Notebook `13` traces |

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
| MedKGI graph shortlist | 24 | 0.833 | 0.875 | 0.744 | 6.21 |
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

### MedKGI-Style Graph Shortlist Rejection

Notebook `16` showed that train-derived graph evidence scores are meaningful: Notebook `13` incorrect trajectories had worse mean graph rank (`9.73`) and lower mean information gain (`0.194`) than correct trajectories (`6.61`, `0.247`). This motivated a live graph-shortlist pilot.

Notebook `17` then kept Notebook `13`'s stop rule fixed and replaced only the evidence shortlist with MedKGI-style graph top-10 legal fields.

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Hybrid v1 selected stop | 0.917 | 0.917 | 6.58 |
| MedKGI graph shortlist | 0.833 | 0.875 | 6.21 |

Graph-request metrics looked strong locally:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 1.76 |
| Mean requested information gain | 0.373 |
| Requests outside graph top-10 | 0 |

But the paired comparison was worse:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| Hybrid v1 only correct | 2 |
| Graph shortlist only correct | 0 |
| Both wrong | 2 |

This is the current graph conclusion:

> Graph scores are useful as analysis/advisory signals, but a hard graph top-10 replacement shortlist over-constrains the action space and can efficiently ask high-scoring questions for the wrong active differential.

Notebook `17` was rejected. Notebook `13` remains the frozen proposed method.

### Graph-Advisory Shortlist Rejection

Notebook `18` tested the next graph hypothesis: graph scores may be useful if they are advisory rather than a hard replacement for the broader Notebook `13` shortlist.

It restored Notebook `13` shortlist diversity, added rare disease-specific support, and kept Notebook `13`'s MLP-guided stop rule fixed.

| System | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Hybrid v1 selected stop | 0.917 | 0.917 | 6.58 |
| Hard MedKGI graph shortlist | 0.833 | 0.875 | 6.21 |
| Graph-advisory hybrid shortlist | 0.875 | 0.917 | 7.67 |

Notebook `18` recovered the Chagas and Ebola failures introduced by Notebook `17`, but it introduced a new Stable angina failure, still failed Croup and Pericarditis, and used more evidence than Notebook `13`. The three wrong Notebook `18` cases all hit the 24-request cap, so the failure is not unsafe early stopping; it is wrong evidence trajectory steering and failure to recover the correct belief.

This result sharpens the graph conclusion:

> Graph signals are useful for audit and rare-disease recovery, but neither hard graph replacement nor advisory graph blending is currently strong enough to replace the Notebook `13` policy.

Notebook `18` was rejected. Notebook `13` remains the frozen proposed method.

### Graph Context As Critic

Notebook `20` corrected the graph-ledger framing. Instead of replacing the LLM's question-selection policy, it kept Notebook `13`'s LLM-led evidence loop and MLP stop rule, then added compact graph-ledger context to the prompt.

| System | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Hybrid v1 selected stop | 0.917 | 0.917 | 0.917 | 6.58 |
| LLM-led graph context | 0.833 | 0.958 | 0.958 | 6.13 |

Notebook `20` did not beat Notebook `13` on top-1, but it produced the best 24-case ranking quality so far: `23/24` top-3 and top-5. This suggested the graph context was helping keep the correct diagnosis in the differential even when the final top-1 was wrong.

Notebook `21` then replayed Notebooks `13`, `17`, `18`, and `20` offline to test graph context as a critic, guardrail, adjudicator, and drift detector.

Main Notebook `21` findings:

- no non-oracle graph adjudication variant beat Notebook `13`
- Notebook `20` oracle top-3/top-5 upper bound reached `23/24 = 0.958`, confirming the ranking signal is real
- wrong Notebook `20` top-1 predictions had much higher graph contradiction than correct predictions
- the best non-oracle variant only matched Notebook `13` and changed one wrong case into another wrong case

Graph contradiction diagnostic:

| Feature | Correct mean | Wrong mean | Wrong - correct |
|---|---:|---:|---:|
| Top contradiction | 0.377 | 4.015 | +3.638 |
| Top contradiction minus support | -6.964 | 0.906 | +7.870 |
| Top net support | 6.964 | -0.906 | -7.870 |

Notebook `21` was not promoted to a live Notebook `22`. It strengthens the research story by showing that graph ledger context has diagnostic signal, but hand-written graph threshold rules are not enough to safely choose the final top-1.

Handoff check on 2026-05-08: the existing Notebook `21` artifacts are not sufficient for a defensible learned/calibrated graph adjudicator with strict train/dev separation. The graph-context ranking signal from Notebook `20` exists only on the 24-case pilot slice, and those 24 cases are all included in the broader Notebook `13` 49-case confirmation. A learned graph adjudicator would therefore need new or reserved graph-context development traces before it should be reported as a method.

### Graph Posterior Final Critic

Notebook `22` tested a different graph-ledger role. Instead of using graph context during question selection or tuning hand-threshold rules on the Notebook `20` pilot, it keeps Notebook `13` evidence acquisition fixed and computes a train-derived graph posterior over the final visible evidence state.

The selected critic uses clipped signed log-odds support from the Notebook `16` graph:

```text
graph_score(disease) =
  sum clip(log_odds_support(revealed outcome -> disease), -3, 3)
```

It overrides Notebook `13` only when the graph top-1 differs, the graph margin is at least `1.0`, the Notebook `13` prediction has negative graph support, and the graph top-1 has positive graph support.

| System | Cases | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 49 | 43/49 = 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Notebook `22` conservative graph critic | 49 | 44/49 = 0.898 | 0.939 | 0.939 | 0.867 | 6.59 |

Notebook `22` changes only `test:81691` Croup, moving the final answer from Anemia to Croup. It introduces no regressions on the saved 49-case trace. At that point it was the first algorithmic-ledger enhancement to beat Notebook `13`, but it remains an offline final-head result rather than a new live evidence-acquisition policy.

The post-run behavior is the important part: the selected critic fires exactly once on the 49-case confirmation. That override has the intended mathematical shape: Notebook `13`'s Anemia answer has negative graph support (`-2.359`), Croup has positive graph support (`1.177`), and the graph margin is `2.073`. The 24-case sanity slice shows the same one-case improvement, from `22/24` to `23/24`, again by fixing Croup. This is not independent validation, but it is a useful consistency check.

The Croup correction is not the full opportunity. The graph posterior has a much higher rescue ceiling: Notebook `13` top-1 plus an oracle chooser over graph top-3 would reach `47/49`, and Notebook `13` top-1 plus graph top-5 would reach `48/49`. The next meaningful algorithmic improvement should therefore be a calibrated graph/LLM/MLP reranker that learns when to trust graph rank-2 or rank-3 alternatives. A hand-tuned rule over the six errors would not be a defensible result.

### Calibrated Graph-Bayes Rescue

Notebook `23` implements that next step offline. It keeps Notebook `13` as the first-pass workup, then applies calibrated graph/Bayes/MLP certificates and a tiny deterministic rescue continuation. The candidate scorer is trained on DDXPlus train/validate synthetic partial evidence states, while the saved 49-case labels are used only for final evaluation.

The selected policy has three rescue mechanisms:

- prior recovery when the initial prior is graph-supported and the final Notebook `13` answer is graph-contradicted
- the Notebook `22` conservative graph critic
- up to three graph/Bayes discriminator questions for suspicious early `agent_stop` cases, followed by the trained L2 reranker with a conservative graph-support accept guard

| System | Cases | Accuracy | Mean requests | Extra requests | Regressions |
|---|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 49 | 43/49 = 0.878 | 6.59 | 0 | 0 |
| Notebook `22` graph critic | 49 | 44/49 = 0.898 | 6.59 | 0 | 0 |
| Notebook `23` graph-Bayes rescue | 49 | 47/49 = 0.959 | 6.96 | 18 total | 0 |

Notebook `23` fixes four Notebook `13` misses: COPD via prior recovery, Croup via the graph critic, and Influenza plus Unstable angina via rescue reranking. It leaves acute-vs-chronic rhinosinusitis and Pericarditis unresolved. This is the first algorithmic-ledger result that materially changes the headline number, but it remains an offline saved-trace enhancement rather than a live-confirmed workup policy.

### Live Graph-Bayes Rescue Confirmation

Notebook `24` tested whether Notebook `23`'s offline rescue policy survives inside a fresh live workup.

| System | Cases | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Original Notebook `13` artifact | 49 | 43/49 = 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Notebook `23` offline rescue | 49 | 47/49 = 0.959 | n/a | n/a | n/a | 6.96 |
| Notebook `24` fresh live base | 49 | 45/49 = 0.918 | 0.939 | 0.939 | 0.895 | 6.20 |
| Notebook `24` live rescue | 49 | 45/49 = 0.918 | 0.939 | 0.939 | 0.895 | 6.39 |

The rescue layer produced zero improvements and zero regressions against the fresh live base. It changed one wrong prediction to another wrong prediction and spent nine extra rescue requests across three already-correct cases. Therefore, Notebook `24` does not promote the graph/Bayes rescue layer. It does strengthen the evidence for the base LLM-led + MLP-stopped architecture by showing a second 49-case live trajectory at `45/49`.

### Offline Branching Trajectory Lab

Notebook `25` collected three additional rescue-disabled Notebook `13`-style base trajectories. Notebook `26` then analyzed five observed base trajectories: the original Notebook `13`, the Notebook `24` fresh base, and Notebook `25` replicates r01-r03.

| Run | Cases | Accuracy | Mean requests |
|---|---:|---:|---:|
| Notebook `13` frozen | 49 | 43/49 = 0.878 | 6.59 |
| Notebook `24` base | 49 | 45/49 = 0.918 | 6.20 |
| Notebook `25` r01 | 49 | 44/49 = 0.898 | 7.00 |
| Notebook `25` r02 | 49 | 42/49 = 0.857 | 7.02 |
| Notebook `25` r03 | 49 | 42/49 = 0.857 | 6.82 |

Across all five trajectories, `41/49` cases had the same final prediction, `8/49` had prediction instability, and `7/49` had correctness instability. Majority vote stayed at `43/49`, while oracle best-of-five reached `47/49`. This means the multi-agent opportunity is not naive voting; it is selective branching plus adjudication.

The strongest diagnostic Notebook `13` branching policy uses a sparse `hybrid_suspicion_v1` trigger, at most two observed alternate branches, and a Bayesian posterior judge. It reaches `47/49` with zero regressions, branch trigger rate `0.184`, mean selected requests `6.86`, and mean total branch requests `9.18`. This is not promoted as a live result because it reuses observed branches, but it gives a concrete next experiment: prospective fixed live branching with current-state suspicion features and graph/Bayes/MLP adjudication.

A quick post-hoc replay over the three Notebook `25` replicates also supports graph/Bayes as an adjudication signal. The strict conservative final override did not fire, but raw graph and raw Bayes top-1 heads improved the replicate accuracies from `44/49`, `42/49`, and `42/49` to `45/49`, `44/49`, and `44/49` with zero regressions. This should stay diagnostic, but it strengthens the case for graph/Bayes branch judging.

Notebook `27` completed that prospective live confirmation. Its own base branch reached `42/49`; targeted branching reached `43/49`, with two wins, one regression, a `9/49` trigger rate, and `11.45` mean total branch requests. The two wins, Myocarditis and Panic attack, show real branch-diversity value. The COPD regression shows that the raw-Bayes-only resolver is too eager to override a graph/Bayes-supported base answer.

The actual live branch pool had an oracle ceiling of only `44/49`, so completed LLM branches alone were not enough to reach the `47/49` target. However, raw graph and raw Bayes top-1 heads over the Notebook `27` base final states each reached `45/49` diagnostically, fixing Myocarditis, Panic attack, and Possible NSTEMI / STEMI without base-correct regressions. The next branching direction should therefore use graph/Bayes pseudo-candidates and a cautious resolver, not branch final predictions alone.

Notebook `28` tested this next direction. It trains `branch_trigger_mlp_v1` on train/validate synthetic partial evidence states, uses a threshold of `0.375`, and spawns up to three fresh LLM branches only when the learned gate fires. The resolver scores base and branch predictions together with graph, Bayes, and MLP pseudo-candidates, while protecting base answers that are graph rank `1` and Bayes rank `1`.

The live Notebook `28` result was `44/49`, with base at `42/49`, two wins, zero regressions, `0.959` top-3/top-5, `6.63` selected requests, and `9.96` total branch requests. It was not promoted, but it revealed a stronger next step: the scored candidate pool itself had only a `44/49` oracle, while adding ranked-differential candidates gives a `47/49` top-2 and `48/49` top-3 oracle. The next iteration should score ranked differential entries from base and branch workups, not only final top-1 branch answers.

## Error Analysis

Notebook `13` originally had six misses on the 49-case confirmation. Notebook `23` fixes four of them, so the remaining final-head errors after graph/Bayes rescue are:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | 8 | selected MLP stop |
| `test:62878` | Pericarditis | Anemia | 15 | agent stop |

The remaining failure pattern is narrower. Acute rhinosinusitis remains a close-neighbor confusion where the graph and model signals prefer Chronic rhinosinusitis. Pericarditis remains an evidence-state failure: the revealed evidence does not support Pericarditis strongly enough for the graph/Bayes rescue layer to recover it. A future live version would need targeted evidence acquisition for these specific confounders, not just final reranking.

In the fresh Notebook `24` live run, the remaining wrong cases are different:

| Case | True pathology | Live rescue prediction |
|---|---|---|
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis |
| `test:38475` | Acute COPD exacerbation / infection | Anemia |
| `test:62878` | Pericarditis | Anemia |
| `test:76022` | Panic attack | PSVT |

This mismatch explains why the offline rescue did not transfer cleanly: the saved-trace failure pattern changed in the fresh live trajectory. Notebook `24` fixed Croup, Influenza, and Unstable angina before rescue, but introduced a Panic attack regression and left COPD wrong with a different wrong answer.

## Claims We Can Make

Supported:

- A faithful DDXPlus deep learning baseline was implemented.
- DDXPlus initial evidence is insufficient for strong diagnosis.
- DDXPlus full evidence is near-ceiling for a neural classifier.
- Sequential evidence acquisition substantially improves over initial-evidence diagnosis on balanced live slices.
- MLP-guided stopping improves evidence efficiency relative to the LLM-only sequential baseline on the 24-case control.
- The final proposed system reaches high-80s accuracy with about 6.6 evidence requests on the 49-case confirmation.
- A train-derived graph-posterior final critic can improve the saved Notebook `13` 49-case trace from `43/49` to `44/49` without adding requests.
- A calibrated graph/Bayes rescue reranker can improve the saved Notebook `13` 49-case trace to `47/49` with `6.96` mean requests and zero regressions.
- A fresh Notebook `13`-style live run inside Notebook `24` reached `45/49` with `6.20` mean requests, supporting the robustness of the base architecture.
- Notebook `27` live targeted branching recovered Myocarditis and Panic attack from wrong base trajectories, but it reached only `43/49` overall and introduced one regression, so it is a partial confirmation rather than a promoted method.
- Notebook `28` live learned-gate branching improved its own base from `42/49` to `44/49` with zero regressions and identified ranked-differential candidate expansion as the next route to `47/49+`.

Not supported:

- The LLM final answer is generally better than a neural classifier.
- Multi-agent systems are better as a final claim; Notebooks `27` and `28` showed useful branch diversity and mathematical adjudication value but did not promote a branching policy.
- Direct MLP-guided question selection is better; notebook `14` rejected that.
- Hard MedKGI-style graph top-10 question shortlisting is better; notebook `17` rejected that.
- Offline Bayesian VOI replacement control is better; notebook `19` rejected that.
- Learned graph-context adjudication is validated; current Notebook `21` data are useful for diagnosis but not enough for a strict held-out learned adjudicator.
- Notebook `22` is held-out or live validated; it is currently an offline final-head enhancement candidate.
- Notebook `23` should be promoted as a live-confirmed improvement; Notebook `24` did not promote the rescue layer, so this claim is not supported.
- The method beats official DDXPlus RL baselines; those were not reproduced.
- The result is a definitive clinical benchmark.

## Final Interpretation

The project should be presented as an **evidence-efficient diagnostic workup system**, not as an unrestricted agentic-diagnosis win.

The strongest final statement is:

> We built a reproducible DDXPlus baseline ladder and proposed a structured sequential workup system where an LLM acquires evidence under a deterministic ledger and a partial-evidence MLP decides when enough information has been gathered. The frozen live 49-case confirmation reached `43/49` accuracy and `0.939` top-5 while requesting about `6.6` evidence fields per case; a fresh live run of the same backbone inside Notebook `24` reached `45/49` with `6.20` requests. Offline train-derived graph-ledger enhancements improve the saved trace to `44/49` with a simple graph critic and to `47/49` with a calibrated graph/Bayes rescue layer, but that rescue was not promoted by the fresh live confirmation. Prospective live branching recovered some wrong trajectories; Notebook `28` improved its own live base to `44/49` with zero regressions, and its candidate-pool analysis points to listwise ranked-differential adjudication as the next credible path to `47/49+`.

Updated graph-ledger statement:

> A calibrated offline graph/Bayes rescue layer over the same Notebook `13` trace reaches `47/49` accuracy with `6.96` mean requests and zero regressions. Notebook `24` did not confirm it as a live improvement, so the strongest defended live method remains the Notebook `13`-style backbone.

This is a credible course-project contribution because it includes:

- full deep learning baseline
- ceiling comparator
- sequential baseline
- matched evidence comparator
- stopping-policy ablation
- negative v2 ablation
- negative graph-shortlist ablation
- offline graph-posterior final critic
- calibrated graph/Bayes rescue reranker
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

### Slide 13. Negative Algorithmic Ablations

MLP-guided question shortlisting was tested and rejected. MedKGI-style graph top-10 shortlisting was also tested and rejected. A later graph-advisory version recovered some rare-disease failures but still underperformed the frozen Notebook 13 method. Bayesian VOI replacement control was also tested offline and rejected: its best fused run reached only `33/49` while asking `22.37` fields on average.

### Slide 14. Graph Critic

Notebook `20` improved ranking quality to `23/24` top-3/top-5 but lost top-1 accuracy. Notebook `21` showed graph contradiction strongly flags wrong top-1 predictions, but no non-oracle hand-rule adjudicator beat Notebook `13`.

Notebook `22` then used the graph after the workup, not during question selection: a conservative graph-posterior final critic improved the saved 49-case trace from `43/49` to `44/49` by fixing Croup with no regressions.

### Slide 15. Calibrated Rescue

Notebook `23` adds calibrated graph/Bayes/MLP rescue over Notebook `13`: prior recovery, graph critic, and up to three discriminator questions for suspicious early stops. It reaches `47/49` with `6.96` mean requests and zero regressions.

### Slide 16. Live Rescue Confirmation

Notebook `24` tested that rescue live. The rescue did not improve over the fresh live base: both ended at `45/49`, while rescue increased mean requests from `6.20` to `6.39`. This is a useful negative confirmation, not a failure of the whole project, because the base architecture itself reproduced strongly.

### Slide 17. Branching Trajectory Lab

Notebook `26` shows why multi-agent branching is interesting: majority vote over five trajectories stays at `43/49`, but oracle best-of-five reaches `47/49`. A sparse suspicious-state trigger plus two observed branches and a Bayes judge reaches `47/49` diagnostically with zero regressions. This motivated Notebook `27`, which tested fixed targeted branching prospectively.

### Slide 18. Live Targeted Branching Confirmation

Notebook `27` is that prospective live test. It improved its own live base from `42/49` to `43/49`, with two wins and one regression. The result is not promoted, but it proves branches can recover real lock-in failures.

### Slide 19. Error Analysis

Errors are mainly wrong-belief convergence and under-adjudicated confounders, not just insufficient evidence budget. Notebook `27` shows that confidence/contradiction triggers miss consensus wrong-answer cases. Notebook `28` shows that some of those misses are already present as rank-2/rank-3 differential entries, but were not scored as final candidates.

### Slide 20. Claims And Limitations

Strong claim: evidence-efficient sequential workup with MLP-guided stopping. Limitation: not a proof of multi-agent superiority or RL baseline dominance.

### Slide 21. Future Work

Build the next listwise differential adjudicator. The candidate pool should include base and branch ranked top-5 diagnoses, graph top-5, Bayes top-5, MLP top-5, and confounder challengers. Notebook `28`'s post-hoc oracle suggests this expansion can recover the `47/49+` target if the resolver can choose safely without test-label tuning.

## Key Files

- `notebooks/01_one_shot_classifier_baselines.ipynb`
- `notebooks/07_full_evidence_one_shot_comparator.ipynb`
- `notebooks/08_cost_sensitive_sequential_lambda_sweep.ipynb`
- `notebooks/10_partial_evidence_one_shot_comparator.ipynb`
- `notebooks/12_stopping_policy_ablation.ipynb`
- `notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb`
- `notebooks/14_hybrid_v2_mlp_discriminative_shortlist.ipynb`
- `notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb`
- `notebooks/17_live_medkgi_graph_shortlist_pilot.ipynb`
- `notebooks/18_graph_advisory_hybrid_shortlist.ipynb`
- `notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb`
- `notebooks/20_llm_led_graph_ledger_context.ipynb`
- `notebooks/21_graph_context_policy_lab.ipynb`
- `notebooks/22_graph_posterior_final_adjudicator.ipynb`
- `notebooks/23_calibrated_graph_bayes_rescue_reranker.ipynb`
- `notebooks/24_live_graph_bayes_rescue_confirmation.ipynb`
- `notebooks/25_live_base_trajectory_replicates.ipynb`
- `notebooks/26_offline_branching_trajectory_lab.ipynb`
- `notebooks/27_live_targeted_branching_confirmation.ipynb`
- `notebooks/28_mlp_gated_confounder_graph_bayes_branching.ipynb`
- `reports/final_results_summary.md`
- `reports/hybrid/live_selected_hybrid_stopping_confirmation.md`
- `reports/hybrid/hybrid_v2_mlp_discriminative_shortlist_report.md`
- `reports/algorithmic_ledger/live_medkgi_graph_shortlist_pilot.md`
- `reports/algorithmic_ledger/graph_advisory_hybrid_shortlist_report.md`
- `reports/algorithmic_ledger/bayesian_voi_ledger_offline_report.md`
- `reports/algorithmic_ledger/llm_led_graph_ledger_context_report.md`
- `reports/algorithmic_ledger/graph_context_policy_lab_report.md`
- `reports/algorithmic_ledger/graph_posterior_final_adjudicator_report.md`
- `reports/algorithmic_ledger/calibrated_graph_bayes_rescue_reranker_report.md`
- `reports/algorithmic_ledger/live_graph_bayes_rescue_confirmation_report.md`
- `reports/algorithmic_ledger/live_base_trajectory_replicates_report.md`
- `reports/algorithmic_ledger/offline_branching_trajectory_lab_report.md`
- `reports/algorithmic_ledger/live_targeted_branching_confirmation_report.md`
- `reports/algorithmic_ledger/mlp_gated_confounder_graph_bayes_branching_report.md`
