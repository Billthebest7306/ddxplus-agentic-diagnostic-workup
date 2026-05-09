# Our Project Versus MEDDxAgent

Purpose: compare our DDXPlus hybrid evidence-ledger project against MEDDxAgent at the system level, so we can clearly explain what is overlapping, what is different, and what claims remain defensible.

Primary source:

- [MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](https://arxiv.org/abs/2502.19175)

Project reference points:

- Notebook 01: initial-evidence one-shot classifier suite
- Notebook 05/08: refined and cost-sensitive single-agent sequential baselines
- Notebook 10: partial-evidence matched MLP comparator
- Notebook 11/13: online MLP feedback hybrid v1
- Notebook 14: hybrid v2 MLP-discriminative shortlist candidate, rejected
- Notebook 15: offline stop-policy sensitivity and trajectory analysis

## 1. High-Level Comparison

| Dimension | MEDDxAgent | Our Project |
|---|---|---|
| Main goal | Modular LLM framework for explainable interactive DDx | DDXPlus-native evidence-efficient diagnostic workup with ledger and neural feedback |
| Core setting | Interactive differential diagnosis across multiple datasets | Structured DDXPlus evidence acquisition from initial evidence |
| Main controller | `DDxDriver` orchestrator | Deterministic evidence ledger plus LLM controller |
| Evidence access | LLM patient simulator answers natural-language questions from full patient profile | Legal DDXPlus root-evidence requests reveal exact structured evidence values |
| Modules | History-taking simulator, knowledge retrieval agent, diagnosis strategy agent | Evidence ledger, LLM question selector, partial-evidence MLP belief head, stopping/adjudication logic |
| External retrieval | PubMed/Wikipedia retrieval agent | No external medical retrieval in current version |
| Neural classifier | LLM diagnosis strategy, dynamic few-shot retrieval | BASD-style MLP trained on DDXPlus slot encodings |
| Main evaluation idea | GTPA@k and average rank across iterative DDx | Accuracy/top-k/F1 versus evidence requests, matched-evidence MLP, full-evidence ceiling |
| Datasets | DDXPlus, iCraft-MD, RareBench | DDXPlus only so far |
| Main novelty area | Modular LLM agent architecture for interactive DDx | Structured legal evidence ledger plus online LLM-MLP feedback and evidence-efficiency analysis |

## 2. Where The Projects Overlap

Both projects share the same broad thesis:

> Differential diagnosis should not be evaluated only as one-shot classification from complete patient profiles. It should be studied as an iterative process where the system starts with incomplete information, gathers more evidence, and updates its diagnosis.

Specific overlaps:

- both use DDXPlus
- both use LLMs for interactive diagnostic reasoning
- both start from incomplete patient information
- both maintain an evolving patient state
- both evaluate top-k diagnostic quality
- both care about traceability/explainability
- both show that more patient information improves diagnosis

This overlap is significant. We should not present our work as if MEDDxAgent does not exist.

## 3. Key Difference: Natural-Language Simulation Versus Structured Evidence Ledger

MEDDxAgent uses a history-taking simulator:

```text
Doctor LLM asks freeform question
Patient LLM sees full profile
Patient LLM answers in natural language
DDxDriver updates patient profile
```

Our project uses a deterministic evidence ledger:

```text
LLM chooses a legal DDXPlus root evidence id
Ledger reveals exact present/absent/value state
Ledger records request history and visible evidence
MLP encodes the updated partial evidence state
LLM/MLP decide whether to stop or continue
```

Why this matters:

- MEDDxAgent is closer to realistic clinical conversation.
- Our system is more controlled and auditable.
- MEDDxAgent has ambiguity from LLM-patient natural-language answers.
- Our system avoids patient-simulator hallucination by revealing exact DDXPlus fields.
- MEDDxAgent can ask flexible questions.
- Our system can rigorously measure exactly which evidence fields were acquired.

This is the central distinction.

## 4. Key Difference: Modular LLM Agents Versus Hybrid Neural Feedback

MEDDxAgent's diagnostic intelligence mainly comes from LLM modules:

- diagnosis strategy agent
- retrieval agent
- orchestrator instructions
- dynamic few-shot examples
- chain-of-thought prompting

Our strongest method, Notebook 13, uses a hybrid control loop:

```text
Partial DDXPlus evidence state
  -> BASD-style MLP probabilities
  -> confidence / margin / entropy / stability
  -> stopping rule and final-head adjudication
```

The LLM still chooses evidence requests, but the MLP provides a grounded DDXPlus-trained belief signal.

This gives us a different research question:

> Can a DDXPlus-trained neural diagnostic head make an LLM workup more evidence-efficient by deciding when enough evidence has been acquired?

MEDDxAgent does not focus on this LLM-plus-neural-diagnostic-head stopping mechanism.

## 5. Key Difference: Matched-Evidence Decomposition

One of our most important evaluation ideas is the matched-evidence comparator.

For each sequential case:

```text
Evidence requested by the agent
  -> build a one-shot MLP input using exactly that evidence
  -> compare MLP diagnosis against LLM final diagnosis
```

This separates two sources of value:

- value from asking better evidence questions
- value from reasoning better over the acquired evidence

MEDDxAgent does not emphasize this decomposition. It evaluates the end-to-end agent result, but does not isolate whether performance comes from:

- history-taking quality
- retrieval quality
- diagnosis strategy quality
- prompt examples
- final LLM reasoning

Our matched-evidence setup is therefore one of the strongest methodological distinctions.

## 6. Key Difference: Evidence Efficiency As A First-Class Outcome

MEDDxAgent evaluates 5, 10, and 15 question settings and reports that gains plateau around 10-15 questions.

Our project makes evidence efficiency more central:

- cost-sensitive lambda sweeps
- request distribution plots
- stop-before-cap rate
- mean and median requested evidence fields
- offline stopping ablation
- live confirmation of a selected stop rule
- stop-threshold sensitivity analysis

Notebook 13 final 49-case result:

| Metric | Result |
|---|---:|
| Top-1 accuracy | 43/49 = 0.878 |
| Top-3 accuracy | 0.918 |
| Top-5 accuracy | 0.939 |
| Macro-F1 | 0.845 |
| Mean requested fields | 6.59 |
| Median requested fields | 5 |
| Stop-before-cap rate | 0.980 |

This makes our current story:

> We can reach MEDDxAgent-like DDXPlus interactive accuracy while requesting fewer structured evidence fields, using a smaller OpenAI model and a DDXPlus-trained MLP stopping signal.

Caveat: this is not a formal apples-to-apples comparison because sample size, case selection, model choice, and interaction format differ.

## 7. Numerical Comparison With Strong Caveats

| System | Setting | Model | Cases | Evidence Budget | Top-1 / GTPA@1 |
|---|---|---|---:|---:|---:|
| MEDDxAgent | DDXPlus interactive | GPT-4o | 100 | 5 questions | 0.74 |
| MEDDxAgent | DDXPlus interactive | GPT-4o | 100 | 10 questions | 0.78 |
| MEDDxAgent | DDXPlus interactive | GPT-4o | 100 | 15 questions | 0.86 |
| Our Notebook 13 | DDXPlus structured evidence | gpt-4.1-mini + MLP | 49 | 6.59 mean fields | 0.878 |

This looks favorable for us, but we must not overclaim.

Differences that prevent direct leaderboard-style comparison:

- MEDDxAgent uses 100 sampled cases; our final pilot uses 49 one-per-class cases.
- MEDDxAgent uses natural-language simulated history taking; ours uses exact structured DDXPlus field reveals.
- MEDDxAgent uses GPT-4o for its strongest result; ours uses gpt-4.1-mini plus a trained MLP.
- MEDDxAgent reports GTPA@1 over ranked DDx; our Notebook 13 top-1 metric is exact pathology accuracy over fixed DDXPlus labels.
- MEDDxAgent may ask broader natural-language questions; our request unit is a DDXPlus evidence root.

Defensible phrasing:

> On our controlled 49-case DDXPlus structured-evidence pilot, the hybrid ledger system achieved 0.878 top-1 accuracy with 6.59 mean evidence requests, which is competitive with the range of MEDDxAgent's reported DDXPlus interactive results. Because the setup and sampling differ, this should be treated as contextual comparison rather than a direct superiority claim.

## 8. What MEDDxAgent Makes Less Novel

After reading MEDDxAgent, these claims are weak:

- "We are the first to use LLM agents for DDXPlus diagnosis."
- "We are the first to do interactive DDx with incomplete information."
- "We are the first to use an orchestrator for differential diagnosis."
- "The novelty is simply that we ask follow-up questions."
- "The novelty is simply that the system is agentic."

MEDDxAgent already covers those broad claims.

## 9. What Remains Distinctive In Our Project

Our more defensible contributions are:

### 9.1 DDXPlus-Native Legal Evidence Ledger

We interact with DDXPlus as a structured environment:

- root evidence IDs
- decoded question text
- present/absent/value states
- hidden versus visible evidence
- request history
- legal/gated action management

MEDDxAgent uses a more general patient-profile simulator rather than this exact DDXPlus evidence-root ledger.

### 9.2 Online Neural Belief Feedback

Notebook 13 uses an MLP after each reveal to compute:

- top-k diagnosis
- confidence
- margin
- entropy
- stability
- agreement with LLM

These signals influence stopping and final diagnosis selection.

MEDDxAgent does not use a DDXPlus-trained MLP belief head as an online verifier/stopping signal.

### 9.3 Matched-Evidence MLP Comparator

Our matched-evidence comparator is scientifically useful because it asks:

> If the LLM gathered these exact evidence fields, would a neural classifier diagnose better from the same information?

This reframes the agent as an evidence acquisition policy, not necessarily the final diagnostic reasoner.

### 9.4 Evidence-Efficiency And Stop Policy Analysis

Notebook 12, 13, and 15 build a specific story:

- offline ablation suggested MLP-guided stopping was efficient
- live confirmation showed strong 49-case performance
- threshold sensitivity showed the selected stop rule is near a plateau
- error analysis showed failures are more about trajectory and belief drift than simple under-requesting

This is a more detailed stop-policy/evidence-efficiency analysis than MEDDxAgent provides.

## 10. Where MEDDxAgent Is Stronger Than Our Project

MEDDxAgent has important strengths:

- evaluates three datasets, not only DDXPlus
- uses 100 cases per dataset
- includes knowledge retrieval from PubMed/Wikipedia
- evaluates multiple LLM backbones
- includes dynamic few-shot diagnosis
- includes broader DDx benchmark framing
- has an accepted ACL 2025 paper-level structure

Our project is currently weaker on:

- statistical sample size
- multi-dataset generalization
- model ablations
- external medical knowledge retrieval
- formal published-benchmark comparability
- real clinical realism

This matters for final presentation. We should present our work as a deep course-project implementation and controlled research pilot, not as a claim of surpassing MEDDxAgent.

## 11. Research Position After MEDDxAgent

The best research question should shift slightly.

Weak framing:

> Can an LLM agent do interactive DDXPlus diagnosis?

Better framing:

> Can a structured DDXPlus evidence ledger plus online neural diagnostic feedback make LLM-led diagnostic workup more evidence-efficient under incomplete evidence?

Even better:

> In DDXPlus interactive diagnosis, how much of sequential-system performance comes from evidence acquisition versus final LLM reasoning, and can an online partial-evidence MLP improve stopping efficiency without sacrificing accuracy?

This framing keeps our work distinct from MEDDxAgent.

## 12. Implications For Next Work

MEDDxAgent suggests several possible extensions, but we should be selective.

Worth considering later:

- compare our final method against a MEDDxAgent-style prompt baseline on the same 49 cases
- add a dynamic similar-case few-shot diagnosis component
- add optional retrieval for external medical explanations
- add progress/rank-improvement metrics like MEDDxAgent's progress metric
- test whether algorithmic ledger signals can reduce hard-case drift

Not recommended immediately:

- copying MEDDxAgent's whole modular multi-agent architecture
- switching away from our structured DDXPlus ledger
- adding PubMed retrieval before finishing our ledger/hybrid claims
- trying to claim broad superiority without an apples-to-apples sample

## 13. Final Takeaway

MEDDxAgent is the closest prior work to our agentic diagnosis idea. It reduces the novelty of broad "LLM interactive DDx" claims, but it does not erase our project.

Our project remains defensible if framed as:

> a controlled DDXPlus-native study of structured evidence acquisition, where a deterministic ledger and online partial-evidence MLP provide stopping and diagnostic feedback to a single LLM workup agent, with matched-evidence evaluation to separate evidence-gathering value from final reasoning value.

