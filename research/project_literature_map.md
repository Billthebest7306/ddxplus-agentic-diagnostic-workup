# Project Literature Map

Last updated: 2026-05-06

Purpose: consolidate the papers and external systems we have been using throughout the DDXPlus project, and explicitly map what each one contributed to our design choices, baselines, claims, and next-step plans.

This is not a generic bibliography. It is a project-specific map of how each paper affected our work.

## 1. DDXPlus: A New Dataset For Automatic Medical Diagnosis

Source:

- [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://arxiv.org/abs/2205.09148)
- [NeurIPS paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)
- [Official GitHub](https://github.com/mila-iqia/ddxplus)

What the paper contributes:

- Defines the DDXPlus dataset:
  - 1.3M synthetic patients
  - 49 pathologies
  - 223 root evidence items
  - binary, categorical, and multi-choice evidence
  - initial evidence
  - full evidence list
  - differential diagnosis
- Frames diagnosis as interactive evidence acquisition:
  - start from initial evidence
  - ask about missing symptoms/antecedents
  - predict pathology or differential
- Provides established baselines:
  - BASD
  - AARLC
- Reports interaction length, evidence recall, pathology accuracy, and differential diagnosis metrics.

How we used it:

- Chose DDXPlus as the primary dataset.
- Used the official 49 pathology label space.
- Used the official evidence metadata and root evidence schema.
- Built `case_id = "<split>:<source_row_index>"` around official splits.
- Implemented BASD-style slot encoding for one-shot and partial-evidence MLPs.
- Used initial-evidence-only diagnosis as the baseline task.
- Used full-evidence diagnosis as a ceiling comparator.
- Treated evidence acquisition efficiency as a core research variable.

Project impact:

- This is the foundation of the whole project.
- It also limits broad novelty claims because the original paper already studies automatic diagnosis systems that interact with patients.

## 2. BASD

Source:

- BASD is discussed in the DDXPlus paper and original DDXPlus baselines.

What it contributes:

- A supervised neural baseline family using structured patient evidence.
- MLP-style diagnosis over DDXPlus/BASD observation slots.
- Evidence state convention:
  - unknown
  - present
  - absent

How we used it:

- Built one-shot MLP classifier notebooks around BASD-style slot encoding.
- Used hidden sizes `[2048, 2048, 2048]` for faithful architecture flavor.
- Used age, sex, and evidence slots rather than a toy bag-of-words representation.
- Trained:
  - initial-evidence one-shot MLP
  - full-evidence MLP ceiling
  - partial-evidence MLP comparator
  - online partial-evidence MLP feedback for Notebook `13`

Project impact:

- BASD gave the project a faithful deep-learning baseline.
- The partial-evidence MLP became the key neural component of our hybrid system.

## 3. AARLC / Adaptive Alignment

Source:

- [Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification](https://arxiv.org/abs/2112.00733)

What the paper contributes:

- Treats automatic medical diagnosis as:
  - symptom inquiry
  - disease classification
  - stopping criterion
- Separates the evidence-inquiring policy from the disease classifier.
- Uses disease-distribution entropy to align inquiry and classification.
- Shows that negative symptoms also help because they rule out diagnoses.
- Optimizes for accurate diagnosis with fewer inquiry turns.

How we used it:

- Justified separating:
  - LLM evidence-acquisition policy
  - MLP diagnostic belief head
- Used MLP entropy, confidence, and margin as stopping signals.
- Motivated Notebook `12` stopping-policy ablation.
- Motivated Notebook `13` online MLP-guided stop policy.
- Motivated the graph-ledger idea that absence and contradiction should be explicitly modeled.

Project impact:

- AARLC is one of the strongest methodological ancestors of our hybrid design.
- It also shows that RL-based evidence acquisition exists, so our novelty is not "sequential evidence acquisition" by itself.

## 4. Active Feature Acquisition

Source:

- [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://papers.nips.cc/paper/7411-joint-active-feature-acquisition-and-classification-with-variable-size-set-encoding)

What the paper contributes:

- Formalizes test-time feature acquisition:
  - collect a feature
  - pay cost
  - stop and predict when enough information is available
- Motivates diagnosis as a natural active feature acquisition problem.
- Notes that acquiring all features/tests can be costly.
- Notes that irrelevant features can add noise and destabilize predictions.
- Jointly learns classifier and acquisition policy in the paper's full method.

How we used it:

- Framed DDXPlus evidence requests as active feature acquisition.
- Treated request count as an evidence cost.
- Built lambda/cost-sensitive sequential experiments.
- Interpreted Notebook `15` finding that more evidence sometimes causes correct-to-wrong drift.
- Motivated future graph action scoring:
  - information value
  - redundancy penalty
  - generic evidence penalty

Project impact:

- This paper supports the scientific question:
  - Can we approach full-evidence diagnosis using only a targeted subset of evidence?

## 5. Cost-Sensitive Feature Acquisition And Classification

Source:

- [Cost-sensitive feature acquisition and classification](https://www.sciencedirect.com/science/article/pii/S0031320306004808)

What the paper contributes:

- Defines diagnosis-like prediction as a balance between:
  - sensing/test cost
  - misclassification cost
  - reward for correct classification
  - when to stop acquiring features
- Discusses POMDP and myopic approaches for computational feasibility.

How we used it:

- Justified evidence-cost lambda sweeps.
- Justified treating max-request caps as safety ceilings rather than the real experimental variable.
- Motivated the move from arbitrary budgets to cost-sensitive stopping.
- Supports the future graph-ledger stop certificate:
  - stop when remaining expected value is below cost.

Project impact:

- This paper grounded our cost-sensitive framing.

## 6. DDxT

Source:

- [DDxT: Deep Generative Transformer Models for Differential Diagnosis](https://arxiv.org/abs/2312.01242)

What the paper contributes:

- A strong transformer-based DDXPlus static/differential diagnosis model.
- Reports near-ceiling DDXPlus diagnosis performance in full-information style settings.

How we used it:

- Used it as evidence that DDXPlus is nearly solved when all evidence is available.
- Used it to motivate why our project should not compete on raw full-evidence classification alone.
- Helped interpret our full-evidence one-shot MLP result near `0.996` accuracy.

Project impact:

- DDxT sets a high ceiling and forces our project to focus on incomplete-evidence workup and evidence efficiency.

## 7. LLM-Driven Medical Document Analysis With LoRA LLaMA-v3

Source:

- [LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis](https://arxiv.org/abs/2506.19702)

What the paper contributes:

- LoRA fine-tuned LLaMA-v3 on DDXPlus.
- Reports approximately:
  - `99.81%` pathology prediction accuracy
  - `99.46%` differential diagnosis accuracy
  - `99.94%` GTPA
- Uses an LLM as a supervised/fine-tuned DDXPlus diagnosis model.

How we used it:

- Resolved Hassan's claim about 99%+ DDXPlus accuracy using an LLM.
- Clarified that 99%+ DDXPlus results are full-information or supervised/fine-tuned diagnosis, not our incomplete-evidence sequential setting.
- Used it as another full-information ceiling reference.

Project impact:

- Reinforces that our project should not claim superiority in static full-information diagnosis.
- Supports the need to focus on evidence acquisition and workup efficiency.

## 8. MEDDxAgent

Source:

- [MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](https://arxiv.org/abs/2502.19175)

What the paper contributes:

- Closest LLM-agent interactive DDx prior work.
- Uses:
  - `DDxDriver` orchestrator
  - history-taking simulator
  - knowledge retrieval agent
  - diagnosis strategy agent
- Evaluates on:
  - DDXPlus
  - iCraft-MD
  - RareBench
- Reports DDXPlus interactive GPT-4o results:
  - `0.74` GTPA@1 at 5 questions
  - `0.78` at 10 questions
  - `0.86` at 15 questions

How we used it:

- Reframed our novelty claims.
- Stopped claiming "LLM interactive DDx on DDXPlus" as novel.
- Used it as the closest external LLM-agent benchmark.
- Compared our Notebook `13` 49-case result contextually:
  - our method: `43/49 = 0.878`, `6.59` mean structured evidence fields
  - MEDDxAgent: `0.86` GTPA@1 at 15 questions on 100 sampled DDXPlus cases
- Identified the need for a future matched comparison before claiming direct superiority.

Project impact:

- MEDDxAgent forced the project to become more precise:
  - DDXPlus-native structured evidence ledger
  - MLP feedback
  - matched-evidence evaluation
  - evidence efficiency

## 9. MediQ

Source:

- [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](https://arxiv.org/abs/2406.00922)

What the paper contributes:

- Shows that static medical QA misses interactive information-seeking.
- Demonstrates that simply prompting LLMs to ask questions can degrade performance.
- Studies abstention/confidence strategies for deciding whether to ask or answer.
- Separates:
  - when to ask
  - what to ask
  - final answer quality

How we used it:

- Justified our concern that LLM-only stop policies are weak.
- Supported Notebook `12` stopping-policy ablation.
- Supported Notebook `13` MLP-guided stopping.
- Motivated the graph stop certificate:
  - do not rely only on the LLM's self-confidence.

Project impact:

- MediQ validates the problem we observed empirically: question asking and stopping are hard for LLMs.

## 10. ALFA

Source:

- [ALFA: Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning](https://arxiv.org/abs/2502.14860)

What the paper contributes:

- Shows LLM question quality can be improved by decomposing "good question" into structured attributes.
- Attributes include:
  - clarity
  - focus
  - answerability
  - medical accuracy
  - diagnostic relevance
  - avoiding differential-diagnosis bias
- Reports improved question quality and reduced diagnostic errors through attribute-based alignment.

How we used it:

- Inspired graph-derived question-quality criteria:
  - legal
  - answerable
  - discriminative
  - non-redundant
  - contradiction-resolving
  - not generic
- Supported the idea that the LLM should receive structured reasons for each candidate evidence request.

Project impact:

- ALFA supports our move from raw prompting to structured question-selection control.

## 11. KG4Diagnosis

Source:

- [KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis](https://arxiv.org/abs/2412.16833)

What the paper contributes:

- Combines LLMs with medical knowledge graph construction.
- Uses hierarchical multi-agent diagnosis:
  - general practitioner agent
  - specialist agents
- Uses KG constraints to reduce hallucination and organize reasoning.

How we used it:

- Supported the broader claim that KGs can constrain medical LLM diagnosis.
- Informed our future multi-agent thinking.
- Clarified that "KG + medical LLM" is not novel by itself.

Project impact:

- KG4Diagnosis supports graph grounding, but also warns us to make our graph contribution DDXPlus-specific.

## 12. MedRAG

Source:

- [MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot](https://arxiv.org/abs/2502.04413)

What the paper contributes:

- Uses a knowledge-graph-enhanced RAG model for healthcare diagnosis and recommendations.
- Evaluates on DDXPlus and a private chronic pain dataset.
- Constructs a hierarchical diagnostic KG.
- Uses KG reasoning to retrieve more specific diagnosis/treatment recommendations and produce follow-up questions.

How we used it:

- Answered the question of whether graph/KG approaches have been used with DDXPlus.
- Confirmed that a broad "graph-based DDXPlus diagnosis" claim is not novel.
- Helped narrow our claim:
  - our graph is not retrieval-oriented
  - our graph is the live evidence-acquisition ledger over official DDXPlus fields

Project impact:

- MedRAG is important prior work for DDXPlus + KG.
- It increases the need for our contribution to emphasize legal sequential evidence reveal, MLP feedback, and matched-evidence evaluation.

## 13. MedKGI

Source:

- [MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring](https://arxiv.org/abs/2512.24181)

What the paper contributes:

- Closest prior work to our planned graph algorithmic ledger.
- Combines:
  - medical KG alignment
  - information-gain-based symptom selection
  - OSCE-style structured diagnostic records
  - hypothesis-driven termination
- Reports that removing KG, removing the clinical record, or replacing information-gain question selection harms performance.

How we used it:

- Strongly influenced the graph-ledger design.
- Motivated implementing a MedKGI-style information-gain baseline before our enhanced graph-cut ledger.
- Motivated:
  - disease/evidence subgraph
  - information-gain action scoring
  - structured state
  - hypothesis/stagnation termination

Project impact:

- MedKGI is the prior graph-algorithm benchmark we should compare against.
- Our next graph work should include:
  - MedKGI-style IG baseline
  - our enhanced graph-cut/discriminator ledger

## 14. TriMediQ

Source:

- [Triplet-Structured Knowledge Integration for Multi-Turn Medical Reasoning](https://arxiv.org/abs/2510.03536)

What the paper contributes:

- Shows that multi-turn medical reasoning suffers when clinical facts are scattered in raw dialogue logs.
- Converts patient responses into triplets.
- Builds a patient-specific KG.
- Uses graph encoding to improve multi-hop reasoning with a frozen LLM.

How we used it:

- Supported the idea that the ledger should be graph-structured rather than a flat text history.
- In DDXPlus, we do not need an LLM triplet extractor because evidence IDs and values are already structured.
- Motivated treating each DDXPlus evidence reveal as a structured graph fact:
  - patient has evidence
  - patient lacks evidence
  - evidence supports diagnosis
  - evidence contradicts diagnosis

Project impact:

- TriMediQ supports the graph-ledger format and the need for patient-specific graphs.

## 15. MedKGI-Style Baseline Versus Our Enhanced Graph-Cut Ledger

This is the next planned research ladder.

### 15.1 MedKGI-Style Baseline

Purpose:

- implement the graph/information-gain approach inspired by prior work.

Behavior:

```text
current top diagnoses
-> DDXPlus train-derived disease/evidence graph
-> score legal evidence roots by information gain
-> shortlist top roots
-> LLM chooses request
-> stop using MLP confidence/stability/simple IG condition
```

This should avoid our custom graph-cut additions.

### 15.2 Our Enhanced Graph-Cut Ledger

Purpose:

- test whether our DDXPlus-specific graph contribution improves over the prior-style IG baseline.

Additions:

- unresolved disease-pair graph
- evidence roots as "cuts" over competing diagnoses
- contradiction edges
- drift barrier
- disagreement subgraph
- graph stop certificate
- redundancy and genericness penalties

Planned comparison:

```text
Notebook 13 hybrid v1
vs
MedKGI-style IG graph baseline
vs
Our enhanced graph-cut ledger
```

This is the clearest future experimental structure.

## 16. Current Best Project Framing

After reviewing all of the above, the strongest project framing is:

> We study evidence-efficient diagnostic workup on DDXPlus. Starting from faithful neural one-shot and sequential LLM baselines, we develop a structured DDXPlus-native hybrid system where an LLM acquires evidence, a partial-evidence MLP estimates diagnostic belief, and a ledger controls stopping. The next phase extends this ledger into a graph algorithm that compares a MedKGI-style information-gain baseline against our DDXPlus-specific differential graph-cut strategy.

Claims we should avoid:

- "We invented interactive diagnosis."
- "We invented graph-based diagnosis."
- "We invented LLM agents for DDXPlus."
- "We beat MEDDxAgent" without a matched evaluation.

Claims we can defend:

- "We built a strong, reproducible baseline ladder."
- "We achieved competitive evidence-efficient DDXPlus workup on a balanced 49-case pilot."
- "We separated evidence acquisition from final diagnosis using matched-evidence MLP comparisons."
- "We identified remaining failures as trajectory/drift issues rather than simple stop-threshold issues."
- "We propose a DDXPlus-native graph evidence ledger to address those failures."

## 17. Bayesian Medical Inquiry And Value Of Information

Source:

- [A Bayesian Approach for Medical Inquiry and Disease Inference in Automated Differential Diagnosis](https://arxiv.org/abs/2110.08393)
- [Efficient Test Selection in Active Diagnosis via Entropy Approximation](https://arxiv.org/abs/1207.1418)
- [Cost-sensitive feature acquisition and classification](https://www.sciencedirect.com/science/article/pii/S0031320306004808)

What these papers contribute:

- Bayesian medical inquiry treats diagnosis and question selection as one loop:
  - infer disease posterior from observed evidence
  - choose the next question by Bayesian experimental design
  - update the posterior after the answer
- Active diagnosis literature frames the next test as the one that maximally reduces diagnostic uncertainty.
- Cost-sensitive feature acquisition frames the stop decision as a tradeoff between expected diagnostic benefit and evidence/test cost.
- These papers make clear that exact optimal sequential test selection is often hard, but greedy/myopic value-of-information is a standard, defensible approximation.

How this changes our project:

- Notebooks `17` and `18` showed that graph edge ranking alone is not enough.
- The next algorithmic version should maintain a full posterior over all `49` DDXPlus pathologies.
- Evidence selection should be based on expected posterior improvement:

```text
VOI(root) = current_entropy - expected_entropy_after_revealing(root)
```

- Stopping should depend on:
  - posterior confidence
  - posterior margin
  - entropy
  - remaining VOI
  - contradiction score
  - MLP/Bayes agreement

Project impact:

- This becomes the strongest mathematical basis for the next phase.
- The graph ledger can be reframed as a support/contradiction feature inside a Bayesian VOI ledger.
- The next notebook should be offline first, because DDXPlus lets us reveal requested fields deterministically without API calls.

Planned notebook:

```text
notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb
```

Research note:

- [bayesian_voi_algorithmic_ledger_research.md](bayesian_voi_algorithmic_ledger_research.md)

## 18. Revised Algorithmic Direction After Notebooks 17 And 18

Notebook `17` and Notebook `18` were useful negative ablations.

Observed result:

| Method | 24-case accuracy | Mean requests | Interpretation |
|---|---:|---:|---|
| Notebook `13` hybrid v1 | 0.917 | 6.58 | Frozen proposed method |
| Notebook `17` hard graph shortlist | 0.833 | 6.21 | Hard graph control over-pruned |
| Notebook `18` graph-advisory shortlist | 0.875 | 7.67 | Safer than hard graph, still worse than Notebook `13` |

Meaning:

- graph information is useful but not sufficient
- the system needs belief correction, not only better ranking
- the next ledger should explicitly model `P(diagnosis | evidence)`
- question value should be computed against the posterior, not just graph edge importance

Revised path:

```text
Notebook 13 frozen proposed method
-> Notebook 19 offline Bayesian VOI ledger
-> only if promising: Notebook 20 live Bayesian-VOI-advisory LLM
```

The core research question becomes:

> Can a DDXPlus-native Bayesian value-of-information ledger improve evidence-efficient diagnostic workup by explicitly modeling posterior belief, contradiction, and remaining information value?
