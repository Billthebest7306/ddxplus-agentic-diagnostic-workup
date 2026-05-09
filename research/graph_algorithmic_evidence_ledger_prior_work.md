# Graph Algorithmic Evidence Ledger: Prior Work Deep Dive

Last updated: 2026-05-06

Purpose: ground the planned graph-based algorithmic evidence ledger in existing research instead of inventing it from scratch. This note identifies what has already been done, what design ideas are worth borrowing, and where our DDXPlus-specific contribution can still sit.

## 1. Executive Summary

The graph-ledger idea is research-grounded, but it is not entirely new. Several recent systems already combine interactive diagnosis, structured state, knowledge graphs, and information-guided questioning.

The most important prior work for us is **MedKGI**:

- [MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring](https://arxiv.org/abs/2512.24181)

MedKGI is close to the graph-ledger idea because it combines:

- medical knowledge graph alignment
- information-gain-based symptom inquiry
- structured diagnostic records
- hypothesis-driven termination

That means we should not claim novelty as simply:

> "We use a graph for interactive diagnosis."

The stronger project-specific claim is:

> We build a DDXPlus-native graph evidence ledger over the official structured root-evidence schema, combine it with a partial-evidence BASD-style MLP belief head, and use graph-derived support, contradiction, unresolved-pair, and expected-information signals to control an LLM evidence-acquisition agent.

This is narrower, more defensible, and better aligned with what we have already built.

## 2. Main Sources Reviewed

| Source | Link | Why It Matters |
|---|---|---|
| DDXPlus | [NeurIPS 2022 paper](https://papers.nips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf) | Defines DDXPlus as structured evidence acquisition with pathology/differential prediction. |
| AARLC / Adaptive Alignment | [arXiv 2112.00733](https://arxiv.org/abs/2112.00733) | Separates symptom inquiry from disease classification and uses classifier entropy to align the two. |
| Active Feature Acquisition | [NeurIPS 2018](https://papers.nips.cc/paper/7411-joint-active-feature-acquisition-and-classification-with-variable-size-set-encoding) | Formalizes stop-and-predict versus collect-feature under cost, including medical examples. |
| Cost-Sensitive Feature Acquisition | [Pattern Recognition 2007](https://www.sciencedirect.com/science/article/pii/S0031320306004808) | Frames diagnosis as balancing test cost, misclassification cost, and stopping. |
| MediQ | [arXiv 2406.00922](https://arxiv.org/abs/2406.00922) | Shows that LLM question-asking under incomplete information is hard and needs abstention/confidence control. |
| ALFA | [arXiv 2502.14860](https://arxiv.org/abs/2502.14860) | Shows clinical question quality improves when decomposed into structured attributes like relevance, answerability, and diagnostic bias mitigation. |
| MEDDxAgent | [arXiv 2502.19175](https://arxiv.org/abs/2502.19175) | Closest LLM-agent DDx prior work on DDXPlus-style interactive diagnosis. |
| KG4Diagnosis | [arXiv 2412.16833](https://arxiv.org/abs/2412.16833) | Shows LLM diagnosis frameworks can be constrained and organized through medical knowledge graphs. |
| TriMediQ | [arXiv 2510.03536](https://arxiv.org/abs/2510.03536) | Converts multi-turn patient responses into triplet-based KGs for more reliable medical reasoning. |
| MedKGI | [arXiv 2512.24181](https://arxiv.org/abs/2512.24181) | Closest direct blueprint: KG-grounded iterative diagnosis, information gain, structured state. |

## 3. What MedKGI Already Does

MedKGI is the closest graph-based prior work to our intended algorithmic ledger.

Its problem statement:

- LLMs hallucinate when not grounded in verified medical knowledge.
- LLMs ask redundant or inefficient questions.
- LLMs lose coherence over multi-turn diagnostic dialogues.

Its solution:

```text
chief complaint / patient profile
  -> candidate differential diagnosis
  -> medical KG alignment
  -> diagnostic subgraph over diseases and symptoms
  -> information-gain symptom selection
  -> OSCE-style structured diagnostic record
  -> hypothesis-driven termination
```

Important mechanisms:

- candidate diseases are mapped into a medical KG
- a task-specific diagnostic subgraph is constructed from candidate diseases and connected symptoms
- candidate symptom questions are scored by expected reduction in diagnostic entropy
- the diagnostic record is maintained in structured JSON to prevent context overload
- termination uses a turn limit and stagnation detection

MedKGI's reported result pattern:

- it improves both diagnostic accuracy and average dialogue efficiency on its tested benchmarks
- removing the KG causes a large accuracy drop
- replacing information-gain symptom selection with random or degree-based selection hurts performance
- removing the clinical record harms coherence

What we should borrow:

- diagnostic subgraph construction
- information-gain question scoring
- structured state as a first-class object
- stagnation / drift detection
- ablation mindset: KG, record, and selection method should be separable

What we should not copy blindly:

- external KG alignment using PubMedBERT
- general OSCE record generation
- freeform symptom natural-language inquiry

Our DDXPlus environment already gives us a cleaner structured schema:

- 223 root evidence fields
- binary/categorical/multi-choice values
- exact hidden evidence reveal through environment
- trained BASD-style partial-evidence MLP

That means our graph can be more deterministic and benchmark-native than MedKGI's.

## 4. What DDXPlus And AARLC Tell Us

The DDXPlus paper already frames diagnosis as:

```text
initial evidence
  -> ask about symptoms / antecedents
  -> stop
  -> predict pathology and/or differential
```

The original DDXPlus baselines matter because:

- BASD is the MLP-style neural classifier family we used for the one-shot and partial-evidence models.
- AARLC separates symptom inquiry and diagnosis.
- AARLC uses entropy from the diagnosis classifier to align the inquiry policy and classification.

This gives direct support for our current hybrid v1 idea:

```text
LLM chooses evidence
partial-evidence MLP estimates diagnostic belief
MLP confidence / margin / entropy influence stopping
```

The graph-ledger phase should extend this:

```text
partial-evidence MLP belief
  + graph support/contradiction
  + unresolved differential pairs
  + expected information value
  -> better question shortlist and stop certificate
```

## 5. What Active Feature Acquisition Tells Us

Active feature acquisition treats diagnosis as:

```text
state = acquired feature subset
action = acquire one new feature or stop and predict
cost = feature/test cost
reward = final prediction quality minus acquisition cost
```

The NeurIPS 2018 work is especially relevant because it explicitly says that acquiring all features/tests can create cost burden and that irrelevant features can add noise or make prediction unstable. This matches our Notebook `15` finding:

- more evidence often helps
- but later evidence sometimes causes correct-to-wrong drift
- wrong cases used more evidence than correct cases

What we should borrow:

- stop-and-predict versus acquire-feature framing
- action value as prediction improvement minus evidence cost
- feature subset representation
- explicit accounting for redundant and noisy features

What we should not do yet:

- train a full RL policy
- jointly train the classifier and acquisition policy

Reason:

- we already have a strong MLP and LLM loop
- we need a controlled course-project improvement first
- deterministic graph algorithms are easier to debug and defend

## 6. What MediQ And ALFA Tell Us

MediQ shows that LLMs do not automatically become good information seekers when given the option to ask questions. It found that direct prompting for questions can degrade performance, and that better abstention/confidence strategies improve interactive clinical reasoning.

ALFA goes one level deeper. It decomposes "good question" into structured attributes:

- clarity
- focus
- answerability
- medical accuracy
- diagnostic relevance
- avoiding differential-diagnosis bias

What we should borrow:

- question quality should be decomposed into explicit signals
- the system should not just ask "what seems useful?"
- the LLM should receive structured reasons for each candidate question

For DDXPlus, our equivalent attributes are:

- legal under parent/child gating
- answerable by the DDXPlus environment
- discriminative between current top diagnoses
- non-redundant with already revealed evidence
- able to resolve contradiction
- not merely globally common

This naturally maps to graph edge scores.

## 7. What TriMediQ Tells Us

TriMediQ argues that multi-turn medical reasoning degrades when clinical facts are spread across raw dialogue logs. It converts patient responses into clinically grounded triplets and builds a patient-specific knowledge graph.

Important idea:

```text
raw dialogue
  -> structured triplets
  -> patient-specific graph
  -> better multi-hop reasoning
```

Our DDXPlus setting is even cleaner:

```text
DDXPlus token reveal
  -> structured evidence outcome
  -> patient-specific graph
  -> support / contradiction / discriminator reasoning
```

We do not need an LLM triplet extractor because DDXPlus already gives structured evidence IDs and values. That is a strength of our project.

## 8. What KG4Diagnosis Tells Us

KG4Diagnosis uses medical knowledge graphs to constrain and coordinate LLM-based diagnosis. It is less directly tied to interactive evidence acquisition than MedKGI, but it supports the broader pattern:

- KGs can constrain hallucination
- KGs can organize multi-agent diagnosis
- KGs can make reasoning more inspectable
- graph construction and validation are central challenges

For us, the graph construction problem is simpler because:

- DDXPlus has official evidence metadata
- train split statistics can provide disease-evidence associations
- patient episode graphs can be constructed deterministically

## 9. Design Lessons Across Prior Work

### Lesson 1. The Graph Should Be Patient-Specific And Dynamic

Static disease-symptom knowledge is not enough. The graph should update after each revealed evidence field.

### Lesson 2. The Graph Should Represent Negative Evidence

AARLC explicitly notes negative symptoms are useful because they rule out diseases. DDXPlus absent evidence is meaningful because a request can reveal absence.

Therefore graph edges must handle:

- present evidence
- absent evidence
- categorical value evidence
- multi-choice value evidence

### Lesson 3. Question Selection Should Optimize Discrimination, Not Popularity

MedKGI's ablations show random and degree/frequency-based selection are weaker than information-gain selection.

Our graph shortlist should avoid:

- high-frequency generic questions
- repeated broad systemic questions
- fields that support many diagnoses equally

### Lesson 4. Stopping Needs A Certificate

MediQ and AARLC both support the idea that stop behavior is a key bottleneck.

Notebook `15` shows threshold tuning is not enough.

The graph ledger should stop only when:

- MLP confidence is adequate
- top diagnosis has graph support
- contradiction is low
- unresolved top competitors are separated
- no high-value legal question remains

### Lesson 5. Graph Signals Should Be Auditable

The final project should be able to show:

- why a field was requested
- which diseases it separates
- whether it supported or contradicted the current top diagnosis
- whether the system stopped despite unresolved alternatives

This is the scientific value of the ledger.

## 10. What Remains Distinctive For Our Project

After this research, the clearest novelty is not "graph plus LLM."

The defensible contribution is:

> a DDXPlus-native graph evidence ledger over official root evidence slots, driven by train-derived disease/evidence statistics and online partial-evidence MLP belief, used to constrain an LLM evidence-acquisition policy and evaluate evidence efficiency under matched neural baselines.

Specific distinguishing features:

- official DDXPlus root-evidence action space
- exact legal evidence reveal rather than simulated natural-language patient answers
- patient graph built from observed evidence only
- partial-evidence MLP belief used as graph prior
- matched-evidence MLP comparison already exists
- graph stop certificate can be tested offline before live API runs

This is a strong course-project direction.

