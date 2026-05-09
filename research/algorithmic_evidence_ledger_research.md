# Algorithmic Evidence Ledger Research Deep Dive

Purpose: define what an algorithmic evidence ledger should do for our DDXPlus diagnostic workup project, grounded in prior work and in the failure analysis from our current notebooks.

This is not an implementation note yet. It is the research/design basis for the next notebook phase.

## 1. Executive Position

The next meaningful project improvement should not be another prompt tweak or another lambda sweep. Notebook `15` showed that the current Notebook `13` hybrid method is already near the local optimum for its simple MLP stop threshold:

- final 49-case result: `43/49 = 0.878`
- mean requests: `6.59`
- no threshold sweep variant improves the correct count
- incorrect cases generally used **more** evidence, not less
- several failures are stable wrong trajectories or diagnostic drift, not simple under-questioning

Therefore the algorithmic ledger should target:

- contradiction detection
- diagnostic drift detection
- unresolved differential separation
- evidence-value scoring
- stop certification
- case-level traceability

The ledger should not merely store evidence. It should compute deterministic, diagnosis-specific control signals from the current partial-evidence state.

Recommended name:

> **Evidence-Gated Differential Ledger**

Short form:

> **EGDL**

Core idea:

```text
Visible DDXPlus evidence
  -> deterministic ledger update
  -> partial-evidence MLP belief
  -> train-derived evidence/pathology statistics
  -> support, contradiction, drift, unresolved-pair, and value-of-information signals
  -> constrained LLM question choice and stop certificate
```

## 2. Relevant Prior Work And Lessons

### 2.1 DDXPlus

Source: [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://papers.nips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)

DDXPlus already defines the benchmark as an evidence-acquisition problem:

- each patient starts with age, sex, and initial evidence
- the model iteratively asks about symptoms or antecedents
- the interaction stops when relevant evidence is collected or max turns are reached
- final output is a differential diagnosis

Important DDXPlus implementation lessons:

- BASD uses an MLP with hidden layers of size `2048`
- AARLC aligns evidence acquisition and classification using classifier entropy
- the official evaluation treats interaction length, evidence collection, pathology accuracy, and differential quality as separate outcomes
- DDXPlus uses `T = 30` as a maximum-turn setting in the paper, but efficiency is explicitly measured

Implication for us:

- Our one-shot and partial-evidence MLPs are faithful to the dataset's neural baseline family.
- Our algorithmic ledger should explicitly track interaction length and evidence value, not only final accuracy.
- Entropy/margin from the classifier are legitimate control signals because DDXPlus prior work already uses classifier uncertainty to align diagnosis and inquiry.

### 2.2 Efficient Symptom Inquiring / AARLC

Source: [Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification](https://arxiv.org/abs/2112.00733)

This work separates symptom inquiry from diagnosis:

- symptom/evidence inquiry is sequential decision-making
- final disease prediction is classification
- distribution entropy is used to align the inquiry policy and classifier
- the goal is higher accuracy with fewer inquiry turns

Implication for us:

- The hybrid pattern is not arbitrary: evidence acquisition and classification can be separated.
- Our MLP can serve as a classifier-side diagnostic belief estimate.
- We are not doing RL yet, but we can borrow the idea of using entropy and margin as deterministic stop/request signals.

### 2.3 Active Feature Acquisition

Source: [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://papers.nips.cc/paper/2018/file/e5841df2166dd424a57127423d276bbe-Paper.pdf)

Active feature acquisition frames the problem as:

```text
observe partial features
choose: acquire another feature or stop and predict
pay acquisition cost
receive final prediction reward/penalty
```

The paper explicitly motivates medical diagnosis:

- clinicians start with a few symptoms
- they ask about symptoms or order tests
- excessive tests add financial burden and can delay care
- irrelevant features can add noise and make predictions unstable

Implication for us:

- DDXPlus evidence requests are active feature acquisition.
- The algorithmic ledger can approximate value-of-information without training a full RL policy.
- The ledger should penalize low-yield/generic/redundant evidence and flag cases where more evidence destabilizes the diagnosis.

### 2.4 Cost-Sensitive Feature Acquisition

Source: [Cost-sensitive feature acquisition and classification](https://www.sciencedirect.com/science/article/pii/S0031320306004808)

This literature treats diagnosis as a balance between:

- feature/test acquisition cost
- misclassification cost
- reward for correct diagnosis
- decision to stop sensing and classify

Implication for us:

- The correct question is not "maximum accuracy no matter what" versus "minimum questions."
- The correct question is "what evidence budget is required to maintain diagnostic quality?"
- A good stop rule should consider both diagnostic confidence and remaining high-value evidence, not just raw request count.

### 2.5 POMDP / Belief-State Active Classification

Source: [Cost-Bounded Active Classification Using Partially Observable Markov Decision Processes](https://arxiv.org/abs/1810.00097)

This line of work models active classification under partial observability using:

- belief over candidate classes
- observation/action choices
- confidence targets
- cost bounds
- finite horizon

Implication for us:

- A formal POMDP would be a major architecture change and is out of scope right now.
- But the algorithmic ledger can borrow a lighter version of the same structure:
  - maintain belief over diseases
  - estimate how observations would change belief
  - stop when belief is sufficiently resolved

### 2.6 MediQ And LLM Question-Asking

Source: [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](https://arxiv.org/abs/2406.00922)

MediQ argues that static medical QA is insufficient and that LLMs need to ask follow-up questions under incomplete information. The important finding for us is that directly prompting LLMs to ask questions can degrade performance, and that confidence/abstention strategies are needed.

Implication for us:

- We should not trust the LLM alone to know when it has enough evidence.
- A deterministic ledger can compensate for LLM weakness in information-seeking.
- Structured state and stop certificates are defensible, not artificial.

### 2.7 ALFA

Source: [ALFA: Aligning LLMs to Ask Good Questions](https://arxiv.org/abs/2502.14860)

ALFA decomposes question quality into fine-grained attributes such as relevance and clarity, then aligns LLMs to ask better clinical questions.

Implication for us:

- Question quality can be improved by explicit criteria.
- We can encode criteria deterministically instead of training a preference model:
  - relevance to current top diagnoses
  - discriminative value
  - non-redundancy
  - legality
  - ability to resolve contradiction

### 2.8 MEDDxAgent

Source: [MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](https://arxiv.org/abs/2502.19175)

MEDDxAgent is a modular LLM-agent framework with:

- DDxDriver orchestrator
- history-taking simulator
- knowledge retrieval agent
- diagnosis strategy agent
- iterative diagnosis refinement

Its DDXPlus interactive GPT-4o results are:

| Questions | GTPA@1 |
|---:|---:|
| 5 | 0.74 |
| 10 | 0.78 |
| 15 | 0.86 |

Implication for us:

- We cannot claim broad novelty for "LLM interactive DDx on DDXPlus."
- Our defensible distinction is DDXPlus-native structured evidence access and MLP/ledger-based evidence efficiency.
- The algorithmic ledger should be our clearer system-level contribution.

### 2.9 TriMediQ

Source: [TriMediQ: A Triplet-Structured Approach for Interactive Medical Question Answering](https://arxiv.org/abs/2510.03536)

TriMediQ argues that flat dialogue logs are weak for interactive medical reasoning because facts are spread across turns without explicit links. It converts patient responses into structured triplets and uses a graph-style representation to improve reasoning.

Implication for us:

- Structure matters.
- The ledger should not be a flat list of text observations.
- Even without a full graph neural network, we should represent:
  - patient has symptom
  - patient lacks symptom
  - evidence supports diagnosis
  - evidence contradicts diagnosis
  - evidence separates diagnosis A from diagnosis B

## 3. What Our Current Results Say The Ledger Must Fix

Notebook `15` provides the strongest empirical guidance.

### 3.1 Failure Is Not Mostly Under-Requesting

Incorrect cases requested more evidence:

| Outcome | Mean requests | Median requests |
|---|---:|---:|
| Correct | 5.86 | 5.0 |
| Incorrect | 11.83 | 11.5 |

This means a better ledger should not simply ask more. It should ask better and detect wrong trajectories earlier.

### 3.2 Extra Evidence Can Help Or Hurt

Turn-level prediction transitions:

| Head | Wrong-to-correct | Correct-to-wrong |
|---|---:|---:|
| LLM / hybrid | 40 | 11 |
| MLP | 38 | 10 |

Extra evidence is usually beneficial, but not always. The ledger must identify when a new finding causes unstable drift rather than genuine resolution.

### 3.3 Persistent Wrong Belief Is The Main Bottleneck

Many turn states are stable wrong. This suggests the current system sometimes locks into a wrong hypothesis and keeps asking without resolving the right discriminator.

The ledger should therefore ask:

- What are the top competing diagnoses?
- What evidence separates them?
- Has that evidence already been checked?
- Did the latest reveal actually move belief in the expected direction?
- Is the top diagnosis contradicted by strong evidence?

## 4. What An Algorithmic Ledger Should Be

The ledger should be a deterministic diagnostic controller with these layers:

```text
Observation Layer
  stores visible evidence and request history

Belief Layer
  stores MLP and LLM differentials, entropy, margin, stability

Support/Contradiction Layer
  scores each revealed evidence against top diagnoses

Discriminator Layer
  identifies unresolved evidence fields separating top competitors

Action-Value Layer
  scores legal next evidence requests by expected diagnostic value

Stop-Certificate Layer
  determines whether stopping is justified

Audit Layer
  explains why the system asked, stopped, drifted, or disagreed
```

This is stronger than memory. It is a structured control algorithm.

## 5. Core Ledger Signals

### 5.1 Belief Uncertainty

From the partial-evidence MLP:

- `top1_confidence = p_1`
- `margin = p_1 - p_2`
- `entropy = -sum_i p_i log(p_i) / log(K)`
- `unresolved_mass = sum probabilities over top competitors close to top1`

Interpretation:

- high confidence and margin suggest stopping may be safe
- high entropy or unresolved mass suggest more evidence is needed

### 5.2 Diagnosis Stability

Track over the last `w` turns:

- whether top-1 diagnosis changed
- whether top-k set changed
- KL divergence between consecutive MLP distributions
- whether LLM and MLP agree

Useful flags:

- `stable`: top diagnosis and top-k are consistent
- `drift`: top diagnosis changes after weak evidence
- `destabilized`: entropy increases or margin collapses after a reveal

### 5.3 Evidence Support And Contradiction

Precompute from train:

```text
P(root status/value | pathology)
```

For revealed evidence `e` and diagnosis `d`, estimate:

```text
support_score(e, d) = log P(e | d)
```

For a top diagnosis `d1` and competitor `d2`:

```text
pairwise_log_ratio(e, d1, d2) = log((P(e | d1) + eps) / (P(e | d2) + eps))
```

Interpretation:

- positive ratio supports `d1` over `d2`
- negative ratio contradicts `d1` relative to `d2`
- near-zero ratio is non-discriminative

This is not a full Bayesian diagnosis engine. It is an auditable evidence-accounting layer.

### 5.4 Pairwise Differential Separation

For current top diagnoses `d1, d2, ..., dk`, identify legal unrevealed roots that best separate them.

Candidate root score components:

- gap in expected present/value rate between top diagnoses
- whether it separates current LLM top1 from MLP top1
- whether it resolves a contradiction
- whether it is generic or redundant
- whether parent gating makes it legal

This targets a specific failure mode from Notebook `15`: the system sometimes asks many questions but does not resolve the right diagnostic pair.

### 5.5 Expected Information Value

Approximate value of asking root `r`:

```text
EIV(r) =
  expected_entropy_reduction(r)
  + expected_margin_gain(r)
  + unresolved_pair_resolution(r)
  + contradiction_resolution_bonus(r)
  - generic_penalty(r)
  - redundancy_penalty(r)
```

The expected entropy reduction can be estimated by counterfactual MLP scoring:

```text
current_H = H(MLP(current_state))
expected_H_after_r = sum_o P(o | current_belief) * H(MLP(current_state + r=o))
entropy_gain = max(0, current_H - expected_H_after_r)
```

This is computationally heavier than current v1, but feasible because DDXPlus has only 223 roots and the shortlist can be restricted.

### 5.6 Stop Certificate

The system should stop only if the ledger can produce a certificate:

```text
confidence_ok
margin_ok
entropy_ok
stability_ok
no_major_llm_mlp_disagreement
no_active_contradiction
no_high_value_unresolved_discriminator
```

This improves over Notebook `13` because Notebook `13` mostly uses MLP confidence/margin/entropy. The algorithmic ledger adds:

- contradiction status
- unresolved-pair status
- remaining evidence value
- drift status

## 6. Design Choice: Deterministic First, Not RL First

We should not jump directly to RL, graph neural networks, or probabilistic belief-state modeling.

Reason:

- the project already has many moving parts
- the current failure modes can be attacked with deterministic analysis
- deterministic ledger signals are easier to debug and present
- the course-project bar favors a clear method over a fragile overbuilt method

The algorithmic ledger can still be research-grounded because it borrows from:

- active feature acquisition
- cost-sensitive classification
- entropy-aligned inquiry/classification
- belief-state active classification
- structured interactive medical QA

The difference is that we implement the first version as a deterministic controller rather than training a new policy.

## 7. What The Ledger Should Not Do Yet

Out of scope for the next phase:

- full RL policy training
- graph neural network reasoning
- probabilistic graphical model inference
- multi-agent debate
- external PubMed retrieval
- replacing the partial-evidence MLP
- using hidden labels or hidden test evidence

These are future extensions. The next version should stay focused:

> better deterministic control over the existing single-agent hybrid system.

## 8. Proposed Research Claim After Implementation

If the algorithmic ledger works, the claim becomes:

> A structured evidence-gated ledger can improve an LLM-led DDXPlus diagnostic workup by detecting unresolved competing diagnoses, contradiction, and diagnostic drift, allowing the system to preserve high accuracy while requesting fewer or more clinically targeted evidence fields.

If it does not improve accuracy, the fallback claim is still useful:

> The ledger provides a rigorous diagnostic audit layer showing that remaining failures arise from persistent wrong-belief trajectories and insufficient discriminator selection, not from stop-threshold tuning alone.

Both outcomes are defensible.

## 9. Recommended Starting Point

Start offline.

Do not spend API money first.

Build Notebook `16` as an offline algorithmic ledger analysis over existing Notebook `13` 49-case traces:

1. Reconstruct every turn's partial-evidence state.
2. Compute MLP belief, entropy, margin, stability.
3. Precompute train-derived evidence/pathology rates.
4. Score each revealed evidence as support or contradiction for top diagnoses.
5. Identify unresolved top diagnosis pairs at each turn.
6. Estimate high-value legal next actions at each turn.
7. Check whether the ledger would have flagged the 6 Notebook `13` errors before final stop.

Only after that should we make a live Notebook `17`.

Offline success criteria:

- the ledger flags most incorrect final cases before stop
- it does not flag too many correct cases as unsafe
- high-value action suggestions are interpretable
- hard cases show actionable warnings such as contradiction, drift, or unresolved pair

Live success criteria:

- top-1 accuracy at least matches Notebook `13` on 49 cases
- mean requests stay near or below `7.5`
- hard-case behavior improves or becomes more diagnosable
- fewer generic requests
- clearer trace explanations

