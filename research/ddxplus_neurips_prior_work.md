# DDXPlus NeurIPS Prior Work Review

Purpose: summarize the original DDXPlus NeurIPS paper and identify what has already been done, what baselines are externally grounded, and where our project can still bring a distinct contribution.

Primary sources:

- Paper: [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://papers.nips.cc/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)
- NeurIPS page: [DDXPlus abstract page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cae73a974390c0edd95ae7aeae09139c-Abstract-Datasets_and_Benchmarks.html)
- Official repo: [mila-iqia/ddxplus](https://github.com/mila-iqia/ddxplus)

## 1. What The DDXPlus Paper Already Does

The DDXPlus paper is not just a dataset paper. It also studies automatic diagnosis systems that interact with patients, collect evidence, and output diagnoses or differential diagnoses.

This is important for our project because a generic claim like "we built a sequential diagnostic agent for DDXPlus" is not novel. The paper already frames DDXPlus around Automatic Symptom Detection (ASD) and Automatic Diagnosis (AD) systems that start from limited initial evidence, ask follow-up evidence questions, and produce a diagnosis or differential.

The paper's stated motivation:

- existing automatic diagnosis datasets often lack differential diagnoses
- clinicians reason using differential diagnoses, not only a single final disease label
- DDXPlus adds structured symptoms, antecedents, categorical evidence, multi-choice evidence, evidence hierarchy, ground-truth pathology, and a generated differential diagnosis

High-level task setup:

- patient begins with age, sex, and an initial evidence item
- model asks about symptoms or antecedents
- model stops when enough relevant evidence is collected or when a maximum turn limit is reached
- model predicts a differential diagnosis at the end

This overlaps directly with our broad project direction.

## 2. Dataset Construction And Schema

The dataset contains roughly 1.3 million synthetic patients.

The paper says the dataset is generated from:

- a proprietary medical knowledge base
- synthesized patient demographics and evidence
- a rule-based automatic diagnosis platform that generates differential diagnoses

Official split:

- 80% train
- 10% validation
- 10% test
- stratified by simulated pathology

Patient fields in the official release:

- `AGE`: synthesized patient age
- `SEX`: synthesized patient sex
- `PATHOLOGY`: ground-truth simulated pathology
- `EVIDENCES`: symptoms and antecedents experienced by the patient
- `INITIAL_EVIDENCE`: one initial evidence item selected from the patient's evidence list
- `DIFFERENTIAL_DIAGNOSIS`: list of pathology-probability pairs

Evidence format:

- binary evidence: `E_48`
- categorical evidence: `E_54_@_V_161`
- multi-choice evidence: multiple `evidence_@_value` entries can appear for the same root evidence

Evidence metadata:

- `release_evidences.json` gives question text, evidence type, default value, possible values, and decoded value meanings
- `release_conditions.json` gives pathology names, ICD-10 IDs, severity, and associated symptoms/antecedents

Key schema numbers:

| Evidence Type | Count |
|---|---:|
| Binary evidences | 208 |
| Categorical evidences | 10 |
| Multi-choice evidences | 5 |
| Total root evidences | 223 |
| Symptoms | 110 |
| Antecedents | 113 |

Patient evidence statistics:

| Field | Mean | Std | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| All evidences | 13.56 | 5.06 | 1 | 10 | 13 | 17 | 36 |
| Symptoms | 10.07 | 4.69 | 1 | 8 | 10 | 12 | 25 |
| Antecedents | 3.49 | 2.23 | 0 | 2 | 3 | 5 | 12 |

Interpretation for us:

- DDXPlus is already designed for evidence-gathering systems.
- There is enough structured information to define legal actions and evidence state cleanly.
- A good project should use the official evidence structure, not flatten everything into plain text without tracking state.

## 3. Relevant Figures And What They Mean

### Figure 1: Dataset Generation Pipeline

Figure 1 explains how synthetic patients and differential diagnoses are produced.

Practical meaning:

- DDXPlus is not real clinical EHR data.
- It is synthetic data generated using a knowledge base and an automatic diagnosis platform.
- This makes the benchmark controlled and useful for algorithm development, but clinical claims must be cautious.

How it affects our work:

- We should frame results as evidence-acquisition behavior on a synthetic diagnostic benchmark.
- We should avoid implying direct clinical deployment value.

### Figure 2: Pathology Distribution

Figure 2 shows the pathology histogram across 49 pathologies.

The paper notes that URTI, Viral pharyngitis, and Anemia dominate, but other pathologies remain represented.

Practical meaning:

- Accuracy can be influenced by class imbalance.
- Macro-F1 and pathology-balanced slices matter.
- Our 1-per-class experiments are useful for probing behavior but are not official full-test estimates.

How it affects our work:

- We should report both full split metrics and balanced-slice metrics.
- For API-based LLM experiments, balanced slices are defensible, but we must label them as small-sample pilots.

### Figure 3: Demographic Distribution

Figure 3 shows sex and age distributions.

The paper states these are aligned with 2015 New York census statistics used during generation.

Practical meaning:

- Demographics are not incidental; they are part of the generative process.
- Age and sex should remain part of the model state.

How it affects our work:

- Our one-shot and sequential systems correctly include age/sex.
- We should keep age-bin and sex encoding aligned with DDXPlus/BASD conventions where possible.

### Figure 4: Differential Diagnosis Size And True-Pathology Rank

Figure 4 shows:

- differential diagnosis can contain multiple diseases
- the simulated true pathology is ranked first for more than 70% of patients

Practical meaning:

- Top-1 pathology accuracy is not the only meaningful metric.
- A model trained to predict the differential may reasonably not place the simulated pathology first every time.

How it affects our work:

- We should report differential metrics, top-k accuracy, and ground-truth-in-differential metrics.
- If we only optimize top-1 accuracy, we may miss the clinical purpose of differential diagnosis.

### Table 1: Evidence Type Distribution

Table 1 establishes the official root evidence universe:

- 223 root evidences
- mostly binary, but with categorical and multi-choice fields

Practical meaning:

- A faithful baseline must respect binary/categorical/multi-choice encoding.
- Our slot-based BASD-style encoding is aligned with this.

### Table 2: Number Of Evidences Per Patient

Table 2 shows that patients average around 13.56 total evidence items.

Practical meaning:

- The useful evidence set is much smaller than the full 223-action universe.
- Evidence acquisition efficiency is a real research dimension.

How it affects our work:

- Cost-sensitive stopping is relevant.
- Asking 20 to 30 questions may recover many evidence fields but may be inefficient.

### Table 3: Official Baseline Results

Table 3 is the most important external benchmark for us.

The paper reports four trained agent variants:

| Method | Differential Training? | IL | GTPA@1 | GTPA | PER | DDR | DDP | DDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AARLC | Yes | 25.75 | 75.39 | 99.92 | 54.55 | 97.73 | 69.53 | 78.24 |
| AARLC | No | 6.73 | 99.21 | 99.97 | 32.78 | 21.96 | 99.19 | 31.28 |
| BASD | Yes | 17.86 | 67.71 | 99.30 | 88.18 | 85.03 | 88.34 | 83.69 |
| BASD | No | 17.99 | 97.15 | 98.82 | 88.45 | 21.89 | 99.38 | 31.31 |

Metric meanings:

- `IL`: interaction length
- `GTPA@1`: ground-truth pathology accuracy at rank 1
- `GTPA`: whether ground-truth pathology appears in the predicted differential
- `PER`: positive evidence recall
- `DDR`: differential diagnosis recall
- `DDP`: differential diagnosis precision
- `DDF1`: differential diagnosis F1

Important interpretation:

- Pathology-trained variants get much higher top-1 ground-truth accuracy.
- Differential-trained variants get much stronger differential recall/F1.
- This means "best" depends on whether the objective is single-label diagnosis or clinically useful differential diagnosis.

Implication for us:

- If we only compare top-1 accuracy, we are using a narrower target than the DDXPlus paper.
- A serious claim should include differential quality and evidence collection quality, not just final pathology accuracy.

## 4. Official Baselines: What They Are

### AARLC

AARLC is an RL-based automatic diagnosis method.

The paper describes it as having:

- an evidence acquisition branch
- a classifier branch
- adaptive alignment between acquisition and classification using classifier entropy
- hyperparameters tuned on validation data

Why it matters:

- This is the closest official baseline to an evidence-acquisition agent.
- If our project claims better evidence acquisition, AARLC is the natural external comparator.

### BASD

BASD is a supervised method adapted from automatic symptom detection.

The paper describes it as:

- a supervised model
- built from the ASD module of prior work, except without the original knowledge graph
- using a classifier network to predict disease at the end
- using an MLP with hidden layers of size 2048

Why it matters:

- Our MLP one-shot work is BASD-inspired, but our direct one-shot setting is not the same as the official BASD interactive baseline.
- The official BASD result is stronger as a published benchmark than our evolving internal one-shot baselines.

## 5. What Is Already Not Novel For Us

The following claims are already covered by DDXPlus prior work and should not be presented as our novelty:

- "We use DDXPlus for automatic diagnosis."
- "We start with age, sex, and initial evidence."
- "The agent asks symptoms or antecedents iteratively."
- "The system predicts a pathology or differential diagnosis."
- "We use a 223-root-evidence action space."
- "We use an MLP/BASD-style classifier."
- "Differential diagnosis can be used as a training signal."
- "Evidence collection and interaction length matter."

If our project is framed only as one of these, it will look like a weaker reimplementation of the DDXPlus paper.

## 6. Where Our Project Can Still Be Distinct

The strongest possible novelty is not "sequential diagnosis exists." The original paper already has that.

Our distinct angle should be:

**LLM-based, ledger-controlled, cost-sensitive diagnostic workup with explicit decomposition of evidence acquisition versus final diagnosis.**

Concrete novelty candidates:

### 1. LLM As A Zero/Few-Shot Evidence Workup Controller

The original paper trains AARLC/BASD on DDXPlus.

Our system can ask:

- can a modern LLM act as an evidence-acquisition controller without training a policy from scratch?
- how close can it get to trained DDXPlus agents?
- what kinds of questions does it ask compared with learned policies?

This is a valid modern angle if we benchmark it against AARLC/BASD.

### 2. Deterministic Evidence Ledger

Our ledger tracks:

- visible evidence
- hidden evidence
- requested evidence history
- revealed values
- legal/gated actions
- repeated or invalid request prevention
- stop behavior and traceability

The DDXPlus paper has an environment/protocol, but our emphasis is on auditable LLM-state control.

Possible claim:

> A ledger-controlled LLM workup system gives more transparent and reproducible evidence-gathering behavior than free-form LLM prompting.

### 3. Cost-Sensitive Stopping

The original paper uses a maximum turn budget of `T = 30`.

Our notebook 08 studies lambda-based evidence cost:

- not just "can the model diagnose?"
- but "how much evidence does it need?"
- and "where does accuracy collapse when evidence cost is too high?"

Possible claim:

> Cost-sensitive stopping exposes an accuracy-efficiency frontier for LLM workup policies.

### 4. Matched-Evidence Decomposition

Our notebook 09/10 work separates:

- value from acquiring better evidence
- value from reasoning over that evidence
- value from the final diagnostic head

This is important because it prevents us from falsely attributing all gains to "agentic reasoning."

Possible claim:

> Much of the improvement may come from selecting informative evidence; final diagnosis can be made by an LLM, a neural classifier, or a hybrid head.

### 5. Hybrid Architecture

The most defensible next architecture:

- LLM sequential policy chooses what to ask
- deterministic ledger controls what is legally revealed
- partial-evidence neural classifier produces calibrated final prediction
- LLM and classifier disagreement triggers adjudication or additional evidence requests

This is a cleaner novelty than simply "multi-agent debate."

### 6. Severity-Aware Workup

The DDXPlus paper explicitly notes that severity exists and could be used for future systems.

This gives us a strong future angle:

- do not treat all pathologies equally
- prioritize ruling out severe conditions
- measure severe-miss rate, severe-recall, or safety-weighted utility

This may be more clinically meaningful than raw accuracy.

## 7. How We Should Reframe Our Project

Weak framing:

> We propose an agentic diagnostic system for DDXPlus.

Problem:

- the DDXPlus paper already evaluates automatic diagnosis agents.

Better framing:

> We evaluate whether a structured LLM-based diagnostic workup controller, governed by an explicit evidence ledger and cost-sensitive stopping rule, can approach published DDXPlus automatic diagnosis baselines while acquiring fewer or more targeted evidence fields.

Even better framing:

> We decompose diagnostic workup performance into evidence acquisition, final diagnostic reasoning, and information-efficiency, using DDXPlus published baselines as the external reference point and matched-evidence classifiers as internal controls.

Best current research question:

> Can a ledger-controlled LLM workup policy efficiently select clinically informative DDXPlus evidence, and should final diagnosis be made by the LLM, a neural diagnostic head, or a hybrid of both?

## 8. What We Need To Add To Be Harder To Criticize

### A. Report Official DDXPlus Metrics

For our sequential runs, compute:

- `IL`: mean interaction length / mean requests
- `PER`: positive evidence recall
- `DDR`: differential recall
- `DDP`: differential precision
- `DDF1`: differential F1
- `GTPA@1`: top-1 ground-truth pathology accuracy
- `GTPA`: ground-truth pathology included in predicted differential

This aligns our results with Table 3.

### B. Compare Against Published Baselines

At minimum:

- cite the official Table 3 values
- compare our sampled results cautiously
- label small live API runs as pilots, not full benchmark results

Better:

- reproduce BASD or AARLC from official code if feasible
- run on the same sampled cases as our LLM policy
- compare evidence efficiency and accuracy directly

### C. Add A Simple Evidence-Acquisition Control

We need a non-LLM control that asks questions using simple heuristics:

- random legal evidence
- most frequent evidence
- evidence most associated with one-shot top-k pathologies

Why:

- if a simple heuristic gets similar evidence acquisition performance, the LLM policy is less interesting
- if the LLM beats these controls, the agentic policy claim is stronger

### D. Stop Changing The Comparator After Interpretation

The matched-evidence classifier should now be versioned and frozen.

If we improve it later, name it as a new comparator:

- `matched_v1_full_model_on_partial_state`
- `matched_v2_partial_policy_masked_classifier`
- `matched_v3_[future_method]`

Do not retroactively replace old interpretations without noting the comparator changed.

## 9. Immediate Action Items For Our Repo

Recommended next implementation steps:

1. Add official DDXPlus metric computation to notebook 08/09 outputs.
2. Add a report table that places our results beside DDXPlus Table 3, with clear caveats about sample size.
3. Add a simple heuristic evidence-acquisition baseline.
4. Run the 49-case balanced test for lambdas `0.10`, `0.22`, and `0.35`.
5. Compare:
   - DDXPlus published AARLC/BASD
   - our cost-sensitive LLM policy
   - partial-evidence diagnostic head
   - simple heuristic acquisition baseline
6. Explore severity-aware metrics as the next genuinely clinical angle.

## 10. Bottom Line

The original DDXPlus paper already did sequential automatic diagnosis with trained agents.

Our novelty cannot be:

> "We built a sequential diagnosis agent."

Our defensible novelty can be:

> "We built a transparent LLM-based workup controller with deterministic evidence-ledger state, cost-sensitive stopping, and matched-information analysis to determine whether value comes from evidence acquisition, final reasoning, or their hybrid combination."

The project is still viable, but only if we benchmark against the DDXPlus paper directly and stop treating our evolving one-shot models as the main external frame of reference.

