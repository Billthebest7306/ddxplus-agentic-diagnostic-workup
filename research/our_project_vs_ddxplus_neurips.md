# Our Project vs. The DDXPlus NeurIPS Paper

Purpose: make the differences between our project and the original DDXPlus NeurIPS paper explicit, so we can decide what is genuinely novel, what is only a reimplementation, and how to frame claims safely.

Reference prior work:

- Paper: [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://papers.nips.cc/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)
- Official repo: [mila-iqia/ddxplus](https://github.com/mila-iqia/ddxplus)

## 1. Short Answer

The DDXPlus paper already did:

- automatic diagnosis on DDXPlus
- iterative evidence collection
- age/sex/initial-evidence start state
- root-evidence question action space
- trained sequential agents
- diagnosis and differential prediction
- interaction length, evidence recall, and differential diagnosis metrics

Our project is different only if we focus on:

- LLM-based workup control rather than a trained RL/supervised policy
- explicit deterministic evidence ledger/state manager
- decoded, auditable evidence traces
- cost-sensitive stopping and evidence-efficiency analysis
- matched-evidence decomposition of evidence acquisition vs final reasoning
- possible hybrid architecture: LLM evidence controller plus neural diagnostic head

Unsafe claim:

> We are the first to build an iterative diagnostic agent on DDXPlus.

Safer claim:

> We study whether a ledger-controlled LLM workup policy can serve as an interpretable, cost-sensitive evidence acquisition controller on DDXPlus, and whether final diagnosis is best handled by the LLM, a neural classifier, or a hybrid of both.

## 2. Side-By-Side Comparison

| Dimension | DDXPlus NeurIPS Paper | Our Current Project | Meaningful Difference? |
|---|---|---|---|
| Dataset | Introduces DDXPlus | Uses DDXPlus | No novelty here |
| Patient start state | Age, sex, initial evidence | Age, sex, initial evidence | Same protocol |
| Evidence space | 223 root evidences, with binary/categorical/multi-choice fields | Same 223-root evidence universe, decoded into readable questions/values | Mostly same, but our decoded ledger improves interpretability |
| Sequential interaction | Yes, models iteratively ask symptoms/antecedents | Yes, LLM sequentially requests evidence fields | Same broad task |
| Max interaction limit | `T = 30` in paper experiments | request caps like `24`, plus lambda-based stopping | Difference: we study cost-sensitive stopping |
| Baseline models | AARLC and BASD | One-shot MLPs, LLM sequential policies, partial-evidence classifier | Different model family |
| Policy learning | AARLC is trained RL; BASD is supervised | LLM policy is not trained on DDXPlus as a policy | Difference: zero/few-shot or prompted LLM controller |
| State management | Environment/protocol exists, but paper emphasizes trained agents and metrics | Explicit deterministic evidence ledger with visible/hidden/requested/legal/revealed state | Meaningful if framed as interpretability/control |
| Evidence presentation | Coded evidence structure, paper examples | Decoded clinical question/value strings in prompts and traces | Meaningful for LLM usability and trace review |
| Final prediction | Pathology or differential, depending on training target | LLM final answer, one-shot neural classifier, matched partial classifier, full-evidence ceiling | Difference: final-head comparison and decomposition |
| Differential training | Central paper contribution: differential diagnosis as training signal | We use differential metrics and some soft targets, but current sequential LLM is not trained on differentials | Not a primary novelty yet |
| Evaluation metrics | IL, PER, DDR, DDP, DDF1, GTPA@1, GTPA | Accuracy, top-k, macro-F1, request count, cost, matched comparisons; should add official metrics | We need to align more with paper |
| Explainability | Doctor comments on example traces; differential alignment emphasized | Ledger traces, requested evidence history, JSON decisions, qualitative trace comparison | Potentially meaningful, but needs stronger reporting |
| Cost/efficiency | Interaction length reported | Lambda/cost-sensitive stopping, accuracy-vs-requests frontier | Meaningful difference |
| Severity | Paper notes severity exists and is future work | Not implemented yet | Opportunity for novelty |
| Multi-agent | Not the focus | Later planned, not implemented yet | Future difference, not current result |

## 3. What The Paper Already Covers

The original paper already establishes that DDXPlus supports:

- patient simulation with pathology, evidence, and differential diagnosis
- structured evidence acquisition
- automatic symptom detection and automatic diagnosis agents
- pathology prediction and differential prediction
- evaluating the tradeoff between interaction length and diagnostic/evidence quality

It also provides official trained baselines:

- **AARLC**: RL-based evidence acquisition plus classifier branch
- **BASD**: supervised automatic symptom detection/diagnosis model with MLP classifier

Because of this, our work cannot be sold as introducing the basic diagnostic workup setup.

## 4. What Our Project Adds So Far

### 4.1 LLM As Workup Controller

The paper's agents are trained DDXPlus-specific models.

Our sequential system uses an LLM as the controller. The LLM receives:

- demographics
- initial evidence
- current visible/revealed evidence
- shortlist of legal next evidence questions
- one-shot prior context, depending on notebook version
- ledger-derived state summary

The LLM decides whether to request another field or stop and diagnose.

Why this matters:

- it tests whether general medical/language reasoning can operate inside the DDXPlus environment without training an RL policy
- it makes the policy more inspectable through prompts and traces
- it supports future multi-agent roles more naturally than monolithic RL

Limitation:

- if it underperforms trained AARLC/BASD, the LLM framing alone is not enough
- LLM API cost makes full-scale evaluation harder

### 4.2 Deterministic Evidence Ledger

Our ledger is a core project component.

It tracks:

- visible evidence at episode start
- hidden evidence
- requestable root evidence fields
- which fields have been requested
- decoded revealed values
- absent vs present evidence
- repeated/invalid request prevention
- trace history
- stopping and final prediction metadata

Why this matters:

- the LLM does not directly access hidden patient data
- every reveal is controlled by the environment
- every request can be audited
- later multi-agent systems can share the same evidence source of truth

Difference from paper:

- the paper has an environment/protocol, but our contribution is the explicit ledger-centered architecture for LLM-controlled workup

Limitation:

- ledger control is an engineering and interpretability contribution, not automatically a performance contribution

### 4.3 Cost-Sensitive Stopping

The paper reports interaction length and uses a maximum turn budget.

Our notebook 08 introduces lambda-based cost-sensitive stopping:

- each evidence request has an implicit cost
- the policy estimates whether another question is worth asking
- larger lambda values encourage earlier stopping
- we plot accuracy vs mean requests and utility vs lambda

Why this matters:

- it turns the project from "ask up to a cap" into "study diagnostic efficiency"
- it produces an accuracy-efficiency frontier
- it gives a clinically intuitive question: how much evidence is enough?

Limitation:

- our marginal-value heuristic is still hand-designed
- small live runs do not establish final statistical claims

### 4.4 Matched-Evidence Decomposition

This is one of our most important differences.

We compare:

- initial-evidence one-shot
- sequential LLM final prediction
- matched-evidence one-shot using exactly the evidence the sequential policy acquired
- full-evidence one-shot ceiling

This lets us ask:

- did the policy acquire useful evidence?
- does the LLM reason better than a neural classifier given the same evidence?
- is final diagnosis better handled by LLM, classifier, or hybrid?

Why this matters:

- it prevents a misleading claim that all gains come from "agentic reasoning"
- it separates evidence acquisition from final diagnostic inference

Limitation:

- the matched classifier's strength depends on training design
- the comparator must be versioned and frozen to avoid moving-goalpost criticism

### 4.5 Hybrid Direction

Our current results suggest the strongest future architecture may be:

1. LLM or structured policy chooses evidence.
2. Ledger controls legal reveal and traceability.
3. Partial-evidence neural classifier produces final diagnosis.
4. LLM and classifier disagreements trigger more evidence or adjudication.

This differs from both:

- pure trained DDXPlus agents
- pure LLM diagnosis
- pure one-shot classifiers

Why this matters:

- it uses each component where it appears strongest
- LLM: flexible question selection and explanation
- classifier: stable direct diagnostic inference from structured state
- ledger: control, reproducibility, and safety

## 5. Where Their Work Is Stronger Than Ours

The DDXPlus paper is stronger in several ways:

- full official benchmark protocol
- full-scale training and evaluation
- published baselines with confidence intervals
- official DDXPlus metrics
- differential diagnosis training analysis
- direct comparison between pathology-trained and differential-trained agents
- clearer external reproducibility target

Our current work is weaker in:

- live LLM runs are small-sample because of API cost
- official DDXPlus metrics are not yet fully integrated into the newest sequential reports
- we have not reproduced AARLC/BASD locally
- the multi-agent system is not implemented yet
- the matched-evidence comparator changed over time, which complicates interpretation

## 6. Where Our Work Can Be Stronger Or More Unique

### 6.1 Interpretability And Auditability

The ledger gives us a concrete explainability artifact:

- what the model knew
- what it asked
- what was revealed
- why it stopped
- what diagnosis it gave

We should make this a primary project theme.

### 6.2 Efficiency Frontier

Instead of one interaction length number, we can show:

- accuracy vs lambda
- mean requests vs lambda
- utility vs lambda
- fraction of full-evidence gain recovered
- severe-disease miss rate vs request count, if added

This can become more interesting than raw top-1 accuracy.

### 6.3 Human-Readable Clinical Workup

Because LLM prompts use decoded questions and values, we can evaluate:

- whether requested questions make clinical sense
- whether the policy asks generic vs discriminative questions
- whether it prematurely stops
- whether it asks safety-critical rule-out questions

This is not the same as simply optimizing an RL reward.

### 6.4 Severity-Aware Diagnosis

The paper explicitly leaves severity-aware handling as future work.

This is a strong opportunity:

- weight errors by severity
- measure severe-condition recall
- penalize missing severe diagnoses more heavily
- test whether the agent asks questions that rule out dangerous alternatives

This could be a genuinely distinct angle if implemented carefully.

### 6.5 Multi-Agent Extension

Multi-agent work is not implemented yet, but a future version could be distinct if agents have meaningful roles:

- hypothesis generator
- evidence acquisition planner
- severity/safety reviewer
- final adjudicator

However, multi-agent debate alone is weak. The novelty must come from role-specific use of the ledger and controlled evidence acquisition.

## 7. Claims We Can Safely Make

Safe claim:

> We build on the DDXPlus automatic diagnosis benchmark by studying a ledger-controlled LLM workup policy rather than a fully trained DDXPlus-specific acquisition policy.

Safe claim:

> Our experiments decompose performance into evidence acquisition and final diagnosis by comparing sequential LLM outputs against matched-evidence neural classifiers.

Safe claim:

> Cost-sensitive stopping provides an interpretable accuracy-efficiency tradeoff for sequential diagnostic workup.

Safe claim:

> Current evidence suggests the strongest architecture may be a hybrid: LLM for evidence acquisition, neural classifier for final diagnosis, and ledger for controlled state management.

Safe claim:

> Our current results are promising but not conclusive because LLM runs are still small-sample and need comparison against official DDXPlus baselines.

## 8. Claims We Should Avoid

Avoid:

> We invented sequential diagnosis on DDXPlus.

Reason:

- DDXPlus was built for this and the paper already evaluates sequential AD/ASD agents.

Avoid:

> Agentic systems are better than one-shot systems.

Reason:

- "one-shot" can mean initial evidence, matched evidence, or full evidence
- different one-shot comparators answer different questions

Avoid:

> The LLM reasons better than neural classifiers.

Reason:

- the partial-evidence classifier is competitive on current small slices
- model capacity and training distribution matter

Avoid:

> Our 24-case API run proves superiority.

Reason:

- it is a useful pilot, not a full benchmark

Avoid:

> Multi-agent architecture will automatically improve performance.

Reason:

- without controlled roles and evidence access, multi-agent systems may just add cost and variance

## 9. Recommended Research Framing

Current best framing:

> This project studies whether a ledger-controlled LLM diagnostic workup policy can efficiently acquire informative evidence on DDXPlus, and whether final diagnosis should be made by the LLM, a trained neural head, or a hybrid adjudication scheme.

Benchmark framing:

> We compare against the DDXPlus paper's published AARLC/BASD baselines as external anchors, and use initial-evidence, matched-evidence, and full-evidence one-shot models as internal controls to decompose performance.

Novelty framing:

> The novelty is not the existence of a sequential diagnostic agent. The novelty is the combination of LLM-based evidence control, deterministic ledger state, cost-sensitive stopping, and matched-information analysis.

## 10. Practical Next Steps

To make the distinction stronger, we should:

1. Add official DDXPlus metrics to our sequential notebooks.
2. Add a table comparing our results to Table 3 from the DDXPlus paper.
3. Implement a simple non-LLM evidence acquisition baseline.
4. Freeze the current partial-evidence matched classifier as `matched_v2`.
5. Run a 49-case balanced experiment with fixed `gpt-4.1-mini` and lambdas `0.10`, `0.22`, `0.35`.
6. Add severity-aware metrics using `release_conditions.json`.
7. Only then move to multi-agent design, with clear role separation and ledger-governed evidence access.

## 11. Bottom Line

The DDXPlus paper is broader and stronger than we initially treated it. It already covers the core idea of interactive automatic diagnosis.

Our project can still be meaningful, but only if we stop claiming novelty at the level of "agent asks questions and diagnoses." The defensible difference is:

> Our system treats the LLM as a transparent evidence-acquisition controller governed by a deterministic ledger, studies cost-sensitive stopping, and compares final diagnostic heads under matched evidence.

That is the angle to build around.

