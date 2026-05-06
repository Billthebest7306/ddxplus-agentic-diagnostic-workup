# Novelty Assessment For Our DDXPlus Diagnostic Workup Project

Last updated: 2026-05-05

Purpose: evaluate the novelty of our project holistically, not by dismissing individual components as "already done." The goal is to understand whether the specific system we are building has been done before in this application, what parts are genuinely distinctive, and how to frame our contribution without overclaiming.

## 1. Executive Answer

The broad research area is not new:

- DDXPlus already supports automatic diagnostic workup.
- The DDXPlus paper already evaluates trained sequential agents.
- Active feature acquisition already studies "which missing feature/test/question should I acquire before classification?"
- Medical LLM-agent papers already study interactive diagnosis and follow-up questioning.
- Modular LLM differential-diagnosis systems already exist, including work that uses DDxPlus.

However, I did **not** find evidence that our exact planned system has already been implemented in this exact form:

> DDXPlus structured evidence environment + deterministic legal evidence ledger + LLM-controlled evidence acquisition over decoded DDXPlus root fields + BASD-style partial-evidence MLP diagnostic head + matched-evidence decomposition + cost-sensitive stopping + planned online LLM-MLP feedback + planned architecture-agnostic algorithmic ledger signals.

The strongest defensible novelty is therefore not "we invented diagnostic agents" or "we invented evidence acquisition." The defensible novelty is:

> We apply and combine existing ideas in a specific way for DDXPlus: an auditable ledger-gated LLM workup controller, evaluated against neural diagnostic heads under matched evidence, with cost-sensitive evidence-efficiency analysis and a path toward online LLM-MLP feedback.

This is a valid applied-research contribution if we frame it as a **specific integration and evaluation design**, not as an entirely new paradigm.

## 2. What Exactly Is "Our Idea"?

The project should be defined as the full system, not as one component.

Current implemented pieces:

- DDXPlus patient episodes.
- Deterministic evidence ledger / state manager.
- Decoded DDXPlus evidence questions and values.
- Legal/gated evidence access.
- Initial-evidence one-shot classifier.
- Full-evidence one-shot ceiling classifier.
- LLM-only sequential evidence-acquisition policy.
- Cost-sensitive lambda stopping.
- Offline matched-evidence MLP comparator:
  - LLM chooses evidence.
  - MLP diagnoses afterward from exactly that evidence.

Planned next pieces:

- Online LLM-MLP hybrid:
  - MLP updates diagnostic belief after each reveal.
  - LLM receives selected MLP belief signals.
  - stop/request policy uses confidence, margin, stability, cost, and LLM-MLP disagreement.

- Architecture-agnostic algorithmic ledger:
  - reusable signals for single-agent, hybrid, multi-agent, heuristic, or future optimization methods.

- Possible later multi-agent system:
  - only if roles are grounded in ledger signals.

- Possible later evolutionary optimization:
  - tune ledger weights/rules/stopping thresholds rather than replace the agentic system.

The core system is therefore:

```text
DDXPlus case
  -> deterministic evidence ledger
  -> algorithmic/diagnostic signals
  -> LLM evidence-acquisition controller
  -> partial-evidence MLP diagnostic belief head
  -> cost/disagreement-aware stop-request loop
  -> final diagnosis or differential
```

That full combination is what we should assess for novelty.

## 3. Closest Prior Work

### 3.1 DDXPlus NeurIPS Paper

Source: [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://papers.nips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)

What it already does:

- introduces DDXPlus
- uses age, sex, and initial evidence
- defines an evidence-acquisition diagnostic task
- evaluates automatic diagnosis systems that collect evidence and predict diagnosis/differential
- reports interaction length, evidence recall, differential recall/precision/F1, and pathology accuracy
- evaluates AARLC and BASD

Important details:

- AARLC has an evidence-acquisition branch trained with RL and a supervised classifier branch.
- AARLC aligns the acquisition and classification branches using classifier entropy.
- BASD is a supervised approach with an MLP classifier, hidden layers of size 2048.
- The paper explicitly treats interaction length and evidence collection as key metrics.

Why this is close:

- The task setup is the same broad application.
- AARLC already combines sequential evidence acquisition with a classifier.
- BASD already uses MLP-style diagnosis over acquired evidence.

Why our project is still different:

- The original DDXPlus baselines are trained DDXPlus-specific RL/supervised agents, not LLM-controlled workup policies.
- Their systems do not use a modern LLM as a legal evidence-action planner over decoded patient evidence.
- They do not perform our matched-evidence decomposition between LLM final reasoning and a partial-evidence neural head.
- They do not study an online LLM-MLP feedback loop.
- They do not focus on an explicit ledger as a reusable state/control/explainability layer for future LLM, hybrid, or multi-agent systems.

Novelty implication:

- We cannot claim novelty in "sequential diagnosis on DDXPlus."
- We can claim novelty in the LLM-led, ledger-gated, matched/hybrid evaluation of this DDXPlus workup problem.

### 3.2 Active Feature Acquisition

Source: [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://proceedings.neurips.cc/paper_files/paper/2018/file/e5841df2166dd424a57127423d276bbe-Paper.pdf)

What it already does:

- frames diagnosis-like problems as sequentially acquiring costly features.
- trains an RL agent to decide whether to stop and predict or acquire a new feature.
- uses a classifier after feature acquisition.
- optimizes prediction performance and feature acquisition cost.
- evaluates on medical datasets.

Why this is close:

- Our evidence requests are equivalent to active feature acquisition.
- Our MLP final diagnosis after acquired evidence fits the same broad template.
- Our lambda/cost-sensitive stopping has the same conceptual motivation.

Why our project is still different:

- Active feature acquisition work is not specifically the DDXPlus structured diagnostic workup environment.
- It does not use an LLM as the evidence acquisition controller.
- It does not use decoded DDXPlus evidence and value schemas.
- It does not build a clinical-language evidence ledger for LLM interaction.
- It does not evaluate LLM final reasoning versus neural final diagnosis under matched acquired evidence.

Novelty implication:

- We should cite active feature acquisition as a methodological ancestor.
- We should not claim the "acquire features before classification" idea is new.
- Our contribution is the DDXPlus-specific LLM/ledger/neural-head integration and evaluation.

### 3.3 MEDDxAgent

Source: [MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](https://arxiv.org/abs/2502.19175)

What it already does:

- proposes a modular explainable LLM-agent framework for interactive differential diagnosis.
- includes an orchestrator called DDxDriver.
- uses a history-taking simulator.
- uses specialized agents for knowledge retrieval and diagnosis strategy.
- evaluates on a benchmark that includes DDxPlus, iCraft-MD, and RareBench.
- emphasizes iterative refinement and explainability.

Why this is close:

- It is an LLM-agent differential diagnosis system.
- It is modular.
- It includes DDxPlus.
- It uses an orchestrator that maintains and updates patient information and differential diagnoses.
- It explicitly targets incomplete patient profiles and iterative diagnosis.

Why our project is still different:

- MEDDxAgent is a broader modular LLM-agent framework with history simulator, retrieval agent, and diagnosis strategy agent.
- Our work is more tightly tied to DDXPlus's official structured evidence fields and legal root-evidence action space.
- Our ledger reveals exact DDXPlus evidence roots as present/absent/value rather than relying primarily on open-ended simulated patient dialogue.
- Our current and planned hybrid uses a BASD-style partial-evidence MLP as a diagnostic belief head.
- Our matched-evidence design directly asks whether the acquired evidence or the LLM final reasoning is responsible for performance.
- Our planned online hybrid uses neural diagnostic confidence/disagreement as runtime control signals for stopping and more evidence acquisition.

Novelty implication:

- We cannot claim "LLM agents for interactive differential diagnosis on DDxPlus have not been studied."
- MEDDxAgent makes that claim unsafe.
- We can still claim a narrower DDXPlus-structured angle:
  - legal evidence-root ledger,
  - neural matched comparator,
  - online LLM-MLP feedback,
  - evidence-efficiency tradeoff over the official DDXPlus evidence schema.

### 3.4 MediQ

Source: [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](https://arxiv.org/abs/2406.00922)

What it already does:

- argues that static single-turn medical benchmarks miss interactive information-seeking behavior.
- introduces a benchmark where LLMs can ask follow-up questions under incomplete information.
- uses abstention/confidence strategies to decide when to ask questions.
- finds that direct prompting for question asking can degrade performance, and that better strategies are needed.

Why this is close:

- MediQ studies LLMs asking clinical questions when initial information is incomplete.
- It treats confidence and abstention as part of the decision to ask more.

Why our project is still different:

- MediQ is not centered on DDXPlus's 223-root structured evidence action space.
- MediQ does not use our BASD-style partial-evidence MLP diagnostic head.
- MediQ does not define matched-evidence neural comparisons on DDXPlus.
- MediQ does not implement the proposed DDXPlus ledger-gated LLM-MLP feedback architecture.

Novelty implication:

- We should cite MediQ to show that LLM question-asking is a known research direction.
- Our DDXPlus-specific structured ledger and neural-head decomposition remain distinct.

### 3.5 AgentClinic

Source: [AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments](https://arxiv.org/abs/2405.07960)

What it already does:

- builds a simulated clinical environment for evaluating LLM agents.
- includes patient interactions, multimodal data collection, incomplete information, and tool use.
- shows sequential decision-making is harder than static medical QA.
- explores tools like persistent notes, retrieval, and reflection cycles.

Why this is close:

- It is explicitly about medical LLM agents in simulated clinical settings.
- It emphasizes incomplete information and sequential data gathering.

Why our project is still different:

- AgentClinic is a general simulated clinical-agent benchmark, not a DDXPlus structured evidence-root workup system.
- It does not focus on a BASD-style neural diagnostic head over DDXPlus slot states.
- It does not provide our matched-evidence decomposition of LLM evidence acquisition versus neural final diagnosis.

Novelty implication:

- We should not claim that LLM clinical agents with tools are new.
- Our novelty is narrower and benchmark-specific.

### 3.6 MedClarify

Source: [MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions](https://arxiv.org/abs/2602.17308)

What it already does:

- computes candidate diagnoses.
- generates follow-up questions to reduce diagnostic uncertainty.
- selects questions using expected information gain.
- shows improved diagnostic performance over single-shot LLM diagnosis.

Why this is close:

- It is explicitly an LLM information-seeking diagnostic agent.
- It uses uncertainty-aware question selection.
- It resembles the direction we discussed for algorithmic ledger signals.

Why our project is still different:

- It is not specifically the DDXPlus structured evidence-root environment.
- It does not use our deterministic DDXPlus evidence ledger.
- It does not combine LLM evidence control with a BASD-style partial-evidence MLP head.
- It does not use matched-evidence decomposition as the central evaluation strategy.

Novelty implication:

- We should be careful claiming information-gain-guided LLM questioning as new.
- Our novelty is the DDXPlus-specific, ledger-controlled, neural-head hybrid implementation and comparison.

## 4. Has The Exact Holistic Thing Been Done?

Based on the reviewed literature, the closest systems cover large parts of the idea, but not the exact combination.

### What has definitely been done

- Sequential diagnostic workup on DDXPlus.
- RL-based evidence acquisition with classifier branches.
- Supervised BASD-style MLP diagnosis over acquired evidence.
- Active feature acquisition for medical diagnosis.
- LLM agents that ask follow-up clinical questions.
- Modular LLM differential diagnosis systems.
- LLM systems evaluated on DDxPlus as part of broader differential diagnosis benchmarks.
- Uncertainty-aware follow-up question generation.

### What I did not find as an exact match

I did not find a paper that combines all of the following as the central system:

- DDXPlus official structured evidence-root environment.
- Deterministic ledger that strictly gates legal evidence access and records visible/hidden/requested/revealed state.
- LLM controller choosing legal DDXPlus evidence roots from decoded question/value fields.
- BASD-style partial-evidence MLP head trained to diagnose from incomplete, policy-shaped evidence states.
- Matched-evidence analysis comparing:
  - LLM final diagnosis,
  - MLP final diagnosis from exactly the same acquired evidence,
  - initial-evidence one-shot,
  - full-evidence ceiling.
- Cost-sensitive lambda stopping over evidence acquisition.
- Planned online feedback where MLP confidence/disagreement affects the LLM's stop/request decisions.
- Planned architecture-agnostic algorithmic ledger that exposes reusable belief, uncertainty, severity, and discriminative-action signals.

This does not prove no such work exists. It means the exact combination is not obvious from the main DDXPlus, active feature acquisition, and medical LLM-agent literature found here.

## 5. What Is Unique About Our Implementation?

## 5A. Combination-Level Uniqueness Matrix

This section separates individual components from combinations of components. A component by itself can be standard while a specific combination can still be distinctive.

Legend:

- `Not unique`: clearly present in prior work or very standard.
- `Partly unique`: similar ideas exist, but our DDXPlus-specific implementation or evaluation differs.
- `Likely distinctive`: I did not find an exact match for this combination in the reviewed literature, though related ideas exist.
- `Future claim only`: not implemented yet, so it can only be claimed as a planned direction.

| Configuration | Status | Why |
|---|---|---|
| DDXPlus automatic diagnosis | Not unique | This is the core task of the DDXPlus NeurIPS paper. |
| Sequential diagnostic agent on DDXPlus | Not unique | DDXPlus evaluates AARLC/BASD agents that collect evidence and diagnose. |
| Single-agent LLM diagnostic questioning | Not unique | MediQ, AgentClinic, MedClarify, MEDDxAgent, and related work study LLMs asking follow-up clinical questions. |
| Evidence acquisition before classification | Not unique | This is the active feature acquisition setup and also appears in DDXPlus AARLC. |
| BASD-style MLP diagnosis | Not unique | BASD is an official DDXPlus baseline using MLP-style diagnosis. |
| Deterministic evidence ledger as a state tracker | Partly unique | Logging/state tracking exists in agent systems, but our implementation is tied to exact DDXPlus legal root evidence, hidden/visible state, and present/absent/value reveal semantics. |
| Decoded DDXPlus evidence prompts | Partly unique | DDXPlus provides evidence metadata, but using it as decoded LLM-facing legal action space is more specific than the original trained-agent setup. |
| Initial-evidence one-shot MLP | Not unique | This is a baseline-style classifier; useful, but not a novelty claim. |
| Full-evidence one-shot MLP ceiling | Not unique | This is a ceiling comparator; useful for analysis, not novel methodologically. |
| Cost-sensitive stopping / lambda sweep | Partly unique | Cost-sensitive acquisition is standard in active feature acquisition, but applying it to a ledger-gated LLM DDXPlus policy is more specific. |
| Matched-evidence comparator | Partly unique | The idea of comparing under equal information is standard experimental control logic, but using it to decompose DDXPlus LLM evidence acquisition vs BASD-style partial MLP diagnosis is more distinctive. |
| Partial-evidence MLP trained on policy-shaped masks | Likely distinctive | Related to missing-feature and active feature acquisition models, but I did not find this exact DDXPlus trace-shaped matched diagnostic head in prior work reviewed. |
| LLM-only sequential policy + deterministic DDXPlus ledger | Partly unique | LLM agents and ledgers exist, but the strict DDXPlus root-evidence legal reveal setup is a narrower contribution. |
| LLM-only sequential policy + cost-sensitive stopping | Partly unique | MedClarify/MediQ-like work uses uncertainty-aware questioning; our lambda-based DDXPlus evidence-cost sweep is a specific implementation. |
| LLM-only sequential policy + matched-evidence MLP analysis | Likely distinctive | I did not find this exact decomposition on DDXPlus: LLM acquires evidence, then MLP diagnoses from exactly that acquired evidence. |
| Deterministic ledger + partial-evidence MLP | Partly unique | This resembles structured missing-feature diagnosis, but our DDXPlus ledger produces the exact slot state consumed by the MLP. |
| Deterministic ledger + LLM controller + partial-evidence MLP offline comparator | Likely distinctive | This is our current strongest implemented combination. The pieces exist separately, but the DDXPlus-specific LLM-led evidence acquisition plus matched neural-head evaluation appears distinctive. |
| Deterministic ledger + LLM controller + partial-evidence MLP online feedback | Future claim only; likely distinctive if implemented | This would move from retrospective comparison to runtime feedback where MLP belief affects stopping and evidence acquisition. I found related active feature acquisition and MEDDxAgent-style orchestration, but not this exact DDXPlus LLM-MLP feedback setup. |
| Algorithmic ledger alone | Future claim only; not unique by itself | Belief-state tracking, uncertainty, graph signals, and discriminative question scoring are known ideas. The ledger alone is not enough as a novelty claim. |
| Algorithmic ledger + single-agent LLM | Future claim only; likely distinctive in DDXPlus form | A structured belief/signal layer feeding an LLM over legal DDXPlus evidence roots would be more distinctive than plain prompting. |
| Algorithmic ledger + partial-evidence MLP | Future claim only; partly unique | Could resemble active feature acquisition or uncertainty-guided classifiers, but DDXPlus-specific severity/discriminative evidence signals may add value. |
| Algorithmic ledger + online LLM-MLP hybrid | Future claim only; likely distinctive | This is one of the strongest future novelty candidates: ledger computes belief/action signals, MLP estimates diagnosis, LLM selects legal evidence, and stop/request depends on feedback. |
| Multi-agent diagnosis by itself | Not unique | MEDDxAgent and other modular/multi-agent medical LLM frameworks already exist. |
| Multi-agent + deterministic DDXPlus evidence ledger | Future claim only; partly unique | Multi-agent is not new, but forcing all agents through one legal evidence ledger on DDXPlus could improve auditability and reduce hallucinated evidence. |
| Multi-agent + algorithmic ledger + online MLP feedback | Future claim only; likely distinctive but high complexity | This could be unique as a full architecture, but only if roles are concrete and metrics improve. It should not be the next implementation step. |
| Evolutionary algorithm for strategy tuning | Future claim only; not unique by itself | Evolutionary optimization of policies/rules is known. |
| Evolutionary optimization of algorithmic ledger weights/rules | Future claim only; partly unique | Could be distinctive if applied to DDXPlus ledger signals and kept interpretable, but it is later-stage work. |
| Evolutionary optimization + LLM/hybrid controller | Future claim only; potentially distinctive | The idea would be offline evolution of ledger scoring/stopping parameters used by the online LLM/hybrid system. Related optimization exists, but exact DDXPlus application may be novel. |

### 5A.1 Main Takeaway From The Matrix

Individual pieces are mostly not novel.

The project becomes distinctive when we combine:

```text
DDXPlus legal evidence schema
  + deterministic ledger
  + LLM evidence controller
  + partial-evidence MLP diagnostic head
  + matched-evidence decomposition
  + cost-sensitive stopping
```

The strongest future novelty would add:

```text
online MLP feedback
  + architecture-agnostic algorithmic ledger signals
  + severity/discriminative evidence scoring
```

The least defensible novelty claims are:

- sequential diagnosis alone
- LLM agent alone
- MLP classifier alone
- evidence acquisition alone
- multi-agent alone

The most defensible novelty claims are:

- DDXPlus-specific ledger-gated LLM evidence acquisition
- matched-evidence decomposition between LLM and neural diagnostic head
- online LLM-MLP feedback for stop/request decisions
- architecture-agnostic algorithmic ledger as a reusable signal layer
- later severity-aware/evidence-efficiency evaluation

### 5A.2 Practical Priority Ranking

If the goal is novelty per implementation effort, prioritize:

| Priority | Direction | Why |
|---:|---|---|
| 1 | Online LLM-MLP hybrid feedback | Directly extends current implemented work and makes the hybrid real rather than retrospective. |
| 2 | Official DDXPlus metric alignment | Makes results comparable to published baselines and harder to dismiss. |
| 3 | Architecture-agnostic algorithmic ledger signals | Creates a reusable technical contribution across single-agent, hybrid, and later multi-agent systems. |
| 4 | Heuristic evidence-acquisition baseline | Helps prove the LLM/ledger strategy is not just equivalent to a simple heuristic. |
| 5 | Severity-aware evaluation | Adds a clinically meaningful angle beyond raw accuracy. |
| 6 | Multi-agent | Only valuable after ledger signals are stable; otherwise likely to add cost/noise. |
| 7 | Evolutionary optimization | Interesting later, but needs stable ledger signals and validation protocol first. |

### 5.1 The Ledger Is Not Just A Log

Our ledger is intended to be the system's source of truth:

- what evidence is initially visible
- what evidence remains hidden
- what root fields are legal to ask
- what has already been requested
- whether a root is present, absent, or value-bearing
- what values were revealed
- how the evidence state changes turn by turn
- what the final diagnosis was based on

This matters because LLM medical agents often operate over conversational text. In our system, the LLM is not allowed to hallucinate access to hidden evidence. It can only request legal evidence roots, and the environment determines what is revealed.

Unique angle:

> A field-level DDXPlus evidence ledger that makes LLM workup state auditable and reusable by neural classifiers, hybrid controllers, and future multi-agent roles.

### 5.2 The LLM Is Not Just The Diagnostician

Many LLM diagnosis systems focus on the LLM as the main answer generator.

Our stronger direction treats the LLM primarily as an evidence-acquisition planner:

- ask the next question
- choose among legal evidence roots
- respond to uncertainty and disagreement
- provide human-readable rationale

Final diagnosis may come from:

- LLM
- MLP
- hybrid/adjudication rule

Unique angle:

> The LLM's main value is tested as evidence-selection intelligence, not assumed as superior final diagnostic reasoning.

### 5.3 Matched-Evidence Decomposition Is Central

This is a major evaluation distinction.

Instead of only asking:

> Did the agent diagnose correctly?

We ask:

> Given the exact evidence the agent acquired, who uses it better: the LLM or the MLP?

That separates:

- value from better evidence acquisition
- value from final diagnostic reasoning
- value from full information availability

Unique angle:

> The project explicitly decomposes whether "agentic improvement" comes from acquiring the right evidence or from superior LLM reasoning.

### 5.4 The Partial-Evidence MLP Is Policy-Shaped

The partial-evidence MLP is not merely a full-evidence classifier with missing fields.

The current direction trains it on incomplete evidence states shaped by observed sequential policy masks:

- demographics always visible
- initial evidence visible
- additional roots sampled from policy-observed request patterns
- unrequested fields unknown

Unique angle:

> The diagnostic head is adapted to the type of partial evidence the agent tends to acquire.

This connects the evidence acquisition policy and the final classifier more directly than a generic one-shot model.

### 5.5 Online Hybrid Feedback Would Be A Stronger Step

The offline matched comparator is already a proof of concept:

```text
LLM acquired evidence -> MLP diagnoses afterward
```

The planned online hybrid would be more meaningful:

```text
current ledger state
  -> MLP belief distribution
  -> confidence/margin/disagreement signals
  -> LLM evidence selection or stop decision
  -> ledger reveal
  -> repeat
```

Unique angle:

> A DDXPlus LLM workup controller that uses a neural diagnostic head as a runtime belief estimator, not just a retrospective evaluator.

This is closer to active feature acquisition, but it differs by keeping the LLM as a transparent legal-action planner and the MLP as a structured belief source.

### 5.6 Cost-Sensitive Evaluation Is Not Just A Request Cap

The project should not only say "we allow 24 questions."

The stronger idea is:

- sweep evidence cost lambda
- measure accuracy vs mean requests
- identify where stopping becomes too aggressive
- compare fraction of full-evidence gain recovered
- evaluate utility under evidence cost

Unique angle:

> An explicit evidence-efficiency frontier for a DDXPlus ledger-controlled LLM/hybrid system.

### 5.7 Architecture-Agnostic Algorithmic Ledger

The future algorithmic ledger should be reusable by multiple controllers:

- single-agent LLM
- online hybrid
- multi-agent
- heuristic baseline
- evolutionary optimizer

It should expose:

- top competing diagnoses
- confidence/margin/entropy
- diagnosis stability
- severity risk
- discriminative evidence candidates
- expected value proxy for asking more
- contradiction/consistency checks

Unique angle:

> The ledger becomes a shared reasoning substrate, not a notebook-specific prompt helper.

## 6. What Claims Are Safe?

### Strong safe claim

> Prior DDXPlus work studies trained RL/supervised diagnostic agents, and recent medical LLM-agent work studies interactive diagnosis. Our project studies a ledger-gated LLM evidence-acquisition controller on DDXPlus, evaluates it against BASD-style neural diagnostic heads under matched evidence, and develops an online hybrid feedback path where neural diagnostic belief can guide stopping and further evidence acquisition.

### Strong safe claim

> The novelty is the application-specific system design and evaluation decomposition: evidence acquisition, final diagnostic reasoning, matched-information neural diagnosis, and full-evidence ceiling are evaluated separately.

### Strong safe claim

> Our approach is not intended to replace RL as the theoretically natural method for learning DDXPlus policies; it investigates whether an LLM/ledger/hybrid approach can provide competitive evidence efficiency, interpretability, and modularity.

### Medium-strength claim

> I did not find prior work implementing this exact DDXPlus ledger-controlled LLM plus partial-evidence MLP hybrid feedback architecture.

Use this carefully. It is a literature-search statement, not a proof of absence.

### Medium-strength claim

> The matched-evidence decomposition is a useful contribution because it prevents over-attributing performance gains to "agentic reasoning."

This is defensible if we report it clearly.

## 7. What Claims Are Unsafe?

Avoid:

> We are the first to do sequential diagnosis on DDXPlus.

False. DDXPlus was built for this and includes AARLC/BASD baselines.

Avoid:

> We are the first to use LLM agents for interactive diagnosis.

False. MediQ, AgentClinic, MEDDxAgent, MedClarify, and related work already study interactive LLM diagnosis.

Avoid:

> We invented hybrid evidence acquisition and classifier diagnosis.

False. Active feature acquisition and AARLC-like methods already combine evidence acquisition with classifiers.

Avoid:

> Our approach is better than RL.

Not established. RL has a natural advantage on simulator-optimized policies.

Avoid:

> Multi-agent is the novelty.

Not yet. Multi-agent only becomes meaningful if role separation and ledger use are concrete and measurable.

Avoid:

> The MLP matched comparator proves LLM reasoning is better.

The matched comparator actually tests whether evidence acquisition or final reasoning drives performance. If the MLP ties or beats the LLM, the project can still be valuable, but the claim changes.

## 8. How To Explain The Novelty To An Instructor

Use this framing:

> DDXPlus already has trained RL and supervised agents for automatic diagnosis. Our work does not claim to invent the task. Instead, we study a different architecture: a ledger-gated LLM workup controller that only accesses legal structured evidence fields, paired with a BASD-style neural diagnostic head trained on partial evidence. The goal is to evaluate whether LLM-controlled evidence acquisition can produce compact, diagnostically useful evidence sets, and whether final diagnosis is better handled by the LLM, the MLP, or an online hybrid of both. This lets us decompose performance into evidence acquisition, diagnostic reasoning, and evidence-efficiency.

If challenged with "isn't this already active feature acquisition?", answer:

> It is related, but active feature acquisition usually learns the acquisition policy directly, often with RL. Our setup keeps the acquisition controller as an LLM operating over decoded DDXPlus evidence questions under a deterministic ledger, then tests neural diagnostic heads under matched evidence. The novelty is the DDXPlus-specific LLM/ledger/neural-head integration and analysis.

If challenged with "isn't this already MEDDxAgent?", answer:

> MEDDxAgent is a modular LLM framework for differential diagnosis and includes DDxPlus, but it is not the same as a field-level DDXPlus evidence ledger plus BASD-style partial-evidence MLP head plus matched-evidence and online-feedback analysis. Our work is narrower, more structured around the official DDXPlus evidence schema, and focused on decomposing evidence acquisition from final diagnosis.

If challenged with "what is the actual research question?", answer:

> Can a ledger-controlled LLM/hybrid diagnostic workup system select a small, targeted subset of DDXPlus evidence that approaches full-evidence diagnostic performance, and does the final diagnostic head work better as an LLM, neural classifier, or online hybrid?

## 9. What Would Make The Novelty Stronger?

### 9.1 Implement Online Hybrid Feedback

This is the most important next step.

The current offline matched comparator is useful but retrospective. Online hybrid feedback would make the system more than a post-hoc pipeline.

Minimum implementation:

- after each evidence reveal, run the partial-evidence MLP
- compute top-k, confidence, margin, entropy
- compare MLP top-k with LLM top-k
- stop if agreement/confidence is high enough
- otherwise let LLM ask another question with these signals in context

### 9.2 Add Official DDXPlus Metrics

To be comparable to DDXPlus prior work:

- interaction length
- ground-truth pathology accuracy
- ground-truth pathology included in predicted differential
- positive evidence recall
- differential diagnosis recall
- differential diagnosis precision
- differential diagnosis F1

### 9.3 Add A Simple Evidence-Acquisition Control

For example:

- random legal evidence
- most frequent evidence
- one-shot-prior top-disease discriminative evidence
- MLP-uncertainty greedy evidence

This is important because if a simple non-LLM policy gets the same result, the LLM controller is less compelling.

### 9.4 Add Architecture-Agnostic Algorithmic Ledger Signals

Useful signals:

- diagnosis margin
- entropy
- stability
- severity risk
- evidence discriminativeness
- expected value proxy
- cost-sensitive stop score

This makes the ledger a contribution rather than just a data structure.

### 9.5 Add Severity-Aware Evaluation

DDXPlus includes condition metadata. Severity-aware metrics could make the project more clinically meaningful:

- severe-disease recall
- severe miss rate
- safety-weighted utility
- evidence requests that rule out severe alternatives

This is a promising route to unique value beyond raw accuracy.

### 9.6 Keep Comparators Versioned

Avoid moving-goalpost criticism:

- `matched_v1`: full-evidence model applied to partial evidence
- `matched_v2`: partial-evidence policy-shaped MLP
- `hybrid_v1`: online MLP confidence-gated policy
- `hybrid_v2`: disagreement-aware policy

## 10. Final Novelty Verdict

The exact project is not a clean "never done before" invention. It is an applied integration of several existing research lines:

- DDXPlus automatic diagnosis
- active feature acquisition
- medical LLM agents
- supervised neural diagnosis
- cost-sensitive sequential decision-making
- interpretable state tracking

But the full system we are building appears to be a distinct combination:

> a DDXPlus-specific, ledger-controlled LLM evidence workup controller with BASD-style partial-evidence neural diagnosis, matched-evidence decomposition, cost-sensitive stopping, and planned online LLM-MLP feedback.

That is enough novelty for a strong course research project if we are precise:

- do not claim to invent evidence acquisition
- do not claim to invent LLM diagnostic agents
- do not claim to beat RL before proving it
- do claim a specific, structured, auditable hybrid approach for DDXPlus
- do evaluate whether the value comes from evidence acquisition, final reasoning, or hybrid feedback

The best one-sentence novelty claim:

> We adapt DDXPlus automatic diagnosis into a ledger-controlled LLM/hybrid workup setting, where structured evidence acquisition, neural partial-evidence diagnosis, and matched-information evaluation are combined to study diagnostic accuracy, evidence efficiency, and interpretability under incomplete clinical evidence.

## References

- Fansi Tchango et al. 2022. [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://papers.nips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf).
- Shim, Hwang, Yang. 2018. [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://proceedings.neurips.cc/paper_files/paper/2018/file/e5841df2166dd424a57127423d276bbe-Paper.pdf).
- Rose et al. 2025. [MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](https://arxiv.org/abs/2502.19175).
- Li et al. 2024. [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](https://arxiv.org/abs/2406.00922).
- Schmidgall et al. 2024. [AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments](https://arxiv.org/abs/2405.07960).
- Wong et al. 2026. [MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions](https://arxiv.org/abs/2602.17308).
