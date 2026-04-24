# Evidence Ledger As The Core Algorithm

## Short Answer

The evidence ledger is a strong starting point, but **the ledger by itself is not enough** to claim that your project is meaningfully different from other agentic systems.

Why:

- a shared memory or shared workspace is already a known multi-agent design pattern
- classic AI blackboard systems use a central shared state
- recent LLM multi-agent systems also use shared blackboards, working memory, or dual-memory designs

So if you present the ledger only as:

- a log of findings
- a memory store
- a place where agents write notes

then it will sound like a standard shared-memory pattern, not a project-specific algorithmic contribution.

The stronger claim is:

> our novelty is not merely “multiple agents share memory,” but that they coordinate through a diagnosis-specific, evidence-gated ledger that constrains what can be asked, what can be claimed, how support vs contradiction is accumulated, and when the system is allowed to stop.

That is the version worth showing to the instructor.

## What The Ledger Already Does In This Repo

The current sequential notebooks already contain a meaningful ledger-centered design.

Main implementation locations:

- [05_single_agent_structured_policy_refinement.ipynb](../notebooks/05_single_agent_structured_policy_refinement.ipynb)
- [06_single_agent_budget_scaling.ipynb](../notebooks/06_single_agent_budget_scaling.ipynb)
- supporting explanation: [ledger_implementation_explained.md](ledger_implementation_explained.md)

Current ledger capabilities already present:

1. **Persistent episode state**
   - the system stores visible findings, request history, diagnosis history, and shortlist history across turns

2. **Decoded clinical evidence**
   - the ledger stores human-readable evidence summaries rather than only raw DDXPlus ids

3. **Legal action gating**
   - follow-up questions are only allowed when parent findings are satisfied

4. **Deterministic reveal/update**
   - when evidence is requested, the environment reveals the corresponding finding and appends a structured ledger entry

5. **Diagnosis-state tracking**
   - the system stores evolving predicted diagnosis, ranked differential, confidence, stop signal, and policy flags turn by turn

6. **Anchored sequential control**
   - in notebook `05`, the deterministic diagnosis state and shortlist are driven by the ledger rather than by free-form prompt context alone

This is already much stronger than “chat history plus tools.”

## Why The Current Ledger Is Promising

The ledger matters because diagnosis is not only about the final label. It is about:

- what was known
- what was requested
- what was revealed
- what hypotheses were competing
- why the next question was chosen
- what evidence separated one diagnosis from another

That makes the ledger a good candidate for the central algorithmic object in the project.

In other words, the ledger lets you move the project away from:

- “an LLM talked to a dataset”

toward:

- “a structured diagnostic controller incrementally updated a medically meaningful state and forced all reasoning to pass through that state”

## Why The Current Ledger Is Still Not Enough As A Novelty Claim

Right now, the ledger is partly:

- a state representation
- a control surface for the single-agent policy

but it is **not yet a fully explicit diagnostic algorithm** in the sense an instructor will want when asking, “what is your project-specific method?”

To be strong enough as the distinguishing algorithm, the ledger should do more than store observations and histories. It should explicitly govern:

1. **what evidence counts as supporting vs contradicting each diagnosis**
2. **which evidence requests are legal and high-value**
3. **which agent is allowed to update which part of the state**
4. **when a hypothesis is strengthened, weakened, or frozen**
5. **when the system is allowed to stop**

That is the difference between:

- `shared memory`

and

- `evidence-ledger coordination algorithm`

## The Better Framing: Evidence-Gated Differential Ledger

The strongest framing for this project is to make the ledger the center of a diagnosis-specific control algorithm.

Suggested name:

- **Evidence-Gated Differential Ledger**
- or **Anchored Evidence Ledger**
- or **Differential Evidence Ledger Controller**

Recommended project claim:

> We propose an evidence-gated differential ledger for multi-agent diagnostic workup. Instead of allowing agents to exchange free-form messages directly, all agents read from and write to a structured ledger that stores revealed evidence, request legality, diagnosis-specific support and contradiction, unresolved competing hypotheses, and stop eligibility. The ledger therefore functions not only as memory but as the coordination and control algorithm of the system.

That is much more concrete.

## What The Improved Ledger Should Contain

To make the ledger the real algorithm, add these structured fields.

### 1. Observation Layer

This is the closest to what you already have.

Each revealed evidence item should store:

- `root_evidence_id`
- question text
- source
- status: `present | absent | unknown`
- decoded values
- turn index
- provenance

### 2. Differential Layer

This should be explicitly diagnosis-centered.

For each candidate pathology, store:

- current score
- support evidence list
- contradiction evidence list
- unresolved discriminators
- confidence band
- last-updated turn

This is important because the project is about differential diagnosis, not only observation logging.

### 3. Request Frontier Layer

This should define the current candidate next actions.

For each legal request, store:

- legality reason
- parent gating status
- estimated discriminatory value
- expected impact on top competing diagnoses
- redundancy / genericness penalty
- whether the request is blocked, allowed, or urgent

### 4. Provenance Layer

Every important state update should say:

- which agent wrote it
- whether it was deterministic vs LLM-generated
- what prior evidence justified it
- whether it was challenged or validated by another agent

This becomes essential in the multi-agent version.

### 5. Stop-Certificate Layer

The system should not stop because a single agent “feels confident.”

The ledger should contain a stop certificate such as:

- top diagnosis margin above threshold
- unresolved evidence mass below threshold
- no remaining high-value discriminators
- critic has not raised an unresolved contradiction

This makes stopping part of the algorithm, not an opinion.

## Suggested Ledger Improvements

These are the highest-value improvements if you want the ledger to become the project’s distinctive method.

### Improvement 1. Support/Contradiction Accounting Per Diagnosis

Add explicit evidence buckets for each active diagnosis:

- `supporting_evidence`
- `contradicting_evidence`
- `missing_but_expected_evidence`

This is probably the single most important upgrade.

Why it matters:

- it makes reasoning inspectable
- it supports critic behavior naturally
- it helps differentiate “not enough evidence yet” from “evidence actively goes against this diagnosis”

### Improvement 2. Pairwise Discriminator Tracking

For the top competing diagnoses, store:

- which evidence best separates `A` vs `B`
- whether that evidence is already known
- whether the current workup has actually resolved the pair

This is better than generic confidence because medicine is often about separating a few nearby hypotheses.

### Improvement 3. Claim Validation Workflow

Do not let any agent directly overwrite diagnosis state without validation.

Better pattern:

1. an agent proposes a claim
2. the claim is written to the ledger as `proposed`
3. a deterministic rule or critic agent validates it
4. only then does it become `accepted`

This prevents “whoever spoke last wins.”

### Improvement 4. Event-Driven Ledger Triggers

Agents should be activated by ledger events, not only by fixed round-robin turns.

Examples:

- new contradictory evidence revealed
- top diagnosis margin falls below threshold
- no high-value requests remain
- critic flags unresolved conflict

This makes the architecture closer to a true blackboard / event-driven system.

### Improvement 5. Agent Permissions By Ledger Field

Different agents should be allowed to update different parts of the ledger.

Example:

- planner can propose requests
- diagnostician can propose differential updates
- critic can attach objections
- deterministic controller can finalize legality and stop status

This avoids unstructured free-form multi-agent chat.

### Improvement 6. Explicit Failure Codes

When the system makes a poor diagnosis, the ledger should help classify why.

Examples:

- `generic_question_drift`
- `premature_stop`
- `anchor_overcommitment`
- `contradiction_ignored`
- `insufficient_discrimination`

This is useful for analysis and for demonstrating scientific maturity in the meeting.

## Concrete Ledger-Centric Algorithm

Here is the version I would present as the algorithm.

### Inputs

- initial evidence
- legal evidence graph from DDXPlus
- pathology evidence statistics / priors
- current revealed evidence

### State

- observation ledger
- active differential table
- request frontier
- stop certificate
- provenance / objections

### Loop

1. initialize ledger from demographics + `INITIAL_EVIDENCE`
2. compute anchored differential state
3. enumerate legal requests
4. score candidate requests by discriminatory value against top competing diagnoses
5. planner proposes next request
6. critic checks whether the request is too generic, redundant, or inconsistent with the current unresolved hypotheses
7. controller approves one request
8. environment reveals evidence
9. ledger updates support / contradiction tables for active diagnoses
10. stop certificate is recomputed
11. if stop certificate passes, finalize diagnosis; otherwise continue

That is a real algorithm, not just “agents talking.”

## Pseudocode

```text
InitializeLedger(case)
while not StopCertificateSatisfied(ledger):
    differential = UpdateDifferentialFromLedger(ledger)
    frontier = BuildLegalRequestFrontier(ledger, differential)
    ranked_requests = ScoreRequestsByDiscrimination(frontier, differential)

    proposed_request = PlannerAgent(ranked_requests, ledger)
    critique = CriticAgent(proposed_request, ledger, differential)
    approved_request = ControllerResolve(proposed_request, critique, ledger)

    revealed_evidence = EnvironmentReveal(approved_request)
    ledger = UpdateObservationLayer(ledger, revealed_evidence)
    ledger = UpdateSupportContradictionTables(ledger, differential, revealed_evidence)
    ledger = RecomputeStopCertificate(ledger, differential)

return FinalDiagnosisFromLedger(ledger)
```

## Why This Is More Distinctive Than Generic Agentic Systems

Many agentic systems differ only by:

- number of agents
- role prompts
- orchestration style

That is weak differentiation.

Your stronger differentiation is:

- the agents are secondary
- the **ledger protocol** is primary
- every agent action must be expressed as a ledger update
- diagnosis progresses through evidence-gated state transitions rather than unconstrained dialogue

That is much easier to defend as a project-specific method.

## Instructor-Facing Answer

If the instructor asks, “is the evidence ledger enough?” the best answer is:

> The ledger is enough only if we present it as the central diagnostic control algorithm, not merely as memory. Shared memory alone is not novel. Our actual contribution is the evidence-gated differential ledger: a structured state that controls legal evidence requests, diagnosis support vs contradiction, inter-agent validation, and stop decisions.

That answer is disciplined and technically credible.

## Recommended Next Upgrades Before Multi-Agent

If you want the ledger to carry the novelty claim, the highest-priority upgrades are:

1. add explicit support vs contradiction tracking per diagnosis
2. add pairwise top-differential discriminator tracking
3. add a formal stop certificate
4. add proposal / validation status for agent-written claims
5. add event-triggered agent activation

## Design Inspiration And Related Work

These sources are useful because they show that shared memory alone is not new, while also supporting your choice to turn the ledger into a central controller.

- IBM event-based blackboard architecture for multi-agent systems: [IBM Research](https://research.ibm.com/publications/event-based-blackboard-architecture-for-multi-agent-systems)
- Blackboard-based LLM MAS with dynamic agent selection and shared blackboard: [Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture](https://www.emergentmind.com/papers/2507.01701)
- Multi-agent neurological reasoning with specialized roles and validation loop: [PLOS Digital Health 2025](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001106)
- Clinical decision-making with orchestration plus mutable working memory: [ClinicalAgents (2026)](https://papers.cool/arxiv/2603.26182)

## Bottom Line

The evidence ledger is the right center for your project.

But to make it a convincing “algorithm” rather than just “state,” you should present it as:

- diagnosis-specific
- evidence-gated
- validation-aware
- stop-controlling
- multi-agent coordinating

That framing is strong enough to differentiate the project if you make the proposed upgrades explicit.
