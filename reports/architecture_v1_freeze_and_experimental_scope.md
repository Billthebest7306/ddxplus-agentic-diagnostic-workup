# Architecture V1 Freeze And Experimental Scope

## Purpose

This note defines what should be treated as **fixed** in the first multi-agent implementation cycle and what should remain **experimental**.

The goal is to stop unnecessary architectural churn.

Right now, the project does **not** need a completely new high-level architecture search. It needs a stable architecture shell and a focused effort on the ledger/control protocol that is most likely to affect results.

## Core Principle

For this project, architecture has two layers:

### 1. Macro-Architecture

This is the high-level arrangement of components:

- shared ledger
- controller
- planner
- differential synthesizer
- critic
- stop agent
- optional retriever

### 2. Micro-Architecture / Coordination Protocol

This is how the system actually behaves:

- what information the ledger stores
- who can write to which fields
- how proposals are validated
- how contradictions are handled
- how requests are scored
- how stop decisions are made

## Main Recommendation

For the next implementation cycle:

- **freeze the macro-architecture**
- **experiment on the ledger/control protocol**

Why:

- the current macro-architecture is already strong enough to build on
- the biggest performance driver is likely the ledger protocol, not swapping one top-level agent diagram for another
- if the macro-architecture keeps changing, it becomes impossible to tell which design decision actually improved results

## What Is Frozen In Architecture V1

These should stay fixed for the next implementation cycle unless something clearly breaks.

### Fixed Component 1. Shared Evidence Ledger

All agents read from and write to a common structured ledger.

Rule:

- agents do **not** communicate through unrestricted peer-to-peer free-form messaging
- the ledger is the single source of truth

### Fixed Component 2. Deterministic Ledger Controller

The controller should remain deterministic-first.

Responsibilities:

- enforce legality
- choose which proposal is accepted
- trigger the next phase
- finalize stop decisions

Rule:

- do not replace this with a fully free-form orchestrator yet

### Fixed Component 3. Workup Planner Agent

There should be one dedicated agent responsible for proposing the next evidence request.

### Fixed Component 4. Differential Synthesizer Agent

There should be one dedicated agent responsible for proposing differential updates from current evidence.

### Fixed Component 5. Skeptic / Critic Agent

There should be one dedicated agent responsible for challenging weak requests, unresolved contradictions, and diagnosis drift.

### Fixed Component 6. Stop Agent

There should be a dedicated stop/disposition component responsible for continue-vs-stop recommendations.

### Fixed Component 7. Optional Retriever

Retrieval may exist as an optional module, but:

- it should be off by default in the core DDXPlus experiments
- retrieval should not be mixed into the first ledger-centered ablations unless explicitly tested

### Fixed Interaction Rule

Use this interaction pattern:

1. planner proposes
2. critic reviews
3. controller approves
4. environment reveals evidence
5. synthesizer updates differential
6. stop agent evaluates
7. controller finalizes continue vs stop

Do not change this loop during the next experimental cycle.

## What Remains Experimental

These are the parts you should actively improve and test.

### Experimental Area 1. Ledger Schema

This is the highest-priority experimental space.

Questions:

- what fields should the ledger contain?
- how much of the differential should be explicit?
- how should support vs contradiction be represented?

Candidate additions:

- diagnosis-specific support evidence
- diagnosis-specific contradiction evidence
- unresolved discriminators
- objection status
- stop certificate state

### Experimental Area 2. Request Frontier Scoring

This is likely a major performance driver.

Questions:

- how should candidate requests be scored?
- how strongly should generic questions be penalized?
- should scores focus on top-1 vs top-2 separation or top-k unresolved mass?

### Experimental Area 3. Claim Validation Rules

Questions:

- which agent claims require validation?
- when can the critic veto?
- when should controller force request vs force stop?

### Experimental Area 4. Differential Update Logic

Questions:

- how should evidence update diagnosis scores?
- how strongly should priors anchor the system?
- how should contradictions reduce confidence?

### Experimental Area 5. Stop Certificate

Questions:

- what exact conditions define sufficient resolution?
- how should unresolved contradictions block stopping?
- how much remaining frontier value is acceptable before stopping?

### Experimental Area 6. Event Triggers

Questions:

- should certain ledger events activate specific agents?
- should contradiction events immediately invoke the critic?
- should low frontier value trigger stop evaluation automatically?

## What Should Not Be Changed Yet

To keep the experiment interpretable, avoid changing these during the next phase:

- number of core agents
- replacing the shared ledger with direct inter-agent dialogue
- replacing the deterministic controller with a pure LLM controller
- introducing hierarchical managers or nested subteams
- mixing retrieval heavily into the core benchmark

Those are valid future ideas, but they would confound the current question.

## Why This Freeze Is The Right Move

If you change both:

- top-level architecture
- and ledger/control protocol

at the same time, then any improvement becomes hard to explain.

If instead you freeze the macro-architecture and vary the ledger protocol, then:

- the experiments are cleaner
- the scientific story is stronger
- the novelty claim is easier to defend

## Working Hypothesis

For this project, the main determinant of multi-agent performance is expected to be:

- the quality of ledger-centered coordination

more than:

- the exact high-level arrangement of agent boxes

This means the likely strongest gains will come from:

- better support/contradiction accounting
- better unresolved-pair tracking
- better request scoring
- better stop control

not from:

- swapping planner/synthesizer/critic for a slightly different role decomposition

## Suggested Experimental Plan

### Phase 1. Freeze V1 Architecture

Keep fixed:

- controller
- planner
- synthesizer
- critic
- stop agent
- shared ledger

### Phase 2. Improve Ledger Protocol

Implement and test:

1. support vs contradiction tables
2. unresolved pair tracking
3. claim proposal / validation states
4. stop certificate

### Phase 3. Run Single Multi-Agent V1 Comparison

Compare:

- refined single-agent notebook `05`
- ledger-centered multi-agent v1

using the same benchmark slice and budgets.

### Phase 4. Only Then Revisit Architecture

If multi-agent v1 underperforms, then revisit:

- agent roles
- hierarchy
- retrieval integration
- decentralized variants

But not before.

## One-Sentence Rule

For the next implementation cycle:

> freeze the top-level architecture and treat the ledger/controller protocol as the main experimental variable.

## Instructor-Facing Version

If asked why you are not still searching for architectures:

> We already have a strong enough architecture shell. The main open question is no longer the boxes-and-arrows arrangement; it is the coordination protocol inside the evidence ledger. That is the part most likely to affect accuracy, stopping quality, and interpretability, so we are freezing the macro-architecture and experimenting on the ledger logic.
