# Design Plan: LLM-Led Graph Evidence Ledger

Created: 2026-05-08

## Corrected Architecture

The graph ledger is not the evidence-acquisition controller.

```text
DDXPlus environment
-> deterministic evidence ledger
-> graph evidence ledger updates support/contradiction/unresolved pairs
-> prompt compiler gives compact graph state to LLM
-> LLM chooses next evidence request
-> DDXPlus reveals answer
-> partial-evidence MLP evaluates stop readiness
-> repeat
```

The LLM remains responsible for question choice. The graph ledger is responsible for state interpretation.

## What The Graph Ledger Should Represent

### Global Train-Derived Graph

Nodes:

- `Diagnosis:<pathology>`
- `EvidenceRoot:<E_id>`
- `Outcome:<E_id>=<state>`

Edges:

- `Diagnosis -> Outcome` with train-derived support score
- `Outcome -> Diagnosis` with contradiction score
- `EvidenceRoot -> Outcome` possible outcome relation
- `EvidenceRoot -> EvidenceRoot` parent/child relation where available

Train-derived weights:

- `P(outcome | diagnosis)`
- `P(outcome | not diagnosis)`
- log-odds support
- mutual information / reliability
- global prevalence
- rare-but-decisive flag

### Episode Graph

Nodes:

- observed outcome nodes
- currently active diagnosis nodes
- unresolved competitor-pair nodes
- requested evidence roots

Edges:

- observed evidence supports diagnosis
- observed evidence contradicts diagnosis
- missing evidence would separate diagnosis pair
- evidence root already requested
- child root legal only if parent observed or implied

The episode graph updates after every DDXPlus reveal.

## Core Ledger Signals

The LLM does not need raw matrices. It needs a compact state summary.

### 1. Current Active Differential

Use the union of:

- LLM top-5 from previous turn
- MLP top-5 from current ledger state
- initial one-shot prior top-5
- optional deterministic prior top-3

For each active diagnosis:

- probability/confidence where available
- support score from observed evidence
- contradiction score from observed evidence
- most supportive revealed findings
- most contradictory revealed findings

### 2. Support/Contradiction Summary

Example prompt block:

```json
{
  "ledger_support_summary": [
    {
      "diagnosis": "Pericarditis",
      "support_score": 2.41,
      "contradiction_score": 0.88,
      "supports": ["pleuritic chest pain present", "worse when breathing deeply present"],
      "contradicts": ["fever absent"]
    }
  ]
}
```

This helps the LLM understand whether its current favorite diagnosis is actually supported by revealed evidence.

### 3. Unresolved Competitor Pairs

For top competing diagnoses, compute pairwise unresolvedness:

```text
unresolved_pair_score(d1, d2)
= similarity of current support
+ disagreement between LLM and MLP
+ remaining high-support missing discriminators
```

Example:

```json
{
  "unresolved_pairs": [
    {
      "pair": ["Croup", "Viral pharyngitis"],
      "why_unresolved": "both explain sore throat/cough; no decisive airway evidence confirmed",
      "useful_discriminators": [
        "stridor/high pitched sound when breathing",
        "barking cough",
        "difficulty breathing"
      ]
    }
  ]
}
```

This should improve LLM question choice without forcing it.

### 4. Missing Discriminator Advisory

The graph can compute candidate roots that separate active diagnoses, but it must present them as advisory context.

Bad:

```text
Only choose from these graph top-10 fields.
```

Good:

```text
The ledger says these unresolved diagnoses would be helped by asking about:
- E_194, high-pitched breathing
- E_66, shortness of breath
- E_220, pain worse on deep inspiration
You may request any legal evidence field from the action menu.
```

### 5. Consistency Warnings

Warnings should be short and explicit:

```json
{
  "consistency_warnings": [
    "LLM top diagnosis is Pericarditis, but MLP top diagnosis is Panic attack.",
    "Current top diagnosis has high contradiction score from revealed evidence.",
    "Two high-probability cardiac diagnoses remain unresolved."
  ]
}
```

These warnings are the core value of an intelligent ledger.

## Prompt Structure

The LLM prompt should contain:

1. demographics
2. visible evidence in decoded language
3. previous request history
4. current ranked differential from LLM and MLP
5. graph ledger summary
6. legal action menu from the Notebook `13` base mechanism
7. strict JSON response schema

Key rule:

```text
The graph ledger is advisory. It may identify support, contradictions, and unresolved competitors, but the LLM still chooses the next evidence request.
```

## What To Keep From Notebook 13

Keep unchanged for first implementation:

- `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- same 49-case slice after pilot validation
- deterministic DDXPlus ledger
- legal parent/child gating
- partial-evidence MLP stop rule:

```text
mlp_confidence >= 0.70
mlp_margin >= 0.20
mlp_entropy <= 0.10
min_requests >= 1
```

Do not change stop policy in the first graph-context version. Otherwise we cannot isolate the effect of graph-ledger context.

## What To Change From Notebook 13

Only change the information shown to the LLM.

Add:

- compact graph support summary
- compact graph contradiction summary
- unresolved differential pairs
- missing discriminator advisory
- consistency warnings

Do not:

- replace the shortlist with graph top-k
- use Bayesian VOI as controller
- force the LLM to ask graph-suggested evidence
- override LLM requests except for legality/schema errors

## Expected Benefit

Notebook `13` already asks good questions. The likely improvement is not broad question quality. The likely improvement is hard-case behavior:

- fewer wrong-belief lock-ins
- fewer cases where the LLM ignores contradictions
- better differentiation in repeated confusion families
- better trace explainability

This means success might be:

- same accuracy with clearer diagnostic evidence traces
- one or two hard-case fixes on the 49-case slice
- fewer cap-hit failures
- less diagnostic drift
- improved top-5/ranked differential even when top-1 does not change

## Hard Cases To Watch

Recurring or informative cases:

- `test:81691` Croup
- `test:62878` Pericarditis
- `test:16097` Stable angina
- `test:51421` Chagas
- `test:77908` Ebola
- `test:125508` Unstable angina

The graph-ledger context should be audited on these cases manually.

## Evaluation Design

Pilot:

- same 24-case slice as Notebooks `13`, `17`, and `18`
- compare to Notebook `13` 24-case reference
- run only if API budget allows

Full:

- same 49-case slice as Notebook `13`
- run only if pilot does not regress

Metrics:

- accuracy
- top-3/top-5
- macro-F1
- mean requests
- cap-hit count
- stop-before-cap rate
- LLM/MLP agreement
- number of graph warnings shown
- number of warnings resolved before stop
- hard-case outcomes
- paired win/loss against Notebook `13`

Trace diagnostics:

- was the graph warning shown before a wrong diagnosis?
- did the LLM act on unresolved-pair advice?
- did the graph context change the requested field?
- did contradiction score decrease over turns?
- did support for the final diagnosis increase over turns?

## Scientific Claim If It Works

Strong claim:

> A DDXPlus-native graph evidence ledger can improve or stabilize LLM-led diagnostic workup by converting raw evidence state into compact support, contradiction, and unresolved-differential context, without taking question choice away from the LLM.

This is more aligned with the original project idea than the failed replacement-controller notebooks.

## Scientific Claim If It Fails

Even if it fails, the conclusion is sharper:

> Notebook `13`'s LLM-led trajectory and MLP stop rule are strong enough that additional graph context does not improve the small 49-case slice; graph ledgers may be more valuable for explanation/audit than for performance on DDXPlus.

That would still be a defensible course-project result.
