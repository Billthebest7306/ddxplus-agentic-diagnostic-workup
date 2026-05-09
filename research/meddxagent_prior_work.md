# MEDDxAgent Prior Work Review

Purpose: summarize MEDDxAgent as the closest LLM-agent prior work to our DDXPlus diagnostic workup project, with enough detail to explain what has already been done and what we should compare against or cite.

Primary sources:

- Paper: [MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](https://arxiv.org/abs/2502.19175)
- PDF: [arXiv PDF](https://arxiv.org/pdf/2502.19175)
- Code link listed in paper: [nec-research/meddxagent](https://github.com/nec-research/meddxagent)

## 1. Executive Summary

MEDDxAgent is a modular LLM-agent framework for interactive differential diagnosis. It is highly relevant to our project because it explicitly argues against complete-profile, single-turn diagnosis and instead evaluates iterative DDx under initially incomplete patient information.

The framework combines:

- a central orchestrator called `DDxDriver`
- a history-taking simulator
- a knowledge retrieval agent
- a diagnosis strategy agent
- iterative updates to patient profile and ranked differential diagnosis

This means our project should not claim that "LLM agents for iterative diagnosis on DDXPlus" are unexplored. MEDDxAgent already establishes that direction.

The strongest reported DDXPlus interactive result in the main table is:

| System | Model | Questions / Turns | DDXPlus GTPA@1 | Avg Rank |
|---|---|---:|---:|---:|
| MEDDxAgent | GPT-4o | 5 | 0.74 | 1.91 |
| MEDDxAgent | GPT-4o | 10 | 0.78 | 1.56 |
| MEDDxAgent | GPT-4o | 15 | 0.86 | 1.29 |
| MEDDxAgent | Llama3.1-70B | 5 | 0.61 | 2.91 |
| MEDDxAgent | Llama3.1-70B | 10 | 0.71 | 2.20 |
| MEDDxAgent | Llama3.1-70B | 15 | 0.68 | 2.30 |
| MEDDxAgent | Llama3.1-8B | 5 | 0.34 | 5.25 |
| MEDDxAgent | Llama3.1-8B | 10 | 0.56 | 3.59 |
| MEDDxAgent | Llama3.1-8B | 15 | 0.58 | 3.10 |

Important: these are not 99% results. The 99% DDXPlus LLM numbers Hassan mentioned are more likely from a different LoRA fine-tuned LLaMA-v3 DDXPlus paper or from DDxT-style full-information/static diagnosis work. MEDDxAgent's interactive DDXPlus result is closer to our research setting.

## 2. Motivation And Problem Framing

MEDDxAgent starts from the observation that differential diagnosis is normally iterative:

- clinicians begin with partial information
- they gather history and symptoms
- they retrieve or recall medical knowledge
- they revise a ranked differential diagnosis over time

The paper criticizes several limitations in prior LLM medical diagnosis work:

- complete patient profiles assumed upfront
- single-attempt diagnosis
- single-dataset evaluation
- isolated optimization of one component instead of a full DDx workflow
- over-reliance on medical QA benchmarks rather than diagnostic interaction

This framing is very close to our own project motivation.

## 3. MEDDxAgent Architecture

### 3.1 DDxDriver

`DDxDriver` is the central orchestrator.

It:

- stores and updates patient information
- maintains the evolving ranked differential diagnosis
- decides which module to call next
- generates agent-specific instructions
- logs intermediate inputs, outputs, and reasoning steps
- enforces stopping criteria such as max iterations or diagnosis stabilization

The paper describes `DDxDriver` as following a ReAct-style pattern:

```text
Observation -> Thought -> Action -> Updated patient profile / DDx
```

This is the main coordination layer of MEDDxAgent.

### 3.2 History-Taking Simulator

The history-taking module simulates doctor-patient interaction.

In the paper's setup:

- one LLM acts as the patient and has access to the full patient profile
- another LLM acts as the doctor and receives only initial patient information plus optional goals
- the doctor asks diagnostic questions
- the patient answers based on the hidden full profile
- the interaction ends when goals are achieved or a question cap is reached

This is different from our project. Our current environment does not use an LLM patient simulator. We reveal exact structured DDXPlus evidence fields through a deterministic ledger.

### 3.3 Knowledge Retrieval Agent

The knowledge retrieval agent searches external medical sources.

The paper uses:

- Wikipedia
- PubMed

The retrieval query is generated from the current patient profile and provisional differential diagnosis. The retrieved content is summarized and fed back into the diagnostic process.

This module is meant to help with rare or complex conditions where internal LLM knowledge may be insufficient.

### 3.4 Diagnosis Strategy Agent

The diagnosis strategy agent generates and ranks candidate diagnoses.

The paper evaluates:

- zero-shot diagnosis
- few-shot diagnosis
- dynamic few-shot diagnosis using embedding-based similar case retrieval
- Chain-of-Thought variants

Dynamic few-shot retrieval uses embeddings such as:

- BioClinicalBERT
- BGE / BAII embeddings

This is a major source of high non-interactive performance in the paper. In DDXPlus full-profile/non-interactive settings, GPT-4o with dynamic few-shot examples reaches approximately `0.96-0.97` GTPA@1 and `1.00` GTPA@5.

## 4. Benchmark Setup

MEDDxAgent evaluates on a broader DDx benchmark containing:

| Dataset | Domain | Size / Scope |
|---|---|---|
| DDXPlus | respiratory/symptom diagnosis | 1.3M synthetic cases, 49 pathologies |
| iCraft-MD | dermatology | 394 skin diseases |
| RareBench | rare diseases | 421 rare diseases |

For evaluation, the paper samples 100 patients from each dataset at a fixed random seed, mainly because LLM-agent experiments are expensive and time-consuming.

The standardized patient format includes:

- optional initial patient information
- full patient profile
- full set of possible diseases

This resembles our use of balanced sampled DDXPlus slices, though our artifact and case-id tracking are more DDXPlus-specific.

## 5. Evaluation Metrics

MEDDxAgent uses:

- `GTPA@k`: whether the ground-truth pathology appears in the top-k diagnoses
- average rank of the correct disease
- progress rate: whether the ground-truth pathology moves upward in the ranked differential across iterations

The paper uses a top-10 disease list and assigns rank 11 if the true pathology does not appear in the top 10.

Our project uses related metrics:

- top-1 accuracy
- top-3 accuracy
- top-5 accuracy
- macro-F1
- mean requested evidence fields
- stop-before-cap rate
- matched-evidence MLP comparison
- full-evidence ceiling comparison

MEDDxAgent focuses more on iterative DDx rank improvement. Our project focuses more on evidence efficiency and structured evidence acquisition.

## 6. Key Results

### 6.1 Non-Interactive Full-Profile Results

When the full patient profile is available, DDXPlus is much easier.

For GPT-4o diagnosis strategy on DDXPlus:

| Method | GTPA@1 | GTPA@5 | Avg Rank |
|---|---:|---:|---:|
| Retrieval / PubMed | 0.69 | 0.90 | 2.27 |
| Retrieval / Wiki | 0.69 | 0.90 | 2.24 |
| Zero-shot standard | 0.69 | 0.90 | 2.21 |
| Zero-shot CoT | 0.71 | 0.92 | 2.10 |
| Few-shot dynamic BAII | 0.96 | 1.00 | 1.06 |
| Few-shot CoT dynamic BERT | 0.96 | 1.00 | 1.05 |
| Few-shot CoT dynamic BAII | 0.97 | 1.00 | 1.03 |

Interpretation:

- DDXPlus becomes close to solved when all relevant patient information and strong similar-case prompting are available.
- Full-profile results are not directly comparable to our sequential incomplete-evidence task.
- These numbers support our use of a full-evidence ceiling comparator.

### 6.2 Interactive DDXPlus Results

For interactive DDXPlus, the main MEDDxAgent results are:

| Model | Method | Questions / Iterations | GTPA@1 | Avg Rank | Progress |
|---|---|---:|---:|---:|---:|
| GPT-4o | KR, n=0 | 0 | 0.18 | 7.33 | - |
| GPT-4o | DS, n=0 | 0 | 0.27 | 6.01 | - |
| GPT-4o | KR, n=5 | 5 | 0.52 | 3.32 | - |
| GPT-4o | DS, n=5 | 5 | 0.72 | 2.14 | - |
| GPT-4o | MEDDx, iter=1 | 5 | 0.74 | 1.91 | 0.00 |
| GPT-4o | MEDDx, iter=2 | 10 | 0.78 | 1.56 | +0.32 |
| GPT-4o | MEDDx, iter=3 | 15 | 0.86 | 1.29 | +0.32 |

For Llama3.1-70B:

| Method | Questions / Iterations | GTPA@1 | Avg Rank |
|---|---:|---:|---:|
| MEDDx, iter=1 | 5 | 0.61 | 2.91 |
| MEDDx, iter=2 | 10 | 0.71 | 2.20 |
| MEDDx, iter=3 | 15 | 0.68 | 2.30 |

For Llama3.1-8B:

| Method | Questions / Iterations | GTPA@1 | Avg Rank |
|---|---:|---:|---:|
| MEDDx, iter=1 | 5 | 0.34 | 5.25 |
| MEDDx, iter=2 | 10 | 0.56 | 3.59 |
| MEDDx, iter=3 | 15 | 0.58 | 3.10 |

Interpretation:

- Iterative information gathering helps substantially.
- Bigger LLMs benefit more reliably.
- Gains plateau around 10-15 questions.
- The interactive task is much harder than full-profile diagnosis.

## 7. Important Limitations From MEDDxAgent

The paper's own limitations matter for our framing:

- Model selection is limited to GPT-4o and Llama3.1 variants.
- Results may not generalize to all LLMs or medical-domain instruction-tuned models.
- Evaluation is English-only.
- The benchmark is text-only, excluding imaging, labs, genomics, and other multimodal evidence.
- The benchmark still does not cover all clinical specialties or real-world patient distributions.
- The modular agent design has high communication cost and latency.

These are useful caveats because our project shares several limitations:

- DDXPlus is synthetic.
- Our LLM experiments use sampled case slices.
- Our method is also text/structured-evidence only.
- Our results should be framed as benchmark evidence, not clinical deployment evidence.

## 8. Why MEDDxAgent Matters For Our Project

MEDDxAgent is the strongest prior-work warning against overclaiming.

Unsafe claims after reading MEDDxAgent:

- "Nobody has studied LLM agents for interactive differential diagnosis."
- "Nobody has used DDXPlus for LLM-based iterative diagnosis."
- "Our project is novel because it uses an orchestrator/agent loop."
- "Our project is novel because it starts with incomplete patient information."

Safer claims:

- "MEDDxAgent establishes LLM-based interactive DDx as an important prior direction."
- "Our project studies a more constrained DDXPlus-native structured evidence setting."
- "Our project replaces freeform patient simulation with deterministic legal evidence-field access."
- "Our project adds an online partial-evidence neural diagnostic head for stopping and adjudication."
- "Our project evaluates whether performance comes from evidence acquisition or from LLM reasoning by using matched-evidence MLP comparisons."

## 9. How We Should Use MEDDxAgent In Our Final Writeup

Recommended framing:

> MEDDxAgent demonstrates that modular LLM systems can improve interactive differential diagnosis when full patient profiles are unavailable. Our work builds in a narrower but more controlled DDXPlus-native direction: instead of simulating freeform patient dialogue, we expose legal structured evidence fields through a deterministic ledger and use an online partial-evidence MLP to guide stopping and final diagnosis. This lets us analyze evidence efficiency and separate the value of evidence acquisition from the value of LLM diagnostic reasoning.

MEDDxAgent should be cited in:

- related work
- motivation for interactive DDx
- comparison against modular LLM-agent systems
- discussion of why our evidence-ledger and matched-evidence design is a narrower contribution

