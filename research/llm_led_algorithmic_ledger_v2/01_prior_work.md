# Prior Work For LLM-Led Algorithmic Ledger V2

Created: 2026-05-08

## Purpose

This note separates the corrected algorithmic-ledger direction from the earlier replacement-controller experiments.

The earlier graph/Bayesian notebooks asked:

```text
Can an algorithmic graph/Bayes controller choose evidence better than the LLM?
```

The corrected research question is:

```text
Can a graph/algorithmic evidence ledger give the LLM a better structured understanding of the case while preserving LLM-led question choice?
```

This is closer to knowledge-graph-augmented LLM reasoning and clinical state tracking than to pure active feature acquisition.

## 1. DDXPlus Supports Structured Evidence Reasoning

Source:

- [DDXPlus NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)

Relevant finding:

- DDXPlus contains roughly `1.3M` synthetic patients, `49` pathologies, `110` symptoms, and `113` antecedents.
- Evidence includes binary, categorical, and multi-choice fields.
- Some symptoms are hierarchical, which supports legal parent/child inquiry.
- Each patient has a ground-truth pathology and a differential diagnosis.

What this means for us:

- DDXPlus is naturally suited to an evidence ledger.
- The evidence ledger should not only store observed fields; it should represent how revealed evidence supports, contradicts, or fails to separate competing diagnoses.
- DDXPlus differentials justify maintaining an active differential rather than only a single top diagnosis.

## 2. MedKGI Is Close, But Its Role Is Not Exactly Ours

Source:

- [MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring](https://arxiv.org/abs/2512.24181)

Relevant finding:

- MedKGI identifies three problems in diagnostic LLMs: weak grounding, inefficient/redundant questions, and loss of coherence in multi-turn dialogue.
- It uses a medical knowledge graph, information-guided inquiry, and structured state tracking.

What to borrow:

- structured state tracking
- knowledge-grounded diagnostic context
- explicit attention to redundant/inefficient questions
- coherence across turns

What not to copy directly:

- In our failed Notebook `17`, we over-converted graph information into a hard evidence shortlist.
- The corrected version should not let graph scores decide the action space.
- The graph should inform the LLM's reasoning, not replace the LLM's trajectory judgment.

## 3. Think-on-Graph Supports LLM-As-Agent Over Graph

Source:

- [Think-on-Graph](https://arxiv.org/abs/2307.07697)

Relevant finding:

- Think-on-Graph frames the LLM as an agent that interactively explores a knowledge graph and reasons over retrieved paths.
- The key idea is not "graph replaces LLM"; it is "LLM reasons with graph structure."

What this means for us:

- This is the strongest conceptual support for the corrected design.
- Our DDXPlus ledger can be a graph substrate the LLM reasons over.
- The LLM should receive compact graph-derived summaries and choose the next question.

## 4. Dr.Knows / KG-Augmented Diagnosis Supports Explainable Diagnostic Pathways

Source:

- [Leveraging Medical Knowledge Graphs Into Large Language Models for Diagnosis Prediction](https://arxiv.org/abs/2308.14321)

Relevant finding:

- The paper combines LLMs with a medical knowledge graph for diagnosis prediction.
- The graph is used as an auxiliary instrument to interpret and summarize complex medical concepts.
- The paper emphasizes explainable diagnostic pathways.

What this means for us:

- Our graph ledger should produce explainable state summaries.
- Useful outputs are not just action scores; they include "why this diagnosis is supported", "why this diagnosis is contradicted", and "what remains unresolved."

## 5. MEDDxAgent Supports Iterative Differential Diagnosis With State Maintenance

Sources:

- [MEDDxAgent arXiv](https://arxiv.org/abs/2502.19175)
- [MEDDxAgent ACL paper PDF](https://aclanthology.org/2025.acl-long.677.pdf)

Relevant finding:

- MEDDxAgent frames differential diagnosis as iterative rather than single-turn.
- It uses an orchestrator that stores, maintains, and updates patient information and ranked differentials.
- It includes history taking and diagnostic strategy components.

What this means for us:

- Our deterministic/graph ledger can play the "state maintenance" role.
- Unlike MEDDxAgent, our immediate next version does not need multiple agents or external retrieval.
- The novelty in our course project can be DDXPlus-native graph state support for a single LLM workup controller.

## 6. Knowledge-Subgraph Prompting Warns Against Dumping The Whole Graph

Source:

- [KoSEL: Knowledge subgraph enhanced large language model for medical question answering](https://www.sciencedirect.com/science/article/abs/pii/S0950705124014710)

Relevant finding:

- KoSEL argues that providing entire knowledge bases to LLMs can create leakage, context-length, privacy, and confusion problems.
- The proposed answer is refined question-relevant subgraphs.

What this means for us:

- The graph ledger should not dump all `223` evidence roots or all disease-evidence edges into the prompt.
- It should compile a small case-relevant subgraph summary:
  - current top diagnoses
  - strongest supporting evidence
  - strongest contradicting evidence
  - unresolved diagnosis pairs
  - suggested discriminators as advisory notes

## 7. Medical KG-Augmented LLM Work Emphasizes Factuality And Grounding

Sources:

- [MedKA](https://www.sciencedirect.com/science/article/pii/S1532046425001005)
- [npj Digital Medicine commentary on KG/RAG for medical LLMs](https://www.nature.com/articles/s41746-024-01081-0)

Relevant finding:

- Medical LLMs struggle with hallucination, inconsistency, and weak integration of domain knowledge.
- Knowledge graphs can provide structured grounding and a model of truth alongside LLM flexibility.

What this means for us:

- The graph ledger should act as the case-specific truth layer.
- The LLM can remain flexible, but every turn should be grounded in the ledger's structured evidence state.

## 8. MDAgents And KG4Diagnosis Show The Broader Multi-Agent/KG Direction

Sources:

- [MDAgents](https://arxiv.org/abs/2404.15155)
- [KG4Diagnosis](https://arxiv.org/abs/2412.16833)

Relevant finding:

- MDAgents studies adaptive LLM collaboration for medical decision-making.
- KG4Diagnosis combines LLMs with knowledge graph construction inside a hierarchical medical-diagnosis setup.

What this means for us:

- These works support the future direction of multi-agent or hierarchical diagnosis.
- But they are not the next step.
- The next step should be a stronger single-agent graph-ledger context system, because our current strongest method is still single-agent Notebook `13`.

## Main Research Conclusion

The corrected approach is not:

```text
graph/Bayes chooses questions for the LLM
```

The corrected approach is:

```text
LLM chooses questions
graph ledger explains the evolving evidence state
MLP decides whether the state is diagnostically stable enough to stop
```

This aligns better with:

- DDXPlus' differential-diagnosis structure
- MedKGI's structured state and knowledge grounding motivation
- Think-on-Graph's LLM-over-graph paradigm
- Dr.Knows' KG-as-auxiliary-diagnostic-pathway framing
- KoSEL's warning that only compact relevant subgraphs should be exposed

## Practical Implication For This Project

Notebooks `17`, `18`, and `19` should be reframed as negative ablations of algorithmic replacement/control.

The next real algorithmic-ledger notebook should be:

```text
Notebook 20: LLM-Led Workup With Graph Ledger Context
```

It should keep Notebook `13`'s strongest parts:

- LLM question choice
- deterministic ledger legality
- partial-evidence MLP stop rule

And add:

- graph-derived support/contradiction summaries
- unresolved diagnosis-pair summaries
- compact advisory evidence notes
- consistency warnings when LLM/MLP/ledger disagree

The success criterion should be improving Notebook `13` hard cases or preserving `43/49` with better trace quality, not proving that graph control beats the LLM.
