# LLM-Led Algorithmic Ledger V2

Created: 2026-05-08

This folder resets the algorithmic-ledger direction after the negative controller-replacement experiments.

The corrected premise:

```text
LLM remains the question chooser.
Algorithmic ledger processes, structures, audits, and explains the evidence state.
The ledger informs the LLM; it does not replace the LLM.
```

Why this folder exists:

- Notebooks `17`, `18`, and `19` tested algorithmic control or heavy algorithmic shortlisting.
- Those experiments were useful negative ablations, but they did not match the intended project idea.
- The intended idea is an intelligent evidence ledger that helps the LLM reason over the evolving case graph.

Files:

- [01_prior_work.md](01_prior_work.md): online research and relevant prior work.
- [02_design_plan.md](02_design_plan.md): proposed architecture for the corrected graph-ledger approach.
- [03_notebook20_plan.md](03_notebook20_plan.md): implementation plan for the next notebook.

Current recommendation:

Do not build another graph/Bayesian replacement controller. Build Notebook `20` as:

```text
Notebook 13 baseline loop
+ graph/algorithmic ledger context block
+ support/contradiction/unresolved-differential summaries
+ same LLM-led question choice
+ same MLP-guided stopping
```

The scientific question becomes:

> Can graph-structured ledger context improve the LLM-led workup without taking control away from the LLM?
