# Simple Multi-Agent Architecture Diagram

This is the simplified meeting-ready version of the architecture.

```mermaid
flowchart LR
    A["Patient Case"] --> B["Controller"]
    B --> C["Evidence Ledger"]
    C --> D["Agents\n(Planner, Synthesizer, Critic, Stop)"]
    D --> B
    B --> E["Final Diagnosis + Trace"]
```

## One-Line Explanation

The controller manages the workflow, all agents read from and write to the shared evidence ledger, and the final diagnosis is produced only after ledger-guided coordination.

## Slightly Expanded Version

```mermaid
flowchart LR
    A["Patient Case\n(Initial Evidence + Demographics)"] --> B["Deterministic Controller"]
    B --> C["Evidence Ledger"]
    C --> D["Planner Agent"]
    C --> E["Differential Synthesizer"]
    C --> F["Critic Agent"]
    C --> G["Stop Agent"]
    D --> B
    E --> B
    F --> B
    G --> B
    B --> H["Final Diagnosis\n+ Evidence Trace"]
```
