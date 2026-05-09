# MedKGI-Style Graph Ledger V1 Report

Last updated: 2026-05-07

## Summary

Notebook `16` implements the first algorithmic graph-ledger version for the DDXPlus project:

- notebook: `notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb`
- artifact root: `artifacts/graph_algorithmic_ledger/medkgi_style_offline_notebook13_49case_v1/`
- source run: `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`

This is an offline MedKGI-style analysis. It does not call an LLM. It builds a DDXPlus-native graph from train-derived evidence/outcome statistics, replays Notebook `13` traces, scores legal evidence questions by expected information gain over the active differential, and tests graph stop certificates along the already-recorded trajectories.

The result is useful but specific: the graph signals are meaningful for evidence-selection analysis, especially in failed cases, but the offline stop-certificate replay does not yet prove a better stop policy. The right next live test is therefore a graph-shortlist pilot, not a claim that Notebook `16` already replaces Notebook `13`.

## Method

The notebook builds a global graph from the DDXPlus training split only:

| Component | Value |
|---|---:|
| Train rows | `1,025,602` |
| Pathologies | `49` |
| Root evidence fields | `223` |
| Max retained outcome states per root | `80` |

Graph nodes are implicit in the exported tables:

- `PATHOLOGY:<label>`
- `ROOT:<E_id>`
- `OUTCOME:<E_id>:<state>`

Outcome states are derived from patient evidence:

- binary roots: `present` or `absent`
- categorical roots: value or `absent`
- multi-choice roots: sorted value-set or `absent`
- rare high-cardinality value-sets are collapsed into `__OTHER_PRESENT__` for laptop-safe information-gain scoring

The MedKGI-style candidate score is:

```text
score = penalty * (0.80 * information_gain + 0.15 * split_balance + 0.05 * global_mi)
```

The active diagnosis set is the union of:

- MLP top-5
- LLM top-5
- deterministic top candidates
- one-shot prior top diagnosis when present in the trace state

The belief mixture is:

```text
belief = normalize(0.75 * MLP_probability + 0.25 * LLM_reciprocal_rank)
```

## Reference System

Notebook `13` remains the frozen current proposed method.

| System | Correct | Accuracy | Top-3 | Top-5 | Mean requests |
|---|---:|---:|---:|---:|---:|
| Notebook `13` hybrid v1, 49-case run | `43/49` | `0.878` | `0.918` | `0.939` | `6.59` |

## Evidence-Selection Findings

Notebook `16` asks whether Notebook `13` usually requested evidence that the graph considered high value.

| Final outcome | Requests | Mean graph rank | Median graph rank | Mean graph score | Mean info gain | Top-3 rate | Top-10 rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Incorrect final diagnosis | `71` | `9.73` | `6.0` | `0.233` | `0.194` | `0.352` | `0.676` |
| Correct final diagnosis | `252` | `6.61` | `4.0` | `0.283` | `0.247` | `0.456` | `0.794` |

This is the main positive finding. Correct trajectories tended to request more graph-informative evidence. Failed trajectories were more likely to request lower-ranked or lower-information fields.

Across all `323` actual evidence requests:

| Graph rank bucket | Requests |
|---|---:|
| Rank `1` | `59` |
| Ranks `2-3` | `81` |
| Ranks `4-5` | `45` |
| Ranks `6-10` | `63` |
| Ranks `11-25` | `60` |
| Ranks `>25` | `15` |

The LLM often chose reasonable questions, but there is enough mismatch with the graph ranking to justify testing a graph-constrained shortlist live.

## Hard-Case Findings

The six Notebook `13` errors remain the most important diagnostic signal.

| Case | True pathology | Predicted | Requests | Terminal graph score | Terminal info gain | Mean actual request rank | Requests outside top-10 |
|---|---|---|---:|---:|---:|---:|---:|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | `24` | `0.297` | `0.211` | `10.17` | `9` |
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | `8` | `0.285` | `0.198` | `7.38` | `1` |
| `test:81691` | Croup | Anemia | `19` | `0.404` | `0.333` | `12.37` | `8` |
| `test:8666` | Influenza | HIV initial infection | `3` | `0.618` | `0.790` | `9.33` | `1` |
| `test:62878` | Pericarditis | Anemia | `15` | `0.343` | `0.281` | `6.73` | `3` |
| `test:125508` | Unstable angina | Anemia | `2` | `0.356` | `0.308` | `12.00` | `1` |

Interpretation:

- The failed cases were not simply under-questioned.
- Several failures still had high remaining graph value at the terminal state.
- `Influenza`, `Croup`, and `Unstable angina` are especially concerning because the graph still saw useful unresolved evidence when the system stopped or drifted.
- COPD and Croup had many actual requests outside the graph top-10, suggesting poor evidence trajectory quality rather than just a poor final classifier.

## Stop-Certificate Replay

The graph stop certificate requires:

- Notebook `13` MLP readiness: confidence `>=0.70`, margin `>=0.20`, entropy `<=0.10`, min requests `>=1`
- best remaining graph action score below a threshold

Offline replay results:

| Graph info threshold | Accuracy | Correct | Top-5 | Mean requests | Graph certificate stops | Terminal fallbacks |
|---:|---:|---:|---:|---:|---:|---:|
| `0.03` | `0.878` | `43` | `0.939` | `6.59` | `0` | `49` |
| `0.05` | `0.878` | `43` | `0.939` | `6.59` | `0` | `49` |
| `0.08` | `0.878` | `43` | `0.939` | `6.59` | `0` | `49` |
| `0.10` | `0.878` | `43` | `0.939` | `6.59` | `0` | `49` |
| `0.15` | `0.878` | `43` | `0.939` | `6.59` | `0` | `49` |
| `0.20` | `0.878` | `43` | `0.939` | `6.59` | `0` | `49` |
| `0.30` | `0.878` | `43` | `0.939` | `6.59` | `6` | `43` |
| `1.00` | `0.878` | `43` | `0.939` | `6.59` | `36` | `13` |

The `1.00` threshold is effectively an MLP-only reference because every graph score is below it. It should not be interpreted as a meaningful graph gate.

The real graph-gated thresholds mostly do not fire on the recorded traces. This means Notebook `16` does **not** prove that the graph stop certificate improves Notebook `13`. The useful result is instead that graph evidence values identify low-quality trajectories and missed high-value evidence fields.

## Scientific Interpretation

The first graph ledger version gives us a stronger diagnosis of the remaining bottleneck:

```text
Notebook 13 is not mainly failing because it stops too early.
It is failing because some trajectories gather lower-value evidence or converge to stable wrong beliefs.
```

That supports the next algorithmic move:

```text
Keep Notebook 13's proven MLP-guided stop rule.
Replace or constrain the question shortlist with MedKGI-style graph information-gain ranking.
Run a small live graph-shortlist pilot.
```

This is aligned with the project goal because the graph ledger is acting as an algorithmic controller over evidence acquisition, not just as a record of what happened.

## Subsequent Live Pilot Result

Notebook `17` implemented the live graph-shortlist pilot recommended by this report. It kept Notebook `13`'s MLP-guided stop rule fixed and changed only the evidence shortlist to MedKGI-style graph top-10 fields.

24-case live result:

| System | Correct | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 22/24 | 0.917 | 0.917 | 6.58 |
| Notebook `17` graph shortlist pilot | 20/24 | 0.833 | 0.875 | 6.21 |

Graph-quality result:

| Metric | Value |
|---|---:|
| Mean requested graph rank | 1.76 |
| Mean requested information gain | 0.373 |
| Requests outside graph top-10 | 0 |

Promotion decision:

- `reject_keep_notebook13_v1`

Updated interpretation:

The offline graph scores remain useful for analysis, but the live pilot shows that a hard graph top-10 replacement shortlist is too restrictive. The graph can efficiently select high-scoring evidence for the current active differential, but if that differential is already biased or missing the true condition, the graph can steer the LLM toward locally informative but globally wrong questions.

Updated recommendation:

- keep Notebook `13` as the frozen proposed method
- do not run Notebook `17` `final49` for this v1 graph-replacement design
- use graph scores next as advisory/blended shortlist features rather than as a hard replacement shortlist
- preserve Notebook `13` shortlist diversity and add graph scores as one component of evidence-quality control

## Artifact Map

- `global_evidence_graph_edges.csv`: outcome-to-pathology support/contradiction edges
- `root_outcome_statistics.csv`: root-level MI, entropy, present rate, retained outcome states
- `turn_level_graph_state.csv`: replayed graph state per Notebook `13` turn
- `candidate_action_scores.csv`: graph top-10 candidate evidence fields per turn
- `actual_request_graph_rank.csv`: actual LLM request ranked under graph scoring
- `graph_stop_replay_summary.csv`: graph stop certificate sweep
- `case_level_graph_replay_results.csv`: case-level replay result for each threshold
- `hard_case_graph_audits.json`: detailed hard-case graph timelines
- `selected_medkgi_graph_policy.json`: selected offline policy metadata
- `analysis_summary.json`: compact summary for future notebooks
