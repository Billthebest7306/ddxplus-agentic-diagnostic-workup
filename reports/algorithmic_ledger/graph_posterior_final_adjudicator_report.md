# Graph Posterior Final Adjudicator

Last updated: 2026-05-08

## Summary

Notebook `22` implements an offline mathematical graph-ledger final adjudicator for the frozen Notebook `13` evidence-acquisition run.

- notebook: `notebooks/22_graph_posterior_final_adjudicator.ipynb`
- artifact root: `artifacts/graph_algorithmic_ledger/graph_posterior_final_adjudicator_49case_v1/`
- API usage: none

The key change is architectural: Notebook `22` does not replace the LLM controller and does not change any evidence requests. It uses the evidence already acquired by Notebook `13`, then computes a train-derived graph posterior over all pathologies.

Primary graph score:

```text
graph_score(disease) =
  sum over revealed evidence outcomes:
    clip(log_odds_support(outcome -> disease), -3, 3)
```

The selected policy is a conservative final critic:

```text
override Notebook 13 top-1 only if:
  graph_top1 differs from Notebook 13 top-1
  graph_margin >= 1.0
  graph_score(Notebook 13 top-1) < 0
  graph_score(graph_top1) > 0
```

## Result

Notebook `22` improves the saved Notebook `13` 49-case final prediction while using the exact same evidence trajectory.

| System | Cases | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid | 49 | 43/49 = 0.878 | 0.918 | 0.939 | 0.845 | 6.59 |
| Graph-only final head | 49 | 44/49 = 0.898 | 0.959 | 0.980 | n/a | 6.59 |
| Notebook `22` conservative graph critic | 49 | 44/49 = 0.898 | 0.939 | 0.939 | 0.867 | 6.59 |

The selected critic changed one prediction:

| Case | True pathology | Notebook `13` | Graph critic | Result |
|---|---|---|---|---|
| `test:81691` | Croup | Anemia | Croup | improvement |

Paired result against Notebook `13`:

| Outcome | Cases |
|---|---:|
| Both correct | 43 |
| Graph critic only correct | 1 |
| Notebook `13` only correct | 0 |
| Both wrong | 5 |

Promotion decision:

```text
offline_candidate_promoted
```

This means Notebook `22` is a promoted offline final-head enhancement candidate. It does not replace Notebook `13` as the evidence-acquisition method.

## Post-Run Analysis

The run is encouraging because the selected rule behaves exactly like a conservative critic should. On the 49-case confirmation it fires once, fixes one Notebook `13` error, and introduces no regressions.

The corrected case is the persistent Croup failure:

| Case | True pathology | Notebook `13` | Graph top-1 | Notebook `13` graph score | Graph top-1 score | Margin | Result |
|---|---|---|---|---:|---:|---:|---|
| `test:81691` | Croup | Anemia | Croup | -2.359 | 1.177 | 2.073 | fixed |

This is the clean target condition for the critic: the Notebook `13` answer has negative graph support, the graph alternative has positive support, and the graph margin is large enough to justify a single override.

The 24-case sanity comparison shows the same pattern. Notebook `13` had `22/24`; the graph critic reaches `23/24`, again by fixing Croup, while Pericarditis remains wrong. This is not a separate held-out validation because those 24 cases are contained in the 49-case confirmation, but it confirms that the rule is not an artifact of only the 49-case CSV layout.

The graph-only head is also informative. It reaches `44/49` top-1, `0.959` top-3, and `0.980` top-5, which means the train-derived graph posterior has real ranking signal. But graph-only is not safe enough to replace Notebook `13`: in the COPD miss, for example, the graph distrusts Myocarditis but prefers Scombroid food poisoning, not COPD. The conservative critic correctly abstains because the graph margin is only `0.290`.

This means the current result should not be treated as the endpoint. The important post-run ceiling is:

| Candidate pool | Oracle accuracy |
|---|---:|
| Notebook `13` top-1 or graph top-1 | 44/49 |
| Notebook `13` top-1 or graph top-2 | 46/49 |
| Notebook `13` top-1 or graph top-3 | 47/49 |
| Notebook `13` top-1 or graph top-5 | 48/49 |
| Notebook `13` top-1 or union of LLM/MLP/graph top-5 | 48/49 |

So the graph posterior contains enough information to support a much larger improvement. The current conservative critic only selects the safest rank-1 graph override. A stronger system would need to learn when to trust graph rank-2/rank-3 candidates without creating regressions on already-correct Notebook `13` cases.

Remaining Notebook `22` errors:

| Case | True pathology | Final prediction | True graph rank | Pattern |
|---|---|---|---:|---|
| `test:38475` | Acute COPD exacerbation / infection | Myocarditis | 4 | graph flags Notebook `13` as weak but has no confident correct alternative |
| `test:111176` | Acute rhinosinusitis | Chronic rhinosinusitis | 2 | graph and Notebook `13` both prefer the chronic neighbor |
| `test:8666` | Influenza | HIV initial infection | 2 | graph and Notebook `13` both prefer HIV initial infection |
| `test:62878` | Pericarditis | Anemia | 15 | true disease is not supported by the revealed evidence state |
| `test:125508` | Unstable angina | Anemia | 3 | true disease is plausible but below Anemia on revealed evidence |

This split matters. Some errors are final-head confusions between close neighbors; others are evidence-trajectory failures where the final visible state does not support the true pathology strongly enough for any posterior critic to recover it.

The realistic next target is therefore not another Croup-specific patch. It is a calibrated graph/LLM/MLP reranker:

- candidate pool: Notebook `13` top-1 plus graph top-3/top-5, MLP top-5, and LLM top-5
- features: graph rank, graph net support, graph contradiction, graph posterior, LLM rank, MLP rank, MLP confidence/margin/entropy, stop reason, request count, and disease-family/confounder indicators
- training source: train/validate-derived partial evidence states or a held-out set of fresh Notebook `13` traces
- evaluation: one final locked evaluation on the 49-case confirmation or a new held-out/live confirmation

Promotion should require a real paired gain, ideally at least `47/49`, with very few or zero regressions. Tuning thresholds directly on the six 49-case misses would not be defensible.

## Why This Is Different From Notebooks 17-21

Earlier graph-ledger experiments mostly tried to use graph information inside the live policy:

- Notebook `17`: hard graph top-10 shortlist, rejected.
- Notebook `18`: graph-advisory shortlist, rejected.
- Notebook `20`: graph prompt context, improved top-5 but reduced top-1.
- Notebook `21`: hand-rule graph adjudication over the 24-case graph-context pilot, no promotable rule.

Notebook `22` uses graph information only after the workup is complete. This avoids the main failure mode from the graph-controller notebooks: graph scores can over-constrain the evidence trajectory if the active differential is already biased.

## Interpretation

The result is a useful algorithmic basis for the project:

- Notebook `13` remains the strong evidence-acquisition controller.
- The train-derived graph ledger provides a mathematical final-state consistency check.
- Signed graph support and contradiction can identify at least one wrong Notebook `13` final answer without causing regressions on the 49-case trace.
- The result supports graph-as-critic, not graph-as-controller: the graph is strongest when asked to adjudicate a completed evidence state, not when it constrains the next-question action space.

The strongest claim is now:

> Sequential LLM-led evidence acquisition plus a conservative train-derived graph posterior critic reaches `44/49` accuracy on the saved 49-case Notebook `13` trace while preserving the same `6.59` mean evidence requests.

## Limitations

- This is an offline final-head adjudicator on saved traces.
- It should not be described as a new live evidence-acquisition policy.
- The fixed conservative rule was chosen before the notebook run, but the result still needs held-out or live confirmation before becoming the final defended method.
- The graph critic fixes Croup but does not fix COPD, rhinosinusitis, Influenza, Pericarditis, or Unstable angina.

## Artifact Contract

Notebook `22` writes:

- `resolved_run_config.json`
- `graph_final_state_features.csv`
- `case_graph_final_state_features.csv`
- `graph_adjudicator_policy_summary.csv`
- `case_level_graph_adjudicator_results.csv`
- `paired_notebook13_vs_graph_adjudicator.csv`
- `hard_case_graph_adjudicator_audits.json`
- `selected_graph_adjudicator.json`
- figures under `figures/`

Static validation and top-to-bottom code-cell execution completed successfully with no API key and no live API path.
