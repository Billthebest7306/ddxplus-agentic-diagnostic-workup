# Hybrid V2 MLP-Discriminative Shortlist Report

Last updated: 2026-05-06

## Purpose

Notebook `14_hybrid_v2_mlp_discriminative_shortlist.ipynb` tested whether the hybrid system should move beyond MLP-guided stopping and also use the partial-evidence MLP to guide question selection.

The candidate change was deliberately narrow:

- keep the notebook `13` ledger, LLM, stop rule, final heads, and 24-case slice fixed
- replace the evidence shortlist with an MLP-discriminative shortlist
- compare v2 directly against the frozen notebook `13` v1 artifact

The v2 shortlist scored legal unrevealed DDXPlus evidence fields using:

```text
score = penalty * (
  0.35 * mlp_pair_gap
  + 0.25 * top1_vs_rest_gap
  + 0.25 * entropy_gain
  + 0.10 * split_balance
  + 0.05 * disagreement_gap
)
```

The intended benefit was better question selection, not a new stopping policy. The stop rule remained the notebook `12` selected MLP stop rule.

## Artifacts

Notebook:

- `notebooks/14_hybrid_v2_mlp_discriminative_shortlist.ipynb`

Live artifact root:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_24case_v1/`

Dry-run smoke artifact:

- `artifacts/sequential_hybrid_mlp_feedback/hybrid_v2_mlp_discriminative_shortlist_dryrun_smoke_v1/`

Key live outputs:

- `metrics.json`
- `predictions.csv`
- `traces.jsonl`
- `v1_v2_paired_comparison.csv`
- `promotion_decision.json`
- `shortlist_score_components.csv`
- `requested_evidence_frequency.csv`
- `hard_case_trace_summary.csv`
- `figures/`

## Live Result

The live run completed on the same 24-case slice as notebook `13`.

| System | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Input tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Notebook `13` hybrid v1 selected stop | 22/24 | 0.917 | 0.917 | 0.917 | 0.867 | 6.58 | 410,536 |
| Notebook `14` hybrid v2 MLP shortlist | 21/24 | 0.875 | 0.958 | 0.958 | 0.840 | 7.38 | 509,158 |

V2 final-head metrics:

| Final head | Accuracy | Top-5 |
|---|---:|---:|
| Agreement hybrid | 0.875 | 0.958 |
| Conservative hybrid | 0.875 | 0.958 |
| LLM final | 0.875 | 0.958 |
| MLP final | 0.833 | 0.917 |

V2 operational metrics:

| Metric | Value |
|---|---:|
| Mean requests | 7.375 |
| Median requests | 3.5 |
| Mean visible roots including initial | 8.375 |
| Stop-before-cap rate | 0.917 |
| Cap hits | 2 |
| Selected stop-rule fired | 18/24 = 0.750 |
| LLM/MLP top-1 agreement | 21/24 = 0.875 |
| Input tokens | 509,158 |
| Output tokens | 23,249 |

## Promotion Decision

The predefined promotion rule rejected v2:

```text
decision = reject_keep_notebook13_v1
```

Reason:

- v2 accuracy dropped from `22/24` to `21/24`
- v2 mean requests increased from `6.58` to `7.38`
- v2 input tokens increased from `410,536` to `509,158`
- v2 improved top-5 from `0.917` to `0.958`, but that does not compensate for lower top-1 accuracy and higher evidence cost

The current main method remains notebook `13`.

## Paired Case Analysis

Paired v1-v2 outcomes:

| Outcome | Cases |
|---|---:|
| Both correct | 20 |
| V1 only correct | 2 |
| V2 only correct | 1 |
| Both wrong | 1 |

V2-only improvement:

| Case | True pathology | V1 prediction | V2 prediction | V1 requests | V2 requests |
|---|---|---|---|---:|---:|
| `test:62878` | `Pericarditis` | `Anemia` | `Pericarditis` | 16 | 24 |

V1-only regressions:

| Case | True pathology | V1 prediction | V2 prediction | V1 requests | V2 requests |
|---|---|---|---|---:|---:|
| `test:51421` | `Chagas` | `Chagas` | `Bronchiolitis` | 23 | 15 |
| `test:8666` | `Influenza` | `Influenza` | `HIV (initial infection)` | 6 | 3 |

Both wrong:

| Case | True pathology | V1 prediction | V2 prediction | Requests |
|---|---|---|---|---:|
| `test:81691` | `Croup` | `Chagas` | `Bronchiolitis` | 23 |

Interpretation:

- v2 fixed one persistent hard case, `Pericarditis`, but only by using the full 24-request cap.
- v2 still failed `Croup`, despite spending 23 requests.
- v2 introduced two new top-1 failures, `Chagas` and `Influenza`.
- v2 therefore changes the error profile but does not improve the evidence-efficiency frontier.

## Evidence-Selection Behavior

The v2 shortlist produced more diagnostically explicit requests, but it was not more efficient.

Most requested evidence fields:

| Evidence id | Request count |
|---|---:|
| `E_129` | 11 |
| `E_41` | 10 |
| `E_201` | 8 |
| `E_77` | 7 |
| `E_214` | 6 |
| `E_148` | 5 |
| `E_124` | 5 |
| `E_45` | 5 |

Mean top-ranked shortlist component values:

| Component | Mean |
|---|---:|
| Score | 0.529 |
| MLP pair gap | 0.567 |
| Top1-vs-rest gap | 0.694 |
| Entropy gain | 0.264 |
| Split balance | 0.666 |
| Disagreement gap | 0.511 |

Interpretation:

- The v2 scoring mechanism is functioning mechanically: it produces high-separation, high-entropy-gain shortlist entries.
- The problem is not implementation failure.
- The problem is policy behavior: the MLP-discriminative shortlist can over-focus on evidence that separates the MLP's current competitors even when those competitors are already wrong or unstable.

## Hard-Case Behavior

### Pericarditis

V2 fixed `test:62878`, true `Pericarditis`.

However:

- it required all `24` requests
- the selected stop rule did not fire
- the case ended by request cap

This is useful diagnostically but not an efficiency win. It suggests the v2 shortlist can eventually find useful discriminators for hard cardiac cases, but the stop policy is not confident enough to stop early.

### Croup

V2 still failed `test:81691`, true `Croup`.

The trace briefly reached `Croup` around turn 4, then drifted away and ended as `Bronchiolitis`. This is important:

- v2 did surface the correct diagnosis transiently
- later evidence and MLP updates destabilized the diagnosis
- the failure is not simply lack of access to the right class label

This points to a future need for stronger diagnostic stability or contradiction handling, not just more MLP-driven evidence selection.

### Chagas And Influenza

V2 introduced two new failures:

- `Chagas` became `Bronchiolitis`
- `Influenza` became `HIV (initial infection)`

Both are signs of overreaction to the MLP-driven competitor set. The shortlist may be too tightly tied to the current MLP belief and not sufficiently anchored to the earlier correct trajectory.

## Scientific Interpretation

Notebook `14` is valuable even though it should not be promoted.

It answers a real research question:

> Does simply replacing v1's shortlist with an MLP-discriminative shortlist improve the current hybrid system?

Current answer:

> No. On the 24-case live slice, MLP-discriminative shortlisting improved top-5 and fixed Pericarditis, but reduced top-1 accuracy and increased evidence usage. Notebook 13 remains the stronger proposed method.

This sharpens the project direction. The MLP is clearly useful as a stopping signal, but letting the MLP drive question selection directly is not automatically beneficial. It may amplify early MLP belief errors and cause diagnostic drift.

## Recommendation

Do not promote notebook `14` to the main method.

Keep notebook `13` as the current frozen proposed system:

```text
LLM-led evidence acquisition
+ deterministic evidence ledger
+ partial-evidence MLP stopping signal
```

Treat notebook `14` as a negative but informative candidate experiment. If v2-style question selection is revisited later, it should be more conservative:

- use MLP-discriminative fields only when MLP belief is stable
- preserve stronger anchor protection for correct early hypotheses
- blend v1 deterministic shortlisting and v2 MLP shortlisting rather than replacing v1
- add safeguards against cases where MLP top competitors are unstable or clinically implausible

For now, the best next step remains either:

- a larger 49-case confirmation of notebook `13`, or
- a live matched-budget LLM-only control to strengthen the stopping-policy claim.
