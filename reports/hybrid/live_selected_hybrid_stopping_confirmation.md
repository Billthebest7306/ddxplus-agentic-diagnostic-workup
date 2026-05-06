# Live Selected Hybrid Stopping Confirmation

## Purpose

Notebook `13_live_selected_hybrid_stopping_confirmation.ipynb` is the live confirmation notebook for the selected stopping rule discovered in notebook `12`.

Notebook `12` was offline replay. It showed that MLP-guided stopping could preserve `22/24` accuracy at about `6.875` requests on saved notebook `08` trajectories. Notebook `13` is designed to test whether that same stop rule still works when it actively controls the live LLM loop.

## Notebook

- `notebooks/13_live_selected_hybrid_stopping_confirmation.ipynb`

Default artifact root:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

Dry-run smoke artifact generated during implementation:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_dryrun_smoke_v1/`

## Selected Stop Policy

The notebook uses the selected policy from notebook `12`:

| Parameter | Value |
|---|---:|
| minimum requested fields | 1 |
| MLP confidence minimum | 0.70 |
| MLP margin minimum | 0.20 |
| MLP entropy maximum | 0.10 |
| MLP stability minimum | 0 |

Policy name:

- `mlp_conf_ge_0.70_margin_ge_0.20_entropy_le_0.10_stab_0`

The LLM still chooses evidence requests. The MLP only supplies online belief feedback and the selected stop signal.

## Fixed Settings

| Setting | Value |
|---|---|
| LLM | `gpt-4.1-mini` |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Request cap | `24` |
| Sample | same 24-case balanced test slice used by notebooks `08`, `11`, and `12` |
| Default live mode | `RUN_LIVE_API = False` |

## How To Run

For a safe smoke test:

```python
RUN_LIVE_API = False
ALLOW_DRY_RUN_BENCHMARK = True
SEQUENTIAL_MAX_CASES = 2
```

For the actual live confirmation:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
SEQUENTIAL_MAX_CASES = 24
```

The notebook uses the same interactive key bootstrap style as notebook `08`. If `LLM_API_KEY` is not already set, it will prompt securely with `getpass`.

Then restart the kernel and run all cells top-to-bottom.

## Validation Completed

Implementation smoke validation was run without live API calls:

- notebook code cells parse cleanly
- dry-run benchmark generated predictions and traces for 2 cases
- selected stop rule fired in the dry-run loop
- root-level artifacts were written
- comparison cells loaded notebook `08`, `11`, and `12` references
- plots and qualitative examples were generated

Dry-run smoke output is not a scientific result. It only validates that notebook `13` is wired correctly before live API usage.

## Live Result

Live artifact:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_24case_v1/`

Run configuration:

| Setting | Value |
|---|---|
| Live API | true |
| LLM | `gpt-4.1-mini` |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Cases | 24 |
| Request cap | 24 |

Main metrics:

| Metric | Value |
|---|---:|
| Agreement-hybrid accuracy | 22/24 = 0.917 |
| LLM-final accuracy | 22/24 = 0.917 |
| MLP-final accuracy | 22/24 = 0.917 |
| Conservative-hybrid accuracy | 22/24 = 0.917 |
| Agreement-hybrid top-3 | 0.917 |
| Agreement-hybrid top-5 | 0.917 |
| MLP-final top-5 | 0.958 |
| Macro-F1 | 0.867 |
| Mean requests | 6.58 |
| Median requests | 4.5 |
| Mean visible roots including initial | 7.58 |
| Stop-before-cap rate | 1.000 |
| Cap hits | 0 |
| Selected stop-rule fired | 20/24 = 0.833 |
| LLM/MLP top-1 agreement | 23/24 = 0.958 |
| Input tokens | 410,536 |
| Output tokens | 20,979 |

Reference comparison:

| Reference | Accuracy | Mean requests |
|---|---:|---:|
| Notebook `08`, lambda `0.10` | 22/24 | 13.04 |
| Notebook `11`, lambda `0.22` | 21/24 | 7.46 |
| Notebook `12`, offline selected stop | 22/24 | 6.875 |
| Notebook `13`, live selected stop | 22/24 | 6.58 |

The live confirmation met the preferred acceptance target: `22/24` correct with fewer than `7.5` mean requests.

Request-efficiency gains:

| Comparison | Request reduction | Input-token reduction |
|---|---:|---:|
| Notebook `13` vs notebook `08` lambda `0.10` | 49.5% fewer requests | 42.2% fewer input tokens |
| Notebook `13` vs notebook `11` lambda `0.22` | 11.7% fewer requests | 9.3% fewer input tokens |

## Error Pattern

Both wrong cases were wrong for all final heads:

| Case | True pathology | Agreement-hybrid prediction | LLM prediction | MLP prediction | Requests | Stop reason |
|---|---|---|---|---|---:|---|
| `test:81691` | `Croup` | `Chagas` | `Chagas` | `Anemia` | 23 | selected MLP stop |
| `test:62878` | `Pericarditis` | `Anemia` | `Anemia` | `Anemia` | 16 | agent stop |

The `Croup` case is the concerning one for live policy quality. Unlike offline replay, where this case stopped earlier with a different wrong diagnosis, the live run spent nearly the full evidence budget and still ended wrong. The LLM and MLP disagreed at the end, but neither head recovered the correct diagnosis.

The `Pericarditis` case remains a persistent hard failure. It was also an offline replay failure and appears to be a trajectory/question-selection problem rather than just a premature stopping problem.

## Interpretation

Notebook `13` confirms the main notebook `12` claim under live interaction. The selected MLP-guided stopping rule preserved the best notebook `08` accuracy while using about half as many requested evidence fields.

This is the strongest evidence-efficiency result in the project so far:

- notebook `08`: `22/24` at `13.04` requests
- notebook `12` offline replay: `22/24` at `6.875` requests
- notebook `13` live confirmation: `22/24` at `6.58` requests

The result supports the hybrid framing:

- LLM is useful as the evidence-acquisition controller
- partial-evidence MLP is useful as an online stop signal
- final-head choice is less important on this run because all final heads had the same top-1 accuracy
- MLP final had better top-5 ranking than the agreement hybrid, so it may still be useful for differential ranking

## 49-Case Confirmation

Notebook `13` was rerun after freezing the method on a broader 49-case balanced test slice.

Artifact:

- `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/`

Run configuration:

| Setting | Value |
|---|---|
| Live API | true |
| LLM | `gpt-4.1-mini` |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Cases | 49 |
| Request cap | 24 |

Main metrics:

| Metric | Value |
|---|---:|
| Agreement-hybrid accuracy | 43/49 = 0.878 |
| LLM-final accuracy | 43/49 = 0.878 |
| MLP-final accuracy | 41/49 = 0.837 |
| Conservative-hybrid accuracy | 43/49 = 0.878 |
| Agreement-hybrid top-3 | 0.918 |
| Agreement-hybrid top-5 | 0.939 |
| Macro-F1 | 0.845 |
| Mean requests | 6.59 |
| Median requests | 5.0 |
| Mean visible roots including initial | 7.59 |
| Stop-before-cap rate | 0.980 |
| Cap hits | 1 |
| Selected stop-rule fired | 36/49 = 0.735 |
| LLM/MLP top-1 agreement | 46/49 = 0.939 |
| Input tokens | 823,478 |
| Output tokens | 42,721 |

Comparison:

| Run | Correct | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| 24-case pilot | 22/24 | 0.917 | 0.917 | 6.58 |
| 49-case confirmation | 43/49 | 0.878 | 0.939 | 6.59 |

Same-case 49-case framing:

| Comparator | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|
| Initial-evidence one-shot | 0.286 | 0.673 | 0 |
| Notebook `13` selected hybrid stop | 0.878 | 0.939 | 6.59 |
| Full-evidence one-shot ceiling | 0.980 | 1.000 | all fields |

The 49-case result is the stronger final result to report. It is less visually impressive than the 24-case pilot, but it is more credible because it covers a broader balanced slice while preserving the same evidence efficiency.

Error cases:

| Case | True pathology | Prediction | Requests | Stop reason |
|---|---|---|---:|---|
| `test:38475` | `Acute COPD exacerbation / infection` | `Myocarditis` | 24 | max requests reached |
| `test:111176` | `Acute rhinosinusitis` | `Chronic rhinosinusitis` | 8 | selected MLP stop |
| `test:81691` | `Croup` | `Anemia` | 19 | agent stop |
| `test:8666` | `Influenza` | `HIV (initial infection)` | 3 | agent stop |
| `test:62878` | `Pericarditis` | `Anemia` | 15 | agent stop |
| `test:125508` | `Unstable angina` | `Anemia` | 2 | agent stop |

Final interpretation:

- notebook `13` remains the strongest current proposed method
- the 49-case confirmation supports the evidence-efficiency claim
- the method is not near-ceiling and should not be framed as solving DDXPlus diagnosis
- the remaining failures are mostly wrong-belief/stability failures, not just insufficient request budget
