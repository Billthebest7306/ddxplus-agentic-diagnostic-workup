# Live Base Trajectory Replicates

Last updated: 2026-05-09

## Summary

Notebook `25` is a replicate runner for the Notebook `13`-style live base workup. It is designed to collect repeated live trajectories on the same 49-case benchmark so a later offline branching lab can measure where the LLM evidence-acquisition path diverges.

- notebook: `notebooks/25_live_base_trajectory_replicates.ipynb`
- dry-run artifact root: `artifacts/trajectory_replicates/notebook13_style_live_base_replicates_dryrun_smoke_v1/`
- live artifact root when enabled: `artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1/`

The graph/Bayes rescue layer is intentionally disabled. This notebook measures the natural live trajectory distribution of the base LLM workup plus MLP-guided stopping.

## Run Design

The notebook runs three replicate roots from one execution:

```text
replicate_r01/
replicate_r02/
replicate_r03/
```

Each replicate uses:

- `gpt-4.1-mini`
- `temperature = 0.0`
- `top_p = 1.0`
- the Notebook `13` selected MLP-guided stop rule
- the same fixed 49-case balanced test scope
- no graph/Bayes rescue intervention

## Artifacts

Each replicate writes:

- `benchmark_cases.csv`
- `predictions.csv`
- `traces.jsonl`
- `raw_api_responses.jsonl`
- `reference_summary.csv`
- `resolved_run_config.json`
- `metrics.json`
- `comparison_summary.json`
- `summary_metrics.csv`

The parent root writes:

- `benchmark_cases.csv`
- `resolved_replicate_study_config.json`
- `replicate_summary.csv`
- `replicate_case_predictions.csv`
- `replicate_case_stability.csv`
- `replicate_stability_summary.json`

## Dry-Run Validation

The notebook was executed in safe dry-run mode with two cases per replicate.

Result:

- static parse passed
- three replicate roots were created
- aggregate replicate/stability files were written
- artifact contract passed
- no live API key was used

The dry-run is not scientific evidence. It validates the runner mechanics before paid live execution.

## Live 49-Case Result

The live run completed and wrote:

```text
artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1/
```

| Replicate | Correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests |
|---|---:|---:|---:|---:|---:|---:|
| `r01` | 44/49 | 0.898 | 0.939 | 0.939 | 0.871 | 7.00 |
| `r02` | 42/49 | 0.857 | 0.918 | 0.918 | 0.816 | 7.02 |
| `r03` | 42/49 | 0.857 | 0.939 | 0.939 | 0.810 | 6.82 |

Stability across the three Notebook `25` replicates:

| Metric | Value |
|---|---:|
| Cases | 49 |
| Same prediction across all three replicates | 45 |
| Prediction instability cases | 4 |
| Correctness instability cases | 3 |

Unstable prediction cases:

| Case | True pathology | Replicate predictions |
|---|---|---|
| `test:35039` | Myocarditis | r01 Myocarditis; r02/r03 Sarcoidosis |
| `test:38475` | Acute COPD exacerbation / infection | r01/r03 COPD; r02 Anemia |
| `test:62878` | Pericarditis | r01/r02 Panic attack; r03 Anemia |
| `test:8666` | Influenza | r01/r02 Influenza; r03 HIV initial infection |

Interpretation:

- The base workup is not fully deterministic even at `temperature = 0.0`.
- Variation is concentrated, not chaotic: most cases are stable, while a few hard cases show meaningful alternate trajectories.
- Naive replicate voting is not enough; the follow-up should study targeted branch triggers and graph/Bayes/MLP adjudication.

## How To Run Live

Open `notebooks/25_live_base_trajectory_replicates.ipynb`.

Set:

```python
RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
REPLICATE_IDS = ["r01", "r02", "r03"]
```

Keep:

```python
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
TOP_P = 1.0
SEQUENTIAL_MAX_CASES = 49
MAX_REQUEST_CAP = 24
```

The live run will write all three replicate artifacts under:

```text
artifacts/trajectory_replicates/notebook13_style_live_base_replicates_49case_v1/
```

## Next Analysis

Notebook `26` now performs the follow-up offline branching trajectory lab over Notebook `13`, Notebook `24` base, and these three Notebook `25` replicates. The goal is to learn label-free divergence features from visible state only: stop/continue instability, near-tied request candidates, MLP entropy/margin, graph contradiction, Bayes posterior ambiguity, and branch-judge feasibility.
