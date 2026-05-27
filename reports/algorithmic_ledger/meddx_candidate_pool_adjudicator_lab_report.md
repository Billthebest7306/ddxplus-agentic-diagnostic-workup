# Notebook 48: MEDDx Candidate-Pool Adjudicator Lab

Notebook `48` is an offline adjudicator lab over the Notebook `46` `v1_eval30` artifacts and the Notebook `47` broad candidate-pool reconstruction. It makes no API calls.

## Control Question

Can a general candidate-pool adjudicator improve Notebook `46` by selecting the right diagnosis from the broad pool, without case-by-case hardcoding or another live run?

## Inputs

- `artifacts/universal_meddx/meddx_aligned_dataset_native_driver_v1_eval30/`
- `artifacts/universal_meddx/meddx_candidate_pool_repair_lab_v1/`
- Notebook `16` train-derived DDXPlus graph edges

## Feature Layer

Notebook `48` builds one row per candidate diagnosis per workup. Features include:

- current final rank, raw LLM rank, and resolver score/rank
- branch rank support and branch confidence
- DDXPlus MLP top-5 rank, confidence, margin, and entropy
- DDXPlus train-derived graph replay over the revealed evidence ledger
- iCraft/RareBench casebase priors and RareBench graph support
- candidate-pool rank and independent signal count

## Results

| Policy | Top-1 | Top-3 | Top-5 | Wins | Regressions | Status |
|---|---:|---:|---:|---:|---:|---|
| Notebook 46 current | 73/90 | 77/90 | 77/90 | 0 | 0 | live baseline |
| Conservative pool educator v1 | 75/90 | 77/90 | 77/90 | 2 | 0 | selected label-free offline candidate |
| Case-blocked HGB educator | 77/90 | 80/90 | 83/90 | 4 | 0 | diagnostic/calibration-style |
| Label-fit logistic educator | 78/90 | 87/90 | 88/90 | 5 | 0 | non-deployable diagnostic |
| Label-fit HGB educator | 86/90 | 88/90 | 88/90 | 13 | 0 | non-deployable diagnostic |
| Candidate-pool oracle | 88/90 | 88/90 | 88/90 | 15 | 0 | non-deployable oracle |

The selected label-free policy is conservative:

```text
1. Protect a high-confidence DDXPlus MLP top-1.
2. Allow a DDXPlus graph override only when graph support is strongly positive and the current diagnosis is graph-contradicted.
3. Allow a source-weighted pool challenger only when it has a large score margin and more independent support than the current answer.
```

In this saved run, step 1 accounts for the two real gains. The graph replay agrees with the MLP on the clean URTI/Bronchitis failures, but it does not safely rescue the remaining DDXPlus misses.

## Interpretation

The candidate-pool architecture is working at the generation layer but not yet at the final adjudication layer. The broad candidate pool contains the truth in `88/90` workups, while Notebook `46` final top-1 is only `73/90`.

The key new finding is the generalization gap:

- label-free rules are safe but modest: `75/90`
- case-blocked learned adjudication improves to `77/90` with zero regressions
- label-fit learned adjudication can reach `86/90`, close to the `88/90` pool ceiling

That means the candidate features contain useful resolver signal, but the current `90`-workup multi-dataset cohort is too small to honestly promote the label-fit educator as final. A deployable learned resolver needs a separate calibration cohort and held-out confirmation, or a frozen calibration rule trained on prior artifacts and tested on future live workups.

## Follow-Up Notebook 49

Notebook `49` implements the calibrated learned resolver suggested here.

Follow-up result:

| Policy | Top-1 | Top-3 | Top-5 | Wins | Regressions | Status |
|---|---:|---:|---:|---:|---:|---|
| Calibrated logistic pool resolver v1 | 78/90 | 80/90 | 81/90 | 5 | 0 | selected offline calibration candidate |
| Strict nested threshold logistic diagnostic | 77/90 | 80/90 | 81/90 | 4 | 0 | threshold stress test |

The selected Notebook `49` model is one system-wide L2 logistic candidate scorer across DDXPlus, iCraft-MD, and RareBench. It is the strongest defensible offline MEDDx resolver candidate so far, but it still needs fresh held-out/live confirmation before promotion.

## Artifacts

- notebook: `notebooks/48_meddx_candidate_pool_adjudicator_lab.ipynb`
- script mirror: `scripts/meddx_candidate_pool_adjudicator_lab_nb48.py`
- artifact root: `artifacts/universal_meddx/meddx_candidate_pool_adjudicator_lab_v1/`

Required outputs:

- `resolved_run_config.json`
- `candidate_level_educator_features.csv`
- `ddxplus_graph_replay_candidate_features.csv`
- `label_free_pool_educator_results.csv`
- `label_free_pool_educator_summary.csv`
- `case_level_learned_diagnostic_results.csv`
- `learned_pool_educator_summary.csv`
- `signal_failure_audit.csv`
- `truth_source_signal_coverage.csv`
- `selected_pool_educator.json`
- figures under `figures/`

## Verification

- `python3 -m py_compile scripts/meddx_candidate_pool_adjudicator_lab_nb48.py` passed
- Notebook `48` code cells parsed with `ast.parse`
- script executed top-to-bottom with no API calls
- artifact contract passed

`jupyter-nbconvert` is not installed in the current environment, so execution verification used the script mirror that generated the notebook.
