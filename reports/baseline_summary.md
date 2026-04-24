# DDXPlus Baseline Suite Summary

The submission is now organized around three notebook-first artifacts:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)
- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)
- [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb)

The only separate code file is [download_ddxplus.py](scripts/download_ddxplus.py), which downloads the official English DDXPlus release from Figshare into the repo-local `dataset/` folder by default.

## What Each Notebook Does

`01_one_shot_classifier_baselines.ipynb`
- self-contained one-shot benchmark suite
- BASD-style DDXPlus slot encoding
- three candidate objectives:
  - `basd_pathology`
  - `basd_differential`
  - `basd_joint`
- validation-based model selection
- writes artifacts under `artifacts/one_shot/`

`02_single_agent_sequential_baseline.ipynb`
- self-contained DDXPlus sequential environment
- evidence ledger and requestable evidence fields
- OpenAI-compatible API placeholders
- dry-run preview mode and live API mode
- resume-safe sampled benchmark under `artifacts/sequential_single_agent/`

`03_compare_baselines.ipynb`
- loads the selected one-shot full artifact
- loads one sequential sampled run
- aligns them on the same `case_id`s
- writes paired comparison artifacts under `artifacts/comparisons/`

## Faithfulness Note

The one-shot baseline remains faithful to the DDXPlus/BASD-style structured observation setup:

- age uses 8 bins
- sex uses 2 slots
- binary evidence uses 1 slot
- categorical evidence uses one slot per possible value, except integer-coded categories which keep one normalized slot
- multi-choice evidence uses one slot per possible value
- the direct diagnosis baseline only reveals the `INITIAL_EVIDENCE` root group at inference time

The sequential notebook uses the same case representation and stable `case_id` scheme so the later comparison is exact rather than approximate.

## Artifact Layout

New runs write to:

- `artifacts/one_shot/`
- `artifacts/sequential_single_agent/`
- `artifacts/comparisons/`

Earlier direct-diagnosis results were archived under:

- `artifacts/_legacy/direct_dx_quick/`
- `artifacts/_legacy/direct_dx_full/`

Those legacy results are still useful as a reference point until notebook 01 is rerun under the new one-shot artifact layout.

## How To Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download DDXPlus:

```bash
python3 scripts/download_ddxplus.py
```

If your local dataset lives somewhere else, set `DDXPLUS_DATASET_DIR` before running the notebooks.

Suggested execution order:

```bash
jupyter lab notebooks/01_one_shot_classifier_baselines.ipynb
jupyter lab notebooks/02_single_agent_sequential_baseline.ipynb
jupyter lab notebooks/03_compare_baselines.ipynb
```

Recommended workflow:

- run notebook 01 in `quick` mode to smoke-test, then in `full` mode for the official one-shot comparator
- run notebook 02 in dry-run mode first, then fill the API placeholders and run the live sampled benchmark
- run notebook 03 after both artifacts exist
