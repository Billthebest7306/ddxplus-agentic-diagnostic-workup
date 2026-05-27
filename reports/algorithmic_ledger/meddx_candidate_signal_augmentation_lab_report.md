# Notebook 50: MEDDx Candidate-Signal Augmentation Lab

Notebook `50` is an offline follow-up to Notebook `49`. It does not make API calls.

## Control Question

If the resolver is still failing, is the problem only model discrimination, or are the candidates themselves under-instrumented?

Notebook `50` keeps the Notebook `46` live traces frozen and augments each candidate with stronger dataset-native signals:

- DDXPlus exact-outcome train-derived Naive Bayes support over visible ledger roots.
- RareBench leave-one-case-out HPO phenotype reference overlap.
- iCraft-MD leave-one-case-out vignette/exemplar TF-IDF support.
- Generic visible-text/candidate-label overlap features.

## Main Results

| Policy | Top-1 | Top-3 | Top-5 | Wins | Regressions | Notes |
|---|---:|---:|---:|---:|---:|---|
| Notebook 46 current | 73/90 | 77/90 | 77/90 | 0 | 0 | live baseline |
| Notebook 49 calibrated logistic resolver | 78/90 | 80/90 | 81/90 | 5 | 0 | strongest current deployable-looking resolver |
| Notebook 50 calibrated augmented resolver | 77/90 | 83/90 | 83/90 | 4 | 0 | improves differential ranking, not top-1 |
| Notebook 50 label-fit augmented logistic | 85/90 | 88/90 | 88/90 | 12 | 0 | non-deployable diagnostic |
| Candidate-pool oracle | 88/90 | 88/90 | 88/90 | 15 | 0 | non-deployable oracle |

## Interpretation

Notebook `50` did **not** beat Notebook `49` on case-blocked top-1. The selected augmented resolver reached `77/90`, so it is not promoted.

The useful finding is more nuanced:

- The augmented signals improve ranked differentials: selected top-3/top-5 rises to `83/90`, above Notebook `49`'s `80/90` and `81/90`.
- The label-fit augmented logistic diagnostic reaches `85/90`, close to the `88/90` oracle, so the new features do contain candidate-discriminating information.
- Under case-blocked evaluation, the model cannot learn those signals robustly from only `30` unique case groups.

This means the remaining gap is not likely to be solved by one more hand threshold. The deployable bottleneck is calibration data: enough held-out, live-like candidate pools to learn when to trust DDXPlus Bayes, RareBench HPO overlap, iCraft exemplar support, or the LLM final answer.

## Failure Analysis

Notebook `50` clarifies several hard limits:

- DDXPlus `test:51945` remains weak: URTI is poorly supported by the visible ledger and does not become a strong Bayes/graph candidate.
- iCraft-MD `icraft_md:14` remains hard because `EBS-Dowling Meara` has no same-label exemplar after leave-one-case exclusion.
- RareBench `MME:21` has no useful same-disease reference exemplar for the true diagnosis.
- RareBench `RAMEDIS:369` is a genuine close-neighbor problem: the visible phenotype overlap favors ornithine transcarbamylase deficiency over citrullinemia type I.

## Decision

Notebook `49` remains the strongest current MEDDx resolver calibration candidate for top-1 accuracy.

Notebook `50` should be cited as a candidate-signal audit:

- It shows why the remaining errors are not just a missing resolver trick.
- It provides stronger top-k differential ranking.
- It supports the argument that a larger calibration corpus, not another retrospective hand rule, is needed for a credible push toward `>90%` deployable multi-dataset top-1.

## Artifacts

- notebook: `notebooks/50_meddx_candidate_signal_augmentation_lab.ipynb`
- script mirror: `scripts/meddx_candidate_signal_augmentation_lab_nb50.py`
- artifact root: `artifacts/universal_meddx/meddx_candidate_signal_augmentation_lab_v1/`

Key outputs:

- `candidate_level_augmented_signal_features.csv`
- `ddxplus_bayes_candidate_features.csv`
- `rarebench_hpo_reference_candidate_features.csv`
- `icraft_text_reference_candidate_features.csv`
- `signal_separability_audit.csv`
- `augmented_resolver_policy_summary.csv`
- `case_level_augmented_resolver_results.csv`
- `final_fit_augmented_summary_diagnostic.csv`
- `selected_augmented_signal_policy.json`
- figures under `figures/`

## Verification

- `python3 -m py_compile scripts/meddx_candidate_signal_augmentation_lab_nb50.py` passed
- Notebook `50` code cells parsed with `ast.parse`
- script executed top-to-bottom with no API calls
- `selected_augmented_signal_policy.json` contains valid JSON with no `NaN`
