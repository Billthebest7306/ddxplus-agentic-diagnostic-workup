# Notebook 47: MEDDx Candidate-Pool Repair Lab

Notebook `47` is an offline repair lab over the Notebook `46` live `v1_eval30` artifacts. It makes no API calls.

## Control Question

Did the earlier DDXPlus candidate-pool/hypothesis-branching idea survive the MEDDx multi-dataset adaptation, and can we repair Notebook `46` without case-by-case hardcoding?

## Inputs

- `artifacts/universal_meddx/meddx_aligned_dataset_native_driver_v1_eval30/`
- `predictions.csv`
- `candidate_level_resolver_scores.csv`
- `branch_case_results.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `universal_cases.csv`

## Candidate-Pool Finding

Notebook `46` does have a strong broad candidate pool:

| Dataset | Budget | Current Top-1 | Current Top-5 | Candidate-Pool Recall | Mean Pool Size |
|---|---:|---:|---:|---:|---:|
| DDXPlus | 5 | 0.60 | 0.60 | 1.00 | 16.0 |
| DDXPlus | 10 | 0.70 | 0.80 | 1.00 | 17.0 |
| DDXPlus | 15 | 0.80 | 0.90 | 0.90 | 17.5 |
| iCraft-MD | 5 | 0.90 | 1.00 | 1.00 | 4.0 |
| iCraft-MD | 10 | 1.00 | 1.00 | 1.00 | 4.0 |
| iCraft-MD | 15 | 0.90 | 1.00 | 1.00 | 4.0 |
| RareBench | 5 | 0.80 | 0.80 | 1.00 | 11.1 |
| RareBench | 10 | 0.80 | 0.80 | 0.90 | 11.3 |
| RareBench | 15 | 0.80 | 0.80 | 1.00 | 11.2 |

Overall, the candidate pool contains the truth in `88/90` workups. This means the universal system is not fundamentally failing to generate candidate diagnoses; it is mostly failing to adjudicate them.

## Policy Variants

| Policy | Top-1 | Top-3 | Top-5 | Wins | Regressions | Status |
|---|---:|---:|---:|---:|---:|---|
| Notebook 46 current | 0.811 | 0.856 | 0.856 | 0 | 0 | live baseline |
| DDXPlus high-confidence MLP guard v1 | 0.833 | 0.856 | 0.856 | 2 | 0 | selected offline candidate |
| DDXPlus all-MLP top-1 diagnostic | 0.833 | 0.856 | 0.856 | 2 | 0 | diagnostic |
| Current top-5 oracle | 0.856 | 0.856 | 0.856 | 4 | 0 | non-deployable oracle |
| Candidate-pool oracle | 0.978 | 0.978 | 0.978 | 15 | 0 | non-deployable oracle |

The selected deployable repair is intentionally small:

```text
If dataset == DDXPlus
and DDXPlus MLP confidence >= 0.70
and DDXPlus MLP margin >= 0.20
then protect the DDXPlus MLP top-1 as final top-1.
```

This fixes two Notebook `46` DDXPlus regressions where a very confident correct MLP top-1 was overridden by branches/resolver logic.

## Failure Taxonomy

| Dataset | Failure type | Workups |
|---|---|---:|
| DDXPlus | truth in broad pool but not final top-5 | 6 |
| DDXPlus | resolver regressed high-confidence DDXPlus MLP | 2 |
| DDXPlus | candidate-pool/acquisition miss | 1 |
| iCraft-MD | small-option top-k adjudication failure | 2 |
| RareBench | wrong LLM/graph lock | 5 |
| RareBench | candidate-pool/acquisition miss | 1 |

## Interpretation

The old architecture idea is still the right direction: generate a broad candidate pool, then resolve. Notebook `46` did generate the pool in most cases, but it used a weak final resolver.

The result argues against spending on another large live run immediately. The next final live notebook should incorporate:

- DDXPlus high-confidence MLP protection
- no branch/resolver override after a strong DDXPlus MLP stop
- a candidate-pool final adjudicator for flagged cases
- stricter RareBench graph-lock rules, especially when graph margin is weak
- small-option top-2 adjudication for iCraft-MD

The candidate-pool oracle of `88/90` shows the upside: a good resolver could plausibly reach or exceed 90% on this cohort. The selected no-API repair only reaches `75/90`, so the remaining work is resolver discrimination, not more evidence collection by itself.

## Follow-Up Notebook 48

Notebook `48` implements the candidate-pool adjudicator lab proposed here.

It adds candidate-level educator features from final/LLM ranks, branch ranks, DDXPlus MLP confidence/margin, DDXPlus train-derived graph replay, RareBench graph support, and casebase priors.

Follow-up results:

| Policy | Top-1 | Top-3 | Top-5 | Wins | Regressions | Status |
|---|---:|---:|---:|---:|---:|---|
| Conservative pool educator v1 | 75/90 | 77/90 | 77/90 | 2 | 0 | selected label-free |
| Case-blocked HGB educator | 77/90 | 80/90 | 83/90 | 4 | 0 | diagnostic |
| Label-fit HGB educator | 86/90 | 88/90 | 88/90 | 13 | 0 | non-deployable diagnostic |

This sharpens the conclusion: the resolver signal exists, but the current `90`-workup cohort is too small to promote a learned universal adjudicator without a separate calibration and held-out confirmation split.

## Artifacts

- notebook: `notebooks/47_meddx_candidate_pool_repair_lab.ipynb`
- script mirror: `scripts/meddx_candidate_pool_repair_lab_nb47.py`
- artifact root: `artifacts/universal_meddx/meddx_candidate_pool_repair_lab_v1/`

Required outputs:

- `resolved_run_config.json`
- `candidate_pool_coverage.csv`
- `candidate_pool_summary.csv`
- `candidate_pool_long.csv`
- `case_level_policy_results.csv`
- `policy_variant_summary.csv`
- `failure_taxonomy.csv`
- `failure_taxonomy_summary.csv`
- `selected_repair_policy.json`
- figures under `figures/`

## Verification

- `python3 -m py_compile scripts/meddx_candidate_pool_repair_lab_nb47.py` passed
- Notebook `47` code cells parsed with `ast.parse`
- script executed top-to-bottom with no API calls
- artifact contract passed
