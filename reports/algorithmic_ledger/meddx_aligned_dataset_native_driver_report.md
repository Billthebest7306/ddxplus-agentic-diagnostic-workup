# Notebook 46: MEDDx-Aligned Dataset-Native Driver

Notebook `46` is the successor to Notebook `45` for the MEDDxAgent-style multi-dataset phase.

## Control Question

Can we keep one MEDDx-style benchmark harness across DDXPlus, iCraft-MD, and RareBench while allowing each dataset to use the evidence interface that actually matches its benchmark?

The answer from implementation is yes: the shared outer driver remains unified, but the inner workup adapter is dataset-native.

## Why Notebook 46 Exists

Notebook `45` tried to port the DDXPlus stop/branch/resolver stack into a single universal profile-retrieval simulator. That was too universal. The live pilot showed a DDXPlus regression on `test:18312` (`Influenza`) even though Notebook `44` solved that same case at all budgets.

The failure was architectural:

- DDXPlus evidence is structured root/value state, including meaningful absent findings.
- The Notebook `45` profile simulator approximated those roots with text spans.
- Early DDXPlus stopping could treat a disagreeing MLP signal as safe.
- The first native-root draft of Notebook `46` also exposed an important bug: child roots such as rash details were selectable before their parent rash root was known to be present.

Notebook `46` fixes this by following the MEDDxAgent pattern more closely: one shared driver, dataset-specific benchmark adapters.

## Implemented Architecture

```text
shared MEDDx budget loop, budgets 5/10/15
  -> dataset-native adapter dispatch
     -> DDXPlus structured root ledger
        -> train/validate root-separation utility shortlist
        -> exact present/absent row reveal
        -> partial-evidence MLP monitor and stop rule
        -> optional hypothesis branch/resolver layer using unused budget
     -> iCraft-MD profile Q/A adapter
     -> RareBench phenotype adapter plus conservative graph gate
  -> common top-1/top-3/top-5/request-cost artifacts
```

For DDXPlus, legal actions now respect parent/child evidence structure: a child root is selectable only after its parent root has been revealed present. This prevents the agent from spending a small MEDDx budget on irrelevant child detail fields.

## Artifacts

- notebook: `notebooks/46_meddx_aligned_dataset_native_driver.ipynb`
- script mirror: `scripts/meddx_aligned_dataset_native_driver_nb46.py`
- dry-run artifact root: `artifacts/universal_meddx/meddx_aligned_dataset_native_driver_dryrun_smoke_v1_pilot1/`

Key outputs:

- `resolved_run_config.json`
- `adapter_preflight.csv`
- `universal_cases.csv`
- `predictions.csv`
- `question_answer_ledger.csv`
- `interaction_traces.jsonl`
- `patient_simulator_retrieval_audit.csv`
- `candidate_level_resolver_scores.csv`
- `meddx_style_metrics_summary.csv`
- figures under `figures/`

## Dry-Run Smoke Result

This is not a performance claim; the no-API agents are scripted. It verifies wiring and artifact integrity.

| Dataset | Case | Architecture mode | Top-1 | Top-3 | Top-5 | Questions |
|---|---|---|---:|---:|---:|---:|
| DDXPlus | `test:18312` | `ddxplus_native_structured_root_driver` | 1 | 1 | 1 | 5 |
| iCraft-MD | `icraft_md:55` | `profile_adapter_branching_resolver` | 0 | 0 | 1 | 5 |
| RareBench | `rarebench:LIRICAL:289` | `profile_adapter_branching_resolver` | 1 | 1 | 1 | 5 |

The DDXPlus smoke case recovered `Influenza` after the legal-root fix. The revealed DDXPlus roots were:

```text
E_129 absent: rash parent root
E_48 absent: household size >= 4
E_41 absent: contact with similar symptoms
E_222 absent: secondhand smoke exposure
E_88 present: severe fatigue / stuck in bed
```

## Verification

- `python3 -m py_compile scripts/meddx_aligned_dataset_native_driver_nb46.py` passed
- all Notebook `46` code cells parsed with `ast.parse`
- no-API dry-run smoke executed successfully
- all three enabled adapters loaded
- DDXPlus MLP monitor loaded successfully
- DDXPlus native train/validate root stats built from `30,000` validate cases
- artifact contract passed

## Live Pilot Result

Live `v1_pilot1` completed the intended tiny pilot shape: one case per dataset across budgets `[5, 10, 15]`, for `9` total workups.

| Slice | Top-1 | Top-3 | Top-5 | Mean questions |
|---|---:|---:|---:|---:|
| Overall | 9/9 | 9/9 | 9/9 | 6.00 |
| Budget 5 | 3/3 | 3/3 | 3/3 | 4.00 |
| Budget 10 | 3/3 | 3/3 | 3/3 | 6.67 |
| Budget 15 | 3/3 | 3/3 | 3/3 | 7.33 |

Per-dataset notes:

- DDXPlus `test:18312`, true `Influenza`, was correct at budgets `5`, `10`, and `15`.
- iCraft-MD `icraft_md:55`, true `Levamisole-induced antineutrophil cytoplasmic antibody vasculitis`, was correct at all budgets.
- RareBench `rarebench:LIRICAL:289`, true `Cockayne syndrome`, was correct at all budgets.
- DDXPlus used the native structured root mode; iCraft-MD and RareBench used the profile/phenotype adapter mode.
- RareBench graph support agreed with `Cockayne syndrome` and no longer caused the high-budget regression seen in earlier Notebook `44` analysis.

This is a strong wiring and behavior result, but it is still only one case per dataset. It should not be presented as a stable multi-dataset accuracy estimate.

## Scaled Evaluation Config

After the successful live pilot, the active Notebook `46` config was advanced to:

```text
RUN_VERSION_SUFFIX = "v1_eval30"
LIVE_TOTAL_MAX_CASES = 30
LIVE_BUDGETS_TO_RUN = [5, 10, 15]
```

This queues `30` unique cases balanced across loaded datasets, or `90` live workups total. The pilot averaged about `18.9k` input tokens, `1.1k` output tokens, and `10` API calls per workup; RareBench is the cost-heavy slice because of large candidate lists and graph/discriminator context.

## Scaled Evaluation Result

Live `v1_eval30` completed `30` unique cases across budgets `[5, 10, 15]`, for `90` total workups.

| Slice | Top-1 | Top-3 | Top-5 | Mean questions |
|---|---:|---:|---:|---:|
| Overall | 73/90 | 77/90 | 77/90 | 6.08 |
| DDXPlus | 21/30 | 23/30 | 23/30 | 8.47 |
| iCraft-MD | 28/30 | 30/30 | 30/30 | 4.63 |
| RareBench | 24/30 | 24/30 | 24/30 | 5.13 |

Notebook `47`, Notebook `48`, and Notebook `49` then analyzed these artifacts offline. Notebook `47` showed broad candidate-pool recall of `88/90` and selected a DDXPlus high-confidence MLP guard that improves the saved run to `75/90`. Notebook `48` built a candidate-level adjudicator feature table; its selected label-free educator remains `75/90`, while diagnostic learned educators reach `77/90` case-blocked and `86/90` label-fit against the `88/90` pool oracle. Notebook `49` trains a system-wide calibrated logistic resolver over those features and reaches `78/90` case-blocked with zero regressions.

## Interpretation

Notebook `46` should supersede Notebook `45` for the MEDDx-aligned dataset-native direction. It gives us the right research framing: we are not claiming that one generic prompt can solve every dataset. We are claiming a shared MEDDx-style evaluation framework with dataset-native evidence acquisition and our DDXPlus-derived stopping/resolution machinery where the structured DDXPlus interface exists.

The scaled result is not strong enough as a final universal claim. The key finding from follow-up repair labs is that candidate generation is strong but adjudication is weak. Further paid live work should freeze a candidate-pool resolver trained/calibrated outside the confirmation set, rather than simply repeating larger live runs.
