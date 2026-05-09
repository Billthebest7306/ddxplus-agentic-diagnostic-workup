# Bayesian VOI Algorithmic Ledger Offline

Last updated: 2026-05-08

## Summary

Notebook `19` is the next algorithmic ledger candidate after the graph experiments in Notebooks `16`, `17`, and `18`.

- notebook: `notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb`
- smoke artifact: `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1_smoke/`
- full artifact: `artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1/`
- API usage: none

The notebook implements an offline DDXPlus-native Bayesian value-of-information ledger. It maintains a posterior over all `49` pathologies, estimates which unrevealed evidence root has the highest expected posterior value, fuses the Bayesian posterior with the partial-evidence MLP, and stops only when confidence and remaining VOI agree.

The current status is implementation-complete, smoke-validated, and full-run evaluated. The full result is a negative ablation: Bayesian VOI does not beat Notebook `13` and should not be promoted.

## Why This Notebook Exists

Notebook `13` remains the current frozen proposed method:

| System | Correct | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1, 49 cases | 43/49 | 0.878 | 0.939 | 6.59 |

The graph work showed a specific failure pattern:

- Notebook `17` hard graph replacement was too restrictive.
- Notebook `18` graph-advisory blending recovered some rare-disease failures but still underperformed Notebook `13`.
- The remaining problem is not only shortlist informativeness; it is belief correction and knowing whether a candidate evidence field is worth acquiring under the current posterior.

Notebook `19` therefore moves from edge-ranking to posterior-level value of information.

## Method

Train-only Bayesian tables:

- disease prior `P(D)`
- age-bin likelihood `P(age_bin | D)`
- sex likelihood `P(sex | D)`
- root outcome likelihood `P(root_outcome | D)`
- root mutual information and reliability weights
- root outcome log-odds support for contradiction and rare-recovery scoring

The likelihood tables are built only from the DDXPlus train split. Test labels and test differentials are not used inside the policy.

Ledger state:

- visible evidence roots
- requested evidence roots
- legal unrevealed roots with parent/child gating
- Bayesian posterior over all `49` diagnoses
- partial-evidence MLP posterior from the same visible evidence
- fused posterior using fixed log-linear pooling
- contradiction/support score
- request history

Posterior fusion:

```text
fused = softmax(
  0.60 * log(MLP posterior)
+ 0.40 * log(Bayesian posterior)
)
```

Question scoring:

```text
utility =
  0.55 * expected_fused_entropy_reduction
+ 0.20 * expected_margin_gain
+ 0.15 * contradiction_resolution_gain
+ 0.10 * rare_recovery_bonus
- lambda_cost
- redundancy_penalty
```

The notebook first evaluates Bayesian VOI over legal roots, then runs batched partial-evidence MLP counterfactual scoring for the top Bayesian candidate roots.

Stop certificate:

```text
stop if:
  fused_confidence >= 0.70
  fused_margin >= 0.20
  fused_entropy <= 0.35
  max_remaining_utility <= 0
  contradiction_score <= 1.5
  min_requests >= 1
  Bayes/MLP/fused agreement is acceptable
```

Lambda sweep:

```python
lambda_cost = [0.00, 0.02, 0.05, 0.10, 0.15]
```

## Fairness Boundaries

Notebook `19` is offline but still enforces the same evidence-access discipline:

- no API calls
- no hidden test labels inside scoring
- no test differentials inside scoring
- no full-evidence predictions inside scoring
- hidden test evidence is revealed only when the offline VOI agent requests that root
- train-derived statistics are allowed because they are part of the learned policy

This makes Notebook `19` a fair algorithmic-policy comparator rather than a leakage-based oracle.

## Smoke Validation

Smoke configuration:

```text
SMOKE_MODE = True
SMOKE_MAX_CASES = 3
MAX_REQUEST_CAP = 5
LAMBDA_COSTS = [0.05]
TRAIN_LIKELIHOOD_NROWS = 20000
CALIBRATION_ROWS = 1000
```

Smoke result:

| Metric | Value |
|---|---:|
| Cases | 3 |
| Fused correct | 2/3 |
| Fused accuracy | 0.667 |
| Mean requests | 2.67 |
| Stop-before-cap rate | 0.667 |
| Cap-hit count | 1 |

These numbers are not scientific. They only prove that the notebook can build tables, load the partial-evidence MLP, run the Bayesian ledger loop, save traces, and write all expected artifact files.

Expected smoke artifacts were written:

- `diagnosis_priors.csv`
- `root_outcome_likelihoods.csv`
- `root_information_stats.csv`
- `posterior_calibration.csv`
- `notebook13_replay_bayesian_diagnostics.csv`
- `policy_sweep_summary.csv`
- `offline_agent_predictions.csv`
- `offline_agent_traces.jsonl`
- `voi_candidate_scores_top20.csv`
- `hard_case_bayes_audits.json`
- `reference_comparison.csv`
- `promotion_decision.json`
- figures under `figures/`

Static validation also passed:

- all code cells parse
- source notebook has no outputs
- no OpenAI/API adapter code is present
- no `LLM_API_KEY` or live HTTP call path is present

## Full 49-Case Result

Full artifact:

```text
artifacts/bayesian_voi_ledger/bayesian_voi_offline_notebook13_49case_v1/
```

Policy sweep:

| Lambda | Fused correct | Accuracy | Top-3 | Top-5 | Macro-F1 | Mean requests | Cap hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 33/49 | 0.673 | 0.837 | 0.878 | 0.620 | 22.37 | 28 |
| 0.02 | 26/49 | 0.531 | 0.735 | 0.837 | 0.461 | 7.98 | 4 |
| 0.05 | 26/49 | 0.531 | 0.633 | 0.776 | 0.447 | 5.65 | 1 |
| 0.10 | 25/49 | 0.510 | 0.653 | 0.776 | 0.424 | 4.43 | 1 |
| 0.15 | 24/49 | 0.490 | 0.633 | 0.776 | 0.396 | 4.33 | 1 |

Reference comparison:

| System | Correct | Accuracy | Top-5 | Mean requests |
|---|---:|---:|---:|---:|
| Notebook `13` selected-stop hybrid v1 | 43/49 | 0.878 | 0.939 | 6.59 |
| Notebook `19` best fused, lambda `0.00` | 33/49 | 0.673 | 0.878 | 22.37 |
| Notebook `19` closest request budget, lambda `0.02` | 26/49 | 0.531 | 0.837 | 7.98 |
| Notebook `19` lower request budget, lambda `0.05` | 26/49 | 0.531 | 0.776 | 5.65 |

Final-head comparison at the best lambda:

| Head | Accuracy | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| Bayes-only | 0.714 | 0.816 | 0.649 |
| MLP-only | 0.673 | 0.857 | 0.620 |
| Fused | 0.673 | 0.878 | 0.620 |

Bayes-only slightly beats the fused head at lambda `0.00`, but it is still far below Notebook `13`. The fused head is pulled toward the partial-evidence MLP, which becomes highly confident on evidence trajectories that differ from the trajectories it was trained on.

## Paired Comparison Against Notebook 13

At lambda `0.00`, Notebook `19` has:

| Outcome | Cases |
|---|---:|
| Both correct | 32 |
| Notebook `19` only correct | 1 |
| Notebook `13` only correct | 11 |
| Both wrong | 5 |

Notebook `19` only fixed one Notebook `13` failure:

| Case | True pathology | Notebook 13 | Notebook 19 |
|---|---|---|---|
| `test:125508` | Unstable angina | Anemia | Unstable angina |

But it regressed eleven Notebook `13` correct cases, including:

| Case | True pathology | Notebook 13 | Notebook 19 |
|---|---|---|---|
| `test:33118` | Acute otitis media | Acute otitis media | Viral pharyngitis |
| `test:88250` | Allergic sinusitis | Allergic sinusitis | Viral pharyngitis |
| `test:11342` | Atrial fibrillation | Atrial fibrillation | Myocarditis |
| `test:11198` | Boerhaave | Boerhaave | Epiglottitis |
| `test:90978` | Bronchiolitis | Bronchiolitis | Croup |
| `test:51421` | Chagas | Chagas | Croup |
| `test:46152` | Cluster headache | Cluster headache | Acute otitis media |
| `test:35039` | Myocarditis | Myocarditis | Pericarditis |
| `test:105129` | Pulmonary neoplasm | Pulmonary neoplasm | Pneumonia |
| `test:36032` | Sarcoidosis | Sarcoidosis | Epiglottitis |
| `test:749` | Spontaneous pneumothorax | Spontaneous pneumothorax | Pericarditis |

At lambda `0.02` and above, there are no fixes over Notebook `13`; there are only regressions.

## Failure Analysis

The result is not a close miss. It points to a real design problem.

### 1. More evidence did not help because the trajectory was wrong

Lambda `0.00` is the most permissive setting. It asks `22.37` evidence fields on average and hits the `24` request cap in `28/49` cases. Despite this, accuracy is only `33/49`.

That means the issue is not simply stopping too early. The VOI controller is often asking many fields and still converging to the wrong diagnosis.

### 2. The partial-evidence MLP is out-of-distribution under VOI-selected trajectories

The partial-evidence MLP was trained on sequential trace-style partial states. Notebook `19` creates a different evidence distribution. The top requested roots are heavily dominated by generic/high-MI fields:

| Root | Question | Count |
|---|---|---:|
| `E_155` | palpitations | 98 |
| `E_201` | cough | 97 |
| `E_50` | sweating | 96 |
| `E_89` | fatigue / non-restful sleep | 88 |
| `E_124` | asthma / bronchodilator history | 81 |
| `E_70` | overweight | 80 |
| `E_125` | GERD history | 77 |

Those fields have global information value, but they do not necessarily form clinically targeted trajectories for each case. The MLP then becomes extremely confident on partial states it was not optimized for.

Example pattern:

- wrong cases often end with fused confidence above `0.95`
- many wrong cases have MLP confidence near `1.0`
- Bayesian confidence is often much lower, showing the fusion is dominated by the MLP head

### 3. Bayesian independence assumptions are too weak for final diagnosis

Bayes-only at lambda `0.00` reaches `0.714` accuracy, better than fused/MLP but still far below Notebook `13`. The naive likelihood model is useful for evidence valuation and audit, but not strong enough as a standalone diagnostic controller.

Likely reasons:

- DDXPlus evidence fields are not conditionally independent
- parent/child evidence relationships are only partly captured
- categorical/multi-choice outcomes are compressed into root-level states
- likelihood smoothing and clipping reduce extreme rare-disease evidence
- myopic VOI optimizes local entropy reduction, not multi-turn diagnostic recovery

### 4. The stop certificate is not the main bottleneck

At lambda `0.00`, many cases hit the cap. At higher lambdas, the system stops earlier but accuracy collapses further. This implies the main bottleneck is question selection and belief updating, not just the stop threshold.

### 5. Notebook 13 remains better because it preserves clinical/LLM trajectory judgment

Notebook `13` asks fewer fields than lambda `0.00` and gets much higher accuracy. Its advantage seems to come from the LLM-led trajectory and MLP-guided stopping, not from a pure information-theoretic score.

The Bayesian VOI score is mathematically clean, but it over-prioritizes generic posterior entropy reduction and under-prioritizes clinically decisive evidence for the active case.

## Promotion Decision

Notebook `19` should not be promoted.

Promotion rule:

- `>=43/49` correct with mean requests `<=6.59`
- or `>=44/49` correct with mean requests `<=9.0`
- or same accuracy as Notebook `13` while fixing at least two persistent hard cases without introducing more than one new failure

Observed best:

- `33/49` correct at `22.37` mean requests
- only one fix over Notebook `13`
- eleven regressions against Notebook `13`

Decision:

```text
do_not_promote_yet
```

## Promotion Rule

Notebook `19` should only become the next live candidate if one of these holds on the full 49-case run:

- `>=43/49` correct with mean requests `<=6.59`
- `>=44/49` correct with mean requests `<=9.0`
- same accuracy as Notebook `13` while fixing at least two persistent hard cases without introducing more than one new failure
- strong offline evidence that VOI-ranked alternatives would have prevented multiple Notebook `13` failures

None of these hold. Notebook `13` remains the frozen proposed method and Notebook `19` becomes a mathematical ablation explaining what Bayesian VOI can and cannot recover.

## Interpretation

Notebook `19` is the most rigorous algorithmic ablation so far because it directly models uncertainty and evidence value instead of only ranking evidence roots by graph statistics. The negative result is useful because it narrows the research direction.

The key question was:

> Can a posterior-level VOI ledger select a smaller or better evidence subset than Notebook `13`, or at least explain the persistent hard-case failures?

The answer in this version is no. Pure offline Bayesian VOI does not beat the current hybrid method.

The scientific takeaway is:

- Bayesian/VOI scoring is valuable as an audit signal.
- It is not ready to replace the LLM-led evidence-acquisition policy.
- The current partial-evidence MLP should not be trusted blindly on arbitrary VOI-generated evidence subsets.
- A future algorithmic version should either train the MLP on VOI-generated partial states, use VOI only as an advisory feature inside the Notebook `13` shortlist, or add stronger calibration/contradiction gating before fusion.

Recommended next step:

- do not make a live Notebook `20` from this version
- keep Notebook `13` as the frozen proposed method
- use Notebook `19` as evidence that naive Bayes/VOI replacement is insufficient
- if continuing algorithmic work, focus on calibrated advisory VOI or trajectory-aware evidence acquisition, not a full replacement controller
