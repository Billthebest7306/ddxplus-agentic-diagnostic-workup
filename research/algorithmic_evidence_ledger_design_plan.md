# Algorithmic Evidence Ledger Design Plan

Purpose: define a concrete next implementation path for the algorithmic evidence ledger phase. This is the engineering plan that follows from `algorithmic_evidence_ledger_research.md`.

## 1. Recommended Implementation Path

Build the algorithmic ledger in two phases:

1. **Notebook 16: offline ledger diagnostics and policy simulation**
   - no API calls
   - reconstruct Notebook `13` 49-case traces
   - compute new ledger signals
   - test whether the ledger would have caught current failures
   - tune deterministic signal thresholds without spending tokens

2. **Notebook 17: live algorithmic-ledger confirmation**
   - run the frozen improved ledger policy on the same 49-case balanced slice
   - compare directly against Notebook `13`
   - only run after Notebook `16` shows meaningful offline evidence

This is the safest path. The algorithmic ledger is complex enough that live API testing before offline validation would waste money.

## 2. What Notebook 16 Should Do

Notebook `16` should be named:

```text
notebooks/16_algorithmic_evidence_ledger_offline_analysis.ipynb
```

Suggested artifact root:

```text
artifacts/algorithmic_evidence_ledger/offline_notebook13_49case_v1/
```

Inputs:

- Notebook `13` 49-case traces:
  - `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/traces.jsonl`
- Notebook `13` predictions:
  - `artifacts/sequential_hybrid_mlp_feedback/selected_stop_live_confirmation_49case_v1/predictions.csv`
- DDXPlus train/test data and metadata:
  - patient splits
  - `release_evidences.json`
  - `release_conditions.json`
- partial-evidence MLP from Notebook `10`

Outputs:

- `turn_level_algorithmic_ledger.csv`
- `evidence_contributions.csv`
- `unresolved_pairs.csv`
- `candidate_action_scores.csv`
- `case_safety_flags.csv`
- `hard_case_algorithmic_audit.json`
- `ledger_threshold_sweep.csv`
- `selected_ledger_policy.json`
- `figures/`
- `resolved_run_config.json`

## 3. Ledger Data Model

The ledger should be represented through explicit records. These can be dataclasses inside the notebook.

### 3.1 EvidenceObservation

```python
@dataclass
class EvidenceObservation:
    turn_index: int
    root_id: str
    question_text: str
    status: str              # present, absent, unknown
    values: list[str]
    decoded_values: list[str]
    source: str              # initial_evidence or request
```

Purpose:

- preserve the factual state of the patient episode
- keep raw IDs and decoded clinical text together

### 3.2 BeliefSnapshot

```python
@dataclass
class BeliefSnapshot:
    turn_index: int
    mlp_top1: str
    mlp_top5: list[str]
    mlp_probs_top5: list[float]
    mlp_confidence: float
    mlp_margin: float
    mlp_entropy: float
    llm_top1: str
    llm_top5: list[str]
    llm_confidence: float
    agreement: bool
    top1_changed: bool
    top5_jaccard: float
    kl_from_prev: float
```

Purpose:

- convert raw model outputs into stable diagnosis-state features
- detect drift and disagreement

### 3.3 EvidenceContribution

```python
@dataclass
class EvidenceContribution:
    case_id: str
    turn_index: int
    root_id: str
    diagnosis: str
    competitor: str
    log_likelihood_ratio: float
    contribution_type: str   # supports_top, contradicts_top, neutral
```

Purpose:

- make support and contradiction explicit
- show why a diagnosis is or is not supported by revealed evidence

### 3.4 CandidateActionScore

```python
@dataclass
class CandidateActionScore:
    case_id: str
    turn_index: int
    root_id: str
    question_text: str
    legal: bool
    score: float
    entropy_gain: float
    pair_gap: float
    contradiction_resolution: float
    split_balance: float
    generic_penalty: float
    redundancy_penalty: float
    reason: str
```

Purpose:

- produce an auditable ranked action frontier
- let the LLM choose from a high-value shortlist rather than the full action space

### 3.5 StopCertificate

```python
@dataclass
class StopCertificate:
    case_id: str
    turn_index: int
    stop_allowed: bool
    confidence_ok: bool
    margin_ok: bool
    entropy_ok: bool
    stability_ok: bool
    agreement_ok: bool
    no_active_contradiction: bool
    no_high_value_action: bool
    unresolved_pair_count: int
    max_action_value: float
    flags: list[str]
```

Purpose:

- make stopping explainable
- prevent premature stop when the ledger knows important uncertainty remains

## 4. Train-Derived Statistics

Notebook `16` should precompute only from the DDXPlus training split:

```text
P(root present | pathology)
P(root absent | pathology)
P(root value=v | pathology)
P(pathology)
global P(root present)
global root frequency
```

Important fairness boundary:

- train-derived disease/evidence statistics are allowed
- hidden test labels are not used during live policy
- hidden test evidence is only used by the environment when the agent requests a root
- full-evidence predictions are not used inside the policy

These statistics are not a replacement classifier. They are ledger-side diagnostics for evidence value and contradiction.

## 5. Core Algorithms

### 5.1 Normalize Evidence Outcomes

Every root should have an observable outcome state:

```text
binary root:
  present or absent

categorical root:
  value id if present, absent otherwise

multi-choice root:
  set of values if present, absent otherwise
```

This lets the ledger compute train-derived rates consistently.

### 5.2 Evidence Contribution Score

For top diagnosis `d_top`, competitor `d_cmp`, and revealed outcome `o`:

```text
llr(o, d_top, d_cmp) =
  log((P(o | d_top) + eps) / (P(o | d_cmp) + eps))
```

Categorize:

```text
if llr >= +tau_support:
    supports_top
elif llr <= -tau_contradiction:
    contradicts_top
else:
    neutral
```

Initial thresholds:

```text
tau_support = 0.75
tau_contradiction = 0.75
eps = 1e-4
```

These should be tuned offline in Notebook `16`.

### 5.3 Unresolved Pair Detection

For each pair among the current MLP top-5 diagnoses:

```text
pair_mass = p(d_i) + p(d_j)
pair_margin = abs(p(d_i) - p(d_j))
pair_evidence_sum = abs(sum revealed llr over pair)
```

Flag unresolved if:

```text
pair_mass >= 0.15
pair_margin <= 0.25
pair_evidence_sum <= 1.0
```

Meaning:

- the two diagnoses still matter probabilistically
- the model has not clearly separated them
- the observed evidence has not strongly supported one over the other

### 5.4 Drift Detection

Flag drift if one of these holds:

```text
top1_changed and latest_evidence_value_score < low_value_threshold
kl_from_prev >= kl_threshold
margin_decreased_by >= margin_drop_threshold
entropy_increased_by >= entropy_rise_threshold
```

Initial thresholds:

```text
low_value_threshold = 0.05
kl_threshold = 0.35
margin_drop_threshold = 0.15
entropy_rise_threshold = 0.10
```

Purpose:

- catch cases like Croup or COPD where extra evidence may push the system toward a wrong diagnosis
- distinguish useful belief revision from unstable overreaction

### 5.5 Candidate Action Value

For each legal unrevealed root `r`, compute:

```text
action_value(r) =
  0.30 * entropy_gain(r)
  + 0.25 * unresolved_pair_gap(r)
  + 0.20 * contradiction_resolution(r)
  + 0.15 * split_balance(r)
  + 0.10 * top1_vs_competitor_gap(r)
  - 0.15 * generic_penalty(r)
  - 0.10 * redundancy_penalty(r)
```

Components:

- `entropy_gain`: expected MLP entropy reduction from counterfactual reveals
- `unresolved_pair_gap`: how much this root separates top unresolved pairs
- `contradiction_resolution`: whether this root can test a contradiction against current top1
- `split_balance`: whether the root is likely to produce informative present/absent variation
- `top1_vs_competitor_gap`: separation between current top1 and next most likely diagnosis
- `generic_penalty`: penalty for broad, frequently asked, weakly discriminative roots
- `redundancy_penalty`: penalty for evidence similar to already revealed roots

This action value should generate the shortlist, not decide the final answer by itself.

### 5.6 Stop Certificate

Notebook `13` stop rule:

```text
mlp_confidence >= 0.70
mlp_margin >= 0.20
mlp_entropy <= 0.10
min_requests >= 1
```

Algorithmic ledger stop rule should extend this:

```text
base_mlp_stop_ok
AND stability_ok
AND no_active_contradiction
AND unresolved_pair_count <= unresolved_pair_limit
AND max_action_value <= action_value_stop_threshold
```

Initial thresholds:

```text
unresolved_pair_limit = 0
action_value_stop_threshold = 0.12
stability_window = 2
```

The key change:

- confidence alone is not enough
- the ledger must certify that there is no obvious unresolved high-value question remaining

## 6. How This Changes The Live Prompt

Notebook `17` should give the LLM a compact ledger card, not a long dump.

Example:

```text
Current diagnostic belief:
- MLP top diagnoses: Croup 0.41, Acute bronchitis 0.22, Bronchospasm 0.14
- LLM top diagnoses: Croup, Acute bronchitis, URTI
- Agreement: yes
- Entropy: 0.18, margin: 0.19

Ledger warnings:
- Stop not certified: unresolved pair Croup vs Acute bronchitis.
- Highest-value unresolved discriminator: stridor / barky cough / fever pattern.
- Do not ask broad generic questions unless no discriminator is available.

Allowed high-value evidence requests:
1. E_xxx: [question text] | reason: separates Croup vs Acute bronchitis
2. E_yyy: [question text] | reason: tests contradiction against current top diagnosis
...
```

The LLM still chooses the final request, but the action space is strongly structured.

## 7. Offline Evaluation Questions

Notebook `16` should answer these before any live run:

1. Did the ledger flag Notebook `13` incorrect cases as unsafe before stop?
2. How many correct cases would it unnecessarily continue?
3. Does `max_action_value` separate correct stops from wrong stops?
4. Do unresolved-pair counts predict errors?
5. Do contradiction flags predict drift or wrong final answers?
6. Which evidence fields are consistently high-value but under-requested?
7. Which generic fields are over-requested?
8. Can an offline stop certificate improve the 49-case trace result without increasing mean requests too much?

## 8. Proposed Notebook 16 Acceptance Criteria

Notebook `16` should be considered successful if it produces at least one of:

- a stop certificate that preserves `43/49` while reducing mean requests
- a stop certificate that identifies at least `4/6` final errors as unsafe before stop
- an action-value analysis showing clear missed discriminators in hard cases
- a contradiction/drift analysis explaining why current hard cases failed

It does **not** need to improve live accuracy yet. It needs to prove the ledger signals are meaningful.

## 9. Proposed Notebook 17 Acceptance Criteria

Only run Notebook `17` after Notebook `16`.

Notebook `17` should compare directly against Notebook `13` on the same 49 cases.

Promotion rule:

- promote algorithmic ledger if `accuracy >= 43/49` and `mean_requests <= 6.59`, or
- promote if `accuracy >= 44/49` and `mean_requests <= 8.0`, or
- promote if accuracy is tied but hard-case trace quality is clearly better and top-5 improves

Reject as main method if:

- accuracy drops below `42/49`
- requests increase without fixing hard cases
- LLM ignores ledger warnings
- action shortlist becomes clinically generic again

## 10. Why This Is Research-Defensible

This plan is not just engineering polish.

It tests a concrete hypothesis:

> A deterministic algorithmic ledger can improve interactive diagnosis by tracking contradiction, unresolved differential pairs, and expected information value, rather than relying on LLM confidence or raw MLP confidence alone.

It also connects directly to prior work:

- active feature acquisition: choose the next feature under cost
- AARLC: use classifier entropy to align inquiry and diagnosis
- MediQ/ALFA: LLM question-asking needs structured guidance
- MEDDxAgent: interactive DDx is an established task, but structured evidence-ledger control is our angle
- TriMediQ: structured representations help multi-turn medical reasoning

## 11. Immediate Next Step

Build Notebook `16`.

Do not run a new live API experiment yet.

Start with these sections:

1. goal and hypothesis
2. load DDXPlus metadata, train split, Notebook `13` traces
3. reconstruct turn states
4. train-derived evidence/pathology statistics
5. MLP belief replay
6. support/contradiction ledger
7. unresolved-pair and drift detection
8. candidate action-value scoring
9. stop-certificate simulation
10. hard-case audits
11. recommendation for Notebook `17`

