# Graph Algorithmic Evidence Ledger Design

Last updated: 2026-05-06

Purpose: define a concrete, research-grounded graph algorithm for the next DDXPlus project phase. This replaces the vague idea of "an algorithmic ledger" with a specific graph-structured method that can be implemented and evaluated.

Recommended method name:

> **Graph Evidence Ledger Controller** (`GEL-C`)

Alternative paper-style name:

> **Evidence-Gated Graph Ledger for DDXPlus Workup**

## 1. Design Goal

The graph ledger should improve the current Notebook `13` hybrid v1 system by addressing the actual bottleneck found in Notebook `15`:

- not simply too few requests
- wrong evidence trajectories
- stable wrong beliefs
- diagnostic drift
- unresolved competing diagnoses
- weak contradiction handling

Current Notebook `13`:

```text
LLM chooses evidence
MLP estimates partial-evidence diagnosis
MLP confidence/margin/entropy decide stopping
```

Graph ledger version:

```text
LLM chooses from graph-ranked evidence shortlist
MLP estimates partial-evidence diagnosis
graph computes support, contradiction, unresolved pairs, and information value
stop requires a graph certificate, not only MLP confidence
```

## 2. Fairness Boundaries

The graph ledger must not leak hidden test labels or full test evidence.

Allowed:

- DDXPlus train split statistics
- DDXPlus evidence metadata for decoding and legal root/value structure
- partial evidence revealed so far in the current case
- MLP belief from revealed evidence only
- LLM ranked differential from revealed evidence only

Use with caution:

- `release_conditions.json` contains pathology-level symptom/antecedent metadata. It may encode generator knowledge. It can be used for explanation or oracle/domain-knowledge ablation, but the main live policy should rely on train-derived statistics to avoid a leakage criticism.

Forbidden inside live policy:

- current test case `PATHOLOGY`
- current test case `DIFFERENTIAL_DIAGNOSIS`
- unrevealed evidence values
- full-evidence one-shot predictions
- test-label-derived evidence/pathology rates

## 3. Graph Types

GEL-C uses two graphs:

1. **Global DDXPlus Evidence Graph**
   - built once from the training split
   - encodes statistical relationships between evidence outcomes and pathologies

2. **Patient Episode Graph**
   - built dynamically during each sequential workup
   - contains only visible evidence and current diagnostic beliefs

## 4. Global DDXPlus Evidence Graph

### 4.1 Node Types

| Node Type | Example | Meaning |
|---|---|---|
| `Disease` | `PATHOLOGY:Croup` | One of the 49 DDXPlus pathologies |
| `RootEvidence` | `ROOT:E_201` | One requestable evidence question |
| `Outcome` | `OUTCOME:E_201:present` | A concrete observation outcome |
| `Outcome` | `OUTCOME:E_54:V_161` | A categorical/multi-choice value outcome |
| `Outcome` | `OUTCOME:E_201:absent` | Explicit absence after request |
| `Demographic` | `AGE_BIN:child`, `SEX:F` | Demographic priors |

### 4.2 Edge Types

| Edge Type | Direction | Weight | Meaning |
|---|---|---|---|
| `HAS_OUTCOME` | `RootEvidence -> Outcome` | `1` | Root can produce this outcome |
| `ASSOCIATED_WITH` | `Outcome -> Disease` | signed log-odds | Outcome supports or contradicts disease |
| `MUTUAL_INFO` | `RootEvidence -> Disease` | mutual information | Root is diagnostically informative for disease |
| `GLOBAL_FREQ` | `RootEvidence -> RootEvidence` metadata | scalar | Used for generic-question penalty |
| `DEMOGRAPHIC_PRIOR` | `Demographic -> Disease` | log-odds | Age/sex support signal |

### 4.3 Train-Derived Statistics

From training split only, compute:

```text
P(outcome | disease)
P(outcome | not disease)
P(outcome)
P(disease)
P(outcome | active disease subset)
```

Use Laplace smoothing:

```text
P_hat(outcome | disease) =
  (count(outcome, disease) + alpha) / (count(disease) + alpha * num_outcomes_for_root)
```

Suggested:

```text
alpha = 1.0
eps = 1e-6
```

Outcome support weight:

```text
w(outcome, disease) =
  log((P(outcome | disease) + eps) / (P(outcome | not disease) + eps))
```

Interpretation:

- positive weight: outcome supports the disease
- negative weight: outcome contradicts or argues against disease
- near-zero weight: outcome is not discriminative

Root-level mutual information:

```text
MI(root, disease) =
  sum_o sum_y P(o, y) log(P(o, y) / (P(o) P(y)))
```

Use MI mainly to identify globally useful roots, but do not rely only on global MI because the best next question is patient-specific.

## 5. Patient Episode Graph

At turn `t`, construct:

```text
Case node
  -> visible outcome nodes
  -> current MLP belief node
  -> current LLM differential node
  -> active disease nodes
  -> legal unrevealed root evidence nodes
  -> unresolved disease-pair nodes
```

### 5.1 Dynamic Node Types

| Node Type | Meaning |
|---|---|
| `Case` | Current patient episode |
| `ObservedOutcome` | Evidence outcome already visible |
| `CandidateRoot` | Legal unrevealed evidence request |
| `ActiveDisease` | Union of MLP top-k and LLM top-k diagnoses |
| `DiseasePair` | Pair of active diseases that still compete |
| `StopCertificate` | Boolean and reasons for stop eligibility |

### 5.2 Dynamic Edge Types

| Edge Type | Direction | Meaning |
|---|---|---|
| `OBSERVED` | `Case -> ObservedOutcome` | Ledger-visible evidence |
| `SUPPORTS` | `ObservedOutcome -> ActiveDisease` | Positive support weight |
| `CONTRADICTS` | `ObservedOutcome -> ActiveDisease` | Negative support weight |
| `COMPETES_WITH` | `ActiveDisease -> ActiveDisease` | Belief-level competition |
| `CAN_RESOLVE` | `CandidateRoot -> DiseasePair` | Candidate root can separate the pair |
| `NEXT_ACTION_VALUE` | `Case -> CandidateRoot` | Final candidate action score |

## 6. Belief State

Start with the partial-evidence MLP because it is trained on DDXPlus.

Let:

```text
p_mlp(d) = MLP probability for disease d
rank_llm(d) = reciprocal-rank score from LLM ranked differential
```

Combined active belief:

```text
b(d) = normalize(0.75 * p_mlp(d) + 0.25 * rank_llm(d))
```

If LLM ranking is missing or malformed:

```text
b(d) = p_mlp(d)
```

Use only top-k active diseases for graph action scoring:

```text
ACTIVE_K = 5
```

Reason:

- keeps graph local to the current diagnostic uncertainty
- avoids asking generic evidence for all 49 pathologies
- aligns with differential diagnosis practice

## 7. Graph Support And Contradiction Scores

For each active disease `d`, compute:

```text
support(d) =
  sum observed outcomes o max(0, w(o, d))

contradiction(d) =
  sum observed outcomes o max(0, -w(o, d))

net_support(d) =
  support(d) - contradiction(d)
```

Graph support margin:

```text
graph_margin =
  net_support(top_disease) - max_{d != top_disease} net_support(d)
```

Active contradiction flag:

```text
active_contradiction =
  contradiction(top_disease) >= contradiction_threshold
  AND graph_margin <= graph_margin_min
```

Initial thresholds for offline tuning:

```text
contradiction_threshold = 2.0
graph_margin_min = 0.5
```

These scores create an evidence audit:

```text
top diagnosis: Croup
supporting evidence: barky cough, age, fever
contradicting evidence: absent stridor, no respiratory distress
unresolved competitor: acute bronchitis
```

## 8. Disease-Pair Graph

Create pair nodes for active diseases:

```text
PAIR(d_i, d_j)
```

Pair priority:

```text
pair_priority(i, j) =
  b(d_i) * b(d_j) / (abs(b(d_i) - b(d_j)) + eps)
```

This gives high priority to pairs that:

- both have meaningful probability
- are close in probability

Pair evidence separation from observed evidence:

```text
observed_pair_separation(i, j) =
  abs(sum_o [w(o, d_i) - w(o, d_j)])
```

Unresolved pair:

```text
pair_priority >= pair_priority_threshold
AND observed_pair_separation <= separation_threshold
```

Suggested starting thresholds:

```text
pair_priority_threshold = 0.02
separation_threshold = 1.0
```

This is the graph version of "the current differential has not actually been separated yet."

## 9. Candidate Evidence Action Scoring

For each legal unrevealed root `r`, score how useful it would be to ask next.

### 9.1 Outcome Distribution Under Current Belief

For root `r`, possible outcomes are:

```text
O_r = {absent, present/value_1, present/value_2, ...}
```

Expected outcome probability:

```text
P(o | b) = sum_d b(d) * P(o | d)
```

### 9.2 Information Gain

Current entropy:

```text
H(b) = -sum_d b(d) log b(d)
```

Posterior if outcome `o` is observed:

```text
b_o(d) = normalize(b(d) * P(o | d))
```

Expected posterior entropy:

```text
E_H_after(r) = sum_o P(o | b) * H(b_o)
```

Information gain:

```text
IG(r) = max(0, H(b) - E_H_after(r))
```

This is the MedKGI-inspired component, adapted to DDXPlus train-derived outcome statistics.

### 9.3 Pair-Separation Score

For each unresolved pair `(i, j)`, compute outcome-distribution distance:

```text
TV_r(i, j) =
  0.5 * sum_o abs(P(o | d_i) - P(o | d_j))
```

Pair score:

```text
pair_separation(r) =
  sum unresolved pairs pair_priority(i, j) * TV_r(i, j)
```

This asks:

> Which legal evidence root most separates the diseases that are currently competing?

### 9.4 Contradiction-Probe Score

If the current top disease has high contradiction or low graph margin, prioritize roots that can test it against the strongest competitor.

```text
contradiction_probe(r) =
  max_competitor TV_r(top_disease, competitor)
  * active_contradiction_indicator
```

### 9.5 Graph Centrality Score

Build a local bipartite graph:

```text
CandidateRoot -> DiseasePair
```

Edge weight:

```text
edge(r, pair) = pair_priority(pair) * TV_r(pair)
```

Centrality:

```text
centrality(r) = weighted_degree(r) / max_weighted_degree
```

Use weighted degree first. PageRank or betweenness can be added later, but they are not necessary for v1.

### 9.6 Redundancy Penalty

Represent each root by its vector of disease associations:

```text
v_r = [P(root present | d_1), ..., P(root present | d_49)]
```

For already revealed roots `R_seen`:

```text
redundancy(r) = max cosine_similarity(v_r, v_seen)
```

Penalty:

```text
redundancy_penalty = max(0, redundancy - 0.75)
```

Purpose:

- avoid repeatedly asking near-duplicate evidence dimensions
- reduce generic broad-system questioning

### 9.7 Generic Penalty

Generic roots are globally frequent but weakly discriminative.

```text
generic_penalty(r) =
  global_present_rate(r) * (1 - normalized_MI(r))
```

This discourages roots that are common across many diagnoses without separating the current top-k.

### 9.8 Final Action Score

Recommended v1 score:

```text
action_score(r) =
  0.30 * normalize(IG(r))
  + 0.25 * normalize(pair_separation(r))
  + 0.15 * normalize(contradiction_probe(r))
  + 0.15 * normalize(centrality(r))
  + 0.10 * normalize(split_balance(r))
  - 0.10 * redundancy_penalty(r)
  - 0.10 * generic_penalty(r)
```

Split balance:

```text
split_balance(r) = 1 - 2 * abs(P(present | b) - 0.5)
```

Interpretation:

- `IG`: expected entropy reduction
- `pair_separation`: separates current differential
- `contradiction_probe`: tests suspected wrong top diagnosis
- `centrality`: useful across multiple unresolved pairs
- `split_balance`: likely to produce non-trivial information
- penalties: avoid redundant/generic evidence

Sort deterministically:

```text
descending action_score
descending pair_separation
ascending is_child
ascending question_text
ascending root_id
```

## 10. Stop Certificate

Notebook `13` stops mostly from:

```text
mlp_confidence >= 0.70
mlp_margin >= 0.20
mlp_entropy <= 0.10
min_requests >= 1
```

GEL-C should extend this:

```text
stop_allowed =
  base_mlp_stop
  AND graph_support_ok
  AND contradiction_ok
  AND unresolved_pairs_ok
  AND no_high_value_action_remaining
  AND drift_ok
```

Concrete v1:

```text
base_mlp_stop =
  mlp_confidence >= 0.70
  AND mlp_margin >= 0.20
  AND mlp_entropy <= 0.10
  AND num_requests >= 1

graph_support_ok =
  graph_margin >= 0.50
  OR mlp_confidence >= 0.85

contradiction_ok =
  contradiction(top_disease) < 2.0
  OR graph_margin >= 1.0

unresolved_pairs_ok =
  unresolved_pair_count == 0
  OR max_pair_priority < 0.02

no_high_value_action_remaining =
  max_action_score < 0.15

drift_ok =
  not unsupported_top1_switch_in_last_2_turns
```

Stop output should be a certificate:

```json
{
  "stop_allowed": false,
  "failed_checks": [
    "unresolved_pair:Croup_vs_Acute_bronchitis",
    "high_value_action_remaining:E_XXX"
  ],
  "recommended_next_actions": [
    {
      "root_id": "E_XXX",
      "reason": "highest pair-separation for Croup vs Acute bronchitis"
    }
  ]
}
```

This makes stopping auditable.

## 11. Drift Detection

A major Notebook `15` finding was correct-to-wrong drift.

Define:

```text
top1_switch = current_top1 != previous_top1
delta_margin = current_margin - previous_margin
delta_entropy = current_entropy - previous_entropy
latest_evidence_graph_value = action_score(requested_root at previous turn)
```

Unsupported drift:

```text
top1_switch
AND latest_evidence_graph_value < 0.10
AND current_top1_graph_margin < 0.50
```

Contradictory drift:

```text
top1_switch
AND contradiction(current_top1) > contradiction(previous_top1)
AND graph_margin(current_top1) < graph_margin(previous_top1)
```

Drift response:

- do not stop on the new top diagnosis yet
- ask a contradiction-probe or pair-separating question
- include warning in LLM prompt

## 12. Live Prompt Card

The LLM should not receive the whole graph. It should receive a compact graph card.

Example:

```text
Graph ledger summary:
- Current MLP top diagnoses: Croup 0.47, Acute bronchitis 0.21, Bronchospasm 0.13.
- Current LLM top diagnoses: Croup, Acute bronchitis, URTI.
- Evidence support: Croup +3.1 support, 0.8 contradiction; Acute bronchitis +2.5 support, 0.6 contradiction.
- Unresolved pair: Croup vs Acute bronchitis, separation score 0.42.
- Stop certificate: NOT SAFE. Reason: high-value discriminator remains.

Allowed graph-ranked evidence requests:
1. E_xxx: [question text] | reason: separates Croup vs Acute bronchitis; score 0.71.
2. E_yyy: [question text] | reason: tests contradiction against Croup; score 0.64.
3. E_zzz: [question text] | reason: high information gain; score 0.58.

Instruction:
- Choose one request from the allowed list unless you are ready to stop and the stop certificate is SAFE.
- Do not ask generic questions if a pair-separating question exists.
```

The LLM still controls the final request, but the graph sharply constrains the action space and explains why each action is offered.

## 13. Notebook 16: Offline Graph Ledger Analysis

Notebook name:

```text
notebooks/16_graph_algorithmic_evidence_ledger_offline_analysis.ipynb
```

Artifact root:

```text
artifacts/graph_algorithmic_ledger/offline_notebook13_49case_v1/
```

Primary purpose:

- test whether graph signals explain Notebook `13` failures before spending API money

Inputs:

- Notebook `13` 49-case traces
- Notebook `13` predictions
- DDXPlus training split
- DDXPlus evidence metadata
- partial-evidence MLP

Outputs:

- `global_evidence_graph_edges.csv`
- `turn_level_graph_ledger.csv`
- `case_graph_stop_certificates.csv`
- `candidate_action_scores.csv`
- `unresolved_pair_timeline.csv`
- `evidence_contribution_matrix.csv`
- `hard_case_graph_audits.json`
- `selected_graph_ledger_policy.json`
- figures under `figures/`

### 13.1 Required Analyses

1. **Stop safety analysis**
   - compare graph certificate at final stop for correct vs incorrect cases
   - report unsafe-stop recall and false-positive rate

2. **Action quality analysis**
   - compare actual requested evidence rank under graph scoring
   - identify cases where the LLM chose low graph-value requests

3. **Hard-case graph audit**
   - produce per-case timelines for:
     - Croup
     - Pericarditis
     - COPD exacerbation
     - Influenza
     - Unstable angina
     - Acute rhinosinusitis

4. **Drift analysis**
   - mark correct-to-wrong and wrong-to-correct transitions
   - measure graph support before and after transition

5. **Unresolved-pair analysis**
   - test whether final errors have higher unresolved-pair scores at stop

6. **Visualization**
   - graph support heatmap
   - max action value vs correctness
   - contradiction score vs correctness
   - unresolved-pair count over turns
   - hard-case network diagrams

### 13.2 Notebook 16 Success Criteria

Notebook `16` is successful if any of these are true:

- graph stop certificate flags at least `4/6` Notebook `13` incorrect cases as unsafe
- graph action scoring identifies clear missed high-value evidence in hard cases
- unresolved-pair or contradiction scores separate many wrong stops from correct stops
- graph replay suggests a modified stop certificate can maintain `43/49` with similar requests

Notebook `16` does not need to improve live accuracy yet. It needs to prove the graph signals are meaningful.

## 14. Notebook 17: Live Graph Ledger Confirmation

Only build Notebook `17` after Notebook `16`.

Notebook name:

```text
notebooks/17_live_graph_algorithmic_ledger_confirmation.ipynb
```

Live changes from Notebook `13`:

- same `gpt-4.1-mini`
- same 49-case balanced slice
- same deterministic API settings
- same partial-evidence MLP
- replace v1 shortlist with graph-ranked shortlist
- replace v1 MLP-only stop with graph stop certificate

Promotion rule:

- promote if accuracy `>= 43/49` and mean requests `<= 6.59`
- promote if accuracy `>= 44/49` and mean requests `<= 8.0`
- promote if accuracy ties but error traces are clinically/graphically better and top-5 improves

Reject if:

- accuracy below `42/49`
- requests increase without hard-case improvements
- graph shortlist becomes too generic
- stop certificate blocks too many safe stops

## 15. Why This Is More Algorithmic Than The Current Ledger

Current ledger:

```text
records evidence and applies legality/stopping rules
```

Graph algorithmic ledger:

```text
constructs disease-evidence graph
computes signed support/contradiction paths
builds active differential-pair graph
scores candidate evidence by information gain and pair separation
detects unsupported diagnostic drift
issues stop certificates with failed checks
```

This is genuinely algorithmic because the next action and stop decision arise from graph operations and information-theoretic scoring, not only prompt judgment.

## 16. Best Research Claim If This Works

If GEL-C improves or matches Notebook `13`, our claim becomes:

> A DDXPlus-native graph evidence ledger can make LLM-led diagnostic workup more evidence-efficient and auditable by converting revealed evidence into signed support/contradiction paths over active differential diagnoses, selecting next questions by graph information value, and allowing final diagnosis only when a graph stop certificate is satisfied.

If it does not improve accuracy, the fallback claim remains useful:

> Graph-ledger analysis exposes why a sequential diagnostic agent fails: stable wrong beliefs, unresolved competing pathologies, unsupported drift, and low-value evidence trajectories.

Either result is scientifically meaningful.

