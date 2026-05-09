# Bayesian VOI Algorithmic Ledger Research

Last updated: 2026-05-08

Purpose: define the next algorithmic/mathematical direction after the Notebook `17` and Notebook `18` graph-shortlist ablations. This note focuses on a Bayesian, value-of-information evidence ledger for DDXPlus.

## 1. Executive Position

The next credible algorithmic improvement should be a **Bayesian value-of-information ledger**, not another shortlist heuristic.

Reason:

- Notebook `13` already has a strong local tradeoff: `43/49 = 0.878` accuracy with `6.59` mean requested evidence fields.
- Notebook `17` showed that hard graph top-10 replacement can ask locally high-information questions while harming diagnosis.
- Notebook `18` showed that graph-advisory blending recovers some rare-disease failures, but still does not beat Notebook `13`.
- The remaining failure mode is not simply early stopping or weak graph ranking. The remaining failure mode is **wrong belief recovery**.

The Bayesian VOI ledger directly targets that failure mode:

```text
visible evidence
-> posterior over all 49 diagnoses
-> contradiction/support accounting
-> expected value of each legal evidence field
-> ask if expected value exceeds cost
-> stop only when posterior, MLP, and remaining VOI agree
```

This gives the project a clearer mathematical core:

```text
P(diagnosis | observed evidence)
```

and a principled acquisition objective:

```text
expected posterior improvement - evidence cost
```

## 2. Why This Is Better Than Another Graph Shortlist

Notebook `17` and Notebook `18` used graph statistics mainly as action-ranking features. They did not maintain a complete posterior belief over all diseases. That matters because a graph score can be high even when it is separating the wrong active diseases.

The Bayesian VOI ledger changes the control logic:

- all 49 pathologies remain in the belief state
- every revealed evidence field updates the same posterior
- evidence can support the current top diagnosis or contradict it
- candidate questions are scored by their expected effect on the posterior
- stop decisions depend on remaining expected value, not just confidence

This makes it harder for the system to get trapped in a wrong active differential without noticing.

## 3. Prior Work Reviewed

### 3.1 Bayesian Medical Inquiry And QMR

Source:

- [A Bayesian Approach for Medical Inquiry and Disease Inference in Automated Differential Diagnosis](https://arxiv.org/abs/2110.08393)

Core contribution:

- uses the Quick Medical Reference belief network
- applies Bayesian inference for disease inference
- applies Bayesian experimental design for medical inquiry
- extends inquiry beyond one-step search to multi-step search
- emphasizes interpretability and avoiding costly training

What we should borrow:

- disease inference and inquiry should be part of one Bayesian loop
- question selection can be framed as Bayesian experimental design
- a training-free or train-statistics-based method is defensible in a medical setting
- multi-step lookahead may improve beyond purely myopic VOI

What differs in our project:

- we do not have QMR; we have DDXPlus root evidence and train split statistics
- DDXPlus gives exact requestable evidence fields and exact environment reveals
- we also have a partial-evidence MLP that can be fused with the Bayesian posterior

### 3.2 Efficient Test Selection In Active Diagnosis

Source:

- [Efficient Test Selection in Active Diagnosis via Entropy Approximation](https://arxiv.org/abs/1207.1418)

Core contribution:

- diagnosis can be represented as recovering hidden states from sequential tests
- selecting the best test by expected entropy reduction is natural
- exact computation can be intractable in general Bayesian networks
- greedy information-gain test selection is a practical approximation

What we should borrow:

- use entropy over diseases as the core uncertainty measure
- score candidate evidence fields by expected conditional entropy reduction
- prefer a greedy/myopic approximation first because it is understandable and computationally feasible
- be explicit that exact optimal sequential test selection is hard

What differs in our project:

- DDXPlus has only 49 diseases and 223 root evidence fields, so exact one-step expected entropy is tractable
- we can avoid loopy belief propagation in V1 by using empirical conditional likelihood tables
- later versions can add two-step rollout only for the top few candidates

### 3.3 Cost-Sensitive Feature Acquisition

Source:

- [Cost-sensitive feature acquisition and classification](https://www.sciencedirect.com/science/article/pii/S0031320306004808)

Core contribution:

- classification can include the decision to acquire features at test time
- feature acquisition has costs
- stopping must balance expected classification benefit against sensing/test cost
- POMDP formulations are principled but often computationally difficult
- myopic approximations are practical

What we should borrow:

- request count should be treated as evidence cost
- the max request cap should remain a safety ceiling, not the real decision variable
- stop when expected marginal value no longer justifies more evidence

What differs in our project:

- all DDXPlus evidence fields can start with equal cost in V1
- later versions can assign disease-sensitive costs or discomfort/complexity weights
- the MLP and Bayesian posterior can jointly estimate benefit

### 3.4 Active Feature Acquisition With Learned Policies

Source:

- [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://papers.nips.cc/paper/7411-joint-active-feature-acquisition-and-classification-with-variable-size-set-encoding)

Core contribution:

- test-time feature acquisition can be learned jointly with classification
- the policy decides whether to stop and predict or collect a new feature
- medical diagnosis is a natural application

What we should borrow:

- DDXPlus evidence requests are active feature acquisition
- acquired feature sets need an order-invariant representation
- evidence cost and predictive performance should be evaluated together

What we should not do yet:

- train a full RL acquisition policy
- train a new policy network before exhausting deterministic Bayesian VOI

Reason:

- the course project benefits from an interpretable mathematical method
- deterministic VOI is easier to audit than another black-box policy
- Notebook `13` already provides a strong neural/LLM baseline

### 3.5 AARLC And Classifier Entropy

Sources:

- [Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification](https://arxiv.org/abs/2112.00733)
- [Artificial Intelligence in Medicine version](https://www.sciencedirect.com/science/article/pii/S0933365723002622)

Core contribution:

- separates evidence inquiry from disease classification
- uses disease-distribution entropy to align inquiry and diagnosis
- learns when to stop asking and classify

What we should borrow:

- entropy is a legitimate bridge between the inquiry process and diagnostic classifier
- the MLP posterior is useful not only as a final head, but as a control signal
- our Bayesian posterior can become another explicit control signal

What differs in our project:

- we use an LLM evidence interface and DDXPlus ledger instead of RL as the main live controller
- the Bayesian ledger can provide deterministic VOI without training an inquiry policy

### 3.6 MedKGI

Source:

- [MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring](https://arxiv.org/abs/2512.24181)

Core contribution:

- combines medical knowledge graphs, information-guided inquiry, structured diagnostic records, and iterative hypothesis refinement
- uses information gain to select questions
- uses structured state to maintain coherence

What we learned from our own graph experiments:

- Notebook `16` confirmed graph scores are informative offline
- Notebook `17` showed hard graph top-10 control is too brittle
- Notebook `18` showed advisory graph support can recover rare-disease failures but still does not beat Notebook `13`

Implication:

- the next algorithmic version should keep information-gain thinking, but move from edge-level graph ranking to posterior-level Bayesian VOI
- graph information may remain as a feature, but the posterior should be the central object

### 3.7 MediQ And ALFA

Sources:

- [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](https://arxiv.org/abs/2406.00922)
- [ALFA: Aligning LLMs to Ask Good Questions](https://arxiv.org/abs/2502.14860)

Core contribution:

- LLMs do not automatically ask good follow-up questions under incomplete information
- question quality can be decomposed into attributes like relevance, clarity, and diagnostic usefulness

What we should borrow:

- do not trust the LLM alone for question selection
- expose a structured candidate list with explicit reasons
- use the LLM as a controlled selector/explainer rather than an unconstrained policy

## 4. What We Have To Work With

We already have enough infrastructure for a strong Bayesian VOI implementation.

### 4.1 Dataset Structure

DDXPlus gives:

- 49 pathologies
- 223 requestable root evidence fields
- train/validate/test splits
- patient age and sex
- initial evidence
- full hidden evidence list
- differential diagnosis targets
- metadata describing evidence type and values

This is unusually suitable for Bayesian evidence acquisition because each requestable field is already discrete and structured.

### 4.2 Existing Artifacts

Useful existing artifacts:

- Notebook `01`: initial-evidence BASD-style MLP
- Notebook `07`: full-evidence MLP ceiling
- Notebook `10`: partial-evidence MLP trained on policy-shaped masks
- Notebook `13`: frozen proposed LLM + MLP stop method
- Notebook `15`: stop-threshold and trajectory analysis
- Notebook `16`: train-derived graph/evidence statistics
- Notebook `17`: hard graph shortlist negative result
- Notebook `18`: advisory graph shortlist negative/diagnostic result

Useful traces:

- Notebook `13` 49-case traces
- Notebook `13` 24-case traces
- Notebook `17` and `18` paired hard-case traces

These let us build and test the Bayesian ledger offline before spending any more API calls.

### 4.3 Train-Derived Tables We Can Build

From the train split:

- disease prior:

```text
P(D = d)
```

- root evidence outcome likelihood:

```text
P(R_j = outcome | D = d)
```

- global root outcome likelihood:

```text
P(R_j = outcome)
```

- log-likelihood ratio:

```text
LLR(d, j, outcome) = log P(outcome | d) - log P(outcome | not d)
```

- evidence reliability:

```text
reliability(j) = mutual_information(D, R_j)
```

- rare-specific support:

```text
support(d, j, outcome) = high LLR and low global outcome rate
```

These are all train-only statistics, so they can be used inside a test-time policy without label leakage.

## 5. Proposed Method: Bayesian VOI Evidence Ledger

Working name:

> BVEL: Bayesian Value-of-Information Evidence Ledger

### 5.1 State

At turn `t`, the ledger stores:

```text
observed outcomes O_t = {(root_id, outcome_state)}
unknown legal roots U_t
posterior p_t(d) over all 49 diagnoses
MLP posterior q_t(d)
optional LLM ranked differential r_t(d)
support/contradiction ledger
request history
```

### 5.2 Posterior Update

The simplest V1 update is Naive Bayes with smoothing and calibration:

```text
log p_t(d)
  = log P(D=d)
  + alpha_age_sex * log P(age_bin, sex | d)
  + tau * sum_j w_j * log P(outcome_j | d)
```

Then normalize:

```text
p_t(d) = softmax(log p_t(d))
```

Important safeguards:

- Laplace/Beta smoothing for sparse evidence outcomes
- likelihood clipping to prevent one rare field from dominating
- reliability weights `w_j` based on train-derived mutual information
- calibration temperature `tau` tuned on validation or fixed conservatively
- optional family/group discount to reduce overcounting correlated evidence

Why this matters:

- simple Naive Bayes may overcount correlated symptoms
- these safeguards keep the posterior useful without pretending evidence fields are fully independent

### 5.3 MLP And Bayesian Posterior Fusion

The partial-evidence MLP is already trained and empirically strong. The Bayesian posterior should not replace it blindly.

Use log-linear pooling:

```text
log fused(d)
  = beta_mlp * log q_t(d)
  + beta_bayes * log p_t(d)
  + beta_prior * log prior_one_shot(d)
```

Then:

```text
fused_t(d) = softmax(log fused(d))
```

V1 choices:

- tune `beta_mlp`, `beta_bayes`, and `beta_prior` on validation or Notebook `13` development traces
- default conservative setting:

```text
beta_mlp = 0.60
beta_bayes = 0.30
beta_prior = 0.10
```

Purpose:

- MLP gives calibrated discriminative power
- Bayes gives interpretable support/contradiction and VOI
- prior prevents unstable drift early in the episode

### 5.4 Value Of Information

For each legal unrevealed root `R`, estimate:

```text
VOI(R) = H(fused_t) - E_outcome[H(fused_{t+1} | R = outcome)]
```

where:

```text
P(outcome | current evidence)
  = sum_d P(outcome | d) * fused_t(d)
```

For each possible outcome:

```text
fused_{t+1}
  = posterior after virtually revealing R = outcome
```

Then utility:

```text
utility(R) =
  a * entropy_reduction
+ b * expected_margin_gain
+ c * expected_top1_stability_gain
+ d * contradiction_resolution_gain
+ e * rare_recovery_bonus
- lambda_cost * cost(R)
- redundancy_penalty(R)
```

The next evidence request is:

```text
argmax_R utility(R)
```

Stop if:

```text
max_R utility(R) <= 0
```

or if the stop certificate fires.

### 5.5 Stop Certificate

The stop certificate should be stricter than a pure MLP threshold:

```text
stop if:
  fused_top1_confidence >= c_min
  and fused_margin >= m_min
  and fused_entropy <= h_max
  and max_remaining_VOI <= voi_max
  and contradiction_score(top1) <= contradiction_max
  and MLP/Bayes top-k are not in severe conflict
```

This is stronger than Notebook `13` because it asks:

> Are we confident, and is there no remaining evidence likely to change the answer?

Notebook `13` mostly asks:

> Is the MLP confident enough right now?

### 5.6 Contradiction Ledger

For current top diagnosis `d*`, each observed evidence outcome gets a support score:

```text
support(d*, outcome_j) = log P(outcome_j | d*) - log P(outcome_j | not d*)
```

Contradiction:

```text
contradiction(d*) = sum max(0, -support(d*, outcome_j))
```

Alternative support:

```text
alt_support(d_alt) = sum support(d_alt, outcome_j)
```

If contradiction is high, the system should not stop even if one head is confident.

If the top diagnosis is contradicted, choose evidence that separates:

```text
d_top vs d_alt
```

where `d_alt` is the best alternative with high support.

This directly targets the current failure mode: wrong belief convergence.

## 6. Proposed Experimental Ladder

### Notebook 19: Offline Bayesian VOI Ledger

No API calls.

Purpose:

- determine whether the Bayesian VOI policy is promising before spending money
- test whether it can improve hard-case evidence selection and stop decisions

Main experiments:

1. Build train-derived likelihood tables.
2. Calibrate Bayesian posterior on validate split.
3. Evaluate Bayesian posterior on:
   - initial evidence
   - Notebook `13` acquired evidence
   - full evidence
4. Replay Notebook `13` traces and compute:
   - posterior trajectory
   - VOI-ranked evidence fields per turn
   - contradiction score per turn
   - remaining VOI at stop
5. Run a deterministic Bayesian VOI agent offline:
   - start from initial evidence
   - choose next root by VOI
   - reveal only that root from the DDXPlus environment
   - stop by VOI certificate
   - diagnose by Bayes, MLP, and fused heads

Important fairness point:

- The offline VOI agent may access hidden test evidence only when it requests that root.
- It must never use true pathology, test differential, or unrevealed evidence in scoring.

Outputs:

- `bayes_likelihood_tables/`
- `posterior_calibration.csv`
- `notebook13_replay_bayes_voi.csv`
- `bayesian_voi_agent_predictions.csv`
- `bayesian_voi_turn_traces.jsonl`
- `hard_case_bayes_audits.json`
- plots for posterior entropy, max VOI, contradiction, accuracy/request frontier

Promotion criterion to a live LLM version:

- match or beat Notebook `13` on the 49-case slice, or
- improve at least two persistent hard cases without losing more than one current correct case, or
- show a clear request reduction at matched accuracy

### Notebook 20: Live Bayesian-VOI-Advisory LLM

Only if Notebook `19` is promising.

Purpose:

- combine Bayesian VOI with the existing LLM interface
- avoid letting the LLM wander into low-value evidence

Live flow:

```text
ledger state
-> Bayesian posterior + MLP posterior
-> top VOI candidates with explanation
-> LLM chooses among top candidates or accepts algorithmic recommendation
-> environment reveals requested evidence
-> posterior updates
-> stop certificate checks
```

The LLM is no longer the main controller. It becomes:

- an interpreter of candidate evidence
- an explanation layer
- a tie-breaker among high-value evidence fields

## 7. What Could Beat Notebook 13

The most plausible route is not "ask fewer questions immediately." The most plausible route is:

```text
same or slightly more evidence on hard cases
+ fewer wasted requests on easy cases
+ better belief recovery
```

Specific improvements that could exceed Notebook `13`:

1. Avoid unsafe wrong stops:
   - stop only when remaining VOI is low.

2. Recover wrong active differentials:
   - contradiction score forces discriminative questions.

3. Improve Croup and Pericarditis:
   - ask evidence that separates true disease from the currently attractive wrong diagnoses.

4. Avoid Stable angina drift:
   - posterior contradiction and prior support prevent late evidence from pulling the model into unrelated diseases.

5. Preserve easy-case efficiency:
   - if posterior and MLP agree and VOI is low, stop early.

## 8. Risks And Mitigations

### Risk 1: Naive Bayes overcounts correlated evidence

Mitigation:

- likelihood clipping
- reliability weights
- group discount
- calibration temperature
- compare Bayes-only, MLP-only, and fused posterior

### Risk 2: Rare evidence dominates too much

Mitigation:

- cap per-evidence log likelihood
- require consistency with more than one support signal
- separate rare-support bonus from posterior update

### Risk 3: VOI asks generic high-entropy questions

Mitigation:

- redundancy penalty
- genericness penalty
- active-differential margin gain
- contradiction-resolution bonus

### Risk 4: Offline deterministic policy beats LLM but does not fit project framing

Mitigation:

- present it as the algorithmic ledger core
- use the LLM only as the clinical-language interface/tie-breaker in Notebook `20`
- the project becomes mathematically stronger, not less agentic

### Risk 5: It still does not beat Notebook 13

Interpretation:

- still valuable
- would show Notebook `13` is a strong simple method
- would provide a principled analysis of why posterior/VOI is insufficient
- gives a defensible negative result for the final report

## 9. Novelty Position

Not novel by itself:

- Bayesian medical diagnosis
- value-of-information test selection
- active feature acquisition
- entropy-based stopping
- LLM question asking

Potentially novel and project-specific:

- DDXPlus-native Bayesian VOI ledger over the official 223 root evidence schema
- fusion of train-derived Bayesian posterior with BASD-style partial-evidence MLP posterior
- contradiction ledger for recovering from wrong active differentials
- using Bayesian VOI as a controlled evidence-acquisition layer for an LLM diagnostic workup agent
- direct comparison against our frozen Notebook `13` hybrid LLM/MLP stop system and graph ablations

The clean claim should be:

> We evaluate whether a DDXPlus-native Bayesian value-of-information ledger can improve evidence-efficient diagnostic workup by explicitly modeling posterior belief, remaining information value, and contradiction, instead of relying only on LLM judgment or graph-edge ranking.

## 10. Recommended Next Step

Build Notebook `19`:

```text
notebooks/19_bayesian_voi_algorithmic_ledger_offline.ipynb
```

Scope:

- offline only
- no API calls
- train-derived likelihoods only
- compare against Notebook `13` on the same 49-case slice
- evaluate Bayes-only, MLP-only, and fused final heads
- produce hard-case audits for Croup, Pericarditis, Stable angina, Chagas, and Ebola

Do not make another live notebook until Notebook `19` proves the Bayesian VOI ledger has a real signal.

## 11. Source Links

- [A Bayesian Approach for Medical Inquiry and Disease Inference in Automated Differential Diagnosis](https://arxiv.org/abs/2110.08393)
- [Efficient Test Selection in Active Diagnosis via Entropy Approximation](https://arxiv.org/abs/1207.1418)
- [Cost-sensitive feature acquisition and classification](https://www.sciencedirect.com/science/article/pii/S0031320306004808)
- [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://papers.nips.cc/paper/7411-joint-active-feature-acquisition-and-classification-with-variable-size-set-encoding)
- [Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification](https://arxiv.org/abs/2112.00733)
- [DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://arxiv.org/abs/2205.09148)
- [MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring](https://arxiv.org/abs/2512.24181)
- [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](https://arxiv.org/abs/2406.00922)
- [ALFA: Aligning LLMs to Ask Good Questions](https://arxiv.org/abs/2502.14860)
