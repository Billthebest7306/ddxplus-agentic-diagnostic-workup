# Ledger Implementation Explained

This note explains where the deterministic ledger was implemented, how it works, and why it helped the sequential DDXPlus system become materially stronger.

Primary implementation files:

- [05_single_agent_structured_policy_refinement.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/05_single_agent_structured_policy_refinement.ipynb)
- [06_single_agent_budget_scaling.ipynb](/Users/bilalawan/claw/assignments/baseline_model/notebooks/06_single_agent_budget_scaling.ipynb)

`06` keeps the same ledger and policy logic as `05`. It only changes the default request budgets.

## 1. Where The Ledger Lives

The ledger is implemented in notebook 05 under:

- `## Deterministic Evidence Ledger And Episode State Manager`

The key pieces are:

- `LedgerEntry`
- `DiagnosisSnapshot`
- `LedgerEpisodeState`
- `DeterministicEvidenceLedger`

These define the state that persists across turns.

## 2. Core Ledger Data Structures

The sequential system needed a structured source of truth for each case episode. That was done with three dataclasses:

```python
@dataclass
class LedgerEntry:
    turn_index: int
    root_evidence_id: str
    question_en: str
    source: str
    status: str
    values: list[str] = field(default_factory=list)
    decoded_values: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class DiagnosisSnapshot:
    turn_index: int
    predicted_pathology: str
    ranked_differential: list[str]
    confidence: float
    decision: str
    requested_evidence_id: str | None
    brief_reasoning: str
    stop_signal_level: str
    top_shortlist_score: float
    deterministic_top_pathology: str | None = None
    deterministic_margin: float = 0.0
    policy_flags: list[str] = field(default_factory=list)


@dataclass
class LedgerEpisodeState:
    case_id: str
    split_name: str
    source_row_index: int
    age: int
    sex: str
    pathology: str
    initial_evidence: str
    evidence_by_root: dict[str, list[str]]
    revealed_roots: set[str]
    evidence_ledger: list[LedgerEntry]
    request_history: list[dict[str, Any]]
    diagnosis_history: list[DiagnosisSnapshot]
    shortlist_history: list[list[dict[str, Any]]]
    prior_differential: list[tuple[str, float]]
```

What this achieves:

- `LedgerEntry` stores every visible clinical finding in a decoded human-readable form.
- `DiagnosisSnapshot` stores the agent's turn-by-turn diagnosis state.
- `LedgerEpisodeState` holds the full case state across the sequential workup.

This is what made the notebook truly stateful instead of just “prompting over a growing text blob.”

## 3. Compiling A DDXPlus Row Into A Case Episode

Each DDXPlus patient row is compiled into a reusable episode state using `from_row(...)`.

```python
def from_row(
    self,
    row: dict[str, Any],
    split_name: str,
    prior_differential: list[tuple[str, float]] | None = None,
) -> LedgerEpisodeState:
    evidences_list = [str(token) for token in safe_parse_list(row["EVIDENCES"])]
    initial_evidence = str(row["INITIAL_EVIDENCE"])
    initial_root, _ = parse_evidence_token(initial_evidence)
    evidence_by_root: dict[str, list[str]] = {}
    for token in evidences_list:
        root_id, _ = parse_evidence_token(token)
        evidence_by_root.setdefault(root_id, []).append(token)
    initial_values = self.tokens_to_values(initial_root, evidence_by_root.get(initial_root, [initial_evidence]))
    initial_entry = self._make_entry(0, initial_root, "initial_evidence", "present", initial_values)
    return LedgerEpisodeState(
        case_id=str(row["case_id"]),
        split_name=split_name,
        source_row_index=int(row["source_row_index"]),
        age=int(row["AGE"]),
        sex=str(row["SEX"]),
        pathology=str(row["PATHOLOGY"]),
        initial_evidence=initial_evidence,
        evidence_by_root=evidence_by_root,
        revealed_roots={initial_root},
        evidence_ledger=[initial_entry],
        request_history=[],
        diagnosis_history=[],
        shortlist_history=[],
        prior_differential=list(prior_differential or []),
    )
```

Important design choice:

- the full evidence is stored in `evidence_by_root`
- but only the `INITIAL_EVIDENCE` root is visible at the start

That matches the intended sequential setup: the environment knows the full case, but the agent only sees the initial clue and whatever it explicitly requests later.

## 4. Decoding DDXPlus Tokens Into Readable Findings

One of the earlier major failures was that the model had been reasoning over opaque ids like `E_66` and `V_20`. The ledger fixed that by decoding values into readable English.

```python
def decode_value(self, root_id: str, value: str) -> str:
    if value == "present":
        return "yes"
    value_meaning = self.evidence_metadata[root_id].get("value_meaning", {})
    if isinstance(value_meaning, dict):
        entry = value_meaning.get(str(value))
        if isinstance(entry, dict):
            human = entry.get("en") or entry.get("fr")
            if human:
                return str(human)
        elif entry:
            return str(entry)
    return str(value)


def summarize_observation(self, root_id: str, values: list[str], status: str) -> str:
    question = self.question_text[root_id]
    if status == "absent":
        return f"{question} -> no"
    if self.data_types[root_id] == "B":
        return f"{question} -> yes"
    decoded_values = self.decode_values(root_id, values)
    return f"{question} -> {', '.join(decoded_values) if decoded_values else 'observed'}"
```

This is what turned the visible ledger from unreadable machine state into clinically meaningful context.

Example of the intended effect:

- before: `E_66`
- after: `Are you experiencing shortness of breath or difficulty breathing in a significant way? -> yes`

## 5. Enforcing Legality With Parent-Child Gating

The ledger is also the place where legal actions are enforced.

```python
def parent_is_satisfied(self, root_id: str, episode: LedgerEpisodeState) -> bool:
    parent_root = self.parent_question.get(root_id, root_id)
    if parent_root == root_id:
        return True
    return self.root_present_or_implied_present(parent_root, episode)


def legal_actions(self, episode: LedgerEpisodeState) -> list[dict[str, Any]]:
    available = []
    for root_id in self.root_ids:
        if root_id in episode.revealed_roots:
            continue
        if not self.parent_is_satisfied(root_id, episode):
            continue
        parent_root = self.parent_question.get(root_id, root_id)
        available.append(
            {
                "root_evidence_id": root_id,
                "question_en": self.question_text[root_id],
                "parent_root_id": parent_root,
                "is_child": bool(parent_root != root_id),
            }
        )
    return available
```

What this does:

- you cannot ask the same root twice
- follow-up questions only become legal if the parent finding is present or implied present

This is why the sequential environment became clinically more sensible. It stopped exposing irrelevant child questions too early.

## 6. Updating The Ledger After A Request

When the agent asks for one piece of evidence, the ledger updates the visible state and writes a structured record of what was revealed.

```python
def reveal(self, episode: LedgerEpisodeState, root_evidence_id: str, turn_index: int) -> dict[str, Any]:
    episode.revealed_roots.add(root_evidence_id)
    revealed_tokens = list(episode.evidence_by_root.get(root_evidence_id, []))
    values = self.tokens_to_values(root_evidence_id, revealed_tokens)
    status = "present" if revealed_tokens else "absent"
    entry = self._make_entry(turn_index, root_evidence_id, "request", status, values)
    episode.evidence_ledger.append(entry)
    episode.request_history.append(
        {
            "turn_index": turn_index,
            "root_evidence_id": root_evidence_id,
            "status": status,
            "values": list(values),
            "decoded_values": list(entry.decoded_values),
            "summary": entry.summary,
        }
    )
    return {
        "root_evidence_id": root_evidence_id,
        "question_en": self.question_text[root_evidence_id],
        "status": status,
        "revealed_tokens": revealed_tokens,
        "revealed_values": list(values),
        "revealed_value_labels": list(entry.decoded_values),
        "summary": entry.summary,
    }
```

This is the central ledger update step:

- the hidden environment is queried
- the result is decoded
- the visible ledger is extended
- the request history is extended
- a structured reveal payload is returned for traces

## 7. Recording The Agent’s Evolving Diagnosis State

The ledger does not only store findings. It also stores the sequence of diagnostic states across turns.

```python
def register_diagnosis(
    self,
    episode: LedgerEpisodeState,
    turn_index: int,
    normalized_response: dict[str, Any],
    shortlist_snapshot: list[dict[str, Any]],
    stop_signal: dict[str, Any],
    state_summary: DiagnosticStateSummary,
    policy_flags: list[str] | None = None,
) -> None:
    top_score = float(shortlist_snapshot[0]["score"]) if shortlist_snapshot else 0.0
    episode.diagnosis_history.append(
        DiagnosisSnapshot(
            turn_index=turn_index,
            predicted_pathology=normalized_response["predicted_pathology"],
            ranked_differential=list(normalized_response["ranked_differential"]),
            confidence=float(normalized_response["confidence"]),
            decision=str(normalized_response["decision"]),
            requested_evidence_id=normalized_response["requested_evidence_id"],
            brief_reasoning=str(normalized_response["brief_reasoning"]),
            stop_signal_level=str(stop_signal["level"]),
            top_shortlist_score=top_score,
            deterministic_top_pathology=(state_summary.top_candidates[0][0] if state_summary.top_candidates else None),
            deterministic_margin=float(state_summary.margin),
            policy_flags=list(policy_flags or []),
        )
    )
```

This is important because the system later uses this history for:

- stability checks
- stop logic
- drift control
- rendering the last few diagnosis turns back into the prompt

## 8. How The Ledger Connects To The Diagnosis State

The ledger made the visible state explicit; the diagnosis-state manager then converts that visible state into a deterministic anchored differential.

```python
class DeterministicDiagnosisStateManager:
    def score_pathologies(self, episode: LedgerEpisodeState) -> dict[str, float]:
        scores = Counter({label: 0.0 for label in self.label_names})
        if episode.prior_differential and USE_ONE_SHOT_PRIOR:
            for rank, (pathology, score) in enumerate(episode.prior_differential[:ONE_SHOT_PRIOR_TOP_K]):
                decay = max(0.35, 1.0 - 0.14 * rank)
                scores[pathology] += PRIOR_BLEND_WEIGHT * anchor_weight * float(score) * decay
        for entry in episode.evidence_ledger:
            root_id = entry.root_evidence_id
            base_rate = float(self.global_root_rates.get(root_id, 0.0))
            ...
            for pathology in self.label_names:
                path_rate = float(self.pathology_root_rates.get(pathology, {}).get(root_id, base_rate))
                ...
                scores[pathology] += float(delta)
        return dict(scores)
```

This is where the ledger stopped being just a record and started driving better policy behavior.

The diagnosis-state manager uses the current ledger contents to compute:

- top competing diagnoses
- margin between top-1 and top-2
- unresolved mass
- prior strength

That anchored state then drives:

- shortlist scoring
- stop decisions
- drift protection

## 9. How The Ledger Enters The Prompt

The prompt no longer hands the LLM a vague text transcript. It explicitly injects the structured ledger, request history, diagnosis history, anchored diagnosis state, and stop guidance.

```python
sections = [
    "Decoded evidence ledger:",
    ledger.render_ledger(episode),
    "",
    "Requested evidence history:",
    ledger.request_history_text(episode),
    "",
    "Recent diagnosis history:",
    ledger.diagnosis_history_text(episode),
    "",
    "Deterministic diagnosis state (current anchor for reasoning and shortlisting):",
    format_state_summary(state_summary),
    "",
    "Shortlisted legal evidence requests for this turn:",
    format_shortlist(shortlist),
]
```

This matters because the model is no longer asked to infer state implicitly from raw text. The state is exposed directly and consistently.

## 10. How The Ledger Is Used In The Main Sequential Loop

The main run loop in notebook 05 uses the ledger at every turn:

```python
episode = ledger.from_row(
    row,
    split_name=SPLIT_NAME,
    prior_differential=prior_lookup.get(case_id, []),
)

for turn_index in range(1, budget + 1):
    legal_actions = ledger.legal_actions(episode)
    state_summary = diagnosis_manager.summarize_state(episode)
    shortlist, pathology_weights = shortlister.shortlist(episode, legal_actions, state_summary)
    stop_signal = build_stop_signal(episode, state_summary, shortlist, remaining_budget=remaining_budget)
    turn_bundle = get_agent_response_with_repair(...)
    final_response, policy_flags = reconcile_agent_response(...)
    ledger.register_diagnosis(...)

    if final_response["decision"] == "request" and shortlist:
        reveal_payload = ledger.reveal(episode, final_response["requested_evidence_id"], turn_index)
        ...
    else:
        break
```

This is the real orchestration flow:

1. compile row into episode state
2. compute legal actions from the ledger
3. compute deterministic diagnosis state from the ledger
4. shortlist next questions
5. ask the LLM
6. log the diagnosis snapshot into the ledger
7. reveal one new finding and update the ledger
8. repeat

## 11. Concrete Example From A Successful Case

A good example is the refined live run for pulmonary embolism:

- [budget_008 traces.jsonl](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1/budget_008/traces.jsonl)

The sequence was:

1. ask if pain worsens with movement -> `no`
2. ask if symptoms worsen when lying down -> `no`
3. ask if coughing blood -> `yes`
4. ask history of DVT -> `no`
5. ask prior spontaneous pneumothorax -> `no`
6. ask fever -> `no`
7. ask recent surgery -> `yes`
8. ask active cancer -> `yes`

This is the kind of stateful workup that the ledger enabled:

- each answer was stored as a structured `LedgerEntry`
- each turn updated the visible evidence state
- the anchored diagnosis state kept revising the top competitors
- by the end, the system converged to `Pulmonary embolism`

That result is visible in:

- [budget_008 predictions.csv](/Users/bilalawan/claw/assignments/baseline_model/artifacts/sequential_single_agent_refined/single_agent_refined_live_test_1perclass_4budgets_anchor_guard_v1/budget_008/predictions.csv)

## 12. Why This Actually Worked

The ledger helped for three concrete reasons:

1. It separated hidden case state from visible agent state.
The environment knows the full DDXPlus case, but the agent only sees what has been explicitly revealed.

2. It made evidence readable and legal.
Decoded English findings plus parent-child gating made the workup clinically interpretable.

3. It gave the policy a stable source of truth.
The shortlist, stop logic, and diagnosis-state manager all read from the same structured episode state instead of each turn being reconstructed loosely from prompt text.

That is why the refined sequential notebook improved so much compared with the earlier sequential versions.

## 13. Best Short Summary

If you need a one-paragraph explanation:

The ledger in notebook 05 is a deterministic state manager for each DDXPlus case. It compiles the hidden full case into an episode, exposes only the initial evidence at the start, enforces legal follow-up questions, decodes every revealed finding into readable English, and records both evidence updates and diagnosis snapshots turn by turn. The sequential policy then uses that ledger as the authoritative source of truth for shortlisting actions, updating the anchored differential, and deciding whether to stop or ask for more evidence.
