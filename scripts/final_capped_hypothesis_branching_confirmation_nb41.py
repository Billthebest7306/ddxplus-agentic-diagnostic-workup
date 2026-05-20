from __future__ import annotations

# %% [markdown]
# # Notebook 41: Final Capped Hypothesis-Branching Confirmation
# 
# This notebook is the final lean live confirmation runner for the hypothesis-branching candidate-pool architecture.
# 
# It intentionally removes the close-confounder extra-root layer and keeps only:
# 
# - Notebook `13`-style base evidence acquisition with MLP-guided stopping
# - hypothesis-forced branch candidates when the learned branch gate fires
# - graph/Bayes/MLP candidate-pool resolution
# - top-1/top-3/top-5 and request-cost accounting
# 
# The cohort is approximately 100 held-out test cases: two per pathology plus two extra held-out cases, excluding prior live benchmark cohorts when possible.
# 
# This is a final confirmation notebook, not a threshold-search notebook. Labels are used only for evaluation after predictions are written.

# %%
# Interactive run controls.
# Edit these variables directly before running the notebook.

RUN_LIVE_API = True
ALLOW_DRY_RUN_BENCHMARK = False
RESUME_IF_AVAILABLE = True

LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4.1-mini"
LLM_API_KEY = ""
INTERACTIVE_API_KEY_BOOTSTRAP = True

# Final confirmation cohort controls.
BALANCED_CASES_PER_PATHOLOGY = 2
FINAL_TARGET_CASES = 100
EXCLUDE_REFERENCE_49_CASES = True
EXCLUDE_ALL_PRIOR_LIVE_BENCHMARK_CASES = True
RUN_VERSION_SUFFIX = "v1"

# Frozen confirmation controls.
ADAPTIVE_BRANCH_TRIGGER_THRESHOLD = 0.20
ADAPTIVE_MAX_BRANCHES = 2
ADAPTIVE_CONTINUATION_VALUE_THRESHOLD = 0.20
FINAL_TARGET_CANDIDATE_POOL_RECALL = 0.95

# Request-safety controls. Base branch keeps the Notebook 13 cap; spawned
# branches are shorter and the whole case has a hard total evidence-request cap.
BASE_REQUEST_CAP = 24
BRANCH_REQUEST_CAP = 8
HARD_TOTAL_REQUEST_CAP_PER_CASE = 24
MIN_REMAINING_REQUESTS_TO_LAUNCH_BRANCH = 2

DATASET_DIR_OVERRIDE = None  # Example: Path("/path/to/ddxplus") if needed.

# %%

import ast
import json
import math
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from IPython.display import display
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

pd.set_option("display.max_colwidth", 160)
pd.set_option("display.max_rows", 30)

ROOT = next(
    (candidate for candidate in [Path.cwd(), *Path.cwd().parents] if (candidate / "scripts" / "download_ddxplus.py").exists()),
    Path.cwd(),
)
DEFAULT_DATASET_DIR = ROOT / "dataset"
LEGACY_DATASET_DIR = ROOT / ".data" / "ddxplus" / "22687585"
DATASET_DIR = (
    Path(DATASET_DIR_OVERRIDE).expanduser()
    if DATASET_DIR_OVERRIDE
    else DEFAULT_DATASET_DIR
    if DEFAULT_DATASET_DIR.exists()
    else LEGACY_DATASET_DIR
    if LEGACY_DATASET_DIR.exists()
    else DEFAULT_DATASET_DIR
)
AUTO_DOWNLOAD_IF_MISSING = True

# Main run controls are set in the top interactive cell.
# Set RUN_LIVE_API=True and ALLOW_DRY_RUN_BENCHMARK=False there for the full live run.

# Final live confirmation cohort settings.
MAX_REQUEST_CAP = int(BASE_REQUEST_CAP)
SEQUENTIAL_SAMPLE_PER_CLASS = int(BALANCED_CASES_PER_PATHOLOGY)
SEQUENTIAL_MAX_CASES = int(FINAL_TARGET_CASES)
DRY_RUN_MAX_CASES = 2
RANDOM_SEED = 4141
SPLIT_NAME = "test"
TEMPERATURE = 0.0
TOP_P = 1.0

# Selected stop policy from notebook 12 / frozen Notebook 13.
SELECTED_STOP_POLICY = {
    "policy_name": "mlp_conf_ge_0.70_margin_ge_0.20_entropy_le_0.10_stab_0",
    "min_requests": 1,
    "mlp_confidence_min": 0.70,
    "mlp_margin_min": 0.20,
    "mlp_entropy_max": 0.10,
    "mlp_stability_min": 0,
    "official_final_head": "agreement_hybrid_final",
    "source_notebook": "12_stopping_policy_ablation.ipynb",
}

# Disabled future option. Do not use for this confirmation run unless explicitly starting a new experiment.
ACCURACY_BIASED_POLICY_DISABLED = {
    "enabled": False,
    "description": "Later optional check targeting ~9-10 requests with MLP final diagnosis.",
    "candidate_policy_name": "llm_conf_ge_0.85_stab_2__mlp_final",
}

SHORTLIST_SIZE = 10
SHORTLIST_STATS_SOURCE = "validate"
SHORTLIST_STATS_MAX_ROWS = 30000
USE_ONE_SHOT_PRIOR = True
ONE_SHOT_PRIOR_TOP_K = 5
PRIOR_BLEND_WEIGHT = 0.55
STATE_TOP_K = 5
STATE_SOFTMAX_TEMPERATURE = 0.33
STOP_CONFIDENCE_THRESHOLD = 0.60
LOW_VALUE_SHORTLIST_THRESHOLD = 0.12
GENERIC_RATE_PENALTY_THRESHOLD = 0.50

DRIFT_GUARD_TOP_K = 4
DRIFT_GUARD_MARGIN_THRESHOLD = 0.08
DRIFT_GUARD_CONFIDENCE_THRESHOLD = 0.72
FORCE_REQUEST_MARGIN_THRESHOLD = 0.07
FORCE_REQUEST_SCORE_THRESHOLD = 0.14
FORCE_STOP_MAX_SHORTLIST_SCORE = 0.10

# Adaptive hypothesis-forced branching policy for Notebook 41 confirmation.
BRANCH_TRIGGER_NAME = "hypothesis_branch_trigger_mlp_v1"
BRANCH_BUDGET = int(ADAPTIVE_MAX_BRANCHES)
BRANCH_JUDGE_NAME = "capped_graph_bayes_mlp_confirmation_resolver_v1"
BRANCH_MIN_REQUESTS = 2
BRANCH_DIVERGENCE_TURNS = 2
BRANCH_AVOID_BASE_FIRST_N_ROOTS = 2
BRANCH_FORCE_DIVERGENT_EARLY_REQUESTS = True
DRY_RUN_FORCE_BRANCH_PATH_SMOKE = True
BRANCH_PROFILE_TEMPLATES = [
    {
        "role_kind": "graph_bayes_challenger",
        "role_name_prefix": "Graph/Bayes hypothesis scout",
        "directive_template": "Treat {target_hypothesis} as the assigned challenger to {base_prediction}. Actively seek evidence that would support or refute {target_hypothesis}, especially roots that separate it from {base_prediction}.",
    },
    {
        "role_kind": "counteranchor_stress_test",
        "role_name_prefix": "Counter-anchor stress-test scout",
        "directive_template": "Assume the base answer {base_prediction} may be an anchoring error. Build the best evidence-grounded case for {target_hypothesis}, then abandon it only if the revealed ledger contradicts it.",
    },
]




SYNTH_TRAIN_STATES = 4000
SYNTH_VALIDATE_STATES = 2000
SYNTHETIC_MAX_SOURCE_ROWS = 60000
PAIR_TOP_ROOTS = 12
BRANCH_TRIGGER_TARGET_RATE = 0.22
BASE_PROTECTION_RESOLVER_MARGIN = 0.35
PSEUDO_CANDIDATE_MIN_RESOLVER_MARGIN = 0.12
CALIBRATION_MIN_ANALYSIS_CASES = 80
PROMOTION_MIN_ACCURACY = 0.94
PROMOTION_MAX_REGRESSION_RATE = 0.05
PROMOTION_MAX_MEAN_SELECTED_REQUESTS = 9.0
PROMOTION_MAX_MEAN_TOTAL_BRANCH_REQUESTS = 16.0

RUN_VERSION_BASE = f"final_capped_hypothesis_branching_confirmation100_{RUN_VERSION_SUFFIX}"
RUN_VERSION = RUN_VERSION_BASE if RUN_LIVE_API else f"final_capped_hypothesis_branching_confirmation100_dryrun_smoke_{RUN_VERSION_SUFFIX}"
RUN_NAME = RUN_VERSION
ARTIFACT_ROOT = ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
REPLICATE_PARENT = ARTIFACT_ROOT  # Compatibility for shared setup cells copied from Notebook 25.
CACHE_DIR = ARTIFACT_ROOT / "cache"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
for directory in [ARTIFACT_ROOT, CACHE_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if RUN_LIVE_API and INTERACTIVE_API_KEY_BOOTSTRAP and not LLM_API_KEY:
    from getpass import getpass
    LLM_API_KEY = getpass("Enter LLM_API_KEY: ")

USE_JSON_MODE = False
LLM_REQUEST_TIMEOUT_SECONDS = 180
LLM_MAX_RETRIES = 5
LLM_RETRY_BACKOFF_SECONDS = 8

INPUT_COST_PER_1K_TOKENS = None
OUTPUT_COST_PER_1K_TOKENS = None

rng = np.random.default_rng(RANDOM_SEED)

print("Project root       :", ROOT)
print("Dataset dir        :", DATASET_DIR)
print("Run live API       :", RUN_LIVE_API)
print("Allow dry-run      :", ALLOW_DRY_RUN_BENCHMARK)
print("LLM model          :", LLM_MODEL)
print("Temperature/top_p  :", TEMPERATURE, TOP_P)
print("Max request cap    :", MAX_REQUEST_CAP)
print("Base/branch caps   :", BASE_REQUEST_CAP, "/", BRANCH_REQUEST_CAP, "total=", HARD_TOTAL_REQUEST_CAP_PER_CASE)
print("Balanced per path. :", SEQUENTIAL_SAMPLE_PER_CLASS)
print("Final target cases :", FINAL_TARGET_CASES)
print("Exclude ref 49     :", EXCLUDE_REFERENCE_49_CASES)
print("Adaptive branches  :", BRANCH_BUDGET, "threshold=", ADAPTIVE_BRANCH_TRIGGER_THRESHOLD, "continuation=", ADAPTIVE_CONTINUATION_VALUE_THRESHOLD)
print("Candidate target   :", FINAL_TARGET_CANDIDATE_POOL_RECALL)
print("Sample per class   :", SEQUENTIAL_SAMPLE_PER_CLASS)
print("Max benchmark rows :", SEQUENTIAL_MAX_CASES)
print("Dry-run max cases  :", DRY_RUN_MAX_CASES)
print("Selected stop rule :", SELECTED_STOP_POLICY["policy_name"])
print("Branch trigger     :", BRANCH_TRIGGER_NAME)
print("Branch budget      :", BRANCH_BUDGET)
print("Branch judge       :", BRANCH_JUDGE_NAME)
print("Dry-run force branch smoke:", DRY_RUN_FORCE_BRANCH_PATH_SMOKE)
print("Synthetic states   :", SYNTH_TRAIN_STATES, "train /", SYNTH_VALIDATE_STATES, "validate")
print("Artifact root      :", ARTIFACT_ROOT)

# %% [markdown]
# 
# ## Dataset And Artifact Helpers
# 
# The improved notebook keeps the same DDXPlus file-validation and split-loading conventions as the rest of the project so that:
# 
# - `case_id` remains stable
# - one-shot predictions can be merged cleanly as priors
# - sequential budget sweeps can be compared later without ad-hoc reformatting

# %%

REQUIRED_FILES = [
    "release_evidences.json",
    "release_conditions.json",
    "release_train_patients.zip",
    "release_validate_patients.zip",
    "release_test_patients.zip",
]

SPLIT_TO_FILENAME = {
    "train": "release_train_patients.zip",
    "validate": "release_validate_patients.zip",
    "test": "release_test_patients.zip",
}


def ensure_dataset_present(dataset_dir: Path, auto_download: bool = True) -> dict[str, Path]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: dataset_dir / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing and auto_download:
        command = [sys.executable, str(ROOT / "scripts" / "download_ddxplus.py"), "--output-dir", str(dataset_dir)]
        subprocess.run(command, check=True)
        missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing DDXPlus files: {missing}")
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_patient_split(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        file_name = next((name for name in members if name.endswith(".csv")), None)
        if file_name is None:
            if not members:
                raise ValueError(f"Archive is empty: {zip_path}")
            file_name = members[0]
        with archive.open(file_name) as handle:
            return pd.read_csv(handle)


def attach_split_metadata(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["split"] = split_name
    enriched["source_row_index"] = np.arange(len(enriched), dtype=int)
    enriched["case_id"] = enriched["source_row_index"].map(lambda idx: f"{split_name}:{idx}")
    return enriched


def sample_fixed_cases_per_pathology(frame: pd.DataFrame, label_col: str, per_class: int, seed: int) -> pd.DataFrame:
    sampled_parts = []
    for pathology, group in frame.groupby(label_col, sort=True):
        take_n = min(per_class, len(group))
        sampled_parts.append(group.sample(n=take_n, random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=False)
    sampled = sampled.sort_values([label_col, "source_row_index"]).reset_index(drop=True)
    return sampled


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def upsert_prediction_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["case_id"] != row["case_id"]]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def safe_parse_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            return json.loads(stripped)
        except Exception:
            return ast.literal_eval(stripped)
    return list(raw)

# %% [markdown]
# ## Deterministic Evidence Ledger And Episode State Manager
# 
# The ledger still records the visible clinical state, but the episode now also stores enough diagnosis-state metadata to support anchored revision instead of free drift.

# %%
def parse_evidence_token(token: str) -> tuple[str, str | None]:
    if "_@_" not in token:
        return token, None
    root_id, value = token.split("_@_", 1)
    return root_id, value


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


class DeterministicEvidenceLedger:
    def __init__(self, evidence_metadata: dict[str, dict[str, Any]]):
        self.evidence_metadata = evidence_metadata
        self.root_ids = list(evidence_metadata.keys())
        self.question_text = {
            root_id: meta.get("question_en", root_id) for root_id, meta in evidence_metadata.items()
        }
        self.data_types = {
            root_id: meta.get("data_type", "B") for root_id, meta in evidence_metadata.items()
        }
        self.parent_question = {
            root_id: str(meta.get("code_question", root_id) or root_id)
            for root_id, meta in evidence_metadata.items()
        }

    def tokens_to_values(self, root_id: str, tokens: list[str]) -> list[str]:
        if self.data_types[root_id] == "B":
            return ["present"] if tokens else []
        return [value for _, value in map(parse_evidence_token, tokens) if value is not None]

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

    def decode_values(self, root_id: str, values: list[str]) -> list[str]:
        return [self.decode_value(root_id, value) for value in values]

    def summarize_observation(self, root_id: str, values: list[str], status: str) -> str:
        question = self.question_text[root_id]
        if status == "absent":
            return f"{question} -> no"
        if self.data_types[root_id] == "B":
            return f"{question} -> yes"
        decoded_values = self.decode_values(root_id, values)
        return f"{question} -> {', '.join(decoded_values) if decoded_values else 'observed'}"

    def _make_entry(self, turn_index: int, root_id: str, source: str, status: str, values: list[str]) -> LedgerEntry:
        decoded_values = self.decode_values(root_id, values)
        return LedgerEntry(
            turn_index=turn_index,
            root_evidence_id=root_id,
            question_en=self.question_text[root_id],
            source=source,
            status=status,
            values=list(values),
            decoded_values=decoded_values,
            summary=self.summarize_observation(root_id, values, status),
        )

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

    def root_present_or_implied_present(self, root_id: str, episode: LedgerEpisodeState) -> bool:
        if root_id in episode.revealed_roots and bool(episode.evidence_by_root.get(root_id, [])):
            return True
        for revealed_root in episode.revealed_roots:
            if (
                self.parent_question.get(revealed_root, revealed_root) == root_id
                and bool(episode.evidence_by_root.get(revealed_root, []))
            ):
                return True
        return False

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
        episode.shortlist_history.append(list(shortlist_snapshot))

    def render_ledger(self, episode: LedgerEpisodeState) -> str:
        return "\n".join(
            f"- {entry.root_evidence_id}: {entry.summary} [source={entry.source}]"
            for entry in episode.evidence_ledger
        )

    def request_history_text(self, episode: LedgerEpisodeState) -> str:
        if not episode.request_history:
            return "- no additional requests yet"
        return "\n".join(
            f"- turn {item['turn_index']}: {item['summary']}"
            for item in episode.request_history
        )

    def diagnosis_history_text(self, episode: LedgerEpisodeState) -> str:
        if not episode.diagnosis_history:
            return "- no diagnosis snapshots yet"
        return "\n".join(
            (
                f"- turn {snap.turn_index}: pred={snap.predicted_pathology}, det_top={snap.deterministic_top_pathology}, "
                f"margin={snap.deterministic_margin:.2f}, decision={snap.decision}, confidence={snap.confidence:.2f}"
            )
            for snap in episode.diagnosis_history[-3:]
        )

    def visible_context(self, episode: LedgerEpisodeState) -> dict[str, Any]:
        return {
            "case_id": episode.case_id,
            "age": episode.age,
            "sex": episode.sex,
            "visible_ledger": self.render_ledger(episode),
            "request_history_text": self.request_history_text(episode),
            "diagnosis_history_text": self.diagnosis_history_text(episode),
        }

# %% [markdown]
# ## One-Shot Priors, Anchored Diagnosis State, Shortlisting, And Stop Logic
# 
# This revision adds a deterministic diagnosis-state manager that converts the one-shot prior plus revealed evidence into a stable differential anchor. The shortlist and stop logic now operate against that state instead of following the LLM's latest guess too closely.

# %%
def discover_selected_one_shot_dir() -> Path | None:
    selected_path = ROOT / "artifacts" / "one_shot" / "selected_model.json"
    if not selected_path.exists():
        return None
    payload = load_json(selected_path)
    artifact_dir = payload.get("selected_artifact_dir")
    if not artifact_dir:
        return None
    return Path(artifact_dir)


def load_one_shot_prior_lookup(split_name: str, top_k: int) -> tuple[dict[str, list[tuple[str, float]]], pd.DataFrame | None]:
    one_shot_dir = discover_selected_one_shot_dir()
    if one_shot_dir is None:
        return {}, None
    predictions_path = one_shot_dir / "predictions.csv"
    if not predictions_path.exists():
        return {}, None
    frame = pd.read_csv(predictions_path)
    frame = frame[frame["split"] == split_name].copy()
    lookup = {}
    for row in frame.itertuples(index=False):
        preds = safe_parse_list(row.top5_predictions)
        scores = safe_parse_list(row.top5_prediction_scores)
        pairs = []
        for pathology, score in zip(preds[:top_k], scores[:top_k]):
            try:
                pairs.append((str(pathology), float(score)))
            except Exception:
                continue
        lookup[str(row.case_id)] = pairs
    return lookup, frame


def build_presence_rate_stats(
    frame: pd.DataFrame,
    evidence_metadata: dict[str, dict[str, Any]],
    source_name: str,
) -> dict[str, Any]:
    pathology_case_counts = Counter()
    pathology_root_counts: dict[str, Counter] = defaultdict(Counter)
    global_root_counts = Counter()
    all_roots = list(evidence_metadata.keys())
    for row in tqdm(frame[["PATHOLOGY", "EVIDENCES"]].itertuples(index=False), total=len(frame), desc=f"Presence stats ({source_name})"):
        pathology = str(row.PATHOLOGY)
        roots = {
            parse_evidence_token(str(token))[0]
            for token in safe_parse_list(row.EVIDENCES)
        }
        pathology_case_counts[pathology] += 1
        for root_id in roots:
            pathology_root_counts[pathology][root_id] += 1
            global_root_counts[root_id] += 1
    pathology_root_rates = {}
    for pathology, root_counter in pathology_root_counts.items():
        denom = max(1, pathology_case_counts[pathology])
        pathology_root_rates[pathology] = {
            root_id: float(root_counter.get(root_id, 0) / denom)
            for root_id in all_roots
        }
    global_root_rates = {
        root_id: float(global_root_counts.get(root_id, 0) / max(1, len(frame)))
        for root_id in all_roots
    }
    return {
        "source_name": source_name,
        "num_rows": int(len(frame)),
        "pathology_case_counts": dict(pathology_case_counts),
        "pathology_root_rates": pathology_root_rates,
        "global_root_rates": global_root_rates,
    }


def load_or_build_presence_rate_stats(
    dataset_paths: dict[str, Path],
    evidence_metadata: dict[str, dict[str, Any]],
    source_name: str,
    max_rows: int | None,
    seed: int,
) -> dict[str, Any]:
    cache_name = f"presence_rates_{source_name}_{max_rows or 'all'}.json"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        return load_json(cache_path)
    if source_name not in SPLIT_TO_FILENAME:
        raise ValueError(f"Unsupported stats source: {source_name}")
    frame = attach_split_metadata(load_patient_split(dataset_paths[SPLIT_TO_FILENAME[source_name]]), source_name)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    payload = build_presence_rate_stats(frame, evidence_metadata, source_name=source_name)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


@dataclass
class DiagnosticStateSummary:
    top_candidates: list[tuple[str, float]]
    normalized_weights: dict[str, float]
    margin: float
    top_score: float
    second_score: float
    stability_turns: int
    prior_strength: str
    anchor_top1: str | None
    anchor_margin: float
    unresolved_mass: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_candidates": [(pathology, float(score)) for pathology, score in self.top_candidates],
            "normalized_weights": {key: float(value) for key, value in self.normalized_weights.items()},
            "margin": float(self.margin),
            "top_score": float(self.top_score),
            "second_score": float(self.second_score),
            "stability_turns": int(self.stability_turns),
            "prior_strength": self.prior_strength,
            "anchor_top1": self.anchor_top1,
            "anchor_margin": float(self.anchor_margin),
            "unresolved_mass": float(self.unresolved_mass),
        }


class DeterministicDiagnosisStateManager:
    def __init__(self, stats_payload: dict[str, Any], ledger: DeterministicEvidenceLedger, label_names: list[str]):
        self.pathology_root_rates = stats_payload["pathology_root_rates"]
        self.global_root_rates = stats_payload["global_root_rates"]
        self.pathology_case_counts = stats_payload["pathology_case_counts"]
        self.label_names = list(label_names)
        self.ledger = ledger
        self.generic_root_ids = {
            root_id
            for root_id, rate in self.global_root_rates.items()
            if rate >= GENERIC_RATE_PENALTY_THRESHOLD
        }

    def _prior_strength(self, episode: LedgerEpisodeState) -> tuple[str, float, str | None, float]:
        if not episode.prior_differential:
            return "none", 0.0, None, 0.0
        top1_label, top1_score = episode.prior_differential[0]
        top2_score = episode.prior_differential[1][1] if len(episode.prior_differential) > 1 else 0.0
        margin = float(top1_score - top2_score)
        if top1_score >= 0.45 and margin >= 0.10:
            return "strong", 1.4 + 2.0 * margin, str(top1_label), margin
        if top1_score >= 0.20 or margin >= 0.05:
            return "moderate", 0.95 + 1.4 * margin, str(top1_label), margin
        return "weak", 0.55 + margin, str(top1_label), margin

    def _stability_turns(self, episode: LedgerEpisodeState) -> int:
        stable = 0
        last_label = None
        for snapshot in reversed(episode.diagnosis_history):
            if snapshot.deterministic_top_pathology is None:
                break
            if last_label is None:
                last_label = snapshot.deterministic_top_pathology
            if snapshot.deterministic_top_pathology == last_label:
                stable += 1
            else:
                break
        return stable

    def score_pathologies(self, episode: LedgerEpisodeState) -> dict[str, float]:
        scores = Counter({label: 0.0 for label in self.label_names})
        prior_strength, anchor_weight, _anchor_label, _anchor_margin = self._prior_strength(episode)
        if episode.prior_differential and USE_ONE_SHOT_PRIOR:
            for rank, (pathology, score) in enumerate(episode.prior_differential[:ONE_SHOT_PRIOR_TOP_K]):
                decay = max(0.35, 1.0 - 0.14 * rank)
                scores[pathology] += PRIOR_BLEND_WEIGHT * anchor_weight * float(score) * decay
        for entry in episode.evidence_ledger:
            root_id = entry.root_evidence_id
            base_rate = float(self.global_root_rates.get(root_id, 0.0))
            question_lower = self.ledger.question_text[root_id].lower()
            generic_multiplier = 1.0
            if root_id in self.generic_root_ids:
                generic_multiplier *= 0.82
            if "pain" in question_lower:
                generic_multiplier *= 0.86
            if self.ledger.parent_question.get(root_id, root_id) != root_id:
                generic_multiplier *= 0.95
            for pathology in self.label_names:
                path_rate = float(self.pathology_root_rates.get(pathology, {}).get(root_id, base_rate))
                discriminativeness = abs(path_rate - base_rate)
                weight = (0.28 + 2.2 * discriminativeness) * generic_multiplier
                if entry.status == "present":
                    positive_support = max(0.0, path_rate - base_rate)
                    negative_support = max(0.0, base_rate - path_rate)
                    delta = weight * (1.25 * positive_support - 0.90 * negative_support)
                    if path_rate >= 0.60 and base_rate <= 0.18:
                        delta += 0.12 * generic_multiplier
                    if path_rate <= 0.12 and base_rate >= 0.35:
                        delta -= 0.08 * generic_multiplier
                else:
                    supportive_absence = max(0.0, base_rate - path_rate)
                    contradictory_absence = max(0.0, path_rate - base_rate)
                    delta = weight * (0.95 * supportive_absence - 1.10 * contradictory_absence)
                    if path_rate >= 0.60 and base_rate <= 0.18:
                        delta -= 0.12 * generic_multiplier
                scores[pathology] += float(delta)
        return dict(scores)

    def summarize_state(self, episode: LedgerEpisodeState) -> DiagnosticStateSummary:
        scores = self.score_pathologies(episode)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_candidates = ranked[: max(STATE_TOP_K, 5)]
        if top_candidates:
            values = np.array([score for _, score in top_candidates], dtype=np.float64)
            shifted = np.exp((values - values.max()) / max(STATE_SOFTMAX_TEMPERATURE, 1e-6))
            shifted = shifted / max(shifted.sum(), 1e-8)
            normalized_weights = {
                pathology: float(weight)
                for (pathology, _), weight in zip(top_candidates, shifted)
            }
            top_score = float(top_candidates[0][1])
            second_score = float(top_candidates[1][1]) if len(top_candidates) > 1 else float(top_score)
            margin = float(top_score - second_score)
            unresolved_mass = float(1.0 - normalized_weights.get(top_candidates[0][0], 0.0))
        else:
            normalized_weights = {}
            top_score = 0.0
            second_score = 0.0
            margin = 0.0
            unresolved_mass = 1.0
        prior_strength, _anchor_weight, anchor_top1, anchor_margin = self._prior_strength(episode)
        return DiagnosticStateSummary(
            top_candidates=[(pathology, float(score)) for pathology, score in top_candidates],
            normalized_weights=normalized_weights,
            margin=margin,
            top_score=top_score,
            second_score=second_score,
            stability_turns=self._stability_turns(episode),
            prior_strength=prior_strength,
            anchor_top1=anchor_top1,
            anchor_margin=anchor_margin,
            unresolved_mass=unresolved_mass,
        )


@dataclass
class ShortlistAction:
    root_evidence_id: str
    question_en: str
    parent_root_id: str
    is_child: bool
    score: float
    weighted_present_rate: float
    pair_gap: float
    top12_gap: float
    global_present_rate: float
    dominant_pair: str
    reason: str


class DeterministicActionShortlister:
    def __init__(self, diagnosis_manager: DeterministicDiagnosisStateManager, shortlist_size: int):
        self.diagnosis_manager = diagnosis_manager
        self.pathology_root_rates = diagnosis_manager.pathology_root_rates
        self.global_root_rates = diagnosis_manager.global_root_rates
        self.shortlist_size = int(shortlist_size)

    def _pairwise_gap(self, labels: list[str], weights: np.ndarray, rates: np.ndarray) -> tuple[float, float, str]:
        if len(labels) < 2:
            return 0.0, 0.0, "single-hypothesis"
        gap_sum = 0.0
        weight_sum = 0.0
        best_gap = -1.0
        best_pair = ""
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pair_weight = float(weights[i] * weights[j])
                gap = abs(float(rates[i] - rates[j]))
                gap_sum += pair_weight * gap
                weight_sum += pair_weight
                if pair_weight * gap > best_gap:
                    best_gap = pair_weight * gap
                    best_pair = f"{labels[i]} vs {labels[j]}"
        top12_gap = abs(float(rates[0] - rates[1])) if len(rates) >= 2 else 0.0
        return (gap_sum / max(weight_sum, 1e-8), top12_gap, best_pair or "no-pair")

    def score_action(
        self,
        action: dict[str, Any],
        state_summary: DiagnosticStateSummary,
    ) -> ShortlistAction:
        root_id = action["root_evidence_id"]
        labels = [pathology for pathology, _ in state_summary.top_candidates[:STATE_TOP_K]]
        if labels:
            weights = np.array([state_summary.normalized_weights.get(label, 0.0) for label in labels], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-8)
            rates = np.array(
                [self.pathology_root_rates.get(label, {}).get(root_id, self.global_root_rates.get(root_id, 0.0)) for label in labels],
                dtype=np.float64,
            )
        else:
            labels = []
            weights = np.array([1.0], dtype=np.float64)
            rates = np.array([self.global_root_rates.get(root_id, 0.0)], dtype=np.float64)
        weighted_present_rate = float(np.dot(weights, rates))
        pair_gap, top12_gap, dominant_pair = self._pairwise_gap(labels, weights, rates)
        global_present_rate = float(self.global_root_rates.get(root_id, 0.0))
        split_balance = 1.0 - abs(weighted_present_rate - 0.5) * 2.0
        question_lower = action["question_en"].lower()
        generic_penalty = 1.0
        if global_present_rate >= GENERIC_RATE_PENALTY_THRESHOLD:
            generic_penalty *= 0.72
        if "pain" in question_lower:
            generic_penalty *= 0.78
        if any(token in question_lower for token in ["where", "how long", "what kind", "which side", "scale"]):
            generic_penalty *= 0.88
        if action["is_child"]:
            generic_penalty *= 0.93
        score = generic_penalty * (
            0.42 * pair_gap
            + 0.24 * top12_gap
            + 0.18 * split_balance
            + 0.10 * (1.0 - global_present_rate)
            + 0.06 * min(1.0, max(rates) if len(rates) else 0.0)
        )
        reason = (
            f"separates {dominant_pair}; pair_gap={pair_gap:.2f}; "
            f"top12_gap={top12_gap:.2f}; global_present={global_present_rate:.2f}"
        )
        return ShortlistAction(
            root_evidence_id=root_id,
            question_en=action["question_en"],
            parent_root_id=action["parent_root_id"],
            is_child=bool(action["is_child"]),
            score=float(score),
            weighted_present_rate=weighted_present_rate,
            pair_gap=float(pair_gap),
            top12_gap=float(top12_gap),
            global_present_rate=global_present_rate,
            dominant_pair=dominant_pair,
            reason=reason,
        )

    def shortlist(
        self,
        episode: LedgerEpisodeState,
        legal_actions: list[dict[str, Any]],
        state_summary: DiagnosticStateSummary,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        scored = [self.score_action(action, state_summary) for action in legal_actions]
        scored.sort(key=lambda item: (-item.score, item.is_child, item.question_en, item.root_evidence_id))
        shortlist = []
        parent_counts = Counter()
        for item in scored:
            parent_key = item.parent_root_id if item.is_child else item.root_evidence_id
            if parent_counts[parent_key] >= 2:
                continue
            shortlist.append(
                {
                    "root_evidence_id": item.root_evidence_id,
                    "question_en": item.question_en,
                    "parent_root_id": item.parent_root_id,
                    "is_child": item.is_child,
                    "score": round(item.score, 4),
                    "weighted_present_rate": round(item.weighted_present_rate, 4),
                    "pair_gap": round(item.pair_gap, 4),
                    "top12_gap": round(item.top12_gap, 4),
                    "global_present_rate": round(item.global_present_rate, 4),
                    "dominant_pair": item.dominant_pair,
                    "reason": item.reason,
                }
            )
            parent_counts[parent_key] += 1
            if len(shortlist) >= self.shortlist_size:
                break
        return shortlist, dict(state_summary.normalized_weights)


def build_stop_signal(
    episode: LedgerEpisodeState,
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    remaining_budget: int,
) -> dict[str, Any]:
    top_shortlist_score = float(shortlist[0]["score"]) if shortlist else 0.0
    if not shortlist:
        return {
            "level": "strong_stop",
            "reason": "No legal evidence requests remain.",
            "top_shortlist_score": top_shortlist_score,
            "deterministic_margin": float(state_summary.margin),
            "stability_turns": int(state_summary.stability_turns),
            "unresolved_mass": float(state_summary.unresolved_mass),
        }
    if (
        state_summary.margin >= 0.18
        and state_summary.stability_turns >= 2
        and state_summary.unresolved_mass <= 0.38
        and top_shortlist_score < FORCE_STOP_MAX_SHORTLIST_SCORE
        and len(episode.request_history) >= 1
    ):
        return {
            "level": "strong_stop",
            "reason": "The deterministic diagnosis state is stable and the remaining questions look low yield.",
            "top_shortlist_score": top_shortlist_score,
            "deterministic_margin": float(state_summary.margin),
            "stability_turns": int(state_summary.stability_turns),
            "unresolved_mass": float(state_summary.unresolved_mass),
        }
    if remaining_budget <= 1 and state_summary.margin >= 0.08:
        return {
            "level": "moderate_stop",
            "reason": "The budget is nearly exhausted and the top competing diagnoses are reasonably separated.",
            "top_shortlist_score": top_shortlist_score,
            "deterministic_margin": float(state_summary.margin),
            "stability_turns": int(state_summary.stability_turns),
            "unresolved_mass": float(state_summary.unresolved_mass),
        }
    if len(episode.request_history) >= 3 and top_shortlist_score < LOW_VALUE_SHORTLIST_THRESHOLD and state_summary.margin >= 0.10:
        return {
            "level": "moderate_stop",
            "reason": "Multiple findings are already visible and the next shortlisted questions are only weakly separating the top hypotheses.",
            "top_shortlist_score": top_shortlist_score,
            "deterministic_margin": float(state_summary.margin),
            "stability_turns": int(state_summary.stability_turns),
            "unresolved_mass": float(state_summary.unresolved_mass),
        }
    return {
        "level": "continue",
        "reason": "The leading diagnoses are not cleanly separated yet and there are still valuable discriminative questions available.",
        "top_shortlist_score": top_shortlist_score,
        "deterministic_margin": float(state_summary.margin),
        "stability_turns": int(state_summary.stability_turns),
        "unresolved_mass": float(state_summary.unresolved_mass),
    }


def reconcile_agent_response(
    normalized_response: dict[str, Any],
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    stop_signal: dict[str, Any],
    remaining_budget: int,
) -> tuple[dict[str, Any], list[str]]:
    result = dict(normalized_response)
    policy_flags: list[str] = []
    deterministic_ranked = [pathology for pathology, _ in state_summary.top_candidates[:5]]
    if deterministic_ranked:
        merged = []
        for label in list(result.get("ranked_differential", [])) + deterministic_ranked:
            if label not in merged:
                merged.append(label)
        result["ranked_differential"] = merged[:5]
        if result["predicted_pathology"] not in result["ranked_differential"]:
            result["ranked_differential"].insert(0, result["predicted_pathology"])
            result["ranked_differential"] = result["ranked_differential"][:5]

    if (
        result["decision"] == "stop"
        and stop_signal["level"] == "continue"
        and remaining_budget > 1
        and shortlist
        and state_summary.margin < FORCE_REQUEST_MARGIN_THRESHOLD
        and shortlist[0]["score"] >= FORCE_REQUEST_SCORE_THRESHOLD
    ):
        result["decision"] = "request"
        result["requested_evidence_id"] = shortlist[0]["root_evidence_id"]
        result["confidence"] = min(float(result["confidence"]), 0.55)
        policy_flags.append("forced_request_due_to_unresolved_competition")

    if (
        result["decision"] == "request"
        and stop_signal["level"] == "strong_stop"
        and shortlist
        and shortlist[0]["score"] < FORCE_STOP_MAX_SHORTLIST_SCORE
    ):
        result["decision"] = "stop"
        result["requested_evidence_id"] = None
        policy_flags.append("forced_stop_due_to_strong_state_signal")

    guard_candidates = deterministic_ranked[:DRIFT_GUARD_TOP_K]
    if (
        deterministic_ranked
        and result["predicted_pathology"] not in guard_candidates
        and (
            state_summary.margin >= DRIFT_GUARD_MARGIN_THRESHOLD
            or state_summary.prior_strength == "strong"
        )
        and float(result["confidence"]) <= DRIFT_GUARD_CONFIDENCE_THRESHOLD
    ):
        result["predicted_pathology"] = deterministic_ranked[0]
        result["ranked_differential"] = deterministic_ranked[:5]
        result["confidence"] = max(float(result["confidence"]), 0.55)
        policy_flags.append("drift_guard_override")
    elif deterministic_ranked:
        if deterministic_ranked[0] not in result["ranked_differential"] and state_summary.margin >= 0.10:
            result["ranked_differential"] = [deterministic_ranked[0], *result["ranked_differential"]]
            deduped = []
            for label in result["ranked_differential"]:
                if label not in deduped:
                    deduped.append(label)
            result["ranked_differential"] = deduped[:5]
            policy_flags.append("deterministic_top1_inserted")

    return result, policy_flags

# %% [markdown]
# ## Partial-Evidence MLP Belief Monitor

# %%
# Hybrid V1 partial-evidence MLP feedback utilities.

@dataclass
class ObservationSchema:
    root_ids: list[str]
    slot_slices: dict[str, tuple[int, int]]
    data_types: dict[str, str]
    possible_values: dict[str, list[str]]
    default_values: dict[str, str | None]
    categorical_integer_roots: set[str]
    question_text: dict[str, str]
    feature_names: list[str]

    @classmethod
    def from_metadata(cls, evidence_metadata: dict[str, dict[str, Any]]) -> "ObservationSchema":
        root_ids = list(evidence_metadata.keys())
        slot_slices = {}
        data_types = {}
        possible_values = {}
        default_values = {}
        categorical_integer_roots = set()
        question_text = {}
        feature_names = [f"age_bin_{idx}" for idx in range(8)] + ["sex_M", "sex_F"]
        cursor = 10
        for root_id in root_ids:
            meta = evidence_metadata[root_id]
            data_type = meta.get("data_type", "B")
            raw_values = meta.get("possible-values", [])
            values = [str(value) for value in raw_values]
            default_value = meta.get("default_value")
            default_value = None if default_value is None else str(default_value)
            question_text[root_id] = meta.get("question_en", root_id)
            data_types[root_id] = data_type
            possible_values[root_id] = values
            default_values[root_id] = default_value
            if data_type == "B":
                slot_slices[root_id] = (cursor, cursor + 1)
                feature_names.append(root_id)
                cursor += 1
            elif data_type == "C":
                if raw_values and not isinstance(raw_values[0], str):
                    categorical_integer_roots.add(root_id)
                    slot_slices[root_id] = (cursor, cursor + 1)
                    feature_names.append(root_id)
                    cursor += 1
                else:
                    slot_slices[root_id] = (cursor, cursor + len(values))
                    feature_names.extend(f"{root_id}__{value}" for value in values)
                    cursor += len(values)
            elif data_type == "M":
                slot_slices[root_id] = (cursor, cursor + len(values))
                feature_names.extend(f"{root_id}__{value}" for value in values)
                cursor += len(values)
            else:
                raise ValueError(f"Unsupported evidence type {data_type} for {root_id}")
        return cls(root_ids, slot_slices, data_types, possible_values, default_values, categorical_integer_roots, question_text, feature_names)

    @property
    def feature_size(self) -> int:
        return len(self.feature_names)

    def initial_state(self, age: int, sex: str) -> np.ndarray:
        state = np.zeros(self.feature_size, dtype=np.float32)
        state[encode_age(int(age))] = 1.0
        state[8 + encode_sex(str(sex))] = 1.0
        return state

    def apply_root_observation(self, state: np.ndarray, root_id: str, present_values: list[str] | None = None) -> np.ndarray:
        if root_id not in self.slot_slices:
            return state
        values = [str(value) for value in (present_values or [])]
        data_type = self.data_types[root_id]
        start, end = self.slot_slices[root_id]
        default_value = self.default_values[root_id]
        if data_type == "B":
            state[start] = 1.0 if values else -1.0
            return state
        if root_id in self.categorical_integer_roots:
            chosen = values[0] if values else default_value
            if chosen is None:
                state[start] = -1.0
            else:
                possible = self.possible_values[root_id]
                denominator = max(1, len(possible) - 1)
                state[start] = float(possible.index(str(chosen))) / denominator if str(chosen) in possible else -1.0
            return state
        state[start:end] = -1.0
        if values:
            for value in values:
                if value in self.possible_values[root_id]:
                    state[start + self.possible_values[root_id].index(value)] = 1.0
        return state


def encode_age(age: int) -> int:
    age = int(age)
    if age < 1:
        return 0
    if age <= 4:
        return 1
    if age <= 14:
        return 2
    if age <= 29:
        return 3
    if age <= 44:
        return 4
    if age <= 59:
        return 5
    if age <= 74:
        return 6
    return 7


def encode_sex(sex: str) -> int:
    sex = str(sex)
    if sex == "M":
        return 0
    if sex == "F":
        return 1
    raise ValueError(f"Unexpected sex value: {sex}")


class DirectDiagnosisMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], num_classes: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(features))


def discover_selected_partial_model_dir() -> Path:
    selected = ROOT / "artifacts" / "one_shot_partial_evidence" / "selected_model.json"
    if not selected.exists():
        raise FileNotFoundError("Run notebook 10 first; missing selected partial-evidence model: " + str(selected))
    payload = load_json(selected)
    model_dir = Path(payload["selected_artifact_dir"])
    if not (model_dir / "best_model.pt").exists():
        local_candidate = ROOT / "artifacts" / "one_shot_partial_evidence" / model_dir.name
        if (local_candidate / "best_model.pt").exists():
            model_dir = local_candidate
    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError("Selected partial-evidence checkpoint not found: " + str(model_dir / "best_model.pt"))
    return model_dir


def load_partial_mlp(model_dir: Path, schema: ObservationSchema) -> tuple[nn.Module, list[str], dict[str, Any]]:
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")
    resolved = checkpoint.get("resolved_run_config", {})
    labels = list(checkpoint.get("label_names", label_names))
    hidden_sizes = list(resolved.get("hidden_sizes", [2048, 2048, 2048]))
    dropout = float(resolved.get("dropout", 0.0))
    model = DirectDiagnosisMLP(schema.feature_size, hidden_sizes, len(labels), dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels, resolved


def encode_episode_state_for_mlp(episode: LedgerEpisodeState, schema: ObservationSchema) -> np.ndarray:
    state = schema.initial_state(episode.age, episode.sex)
    for root_id in sorted(episode.revealed_roots):
        tokens = episode.evidence_by_root.get(root_id, [])
        values = []
        for token in tokens:
            _, value = parse_evidence_token(str(token))
            values.append("present" if value is None else str(value))
        schema.apply_root_observation(state, root_id, values)
    return state


def compute_mlp_feedback(
    episode: LedgerEpisodeState,
    schema: ObservationSchema,
    model: nn.Module,
    labels: list[str],
    mlp_history: list[dict[str, Any]],
) -> dict[str, Any]:
    features = encode_episode_state_for_mlp(episode, schema)
    with torch.no_grad():
        logits = model(torch.tensor(features[None, :], dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    top_indices = order[:5]
    top_predictions = [labels[int(idx)] for idx in top_indices]
    top_probs = [float(probs[int(idx)]) for idx in top_indices]
    confidence = top_probs[0] if top_probs else 0.0
    second = top_probs[1] if len(top_probs) > 1 else 0.0
    margin = confidence - second
    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)) / np.log(len(probs)))
    previous_top = [item.get("top1") for item in mlp_history]
    stability_turns = 0
    for prev in reversed(previous_top):
        if prev == top_predictions[0]:
            stability_turns += 1
        else:
            break
    return {
        "top1": top_predictions[0],
        "top_predictions": top_predictions,
        "top_probabilities": top_probs,
        "confidence": float(confidence),
        "margin": float(margin),
        "entropy": float(entropy),
        "stability_turns": int(stability_turns),
        "revealed_root_count": int(len(episode.revealed_roots)),
    }


def format_mlp_feedback(mlp_feedback: dict[str, Any]) -> str:
    ranked = ", ".join(
        f"{label} ({prob:.3f})"
        for label, prob in zip(mlp_feedback["top_predictions"], mlp_feedback["top_probabilities"])
    )
    return "\n".join([
        f"- top diagnoses: {ranked}",
        f"- top1: {mlp_feedback['top1']}",
        f"- confidence: {mlp_feedback['confidence']:.3f}",
        f"- margin: {mlp_feedback['margin']:.3f}",
        f"- normalized_entropy: {mlp_feedback['entropy']:.3f}",
        f"- stable_same_top1_previous_turns: {mlp_feedback['stability_turns']}",
        f"- revealed_root_count: {mlp_feedback['revealed_root_count']}",
    ])


def mlp_ranked_json(mlp_feedback: dict[str, Any]) -> str:
    return json.dumps(mlp_feedback.get("top_predictions", []))


mlp_schema = None
partial_mlp_model = None
partial_mlp_labels = None
partial_mlp_config = None

# %% [markdown]
# ## Prompt Contract And Policy Controller
# 
# The LLM remains a single diagnostic workup agent, but it is now guided by a stronger deterministic diagnosis state:
# 
# - it sees decoded evidence only
# - it sees the anchored deterministic differential and prior-strength signal
# - it gets a shortlist whose reasons explicitly separate the current competing diagnoses
# - a lightweight policy controller can veto premature stop decisions or drift-heavy diagnosis jumps
# 
# The goal is to improve question quality and reduce late-turn collapse without changing the overall architecture.

# %%
SEQUENTIAL_RESPONSE_SCHEMA = {
    "decision": "request | stop",
    "requested_evidence_id": "nullable shortlisted root evidence id",
    "predicted_pathology": "one of the 49 DDXPlus pathologies",
    "ranked_differential": "ordered list of up to 5 pathology names",
    "confidence": "float between 0 and 1",
    "brief_reasoning": "one short sentence",
}


def format_prior_text(episode: LedgerEpisodeState, state_summary: DiagnosticStateSummary) -> str:
    if not episode.prior_differential:
        return "- no one-shot prior available"
    header = f"- strength={state_summary.prior_strength}, top1={state_summary.anchor_top1}, margin={state_summary.anchor_margin:.3f}"
    rows = [
        f"- {pathology}: {score:.3f}"
        for pathology, score in episode.prior_differential[:ONE_SHOT_PRIOR_TOP_K]
    ]
    return "\n".join([header, *rows])


def format_state_summary(state_summary: DiagnosticStateSummary) -> str:
    if not state_summary.top_candidates:
        return "- no deterministic candidates available"
    rows = [
        f"- {pathology}: raw_score={score:.3f}, weight={state_summary.normalized_weights.get(pathology, 0.0):.3f}"
        for pathology, score in state_summary.top_candidates[:5]
    ]
    footer = [
        f"- margin(top1-top2)={state_summary.margin:.3f}",
        f"- stability_turns={state_summary.stability_turns}",
        f"- unresolved_mass={state_summary.unresolved_mass:.3f}",
    ]
    return "\n".join(rows + footer)


def format_shortlist(shortlist: list[dict[str, Any]]) -> str:
    if not shortlist:
        return "- no legal actions remain"
    return "\n".join(
        (
            f"- {item['root_evidence_id']} | score={item['score']:.3f} | "
            f"pair_gap={item['pair_gap']:.2f} | top12_gap={item['top12_gap']:.2f} | "
            f"{item['question_en']} | {item['reason']}"
        )
        for item in shortlist
    )


def build_system_prompt(label_names: list[str]) -> str:
    pathology_text = ", ".join(label_names)
    return (
        "You are a single diagnostic workup agent operating inside a structured DDXPlus environment. "
        "The deterministic evidence ledger and deterministic diagnosis state are the authoritative state summary for the episode. "
        "Use the shortlist to choose the single best next question when the leading diagnoses are not yet separated. "
        "Do not abandon a strong anchored diagnosis unless newly revealed evidence clearly contradicts it. "
        "When the deterministic state shows clear separation and the shortlist looks low-yield, stop instead of requesting generic questions. "
        "Never request evidence outside the shortlisted root ids. "
        "Always return exactly one JSON object and no extra text. "
        f"Valid pathology labels are: {pathology_text}."
    )


def build_user_prompt(
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    turn_index: int,
    max_budget: int,
) -> str:
    remaining_budget = max_budget - turn_index + 1
    sections = [
        f"Case ID: {episode.case_id}",
        f"Turn: {turn_index}",
        f"Max request budget for this run: {max_budget}",
        f"Remaining request budget including this turn: {remaining_budget}",
        f"Demographics: age={episode.age}, sex={episode.sex}",
        "",
        "Decoded evidence ledger:",
        ledger.render_ledger(episode),
        "",
        "Requested evidence history:",
        ledger.request_history_text(episode),
        "",
        "Recent diagnosis history:",
        ledger.diagnosis_history_text(episode),
        "",
        "One-shot prior differential (initial evidence only):",
        format_prior_text(episode, state_summary),
        "",
        "Deterministic diagnosis state (current anchor for reasoning and shortlisting):",
        format_state_summary(state_summary),
        "",
        "Stop guidance:",
        f"- level: {stop_signal['level']}",
        f"- rationale: {stop_signal['reason']}",
        f"- deterministic_margin: {stop_signal['deterministic_margin']:.3f}",
        f"- stable_prediction_turns: {stop_signal['stability_turns']}",
        f"- unresolved_mass: {stop_signal['unresolved_mass']:.3f}",
        f"- top_shortlist_score: {stop_signal['top_shortlist_score']:.3f}",
        "",
        "Shortlisted legal evidence requests for this turn:",
        format_shortlist(shortlist),
        "",
        "Return JSON with the following keys only:",
        json.dumps(SEQUENTIAL_RESPONSE_SCHEMA, indent=2),
        "",
        "Rules:",
        "- If decision is request, requested_evidence_id must be one of the shortlisted root ids shown above.",
        "- If decision is stop, requested_evidence_id must be null.",
        "- ranked_differential must contain at most 5 labels and predicted_pathology must appear in it.",
        "- Prefer questions that separate the current top competing diagnoses rather than generic high-frequency questions.",
        "- If the deterministic margin is small, request a separating question instead of stopping.",
        "- If the deterministic margin is large and the shortlist is low-value, stop instead of drifting to a new unsupported diagnosis.",
    ]
    return "\n".join(sections)


def build_messages(
    label_names: list[str],
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    turn_index: int,
    max_budget: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(label_names)},
        {
            "role": "user",
            "content": build_user_prompt(
                ledger=ledger,
                episode=episode,
                shortlist=shortlist,
                state_summary=state_summary,
                stop_signal=stop_signal,
                turn_index=turn_index,
                max_budget=max_budget,
            ),
        },
    ]


def chat_completion_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def extract_content_text(response_payload: dict[str, Any]) -> str:
    message = response_payload["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        pieces = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(item.get("text", ""))
        return "".join(pieces)
    raise ValueError(f"Unsupported message content payload: {message!r}")


def call_openai_compatible(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], dict[str, int]]:
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is empty. Fill LLM_API_KEY in the first cell or use the secure notebook bootstrap prompt.")
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if USE_JSON_MODE:
        body["response_format"] = {"type": "json_object"}
    url = chat_completion_url(LLM_BASE_URL)
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    response = None
    last_error: Exception | None = None
    for attempt_index in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in retryable_statuses and attempt_index < LLM_MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else LLM_RETRY_BACKOFF_SECONDS * attempt_index
                print(f"Transient API status {response.status_code}; retrying in {sleep_seconds:.1f}s ({attempt_index}/{LLM_MAX_RETRIES}).")
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None and status_code not in retryable_statuses:
                raise
            if attempt_index >= LLM_MAX_RETRIES:
                raise
            sleep_seconds = LLM_RETRY_BACKOFF_SECONDS * attempt_index
            print(f"Transient API connection error: {exc}; retrying in {sleep_seconds:.1f}s ({attempt_index}/{LLM_MAX_RETRIES}).")
            time.sleep(sleep_seconds)
    if response is None:
        raise RuntimeError(f"No API response after {LLM_MAX_RETRIES} attempts") from last_error
    response_payload = response.json()
    usage = response_payload.get("usage", {})
    token_usage = {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
    }
    return extract_content_text(response_payload), response_payload, token_usage


def parse_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in the model response.")
    return json.loads(match.group(0))


def _coerce_ranked_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        return [item.strip() for item in stripped.split("|") if item.strip()]
    return []


def normalize_agent_response(payload: dict[str, Any], label_names: list[str]) -> dict[str, Any]:
    decision = str(payload.get("decision", "stop")).strip().lower()
    if decision not in {"request", "stop"}:
        decision = "stop"
    requested_evidence_id = payload.get("requested_evidence_id")
    if requested_evidence_id in ("", "null", "None"):
        requested_evidence_id = None
    if requested_evidence_id is not None:
        requested_evidence_id = str(requested_evidence_id).strip()
    predicted_pathology = str(payload.get("predicted_pathology", "")).strip()
    ranked_differential = _coerce_ranked_list(payload.get("ranked_differential"))
    valid_ranked = []
    for item in ranked_differential:
        if item in label_names and item not in valid_ranked:
            valid_ranked.append(item)
    if predicted_pathology in label_names and predicted_pathology not in valid_ranked:
        valid_ranked.insert(0, predicted_pathology)
    if not predicted_pathology and valid_ranked:
        predicted_pathology = valid_ranked[0]
    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    brief_reasoning = str(payload.get("brief_reasoning", "")).strip()[:400]
    return {
        "decision": decision,
        "requested_evidence_id": requested_evidence_id if decision == "request" else None,
        "predicted_pathology": predicted_pathology,
        "ranked_differential": valid_ranked[:5],
        "confidence": confidence,
        "brief_reasoning": brief_reasoning,
    }


def validate_agent_response(
    normalized_response: dict[str, Any],
    shortlist_ids: set[str],
    label_names: list[str],
) -> str | None:
    if normalized_response["predicted_pathology"] not in label_names:
        return "predicted_pathology is missing or outside the DDXPlus label set."
    if not normalized_response["ranked_differential"]:
        return "ranked_differential is empty."
    if normalized_response["decision"] == "request":
        requested = normalized_response["requested_evidence_id"]
        if requested is None:
            return "decision=request but requested_evidence_id is null."
        if requested not in shortlist_ids:
            return "requested_evidence_id is not in the current shortlist."
    return None


def build_repair_messages(
    messages: list[dict[str, str]],
    raw_text: str,
    error_message: str,
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": raw_text},
        {
            "role": "user",
            "content": (
                "Your previous response violated the required JSON contract. "
                f"Problem: {error_message}. "
                "Return one corrected JSON object only."
            ),
        },
    ]


def scripted_dry_run_response(
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    turn_index: int,
    max_budget: int,
    label_names: list[str],
) -> dict[str, Any]:
    ranked = [pathology for pathology, _ in state_summary.top_candidates[:5]] or list(label_names[:5])
    predicted = ranked[0]
    confidence = float(np.clip(0.42 + 0.22 * state_summary.margin + 0.05 * len(episode.request_history), 0.0, 0.95))
    if stop_signal["level"] == "strong_stop":
        return {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": predicted,
            "ranked_differential": ranked,
            "confidence": max(confidence, 0.68),
            "brief_reasoning": "Dry-run mode stops because the deterministic state is stable and the remaining shortlist is weak.",
        }
    if turn_index >= max_budget or not shortlist:
        return {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": predicted,
            "ranked_differential": ranked,
            "confidence": max(confidence, 0.55),
            "brief_reasoning": "Dry-run mode stops at the budget limit using the deterministic diagnosis state.",
        }
    if stop_signal["level"] == "moderate_stop" and state_summary.margin >= 0.10:
        return {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": predicted,
            "ranked_differential": ranked,
            "confidence": max(confidence, 0.60),
            "brief_reasoning": "Dry-run mode stops because the deterministic margin is already reasonably separated.",
        }
    return {
        "decision": "request",
        "requested_evidence_id": shortlist[0]["root_evidence_id"],
        "predicted_pathology": predicted,
        "ranked_differential": ranked,
        "confidence": min(confidence, 0.62),
        "brief_reasoning": "Dry-run mode requests the best separating question from the deterministic shortlist.",
    }


def get_agent_response_with_repair(
    label_names: list[str],
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    turn_index: int,
    max_budget: int,
) -> dict[str, Any]:
    messages = build_messages(
        label_names=label_names,
        ledger=ledger,
        episode=episode,
        shortlist=shortlist,
        state_summary=state_summary,
        stop_signal=stop_signal,
        turn_index=turn_index,
        max_budget=max_budget,
    )
    shortlist_ids = {item["root_evidence_id"] for item in shortlist}
    raw_attempts = []
    api_calls = 0
    input_tokens = 0
    output_tokens = 0
    estimated_cost = None

    if not RUN_LIVE_API:
        raw_text = json.dumps(
            scripted_dry_run_response(
                episode=episode,
                shortlist=shortlist,
                state_summary=state_summary,
                stop_signal=stop_signal,
                turn_index=turn_index,
                max_budget=max_budget,
                label_names=label_names,
            )
        )
        payload = {"dry_run": True, "message": raw_text}
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        api_calls = 1
    else:
        raw_text, payload, token_usage = call_openai_compatible(messages)
        api_calls = 1
    raw_attempts.append({"attempt_index": 1, "raw_text": raw_text, "payload": payload})
    input_tokens += token_usage["input_tokens"]
    output_tokens += token_usage["output_tokens"]

    try:
        normalized = normalize_agent_response(parse_json_response(raw_text), label_names)
        error_message = validate_agent_response(normalized, shortlist_ids, label_names)
    except Exception as exc:
        normalized = None
        error_message = str(exc)

    if error_message is not None:
        repair_messages = build_repair_messages(messages, raw_text, error_message)
        if not RUN_LIVE_API:
            repair_text = json.dumps(
                scripted_dry_run_response(
                    episode=episode,
                    shortlist=shortlist,
                    state_summary=state_summary,
                    stop_signal={**stop_signal, "level": "strong_stop"},
                    turn_index=turn_index,
                    max_budget=max_budget,
                    label_names=label_names,
                )
            )
            repair_payload = {"dry_run": True, "message": repair_text}
            repair_usage = {"input_tokens": 0, "output_tokens": 0}
            api_calls += 1
        else:
            repair_text, repair_payload, repair_usage = call_openai_compatible(repair_messages)
            api_calls += 1
        raw_attempts.append({"attempt_index": 2, "raw_text": repair_text, "payload": repair_payload})
        input_tokens += repair_usage["input_tokens"]
        output_tokens += repair_usage["output_tokens"]
        normalized = normalize_agent_response(parse_json_response(repair_text), label_names)
        error_message = validate_agent_response(normalized, shortlist_ids, label_names)

    error_flags = []
    if error_message is not None:
        error_flags.append(error_message)
        fallback_labels = [pathology for pathology, _ in state_summary.top_candidates[:5]]
        if not fallback_labels:
            fallback_labels = [label_names[0]]
        normalized = {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": fallback_labels[0],
            "ranked_differential": fallback_labels,
            "confidence": 0.0,
            "brief_reasoning": "Forced stop after invalid responses.",
        }

    return {
        "messages": messages,
        "normalized_response": normalized,
        "raw_attempts": raw_attempts,
        "api_calls": api_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "error_flags": error_flags,
    }

# %% [markdown]
# ## Hybrid Prompt Contract

# %%
# Hybrid prompt wrappers. These keep the response contract from notebook 08 and add MLP belief context.

def build_hybrid_user_prompt(
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
) -> str:
    base_prompt = build_user_prompt(
        ledger=ledger,
        episode=episode,
        shortlist=shortlist,
        state_summary=state_summary,
        stop_signal=stop_signal,
        turn_index=turn_index,
        max_budget=max_budget,
    )
    hybrid_section = "\n".join([
        "",
        "Hybrid MLP belief monitor (partial-evidence neural diagnostic head):",
        format_mlp_feedback(mlp_feedback),
        "",
        "Hybrid guidance:",
        "- Treat the MLP as a structured belief monitor, not as hidden ground truth.",
        "- If the MLP is confident and your differential agrees, stop rather than asking low-yield questions.",
        "- If your differential conflicts with the MLP top diagnoses, request evidence that separates the disagreement.",
        "- Do not change diagnosis without revealed evidence that supports the change.",
    ])
    return base_prompt + hybrid_section


def build_hybrid_messages(
    label_names: list[str],
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(label_names)},
        {
            "role": "user",
            "content": build_hybrid_user_prompt(
                ledger=ledger,
                episode=episode,
                shortlist=shortlist,
                state_summary=state_summary,
                stop_signal=stop_signal,
                mlp_feedback=mlp_feedback,
                turn_index=turn_index,
                max_budget=max_budget,
            ),
        },
    ]


def scripted_hybrid_dry_run_response(
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
    label_names: list[str],
) -> dict[str, Any]:
    ranked = list(mlp_feedback.get("top_predictions", [])) or [pathology for pathology, _ in state_summary.top_candidates[:5]] or list(label_names[:5])
    predicted = ranked[0]
    confidence = float(max(mlp_feedback.get("confidence", 0.0), 0.42 + 0.12 * state_summary.margin))
    if stop_signal["level"] in {"strong_stop", "hybrid_mlp_stop"}:
        return {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": predicted,
            "ranked_differential": ranked[:5],
            "confidence": max(confidence, 0.65),
            "brief_reasoning": "Dry-run hybrid stops because MLP belief and deterministic state are sufficiently stable.",
        }
    if turn_index >= max_budget or not shortlist:
        return {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": predicted,
            "ranked_differential": ranked[:5],
            "confidence": max(confidence, 0.55),
            "brief_reasoning": "Dry-run hybrid stops at the cap or because no legal questions remain.",
        }
    return {
        "decision": "request",
        "requested_evidence_id": shortlist[0]["root_evidence_id"],
        "predicted_pathology": predicted,
        "ranked_differential": ranked[:5],
        "confidence": min(confidence, 0.62),
        "brief_reasoning": "Dry-run hybrid requests the best shortlisted evidence while MLP uncertainty remains.",
    }


def get_hybrid_agent_response_with_repair(
    label_names: list[str],
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
) -> dict[str, Any]:
    messages = build_hybrid_messages(
        label_names=label_names,
        ledger=ledger,
        episode=episode,
        shortlist=shortlist,
        state_summary=state_summary,
        stop_signal=stop_signal,
        mlp_feedback=mlp_feedback,
        turn_index=turn_index,
        max_budget=max_budget,
    )
    shortlist_ids = {item["root_evidence_id"] for item in shortlist}
    raw_attempts = []
    api_calls = 0
    input_tokens = 0
    output_tokens = 0
    estimated_cost = None

    if not RUN_LIVE_API:
        raw_text = json.dumps(scripted_hybrid_dry_run_response(episode, shortlist, state_summary, stop_signal, mlp_feedback, turn_index, max_budget, label_names))
        payload = {"dry_run": True, "message": raw_text}
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        api_calls = 1
    else:
        raw_text, payload, token_usage = call_openai_compatible(messages)
        api_calls = 1
    raw_attempts.append({"attempt_index": 1, "raw_text": raw_text, "payload": payload})
    input_tokens += token_usage["input_tokens"]
    output_tokens += token_usage["output_tokens"]

    try:
        normalized = normalize_agent_response(parse_json_response(raw_text), label_names)
        error_message = validate_agent_response(normalized, shortlist_ids, label_names)
    except Exception as exc:
        normalized = None
        error_message = str(exc)

    if error_message is not None:
        repair_messages = build_repair_messages(messages, raw_text, error_message)
        if not RUN_LIVE_API:
            repair_text = json.dumps(scripted_hybrid_dry_run_response(episode, shortlist, state_summary, {**stop_signal, "level": "strong_stop"}, mlp_feedback, turn_index, max_budget, label_names))
            repair_payload = {"dry_run": True, "message": repair_text}
            repair_usage = {"input_tokens": 0, "output_tokens": 0}
            api_calls += 1
        else:
            repair_text, repair_payload, repair_usage = call_openai_compatible(repair_messages)
            api_calls += 1
        raw_attempts.append({"attempt_index": 2, "raw_text": repair_text, "payload": repair_payload})
        input_tokens += repair_usage["input_tokens"]
        output_tokens += repair_usage["output_tokens"]
        normalized = normalize_agent_response(parse_json_response(repair_text), label_names)
        error_message = validate_agent_response(normalized, shortlist_ids, label_names)

    error_flags = []
    if error_message is not None:
        error_flags.append(error_message)
        fallback_labels = list(mlp_feedback.get("top_predictions", [])) or [pathology for pathology, _ in state_summary.top_candidates[:5]] or [label_names[0]]
        normalized = {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": fallback_labels[0],
            "ranked_differential": fallback_labels[:5],
            "confidence": 0.0,
            "brief_reasoning": "Forced stop after invalid responses.",
        }

    return {
        "messages": messages,
        "normalized_response": normalized,
        "raw_attempts": raw_attempts,
        "api_calls": api_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "error_flags": error_flags,
    }

# %% [markdown]
# ## Prepare Benchmark Cases And Load Hybrid Components

# %%
# Load dataset, benchmark cases, priors, shortlist statistics, and the selected partial-evidence MLP.

dataset_paths = ensure_dataset_present(DATASET_DIR, auto_download=AUTO_DOWNLOAD_IF_MISSING)
evidences = load_json(dataset_paths["release_evidences.json"])
conditions = load_json(dataset_paths["release_conditions.json"])
label_names = list(conditions.keys())

raw_split = attach_split_metadata(load_patient_split(dataset_paths[SPLIT_TO_FILENAME[SPLIT_NAME]]), SPLIT_NAME)
split_lookup = {
    str(row["case_id"]): row
    for row in raw_split.to_dict(orient="records")
}

REFERENCE_CASE_SOURCES = [
    ROOT / "artifacts" / "sequential_hybrid_mlp_feedback" / "selected_stop_live_confirmation_49case_v1" / "benchmark_cases.csv",
    ROOT / "artifacts" / "graph_algorithmic_ledger" / "live_graph_bayes_rescue_confirmation_49case_v1" / "benchmark_cases.csv",
    ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1" / "benchmark_cases.csv",
    ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_balanced2_v1" / "benchmark_cases.csv",
]
if EXCLUDE_ALL_PRIOR_LIVE_BENCHMARK_CASES:
    for discovered_path in sorted((ROOT / "artifacts").glob("**/benchmark_cases.csv")):
        try:
            if discovered_path.resolve().is_relative_to(ARTIFACT_ROOT.resolve()):
                continue
        except AttributeError:
            if str(ARTIFACT_ROOT.resolve()) in str(discovered_path.resolve()):
                continue
        REFERENCE_CASE_SOURCES.append(discovered_path)
REFERENCE_CASE_SOURCES = sorted(set(REFERENCE_CASE_SOURCES))
reference_case_ids: set[str] = set()
reference_rows = []
if EXCLUDE_REFERENCE_49_CASES or EXCLUDE_ALL_PRIOR_LIVE_BENCHMARK_CASES:
    for reference_path in REFERENCE_CASE_SOURCES:
        if reference_path.exists():
            reference_frame = pd.read_csv(reference_path)
            if "case_id" in reference_frame.columns:
                ids = set(reference_frame["case_id"].astype(str))
                reference_case_ids.update(ids)
                reference_rows.append({"source": str(reference_path), "case_ids": len(ids)})
reference_exclusion_summary = pd.DataFrame(reference_rows, columns=["source", "case_ids"])
reference_exclusion_summary.to_csv(REPLICATE_PARENT / "reference_case_exclusion_summary.csv", index=False)

eligible_split = raw_split[~raw_split["case_id"].astype(str).isin(reference_case_ids)].copy() if reference_case_ids else raw_split.copy()
benchmark_df = sample_fixed_cases_per_pathology(
    eligible_split,
    label_col="PATHOLOGY",
    per_class=SEQUENTIAL_SAMPLE_PER_CLASS,
    seed=RANDOM_SEED,
)
if RUN_LIVE_API and FINAL_TARGET_CASES is not None and len(benchmark_df) < FINAL_TARGET_CASES:
    selected_case_ids = set(benchmark_df["case_id"].astype(str))
    extra_pool = eligible_split[~eligible_split["case_id"].astype(str).isin(selected_case_ids)].copy()
    extra_n = min(int(FINAL_TARGET_CASES) - len(benchmark_df), len(extra_pool))
    if extra_n > 0:
        extra_cases = extra_pool.sample(n=extra_n, random_state=RANDOM_SEED + 41)
        benchmark_df = (
            pd.concat([benchmark_df, extra_cases], ignore_index=False)
            .sort_values(["PATHOLOGY", "source_row_index"])
            .reset_index(drop=True)
        )
effective_max_cases = DRY_RUN_MAX_CASES if (not RUN_LIVE_API and ALLOW_DRY_RUN_BENCHMARK) else SEQUENTIAL_MAX_CASES
if effective_max_cases is not None:
    benchmark_df = (
        benchmark_df.sample(n=min(effective_max_cases, len(benchmark_df)), random_state=RANDOM_SEED)
        .sort_values(["PATHOLOGY", "source_row_index"])
        .reset_index(drop=True)
    )
pathology_case_counts = benchmark_df.groupby("PATHOLOGY")["case_id"].nunique().reset_index(name="selected_cases")
pathology_case_counts.to_csv(REPLICATE_PARENT / "balanced_pathology_case_counts.csv", index=False)

prior_lookup, one_shot_prediction_frame = load_one_shot_prior_lookup(SPLIT_NAME, ONE_SHOT_PRIOR_TOP_K)
benchmark_df["prior_available"] = benchmark_df["case_id"].map(lambda case_id: case_id in prior_lookup)

presence_stats = load_or_build_presence_rate_stats(
    dataset_paths=dataset_paths,
    evidence_metadata=evidences,
    source_name=SHORTLIST_STATS_SOURCE,
    max_rows=SHORTLIST_STATS_MAX_ROWS,
    seed=RANDOM_SEED,
)

ledger = DeterministicEvidenceLedger(evidences)
diagnosis_manager = DeterministicDiagnosisStateManager(presence_stats, ledger=ledger, label_names=label_names)
shortlister = DeterministicActionShortlister(diagnosis_manager, shortlist_size=SHORTLIST_SIZE)

mlp_schema = ObservationSchema.from_metadata(evidences)
partial_mlp_dir = discover_selected_partial_model_dir()
partial_mlp_model, partial_mlp_labels, partial_mlp_config = load_partial_mlp(partial_mlp_dir, mlp_schema)

benchmark_case_export = benchmark_df[["case_id", "PATHOLOGY", "AGE", "SEX", "INITIAL_EVIDENCE", "prior_available"]]
benchmark_case_export.to_csv(REPLICATE_PARENT / "benchmark_cases.csv", index=False)

print("Benchmark size           :", len(benchmark_df))
print("Prior live cases excluded:", len(reference_case_ids))
print("Unique pathologies       :", benchmark_df["PATHOLOGY"].nunique())
print("One-shot prior coverage  :", int(benchmark_df["prior_available"].sum()), "/", len(benchmark_df))
print("Shortlist stats source   :", presence_stats["source_name"], "rows=", presence_stats["num_rows"])
print("Partial MLP dir          :", partial_mlp_dir)
print("Partial MLP labels       :", len(partial_mlp_labels))
print("Partial MLP feature size :", mlp_schema.feature_size)
display(pathology_case_counts.head(10))
display(benchmark_df[["case_id", "PATHOLOGY", "AGE", "SEX", "INITIAL_EVIDENCE", "prior_available"]].head(10))

# %% [markdown]
# ## Lambda-Aware Policy Controls
# 
# These helpers estimate the marginal value of another evidence request and convert that estimate into cost-sensitive stop/request guidance.

# %%
def lambda_slug(value: float) -> str:
    return f"lambda_{value:.3f}".replace(".", "p")


def estimate_marginal_evidence_value(
    episode: LedgerEpisodeState,
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    remaining_budget: int,
) -> dict[str, float]:
    top_shortlist_score = float(shortlist[0]["score"]) if shortlist else 0.0
    ambiguity = max(0.0, 0.24 - float(state_summary.margin)) / 0.24
    unresolved = float(np.clip(state_summary.unresolved_mass, 0.0, 1.0))
    shortlist_value = float(np.clip(top_shortlist_score, 0.0, 1.0))
    stability_penalty = min(0.22, 0.055 * max(0, state_summary.stability_turns))
    exhausted_penalty = 0.06 if remaining_budget <= 1 else 0.0
    recent_gain = 0.0
    if episode.diagnosis_history:
        previous_margin = float(episode.diagnosis_history[-1].deterministic_margin)
        recent_gain = max(0.0, float(state_summary.margin) - previous_margin)
    marginal_value = (
        0.52 * shortlist_value
        + 0.26 * ambiguity
        + 0.22 * unresolved
        + 0.18 * recent_gain
        - stability_penalty
        - exhausted_penalty
    )
    return {
        "marginal_value": float(np.clip(marginal_value, 0.0, 1.0)),
        "top_shortlist_score": top_shortlist_score,
        "ambiguity": float(ambiguity),
        "unresolved_mass": unresolved,
        "recent_margin_gain": float(recent_gain),
        "stability_penalty": float(stability_penalty),
    }


def build_cost_sensitive_stop_signal(
    episode: LedgerEpisodeState,
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    remaining_budget: int,
    evidence_cost_lambda: float,
) -> dict[str, Any]:
    base = build_stop_signal(episode, state_summary, shortlist, remaining_budget)
    value_parts = estimate_marginal_evidence_value(episode, state_summary, shortlist, remaining_budget)
    utility_continue = value_parts["marginal_value"] - float(evidence_cost_lambda)
    signal = {
        **base,
        **value_parts,
        "evidence_cost_lambda": float(evidence_cost_lambda),
        "utility_continue": float(utility_continue),
        "max_request_cap": int(MAX_REQUEST_CAP),
    }
    if not shortlist:
        signal["level"] = "strong_stop"
        signal["reason"] = "No legal evidence requests remain."
        return signal
    if (
        len(episode.request_history) >= MIN_REQUESTS_BEFORE_COST_STOP
        and utility_continue <= COST_FORCE_STOP_UTILITY
        and (state_summary.margin >= 0.08 or state_summary.stability_turns >= 1)
    ):
        signal["level"] = "strong_stop"
        signal["reason"] = (
            f"Estimated marginal evidence value ({value_parts['marginal_value']:.3f}) is below "
            f"lambda cost ({evidence_cost_lambda:.3f}) and the current differential is sufficiently stable."
        )
        return signal
    if utility_continue <= 0 and len(episode.request_history) >= MIN_REQUESTS_BEFORE_COST_STOP:
        signal["level"] = "moderate_stop"
        signal["reason"] = (
            f"Another request has weak expected utility: marginal={value_parts['marginal_value']:.3f}, "
            f"lambda={evidence_cost_lambda:.3f}."
        )
        return signal
    signal["level"] = "continue"
    signal["reason"] = (
        f"A shortlisted question still appears worth asking: marginal={value_parts['marginal_value']:.3f}, "
        f"lambda={evidence_cost_lambda:.3f}, utility={utility_continue:.3f}."
    )
    return signal


def reconcile_cost_sensitive_response(
    normalized_response: dict[str, Any],
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    stop_signal: dict[str, Any],
    remaining_budget: int,
) -> tuple[dict[str, Any], list[str]]:
    result, policy_flags = reconcile_agent_response(
        normalized_response=normalized_response,
        state_summary=state_summary,
        shortlist=shortlist,
        stop_signal=stop_signal,
        remaining_budget=remaining_budget,
    )
    utility_continue = float(stop_signal.get("utility_continue", 0.0))
    if (
        result["decision"] == "request"
        and stop_signal.get("level") in {"strong_stop", "moderate_stop"}
        and utility_continue <= 0
    ):
        result["decision"] = "stop"
        result["requested_evidence_id"] = None
        policy_flags.append("cost_sensitive_forced_stop")
    if (
        result["decision"] == "stop"
        and shortlist
        and remaining_budget > 1
        and utility_continue >= COST_FORCE_REQUEST_MARGIN
        and state_summary.margin < 0.20
    ):
        result["decision"] = "request"
        result["requested_evidence_id"] = shortlist[0]["root_evidence_id"]
        result["confidence"] = min(float(result["confidence"]), 0.56)
        policy_flags.append("cost_sensitive_forced_request")
    return result, policy_flags

# %% [markdown]
# ## Selected MLP-Guided Stop And Final-Head Logic

# %%
# Live-confirmation stop/reconciliation/final-head rules.


def selected_mlp_stop_rule_fired(episode: LedgerEpisodeState, mlp_feedback: dict[str, Any]) -> bool:
    return bool(
        len(episode.request_history) >= SELECTED_STOP_POLICY["min_requests"]
        and float(mlp_feedback["confidence"]) >= SELECTED_STOP_POLICY["mlp_confidence_min"]
        and float(mlp_feedback["margin"]) >= SELECTED_STOP_POLICY["mlp_margin_min"]
        and float(mlp_feedback["entropy"]) <= SELECTED_STOP_POLICY["mlp_entropy_max"]
        and int(mlp_feedback["stability_turns"]) >= SELECTED_STOP_POLICY["mlp_stability_min"]
    )


def build_selected_stop_signal(
    episode: LedgerEpisodeState,
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    remaining_budget: int,
    mlp_feedback: dict[str, Any],
) -> dict[str, Any]:
    base = build_stop_signal(episode, state_summary, shortlist, remaining_budget)
    selected_fired = selected_mlp_stop_rule_fired(episode, mlp_feedback)
    signal = {
        **base,
        "selected_policy_name": SELECTED_STOP_POLICY["policy_name"],
        "selected_stop_rule_fired": bool(selected_fired),
        "selected_min_requests": int(SELECTED_STOP_POLICY["min_requests"]),
        "mlp_top1": mlp_feedback["top1"],
        "mlp_top5": list(mlp_feedback["top_predictions"]),
        "mlp_confidence": float(mlp_feedback["confidence"]),
        "mlp_margin": float(mlp_feedback["margin"]),
        "mlp_entropy": float(mlp_feedback["entropy"]),
        "mlp_stability_turns": int(mlp_feedback["stability_turns"]),
        "mlp_confidence_min": float(SELECTED_STOP_POLICY["mlp_confidence_min"]),
        "mlp_margin_min": float(SELECTED_STOP_POLICY["mlp_margin_min"]),
        "mlp_entropy_max": float(SELECTED_STOP_POLICY["mlp_entropy_max"]),
        "mlp_stability_min": int(SELECTED_STOP_POLICY["mlp_stability_min"]),
    }
    if not shortlist:
        signal["level"] = "strong_stop"
        signal["reason"] = "No legal evidence requests remain."
    elif selected_fired:
        signal["level"] = "hybrid_mlp_stop"
        signal["reason"] = (
            "Selected notebook 12 stop rule fired: "
            f"requests={len(episode.request_history)}, "
            f"mlp_conf={mlp_feedback['confidence']:.3f}, "
            f"mlp_margin={mlp_feedback['margin']:.3f}, "
            f"mlp_entropy={mlp_feedback['entropy']:.3f}, "
            f"mlp_stability={mlp_feedback['stability_turns']}."
        )
    else:
        signal["level"] = "continue"
        signal["reason"] = (
            "Selected MLP stop rule has not fired; continue if a shortlisted question is useful. "
            f"Current MLP conf={mlp_feedback['confidence']:.3f}, margin={mlp_feedback['margin']:.3f}, "
            f"entropy={mlp_feedback['entropy']:.3f}, requests={len(episode.request_history)}."
        )
    return signal


def agreement_hybrid_prediction(llm_pred: str, llm_ranked: list[str], mlp_pred: str, mlp_ranked: list[str]) -> tuple[str, list[str], str]:
    if llm_pred == mlp_pred:
        final = llm_pred
        source = "agreement"
    else:
        final = llm_pred
        source = "fallback_llm_disagreement"
    merged: list[str] = []
    for label in [final, *llm_ranked, *mlp_ranked]:
        if label and label not in merged:
            merged.append(label)
    return final, merged[:5], source


def conservative_hybrid_prediction(
    llm_pred: str,
    llm_ranked: list[str],
    mlp_pred: str,
    mlp_ranked: list[str],
    mlp_feedback: dict[str, Any],
    llm_confidence: float,
) -> tuple[str, list[str], str]:
    if llm_pred == mlp_pred:
        final = llm_pred
        source = "llm_mlp_agreement"
    elif mlp_feedback["confidence"] >= 0.97 and mlp_feedback["margin"] >= 0.90 and llm_confidence < 0.70:
        final = mlp_pred
        source = "strict_mlp_override"
    else:
        final = llm_pred
        source = "prefer_llm_disagreement"
    merged: list[str] = []
    for label in [final, *llm_ranked, *mlp_ranked]:
        if label and label not in merged:
            merged.append(label)
    return final, merged[:5], source


def reconcile_selected_stop_response(
    normalized_response: dict[str, Any],
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    remaining_budget: int,
) -> tuple[dict[str, Any], list[str]]:
    result, policy_flags = reconcile_agent_response(
        normalized_response=normalized_response,
        state_summary=state_summary,
        shortlist=shortlist,
        stop_signal=stop_signal,
        remaining_budget=remaining_budget,
    )
    if result["decision"] == "request" and stop_signal.get("level") == "hybrid_mlp_stop":
        result["decision"] = "stop"
        result["requested_evidence_id"] = None
        policy_flags.append("selected_mlp_stop_forced_stop")
    return result, policy_flags

# %% [markdown]
# ## Preview Selected-Stop Hybrid Prompt

# %%
preview_row = benchmark_df.iloc[0].to_dict()
preview_episode = ledger.from_row(preview_row, split_name=SPLIT_NAME, prior_differential=prior_lookup.get(preview_row["case_id"], []))
preview_legal = ledger.legal_actions(preview_episode)
preview_state = diagnosis_manager.summarize_state(preview_episode)
preview_shortlist, _ = shortlister.shortlist(preview_episode, preview_legal, preview_state)
preview_mlp = compute_mlp_feedback(preview_episode, mlp_schema, partial_mlp_model, partial_mlp_labels, [])
preview_stop = build_selected_stop_signal(preview_episode, preview_state, preview_shortlist, MAX_REQUEST_CAP, preview_mlp)
print(build_hybrid_user_prompt(ledger, preview_episode, preview_shortlist[:5], preview_state, preview_stop, preview_mlp, 1, MAX_REQUEST_CAP)[:5000])

# %% [markdown]
# ## Graph/Bayes Branch Ledger
# 
# The branch trigger and judge use only current revealed evidence. Graph support comes from Notebook `16`; Bayesian likelihoods and priors come from Notebook `19`. Both are train/validate-derived artifacts and do not use the 49-case labels for decisions.

# %%
GRAPH_ROOT = ROOT / "artifacts" / "graph_algorithmic_ledger" / "medkgi_style_offline_notebook13_49case_v1"
BAYES_ROOT = ROOT / "artifacts" / "bayesian_voi_ledger" / "bayesian_voi_offline_notebook13_49case_v1"
BRANCH_LAB_ROOT = ROOT / "artifacts" / "trajectory_replicates" / "offline_branching_trajectory_lab_49case_v1"
GRAPH_EDGE_PATH = GRAPH_ROOT / "global_evidence_graph_edges.csv"
BAYES_LIKELIHOOD_PATH = BAYES_ROOT / "root_outcome_likelihoods.csv"
BAYES_PRIOR_PATH = BAYES_ROOT / "diagnosis_priors.csv"
RECOMMENDED_BRANCH_POLICY_PATH = BRANCH_LAB_ROOT / "recommended_branching_policy.json"

for required_path in [GRAPH_EDGE_PATH, BAYES_LIKELIHOOD_PATH, BAYES_PRIOR_PATH, RECOMMENDED_BRANCH_POLICY_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

GRAPH_CLIP = 3.0
EPS = 1e-9
ALL_PATHOLOGIES = list(partial_mlp_labels)
PATHOLOGY_TO_INDEX = {label: idx for idx, label in enumerate(ALL_PATHOLOGIES)}

recommended_branching_policy = load_json(RECOMMENDED_BRANCH_POLICY_PATH)
graph_edges = pd.read_csv(GRAPH_EDGE_PATH)
graph_edges = graph_edges[graph_edges["outcome_state"].isin(["present", "absent"])].copy()
graph_lookup = {
    (str(row.root_evidence_id), str(row.outcome_state), str(row.pathology)): float(np.clip(row.log_odds_support, -GRAPH_CLIP, GRAPH_CLIP))
    for row in graph_edges.itertuples()
}

bayes_likelihoods = pd.read_csv(BAYES_LIKELIHOOD_PATH)
bayes_likelihoods = bayes_likelihoods[bayes_likelihoods["outcome_state"].isin(["__ABSENT__", "__PRESENT__"])].copy()
bayes_index = bayes_likelihoods.set_index(["root_evidence_id", "outcome_state"])
bayes_priors_df = pd.read_csv(BAYES_PRIOR_PATH)
bayes_priors = dict(zip(bayes_priors_df["pathology"], bayes_priors_df["prior"]))


def softmax_dict(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    keys = list(scores)
    values = np.array([scores[key] for key in keys], dtype=float)
    values = values - np.max(values)
    exp = np.exp(values)
    probs = exp / max(float(exp.sum()), EPS)
    return dict(zip(keys, probs))


def rank_of(scores: dict[str, float], label: str) -> int:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for idx, (candidate, _) in enumerate(ordered, start=1):
        if candidate == label:
            return idx
    return len(ordered) + 1


def top_margin(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    return float(values[0] - values[1]) if len(values) >= 2 else 0.0


def episode_visible_evidence(episode: LedgerEpisodeState) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for entry in episode.evidence_ledger:
        evidence[str(entry.root_evidence_id)] = str(entry.status)
    return evidence


def bayes_state(status: str) -> str | None:
    if status == "present":
        return "__PRESENT__"
    if status == "absent":
        return "__ABSENT__"
    return None


def compute_graph_scores(evidence: dict[str, str]) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    scores = {pathology: 0.0 for pathology in ALL_PATHOLOGIES}
    positive = {pathology: 0.0 for pathology in ALL_PATHOLOGIES}
    contradiction = {pathology: 0.0 for pathology in ALL_PATHOLOGIES}
    for root_id, status in evidence.items():
        if status not in {"present", "absent"}:
            continue
        for pathology in ALL_PATHOLOGIES:
            value = float(graph_lookup.get((root_id, status, pathology), 0.0))
            scores[pathology] += value
            if value >= 0:
                positive[pathology] += value
            else:
                contradiction[pathology] += -value
    return scores, positive, contradiction, softmax_dict(scores)


def compute_bayes_scores(evidence: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    scores = {pathology: math.log(max(float(bayes_priors.get(pathology, EPS)), EPS)) for pathology in ALL_PATHOLOGIES}
    for root_id, status in evidence.items():
        state = bayes_state(status)
        if state is None or (root_id, state) not in bayes_index.index:
            continue
        likelihood_row = bayes_index.loc[(root_id, state)]
        if isinstance(likelihood_row, pd.DataFrame):
            likelihood_row = likelihood_row.iloc[0]
        for pathology in ALL_PATHOLOGIES:
            col = f"p__{pathology}"
            if col in likelihood_row.index:
                scores[pathology] += math.log(max(float(likelihood_row[col]), EPS))
    return scores, softmax_dict(scores)


def mlp_probs_for_episode(episode: LedgerEpisodeState) -> np.ndarray:
    features = encode_episode_state_for_mlp(episode, mlp_schema)
    with torch.no_grad():
        logits = partial_mlp_model(torch.tensor(features[None, :], dtype=torch.float32))
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def top_labels_from_scores(scores: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


def graph_bayes_context_text(episode: LedgerEpisodeState, predicted_pathology: str | None = None, k: int = 5) -> str:
    evidence = episode_visible_evidence(episode)
    graph_scores, _, _, graph_post = compute_graph_scores(evidence)
    bayes_scores, bayes_post = compute_bayes_scores(evidence)
    graph_top = top_labels_from_scores(graph_scores, k)
    bayes_top = top_labels_from_scores(bayes_post, k)
    lines_out = [
        "Raw graph/Bayes ledger context from currently revealed evidence:",
        "- graph_top: " + json.dumps([{"pathology": p, "score": round(s, 3)} for p, s in graph_top]),
        "- bayes_top: " + json.dumps([{"pathology": p, "posterior": round(v, 4)} for p, v in bayes_top]),
    ]
    if predicted_pathology in ALL_PATHOLOGIES:
        lines_out.append(f"- current_prediction_graph_rank: {rank_of(graph_scores, predicted_pathology)}")
        lines_out.append(f"- current_prediction_bayes_rank: {rank_of(bayes_post, predicted_pathology)}")
        lines_out.append(f"- current_prediction_bayes_posterior: {bayes_post.get(predicted_pathology, 0.0):.4f}")
    return "\n".join(lines_out)

print("Graph status edges:", len(graph_edges))
print("Bayes status likelihood rows:", len(bayes_likelihoods))
print("Notebook 26 recommended policy:", recommended_branching_policy.get("pre_registered_next_live_candidate", {}).get("policy_name"))

# %% [markdown]
# ## Train Branch Trigger MLP And Candidate Resolver

# %%

# Train/validate learned branch gate and candidate resolver.
# No 49-case labels are used here.

TRAIN_SPLIT = attach_split_metadata(load_patient_split(dataset_paths["release_train_patients.zip"]), "train")
VALIDATE_SPLIT = attach_split_metadata(load_patient_split(dataset_paths["release_validate_patients.zip"]), "validate")
if SYNTHETIC_MAX_SOURCE_ROWS:
    TRAIN_SPLIT = TRAIN_SPLIT.sample(n=min(SYNTHETIC_MAX_SOURCE_ROWS, len(TRAIN_SPLIT)), random_state=RANDOM_SEED).reset_index(drop=True)
    VALIDATE_SPLIT = VALIDATE_SPLIT.sample(n=min(SYNTHETIC_MAX_SOURCE_ROWS, len(VALIDATE_SPLIT)), random_state=RANDOM_SEED + 1).reset_index(drop=True)

ROOT_INFO = pd.read_csv(BAYES_ROOT / "root_information_stats.csv") if (BAYES_ROOT / "root_information_stats.csv").exists() else pd.DataFrame()
ROOT_MI = dict(zip(ROOT_INFO.get("root_evidence_id", []), ROOT_INFO.get("root_mi_norm", [])))
ALL_ROOTS = list(evidences.keys())
BRANCH_FEATURE_COLUMNS: list[str] = []
RESOLVER_FEATURE_COLUMNS: list[str] = []
PAIR_UTILITY_CACHE: dict[tuple[str, str], list[tuple[str, float]]] = {}


def evidence_tokens(raw: Any) -> list[str]:
    return [str(item) for item in safe_parse_list(raw)]


def token_to_root_value(token: str) -> tuple[str, str | None]:
    if "_@_" in token:
        root, value = token.split("_@_", 1)
        return root, value
    return token, None


def row_root_values(row: dict[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for token in evidence_tokens(row.get("EVIDENCES", [])):
        root, value = token_to_root_value(token)
        if value is None:
            values[root].append("1")
        else:
            values[root].append(value)
    init = str(row.get("INITIAL_EVIDENCE", ""))
    if init and init not in values:
        values[init].append("1")
    return dict(values)


def evidence_from_roots(row: dict[str, Any], visible_roots: set[str]) -> dict[str, str]:
    present = set(row_root_values(row))
    return {root: ("present" if root in present else "absent") for root in visible_roots if root in evidences}


def mlp_probs_for_row_evidence(row: dict[str, Any], evidence: dict[str, str]) -> np.ndarray:
    state = mlp_schema.initial_state(int(row["AGE"]), str(row["SEX"]))
    values_by_root = row_root_values(row)
    for root_id, status in evidence.items():
        state = mlp_schema.apply_root_observation(state, root_id, values_by_root.get(root_id, []) if status == "present" else [])
    with torch.no_grad():
        logits = partial_mlp_model(torch.tensor(state[None, :], dtype=torch.float32))
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def ranked_from_probs(probs: np.ndarray) -> tuple[list[str], dict[str, int]]:
    order = list(np.argsort(-probs))
    labels = [partial_mlp_labels[idx] for idx in order]
    return labels, {label: rank for rank, label in enumerate(labels, start=1)}


def entropy01(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs / max(float(probs.sum()), EPS)
    return float(-(probs * np.log(np.clip(probs, EPS, 1.0))).sum() / math.log(len(probs)))


def binary_root_prob(root: str, pathology: str, state: str) -> float:
    key = (root, state)
    if key not in bayes_index.index:
        return 0.5
    row = bayes_index.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return float(row.get(f"p__{pathology}", 0.5))


def js_binary_for_root(root: str, disease_a: str, disease_b: str) -> float:
    pa = np.array([binary_root_prob(root, disease_a, "__PRESENT__"), binary_root_prob(root, disease_a, "__ABSENT__")], dtype=float)
    pb = np.array([binary_root_prob(root, disease_b, "__PRESENT__"), binary_root_prob(root, disease_b, "__ABSENT__")], dtype=float)
    pa = pa / max(pa.sum(), EPS)
    pb = pb / max(pb.sum(), EPS)
    m = 0.5 * (pa + pb)
    kl_a = float((pa * np.log(np.clip(pa / np.clip(m, EPS, None), EPS, None))).sum())
    kl_b = float((pb * np.log(np.clip(pb / np.clip(m, EPS, None), EPS, None))).sum())
    return 0.5 * (kl_a + kl_b)


def pair_root_utilities(disease_a: str, disease_b: str) -> list[tuple[str, float]]:
    key = tuple(sorted([str(disease_a), str(disease_b)]))
    if key in PAIR_UTILITY_CACHE:
        return PAIR_UTILITY_CACHE[key]
    values = []
    if disease_a == disease_b:
        PAIR_UTILITY_CACHE[key] = []
        return []
    for root in ALL_ROOTS:
        utility = js_binary_for_root(root, disease_a, disease_b) * float(ROOT_MI.get(root, 1.0))
        if np.isfinite(utility) and utility > 0:
            values.append((root, float(utility)))
    values.sort(key=lambda item: item[1], reverse=True)
    PAIR_UTILITY_CACHE[key] = values
    return values


def pair_coverage(disease_a: str, disease_b: str, observed_roots: set[str], top_k: int = PAIR_TOP_ROOTS) -> tuple[float, float, str | None, list[str]]:
    utilities = pair_root_utilities(disease_a, disease_b)[:top_k]
    denom = float(sum(value for _, value in utilities))
    if denom <= EPS:
        return 1.0, 0.0, None, []
    observed = float(sum(value for root, value in utilities if root in observed_roots))
    missing = [root for root, _ in utilities if root not in observed_roots]
    return float(observed / denom), float(denom - observed), (missing[0] if missing else None), missing[:5]


def state_candidates(evidence: dict[str, str], mlp_probs: np.ndarray) -> dict[str, Any]:
    graph_scores, graph_pos, graph_contra, graph_post = compute_graph_scores(evidence)
    bayes_scores, bayes_post = compute_bayes_scores(evidence)
    mlp_labels, mlp_ranks = ranked_from_probs(mlp_probs)
    graph_order = [label for label, _ in top_labels_from_scores(graph_scores, 10)]
    bayes_order = [label for label, _ in top_labels_from_scores(bayes_post, 10)]
    anchor = mlp_labels[0]
    if float(max(bayes_post.values())) > 0.70:
        anchor = max(bayes_post, key=bayes_post.get)
    elif top_margin(graph_scores) > 2.5:
        anchor = graph_order[0]
    return {
        "anchor": anchor,
        "graph_scores": graph_scores,
        "graph_pos": graph_pos,
        "graph_contra": graph_contra,
        "graph_post": graph_post,
        "graph_order": graph_order,
        "bayes_scores": bayes_scores,
        "bayes_post": bayes_post,
        "bayes_order": bayes_order,
        "mlp_probs": mlp_probs,
        "mlp_order": mlp_labels,
        "mlp_ranks": mlp_ranks,
    }


def branch_feature_dict(row: dict[str, Any], evidence: dict[str, str], request_count: int, anchor: str | None = None) -> dict[str, float | str | bool]:
    bundle = state_candidates(evidence, mlp_probs_for_row_evidence(row, evidence))
    anchor = str(anchor or bundle["anchor"])
    graph_scores = bundle["graph_scores"]
    bayes_post = bundle["bayes_post"]
    mlp_probs = bundle["mlp_probs"]
    mlp_order = bundle["mlp_order"]
    top_candidates = []
    for label in [anchor, *bundle["graph_order"][:3], *bundle["bayes_order"][:3], *mlp_order[:3]]:
        if label not in top_candidates:
            top_candidates.append(label)
    challenger = next((label for label in top_candidates if label != anchor), anchor)
    coverage, missing_mass, top_missing_root, missing_roots = pair_coverage(anchor, challenger, set(evidence))
    mlp_idx = PATHOLOGY_TO_INDEX.get(anchor)
    top_prob = float(mlp_probs[np.argsort(-mlp_probs)[0]])
    second_prob = float(mlp_probs[np.argsort(-mlp_probs)[1]])
    return {
        "anchor_prediction": anchor,
        "graph_top1": bundle["graph_order"][0],
        "bayes_top1": bundle["bayes_order"][0],
        "mlp_top1": mlp_order[0],
        "challenger_prediction": challenger,
        "request_count": float(request_count),
        "visible_root_count": float(len(evidence)),
        "graph_top_score": float(graph_scores[bundle["graph_order"][0]]),
        "graph_margin": top_margin(graph_scores),
        "anchor_graph_score": float(graph_scores.get(anchor, 0.0)),
        "anchor_graph_rank": float(rank_of(graph_scores, anchor)),
        "bayes_top_posterior": float(bayes_post[bundle["bayes_order"][0]]),
        "bayes_margin": top_margin(bayes_post),
        "anchor_bayes_posterior": float(bayes_post.get(anchor, 0.0)),
        "anchor_bayes_rank": float(rank_of(bayes_post, anchor)),
        "mlp_confidence": top_prob,
        "mlp_margin": top_prob - second_prob,
        "mlp_entropy": entropy01(mlp_probs),
        "anchor_mlp_posterior": float(mlp_probs[mlp_idx]) if mlp_idx is not None else 0.0,
        "anchor_mlp_rank": float(bundle["mlp_ranks"].get(anchor, len(ALL_PATHOLOGIES) + 1)),
        "graph_bayes_agree": float(bundle["graph_order"][0] == bundle["bayes_order"][0]),
        "all_heads_anchor": float(anchor == bundle["graph_order"][0] == bundle["bayes_order"][0] == mlp_order[0]),
        "ledger_disagrees_anchor": float(bundle["graph_order"][0] != anchor and bundle["bayes_order"][0] != anchor),
        "pair_coverage": float(coverage),
        "pair_missing_utility": float(missing_mass),
        "top_missing_pair_root": top_missing_root or "",
        "missing_pair_roots": json.dumps(missing_roots),
    }


def sample_visible_roots(row: dict[str, Any], rng: np.random.Generator) -> set[str]:
    present_roots = set(row_root_values(row))
    initial = str(row.get("INITIAL_EVIDENCE", ""))
    sample_n = int(rng.choice(np.arange(1, 13)))
    candidate_roots = list(set(ALL_ROOTS) - {initial})
    requested = set(rng.choice(candidate_roots, size=min(sample_n, len(candidate_roots)), replace=False))
    if rng.random() < 0.65 and present_roots:
        positives = list(present_roots - {initial})
        if positives:
            requested.update(rng.choice(positives, size=min(2, len(positives)), replace=False))
    return {initial, *requested} if initial else requested


def build_synthetic_rows(split_df: pd.DataFrame, split_name: str, n_states: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch_rows = []
    resolver_rows = []
    records = split_df.to_dict(orient="records")
    for idx in tqdm(range(n_states), desc=f"Synthetic {split_name} branch states"):
        row = dict(records[int(rng.integers(0, len(records)))])
        true_pathology = str(row["PATHOLOGY"])
        visible_roots = sample_visible_roots(row, rng)
        evidence = evidence_from_roots(row, visible_roots)
        request_count = max(0, len(evidence) - 1)
        feats = branch_feature_dict(row, evidence, request_count)
        y_branch = int(str(feats["anchor_prediction"]) != true_pathology)
        branch_rows.append({
            "split": split_name,
            "synthetic_state_id": f"{split_name}:{idx}",
            "true_pathology": true_pathology,
            "branch_label": y_branch,
            **feats,
        })
        bundle = state_candidates(evidence, mlp_probs_for_row_evidence(row, evidence))
        candidate_pool = []
        for label in [feats["anchor_prediction"], *bundle["graph_order"][:5], *bundle["bayes_order"][:5], *bundle["mlp_order"][:5]]:
            if label not in candidate_pool:
                candidate_pool.append(label)
        for order, candidate in enumerate(candidate_pool):
            cfeats = candidate_feature_dict_from_bundle(bundle, evidence, candidate, feats, source="synthetic_candidate", order=order)
            resolver_rows.append({
                "split": split_name,
                "synthetic_state_id": f"{split_name}:{idx}",
                "true_pathology": true_pathology,
                "candidate_pathology": candidate,
                "candidate_label": int(candidate == true_pathology),
                **cfeats,
            })
    return pd.DataFrame(branch_rows), pd.DataFrame(resolver_rows)


def candidate_feature_dict_from_bundle(bundle: dict[str, Any], evidence: dict[str, str], candidate: str, state_feats: dict[str, Any], source: str, order: int) -> dict[str, float | str]:
    graph_scores = bundle["graph_scores"]
    graph_post = bundle["graph_post"]
    graph_pos = bundle["graph_pos"]
    graph_contra = bundle["graph_contra"]
    bayes_post = bundle["bayes_post"]
    bayes_scores = bundle["bayes_scores"]
    mlp_probs = bundle["mlp_probs"]
    mlp_idx = PATHOLOGY_TO_INDEX.get(candidate)
    is_base = float(source == "base")
    is_branch = float(source == "branch")
    is_pseudo = float(source.startswith("pseudo"))
    return {
        "candidate_source": source,
        "candidate_order": float(order),
        "candidate_graph_score": float(graph_scores.get(candidate, 0.0)),
        "candidate_graph_posterior": float(graph_post.get(candidate, 0.0)),
        "candidate_graph_rank": float(rank_of(graph_scores, candidate)),
        "candidate_graph_positive_support": float(graph_pos.get(candidate, 0.0)),
        "candidate_graph_contradiction": float(graph_contra.get(candidate, 0.0)),
        "candidate_bayes_log_score": float(bayes_scores.get(candidate, -999.0)),
        "candidate_bayes_posterior": float(bayes_post.get(candidate, 0.0)),
        "candidate_bayes_rank": float(rank_of(bayes_post, candidate)),
        "candidate_mlp_posterior": float(mlp_probs[mlp_idx]) if mlp_idx is not None else 0.0,
        "candidate_mlp_rank": float(bundle["mlp_ranks"].get(candidate, len(ALL_PATHOLOGIES) + 1)),
        "is_base_candidate": is_base,
        "is_branch_candidate": is_branch,
        "is_pseudo_candidate": is_pseudo,
        "request_count": float(state_feats.get("request_count", 0.0)),
        "visible_root_count": float(state_feats.get("visible_root_count", len(evidence))),
        "branch_trigger_probability": float(state_feats.get("branch_trigger_probability", 0.0)),
        "pair_coverage": float(state_feats.get("pair_coverage", 1.0)),
        "pair_missing_utility": float(state_feats.get("pair_missing_utility", 0.0)),
    }


# candidate_feature_dict_from_bundle is referenced while synthetic rows are built.
# Python resolves it at call time, so defining it before invocation is enough.
train_branch_features, train_resolver_features = build_synthetic_rows(TRAIN_SPLIT, "train", SYNTH_TRAIN_STATES)
validate_branch_features, validate_resolver_features = build_synthetic_rows(VALIDATE_SPLIT, "validate", SYNTH_VALIDATE_STATES)
branch_feature_frame = pd.concat([train_branch_features, validate_branch_features], ignore_index=True)
resolver_feature_frame = pd.concat([train_resolver_features, validate_resolver_features], ignore_index=True)

BRANCH_FEATURE_COLUMNS = [
    "request_count", "visible_root_count", "graph_top_score", "graph_margin", "anchor_graph_score", "anchor_graph_rank",
    "bayes_top_posterior", "bayes_margin", "anchor_bayes_posterior", "anchor_bayes_rank",
    "mlp_confidence", "mlp_margin", "mlp_entropy", "anchor_mlp_posterior", "anchor_mlp_rank",
    "graph_bayes_agree", "all_heads_anchor", "ledger_disagrees_anchor", "pair_coverage", "pair_missing_utility",
]
RESOLVER_FEATURE_COLUMNS = [
    "candidate_order", "candidate_graph_score", "candidate_graph_posterior", "candidate_graph_rank",
    "candidate_graph_positive_support", "candidate_graph_contradiction", "candidate_bayes_log_score",
    "candidate_bayes_posterior", "candidate_bayes_rank", "candidate_mlp_posterior", "candidate_mlp_rank",
    "is_base_candidate", "is_branch_candidate", "is_pseudo_candidate", "request_count", "visible_root_count",
    "branch_trigger_probability", "pair_coverage", "pair_missing_utility",
]

branch_trigger_mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, max_iter=160, random_state=RANDOM_SEED, early_stopping=True),
)
train_mask = branch_feature_frame["split"].eq("train")
valid_mask = branch_feature_frame["split"].eq("validate")
branch_trigger_mlp.fit(branch_feature_frame.loc[train_mask, BRANCH_FEATURE_COLUMNS], branch_feature_frame.loc[train_mask, "branch_label"].astype(int))
validate_branch_prob = branch_trigger_mlp.predict_proba(branch_feature_frame.loc[valid_mask, BRANCH_FEATURE_COLUMNS])[:, 1]
validate_branch_features = branch_feature_frame.loc[valid_mask].copy()
validate_branch_features["branch_trigger_probability"] = validate_branch_prob

threshold_rows = []
for threshold in np.linspace(0.05, 0.95, 37):
    pred = validate_branch_prob >= threshold
    y = validate_branch_features["branch_label"].astype(bool).to_numpy()
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    branch_rate = float(pred.mean())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    objective = recall - 0.70 * (fp / max(fp + tn, 1)) - 0.05 * branch_rate
    threshold_rows.append({"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "branch_rate": branch_rate, "validation_objective": objective})
branch_threshold_sweep = pd.DataFrame(threshold_rows)
selected_threshold_row = branch_threshold_sweep.sort_values(["validation_objective", "threshold"], ascending=[False, False]).iloc[0].to_dict()
BRANCH_TRIGGER_THRESHOLD = float(ADAPTIVE_BRANCH_TRIGGER_THRESHOLD)
selected_threshold_row = dict(selected_threshold_row)
selected_threshold_row["threshold"] = BRANCH_TRIGGER_THRESHOLD
matched_threshold_rows = branch_threshold_sweep[np.isclose(branch_threshold_sweep["threshold"].astype(float), BRANCH_TRIGGER_THRESHOLD)]
if len(matched_threshold_rows):
    selected_threshold_row.update(matched_threshold_rows.iloc[0].to_dict())

resolver_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=RANDOM_SEED))
resolver_train = resolver_feature_frame["split"].eq("train")
resolver_model.fit(resolver_feature_frame.loc[resolver_train, RESOLVER_FEATURE_COLUMNS], resolver_feature_frame.loc[resolver_train, "candidate_label"].astype(int))
resolver_valid = resolver_feature_frame["split"].eq("validate")
resolver_validate_scores = resolver_model.predict_proba(resolver_feature_frame.loc[resolver_valid, RESOLVER_FEATURE_COLUMNS])[:, 1]
resolver_validation_summary = pd.DataFrame([{
    "validate_candidate_rows": int(resolver_valid.sum()),
    "validate_positive_rate": float(resolver_feature_frame.loc[resolver_valid, "candidate_label"].mean()),
    "candidate_auc": float(roc_auc_score(resolver_feature_frame.loc[resolver_valid, "candidate_label"], resolver_validate_scores)),
    "candidate_average_precision": float(average_precision_score(resolver_feature_frame.loc[resolver_valid, "candidate_label"], resolver_validate_scores)),
}])

branch_auc = float(roc_auc_score(validate_branch_features["branch_label"], validate_branch_prob))
branch_ap = float(average_precision_score(validate_branch_features["branch_label"], validate_branch_prob))
branch_brier = float(brier_score_loss(validate_branch_features["branch_label"], validate_branch_prob))
branch_validation_summary = pd.DataFrame([{
    "train_states": int(train_mask.sum()),
    "validate_states": int(valid_mask.sum()),
    "branch_label_rate_train": float(branch_feature_frame.loc[train_mask, "branch_label"].mean()),
    "branch_label_rate_validate": float(validate_branch_features["branch_label"].mean()),
    "selected_threshold": BRANCH_TRIGGER_THRESHOLD,
    "selected_threshold_precision": float(selected_threshold_row["precision"]),
    "selected_threshold_recall": float(selected_threshold_row["recall"]),
    "selected_threshold_branch_rate": float(selected_threshold_row["branch_rate"]),
    "auc": branch_auc,
    "average_precision": branch_ap,
    "brier_score": branch_brier,
}])

# Persist model-selection artifacts immediately, even if live execution is skipped.
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
branch_feature_frame.to_csv(ARTIFACT_ROOT / "branch_mlp_train_validate_features.csv", index=False)
resolver_feature_frame.to_csv(ARTIFACT_ROOT / "candidate_resolver_train_validate_features.csv", index=False)
branch_validation_summary.to_csv(ARTIFACT_ROOT / "branch_mlp_validation_summary.csv", index=False)
branch_threshold_sweep.to_csv(ARTIFACT_ROOT / "branch_mlp_threshold_sweep.csv", index=False)
resolver_validation_summary.to_csv(ARTIFACT_ROOT / "candidate_resolver_validation_summary.csv", index=False)

pair_rows = []
for key, utilities in list(PAIR_UTILITY_CACHE.items())[:2000]:
    disease_a, disease_b = key
    for rank, (root, utility) in enumerate(utilities[:PAIR_TOP_ROOTS], start=1):
        pair_rows.append({"disease_a": disease_a, "disease_b": disease_b, "root_evidence_id": root, "pair_root_rank": rank, "pair_root_utility": utility})
pairwise_evidence_separation_graph = pd.DataFrame(pair_rows)
pairwise_evidence_separation_graph.to_csv(ARTIFACT_ROOT / "pairwise_evidence_separation_graph.csv", index=False)

print("Branch MLP validation summary")
display(branch_validation_summary)
print("Resolver validation summary")
display(resolver_validation_summary)
print("Selected branch threshold:", BRANCH_TRIGGER_THRESHOLD)

# %% [markdown]
# ## Branch Prompt, Hypothesis Assignment, And Resolver
# 
# Branches are logical agents, not persistent chat sessions. Each branch starts from a fresh message list and the original deterministic ledger state, but the coordinator now assigns an explicit target hypothesis before the branch starts. The target is computed only from visible base-branch evidence using graph, Bayes, MLP, and ranked-differential signals.
# 
# The branch prompt deliberately avoids carrying over the base branch's free-text reasoning; it passes structured facts only: base final prediction, assigned challenger diagnosis, discriminator roots, graph/Bayes support summaries, and base-requested roots to avoid early when possible.

# %%
def branch_role_section(branch_profile: dict[str, Any] | None, base_result: dict[str, Any] | None, episode: LedgerEpisodeState) -> str:
    if not branch_profile:
        return ""
    base_prediction = base_result.get("predicted_pathology") if base_result else None
    base_requested_roots = list((base_result or {}).get("requested_roots", []))
    avoid_roots = list(branch_profile.get("avoid_roots", []))
    preferred_roots = list(branch_profile.get("preferred_roots", []))
    target_hypothesis = branch_profile.get("target_hypothesis")
    support_summary = branch_profile.get("support_summary", {})
    return "\n".join([
        "",
        "Independent hypothesis-forced branch role:",
        f"- branch_id: {branch_profile['branch_id']}",
        f"- role_name: {branch_profile['role_name']}",
        f"- assigned_target_hypothesis: {target_hypothesis}",
        f"- base_branch_final_prediction_for_counterchecking: {base_prediction}",
        f"- directive: {branch_profile['directive']}",
        "- You are a fresh diagnostic branch. Do not rely on another branch's hidden reasoning.",
        "- You may reject the assigned hypothesis, but first give it a fair evidence-seeking test.",
        "- Use only the decoded evidence ledger, deterministic state, MLP monitor, shortlist, graph/Bayes context, and structured branch assignment shown here.",
        f"- target_discriminator_roots_preferred_if_legal: {json.dumps(preferred_roots)}",
        f"- base_branch_early_requested_roots_to_avoid_if_reasonable: {json.dumps(avoid_roots)}",
        f"- base_branch_all_requested_roots: {json.dumps(base_requested_roots)}",
        f"- target_support_summary_from_base_visible_evidence: {json.dumps(support_summary, sort_keys=True)}",
        "- During early turns, prefer a target discriminator root if it appears in the legal shortlist; otherwise choose the best legal root that separates the assigned hypothesis from the base answer.",
        "",
        graph_bayes_context_text(episode, predicted_pathology=base_prediction),
    ])


def build_branch_hybrid_user_prompt(
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
    branch_profile: dict[str, Any] | None = None,
    base_result: dict[str, Any] | None = None,
) -> str:
    return build_hybrid_user_prompt(
        ledger=ledger,
        episode=episode,
        shortlist=shortlist,
        state_summary=state_summary,
        stop_signal=stop_signal,
        mlp_feedback=mlp_feedback,
        turn_index=turn_index,
        max_budget=max_budget,
    ) + branch_role_section(branch_profile, base_result, episode)


def build_branch_hybrid_messages(
    label_names: list[str],
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
    branch_profile: dict[str, Any] | None = None,
    base_result: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(label_names)},
        {
            "role": "user",
            "content": build_branch_hybrid_user_prompt(
                ledger=ledger,
                episode=episode,
                shortlist=shortlist,
                state_summary=state_summary,
                stop_signal=stop_signal,
                mlp_feedback=mlp_feedback,
                turn_index=turn_index,
                max_budget=max_budget,
                branch_profile=branch_profile,
                base_result=base_result,
            ),
        },
    ]


def choose_divergent_shortlist_item(shortlist: list[dict[str, Any]], avoid_roots: set[str], preferred_roots: list[str] | None = None) -> dict[str, Any] | None:
    if not shortlist:
        return None
    shortlist_by_root = {item["root_evidence_id"]: item for item in shortlist}
    for root_id in preferred_roots or []:
        if root_id in shortlist_by_root and root_id not in avoid_roots:
            return shortlist_by_root[root_id]
    for item in shortlist:
        if item["root_evidence_id"] not in avoid_roots:
            return item
    return shortlist[0]


def scripted_branch_dry_run_response(
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
    label_names: list[str],
    branch_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if branch_profile and turn_index <= BRANCH_DIVERGENCE_TURNS and shortlist:
        item = choose_divergent_shortlist_item(shortlist, set(branch_profile.get("avoid_roots", [])), list(branch_profile.get("preferred_roots", [])))
        ranked = list(mlp_feedback.get("top_predictions", [])) or [pathology for pathology, _ in state_summary.top_candidates[:5]] or list(label_names[:5])
        target = branch_profile.get("target_hypothesis") if branch_profile else None
        if target in label_names and target not in ranked[:1]:
            ranked = [target] + [label for label in ranked if label != target]
        return {
            "decision": "request",
            "requested_evidence_id": item["root_evidence_id"],
            "predicted_pathology": ranked[0],
            "ranked_differential": ranked[:5],
            "confidence": min(float(mlp_feedback.get("confidence", 0.5)), 0.62),
            "brief_reasoning": "Dry-run branch requests a target-hypothesis discriminator root.",
        }
    return scripted_hybrid_dry_run_response(episode, shortlist, state_summary, stop_signal, mlp_feedback, turn_index, max_budget, label_names)


def get_branch_agent_response_with_repair(
    label_names: list[str],
    ledger: DeterministicEvidenceLedger,
    episode: LedgerEpisodeState,
    shortlist: list[dict[str, Any]],
    state_summary: DiagnosticStateSummary,
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    turn_index: int,
    max_budget: int,
    branch_profile: dict[str, Any] | None = None,
    base_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = build_branch_hybrid_messages(
        label_names=label_names,
        ledger=ledger,
        episode=episode,
        shortlist=shortlist,
        state_summary=state_summary,
        stop_signal=stop_signal,
        mlp_feedback=mlp_feedback,
        turn_index=turn_index,
        max_budget=max_budget,
        branch_profile=branch_profile,
        base_result=base_result,
    )
    shortlist_ids = {item["root_evidence_id"] for item in shortlist}
    raw_attempts = []
    api_calls = 0
    input_tokens = 0
    output_tokens = 0
    estimated_cost = None

    if not RUN_LIVE_API:
        raw_text = json.dumps(scripted_branch_dry_run_response(episode, shortlist, state_summary, stop_signal, mlp_feedback, turn_index, max_budget, label_names, branch_profile))
        payload = {"dry_run": True, "message": raw_text}
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        api_calls = 1
    else:
        raw_text, payload, token_usage = call_openai_compatible(messages)
        api_calls = 1
    raw_attempts.append({"attempt_index": 1, "raw_text": raw_text, "payload": payload})
    input_tokens += token_usage["input_tokens"]
    output_tokens += token_usage["output_tokens"]

    try:
        normalized = normalize_agent_response(parse_json_response(raw_text), label_names)
        error_message = validate_agent_response(normalized, shortlist_ids, label_names)
    except Exception as exc:
        normalized = None
        error_message = str(exc)

    if error_message is not None:
        repair_messages = build_repair_messages(messages, raw_text, error_message)
        if not RUN_LIVE_API:
            repair_text = json.dumps(scripted_branch_dry_run_response(episode, shortlist, state_summary, {**stop_signal, "level": "strong_stop"}, mlp_feedback, turn_index, max_budget, label_names, branch_profile))
            repair_payload = {"dry_run": True, "message": repair_text}
            repair_usage = {"input_tokens": 0, "output_tokens": 0}
            api_calls += 1
        else:
            repair_text, repair_payload, repair_usage = call_openai_compatible(repair_messages)
            api_calls += 1
        raw_attempts.append({"attempt_index": 2, "raw_text": repair_text, "payload": repair_payload})
        input_tokens += repair_usage["input_tokens"]
        output_tokens += repair_usage["output_tokens"]
        normalized = normalize_agent_response(parse_json_response(repair_text), label_names)
        error_message = validate_agent_response(normalized, shortlist_ids, label_names)

    error_flags = []
    if error_message is not None:
        error_flags.append(error_message)
        fallback_labels = list(mlp_feedback.get("top_predictions", [])) or [pathology for pathology, _ in state_summary.top_candidates[:5]] or [label_names[0]]
        normalized = {
            "decision": "stop",
            "requested_evidence_id": None,
            "predicted_pathology": fallback_labels[0],
            "ranked_differential": fallback_labels[:5],
            "confidence": 0.0,
            "brief_reasoning": "Forced stop after invalid responses.",
        }

    return {
        "messages": messages,
        "normalized_response": normalized,
        "raw_attempts": raw_attempts,
        "api_calls": api_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "error_flags": error_flags,
    }


def reconcile_branch_response(
    normalized_response: dict[str, Any],
    state_summary: DiagnosticStateSummary,
    shortlist: list[dict[str, Any]],
    stop_signal: dict[str, Any],
    mlp_feedback: dict[str, Any],
    remaining_budget: int,
    episode: LedgerEpisodeState,
    turn_index: int,
    branch_profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    result, flags = reconcile_selected_stop_response(
        normalized_response=normalized_response,
        state_summary=state_summary,
        shortlist=shortlist,
        stop_signal=stop_signal,
        mlp_feedback=mlp_feedback,
        remaining_budget=remaining_budget,
    )
    if not branch_profile or not shortlist or remaining_budget <= 0:
        return result, flags

    avoid_roots = set(branch_profile.get("avoid_roots", []))
    current_requests = len(episode.request_history)
    should_force_min_request = current_requests < BRANCH_MIN_REQUESTS and result["decision"] == "stop"
    should_force_divergent_request = (
        BRANCH_FORCE_DIVERGENT_EARLY_REQUESTS
        and turn_index <= BRANCH_DIVERGENCE_TURNS
        and result["decision"] == "request"
        and result.get("requested_evidence_id") in avoid_roots
    )
    if should_force_min_request or should_force_divergent_request:
        item = choose_divergent_shortlist_item(shortlist, avoid_roots, list(branch_profile.get("preferred_roots", [])))
        if item is not None:
            result = dict(result)
            result["decision"] = "request"
            result["requested_evidence_id"] = item["root_evidence_id"]
            result["brief_reasoning"] = str(result.get("brief_reasoning", "")) + " [deterministic hypothesis-branch divergence guard]"
            flags = list(flags) + ["branch_min_request" if should_force_min_request else "branch_divergence_root_substitution"]
    return result, flags


def terminal_features_for_result(branch_result: dict[str, Any]) -> dict[str, Any]:
    episode = branch_result["episode"]
    predicted = branch_result["predicted_pathology"]
    evidence = episode_visible_evidence(episode)
    graph_scores, graph_pos, graph_contra, graph_post = compute_graph_scores(evidence)
    bayes_scores, bayes_post = compute_bayes_scores(evidence)
    mlp_probs = mlp_probs_for_episode(episode)
    mlp_idx = PATHOLOGY_TO_INDEX.get(predicted)
    graph_top1, graph_top_score = top_labels_from_scores(graph_scores, 1)[0]
    bayes_top1, bayes_top_posterior = top_labels_from_scores(bayes_post, 1)[0]
    pred_graph_score = float(graph_scores.get(predicted, 0.0))
    pred_bayes_posterior = float(bayes_post.get(predicted, 0.0))
    return {
        "visible_evidence_json": json.dumps(evidence, sort_keys=True),
        "graph_top1": graph_top1,
        "graph_top_score": float(graph_top_score),
        "graph_margin": top_margin(graph_scores),
        "pred_graph_score": pred_graph_score,
        "pred_graph_rank": rank_of(graph_scores, predicted),
        "pred_graph_posterior": float(graph_post.get(predicted, 0.0)),
        "pred_graph_positive_support": float(graph_pos.get(predicted, 0.0)),
        "pred_graph_contradiction": float(graph_contra.get(predicted, 0.0)),
        "bayes_top1": bayes_top1,
        "bayes_top_posterior": float(bayes_top_posterior),
        "bayes_margin": top_margin(bayes_post),
        "pred_bayes_log_score": float(bayes_scores.get(predicted, float("-inf"))),
        "pred_bayes_rank": rank_of(bayes_post, predicted),
        "pred_bayes_posterior": pred_bayes_posterior,
        "pred_mlp_posterior": float(mlp_probs[mlp_idx]) if mlp_idx is not None else 0.0,
    }


def suspicion_features_for_base(base_result: dict[str, Any]) -> dict[str, Any]:
    feats = terminal_features_for_result(base_result)
    signal_values = {
        "signal_llm_mlp_disagree": not bool(base_result.get("llm_mlp_agreement", False)),
        "signal_cap_hit": base_result.get("stop_reason") == "max_requests_reached",
        "signal_uncertain_mlp": float(base_result.get("final_mlp_entropy", 1.0)) > 0.10 or float(base_result.get("final_mlp_margin", 0.0)) < 0.20,
        "signal_early_uncertain_stop": int(base_result.get("num_requests", 0)) <= 3 and (float(base_result.get("final_mlp_entropy", 1.0)) > 0.10 or float(base_result.get("final_mlp_margin", 0.0)) < 0.70),
        "signal_graph_conflict": int(feats["pred_graph_rank"]) > 5 or float(feats["pred_graph_score"]) < 0.0,
        "signal_bayes_conflict": int(feats["pred_bayes_rank"]) > 5 or float(feats["pred_bayes_posterior"]) < 0.05,
        "signal_ledger_disagrees": feats["graph_top1"] != base_result["predicted_pathology"] and feats["bayes_top1"] != base_result["predicted_pathology"],
    }
    signal_count = int(sum(bool(v) for v in signal_values.values()))
    graph_bayes = bool(signal_values["signal_graph_conflict"] and signal_values["signal_bayes_conflict"])
    trigger_fired = bool(signal_count >= 2 or (signal_values["signal_early_uncertain_stop"] and graph_bayes))
    return {
        **feats,
        **signal_values,
        "suspicion_signal_count": signal_count,
        "branch_trigger_name": BRANCH_TRIGGER_NAME,
        "branch_trigger_fired": trigger_fired,
    }


def minmax_list(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [float((v - lo) / (hi - lo)) for v in arr]


def score_branch_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    vote_counts = Counter(candidate["predicted_pathology"] for candidate in candidates)
    for order, candidate in enumerate(candidates):
        features = terminal_features_for_result(candidate)
        row = {k: v for k, v in candidate.items() if k != "episode"}
        row.update(features)
        row["candidate_order"] = order
        row["vote_share"] = vote_counts[candidate["predicted_pathology"]] / len(candidates)
        rows.append(row)
    graph_scaled = minmax_list([row["pred_graph_score"] for row in rows])
    bayes_scaled = minmax_list([row["pred_bayes_log_score"] for row in rows])
    mlp_conf_scaled = minmax_list([row["final_mlp_confidence"] for row in rows])
    mlp_margin_scaled = minmax_list([row["final_mlp_margin"] for row in rows])
    entropy_low_scaled = minmax_list([-row["final_mlp_entropy"] for row in rows])
    for idx, row in enumerate(rows):
        row["graph_scaled"] = graph_scaled[idx]
        row["bayes_scaled"] = bayes_scaled[idx]
        row["mlp_conf_scaled"] = mlp_conf_scaled[idx]
        row["mlp_margin_scaled"] = mlp_margin_scaled[idx]
        row["entropy_low_scaled"] = entropy_low_scaled[idx]
        row["fused_diagnostic_score"] = (
            1.30 * row["vote_share"]
            + 0.80 * row["graph_scaled"]
            + 0.80 * row["bayes_scaled"]
            + 0.45 * row["mlp_conf_scaled"]
            + 0.35 * row["mlp_margin_scaled"]
            + 0.25 * row["entropy_low_scaled"]
            + 0.15 * float(bool(row["llm_mlp_agreement"]))
        )
        row["raw_bayes_judge_score"] = float(row["pred_bayes_posterior"])
    return rows


def choose_branch_candidate(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = score_branch_candidates(candidates)
    chosen = sorted(
        scored,
        key=lambda row: (
            -float(row["raw_bayes_judge_score"]),
            -float(row["pred_graph_score"]),
            -float(row["final_mlp_confidence"]),
            int(row["candidate_order"]),
        ),
    )[0]
    return chosen, scored

# %% [markdown]
# ## Learned Branch Gate And Resolver Overrides

# %%

# Notebook 41 final controller: learned branch gate, capped hypothesis branches, pseudo-candidates, and graph/Bayes/MLP resolver.


def mlp_branch_probability_from_features(feats: dict[str, Any]) -> float:
    row = pd.DataFrame([{col: float(feats.get(col, 0.0)) for col in BRANCH_FEATURE_COLUMNS}])
    return float(branch_trigger_mlp.predict_proba(row[BRANCH_FEATURE_COLUMNS])[:, 1][0])


def slugify_hypothesis_name(label: str, max_len: int = 34) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
    return slug[:max_len] or "hypothesis"


def candidate_hypothesis_rows_for_base(base_result: dict[str, Any], suspicion: dict[str, Any] | None = None, top_k: int = 8) -> list[dict[str, Any]]:
    suspicion = suspicion or terminal_features_for_result(base_result)
    anchor = str(base_result["predicted_pathology"])
    evidence = episode_visible_evidence(base_result["episode"])
    observed_roots = set(evidence)
    graph_scores, graph_pos, graph_contra, graph_post = compute_graph_scores(evidence)
    bayes_scores, bayes_post = compute_bayes_scores(evidence)
    mlp_probs = mlp_probs_for_episode(base_result["episode"])
    mlp_order, mlp_ranks = ranked_from_probs(mlp_probs)
    graph_order = [label for label, _ in top_labels_from_scores(graph_scores, 10)]
    bayes_order = [label for label, _ in top_labels_from_scores(bayes_post, 10)]
    llm_ranked = [str(label) for label in base_result.get("llm_ranked_differential", [])]
    hybrid_ranked = [str(label) for label in base_result.get("ranked_differential", [])]

    candidates: dict[str, dict[str, Any]] = {}

    def touch(label: str, source: str, rank: int, score: float) -> None:
        if label not in ALL_PATHOLOGIES or label == anchor:
            return
        row = candidates.setdefault(label, {"target_hypothesis": label, "sources": [], "source_score": 0.0})
        row["sources"].append({"source": source, "rank": int(rank), "score": float(score)})
        row["source_score"] += float(score)

    for rank, label in enumerate(graph_order[:5], start=1):
        touch(label, "graph_top5", rank, 1.35 / rank)
    for rank, label in enumerate(bayes_order[:5], start=1):
        touch(label, "bayes_top5", rank, 1.35 / rank)
    for rank, label in enumerate(mlp_order[:5], start=1):
        touch(label, "mlp_top5", rank, 0.85 / rank)
    for rank, label in enumerate(hybrid_ranked[:5], start=1):
        touch(label, "hybrid_ranked_diff", rank, 0.75 / rank)
    for rank, label in enumerate(llm_ranked[:5], start=1):
        touch(label, "llm_ranked_diff", rank, 0.65 / rank)

    rows = []
    for label, row in candidates.items():
        coverage, missing_mass, top_missing_root, missing_roots = pair_coverage(anchor, label, observed_roots)
        graph_rank = rank_of(graph_scores, label)
        bayes_rank = rank_of(bayes_post, label)
        mlp_rank = mlp_ranks.get(label, len(ALL_PATHOLOGIES) + 1)
        rank_bonus = (1.0 / max(graph_rank, 1)) + (1.0 / max(bayes_rank, 1)) + (0.5 / max(mlp_rank, 1))
        support_gap = float(graph_scores.get(label, 0.0) - graph_scores.get(anchor, 0.0))
        bayes_gap = float(bayes_post.get(label, 0.0) - bayes_post.get(anchor, 0.0))
        unresolved_bonus = float(missing_mass) * (1.0 - float(coverage))
        hypothesis_priority = float(row["source_score"] + 0.35 * rank_bonus + 0.15 * max(support_gap, 0.0) + 0.50 * max(bayes_gap, 0.0) + 0.20 * unresolved_bonus)
        roots = missing_roots[:6]
        if top_missing_root and top_missing_root not in roots:
            roots = [top_missing_root] + roots
        rows.append({
            "case_id": base_result.get("case_id"),
            "base_prediction": anchor,
            "target_hypothesis": label,
            "hypothesis_priority": hypothesis_priority,
            "source_score": float(row["source_score"]),
            "source_mix_json": json.dumps(row["sources"], sort_keys=True),
            "target_graph_score": float(graph_scores.get(label, 0.0)),
            "target_graph_rank": int(graph_rank),
            "target_graph_posterior": float(graph_post.get(label, 0.0)),
            "target_graph_positive_support": float(graph_pos.get(label, 0.0)),
            "target_graph_contradiction": float(graph_contra.get(label, 0.0)),
            "target_bayes_log_score": float(bayes_scores.get(label, -999.0)),
            "target_bayes_rank": int(bayes_rank),
            "target_bayes_posterior": float(bayes_post.get(label, 0.0)),
            "target_mlp_rank": int(mlp_rank),
            "target_mlp_posterior": float(mlp_probs[PATHOLOGY_TO_INDEX[label]]) if label in PATHOLOGY_TO_INDEX else 0.0,
            "base_graph_score": float(graph_scores.get(anchor, 0.0)),
            "base_bayes_posterior": float(bayes_post.get(anchor, 0.0)),
            "pair_coverage": float(coverage),
            "pair_missing_utility": float(missing_mass),
            "top_missing_pair_root": top_missing_root or "",
            "target_discriminator_roots_json": json.dumps(roots),
        })
    rows.sort(key=lambda item: item["hypothesis_priority"], reverse=True)
    for order, row in enumerate(rows, start=1):
        row["hypothesis_order"] = order
        row["selected_for_branch"] = bool(order <= top_k)
    return rows[:top_k]


def build_hypothesis_branch_profiles(base_result: dict[str, Any], suspicion: dict[str, Any], budget: int = BRANCH_BUDGET) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = candidate_hypothesis_rows_for_base(base_result, suspicion, top_k=max(budget, 1))
    profiles = []
    base_prediction = str(base_result["predicted_pathology"])
    avoid_roots = list(base_result.get("requested_roots", []))[:BRANCH_AVOID_BASE_FIRST_N_ROOTS]
    for idx, row in enumerate(rows[:budget]):
        template = BRANCH_PROFILE_TEMPLATES[idx % len(BRANCH_PROFILE_TEMPLATES)]
        target = str(row["target_hypothesis"])
        preferred_roots = safe_parse_list(row.get("target_discriminator_roots_json", "[]"))
        directive = template["directive_template"].format(base_prediction=base_prediction, target_hypothesis=target)
        profile = {
            "branch_id": f"hypothesis_{idx + 1:02d}_{slugify_hypothesis_name(target)}",
            "role_name": f"{template['role_name_prefix']}: {target}",
            "role_kind": template["role_kind"],
            "target_hypothesis": target,
            "base_prediction": base_prediction,
            "directive": directive,
            "preferred_roots": preferred_roots,
            "avoid_roots": avoid_roots,
            "hypothesis_priority": float(row["hypothesis_priority"]),
            "support_summary": {
                "source_mix": safe_parse_list(row.get("source_mix_json", "[]")),
                "graph_rank": int(row["target_graph_rank"]),
                "graph_score": round(float(row["target_graph_score"]), 4),
                "bayes_rank": int(row["target_bayes_rank"]),
                "bayes_posterior": round(float(row["target_bayes_posterior"]), 6),
                "mlp_rank": int(row["target_mlp_rank"]),
                "mlp_posterior": round(float(row["target_mlp_posterior"]), 6),
                "pair_coverage": round(float(row["pair_coverage"]), 4),
                "pair_missing_utility": round(float(row["pair_missing_utility"]), 6),
            },
        }
        row["branch_id"] = profile["branch_id"]
        row["branch_role_name"] = profile["role_name"]
        row["role_kind"] = profile["role_kind"]
        row["selected_for_branch"] = True
        profiles.append(profile)
    return profiles, rows


def suspicion_features_for_base(base_result: dict[str, Any]) -> dict[str, Any]:
    feats = terminal_features_for_result(base_result)
    evidence = json.loads(feats["visible_evidence_json"])
    anchor = str(base_result["predicted_pathology"])
    mlp_ranked = list(base_result.get("mlp_ranked_differential", []))
    challenger_candidates = [feats["graph_top1"], feats["bayes_top1"], base_result.get("mlp_predicted_pathology"), *mlp_ranked[:3]]
    challenger = next((str(label) for label in challenger_candidates if label and str(label) != anchor), anchor)
    coverage, missing_mass, top_missing_root, missing_roots = pair_coverage(anchor, challenger, set(evidence))
    anchor_mlp_rank = mlp_ranked.index(anchor) + 1 if anchor in mlp_ranked else len(ALL_PATHOLOGIES) + 1
    state_feats = {
        "anchor_prediction": anchor,
        "graph_top1": feats["graph_top1"],
        "bayes_top1": feats["bayes_top1"],
        "mlp_top1": base_result.get("mlp_predicted_pathology"),
        "challenger_prediction": challenger,
        "request_count": float(base_result.get("num_requests", 0)),
        "visible_root_count": float(base_result.get("visible_root_count", len(evidence))),
        "graph_top_score": float(feats["graph_top_score"]),
        "graph_margin": float(feats["graph_margin"]),
        "anchor_graph_score": float(feats["pred_graph_score"]),
        "anchor_graph_rank": float(feats["pred_graph_rank"]),
        "bayes_top_posterior": float(feats["bayes_top_posterior"]),
        "bayes_margin": float(feats["bayes_margin"]),
        "anchor_bayes_posterior": float(feats["pred_bayes_posterior"]),
        "anchor_bayes_rank": float(feats["pred_bayes_rank"]),
        "mlp_confidence": float(base_result.get("final_mlp_confidence", 0.0)),
        "mlp_margin": float(base_result.get("final_mlp_margin", 0.0)),
        "mlp_entropy": float(base_result.get("final_mlp_entropy", 1.0)),
        "anchor_mlp_posterior": float(feats.get("pred_mlp_posterior", 0.0)),
        "anchor_mlp_rank": float(anchor_mlp_rank),
        "graph_bayes_agree": float(feats["graph_top1"] == feats["bayes_top1"]),
        "all_heads_anchor": float(anchor == feats["graph_top1"] == feats["bayes_top1"] == base_result.get("mlp_predicted_pathology")),
        "ledger_disagrees_anchor": float(feats["graph_top1"] != anchor and feats["bayes_top1"] != anchor),
        "pair_coverage": float(coverage),
        "pair_missing_utility": float(missing_mass),
        "top_missing_pair_root": top_missing_root or "",
        "missing_pair_roots": json.dumps(missing_roots),
    }
    probability = mlp_branch_probability_from_features(state_feats)
    signal_values = {
        "signal_llm_mlp_disagree": not bool(base_result.get("llm_mlp_agreement", False)),
        "signal_cap_hit": base_result.get("stop_reason") == "max_requests_reached",
        "signal_uncertain_mlp": float(base_result.get("final_mlp_entropy", 1.0)) > 0.10 or float(base_result.get("final_mlp_margin", 0.0)) < 0.20,
        "signal_early_uncertain_stop": int(base_result.get("num_requests", 0)) <= 3 and (float(base_result.get("final_mlp_entropy", 1.0)) > 0.10 or float(base_result.get("final_mlp_margin", 0.0)) < 0.70),
        "signal_graph_conflict": int(feats["pred_graph_rank"]) > 5 or float(feats["pred_graph_score"]) < 0.0,
        "signal_bayes_conflict": int(feats["pred_bayes_rank"]) > 5 or float(feats["pred_bayes_posterior"]) < 0.05,
        "signal_ledger_disagrees": feats["graph_top1"] != base_result["predicted_pathology"] and feats["bayes_top1"] != base_result["predicted_pathology"],
    }
    signal_count = int(sum(bool(v) for v in signal_values.values()))
    return {
        **feats,
        **state_feats,
        **signal_values,
        "suspicion_signal_count": signal_count,
        "branch_trigger_name": BRANCH_TRIGGER_NAME,
        "branch_trigger_probability": probability,
        "branch_trigger_threshold": BRANCH_TRIGGER_THRESHOLD,
        "branch_trigger_fired": bool(probability >= BRANCH_TRIGGER_THRESHOLD),
    }

def pseudo_candidate_from_base(base_result: dict[str, Any], disease: str, pseudo_source: str, order: int) -> dict[str, Any]:
    ranked = [disease]
    for label in list(base_result.get("ranked_differential", [])) + list(base_result.get("mlp_ranked_differential", [])):
        if label and label not in ranked:
            ranked.append(label)
    item = dict(base_result)
    item.update({
        "branch_id": pseudo_source,
        "candidate_role": "pseudo",
        "branch_role_name": pseudo_source.replace("pseudo_", "Pseudo ").replace("_", " ").title(),
        "predicted_pathology": disease,
        "agreement_hybrid_predicted_pathology": disease,
        "ranked_differential": ranked[:5],
        "candidate_order": 100 + order,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": None,
        "policy_flags": ["pseudo_candidate"],
        "error_flags": [],
    })
    return item


def add_pseudo_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = candidates[0]
    base_features = terminal_features_for_result(base)
    evidence = episode_visible_evidence(base["episode"])
    graph_scores, _, _, _ = compute_graph_scores(evidence)
    _, bayes_post = compute_bayes_scores(evidence)
    graph_top = [label for label, _ in top_labels_from_scores(graph_scores, 3)]
    bayes_top = [label for label, _ in top_labels_from_scores(bayes_post, 3)]
    hypothesis_targets = [str(candidate.get("target_hypothesis")) for candidate in candidates if candidate.get("target_hypothesis")]
    pseudo_specs = []
    for label in graph_top:
        pseudo_specs.append((label, "pseudo_graph_topk"))
    for label in bayes_top:
        pseudo_specs.append((label, "pseudo_bayes_topk"))
    pseudo_specs.append((base_features["graph_top1"], "pseudo_graph_top1"))
    pseudo_specs.append((base_features["bayes_top1"], "pseudo_bayes_top1"))
    pseudo_specs.append((base.get("mlp_predicted_pathology"), "pseudo_mlp_top1"))
    for label in hypothesis_targets:
        pseudo_specs.append((label, "pseudo_assigned_hypothesis"))
    diseases_seen = {str(candidate["predicted_pathology"]) for candidate in candidates}
    out = list(candidates)
    for order, (disease, source) in enumerate(pseudo_specs):
        disease = str(disease)
        if disease and disease in ALL_PATHOLOGIES and disease not in diseases_seen:
            out.append(pseudo_candidate_from_base(base, disease, source, order))
            diseases_seen.add(disease)
    return out


def resolver_features_for_scored_row(row: dict[str, Any], state_feats: dict[str, Any] | None = None) -> dict[str, float]:
    state_feats = state_feats or {}
    source = str(row.get("candidate_role", "branch"))
    return {
        "candidate_order": float(row.get("candidate_order", 0)),
        "candidate_graph_score": float(row.get("pred_graph_score", 0.0)),
        "candidate_graph_posterior": float(row.get("pred_graph_posterior", 0.0)),
        "candidate_graph_rank": float(row.get("pred_graph_rank", len(ALL_PATHOLOGIES) + 1)),
        "candidate_graph_positive_support": float(row.get("pred_graph_positive_support", 0.0)),
        "candidate_graph_contradiction": float(row.get("pred_graph_contradiction", 0.0)),
        "candidate_bayes_log_score": float(row.get("pred_bayes_log_score", -999.0)),
        "candidate_bayes_posterior": float(row.get("pred_bayes_posterior", 0.0)),
        "candidate_bayes_rank": float(row.get("pred_bayes_rank", len(ALL_PATHOLOGIES) + 1)),
        "candidate_mlp_posterior": float(row.get("pred_mlp_posterior", 0.0)),
        "candidate_mlp_rank": 1.0 if row.get("predicted_pathology") == row.get("mlp_predicted_pathology") else float(len(ALL_PATHOLOGIES) + 1),
        "is_base_candidate": float(row.get("branch_id") == "base"),
        "is_branch_candidate": float(source == "branch"),
        "is_pseudo_candidate": float(source == "pseudo"),
        "request_count": float(row.get("num_requests", 0.0)),
        "visible_root_count": float(row.get("visible_root_count", 0.0)),
        "branch_trigger_probability": float(state_feats.get("branch_trigger_probability", 0.0)),
        "pair_coverage": float(state_feats.get("pair_coverage", 1.0)),
        "pair_missing_utility": float(state_feats.get("pair_missing_utility", 0.0)),
    }


def score_branch_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = add_pseudo_candidates(candidates)
    rows = []
    vote_counts = Counter(candidate["predicted_pathology"] for candidate in candidates if candidate.get("candidate_role") != "pseudo")
    base_state_feats = suspicion_features_for_base(candidates[0]) if candidates and candidates[0].get("branch_id") == "base" else {}
    for order, candidate in enumerate(candidates):
        features = terminal_features_for_result(candidate)
        row = {k: v for k, v in candidate.items() if k != "episode"}
        row.update(features)
        row["candidate_order"] = int(candidate.get("candidate_order", order))
        row["vote_share"] = vote_counts.get(candidate["predicted_pathology"], 0) / max(sum(vote_counts.values()), 1)
        row.update({f"resolver_{k}": v for k, v in resolver_features_for_scored_row(row, base_state_feats).items()})
        resolver_input = pd.DataFrame([{col: row.get(f"resolver_{col}", 0.0) for col in RESOLVER_FEATURE_COLUMNS}])
        row["resolver_score"] = float(resolver_model.predict_proba(resolver_input[RESOLVER_FEATURE_COLUMNS])[:, 1][0])
        row["raw_bayes_judge_score"] = float(row["pred_bayes_posterior"])
        row["base_protection_candidate"] = bool(row.get("branch_id") == "base" and int(row.get("pred_graph_rank", 999)) == 1 and int(row.get("pred_bayes_rank", 999)) == 1)
        rows.append(row)
    return rows


def choose_branch_candidate(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = score_branch_candidates(candidates)
    base_row = next((row for row in scored if row.get("branch_id") == "base"), scored[0])
    chosen = sorted(
        scored,
        key=lambda row: (
            -float(row["resolver_score"]),
            -float(row["pred_graph_score"]),
            -float(row["pred_bayes_posterior"]),
            int(row["candidate_order"]),
        ),
    )[0]
    base_is_protected = int(base_row.get("pred_graph_rank", 999)) == 1 and int(base_row.get("pred_bayes_rank", 999)) == 1
    if chosen.get("branch_id") != "base" and base_is_protected:
        margin = float(chosen["resolver_score"]) - float(base_row["resolver_score"])
        challenger_strong = int(chosen.get("pred_graph_rank", 999)) <= 2 or float(chosen.get("pred_bayes_posterior", 0.0)) >= 0.90
        if margin < BASE_PROTECTION_RESOLVER_MARGIN or not challenger_strong:
            chosen = dict(base_row)
            chosen["base_protection_applied"] = True
        else:
            chosen["base_protection_applied"] = False
    else:
        chosen["base_protection_applied"] = False
    return chosen, scored

# %% [markdown]
# ## Live Adaptive Value Branching Execution
# 
# The base branch is run first. Branches are paid live workups only for suspicious terminal states. Unlike Notebook `28`, each spawned branch receives a concrete target diagnosis and pairwise discriminator roots computed before the branch starts. In dry-run mode, the same machinery executes on two cases with scripted responses so the artifact contract can be tested without API spend.

# %%
def top_k_accuracy_from_lists(true_labels: list[str], ranked_lists: list[list[str]], k: int) -> float:
    return float(np.mean([true_label in list(ranked)[:k] for true_label, ranked in zip(true_labels, ranked_lists)])) if true_labels else 0.0


def load_reference_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nb13_path = ROOT / "artifacts" / "sequential_hybrid_mlp_feedback" / "selected_stop_live_confirmation_49case_v1" / "metrics.json"
    nb22_path = ROOT / "artifacts" / "graph_algorithmic_ledger" / "graph_posterior_final_adjudicator_49case_v1" / "selected_graph_adjudicator.json"
    nb23_path = ROOT / "artifacts" / "graph_algorithmic_ledger" / "calibrated_graph_bayes_rescue_reranker_49case_v1" / "selected_rescue_policy.json"
    nb24_path = ROOT / "artifacts" / "graph_algorithmic_ledger" / "live_graph_bayes_rescue_confirmation_49case_v1" / "metrics.json"
    nb26_path = BRANCH_LAB_ROOT / "analysis_summary.json"
    for system, path in [
        ("notebook13_frozen_selected_stop", nb13_path),
        ("notebook22_offline_graph_adjudicator", nb22_path),
        ("notebook23_offline_graph_bayes_rescue", nb23_path),
        ("notebook24_live_rescue_confirmation", nb24_path),
        ("notebook26_offline_branching_lab", nb26_path),
    ]:
        if path.exists():
            data = load_json(path)
            rows.append({"system": system, "source": str(path), "summary": json.dumps(data)[:2000]})
    return pd.DataFrame(rows)


def upsert_rows_by_keys(path: Path, rows: list[dict[str, Any]], key_cols: list[str]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        for col in key_cols:
            if col not in old.columns:
                old[col] = None
        old_key = old[key_cols].astype(str).agg("||".join, axis=1)
        new_key = set(new[key_cols].astype(str).agg("||".join, axis=1))
        old = old.loc[~old_key.isin(new_key)].copy()
        out = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        out = new
    out.to_csv(path, index=False)


def run_workup_branch(
    row: dict[str, Any],
    branch_id: str,
    candidate_role: str,
    raw_api_path: Path,
    branch_profile: dict[str, Any] | None = None,
    base_result: dict[str, Any] | None = None,
    request_cap: int | None = None,
) -> dict[str, Any]:
    effective_request_cap = int(request_cap if request_cap is not None else MAX_REQUEST_CAP)
    episode = ledger.from_row(row, split_name=SPLIT_NAME, prior_differential=prior_lookup.get(str(row["case_id"]), []))
    case_trace: list[dict[str, Any]] = []
    mlp_history: list[dict[str, Any]] = []
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost = 0.0
    has_cost = False
    error_flags: list[str] = []
    policy_flags_all: list[str] = []
    raw_normalized = None
    final_state_summary = diagnosis_manager.summarize_state(episode)
    final_stop_signal: dict[str, Any] = {}
    final_mlp_feedback = compute_mlp_feedback(episode, mlp_schema, partial_mlp_model, partial_mlp_labels, mlp_history)
    stop_reason = "max_requests_reached"
    stop_rule_fired = False

    for turn_index in range(1, effective_request_cap + 1):
        legal_actions = ledger.legal_actions(episode)
        state_summary = diagnosis_manager.summarize_state(episode)
        shortlist, pathology_weights = shortlister.shortlist(episode, legal_actions, state_summary)
        remaining_budget = effective_request_cap - turn_index + 1
        mlp_feedback = compute_mlp_feedback(episode, mlp_schema, partial_mlp_model, partial_mlp_labels, mlp_history)
        stop_signal = build_selected_stop_signal(
            episode=episode,
            state_summary=state_summary,
            shortlist=shortlist,
            remaining_budget=remaining_budget,
            mlp_feedback=mlp_feedback,
        )
        turn_bundle = get_branch_agent_response_with_repair(
            label_names=label_names,
            ledger=ledger,
            episode=episode,
            shortlist=shortlist,
            state_summary=state_summary,
            stop_signal=stop_signal,
            mlp_feedback=mlp_feedback,
            turn_index=turn_index,
            max_budget=effective_request_cap,
            branch_profile=branch_profile,
            base_result=base_result,
        )
        raw_normalized = turn_bundle["normalized_response"]
        final_response, policy_flags = reconcile_branch_response(
            normalized_response=raw_normalized,
            state_summary=state_summary,
            shortlist=shortlist,
            stop_signal=stop_signal,
            mlp_feedback=mlp_feedback,
            remaining_budget=remaining_budget,
            episode=episode,
            turn_index=turn_index,
            branch_profile=branch_profile,
        )
        final_state_summary = state_summary
        final_stop_signal = stop_signal
        final_mlp_feedback = mlp_feedback
        stop_rule_fired = bool(stop_signal.get("selected_stop_rule_fired", False))
        mlp_history.append(mlp_feedback)
        total_api_calls += turn_bundle["api_calls"]
        total_input_tokens += turn_bundle["input_tokens"]
        total_output_tokens += turn_bundle["output_tokens"]
        if turn_bundle["estimated_cost"] is not None:
            total_estimated_cost += float(turn_bundle["estimated_cost"])
            has_cost = True
        error_flags.extend(turn_bundle["error_flags"])
        policy_flags_all.extend(policy_flags)

        ledger.register_diagnosis(
            episode=episode,
            turn_index=turn_index,
            normalized_response=final_response,
            shortlist_snapshot=shortlist,
            stop_signal=stop_signal,
            state_summary=state_summary,
            policy_flags=policy_flags,
        )
        for raw_attempt in turn_bundle["raw_attempts"]:
            append_jsonl(raw_api_path, {"case_id": episode.case_id, "branch_id": branch_id, "candidate_role": candidate_role, "turn_index": turn_index, **raw_attempt})
        trace_step = {
            "turn_index": turn_index,
            "branch_id": branch_id,
            "candidate_role": candidate_role,
            "branch_profile": branch_profile,
            "visible_context_before": ledger.visible_context(episode),
            "deterministic_state": state_summary.to_dict(),
            "mlp_feedback": mlp_feedback,
            "shortlist": shortlist,
            "stop_signal": stop_signal,
            "raw_agent_response": raw_normalized,
            "agent_response": final_response,
            "policy_response": final_response,
            "policy_flags": list(policy_flags),
            "error_flags": list(turn_bundle["error_flags"]),
        }
        if final_response["decision"] == "request" and shortlist:
            reveal_payload = ledger.reveal(episode, final_response["requested_evidence_id"], turn_index)
            trace_step["reveal_payload"] = reveal_payload
            stop_reason = "max_requests_reached" if turn_index == effective_request_cap else "requested_more_evidence"
            case_trace.append(trace_step)
            if turn_index == effective_request_cap:
                break
            continue
        stop_reason = "selected_mlp_stop" if stop_rule_fired else "agent_stop"
        case_trace.append(trace_step)
        break

    if raw_normalized is None:
        raw_normalized = {"predicted_pathology": final_mlp_feedback["top1"], "ranked_differential": final_mlp_feedback["top_predictions"], "confidence": final_mlp_feedback["confidence"], "decision": "stop", "requested_evidence_id": None, "brief_reasoning": "empty fallback"}
    llm_pred = str(raw_normalized["predicted_pathology"])
    llm_ranked = list(raw_normalized["ranked_differential"])
    mlp_pred = str(final_mlp_feedback["top1"])
    mlp_ranked = list(final_mlp_feedback["top_predictions"])
    llm_confidence = float(raw_normalized.get("confidence", 0.0))
    agreement_pred, agreement_ranked, agreement_source = agreement_hybrid_prediction(llm_pred, llm_ranked, mlp_pred, mlp_ranked)
    conservative_pred, conservative_ranked, conservative_source = conservative_hybrid_prediction(
        llm_pred, llm_ranked, mlp_pred, mlp_ranked, final_mlp_feedback, llm_confidence
    )
    requested_roots = [str(item["root_evidence_id"]) for item in episode.request_history]
    return {
        "case_id": episode.case_id,
        "branch_id": branch_id,
        "candidate_role": candidate_role,
        "branch_role_name": branch_profile.get("role_name") if branch_profile else "base",
        "request_cap": int(effective_request_cap),
        "source_row_index": episode.source_row_index,
        "split": episode.split_name,
        "AGE": episode.age,
        "SEX": episode.sex,
        "true_pathology": episode.pathology,
        "predicted_pathology": agreement_pred,
        "agreement_hybrid_predicted_pathology": agreement_pred,
        "conservative_hybrid_predicted_pathology": conservative_pred,
        "llm_predicted_pathology": llm_pred,
        "mlp_predicted_pathology": mlp_pred,
        "ranked_differential": agreement_ranked,
        "agreement_hybrid_ranked_differential": agreement_ranked,
        "conservative_hybrid_ranked_differential": conservative_ranked,
        "llm_ranked_differential": llm_ranked,
        "mlp_ranked_differential": mlp_ranked,
        "agreement_hybrid_final_source": agreement_source,
        "conservative_hybrid_final_source": conservative_source,
        "final_confidence": float(max(llm_confidence, final_mlp_feedback["confidence"])),
        "llm_confidence": llm_confidence,
        "final_mlp_confidence": float(final_mlp_feedback["confidence"]),
        "final_mlp_margin": float(final_mlp_feedback["margin"]),
        "final_mlp_entropy": float(final_mlp_feedback["entropy"]),
        "final_mlp_stability_turns": int(final_mlp_feedback["stability_turns"]),
        "llm_mlp_agreement": bool(llm_pred == mlp_pred),
        "mlp_in_llm_top3": bool(mlp_pred in llm_ranked[:3]),
        "num_requests": len(episode.request_history),
        "requested_roots": requested_roots,
        "visible_root_count": len(episode.revealed_roots),
        "stop_reason": stop_reason,
        "stop_rule_fired": bool(stop_rule_fired),
        "api_calls": total_api_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost": float(total_estimated_cost) if has_cost else None,
        "initial_evidence": episode.initial_evidence,
        "prior_top1": episode.prior_differential[0][0] if episode.prior_differential else None,
        "prior_top1_score": episode.prior_differential[0][1] if episode.prior_differential else None,
        "deterministic_top1": final_state_summary.top_candidates[0][0] if final_state_summary.top_candidates else None,
        "deterministic_margin": float(final_state_summary.margin),
        "deterministic_unresolved_mass": float(final_state_summary.unresolved_mass),
        "final_top_shortlist_score": float(final_stop_signal.get("top_shortlist_score", 0.0)),
        "selected_policy_name": SELECTED_STOP_POLICY["policy_name"],
        "policy_flags": list(policy_flags_all),
        "error_flags": list(error_flags),
        "trace": case_trace,
        "episode": episode,
    }


def metrics_for_branching_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    true_labels = frame["true_pathology"].astype(str).tolist()
    selected_preds = frame["predicted_pathology"].astype(str).tolist()
    base_preds = frame["base_predicted_pathology"].astype(str).tolist()
    ranked_lists = frame["ranked_differential"].map(safe_parse_list).tolist()
    base_ranked_lists = frame["base_ranked_differential"].map(safe_parse_list).tolist() if "base_ranked_differential" in frame.columns else [[pred] for pred in base_preds]
    return {
        "num_cases": int(len(frame)),
        "accuracy": float(np.mean([a == b for a, b in zip(true_labels, selected_preds)])),
        "num_correct": int(sum(a == b for a, b in zip(true_labels, selected_preds))),
        "top3_accuracy": top_k_accuracy_from_lists(true_labels, ranked_lists, 3),
        "top5_accuracy": top_k_accuracy_from_lists(true_labels, ranked_lists, 5),
        "base_top3_accuracy": top_k_accuracy_from_lists(true_labels, base_ranked_lists, 3),
        "base_top5_accuracy": top_k_accuracy_from_lists(true_labels, base_ranked_lists, 5),
        "macro_f1": float(f1_score(true_labels, selected_preds, average="macro")),
        "base_accuracy": float(np.mean([a == b for a, b in zip(true_labels, base_preds)])),
        "base_num_correct": int(sum(a == b for a, b in zip(true_labels, base_preds))),
        "wins_vs_base": int((~frame["base_correct"].astype(bool) & frame["correct"].astype(bool)).sum()),
        "regressions_vs_base": int((frame["base_correct"].astype(bool) & ~frame["correct"].astype(bool)).sum()),
        "changed_predictions": int(frame["changed_prediction"].sum()),
        "branch_trigger_rate": float(frame["branch_trigger_fired"].mean()),
        "branches_spawned_total": int(frame["branches_spawned"].sum()),
        "mean_branches_spawned": float(frame["branches_spawned"].mean()),
        "mean_selected_requests": float(frame["num_requests"].mean()),
        "mean_base_requests": float(frame["num_requests_base"].mean()),
        "mean_total_branch_requests": float(frame["total_branch_requests"].mean()),
        "total_api_calls": int(frame["total_api_calls"].sum()),
        "total_input_tokens": int(frame["input_tokens"].fillna(0).sum()),
        "total_output_tokens": int(frame["output_tokens"].fillna(0).sum()),
        "selection_source_distribution": {str(k): int(v) for k, v in frame["selected_branch_id"].value_counts().sort_index().items()},
        "stop_reason_distribution": {str(k): int(v) for k, v in frame["selected_stop_reason"].value_counts().sort_index().items()},
    }


def resolver_state_from_scored(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        scored_candidates,
        key=lambda row: (
            -float(row.get("resolver_score", 0.0)),
            -float(row.get("pred_graph_score", 0.0)),
            -float(row.get("pred_bayes_posterior", 0.0)),
            int(row.get("candidate_order", 999)),
        ),
    )
    scores = np.array([float(row.get("resolver_score", 0.0)) for row in ordered], dtype=float)
    probs = np.exp(scores - np.max(scores)) if len(scores) else np.array([], dtype=float)
    probs = probs / max(float(probs.sum()), EPS) if len(probs) else probs
    entropy = float(-(probs * np.log(np.clip(probs, EPS, 1.0))).sum() / np.log(len(probs))) if len(probs) > 1 else 0.0
    top = ordered[0] if ordered else {}
    second = ordered[1] if len(ordered) > 1 else {}
    margin = float(scores[0] - scores[1]) if len(scores) > 1 else 99.0
    top_pathology = str(top.get("predicted_pathology", ""))
    math_support_count = int(int(top.get("pred_graph_rank", 999)) == 1) + int(int(top.get("pred_bayes_rank", 999)) == 1) + int(top_pathology == str(top.get("mlp_predicted_pathology", "")))
    return {
        "top_pathology": top_pathology,
        "second_pathology": str(second.get("predicted_pathology", "")),
        "resolver_raw_margin": margin,
        "resolver_entropy": entropy,
        "math_support_count": math_support_count,
        "ledger_disagreement": 1.0 - (math_support_count / 3.0),
        "candidate_count": len(ordered),
    }


def adaptive_continuation_decision(
    *,
    case_id: str,
    branch_candidates: list[dict[str, Any]],
    previous_state: dict[str, Any],
    next_profile: dict[str, Any],
    first_priority: float,
    decision_step: int,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    _, scored_now = choose_branch_candidate(branch_candidates)
    current_state = resolver_state_from_scored(scored_now)
    margin_uncertainty = 1.0 - min(float(current_state["resolver_raw_margin"]) / 0.25, 1.0)
    unresolved_mass = (
        0.45 * float(current_state["ledger_disagreement"])
        + 0.30 * float(current_state["resolver_entropy"])
        + 0.25 * margin_uncertainty
    )
    next_priority_ratio = min(float(next_profile.get("hypothesis_priority", 0.0)) / max(float(first_priority), EPS), 1.0)
    top_changed = str(previous_state.get("top_pathology")) != str(current_state.get("top_pathology"))
    sufficient_update = (top_changed and int(current_state["math_support_count"]) >= 2) or int(current_state["math_support_count"]) == 3
    suppression = 0.35 if sufficient_update else 1.0
    continuation_value = float(unresolved_mass * next_priority_ratio * suppression)
    launch = bool(continuation_value >= ADAPTIVE_CONTINUATION_VALUE_THRESHOLD)
    row = {
        "case_id": case_id,
        "decision_step": int(decision_step),
        "action": "launch_branch" if launch else "stop_branching",
        "next_branch_id": next_profile.get("branch_id", ""),
        "next_target_hypothesis": next_profile.get("target_hypothesis", ""),
        "next_role_kind": next_profile.get("role_kind", ""),
        "top_pathology": current_state.get("top_pathology", ""),
        "second_pathology": current_state.get("second_pathology", ""),
        "previous_top_pathology": previous_state.get("top_pathology", ""),
        "resolver_raw_margin": float(current_state["resolver_raw_margin"]),
        "resolver_entropy": float(current_state["resolver_entropy"]),
        "ledger_disagreement": float(current_state["ledger_disagreement"]),
        "math_support_count": int(current_state["math_support_count"]),
        "unresolved_mass": float(unresolved_mass),
        "next_priority_ratio": float(next_priority_ratio),
        "top_changed_after_last_branch": bool(top_changed),
        "sufficient_update": bool(sufficient_update),
        "suppression": float(suppression),
        "continuation_value": float(continuation_value),
        "continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
        "reason": "continuation_value_above_threshold" if launch else "continuation_value_below_threshold_or_sufficient_update",
    }
    return launch, row, current_state


def run_hypothesis_forced_differential_branching() -> pd.DataFrame:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_case_export.to_csv(ARTIFACT_ROOT / "benchmark_cases.csv", index=False)
    predictions_path = ARTIFACT_ROOT / "predictions.csv"
    branch_case_path = ARTIFACT_ROOT / "branch_case_results.csv"
    hypothesis_assignment_path = ARTIFACT_ROOT / "hypothesis_branch_assignments.csv"
    adaptive_decision_trace_path = ARTIFACT_ROOT / "adaptive_branch_decision_trace.csv"
    candidate_scores_path = ARTIFACT_ROOT / "candidate_level_live_scores.csv"
    paired_path = ARTIFACT_ROOT / "paired_notebook13_vs_notebook41.csv"
    traces_path = ARTIFACT_ROOT / "traces.jsonl"
    branch_traces_path = ARTIFACT_ROOT / "branch_traces.jsonl"
    raw_api_path = ARTIFACT_ROOT / "raw_api_responses.jsonl"
    reference_summary = load_reference_summary()
    reference_summary.to_csv(ARTIFACT_ROOT / "reference_summary.csv", index=False)

    resolved_run_config = {
        "notebook": "41_final_capped_hypothesis_branching_confirmation.ipynb",
        "run_name": RUN_NAME,
        "artifact_root": str(ARTIFACT_ROOT),
        "run_live_api": bool(RUN_LIVE_API),
        "allow_dry_run_benchmark": bool(ALLOW_DRY_RUN_BENCHMARK),
        "resume_if_available": bool(RESUME_IF_AVAILABLE),
        "max_request_cap_per_branch": int(MAX_REQUEST_CAP),
        "base_request_cap": int(BASE_REQUEST_CAP),
        "branch_request_cap": int(BRANCH_REQUEST_CAP),
        "hard_total_request_cap_per_case": int(HARD_TOTAL_REQUEST_CAP_PER_CASE),
        "min_remaining_requests_to_launch_branch": int(MIN_REMAINING_REQUESTS_TO_LAUNCH_BRANCH),
        "split_name": SPLIT_NAME,
        "random_seed": RANDOM_SEED,
        "llm_model": LLM_MODEL,
        "temperature": float(TEMPERATURE),
        "top_p": float(TOP_P),
        "selected_stop_policy": SELECTED_STOP_POLICY,
        "branch_trigger_name": BRANCH_TRIGGER_NAME,
        "branch_trigger_threshold": float(BRANCH_TRIGGER_THRESHOLD),
        "max_branches": int(BRANCH_BUDGET),
        "adaptive_continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
        "branch_judge_name": BRANCH_JUDGE_NAME,
        "branch_profile_templates": BRANCH_PROFILE_TEMPLATES,
        "branch_min_requests": int(BRANCH_MIN_REQUESTS),
        "branch_divergence_turns": int(BRANCH_DIVERGENCE_TURNS),
        "branch_avoid_base_first_n_roots": int(BRANCH_AVOID_BASE_FIRST_N_ROOTS),
        "dry_run_force_branch_path_smoke": bool(DRY_RUN_FORCE_BRANCH_PATH_SMOKE),
        "graph_edge_path": str(GRAPH_EDGE_PATH),
        "bayes_likelihood_path": str(BAYES_LIKELIHOOD_PATH),
        "bayes_prior_path": str(BAYES_PRIOR_PATH),
        "notebook26_recommended_policy": recommended_branching_policy,
        "decision_rule": "Final capped confirmation: run Notebook 13-style base, launch at most two short hypothesis branches only when the learned branch gate fires, resolve with graph/Bayes/MLP candidate scoring, and do not use the close-confounder extra-root layer.",
        "branch_trigger_threshold": float(BRANCH_TRIGGER_THRESHOLD),
        "branch_trigger_validation_summary": branch_validation_summary.to_dict(orient="records"),
        "resolver_validation_summary": resolver_validation_summary.to_dict(orient="records"),
        "leakage_rule": "Labels and hidden full evidence are used only for metric computation. Branch trigger, branch prompts, and judge use current revealed evidence plus train/validate-derived graph/Bayes/MLP artifacts.",
        "reference_systems": reference_summary.to_dict(orient="records"),
    }
    with (ARTIFACT_ROOT / "resolved_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_run_config, handle, indent=2)

    if not RUN_LIVE_API and not ALLOW_DRY_RUN_BENCHMARK:
        print("Notebook 41 live path skipped. Set RUN_LIVE_API=True or ALLOW_DRY_RUN_BENCHMARK=True for smoke artifacts.")
        display(reference_summary)
        return pd.DataFrame()
    if RUN_LIVE_API and not LLM_API_KEY:
        raise ValueError("RUN_LIVE_API=True but LLM_API_KEY is empty. Provide LLM_API_KEY or use the interactive prompt.")

    processed_case_ids: set[str] = set()
    if RESUME_IF_AVAILABLE and predictions_path.exists():
        processed_case_ids = set(pd.read_csv(predictions_path)["case_id"].astype(str).tolist())
        print(f"Resuming with {len(processed_case_ids)} completed cases.")

    started_at = time.time()
    dry_run_forced_branch_path_smoke_used = False
    for row in tqdm(list(benchmark_df.to_dict(orient="records")), desc="Final capped hypothesis-branching confirmation"):
        case_id = str(row["case_id"])
        if case_id in processed_case_ids:
            continue

        base_result = run_workup_branch(
            row=row,
            branch_id="base",
            candidate_role="base",
            raw_api_path=raw_api_path,
            request_cap=BASE_REQUEST_CAP,
        )
        suspicion = suspicion_features_for_base(base_result)
        branch_candidates = [base_result]
        trigger_fired = bool(suspicion["branch_trigger_fired"])
        dry_run_forced_branch_path_smoke = False
        if (not RUN_LIVE_API) and DRY_RUN_FORCE_BRANCH_PATH_SMOKE and not dry_run_forced_branch_path_smoke_used:
            trigger_fired = True
            dry_run_forced_branch_path_smoke = True
            dry_run_forced_branch_path_smoke_used = True
            suspicion["branch_trigger_fired"] = True
            suspicion["dry_run_forced_branch_path_smoke"] = True
        branch_profiles_used = []
        adaptive_decision_rows = []
        branch_profiles_planned, hypothesis_rows = build_hypothesis_branch_profiles(base_result, suspicion, BRANCH_BUDGET)
        for row_idx, hypothesis_row in enumerate(hypothesis_rows, start=1):
            hypothesis_row.update({
                "case_id": case_id,
                "true_pathology": base_result["true_pathology"],
                "branch_trigger_fired": trigger_fired,
                "branch_trigger_probability": float(suspicion["branch_trigger_probability"]),
                "branch_trigger_threshold": float(suspicion["branch_trigger_threshold"]),
                "dry_run_forced_branch_path_smoke": bool(dry_run_forced_branch_path_smoke),
                "selected_for_branch": False,
            })
        if trigger_fired and branch_profiles_planned:
            _, base_scored_for_state = choose_branch_candidate(branch_candidates)
            previous_state = resolver_state_from_scored(base_scored_for_state)
            first_priority = float(branch_profiles_planned[0].get("hypothesis_priority", 1.0))
            for branch_order, branch_profile in enumerate(branch_profiles_planned[:BRANCH_BUDGET], start=1):
                if branch_order == 1:
                    launch_branch = True
                    adaptive_row = {
                        "case_id": case_id,
                        "decision_step": branch_order,
                        "action": "launch_branch",
                        "next_branch_id": branch_profile.get("branch_id", ""),
                        "next_target_hypothesis": branch_profile.get("target_hypothesis", ""),
                        "next_role_kind": branch_profile.get("role_kind", ""),
                        "top_pathology": previous_state.get("top_pathology", ""),
                        "second_pathology": previous_state.get("second_pathology", ""),
                        "previous_top_pathology": "",
                        "resolver_raw_margin": float(previous_state["resolver_raw_margin"]),
                        "resolver_entropy": float(previous_state["resolver_entropy"]),
                        "ledger_disagreement": float(previous_state["ledger_disagreement"]),
                        "math_support_count": int(previous_state["math_support_count"]),
                        "unresolved_mass": np.nan,
                        "next_priority_ratio": 1.0,
                        "top_changed_after_last_branch": False,
                        "sufficient_update": False,
                        "suppression": 1.0,
                        "continuation_value": 1.0,
                        "continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
                        "reason": "initial_branch_trigger_fired",
                    }
                else:
                    launch_branch, adaptive_row, previous_state = adaptive_continuation_decision(
                        case_id=case_id,
                        branch_candidates=branch_candidates,
                        previous_state=previous_state,
                        next_profile=branch_profile,
                        first_priority=first_priority,
                        decision_step=branch_order,
                    )
                adaptive_decision_rows.append(adaptive_row)
                if not launch_branch:
                    break
                consumed_requests = int(sum(candidate["num_requests"] for candidate in branch_candidates))
                remaining_case_budget = int(HARD_TOTAL_REQUEST_CAP_PER_CASE - consumed_requests)
                if remaining_case_budget < MIN_REMAINING_REQUESTS_TO_LAUNCH_BRANCH:
                    adaptive_row["action"] = "stop_branching"
                    adaptive_row["reason"] = "hard_total_request_cap_reached"
                    adaptive_row["remaining_case_budget"] = remaining_case_budget
                    break
                branch_request_cap = int(min(BRANCH_REQUEST_CAP, remaining_case_budget))
                adaptive_row["branch_request_cap"] = branch_request_cap
                adaptive_row["remaining_case_budget"] = remaining_case_budget
                branch_profiles_used.append(branch_profile)
                for hypothesis_row in hypothesis_rows:
                    if hypothesis_row.get("branch_id") == branch_profile.get("branch_id"):
                        hypothesis_row["selected_for_branch"] = True
                branch_result = run_workup_branch(
                    row=row,
                    branch_id=branch_profile["branch_id"],
                    candidate_role="branch",
                    raw_api_path=raw_api_path,
                    branch_profile=branch_profile,
                    base_result=base_result,
                    request_cap=branch_request_cap,
                )
                branch_result["target_hypothesis"] = branch_profile.get("target_hypothesis")
                branch_result["target_role_kind"] = branch_profile.get("role_kind")
                branch_result["target_preferred_roots"] = branch_profile.get("preferred_roots", [])
                branch_candidates.append(branch_result)
                _, scored_after_branch = choose_branch_candidate(branch_candidates)
                previous_state = resolver_state_from_scored(scored_after_branch)
        elif not trigger_fired:
            adaptive_decision_rows.append({
                "case_id": case_id,
                "decision_step": 0,
                "action": "stop_no_initial_branch",
                "next_branch_id": "",
                "next_target_hypothesis": "",
                "next_role_kind": "",
                "top_pathology": base_result.get("predicted_pathology", ""),
                "second_pathology": "",
                "previous_top_pathology": "",
                "resolver_raw_margin": np.nan,
                "resolver_entropy": np.nan,
                "ledger_disagreement": np.nan,
                "math_support_count": np.nan,
                "unresolved_mass": np.nan,
                "next_priority_ratio": 0.0,
                "top_changed_after_last_branch": False,
                "sufficient_update": False,
                "suppression": 1.0,
                "continuation_value": 0.0,
                "continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
                "reason": "branch_trigger_below_threshold",
            })
        if hypothesis_rows:
            upsert_rows_by_keys(hypothesis_assignment_path, hypothesis_rows, ["case_id", "target_hypothesis"])
        if adaptive_decision_rows:
            upsert_rows_by_keys(adaptive_decision_trace_path, adaptive_decision_rows, ["case_id", "decision_step"])

        chosen, scored_candidates = choose_branch_candidate(branch_candidates)
        selected_ranked = list(chosen["ranked_differential"])
        selected_pred = str(chosen["predicted_pathology"])
        base_pred = str(base_result["predicted_pathology"])
        base_correct = bool(base_pred == base_result["true_pathology"])
        selected_correct = bool(selected_pred == base_result["true_pathology"])
        branches_spawned = len(branch_candidates) - 1
        total_branch_requests = int(sum(candidate["num_requests"] for candidate in branch_candidates))
        total_api_calls = int(sum(candidate["api_calls"] for candidate in branch_candidates))
        total_input_tokens = int(sum(candidate["input_tokens"] for candidate in branch_candidates))
        total_output_tokens = int(sum(candidate["output_tokens"] for candidate in branch_candidates))

        branch_rows = []
        for candidate in scored_candidates:
            branch_rows.append({
                **{k: v for k, v in candidate.items() if k not in {"trace", "ranked_differential", "agreement_hybrid_ranked_differential", "conservative_hybrid_ranked_differential", "llm_ranked_differential", "mlp_ranked_differential", "requested_roots"}},
                "true_pathology": base_result["true_pathology"],
                "correct": bool(candidate["predicted_pathology"] == base_result["true_pathology"]),
                "selected_by_judge": bool(candidate["branch_id"] == chosen["branch_id"]),
                "ranked_differential": json.dumps(candidate["ranked_differential"]),
                "requested_roots": json.dumps(candidate["requested_roots"]),
                "policy_flags": json.dumps(candidate.get("policy_flags", [])),
                "error_flags": json.dumps(candidate.get("error_flags", [])),
            })
        upsert_rows_by_keys(branch_case_path, branch_rows, ["case_id", "branch_id"])
        upsert_rows_by_keys(candidate_scores_path, branch_rows, ["case_id", "branch_id"])

        prediction_row = {
            "case_id": case_id,
            "max_request_cap_per_branch": int(MAX_REQUEST_CAP),
            "base_request_cap": int(BASE_REQUEST_CAP),
            "branch_request_cap": int(BRANCH_REQUEST_CAP),
            "hard_total_request_cap_per_case": int(HARD_TOTAL_REQUEST_CAP_PER_CASE),
            "source_row_index": base_result["source_row_index"],
            "split": base_result["split"],
            "AGE": base_result["AGE"],
            "SEX": base_result["SEX"],
            "true_pathology": base_result["true_pathology"],
            "base_predicted_pathology": base_pred,
            "predicted_pathology": selected_pred,
            "correct": selected_correct,
            "base_correct": base_correct,
            "changed_prediction": bool(selected_pred != base_pred),
            "improvement_vs_base": bool((not base_correct) and selected_correct),
            "regression_vs_base": bool(base_correct and (not selected_correct)),
            "selected_branch_id": chosen["branch_id"],
            "selected_candidate_role": chosen["candidate_role"],
            "selected_branch_role_name": chosen["branch_role_name"],
            "branch_trigger_fired": trigger_fired,
            "branch_trigger_name": BRANCH_TRIGGER_NAME,
            "branch_trigger_probability": float(suspicion["branch_trigger_probability"]),
            "branch_trigger_threshold": float(suspicion["branch_trigger_threshold"]),
            "dry_run_forced_branch_path_smoke": bool(dry_run_forced_branch_path_smoke),
            "pair_coverage": float(suspicion.get("pair_coverage", 1.0)),
            "pair_missing_utility": float(suspicion.get("pair_missing_utility", 0.0)),
            "challenger_prediction": suspicion.get("challenger_prediction"),
            "top_missing_pair_root": suspicion.get("top_missing_pair_root"),
            "branches_spawned": branches_spawned,
            "branch_profiles_used": json.dumps(branch_profiles_used),
            "suspicion_signal_count": int(suspicion["suspicion_signal_count"]),
            "signal_llm_mlp_disagree": bool(suspicion["signal_llm_mlp_disagree"]),
            "signal_cap_hit": bool(suspicion["signal_cap_hit"]),
            "signal_uncertain_mlp": bool(suspicion["signal_uncertain_mlp"]),
            "signal_early_uncertain_stop": bool(suspicion["signal_early_uncertain_stop"]),
            "signal_graph_conflict": bool(suspicion["signal_graph_conflict"]),
            "signal_bayes_conflict": bool(suspicion["signal_bayes_conflict"]),
            "signal_ledger_disagrees": bool(suspicion["signal_ledger_disagrees"]),
            "ranked_differential": json.dumps(selected_ranked),
            "base_ranked_differential": json.dumps(base_result["ranked_differential"]),
            "num_requests": int(chosen["num_requests"]),
            "num_requests_base": int(base_result["num_requests"]),
            "num_requests_selected": int(chosen["num_requests"]),
            "total_branch_requests": total_branch_requests,
            "visible_root_count_selected": int(chosen["visible_root_count"]),
            "selected_stop_reason": chosen["stop_reason"],
            "base_stop_reason": base_result["stop_reason"],
            "selected_resolver_score": float(chosen.get("resolver_score", 0.0)),
            "selected_raw_bayes_judge_score": float(chosen["raw_bayes_judge_score"]),
            "base_protection_applied": bool(chosen.get("base_protection_applied", False)),
            "selected_pred_bayes_posterior": float(chosen["pred_bayes_posterior"]),
            "selected_pred_graph_score": float(chosen["pred_graph_score"]),
            "selected_pred_graph_rank": int(chosen["pred_graph_rank"]),
            "base_pred_bayes_posterior": float(suspicion["pred_bayes_posterior"]),
            "base_pred_graph_score": float(suspicion["pred_graph_score"]),
            "base_pred_graph_rank": int(suspicion["pred_graph_rank"]),
            "final_mlp_confidence": float(chosen["final_mlp_confidence"]),
            "final_mlp_margin": float(chosen["final_mlp_margin"]),
            "final_mlp_entropy": float(chosen["final_mlp_entropy"]),
            "llm_mlp_agreement": bool(chosen["llm_mlp_agreement"]),
            "api_calls": total_api_calls,
            "total_api_calls": total_api_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "initial_evidence": base_result["initial_evidence"],
            "base_requested_roots": json.dumps(base_result["requested_roots"]),
            "selected_requested_roots": json.dumps(chosen["requested_roots"]),
            "policy_name": f"final_capped|trigger={BRANCH_TRIGGER_NAME}|threshold={BRANCH_TRIGGER_THRESHOLD:.2f}|max_branches={BRANCH_BUDGET}|branch_cap={BRANCH_REQUEST_CAP}|total_cap={HARD_TOTAL_REQUEST_CAP_PER_CASE}|continuation={ADAPTIVE_CONTINUATION_VALUE_THRESHOLD:.2f}|judge={BRANCH_JUDGE_NAME}",
        }
        upsert_prediction_row(predictions_path, prediction_row)
        upsert_prediction_row(paired_path, prediction_row)

        branch_trace_payload = []
        for candidate in branch_candidates:
            trace_obj = {
                "case_id": case_id,
                "branch_id": candidate["branch_id"],
                "candidate_role": candidate["candidate_role"],
                "true_pathology": candidate["true_pathology"],
                "predicted_pathology": candidate["predicted_pathology"],
                "target_hypothesis": candidate.get("target_hypothesis"),
                "selected_by_judge": bool(candidate["branch_id"] == chosen["branch_id"]),
                "num_requests": int(candidate["num_requests"]),
                "requested_roots": candidate["requested_roots"],
                "trace": candidate["trace"],
            }
            branch_trace_payload.append(trace_obj)
            append_jsonl(branch_traces_path, trace_obj)
        append_jsonl(traces_path, {
            "case_id": case_id,
            "true_pathology": base_result["true_pathology"],
            "base_predicted_pathology": base_pred,
            "selected_prediction": selected_pred,
            "selected_branch_id": chosen["branch_id"],
            "branch_trigger_fired": trigger_fired,
            "suspicion": {k: v for k, v in suspicion.items() if k != "visible_evidence_json"},
            "candidate_scores": [{k: v for k, v in candidate.items() if k not in {"trace", "episode"}} for candidate in scored_candidates],
            "branches": branch_trace_payload,
        })

    if not predictions_path.exists():
        print("No predictions were generated.")
        return pd.DataFrame()

    prediction_frame = pd.read_csv(predictions_path)
    metrics = metrics_for_branching_predictions(prediction_frame)
    metrics.update({
        "run_name": RUN_NAME,
        "artifact_root": str(ARTIFACT_ROOT),
        "runtime_seconds": float(time.time() - started_at),
        "run_live_api": bool(RUN_LIVE_API),
        "allow_dry_run_benchmark": bool(ALLOW_DRY_RUN_BENCHMARK),
        "branch_trigger_name": BRANCH_TRIGGER_NAME,
        "branch_trigger_threshold": float(BRANCH_TRIGGER_THRESHOLD),
        "max_branches": int(BRANCH_BUDGET),
        "adaptive_continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
        "branch_judge_name": BRANCH_JUDGE_NAME,
        "dry_run_force_branch_path_smoke": bool(DRY_RUN_FORCE_BRANCH_PATH_SMOKE),
        "base_request_cap": int(BASE_REQUEST_CAP),
        "branch_request_cap": int(BRANCH_REQUEST_CAP),
        "hard_total_request_cap_per_case": int(HARD_TOTAL_REQUEST_CAP_PER_CASE),
        "promotion_decision": "awaiting_full_confirmation_run" if len(prediction_frame) < CALIBRATION_MIN_ANALYSIS_CASES else "confirmation_complete_review_metrics",
    })
    with (ARTIFACT_ROOT / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (ARTIFACT_ROOT / "selected_hypothesis_branch_policy.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "policy_name": f"final_capped_hypothesis_branching|trigger={BRANCH_TRIGGER_NAME}|threshold={BRANCH_TRIGGER_THRESHOLD:.2f}|max_branches={BRANCH_BUDGET}|branch_cap={BRANCH_REQUEST_CAP}|total_cap={HARD_TOTAL_REQUEST_CAP_PER_CASE}|continuation={ADAPTIVE_CONTINUATION_VALUE_THRESHOLD:.2f}|judge={BRANCH_JUDGE_NAME}",
            "selected_before_live_results": True,
            "source_analysis": "Final capped confirmation using Notebook 38 lessons, with close-confounder extra-root adjudication removed and explicit request caps added.",
            "inputs_used_for_decisions": ["visible evidence ledger", "learned branch-trigger MLP", "target-hypothesis assignment table", "partial-evidence diagnostic MLP", "Notebook 16 graph edges", "Notebook 19 Bayes likelihoods/priors"],
            "branch_trigger_threshold": float(BRANCH_TRIGGER_THRESHOLD),
            "adaptive_continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
            "max_branches": int(BRANCH_BUDGET),
            "base_request_cap": int(BASE_REQUEST_CAP),
            "branch_request_cap": int(BRANCH_REQUEST_CAP),
            "hard_total_request_cap_per_case": int(HARD_TOTAL_REQUEST_CAP_PER_CASE),
            "branch_mlp_validation_summary": branch_validation_summary.to_dict(orient="records"),
            "resolver_validation_summary": resolver_validation_summary.to_dict(orient="records"),
            "branch_profile_templates": BRANCH_PROFILE_TEMPLATES,
            "metrics": metrics,
        }, handle, indent=2)
    summary_frame = pd.DataFrame([{**metrics, "system": "notebook41_final_capped_branch_judge"}])
    summary_frame.to_csv(ARTIFACT_ROOT / "summary_metrics.csv", index=False)

    hard_rows = prediction_frame[(prediction_frame["branch_trigger_fired"].astype(bool)) | (prediction_frame["changed_prediction"].astype(bool)) | (~prediction_frame["correct"].astype(bool))]
    hard_audits = hard_rows.to_dict(orient="records")
    with (ARTIFACT_ROOT / "hard_case_branch_audits.json").open("w", encoding="utf-8") as handle:
        json.dump(hard_audits, handle, indent=2)

    print("Base correct:", metrics.get("base_num_correct"), "/", metrics.get("num_cases"))
    print("Final capped branch-selected correct:", metrics.get("num_correct"), "/", metrics.get("num_cases"))
    print("Wins/regressions:", metrics.get("wins_vs_base"), "/", metrics.get("regressions_vs_base"))
    print("Mean total branch requests:", metrics.get("mean_total_branch_requests"))
    display(summary_frame)
    display(prediction_frame[["case_id", "true_pathology", "base_predicted_pathology", "predicted_pathology", "selected_branch_id", "branch_trigger_fired", "branches_spawned", "correct", "base_correct", "total_branch_requests"]])
    return prediction_frame


branch_prediction_frame = run_hypothesis_forced_differential_branching()

# %% [markdown]
# ## Figures

# %%

if len(branch_prediction_frame):
    # Accuracy comparison.
    plt.figure(figsize=(7, 4))
    scores = [branch_prediction_frame["base_correct"].mean(), branch_prediction_frame["correct"].mean()]
    plt.bar(["Base branch", "Adaptive branching"], scores, color=["#6d7f95", "#2f8f83"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Notebook 41 final capped branch-judge accuracy")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=180)
    plt.close()

    paired_counts = branch_prediction_frame.apply(
        lambda row: "both_correct" if row.base_correct and row.correct else "branch_only_correct" if ((not row.base_correct) and row.correct) else "base_only_correct" if (row.base_correct and (not row.correct)) else "both_wrong",
        axis=1,
    ).value_counts().reindex(["both_correct", "branch_only_correct", "base_only_correct", "both_wrong"], fill_value=0)
    paired_counts.to_csv(ARTIFACT_ROOT / "paired_outcome_counts.csv")
    plt.figure(figsize=(7, 4))
    paired_counts.plot(kind="bar", color=["#6f9f76", "#2f8f83", "#c56b55", "#8a8f99"])
    plt.ylabel("Cases")
    plt.title("Paired base vs Adaptive branching outcomes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "paired_outcomes.png", dpi=180)
    plt.close()

    # Trigger probability diagnostics.
    plt.figure(figsize=(7, 4))
    plt.hist(branch_prediction_frame["branch_trigger_probability"], bins=np.linspace(0, 1, 16), color="#4477aa", edgecolor="white")
    plt.axvline(BRANCH_TRIGGER_THRESHOLD, color="#b24a3b", linestyle="--", linewidth=1.5)
    plt.xlabel("Branch-trigger MLP probability")
    plt.ylabel("Cases")
    plt.title("Branch trigger probability histogram")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "branch_trigger_probability_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for label, group in branch_prediction_frame.groupby("base_correct"):
        plt.hist(group["branch_trigger_probability"], bins=np.linspace(0, 1, 12), alpha=0.65, label=f"base_correct={label}")
    plt.axvline(BRANCH_TRIGGER_THRESHOLD, color="#222222", linestyle="--", linewidth=1.2)
    plt.xlabel("Branch-trigger MLP probability")
    plt.ylabel("Cases")
    plt.legend()
    plt.title("Trigger probability by base correctness")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "branch_trigger_probability_by_correctness.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(branch_threshold_sweep["threshold"], branch_threshold_sweep["branch_rate"], marker="o", color="#4477aa")
    plt.axvline(BRANCH_TRIGGER_THRESHOLD, color="#b24a3b", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Validation branch rate")
    plt.title("Branch rate vs threshold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "branch_rate_vs_threshold.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(branch_threshold_sweep["threshold"], branch_threshold_sweep["validation_objective"], marker="o", color="#2f8f83")
    plt.axvline(BRANCH_TRIGGER_THRESHOLD, color="#b24a3b", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Validation objective")
    plt.title("Validation gain vs threshold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "validation_gain_vs_threshold.png", dpi=180)
    plt.close()

    # Request cost plots.
    plt.figure(figsize=(7, 4))
    plt.hist(branch_prediction_frame["total_branch_requests"], bins=range(0, int(branch_prediction_frame["total_branch_requests"].max()) + 3), color="#4477aa", edgecolor="white")
    plt.xlabel("Total requests across base and spawned branches")
    plt.ylabel("Cases")
    plt.title("Total branch request cost")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "request_cost_distribution.png", dpi=180)
    plt.savefig(FIGURE_DIR / "total_branch_request_cost.png", dpi=180)
    plt.close()

    cost_frame = branch_prediction_frame.sort_values("total_branch_requests")[["case_id", "total_branch_requests", "branches_spawned"]].reset_index(drop=True)
    cost_frame.to_csv(ARTIFACT_ROOT / "total_branch_requests_by_case.csv", index=False)
    plt.figure(figsize=(9, 4))
    plt.bar(np.arange(len(cost_frame)), cost_frame["total_branch_requests"], color="#8a7aa8")
    plt.xlabel("Cases sorted by total branch requests")
    plt.ylabel("Total branch requests")
    plt.title("Total branch requests by case")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "total_branch_requests_by_case.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    branch_prediction_frame["branches_spawned"].value_counts().sort_index().plot(kind="bar", color="#8a7aa8")
    plt.xlabel("Branches spawned")
    plt.ylabel("Cases")
    plt.title("Branch trigger usage")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "branch_trigger_usage.png", dpi=180)
    plt.close()

    candidate_scores = pd.read_csv(ARTIFACT_ROOT / "candidate_level_live_scores.csv") if (ARTIFACT_ROOT / "candidate_level_live_scores.csv").exists() else pd.DataFrame()
    if len(candidate_scores):
        plt.figure(figsize=(7, 4))
        candidate_scores["candidate_role"].value_counts().plot(kind="bar", color="#6677aa")
        plt.ylabel("Candidate rows")
        plt.title("Candidate source counts")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "pseudo_candidate_source_counts.png", dpi=180)
        plt.close()

        if "resolver_score" in candidate_scores.columns:
            plt.figure(figsize=(7, 4))
            for selected, group in candidate_scores.groupby("selected_by_judge"):
                plt.hist(group["resolver_score"], bins=np.linspace(0, 1, 15), alpha=0.65, label=f"selected={selected}")
            plt.xlabel("Resolver score")
            plt.ylabel("Candidate rows")
            plt.legend()
            plt.title("Resolver score distribution")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "resolver_score_margin_by_outcome.png", dpi=180)
            plt.close()

        agreement = candidate_scores.assign(
            graph_top_is_pred=lambda df: df["graph_top1"].astype(str).eq(df["predicted_pathology"].astype(str)),
            bayes_top_is_pred=lambda df: df["bayes_top1"].astype(str).eq(df["predicted_pathology"].astype(str)),
            mlp_top_is_pred=lambda df: df["mlp_predicted_pathology"].astype(str).eq(df["predicted_pathology"].astype(str)),
        )[["graph_top_is_pred", "bayes_top_is_pred", "mlp_top_is_pred"]].corr()
        agreement.to_csv(ARTIFACT_ROOT / "graph_bayes_mlp_agreement_heatmap.csv")
        plt.figure(figsize=(5, 4))
        plt.imshow(agreement, vmin=-1, vmax=1, cmap="coolwarm")
        plt.xticks(range(3), agreement.columns, rotation=30, ha="right")
        plt.yticks(range(3), agreement.index)
        plt.colorbar(label="Correlation")
        plt.title("Graph/Bayes/MLP agreement heatmap")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "graph_bayes_mlp_agreement_heatmap.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(branch_prediction_frame["pair_coverage"], bins=np.linspace(0, 1, 12), color="#55a868", edgecolor="white")
    plt.xlabel("Top pairwise separation-root coverage")
    plt.ylabel("Cases")
    plt.title("Pairwise separation coverage by case")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pairwise_separation_coverage_by_case.png", dpi=180)
    plt.close()

    hypothesis_assignments = pd.read_csv(ARTIFACT_ROOT / "hypothesis_branch_assignments.csv") if (ARTIFACT_ROOT / "hypothesis_branch_assignments.csv").exists() else pd.DataFrame()
    if len(hypothesis_assignments):
        source_counts = hypothesis_assignments["role_kind"].fillna("not_spawned").value_counts()
        plt.figure(figsize=(7, 4))
        source_counts.plot(kind="bar", color="#4477aa")
        plt.ylabel("Assigned hypotheses")
        plt.title("Hypothesis branch role mix")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "hypothesis_branch_role_mix.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist(hypothesis_assignments["hypothesis_priority"], bins=12, color="#8a7aa8", edgecolor="white")
        plt.xlabel("Hypothesis assignment priority")
        plt.ylabel("Candidate hypotheses")
        plt.title("Assigned hypothesis priority distribution")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "hypothesis_assignment_priority_distribution.png", dpi=180)
        plt.close()

    hard_rank = branch_prediction_frame[(branch_prediction_frame["changed_prediction"]) | (~branch_prediction_frame["correct"])]
    hard_rank[["case_id", "true_pathology", "base_predicted_pathology", "predicted_pathology", "base_pred_graph_rank", "selected_pred_graph_rank", "base_pred_bayes_posterior", "selected_pred_bayes_posterior"]].to_csv(ARTIFACT_ROOT / "hard_case_rank_movement.csv", index=False)
    plt.figure(figsize=(8, 4))
    if len(hard_rank):
        x = np.arange(len(hard_rank))
        plt.plot(x, hard_rank["base_pred_graph_rank"], marker="o", label="base graph rank")
        plt.plot(x, hard_rank["selected_pred_graph_rank"], marker="o", label="selected graph rank")
        plt.xticks(x, hard_rank["case_id"], rotation=45, ha="right")
        plt.ylabel("Graph rank lower is better")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No changed or incorrect cases in this dry run", ha="center", va="center")
        plt.xticks([])
        plt.yticks([])
    plt.title("Hard-case rank movement")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hard_case_rank_movement.png", dpi=180)
    plt.close()

# Validation-only model diagnostics are useful even if live execution is skipped.
fpr, tpr, _ = roc_curve(validate_branch_features["branch_label"], validate_branch_features["branch_trigger_probability"])
precision, recall, _ = precision_recall_curve(validate_branch_features["branch_label"], validate_branch_features["branch_trigger_probability"])
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(ARTIFACT_ROOT / "branch_mlp_roc_curve.csv", index=False)
pd.DataFrame({"precision": precision, "recall": recall}).to_csv(ARTIFACT_ROOT / "branch_mlp_pr_curve.csv", index=False)
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="#4477aa")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Branch MLP ROC")
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color="#2f8f83")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Branch MLP PR")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "branch_mlp_roc_pr_curves.png", dpi=180)
plt.close()

prob_true, prob_pred = calibration_curve(validate_branch_features["branch_label"], validate_branch_features["branch_trigger_probability"], n_bins=10, strategy="quantile")
pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).to_csv(ARTIFACT_ROOT / "branch_mlp_calibration_curve.csv", index=False)
plt.figure(figsize=(5, 5))
plt.plot(prob_pred, prob_true, marker="o", color="#4477aa")
plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
plt.xlabel("Predicted probability")
plt.ylabel("Observed wrong-anchor rate")
plt.title("Branch MLP calibration")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "branch_mlp_calibration_curve.png", dpi=180)
plt.close()

try:
    perm = permutation_importance(
        branch_trigger_mlp,
        validate_branch_features[BRANCH_FEATURE_COLUMNS].sample(n=min(500, len(validate_branch_features)), random_state=RANDOM_SEED),
        validate_branch_features.loc[validate_branch_features[BRANCH_FEATURE_COLUMNS].sample(n=min(500, len(validate_branch_features)), random_state=RANDOM_SEED).index, "branch_label"].astype(int),
        n_repeats=2,
        random_state=RANDOM_SEED,
        scoring="average_precision",
    )
    importance = pd.DataFrame({"feature": BRANCH_FEATURE_COLUMNS, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(ARTIFACT_ROOT / "feature_importance_or_permutation_importance.csv", index=False)
    plt.figure(figsize=(8, 5))
    top_imp = importance.head(15).iloc[::-1]
    plt.barh(top_imp["feature"], top_imp["importance_mean"], color="#6677aa")
    plt.xlabel("Permutation importance, average precision")
    plt.title("Branch MLP feature importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance_or_permutation_importance.png", dpi=180)
    plt.close()
except Exception as exc:
    print("Permutation importance skipped:", exc)

# %% [markdown]
# ## Candidate-Pool Resolver And Top-K Differentials
# 
# This section runs after live branch collection. It restores top-3/top-5 reporting by ranking unique candidate diagnoses with the fixed Notebook `32` deployable-looking GBM resolver. No close-confounder rescue or extra-root adjudicator is applied in this final confirmation notebook.

# %%
SELECTED_LIVE_RESOLVER_NAME = "gradient_boosting_name_family"
SELECTED_LIVE_SCORE_COL = "score__gradient_boosting_name_family"

DISEASE_FAMILY_KEYWORDS = {
    "respiratory": ["bronch", "pneum", "copd", "asthma", "urti", "rhino", "sinus", "influenza", "croup", "epiglott", "whooping", "pertussis", "sarcoid", "tb", "tuberculosis"],
    "upper_airway": ["urti", "rhino", "sinus", "croup", "epiglott", "pharyng", "tonsil"],
    "cardiac": ["angina", "myocard", "infarction", "pericard", "psvt", "fibrillation", "heart", "edema", "embolism"],
    "neuro": ["stroke", "seizure", "headache", "migraine", "dystonic", "mening", "vertigo"],
    "gi": ["append", "divert", "gastro", "pancrea", "chole", "reflux", "ulcer", "obstruction", "gerd"],
    "skin_allergy": ["anaphyl", "dermat", "urtic", "cellul", "rash", "scombroid"],
    "metabolic_heme": ["anemia", "diabetes", "hypogly", "thyroid", "electrolyte"],
    "renal_uro": ["renal", "kidney", "stone", "pyelo", "cystitis", "urinary"],
    "msk": ["fracture", "sprain", "arthritis", "back pain", "gout"],
    "infection": ["infection", "sepsis", "cellul", "abscess", "influenza", "pneum", "mening", "tb", "tuberculosis"],
}

BASE_FEATURES_FOR_RESOLVER = [
    "candidate_order",
    "candidate_graph_score",
    "candidate_graph_posterior",
    "candidate_graph_rank",
    "candidate_graph_positive_support",
    "candidate_graph_contradiction",
    "candidate_bayes_log_score",
    "candidate_bayes_posterior",
    "candidate_bayes_rank",
    "candidate_mlp_posterior",
    "candidate_mlp_rank",
    "is_base_candidate",
    "is_branch_candidate",
    "is_pseudo_candidate",
    "request_count",
    "visible_root_count",
    "branch_trigger_probability",
    "pair_coverage",
    "pair_missing_utility",
]
LIVE_TO_TRAIN_COLUMNS = {
    "case_id": "case_id",
    "branch_id": "branch_id",
    "candidate_role": "candidate_role",
    "branch_role_name": "branch_role_name",
    "true_pathology": "true_pathology",
    "predicted_pathology": "candidate_pathology",
    "correct": "candidate_label",
    "resolver_candidate_order": "candidate_order",
    "resolver_candidate_graph_score": "candidate_graph_score",
    "resolver_candidate_graph_posterior": "candidate_graph_posterior",
    "resolver_candidate_graph_rank": "candidate_graph_rank",
    "resolver_candidate_graph_positive_support": "candidate_graph_positive_support",
    "resolver_candidate_graph_contradiction": "candidate_graph_contradiction",
    "resolver_candidate_bayes_log_score": "candidate_bayes_log_score",
    "resolver_candidate_bayes_posterior": "candidate_bayes_posterior",
    "resolver_candidate_bayes_rank": "candidate_bayes_rank",
    "resolver_candidate_mlp_posterior": "candidate_mlp_posterior",
    "resolver_candidate_mlp_rank": "candidate_mlp_rank",
    "resolver_is_base_candidate": "is_base_candidate",
    "resolver_is_branch_candidate": "is_branch_candidate",
    "resolver_is_pseudo_candidate": "is_pseudo_candidate",
    "resolver_request_count": "request_count",
    "resolver_visible_root_count": "visible_root_count",
    "resolver_branch_trigger_probability": "branch_trigger_probability",
    "resolver_pair_coverage": "pair_coverage",
    "resolver_pair_missing_utility": "pair_missing_utility",
    "resolver_score": "notebook30_resolver_score",
    "raw_bayes_judge_score": "raw_bayes_judge_score",
    "selected_by_judge": "notebook30_selected_candidate",
}
EXCLUDED_RESOLVER_FEATURE_COLUMNS = {
    "split",
    "synthetic_state_id",
    "case_id",
    "branch_id",
    "candidate_role",
    "branch_role_name",
    "true_pathology",
    "candidate_pathology",
    "candidate_label",
    "candidate_source",
    "notebook30_selected_candidate",
}


def clean_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=-1e6, upper=1e6)


def clean_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return clean_numeric_frame(df, columns).to_numpy(dtype=np.float32)


def make_live_training_schema(live_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for source_col, target_col in LIVE_TO_TRAIN_COLUMNS.items():
        if source_col in live_df.columns:
            out[target_col] = live_df[source_col]
    out["synthetic_state_id"] = out["case_id"]
    out["split"] = "test"
    out["candidate_source"] = out["candidate_role"]
    return out


def add_name_family_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["candidate_pathology"].fillna("").str.lower()
    features: dict[str, Any] = {"candidate_name_len": text.str.len().astype(float)}
    for token in ["acute", "chronic", "infection", "viral", "bacterial"]:
        features[f"name_has_{token}"] = text.str.contains(token, regex=False).astype(float)
    for family, keywords in DISEASE_FAMILY_KEYWORDS.items():
        pattern = "|".join(re.escape(token) for token in keywords)
        features[f"family_{family}"] = text.str.contains(pattern, regex=True).astype(float)
    feature_df = pd.DataFrame(features, index=out.index)
    out = pd.concat([out, feature_df], axis=1)
    group = out.groupby("synthetic_state_id")
    competitor_features: dict[str, Any] = {}
    family_cols = [col for col in feature_df.columns if col.startswith("family_")]
    for col in family_cols + ["name_has_acute", "name_has_chronic"]:
        pool_count = group[col].transform("sum").astype(float)
        competitor_features[f"{col}_pool_count"] = pool_count
        competitor_features[f"{col}_has_competitor"] = ((pool_count - out[col]) > 0).astype(float)
    out = pd.concat([out, pd.DataFrame(competitor_features, index=out.index)], axis=1)
    group = out.groupby("synthetic_state_id")
    out["acute_vs_chronic_pool_conflict"] = (
        ((out["name_has_acute"] > 0) & (group["name_has_chronic"].transform("sum") > 0))
        | ((out["name_has_chronic"] > 0) & (group["name_has_acute"].transform("sum") > 0))
    ).astype(float)
    return out.copy()


def add_resolver_features(df: pd.DataFrame, *, add_name_features: bool = False) -> pd.DataFrame:
    out = df.copy()
    for col in BASE_FEATURES_FOR_RESOLVER:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[BASE_FEATURES_FOR_RESOLVER] = out[BASE_FEATURES_FOR_RESOLVER].replace([np.inf, -np.inf], np.nan)
    for col in BASE_FEATURES_FOR_RESOLVER:
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)
    engineered: dict[str, Any] = {}
    eps = 1e-6
    for col in ["candidate_graph_posterior", "candidate_bayes_posterior", "candidate_mlp_posterior"]:
        clipped = np.clip(out[col].astype(float), eps, 1 - eps)
        engineered[f"{col}_logit"] = np.log(clipped / (1 - clipped))
    for col in ["candidate_graph_rank", "candidate_bayes_rank", "candidate_mlp_rank", "candidate_order"]:
        engineered[f"{col}_inv"] = 1.0 / (1.0 + out[col].clip(lower=0))
        engineered[f"{col}_neg"] = -out[col]
    out = pd.concat([out, pd.DataFrame(engineered, index=out.index)], axis=1)
    group = out.groupby("synthetic_state_id")
    group_features: dict[str, Any] = {"candidate_pool_size": group["candidate_pathology"].transform("count")}
    group_context_cols = [
        "candidate_graph_score",
        "candidate_graph_posterior",
        "candidate_bayes_log_score",
        "candidate_bayes_posterior",
        "candidate_mlp_posterior",
        "candidate_graph_rank_neg",
        "candidate_bayes_rank_neg",
        "candidate_mlp_rank_neg",
        "candidate_order_neg",
    ]
    for col in group_context_cols:
        group_max = group[col].transform("max")
        group_features[f"{col}_group_max"] = group_max
        group_features[f"{col}_minus_group_max"] = out[col] - group_max
    posterior_cols = ["candidate_graph_posterior", "candidate_bayes_posterior", "candidate_mlp_posterior"]
    posterior_frame = out[posterior_cols]
    group_features["graph_minus_bayes_rank"] = out["candidate_graph_rank"] - out["candidate_bayes_rank"]
    group_features["graph_minus_mlp_rank"] = out["candidate_graph_rank"] - out["candidate_mlp_rank"]
    group_features["bayes_minus_mlp_rank"] = out["candidate_bayes_rank"] - out["candidate_mlp_rank"]
    group_features["graph_bayes_score_product"] = out["candidate_graph_score"] * out["candidate_bayes_posterior"]
    group_features["mlp_bayes_product"] = out["candidate_mlp_posterior"] * out["candidate_bayes_posterior"]
    group_features["graph_bayes_mlp_product"] = out["candidate_graph_posterior"] * out["candidate_bayes_posterior"] * out["candidate_mlp_posterior"]
    group_features["rank_consensus_sum"] = out["candidate_graph_rank"] + out["candidate_bayes_rank"] + out["candidate_mlp_rank"]
    group_features["posterior_mean"] = posterior_frame.mean(axis=1)
    group_features["posterior_min"] = posterior_frame.min(axis=1)
    group_features["posterior_max"] = posterior_frame.max(axis=1)
    group_features["posterior_range"] = posterior_frame.max(axis=1) - posterior_frame.min(axis=1)
    out = pd.concat([out, pd.DataFrame(group_features, index=out.index)], axis=1)
    if add_name_features:
        out = add_name_family_features(out)
    return out.copy()


def rank_unique_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in scored.groupby("case_id", sort=False):
        sorted_group = group.sort_values(
            [SELECTED_LIVE_SCORE_COL, "resolver_candidate_graph_posterior", "resolver_candidate_bayes_posterior", "resolver_candidate_mlp_posterior"],
            ascending=[False, False, False, False],
        ).copy()
        unique = sorted_group.drop_duplicates("candidate_pathology", keep="first").copy()
        unique["candidate_rank"] = np.arange(1, len(unique) + 1)
        rows.append(unique)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_topk(name: str, predictions: pd.DataFrame, ranked_col: str, pred_col: str, request_col: str | None = None, total_request_col: str | None = None) -> dict[str, Any]:
    true_labels = predictions["true_pathology"].astype(str).tolist()
    ranked_lists = predictions[ranked_col].map(safe_parse_list).tolist()
    preds = predictions[pred_col].astype(str).tolist()
    row = {
        "system": name,
        "num_cases": int(len(predictions)),
        "num_correct": int(sum(t == p for t, p in zip(true_labels, preds))),
        "accuracy": float(np.mean([t == p for t, p in zip(true_labels, preds)])) if len(predictions) else np.nan,
        "top3_accuracy": top_k_accuracy_from_lists(true_labels, ranked_lists, 3),
        "top5_accuracy": top_k_accuracy_from_lists(true_labels, ranked_lists, 5),
    }
    if request_col and request_col in predictions.columns:
        row["mean_selected_requests"] = float(pd.to_numeric(predictions[request_col], errors="coerce").mean())
    if total_request_col and total_request_col in predictions.columns:
        row["mean_total_requests"] = float(pd.to_numeric(predictions[total_request_col], errors="coerce").mean())
        row["p90_total_requests"] = float(pd.to_numeric(predictions[total_request_col], errors="coerce").quantile(0.90))
        row["max_total_requests"] = float(pd.to_numeric(predictions[total_request_col], errors="coerce").max())
    return row


if len(branch_prediction_frame):
    train_validate_raw = pd.read_csv(ARTIFACT_ROOT / "candidate_resolver_train_validate_features.csv")
    live_candidates_raw = pd.read_csv(ARTIFACT_ROOT / "candidate_level_live_scores.csv")
    live_schema = make_live_training_schema(live_candidates_raw)
    train_validate_name = add_resolver_features(train_validate_raw, add_name_features=True)
    live_name = add_resolver_features(live_schema, add_name_features=True)
    name_feature_columns = [
        col for col in train_validate_name.columns
        if col not in EXCLUDED_RESOLVER_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(train_validate_name[col])
    ]
    for col in name_feature_columns:
        if col not in live_name.columns:
            live_name[col] = 0.0
    train_mask = train_validate_name["split"].eq("train")
    validate_mask = train_validate_name["split"].eq("validate")
    selected_model = GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=3, random_state=RANDOM_SEED)
    selected_model.fit(clean_matrix(train_validate_name.loc[train_mask], name_feature_columns), train_validate_name.loc[train_mask, "candidate_label"].astype(int))
    validate_scores = selected_model.predict_proba(clean_matrix(train_validate_name.loc[validate_mask], name_feature_columns))[:, 1]
    live_scores = selected_model.predict_proba(clean_matrix(live_name, name_feature_columns))[:, 1]
    resolver_validation_summary = pd.DataFrame([{
        "resolver_name": SELECTED_LIVE_RESOLVER_NAME,
        "feature_count": int(len(name_feature_columns)),
        "validate_rows": int(validate_mask.sum()),
        "validate_candidate_pool_recall": float((train_validate_name.loc[validate_mask].groupby("synthetic_state_id")["candidate_label"].sum() > 0).mean()),
        "validate_row_auc": float(roc_auc_score(train_validate_name.loc[validate_mask, "candidate_label"].astype(int), validate_scores)),
        "validate_row_average_precision": float(average_precision_score(train_validate_name.loc[validate_mask, "candidate_label"].astype(int), validate_scores)),
    }])
    resolver_validation_summary.to_csv(ARTIFACT_ROOT / "live_resolver_validation_summary.csv", index=False)

    live_scored = live_candidates_raw.copy()
    live_scored["candidate_pathology"] = live_schema["candidate_pathology"].astype(str).values
    live_scored["candidate_label"] = live_schema["candidate_label"].astype(bool).values
    live_scored[SELECTED_LIVE_SCORE_COL] = live_scores
    ranked_candidates = rank_unique_candidates(live_scored)

    final_rows = []
    resolver_trace_rows = []
    prediction_lookup = pd.read_csv(ARTIFACT_ROOT / "predictions.csv").set_index("case_id")
    for case_id, group in ranked_candidates.groupby("case_id", sort=False):
        group = group.sort_values("candidate_rank")
        top = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        ranked_list = group["candidate_pathology"].astype(str).tolist()
        anchor = ranked_list[0]
        challenger = ranked_list[1] if len(ranked_list) > 1 else ""
        score_margin = float(top[SELECTED_LIVE_SCORE_COL] - (second[SELECTED_LIVE_SCORE_COL] if second is not None else 0.0))
        selected_row = prediction_lookup.loc[case_id].to_dict()
        final_prediction = anchor
        final_ranked = ranked_list
        total_requests = float(selected_row.get("total_branch_requests", np.nan))
        selected_requests = float(top.get("num_requests", selected_row.get("num_requests", np.nan)))
        final_rows.append({
            "case_id": case_id,
            "true_pathology": str(top["true_pathology"]),
            "selected_pathology_by_resolver": anchor,
            "predicted_pathology": final_prediction,
            "correct": final_prediction == str(top["true_pathology"]),
            "ranked_differential": json.dumps(final_ranked),
            "candidate_pool_has_true": bool(group["candidate_label"].any()),
            "candidate_pool_rows": int(len(group)),
            "selected_branch_id": str(top.get("branch_id", "")),
            "selected_candidate_role": str(top.get("candidate_role", "")),
            "selected_resolver_score": float(top[SELECTED_LIVE_SCORE_COL]),
            "score_margin_top2": score_margin,
            "num_requests_selected": selected_requests,
            "total_branch_requests": total_requests,
        })
        resolver_trace_rows.append({
            "case_id": case_id,
            "resolver_top1": anchor,
            "resolver_top2": challenger,
            "score_margin_top2": score_margin,
            "final_prediction": final_prediction,
            "candidate_pool_has_true": bool(group["candidate_label"].any()),
            "candidate_pool_rows": int(len(group)),
        })

    final_predictions = pd.DataFrame(final_rows)
    resolver_trace = pd.DataFrame(resolver_trace_rows)
    ranked_candidates.to_csv(ARTIFACT_ROOT / "candidate_pool_topk_rankings.csv", index=False)
    live_scored.to_csv(ARTIFACT_ROOT / "candidate_level_live_resolver_scores.csv", index=False)
    final_predictions.to_csv(ARTIFACT_ROOT / "adaptive_live_final_predictions.csv", index=False)
    resolver_trace.to_csv(ARTIFACT_ROOT / "final_resolver_trace.csv", index=False)

    branch_predictions = pd.read_csv(ARTIFACT_ROOT / "predictions.csv")
    base_eval = branch_predictions.copy()
    base_eval["base_prediction_for_eval"] = base_eval["base_predicted_pathology"]
    base_eval["base_ranked_for_eval"] = base_eval["base_ranked_differential"]
    topk_summary = pd.DataFrame([
        summarize_topk("base_notebook13_style_branch", base_eval.rename(columns={"base_ranked_for_eval": "ranked_eval"}), "ranked_eval", "base_prediction_for_eval", "num_requests_base", "num_requests_base"),
        summarize_topk("notebook41_final_capped_branch_judge_selected", branch_predictions, "ranked_differential", "predicted_pathology", "num_requests_selected", "total_branch_requests"),
        summarize_topk("notebook41_final_capped_gbm_resolver", final_predictions, "ranked_differential", "predicted_pathology", "num_requests_selected", "total_branch_requests"),
    ])
    topk_summary["candidate_pool_recall"] = np.nan
    topk_summary.loc[topk_summary["system"].eq("notebook41_final_capped_gbm_resolver"), "candidate_pool_recall"] = float(final_predictions["candidate_pool_has_true"].mean())
    topk_summary.to_csv(ARTIFACT_ROOT / "topk_summary.csv", index=False)

    final_metrics = topk_summary[topk_summary["system"].eq("notebook41_final_capped_gbm_resolver")].iloc[0].to_dict()
    final_metrics.update({
        "run_name": RUN_NAME,
        "final_confirmation_run": True,
        "artifact_root": str(ARTIFACT_ROOT),
        "resolver_name": SELECTED_LIVE_RESOLVER_NAME,
        "close_confounder_extra_root_layer_enabled": False,
        "candidate_pool_recall_count": int(final_predictions["candidate_pool_has_true"].sum()),
        "candidate_pool_recall": float(final_predictions["candidate_pool_has_true"].mean()),
        "branch_trigger_rate": float(branch_predictions["branch_trigger_fired"].astype(bool).mean()),
        "mean_branches_spawned": float(branch_predictions["branches_spawned"].mean()),
        "branches_spawned_distribution": {str(k): int(v) for k, v in branch_predictions["branches_spawned"].value_counts().sort_index().items()},
    })
    with (ARTIFACT_ROOT / "metrics_final.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)
    with (ARTIFACT_ROOT / "selected_final_capped_policy.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "policy_name": "final_capped_hypothesis_branching_confirmation_v1",
            "selected_before_live_results": True,
            "cohort_rule": "two held-out test cases per pathology plus two additional held-out cases, excluding prior live benchmark cohorts when possible",
            "llm_model": LLM_MODEL,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "branch_trigger_threshold": float(BRANCH_TRIGGER_THRESHOLD),
            "max_branches": int(BRANCH_BUDGET),
            "continuation_value_threshold": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
            "base_request_cap": int(BASE_REQUEST_CAP),
            "branch_request_cap": int(BRANCH_REQUEST_CAP),
            "hard_total_request_cap_per_case": int(HARD_TOTAL_REQUEST_CAP_PER_CASE),
            "resolver_name": SELECTED_LIVE_RESOLVER_NAME,
            "close_confounder_rule": "excluded_from_notebook41_final_confirmation",
            "metrics_final": final_metrics,
            "target_candidate_pool_recall": FINAL_TARGET_CANDIDATE_POOL_RECALL,
        }, handle, indent=2)

    display(topk_summary)
    display(final_predictions.sort_values(["correct", "score_margin_top2"]).head(12))
else:
    final_predictions = pd.DataFrame()
    ranked_candidates = pd.DataFrame()
    topk_summary = pd.DataFrame()
    print("No live or dry-run predictions are available yet; run the live cells to produce final top-k artifacts.")

# %% [markdown]
# ## Final Confirmation Diagnostics
# 
# This section summarizes where the final capped run still fails. It does not select thresholds or alter predictions.

# %%
def calibration_truth_rank_table(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in ranked.groupby("case_id", sort=False):
        true_label = str(group["true_pathology"].iloc[0])
        hit = group[group["candidate_pathology"].astype(str).eq(true_label)]
        rows.append({
            "case_id": case_id,
            "true_pathology": true_label,
            "truth_rank": int(hit["candidate_rank"].iloc[0]) if len(hit) else np.nan,
            "candidate_pool_size": int(group["candidate_pathology"].nunique()),
            "candidate_pool_has_true": bool(len(hit)),
        })
    return pd.DataFrame(rows)


def recall_for_candidate_subset(ranked: pd.DataFrame, subset_name: str, mask: pd.Series) -> dict[str, Any]:
    subset = ranked[mask].copy()
    rows = []
    for case_id, group in ranked.groupby("case_id", sort=False):
        true_label = str(group["true_pathology"].iloc[0])
        subgroup = subset[subset["case_id"].eq(case_id)]
        rows.append({
            "case_id": case_id,
            "has_true": bool((subgroup["candidate_pathology"].astype(str) == true_label).any()),
            "pool_size": int(subgroup["candidate_pathology"].nunique()),
        })
    frame = pd.DataFrame(rows)
    return {
        "subset_name": subset_name,
        "num_cases": int(frame["case_id"].nunique()),
        "candidate_pool_recall_count": int(frame["has_true"].sum()),
        "candidate_pool_recall": float(frame["has_true"].mean()) if len(frame) else np.nan,
        "mean_pool_size": float(frame["pool_size"].mean()) if len(frame) else np.nan,
        "max_pool_size": int(frame["pool_size"].max()) if len(frame) else 0,
    }


if len(final_predictions):
    branch_predictions = pd.read_csv(ARTIFACT_ROOT / "predictions.csv")
    paired = final_predictions.merge(
        branch_predictions[
            [
                "case_id",
                "true_pathology",
                "base_predicted_pathology",
                "base_correct",
                "predicted_pathology",
                "correct",
                "branch_trigger_probability",
                "branch_trigger_fired",
                "branches_spawned",
                "suspicion_signal_count",
                "signal_llm_mlp_disagree",
                "signal_cap_hit",
                "signal_uncertain_mlp",
                "signal_graph_conflict",
                "signal_bayes_conflict",
                "signal_ledger_disagrees",
                "num_requests_base",
            ]
        ],
        on=["case_id", "true_pathology"],
        how="left",
        suffixes=("_final", "_branchjudge"),
    )
    paired["base_correct"] = paired["base_correct"].astype(bool)
    paired["branchjudge_correct"] = paired["correct_branchjudge"].astype(bool)
    paired["final_correct"] = paired["correct_final"].astype(bool)
    paired["final_outcome_vs_base"] = np.select(
        [
            (~paired["base_correct"]) & paired["final_correct"],
            paired["base_correct"] & (~paired["final_correct"]),
            paired["base_correct"] & paired["final_correct"],
        ],
        ["win", "regression", "both_correct"],
        default="both_wrong",
    )
    paired.to_csv(ARTIFACT_ROOT / "final_confirmation_paired_outcomes.csv", index=False)

    truth_rank = calibration_truth_rank_table(ranked_candidates)
    truth_rank.to_csv(ARTIFACT_ROOT / "final_confirmation_truth_rank_analysis.csv", index=False)

    failure_rows = []
    for row in paired[~paired["final_correct"]].itertuples(index=False):
        failure_rows.append({
            "case_id": row.case_id,
            "true_pathology": row.true_pathology,
            "failure_mode": "candidate_pool_miss" if not bool(row.candidate_pool_has_true) else "resolver_miss_truth_in_pool",
            "base_prediction": row.base_predicted_pathology,
            "branchjudge_prediction": row.predicted_pathology_branchjudge,
            "final_prediction": row.predicted_pathology_final,
            "branch_trigger_probability": float(row.branch_trigger_probability),
            "branch_trigger_fired": bool(row.branch_trigger_fired),
            "branches_spawned": int(row.branches_spawned),
            "score_margin_top2": float(row.score_margin_top2),
            "suspicion_signal_count": int(row.suspicion_signal_count),
            "signal_llm_mlp_disagree": bool(row.signal_llm_mlp_disagree),
            "signal_cap_hit": bool(row.signal_cap_hit),
            "signal_uncertain_mlp": bool(row.signal_uncertain_mlp),
            "signal_graph_conflict": bool(row.signal_graph_conflict),
            "signal_bayes_conflict": bool(row.signal_bayes_conflict),
            "signal_ledger_disagrees": bool(row.signal_ledger_disagrees),
        })
    failure_modes = pd.DataFrame(failure_rows)
    failure_modes.to_csv(ARTIFACT_ROOT / "final_confirmation_failure_modes.csv", index=False)

    threshold_rows = []
    total_pool_misses = max(int((~paired["candidate_pool_has_true"].astype(bool)).sum()), 1)
    total_final_errors = max(int((~paired["final_correct"].astype(bool)).sum()), 1)
    for threshold in np.round(np.linspace(0.0, 0.95, 20), 3):
        triggered = paired[paired["branch_trigger_probability"].ge(threshold)]
        threshold_rows.append({
            "threshold": float(threshold),
            "cases_triggered": int(len(triggered)),
            "case_rate": float(len(triggered) / max(len(paired), 1)),
            "base_incorrect_triggered": int((~triggered["base_correct"].astype(bool)).sum()),
            "candidate_pool_misses_triggered": int((~triggered["candidate_pool_has_true"].astype(bool)).sum()),
            "candidate_pool_miss_capture_rate": float((~triggered["candidate_pool_has_true"].astype(bool)).sum() / total_pool_misses),
            "final_errors_triggered": int((~triggered["final_correct"].astype(bool)).sum()),
            "final_error_capture_rate": float((~triggered["final_correct"].astype(bool)).sum() / total_final_errors),
        })
    branch_threshold_sweep_live = pd.DataFrame(threshold_rows)
    branch_threshold_sweep_live.to_csv(ARTIFACT_ROOT / "final_confirmation_branch_trigger_threshold_diagnostics.csv", index=False)

    margin_rows = []
    for threshold in np.round(np.linspace(0.0, 1.0, 21), 3):
        flagged = paired[paired["score_margin_top2"].le(threshold)]
        margin_rows.append({
            "score_margin_threshold": float(threshold),
            "cases_flagged": int(len(flagged)),
            "case_rate": float(len(flagged) / max(len(paired), 1)),
            "final_errors_flagged": int((~flagged["final_correct"].astype(bool)).sum()),
            "final_error_capture_rate": float((~flagged["final_correct"].astype(bool)).sum() / total_final_errors),
            "resolver_misses_flagged": int(((~flagged["final_correct"].astype(bool)) & flagged["candidate_pool_has_true"].astype(bool)).sum()),
        })
    resolver_margin_sweep = pd.DataFrame(margin_rows)
    resolver_margin_sweep.to_csv(ARTIFACT_ROOT / "final_confirmation_resolver_margin_diagnostics.csv", index=False)

    source_rows = [
        recall_for_candidate_subset(ranked_candidates, "base_only", ranked_candidates["candidate_role"].eq("base")),
        recall_for_candidate_subset(ranked_candidates, "base_plus_pseudo", ranked_candidates["candidate_role"].isin(["base", "pseudo"])),
        recall_for_candidate_subset(ranked_candidates, "base_plus_real_branch", ranked_candidates["candidate_role"].isin(["base", "branch"])),
        recall_for_candidate_subset(ranked_candidates, "all_candidates", pd.Series(True, index=ranked_candidates.index)),
    ]
    candidate_source_recall = pd.DataFrame(source_rows)
    candidate_source_recall.to_csv(ARTIFACT_ROOT / "final_confirmation_candidate_source_recall.csv", index=False)

    confirmation_summary = {
        "run_name": RUN_NAME,
        "confirmation_cases": int(len(paired)),
        "target_candidate_pool_recall": float(FINAL_TARGET_CANDIDATE_POOL_RECALL),
        "base_correct": int(paired["base_correct"].sum()),
        "branchjudge_correct": int(paired["branchjudge_correct"].sum()),
        "final_correct": int(paired["final_correct"].sum()),
        "candidate_pool_recall_count": int(paired["candidate_pool_has_true"].sum()),
        "candidate_pool_recall": float(paired["candidate_pool_has_true"].mean()),
        "mean_candidate_pool_size": float(truth_rank["candidate_pool_size"].mean()),
        "branch_trigger_threshold_used": float(BRANCH_TRIGGER_THRESHOLD),
        "continuation_value_threshold_used": float(ADAPTIVE_CONTINUATION_VALUE_THRESHOLD),
        "base_request_cap": int(BASE_REQUEST_CAP),
        "branch_request_cap": int(BRANCH_REQUEST_CAP),
        "hard_total_request_cap_per_case": int(HARD_TOTAL_REQUEST_CAP_PER_CASE),
        "branch_trigger_diagnostics_file": "final_confirmation_branch_trigger_threshold_diagnostics.csv",
        "resolver_margin_diagnostics_file": "final_confirmation_resolver_margin_diagnostics.csv",
        "candidate_source_recall_file": "final_confirmation_candidate_source_recall.csv",
        "use_rule": "Final confirmation diagnostics only. Do not retune this notebook from these labels.",
    }
    write_json(ARTIFACT_ROOT / "final_confirmation_summary.json", confirmation_summary)
    display(pd.DataFrame([confirmation_summary]))
    display(candidate_source_recall)
else:
    print("Final confirmation diagnostics wait for live predictions.")

# %% [markdown]
# ## Final Top-K Figures

# %%
if len(topk_summary):
    metric_plot = topk_summary.set_index("system")[["accuracy", "top3_accuracy", "top5_accuracy"]]
    plt.figure(figsize=(9, 4.8))
    metric_plot.plot(kind="bar", ax=plt.gca(), color=["#3b6ea8", "#6aa86a", "#b88a3b"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Notebook 41 final capped top-1/top-3/top-5 comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "topk_accuracy_comparison.png", dpi=180)
    plt.close()

    if "branches_spawned" in branch_prediction_frame.columns:
        plt.figure(figsize=(7, 4))
        branch_prediction_frame["branches_spawned"].value_counts().sort_index().plot(kind="bar", color="#4f7f9f")
        plt.xlabel("Branches spawned per case")
        plt.ylabel("Cases")
        plt.title("Adaptive branch usage on final confirmation cohort")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "adaptive_branch_usage_distribution.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7, 4))
    final_predictions["total_branch_requests"].plot(kind="hist", bins=18, color="#6d7f95", edgecolor="white")
    plt.xlabel("Total LLM evidence requests across base and branches")
    plt.ylabel("Cases")
    plt.title("Notebook 41 final capped total request distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_total_request_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    final_predictions["candidate_pool_has_true"].value_counts().sort_index().plot(kind="bar", color=["#b65d54", "#3b8f6a"])
    plt.xticks([0, 1], ["miss", "hit"], rotation=0)
    plt.ylabel("Cases")
    plt.title("Candidate-pool recall")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "candidate_pool_recall_counts.png", dpi=180)
    plt.close()

    print("Final top-k figures written to", FIGURE_DIR)

# %% [markdown]
# ## Final Summary And Artifact Contract

# %%
required_artifacts = [
    "benchmark_cases.csv",
    "balanced_pathology_case_counts.csv",
    "reference_case_exclusion_summary.csv",
    "reference_summary.csv",
    "resolved_run_config.json",
    "branch_mlp_train_validate_features.csv",
    "candidate_resolver_train_validate_features.csv",
    "branch_mlp_validation_summary.csv",
    "branch_mlp_threshold_sweep.csv",
    "candidate_resolver_validation_summary.csv",
    "pairwise_evidence_separation_graph.csv",
]
if len(branch_prediction_frame):
    required_artifacts += [
        "predictions.csv",
        "paired_notebook13_vs_notebook41.csv",
        "branch_case_results.csv",
        "candidate_level_live_scores.csv",
        "candidate_level_live_resolver_scores.csv",
        "candidate_pool_topk_rankings.csv",
        "adaptive_live_final_predictions.csv",
        "final_resolver_trace.csv",
        "adaptive_branch_decision_trace.csv",
        "topk_summary.csv",
        "metrics.json",
        "metrics_final.json",
        "final_confirmation_paired_outcomes.csv",
        "final_confirmation_truth_rank_analysis.csv",
        "final_confirmation_failure_modes.csv",
        "final_confirmation_branch_trigger_threshold_diagnostics.csv",
        "final_confirmation_resolver_margin_diagnostics.csv",
        "final_confirmation_candidate_source_recall.csv",
        "final_confirmation_summary.json",
        "summary_metrics.csv",
        "selected_hypothesis_branch_policy.json",
        "selected_final_capped_policy.json",
        "hypothesis_branch_assignments.csv",
        "traces.jsonl",
        "branch_traces.jsonl",
        "raw_api_responses.jsonl",
        "hard_case_branch_audits.json",
    ]
missing = [name for name in required_artifacts if not (ARTIFACT_ROOT / name).exists()]
if missing:
    raise AssertionError(f"Missing artifacts: {missing}")

with (ARTIFACT_ROOT / "resolved_run_config.json").open() as handle:
    config = json.load(handle)
config.update({
    "artifact_contract": required_artifacts,
    "completed_prediction_rows": int(len(branch_prediction_frame)),
    "figures_dir": str(FIGURE_DIR),
    "final_topk_artifacts": bool(len(topk_summary)),
})
with (ARTIFACT_ROOT / "resolved_run_config.json").open("w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)

print("Notebook 41 final capped confirmation artifact contract OK")
print("Artifact root:", ARTIFACT_ROOT)
if len(topk_summary):
    display(topk_summary)
elif len(branch_prediction_frame):
    display(pd.read_csv(ARTIFACT_ROOT / "summary_metrics.csv"))
