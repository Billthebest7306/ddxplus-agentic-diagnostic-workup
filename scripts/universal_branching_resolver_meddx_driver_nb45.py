from __future__ import annotations

# %% [markdown]
# # Notebook 45: Universal Branching Resolver MEDDx Driver
#
# This notebook ports the main DDXPlus architecture ideas into the multi-dataset MEDDx-style harness:
#
# 1. universal patient-profile/evidence ledger over DDXPlus, iCraft-MD, and RareBench
# 2. cap-aware early stopping at MEDDx budgets 5/10/15
# 3. DDXPlus partial-evidence MLP monitor when structured roots are available
# 4. universal hypothesis-forced branch continuation using unused budget
# 5. candidate-pool resolver over base, branches, casebase, and graph/HPO support
# 6. conservative graph/discriminator gates that block unsupported RareBench regressions
#
# This is still a live pilot harness. Dry-run validates wiring only; live metrics are required for claims.

# %%
# Interactive run controls.
# Keep this dry by default; set RUN_LIVE_API=True only when intentionally spending API budget.

RUN_LIVE_API = False
ALLOW_DRY_RUN_BENCHMARK = True
RESUME_IF_AVAILABLE = True

LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4.1-mini"
LLM_API_KEY = ""
INTERACTIVE_API_KEY_BOOTSTRAP = True
TEMPERATURE = 0.0
TOP_P = 1.0

RUN_VERSION_SUFFIX = "v1_pilot4"
RANDOM_SEED = 4242

# MEDDx-style budgets. Active config is a cautious first live pilot for the new branch/resolver stack.
# For a larger run after this pilot passes, use a fresh suffix and raise LIVE_TOTAL_MAX_CASES.
MEDDX_REFERENCE_BUDGETS = [5, 10, 15]
LIVE_BUDGETS_TO_RUN = [5, 10, 15]
DRY_RUN_ALL_BUDGETS = False
# Global unique-case cap across all loaded datasets, not per dataset.
LIVE_TOTAL_MAX_CASES = 3
DRY_RUN_TOTAL_MAX_CASES = 3

ENABLED_DATASETS = ["ddxplus", "icraft_md", "rarebench"]
REQUIRE_ALL_ENABLED_DATASETS = True

DATASET_DIR_OVERRIDE = None

# MEDDxAgent benchmark data defaults. These are plain variables so a notebook user can edit them directly.
MEDDXAGENT_DATA_ROOT = "external/meddxagent/ddxdriver/benchmarks/data"
ICRAFT_MD_PATH = "external/meddxagent/ddxdriver/benchmarks/data/icraftmd/all_craft_md.jsonl"
RAREBENCH_MAPPING_DIR = "external/meddxagent/ddxdriver/benchmarks/data/rarebench"
RAREBENCH_DATA_ZIP_PATH = "artifacts/universal_meddx/cache/rarebench_data.zip"
RAREBENCH_DATA_ZIP_URL = "https://huggingface.co/datasets/chenxz/RareBench/resolve/main/data.zip"
RAREBENCH_SUBSETS = ["RAMEDIS", "MME", "HMS", "LIRICAL"]

CANDIDATE_TEXT_MAX_CHARS = 50000
PATIENT_SIMULATOR_MAX_SPANS = 5
PATIENT_SIMULATOR_MIN_OVERLAP = 1

# MEDDx-style hybrid layer:
# - add a weak, visible-evidence-only reference-case prior
# - encourage one broad first-turn inventory question
# - let the final resolver combine LLM rank with the casebase prior
ENABLE_CASEBASE_PRIOR = True
BROAD_INVENTORY_FIRST_TURN = True
CASEBASE_REFERENCE_MAX_CASES_PER_DATASET = 1500
CASEBASE_PRIOR_CONTEXT_LABELS = 6
CASEBASE_PRIOR_RERANK_LABELS = 10
CASEBASE_PRIOR_MIN_CANDIDATES = 10
CASEBASE_PRIOR_WEIGHT = 1.25
CASEBASE_PRIOR_MIN_NORMALIZED_SCORE = 0.05
CASEBASE_PRIOR_PROMOTION_MARGIN = 0.12
CASEBASE_PRIOR_MATCH_TERMS = 8

# RareBench remains an HPO phenotype-to-disease task. The graph layer treats exact phenotype names as
# atomic nodes and scores candidate diseases by leave-one-case-out exemplar overlap within the same RareBench subset.
ENABLE_RAREBENCH_GRAPH_PHENOTYPE_RESOLVER = True
RAREBENCH_GRAPH_CONTEXT_LABELS = 8
RAREBENCH_GRAPH_RERANK_LABELS = 12
RAREBENCH_GRAPH_MIN_VISIBLE_PHENOTYPES = 3
RAREBENCH_GRAPH_MIN_NORMALIZED_SCORE = 0.05
RAREBENCH_GRAPH_PRIOR_WEIGHT = 2.25
RAREBENCH_GRAPH_LLM_RANK_WEIGHT = 0.25
RAREBENCH_GRAPH_DESCRIPTIVE_LABEL_PENALTY = 0.20
RAREBENCH_GRAPH_MATCH_TERMS = 10
RAREBENCH_LLM_DISCRIMINATOR = True

# Notebook 45 architecture layer: cap-aware stopping and universal hypothesis branching.
ENABLE_CAP_AWARE_EARLY_STOP = True
ENABLE_DDXPLUS_PARTIAL_MLP_MONITOR = True
EARLY_STOP_MIN_QUESTIONS_BY_BUDGET = {5: 3, 10: 4, 15: 5}
EARLY_STOP_CONFIDENCE_MIN = 0.82
EARLY_STOP_RANK_MARGIN_MIN = 0.18
EARLY_STOP_STABILITY_MIN = 1
EARLY_STOP_DDXPLUS_MLP_CONFIDENCE_MIN = 0.70
EARLY_STOP_DDXPLUS_MLP_MARGIN_MIN = 0.20
EARLY_STOP_DDXPLUS_MLP_ENTROPY_MAX = 0.10

ENABLE_HYPOTHESIS_BRANCHING = True
BRANCH_MAX_BRANCHES = 2
BRANCH_MAX_QUESTIONS_PER_BRANCH = 2
BRANCH_MIN_REMAINING_BUDGET = 2
BRANCH_TRIGGER_CONFIDENCE_MAX = 0.84
BRANCH_TRIGGER_MARGIN_MAX = 0.22
BRANCH_TRIGGER_DISAGREEMENT_MIN = 0.34
BRANCH_CANDIDATE_POOL_LABELS = 8
RESOLVER_BASE_PROTECTION_MARGIN = 0.18
RESOLVER_MIN_INDEPENDENT_SUPPORT_TO_OVERRIDE = 2

# RareBench graph safety: the graph/discriminator may revise only when independent support is strong.
RAREBENCH_CONSERVATIVE_GRAPH_GATE = True
RAREBENCH_GRAPH_OVERRIDE_MARGIN_MIN = 0.18
RAREBENCH_DISCRIMINATOR_THIRD_OPTION_MARGIN_MIN = 0.28
RAREBENCH_LOCK_WHEN_LLM_AND_GRAPH_AGREE = True

# Generic adapter column defaults. Override these if the external files use different names.
GENERIC_CASE_ID_COLUMNS = ["case_id", "id", "patient_id", "uid"]
GENERIC_INITIAL_INFO_COLUMNS = ["initial_patient_info", "initial_info", "chief_complaint", "presentation", "initial"]
GENERIC_FULL_PROFILE_COLUMNS = ["full_patient_profile", "full_profile", "patient_profile", "profile", "case_text", "text", "description"]
GENERIC_DIAGNOSIS_COLUMNS = ["ground_truth_diagnosis", "diagnosis", "label", "disease", "pathology"]
GENERIC_CANDIDATE_COLUMNS = ["candidate_disease_list", "candidate_diseases", "disease_list", "possible_diseases"]

RUN_NAME_BASE = f"universal_branching_resolver_meddx_driver_{RUN_VERSION_SUFFIX}"
RUN_NAME = RUN_NAME_BASE if RUN_LIVE_API else f"universal_branching_resolver_meddx_driver_dryrun_smoke_{RUN_VERSION_SUFFIX}"

# %%
import ast
import csv
import json
import math
import random
import re
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from getpass import getpass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover - script fallback.
    def display(obj: Any) -> None:
        print(obj)

pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 80)

ROOT = next((candidate for candidate in [Path.cwd(), *Path.cwd().parents] if (candidate / "notebooks").exists() and (candidate / "reports").exists()), Path.cwd())
DEFAULT_DATASET_DIR = ROOT / "dataset"
DATASET_DIR = Path(DATASET_DIR_OVERRIDE).expanduser() if DATASET_DIR_OVERRIDE else DEFAULT_DATASET_DIR
DEFAULT_MEDDXAGENT_DATA_ROOT = ROOT / "external" / "meddxagent" / "ddxdriver" / "benchmarks" / "data"

def project_path(value: str, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else ROOT / path

MEDDXAGENT_DATA_ROOT = project_path(MEDDXAGENT_DATA_ROOT, DEFAULT_MEDDXAGENT_DATA_ROOT)
ICRAFT_MD_PATH = project_path(str(ICRAFT_MD_PATH), MEDDXAGENT_DATA_ROOT / "icraftmd" / "all_craft_md.jsonl")
RAREBENCH_MAPPING_DIR = project_path(str(RAREBENCH_MAPPING_DIR), MEDDXAGENT_DATA_ROOT / "rarebench")
RAREBENCH_DATA_ZIP_PATH = project_path(str(RAREBENCH_DATA_ZIP_PATH), ROOT / "artifacts" / "universal_meddx" / "cache" / "rarebench_data.zip")

ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
CACHE_DIR = ARTIFACT_ROOT / "cache"
for directory in [ARTIFACT_ROOT, FIGURE_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if RUN_LIVE_API and not LLM_API_KEY:
    if INTERACTIVE_API_KEY_BOOTSTRAP:
        print("RUN_LIVE_API=True and LLM_API_KEY is empty. Waiting for the secure API-key prompt now.")
        print("If your notebook UI does not show the prompt, stop the cell and either paste the key into LLM_API_KEY or set RUN_LIVE_API=False for dry-run smoke.")
        LLM_API_KEY = getpass("Enter LLM_API_KEY: ")
    else:
        raise ValueError("RUN_LIVE_API=True but LLM_API_KEY is empty. Set LLM_API_KEY or enable INTERACTIVE_API_KEY_BOOTSTRAP.")

print("Project root       :", ROOT)
print("Dataset dir        :", DATASET_DIR)
print("Run live API       :", RUN_LIVE_API)
print("Allow dry-run      :", ALLOW_DRY_RUN_BENCHMARK)
print("LLM model          :", LLM_MODEL)
print("Temperature/top_p  :", TEMPERATURE, TOP_P)
print("Run name           :", RUN_NAME)
print("Artifact root      :", ARTIFACT_ROOT)
print("Enabled datasets   :", ENABLED_DATASETS)
print("Require all        :", REQUIRE_ALL_ENABLED_DATASETS)
print("MEDDx ref budgets  :", MEDDX_REFERENCE_BUDGETS)
print("Live budgets       :", LIVE_BUDGETS_TO_RUN)
print("Early stopping     :", ENABLE_CAP_AWARE_EARLY_STOP)
print("Hypothesis branches:", ENABLE_HYPOTHESIS_BRANCHING, "max", BRANCH_MAX_BRANCHES)
print("MEDDxAgent data    :", MEDDXAGENT_DATA_ROOT)
print("iCraft-MD path     :", ICRAFT_MD_PATH)
print("RareBench mappings :", RAREBENCH_MAPPING_DIR)
print("RareBench zip      :", RAREBENCH_DATA_ZIP_PATH)

# %% [markdown]
# ## 1. Utility Functions And Universal Schema

# %%
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def safe_parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def normalize_list(value: Any) -> list[str]:
    parsed = safe_parse_jsonish(value, [])
    if isinstance(parsed, list):
        out = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and item:
                out.append(str(item[0]))
            elif isinstance(item, dict):
                label = item.get("name") or item.get("diagnosis") or item.get("disease") or item.get("label")
                if label:
                    out.append(str(label))
            elif item is not None:
                out.append(str(item))
        return [item for item in out if item]
    if isinstance(parsed, str) and parsed:
        return [part.strip() for part in re.split(r"[|;,]", parsed) if part.strip()]
    return []


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_actual = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def tokenize(text: str) -> set[str]:
    stop = {
        "the", "and", "with", "without", "that", "this", "have", "has", "had", "are", "was", "were",
        "from", "into", "about", "what", "when", "where", "which", "there", "their", "patient", "reported",
        "tell", "more", "any", "you", "your", "does", "did", "for", "can", "please", "other", "additional",
    }
    return {tok for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", text.lower()) if tok not in stop}


def split_profile_into_spans(profile: str) -> list[str]:
    spans: list[str] = []
    for raw_line in str(profile).replace("\r", "\n").split("\n"):
        line = raw_line.strip(" -\t")
        if not line:
            continue
        if "Answer:" in line:
            spans.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                spans.append(cleaned)
    return spans


def normalize_label_text(text: str) -> str:
    text = str(text).lower()
    roman_map = {
        " type iia": " type 2a",
        " type iib": " type 2b",
        " type iii": " type 3",
        " type ii": " type 2",
        " type iv": " type 4",
        " type i": " type 1",
    }
    for src, dst in roman_map.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def label_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, normalize_label_text(a), normalize_label_text(b)).ratio()


def canonicalize_to_candidates(label: str, candidates: list[str], threshold: float = 0.92) -> str:
    if not label or not candidates:
        return str(label)
    exact = {str(candidate).strip().lower(): str(candidate) for candidate in candidates}
    key = str(label).strip().lower()
    if key in exact:
        return exact[key]
    best = max(candidates, key=lambda candidate: label_similarity(label, candidate))
    return str(best) if label_similarity(label, best) >= threshold else str(label)


def canonicalize_ranked_differential(ranked: list[str], candidates: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in ranked:
        canonical = canonicalize_to_candidates(str(label), candidates)
        key = canonical.strip().lower()
        if key and key not in seen:
            out.append(canonical)
            seen.add(key)
    return out[:10]


def rank_true_label(ranked: list[str], true_label: str, missing_rank: int = 11) -> int:
    truth = normalize_label_text(true_label)
    for idx, label in enumerate(ranked[:10], start=1):
        if normalize_label_text(label) == truth:
            return idx
    return missing_rank


def topk_hit(ranked: list[str], true_label: str, k: int) -> bool:
    truth = normalize_label_text(true_label)
    return any(normalize_label_text(label) == truth for label in ranked[:k])


@dataclass
class UniversalCase:
    case_id: str
    dataset_name: str
    initial_patient_info: str
    hidden_full_profile: str
    ground_truth_diagnosis: str
    candidate_disease_list: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CasebaseReference:
    case_id: str
    dataset_name: str
    diagnosis: str
    text: str
    tokens: set[str]


@dataclass
class AdapterResult:
    dataset_name: str
    status: str
    cases: list[UniversalCase]
    message: str
    source_path: str = ""


@dataclass
class WorkupTurn:
    turn_index: int
    question: str
    answer: str
    retrieved_spans: list[str]
    decision_before_answer: str
    predicted_before_answer: str
    ranked_before_answer: list[str]


@dataclass
class StopProbe:
    turn_index: int
    predicted_diagnosis: str
    ranked_differential: list[str]
    confidence: float
    rank_margin: float
    stability_turns: int
    stop_signal: dict[str, Any]


@dataclass
class HypothesisBranchResult:
    branch_id: str
    target_hypothesis: str
    branch_role: str
    turns: list[WorkupTurn]
    predicted_diagnosis: str
    ranked_differential: list[str]
    confidence: float
    input_tokens: int
    output_tokens: int
    api_calls: int
    raw_responses: list[dict[str, Any]]


@dataclass
class WorkupResult:
    case_id: str
    dataset_name: str
    budget: int
    predicted_diagnosis: str
    ranked_differential: list[str]
    confidence: float
    num_questions: int
    stopped_early: bool
    stop_reason: str
    correct_top1: bool
    gtpa_at_3: bool
    gtpa_at_5: bool
    true_rank: int
    initial_true_rank: int
    progress_improved: bool
    input_tokens: int
    output_tokens: int
    api_calls: int
    estimated_cost: float | None
    llm_predicted_diagnosis: str
    llm_ranked_differential: list[str]
    casebase_prior_top_label: str
    casebase_prior_top_score: float
    casebase_prior_margin: float
    casebase_resolver_changed: bool
    rarebench_graph_top_label: str
    rarebench_graph_top_score: float
    rarebench_graph_margin: float
    rarebench_graph_visible_phenotypes: int
    rarebench_graph_resolver_changed: bool
    rarebench_discriminator_used: bool
    stop_probe_count: int
    early_stop_signal: str
    ddxplus_mlp_available: bool
    ddxplus_mlp_top1: str
    ddxplus_mlp_top5: list[str]
    ddxplus_mlp_confidence: float
    ddxplus_mlp_margin: float
    ddxplus_mlp_entropy: float
    branch_triggered: bool
    branch_count: int
    branch_question_count: int
    resolver_margin: float
    resolver_support_count: int
    resolver_changed_top1: bool
    rarebench_gate_action: str
    turns: list[WorkupTurn]
    branches: list[HypothesisBranchResult]
    stop_probes: list[StopProbe]
    candidate_scores: list[dict[str, Any]]
    raw_responses: list[dict[str, Any]]

# %% [markdown]
# ## 2. Dataset Adapters

# %%
def load_ddxplus_split(dataset_dir: Path, split: str = "test") -> pd.DataFrame:
    zip_path = dataset_dir / f"release_{split}_patients.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing DDXPlus split archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            raise ValueError(f"Empty DDXPlus archive: {zip_path}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle)
    frame = frame.copy()
    frame["case_id"] = [f"{split}:{idx}" for idx in frame.index]
    return frame


def parse_ddxplus_token(token: str) -> tuple[str, str | None]:
    token = str(token)
    if "_@_" not in token:
        return token, None
    root, value = token.split("_@_", 1)
    return root, value


def decode_ddxplus_evidence_token(token: str, evidence_map: dict[str, Any]) -> dict[str, str]:
    root, value = parse_ddxplus_token(token)
    item = evidence_map.get(root, {})
    question = item.get("question_en") or item.get("name") or root
    data_type = item.get("data_type", "")
    if value is None:
        answer = "yes"
    elif data_type in {"M", "C"}:
        answer = item.get("value_meaning", {}).get(value, {}).get("en", value)
    elif data_type == "B":
        answer = "yes" if str(value).lower() not in {"0", "false", "no"} else "no"
    else:
        answer = value
    return {"root": root, "question": str(question), "answer": str(answer), "data_type": str(data_type)}


def ddxplus_case_to_profile(row: pd.Series, evidence_map: dict[str, Any]) -> tuple[str, str, list[str]]:
    evidence_tokens = normalize_list(row.get("EVIDENCES", "[]"))
    initial_token = str(row.get("INITIAL_EVIDENCE", ""))
    decoded_initial = decode_ddxplus_evidence_token(initial_token, evidence_map) if initial_token else {"question": "Initial evidence", "answer": "not provided"}
    initial_info = "\n".join([
        f"Age: {row.get('AGE')}",
        f"Sex: {row.get('SEX')}",
        f"Initial finding: {decoded_initial['question']} Answer: {decoded_initial['answer']}.",
    ])
    profile_lines = [
        f"Age: {row.get('AGE')}.",
        f"Sex: {row.get('SEX')}.",
        "Known patient findings from the hidden profile:",
    ]
    seen: set[tuple[str, str]] = set()
    for token in evidence_tokens:
        decoded = decode_ddxplus_evidence_token(token, evidence_map)
        key = (decoded["root"], decoded["answer"])
        if key in seen:
            continue
        seen.add(key)
        profile_lines.append(f"- {decoded['question']} Answer: {decoded['answer']}.")
    profile = "\n".join(profile_lines)
    official_diff = normalize_list(row.get("DIFFERENTIAL_DIAGNOSIS", "[]"))
    return initial_info, profile, official_diff


def ddxplus_span_token_map(row: pd.Series, evidence_map: dict[str, Any]) -> dict[str, str]:
    span_map: dict[str, str] = {}
    for token in normalize_list(row.get("EVIDENCES", "[]")):
        decoded = decode_ddxplus_evidence_token(token, evidence_map)
        span = f"{decoded['question']} Answer: {decoded['answer']}."
        span_map[span] = str(token)
    return span_map


def sample_ddxplus_cases(frame: pd.DataFrame, max_cases: int, seed: int) -> pd.DataFrame:
    if max_cases <= 0 or len(frame) <= max_cases:
        return frame.copy()
    pathology_counts = frame["PATHOLOGY"].value_counts()
    if max_cases >= min(len(pathology_counts) * 2, len(frame)):
        rows = []
        per_pathology = max(1, max_cases // max(len(pathology_counts), 1))
        for pathology, group in frame.groupby("PATHOLOGY", sort=True):
            rows.append(group.sample(n=min(per_pathology, len(group)), random_state=seed))
        sampled = pd.concat(rows, ignore_index=False)
        if len(sampled) < max_cases:
            remaining = frame[~frame["case_id"].isin(sampled["case_id"])]
            extra = remaining.sample(n=min(max_cases - len(sampled), len(remaining)), random_state=seed + 1)
            sampled = pd.concat([sampled, extra], ignore_index=False)
        return sampled.sample(frac=1.0, random_state=seed).head(max_cases).copy()
    return frame.sample(n=max_cases, random_state=seed).copy()


def load_ddxplus_adapter(max_cases: int) -> AdapterResult:
    try:
        evidence_map = read_json(DATASET_DIR / "release_evidences.json")
        condition_map = read_json(DATASET_DIR / "release_conditions.json")
        frame = load_ddxplus_split(DATASET_DIR, split="test")
        sampled = sample_ddxplus_cases(frame, max_cases=max_cases, seed=RANDOM_SEED)
        all_conditions = sorted(str(name) for name in condition_map.keys())
        cases: list[UniversalCase] = []
        for _, row in sampled.iterrows():
            initial_info, profile, official_diff = ddxplus_case_to_profile(row, evidence_map)
            evidence_tokens = normalize_list(row.get("EVIDENCES", "[]"))
            candidate_list = all_conditions
            metadata = {
                "age": row.get("AGE"),
                "sex": row.get("SEX"),
                "official_differential": official_diff[:10],
                "initial_evidence": row.get("INITIAL_EVIDENCE"),
                "evidence_tokens": evidence_tokens,
                "span_token_map": ddxplus_span_token_map(row, evidence_map),
                "source": "ddxplus_structured_to_profile_v1",
            }
            cases.append(
                UniversalCase(
                    case_id=str(row["case_id"]),
                    dataset_name="ddxplus",
                    initial_patient_info=initial_info,
                    hidden_full_profile=profile,
                    ground_truth_diagnosis=str(row["PATHOLOGY"]),
                    candidate_disease_list=candidate_list,
                    metadata=metadata,
                )
            )
        return AdapterResult("ddxplus", "loaded", cases, f"Loaded {len(cases)} DDXPlus cases from test split.", str(DATASET_DIR))
    except Exception as exc:
        return AdapterResult("ddxplus", "error", [], str(exc), str(DATASET_DIR))


def read_generic_cases(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = payload.get("cases") or payload.get("data") or payload.get("records") or []
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported generic dataset extension for {path}. Use CSV, JSONL, NDJSON, or JSON.")


def load_generic_profile_adapter(dataset_name: str, source_path: str, max_cases: int) -> AdapterResult:
    if not source_path:
        return AdapterResult(dataset_name, "skipped_missing_path", [], "No local dataset path was supplied.", "")
    path = Path(source_path).expanduser()
    if not path.exists():
        return AdapterResult(dataset_name, "skipped_missing_path", [], f"Path does not exist: {path}", str(path))
    try:
        frame = read_generic_cases(path)
        if frame.empty:
            return AdapterResult(dataset_name, "error", [], f"No rows found in {path}.", str(path))
        case_col = first_existing_column(list(frame.columns), GENERIC_CASE_ID_COLUMNS)
        initial_col = first_existing_column(list(frame.columns), GENERIC_INITIAL_INFO_COLUMNS)
        profile_col = first_existing_column(list(frame.columns), GENERIC_FULL_PROFILE_COLUMNS)
        diagnosis_col = first_existing_column(list(frame.columns), GENERIC_DIAGNOSIS_COLUMNS)
        candidate_col = first_existing_column(list(frame.columns), GENERIC_CANDIDATE_COLUMNS)
        if profile_col is None or diagnosis_col is None:
            return AdapterResult(
                dataset_name,
                "error",
                [],
                f"Could not find required profile/diagnosis columns. Columns={list(frame.columns)}",
                str(path),
            )
        if len(frame) > max_cases:
            frame = frame.sample(n=max_cases, random_state=RANDOM_SEED).copy()
        dataset_candidates = sorted(str(x) for x in frame[diagnosis_col].dropna().unique())
        cases: list[UniversalCase] = []
        for idx, row in frame.reset_index(drop=True).iterrows():
            case_id = str(row[case_col]) if case_col else f"{dataset_name}:{idx}"
            profile = str(row[profile_col])
            initial = str(row[initial_col]) if initial_col else profile[:600]
            candidates = normalize_list(row[candidate_col]) if candidate_col else dataset_candidates
            if not candidates:
                candidates = dataset_candidates
            cases.append(
                UniversalCase(
                    case_id=case_id,
                    dataset_name=dataset_name,
                    initial_patient_info=initial,
                    hidden_full_profile=profile,
                    ground_truth_diagnosis=str(row[diagnosis_col]),
                    candidate_disease_list=candidates,
                    metadata={
                        "source_path": str(path),
                        "profile_column": profile_col,
                        "diagnosis_column": diagnosis_col,
                        "candidate_column": candidate_col or "",
                    },
                )
            )
        return AdapterResult(dataset_name, "loaded", cases, f"Loaded {len(cases)} cases from {path}.", str(path))
    except Exception as exc:
        return AdapterResult(dataset_name, "error", [], str(exc), str(path))


def read_diagnosis_options_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_icraft_md_adapter(max_cases: int) -> AdapterResult:
    path = Path(ICRAFT_MD_PATH)
    if not path.exists():
        return AdapterResult(
            "icraft_md",
            "missing_required_path",
            [],
            f"iCraft-MD file not found: {path}. Clone MEDDxAgent into external/meddxagent or edit ICRAFT_MD_PATH.",
            str(path),
        )
    try:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return AdapterResult("icraft_md", "error", [], f"No rows found in {path}.", str(path))
        frame = pd.DataFrame(records)
        if len(frame) > max_cases:
            frame = frame.sample(n=max_cases, random_state=RANDOM_SEED).copy()
        data_dir = path.parent
        disease_mapping = read_json(data_dir / "disease_mapping.json") if (data_dir / "disease_mapping.json").exists() else {}
        diagnosis_options = read_diagnosis_options_txt(data_dir / "diagnosis_options.txt")
        if not diagnosis_options:
            diagnosis_options = sorted(str(item) for item in pd.DataFrame(records)["answer"].dropna().unique())
        cases: list[UniversalCase] = []
        for _, row in frame.reset_index(drop=True).iterrows():
            patient = row.get("patient") if isinstance(row.get("patient"), dict) else {}
            age = patient.get("age", "unknown")
            gender = patient.get("gender", "unknown")
            context = row.get("context") if isinstance(row.get("context"), list) else normalize_list(row.get("context", []))
            facts = row.get("facts") if isinstance(row.get("facts"), list) else []
            profile_lines = [str(item) for item in context]
            if facts:
                profile_lines.extend(str(item) for item in facts)
            option_values = []
            options = row.get("options")
            if isinstance(options, dict):
                option_values = [str(value) for _, value in sorted(options.items())]
            answer_idx = str(row.get("answer_idx", ""))
            answer = str(row.get("answer", ""))
            ground_truth = str(options.get(answer_idx, answer)) if isinstance(options, dict) and answer_idx in options else str(disease_mapping.get(answer, answer))
            candidate_list = option_values if option_values else diagnosis_options
            cases.append(
                UniversalCase(
                    case_id=f"icraft_md:{row.get('id')}",
                    dataset_name="icraft_md",
                    initial_patient_info="\n".join([
                        f"Age: {age}",
                        f"Sex: {gender}",
                        f"Chief complaint: {context[0] if context else 'not provided'}",
                    ]),
                    hidden_full_profile="\n".join(f"- {item}" for item in profile_lines if str(item).strip()),
                    ground_truth_diagnosis=ground_truth,
                    candidate_disease_list=candidate_list,
                    metadata={
                        "source_path": str(path),
                        "source": "meddxagent_icraftmd_jsonl_v1",
                        "answer_idx": answer_idx,
                        "all_diagnosis_options_count": len(diagnosis_options),
                        "case_option_count": len(option_values),
                    },
                )
            )
        return AdapterResult("icraft_md", "loaded", cases, f"Loaded {len(cases)} iCraft-MD cases from MEDDxAgent JSONL.", str(path))
    except Exception as exc:
        return AdapterResult("icraft_md", "error", [], str(exc), str(path))


def ensure_rarebench_data_zip() -> Path:
    path = Path(RAREBENCH_DATA_ZIP_PATH)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(RAREBENCH_DATA_ZIP_URL, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_rarebench_adapter(max_cases: int) -> AdapterResult:
    mapping_dir = Path(RAREBENCH_MAPPING_DIR)
    required = [
        mapping_dir / "rarebench_phenotype_mapping.json",
        mapping_dir / "rarebench_disease_mapping.json",
        mapping_dir / "disease_mapping.json",
        mapping_dir / "diagnosis_options.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return AdapterResult(
            "rarebench",
            "missing_required_path",
            [],
            "RareBench mapping files not found. Clone MEDDxAgent into external/meddxagent or edit RAREBENCH_MAPPING_DIR. Missing: " + "; ".join(missing),
            str(mapping_dir),
        )
    try:
        phenotype_mapping = read_json(mapping_dir / "rarebench_phenotype_mapping.json")
        code_to_disease = read_json(mapping_dir / "rarebench_disease_mapping.json")
        disease_mapping = read_json(mapping_dir / "disease_mapping.json")
        diagnosis_options_by_subset = read_json(mapping_dir / "diagnosis_options.json")
        data_zip = ensure_rarebench_data_zip()
        records: list[dict[str, Any]] = []
        with zipfile.ZipFile(data_zip) as archive:
            available = set(archive.namelist())
            for subset in RAREBENCH_SUBSETS:
                member = f"data/{subset}.jsonl"
                if member not in available:
                    continue
                with archive.open(member) as handle:
                    for idx, raw in enumerate(handle):
                        row = json.loads(raw.decode("utf-8"))
                        records.append({"subset": subset, "subset_index": idx, **row})
        if not records:
            return AdapterResult("rarebench", "error", [], f"No RareBench rows found in {data_zip} for subsets {RAREBENCH_SUBSETS}.", str(data_zip))
        frame = pd.DataFrame(records)
        if len(frame) > max_cases:
            frame = frame.sample(n=max_cases, random_state=RANDOM_SEED).copy()
        cases: list[UniversalCase] = []
        for _, row in frame.reset_index(drop=True).iterrows():
            subset = str(row["subset"])
            phenotype_codes = row.get("Phenotype") if isinstance(row.get("Phenotype"), list) else []
            disease_codes = row.get("RareDisease") if isinstance(row.get("RareDisease"), list) else []
            phenotypes = [str(phenotype_mapping[code]) for code in phenotype_codes if code in phenotype_mapping]
            disease_names = [str(code_to_disease[code]) for code in disease_codes if code in code_to_disease]
            pathology_key = ", ".join(disease_names)
            subset_mapping = disease_mapping.get(subset, {})
            ground_truth = str(subset_mapping.get(pathology_key, disease_names[0] if disease_names else pathology_key))
            num_initial = max(1, round(len(phenotypes) * 0.2)) if phenotypes else 0
            initial_phenotypes = phenotypes[:num_initial]
            candidate_list = [str(option) for option in diagnosis_options_by_subset.get(subset, [])]
            ground_truth = canonicalize_to_candidates(ground_truth, candidate_list, threshold=0.90)
            cases.append(
                UniversalCase(
                    case_id=f"rarebench:{subset}:{row.get('subset_index')}",
                    dataset_name="rarebench",
                    initial_patient_info="\n".join(f"- {item}" for item in initial_phenotypes) or "Initial symptoms not provided.",
                    hidden_full_profile="\n".join(f"- {item}" for item in phenotypes) or "No phenotype profile available.",
                    ground_truth_diagnosis=ground_truth,
                    candidate_disease_list=candidate_list,
                    metadata={
                        "source_path": str(data_zip),
                        "mapping_dir": str(mapping_dir),
                        "source": "rarebench_hf_zip_with_meddxagent_mappings_v1",
                        "subset": subset,
                        "subset_candidate_count": len(candidate_list),
                        "raw_disease_codes": disease_codes,
                    },
                )
            )
        return AdapterResult("rarebench", "loaded", cases, f"Loaded {len(cases)} RareBench cases from HuggingFace zip and MEDDxAgent mappings.", str(data_zip))
    except Exception as exc:
        return AdapterResult("rarebench", "error", [], str(exc), str(mapping_dir))


adapter_load_cap = LIVE_TOTAL_MAX_CASES if RUN_LIVE_API else DRY_RUN_TOTAL_MAX_CASES
adapter_results: list[AdapterResult] = []
if "ddxplus" in ENABLED_DATASETS:
    adapter_results.append(load_ddxplus_adapter(adapter_load_cap))
if "icraft_md" in ENABLED_DATASETS:
    adapter_results.append(load_icraft_md_adapter(adapter_load_cap))
if "rarebench" in ENABLED_DATASETS:
    adapter_results.append(load_rarebench_adapter(adapter_load_cap))

adapter_preflight = pd.DataFrame([
    {
        "dataset_name": result.dataset_name,
        "status": result.status,
        "num_cases": len(result.cases),
        "source_path": result.source_path,
        "message": result.message,
    }
    for result in adapter_results
])
adapter_preflight.to_csv(ARTIFACT_ROOT / "adapter_preflight.csv", index=False)
display(adapter_preflight)

if REQUIRE_ALL_ENABLED_DATASETS:
    loaded_datasets = {result.dataset_name for result in adapter_results if result.status == "loaded" and result.cases}
    missing_enabled = [dataset for dataset in ENABLED_DATASETS if dataset not in loaded_datasets]
    if missing_enabled:
        detail = adapter_preflight[adapter_preflight["dataset_name"].isin(missing_enabled)].to_dict(orient="records")
        raise RuntimeError(
            "Notebook 45 is configured to test all enabled datasets, but not all adapters loaded. "
            "Resolve the adapter_preflight errors or set REQUIRE_ALL_ENABLED_DATASETS=False for a partial smoke run. "
            f"Missing enabled datasets: {missing_enabled}. Details: {detail}"
        )

def cap_universal_cases_by_dataset(results: list[AdapterResult], total_cap: int, seed: int) -> list[UniversalCase]:
    loaded = [result for result in results if result.status == "loaded" and result.cases]
    if total_cap <= 0 or not loaded:
        return []
    if sum(len(result.cases) for result in loaded) <= total_cap:
        return [case for result in loaded for case in result.cases]
    rng_local = random.Random(seed)
    base = total_cap // len(loaded)
    remainder = total_cap % len(loaded)
    selected: list[UniversalCase] = []
    for idx, result in enumerate(sorted(loaded, key=lambda item: item.dataset_name)):
        quota = base + (1 if idx < remainder else 0)
        cases = list(result.cases)
        rng_local.shuffle(cases)
        selected.extend(cases[: min(quota, len(cases))])
    if len(selected) < total_cap:
        selected_ids = {(case.dataset_name, case.case_id) for case in selected}
        leftovers = [
            case
            for result in sorted(loaded, key=lambda item: item.dataset_name)
            for case in result.cases
            if (case.dataset_name, case.case_id) not in selected_ids
        ]
        rng_local.shuffle(leftovers)
        selected.extend(leftovers[: total_cap - len(selected)])
    selected.sort(key=lambda case: (case.dataset_name, case.case_id))
    return selected[:total_cap]


effective_total_case_cap = LIVE_TOTAL_MAX_CASES if RUN_LIVE_API else DRY_RUN_TOTAL_MAX_CASES
universal_cases = cap_universal_cases_by_dataset(adapter_results, effective_total_case_cap, RANDOM_SEED)
case_frame = pd.DataFrame([
    {
        "case_id": case.case_id,
        "dataset_name": case.dataset_name,
        "initial_patient_info": case.initial_patient_info,
        "hidden_full_profile": case.hidden_full_profile,
        "ground_truth_diagnosis": case.ground_truth_diagnosis,
        "candidate_disease_list": json.dumps(case.candidate_disease_list),
        "metadata": json.dumps(case.metadata, ensure_ascii=True),
    }
    for case in universal_cases
])
case_frame.to_csv(ARTIFACT_ROOT / "universal_cases.csv", index=False)
print("Universal cases selected for run:", len(universal_cases), "of global cap", effective_total_case_cap)

# %% [markdown]
# ## 3. Reference Casebase Prior

# %%
def make_casebase_reference(case_id: str, dataset_name: str, diagnosis: str, text: str) -> CasebaseReference | None:
    diagnosis = str(diagnosis).strip()
    text = str(text).strip()
    if not diagnosis or not text:
        return None
    tokens = tokenize(text)
    if not tokens:
        return None
    return CasebaseReference(
        case_id=str(case_id),
        dataset_name=str(dataset_name),
        diagnosis=diagnosis,
        text=text,
        tokens=tokens,
    )


def build_ddxplus_casebase(max_cases: int) -> list[CasebaseReference]:
    try:
        evidence_map = read_json(DATASET_DIR / "release_evidences.json")
        frame = load_ddxplus_split(DATASET_DIR, split="train")
        sampled = sample_ddxplus_cases(frame, max_cases=max_cases, seed=RANDOM_SEED + 101)
        refs: list[CasebaseReference] = []
        for _, row in sampled.iterrows():
            _, profile, _ = ddxplus_case_to_profile(row, evidence_map)
            ref = make_casebase_reference(str(row["case_id"]), "ddxplus", str(row["PATHOLOGY"]), profile)
            if ref:
                refs.append(ref)
        return refs
    except Exception as exc:
        print("DDXPlus casebase prior skipped:", exc)
        return []


def build_icraft_casebase(max_cases: int) -> list[CasebaseReference]:
    path = Path(ICRAFT_MD_PATH)
    if not path.exists():
        return []
    try:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if max_cases > 0 and len(records) > max_cases:
            rng = random.Random(RANDOM_SEED + 202)
            records = rng.sample(records, max_cases)
        disease_mapping = read_json(path.parent / "disease_mapping.json") if (path.parent / "disease_mapping.json").exists() else {}
        refs: list[CasebaseReference] = []
        for row in records:
            context = row.get("context") if isinstance(row.get("context"), list) else normalize_list(row.get("context", []))
            facts = row.get("facts") if isinstance(row.get("facts"), list) else []
            profile = "\n".join(str(item) for item in [*context, *facts] if str(item).strip())
            options = row.get("options")
            answer_idx = str(row.get("answer_idx", ""))
            answer = str(row.get("answer", ""))
            diagnosis = str(options.get(answer_idx, answer)) if isinstance(options, dict) and answer_idx in options else str(disease_mapping.get(answer, answer))
            ref = make_casebase_reference(f"icraft_md:{row.get('id')}", "icraft_md", diagnosis, profile)
            if ref:
                refs.append(ref)
        return refs
    except Exception as exc:
        print("iCraft-MD casebase prior skipped:", exc)
        return []


def build_rarebench_casebase(max_cases: int) -> list[CasebaseReference]:
    mapping_dir = Path(RAREBENCH_MAPPING_DIR)
    required = [
        mapping_dir / "rarebench_phenotype_mapping.json",
        mapping_dir / "rarebench_disease_mapping.json",
        mapping_dir / "disease_mapping.json",
        mapping_dir / "diagnosis_options.json",
    ]
    if any(not path.exists() for path in required):
        return []
    try:
        phenotype_mapping = read_json(mapping_dir / "rarebench_phenotype_mapping.json")
        code_to_disease = read_json(mapping_dir / "rarebench_disease_mapping.json")
        disease_mapping = read_json(mapping_dir / "disease_mapping.json")
        data_zip = ensure_rarebench_data_zip()
        refs: list[CasebaseReference] = []
        with zipfile.ZipFile(data_zip) as archive:
            available = set(archive.namelist())
            for subset in RAREBENCH_SUBSETS:
                member = f"data/{subset}.jsonl"
                if member not in available:
                    continue
                subset_mapping = disease_mapping.get(subset, {})
                with archive.open(member) as handle:
                    for idx, raw in enumerate(handle):
                        row = json.loads(raw.decode("utf-8"))
                        phenotype_codes = row.get("Phenotype") if isinstance(row.get("Phenotype"), list) else []
                        disease_codes = row.get("RareDisease") if isinstance(row.get("RareDisease"), list) else []
                        phenotypes = [str(phenotype_mapping[code]) for code in phenotype_codes if code in phenotype_mapping]
                        disease_names = [str(code_to_disease[code]) for code in disease_codes if code in code_to_disease]
                        pathology_key = ", ".join(disease_names)
                        diagnosis = str(subset_mapping.get(pathology_key, disease_names[0] if disease_names else pathology_key))
                        ref = make_casebase_reference(
                            f"rarebench:{subset}:{idx}",
                            "rarebench",
                            diagnosis,
                            "\n".join(phenotypes),
                        )
                        if ref:
                            refs.append(ref)
        if max_cases > 0 and len(refs) > max_cases:
            rng = random.Random(RANDOM_SEED + 303)
            refs = rng.sample(refs, max_cases)
        return refs
    except Exception as exc:
        print("RareBench casebase prior skipped:", exc)
        return []


def build_reference_casebase() -> dict[str, list[CasebaseReference]]:
    if not ENABLE_CASEBASE_PRIOR:
        return {}
    reference_by_dataset = {
        "ddxplus": build_ddxplus_casebase(CASEBASE_REFERENCE_MAX_CASES_PER_DATASET),
        "icraft_md": build_icraft_casebase(CASEBASE_REFERENCE_MAX_CASES_PER_DATASET),
        "rarebench": build_rarebench_casebase(CASEBASE_REFERENCE_MAX_CASES_PER_DATASET),
    }
    summary = pd.DataFrame([
        {"dataset_name": dataset_name, "reference_cases": len(refs)}
        for dataset_name, refs in reference_by_dataset.items()
    ])
    summary.to_csv(ARTIFACT_ROOT / "casebase_prior_reference_summary.csv", index=False)
    display(summary)
    return reference_by_dataset


CASEBASE_REFERENCES = build_reference_casebase()


# %% [markdown]
# ## 4. DDXPlus Partial-Evidence MLP Monitor
#
# The original DDXPlus architecture used a partial-evidence MLP as the online stop signal. That model is only valid when
# we can reconstruct structured DDXPlus evidence roots. For iCraft-MD and RareBench we use the universal confidence
# monitor instead of pretending the DDXPlus MLP transfers across schemas.

# %%
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
        slot_slices: dict[str, tuple[int, int]] = {}
        data_types: dict[str, str] = {}
        possible_values: dict[str, list[str]] = {}
        default_values: dict[str, str | None] = {}
        categorical_integer_roots: set[str] = set()
        question_text: dict[str, str] = {}
        feature_names = [f"age_bin_{idx}" for idx in range(8)] + ["sex_M", "sex_F"]
        cursor = 10
        for root_id in root_ids:
            meta = evidence_metadata[root_id]
            data_type = meta.get("data_type", "B")
            raw_values = meta.get("possible-values", [])
            values = [str(value) for value in raw_values]
            default_value = meta.get("default_value")
            default_values[root_id] = None if default_value is None else str(default_value)
            question_text[root_id] = meta.get("question_en", root_id)
            data_types[root_id] = data_type
            possible_values[root_id] = values
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

    def apply_root_observation(self, state: np.ndarray, root_id: str, values: list[str] | None = None) -> np.ndarray:
        if root_id not in self.slot_slices:
            return state
        values = [str(value) for value in (values or [])]
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
    sex = str(sex).upper()
    if sex == "M":
        return 0
    if sex == "F":
        return 1
    return 0


if nn is not None:
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

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.backbone(features))
else:
    DirectDiagnosisMLP = None


def discover_selected_partial_model_dir() -> Path | None:
    selected = ROOT / "artifacts" / "one_shot_partial_evidence" / "selected_model.json"
    if not selected.exists():
        return None
    payload = read_json(selected)
    model_dir = Path(payload.get("selected_artifact_dir", ""))
    if not (model_dir / "best_model.pt").exists():
        local_candidate = ROOT / "artifacts" / "one_shot_partial_evidence" / model_dir.name
        if (local_candidate / "best_model.pt").exists():
            model_dir = local_candidate
    return model_dir if (model_dir / "best_model.pt").exists() else None


def load_ddxplus_partial_mlp_monitor() -> dict[str, Any]:
    if not ENABLE_DDXPLUS_PARTIAL_MLP_MONITOR:
        return {"available": False, "reason": "disabled"}
    if torch is None or nn is None or DirectDiagnosisMLP is None:
        return {"available": False, "reason": "torch_not_available"}
    try:
        model_dir = discover_selected_partial_model_dir()
        if model_dir is None:
            return {"available": False, "reason": "selected_model_missing"}
        evidence_metadata = read_json(DATASET_DIR / "release_evidences.json")
        schema = ObservationSchema.from_metadata(evidence_metadata)
        checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")
        resolved = checkpoint.get("resolved_run_config", {})
        labels = list(checkpoint.get("label_names", []))
        hidden_sizes = list(resolved.get("hidden_sizes", [2048, 2048, 2048]))
        dropout = float(resolved.get("dropout", 0.0))
        model = DirectDiagnosisMLP(schema.feature_size, hidden_sizes, len(labels), dropout=dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "available": True,
            "model_dir": str(model_dir),
            "schema": schema,
            "model": model,
            "labels": labels,
            "resolved": resolved,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


DDXPLUS_MLP_MONITOR = load_ddxplus_partial_mlp_monitor()
print("DDXPlus MLP monitor:", "available" if DDXPLUS_MLP_MONITOR.get("available") else DDXPLUS_MLP_MONITOR.get("reason"))


def ddxplus_observed_tokens(case: UniversalCase, turns: list[WorkupTurn]) -> list[str]:
    if case.dataset_name != "ddxplus":
        return []
    tokens = []
    initial = str(case.metadata.get("initial_evidence", "") or "")
    if initial:
        tokens.append(initial)
    span_map = case.metadata.get("span_token_map", {})
    if isinstance(span_map, str):
        span_map = safe_parse_jsonish(span_map, {})
    if not isinstance(span_map, dict):
        span_map = {}
    for turn in turns:
        for span in turn.retrieved_spans:
            token = span_map.get(str(span).strip())
            if token:
                tokens.append(str(token))
    deduped = []
    seen = set()
    for token in tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def compute_ddxplus_mlp_feedback(case: UniversalCase, turns: list[WorkupTurn], mlp_history: list[dict[str, Any]]) -> dict[str, Any]:
    if case.dataset_name != "ddxplus" or not DDXPLUS_MLP_MONITOR.get("available"):
        return {"available": False}
    schema: ObservationSchema = DDXPLUS_MLP_MONITOR["schema"]
    model = DDXPLUS_MLP_MONITOR["model"]
    labels = DDXPLUS_MLP_MONITOR["labels"]
    state = schema.initial_state(int(case.metadata.get("age", 0) or 0), str(case.metadata.get("sex", "M")))
    root_values: dict[str, list[str]] = {}
    for token in ddxplus_observed_tokens(case, turns):
        root_id, value = parse_ddxplus_token(str(token))
        root_values.setdefault(root_id, [])
        root_values[root_id].append("present" if value is None else str(value))
    for root_id, values in root_values.items():
        schema.apply_root_observation(state, root_id, values)
    with torch.no_grad():
        logits = model(torch.tensor(state[None, :], dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    top_indices = order[:5]
    top_predictions = [labels[int(idx)] for idx in top_indices]
    top_probs = [float(probs[int(idx)]) for idx in top_indices]
    confidence = top_probs[0] if top_probs else 0.0
    second = top_probs[1] if len(top_probs) > 1 else 0.0
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))) if len(probs) else 1.0
    stability_turns = 0
    for prev in reversed([row.get("top1") for row in mlp_history if row.get("available")]):
        if prev == (top_predictions[0] if top_predictions else ""):
            stability_turns += 1
        else:
            break
    return {
        "available": True,
        "top1": top_predictions[0] if top_predictions else "",
        "top_predictions": top_predictions,
        "top_probabilities": top_probs,
        "confidence": float(confidence),
        "margin": float(confidence - second),
        "entropy": float(entropy),
        "stability_turns": int(stability_turns),
        "revealed_root_count": int(len(root_values)),
    }


def visible_case_text(case: UniversalCase, turns: list[WorkupTurn]) -> str:
    lines = [case.initial_patient_info]
    for turn in turns:
        lines.append(turn.question)
        lines.append(turn.answer)
    return "\n".join(str(line) for line in lines if str(line).strip())


def canonical_candidate_label(label: str, candidates: list[str]) -> str:
    if not candidates:
        return str(label)
    canonical = canonicalize_to_candidates(str(label), candidates, threshold=0.90)
    return canonical if canonical in candidates else ""


def casebase_prior_for_case(case: UniversalCase, turns: list[WorkupTurn]) -> list[dict[str, Any]]:
    if not ENABLE_CASEBASE_PRIOR:
        return []
    if len(case.candidate_disease_list) < CASEBASE_PRIOR_MIN_CANDIDATES:
        return []
    refs = CASEBASE_REFERENCES.get(case.dataset_name, [])
    if not refs:
        return []
    query_tokens = tokenize(visible_case_text(case, turns))
    if not query_tokens:
        return []
    label_stats: dict[str, dict[str, Any]] = {}
    for ref in refs:
        if ref.case_id == case.case_id:
            continue
        label = canonical_candidate_label(ref.diagnosis, case.candidate_disease_list)
        if not label:
            continue
        overlap = query_tokens & ref.tokens
        if not overlap:
            continue
        jaccard = len(overlap) / max(len(query_tokens | ref.tokens), 1)
        if jaccard <= 0:
            continue
        stats = label_stats.setdefault(label, {"label": label, "total": 0.0, "best": 0.0, "count": 0, "terms": Counter()})
        stats["total"] += jaccard
        stats["best"] = max(stats["best"], jaccard)
        stats["count"] += 1
        stats["terms"].update(overlap)
    priors = []
    for stats in label_stats.values():
        score = float(stats["best"] + 0.25 * math.log1p(stats["total"]) + 0.05 * math.log1p(stats["count"]))
        priors.append({
            "label": stats["label"],
            "score": score,
            "best_reference_score": float(stats["best"]),
            "supporting_reference_count": int(stats["count"]),
            "matched_terms": [term for term, _ in stats["terms"].most_common(CASEBASE_PRIOR_MATCH_TERMS)],
        })
    priors.sort(key=lambda row: (row["score"], row["supporting_reference_count"], row["label"]), reverse=True)
    if priors:
        top_score = max(priors[0]["score"], 1e-9)
        for row in priors:
            row["normalized_score"] = float(row["score"] / top_score)
    return priors[:CASEBASE_PRIOR_RERANK_LABELS]


def dynamic_fewshot_examples(case: UniversalCase, turns: list[WorkupTurn], k: int = 5) -> list[dict[str, Any]]:
    refs = CASEBASE_REFERENCES.get(case.dataset_name, [])
    if not refs or k <= 0:
        return []
    query_tokens = tokenize(visible_case_text(case, turns))
    if not query_tokens:
        return []
    scored = []
    seen_labels: set[str] = set()
    for ref in refs:
        if ref.case_id == case.case_id:
            continue
        label = canonical_candidate_label(ref.diagnosis, case.candidate_disease_list)
        if not label:
            continue
        overlap = query_tokens & ref.tokens
        if not overlap:
            continue
        score = len(overlap) / max(len(query_tokens | ref.tokens), 1)
        if score <= 0:
            continue
        scored.append((score, len(overlap), label, ref))
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    examples = []
    for score, overlap_count, label, ref in scored:
        # Keep examples diagnostically diverse; MEDDxAgent uses similar-case examples,
        # but too many same-label examples can over-anchor the final judge.
        if label in seen_labels and len(seen_labels) >= max(1, k // 2):
            continue
        seen_labels.add(label)
        profile = ref.text.strip()
        if len(profile) > 1200:
            profile = profile[:1200].rsplit("\n", 1)[0] + "\n[truncated]"
        examples.append({
            "case_id": ref.case_id,
            "diagnosis": label,
            "similarity": float(score),
            "overlap_terms": int(overlap_count),
            "patient_profile": profile,
        })
        if len(examples) >= k:
            break
    return examples


def dynamic_fewshot_text(case: UniversalCase, turns: list[WorkupTurn], k: int = 5) -> str:
    examples = dynamic_fewshot_examples(case, turns, k=k)
    if not examples:
        return "No dynamic reference examples were available."
    blocks = []
    for idx, example in enumerate(examples, start=1):
        blocks.append(
            f"Example {idx} with ground truth pathology: {example['diagnosis']}\n"
            f"Similarity: {example['similarity']:.3f}\n"
            f"Patient Profile:\n{example['patient_profile']}"
        )
    return "\n\n".join(blocks)


def casebase_prior_text(case: UniversalCase, turns: list[WorkupTurn]) -> str:
    priors = [
        row for row in casebase_prior_for_case(case, turns)[:CASEBASE_PRIOR_CONTEXT_LABELS]
        if row.get("normalized_score", 0.0) >= CASEBASE_PRIOR_MIN_NORMALIZED_SCORE
    ]
    if not priors:
        return "No useful reference-case prior was available from the currently visible evidence."
    if len(priors) > 1:
        margin = float(priors[0].get("normalized_score", 0.0) - priors[1].get("normalized_score", 0.0))
        if margin < CASEBASE_PRIOR_PROMOTION_MARGIN:
            return (
                "Reference-case prior is low-margin under the currently visible evidence; "
                "do not use it to force a diagnosis."
            )
    lines = []
    for row in priors:
        terms = ", ".join(row["matched_terms"]) if row["matched_terms"] else "visible text overlap"
        lines.append(
            f"- {row['label']}: prior={row['normalized_score']:.3f}, "
            f"refs={row['supporting_reference_count']}, matched_terms={terms}"
        )
    return "\n".join(lines)


def apply_casebase_prior_rerank(
    llm_ranked: list[str],
    candidates: list[str],
    priors: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    clean_llm = canonicalize_ranked_differential(llm_ranked, candidates)
    prior_rows = [row for row in priors if row.get("normalized_score", 0.0) >= CASEBASE_PRIOR_MIN_NORMALIZED_SCORE]
    if not ENABLE_CASEBASE_PRIOR or not prior_rows:
        return clean_llm, {"changed": False, "top_label": "", "top_score": 0.0, "margin": 0.0}
    margin = 0.0
    if len(prior_rows) > 1:
        margin = float(prior_rows[0].get("normalized_score", 0.0) - prior_rows[1].get("normalized_score", 0.0))
    elif prior_rows:
        margin = float(prior_rows[0].get("normalized_score", 0.0))
    if margin < CASEBASE_PRIOR_PROMOTION_MARGIN:
        prior_top = prior_rows[0] if prior_rows else {}
        return clean_llm, {
            "changed": False,
            "top_label": str(prior_top.get("label", "")),
            "top_score": float(prior_top.get("normalized_score", 0.0) or 0.0),
            "margin": margin,
        }
    scores: dict[str, float] = {}
    tie_break: dict[str, int] = {}
    for rank, label in enumerate(clean_llm, start=1):
        scores[label] = scores.get(label, 0.0) + 1.0 / rank
        tie_break.setdefault(label, rank)
    next_rank = len(clean_llm) + 1
    for idx, row in enumerate(prior_rows, start=1):
        label = canonical_candidate_label(row["label"], candidates)
        if not label:
            continue
        scores[label] = scores.get(label, 0.0) + CASEBASE_PRIOR_WEIGHT * float(row.get("normalized_score", 0.0))
        tie_break.setdefault(label, next_rank + idx)
    ranked = sorted(scores, key=lambda label: (scores[label], -tie_break.get(label, 999), label), reverse=True)
    for label in clean_llm:
        if label not in ranked:
            ranked.append(label)
    prior_top = prior_rows[0] if prior_rows else {}
    return ranked[:10], {
        "changed": bool(ranked and clean_llm and ranked[0] != clean_llm[0]),
        "top_label": str(prior_top.get("label", "")),
        "top_score": float(prior_top.get("normalized_score", 0.0) or 0.0),
        "margin": margin,
    }


# %% [markdown]
# ## 5. RareBench Graph-Phenotype Resolver
#
# Notebook 43's RareBench failure came from treating phenotype strings as ordinary prose. Long generic phenotype profiles
# won lexical overlap even when a named syndrome was the closest exact phenotype exemplar. This layer converts RareBench
# phenotype names into atomic graph nodes and scores disease candidates by leave-one-case-out phenotype-set support.

# %%
@dataclass
class RarebenchPhenotypeReference:
    case_id: str
    subset: str
    diagnosis: str
    phenotypes: set[str]


def load_rarebench_phenotype_labels() -> list[str]:
    mapping_path = RAREBENCH_MAPPING_DIR / "rarebench_phenotype_mapping.json"
    if not mapping_path.exists():
        return []
    labels = sorted({str(value).strip() for value in read_json(mapping_path).values() if str(value).strip()}, key=len, reverse=True)
    # Very short/general HPO names are useful in ontology browsers but harmful as substring matches in profile text.
    blocked = {"All", "Mode of inheritance", "Bilateral", "Unilateral"}
    return [label for label in labels if label not in blocked and len(label) >= 4]


RAREBENCH_PHENOTYPE_LABELS = load_rarebench_phenotype_labels()


def build_rarebench_phenotype_references() -> list[RarebenchPhenotypeReference]:
    if not ENABLE_RAREBENCH_GRAPH_PHENOTYPE_RESOLVER:
        return []
    mapping_dir = RAREBENCH_MAPPING_DIR
    required = [
        mapping_dir / "rarebench_phenotype_mapping.json",
        mapping_dir / "rarebench_disease_mapping.json",
        mapping_dir / "disease_mapping.json",
    ]
    if any(not path.exists() for path in required):
        return []
    try:
        phenotype_mapping = read_json(mapping_dir / "rarebench_phenotype_mapping.json")
        code_to_disease = read_json(mapping_dir / "rarebench_disease_mapping.json")
        disease_mapping = read_json(mapping_dir / "disease_mapping.json")
        data_zip = ensure_rarebench_data_zip()
        refs: list[RarebenchPhenotypeReference] = []
        with zipfile.ZipFile(data_zip) as archive:
            available = set(archive.namelist())
            for subset in RAREBENCH_SUBSETS:
                member = f"data/{subset}.jsonl"
                if member not in available:
                    continue
                subset_mapping = disease_mapping.get(subset, {})
                with archive.open(member) as handle:
                    for idx, raw in enumerate(handle):
                        row = json.loads(raw.decode("utf-8"))
                        phenotype_codes = row.get("Phenotype") if isinstance(row.get("Phenotype"), list) else []
                        disease_codes = row.get("RareDisease") if isinstance(row.get("RareDisease"), list) else []
                        phenotypes = {str(phenotype_mapping[code]).strip() for code in phenotype_codes if code in phenotype_mapping}
                        disease_names = [str(code_to_disease[code]) for code in disease_codes if code in code_to_disease]
                        pathology_key = ", ".join(disease_names)
                        diagnosis = str(subset_mapping.get(pathology_key, disease_names[0] if disease_names else pathology_key))
                        if diagnosis and phenotypes:
                            refs.append(
                                RarebenchPhenotypeReference(
                                    case_id=f"rarebench:{subset}:{idx}",
                                    subset=subset,
                                    diagnosis=diagnosis,
                                    phenotypes=phenotypes,
                                )
                            )
        return refs
    except Exception as exc:
        print("RareBench graph-phenotype resolver skipped:", exc)
        return []


RAREBENCH_PHENOTYPE_REFERENCES = build_rarebench_phenotype_references()
if ENABLE_RAREBENCH_GRAPH_PHENOTYPE_RESOLVER:
    graph_reference_summary = pd.DataFrame([
        {
            "subset": subset,
            "reference_cases": sum(1 for ref in RAREBENCH_PHENOTYPE_REFERENCES if ref.subset == subset),
            "unique_diagnoses": len({ref.diagnosis for ref in RAREBENCH_PHENOTYPE_REFERENCES if ref.subset == subset}),
        }
        for subset in RAREBENCH_SUBSETS
    ])
    graph_reference_summary.to_csv(ARTIFACT_ROOT / "rarebench_graph_phenotype_reference_summary.csv", index=False)
    display(graph_reference_summary)

RAREBENCH_PHENOTYPE_IDF: dict[str, float] = {}
if RAREBENCH_PHENOTYPE_REFERENCES:
    ref_count = len(RAREBENCH_PHENOTYPE_REFERENCES)
    doc_freq = Counter()
    for ref in RAREBENCH_PHENOTYPE_REFERENCES:
        doc_freq.update(ref.phenotypes)
    RAREBENCH_PHENOTYPE_IDF = {
        phenotype: float(math.log((1 + ref_count) / (1 + freq)) + 1.0)
        for phenotype, freq in doc_freq.items()
    }


def remove_subsumed_phenotype_matches(phenotypes: set[str]) -> set[str]:
    kept = set(phenotypes)
    for item in sorted(phenotypes, key=len):
        if any(item != other and item.lower() in other.lower() for other in phenotypes):
            kept.discard(item)
    return kept


def extract_rarebench_visible_phenotypes(case: UniversalCase, turns: list[WorkupTurn]) -> set[str]:
    if case.dataset_name != "rarebench":
        return set()
    text_blocks = [case.initial_patient_info]
    for turn in turns:
        if turn.answer and "not mention" not in turn.answer.lower():
            text_blocks.append(turn.answer)
    text = "\n".join(text_blocks)
    direct = set()
    for line in text.splitlines():
        clean = line.strip().lstrip("-").strip()
        if clean in RAREBENCH_PHENOTYPE_LABELS:
            direct.add(clean)
    # Patient-simulator answers may concatenate several phenotype names without punctuation, so also use conservative
    # longest-label substring matching, then remove short labels subsumed by longer matches.
    for label in RAREBENCH_PHENOTYPE_LABELS:
        if label in text:
            direct.add(label)
    return remove_subsumed_phenotype_matches(direct)


def is_descriptive_rarebench_label(label: str) -> bool:
    text = str(label).lower()
    patterns = [
        "with or without",
        "neurodevelopmental disorder with",
        "intellectual developmental disorder with",
        "developmental and epileptic encephalopathy",
        "abnormalities of",
    ]
    return any(pattern in text for pattern in patterns)


def rarebench_graph_phenotype_prior_for_case(case: UniversalCase, turns: list[WorkupTurn]) -> list[dict[str, Any]]:
    if not ENABLE_RAREBENCH_GRAPH_PHENOTYPE_RESOLVER or case.dataset_name != "rarebench":
        return []
    visible = extract_rarebench_visible_phenotypes(case, turns)
    if len(visible) < RAREBENCH_GRAPH_MIN_VISIBLE_PHENOTYPES:
        return []
    subset = str(case.metadata.get("subset", ""))
    candidates = set(case.candidate_disease_list)
    label_best: dict[str, dict[str, Any]] = {}
    label_support_count: Counter[str] = Counter()
    for ref in RAREBENCH_PHENOTYPE_REFERENCES:
        if ref.case_id == case.case_id or (subset and ref.subset != subset):
            continue
        label = canonical_candidate_label(ref.diagnosis, case.candidate_disease_list)
        if not label or label not in candidates:
            continue
        overlap = visible & ref.phenotypes
        if not overlap:
            continue
        jaccard = len(overlap) / max(len(visible | ref.phenotypes), 1)
        visible_recall = len(overlap) / max(len(visible), 1)
        ref_precision = len(overlap) / max(len(ref.phenotypes), 1)
        score = jaccard + 0.05 * visible_recall + 0.05 * ref_precision
        label_support_count[label] += 1
        current = label_best.get(label)
        if current is None or score > current["score"]:
            label_best[label] = {
                "label": label,
                "score": float(score),
                "best_jaccard": float(jaccard),
                "visible_recall": float(visible_recall),
                "reference_precision": float(ref_precision),
                "overlap_count": int(len(overlap)),
                "best_reference_id": ref.case_id,
                "best_reference_phenotype_count": int(len(ref.phenotypes)),
                "matched_phenotypes": sorted(overlap)[:RAREBENCH_GRAPH_MATCH_TERMS],
                "visible_phenotype_count": int(len(visible)),
                "descriptive_label": is_descriptive_rarebench_label(label),
            }
    rows = list(label_best.values())
    for row in rows:
        row["supporting_reference_count"] = int(label_support_count[row["label"]])
    rows.sort(
        key=lambda row: (
            row["score"],
            row["overlap_count"],
            -int(row["descriptive_label"]),
            row["supporting_reference_count"],
            row["label"],
        ),
        reverse=True,
    )
    if rows:
        top_score = max(float(rows[0]["score"]), 1e-9)
        for row in rows:
            row["normalized_score"] = float(row["score"] / top_score)
    return rows[:RAREBENCH_GRAPH_RERANK_LABELS]


def rarebench_graph_phenotype_text(case: UniversalCase, turns: list[WorkupTurn]) -> str:
    if case.dataset_name != "rarebench" or not ENABLE_RAREBENCH_GRAPH_PHENOTYPE_RESOLVER:
        return "Not applicable."
    visible = sorted(extract_rarebench_visible_phenotypes(case, turns))
    rows = rarebench_graph_phenotype_prior_for_case(case, turns)[:RAREBENCH_GRAPH_CONTEXT_LABELS]
    if not visible:
        return "No exact visible RareBench phenotype nodes have been recovered yet."
    lines = ["Visible exact phenotype nodes: " + "; ".join(visible)]
    if not rows:
        lines.append("No graph-phenotype candidate support met the minimum evidence threshold.")
        return "\n".join(lines)
    lines.append("Leave-one-case-out graph-phenotype disease support:")
    for row in rows:
        matched = "; ".join(row["matched_phenotypes"]) if row["matched_phenotypes"] else "none"
        descriptor = "descriptive-label" if row["descriptive_label"] else "specific-label"
        lines.append(
            f"- {row['label']}: graph_prior={row['normalized_score']:.3f}, "
            f"jaccard={row['best_jaccard']:.3f}, overlap={row['overlap_count']}/{row['visible_phenotype_count']}, "
            f"best_ref={row['best_reference_id']}, {descriptor}, matched={matched}"
        )
    return "\n".join(lines)


def apply_rarebench_graph_rerank(
    current_ranked: list[str],
    candidates: list[str],
    graph_rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    clean_ranked = canonicalize_ranked_differential(current_ranked, candidates)
    if not graph_rows:
        return clean_ranked, {
            "changed": False,
            "top_label": "",
            "top_score": 0.0,
            "margin": 0.0,
            "visible_phenotype_count": 0,
        }
    usable_rows = [row for row in graph_rows if row.get("normalized_score", 0.0) >= RAREBENCH_GRAPH_MIN_NORMALIZED_SCORE]
    if not usable_rows:
        return clean_ranked, {
            "changed": False,
            "top_label": "",
            "top_score": 0.0,
            "margin": 0.0,
            "visible_phenotype_count": int(graph_rows[0].get("visible_phenotype_count", 0)),
        }
    scores: dict[str, float] = {}
    tie_break: dict[str, int] = {}
    for rank, label in enumerate(clean_ranked, start=1):
        scores[label] = scores.get(label, 0.0) + RAREBENCH_GRAPH_LLM_RANK_WEIGHT * (1.0 / rank)
        tie_break.setdefault(label, rank)
        if is_descriptive_rarebench_label(label):
            scores[label] -= RAREBENCH_GRAPH_DESCRIPTIVE_LABEL_PENALTY
    next_rank = len(clean_ranked) + 1
    for idx, row in enumerate(usable_rows, start=1):
        label = canonical_candidate_label(row["label"], candidates)
        if not label:
            continue
        graph_score = float(row.get("normalized_score", 0.0))
        scores[label] = scores.get(label, 0.0) + RAREBENCH_GRAPH_PRIOR_WEIGHT * graph_score
        if row.get("descriptive_label"):
            scores[label] -= RAREBENCH_GRAPH_DESCRIPTIVE_LABEL_PENALTY
        tie_break.setdefault(label, next_rank + idx)
    ranked = sorted(scores, key=lambda label: (scores[label], -tie_break.get(label, 999), label), reverse=True)
    for label in clean_ranked:
        if label not in ranked:
            ranked.append(label)
    top = usable_rows[0]
    margin = 0.0
    if len(usable_rows) > 1:
        margin = float(usable_rows[0].get("normalized_score", 0.0) - usable_rows[1].get("normalized_score", 0.0))
    else:
        margin = float(usable_rows[0].get("normalized_score", 0.0))
    return ranked[:10], {
        "changed": bool(ranked and clean_ranked and ranked[0] != clean_ranked[0]),
        "top_label": str(top.get("label", "")),
        "top_score": float(top.get("normalized_score", 0.0) or 0.0),
        "margin": margin,
        "visible_phenotype_count": int(top.get("visible_phenotype_count", 0) or 0),
    }


# %% [markdown]
# ## 6. Guarded Patient Simulator

# %%
class RetrievalPatientSimulator:
    def __init__(self, min_overlap: int = 1, max_spans: int = 3):
        self.min_overlap = min_overlap
        self.max_spans = max_spans

    TOPIC_GROUPS = {
        "respiratory": {"cough", "sputum", "phlegm", "wheeze", "wheezing", "dyspnea", "breath", "shortness", "chest", "stridor"},
        "infectious": {"fever", "chill", "chills", "sweat", "sweating", "night", "infection", "sick", "contact"},
        "ent": {"throat", "nasal", "nose", "congestion", "ear", "sinus", "hoarseness", "swallow"},
        "skin": {"rash", "skin", "lesion", "redness", "itch", "urticarial", "blister"},
        "pain": {"pain", "ache", "cramp", "radiate", "intense", "tender"},
        "gastro": {"abdominal", "belly", "epigastric", "vomit", "nausea", "diarrhea", "appetite", "jaundice"},
        "neuro": {"seizure", "developmental", "delay", "intellectual", "ataxia", "dystonia", "hypotonia", "regression", "microcephaly"},
        "metabolic": {"hypoglycemia", "acidosis", "ammonia", "hyperammonemia", "carnitine", "organic", "urine", "odor"},
    }

    def topic_hits(self, text: str) -> set[str]:
        tokens = tokenize(text)
        lowered = str(text).lower()
        hits = set()
        for topic, terms in self.TOPIC_GROUPS.items():
            if tokens & terms or any(term in lowered for term in terms if " " in term):
                hits.add(topic)
        return hits

    def question_focus_terms(self, question: str) -> set[str]:
        tokens = tokenize(question)
        generic = {"patient", "reported", "available", "additional", "other", "profile", "finding", "findings", "symptom", "symptoms", "present"}
        return {tok for tok in tokens if tok not in generic}

    def semantic_alignment_score(self, case: UniversalCase, question: str, span: str) -> float:
        if self.is_broad_inventory_question(question):
            return 0.0
        # DDXPlus spans are structured evidence-field answers. A broad infectious
        # question can legitimately reveal co-traveling systemic fields such as
        # appetite, fatigue, and myalgia. The topic-mismatch penalty is useful for
        # RareBench phenotype/prose retrieval, but it filtered out those DDXPlus
        # fields in the live pilot.
        if case.dataset_name == "ddxplus":
            return 0.0
        q_topics = self.topic_hits(question)
        s_topics = self.topic_hits(span)
        q_focus = self.question_focus_terms(question)
        s_tokens = tokenize(span)
        score = 0.0
        if q_topics:
            if q_topics & s_topics:
                score += 0.35
            elif s_topics:
                score -= 0.30
        critical_terms = {
            "fever", "chill", "chills", "sputum", "phlegm", "blood", "hemoptysis", "wheezing",
            "rash", "seizure", "hearing", "vision", "hypoglycemia", "acidosis", "jaundice",
        }
        focused = q_focus & critical_terms
        if focused and not (focused & s_tokens):
            score -= 0.35
        if case.dataset_name == "rarebench" and span.strip().lstrip("-").strip() in RAREBENCH_PHENOTYPE_IDF:
            score += 0.15 * RAREBENCH_PHENOTYPE_IDF[span.strip().lstrip("-").strip()]
        return score

    def is_broad_inventory_question(self, question: str) -> bool:
        text = str(question).lower()
        broad_terms = [
            "additional", "other", "more", "profile", "symptoms", "findings", "phenotypes",
            "features", "review of systems", "anything else", "reported", "present",
        ]
        return any(term in text for term in broad_terms)

    def broad_span_score(self, case: UniversalCase, span: str, span_index: int) -> float:
        text = str(span).lower()
        score = 1.0 / (span_index + 1)
        negative_patterns = [
            "answer: n", "answer: no", "answer: 0", "answer: na", "not mention", "nowhere",
        ]
        if any(pattern in text for pattern in negative_patterns):
            score -= 3.0
        if case.dataset_name == "ddxplus":
            high_yield = {
                "fever": 8.0,
                "tired": 7.0,
                "bed": 7.0,
                "fatigue": 7.0,
                "muscle": 7.0,
                "appetite": 6.0,
                "cough": 6.0,
                "sore throat": 5.0,
                "immunosuppressed": 5.0,
                "sweating": 3.0,
                "travel": 2.0,
                "smoke": 1.0,
            }
        elif case.dataset_name == "rarebench":
            high_yield = {
                "calcification": 8.0,
                "intracerebral": 8.0,
                "growth": 7.0,
                "hypoplasia": 6.0,
                "microphthalmia": 6.0,
                "intellectual": 6.0,
                "hearing": 5.0,
                "feeding": 5.0,
                "microcephaly": 4.0,
                "short stature": 4.0,
                "contracture": 4.0,
                "scoliosis": 3.0,
            }
        else:
            high_yield = {
                "positive": 3.0,
                "examination": 3.0,
                "laboratory": 3.0,
                "toxicology": 3.0,
                "history": 2.0,
                "denies": 1.0,
                "rash": 1.0,
            }
        for term, weight in high_yield.items():
            if term in text:
                score += weight
        if case.dataset_name == "rarebench":
            phenotype = str(span).strip().lstrip("-").strip()
            score += 0.35 * RAREBENCH_PHENOTYPE_IDF.get(phenotype, 0.0)
        return score

    def broad_inventory(self, case: UniversalCase, seen_spans: set[str] | None = None) -> list[dict[str, Any]]:
        spans = split_profile_into_spans(case.hidden_full_profile)
        seen_spans = seen_spans or set()
        scored = []
        for idx, span in enumerate(spans):
            if span in seen_spans:
                continue
            score = self.broad_span_score(case, span, idx)
            scored.append({"span_index": idx, "span": span, "overlap": 0, "score": score})
        scored.sort(key=lambda row: (row["score"], -row["span_index"]), reverse=True)
        kept = [row for row in scored if row["score"] > -1.0]
        if not kept and seen_spans:
            return self.broad_inventory(case, seen_spans=set())
        return kept[: self.max_spans]

    def retrieve(self, case: UniversalCase, question: str, seen_spans: set[str] | None = None) -> list[dict[str, Any]]:
        if self.is_broad_inventory_question(question):
            return self.broad_inventory(case, seen_spans=seen_spans)
        q_tokens = tokenize(question)
        spans = split_profile_into_spans(case.hidden_full_profile)
        scored = []
        seen_spans = seen_spans or set()
        for idx, span in enumerate(spans):
            if span in seen_spans:
                continue
            s_tokens = tokenize(span)
            overlap = len(q_tokens & s_tokens)
            if not q_tokens:
                score = 0.0
            else:
                score = overlap / max(len(q_tokens), 1)
            score += self.semantic_alignment_score(case, question, span)
            # Generic "tell me more" questions should still reveal broad profile context.
            if any(token in question.lower() for token in ["additional", "other", "more", "profile", "symptoms", "findings"]):
                score += 0.05 if idx < 10 else 0.0
            scored.append({"span_index": idx, "span": span, "overlap": overlap, "score": score})
        scored.sort(key=lambda row: (row["score"], row["overlap"], -row["span_index"]), reverse=True)
        kept = [row for row in scored if (row["overlap"] >= self.min_overlap and row["score"] > -0.10) or row["score"] > 0.18]
        if not kept and seen_spans:
            return self.retrieve(case, question, seen_spans=set())
        return kept[: self.max_spans]

    def answer(self, case: UniversalCase, question: str, seen_spans: set[str] | None = None) -> tuple[str, list[str], dict[str, Any]]:
        retrieved = self.retrieve(case, question, seen_spans=seen_spans)
        if not retrieved:
            answer = "The available patient profile does not mention that."
            spans: list[str] = []
        else:
            spans = [row["span"] for row in retrieved]
            answer = " ".join(spans)
        audit = {
            "case_id": case.case_id,
            "dataset_name": case.dataset_name,
            "question": question,
            "answer": answer,
            "retrieved_span_count": len(spans),
            "retrieved_spans": spans,
        }
        return answer, spans, audit


patient_simulator = RetrievalPatientSimulator(min_overlap=PATIENT_SIMULATOR_MIN_OVERLAP, max_spans=PATIENT_SIMULATOR_MAX_SPANS)

# %% [markdown]
# ## 7. MEDDx-Style Agents

# %%
def call_openai_compatible(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any], dict[str, int]]:
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is empty. Fill LLM_API_KEY in the first cell or use the interactive prompt.")
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(5):
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code < 500 and response.status_code != 429:
            response.raise_for_status()
            raw_payload = response.json()
            text = raw_payload["choices"][0]["message"]["content"]
            usage_raw = raw_payload.get("usage", {})
            usage = {
                "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            }
            return text, raw_payload, usage
        time.sleep(4 + attempt * 4)
    response.raise_for_status()
    raise RuntimeError("unreachable")


def parse_agent_response(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        payload = json.loads(match.group(0)) if match else {}
    decision = str(payload.get("decision", "stop")).lower().strip()
    if decision not in {"request", "stop"}:
        decision = "stop"
    ranked = normalize_list(payload.get("ranked_differential", []))[:10]
    predicted = str(payload.get("predicted_diagnosis") or payload.get("predicted_pathology") or (ranked[0] if ranked else ""))
    if predicted and predicted not in ranked:
        ranked = [predicted] + [item for item in ranked if item != predicted]
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return {
        "decision": decision,
        "question": str(payload.get("question") or payload.get("requested_question") or "").strip(),
        "predicted_diagnosis": predicted,
        "ranked_differential": ranked[:10],
        "confidence": confidence,
        "brief_reasoning": str(payload.get("brief_reasoning") or payload.get("reasoning") or ""),
    }


def candidate_text(candidates: list[str], max_chars: int = CANDIDATE_TEXT_MAX_CHARS) -> str:
    text = json.dumps(candidates, ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    kept = []
    total = 2
    for candidate in candidates:
        extra = len(candidate) + 4
        if total + extra > max_chars:
            break
        kept.append(candidate)
        total += extra
    return json.dumps(kept, ensure_ascii=True) + "\n[truncated candidate list]"


def ledger_text(turns: list[WorkupTurn]) -> str:
    if not turns:
        return "No follow-up questions have been asked yet."
    lines = []
    for turn in turns:
        lines.append(f"Q{turn.turn_index}: {turn.question}")
        lines.append(f"A{turn.turn_index}: {turn.answer}")
    return "\n".join(lines)


def dataset_guidance(case: UniversalCase) -> str:
    if case.dataset_name == "ddxplus":
        return (
            "This case was converted from structured DDXPlus evidence. Patient answers often look like "
            "'Question? Answer: yes/no/value.' Ask targeted symptom, risk-factor, timing, location, severity, "
            "exam, travel, exposure, or comorbidity questions. Do not repeat a question already present in the ledger."
        )
    if case.dataset_name == "icraft_md":
        return (
            "This is an iCraft-MD dermatology multiple-choice case. The allowed candidate diagnoses are the case options. "
            "Ask about lesion morphology, distribution, timing, exposures, medications, systemic symptoms, exam findings, "
            "and pathology clues. Final diagnoses must use one of the listed option names."
        )
    if case.dataset_name == "rarebench":
        return (
            "This is a RareBench rare-disease case represented mainly by phenotype terms. Ask for additional phenotype "
            "findings, affected organ systems, developmental/neurologic/metabolic features, inheritance/family clues, "
            "and onset information. If a specific pathophysiology question is not mentioned, switch back to asking for "
            "additional phenotypes. Do not bias toward early candidate-list items."
        )
    return "Ask concise clinical questions that distinguish the leading candidate diagnoses."


def known_profile_text(case: UniversalCase, turns: list[WorkupTurn]) -> str:
    lines = [case.initial_patient_info]
    for turn in turns:
        if turn.answer and "not mention" not in turn.answer.lower():
            lines.append(f"- {turn.answer}")
    return "\n".join(lines)


def build_history_taking_messages(case: UniversalCase, turns: list[WorkupTurn], budget: int) -> list[dict[str, str]]:
    remaining = max(budget - len(turns), 0)
    system = (
        "You are the MEDDx-style history-taking agent. Your job is to ask exactly one concise clinical "
        "question that would reveal useful symptoms, antecedents, exam findings, labs, imaging, exposures, "
        "or phenotype clues. You cannot inspect hidden patient data directly. Return valid JSON only."
    )
    first_turn_instruction = ""
    if BROAD_INVENTORY_FIRST_TURN and not turns:
        first_turn_instruction = (
            "First-turn preference: ask a broad inventory question for additional positive symptoms, findings, "
            "or phenotype features before committing to a narrow disease-specific path."
        )
    user = f"""
Dataset: {case.dataset_name}
Case id: {case.case_id}

Initial patient information:
{case.initial_patient_info}

Allowed candidate diagnoses:
{candidate_text(case.candidate_disease_list)}

Dataset-specific guidance:
{dataset_guidance(case)}

Reference-case prior from visible evidence only:
{casebase_prior_text(case, turns)}

Question budget: {budget}
Questions already asked: {len(turns)}
Questions remaining: {remaining}

Question-answer ledger:
{ledger_text(turns)}

{first_turn_instruction}

Return JSON with exactly these keys:
{{
  "question": "single natural-language clinical question, or \"None\" if no useful question remains",
  "target": "what clinical distinction this question is meant to resolve",
  "brief_reasoning": "one short sentence"
}}

Rules:
- Do not ask for a dataset field id.
- Do not ask multiple questions at once.
- Do not repeat or paraphrase an already asked question.
- Keep questions clinically meaningful and answerable from a patient profile.
- Candidate-list order is arbitrary; do not prefer a diagnosis because it appears earlier.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def build_final_diagnosis_messages(case: UniversalCase, turns: list[WorkupTurn]) -> list[dict[str, str]]:
    system = (
        "You are the MEDDx-style diagnosis strategy agent. Given the known patient profile, dynamic reference "
        "examples, previous evidence ledger, and exact diagnosis options, return the top ranked differential. "
        "Use exact disease names from the candidate list. Return valid JSON only."
    )
    user = f"""
Dataset: {case.dataset_name}
Case id: {case.case_id}

Known patient profile:
{known_profile_text(case, turns)}

Question-answer ledger:
{ledger_text(turns)}

Allowed candidate diagnoses:
{candidate_text(case.candidate_disease_list)}

Dataset-specific guidance:
{dataset_guidance(case)}

Dynamic similar-patient examples:
{dynamic_fewshot_text(case, turns, k=5)}

Reference-case prior from visible evidence only:
{casebase_prior_text(case, turns)}

RareBench exact graph-phenotype support, when applicable:
{rarebench_graph_phenotype_text(case, turns)}

Return JSON with exactly these keys:
{{
  "predicted_diagnosis": "best diagnosis from the candidate list",
  "ranked_differential": ["exactly 10 candidate diagnoses if available, best first"],
  "confidence": 0.0,
  "brief_reasoning": "one short sentence"
}}

Rules:
- Use exact disease names from the candidate list.
- Do not output diagnoses outside the candidate list.
- Candidate-list order is arbitrary; do not prefer a diagnosis because it appears earlier.
- Use the dynamic examples as MEDDx-style few-shot support, but do not copy them blindly.
- For RareBench, treat exact phenotype-node graph support as a mathematical audit signal. Prefer a specific named disease
  over a broad descriptive category when the phenotype graph supports the named disease at least as well.
- If uncertain, keep plausible alternatives in the ranked differential rather than collapsing the list.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def parse_history_response(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        payload = json.loads(match.group(0)) if match else {}
    question = str(payload.get("question") or "").strip()
    return {
        "question": question,
        "target": str(payload.get("target") or ""),
        "brief_reasoning": str(payload.get("brief_reasoning") or payload.get("reasoning") or ""),
    }


def scripted_history_response(case: UniversalCase, turns: list[WorkupTurn], budget: int) -> dict[str, Any]:
    question_bank = [
        "What additional symptoms, signs, findings, or phenotype features are reported?",
        "What timing, duration, exposures, medications, family history, or context are reported?",
        "Are there any exam, lab, imaging, skin, neurologic, or rare-disease clues mentioned?",
    ]
    if len(turns) >= min(2, budget):
        return {"question": "None", "target": "dry-run stop", "brief_reasoning": "Dry-run scripted history stop."}
    return {
        "question": question_bank[min(len(turns), len(question_bank) - 1)],
        "target": "dry-run evidence inventory",
        "brief_reasoning": "Dry-run scripted history request for artifact validation.",
    }


def scripted_final_response(case: UniversalCase, turns: list[WorkupTurn]) -> dict[str, Any]:
    candidates = case.candidate_disease_list[:10] if case.candidate_disease_list else [case.ground_truth_diagnosis]
    predicted = candidates[0] if candidates else "Unknown"
    return {
        "predicted_diagnosis": predicted,
        "ranked_differential": candidates[:10],
        "confidence": 0.35,
        "brief_reasoning": "Dry-run scripted diagnosis for artifact validation.",
    }


def get_history_response(case: UniversalCase, turns: list[WorkupTurn], budget: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if not RUN_LIVE_API:
        response = scripted_history_response(case, turns, budget)
        return response, {"dry_run": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    messages = build_history_taking_messages(case, turns, budget)
    raw_text, payload, usage = call_openai_compatible(messages)
    try:
        return parse_history_response(raw_text), payload, usage
    except Exception:
        repair_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": "Repair the previous answer into valid JSON with the required keys only."},
        ]
        repair_text, repair_payload, repair_usage = call_openai_compatible(repair_messages)
        usage = {
            "input_tokens": usage["input_tokens"] + repair_usage["input_tokens"],
            "output_tokens": usage["output_tokens"] + repair_usage["output_tokens"],
        }
        return parse_history_response(repair_text), {"first_payload": payload, "repair_payload": repair_payload}, usage


def get_final_diagnosis_response(case: UniversalCase, turns: list[WorkupTurn]) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if not RUN_LIVE_API:
        response = scripted_final_response(case, turns)
        return response, {"dry_run": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    messages = build_final_diagnosis_messages(case, turns)
    raw_text, payload, usage = call_openai_compatible(messages)
    try:
        return parse_agent_response(raw_text), payload, usage
    except Exception:
        repair_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": "Repair the previous answer into valid JSON with the required keys only."},
        ]
        repair_text, repair_payload, repair_usage = call_openai_compatible(repair_messages)
        usage = {
            "input_tokens": usage["input_tokens"] + repair_usage["input_tokens"],
            "output_tokens": usage["output_tokens"] + repair_usage["output_tokens"],
        }
        return parse_agent_response(repair_text), {"first_payload": payload, "repair_payload": repair_payload}, usage


def build_rarebench_discriminator_messages(
    case: UniversalCase,
    turns: list[WorkupTurn],
    current_ranked: list[str],
    graph_ranked: list[str],
    graph_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    graph_pool = []
    for row in graph_rows[:RAREBENCH_GRAPH_CONTEXT_LABELS]:
        graph_pool.append(str(row.get("label", "")))
    candidate_pool = []
    for label in [*graph_ranked, *current_ranked, *graph_pool]:
        clean = canonical_candidate_label(label, case.candidate_disease_list)
        if clean and clean not in candidate_pool:
            candidate_pool.append(clean)
    candidate_pool = candidate_pool[:18]
    system = (
        "You are a rare disease differential diagnosis adjudicator. You receive a visible phenotype ledger, "
        "a first-pass ranked differential, and a graph-phenotype audit built from exact HPO phenotype-node overlap. "
        "Return valid JSON only."
    )
    user = f"""
Dataset: RareBench
Case id: {case.case_id}

Known patient profile:
{known_profile_text(case, turns)}

Question-answer ledger:
{ledger_text(turns)}

First-pass ranked differential:
{json.dumps(current_ranked[:10], ensure_ascii=True)}

Graph-phenotype reranked differential:
{json.dumps(graph_ranked[:10], ensure_ascii=True)}

Candidate pool for final adjudication:
{json.dumps(candidate_pool, ensure_ascii=True)}

Exact graph-phenotype audit:
{rarebench_graph_phenotype_text(case, turns)}

Return JSON with exactly these keys:
{{
  "predicted_diagnosis": "best disease from the candidate pool",
  "ranked_differential": ["up to 10 candidate-pool diseases, best first"],
  "confidence": 0.0,
  "brief_reasoning": "one short sentence comparing phenotype support"
}}

Rules:
- Use exact candidate-pool names only.
- The graph audit is label-free and based only on visible phenotypes plus other RareBench cases excluding this case.
- Do not choose a broad descriptive phenotype-category label if a specific named syndrome in the pool explains the phenotype constellation at least as well.
- Keep close alternatives in the ranked differential.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def scripted_rarebench_discriminator_response(
    case: UniversalCase,
    graph_ranked: list[str],
    current_ranked: list[str],
) -> dict[str, Any]:
    ranked = graph_ranked[:10] if graph_ranked else current_ranked[:10]
    predicted = ranked[0] if ranked else (case.candidate_disease_list[0] if case.candidate_disease_list else "")
    return {
        "predicted_diagnosis": predicted,
        "ranked_differential": ranked[:10],
        "confidence": 0.40,
        "brief_reasoning": "Dry-run graph-phenotype discriminator response for artifact validation.",
    }


def get_rarebench_discriminator_response(
    case: UniversalCase,
    turns: list[WorkupTurn],
    current_ranked: list[str],
    graph_ranked: list[str],
    graph_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if not RAREBENCH_LLM_DISCRIMINATOR or case.dataset_name != "rarebench" or not graph_rows:
        response = scripted_rarebench_discriminator_response(case, graph_ranked, current_ranked)
        return response, {"skipped": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    if not RUN_LIVE_API:
        response = scripted_rarebench_discriminator_response(case, graph_ranked, current_ranked)
        return response, {"dry_run": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    messages = build_rarebench_discriminator_messages(case, turns, current_ranked, graph_ranked, graph_rows)
    raw_text, payload, usage = call_openai_compatible(messages)
    try:
        return parse_agent_response(raw_text), payload, usage
    except Exception:
        repair_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": "Repair the previous answer into valid JSON with the required keys only."},
        ]
        repair_text, repair_payload, repair_usage = call_openai_compatible(repair_messages)
        usage = {
            "input_tokens": usage["input_tokens"] + repair_usage["input_tokens"],
            "output_tokens": usage["output_tokens"] + repair_usage["output_tokens"],
        }
        return parse_agent_response(repair_text), {"first_payload": payload, "repair_payload": repair_payload}, usage


def canonicalize_agent_response(response: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    response = dict(response)
    ranked = canonicalize_ranked_differential(normalize_list(response.get("ranked_differential", [])), candidates)
    predicted = canonicalize_to_candidates(str(response.get("predicted_diagnosis", "")), candidates)
    if predicted and predicted not in ranked:
        ranked = [predicted] + [item for item in ranked if item != predicted]
    response["predicted_diagnosis"] = predicted
    response["ranked_differential"] = ranked[:10]
    return response


# %% [markdown]
# ## 8. Cap-Aware Stop, Branch, And Resolver Stack

# %%
def reciprocal_rank(label: str, ranked: list[str]) -> float:
    norm = normalize_label_text(label)
    for idx, candidate in enumerate(ranked, start=1):
        if normalize_label_text(candidate) == norm:
            return 1.0 / idx
    return 0.0


def add_candidate_score(
    scores: dict[str, dict[str, Any]],
    label: str,
    value: float,
    source: str,
    candidates: list[str],
    support_if_positive: bool = True,
) -> None:
    clean = canonical_candidate_label(label, candidates)
    if not clean:
        return
    row = scores.setdefault(clean, {"label": clean, "score": 0.0, "sources": [], "support_count": 0})
    row["score"] += float(value)
    row["sources"].append({"source": source, "value": float(value)})
    if support_if_positive and value > 0:
        row["support_count"] += 1


def resolver_from_candidate_sources(
    case: UniversalCase,
    turns: list[WorkupTurn],
    base_ranked: list[str],
    base_confidence: float,
    branches: list[HypothesisBranchResult] | None = None,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    branches = branches or []
    candidates = case.candidate_disease_list
    clean_base = canonicalize_ranked_differential(base_ranked, candidates)
    scores: dict[str, dict[str, Any]] = {}
    for rank, label in enumerate(clean_base[:10], start=1):
        add_candidate_score(scores, label, 1.15 / rank, "base_llm_rank", candidates)
    if clean_base:
        add_candidate_score(scores, clean_base[0], 0.35 * float(base_confidence), "base_llm_confidence", candidates)

    priors = casebase_prior_for_case(case, turns)
    for row in priors[:CASEBASE_PRIOR_RERANK_LABELS]:
        add_candidate_score(
            scores,
            str(row.get("label", "")),
            0.80 * float(row.get("normalized_score", 0.0)),
            "casebase_prior",
            candidates,
        )

    graph_rows = rarebench_graph_phenotype_prior_for_case(case, turns)
    for row in graph_rows[:RAREBENCH_GRAPH_RERANK_LABELS]:
        add_candidate_score(
            scores,
            str(row.get("label", "")),
            1.15 * float(row.get("normalized_score", 0.0)),
            "rarebench_graph",
            candidates,
        )

    for branch in branches:
        for rank, label in enumerate(branch.ranked_differential[:10], start=1):
            add_candidate_score(
                scores,
                label,
                0.95 / rank,
                f"branch:{branch.branch_id}",
                candidates,
            )
        if branch.predicted_diagnosis:
            add_candidate_score(
                scores,
                branch.predicted_diagnosis,
                0.20 * float(branch.confidence),
                f"branch_conf:{branch.branch_id}",
                candidates,
                support_if_positive=False,
            )

    ordered = sorted(scores.values(), key=lambda row: (row["score"], row["support_count"], row["label"]), reverse=True)
    if not ordered and clean_base:
        ordered = [{"label": label, "score": 1.0 / idx, "support_count": 1, "sources": [{"source": "base_fallback", "value": 1.0 / idx}]} for idx, label in enumerate(clean_base, start=1)]
    top = ordered[0] if ordered else {"label": "", "score": 0.0, "support_count": 0}
    second = ordered[1] if len(ordered) > 1 else {"label": "", "score": 0.0, "support_count": 0}
    margin = float(top.get("score", 0.0) - second.get("score", 0.0))
    base_top = clean_base[0] if clean_base else ""
    selected_top = str(top.get("label", ""))
    changed = bool(base_top and selected_top and normalize_label_text(base_top) != normalize_label_text(selected_top))
    protected = False
    if changed:
        challenger_support = int(top.get("support_count", 0))
        base_row = next((row for row in ordered if normalize_label_text(row["label"]) == normalize_label_text(base_top)), None)
        base_score = float(base_row.get("score", 0.0)) if base_row else 0.0
        challenger_margin = float(top.get("score", 0.0) - base_score)
        if challenger_margin < RESOLVER_BASE_PROTECTION_MARGIN or challenger_support < RESOLVER_MIN_INDEPENDENT_SUPPORT_TO_OVERRIDE:
            protected = True
            selected_top = base_top
            ordered = [
                {"label": base_top, "score": base_score + 0.01, "support_count": int(base_row.get("support_count", 1) if base_row else 1), "sources": (base_row or {}).get("sources", [])}
            ] + [row for row in ordered if normalize_label_text(row["label"]) != normalize_label_text(base_top)]
    ranked = []
    seen = set()
    for row in ordered:
        label = str(row["label"])
        key = normalize_label_text(label)
        if key and key not in seen:
            ranked.append(label)
            seen.add(key)
    for label in clean_base:
        key = normalize_label_text(label)
        if key not in seen:
            ranked.append(label)
            seen.add(key)
    detail = {
        "resolver_margin": margin,
        "resolver_support_count": int(top.get("support_count", 0)),
        "resolver_changed_top1": bool(changed and not protected),
        "resolver_base_protected": bool(protected),
        "base_top": base_top,
        "raw_top": str(top.get("label", "")),
        "selected_top": ranked[0] if ranked else selected_top,
    }
    rows = []
    for idx, row in enumerate(ordered[:BRANCH_CANDIDATE_POOL_LABELS], start=1):
        rows.append({
            "candidate_rank": idx,
            "label": row["label"],
            "resolver_score": float(row["score"]),
            "support_count": int(row["support_count"]),
            "sources": json.dumps(row["sources"], ensure_ascii=True),
        })
    return ranked[:10], detail, rows


def stop_probe_margin_from_ranked(ranked: list[str], case: UniversalCase, turns: list[WorkupTurn], confidence: float) -> float:
    resolved_ranked, detail, _rows = resolver_from_candidate_sources(case, turns, ranked, confidence, branches=[])
    return float(detail.get("resolver_margin", 0.0))


def make_universal_stop_signal(
    case: UniversalCase,
    turns: list[WorkupTurn],
    probe_response: dict[str, Any],
    prior_probes: list[StopProbe],
    budget: int,
    mlp_history: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranked = canonicalize_ranked_differential(normalize_list(probe_response.get("ranked_differential", [])), case.candidate_disease_list)
    predicted = str(probe_response.get("predicted_diagnosis") or (ranked[0] if ranked else ""))
    confidence = float(probe_response.get("confidence", 0.0) or 0.0)
    rank_margin = stop_probe_margin_from_ranked(ranked, case, turns, confidence)
    stability_turns = 0
    for previous in reversed(prior_probes):
        if normalize_label_text(previous.predicted_diagnosis) == normalize_label_text(predicted):
            stability_turns += 1
        else:
            break
    mlp_feedback = compute_ddxplus_mlp_feedback(case, turns, mlp_history)
    if mlp_feedback.get("available"):
        mlp_history.append(mlp_feedback)
    mlp_top1 = str(mlp_feedback.get("top1", "") or "")
    mlp_agrees_with_probe = bool(
        mlp_feedback.get("available")
        and predicted
        and mlp_top1
        and normalize_label_text(mlp_top1) == normalize_label_text(predicted)
    )
    ddxplus_mlp_disagrees_with_probe = bool(
        case.dataset_name == "ddxplus"
        and mlp_feedback.get("available")
        and predicted
        and mlp_top1
        and normalize_label_text(mlp_top1) != normalize_label_text(predicted)
    )
    min_questions = int(EARLY_STOP_MIN_QUESTIONS_BY_BUDGET.get(int(budget), max(3, int(budget) // 2)))
    remaining = int(budget - len(turns))
    universal_fired = bool(
        len(turns) >= min_questions
        and confidence >= EARLY_STOP_CONFIDENCE_MIN
        and rank_margin >= EARLY_STOP_RANK_MARGIN_MIN
        and stability_turns >= EARLY_STOP_STABILITY_MIN
    )
    if case.dataset_name == "ddxplus" and mlp_feedback.get("available"):
        universal_fired = bool(universal_fired and mlp_agrees_with_probe)
    mlp_fired = bool(
        mlp_feedback.get("available")
        and len(turns) >= min_questions
        and mlp_agrees_with_probe
        and float(mlp_feedback.get("confidence", 0.0)) >= EARLY_STOP_DDXPLUS_MLP_CONFIDENCE_MIN
        and float(mlp_feedback.get("margin", 0.0)) >= EARLY_STOP_DDXPLUS_MLP_MARGIN_MIN
        and float(mlp_feedback.get("entropy", 1.0)) <= EARLY_STOP_DDXPLUS_MLP_ENTROPY_MAX
    )
    force_continue = remaining > 0 and len(turns) < min_questions
    should_stop = bool((universal_fired or mlp_fired or remaining <= 0) and not force_continue)
    reason = "continue"
    if remaining <= 0:
        reason = "budget_exhausted"
    elif mlp_fired:
        reason = "ddxplus_partial_mlp_stop"
    elif universal_fired:
        reason = "universal_confidence_stop"
    return {
        "should_stop": should_stop,
        "reason": reason,
        "confidence": confidence,
        "rank_margin": rank_margin,
        "stability_turns": int(stability_turns),
        "min_questions": min_questions,
        "remaining_budget": remaining,
        "universal_stop_rule_fired": universal_fired,
        "ddxplus_mlp_stop_rule_fired": mlp_fired,
        "ddxplus_mlp_available": bool(mlp_feedback.get("available", False)),
        "ddxplus_mlp_top1": mlp_top1,
        "ddxplus_mlp_top5": list(mlp_feedback.get("top_predictions", []) or [])[:5],
        "ddxplus_mlp_top_probabilities": list(mlp_feedback.get("top_probabilities", []) or [])[:5],
        "ddxplus_mlp_agrees_with_probe": mlp_agrees_with_probe,
        "ddxplus_mlp_disagrees_with_probe": ddxplus_mlp_disagrees_with_probe,
        "ddxplus_mlp_revealed_root_count": int(mlp_feedback.get("revealed_root_count", 0) or 0),
        "ddxplus_mlp_confidence": float(mlp_feedback.get("confidence", 0.0) or 0.0),
        "ddxplus_mlp_margin": float(mlp_feedback.get("margin", 0.0) or 0.0),
        "ddxplus_mlp_entropy": float(mlp_feedback.get("entropy", 1.0) or 1.0),
        "predicted_diagnosis": predicted,
    }, mlp_feedback


def branch_trigger_features(
    case: UniversalCase,
    turns: list[WorkupTurn],
    base_ranked: list[str],
    base_confidence: float,
    resolver_detail: dict[str, Any],
) -> dict[str, Any]:
    priors = casebase_prior_for_case(case, turns)
    graph_rows = rarebench_graph_phenotype_prior_for_case(case, turns)
    source_tops = []
    if base_ranked:
        source_tops.append(base_ranked[0])
    if priors:
        source_tops.append(str(priors[0].get("label", "")))
    if graph_rows:
        source_tops.append(str(graph_rows[0].get("label", "")))
    nonempty = [normalize_label_text(label) for label in source_tops if label]
    disagreement = 0.0 if len(nonempty) <= 1 else 1.0 - (max(Counter(nonempty).values()) / len(nonempty))
    margin = float(resolver_detail.get("resolver_margin", 0.0))
    confidence = float(base_confidence)
    triggered = bool(
        ENABLE_HYPOTHESIS_BRANCHING
        and len(case.candidate_disease_list) > 1
        and (
            confidence <= BRANCH_TRIGGER_CONFIDENCE_MAX
            or margin <= BRANCH_TRIGGER_MARGIN_MAX
            or disagreement >= BRANCH_TRIGGER_DISAGREEMENT_MIN
        )
    )
    return {
        "triggered": triggered,
        "confidence": confidence,
        "resolver_margin": margin,
        "source_disagreement": float(disagreement),
        "source_tops": source_tops,
    }


def propose_branch_hypotheses(
    case: UniversalCase,
    turns: list[WorkupTurn],
    base_ranked: list[str],
    extra_hypothesis_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    hypotheses = []
    seen = {normalize_label_text(base_ranked[0])} if base_ranked else set()
    for source, labels in [
        ("ddxplus_mlp_monitor", list(extra_hypothesis_labels or [])[:4]),
        ("base_ranked", base_ranked[1:5]),
        ("casebase_prior", [row.get("label", "") for row in casebase_prior_for_case(case, turns)[:4]]),
        ("rarebench_graph", [row.get("label", "") for row in rarebench_graph_phenotype_prior_for_case(case, turns)[:4]]),
    ]:
        for label in labels:
            clean = canonical_candidate_label(str(label), case.candidate_disease_list)
            key = normalize_label_text(clean)
            if clean and key not in seen:
                hypotheses.append({"target_hypothesis": clean, "source": source})
                seen.add(key)
            if len(hypotheses) >= BRANCH_MAX_BRANCHES:
                return hypotheses
    return hypotheses[:BRANCH_MAX_BRANCHES]


def build_branch_question_messages(
    case: UniversalCase,
    base_turns: list[WorkupTurn],
    branch_turns: list[WorkupTurn],
    target_hypothesis: str,
    base_top: str,
    remaining_branch_questions: int,
) -> list[dict[str, str]]:
    system = (
        "You are a hypothesis-forced diagnostic branch agent. Explore one plausible alternative diagnosis without "
        "ignoring the shared evidence ledger. Ask exactly one concise question that could distinguish the branch "
        "hypothesis from the current leading diagnosis. Return valid JSON only."
    )
    user = f"""
Dataset: {case.dataset_name}
Case id: {case.case_id}

Current leading diagnosis: {base_top}
Forced branch hypothesis: {target_hypothesis}
Remaining branch questions: {remaining_branch_questions}

Shared known profile:
{known_profile_text(case, base_turns)}

Shared base ledger:
{ledger_text(base_turns)}

Branch-specific ledger:
{ledger_text(branch_turns)}

Allowed candidate diagnoses:
{candidate_text(case.candidate_disease_list)}

Dataset guidance:
{dataset_guidance(case)}

Return JSON with exactly these keys:
{{
  "question": "single natural-language clinical question, or \"None\" if no useful question remains",
  "target": "what evidence would separate the branch hypothesis",
  "brief_reasoning": "one short sentence"
}}

Rules:
- Ask one differentiating question, not a general diagnosis request.
- Do not repeat the base or branch ledger.
- The branch may disagree with the base, but must remain evidence-seeking rather than argumentative.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def get_branch_question_response(
    case: UniversalCase,
    base_turns: list[WorkupTurn],
    branch_turns: list[WorkupTurn],
    target_hypothesis: str,
    base_top: str,
    remaining_branch_questions: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if not RUN_LIVE_API:
        response = {
            "question": "What finding would best distinguish this alternative diagnosis from the current leading diagnosis?",
            "target": target_hypothesis,
            "brief_reasoning": "Dry-run hypothesis branch question.",
        }
        return response, {"dry_run": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    messages = build_branch_question_messages(case, base_turns, branch_turns, target_hypothesis, base_top, remaining_branch_questions)
    raw_text, payload, usage = call_openai_compatible(messages)
    try:
        return parse_history_response(raw_text), payload, usage
    except Exception:
        repair_text, repair_payload, repair_usage = call_openai_compatible(messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": "Repair the previous answer into valid JSON with the required keys only."},
        ])
        usage = {
            "input_tokens": usage["input_tokens"] + repair_usage["input_tokens"],
            "output_tokens": usage["output_tokens"] + repair_usage["output_tokens"],
        }
        return parse_history_response(repair_text), {"first_payload": payload, "repair_payload": repair_payload}, usage


def build_branch_final_messages(case: UniversalCase, base_turns: list[WorkupTurn], branch: HypothesisBranchResult | dict[str, Any], target_hypothesis: str, base_top: str) -> list[dict[str, str]]:
    branch_turns = branch.turns if isinstance(branch, HypothesisBranchResult) else branch.get("turns", [])
    system = (
        "You are the final judge for a hypothesis-forced diagnostic branch. Rank exact candidate diagnoses using the "
        "shared base ledger plus the branch-specific evidence. Return valid JSON only."
    )
    user = f"""
Dataset: {case.dataset_name}
Case id: {case.case_id}

Base leading diagnosis: {base_top}
Branch target hypothesis: {target_hypothesis}

Known profile from base:
{known_profile_text(case, base_turns)}

Base ledger:
{ledger_text(base_turns)}

Branch ledger:
{ledger_text(branch_turns)}

Allowed candidate diagnoses:
{candidate_text(case.candidate_disease_list)}

Graph/casebase audit:
{casebase_prior_text(case, [*base_turns, *branch_turns])}
{rarebench_graph_phenotype_text(case, [*base_turns, *branch_turns])}

Return JSON with exactly these keys:
{{
  "predicted_diagnosis": "best diagnosis from the candidate list",
  "ranked_differential": ["up to 10 exact candidate diagnoses, best first"],
  "confidence": 0.0,
  "brief_reasoning": "one short sentence"
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def get_branch_final_response(case: UniversalCase, base_turns: list[WorkupTurn], branch_turns: list[WorkupTurn], target_hypothesis: str, base_top: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if not RUN_LIVE_API:
        ranked = [target_hypothesis] + [label for label in case.candidate_disease_list[:10] if normalize_label_text(label) != normalize_label_text(target_hypothesis)]
        response = {"predicted_diagnosis": ranked[0], "ranked_differential": ranked[:10], "confidence": 0.45, "brief_reasoning": "Dry-run branch final response."}
        return response, {"dry_run": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    messages = build_branch_final_messages(case, base_turns, {"turns": branch_turns}, target_hypothesis, base_top)
    raw_text, payload, usage = call_openai_compatible(messages)
    try:
        return parse_agent_response(raw_text), payload, usage
    except Exception:
        repair_text, repair_payload, repair_usage = call_openai_compatible(messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": "Repair the previous answer into valid JSON with the required keys only."},
        ])
        usage = {
            "input_tokens": usage["input_tokens"] + repair_usage["input_tokens"],
            "output_tokens": usage["output_tokens"] + repair_usage["output_tokens"],
        }
        return parse_agent_response(repair_text), {"first_payload": payload, "repair_payload": repair_payload}, usage


def run_hypothesis_branches(
    case: UniversalCase,
    base_turns: list[WorkupTurn],
    base_ranked: list[str],
    budget_remaining: int,
    extra_hypothesis_labels: list[str] | None = None,
) -> tuple[list[HypothesisBranchResult], list[dict[str, Any]], int, int, int]:
    if not ENABLE_HYPOTHESIS_BRANCHING or budget_remaining < BRANCH_MIN_REMAINING_BUDGET or not base_ranked:
        return [], [], 0, 0, 0
    hypotheses = propose_branch_hypotheses(case, base_turns, base_ranked, extra_hypothesis_labels=extra_hypothesis_labels)
    branches: list[HypothesisBranchResult] = []
    raw_responses: list[dict[str, Any]] = []
    input_tokens = output_tokens = api_calls = 0
    total_branch_questions = 0
    base_top = base_ranked[0]
    for branch_index, hypothesis in enumerate(hypotheses, start=1):
        if budget_remaining <= 0:
            break
        branch_turns: list[WorkupTurn] = []
        branch_question_cap = min(BRANCH_MAX_QUESTIONS_PER_BRANCH, budget_remaining)
        target = str(hypothesis["target_hypothesis"])
        for local_turn in range(1, branch_question_cap + 1):
            response, raw_payload, usage = get_branch_question_response(case, base_turns, branch_turns, target, base_top, branch_question_cap - local_turn + 1)
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
            api_calls += 0 if not RUN_LIVE_API else 1
            raw_responses.append({"branch_id": f"branch_{branch_index}", "agent": "branch_question", "response": response, "raw_payload": raw_payload})
            question = str(response.get("question") or "").strip()
            if not question or question.lower() == "none":
                break
            seen_spans = {span for turn in [*base_turns, *branch_turns] for span in turn.retrieved_spans}
            answer, spans, _audit = patient_simulator.answer(case, question, seen_spans=seen_spans)
            branch_turns.append(
                WorkupTurn(
                    turn_index=local_turn,
                    question=question,
                    answer=answer,
                    retrieved_spans=spans,
                    decision_before_answer=f"branch_{branch_index}",
                    predicted_before_answer=target,
                    ranked_before_answer=base_ranked[:10],
                )
            )
            budget_remaining -= 1
            total_branch_questions += 1
            if budget_remaining <= 0:
                break
        final_response, raw_payload, usage = get_branch_final_response(case, base_turns, branch_turns, target, base_top)
        final_response = canonicalize_agent_response(final_response, case.candidate_disease_list)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        api_calls += 0 if not RUN_LIVE_API else 1
        raw_responses.append({"branch_id": f"branch_{branch_index}", "agent": "branch_final", "response": final_response, "raw_payload": raw_payload})
        ranked = list(final_response.get("ranked_differential") or [])
        predicted = str(final_response.get("predicted_diagnosis") or (ranked[0] if ranked else target))
        branches.append(
            HypothesisBranchResult(
                branch_id=f"branch_{branch_index}",
                target_hypothesis=target,
                branch_role=str(hypothesis["source"]),
                turns=branch_turns,
                predicted_diagnosis=predicted,
                ranked_differential=ranked[:10],
                confidence=float(final_response.get("confidence", 0.0) or 0.0),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                api_calls=api_calls,
                raw_responses=list(raw_responses),
            )
        )
    return branches, raw_responses, input_tokens, output_tokens, api_calls


def apply_rarebench_conservative_gate(
    *,
    case: UniversalCase,
    llm_ranked: list[str],
    graph_ranked: list[str],
    discriminator_ranked: list[str],
    graph_detail: dict[str, Any],
) -> tuple[list[str], str]:
    if case.dataset_name != "rarebench" or not RAREBENCH_CONSERVATIVE_GRAPH_GATE:
        return discriminator_ranked or graph_ranked or llm_ranked, "not_applicable"
    llm_top = llm_ranked[0] if llm_ranked else ""
    graph_top = graph_ranked[0] if graph_ranked else ""
    disc_top = discriminator_ranked[0] if discriminator_ranked else ""
    llm_key = normalize_label_text(llm_top)
    graph_key = normalize_label_text(graph_top)
    disc_key = normalize_label_text(disc_top)
    graph_margin = float(graph_detail.get("margin", 0.0) or 0.0)
    if RAREBENCH_LOCK_WHEN_LLM_AND_GRAPH_AGREE and llm_top and graph_top and llm_key == graph_key:
        merged = [llm_top] + [x for x in (discriminator_ranked + graph_ranked + llm_ranked) if normalize_label_text(x) != normalize_label_text(llm_top)]
        return canonicalize_ranked_differential(merged, case.candidate_disease_list), "locked_llm_graph_agreement"
    if disc_top and disc_key not in {llm_key, graph_key} and graph_margin < RAREBENCH_DISCRIMINATOR_THIRD_OPTION_MARGIN_MIN:
        return llm_ranked, "blocked_unsupported_third_option"
    if graph_top and llm_top and graph_key != llm_key:
        if graph_margin < RAREBENCH_GRAPH_OVERRIDE_MARGIN_MIN:
            return llm_ranked, "blocked_weak_graph_override"
    return discriminator_ranked or graph_ranked or llm_ranked, "accepted"

# %% [markdown]
# ## 9. Workup Execution

# %%
def run_case(case: UniversalCase, budget: int) -> WorkupResult:
    turns: list[WorkupTurn] = []
    stop_probes: list[StopProbe] = []
    mlp_history: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    api_calls = 0
    initial_ranked: list[str] = casebase_prior_for_case(case, [])[:CASEBASE_PRIOR_CONTEXT_LABELS]
    initial_ranked_labels = [row["label"] for row in initial_ranked]
    early_stop_reason = "budget_exhausted_forced_final"
    last_probe_response: dict[str, Any] | None = None
    last_stop_signal: dict[str, Any] = {}

    for turn_index in range(1, budget + 1):
        response, raw_payload, usage = get_history_response(case, turns, budget)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        api_calls += 0 if not RUN_LIVE_API else 1
        raw_responses.append({"turn_index": turn_index, "agent": "history_taking", "response": response, "raw_payload": raw_payload})
        question = str(response.get("question") or "").strip()
        if not question or question.lower() == "none":
            early_stop_reason = "history_agent_stop"
            break
        seen_spans = {span for turn in turns for span in turn.retrieved_spans}
        answer, spans, audit = patient_simulator.answer(case, question, seen_spans=seen_spans)
        turns.append(
            WorkupTurn(
                turn_index=turn_index,
                question=question,
                answer=answer,
                retrieved_spans=spans,
                decision_before_answer="history_taking",
                predicted_before_answer="",
                ranked_before_answer=[],
            )
        )
        if ENABLE_CAP_AWARE_EARLY_STOP and len(turns) < budget:
            probe_response, probe_payload, probe_usage = get_final_diagnosis_response(case, turns)
            probe_response = canonicalize_agent_response(probe_response, case.candidate_disease_list)
            input_tokens += probe_usage.get("input_tokens", 0)
            output_tokens += probe_usage.get("output_tokens", 0)
            api_calls += 0 if not RUN_LIVE_API else 1
            stop_signal, _mlp_feedback = make_universal_stop_signal(case, turns, probe_response, stop_probes, budget, mlp_history)
            probe_ranked = list(probe_response.get("ranked_differential") or [])
            stop_probe = StopProbe(
                turn_index=turn_index,
                predicted_diagnosis=str(probe_response.get("predicted_diagnosis") or (probe_ranked[0] if probe_ranked else "")),
                ranked_differential=probe_ranked[:10],
                confidence=float(probe_response.get("confidence", 0.0) or 0.0),
                rank_margin=float(stop_signal.get("rank_margin", 0.0) or 0.0),
                stability_turns=int(stop_signal.get("stability_turns", 0) or 0),
                stop_signal=stop_signal,
            )
            stop_probes.append(stop_probe)
            last_probe_response = probe_response
            last_stop_signal = stop_signal
            raw_responses.append({
                "turn_index": turn_index,
                "agent": "cap_aware_stop_probe",
                "response": probe_response,
                "stop_signal": stop_signal,
                "raw_payload": probe_payload,
            })
            if stop_signal.get("should_stop"):
                early_stop_reason = str(stop_signal.get("reason", "cap_aware_stop"))
                break

    if last_probe_response is not None and len(turns) < budget and str(last_stop_signal.get("reason", "")).endswith("stop"):
        final_response = canonicalize_agent_response(last_probe_response, case.candidate_disease_list)
        raw_payload = {"reused_stop_probe": True}
    else:
        final_response, raw_payload, usage = get_final_diagnosis_response(case, turns)
        final_response = canonicalize_agent_response(final_response, case.candidate_disease_list)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        api_calls += 0 if not RUN_LIVE_API else 1
    raw_responses.append({"turn_index": len(turns) + 1, "agent": "base_diagnosis", "response": final_response, "raw_payload": raw_payload})

    ranked = list(final_response.get("ranked_differential") or [])
    predicted = str(final_response.get("predicted_diagnosis") or (ranked[0] if ranked else ""))
    if predicted and predicted not in ranked:
        ranked = [predicted] + ranked
    if not ranked and case.candidate_disease_list:
        ranked = case.candidate_disease_list[:10]
        predicted = ranked[0]
    llm_ranked = ranked[:10]
    llm_predicted = predicted

    final_priors = casebase_prior_for_case(case, turns)
    ranked, prior_detail = apply_casebase_prior_rerank(llm_ranked, case.candidate_disease_list, final_priors)
    predicted = ranked[0] if ranked else predicted
    graph_rows = rarebench_graph_phenotype_prior_for_case(case, turns)
    graph_ranked, graph_detail = apply_rarebench_graph_rerank(ranked, case.candidate_disease_list, graph_rows)
    discriminator_used = False
    rarebench_gate_action = "not_applicable"
    if case.dataset_name == "rarebench" and graph_rows:
        discriminator_response, discriminator_payload, discriminator_usage = get_rarebench_discriminator_response(
            case,
            turns,
            current_ranked=ranked,
            graph_ranked=graph_ranked,
            graph_rows=graph_rows,
        )
        discriminator_response = canonicalize_agent_response(discriminator_response, case.candidate_disease_list)
        input_tokens += discriminator_usage.get("input_tokens", 0)
        output_tokens += discriminator_usage.get("output_tokens", 0)
        api_calls += 0 if not RUN_LIVE_API or discriminator_payload.get("skipped") else 1
        raw_responses.append({
            "turn_index": len(turns) + 2,
            "agent": "rarebench_graph_phenotype_discriminator",
            "response": discriminator_response,
            "raw_payload": discriminator_payload,
        })
        discriminator_ranked = list(discriminator_response.get("ranked_differential") or [])
        if discriminator_ranked:
            gated_ranked, rarebench_gate_action = apply_rarebench_conservative_gate(
                case=case,
                llm_ranked=ranked,
                graph_ranked=graph_ranked,
                discriminator_ranked=discriminator_ranked[:10],
                graph_detail=graph_detail,
            )
            ranked = gated_ranked[:10]
            predicted = str(discriminator_response.get("predicted_diagnosis") or ranked[0])
            if predicted and predicted not in ranked:
                ranked = [predicted] + [label for label in ranked if label != predicted]
            if ranked:
                predicted = ranked[0]
            discriminator_used = bool(not discriminator_payload.get("skipped"))
        else:
            ranked, rarebench_gate_action = apply_rarebench_conservative_gate(
                case=case,
                llm_ranked=ranked,
                graph_ranked=graph_ranked,
                discriminator_ranked=[],
                graph_detail=graph_detail,
            )
            predicted = ranked[0] if ranked else predicted
    elif graph_rows:
        ranked = graph_ranked
        predicted = ranked[0] if ranked else predicted

    pre_branch_ranked = ranked[:10]
    pre_branch_confidence = float(final_response.get("confidence", 0.0) or 0.0)
    preliminary_resolved, preliminary_detail, _preliminary_scores = resolver_from_candidate_sources(
        case, turns, pre_branch_ranked, pre_branch_confidence, branches=[]
    )
    trigger = branch_trigger_features(case, turns, pre_branch_ranked, pre_branch_confidence, preliminary_detail)
    latest_mlp_for_trigger = next((row for row in reversed(mlp_history) if row.get("available")), {})
    latest_mlp_top1 = str(latest_mlp_for_trigger.get("top1", "") or "")
    latest_mlp_top5 = list(latest_mlp_for_trigger.get("top_predictions", []) or [])[:5]
    mlp_agrees_with_base_top = bool(
        latest_mlp_top1
        and pre_branch_ranked
        and normalize_label_text(latest_mlp_top1) == normalize_label_text(pre_branch_ranked[0])
    )
    mlp_high_confidence = bool(
        bool(latest_mlp_for_trigger.get("available", False))
        and float(latest_mlp_for_trigger.get("confidence", 0.0)) >= 0.90
        and float(latest_mlp_for_trigger.get("margin", 0.0)) >= 0.50
        and float(latest_mlp_for_trigger.get("entropy", 1.0)) <= 0.10
    )
    if (
        mlp_high_confidence
        and mlp_agrees_with_base_top
    ):
        trigger["triggered"] = False
        trigger["suppressed_by"] = "high_confidence_agreeing_ddxplus_mlp_monitor"
    elif (
        bool(latest_mlp_for_trigger.get("available", False))
        and latest_mlp_top1
        and pre_branch_ranked
        and not mlp_agrees_with_base_top
    ):
        trigger["triggered"] = True
        trigger["forced_by"] = "ddxplus_mlp_llm_disagreement"
        trigger["ddxplus_mlp_top1"] = latest_mlp_top1
        trigger["ddxplus_mlp_top5"] = latest_mlp_top5
    remaining_budget = max(0, int(budget) - len(turns))
    branches: list[HypothesisBranchResult] = []
    branch_raw_responses: list[dict[str, Any]] = []
    branch_input = branch_output = branch_api = 0
    if trigger.get("triggered") and remaining_budget >= BRANCH_MIN_REMAINING_BUDGET:
        branches, branch_raw_responses, branch_input, branch_output, branch_api = run_hypothesis_branches(
            case=case,
            base_turns=turns,
            base_ranked=pre_branch_ranked,
            budget_remaining=remaining_budget,
            extra_hypothesis_labels=latest_mlp_top5,
        )
        input_tokens += branch_input
        output_tokens += branch_output
        api_calls += branch_api
        raw_responses.extend(branch_raw_responses)

    ranked, resolver_detail, candidate_scores = resolver_from_candidate_sources(
        case, turns, pre_branch_ranked, pre_branch_confidence, branches=branches
    )
    predicted = ranked[0] if ranked else predicted

    true_rank_final = rank_true_label(ranked, case.ground_truth_diagnosis)
    true_rank_initial = rank_true_label(initial_ranked_labels or ranked, case.ground_truth_diagnosis)
    branch_question_count = sum(len(branch.turns) for branch in branches)
    total_questions = len(turns) + branch_question_count
    latest_mlp = next((row for row in reversed(mlp_history) if row.get("available")), {})
    return WorkupResult(
        case_id=case.case_id,
        dataset_name=case.dataset_name,
        budget=budget,
        predicted_diagnosis=predicted,
        ranked_differential=ranked[:10],
        confidence=float(final_response.get("confidence", 0.0)),
        num_questions=total_questions,
        stopped_early=total_questions < budget,
        stop_reason=early_stop_reason if total_questions < budget else "budget_exhausted_forced_final",
        correct_top1=str(predicted).strip().lower() == str(case.ground_truth_diagnosis).strip().lower(),
        gtpa_at_3=topk_hit(ranked, case.ground_truth_diagnosis, 3),
        gtpa_at_5=topk_hit(ranked, case.ground_truth_diagnosis, 5),
        true_rank=true_rank_final,
        initial_true_rank=true_rank_initial,
        progress_improved=true_rank_final < true_rank_initial,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=api_calls,
        estimated_cost=None,
        llm_predicted_diagnosis=llm_predicted,
        llm_ranked_differential=llm_ranked,
        casebase_prior_top_label=str(prior_detail.get("top_label", "")),
        casebase_prior_top_score=float(prior_detail.get("top_score", 0.0) or 0.0),
        casebase_prior_margin=float(prior_detail.get("margin", 0.0) or 0.0),
        casebase_resolver_changed=bool(prior_detail.get("changed", False)),
        rarebench_graph_top_label=str(graph_detail.get("top_label", "")),
        rarebench_graph_top_score=float(graph_detail.get("top_score", 0.0) or 0.0),
        rarebench_graph_margin=float(graph_detail.get("margin", 0.0) or 0.0),
        rarebench_graph_visible_phenotypes=int(graph_detail.get("visible_phenotype_count", 0) or 0),
        rarebench_graph_resolver_changed=bool(graph_detail.get("changed", False)),
        rarebench_discriminator_used=bool(discriminator_used),
        stop_probe_count=len(stop_probes),
        early_stop_signal=str(last_stop_signal.get("reason", early_stop_reason)),
        ddxplus_mlp_available=bool(latest_mlp.get("available", False)),
        ddxplus_mlp_top1=str(latest_mlp.get("top1", "") or ""),
        ddxplus_mlp_top5=list(latest_mlp.get("top_predictions", []) or [])[:5],
        ddxplus_mlp_confidence=float(latest_mlp.get("confidence", 0.0) or 0.0),
        ddxplus_mlp_margin=float(latest_mlp.get("margin", 0.0) or 0.0),
        ddxplus_mlp_entropy=float(latest_mlp.get("entropy", 1.0) or 1.0),
        branch_triggered=bool(branches),
        branch_count=len(branches),
        branch_question_count=branch_question_count,
        resolver_margin=float(resolver_detail.get("resolver_margin", 0.0) or 0.0),
        resolver_support_count=int(resolver_detail.get("resolver_support_count", 0) or 0),
        resolver_changed_top1=bool(resolver_detail.get("resolver_changed_top1", False)),
        rarebench_gate_action=rarebench_gate_action,
        turns=turns,
        branches=branches,
        stop_probes=stop_probes,
        candidate_scores=candidate_scores,
        raw_responses=raw_responses,
    )


predictions_path = ARTIFACT_ROOT / "predictions.csv"
ledger_path = ARTIFACT_ROOT / "question_answer_ledger.csv"
trace_path = ARTIFACT_ROOT / "interaction_traces.jsonl"
simulator_audit_path = ARTIFACT_ROOT / "patient_simulator_retrieval_audit.csv"
branch_path = ARTIFACT_ROOT / "branch_case_results.csv"
candidate_scores_path = ARTIFACT_ROOT / "candidate_level_resolver_scores.csv"

if RUN_LIVE_API:
    active_budgets = LIVE_BUDGETS_TO_RUN
else:
    active_budgets = MEDDX_REFERENCE_BUDGETS if DRY_RUN_ALL_BUDGETS else MEDDX_REFERENCE_BUDGETS[:1]
existing_keys: set[tuple[str, str, int]] = set()
if RESUME_IF_AVAILABLE and predictions_path.exists():
    existing = pd.read_csv(predictions_path)
    existing_keys = set(zip(existing["dataset_name"].astype(str), existing["case_id"].astype(str), existing["budget"].astype(int)))

prediction_rows: list[dict[str, Any]] = []
ledger_rows: list[dict[str, Any]] = []
audit_rows: list[dict[str, Any]] = []
branch_rows: list[dict[str, Any]] = []
candidate_score_rows: list[dict[str, Any]] = []

if not universal_cases:
    print("No dataset adapters produced cases. Supply external paths or check DDXPlus files.")
elif not RUN_LIVE_API and not ALLOW_DRY_RUN_BENCHMARK:
    print("Benchmark execution skipped. Set RUN_LIVE_API=True or ALLOW_DRY_RUN_BENCHMARK=True.")
else:
    tasks = [(case, budget) for case in universal_cases for budget in active_budgets]
    print("Active budgets      :", active_budgets)
    print("Total workups queued:", len(tasks), f"({len(universal_cases)} cases x {len(active_budgets)} budgets)")
    for case, budget in tqdm(tasks, desc="Unified MEDDx-style workups"):
        key = (case.dataset_name, case.case_id, int(budget))
        if key in existing_keys:
            continue
        result = run_case(case, budget)
        prediction_row = {
            "dataset_name": result.dataset_name,
            "case_id": result.case_id,
            "budget": result.budget,
            "ground_truth_diagnosis": case.ground_truth_diagnosis,
            "predicted_diagnosis": result.predicted_diagnosis,
            "ranked_differential": json.dumps(result.ranked_differential, ensure_ascii=True),
            "llm_predicted_diagnosis": result.llm_predicted_diagnosis,
            "llm_ranked_differential": json.dumps(result.llm_ranked_differential, ensure_ascii=True),
            "casebase_prior_top_label": result.casebase_prior_top_label,
            "casebase_prior_top_score": result.casebase_prior_top_score,
            "casebase_prior_margin": result.casebase_prior_margin,
            "casebase_resolver_changed": result.casebase_resolver_changed,
            "rarebench_graph_top_label": result.rarebench_graph_top_label,
            "rarebench_graph_top_score": result.rarebench_graph_top_score,
            "rarebench_graph_margin": result.rarebench_graph_margin,
            "rarebench_graph_visible_phenotypes": result.rarebench_graph_visible_phenotypes,
            "rarebench_graph_resolver_changed": result.rarebench_graph_resolver_changed,
            "rarebench_discriminator_used": result.rarebench_discriminator_used,
            "rarebench_gate_action": result.rarebench_gate_action,
            "stop_probe_count": result.stop_probe_count,
            "early_stop_signal": result.early_stop_signal,
            "ddxplus_mlp_available": result.ddxplus_mlp_available,
            "ddxplus_mlp_top1": result.ddxplus_mlp_top1,
            "ddxplus_mlp_top5": json.dumps(result.ddxplus_mlp_top5, ensure_ascii=True),
            "ddxplus_mlp_confidence": result.ddxplus_mlp_confidence,
            "ddxplus_mlp_margin": result.ddxplus_mlp_margin,
            "ddxplus_mlp_entropy": result.ddxplus_mlp_entropy,
            "branch_triggered": result.branch_triggered,
            "branch_count": result.branch_count,
            "branch_question_count": result.branch_question_count,
            "resolver_margin": result.resolver_margin,
            "resolver_support_count": result.resolver_support_count,
            "resolver_changed_top1": result.resolver_changed_top1,
            "confidence": result.confidence,
            "num_questions": result.num_questions,
            "stopped_early": result.stopped_early,
            "stop_reason": result.stop_reason,
            "correct_top1": result.correct_top1,
            "gtpa_at_3": result.gtpa_at_3,
            "gtpa_at_5": result.gtpa_at_5,
            "true_rank": result.true_rank,
            "initial_true_rank": result.initial_true_rank,
            "progress_improved": result.progress_improved,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "api_calls": result.api_calls,
            "run_live_api": bool(RUN_LIVE_API),
        }
        prediction_rows.append(prediction_row)
        append_jsonl(trace_path, {
            "dataset_name": result.dataset_name,
            "case_id": result.case_id,
            "budget": result.budget,
            "ground_truth_diagnosis": case.ground_truth_diagnosis,
            "initial_patient_info": case.initial_patient_info,
            "turns": [asdict(turn) for turn in result.turns],
            "raw_responses": result.raw_responses,
            "llm_prediction_before_casebase": result.llm_predicted_diagnosis,
            "llm_ranked_differential_before_casebase": result.llm_ranked_differential,
            "casebase_prior_top_label": result.casebase_prior_top_label,
            "casebase_prior_top_score": result.casebase_prior_top_score,
            "casebase_prior_margin": result.casebase_prior_margin,
            "casebase_resolver_changed": result.casebase_resolver_changed,
            "rarebench_graph_top_label": result.rarebench_graph_top_label,
            "rarebench_graph_top_score": result.rarebench_graph_top_score,
            "rarebench_graph_margin": result.rarebench_graph_margin,
            "rarebench_graph_visible_phenotypes": result.rarebench_graph_visible_phenotypes,
            "rarebench_graph_resolver_changed": result.rarebench_graph_resolver_changed,
            "rarebench_discriminator_used": result.rarebench_discriminator_used,
            "rarebench_gate_action": result.rarebench_gate_action,
            "ddxplus_mlp_top1": result.ddxplus_mlp_top1,
            "ddxplus_mlp_top5": result.ddxplus_mlp_top5,
            "stop_probes": [asdict(probe) for probe in result.stop_probes],
            "branches": [asdict(branch) for branch in result.branches],
            "candidate_scores": result.candidate_scores,
            "final_prediction": result.predicted_diagnosis,
            "ranked_differential": result.ranked_differential,
        })
        for turn in result.turns:
            ledger_rows.append({
                "dataset_name": result.dataset_name,
                "case_id": result.case_id,
                "budget": result.budget,
                "turn_index": turn.turn_index,
                "question": turn.question,
                "answer": turn.answer,
                "retrieved_spans": json.dumps(turn.retrieved_spans, ensure_ascii=True),
                "predicted_before_answer": turn.predicted_before_answer,
                "ranked_before_answer": json.dumps(turn.ranked_before_answer, ensure_ascii=True),
            })
            audit_rows.append({
                "dataset_name": result.dataset_name,
                "case_id": result.case_id,
                "budget": result.budget,
                "turn_index": turn.turn_index,
                "question": turn.question,
                "answer": turn.answer,
                "retrieved_span_count": len(turn.retrieved_spans),
                "retrieved_spans": json.dumps(turn.retrieved_spans, ensure_ascii=True),
            })
        for branch in result.branches:
            branch_rows.append({
                "dataset_name": result.dataset_name,
                "case_id": result.case_id,
                "budget": result.budget,
                "branch_id": branch.branch_id,
                "target_hypothesis": branch.target_hypothesis,
                "branch_role": branch.branch_role,
                "predicted_diagnosis": branch.predicted_diagnosis,
                "ranked_differential": json.dumps(branch.ranked_differential, ensure_ascii=True),
                "confidence": branch.confidence,
                "num_branch_questions": len(branch.turns),
                "input_tokens": branch.input_tokens,
                "output_tokens": branch.output_tokens,
                "api_calls": branch.api_calls,
            })
            for turn in branch.turns:
                ledger_rows.append({
                    "dataset_name": result.dataset_name,
                    "case_id": result.case_id,
                    "budget": result.budget,
                    "turn_index": f"{branch.branch_id}:{turn.turn_index}",
                    "question": turn.question,
                    "answer": turn.answer,
                    "retrieved_spans": json.dumps(turn.retrieved_spans, ensure_ascii=True),
                    "predicted_before_answer": turn.predicted_before_answer,
                    "ranked_before_answer": json.dumps(turn.ranked_before_answer, ensure_ascii=True),
                })
        for row in result.candidate_scores:
            candidate_score_rows.append({
                "dataset_name": result.dataset_name,
                "case_id": result.case_id,
                "budget": result.budget,
                **row,
            })

    if prediction_rows:
        new_predictions = pd.DataFrame(prediction_rows)
        if predictions_path.exists() and RESUME_IF_AVAILABLE:
            old_predictions = pd.read_csv(predictions_path)
            combined = pd.concat([old_predictions, new_predictions], ignore_index=True)
            combined = combined.drop_duplicates(["dataset_name", "case_id", "budget"], keep="last")
        else:
            combined = new_predictions
        combined.to_csv(predictions_path, index=False)
    if ledger_rows:
        ledger_frame = pd.DataFrame(ledger_rows)
        if ledger_path.exists() and RESUME_IF_AVAILABLE:
            old_ledger = pd.read_csv(ledger_path)
            ledger_frame = pd.concat([old_ledger, ledger_frame], ignore_index=True)
            ledger_frame = ledger_frame.drop_duplicates(["dataset_name", "case_id", "budget", "turn_index"], keep="last")
        ledger_frame.to_csv(ledger_path, index=False)
    if audit_rows:
        audit_frame = pd.DataFrame(audit_rows)
        if simulator_audit_path.exists() and RESUME_IF_AVAILABLE:
            old_audit = pd.read_csv(simulator_audit_path)
            audit_frame = pd.concat([old_audit, audit_frame], ignore_index=True)
            audit_frame = audit_frame.drop_duplicates(["dataset_name", "case_id", "budget", "turn_index"], keep="last")
        audit_frame.to_csv(simulator_audit_path, index=False)
    if branch_rows:
        branch_frame = pd.DataFrame(branch_rows)
        if branch_path.exists() and RESUME_IF_AVAILABLE:
            old_branch = pd.read_csv(branch_path)
            branch_frame = pd.concat([old_branch, branch_frame], ignore_index=True)
            branch_frame = branch_frame.drop_duplicates(["dataset_name", "case_id", "budget", "branch_id"], keep="last")
        branch_frame.to_csv(branch_path, index=False)
    if candidate_score_rows:
        candidate_frame = pd.DataFrame(candidate_score_rows)
        if candidate_scores_path.exists() and RESUME_IF_AVAILABLE:
            old_candidates = pd.read_csv(candidate_scores_path)
            candidate_frame = pd.concat([old_candidates, candidate_frame], ignore_index=True)
            candidate_frame = candidate_frame.drop_duplicates(["dataset_name", "case_id", "budget", "candidate_rank"], keep="last")
        candidate_frame.to_csv(candidate_scores_path, index=False)

predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
display(predictions.head())

# %% [markdown]
# ## 10. MEDDx-Style Evaluation

# %%
def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "num_cases": int(len(group)),
        "gtpa_at_1": float(group["correct_top1"].astype(bool).mean()) if len(group) else np.nan,
        "gtpa_at_3": float(group["gtpa_at_3"].astype(bool).mean()) if len(group) else np.nan,
        "gtpa_at_5": float(group["gtpa_at_5"].astype(bool).mean()) if len(group) else np.nan,
        "mean_true_rank_capped_11": float(pd.to_numeric(group["true_rank"], errors="coerce").mean()) if len(group) else np.nan,
        "progress_rate": float(group["progress_improved"].astype(bool).mean()) if len(group) else np.nan,
        "mean_questions": float(pd.to_numeric(group["num_questions"], errors="coerce").mean()) if len(group) else np.nan,
        "median_questions": float(pd.to_numeric(group["num_questions"], errors="coerce").median()) if len(group) else np.nan,
        "stop_before_budget_rate": float(group["stopped_early"].astype(bool).mean()) if len(group) else np.nan,
        "graph_resolver_change_rate": float(group.get("rarebench_graph_resolver_changed", pd.Series([False] * len(group))).astype(bool).mean()) if len(group) else np.nan,
        "graph_discriminator_use_rate": float(group.get("rarebench_discriminator_used", pd.Series([False] * len(group))).astype(bool).mean()) if len(group) else np.nan,
        "branch_trigger_rate": float(group.get("branch_triggered", pd.Series([False] * len(group))).astype(bool).mean()) if len(group) else np.nan,
        "mean_branch_count": float(pd.to_numeric(group.get("branch_count", pd.Series([0] * len(group))), errors="coerce").mean()) if len(group) else np.nan,
        "mean_branch_questions": float(pd.to_numeric(group.get("branch_question_count", pd.Series([0] * len(group))), errors="coerce").mean()) if len(group) else np.nan,
        "resolver_change_rate": float(group.get("resolver_changed_top1", pd.Series([False] * len(group))).astype(bool).mean()) if len(group) else np.nan,
        "mean_resolver_margin": float(pd.to_numeric(group.get("resolver_margin", pd.Series([0] * len(group))), errors="coerce").mean()) if len(group) else np.nan,
        "mean_input_tokens": float(pd.to_numeric(group["input_tokens"], errors="coerce").mean()) if len(group) else np.nan,
        "mean_output_tokens": float(pd.to_numeric(group["output_tokens"], errors="coerce").mean()) if len(group) else np.nan,
    }


if len(predictions):
    rows = []
    for (dataset_name, budget), group in predictions.groupby(["dataset_name", "budget"], sort=True):
        rows.append({"dataset_name": dataset_name, "budget": int(budget), **summarize_group(group)})
    metrics_summary = pd.DataFrame(rows).sort_values(["dataset_name", "budget"])
else:
    metrics_summary = pd.DataFrame(columns=["dataset_name", "budget", "num_cases", "gtpa_at_1", "gtpa_at_3", "gtpa_at_5"])

metrics_summary.to_csv(ARTIFACT_ROOT / "meddx_style_metrics_summary.csv", index=False)
display(metrics_summary)

resolved_run_config = {
    "run_name": RUN_NAME,
    "run_live_api": bool(RUN_LIVE_API),
    "allow_dry_run_benchmark": bool(ALLOW_DRY_RUN_BENCHMARK),
    "require_all_enabled_datasets": bool(REQUIRE_ALL_ENABLED_DATASETS),
    "llm_model": LLM_MODEL,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "meddx_reference_budgets": MEDDX_REFERENCE_BUDGETS,
    "live_budgets_to_run": LIVE_BUDGETS_TO_RUN,
    "active_budgets": active_budgets,
    "enabled_datasets": ENABLED_DATASETS,
    "live_total_max_cases": LIVE_TOTAL_MAX_CASES,
    "dry_run_total_max_cases": DRY_RUN_TOTAL_MAX_CASES,
    "effective_total_case_cap": effective_total_case_cap,
    "meddxagent_data_root": str(MEDDXAGENT_DATA_ROOT),
    "icraft_md_path": str(ICRAFT_MD_PATH),
    "rarebench_mapping_dir": str(RAREBENCH_MAPPING_DIR),
    "rarebench_data_zip_path": str(RAREBENCH_DATA_ZIP_PATH),
    "rarebench_subsets": RAREBENCH_SUBSETS,
    "rarebench_data_zip_url": RAREBENCH_DATA_ZIP_URL,
    "candidate_text_max_chars": CANDIDATE_TEXT_MAX_CHARS,
    "driver_architecture": {
        "name": "universal_branching_resolver_meddx_driver_v1",
        "phases": [
            "history_taking",
            "deterministic_patient_simulator",
            "cap_aware_stop_probe",
            "ddxplus_partial_mlp_monitor_when_valid",
            "hypothesis_forced_branching_with_unused_budget",
            "candidate_pool_resolver",
            "conservative_rarebench_graph_gate",
        ],
        "history_policy": "one natural-language question per step, with broad first-turn inventory preference and cap-aware early stop probes",
        "diagnosis_policy": "base final diagnosis plus candidate-pool resolver over base, branch, casebase, and graph/HPO support",
        "meddxagent_alignment": "shared patient schema plus fixed MEDDx budgets, adapted with our DDXPlus stopping/branch/resolver architecture under the same cap",
    },
    "cap_aware_stopping": {
        "enabled": bool(ENABLE_CAP_AWARE_EARLY_STOP),
        "min_questions_by_budget": EARLY_STOP_MIN_QUESTIONS_BY_BUDGET,
        "universal_confidence_min": EARLY_STOP_CONFIDENCE_MIN,
        "universal_rank_margin_min": EARLY_STOP_RANK_MARGIN_MIN,
        "universal_stability_min": EARLY_STOP_STABILITY_MIN,
        "ddxplus_partial_mlp_monitor_enabled": bool(ENABLE_DDXPLUS_PARTIAL_MLP_MONITOR),
        "ddxplus_partial_mlp_available": bool(DDXPLUS_MLP_MONITOR.get("available", False)),
        "ddxplus_partial_mlp_reason": DDXPLUS_MLP_MONITOR.get("reason", ""),
        "ddxplus_selected_policy": {
            "mlp_confidence_min": EARLY_STOP_DDXPLUS_MLP_CONFIDENCE_MIN,
            "mlp_margin_min": EARLY_STOP_DDXPLUS_MLP_MARGIN_MIN,
            "mlp_entropy_max": EARLY_STOP_DDXPLUS_MLP_ENTROPY_MAX,
        },
    },
    "hypothesis_branching": {
        "enabled": bool(ENABLE_HYPOTHESIS_BRANCHING),
        "max_branches": int(BRANCH_MAX_BRANCHES),
        "max_questions_per_branch": int(BRANCH_MAX_QUESTIONS_PER_BRANCH),
        "min_remaining_budget": int(BRANCH_MIN_REMAINING_BUDGET),
        "total_question_cap_policy": "base plus branch questions may not exceed the active MEDDx budget",
        "trigger_confidence_max": BRANCH_TRIGGER_CONFIDENCE_MAX,
        "trigger_margin_max": BRANCH_TRIGGER_MARGIN_MAX,
        "trigger_disagreement_min": BRANCH_TRIGGER_DISAGREEMENT_MIN,
    },
    "candidate_pool_resolver": {
        "base_protection_margin": RESOLVER_BASE_PROTECTION_MARGIN,
        "min_independent_support_to_override": RESOLVER_MIN_INDEPENDENT_SUPPORT_TO_OVERRIDE,
        "sources": ["base_llm_rank", "branch_rank", "casebase_prior", "rarebench_graph"],
    },
    "casebase_prior": {
        "enabled": bool(ENABLE_CASEBASE_PRIOR),
        "reference_max_cases_per_dataset": CASEBASE_REFERENCE_MAX_CASES_PER_DATASET,
        "context_labels": CASEBASE_PRIOR_CONTEXT_LABELS,
        "rerank_labels": CASEBASE_PRIOR_RERANK_LABELS,
        "min_candidates": CASEBASE_PRIOR_MIN_CANDIDATES,
        "prior_weight": CASEBASE_PRIOR_WEIGHT,
        "min_normalized_score": CASEBASE_PRIOR_MIN_NORMALIZED_SCORE,
        "promotion_margin": CASEBASE_PRIOR_PROMOTION_MARGIN,
        "reference_counts": {dataset_name: len(refs) for dataset_name, refs in CASEBASE_REFERENCES.items()},
        "policy": "visible-evidence Jaccard casebase prior plus margin-gated LLM rank fusion",
    },
    "rarebench_graph_phenotype_resolver": {
        "enabled": bool(ENABLE_RAREBENCH_GRAPH_PHENOTYPE_RESOLVER),
        "reference_count": len(RAREBENCH_PHENOTYPE_REFERENCES),
        "context_labels": RAREBENCH_GRAPH_CONTEXT_LABELS,
        "rerank_labels": RAREBENCH_GRAPH_RERANK_LABELS,
        "min_visible_phenotypes": RAREBENCH_GRAPH_MIN_VISIBLE_PHENOTYPES,
        "prior_weight": RAREBENCH_GRAPH_PRIOR_WEIGHT,
        "llm_rank_weight": RAREBENCH_GRAPH_LLM_RANK_WEIGHT,
        "descriptive_label_penalty": RAREBENCH_GRAPH_DESCRIPTIVE_LABEL_PENALTY,
        "llm_discriminator": bool(RAREBENCH_LLM_DISCRIMINATOR),
        "conservative_graph_gate": bool(RAREBENCH_CONSERVATIVE_GRAPH_GATE),
        "lock_when_llm_and_graph_agree": bool(RAREBENCH_LOCK_WHEN_LLM_AND_GRAPH_AGREE),
        "graph_override_margin_min": RAREBENCH_GRAPH_OVERRIDE_MARGIN_MIN,
        "discriminator_third_option_margin_min": RAREBENCH_DISCRIMINATOR_THIRD_OPTION_MARGIN_MIN,
        "policy": "exact visible HPO phenotype nodes -> leave-one-case-out disease exemplar support -> conservative rank fusion/adjudication",
    },
    "broad_inventory_first_turn": bool(BROAD_INVENTORY_FIRST_TURN),
    "adapter_status": adapter_preflight.to_dict(orient="records"),
    "universal_schema": list(UniversalCase.__dataclass_fields__.keys()),
    "patient_simulator": {
        "name": "retrieval_patient_simulator_v1",
        "answer_policy": "answer only from hidden_full_profile spans; otherwise report not mentioned",
        "max_spans": patient_simulator.max_spans,
        "min_overlap": patient_simulator.min_overlap,
        "skip_previously_seen_spans": True,
    },
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

# %% [markdown]
# ## 11. Figures And Artifact Contract

# %%
if len(metrics_summary):
    plt.figure(figsize=(7, 4))
    for dataset_name, group in metrics_summary.groupby("dataset_name"):
        plt.plot(group["budget"], group["gtpa_at_1"], marker="o", label=dataset_name)
    plt.ylim(0, 1.05)
    plt.xlabel("Question budget")
    plt.ylabel("GTPA@1 / top-1")
    plt.title("Unified MEDDx-style hybrid accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gtpa_at_1_by_budget.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for dataset_name, group in metrics_summary.groupby("dataset_name"):
        plt.plot(group["budget"], group["mean_questions"], marker="o", label=dataset_name)
    plt.xlabel("Question budget")
    plt.ylabel("Mean questions asked")
    plt.title("Question use by budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mean_questions_by_budget.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    width = 0.25
    x = np.arange(len(metrics_summary))
    plt.bar(x - width, metrics_summary["gtpa_at_1"], width=width, label="GTPA@1")
    plt.bar(x, metrics_summary["gtpa_at_3"], width=width, label="GTPA@3")
    plt.bar(x + width, metrics_summary["gtpa_at_5"], width=width, label="GTPA@5")
    labels = [f"{row.dataset_name}\nB{int(row.budget)}" for row in metrics_summary.itertuples(index=False)]
    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 1.05)
    plt.ylabel("Hit rate")
    plt.title("Top-k diagnosis quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "topk_quality_by_dataset_budget.png", dpi=180)
    plt.close()

    if "rarebench_graph_visible_phenotypes" in predictions.columns:
        graph_frame = predictions[predictions["dataset_name"].astype(str).eq("rarebench")].copy()
        if len(graph_frame):
            plt.figure(figsize=(7, 4))
            plt.plot(graph_frame["budget"], graph_frame["rarebench_graph_visible_phenotypes"], marker="o", label="Visible phenotype nodes")
            plt.plot(graph_frame["budget"], graph_frame["rarebench_graph_top_score"], marker="o", label="Top graph support")
            plt.xlabel("Question budget")
            plt.title("RareBench graph-phenotype evidence")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "rarebench_graph_phenotype_support.png", dpi=180)
            plt.close()

    if "branch_count" in predictions.columns:
        plt.figure(figsize=(7, 4))
        branch_summary = predictions.groupby(["dataset_name", "budget"], as_index=False).agg(
            branch_rate=("branch_triggered", lambda s: s.astype(bool).mean()),
            mean_branch_questions=("branch_question_count", "mean"),
        )
        for dataset_name, group in branch_summary.groupby("dataset_name"):
            plt.plot(group["budget"], group["branch_rate"], marker="o", label=dataset_name)
        plt.ylim(0, 1.05)
        plt.xlabel("Question budget")
        plt.ylabel("Branch trigger rate")
        plt.title("Hypothesis branching under MEDDx caps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "branch_trigger_rate_by_budget.png", dpi=180)
        plt.close()

    if "resolver_margin" in predictions.columns:
        plt.figure(figsize=(7, 4))
        for dataset_name, group in predictions.groupby("dataset_name"):
            plt.scatter(group["resolver_margin"], group["correct_top1"].astype(int), label=dataset_name, alpha=0.75)
        plt.xlabel("Resolver margin")
        plt.ylabel("Correct top-1")
        plt.yticks([0, 1], ["wrong", "correct"])
        plt.title("Resolver margin vs correctness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "resolver_margin_vs_correctness.png", dpi=180)
        plt.close()

required_artifacts = [
    "resolved_run_config.json",
    "adapter_preflight.csv",
    "universal_cases.csv",
    "casebase_prior_reference_summary.csv",
    "rarebench_graph_phenotype_reference_summary.csv",
    "meddx_style_metrics_summary.csv",
]
if len(predictions):
    required_artifacts += [
        "predictions.csv",
        "question_answer_ledger.csv",
        "interaction_traces.jsonl",
        "patient_simulator_retrieval_audit.csv",
        "candidate_level_resolver_scores.csv",
    ]
    if predictions.get("branch_triggered", pd.Series(dtype=bool)).astype(bool).any():
        required_artifacts.append("branch_case_results.csv")
missing = [name for name in required_artifacts if not (ARTIFACT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing required artifacts: {missing}")
print("Notebook 45 artifact contract OK")
print("Artifact root:", ARTIFACT_ROOT)
