from __future__ import annotations

# %% [markdown]
# # Notebook 42: Universal MEDDx Benchmark Adapter
#
# This notebook starts the cross-dataset phase of the project.
#
# The goal is to keep the diagnostic workup architecture, but remove the hard dependency on DDXPlus evidence-root actions.
# Instead of asking for dataset-specific fields like `E_201`, the diagnostic agent asks natural-language clinical questions.
# A guarded patient simulator answers only from the hidden patient profile supplied by the active dataset adapter.
#
# Version 3 supports:
#
# - DDXPlus immediately, by converting structured evidence rows into hidden patient profiles
# - iCraft-MD through the MEDDxAgent benchmark JSONL format
# - RareBench through the MEDDxAgent mapping files plus the public HuggingFace RareBench data zip
#
# This is a harness and adapter notebook, not a final accuracy claim.

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

RUN_VERSION_SUFFIX = "v6_pilot3"
RANDOM_SEED = 4242

# MEDDx-style budgets. Active config is a small combined live pilot before the full v6 run.
# For the full run after pilot validation, use RUN_VERSION_SUFFIX = "v6" and LIVE_TOTAL_MAX_CASES = 49.
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

# Universal v6 repair layer:
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

# Generic adapter column defaults. Override these if the external files use different names.
GENERIC_CASE_ID_COLUMNS = ["case_id", "id", "patient_id", "uid"]
GENERIC_INITIAL_INFO_COLUMNS = ["initial_patient_info", "initial_info", "chief_complaint", "presentation", "initial"]
GENERIC_FULL_PROFILE_COLUMNS = ["full_patient_profile", "full_profile", "patient_profile", "profile", "case_text", "text", "description"]
GENERIC_DIAGNOSIS_COLUMNS = ["ground_truth_diagnosis", "diagnosis", "label", "disease", "pathology"]
GENERIC_CANDIDATE_COLUMNS = ["candidate_disease_list", "candidate_diseases", "disease_list", "possible_diseases"]

RUN_NAME_BASE = f"universal_meddx_benchmark_adapter_{RUN_VERSION_SUFFIX}"
RUN_NAME = RUN_NAME_BASE if RUN_LIVE_API else f"universal_meddx_benchmark_adapter_dryrun_smoke_{RUN_VERSION_SUFFIX}"

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
    turns: list[WorkupTurn]
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
            candidate_list = all_conditions
            metadata = {
                "age": row.get("AGE"),
                "sex": row.get("SEX"),
                "official_differential": official_diff[:10],
                "initial_evidence": row.get("INITIAL_EVIDENCE"),
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
            "Notebook 42 is configured to test all enabled datasets, but not all adapters loaded. "
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
# ## 4. Guarded Patient Simulator

# %%
class RetrievalPatientSimulator:
    def __init__(self, min_overlap: int = 1, max_spans: int = 3):
        self.min_overlap = min_overlap
        self.max_spans = max_spans

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
            # Generic "tell me more" questions should still reveal broad profile context.
            if any(token in question.lower() for token in ["additional", "other", "more", "profile", "symptoms", "findings"]):
                score += 0.05 if idx < 10 else 0.0
            scored.append({"span_index": idx, "span": span, "overlap": overlap, "score": score})
        scored.sort(key=lambda row: (row["score"], row["overlap"], -row["span_index"]), reverse=True)
        kept = [row for row in scored if row["overlap"] >= self.min_overlap or row["score"] > 0.04]
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
# ## 5. Universal LLM Diagnostic Agent

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


def build_agent_messages(case: UniversalCase, turns: list[WorkupTurn], budget: int, force_stop: bool = False) -> list[dict[str, str]]:
    remaining = max(budget - len(turns), 0)
    system = (
        "You are a differential-diagnosis workup agent. Ask one concise clinical question at a time, "
        "or stop with a ranked differential diagnosis. You cannot inspect hidden patient data directly. "
        "Use only the initial patient information and answered questions. Return valid JSON only."
    )
    decision_rule = (
        "You must stop now because the question budget is exhausted."
        if force_stop
        else "Ask a question if it is likely to distinguish the top competing diagnoses; otherwise stop."
    )
    first_turn_instruction = ""
    if BROAD_INVENTORY_FIRST_TURN and not turns and not force_stop:
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

Decision rule:
{decision_rule}
{first_turn_instruction}

Return JSON with exactly these keys:
{{
  "decision": "request | stop",
  "question": "single natural-language clinical question if requesting, otherwise empty string",
  "predicted_diagnosis": "best diagnosis from the candidate list",
  "ranked_differential": ["up to 10 candidate diagnoses, best first"],
  "confidence": 0.0,
  "brief_reasoning": "one short sentence"
}}

Rules:
- Do not ask for a dataset field id.
- Do not ask multiple questions at once.
- Do not repeat or paraphrase an already asked question.
- Keep questions clinically meaningful and answerable from a patient profile.
- Use exact disease names from the candidate list when possible.
- Candidate-list order is arbitrary; do not prefer a diagnosis because it appears earlier.
- If uncertain, keep the correct-looking alternatives in the ranked differential rather than overcommitting.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def scripted_dry_run_response(case: UniversalCase, turns: list[WorkupTurn], budget: int, force_stop: bool = False) -> dict[str, Any]:
    candidates = case.candidate_disease_list[:10] if case.candidate_disease_list else [case.ground_truth_diagnosis]
    predicted = candidates[0] if candidates else "Unknown"
    if force_stop or len(turns) >= min(2, budget):
        return {
            "decision": "stop",
            "question": "",
            "predicted_diagnosis": predicted,
            "ranked_differential": candidates[:10],
            "confidence": 0.35,
            "brief_reasoning": "Dry-run scripted stop for artifact validation.",
        }
    question_bank = [
        "What additional symptoms or findings are reported in the patient profile?",
        "What timing, duration, exposures, or context are reported?",
        "Are there any exam, lab, imaging, skin, or rare-disease clues mentioned?",
    ]
    return {
        "decision": "request",
        "question": question_bank[min(len(turns), len(question_bank) - 1)],
        "predicted_diagnosis": predicted,
        "ranked_differential": candidates[:10],
        "confidence": 0.20,
        "brief_reasoning": "Dry-run scripted request for artifact validation.",
    }


def get_agent_response(case: UniversalCase, turns: list[WorkupTurn], budget: int, force_stop: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if not RUN_LIVE_API:
        response = scripted_dry_run_response(case, turns, budget, force_stop=force_stop)
        return response, {"dry_run": True, "response": response}, {"input_tokens": 0, "output_tokens": 0}
    messages = build_agent_messages(case, turns, budget, force_stop=force_stop)
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
# ## 6. Workup Execution

# %%
def run_case(case: UniversalCase, budget: int) -> WorkupResult:
    turns: list[WorkupTurn] = []
    raw_responses: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    api_calls = 0
    initial_ranked: list[str] = []
    final_response: dict[str, Any] | None = None
    simulator_audits: list[dict[str, Any]] = []

    for turn_index in range(1, budget + 1):
        response, raw_payload, usage = get_agent_response(case, turns, budget, force_stop=False)
        response = canonicalize_agent_response(response, case.candidate_disease_list)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        api_calls += 0 if not RUN_LIVE_API else 1
        raw_responses.append({"turn_index": turn_index, "response": response, "raw_payload": raw_payload})
        if not initial_ranked and response.get("ranked_differential"):
            initial_ranked = list(response["ranked_differential"])
        final_response = response
        if response["decision"] == "stop" or not response.get("question"):
            break
        question = response["question"]
        seen_spans = {span for turn in turns for span in turn.retrieved_spans}
        answer, spans, audit = patient_simulator.answer(case, question, seen_spans=seen_spans)
        simulator_audits.append(audit)
        turns.append(
            WorkupTurn(
                turn_index=turn_index,
                question=question,
                answer=answer,
                retrieved_spans=spans,
                decision_before_answer=response["decision"],
                predicted_before_answer=response["predicted_diagnosis"],
                ranked_before_answer=response["ranked_differential"],
            )
        )

    if final_response is None or final_response.get("decision") == "request":
        response, raw_payload, usage = get_agent_response(case, turns, budget, force_stop=True)
        response = canonicalize_agent_response(response, case.candidate_disease_list)
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        api_calls += 0 if not RUN_LIVE_API else 1
        raw_responses.append({"turn_index": len(turns) + 1, "response": response, "raw_payload": raw_payload, "forced_final": True})
        final_response = response

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
    true_rank_final = rank_true_label(ranked, case.ground_truth_diagnosis)
    true_rank_initial = rank_true_label(initial_ranked or ranked, case.ground_truth_diagnosis)
    return WorkupResult(
        case_id=case.case_id,
        dataset_name=case.dataset_name,
        budget=budget,
        predicted_diagnosis=predicted,
        ranked_differential=ranked[:10],
        confidence=float(final_response.get("confidence", 0.0)),
        num_questions=len(turns),
        stopped_early=len(turns) < budget,
        stop_reason="agent_stop" if len(turns) < budget else "budget_exhausted_forced_final",
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
        turns=turns,
        raw_responses=raw_responses,
    )


predictions_path = ARTIFACT_ROOT / "predictions.csv"
ledger_path = ARTIFACT_ROOT / "question_answer_ledger.csv"
trace_path = ARTIFACT_ROOT / "interaction_traces.jsonl"
simulator_audit_path = ARTIFACT_ROOT / "patient_simulator_retrieval_audit.csv"

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

if not universal_cases:
    print("No dataset adapters produced cases. Supply external paths or check DDXPlus files.")
elif not RUN_LIVE_API and not ALLOW_DRY_RUN_BENCHMARK:
    print("Benchmark execution skipped. Set RUN_LIVE_API=True or ALLOW_DRY_RUN_BENCHMARK=True.")
else:
    tasks = [(case, budget) for case in universal_cases for budget in active_budgets]
    print("Active budgets      :", active_budgets)
    print("Total workups queued:", len(tasks), f"({len(universal_cases)} cases x {len(active_budgets)} budgets)")
    for case, budget in tqdm(tasks, desc="Universal MEDDx workups"):
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

predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
display(predictions.head())

# %% [markdown]
# ## 7. MEDDx-Style Evaluation

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
# ## 8. Figures And Artifact Contract

# %%
if len(metrics_summary):
    plt.figure(figsize=(7, 4))
    for dataset_name, group in metrics_summary.groupby("dataset_name"):
        plt.plot(group["budget"], group["gtpa_at_1"], marker="o", label=dataset_name)
    plt.ylim(0, 1.05)
    plt.xlabel("Question budget")
    plt.ylabel("GTPA@1 / top-1")
    plt.title("Universal MEDDx adapter accuracy")
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

required_artifacts = [
    "resolved_run_config.json",
    "adapter_preflight.csv",
    "universal_cases.csv",
    "casebase_prior_reference_summary.csv",
    "meddx_style_metrics_summary.csv",
]
if len(predictions):
    required_artifacts += [
        "predictions.csv",
        "question_answer_ledger.csv",
        "interaction_traces.jsonl",
        "patient_simulator_retrieval_audit.csv",
    ]
missing = [name for name in required_artifacts if not (ARTIFACT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing required artifacts: {missing}")
print("Notebook 42 artifact contract OK")
print("Artifact root:", ARTIFACT_ROOT)
