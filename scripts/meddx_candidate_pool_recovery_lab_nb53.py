from __future__ import annotations

# %% [markdown]
# # Notebook 53: MEDDx Candidate-Pool Recovery Lab
#
# Notebook 52 established that the current MEDDx-scale system is capped by candidate-pool formation:
# the final resolver can only choose the truth in `809/900` saved workups because the truth is absent
# from the candidate pool in `91/900` workups.
#
# This notebook is an offline-only Stage 1 lab. It keeps the Notebook 51 live traces frozen and asks:
#
# > Can label-free, dataset-native candidate expansion raise candidate-pool recall enough to make a
# > stronger resolver worth building?
#
# The selected Stage 1 policy is intentionally simple and deployable at final-resolution time:
#
# ```text
# recovered_candidate_pool_v1 =
#   current resolver pool
#   + saved final ranked differential top-10
#   + saved raw LLM ranked differential top-10
#   + DDXPlus MLP top-5 when available
#   + branch ranked differentials top-10
#   + casebase prior top label
#   + RareBench graph top label
#   + DDXPlus visible-evidence Bayes top-10
#   + RareBench visible-HPO exemplar top-10
# ```
#
# It makes no API calls and uses only evidence visible in the Notebook 51 artifacts.

# %%
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

pd.set_option("display.max_columns", 140)
pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 160)

ROOT = next(
    (
        candidate
        for candidate in [Path.cwd(), *Path.cwd().parents]
        if (candidate / "notebooks").exists() and (candidate / "reports").exists()
    ),
    Path.cwd(),
)

SCALE_RUN_NAME = "meddx_scale_hypothesis_branching_confirmation_v1_meddx100"
TRANSFER_RUN_NAME = "meddx_aligned_dataset_native_driver_v1_eval30"
RUN_NAME = "meddx_candidate_pool_recovery_lab_v1"

SCALE_ROOT = ROOT / "artifacts" / "universal_meddx" / SCALE_RUN_NAME
TRANSFER_ROOT = ROOT / "artifacts" / "universal_meddx" / TRANSFER_RUN_NAME
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DDXPLUS_BAYES_ROOT = ROOT / "artifacts" / "bayesian_voi_ledger" / "bayesian_voi_offline_notebook13_49case_v1"
RAREBENCH_ZIP = ROOT / "artifacts" / "universal_meddx" / "cache" / "rarebench_data.zip"
RAREBENCH_MAPPING_DIR = ROOT / "external" / "meddxagent" / "ddxdriver" / "benchmarks" / "data" / "rarebench"

SPLIT_SEED = 53
TRAIN_FRACTION = 0.60
VALIDATE_FRACTION = 0.20
TEST_FRACTION = 0.20

SELECTED_POLICY_NAME = "saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1"
SELECTED_SOURCE_ORDER = [
    "current_pool",
    "ranked_diff_top10",
    "llm_diff_top10",
    "ddxplus_mlp_top5",
    "branch_top10",
    "casebase_prior_top1",
    "rarebench_graph_top1",
    "ddxplus_visible_bayes_top10",
    "rarebench_visible_hpo_top10",
]

print("Project root :", ROOT)
print("Scale input  :", SCALE_ROOT)
print("Transfer set :", TRANSFER_ROOT)
print("Artifact root:", ARTIFACT_ROOT)

# %% [markdown]
# ## 1. Utility Functions

# %%
def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_label(value: Any) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("é", "e")
        .replace("è", "e")
        .replace("á", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ï", "i")
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def root_from_token(token: Any) -> str:
    return str(token or "").split("_@_")[0]


def outcome_from_token(token: Any) -> str:
    text = str(token or "")
    if "_@_" not in text:
        return "__PRESENT__"
    return text.split("_@_", 1)[1]


def extract_root_from_question(question: Any) -> str | None:
    match = re.search(r"\[(E_\d+)\]", str(question or ""))
    return match.group(1) if match else None


def outcome_from_ddxplus_answer(answer: Any) -> str | None:
    text = str(answer or "").lower()
    if "no / not reported" in text or "does not mention" in text or "answer: no" in text or "answer: n." in text:
        return "__ABSENT__"
    if "answer: present" in text or "answer: yes" in text or "answer: y." in text:
        return "__PRESENT__"
    return None


def sort_outcome_parts(parts: list[str]) -> str:
    def key(part: str) -> tuple[str, int | str]:
        if part.startswith("V_") and part.split("_", 1)[1].isdigit():
            return ("V", int(part.split("_", 1)[1]))
        if part.isdigit():
            return ("N", int(part))
        return ("Z", part)

    return "|".join(sorted(set(parts), key=key))


def add_unique(pool: list[dict[str, Any]], label: str, source_name: str, source_rank: int, score: float = 0.0) -> None:
    key = normalize_label(label)
    if not key:
        return
    for item in pool:
        if item["label_key"] == key:
            item["sources"].append(source_name)
            item["source_ranks"][source_name] = min(int(source_rank), int(item["source_ranks"].get(source_name, source_rank)))
            item["source_scores"][source_name] = max(float(score), float(item["source_scores"].get(source_name, score)))
            return
    pool.append(
        {
            "label": str(label),
            "label_key": key,
            "sources": [source_name],
            "source_ranks": {source_name: int(source_rank)},
            "source_scores": {source_name: float(score)},
        }
    )


def unique_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = normalize_label(label)
        if key and key not in seen:
            seen.add(key)
            out.append(str(label))
    return out


def rank_in_pool(truth: str, pool_labels: list[str], missing_rank: int = 999) -> int:
    truth_key = normalize_label(truth)
    for idx, label in enumerate(pool_labels, start=1):
        if normalize_label(label) == truth_key:
            return idx
    return missing_rank


def pool_has_truth(truth: str, pool_labels: list[str]) -> bool:
    truth_key = normalize_label(truth)
    return truth_key in {normalize_label(label) for label in pool_labels}


def score_current_ranked(row: pd.Series) -> list[str]:
    ranked = parse_json_list(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "").strip()
    if prediction and normalize_label(prediction) not in {normalize_label(label) for label in ranked}:
        ranked = [prediction, *ranked]
    return unique_labels(ranked)[:10]

# %% [markdown]
# ## 2. Load Frozen Notebook 51 / 46 Artifacts

# %%
REQUIRED_INPUTS = [
    SCALE_ROOT / "predictions.csv",
    SCALE_ROOT / "candidate_level_resolver_scores.csv",
    SCALE_ROOT / "question_answer_ledger.csv",
    SCALE_ROOT / "universal_cases.csv",
    TRANSFER_ROOT / "predictions.csv",
    TRANSFER_ROOT / "candidate_level_resolver_scores.csv",
    TRANSFER_ROOT / "question_answer_ledger.csv",
    TRANSFER_ROOT / "universal_cases.csv",
]
missing = [str(path) for path in REQUIRED_INPUTS if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")


def load_run(root: Path, cohort_name: str) -> dict[str, pd.DataFrame]:
    frames = {
        "predictions": pd.read_csv(root / "predictions.csv"),
        "candidates": pd.read_csv(root / "candidate_level_resolver_scores.csv"),
        "ledger": pd.read_csv(root / "question_answer_ledger.csv"),
        "cases": pd.read_csv(root / "universal_cases.csv"),
    }
    branch_path = root / "branch_case_results.csv"
    frames["branches"] = pd.read_csv(branch_path) if branch_path.exists() else pd.DataFrame()
    for frame_name, frame in frames.items():
        if len(frame):
            frame["cohort"] = cohort_name
    return frames


scale = load_run(SCALE_ROOT, "scale_meddx100")
transfer = load_run(TRANSFER_ROOT, "transfer_old90")

display(
    scale["predictions"].groupby(["dataset_name", "budget"], as_index=False).agg(
        n=("case_id", "size"),
        top1=("correct_top1", lambda s: s.map(boolish).mean()),
        top3=("gtpa_at_3", lambda s: s.map(boolish).mean()),
        top5=("gtpa_at_5", lambda s: s.map(boolish).mean()),
        mean_questions=("num_questions", "mean"),
    )
)

# %% [markdown]
# ## 3. Case-Blocked Split And Baseline Failure Observatory

# %%
def make_case_split(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SPLIT_SEED)
    rows: list[dict[str, Any]] = []
    for dataset_name, group in predictions[["dataset_name", "case_id"]].drop_duplicates().groupby("dataset_name", sort=True):
        case_ids = group["case_id"].astype(str).tolist()
        rng.shuffle(case_ids)
        n_cases = len(case_ids)
        n_train = int(round(TRAIN_FRACTION * n_cases))
        n_validate = int(round(VALIDATE_FRACTION * n_cases))
        train_ids = set(case_ids[:n_train])
        validate_ids = set(case_ids[n_train : n_train + n_validate])
        for case_id in case_ids:
            split = "train" if case_id in train_ids else "validate" if case_id in validate_ids else "test"
            rows.append({"dataset_name": str(dataset_name), "case_id": str(case_id), "split": split})
    return pd.DataFrame(rows)


case_split = make_case_split(scale["predictions"])
case_split.to_csv(ARTIFACT_ROOT / "case_split_assignment.csv", index=False)


def current_pool_map(candidates: pd.DataFrame) -> dict[tuple[str, str, int], list[str]]:
    pool_by_key: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    seen_by_key: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for _, row in candidates.sort_values(["dataset_name", "case_id", "budget", "candidate_rank"]).iterrows():
        key = (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]))
        label = str(row["label"])
        label_key = normalize_label(label)
        if label_key and label_key not in seen_by_key[key]:
            seen_by_key[key].add(label_key)
            pool_by_key[key].append(label)
    return dict(pool_by_key)


def baseline_failure_map(frames: dict[str, pd.DataFrame], cohort_name: str, split_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    predictions = frames["predictions"].copy()
    current = current_pool_map(frames["candidates"])
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        key = (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]))
        pool = current.get(key, [])
        truth = str(row["ground_truth_diagnosis"])
        current_ranked = score_current_ranked(row)
        rows.append(
            {
                "cohort": cohort_name,
                "dataset_name": key[0],
                "case_id": key[1],
                "budget": key[2],
                "ground_truth_diagnosis": truth,
                "current_prediction": row.get("predicted_diagnosis", ""),
                "current_correct_top1": boolish(row.get("correct_top1", False)),
                "current_gtpa_at_3": boolish(row.get("gtpa_at_3", False)),
                "current_gtpa_at_5": boolish(row.get("gtpa_at_5", False)),
                "current_ranked_has_truth_top10": pool_has_truth(truth, current_ranked),
                "current_candidate_pool_has_truth": pool_has_truth(truth, pool),
                "current_candidate_pool_size": len(pool),
                "current_candidate_pool_truth_rank": rank_in_pool(truth, pool),
                "branch_triggered": boolish(row.get("branch_triggered", False)),
                "branch_count": int(safe_float(row.get("branch_count", 0), 0)),
                "num_questions": safe_float(row.get("num_questions", np.nan), np.nan),
                "resolver_margin": safe_float(row.get("resolver_margin", 0.0), 0.0),
                "confidence": safe_float(row.get("confidence", 0.0), 0.0),
            }
        )
    out = pd.DataFrame(rows)
    if split_frame is not None:
        out = out.merge(split_frame, on=["dataset_name", "case_id"], how="left")
    else:
        out["split"] = "transfer"
    return out


scale_baseline = baseline_failure_map(scale, "scale_meddx100", case_split)
transfer_baseline = baseline_failure_map(transfer, "transfer_old90", None)

baseline_policy_summary = (
    scale_baseline.groupby(["cohort", "dataset_name", "budget", "split"], as_index=False)
    .agg(
        workups=("case_id", "size"),
        current_top1=("current_correct_top1", "sum"),
        current_top3=("current_gtpa_at_3", "sum"),
        current_top5=("current_gtpa_at_5", "sum"),
        current_pool_recall=("current_candidate_pool_has_truth", "sum"),
        mean_pool_size=("current_candidate_pool_size", "mean"),
        mean_questions=("num_questions", "mean"),
    )
)
baseline_policy_summary.to_csv(ARTIFACT_ROOT / "baseline_policy_summary.csv", index=False)
scale_baseline.to_csv(ARTIFACT_ROOT / "baseline_failure_map.csv", index=False)

candidate_pool_miss_audit = scale_baseline[~scale_baseline["current_candidate_pool_has_truth"]].copy()
candidate_pool_miss_audit.to_csv(ARTIFACT_ROOT / "candidate_pool_miss_audit.csv", index=False)
resolver_miss_audit = scale_baseline[
    scale_baseline["current_candidate_pool_has_truth"] & ~scale_baseline["current_correct_top1"]
].copy()
resolver_miss_audit.to_csv(ARTIFACT_ROOT / "resolver_miss_audit.csv", index=False)
questioning_audit = scale_baseline[
    [
        "dataset_name",
        "case_id",
        "budget",
        "split",
        "ground_truth_diagnosis",
        "current_candidate_pool_has_truth",
        "current_correct_top1",
        "branch_triggered",
        "branch_count",
        "num_questions",
        "resolver_margin",
        "confidence",
    ]
].copy()
questioning_audit.to_csv(ARTIFACT_ROOT / "questioning_audit.csv", index=False)

dataset_budget_failure_summary = (
    scale_baseline.groupby(["dataset_name", "budget"], as_index=False)
    .agg(
        workups=("case_id", "size"),
        current_top1=("current_correct_top1", "sum"),
        current_top3=("current_gtpa_at_3", "sum"),
        current_top5=("current_gtpa_at_5", "sum"),
        current_pool_recall=("current_candidate_pool_has_truth", "sum"),
        pool_misses=("current_candidate_pool_has_truth", lambda s: int((~s).sum())),
        resolver_misses=("current_correct_top1", lambda s: int((~s).sum())),
        branch_rate=("branch_triggered", "mean"),
        mean_questions=("num_questions", "mean"),
    )
)
dataset_budget_failure_summary.to_csv(ARTIFACT_ROOT / "dataset_budget_failure_summary.csv", index=False)

display(dataset_budget_failure_summary)

# %% [markdown]
# ## 4. Dataset-Native Candidate Sources

# %%
def load_ddxplus_bayes_tables() -> tuple[list[str], dict[str, float], dict[tuple[str, str, str], float]]:
    likelihood_path = DDXPLUS_BAYES_ROOT / "root_outcome_likelihoods.csv"
    prior_path = DDXPLUS_BAYES_ROOT / "diagnosis_priors.csv"
    if not likelihood_path.exists() or not prior_path.exists():
        return [], {}, {}
    likelihoods = pd.read_csv(likelihood_path)
    priors = pd.read_csv(prior_path)
    pathologies = [str(item) for item in priors["pathology"].tolist()]
    prior_lookup = {str(row["pathology"]): max(safe_float(row["prior"], 1e-12), 1e-12) for _, row in priors.iterrows()}
    likelihood_lookup: dict[tuple[str, str, str], float] = {}
    for _, row in likelihoods.iterrows():
        root_id = str(row["root_evidence_id"])
        outcome = str(row["outcome_state"])
        for pathology in pathologies:
            likelihood_lookup[(root_id, outcome, pathology)] = max(
                min(safe_float(row.get(f"p__{pathology}", 1e-9), 1e-9), 1 - 1e-9),
                1e-9,
            )
    return pathologies, prior_lookup, likelihood_lookup


DDXPLUS_PATHOLOGIES, DDXPLUS_PRIORS, DDXPLUS_LIKELIHOODS = load_ddxplus_bayes_tables()


def ddxplus_observed_roots(frames: dict[str, pd.DataFrame], case_id: str, budget: int) -> dict[str, str]:
    cases = frames["cases"]
    ledger = frames["ledger"]
    case_rows = cases[(cases["dataset_name"].astype(str) == "ddxplus") & (cases["case_id"].astype(str) == str(case_id))]
    metadata: dict[str, Any] = {}
    if len(case_rows):
        try:
            metadata = json.loads(str(case_rows.iloc[0].get("metadata", "{}")))
        except Exception:
            metadata = {}

    observed: dict[str, str] = {}
    initial = str(metadata.get("initial_evidence", "") or "")
    if initial.startswith("E_"):
        observed[root_from_token(initial)] = outcome_from_token(initial)

    span_token_map = metadata.get("span_token_map", {}) if isinstance(metadata.get("span_token_map", {}), dict) else {}
    workup_ledger = ledger[
        (ledger["dataset_name"].astype(str) == "ddxplus")
        & (ledger["case_id"].astype(str) == str(case_id))
        & (ledger["budget"].astype(int) == int(budget))
    ].sort_values("turn_index")
    for _, row in workup_ledger.iterrows():
        root_id = extract_root_from_question(row.get("question", ""))
        if not root_id:
            continue
        outcome = outcome_from_ddxplus_answer(row.get("answer", ""))
        exact_parts: list[str] = []
        for span in parse_json_list(row.get("retrieved_spans", "")):
            token = span_token_map.get(span)
            if token and root_from_token(token) == root_id:
                token_outcome = outcome_from_token(token)
                if token_outcome != "__PRESENT__":
                    exact_parts.extend(str(token_outcome).split("|"))
        if exact_parts:
            outcome = sort_outcome_parts(exact_parts)
        if outcome:
            observed[root_id] = outcome
    return observed


def ddxplus_bayes_ranked(frames: dict[str, pd.DataFrame], case_id: str, budget: int, limit: int = 10) -> list[tuple[str, float]]:
    if not DDXPLUS_PATHOLOGIES:
        return []
    observed = ddxplus_observed_roots(frames, case_id, budget)
    log_scores = {pathology: math.log(DDXPLUS_PRIORS.get(pathology, 1e-12)) for pathology in DDXPLUS_PATHOLOGIES}
    for root_id, outcome in observed.items():
        for pathology in DDXPLUS_PATHOLOGIES:
            log_scores[pathology] += math.log(DDXPLUS_LIKELIHOODS.get((root_id, outcome, pathology), 1e-6))
    return [(label, float(log_scores[label])) for label in sorted(DDXPLUS_PATHOLOGIES, key=lambda item: log_scores[item], reverse=True)[:limit]]


def canonicalize_rarebench_disease(
    subset: str,
    disease_codes: list[str],
    raw_disease_mapping: dict[str, str],
    subset_mapping: dict[str, dict[str, str]],
) -> str:
    raw_names = [raw_disease_mapping[code] for code in disease_codes if code in raw_disease_mapping]
    joined = ", ".join(raw_names)
    mapping = subset_mapping.get(subset, {})
    if joined in mapping:
        return str(mapping[joined])
    slash_joined = "/".join(raw_names)
    if slash_joined in mapping:
        return str(mapping[slash_joined])
    mapped_parts = [mapping.get(name, name.split("/")[0]) for name in raw_names]
    if not mapped_parts:
        return ""
    return str(Counter(mapped_parts).most_common(1)[0][0])


def load_rarebench_reference_records() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, float]]:
    if not RAREBENCH_ZIP.exists():
        return [], {}, {}
    phenotype_mapping = read_json(RAREBENCH_MAPPING_DIR / "rarebench_phenotype_mapping.json")
    raw_disease_mapping = read_json(RAREBENCH_MAPPING_DIR / "rarebench_disease_mapping.json")
    disease_mapping = read_json(RAREBENCH_MAPPING_DIR / "disease_mapping.json")

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(RAREBENCH_ZIP) as archive:
        for name in archive.namelist():
            if not name.startswith("data/") or not name.endswith(".jsonl"):
                continue
            subset = Path(name).stem
            with archive.open(name) as handle:
                for idx, raw_line in enumerate(handle):
                    item = json.loads(raw_line)
                    hpo_ids = {str(hpo) for hpo in item.get("Phenotype", [])}
                    disease_codes = [str(code) for code in item.get("RareDisease", [])]
                    canonical = canonicalize_rarebench_disease(subset, disease_codes, raw_disease_mapping, disease_mapping)
                    if canonical:
                        records.append(
                            {
                                "subset": subset,
                                "record_index": idx,
                                "label": canonical,
                                "label_key": normalize_label(canonical),
                                "hpo_ids": hpo_ids,
                            }
                        )
    document_frequency = Counter()
    for record in records:
        document_frequency.update(record["hpo_ids"])
    total = max(len(records), 1)
    idf = {hpo: math.log((1 + total) / (1 + count)) + 1.0 for hpo, count in document_frequency.items()}
    phenotype_name_to_hpo = {normalize_label(name): hpo for hpo, name in phenotype_mapping.items()}
    return records, phenotype_name_to_hpo, idf


RAREBENCH_RECORDS, RAREBENCH_PHENOTYPE_NAME_TO_HPO, RAREBENCH_HPO_IDF = load_rarebench_reference_records()
RAREBENCH_RECORDS_BY_LABEL: dict[str, list[dict[str, Any]]] = defaultdict(list)
for record in RAREBENCH_RECORDS:
    RAREBENCH_RECORDS_BY_LABEL[record["label_key"]].append(record)


def rarebench_case_ref(case_id: str) -> tuple[str, int] | None:
    parts = str(case_id).split(":")
    if len(parts) >= 3:
        try:
            return parts[1], int(parts[2])
        except Exception:
            return None
    return None


def rarebench_visible_hpos(frames: dict[str, pd.DataFrame], case_id: str, budget: int) -> set[str]:
    cases = frames["cases"]
    ledger = frames["ledger"]
    names: list[str] = []
    case_rows = cases[(cases["dataset_name"].astype(str) == "rarebench") & (cases["case_id"].astype(str) == str(case_id))]
    if len(case_rows):
        for line in str(case_rows.iloc[0].get("initial_patient_info", "")).splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                names.append(line)
    workup_ledger = ledger[
        (ledger["dataset_name"].astype(str) == "rarebench")
        & (ledger["case_id"].astype(str) == str(case_id))
        & (ledger["budget"].astype(int) == int(budget))
    ].sort_values("turn_index")
    for _, row in workup_ledger.iterrows():
        names.extend(parse_json_list(row.get("retrieved_spans", "")))
    return {
        RAREBENCH_PHENOTYPE_NAME_TO_HPO[normalize_label(name)]
        for name in names
        if normalize_label(name) in RAREBENCH_PHENOTYPE_NAME_TO_HPO
    }


def rarebench_hpo_ranked(frames: dict[str, pd.DataFrame], case_id: str, budget: int, limit: int = 10) -> list[tuple[str, float]]:
    if not RAREBENCH_RECORDS:
        return []
    cases = frames["cases"]
    case_rows = cases[(cases["dataset_name"].astype(str) == "rarebench") & (cases["case_id"].astype(str) == str(case_id))]
    if not len(case_rows):
        return []
    candidate_labels = parse_json_list(case_rows.iloc[0].get("candidate_disease_list", ""))
    visible_hpos = rarebench_visible_hpos(frames, case_id, budget)
    denominator = sum(RAREBENCH_HPO_IDF.get(hpo, 1.0) for hpo in visible_hpos) or 1.0
    current_ref = rarebench_case_ref(case_id)
    scored: list[tuple[float, float, int, str]] = []
    for label in candidate_labels:
        label_key = normalize_label(label)
        values: list[tuple[float, float, int]] = []
        for record in RAREBENCH_RECORDS_BY_LABEL.get(label_key, []):
            if current_ref and record["subset"] == current_ref[0] and int(record["record_index"]) == int(current_ref[1]):
                continue
            overlap = visible_hpos & record["hpo_ids"]
            union = visible_hpos | record["hpo_ids"]
            idf_fraction = sum(RAREBENCH_HPO_IDF.get(hpo, 1.0) for hpo in overlap) / denominator
            jaccard = len(overlap) / len(union) if union else 0.0
            values.append((float(idf_fraction), float(jaccard), int(len(overlap))))
        best = max(values) if values else (0.0, 0.0, 0)
        scored.append((best[0], best[1], best[2], label))
    scored.sort(reverse=True)
    return [(label, float(idf_fraction)) for idf_fraction, _jaccard, _overlap, label in scored[:limit]]


print("DDXPlus Bayes pathologies:", len(DDXPLUS_PATHOLOGIES))
print("RareBench reference records:", len(RAREBENCH_RECORDS))

# %% [markdown]
# ## 5. Build Expanded Candidate Pools

# %%
def candidate_sources_for_run(frames: dict[str, pd.DataFrame]) -> dict[str, dict[tuple[str, str, int], list[tuple[str, float]]]]:
    predictions = frames["predictions"]
    candidates = frames["candidates"]
    branches = frames["branches"]
    sources: dict[str, dict[tuple[str, str, int], list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

    current = current_pool_map(candidates)
    for key, labels in current.items():
        sources["current_pool"][key] = [(label, float(len(labels) - idx)) for idx, label in enumerate(labels)]

    for _, row in predictions.iterrows():
        key = (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]))
        for idx, label in enumerate(parse_json_list(row.get("ranked_differential", ""))[:10], start=1):
            sources["ranked_diff_top10"][key].append((label, float(11 - idx)))
        for idx, label in enumerate(parse_json_list(row.get("llm_ranked_differential", ""))[:10], start=1):
            sources["llm_diff_top10"][key].append((label, float(11 - idx)))
        for idx, label in enumerate(parse_json_list(row.get("ddxplus_mlp_top5", ""))[:5], start=1):
            sources["ddxplus_mlp_top5"][key].append((label, float(6 - idx)))
        prior_label = row.get("casebase_prior_top_label", "")
        if isinstance(prior_label, str) and prior_label.strip():
            sources["casebase_prior_top1"][key].append((prior_label, safe_float(row.get("casebase_prior_top_score", 0.0), 0.0)))
        rare_label = row.get("rarebench_graph_top_label", "")
        if isinstance(rare_label, str) and rare_label.strip():
            sources["rarebench_graph_top1"][key].append((rare_label, safe_float(row.get("rarebench_graph_top_score", 0.0), 0.0)))

        if key[0] == "ddxplus":
            sources["ddxplus_visible_bayes_top10"][key].extend(ddxplus_bayes_ranked(frames, key[1], key[2], limit=10))
        if key[0] == "rarebench":
            sources["rarebench_visible_hpo_top10"][key].extend(rarebench_hpo_ranked(frames, key[1], key[2], limit=10))

    if len(branches):
        for _, row in branches.iterrows():
            key = (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]))
            for idx, label in enumerate(parse_json_list(row.get("ranked_differential", ""))[:10], start=1):
                sources["branch_top10"][key].append((label, float(11 - idx)))
    return {name: dict(values) for name, values in sources.items()}


scale_sources = candidate_sources_for_run(scale)
transfer_sources = candidate_sources_for_run(transfer)

POLICY_VARIANTS = {
    "current_pool": ["current_pool"],
    "saved_sources_union": [
        "current_pool",
        "ranked_diff_top10",
        "llm_diff_top10",
        "ddxplus_mlp_top5",
        "branch_top10",
        "casebase_prior_top1",
        "rarebench_graph_top1",
    ],
    "saved_plus_ddx_bayes10": [
        "current_pool",
        "ranked_diff_top10",
        "llm_diff_top10",
        "ddxplus_mlp_top5",
        "branch_top10",
        "casebase_prior_top1",
        "rarebench_graph_top1",
        "ddxplus_visible_bayes_top10",
    ],
    "saved_plus_visible_rare_hpo10": [
        "current_pool",
        "ranked_diff_top10",
        "llm_diff_top10",
        "ddxplus_mlp_top5",
        "branch_top10",
        "casebase_prior_top1",
        "rarebench_graph_top1",
        "rarebench_visible_hpo_top10",
    ],
    SELECTED_POLICY_NAME: SELECTED_SOURCE_ORDER,
}


def materialize_pool(
    sources: dict[str, dict[tuple[str, str, int], list[tuple[str, float]]]],
    key: tuple[str, str, int],
    source_order: list[str],
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for source_name in source_order:
        labels = sources.get(source_name, {}).get(key, [])
        for idx, (label, score) in enumerate(labels, start=1):
            add_unique(pool, label, source_name, idx, score)
    return pool


def evaluate_pool_policy(
    frames: dict[str, pd.DataFrame],
    sources: dict[str, dict[tuple[str, str, int], list[tuple[str, float]]]],
    cohort_name: str,
    policy_name: str,
    source_order: list[str],
    split_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    current = current_pool_map(frames["candidates"])

    for _, row in frames["predictions"].iterrows():
        key = (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]))
        truth = str(row["ground_truth_diagnosis"])
        current_labels = current.get(key, [])
        pool = materialize_pool(sources, key, source_order)
        labels = [item["label"] for item in pool]
        truth_key = normalize_label(truth)
        current_has_truth = pool_has_truth(truth, current_labels)
        selected_has_truth = pool_has_truth(truth, labels)
        case_rows.append(
            {
                "cohort": cohort_name,
                "policy_name": policy_name,
                "dataset_name": key[0],
                "case_id": key[1],
                "budget": key[2],
                "ground_truth_diagnosis": truth,
                "current_prediction": row.get("predicted_diagnosis", ""),
                "current_correct_top1": boolish(row.get("correct_top1", False)),
                "current_gtpa_at_3": boolish(row.get("gtpa_at_3", False)),
                "current_gtpa_at_5": boolish(row.get("gtpa_at_5", False)),
                "current_candidate_pool_has_truth": current_has_truth,
                "candidate_pool_has_truth": selected_has_truth,
                "recovered_truth_vs_current_pool": bool((not current_has_truth) and selected_has_truth),
                "lost_truth_vs_current_pool": bool(current_has_truth and not selected_has_truth),
                "current_candidate_pool_size": len(current_labels),
                "candidate_pool_size": len(labels),
                "pool_size_delta": len(labels) - len(current_labels),
                "candidate_pool_truth_rank": rank_in_pool(truth, labels),
                "candidate_pool_labels": json.dumps(labels, ensure_ascii=True),
                "num_questions": safe_float(row.get("num_questions", np.nan), np.nan),
                "branch_triggered": boolish(row.get("branch_triggered", False)),
                "branch_count": int(safe_float(row.get("branch_count", 0), 0)),
            }
        )
        for idx, item in enumerate(pool, start=1):
            candidate_rows.append(
                {
                    "cohort": cohort_name,
                    "policy_name": policy_name,
                    "dataset_name": key[0],
                    "case_id": key[1],
                    "budget": key[2],
                    "candidate_rank_added_order": idx,
                    "label": item["label"],
                    "label_key": item["label_key"],
                    "ground_truth_diagnosis": truth,
                    "is_truth_candidate": int(item["label_key"] == truth_key),
                    "sources": json.dumps(item["sources"], ensure_ascii=True),
                    "source_ranks": json.dumps(item["source_ranks"], ensure_ascii=True),
                    "source_scores": json.dumps(item["source_scores"], ensure_ascii=True),
                    **{f"has_source__{source}": int(source in item["sources"]) for source in SELECTED_SOURCE_ORDER},
                }
            )

    case_frame = pd.DataFrame(case_rows)
    if split_frame is not None:
        case_frame = case_frame.merge(split_frame, on=["dataset_name", "case_id"], how="left")
    else:
        case_frame["split"] = "transfer"
    candidate_frame = pd.DataFrame(candidate_rows)
    if split_frame is not None and len(candidate_frame):
        candidate_frame = candidate_frame.merge(split_frame, on=["dataset_name", "case_id"], how="left")
    elif len(candidate_frame):
        candidate_frame["split"] = "transfer"
    return case_frame, candidate_frame


all_case_results: list[pd.DataFrame] = []
all_candidate_results: list[pd.DataFrame] = []
for policy_name, source_order in POLICY_VARIANTS.items():
    case_frame, candidate_frame = evaluate_pool_policy(scale, scale_sources, "scale_meddx100", policy_name, source_order, case_split)
    all_case_results.append(case_frame)
    all_candidate_results.append(candidate_frame)
    transfer_case_frame, transfer_candidate_frame = evaluate_pool_policy(
        transfer,
        transfer_sources,
        "transfer_old90",
        policy_name,
        source_order,
        None,
    )
    all_case_results.append(transfer_case_frame)
    all_candidate_results.append(transfer_candidate_frame)

case_level_results = pd.concat(all_case_results, ignore_index=True)
candidate_level_results = pd.concat(all_candidate_results, ignore_index=True)

case_level_results.to_csv(ARTIFACT_ROOT / "case_level_candidate_pool_recovery_results.csv", index=False)
candidate_level_results.to_csv(ARTIFACT_ROOT / "expanded_candidate_pool_long.csv", index=False)

display(case_level_results[case_level_results["policy_name"].eq(SELECTED_POLICY_NAME)].head())

# %% [markdown]
# ## 6. Candidate-Pool Recovery Evaluation

# %%
def summarize_pool(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slices = [("ALL", -1, "ALL", frame)]
    slices.extend(
        (dataset_name, int(budget), "ALL", group)
        for (dataset_name, budget), group in frame.groupby(["dataset_name", "budget"], sort=True)
    )
    if "split" in frame:
        slices.extend(
            ("ALL", -1, split, group)
            for split, group in frame.groupby("split", sort=True)
        )
        slices.extend(
            (dataset_name, int(budget), split, group)
            for (dataset_name, budget, split), group in frame.groupby(["dataset_name", "budget", "split"], sort=True)
        )
    for dataset_name, budget, split, group in slices:
        if not len(group):
            continue
        rows.append(
            {
                "cohort": str(group["cohort"].iloc[0]),
                "policy_name": str(group["policy_name"].iloc[0]),
                "dataset_name": dataset_name,
                "budget": int(budget),
                "split": split,
                "num_workups": int(len(group)),
                "candidate_pool_recall_count": int(group["candidate_pool_has_truth"].sum()),
                "candidate_pool_recall": float(group["candidate_pool_has_truth"].mean()),
                "current_pool_recall_count": int(group["current_candidate_pool_has_truth"].sum()),
                "current_pool_recall": float(group["current_candidate_pool_has_truth"].mean()),
                "recoveries_vs_current_pool": int(group["recovered_truth_vs_current_pool"].sum()),
                "lost_truth_vs_current_pool": int(group["lost_truth_vs_current_pool"].sum()),
                "mean_pool_size": float(group["candidate_pool_size"].mean()),
                "median_pool_size": float(group["candidate_pool_size"].median()),
                "p90_pool_size": float(group["candidate_pool_size"].quantile(0.90)),
                "mean_pool_size_delta": float(group["pool_size_delta"].mean()),
                "mean_questions": float(group["num_questions"].mean()),
            }
        )
    return pd.DataFrame(rows)


candidate_pool_recovery_summary = pd.concat(
    [summarize_pool(group) for _, group in case_level_results.groupby(["cohort", "policy_name"], sort=True)],
    ignore_index=True,
)
candidate_pool_recovery_summary.to_csv(ARTIFACT_ROOT / "candidate_pool_recovery_summary.csv", index=False)

selected_scale = case_level_results[
    (case_level_results["cohort"].eq("scale_meddx100")) & (case_level_results["policy_name"].eq(SELECTED_POLICY_NAME))
].copy()
selected_transfer = case_level_results[
    (case_level_results["cohort"].eq("transfer_old90")) & (case_level_results["policy_name"].eq(SELECTED_POLICY_NAME))
].copy()
paired_current_vs_candidate = selected_scale[
    [
        "cohort",
        "policy_name",
        "dataset_name",
        "case_id",
        "budget",
        "split",
        "ground_truth_diagnosis",
        "current_candidate_pool_has_truth",
        "candidate_pool_has_truth",
        "recovered_truth_vs_current_pool",
        "lost_truth_vs_current_pool",
        "current_candidate_pool_size",
        "candidate_pool_size",
        "pool_size_delta",
        "candidate_pool_truth_rank",
        "num_questions",
    ]
].copy()
paired_current_vs_candidate.to_csv(ARTIFACT_ROOT / "paired_current_vs_candidate.csv", index=False)

display(
    candidate_pool_recovery_summary[
        candidate_pool_recovery_summary["dataset_name"].eq("ALL")
        & candidate_pool_recovery_summary["budget"].eq(-1)
        & candidate_pool_recovery_summary["split"].eq("ALL")
    ].sort_values(["cohort", "candidate_pool_recall_count"], ascending=[True, False])
)

display(
    candidate_pool_recovery_summary[
        candidate_pool_recovery_summary["policy_name"].eq(SELECTED_POLICY_NAME)
        & candidate_pool_recovery_summary["dataset_name"].ne("ALL")
        & candidate_pool_recovery_summary["split"].eq("ALL")
    ].sort_values(["cohort", "dataset_name", "budget"])
)

# %% [markdown]
# ## 7. Source Contribution And Risk Diagnostics

# %%
def source_contribution_rows(
    frames: dict[str, pd.DataFrame],
    sources: dict[str, dict[tuple[str, str, int], list[tuple[str, float]]]],
    source_order: list[str],
    cohort_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_order: list[str] = []
    previous_has: dict[tuple[str, str, int], bool] = {}
    predictions = frames["predictions"]
    for step_index, source in enumerate(source_order, start=1):
        current_order = [*previous_order, source]
        step_hits: dict[tuple[str, str, int], bool] = {}
        for _, row in predictions.iterrows():
            key = (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]))
            truth = str(row["ground_truth_diagnosis"])
            labels = [item["label"] for item in materialize_pool(sources, key, current_order)]
            step_hits[key] = pool_has_truth(truth, labels)
        if not previous_order:
            recoveries = int(sum(step_hits.values()))
        else:
            recoveries = int(sum(hit and not previous_has.get(key, False) for key, hit in step_hits.items()))
        rows.append(
            {
                "cohort": cohort_name,
                "step_index": step_index,
                "added_source": source,
                "cumulative_sources": json.dumps(current_order, ensure_ascii=True),
                "candidate_pool_recall_count": int(sum(step_hits.values())),
                "candidate_pool_recall": float(sum(step_hits.values()) / max(len(step_hits), 1)),
                "new_recoveries_at_step": recoveries,
            }
        )
        previous_order = current_order
        previous_has = step_hits
    return pd.DataFrame(rows)


source_contribution = pd.concat(
    [
        source_contribution_rows(scale, scale_sources, SELECTED_SOURCE_ORDER, "scale_meddx100"),
        source_contribution_rows(transfer, transfer_sources, SELECTED_SOURCE_ORDER, "transfer_old90"),
    ],
    ignore_index=True,
)
source_contribution.to_csv(ARTIFACT_ROOT / "pool_source_contribution.csv", index=False)

failure_decomposition_summary = (
    selected_scale.assign(
        failure_type=np.select(
            [
                selected_scale["current_candidate_pool_has_truth"] & selected_scale["current_correct_top1"],
                selected_scale["current_candidate_pool_has_truth"] & ~selected_scale["current_correct_top1"],
                ~selected_scale["current_candidate_pool_has_truth"] & selected_scale["candidate_pool_has_truth"],
                ~selected_scale["candidate_pool_has_truth"],
            ],
            [
                "already_correct",
                "resolver_failure_truth_already_in_current_pool",
                "candidate_pool_recovered_truth",
                "remaining_pool_miss",
            ],
            default="unknown",
        )
    )
    .groupby(["dataset_name", "budget", "split", "failure_type"], as_index=False)
    .agg(num_workups=("case_id", "size"))
)
failure_decomposition_summary.to_csv(ARTIFACT_ROOT / "failure_decomposition_summary.csv", index=False)

hard_case_audits = {
    "remaining_pool_misses": selected_scale[~selected_scale["candidate_pool_has_truth"]]
    .sort_values(["dataset_name", "ground_truth_diagnosis", "budget"])
    .head(80)
    .to_dict(orient="records"),
    "pool_recoveries": selected_scale[selected_scale["recovered_truth_vs_current_pool"]]
    .sort_values(["dataset_name", "ground_truth_diagnosis", "budget"])
    .head(80)
    .to_dict(orient="records"),
    "transfer_remaining_pool_misses": selected_transfer[~selected_transfer["candidate_pool_has_truth"]]
    .sort_values(["dataset_name", "ground_truth_diagnosis", "budget"])
    .head(40)
    .to_dict(orient="records"),
}
write_json(ARTIFACT_ROOT / "hard_case_audits.json", hard_case_audits)


def build_pool_miss_risk_features(baseline: pd.DataFrame) -> pd.DataFrame:
    out = baseline.copy()
    out["workup_id"] = out["dataset_name"].astype(str) + "|" + out["case_id"].astype(str) + "|" + out["budget"].astype(int).astype(str)
    out["pool_miss_label"] = (~out["current_candidate_pool_has_truth"]).astype(int)
    out["weak_confidence"] = (out["confidence"] < 0.70).astype(int)
    out["low_resolver_margin"] = (out["resolver_margin"] < 0.25).astype(int)
    out["small_pool"] = (out["current_candidate_pool_size"] < 5).astype(int)
    return out[
        [
            "dataset_name",
            "case_id",
            "budget",
            "split",
            "workup_id",
            "pool_miss_label",
            "current_candidate_pool_size",
            "num_questions",
            "branch_triggered",
            "branch_count",
            "resolver_margin",
            "confidence",
            "weak_confidence",
            "low_resolver_margin",
            "small_pool",
        ]
    ]


pool_miss_risk_features = build_pool_miss_risk_features(scale_baseline)
pool_miss_risk_features.to_csv(ARTIFACT_ROOT / "pool_miss_risk_features.csv", index=False)

numeric_features = [
    "budget",
    "current_candidate_pool_size",
    "num_questions",
    "branch_count",
    "resolver_margin",
    "confidence",
    "weak_confidence",
    "low_resolver_margin",
    "small_pool",
]
categorical_features = ["dataset_name", "branch_triggered"]
preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
risk_model = Pipeline(
    [
        ("preprocess", preprocess),
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SPLIT_SEED)),
    ]
)
risk_rows: list[dict[str, Any]] = []
train_risk = pool_miss_risk_features[pool_miss_risk_features["split"].eq("train")]
if train_risk["pool_miss_label"].nunique() > 1:
    risk_model.fit(train_risk[numeric_features + categorical_features], train_risk["pool_miss_label"])
    for split_name, group in pool_miss_risk_features.groupby("split", sort=True):
        if not len(group):
            continue
        scores = risk_model.predict_proba(group[numeric_features + categorical_features])[:, 1]
        if group["pool_miss_label"].nunique() > 1:
            auroc = roc_auc_score(group["pool_miss_label"], scores)
            auprc = average_precision_score(group["pool_miss_label"], scores)
        else:
            auroc = np.nan
            auprc = np.nan
        threshold = float(np.quantile(scores, 0.80))
        trigger = scores >= threshold
        risk_rows.append(
            {
                "split": split_name,
                "num_workups": int(len(group)),
                "pool_misses": int(group["pool_miss_label"].sum()),
                "auroc": float(auroc) if not math.isnan(auroc) else np.nan,
                "auprc": float(auprc) if not math.isnan(auprc) else np.nan,
                "p80_trigger_rate": float(trigger.mean()),
                "p80_pool_miss_recall": float((trigger & group["pool_miss_label"].astype(bool)).sum() / max(group["pool_miss_label"].sum(), 1)),
            }
        )
pool_miss_risk_validation_summary = pd.DataFrame(risk_rows)
pool_miss_risk_validation_summary.to_csv(ARTIFACT_ROOT / "pool_miss_risk_validation_summary.csv", index=False)

display(source_contribution)
display(pool_miss_risk_validation_summary)

# %% [markdown]
# ## 8. Figures

# %%
plt.style.use("seaborn-v0_8-whitegrid")

summary_all = candidate_pool_recovery_summary[
    (candidate_pool_recovery_summary["cohort"].eq("scale_meddx100"))
    & (candidate_pool_recovery_summary["dataset_name"].eq("ALL"))
    & (candidate_pool_recovery_summary["budget"].eq(-1))
    & (candidate_pool_recovery_summary["split"].eq("ALL"))
].copy()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(summary_all["policy_name"], summary_all["candidate_pool_recall_count"], color="#3b6ea8")
ax.axhline(850, color="#c23b22", linestyle="--", linewidth=1.5, label="minimum target 850/900")
ax.axhline(865, color="#2e8b57", linestyle="--", linewidth=1.5, label="strong target 865/900")
ax.set_ylabel("Candidate-pool recall count")
ax.set_title("Notebook 53 Candidate-Pool Recall By Policy")
ax.tick_params(axis="x", rotation=30)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "candidate_pool_recall_by_policy.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(summary_all["mean_pool_size"], summary_all["candidate_pool_recall_count"], s=80, color="#5c8a3c")
for _, row in summary_all.iterrows():
    ax.annotate(str(row["policy_name"]).replace("_", "\n"), (row["mean_pool_size"], row["candidate_pool_recall_count"]), fontsize=8)
ax.axhline(850, color="#c23b22", linestyle="--", linewidth=1)
ax.set_xlabel("Mean candidate-pool size")
ax.set_ylabel("Candidate-pool recall count")
ax.set_title("Pool Size Versus Recall")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "pool_size_vs_recall.png", dpi=180)
plt.close(fig)

selected_by_slice = candidate_pool_recovery_summary[
    (candidate_pool_recovery_summary["cohort"].eq("scale_meddx100"))
    & (candidate_pool_recovery_summary["policy_name"].eq(SELECTED_POLICY_NAME))
    & (candidate_pool_recovery_summary["dataset_name"].ne("ALL"))
    & (candidate_pool_recovery_summary["split"].eq("ALL"))
].copy()
selected_by_slice["slice"] = selected_by_slice["dataset_name"] + "@" + selected_by_slice["budget"].astype(str)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(selected_by_slice["slice"], selected_by_slice["candidate_pool_recall"], color="#7d5a9b")
ax.set_ylim(0, 1.02)
ax.set_ylabel("Candidate-pool recall")
ax.set_title("Selected Pool Recovery By Dataset And Budget")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "selected_pool_recall_by_dataset_budget.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
scale_contribution = source_contribution[source_contribution["cohort"].eq("scale_meddx100")]
ax.bar(scale_contribution["added_source"], scale_contribution["new_recoveries_at_step"], color="#d28e39")
ax.set_ylabel("New truth recoveries at step")
ax.set_title("Sequential Source Contribution")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "source_contribution_waterfall.png", dpi=180)
plt.close(fig)

if len(pool_miss_risk_validation_summary):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pool_miss_risk_validation_summary["split"], pool_miss_risk_validation_summary["auroc"], color="#668c99")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("Pool-Miss Risk Detector Diagnostic")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pool_miss_risk_auroc.png", dpi=180)
    plt.close(fig)

# %% [markdown]
# ## 9. Final Summary And Artifact Contract

# %%
selected_all = candidate_pool_recovery_summary[
    (candidate_pool_recovery_summary["cohort"].eq("scale_meddx100"))
    & (candidate_pool_recovery_summary["policy_name"].eq(SELECTED_POLICY_NAME))
    & (candidate_pool_recovery_summary["dataset_name"].eq("ALL"))
    & (candidate_pool_recovery_summary["budget"].eq(-1))
    & (candidate_pool_recovery_summary["split"].eq("ALL"))
].iloc[0]
selected_test = candidate_pool_recovery_summary[
    (candidate_pool_recovery_summary["cohort"].eq("scale_meddx100"))
    & (candidate_pool_recovery_summary["policy_name"].eq(SELECTED_POLICY_NAME))
    & (candidate_pool_recovery_summary["dataset_name"].eq("ALL"))
    & (candidate_pool_recovery_summary["budget"].eq(-1))
    & (candidate_pool_recovery_summary["split"].eq("test"))
].iloc[0]
transfer_all = candidate_pool_recovery_summary[
    (candidate_pool_recovery_summary["cohort"].eq("transfer_old90"))
    & (candidate_pool_recovery_summary["policy_name"].eq(SELECTED_POLICY_NAME))
    & (candidate_pool_recovery_summary["dataset_name"].eq("ALL"))
    & (candidate_pool_recovery_summary["budget"].eq(-1))
    & (candidate_pool_recovery_summary["split"].eq("transfer"))
].iloc[0]

stage1_passed = bool(
    int(selected_all["candidate_pool_recall_count"]) >= 850
    and int(selected_all["lost_truth_vs_current_pool"]) == 0
    and int(transfer_all["candidate_pool_recall_count"]) >= int(transfer_all["current_pool_recall_count"])
)

selected_policy = {
    "selected_policy_name": SELECTED_POLICY_NAME,
    "stage": "stage1_candidate_pool_recovery",
    "deployability": "offline_deployable_final_resolution_layer",
    "inputs_used": {
        "scale_run": SCALE_RUN_NAME,
        "transfer_run": TRANSFER_RUN_NAME,
        "ddxplus_bayes_tables": str(DDXPLUS_BAYES_ROOT.relative_to(ROOT)),
        "rarebench_zip": str(RAREBENCH_ZIP.relative_to(ROOT)),
        "rarebench_mapping_dir": str(RAREBENCH_MAPPING_DIR.relative_to(ROOT)),
    },
    "source_order": SELECTED_SOURCE_ORDER,
    "split_strategy": {
        "seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "validate_fraction": VALIDATE_FRACTION,
        "test_fraction": TEST_FRACTION,
        "grouping": "dataset_name|case_id; all budgets for each case stay in the same split",
    },
    "scale_all": selected_all.to_dict(),
    "scale_test": selected_test.to_dict(),
    "transfer_old90": transfer_all.to_dict(),
    "promotion_decision": {
        "stage1_candidate_pool_recovery_passed": stage1_passed,
        "minimum_pool_recall_target": ">=850/900",
        "strong_pool_recall_target": ">=865/900",
        "transfer_requirement": "selected policy candidate-pool recall non-negative versus current pool on old Notebook 46 artifact",
        "decision": "promote_to_stage2_resolver_lab" if stage1_passed else "diagnostic_only_do_not_promote",
    },
    "guardrails": [
        "No API calls.",
        "No test labels are used as decision-time features.",
        "Truth/correctness fields are used only for evaluation.",
        "RareBench HPO scorer uses initial patient phenotypes plus revealed retrieved spans, not the hidden full profile.",
        "DDXPlus Bayes scorer distinguishes absent, present, and unknown evidence outcomes.",
    ],
}
write_json(ARTIFACT_ROOT / "selected_policy.json", selected_policy)

resolved_run_config = {
    "run_name": RUN_NAME,
    "notebook": "notebooks/53_meddx_candidate_pool_recovery_lab.ipynb",
    "script": "scripts/meddx_candidate_pool_recovery_lab_nb53.py",
    "scale_input": SCALE_RUN_NAME,
    "transfer_input": TRANSFER_RUN_NAME,
    "selected_policy": SELECTED_POLICY_NAME,
    "selected_source_order": SELECTED_SOURCE_ORDER,
    "no_api_calls": True,
    "artifact_root": str(ARTIFACT_ROOT.relative_to(ROOT)),
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

print(json.dumps(selected_policy["promotion_decision"], indent=2))
display(
    candidate_pool_recovery_summary[
        candidate_pool_recovery_summary["policy_name"].eq(SELECTED_POLICY_NAME)
        & candidate_pool_recovery_summary["dataset_name"].eq("ALL")
        & candidate_pool_recovery_summary["budget"].eq(-1)
    ].sort_values(["cohort", "split"])
)
