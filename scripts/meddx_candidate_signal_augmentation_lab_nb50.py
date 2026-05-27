from __future__ import annotations

# %% [markdown]
# # Notebook 50: MEDDx Candidate-Signal Augmentation Lab
#
# Notebook 49 showed that the broad candidate pool is usually sufficient, but not always separable:
# the correct diagnosis is present in 88/90 Notebook 46 workups, while the calibrated resolver reaches 78/90.
# This notebook steps back from "just train a stronger resolver" and asks a more structural question:
#
# > Are the candidates carrying enough dataset-native evidence signal for a resolver to choose correctly?
#
# The notebook is offline-only. It keeps the Notebook 46 live traces frozen, then augments each candidate with
# additional non-leaky support signals:
#
# - DDXPlus train-derived exact-outcome Naive Bayes support over the visible evidence ledger.
# - RareBench leave-one-case-out HPO phenotype reference overlap.
# - iCraft-MD leave-one-case-out vignette/exemplar TF-IDF support.
# - Generic visible-text/candidate-label overlap features.
#
# The selected policy remains conservative: a system-wide calibrated logistic candidate scorer may override the
# Notebook 46 answer only through a zero-regression base-protection gate. Stronger label-fit and tree models are
# diagnostic only.

# %%
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 180)

ROOT = next(
    (
        candidate
        for candidate in [Path.cwd(), *Path.cwd().parents]
        if (candidate / "notebooks").exists() and (candidate / "reports").exists()
    ),
    Path.cwd(),
)

INPUT_RUN_NAME = "meddx_aligned_dataset_native_driver_v1_eval30"
ADJUDICATOR_RUN_NAME = "meddx_candidate_pool_adjudicator_lab_v1"
CALIBRATED_RUN_NAME = "meddx_calibrated_candidate_pool_resolver_v1"
RUN_NAME = "meddx_candidate_signal_augmentation_lab_v1"

INPUT_ROOT = ROOT / "artifacts" / "universal_meddx" / INPUT_RUN_NAME
ADJUDICATOR_ROOT = ROOT / "artifacts" / "universal_meddx" / ADJUDICATOR_RUN_NAME
CALIBRATED_ROOT = ROOT / "artifacts" / "universal_meddx" / CALIBRATED_RUN_NAME
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DDXPLUS_BAYES_ROOT = ROOT / "artifacts" / "bayesian_voi_ledger" / "bayesian_voi_offline_notebook13_49case_v1"
RAREBENCH_ZIP = ROOT / "artifacts" / "universal_meddx" / "cache" / "rarebench_data.zip"
RAREBENCH_MAPPING_DIR = ROOT / "external" / "meddxagent" / "ddxdriver" / "benchmarks" / "data" / "rarebench"
ICRAFT_JSONL = ROOT / "external" / "meddxagent" / "ddxdriver" / "benchmarks" / "data" / "icraftmd" / "all_craft_md.jsonl"

MODEL_C = 1.0
RANDOM_STATE = 50
RUN_HGB_DIAGNOSTIC = False
STRICT_NESTED_DIAGNOSTIC = False

print("Project root:", ROOT)
print("Input root  :", INPUT_ROOT)
print("Base feature root:", ADJUDICATOR_ROOT)
print("Artifact root:", ARTIFACT_ROOT)

# %% [markdown]
# ## 1. Utility Functions

# %%
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, ensure_ascii=True)


def normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def tokenize(value: Any) -> list[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
        "of", "on", "or", "patient", "the", "to", "type", "with", "without",
    }
    return [token for token in normalize_token(value).split() if len(token) > 2 and token not in stop]


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except Exception:
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def label_rank(label: Any, ranked: list[str], missing_rank: int = 99) -> int:
    key = normalize_label(label)
    for idx, item in enumerate(ranked, start=1):
        if normalize_label(item) == key:
            return idx
    return missing_rank


def reciprocal_rank(label: Any, ranked: list[str]) -> float:
    rank = label_rank(label, ranked)
    return 0.0 if rank == 99 else 1.0 / rank


def insert_ranked_prefix(prefix: list[str], ranked: list[str], limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in [*prefix, *ranked]:
        key = normalize_label(label)
        if key and key not in seen:
            out.append(str(label))
            seen.add(key)
    return out[:limit]


def ranked_from_prediction_row(row: pd.Series) -> list[str]:
    ranked = parse_json_list(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "").strip()
    if prediction and normalize_label(prediction) not in {normalize_label(label) for label in ranked}:
        ranked = [prediction] + ranked
    return ranked[:10]


def score_ranked(row: pd.Series, prediction: str, ranked: list[str]) -> dict[str, Any]:
    truth = normalize_label(row["ground_truth_diagnosis"])
    keys = [normalize_label(label) for label in ranked]
    return {
        "correct_top1": normalize_label(prediction) == truth,
        "gtpa_at_3": truth in set(keys[:3]),
        "gtpa_at_5": truth in set(keys[:5]),
        "true_rank": keys.index(truth) + 1 if truth in keys else 11,
    }


def per_slice_summaries(frame: pd.DataFrame, policy_name: str) -> list[dict[str, Any]]:
    rows = []
    for dataset_name, budget, group in [("ALL", -1, frame), *[
        (dataset_name, int(budget), group)
        for (dataset_name, budget), group in frame.groupby(["dataset_name", "budget"], sort=True)
    ]]:
        rows.append({
            "policy_name": policy_name,
            "dataset_name": dataset_name,
            "budget": int(budget),
            "num_workups": int(len(group)),
            "top1_count": int(group["correct_top1"].sum()),
            "top1": float(group["correct_top1"].mean()),
            "top3_count": int(group["gtpa_at_3"].sum()),
            "top3": float(group["gtpa_at_3"].mean()),
            "top5_count": int(group["gtpa_at_5"].sum()),
            "top5": float(group["gtpa_at_5"].mean()),
            "wins_vs_current": int((group["correct_top1"] & ~group["original_correct_top1"]).sum()),
            "regressions_vs_current": int((~group["correct_top1"] & group["original_correct_top1"]).sum()),
            "changed_predictions": int(group["changed_prediction"].sum()),
        })
    return rows


def normalize_by_workup(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        max_value = out.groupby("workup_id")[column].transform("max")
        min_value = out.groupby("workup_id")[column].transform("min")
        denom = (max_value - min_value).replace(0, np.nan)
        out[f"{column}_pool_minmax"] = ((out[column] - min_value) / denom).fillna(0.0)
        out[f"{column}_minus_max"] = (out[column] - max_value).fillna(0.0)
        out[f"{column}_rank_within_pool"] = out.groupby("workup_id")[column].rank(ascending=False, method="min")
    return out


def softmax_from_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_values = {key: math.exp(max(min(value - max_score, 60.0), -60.0)) for key, value in scores.items()}
    total = sum(exp_values.values()) or 1.0
    return {key: value / total for key, value in exp_values.items()}


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


# %% [markdown]
# ## 2. Load Frozen Notebook 46/48/49 Artifacts

# %%
required = [
    INPUT_ROOT / "predictions.csv",
    INPUT_ROOT / "question_answer_ledger.csv",
    INPUT_ROOT / "universal_cases.csv",
    ADJUDICATOR_ROOT / "candidate_level_educator_features.csv",
    ADJUDICATOR_ROOT / "label_free_pool_educator_summary.csv",
    CALIBRATED_ROOT / "calibrated_resolver_policy_summary.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

predictions = pd.read_csv(INPUT_ROOT / "predictions.csv")
ledger = pd.read_csv(INPUT_ROOT / "question_answer_ledger.csv")
universal_cases = pd.read_csv(INPUT_ROOT / "universal_cases.csv")
base_features = pd.read_csv(ADJUDICATOR_ROOT / "candidate_level_educator_features.csv")
nb48_policy_summary = pd.read_csv(ADJUDICATOR_ROOT / "label_free_pool_educator_summary.csv")
nb49_policy_summary = pd.read_csv(CALIBRATED_ROOT / "calibrated_resolver_policy_summary.csv")

for column in base_features.columns:
    if base_features[column].dtype == bool:
        base_features[column] = base_features[column].astype(int)

prediction_by_key = {
    (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"])): row
    for _, row in predictions.iterrows()
}
case_by_key = {
    (str(row["dataset_name"]), str(row["case_id"])): row
    for _, row in universal_cases.iterrows()
}

display(predictions.groupby(["dataset_name", "budget"], as_index=False).agg(top1=("correct_top1", "mean"), top5=("gtpa_at_5", "mean"), n=("case_id", "count")))
display(nb49_policy_summary[nb49_policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))


def visible_text_for_workup(dataset_name: str, case_id: str, budget: int) -> str:
    case_row = case_by_key.get((dataset_name, case_id))
    parts: list[str] = []
    if case_row is not None:
        parts.append(str(case_row.get("initial_patient_info", "")))
    workup_ledger = ledger[
        (ledger["dataset_name"].astype(str) == dataset_name)
        & (ledger["case_id"].astype(str) == case_id)
        & (ledger["budget"].astype(int) == int(budget))
    ].sort_values("turn_index")
    for _, row in workup_ledger.iterrows():
        parts.append(str(row.get("question", "")))
        parts.append(str(row.get("answer", "")))
        parts.extend(parse_json_list(row.get("retrieved_spans", "")))
    return "\n".join(part for part in parts if part and part != "nan")


visible_text_by_workup = {
    f"{row.dataset_name}|{row.case_id}|{int(row.budget)}": visible_text_for_workup(str(row.dataset_name), str(row.case_id), int(row.budget))
    for _, row in predictions.iterrows()
}

# %% [markdown]
# ## 3. DDXPlus Exact-Outcome Bayes Support

# %%
def build_ddxplus_bayes_features() -> pd.DataFrame:
    likelihood_path = DDXPLUS_BAYES_ROOT / "root_outcome_likelihoods.csv"
    prior_path = DDXPLUS_BAYES_ROOT / "diagnosis_priors.csv"
    if not likelihood_path.exists() or not prior_path.exists():
        return pd.DataFrame()

    likelihoods = pd.read_csv(likelihood_path)
    priors = pd.read_csv(prior_path)
    pathologies = [str(item) for item in priors["pathology"].tolist()]
    prior_lookup = {str(row["pathology"]): max(safe_float(row["prior"]), 1e-12) for _, row in priors.iterrows()}
    likelihood_lookup: dict[tuple[str, str, str], float] = {}
    for _, row in likelihoods.iterrows():
        root_id = str(row["root_evidence_id"])
        outcome = str(row["outcome_state"])
        for pathology in pathologies:
            likelihood_lookup[(root_id, outcome, pathology)] = max(min(safe_float(row.get(f"p__{pathology}", 1e-9)), 1 - 1e-9), 1e-9)

    metadata_by_case: dict[str, dict[str, Any]] = {}
    for _, row in universal_cases[universal_cases["dataset_name"].astype(str).eq("ddxplus")].iterrows():
        try:
            metadata_by_case[str(row["case_id"])] = json.loads(str(row.get("metadata", "{}")))
        except Exception:
            metadata_by_case[str(row["case_id"])] = {}

    def observed_for_case(case_id: str, budget: int) -> dict[str, str]:
        metadata = metadata_by_case.get(str(case_id), {})
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

    rows: list[dict[str, Any]] = []
    for _, pred_row in predictions[predictions["dataset_name"].astype(str).eq("ddxplus")].iterrows():
        case_id = str(pred_row["case_id"])
        budget = int(pred_row["budget"])
        observed = observed_for_case(case_id, budget)
        log_scores = {pathology: math.log(prior_lookup.get(pathology, 1e-12)) for pathology in pathologies}
        matched_outcomes = 0
        fallback_outcomes = 0
        for root_id, outcome in observed.items():
            for pathology in pathologies:
                value = likelihood_lookup.get((root_id, outcome, pathology))
                if value is None:
                    fallback_outcomes += 1
                    value = 1e-6
                else:
                    matched_outcomes += 1
                log_scores[pathology] += math.log(value)
        posterior = softmax_from_scores(log_scores)
        ranked = sorted(pathologies, key=lambda pathology: log_scores[pathology], reverse=True)
        top_score = log_scores[ranked[0]]
        second_score = log_scores[ranked[1]] if len(ranked) > 1 else top_score
        for pathology in pathologies:
            rank = ranked.index(pathology) + 1
            rows.append({
                "dataset_name": "ddxplus",
                "case_id": case_id,
                "budget": budget,
                "label_key": normalize_label(pathology),
                "ddxplus_bayes_log_score": float(log_scores[pathology]),
                "ddxplus_bayes_rank": int(rank),
                "ddxplus_bayes_rr": float(1.0 / rank),
                "ddxplus_bayes_posterior": float(posterior.get(pathology, 0.0)),
                "ddxplus_bayes_top1": int(rank == 1),
                "ddxplus_bayes_margin": float(top_score - second_score),
                "ddxplus_bayes_visible_roots": int(len(observed)),
                "ddxplus_bayes_matched_likelihood_cells": int(matched_outcomes),
                "ddxplus_bayes_fallback_likelihood_cells": int(fallback_outcomes),
            })
    return pd.DataFrame(rows)


ddxplus_bayes_features = build_ddxplus_bayes_features()
if len(ddxplus_bayes_features):
    ddxplus_bayes_features.to_csv(ARTIFACT_ROOT / "ddxplus_bayes_candidate_features.csv", index=False)
    bayes_top = (
        ddxplus_bayes_features[ddxplus_bayes_features["ddxplus_bayes_rank"].eq(1)]
        .merge(predictions[predictions["dataset_name"].astype(str).eq("ddxplus")], on=["dataset_name", "case_id", "budget"], how="left")
    )
    bayes_top["bayes_correct"] = bayes_top["label_key"].eq(bayes_top["ground_truth_diagnosis"].map(normalize_label))
    display(bayes_top.groupby("budget", as_index=False).agg(top1=("bayes_correct", "mean"), n=("case_id", "count")))
else:
    display("DDXPlus Bayes artifacts were not available; Bayes augmentation will be zeroed.")

# %% [markdown]
# ## 4. RareBench HPO Reference Separability

# %%
def canonicalize_rarebench_disease(subset: str, disease_codes: list[str], raw_disease_mapping: dict[str, str], subset_mapping: dict[str, dict[str, str]]) -> str:
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
    counts = Counter(mapped_parts)
    return str(counts.most_common(1)[0][0])


def load_rarebench_reference_records() -> tuple[list[dict[str, Any]], dict[tuple[str, int], set[str]], dict[str, float]]:
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
                    phenotypes = [phenotype_mapping[hpo] for hpo in hpo_ids if hpo in phenotype_mapping]
                    disease_codes = [str(code) for code in item.get("RareDisease", [])]
                    canonical = canonicalize_rarebench_disease(subset, disease_codes, raw_disease_mapping, disease_mapping)
                    if canonical:
                        records.append({
                            "subset": subset,
                            "record_index": idx,
                            "label": canonical,
                            "label_key": normalize_label(canonical),
                            "hpo_ids": hpo_ids,
                            "phenotype_names": phenotypes,
                        })

    document_frequency = Counter()
    for record in records:
        document_frequency.update(record["hpo_ids"])
    total = max(len(records), 1)
    idf = {hpo: math.log((1 + total) / (1 + count)) + 1.0 for hpo, count in document_frequency.items()}
    hpo_by_case = {(record["subset"], int(record["record_index"])): set(record["hpo_ids"]) for record in records}
    return records, hpo_by_case, idf


rarebench_records, rarebench_hpo_by_case, rarebench_hpo_idf = load_rarebench_reference_records()
rarebench_records_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
for record in rarebench_records:
    rarebench_records_by_label[record["label_key"]].append(record)


def rarebench_case_ref(case_id: str) -> tuple[str, int] | None:
    parts = str(case_id).split(":")
    if len(parts) >= 3:
        try:
            return parts[1], int(parts[2])
        except Exception:
            return None
    return None


def rarebench_candidate_features_for(dataset_name: str, case_id: str, budget: int, label: str) -> dict[str, float]:
    if dataset_name != "rarebench" or not rarebench_records:
        return {
            "rarebench_hpo_ref_count": 0.0,
            "rarebench_hpo_no_reference": 0.0,
            "rarebench_hpo_patient_count": 0.0,
            "rarebench_hpo_max_overlap": 0.0,
            "rarebench_hpo_mean_overlap": 0.0,
            "rarebench_hpo_max_jaccard": 0.0,
            "rarebench_hpo_mean_jaccard": 0.0,
            "rarebench_hpo_max_idf_fraction": 0.0,
            "rarebench_hpo_mean_idf_fraction": 0.0,
        }
    case_ref = rarebench_case_ref(case_id)
    patient_hpos = rarebench_hpo_by_case.get(case_ref or ("", -1), set())
    patient_idf_total = sum(rarebench_hpo_idf.get(hpo, 1.0) for hpo in patient_hpos) or 1.0
    label_key = normalize_label(label)
    candidate_records = [
        record for record in rarebench_records_by_label.get(label_key, [])
        if not case_ref or not (record["subset"] == case_ref[0] and int(record["record_index"]) == int(case_ref[1]))
    ]
    overlaps: list[float] = []
    jaccards: list[float] = []
    idf_fractions: list[float] = []
    for record in candidate_records:
        overlap = patient_hpos & record["hpo_ids"]
        union = patient_hpos | record["hpo_ids"]
        overlaps.append(float(len(overlap)))
        jaccards.append(float(len(overlap) / len(union)) if union else 0.0)
        idf_fractions.append(float(sum(rarebench_hpo_idf.get(hpo, 1.0) for hpo in overlap) / patient_idf_total))
    return {
        "rarebench_hpo_ref_count": float(len(candidate_records)),
        "rarebench_hpo_no_reference": float(len(candidate_records) == 0),
        "rarebench_hpo_patient_count": float(len(patient_hpos)),
        "rarebench_hpo_max_overlap": float(max(overlaps) if overlaps else 0.0),
        "rarebench_hpo_mean_overlap": float(np.mean(overlaps) if overlaps else 0.0),
        "rarebench_hpo_max_jaccard": float(max(jaccards) if jaccards else 0.0),
        "rarebench_hpo_mean_jaccard": float(np.mean(jaccards) if jaccards else 0.0),
        "rarebench_hpo_max_idf_fraction": float(max(idf_fractions) if idf_fractions else 0.0),
        "rarebench_hpo_mean_idf_fraction": float(np.mean(idf_fractions) if idf_fractions else 0.0),
    }


# %% [markdown]
# ## 5. iCraft-MD Text Exemplar Separability

# %%
def load_icraft_reference() -> tuple[list[dict[str, Any]], TfidfVectorizer | None, Any]:
    if not ICRAFT_JSONL.exists():
        return [], None, None
    records = []
    for line in ICRAFT_JSONL.open("r", encoding="utf-8"):
        item = json.loads(line)
        text = "\n".join([*item.get("context", []), *item.get("facts", [])])
        records.append({
            "id": int(item["id"]),
            "label": str(item["answer"]),
            "label_key": normalize_label(item["answer"]),
            "text": text,
            "options": item.get("options", {}),
        })
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=8000)
    matrix = vectorizer.fit_transform([record["text"] for record in records])
    return records, vectorizer, matrix


icraft_records, icraft_vectorizer, icraft_matrix = load_icraft_reference()
icraft_indices_by_label: dict[str, list[int]] = defaultdict(list)
for idx, record in enumerate(icraft_records):
    icraft_indices_by_label[record["label_key"]].append(idx)


def icraft_record_id(case_id: str) -> int | None:
    parts = str(case_id).split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except Exception:
            return None
    return None


def icraft_candidate_features_for(dataset_name: str, case_id: str, budget: int, label: str) -> dict[str, float]:
    if dataset_name != "icraft_md" or not icraft_records or icraft_vectorizer is None:
        return {
            "icraft_ref_count": 0.0,
            "icraft_no_reference": 0.0,
            "icraft_tfidf_max_similarity": 0.0,
            "icraft_tfidf_mean_similarity": 0.0,
        }
    label_key = normalize_label(label)
    current_id = icraft_record_id(case_id)
    indices = [idx for idx in icraft_indices_by_label.get(label_key, []) if icraft_records[idx]["id"] != current_id]
    if not indices:
        return {
            "icraft_ref_count": 0.0,
            "icraft_no_reference": 1.0,
            "icraft_tfidf_max_similarity": 0.0,
            "icraft_tfidf_mean_similarity": 0.0,
        }
    query = icraft_vectorizer.transform([visible_text_by_workup.get(f"{dataset_name}|{case_id}|{int(budget)}", "")])
    sims = cosine_similarity(query, icraft_matrix[indices]).ravel()
    return {
        "icraft_ref_count": float(len(indices)),
        "icraft_no_reference": 0.0,
        "icraft_tfidf_max_similarity": float(np.max(sims) if len(sims) else 0.0),
        "icraft_tfidf_mean_similarity": float(np.mean(sims) if len(sims) else 0.0),
    }


# %% [markdown]
# ## 6. Augmented Candidate Feature Table

# %%
ddxplus_bayes_by_key = {
    (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]), str(row["label_key"])): row
    for _, row in ddxplus_bayes_features.iterrows()
} if len(ddxplus_bayes_features) else {}

rare_rows: list[dict[str, Any]] = []
icraft_rows: list[dict[str, Any]] = []
generic_rows: list[dict[str, Any]] = []
augmented_rows: list[dict[str, Any]] = []

for _, row in base_features.iterrows():
    dataset_name = str(row["dataset_name"])
    case_id = str(row["case_id"])
    budget = int(row["budget"])
    label = str(row["label"])
    label_key = str(row["label_key"])
    workup_id = str(row["workup_id"])

    bayes = ddxplus_bayes_by_key.get((dataset_name, case_id, budget, label_key), {})
    rare = rarebench_candidate_features_for(dataset_name, case_id, budget, label)
    icraft = icraft_candidate_features_for(dataset_name, case_id, budget, label)

    text_tokens = set(tokenize(visible_text_by_workup.get(workup_id, "")))
    label_tokens = set(tokenize(label))
    overlap = text_tokens & label_tokens
    generic = {
        "visible_text_token_count": float(len(text_tokens)),
        "candidate_label_token_count": float(len(label_tokens)),
        "candidate_label_visible_overlap_count": float(len(overlap)),
        "candidate_label_visible_overlap_fraction": float(len(overlap) / len(label_tokens)) if label_tokens else 0.0,
        "candidate_label_current_overlap_fraction": float(
            len(label_tokens & set(tokenize(row.get("current_prediction", "")))) / len(label_tokens)
        ) if label_tokens else 0.0,
    }

    rare_rows.append({"workup_id": workup_id, "label_key": label_key, **rare})
    icraft_rows.append({"workup_id": workup_id, "label_key": label_key, **icraft})
    generic_rows.append({"workup_id": workup_id, "label_key": label_key, **generic})

    augmented = row.to_dict()
    augmented.update({
        "ddxplus_bayes_log_score": safe_float(bayes.get("ddxplus_bayes_log_score", 0.0)),
        "ddxplus_bayes_rank": safe_int(bayes.get("ddxplus_bayes_rank", 99), 99),
        "ddxplus_bayes_rr": safe_float(bayes.get("ddxplus_bayes_rr", 0.0)),
        "ddxplus_bayes_posterior": safe_float(bayes.get("ddxplus_bayes_posterior", 0.0)),
        "ddxplus_bayes_top1": safe_int(bayes.get("ddxplus_bayes_top1", 0), 0),
        "ddxplus_bayes_margin": safe_float(bayes.get("ddxplus_bayes_margin", 0.0)),
        "ddxplus_bayes_visible_roots": safe_int(bayes.get("ddxplus_bayes_visible_roots", 0), 0),
        "ddxplus_bayes_fallback_likelihood_cells": safe_int(bayes.get("ddxplus_bayes_fallback_likelihood_cells", 0), 0),
        **rare,
        **icraft,
        **generic,
    })
    augmented_rows.append(augmented)

rare_features = pd.DataFrame(rare_rows).drop_duplicates(["workup_id", "label_key"])
icraft_features = pd.DataFrame(icraft_rows).drop_duplicates(["workup_id", "label_key"])
generic_features = pd.DataFrame(generic_rows).drop_duplicates(["workup_id", "label_key"])

candidate_features = pd.DataFrame(augmented_rows)
candidate_features = normalize_by_workup(
    candidate_features,
    [
        "ddxplus_bayes_log_score",
        "ddxplus_bayes_posterior",
        "rarebench_hpo_max_overlap",
        "rarebench_hpo_max_jaccard",
        "rarebench_hpo_max_idf_fraction",
        "icraft_tfidf_max_similarity",
        "icraft_tfidf_mean_similarity",
        "candidate_label_visible_overlap_fraction",
    ],
)
candidate_features["augmented_source_weighted_score"] = (
    candidate_features["source_weighted_score"]
    + 0.40 * candidate_features["ddxplus_bayes_rr"]
    + 0.35 * candidate_features["ddxplus_bayes_posterior_pool_minmax"]
    + 0.45 * candidate_features["rarebench_hpo_max_idf_fraction_pool_minmax"]
    + 0.25 * candidate_features["rarebench_hpo_max_jaccard_pool_minmax"]
    + 0.45 * candidate_features["icraft_tfidf_max_similarity_pool_minmax"]
    + 0.10 * candidate_features["candidate_label_visible_overlap_fraction_pool_minmax"]
)
candidate_features["augmented_independent_signal_count"] = candidate_features["independent_signal_count"] + (
    (candidate_features["ddxplus_bayes_rr"] > 0).astype(int)
    + (candidate_features["rarebench_hpo_max_idf_fraction"] > 0).astype(int)
    + (candidate_features["icraft_tfidf_max_similarity"] > 0).astype(int)
    + (candidate_features["candidate_label_visible_overlap_fraction"] > 0).astype(int)
)
candidate_features["augmented_source_weighted_score_minus_current"] = (
    candidate_features["augmented_source_weighted_score"]
    - candidate_features.groupby("workup_id")["augmented_source_weighted_score"].transform(
        lambda values: values[candidate_features.loc[values.index, "is_current_prediction"].astype(bool)].iloc[0]
        if candidate_features.loc[values.index, "is_current_prediction"].any()
        else values.max()
    )
)

candidate_features.to_csv(ARTIFACT_ROOT / "candidate_level_augmented_signal_features.csv", index=False)
rare_features.to_csv(ARTIFACT_ROOT / "rarebench_hpo_reference_candidate_features.csv", index=False)
icraft_features.to_csv(ARTIFACT_ROOT / "icraft_text_reference_candidate_features.csv", index=False)
generic_features.to_csv(ARTIFACT_ROOT / "generic_text_overlap_candidate_features.csv", index=False)

display(candidate_features.groupby("dataset_name", as_index=False).agg(candidate_rows=("label", "count"), truth_rows=("is_truth", "sum")))
display(candidate_features.head())

# %% [markdown]
# ## 7. Candidate Separability Diagnostics

# %%
signal_rank_columns = [
    "final_rank",
    "llm_rank",
    "mlp_rank",
    "candidate_resolver_rank",
    "ddxplus_graph_rank",
    "ddxplus_bayes_rank",
    "rarebench_hpo_max_idf_fraction_rank_within_pool",
    "rarebench_hpo_max_jaccard_rank_within_pool",
    "icraft_tfidf_max_similarity_rank_within_pool",
    "augmented_source_weighted_score_rank_within_pool",
]

audit_rows: list[dict[str, Any]] = []
for workup_id, group in candidate_features.groupby("workup_id", sort=True):
    truth_rows = group[group["is_truth"].astype(bool)]
    top_aug = group.sort_values(["augmented_source_weighted_score", "pool_rr"], ascending=[False, False]).iloc[0]
    if len(truth_rows):
        truth_row = truth_rows.iloc[0]
        best_rank = min(safe_int(truth_row.get(column, 99), 99) for column in signal_rank_columns if column in truth_row.index)
        truth_signal = {
            f"truth_{column}": safe_float(truth_row.get(column, 99), 99)
            for column in signal_rank_columns
            if column in truth_row.index
        }
    else:
        truth_row = group.iloc[0]
        best_rank = 99
        truth_signal = {}
    dataset_name, case_id, budget_text = workup_id.split("|", 2)
    pred_row = prediction_by_key[(dataset_name, case_id, int(budget_text))]
    audit_rows.append({
        "workup_id": workup_id,
        "dataset_name": dataset_name,
        "case_id": case_id,
        "budget": int(budget_text),
        "ground_truth_diagnosis": pred_row["ground_truth_diagnosis"],
        "current_prediction": pred_row["predicted_diagnosis"],
        "current_correct": boolish(pred_row.get("correct_top1", False)),
        "truth_in_pool": bool(len(truth_rows)),
        "truth_best_signal_rank": int(best_rank),
        "augmented_weighted_top_label": str(top_aug["label"]),
        "augmented_weighted_top_is_truth": bool(top_aug["is_truth"]),
        "candidate_pool_size": int(len(group)),
        **truth_signal,
    })

signal_audit = pd.DataFrame(audit_rows)
signal_audit.to_csv(ARTIFACT_ROOT / "signal_separability_audit.csv", index=False)

signal_summary = (
    signal_audit.groupby("dataset_name", as_index=False)
    .agg(
        workups=("workup_id", "count"),
        truth_in_pool=("truth_in_pool", "mean"),
        best_signal_top1=("truth_best_signal_rank", lambda values: float(np.mean(np.array(values) <= 1))),
        best_signal_top3=("truth_best_signal_rank", lambda values: float(np.mean(np.array(values) <= 3))),
        augmented_weighted_top1=("augmented_weighted_top_is_truth", "mean"),
    )
)
signal_summary.to_csv(ARTIFACT_ROOT / "signal_separability_summary.csv", index=False)
display(signal_summary)
display(signal_audit[~signal_audit["current_correct"]].sort_values(["dataset_name", "case_id", "budget"]))

# %% [markdown]
# ## 8. Case-Blocked Augmented Resolvers

# %%
excluded_feature_columns = {
    "workup_id",
    "case_group",
    "case_id",
    "label",
    "label_key",
    "ground_truth_diagnosis",
    "is_truth",
    "current_prediction",
    "current_correct",
    "ddxplus_graph_top1_label",
}
numeric_columns = [
    column
    for column in candidate_features.columns
    if column not in excluded_feature_columns
    and column != "dataset_name"
    and pd.api.types.is_numeric_dtype(candidate_features[column])
]
categorical_columns = ["dataset_name"]
feature_columns = numeric_columns + categorical_columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)


def make_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=3000, C=MODEL_C, class_weight="balanced")),
    ])


def make_hgb_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ]
        )),
        ("model", HistGradientBoostingClassifier(
            max_iter=35,
            max_leaf_nodes=7,
            learning_rate=0.06,
            l2_regularization=0.25,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )),
    ])


X = candidate_features[feature_columns]
y = candidate_features["is_truth"].astype(int)
groups = candidate_features["case_group"].astype(str)
logo = LeaveOneGroupOut()

feature_contract = {
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "excluded_columns": sorted(excluded_feature_columns),
    "case_blocking": "case_group = dataset_name|case_id; all budgets for one case are held out together",
    "added_signal_families": [
        "ddxplus_exact_outcome_bayes",
        "rarebench_leave_one_case_hpo_reference_overlap",
        "icraft_leave_one_case_tfidf_exemplar_similarity",
        "visible_text_candidate_label_overlap",
    ],
}
write_json(ARTIFACT_ROOT / "augmented_feature_contract.json", feature_contract)
print(f"Numeric features: {len(numeric_columns)}")
print(f"Candidate rows: {len(candidate_features)}")
print(f"Case groups: {groups.nunique()}")


def add_group_probabilities(frame: pd.DataFrame, probability_column: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    raw_col = f"{prefix}_raw_probability"
    group_col = f"{prefix}_group_probability"
    rank_col = f"{prefix}_rank"
    out[raw_col] = out[probability_column].astype(float)
    out[group_col] = out.groupby("workup_id")[raw_col].transform(lambda values: values / (values.sum() + 1e-12))
    out[rank_col] = out.groupby("workup_id")[group_col].rank(ascending=False, method="first").astype(int)
    return out


def oof_scores_for_pipeline(make_pipeline, probability_name: str) -> pd.DataFrame:
    probabilities = np.zeros(len(candidate_features))
    for train_idx, test_idx in logo.split(X, y, groups):
        model = make_pipeline()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    scored = candidate_features.copy()
    scored[probability_name] = probabilities
    return add_group_probabilities(scored, probability_name, probability_name.replace("_probability", ""))


logistic_scores = oof_scores_for_pipeline(make_logistic_pipeline, "case_blocked_logistic_probability")
logistic_scores.to_csv(ARTIFACT_ROOT / "case_blocked_augmented_logistic_candidate_scores.csv", index=False)

hgb_scores = pd.DataFrame()
if RUN_HGB_DIAGNOSTIC:
    hgb_scores = oof_scores_for_pipeline(make_hgb_pipeline, "case_blocked_hgb_probability")
    hgb_scores.to_csv(ARTIFACT_ROOT / "case_blocked_augmented_hgb_candidate_scores.csv", index=False)

# %% [markdown]
# ## 9. Policy Evaluation And Base-Protection Gate

# %%
def workup_decision_frame(scored_candidates: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for workup_id, group in scored_candidates.groupby("workup_id", sort=True):
        top = group.sort_values([score_column, "augmented_source_weighted_score", "pool_rr"], ascending=[False, False, False]).iloc[0]
        current_rows = group[group["is_current_prediction"].astype(bool)]
        current = current_rows.iloc[0] if len(current_rows) else group.sort_values(["final_rr", "pool_rr"], ascending=[False, False]).iloc[0]
        rows.append({
            "workup_id": workup_id,
            "case_group": str(top["case_group"]),
            "dataset_name": str(top["dataset_name"]),
            "case_id": str(top["case_id"]),
            "budget": int(top["budget"]),
            "top_label": str(top["label"]),
            "top_is_truth": bool(top["is_truth"]),
            "top_score": float(top[score_column]),
            "top_augmented_signal_count": float(top["augmented_independent_signal_count"]),
            "top_augmented_weighted_score": float(top["augmented_source_weighted_score"]),
            "current_label": str(current["label"]),
            "current_is_truth": bool(current["is_truth"]),
            "current_score": float(current[score_column]),
            "current_augmented_signal_count": float(current["augmented_independent_signal_count"]),
            "current_augmented_weighted_score": float(current["augmented_source_weighted_score"]),
            "score_margin_vs_current": float(top[score_column] - current[score_column]),
            "augmented_signal_delta_vs_current": float(top["augmented_independent_signal_count"] - current["augmented_independent_signal_count"]),
            "augmented_weighted_delta_vs_current": float(top["augmented_source_weighted_score"] - current["augmented_source_weighted_score"]),
            "learned_changed_from_current": normalize_label(top["label"]) != normalize_label(current["label"]),
        })
    return pd.DataFrame(rows)


logistic_decisions = workup_decision_frame(logistic_scores, "case_blocked_logistic_group_probability")
logistic_decisions.to_csv(ARTIFACT_ROOT / "case_blocked_augmented_logistic_workup_decisions.csv", index=False)


def evaluate_gate(decisions: pd.DataFrame, threshold: dict[str, float]) -> dict[str, Any]:
    choose_learned = (
        decisions["learned_changed_from_current"].astype(bool)
        & (decisions["score_margin_vs_current"] >= threshold["min_score_margin_vs_current"])
        & (decisions["top_score"] >= threshold["min_top_score"])
        & (decisions["augmented_signal_delta_vs_current"] >= threshold["min_augmented_signal_delta_vs_current"])
        & (decisions["augmented_weighted_delta_vs_current"] >= threshold["min_augmented_weighted_delta_vs_current"])
    )
    correct = np.where(choose_learned, decisions["top_is_truth"], decisions["current_is_truth"]).astype(bool)
    return {
        **threshold,
        "top1_count": int(correct.sum()),
        "top1": float(correct.mean()),
        "wins_vs_current": int((correct & ~decisions["current_is_truth"].astype(bool)).sum()),
        "regressions_vs_current": int((~correct & decisions["current_is_truth"].astype(bool)).sum()),
        "changed_predictions": int(choose_learned.sum()),
    }


threshold_grid: list[dict[str, float]] = []
for min_margin in [0.0, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    for min_top in [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for min_signal_delta in [-99.0, -1.0, 0.0, 1.0, 2.0]:
            for min_weighted_delta in [-999.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0]:
                threshold_grid.append({
                    "min_score_margin_vs_current": min_margin,
                    "min_top_score": min_top,
                    "min_augmented_signal_delta_vs_current": min_signal_delta,
                    "min_augmented_weighted_delta_vs_current": min_weighted_delta,
                })

threshold_sweep = pd.DataFrame([evaluate_gate(logistic_decisions, threshold) for threshold in threshold_grid])
threshold_sweep = threshold_sweep.sort_values(
    ["top1_count", "regressions_vs_current", "changed_predictions", "wins_vs_current"],
    ascending=[False, True, True, False],
).reset_index(drop=True)
threshold_sweep.to_csv(ARTIFACT_ROOT / "augmented_resolver_threshold_sweep.csv", index=False)
zero_regression = threshold_sweep[threshold_sweep["regressions_vs_current"].eq(0)]
selected_threshold = zero_regression.iloc[0][[
    "min_score_margin_vs_current",
    "min_top_score",
    "min_augmented_signal_delta_vs_current",
    "min_augmented_weighted_delta_vs_current",
]].to_dict()

display(threshold_sweep.head(12))
print("Selected threshold:", selected_threshold)


def policy_case_results_from_scores(
    scored_candidates: pd.DataFrame,
    policy_name: str,
    score_column: str,
    threshold: dict[str, float] | None = None,
    force_learned_top1: bool = False,
) -> pd.DataFrame:
    decisions = workup_decision_frame(scored_candidates, score_column)
    decision_by_workup = {row["workup_id"]: row for _, row in decisions.iterrows()}
    rows: list[dict[str, Any]] = []
    for workup_id in sorted(scored_candidates["workup_id"].astype(str).unique()):
        dataset_name, case_id, budget_text = workup_id.split("|", 2)
        budget = int(budget_text)
        pred_row = prediction_by_key[(dataset_name, case_id, budget)]
        current_ranked = ranked_from_prediction_row(pred_row)
        current_prediction = current_ranked[0] if current_ranked else str(pred_row.get("predicted_diagnosis", ""))
        candidate_group = scored_candidates[scored_candidates["workup_id"].eq(workup_id)].copy()
        learned_ranked = [
            str(row["label"])
            for _, row in candidate_group.sort_values([score_column, "augmented_source_weighted_score", "pool_rr"], ascending=[False, False, False]).iterrows()
        ]
        decision = decision_by_workup[workup_id]
        if force_learned_top1:
            policy_prediction = str(decision["top_label"])
            action = "learned_top1"
            ranked = insert_ranked_prefix([policy_prediction], learned_ranked + current_ranked)
        else:
            assert threshold is not None
            choose_learned = (
                bool(decision["learned_changed_from_current"])
                and decision["score_margin_vs_current"] >= threshold["min_score_margin_vs_current"]
                and decision["top_score"] >= threshold["min_top_score"]
                and decision["augmented_signal_delta_vs_current"] >= threshold["min_augmented_signal_delta_vs_current"]
                and decision["augmented_weighted_delta_vs_current"] >= threshold["min_augmented_weighted_delta_vs_current"]
            )
            if choose_learned:
                policy_prediction = str(decision["top_label"])
                action = "accepted_learned_challenger"
                ranked = insert_ranked_prefix([policy_prediction], learned_ranked + current_ranked)
            else:
                policy_prediction = current_prediction
                action = "kept_current_base"
                ranked = insert_ranked_prefix([policy_prediction], learned_ranked + current_ranked)
        metrics = score_ranked(pred_row, policy_prediction, ranked)
        rows.append({
            "policy_name": policy_name,
            "dataset_name": dataset_name,
            "case_id": case_id,
            "budget": budget,
            "ground_truth_diagnosis": pred_row["ground_truth_diagnosis"],
            "original_prediction": pred_row["predicted_diagnosis"],
            "policy_prediction": policy_prediction,
            "policy_action": action,
            "policy_ranked_differential": json.dumps(ranked, ensure_ascii=True),
            "learned_top_label": decision["top_label"],
            "top_score": decision["top_score"],
            "current_score": decision["current_score"],
            "score_margin_vs_current": decision["score_margin_vs_current"],
            **metrics,
            "original_correct_top1": boolish(pred_row.get("correct_top1", False)),
            "changed_prediction": normalize_label(policy_prediction) != normalize_label(pred_row["predicted_diagnosis"]),
        })
    return pd.DataFrame(rows)


current_rows: list[dict[str, Any]] = []
for (dataset_name, case_id, budget), pred_row in prediction_by_key.items():
    ranked = ranked_from_prediction_row(pred_row)
    prediction = ranked[0] if ranked else str(pred_row.get("predicted_diagnosis", ""))
    metrics = score_ranked(pred_row, prediction, ranked)
    current_rows.append({
        "policy_name": "notebook46_current",
        "dataset_name": dataset_name,
        "case_id": case_id,
        "budget": int(budget),
        "ground_truth_diagnosis": pred_row["ground_truth_diagnosis"],
        "original_prediction": pred_row["predicted_diagnosis"],
        "policy_prediction": prediction,
        "policy_action": "current",
        "policy_ranked_differential": json.dumps(ranked, ensure_ascii=True),
        "learned_top_label": "",
        "top_score": 0.0,
        "current_score": 0.0,
        "score_margin_vs_current": 0.0,
        **metrics,
        "original_correct_top1": boolish(pred_row.get("correct_top1", False)),
        "changed_prediction": False,
    })
current_results = pd.DataFrame(current_rows)

weighted_scores = candidate_features.copy()
weighted_scores["weighted_signal_probability"] = weighted_scores.groupby("workup_id")["augmented_source_weighted_score"].transform(
    lambda values: (values - values.min() + 1e-9) / ((values - values.min() + 1e-9).sum())
)

weighted_top1_results = policy_case_results_from_scores(
    weighted_scores,
    "augmented_weighted_signal_top1_diagnostic",
    "weighted_signal_probability",
    force_learned_top1=True,
)
logistic_top1_results = policy_case_results_from_scores(
    logistic_scores,
    "case_blocked_augmented_logistic_top1_diagnostic",
    "case_blocked_logistic_group_probability",
    force_learned_top1=True,
)
selected_results = policy_case_results_from_scores(
    logistic_scores,
    "calibrated_augmented_signal_resolver_v1",
    "case_blocked_logistic_group_probability",
    threshold=selected_threshold,
    force_learned_top1=False,
)

case_level_results = pd.concat([
    current_results,
    weighted_top1_results,
    logistic_top1_results,
    selected_results,
], ignore_index=True)
if len(hgb_scores):
    hgb_top1_results = policy_case_results_from_scores(
        hgb_scores,
        "case_blocked_augmented_hgb_top1_diagnostic",
        "case_blocked_hgb_group_probability",
        force_learned_top1=True,
    )
    case_level_results = pd.concat([case_level_results, hgb_top1_results], ignore_index=True)
case_level_results.to_csv(ARTIFACT_ROOT / "case_level_augmented_resolver_results.csv", index=False)

summary_rows: list[dict[str, Any]] = []
for policy_name, group in case_level_results.groupby("policy_name", sort=True):
    summary_rows.extend(per_slice_summaries(group, policy_name))

nb48_selected = nb48_policy_summary[nb48_policy_summary["policy_name"].eq("candidate_pool_oracle_diagnostic")].copy()
nb49_rows = nb49_policy_summary[
    nb49_policy_summary["policy_name"].isin([
        "calibrated_logistic_pool_resolver_v1",
        "strict_nested_threshold_logistic_diagnostic",
    ])
].copy()
policy_summary = pd.concat([pd.DataFrame(summary_rows), nb49_rows, nb48_selected], ignore_index=True)
policy_summary.to_csv(ARTIFACT_ROOT / "augmented_resolver_policy_summary.csv", index=False)
display(policy_summary[policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))

# %% [markdown]
# ## 10. Strict Nested Threshold Diagnostic

# %%
def select_threshold_from_decisions(decisions: pd.DataFrame) -> dict[str, float]:
    sweep = pd.DataFrame([evaluate_gate(decisions, threshold) for threshold in threshold_grid])
    sweep = sweep.sort_values(
        ["top1_count", "regressions_vs_current", "changed_predictions", "wins_vs_current"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    zero_reg = sweep[sweep["regressions_vs_current"].eq(0)]
    selected = zero_reg.iloc[0] if len(zero_reg) else sweep.iloc[0]
    return selected[[
        "min_score_margin_vs_current",
        "min_top_score",
        "min_augmented_signal_delta_vs_current",
        "min_augmented_weighted_delta_vs_current",
    ]].to_dict()


nested_rows: list[pd.DataFrame] = []
nested_threshold_rows: list[dict[str, Any]] = []
if STRICT_NESTED_DIAGNOSTIC:
    for outer_train_idx, outer_test_idx in logo.split(X, y, groups):
        outer_train_groups = groups.iloc[outer_train_idx]
        inner_probabilities = np.zeros(len(outer_train_idx))
        inner_X = X.iloc[outer_train_idx]
        inner_y = y.iloc[outer_train_idx]
        inner_logo = LeaveOneGroupOut()
        for inner_train_local, inner_test_local in inner_logo.split(inner_X, inner_y, outer_train_groups):
            inner_train_idx = outer_train_idx[inner_train_local]
            inner_test_idx = outer_train_idx[inner_test_local]
            model = make_logistic_pipeline()
            model.fit(X.iloc[inner_train_idx], y.iloc[inner_train_idx])
            inner_probabilities[inner_test_local] = model.predict_proba(X.iloc[inner_test_idx])[:, 1]

        inner_scores = candidate_features.iloc[outer_train_idx].copy()
        inner_scores["inner_probability"] = inner_probabilities
        inner_scores = add_group_probabilities(inner_scores, "inner_probability", "inner")
        inner_decisions = workup_decision_frame(inner_scores, "inner_group_probability")
        inner_threshold = select_threshold_from_decisions(inner_decisions)
        nested_threshold_rows.append({"heldout_case_group": str(groups.iloc[outer_test_idx].iloc[0]), **inner_threshold})

        outer_model = make_logistic_pipeline()
        outer_model.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
        outer_scores = candidate_features.iloc[outer_test_idx].copy()
        outer_scores["outer_probability"] = outer_model.predict_proba(X.iloc[outer_test_idx])[:, 1]
        outer_scores = add_group_probabilities(outer_scores, "outer_probability", "outer")
        nested_result = policy_case_results_from_scores(
            outer_scores,
            "strict_nested_augmented_logistic_diagnostic",
            "outer_group_probability",
            threshold=inner_threshold,
            force_learned_top1=False,
        )
        nested_rows.append(nested_result)

if nested_rows:
    nested_results = pd.concat(nested_rows, ignore_index=True).drop_duplicates(["dataset_name", "case_id", "budget"])
    nested_results.to_csv(ARTIFACT_ROOT / "strict_nested_augmented_results.csv", index=False)
    pd.DataFrame(nested_threshold_rows).to_csv(ARTIFACT_ROOT / "strict_nested_augmented_thresholds_by_case.csv", index=False)
    nested_summary = pd.DataFrame(per_slice_summaries(nested_results, "strict_nested_augmented_logistic_diagnostic"))
    nested_summary.to_csv(ARTIFACT_ROOT / "strict_nested_augmented_summary.csv", index=False)
    policy_summary = pd.concat([policy_summary, nested_summary], ignore_index=True)
    policy_summary.to_csv(ARTIFACT_ROOT / "augmented_resolver_policy_summary.csv", index=False)
    display(nested_summary[nested_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 11. Final Fit, Error Analysis, And Figures

# %%
final_model = make_logistic_pipeline()
final_model.fit(X, y)
model_path = ARTIFACT_ROOT / "calibrated_augmented_signal_resolver_v1.joblib"
joblib.dump(final_model, model_path)

try:
    transformed_feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
except Exception:
    transformed_feature_names = np.array([f"feature_{idx}" for idx in range(len(final_model.named_steps["model"].coef_[0]))])
coefficients = pd.DataFrame({
    "feature": transformed_feature_names,
    "coefficient": final_model.named_steps["model"].coef_[0],
})
coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
coefficients = coefficients.sort_values("abs_coefficient", ascending=False)
coefficients.to_csv(ARTIFACT_ROOT / "augmented_logistic_feature_coefficients.csv", index=False)
display(coefficients.head(30))

final_fit_scores = candidate_features.copy()
final_fit_scores["final_fit_probability"] = final_model.predict_proba(X)[:, 1]
final_fit_scores = add_group_probabilities(final_fit_scores, "final_fit_probability", "final_fit")
final_fit_scores.to_csv(ARTIFACT_ROOT / "final_fit_augmented_candidate_scores_diagnostic.csv", index=False)
final_fit_results = policy_case_results_from_scores(
    final_fit_scores,
    "label_fit_augmented_logistic_top1_diagnostic",
    "final_fit_group_probability",
    force_learned_top1=True,
)
final_fit_results.to_csv(ARTIFACT_ROOT / "final_fit_augmented_case_results_diagnostic.csv", index=False)
final_fit_summary = pd.DataFrame(per_slice_summaries(final_fit_results, "label_fit_augmented_logistic_top1_diagnostic"))
final_fit_summary.to_csv(ARTIFACT_ROOT / "final_fit_augmented_summary_diagnostic.csv", index=False)
policy_summary = pd.concat([policy_summary, final_fit_summary], ignore_index=True)
policy_summary.to_csv(ARTIFACT_ROOT / "augmented_resolver_policy_summary.csv", index=False)
display(final_fit_summary[final_fit_summary["dataset_name"].eq("ALL")])

selected_failures = selected_results[~selected_results["correct_top1"]].copy()
selected_failures.to_csv(ARTIFACT_ROOT / "selected_augmented_resolver_failure_audit.csv", index=False)

selected_changes = selected_results[selected_results["changed_prediction"].astype(bool)].copy()
selected_changes.to_csv(ARTIFACT_ROOT / "selected_augmented_resolver_changes.csv", index=False)

diagnostic_failures = signal_audit[~signal_audit["current_correct"]].copy()
diagnostic_failures.to_csv(ARTIFACT_ROOT / "remaining_signal_quality_failure_audit.csv", index=False)
display(selected_failures[[
    "dataset_name",
    "case_id",
    "budget",
    "ground_truth_diagnosis",
    "original_prediction",
    "policy_prediction",
    "learned_top_label",
    "score_margin_vs_current",
    "top_score",
    "current_score",
]])

plot_summary = policy_summary[policy_summary["dataset_name"].eq("ALL")].copy()
plot_summary = plot_summary.sort_values("top1", ascending=True)
plt.figure(figsize=(10, 5))
colors = ["#F58518" if "diagnostic" in name or "oracle" in name else "#4C78A8" for name in plot_summary["policy_name"]]
plt.barh(plot_summary["policy_name"], plot_summary["top1"], color=colors)
plt.xlim(0, 1)
plt.xlabel("Top-1 accuracy")
plt.title("Candidate-Signal Augmentation Policies")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "augmented_policy_top1.png", dpi=180)
plt.close()

selected_overall = policy_summary[
    policy_summary["policy_name"].eq("calibrated_augmented_signal_resolver_v1") & policy_summary["dataset_name"].eq("ALL")
].iloc[0]
plt.figure(figsize=(5, 4))
plt.bar(["Wins", "Regressions"], [selected_overall["wins_vs_current"], selected_overall["regressions_vs_current"]], color=["#54A24B", "#E45756"])
plt.ylabel("Workups")
plt.title("Selected Augmented Resolver vs Notebook 46")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "selected_augmented_wins_regressions.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
for dataset_name, group in signal_audit.groupby("dataset_name"):
    plt.hist(group["truth_best_signal_rank"], bins=[1, 2, 3, 4, 6, 11, 30, 100], alpha=0.55, label=dataset_name)
plt.xlabel("Best rank of truth under any candidate signal")
plt.ylabel("Workups")
plt.title("Candidate Signal Separability")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "truth_best_signal_rank_distribution.png", dpi=180)
plt.close()

coef_plot = coefficients.head(22).sort_values("coefficient")
plt.figure(figsize=(9, 6))
plt.barh(coef_plot["feature"], coef_plot["coefficient"], color=np.where(coef_plot["coefficient"] >= 0, "#4C78A8", "#E45756"))
plt.xlabel("Logistic coefficient")
plt.title("Top Augmented Resolver Coefficients")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "top_augmented_logistic_coefficients.png", dpi=180)
plt.close()

slice_plot = policy_summary[
    policy_summary["policy_name"].eq("calibrated_augmented_signal_resolver_v1")
    & ~policy_summary["dataset_name"].eq("ALL")
].copy()
slice_plot["slice"] = slice_plot["dataset_name"] + " B" + slice_plot["budget"].astype(str)
plt.figure(figsize=(9, 4))
plt.bar(slice_plot["slice"], slice_plot["top1"], color="#72B7B2")
plt.ylim(0, 1.05)
plt.ylabel("Top-1 accuracy")
plt.xticks(rotation=35, ha="right")
plt.title("Selected Augmented Resolver by Dataset and Budget")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "selected_augmented_by_slice.png", dpi=180)
plt.close()

# %% [markdown]
# ## 12. Final Summary And Artifact Contract

# %%
current_overall = policy_summary[
    policy_summary["policy_name"].eq("notebook46_current") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
selected_overall = policy_summary[
    policy_summary["policy_name"].eq("calibrated_augmented_signal_resolver_v1") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
nb49_overall = policy_summary[
    policy_summary["policy_name"].eq("calibrated_logistic_pool_resolver_v1") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
oracle_overall = policy_summary[
    policy_summary["policy_name"].eq("candidate_pool_oracle_diagnostic") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
final_fit_overall = policy_summary[
    policy_summary["policy_name"].eq("label_fit_augmented_logistic_top1_diagnostic") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
nested_overall = None
if (ARTIFACT_ROOT / "strict_nested_augmented_summary.csv").exists():
    nested_frame = pd.read_csv(ARTIFACT_ROOT / "strict_nested_augmented_summary.csv")
    nested_overall = nested_frame[nested_frame["dataset_name"].eq("ALL")].iloc[0].to_dict()

resolved_config = {
    "run_name": RUN_NAME,
    "input_run_name": INPUT_RUN_NAME,
    "base_feature_run_name": ADJUDICATOR_RUN_NAME,
    "calibrated_reference_run_name": CALIBRATED_RUN_NAME,
    "artifact_root": str(ARTIFACT_ROOT),
    "live_api_used": False,
    "selected_policy": "calibrated_augmented_signal_resolver_v1",
    "selected_threshold": selected_threshold,
    "model_family": "L2-regularized logistic regression over augmented candidate-level signals",
    "hgb_diagnostic_run": RUN_HGB_DIAGNOSTIC,
    "strict_nested_diagnostic_run": STRICT_NESTED_DIAGNOSTIC,
    "case_blocking": "dataset_name|case_id; all budgets for a held-out case are predicted by models trained without that case",
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

artifact_contract = [
    "resolved_run_config.json",
    "augmented_feature_contract.json",
    "candidate_level_augmented_signal_features.csv",
    "ddxplus_bayes_candidate_features.csv",
    "rarebench_hpo_reference_candidate_features.csv",
    "icraft_text_reference_candidate_features.csv",
    "generic_text_overlap_candidate_features.csv",
    "signal_separability_audit.csv",
    "signal_separability_summary.csv",
    "case_blocked_augmented_logistic_candidate_scores.csv",
    "augmented_resolver_threshold_sweep.csv",
    "case_level_augmented_resolver_results.csv",
    "augmented_resolver_policy_summary.csv",
    "final_fit_augmented_candidate_scores_diagnostic.csv",
    "final_fit_augmented_summary_diagnostic.csv",
    "augmented_logistic_feature_coefficients.csv",
    "selected_augmented_resolver_failure_audit.csv",
    "remaining_signal_quality_failure_audit.csv",
    "selected_augmented_signal_policy.json",
    "figures/",
]
if RUN_HGB_DIAGNOSTIC:
    artifact_contract.append("case_blocked_augmented_hgb_candidate_scores.csv")
if STRICT_NESTED_DIAGNOSTIC:
    artifact_contract.extend(["strict_nested_augmented_results.csv", "strict_nested_augmented_summary.csv"])

selected_status = (
    "offline_candidate_promoted_over_notebook49"
    if selected_overall["top1"] > nb49_overall["top1"]
    else "diagnostic_not_promoted_notebook49_remains_stronger_top1"
)
selected_payload = {
    "selected_policy_name": "calibrated_augmented_signal_resolver_v1",
    "status": selected_status,
    "current_overall": current_overall,
    "notebook49_reference_overall": nb49_overall,
    "selected_case_blocked_overall": selected_overall,
    "label_fit_augmented_logistic_diagnostic_overall": final_fit_overall,
    "strict_nested_augmented_overall": nested_overall,
    "candidate_pool_oracle_overall": oracle_overall,
    "selected_threshold": selected_threshold,
    "model_path": str(model_path),
    "feature_contract_path": str(ARTIFACT_ROOT / "augmented_feature_contract.json"),
    "interpretation": [
        "This notebook tests candidate quality and separability, not just resolver capacity.",
        "DDXPlus Bayes adds a mathematically clean train-derived evidence signal but does not fully rescue the URTI/acquisition failures.",
        "RareBench HPO overlap exposes genuine reference limitations: some true diseases have no leave-one-case exemplar and some close metabolic neighbors have stronger phenotype overlap than the truth.",
        "iCraft-MD exemplar similarity is constrained by one-off labels; the hardest EBS-Dowling Meara case has no same-label exemplar after leave-one-case exclusion.",
        "If the selected policy does not materially beat Notebook 49, the main lesson is that the remaining gap is candidate-signal quality and acquisition, not a missing black-box resolver trick.",
    ],
    "artifact_contract": artifact_contract,
}
write_json(ARTIFACT_ROOT / "selected_augmented_signal_policy.json", selected_payload)

print("Wrote artifacts to:", ARTIFACT_ROOT)
display(policy_summary[policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))
display(pd.DataFrame([selected_payload]))
