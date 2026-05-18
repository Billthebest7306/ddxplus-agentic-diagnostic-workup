from __future__ import annotations

import ast
import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RANDOM_SEED = 32
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (candidate_root / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1").exists():
        PROJECT_ROOT = candidate_root
        break
NOTEBOOK30_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1"
NOTEBOOK31_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "neural_candidate_pool_resolver_49case_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "resolver_ablation_lab_49case_v1"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "resolver_ablation_lab_report.md"
RUN_NAME = "resolver_ablation_lab_49case_v1"

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except Exception:
                return []
    return []


def softmax(values: Any) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=-1e9, posinf=1e9, neginf=-1e9)
    arr = arr - np.max(arr)
    exp = np.exp(np.clip(arr, -80, 80))
    denom = exp.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(exp) / len(exp)
    return exp / denom


def clean_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=-1e6, upper=1e6)


def clean_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return clean_numeric_frame(df, columns).to_numpy(dtype=np.float32)


def compact_miss_list(selected: pd.DataFrame, prediction_col: str = "candidate_pathology") -> str:
    misses = selected[~selected["candidate_label"].astype(bool)]
    parts = []
    for _, row in misses.iterrows():
        parts.append(f"{row['case_id']}:{row['true_pathology']}->{row[prediction_col]}")
    return "; ".join(parts)


required_inputs = {
    "candidate_resolver_train_validate_features": NOTEBOOK30_ROOT / "candidate_resolver_train_validate_features.csv",
    "candidate_level_live_scores": NOTEBOOK30_ROOT / "candidate_level_live_scores.csv",
    "predictions": NOTEBOOK30_ROOT / "predictions.csv",
    "metrics": NOTEBOOK30_ROOT / "metrics.json",
}
missing = [name for name, path in required_inputs.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required Notebook 30 artifacts: {missing}")

train_validate_raw = pd.read_csv(required_inputs["candidate_resolver_train_validate_features"])
live_candidates_raw = pd.read_csv(required_inputs["candidate_level_live_scores"])
notebook30_predictions = pd.read_csv(required_inputs["predictions"])
notebook30_metrics = json.loads(required_inputs["metrics"].read_text(encoding="utf-8"))

nb31_candidate_scores = (
    pd.read_csv(NOTEBOOK31_ROOT / "candidate_level_neural_scores.csv")
    if (NOTEBOOK31_ROOT / "candidate_level_neural_scores.csv").exists()
    else pd.DataFrame()
)
nb31_validation_summary = (
    pd.read_csv(NOTEBOOK31_ROOT / "neural_resolver_validation_summary.csv")
    if (NOTEBOOK31_ROOT / "neural_resolver_validation_summary.csv").exists()
    else pd.DataFrame()
)
nb31_summary_metrics = (
    pd.read_csv(NOTEBOOK31_ROOT / "summary_metrics.csv")
    if (NOTEBOOK31_ROOT / "summary_metrics.csv").exists()
    else pd.DataFrame()
)

print("Notebook 30 train/validate rows:", train_validate_raw.shape)
print("Notebook 30 live candidate rows:", live_candidates_raw.shape)
print("Notebook 30 live cases:", live_candidates_raw["case_id"].nunique())
print("Notebook 30 selected accuracy:", notebook30_metrics.get("num_correct"), "/", notebook30_metrics.get("num_cases"))
if not nb31_summary_metrics.empty:
    print("Notebook 31 reference accuracy:", int(nb31_summary_metrics.loc[0, "num_correct"]), "/", int(nb31_summary_metrics.loc[0, "num_cases"]))

BASE_FEATURES = [
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

EXCLUDED_FEATURE_COLUMNS = {
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
    for col in BASE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[BASE_FEATURES] = out[BASE_FEATURES].replace([np.inf, -np.inf], np.nan)
    for col in BASE_FEATURES:
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


def candidate_pool_oracle(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in df.groupby("case_id"):
        true_pathology = group["true_pathology"].iloc[0]
        unique_candidates = sorted(set(group["predicted_pathology"].dropna()))
        rows.append({
            "case_id": case_id,
            "true_pathology": true_pathology,
            "candidate_rows": int(len(group)),
            "unique_candidate_diagnoses": int(len(unique_candidates)),
            "true_in_candidate_pool": bool(true_pathology in unique_candidates),
        })
    return pd.DataFrame(rows)


train_validate_base = add_resolver_features(train_validate_raw, add_name_features=False)
live_candidates_base = add_resolver_features(make_live_training_schema(live_candidates_raw), add_name_features=False)
train_validate_name = add_resolver_features(train_validate_raw, add_name_features=True)
live_candidates_name = add_resolver_features(make_live_training_schema(live_candidates_raw), add_name_features=True)

BASE_FEATURE_COLUMNS = [
    col for col in train_validate_base.columns
    if col not in EXCLUDED_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(train_validate_base[col])
]
NAME_FEATURE_COLUMNS = [
    col for col in train_validate_name.columns
    if col not in EXCLUDED_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(train_validate_name[col])
]
for col in BASE_FEATURE_COLUMNS:
    if col not in live_candidates_base.columns:
        live_candidates_base[col] = 0.0
for col in NAME_FEATURE_COLUMNS:
    if col not in live_candidates_name.columns:
        live_candidates_name[col] = 0.0

pool_oracle = candidate_pool_oracle(live_candidates_raw)
validate_base = train_validate_base[train_validate_base["split"] == "validate"].copy()
train_base = train_validate_base[train_validate_base["split"] == "train"].copy()
validate_name = train_validate_name[train_validate_name["split"] == "validate"].copy()
train_name = train_validate_name[train_validate_name["split"] == "train"].copy()
validate_pool_recall = float((validate_base.groupby("synthetic_state_id")["candidate_label"].sum() > 0).mean())

print("Base feature count:", len(BASE_FEATURE_COLUMNS))
print("Name/family feature count:", len(NAME_FEATURE_COLUMNS))
print("Candidate-pool oracle:", int(pool_oracle["true_in_candidate_pool"].sum()), "/", len(pool_oracle))

results: list[dict[str, Any]] = []
selected_by_resolver: dict[str, pd.DataFrame] = {}
identity_cols = ["case_id", "branch_id", "candidate_role", "branch_role_name", "true_pathology", "candidate_pathology", "candidate_label"]
candidate_score_table = live_candidates_base[identity_cols].copy()


def safe_score_column_name(name: str) -> str:
    return "score__" + re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()


def select_by_score(df: pd.DataFrame, score_col: str, resolver_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1e9)
    tie_cols = ["candidate_graph_posterior", "candidate_bayes_posterior", "candidate_mlp_posterior", "candidate_graph_score"]
    for col in tie_cols:
        if col not in out.columns:
            out[col] = 0.0
    out = out.sort_values(["synthetic_state_id", score_col] + tie_cols, ascending=[True, False, False, False, False, False]).copy()
    out[f"{resolver_name}_rank"] = out.groupby("synthetic_state_id").cumcount() + 1
    out[f"selected_by_{resolver_name}"] = out[f"{resolver_name}_rank"] == 1
    out[f"{resolver_name}_score_margin"] = out.groupby("synthetic_state_id")[score_col].transform(lambda s: s.iloc[0] - (s.iloc[1] if len(s) > 1 else np.nan))
    out[f"{resolver_name}_group_probability"] = out.groupby("synthetic_state_id")[score_col].transform(lambda s: softmax(s).tolist())
    selected = out[out[f"selected_by_{resolver_name}"]].copy()
    return out, selected


def evaluate_scores(
    *,
    resolver_name: str,
    resolver_family: str,
    live_df: pd.DataFrame,
    live_score: np.ndarray,
    validate_df: pd.DataFrame | None = None,
    validate_score: np.ndarray | None = None,
    validation_available: bool = True,
    validatable_without_49_labels: bool = True,
    uses_49_labels_for_selection: bool = False,
    uses_unobserved_full_evidence: bool = False,
    notes: str = "",
) -> pd.DataFrame:
    score_col = safe_score_column_name(resolver_name)
    live_scored = live_df.copy()
    live_scored[score_col] = np.asarray(live_score, dtype=float)
    _, live_selected = select_by_score(live_scored, score_col, resolver_name)
    live_correct = int(live_selected["candidate_label"].astype(bool).sum())
    live_cases = int(live_selected["synthetic_state_id"].nunique())

    row: dict[str, Any] = {
        "resolver_name": resolver_name,
        "resolver_family": resolver_family,
        "validation_available": bool(validation_available and validate_df is not None and validate_score is not None),
        "validatable_without_49_labels": bool(validatable_without_49_labels),
        "uses_49_labels_for_selection": bool(uses_49_labels_for_selection),
        "uses_unobserved_full_evidence": bool(uses_unobserved_full_evidence),
        "validate_candidate_pool_recall": validate_pool_recall if validation_available else np.nan,
        "validate_row_auc": np.nan,
        "validate_row_average_precision": np.nan,
        "validate_group_argmax_accuracy_all_groups": np.nan,
        "validate_group_argmax_accuracy_conditional_on_candidate_pool": np.nan,
        "live_cases": live_cases,
        "live_correct": live_correct,
        "live_accuracy": live_correct / live_cases if live_cases else np.nan,
        "live_misses": compact_miss_list(live_selected),
        "additional_api_calls": 0,
        "total_api_calls_if_run_after_pool": int(notebook30_metrics.get("total_api_calls", 0)),
        "mean_selected_requests_if_run_after_pool": float(notebook30_metrics.get("mean_selected_requests", np.nan)),
        "mean_total_branch_requests_if_run_after_pool": float(notebook30_metrics.get("mean_total_branch_requests", np.nan)),
        "notes": notes,
    }

    if validation_available and validate_df is not None and validate_score is not None:
        validate_scored = validate_df.copy()
        validate_scored[score_col] = np.asarray(validate_score, dtype=float)
        _, validate_selected = select_by_score(validate_scored, score_col, resolver_name)
        validate_has_truth = (validate_df.groupby("synthetic_state_id")["candidate_label"].sum() > 0).to_dict()
        validate_selected["candidate_pool_contains_truth"] = validate_selected["synthetic_state_id"].map(validate_has_truth).fillna(False)
        row["validate_group_argmax_accuracy_all_groups"] = float(validate_selected["candidate_label"].astype(bool).mean())
        conditional = validate_selected[validate_selected["candidate_pool_contains_truth"]]
        row["validate_group_argmax_accuracy_conditional_on_candidate_pool"] = float(conditional["candidate_label"].astype(bool).mean()) if len(conditional) else np.nan
        try:
            y_validate = validate_df["candidate_label"].astype(int).to_numpy()
            row["validate_row_auc"] = float(roc_auc_score(y_validate, validate_score))
            row["validate_row_average_precision"] = float(average_precision_score(y_validate, validate_score))
        except ValueError:
            pass

    results.append(row)
    selected_by_resolver[resolver_name] = live_selected
    candidate_score_table[score_col] = np.asarray(live_score, dtype=float)
    return live_selected


def add_reference_result(
    *,
    resolver_name: str,
    resolver_family: str,
    live_correct: int,
    live_cases: int,
    live_misses: str,
    validation_available: bool = False,
    validatable_without_49_labels: bool = True,
    uses_49_labels_for_selection: bool = False,
    uses_unobserved_full_evidence: bool = False,
    validate_group_argmax_accuracy_all_groups: float = np.nan,
    validate_group_argmax_accuracy_conditional_on_candidate_pool: float = np.nan,
    validate_row_auc: float = np.nan,
    validate_row_average_precision: float = np.nan,
    notes: str = "",
) -> None:
    results.append({
        "resolver_name": resolver_name,
        "resolver_family": resolver_family,
        "validation_available": validation_available,
        "validatable_without_49_labels": validatable_without_49_labels,
        "uses_49_labels_for_selection": uses_49_labels_for_selection,
        "uses_unobserved_full_evidence": uses_unobserved_full_evidence,
        "validate_candidate_pool_recall": validate_pool_recall if validation_available else np.nan,
        "validate_row_auc": validate_row_auc,
        "validate_row_average_precision": validate_row_average_precision,
        "validate_group_argmax_accuracy_all_groups": validate_group_argmax_accuracy_all_groups,
        "validate_group_argmax_accuracy_conditional_on_candidate_pool": validate_group_argmax_accuracy_conditional_on_candidate_pool,
        "live_cases": live_cases,
        "live_correct": live_correct,
        "live_accuracy": live_correct / live_cases if live_cases else np.nan,
        "live_misses": live_misses,
        "additional_api_calls": 0,
        "total_api_calls_if_run_after_pool": int(notebook30_metrics.get("total_api_calls", 0)),
        "mean_selected_requests_if_run_after_pool": float(notebook30_metrics.get("mean_selected_requests", np.nan)),
        "mean_total_branch_requests_if_run_after_pool": float(notebook30_metrics.get("mean_total_branch_requests", np.nan)),
        "notes": notes,
    })


def selected_from_boolean(df: pd.DataFrame, flag_col: str, prediction_col: str = "predicted_pathology") -> pd.DataFrame:
    selected = df[df[flag_col].astype(bool)].copy().drop_duplicates("case_id")
    selected = selected.rename(columns={prediction_col: "candidate_pathology", "correct": "candidate_label"})
    selected["synthetic_state_id"] = selected["case_id"]
    return selected


nb30_selected = selected_from_boolean(live_candidates_raw, "selected_by_judge")
selected_by_resolver["notebook30_hand_resolver_reference"] = nb30_selected
add_reference_result(
    resolver_name="notebook30_hand_resolver_reference",
    resolver_family="reference",
    live_correct=int(nb30_selected["candidate_label"].astype(bool).sum()),
    live_cases=int(nb30_selected["case_id"].nunique()),
    live_misses=compact_miss_list(nb30_selected),
    notes="Notebook 30 production hand resolver over the hypothesis-branching pool.",
)

if not nb31_candidate_scores.empty:
    nb31_eval_df = nb31_candidate_scores.copy()
    nb31_eval_df["synthetic_state_id"] = nb31_eval_df["case_id"]
    evaluate_scores(
        resolver_name="notebook31_compact_neural_reference",
        resolver_family="reference",
        live_df=nb31_eval_df,
        live_score=nb31_eval_df["neural_score"].to_numpy(),
        validation_available=False,
        notes="Notebook 31 committed compact neural resolver artifact.",
    )
    if not nb31_validation_summary.empty:
        row_mask = nb31_validation_summary["model_name"].eq("compact_neural_candidate_resolver_v1")
        if row_mask.any():
            vrow = nb31_validation_summary[row_mask].iloc[0]
            for result in results:
                if result["resolver_name"] == "notebook31_compact_neural_reference":
                    result["validation_available"] = True
                    result["validate_candidate_pool_recall"] = float(vrow.get("validate_candidate_pool_recall", np.nan))
                    result["validate_row_auc"] = float(vrow.get("validate_row_auc", np.nan))
                    result["validate_row_average_precision"] = float(vrow.get("validate_row_average_precision", np.nan))
                    result["validate_group_argmax_accuracy_all_groups"] = float(vrow.get("validate_group_argmax_accuracy_all_groups", np.nan))
                    result["validate_group_argmax_accuracy_conditional_on_candidate_pool"] = float(vrow.get("validate_group_argmax_accuracy_conditional_on_candidate_pool", np.nan))

heuristic_specs = [
    ("graph_posterior", "candidate_graph_posterior", 1.0),
    ("bayes_posterior", "candidate_bayes_posterior", 1.0),
    ("mlp_posterior", "candidate_mlp_posterior", 1.0),
    ("posterior_mean", "posterior_mean", 1.0),
    ("posterior_min", "posterior_min", 1.0),
    ("graph_bayes_mlp_product", "graph_bayes_mlp_product", 1.0),
    ("rank_consensus_negative", "rank_consensus_sum", -1.0),
]
for resolver_name, col, sign in heuristic_specs:
    evaluate_scores(
        resolver_name=resolver_name,
        resolver_family="heuristic_rank_or_posterior",
        validate_df=validate_name,
        validate_score=sign * validate_name[col].fillna(0).to_numpy(),
        live_df=live_candidates_name,
        live_score=sign * live_candidates_name[col].fillna(0).to_numpy(),
        notes="Single-signal or simple posterior/rank heuristic.",
    )

for weights in [(1, 1, 1), (2, 2, 1), (3, 2, 1), (1, 2, 2), (1, 1, 2), (1, 3, 2)]:
    name = "reciprocal_rank_fusion_gbm_" + "_".join(map(str, weights))

    def rrf(df: pd.DataFrame, weights: tuple[int, int, int] = weights) -> np.ndarray:
        return (
            weights[0] / (10.0 + df["candidate_graph_rank"].astype(float))
            + weights[1] / (10.0 + df["candidate_bayes_rank"].astype(float))
            + weights[2] / (10.0 + df["candidate_mlp_rank"].astype(float))
        ).to_numpy()

    evaluate_scores(
        resolver_name=name,
        resolver_family="heuristic_rank_fusion",
        validate_df=validate_name,
        validate_score=rrf(validate_name),
        live_df=live_candidates_name,
        live_score=rrf(live_candidates_name),
        notes="Reciprocal-rank fusion over graph, Bayes, and MLP ranks.",
    )


def fit_predict_model(
    *,
    resolver_name: str,
    resolver_family: str,
    model: Any,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    live_df: pd.DataFrame,
    notes: str,
) -> None:
    x_train = clean_matrix(train_df, feature_columns)
    y_train = train_df["candidate_label"].astype(int).to_numpy()
    x_validate = clean_matrix(validate_df, feature_columns)
    x_live = clean_matrix(live_df, feature_columns)
    model.fit(x_train, y_train)
    validate_score = model.predict_proba(x_validate)[:, 1]
    live_score = model.predict_proba(x_live)[:, 1]
    evaluate_scores(
        resolver_name=resolver_name,
        resolver_family=resolver_family,
        validate_df=validate_df,
        validate_score=validate_score,
        live_df=live_df,
        live_score=live_score,
        notes=notes,
    )


base_model_specs = {
    "logistic_l2_base_features": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)),
    "mlp_64_32_base_features": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=150, early_stopping=True, validation_fraction=0.15, n_iter_no_change=12, random_state=RANDOM_SEED)),
    "extra_trees_base_features": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
}

name_model_specs = {
    "logistic_l2_name_family": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)),
    "logistic_balanced_l2_name_family": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs", class_weight="balanced", random_state=RANDOM_SEED)),
    "logistic_sparse_l1_balanced_name_family": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.08, solver="liblinear", penalty="l1", class_weight="balanced", random_state=RANDOM_SEED)),
    "mlp_64_32_name_family": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=150, early_stopping=True, validation_fraction=0.15, n_iter_no_change=12, random_state=RANDOM_SEED)),
    "mlp_128_64_regularized_name_family": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=150, early_stopping=True, validation_fraction=0.15, n_iter_no_change=12, random_state=RANDOM_SEED)),
    "random_forest_name_family": RandomForestClassifier(n_estimators=300, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED),
    "extra_trees_name_family": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
    "gradient_boosting_name_family": GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=3, random_state=RANDOM_SEED),
    "hist_gradient_boosting_name_family": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.05, random_state=RANDOM_SEED),
}

for name, model in base_model_specs.items():
    fit_predict_model(
        resolver_name=name,
        resolver_family="supervised_row_model_base_features",
        model=model,
        feature_columns=BASE_FEATURE_COLUMNS,
        train_df=train_base,
        validate_df=validate_base,
        live_df=live_candidates_base,
        notes="Supervised candidate row scorer using Notebook 31-style numeric features.",
    )

for name, model in name_model_specs.items():
    fit_predict_model(
        resolver_name=name,
        resolver_family="supervised_row_model_name_family_features",
        model=model,
        feature_columns=NAME_FEATURE_COLUMNS,
        train_df=train_name,
        validate_df=validate_name,
        live_df=live_candidates_name,
        notes="Supervised row scorer with candidate-name and disease-family conflict features added.",
    )


def build_pairwise_training_matrix(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_negatives_per_positive: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    pair_blocks = []
    labels: list[int] = []
    for _, group in train_df.groupby("synthetic_state_id"):
        positives = group[group["candidate_label"].astype(int) == 1]
        negatives = group[group["candidate_label"].astype(int) == 0]
        if positives.empty or negatives.empty:
            continue
        if len(negatives) > max_negatives_per_positive:
            negatives = negatives.sample(max_negatives_per_positive, random_state=RANDOM_SEED)
        pos_x = clean_matrix(positives, feature_columns)
        neg_x = clean_matrix(negatives, feature_columns)
        for pos in pos_x:
            diff = pos - neg_x
            pair_blocks.append(diff)
            labels.extend([1] * len(diff))
            pair_blocks.append(-diff)
            labels.extend([0] * len(diff))
    return np.vstack(pair_blocks).astype(np.float32), np.asarray(labels, dtype=int)


def pairwise_mean_win_scores(model: Any, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    scores = np.zeros(len(df), dtype=float)
    matrix = clean_matrix(df, feature_columns)
    position_by_index = {idx: pos for pos, idx in enumerate(df.index)}
    for _, idxs in df.groupby("synthetic_state_id").groups.items():
        positions = np.asarray([position_by_index[idx] for idx in idxs])
        if len(positions) == 1:
            scores[positions[0]] = 1.0
            continue
        group_matrix = matrix[positions]
        for local_pos, global_pos in enumerate(positions):
            opponents = np.delete(group_matrix, local_pos, axis=0)
            scores[global_pos] = float(model.predict_proba(group_matrix[local_pos] - opponents)[:, 1].mean())
    return scores


pair_x, pair_y = build_pairwise_training_matrix(train_name, NAME_FEATURE_COLUMNS, max_negatives_per_positive=5)
pairwise_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs", class_weight="balanced", random_state=RANDOM_SEED))
pairwise_model.fit(pair_x, pair_y)
evaluate_scores(
    resolver_name="pairwise_logistic_mean_win_name_family",
    resolver_family="pairwise_ranker",
    validate_df=validate_name,
    validate_score=pairwise_mean_win_scores(pairwise_model, validate_name, NAME_FEATURE_COLUMNS),
    live_df=live_candidates_name,
    live_score=pairwise_mean_win_scores(pairwise_model, live_candidates_name, NAME_FEATURE_COLUMNS),
    notes="Listwise-ish pairwise logistic ranker: mean predicted win probability against pool peers.",
)


def differential_vote_scores(live_df: pd.DataFrame, *, confidence_weighted: bool = False) -> np.ndarray:
    score_by_key: dict[tuple[str, str], float] = {}
    for _, row in live_candidates_raw.iterrows():
        ranked = safe_parse_list(row.get("ranked_differential"))
        weight = 1.0
        if confidence_weighted:
            raw_weight = pd.to_numeric(row.get("final_confidence"), errors="coerce")
            weight = float(raw_weight) if not pd.isna(raw_weight) else 1.0
        for rank, diagnosis in enumerate(ranked, start=1):
            key = (row["case_id"], str(diagnosis))
            score_by_key[key] = score_by_key.get(key, 0.0) + weight / rank
    return np.asarray([score_by_key.get((row.case_id, row.candidate_pathology), 0.0) for row in live_df.itertuples()], dtype=float)


def prediction_vote_scores(live_df: pd.DataFrame) -> np.ndarray:
    vote_cols = [
        "llm_predicted_pathology",
        "mlp_predicted_pathology",
        "agreement_hybrid_predicted_pathology",
        "conservative_hybrid_predicted_pathology",
        "graph_top1",
        "bayes_top1",
    ]
    vote_cols = [col for col in vote_cols if col in live_candidates_raw.columns]
    score_by_key: dict[tuple[str, str], float] = {}
    for _, row in live_candidates_raw.iterrows():
        for col in vote_cols:
            diagnosis = row.get(col)
            if isinstance(diagnosis, str) and diagnosis:
                key = (row["case_id"], diagnosis)
                score_by_key[key] = score_by_key.get(key, 0.0) + 1.0
    return np.asarray([score_by_key.get((row.case_id, row.candidate_pathology), 0.0) for row in live_df.itertuples()], dtype=float)


diff_vote = differential_vote_scores(live_candidates_name, confidence_weighted=False)
diff_vote_conf = differential_vote_scores(live_candidates_name, confidence_weighted=True)
pred_vote = prediction_vote_scores(live_candidates_name)
evaluate_scores(
    resolver_name="branch_ranked_differential_reciprocal_vote",
    resolver_family="live_artifact_aggregation",
    live_df=live_candidates_name,
    live_score=diff_vote,
    validation_available=False,
    notes="Aggregates all branch ranked differentials by reciprocal rank. Diagnostic only: no synthetic validation analogue.",
)
evaluate_scores(
    resolver_name="branch_ranked_differential_confidence_vote",
    resolver_family="live_artifact_aggregation",
    live_df=live_candidates_name,
    live_score=diff_vote_conf,
    validation_available=False,
    notes="Confidence-weighted branch differential vote. Diagnostic only: no synthetic validation analogue.",
)
evaluate_scores(
    resolver_name="branch_prediction_vote",
    resolver_family="live_artifact_aggregation",
    live_df=live_candidates_name,
    live_score=pred_vote,
    validation_available=False,
    notes="Counts branch LLM, MLP, graph, Bayes, agreement, and conservative top-1 votes.",
)

nb31_neural_for_blend = None
if not nb31_candidate_scores.empty:
    aligned_nb31 = nb31_candidate_scores[["case_id", "branch_id", "candidate_role", "candidate_pathology", "neural_score"]].copy()
    nb31_neural_for_blend = live_candidates_name.merge(
        aligned_nb31,
        on=["case_id", "branch_id", "candidate_role", "candidate_pathology"],
        how="left",
    )["neural_score"].fillna(0).to_numpy()
    for weight in [0.05, 0.1, 0.25, 0.5]:
        evaluate_scores(
            resolver_name=f"notebook31_neural_plus_differential_vote_{weight:g}",
            resolver_family="live_artifact_aggregation_blend",
            live_df=live_candidates_name,
            live_score=nb31_neural_for_blend + weight * diff_vote,
            validation_available=False,
            notes="Diagnostic blend of Notebook 31 neural score and branch differential vote.",
        )


def one_shot_top5_candidate_scores(prediction_path: Path, live_df: pd.DataFrame) -> np.ndarray | None:
    if not prediction_path.exists() or "source_row_index" not in live_candidates_raw.columns:
        return None
    predictions = pd.read_csv(prediction_path)
    required = {"source_row_index", "top5_predictions", "top5_prediction_scores"}
    if not required.issubset(predictions.columns):
        return None
    score_map: dict[int, dict[str, float]] = {}
    for _, row in predictions.iterrows():
        diagnoses = safe_parse_list(row.get("top5_predictions"))
        scores = safe_parse_list(row.get("top5_prediction_scores"))
        score_map[int(row["source_row_index"])] = {str(dx): float(sc) for dx, sc in zip(diagnoses, scores)}
    live_with_source = live_df.merge(
        live_candidates_raw[["case_id", "branch_id", "candidate_role", "predicted_pathology", "source_row_index"]],
        left_on=["case_id", "branch_id", "candidate_role", "candidate_pathology"],
        right_on=["case_id", "branch_id", "candidate_role", "predicted_pathology"],
        how="left",
    )
    scores = []
    for _, row in live_with_source.iterrows():
        source_idx = row.get("source_row_index")
        if pd.isna(source_idx):
            scores.append(0.0)
            continue
        scores.append(score_map.get(int(source_idx), {}).get(row["candidate_pathology"], 0.0))
    return np.asarray(scores, dtype=float)


one_shot_paths = {
    "initial_evidence_pathology_top5": PROJECT_ROOT / "artifacts" / "one_shot" / "basd_pathology_full" / "predictions.csv",
    "initial_evidence_differential_top5": PROJECT_ROOT / "artifacts" / "one_shot" / "basd_differential_full" / "predictions.csv",
    "initial_evidence_joint_top5": PROJECT_ROOT / "artifacts" / "one_shot" / "basd_joint_full" / "predictions.csv",
    "full_evidence_pathology_top5_non_deployable": PROJECT_ROOT / "artifacts" / "one_shot_full_evidence" / "full_evidence_pathology_full" / "predictions.csv",
}

for name, path in one_shot_paths.items():
    score = one_shot_top5_candidate_scores(path, live_candidates_name)
    if score is None:
        continue
    uses_full_evidence = name.startswith("full_evidence")
    evaluate_scores(
        resolver_name=name,
        resolver_family="external_static_mlp_signal",
        live_df=live_candidates_name,
        live_score=score,
        validation_available=False,
        uses_unobserved_full_evidence=uses_full_evidence,
        notes="Candidate score from prior one-shot artifact top-5 probabilities." + (" Uses all unobserved evidence roots, so this is a non-deployable ceiling." if uses_full_evidence else ""),
    )
    if uses_full_evidence and nb31_neural_for_blend is not None:
        for weight in [0.25, 0.5, 1.0, 2.0]:
            evaluate_scores(
                resolver_name=f"notebook31_neural_plus_full_evidence_{weight:g}_non_deployable",
                resolver_family="external_full_evidence_blend_non_deployable",
                live_df=live_candidates_name,
                live_score=nb31_neural_for_blend + weight * score,
                validation_available=False,
                uses_unobserved_full_evidence=True,
                notes="Diagnostic blend with full-evidence model. Uses unobserved roots and is not a resolver-only policy.",
            )

evaluate_scores(
    resolver_name="candidate_pool_label_oracle_non_deployable",
    resolver_family="label_oracle",
    live_df=live_candidates_name,
    live_score=live_candidates_name["candidate_label"].astype(int).to_numpy(),
    validation_available=False,
    validatable_without_49_labels=False,
    uses_49_labels_for_selection=True,
    notes="Selects the true candidate using the held-out 49-case labels. Diagnostic ceiling only.",
)

summary_df = pd.DataFrame(results).drop_duplicates("resolver_name", keep="last")
if not nb31_summary_metrics.empty:
    mask = summary_df["resolver_name"].eq("notebook31_compact_neural_reference")
    if mask.any():
        nb31_correct = int(nb31_summary_metrics.loc[0, "num_correct"])
        nb31_cases = int(nb31_summary_metrics.loc[0, "num_cases"])
        summary_df.loc[mask, "live_correct"] = nb31_correct
        summary_df.loc[mask, "live_cases"] = nb31_cases
        summary_df.loc[mask, "live_accuracy"] = nb31_correct / nb31_cases

summary_df = summary_df.sort_values(
    ["live_correct", "validate_group_argmax_accuracy_all_groups", "validate_row_average_precision"],
    ascending=[False, False, False],
).reset_index(drop=True)

validatable_mask = (
    summary_df["validation_available"].astype(bool)
    & summary_df["validatable_without_49_labels"].astype(bool)
    & ~summary_df["uses_49_labels_for_selection"].astype(bool)
    & ~summary_df["uses_unobserved_full_evidence"].astype(bool)
)
validatable_df = summary_df[validatable_mask].copy()
best_validation_selected = validatable_df.sort_values(
    ["validate_group_argmax_accuracy_all_groups", "validate_row_average_precision", "resolver_name"],
    ascending=[False, False, True],
).iloc[0]
best_live_deployable_diagnostic = validatable_df.sort_values(
    ["live_correct", "validate_group_argmax_accuracy_all_groups", "validate_row_average_precision"],
    ascending=[False, False, False],
).iloc[0]
best_overall_diagnostic = summary_df.sort_values(["live_correct", "live_accuracy"], ascending=[False, False]).iloc[0]

summary_df.to_csv(ARTIFACT_ROOT / "resolver_ablation_summary.csv", index=False)
candidate_score_table.to_csv(ARTIFACT_ROOT / "candidate_level_resolver_ablation_scores.csv", index=False)
pool_oracle.to_csv(ARTIFACT_ROOT / "candidate_pool_oracle_summary.csv", index=False)
write_json(ARTIFACT_ROOT / "candidate_pool_oracle_summary.json", {
    "num_cases": int(len(pool_oracle)),
    "candidate_pool_oracle_correct": int(pool_oracle["true_in_candidate_pool"].sum()),
    "candidate_pool_oracle_accuracy": float(pool_oracle["true_in_candidate_pool"].mean()),
    "mean_candidate_rows": float(pool_oracle["candidate_rows"].mean()),
    "mean_unique_candidate_diagnoses": float(pool_oracle["unique_candidate_diagnoses"].mean()),
})

case_result_rows = []
for resolver_name, selected in selected_by_resolver.items():
    for _, row in selected.iterrows():
        case_result_rows.append({
            "resolver_name": resolver_name,
            "case_id": row.get("case_id"),
            "true_pathology": row.get("true_pathology"),
            "selected_pathology": row.get("candidate_pathology"),
            "selected_candidate_role": row.get("candidate_role"),
            "selected_branch_id": row.get("branch_id"),
            "correct": bool(row.get("candidate_label")),
        })
case_results_df = pd.DataFrame(case_result_rows)
case_results_df.to_csv(ARTIFACT_ROOT / "case_level_resolver_ablation_results.csv", index=False)

hard_case_ids = sorted(set(case_results_df.loc[~case_results_df["correct"].astype(bool), "case_id"].dropna().tolist() + ["test:111176", "test:11655", "test:62878"]))
key_resolvers = list(dict.fromkeys([
    "notebook31_compact_neural_reference",
    str(best_validation_selected["resolver_name"]),
    str(best_live_deployable_diagnostic["resolver_name"]),
    "notebook31_neural_plus_full_evidence_1_non_deployable",
    "candidate_pool_label_oracle_non_deployable",
]))
key_score_cols = [safe_score_column_name(name) for name in key_resolvers if safe_score_column_name(name) in candidate_score_table.columns]
hard_case_ladders = []
for case_id in hard_case_ids:
    subset = candidate_score_table[candidate_score_table["case_id"] == case_id].copy()
    for score_col in key_score_cols:
        ranked = subset.sort_values(score_col, ascending=False).head(10)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            hard_case_ladders.append({
                "case_id": case_id,
                "resolver_score_col": score_col,
                "rank": rank,
                "candidate_pathology": row["candidate_pathology"],
                "true_pathology": row["true_pathology"],
                "candidate_label": bool(row["candidate_label"]),
                "candidate_role": row["candidate_role"],
                "score": float(row[score_col]),
            })
hard_case_ladders_df = pd.DataFrame(hard_case_ladders)
hard_case_ladders_df.to_csv(ARTIFACT_ROOT / "hard_case_resolver_score_ladders.csv", index=False)
write_json(ARTIFACT_ROOT / "hard_case_resolver_audits.json", hard_case_ladders_df.to_dict(orient="records"))

selected_policy = {
    "run_name": RUN_NAME,
    "artifact_root": str(ARTIFACT_ROOT.resolve()),
    "selected_by_validation": best_validation_selected.to_dict(),
    "best_live_deployable_diagnostic": best_live_deployable_diagnostic.to_dict(),
    "best_overall_diagnostic": best_overall_diagnostic.to_dict(),
    "candidate_pool_oracle": {
        "correct": int(pool_oracle["true_in_candidate_pool"].sum()),
        "cases": int(len(pool_oracle)),
        "accuracy": float(pool_oracle["true_in_candidate_pool"].mean()),
    },
    "decision": "validation_selected_policy_did_not_exceed_notebook31_but_live_diagnostic_candidate_reached_47_of_49",
    "promotion_status": "do_not_promote_without_independent_confirmation",
    "important_caveat": "Rows using 49-case labels or full unobserved evidence are diagnostic ceilings, not deployable resolver policies.",
}
write_json(ARTIFACT_ROOT / "selected_resolver_ablation_policy.json", selected_policy)

resolved_run_config = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "offline_only": True,
    "additional_api_calls": 0,
    "notebook30_artifact_root": str(NOTEBOOK30_ROOT.resolve()),
    "notebook31_artifact_root": str(NOTEBOOK31_ROOT.resolve()),
    "artifact_root": str(ARTIFACT_ROOT.resolve()),
    "base_feature_count": int(len(BASE_FEATURE_COLUMNS)),
    "name_family_feature_count": int(len(NAME_FEATURE_COLUMNS)),
    "resolver_families_tested": sorted(summary_df["resolver_family"].dropna().unique().tolist()),
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

summary_metrics = pd.DataFrame([{
    "num_cases": int(live_candidates_raw["case_id"].nunique()),
    "candidate_pool_oracle_correct": int(pool_oracle["true_in_candidate_pool"].sum()),
    "candidate_pool_oracle_accuracy": float(pool_oracle["true_in_candidate_pool"].mean()),
    "notebook30_reference_correct": int(nb30_selected["candidate_label"].astype(bool).sum()),
    "notebook31_reference_correct": int(nb31_summary_metrics.loc[0, "num_correct"]) if not nb31_summary_metrics.empty else np.nan,
    "best_validation_selected_resolver": best_validation_selected["resolver_name"],
    "best_validation_selected_live_correct": int(best_validation_selected["live_correct"]),
    "best_live_deployable_diagnostic_resolver": best_live_deployable_diagnostic["resolver_name"],
    "best_live_deployable_diagnostic_correct": int(best_live_deployable_diagnostic["live_correct"]),
    "best_overall_diagnostic_resolver": best_overall_diagnostic["resolver_name"],
    "best_overall_diagnostic_correct": int(best_overall_diagnostic["live_correct"]),
    "additional_api_calls": 0,
    "total_api_calls_if_run_after_pool": int(notebook30_metrics.get("total_api_calls", 0)),
    "mean_selected_requests_if_run_after_pool": float(notebook30_metrics.get("mean_selected_requests", np.nan)),
    "mean_total_branch_requests_if_run_after_pool": float(notebook30_metrics.get("mean_total_branch_requests", np.nan)),
    "promotion_status": "do_not_promote_without_independent_confirmation",
}])
summary_metrics.to_csv(ARTIFACT_ROOT / "summary_metrics.csv", index=False)

plot_df = summary_df.copy()
plot_df["deployable_validatable"] = (
    plot_df["validation_available"].astype(bool)
    & plot_df["validatable_without_49_labels"].astype(bool)
    & ~plot_df["uses_49_labels_for_selection"].astype(bool)
    & ~plot_df["uses_unobserved_full_evidence"].astype(bool)
)

fig, ax = plt.subplots(figsize=(10, 7))
top_plot = plot_df.sort_values(["live_correct", "deployable_validatable"], ascending=[False, False]).head(18).iloc[::-1]
colors = np.where(top_plot["deployable_validatable"], "#4C78A8", "#F58518")
ax.barh(top_plot["resolver_name"], top_plot["live_correct"], color=colors)
ax.axvline(46, color="#333333", linestyle="--", linewidth=1, label="Notebook 31 reference")
ax.axvline(49, color="#777777", linestyle=":", linewidth=1, label="Candidate-pool oracle")
ax.set_xlabel("Correct cases out of 49")
ax.set_title("Notebook 32 Resolver Ablation: Live Diagnostic Accuracy")
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "resolver_live_accuracy_comparison.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
valid_plot = plot_df[plot_df["validation_available"].astype(bool)].copy()
ax.scatter(
    valid_plot["validate_group_argmax_accuracy_all_groups"],
    valid_plot["live_correct"],
    c=np.where(valid_plot["deployable_validatable"], "#4C78A8", "#F58518"),
    alpha=0.85,
)
for _, row in valid_plot.sort_values("live_correct", ascending=False).head(8).iterrows():
    ax.annotate(row["resolver_name"].replace("_", " ")[:28], (row["validate_group_argmax_accuracy_all_groups"], row["live_correct"]), fontsize=7, alpha=0.8)
ax.set_xlabel("Synthetic validation group argmax accuracy")
ax.set_ylabel("Live correct cases")
ax.set_title("Validation Accuracy vs 49-Case Live Diagnostic Accuracy")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "validation_vs_live_accuracy.png", dpi=170)
plt.close(fig)

top_deployable = summary_df[
    summary_df["validation_available"].astype(bool)
    & summary_df["validatable_without_49_labels"].astype(bool)
    & ~summary_df["uses_49_labels_for_selection"].astype(bool)
    & ~summary_df["uses_unobserved_full_evidence"].astype(bool)
].sort_values(["live_correct", "validate_group_argmax_accuracy_all_groups"], ascending=[False, False]).head(10)
top_overall = summary_df.sort_values(["live_correct", "live_accuracy"], ascending=[False, False]).head(10)

report_lines = [
    "# Resolver Ablation Lab Report",
    "",
    f"Generated by `notebooks/32_resolver_ablation_lab.ipynb` on {datetime.now().isoformat(timespec='seconds')}.",
    "",
    "## Inputs",
    "",
    f"- Notebook 30 candidate pool: `{NOTEBOOK30_ROOT}`",
    f"- Notebook 31 reference resolver: `{NOTEBOOK31_ROOT}`",
    f"- New artifact root: `{ARTIFACT_ROOT}`",
    "- Offline only: no new API calls; all deployable resolver rows inherit Notebook 30's candidate-pool cost.",
    "",
    "## Key Results",
    "",
    f"- Candidate-pool oracle remains `{int(pool_oracle['true_in_candidate_pool'].sum())}/{len(pool_oracle)}`. This is a label-using ceiling, not a resolver policy.",
    f"- Notebook 30 hand resolver reference: `{int(nb30_selected['candidate_label'].astype(bool).sum())}/{int(nb30_selected['case_id'].nunique())}`.",
]
if not nb31_summary_metrics.empty:
    report_lines.append(f"- Notebook 31 compact neural reference: `{int(nb31_summary_metrics.loc[0, 'num_correct'])}/{int(nb31_summary_metrics.loc[0, 'num_cases'])}`.")
report_lines.extend([
    f"- Best validation-selected deployable resolver in this lab: `{best_validation_selected['resolver_name']}` at `{int(best_validation_selected['live_correct'])}/49` live diagnostic accuracy.",
    f"- Best deployable live diagnostic resolver found: `{best_live_deployable_diagnostic['resolver_name']}` at `{int(best_live_deployable_diagnostic['live_correct'])}/49`.",
    f"- Best overall diagnostic row: `{best_overall_diagnostic['resolver_name']}` at `{int(best_overall_diagnostic['live_correct'])}/49`; deployability flags: `uses_49_labels={bool(best_overall_diagnostic['uses_49_labels_for_selection'])}`, `uses_full_evidence={bool(best_overall_diagnostic['uses_unobserved_full_evidence'])}`.",
    "",
    "## Interpretation",
    "",
    "The strict validation-selected policy did not improve on Notebook 31: it selected a sparse balanced logistic resolver that scored `45/49` on the live diagnostic. The ablation sweep did uncover one deployable model, `gradient_boosting_name_family`, at `47/49`, but that row was identified by inspecting the 49-case outcomes across many ablations. Treat it as a candidate for independent confirmation, not as a promoted final resolver yet.",
    "",
    "The non-deployable full-evidence blend can reach `48/49` because it uses evidence roots that the Notebook 30/31 resolver did not pay to observe. That result is useful as a diagnostic signal: the remaining errors are information/resolution problems, not proof that a scalar graph/Bayes/neural reranker is sufficient.",
    "",
    "## Top Deployable Rows",
    "",
    top_deployable[["resolver_name", "resolver_family", "validate_group_argmax_accuracy_all_groups", "live_correct", "live_misses"]].to_markdown(index=False),
    "",
    "## Top Overall Diagnostic Rows",
    "",
    top_overall[["resolver_name", "resolver_family", "validation_available", "uses_49_labels_for_selection", "uses_unobserved_full_evidence", "live_correct", "live_misses"]].to_markdown(index=False),
    "",
    "## Recommended Next Step",
    "",
    "Stop treating the last three misses as a generic resolver-ranking problem. The next credible experiment should be a targeted close-confounder adjudicator that asks for or verifies discriminating evidence for the specific competing diagnoses before final selection. If that adjudicator uses extra API calls or hidden evidence, it must be reported as a different cost tier, not as a free resolver improvement.",
    "",
])
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print("Best validation-selected resolver:")
print(best_validation_selected[["resolver_name", "validate_group_argmax_accuracy_all_groups", "live_correct", "live_misses"]].to_string())
print("\nBest deployable live diagnostic resolver:")
print(best_live_deployable_diagnostic[["resolver_name", "validate_group_argmax_accuracy_all_groups", "live_correct", "live_misses"]].to_string())
print("\nBest overall diagnostic row:")
print(best_overall_diagnostic[["resolver_name", "live_correct", "uses_49_labels_for_selection", "uses_unobserved_full_evidence", "live_misses"]].to_string())
print("\nArtifacts written to", ARTIFACT_ROOT)
print("Report written to", REPORT_PATH)

try:
    display(summary_df[[
        "resolver_name",
        "resolver_family",
        "validation_available",
        "validatable_without_49_labels",
        "uses_49_labels_for_selection",
        "uses_unobserved_full_evidence",
        "validate_group_argmax_accuracy_all_groups",
        "live_correct",
        "live_accuracy",
        "live_misses",
    ]].head(25))
except NameError:
    print(summary_df[[
        "resolver_name",
        "resolver_family",
        "validation_available",
        "validatable_without_49_labels",
        "uses_49_labels_for_selection",
        "uses_unobserved_full_evidence",
        "validate_group_argmax_accuracy_all_groups",
        "live_correct",
        "live_accuracy",
        "live_misses",
    ]].head(25).to_string(index=False))
