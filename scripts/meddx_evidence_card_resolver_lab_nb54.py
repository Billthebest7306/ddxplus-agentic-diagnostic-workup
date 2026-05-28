from __future__ import annotations

# %% [markdown]
# # Notebook 54: MEDDx Evidence-Card Resolver Lab
#
# Notebook 53 raised candidate-pool recall from `809/900` to `866/900` by admitting
# independently supported candidates from saved LLM traces, DDXPlus Bayes, and RareBench visible-HPO
# similarity. This notebook is Stage 2: choose among that recovered pool.
#
# The selected resolver is intentionally modest:
#
# - train a label-free L2 logistic candidate scorer on case-blocked train cases only
# - select one override threshold on validation only
# - preserve the current answer unless the learned candidate clears that threshold
# - evaluate on the held-out case-blocked test split and the old Notebook 46 transfer artifact
#
# Oracle and full-fit rows are reported as diagnostics only.

# %%
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

pd.set_option("display.max_columns", 160)
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

POOL_RUN_NAME = "meddx_candidate_pool_recovery_lab_v1"
SCALE_RUN_NAME = "meddx_scale_hypothesis_branching_confirmation_v1_meddx100"
TRANSFER_RUN_NAME = "meddx_aligned_dataset_native_driver_v1_eval30"
RUN_NAME = "meddx_evidence_card_resolver_lab_v1"

POOL_ROOT = ROOT / "artifacts" / "universal_meddx" / POOL_RUN_NAME
SCALE_ROOT = ROOT / "artifacts" / "universal_meddx" / SCALE_RUN_NAME
TRANSFER_ROOT = ROOT / "artifacts" / "universal_meddx" / TRANSFER_RUN_NAME
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_POOL_POLICY = "saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1"
SELECTED_RESOLVER_POLICY = "recovered_pool_logistic_evidence_card_resolver_v1"
SELECTED_MODEL_NAME = "logistic_C2_balanced"
MODEL_GRID = {
    "logistic_C2_balanced": Pipeline(
        [
            ("preprocess", "passthrough"),
            (
                "classifier",
                LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    ),
    "logistic_C05_balanced": Pipeline(
        [
            ("preprocess", "passthrough"),
            (
                "classifier",
                LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    ),
    "extra_trees_diagnostic": Pipeline(
        [
            ("preprocess", "passthrough"),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=54,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "hgb_diagnostic": Pipeline(
        [
            ("preprocess", "passthrough"),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.05,
                    l2_regularization=0.10,
                    random_state=54,
                ),
            ),
        ]
    ),
}
PRIMARY_MODEL_NAMES = ["logistic_C2_balanced", "logistic_C05_balanced"]
THRESHOLD_GRID = [round(float(value), 2) for value in np.arange(0.0, 0.81, 0.02)] + [999.0]
OOF_FOLDS = 5
OOF_SEED = 540

print("Project root :", ROOT)
print("Pool input   :", POOL_ROOT)
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


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


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


def rank_in_list(label: Any, ranked: list[str], missing_rank: int = 99) -> int:
    key = normalize_label(label)
    for idx, item in enumerate(ranked, start=1):
        if normalize_label(item) == key:
            return idx
    return missing_rank


def topk_hit(ranked: list[str], truth: str, k: int) -> bool:
    truth_key = normalize_label(truth)
    return truth_key in {normalize_label(label) for label in ranked[:k]}


def unique_ranked(labels: list[str], limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = normalize_label(label)
        if key and key not in seen:
            seen.add(key)
            out.append(str(label))
    return out[:limit]


def current_ranked(row: pd.Series) -> list[str]:
    ranked = parse_json_list(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "").strip()
    if prediction and normalize_label(prediction) not in {normalize_label(label) for label in ranked}:
        ranked = [prediction, *ranked]
    return unique_ranked(ranked, 10)

# %% [markdown]
# ## 2. Load Pool, Prediction, And Candidate Artifacts

# %%
required = [
    POOL_ROOT / "expanded_candidate_pool_long.csv",
    POOL_ROOT / "case_level_candidate_pool_recovery_results.csv",
    POOL_ROOT / "case_split_assignment.csv",
    SCALE_ROOT / "predictions.csv",
    SCALE_ROOT / "candidate_level_resolver_scores.csv",
    TRANSFER_ROOT / "predictions.csv",
    TRANSFER_ROOT / "candidate_level_resolver_scores.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

expanded_candidates = pd.read_csv(POOL_ROOT / "expanded_candidate_pool_long.csv")
pool_case_results = pd.read_csv(POOL_ROOT / "case_level_candidate_pool_recovery_results.csv")
case_split = pd.read_csv(POOL_ROOT / "case_split_assignment.csv")
scale_predictions = pd.read_csv(SCALE_ROOT / "predictions.csv").assign(cohort="scale_meddx100")
transfer_predictions = pd.read_csv(TRANSFER_ROOT / "predictions.csv").assign(cohort="transfer_old90")
scale_current_candidates = pd.read_csv(SCALE_ROOT / "candidate_level_resolver_scores.csv").assign(cohort="scale_meddx100")
transfer_current_candidates = pd.read_csv(TRANSFER_ROOT / "candidate_level_resolver_scores.csv").assign(cohort="transfer_old90")

expanded_candidates = expanded_candidates[expanded_candidates["policy_name"].eq(SELECTED_POOL_POLICY)].copy()
pool_case_results = pool_case_results[pool_case_results["policy_name"].eq(SELECTED_POOL_POLICY)].copy()

display(pool_case_results.groupby(["cohort", "dataset_name", "budget"], as_index=False).agg(
    workups=("case_id", "size"),
    pool_recall=("candidate_pool_has_truth", "mean"),
    mean_pool_size=("candidate_pool_size", "mean"),
))

# %% [markdown]
# ## 3. Evidence-Card Candidate Features

# %%
SOURCE_NAMES = [
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


def build_candidate_feature_table() -> pd.DataFrame:
    predictions = pd.concat([scale_predictions, transfer_predictions], ignore_index=True)
    current_candidates = pd.concat([scale_current_candidates, transfer_current_candidates], ignore_index=True)
    current_candidates = current_candidates.copy()
    current_candidates["label_key"] = current_candidates["label"].map(normalize_label)
    current_candidates = current_candidates[
        ["cohort", "dataset_name", "case_id", "budget", "label_key", "candidate_rank", "resolver_score", "support_count"]
    ].copy()

    prediction_columns = [
        "cohort",
        "dataset_name",
        "case_id",
        "budget",
        "predicted_diagnosis",
        "llm_predicted_diagnosis",
        "casebase_prior_top_label",
        "rarebench_graph_top_label",
        "ddxplus_mlp_top1",
        "ddxplus_mlp_top5",
        "ranked_differential",
        "llm_ranked_differential",
        "confidence",
        "num_questions",
        "branch_count",
        "branch_question_count",
        "resolver_margin",
        "resolver_support_count",
        "ddxplus_mlp_confidence",
        "ddxplus_mlp_margin",
        "ddxplus_mlp_entropy",
        "rarebench_graph_top_score",
        "rarebench_graph_margin",
        "rarebench_graph_visible_phenotypes",
        "casebase_prior_top_score",
        "casebase_prior_margin",
        "correct_top1",
        "gtpa_at_3",
        "gtpa_at_5",
        "true_rank",
    ]
    frame = expanded_candidates.merge(
        predictions[[column for column in prediction_columns if column in predictions.columns]],
        on=["cohort", "dataset_name", "case_id", "budget"],
        how="left",
    )
    frame = frame.merge(
        current_candidates,
        on=["cohort", "dataset_name", "case_id", "budget", "label_key"],
        how="left",
    )
    for column, default in [("candidate_rank", 99), ("resolver_score", 0.0), ("support_count", 0.0)]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

    for source in SOURCE_NAMES:
        source_flag_column = f"has_source__{source}"
        frame[f"has_{source}"] = pd.to_numeric(frame.get(source_flag_column, 0), errors="coerce").fillna(0).astype(int)
        ranks: list[float] = []
        scores: list[float] = []
        for raw_ranks, raw_scores in zip(frame["source_ranks"], frame["source_scores"]):
            rank_map = parse_json_dict(raw_ranks)
            score_map = parse_json_dict(raw_scores)
            rank = safe_float(rank_map.get(source, 99), 99)
            score = safe_float(score_map.get(source, 0.0), 0.0)
            ranks.append(rank)
            scores.append(score)
        frame[f"rank_{source}"] = ranks
        frame[f"rr_{source}"] = [0.0 if rank >= 99 else 1.0 / max(rank, 1.0) for rank in ranks]
        frame[f"score_{source}"] = scores

    for list_column in ["ranked_differential", "llm_ranked_differential", "ddxplus_mlp_top5"]:
        if list_column not in frame.columns:
            continue
        ranks = [rank_in_list(label, parse_json_list(raw)) for label, raw in zip(frame["label"], frame[list_column])]
        frame[f"{list_column}_candidate_rank"] = ranks
        frame[f"{list_column}_candidate_rr"] = [0.0 if rank >= 99 else 1.0 / rank for rank in ranks]

    for reference_column in [
        "predicted_diagnosis",
        "llm_predicted_diagnosis",
        "casebase_prior_top_label",
        "rarebench_graph_top_label",
        "ddxplus_mlp_top1",
    ]:
        if reference_column in frame.columns:
            frame[f"is_{reference_column}"] = [
                int(normalize_label(label) == normalize_label(reference))
                for label, reference in zip(frame["label"], frame[reference_column])
            ]

    group_cols = ["cohort", "dataset_name", "case_id", "budget"]
    frame["workup_id"] = (
        frame["cohort"].astype(str)
        + "|"
        + frame["dataset_name"].astype(str)
        + "|"
        + frame["case_id"].astype(str)
        + "|"
        + frame["budget"].astype(int).astype(str)
    )
    frame["case_group"] = frame["cohort"].astype(str) + "|" + frame["dataset_name"].astype(str) + "|" + frame["case_id"].astype(str)
    frame["pool_size"] = frame.groupby(group_cols)["label"].transform("count")
    frame["added_rank_inv"] = 1.0 / pd.to_numeric(frame["candidate_rank_added_order"], errors="coerce").fillna(99).clip(lower=1)
    frame["current_rank_inv"] = 1.0 / pd.to_numeric(frame["candidate_rank"], errors="coerce").fillna(99).clip(lower=1)
    frame["is_current_prediction"] = (frame["candidate_rank"].astype(float) == 1.0).astype(int)
    frame["source_count"] = frame[[f"has_{source}" for source in SOURCE_NAMES]].sum(axis=1)

    for column in [
        "resolver_score",
        "score_ranked_diff_top10",
        "score_llm_diff_top10",
        "score_branch_top10",
        "score_ddxplus_visible_bayes_top10",
        "score_rarebench_visible_hpo_top10",
        "source_count",
    ]:
        if column not in frame.columns:
            continue
        group = frame.groupby(group_cols)[column]
        min_value = group.transform("min")
        max_value = group.transform("max")
        denom = (max_value - min_value).replace(0, np.nan)
        frame[f"{column}_pool_minmax"] = ((frame[column] - min_value) / denom).fillna(0.0)
        frame[f"{column}_rank_within_pool"] = frame.groupby(group_cols)[column].rank(ascending=False, method="min")
        frame[f"{column}_minus_pool_max"] = frame[column] - max_value

    return frame


candidate_features = build_candidate_feature_table()

numeric_feature_columns: list[str] = []
allowed_exact = {
    "candidate_rank_added_order",
    "candidate_rank",
    "resolver_score",
    "support_count",
    "budget",
    "confidence",
    "num_questions",
    "branch_count",
    "branch_question_count",
    "resolver_margin",
    "resolver_support_count",
    "ddxplus_mlp_confidence",
    "ddxplus_mlp_margin",
    "ddxplus_mlp_entropy",
    "rarebench_graph_top_score",
    "rarebench_graph_margin",
    "rarebench_graph_visible_phenotypes",
    "casebase_prior_top_score",
    "casebase_prior_margin",
    "pool_size",
    "added_rank_inv",
    "current_rank_inv",
    "is_current_prediction",
    "source_count",
}
for column in candidate_features.columns:
    if any(fragment in column.lower() for fragment in ["truth", "correct", "gtpa", "true_rank"]):
        continue
    if (
        column in allowed_exact
        or column.startswith(("has_", "rank_", "rr_", "score_", "is_"))
        or column.endswith(("_candidate_rank", "_candidate_rr", "_pool_minmax", "_rank_within_pool", "_minus_pool_max"))
    ):
        candidate_features[column] = pd.to_numeric(candidate_features[column], errors="coerce")
        if candidate_features[column].notna().any():
            numeric_feature_columns.append(column)

categorical_feature_columns = ["dataset_name"]
FEATURE_COLUMNS = numeric_feature_columns + categorical_feature_columns
FORBIDDEN_FEATURE_SUBSTRINGS = ("truth", "correct", "gtpa", "true_rank", "progress_improved")
leaky_features = [
    column for column in FEATURE_COLUMNS if any(fragment in column.lower() for fragment in FORBIDDEN_FEATURE_SUBSTRINGS)
]
if leaky_features:
    raise ValueError(f"Label-leakage feature columns detected: {leaky_features}")

preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_feature_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feature_columns),
    ]
)
for model in MODEL_GRID.values():
    model.steps[0] = ("preprocess", preprocess)

candidate_features.to_csv(ARTIFACT_ROOT / "candidate_level_evidence_card_features.csv", index=False)
write_json(ARTIFACT_ROOT / "feature_contract.json", {"feature_columns": FEATURE_COLUMNS})

print("Candidate rows:", len(candidate_features))
print("Feature columns:", len(FEATURE_COLUMNS))
display(candidate_features.head())

# %% [markdown]
# ## 4. Policy Scoring Utilities

# %%
CASE_KEYS = ["cohort", "dataset_name", "case_id", "budget"]


def ranked_from_group(group: pd.DataFrame, score_column: str, chosen_label: str) -> list[str]:
    labels = [chosen_label]
    labels.extend(
        group.sort_values([score_column, "resolver_score", "source_count"], ascending=[False, False, False])["label"].tolist()
    )
    return unique_ranked(labels, 10)


def current_case_results(predictions: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        ranked = current_ranked(row)
        truth = str(row["ground_truth_diagnosis"])
        rows.append(
            {
                "policy_name": policy_name,
                "cohort": row["cohort"],
                "dataset_name": row["dataset_name"],
                "case_id": row["case_id"],
                "budget": int(row["budget"]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": row["predicted_diagnosis"],
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": boolish(row["correct_top1"]),
                "gtpa_at_3": boolish(row["gtpa_at_3"]),
                "gtpa_at_5": boolish(row["gtpa_at_5"]),
                "true_rank": int(row["true_rank"]) if "true_rank" in row and not pd.isna(row["true_rank"]) else rank_in_list(truth, ranked, 11),
                "original_correct_top1": boolish(row["correct_top1"]),
                "changed_prediction": False,
                "policy_action": "current",
                "learned_top_label": "",
                "learned_top_score": np.nan,
                "current_candidate_score": np.nan,
                "score_delta_vs_current": np.nan,
                "candidate_pool_has_truth": np.nan,
                "candidate_pool_size": np.nan,
                "split": row.get("split", "transfer"),
            }
        )
    return pd.DataFrame(rows)


def resolver_policy_results(
    frame: pd.DataFrame,
    score_column: str,
    policy_name: str,
    threshold: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(CASE_KEYS, sort=False):
        group = group.copy()
        current_candidates = group[group["is_current_prediction"].eq(1)]
        if not len(current_candidates):
            current_candidates = group.sort_values("candidate_rank_added_order").head(1)
        current_candidate = current_candidates.iloc[0]
        learned_top = group.sort_values([score_column, "resolver_score", "source_count"], ascending=[False, False, False]).iloc[0]
        learned_score = safe_float(learned_top.get(score_column, 0.0), 0.0)
        current_score = safe_float(current_candidate.get(score_column, 0.0), 0.0)
        score_delta = learned_score - current_score
        choose_learned = True if threshold is None else (
            normalize_label(learned_top["label"]) != normalize_label(current_candidate["label"])
            and score_delta >= threshold
        )
        chosen = learned_top if choose_learned else current_candidate
        prediction = str(chosen["label"])
        truth = str(chosen["ground_truth_diagnosis"])
        ranked = ranked_from_group(group, score_column, prediction)
        rows.append(
            {
                "policy_name": policy_name,
                "cohort": key[0],
                "dataset_name": key[1],
                "case_id": key[2],
                "budget": int(key[3]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": prediction,
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": normalize_label(prediction) == normalize_label(truth),
                "gtpa_at_3": topk_hit(ranked, truth, 3),
                "gtpa_at_5": topk_hit(ranked, truth, 5),
                "true_rank": rank_in_list(truth, ranked, 11),
                "original_correct_top1": boolish(chosen.get("correct_top1", False)),
                "changed_prediction": normalize_label(prediction) != normalize_label(current_candidate["label"]),
                "policy_action": "learned_override" if choose_learned else "preserve_current",
                "learned_top_label": str(learned_top["label"]),
                "learned_top_score": learned_score,
                "current_candidate_score": current_score,
                "score_delta_vs_current": score_delta,
                "candidate_pool_has_truth": bool(group["is_truth_candidate"].max()),
                "candidate_pool_size": int(len(group)),
                "split": str(chosen.get("split", "transfer")),
            }
        )
    return pd.DataFrame(rows)


def candidate_pool_oracle_results(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(CASE_KEYS, sort=False):
        truth_rows = group[group["is_truth_candidate"].eq(1)]
        if len(truth_rows):
            chosen = truth_rows.sort_values("candidate_rank_added_order").iloc[0]
            prediction = str(chosen["label"])
            action = "oracle_truth_in_pool"
        else:
            chosen = group.sort_values("candidate_rank_added_order").iloc[0]
            prediction = str(chosen["label"])
            action = "oracle_pool_miss"
        truth = str(chosen["ground_truth_diagnosis"])
        ranked = ranked_from_group(group.assign(oracle_score=0.0), "oracle_score", prediction)
        rows.append(
            {
                "policy_name": policy_name,
                "cohort": key[0],
                "dataset_name": key[1],
                "case_id": key[2],
                "budget": int(key[3]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": prediction,
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": normalize_label(prediction) == normalize_label(truth),
                "gtpa_at_3": topk_hit(ranked, truth, 3),
                "gtpa_at_5": topk_hit(ranked, truth, 5),
                "true_rank": rank_in_list(truth, ranked, 11),
                "original_correct_top1": boolish(chosen.get("correct_top1", False)),
                "changed_prediction": normalize_label(prediction) != normalize_label(chosen.get("predicted_diagnosis", "")),
                "policy_action": action,
                "learned_top_label": "",
                "learned_top_score": np.nan,
                "current_candidate_score": np.nan,
                "score_delta_vs_current": np.nan,
                "candidate_pool_has_truth": bool(group["is_truth_candidate"].max()),
                "candidate_pool_size": int(len(group)),
                "split": str(chosen.get("split", "transfer")),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slices = [("ALL", -1, "ALL", frame)]
    slices.extend(
        (dataset_name, int(budget), "ALL", group)
        for (dataset_name, budget), group in frame.groupby(["dataset_name", "budget"], sort=True)
    )
    if "split" in frame:
        slices.extend(("ALL", -1, split, group) for split, group in frame.groupby("split", sort=True))
    for dataset_name, budget, split, group in slices:
        if not len(group):
            continue
        rows.append(
            {
                "policy_name": policy_name,
                "cohort": str(group["cohort"].iloc[0]),
                "dataset_name": dataset_name,
                "budget": int(budget),
                "split": split,
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
                "candidate_pool_recall": float(group["candidate_pool_has_truth"].mean())
                if group["candidate_pool_has_truth"].notna().any()
                else np.nan,
                "mean_candidate_pool_size": float(group["candidate_pool_size"].mean())
                if group["candidate_pool_size"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)

# %% [markdown]
# ## 5. Train/Validate/Test Resolver Selection

# %%
scale_features = candidate_features[candidate_features["cohort"].eq("scale_meddx100")].copy()
transfer_features = candidate_features[candidate_features["cohort"].eq("transfer_old90")].copy()
train_features = scale_features[scale_features["split"].eq("train")].copy()
validate_features = scale_features[scale_features["split"].eq("validate")].copy()
test_features = scale_features[scale_features["split"].eq("test")].copy()

baseline_scale_current = current_case_results(scale_predictions.merge(case_split, on=["dataset_name", "case_id"], how="left"), "notebook51_current_live")
baseline_transfer_current = current_case_results(transfer_predictions.assign(split="transfer"), "notebook46_current_live")
scale_oracle = candidate_pool_oracle_results(scale_features, "recovered_candidate_pool_oracle_non_deployable")
transfer_oracle = candidate_pool_oracle_results(transfer_features, "transfer_recovered_candidate_pool_oracle_non_deployable")


def fit_candidate_model(model_name: str, train_frame: pd.DataFrame) -> Any:
    model = clone(MODEL_GRID[model_name])
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["is_truth_candidate"].astype(int))
    return model


def add_model_score(model: Any, frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = frame.copy()
    out[score_column] = model.predict_proba(out[FEATURE_COLUMNS])[:, 1]
    return out


validation_rows: list[dict[str, Any]] = []
fitted_models: dict[str, Any] = {}
scored_frames: dict[tuple[str, str], pd.DataFrame] = {}
for model_name in MODEL_GRID:
    model = fit_candidate_model(model_name, train_features)
    fitted_models[model_name] = model
    score_column = f"{model_name}_score"
    for frame_name, frame in [
        ("train", train_features),
        ("validate", validate_features),
        ("test", test_features),
        ("scale_all", scale_features),
        ("transfer", transfer_features),
    ]:
        scored_frames[(model_name, frame_name)] = add_model_score(model, frame, score_column)
    for threshold in THRESHOLD_GRID:
        validate_results = resolver_policy_results(
            scored_frames[(model_name, "validate")],
            score_column,
            f"{model_name}_threshold_{threshold}",
            threshold,
        )
        validation_rows.append(
            {
                "model_name": model_name,
                "threshold": threshold,
                "validate_top1_count": int(validate_results["correct_top1"].sum()),
                "validate_top1": float(validate_results["correct_top1"].mean()),
                "validate_top3": float(validate_results["gtpa_at_3"].mean()),
                "validate_top5": float(validate_results["gtpa_at_5"].mean()),
                "wins_vs_current": int((validate_results["correct_top1"] & ~validate_results["original_correct_top1"]).sum()),
                "regressions_vs_current": int((~validate_results["correct_top1"] & validate_results["original_correct_top1"]).sum()),
                "changed_predictions": int(validate_results["changed_prediction"].sum()),
                "is_primary_model": model_name in PRIMARY_MODEL_NAMES,
            }
        )

validation_threshold_sweep = pd.DataFrame(validation_rows)
validation_threshold_sweep.to_csv(ARTIFACT_ROOT / "resolver_validation_threshold_sweep.csv", index=False)

primary_sweep = validation_threshold_sweep[validation_threshold_sweep["is_primary_model"]].copy()
primary_sweep["selection_tuple"] = list(
    zip(
        primary_sweep["validate_top1_count"],
        -primary_sweep["regressions_vs_current"],
        primary_sweep["wins_vs_current"],
        -primary_sweep["changed_predictions"],
        -primary_sweep["threshold"],
        primary_sweep["model_name"].eq(SELECTED_MODEL_NAME).astype(int),
    )
)
selected_row = primary_sweep.sort_values(
    ["validate_top1_count", "regressions_vs_current", "wins_vs_current", "changed_predictions", "threshold"],
    ascending=[False, True, False, True, False],
).iloc[0]
selected_model_name = str(selected_row["model_name"])
selected_threshold = float(selected_row["threshold"])
selected_score_column = f"{selected_model_name}_score"

print("Selected model:", selected_model_name)
print("Selected threshold:", selected_threshold)
display(validation_threshold_sweep.sort_values(["validate_top1_count", "regressions_vs_current"], ascending=[False, True]).head(20))

# %% [markdown]
# ## 6. Selected Resolver Evaluation

# %%
selected_train = resolver_policy_results(
    scored_frames[(selected_model_name, "train")],
    selected_score_column,
    SELECTED_RESOLVER_POLICY,
    selected_threshold,
)
selected_validate = resolver_policy_results(
    scored_frames[(selected_model_name, "validate")],
    selected_score_column,
    SELECTED_RESOLVER_POLICY,
    selected_threshold,
)
selected_test = resolver_policy_results(
    scored_frames[(selected_model_name, "test")],
    selected_score_column,
    SELECTED_RESOLVER_POLICY,
    selected_threshold,
)
selected_scale_all = resolver_policy_results(
    scored_frames[(selected_model_name, "scale_all")],
    selected_score_column,
    SELECTED_RESOLVER_POLICY,
    selected_threshold,
)
selected_transfer = resolver_policy_results(
    scored_frames[(selected_model_name, "transfer")],
    selected_score_column,
    SELECTED_RESOLVER_POLICY,
    selected_threshold,
)

selected_train.to_csv(ARTIFACT_ROOT / "selected_train_case_results.csv", index=False)
selected_validate.to_csv(ARTIFACT_ROOT / "selected_validate_case_results.csv", index=False)
selected_test.to_csv(ARTIFACT_ROOT / "case_blocked_test_case_results.csv", index=False)
selected_scale_all.to_csv(ARTIFACT_ROOT / "selected_resolver_case_results.csv", index=False)
selected_transfer.to_csv(ARTIFACT_ROOT / "transfer_old90_resolver_results.csv", index=False)

selected_scored_scale = scored_frames[(selected_model_name, "scale_all")].copy()
selected_scored_transfer = scored_frames[(selected_model_name, "transfer")].copy()
selected_scored_scale.to_csv(ARTIFACT_ROOT / "candidate_level_scale_resolver_scores.csv", index=False)
selected_scored_transfer.to_csv(ARTIFACT_ROOT / "candidate_level_transfer_resolver_scores.csv", index=False)
pd.concat([selected_scored_scale, selected_scored_transfer], ignore_index=True).to_csv(
    ARTIFACT_ROOT / "candidate_level_resolver_scores.csv",
    index=False,
)

policy_summary = pd.concat(
    [
        summarize_policy(baseline_scale_current, "notebook51_current_live"),
        summarize_policy(scale_oracle, "recovered_candidate_pool_oracle_non_deployable"),
        summarize_policy(selected_scale_all, SELECTED_RESOLVER_POLICY),
        summarize_policy(baseline_transfer_current, "notebook46_current_live"),
        summarize_policy(transfer_oracle, "transfer_recovered_candidate_pool_oracle_non_deployable"),
        summarize_policy(selected_transfer, SELECTED_RESOLVER_POLICY),
    ],
    ignore_index=True,
)
resolver_policy_summary = policy_summary
resolver_policy_summary.to_csv(ARTIFACT_ROOT / "resolver_policy_summary.csv", index=False)

paired_current_vs_resolver = selected_scale_all[
    [
        "policy_name",
        "cohort",
        "dataset_name",
        "case_id",
        "budget",
        "split",
        "ground_truth_diagnosis",
        "policy_prediction",
        "correct_top1",
        "gtpa_at_3",
        "gtpa_at_5",
        "original_correct_top1",
        "changed_prediction",
        "policy_action",
        "learned_top_label",
        "learned_top_score",
        "current_candidate_score",
        "score_delta_vs_current",
        "candidate_pool_has_truth",
        "candidate_pool_size",
    ]
].copy()
paired_current_vs_resolver.to_csv(ARTIFACT_ROOT / "paired_current_vs_resolver.csv", index=False)

display(
    resolver_policy_summary[
        resolver_policy_summary["dataset_name"].eq("ALL")
        & resolver_policy_summary["budget"].eq(-1)
    ].sort_values(["cohort", "top1_count"], ascending=[True, False])
)

# %% [markdown]
# ## 7. Out-Of-Fold And Diagnostic Models

# %%
oof_rows: list[dict[str, Any]] = []
groups = scale_features["dataset_name"].astype(str) + "|" + scale_features["case_id"].astype(str)
y = scale_features["is_truth_candidate"].astype(int)
sgkf = StratifiedGroupKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=OOF_SEED)
for model_name in MODEL_GRID:
    oof_scores = np.full(len(scale_features), np.nan)
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(scale_features[FEATURE_COLUMNS], y, groups), start=1):
        model = fit_candidate_model(model_name, scale_features.iloc[train_idx])
        oof_scores[test_idx] = model.predict_proba(scale_features.iloc[test_idx][FEATURE_COLUMNS])[:, 1]
    scored = scale_features.copy()
    score_column = f"oof_{model_name}_score"
    scored[score_column] = oof_scores
    raw_results = resolver_policy_results(scored, score_column, f"oof_{model_name}_raw_diagnostic", None)
    thresholded_results = resolver_policy_results(scored, score_column, f"oof_{model_name}_selected_threshold_diagnostic", selected_threshold)
    for policy_name, results in [
        (f"oof_{model_name}_raw_diagnostic", raw_results),
        (f"oof_{model_name}_selected_threshold_diagnostic", thresholded_results),
    ]:
        oof_rows.append(
            {
                "policy_name": policy_name,
                "model_name": model_name,
                "threshold": np.nan if "raw" in policy_name else selected_threshold,
                "top1_count": int(results["correct_top1"].sum()),
                "top1": float(results["correct_top1"].mean()),
                "top3": float(results["gtpa_at_3"].mean()),
                "top5": float(results["gtpa_at_5"].mean()),
                "wins_vs_current": int((results["correct_top1"] & ~results["original_correct_top1"]).sum()),
                "regressions_vs_current": int((~results["correct_top1"] & results["original_correct_top1"]).sum()),
                "changed_predictions": int(results["changed_prediction"].sum()),
                "diagnostic_only": True,
            }
        )
oof_diagnostic_summary = pd.DataFrame(oof_rows)
oof_diagnostic_summary.to_csv(ARTIFACT_ROOT / "oof_diagnostic_resolver_summary.csv", index=False)
display(oof_diagnostic_summary.sort_values("top1_count", ascending=False))

# %% [markdown]
# ## 8. Figures

# %%
plt.style.use("seaborn-v0_8-whitegrid")

summary_all = resolver_policy_summary[
    (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["split"].eq("ALL"))
].copy()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(summary_all["policy_name"] + "\n" + summary_all["cohort"], summary_all["top1_count"], color="#416f9f")
ax.axhline(760, color="#c23b22", linestyle="--", linewidth=1.5, label="minimum final target 760/900")
ax.axhline(780, color="#2e8b57", linestyle="--", linewidth=1.5, label="strong final target 780/900")
ax.set_ylabel("Top-1 count")
ax.set_title("Resolver Top-1 Summary")
ax.tick_params(axis="x", rotation=35)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "resolver_top1_summary.png", dpi=180)
plt.close(fig)

selected_splits = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY))
    & (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["cohort"].eq("scale_meddx100"))
    & (resolver_policy_summary["split"].isin(["train", "validate", "test", "ALL"]))
].copy()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(selected_splits["split"], selected_splits["top1"], color="#7d5a9b")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Top-1 accuracy")
ax.set_title("Selected Resolver By Split")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "selected_resolver_by_split.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
changed = selected_scale_all[selected_scale_all["changed_prediction"]].copy()
if len(changed):
    counts = pd.Series(
        {
            "wins": int((changed["correct_top1"] & ~changed["original_correct_top1"]).sum()),
            "regressions": int((~changed["correct_top1"] & changed["original_correct_top1"]).sum()),
            "neutral": int((changed["correct_top1"] == changed["original_correct_top1"]).sum()),
        }
    )
    ax.bar(counts.index, counts.values, color=["#2e8b57", "#c23b22", "#8a8f99"])
else:
    ax.bar(["wins", "regressions", "neutral"], [0, 0, 0])
ax.set_ylabel("Changed workups")
ax.set_title("Selected Resolver Paired Changes")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "paired_wins_regressions.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
dataset_summary = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY))
    & (resolver_policy_summary["cohort"].eq("scale_meddx100"))
    & (resolver_policy_summary["dataset_name"].ne("ALL"))
    & (resolver_policy_summary["split"].eq("ALL"))
].copy()
dataset_summary["slice"] = dataset_summary["dataset_name"] + "@" + dataset_summary["budget"].astype(str)
ax.bar(dataset_summary["slice"], dataset_summary["top1"], color="#d28e39")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Top-1 accuracy")
ax.set_title("Selected Resolver By Dataset And Budget")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "selected_resolver_by_dataset_budget.png", dpi=180)
plt.close(fig)

# %% [markdown]
# ## 9. Final Summary And Artifact Contract

# %%
selected_validate_summary = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY))
    & (resolver_policy_summary["cohort"].eq("scale_meddx100"))
    & (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["split"].eq("validate"))
].iloc[0]
selected_test_summary = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY))
    & (resolver_policy_summary["cohort"].eq("scale_meddx100"))
    & (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["split"].eq("test"))
].iloc[0]
selected_all_summary = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY))
    & (resolver_policy_summary["cohort"].eq("scale_meddx100"))
    & (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["split"].eq("ALL"))
].iloc[0]
transfer_summary = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY))
    & (resolver_policy_summary["cohort"].eq("transfer_old90"))
    & (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["split"].eq("ALL"))
].iloc[0]
transfer_current_summary = resolver_policy_summary[
    (resolver_policy_summary["policy_name"].eq("notebook46_current_live"))
    & (resolver_policy_summary["cohort"].eq("transfer_old90"))
    & (resolver_policy_summary["dataset_name"].eq("ALL"))
    & (resolver_policy_summary["budget"].eq(-1))
    & (resolver_policy_summary["split"].eq("ALL"))
].iloc[0]

stage2_minimum_passed = bool(int(selected_test_summary["top1_count"]) >= 152)
transfer_nonnegative = bool(int(transfer_summary["top1_count"]) >= int(transfer_current_summary["top1_count"]))

selected_policy = {
    "selected_policy_name": SELECTED_RESOLVER_POLICY,
    "stage": "stage2_evidence_card_resolver",
    "selected_pool_policy": SELECTED_POOL_POLICY,
    "selected_model_name": selected_model_name,
    "selected_threshold": selected_threshold,
    "feature_columns": FEATURE_COLUMNS,
    "feature_guard": {
        "forbidden_substrings": FORBIDDEN_FEATURE_SUBSTRINGS,
        "leaky_features_detected": leaky_features,
    },
    "training_split": "scale_meddx100 train cases only",
    "threshold_selection": "scale_meddx100 validation cases only",
    "scale_validate": selected_validate_summary.to_dict(),
    "scale_test": selected_test_summary.to_dict(),
    "scale_all_diagnostic": selected_all_summary.to_dict(),
    "transfer_old90": transfer_summary.to_dict(),
    "promotion_decision": {
        "minimum_final_target_case_blocked_test": ">=152/180, equivalent to >=760/900",
        "stage2_minimum_passed": stage2_minimum_passed,
        "transfer_nonnegative": transfer_nonnegative,
        "decision": "promote_to_stage3_integrated_policy" if stage2_minimum_passed and transfer_nonnegative else "diagnostic_only_do_not_promote",
    },
    "diagnostic_caveat": (
        "The scale_all row includes train cases and is not the generalization claim. "
        "The promotion gate is the held-out case-blocked test split plus old90 transfer."
    ),
}
write_json(ARTIFACT_ROOT / "selected_policy.json", selected_policy)

resolved_run_config = {
    "run_name": RUN_NAME,
    "notebook": "notebooks/54_meddx_evidence_card_resolver_lab.ipynb",
    "script": "scripts/meddx_evidence_card_resolver_lab_nb54.py",
    "pool_input": POOL_RUN_NAME,
    "scale_input": SCALE_RUN_NAME,
    "transfer_input": TRANSFER_RUN_NAME,
    "selected_pool_policy": SELECTED_POOL_POLICY,
    "selected_resolver_policy": SELECTED_RESOLVER_POLICY,
    "no_api_calls": True,
    "artifact_root": str(ARTIFACT_ROOT.relative_to(ROOT)),
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

hard_case_audits = {
    "selected_test_failures": selected_test[~selected_test["correct_top1"]].head(80).to_dict(orient="records"),
    "selected_test_wins": selected_test[selected_test["correct_top1"] & ~selected_test["original_correct_top1"]].head(80).to_dict(orient="records"),
    "selected_test_regressions": selected_test[~selected_test["correct_top1"] & selected_test["original_correct_top1"]].head(80).to_dict(orient="records"),
    "transfer_regressions": selected_transfer[
        ~selected_transfer["correct_top1"] & selected_transfer["original_correct_top1"]
    ].head(40).to_dict(orient="records"),
}
write_json(ARTIFACT_ROOT / "hard_case_audits.json", hard_case_audits)

print(json.dumps(selected_policy["promotion_decision"], indent=2))
display(
    resolver_policy_summary[
        resolver_policy_summary["policy_name"].isin(
            ["notebook51_current_live", "notebook46_current_live", SELECTED_RESOLVER_POLICY]
        )
        & resolver_policy_summary["dataset_name"].eq("ALL")
        & resolver_policy_summary["budget"].eq(-1)
    ].sort_values(["cohort", "policy_name", "split"])
)
