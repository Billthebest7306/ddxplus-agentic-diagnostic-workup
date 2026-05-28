from __future__ import annotations

# %% [markdown]
# # Notebook 52: MEDDx-Scale Offline Resolver Calibration
#
# Notebook 51 produced the first MEDDx-scale live artifact for this project:
# 100 cases per dataset, budgets 5/10/15, and 900 total workups.
#
# The live result was disappointing relative to the earlier 90-workup pilot. This notebook keeps the live traces
# frozen and asks:
#
# 1. Is the new failure mostly candidate generation or final candidate-pool resolution?
# 2. Can a learned offline resolver calibrated on the new MEDDx-scale cohort improve held-out scale cases?
# 3. If trained on the new large cohort, does the frozen resolver transfer back to the older 90-workup artifact?
#
# The notebook makes no API calls. It is a calibration and audit lab, not a live controller.

# %%
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 120)

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
RUN_NAME = "meddx_scale_offline_resolver_calibration_v1"

SCALE_ROOT = ROOT / "artifacts" / "universal_meddx" / SCALE_RUN_NAME
TRANSFER_ROOT = ROOT / "artifacts" / "universal_meddx" / TRANSFER_RUN_NAME
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 52
OOF_SEED = 520
TRAIN_FRACTION = 0.60
VALIDATE_FRACTION = 0.20
TEST_FRACTION = 0.20
SELECTED_MODEL_NAME = "scale_logistic_candidate_resolver_v1"
SELECTED_MODEL = Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                max_iter=3000,
                C=0.30,
                class_weight="balanced",
                solver="lbfgs",
                random_state=SPLIT_SEED,
            ),
        ),
    ]
)
THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.0, 0.81, 0.05)] + [999.0]

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
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
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
    text = text.replace("é", "e").replace("è", "e").replace("á", "a")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
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


def source_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def rank_in_list(label: Any, ranked: list[str], missing_rank: int = 99) -> int:
    key = normalize_label(label)
    for idx, item in enumerate(ranked, start=1):
        if normalize_label(item) == key:
            return idx
    return missing_rank


def topk_hit(ranked: list[str], truth: str, k: int) -> bool:
    truth_key = normalize_label(truth)
    return truth_key in {normalize_label(label) for label in ranked[:k]}


def original_ranked(row: pd.Series) -> list[str]:
    ranked = parse_json_list(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "")
    if prediction and normalize_label(prediction) not in {normalize_label(label) for label in ranked}:
        ranked = [prediction, *ranked]
    return ranked[:10]


def ranked_from_candidates(group: pd.DataFrame, score_column: str, first_label: str | None = None) -> list[str]:
    labels = group.sort_values([score_column, "resolver_score"], ascending=[False, False])["label"].tolist()
    ordered = [str(first_label)] if first_label else []
    ordered.extend(str(label) for label in labels)
    out: list[str] = []
    seen: set[str] = set()
    for label in ordered:
        key = normalize_label(label)
        if key and key not in seen:
            out.append(label)
            seen.add(key)
    return out[:10]

# %% [markdown]
# ## 2. Load Live Artifacts And Build Candidate Features

# %%
REQUIRED_INPUTS = [
    SCALE_ROOT / "predictions.csv",
    SCALE_ROOT / "candidate_level_resolver_scores.csv",
    TRANSFER_ROOT / "predictions.csv",
    TRANSFER_ROOT / "candidate_level_resolver_scores.csv",
]
missing = [str(path) for path in REQUIRED_INPUTS if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required input artifacts: {missing}")


def build_candidate_features(root: Path, cohort_name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    predictions = pd.read_csv(root / "predictions.csv")
    candidates = pd.read_csv(root / "candidate_level_resolver_scores.csv")
    frame = candidates.merge(
        predictions,
        on=["dataset_name", "case_id", "budget"],
        how="left",
        suffixes=("", "_prediction"),
    )
    frame["cohort"] = cohort_name
    frame["workup_id"] = (
        frame["dataset_name"].astype(str) + "|" + frame["case_id"].astype(str) + "|" + frame["budget"].astype(int).astype(str)
    )
    frame["case_group"] = frame["dataset_name"].astype(str) + "|" + frame["case_id"].astype(str)
    frame["label_key"] = frame["label"].map(normalize_label)
    frame["truth_key"] = frame["ground_truth_diagnosis"].map(normalize_label)
    frame["is_truth_candidate"] = (frame["label_key"] == frame["truth_key"]).astype(int)
    frame["original_correct_top1"] = frame["correct_top1"].map(boolish)
    frame["candidate_rank"] = pd.to_numeric(frame["candidate_rank"], errors="coerce").fillna(99).astype(int)

    source_names = [
        "base_llm_rank",
        "base_llm_confidence",
        "casebase_prior",
        "rarebench_graph",
        "branch:branch_1",
        "branch:branch_2",
        "branch_conf:branch_1",
        "branch_conf:branch_2",
    ]
    for source_name in source_names:
        key = source_key(source_name)
        frame[f"src_{key}"] = 0.0
        frame[f"has_{key}"] = 0

    for row_index, raw_sources in frame["sources"].items():
        try:
            parsed = json.loads(raw_sources) if isinstance(raw_sources, str) else []
        except Exception:
            parsed = []
        for item in parsed:
            source_name = item.get("source")
            if source_name not in source_names:
                continue
            key = source_key(source_name)
            value = safe_float(item.get("value", 0.0))
            frame.at[row_index, f"src_{key}"] = max(safe_float(frame.at[row_index, f"src_{key}"]), value)
            frame.at[row_index, f"has_{key}"] = 1

    for column in [
        "predicted_diagnosis",
        "llm_predicted_diagnosis",
        "casebase_prior_top_label",
        "rarebench_graph_top_label",
        "ddxplus_mlp_top1",
    ]:
        frame[f"is_{column}"] = [
            int(normalize_label(label) == normalize_label(reference))
            for label, reference in zip(frame["label"], frame[column])
        ]

    frame["in_ddxplus_mlp_top5"] = [
        int(normalize_label(label) in {normalize_label(item) for item in parse_json_list(raw)})
        for label, raw in zip(frame["label"], frame.get("ddxplus_mlp_top5", pd.Series(["[]"] * len(frame))))
    ]
    for column in ["ranked_differential", "llm_ranked_differential"]:
        rank_column = f"{column}_candidate_rank"
        frame[rank_column] = [
            rank_in_list(label, parse_json_list(raw))
            for label, raw in zip(frame["label"], frame[column])
        ]
        frame[f"in_{column}"] = (frame[rank_column] < 99).astype(int)

    for dataset_name in ["ddxplus", "icraft_md", "rarebench"]:
        frame[f"dataset_{dataset_name}"] = (frame["dataset_name"].astype(str) == dataset_name).astype(int)

    group_cols = ["dataset_name", "case_id", "budget"]
    frame["pool_size"] = frame.groupby(group_cols)["label"].transform("count")
    frame["pool_unique_size"] = frame.groupby(group_cols)["label"].transform("nunique")
    frame["candidate_pool_has_truth"] = frame.groupby(group_cols)["is_truth_candidate"].transform("max").astype(bool)
    frame["max_resolver_score"] = frame.groupby(group_cols)["resolver_score"].transform("max")
    frame["score_gap_to_current"] = frame["max_resolver_score"] - frame["resolver_score"]
    frame["score_share"] = (
        frame["resolver_score"] / frame.groupby(group_cols)["resolver_score"].transform("sum").replace(0, np.nan)
    ).fillna(0.0)
    frame["rank_inv"] = 1.0 / frame["candidate_rank"].clip(lower=1).astype(float)
    source_presence_cols = [f"has_{source_key(name)}" for name in source_names]
    frame["independent_source_count"] = frame[source_presence_cols].sum(axis=1)
    frame["is_current_prediction"] = (frame["candidate_rank"] == 1).astype(int)

    numeric_columns = []
    allowed_exact = {
        "candidate_rank",
        "rank_inv",
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
        "pool_unique_size",
        "max_resolver_score",
        "score_gap_to_current",
        "score_share",
        "independent_source_count",
        "ranked_differential_candidate_rank",
        "llm_ranked_differential_candidate_rank",
    }
    prefixes = ("src_", "has_", "is_", "in_", "dataset_")
    forbidden_feature_columns = {
        "is_truth_candidate",
        "candidate_pool_has_truth",
        "original_correct_top1",
        "correct_top1",
        "gtpa_at_3",
        "gtpa_at_5",
        "true_rank",
        "initial_true_rank",
        "progress_improved",
    }
    for column in frame.columns:
        if column in forbidden_feature_columns:
            continue
        if column in allowed_exact or (column.startswith(prefixes) and column != "dataset_name"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                frame[column] = values.fillna(0.0)
                numeric_columns.append(column)

    workups = predictions.copy()
    workups["workup_id"] = (
        workups["dataset_name"].astype(str) + "|" + workups["case_id"].astype(str) + "|" + workups["budget"].astype(int).astype(str)
    )
    workups["case_group"] = workups["dataset_name"].astype(str) + "|" + workups["case_id"].astype(str)
    pool_summary = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            candidate_pool_has_truth=("is_truth_candidate", "max"),
            candidate_pool_size=("label", "nunique"),
            truth_pool_rank=("candidate_rank", lambda values: int(values.min()) if len(values) else 999),
        )
    )
    truth_ranks = (
        frame[frame["is_truth_candidate"].eq(1)]
        .groupby(group_cols, as_index=False)["candidate_rank"]
        .min()
        .rename(columns={"candidate_rank": "truth_pool_rank_true_only"})
    )
    pool_summary = pool_summary.merge(truth_ranks, on=group_cols, how="left")
    pool_summary["truth_pool_rank"] = pool_summary["truth_pool_rank_true_only"].fillna(999).astype(int)
    pool_summary = pool_summary.drop(columns=["truth_pool_rank_true_only"])
    workups = workups.merge(pool_summary, on=group_cols, how="left")
    workups["candidate_pool_has_truth"] = workups["candidate_pool_has_truth"].fillna(0).astype(bool)
    return frame, workups, sorted(set(numeric_columns))


scale_candidates, scale_workups, scale_features = build_candidate_features(SCALE_ROOT, "scale_meddx100")
transfer_candidates, transfer_workups, transfer_features = build_candidate_features(TRANSFER_ROOT, "transfer_old90")
FEATURE_COLUMNS = [column for column in scale_features if column in set(transfer_features)]
FORBIDDEN_FEATURE_SUBSTRINGS = ("truth", "correct", "gtpa", "true_rank", "progress_improved")
leaky_features = [
    column
    for column in FEATURE_COLUMNS
    if any(fragment in column.lower() for fragment in FORBIDDEN_FEATURE_SUBSTRINGS)
]
if leaky_features:
    raise ValueError(f"Label-leakage feature columns detected: {leaky_features}")

scale_candidates.to_csv(ARTIFACT_ROOT / "candidate_level_scale_features.csv", index=False)
transfer_candidates.to_csv(ARTIFACT_ROOT / "candidate_level_transfer_features.csv", index=False)
scale_workups.to_csv(ARTIFACT_ROOT / "scale_workup_diagnostics.csv", index=False)
transfer_workups.to_csv(ARTIFACT_ROOT / "transfer_workup_diagnostics.csv", index=False)

print("Scale workups:", len(scale_workups), "unique cases:", scale_workups["case_group"].nunique())
print("Transfer workups:", len(transfer_workups), "unique cases:", transfer_workups["case_group"].nunique())
print("Candidate features:", len(FEATURE_COLUMNS))
display(scale_workups.groupby(["dataset_name", "budget"]).agg(
    n=("case_id", "size"),
    top1=("correct_top1", lambda s: s.map(boolish).mean()),
    top3=("gtpa_at_3", lambda s: s.map(boolish).mean()),
    top5=("gtpa_at_5", lambda s: s.map(boolish).mean()),
    pool_recall=("candidate_pool_has_truth", "mean"),
    mean_questions=("num_questions", "mean"),
))

# %% [markdown]
# ## 3. Candidate-Pool Ceilings And Baseline Summaries

# %%
def summarize_cases(frame: pd.DataFrame, policy_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slices = [("ALL", -1, frame)]
    slices.extend(
        (dataset_name, int(budget), group)
        for (dataset_name, budget), group in frame.groupby(["dataset_name", "budget"], sort=True)
    )
    for dataset_name, budget, group in slices:
        rows.append(
            {
                "policy_name": policy_name,
                "dataset_name": dataset_name,
                "budget": int(budget),
                "num_workups": int(len(group)),
                "top1_count": int(group["correct_top1"].sum()),
                "top1": float(group["correct_top1"].mean()) if len(group) else np.nan,
                "top3_count": int(group["gtpa_at_3"].sum()),
                "top3": float(group["gtpa_at_3"].mean()) if len(group) else np.nan,
                "top5_count": int(group["gtpa_at_5"].sum()),
                "top5": float(group["gtpa_at_5"].mean()) if len(group) else np.nan,
                "wins_vs_current": int((group["correct_top1"] & ~group["original_correct_top1"]).sum())
                if "original_correct_top1" in group
                else 0,
                "regressions_vs_current": int((~group["correct_top1"] & group["original_correct_top1"]).sum())
                if "original_correct_top1" in group
                else 0,
                "changed_predictions": int(group["changed_prediction"].sum()) if "changed_prediction" in group else 0,
                "mean_questions": float(pd.to_numeric(group.get("num_questions", pd.Series(dtype=float)), errors="coerce").mean())
                if "num_questions" in group
                else np.nan,
                "candidate_pool_recall": float(group["candidate_pool_has_truth"].mean())
                if "candidate_pool_has_truth" in group
                else np.nan,
            }
        )
    return rows


def current_policy_results(workups: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows = []
    for _, row in workups.iterrows():
        ranked = original_ranked(row)
        truth = str(row["ground_truth_diagnosis"])
        rows.append(
            {
                "policy_name": policy_name,
                "dataset_name": row["dataset_name"],
                "case_id": row["case_id"],
                "budget": int(row["budget"]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": row["predicted_diagnosis"],
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": boolish(row["correct_top1"]),
                "gtpa_at_3": boolish(row["gtpa_at_3"]),
                "gtpa_at_5": boolish(row["gtpa_at_5"]),
                "true_rank": int(row["true_rank"]),
                "original_correct_top1": boolish(row["correct_top1"]),
                "changed_prediction": False,
                "candidate_pool_has_truth": boolish(row["candidate_pool_has_truth"]),
                "candidate_pool_size": int(row["candidate_pool_size"]),
                "num_questions": safe_float(row.get("num_questions", np.nan), np.nan),
                "policy_action": "current",
            }
        )
    return pd.DataFrame(rows)


def candidate_pool_oracle_results(candidates: pd.DataFrame, workups: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows = []
    current_lookup = current_policy_results(workups, "tmp").set_index(["dataset_name", "case_id", "budget"])
    for key, group in candidates.groupby(["dataset_name", "case_id", "budget"], sort=False):
        current = current_lookup.loc[key]
        hit = group[group["is_truth_candidate"].eq(1)]
        if len(hit):
            chosen = hit.sort_values("candidate_rank").iloc[0]
            prediction = str(chosen["label"])
            action = "oracle_truth_in_pool"
        else:
            chosen = group.sort_values("candidate_rank").iloc[0]
            prediction = str(chosen["label"])
            action = "oracle_pool_miss_keep_current_pool_top"
        ranked = ranked_from_candidates(group, "resolver_score", prediction)
        truth = str(chosen["ground_truth_diagnosis"])
        rows.append(
            {
                "policy_name": policy_name,
                "dataset_name": key[0],
                "case_id": key[1],
                "budget": int(key[2]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": prediction,
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": normalize_label(prediction) == normalize_label(truth),
                "gtpa_at_3": topk_hit(ranked, truth, 3),
                "gtpa_at_5": topk_hit(ranked, truth, 5),
                "true_rank": rank_in_list(truth, ranked, 11),
                "original_correct_top1": bool(current["original_correct_top1"]),
                "changed_prediction": normalize_label(prediction) != normalize_label(current["policy_prediction"]),
                "candidate_pool_has_truth": bool(group["is_truth_candidate"].max()),
                "candidate_pool_size": int(group["label"].nunique()),
                "num_questions": safe_float(current.get("num_questions", np.nan), np.nan),
                "policy_action": action,
            }
        )
    return pd.DataFrame(rows)


scale_current = current_policy_results(scale_workups, "notebook51_current_live")
scale_oracle = candidate_pool_oracle_results(scale_candidates, scale_workups, "candidate_pool_oracle_non_deployable")
transfer_current = current_policy_results(transfer_workups, "notebook46_current_live")
transfer_oracle = candidate_pool_oracle_results(transfer_candidates, transfer_workups, "transfer_candidate_pool_oracle_non_deployable")

baseline_summary = pd.DataFrame(
    summarize_cases(scale_current, "notebook51_current_live")
    + summarize_cases(scale_oracle, "candidate_pool_oracle_non_deployable")
    + summarize_cases(transfer_current, "notebook46_current_live")
    + summarize_cases(transfer_oracle, "transfer_candidate_pool_oracle_non_deployable")
)
baseline_summary.to_csv(ARTIFACT_ROOT / "baseline_and_oracle_summary.csv", index=False)
display(baseline_summary[baseline_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 4. Train/Validate/Test Split On The MEDDx-Scale Cohort

# %%
def make_case_split(workups: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SPLIT_SEED)
    rows = []
    for dataset_name, group in workups[["dataset_name", "case_id", "case_group"]].drop_duplicates().groupby("dataset_name"):
        case_ids = group["case_id"].astype(str).tolist()
        rng.shuffle(case_ids)
        n_cases = len(case_ids)
        n_train = int(round(TRAIN_FRACTION * n_cases))
        n_validate = int(round(VALIDATE_FRACTION * n_cases))
        train_ids = set(case_ids[:n_train])
        validate_ids = set(case_ids[n_train : n_train + n_validate])
        for case_id in case_ids:
            split = "train" if case_id in train_ids else "validate" if case_id in validate_ids else "test"
            rows.append({"dataset_name": dataset_name, "case_id": case_id, "split": split})
    return pd.DataFrame(rows)


split_assignment = make_case_split(scale_workups)
split_assignment.to_csv(ARTIFACT_ROOT / "scale_case_split_assignment.csv", index=False)
scale_candidates = scale_candidates.merge(split_assignment, on=["dataset_name", "case_id"], how="left")
scale_workups = scale_workups.merge(split_assignment, on=["dataset_name", "case_id"], how="left")
display(split_assignment.groupby(["dataset_name", "split"]).size().unstack(fill_value=0))

# %% [markdown]
# ## 5. Selected Logistic Resolver And Threshold Calibration

# %%
def fit_model(model: Any, train_frame: pd.DataFrame) -> Any:
    fitted = clone(model)
    fitted.fit(train_frame[FEATURE_COLUMNS], train_frame["is_truth_candidate"].astype(int))
    return fitted


def add_model_scores(model: Any, frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = frame.copy()
    out[score_column] = model.predict_proba(out[FEATURE_COLUMNS])[:, 1]
    return out


def learned_policy_results(
    candidates: pd.DataFrame,
    workups: pd.DataFrame,
    score_column: str,
    policy_name: str,
    threshold: float | None,
    require_support_noninferior: bool = False,
) -> pd.DataFrame:
    current_lookup = current_policy_results(workups, "tmp").set_index(["dataset_name", "case_id", "budget"])
    rows = []
    for key, group in candidates.groupby(["dataset_name", "case_id", "budget"], sort=False):
        group = group.copy()
        current_candidate = group[group["candidate_rank"].eq(1)].iloc[0]
        learned_top = group.sort_values([score_column, "resolver_score"], ascending=[False, False]).iloc[0]
        current_score = safe_float(current_candidate.get(score_column, 0.0))
        learned_score = safe_float(learned_top.get(score_column, 0.0))
        delta = learned_score - current_score
        choose_learned = threshold is None
        if threshold is not None:
            choose_learned = (
                normalize_label(learned_top["label"]) != normalize_label(current_candidate["label"])
                and delta >= threshold
            )
            if choose_learned and require_support_noninferior:
                choose_learned = safe_float(learned_top["support_count"]) >= safe_float(current_candidate["support_count"])
        chosen = learned_top if choose_learned else current_candidate
        prediction = str(chosen["label"])
        ranked = ranked_from_candidates(group, score_column, prediction)
        truth = str(chosen["ground_truth_diagnosis"])
        current = current_lookup.loc[key]
        rows.append(
            {
                "policy_name": policy_name,
                "dataset_name": key[0],
                "case_id": key[1],
                "budget": int(key[2]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": prediction,
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": normalize_label(prediction) == normalize_label(truth),
                "gtpa_at_3": topk_hit(ranked, truth, 3),
                "gtpa_at_5": topk_hit(ranked, truth, 5),
                "true_rank": rank_in_list(truth, ranked, 11),
                "original_correct_top1": bool(current["original_correct_top1"]),
                "changed_prediction": normalize_label(prediction) != normalize_label(current["policy_prediction"]),
                "candidate_pool_has_truth": bool(group["is_truth_candidate"].max()),
                "candidate_pool_size": int(group["label"].nunique()),
                "num_questions": safe_float(current.get("num_questions", np.nan), np.nan),
                "policy_action": "learned_override" if choose_learned else "keep_current",
                "learned_top_label": str(learned_top["label"]),
                "learned_top_score": learned_score,
                "current_candidate_score": current_score,
                "learned_delta_vs_current": delta,
            }
        )
    return pd.DataFrame(rows)


train_candidates = scale_candidates[scale_candidates["split"].eq("train")].copy()
validate_candidates = scale_candidates[scale_candidates["split"].eq("validate")].copy()
test_candidates = scale_candidates[scale_candidates["split"].eq("test")].copy()
train_workups = scale_workups[scale_workups["split"].eq("train")].copy()
validate_workups = scale_workups[scale_workups["split"].eq("validate")].copy()
test_workups = scale_workups[scale_workups["split"].eq("test")].copy()

selected_train_model = fit_model(SELECTED_MODEL, train_candidates)
joblib.dump(selected_train_model, ARTIFACT_ROOT / "selected_logistic_train_split_model.joblib")

validate_scored = add_model_scores(selected_train_model, validate_candidates, "selected_model_probability")
test_scored = add_model_scores(selected_train_model, test_candidates, "selected_model_probability")

threshold_rows = []
for threshold in THRESHOLD_GRID:
    policy = learned_policy_results(
        validate_scored,
        validate_workups,
        "selected_model_probability",
        f"{SELECTED_MODEL_NAME}_threshold_{threshold}",
        threshold=threshold,
    )
    overall = pd.DataFrame(summarize_cases(policy, "tmp")).query("dataset_name == 'ALL'").iloc[0].to_dict()
    threshold_rows.append(
        {
            "threshold": float(threshold),
            "top1_count": int(overall["top1_count"]),
            "top1": float(overall["top1"]),
            "top3": float(overall["top3"]),
            "top5": float(overall["top5"]),
            "wins_vs_current": int(overall["wins_vs_current"]),
            "regressions_vs_current": int(overall["regressions_vs_current"]),
            "changed_predictions": int(overall["changed_predictions"]),
        }
    )

threshold_sweep = pd.DataFrame(threshold_rows)
threshold_sweep.to_csv(ARTIFACT_ROOT / "validation_threshold_sweep.csv", index=False)

# Conservative tie-break: maximize validation top-1, then minimize regressions, then minimize changed predictions.
selected_row = (
    threshold_sweep.sort_values(
        ["top1_count", "regressions_vs_current", "changed_predictions", "threshold"],
        ascending=[False, True, True, False],
    )
    .iloc[0]
    .to_dict()
)
SELECTED_THRESHOLD = float(selected_row["threshold"])
print("Selected threshold:", SELECTED_THRESHOLD)
display(threshold_sweep)

selected_validate_results = learned_policy_results(
    validate_scored,
    validate_workups,
    "selected_model_probability",
    "selected_logistic_resolver_validate",
    threshold=SELECTED_THRESHOLD,
)
selected_test_results = learned_policy_results(
    test_scored,
    test_workups,
    "selected_model_probability",
    "selected_logistic_resolver_internal_test",
    threshold=SELECTED_THRESHOLD,
)
internal_summary = pd.DataFrame(
    summarize_cases(current_policy_results(validate_workups, "scale_validate_current"), "scale_validate_current")
    + summarize_cases(selected_validate_results, "selected_logistic_resolver_validate")
    + summarize_cases(current_policy_results(test_workups, "scale_internal_test_current"), "scale_internal_test_current")
    + summarize_cases(selected_test_results, "selected_logistic_resolver_internal_test")
)
internal_summary.to_csv(ARTIFACT_ROOT / "internal_split_policy_summary.csv", index=False)
selected_validate_results.to_csv(ARTIFACT_ROOT / "selected_validate_case_results.csv", index=False)
selected_test_results.to_csv(ARTIFACT_ROOT / "selected_internal_test_case_results.csv", index=False)
display(internal_summary[internal_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 6. Case-Blocked Cross-Validation Diagnostics

# %%
DIAGNOSTIC_MODELS = {
    "oof_logistic_balanced": SELECTED_MODEL,
    "oof_hgb_diagnostic": HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=12,
        l2_regularization=0.50,
        random_state=OOF_SEED,
    ),
    "oof_extra_trees_diagnostic": ExtraTreesClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=OOF_SEED,
        n_jobs=-1,
    ),
}

groups = scale_candidates["case_group"].astype(str).values
y = scale_candidates["is_truth_candidate"].astype(int).values
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)

oof_summary_rows = []
oof_case_results = []
oof_score_frame = scale_candidates[["dataset_name", "case_id", "budget", "candidate_rank", "label", "ground_truth_diagnosis"]].copy()

for model_name, model in DIAGNOSTIC_MODELS.items():
    probabilities = np.zeros(len(scale_candidates), dtype=float)
    for train_index, test_index in cv.split(scale_candidates[FEATURE_COLUMNS], y, groups):
        fitted = clone(model)
        fitted.fit(scale_candidates.iloc[train_index][FEATURE_COLUMNS], y[train_index])
        probabilities[test_index] = fitted.predict_proba(scale_candidates.iloc[test_index][FEATURE_COLUMNS])[:, 1]
    score_column = f"{model_name}_probability"
    scored = scale_candidates.copy()
    scored[score_column] = probabilities
    oof_score_frame[score_column] = probabilities
    results = learned_policy_results(scored, scale_workups, score_column, model_name, threshold=None)
    oof_case_results.append(results)
    oof_summary_rows.extend(summarize_cases(results, model_name))

oof_summary = pd.DataFrame(oof_summary_rows)
oof_results = pd.concat(oof_case_results, ignore_index=True)
oof_summary.to_csv(ARTIFACT_ROOT / "case_blocked_oof_resolver_summary.csv", index=False)
oof_results.to_csv(ARTIFACT_ROOT / "case_blocked_oof_case_results.csv", index=False)
oof_score_frame.to_csv(ARTIFACT_ROOT / "case_blocked_oof_candidate_scores.csv", index=False)
display(oof_summary[oof_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 7. Frozen Transfer Back To The Old 90-Workup Artifact

# %%
final_model = fit_model(SELECTED_MODEL, scale_candidates)
joblib.dump(final_model, ARTIFACT_ROOT / "selected_logistic_full_scale_model.joblib")

scale_full_scored = add_model_scores(final_model, scale_candidates, "selected_full_scale_probability")
transfer_scored = add_model_scores(final_model, transfer_candidates, "selected_full_scale_probability")

scale_full_selected = learned_policy_results(
    scale_full_scored,
    scale_workups,
    "selected_full_scale_probability",
    "selected_logistic_resolver_full_scale_fit_diagnostic",
    threshold=SELECTED_THRESHOLD,
)
transfer_selected = learned_policy_results(
    transfer_scored,
    transfer_workups,
    "selected_full_scale_probability",
    "selected_logistic_resolver_transfer_old90",
    threshold=SELECTED_THRESHOLD,
)

transfer_summary = pd.DataFrame(
    summarize_cases(transfer_current, "notebook46_current_live")
    + summarize_cases(transfer_oracle, "transfer_candidate_pool_oracle_non_deployable")
    + summarize_cases(transfer_selected, "selected_logistic_resolver_transfer_old90")
)
scale_full_summary = pd.DataFrame(
    summarize_cases(scale_current, "notebook51_current_live")
    + summarize_cases(scale_oracle, "candidate_pool_oracle_non_deployable")
    + summarize_cases(scale_full_selected, "selected_logistic_resolver_full_scale_fit_diagnostic")
)
transfer_selected.to_csv(ARTIFACT_ROOT / "transfer_old90_selected_case_results.csv", index=False)
scale_full_selected.to_csv(ARTIFACT_ROOT / "scale_full_fit_selected_case_results_diagnostic.csv", index=False)
transfer_summary.to_csv(ARTIFACT_ROOT / "transfer_policy_summary.csv", index=False)
scale_full_summary.to_csv(ARTIFACT_ROOT / "scale_full_fit_policy_summary_diagnostic.csv", index=False)
display(transfer_summary[transfer_summary["dataset_name"].eq("ALL")])
display(scale_full_summary[scale_full_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 8. Failure Analysis

# %%
def failure_decomposition(workups: pd.DataFrame, policy_results: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    merged = policy_results.merge(
        workups[["dataset_name", "case_id", "budget", "candidate_pool_has_truth", "num_questions", "branch_triggered", "branch_count"]],
        on=["dataset_name", "case_id", "budget"],
        how="left",
        suffixes=("", "_workup"),
    )
    failures = merged[~merged["correct_top1"].astype(bool)].copy()
    failures["failure_type"] = np.where(
        failures["candidate_pool_has_truth_workup"].astype(bool),
        "resolver_failed_truth_in_pool",
        "candidate_generation_miss",
    )
    failures["policy_name"] = policy_name
    return failures


scale_failure_audit = failure_decomposition(scale_workups, scale_current, "notebook51_current_live")
scale_selected_failure_audit = failure_decomposition(scale_workups, scale_full_selected, "selected_logistic_resolver_full_scale_fit_diagnostic")
transfer_failure_audit = failure_decomposition(transfer_workups, transfer_selected, "selected_logistic_resolver_transfer_old90")

failure_summary = pd.concat([scale_failure_audit, scale_selected_failure_audit, transfer_failure_audit], ignore_index=True)
failure_summary.to_csv(ARTIFACT_ROOT / "failure_decomposition.csv", index=False)

failure_counts = (
    failure_summary.groupby(["policy_name", "dataset_name", "budget", "failure_type"], as_index=False)
    .size()
    .rename(columns={"size": "num_failures"})
)
failure_counts.to_csv(ARTIFACT_ROOT / "failure_decomposition_summary.csv", index=False)
display(failure_counts)

hard_case_audit = {
    "scale_current_pool_misses": scale_failure_audit[scale_failure_audit["failure_type"].eq("candidate_generation_miss")]
    .head(40)
    .to_dict(orient="records"),
    "scale_selected_truth_in_pool_misses": scale_selected_failure_audit[
        scale_selected_failure_audit["failure_type"].eq("resolver_failed_truth_in_pool")
    ]
    .head(40)
    .to_dict(orient="records"),
    "transfer_selected_changes": transfer_selected[transfer_selected["changed_prediction"].astype(bool)].to_dict(orient="records"),
}
write_json(ARTIFACT_ROOT / "hard_case_scale_resolver_audits.json", hard_case_audit)

# %% [markdown]
# ## 9. Figures

# %%
plot_summary = pd.concat(
    [
        baseline_summary,
        internal_summary,
        oof_summary,
        transfer_summary,
        scale_full_summary,
    ],
    ignore_index=True,
)
plot_summary.to_csv(ARTIFACT_ROOT / "combined_policy_summary.csv", index=False)

overall = plot_summary[plot_summary["dataset_name"].eq("ALL")].copy()
plt.figure(figsize=(9, 4.5))
show = overall[
    overall["policy_name"].isin(
        [
            "notebook51_current_live",
            "candidate_pool_oracle_non_deployable",
            "selected_logistic_resolver_internal_test",
            "oof_logistic_balanced",
            "oof_hgb_diagnostic",
            "notebook46_current_live",
            "selected_logistic_resolver_transfer_old90",
        ]
    )
].copy()
show["label"] = show["policy_name"].str.replace("_", "\n")
plt.bar(show["label"], show["top1"], color="#4f7f9f")
plt.ylim(0, 1.02)
plt.ylabel("Top-1 accuracy")
plt.xticks(rotation=0, fontsize=8)
plt.title("MEDDx offline resolver calibration summary")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "overall_policy_top1_comparison.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
pool_slice = baseline_summary[
    baseline_summary["policy_name"].isin(["notebook51_current_live", "candidate_pool_oracle_non_deployable"])
    & ~baseline_summary["dataset_name"].eq("ALL")
].copy()
for policy_name, group in pool_slice.groupby("policy_name"):
    plt.plot(
        range(len(group)),
        group["top1"],
        marker="o",
        label=policy_name,
    )
plt.xticks(range(len(group)), [f"{r.dataset_name}\nB{int(r.budget)}" for r in group.itertuples()], fontsize=8)
plt.ylim(0, 1.02)
plt.ylabel("Top-1 / oracle ceiling")
plt.title("Current accuracy versus candidate-pool ceiling")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "candidate_pool_ceiling_by_slice.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(threshold_sweep["threshold"].replace(999.0, 0.95), threshold_sweep["top1"], marker="o", label="validation top-1")
plt.plot(threshold_sweep["threshold"].replace(999.0, 0.95), threshold_sweep["changed_predictions"], marker="o", label="changed predictions")
plt.xlabel("Override threshold; 0.95 denotes keep-current sentinel")
plt.title("Validation threshold sweep")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "validation_threshold_sweep.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
fail_plot = failure_counts[
    failure_counts["policy_name"].eq("notebook51_current_live")
].copy()
if len(fail_plot):
    pivot = fail_plot.pivot_table(
        index=["dataset_name", "budget"],
        columns="failure_type",
        values="num_failures",
        fill_value=0,
        aggfunc="sum",
    )
    pivot.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.ylabel("Failure count")
    plt.title("Notebook 51 failure decomposition")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "notebook51_failure_decomposition.png", dpi=180)
    plt.close()

# %% [markdown]
# ## 10. Final Summary And Artifact Contract

# %%
scale_current_overall = baseline_summary[
    baseline_summary["policy_name"].eq("notebook51_current_live") & baseline_summary["dataset_name"].eq("ALL")
].iloc[0]
scale_oracle_overall = baseline_summary[
    baseline_summary["policy_name"].eq("candidate_pool_oracle_non_deployable") & baseline_summary["dataset_name"].eq("ALL")
].iloc[0]
transfer_current_overall = transfer_summary[
    transfer_summary["policy_name"].eq("notebook46_current_live") & transfer_summary["dataset_name"].eq("ALL")
].iloc[0]
transfer_selected_overall = transfer_summary[
    transfer_summary["policy_name"].eq("selected_logistic_resolver_transfer_old90") & transfer_summary["dataset_name"].eq("ALL")
].iloc[0]
internal_test_overall = internal_summary[
    internal_summary["policy_name"].eq("selected_logistic_resolver_internal_test") & internal_summary["dataset_name"].eq("ALL")
].iloc[0]

promotion_decision = (
    "not_promoted"
    if int(transfer_selected_overall["top1_count"]) <= int(transfer_current_overall["top1_count"])
    else "candidate_pending_confirmation"
)

selected_policy = {
    "selected_policy_name": "selected_logistic_resolver_transfer_old90",
    "claim_status": promotion_decision,
    "scale_input_run": SCALE_RUN_NAME,
    "transfer_input_run": TRANSFER_RUN_NAME,
    "model": SELECTED_MODEL_NAME,
    "feature_count": len(FEATURE_COLUMNS),
    "split_seed": SPLIT_SEED,
    "case_split": {
        "train_fraction": TRAIN_FRACTION,
        "validate_fraction": VALIDATE_FRACTION,
        "test_fraction": TEST_FRACTION,
        "grouping": "all budgets for one dataset/case stay in the same split",
    },
    "selected_threshold": SELECTED_THRESHOLD,
    "threshold_selection_rule": "maximize validation top1, then minimize regressions, then minimize changed predictions",
    "scale_live_current": {
        "top1_count": int(scale_current_overall["top1_count"]),
        "num_workups": int(scale_current_overall["num_workups"]),
        "top1": float(scale_current_overall["top1"]),
        "top3": float(scale_current_overall["top3"]),
        "top5": float(scale_current_overall["top5"]),
    },
    "scale_candidate_pool_oracle": {
        "top1_count": int(scale_oracle_overall["top1_count"]),
        "num_workups": int(scale_oracle_overall["num_workups"]),
        "top1": float(scale_oracle_overall["top1"]),
    },
    "internal_test_selected": {
        "top1_count": int(internal_test_overall["top1_count"]),
        "num_workups": int(internal_test_overall["num_workups"]),
        "top1": float(internal_test_overall["top1"]),
        "wins_vs_current": int(internal_test_overall["wins_vs_current"]),
        "regressions_vs_current": int(internal_test_overall["regressions_vs_current"]),
    },
    "old90_transfer_current": {
        "top1_count": int(transfer_current_overall["top1_count"]),
        "num_workups": int(transfer_current_overall["num_workups"]),
        "top1": float(transfer_current_overall["top1"]),
    },
    "old90_transfer_selected": {
        "top1_count": int(transfer_selected_overall["top1_count"]),
        "num_workups": int(transfer_selected_overall["num_workups"]),
        "top1": float(transfer_selected_overall["top1"]),
        "wins_vs_current": int(transfer_selected_overall["wins_vs_current"]),
        "regressions_vs_current": int(transfer_selected_overall["regressions_vs_current"]),
    },
    "interpretation": [
        "The 900-workup live run is not mainly a solved-resolver problem; candidate-pool recall is only about 0.899.",
        "The selected logistic resolver gives at most a very small internal split improvement and does not improve the old 90-workup transfer set.",
        "Notebook 51 should be treated as a calibration corpus and failure map, not as evidence that the current universal MEDDx architecture is ready to claim superiority.",
    ],
}
write_json(ARTIFACT_ROOT / "selected_offline_resolver_policy.json", selected_policy)

resolved_run_config = {
    "run_name": RUN_NAME,
    "scale_input_run": SCALE_RUN_NAME,
    "transfer_input_run": TRANSFER_RUN_NAME,
    "artifact_root": str(ARTIFACT_ROOT),
    "feature_columns": FEATURE_COLUMNS,
    "selected_policy": selected_policy,
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

required_outputs = [
    "resolved_run_config.json",
    "candidate_level_scale_features.csv",
    "candidate_level_transfer_features.csv",
    "scale_workup_diagnostics.csv",
    "transfer_workup_diagnostics.csv",
    "baseline_and_oracle_summary.csv",
    "scale_case_split_assignment.csv",
    "validation_threshold_sweep.csv",
    "internal_split_policy_summary.csv",
    "selected_validate_case_results.csv",
    "selected_internal_test_case_results.csv",
    "case_blocked_oof_resolver_summary.csv",
    "case_blocked_oof_case_results.csv",
    "transfer_policy_summary.csv",
    "transfer_old90_selected_case_results.csv",
    "scale_full_fit_policy_summary_diagnostic.csv",
    "failure_decomposition.csv",
    "failure_decomposition_summary.csv",
    "hard_case_scale_resolver_audits.json",
    "selected_offline_resolver_policy.json",
]
missing_outputs = [name for name in required_outputs if not (ARTIFACT_ROOT / name).exists()]
if missing_outputs:
    raise FileNotFoundError(f"Missing required outputs: {missing_outputs}")

print("Notebook 52 artifact contract OK")
print("Selected policy:", selected_policy["selected_policy_name"], selected_policy["claim_status"])
print("Artifact root:", ARTIFACT_ROOT)
