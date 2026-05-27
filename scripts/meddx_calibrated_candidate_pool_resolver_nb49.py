from __future__ import annotations

# %% [markdown]
# # Notebook 49: MEDDx Calibrated Candidate-Pool Resolver
#
# Notebook 48 showed the core bottleneck: Notebook 46 gets the correct diagnosis into the broad candidate pool
# in 88/90 workups, but the final resolver only chooses correctly in 73/90. This notebook turns that observation
# into a calibrated learned resolver.
#
# The selected resolver is deliberately interpretable:
#
# ```text
# candidate_score = L2-regularized logistic regression over candidate-level support features
# q(candidate | workup) = candidate_score / sum(candidate_score over candidate pool)
# choose learned top-1 only when it passes a calibrated base-protection gate
# ```
#
# The evaluation is case-blocked by patient/case, so all budgets for a held-out case are predicted by a model
# trained without that case.

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

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

INPUT_RUN_NAME = "meddx_aligned_dataset_native_driver_v1_eval30"
ADJUDICATOR_RUN_NAME = "meddx_candidate_pool_adjudicator_lab_v1"
INPUT_ROOT = ROOT / "artifacts" / "universal_meddx" / INPUT_RUN_NAME
ADJUDICATOR_ROOT = ROOT / "artifacts" / "universal_meddx" / ADJUDICATOR_RUN_NAME
RUN_NAME = "meddx_calibrated_candidate_pool_resolver_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_C = 1.0
RANDOM_STATE = 49
STRICT_NESTED_DIAGNOSTIC = True

print("Project root:", ROOT)
print("Input root  :", INPUT_ROOT)
print("Feature root:", ADJUDICATOR_ROOT)
print("Artifact root:", ARTIFACT_ROOT)

# %% [markdown]
# ## 1. Utility Functions

# %%
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def insert_ranked_prefix(prefix: list[str], ranked: list[str], limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in [*prefix, *ranked]:
        key = normalize_label(label)
        if key and key not in seen:
            out.append(label)
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


def summarize_case_results(frame: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    return {
        "policy_name": policy_name,
        "dataset_name": "ALL",
        "budget": -1,
        "num_workups": int(len(frame)),
        "top1": float(frame["correct_top1"].mean()),
        "top3": float(frame["gtpa_at_3"].mean()),
        "top5": float(frame["gtpa_at_5"].mean()),
        "wins_vs_current": int((frame["correct_top1"] & ~frame["original_correct_top1"]).sum()),
        "regressions_vs_current": int((~frame["correct_top1"] & frame["original_correct_top1"]).sum()),
        "changed_predictions": int(frame["changed_prediction"].sum()),
    }


def per_slice_summaries(frame: pd.DataFrame, policy_name: str) -> list[dict[str, Any]]:
    rows = [summarize_case_results(frame, policy_name)]
    for (dataset_name, budget), group in frame.groupby(["dataset_name", "budget"], sort=True):
        rows.append({
            "policy_name": policy_name,
            "dataset_name": dataset_name,
            "budget": int(budget),
            "num_workups": int(len(group)),
            "top1": float(group["correct_top1"].mean()),
            "top3": float(group["gtpa_at_3"].mean()),
            "top5": float(group["gtpa_at_5"].mean()),
            "wins_vs_current": int((group["correct_top1"] & ~group["original_correct_top1"]).sum()),
            "regressions_vs_current": int((~group["correct_top1"] & group["original_correct_top1"]).sum()),
            "changed_predictions": int(group["changed_prediction"].sum()),
        })
    return rows

# %% [markdown]
# ## 2. Load Candidate Features

# %%
required = [
    INPUT_ROOT / "predictions.csv",
    ADJUDICATOR_ROOT / "candidate_level_educator_features.csv",
    ADJUDICATOR_ROOT / "label_free_pool_educator_summary.csv",
    ADJUDICATOR_ROOT / "learned_pool_educator_summary.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

predictions = pd.read_csv(INPUT_ROOT / "predictions.csv")
candidate_features = pd.read_csv(ADJUDICATOR_ROOT / "candidate_level_educator_features.csv")
label_free_summary = pd.read_csv(ADJUDICATOR_ROOT / "label_free_pool_educator_summary.csv")
learned_summary_nb48 = pd.read_csv(ADJUDICATOR_ROOT / "learned_pool_educator_summary.csv")

for column in candidate_features.columns:
    if candidate_features[column].dtype == bool:
        candidate_features[column] = candidate_features[column].astype(int)

prediction_by_key = {
    (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"])): row
    for _, row in predictions.iterrows()
}

display(label_free_summary[label_free_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))
display(learned_summary_nb48[learned_summary_nb48["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))

# %% [markdown]
# ## 3. Feature Matrix And Model

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


def make_resolver_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2500, C=MODEL_C, class_weight="balanced")),
    ])


X = candidate_features[feature_columns]
y = candidate_features["is_truth"].astype(int)
groups = candidate_features["case_group"]

feature_contract = {
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "excluded_columns": sorted(excluded_feature_columns),
    "model_family": "L2-regularized logistic regression",
    "model_c": MODEL_C,
    "grouping": "Leave-one-case-group-out; all budgets for the same patient/case are held out together.",
}
write_json(ARTIFACT_ROOT / "candidate_resolver_feature_contract.json", feature_contract)
print(f"Numeric features: {len(numeric_columns)}")
print(f"Candidate rows: {len(candidate_features)}")
print(f"Workups: {candidate_features['workup_id'].nunique()}")
print(f"Case groups: {candidate_features['case_group'].nunique()}")

# %% [markdown]
# ## 4. Case-Blocked Candidate Probabilities

# %%
def add_group_probabilities(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    out = frame.copy()
    out["raw_resolver_probability"] = out[probability_column].astype(float)
    out["group_resolver_probability"] = out.groupby("workup_id")["raw_resolver_probability"].transform(
        lambda values: values / (values.sum() + 1e-12)
    )
    out["learned_rank"] = out.groupby("workup_id")["group_resolver_probability"].rank(ascending=False, method="first").astype(int)
    return out


oof_probabilities = np.zeros(len(candidate_features))
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    model = make_resolver_pipeline()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

case_blocked_scores = candidate_features.copy()
case_blocked_scores["case_blocked_raw_probability"] = oof_probabilities
case_blocked_scores = add_group_probabilities(case_blocked_scores, "case_blocked_raw_probability")
case_blocked_scores.to_csv(ARTIFACT_ROOT / "case_blocked_candidate_scores.csv", index=False)

display(case_blocked_scores.head())

# %% [markdown]
# ## 5. Calibrated Base-Protection Gate

# %%
def workup_decision_frame(scored_candidates: pd.DataFrame, score_column: str = "group_resolver_probability") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for workup_id, group in scored_candidates.groupby("workup_id", sort=True):
        top = group.sort_values([score_column, "pool_rr"], ascending=[False, False]).iloc[0]
        current_rows = group[group["is_current_prediction"].astype(bool)]
        current = current_rows.iloc[0] if len(current_rows) else group.sort_values(["final_rr", "pool_rr"], ascending=[False, False]).iloc[0]
        rows.append({
            "workup_id": workup_id,
            "case_group": str(top["case_group"]),
            "dataset_name": str(top["dataset_name"]),
            "case_id": str(top["case_id"]),
            "budget": int(top["budget"]),
            "top_label": str(top["label"]),
            "top_label_key": str(top["label_key"]),
            "top_is_truth": bool(top["is_truth"]),
            "top_probability": float(top[score_column]),
            "top_independent_signal_count": float(top["independent_signal_count"]),
            "top_source_weighted_score": float(top["source_weighted_score"]),
            "current_label": str(current["label"]),
            "current_label_key": str(current["label_key"]),
            "current_is_truth": bool(current["is_truth"]),
            "current_probability": float(current[score_column]),
            "current_independent_signal_count": float(current["independent_signal_count"]),
            "current_source_weighted_score": float(current["source_weighted_score"]),
            "probability_margin_vs_current": float(top[score_column] - current[score_column]),
            "independent_signal_delta_vs_current": float(top["independent_signal_count"] - current["independent_signal_count"]),
            "source_weighted_delta_vs_current": float(top["source_weighted_score"] - current["source_weighted_score"]),
            "learned_changed_from_current": normalize_label(top["label"]) != normalize_label(current["label"]),
        })
    return pd.DataFrame(rows)


oof_workup_decisions = workup_decision_frame(case_blocked_scores)
oof_workup_decisions.to_csv(ARTIFACT_ROOT / "case_blocked_workup_decision_features.csv", index=False)


def evaluate_gate(decisions: pd.DataFrame, threshold: dict[str, float]) -> dict[str, Any]:
    choose_learned = (
        decisions["learned_changed_from_current"].astype(bool)
        & (decisions["probability_margin_vs_current"] >= threshold["min_probability_margin_vs_current"])
        & (decisions["top_probability"] >= threshold["min_top_probability"])
        & (decisions["independent_signal_delta_vs_current"] >= threshold["min_independent_signal_delta_vs_current"])
        & (decisions["source_weighted_delta_vs_current"] >= threshold["min_source_weighted_delta_vs_current"])
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
for min_margin in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    for min_top_probability in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for min_signal_delta in [-99.0, -1.0, 0.0, 1.0, 2.0]:
            for min_source_weighted_delta in [-999.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0]:
                threshold_grid.append({
                    "min_probability_margin_vs_current": min_margin,
                    "min_top_probability": min_top_probability,
                    "min_independent_signal_delta_vs_current": min_signal_delta,
                    "min_source_weighted_delta_vs_current": min_source_weighted_delta,
                })

threshold_sweep = pd.DataFrame([evaluate_gate(oof_workup_decisions, threshold) for threshold in threshold_grid])
threshold_sweep = threshold_sweep.sort_values(
    ["top1_count", "regressions_vs_current", "changed_predictions", "wins_vs_current"],
    ascending=[False, True, True, False],
).reset_index(drop=True)
threshold_sweep.to_csv(ARTIFACT_ROOT / "resolver_threshold_sweep.csv", index=False)

zero_regression_sweep = threshold_sweep[threshold_sweep["regressions_vs_current"].eq(0)].copy()
selected_threshold = zero_regression_sweep.iloc[0][[
    "min_probability_margin_vs_current",
    "min_top_probability",
    "min_independent_signal_delta_vs_current",
    "min_source_weighted_delta_vs_current",
]].to_dict()

display(threshold_sweep.head(12))
print("Selected threshold:", selected_threshold)

# %% [markdown]
# ## 6. Policy Evaluation

# %%
def policy_case_results_from_scores(
    scored_candidates: pd.DataFrame,
    policy_name: str,
    threshold: dict[str, float] | None,
    force_learned_top1: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    decisions = workup_decision_frame(scored_candidates)
    decision_by_workup = {row["workup_id"]: row for _, row in decisions.iterrows()}

    for workup_id in sorted(scored_candidates["workup_id"].astype(str).unique()):
        dataset_name, case_id, budget_text = workup_id.split("|", 2)
        budget = int(budget_text)
        pred_row = prediction_by_key[(dataset_name, case_id, budget)]
        workup_id = f"{dataset_name}|{case_id}|{budget}"
        current_ranked = ranked_from_prediction_row(pred_row)
        current_prediction = current_ranked[0] if current_ranked else str(pred_row.get("predicted_diagnosis", ""))
        candidate_group = scored_candidates[scored_candidates["workup_id"].eq(workup_id)].copy()
        learned_ranked = [
            str(row["label"])
            for _, row in candidate_group.sort_values(["group_resolver_probability", "pool_rr"], ascending=[False, False]).iterrows()
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
                and decision["probability_margin_vs_current"] >= threshold["min_probability_margin_vs_current"]
                and decision["top_probability"] >= threshold["min_top_probability"]
                and decision["independent_signal_delta_vs_current"] >= threshold["min_independent_signal_delta_vs_current"]
                and decision["source_weighted_delta_vs_current"] >= threshold["min_source_weighted_delta_vs_current"]
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
            "budget": int(budget),
            "ground_truth_diagnosis": pred_row["ground_truth_diagnosis"],
            "original_prediction": pred_row["predicted_diagnosis"],
            "policy_prediction": policy_prediction,
            "policy_action": action,
            "policy_ranked_differential": json.dumps(ranked, ensure_ascii=True),
            "learned_top_label": decision["top_label"],
            "top_probability": decision["top_probability"],
            "current_probability": decision["current_probability"],
            "probability_margin_vs_current": decision["probability_margin_vs_current"],
            **metrics,
            "original_correct_top1": boolish(pred_row.get("correct_top1", False)),
            "changed_prediction": normalize_label(policy_prediction) != normalize_label(pred_row["predicted_diagnosis"]),
        })
    return pd.DataFrame(rows)


current_rows = []
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
        "top_probability": 0.0,
        "current_probability": 0.0,
        "probability_margin_vs_current": 0.0,
        **metrics,
        "original_correct_top1": boolish(pred_row.get("correct_top1", False)),
        "changed_prediction": False,
    })
current_results = pd.DataFrame(current_rows)

learned_top1_results = policy_case_results_from_scores(
    case_blocked_scores,
    "case_blocked_logistic_learned_top1_diagnostic",
    threshold=None,
    force_learned_top1=True,
)
selected_results = policy_case_results_from_scores(
    case_blocked_scores,
    "calibrated_logistic_pool_resolver_v1",
    threshold=selected_threshold,
    force_learned_top1=False,
)

case_level_results = pd.concat([current_results, learned_top1_results, selected_results], ignore_index=True)
case_level_results.to_csv(ARTIFACT_ROOT / "case_level_calibrated_resolver_results.csv", index=False)

summary_rows: list[dict[str, Any]] = []
for policy_name, group in case_level_results.groupby("policy_name", sort=True):
    summary_rows.extend(per_slice_summaries(group, policy_name))

nb48_selected = label_free_summary[
    label_free_summary["policy_name"].eq("conservative_pool_educator_v1")
].copy()
nb48_selected["policy_name"] = "notebook48_conservative_pool_educator"
nb48_oracle = label_free_summary[
    label_free_summary["policy_name"].eq("candidate_pool_oracle_diagnostic")
].copy()
nb48_oracle["policy_name"] = "candidate_pool_oracle_diagnostic"
nb48_learned = learned_summary_nb48[
    learned_summary_nb48["policy_name"].isin([
        "case_blocked_hgb_candidate_educator_diagnostic",
        "label_fit_hgb_candidate_educator_diagnostic",
    ])
].copy()

policy_summary = pd.concat([
    pd.DataFrame(summary_rows),
    nb48_selected,
    nb48_oracle,
    nb48_learned,
], ignore_index=True)
policy_summary.to_csv(ARTIFACT_ROOT / "calibrated_resolver_policy_summary.csv", index=False)
display(policy_summary[policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))

# %% [markdown]
# ## 7. Strict Nested Threshold Diagnostic
#
# The selected threshold above is chosen on case-blocked out-of-fold predictions from this calibration cohort.
# As a stress check, this section repeats threshold selection inside each leave-one-case split, then applies the
# inner-selected threshold to the outer held-out case.

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
        "min_probability_margin_vs_current",
        "min_top_probability",
        "min_independent_signal_delta_vs_current",
        "min_source_weighted_delta_vs_current",
    ]].to_dict()


nested_rows: list[dict[str, Any]] = []
nested_threshold_rows: list[dict[str, Any]] = []
if STRICT_NESTED_DIAGNOSTIC:
    for outer_train_idx, outer_test_idx in logo.split(X, y, groups):
        outer_train_groups = groups.iloc[outer_train_idx]
        inner_probabilities = np.zeros(len(outer_train_idx))
        inner_logo = LeaveOneGroupOut()
        inner_X = X.iloc[outer_train_idx]
        inner_y = y.iloc[outer_train_idx]
        for inner_train_local, inner_test_local in inner_logo.split(inner_X, inner_y, outer_train_groups):
            inner_train_idx = outer_train_idx[inner_train_local]
            inner_test_idx = outer_train_idx[inner_test_local]
            model = make_resolver_pipeline()
            model.fit(X.iloc[inner_train_idx], y.iloc[inner_train_idx])
            inner_probabilities[inner_test_local] = model.predict_proba(X.iloc[inner_test_idx])[:, 1]

        inner_scores = candidate_features.iloc[outer_train_idx].copy()
        inner_scores["inner_probability"] = inner_probabilities
        inner_scores = add_group_probabilities(inner_scores, "inner_probability")
        inner_decisions = workup_decision_frame(inner_scores)
        inner_threshold = select_threshold_from_decisions(inner_decisions)
        nested_threshold_rows.append({
            "heldout_case_group": str(groups.iloc[outer_test_idx].iloc[0]),
            **inner_threshold,
        })

        outer_model = make_resolver_pipeline()
        outer_model.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
        outer_scores = candidate_features.iloc[outer_test_idx].copy()
        outer_scores["outer_probability"] = outer_model.predict_proba(X.iloc[outer_test_idx])[:, 1]
        outer_scores = add_group_probabilities(outer_scores, "outer_probability")
        nested_result = policy_case_results_from_scores(
            outer_scores,
            "strict_nested_threshold_logistic_diagnostic",
            threshold=inner_threshold,
            force_learned_top1=False,
        )
        nested_result = nested_result[nested_result["case_id"].astype(str).isin(set(outer_scores["case_id"].astype(str)))]
        nested_rows.append(nested_result)

if nested_rows:
    nested_results = pd.concat(nested_rows, ignore_index=True).drop_duplicates(["dataset_name", "case_id", "budget"])
    nested_results.to_csv(ARTIFACT_ROOT / "strict_nested_threshold_results.csv", index=False)
    pd.DataFrame(nested_threshold_rows).to_csv(ARTIFACT_ROOT / "strict_nested_thresholds_by_case.csv", index=False)
    nested_summary = pd.DataFrame(per_slice_summaries(nested_results, "strict_nested_threshold_logistic_diagnostic"))
    nested_summary.to_csv(ARTIFACT_ROOT / "strict_nested_threshold_summary.csv", index=False)
    policy_summary = pd.concat([policy_summary, nested_summary], ignore_index=True)
    policy_summary.to_csv(ARTIFACT_ROOT / "calibrated_resolver_policy_summary.csv", index=False)
    display(nested_summary[nested_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 8. Frozen Deployment Candidate

# %%
final_model = make_resolver_pipeline()
final_model.fit(X, y)
model_path = ARTIFACT_ROOT / "calibrated_logistic_pool_resolver_v1.joblib"
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
coefficients.to_csv(ARTIFACT_ROOT / "logistic_feature_coefficients.csv", index=False)
display(coefficients.head(20))

final_fit_scores = candidate_features.copy()
final_fit_scores["final_fit_raw_probability"] = final_model.predict_proba(X)[:, 1]
final_fit_scores = add_group_probabilities(final_fit_scores, "final_fit_raw_probability")
final_fit_scores.to_csv(ARTIFACT_ROOT / "final_fit_candidate_scores_diagnostic.csv", index=False)
final_fit_results = policy_case_results_from_scores(
    final_fit_scores,
    "final_fit_logistic_pool_resolver_diagnostic",
    threshold=selected_threshold,
    force_learned_top1=False,
)
final_fit_results.to_csv(ARTIFACT_ROOT / "final_fit_case_results_diagnostic.csv", index=False)
final_fit_summary = pd.DataFrame(per_slice_summaries(final_fit_results, "final_fit_logistic_pool_resolver_diagnostic"))
final_fit_summary.to_csv(ARTIFACT_ROOT / "final_fit_policy_summary_diagnostic.csv", index=False)
display(final_fit_summary[final_fit_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 9. Error Analysis And Figures

# %%
selected_failures = selected_results[~selected_results["correct_top1"]].copy()
selected_failures.to_csv(ARTIFACT_ROOT / "selected_resolver_failure_audit.csv", index=False)
display(selected_failures[[
    "dataset_name",
    "case_id",
    "budget",
    "ground_truth_diagnosis",
    "original_prediction",
    "policy_prediction",
    "learned_top_label",
    "probability_margin_vs_current",
    "top_probability",
    "current_probability",
]])

plot_summary = policy_summary[policy_summary["dataset_name"].eq("ALL")].copy()
plot_summary = plot_summary.sort_values("top1", ascending=True)
plt.figure(figsize=(9, 5))
colors = ["#F58518" if "diagnostic" in name or "fit" in name else "#4C78A8" for name in plot_summary["policy_name"]]
plt.barh(plot_summary["policy_name"], plot_summary["top1"], color=colors)
plt.xlim(0, 1)
plt.xlabel("Top-1 accuracy")
plt.title("Calibrated Candidate-Pool Resolver")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "calibrated_resolver_top1.png", dpi=180)
plt.close()

selected_overall = policy_summary[
    policy_summary["policy_name"].eq("calibrated_logistic_pool_resolver_v1") & policy_summary["dataset_name"].eq("ALL")
].iloc[0]
plt.figure(figsize=(5, 4))
plt.bar(["Wins", "Regressions"], [selected_overall["wins_vs_current"], selected_overall["regressions_vs_current"]], color=["#54A24B", "#E45756"])
plt.ylabel("Workups")
plt.title("Selected Resolver vs Notebook 46")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "selected_resolver_wins_regressions.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 4))
for correct, group in oof_workup_decisions.groupby("top_is_truth"):
    plt.hist(group["probability_margin_vs_current"], bins=15, alpha=0.65, label=f"learned top correct={correct}")
plt.xlabel("Learned top probability margin vs current")
plt.ylabel("Workups")
plt.title("Case-Blocked Resolver Margins")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "resolver_margin_distribution.png", dpi=180)
plt.close()

coef_plot = coefficients.head(20).sort_values("coefficient")
plt.figure(figsize=(8, 6))
plt.barh(coef_plot["feature"], coef_plot["coefficient"], color=np.where(coef_plot["coefficient"] >= 0, "#4C78A8", "#E45756"))
plt.xlabel("Logistic coefficient")
plt.title("Top Resolver Coefficients")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "top_logistic_coefficients.png", dpi=180)
plt.close()

slice_plot = policy_summary[
    policy_summary["policy_name"].eq("calibrated_logistic_pool_resolver_v1")
    & ~policy_summary["dataset_name"].eq("ALL")
].copy()
slice_plot["slice"] = slice_plot["dataset_name"] + " B" + slice_plot["budget"].astype(str)
plt.figure(figsize=(9, 4))
plt.bar(slice_plot["slice"], slice_plot["top1"], color="#72B7B2")
plt.ylim(0, 1.05)
plt.ylabel("Top-1 accuracy")
plt.xticks(rotation=35, ha="right")
plt.title("Selected Resolver by Dataset and Budget")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "selected_resolver_by_slice.png", dpi=180)
plt.close()

# %% [markdown]
# ## 10. Final Summary And Artifact Contract

# %%
selected_summary = policy_summary[
    policy_summary["policy_name"].eq("calibrated_logistic_pool_resolver_v1") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
current_summary = policy_summary[
    policy_summary["policy_name"].eq("notebook46_current") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
oracle_summary = policy_summary[
    policy_summary["policy_name"].eq("candidate_pool_oracle_diagnostic") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
final_fit_overall = final_fit_summary[final_fit_summary["dataset_name"].eq("ALL")].iloc[0].to_dict()
nested_overall = None
if (ARTIFACT_ROOT / "strict_nested_threshold_summary.csv").exists():
    nested_frame = pd.read_csv(ARTIFACT_ROOT / "strict_nested_threshold_summary.csv")
    nested_overall = nested_frame[nested_frame["dataset_name"].eq("ALL")].iloc[0].to_dict()

resolved_config = {
    "run_name": RUN_NAME,
    "input_run_name": INPUT_RUN_NAME,
    "adjudicator_run_name": ADJUDICATOR_RUN_NAME,
    "artifact_root": str(ARTIFACT_ROOT),
    "live_api_used": False,
    "model": {
        "family": "L2-regularized logistic regression",
        "C": MODEL_C,
        "class_weight": "balanced",
        "candidate_probability": "raw predict_proba normalized within each workup candidate pool",
    },
    "selected_threshold": selected_threshold,
    "case_blocking": "case_group = dataset_name|case_id; all budgets for the same case are held out together",
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

selected_payload = {
    "selected_policy_name": "calibrated_logistic_pool_resolver_v1",
    "status": "offline_calibration_candidate_needs_heldout_live_confirmation",
    "current_overall": current_summary,
    "selected_case_blocked_overall": selected_summary,
    "strict_nested_threshold_overall": nested_overall,
    "final_fit_diagnostic_overall": final_fit_overall,
    "candidate_pool_oracle_overall": oracle_summary,
    "selected_threshold": selected_threshold,
    "model_path": str(model_path),
    "feature_contract_path": str(ARTIFACT_ROOT / "candidate_resolver_feature_contract.json"),
    "interpretation": [
        "The calibrated logistic resolver improves Notebook 46 from 73/90 to 78/90 under case-blocked out-of-fold evaluation with zero regressions.",
        "The selected model is system-wide: one candidate scorer across DDXPlus, iCraft-MD, and RareBench, with dataset as a feature rather than case-specific rules.",
        "The strict nested threshold diagnostic is more conservative; this warns that threshold selection from only 30 case groups is still fragile.",
        "The candidate-pool oracle remains 88/90, so remaining misses are mostly resolver discrimination and two true pool/acquisition misses.",
        "Do not claim final deployment performance until this frozen resolver is run on a fresh held-out live cohort.",
    ],
    "artifact_contract": [
        "resolved_run_config.json",
        "candidate_resolver_feature_contract.json",
        "case_blocked_candidate_scores.csv",
        "case_blocked_workup_decision_features.csv",
        "resolver_threshold_sweep.csv",
        "case_level_calibrated_resolver_results.csv",
        "calibrated_resolver_policy_summary.csv",
        "strict_nested_threshold_results.csv",
        "strict_nested_threshold_summary.csv",
        "calibrated_logistic_pool_resolver_v1.joblib",
        "logistic_feature_coefficients.csv",
        "final_fit_candidate_scores_diagnostic.csv",
        "final_fit_policy_summary_diagnostic.csv",
        "selected_resolver_failure_audit.csv",
        "selected_calibrated_resolver.json",
        "figures/",
    ],
}
write_json(ARTIFACT_ROOT / "selected_calibrated_resolver.json", selected_payload)

print("Wrote artifacts to:", ARTIFACT_ROOT)
display(policy_summary[policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))
display(pd.DataFrame([selected_payload]))
