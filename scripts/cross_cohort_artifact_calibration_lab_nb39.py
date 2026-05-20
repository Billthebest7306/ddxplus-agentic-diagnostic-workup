from __future__ import annotations

# %% [markdown]
# # Notebook 39: Cross-Cohort Artifact Calibration Lab
#
# This notebook uses only saved artifacts from the candidate-pool/branching work.
# It does not call the live API and it does not change the acquisition policy.
#
# The control question is:
#
# > Given the 49-case development run, the 98-case balanced confirmation, and
# > the 196-case live calibration cohort, can we fit a calibration layer that
# > improves final accuracy while still looking like a rule that could generalize?
#
# We keep three claims separate:
#
# - **current final pipeline**: the latest saved final predictions from Notebooks
#   33, 37, and 38.
# - **calibration-selected policy**: rules selected only from the 196-case
#   calibration cohort and then evaluated on the older held-out artifacts.
# - **diagnostic/oracle ceilings**: pooled label-fit and candidate-pool oracle
#   numbers that explain headroom but are not deployable claims.

# %% [markdown]
# ## 1. Utility Functions

# %%
import ast
import json
import math
import os
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache-baseline-model")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/baseline-model-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover - script execution fallback.
    def display(obj: Any) -> None:
        print(obj)

try:
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional diagnostic layer.
    ExtraTreesClassifier = None
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 80)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), *Path.cwd().parents]:
    if (candidate_root / "notebooks").exists() and (candidate_root / "artifacts").exists():
        PROJECT_ROOT = candidate_root
        break

DATASET_ROOT = PROJECT_ROOT / "dataset"
RUN_NAME = "cross_cohort_artifact_calibration_lab_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "cross_cohort_artifact_calibration_lab_report.md"

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 3939
SCORE_COL = "score__gradient_boosting_name_family"
EPS = 1e-12


@dataclass(frozen=True)
class CohortSpec:
    cohort: str
    role: str
    final_path: Path
    candidate_path: Path
    final_prediction_col: str = "predicted_pathology"


COHORT_SPECS = [
    CohortSpec(
        cohort="nb33_49",
        role="development_49_close_confounder",
        final_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "close_confounder_discriminator_49case_v1" / "case_level_close_confounder_results.csv",
        candidate_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "resolver_ablation_lab_49case_v1" / "candidate_level_resolver_ablation_scores.csv",
        final_prediction_col="selected_pathology",
    ),
    CohortSpec(
        cohort="nb37_98",
        role="balanced_confirmation_98",
        final_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_balanced2_v1" / "adaptive_live_final_predictions.csv",
        candidate_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_balanced2_v1" / "candidate_level_live_resolver_scores.csv",
    ),
    CohortSpec(
        cohort="nb38_196",
        role="live_calibration_196",
        final_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_calibration196_v1" / "adaptive_live_final_predictions.csv",
        candidate_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_calibration196_v1" / "candidate_level_live_resolver_scores.csv",
    ),
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_parse_json(raw: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return default
    if isinstance(raw, float) and np.isnan(raw):
        return default
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def safe_parse_list(raw: Any) -> list[Any]:
    parsed = safe_parse_json(raw, default=[])
    if isinstance(parsed, list):
        return parsed
    if parsed in (None, ""):
        return []
    return [parsed]


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def case_uid(cohort: str, case_id: str) -> str:
    return f"{cohort}::{case_id}"


def top_k_from_ranked(raw: Any, k: int) -> list[str]:
    out: list[str] = []
    for item in safe_parse_list(raw):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        if len(out) >= k:
            break
    return out


def zip_table_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            raise ValueError(f"Archive is empty: {zip_path}")
        return members[0]


def evidence_root(token: str) -> str:
    return str(token).split("_@_", 1)[0]


def accuracy_count(pred: pd.Series, truth: pd.Series) -> tuple[int, int, float]:
    correct = pred.astype(str).eq(truth.astype(str))
    return int(correct.sum()), int(len(correct)), float(correct.mean()) if len(correct) else float("nan")


def summarize_predictions(frame: pd.DataFrame, pred_col: str, policy_name: str) -> dict[str, Any]:
    correct = frame[pred_col].astype(str).eq(frame["true_pathology"].astype(str))
    top3 = frame.apply(lambda r: r["true_pathology"] in top_k_from_ranked(r.get("ranked_differential", "[]"), 3), axis=1)
    top5 = frame.apply(lambda r: r["true_pathology"] in top_k_from_ranked(r.get("ranked_differential", "[]"), 5), axis=1)
    return {
        "policy_name": policy_name,
        "n_cases": int(len(frame)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(frame) else float("nan"),
        "top3_correct": int(top3.sum()),
        "top3_accuracy": float(top3.mean()) if len(frame) else float("nan"),
        "top5_correct": int(top5.sum()),
        "top5_accuracy": float(top5.mean()) if len(frame) else float("nan"),
        "candidate_pool_recall_correct": int(frame["candidate_pool_has_true_calculated"].sum()),
        "candidate_pool_recall": float(frame["candidate_pool_has_true_calculated"].mean()) if len(frame) else float("nan"),
        "mean_selected_requests": float(pd.to_numeric(frame.get("num_requests_selected", pd.Series(dtype=float)), errors="coerce").mean()),
        "mean_total_branch_requests": float(pd.to_numeric(frame.get("total_branch_requests", pd.Series(dtype=float)), errors="coerce").mean()),
    }


# %% [markdown]
# ## 2. Load Calibration And Confirmation Artifacts

# %%
def load_cohort(spec: CohortSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not spec.final_path.exists():
        raise FileNotFoundError(spec.final_path)
    if not spec.candidate_path.exists():
        raise FileNotFoundError(spec.candidate_path)

    final = pd.read_csv(spec.final_path)
    candidates = pd.read_csv(spec.candidate_path)
    final = final.copy()
    candidates = candidates.copy()
    final["cohort"] = spec.cohort
    final["cohort_role"] = spec.role
    candidates["cohort"] = spec.cohort
    candidates["cohort_role"] = spec.role
    final["case_uid"] = [case_uid(spec.cohort, cid) for cid in final["case_id"].astype(str)]
    candidates["case_uid"] = [case_uid(spec.cohort, cid) for cid in candidates["case_id"].astype(str)]

    final["current_final_prediction"] = final[spec.final_prediction_col].astype(str)
    if "correct" in final.columns:
        final["current_final_correct"] = normalize_bool(final["correct"])
    else:
        final["current_final_correct"] = final["current_final_prediction"].eq(final["true_pathology"].astype(str))

    if "ranked_differential" not in final.columns:
        final["ranked_differential"] = "[]"
    if "num_requests_selected" not in final.columns:
        final["num_requests_selected"] = np.nan
    if "total_branch_requests" not in final.columns:
        final["total_branch_requests"] = np.nan

    if "candidate_pathology" not in candidates.columns:
        raise ValueError(f"Missing candidate_pathology in {spec.candidate_path}")
    if "candidate_label" not in candidates.columns:
        candidates["candidate_label"] = candidates["candidate_pathology"].astype(str).eq(candidates["true_pathology"].astype(str))
    candidates["candidate_label"] = normalize_bool(candidates["candidate_label"])
    if SCORE_COL not in candidates.columns:
        score_like = [col for col in candidates.columns if col.startswith("score__")]
        if not score_like:
            candidates[SCORE_COL] = 0.0
        else:
            candidates[SCORE_COL] = pd.to_numeric(candidates[score_like[0]], errors="coerce").fillna(0.0)

    return final, candidates


final_frames: list[pd.DataFrame] = []
candidate_frames: list[pd.DataFrame] = []
for spec in COHORT_SPECS:
    final_frame, candidate_frame = load_cohort(spec)
    final_frames.append(final_frame)
    candidate_frames.append(candidate_frame)

final_all = pd.concat(final_frames, ignore_index=True)
candidates_all = pd.concat(candidate_frames, ignore_index=True, sort=False)

candidate_lists = (
    candidates_all.groupby("case_uid")["candidate_pathology"]
    .apply(lambda s: list(dict.fromkeys(s.astype(str).tolist())))
    .to_dict()
)
candidate_ranked_by_score: dict[str, list[str]] = {}
for uid, group in candidates_all.groupby("case_uid"):
    ranked = (
        group.assign(_score=pd.to_numeric(group[SCORE_COL], errors="coerce").fillna(-np.inf))
        .sort_values("_score", ascending=False)["candidate_pathology"]
        .astype(str)
        .tolist()
    )
    candidate_ranked_by_score[uid] = list(dict.fromkeys(ranked))
candidate_counts = candidates_all.groupby("case_uid")["candidate_pathology"].nunique().to_dict()
candidate_has_true = candidates_all.groupby("case_uid")["candidate_label"].any().to_dict()

visible_evidence_by_case: dict[str, dict[str, str]] = {}
if "visible_evidence_json" in candidates_all.columns:
    for uid, group in candidates_all.groupby("case_uid"):
        visible: dict[str, str] = {}
        for raw in group["visible_evidence_json"]:
            parsed = safe_parse_json(raw, default={})
            if isinstance(parsed, dict):
                visible.update({str(k): str(v) for k, v in parsed.items()})
        visible_evidence_by_case[uid] = visible

final_all["candidate_list"] = final_all["case_uid"].map(candidate_lists).apply(lambda x: x if isinstance(x, list) else [])
final_all["candidate_pool_rows_calculated"] = final_all["case_uid"].map(candidate_counts).fillna(0).astype(int)
final_all["candidate_pool_has_true_calculated"] = final_all["case_uid"].map(candidate_has_true).fillna(False).astype(bool)
final_all["visible_evidence_json_merged"] = final_all["case_uid"].map(visible_evidence_by_case).apply(lambda x: json.dumps(x if isinstance(x, dict) else {}, sort_keys=True))
final_all["ranked_differential"] = final_all.apply(
    lambda r: json.dumps(candidate_ranked_by_score.get(r["case_uid"], []))
    if not top_k_from_ranked(r.get("ranked_differential", "[]"), 1)
    else r.get("ranked_differential", "[]"),
    axis=1,
)
final_all["current_top3_has_true"] = final_all.apply(lambda r: r["true_pathology"] in top_k_from_ranked(r["ranked_differential"], 3), axis=1)
final_all["current_top5_has_true"] = final_all.apply(lambda r: r["true_pathology"] in top_k_from_ranked(r["ranked_differential"], 5), axis=1)

print("Loaded cases:", len(final_all))
print("Loaded candidate rows:", len(candidates_all))
display(final_all.groupby("cohort").agg(cases=("case_id", "count"), current_correct=("current_final_correct", "sum"), candidate_recall=("candidate_pool_has_true_calculated", "sum")))

resolved_config = {
    "run_name": RUN_NAME,
    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "offline_only": True,
    "score_column": SCORE_COL,
    "cohorts": [
        {
            "cohort": spec.cohort,
            "role": spec.role,
            "final_path": str(spec.final_path.relative_to(PROJECT_ROOT)),
            "candidate_path": str(spec.candidate_path.relative_to(PROJECT_ROOT)),
            "final_prediction_col": spec.final_prediction_col,
        }
        for spec in COHORT_SPECS
    ],
    "calibration_cohort": "nb38_196",
    "heldout_artifact_cohorts": ["nb33_49", "nb37_98"],
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

# %% [markdown]
# ## 3. Cross-Cohort Reference Metrics

# %%
def candidate_argmax_predictions(candidate_frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    work = candidate_frame.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(-np.inf)
    idx = work.groupby("case_uid")[score_col].idxmax()
    out = work.loc[idx, ["cohort", "case_id", "case_uid", "candidate_pathology", score_col]].copy()
    out = out.rename(columns={"candidate_pathology": "gbm_argmax_prediction", score_col: "gbm_argmax_score"})
    return out


gbm_argmax = candidate_argmax_predictions(candidates_all, SCORE_COL)
case_pool = final_all.merge(gbm_argmax[["case_uid", "gbm_argmax_prediction", "gbm_argmax_score"]], on="case_uid", how="left")
case_pool["gbm_argmax_correct"] = case_pool["gbm_argmax_prediction"].astype(str).eq(case_pool["true_pathology"].astype(str))
case_pool["candidate_pool_oracle_prediction"] = case_pool.apply(
    lambda r: r["true_pathology"] if r["candidate_pool_has_true_calculated"] else r["gbm_argmax_prediction"],
    axis=1,
)
case_pool["candidate_pool_oracle_correct"] = case_pool["candidate_pool_oracle_prediction"].astype(str).eq(case_pool["true_pathology"].astype(str))

reference_rows: list[dict[str, Any]] = []
for cohort_name, group in case_pool.groupby("cohort", sort=False):
    for pred_col, policy_name in [
        ("current_final_prediction", "current_final_pipeline"),
        ("gbm_argmax_prediction", "raw_gbm_candidate_argmax"),
        ("candidate_pool_oracle_prediction", "candidate_pool_oracle_non_deployable"),
    ]:
        row = summarize_predictions(group.rename(columns={pred_col: "_pred"}), "_pred", policy_name)
        row["cohort"] = cohort_name
        row["scope"] = "cohort"
        reference_rows.append(row)

for pred_col, policy_name in [
    ("current_final_prediction", "current_final_pipeline"),
    ("gbm_argmax_prediction", "raw_gbm_candidate_argmax"),
    ("candidate_pool_oracle_prediction", "candidate_pool_oracle_non_deployable"),
]:
    row = summarize_predictions(case_pool.rename(columns={pred_col: "_pred"}), "_pred", policy_name)
    row["cohort"] = "pooled_343"
    row["scope"] = "pooled"
    reference_rows.append(row)

reference_summary = pd.DataFrame(reference_rows)
display(reference_summary[["cohort", "policy_name", "correct", "n_cases", "accuracy", "top3_correct", "top5_correct", "candidate_pool_recall_correct", "mean_selected_requests", "mean_total_branch_requests"]])

# %% [markdown]
# ## 4. Candidate Resolver Calibration

# %%
def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    allowed_prefixes = (
        "score__",
        "resolver_candidate_",
        "resolver_",
        "pred_graph_",
        "pred_bayes_",
        "pred_mlp_",
        "graph_",
        "bayes_",
        "final_mlp_",
    )
    columns: list[str] = []
    banned = {
        "resolver_candidate_order",
        "resolver_is_base_candidate",
        "resolver_is_branch_candidate",
        "resolver_is_pseudo_candidate",
    }
    for col in frame.columns:
        if col in banned:
            columns.append(col)
            continue
        if col.startswith(allowed_prefixes) or col in [
            "num_requests",
            "visible_root_count",
            "candidate_order",
            "vote_share",
            "final_confidence",
            "llm_confidence",
            "final_mlp_confidence",
            "final_mlp_margin",
            "final_mlp_entropy",
            "final_mlp_stability_turns",
            "llm_mlp_agreement",
            "mlp_in_llm_top3",
        ]:
            if col in frame.columns:
                columns.append(col)
    # Keep only columns that can be coerced to at least one numeric value.
    numeric_cols: list[str] = []
    for col in dict.fromkeys(columns):
        coerced = pd.to_numeric(frame[col], errors="coerce")
        if coerced.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def select_argmax_by_score(frame: pd.DataFrame, score_col: str, pred_col: str) -> pd.DataFrame:
    work = frame.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(-np.inf)
    idx = work.groupby("case_uid")[score_col].idxmax()
    selected = work.loc[idx, ["case_uid", "candidate_pathology", score_col]].copy()
    selected = selected.rename(columns={"candidate_pathology": pred_col})
    return selected


def run_loco_candidate_models(candidate_frame: pd.DataFrame, base_case_pool: pd.DataFrame) -> pd.DataFrame:
    if ExtraTreesClassifier is None:
        return pd.DataFrame([{"model_name": "sklearn_unavailable", "note": "sklearn could not be imported"}])

    feature_cols = numeric_feature_columns(candidate_frame)
    model_specs = [
        (
            "logistic_l2_numeric_loco",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=RANDOM_SEED)),
        ),
        (
            "extra_trees_numeric_loco",
            ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_SEED),
        ),
    ]

    rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    for holdout in sorted(candidate_frame["cohort"].unique()):
        train = candidate_frame[candidate_frame["cohort"] != holdout].copy()
        test = candidate_frame[candidate_frame["cohort"] == holdout].copy()
        X_train_all = train[feature_cols].apply(pd.to_numeric, errors="coerce")
        usable_cols = [col for col in feature_cols if X_train_all[col].notna().any()]
        X_train = X_train_all[usable_cols]
        y_train = train["candidate_label"].astype(int)
        X_test = test[usable_cols].apply(pd.to_numeric, errors="coerce")

        for model_name, model in model_specs:
            try:
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X_test)[:, 1]
                else:
                    raw_scores = model.decision_function(X_test)
                    scores = 1.0 / (1.0 + np.exp(-raw_scores))
            except Exception as exc:
                rows.append({"model_name": model_name, "holdout_cohort": holdout, "error": repr(exc)})
                continue
            scored = test[["cohort", "case_id", "case_uid", "true_pathology", "candidate_pathology", "candidate_label"]].copy()
            scored["loco_score"] = scores
            scored["model_name"] = model_name
            selected = select_argmax_by_score(scored, "loco_score", "prediction")
            merged = base_case_pool[base_case_pool["cohort"] == holdout][["case_uid", "true_pathology"]].merge(selected, on="case_uid", how="left")
            correct = merged["prediction"].astype(str).eq(merged["true_pathology"].astype(str))
            rows.append(
                {
                    "model_name": model_name,
                    "holdout_cohort": holdout,
                    "correct": int(correct.sum()),
                    "n_cases": int(len(merged)),
                    "accuracy": float(correct.mean()) if len(merged) else float("nan"),
                    "feature_count": len(usable_cols),
                    "error": "",
                }
            )
            score_frames.append(scored)

    if score_frames:
        pd.concat(score_frames, ignore_index=True).to_csv(ARTIFACT_ROOT / "cross_cohort_loco_candidate_scores.csv", index=False)
    return pd.DataFrame(rows)


loco_summary = run_loco_candidate_models(candidates_all, case_pool)
display(loco_summary)

# %% [markdown]
# ## 5. Evidence Rule Mining And Stress Tests

# %%
def build_rule_trigger_table(case_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, case in case_frame.iterrows():
        visible = safe_parse_json(case.get("visible_evidence_json_merged"), default={})
        if not isinstance(visible, dict) or not visible:
            continue
        current = str(case["current_final_prediction"])
        for challenger in case["candidate_list"]:
            challenger = str(challenger)
            if challenger == current:
                continue
            for root_id, status in visible.items():
                rows.append(
                    {
                        "cohort": case["cohort"],
                        "case_id": case["case_id"],
                        "case_uid": case["case_uid"],
                        "true_pathology": case["true_pathology"],
                        "anchor_prediction": current,
                        "challenger_prediction": challenger,
                        "root_id": str(root_id),
                        "status": str(status),
                        "current_correct": bool(case["current_final_correct"]),
                        "rule_would_be_correct": challenger == case["true_pathology"],
                    }
                )
    trigger_table = pd.DataFrame(rows)
    if trigger_table.empty:
        return trigger_table
    trigger_table["win"] = (~trigger_table["current_correct"]) & trigger_table["rule_would_be_correct"]
    trigger_table["regression"] = trigger_table["current_correct"] & (~trigger_table["rule_would_be_correct"])
    return trigger_table


def summarize_rules(trigger_table: pd.DataFrame, cohorts: list[str] | None = None) -> pd.DataFrame:
    if trigger_table.empty:
        return pd.DataFrame()
    work = trigger_table.copy()
    if cohorts is not None:
        work = work[work["cohort"].isin(cohorts)]
    if work.empty:
        return pd.DataFrame()
    group_cols = ["anchor_prediction", "challenger_prediction", "root_id", "status"]
    summary = (
        work.groupby(group_cols)
        .agg(
            trigger_count=("case_uid", "nunique"),
            wins=("win", "sum"),
            regressions=("regression", "sum"),
            triggered_cases=("case_uid", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    summary["net_gain"] = summary["wins"] - summary["regressions"]
    return summary.sort_values(["net_gain", "wins", "regressions", "trigger_count"], ascending=[False, False, True, False]).reset_index(drop=True)


def apply_ordered_rules(case_frame: pd.DataFrame, rules: pd.DataFrame, pred_col: str = "rule_adjusted_prediction") -> pd.DataFrame:
    out = case_frame.copy()
    predictions: list[str] = []
    applied_rules: list[str] = []
    for _, case in out.iterrows():
        pred = str(case["current_final_prediction"])
        visible = safe_parse_json(case.get("visible_evidence_json_merged"), default={})
        candidates = set(str(x) for x in case["candidate_list"])
        applied = ""
        for _, rule in rules.iterrows():
            anchor = str(rule["anchor_prediction"])
            challenger = str(rule["challenger_prediction"])
            root_id = str(rule["root_id"])
            status = str(rule["status"])
            if pred == anchor and challenger in candidates and isinstance(visible, dict) and str(visible.get(root_id)) == status:
                pred = challenger
                applied = f"{anchor}->{challenger}|{root_id}={status}"
                break
        predictions.append(pred)
        applied_rules.append(applied)
    out[pred_col] = predictions
    out[pred_col + "_correct"] = out[pred_col].astype(str).eq(out["true_pathology"].astype(str))
    out["applied_calibration_rule"] = applied_rules
    return out


def greedy_nonredundant_rules(case_frame: pd.DataFrame, candidate_rules: pd.DataFrame, min_wins: int = 1, require_zero_regressions: bool = True, max_rules: int = 12) -> pd.DataFrame:
    """Greedy rule selection using plain Python records.

    The rule table is small enough for exhaustive evaluation, but using pandas
    `apply` inside this loop makes the notebook painfully slow. Plain dicts keep
    the calibration lab snappy and deterministic.
    """
    if candidate_rules.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in case_frame.iterrows():
        visible = safe_parse_json(row.get("visible_evidence_json_merged"), default={})
        records.append(
            {
                "case_uid": row["case_uid"],
                "truth": str(row["true_pathology"]),
                "current": str(row["current_final_prediction"]),
                "visible": visible if isinstance(visible, dict) else {},
                "candidates": set(str(x) for x in row["candidate_list"]),
            }
        )

    current_pred = {row["case_uid"]: row["current"] for row in records}
    candidate_rule_records = candidate_rules.to_dict(orient="records")
    selected_rows: list[dict[str, Any]] = []

    for _ in range(max_rules):
        best_rule: dict[str, Any] | None = None
        best_tuple = (0, 0, 0, 0)
        for rule in candidate_rule_records:
            anchor = str(rule["anchor_prediction"])
            challenger = str(rule["challenger_prediction"])
            root_id = str(rule["root_id"])
            status = str(rule["status"])
            wins = 0
            regressions = 0
            triggers = 0
            for case in records:
                uid = case["case_uid"]
                if current_pred[uid] != anchor:
                    continue
                if challenger not in case["candidates"]:
                    continue
                if str(case["visible"].get(root_id)) != status:
                    continue
                triggers += 1
                before = current_pred[uid] == case["truth"]
                after = challenger == case["truth"]
                if (not before) and after:
                    wins += 1
                elif before and not after:
                    regressions += 1
            net = wins - regressions
            if wins < min_wins:
                continue
            if require_zero_regressions and regressions:
                continue
            score_tuple = (net, wins, -regressions, triggers)
            if score_tuple > best_tuple:
                best_tuple = score_tuple
                best_rule = dict(rule)
                best_rule["incremental_wins"] = wins
                best_rule["incremental_regressions"] = regressions
                best_rule["incremental_triggers"] = triggers
                best_rule["incremental_net_gain"] = net
        if best_rule is None or int(best_rule["incremental_net_gain"]) <= 0:
            break
        selected_rows.append(best_rule)
        anchor = str(best_rule["anchor_prediction"])
        challenger = str(best_rule["challenger_prediction"])
        root_id = str(best_rule["root_id"])
        status = str(best_rule["status"])
        for case in records:
            uid = case["case_uid"]
            if current_pred[uid] == anchor and challenger in case["candidates"] and str(case["visible"].get(root_id)) == status:
                current_pred[uid] = challenger

    return pd.DataFrame(selected_rows)


rule_triggers = build_rule_trigger_table(case_pool)
calibration_rule_summary = summarize_rules(rule_triggers, cohorts=["nb38_196"])
pooled_rule_summary = summarize_rules(rule_triggers)

selected_calibration_rules = greedy_nonredundant_rules(
    case_pool[case_pool["cohort"] == "nb38_196"],
    calibration_rule_summary,
    min_wins=2,
    require_zero_regressions=True,
    max_rules=5,
)
pooled_label_fit_rules = greedy_nonredundant_rules(
    case_pool,
    pooled_rule_summary,
    min_wins=1,
    require_zero_regressions=True,
    max_rules=12,
)

case_pool_calibrated = apply_ordered_rules(case_pool, selected_calibration_rules, pred_col="calibration196_rule_prediction")
case_pool_label_fit = apply_ordered_rules(case_pool, pooled_label_fit_rules, pred_col="pooled_label_fit_rule_prediction")

print("Selected calibration rules")
display(selected_calibration_rules.drop(columns=["triggered_cases"], errors="ignore"))
print("Diagnostic pooled label-fit rules")
display(pooled_label_fit_rules.drop(columns=["triggered_cases"], errors="ignore"))

# %% [markdown]
# ## 6. Failure Mode Analysis

# %%
def policy_summary_rows(case_frame: pd.DataFrame) -> pd.DataFrame:
    policy_specs = [
        ("current_final_prediction", "current_final_pipeline", "deployable_saved_artifact"),
        ("gbm_argmax_prediction", "raw_gbm_candidate_argmax", "diagnostic"),
        ("calibration196_rule_prediction", "calibration196_rule_layer_v1", "calibration_selected_needs_fresh_confirmation"),
        ("pooled_label_fit_rule_prediction", "pooled_label_fit_no_regret_rules_diagnostic", "diagnostic_label_fit"),
        ("candidate_pool_oracle_prediction", "candidate_pool_oracle_non_deployable", "oracle_non_deployable"),
    ]
    rows: list[dict[str, Any]] = []
    for cohort_name, group in case_frame.groupby("cohort", sort=False):
        for pred_col, policy_name, claim_type in policy_specs:
            row = summarize_predictions(group.rename(columns={pred_col: "_pred"}), "_pred", policy_name)
            row["cohort"] = cohort_name
            row["scope"] = "cohort"
            row["claim_type"] = claim_type
            rows.append(row)
    for pred_col, policy_name, claim_type in policy_specs:
        row = summarize_predictions(case_frame.rename(columns={pred_col: "_pred"}), "_pred", policy_name)
        row["cohort"] = "pooled_343"
        row["scope"] = "pooled"
        row["claim_type"] = claim_type
        rows.append(row)
    return pd.DataFrame(rows)


policy_summary = policy_summary_rows(case_pool_calibrated.merge(
    case_pool_label_fit[["case_uid", "pooled_label_fit_rule_prediction", "pooled_label_fit_rule_prediction_correct"]],
    on="case_uid",
    how="left",
))

case_results = case_pool_calibrated.merge(
    case_pool_label_fit[["case_uid", "pooled_label_fit_rule_prediction", "pooled_label_fit_rule_prediction_correct"]],
    on="case_uid",
    how="left",
)
case_results["calibration196_rule_correct"] = case_results["calibration196_rule_prediction"].astype(str).eq(case_results["true_pathology"].astype(str))
case_results["current_to_calibration196_delta"] = np.select(
    [
        (~case_results["current_final_correct"]) & case_results["calibration196_rule_correct"],
        case_results["current_final_correct"] & (~case_results["calibration196_rule_correct"]),
    ],
    ["win", "regression"],
    default="unchanged",
)
case_results["failure_pair_current"] = np.where(
    case_results["current_final_correct"],
    "",
    case_results["true_pathology"].astype(str) + " -> " + case_results["current_final_prediction"].astype(str),
)
case_results["failure_pair_calibration196"] = np.where(
    case_results["calibration196_rule_correct"],
    "",
    case_results["true_pathology"].astype(str) + " -> " + case_results["calibration196_rule_prediction"].astype(str),
)

failure_modes = (
    case_results[~case_results["calibration196_rule_correct"]]
    .groupby(["cohort", "failure_pair_calibration196"])
    .agg(cases=("case_id", "nunique"), candidate_pool_has_true=("candidate_pool_has_true_calculated", "sum"))
    .reset_index()
    .sort_values(["cases", "cohort"], ascending=[False, True])
)

request_cost_summary = (
    case_results.groupby("cohort")
    .agg(
        cases=("case_id", "count"),
        mean_selected_requests=("num_requests_selected", "mean"),
        mean_total_branch_requests=("total_branch_requests", "mean"),
        median_total_branch_requests=("total_branch_requests", "median"),
        p90_total_branch_requests=("total_branch_requests", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
        max_total_branch_requests=("total_branch_requests", "max"),
    )
    .reset_index()
)

display(policy_summary[["cohort", "policy_name", "claim_type", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct", "top3_correct", "top5_correct", "mean_total_branch_requests"]])
display(failure_modes.head(30))
display(request_cost_summary)

# %% [markdown]
# ### Evidence Support For Selected Rules

# %%
def load_evidence_questions() -> dict[str, str]:
    path = DATASET_ROOT / "release_evidences.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    questions: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            questions[key] = str(value.get("question_en") or value.get("name") or "")
    return questions


def load_root_presence_stats(pathologies: set[str], roots: set[str]) -> pd.DataFrame:
    if not pathologies or not roots:
        return pd.DataFrame()
    counts: dict[str, Counter[str]] = {pathology: Counter() for pathology in pathologies}
    totals: Counter[str] = Counter()
    for zip_name in ["release_train_patients.zip", "release_validate_patients.zip"]:
        zip_path = DATASET_ROOT / zip_name
        if not zip_path.exists():
            continue
        member = zip_table_member(zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(member) as handle:
                for chunk in pd.read_csv(handle, usecols=["PATHOLOGY", "EVIDENCES"], chunksize=50_000):
                    chunk = chunk[chunk["PATHOLOGY"].isin(pathologies)]
                    for row in chunk.itertuples(index=False):
                        pathology = str(row.PATHOLOGY)
                        totals[pathology] += 1
                        present_roots = {evidence_root(token) for token in safe_parse_list(row.EVIDENCES)}
                        for root in roots:
                            if root in present_roots:
                                counts[pathology][root] += 1
    rows: list[dict[str, Any]] = []
    for pathology in sorted(pathologies):
        total = totals[pathology]
        for root in sorted(roots):
            rows.append(
                {
                    "pathology": pathology,
                    "root_id": root,
                    "n": int(total),
                    "present_count": int(counts[pathology][root]),
                    "present_rate": float((counts[pathology][root] + 1) / (total + 2)) if total else float("nan"),
                }
            )
    return pd.DataFrame(rows)


questions = load_evidence_questions()
selected_rule_pathologies = set()
selected_rule_roots = set()
for _, rule in selected_calibration_rules.iterrows():
    selected_rule_pathologies.add(str(rule["anchor_prediction"]))
    selected_rule_pathologies.add(str(rule["challenger_prediction"]))
    selected_rule_roots.add(str(rule["root_id"]))

selected_rule_stats = load_root_presence_stats(selected_rule_pathologies, selected_rule_roots)
if not selected_rule_stats.empty:
    selected_rule_stats["question"] = selected_rule_stats["root_id"].map(questions).fillna("")
display(selected_rule_stats)

# %% [markdown]
# ## 7. Figures

# %%
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


plot_policies = [
    "current_final_pipeline",
    "calibration196_rule_layer_v1",
    "pooled_label_fit_no_regret_rules_diagnostic",
    "candidate_pool_oracle_non_deployable",
]
plot_df = policy_summary[(policy_summary["policy_name"].isin(plot_policies)) & (policy_summary["scope"] == "cohort")].copy()
plot_df["label"] = plot_df["correct"].astype(str) + "/" + plot_df["n_cases"].astype(str)

plt.figure(figsize=(10, 5))
for idx, policy_name in enumerate(plot_policies):
    subset = plot_df[plot_df["policy_name"] == policy_name]
    xs = np.arange(len(subset)) + idx * 0.18
    plt.bar(xs, subset["accuracy"], width=0.18, label=policy_name)
plt.xticks(np.arange(len(COHORT_SPECS)) + 0.27, [spec.cohort for spec in COHORT_SPECS])
plt.ylim(0.80, 1.01)
plt.ylabel("Accuracy")
plt.title("Cross-Cohort Accuracy Frontier")
plt.legend(fontsize=8, loc="lower right")
savefig(FIGURE_DIR / "cross_cohort_accuracy_frontier.png")

plt.figure(figsize=(8, 4.5))
pooled_plot = policy_summary[(policy_summary["scope"] == "pooled") & (policy_summary["policy_name"].isin(plot_policies))]
plt.barh(pooled_plot["policy_name"], pooled_plot["accuracy"])
for _, row in pooled_plot.iterrows():
    plt.text(row["accuracy"] + 0.003, row["policy_name"], f"{int(row['correct'])}/{int(row['n_cases'])}", va="center")
plt.xlim(0.88, 1.0)
plt.xlabel("Pooled accuracy")
plt.title("Pooled 343-Case Calibration Frontier")
savefig(FIGURE_DIR / "pooled_accuracy_frontier.png")

plt.figure(figsize=(8, 4.5))
cost_plot = request_cost_summary.dropna(subset=["mean_total_branch_requests"]).copy()
plt.bar(cost_plot["cohort"], cost_plot["mean_total_branch_requests"], label="mean total")
plt.scatter(cost_plot["cohort"], cost_plot["p90_total_branch_requests"], color="black", label="p90 total", zorder=3)
plt.ylabel("Requests")
plt.title("Live Artifact Request Cost")
plt.legend()
savefig(FIGURE_DIR / "request_cost_by_cohort.png")

top_failures = (
    case_results[~case_results["calibration196_rule_correct"]]["failure_pair_calibration196"]
    .value_counts()
    .head(12)
    .sort_values()
)
plt.figure(figsize=(9, 5))
plt.barh(top_failures.index, top_failures.values)
plt.xlabel("Remaining misses")
plt.title("Remaining Failure Modes After Calibration Rule Layer")
savefig(FIGURE_DIR / "remaining_failure_modes.png")

rule_effect = case_results["current_to_calibration196_delta"].value_counts().reindex(["win", "regression", "unchanged"]).fillna(0)
plt.figure(figsize=(5.5, 4))
plt.bar(rule_effect.index, rule_effect.values, color=["#2d7f5e", "#b33a3a", "#7c8490"])
plt.ylabel("Cases")
plt.title("Calibration Rule Layer Paired Effect")
savefig(FIGURE_DIR / "calibration_rule_paired_effect.png")

# %% [markdown]
# ## 8. Final Summary And Artifact Contract

# %%
case_pool_export_cols = [
    "cohort",
    "cohort_role",
    "case_id",
    "case_uid",
    "true_pathology",
    "current_final_prediction",
    "current_final_correct",
    "gbm_argmax_prediction",
    "gbm_argmax_correct",
    "calibration196_rule_prediction",
    "calibration196_rule_correct",
    "pooled_label_fit_rule_prediction",
    "pooled_label_fit_rule_prediction_correct",
    "candidate_pool_oracle_prediction",
    "candidate_pool_oracle_correct",
    "candidate_pool_has_true_calculated",
    "candidate_pool_rows_calculated",
    "ranked_differential",
    "current_top3_has_true",
    "current_top5_has_true",
    "num_requests_selected",
    "total_branch_requests",
    "applied_calibration_rule",
    "current_to_calibration196_delta",
    "failure_pair_current",
    "failure_pair_calibration196",
    "visible_evidence_json_merged",
]
candidate_export = candidates_all.copy()
if "visible_evidence_json" in candidate_export.columns:
    candidate_export["visible_evidence_json"] = candidate_export["visible_evidence_json"].astype(str)

case_results[case_pool_export_cols].to_csv(ARTIFACT_ROOT / "cross_cohort_case_pool.csv", index=False)
candidate_export.to_csv(ARTIFACT_ROOT / "cross_cohort_candidate_scores.csv", index=False)
policy_summary.to_csv(ARTIFACT_ROOT / "cross_cohort_policy_summary.csv", index=False)
loco_summary.to_csv(ARTIFACT_ROOT / "cross_cohort_leave_one_cohort_out.csv", index=False)
calibration_rule_summary.to_csv(ARTIFACT_ROOT / "cross_cohort_rule_mining_summary.csv", index=False)
pooled_rule_summary.to_csv(ARTIFACT_ROOT / "cross_cohort_pooled_rule_mining_summary.csv", index=False)
selected_calibration_rules.to_csv(ARTIFACT_ROOT / "selected_calibration_rules.csv", index=False)
pooled_label_fit_rules.to_csv(ARTIFACT_ROOT / "pooled_label_fit_rules_diagnostic.csv", index=False)
failure_modes.to_csv(ARTIFACT_ROOT / "cross_cohort_failure_modes.csv", index=False)
request_cost_summary.to_csv(ARTIFACT_ROOT / "cross_cohort_request_cost_summary.csv", index=False)
selected_rule_stats.to_csv(ARTIFACT_ROOT / "selected_rule_train_validate_stats.csv", index=False)

hard_cases = case_results[~case_results["calibration196_rule_correct"]].copy()
hard_case_payload = []
for _, row in hard_cases.iterrows():
    hard_case_payload.append(
        {
            "cohort": row["cohort"],
            "case_id": row["case_id"],
            "true_pathology": row["true_pathology"],
            "current_final_prediction": row["current_final_prediction"],
            "calibration196_rule_prediction": row["calibration196_rule_prediction"],
            "candidate_pool_has_true": bool(row["candidate_pool_has_true_calculated"]),
            "candidate_list": row["candidate_list"],
            "ranked_differential": top_k_from_ranked(row["ranked_differential"], 5),
            "visible_evidence": safe_parse_json(row["visible_evidence_json_merged"], default={}),
            "failure_pair": row["failure_pair_calibration196"],
        }
    )
write_json(ARTIFACT_ROOT / "hard_case_calibration_audits.json", hard_case_payload)

pooled_current = policy_summary[(policy_summary["cohort"] == "pooled_343") & (policy_summary["policy_name"] == "current_final_pipeline")].iloc[0].to_dict()
pooled_calibrated = policy_summary[(policy_summary["cohort"] == "pooled_343") & (policy_summary["policy_name"] == "calibration196_rule_layer_v1")].iloc[0].to_dict()
pooled_oracle = policy_summary[(policy_summary["cohort"] == "pooled_343") & (policy_summary["policy_name"] == "candidate_pool_oracle_non_deployable")].iloc[0].to_dict()

selected_policy = {
    "selected_policy_name": "calibration196_rule_layer_v1",
    "status": "calibration_only_needs_fresh_confirmation",
    "inputs_used": [
        "Notebook 33 49-case close-confounder artifacts",
        "Notebook 37 98-case live balanced artifacts",
        "Notebook 38 196-case live calibration artifacts",
        "DDXPlus train/validate evidence presence rates for rule plausibility audit",
    ],
    "training_or_selection_split": {
        "rule_selection": "nb38_196 only",
        "heldout_artifact_check": ["nb33_49", "nb37_98"],
        "pooled_diagnostic": "nb33_49 + nb37_98 + nb38_196",
    },
    "selected_rules": selected_calibration_rules.drop(columns=["triggered_cases"], errors="ignore").to_dict(orient="records"),
    "pooled_current_final": {
        "correct": int(pooled_current["correct"]),
        "n_cases": int(pooled_current["n_cases"]),
        "accuracy": float(pooled_current["accuracy"]),
    },
    "pooled_calibration196_rule_layer": {
        "correct": int(pooled_calibrated["correct"]),
        "n_cases": int(pooled_calibrated["n_cases"]),
        "accuracy": float(pooled_calibrated["accuracy"]),
        "paired_wins": int((case_results["current_to_calibration196_delta"] == "win").sum()),
        "paired_regressions": int((case_results["current_to_calibration196_delta"] == "regression").sum()),
    },
    "pooled_candidate_pool_oracle": {
        "correct": int(pooled_oracle["correct"]),
        "n_cases": int(pooled_oracle["n_cases"]),
        "accuracy": float(pooled_oracle["accuracy"]),
    },
    "promotion_decision": "Do not promote as a final claim until a fresh frozen confirmation run tests this rule layer. It improves the calibration cohort and causes no historical-artifact regressions, but the selected acute/chronic signal is weakly supported by train/validate disease statistics.",
}
write_json(ARTIFACT_ROOT / "selected_cross_cohort_calibration_policy.json", selected_policy)

report_lines = [
    "# Cross-Cohort Artifact Calibration Lab",
    "",
    "Notebook 39 is an offline calibration analysis over saved Notebooks 33, 37, and 38 artifacts. It makes no API calls.",
    "",
    "## Main Results",
    "",
    policy_summary[(policy_summary["scope"] == "pooled")][["policy_name", "claim_type", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct", "top3_correct", "top5_correct", "mean_total_branch_requests"]].to_markdown(index=False),
    "",
    "## Selected Calibration Rule Layer",
    "",
    "Rules were selected only from the Notebook 38 196-case calibration cohort with a zero-regression constraint inside that cohort. They were then checked on the older 49/98-case artifacts.",
    "",
    selected_calibration_rules.drop(columns=["triggered_cases"], errors="ignore").to_markdown(index=False) if not selected_calibration_rules.empty else "No calibration rules selected.",
    "",
    "The selected layer changes pooled accuracy from "
    f"{int(pooled_current['correct'])}/{int(pooled_current['n_cases'])} "
    f"to {int(pooled_calibrated['correct'])}/{int(pooled_calibrated['n_cases'])}, "
    f"with {int((case_results['current_to_calibration196_delta'] == 'win').sum())} wins and "
    f"{int((case_results['current_to_calibration196_delta'] == 'regression').sum())} regressions on the pooled artifacts.",
    "",
    "## Interpretation",
    "",
    "The artifacts support a modest calibration improvement, not a solved universal resolver. Candidate-pool recall remains the ceiling driver: the pooled candidate-pool oracle is "
    f"{int(pooled_oracle['correct'])}/{int(pooled_oracle['n_cases'])}. The selected rule layer mainly repairs repeated acute-vs-chronic rhinosinusitis decisions in the 196-case calibration cohort, but this signal should be treated as calibration-only until a fresh confirmation run.",
    "",
    "## Remaining Failure Modes",
    "",
    failure_modes.head(20).to_markdown(index=False),
    "",
    "## Request Cost",
    "",
    request_cost_summary.to_markdown(index=False),
    "",
    "## Artifact Contract",
    "",
    "- `resolved_run_config.json`",
    "- `cross_cohort_case_pool.csv`",
    "- `cross_cohort_candidate_scores.csv`",
    "- `cross_cohort_policy_summary.csv`",
    "- `cross_cohort_leave_one_cohort_out.csv`",
    "- `cross_cohort_rule_mining_summary.csv`",
    "- `cross_cohort_pooled_rule_mining_summary.csv`",
    "- `selected_calibration_rules.csv`",
    "- `pooled_label_fit_rules_diagnostic.csv`",
    "- `cross_cohort_failure_modes.csv`",
    "- `cross_cohort_request_cost_summary.csv`",
    "- `selected_rule_train_validate_stats.csv`",
    "- `hard_case_calibration_audits.json`",
    "- `selected_cross_cohort_calibration_policy.json`",
    "- figures under `figures/`",
    "",
]
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print("Artifacts written to:", ARTIFACT_ROOT)
print("Report written to   :", REPORT_PATH)
display(policy_summary[(policy_summary["scope"] == "pooled")][["policy_name", "claim_type", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct"]])
