from __future__ import annotations

# %% [markdown]
# # Notebook 40: Synthetic-to-Live Listwise Resolver
#
# Offline resolver lab. No live API calls.
#
# The goal is to test whether the resolver bottleneck can be attacked cleanly:
#
# ```text
# DDXPlus train/validate synthetic partial states -> candidate-pool resolver
# saved live candidate pools -> transfer evaluation
# ```
#
# The key scientific distinction:
#
# - synthetic DDXPlus states teach evidence-to-disease compatibility at scale
# - live artifacts test whether that compatibility transfers to the actual
#   candidate pools produced by our agentic system
# - artifact labels may be used only in explicit calibration/diagnostic sections,
#   with leave-one-cohort-out evaluation separated from pooled label-fit results

# %% [markdown]
# ## 1. Utility Functions

# %%
import ast
import json
import os
import re
import warnings
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
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj: Any) -> None:
        print(obj)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 160)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), *Path.cwd().parents]:
    if (candidate_root / "notebooks").exists() and (candidate_root / "artifacts").exists():
        PROJECT_ROOT = candidate_root
        break

RUN_NAME = "synthetic_to_live_listwise_resolver_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "synthetic_to_live_listwise_resolver_report.md"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 4040
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def log_step(message: str) -> None:
    print(f"[Notebook40] {message}", flush=True)

SYNTHETIC_FEATURE_PATH = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_calibration196_v1" / "candidate_resolver_train_validate_features.csv"
NB39_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "cross_cohort_artifact_calibration_lab_v1" / "cross_cohort_policy_summary.csv"

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
    "cohort",
    "cohort_role",
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


@dataclass(frozen=True)
class LiveCohortSpec:
    cohort: str
    role: str
    candidate_path: Path
    final_path: Path
    final_prediction_col: str


LIVE_COHORTS = [
    LiveCohortSpec(
        cohort="nb33_49",
        role="development_49_candidate_pool",
        candidate_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1" / "candidate_level_live_scores.csv",
        final_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "close_confounder_discriminator_49case_v1" / "case_level_close_confounder_results.csv",
        final_prediction_col="selected_pathology",
    ),
    LiveCohortSpec(
        cohort="nb37_98",
        role="fresh_balanced_confirmation_98",
        candidate_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_balanced2_v1" / "candidate_level_live_resolver_scores.csv",
        final_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_balanced2_v1" / "adaptive_live_final_predictions.csv",
        final_prediction_col="predicted_pathology",
    ),
    LiveCohortSpec(
        cohort="nb38_196",
        role="live_calibration_196",
        candidate_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_calibration196_v1" / "candidate_level_live_resolver_scores.csv",
        final_path=PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_live_calibration196_v1" / "adaptive_live_final_predictions.csv",
        final_prediction_col="predicted_pathology",
    ),
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_parse_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
    return parsed if isinstance(parsed, list) else [parsed]


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def clean_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame[columns].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=-1e6, upper=1e6)


def add_name_family_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    text = out["candidate_pathology"].fillna("").astype(str).str.lower()
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


def add_resolver_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in BASE_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
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
    out = add_name_family_features(out)
    return out.copy()


def make_live_training_schema(live_df: pd.DataFrame, spec: LiveCohortSpec) -> pd.DataFrame:
    mapping = {
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
    }
    out = pd.DataFrame()
    for source_col, target_col in mapping.items():
        if source_col in live_df.columns:
            out[target_col] = live_df[source_col]
    out["synthetic_state_id"] = spec.cohort + "::" + out["case_id"].astype(str)
    out["split"] = "artifact"
    out["cohort"] = spec.cohort
    out["cohort_role"] = spec.role
    if "candidate_label" not in out.columns:
        out["candidate_label"] = out["candidate_pathology"].astype(str).eq(out["true_pathology"].astype(str))
    out["candidate_label"] = bool_series(out["candidate_label"])
    out["candidate_source"] = out.get("candidate_role", "live")
    return out


def select_unique_by_score(frame: pd.DataFrame, score_col: str, prediction_col: str) -> pd.DataFrame:
    work = frame.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(-1e9)
    rows = []
    for state_id, group in work.groupby("synthetic_state_id", sort=False):
        unique = (
            group.sort_values(score_col, ascending=False)
            .drop_duplicates("candidate_pathology", keep="first")
            .sort_values(score_col, ascending=False)
            .copy()
        )
        if unique.empty:
            continue
        top = unique.iloc[0]
        rows.append(
            {
                "cohort": top.get("cohort", ""),
                "case_id": top.get("case_id", state_id),
                "synthetic_state_id": state_id,
                "true_pathology": top["true_pathology"],
                prediction_col: top["candidate_pathology"],
                "selected_score": float(top[score_col]),
                "candidate_pool_has_true": bool(group["candidate_label"].astype(bool).any()),
                "candidate_pool_rows": int(len(group)),
                "candidate_pool_unique": int(unique["candidate_pathology"].nunique()),
                "ranked_differential": json.dumps(unique["candidate_pathology"].astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out[prediction_col + "_correct"] = out[prediction_col].astype(str).eq(out["true_pathology"].astype(str))
    return out


def topk_accuracy_from_ranked(frame: pd.DataFrame, ranked_col: str, k: int) -> float:
    if frame.empty:
        return float("nan")
    hits = []
    for _, row in frame.iterrows():
        ranked = [str(x[0] if isinstance(x, (list, tuple)) and x else x) for x in safe_parse_list(row[ranked_col])]
        hits.append(str(row["true_pathology"]) in ranked[:k])
    return float(np.mean(hits))


def summarize_case_predictions(frame: pd.DataFrame, prediction_col: str, policy_name: str, claim_type: str) -> dict[str, Any]:
    correct = frame[prediction_col].astype(str).eq(frame["true_pathology"].astype(str))
    return {
        "policy_name": policy_name,
        "claim_type": claim_type,
        "cohort": str(frame["cohort"].iloc[0]) if "cohort" in frame.columns and len(frame) else "pooled",
        "n_cases": int(len(frame)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(frame) else float("nan"),
        "candidate_pool_recall_correct": int(frame["candidate_pool_has_true"].sum()) if "candidate_pool_has_true" in frame.columns else np.nan,
        "candidate_pool_recall": float(frame["candidate_pool_has_true"].mean()) if "candidate_pool_has_true" in frame.columns and len(frame) else np.nan,
        "top3_accuracy": topk_accuracy_from_ranked(frame, "ranked_differential", 3) if "ranked_differential" in frame.columns else np.nan,
        "top5_accuracy": topk_accuracy_from_ranked(frame, "ranked_differential", 5) if "ranked_differential" in frame.columns else np.nan,
    }


# %% [markdown]
# ## 2. Load Synthetic And Live Candidate Pools

# %%
resolved_config = {
    "run_name": RUN_NAME,
    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "offline_only": True,
    "synthetic_feature_path": str(SYNTHETIC_FEATURE_PATH.relative_to(PROJECT_ROOT)),
    "live_cohorts": [
        {
            "cohort": spec.cohort,
            "role": spec.role,
            "candidate_path": str(spec.candidate_path.relative_to(PROJECT_ROOT)),
            "final_path": str(spec.final_path.relative_to(PROJECT_ROOT)),
            "final_prediction_col": spec.final_prediction_col,
        }
        for spec in LIVE_COHORTS
    ],
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

if not SYNTHETIC_FEATURE_PATH.exists():
    raise FileNotFoundError(SYNTHETIC_FEATURE_PATH)

log_step("loading synthetic candidate features and live candidate pools")
synthetic_raw = pd.read_csv(SYNTHETIC_FEATURE_PATH)
synthetic_features = add_resolver_features(synthetic_raw)
synthetic_features["candidate_label"] = bool_series(synthetic_features["candidate_label"])
feature_columns = [
    col
    for col in synthetic_features.columns
    if col not in EXCLUDED_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(synthetic_features[col])
]

live_feature_frames = []
current_final_frames = []
for spec in LIVE_COHORTS:
    candidate_raw = pd.read_csv(spec.candidate_path)
    live_schema = make_live_training_schema(candidate_raw, spec)
    live_features = add_resolver_features(live_schema)
    for col in feature_columns:
        if col not in live_features.columns:
            live_features[col] = 0.0
    live_feature_frames.append(live_features)

    final = pd.read_csv(spec.final_path).copy()
    final["cohort"] = spec.cohort
    final["cohort_role"] = spec.role
    final["current_final_prediction"] = final[spec.final_prediction_col].astype(str)
    if "correct" in final.columns:
        final["current_final_correct"] = bool_series(final["correct"])
    else:
        final["current_final_correct"] = final["current_final_prediction"].astype(str).eq(final["true_pathology"].astype(str))
    if "ranked_differential" not in final.columns:
        final["ranked_differential"] = "[]"
    current_final_frames.append(final)

live_features_all = pd.concat(live_feature_frames, ignore_index=True, sort=False)
current_final_all = pd.concat(current_final_frames, ignore_index=True, sort=False)
live_ranked_by_case = {}
for (cohort, case_id), group in live_features_all.sort_values(["cohort", "case_id", "candidate_order"]).groupby(["cohort", "case_id"], sort=False):
    ranked = list(dict.fromkeys(group["candidate_pathology"].astype(str).tolist()))
    live_ranked_by_case[(cohort, str(case_id))] = json.dumps(ranked)


def fill_missing_ranked_differentials(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ranked_differential" not in out.columns:
        out["ranked_differential"] = "[]"
    is_missing = out["ranked_differential"].apply(lambda value: len(safe_parse_list(value)) == 0)
    out.loc[is_missing, "ranked_differential"] = out.loc[is_missing].apply(
        lambda row: live_ranked_by_case.get((row["cohort"], str(row["case_id"])), "[]"),
        axis=1,
    )
    return out


current_final_all = fill_missing_ranked_differentials(current_final_all)
log_step(f"loaded {len(synthetic_features):,} synthetic candidate rows and {len(live_features_all):,} live candidate rows")

synthetic_group_summary = (
    synthetic_features.groupby("split")
    .agg(
        candidate_rows=("candidate_pathology", "count"),
        states=("synthetic_state_id", "nunique"),
        positive_rate=("candidate_label", "mean"),
        candidate_pool_recall=("candidate_label", lambda s: np.nan),
    )
    .reset_index()
)
synthetic_recall = synthetic_features.groupby(["split", "synthetic_state_id"])["candidate_label"].max().groupby("split").mean()
synthetic_group_summary["candidate_pool_recall"] = synthetic_group_summary["split"].map(synthetic_recall).astype(float)

live_pool_summary = (
    live_features_all.groupby("cohort")
    .agg(
        candidate_rows=("candidate_pathology", "count"),
        cases=("synthetic_state_id", "nunique"),
        unique_candidates_mean=("candidate_pathology", lambda s: np.nan),
        candidate_pool_recall=("candidate_label", lambda s: np.nan),
    )
    .reset_index()
)
unique_mean = live_features_all.groupby(["cohort", "synthetic_state_id"])["candidate_pathology"].nunique().groupby("cohort").mean()
live_recall = live_features_all.groupby(["cohort", "synthetic_state_id"])["candidate_label"].max().groupby("cohort").mean()
live_pool_summary["unique_candidates_mean"] = live_pool_summary["cohort"].map(unique_mean).astype(float)
live_pool_summary["candidate_pool_recall"] = live_pool_summary["cohort"].map(live_recall).astype(float)

display(synthetic_group_summary)
display(live_pool_summary)

# %% [markdown]
# ## 3. Train Synthetic Listwise And Pairwise Resolvers

# %%
train_mask = synthetic_features["split"].eq("train")
validate_mask = synthetic_features["split"].eq("validate")
X_train = clean_numeric_frame(synthetic_features.loc[train_mask], feature_columns)
y_train = synthetic_features.loc[train_mask, "candidate_label"].astype(int)
X_validate = clean_numeric_frame(synthetic_features.loc[validate_mask], feature_columns)
y_validate = synthetic_features.loc[validate_mask, "candidate_label"].astype(int)

row_models: dict[str, Any] = {
    "synthetic_logistic_group_softmax": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, class_weight="balanced", C=0.5, random_state=RANDOM_SEED),
    ),
    "synthetic_hist_gradient_boosting_group_softmax": HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    ),
}

synthetic_scored = synthetic_features.copy()
for model_name, model in row_models.items():
    log_step(f"training {model_name}")
    model.fit(X_train, y_train)
    synthetic_scored.loc[validate_mask, model_name] = model.predict_proba(X_validate)[:, 1]


class ListwiseMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_group_codes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(frame["synthetic_state_id"].astype(str), sort=True)
    return codes.astype(np.int64), uniques.astype(str)


def train_listwise_mlp(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, columns: list[str], epochs: int = 35) -> tuple[ListwiseMLP, StandardScaler, dict[str, float]]:
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(clean_numeric_frame(train_frame, columns)).astype(np.float32)
    y_train_np = train_frame["candidate_label"].astype(float).to_numpy(dtype=np.float32)
    group_codes, _ = make_group_codes(train_frame)
    group_has_true = pd.Series(y_train_np).groupby(group_codes).sum().to_numpy() > 0

    x = torch.tensor(x_train_np, dtype=torch.float32)
    y = torch.tensor(y_train_np, dtype=torch.float32)
    group = torch.tensor(group_codes, dtype=torch.long)
    n_groups = int(group.max().item()) + 1
    has_true = torch.tensor(group_has_true, dtype=torch.bool)

    model = ListwiseMLP(len(columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=2e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        scores = model(x).clamp(min=-20.0, max=20.0)
        exp_scores = torch.exp(scores)
        group_sum = torch.zeros(n_groups, dtype=torch.float32).scatter_add_(0, group, exp_scores)
        true_score = torch.zeros(n_groups, dtype=torch.float32).scatter_add_(0, group, scores * y)
        loss = (-true_score[has_true] + torch.log(group_sum[has_true] + 1e-8)).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_scores = model(torch.tensor(scaler.transform(clean_numeric_frame(valid_frame, columns)).astype(np.float32))).numpy()
    valid_tmp = valid_frame.copy()
    valid_tmp["synthetic_listwise_mlp"] = valid_scores
    selected = select_unique_by_score(valid_tmp, "synthetic_listwise_mlp", "prediction")
    metrics = summarize_case_predictions(selected.assign(cohort="synthetic_validate"), "prediction", "synthetic_listwise_mlp", "synthetic_validate")
    metrics["final_training_loss"] = float(loss.item())
    return model, scaler, metrics


log_step("training synthetic listwise MLP")
listwise_model, listwise_scaler, listwise_validate_metrics = train_listwise_mlp(
    synthetic_features.loc[train_mask].copy(),
    synthetic_features.loc[validate_mask].copy(),
    feature_columns,
)
log_step("trained synthetic listwise MLP")
with torch.no_grad():
    synthetic_scored.loc[validate_mask, "synthetic_listwise_mlp"] = listwise_model(
        torch.tensor(listwise_scaler.transform(clean_numeric_frame(synthetic_features.loc[validate_mask], feature_columns)).astype(np.float32))
    ).numpy()


def make_pairwise_training(frame: pd.DataFrame, columns: list[str], max_pairs_per_group: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(RANDOM_SEED)
    diffs = []
    labels = []
    for _, group in frame.groupby("synthetic_state_id", sort=False):
        positives = group[group["candidate_label"].astype(bool)]
        negatives = group[~group["candidate_label"].astype(bool)]
        if positives.empty or negatives.empty:
            continue
        pos = positives.iloc[0]
        neg_indices = negatives.index.to_numpy()
        if len(neg_indices) > max_pairs_per_group:
            neg_indices = rng.choice(neg_indices, size=max_pairs_per_group, replace=False)
        for neg_idx in neg_indices:
            neg = group.loc[neg_idx]
            pos_values = pd.to_numeric(pos[columns], errors="coerce").fillna(0.0).astype(float)
            neg_values = pd.to_numeric(neg[columns], errors="coerce").fillna(0.0).astype(float)
            diffs.append((pos_values - neg_values).to_dict())
            labels.append(1)
            diffs.append((neg_values - pos_values).to_dict())
            labels.append(0)
    return pd.DataFrame(diffs).replace([np.inf, -np.inf], 0.0).fillna(0.0), pd.Series(labels, dtype=int)


pair_X, pair_y = make_pairwise_training(synthetic_features.loc[train_mask], feature_columns)
pairwise_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=300, C=0.5, class_weight="balanced", random_state=RANDOM_SEED),
)
log_step("training synthetic pairwise Bradley-Terry resolver")
pairwise_model.fit(pair_X[feature_columns], pair_y)


def pairwise_scores(frame: pd.DataFrame, columns: list[str], model: Any) -> pd.Series:
    scores = pd.Series(0.0, index=frame.index)
    for _, group in frame.groupby("synthetic_state_id", sort=False):
        indices = list(group.index)
        if len(indices) <= 1:
            continue
        matrix = clean_numeric_frame(group, columns)
        agg = {idx: 0.0 for idx in indices}
        counts = {idx: 0 for idx in indices}
        diffs = []
        pair_indices = []
        for i, idx_i in enumerate(indices):
            for idx_j in indices[i + 1:]:
                diffs.append((matrix.loc[idx_i] - matrix.loc[idx_j]).to_numpy(dtype=float))
                pair_indices.append((idx_i, idx_j))
        if not diffs:
            continue
        diff_frame = pd.DataFrame(diffs, columns=columns)
        probs = model.predict_proba(diff_frame[columns])[:, 1]
        logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1 - 1e-6))
        for (idx_i, idx_j), logit in zip(pair_indices, logits):
                agg[idx_i] += logit
                agg[idx_j] -= logit
                counts[idx_i] += 1
                counts[idx_j] += 1
        for idx in indices:
            scores.loc[idx] = agg[idx] / max(counts[idx], 1)
    return scores


synthetic_scored.loc[validate_mask, "synthetic_pairwise_bradley_terry"] = pairwise_scores(
    synthetic_features.loc[validate_mask].copy(),
    feature_columns,
    pairwise_model,
)

synthetic_model_rows = []
for model_name in [
    "synthetic_logistic_group_softmax",
    "synthetic_hist_gradient_boosting_group_softmax",
    "synthetic_listwise_mlp",
    "synthetic_pairwise_bradley_terry",
]:
    selected = select_unique_by_score(synthetic_scored.loc[validate_mask], model_name, "prediction")
    row = summarize_case_predictions(selected.assign(cohort="synthetic_validate"), "prediction", model_name, "synthetic_validate")
    row["row_average_precision"] = float(average_precision_score(y_validate, synthetic_scored.loc[validate_mask, model_name]))
    try:
        row["row_auc"] = float(roc_auc_score(y_validate, synthetic_scored.loc[validate_mask, model_name]))
    except Exception:
        row["row_auc"] = np.nan
    synthetic_model_rows.append(row)

synthetic_model_summary = pd.DataFrame(synthetic_model_rows)
display(synthetic_model_summary)

# %% [markdown]
# ## 4. Synthetic-Only Transfer To Live Candidate Pools

# %%
live_scored = live_features_all.copy()
for col in feature_columns:
    if col not in live_scored.columns:
        live_scored[col] = 0.0

log_step("scoring live candidate pools with synthetic-only resolvers")
X_live = clean_numeric_frame(live_scored, feature_columns)
for model_name, model in row_models.items():
    live_scored[model_name] = model.predict_proba(X_live)[:, 1]
with torch.no_grad():
    live_scored["synthetic_listwise_mlp"] = listwise_model(
        torch.tensor(listwise_scaler.transform(X_live).astype(np.float32))
    ).numpy()
live_scored["synthetic_pairwise_bradley_terry"] = pairwise_scores(live_scored, feature_columns, pairwise_model)

synthetic_transfer_case_frames = []
synthetic_transfer_rows = []
for model_name in [
    "synthetic_logistic_group_softmax",
    "synthetic_hist_gradient_boosting_group_softmax",
    "synthetic_listwise_mlp",
    "synthetic_pairwise_bradley_terry",
]:
    selected = select_unique_by_score(live_scored, model_name, f"{model_name}_prediction")
    selected["policy_name"] = model_name
    synthetic_transfer_case_frames.append(selected)
    for cohort, group in selected.groupby("cohort", sort=False):
        synthetic_transfer_rows.append(summarize_case_predictions(group, f"{model_name}_prediction", model_name, "synthetic_only_transfer"))
    pooled_row = summarize_case_predictions(selected.assign(cohort="pooled_343"), f"{model_name}_prediction", model_name, "synthetic_only_transfer")
    synthetic_transfer_rows.append(pooled_row)

synthetic_transfer_summary = pd.DataFrame(synthetic_transfer_rows)
synthetic_transfer_cases = pd.concat(synthetic_transfer_case_frames, ignore_index=True, sort=False)
display(synthetic_transfer_summary)

# %% [markdown]
# ## 5. Artifact-Calibrated Leave-One-Cohort-Out Resolver

# %%
def artifact_loco_train_predict(model_kind: str = "logistic", artifact_repeat: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    case_frames = []
    synthetic_all = synthetic_features[synthetic_features["split"].isin(["train", "validate"])].copy()
    for holdout in [spec.cohort for spec in LIVE_COHORTS]:
        log_step(f"training artifact-calibrated {model_kind} resolver with {holdout} held out")
        train_art = live_features_all[live_features_all["cohort"] != holdout].copy()
        repeated_art = pd.concat([train_art] * artifact_repeat, ignore_index=True, sort=False) if len(train_art) else train_art
        train_frame = pd.concat([synthetic_all, repeated_art], ignore_index=True, sort=False)
        test_frame = live_features_all[live_features_all["cohort"] == holdout].copy()
        for col in feature_columns:
            if col not in train_frame.columns:
                train_frame[col] = 0.0
            if col not in test_frame.columns:
                test_frame[col] = 0.0
        if model_kind == "gbm":
            model = HistGradientBoostingClassifier(
                max_iter=70,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=0.05,
                random_state=RANDOM_SEED,
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, C=0.5, class_weight="balanced", random_state=RANDOM_SEED),
            )
        model.fit(clean_numeric_frame(train_frame, feature_columns), train_frame["candidate_label"].astype(int))
        score_col = f"artifact_loco_{model_kind}_score"
        test_frame[score_col] = model.predict_proba(clean_numeric_frame(test_frame, feature_columns))[:, 1]
        selected = select_unique_by_score(test_frame, score_col, f"artifact_loco_{model_kind}_prediction")
        selected["policy_name"] = f"artifact_loco_{model_kind}"
        selected["heldout_cohort"] = holdout
        case_frames.append(selected)
        rows.append(summarize_case_predictions(selected, f"artifact_loco_{model_kind}_prediction", f"artifact_loco_{model_kind}", "synthetic_plus_loco_artifact_calibration"))
    all_cases = pd.concat(case_frames, ignore_index=True, sort=False)
    rows.append(summarize_case_predictions(all_cases.assign(cohort="pooled_343"), f"artifact_loco_{model_kind}_prediction", f"artifact_loco_{model_kind}", "synthetic_plus_loco_artifact_calibration"))
    return pd.DataFrame(rows), all_cases


loco_logistic_summary, loco_logistic_cases = artifact_loco_train_predict("logistic", artifact_repeat=8)
loco_gbm_summary, loco_gbm_cases = artifact_loco_train_predict("gbm", artifact_repeat=8)
loco_summary = pd.concat([loco_logistic_summary, loco_gbm_summary], ignore_index=True)
loco_cases = pd.concat([loco_logistic_cases, loco_gbm_cases], ignore_index=True, sort=False)
display(loco_summary)

# %% [markdown]
# ## 6. Diagnostic Artifact Fit And Reference Frontier

# %%
def train_all_artifact_diagnostic(model_kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_step(f"training diagnostic artifact-fit {model_kind} resolver")
    train_frame = pd.concat([synthetic_features, live_features_all], ignore_index=True, sort=False)
    for col in feature_columns:
        if col not in train_frame.columns:
            train_frame[col] = 0.0
    if model_kind == "gbm":
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_SEED,
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, C=0.5, class_weight="balanced", random_state=RANDOM_SEED),
        )
    model.fit(clean_numeric_frame(train_frame, feature_columns), train_frame["candidate_label"].astype(int))
    scored = live_features_all.copy()
    for col in feature_columns:
        if col not in scored.columns:
            scored[col] = 0.0
    score_col = f"artifact_fit_{model_kind}_score"
    scored[score_col] = model.predict_proba(clean_numeric_frame(scored, feature_columns))[:, 1]
    selected = select_unique_by_score(scored, score_col, f"artifact_fit_{model_kind}_prediction")
    selected["policy_name"] = f"artifact_fit_{model_kind}"
    rows = []
    for cohort, group in selected.groupby("cohort", sort=False):
        rows.append(summarize_case_predictions(group, f"artifact_fit_{model_kind}_prediction", f"artifact_fit_{model_kind}", "diagnostic_artifact_label_fit"))
    rows.append(summarize_case_predictions(selected.assign(cohort="pooled_343"), f"artifact_fit_{model_kind}_prediction", f"artifact_fit_{model_kind}", "diagnostic_artifact_label_fit"))
    return pd.DataFrame(rows), selected


artifact_fit_logistic_summary, artifact_fit_logistic_cases = train_all_artifact_diagnostic("logistic")
artifact_fit_gbm_summary, artifact_fit_gbm_cases = train_all_artifact_diagnostic("gbm")
artifact_fit_summary = pd.concat([artifact_fit_logistic_summary, artifact_fit_gbm_summary], ignore_index=True)
artifact_fit_cases = pd.concat([artifact_fit_logistic_cases, artifact_fit_gbm_cases], ignore_index=True, sort=False)

reference_rows = []
for cohort, group in current_final_all.groupby("cohort", sort=False):
    temp = group.rename(columns={"current_final_prediction": "_prediction"}).copy()
    if "candidate_pool_has_true" not in temp.columns:
        recall = live_features_all[live_features_all["cohort"] == cohort].groupby("synthetic_state_id")["candidate_label"].max()
        temp["candidate_pool_has_true"] = temp["case_id"].map({uid.split("::", 1)[1]: val for uid, val in recall.items()}).fillna(False).astype(bool)
    reference_rows.append(summarize_case_predictions(temp, "_prediction", "current_final_pipeline", "saved_artifact_reference"))

current_temp = current_final_all.rename(columns={"current_final_prediction": "_prediction"}).copy()
candidate_recall_by_case = live_features_all.groupby(["cohort", "case_id"])["candidate_label"].max().to_dict()
current_temp["candidate_pool_has_true"] = current_temp.apply(lambda r: bool(candidate_recall_by_case.get((r["cohort"], r["case_id"]), False)), axis=1)
reference_rows.append(summarize_case_predictions(current_temp.assign(cohort="pooled_343"), "_prediction", "current_final_pipeline", "saved_artifact_reference"))

oracle_selected = []
for state_id, group in live_features_all.groupby("synthetic_state_id", sort=False):
    top = group.iloc[0]
    true_in_pool = bool(group["candidate_label"].astype(bool).any())
    prediction = str(top["true_pathology"]) if true_in_pool else str(group.sort_values("candidate_order").iloc[0]["candidate_pathology"])
    unique = list(dict.fromkeys(group["candidate_pathology"].astype(str).tolist()))
    oracle_selected.append(
        {
            "cohort": top["cohort"],
            "case_id": top["case_id"],
            "synthetic_state_id": state_id,
            "true_pathology": top["true_pathology"],
            "candidate_pool_oracle_prediction": prediction,
            "candidate_pool_has_true": true_in_pool,
            "ranked_differential": json.dumps([prediction] + [x for x in unique if x != prediction]),
        }
    )
oracle_cases = pd.DataFrame(oracle_selected)
for cohort, group in oracle_cases.groupby("cohort", sort=False):
    reference_rows.append(summarize_case_predictions(group, "candidate_pool_oracle_prediction", "candidate_pool_oracle_non_deployable", "oracle_non_deployable"))
reference_rows.append(summarize_case_predictions(oracle_cases.assign(cohort="pooled_343"), "candidate_pool_oracle_prediction", "candidate_pool_oracle_non_deployable", "oracle_non_deployable"))

reference_summary = pd.DataFrame(reference_rows)

nb39_summary = pd.DataFrame()
if NB39_SUMMARY_PATH.exists():
    nb39_summary = pd.read_csv(NB39_SUMMARY_PATH)
    nb39_summary = nb39_summary[nb39_summary["cohort"].eq("pooled_343")].copy()

all_policy_summary = pd.concat(
    [
        reference_summary,
        synthetic_transfer_summary,
        loco_summary,
        artifact_fit_summary,
    ],
    ignore_index=True,
    sort=False,
)
display(all_policy_summary[all_policy_summary["cohort"].eq("pooled_343")][["policy_name", "claim_type", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct"]])

# %% [markdown]
# ## 7. Error Analysis

# %%
def case_result_table_for_policy(cases: pd.DataFrame, prediction_col: str, policy_name: str) -> pd.DataFrame:
    out = cases.copy()
    out["policy_name"] = policy_name
    out["prediction"] = out[prediction_col]
    out["correct"] = out["prediction"].astype(str).eq(out["true_pathology"].astype(str))
    out["failure_pair"] = np.where(out["correct"], "", out["true_pathology"].astype(str) + " -> " + out["prediction"].astype(str))
    return out[["policy_name", "cohort", "case_id", "synthetic_state_id", "true_pathology", "prediction", "correct", "candidate_pool_has_true", "candidate_pool_unique", "ranked_differential", "failure_pair"]]


selected_candidate_policy_name = "artifact_loco_logistic"
selected_cases = loco_logistic_cases.copy()
selected_case_results = case_result_table_for_policy(selected_cases, "artifact_loco_logistic_prediction", selected_candidate_policy_name)

current_case_lookup = current_temp[["cohort", "case_id", "true_pathology", "_prediction"]].rename(columns={"_prediction": "current_final_prediction"})
selected_case_results = selected_case_results.merge(current_case_lookup, on=["cohort", "case_id", "true_pathology"], how="left")
selected_case_results["current_final_correct"] = selected_case_results["current_final_prediction"].astype(str).eq(selected_case_results["true_pathology"].astype(str))
selected_case_results["paired_vs_current"] = np.select(
    [
        (~selected_case_results["current_final_correct"]) & selected_case_results["correct"],
        selected_case_results["current_final_correct"] & (~selected_case_results["correct"]),
    ],
    ["win", "regression"],
    default="unchanged",
)

failure_modes = (
    selected_case_results[~selected_case_results["correct"]]
    .groupby(["cohort", "failure_pair"])
    .agg(cases=("case_id", "nunique"), candidate_pool_has_true=("candidate_pool_has_true", "sum"))
    .reset_index()
    .sort_values(["cases", "cohort"], ascending=[False, True])
)

paired_counts = selected_case_results["paired_vs_current"].value_counts().reindex(["win", "regression", "unchanged"]).fillna(0).astype(int)
display(selected_case_results.groupby(["cohort", "paired_vs_current"]).size().unstack(fill_value=0))
display(failure_modes.head(30))

# %% [markdown]
# ## 8. Figures

# %%
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


plot_policies = [
    "current_final_pipeline",
    "synthetic_logistic_group_softmax",
    "synthetic_listwise_mlp",
    "synthetic_pairwise_bradley_terry",
    "artifact_loco_logistic",
    "artifact_fit_gbm",
    "candidate_pool_oracle_non_deployable",
]
plot_df = all_policy_summary[(all_policy_summary["cohort"].eq("pooled_343")) & (all_policy_summary["policy_name"].isin(plot_policies))].copy()
plot_df = plot_df.sort_values("accuracy")
plt.figure(figsize=(9, 5))
plt.barh(plot_df["policy_name"], plot_df["accuracy"])
for _, row in plot_df.iterrows():
    plt.text(row["accuracy"] + 0.002, row["policy_name"], f"{int(row['correct'])}/{int(row['n_cases'])}", va="center", fontsize=8)
plt.xlim(0.84, 1.0)
plt.xlabel("Pooled artifact accuracy")
plt.title("Synthetic-to-Live Resolver Frontier")
savefig(FIGURE_DIR / "pooled_resolver_frontier.png")

cohort_plot = all_policy_summary[all_policy_summary["policy_name"].isin(["current_final_pipeline", "synthetic_listwise_mlp", "artifact_loco_logistic", "candidate_pool_oracle_non_deployable"])]
cohort_plot = cohort_plot[~cohort_plot["cohort"].eq("pooled_343")].copy()
plt.figure(figsize=(10, 5))
cohorts = [spec.cohort for spec in LIVE_COHORTS]
for idx, policy in enumerate(["current_final_pipeline", "synthetic_listwise_mlp", "artifact_loco_logistic", "candidate_pool_oracle_non_deployable"]):
    sub = cohort_plot[cohort_plot["policy_name"].eq(policy)].set_index("cohort").reindex(cohorts)
    plt.bar(np.arange(len(cohorts)) + idx * 0.18, sub["accuracy"], width=0.18, label=policy)
plt.xticks(np.arange(len(cohorts)) + 0.27, cohorts)
plt.ylim(0.82, 1.01)
plt.ylabel("Accuracy")
plt.title("Resolver Transfer By Cohort")
plt.legend(fontsize=8, loc="lower right")
savefig(FIGURE_DIR / "resolver_transfer_by_cohort.png")

plt.figure(figsize=(5.5, 4))
plt.bar(paired_counts.index, paired_counts.values, color=["#2d7f5e", "#b33a3a", "#7c8490"])
plt.ylabel("Cases")
plt.title("Selected LOCO Resolver Paired Effect")
savefig(FIGURE_DIR / "loco_paired_effect.png")

top_failures = selected_case_results[~selected_case_results["correct"]]["failure_pair"].value_counts().head(12).sort_values()
plt.figure(figsize=(9, 5))
plt.barh(top_failures.index, top_failures.values)
plt.xlabel("Misses")
plt.title("Selected LOCO Resolver Failure Modes")
savefig(FIGURE_DIR / "loco_failure_modes.png")

# %% [markdown]
# ## 9. Final Summary And Artifact Contract

# %%
synthetic_group_summary.to_csv(ARTIFACT_ROOT / "synthetic_state_generation_summary.csv", index=False)
synthetic_model_summary.to_csv(ARTIFACT_ROOT / "synthetic_model_validation_summary.csv", index=False)
live_pool_summary.to_csv(ARTIFACT_ROOT / "live_candidate_pool_summary.csv", index=False)
synthetic_transfer_summary.to_csv(ARTIFACT_ROOT / "synthetic_only_live_transfer_summary.csv", index=False)
synthetic_transfer_cases.to_csv(ARTIFACT_ROOT / "synthetic_only_live_transfer_case_results.csv", index=False)
loco_summary.to_csv(ARTIFACT_ROOT / "leave_one_cohort_out_summary.csv", index=False)
loco_cases.to_csv(ARTIFACT_ROOT / "leave_one_cohort_out_case_predictions.csv", index=False)
artifact_fit_summary.to_csv(ARTIFACT_ROOT / "diagnostic_artifact_fit_summary.csv", index=False)
artifact_fit_cases.to_csv(ARTIFACT_ROOT / "diagnostic_artifact_fit_case_predictions.csv", index=False)
all_policy_summary.to_csv(ARTIFACT_ROOT / "live_transfer_policy_summary.csv", index=False)
selected_case_results.to_csv(ARTIFACT_ROOT / "selected_loco_resolver_case_results.csv", index=False)
live_scored.to_csv(ARTIFACT_ROOT / "live_transfer_candidate_scores.csv", index=False)
failure_modes.to_csv(ARTIFACT_ROOT / "selected_loco_failure_modes.csv", index=False)

hard_case_payload = selected_case_results[~selected_case_results["correct"]].to_dict(orient="records")
write_json(ARTIFACT_ROOT / "hard_case_listwise_resolver_audits.json", hard_case_payload)

pooled_current = all_policy_summary[(all_policy_summary["cohort"].eq("pooled_343")) & (all_policy_summary["policy_name"].eq("current_final_pipeline"))].iloc[0].to_dict()
pooled_loco = all_policy_summary[(all_policy_summary["cohort"].eq("pooled_343")) & (all_policy_summary["policy_name"].eq(selected_candidate_policy_name))].iloc[0].to_dict()
pooled_oracle = all_policy_summary[(all_policy_summary["cohort"].eq("pooled_343")) & (all_policy_summary["policy_name"].eq("candidate_pool_oracle_non_deployable"))].iloc[0].to_dict()

promotion_decision = (
    "not_promoted"
    if int(pooled_loco["correct"]) <= int(pooled_current["correct"])
    else "candidate_needs_fresh_confirmation"
)

selected_policy = {
    "selected_policy_name": selected_candidate_policy_name,
    "status": promotion_decision,
    "training_data": {
        "synthetic_source": str(SYNTHETIC_FEATURE_PATH.relative_to(PROJECT_ROOT)),
        "synthetic_train_states": int(synthetic_features[synthetic_features["split"].eq("train")]["synthetic_state_id"].nunique()),
        "synthetic_validate_states": int(synthetic_features[synthetic_features["split"].eq("validate")]["synthetic_state_id"].nunique()),
        "artifact_calibration": "leave-one-cohort-out only for selected policy summary",
    },
    "feature_count": int(len(feature_columns)),
    "pooled_current_final": {
        "correct": int(pooled_current["correct"]),
        "n_cases": int(pooled_current["n_cases"]),
        "accuracy": float(pooled_current["accuracy"]),
    },
    "pooled_selected_loco": {
        "correct": int(pooled_loco["correct"]),
        "n_cases": int(pooled_loco["n_cases"]),
        "accuracy": float(pooled_loco["accuracy"]),
        "paired_wins_vs_current": int(paired_counts.get("win", 0)),
        "paired_regressions_vs_current": int(paired_counts.get("regression", 0)),
    },
    "pooled_candidate_pool_oracle": {
        "correct": int(pooled_oracle["correct"]),
        "n_cases": int(pooled_oracle["n_cases"]),
        "accuracy": float(pooled_oracle["accuracy"]),
    },
    "interpretation": "Synthetic DDXPlus resolver training does not by itself solve live artifact candidate selection. The selected leave-one-cohort-out calibrated resolver is reported as a transfer diagnostic, not a replacement for the current final pipeline unless it beats the current saved final result.",
}
write_json(ARTIFACT_ROOT / "selected_listwise_resolver_policy.json", selected_policy)

pooled_table = all_policy_summary[all_policy_summary["cohort"].eq("pooled_343")][
    ["policy_name", "claim_type", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct"]
].copy()

report_lines = [
    "# Synthetic-to-Live Listwise Resolver",
    "",
    "Notebook 40 tests whether a resolver trained on DDXPlus-derived synthetic partial evidence states transfers to the saved live candidate pools from Notebooks 33, 37, and 38. It makes no API calls.",
    "",
    "## Main Pooled Results",
    "",
    pooled_table.to_markdown(index=False),
    "",
    "## Synthetic Validation",
    "",
    synthetic_model_summary[["policy_name", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct", "row_average_precision", "row_auc"]].to_markdown(index=False),
    "",
    "## Leave-One-Cohort-Out Artifact Calibration",
    "",
    loco_summary[["cohort", "policy_name", "correct", "n_cases", "accuracy", "candidate_pool_recall_correct"]].to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    f"The current saved final pipeline is `{int(pooled_current['correct'])}/{int(pooled_current['n_cases'])}`. The selected leave-one-cohort-out resolver `{selected_candidate_policy_name}` is `{int(pooled_loco['correct'])}/{int(pooled_loco['n_cases'])}` with `{int(paired_counts.get('win', 0))}` wins and `{int(paired_counts.get('regression', 0))}` regressions versus the current final pipeline.",
    "",
    "This means synthetic DDXPlus states are useful for building and testing resolver families, but the current synthetic-to-live transfer does not by itself produce the desired near-oracle resolver. The candidate-pool oracle remains the upper bound, not a final result.",
    "",
    "## Remaining Failure Modes",
    "",
    failure_modes.head(25).to_markdown(index=False),
    "",
    "## Artifact Contract",
    "",
    "- `resolved_run_config.json`",
    "- `synthetic_state_generation_summary.csv`",
    "- `synthetic_model_validation_summary.csv`",
    "- `live_candidate_pool_summary.csv`",
    "- `synthetic_only_live_transfer_summary.csv`",
    "- `synthetic_only_live_transfer_case_results.csv`",
    "- `leave_one_cohort_out_summary.csv`",
    "- `leave_one_cohort_out_case_predictions.csv`",
    "- `diagnostic_artifact_fit_summary.csv`",
    "- `diagnostic_artifact_fit_case_predictions.csv`",
    "- `live_transfer_policy_summary.csv`",
    "- `live_transfer_candidate_scores.csv`",
    "- `selected_loco_resolver_case_results.csv`",
    "- `selected_loco_failure_modes.csv`",
    "- `hard_case_listwise_resolver_audits.json`",
    "- `selected_listwise_resolver_policy.json`",
    "- figures under `figures/`",
    "",
]
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print("Artifacts written to:", ARTIFACT_ROOT)
print("Report written to   :", REPORT_PATH)
display(pooled_table)
