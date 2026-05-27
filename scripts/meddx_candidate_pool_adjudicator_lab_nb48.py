from __future__ import annotations

# %% [markdown]
# # Notebook 48: MEDDx Candidate-Pool Adjudicator Lab
#
# Notebook 46 showed that the MEDDx-aligned native driver can usually generate the right answer somewhere in
# the candidate pool, but the final resolver often chooses the wrong close neighbor. Notebook 47 verified that
# the broad pool contains the truth in 88/90 workups. This notebook asks the next control question:
#
# Can we build a general candidate-pool adjudicator that improves the final top-1 decision without using new API
# calls or case-by-case hardcoding?
#
# The notebook keeps all live evidence traces frozen. It builds candidate-level features from independent signals
# already present in the run: final/LLM ranks, branch ranks, casebase priors, RareBench graph scores, DDXPlus MLP
# scores, and a train-derived DDXPlus graph posterior replay over the revealed ledger.

# %%
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:
    raise RuntimeError("Notebook 48 requires scikit-learn for the diagnostic educator section.") from exc

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
REPAIR_RUN_NAME = "meddx_candidate_pool_repair_lab_v1"
INPUT_ROOT = ROOT / "artifacts" / "universal_meddx" / INPUT_RUN_NAME
REPAIR_ROOT = ROOT / "artifacts" / "universal_meddx" / REPAIR_RUN_NAME
RUN_NAME = "meddx_candidate_pool_adjudicator_lab_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DDXPLUS_MLP_GUARD_CONFIDENCE_MIN = 0.70
DDXPLUS_MLP_GUARD_MARGIN_MIN = 0.20
DDXPLUS_GRAPH_OVERRIDE_MARGIN_MIN = 1.00
DDXPLUS_GRAPH_CURRENT_SCORE_MAX = 0.00
SOURCE_WEIGHTED_OVERRIDE_MARGIN_MIN = 0.50

print("Project root:", ROOT)
print("Input root  :", INPUT_ROOT)
print("Repair root :", REPAIR_ROOT)
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


def label_rank(label: Any, ranked: list[str], missing_rank: int = 99) -> int:
    key = normalize_label(label)
    for idx, item in enumerate(ranked, start=1):
        if normalize_label(item) == key:
            return idx
    return missing_rank


def reciprocal_rank(label: Any, ranked: list[str]) -> float:
    rank = label_rank(label, ranked)
    return 0.0 if rank == 99 else 1.0 / rank


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


def insert_top(ranked: list[str], label: str, limit: int = 10) -> list[str]:
    label_key = normalize_label(label)
    out = [label]
    for candidate in ranked:
        if normalize_label(candidate) != label_key:
            out.append(candidate)
    return out[:limit]


def score_ranked(row: pd.Series, prediction: str, ranked: list[str]) -> dict[str, Any]:
    truth = normalize_label(row["ground_truth_diagnosis"])
    keys = [normalize_label(label) for label in ranked]
    return {
        "correct_top1": normalize_label(prediction) == truth,
        "gtpa_at_3": truth in set(keys[:3]),
        "gtpa_at_5": truth in set(keys[:5]),
        "true_rank": keys.index(truth) + 1 if truth in keys else 11,
    }


def ranked_from_prediction_row(row: pd.Series) -> list[str]:
    ranked = parse_json_list(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "").strip()
    if prediction and normalize_label(prediction) not in {normalize_label(label) for label in ranked}:
        ranked = [prediction] + ranked
    return ranked[:10]


def outcome_from_ddxplus_answer(answer: Any) -> str | None:
    text = str(answer or "").lower()
    if "no / not reported" in text or "does not mention" in text or "answer: no" in text or "answer: n." in text:
        return "absent"
    if "answer: present" in text or "answer: yes" in text or "answer: y." in text:
        return "present"
    return None


def extract_root_from_question(question: Any) -> str | None:
    match = re.search(r"\[(E_\d+)\]", str(question or ""))
    return match.group(1) if match else None


def parse_source_values(value: Any) -> dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    out: dict[str, float] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("source"):
                out[str(item["source"])] = safe_float(item.get("value", 0.0))
    return out


def normalize_by_workup(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        max_value = out.groupby("workup_id")[column].transform("max")
        out[f"{column}_norm"] = np.where(max_value.abs() > 1e-12, out[column] / max_value.replace(0, np.nan), 0.0)
        out[f"{column}_minus_max"] = out[column] - max_value
        out[f"{column}_rank_within_pool"] = out.groupby("workup_id")[column].rank(ascending=False, method="min")
    return out.fillna(0.0)

# %% [markdown]
# ## 2. Load Notebook 46 And Notebook 47 Artifacts

# %%
required_input = [
    "resolved_run_config.json",
    "predictions.csv",
    "candidate_level_resolver_scores.csv",
    "question_answer_ledger.csv",
    "interaction_traces.jsonl",
    "universal_cases.csv",
]
required_repair = ["candidate_pool_long.csv", "candidate_pool_coverage.csv", "candidate_pool_summary.csv"]
missing = [name for name in required_input if not (INPUT_ROOT / name).exists()]
missing += [name for name in required_repair if not (REPAIR_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing required artifacts: {missing}")

input_config = read_json(INPUT_ROOT / "resolved_run_config.json")
predictions = pd.read_csv(INPUT_ROOT / "predictions.csv")
candidate_scores = pd.read_csv(INPUT_ROOT / "candidate_level_resolver_scores.csv")
ledger = pd.read_csv(INPUT_ROOT / "question_answer_ledger.csv")
universal_cases = pd.read_csv(INPUT_ROOT / "universal_cases.csv")
branches = pd.read_csv(INPUT_ROOT / "branch_case_results.csv") if (INPUT_ROOT / "branch_case_results.csv").exists() else pd.DataFrame()
candidate_pool_long = pd.read_csv(REPAIR_ROOT / "candidate_pool_long.csv")
candidate_pool_coverage = pd.read_csv(REPAIR_ROOT / "candidate_pool_coverage.csv")

graph_edges_path = ROOT / "artifacts" / "graph_algorithmic_ledger" / "medkgi_style_offline_notebook13_49case_v1" / "global_evidence_graph_edges.csv"
graph_edges = pd.read_csv(graph_edges_path) if graph_edges_path.exists() else pd.DataFrame()

traces = []
with (INPUT_ROOT / "interaction_traces.jsonl").open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            traces.append(json.loads(line))

prediction_by_key = {
    (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"])): row
    for _, row in predictions.iterrows()
}
case_by_key = {
    (str(row["dataset_name"]), str(row["case_id"])): row
    for _, row in universal_cases.iterrows()
}
score_by_key = {
    (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]), normalize_label(row["label"])): row
    for _, row in candidate_scores.iterrows()
}
branch_groups = {
    (str(dataset_name), str(case_id), int(budget)): group.copy()
    for (dataset_name, case_id, budget), group in branches.groupby(["dataset_name", "case_id", "budget"])
} if len(branches) else {}

display(predictions.groupby(["dataset_name", "budget"], as_index=False).agg(top1=("correct_top1", "mean"), top5=("gtpa_at_5", "mean"), n=("case_id", "count")))
display(candidate_pool_coverage.groupby("dataset_name", as_index=False).agg(pool_recall=("candidate_pool_hit", "mean"), mean_pool_size=("candidate_pool_size", "mean")))

# %% [markdown]
# ## 3. DDXPlus Graph Replay Features

# %%
def build_ddxplus_graph_features() -> pd.DataFrame:
    if graph_edges.empty:
        return pd.DataFrame()

    graph_paths = sorted(graph_edges["pathology"].astype(str).unique())
    graph_lookup = {
        (str(row["root_evidence_id"]), str(row["outcome_state"]).lower(), str(row["pathology"])): safe_float(row["log_odds_support"])
        for _, row in graph_edges.iterrows()
    }

    metadata_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in universal_cases.iterrows():
        if str(row["dataset_name"]) != "ddxplus":
            continue
        try:
            metadata_by_case[(str(row["dataset_name"]), str(row["case_id"]))] = json.loads(str(row.get("metadata", "{}")))
        except Exception:
            metadata_by_case[(str(row["dataset_name"]), str(row["case_id"]))] = {}

    rows: list[dict[str, Any]] = []
    for _, pred_row in predictions[predictions["dataset_name"].astype(str).eq("ddxplus")].iterrows():
        dataset_name = str(pred_row["dataset_name"])
        case_id = str(pred_row["case_id"])
        budget = int(pred_row["budget"])
        observed: dict[str, str] = {}

        metadata = metadata_by_case.get((dataset_name, case_id), {})
        initial_token = str(metadata.get("initial_evidence", "") or "")
        if initial_token.startswith("E_"):
            observed[initial_token.split("_@_")[0]] = "present"

        ledger_rows = ledger[
            (ledger["dataset_name"].astype(str) == dataset_name)
            & (ledger["case_id"].astype(str) == case_id)
            & (ledger["budget"].astype(int) == budget)
        ].copy()
        for _, ledger_row in ledger_rows.iterrows():
            root = extract_root_from_question(ledger_row.get("question", ""))
            outcome = outcome_from_ddxplus_answer(ledger_row.get("answer", ""))
            if root and outcome:
                observed[root] = outcome

        scores = {pathology: 0.0 for pathology in graph_paths}
        positive_support = {pathology: 0.0 for pathology in graph_paths}
        contradiction = {pathology: 0.0 for pathology in graph_paths}
        for root, outcome in observed.items():
            for pathology in graph_paths:
                value = max(-3.0, min(3.0, graph_lookup.get((root, outcome, pathology), 0.0)))
                scores[pathology] += value
                if value > 0:
                    positive_support[pathology] += value
                elif value < 0:
                    contradiction[pathology] += -value

        ranked = sorted(scores, key=scores.get, reverse=True)
        top_score = scores[ranked[0]] if ranked else 0.0
        second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
        exp_values = np.exp(np.array([scores[pathology] for pathology in ranked]) - top_score) if ranked else np.array([1.0])
        posterior = exp_values / exp_values.sum()
        posterior_by_pathology = {pathology: float(posterior[idx]) for idx, pathology in enumerate(ranked)}

        for pathology in graph_paths:
            rows.append({
                "dataset_name": dataset_name,
                "case_id": case_id,
                "budget": budget,
                "label_key": normalize_label(pathology),
                "ddxplus_graph_score": float(scores[pathology]),
                "ddxplus_graph_positive_support": float(positive_support[pathology]),
                "ddxplus_graph_contradiction": float(contradiction[pathology]),
                "ddxplus_graph_rank": int(ranked.index(pathology) + 1),
                "ddxplus_graph_rr": float(1.0 / (ranked.index(pathology) + 1)),
                "ddxplus_graph_posterior": float(posterior_by_pathology.get(pathology, 0.0)),
                "ddxplus_graph_top1": ranked[0] if ranked else "",
                "ddxplus_graph_margin": float(top_score - second_score),
                "ddxplus_graph_visible_roots": int(len(observed)),
            })

    return pd.DataFrame(rows)


ddxplus_graph_features = build_ddxplus_graph_features()
if len(ddxplus_graph_features):
    ddxplus_graph_features.to_csv(ARTIFACT_ROOT / "ddxplus_graph_replay_candidate_features.csv", index=False)
    graph_top1 = (
        ddxplus_graph_features[ddxplus_graph_features["ddxplus_graph_rank"].eq(1)]
        .merge(predictions[predictions["dataset_name"].astype(str).eq("ddxplus")], on=["dataset_name", "case_id", "budget"], how="left")
    )
    display(graph_top1[["case_id", "budget", "ground_truth_diagnosis", "ddxplus_graph_top1", "ddxplus_graph_margin"]].head())
else:
    display("No DDXPlus graph edge artifact was found; graph replay features will be zeroed.")

# %% [markdown]
# ## 4. Candidate-Level Educator Features

# %%
def branch_features_for_candidate(dataset_name: str, case_id: str, budget: int, label: str) -> dict[str, float]:
    group = branch_groups.get((dataset_name, case_id, budget))
    if group is None or not len(group):
        return {
            "branch_best_rr": 0.0,
            "branch_hit_count": 0.0,
            "branch_confidence_sum": 0.0,
            "branch_top1_count": 0.0,
        }

    best_rr = 0.0
    hit_count = 0.0
    confidence_sum = 0.0
    top1_count = 0.0
    for _, branch_row in group.iterrows():
        ranked = parse_json_list(branch_row.get("ranked_differential", ""))
        rr = reciprocal_rank(label, ranked)
        if rr > 0:
            best_rr = max(best_rr, rr)
            hit_count += 1.0
            confidence_sum += safe_float(branch_row.get("confidence", 0.0))
            if label_rank(label, ranked) == 1:
                top1_count += 1.0
    return {
        "branch_best_rr": float(best_rr),
        "branch_hit_count": float(hit_count),
        "branch_confidence_sum": float(confidence_sum),
        "branch_top1_count": float(top1_count),
    }


def case_candidate_list(dataset_name: str, case_id: str) -> list[str]:
    case_row = case_by_key.get((dataset_name, case_id))
    if case_row is None:
        return []
    return parse_json_list(case_row.get("candidate_disease_list", ""))


candidate_rows: list[dict[str, Any]] = []
deduped_pool = candidate_pool_long.copy()
deduped_pool["label_key"] = deduped_pool["label"].map(normalize_label)
deduped_pool = deduped_pool.drop_duplicates(["dataset_name", "case_id", "budget", "label_key"])

ddx_graph_by_key = {
    (str(row["dataset_name"]), str(row["case_id"]), int(row["budget"]), str(row["label_key"])): row
    for _, row in ddxplus_graph_features.iterrows()
} if len(ddxplus_graph_features) else {}

for _, pool_row in deduped_pool.iterrows():
    dataset_name = str(pool_row["dataset_name"])
    case_id = str(pool_row["case_id"])
    budget = int(pool_row["budget"])
    label = str(pool_row["label"])
    label_key = normalize_label(label)
    pred_row = prediction_by_key[(dataset_name, case_id, budget)]
    score_row = score_by_key.get((dataset_name, case_id, budget, label_key))
    source_values = parse_source_values(score_row.get("sources", "") if score_row is not None else "")

    final_ranked = parse_json_list(pred_row.get("ranked_differential", ""))
    llm_ranked = parse_json_list(pred_row.get("llm_ranked_differential", ""))
    mlp_ranked = parse_json_list(pred_row.get("ddxplus_mlp_top5", ""))
    allowed_list = case_candidate_list(dataset_name, case_id)
    branch_features = branch_features_for_candidate(dataset_name, case_id, budget, label)
    ddx_graph = ddx_graph_by_key.get((dataset_name, case_id, budget, label_key), {})

    current_key = normalize_label(pred_row.get("predicted_diagnosis", ""))
    truth_key = normalize_label(pred_row.get("ground_truth_diagnosis", ""))

    independent_signal_count = sum([
        reciprocal_rank(label, final_ranked) > 0,
        reciprocal_rank(label, llm_ranked) > 0,
        reciprocal_rank(label, mlp_ranked) > 0,
        source_values.get("casebase_prior", 0.0) > 0,
        source_values.get("rarebench_graph", 0.0) > 0,
        branch_features["branch_best_rr"] > 0,
        safe_float(ddx_graph.get("ddxplus_graph_score", 0.0)) > 0,
    ])

    candidate_rows.append({
        "workup_id": f"{dataset_name}|{case_id}|{budget}",
        "case_group": f"{dataset_name}|{case_id}",
        "dataset_name": dataset_name,
        "case_id": case_id,
        "budget": budget,
        "label": label,
        "label_key": label_key,
        "ground_truth_diagnosis": pred_row.get("ground_truth_diagnosis", ""),
        "is_truth": label_key == truth_key,
        "current_prediction": pred_row.get("predicted_diagnosis", ""),
        "current_correct": boolish(pred_row.get("correct_top1", False)),
        "is_current_prediction": label_key == current_key,
        "pool_rank": safe_int(pool_row.get("pool_rank", 99), 99),
        "pool_rr": 1.0 / safe_int(pool_row.get("pool_rank", 99), 99),
        "final_rank": label_rank(label, final_ranked),
        "final_rr": reciprocal_rank(label, final_ranked),
        "final_top1": int(label_rank(label, final_ranked) == 1),
        "final_top3": int(label_rank(label, final_ranked) <= 3),
        "final_top5": int(label_rank(label, final_ranked) <= 5),
        "llm_rank": label_rank(label, llm_ranked),
        "llm_rr": reciprocal_rank(label, llm_ranked),
        "llm_top1": int(label_rank(label, llm_ranked) == 1),
        "llm_top3": int(label_rank(label, llm_ranked) <= 3),
        "llm_top5": int(label_rank(label, llm_ranked) <= 5),
        "mlp_rank": label_rank(label, mlp_ranked),
        "mlp_rr": reciprocal_rank(label, mlp_ranked),
        "mlp_top1": int(normalize_label(pred_row.get("ddxplus_mlp_top1", "")) == label_key),
        "mlp_top5": int(label_rank(label, mlp_ranked) <= 5),
        "candidate_resolver_rank": safe_int(score_row.get("candidate_rank", 99), 99) if score_row is not None else 99,
        "candidate_resolver_rr": 1.0 / safe_int(score_row.get("candidate_rank", 99), 99) if score_row is not None else 0.0,
        "resolver_score": safe_float(score_row.get("resolver_score", 0.0)) if score_row is not None else 0.0,
        "support_count": safe_float(score_row.get("support_count", 0.0)) if score_row is not None else 0.0,
        "source_count": float(len(source_values)),
        "base_llm_rank_value": source_values.get("base_llm_rank", 0.0),
        "base_llm_confidence_value": source_values.get("base_llm_confidence", 0.0),
        "casebase_prior_value": source_values.get("casebase_prior", 0.0),
        "rarebench_graph_value": source_values.get("rarebench_graph", 0.0),
        "branch1_value": source_values.get("branch:branch_1", 0.0),
        "branch2_value": source_values.get("branch:branch_2", 0.0),
        "branch1_conf_value": source_values.get("branch_conf:branch_1", 0.0),
        "branch2_conf_value": source_values.get("branch_conf:branch_2", 0.0),
        **branch_features,
        "ddxplus_graph_score": safe_float(ddx_graph.get("ddxplus_graph_score", 0.0)),
        "ddxplus_graph_positive_support": safe_float(ddx_graph.get("ddxplus_graph_positive_support", 0.0)),
        "ddxplus_graph_contradiction": safe_float(ddx_graph.get("ddxplus_graph_contradiction", 0.0)),
        "ddxplus_graph_rank": safe_int(ddx_graph.get("ddxplus_graph_rank", 99), 99),
        "ddxplus_graph_rr": safe_float(ddx_graph.get("ddxplus_graph_rr", 0.0)),
        "ddxplus_graph_posterior": safe_float(ddx_graph.get("ddxplus_graph_posterior", 0.0)),
        "ddxplus_graph_top1_label": str(ddx_graph.get("ddxplus_graph_top1", "")),
        "ddxplus_graph_margin": safe_float(ddx_graph.get("ddxplus_graph_margin", 0.0)),
        "ddxplus_graph_visible_roots": safe_int(ddx_graph.get("ddxplus_graph_visible_roots", 0), 0),
        "candidate_in_allowed_list": int(any(normalize_label(item) == label_key for item in allowed_list)),
        "allowed_list_size": len(allowed_list),
        "num_questions": safe_float(pred_row.get("num_questions", 0.0)),
        "resolver_margin": safe_float(pred_row.get("resolver_margin", 0.0)),
        "resolver_support_count": safe_float(pred_row.get("resolver_support_count", 0.0)),
        "branch_triggered": int(boolish(pred_row.get("branch_triggered", False))),
        "branch_count": safe_int(pred_row.get("branch_count", 0)),
        "ddxplus_mlp_available": int(boolish(pred_row.get("ddxplus_mlp_available", False))),
        "ddxplus_mlp_confidence": safe_float(pred_row.get("ddxplus_mlp_confidence", 0.0)),
        "ddxplus_mlp_margin": safe_float(pred_row.get("ddxplus_mlp_margin", 0.0)),
        "ddxplus_mlp_entropy": safe_float(pred_row.get("ddxplus_mlp_entropy", 0.0)),
        "casebase_prior_top_score": safe_float(pred_row.get("casebase_prior_top_score", 0.0)),
        "casebase_prior_margin": safe_float(pred_row.get("casebase_prior_margin", 0.0)),
        "rarebench_graph_top_score": safe_float(pred_row.get("rarebench_graph_top_score", 0.0)),
        "rarebench_graph_margin": safe_float(pred_row.get("rarebench_graph_margin", 0.0)),
        "independent_signal_count": float(independent_signal_count),
    })

candidate_features = pd.DataFrame(candidate_rows)
candidate_features = normalize_by_workup(
    candidate_features,
    [
        "resolver_score",
        "support_count",
        "base_llm_rank_value",
        "casebase_prior_value",
        "rarebench_graph_value",
        "branch_best_rr",
        "branch_confidence_sum",
        "ddxplus_graph_score",
        "ddxplus_graph_posterior",
        "independent_signal_count",
    ],
)

candidate_features["source_weighted_score"] = (
    1.00 * candidate_features["final_rr"]
    + 0.70 * candidate_features["llm_rr"]
    + 0.80 * candidate_features["candidate_resolver_rr"]
    + 0.85 * candidate_features["mlp_rr"]
    + 0.35 * candidate_features["casebase_prior_value_norm"]
    + 0.45 * candidate_features["rarebench_graph_value_norm"]
    + 0.50 * candidate_features["branch_best_rr"]
    + 0.30 * candidate_features["ddxplus_graph_rr"]
    + 0.08 * candidate_features["support_count"]
)
candidate_features["source_weighted_score_minus_current"] = (
    candidate_features["source_weighted_score"]
    - candidate_features.groupby("workup_id")["source_weighted_score"].transform(
        lambda values: values[candidate_features.loc[values.index, "is_current_prediction"].astype(bool)].iloc[0]
        if candidate_features.loc[values.index, "is_current_prediction"].any()
        else values.max()
    )
)

candidate_features.to_csv(ARTIFACT_ROOT / "candidate_level_educator_features.csv", index=False)
display(candidate_features.head())
display(candidate_features.groupby("dataset_name", as_index=False).agg(num_candidates=("label", "count"), truth_rows=("is_truth", "sum")))

# %% [markdown]
# ## 5. Label-Free Candidate-Pool Policies

# %%
def candidates_for_workup(row: pd.Series) -> pd.DataFrame:
    return candidate_features[candidate_features["workup_id"].eq(f"{row['dataset_name']}|{row['case_id']}|{int(row['budget'])}")].copy()


def current_candidate_row(group: pd.DataFrame) -> pd.Series:
    current_rows = group[group["is_current_prediction"].astype(bool)]
    if len(current_rows):
        return current_rows.iloc[0]
    return group.sort_values(["final_rr", "pool_rr"], ascending=[False, False]).iloc[0]


def ddxplus_high_conf_mlp_label(row: pd.Series, group: pd.DataFrame) -> str | None:
    if (
        str(row["dataset_name"]) == "ddxplus"
        and boolish(row.get("ddxplus_mlp_available", False))
        and str(row.get("ddxplus_mlp_top1", "")).strip()
        and safe_float(row.get("ddxplus_mlp_confidence", 0.0)) >= DDXPLUS_MLP_GUARD_CONFIDENCE_MIN
        and safe_float(row.get("ddxplus_mlp_margin", 0.0)) >= DDXPLUS_MLP_GUARD_MARGIN_MIN
    ):
        label = str(row["ddxplus_mlp_top1"])
        if normalize_label(label) in set(group["label_key"]):
            return label
    return None


def ddxplus_graph_override_label(row: pd.Series, group: pd.DataFrame) -> str | None:
    if str(row["dataset_name"]) != "ddxplus":
        return None
    current = current_candidate_row(group)
    graph_top_rows = group[group["ddxplus_graph_rank"].eq(1)]
    if not len(graph_top_rows):
        return None
    graph_top = graph_top_rows.iloc[0]
    if normalize_label(graph_top["label"]) == normalize_label(row.get("predicted_diagnosis", "")):
        return None
    if (
        safe_float(graph_top["ddxplus_graph_margin"]) >= DDXPLUS_GRAPH_OVERRIDE_MARGIN_MIN
        and safe_float(current["ddxplus_graph_score"]) < DDXPLUS_GRAPH_CURRENT_SCORE_MAX
        and safe_float(graph_top["ddxplus_graph_score"]) > 0.0
    ):
        return str(graph_top["label"])
    return None


def source_weighted_challenger_label(group: pd.DataFrame) -> str | None:
    current = current_candidate_row(group)
    top = group.sort_values(["source_weighted_score", "independent_signal_count", "pool_rr"], ascending=[False, False, False]).iloc[0]
    score_margin = safe_float(top["source_weighted_score"] - current["source_weighted_score"])
    signal_delta = safe_float(top["independent_signal_count"] - current["independent_signal_count"])
    if normalize_label(top["label"]) == normalize_label(current["label"]):
        return None
    if score_margin >= SOURCE_WEIGHTED_OVERRIDE_MARGIN_MIN and signal_delta >= 1.0:
        return str(top["label"])
    return None


def ranked_by_score(group: pd.DataFrame, score_column: str, current_ranked: list[str]) -> list[str]:
    labels = [
        str(row["label"])
        for _, row in group.sort_values([score_column, "pool_rr"], ascending=[False, False]).iterrows()
    ]
    seen: set[str] = set()
    ranked: list[str] = []
    for label in [*labels, *current_ranked]:
        key = normalize_label(label)
        if key and key not in seen:
            ranked.append(label)
            seen.add(key)
    return ranked[:10]


def apply_label_free_policy(row: pd.Series, policy_name: str) -> tuple[str, list[str], str]:
    current_ranked = ranked_from_prediction_row(row)
    current_prediction = current_ranked[0] if current_ranked else str(row.get("predicted_diagnosis", ""))
    group = candidates_for_workup(row)

    if policy_name == "notebook46_current":
        return current_prediction, current_ranked, "current"

    if policy_name == "ddxplus_high_conf_mlp_guard_v1":
        mlp_label = ddxplus_high_conf_mlp_label(row, group)
        if mlp_label:
            return mlp_label, insert_top(current_ranked, mlp_label), "ddxplus_high_conf_mlp_guard"
        return current_prediction, current_ranked, "current"

    if policy_name == "ddxplus_graph_mlp_guard_v1":
        mlp_label = ddxplus_high_conf_mlp_label(row, group)
        if mlp_label:
            return mlp_label, insert_top(current_ranked, mlp_label), "ddxplus_high_conf_mlp_guard"
        graph_label = ddxplus_graph_override_label(row, group)
        if graph_label:
            return graph_label, insert_top(current_ranked, graph_label), "ddxplus_graph_conservative_override"
        return current_prediction, current_ranked, "current"

    if policy_name == "conservative_pool_educator_v1":
        mlp_label = ddxplus_high_conf_mlp_label(row, group)
        if mlp_label:
            return mlp_label, insert_top(current_ranked, mlp_label), "ddxplus_high_conf_mlp_guard"
        graph_label = ddxplus_graph_override_label(row, group)
        if graph_label:
            return graph_label, insert_top(current_ranked, graph_label), "ddxplus_graph_conservative_override"
        challenger = source_weighted_challenger_label(group)
        if challenger:
            return challenger, insert_top(ranked_by_score(group, "source_weighted_score", current_ranked), challenger), "source_weighted_pool_challenger"
        return current_prediction, current_ranked, "current"

    if policy_name == "source_weighted_pool_top1_diagnostic":
        ranked = ranked_by_score(group, "source_weighted_score", current_ranked)
        return ranked[0], ranked, "source_weighted_top1"

    if policy_name == "candidate_pool_oracle_diagnostic":
        truth = str(row["ground_truth_diagnosis"])
        if normalize_label(truth) in set(group["label_key"]):
            ranked = ranked_by_score(group.assign(candidate_pool_oracle_score=group["is_truth"].astype(float)), "candidate_pool_oracle_score", current_ranked)
            return truth, insert_top(ranked, truth), "candidate_pool_oracle"
        return current_prediction, current_ranked, "current"

    raise ValueError(policy_name)


label_free_policy_names = [
    "notebook46_current",
    "ddxplus_high_conf_mlp_guard_v1",
    "ddxplus_graph_mlp_guard_v1",
    "conservative_pool_educator_v1",
    "source_weighted_pool_top1_diagnostic",
    "candidate_pool_oracle_diagnostic",
]

label_free_rows: list[dict[str, Any]] = []
for policy_name in label_free_policy_names:
    for _, row in predictions.iterrows():
        prediction, ranked, action = apply_label_free_policy(row, policy_name)
        metrics = score_ranked(row, prediction, ranked)
        label_free_rows.append({
            "policy_name": policy_name,
            "dataset_name": row["dataset_name"],
            "case_id": row["case_id"],
            "budget": int(row["budget"]),
            "ground_truth_diagnosis": row["ground_truth_diagnosis"],
            "original_prediction": row["predicted_diagnosis"],
            "policy_prediction": prediction,
            "policy_action": action,
            "policy_ranked_differential": json.dumps(ranked[:10], ensure_ascii=True),
            **metrics,
            "original_correct_top1": boolish(row.get("correct_top1", False)),
            "original_gtpa_at_3": boolish(row.get("gtpa_at_3", False)),
            "original_gtpa_at_5": boolish(row.get("gtpa_at_5", False)),
            "changed_prediction": normalize_label(prediction) != normalize_label(row["predicted_diagnosis"]),
        })

label_free_results = pd.DataFrame(label_free_rows)
label_free_results.to_csv(ARTIFACT_ROOT / "label_free_pool_educator_results.csv", index=False)

summary_rows: list[dict[str, Any]] = []
for (policy_name, dataset_name, budget), group in label_free_results.groupby(["policy_name", "dataset_name", "budget"], sort=True):
    summary_rows.append({
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
for policy_name, group in label_free_results.groupby("policy_name", sort=True):
    summary_rows.append({
        "policy_name": policy_name,
        "dataset_name": "ALL",
        "budget": -1,
        "num_workups": int(len(group)),
        "top1": float(group["correct_top1"].mean()),
        "top3": float(group["gtpa_at_3"].mean()),
        "top5": float(group["gtpa_at_5"].mean()),
        "wins_vs_current": int((group["correct_top1"] & ~group["original_correct_top1"]).sum()),
        "regressions_vs_current": int((~group["correct_top1"] & group["original_correct_top1"]).sum()),
        "changed_predictions": int(group["changed_prediction"].sum()),
    })
label_free_summary = pd.DataFrame(summary_rows)
label_free_summary.to_csv(ARTIFACT_ROOT / "label_free_pool_educator_summary.csv", index=False)
display(label_free_summary[label_free_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))

# %% [markdown]
# ## 6. Learned Diagnostic Educator
#
# These models use labels from the Notebook 46 evaluation cohort, so they are diagnostic unless trained on a
# separate calibration set and confirmed on held-out workups. They are included to answer a different question:
# do the current candidate features contain enough information to identify the correct candidate?

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
feature_columns = [
    column for column in candidate_features.columns
    if column not in excluded_feature_columns and candidate_features[column].dtype != "object"
]
categorical_columns = ["dataset_name"]
numeric_columns = feature_columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

candidate_X = candidate_features[numeric_columns + categorical_columns]
candidate_y = candidate_features["is_truth"].astype(int)
case_groups = candidate_features["case_group"]


def evaluate_candidate_probabilities(probabilities: np.ndarray, policy_name: str) -> pd.DataFrame:
    scored = candidate_features.copy()
    scored["educator_probability"] = probabilities
    rows: list[dict[str, Any]] = []
    for workup_id, group in scored.groupby("workup_id", sort=True):
        dataset_name, case_id, budget_text = workup_id.split("|", 2)
        budget = int(budget_text)
        pred_row = prediction_by_key[(dataset_name, case_id, budget)]
        current_ranked = ranked_from_prediction_row(pred_row)
        ranked = [
            str(row["label"])
            for _, row in group.sort_values(["educator_probability", "pool_rr"], ascending=[False, False]).iterrows()
        ]
        ranked = [*ranked, *current_ranked]
        deduped: list[str] = []
        seen: set[str] = set()
        for label in ranked:
            key = normalize_label(label)
            if key and key not in seen:
                deduped.append(label)
                seen.add(key)
        prediction = deduped[0]
        metrics = score_ranked(pred_row, prediction, deduped[:10])
        rows.append({
            "policy_name": policy_name,
            "dataset_name": dataset_name,
            "case_id": case_id,
            "budget": budget,
            "ground_truth_diagnosis": pred_row["ground_truth_diagnosis"],
            "original_prediction": pred_row["predicted_diagnosis"],
            "policy_prediction": prediction,
            "educator_probability": float(group.sort_values(["educator_probability", "pool_rr"], ascending=[False, False]).iloc[0]["educator_probability"]),
            **metrics,
            "original_correct_top1": boolish(pred_row.get("correct_top1", False)),
            "changed_prediction": normalize_label(prediction) != normalize_label(pred_row["predicted_diagnosis"]),
        })
    return pd.DataFrame(rows)


label_fit_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, l2_regularization=0.5, random_state=48)),
])
label_fit_model.fit(candidate_X, candidate_y)
label_fit_prob = label_fit_model.predict_proba(candidate_X)[:, 1]
label_fit_results = evaluate_candidate_probabilities(label_fit_prob, "label_fit_hgb_candidate_educator_diagnostic")

case_blocked_prob = np.zeros(len(candidate_features))
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(candidate_X, candidate_y, case_groups):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, l2_regularization=0.5, random_state=48)),
    ])
    model.fit(candidate_X.iloc[train_idx], candidate_y.iloc[train_idx])
    case_blocked_prob[test_idx] = model.predict_proba(candidate_X.iloc[test_idx])[:, 1]
case_blocked_results = evaluate_candidate_probabilities(case_blocked_prob, "case_blocked_hgb_candidate_educator_diagnostic")

logistic_label_fit = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
])
logistic_label_fit.fit(candidate_X, candidate_y)
logistic_fit_prob = logistic_label_fit.predict_proba(candidate_X)[:, 1]
logistic_fit_results = evaluate_candidate_probabilities(logistic_fit_prob, "label_fit_logistic_candidate_educator_diagnostic")

learned_results = pd.concat([label_fit_results, case_blocked_results, logistic_fit_results], ignore_index=True)
learned_results.to_csv(ARTIFACT_ROOT / "case_level_learned_diagnostic_results.csv", index=False)

learned_summary_rows = []
for policy_name, group in learned_results.groupby("policy_name", sort=True):
    learned_summary_rows.append({
        "policy_name": policy_name,
        "dataset_name": "ALL",
        "budget": -1,
        "num_workups": int(len(group)),
        "top1": float(group["correct_top1"].mean()),
        "top3": float(group["gtpa_at_3"].mean()),
        "top5": float(group["gtpa_at_5"].mean()),
        "wins_vs_current": int((group["correct_top1"] & ~group["original_correct_top1"]).sum()),
        "regressions_vs_current": int((~group["correct_top1"] & group["original_correct_top1"]).sum()),
        "changed_predictions": int(group["changed_prediction"].sum()),
    })
for (policy_name, dataset_name, budget), group in learned_results.groupby(["policy_name", "dataset_name", "budget"], sort=True):
    learned_summary_rows.append({
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
learned_summary = pd.DataFrame(learned_summary_rows)
learned_summary.to_csv(ARTIFACT_ROOT / "learned_pool_educator_summary.csv", index=False)
display(learned_summary[learned_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))

# %% [markdown]
# ## 7. Error And Signal Analysis

# %%
selected_policy_name = "conservative_pool_educator_v1"
selected_results = label_free_results[label_free_results["policy_name"].eq(selected_policy_name)].copy()
current_results = label_free_results[label_free_results["policy_name"].eq("notebook46_current")].copy()
pool_oracle_results = label_free_results[label_free_results["policy_name"].eq("candidate_pool_oracle_diagnostic")].copy()

audit_rows: list[dict[str, Any]] = []
for _, current_row in current_results[~current_results["correct_top1"]].iterrows():
    group = candidate_features[candidate_features["workup_id"].eq(f"{current_row['dataset_name']}|{current_row['case_id']}|{int(current_row['budget'])}")].copy()
    truth_rows = group[group["is_truth"].astype(bool)]
    current_candidate = current_candidate_row(group)
    truth_candidate = truth_rows.iloc[0] if len(truth_rows) else None
    selected_row = selected_results[
        (selected_results["dataset_name"].astype(str) == str(current_row["dataset_name"]))
        & (selected_results["case_id"].astype(str) == str(current_row["case_id"]))
        & (selected_results["budget"].astype(int) == int(current_row["budget"]))
    ].iloc[0]

    audit_rows.append({
        "dataset_name": current_row["dataset_name"],
        "case_id": current_row["case_id"],
        "budget": int(current_row["budget"]),
        "ground_truth_diagnosis": current_row["ground_truth_diagnosis"],
        "current_prediction": current_row["policy_prediction"],
        "selected_prediction": selected_row["policy_prediction"],
        "selected_action": selected_row["policy_action"],
        "candidate_pool_hit": bool(len(truth_rows)),
        "truth_pool_rank": int(truth_candidate["pool_rank"]) if truth_candidate is not None else 99,
        "truth_final_rank": int(truth_candidate["final_rank"]) if truth_candidate is not None else 99,
        "truth_source_weighted_score": float(truth_candidate["source_weighted_score"]) if truth_candidate is not None else 0.0,
        "current_source_weighted_score": float(current_candidate["source_weighted_score"]),
        "truth_independent_signal_count": float(truth_candidate["independent_signal_count"]) if truth_candidate is not None else 0.0,
        "current_independent_signal_count": float(current_candidate["independent_signal_count"]),
        "truth_ddxplus_graph_rank": int(truth_candidate["ddxplus_graph_rank"]) if truth_candidate is not None else 99,
        "current_ddxplus_graph_rank": int(current_candidate["ddxplus_graph_rank"]),
        "truth_mlp_rank": int(truth_candidate["mlp_rank"]) if truth_candidate is not None else 99,
        "current_mlp_rank": int(current_candidate["mlp_rank"]),
    })

signal_audit = pd.DataFrame(audit_rows)
signal_audit.to_csv(ARTIFACT_ROOT / "signal_failure_audit.csv", index=False)
display(signal_audit)

source_presence_rows = []
for source_name, column in [
    ("final_top5", "final_top5"),
    ("llm_top5", "llm_top5"),
    ("ddxplus_mlp_top5", "mlp_top5"),
    ("casebase_prior", "casebase_prior_value"),
    ("rarebench_graph", "rarebench_graph_value"),
    ("branch_rank", "branch_best_rr"),
    ("ddxplus_graph_top5", "ddxplus_graph_rank"),
]:
    truth_candidates = candidate_features[candidate_features["is_truth"].astype(bool)].copy()
    if column == "ddxplus_graph_rank":
        present = truth_candidates[column].le(5)
    else:
        present = truth_candidates[column].astype(float).gt(0)
    source_presence_rows.append({
        "source_signal": source_name,
        "truth_coverage_rate": float(present.mean()),
        "truth_coverage_count": int(present.sum()),
        "num_workups_with_truth_in_pool": int(len(truth_candidates)),
    })
source_presence = pd.DataFrame(source_presence_rows)
source_presence.to_csv(ARTIFACT_ROOT / "truth_source_signal_coverage.csv", index=False)
display(source_presence)

# %% [markdown]
# ## 8. Figures

# %%
all_policy_summary = pd.concat([
    label_free_summary[label_free_summary["dataset_name"].eq("ALL")],
    learned_summary[learned_summary["dataset_name"].eq("ALL")],
], ignore_index=True)
plot_summary = all_policy_summary.sort_values("top1", ascending=True)
plt.figure(figsize=(9, 5))
colors = ["#4C78A8" if "diagnostic" not in name else "#F58518" for name in plot_summary["policy_name"]]
plt.barh(plot_summary["policy_name"], plot_summary["top1"], color=colors)
plt.xlim(0, 1.0)
plt.xlabel("Top-1 accuracy")
plt.title("Candidate-Pool Adjudicator Policies")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "candidate_pool_adjudicator_top1.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
pool_truth = candidate_features[candidate_features["is_truth"].astype(bool)]["pool_rank"].value_counts().sort_index()
plt.bar(pool_truth.index.astype(str), pool_truth.values, color="#54A24B")
plt.xlabel("Truth rank in broad candidate pool")
plt.ylabel("Workups")
plt.title("Broad Candidate-Pool Truth Rank")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "truth_pool_rank_distribution.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
plt.bar(source_presence["source_signal"], source_presence["truth_coverage_rate"], color="#72B7B2")
plt.ylim(0, 1.05)
plt.ylabel("Truth coverage rate")
plt.xticks(rotation=35, ha="right")
plt.title("Where The Correct Candidate Appears")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "truth_source_signal_coverage.png", dpi=180)
plt.close()

gap_rows = all_policy_summary[
    all_policy_summary["policy_name"].isin([
        "notebook46_current",
        selected_policy_name,
        "case_blocked_hgb_candidate_educator_diagnostic",
        "label_fit_hgb_candidate_educator_diagnostic",
        "candidate_pool_oracle_diagnostic",
    ])
].copy()
plt.figure(figsize=(8, 4))
plt.plot(gap_rows["policy_name"], gap_rows["top1"], marker="o", color="#E45756")
plt.ylim(0, 1.0)
plt.ylabel("Top-1 accuracy")
plt.xticks(rotation=30, ha="right")
plt.title("Resolver Signal vs Candidate-Pool Ceiling")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "resolver_generalization_gap.png", dpi=180)
plt.close()

# %% [markdown]
# ## 9. Final Summary And Artifact Contract

# %%
selected_overall = label_free_summary[
    label_free_summary["policy_name"].eq(selected_policy_name) & label_free_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
current_overall = label_free_summary[
    label_free_summary["policy_name"].eq("notebook46_current") & label_free_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
pool_oracle_overall = label_free_summary[
    label_free_summary["policy_name"].eq("candidate_pool_oracle_diagnostic") & label_free_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
case_blocked_overall = learned_summary[
    learned_summary["policy_name"].eq("case_blocked_hgb_candidate_educator_diagnostic") & learned_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
label_fit_overall = learned_summary[
    learned_summary["policy_name"].eq("label_fit_hgb_candidate_educator_diagnostic") & learned_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()

resolved_config = {
    "run_name": RUN_NAME,
    "input_run_name": INPUT_RUN_NAME,
    "repair_run_name": REPAIR_RUN_NAME,
    "input_root": str(INPUT_ROOT),
    "repair_root": str(REPAIR_ROOT),
    "artifact_root": str(ARTIFACT_ROOT),
    "live_api_used": False,
    "selected_policy_name": selected_policy_name,
    "label_free_thresholds": {
        "ddxplus_mlp_guard_confidence_min": DDXPLUS_MLP_GUARD_CONFIDENCE_MIN,
        "ddxplus_mlp_guard_margin_min": DDXPLUS_MLP_GUARD_MARGIN_MIN,
        "ddxplus_graph_override_margin_min": DDXPLUS_GRAPH_OVERRIDE_MARGIN_MIN,
        "ddxplus_graph_current_score_max": DDXPLUS_GRAPH_CURRENT_SCORE_MAX,
        "source_weighted_override_margin_min": SOURCE_WEIGHTED_OVERRIDE_MARGIN_MIN,
    },
    "feature_sources": [
        "Notebook 46 final and LLM ranked differentials",
        "Notebook 46 branch ranked differentials",
        "Notebook 46 casebase prior and RareBench graph scores",
        "Notebook 46 DDXPlus MLP top-5 and confidence fields",
        "Notebook 47 broad candidate pool",
        "Notebook 16 train-derived DDXPlus graph edges replayed over Notebook 46 revealed evidence",
    ],
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

selected_payload = {
    "selected_policy_name": selected_policy_name,
    "status": "offline_candidate_not_yet_live_confirmed",
    "input_run_name": INPUT_RUN_NAME,
    "current_overall": current_overall,
    "selected_overall": selected_overall,
    "case_blocked_learned_diagnostic_overall": case_blocked_overall,
    "label_fit_learned_diagnostic_overall": label_fit_overall,
    "candidate_pool_oracle_overall": pool_oracle_overall,
    "promotion_decision": "Do not claim the learned educator as final. The selected label-free policy is safe but only modestly improves Notebook 46; the label-fit diagnostic exposes resolver headroom but needs a separate calibration cohort before deployment.",
    "interpretation": [
        "The broad pool remains the right abstraction: candidate-pool oracle is near ceiling, so acquisition is no longer the main failure on this run.",
        "The conservative educator mainly protects high-confidence DDXPlus MLP decisions and avoids regressions.",
        "Train-derived DDXPlus graph replay agrees with the MLP on the clean URTI/Bronchitis failures, but does not rescue the remaining ambiguous DDXPlus cases.",
        "A label-fit learned educator can approach the candidate-pool ceiling, but case-blocked validation is much weaker, meaning the current 90-workup cohort is too small to deploy a learned universal resolver honestly.",
    ],
    "artifact_contract": [
        "resolved_run_config.json",
        "candidate_level_educator_features.csv",
        "ddxplus_graph_replay_candidate_features.csv",
        "label_free_pool_educator_results.csv",
        "label_free_pool_educator_summary.csv",
        "case_level_learned_diagnostic_results.csv",
        "learned_pool_educator_summary.csv",
        "signal_failure_audit.csv",
        "truth_source_signal_coverage.csv",
        "selected_pool_educator.json",
        "figures/",
    ],
}
write_json(ARTIFACT_ROOT / "selected_pool_educator.json", selected_payload)

print("Wrote artifacts to:", ARTIFACT_ROOT)
display(label_free_summary[label_free_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))
display(learned_summary[learned_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))
