from __future__ import annotations

# %% [markdown]
# # Notebook 47: MEDDx Candidate-Pool Repair Lab
#
# This offline lab tests whether the successful DDXPlus candidate-pool architecture survived the Notebook 46
# MEDDx-aligned multi-dataset run. It does not make API calls. Its job is to answer:
#
# 1. Is the correct diagnosis present in the candidate pool?
# 2. Are the remaining errors mostly acquisition failures or resolver failures?
# 3. Which general, label-free repair rules are safe enough to move into the next live notebook?

# %%
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
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

pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_rows", 120)

ROOT = next((candidate for candidate in [Path.cwd(), *Path.cwd().parents] if (candidate / "notebooks").exists() and (candidate / "reports").exists()), Path.cwd())

INPUT_RUN_NAME = "meddx_aligned_dataset_native_driver_v1_eval30"
INPUT_ROOT = ROOT / "artifacts" / "universal_meddx" / INPUT_RUN_NAME
RUN_NAME = "meddx_candidate_pool_repair_lab_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DDXPLUS_MLP_GUARD_CONFIDENCE_MIN = 0.70
DDXPLUS_MLP_GUARD_MARGIN_MIN = 0.20

print("Project root:", ROOT)
print("Input root  :", INPUT_ROOT)
print("Artifact root:", ARTIFACT_ROOT)

# %% [markdown]
# ## 1. Load Notebook 46 Artifacts

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


required = [
    "resolved_run_config.json",
    "predictions.csv",
    "candidate_level_resolver_scores.csv",
    "question_answer_ledger.csv",
    "interaction_traces.jsonl",
    "universal_cases.csv",
]
missing = [name for name in required if not (INPUT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing Notebook 46 artifacts: {missing}")

config = read_json(INPUT_ROOT / "resolved_run_config.json")
predictions = pd.read_csv(INPUT_ROOT / "predictions.csv")
candidate_scores = pd.read_csv(INPUT_ROOT / "candidate_level_resolver_scores.csv")
ledger = pd.read_csv(INPUT_ROOT / "question_answer_ledger.csv")
universal_cases = pd.read_csv(INPUT_ROOT / "universal_cases.csv")
branches = pd.read_csv(INPUT_ROOT / "branch_case_results.csv") if (INPUT_ROOT / "branch_case_results.csv").exists() else pd.DataFrame()

traces = []
with (INPUT_ROOT / "interaction_traces.jsonl").open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            traces.append(json.loads(line))
trace_by_key = {(row["dataset_name"], row["case_id"], int(row["budget"])): row for row in traces}
case_by_key = {(row["dataset_name"], row["case_id"]): row for _, row in universal_cases.iterrows()}

display(predictions.head())
display(pd.read_csv(INPUT_ROOT / "meddx_style_metrics_summary.csv"))

# %% [markdown]
# ## 2. Candidate-Pool Reconstruction

# %%
def add_candidate(pool: list[dict[str, Any]], label: Any, source: str, rank: int | None = None, score: float | None = None) -> None:
    label = str(label or "").strip()
    key = normalize_label(label)
    if not key or key == "nan":
        return
    pool.append({"label": label, "key": key, "source": source, "rank": rank, "score": score})


def candidate_pool_for_row(row: pd.Series) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for source, column in [
        ("final_rank", "ranked_differential"),
        ("llm_rank", "llm_ranked_differential"),
        ("ddxplus_mlp_top5", "ddxplus_mlp_top5"),
    ]:
        for idx, label in enumerate(parse_json_list(row.get(column, "")), start=1):
            add_candidate(pool, label, source, idx)
    for source, column in [
        ("final_top1", "predicted_diagnosis"),
        ("llm_top1", "llm_predicted_diagnosis"),
        ("casebase_top1", "casebase_prior_top_label"),
        ("rarebench_graph_top1", "rarebench_graph_top_label"),
        ("ddxplus_mlp_top1", "ddxplus_mlp_top1"),
    ]:
        add_candidate(pool, row.get(column, ""), source, 1)
    score_rows = candidate_scores[
        (candidate_scores["dataset_name"].astype(str) == str(row["dataset_name"]))
        & (candidate_scores["case_id"].astype(str) == str(row["case_id"]))
        & (candidate_scores["budget"].astype(int) == int(row["budget"]))
    ].copy()
    for _, score_row in score_rows.iterrows():
        add_candidate(
            pool,
            score_row["label"],
            "candidate_resolver_score",
            int(score_row.get("candidate_rank", 99)),
            float(score_row.get("resolver_score", 0.0)),
        )
    if len(branches):
        branch_rows = branches[
            (branches["dataset_name"].astype(str) == str(row["dataset_name"]))
            & (branches["case_id"].astype(str) == str(row["case_id"]))
            & (branches["budget"].astype(int) == int(row["budget"]))
        ].copy()
        for _, branch_row in branch_rows.iterrows():
            branch_id = str(branch_row.get("branch_id", "branch"))
            add_candidate(pool, branch_row.get("predicted_diagnosis", ""), f"{branch_id}_top1", 1)
            for idx, label in enumerate(parse_json_list(branch_row.get("ranked_differential", "")), start=1):
                add_candidate(pool, label, f"{branch_id}_rank", idx)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in pool:
        if item["key"] not in seen:
            deduped.append(item)
            seen.add(item["key"])
    return deduped


pool_rows = []
candidate_level_rows = []
for _, row in predictions.iterrows():
    pool = candidate_pool_for_row(row)
    truth_key = normalize_label(row["ground_truth_diagnosis"])
    pool_keys = [item["key"] for item in pool]
    hit = truth_key in set(pool_keys)
    rank = pool_keys.index(truth_key) + 1 if hit else 99
    pool_rows.append({
        "dataset_name": row["dataset_name"],
        "case_id": row["case_id"],
        "budget": int(row["budget"]),
        "ground_truth_diagnosis": row["ground_truth_diagnosis"],
        "current_prediction": row["predicted_diagnosis"],
        "current_correct": bool(row["correct_top1"]),
        "current_top3": bool(row["gtpa_at_3"]),
        "current_top5": bool(row["gtpa_at_5"]),
        "candidate_pool_size": len(pool),
        "candidate_pool_hit": bool(hit),
        "candidate_pool_rank": int(rank),
        "candidate_pool_labels": json.dumps([item["label"] for item in pool], ensure_ascii=True),
        "candidate_pool_sources": json.dumps(pool, ensure_ascii=True),
    })
    for idx, item in enumerate(pool, start=1):
        candidate_level_rows.append({
            "dataset_name": row["dataset_name"],
            "case_id": row["case_id"],
            "budget": int(row["budget"]),
            "pool_rank": idx,
            "label": item["label"],
            "source": item["source"],
            "source_rank": item["rank"],
            "source_score": item["score"],
            "is_truth": item["key"] == truth_key,
        })

pool_frame = pd.DataFrame(pool_rows)
candidate_pool_long = pd.DataFrame(candidate_level_rows)
pool_frame.to_csv(ARTIFACT_ROOT / "candidate_pool_coverage.csv", index=False)
candidate_pool_long.to_csv(ARTIFACT_ROOT / "candidate_pool_long.csv", index=False)

pool_summary = (
    pool_frame.groupby(["dataset_name", "budget"], as_index=False)
    .agg(
        num_workups=("case_id", "count"),
        current_top1=("current_correct", "mean"),
        current_top5=("current_top5", "mean"),
        candidate_pool_recall=("candidate_pool_hit", "mean"),
        mean_pool_size=("candidate_pool_size", "mean"),
        median_truth_pool_rank=("candidate_pool_rank", "median"),
    )
)
pool_summary.to_csv(ARTIFACT_ROOT / "candidate_pool_summary.csv", index=False)
display(pool_summary)
display(pool_frame[~pool_frame["candidate_pool_hit"]])

# %% [markdown]
# ## 3. General Policy Variants

# %%
def ranked_from_row(row: pd.Series) -> list[str]:
    ranked = parse_json_list(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "").strip()
    if prediction and normalize_label(prediction) not in {normalize_label(label) for label in ranked}:
        ranked = [prediction] + ranked
    return ranked


def insert_top(ranked: list[str], label: str) -> list[str]:
    label_key = normalize_label(label)
    out = [label]
    for candidate in ranked:
        if normalize_label(candidate) != label_key:
            out.append(candidate)
    return out[:10]


def score_prediction(row: pd.Series, prediction: str, ranked: list[str]) -> dict[str, Any]:
    truth = normalize_label(row["ground_truth_diagnosis"])
    keys = [normalize_label(label) for label in ranked]
    return {
        "correct_top1": normalize_label(prediction) == truth,
        "gtpa_at_3": truth in set(keys[:3]),
        "gtpa_at_5": truth in set(keys[:5]),
        "true_rank": keys.index(truth) + 1 if truth in keys else 11,
    }


def apply_policy(row: pd.Series, policy_name: str) -> tuple[str, list[str], str]:
    current_ranked = ranked_from_row(row)
    current_prediction = current_ranked[0] if current_ranked else str(row.get("predicted_diagnosis", ""))
    if policy_name == "notebook46_current":
        return current_prediction, current_ranked, "current"
    if policy_name == "ddxplus_high_conf_mlp_guard_v1":
        if (
            str(row["dataset_name"]) == "ddxplus"
            and bool(row.get("ddxplus_mlp_available", False))
            and str(row.get("ddxplus_mlp_top1", "")).strip()
            and float(row.get("ddxplus_mlp_confidence", 0.0) or 0.0) >= DDXPLUS_MLP_GUARD_CONFIDENCE_MIN
            and float(row.get("ddxplus_mlp_margin", 0.0) or 0.0) >= DDXPLUS_MLP_GUARD_MARGIN_MIN
        ):
            label = str(row["ddxplus_mlp_top1"])
            return label, insert_top(current_ranked, label), "ddxplus_high_conf_mlp_guard"
        return current_prediction, current_ranked, "current"
    if policy_name == "ddxplus_all_mlp_top1_diagnostic":
        if str(row["dataset_name"]) == "ddxplus" and str(row.get("ddxplus_mlp_top1", "")).strip():
            label = str(row["ddxplus_mlp_top1"])
            return label, insert_top(current_ranked, label), "ddxplus_mlp_top1"
        return current_prediction, current_ranked, "current"
    if policy_name == "current_top5_oracle_diagnostic":
        truth = str(row["ground_truth_diagnosis"])
        keys = [normalize_label(label) for label in current_ranked[:5]]
        if normalize_label(truth) in keys:
            return truth, insert_top(current_ranked, truth), "top5_oracle"
        return current_prediction, current_ranked, "current"
    if policy_name == "candidate_pool_oracle_diagnostic":
        pool = candidate_pool_for_row(row)
        truth = str(row["ground_truth_diagnosis"])
        if normalize_label(truth) in {item["key"] for item in pool}:
            pool_ranked = [item["label"] for item in pool]
            return truth, insert_top(pool_ranked, truth), "candidate_pool_oracle"
        return current_prediction, current_ranked, "current"
    raise ValueError(policy_name)


policy_names = [
    "notebook46_current",
    "ddxplus_high_conf_mlp_guard_v1",
    "ddxplus_all_mlp_top1_diagnostic",
    "current_top5_oracle_diagnostic",
    "candidate_pool_oracle_diagnostic",
]

policy_rows = []
case_policy_rows = []
for policy_name in policy_names:
    for _, row in predictions.iterrows():
        prediction, ranked, action = apply_policy(row, policy_name)
        metrics = score_prediction(row, prediction, ranked)
        case_policy_rows.append({
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
            "original_correct_top1": bool(row["correct_top1"]),
            "original_gtpa_at_3": bool(row["gtpa_at_3"]),
            "original_gtpa_at_5": bool(row["gtpa_at_5"]),
            "changed_prediction": normalize_label(prediction) != normalize_label(row["predicted_diagnosis"]),
        })

case_policy_frame = pd.DataFrame(case_policy_rows)
case_policy_frame.to_csv(ARTIFACT_ROOT / "case_level_policy_results.csv", index=False)

for (policy_name, dataset_name, budget), group in case_policy_frame.groupby(["policy_name", "dataset_name", "budget"], sort=True):
    policy_rows.append({
        "policy_name": policy_name,
        "dataset_name": dataset_name,
        "budget": int(budget),
        "num_workups": int(len(group)),
        "top1": float(group["correct_top1"].mean()),
        "top3": float(group["gtpa_at_3"].mean()),
        "top5": float(group["gtpa_at_5"].mean()),
        "wins_vs_current": int(((group["correct_top1"]) & (~group["original_correct_top1"])).sum()),
        "regressions_vs_current": int(((~group["correct_top1"]) & (group["original_correct_top1"])).sum()),
        "changed_predictions": int(group["changed_prediction"].sum()),
    })
policy_summary = pd.DataFrame(policy_rows)

overall_rows = []
for policy_name, group in case_policy_frame.groupby("policy_name", sort=True):
    overall_rows.append({
        "policy_name": policy_name,
        "dataset_name": "ALL",
        "budget": -1,
        "num_workups": int(len(group)),
        "top1": float(group["correct_top1"].mean()),
        "top3": float(group["gtpa_at_3"].mean()),
        "top5": float(group["gtpa_at_5"].mean()),
        "wins_vs_current": int(((group["correct_top1"]) & (~group["original_correct_top1"])).sum()),
        "regressions_vs_current": int(((~group["correct_top1"]) & (group["original_correct_top1"])).sum()),
        "changed_predictions": int(group["changed_prediction"].sum()),
    })
policy_summary = pd.concat([pd.DataFrame(overall_rows), policy_summary], ignore_index=True)
policy_summary.to_csv(ARTIFACT_ROOT / "policy_variant_summary.csv", index=False)
display(policy_summary[policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=False))

# %% [markdown]
# ## 4. Failure Taxonomy

# %%
def trace_visible_text(row: pd.Series) -> str:
    trace = trace_by_key.get((row["dataset_name"], row["case_id"], int(row["budget"])), {})
    parts = [str(trace.get("initial_patient_info", ""))]
    for turn in trace.get("turns", []):
        parts.append(str(turn.get("answer", "")))
    return "\n".join(parts)


def classify_failure(row: pd.Series, pool_hit: bool) -> str:
    if bool(row["correct_top1"]):
        return "correct"
    if not pool_hit:
        return "candidate_pool_miss_or_acquisition_failure"
    if (
        str(row["dataset_name"]) == "ddxplus"
        and str(row.get("ddxplus_mlp_top1", "")).strip()
        and normalize_label(row.get("ddxplus_mlp_top1", "")) == normalize_label(row["ground_truth_diagnosis"])
        and float(row.get("ddxplus_mlp_confidence", 0.0) or 0.0) >= DDXPLUS_MLP_GUARD_CONFIDENCE_MIN
        and float(row.get("ddxplus_mlp_margin", 0.0) or 0.0) >= DDXPLUS_MLP_GUARD_MARGIN_MIN
    ):
        return "resolver_regressed_high_confidence_ddxplus_mlp"
    if str(row["dataset_name"]) == "rarebench" and str(row.get("rarebench_gate_action", "")) == "locked_llm_graph_agreement":
        return "rarebench_wrong_llm_graph_lock"
    if str(row["dataset_name"]) == "icraft_md" and bool(row.get("gtpa_at_3", False)):
        return "small_option_topk_adjudication_failure"
    if not bool(row.get("gtpa_at_5", False)) and pool_hit:
        return "truth_in_broad_pool_but_not_final_top5"
    return "resolver_ranking_failure"


failure_rows = []
for _, row in predictions.iterrows():
    pool_match = pool_frame[
        (pool_frame["dataset_name"].astype(str) == str(row["dataset_name"]))
        & (pool_frame["case_id"].astype(str) == str(row["case_id"]))
        & (pool_frame["budget"].astype(int) == int(row["budget"]))
    ].iloc[0]
    failure_rows.append({
        "dataset_name": row["dataset_name"],
        "case_id": row["case_id"],
        "budget": int(row["budget"]),
        "ground_truth_diagnosis": row["ground_truth_diagnosis"],
        "prediction": row["predicted_diagnosis"],
        "correct_top1": bool(row["correct_top1"]),
        "gtpa_at_3": bool(row["gtpa_at_3"]),
        "gtpa_at_5": bool(row["gtpa_at_5"]),
        "candidate_pool_hit": bool(pool_match["candidate_pool_hit"]),
        "candidate_pool_rank": int(pool_match["candidate_pool_rank"]),
        "failure_type": classify_failure(row, bool(pool_match["candidate_pool_hit"])),
        "num_questions": int(row["num_questions"]),
        "stop_reason": row.get("stop_reason", ""),
        "ddxplus_mlp_top1": row.get("ddxplus_mlp_top1", ""),
        "ddxplus_mlp_confidence": float(row.get("ddxplus_mlp_confidence", 0.0) or 0.0),
        "ddxplus_mlp_margin": float(row.get("ddxplus_mlp_margin", 0.0) or 0.0),
        "rarebench_graph_top_label": row.get("rarebench_graph_top_label", ""),
        "rarebench_gate_action": row.get("rarebench_gate_action", ""),
    })
failure_frame = pd.DataFrame(failure_rows)
failure_frame.to_csv(ARTIFACT_ROOT / "failure_taxonomy.csv", index=False)
failure_summary = (
    failure_frame.groupby(["dataset_name", "failure_type"], as_index=False)
    .agg(num_workups=("case_id", "count"))
    .sort_values(["dataset_name", "num_workups"], ascending=[True, False])
)
failure_summary.to_csv(ARTIFACT_ROOT / "failure_taxonomy_summary.csv", index=False)
display(failure_summary)
display(failure_frame[~failure_frame["correct_top1"]].sort_values(["dataset_name", "case_id", "budget"]))

# %% [markdown]
# ## 5. Figures

# %%
overall = policy_summary[policy_summary["dataset_name"].eq("ALL")].sort_values("top1", ascending=True)
plt.figure(figsize=(8, 4))
plt.barh(overall["policy_name"], overall["top1"], color="#4C78A8")
plt.xlim(0, 1.0)
plt.xlabel("Top-1 accuracy")
plt.title("Notebook 46 Repair Policy Variants")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "policy_variant_top1.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 4))
for dataset_name, group in pool_summary.groupby("dataset_name"):
    plt.plot(group["budget"], group["candidate_pool_recall"], marker="o", label=f"{dataset_name} pool")
    plt.plot(group["budget"], group["current_top1"], marker="x", linestyle="--", label=f"{dataset_name} current")
plt.ylim(0, 1.05)
plt.xlabel("Budget")
plt.ylabel("Rate")
plt.title("Candidate-Pool Recall vs Current Top-1")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "candidate_pool_recall_vs_current.png", dpi=180)
plt.close()

wrong = failure_frame[~failure_frame["correct_top1"]].copy()
plt.figure(figsize=(8, 4))
counts = wrong["failure_type"].value_counts().sort_values()
plt.barh(counts.index, counts.values, color="#F58518")
plt.xlabel("Wrong workups")
plt.title("Failure Taxonomy")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "failure_taxonomy.png", dpi=180)
plt.close()

# %% [markdown]
# ## 6. Selected Repair Policy And Artifact Contract

# %%
selected_policy_name = "ddxplus_high_conf_mlp_guard_v1"
selected_overall = policy_summary[
    policy_summary["policy_name"].eq(selected_policy_name) & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
current_overall = policy_summary[
    policy_summary["policy_name"].eq("notebook46_current") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()
oracle_overall = policy_summary[
    policy_summary["policy_name"].eq("candidate_pool_oracle_diagnostic") & policy_summary["dataset_name"].eq("ALL")
].iloc[0].to_dict()

selected_policy = {
    "selected_policy_name": selected_policy_name,
    "status": "offline_candidate_for_next_live_notebook",
    "label_use_note": "Ground-truth labels are used for evaluation only. The selected rule uses only dataset name, DDXPlus MLP confidence, and DDXPlus MLP margin.",
    "input_run": INPUT_RUN_NAME,
    "input_root": str(INPUT_ROOT),
    "policy_rule": {
        "if": "dataset_name == 'ddxplus' and ddxplus_mlp_available and ddxplus_mlp_confidence >= threshold and ddxplus_mlp_margin >= threshold",
        "then": "protect the DDXPlus MLP top-1 as the final top-1 and move it to rank 1",
        "else": "keep Notebook 46 final prediction",
        "confidence_threshold": DDXPLUS_MLP_GUARD_CONFIDENCE_MIN,
        "margin_threshold": DDXPLUS_MLP_GUARD_MARGIN_MIN,
    },
    "current_overall": current_overall,
    "selected_overall": selected_overall,
    "candidate_pool_oracle_overall": oracle_overall,
    "candidate_pool_recall": {
        "overall": float(pool_frame["candidate_pool_hit"].mean()),
        "hits": int(pool_frame["candidate_pool_hit"].sum()),
        "num_workups": int(len(pool_frame)),
    },
    "interpretation": [
        "Notebook 46 already has a strong broad candidate pool: 88/90 workups contain the true diagnosis.",
        "The selected no-API repair fixes high-confidence DDXPlus MLP regressions but cannot fully solve the run.",
        "The remaining gap is resolver discrimination over a mostly-correct candidate pool, not simply a need for more evidence.",
        "A final live notebook should add this DDXPlus guard plus a budget-aware candidate-pool adjudicator for flagged non-guarded cases.",
    ],
}
write_json(ARTIFACT_ROOT / "selected_repair_policy.json", selected_policy)

resolved_run_config = {
    "run_name": RUN_NAME,
    "input_run_name": INPUT_RUN_NAME,
    "input_root": str(INPUT_ROOT),
    "offline_only": True,
    "api_calls": 0,
    "policies_evaluated": policy_names,
    "selected_policy": selected_policy_name,
    "ddxplus_mlp_guard_confidence_min": DDXPLUS_MLP_GUARD_CONFIDENCE_MIN,
    "ddxplus_mlp_guard_margin_min": DDXPLUS_MLP_GUARD_MARGIN_MIN,
    "required_outputs": [
        "resolved_run_config.json",
        "candidate_pool_coverage.csv",
        "candidate_pool_summary.csv",
        "candidate_pool_long.csv",
        "case_level_policy_results.csv",
        "policy_variant_summary.csv",
        "failure_taxonomy.csv",
        "failure_taxonomy_summary.csv",
        "selected_repair_policy.json",
    ],
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

required_outputs = resolved_run_config["required_outputs"]
missing_outputs = [name for name in required_outputs if not (ARTIFACT_ROOT / name).exists()]
if missing_outputs:
    raise FileNotFoundError(f"Missing required Notebook 47 artifacts: {missing_outputs}")

print("Notebook 47 artifact contract OK")
print("Selected policy:", selected_policy_name)
print("Current top-1:", current_overall["top1"])
print("Selected top-1:", selected_overall["top1"])
print("Candidate-pool oracle top-1:", oracle_overall["top1"])
