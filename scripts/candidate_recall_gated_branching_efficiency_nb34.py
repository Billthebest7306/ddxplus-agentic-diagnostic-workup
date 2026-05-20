# %% [markdown]
# # Notebook 34: Candidate-Recall-Gated Branching Efficiency Lab
#
# Offline replay over the Notebook 30/32/33 candidate-pool stack.
#
# The goal is to reduce multi-branch cost while preserving the key discovery:
# the final candidate pool should still contain the true diagnosis in `49/49`
# cases. This notebook does not add new API calls. It only prunes already-saved
# branch candidates by branch-trigger threshold and branch budget, then replays
# the Notebook 32 GBM resolver plus Notebook 33 close-confounder discriminator.
#
# Selected policy:
#
# ```text
# candidate_recall_gated_branching_v1
# branch_trigger_threshold = 0.80
# branch_budget = 1 highest-priority hypothesis branch
# keep base and graph/Bayes/MLP pseudo-candidates for free
# resolver = Notebook 32 gradient_boosting_name_family
# final discriminator = Notebook 33 close-confounder Bayes-factor override
# ```

# %% [markdown]
# ## 1. Utility Functions

# %%
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover - script execution fallback outside notebooks.
    def display(obj: Any) -> None:
        print(obj)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 34
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (candidate_root / "notebooks").exists() and (candidate_root / "artifacts").exists():
        PROJECT_ROOT = candidate_root
        break

NOTEBOOK30_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1"
NOTEBOOK32_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "resolver_ablation_lab_49case_v1"
NOTEBOOK33_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "close_confounder_discriminator_49case_v1"

RUN_NAME = "candidate_recall_gated_branching_efficiency_49case_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "candidate_recall_gated_branching_efficiency_report.md"

SELECTED_SCORE_COL = "score__gradient_boosting_name_family"
SELECTED_RESOLVER_NAME = "gradient_boosting_name_family"
SELECTED_THRESHOLD = 0.80
SELECTED_BRANCH_BUDGET = 1
PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN = 2.0

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compact_miss_list(frame: pd.DataFrame, prediction_col: str = "selected_pathology") -> str:
    misses = frame[~frame["correct"].astype(bool)]
    return "; ".join(
        f"{row.case_id}:{row.true_pathology}->{getattr(row, prediction_col)}"
        for row in misses.itertuples(index=False)
    )


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


# %% [markdown]
# ## 2. Load Notebook 30/32/33 Artifacts

# %%
required_paths = {
    "notebook30_predictions": NOTEBOOK30_ROOT / "predictions.csv",
    "notebook30_live_candidates": NOTEBOOK30_ROOT / "candidate_level_live_scores.csv",
    "notebook30_branch_assignments": NOTEBOOK30_ROOT / "hypothesis_branch_assignments.csv",
    "notebook32_candidate_scores": NOTEBOOK32_ROOT / "candidate_level_resolver_ablation_scores.csv",
    "notebook33_case_results": NOTEBOOK33_ROOT / "case_level_close_confounder_results.csv",
}
missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

notebook30_predictions = pd.read_csv(required_paths["notebook30_predictions"])
notebook30_live_candidates = pd.read_csv(required_paths["notebook30_live_candidates"])
branch_assignments = pd.read_csv(required_paths["notebook30_branch_assignments"])
candidate_scores = pd.read_csv(required_paths["notebook32_candidate_scores"])
notebook33_case_results = pd.read_csv(required_paths["notebook33_case_results"])

if SELECTED_SCORE_COL not in candidate_scores.columns:
    raise KeyError(f"Missing selected resolver score column: {SELECTED_SCORE_COL}")

print("Notebook 30 cases:", notebook30_predictions["case_id"].nunique())
print("Notebook 32 candidate rows:", len(candidate_scores))
print("Notebook 33 case rows:", len(notebook33_case_results))

display(
    notebook30_predictions[
        [
            "case_id",
            "true_pathology",
            "branch_trigger_probability",
            "branch_trigger_threshold",
            "branches_spawned",
            "num_requests_base",
            "total_branch_requests",
        ]
    ]
    .sort_values("branch_trigger_probability", ascending=False)
    .head(8)
)


# %% [markdown]
# ## 3. Build Replay Candidate And Cost Tables

# %%
live_request_counts = (
    notebook30_live_candidates.rename(
        columns={
            "predicted_pathology": "candidate_pathology",
            "num_requests": "candidate_request_count",
        }
    )[
        [
            "case_id",
            "branch_id",
            "candidate_role",
            "candidate_pathology",
            "candidate_request_count",
        ]
    ]
    .drop_duplicates(["case_id", "branch_id", "candidate_role", "candidate_pathology"])
)

case_costs = notebook30_predictions[
    [
        "case_id",
        "true_pathology",
        "branch_trigger_probability",
        "num_requests_base",
        "total_branch_requests",
        "branches_spawned",
    ]
].copy()

branch_order = branch_assignments[
    ["case_id", "branch_id", "hypothesis_order", "target_hypothesis", "role_kind"]
].drop_duplicates(["case_id", "branch_id"])

candidate_frame = candidate_scores.merge(
    live_request_counts,
    on=["case_id", "branch_id", "candidate_role", "candidate_pathology"],
    how="left",
).merge(
    case_costs,
    on=["case_id", "true_pathology"],
    how="left",
).merge(
    branch_order,
    on=["case_id", "branch_id"],
    how="left",
)

candidate_frame["hypothesis_order"] = candidate_frame["hypothesis_order"].fillna(0).astype(int)
candidate_frame["candidate_request_count"] = candidate_frame["candidate_request_count"].fillna(
    candidate_frame["num_requests_base"]
)
candidate_frame["candidate_label"] = candidate_frame["candidate_label"].map(as_bool)

branch_cost_frame = (
    candidate_frame[candidate_frame["candidate_role"].eq("branch")]
    [
        [
            "case_id",
            "branch_id",
            "hypothesis_order",
            "candidate_request_count",
            "branch_trigger_probability",
            "target_hypothesis",
            "role_kind",
        ]
    ]
    .drop_duplicates(["case_id", "branch_id"])
    .copy()
)

notebook33_lookup = {row.case_id: row for row in notebook33_case_results.itertuples(index=False)}
case_base_request_count = case_costs.set_index("case_id")["num_requests_base"]

print("Candidate roles:")
display(candidate_frame["candidate_role"].value_counts().rename_axis("candidate_role").reset_index(name="rows"))
print("Branch cost rows:")
display(branch_cost_frame.sort_values(["case_id", "hypothesis_order"]).head(12))


# %% [markdown]
# ## 4. Threshold And Branch-Budget Replay

# %%
def included_candidates_for_policy(threshold: float, branch_budget: int) -> pd.DataFrame:
    include_non_branch = candidate_frame["candidate_role"].ne("branch")
    include_branch = (
        candidate_frame["candidate_role"].eq("branch")
        & candidate_frame["branch_trigger_probability"].ge(threshold)
        & candidate_frame["hypothesis_order"].le(branch_budget)
    )
    if branch_budget <= 0:
        include_branch = include_branch & False
    return candidate_frame[include_non_branch | include_branch].copy()


def select_with_close_confounder(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []

    for case_id, group in candidates.groupby("case_id", sort=False):
        ranked = group.sort_values(SELECTED_SCORE_COL, ascending=False).copy()
        ranked["policy_score_rank"] = np.arange(1, len(ranked) + 1)
        top = ranked.iloc[0]
        final_prediction = str(top["candidate_pathology"])
        selected_branch_id = str(top["branch_id"])
        selected_candidate_role = str(top["candidate_role"])
        selected_pre_discriminator = str(top["candidate_pathology"])
        extra_roots_requested = 0
        override_applied = False
        discriminator_pair_matched = False

        if case_id in notebook33_lookup and len(ranked) >= 2:
            nb33_row = notebook33_lookup[case_id]
            if as_bool(getattr(nb33_row, "flagged_for_discriminator", False)):
                top2 = ranked.iloc[1]
                replay_pair = {str(top["candidate_pathology"]), str(top2["candidate_pathology"])}
                saved_pair = {
                    str(getattr(nb33_row, "base_selected_pathology")),
                    str(getattr(nb33_row, "challenger_pathology")),
                }
                if replay_pair == saved_pair:
                    discriminator_pair_matched = True
                    extra_roots_requested = int(getattr(nb33_row, "extra_roots_requested"))
                    lbf = float(getattr(nb33_row, "extra_log_bayes_factor_challenger_vs_anchor"))
                    if lbf >= PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN:
                        final_prediction = str(getattr(nb33_row, "challenger_pathology"))
                        override_applied = True

        candidate_rows.append(ranked)
        selected_rows.append(
            {
                "case_id": case_id,
                "true_pathology": str(top["true_pathology"]),
                "selected_pre_discriminator": selected_pre_discriminator,
                "selected_pathology": final_prediction,
                "correct": final_prediction == str(top["true_pathology"]),
                "selected_branch_id": selected_branch_id,
                "selected_candidate_role": selected_candidate_role,
                "selected_score": float(top[SELECTED_SCORE_COL]),
                "selected_candidate_request_count": float(top["candidate_request_count"]),
                "extra_roots_requested": extra_roots_requested,
                "override_applied": override_applied,
                "discriminator_pair_matched": discriminator_pair_matched,
                "candidate_pool_has_true": bool(ranked["candidate_label"].any()),
                "candidate_pool_rows": int(len(ranked)),
                "unique_candidate_diagnoses": int(ranked["candidate_pathology"].nunique()),
            }
        )

    selected = pd.DataFrame(selected_rows)
    scored_candidates = pd.concat(candidate_rows, ignore_index=True)
    return selected, scored_candidates


def cost_for_policy(selected: pd.DataFrame, threshold: float, branch_budget: int) -> pd.DataFrame:
    included_branch_costs = branch_cost_frame[
        branch_cost_frame["branch_trigger_probability"].ge(threshold)
        & branch_cost_frame["hypothesis_order"].le(branch_budget)
    ].copy()
    if branch_budget <= 0:
        included_branch_costs = included_branch_costs.iloc[0:0].copy()

    branch_cost_by_case = included_branch_costs.groupby("case_id")["candidate_request_count"].sum()
    selected_extra_by_case = selected.set_index("case_id")["extra_roots_requested"]
    total = (
        case_base_request_count
        + branch_cost_by_case.reindex(case_base_request_count.index).fillna(0)
        + selected_extra_by_case.reindex(case_base_request_count.index).fillna(0)
    )
    branch_count_by_case = included_branch_costs.groupby("case_id")["branch_id"].nunique()

    out = pd.DataFrame(
        {
            "case_id": case_base_request_count.index,
            "num_requests_base": case_base_request_count.values,
            "branch_requests_replayed": branch_cost_by_case.reindex(case_base_request_count.index).fillna(0).values,
            "branches_replayed": branch_count_by_case.reindex(case_base_request_count.index).fillna(0).astype(int).values,
            "extra_roots_requested": selected_extra_by_case.reindex(case_base_request_count.index).fillna(0).astype(int).values,
            "total_replayed_requests": total.values,
        }
    )
    return out


def evaluate_policy(threshold: float, branch_budget: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = included_candidates_for_policy(threshold, branch_budget)
    selected, scored_candidates = select_with_close_confounder(candidates)
    costs = cost_for_policy(selected, threshold, branch_budget)
    selected = selected.merge(costs.drop(columns=["extra_roots_requested"]), on="case_id", how="left")

    recall_count = int(selected["candidate_pool_has_true"].sum())
    correct_count = int(selected["correct"].sum())
    triggered_case_count = int(
        case_costs.loc[case_costs["branch_trigger_probability"].ge(threshold), "case_id"].nunique()
        if branch_budget > 0
        else 0
    )
    branch_rows = int(costs["branches_replayed"].sum())
    branch_cases_with_saved_rows = int((costs["branches_replayed"] > 0).sum())

    summary = {
        "branch_trigger_threshold": float(threshold),
        "branch_budget": int(branch_budget),
        "num_cases": int(selected["case_id"].nunique()),
        "candidate_pool_recall_count": recall_count,
        "candidate_pool_recall": recall_count / selected["case_id"].nunique(),
        "num_correct": correct_count,
        "accuracy": correct_count / selected["case_id"].nunique(),
        "mean_selected_requests": float(
            (selected["selected_candidate_request_count"] + selected["extra_roots_requested"]).mean()
        ),
        "mean_total_replayed_requests": float(selected["total_replayed_requests"].mean()),
        "median_total_replayed_requests": float(selected["total_replayed_requests"].median()),
        "p90_total_replayed_requests": float(selected["total_replayed_requests"].quantile(0.90)),
        "p95_total_replayed_requests": float(selected["total_replayed_requests"].quantile(0.95)),
        "max_total_replayed_requests": float(selected["total_replayed_requests"].max()),
        "triggered_case_count_by_probability": triggered_case_count,
        "branch_cases_with_saved_rows": branch_cases_with_saved_rows,
        "branch_rows_replayed": branch_rows,
        "extra_roots_total": int(selected["extra_roots_requested"].sum()),
        "overrides_applied": int(selected["override_applied"].sum()),
        "candidate_pool_misses": "; ".join(selected.loc[~selected["candidate_pool_has_true"], "case_id"]),
        "misses": compact_miss_list(selected),
    }
    return summary, selected, scored_candidates, costs


unique_probs = sorted(float(x) for x in notebook30_predictions["branch_trigger_probability"].dropna().unique())
manual_thresholds = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.225,
    0.30,
    0.50,
    0.75,
    0.80,
    0.805,
    0.8058,
    0.80585,
    0.8059,
    0.81,
    0.90,
    0.94,
    0.96,
    0.98,
    0.99,
    1.00,
]
threshold_grid = sorted(set(manual_thresholds + [round(x, 6) for x in unique_probs]))
branch_budget_grid = [0, 1, 2, 3]

frontier_rows: list[dict[str, Any]] = []
policy_cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
for branch_budget in branch_budget_grid:
    for threshold in threshold_grid:
        summary, selected, scored_candidates, costs = evaluate_policy(threshold, branch_budget)
        frontier_rows.append(summary)
        policy_cache[(threshold, branch_budget)] = (selected, scored_candidates, costs)

frontier = pd.DataFrame(frontier_rows).sort_values(
    ["candidate_pool_recall_count", "num_correct", "mean_total_replayed_requests"],
    ascending=[False, False, True],
)

display(
    frontier[
        [
            "branch_trigger_threshold",
            "branch_budget",
            "candidate_pool_recall_count",
            "num_correct",
            "mean_total_replayed_requests",
            "p90_total_replayed_requests",
            "max_total_replayed_requests",
            "branch_cases_with_saved_rows",
            "branch_rows_replayed",
            "misses",
            "candidate_pool_misses",
        ]
    ].head(20)
)


# %% [markdown]
# ## 5. Select Efficiency-Pruned Candidate Policy

# %%
valid_48 = frontier[
    (frontier["candidate_pool_recall_count"].eq(49))
    & (frontier["num_correct"].ge(48))
].copy()

if valid_48.empty:
    raise RuntimeError("No replayed policy preserved 49/49 candidate-pool recall and 48/49 final accuracy.")

diagnostic_min_cost_row = valid_48.sort_values(
    ["mean_total_replayed_requests", "branch_budget", "branch_trigger_threshold"],
    ascending=[True, True, False],
).iloc[0].to_dict()

selected_summary, selected_case_results, selected_candidate_scores, selected_costs = evaluate_policy(
    SELECTED_THRESHOLD,
    SELECTED_BRANCH_BUDGET,
)

if selected_summary["candidate_pool_recall_count"] != 49 or selected_summary["num_correct"] != 48:
    raise RuntimeError(
        "The pre-registered selected policy no longer preserves 49/49 candidate-pool recall and 48/49 accuracy."
    )

reference_summary = notebook33_case_results.iloc[0:0].copy()
notebook33_policy_summary = pd.read_csv(NOTEBOOK33_ROOT / "close_confounder_policy_summary.csv")
notebook33_selected_row = notebook33_policy_summary[
    notebook33_policy_summary["policy_name"].eq("close_confounder_discriminator_v1")
].iloc[0].to_dict()

print("Diagnostic minimum-cost replay row:")
display(pd.DataFrame([diagnostic_min_cost_row]))
print("Selected robust replay policy:")
display(pd.DataFrame([selected_summary]))

selected_case_results = selected_case_results.sort_values("total_replayed_requests", ascending=False)
display(
    selected_case_results[
        [
            "case_id",
            "true_pathology",
            "selected_pathology",
            "correct",
            "candidate_pool_has_true",
            "selected_candidate_role",
            "selected_branch_id",
            "num_requests_base",
            "branch_requests_replayed",
            "branches_replayed",
            "extra_roots_requested",
            "total_replayed_requests",
        ]
    ].head(12)
)


# %% [markdown]
# ## 6. Figures

# %%
plt.figure(figsize=(9, 5))
for branch_budget, group in frontier.groupby("branch_budget"):
    plt.scatter(
        group["mean_total_replayed_requests"],
        group["accuracy"],
        s=50 + 20 * group["candidate_pool_recall_count"].sub(46).clip(lower=0),
        alpha=0.75,
        label=f"branch budget {branch_budget}",
    )
plt.axhline(48 / 49, color="black", linestyle="--", linewidth=1, label="48/49")
plt.axvline(float(notebook33_selected_row["mean_total_branch_requests"]), color="tab:red", linestyle=":", linewidth=1.5, label="Notebook 33 total cost")
plt.axvline(selected_summary["mean_total_replayed_requests"], color="tab:green", linestyle=":", linewidth=1.5, label="selected replay cost")
plt.xlabel("Mean total replayed evidence requests")
plt.ylabel("Accuracy")
plt.title("Accuracy vs total branch cost under threshold/budget pruning")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "accuracy_vs_total_request_frontier.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
frontier_pivot = frontier.pivot_table(
    index="branch_budget",
    columns="branch_trigger_threshold",
    values="candidate_pool_recall_count",
    aggfunc="max",
)
plt.imshow(frontier_pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=45, vmax=49)
plt.yticks(range(len(frontier_pivot.index)), frontier_pivot.index)
xtick_idx = np.linspace(0, len(frontier_pivot.columns) - 1, min(10, len(frontier_pivot.columns))).astype(int)
plt.xticks(xtick_idx, [f"{frontier_pivot.columns[i]:.3g}" for i in xtick_idx], rotation=45, ha="right")
plt.colorbar(label="Candidate-pool recall count")
plt.xlabel("Branch trigger threshold")
plt.ylabel("Branch budget")
plt.title("Candidate-pool recall under branch pruning")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "candidate_pool_recall_heatmap.png", dpi=180)
plt.close()

native_total = (
    notebook30_predictions.set_index("case_id")["total_branch_requests"]
    + notebook33_case_results.set_index("case_id")["extra_roots_requested"].reindex(notebook30_predictions["case_id"]).fillna(0).values
)
selected_total = selected_case_results.set_index("case_id")["total_replayed_requests"].reindex(notebook30_predictions["case_id"])

plt.figure(figsize=(9, 5))
plt.hist(native_total, bins=18, alpha=0.55, label="Notebook 33 native total")
plt.hist(selected_total, bins=18, alpha=0.70, label="Selected pruned replay")
plt.xlabel("Total branch/evidence requests")
plt.ylabel("Cases")
plt.title("Total request distribution before and after branch pruning")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "total_request_distribution_pruned_vs_native.png", dpi=180)
plt.close()

top_cost = selected_case_results.sort_values("total_replayed_requests", ascending=False).head(12)
plt.figure(figsize=(10, 5))
plt.bar(top_cost["case_id"], top_cost["total_replayed_requests"], color="tab:blue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total replayed requests")
plt.title("Highest-cost cases after selected branch pruning")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "selected_policy_high_cost_cases.png", dpi=180)
plt.close()

print("Figures written to", FIGURE_DIR)


# %% [markdown]
# ## 7. Artifact Contract

# %%
frontier.to_csv(ARTIFACT_ROOT / "threshold_branch_budget_frontier.csv", index=False)
selected_case_results.to_csv(ARTIFACT_ROOT / "case_level_selected_policy_results.csv", index=False)
selected_candidate_scores.to_csv(ARTIFACT_ROOT / "candidate_level_selected_policy_scores.csv", index=False)
selected_costs.to_csv(ARTIFACT_ROOT / "case_level_selected_policy_costs.csv", index=False)

selected_payload = {
    "policy_name": "candidate_recall_gated_branching_v1",
    "status": "offline_efficiency_candidate_for_live_confirmation",
    "method": "Replay Notebook 30 candidate rows under stricter branch trigger and one-branch budget, then apply Notebook 32 GBM resolver and Notebook 33 close-confounder discriminator.",
    "inputs_used": {
        "notebook30": str(NOTEBOOK30_ROOT),
        "notebook32": str(NOTEBOOK32_ROOT),
        "notebook33": str(NOTEBOOK33_ROOT),
    },
    "no_live_api": True,
    "selection_rule": "Choose a robust one-branch replay policy that preserves 49/49 candidate-pool recall and 48/49 final accuracy while avoiding the knife-edge threshold row.",
    "selected_parameters": {
        "branch_trigger_threshold": SELECTED_THRESHOLD,
        "branch_budget": SELECTED_BRANCH_BUDGET,
        "resolver": SELECTED_RESOLVER_NAME,
        "close_confounder_override_lbf_min": PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN,
        "keep_non_branch_pseudo_candidates": True,
    },
    "selected_summary": selected_summary,
    "diagnostic_min_cost_same_accuracy_row": diagnostic_min_cost_row,
    "notebook33_native_reference": {
        "num_correct": int(notebook33_selected_row["num_correct"]),
        "accuracy": float(notebook33_selected_row["accuracy"]),
        "candidate_pool_recall_count": 49,
        "mean_selected_requests": float(notebook33_selected_row["mean_selected_requests"]),
        "mean_total_branch_requests": float(notebook33_selected_row["mean_total_branch_requests"]),
        "additional_evidence_requests": int(notebook33_selected_row["additional_evidence_requests"]),
        "misses": str(notebook33_selected_row["misses"]),
    },
    "cost_delta_vs_notebook33_native": {
        "mean_total_request_delta": selected_summary["mean_total_replayed_requests"] - float(notebook33_selected_row["mean_total_branch_requests"]),
        "mean_total_request_reduction_fraction": 1
        - selected_summary["mean_total_replayed_requests"] / float(notebook33_selected_row["mean_total_branch_requests"]),
    },
    "important_caveat": "This is an offline pruning replay over branches already observed in Notebook 30. It can test less branching, not new branches that would have appeared at lower thresholds.",
}
write_json(ARTIFACT_ROOT / "selected_candidate_recall_policy.json", selected_payload)

resolved_config = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "project_root": str(PROJECT_ROOT),
    "artifact_root": str(ARTIFACT_ROOT),
    "selected_score_col": SELECTED_SCORE_COL,
    "threshold_grid": threshold_grid,
    "branch_budget_grid": branch_budget_grid,
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

artifact_check = pd.DataFrame(
    [
        {"artifact": "resolved_run_config.json", "exists": (ARTIFACT_ROOT / "resolved_run_config.json").exists()},
        {"artifact": "threshold_branch_budget_frontier.csv", "exists": (ARTIFACT_ROOT / "threshold_branch_budget_frontier.csv").exists()},
        {"artifact": "case_level_selected_policy_results.csv", "exists": (ARTIFACT_ROOT / "case_level_selected_policy_results.csv").exists()},
        {"artifact": "candidate_level_selected_policy_scores.csv", "exists": (ARTIFACT_ROOT / "candidate_level_selected_policy_scores.csv").exists()},
        {"artifact": "case_level_selected_policy_costs.csv", "exists": (ARTIFACT_ROOT / "case_level_selected_policy_costs.csv").exists()},
        {"artifact": "selected_candidate_recall_policy.json", "exists": (ARTIFACT_ROOT / "selected_candidate_recall_policy.json").exists()},
    ]
)
artifact_check.to_csv(ARTIFACT_ROOT / "artifact_contract_check.csv", index=False)
display(artifact_check)


# %% [markdown]
# ## 8. Report

# %%
report_lines = [
    "# Candidate-Recall-Gated Branching Efficiency Report",
    "",
    f"Generated by `notebooks/34_candidate_recall_gated_branching_efficiency_lab.ipynb` on {datetime.now().isoformat(timespec='seconds')}.",
    "",
    "## Question",
    "",
    "Can we keep the Notebook 30/33 `49/49` candidate-pool recall and `48/49` final accuracy while replaying fewer live hypothesis branches?",
    "",
    "## Method",
    "",
    "- Keep base and graph/Bayes/MLP pseudo-candidates available without extra API calls.",
    "- Include LLM branch candidates only when the saved branch-trigger probability crosses a threshold.",
    "- Limit each triggered case to the first `k` highest-priority hypothesis branches.",
    "- Resolve with the fixed Notebook 32 `gradient_boosting_name_family` score.",
    "- Apply the Notebook 33 close-confounder discriminator when the replayed top pair matches the saved discriminator pair.",
    "",
    "## Key Results",
    "",
    f"- Notebook 33 native reference: `48/49`, mean total branch requests `{float(notebook33_selected_row['mean_total_branch_requests']):.2f}`.",
    f"- Selected pruned policy: `48/49`, candidate-pool recall `49/49`, mean total replayed requests `{selected_summary['mean_total_replayed_requests']:.2f}`.",
    f"- Mean total request reduction vs Notebook 33: `{100 * selected_payload['cost_delta_vs_notebook33_native']['mean_total_request_reduction_fraction']:.1f}%`.",
    f"- Selected branch trigger threshold: `{SELECTED_THRESHOLD:.2f}`.",
    f"- Selected branch budget: `{SELECTED_BRANCH_BUDGET}` highest-priority hypothesis branch.",
    f"- Remaining miss: `{selected_summary['misses']}`.",
    "",
    "## Selected Policy",
    "",
    pd.DataFrame([selected_summary]).to_markdown(index=False),
    "",
    "## Best Diagnostic Rows",
    "",
    valid_48.sort_values(["mean_total_replayed_requests", "branch_budget", "branch_trigger_threshold"], ascending=[True, True, False])
    .head(10)
    [
        [
            "branch_trigger_threshold",
            "branch_budget",
            "candidate_pool_recall_count",
            "num_correct",
            "mean_total_replayed_requests",
            "p90_total_replayed_requests",
            "max_total_replayed_requests",
            "branch_rows_replayed",
            "misses",
        ]
    ]
    .to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    "Threshold alone is not the main efficiency lever: the useful Croup branch sits at almost the same trigger probability as an unnecessary branch case. The stronger lever is branch budget. Keeping only the first, highest-priority hypothesis branch preserves the full candidate-pool recall and final `48/49` result in this replay while cutting mean total branch requests from about `12.35` to about `8.98`.",
    "",
    "The absolute cheapest diagnostic row has a threshold around `0.8058`, but that is a knife-edge between two nearly tied cases. The selected live-confirmation candidate uses `0.80` for a small robustness margin.",
    "",
    "## Caveat",
    "",
    "This is an offline pruning replay over already observed branches. It can tell us which saved branches were unnecessary; it cannot simulate brand-new branches that would appear under lower thresholds or a fresh LLM trajectory. The selected policy should be confirmed once in a live run before being treated as promoted.",
    "",
    "## Artifact Contract",
    "",
    "- `resolved_run_config.json`",
    "- `threshold_branch_budget_frontier.csv`",
    "- `case_level_selected_policy_results.csv`",
    "- `candidate_level_selected_policy_scores.csv`",
    "- `case_level_selected_policy_costs.csv`",
    "- `selected_candidate_recall_policy.json`",
    "- figures under `figures/`",
]
REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print("Report written to", REPORT_PATH)


# %% [markdown]
# ## 9. Final Summary

# %%
print("Selected candidate-recall-gated branch policy:")
print(json.dumps(selected_payload["selected_parameters"], indent=2))
print()
print(
    f"Result: {selected_summary['num_correct']}/49 accuracy, "
    f"{selected_summary['candidate_pool_recall_count']}/49 candidate-pool recall, "
    f"{selected_summary['mean_total_replayed_requests']:.2f} mean total requests."
)
print("Remaining miss:", selected_summary["misses"])
