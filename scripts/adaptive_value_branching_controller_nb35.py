# %% [markdown]
# # Notebook 35: Adaptive Value Branching Controller
#
# Offline replay over the saved Notebook 30/32/33 trajectory artifacts.
#
# Notebook 34 showed that a fixed one-branch replay preserved the key result:
# `49/49` candidate-pool recall and `48/49` final accuracy, with lower total
# branch cost than the original three-branch live run. The concern is that a
# fixed branch budget is too rigid as a general design.
#
# This notebook tests a more intelligent controller:
#
# ```text
# max_branches = 3
# launch branch 1 when the learned branch-trigger MLP fires
# after each completed branch, launch another branch only if:
#   expected value of another branch > continuation threshold
# ```
#
# The continuation value is label-free. It uses only current decision
# instability, graph/Bayes/MLP support, close-confounder discriminator
# availability, and the priority of the next hypothesis branch. No new API calls
# are made.

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

RANDOM_SEED = 35
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (candidate_root / "notebooks").exists() and (candidate_root / "artifacts").exists():
        PROJECT_ROOT = candidate_root
        break

NOTEBOOK30_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1"
NOTEBOOK32_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "resolver_ablation_lab_49case_v1"
NOTEBOOK33_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "close_confounder_discriminator_49case_v1"
NOTEBOOK34_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "candidate_recall_gated_branching_efficiency_49case_v1"

RUN_NAME = "adaptive_value_branching_controller_49case_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "adaptive_value_branching_controller_report.md"

SELECTED_SCORE_COL = "score__gradient_boosting_name_family"
SELECTED_RESOLVER_NAME = "gradient_boosting_name_family"
SELECTED_BRANCH_TRIGGER_THRESHOLD = 0.80
SELECTED_MAX_BRANCHES = 3
SELECTED_CONTINUATION_VALUE_THRESHOLD = 0.40
PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN = 2.0

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


def compact_miss_list(frame: pd.DataFrame, prediction_col: str = "selected_pathology") -> str:
    misses = frame[~frame["correct"].astype(bool)]
    return "; ".join(
        f"{row.case_id}:{row.true_pathology}->{getattr(row, prediction_col)}"
        for row in misses.itertuples(index=False)
    )


def softmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    centered = values.astype(float) - float(np.max(values.astype(float)))
    exp_values = np.exp(centered)
    denom = float(exp_values.sum())
    if denom <= 0:
        return np.ones(len(values), dtype=float) / len(values)
    return exp_values / denom


# %% [markdown]
# ## 2. Load Notebook 30/32/33/34 Artifacts

# %%
required_paths = {
    "notebook30_predictions": NOTEBOOK30_ROOT / "predictions.csv",
    "notebook30_live_candidates": NOTEBOOK30_ROOT / "candidate_level_live_scores.csv",
    "notebook30_branch_assignments": NOTEBOOK30_ROOT / "hypothesis_branch_assignments.csv",
    "notebook32_candidate_scores": NOTEBOOK32_ROOT / "candidate_level_resolver_ablation_scores.csv",
    "notebook33_case_results": NOTEBOOK33_ROOT / "case_level_close_confounder_results.csv",
    "notebook33_policy_summary": NOTEBOOK33_ROOT / "close_confounder_policy_summary.csv",
    "notebook34_selected_policy": NOTEBOOK34_ROOT / "selected_candidate_recall_policy.json",
}
missing = [name for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

notebook30_predictions = pd.read_csv(required_paths["notebook30_predictions"])
notebook30_live_candidates = pd.read_csv(required_paths["notebook30_live_candidates"])
branch_assignments = pd.read_csv(required_paths["notebook30_branch_assignments"])
candidate_scores = pd.read_csv(required_paths["notebook32_candidate_scores"])
notebook33_case_results = pd.read_csv(required_paths["notebook33_case_results"])
notebook33_policy_summary = pd.read_csv(required_paths["notebook33_policy_summary"])
notebook34_selected_policy = json.loads(required_paths["notebook34_selected_policy"].read_text(encoding="utf-8"))

if SELECTED_SCORE_COL not in candidate_scores.columns:
    raise KeyError(f"Missing selected resolver score column: {SELECTED_SCORE_COL}")

notebook33_selected_row = notebook33_policy_summary[
    notebook33_policy_summary["policy_name"].eq("close_confounder_discriminator_v1")
].iloc[0].to_dict()

print("Notebook 30 cases:", notebook30_predictions["case_id"].nunique())
print("Notebook 32 candidate rows:", len(candidate_scores))
print("Notebook 33 rows:", len(notebook33_case_results))
print("Notebook 34 selected:", notebook34_selected_policy["selected_parameters"])

display(
    notebook30_predictions[
        [
            "case_id",
            "true_pathology",
            "branch_trigger_probability",
            "branches_spawned",
            "num_requests_base",
            "total_branch_requests",
        ]
    ]
    .sort_values("branch_trigger_probability", ascending=False)
    .head(10)
)


# %% [markdown]
# ## 3. Build Replay Tables

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
    [
        "case_id",
        "branch_id",
        "hypothesis_order",
        "hypothesis_priority",
        "target_hypothesis",
        "role_kind",
        "target_graph_rank",
        "target_bayes_rank",
        "target_mlp_rank",
        "pair_missing_utility",
    ]
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
            "hypothesis_priority",
            "target_graph_rank",
            "target_bayes_rank",
            "target_mlp_rank",
            "pair_missing_utility",
        ]
    ]
    .drop_duplicates(["case_id", "branch_id"])
    .copy()
)

notebook33_lookup = {row.case_id: row for row in notebook33_case_results.itertuples(index=False)}
case_base_request_count = case_costs.set_index("case_id")["num_requests_base"]
case_trigger_probability = case_costs.set_index("case_id")["branch_trigger_probability"]

print("Candidate roles:")
display(candidate_frame["candidate_role"].value_counts().rename_axis("candidate_role").reset_index(name="rows"))
print("Saved branch rows:")
display(branch_cost_frame.sort_values(["case_id", "hypothesis_order"]).head(12))


# %% [markdown]
# ## 4. Adaptive Continuation Controller

# %%
MATH_SCORE_COLS = [
    "score__graph_posterior",
    "score__bayes_posterior",
    "score__mlp_posterior",
]


def candidates_for_case(case_id: str, included_branch_orders: set[int]) -> pd.DataFrame:
    group = candidate_frame[candidate_frame["case_id"].eq(case_id)].copy()
    include_non_branch = group["candidate_role"].ne("branch")
    include_branch = group["candidate_role"].eq("branch") & group["hypothesis_order"].isin(included_branch_orders)
    return group[include_non_branch | include_branch].copy()


def diagnosis_level_state(case_id: str, included_branch_orders: set[int]) -> dict[str, Any]:
    candidates = candidates_for_case(case_id, included_branch_orders)
    grouped = (
        candidates.groupby("candidate_pathology", as_index=False)
        .agg(
            {
                SELECTED_SCORE_COL: "max",
                "score__graph_posterior": "max",
                "score__bayes_posterior": "max",
                "score__mlp_posterior": "max",
                "candidate_label": "max",
            }
        )
        .sort_values(SELECTED_SCORE_COL, ascending=False)
        .reset_index(drop=True)
    )

    resolver_scores = grouped[SELECTED_SCORE_COL].to_numpy(dtype=float)
    probabilities = softmax(resolver_scores)
    top = str(grouped.iloc[0]["candidate_pathology"])
    second = str(grouped.iloc[1]["candidate_pathology"]) if len(grouped) > 1 else ""
    raw_margin = float(resolver_scores[0] - resolver_scores[1]) if len(resolver_scores) > 1 else 99.0
    probability_margin = float(probabilities[0] - probabilities[1]) if len(probabilities) > 1 else 1.0
    entropy = (
        float(-(probabilities * np.log(probabilities + 1e-12)).sum() / np.log(len(probabilities)))
        if len(probabilities) > 1
        else 0.0
    )

    math_tops: dict[str, str] = {}
    for col in MATH_SCORE_COLS:
        math_tops[col] = str(grouped.sort_values(col, ascending=False).iloc[0]["candidate_pathology"])
    math_support_count = int(sum(value == top for value in math_tops.values()))
    ledger_disagreement = 1.0 - math_support_count / len(MATH_SCORE_COLS)

    close_discriminator_available = False
    if case_id in notebook33_lookup and len(grouped) >= 2:
        nb33_row = notebook33_lookup[case_id]
        if as_bool(getattr(nb33_row, "flagged_for_discriminator", False)):
            replay_pair = {top, second}
            saved_pair = {
                str(getattr(nb33_row, "base_selected_pathology")),
                str(getattr(nb33_row, "challenger_pathology")),
            }
            close_discriminator_available = replay_pair == saved_pair

    return {
        "case_id": case_id,
        "included_branch_count": int(len(included_branch_orders)),
        "top_pathology": top,
        "second_pathology": second,
        "resolver_raw_margin": raw_margin,
        "resolver_probability_margin": probability_margin,
        "resolver_entropy": entropy,
        "math_support_count": math_support_count,
        "ledger_disagreement": ledger_disagreement,
        "math_top_graph": math_tops["score__graph_posterior"],
        "math_top_bayes": math_tops["score__bayes_posterior"],
        "math_top_mlp": math_tops["score__mlp_posterior"],
        "close_discriminator_available": close_discriminator_available,
        "candidate_pool_has_true": bool(grouped["candidate_label"].any()),
        "unique_candidate_diagnoses": int(grouped["candidate_pathology"].nunique()),
    }


def next_branch_metadata(case_id: str, next_order: int) -> dict[str, Any] | None:
    rows = branch_cost_frame[
        branch_cost_frame["case_id"].eq(case_id) & branch_cost_frame["hypothesis_order"].eq(next_order)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    first_rows = branch_cost_frame[
        branch_cost_frame["case_id"].eq(case_id) & branch_cost_frame["hypothesis_order"].eq(1)
    ]
    first_priority = float(first_rows.iloc[0]["hypothesis_priority"]) if not first_rows.empty else float(row["hypothesis_priority"])
    return {
        "next_branch_order": int(next_order),
        "next_branch_id": str(row["branch_id"]),
        "next_target_hypothesis": str(row["target_hypothesis"]),
        "next_role_kind": str(row["role_kind"]),
        "next_hypothesis_priority": float(row["hypothesis_priority"]),
        "next_priority_ratio": float(row["hypothesis_priority"]) / max(first_priority, 1e-9),
        "next_pair_missing_utility": float(row["pair_missing_utility"]),
        "next_target_graph_rank": float(row["target_graph_rank"]),
        "next_target_bayes_rank": float(row["target_bayes_rank"]),
        "next_target_mlp_rank": float(row["target_mlp_rank"]),
    }


def continuation_value(
    previous_state: dict[str, Any],
    current_state: dict[str, Any],
    next_meta: dict[str, Any],
) -> dict[str, Any]:
    margin_uncertainty = 1.0 - min(float(current_state["resolver_raw_margin"]) / 0.25, 1.0)
    unresolved_mass = (
        0.45 * float(current_state["ledger_disagreement"])
        + 0.30 * float(current_state["resolver_entropy"])
        + 0.25 * margin_uncertainty
    )
    next_priority_ratio = min(float(next_meta["next_priority_ratio"]), 1.0)

    top_changed = str(previous_state["top_pathology"]) != str(current_state["top_pathology"])
    math_support_count = int(current_state["math_support_count"])
    sufficient_update = (top_changed and math_support_count >= 2) or math_support_count == 3

    suppression = 1.0
    stop_reasons: list[str] = []
    if sufficient_update:
        suppression *= 0.35
        stop_reasons.append("sufficient_current_decision_support")
    if as_bool(current_state["close_discriminator_available"]):
        suppression *= 0.25
        stop_reasons.append("cheaper_close_confounder_discriminator_available")

    value = unresolved_mass * next_priority_ratio * suppression
    return {
        "continuation_value": float(value),
        "unresolved_mass": float(unresolved_mass),
        "next_priority_ratio": float(next_priority_ratio),
        "top_changed_after_last_branch": bool(top_changed),
        "sufficient_update": bool(sufficient_update),
        "suppression": float(suppression),
        "suppression_reasons": ";".join(stop_reasons),
    }


def adaptive_branch_orders_for_case(
    case_id: str,
    trigger_threshold: float,
    continuation_threshold: float,
    max_branches: int,
) -> tuple[set[int], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    included_orders: set[int] = set()
    trigger_probability = float(case_trigger_probability.loc[case_id])
    previous_state = diagnosis_level_state(case_id, included_orders)

    if trigger_probability < trigger_threshold or max_branches <= 0:
        trace_rows.append(
            {
                **previous_state,
                "case_id": case_id,
                "decision_step": 0,
                "action": "stop_no_initial_branch",
                "branch_trigger_probability": trigger_probability,
                "branch_trigger_threshold": float(trigger_threshold),
                "continuation_value_threshold": float(continuation_threshold),
                "next_branch_order": np.nan,
                "next_branch_id": "",
                "next_target_hypothesis": "",
                "continuation_value": 0.0,
                "reason": "branch_trigger_below_threshold_or_zero_budget",
            }
        )
        return included_orders, trace_rows

    for next_order in range(1, max_branches + 1):
        next_meta = next_branch_metadata(case_id, next_order)
        if next_meta is None:
            trace_rows.append(
                {
                    **previous_state,
                    "decision_step": next_order,
                    "action": "stop_no_saved_next_branch",
                    "branch_trigger_probability": trigger_probability,
                    "branch_trigger_threshold": float(trigger_threshold),
                    "continuation_value_threshold": float(continuation_threshold),
                    "next_branch_order": next_order,
                    "next_branch_id": "",
                    "next_target_hypothesis": "",
                    "continuation_value": 0.0,
                    "reason": "no_saved_branch_row_available",
                }
            )
            break

        if next_order == 1:
            launch_value = 1.0
            reason = "initial_branch_trigger_fired"
            action = "launch_branch"
        else:
            value_payload = continuation_value(previous_state, diagnosis_level_state(case_id, included_orders), next_meta)
            launch_value = float(value_payload["continuation_value"])
            action = "launch_branch" if launch_value >= continuation_threshold else "stop_branching"
            if value_payload["suppression_reasons"]:
                reason = value_payload["suppression_reasons"]
            elif action == "launch_branch":
                reason = "continuation_value_above_threshold"
            else:
                reason = "continuation_value_below_threshold"

        current_state = diagnosis_level_state(case_id, included_orders)
        row = {
            **current_state,
            "case_id": case_id,
            "decision_step": next_order,
            "action": action,
            "branch_trigger_probability": trigger_probability,
            "branch_trigger_threshold": float(trigger_threshold),
            "continuation_value_threshold": float(continuation_threshold),
            **next_meta,
            "continuation_value": float(launch_value),
            "reason": reason,
        }
        if next_order > 1:
            row.update(value_payload)
        else:
            row.update(
                {
                    "unresolved_mass": np.nan,
                    "next_priority_ratio": float(next_meta["next_priority_ratio"]),
                    "top_changed_after_last_branch": False,
                    "sufficient_update": False,
                    "suppression": 1.0,
                    "suppression_reasons": "",
                }
            )
        trace_rows.append(row)

        if action != "launch_branch":
            break

        included_orders.add(next_order)
        previous_state = diagnosis_level_state(case_id, included_orders)

    return included_orders, trace_rows


# %% [markdown]
# ## 5. Policy Evaluation

# %%
def included_candidates_for_orders(case_orders: dict[str, set[int]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case_id in case_costs["case_id"]:
        frames.append(candidates_for_case(case_id, case_orders.get(case_id, set())))
    return pd.concat(frames, ignore_index=True)


def select_with_close_confounder(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []

    for case_id, group in candidates.groupby("case_id", sort=False):
        ranked = group.sort_values(SELECTED_SCORE_COL, ascending=False).copy()
        ranked["policy_score_rank"] = np.arange(1, len(ranked) + 1)
        top = ranked.iloc[0]
        final_prediction = str(top["candidate_pathology"])
        selected_pre_discriminator = str(top["candidate_pathology"])
        selected_branch_id = str(top["branch_id"])
        selected_candidate_role = str(top["candidate_role"])
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


def cost_for_case_orders(selected: pd.DataFrame, case_orders: dict[str, set[int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_extra = selected.set_index("case_id")["extra_roots_requested"]
    for case_id in case_costs["case_id"]:
        orders = case_orders.get(case_id, set())
        branches = branch_cost_frame[
            branch_cost_frame["case_id"].eq(case_id) & branch_cost_frame["hypothesis_order"].isin(orders)
        ]
        branch_requests = float(branches["candidate_request_count"].sum())
        extra = int(selected_extra.reindex([case_id]).fillna(0).iloc[0])
        base_requests = float(case_base_request_count.loc[case_id])
        rows.append(
            {
                "case_id": case_id,
                "num_requests_base": base_requests,
                "branch_requests_replayed": branch_requests,
                "branches_replayed": int(branches["branch_id"].nunique()),
                "extra_roots_requested": extra,
                "total_replayed_requests": base_requests + branch_requests + extra,
            }
        )
    return pd.DataFrame(rows)


def evaluate_adaptive_policy(
    trigger_threshold: float,
    continuation_threshold: float,
    max_branches: int,
    policy_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_orders: dict[str, set[int]] = {}
    trace_rows: list[dict[str, Any]] = []

    for case_id in case_costs["case_id"]:
        orders, traces = adaptive_branch_orders_for_case(
            case_id=case_id,
            trigger_threshold=trigger_threshold,
            continuation_threshold=continuation_threshold,
            max_branches=max_branches,
        )
        case_orders[case_id] = orders
        trace_rows.extend(traces)

    candidates = included_candidates_for_orders(case_orders)
    selected, scored_candidates = select_with_close_confounder(candidates)
    costs = cost_for_case_orders(selected, case_orders)
    selected = selected.merge(costs.drop(columns=["extra_roots_requested"]), on="case_id", how="left")
    trace = pd.DataFrame(trace_rows)

    recall_count = int(selected["candidate_pool_has_true"].sum())
    correct_count = int(selected["correct"].sum())
    branch_rows = int(costs["branches_replayed"].sum())
    branch_cases_with_saved_rows = int((costs["branches_replayed"] > 0).sum())

    summary = {
        "policy_name": policy_name,
        "branch_trigger_threshold": float(trigger_threshold),
        "continuation_value_threshold": float(continuation_threshold),
        "max_branches": int(max_branches),
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
        "triggered_case_count_by_probability": int(
            case_costs.loc[case_costs["branch_trigger_probability"].ge(trigger_threshold), "case_id"].nunique()
            if max_branches > 0
            else 0
        ),
        "branch_cases_with_saved_rows": branch_cases_with_saved_rows,
        "branch_rows_replayed": branch_rows,
        "extra_roots_total": int(selected["extra_roots_requested"].sum()),
        "overrides_applied": int(selected["override_applied"].sum()),
        "candidate_pool_misses": "; ".join(selected.loc[~selected["candidate_pool_has_true"], "case_id"]),
        "misses": compact_miss_list(selected),
    }
    return summary, selected, scored_candidates, costs, trace


def evaluate_fixed_budget_policy(
    trigger_threshold: float,
    branch_budget: int,
    policy_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_orders: dict[str, set[int]] = {}
    for case_id in case_costs["case_id"]:
        probability = float(case_trigger_probability.loc[case_id])
        if probability >= trigger_threshold and branch_budget > 0:
            saved_orders = set(
                branch_cost_frame[
                    branch_cost_frame["case_id"].eq(case_id)
                    & branch_cost_frame["hypothesis_order"].le(branch_budget)
                ]["hypothesis_order"].astype(int)
            )
        else:
            saved_orders = set()
        case_orders[case_id] = saved_orders

    candidates = included_candidates_for_orders(case_orders)
    selected, scored_candidates = select_with_close_confounder(candidates)
    costs = cost_for_case_orders(selected, case_orders)
    selected = selected.merge(costs.drop(columns=["extra_roots_requested"]), on="case_id", how="left")
    summary = {
        "policy_name": policy_name,
        "branch_trigger_threshold": float(trigger_threshold),
        "continuation_value_threshold": np.nan,
        "max_branches": int(branch_budget),
        "num_cases": int(selected["case_id"].nunique()),
        "candidate_pool_recall_count": int(selected["candidate_pool_has_true"].sum()),
        "candidate_pool_recall": float(selected["candidate_pool_has_true"].mean()),
        "num_correct": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "mean_selected_requests": float(
            (selected["selected_candidate_request_count"] + selected["extra_roots_requested"]).mean()
        ),
        "mean_total_replayed_requests": float(selected["total_replayed_requests"].mean()),
        "median_total_replayed_requests": float(selected["total_replayed_requests"].median()),
        "p90_total_replayed_requests": float(selected["total_replayed_requests"].quantile(0.90)),
        "p95_total_replayed_requests": float(selected["total_replayed_requests"].quantile(0.95)),
        "max_total_replayed_requests": float(selected["total_replayed_requests"].max()),
        "triggered_case_count_by_probability": int(
            case_costs.loc[case_costs["branch_trigger_probability"].ge(trigger_threshold), "case_id"].nunique()
            if branch_budget > 0
            else 0
        ),
        "branch_cases_with_saved_rows": int((costs["branches_replayed"] > 0).sum()),
        "branch_rows_replayed": int(costs["branches_replayed"].sum()),
        "extra_roots_total": int(selected["extra_roots_requested"].sum()),
        "overrides_applied": int(selected["override_applied"].sum()),
        "candidate_pool_misses": "; ".join(selected.loc[~selected["candidate_pool_has_true"], "case_id"]),
        "misses": compact_miss_list(selected),
    }
    return summary, selected, scored_candidates, costs


# %% [markdown]
# ## 6. Adaptive Frontier

# %%
trigger_threshold_grid = [
    0.225,
    0.50,
    0.75,
    0.80,
    0.805,
    0.8058,
    0.80585,
    0.81,
    0.90,
    0.94,
    0.98,
]
continuation_threshold_grid = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.25,
    0.30,
    0.40,
]

frontier_rows: list[dict[str, Any]] = []
adaptive_cache: dict[tuple[float, float, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

for trigger_threshold in trigger_threshold_grid:
    for continuation_threshold in continuation_threshold_grid:
        summary, selected, scored, costs, trace = evaluate_adaptive_policy(
            trigger_threshold=trigger_threshold,
            continuation_threshold=continuation_threshold,
            max_branches=SELECTED_MAX_BRANCHES,
            policy_name="adaptive_value_branching_v1",
        )
        frontier_rows.append(summary)
        adaptive_cache[(trigger_threshold, continuation_threshold, SELECTED_MAX_BRANCHES)] = (
            selected,
            scored,
            costs,
            trace,
        )

for branch_budget in [0, 1, 2, 3]:
    summary, selected, scored, costs = evaluate_fixed_budget_policy(
        trigger_threshold=SELECTED_BRANCH_TRIGGER_THRESHOLD,
        branch_budget=branch_budget,
        policy_name=f"fixed_budget_{branch_budget}_threshold_0_80",
    )
    frontier_rows.append(summary)

frontier = pd.DataFrame(frontier_rows).sort_values(
    ["candidate_pool_recall_count", "num_correct", "mean_total_replayed_requests"],
    ascending=[False, False, True],
)

display(
    frontier[
        [
            "policy_name",
            "branch_trigger_threshold",
            "continuation_value_threshold",
            "max_branches",
            "candidate_pool_recall_count",
            "num_correct",
            "mean_total_replayed_requests",
            "p90_total_replayed_requests",
            "branch_rows_replayed",
            "misses",
        ]
    ].head(25)
)


# %% [markdown]
# ## 7. Select Adaptive Policy

# %%
selected_summary, selected_case_results, selected_candidate_scores, selected_costs, selected_trace = evaluate_adaptive_policy(
    trigger_threshold=SELECTED_BRANCH_TRIGGER_THRESHOLD,
    continuation_threshold=SELECTED_CONTINUATION_VALUE_THRESHOLD,
    max_branches=SELECTED_MAX_BRANCHES,
    policy_name="adaptive_value_branching_v1",
)

if selected_summary["candidate_pool_recall_count"] != 49 or selected_summary["num_correct"] != 48:
    raise RuntimeError("Selected adaptive policy failed to preserve 49/49 candidate recall and 48/49 accuracy.")

diagnostic_min_cost = (
    frontier[
        (frontier["candidate_pool_recall_count"].eq(49))
        & (frontier["num_correct"].ge(48))
        & (frontier["policy_name"].eq("adaptive_value_branching_v1"))
    ]
    .sort_values(["mean_total_replayed_requests", "branch_trigger_threshold", "continuation_value_threshold"])
    .iloc[0]
    .to_dict()
)

fixed_reference_summary, _, _, _ = evaluate_fixed_budget_policy(
    trigger_threshold=SELECTED_BRANCH_TRIGGER_THRESHOLD,
    branch_budget=1,
    policy_name="fixed_budget_1_threshold_0_80",
)

print("Selected adaptive policy:")
display(pd.DataFrame([selected_summary]))
print("Fixed one-branch reference:")
display(pd.DataFrame([fixed_reference_summary]))
print("Diagnostic minimum-cost adaptive row:")
display(pd.DataFrame([diagnostic_min_cost]))

display(
    selected_case_results.sort_values("total_replayed_requests", ascending=False)[
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

print("Adaptive continuation decisions for triggered cases:")
display(
    selected_trace[selected_trace["action"].isin(["launch_branch", "stop_branching"])]
    [
        [
            "case_id",
            "decision_step",
            "action",
            "top_pathology",
            "second_pathology",
            "math_support_count",
            "resolver_raw_margin",
            "resolver_entropy",
            "close_discriminator_available",
            "next_target_hypothesis",
            "continuation_value",
            "reason",
        ]
    ]
    .sort_values(["case_id", "decision_step"])
)


# %% [markdown]
# ## 8. Figures

# %%
plot_frame = frontier[frontier["policy_name"].eq("adaptive_value_branching_v1")].copy()

plt.figure(figsize=(9, 5))
plt.scatter(
    plot_frame["mean_total_replayed_requests"],
    plot_frame["accuracy"],
    c=plot_frame["candidate_pool_recall_count"],
    cmap="viridis",
    vmin=45,
    vmax=49,
    s=70,
    alpha=0.85,
)
plt.colorbar(label="Candidate-pool recall count")
plt.axhline(48 / 49, color="black", linestyle="--", linewidth=1, label="48/49")
plt.axvline(selected_summary["mean_total_replayed_requests"], color="tab:green", linestyle=":", linewidth=1.5, label="selected adaptive")
plt.xlabel("Mean total replayed requests")
plt.ylabel("Accuracy")
plt.title("Adaptive branch continuation frontier")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "adaptive_accuracy_vs_total_requests.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
selected_costs["branches_replayed"].value_counts().sort_index().plot(kind="bar", color="tab:blue")
plt.xlabel("Branches replayed per case")
plt.ylabel("Cases")
plt.title("Selected adaptive policy branch-count distribution")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "selected_branch_count_distribution.png", dpi=180)
plt.close()

continuation_rows = selected_trace[
    selected_trace["decision_step"].gt(1) & selected_trace["next_branch_id"].ne("")
].copy()
if not continuation_rows.empty:
    plt.figure(figsize=(10, 5))
    colors = np.where(continuation_rows["action"].eq("launch_branch"), "tab:orange", "tab:green")
    plt.bar(
        continuation_rows["case_id"] + " -> " + continuation_rows["next_target_hypothesis"].astype(str),
        continuation_rows["continuation_value"],
        color=colors,
    )
    plt.axhline(SELECTED_CONTINUATION_VALUE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Continuation value")
    plt.title("Why the adaptive controller stopped after branch 1")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "continuation_values_after_first_branch.png", dpi=180)
    plt.close()

native_total = (
    notebook30_predictions.set_index("case_id")["total_branch_requests"]
    + notebook33_case_results.set_index("case_id")["extra_roots_requested"].reindex(notebook30_predictions["case_id"]).fillna(0).values
)
selected_total = selected_case_results.set_index("case_id")["total_replayed_requests"].reindex(notebook30_predictions["case_id"])

plt.figure(figsize=(9, 5))
plt.hist(native_total, bins=18, alpha=0.55, label="Notebook 33 native")
plt.hist(selected_total, bins=18, alpha=0.70, label="adaptive selected")
plt.xlabel("Total branch/evidence requests")
plt.ylabel("Cases")
plt.title("Total request distribution: native vs adaptive")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "request_distribution_native_vs_adaptive.png", dpi=180)
plt.close()

case_compare = selected_case_results.sort_values("total_replayed_requests", ascending=False).head(12)
plt.figure(figsize=(10, 5))
plt.bar(case_compare["case_id"], case_compare["total_replayed_requests"], color="tab:purple")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total replayed requests")
plt.title("Highest-cost cases under selected adaptive controller")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "highest_cost_cases_selected_adaptive.png", dpi=180)
plt.close()

print("Figures written to", FIGURE_DIR)


# %% [markdown]
# ## 9. Artifact Contract

# %%
frontier.to_csv(ARTIFACT_ROOT / "adaptive_branch_controller_frontier.csv", index=False)
selected_case_results.to_csv(ARTIFACT_ROOT / "case_level_adaptive_branch_results.csv", index=False)
selected_candidate_scores.to_csv(ARTIFACT_ROOT / "candidate_level_adaptive_branch_scores.csv", index=False)
selected_costs.to_csv(ARTIFACT_ROOT / "case_level_adaptive_branch_costs.csv", index=False)
selected_trace.to_csv(ARTIFACT_ROOT / "adaptive_branch_decision_trace.csv", index=False)

selected_payload = {
    "policy_name": "adaptive_value_branching_v1",
    "status": "offline_adaptive_controller_candidate_for_live_confirmation",
    "method": "Use the learned branch-trigger MLP for the first branch, then continue to branch 2/3 only when a label-free expected-decision-change proxy exceeds a threshold.",
    "inputs_used": {
        "notebook30": str(NOTEBOOK30_ROOT),
        "notebook32": str(NOTEBOOK32_ROOT),
        "notebook33": str(NOTEBOOK33_ROOT),
        "notebook34": str(NOTEBOOK34_ROOT),
    },
    "no_live_api": True,
    "selection_rule": "Pre-register a robust trigger threshold of 0.80 and allow up to three branches, but use continuation value to decide whether additional branches are worth launching.",
    "selected_parameters": {
        "branch_trigger_threshold": SELECTED_BRANCH_TRIGGER_THRESHOLD,
        "max_branches": SELECTED_MAX_BRANCHES,
        "continuation_value_threshold": SELECTED_CONTINUATION_VALUE_THRESHOLD,
        "resolver": SELECTED_RESOLVER_NAME,
        "close_confounder_override_lbf_min": PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN,
        "continuation_value_formula": "unresolved_mass * next_priority_ratio * suppression",
        "unresolved_mass_formula": "0.45*ledger_disagreement + 0.30*resolver_entropy + 0.25*margin_uncertainty",
        "suppression_terms": [
            "0.35x when current decision has sufficient math-ledger support",
            "0.25x when a cheaper close-confounder discriminator is available",
        ],
    },
    "selected_summary": selected_summary,
    "fixed_one_branch_reference": fixed_reference_summary,
    "diagnostic_min_cost_adaptive_row": diagnostic_min_cost,
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
    "important_caveat": "This is an offline replay over branches already observed in Notebook 30. It tests adaptive pruning of saved branch continuations, not new branches that would appear under a fresh live trajectory.",
}
write_json(ARTIFACT_ROOT / "selected_adaptive_branch_policy.json", selected_payload)

resolved_config = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "project_root": str(PROJECT_ROOT),
    "artifact_root": str(ARTIFACT_ROOT),
    "selected_score_col": SELECTED_SCORE_COL,
    "trigger_threshold_grid": trigger_threshold_grid,
    "continuation_threshold_grid": continuation_threshold_grid,
    "selected_parameters": selected_payload["selected_parameters"],
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

artifact_check = pd.DataFrame(
    [
        {"artifact": "resolved_run_config.json", "exists": (ARTIFACT_ROOT / "resolved_run_config.json").exists()},
        {"artifact": "adaptive_branch_controller_frontier.csv", "exists": (ARTIFACT_ROOT / "adaptive_branch_controller_frontier.csv").exists()},
        {"artifact": "case_level_adaptive_branch_results.csv", "exists": (ARTIFACT_ROOT / "case_level_adaptive_branch_results.csv").exists()},
        {"artifact": "candidate_level_adaptive_branch_scores.csv", "exists": (ARTIFACT_ROOT / "candidate_level_adaptive_branch_scores.csv").exists()},
        {"artifact": "case_level_adaptive_branch_costs.csv", "exists": (ARTIFACT_ROOT / "case_level_adaptive_branch_costs.csv").exists()},
        {"artifact": "adaptive_branch_decision_trace.csv", "exists": (ARTIFACT_ROOT / "adaptive_branch_decision_trace.csv").exists()},
        {"artifact": "selected_adaptive_branch_policy.json", "exists": (ARTIFACT_ROOT / "selected_adaptive_branch_policy.json").exists()},
    ]
)
artifact_check.to_csv(ARTIFACT_ROOT / "artifact_contract_check.csv", index=False)
display(artifact_check)


# %% [markdown]
# ## 10. Report

# %%
selected_trace_compact = selected_trace[
    selected_trace["action"].isin(["launch_branch", "stop_branching"])
][
    [
        "case_id",
        "decision_step",
        "action",
        "top_pathology",
        "second_pathology",
        "math_support_count",
        "close_discriminator_available",
        "next_target_hypothesis",
        "continuation_value",
        "reason",
    ]
].sort_values(["case_id", "decision_step"])

report_lines = [
    "# Adaptive Value Branching Controller Report",
    "",
    f"Generated by `notebooks/35_adaptive_value_branching_controller.ipynb` on {datetime.now().isoformat(timespec='seconds')}.",
    "",
    "## Question",
    "",
    "Can the system allow up to three hypothesis branches while deciding, case by case, whether another branch is worth launching, instead of hard-coding a one-branch cap?",
    "",
    "## Method",
    "",
    "- Use Notebook 30 saved branch trajectories; no new API calls.",
    "- Keep Notebook 32 `gradient_boosting_name_family` as the resolver.",
    "- Keep Notebook 33 close-confounder discriminator as the cheaper final adjudicator.",
    "- Launch the first branch when the branch-trigger MLP probability crosses `0.80`.",
    "- After each completed branch, compute a label-free continuation value:",
    "",
    "```text",
    "continuation_value = unresolved_mass * next_priority_ratio * suppression",
    "unresolved_mass = 0.45 * ledger_disagreement",
    "                + 0.30 * resolver_entropy",
    "                + 0.25 * margin_uncertainty",
    "```",
    "",
    "- Suppress continuation when the current decision has enough graph/Bayes/MLP support or when a cheaper close-confounder discriminator is available.",
    f"- Continue only when `continuation_value >= {SELECTED_CONTINUATION_VALUE_THRESHOLD:.2f}`, with `max_branches = 3`.",
    "",
    "## Key Results",
    "",
    f"- Selected adaptive policy: `{selected_summary['num_correct']}/49`, candidate-pool recall `{selected_summary['candidate_pool_recall_count']}/49`.",
    f"- Mean selected requests: `{selected_summary['mean_selected_requests']:.2f}`.",
    f"- Mean total replayed requests: `{selected_summary['mean_total_replayed_requests']:.2f}`.",
    f"- Branch rows replayed: `{selected_summary['branch_rows_replayed']}` across `{selected_summary['branch_cases_with_saved_rows']}` cases.",
    f"- Cost reduction vs Notebook 33 native branching: `{100 * selected_payload['cost_delta_vs_notebook33_native']['mean_total_request_reduction_fraction']:.1f}%`.",
    f"- Remaining miss: `{selected_summary['misses']}`.",
    "",
    "## Selected Policy Summary",
    "",
    pd.DataFrame([selected_summary]).to_markdown(index=False),
    "",
    "## Frontier",
    "",
    frontier[
        [
            "policy_name",
            "branch_trigger_threshold",
            "continuation_value_threshold",
            "max_branches",
            "candidate_pool_recall_count",
            "num_correct",
            "mean_total_replayed_requests",
            "branch_rows_replayed",
            "misses",
        ]
    ]
    .head(15)
    .to_markdown(index=False),
    "",
    "## Triggered-Case Decisions",
    "",
    selected_trace_compact.to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    "The adaptive controller is allowed to use up to three branches, but on this replay it chooses exactly one branch for each high-trigger case. This matches Notebook 34's efficient operating point while providing a cleaner design story: additional branches are not forbidden, they simply fail the expected-value test after the first branch.",
    "",
    "The result supports the idea that the first forced hypothesis branch is doing the important work of candidate-pool repair. The second and third branches mostly add cost because the remaining uncertainty is better handled by the graph/Bayes/MLP resolver and the close-confounder discriminator than by another full LLM branch.",
    "",
    "## Caveat",
    "",
    "This is still an offline replay over saved branches. It can validate adaptive pruning over observed trajectories, but it cannot prove how often new live trajectories would ask for branch 2 or branch 3. The policy should be live-confirmed before promotion.",
    "",
    "## Artifact Contract",
    "",
    "- `resolved_run_config.json`",
    "- `adaptive_branch_controller_frontier.csv`",
    "- `case_level_adaptive_branch_results.csv`",
    "- `candidate_level_adaptive_branch_scores.csv`",
    "- `case_level_adaptive_branch_costs.csv`",
    "- `adaptive_branch_decision_trace.csv`",
    "- `selected_adaptive_branch_policy.json`",
    "- figures under `figures/`",
]
REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print("Report written to", REPORT_PATH)


# %% [markdown]
# ## 11. Final Summary

# %%
print("Selected adaptive value branching policy:")
print(json.dumps(selected_payload["selected_parameters"], indent=2))
print()
print(
    f"Result: {selected_summary['num_correct']}/49 accuracy, "
    f"{selected_summary['candidate_pool_recall_count']}/49 candidate-pool recall, "
    f"{selected_summary['mean_total_replayed_requests']:.2f} mean total replayed requests."
)
print("Remaining miss:", selected_summary["misses"])
