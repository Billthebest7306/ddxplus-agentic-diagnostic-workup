# %% [markdown]
# # Notebook 36: Adaptive Branching Stress Test
#
# Offline artificial stress test for the Notebook 35 adaptive continuation
# controller.
#
# Notebook 35 showed that, on the saved 49-case replay, allowing up to three
# branches still results in exactly one useful branch per high-trigger case.
# That is efficient, but it leaves a design question:
#
# ```text
# Would the controller ever spend branch 2/3 if branch 1 failed?
# ```
#
# The current saved 49-case pool has no natural case where branch 2/3 are needed
# after a successful branch 1. This notebook therefore runs adversarial offline
# probes:
#
# - remove branch 1 and let branch 2 become the first available branch
# - spend branch 1 but hide its candidate contribution, simulating a no-signal
#   first branch
# - sweep continuation thresholds and a diagnostic support-suppression ablation
#
# This notebook makes no API calls and does not promote a new live policy.

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
except Exception:  # pragma: no cover - script fallback outside notebooks.
    def display(obj: Any) -> None:
        print(obj)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 36
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (candidate_root / "notebooks").exists() and (candidate_root / "artifacts").exists():
        PROJECT_ROOT = candidate_root
        break

NOTEBOOK30_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1"
NOTEBOOK32_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "resolver_ablation_lab_49case_v1"
NOTEBOOK33_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "close_confounder_discriminator_49case_v1"
NOTEBOOK35_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "adaptive_value_branching_controller_49case_v1"

RUN_NAME = "adaptive_branching_stress_test_49case_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "adaptive_branching_stress_test_report.md"

SELECTED_SCORE_COL = "score__gradient_boosting_name_family"
SELECTED_BRANCH_TRIGGER_THRESHOLD = 0.80
SELECTED_CONTINUATION_THRESHOLD = 0.40
SELECTED_MAX_BRANCHES = 3
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


def softmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    centered = values.astype(float) - float(np.max(values.astype(float)))
    exp_values = np.exp(centered)
    denom = float(exp_values.sum())
    if denom <= 0:
        return np.ones(len(values), dtype=float) / len(values)
    return exp_values / denom


def compact_miss_list(frame: pd.DataFrame, prediction_col: str = "selected_pathology") -> str:
    misses = frame[~frame["correct"].astype(bool)]
    return "; ".join(
        f"{row.case_id}:{row.true_pathology}->{getattr(row, prediction_col)}"
        for row in misses.itertuples(index=False)
    )


# %% [markdown]
# ## 2. Load Notebook 30/32/33/35 Artifacts

# %%
required_paths = {
    "notebook30_predictions": NOTEBOOK30_ROOT / "predictions.csv",
    "notebook30_live_candidates": NOTEBOOK30_ROOT / "candidate_level_live_scores.csv",
    "notebook30_branch_assignments": NOTEBOOK30_ROOT / "hypothesis_branch_assignments.csv",
    "notebook32_candidate_scores": NOTEBOOK32_ROOT / "candidate_level_resolver_ablation_scores.csv",
    "notebook33_case_results": NOTEBOOK33_ROOT / "case_level_close_confounder_results.csv",
    "notebook33_policy_summary": NOTEBOOK33_ROOT / "close_confounder_policy_summary.csv",
    "notebook35_selected_policy": NOTEBOOK35_ROOT / "selected_adaptive_branch_policy.json",
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
notebook35_selected_policy = json.loads(required_paths["notebook35_selected_policy"].read_text(encoding="utf-8"))

if SELECTED_SCORE_COL not in candidate_scores.columns:
    raise KeyError(f"Missing selected resolver score column: {SELECTED_SCORE_COL}")

notebook33_selected_row = notebook33_policy_summary[
    notebook33_policy_summary["policy_name"].eq("close_confounder_discriminator_v1")
].iloc[0].to_dict()

print("Notebook 30 cases:", notebook30_predictions["case_id"].nunique())
print("Notebook 32 candidate rows:", len(candidate_scores))
print("Notebook 35 selected:", notebook35_selected_policy["selected_parameters"])

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

display(candidate_frame["candidate_role"].value_counts().rename_axis("candidate_role").reset_index(name="rows"))
display(branch_cost_frame.sort_values(["case_id", "hypothesis_order"]).head(12))


# %% [markdown]
# ## 4. Controller Helpers

# %%
MATH_SCORE_COLS = [
    "score__graph_posterior",
    "score__bayes_posterior",
    "score__mlp_posterior",
]


def candidates_for_case(case_id: str, contributing_orders: set[int]) -> pd.DataFrame:
    group = candidate_frame[candidate_frame["case_id"].eq(case_id)].copy()
    include_non_branch = group["candidate_role"].ne("branch")
    include_branch = group["candidate_role"].eq("branch") & group["hypothesis_order"].isin(contributing_orders)
    return group[include_non_branch | include_branch].copy()


def diagnosis_level_state(case_id: str, contributing_orders: set[int]) -> dict[str, Any]:
    candidates = candidates_for_case(case_id, contributing_orders)
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
        "included_branch_count": int(len(contributing_orders)),
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


def branch_metadata(case_id: str, order: int, first_available_order: int = 1) -> dict[str, Any] | None:
    rows = branch_cost_frame[
        branch_cost_frame["case_id"].eq(case_id) & branch_cost_frame["hypothesis_order"].eq(order)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    first_rows = branch_cost_frame[
        branch_cost_frame["case_id"].eq(case_id) & branch_cost_frame["hypothesis_order"].eq(first_available_order)
    ]
    first_priority = float(first_rows.iloc[0]["hypothesis_priority"]) if not first_rows.empty else float(row["hypothesis_priority"])
    return {
        "next_branch_order": int(order),
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
    suppress_sufficient_support: bool = True,
    suppress_close_discriminator: bool = True,
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
    if suppress_sufficient_support and sufficient_update:
        suppression *= 0.35
        stop_reasons.append("sufficient_current_decision_support")
    if suppress_close_discriminator and as_bool(current_state["close_discriminator_available"]):
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


# %% [markdown]
# ## 5. Scenario Policies

# %%
def scenario_orders_for_case(
    case_id: str,
    scenario: str,
    continuation_threshold: float,
    suppress_sufficient_support: bool,
    max_branches: int = 3,
) -> tuple[set[int], set[int], list[dict[str, Any]]]:
    """Return contributing orders, spent orders, and decision trace."""
    trace_rows: list[dict[str, Any]] = []
    contributing_orders: set[int] = set()
    spent_orders: set[int] = set()
    trigger_probability = float(case_trigger_probability.loc[case_id])
    base_state = diagnosis_level_state(case_id, contributing_orders)

    if trigger_probability < SELECTED_BRANCH_TRIGGER_THRESHOLD or max_branches <= 0:
        trace_rows.append(
            {
                **base_state,
                "scenario": scenario,
                "case_id": case_id,
                "decision_step": 0,
                "action": "stop_no_initial_branch",
                "branch_trigger_probability": trigger_probability,
                "continuation_value_threshold": continuation_threshold,
                "next_branch_order": np.nan,
                "next_branch_id": "",
                "next_target_hypothesis": "",
                "continuation_value": 0.0,
                "reason": "branch_trigger_below_threshold_or_zero_budget",
            }
        )
        return contributing_orders, spent_orders, trace_rows

    available_orders = [1, 2, 3]
    if scenario == "branch1_removed_slot_remap":
        available_orders = [2, 3]

    previous_state = base_state

    if scenario.startswith("branch1_no_signal"):
        meta = branch_metadata(case_id, 1, first_available_order=1)
        if meta is not None:
            spent_orders.add(1)
            trace_rows.append(
                {
                    **base_state,
                    "scenario": scenario,
                    "case_id": case_id,
                    "decision_step": 1,
                    "action": "spend_branch_no_signal",
                    "branch_trigger_probability": trigger_probability,
                    "continuation_value_threshold": continuation_threshold,
                    **meta,
                    "continuation_value": 1.0,
                    "reason": "stress_hidden_first_branch_contribution",
                    "unresolved_mass": np.nan,
                    "top_changed_after_last_branch": False,
                    "sufficient_update": False,
                    "suppression": 1.0,
                    "suppression_reasons": "",
                }
            )
        available_orders = [2, 3]

    for step_index, order in enumerate(available_orders, start=1):
        meta = branch_metadata(
            case_id,
            order,
            first_available_order=available_orders[0] if scenario == "branch1_removed_slot_remap" else 1,
        )
        if meta is None:
            continue

        current_state = diagnosis_level_state(case_id, contributing_orders)
        if not spent_orders and scenario in {"native_adaptive_reference", "branch1_removed_slot_remap"}:
            action = "launch_branch"
            launch_value = 1.0
            reason = "initial_branch_trigger_fired"
            value_payload = {
                "unresolved_mass": np.nan,
                "next_priority_ratio": float(meta["next_priority_ratio"]),
                "top_changed_after_last_branch": False,
                "sufficient_update": False,
                "suppression": 1.0,
                "suppression_reasons": "",
            }
        else:
            value_payload = continuation_value(
                previous_state=previous_state,
                current_state=current_state,
                next_meta=meta,
                suppress_sufficient_support=suppress_sufficient_support,
                suppress_close_discriminator=True,
            )
            launch_value = float(value_payload["continuation_value"])
            action = "launch_branch" if launch_value >= continuation_threshold else "stop_branching"
            if value_payload["suppression_reasons"]:
                reason = value_payload["suppression_reasons"]
            elif action == "launch_branch":
                reason = "continuation_value_above_threshold"
            else:
                reason = "continuation_value_below_threshold"

        trace_rows.append(
            {
                **current_state,
                "scenario": scenario,
                "case_id": case_id,
                "decision_step": step_index + 1 if scenario.startswith("branch1_no_signal") else step_index,
                "action": action,
                "branch_trigger_probability": trigger_probability,
                "continuation_value_threshold": continuation_threshold,
                **meta,
                "continuation_value": float(launch_value),
                "reason": reason,
                **value_payload,
            }
        )

        if action != "launch_branch":
            break

        spent_orders.add(order)
        contributing_orders.add(order)
        previous_state = diagnosis_level_state(case_id, contributing_orders)

        if len(spent_orders) >= max_branches:
            break

    return contributing_orders, spent_orders, trace_rows


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


def costs_for_spent_orders(selected: pd.DataFrame, spent_orders_by_case: dict[str, set[int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_extra = selected.set_index("case_id")["extra_roots_requested"]
    for case_id in case_costs["case_id"]:
        orders = spent_orders_by_case.get(case_id, set())
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
                "branches_spent": int(branches["branch_id"].nunique()),
                "extra_roots_requested": extra,
                "total_replayed_requests": base_requests + branch_requests + extra,
            }
        )
    return pd.DataFrame(rows)


def evaluate_stress_scenario(
    scenario: str,
    continuation_threshold: float,
    suppress_sufficient_support: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contributing_orders_by_case: dict[str, set[int]] = {}
    spent_orders_by_case: dict[str, set[int]] = {}
    trace_rows: list[dict[str, Any]] = []

    for case_id in case_costs["case_id"]:
        contributing_orders, spent_orders, trace = scenario_orders_for_case(
            case_id=case_id,
            scenario=scenario,
            continuation_threshold=continuation_threshold,
            suppress_sufficient_support=suppress_sufficient_support,
            max_branches=SELECTED_MAX_BRANCHES,
        )
        contributing_orders_by_case[case_id] = contributing_orders
        spent_orders_by_case[case_id] = spent_orders
        trace_rows.extend(trace)

    candidates = included_candidates_for_orders(contributing_orders_by_case)
    selected, scored_candidates = select_with_close_confounder(candidates)
    costs = costs_for_spent_orders(selected, spent_orders_by_case)
    selected = selected.merge(costs.drop(columns=["extra_roots_requested"]), on="case_id", how="left")
    trace = pd.DataFrame(trace_rows)

    summary = {
        "scenario": scenario,
        "continuation_value_threshold": float(continuation_threshold),
        "suppress_sufficient_support": bool(suppress_sufficient_support),
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
        "max_total_replayed_requests": float(selected["total_replayed_requests"].max()),
        "branch_cases_with_spend": int((costs["branches_spent"] > 0).sum()),
        "branches_spent_total": int(costs["branches_spent"].sum()),
        "branch2_launch_count": int((trace["action"].eq("launch_branch") & trace["next_branch_order"].eq(2)).sum()),
        "branch3_launch_count": int((trace["action"].eq("launch_branch") & trace["next_branch_order"].eq(3)).sum()),
        "extra_roots_total": int(selected["extra_roots_requested"].sum()),
        "overrides_applied": int(selected["override_applied"].sum()),
        "candidate_pool_misses": "; ".join(selected.loc[~selected["candidate_pool_has_true"], "case_id"]),
        "misses": compact_miss_list(selected),
    }
    return summary, selected, scored_candidates, costs, trace


# %% [markdown]
# ## 6. Prefix And Necessity Diagnostics

# %%
def evaluate_fixed_orders(orders: set[int]) -> pd.DataFrame:
    case_orders = {case_id: set(orders) for case_id in case_costs["case_id"]}
    candidates = included_candidates_for_orders(case_orders)
    selected, _ = select_with_close_confounder(candidates)
    return selected


prefix_rows: list[dict[str, Any]] = []
for label, orders in [
    ("no_branch", set()),
    ("branch1_only", {1}),
    ("branch2_only", {2}),
    ("branch3_only", {3}),
    ("branch1_2", {1, 2}),
    ("branch1_2_3", {1, 2, 3}),
]:
    selected = evaluate_fixed_orders(orders)
    prefix_rows.append(
        {
            "prefix": label,
            "candidate_pool_recall_count": int(selected["candidate_pool_has_true"].sum()),
            "num_correct": int(selected["correct"].sum()),
            "candidate_pool_misses": "; ".join(selected.loc[~selected["candidate_pool_has_true"], "case_id"]),
            "misses": compact_miss_list(selected),
        }
    )

prefix_diagnostics = pd.DataFrame(prefix_rows)
display(prefix_diagnostics)

necessity_rows: list[dict[str, Any]] = []
for case_id in case_costs["case_id"]:
    no_branch = evaluate_fixed_orders(set())
    break

no_branch_by_case = evaluate_fixed_orders(set()).set_index("case_id")
branch1_by_case = evaluate_fixed_orders({1}).set_index("case_id")
branch2_by_case = evaluate_fixed_orders({2}).set_index("case_id")
branch3_by_case = evaluate_fixed_orders({3}).set_index("case_id")

for case_id, case_row in case_costs.set_index("case_id").iterrows():
    if float(case_row["branch_trigger_probability"]) < SELECTED_BRANCH_TRIGGER_THRESHOLD:
        continue
    base_state = diagnosis_level_state(case_id, set())
    branch2_meta = branch_metadata(case_id, 2)
    if branch2_meta is not None:
        value_payload = continuation_value(
            previous_state=base_state,
            current_state=base_state,
            next_meta=branch2_meta,
            suppress_sufficient_support=True,
            suppress_close_discriminator=True,
        )
        no_support_value_payload = continuation_value(
            previous_state=base_state,
            current_state=base_state,
            next_meta=branch2_meta,
            suppress_sufficient_support=False,
            suppress_close_discriminator=True,
        )
    else:
        value_payload = {"continuation_value": np.nan, "suppression_reasons": ""}
        no_support_value_payload = {"continuation_value": np.nan, "suppression_reasons": ""}

    necessity_rows.append(
        {
            "case_id": case_id,
            "true_pathology": str(case_row["true_pathology"]),
            "branch_trigger_probability": float(case_row["branch_trigger_probability"]),
            "no_branch_candidate_has_true": bool(no_branch_by_case.loc[case_id, "candidate_pool_has_true"]),
            "branch1_candidate_has_true": bool(branch1_by_case.loc[case_id, "candidate_pool_has_true"]),
            "branch2_candidate_has_true": bool(branch2_by_case.loc[case_id, "candidate_pool_has_true"]),
            "branch3_candidate_has_true": bool(branch3_by_case.loc[case_id, "candidate_pool_has_true"]),
            "no_branch_prediction": str(no_branch_by_case.loc[case_id, "selected_pathology"]),
            "branch1_prediction": str(branch1_by_case.loc[case_id, "selected_pathology"]),
            "branch2_prediction": str(branch2_by_case.loc[case_id, "selected_pathology"]),
            "branch3_prediction": str(branch3_by_case.loc[case_id, "selected_pathology"]),
            "branch2_value_after_no_signal_branch1": float(value_payload["continuation_value"]),
            "branch2_value_without_support_suppression": float(no_support_value_payload["continuation_value"]),
            "branch2_stop_reason_after_no_signal": str(value_payload.get("suppression_reasons", "")),
            "base_top_pathology": str(base_state["top_pathology"]),
            "base_math_support_count": int(base_state["math_support_count"]),
            "base_resolver_raw_margin": float(base_state["resolver_raw_margin"]),
            "base_resolver_entropy": float(base_state["resolver_entropy"]),
        }
    )

necessity_diagnostics = pd.DataFrame(necessity_rows)
display(necessity_diagnostics)


# %% [markdown]
# ## 7. Stress Scenario Evaluation

# %%
scenario_specs = [
    {
        "scenario": "native_adaptive_reference",
        "continuation_threshold": SELECTED_CONTINUATION_THRESHOLD,
        "suppress_sufficient_support": True,
    },
    {
        "scenario": "branch1_removed_slot_remap",
        "continuation_threshold": SELECTED_CONTINUATION_THRESHOLD,
        "suppress_sufficient_support": True,
    },
    {
        "scenario": "branch1_no_signal_selected",
        "continuation_threshold": SELECTED_CONTINUATION_THRESHOLD,
        "suppress_sufficient_support": True,
    },
    {
        "scenario": "branch1_no_signal_no_support_suppression",
        "continuation_threshold": SELECTED_CONTINUATION_THRESHOLD,
        "suppress_sufficient_support": False,
    },
]

for threshold in [0.05, 0.10, 0.20, 0.30, 0.40]:
    scenario_specs.append(
        {
            "scenario": "branch1_no_signal_threshold_sweep",
            "continuation_threshold": threshold,
            "suppress_sufficient_support": True,
        }
    )

scenario_rows: list[dict[str, Any]] = []
case_result_frames: list[pd.DataFrame] = []
candidate_score_frames: list[pd.DataFrame] = []
cost_frames: list[pd.DataFrame] = []
trace_frames: list[pd.DataFrame] = []

for spec in scenario_specs:
    summary, selected, scored, costs, trace = evaluate_stress_scenario(**spec)
    scenario_rows.append(summary)
    scenario_key = f"{spec['scenario']}|thr={spec['continuation_threshold']:.2f}|suppress={spec['suppress_sufficient_support']}"
    selected.insert(0, "scenario_key", scenario_key)
    scored.insert(0, "scenario_key", scenario_key)
    costs.insert(0, "scenario_key", scenario_key)
    trace.insert(0, "scenario_key", scenario_key)
    case_result_frames.append(selected)
    candidate_score_frames.append(scored)
    cost_frames.append(costs)
    trace_frames.append(trace)

scenario_summary = pd.DataFrame(scenario_rows).sort_values(
    ["candidate_pool_recall_count", "num_correct", "mean_total_replayed_requests"],
    ascending=[False, False, True],
)
case_results = pd.concat(case_result_frames, ignore_index=True)
candidate_scores_out = pd.concat(candidate_score_frames, ignore_index=True)
costs_out = pd.concat(cost_frames, ignore_index=True)
decision_trace = pd.concat(trace_frames, ignore_index=True)

display(
    scenario_summary[
        [
            "scenario",
            "continuation_value_threshold",
            "suppress_sufficient_support",
            "candidate_pool_recall_count",
            "num_correct",
            "mean_total_replayed_requests",
            "branches_spent_total",
            "branch2_launch_count",
            "branch3_launch_count",
            "candidate_pool_misses",
            "misses",
        ]
    ]
)


# %% [markdown]
# ## 8. Figures

# %%
main_plot = scenario_summary[
    scenario_summary["scenario"].isin(
        [
            "native_adaptive_reference",
            "branch1_removed_slot_remap",
            "branch1_no_signal_selected",
            "branch1_no_signal_no_support_suppression",
        ]
    )
].copy()

plt.figure(figsize=(10, 5))
x = np.arange(len(main_plot))
plt.bar(x - 0.18, main_plot["num_correct"], width=0.36, label="correct", color="tab:blue")
plt.bar(x + 0.18, main_plot["candidate_pool_recall_count"], width=0.36, label="candidate recall", color="tab:green")
plt.axhline(48, color="black", linestyle="--", linewidth=1)
plt.xticks(x, main_plot["scenario"], rotation=35, ha="right")
plt.ylabel("Cases out of 49")
plt.title("Artificial branch-stress outcomes")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "stress_scenario_accuracy_recall.png", dpi=180)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(main_plot["scenario"], main_plot["mean_total_replayed_requests"], color="tab:purple")
plt.axhline(8.979591836734693, color="black", linestyle="--", linewidth=1, label="Notebook 35 selected")
plt.xticks(rotation=35, ha="right")
plt.ylabel("Mean total replayed requests")
plt.title("Stress scenario request cost")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "stress_scenario_request_cost.png", dpi=180)
plt.close()

sweep = scenario_summary[scenario_summary["scenario"].eq("branch1_no_signal_threshold_sweep")].copy()
plt.figure(figsize=(9, 5))
plt.plot(sweep["continuation_value_threshold"], sweep["branch2_launch_count"], marker="o", label="branch 2 launches")
plt.plot(sweep["continuation_value_threshold"], sweep["branch3_launch_count"], marker="o", label="branch 3 launches")
plt.gca().invert_xaxis()
plt.xlabel("Continuation threshold")
plt.ylabel("Launch count")
plt.title("No-signal branch-1 stress: continuation sensitivity")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "no_signal_threshold_sweep_branch_launches.png", dpi=180)
plt.close()

triggered_values = necessity_diagnostics.sort_values("branch2_value_after_no_signal_branch1", ascending=False)
plt.figure(figsize=(10, 5))
plt.bar(triggered_values["case_id"], triggered_values["branch2_value_after_no_signal_branch1"], color="tab:orange")
plt.axhline(SELECTED_CONTINUATION_THRESHOLD, color="black", linestyle="--", linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Branch 2 continuation value")
plt.title("Would branch 2 fire after a no-signal branch 1?")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "branch2_value_after_no_signal_branch1.png", dpi=180)
plt.close()

print("Figures written to", FIGURE_DIR)


# %% [markdown]
# ## 9. Artifact Contract

# %%
scenario_summary.to_csv(ARTIFACT_ROOT / "stress_scenario_summary.csv", index=False)
case_results.to_csv(ARTIFACT_ROOT / "case_level_stress_results.csv", index=False)
candidate_scores_out.to_csv(ARTIFACT_ROOT / "candidate_level_stress_scores.csv", index=False)
costs_out.to_csv(ARTIFACT_ROOT / "case_level_stress_costs.csv", index=False)
decision_trace.to_csv(ARTIFACT_ROOT / "stress_decision_trace.csv", index=False)
prefix_diagnostics.to_csv(ARTIFACT_ROOT / "branch_prefix_diagnostics.csv", index=False)
necessity_diagnostics.to_csv(ARTIFACT_ROOT / "branch_necessity_diagnostics.csv", index=False)

selected_reference = scenario_summary[
    scenario_summary["scenario"].eq("native_adaptive_reference")
    & scenario_summary["continuation_value_threshold"].eq(SELECTED_CONTINUATION_THRESHOLD)
].iloc[0].to_dict()
no_signal_selected = scenario_summary[
    scenario_summary["scenario"].eq("branch1_no_signal_selected")
    & scenario_summary["continuation_value_threshold"].eq(SELECTED_CONTINUATION_THRESHOLD)
].iloc[0].to_dict()
no_support_suppression = scenario_summary[
    scenario_summary["scenario"].eq("branch1_no_signal_no_support_suppression")
].iloc[0].to_dict()

selected_payload = {
    "policy_name": "adaptive_branching_stress_test_v1",
    "status": "offline_artificial_stress_test_not_promoted",
    "method": "Adversarial replay over saved Notebook 30 branches: branch1 removed, branch1 no-signal, threshold sweeps, and support-suppression diagnostic ablation.",
    "inputs_used": {
        "notebook30": str(NOTEBOOK30_ROOT),
        "notebook32": str(NOTEBOOK32_ROOT),
        "notebook33": str(NOTEBOOK33_ROOT),
        "notebook35": str(NOTEBOOK35_ROOT),
    },
    "no_live_api": True,
    "selected_notebook35_reference": selected_reference,
    "branch1_no_signal_selected_controller": no_signal_selected,
    "branch1_no_signal_no_support_suppression_diagnostic": no_support_suppression,
    "main_findings": [
        "The natural saved 49-case replay contains no case where branch 2/3 improve beyond branch 1.",
        "When branch 1 is artificially removed or made no-signal, the selected controller does not reliably recover all branch-1-dependent cases.",
        "The main failure mode is false stability: graph/Bayes/MLP can agree on the wrong current top candidate, suppressing continuation.",
        "This does not invalidate Notebook 35 for the current replay, but it means live or larger-slice testing is needed to prove adaptive branch 2/3 spending under natural need.",
    ],
}
write_json(ARTIFACT_ROOT / "selected_stress_test_findings.json", selected_payload)

resolved_config = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "project_root": str(PROJECT_ROOT),
    "artifact_root": str(ARTIFACT_ROOT),
    "selected_score_col": SELECTED_SCORE_COL,
    "selected_branch_trigger_threshold": SELECTED_BRANCH_TRIGGER_THRESHOLD,
    "selected_continuation_threshold": SELECTED_CONTINUATION_THRESHOLD,
    "selected_max_branches": SELECTED_MAX_BRANCHES,
    "scenario_specs": scenario_specs,
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_config)

artifact_check = pd.DataFrame(
    [
        {"artifact": "resolved_run_config.json", "exists": (ARTIFACT_ROOT / "resolved_run_config.json").exists()},
        {"artifact": "stress_scenario_summary.csv", "exists": (ARTIFACT_ROOT / "stress_scenario_summary.csv").exists()},
        {"artifact": "case_level_stress_results.csv", "exists": (ARTIFACT_ROOT / "case_level_stress_results.csv").exists()},
        {"artifact": "candidate_level_stress_scores.csv", "exists": (ARTIFACT_ROOT / "candidate_level_stress_scores.csv").exists()},
        {"artifact": "case_level_stress_costs.csv", "exists": (ARTIFACT_ROOT / "case_level_stress_costs.csv").exists()},
        {"artifact": "stress_decision_trace.csv", "exists": (ARTIFACT_ROOT / "stress_decision_trace.csv").exists()},
        {"artifact": "branch_prefix_diagnostics.csv", "exists": (ARTIFACT_ROOT / "branch_prefix_diagnostics.csv").exists()},
        {"artifact": "branch_necessity_diagnostics.csv", "exists": (ARTIFACT_ROOT / "branch_necessity_diagnostics.csv").exists()},
        {"artifact": "selected_stress_test_findings.json", "exists": (ARTIFACT_ROOT / "selected_stress_test_findings.json").exists()},
    ]
)
artifact_check.to_csv(ARTIFACT_ROOT / "artifact_contract_check.csv", index=False)
display(artifact_check)


# %% [markdown]
# ## 10. Report

# %%
report_lines = [
    "# Adaptive Branching Stress Test Report",
    "",
    f"Generated by `notebooks/36_adaptive_branching_stress_test.ipynb` on {datetime.now().isoformat(timespec='seconds')}.",
    "",
    "## Question",
    "",
    "Can we show, without new API calls, that the Notebook `35` adaptive branch controller would launch branch 2/3 when branch 1 fails or is unavailable?",
    "",
    "## Method",
    "",
    "- Reuse the saved Notebook `30` branch pool, Notebook `32` resolver scores, Notebook `33` discriminator, and Notebook `35` continuation logic.",
    "- Run adversarial replays rather than live calls.",
    "- Stress scenarios:",
    "  - native Notebook `35` adaptive reference",
    "  - branch 1 removed, so branch 2 becomes the first available branch",
    "  - branch 1 spent but hidden from the candidate pool, simulating a no-signal first branch",
    "  - no-signal branch 1 with support-suppression disabled as a diagnostic ablation",
    "  - continuation threshold sweep over the no-signal branch-1 setting",
    "",
    "## Prefix Diagnostics",
    "",
    prefix_diagnostics.to_markdown(index=False),
    "",
    "## Scenario Summary",
    "",
    scenario_summary[
        [
            "scenario",
            "continuation_value_threshold",
            "suppress_sufficient_support",
            "candidate_pool_recall_count",
            "num_correct",
            "mean_total_replayed_requests",
            "branches_spent_total",
            "branch2_launch_count",
            "branch3_launch_count",
            "candidate_pool_misses",
            "misses",
        ]
    ].to_markdown(index=False),
    "",
    "## Branch-2 Necessity Probe",
    "",
    necessity_diagnostics[
        [
            "case_id",
            "true_pathology",
            "no_branch_candidate_has_true",
            "branch1_candidate_has_true",
            "branch2_candidate_has_true",
            "branch3_candidate_has_true",
            "branch2_value_after_no_signal_branch1",
            "branch2_value_without_support_suppression",
            "base_top_pathology",
            "base_math_support_count",
        ]
    ].to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    "The natural saved 49-case replay still has no example where branch 2 or branch 3 is needed after branch 1 succeeds: branch 1 already recovers all candidate-pool misses. The artificial stress test therefore checks mechanics rather than natural generalization.",
    "",
    "The stress test is mixed. When branch 1 is artificially removed or hidden, branch 2/3 do not reliably recover all branch-1-dependent cases under the selected Notebook `35` continuation rule. The failure mode is false stability: graph/Bayes/MLP can agree on the current wrong top diagnosis, which suppresses continuation even though an alternate branch would be desirable in hindsight.",
    "",
    "This does not invalidate Notebook `35` for the current saved replay. It does mean the current 49 cases can only prove that the controller avoids wasteful extra branches, not that it will reliably spend branch 2/3 under natural need. To prove that, we need either a larger saved branch pool or a live confirmation where branch 2/3 opportunities occur naturally.",
    "",
    "## Artifact Contract",
    "",
    "- `resolved_run_config.json`",
    "- `stress_scenario_summary.csv`",
    "- `case_level_stress_results.csv`",
    "- `candidate_level_stress_scores.csv`",
    "- `case_level_stress_costs.csv`",
    "- `stress_decision_trace.csv`",
    "- `branch_prefix_diagnostics.csv`",
    "- `branch_necessity_diagnostics.csv`",
    "- `selected_stress_test_findings.json`",
    "- figures under `figures/`",
]
REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print("Report written to", REPORT_PATH)


# %% [markdown]
# ## 11. Final Summary

# %%
print("Stress test complete.")
print("Native adaptive reference:")
print(json.dumps(selected_reference, indent=2))
print()
print("Branch-1 no-signal selected controller:")
print(json.dumps(no_signal_selected, indent=2))
print()
print("Main conclusion:")
for item in selected_payload["main_findings"]:
    print("-", item)
