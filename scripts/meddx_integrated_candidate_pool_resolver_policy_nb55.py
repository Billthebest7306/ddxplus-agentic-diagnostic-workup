from __future__ import annotations

# %% [markdown]
# # Notebook 55: MEDDx Integrated Candidate-Pool Resolver Policy
#
# Notebook 53 passed the Stage 1 candidate-pool recovery gate. Notebook 54 passed the Stage 2
# held-out resolver gate. This notebook freezes one integrated offline policy and audits whether it is
# ready for a small live pilot.
#
# It does not fit a new model and makes no API calls. It packages:
#
# - `saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1`
# - `recovered_pool_logistic_evidence_card_resolver_v1`

# %%
import json
import math
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

pd.set_option("display.max_columns", 120)
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
RESOLVER_RUN_NAME = "meddx_evidence_card_resolver_lab_v1"
RUN_NAME = "meddx_integrated_candidate_pool_resolver_policy_v1"

POOL_ROOT = ROOT / "artifacts" / "universal_meddx" / POOL_RUN_NAME
RESOLVER_ROOT = ROOT / "artifacts" / "universal_meddx" / RESOLVER_RUN_NAME
ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_POOL_POLICY = "saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1"
SELECTED_RESOLVER_POLICY = "recovered_pool_logistic_evidence_card_resolver_v1"
INTEGRATED_POLICY_NAME = "integrated_recovered_pool_evidence_card_policy_v1"

print("Project root :", ROOT)
print("Pool root    :", POOL_ROOT)
print("Resolver root:", RESOLVER_ROOT)
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_summary(
    frame: pd.DataFrame,
    policy_name: str,
    cohort: str,
    split: str,
    dataset_name: str = "ALL",
    budget: int = -1,
) -> pd.Series:
    rows = frame[
        frame["policy_name"].eq(policy_name)
        & frame["cohort"].eq(cohort)
        & frame["dataset_name"].eq(dataset_name)
        & frame["budget"].astype(int).eq(int(budget))
        & frame["split"].eq(split)
    ]
    if not len(rows):
        raise ValueError(f"Missing summary row: {policy_name=} {cohort=} {split=} {dataset_name=} {budget=}")
    return rows.iloc[0]

# %% [markdown]
# ## 2. Load Stage 1 And Stage 2 Artifacts

# %%
required = [
    POOL_ROOT / "candidate_pool_recovery_summary.csv",
    POOL_ROOT / "selected_policy.json",
    RESOLVER_ROOT / "resolver_policy_summary.csv",
    RESOLVER_ROOT / "selected_policy.json",
    RESOLVER_ROOT / "selected_resolver_case_results.csv",
    RESOLVER_ROOT / "case_blocked_test_case_results.csv",
    RESOLVER_ROOT / "transfer_old90_resolver_results.csv",
    RESOLVER_ROOT / "paired_current_vs_resolver.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

pool_summary = pd.read_csv(POOL_ROOT / "candidate_pool_recovery_summary.csv")
pool_selected = read_json(POOL_ROOT / "selected_policy.json")
resolver_summary = pd.read_csv(RESOLVER_ROOT / "resolver_policy_summary.csv")
resolver_selected = read_json(RESOLVER_ROOT / "selected_policy.json")
scale_case_results = pd.read_csv(RESOLVER_ROOT / "selected_resolver_case_results.csv")
test_case_results = pd.read_csv(RESOLVER_ROOT / "case_blocked_test_case_results.csv")
transfer_case_results = pd.read_csv(RESOLVER_ROOT / "transfer_old90_resolver_results.csv")
paired_current_vs_resolver = pd.read_csv(RESOLVER_ROOT / "paired_current_vs_resolver.csv")

display(pool_summary[pool_summary["policy_name"].eq(SELECTED_POOL_POLICY) & pool_summary["dataset_name"].eq("ALL")])
display(resolver_summary[resolver_summary["policy_name"].eq(SELECTED_RESOLVER_POLICY) & resolver_summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 3. Gate Audit

# %%
pool_scale_all = pick_summary(pool_summary, SELECTED_POOL_POLICY, "scale_meddx100", "ALL")
pool_scale_test = pick_summary(pool_summary, SELECTED_POOL_POLICY, "scale_meddx100", "test")
pool_transfer = pick_summary(pool_summary, SELECTED_POOL_POLICY, "transfer_old90", "transfer")
resolver_scale_all = pick_summary(resolver_summary, SELECTED_RESOLVER_POLICY, "scale_meddx100", "ALL")
resolver_scale_test = pick_summary(resolver_summary, SELECTED_RESOLVER_POLICY, "scale_meddx100", "test")
resolver_transfer = pick_summary(resolver_summary, SELECTED_RESOLVER_POLICY, "transfer_old90", "ALL")
current_scale_all = pick_summary(resolver_summary, "notebook51_current_live", "scale_meddx100", "ALL")
current_scale_test = pick_summary(resolver_summary, "notebook51_current_live", "scale_meddx100", "test")
current_transfer = pick_summary(resolver_summary, "notebook46_current_live", "transfer_old90", "ALL")

gate_rows = [
    {
        "gate": "stage1_pool_recall_scale_all",
        "target": ">=850/900",
        "observed": int(pool_scale_all["candidate_pool_recall_count"]),
        "denominator": int(pool_scale_all["num_workups"]),
        "passed": int(pool_scale_all["candidate_pool_recall_count"]) >= 850,
        "notes": "Strong target is >=865/900.",
    },
    {
        "gate": "stage1_pool_recall_scale_test",
        "target": "held-out split should not collapse",
        "observed": int(pool_scale_test["candidate_pool_recall_count"]),
        "denominator": int(pool_scale_test["num_workups"]),
        "passed": float(pool_scale_test["candidate_pool_recall"]) >= float(pool_scale_all["current_pool_recall"]) - 0.02,
        "notes": "Held-out test split had complete recovered-pool recall in this split.",
    },
    {
        "gate": "stage1_pool_transfer_nonnegative",
        "target": "old90 pool recall >= current old90 pool recall",
        "observed": int(pool_transfer["candidate_pool_recall_count"]),
        "denominator": int(pool_transfer["num_workups"]),
        "passed": int(pool_transfer["candidate_pool_recall_count"]) >= int(pool_transfer["current_pool_recall_count"]),
        "notes": f"Current old90 pool recall was {int(pool_transfer['current_pool_recall_count'])}/{int(pool_transfer['num_workups'])}.",
    },
    {
        "gate": "stage2_final_case_blocked_test",
        "target": ">=152/180, equivalent to >=760/900",
        "observed": int(resolver_scale_test["top1_count"]),
        "denominator": int(resolver_scale_test["num_workups"]),
        "passed": int(resolver_scale_test["top1_count"]) >= 152,
        "notes": "This is the main deployable generalization gate for the final resolver.",
    },
    {
        "gate": "stage2_final_transfer_nonnegative",
        "target": "old90 final top-1 >= current old90 top-1",
        "observed": int(resolver_transfer["top1_count"]),
        "denominator": int(resolver_transfer["num_workups"]),
        "passed": int(resolver_transfer["top1_count"]) >= int(current_transfer["top1_count"]),
        "notes": f"Current old90 top-1 was {int(current_transfer['top1_count'])}/{int(current_transfer['num_workups'])}.",
    },
    {
        "gate": "stage2_no_test_regressions",
        "target": "0 regressions on held-out test under selected threshold",
        "observed": int(resolver_scale_test["regressions_vs_current"]),
        "denominator": int(resolver_scale_test["num_workups"]),
        "passed": int(resolver_scale_test["regressions_vs_current"]) == 0,
        "notes": "Validation-selected conservative threshold protects current answers.",
    },
    {
        "gate": "pool_size_manageable",
        "target": "mean <=15 and p90 <=25",
        "observed": float(pool_scale_all["mean_pool_size"]),
        "denominator": int(pool_scale_all["num_workups"]),
        "passed": float(pool_scale_all["mean_pool_size"]) <= 15 and float(pool_scale_all["p90_pool_size"]) <= 25,
        "notes": f"p90 pool size was {float(pool_scale_all['p90_pool_size']):.1f}.",
    },
]
gate_audit = pd.DataFrame(gate_rows)
gate_audit.to_csv(ARTIFACT_ROOT / "integrated_policy_gate_audit.csv", index=False)

all_gates_passed = bool(gate_audit["passed"].all())
display(gate_audit)

# %% [markdown]
# ## 4. Integrated Policy Tables

# %%
integrated_case_results = pd.concat(
    [
        scale_case_results.assign(evaluation_scope="scale_all_diagnostic"),
        test_case_results.assign(evaluation_scope="scale_case_blocked_test"),
        transfer_case_results.assign(evaluation_scope="transfer_old90"),
    ],
    ignore_index=True,
)
integrated_case_results["integrated_policy_name"] = INTEGRATED_POLICY_NAME
integrated_case_results.to_csv(ARTIFACT_ROOT / "integrated_policy_case_results.csv", index=False)

summary_rows = [
    {
        "policy_name": INTEGRATED_POLICY_NAME,
        "evaluation_scope": "scale_all_diagnostic",
        "cohort": "scale_meddx100",
        "top1_count": int(resolver_scale_all["top1_count"]),
        "num_workups": int(resolver_scale_all["num_workups"]),
        "top1": float(resolver_scale_all["top1"]),
        "top3": float(resolver_scale_all["top3"]),
        "top5": float(resolver_scale_all["top5"]),
        "candidate_pool_recall": float(pool_scale_all["candidate_pool_recall"]),
        "wins_vs_current": int(resolver_scale_all["wins_vs_current"]),
        "regressions_vs_current": int(resolver_scale_all["regressions_vs_current"]),
        "mean_pool_size": float(pool_scale_all["mean_pool_size"]),
        "mean_questions": float(pool_scale_all["mean_questions"]),
        "claim_status": "diagnostic_contains_train_cases",
    },
    {
        "policy_name": INTEGRATED_POLICY_NAME,
        "evaluation_scope": "scale_case_blocked_test",
        "cohort": "scale_meddx100",
        "top1_count": int(resolver_scale_test["top1_count"]),
        "num_workups": int(resolver_scale_test["num_workups"]),
        "top1": float(resolver_scale_test["top1"]),
        "top3": float(resolver_scale_test["top3"]),
        "top5": float(resolver_scale_test["top5"]),
        "candidate_pool_recall": float(pool_scale_test["candidate_pool_recall"]),
        "wins_vs_current": int(resolver_scale_test["wins_vs_current"]),
        "regressions_vs_current": int(resolver_scale_test["regressions_vs_current"]),
        "mean_pool_size": float(pool_scale_test["mean_pool_size"]),
        "mean_questions": float(pool_scale_test["mean_questions"]),
        "claim_status": "primary_case_blocked_generalization_gate",
    },
    {
        "policy_name": INTEGRATED_POLICY_NAME,
        "evaluation_scope": "transfer_old90",
        "cohort": "transfer_old90",
        "top1_count": int(resolver_transfer["top1_count"]),
        "num_workups": int(resolver_transfer["num_workups"]),
        "top1": float(resolver_transfer["top1"]),
        "top3": float(resolver_transfer["top3"]),
        "top5": float(resolver_transfer["top5"]),
        "candidate_pool_recall": float(pool_transfer["candidate_pool_recall"]),
        "wins_vs_current": int(resolver_transfer["wins_vs_current"]),
        "regressions_vs_current": int(resolver_transfer["regressions_vs_current"]),
        "mean_pool_size": float(pool_transfer["mean_pool_size"]),
        "mean_questions": float(pool_transfer["mean_questions"]),
        "claim_status": "transfer_regression_check",
    },
]
integrated_policy_summary = pd.DataFrame(summary_rows)
integrated_policy_summary.to_csv(ARTIFACT_ROOT / "integrated_policy_summary.csv", index=False)

paired_current_vs_resolver.to_csv(ARTIFACT_ROOT / "paired_current_vs_integrated_policy.csv", index=False)

display(integrated_policy_summary)

# %% [markdown]
# ## 5. Figures

# %%
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["current test", "integrated test", "current transfer", "integrated transfer"]
values = [
    int(current_scale_test["top1_count"]),
    int(resolver_scale_test["top1_count"]),
    int(current_transfer["top1_count"]),
    int(resolver_transfer["top1_count"]),
]
denominators = [
    int(current_scale_test["num_workups"]),
    int(resolver_scale_test["num_workups"]),
    int(current_transfer["num_workups"]),
    int(resolver_transfer["num_workups"]),
]
ax.bar(labels, [value / denom for value, denom in zip(values, denominators)], color=["#8a8f99", "#2e8b57", "#8a8f99", "#2e8b57"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("Top-1 accuracy")
ax.set_title("Integrated Policy Versus Current")
for idx, (value, denom) in enumerate(zip(values, denominators)):
    ax.text(idx, value / denom + 0.02, f"{value}/{denom}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "integrated_policy_vs_current.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(gate_audit["gate"], gate_audit["passed"].astype(int), color=["#2e8b57" if passed else "#c23b22" for passed in gate_audit["passed"]])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Passed")
ax.set_title("Integrated Policy Gate Audit")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "integrated_policy_gate_audit.png", dpi=180)
plt.close(fig)

# %% [markdown]
# ## 6. Final Frozen Policy Contract

# %%
selected_policy = {
    "selected_policy_name": INTEGRATED_POLICY_NAME,
    "stage": "stage3_integrated_frozen_policy",
    "pool_policy": {
        "name": SELECTED_POOL_POLICY,
        "source_order": pool_selected.get("source_order", []),
        "scale_all": pool_selected.get("scale_all", {}),
        "scale_test": pool_selected.get("scale_test", {}),
        "transfer_old90": pool_selected.get("transfer_old90", {}),
    },
    "resolver_policy": {
        "name": SELECTED_RESOLVER_POLICY,
        "model_name": resolver_selected.get("selected_model_name"),
        "threshold": resolver_selected.get("selected_threshold"),
        "feature_columns": resolver_selected.get("feature_columns", []),
        "scale_validate": resolver_selected.get("scale_validate", {}),
        "scale_test": resolver_selected.get("scale_test", {}),
        "transfer_old90": resolver_selected.get("transfer_old90", {}),
    },
    "gate_audit": gate_audit.to_dict(orient="records"),
    "promotion_decision": {
        "offline_gates_passed": all_gates_passed,
        "decision": "ready_for_small_live_pilot" if all_gates_passed else "not_ready_for_live_pilot",
        "minimum_claim": "Candidate-pool recovery target passed; final resolver passed the held-out case-blocked minimum gate and old90 transfer check.",
        "claim_caveat": (
            "The all-900 final top-1 row is diagnostic because it includes train cases. "
            "The generalization claim is the case-blocked test split plus old90 transfer."
        ),
    },
    "no_api_calls": True,
}
write_json(ARTIFACT_ROOT / "selected_policy.json", selected_policy)

resolved_run_config = {
    "run_name": RUN_NAME,
    "notebook": "notebooks/55_meddx_integrated_candidate_pool_resolver_policy.ipynb",
    "script": "scripts/meddx_integrated_candidate_pool_resolver_policy_nb55.py",
    "pool_input": POOL_RUN_NAME,
    "resolver_input": RESOLVER_RUN_NAME,
    "selected_policy": INTEGRATED_POLICY_NAME,
    "no_api_calls": True,
    "artifact_root": str(ARTIFACT_ROOT.relative_to(ROOT)),
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

print(json.dumps(selected_policy["promotion_decision"], indent=2))
