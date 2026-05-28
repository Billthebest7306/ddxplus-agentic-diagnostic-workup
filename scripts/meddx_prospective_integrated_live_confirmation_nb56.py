from __future__ import annotations

# %% [markdown]
# # Notebook 56: Prospective Integrated MEDDx Live Confirmation
#
# This notebook is the prospective confirmation runner for the frozen MEDDx integrated policy.
#
# It intentionally separates three ideas:
#
# 1. run the already-frozen MEDDx-aligned live workup architecture from Notebook 51
# 2. sample a fresh prospective cohort, excluding prior artifact case IDs when possible
# 3. apply the frozen Notebook 55 candidate-pool/recovered-resolver stack without reselecting thresholds
#
# Default mode is a cheap dry-run smoke. For the real confirmation run, set `RUN_LIVE_API=True`.

# %%
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

pd.set_option("display.max_columns", 180)
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

# %% [markdown]
# ## 1. Frozen Prospective Run Controls

# %%
# Switch this to True for the paid prospective run.
RUN_LIVE_API = False
ALLOW_DRY_RUN_BENCHMARK = True
RESUME_IF_AVAILABLE = True

# Keep API/model settings as notebook variables, not environment variables.
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4.1-mini"
LLM_API_KEY = ""
INTERACTIVE_API_KEY_BOOTSTRAP = True
TEMPERATURE = 0.0
TOP_P = 1.0

# Prospective confirmation shape:
# - live: 10 cases per dataset x 3 datasets x 3 budgets = 90 workups
# - dry: 1 case per dataset x 3 datasets x 3 budgets = 9 workups
RUN_VERSION_SUFFIX = "v1_prospective90"
RANDOM_SEED = 560
MEDDX_REFERENCE_BUDGETS = [5, 10, 15]
LIVE_BUDGETS_TO_RUN = [5, 10, 15]
DRY_RUN_ALL_BUDGETS = True
LIVE_CASES_PER_DATASET = 10
DRY_RUN_CASES_PER_DATASET = 1
LIVE_TOTAL_MAX_CASES = 30
DRY_RUN_TOTAL_MAX_CASES = 3
ENABLED_DATASETS = ["ddxplus", "icraft_md", "rarebench"]
REQUIRE_ALL_ENABLED_DATASETS = True

# Load more than the final cohort size so prior artifacts can be excluded before balanced sampling.
EXCLUDE_PRIOR_ARTIFACT_CASE_IDS = True
LIVE_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET = 250
DRY_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET = 50
PRIOR_EXCLUSION_RUN_NAMES = [
    "meddx_scale_hypothesis_branching_confirmation_v1_meddx100",
    "meddx_aligned_dataset_native_driver_v1_eval30",
    "meddx_prospective_integrated_live_confirmation_v1_prospective90",
]

BASE_DRIVER_SCRIPT = ROOT / "scripts" / "meddx_scale_hypothesis_branching_confirmation_nb51.py"
POOL_RECOVERY_SCRIPT = ROOT / "scripts" / "meddx_candidate_pool_recovery_lab_nb53.py"

FROZEN_POOL_POLICY = "saved_sources_plus_ddx_bayes10_plus_visible_rare_hpo10_v1"
FROZEN_RESOLVER_POLICY = "recovered_pool_logistic_evidence_card_resolver_v1"
FROZEN_INTEGRATED_POLICY = "integrated_recovered_pool_evidence_card_policy_v1"
FROZEN_RESOLVER_ARTIFACT = ROOT / "artifacts" / "universal_meddx" / "meddx_evidence_card_resolver_lab_v1"

PROSPECTIVE_COHORT_NAME = "prospective90"
PROSPECTIVE_POOL_RUN_NAME_STEM = "candidate_pool"
PROSPECTIVE_FINAL_RUN_NAME_STEM = "frozen_policy"

print("Project root       :", ROOT)
print("Run live API       :", RUN_LIVE_API)
print("Live workups target:", LIVE_TOTAL_MAX_CASES * len(LIVE_BUDGETS_TO_RUN))
print("Dry workups target :", DRY_RUN_TOTAL_MAX_CASES * len(MEDDX_REFERENCE_BUDGETS))
print("Model              :", LLM_MODEL)
print("Temperature/top_p  :", TEMPERATURE, TOP_P)
print("Seed               :", RANDOM_SEED)
print("Exclude prior cases:", EXCLUDE_PRIOR_ARTIFACT_CASE_IDS)

# %% [markdown]
# ## 2. Execute Frozen Live Driver

# %%
def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}, found {count}.")
    return source.replace(old, new, 1)


driver_source = BASE_DRIVER_SCRIPT.read_text(encoding="utf-8")
driver_replacements = {
    'RUN_LIVE_API = False': 'RUN_LIVE_API = CONFIRMATION_RUN_LIVE_API',
    'ALLOW_DRY_RUN_BENCHMARK = True': 'ALLOW_DRY_RUN_BENCHMARK = CONFIRMATION_ALLOW_DRY_RUN_BENCHMARK',
    'RESUME_IF_AVAILABLE = True': 'RESUME_IF_AVAILABLE = CONFIRMATION_RESUME_IF_AVAILABLE',
    'LLM_BASE_URL = "https://api.openai.com/v1"': 'LLM_BASE_URL = CONFIRMATION_LLM_BASE_URL',
    'LLM_MODEL = "gpt-4.1-mini"': 'LLM_MODEL = CONFIRMATION_LLM_MODEL',
    'LLM_API_KEY = ""': 'LLM_API_KEY = CONFIRMATION_LLM_API_KEY',
    'INTERACTIVE_API_KEY_BOOTSTRAP = True': 'INTERACTIVE_API_KEY_BOOTSTRAP = CONFIRMATION_INTERACTIVE_API_KEY_BOOTSTRAP',
    'TEMPERATURE = 0.0': 'TEMPERATURE = CONFIRMATION_TEMPERATURE',
    'TOP_P = 1.0': 'TOP_P = CONFIRMATION_TOP_P',
    'RUN_VERSION_SUFFIX = "v1_meddx100"': 'RUN_VERSION_SUFFIX = CONFIRMATION_RUN_VERSION_SUFFIX',
    'RANDOM_SEED = 42': 'RANDOM_SEED = CONFIRMATION_RANDOM_SEED',
    'MEDDX_SAMPLE_MODE = "meddxagent_seed42_shuffle_first_n"': 'MEDDX_SAMPLE_MODE = "prospective_seed560_shuffle_first_n_excluding_prior_artifacts"',
    'MEDDX_REFERENCE_BUDGETS = [5, 10, 15]': 'MEDDX_REFERENCE_BUDGETS = CONFIRMATION_MEDDX_REFERENCE_BUDGETS',
    'LIVE_BUDGETS_TO_RUN = [5, 10, 15]': 'LIVE_BUDGETS_TO_RUN = CONFIRMATION_LIVE_BUDGETS_TO_RUN',
    'DRY_RUN_ALL_BUDGETS = True': 'DRY_RUN_ALL_BUDGETS = CONFIRMATION_DRY_RUN_ALL_BUDGETS',
    'LIVE_CASES_PER_DATASET = 100': 'LIVE_CASES_PER_DATASET = CONFIRMATION_LIVE_CASES_PER_DATASET',
    'DRY_RUN_CASES_PER_DATASET = 1': 'DRY_RUN_CASES_PER_DATASET = CONFIRMATION_DRY_RUN_CASES_PER_DATASET',
    'LIVE_TOTAL_MAX_CASES = 300': 'LIVE_TOTAL_MAX_CASES = CONFIRMATION_LIVE_TOTAL_MAX_CASES',
    'DRY_RUN_TOTAL_MAX_CASES = 3': 'DRY_RUN_TOTAL_MAX_CASES = CONFIRMATION_DRY_RUN_TOTAL_MAX_CASES',
    'ENABLED_DATASETS = ["ddxplus", "icraft_md", "rarebench"]': 'ENABLED_DATASETS = CONFIRMATION_ENABLED_DATASETS',
    'REQUIRE_ALL_ENABLED_DATASETS = True': 'REQUIRE_ALL_ENABLED_DATASETS = CONFIRMATION_REQUIRE_ALL_ENABLED_DATASETS',
    'RUN_NAME_BASE = f"meddx_scale_hypothesis_branching_confirmation_{RUN_VERSION_SUFFIX}"': 'RUN_NAME_BASE = f"meddx_prospective_integrated_live_confirmation_{RUN_VERSION_SUFFIX}"',
    'RUN_NAME = RUN_NAME_BASE if RUN_LIVE_API else f"meddx_scale_hypothesis_branching_confirmation_dryrun_smoke_{RUN_VERSION_SUFFIX}"': 'RUN_NAME = RUN_NAME_BASE if RUN_LIVE_API else f"meddx_prospective_integrated_live_confirmation_dryrun_smoke_{RUN_VERSION_SUFFIX}"',
}
for old, new in driver_replacements.items():
    driver_source = replace_once(driver_source, old, new)

adapter_block_old = '''adapter_load_cap = LIVE_CASES_PER_DATASET if RUN_LIVE_API else DRY_RUN_CASES_PER_DATASET
adapter_results: list[AdapterResult] = []
if "ddxplus" in ENABLED_DATASETS:
    adapter_results.append(load_ddxplus_adapter(adapter_load_cap))
if "icraft_md" in ENABLED_DATASETS:
    adapter_results.append(load_icraft_md_adapter(adapter_load_cap))
if "rarebench" in ENABLED_DATASETS:
    adapter_results.append(load_rarebench_adapter(adapter_load_cap))

adapter_preflight = pd.DataFrame(['''

adapter_block_new = '''adapter_load_cap = (
    CONFIRMATION_LIVE_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET
    if RUN_LIVE_API
    else CONFIRMATION_DRY_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET
)
adapter_results: list[AdapterResult] = []
if "ddxplus" in ENABLED_DATASETS:
    adapter_results.append(load_ddxplus_adapter(adapter_load_cap))
if "icraft_md" in ENABLED_DATASETS:
    adapter_results.append(load_icraft_md_adapter(adapter_load_cap))
if "rarebench" in ENABLED_DATASETS:
    adapter_results.append(load_rarebench_adapter(adapter_load_cap))


def confirmation_load_prior_case_ids(root: Path, run_names: list[str]) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    for run_name in run_names:
        run_root = root / "artifacts" / "universal_meddx" / run_name
        for filename in ["universal_cases.csv", "predictions.csv"]:
            path = run_root / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if {"dataset_name", "case_id"}.issubset(frame.columns):
                excluded.update(zip(frame["dataset_name"].astype(str), frame["case_id"].astype(str)))
    return excluded


if CONFIRMATION_EXCLUDE_PRIOR_ARTIFACT_CASE_IDS:
    confirmation_excluded_case_ids = confirmation_load_prior_case_ids(ROOT, CONFIRMATION_PRIOR_EXCLUSION_RUN_NAMES)
    filtered_results: list[AdapterResult] = []
    for result in adapter_results:
        before_count = len(result.cases)
        kept_cases = [
            case
            for case in result.cases
            if (str(case.dataset_name), str(case.case_id)) not in confirmation_excluded_case_ids
        ]
        filtered_results.append(
            AdapterResult(
                dataset_name=result.dataset_name,
                status=result.status,
                cases=kept_cases,
                message=f"{result.message} Filtered prospective exclusions: {before_count - len(kept_cases)} removed.",
                source_path=result.source_path,
            )
        )
    adapter_results = filtered_results

adapter_preflight = pd.DataFrame(['''
driver_source = replace_once(driver_source, adapter_block_old, adapter_block_new)

CONFIRMATION_RUN_LIVE_API = RUN_LIVE_API
CONFIRMATION_ALLOW_DRY_RUN_BENCHMARK = ALLOW_DRY_RUN_BENCHMARK
CONFIRMATION_RESUME_IF_AVAILABLE = RESUME_IF_AVAILABLE
CONFIRMATION_LLM_BASE_URL = LLM_BASE_URL
CONFIRMATION_LLM_MODEL = LLM_MODEL
CONFIRMATION_LLM_API_KEY = LLM_API_KEY
CONFIRMATION_INTERACTIVE_API_KEY_BOOTSTRAP = INTERACTIVE_API_KEY_BOOTSTRAP
CONFIRMATION_TEMPERATURE = TEMPERATURE
CONFIRMATION_TOP_P = TOP_P
CONFIRMATION_RUN_VERSION_SUFFIX = RUN_VERSION_SUFFIX
CONFIRMATION_RANDOM_SEED = RANDOM_SEED
CONFIRMATION_MEDDX_REFERENCE_BUDGETS = MEDDX_REFERENCE_BUDGETS
CONFIRMATION_LIVE_BUDGETS_TO_RUN = LIVE_BUDGETS_TO_RUN
CONFIRMATION_DRY_RUN_ALL_BUDGETS = DRY_RUN_ALL_BUDGETS
CONFIRMATION_LIVE_CASES_PER_DATASET = LIVE_CASES_PER_DATASET
CONFIRMATION_DRY_RUN_CASES_PER_DATASET = DRY_RUN_CASES_PER_DATASET
CONFIRMATION_LIVE_TOTAL_MAX_CASES = LIVE_TOTAL_MAX_CASES
CONFIRMATION_DRY_RUN_TOTAL_MAX_CASES = DRY_RUN_TOTAL_MAX_CASES
CONFIRMATION_ENABLED_DATASETS = ENABLED_DATASETS
CONFIRMATION_REQUIRE_ALL_ENABLED_DATASETS = REQUIRE_ALL_ENABLED_DATASETS
CONFIRMATION_LIVE_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET = LIVE_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET
CONFIRMATION_DRY_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET = DRY_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET
CONFIRMATION_EXCLUDE_PRIOR_ARTIFACT_CASE_IDS = EXCLUDE_PRIOR_ARTIFACT_CASE_IDS
CONFIRMATION_PRIOR_EXCLUSION_RUN_NAMES = PRIOR_EXCLUSION_RUN_NAMES

exec(compile(driver_source, str(BASE_DRIVER_SCRIPT), "exec"), globals())

CONFIRMATION_LIVE_RUN_NAME = RUN_NAME
CONFIRMATION_LIVE_ARTIFACT_ROOT = ARTIFACT_ROOT
print("Frozen live-driver artifact root:", CONFIRMATION_LIVE_ARTIFACT_ROOT)

# %% [markdown]
# ## 3. Materialize Frozen Candidate Pool

# %%
if not (CONFIRMATION_LIVE_ARTIFACT_ROOT / "predictions.csv").exists():
    raise FileNotFoundError("The live-driver step did not produce predictions.csv.")

pool_source = POOL_RECOVERY_SCRIPT.read_text(encoding="utf-8")
pool_source = replace_once(
    pool_source,
    'SCALE_RUN_NAME = "meddx_scale_hypothesis_branching_confirmation_v1_meddx100"',
    'SCALE_RUN_NAME = CONFIRMATION_LIVE_RUN_NAME',
)
pool_source = replace_once(
    pool_source,
    'RUN_NAME = "meddx_candidate_pool_recovery_lab_v1"',
    'RUN_NAME = CONFIRMATION_POOL_RUN_NAME',
)
pool_source = pool_source.replace('"scale_meddx100"', f'"{PROSPECTIVE_COHORT_NAME}"')
pool_final_marker = "# %% [markdown]\n# ## 9. Final Summary And Artifact Contract"
if pool_final_marker in pool_source:
    pool_source = pool_source.split(pool_final_marker, 1)[0]

CONFIRMATION_POOL_RUN_NAME = f"{CONFIRMATION_LIVE_RUN_NAME}_{PROSPECTIVE_POOL_RUN_NAME_STEM}"
exec(compile(pool_source, str(POOL_RECOVERY_SCRIPT), "exec"), globals())

CONFIRMATION_POOL_ARTIFACT_ROOT = ARTIFACT_ROOT
write_json(
    CONFIRMATION_POOL_ARTIFACT_ROOT / "selected_policy.json",
    {
        "selected_policy_name": FROZEN_POOL_POLICY,
        "stage": "prospective_candidate_pool_materialization",
        "base_live_run_name": CONFIRMATION_LIVE_RUN_NAME,
        "cohort_name": PROSPECTIVE_COHORT_NAME,
        "source_script": str(POOL_RECOVERY_SCRIPT.relative_to(ROOT)),
        "final_summary_truncated_for_small_smoke_runs": True,
    },
)
print("Prospective pool artifact root:", CONFIRMATION_POOL_ARTIFACT_ROOT)

# %% [markdown]
# ## 4. Apply Frozen Evidence-Card Resolver

# %%
CONFIRMATION_FINAL_RUN_NAME = f"{CONFIRMATION_LIVE_RUN_NAME}_{PROSPECTIVE_FINAL_RUN_NAME_STEM}"
FINAL_ARTIFACT_ROOT = ROOT / "artifacts" / "universal_meddx" / CONFIRMATION_FINAL_RUN_NAME
FINAL_FIGURE_DIR = FINAL_ARTIFACT_ROOT / "figures"
FINAL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def write_json_clean(path: Path, payload: Any) -> None:
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


def normalize_label_for_policy(value: Any) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("é", "e")
        .replace("è", "e")
        .replace("á", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ï", "i")
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def parse_json_list_for_policy(value: Any) -> list[str]:
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


def parse_json_dict_for_policy(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def boolish_for_policy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float_for_policy(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def rank_in_list_for_policy(label: Any, ranked: list[str], missing_rank: int = 99) -> int:
    key = normalize_label_for_policy(label)
    for idx, item in enumerate(ranked, start=1):
        if normalize_label_for_policy(item) == key:
            return idx
    return missing_rank


def topk_hit_for_policy(ranked: list[str], truth: str, k: int) -> bool:
    truth_key = normalize_label_for_policy(truth)
    return truth_key in {normalize_label_for_policy(label) for label in ranked[:k]}


def unique_ranked_for_policy(labels: list[str], limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = normalize_label_for_policy(label)
        if key and key not in seen:
            seen.add(key)
            out.append(str(label))
    return out[:limit]


def current_ranked_for_policy(row: pd.Series) -> list[str]:
    ranked = parse_json_list_for_policy(row.get("ranked_differential", ""))
    prediction = str(row.get("predicted_diagnosis", "") or "").strip()
    if prediction and normalize_label_for_policy(prediction) not in {normalize_label_for_policy(label) for label in ranked}:
        ranked = [prediction, *ranked]
    return unique_ranked_for_policy(ranked, 10)


SOURCE_NAMES = [
    "current_pool",
    "ranked_diff_top10",
    "llm_diff_top10",
    "ddxplus_mlp_top5",
    "branch_top10",
    "casebase_prior_top1",
    "rarebench_graph_top1",
    "ddxplus_visible_bayes_top10",
    "rarebench_visible_hpo_top10",
]
CASE_KEYS = ["cohort", "dataset_name", "case_id", "budget"]

selected_resolver = json.loads((FROZEN_RESOLVER_ARTIFACT / "selected_policy.json").read_text(encoding="utf-8"))
training_features = pd.read_csv(FROZEN_RESOLVER_ARTIFACT / "candidate_level_evidence_card_features.csv")
feature_columns = list(selected_resolver["feature_columns"])
selected_model_name = str(selected_resolver["selected_model_name"])
selected_threshold = float(selected_resolver["selected_threshold"])

if selected_model_name != "logistic_C2_balanced":
    raise ValueError(f"Notebook 56 is frozen for logistic_C2_balanced; found {selected_model_name!r}.")

prospective_predictions = pd.read_csv(CONFIRMATION_LIVE_ARTIFACT_ROOT / "predictions.csv").assign(cohort=PROSPECTIVE_COHORT_NAME)
prospective_current_candidates = pd.read_csv(CONFIRMATION_LIVE_ARTIFACT_ROOT / "candidate_level_resolver_scores.csv").assign(
    cohort=PROSPECTIVE_COHORT_NAME
)
expanded_candidates = pd.read_csv(CONFIRMATION_POOL_ARTIFACT_ROOT / "expanded_candidate_pool_long.csv")
expanded_candidates = expanded_candidates[
    expanded_candidates["policy_name"].eq(FROZEN_POOL_POLICY)
    & expanded_candidates["cohort"].eq(PROSPECTIVE_COHORT_NAME)
].copy()
pool_case_results = pd.read_csv(CONFIRMATION_POOL_ARTIFACT_ROOT / "case_level_candidate_pool_recovery_results.csv")
pool_case_results = pool_case_results[
    pool_case_results["policy_name"].eq(FROZEN_POOL_POLICY)
    & pool_case_results["cohort"].eq(PROSPECTIVE_COHORT_NAME)
].copy()


def build_prospective_candidate_feature_table() -> pd.DataFrame:
    current_candidates = prospective_current_candidates.copy()
    current_candidates["label_key"] = current_candidates["label"].map(normalize_label_for_policy)
    current_columns = [
        "cohort",
        "dataset_name",
        "case_id",
        "budget",
        "label_key",
        "candidate_rank",
        "resolver_score",
        "support_count",
    ]
    for column in current_columns:
        if column not in current_candidates.columns:
            current_candidates[column] = np.nan
    current_candidates = current_candidates[current_columns].copy()

    prediction_columns = [
        "cohort",
        "dataset_name",
        "case_id",
        "budget",
        "predicted_diagnosis",
        "llm_predicted_diagnosis",
        "casebase_prior_top_label",
        "rarebench_graph_top_label",
        "ddxplus_mlp_top1",
        "ddxplus_mlp_top5",
        "ranked_differential",
        "llm_ranked_differential",
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
        "correct_top1",
        "gtpa_at_3",
        "gtpa_at_5",
        "true_rank",
    ]
    for column in prediction_columns:
        if column not in prospective_predictions.columns:
            prospective_predictions[column] = np.nan

    frame = expanded_candidates.merge(
        prospective_predictions[prediction_columns],
        on=["cohort", "dataset_name", "case_id", "budget"],
        how="left",
    )
    frame = frame.merge(
        current_candidates,
        on=["cohort", "dataset_name", "case_id", "budget", "label_key"],
        how="left",
    )
    for column, default in [("candidate_rank", 99), ("resolver_score", 0.0), ("support_count", 0.0)]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

    for source in SOURCE_NAMES:
        source_flag_column = f"has_source__{source}"
        frame[f"has_{source}"] = pd.to_numeric(frame.get(source_flag_column, 0), errors="coerce").fillna(0).astype(int)
        ranks: list[float] = []
        scores: list[float] = []
        for raw_ranks, raw_scores in zip(frame["source_ranks"], frame["source_scores"]):
            rank_map = parse_json_dict_for_policy(raw_ranks)
            score_map = parse_json_dict_for_policy(raw_scores)
            rank = safe_float_for_policy(rank_map.get(source, 99), 99)
            score = safe_float_for_policy(score_map.get(source, 0.0), 0.0)
            ranks.append(rank)
            scores.append(score)
        frame[f"rank_{source}"] = ranks
        frame[f"rr_{source}"] = [0.0 if rank >= 99 else 1.0 / max(rank, 1.0) for rank in ranks]
        frame[f"score_{source}"] = scores

    for list_column in ["ranked_differential", "llm_ranked_differential", "ddxplus_mlp_top5"]:
        ranks = [rank_in_list_for_policy(label, parse_json_list_for_policy(raw)) for label, raw in zip(frame["label"], frame[list_column])]
        frame[f"{list_column}_candidate_rank"] = ranks
        frame[f"{list_column}_candidate_rr"] = [0.0 if rank >= 99 else 1.0 / rank for rank in ranks]

    for reference_column in [
        "predicted_diagnosis",
        "llm_predicted_diagnosis",
        "casebase_prior_top_label",
        "rarebench_graph_top_label",
        "ddxplus_mlp_top1",
    ]:
        frame[f"is_{reference_column}"] = [
            int(normalize_label_for_policy(label) == normalize_label_for_policy(reference))
            for label, reference in zip(frame["label"], frame[reference_column])
        ]

    group_cols = ["cohort", "dataset_name", "case_id", "budget"]
    frame["workup_id"] = (
        frame["cohort"].astype(str)
        + "|"
        + frame["dataset_name"].astype(str)
        + "|"
        + frame["case_id"].astype(str)
        + "|"
        + frame["budget"].astype(int).astype(str)
    )
    frame["case_group"] = frame["cohort"].astype(str) + "|" + frame["dataset_name"].astype(str) + "|" + frame["case_id"].astype(str)
    frame["pool_size"] = frame.groupby(group_cols)["label"].transform("count")
    frame["added_rank_inv"] = 1.0 / pd.to_numeric(frame["candidate_rank_added_order"], errors="coerce").fillna(99).clip(lower=1)
    frame["current_rank_inv"] = 1.0 / pd.to_numeric(frame["candidate_rank"], errors="coerce").fillna(99).clip(lower=1)
    frame["is_current_prediction"] = (frame["candidate_rank"].astype(float) == 1.0).astype(int)
    frame["source_count"] = frame[[f"has_{source}" for source in SOURCE_NAMES]].sum(axis=1)

    for column in [
        "resolver_score",
        "score_ranked_diff_top10",
        "score_llm_diff_top10",
        "score_branch_top10",
        "score_ddxplus_visible_bayes_top10",
        "score_rarebench_visible_hpo_top10",
        "source_count",
    ]:
        group = frame.groupby(group_cols)[column]
        min_value = group.transform("min")
        max_value = group.transform("max")
        denom = (max_value - min_value).replace(0, np.nan)
        frame[f"{column}_pool_minmax"] = ((frame[column] - min_value) / denom).fillna(0.0)
        frame[f"{column}_rank_within_pool"] = frame.groupby(group_cols)[column].rank(ascending=False, method="min")
        frame[f"{column}_minus_pool_max"] = frame[column] - max_value

    return frame


prospective_features = build_prospective_candidate_feature_table()

numeric_feature_columns = [column for column in feature_columns if column != "dataset_name"]
categorical_feature_columns = ["dataset_name"]
for frame in [training_features, prospective_features]:
    for column in numeric_feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in categorical_feature_columns:
        if column not in frame.columns:
            frame[column] = "unknown"
        frame[column] = frame[column].fillna("unknown").astype(str)

train_features = training_features[
    training_features["cohort"].eq("scale_meddx100")
    & training_features["split"].eq("train")
].copy()

preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_feature_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feature_columns),
    ]
)
frozen_model = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "classifier",
            LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced", solver="lbfgs"),
        ),
    ]
)
frozen_model.fit(train_features[feature_columns], train_features["is_truth_candidate"].astype(int))

score_column = "frozen_logistic_C2_balanced_score"
prospective_features[score_column] = frozen_model.predict_proba(prospective_features[feature_columns])[:, 1]


def ranked_from_group_for_policy(group: pd.DataFrame, chosen_label: str) -> list[str]:
    labels = [chosen_label]
    labels.extend(
        group.sort_values([score_column, "resolver_score", "source_count"], ascending=[False, False, False])["label"].tolist()
    )
    return unique_ranked_for_policy(labels, 10)


def current_case_results_for_policy(predictions: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        ranked = current_ranked_for_policy(row)
        truth = str(row["ground_truth_diagnosis"])
        rows.append(
            {
                "policy_name": policy_name,
                "cohort": row["cohort"],
                "dataset_name": row["dataset_name"],
                "case_id": row["case_id"],
                "budget": int(row["budget"]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": row["predicted_diagnosis"],
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": boolish_for_policy(row.get("correct_top1", False)),
                "gtpa_at_3": boolish_for_policy(row.get("gtpa_at_3", False)),
                "gtpa_at_5": boolish_for_policy(row.get("gtpa_at_5", False)),
                "true_rank": int(row["true_rank"]) if "true_rank" in row and not pd.isna(row["true_rank"]) else rank_in_list_for_policy(truth, ranked, 11),
                "original_correct_top1": boolish_for_policy(row.get("correct_top1", False)),
                "changed_prediction": False,
                "policy_action": "current",
                "learned_top_label": "",
                "learned_top_score": np.nan,
                "current_candidate_score": np.nan,
                "score_delta_vs_current": np.nan,
                "candidate_pool_has_truth": np.nan,
                "candidate_pool_size": np.nan,
                "num_questions": safe_float_for_policy(row.get("num_questions", np.nan), np.nan),
                "branch_count": safe_float_for_policy(row.get("branch_count", 0), 0),
                "branch_question_count": safe_float_for_policy(row.get("branch_question_count", 0), 0),
            }
        )
    return pd.DataFrame(rows)


def frozen_resolver_policy_results(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(CASE_KEYS, sort=False):
        group = group.copy()
        current_candidates = group[group["is_current_prediction"].eq(1)]
        if not len(current_candidates):
            current_candidates = group.sort_values("candidate_rank_added_order").head(1)
        current_candidate = current_candidates.iloc[0]
        learned_top = group.sort_values([score_column, "resolver_score", "source_count"], ascending=[False, False, False]).iloc[0]
        learned_score = safe_float_for_policy(learned_top.get(score_column, 0.0), 0.0)
        current_score = safe_float_for_policy(current_candidate.get(score_column, 0.0), 0.0)
        score_delta = learned_score - current_score
        choose_learned = (
            normalize_label_for_policy(learned_top["label"]) != normalize_label_for_policy(current_candidate["label"])
            and score_delta >= selected_threshold
        )
        chosen = learned_top if choose_learned else current_candidate
        prediction = str(chosen["label"])
        truth = str(chosen["ground_truth_diagnosis"])
        ranked = ranked_from_group_for_policy(group, prediction)
        rows.append(
            {
                "policy_name": FROZEN_INTEGRATED_POLICY,
                "cohort": key[0],
                "dataset_name": key[1],
                "case_id": key[2],
                "budget": int(key[3]),
                "ground_truth_diagnosis": truth,
                "policy_prediction": prediction,
                "ranked_differential": json.dumps(ranked, ensure_ascii=True),
                "correct_top1": normalize_label_for_policy(prediction) == normalize_label_for_policy(truth),
                "gtpa_at_3": topk_hit_for_policy(ranked, truth, 3),
                "gtpa_at_5": topk_hit_for_policy(ranked, truth, 5),
                "true_rank": rank_in_list_for_policy(truth, ranked, 11),
                "original_correct_top1": boolish_for_policy(chosen.get("correct_top1", False)),
                "changed_prediction": normalize_label_for_policy(prediction) != normalize_label_for_policy(current_candidate["label"]),
                "policy_action": "learned_override" if choose_learned else "preserve_current",
                "learned_top_label": str(learned_top["label"]),
                "learned_top_score": learned_score,
                "current_candidate_score": current_score,
                "score_delta_vs_current": score_delta,
                "candidate_pool_has_truth": bool(group["is_truth_candidate"].max()),
                "candidate_pool_size": int(len(group)),
                "num_questions": safe_float_for_policy(chosen.get("num_questions", np.nan), np.nan),
                "branch_count": safe_float_for_policy(chosen.get("branch_count", 0), 0),
                "branch_question_count": safe_float_for_policy(chosen.get("branch_question_count", 0), 0),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy_for_confirmation(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slices: list[tuple[str, int, pd.DataFrame]] = [("ALL", -1, frame)]
    slices.extend(
        (dataset_name, int(budget), group)
        for (dataset_name, budget), group in frame.groupby(["dataset_name", "budget"], sort=True)
    )
    slices.extend(
        (dataset_name, -1, group)
        for dataset_name, group in frame.groupby("dataset_name", sort=True)
    )
    for dataset_name, budget, group in slices:
        if not len(group):
            continue
        rows.append(
            {
                "policy_name": policy_name,
                "cohort": PROSPECTIVE_COHORT_NAME,
                "dataset_name": dataset_name,
                "budget": int(budget),
                "num_workups": int(len(group)),
                "top1_count": int(group["correct_top1"].sum()),
                "top1": float(group["correct_top1"].mean()),
                "top3_count": int(group["gtpa_at_3"].sum()),
                "top3": float(group["gtpa_at_3"].mean()),
                "top5_count": int(group["gtpa_at_5"].sum()),
                "top5": float(group["gtpa_at_5"].mean()),
                "wins_vs_current": int((group["correct_top1"] & ~group["original_correct_top1"]).sum()),
                "regressions_vs_current": int((~group["correct_top1"] & group["original_correct_top1"]).sum()),
                "changed_predictions": int(group["changed_prediction"].sum()),
                "candidate_pool_recall": float(group["candidate_pool_has_truth"].mean())
                if group["candidate_pool_has_truth"].notna().any()
                else np.nan,
                "mean_candidate_pool_size": float(group["candidate_pool_size"].mean())
                if group["candidate_pool_size"].notna().any()
                else np.nan,
                "mean_questions": float(group["num_questions"].mean()) if "num_questions" in group else np.nan,
                "mean_branch_count": float(group["branch_count"].mean()) if "branch_count" in group else np.nan,
                "mean_branch_questions": float(group["branch_question_count"].mean()) if "branch_question_count" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


current_results = current_case_results_for_policy(prospective_predictions, "prospective_current_live")
integrated_results = frozen_resolver_policy_results(prospective_features)
current_pool_results = pool_case_results[
    [
        "dataset_name",
        "case_id",
        "budget",
        "candidate_pool_has_truth",
        "candidate_pool_size",
        "candidate_pool_truth_rank",
    ]
].copy()
current_pool_results["cohort"] = PROSPECTIVE_COHORT_NAME
integrated_results = integrated_results.merge(
    current_pool_results,
    on=["cohort", "dataset_name", "case_id", "budget"],
    how="left",
    suffixes=("", "_pool_table"),
)
for column in ["candidate_pool_has_truth", "candidate_pool_size"]:
    pool_column = f"{column}_pool_table"
    if pool_column in integrated_results.columns:
        integrated_results[column] = integrated_results[column].combine_first(integrated_results[pool_column])
        integrated_results = integrated_results.drop(columns=[pool_column])

candidate_pool_oracle = integrated_results.copy()
candidate_pool_oracle["policy_name"] = "prospective_candidate_pool_oracle_non_deployable"
candidate_pool_oracle["correct_top1"] = candidate_pool_oracle["candidate_pool_has_truth"].astype(bool)
candidate_pool_oracle["gtpa_at_3"] = candidate_pool_oracle["candidate_pool_has_truth"].astype(bool)
candidate_pool_oracle["gtpa_at_5"] = candidate_pool_oracle["candidate_pool_has_truth"].astype(bool)
candidate_pool_oracle["policy_action"] = np.where(candidate_pool_oracle["candidate_pool_has_truth"], "oracle_truth_in_pool", "oracle_pool_miss")
candidate_pool_oracle["changed_prediction"] = np.nan
candidate_pool_oracle["original_correct_top1"] = current_results["original_correct_top1"].values if len(current_results) == len(candidate_pool_oracle) else candidate_pool_oracle["original_correct_top1"]

summary = pd.concat(
    [
        summarize_policy_for_confirmation(current_results, "prospective_current_live"),
        summarize_policy_for_confirmation(integrated_results, FROZEN_INTEGRATED_POLICY),
        summarize_policy_for_confirmation(candidate_pool_oracle, "prospective_candidate_pool_oracle_non_deployable"),
    ],
    ignore_index=True,
)

prospective_features.to_csv(FINAL_ARTIFACT_ROOT / "prospective_candidate_level_evidence_card_features.csv", index=False)
prospective_features.to_csv(FINAL_ARTIFACT_ROOT / "prospective_candidate_level_resolver_scores.csv", index=False)
pool_case_results.to_csv(FINAL_ARTIFACT_ROOT / "prospective_candidate_pool_results.csv", index=False)
expanded_candidates.to_csv(FINAL_ARTIFACT_ROOT / "prospective_expanded_candidate_pool_long.csv", index=False)
current_results.to_csv(FINAL_ARTIFACT_ROOT / "prospective_current_case_results.csv", index=False)
integrated_results.to_csv(FINAL_ARTIFACT_ROOT / "prospective_integrated_policy_case_results.csv", index=False)
summary.to_csv(FINAL_ARTIFACT_ROOT / "prospective_integrated_policy_summary.csv", index=False)

paired = integrated_results[
    [
        "policy_name",
        "cohort",
        "dataset_name",
        "case_id",
        "budget",
        "ground_truth_diagnosis",
        "policy_prediction",
        "correct_top1",
        "gtpa_at_3",
        "gtpa_at_5",
        "original_correct_top1",
        "changed_prediction",
        "policy_action",
        "learned_top_label",
        "learned_top_score",
        "current_candidate_score",
        "score_delta_vs_current",
        "candidate_pool_has_truth",
        "candidate_pool_size",
        "candidate_pool_truth_rank",
        "num_questions",
        "branch_count",
        "branch_question_count",
    ]
].copy()
paired.to_csv(FINAL_ARTIFACT_ROOT / "paired_live_current_vs_integrated_policy.csv", index=False)

hard_case_audits = {
    "integrated_failures": integrated_results[~integrated_results["correct_top1"]].to_dict(orient="records"),
    "integrated_wins": integrated_results[
        integrated_results["correct_top1"] & ~integrated_results["original_correct_top1"]
    ].to_dict(orient="records"),
    "integrated_regressions": integrated_results[
        ~integrated_results["correct_top1"] & integrated_results["original_correct_top1"]
    ].to_dict(orient="records"),
    "candidate_pool_misses": integrated_results[~integrated_results["candidate_pool_has_truth"].astype(bool)].to_dict(orient="records"),
}
write_json_clean(FINAL_ARTIFACT_ROOT / "hard_case_audits.json", hard_case_audits)

selected_confirmation_policy = {
    "selected_policy_name": FROZEN_INTEGRATED_POLICY,
    "stage": "prospective_live_confirmation",
    "base_live_run_name": CONFIRMATION_LIVE_RUN_NAME,
    "base_live_artifact_root": str(CONFIRMATION_LIVE_ARTIFACT_ROOT.relative_to(ROOT)),
    "candidate_pool_run_name": CONFIRMATION_POOL_RUN_NAME,
    "final_artifact_root": str(FINAL_ARTIFACT_ROOT.relative_to(ROOT)),
    "run_live_api": bool(RUN_LIVE_API),
    "llm_model": LLM_MODEL,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "case_sampling": {
        "seed": RANDOM_SEED,
        "live_cases_per_dataset": LIVE_CASES_PER_DATASET,
        "live_total_max_cases": LIVE_TOTAL_MAX_CASES,
        "budgets": LIVE_BUDGETS_TO_RUN,
        "prior_exclusion_enabled": EXCLUDE_PRIOR_ARTIFACT_CASE_IDS,
        "prior_exclusion_run_names": PRIOR_EXCLUSION_RUN_NAMES,
        "adapter_load_buffer_cases_per_dataset": LIVE_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET if RUN_LIVE_API else DRY_ADAPTER_LOAD_BUFFER_CASES_PER_DATASET,
    },
    "frozen_pool_policy": FROZEN_POOL_POLICY,
    "frozen_resolver_policy": FROZEN_RESOLVER_POLICY,
    "frozen_resolver_training": "Notebook 54 scale_meddx100 train split only",
    "frozen_threshold_selection": "Notebook 54 scale_meddx100 validation split only",
    "selected_model_name": selected_model_name,
    "selected_threshold": selected_threshold,
    "feature_columns": feature_columns,
    "no_recalibration_on_prospective_labels": True,
    "summary": summary.to_dict(orient="records"),
}
write_json_clean(FINAL_ARTIFACT_ROOT / "selected_live_confirmation_policy.json", selected_confirmation_policy)

display(summary[summary["dataset_name"].eq("ALL")])

# %% [markdown]
# ## 5. Figures And Artifact Contract

# %%
plt.style.use("seaborn-v0_8-whitegrid")

if len(summary):
    all_rows = summary[summary["dataset_name"].eq("ALL") & summary["budget"].eq(-1)].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(all_rows["policy_name"], all_rows["top1"], color=["#7a8793", "#416f9f", "#8a6f3d"][: len(all_rows)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Prospective Confirmation Top-1")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FINAL_FIGURE_DIR / "prospective_top1_by_policy.png", dpi=180)
    plt.close(fig)

    topk_rows = summary[
        summary["policy_name"].eq(FROZEN_INTEGRATED_POLICY)
        & summary["dataset_name"].ne("ALL")
        & summary["budget"].ne(-1)
    ].copy()
    if len(topk_rows):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(topk_rows))
        width = 0.25
        ax.bar(x - width, topk_rows["top1"], width=width, label="Top-1")
        ax.bar(x, topk_rows["top3"], width=width, label="Top-3")
        ax.bar(x + width, topk_rows["top5"], width=width, label="Top-5")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row.dataset_name}\nB{int(row.budget)}" for row in topk_rows.itertuples(index=False)])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Hit rate")
        ax.set_title("Prospective Integrated Top-k By Dataset And Budget")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FINAL_FIGURE_DIR / "prospective_topk_by_dataset_budget.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    changed = integrated_results[integrated_results["changed_prediction"]].copy()
    counts = pd.Series(
        {
            "wins": int((changed["correct_top1"] & ~changed["original_correct_top1"]).sum()) if len(changed) else 0,
            "regressions": int((~changed["correct_top1"] & changed["original_correct_top1"]).sum()) if len(changed) else 0,
            "neutral": int((changed["correct_top1"] == changed["original_correct_top1"]).sum()) if len(changed) else 0,
        }
    )
    ax.bar(counts.index, counts.values, color=["#2e8b57", "#c23b22", "#8a8f99"])
    ax.set_ylabel("Changed workups")
    ax.set_title("Frozen Resolver Paired Changes")
    fig.tight_layout()
    fig.savefig(FINAL_FIGURE_DIR / "prospective_paired_wins_regressions.png", dpi=180)
    plt.close(fig)

    pool_rows = summary[
        summary["policy_name"].eq(FROZEN_INTEGRATED_POLICY)
        & summary["dataset_name"].ne("ALL")
        & summary["budget"].ne(-1)
    ].copy()
    if len(pool_rows):
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [f"{row.dataset_name}\nB{int(row.budget)}" for row in pool_rows.itertuples(index=False)]
        ax.bar(labels, pool_rows["candidate_pool_recall"], color="#5c8a3c")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Candidate-pool recall")
        ax.set_title("Prospective Recovered Pool Recall")
        fig.tight_layout()
        fig.savefig(FINAL_FIGURE_DIR / "prospective_candidate_pool_recall.png", dpi=180)
        plt.close(fig)

required_artifacts = [
    "prospective_candidate_pool_results.csv",
    "prospective_expanded_candidate_pool_long.csv",
    "prospective_candidate_level_evidence_card_features.csv",
    "prospective_candidate_level_resolver_scores.csv",
    "prospective_current_case_results.csv",
    "prospective_integrated_policy_case_results.csv",
    "prospective_integrated_policy_summary.csv",
    "paired_live_current_vs_integrated_policy.csv",
    "hard_case_audits.json",
    "selected_live_confirmation_policy.json",
]
missing = [name for name in required_artifacts if not (FINAL_ARTIFACT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing required prospective confirmation artifacts: {missing}")

print("Notebook 56 artifact contract OK")
print("Live driver root:", CONFIRMATION_LIVE_ARTIFACT_ROOT)
print("Pool root       :", CONFIRMATION_POOL_ARTIFACT_ROOT)
print("Final root      :", FINAL_ARTIFACT_ROOT)
