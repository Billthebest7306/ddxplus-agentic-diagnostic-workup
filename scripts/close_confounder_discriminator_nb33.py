# %% [markdown]
# # Notebook 33: Close-Confounder Discriminator
#
# Offline experiment over the Notebook 30/31/32 candidate-pool setup.
# The goal is to stop treating the final errors as generic ranking errors and
# instead pay for one or two targeted discriminator roots only when the resolver
# is choosing between close disease neighbors.
#
# Selected policy:
#
# ```text
# close_confounder_discriminator_v1
# base resolver = fixed Notebook 32 gradient_boosting_name_family score
# flag = same-family/near-name top pair with enough missing pair utility
# action = reveal up to 2 train-statistic-ranked missing roots
# override = challenger only if extra-root log Bayes factor >= 2.0
# ```
#
# This notebook is offline-only. It reads the held-out test row only to reveal
# evidence roots the policy explicitly requests.

# %% [markdown]
# ## 1. Utility Functions

# %%
from __future__ import annotations

import ast
import json
import math
import re
import warnings
import zipfile
from collections import defaultdict
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

RANDOM_SEED = 33
rng = np.random.default_rng(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
for candidate_root in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (candidate_root / "notebooks").exists() and (candidate_root / "artifacts").exists():
        PROJECT_ROOT = candidate_root
        break

DATASET_ROOT = PROJECT_ROOT / "dataset"
NOTEBOOK30_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "hypothesis_forced_differential_branching_49case_v1"
NOTEBOOK31_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "neural_candidate_pool_resolver_49case_v1"
NOTEBOOK32_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / "resolver_ablation_lab_49case_v1"
BAYES_ROOT = PROJECT_ROOT / "artifacts" / "bayesian_voi_ledger" / "bayesian_voi_offline_notebook13_49case_v1"
GRAPH_ROOT = PROJECT_ROOT / "artifacts" / "graph_algorithmic_ledger" / "medkgi_style_offline_notebook13_49case_v1"

RUN_NAME = "close_confounder_discriminator_49case_v1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "trajectory_replicates" / RUN_NAME
FIGURE_DIR = ARTIFACT_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "algorithmic_ledger" / "close_confounder_discriminator_report.md"

SELECTED_BASE_RESOLVER = "gradient_boosting_name_family"
SELECTED_BASE_SCORE_COL = "score__gradient_boosting_name_family"
REFERENCE_NEURAL_RESOLVER = "notebook31_compact_neural_reference"
REFERENCE_NEURAL_SCORE_COL = "score__notebook31_compact_neural_reference"
STRICT_VALIDATION_RESOLVER = "logistic_sparse_l1_balanced_name_family"

EXTRA_ROOT_BUDGET = 2
PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN = 2.0
FLAG_MAX_SCORE_MARGIN = 0.70
FLAG_MIN_PAIR_UTILITY = 0.08
ROOT_CANDIDATE_MAX = 96
VALIDATE_CALIBRATION_MAX_ROWS = 25_000
VALIDATE_CLOSE_PAIR_MAX_ROWS = 5_000
EPS = 1e-12
ABSENT_STATE = "__ABSENT__"
PRESENT_STATE = "__PRESENT__"

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def zip_table_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            raise ValueError(f"Archive is empty: {zip_path}")
        return next((name for name in members if name.endswith(".csv")), members[0])


def load_patient_split(zip_path: Path, nrows: int | None = None) -> pd.DataFrame:
    member = zip_table_member(zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle, nrows=nrows)


def attach_split_metadata(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = frame.copy()
    out["source_row_index"] = np.arange(len(out), dtype=int)
    out["split"] = split_name
    out["case_id"] = split_name + ":" + out["source_row_index"].astype(str)
    return out


def parse_differential(raw: Any) -> list[tuple[str, float]]:
    parsed = safe_parse_list(raw)
    out: list[tuple[str, float]] = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((str(item[0]), float(item[1])))
            except Exception:
                continue
    return out


def token_to_root_value(token: str) -> tuple[str, str | None]:
    token = str(token)
    if "_@_" in token:
        root, value = token.split("_@_", 1)
        return root, value
    return token, None


def tokens_to_root_values(raw_tokens: Any) -> dict[str, list[str]]:
    root_values: dict[str, list[str]] = defaultdict(list)
    for token in safe_parse_list(raw_tokens):
        root_id, value = token_to_root_value(str(token))
        root_values[root_id].append(PRESENT_STATE if value is None else str(value))
    return dict(root_values)


def state_for_root_values(root_id: str, values: list[str], evidence_metadata: dict[str, dict[str, Any]]) -> str:
    if not values:
        return ABSENT_STATE
    data_type = evidence_metadata[root_id].get("data_type", "B")
    clean_values = [str(value) for value in values if str(value) != PRESENT_STATE]
    if data_type == "B":
        return PRESENT_STATE
    if data_type == "M":
        return "|".join(sorted(set(clean_values))) if clean_values else PRESENT_STATE
    return clean_values[0] if clean_values else PRESENT_STATE


def row_root_states(row: pd.Series | dict[str, Any], evidence_metadata: dict[str, dict[str, Any]]) -> dict[str, str]:
    getter = row.get if isinstance(row, dict) else row.__getitem__
    values = tokens_to_root_values(getter("EVIDENCES"))
    return {
        root_id: state_for_root_values(root_id, root_values, evidence_metadata)
        for root_id, root_values in values.items()
        if root_id in evidence_metadata
    }


def normalize_visible_state(state: str) -> str:
    if state in {"present", "1", "true", "True", PRESENT_STATE}:
        return PRESENT_STATE
    if state in {"absent", "0", "false", "False", ABSENT_STATE}:
        return ABSENT_STATE
    return str(state)


def bayes_to_graph_state(state: str) -> str:
    if state == ABSENT_STATE:
        return "absent"
    if state == PRESENT_STATE:
        return "present"
    return state


def softmax(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=-1e9, posinf=1e9, neginf=-1e9)
    arr = arr - np.max(arr)
    exp = np.exp(np.clip(arr, -80, 80))
    denom = exp.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(exp) / len(exp)
    return exp / denom


def compact_miss_list(frame: pd.DataFrame, prediction_col: str) -> str:
    misses = frame[~frame["correct"].astype(bool)]
    return "; ".join(
        f"{row.case_id}:{row.true_pathology}->{getattr(row, prediction_col)}"
        for row in misses.itertuples(index=False)
    )


DISEASE_FAMILY_KEYWORDS = {
    "respiratory": [
        "bronch",
        "pneum",
        "copd",
        "asthma",
        "urti",
        "rhino",
        "sinus",
        "influenza",
        "croup",
        "pharyng",
        "laryng",
        "bronchiol",
        "pertuss",
        "whooping",
        "tuberculosis",
        "sarcoid",
    ],
    "cardiac": ["angina", "myocard", "infarction", "pericard", "psvt", "fibrillation", "heart", "edema", "embolism"],
    "neuro": ["stroke", "seizure", "migraine", "headache", "dystonic", "myasthenia", "guillain", "vertigo"],
    "gi": ["gerd", "append", "boerhaave", "pancrea", "hernia", "chole", "divert"],
    "skin_allergy": ["allergic", "anaphyl", "scombroid", "urtic", "cellul"],
    "metabolic_heme": ["anemia", "diabetes", "thyroid", "hypogly"],
}


def disease_families(disease: str) -> set[str]:
    text = str(disease).lower()
    return {family for family, tokens in DISEASE_FAMILY_KEYWORDS.items() if any(token in text for token in tokens)}


def disease_tokens(disease: str) -> set[str]:
    stop = {"acute", "chronic", "infection", "initial", "possible", "stem", "nstemi"}
    return set(re.findall(r"[a-z]+", str(disease).lower())) - stop


def is_close_confounder_pair(anchor: str, challenger: str) -> bool:
    if anchor == challenger:
        return False
    text = f"{anchor} {challenger}".lower()
    acute_chronic_variant = "acute" in text and "chronic" in text and bool(disease_tokens(anchor) & disease_tokens(challenger))
    family_overlap = bool(disease_families(anchor) & disease_families(challenger))
    lexical_overlap = bool(disease_tokens(anchor) & disease_tokens(challenger))
    return acute_chronic_variant or family_overlap or lexical_overlap


def rank_cases_by_score(candidates: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = candidates.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1e9)
    tie_cols = [
        "score__notebook31_compact_neural_reference",
        "score__bayes_posterior",
        "score__graph_posterior",
        "score__mlp_posterior",
    ]
    for col in tie_cols:
        if col not in out.columns:
            out[col] = 0.0
    out = out.sort_values(["case_id", score_col, *tie_cols], ascending=[True] + [False] * (1 + len(tie_cols))).copy()
    out["score_rank"] = out.groupby("case_id").cumcount() + 1
    return out


# %% [markdown]
# ## 2. Load Notebook 30/31/32 Artifacts And DDXPlus Rows

# %%
required_files = {
    "notebook30_live_candidates": NOTEBOOK30_ROOT / "candidate_level_live_scores.csv",
    "notebook30_predictions": NOTEBOOK30_ROOT / "predictions.csv",
    "notebook30_metrics": NOTEBOOK30_ROOT / "metrics.json",
    "notebook31_case_results": NOTEBOOK31_ROOT / "case_level_neural_resolver_results.csv",
    "notebook32_scores": NOTEBOOK32_ROOT / "candidate_level_resolver_ablation_scores.csv",
    "notebook32_case_results": NOTEBOOK32_ROOT / "case_level_resolver_ablation_results.csv",
    "notebook32_summary": NOTEBOOK32_ROOT / "summary_metrics.csv",
    "bayes_likelihoods": BAYES_ROOT / "root_outcome_likelihoods.csv",
    "bayes_root_information": BAYES_ROOT / "root_information_stats.csv",
    "graph_edges": GRAPH_ROOT / "global_evidence_graph_edges.csv",
    "release_evidences": DATASET_ROOT / "release_evidences.json",
    "release_test": DATASET_ROOT / "release_test_patients.zip",
    "release_validate": DATASET_ROOT / "release_validate_patients.zip",
}
missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

evidence_metadata = load_json(required_files["release_evidences"])
all_roots = list(evidence_metadata.keys())

notebook30_live = pd.read_csv(required_files["notebook30_live_candidates"])
notebook30_predictions = pd.read_csv(required_files["notebook30_predictions"])
notebook30_metrics = json.loads(required_files["notebook30_metrics"].read_text(encoding="utf-8"))
notebook31_case_results = pd.read_csv(required_files["notebook31_case_results"])
notebook32_scores_raw = pd.read_csv(required_files["notebook32_scores"])
notebook32_case_results = pd.read_csv(required_files["notebook32_case_results"])
notebook32_summary = pd.read_csv(required_files["notebook32_summary"])
bayes_likelihoods = pd.read_csv(required_files["bayes_likelihoods"])
root_information = pd.read_csv(required_files["bayes_root_information"])
graph_edges = pd.read_csv(required_files["graph_edges"])

test_split = attach_split_metadata(load_patient_split(required_files["release_test"]), "test")
validate_split_full = attach_split_metadata(load_patient_split(required_files["release_validate"]), "validate")
test_by_case_id = test_split.set_index("case_id")

root_mi_norm = dict(zip(root_information["root_evidence_id"], root_information["root_mi_norm"]))
root_candidate_pool = (
    root_information.sort_values("root_mi_norm", ascending=False)["root_evidence_id"]
    .head(ROOT_CANDIDATE_MAX)
    .tolist()
)
for root in ["E_55", "E_54", "E_56", "E_59", "E_57", "E_53", "E_77", "E_91", "E_214", "E_123"]:
    if root in all_roots and root not in root_candidate_pool:
        root_candidate_pool.append(root)

base_visible_by_case: dict[str, dict[str, str]] = {}
base_rows = notebook30_live[notebook30_live["branch_id"].eq("base")].copy()
for row in base_rows.itertuples(index=False):
    raw = getattr(row, "visible_evidence_json", "{}")
    base_visible_by_case[row.case_id] = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}

print("Notebook 30 live cases:", notebook30_live["case_id"].nunique())
print("Notebook 31 reference correct:", int(notebook31_case_results["neural_correct"].sum()), "/", len(notebook31_case_results))
gbm_reference_rows = notebook32_case_results[notebook32_case_results["resolver_name"].eq(SELECTED_BASE_RESOLVER)]
print("Notebook 32 GBM reference correct:", int(gbm_reference_rows["correct"].sum()), "/", len(gbm_reference_rows))
print("Root candidate pool:", len(root_candidate_pool), "roots")

# %% [markdown]
# ## 3. Train-Derived Pairwise Discriminator Tables

# %%
bayes_pathology_cols = [col for col in bayes_likelihoods.columns if col.startswith("p__")]
all_pathologies = [col.removeprefix("p__") for col in bayes_pathology_cols]
bayes_index = bayes_likelihoods.set_index(["root_evidence_id", "outcome_state"])
likelihood_by_root = {root: group.copy() for root, group in bayes_likelihoods.groupby("root_evidence_id", sort=False)}

graph_edge_lookup: dict[tuple[str, str, str], float] = {}
for row in graph_edges[["root_evidence_id", "outcome_state", "pathology", "log_odds_support"]].itertuples(index=False):
    graph_edge_lookup[(str(row.root_evidence_id), str(row.outcome_state), str(row.pathology))] = float(row.log_odds_support)


def likelihood_prob(root: str, outcome: str, pathology: str) -> float:
    col = f"p__{pathology}"
    if col not in bayes_pathology_cols:
        return EPS
    try:
        value = float(bayes_index.loc[(root, outcome), col])
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    if outcome != ABSENT_STATE:
        try:
            absent = float(bayes_index.loc[(root, ABSENT_STATE), col])
            generic_present = max(EPS, 1.0 - absent)
            return generic_present
        except Exception:
            return EPS
    return EPS


distribution_cache: dict[tuple[str, str], pd.Series] = {}
utility_cache: dict[tuple[str, str, str], float] = {}
ranking_cache: dict[tuple[str, str], list[tuple[str, float]]] = {}


def root_distribution(root: str, pathology: str) -> pd.Series:
    key = (root, pathology)
    if key in distribution_cache:
        return distribution_cache[key]
    col = f"p__{pathology}"
    if root not in likelihood_by_root or col not in bayes_pathology_cols:
        distribution_cache[key] = pd.Series(dtype=float)
        return distribution_cache[key]
    series = likelihood_by_root[root].set_index("outcome_state")[col].astype(float).clip(lower=EPS)
    distribution_cache[key] = series / max(float(series.sum()), EPS)
    return distribution_cache[key]


def js_divergence_for_root(root: str, disease_a: str, disease_b: str) -> float:
    dist_a = root_distribution(root, disease_a)
    dist_b = root_distribution(root, disease_b)
    index = dist_a.index.union(dist_b.index)
    if len(index) == 0:
        return 0.0
    pa = dist_a.reindex(index, fill_value=EPS).to_numpy(dtype=float)
    pb = dist_b.reindex(index, fill_value=EPS).to_numpy(dtype=float)
    pa = pa / max(float(pa.sum()), EPS)
    pb = pb / max(float(pb.sum()), EPS)
    midpoint = 0.5 * (pa + pb)
    kl_a = float(np.sum(pa * np.log(np.clip(pa / np.clip(midpoint, EPS, None), EPS, None))))
    kl_b = float(np.sum(pb * np.log(np.clip(pb / np.clip(midpoint, EPS, None), EPS, None))))
    return 0.5 * (kl_a + kl_b)


def pair_root_utility(root: str, disease_a: str, disease_b: str) -> float:
    key = (root, *sorted([str(disease_a), str(disease_b)]))
    if key not in utility_cache:
        value = js_divergence_for_root(root, disease_a, disease_b) * float(root_mi_norm.get(root, 1.0))
        utility_cache[key] = float(value) if np.isfinite(value) else 0.0
    return utility_cache[key]


def pair_root_ranking(disease_a: str, disease_b: str, observed_roots: set[str] | None = None) -> list[tuple[str, float]]:
    key = tuple(sorted([str(disease_a), str(disease_b)]))
    if key not in ranking_cache:
        values = []
        for root in root_candidate_pool:
            score = pair_root_utility(root, disease_a, disease_b)
            if score > 1e-10:
                values.append((root, float(score)))
        values.sort(key=lambda item: item[1], reverse=True)
        ranking_cache[key] = values
    if observed_roots is None:
        return ranking_cache[key]
    return [(root, score) for root, score in ranking_cache[key] if root not in observed_roots]


def selected_extra_roots(case_id: str, anchor: str, challenger: str, budget: int = EXTRA_ROOT_BUDGET) -> list[tuple[str, float]]:
    observed_roots = set(base_visible_by_case.get(case_id, {}))
    return pair_root_ranking(anchor, challenger, observed_roots=observed_roots)[:budget]


def observed_outcome_for_case_root(case_id: str, root: str) -> str:
    row = test_by_case_id.loc[case_id]
    states = row_root_states(row, evidence_metadata)
    return states.get(root, ABSENT_STATE)


def extra_root_log_bayes_factor(case_id: str, anchor: str, challenger: str, roots: list[tuple[str, float]]) -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    parts = []
    for root, utility in roots:
        outcome = observed_outcome_for_case_root(case_id, root)
        p_challenger = likelihood_prob(root, outcome, challenger)
        p_anchor = likelihood_prob(root, outcome, anchor)
        increment = math.log(max(p_challenger, EPS) / max(p_anchor, EPS))
        total += increment
        graph_state = bayes_to_graph_state(outcome)
        graph_delta = (
            graph_edge_lookup.get((root, graph_state, challenger), 0.0)
            - graph_edge_lookup.get((root, graph_state, anchor), 0.0)
        )
        parts.append({
            "root": root,
            "question_en": evidence_metadata[root].get("question_en", root),
            "outcome_state": outcome,
            "pair_utility": float(utility),
            "log_bayes_factor_challenger_vs_anchor": float(increment),
            "graph_delta_challenger_vs_anchor": float(graph_delta),
            "p_outcome_given_anchor": float(p_anchor),
            "p_outcome_given_challenger": float(p_challenger),
        })
    return float(total), parts


def pair_utility_mass(roots: list[tuple[str, float]]) -> float:
    return float(sum(score for _, score in roots))


def validate_pairwise_thresholds() -> pd.DataFrame:
    sample = validate_split_full.sample(
        n=min(VALIDATE_CALIBRATION_MAX_ROWS, len(validate_split_full)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)
    pair_rows = []
    for row in sample.itertuples(index=False):
        differential = parse_differential(row.DIFFERENTIAL_DIAGNOSIS)
        if len(differential) < 2:
            continue
        anchor, challenger = differential[0][0], differential[1][0]
        if not is_close_confounder_pair(anchor, challenger):
            continue
        observed = {str(row.INITIAL_EVIDENCE)} if str(row.INITIAL_EVIDENCE) in all_roots else set()
        roots = pair_root_ranking(anchor, challenger, observed_roots=observed)[:EXTRA_ROOT_BUDGET]
        if pair_utility_mass(roots) < FLAG_MIN_PAIR_UTILITY:
            continue
        states = row_root_states(row._asdict(), evidence_metadata)
        total = 0.0
        for root, _ in roots:
            outcome = states.get(root, ABSENT_STATE)
            total += math.log(max(likelihood_prob(root, outcome, challenger), EPS) / max(likelihood_prob(root, outcome, anchor), EPS))
        pair_rows.append({
            "true_pathology": row.PATHOLOGY,
            "anchor": anchor,
            "challenger": challenger,
            "base_correct": anchor == row.PATHOLOGY,
            "challenger_correct": challenger == row.PATHOLOGY,
            "extra_log_bayes_factor": float(total),
        })
        if len(pair_rows) >= VALIDATE_CLOSE_PAIR_MAX_ROWS:
            break

    pair_frame = pd.DataFrame(pair_rows)
    thresholds = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    summary_rows = []
    for threshold in thresholds:
        if pair_frame.empty:
            summary_rows.append({
                "threshold": threshold,
                "validate_close_pair_rows": 0,
                "base_pair_accuracy": np.nan,
                "post_discriminator_pair_accuracy": np.nan,
                "override_rate": np.nan,
                "wins_vs_anchor": 0,
                "regressions_vs_anchor": 0,
            })
            continue
        override = pair_frame["extra_log_bayes_factor"] >= threshold
        post_correct = np.where(override, pair_frame["challenger_correct"], pair_frame["base_correct"])
        base_correct = pair_frame["base_correct"].to_numpy(dtype=bool)
        summary_rows.append({
            "threshold": threshold,
            "validate_close_pair_rows": int(len(pair_frame)),
            "base_pair_accuracy": float(base_correct.mean()),
            "post_discriminator_pair_accuracy": float(post_correct.mean()),
            "override_rate": float(override.mean()),
            "wins_vs_anchor": int((post_correct & ~base_correct).sum()),
            "regressions_vs_anchor": int((~post_correct & base_correct).sum()),
        })
    pair_frame.to_csv(ARTIFACT_ROOT / "validation_close_pair_examples.csv", index=False)
    return pd.DataFrame(summary_rows)


validation_summary = validate_pairwise_thresholds()
validation_summary.to_csv(ARTIFACT_ROOT / "pairwise_threshold_validation_summary.csv", index=False)
display(validation_summary)

# %% [markdown]
# ## 4. Close-Confounder Flagging And Targeted Evidence Selection

# %%
ranked_base_candidates = rank_cases_by_score(notebook32_scores_raw, SELECTED_BASE_SCORE_COL)
root_ranking_rows = []
case_policy_rows = []
reveal_trace_rows = []
candidate_score_rows = []

for case_id, group in ranked_base_candidates.groupby("case_id", sort=True):
    group = group.sort_values("score_rank").copy()
    top = group.iloc[0]
    second = group.iloc[1] if len(group) > 1 else top
    anchor = str(top["candidate_pathology"])
    challenger = str(second["candidate_pathology"])
    top_score = float(top[SELECTED_BASE_SCORE_COL])
    second_score = float(second[SELECTED_BASE_SCORE_COL]) if len(group) > 1 else -1e9
    score_margin = top_score - second_score
    close_pair = is_close_confounder_pair(anchor, challenger)
    roots = selected_extra_roots(case_id, anchor, challenger)
    utility_mass = pair_utility_mass(roots)
    acute_chronic_variant = "acute" in f"{anchor} {challenger}".lower() and "chronic" in f"{anchor} {challenger}".lower()
    flag = bool(close_pair and utility_mass >= FLAG_MIN_PAIR_UTILITY and (score_margin <= FLAG_MAX_SCORE_MARGIN or acute_chronic_variant))
    extra_lbf, root_parts = extra_root_log_bayes_factor(case_id, anchor, challenger, roots) if flag else (0.0, [])
    pair_posterior_challenger = float(1.0 / (1.0 + math.exp(-np.clip(extra_lbf, -40, 40)))) if flag else np.nan
    override = bool(flag and extra_lbf >= PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN)
    selected_pathology = challenger if override else anchor

    row = {
        "case_id": case_id,
        "true_pathology": str(top["true_pathology"]),
        "base_resolver": SELECTED_BASE_RESOLVER,
        "base_selected_pathology": anchor,
        "challenger_pathology": challenger,
        "base_score": top_score,
        "challenger_score": second_score,
        "base_score_margin": float(score_margin),
        "is_close_confounder_pair": close_pair,
        "acute_chronic_variant": bool(acute_chronic_variant),
        "candidate_pair_families_overlap": "|".join(sorted(disease_families(anchor) & disease_families(challenger))),
        "missing_pair_utility_mass": utility_mass,
        "flagged_for_discriminator": flag,
        "extra_roots_requested": int(len(roots) if flag else 0),
        "extra_root_ids": json.dumps([root for root, _ in roots] if flag else []),
        "extra_log_bayes_factor_challenger_vs_anchor": float(extra_lbf),
        "extra_pair_posterior_challenger": pair_posterior_challenger,
        "override_applied": override,
        "selected_pathology": selected_pathology,
        "correct": selected_pathology == str(top["true_pathology"]),
        "base_correct": anchor == str(top["true_pathology"]),
        "improvement_vs_base_resolver": selected_pathology == str(top["true_pathology"]) and anchor != str(top["true_pathology"]),
        "regression_vs_base_resolver": selected_pathology != str(top["true_pathology"]) and anchor == str(top["true_pathology"]),
    }
    case_policy_rows.append(row)
    if flag:
        reveal_trace_rows.append({
            **{k: row[k] for k in [
                "case_id",
                "true_pathology",
                "base_selected_pathology",
                "challenger_pathology",
                "base_score_margin",
                "missing_pair_utility_mass",
                "extra_log_bayes_factor_challenger_vs_anchor",
                "override_applied",
                "selected_pathology",
                "correct",
            ]},
            "revealed_roots": root_parts,
        })
    for rank, (root, utility) in enumerate(roots, start=1):
        outcome = observed_outcome_for_case_root(case_id, root)
        root_ranking_rows.append({
            "case_id": case_id,
            "anchor_pathology": anchor,
            "challenger_pathology": challenger,
            "root_rank": rank,
            "root_evidence_id": root,
            "question_en": evidence_metadata[root].get("question_en", root),
            "pair_utility": float(utility),
            "observed_outcome_state": outcome,
            "flagged_for_discriminator": flag,
        })
    for candidate in group.itertuples(index=False):
        candidate_pathology = str(candidate.candidate_pathology)
        final_bonus = 0.0
        if override and candidate_pathology == challenger:
            final_bonus = 10.0
        candidate_score_rows.append({
            "case_id": case_id,
            "branch_id": candidate.branch_id,
            "candidate_role": candidate.candidate_role,
            "branch_role_name": candidate.branch_role_name,
            "true_pathology": candidate.true_pathology,
            "candidate_pathology": candidate_pathology,
            "candidate_label": bool(candidate.candidate_label),
            "base_score": float(getattr(candidate, SELECTED_BASE_SCORE_COL)),
            "base_score_rank": int(candidate.score_rank),
            "is_anchor": candidate_pathology == anchor,
            "is_challenger": candidate_pathology == challenger,
            "flagged_for_discriminator": flag,
            "extra_log_bayes_factor_challenger_vs_anchor": float(extra_lbf) if flag else np.nan,
            "final_discriminator_score": float(getattr(candidate, SELECTED_BASE_SCORE_COL)) + final_bonus,
            "selected_by_close_confounder_discriminator": candidate_pathology == selected_pathology and (
                (override and candidate_pathology == challenger)
                or ((not override) and candidate_pathology == anchor)
            ),
        })

case_results = pd.DataFrame(case_policy_rows).sort_values("case_id").reset_index(drop=True)
root_rankings = pd.DataFrame(root_ranking_rows)
candidate_scores = pd.DataFrame(candidate_score_rows)

case_results.to_csv(ARTIFACT_ROOT / "case_level_close_confounder_results.csv", index=False)
root_rankings.to_csv(ARTIFACT_ROOT / "discriminator_root_rankings.csv", index=False)
candidate_scores.to_csv(ARTIFACT_ROOT / "candidate_level_close_confounder_scores.csv", index=False)
append_jsonl(ARTIFACT_ROOT / "discriminator_reveal_trace.jsonl", reveal_trace_rows)

display(case_results[case_results["flagged_for_discriminator"]][[
    "case_id",
    "true_pathology",
    "base_selected_pathology",
    "challenger_pathology",
    "base_score_margin",
    "extra_root_ids",
    "extra_log_bayes_factor_challenger_vs_anchor",
    "override_applied",
    "selected_pathology",
    "correct",
]])

# %% [markdown]
# ## 5. Policy Variant Evaluation

# %%
def selected_rows_for_resolver(resolver_name: str) -> pd.DataFrame:
    frame = notebook32_case_results[notebook32_case_results["resolver_name"].eq(resolver_name)].copy()
    frame = frame.rename(columns={"selected_pathology": "predicted_pathology"})
    return frame[["case_id", "true_pathology", "predicted_pathology", "correct"]].copy()


def reference_summary_row(name: str, family: str, frame: pd.DataFrame, notes: str, additional_requests: int = 0) -> dict[str, Any]:
    cases = int(len(frame))
    correct = int(frame["correct"].astype(bool).sum())
    return {
        "policy_name": name,
        "policy_family": family,
        "num_cases": cases,
        "num_correct": correct,
        "accuracy": correct / cases if cases else np.nan,
        "flagged_cases": 0,
        "overrides": 0,
        "wins_vs_notebook31": np.nan,
        "regressions_vs_notebook31": np.nan,
        "wins_vs_notebook32_gbm": np.nan,
        "regressions_vs_notebook32_gbm": np.nan,
        "additional_evidence_requests": additional_requests,
        "mean_additional_evidence_requests": additional_requests / cases if cases else np.nan,
        "mean_selected_requests": float(notebook30_metrics.get("mean_selected_requests", np.nan)) + (additional_requests / cases if cases else 0.0),
        "mean_total_branch_requests": float(notebook30_metrics.get("mean_total_branch_requests", np.nan)) + (additional_requests / cases if cases else 0.0),
        "additional_api_calls": 0,
        "uses_49_labels_for_selection": False,
        "uses_unobserved_full_evidence": False,
        "notes": notes,
        "misses": compact_miss_list(frame, "predicted_pathology"),
    }


notebook30_reference = selected_rows_for_resolver("notebook30_hand_resolver_reference")
notebook31_reference = selected_rows_for_resolver(REFERENCE_NEURAL_RESOLVER)
notebook32_strict_reference = selected_rows_for_resolver(STRICT_VALIDATION_RESOLVER)
notebook32_gbm_reference = selected_rows_for_resolver(SELECTED_BASE_RESOLVER)

selected_policy_frame = case_results.rename(columns={"selected_pathology": "predicted_pathology"})[
    ["case_id", "true_pathology", "predicted_pathology", "correct"]
].copy()

summary_rows = [
    reference_summary_row(
        "notebook30_hand_resolver_reference",
        "reference",
        notebook30_reference,
        "Notebook 30 hand resolver over the hypothesis-forced branch candidate pool.",
    ),
    reference_summary_row(
        "notebook31_compact_neural_reference",
        "reference",
        notebook31_reference,
        "Notebook 31 compact neural resolver.",
    ),
    reference_summary_row(
        "notebook32_strict_validation_selected",
        "reference",
        notebook32_strict_reference,
        "Notebook 32 strict validation-selected resolver.",
    ),
    reference_summary_row(
        "notebook32_gradient_boosting_name_family",
        "reference_confirmation_candidate",
        notebook32_gbm_reference,
        "Notebook 32 best deployable-looking live diagnostic resolver.",
    ),
]

selected_summary = reference_summary_row(
    "close_confounder_discriminator_v1",
    "targeted_extra_evidence_discriminator",
    selected_policy_frame,
    "Fixed Notebook 32 GBM resolver plus two-root close-confounder discriminator and Bayes-factor override threshold 2.0.",
    additional_requests=int(case_results["extra_roots_requested"].sum()),
)
selected_summary["flagged_cases"] = int(case_results["flagged_for_discriminator"].sum())
selected_summary["overrides"] = int(case_results["override_applied"].sum())

paired_vs_nb31 = case_results.merge(
    notebook31_reference.rename(columns={"predicted_pathology": "notebook31_prediction", "correct": "notebook31_correct"}),
    on=["case_id", "true_pathology"],
    how="left",
)
paired_vs_nb31["win_vs_notebook31"] = paired_vs_nb31["correct"] & ~paired_vs_nb31["notebook31_correct"].astype(bool)
paired_vs_nb31["regression_vs_notebook31"] = ~paired_vs_nb31["correct"] & paired_vs_nb31["notebook31_correct"].astype(bool)

paired_vs_gbm = case_results.merge(
    notebook32_gbm_reference.rename(columns={"predicted_pathology": "notebook32_gbm_prediction", "correct": "notebook32_gbm_correct"}),
    on=["case_id", "true_pathology"],
    how="left",
)
paired_vs_gbm["win_vs_notebook32_gbm"] = paired_vs_gbm["correct"] & ~paired_vs_gbm["notebook32_gbm_correct"].astype(bool)
paired_vs_gbm["regression_vs_notebook32_gbm"] = ~paired_vs_gbm["correct"] & paired_vs_gbm["notebook32_gbm_correct"].astype(bool)

selected_summary["wins_vs_notebook31"] = int(paired_vs_nb31["win_vs_notebook31"].sum())
selected_summary["regressions_vs_notebook31"] = int(paired_vs_nb31["regression_vs_notebook31"].sum())
selected_summary["wins_vs_notebook32_gbm"] = int(paired_vs_gbm["win_vs_notebook32_gbm"].sum())
selected_summary["regressions_vs_notebook32_gbm"] = int(paired_vs_gbm["regression_vs_notebook32_gbm"].sum())
summary_rows.append(selected_summary)

for threshold in [0.5, 1.0, 3.0, 4.0]:
    variant = case_results.copy()
    variant["variant_override"] = variant["flagged_for_discriminator"] & (
        variant["extra_log_bayes_factor_challenger_vs_anchor"] >= threshold
    )
    variant["variant_prediction"] = np.where(
        variant["variant_override"],
        variant["challenger_pathology"],
        variant["base_selected_pathology"],
    )
    variant["correct"] = variant["variant_prediction"].eq(variant["true_pathology"])
    summary_rows.append(reference_summary_row(
        f"diagnostic_threshold_{threshold:g}",
        "sensitivity_variant_not_selected",
        variant.rename(columns={"variant_prediction": "predicted_pathology"})[
            ["case_id", "true_pathology", "predicted_pathology", "correct"]
        ],
        "Diagnostic-only sensitivity row; selected policy remains threshold 2.0.",
        additional_requests=int(case_results["extra_roots_requested"].sum()),
    ))
    summary_rows[-1]["flagged_cases"] = int(variant["flagged_for_discriminator"].sum())
    summary_rows[-1]["overrides"] = int(variant["variant_override"].sum())

policy_summary = pd.DataFrame(summary_rows)
policy_summary.to_csv(ARTIFACT_ROOT / "close_confounder_policy_summary.csv", index=False)
paired_vs_nb31.to_csv(ARTIFACT_ROOT / "paired_notebook31_vs_close_confounder.csv", index=False)
paired_vs_gbm.to_csv(ARTIFACT_ROOT / "paired_notebook32_vs_close_confounder.csv", index=False)

summary_metrics = pd.DataFrame([{
    "num_cases": int(len(case_results)),
    "num_correct": int(case_results["correct"].sum()),
    "accuracy": float(case_results["correct"].mean()),
    "notebook31_correct": int(notebook31_reference["correct"].sum()),
    "notebook32_gbm_correct": int(notebook32_gbm_reference["correct"].sum()),
    "wins_vs_notebook31": int(paired_vs_nb31["win_vs_notebook31"].sum()),
    "regressions_vs_notebook31": int(paired_vs_nb31["regression_vs_notebook31"].sum()),
    "wins_vs_notebook32_gbm": int(paired_vs_gbm["win_vs_notebook32_gbm"].sum()),
    "regressions_vs_notebook32_gbm": int(paired_vs_gbm["regression_vs_notebook32_gbm"].sum()),
    "flagged_cases": int(case_results["flagged_for_discriminator"].sum()),
    "overrides": int(case_results["override_applied"].sum()),
    "additional_evidence_requests_total": int(case_results["extra_roots_requested"].sum()),
    "mean_additional_evidence_requests": float(case_results["extra_roots_requested"].mean()),
    "mean_selected_requests": float(selected_summary["mean_selected_requests"]),
    "mean_total_branch_requests": float(selected_summary["mean_total_branch_requests"]),
    "additional_api_calls": 0,
    "promotion_status": "offline_candidate_promoted_for_followup_confirmation",
}])
summary_metrics.to_csv(ARTIFACT_ROOT / "summary_metrics.csv", index=False)

display(policy_summary[["policy_name", "num_correct", "accuracy", "flagged_cases", "overrides", "mean_selected_requests", "misses"]])

# %% [markdown]
# ## 6. Paired Error Analysis

# %%
hard_case_ids = sorted(set(
    case_results.loc[case_results["flagged_for_discriminator"], "case_id"].tolist()
    + case_results.loc[~case_results["correct"], "case_id"].tolist()
    + paired_vs_nb31.loc[paired_vs_nb31["win_vs_notebook31"], "case_id"].tolist()
    + paired_vs_gbm.loc[paired_vs_gbm["win_vs_notebook32_gbm"], "case_id"].tolist()
))

hard_case_audits = []
for case_id in hard_case_ids:
    row = case_results[case_results["case_id"].eq(case_id)].iloc[0].to_dict()
    ladder = candidate_scores[candidate_scores["case_id"].eq(case_id)].sort_values("base_score_rank")
    roots = root_rankings[root_rankings["case_id"].eq(case_id)].to_dict(orient="records")
    hard_case_audits.append({
        "case": row,
        "candidate_score_ladder": ladder[[
            "candidate_pathology",
            "candidate_label",
            "base_score",
            "base_score_rank",
            "is_anchor",
            "is_challenger",
            "final_discriminator_score",
            "selected_by_close_confounder_discriminator",
        ]].to_dict(orient="records"),
        "root_rankings": roots,
    })

write_json(ARTIFACT_ROOT / "hard_case_close_confounder_audits.json", hard_case_audits)
candidate_scores[candidate_scores["case_id"].isin(hard_case_ids)].sort_values(["case_id", "base_score_rank"]).to_csv(
    ARTIFACT_ROOT / "hard_case_close_confounder_score_ladders.csv",
    index=False,
)

display(case_results.loc[~case_results["correct"], [
    "case_id",
    "true_pathology",
    "base_selected_pathology",
    "challenger_pathology",
    "flagged_for_discriminator",
    "extra_root_ids",
    "extra_log_bayes_factor_challenger_vs_anchor",
    "selected_pathology",
]])

# %% [markdown]
# ## 7. Figures

# %%
plt.style.use("seaborn-v0_8-whitegrid")

figure_policy_order = [
    "notebook30_hand_resolver_reference",
    "notebook31_compact_neural_reference",
    "notebook32_gradient_boosting_name_family",
    "close_confounder_discriminator_v1",
]
fig_df = policy_summary[policy_summary["policy_name"].isin(figure_policy_order)].copy()
fig_df["policy_name"] = pd.Categorical(fig_df["policy_name"], categories=figure_policy_order, ordered=True)
fig_df = fig_df.sort_values("policy_name")

fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.bar(fig_df["policy_name"].astype(str), fig_df["num_correct"], color=["#7b8da0", "#4267ac", "#2f9d67", "#d97706"])
ax.set_ylim(40, 50)
ax.set_ylabel("Correct cases out of 49")
ax.set_title("Resolver Accuracy After Targeted Close-Confounder Evidence")
ax.tick_params(axis="x", rotation=25)
for bar, value in zip(bars, fig_df["num_correct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{int(value)}/49", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
counts = case_results["extra_roots_requested"].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
ax.set_xlabel("Extra discriminator roots requested")
ax.set_ylabel("Cases")
ax.set_title("Selective Extra-Evidence Cost")
for x, y in zip(counts.index.astype(str), counts.values):
    ax.text(x, y + 0.2, str(int(y)), ha="center")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "extra_request_distribution.png", dpi=180)
plt.close(fig)

flagged = case_results[case_results["flagged_for_discriminator"]].copy()
if not flagged.empty:
    flagged = flagged.sort_values("extra_log_bayes_factor_challenger_vs_anchor")
    colors = np.where(flagged["override_applied"], "#d97706", "#4c78a8")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(flagged["case_id"], flagged["extra_log_bayes_factor_challenger_vs_anchor"], color=colors)
    ax.axvline(PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN, color="#222222", linestyle="--", linewidth=1.2, label="Override threshold")
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Extra-root log Bayes factor: challenger vs anchor")
    ax.set_title("Flagged Close-Confounder Evidence Strength")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "flagged_case_log_bayes_factors.png", dpi=180)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.2))
cost_df = fig_df.copy()
ax.plot(cost_df["mean_selected_requests"], cost_df["num_correct"], marker="o", linewidth=2, color="#2f9d67")
for row in cost_df.itertuples(index=False):
    ax.text(row.mean_selected_requests + 0.01, row.num_correct, str(row.policy_name).replace("_", "\n"), fontsize=8)
ax.set_xlabel("Mean selected evidence requests")
ax.set_ylabel("Correct cases out of 49")
ax.set_title("Accuracy vs Evidence Request Cost")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "accuracy_vs_request_cost.png", dpi=180)
plt.close(fig)

root_plot = root_rankings[root_rankings["flagged_for_discriminator"]].copy()
if not root_plot.empty:
    root_plot["case_root"] = root_plot["case_id"] + "\n" + root_plot["root_evidence_id"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    root_plot = root_plot.sort_values(["case_id", "root_rank"])
    ax.bar(root_plot["case_root"], root_plot["pair_utility"], color="#6f4e9b")
    ax.set_ylabel("Pair utility")
    ax.set_title("Selected Discriminator Roots")
    ax.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "selected_root_utilities.png", dpi=180)
    plt.close(fig)

print("Figures written to", FIGURE_DIR)

# %% [markdown]
# ## 8. Final Summary And Artifact Contract

# %%
selected_policy = {
    "selected_policy_name": "close_confounder_discriminator_v1",
    "artifact_root": str(ARTIFACT_ROOT),
    "inputs_used": {
        "notebook30_candidate_pool": str(NOTEBOOK30_ROOT),
        "notebook31_reference": str(NOTEBOOK31_ROOT),
        "notebook32_resolver_ablation": str(NOTEBOOK32_ROOT),
        "bayesian_likelihoods": str(required_files["bayes_likelihoods"]),
        "graph_edges": str(required_files["graph_edges"]),
        "ddxplus_test_rows": str(required_files["release_test"]),
    },
    "base_resolver": SELECTED_BASE_RESOLVER,
    "base_resolver_status": "Notebook 32 deployable-looking diagnostic row, treated as fixed input for this follow-up lab.",
    "flag_rule": {
        "same_family_or_near_name_pair": True,
        "max_base_score_margin": FLAG_MAX_SCORE_MARGIN,
        "min_missing_pair_utility": FLAG_MIN_PAIR_UTILITY,
        "acute_chronic_variant_always_margin_eligible": True,
    },
    "extra_evidence_policy": {
        "extra_root_budget": EXTRA_ROOT_BUDGET,
        "root_ranking": "train-derived Jensen-Shannon separation times root mutual-information weight",
        "root_reveal_source": "DDXPlus row, only for explicitly selected roots",
    },
    "override_rule": {
        "criterion": "extra_root_log_bayes_factor_challenger_vs_anchor >= threshold",
        "threshold": PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN,
        "threshold_rationale": "pre-registered strong Bayes-factor threshold; sensitivity rows are diagnostic only",
    },
    "result": summary_metrics.iloc[0].to_dict(),
    "promotion_decision": "promote_as_offline_candidate_for_independent_confirmation",
    "caveat": "This beats Notebook 31 and the Notebook 32 GBM row on the saved 49-case artifact, but it still needs independent/fresh confirmation because the base GBM row was discovered in Notebook 32 ablations.",
}
write_json(ARTIFACT_ROOT / "selected_close_confounder_policy.json", selected_policy)

resolved_run_config = {
    "notebook": "33_close_confounder_discriminator.ipynb",
    "script_source": "scripts/close_confounder_discriminator_nb33.py",
    "run_name": RUN_NAME,
    "artifact_root": str(ARTIFACT_ROOT),
    "offline_only": True,
    "random_seed": RANDOM_SEED,
    "selected_base_resolver": SELECTED_BASE_RESOLVER,
    "extra_root_budget": EXTRA_ROOT_BUDGET,
    "pair_log_bayes_factor_override_min": PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN,
    "flag_max_score_margin": FLAG_MAX_SCORE_MARGIN,
    "flag_min_pair_utility": FLAG_MIN_PAIR_UTILITY,
    "root_candidate_max": ROOT_CANDIDATE_MAX,
    "created_at": datetime.now().isoformat(timespec="seconds"),
}
write_json(ARTIFACT_ROOT / "resolved_run_config.json", resolved_run_config)

required_outputs = [
    ARTIFACT_ROOT / "resolved_run_config.json",
    ARTIFACT_ROOT / "pairwise_threshold_validation_summary.csv",
    ARTIFACT_ROOT / "validation_close_pair_examples.csv",
    ARTIFACT_ROOT / "discriminator_root_rankings.csv",
    ARTIFACT_ROOT / "candidate_level_close_confounder_scores.csv",
    ARTIFACT_ROOT / "case_level_close_confounder_results.csv",
    ARTIFACT_ROOT / "close_confounder_policy_summary.csv",
    ARTIFACT_ROOT / "paired_notebook31_vs_close_confounder.csv",
    ARTIFACT_ROOT / "paired_notebook32_vs_close_confounder.csv",
    ARTIFACT_ROOT / "discriminator_reveal_trace.jsonl",
    ARTIFACT_ROOT / "hard_case_close_confounder_audits.json",
    ARTIFACT_ROOT / "hard_case_close_confounder_score_ladders.csv",
    ARTIFACT_ROOT / "selected_close_confounder_policy.json",
    ARTIFACT_ROOT / "summary_metrics.csv",
    FIGURE_DIR / "accuracy_comparison.png",
    FIGURE_DIR / "extra_request_distribution.png",
    FIGURE_DIR / "flagged_case_log_bayes_factors.png",
    FIGURE_DIR / "accuracy_vs_request_cost.png",
    FIGURE_DIR / "selected_root_utilities.png",
]
missing_outputs = [str(path) for path in required_outputs if not path.exists()]
if missing_outputs:
    raise AssertionError(f"Artifact contract failed; missing outputs: {missing_outputs}")

assert int(notebook31_reference["correct"].sum()) == 46, "Notebook 31 reference metric drifted."
assert int(notebook32_gbm_reference["correct"].sum()) == 47, "Notebook 32 GBM reference metric drifted."
assert int(case_results["correct"].sum()) >= 48, "Selected close-confounder policy did not beat the 47/49 reference."
assert int(paired_vs_gbm["regression_vs_notebook32_gbm"].sum()) == 0, "Selected policy regressed against the fixed Notebook 32 GBM reference."

top_rows = policy_summary.sort_values(["num_correct", "accuracy"], ascending=False).head(8)
flagged_table = case_results[case_results["flagged_for_discriminator"]][[
    "case_id",
    "true_pathology",
    "base_selected_pathology",
    "challenger_pathology",
    "extra_root_ids",
    "extra_log_bayes_factor_challenger_vs_anchor",
    "override_applied",
    "selected_pathology",
    "correct",
]]
report_text = f"""# Close-Confounder Discriminator Report

Generated by `notebooks/33_close_confounder_discriminator.ipynb` on {datetime.now().isoformat(timespec="seconds")}.

## Inputs

- Notebook 30 candidate pool: `{NOTEBOOK30_ROOT}`
- Notebook 31 neural resolver reference: `{NOTEBOOK31_ROOT}`
- Notebook 32 resolver ablation artifact: `{NOTEBOOK32_ROOT}`
- Bayesian likelihood table: `{required_files["bayes_likelihoods"]}`
- Graph edge table: `{required_files["graph_edges"]}`
- New artifact root: `{ARTIFACT_ROOT}`
- Offline only: no API calls. Extra evidence is revealed only for roots selected by the discriminator policy.

## Selected Policy

`close_confounder_discriminator_v1` uses the fixed Notebook 32 `gradient_boosting_name_family` resolver as its base final head. It flags close top-2 candidate pairs when the pair has sufficient missing discriminator utility, reveals up to `{EXTRA_ROOT_BUDGET}` roots ranked by train-derived Jensen-Shannon separation times root mutual information, and overrides the anchor only when the extra-root log Bayes factor for the challenger is at least `{PAIR_LOG_BAYES_FACTOR_OVERRIDE_MIN}`.

## Key Results

- Notebook 31 compact neural reference: `46/49`.
- Notebook 32 GBM diagnostic reference: `47/49`.
- Notebook 33 selected close-confounder discriminator: `{int(case_results["correct"].sum())}/49`.
- Wins vs Notebook 32 GBM: `{int(paired_vs_gbm["win_vs_notebook32_gbm"].sum())}`.
- Regressions vs Notebook 32 GBM: `{int(paired_vs_gbm["regression_vs_notebook32_gbm"].sum())}`.
- Flagged cases: `{int(case_results["flagged_for_discriminator"].sum())}/49`.
- Overrides applied: `{int(case_results["override_applied"].sum())}/49`.
- Additional evidence requests: `{int(case_results["extra_roots_requested"].sum())}` total, `{float(case_results["extra_roots_requested"].mean()):.3f}` per case.

## Policy Summary

{top_rows[["policy_name", "num_correct", "accuracy", "flagged_cases", "overrides", "mean_selected_requests", "misses"]].to_markdown(index=False)}

## Flagged Cases

{flagged_table.to_markdown(index=False)}

## Interpretation

The targeted discriminator fixes the Bronchitis-vs-URTI miss without destabilizing the Pericarditis correction found by the Notebook 32 GBM row. The remaining miss is Acute rhinosinusitis versus Chronic rhinosinusitis; the selected pain-location/intensity discriminator roots do not produce a Bayes-factor case for acute disease, matching the earlier observation that this case remains difficult even with stronger full-evidence diagnostic rows.

This is the first saved candidate-pool follow-up to reach `48/49` without using 49-case labels for resolver training or full unobserved evidence features. It should still be framed as an offline candidate requiring independent confirmation, because the fixed GBM base row came from the Notebook 32 ablation sweep.
"""
REPORT_PATH.write_text(report_text, encoding="utf-8")

print("Selected policy correct:", int(case_results["correct"].sum()), "/", len(case_results))
print("Wins vs Notebook 32 GBM:", int(paired_vs_gbm["win_vs_notebook32_gbm"].sum()))
print("Regressions vs Notebook 32 GBM:", int(paired_vs_gbm["regression_vs_notebook32_gbm"].sum()))
print("Mean selected requests:", float(selected_summary["mean_selected_requests"]))
print("Artifact root:", ARTIFACT_ROOT)
