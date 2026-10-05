"""Scene-level stress-set transfer, conditional references, and reuse regret.

Measures scene identity, captured target loss, and same-budget efficiency.
References account for acquisition-log composition and clean-detection
opportunity. Boundary ties are averaged exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

import numpy as np


MODEL_ORDER = ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
OUTCOME_ORDER = ("NDS_like_loss", "lost_clean_tp_fraction", "lost_clean_tp_count")
DEFAULT_FRACTIONS = (0.10, 0.20, 0.30)
DEFAULT_SEED = 20260724
DEFAULT_PERMUTATIONS = 10_000
TIE_REL_TOLERANCE = 1e-9
TIE_ABS_TOLERANCE = 1e-12
DECISION_TOLERANCE = 1e-12


def _is_boundary_tie(value: float, boundary: float) -> bool:
    """Use one tie predicate for exact expectations and Monte Carlo masks."""
    return math.isclose(
        float(value),
        float(boundary),
        rel_tol=TIE_REL_TOLERANCE,
        abs_tol=TIE_ABS_TOLERANCE,
    )


def _percentile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot take a percentile of an empty collection")
    return float(np.quantile(array, probability))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including exact-tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def stratified_spearman(
    source_scores: dict[str, float],
    target_scores: dict[str, float],
    scene_to_log: dict[str, str],
) -> float | None:
    """Correlation of within-log centered ranks.

    Centering ranks inside each acquisition log removes between-log location
    effects before measuring continuous scene-order agreement.
    """
    source_residuals: list[float] = []
    target_residuals: list[float] = []
    by_log: dict[str, list[str]] = defaultdict(list)
    for scene in source_scores:
        by_log[scene_to_log[scene]].append(scene)
    for scenes in by_log.values():
        scenes = sorted(scenes)
        source = _average_ranks(
            np.asarray([source_scores[scene] for scene in scenes], dtype=float)
        )
        target = _average_ranks(
            np.asarray([target_scores[scene] for scene in scenes], dtype=float)
        )
        source_residuals.extend((source - source.mean()).tolist())
        target_residuals.extend((target - target.mean()).tolist())
    source_array = np.asarray(source_residuals, dtype=float)
    target_array = np.asarray(target_residuals, dtype=float)
    denominator = float(
        np.sqrt(np.sum(source_array**2) * np.sum(target_array**2))
    )
    if denominator <= 1e-12:
        return None
    return float(np.sum(source_array * target_array) / denominator)


def deterministic_top_set(scores: dict[str, float], k: int) -> set[str]:
    """Select exactly k highest scores with a stable lexical tie break."""
    if not 0 < k <= len(scores):
        raise ValueError("k must lie in [1, number of scenes]")
    return set(
        sorted(scores, key=lambda scene: (-float(scores[scene]), scene))[:k]
    )


def boundary_tie_size(scores: dict[str, float], selected: set[str]) -> int:
    boundary = min(float(scores[scene]) for scene in selected)
    return sum(
        _is_boundary_tie(value, boundary)
        for value in scores.values()
    )


def top_k_inclusion_probabilities(
    scores: dict[str, float], k: int
) -> dict[str, float]:
    """Exact inclusion probabilities under uniform random boundary tie breaks."""
    selected = deterministic_top_set(scores, k)
    boundary = min(float(scores[scene]) for scene in selected)
    tied = {
        scene
        for scene, value in scores.items()
        if _is_boundary_tie(value, boundary)
    }
    above = {
        scene
        for scene, value in scores.items()
        if float(value) > boundary and scene not in tied
    }
    remaining = k - len(above)
    if not 0 <= remaining <= len(tied):
        raise RuntimeError("Boundary partition cannot produce an exact top-k set")
    probability = remaining / len(tied)
    probabilities = {
        scene: 1.0 if scene in above else probability if scene in tied else 0.0
        for scene in scores
    }
    if not math.isclose(
        sum(probabilities.values()), float(k), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Top-k inclusion probabilities do not sum to k")
    return probabilities


def tie_averaged_transfer(
    source_scores: dict[str, float],
    target_scores: dict[str, float],
    scene_to_log: dict[str, str],
    k: int,
    target_opportunity: dict[str, float] | None = None,
) -> dict:
    """Exact expectation over independent uniform source/target tie breaks."""
    source_probability = top_k_inclusion_probabilities(source_scores, k)
    target_probability = top_k_inclusion_probabilities(target_scores, k)
    expected_intersection = sum(
        source_probability[scene] * target_probability[scene]
        for scene in source_scores
    )
    by_log: dict[str, list[str]] = defaultdict(list)
    for scene, log in scene_to_log.items():
        by_log[log].append(scene)
    null_intersection = 0.0
    for scenes in by_log.values():
        expected_source_count = sum(source_probability[scene] for scene in scenes)
        expected_target_count = sum(target_probability[scene] for scene in scenes)
        null_intersection += (
            expected_source_count * expected_target_count / len(scenes)
        )
    positive_target = {
        scene: max(float(value), 0.0) for scene, value in target_scores.items()
    }
    observed_mass = sum(
        source_probability[scene] * positive_target[scene]
        for scene in source_scores
    )
    total_positive_mass = sum(positive_target.values())
    oracle_mass = sum(
        target_probability[scene] * positive_target[scene]
        for scene in target_scores
    )
    expected_random_mass = 0.0
    for scenes in by_log.values():
        expected_source_count = sum(source_probability[scene] for scene in scenes)
        expected_random_mass += expected_source_count / len(scenes) * sum(
            positive_target[scene] for scene in scenes
        )
    coverage = observed_mass / oracle_mass if oracle_mass > 1e-12 else None
    absolute_coverage = (
        observed_mass / total_positive_mass
        if total_positive_mass > 1e-12
        else None
    )
    null_coverage = (
        expected_random_mass / oracle_mass if oracle_mass > 1e-12 else None
    )
    opportunity_mass = None
    opportunity_coverage = None
    opportunity_adjusted_coverage = None
    opportunity_lift = None
    if target_opportunity is not None:
        expected_opportunity_mass = 0.0
        for scenes in by_log.values():
            total_opportunity = sum(target_opportunity[scene] for scene in scenes)
            if total_opportunity <= 1e-12:
                continue
            log_rate = (
                sum(positive_target[scene] for scene in scenes)
                / total_opportunity
            )
            selected_opportunity = sum(
                source_probability[scene] * target_opportunity[scene]
                for scene in scenes
            )
            expected_opportunity_mass += log_rate * selected_opportunity
        opportunity_mass = expected_opportunity_mass
        if oracle_mass > 1e-12:
            opportunity_coverage = opportunity_mass / oracle_mass
            opportunity_adjusted_coverage = adjusted_coverage(
                coverage, opportunity_coverage
            )
        if opportunity_mass > 1e-12:
            opportunity_lift = observed_mass / opportunity_mass
    return {
        "tie_random_expected_observed_target_risk_mass": observed_mass,
        "tie_random_expected_total_target_positive_risk_mass": (
            total_positive_mass
        ),
        "tie_random_expected_target_oracle_risk_mass": oracle_mass,
        "tie_random_expected_absolute_target_risk_coverage": absolute_coverage,
        "tie_random_expected_raw_overlap": expected_intersection / k,
        "tie_random_expected_log_stratified_chance_overlap": (
            null_intersection / k
        ),
        "tie_random_expected_log_stratified_adjusted_overlap": adjusted_overlap(
            expected_intersection, null_intersection, k
        ),
        "tie_random_expected_RiskCoverage": coverage,
        "tie_random_expected_log_stratified_random_RiskCoverage": null_coverage,
        "tie_random_expected_log_stratified_adjusted_RiskCoverage": (
            adjusted_coverage(coverage, null_coverage)
        ),
        "tie_random_expected_log_opportunity_expected_target_risk_mass": (
            opportunity_mass
        ),
        "tie_random_expected_log_opportunity_expected_RiskCoverage": (
            opportunity_coverage
        ),
        "tie_random_expected_log_opportunity_adjusted_RiskCoverage": (
            opportunity_adjusted_coverage
        ),
        "tie_random_expected_log_opportunity_risk_lift": opportunity_lift,
    }


def log_stratified_expected_intersection(
    source_set: set[str],
    target_set: set[str],
    scene_to_log: dict[str, str],
) -> float:
    """Expected intersection under independent uniform sets within each log."""
    scenes_by_log: dict[str, set[str]] = defaultdict(set)
    for scene, log in scene_to_log.items():
        scenes_by_log[log].add(scene)
    expected = 0.0
    for scenes in scenes_by_log.values():
        n_log = len(scenes)
        source_count = len(source_set & scenes)
        target_count = len(target_set & scenes)
        expected += source_count * target_count / n_log
    return expected


def adjusted_overlap(
    observed_intersection: int,
    expected_intersection: float,
    maximum_intersection: int,
) -> float | None:
    """Chance-adjusted overlap, 0 at the null and 1 at perfect agreement."""
    denominator = maximum_intersection - expected_intersection
    if denominator <= 1e-12:
        return None
    return (observed_intersection - expected_intersection) / denominator


def log_stratified_expected_risk_mass(
    source_set: set[str],
    target_positive_risk: dict[str, float],
    scene_to_log: dict[str, str],
) -> float:
    """Expected target risk captured by a same-size, same-log-composition set."""
    scenes_by_log: dict[str, list[str]] = defaultdict(list)
    for scene, log in scene_to_log.items():
        scenes_by_log[log].append(scene)
    expected = 0.0
    for scenes in scenes_by_log.values():
        source_count = sum(scene in source_set for scene in scenes)
        if not source_count:
            continue
        expected += source_count / len(scenes) * sum(
            target_positive_risk[scene] for scene in scenes
        )
    return expected


def log_opportunity_expected_risk_mass(
    source_set: set[str],
    target_positive_risk: dict[str, float],
    target_opportunity: dict[str, float],
    scene_to_log: dict[str, str],
) -> float:
    """Expected captured risk under a constant per-opportunity log rate."""
    scenes_by_log: dict[str, list[str]] = defaultdict(list)
    for scene, log in scene_to_log.items():
        scenes_by_log[log].append(scene)
    expected = 0.0
    for scenes in scenes_by_log.values():
        total_opportunity = sum(target_opportunity[scene] for scene in scenes)
        if total_opportunity <= 1e-12:
            continue
        log_rate = (
            sum(target_positive_risk[scene] for scene in scenes)
            / total_opportunity
        )
        selected_opportunity = sum(
            target_opportunity[scene]
            for scene in scenes
            if scene in source_set
        )
        expected += log_rate * selected_opportunity
    return expected


def adjusted_coverage(
    observed_coverage: float | None, null_coverage: float | None
) -> float | None:
    """Normalize excess coverage so 0 is random and 1 is oracle coverage."""
    if observed_coverage is None or null_coverage is None:
        return None
    denominator = 1.0 - null_coverage
    if denominator <= 1e-12:
        return None
    return (observed_coverage - null_coverage) / denominator


def transfer_cell(
    source_scores: dict[str, float],
    target_scores: dict[str, float],
    scene_to_log: dict[str, str],
    fraction: float,
    target_opportunity: dict[str, float] | None = None,
) -> dict:
    """Compute raw and corrected transfer estimands for one directed pair."""
    if source_scores.keys() != target_scores.keys():
        raise ValueError("Source and target must use exactly the same scenes")
    n_scenes = len(source_scores)
    k = max(1, math.ceil(fraction * n_scenes))
    source_set = deterministic_top_set(source_scores, k)
    target_set = deterministic_top_set(target_scores, k)
    intersection = len(source_set & target_set)
    global_expected_intersection = k * k / n_scenes
    stratified_expected_intersection = log_stratified_expected_intersection(
        source_set, target_set, scene_to_log
    )
    positive_target = {
        scene: max(float(value), 0.0) for scene, value in target_scores.items()
    }
    observed_mass = sum(positive_target[scene] for scene in source_set)
    total_positive_mass = sum(positive_target.values())
    oracle_mass = sum(positive_target[scene] for scene in target_set)
    coverage = observed_mass / oracle_mass if oracle_mass > 1e-12 else None
    absolute_coverage = (
        observed_mass / total_positive_mass
        if total_positive_mass > 1e-12
        else None
    )
    expected_mass = log_stratified_expected_risk_mass(
        source_set, positive_target, scene_to_log
    )
    null_coverage = expected_mass / oracle_mass if oracle_mass > 1e-12 else None
    opportunity_mass = None
    opportunity_coverage = None
    opportunity_adjusted_coverage = None
    opportunity_lift = None
    if target_opportunity is not None:
        if target_opportunity.keys() != target_scores.keys():
            raise ValueError("Target opportunity must use the same scenes")
        opportunity_mass = log_opportunity_expected_risk_mass(
            source_set,
            positive_target,
            target_opportunity,
            scene_to_log,
        )
        if oracle_mass > 1e-12:
            opportunity_coverage = opportunity_mass / oracle_mass
            opportunity_adjusted_coverage = adjusted_coverage(
                coverage, opportunity_coverage
            )
        if opportunity_mass > 1e-12:
            opportunity_lift = observed_mass / opportunity_mass
    tie_averaged = tie_averaged_transfer(
        source_scores,
        target_scores,
        scene_to_log,
        k,
        target_opportunity=target_opportunity,
    )
    return {
        "n_scenes": n_scenes,
        "top_k": k,
        "source_top_scenes": sorted(source_set),
        "target_top_scenes": sorted(target_set),
        "source_boundary_tie_size": boundary_tie_size(source_scores, source_set),
        "target_boundary_tie_size": boundary_tie_size(target_scores, target_set),
        "observed_intersection": intersection,
        "lexical_tie_break_raw_overlap": intersection / k,
        "global_chance_overlap": global_expected_intersection / k,
        "lexical_tie_break_global_adjusted_overlap": adjusted_overlap(
            intersection, global_expected_intersection, k
        ),
        "lexical_tie_break_log_stratified_chance_overlap": (
            stratified_expected_intersection / k
        ),
        "lexical_tie_break_log_stratified_adjusted_overlap": adjusted_overlap(
            intersection, stratified_expected_intersection, k
        ),
        "lexical_tie_break_observed_target_risk_mass": observed_mass,
        "lexical_tie_break_total_target_positive_risk_mass": (
            total_positive_mass
        ),
        "lexical_tie_break_target_oracle_risk_mass": oracle_mass,
        "lexical_tie_break_absolute_target_risk_coverage": absolute_coverage,
        "lexical_tie_break_raw_RiskCoverage": coverage,
        "lexical_tie_break_log_stratified_random_RiskCoverage": null_coverage,
        "lexical_tie_break_log_stratified_adjusted_RiskCoverage": adjusted_coverage(
            coverage, null_coverage
        ),
        "lexical_tie_break_log_opportunity_expected_target_risk_mass": (
            opportunity_mass
        ),
        "lexical_tie_break_log_opportunity_expected_RiskCoverage": (
            opportunity_coverage
        ),
        "lexical_tie_break_log_opportunity_adjusted_RiskCoverage": (
            opportunity_adjusted_coverage
        ),
        "lexical_tie_break_log_opportunity_risk_lift": opportunity_lift,
        # Exact expectations over uniform boundary-tie breaks are the primary
        # estimands. Lexical selections above are retained only as sensitivity.
        "raw_overlap": tie_averaged["tie_random_expected_raw_overlap"],
        "global_adjusted_overlap": adjusted_overlap(
            tie_averaged["tie_random_expected_raw_overlap"] * k,
            global_expected_intersection,
            k,
        ),
        "log_stratified_chance_overlap": tie_averaged[
            "tie_random_expected_log_stratified_chance_overlap"
        ],
        "log_stratified_adjusted_overlap": tie_averaged[
            "tie_random_expected_log_stratified_adjusted_overlap"
        ],
        "observed_target_risk_mass": tie_averaged[
            "tie_random_expected_observed_target_risk_mass"
        ],
        "total_target_positive_risk_mass": tie_averaged[
            "tie_random_expected_total_target_positive_risk_mass"
        ],
        "target_oracle_risk_mass": tie_averaged[
            "tie_random_expected_target_oracle_risk_mass"
        ],
        "absolute_target_risk_coverage": tie_averaged[
            "tie_random_expected_absolute_target_risk_coverage"
        ],
        "raw_RiskCoverage": tie_averaged[
            "tie_random_expected_RiskCoverage"
        ],
        "log_stratified_random_RiskCoverage": tie_averaged[
            "tie_random_expected_log_stratified_random_RiskCoverage"
        ],
        "log_stratified_adjusted_RiskCoverage": tie_averaged[
            "tie_random_expected_log_stratified_adjusted_RiskCoverage"
        ],
        "log_opportunity_expected_target_risk_mass": tie_averaged[
            "tie_random_expected_log_opportunity_expected_target_risk_mass"
        ],
        "log_opportunity_expected_RiskCoverage": tie_averaged[
            "tie_random_expected_log_opportunity_expected_RiskCoverage"
        ],
        "log_opportunity_adjusted_RiskCoverage": tie_averaged[
            "tie_random_expected_log_opportunity_adjusted_RiskCoverage"
        ],
        "log_opportunity_risk_lift": tie_averaged[
            "tie_random_expected_log_opportunity_risk_lift"
        ],
        "within_log_rank_correlation": stratified_spearman(
            source_scores, target_scores, scene_to_log
        ),
        **tie_averaged,
    }


def _load_outcome_tables(primary_path: Path, mechanism_path: Path) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, float]]],
    dict[tuple[str, str], dict[str, float]],
    dict[str, str],
    dict,
]:
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    if primary.get("partition") not in {"study", "confirmation"}:
        raise ValueError("Transfer analysis requires a development or follow-up partition")
    scene_to_log = {
        str(scene): str(log) for scene, log in primary["scene_log_tokens"].items()
    }
    tables: dict[str, dict[tuple[str, str], dict[str, float]]] = {
        outcome: {} for outcome in OUTCOME_ORDER
    }
    opportunity_table: dict[tuple[str, str], dict[str, float]] = {}
    for row in primary["scene_rows"]:
        key = (str(row["model"]), str(row["candidate"]))
        tables["NDS_like_loss"][key] = {
            str(scene): float(value)
            for scene, value in row["NDS_losses_by_scene"].items()
        }
    for row in mechanism["deep_mechanism"]["tail_concentration"]:
        key = (str(row["model"]), str(row["condition"]))
        tables["lost_clean_tp_fraction"].setdefault(key, {})[
            str(row["scene"])
        ] = float(row["L_status"])
        tables["lost_clean_tp_count"].setdefault(key, {})[
            str(row["scene"])
        ] = float(row["lost_clean_tp_count"])
        opportunity_table.setdefault(key, {})[str(row["scene"])] = float(
            row["clean_tp_count"]
        )
    expected_keys = {
        (model, condition)
        for model in MODEL_ORDER
        for condition in sorted(
            {
                condition
                for _, condition in tables["lost_clean_tp_fraction"].keys()
            }
        )
    }
    for outcome, table in tables.items():
        if set(table) != expected_keys:
            raise ValueError(f"{outcome} does not contain the full model-condition grid")
        for scores in table.values():
            if set(scores) != set(scene_to_log):
                raise ValueError(f"{outcome} does not contain the full scene universe")
    if set(opportunity_table) != expected_keys:
        raise ValueError("Opportunity table does not contain the full grid")
    for opportunities in opportunity_table.values():
        if set(opportunities) != set(scene_to_log):
            raise ValueError("Opportunity table does not contain all scenes")
        if any(value < 0 for value in opportunities.values()):
            raise ValueError("Opportunity counts must be nonnegative")
    provenance = {
        "primary": str(primary_path),
        "mechanism": str(mechanism_path),
        "primary_artifact_type": primary.get("artifact_type"),
        "mechanism_artifact_type": mechanism.get("artifact_type"),
        "partition": primary.get("partition"),
    }
    return tables, opportunity_table, scene_to_log, provenance


def compute_cells(
    table: dict[tuple[str, str], dict[str, float]],
    scene_to_log: dict[str, str],
    fraction: float,
    opportunity_table: dict[tuple[str, str], dict[str, float]] | None = None,
) -> list[dict]:
    conditions = sorted({condition for _, condition in table})
    rows: list[dict] = []
    for condition in conditions:
        for source in MODEL_ORDER:
            for target in MODEL_ORDER:
                if source == target:
                    continue
                cell = transfer_cell(
                    table[(source, condition)],
                    table[(target, condition)],
                    scene_to_log,
                    fraction,
                    (
                        opportunity_table[(target, condition)]
                        if opportunity_table is not None
                        else None
                    ),
                )
                rows.append(
                    {
                        "condition": condition,
                        "source": source,
                        "target": target,
                        **cell,
                    }
                )
    return rows


def _summary(rows: list[dict]) -> dict:
    metric_names = (
        "raw_overlap",
        "global_chance_overlap",
        "global_adjusted_overlap",
        "log_stratified_chance_overlap",
        "log_stratified_adjusted_overlap",
        "raw_RiskCoverage",
        "absolute_target_risk_coverage",
        "log_stratified_random_RiskCoverage",
        "log_stratified_adjusted_RiskCoverage",
        "within_log_rank_correlation",
        "tie_random_expected_raw_overlap",
        "tie_random_expected_log_stratified_chance_overlap",
        "tie_random_expected_log_stratified_adjusted_overlap",
        "tie_random_expected_RiskCoverage",
        "tie_random_expected_absolute_target_risk_coverage",
        "tie_random_expected_log_stratified_random_RiskCoverage",
        "tie_random_expected_log_stratified_adjusted_RiskCoverage",
        "log_opportunity_expected_RiskCoverage",
        "log_opportunity_adjusted_RiskCoverage",
        "log_opportunity_risk_lift",
        "lexical_tie_break_raw_overlap",
        "lexical_tie_break_log_stratified_adjusted_overlap",
        "lexical_tie_break_raw_RiskCoverage",
        "lexical_tie_break_absolute_target_risk_coverage",
        "lexical_tie_break_log_stratified_adjusted_RiskCoverage",
        "lexical_tie_break_log_opportunity_adjusted_RiskCoverage",
    )
    result: dict[str, dict] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        if not values:
            result[metric] = {
                "estimable_cells": 0,
                "min": None,
                "median": None,
                "mean": None,
                "max": None,
            }
            continue
        result[metric] = {
            "estimable_cells": len(values),
            "min": min(values),
            "median": median(values),
            "mean": fmean(values),
            "max": max(values),
        }
    result["cells"] = len(rows)
    result["source_boundary_tied_cells"] = sum(
        row["source_boundary_tie_size"] > 1 for row in rows
    )
    result["target_boundary_tied_cells"] = sum(
        row["target_boundary_tie_size"] > 1 for row in rows
    )
    return result


def _metric_maximizers(rows: list[dict], metric: str) -> list[dict]:
    values = [row[metric] for row in rows]
    if any(value is None for value in values):
        return []
    maximum = max(float(value) for value in values)
    return [
        row
        for row in rows
        if math.isclose(
            float(row[metric]),
            maximum,
            rel_tol=0.0,
            abs_tol=DECISION_TOLERANCE,
        )
    ]


def decision_consequence(rows: list[dict]) -> dict:
    """Decompose identity-guided reuse regret against the target oracle.

    Each target-condition decision has exactly two candidate source detectors.
    Boundary-tie-averaged transfer metrics are used throughout. If a decision
    metric ties, utility is averaged uniformly over the maximizing sources.
    Total regret separates exactly into candidate-availability and
    identity-selection components.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target"]), str(row["condition"]))].append(row)

    expected_groups = len({str(row["target"]) for row in rows}) * len(
        {str(row["condition"]) for row in rows}
    )
    if len(grouped) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} target-condition decisions, "
            f"found {len(grouped)}"
        )

    relation_errors: list[float] = []
    for row in rows:
        efficiency = row["raw_RiskCoverage"]
        coverage = row["absolute_target_risk_coverage"]
        oracle = row["target_oracle_risk_mass"]
        total = row["total_target_positive_risk_mass"]
        if None in (efficiency, coverage, oracle, total) or float(total) <= 0:
            continue
        implied = float(efficiency) * float(oracle) / float(total)
        relation_errors.append(abs(float(coverage) - implied))

    rule_results: dict[str, dict] = {}
    for rule in ("raw_overlap", "log_stratified_adjusted_overlap"):
        units: list[dict] = []
        for (target, condition), candidates in sorted(grouped.items()):
            if len(candidates) != len(MODEL_ORDER) - 1:
                raise RuntimeError(
                    f"{target}/{condition}: expected two candidate sources"
                )
            sources = {str(row["source"]) for row in candidates}
            expected_sources = set(MODEL_ORDER) - {target}
            if sources != expected_sources:
                raise RuntimeError(
                    f"{target}/{condition}: candidate source identities differ"
                )
            top_k = {int(row["top_k"]) for row in candidates}
            if len(top_k) != 1:
                raise RuntimeError(f"{target}/{condition}: inconsistent top-k")
            for field in (
                "target_oracle_risk_mass",
                "total_target_positive_risk_mass",
            ):
                values = [row[field] for row in candidates]
                if any(value is None for value in values):
                    raise RuntimeError(
                        f"{target}/{condition}: missing target mass for {field}"
                    )
                if not math.isclose(
                    float(values[0]),
                    float(values[1]),
                    rel_tol=0.0,
                    abs_tol=DECISION_TOLERANCE,
                ):
                    raise RuntimeError(
                        f"{target}/{condition}: inconsistent {field}"
                    )

            selected = _metric_maximizers(candidates, rule)
            best = _metric_maximizers(candidates, "raw_RiskCoverage")
            if not selected or not best:
                continue
            selected_utility = fmean(
                float(row["raw_RiskCoverage"]) for row in selected
            )
            best_utility = max(
                float(row["raw_RiskCoverage"]) for row in candidates
            )
            random_utility = fmean(
                float(row["raw_RiskCoverage"]) for row in candidates
            )
            total_reuse_regret = max(0.0, 1.0 - selected_utility)
            availability_regret = max(0.0, 1.0 - best_utility)
            selection_regret = max(0.0, best_utility - selected_utility)
            random_selection_regret = max(0.0, best_utility - random_utility)
            relative_selection_regret = (
                selection_regret / best_utility
                if best_utility > DECISION_TOLERANCE
                else None
            )
            decomposition_error = abs(
                total_reuse_regret
                - availability_regret
                - selection_regret
            )
            selected_sources = sorted(str(row["source"]) for row in selected)
            best_sources = sorted(str(row["source"]) for row in best)
            units.append(
                {
                    "target": target,
                    "condition": condition,
                    "top_k": top_k.pop(),
                    "candidate_sources": sorted(sources),
                    "selection_scores": {
                        str(row["source"]): float(row[rule])
                        for row in candidates
                    },
                    "candidate_efficiencies": {
                        str(row["source"]): float(row["raw_RiskCoverage"])
                        for row in candidates
                    },
                    "selected_sources": selected_sources,
                    "best_utility_sources": best_sources,
                    "selection_tie": len(selected_sources) > 1,
                    "utility_tie": len(best_sources) > 1,
                    "strict_reversal": (
                        len(selected_sources) == 1
                        and len(best_sources) == 1
                        and selected_sources != best_sources
                    ),
                    "suboptimal_choice": (
                        selection_regret > DECISION_TOLERANCE
                    ),
                    "selected_efficiency": selected_utility,
                    "best_candidate_efficiency": best_utility,
                    "random_candidate_efficiency": random_utility,
                    "total_reuse_regret": total_reuse_regret,
                    "availability_regret": availability_regret,
                    "selection_regret": selection_regret,
                    "relative_selection_regret": relative_selection_regret,
                    "random_selection_regret": random_selection_regret,
                    "selection_advantage_over_random": (
                        random_selection_regret - selection_regret
                    ),
                    "regret_decomposition_error": decomposition_error,
                }
            )

        total_regrets = [float(unit["total_reuse_regret"]) for unit in units]
        availability_regrets = [
            float(unit["availability_regret"]) for unit in units
        ]
        selection_regrets = [
            float(unit["selection_regret"]) for unit in units
        ]
        positive_selection = [
            value
            for value in selection_regrets
            if value > DECISION_TOLERANCE
        ]
        relative_selection = [
            float(unit["relative_selection_regret"])
            for unit in units
            if unit["relative_selection_regret"] is not None
        ]
        random_selection_regrets = [
            float(unit["random_selection_regret"]) for unit in units
        ]
        advantages = [
            float(unit["selection_advantage_over_random"]) for unit in units
        ]
        decomposition_errors = [
            float(unit["regret_decomposition_error"]) for unit in units
        ]
        count = len(units)
        mean_total = fmean(total_regrets) if total_regrets else None
        mean_availability = (
            fmean(availability_regrets) if availability_regrets else None
        )
        mean_selection = (
            fmean(selection_regrets) if selection_regrets else None
        )
        rule_results[rule] = {
            "selection_metric": rule,
            "utility_metric": "raw_RiskCoverage",
            "units": units,
            "summary": {
                "decision_units": expected_groups,
                "estimable_units": count,
                "strict_reversals": sum(
                    bool(unit["strict_reversal"]) for unit in units
                ),
                "strict_reversal_fraction": (
                    sum(bool(unit["strict_reversal"]) for unit in units) / count
                    if count
                    else None
                ),
                "selection_ties": sum(
                    bool(unit["selection_tie"]) for unit in units
                ),
                "utility_ties": sum(
                    bool(unit["utility_tie"]) for unit in units
                ),
                "positive_selection_regret_units": len(positive_selection),
                "positive_selection_regret_fraction": (
                    len(positive_selection) / count if count else None
                ),
                "total_reuse_regret_at_least_0_02_units": sum(
                    value >= 0.02 - DECISION_TOLERANCE
                    for value in total_regrets
                ),
                "mean_total_reuse_regret": mean_total,
                "median_total_reuse_regret": (
                    median(total_regrets) if total_regrets else None
                ),
                "q25_total_reuse_regret": (
                    _percentile(total_regrets, 0.25)
                    if total_regrets
                    else None
                ),
                "q75_total_reuse_regret": (
                    _percentile(total_regrets, 0.75)
                    if total_regrets
                    else None
                ),
                "max_total_reuse_regret": (
                    max(total_regrets) if total_regrets else None
                ),
                "mean_availability_regret": mean_availability,
                "median_availability_regret": (
                    median(availability_regrets)
                    if availability_regrets
                    else None
                ),
                "mean_selection_regret": mean_selection,
                "median_selection_regret": (
                    median(selection_regrets) if selection_regrets else None
                ),
                "median_positive_selection_regret": (
                    median(positive_selection) if positive_selection else 0.0
                ),
                "mean_relative_selection_regret": (
                    fmean(relative_selection)
                    if relative_selection
                    else None
                ),
                "selection_share_of_mean_total_regret": (
                    mean_selection / mean_total
                    if mean_total is not None
                    and mean_total > DECISION_TOLERANCE
                    and mean_selection is not None
                    else None
                ),
                "mean_random_selection_regret": (
                    fmean(random_selection_regrets)
                    if random_selection_regrets
                    else None
                ),
                "mean_selection_advantage_over_random": (
                    fmean(advantages) if advantages else None
                ),
                "max_regret_decomposition_error": (
                    max(decomposition_errors)
                    if decomposition_errors
                    else None
                ),
            },
        }

    return {
        "scope": (
            "retrospective decomposition of target-oracle reuse regret among "
            "the two non-target detector stress sets; not a deployment policy"
        ),
        "regret_decomposition": {
            "formula": (
                "1 - E_selected = (1 - E_best_available) "
                "+ (E_best_available - E_selected)"
            ),
            "total_component": "target-oracle reuse regret",
            "availability_component": (
                "regret remaining even for the best available source set"
            ),
            "selection_component": (
                "additional regret from identity-guided source choice"
            ),
        },
        "coverage_efficiency_relation": {
            "formula": "C_all = E_k * target_oracle_mass / total_positive_mass",
            "checked_cells": len(relation_errors),
            "max_absolute_error": max(relation_errors, default=None),
        },
        "rules": rule_results,
    }


def _random_source_masks_from_observed_masks(
    observed_masks: np.ndarray,
    scenes: list[str],
    scene_to_log: dict[str, str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize each sampled source set within logs at its sampled log counts."""
    masks = np.zeros_like(observed_masks)
    scene_position = {scene: index for index, scene in enumerate(scenes)}
    by_log: dict[str, list[str]] = defaultdict(list)
    for scene in scenes:
        by_log[scene_to_log[scene]].append(scene)
    for log_scenes in by_log.values():
        positions = np.asarray(
            [scene_position[scene] for scene in sorted(log_scenes)]
        )
        counts = observed_masks[:, positions].sum(axis=1)
        order = np.argsort(
            rng.random((len(observed_masks), len(positions))), axis=1
        )
        for count in np.unique(counts):
            if count == 0:
                continue
            rows = np.flatnonzero(counts == count)
            chosen = positions[order[rows, : int(count)]]
            masks[rows[:, None], chosen] = True
    return masks


def conditional_randomization_test(
    table: dict[tuple[str, str], dict[str, float]],
    scene_to_log: dict[str, str],
    fraction: float,
    replicates: int,
    seed: int,
) -> dict:
    """Outcome-level conditional randomization test for median transfer.

    Each source detector-condition set is randomized within acquisition logs
    while retaining its observed per-log set sizes. The same randomized source
    set is reused for both directed target comparisons.
    """
    rows = compute_cells(table, scene_to_log, fraction)
    scenes = sorted(scene_to_log)
    rng = np.random.default_rng(seed)
    observed_masks: dict[tuple[str, str], np.ndarray] = {}
    randomized_masks: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        key = (row["condition"], row["source"])
        if key not in observed_masks:
            sampled = _sample_observed_top_masks(
                table[(row["source"], row["condition"])],
                scenes,
                int(row["top_k"]),
                replicates,
                rng,
            )
            observed_masks[key] = sampled
            randomized_masks[key] = _random_source_masks_from_observed_masks(
                sampled, scenes, scene_to_log, rng
            )
    target_masks: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        key = (row["condition"], row["target"])
        if key not in target_masks:
            target_masks[key] = _sample_observed_top_masks(
                table[(row["target"], row["condition"])],
                scenes,
                int(row["top_k"]),
                replicates,
                rng,
            )
    overlap_null = np.empty((replicates, len(rows)), dtype=float)
    coverage_null = np.full((replicates, len(rows)), np.nan, dtype=float)
    for column, row in enumerate(rows):
        masks = randomized_masks[(row["condition"], row["source"])]
        target_indicator = target_masks[(row["condition"], row["target"])]
        overlap_null[:, column] = (
            np.sum(masks & target_indicator, axis=1) / float(row["top_k"])
        )
        target_scores = table[(row["target"], row["condition"])]
        target_risk = np.asarray(
            [max(float(target_scores[scene]), 0.0) for scene in scenes]
        )
        oracle = target_indicator @ target_risk
        captured = masks @ target_risk
        coverage_null[:, column] = np.divide(
            captured,
            oracle,
            out=np.full(replicates, np.nan, dtype=float),
            where=oracle > 1e-12,
        )
    observed_overlap = median(float(row["raw_overlap"]) for row in rows)
    observed_coverage_values = [
        float(row["raw_RiskCoverage"])
        for row in rows
        if row["raw_RiskCoverage"] is not None
    ]
    observed_coverage = median(observed_coverage_values)
    null_overlap = np.median(overlap_null, axis=1)
    null_coverage = np.nanmedian(coverage_null, axis=1)

    def test_block(observed: float, null: np.ndarray) -> dict:
        return {
            "observed_median": observed,
            "null_median": float(np.median(null)),
            "null_ci95": [
                float(np.quantile(null, 0.025)),
                float(np.quantile(null, 0.975)),
            ],
            "excess_over_null_median": observed - float(np.median(null)),
            "one_sided_p_greater": float(
                (1 + np.count_nonzero(null >= observed)) / (replicates + 1)
            ),
        }

    return {
        "replicates": replicates,
        "seed": seed,
        "null": (
            "source top-k scene identities are exchangeable within acquisition "
            "logs, conditional on observed source per-log selected counts"
        ),
        "overlap": test_block(observed_overlap, null_overlap),
        "RiskCoverage": test_block(observed_coverage, null_coverage),
    }


def _sample_observed_top_masks(
    scores: dict[str, float],
    scenes: list[str],
    k: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exact top-k sets only over an observed boundary tie."""
    values = np.asarray([scores[scene] for scene in scenes], dtype=float)
    selected = deterministic_top_set(scores, k)
    boundary = min(scores[scene] for scene in selected)
    tied_mask = np.fromiter(
        (_is_boundary_tie(value, boundary) for value in values),
        dtype=bool,
        count=len(values),
    )
    tied = np.flatnonzero(tied_mask)
    above = np.flatnonzero((values > boundary) & ~tied_mask)
    remaining = k - len(above)
    if not 0 <= remaining <= len(tied):
        raise RuntimeError("Boundary partition cannot produce an exact top-k set")
    masks = np.zeros((replicates, len(scenes)), dtype=bool)
    masks[:, above] = True
    if remaining:
        order = np.argsort(rng.random((replicates, len(tied))), axis=1)
        chosen = tied[order[:, :remaining]]
        masks[np.arange(replicates)[:, None], chosen] = True
    if not np.all(masks.sum(axis=1) == k):
        raise RuntimeError("Sampled boundary-tie masks are not exact top-k sets")
    return masks


def _opportunity_null_counts(
    target_scores: dict[str, float],
    target_opportunity: dict[str, float],
    scenes: list[str],
    scene_to_log: dict[str, str],
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Redistribute integer failures within logs over clean-TP opportunities."""
    draws = np.zeros((replicates, len(scenes)), dtype=int)
    scene_position = {scene: index for index, scene in enumerate(scenes)}
    by_log: dict[str, list[str]] = defaultdict(list)
    for scene in scenes:
        by_log[scene_to_log[scene]].append(scene)
    for log_scenes in by_log.values():
        log_scenes = sorted(log_scenes)
        colors = np.asarray(
            [round(target_opportunity[scene]) for scene in log_scenes],
            dtype=int,
        )
        failures = round(sum(target_scores[scene] for scene in log_scenes))
        if failures < 0 or failures > int(colors.sum()):
            raise ValueError("Lost counts must lie within clean-TP opportunities")
        positions = np.asarray([scene_position[scene] for scene in log_scenes])
        draws[:, positions] = rng.multivariate_hypergeometric(
            colors, failures, size=replicates
        )
    return draws


def opportunity_conditional_randomization_test(
    table: dict[tuple[str, str], dict[str, float]],
    opportunity_table: dict[tuple[str, str], dict[str, float]],
    scene_to_log: dict[str, str],
    fraction: float,
    replicates: int,
    seed: int,
) -> dict:
    """Test count transfer beyond log-specific clean-TP opportunity.

    The source top set is held fixed apart from exact boundary ties. Target
    failures are redistributed within each log across its clean true-positive
    opportunities while preserving every target detector-condition log total.
    """
    rows = compute_cells(
        table,
        scene_to_log,
        fraction,
        opportunity_table=opportunity_table,
    )
    scenes = sorted(scene_to_log)
    k = rows[0]["top_k"]
    rng = np.random.default_rng(seed)
    source_masks: dict[tuple[str, str], np.ndarray] = {}
    target_draws: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        source_key = (row["source"], row["condition"])
        if source_key not in source_masks:
            source_masks[source_key] = _sample_observed_top_masks(
                table[source_key], scenes, k, replicates, rng
            )
        target_key = (row["target"], row["condition"])
        if target_key not in target_draws:
            target_draws[target_key] = _opportunity_null_counts(
                table[target_key],
                opportunity_table[target_key],
                scenes,
                scene_to_log,
                replicates,
                rng,
            )

    overlap_null = np.empty((replicates, len(rows)), dtype=float)
    coverage_null = np.empty((replicates, len(rows)), dtype=float)
    for column, row in enumerate(rows):
        masks = source_masks[(row["source"], row["condition"])]
        draws = target_draws[(row["target"], row["condition"])]
        # Random jitter breaks simulated count ties without changing count order.
        target_order = np.argsort(
            -draws + 0.25 * rng.random(draws.shape),
            axis=1,
            kind="stable",
        )[:, :k]
        target_masks = np.zeros_like(masks)
        target_masks[np.arange(replicates)[:, None], target_order] = True
        overlap_null[:, column] = np.sum(masks & target_masks, axis=1) / k
        oracle_mass = np.take_along_axis(draws, target_order, axis=1).sum(axis=1)
        captured_mass = np.sum(draws * masks, axis=1)
        coverage_null[:, column] = np.divide(
            captured_mass,
            oracle_mass,
            out=np.zeros(replicates, dtype=float),
            where=oracle_mass > 0,
        )

    observed_overlap = median(
        float(row["tie_random_expected_raw_overlap"]) for row in rows
    )
    observed_coverage = median(
        float(row["tie_random_expected_RiskCoverage"]) for row in rows
    )
    null_overlap = np.median(overlap_null, axis=1)
    null_coverage = np.median(coverage_null, axis=1)

    def test_block(observed: float, null: np.ndarray) -> dict:
        return {
            "observed_tie_averaged_median": observed,
            "null_median": float(np.median(null)),
            "null_ci95": [
                float(np.quantile(null, 0.025)),
                float(np.quantile(null, 0.975)),
            ],
            "excess_over_null_median": observed - float(np.median(null)),
            "one_sided_p_greater": float(
                (1 + np.count_nonzero(null >= observed)) / (replicates + 1)
            ),
        }

    overlap = test_block(observed_overlap, null_overlap)
    coverage = test_block(observed_coverage, null_coverage)
    adjusted = _holm_adjust(
        {
            "overlap": overlap["one_sided_p_greater"],
            "RiskCoverage": coverage["one_sided_p_greater"],
        }
    )
    overlap["holm_adjusted_p_across_metrics"] = adjusted["overlap"]
    coverage["holm_adjusted_p_across_metrics"] = adjusted["RiskCoverage"]
    return {
        "replicates": replicates,
        "seed": seed,
        "null": (
            "within each target detector-condition acquisition log, observed "
            "lost-clean-TP totals are redistributed without replacement over "
            "scene clean-TP opportunities; source boundary ties and simulated "
            "target count ties are randomized"
        ),
        "overlap": overlap,
        "RiskCoverage": coverage,
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (count - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _leave_one_log_out(
    table: dict[tuple[str, str], dict[str, float]],
    scene_to_log: dict[str, str],
    fraction: float,
    opportunity_table: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict:
    results = {}
    for held_out in sorted(set(scene_to_log.values())):
        retained_mapping = {
            scene: log for scene, log in scene_to_log.items() if log != held_out
        }
        retained_table = {
            key: {
                scene: value
                for scene, value in scores.items()
                if scene in retained_mapping
            }
            for key, scores in table.items()
        }
        retained_opportunity = (
            {
                key: {
                    scene: value
                    for scene, value in opportunities.items()
                    if scene in retained_mapping
                }
                for key, opportunities in opportunity_table.items()
            }
            if opportunity_table is not None
            else None
        )
        cells = compute_cells(
            retained_table,
            retained_mapping,
            fraction,
            opportunity_table=retained_opportunity,
        )
        results[held_out] = _summary(cells)
        results[held_out]["decision_consequence"] = decision_consequence(cells)
    return results


def analyze(
    primary_path: Path,
    mechanism_path: Path,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict:
    tables, opportunity_table, scene_to_log, provenance = _load_outcome_tables(
        primary_path, mechanism_path
    )
    outcomes: dict[str, dict] = {}
    p_overlap: dict[str, float] = {}
    p_coverage: dict[str, float] = {}
    for outcome_index, outcome in enumerate(OUTCOME_ORDER):
        fraction_results = {}
        outcome_opportunity = (
            opportunity_table if outcome == "lost_clean_tp_count" else None
        )
        for fraction in fractions:
            cells = compute_cells(
                tables[outcome],
                scene_to_log,
                fraction,
                opportunity_table=outcome_opportunity,
            )
            fraction_results[f"{fraction:.2f}"] = {
                "summary": _summary(cells),
                "cells": cells,
                "decision_consequence": decision_consequence(cells),
            }
        primary_key = "0.20"
        randomization = conditional_randomization_test(
            tables[outcome],
            scene_to_log,
            0.20,
            permutations,
            seed + 1000 * outcome_index,
        )
        p_overlap[outcome] = randomization["overlap"]["one_sided_p_greater"]
        p_coverage[outcome] = randomization["RiskCoverage"][
            "one_sided_p_greater"
        ]
        outcomes[outcome] = {
            "fractions": fraction_results,
            "primary_fraction": 0.20,
            "conditional_randomization": randomization,
            "leave_one_log_out": _leave_one_log_out(
                tables[outcome],
                scene_to_log,
                0.20,
                opportunity_table=outcome_opportunity,
            ),
            "primary_summary": fraction_results[primary_key]["summary"],
        }
        if outcome_opportunity is not None:
            outcomes[outcome]["opportunity_conditional_randomization"] = (
                opportunity_conditional_randomization_test(
                    tables[outcome],
                    outcome_opportunity,
                    scene_to_log,
                    0.20,
                    permutations,
                    seed + 10_000,
                )
            )
    overlap_holm = _holm_adjust(p_overlap)
    coverage_holm = _holm_adjust(p_coverage)
    for outcome in OUTCOME_ORDER:
        outcomes[outcome]["conditional_randomization"]["overlap"][
            "holm_adjusted_p_across_outcomes"
        ] = overlap_holm[outcome]
        outcomes[outcome]["conditional_randomization"]["RiskCoverage"][
            "holm_adjusted_p_across_outcomes"
        ] = coverage_holm[outcome]
    is_confirmation = provenance["partition"] == "confirmation"
    return {
        "artifact_type": "chance_corrected_log_stratified_transfer_analysis",
        "status": (
            "log_disjoint_confirmation_analysis_completed"
            if is_confirmation
            else "development_analysis_completed"
        ),
        "evidence_scope": (
            "frozen log-disjoint confirmation partition only"
            if is_confirmation
            else "39-panel study only"
        ),
        "provenance": provenance,
        "scene_count": len(scene_to_log),
        "log_count": len(set(scene_to_log.values())),
        "model_count": len(MODEL_ORDER),
        "condition_count": len(
            {condition for _, condition in tables[OUTCOME_ORDER[0]]}
        ),
        "correction_contract": {
            "identity_null": (
                "independent uniform source/target sets conditional on actual "
                "per-log selected counts"
            ),
            "risk_mass_null": (
                "uniform source set within each log, preserving actual source "
                "per-log selected counts"
            ),
            "absolute_target_risk_coverage": (
                "captured positive target loss divided by positive target loss "
                "over all scenes"
            ),
            "raw_RiskCoverage": (
                "captured positive target loss divided by the target top-k "
                "oracle mass; this is oracle-relative budget efficiency, not "
                "coverage of all target loss"
            ),
            "adjusted_overlap": "(I-E0[I])/(k-E0[I])",
            "adjusted_RiskCoverage": "(Coverage-E0[Coverage])/(1-E0[Coverage])",
            "continuous_sensitivity": "Spearman correlation of within-log ranks",
            "tie_sensitivity": (
                "exact expectation under independent uniform random selection "
                "among scenes tied at the top-k boundary"
            ),
            "primary_test": "within-log conditional randomization, one-sided",
            "multiplicity": (
                "Holm correction across the three outcomes for each "
                "log-conditional metric family, and across overlap and "
                "RiskCoverage for the post-hoc opportunity-null family"
            ),
            "opportunity_null": (
                "for lost counts only, failures are redistributed within target "
                "detector-condition logs over scene clean-TP opportunities while "
                "preserving observed log totals"
            ),
        },
        "outcomes": outcomes,
        "claim_boundary": (
            (
                "Transfer analysis on 18 follow-up scenes from nine separate "
                "acquisition logs, evaluated under six signed rotations."
            )
            if is_confirmation
            else (
                "Retrospective transfer analysis on 50 development scenes, "
                "three detectors, and 12 signed calibration perturbations."
            )
        ),
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Chance-corrected stress-set transfer",
        "",
        f"Scope: {payload['evidence_scope']}.",
        "",
        "## Primary top-20% result",
        "",
        "| Outcome | Raw overlap | Within-log null | Adjusted overlap | "
        "Tie-avg adjusted overlap | "
        "Total-mass coverage | Oracle-relative coverage | Random coverage | "
        "Adjusted coverage | "
        "Opportunity-adjusted coverage | Rank rho | "
        "Overlap p (Holm) | Coverage p (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for outcome in OUTCOME_ORDER:
        block = payload["outcomes"][outcome]
        summary = block["primary_summary"]
        test = block["conditional_randomization"]
        opportunity_adjusted = summary[
            "log_opportunity_adjusted_RiskCoverage"
        ]["median"]
        opportunity_text = (
            f"{opportunity_adjusted:.3f}"
            if opportunity_adjusted is not None
            else "n/a"
        )
        lines.append(
            f"| {outcome} | "
            f"{summary['raw_overlap']['median']:.3f} | "
            f"{summary['log_stratified_chance_overlap']['median']:.3f} | "
            f"{summary['log_stratified_adjusted_overlap']['median']:.3f} | "
            f"{summary['tie_random_expected_log_stratified_adjusted_overlap']['median']:.3f} | "
            f"{summary['absolute_target_risk_coverage']['median']:.3f} | "
            f"{summary['raw_RiskCoverage']['median']:.3f} | "
            f"{summary['log_stratified_random_RiskCoverage']['median']:.3f} | "
            f"{summary['log_stratified_adjusted_RiskCoverage']['median']:.3f} | "
            f"{opportunity_text} | "
            f"{summary['within_log_rank_correlation']['median']:.3f} | "
            f"{test['overlap']['one_sided_p_greater']:.4f} "
            f"({test['overlap']['holm_adjusted_p_across_outcomes']:.4f}) | "
            f"{test['RiskCoverage']['one_sided_p_greater']:.4f} "
            f"({test['RiskCoverage']['holm_adjusted_p_across_outcomes']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "The within-log null preserves the number of selected scenes from "
            "each acquisition log. A corrected value of zero means no transfer "
            "beyond set size and log composition; negative values mean worse "
            "than that null.",
            "",
            "For lost counts, opportunity adjustment additionally preserves each "
            "target detector-condition log's total failures and uses clean true "
            "positives as the scene-level failure opportunity.",
            "",
            "## Cutoff and leave-one-log sensitivity",
            "",
        ]
    )
    for outcome in OUTCOME_ORDER:
        block = payload["outcomes"][outcome]
        lines.append(f"### {outcome}")
        lines.append("")
        for fraction, result in block["fractions"].items():
            summary = result["summary"]
            lines.append(
                f"- top-{float(fraction):.0%}: raw/adjusted overlap "
                f"{summary['raw_overlap']['median']:.3f}/"
                f"{summary['log_stratified_adjusted_overlap']['median']:.3f}; "
                f"raw/adjusted coverage "
                f"{summary['raw_RiskCoverage']['median']:.3f}/"
                f"{summary['log_stratified_adjusted_RiskCoverage']['median']:.3f}."
            )
        lolo = block["leave_one_log_out"]
        overlap_values = [
            row["log_stratified_adjusted_overlap"]["median"]
            for row in lolo.values()
        ]
        coverage_values = [
            row["log_stratified_adjusted_RiskCoverage"]["median"]
            for row in lolo.values()
        ]
        lines.append(
            f"- Leave-one-log-out adjusted-overlap median range: "
            f"{min(overlap_values):.3f} to {max(overlap_values):.3f}; "
            f"adjusted-coverage range: {min(coverage_values):.3f} to "
            f"{max(coverage_values):.3f}."
        )
        opportunity_values = [
            row["log_opportunity_adjusted_RiskCoverage"]["median"]
            for row in lolo.values()
            if row["log_opportunity_adjusted_RiskCoverage"]["median"] is not None
        ]
        if opportunity_values:
            lines.append(
                "- Leave-one-log-out opportunity-adjusted coverage range: "
                f"{min(opportunity_values):.3f} to "
                f"{max(opportunity_values):.3f}."
            )
        opportunity_test = block.get("opportunity_conditional_randomization")
        if opportunity_test is not None:
            lines.append(
                "- Opportunity-conditioned randomization: tie-averaged median "
                f"overlap {opportunity_test['overlap']['observed_tie_averaged_median']:.3f} "
                f"vs null {opportunity_test['overlap']['null_median']:.3f}, "
                "Holm p="
                f"{opportunity_test['overlap']['holm_adjusted_p_across_metrics']:.4f}; "
                "median RiskCoverage "
                f"{opportunity_test['RiskCoverage']['observed_tie_averaged_median']:.3f} "
                f"vs null {opportunity_test['RiskCoverage']['null_median']:.3f}, "
                "Holm p="
                f"{opportunity_test['RiskCoverage']['holm_adjusted_p_across_metrics']:.4f}."
            )
        lines.append("")
    decision = payload["outcomes"]["lost_clean_tp_count"]["fractions"]["0.20"][
        "decision_consequence"
    ]
    raw_decision = decision["rules"]["raw_overlap"]["summary"]
    adjusted_decision = decision["rules"]["log_stratified_adjusted_overlap"][
        "summary"
    ]
    relation = decision["coverage_efficiency_relation"]
    lines.extend(
        [
            "## Retrospective reuse-regret decomposition",
            "",
            "For each target detector and condition, this diagnostic chooses "
            "between the two non-target source sets by identity overlap and "
            "decomposes the chosen set's target-oracle regret into the regret "
            "remaining for the best available source and the additional regret "
            "from identity-guided selection. Exact boundary and source-choice "
            "ties are averaged. The diagnostic uses observed target outcomes.",
            "",
            f"- Raw identity: {raw_decision['strict_reversals']}/"
            f"{raw_decision['decision_units']} strict source-order reversals; "
            f"{raw_decision['positive_selection_regret_units']}/"
            f"{raw_decision['decision_units']} positive selection-regret choices; mean "
            f"total/availability/selection regret "
            f"{raw_decision['mean_total_reuse_regret']:.4f}/"
            f"{raw_decision['mean_availability_regret']:.4f}/"
            f"{raw_decision['mean_selection_regret']:.4f}; selection accounts "
            "for "
            f"{raw_decision['selection_share_of_mean_total_regret']:.1%} "
            "of mean total regret.",
            f"- Log-adjusted identity: {adjusted_decision['strict_reversals']}/"
            f"{adjusted_decision['decision_units']} strict reversals; mean "
            "total/availability/selection regret "
            f"{adjusted_decision['mean_total_reuse_regret']:.4f}/"
            f"{adjusted_decision['mean_availability_regret']:.4f}/"
            f"{adjusted_decision['mean_selection_regret']:.4f}.",
            "- Exact regret decomposition: "
            f"`{decision['regret_decomposition']['formula']}`; maximum "
            "absolute reconstruction error "
            f"{raw_decision['max_regret_decomposition_error']:.3g}.",
            f"- Exact scale relation checked in {relation['checked_cells']} "
            "directed cells: "
            f"`{relation['formula']}`; maximum absolute reconstruction error "
            f"{relation['max_absolute_error']:.3g}.",
            "",
        ]
    )
    lines.extend(
        [
            "## Analysis scope",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("data/primary_checkpoint.json"),
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=Path("data/mechanism.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/development.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/development.md"),
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.permutations < 999:
        raise ValueError("At least 999 permutations are required")
    payload = analyze(
        args.primary,
        args.mechanism,
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
