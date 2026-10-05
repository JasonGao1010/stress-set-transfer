"""Failure-sharing references and cross-fitted target-risk analysis."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from transfer import compute_cells
except ImportError:
    from analysis.transfer import compute_cells


MODEL_ORDER = ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
METRICS = ("L_status", "lost_clean_tp_count")
QUANTILES = (0.80, 0.90, 0.95)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260721
IDENTIFICATION_REPLICATES = 10_000
IDENTIFICATION_SEED = 20260726
TARGET_FREQUENCY_SEED = 20260727
TARGET_FREQUENCY_DIAGNOSTIC_CHAINS = 4
TARGET_FREQUENCY_DIAGNOSTIC_REPLICATES = 2_500
TRANSFER_SEED = 20260726
SENSITIVITY_REPLICATES = 2_000
SENSITIVITY_CONFIGS = tuple(
    (score, distance) for score in (0.20, 0.25, 0.30) for distance in (1.5, 2.0, 2.5)
)
FAILURE_PATHS = (
    "score_collapse",
    "localization_drift",
    "class_switch",
    "compound_degradation",
    "assignment_competition",
    "output_disappearance",
)

TRANSFER_CONTEXT_NUMERIC = ("log_distance_m", "log_size_m3", "visibility")
TRANSFER_CATEGORICAL = ("condition", "class")
TRANSFER_SPECIFICATIONS = {
    "context": TRANSFER_CONTEXT_NUMERIC,
    "vulnerability": (*TRANSFER_CONTEXT_NUMERIC, "v_clean"),
    "source": (*TRANSFER_CONTEXT_NUMERIC, "source_failure_rate"),
    "vulnerability_source_rate": (
        *TRANSFER_CONTEXT_NUMERIC,
        "v_clean",
        "source_failure_rate",
    ),
}
MEAN_SINGLE_SOURCE_SPECIFICATION = "vulnerability_source_mean_single"


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must have positive total mass")
    return sum(value * weight for value, weight in zip(values, weights)) / total


def weighted_cvar(
    values: list[float], quantile: float, weights: list[float] | None = None
) -> float:
    """Return the exact upper-tail empirical mean with fractional boundary mass."""
    if not values:
        raise ValueError("CVaR requires at least one value")
    if not 0 <= quantile < 1:
        raise ValueError("CVaR quantile must lie in [0, 1)")
    if weights is None:
        weights = [1.0] * len(values)
    if len(values) != len(weights):
        raise ValueError("Values and weights must have equal length")
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0], reverse=True)
    tail_mass = (1.0 - quantile) * sum(weights)
    if tail_mass <= 0:
        raise ValueError("CVaR tail has zero mass")
    remaining = tail_mass
    weighted_sum = 0.0
    for value, weight in pairs:
        take = min(weight, remaining)
        weighted_sum += value * take
        remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-9:
        raise RuntimeError("CVaR failed to consume the requested tail mass")
    return weighted_sum / tail_mass


def condition_summary(
    rows: list[dict], metric: str, equal_log_weight: bool = False
) -> dict:
    values = [float(row[metric]) for row in rows]
    if equal_log_weight:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["log"]] += 1
        weights = [1.0 / counts[row["log"]] for row in rows]
    else:
        weights = [1.0] * len(rows)
    result = {"mean": weighted_mean(values, weights)}
    for quantile in QUANTILES:
        result[f"CVaR{int(quantile * 100)}"] = weighted_cvar(values, quantile, weights)
    result["CVaR90_minus_mean"] = result["CVaR90"] - result["mean"]
    result["CVaR90_over_mean"] = (
        result["CVaR90"] / result["mean"] if result["mean"] > 0 else None
    )
    return result


def average_condition_summaries(summaries: list[dict]) -> dict:
    result = {
        key: fmean(row[key] for row in summaries)
        for key in ("mean", "CVaR80", "CVaR90", "CVaR95")
    }
    result["CVaR90_minus_mean"] = result["CVaR90"] - result["mean"]
    result["CVaR90_over_mean"] = (
        result["CVaR90"] / result["mean"] if result["mean"] > 0 else None
    )
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_tail(
    grouped: dict[tuple[str, str], list[dict]],
    model: str,
    conditions: list[str],
    logs: list[str],
    metric: str,
    draws: list[list[str]],
) -> dict:
    by_log: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for condition in conditions:
        for row in grouped[(model, condition)]:
            by_log[(condition, row["log"])].append(row)
    statistics = {
        "mean": [],
        "CVaR90": [],
        "CVaR90_minus_mean": [],
        "CVaR90_over_mean": [],
    }
    for sampled_logs in draws:
        summaries = []
        for condition in conditions:
            sample = [row for log in sampled_logs for row in by_log[(condition, log)]]
            summaries.append(condition_summary(sample, metric))
        aggregate = average_condition_summaries(summaries)
        for key in statistics:
            statistics[key].append(aggregate[key])
    return {
        "replicates": len(draws),
        "seed": BOOTSTRAP_SEED,
        "unit": "nuScenes_acquisition_log",
        "ci95": {
            key: [percentile(values, 0.025), percentile(values, 0.975)]
            for key, values in statistics.items()
        },
    }


def summarize_tail(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    logs = sorted({row["log"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[rng.choice(logs) for _ in logs] for _ in range(BOOTSTRAP_REPLICATES)]
    for row in rows:
        grouped[(row["model"], row["condition"])].append(row)

    result = {}
    for metric in METRICS:
        metric_result = {}
        for model in MODEL_ORDER:
            ordinary = [
                condition_summary(grouped[(model, condition)], metric)
                for condition in conditions
            ]
            log_balanced = [
                condition_summary(
                    grouped[(model, condition)], metric, equal_log_weight=True
                )
                for condition in conditions
            ]

            lolo = {}
            for held_log in logs:
                summaries = []
                for condition in conditions:
                    subset = [
                        row
                        for row in grouped[(model, condition)]
                        if row["log"] != held_log
                    ]
                    summaries.append(condition_summary(subset, metric))
                lolo[held_log] = average_condition_summaries(summaries)
            metric_result[model] = {
                "scene_equal": average_condition_summaries(ordinary),
                "log_equal": average_condition_summaries(log_balanced),
                "log_cluster_bootstrap": cluster_bootstrap_tail(
                    grouped, model, conditions, logs, metric, draws
                ),
                "leave_one_log_out": lolo,
                "lolo_min_CVaR90_minus_mean": min(
                    row["CVaR90_minus_mean"] for row in lolo.values()
                ),
                "lolo_min_CVaR90_over_mean": min(
                    row["CVaR90_over_mean"]
                    for row in lolo.values()
                    if row["CVaR90_over_mean"] is not None
                ),
            }
        result[metric] = metric_result
    return result


def directed_transfer(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition"])].append(row)
    conditions = sorted({row["condition"] for row in rows})
    result = []
    for condition in conditions:
        for source in MODEL_ORDER:
            for target in MODEL_ORDER:
                if source == target:
                    continue
                source_rows = grouped[(source, condition)]
                target_rows = grouped[(target, condition)]
                target_by_scene = {row["scene"]: row for row in target_rows}
                if {row["scene"] for row in source_rows} != set(target_by_scene):
                    raise ValueError("Source and target scene sets differ")
                top_k = max(1, math.ceil(0.20 * len(source_rows)))
                source_top = sorted(
                    source_rows, key=lambda row: (-float(row[metric]), row["scene"])
                )[:top_k]
                target_top = sorted(
                    target_rows, key=lambda row: (-float(row[metric]), row["scene"])
                )[:top_k]
                source_scenes = {row["scene"] for row in source_top}
                target_scenes = {row["scene"] for row in target_top}
                captured = sum(
                    max(float(target_by_scene[scene][metric]), 0.0)
                    for scene in source_scenes
                )
                oracle = sum(
                    max(float(target_by_scene[scene][metric]), 0.0)
                    for scene in target_scenes
                )
                coverage = captured / oracle if oracle > 0 else None
                result.append(
                    {
                        "condition": condition,
                        "source": source,
                        "target": target,
                        "top_k": top_k,
                        "overlap": len(source_scenes & target_scenes) / top_k,
                        "RiskCoverage": coverage,
                        "RiskSetTransferRegret": (
                            1.0 - coverage if coverage is not None else None
                        ),
                    }
                )
    return result


def summarize_transfer(rows: list[dict]) -> dict:
    logs = sorted({row["log"] for row in rows})
    summary = {}
    for metric in METRICS:
        primary = directed_transfer(rows, metric)
        estimable = [row for row in primary if row["RiskSetTransferRegret"] is not None]
        regrets = [row["RiskSetTransferRegret"] for row in estimable]
        overlaps = [row["overlap"] for row in estimable]
        lolo_by_row: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for held_log in logs:
            subset = [row for row in rows if row["log"] != held_log]
            for row in directed_transfer(subset, metric):
                regret = row["RiskSetTransferRegret"]
                if regret is not None:
                    key = (row["condition"], row["source"], row["target"])
                    lolo_by_row[key].append(regret)
        summary[metric] = {
            "rows": primary,
            "estimable_rows": len(estimable),
            "regret": {
                "min": min(regrets),
                "median": median(regrets),
                "mean": fmean(regrets),
                "max": max(regrets),
            },
            "overlap": {
                "min": min(overlaps),
                "median": median(overlaps),
                "mean": fmean(overlaps),
                "max": max(overlaps),
            },
            "lolo_rows_estimable_in_all_logs": sum(
                len(values) == len(logs) for values in lolo_by_row.values()
            ),
            "lolo_rows_positive_in_all_logs": sum(
                len(values) == len(logs) and min(values) > 0
                for values in lolo_by_row.values()
            ),
            "lolo_rows_at_or_above_0_25_in_all_logs": sum(
                len(values) == len(logs) and min(values) >= 0.25
                for values in lolo_by_row.values()
            ),
            "lolo_min_regret": min(
                value for values in lolo_by_row.values() for value in values
            ),
        }
    return summary


def summarize_failure_sensitivity(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["assignment_mode"],
            float(row["score_threshold"]),
            float(row["match_distance"]),
        )
        grouped[key].append(row)
    configurations = []
    for (mode, score, distance), cells in sorted(grouped.items()):
        totals = {path: 0 for path in FAILURE_PATHS}
        dominant_cells = 0
        for cell in cells:
            for path in FAILURE_PATHS:
                totals[path] += int(cell["counts"][path])
            cell_max = max(cell["counts"][path] for path in FAILURE_PATHS)
            if cell["counts"]["score_collapse"] == cell_max:
                dominant_cells += 1
        nonstable = sum(totals.values())
        configurations.append(
            {
                "assignment_mode": mode,
                "score_threshold": score,
                "match_distance": distance,
                "cell_count": len(cells),
                "score_collapse_dominant_cells": dominant_cells,
                "score_collapse_share": totals["score_collapse"] / nonstable,
                "path_counts": totals,
            }
        )
    return {
        "configuration_count": len(configurations),
        "configurations": configurations,
        "all_cells_dominant_configuration_count": sum(
            row["score_collapse_dominant_cells"] == row["cell_count"]
            for row in configurations
        ),
        "min_score_collapse_share": min(
            row["score_collapse_share"] for row in configurations
        ),
        "max_score_collapse_share": max(
            row["score_collapse_share"] for row in configurations
        ),
    }


def _fraction_summary(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def _null_composition_summary(
    exactly_one: np.ndarray,
    at_least_two: np.ndarray,
    all_three: np.ndarray,
    pair_intersections: dict[str, np.ndarray],
    pair_unions: dict[str, np.ndarray],
) -> tuple[dict, np.ndarray, np.ndarray]:
    any_failure = exactly_one + at_least_two
    specific_fraction = np.divide(
        exactly_one,
        any_failure,
        out=np.zeros(len(any_failure), dtype=float),
        where=any_failure > 0,
    )
    shared_fraction = np.divide(
        at_least_two,
        any_failure,
        out=np.zeros(len(any_failure), dtype=float),
        where=any_failure > 0,
    )
    triple_fraction = np.divide(
        all_three,
        any_failure,
        out=np.zeros(len(any_failure), dtype=float),
        where=any_failure > 0,
    )
    return (
        {
            "specific_fraction_among_any_failure": _fraction_summary(specific_fraction),
            "shared_fraction_among_any_failure": _fraction_summary(shared_fraction),
            "triple_fraction_among_any_failure": _fraction_summary(triple_fraction),
            "pairwise_jaccard": {
                pair: _fraction_summary(
                    np.divide(
                        pair_intersections[pair],
                        pair_unions[pair],
                        out=np.zeros(len(any_failure), dtype=float),
                        where=pair_unions[pair] > 0,
                    )
                )
                for pair in pair_intersections
            },
        },
        specific_fraction,
        shared_fraction,
    )


def _mcmc_diagnostics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    quarter_medians = [
        float(np.median(block)) for block in np.array_split(values, 4) if len(block)
    ]
    centered = values - float(np.mean(values))
    variance = float(centered @ centered)
    if variance <= 1e-18:
        return {
            "lag1_autocorrelation": 0.0,
            "effective_sample_size": len(values),
            "quarter_medians": quarter_medians,
            "quarter_median_range": [
                min(quarter_medians),
                max(quarter_medians),
            ],
        }
    correlations = []
    for lag in range(1, min(1_000, len(values) // 2) + 1):
        correlation = float(centered[:-lag] @ centered[lag:] / variance)
        if correlation <= 0:
            break
        correlations.append(correlation)
    integrated_time = 1.0 + 2.0 * sum(correlations)
    return {
        "lag1_autocorrelation": (correlations[0] if correlations else 0.0),
        "positive_sequence_lags": len(correlations),
        "effective_sample_size": len(values) / integrated_time,
        "quarter_medians": quarter_medians,
        "quarter_median_range": [
            min(quarter_medians),
            max(quarter_medians),
        ],
    }


def marginal_failure_null(
    strata: list[dict],
    replicates: int = IDENTIFICATION_REPLICATES,
    seed: int = IDENTIFICATION_SEED,
) -> dict:
    """Match each detector's failure count within condition-log strata.

    Failure sets are drawn independently over the common clean-TP population.
    The construction preserves every detector-condition-log marginal exactly,
    so residual agreement cannot be attributed to different base failure rates
    or acquisition-log composition.
    """
    rng = np.random.default_rng(seed)
    exactly_one = np.zeros(replicates, dtype=np.int64)
    at_least_two = np.zeros(replicates, dtype=np.int64)
    all_three = np.zeros(replicates, dtype=np.int64)
    pair_intersections = {
        "bevfusion_mit__sparsefusion_r50": np.zeros(replicates, dtype=np.int64),
        "bevfusion_mit__deepinteraction_base": np.zeros(replicates, dtype=np.int64),
        "sparsefusion_r50__deepinteraction_base": np.zeros(replicates, dtype=np.int64),
    }
    pair_unions = {
        pair: np.zeros(replicates, dtype=np.int64) for pair in pair_intersections
    }
    by_log: dict[str, dict] = {}

    for stratum in strata:
        log = str(stratum["log"])
        if log not in by_log:
            by_log[log] = {
                "exactly_one": np.zeros(replicates, dtype=np.int64),
                "at_least_two": np.zeros(replicates, dtype=np.int64),
                "all_three": np.zeros(replicates, dtype=np.int64),
                "pair_intersections": {
                    pair: np.zeros(replicates, dtype=np.int64)
                    for pair in pair_intersections
                },
                "pair_unions": {
                    pair: np.zeros(replicates, dtype=np.int64)
                    for pair in pair_intersections
                },
            }
        n = int(stratum["n"])
        ka, kb, kc = (int(stratum["failures"][model]) for model in MODEL_ORDER)
        if any(value < 0 or value > n for value in (ka, kb, kc)):
            raise ValueError("Marginal failure counts must lie within the stratum")

        # Draw B over A, then allocate C over the four A/B cells.  This is
        # exactly equivalent to three independent fixed-size sets.
        ab = rng.hypergeometric(ka, n - ka, kb, size=replicates)
        c_ab = rng.hypergeometric(ab, n - ab, kc)
        c_remaining = kc - c_ab
        a_only_size = ka - ab
        c_a_only = rng.hypergeometric(a_only_size, n - ab - a_only_size, c_remaining)
        c_remaining = c_remaining - c_a_only
        b_only_size = kb - ab
        c_b_only = rng.hypergeometric(
            b_only_size,
            n - ka - b_only_size,
            c_remaining,
        )

        n111 = c_ab
        n110 = ab - c_ab
        n101 = c_a_only
        n011 = c_b_only
        n100 = a_only_size - c_a_only
        n010 = b_only_size - c_b_only
        n001 = kc - c_ab - c_a_only - c_b_only

        stratum_exactly_one = n100 + n010 + n001
        stratum_at_least_two = n110 + n101 + n011 + n111
        exactly_one += stratum_exactly_one
        at_least_two += stratum_at_least_two
        all_three += n111
        by_log[log]["exactly_one"] += stratum_exactly_one
        by_log[log]["at_least_two"] += stratum_at_least_two
        by_log[log]["all_three"] += n111

        pair_values = {
            "bevfusion_mit__sparsefusion_r50": ab,
            "bevfusion_mit__deepinteraction_base": n101 + n111,
            "sparsefusion_r50__deepinteraction_base": n011 + n111,
        }
        pair_sizes = {
            "bevfusion_mit__sparsefusion_r50": (ka, kb),
            "bevfusion_mit__deepinteraction_base": (ka, kc),
            "sparsefusion_r50__deepinteraction_base": (kb, kc),
        }
        for pair, intersection in pair_values.items():
            pair_intersections[pair] += intersection
            left, right = pair_sizes[pair]
            pair_unions[pair] += left + right - intersection
            by_log[log]["pair_intersections"][pair] += intersection
            by_log[log]["pair_unions"][pair] += left + right - intersection

    summary, specific_fraction, shared_fraction = _null_composition_summary(
        exactly_one,
        at_least_two,
        all_three,
        pair_intersections,
        pair_unions,
    )
    lolo = {}
    for held_log, block in by_log.items():
        lolo[held_log] = _null_composition_summary(
            exactly_one - block["exactly_one"],
            at_least_two - block["at_least_two"],
            all_three - block["all_three"],
            {
                pair: pair_intersections[pair] - block["pair_intersections"][pair]
                for pair in pair_intersections
            },
            {
                pair: pair_unions[pair] - block["pair_unions"][pair]
                for pair in pair_unions
            },
        )[0]
    return {
        "replicates": replicates,
        "seed": seed,
        "unit": "annotation-condition within detector-condition-log",
        **summary,
        "leave_one_log_out": lolo,
        "_specific_draws": specific_fraction,
        "_shared_draws": shared_fraction,
    }


def _switch_state(matrix: np.ndarray) -> dict:
    edge_rows, edge_columns = np.nonzero(matrix)
    masks = np.zeros(len(matrix), dtype=np.uint16)
    for column in range(matrix.shape[1]):
        masks |= matrix[:, column].astype(np.uint16) << column
    return {
        "masks": masks,
        "edge_rows": edge_rows.astype(np.int32),
        "edge_columns": edge_columns.astype(np.int16),
    }


def _switch_edges(state: dict, attempts: int, rng: np.random.Generator) -> int:
    """Apply binary two-edge swaps while preserving row and column margins."""
    rows = state["edge_rows"]
    columns = state["edge_columns"]
    masks = state["masks"]
    if len(rows) < 2:
        return 0
    accepted = 0
    left_draws = rng.integers(0, len(rows), size=attempts)
    right_draws = rng.integers(0, len(rows), size=attempts)
    for left, right in zip(left_draws, right_draws):
        row_left, row_right = int(rows[left]), int(rows[right])
        col_left, col_right = int(columns[left]), int(columns[right])
        if row_left == row_right or col_left == col_right:
            continue
        bit_left = np.uint16(1 << col_left)
        bit_right = np.uint16(1 << col_right)
        if masks[row_left] & bit_right or masks[row_right] & bit_left:
            continue
        masks[row_left] ^= bit_left | bit_right
        masks[row_right] ^= bit_left | bit_right
        columns[left], columns[right] = col_right, col_left
        accepted += 1
    return accepted


def target_frequency_null(
    common: pd.DataFrame,
    replicates: int = IDENTIFICATION_REPLICATES,
    seed: int = TARGET_FREQUENCY_SEED,
    condition_blocks: tuple[tuple[str, ...], ...] | None = None,
) -> dict:
    """Preserve target-frequency and detector-condition-log binary margins.

    Independent constrained switch chains are run for every detector-log
    block. Each matrix keeps an annotation's number of failures within the
    block and every condition's observed failure total.
    """
    conditions = sorted(common["condition"].unique())
    logs = sorted(common["log"].unique())
    if len(conditions) > 16:
        raise ValueError("Bit-mask switch randomization supports at most 16 conditions")
    if condition_blocks is None:
        condition_blocks = (tuple(conditions),)
    flattened = [condition for block in condition_blocks for condition in block]
    if sorted(flattened) != conditions or len(flattened) != len(set(flattened)):
        raise ValueError("Condition blocks must partition the observed conditions")
    positions = {condition: index for index, condition in enumerate(conditions)}
    for block in condition_blocks:
        block_positions = [positions[condition] for condition in block]
        if block_positions != list(
            range(block_positions[0], block_positions[0] + len(block_positions))
        ):
            raise ValueError("Condition blocks must be contiguous in sorted order")
    rng = np.random.default_rng(seed)
    states: dict[tuple[str, str, int], dict] = {}
    annotations_by_log: dict[str, list[str]] = {}
    for log in logs:
        annotations = sorted(
            common.loc[common["log"] == log, "annotation_token"].unique()
        )
        annotations_by_log[str(log)] = annotations
        for model in MODEL_ORDER:
            for block_index, block in enumerate(condition_blocks):
                matrix = (
                    common.loc[
                        common["log"] == log,
                        ["annotation_token", "condition", model],
                    ]
                    .pivot(
                        index="annotation_token",
                        columns="condition",
                        values=model,
                    )
                    .reindex(index=annotations, columns=list(block))
                )
                if matrix.isna().any().any():
                    raise ValueError("Common cohort must be complete across conditions")
                state = _switch_state(matrix.to_numpy(dtype=bool))
                state["bit_offset"] = positions[block[0]]
                states[(model, str(log), block_index)] = state

    attempted = 0
    accepted = 0
    for state in states.values():
        burn_in = max(2_000, 20 * len(state["edge_rows"]))
        attempted += burn_in
        accepted += _switch_edges(state, burn_in, rng)

    pair_names = (
        "bevfusion_mit__sparsefusion_r50",
        "bevfusion_mit__deepinteraction_base",
        "sparsefusion_r50__deepinteraction_base",
    )
    per_log = {
        log: {
            "exactly_one": np.zeros(replicates, dtype=np.int64),
            "at_least_two": np.zeros(replicates, dtype=np.int64),
            "all_three": np.zeros(replicates, dtype=np.int64),
            "pair_intersections": {
                pair: np.zeros(replicates, dtype=np.int64) for pair in pair_names
            },
            "pair_unions": {
                pair: np.zeros(replicates, dtype=np.int64) for pair in pair_names
            },
        }
        for log in logs
    }
    popcount = np.asarray([value.bit_count() for value in range(1 << len(conditions))])

    def current_masks(model: str, log: str) -> np.ndarray:
        masks = np.zeros(len(annotations_by_log[str(log)]), dtype=np.uint16)
        for block_index in range(len(condition_blocks)):
            state = states[(model, str(log), block_index)]
            masks |= state["masks"] << np.uint16(state["bit_offset"])
        return masks

    for replicate in range(replicates):
        for state in states.values():
            thinning = max(50, len(state["edge_rows"]) // 5)
            attempted += thinning
            accepted += _switch_edges(state, thinning, rng)
        for log in logs:
            a, b, c = (current_masks(model, str(log)) for model in MODEL_ORDER)
            only_one = ((a & ~b & ~c) | (~a & b & ~c) | (~a & ~b & c)) & np.uint16(
                (1 << len(conditions)) - 1
            )
            shared = (a & b) | (a & c) | (b & c)
            block = per_log[log]
            block["exactly_one"][replicate] = int(popcount[only_one].sum())
            block["at_least_two"][replicate] = int(popcount[shared].sum())
            block["all_three"][replicate] = int(popcount[a & b & c].sum())
            pair_masks = {
                pair_names[0]: (a, b),
                pair_names[1]: (a, c),
                pair_names[2]: (b, c),
            }
            for pair, (left, right) in pair_masks.items():
                block["pair_intersections"][pair][replicate] = int(
                    popcount[left & right].sum()
                )
                block["pair_unions"][pair][replicate] = int(
                    popcount[left | right].sum()
                )

    exactly_one = sum(
        (block["exactly_one"] for block in per_log.values()),
        start=np.zeros(replicates, dtype=np.int64),
    )
    at_least_two = sum(
        (block["at_least_two"] for block in per_log.values()),
        start=np.zeros(replicates, dtype=np.int64),
    )
    all_three = sum(
        (block["all_three"] for block in per_log.values()),
        start=np.zeros(replicates, dtype=np.int64),
    )
    pair_intersections = {
        pair: sum(
            (block["pair_intersections"][pair] for block in per_log.values()),
            start=np.zeros(replicates, dtype=np.int64),
        )
        for pair in pair_names
    }
    pair_unions = {
        pair: sum(
            (block["pair_unions"][pair] for block in per_log.values()),
            start=np.zeros(replicates, dtype=np.int64),
        )
        for pair in pair_names
    }
    summary, specific_draws, shared_draws = _null_composition_summary(
        exactly_one,
        at_least_two,
        all_three,
        pair_intersections,
        pair_unions,
    )
    lolo = {}
    for held_log, block in per_log.items():
        lolo[str(held_log)] = _null_composition_summary(
            exactly_one - block["exactly_one"],
            at_least_two - block["at_least_two"],
            all_three - block["all_three"],
            {
                pair: pair_intersections[pair] - block["pair_intersections"][pair]
                for pair in pair_names
            },
            {
                pair: pair_unions[pair] - block["pair_unions"][pair]
                for pair in pair_names
            },
        )[0]
    return {
        "replicates": replicates,
        "seed": seed,
        "unit": "annotation-condition within detector-log",
        "null": (
            "independent detector-log-block binary switch chains preserving every "
            "annotation row sum within each block and every condition column sum"
        ),
        "condition_blocks": [list(block) for block in condition_blocks],
        "burn_in_attempts_per_matrix": "max(2000, 20 * failure_edges)",
        "thinning_attempts_per_draw_per_matrix": "max(50, failure_edges / 5)",
        "attempted_swaps": attempted,
        "accepted_swaps": accepted,
        "acceptance_rate": accepted / attempted if attempted else None,
        "shared_fraction_chain_diagnostics": _mcmc_diagnostics(shared_draws),
        **summary,
        "leave_one_log_out": lolo,
        "_specific_draws": specific_draws,
        "_shared_draws": shared_draws,
    }


def _classic_r_hat(chains: list[np.ndarray]) -> float | None:
    """Return the classical between/within-chain potential scale reduction."""
    if len(chains) < 2:
        return None
    length = min(len(chain) for chain in chains)
    if length < 2:
        return None
    values = np.stack([np.asarray(chain[:length], dtype=float) for chain in chains])
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(length * np.var(np.mean(values, axis=1), ddof=1))
    if within <= 1e-18:
        return 1.0 if between <= 1e-18 else math.inf
    variance = ((length - 1) / length) * within + between / length
    return float(math.sqrt(variance / within))


def _split_r_hat(chains: list[np.ndarray]) -> float | None:
    """Split each chain before applying the classical R-hat diagnostic."""
    if not chains:
        return None
    half = min(len(chain) for chain in chains) // 2
    if half < 2:
        return None
    split = []
    for chain in chains:
        values = np.asarray(chain, dtype=float)
        split.extend((values[:half], values[-half:]))
    return _classic_r_hat(split)


def multi_chain_target_frequency_diagnostics(
    common: pd.DataFrame,
    *,
    replicates_per_chain: int = TARGET_FREQUENCY_DIAGNOSTIC_REPLICATES,
    chains: int = TARGET_FREQUENCY_DIAGNOSTIC_CHAINS,
    seed: int = TARGET_FREQUENCY_SEED,
    condition_blocks: tuple[tuple[str, ...], ...] | None = None,
) -> dict:
    """Run independent randomized switch chains and summarize convergence.

    Every chain starts its retained sequence after a seed-specific randomized
    burn-in. The retained states diagnose the reference distribution; they are
    not independent data and are never converted into additional p-values.
    """
    if chains < 2:
        raise ValueError("Multi-chain diagnostics require at least two chains")
    if replicates_per_chain < 4:
        raise ValueError("Each diagnostic chain requires at least four states")
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(seed).spawn(chains)
    ]
    results = [
        target_frequency_null(
            common,
            replicates=replicates_per_chain,
            seed=chain_seed,
            condition_blocks=condition_blocks,
        )
        for chain_seed in seeds
    ]

    metrics = {}
    for private_key, label in (
        ("_specific_draws", "specific_fraction"),
        ("_shared_draws", "shared_fraction"),
    ):
        draws = [np.asarray(result[private_key], dtype=float) for result in results]
        chain_rows = []
        for index, (chain_seed, result, values) in enumerate(
            zip(seeds, results, draws), start=1
        ):
            diagnostics = _mcmc_diagnostics(values)
            chain_rows.append(
                {
                    "chain": index,
                    "seed": chain_seed,
                    "retained_states": len(values),
                    "median": float(np.median(values)),
                    "range": [float(np.min(values)), float(np.max(values))],
                    "acceptance_rate": result["acceptance_rate"],
                    "effective_sample_size": diagnostics["effective_sample_size"],
                    "lag1_autocorrelation": diagnostics["lag1_autocorrelation"],
                }
            )
        pooled = np.concatenate(draws)
        metrics[label] = {
            "chains": chain_rows,
            "chain_median_range": [
                min(row["median"] for row in chain_rows),
                max(row["median"] for row in chain_rows),
            ],
            "pooled_range": [float(np.min(pooled)), float(np.max(pooled))],
            "total_within_chain_effective_sample_size": float(
                sum(row["effective_sample_size"] for row in chain_rows)
            ),
            "classic_r_hat": _classic_r_hat(draws),
            "split_r_hat": _split_r_hat(draws),
        }
    return {
        "chain_count": chains,
        "retained_states_per_chain": replicates_per_chain,
        "condition_blocks": results[0]["condition_blocks"],
        "initialization": (
            "independent seed-specific switch burn-ins before retained states"
        ),
        "diagnostic_boundary": (
            "Retained Markov-chain states are autocorrelated reference draws. "
            "R-hat, per-chain ranges, acceptance, and ESS assess mixing; chains "
            "do not provide independent p-values."
        ),
        "metrics": metrics,
    }


def signed_axis_condition_blocks(
    conditions: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Pair the negative and positive LOW conditions for each physical axis."""
    by_axis: dict[str, list[str]] = defaultdict(list)
    for condition in sorted(conditions):
        parts = str(condition).split("_")
        if len(parts) != 4 or parts[0] != "D" or parts[3] != "LOW":
            raise ValueError(f"Unexpected condition name: {condition}")
        by_axis[parts[1]].append(str(condition))
    canonical = ("RX", "RY", "RZ", "TX", "TY", "TZ")
    if not by_axis or any(axis not in canonical for axis in by_axis):
        raise ValueError("Conditions must contain recognized signed physical axes")
    blocks = tuple(
        tuple(sorted(by_axis[axis])) for axis in canonical if axis in by_axis
    )
    if any(
        len(block) != 2
        or not any("NEGATIVE" in condition for condition in block)
        or not any("POSITIVE" in condition for condition in block)
        for block in blocks
    ):
        raise ValueError("Every axis block must contain one negative and one positive")
    return blocks


def _failure_composition(frame: pd.DataFrame) -> dict:
    failures = frame[list(MODEL_ORDER)].astype(bool).sum(axis=1)
    counts = {
        f"{number}_models": int((failures == number).sum()) for number in range(4)
    }
    any_failure = len(frame) - counts["0_models"]
    return {
        "counts": counts,
        "any_failure": any_failure,
        "specific_fraction_among_any_failure": (
            counts["1_models"] / any_failure if any_failure else None
        ),
        "shared_fraction_among_any_failure": (
            (counts["2_models"] + counts["3_models"]) / any_failure
            if any_failure
            else None
        ),
        "triple_fraction_among_any_failure": (
            counts["3_models"] / any_failure if any_failure else None
        ),
    }


def _parquet_assignment_mode(path: Path) -> str:
    metadata = pq.read_schema(path).metadata or {}
    raw_mode = metadata.get(b"assignment_mode")
    if raw_mode is None:
        raise ValueError(f"Parquet assignment_mode metadata is missing: {path}")
    mode = raw_mode.decode()
    if mode not in {"class_agnostic", "class_aware"}:
        raise ValueError(f"Unsupported Parquet assignment mode: {mode}")
    return mode


def validate_target_table_metadata(path: Path, assignment_mode: str) -> None:
    """Reject target tables not emitted under the declared primary matcher."""
    schema = pq.read_schema(path)
    metadata = dict(schema.metadata or {})
    expected = {
        b"producer": b"analysis/mechanism.py",
        b"assignment_mode": assignment_mode.encode(),
        b"qualification_score_threshold": b"0.25",
        b"qualification_distance_m": b"2.0",
        b"qualification_assignment": (
            b"global_one_to_one_max_cardinality_then_min_center_distance"
        ),
    }
    mismatches = {
        key.decode(): (
            metadata.get(key).decode(errors="replace")
            if metadata.get(key) is not None
            else None
        )
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Target table mechanism metadata is missing or incompatible: {mismatches}"
        )
    expected_types = {
        "model": pa.string(),
        "condition": pa.string(),
        "log": pa.string(),
        "scene": pa.string(),
        "sample_token": pa.string(),
        "annotation_token": pa.string(),
        "class": pa.string(),
        "clean_score": pa.float64(),
        "distance_m": pa.float64(),
        "size_m3": pa.float64(),
        "visibility": pa.int32(),
        "failure": pa.bool_(),
    }
    wrong_types = {
        column: str(schema.field(column).type) if column in schema.names else None
        for column, expected_type in expected_types.items()
        if column not in schema.names or schema.field(column).type != expected_type
    }
    if wrong_types:
        raise ValueError(
            f"Target table mechanism schema is incompatible: {wrong_types}"
        )


def validate_target_table_semantics(targets: pd.DataFrame) -> None:
    """Protect clean-side and GT covariates from condition-dependent drift."""
    required = {
        "model",
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
        "class",
        "clean_score",
        "distance_m",
        "size_m3",
        "visibility",
        "failure",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    identity = ["model", "condition", "annotation_token"]
    if targets.empty or targets[list(required)].isna().any().any():
        raise ValueError("Target table values must be complete")
    if targets.duplicated(identity).any():
        raise ValueError("Target table contains duplicate detector-condition targets")

    numeric = {
        column: pd.to_numeric(targets[column], errors="coerce").to_numpy(float)
        for column in ("clean_score", "distance_m", "size_m3", "visibility")
    }
    if any(not np.isfinite(values).all() for values in numeric.values()):
        raise ValueError("Target table numeric covariates must be finite")
    if np.any((numeric["clean_score"] < 0) | (numeric["clean_score"] > 1)):
        raise ValueError("Target Clean scores must lie in [0, 1]")
    if np.any(numeric["distance_m"] < 0) or np.any(numeric["size_m3"] <= 0):
        raise ValueError("Target distance and volume must be physically valid")
    if not np.equal(numeric["visibility"], np.floor(numeric["visibility"])).all():
        raise ValueError("Target visibility must be integer-valued")

    condition_count = targets["condition"].nunique()
    coverage = targets.groupby(
        ["model", "annotation_token"], observed=True
    )["condition"].nunique()
    if condition_count < 1 or not (coverage == condition_count).all():
        raise ValueError("Every detector-annotation target must cover every condition")
    clean_variants = targets.groupby(
        ["model", "annotation_token"], observed=True
    )["clean_score"].nunique(dropna=False)
    if (clean_variants != 1).any():
        raise ValueError(
            "A detector-annotation Clean score must be invariant across conditions"
        )

    physical_columns = [
        "log",
        "scene",
        "sample_token",
        "class",
        "distance_m",
        "size_m3",
        "visibility",
    ]
    physical_variants = targets.groupby(
        "annotation_token", observed=True
    )[physical_columns].nunique(dropna=False)
    if physical_variants.ne(1).any().any():
        raise ValueError(
            "Annotation identity and GT covariates must be invariant across "
            "detectors and conditions"
        )


def _decode_json_metadata(metadata: dict[bytes, bytes], key: bytes, path: Path):
    raw = metadata.get(key)
    if raw is None:
        raise ValueError(f"Parquet {key.decode()} metadata is missing: {path}")
    try:
        return json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Parquet {key.decode()} metadata is invalid JSON: {path}"
        ) from error


def load_target_sensitivity_table(
    path: Path,
    assignment_mode: str,
    target_conditions: set[str] | frozenset[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load the mechanism-produced sensitivity estimand without reclassification."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Mechanism-produced target sensitivity table is missing: {path}"
        )
    schema = pq.read_schema(path)
    metadata = dict(schema.metadata or {})
    producer = metadata.get(b"producer")
    if producer != b"analysis/mechanism.py":
        found = producer.decode(errors="replace") if producer is not None else None
        raise ValueError(
            "Target sensitivity must be produced by analysis/mechanism.py; "
            f"found {found!r}"
        )
    sensitivity_mode = _parquet_assignment_mode(path)
    if sensitivity_mode != assignment_mode:
        raise ValueError(
            "Target sensitivity assignment mode does not match the mechanism "
            f"artifact: {sensitivity_mode} != {assignment_mode}"
        )
    expected_config_order = [
        {"score_threshold": score, "match_distance_m": distance}
        for score, distance in SENSITIVITY_CONFIGS
    ]
    config_order = _decode_json_metadata(metadata, b"config_order", path)
    if config_order != expected_config_order:
        raise ValueError(
            "Target sensitivity config_order does not match the consumer grid"
        )
    condition_order = _decode_json_metadata(metadata, b"condition_order", path)
    if (
        not isinstance(condition_order, list)
        or not condition_order
        or len(condition_order) > 16
        or any(not isinstance(condition, str) for condition in condition_order)
        or len(condition_order) != len(set(condition_order))
    ):
        raise ValueError("Target sensitivity condition_order is invalid")
    conditions = tuple(condition_order)
    if set(conditions) != set(target_conditions):
        raise ValueError(
            "Target sensitivity condition_order does not match the primary target table"
        )

    identity = [
        "model",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
    ]
    mask_columns = [
        "condition_mask",
        "eligible_config_mask",
        *[f"failure_mask_{index}" for index in range(len(SENSITIVITY_CONFIGS))],
    ]
    required = identity + mask_columns
    missing = set(required).difference(schema.names)
    if missing:
        raise ValueError(
            f"Target sensitivity table is missing columns: {sorted(missing)}"
        )
    wrong_mask_types = {
        column: str(schema.field(column).type)
        for column in mask_columns
        if schema.field(column).type != pa.uint16()
    }
    if wrong_mask_types:
        raise ValueError(
            f"Target sensitivity masks must use Parquet uint16: {wrong_mask_types}"
        )
    table = pd.read_parquet(path, columns=required)
    if table.empty:
        raise ValueError("Target sensitivity table is empty")
    if table[identity].isna().any().any():
        raise ValueError("Target sensitivity identities must be complete")
    if table.duplicated(["model", "annotation_token"]).any():
        raise ValueError(
            "Target sensitivity must contain one row per detector-annotation"
        )
    if set(table["model"].astype(str)) != set(MODEL_ORDER):
        raise ValueError("Target sensitivity detector set is incomplete or unexpected")
    detector_counts = table.groupby("annotation_token", observed=True)[
        "model"
    ].nunique()
    if not (detector_counts == len(MODEL_ORDER)).all():
        raise ValueError(
            "Every sensitivity annotation must occur once for every detector"
        )
    physical_identity = ["log", "scene", "sample_token"]
    inconsistent_identity = (
        table.groupby("annotation_token", observed=True)[physical_identity]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent_identity.any():
        raise ValueError("Sensitivity annotation identity disagrees across detectors")

    masks: dict[str, np.ndarray] = {}
    for column in mask_columns:
        numeric = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        if (
            not np.isfinite(numeric).all()
            or (numeric < 0).any()
            or (numeric > np.iinfo(np.uint16).max).any()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            raise ValueError(f"Target sensitivity {column} is not a uint16 mask")
        masks[column] = numeric.astype(np.uint16)
        table[column] = masks[column]

    complete_condition_mask = (1 << len(conditions)) - 1
    if not np.all(masks["condition_mask"] == complete_condition_mask):
        raise ValueError(
            "Every target sensitivity row must cover every declared condition"
        )
    config_bits = (1 << len(SENSITIVITY_CONFIGS)) - 1
    unknown_config_bits = np.uint16(np.iinfo(np.uint16).max ^ config_bits)
    if np.any(masks["eligible_config_mask"] & unknown_config_bits):
        raise ValueError("Target sensitivity eligibility has unknown config bits")
    unknown_condition_bits = np.uint16(
        np.iinfo(np.uint16).max ^ complete_condition_mask
    )
    for index in range(len(SENSITIVITY_CONFIGS)):
        failure = masks[f"failure_mask_{index}"]
        if np.any(failure & unknown_condition_bits):
            raise ValueError(
                f"Target sensitivity failure_mask_{index} has unknown condition bits"
            )
        eligible = (masks["eligible_config_mask"] & np.uint16(1 << index)) != 0
        if np.any(failure[~eligible]):
            raise ValueError(
                f"Target sensitivity failure_mask_{index} marks ineligible targets"
            )
    return table, conditions


def expand_target_sensitivity(
    table: pd.DataFrame, config_index: int, conditions: tuple[str, ...]
) -> pd.DataFrame:
    """Expand one packed configuration into the seven-column target table."""
    eligibility_bit = np.uint16(1 << config_index)
    eligible = table[
        (table["eligible_config_mask"].to_numpy(dtype=np.uint16) & eligibility_bit) != 0
    ]
    repeats = len(conditions)
    failure_masks = eligible[f"failure_mask_{config_index}"].to_numpy(dtype=np.uint16)
    expanded_masks = np.repeat(failure_masks, repeats)
    condition_bits = np.tile(
        np.asarray([1 << index for index in range(repeats)], dtype=np.uint16),
        len(eligible),
    )
    return pd.DataFrame(
        {
            column: np.repeat(eligible[column].to_numpy(), repeats)
            for column in (
                "model",
                "log",
                "scene",
                "sample_token",
                "annotation_token",
            )
        }
        | {
            "condition": np.tile(np.asarray(conditions), len(eligible)),
            "failure": (expanded_masks & condition_bits) != 0,
        }
    )


def validate_primary_target_alignment(
    target_table: pd.DataFrame,
    packed_sensitivity: pd.DataFrame,
    conditions: tuple[str, ...],
) -> None:
    """Require the packed primary mask to reproduce every target outcome."""
    identity = [
        "model",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
        "condition",
    ]
    required = {*identity, "failure"}
    missing = required.difference(target_table.columns)
    if missing:
        raise ValueError(
            f"Primary target table is missing alignment columns: {sorted(missing)}"
        )
    target = target_table[[*identity, "failure"]].copy()
    if (
        target[identity].isna().any().any()
        or target["failure"].isna().any()
        or target.duplicated(identity).any()
    ):
        raise ValueError("Primary target table has invalid or duplicate identities")

    primary_index = SENSITIVITY_CONFIGS.index((0.25, 2.0))
    expanded = expand_target_sensitivity(
        packed_sensitivity,
        primary_index,
        conditions,
    )
    if (
        expanded[identity].isna().any().any()
        or expanded["failure"].isna().any()
        or expanded.duplicated(identity).any()
    ):
        raise ValueError(
            "Packed sensitivity primary configuration has invalid or duplicate "
            "identities"
        )
    aligned = target.merge(
        expanded,
        on=identity,
        how="outer",
        suffixes=("_target", "_sensitivity"),
        indicator=True,
        validate="one_to_one",
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError(
            "Packed sensitivity primary cohort does not match the primary target table"
        )
    if not (
        aligned["failure_target"].astype(bool)
        == aligned["failure_sensitivity"].astype(bool)
    ).all():
        raise ValueError(
            "Packed sensitivity primary failure identities do not match the "
            "primary target table"
        )


def _common_failure_frame(target_table: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
    ]
    wide = target_table.pivot(index=identity, columns="model", values="failure")
    common = wide.dropna(subset=list(MODEL_ORDER)).reset_index()
    for model in MODEL_ORDER:
        common[model] = common[model].astype(bool)
    return common


def _compact_null_summary(block: dict) -> dict:
    return {
        "specific_fraction": block["specific_fraction_among_any_failure"],
        "shared_fraction": block["shared_fraction_among_any_failure"],
    }


def summarize_failure_estimand_sensitivity(
    packed: pd.DataFrame,
    conditions: tuple[str, ...],
    assignment_mode: str,
    replicates: int = SENSITIVITY_REPLICATES,
) -> dict:
    """Recompute transfer and sharing across the full score/radius grid."""
    if assignment_mode not in {"class_agnostic", "class_aware"}:
        raise ValueError(f"Unsupported assignment mode: {assignment_mode}")
    scene_to_log = {
        str(scene): str(log)
        for scene, log in packed[["scene", "log"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    scenes = sorted(scene_to_log)
    rows = []
    for config_index, (score, distance) in enumerate(SENSITIVITY_CONFIGS):
        target = expand_target_sensitivity(packed, config_index, conditions)
        grouped = (
            target.groupby(["model", "condition", "scene"], observed=True, sort=False)[
                "failure"
            ]
            .agg(["size", "sum"])
            .rename(columns={"size": "clean_tp_count", "sum": "lost_count"})
            .sort_index()
        )
        fraction_table = {}
        count_table = {}
        opportunity_table = {}
        for model in MODEL_ORDER:
            for condition in conditions:
                local = grouped.loc[(model, condition)]
                counts = {
                    scene: float(local.loc[scene, "lost_count"])
                    if scene in local.index
                    else 0.0
                    for scene in scenes
                }
                opportunities = {
                    scene: float(local.loc[scene, "clean_tp_count"])
                    if scene in local.index
                    else 0.0
                    for scene in scenes
                }
                count_table[(model, condition)] = counts
                opportunity_table[(model, condition)] = opportunities
                fraction_table[(model, condition)] = {
                    scene: (
                        counts[scene] / opportunities[scene]
                        if opportunities[scene] > 0
                        else 0.0
                    )
                    for scene in scenes
                }

        def transfer_summary(table: dict, opportunity: dict | None = None) -> dict:
            cells = compute_cells(
                table,
                scene_to_log,
                0.20,
                opportunity_table=opportunity,
            )
            fields = (
                "raw_overlap",
                "log_stratified_adjusted_overlap",
                "raw_RiskCoverage",
                "log_stratified_adjusted_RiskCoverage",
                "log_opportunity_adjusted_RiskCoverage",
            )
            return {
                field: (
                    median(
                        float(cell[field]) for cell in cells if cell[field] is not None
                    )
                    if any(cell[field] is not None for cell in cells)
                    else None
                )
                for field in fields
            }

        common = _common_failure_frame(target)
        observed = _failure_composition(common)
        strata = [
            {
                "condition": str(condition),
                "log": str(log),
                "n": int(len(local)),
                "failures": {model: int(local[model].sum()) for model in MODEL_ORDER},
            }
            for (condition, log), local in common.groupby(
                ["condition", "log"], observed=True, sort=True
            )
        ]
        fixed = marginal_failure_null(
            strata,
            replicates=replicates,
            seed=IDENTIFICATION_SEED + config_index + 1,
        )
        frequency = target_frequency_null(
            common,
            replicates=replicates,
            seed=TARGET_FREQUENCY_SEED + config_index + 1,
        )
        fixed.pop("_specific_draws")
        fixed.pop("_shared_draws")
        frequency.pop("_specific_draws")
        frequency.pop("_shared_draws")
        rows.append(
            {
                "assignment_mode": assignment_mode,
                "score_threshold": score,
                "match_distance": distance,
                "detector_annotation_targets": int(len(target) / len(conditions)),
                "common_unique_annotations": int(common["annotation_token"].nunique()),
                "lost_fraction_transfer": transfer_summary(fraction_table),
                "lost_count_transfer": transfer_summary(count_table, opportunity_table),
                "observed": observed,
                "fixed_rate_null": _compact_null_summary(fixed),
                "target_frequency_null": _compact_null_summary(frequency),
            }
        )
        del (
            target,
            grouped,
            fraction_table,
            count_table,
            opportunity_table,
            common,
            strata,
            fixed,
            frequency,
        )
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass
    return {
        "assignment_mode": assignment_mode,
        "configuration_count": len(rows),
        "replicates_per_configuration": replicates,
        "configurations": rows,
        "boundary": (
            f"{assignment_mode.replace('_', '-').title()} assignment is fixed; "
            "the complete 3x3 score/radius "
            "grid changes clean eligibility and the binary lost-TP estimand."
        ),
    }


def _merge_common_gt_metadata(
    target_table: pd.DataFrame,
    common: pd.DataFrame,
    identity: list[str],
) -> pd.DataFrame:
    """Attach GT context only after validating cross-detector identity semantics."""
    metadata_columns = ["class", "visibility", "distance_m", "size_m3"]
    missing = set(metadata_columns).difference(target_table.columns)
    if missing:
        raise ValueError(f"Target table is missing GT metadata: {sorted(missing)}")
    grouped = target_table.groupby(identity, observed=True, sort=False)
    inconsistent = {
        column: int((grouped[column].nunique(dropna=False) > 1).sum())
        for column in metadata_columns
    }
    inconsistent = {key: value for key, value in inconsistent.items() if value}
    if inconsistent:
        raise ValueError(
            "GT metadata disagree across detectors for the same identity: "
            f"{inconsistent}"
        )
    metadata = target_table.drop_duplicates(identity)[identity + metadata_columns]
    if metadata[metadata_columns].isna().any().any():
        raise ValueError("Context-stratified null requires complete GT metadata")
    numeric = metadata[["visibility", "distance_m", "size_m3"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Context-stratified GT metadata must be finite")
    if (metadata["distance_m"].to_numpy(float) < 0).any() or (
        metadata["size_m3"].to_numpy(float) <= 0
    ).any():
        raise ValueError("Distance must be nonnegative and size must be positive")
    return common.merge(metadata, on=identity, how="left", validate="one_to_one")


def _tertile_definition(values: np.ndarray, labels: tuple[str, str, str]) -> dict:
    lower, upper = np.quantile(np.asarray(values, dtype=float), [1 / 3, 2 / 3])
    return {
        "cutpoints": [float(lower), float(upper)],
        "labels": list(labels),
        "rule": (
            f"{labels[0]}: x <= first; {labels[1]}: first < x <= second; "
            f"{labels[2]}: x > second"
        ),
    }


def _apply_tertiles(values: pd.Series, definition: dict) -> pd.Categorical:
    lower, upper = definition["cutpoints"]
    bins = np.searchsorted(
        np.asarray([lower, upper], dtype=float),
        values.to_numpy(float),
        side="left",
    )
    return pd.Categorical.from_codes(
        bins,
        categories=definition["labels"],
        ordered=True,
    )


def context_stratified_failure_reference(
    common: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """Build a stronger descriptive fixed-rate reference over observed GT context."""
    physical_identity = ["log", "scene", "sample_token", "annotation_token"]
    physical = common.drop_duplicates(physical_identity)
    distance_definition = _tertile_definition(
        physical["distance_m"].to_numpy(float),
        ("near", "middle", "far"),
    )
    size_definition = _tertile_definition(
        physical["size_m3"].to_numpy(float),
        ("small", "middle", "large"),
    )
    enriched = common.copy()
    enriched["distance_tertile"] = _apply_tertiles(
        enriched["distance_m"], distance_definition
    )
    enriched["size_tertile"] = _apply_tertiles(enriched["size_m3"], size_definition)
    group_columns = [
        "condition",
        "log",
        "class",
        "visibility",
        "distance_tertile",
        "size_tertile",
    ]
    strata = []
    for key, rows in enriched.groupby(
        group_columns,
        observed=True,
        sort=True,
    ):
        strata.append(
            {
                **{
                    column: (int(value) if column == "visibility" else str(value))
                    for column, value in zip(group_columns, key)
                },
                "n": int(len(rows)),
                "failures": {model: int(rows[model].sum()) for model in MODEL_ORDER},
            }
        )
    singleton = sum(row["n"] == 1 for row in strata)
    detector_degenerate = sum(
        any(row["failures"][model] in {0, row["n"]} for model in MODEL_ORDER)
        for row in strata
    )
    fully_deterministic = sum(
        all(row["failures"][model] in {0, row["n"]} for model in MODEL_ORDER)
        for row in strata
    )
    count = len(strata)
    diagnostics = {
        "stratification": group_columns,
        "strata": count,
        "singleton_strata": singleton,
        "singleton_fraction": singleton / count if count else None,
        "any_detector_degenerate_margin_strata": detector_degenerate,
        "any_detector_degenerate_margin_fraction": (
            detector_degenerate / count if count else None
        ),
        "fully_deterministic_margin_strata": fully_deterministic,
        "fully_deterministic_margin_fraction": (
            fully_deterministic / count if count else None
        ),
        "bin_definitions": {
            "distance_m_tertiles": distance_definition,
            "size_m3_tertiles": size_definition,
            "quantile_population": (
                "unique common-clean-TP physical GT identities, pooled across logs"
            ),
        },
        "interpretation": (
            "Stronger descriptive fixed-rate reference, not causal adjustment. "
            "Fine strata can be singleton or margin-degenerate."
        ),
    }
    return strata, diagnostics


def summarize_identification(
    target_table: pd.DataFrame,
    cohort_comparison: list[dict],
    assignment_mode: str,
    replicates: int = IDENTIFICATION_REPLICATES,
    seed: int = IDENTIFICATION_SEED,
    frequency_seed: int = TARGET_FREQUENCY_SEED,
    diagnostic_chains: int = TARGET_FREQUENCY_DIAGNOSTIC_CHAINS,
    diagnostic_replicates: int | None = None,
) -> dict:
    """Audit detector specificity under common eligibility and matched rates."""
    if assignment_mode not in {"class_agnostic", "class_aware"}:
        raise ValueError(f"Unsupported assignment mode: {assignment_mode}")
    required = {
        "model",
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
        "failure",
    }
    missing = required - set(target_table.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    models = tuple(sorted(target_table["model"].unique()))
    if set(models) != set(MODEL_ORDER):
        raise ValueError("Target table does not contain the expected detectors")
    conditions = sorted(target_table["condition"].unique())
    signed_axis_condition_blocks(conditions)
    identity = [
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
    ]
    if target_table.duplicated(["model", *identity]).any():
        raise ValueError("Target table has duplicate detector-target-condition rows")

    coverage = target_table.groupby(["model", "annotation_token"], observed=True)[
        "condition"
    ].nunique()
    if not bool((coverage == len(conditions)).all()):
        raise ValueError("Every detector-annotation target must cover all conditions")

    wide = target_table.pivot(index=identity, columns="model", values="failure")
    common = wide.dropna(subset=list(MODEL_ORDER)).reset_index()
    if common.empty:
        raise ValueError("No common clean-TP population is available")
    for model in MODEL_ORDER:
        common[model] = common[model].astype(bool)
    common = _merge_common_gt_metadata(target_table, common, identity)

    detector_specific = []
    for (model, condition), rows in target_table.groupby(
        ["model", "condition"], observed=True, sort=True
    ):
        detector_specific.append(
            {
                "model": str(model),
                "condition": str(condition),
                "n": int(len(rows)),
                "failure_rate": float(rows["failure"].mean()),
            }
        )
    common_rates = []
    for condition, rows in common.groupby("condition", observed=True, sort=True):
        for model in MODEL_ORDER:
            common_rates.append(
                {
                    "model": model,
                    "condition": str(condition),
                    "n": int(len(rows)),
                    "failure_rate": float(rows[model].mean()),
                }
            )

    observed = _failure_composition(common)
    pairwise = {}
    for index, left in enumerate(MODEL_ORDER):
        for right in MODEL_ORDER[index + 1 :]:
            intersection = int((common[left] & common[right]).sum())
            union = int((common[left] | common[right]).sum())
            pairwise[f"{left}__{right}"] = {
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else None,
            }
    observed["pairwise_jaccard"] = pairwise

    strata = []
    for (condition, log), rows in common.groupby(
        ["condition", "log"], observed=True, sort=True
    ):
        strata.append(
            {
                "condition": str(condition),
                "log": str(log),
                "n": int(len(rows)),
                "failures": {model: int(rows[model].sum()) for model in MODEL_ORDER},
            }
        )
    null = marginal_failure_null(strata, replicates=replicates, seed=seed)
    specific_draws = null.pop("_specific_draws")
    shared_draws = null.pop("_shared_draws")
    observed_specific = float(observed["specific_fraction_among_any_failure"])
    observed_shared = float(observed["shared_fraction_among_any_failure"])
    null["tests"] = {
        "specificity_greater": {
            "observed": observed_specific,
            "one_sided_p": float(
                (1 + np.count_nonzero(specific_draws >= observed_specific))
                / (replicates + 1)
            ),
        },
        "sharing_greater": {
            "observed": observed_shared,
            "one_sided_p": float(
                (1 + np.count_nonzero(shared_draws >= observed_shared))
                / (replicates + 1)
            ),
        },
    }
    context_strata, context_diagnostics = context_stratified_failure_reference(common)
    context_null = marginal_failure_null(
        context_strata,
        replicates=replicates,
        seed=seed + 1,
    )
    context_null.pop("_specific_draws")
    context_null.pop("_shared_draws")
    context_null["unit"] = (
        "annotation-condition within detector-condition-log-GT-class-visibility-"
        "distance-tertile-size-tertile"
    )
    context_null["context_diagnostics"] = context_diagnostics
    context_null["reference_boundary"] = (
        "This class-, visibility-, distance-, size-, condition-, and log-stratified "
        "fixed-rate reference is stronger descriptively than the detector-condition-"
        "log null. It does not identify causal effects or preserve physical-instance "
        "trajectory dependence."
    )

    if diagnostic_replicates is None:
        diagnostic_replicates = min(
            TARGET_FREQUENCY_DIAGNOSTIC_REPLICATES,
            replicates,
        )
    frequency = target_frequency_null(
        common, replicates=replicates, seed=frequency_seed
    )
    frequency_specific_draws = frequency.pop("_specific_draws")
    frequency_shared_draws = frequency.pop("_shared_draws")
    frequency["retained_state_diagnostic"] = {
        "observed_specific_fraction": observed_specific,
        "specific_states_at_or_above_observed": int(
            np.count_nonzero(frequency_specific_draws >= observed_specific)
        ),
        "observed_shared_fraction": observed_shared,
        "shared_states_at_or_above_observed": int(
            np.count_nonzero(frequency_shared_draws >= observed_shared)
        ),
        "retained_states": replicates,
    }
    frequency["multi_chain_diagnostics"] = multi_chain_target_frequency_diagnostics(
        common,
        replicates_per_chain=diagnostic_replicates,
        chains=diagnostic_chains,
        seed=frequency_seed + 10_000,
    )
    rotation = tuple(
        condition for condition in conditions if condition.startswith("D_R")
    )
    translation = tuple(
        condition for condition in conditions if condition.startswith("D_T")
    )
    family_blocks = tuple(
        block for block in (rotation, translation) if block
    )
    family_frequency = target_frequency_null(
        common,
        replicates=replicates,
        seed=frequency_seed + 1,
        condition_blocks=family_blocks,
    )
    family_frequency.pop("_specific_draws")
    family_shared_draws = family_frequency.pop("_shared_draws")
    family_frequency["retained_state_diagnostic"] = {
        "observed_shared_fraction": observed_shared,
        "shared_states_at_or_above_observed": int(
            np.count_nonzero(family_shared_draws >= observed_shared)
        ),
        "retained_states": replicates,
    }
    family_frequency["multi_chain_diagnostics"] = (
        multi_chain_target_frequency_diagnostics(
            common,
            replicates_per_chain=diagnostic_replicates,
            chains=diagnostic_chains,
            seed=frequency_seed + 20_000,
            condition_blocks=family_blocks,
        )
    )

    axis_blocks = signed_axis_condition_blocks(conditions)
    axis_frequency = target_frequency_null(
        common,
        replicates=replicates,
        seed=frequency_seed + 2,
        condition_blocks=axis_blocks,
    )
    axis_frequency.pop("_specific_draws")
    axis_shared_draws = axis_frequency.pop("_shared_draws")
    axis_frequency["retained_state_diagnostic"] = {
        "observed_shared_fraction": observed_shared,
        "shared_states_at_or_above_observed": int(
            np.count_nonzero(axis_shared_draws >= observed_shared)
        ),
        "retained_states": replicates,
    }
    axis_frequency["multi_chain_diagnostics"] = (
        multi_chain_target_frequency_diagnostics(
            common,
            replicates_per_chain=diagnostic_replicates,
            chains=diagnostic_chains,
            seed=frequency_seed + 30_000,
            condition_blocks=axis_blocks,
        )
    )

    lolo = {}
    for held_log in sorted(common["log"].unique()):
        subset = common[common["log"] != held_log]
        observed_lolo = _failure_composition(subset)
        lolo[str(held_log)] = {
            "observed": observed_lolo,
            "fixed_margin_null": null["leave_one_log_out"][str(held_log)],
            "context_stratified_fixed_rate_reference": context_null[
                "leave_one_log_out"
            ][str(held_log)],
            "target_frequency_null": frequency["leave_one_log_out"][str(held_log)],
            "target_family_frequency_null": family_frequency["leave_one_log_out"][
                str(held_log)
            ],
            "target_axis_frequency_null": axis_frequency["leave_one_log_out"][
                str(held_log)
            ],
        }

    return {
        "assignment_mode": assignment_mode,
        "source_rows": int(len(target_table)),
        "complete_detector_annotation_targets": int(len(coverage)),
        "conditions_per_target": len(conditions),
        "common_clean_tp_annotation_condition_rows": int(len(common)),
        "common_clean_tp_unique_annotations": int(common["annotation_token"].nunique()),
        "detector_specific_rates": detector_specific,
        "common_clean_tp_rates": common_rates,
        "all_gt_continuous_utilities": cohort_comparison,
        "observed_common_clean_tp_overlap": observed,
        "marginal_failure_rate_null": null,
        "context_stratified_fixed_rate_reference": context_null,
        "target_frequency_null": frequency,
        "target_family_frequency_null": family_frequency,
        "target_axis_frequency_null": axis_frequency,
        "leave_one_log_out": lolo,
        "claim_boundary": (
            "Common clean-TP results condition on targets detected cleanly by all "
            "three detectors. The fixed-margin null controls detector-condition-log "
            "rates; a stronger descriptive reference additionally stratifies by GT "
            "class, visibility, distance tertile, and size tertile. The constrained-"
            "switch null additionally fixes every annotation's failure frequency "
            "across conditions; blocked versions fix it separately by perturbation "
            "family and by each signed-axis pair. Multi-chain diagnostics assess "
            "reference-chain mixing. Annotation-condition membership defines "
            "the resampling unit; physical-instance trajectories vary across "
            "reference draws."
        ),
    }


def fold_clean_score_vulnerability(targets: pd.DataFrame, held_log: str) -> np.ndarray:
    """Calibrate the inverse Clean-score percentile within one held-log fold.

    ``targets`` is already restricted to common-clean-TP annotation identities.
    For each detector, one empirical CDF is fitted on this GT-conditioned
    population in all outer-training logs and applied unchanged to training and
    test rows. No held-log score or perturbed failure enters the fitted CDF; a
    test row's Clean score is only the CDF input.
    """
    required = {"model", "log", "annotation_token", "clean_score"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Missing Clean-score columns: {sorted(missing)}")
    if held_log not in set(targets["log"]):
        raise ValueError(f"Unknown held log: {held_log}")
    clean_scores = pd.to_numeric(targets["clean_score"], errors="coerce").to_numpy(
        float
    )
    if not np.isfinite(clean_scores).all():
        raise ValueError("Clean-score calibration requires finite scores")
    clean_score_variants = targets.groupby(
        ["model", "log", "annotation_token"], observed=True
    )["clean_score"].nunique(dropna=False)
    if (clean_score_variants != 1).any():
        raise ValueError(
            "A detector-annotation Clean score must be invariant across conditions"
        )

    reference = targets[
        ["model", "log", "annotation_token", "clean_score"]
    ].drop_duplicates(["model", "log", "annotation_token"])
    result = np.full(len(targets), np.nan, dtype=float)
    for model in sorted(targets["model"].unique()):
        model_reference = reference.loc[reference["model"].eq(model)]
        row_mask = targets["model"].eq(model).to_numpy()
        calibration = np.sort(
            model_reference.loc[
                model_reference["log"].ne(held_log),
                "clean_score",
            ].to_numpy(float)
        )
        if not len(calibration):
            raise ValueError(
                f"Model {model} has no Clean-score calibration rows "
                f"outside held log {held_log}"
            )
        row_positions = np.flatnonzero(row_mask)
        percentile = np.searchsorted(
            calibration,
            targets.loc[row_mask, "clean_score"].to_numpy(float),
            side="right",
        ) / len(calibration)
        result[row_positions] = 1.0 - percentile
    if np.isnan(result).any():
        raise RuntimeError("Fold-specific Clean-score calibration is incomplete")
    return result


def build_aligned_failure_transfer_table(targets: pd.DataFrame) -> pd.DataFrame:
    """Restrict to common Clean-TP identities and count other-detector failures."""
    required = {
        "model",
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
        "class",
        "clean_score",
        "distance_m",
        "size_m3",
        "visibility",
        "failure",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Missing failure-transfer columns: {sorted(missing)}")
    validate_target_table_semantics(targets)
    identity = [
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
    ]
    if targets.duplicated([*identity, "model"]).any():
        raise ValueError("Detector--condition--annotation rows must be unique")
    model_count = targets.groupby(identity, observed=True)["model"].transform("nunique")
    aligned = targets.loc[model_count == len(MODEL_ORDER)].copy()
    if set(aligned["model"].unique()) != set(MODEL_ORDER):
        raise ValueError("Aligned cohort does not contain all study detectors")
    failure_total = aligned.groupby(identity, observed=True)["failure"].transform("sum")
    aligned["source_failures"] = failure_total.astype(np.int8) - aligned[
        "failure"
    ].astype(np.int8)
    if not aligned["source_failures"].between(0, len(MODEL_ORDER) - 1).all():
        raise RuntimeError("Other-detector failure count is outside its support")
    failure_wide = (
        aligned.pivot(
            index=identity,
            columns="model",
            values="failure",
        )
        .rename(columns=lambda model: f"failure__{model}")
        .reset_index()
    )
    aligned = aligned.merge(
        failure_wide,
        on=identity,
        how="left",
        validate="many_to_one",
    )
    aligned["log_distance_m"] = np.log1p(aligned["distance_m"].astype(float))
    aligned["log_size_m3"] = np.log1p(aligned["size_m3"].clip(lower=0).astype(float))
    return aligned.reset_index(drop=True)


def fold_source_failure_rate(aligned: pd.DataFrame, held_detector: str) -> np.ndarray:
    """Return source evidence without reading the held detector's failures.

    Training targets come from the two non-held detectors and therefore have
    one available source. Test targets are the held detector and have two.
    A rate keeps the feature on the same [0, 1] scale in both populations.
    """
    available = [model for model in MODEL_ORDER if model != held_detector]
    total = (
        aligned[[f"failure__{model}" for model in available]]
        .astype(np.int8)
        .sum(axis=1)
        .to_numpy()
    )
    is_test_detector = aligned["model"].eq(held_detector).to_numpy()
    own_failure = aligned["failure"].astype(np.int8).to_numpy()
    numerator = total - np.where(is_test_detector, 0, own_failure)
    denominator = np.where(is_test_detector, len(available), len(available) - 1)
    result = numerator / denominator
    if np.any((result < 0) | (result > 1)):
        raise RuntimeError("Fold-specific source failure rate is invalid")
    return result.astype(float)


def _safe_binary_metric(metric, target: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(target)) < 2:
        return None
    return float(metric(target, score))


def _macro(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def _metric_direction(delta_auprc: float, delta_brier: float) -> str:
    if delta_auprc < 0 and delta_brier < 0:
        return "ranking_declined_brier_improved"
    if delta_auprc > 0 and delta_brier < 0:
        return "ranking_and_brier_improved"
    if delta_auprc < 0 and delta_brier > 0:
        return "ranking_and_brier_declined"
    return "mixed_or_unchanged"


def _transfer_estimator(numeric: tuple[str, ...]):
    columns = [*numeric, *TRANSFER_CATEGORICAL]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                ),
                list(numeric),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(TRANSFER_CATEGORICAL),
            ),
        ]
    )
    return (
        columns,
        make_pipeline(
            transformer,
            LogisticRegression(
                C=1,
                max_iter=1_000,
                random_state=TRANSFER_SEED,
            ),
        ),
    )


def single_source_failure_probabilities(
    estimator,
    test_rows: pd.DataFrame,
    columns: list[str],
    source_detectors: list[str],
) -> dict[str, np.ndarray]:
    """Apply a one-source training feature separately for every test source."""
    probabilities = {}
    for source in source_detectors:
        source_column = f"failure__{source}"
        if source_column not in test_rows:
            raise ValueError(f"Missing directed source column: {source_column}")
        source_rows = test_rows.copy()
        source_rows["source_failure_rate"] = source_rows[source_column].astype(float)
        probabilities[source] = estimator.predict_proba(source_rows[columns])[:, 1]
    return probabilities


def summarize_failure_evidence_transfer(targets: pd.DataFrame) -> dict:
    """Cross-fit source-failure evidence with detector and log held together."""
    aligned = build_aligned_failure_transfer_table(targets)
    models = sorted(aligned["model"].unique())
    logs = sorted(aligned["log"].unique())
    conditions = sorted(aligned["condition"].unique())
    target = aligned["failure"].astype(np.int8).to_numpy()
    group_size = aligned.groupby(["model", "log", "condition"], observed=True)[
        "failure"
    ].transform("size")
    weights = 1.0 / group_size.to_numpy(float)
    predictions = {
        name: np.full(len(aligned), np.nan, dtype=float)
        for name in (*TRANSFER_SPECIFICATIONS, MEAN_SINGLE_SOURCE_SPECIFICATION)
    }
    fold_rows = []
    condition_rows = []
    directed_fold_rows = []

    def record_predictions(
        held_model: str,
        held_log: str,
        specification: str,
        train: np.ndarray,
        test: np.ndarray,
        probability: np.ndarray,
    ) -> None:
        predictions[specification][test] = probability
        local_target = target[test]
        fold_rows.append(
            {
                "held_detector": held_model,
                "held_log": held_log,
                "specification": specification,
                "n": int(test.sum()),
                "positives": int(local_target.sum()),
                "prevalence": float(local_target.mean()),
                "AUPRC": _safe_binary_metric(
                    average_precision_score, local_target, probability
                ),
                "AUROC": _safe_binary_metric(roc_auc_score, local_target, probability),
                "brier": float(brier_score_loss(local_target, probability)),
                "train_excludes_held_detector": bool(
                    aligned.loc[train, "model"].ne(held_model).all()
                ),
                "train_excludes_held_log": bool(
                    aligned.loc[train, "log"].ne(held_log).all()
                ),
                "clean_score_calibration_excludes_held_log": True,
                "source_feature_excludes_held_detector_failure": True,
            }
        )
        test_frame = aligned.loc[test, ["condition"]].copy()
        test_frame["target"] = local_target
        test_frame["probability"] = probability
        for condition, group in test_frame.groupby(
            "condition", sort=True, observed=True
        ):
            local_y = group["target"].to_numpy()
            local_p = group["probability"].to_numpy()
            condition_rows.append(
                {
                    "held_detector": held_model,
                    "held_log": held_log,
                    "condition": condition,
                    "specification": specification,
                    "n": len(group),
                    "positives": int(local_y.sum()),
                    "negatives": int(len(local_y) - local_y.sum()),
                    "AUPRC": _safe_binary_metric(
                        average_precision_score, local_y, local_p
                    ),
                    "AUROC": _safe_binary_metric(roc_auc_score, local_y, local_p),
                    "brier": float(brier_score_loss(local_y, local_p)),
                }
            )

    for held_model in models:
        aligned["source_failure_rate"] = fold_source_failure_rate(aligned, held_model)
        source_detectors = [model for model in models if model != held_model]
        for held_log in logs:
            # Fit the Clean-score transformation inside the outer log fold.
            aligned["v_clean"] = fold_clean_score_vulnerability(aligned, held_log)
            test = (
                aligned["model"].eq(held_model) & aligned["log"].eq(held_log)
            ).to_numpy()
            train = (
                aligned["model"].ne(held_model) & aligned["log"].ne(held_log)
            ).to_numpy()
            if not test.any() or len(np.unique(target[train])) < 2:
                raise RuntimeError(
                    f"Unestimable transfer fold: {held_model}, {held_log}"
                )
            train_indices = np.flatnonzero(train)
            test_indices = np.flatnonzero(test)
            local_estimators = {}
            local_columns = {}
            local_probabilities = {}
            for name, numeric in TRANSFER_SPECIFICATIONS.items():
                columns, estimator = _transfer_estimator(numeric)
                estimator.fit(
                    aligned.iloc[train_indices][columns],
                    target[train],
                    logisticregression__sample_weight=weights[train],
                )
                probability = estimator.predict_proba(
                    aligned.iloc[test_indices][columns]
                )[:, 1]
                local_estimators[name] = estimator
                local_columns[name] = columns
                local_probabilities[name] = probability
                record_predictions(held_model, held_log, name, train, test, probability)

            directed = single_source_failure_probabilities(
                local_estimators["vulnerability_source_rate"],
                aligned.iloc[test_indices],
                local_columns["vulnerability_source_rate"],
                source_detectors,
            )
            vulnerability_probability = local_probabilities["vulnerability"]
            local_target = target[test]
            for source_detector, probability in directed.items():
                directed_fold_rows.append(
                    {
                        "source_detector": source_detector,
                        "target_detector": held_model,
                        "held_log": held_log,
                        "n": int(test.sum()),
                        "positives": int(local_target.sum()),
                        "AUPRC": _safe_binary_metric(
                            average_precision_score,
                            local_target,
                            probability,
                        ),
                        "brier": float(brier_score_loss(local_target, probability)),
                        "delta_AUPRC": (
                            _safe_binary_metric(
                                average_precision_score,
                                local_target,
                                probability,
                            )
                            - _safe_binary_metric(
                                average_precision_score,
                                local_target,
                                vulnerability_probability,
                            )
                        ),
                        "delta_brier": float(
                            brier_score_loss(local_target, probability)
                            - brier_score_loss(local_target, vulnerability_probability)
                        ),
                    }
                )
            mean_probability = np.mean(
                np.stack(list(directed.values()), axis=0), axis=0
            )
            record_predictions(
                held_model,
                held_log,
                MEAN_SINGLE_SOURCE_SPECIFICATION,
                train,
                test,
                mean_probability,
            )
    if any(np.isnan(values).any() for values in predictions.values()):
        raise RuntimeError("Double-held cross-fitting left predictions missing")

    summary = {}
    for name, probability in predictions.items():
        spec_folds = [row for row in fold_rows if row["specification"] == name]
        spec_conditions = [
            row for row in condition_rows if row["specification"] == name
        ]
        fold_auprc = [row["AUPRC"] for row in spec_folds if row["AUPRC"] is not None]
        condition_auprc = [
            row["AUPRC"] for row in spec_conditions if row["AUPRC"] is not None
        ]
        summary[name] = {
            "pooled_AUPRC": float(average_precision_score(target, probability)),
            "pooled_AUROC": float(roc_auc_score(target, probability)),
            "pooled_brier": float(brier_score_loss(target, probability)),
            "macro_detector_log_AUPRC": _macro(spec_folds, "AUPRC"),
            "macro_detector_log_AUROC": _macro(spec_folds, "AUROC"),
            "macro_detector_log_brier": _macro(spec_folds, "brier"),
            "balanced_detector_log_condition_AUPRC": _macro(spec_conditions, "AUPRC"),
            "balanced_detector_log_condition_AUROC": _macro(spec_conditions, "AUROC"),
            "balanced_detector_log_condition_brier": _macro(spec_conditions, "brier"),
            "balanced_detector_log_condition_AUPRC_estimable_cells": len(
                condition_auprc
            ),
            "balanced_detector_log_condition_total_cells": len(spec_conditions),
            "balanced_detector_log_condition_zero_positive_cells": sum(
                row["positives"] == 0 for row in spec_conditions
            ),
            "fold_AUPRC_range": (
                [min(fold_auprc), max(fold_auprc)] if fold_auprc else []
            ),
            "fold_brier_range": [
                min(row["brier"] for row in spec_folds),
                max(row["brier"] for row in spec_folds),
            ],
        }

    def paired_comparison(combined_name: str, aggregation: str) -> dict:
        paired = []
        for held_model in models:
            for held_log in logs:
                vulnerability = next(
                    row
                    for row in fold_rows
                    if row["held_detector"] == held_model
                    and row["held_log"] == held_log
                    and row["specification"] == "vulnerability"
                )
                combined = next(
                    row
                    for row in fold_rows
                    if row["held_detector"] == held_model
                    and row["held_log"] == held_log
                    and row["specification"] == combined_name
                )
                paired.append(
                    {
                        "held_detector": held_model,
                        "held_log": held_log,
                        "delta_AUPRC": (combined["AUPRC"] - vulnerability["AUPRC"]),
                        "delta_brier": (combined["brier"] - vulnerability["brier"]),
                    }
                )
        by_detector = []
        for model in models:
            rows = [row for row in paired if row["held_detector"] == model]
            vulnerability_rows = [
                row
                for row in fold_rows
                if row["held_detector"] == model
                and row["specification"] == "vulnerability"
            ]
            combined_rows = [
                row
                for row in fold_rows
                if row["held_detector"] == model
                and row["specification"] == combined_name
            ]
            mean_delta_auprc = float(np.mean([row["delta_AUPRC"] for row in rows]))
            mean_delta_brier = float(np.mean([row["delta_brier"] for row in rows]))
            lolo_delta_auprc = [
                float(
                    np.mean(
                        [row["delta_AUPRC"] for row in rows if row["held_log"] != log]
                    )
                )
                for log in logs
            ]
            lolo_delta_brier = [
                float(
                    np.mean(
                        [row["delta_brier"] for row in rows if row["held_log"] != log]
                    )
                )
                for log in logs
            ]
            by_detector.append(
                {
                    "held_detector": model,
                    "folds": len(rows),
                    "vulnerability_AUPRC": _macro(vulnerability_rows, "AUPRC"),
                    "vulnerability_source_AUPRC": _macro(combined_rows, "AUPRC"),
                    "mean_delta_AUPRC": mean_delta_auprc,
                    "leave_one_held_log_out_mean_delta_AUPRC_range": [
                        min(lolo_delta_auprc),
                        max(lolo_delta_auprc),
                    ],
                    "AUPRC_positive_folds": sum(row["delta_AUPRC"] > 0 for row in rows),
                    "vulnerability_brier": _macro(vulnerability_rows, "brier"),
                    "vulnerability_source_brier": _macro(combined_rows, "brier"),
                    "mean_delta_brier": mean_delta_brier,
                    "leave_one_held_log_out_mean_delta_brier_range": [
                        min(lolo_delta_brier),
                        max(lolo_delta_brier),
                    ],
                    "brier_improved_folds": sum(row["delta_brier"] < 0 for row in rows),
                    "metric_direction": _metric_direction(
                        mean_delta_auprc, mean_delta_brier
                    ),
                }
            )
        by_log = []
        for log in logs:
            rows = [row for row in paired if row["held_log"] == log]
            by_log.append(
                {
                    "held_log": log,
                    "mean_delta_AUPRC": float(
                        np.mean([row["delta_AUPRC"] for row in rows])
                    ),
                    "positive_detectors": sum(row["delta_AUPRC"] > 0 for row in rows),
                    "mean_delta_brier": float(
                        np.mean([row["delta_brier"] for row in rows])
                    ),
                    "brier_improved_detectors": sum(
                        row["delta_brier"] < 0 for row in rows
                    ),
                }
            )
        delta_auprc = [row["delta_AUPRC"] for row in paired]
        delta_brier = [row["delta_brier"] for row in paired]
        return {
            "source_aggregation": aggregation,
            "AUPRC_mean_gain": float(np.mean(delta_auprc)),
            "AUPRC_positive_folds": sum(value > 0 for value in delta_auprc),
            "AUPRC_gain_range": [min(delta_auprc), max(delta_auprc)],
            "brier_mean_change": float(np.mean(delta_brier)),
            "brier_improved_folds": sum(value < 0 for value in delta_brier),
            "brier_change_range": [min(delta_brier), max(delta_brier)],
            "by_held_detector": by_detector,
            "by_held_log": by_log,
            "leave_one_detector_out_mean_AUPRC_gain": {
                model: float(
                    np.mean(
                        [
                            row["delta_AUPRC"]
                            for row in paired
                            if row["held_detector"] != model
                        ]
                    )
                )
                for model in models
            },
            "leave_one_log_out_mean_AUPRC_gain_range": [
                min(
                    np.mean(
                        [row["delta_AUPRC"] for row in paired if row["held_log"] != log]
                    )
                    for log in logs
                ),
                max(
                    np.mean(
                        [row["delta_AUPRC"] for row in paired if row["held_log"] != log]
                    )
                    for log in logs
                ),
            ],
            "interpretation": (
                "Descriptive paired gains under shared training folds. They are "
                "not independent-fold confidence intervals, and no practical-"
                "minimum effect threshold was prespecified."
            ),
        }

    summary["vulnerability_source_rate_vs_vulnerability"] = paired_comparison(
        "vulnerability_source_rate",
        "mean failure rate across both available test sources",
    )
    summary["vulnerability_source_mean_single_vs_vulnerability"] = paired_comparison(
        MEAN_SINGLE_SOURCE_SPECIFICATION,
        "mean of two probabilities obtained by applying each source separately",
    )

    directed_pairs = []
    for target_detector in models:
        for source_detector in [model for model in models if model != target_detector]:
            rows = [
                row
                for row in directed_fold_rows
                if row["target_detector"] == target_detector
                and row["source_detector"] == source_detector
            ]
            vulnerability_rows = [
                row
                for row in fold_rows
                if row["held_detector"] == target_detector
                and row["specification"] == "vulnerability"
            ]
            directed_pairs.append(
                {
                    "source_detector": source_detector,
                    "target_detector": target_detector,
                    "folds": len(rows),
                    "vulnerability_AUPRC": _macro(vulnerability_rows, "AUPRC"),
                    "single_source_AUPRC": _macro(rows, "AUPRC"),
                    "mean_delta_AUPRC": float(
                        np.mean([row["delta_AUPRC"] for row in rows])
                    ),
                    "AUPRC_positive_folds": sum(row["delta_AUPRC"] > 0 for row in rows),
                    "vulnerability_brier": _macro(vulnerability_rows, "brier"),
                    "single_source_brier": _macro(rows, "brier"),
                    "mean_delta_brier": float(
                        np.mean([row["delta_brier"] for row in rows])
                    ),
                    "brier_improved_folds": sum(row["delta_brier"] < 0 for row in rows),
                }
            )
    summary["directed_source_to_target"] = {
        "training_feature": ("one available source detector per non-held target row"),
        "test_feature": "one named source detector, without source averaging",
        "pairs": directed_pairs,
    }

    condition_reference = [
        row for row in condition_rows if row["specification"] == "vulnerability"
    ]
    zero_positive_cells = [
        {
            "held_detector": row["held_detector"],
            "held_log": row["held_log"],
            "condition": row["condition"],
            "target_rows": row["n"],
            "positive_events": row["positives"],
        }
        for row in condition_reference
        if row["positives"] == 0
    ]
    all_positive_cells = [
        {
            "held_detector": row["held_detector"],
            "held_log": row["held_log"],
            "condition": row["condition"],
            "target_rows": row["n"],
            "positive_events": row["positives"],
        }
        for row in condition_reference
        if row["negatives"] == 0
    ]
    condition_metric_support = {
        "total_detector_log_condition_cells": len(condition_reference),
        "AUPRC_estimable_cells": sum(
            row["AUPRC"] is not None for row in condition_reference
        ),
        "total_target_rows": int(sum(row["n"] for row in condition_reference)),
        "total_positive_events": int(
            sum(row["positives"] for row in condition_reference)
        ),
        "estimable_target_rows": int(
            sum(row["n"] for row in condition_reference if row["AUPRC"] is not None)
        ),
        "estimable_positive_events": int(
            sum(
                row["positives"]
                for row in condition_reference
                if row["AUPRC"] is not None
            )
        ),
        "zero_positive_cells": len(zero_positive_cells),
        "all_positive_cells": len(all_positive_cells),
        "zero_positive_cell_identities": zero_positive_cells,
        "all_positive_cell_identities": all_positive_cells,
    }
    strata = []
    for source_failures, rows in aligned.groupby(
        "source_failures", sort=True, observed=True
    ):
        failures = int(rows["failure"].sum())
        strata.append(
            {
                "source_failures": int(source_failures),
                "n": len(rows),
                "target_failures": failures,
                "target_failure_rate": failures / len(rows),
            }
        )
    strata_by_detector = []
    for (model, source_failures), rows in aligned.groupby(
        ["model", "source_failures"], sort=True, observed=True
    ):
        failures = int(rows["failure"].sum())
        strata_by_detector.append(
            {
                "target_detector": model,
                "source_failures": int(source_failures),
                "n": len(rows),
                "target_failures": failures,
                "target_failure_rate": failures / len(rows),
            }
        )
    strata_by_condition = []
    for (condition, source_failures), rows in aligned.groupby(
        ["condition", "source_failures"], sort=True, observed=True
    ):
        failures = int(rows["failure"].sum())
        strata_by_condition.append(
            {
                "condition": condition,
                "source_failures": int(source_failures),
                "n": len(rows),
                "target_failures": failures,
                "target_failure_rate": failures / len(rows),
            }
        )
    unique_identities = len(aligned) // len(MODEL_ORDER)
    return {
        "design": {
            "split": ("leave one target detector and one acquisition log out together"),
            "target_population": (
                "annotation-condition identities detected cleanly by all three "
                "detectors"
            ),
            "inverse_clean_score_percentile_calibration": (
                "v_clean = 1 - empirical CDF(clean_score); within every outer "
                "detector-log fold, one detector-specific CDF is fitted on all "
                "common-clean-TP identities in the outer-training logs and applied "
                "unchanged to training and test rows; no held-log score or "
                "perturbed failure enters the fitted CDF, but the population uses "
                "GT-backed eligibility, and v_clean is not a physical margin"
            ),
            "source_feature": (
                "primary specification uses the failure rate among both "
                "available test sources; sensitivity applies the one-source "
                "training feature to each named test source separately and "
                "averages the resulting probabilities"
            ),
            "training_weight": (
                "inverse detector-log-condition target count, so acquisition "
                "groups and perturbation conditions have equal fitting mass"
            ),
            "models": models,
            "logs": logs,
            "conditions": conditions,
            "common_annotation_condition_identities": unique_identities,
            "aligned_target_rows": len(aligned),
            "folds": len(models) * len(logs),
            "retrospective": True,
        },
        "prevalence": float(target.mean()),
        "source_failure_strata": strata,
        "source_failure_strata_by_detector": strata_by_detector,
        "source_failure_strata_by_condition": strata_by_condition,
        "condition_metric_support": condition_metric_support,
        "summary": summary,
        "folds": fold_rows,
        "directed_source_folds": directed_fold_rows,
        "condition_metrics": condition_rows,
        "claim_boundary": (
            "The analysis estimates conditional prediction within the complete "
            "paired corpus. Common Clean-TP eligibility is a selected target "
            "population with target-detector Clean outputs available. "
            "Results report held-log predictive differences within this corpus."
        ),
    }


def render_summary(payload: dict) -> str:
    lines = [
        "# Failure-sharing references and target-risk analysis",
        "",
        f"Scope: {payload['evidence_scope']}; no unobserved panels are implied.",
        "",
        "## Decomposable scene-tail outcomes",
        "",
    ]
    for metric in METRICS:
        lines.append(f"### `{metric}`")
        lines.append("")
        lines.append(
            "| Model | Mean | CVaR90 | CVaR90/Mean [95% log-cluster CI] | "
            "LOLO min ratio | Log-equal ratio |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODEL_ORDER:
            row = payload["decomposable_tail"][metric][model]
            primary = row["scene_equal"]
            balanced = row["log_equal"]
            ratio_ci = row["log_cluster_bootstrap"]["ci95"]["CVaR90_over_mean"]
            lines.append(
                f"| {model} | {primary['mean']:.6f} | "
                f"{primary['CVaR90']:.6f} | "
                f"{primary['CVaR90_over_mean']:.3f} "
                f"[{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}] | "
                f"{row['lolo_min_CVaR90_over_mean']:.3f} | "
                f"{balanced['CVaR90_over_mean']:.3f} |"
            )
        lines.append("")
    lines.extend(["## Risk-set transfer using decomposable outcomes", ""])
    lines.append(
        "| Metric | Estimable | Median overlap | Median regret | "
        "LOLO positive | LOLO >= 0.25 | LOLO minimum |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        row = payload["decomposable_transfer"][metric]
        lines.append(
            f"| {metric} | {row['estimable_rows']} | "
            f"{row['overlap']['median']:.3f} | "
            f"{row['regret']['median']:.3f} | "
            f"{row['lolo_rows_positive_in_all_logs']}/"
            f"{row['lolo_rows_estimable_in_all_logs']} | "
            f"{row['lolo_rows_at_or_above_0_25_in_all_logs']}/"
            f"{row['lolo_rows_estimable_in_all_logs']} | "
            f"{row['lolo_min_regret']:.3f} |"
        )
    sensitivity = payload["failure_path_sensitivity"]
    identification = payload["eligibility_and_marginal_null"]
    observed = identification["observed_common_clean_tp_overlap"]
    null = identification["marginal_failure_rate_null"]
    frequency = identification["target_frequency_null"]
    family_frequency = identification["target_family_frequency_null"]
    axis_frequency = identification["target_axis_frequency_null"]
    context_frequency = identification["context_stratified_fixed_rate_reference"]
    evidence_transfer = payload["cross_fitted_failure_evidence_transfer"]
    evidence_comparison = evidence_transfer["summary"][
        "vulnerability_source_rate_vs_vulnerability"
    ]
    mean_single_comparison = evidence_transfer["summary"][
        "vulnerability_source_mean_single_vs_vulnerability"
    ]
    condition_support = evidence_transfer["condition_metric_support"]
    lines.extend(
        [
            "",
            "## Cross-fitted source-failure evidence",
            "",
            "- Split: leave one target detector and one acquisition log out "
            "together (27 folds).",
            f"- Common annotation-condition identities: "
            f"{evidence_transfer['design']['common_annotation_condition_identities']:,}; "
            f"aligned target rows: "
            f"{evidence_transfer['design']['aligned_target_rows']:,}.",
            "- Detector-log macro AUPRC, inverse Clean-score percentile "
            "only / percentile + dual-source failure rate: "
            f"{evidence_transfer['summary']['vulnerability']['macro_detector_log_AUPRC']:.3f} / "
            f"{evidence_transfer['summary']['vulnerability_source_rate']['macro_detector_log_AUPRC']:.3f}.",
            f"- Dual-source-rate mean paired AUPRC gain: "
            f"{evidence_comparison['AUPRC_mean_gain']:+.3f}; positive in "
            f"{evidence_comparison['AUPRC_positive_folds']}/27 folds.",
            f"- Dual-source-rate mean Brier change: "
            f"{evidence_comparison['brier_mean_change']:+.6f}; improved in "
            f"{evidence_comparison['brier_improved_folds']}/27 folds.",
            "- Mean-of-single-source-probabilities sensitivity: AUPRC gain "
            f"{mean_single_comparison['AUPRC_mean_gain']:+.3f} "
            f"({mean_single_comparison['AUPRC_positive_folds']}/27 positive); "
            "Brier change "
            f"{mean_single_comparison['brier_mean_change']:+.6f} "
            f"({mean_single_comparison['brier_improved_folds']}/27 improved).",
            "- Balanced condition AUPRC support: "
            f"{condition_support['AUPRC_estimable_cells']}/"
            f"{condition_support['total_detector_log_condition_cells']} cells; "
            f"{condition_support['zero_positive_cells']} zero-positive cells; "
            f"{condition_support['total_positive_events']:,} positive events "
            f"among {condition_support['total_target_rows']:,} target rows.",
            "- The paired fold counts are descriptive because folds share "
            "training observations.",
            "",
        ]
    )
    direction_labels = {
        "ranking_declined_brier_improved": (
            "ranking declined while Brier calibration improved"
        ),
        "ranking_and_brier_improved": "ranking and Brier calibration improved",
        "ranking_and_brier_declined": "ranking and Brier calibration declined",
        "mixed_or_unchanged": "metric directions were mixed or unchanged",
    }
    for row in evidence_comparison["by_held_detector"]:
        lines.append(
            f"- {row['held_detector']}: inverse Clean-score percentile "
            "AUPRC "
            f"{row['vulnerability_AUPRC']:.3f}, combined "
            f"{row['vulnerability_source_AUPRC']:.3f}, delta "
            f"{row['mean_delta_AUPRC']:+.3f} "
            f"({row['AUPRC_positive_folds']}/{row['folds']} positive); "
            f"Brier {row['vulnerability_brier']:.5f} / "
            f"{row['vulnerability_source_brier']:.5f}, delta "
            f"{row['mean_delta_brier']:+.6f} "
            f"({row['brier_improved_folds']}/{row['folds']} improved); "
            f"{direction_labels[row['metric_direction']]}."
        )
    lines.extend(["", "### Directed single-source sensitivity", ""])
    for row in evidence_transfer["summary"]["directed_source_to_target"]["pairs"]:
        lines.append(
            f"- {row['source_detector']} -> {row['target_detector']}: "
            f"AUPRC {row['vulnerability_AUPRC']:.3f} / "
            f"{row['single_source_AUPRC']:.3f}, delta "
            f"{row['mean_delta_AUPRC']:+.3f} "
            f"({row['AUPRC_positive_folds']}/{row['folds']} positive); "
            f"Brier delta {row['mean_delta_brier']:+.6f} "
            f"({row['brier_improved_folds']}/{row['folds']} improved)."
        )
    lines.extend(
        [
            "",
            "## Eligibility and fixed-margin identification",
            "",
            f"- Complete detector–annotation targets: "
            f"{identification['complete_detector_annotation_targets']:,}; "
            f"condition rows: {identification['source_rows']:,}.",
            f"- Common clean-TP annotation–condition rows: "
            f"{identification['common_clean_tp_annotation_condition_rows']:,}.",
            "- Observed detector-specific fraction among common-eligible "
            f"failures: {observed['specific_fraction_among_any_failure']:.3f}.",
            "- Marginal-rate null median [95% interval]: "
            f"{null['specific_fraction_among_any_failure']['median']:.3f} "
            f"[{null['specific_fraction_among_any_failure']['ci95'][0]:.3f}, "
            f"{null['specific_fraction_among_any_failure']['ci95'][1]:.3f}].",
            "- One-sided test that specificity exceeds the matched-rate null: "
            f"p={null['tests']['specificity_greater']['one_sided_p']:.6f}.",
            "- One-sided test that cross-detector sharing exceeds the matched-rate "
            f"null: p={null['tests']['sharing_greater']['one_sided_p']:.6f}.",
            "- Target-frequency null shared-failure median [95% chain range]: "
            f"{frequency['shared_fraction_among_any_failure']['median']:.3f} "
            f"[{frequency['shared_fraction_among_any_failure']['ci95'][0]:.3f}, "
            f"{frequency['shared_fraction_among_any_failure']['ci95'][1]:.3f}]; "
            f"{frequency['retained_state_diagnostic']['shared_states_at_or_above_observed']}/"
            f"{frequency['retained_state_diagnostic']['retained_states']} retained "
            "states reached the observed statistic.",
            "- Family-blocked target-frequency null shared-failure median "
            f"[95% chain range]: "
            f"{family_frequency['shared_fraction_among_any_failure']['median']:.3f} "
            f"[{family_frequency['shared_fraction_among_any_failure']['ci95'][0]:.3f}, "
            f"{family_frequency['shared_fraction_among_any_failure']['ci95'][1]:.3f}].",
            "- Axis-pair-blocked target-frequency null shared-failure median "
            f"[95% chain range]: "
            f"{axis_frequency['shared_fraction_among_any_failure']['median']:.3f} "
            f"[{axis_frequency['shared_fraction_among_any_failure']['ci95'][0]:.3f}, "
            f"{axis_frequency['shared_fraction_among_any_failure']['ci95'][1]:.3f}].",
            "- Context-stratified fixed-rate reference shared-failure median "
            f"[95% interval]: "
            f"{context_frequency['shared_fraction_among_any_failure']['median']:.3f} "
            f"[{context_frequency['shared_fraction_among_any_failure']['ci95'][0]:.3f}, "
            f"{context_frequency['shared_fraction_among_any_failure']['ci95'][1]:.3f}]; "
            f"{context_frequency['context_diagnostics']['strata']} strata, "
            f"{context_frequency['context_diagnostics']['singleton_fraction']:.1%} "
            "singleton.",
            "- Multi-chain split R-hat for shared-failure fraction "
            "(target/family/axis): "
            f"{frequency['multi_chain_diagnostics']['metrics']['shared_fraction']['split_r_hat']:.3f} / "
            f"{family_frequency['multi_chain_diagnostics']['metrics']['shared_fraction']['split_r_hat']:.3f} / "
            f"{axis_frequency['multi_chain_diagnostics']['metrics']['shared_fraction']['split_r_hat']:.3f}.",
            "",
            "## Failure-path definition sensitivity",
            "",
            f"- Configurations: {sensitivity['configuration_count']}.",
            "- Score-collapse share range: "
            f"{sensitivity['min_score_collapse_share']:.3f}–"
            f"{sensitivity['max_score_collapse_share']:.3f}.",
            "- Configurations where score collapse is the largest path in every "
            "model-condition cell: "
            f"{sensitivity['all_cells_dominant_configuration_count']}/"
            f"{sensitivity['configuration_count']}.",
            "",
            "## Analysis scope",
            "",
            "Upper-tail CVaR is at least the full mean by construction. The "
            "informative result is the estimated concentration magnitude and "
            "its stability, not the sign of the difference alone.",
            "",
            "Absolute lost count is a burden outcome that also depends on the "
            "number of clean true positives in a scene; L_status is the more "
            "direct normalized target-vulnerability outcome.",
            "",
            "These retrospective analyses compare reference-dependent "
            "failure patterns within the paired detector corpus.",
            "",
        ]
    )
    return "\n".join(lines)


def _tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _tex_number(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def render_supplement_tables(corrected: dict, strengthening: dict) -> str:
    """Render deterministic extended tables from the machine-readable results."""
    outcome_labels = {
        "NDS_like_loss": "NDS-like loss",
        "lost_clean_tp_fraction": "Lost fraction",
        "lost_clean_tp_count": "Lost count",
    }
    compact_outcome_labels = {
        "NDS_like_loss": "NDS",
        "lost_clean_tp_fraction": "Frac.",
        "lost_clean_tp_count": "Count",
    }
    compact_model_labels = {
        "bevfusion_mit": "B",
        "sparsefusion_r50": "S",
        "deepinteraction_base": "D",
    }
    paper_model_labels = {
        "bevfusion_mit": "BEVFusion",
        "sparsefusion_r50": "SparseFusion",
        "deepinteraction_base": "DeepInteraction",
    }

    def compact_condition(condition: str) -> str:
        _, axis, direction, _ = condition.split("_")
        sign = "+" if direction == "POSITIVE" else "-"
        return f"{axis}{sign}"

    transfer = strengthening["cross_fitted_failure_evidence_transfer"]
    transfer_summary = transfer["summary"]
    lines = [
        "% Generated by analysis/reference_models.py. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model definition & Pooled AUPRC & Macro AUPRC & Balanced AUPRC & Macro AUROC & "
        r"Macro Brier \\",
        r"\midrule",
    ]
    for key, label in (
        ("context", "Context"),
        ("source", "Context + source"),
        ("vulnerability", "Context + Clean score"),
        (
            "vulnerability_source_rate",
            "Context + Clean score + rate",
        ),
        (
            MEAN_SINGLE_SOURCE_SPECIFICATION,
            "Context + Clean score + mean single-source",
        ),
    ):
        row = transfer_summary[key]
        lines.append(
            f"{label} & {row['pooled_AUPRC']:.3f} & "
            f"{row['macro_detector_log_AUPRC']:.3f} & "
            f"{row['balanced_detector_log_condition_AUPRC']:.3f} & "
            f"{row['macro_detector_log_AUROC']:.3f} & "
            f"{row['macro_detector_log_brier']:.5f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Joint detector--log holdout performance. Macro averages "
            r"give each of 27 detector--log splits equal mass; balanced AUPRC "
            r"additionally gives every estimable condition within a split equal "
            f"mass ({transfer['condition_metric_support']['AUPRC_estimable_cells']}/"
            f"{transfer['condition_metric_support']['total_detector_log_condition_cells']} "
            r"combinations; the remainder have one outcome class).}",
            r"\label{tab:supp-predictive}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Held detector & Base AUPRC & Rate AUPRC & $\Delta$AUPRC & AUPRC+ & "
            r"$\Delta$Brier & Brier+ \\",
            r"\midrule",
        ]
    )
    for row in transfer_summary["vulnerability_source_rate_vs_vulnerability"][
        "by_held_detector"
    ]:
        lines.append(
            f"{paper_model_labels[row['held_detector']]} & "
            f"{row['vulnerability_AUPRC']:.3f} & "
            f"{row['vulnerability_source_AUPRC']:.3f} & "
            f"{row['mean_delta_AUPRC']:+.3f} & "
            f"{row['AUPRC_positive_folds']}/{row['folds']} & "
            f"{row['mean_delta_brier']:+.5f} & "
            f"{row['brier_improved_folds']}/{row['folds']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Target-detector heterogeneity under the same joint "
            r"holdout design. Rate AUPRC uses context, the inverse Clean-score "
            r"percentile, "
            r"and the mean failure rate across both test sources. Brier+ counts "
            r"logs with lower Brier score.}",
            r"\label{tab:supp-predictive-targets}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Held detector & Base AUPRC & Mean-single AUPRC & "
            r"$\Delta$AUPRC & AUPRC+ & "
            r"$\Delta$Brier & Brier+ \\",
            r"\midrule",
        ]
    )
    for row in transfer_summary["vulnerability_source_mean_single_vs_vulnerability"][
        "by_held_detector"
    ]:
        lines.append(
            f"{paper_model_labels[row['held_detector']]} & "
            f"{row['vulnerability_AUPRC']:.3f} & "
            f"{row['vulnerability_source_AUPRC']:.3f} & "
            f"{row['mean_delta_AUPRC']:+.3f} & "
            f"{row['AUPRC_positive_folds']}/{row['folds']} & "
            f"{row['mean_delta_brier']:+.5f} & "
            f"{row['brier_improved_folds']}/{row['folds']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Semantic-alignment sensitivity. The fitted one-source "
            r"feature is applied once per named test source, and the two "
            r"probabilities are averaged.}",
            r"\label{tab:supp-predictive-mean-single}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Source & Target & Single AUPRC & $\Delta$AUPRC & AUPRC+ & "
            r"$\Delta$Brier & Brier+ \\",
            r"\midrule",
        ]
    )
    for row in transfer_summary["directed_source_to_target"]["pairs"]:
        lines.append(
            f"{paper_model_labels[row['source_detector']]} & "
            f"{paper_model_labels[row['target_detector']]} & "
            f"{row['single_source_AUPRC']:.3f} & "
            f"{row['mean_delta_AUPRC']:+.3f} & "
            f"{row['AUPRC_positive_folds']}/{row['folds']} & "
            f"{row['mean_delta_brier']:+.5f} & "
            f"{row['brier_improved_folds']}/{row['folds']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Directed one-source sensitivity. Each row preserves the "
            r"source identity instead of pooling the two available sources. "
            r"Deltas are relative to context plus the inverse Clean-score "
            r"percentile.}",
            r"\label{tab:supp-predictive-directed}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{rrrr}",
            r"\toprule",
            r"Failing sources & Target rows & Target failures & Failure rate \\",
            r"\midrule",
        ]
    )
    for row in transfer["source_failure_strata"]:
        lines.append(
            f"{row['source_failures']} & {row['n']:,} & "
            f"{row['target_failures']:,} & "
            f"{100 * row['target_failure_rate']:.2f}\\% \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Observed target failure rate by the number of failing "
            r"source detectors in the common clean-TP population. Rows are "
            r"paired and clustered; these are descriptive conditional rates.}",
            r"\label{tab:supp-source-strata}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Outcome & Top fraction & Raw $O$ & $O_{\rm adj}$ & $C_{\rm all}$ & "
            r"$E_k$ & $E_{\rm adj}$ \\",
            r"\midrule",
        ]
    )
    for outcome, label in outcome_labels.items():
        for fraction in ("0.10", "0.20", "0.30"):
            summary = corrected["outcomes"][outcome]["fractions"][fraction]["summary"]
            lines.append(
                f"{label} & {fraction} & "
                f"{_tex_number(summary['raw_overlap']['median'])} & "
                f"{_tex_number(summary['log_stratified_adjusted_overlap']['median'])} & "
                f"{_tex_number(summary['absolute_target_risk_coverage']['median'])} & "
                f"{_tex_number(summary['raw_RiskCoverage']['median'])} & "
                f"{_tex_number(summary['log_stratified_adjusted_RiskCoverage']['median'])} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{Top-set cutoff sensitivity. Values are medians over "
            rf"{6 * strengthening['condition_count']} directed "
            r"detector--condition comparisons.}",
            r"\label{tab:supp-cutoffs}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Outcome & Raw $O$ & $O_{\rm adj}$ & $E_k$ & "
            r"$E_{\rm adj}$ & $E_{\rm opp}$ \\",
            r"\midrule",
        ]
    )
    for outcome, label in outcome_labels.items():
        lolo = list(corrected["outcomes"][outcome]["leave_one_log_out"].values())
        ranges = []
        for field in (
            "raw_overlap",
            "log_stratified_adjusted_overlap",
            "raw_RiskCoverage",
            "log_stratified_adjusted_RiskCoverage",
            "log_opportunity_adjusted_RiskCoverage",
        ):
            values = [
                row[field]["median"] for row in lolo if row[field]["median"] is not None
            ]
            ranges.append(
                "--" if not values else f"[{min(values):.3f}, {max(values):.3f}]"
            )
        lines.append(f"{label} & " + " & ".join(ranges) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Leave-one-log-out ranges of all headline transfer "
            r"medians. These are deletion sensitivities, not confidence "
            r"intervals.}",
            r"\label{tab:supp-lolo}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Detector & Own eligible $n$ & Own rate & Common $n$ & Common rate \\",
            r"\midrule",
        ]
    )
    identification = strengthening["eligibility_and_marginal_null"]
    for model in MODEL_ORDER:
        own = [
            row
            for row in identification["detector_specific_rates"]
            if row["model"] == model
        ]
        common = [
            row
            for row in identification["common_clean_tp_rates"]
            if row["model"] == model
        ]
        own_n = sum(row["n"] for row in own)
        common_n = sum(row["n"] for row in common)
        own_rate = sum(row["n"] * row["failure_rate"] for row in own) / own_n
        common_rate = sum(row["n"] * row["failure_rate"] for row in common) / common_n
        lines.append(
            f"{_tex(model)} & {own_n:,} & {own_rate:.4f} & "
            f"{common_n:,} & {common_rate:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{Eligibility summary aggregated over "
            rf"{strengthening['condition_count']} small-magnitude perturbations.}}",
            r"\label{tab:supp-eligibility}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Reference & Specific & Shared & Shared 95\% interval & "
            r"Shared LOLO range \\",
            r"\midrule",
        ]
    )
    observed = identification["observed_common_clean_tp_overlap"]
    observed_lolo = [
        row["observed"]["shared_fraction_among_any_failure"]
        for row in identification["leave_one_log_out"].values()
    ]
    lines.append(
        f"Observed & {observed['specific_fraction_among_any_failure']:.3f} & "
        f"{observed['shared_fraction_among_any_failure']:.3f} & -- & "
        f"[{min(observed_lolo):.3f}, {max(observed_lolo):.3f}] \\\\"
    )
    for key, label, lolo_key in (
        ("marginal_failure_rate_null", "Fixed-rate null", "fixed_margin_null"),
        (
            "context_stratified_fixed_rate_reference",
            "Context-stratified reference",
            "context_stratified_fixed_rate_reference",
        ),
        (
            "target_frequency_null",
            "Target-frequency null",
            "target_frequency_null",
        ),
        (
            "target_family_frequency_null",
            "Family-frequency null",
            "target_family_frequency_null",
        ),
        (
            "target_axis_frequency_null",
            "Axis-pair-frequency null",
            "target_axis_frequency_null",
        ),
    ):
        block = identification[key]
        shared = block["shared_fraction_among_any_failure"]
        lolo_values = (
            [
                row[lolo_key]["shared_fraction_among_any_failure"]["median"]
                for row in identification["leave_one_log_out"].values()
            ]
            if lolo_key is not None
            else []
        )
        lolo_text = (
            f"[{min(lolo_values):.3f}, {max(lolo_values):.3f}]" if lolo_values else "--"
        )
        lines.append(
            f"{label} & "
            f"{block['specific_fraction_among_any_failure']['median']:.3f} & "
            f"{shared['median']:.3f} & "
            f"[{shared['ci95'][0]:.3f}, {shared['ci95'][1]:.3f}] & "
            f"{lolo_text} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Common-eligibility failure composition. The "
            r"target-frequency null preserves annotation totals across all "
            r"conditions; the family-frequency null preserves them separately "
            r"within rotation and translation, and the axis-pair null does so "
            r"within each signed physical axis. The context-stratified reference "
            r"also fixes rates within GT class, visibility, distance-tertile, and "
            r"size-tertile groups; it is descriptive rather than causal.}",
            r"\label{tab:supp-identification}",
            r"\end{table}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"{\small",
            r"\begin{tabular}{@{}rrrrrrrr@{}}",
            r"\toprule",
            r"Score & Radius & Frac. raw $O$ & Frac. $O_{\rm adj}$ & "
            r"Count raw $O$ & Count $O_{\rm adj}$ & Observed shared & "
            r"Target-freq. null \\",
            r"\midrule",
        ]
    )
    for row in strengthening["failure_estimand_sensitivity"]["configurations"]:
        fraction = row["lost_fraction_transfer"]
        count = row["lost_count_transfer"]
        lines.append(
            f"{row['score_threshold']:.2f} & {row['match_distance']:.1f} & "
            f"{fraction['raw_overlap']:.3f} & "
            f"{fraction['log_stratified_adjusted_overlap']:.3f} & "
            f"{count['raw_overlap']:.3f} & "
            f"{count['log_stratified_adjusted_overlap']:.3f} & "
            f"{row['observed']['shared_fraction_among_any_failure']:.3f} & "
            f"{row['target_frequency_null']['shared_fraction']['median']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Central estimand sensitivity over the complete "
            r"score-threshold and center-radius grid under the fixed "
            f"{_tex(identification['assignment_mode'].replace('_', '-'))} "
            r"assignment. Each null entry is based on 2,000 "
            r"thinned switch-chain states; it is a robustness diagnostic, not "
            r"an exact test.}",
            r"\label{tab:supp-estimand-sensitivity}",
            r"\end{table*}",
            "",
            r"\begingroup",
            r"\small",
            r"\begin{longtable}{@{}llllrrrrrr@{}}",
            r"\toprule",
            r"Outcome & Condition & Source & Target & Raw $O$ & $O_{\rm adj}$ & "
            r"$C_{\rm all}$ & $E_k$ & $E_{\rm adj}$ & $E_{\rm opp}$ \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Outcome & Condition & Source & Target & Raw $O$ & $O_{\rm adj}$ & "
            r"$C_{\rm all}$ & $E_k$ & $E_{\rm adj}$ & $E_{\rm opp}$ \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for outcome, label in compact_outcome_labels.items():
        cells = corrected["outcomes"][outcome]["fractions"]["0.20"]["cells"]
        for row in cells:
            lines.append(
                f"{label} & {compact_condition(row['condition'])} & "
                f"{compact_model_labels[row['source']]} & "
                f"{compact_model_labels[row['target']]} & "
                f"{_tex_number(row['raw_overlap'])} & "
                f"{_tex_number(row['log_stratified_adjusted_overlap'])} & "
                f"{_tex_number(row['absolute_target_risk_coverage'])} & "
                f"{_tex_number(row['raw_RiskCoverage'])} & "
                f"{_tex_number(row['log_stratified_adjusted_RiskCoverage'])} & "
                f"{_tex_number(row['log_opportunity_adjusted_RiskCoverage'])} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\caption{\normalsize All top-20\% directed transfer comparisons. "
            r"$C_{\rm all}$ divides captured positive loss by positive target "
            r"loss over all scenes. $E_k$ is relative to the target top-$k$ "
            r"oracle, and $E_{\rm opp}$ is "
            r"defined only for lost counts. B, S, and D denote BEVFusion, "
            r"SparseFusion, and DeepInteraction; comparisons share scenes and "
            r"predictions.}",
            r"\label{tab:supp-all-cells}\\",
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=Path("data/mechanism.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/development-references.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/development-references.md"),
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("data/target_table.parquet"),
    )
    parser.add_argument(
        "--target-sensitivity-table",
        type=Path,
        default=Path("data/target_sensitivity.parquet"),
    )
    parser.add_argument(
        "--identification-replicates",
        type=int,
        default=IDENTIFICATION_REPLICATES,
    )
    parser.add_argument(
        "--identification-seed",
        type=int,
        default=IDENTIFICATION_SEED,
    )
    parser.add_argument(
        "--frequency-seed",
        type=int,
        default=TARGET_FREQUENCY_SEED,
    )
    parser.add_argument(
        "--sensitivity-replicates",
        type=int,
        default=SENSITIVITY_REPLICATES,
    )
    parser.add_argument(
        "--corrected",
        type=Path,
        default=Path("results/development.json"),
    )
    parser.add_argument("--supplement-tables", type=Path)
    args = parser.parse_args()

    source = json.loads(args.mechanism.read_text(encoding="utf-8"))
    assignment_mode = source["deep_mechanism"].get("primary_assignment_mode")
    if assignment_mode not in {"class_agnostic", "class_aware"}:
        raise ValueError("Mechanism artifact lacks a valid primary_assignment_mode")
    target_assignment_mode = _parquet_assignment_mode(args.target_table)
    if target_assignment_mode != assignment_mode:
        raise ValueError(
            "Target table assignment mode does not match the mechanism artifact: "
            f"{target_assignment_mode} != {assignment_mode}"
        )
    validate_target_table_metadata(args.target_table, assignment_mode)
    rows = source["deep_mechanism"]["tail_concentration"]
    conditions_from_rows = sorted({str(row["condition"]) for row in rows})
    signed_axis_condition_blocks(conditions_from_rows)
    condition_count = len(conditions_from_rows)
    expected = int(source["scene_count"]) * condition_count * len(MODEL_ORDER)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} scene rows, found {len(rows)}")
    sensitivity_rows = source["deep_mechanism"]["failure_path_sensitivity"]
    target_columns = [
        "model",
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
        "class",
        "clean_score",
        "distance_m",
        "size_m3",
        "visibility",
        "failure",
    ]
    target_table = pd.read_parquet(args.target_table, columns=target_columns)
    validate_target_table_semantics(target_table)
    target_sensitivity, conditions = load_target_sensitivity_table(
        args.target_sensitivity_table,
        assignment_mode,
        set(target_table["condition"].astype(str).unique()),
    )
    validate_primary_target_alignment(
        target_table,
        target_sensitivity,
        conditions,
    )
    corrected = json.loads(args.corrected.read_text(encoding="utf-8"))
    estimand_sensitivity = summarize_failure_estimand_sensitivity(
        target_sensitivity,
        conditions,
        assignment_mode,
        replicates=args.sensitivity_replicates,
    )
    identification = summarize_identification(
        target_table,
        source["deep_mechanism"]["cross_model"]["cohort_comparison"],
        assignment_mode,
        replicates=args.identification_replicates,
        seed=args.identification_seed,
        frequency_seed=args.frequency_seed,
    )
    primary_sensitivity = next(
        row
        for row in estimand_sensitivity["configurations"]
        if row["score_threshold"] == 0.25 and row["match_distance"] == 2.0
    )
    if (
        primary_sensitivity["detector_annotation_targets"]
        != identification["complete_detector_annotation_targets"]
        or primary_sensitivity["common_unique_annotations"]
        != identification["common_clean_tp_unique_annotations"]
        or primary_sensitivity["observed"]["counts"]
        != identification["observed_common_clean_tp_overlap"]["counts"]
    ):
        raise RuntimeError(
            "Packed sensitivity table does not reproduce the primary target cohort"
        )
    for outcome, sensitivity_key in (
        ("lost_clean_tp_fraction", "lost_fraction_transfer"),
        ("lost_clean_tp_count", "lost_count_transfer"),
    ):
        expected_summary = corrected["outcomes"][outcome]["primary_summary"]
        for field in (
            "raw_overlap",
            "log_stratified_adjusted_overlap",
            "raw_RiskCoverage",
            "log_stratified_adjusted_RiskCoverage",
        ):
            if not math.isclose(
                primary_sensitivity[sensitivity_key][field],
                expected_summary[field]["median"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Packed sensitivity table does not reproduce {outcome} {field}"
                )
    payload = {
        "artifact_type": "reference_models_experiments",
        "evidence_scope": source["evidence_scope"],
        "source_artifact": str(args.mechanism),
        "scene_count": source["scene_count"],
        "log_count": source["log_count"],
        "model_count": len(MODEL_ORDER),
        "condition_count": condition_count,
        "decomposable_tail": summarize_tail(rows),
        "decomposable_transfer": summarize_transfer(rows),
        "failure_path_sensitivity": summarize_failure_sensitivity(sensitivity_rows),
        "failure_estimand_sensitivity": estimand_sensitivity,
        "eligibility_and_marginal_null": identification,
        "cross_fitted_failure_evidence_transfer": (
            summarize_failure_evidence_transfer(target_table)
        ),
        "claim_boundary": (
            (
                "Analysis of the log-disjoint follow-up partition covering "
                "six signed rotation conditions."
            )
            if source["evidence_scope"] == "log_disjoint_confirmation"
            else (
                "Retrospective analysis of the complete paired development "
                "corpus under 12 signed calibration perturbations."
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary.write_text(render_summary(payload), encoding="utf-8")
    if args.supplement_tables is not None:
        args.supplement_tables.parent.mkdir(parents=True, exist_ok=True)
        args.supplement_tables.write_text(
            render_supplement_tables(corrected, payload),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
