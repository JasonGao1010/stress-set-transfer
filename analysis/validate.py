"""Independent arithmetic audit of persisted corrected-transfer cells.

The validator intentionally does not import analysis.transfer. It
reconstructs every top-20% cell, including exact boundary-tie expectations,
from the two source artifacts and checks the persisted estimands.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median


MODELS = ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
OUTCOMES = ("NDS_like_loss", "lost_clean_tp_fraction", "lost_clean_tp_count")
TOLERANCE = 1e-12
TIE_REL_TOLERANCE = 1e-9
TIE_ABS_TOLERANCE = 1e-12


def _is_boundary_tie(value: float, boundary: float) -> bool:
    """Mirror the declared producer tie predicate independently."""
    return math.isclose(
        float(value),
        float(boundary),
        rel_tol=TIE_REL_TOLERANCE,
        abs_tol=TIE_ABS_TOLERANCE,
    )


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), abs_tol=TOLERANCE, rel_tol=1e-10)


def _top(scores: dict[str, float], k: int) -> set[str]:
    return set(sorted(scores, key=lambda scene: (-scores[scene], scene))[:k])


def _probabilities(scores: dict[str, float], k: int) -> dict[str, float]:
    selected = _top(scores, k)
    boundary = min(scores[scene] for scene in selected)
    tied = {
        scene
        for scene, value in scores.items()
        if _is_boundary_tie(value, boundary)
    }
    above = {
        scene
        for scene, value in scores.items()
        if value > boundary and scene not in tied
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
        sum(probabilities.values()), float(k), rel_tol=0.0, abs_tol=TOLERANCE
    ):
        raise RuntimeError("Top-k inclusion probabilities do not sum to k")
    return probabilities


def _tables(primary: dict, mechanism: dict) -> tuple[dict, dict]:
    tables = {outcome: {} for outcome in OUTCOMES}
    opportunities = {}
    for row in primary["scene_rows"]:
        tables["NDS_like_loss"][(row["model"], row["candidate"])] = {
            scene: float(value)
            for scene, value in row["NDS_losses_by_scene"].items()
        }
    for row in mechanism["deep_mechanism"]["tail_concentration"]:
        key = (row["model"], row["condition"])
        tables["lost_clean_tp_fraction"].setdefault(key, {})[row["scene"]] = float(
            row["L_status"]
        )
        tables["lost_clean_tp_count"].setdefault(key, {})[row["scene"]] = float(
            row["lost_clean_tp_count"]
        )
        opportunities.setdefault(key, {})[row["scene"]] = float(
            row["clean_tp_count"]
        )
    return tables, opportunities


def _maximizers(rows: list[dict], metric: str) -> list[dict]:
    maximum = max(float(row[metric]) for row in rows)
    return [
        row
        for row in rows
        if math.isclose(
            float(row[metric]), maximum, rel_tol=0.0, abs_tol=TOLERANCE
        )
    ]


def _decision_expected(rows: list[dict], metric: str) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target"]), str(row["condition"]))].append(row)
    units = []
    for (target, condition), candidates in sorted(grouped.items()):
        if len(candidates) != len(MODELS) - 1:
            raise RuntimeError("Independent decision audit requires two sources")
        selected = _maximizers(candidates, metric)
        best = _maximizers(candidates, "raw_RiskCoverage")
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
        random_selection_regret = max(
            0.0, best_utility - random_utility
        )
        selected_sources = sorted(str(row["source"]) for row in selected)
        best_sources = sorted(str(row["source"]) for row in best)
        units.append(
            {
                "target": target,
                "condition": condition,
                "selected_sources": selected_sources,
                "best_utility_sources": best_sources,
                "selection_tie": len(selected_sources) > 1,
                "utility_tie": len(best_sources) > 1,
                "strict_reversal": (
                    len(selected_sources) == 1
                    and len(best_sources) == 1
                    and selected_sources != best_sources
                ),
                "suboptimal_choice": selection_regret > TOLERANCE,
                "selected_efficiency": selected_utility,
                "best_candidate_efficiency": best_utility,
                "random_candidate_efficiency": random_utility,
                "total_reuse_regret": total_reuse_regret,
                "availability_regret": availability_regret,
                "selection_regret": selection_regret,
                "relative_selection_regret": (
                    selection_regret / best_utility
                    if best_utility > TOLERANCE
                    else None
                ),
                "random_selection_regret": random_selection_regret,
                "selection_advantage_over_random": (
                    random_selection_regret - selection_regret
                ),
                "regret_decomposition_error": abs(
                    total_reuse_regret
                    - availability_regret
                    - selection_regret
                ),
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
        value for value in selection_regrets if value > TOLERANCE
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
    mean_total = fmean(total_regrets)
    mean_availability = fmean(availability_regrets)
    mean_selection = fmean(selection_regrets)
    return units, {
        "decision_units": len({str(row["target"]) for row in rows})
        * len({str(row["condition"]) for row in rows}),
        "estimable_units": count,
        "strict_reversals": sum(bool(unit["strict_reversal"]) for unit in units),
        "strict_reversal_fraction": (
            sum(bool(unit["strict_reversal"]) for unit in units) / count
        ),
        "selection_ties": sum(bool(unit["selection_tie"]) for unit in units),
        "utility_ties": sum(bool(unit["utility_tie"]) for unit in units),
        "positive_selection_regret_units": len(positive_selection),
        "positive_selection_regret_fraction": (
            len(positive_selection) / count
        ),
        "total_reuse_regret_at_least_0_02_units": sum(
            value >= 0.02 - TOLERANCE for value in total_regrets
        ),
        "mean_total_reuse_regret": mean_total,
        "median_total_reuse_regret": median(total_regrets),
        "max_total_reuse_regret": max(total_regrets),
        "mean_availability_regret": mean_availability,
        "median_availability_regret": median(availability_regrets),
        "mean_selection_regret": mean_selection,
        "median_selection_regret": median(selection_regrets),
        "median_positive_selection_regret": (
            median(positive_selection) if positive_selection else 0.0
        ),
        "mean_relative_selection_regret": fmean(relative_selection),
        "selection_share_of_mean_total_regret": (
            mean_selection / mean_total if mean_total > TOLERANCE else None
        ),
        "mean_random_selection_regret": fmean(random_selection_regrets),
        "mean_selection_advantage_over_random": fmean(advantages),
        "max_regret_decomposition_error": max(decomposition_errors),
    }


def validate(primary: dict, mechanism: dict, corrected: dict) -> dict:
    mapping = primary["scene_log_tokens"]
    tables, opportunities = _tables(primary, mechanism)
    condition_count = len(
        {condition for _, condition in tables["NDS_like_loss"]}
    )
    expected_primary_cells = (
        len(OUTCOMES) * condition_count * len(MODELS) * (len(MODELS) - 1)
    )
    failures: list[str] = []
    checked = 0
    checked_decisions = 0
    for outcome in OUTCOMES:
        cells = corrected["outcomes"][outcome]["fractions"]["0.20"]["cells"]
        recomputed_cells: list[dict] = []
        table = tables[outcome]
        conditions = sorted({condition for _, condition in table})
        expected_cell_count = len(conditions) * len(MODELS) * (len(MODELS) - 1)
        if len(cells) != expected_cell_count:
            failures.append(
                f"{outcome}: expected {expected_cell_count} cells, found {len(cells)}"
            )
        expected_keys = {
            (condition, source, target)
            for condition in conditions
            for source in MODELS
            for target in MODELS
            if source != target
        }
        actual_keys = {
            (row["condition"], row["source"], row["target"]) for row in cells
        }
        if actual_keys != expected_keys:
            failures.append(f"{outcome}: directed cell identities differ")
        for row in cells:
            checked += 1
            condition = row["condition"]
            source_scores = table[(row["source"], condition)]
            target_scores = table[(row["target"], condition)]
            n = len(source_scores)
            k = math.ceil(0.20 * n)
            source = _top(source_scores, k)
            target = _top(target_scores, k)
            intersection = len(source & target)
            source_probability = _probabilities(source_scores, k)
            target_probability = _probabilities(target_scores, k)
            expected_observed_intersection = sum(
                source_probability[scene] * target_probability[scene]
                for scene in source_scores
            )
            by_log: dict[str, set[str]] = defaultdict(set)
            for scene, log in mapping.items():
                by_log[log].add(scene)
            expected_intersection = sum(
                sum(source_probability[scene] for scene in scenes)
                * sum(target_probability[scene] for scene in scenes)
                / len(scenes)
                for scenes in by_log.values()
            )
            adjusted = (expected_observed_intersection - expected_intersection) / (
                k - expected_intersection
            )
            target_risk = {
                scene: max(value, 0.0) for scene, value in target_scores.items()
            }
            observed_mass = sum(
                source_probability[scene] * target_risk[scene]
                for scene in source_scores
            )
            total_positive_mass = sum(target_risk.values())
            oracle_mass = sum(
                target_probability[scene] * target_risk[scene]
                for scene in target_scores
            )
            coverage = observed_mass / oracle_mass
            absolute_coverage = observed_mass / total_positive_mass
            expected_mass = sum(
                sum(source_probability[scene] for scene in scenes)
                / len(scenes)
                * sum(target_risk[scene] for scene in scenes)
                for scenes in by_log.values()
            )
            null_coverage = expected_mass / oracle_mass
            adjusted_coverage = (coverage - null_coverage) / (1 - null_coverage)
            recomputed_cells.append(
                {
                    "condition": condition,
                    "source": row["source"],
                    "target": row["target"],
                    "raw_overlap": expected_observed_intersection / k,
                    "log_stratified_adjusted_overlap": adjusted,
                    "raw_RiskCoverage": coverage,
                    "absolute_target_risk_coverage": absolute_coverage,
                    "target_oracle_risk_mass": oracle_mass,
                    "total_target_positive_risk_mass": total_positive_mass,
                }
            )
            checks = {
                "top_k": (row["top_k"], k),
                "observed_intersection": (
                    row["observed_intersection"],
                    intersection,
                ),
                "lexical_tie_break_raw_overlap": (
                    row["lexical_tie_break_raw_overlap"],
                    intersection / k,
                ),
                "raw_overlap": (
                    row["raw_overlap"],
                    expected_observed_intersection / k,
                ),
                "log_stratified_chance_overlap": (
                    row["log_stratified_chance_overlap"],
                    expected_intersection / k,
                ),
                "log_stratified_adjusted_overlap": (
                    row["log_stratified_adjusted_overlap"],
                    adjusted,
                ),
                "raw_RiskCoverage": (row["raw_RiskCoverage"], coverage),
                "total_target_positive_risk_mass": (
                    row["total_target_positive_risk_mass"],
                    total_positive_mass,
                ),
                "absolute_target_risk_coverage": (
                    row["absolute_target_risk_coverage"],
                    absolute_coverage,
                ),
                "log_stratified_random_RiskCoverage": (
                    row["log_stratified_random_RiskCoverage"],
                    null_coverage,
                ),
                "log_stratified_adjusted_RiskCoverage": (
                    row["log_stratified_adjusted_RiskCoverage"],
                    adjusted_coverage,
                ),
            }
            if outcome == "lost_clean_tp_count":
                target_opportunity = opportunities[(row["target"], condition)]
                opportunity_mass = 0.0
                for scenes in by_log.values():
                    total_opportunity = sum(
                        target_opportunity[scene] for scene in scenes
                    )
                    log_rate = (
                        sum(target_risk[scene] for scene in scenes)
                        / total_opportunity
                    )
                    opportunity_mass += log_rate * sum(
                        source_probability[scene] * target_opportunity[scene]
                        for scene in scenes
                    )
                opportunity_coverage = opportunity_mass / oracle_mass
                opportunity_adjusted = (
                    coverage - opportunity_coverage
                ) / (1 - opportunity_coverage)
                checks.update(
                    {
                        "log_opportunity_expected_target_risk_mass": (
                            row["log_opportunity_expected_target_risk_mass"],
                            opportunity_mass,
                        ),
                        "log_opportunity_expected_RiskCoverage": (
                            row["log_opportunity_expected_RiskCoverage"],
                            opportunity_coverage,
                        ),
                        "log_opportunity_adjusted_RiskCoverage": (
                            row["log_opportunity_adjusted_RiskCoverage"],
                            opportunity_adjusted,
                        ),
                        "log_opportunity_risk_lift": (
                            row["log_opportunity_risk_lift"],
                            observed_mass / opportunity_mass,
                        ),
                    }
                )
            for field, (persisted, recomputed) in checks.items():
                if not _close(persisted, recomputed):
                    failures.append(
                        f"{outcome}/{condition}/{row['source']}->{row['target']}: "
                        f"{field} persisted={persisted} recomputed={recomputed}"
                    )
        persisted_decision = corrected["outcomes"][outcome]["fractions"]["0.20"][
            "decision_consequence"
        ]
        relation_errors = [
            abs(
                float(row["absolute_target_risk_coverage"])
                - float(row["raw_RiskCoverage"])
                * float(row["target_oracle_risk_mass"])
                / float(row["total_target_positive_risk_mass"])
            )
            for row in recomputed_cells
        ]
        relation = persisted_decision["coverage_efficiency_relation"]
        if int(relation["checked_cells"]) != len(relation_errors):
            failures.append(f"{outcome}: coverage-efficiency cell count differs")
        if not _close(
            relation["max_absolute_error"], max(relation_errors, default=None)
        ):
            failures.append(f"{outcome}: coverage-efficiency identity differs")
        for metric in ("raw_overlap", "log_stratified_adjusted_overlap"):
            expected_units, expected_summary = _decision_expected(
                recomputed_cells, metric
            )
            checked_decisions += len(expected_units)
            persisted_rule = persisted_decision["rules"][metric]
            persisted_units = {
                (unit["target"], unit["condition"]): unit
                for unit in persisted_rule["units"]
            }
            if len(persisted_units) != len(expected_units):
                failures.append(f"{outcome}/{metric}: decision unit count differs")
                continue
            for expected in expected_units:
                key = (expected["target"], expected["condition"])
                actual = persisted_units.get(key)
                if actual is None:
                    failures.append(
                        f"{outcome}/{metric}/{key}: decision unit missing"
                    )
                    continue
                for field in (
                    "selected_sources",
                    "best_utility_sources",
                    "selection_tie",
                    "utility_tie",
                    "strict_reversal",
                    "suboptimal_choice",
                ):
                    if actual[field] != expected[field]:
                        failures.append(
                            f"{outcome}/{metric}/{key}: {field} differs"
                        )
                for field in (
                    "selected_efficiency",
                    "best_candidate_efficiency",
                    "random_candidate_efficiency",
                    "total_reuse_regret",
                    "availability_regret",
                    "selection_regret",
                    "relative_selection_regret",
                    "random_selection_regret",
                    "selection_advantage_over_random",
                    "regret_decomposition_error",
                ):
                    if not _close(actual[field], expected[field]):
                        failures.append(
                            f"{outcome}/{metric}/{key}: {field} differs"
                        )
            actual_summary = persisted_rule["summary"]
            for field, expected in expected_summary.items():
                actual = actual_summary[field]
                if isinstance(expected, float):
                    matches = _close(actual, expected)
                else:
                    matches = actual == expected
                if not matches:
                    failures.append(
                        f"{outcome}/{metric}: summary {field} differs"
                    )
    for outcome in OUTCOMES:
        test = corrected["outcomes"][outcome]["conditional_randomization"]
        for family in ("overlap", "RiskCoverage"):
            raw_p = float(test[family]["one_sided_p_greater"])
            adjusted_p = float(test[family]["holm_adjusted_p_across_outcomes"])
            if not 0 <= raw_p <= adjusted_p <= 1:
                failures.append(
                    f"{outcome}/{family}: invalid raw/Holm p-value ordering"
                )
        if int(test["replicates"]) != 10_000:
            failures.append(f"{outcome}: randomization replicate count differs")
    opportunity_test = corrected["outcomes"]["lost_clean_tp_count"].get(
        "opportunity_conditional_randomization"
    )
    if opportunity_test is None:
        failures.append("lost_clean_tp_count: missing opportunity randomization")
    else:
        if int(opportunity_test["replicates"]) != 10_000:
            failures.append("opportunity randomization replicate count differs")
        for family in ("overlap", "RiskCoverage"):
            raw_p = float(opportunity_test[family]["one_sided_p_greater"])
            adjusted_p = float(
                opportunity_test[family]["holm_adjusted_p_across_metrics"]
            )
            if not 0 <= raw_p <= adjusted_p <= 1:
                failures.append(
                    f"opportunity/{family}: invalid raw/Holm p-value ordering"
                )
    return {
        "artifact_type": "transfer_independent_arithmetic_validation",
        "passed": not failures,
        "checked_primary_cells": checked,
        "expected_primary_cells": expected_primary_cells,
        "checked_lexical_tie_sensitivity_cells": checked,
        "checked_primary_decision_rules": checked_decisions,
        "formula_tolerance": TOLERANCE,
        "checks": [
            "cell identity and count",
            "exact top-k boundary-tie inclusion probabilities",
            "lexical tie-break sensitivity reconstruction",
            "raw overlap",
            "log-stratified overlap null",
            "chance-adjusted overlap",
            "raw risk coverage",
            "log-stratified random risk-coverage null",
            "chance-adjusted risk coverage",
            "log-specific clean-TP opportunity-adjusted risk coverage",
            "randomization replicate metadata",
            "raw and Holm p-value ordering for both randomization families",
            "coverage-efficiency algebraic identity",
            "identity-based source choice and target-oracle reuse-regret decomposition",
        ],
        "failures": failures,
        "boundary": (
            "Independent arithmetic reconstruction of the released "
            "scene-level inputs and transfer results."
        ),
    }


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
        "--corrected",
        type=Path,
        default=Path("results/development.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/development-validation.json"),
    )
    args = parser.parse_args()
    result = validate(
        json.loads(args.primary.read_text(encoding="utf-8")),
        json.loads(args.mechanism.read_text(encoding="utf-8")),
        json.loads(args.corrected.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["passed"]:
        raise SystemExit("Corrected-transfer validation failed")


if __name__ == "__main__":
    main()
