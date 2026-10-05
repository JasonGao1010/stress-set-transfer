import math

import numpy as np

from analysis.transfer import (
    _holm_adjust,
    _opportunity_null_counts,
    _random_source_masks_from_observed_masks,
    _sample_observed_top_masks,
    adjusted_coverage,
    adjusted_overlap,
    decision_consequence,
    log_opportunity_expected_risk_mass,
    log_stratified_expected_intersection,
    log_stratified_expected_risk_mass,
    stratified_spearman,
    tie_averaged_transfer,
    top_k_inclusion_probabilities,
    transfer_cell,
)


def _decision_rows(
    overlap_by_source: dict[str, float],
    efficiency_by_source: dict[str, float],
    *,
    target: str = "deepinteraction_base",
) -> list[dict]:
    rows = []
    for source in ("bevfusion_mit", "sparsefusion_r50"):
        efficiency = efficiency_by_source[source]
        rows.append(
            {
                "condition": "condition",
                "source": source,
                "target": target,
                "top_k": 2,
                "raw_overlap": overlap_by_source[source],
                "log_stratified_adjusted_overlap": overlap_by_source[source],
                "raw_RiskCoverage": efficiency,
                "absolute_target_risk_coverage": efficiency * 0.5,
                "target_oracle_risk_mass": 5.0,
                "total_target_positive_risk_mass": 10.0,
            }
        )
    return rows


def test_decision_consequence_detects_strict_source_reversal() -> None:
    result = decision_consequence(
        _decision_rows(
            {"bevfusion_mit": 0.8, "sparsefusion_r50": 0.4},
            {"bevfusion_mit": 0.6, "sparsefusion_r50": 0.9},
        )
    )
    rule = result["rules"]["raw_overlap"]
    assert rule["summary"]["decision_units"] == 1
    assert rule["summary"]["estimable_units"] == 1
    assert rule["summary"]["strict_reversals"] == 1
    unit = rule["units"][0]
    assert math.isclose(unit["total_reuse_regret"], 0.4)
    assert math.isclose(unit["availability_regret"], 0.1)
    assert math.isclose(unit["selection_regret"], 0.3)
    assert math.isclose(unit["regret_decomposition_error"], 0.0)


def test_decision_consequence_averages_selection_ties() -> None:
    result = decision_consequence(
        _decision_rows(
            {"bevfusion_mit": 0.5, "sparsefusion_r50": 0.5},
            {"bevfusion_mit": 0.6, "sparsefusion_r50": 1.0},
        )
    )
    unit = result["rules"]["raw_overlap"]["units"][0]
    assert unit["selection_tie"]
    assert not unit["strict_reversal"]
    assert math.isclose(unit["selected_efficiency"], 0.8)
    assert math.isclose(unit["total_reuse_regret"], 0.2)
    assert math.isclose(unit["availability_regret"], 0.0)
    assert math.isclose(unit["selection_regret"], 0.2)


def test_decision_consequence_exposes_shared_candidate_failure() -> None:
    result = decision_consequence(
        _decision_rows(
            {"bevfusion_mit": 0.8, "sparsefusion_r50": 0.4},
            {"bevfusion_mit": 0.3, "sparsefusion_r50": 0.2},
        )
    )
    unit = result["rules"]["raw_overlap"]["units"][0]
    assert not unit["suboptimal_choice"]
    assert math.isclose(unit["selection_regret"], 0.0)
    assert math.isclose(unit["availability_regret"], 0.7)
    assert math.isclose(unit["total_reuse_regret"], 0.7)


def test_decision_consequence_verifies_coverage_efficiency_identity() -> None:
    result = decision_consequence(
        _decision_rows(
            {"bevfusion_mit": 0.4, "sparsefusion_r50": 0.6},
            {"bevfusion_mit": 0.7, "sparsefusion_r50": 0.8},
        )
    )
    relation = result["coverage_efficiency_relation"]
    assert relation["checked_cells"] == 2
    assert relation["max_absolute_error"] == 0.0


def test_decision_consequence_rejects_incomplete_source_choices() -> None:
    rows = _decision_rows(
        {"bevfusion_mit": 0.4, "sparsefusion_r50": 0.6},
        {"bevfusion_mit": 0.7, "sparsefusion_r50": 0.8},
    )
    with np.testing.assert_raises(RuntimeError):
        decision_consequence(rows[:1])


def test_global_top_k_overlap_has_nonzero_chance_baseline() -> None:
    mapping = {f"s{i}": "log" for i in range(10)}
    source = {f"s{i}": float(i < 2) for i in range(10)}
    target = {f"s{i}": float(2 <= i < 4) for i in range(10)}
    row = transfer_cell(source, target, mapping, 0.20)
    assert row["raw_overlap"] == 0.0
    assert row["global_chance_overlap"] == 0.2
    assert row["global_adjusted_overlap"] == -0.25


def test_log_composition_can_create_spurious_perfect_overlap() -> None:
    mapping = {
        **{f"a{i}": "a" for i in range(2)},
        **{f"b{i}": "b" for i in range(8)},
    }
    source_set = {"a0", "a1"}
    target_set = {"a0", "a1"}
    expected = log_stratified_expected_intersection(
        source_set, target_set, mapping
    )
    assert expected == 2.0
    assert adjusted_overlap(2, expected, 2) is None


def test_log_stratified_random_risk_baseline_uses_actual_log_counts() -> None:
    mapping = {
        "a0": "a",
        "a1": "a",
        "b0": "b",
        "b1": "b",
    }
    source_set = {"a0", "b0"}
    target_risk = {"a0": 4.0, "a1": 0.0, "b0": 2.0, "b1": 0.0}
    expected = log_stratified_expected_risk_mass(
        source_set, target_risk, mapping
    )
    assert expected == 3.0


def test_opportunity_baseline_uses_log_rate_and_scene_exposure() -> None:
    mapping = {"a": "log", "b": "log"}
    expected = log_opportunity_expected_risk_mass(
        {"a"},
        {"a": 2.0, "b": 8.0},
        {"a": 10.0, "b": 10.0},
        mapping,
    )
    assert expected == 5.0


def test_transfer_cell_reports_opportunity_adjusted_coverage() -> None:
    mapping = {f"s{i}": "log" for i in range(4)}
    source = {"s0": 4.0, "s1": 3.0, "s2": 2.0, "s3": 1.0}
    target = {"s0": 2.0, "s1": 8.0, "s2": 0.0, "s3": 0.0}
    opportunity = {scene: 10.0 for scene in source}
    row = transfer_cell(source, target, mapping, 0.25, opportunity)
    assert row["log_opportunity_expected_target_risk_mass"] == 2.5
    assert row["log_opportunity_risk_lift"] == 0.8
    assert row["log_opportunity_expected_RiskCoverage"] == 2.5 / 8.0
    assert row["total_target_positive_risk_mass"] == 10.0
    assert row["absolute_target_risk_coverage"] == 0.2


def test_adjusted_metrics_are_zero_at_null_and_one_at_oracle() -> None:
    assert adjusted_overlap(2, 2.0, 5) == 0.0
    assert adjusted_overlap(5, 2.0, 5) == 1.0
    assert adjusted_coverage(0.4, 0.4) == 0.0
    assert adjusted_coverage(1.0, 0.4) == 1.0


def test_within_log_rank_correlation_removes_between_log_location_shift() -> None:
    mapping = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
    source = {"a0": 100.0, "a1": 99.0, "b0": 2.0, "b1": 1.0}
    target = {"a0": 2.0, "a1": 1.0, "b0": 100.0, "b1": 99.0}
    rho = stratified_spearman(source, target, mapping)
    assert rho is not None
    assert math.isclose(rho, 1.0)


def test_boundary_tie_probabilities_select_exactly_k_in_expectation() -> None:
    scores = {"a": 3.0, "b": 2.0, "c": 2.0, "d": 2.0, "e": 0.0}
    probabilities = top_k_inclusion_probabilities(scores, 2)
    assert probabilities["a"] == 1.0
    assert probabilities["b"] == probabilities["c"] == probabilities["d"] == 1 / 3
    assert probabilities["e"] == 0.0
    assert math.isclose(sum(probabilities.values()), 2.0)


def test_near_boundary_values_form_one_disjoint_tie_partition() -> None:
    scores = {"a": 1.0 + 5e-10, "b": 1.0, "c": 1.0, "d": 0.0}
    probabilities = top_k_inclusion_probabilities(scores, 2)
    assert probabilities == {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3, "d": 0.0}
    assert math.isclose(sum(probabilities.values()), 2.0)

    scenes = list(scores)
    masks = _sample_observed_top_masks(
        scores, scenes, 2, 1_000, np.random.default_rng(7)
    )
    assert np.all(masks.sum(axis=1) == 2)


def test_exact_probabilities_and_sampled_masks_share_tie_tolerance() -> None:
    scores = {"a": 1.0 + 5e-7, "b": 1.0, "c": 1.0, "d": 0.0}
    probabilities = top_k_inclusion_probabilities(scores, 2)
    assert probabilities == {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.0}

    scenes = list(scores)
    masks = _sample_observed_top_masks(
        scores, scenes, 2, 1_000, np.random.default_rng(11)
    )
    assert np.all(masks.sum(axis=1) == 2)
    assert masks[:, scenes.index("a")].all()


def test_tie_predicate_is_identical_at_the_formula_boundary() -> None:
    scores = {"a": 1.0 + 1.0005e-9, "b": 1.0, "c": 1.0, "d": 0.0}
    probabilities = top_k_inclusion_probabilities(scores, 2)
    assert probabilities == {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.0}

    scenes = list(scores)
    masks = _sample_observed_top_masks(
        scores, scenes, 2, 10_000, np.random.default_rng(13)
    )
    assert masks[:, scenes.index("a")].all()
    assert np.all(masks.sum(axis=1) == 2)


def test_tie_averaged_transfer_is_exact_for_all_tied_sets() -> None:
    source = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    target = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    mapping = {scene: "log" for scene in source}
    result = tie_averaged_transfer(source, target, mapping, 2)
    assert result["tie_random_expected_raw_overlap"] == 0.5
    assert result["tie_random_expected_log_stratified_chance_overlap"] == 0.5
    assert result["tie_random_expected_log_stratified_adjusted_overlap"] == 0.0


def test_source_randomization_preserves_every_replicate_log_count() -> None:
    scenes = ["a0", "a1", "b0", "b1", "b2"]
    scene_to_log = {
        "a0": "a",
        "a1": "a",
        "b0": "b",
        "b1": "b",
        "b2": "b",
    }
    observed = np.asarray(
        [
            [True, False, True, False, False],
            [False, True, True, True, False],
            [True, True, False, False, True],
            [False, False, True, True, True],
        ],
        dtype=bool,
    )
    randomized = _random_source_masks_from_observed_masks(
        observed,
        scenes,
        scene_to_log,
        np.random.default_rng(17),
    )
    for positions in ([0, 1], [2, 3, 4]):
        assert np.array_equal(
            randomized[:, positions].sum(axis=1),
            observed[:, positions].sum(axis=1),
        )


def test_opportunity_draws_preserve_log_totals_and_capacity() -> None:
    scenes = ["a0", "a1", "b0", "b1"]
    scene_to_log = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
    scores = {"a0": 2.0, "a1": 1.0, "b0": 0.0, "b1": 1.0}
    opportunities = {"a0": 3.0, "a1": 2.0, "b0": 4.0, "b1": 1.0}
    draws = _opportunity_null_counts(
        scores,
        opportunities,
        scenes,
        scene_to_log,
        1_000,
        np.random.default_rng(19),
    )
    assert np.all(draws >= 0)
    assert np.all(draws <= np.asarray([3, 2, 4, 1]))
    assert np.all(draws[:, [0, 1]].sum(axis=1) == 3)
    assert np.all(draws[:, [2, 3]].sum(axis=1) == 1)


def test_holm_adjustment_is_bounded_and_monotone_in_rank() -> None:
    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.8}
    adjusted = _holm_adjust(raw)
    assert adjusted == {"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.8}
    ordered = sorted(raw, key=raw.get)
    assert all(0 <= adjusted[key] <= 1 for key in raw)
    assert all(adjusted[key] >= raw[key] for key in raw)
    assert all(
        adjusted[left] <= adjusted[right]
        for left, right in zip(ordered, ordered[1:])
    )


def test_transfer_cell_uses_tie_expectation_as_primary() -> None:
    source = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    target = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    mapping = {scene: "log" for scene in source}
    row = transfer_cell(source, target, mapping, 0.50)
    assert row["lexical_tie_break_raw_overlap"] == 1.0
    assert row["raw_overlap"] == 0.5
    assert row["lexical_tie_break_raw_RiskCoverage"] == 1.0
    assert math.isclose(row["raw_RiskCoverage"], 5.0 / 7.0)
    assert row["lexical_tie_break_absolute_target_risk_coverage"] == 0.7
    assert row["absolute_target_risk_coverage"] == 0.5
