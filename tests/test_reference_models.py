import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analysis.reference_models import (
    SENSITIVITY_CONFIGS,
    _merge_common_gt_metadata,
    _metric_direction,
    _switch_edges,
    _switch_state,
    average_condition_summaries,
    build_aligned_failure_transfer_table,
    context_stratified_failure_reference,
    directed_transfer,
    expand_target_sensitivity,
    fold_clean_score_vulnerability,
    fold_source_failure_rate,
    load_target_sensitivity_table,
    marginal_failure_null,
    multi_chain_target_frequency_diagnostics,
    signed_axis_condition_blocks,
    single_source_failure_probabilities,
    summarize_failure_evidence_transfer,
    summarize_failure_estimand_sensitivity,
    summarize_identification,
    target_frequency_null,
    validate_primary_target_alignment,
    validate_target_table_metadata,
    weighted_cvar,
)


def test_weighted_cvar_uses_fractional_boundary_mass() -> None:
    assert math.isclose(weighted_cvar([0.0, 1.0, 2.0, 3.0], 0.625), 8.0 / 3.0)
    assert math.isclose(
        weighted_cvar([0.0, 1.0, 3.0], 0.5, [1.0, 1.0, 2.0]),
        3.0,
    )


def test_metric_direction_distinguishes_ranking_from_brier() -> None:
    assert _metric_direction(-0.01, -0.001) == "ranking_declined_brier_improved"


def test_aggregate_ratio_is_ratio_of_averages() -> None:
    summaries = [
        {"mean": 1.0, "CVaR80": 2.0, "CVaR90": 2.0, "CVaR95": 2.0},
        {"mean": 3.0, "CVaR80": 3.0, "CVaR90": 3.0, "CVaR95": 3.0},
    ]
    result = average_condition_summaries(summaries)
    assert result["mean"] == 2.0
    assert result["CVaR90"] == 2.5
    assert result["CVaR90_over_mean"] == 1.25


def test_directed_transfer_uses_source_set_and_target_risk_mass() -> None:
    rows = []
    source = [4.0, 3.0, 2.0, 1.0, 0.0]
    target = [0.0, 1.0, 2.0, 3.0, 4.0]
    middle = [0.0, 1.0, 4.0, 3.0, 2.0]
    for model, values in (
        ("bevfusion_mit", source),
        ("sparsefusion_r50", target),
        ("deepinteraction_base", middle),
    ):
        for index, value in enumerate(values):
            rows.append(
                {
                    "model": model,
                    "condition": "c",
                    "scene": f"s{index}",
                    "L_status": value,
                }
            )

    result = directed_transfer(rows, "L_status")
    row = next(
        item
        for item in result
        if item["source"] == "bevfusion_mit" and item["target"] == "sparsefusion_r50"
    )
    assert row["top_k"] == 1
    assert row["overlap"] == 0.0
    assert row["RiskCoverage"] == 0.0
    assert row["RiskSetTransferRegret"] == 1.0


def test_marginal_failure_null_preserves_pairwise_margins() -> None:
    result = marginal_failure_null(
        [
            {
                "condition": "c",
                "log": "l",
                "n": 100,
                "failures": {
                    "bevfusion_mit": 10,
                    "sparsefusion_r50": 20,
                    "deepinteraction_base": 30,
                },
            }
        ],
        replicates=2_000,
        seed=7,
    )
    expected_ab_jaccard = 2.0 / (10 + 20 - 2.0)
    observed = result["pairwise_jaccard"]["bevfusion_mit__sparsefusion_r50"]["median"]
    assert abs(observed - expected_ab_jaccard) < 0.03
    assert (
        result["specific_fraction_among_any_failure"]["ci95"][0]
        <= result["specific_fraction_among_any_failure"]["median"]
        <= result["specific_fraction_among_any_failure"]["ci95"][1]
    )


def test_binary_switch_chain_preserves_annotation_and_condition_margins() -> None:
    matrix = np.asarray(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    state = _switch_state(matrix)
    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    _switch_edges(state, 10_000, np.random.default_rng(7))
    reconstructed = np.zeros_like(matrix)
    reconstructed[state["edge_rows"], state["edge_columns"]] = True
    assert np.array_equal(reconstructed.sum(axis=1), row_sums)
    assert np.array_equal(reconstructed.sum(axis=0), column_sums)


def test_family_blocked_target_frequency_null_preserves_blocks() -> None:
    conditions = ("D_RX_A_LOW", "D_RX_B_LOW", "D_TX_A_LOW", "D_TX_B_LOW")
    rows = []
    for log in ("l1",):
        for annotation in ("a1", "a2", "a3", "a4"):
            for condition_index, condition in enumerate(conditions):
                row = {
                    "log": log,
                    "annotation_token": annotation,
                    "condition": condition,
                }
                annotation_index = int(annotation[-1])
                for model_index, model in enumerate(
                    ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
                ):
                    row[model] = (
                        annotation_index + condition_index + model_index
                    ) % 3 == 0
                rows.append(row)
    result = target_frequency_null(
        pd.DataFrame(rows),
        replicates=20,
        seed=7,
        condition_blocks=(conditions[:2], conditions[2:]),
    )
    assert result["condition_blocks"] == [
        list(conditions[:2]),
        list(conditions[2:]),
    ]
    assert result["replicates"] == 20


def test_axis_blocks_and_multichain_diagnostics_are_explicit() -> None:
    conditions = tuple(
        f"D_{axis}_{direction}_LOW"
        for axis in ("RX", "RY", "RZ", "TX", "TY", "TZ")
        for direction in ("NEGATIVE", "POSITIVE")
    )
    blocks = signed_axis_condition_blocks(conditions)
    rows = []
    for annotation_index in range(6):
        for condition_index, condition in enumerate(conditions):
            row = {
                "log": "l1",
                "annotation_token": f"a{annotation_index}",
                "condition": condition,
            }
            for model_index, model in enumerate(
                ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
            ):
                row[model] = (annotation_index + condition_index + model_index) % 4 == 0
            rows.append(row)
    result = multi_chain_target_frequency_diagnostics(
        pd.DataFrame(rows),
        replicates_per_chain=8,
        chains=2,
        seed=7,
        condition_blocks=blocks,
    )
    assert len(result["condition_blocks"]) == 6
    shared = result["metrics"]["shared_fraction"]
    assert len(shared["chains"]) == 2
    assert all("effective_sample_size" in chain for chain in shared["chains"])
    assert "split_r_hat" in shared
    assert "not provide independent p-values" in result["diagnostic_boundary"]


def test_axis_blocks_accept_frozen_rotation_only_confirmation_scope() -> None:
    conditions = tuple(
        f"D_{axis}_{direction}_LOW"
        for axis in ("RX", "RY", "RZ")
        for direction in ("NEGATIVE", "POSITIVE")
    )
    assert signed_axis_condition_blocks(conditions) == (
        ("D_RX_NEGATIVE_LOW", "D_RX_POSITIVE_LOW"),
        ("D_RY_NEGATIVE_LOW", "D_RY_POSITIVE_LOW"),
        ("D_RZ_NEGATIVE_LOW", "D_RZ_POSITIVE_LOW"),
    )


def test_context_strata_validate_gt_metadata_and_report_degeneracy() -> None:
    identity = [
        "condition",
        "log",
        "scene",
        "sample_token",
        "annotation_token",
    ]
    rows = []
    for annotation_index in range(6):
        for model_index, model in enumerate(
            ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
        ):
            rows.append(
                {
                    "model": model,
                    "condition": "D_RX_NEGATIVE_LOW",
                    "log": "l1",
                    "scene": "s1",
                    "sample_token": f"p{annotation_index}",
                    "annotation_token": f"a{annotation_index}",
                    "class": "car",
                    "visibility": 4,
                    "distance_m": float(annotation_index + 1),
                    "size_m3": float(annotation_index + 2),
                    "failure": (annotation_index + model_index) % 3 == 0,
                }
            )
    table = pd.DataFrame(rows)
    wide = table.pivot(index=identity, columns="model", values="failure").reset_index()
    common = _merge_common_gt_metadata(table, wide, identity)
    strata, diagnostics = context_stratified_failure_reference(common)
    assert strata
    assert diagnostics["strata"] == len(strata)
    assert "distance_m_tertiles" in diagnostics["bin_definitions"]
    assert "not causal adjustment" in diagnostics["interpretation"]

    inconsistent = table.copy()
    inconsistent.loc[0, "class"] = "truck"
    with pytest.raises(ValueError, match="GT metadata disagree"):
        _merge_common_gt_metadata(inconsistent, wide, identity)


def test_identification_reports_axis_context_and_multichain_boundaries() -> None:
    conditions = tuple(
        f"D_{axis}_{direction}_LOW"
        for axis in ("RX", "RY", "RZ", "TX", "TY", "TZ")
        for direction in ("NEGATIVE", "POSITIVE")
    )
    rows = []
    for annotation_index in range(6):
        for model_index, model in enumerate(
            ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
        ):
            for condition_index, condition in enumerate(conditions):
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "log": "l1",
                        "scene": "s1",
                        "sample_token": f"p{annotation_index}",
                        "annotation_token": f"a{annotation_index}",
                        "class": "car",
                        "visibility": 4,
                        "distance_m": float(annotation_index + 1),
                        "size_m3": float(annotation_index + 2),
                        "failure": (annotation_index + condition_index + model_index)
                        % 4
                        == 0,
                    }
                )
    result = summarize_identification(
        pd.DataFrame(rows),
        [],
        "class_aware",
        replicates=4,
        seed=7,
        frequency_seed=9,
        diagnostic_chains=2,
        diagnostic_replicates=4,
    )
    assert result["assignment_mode"] == "class_aware"
    assert len(result["target_axis_frequency_null"]["condition_blocks"]) == 6
    assert result["context_stratified_fixed_rate_reference"]["unit"].endswith(
        "distance-tertile-size-tertile"
    )
    assert (
        "context_stratified_fixed_rate_reference" in result["leave_one_log_out"]["l1"]
    )
    assert "physical-instance trajectories vary across reference draws" in result["claim_boundary"]


def test_failure_estimand_assignment_mode_is_not_hard_coded() -> None:
    with pytest.raises(ValueError, match="Unsupported assignment mode"):
        summarize_failure_estimand_sensitivity(
            pd.DataFrame(),
            (),
            "not_a_matching_mode",
            replicates=4,
        )


def _write_mechanism_sensitivity(
    path,
    *,
    producer="analysis/mechanism.py",
    config_order=None,
):
    frame = pd.DataFrame(
        [
            {
                "model": model,
                "log": "log",
                "scene": "scene",
                "sample_token": "sample",
                "annotation_token": "annotation",
                "condition_mask": 0b11,
                "eligible_config_mask": 1 << 4,
                **{
                    f"failure_mask_{index}": (
                        1 if index == 4 and model != "sparsefusion_r50" else 0
                    )
                    for index in range(len(SENSITIVITY_CONFIGS))
                },
            }
            for model in (
                "bevfusion_mit",
                "sparsefusion_r50",
                "deepinteraction_base",
            )
        ]
    )
    for column in (
        "condition_mask",
        "eligible_config_mask",
        *[f"failure_mask_{index}" for index in range(len(SENSITIVITY_CONFIGS))],
    ):
        frame[column] = frame[column].astype("uint16")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            b"producer": producer.encode(),
            b"assignment_mode": b"class_aware",
            b"config_order": json.dumps(
                config_order
                if config_order is not None
                else [
                    {
                        "score_threshold": score,
                        "match_distance_m": distance,
                    }
                    for score, distance in SENSITIVITY_CONFIGS
                ]
            ).encode(),
            b"condition_order": json.dumps(["condition_b", "condition_a"]).encode(),
        }
    )
    pq.write_table(table, path)


def test_mechanism_sensitivity_loader_preserves_declared_mask_order(tmp_path) -> None:
    path = tmp_path / "target_sensitivity.parquet"
    _write_mechanism_sensitivity(path)
    table, conditions = load_target_sensitivity_table(
        path,
        "class_aware",
        {"condition_a", "condition_b"},
    )
    assert conditions == ("condition_b", "condition_a")
    expanded = expand_target_sensitivity(table, 4, conditions)
    bev = expanded.loc[expanded["model"] == "bevfusion_mit"]
    assert bev.set_index("condition")["failure"].to_dict() == {
        "condition_b": True,
        "condition_a": False,
    }


def test_primary_alignment_rejects_identity_swap_with_same_failure_totals(
    tmp_path,
) -> None:
    path = tmp_path / "target_sensitivity.parquet"
    _write_mechanism_sensitivity(path)
    packed, conditions = load_target_sensitivity_table(
        path,
        "class_aware",
        {"condition_a", "condition_b"},
    )
    target = expand_target_sensitivity(packed, 4, conditions)
    first_model = target["model"] == "bevfusion_mit"
    first_condition = target["condition"] == "condition_b"
    second_model = target["model"] == "sparsefusion_r50"
    second_condition = target["condition"] == "condition_b"
    target.loc[first_model & first_condition, "failure"] = False
    target.loc[second_model & second_condition, "failure"] = True

    with pytest.raises(ValueError, match="failure identities"):
        validate_primary_target_alignment(target, packed, conditions)


def test_target_metadata_rejects_non_mechanism_producer(tmp_path) -> None:
    path = tmp_path / "target.parquet"
    table = pa.table({"failure": pa.array([False], type=pa.bool_())})
    table = table.replace_schema_metadata(
        {
            b"producer": b"analysis/reference_models.py",
            b"assignment_mode": b"class_aware",
            b"qualification_score_threshold": b"0.25",
            b"qualification_distance_m": b"2.0",
            b"qualification_assignment": (
                b"global_one_to_one_max_cardinality_then_min_center_distance"
            ),
        }
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="mechanism metadata"):
        validate_target_table_metadata(path, "class_aware")


def test_sensitivity_loader_rejects_non_mechanism_producer(tmp_path) -> None:
    path = tmp_path / "target_sensitivity.parquet"
    _write_mechanism_sensitivity(
        path,
        producer="analysis/reference_models.py",
    )
    with pytest.raises(ValueError, match="must be produced"):
        load_target_sensitivity_table(
            path,
            "class_aware",
            {"condition_a", "condition_b"},
        )


def test_sensitivity_loader_rejects_different_config_order(tmp_path) -> None:
    path = tmp_path / "target_sensitivity.parquet"
    reversed_order = [
        {"score_threshold": score, "match_distance_m": distance}
        for score, distance in reversed(SENSITIVITY_CONFIGS)
    ]
    _write_mechanism_sensitivity(path, config_order=reversed_order)
    with pytest.raises(ValueError, match="config_order"):
        load_target_sensitivity_table(
            path,
            "class_aware",
            {"condition_a", "condition_b"},
        )


def test_sensitivity_loader_has_no_cache_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Mechanism-produced"):
        load_target_sensitivity_table(
            tmp_path / "missing.parquet",
            "class_aware",
            {"condition_a"},
        )


def _transfer_fixture() -> pd.DataFrame:
    rows = []
    models = ("bevfusion_mit", "sparsefusion_r50", "deepinteraction_base")
    for log_index, log in enumerate(("l1", "l2", "l3")):
        for annotation_index, annotation in enumerate(("a1", "a2")):
            for model_index, model in enumerate(models):
                for condition_index, condition in enumerate(("c1", "c2")):
                    rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "log": log,
                            "scene": f"s{log_index}",
                            "sample_token": f"p{log_index}",
                            "annotation_token": f"{log}_{annotation}",
                            "class": "car",
                            "clean_score": (
                                0.1
                                + 0.2 * log_index
                                + 0.05 * annotation_index
                                + 0.01 * model_index
                            ),
                            "distance_m": 10.0,
                            "size_m3": 8.0,
                            "visibility": 4,
                            "failure": (
                                model_index + annotation_index + condition_index
                            )
                            % 3
                            == 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_aligned_transfer_counts_only_other_detector_failures() -> None:
    result = build_aligned_failure_transfer_table(_transfer_fixture())
    identity = result[
        (result["condition"] == "c1") & (result["annotation_token"] == "l1_a1")
    ]
    assert len(identity) == 3
    for row in identity.itertuples():
        expected = int(identity["failure"].sum()) - int(row.failure)
        assert row.source_failures == expected
    assert result["source_failures"].between(0, 2).all()


def test_aligned_transfer_rejects_condition_dependent_gt_covariates() -> None:
    targets = _transfer_fixture()
    targets.loc[targets.index[0], "distance_m"] += 1.0
    with pytest.raises(ValueError, match="GT covariates"):
        build_aligned_failure_transfer_table(targets)


def test_fold_source_feature_never_reads_held_detector_failure() -> None:
    aligned = build_aligned_failure_transfer_table(_transfer_fixture())
    held = "bevfusion_mit"
    before = fold_source_failure_rate(aligned, held)
    changed = aligned.copy()
    changed[f"failure__{held}"] = ~changed[f"failure__{held}"]
    after = fold_source_failure_rate(changed, held)
    assert np.array_equal(before, after)


def test_fold_clean_score_calibration_excludes_held_log_distribution() -> None:
    aligned = build_aligned_failure_transfer_table(_transfer_fixture())
    before = fold_clean_score_vulnerability(aligned, "l1")
    reference = aligned[
        ["model", "log", "annotation_token", "clean_score"]
    ].drop_duplicates(["model", "log", "annotation_token"])
    expected = np.full(len(aligned), np.nan)
    for model in sorted(aligned["model"].unique()):
        mask = aligned["model"].eq(model).to_numpy()
        calibration = np.sort(
            reference.loc[
                reference["model"].eq(model) & reference["log"].ne("l1"),
                "clean_score",
            ].to_numpy(float)
        )
        expected[mask] = 1.0 - (
            np.searchsorted(
                calibration,
                aligned.loc[mask, "clean_score"].to_numpy(float),
                side="right",
            )
            / len(calibration)
        )
    assert np.array_equal(before, expected)

    changed = aligned.copy()
    changed.loc[changed["log"] == "l1", "clean_score"] = 10_000.0
    after = fold_clean_score_vulnerability(changed, "l1")
    training_rows = aligned["log"].ne("l1").to_numpy()
    assert np.array_equal(before[training_rows], after[training_rows])


def test_fold_clean_score_calibration_rejects_nonfinite_scores() -> None:
    aligned = build_aligned_failure_transfer_table(_transfer_fixture())
    aligned.loc[aligned.index[0], "clean_score"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fold_clean_score_vulnerability(aligned, "l1")


def test_fold_clean_score_calibration_rejects_condition_drift() -> None:
    aligned = build_aligned_failure_transfer_table(_transfer_fixture())
    identity = aligned.loc[aligned.index[0], ["model", "log", "annotation_token"]]
    mask = (
        aligned["model"].eq(identity["model"])
        & aligned["log"].eq(identity["log"])
        & aligned["annotation_token"].eq(identity["annotation_token"])
    )
    changed_index = aligned.index[mask][-1]
    aligned.loc[changed_index, "clean_score"] += 0.01
    with pytest.raises(ValueError, match="invariant"):
        fold_clean_score_vulnerability(aligned, "l1")


def test_single_source_predictions_preserve_source_identity() -> None:
    class IdentityEstimator:
        @staticmethod
        def predict_proba(rows):
            probability = rows["source_failure_rate"].to_numpy(float)
            return np.column_stack((1.0 - probability, probability))

    frame = pd.DataFrame(
        {
            "failure__bevfusion_mit": [0, 1],
            "failure__sparsefusion_r50": [1, 0],
            "source_failure_rate": [0.5, 0.5],
        }
    )
    result = single_source_failure_probabilities(
        IdentityEstimator(),
        frame,
        ["source_failure_rate"],
        ["bevfusion_mit", "sparsefusion_r50"],
    )
    assert np.array_equal(result["bevfusion_mit"], np.asarray([0.0, 1.0]))
    assert np.array_equal(result["sparsefusion_r50"], np.asarray([1.0, 0.0]))
    assert np.array_equal(
        np.mean(np.stack(list(result.values())), axis=0),
        np.asarray([0.5, 0.5]),
    )


def test_double_holdout_excludes_both_target_groups() -> None:
    result = summarize_failure_evidence_transfer(_transfer_fixture())
    assert result["design"]["folds"] == 9
    assert len(result["folds"]) == 45
    assert (
        "1 - empirical CDF(clean_score)"
        in result["design"]["inverse_clean_score_percentile_calibration"]
    )
    assert not any("margin" in row["specification"].lower() for row in result["folds"])
    assert all(row["train_excludes_held_detector"] for row in result["folds"])
    assert all(row["train_excludes_held_log"] for row in result["folds"])
    assert all(
        row["clean_score_calibration_excludes_held_log"] for row in result["folds"]
    )
    assert all(
        row["source_feature_excludes_held_detector_failure"] for row in result["folds"]
    )
    assert len(result["summary"]["directed_source_to_target"]["pairs"]) == 6
    assert all(
        row["source_detector"] != row["target_detector"]
        for row in result["summary"]["directed_source_to_target"]["pairs"]
    )
    for comparison in (
        "vulnerability_source_rate_vs_vulnerability",
        "vulnerability_source_mean_single_vs_vulnerability",
    ):
        for row in result["summary"][comparison]["by_held_detector"]:
            assert len(row["leave_one_held_log_out_mean_delta_AUPRC_range"]) == 2
            assert len(row["leave_one_held_log_out_mean_delta_brier_range"]) == 2
    assert "held-log predictive differences within this corpus" in result["claim_boundary"]
    assert (
        result["condition_metric_support"]["total_detector_log_condition_cells"] == 18
    )
    assert (
        result["condition_metric_support"]["AUPRC_estimable_cells"]
        + result["condition_metric_support"]["zero_positive_cells"]
        + result["condition_metric_support"]["all_positive_cells"]
        == 18
    )
    assert result["condition_metric_support"]["total_target_rows"] == len(
        build_aligned_failure_transfer_table(_transfer_fixture())
    )
    assert result["condition_metric_support"]["total_positive_events"] == int(
        build_aligned_failure_transfer_table(_transfer_fixture())["failure"].sum()
    )
