from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mobo_kit.observation_influence import (
    InfluenceRunInput,
    acquisition_rank_metrics,
    hyperparameter_displacement,
    pareto_membership_change,
    prediction_change_metrics,
    run_observation_influence_study,
)


POOL_HASH = "a" * 64
OBJECTIVES = ("uniformity", "optoelectronic", "thickness")
SCALES = (1.0, 2.0, 1.0)


def _run(
    label: str,
    omitted_sample_id: int | None,
    *,
    selected: np.ndarray,
    scores: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    pareto: tuple[int, ...],
    hyperparameters: dict[str, float],
    pool_hash: str = POOL_HASH,
) -> InfluenceRunInput:
    return InfluenceRunInput(
        run_label=label,
        omitted_sample_id=omitted_sample_id,
        common_pool_sha256=pool_hash,
        selected_X_norm=selected,
        pool_acquisition_scores=scores,
        prediction_mean_at_full_candidates=means,
        prediction_std_at_full_candidates=stds,
        pareto_sample_ids=pareto,
        hyperparameters=hyperparameters,
    )


def _study_inputs() -> tuple[InfluenceRunInput, tuple[InfluenceRunInput, ...]]:
    selected = np.array([[0.0, 0.0], [1.0, 1.0]])
    means = np.array([[0.4, -8.0, 0.7], [0.6, -7.5, 0.8]])
    stds = np.full((2, 3), 0.1)
    full = _run(
        "full",
        None,
        selected=selected,
        scores=np.array([4.0, 3.0, 2.0, 1.0]),
        means=means,
        stds=stds,
        pareto=(1, 2),
        hyperparameters={"noise": 0.1, "lengthscale": 0.5},
    )
    omit_1 = _run(
        "omit-1",
        1,
        selected=np.array([[0.7, 0.0], [0.2, 1.0]]),
        scores=np.array([1.0, 2.0, 3.0, 4.0]),
        means=means + np.array([[0.4, 1.0, 0.3], [0.4, 1.0, 0.3]]),
        stds=stds + 0.2,
        pareto=(2,),
        hyperparameters={"noise": 0.4, "lengthscale": 1.5},
    )
    omit_2 = _run(
        "omit-2",
        2,
        selected=np.array([[0.05, 0.0], [1.0, 0.95]]),
        scores=np.array([4.0, 2.9, 2.1, 1.0]),
        means=means + 0.02,
        stds=stds + 0.01,
        pareto=(1,),
        hyperparameters={"noise": 0.11, "lengthscale": 0.52},
    )
    omit_3 = _run(
        "omit-3",
        3,
        selected=selected.copy(),
        scores=np.array([4.0, 3.0, 2.0, 1.0]),
        means=means.copy(),
        stds=stds.copy(),
        pareto=(1, 2),
        hyperparameters={"noise": 0.1, "lengthscale": 0.5},
    )
    return full, (omit_1, omit_2, omit_3)


def test_run_input_defensively_copies_arrays_and_hyperparameters() -> None:
    selected = np.array([[0.0, 0.0], [1.0, 1.0]])
    scores = np.array([2.0, 1.0])
    means = np.ones((2, 3))
    stds = np.full((2, 3), 0.2)
    hyperparameters = {"noise": 0.1}
    run = _run(
        "full",
        None,
        selected=selected,
        scores=scores,
        means=means,
        stds=stds,
        pareto=(1,),
        hyperparameters=hyperparameters,
    )
    selected[0, 0] = 0.8
    scores[0] = 0.0
    means[0, 0] = 9.0
    stds[0, 0] = 9.0
    hyperparameters["noise"] = 9.0

    assert run.selected_X_norm[0, 0] == 0.0
    assert run.pool_acquisition_scores[0] == 2.0
    assert run.prediction_mean_at_full_candidates[0, 0] == 1.0
    assert run.prediction_std_at_full_candidates[0, 0] == 0.2
    assert run.hyperparameters["noise"] == 0.1
    with pytest.raises(ValueError):
        run.selected_X_norm[0, 0] = 1.0
    with pytest.raises(TypeError):
        run.hyperparameters["noise"] = 1.0


def test_prediction_change_metrics_are_objective_scaled_and_long_form() -> None:
    full_mean = np.array([[0.0, 0.0], [1.0, 2.0]])
    omitted_mean = np.array([[0.1, 0.4], [0.8, 2.4]])
    full_std = np.full((2, 2), 0.1)
    omitted_std = np.array([[0.2, 0.3], [0.1, 0.5]])
    metrics, rows = prediction_change_metrics(
        full_mean,
        full_std,
        omitted_mean,
        omitted_std,
        omitted_sample_id=7,
        objective_names=("a", "b"),
        objective_scales=(1.0, 2.0),
    )

    assert len(rows) == 4
    assert rows[1].mean_delta == pytest.approx(0.4)
    assert rows[1].normalized_absolute_mean_delta == pytest.approx(0.2)
    assert metrics.mean_absolute_mean_change == pytest.approx(0.275)
    assert metrics.maximum_absolute_mean_change == pytest.approx(0.4)
    assert metrics.mean_normalized_absolute_mean_change == pytest.approx(0.175)
    assert metrics.mean_normalized_absolute_std_change == pytest.approx(0.1)
    assert metrics.component_value == pytest.approx(0.1375)


def test_acquisition_rank_and_top_k_changes_match_hand_values() -> None:
    metrics = acquisition_rank_metrics(
        np.array([4.0, 3.0, 2.0, 1.0]),
        np.array([4.0, 2.0, 3.0, 1.0]),
        top_k=2,
    )

    assert metrics.pool_size == 4
    assert metrics.top_k == 2
    assert metrics.top_k_overlap_count == 1
    assert metrics.top_k_jaccard == pytest.approx(1.0 / 3.0)
    assert metrics.mean_absolute_rank_change == pytest.approx(0.5)
    assert metrics.maximum_absolute_rank_change == pytest.approx(1.0)
    assert metrics.normalized_mean_absolute_rank_change == pytest.approx(1.0 / 6.0)
    assert metrics.spearman_rank_correlation == pytest.approx(0.8)
    assert metrics.component_value == pytest.approx(5.0 / 12.0)

    identical = acquisition_rank_metrics(np.ones(3), np.ones(3), top_k=10)
    assert identical.top_k == 3
    assert identical.spearman_rank_correlation == 1.0
    assert identical.component_value == 0.0


def test_pareto_and_hyperparameter_changes_are_transparent() -> None:
    pareto = pareto_membership_change((1, 2), (2, 3))
    assert pareto.intersection_count == 1
    assert pareto.jaccard == pytest.approx(1.0 / 3.0)
    assert pareto.added_sample_ids == (3,)
    assert pareto.removed_sample_ids == (1,)
    assert pareto.component_value == pytest.approx(2.0 / 3.0)

    displacement = hyperparameter_displacement(
        {"lengthscale": 2.0, "noise": 1.0},
        {"lengthscale": 1.0, "noise": 2.0},
    )
    assert displacement.parameter_count == 2
    assert displacement.mean_absolute_log_ratio == pytest.approx(np.log(2.0))
    assert displacement.maximum_absolute_log_ratio == pytest.approx(np.log(2.0))
    assert dict(displacement.absolute_log_ratio_by_parameter) == {
        "lengthscale": pytest.approx(np.log(2.0)),
        "noise": pytest.approx(np.log(2.0)),
    }


def test_full_study_ranks_sample_1_and_preserves_fixed_policy() -> None:
    full, omissions = _study_inputs()
    row_roles = {1: "control", 2: "r0_lhs", 3: "r0_lhs"}
    include_policy = {1: True, 2: True, 3: True}
    row_roles_before = dict(row_roles)
    include_before = dict(include_policy)

    result = run_observation_influence_study(
        full,
        omissions,
        expected_common_pool_sha256=POOL_HASH,
        objective_names=OBJECTIVES,
        objective_scales=SCALES,
        row_roles=row_roles,
        primary_include_policy=include_policy,
        regional_thresholds=(0.10, 0.15, 0.20),
        top_k=2,
    )

    assert row_roles == row_roles_before
    assert include_policy == include_before
    assert dict(result.row_roles) == row_roles_before
    assert dict(result.primary_include_policy) == include_before
    with pytest.raises(TypeError):
        result.row_roles[1] = "changed"
    sample_1 = result.rank_for_sample(1)
    assert sample_1.influence_rank == 1
    assert sample_1.influence_percentile == 100.0
    assert result.sample_1_rank() == (1, 100.0)
    assert sample_1.omitted_row_role == "control"
    assert sample_1.omitted_primary_include_in_model is True
    assert len(result.prediction_changes) == len(omissions) * 2 * 3
    assert set(sample_1.raw_components) == {
        "batch_displacement",
        "prediction_change",
        "acquisition_change",
        "pareto_change",
        "hyperparameter_change",
    }
    assert set(sample_1.normalized_components) == set(sample_1.raw_components)
    assert sum(result.component_weights.values()) == pytest.approx(1.0)

    summary = result.summary_frame()
    assert summary.shape[0] == 3
    assert summary.iloc[0]["omitted_sample_id"] == 1
    assert summary.iloc[0]["influence_rank"] == 1
    assert {
        "regional_matches_within_0_10",
        "regional_matches_within_0_15",
        "regional_matches_within_0_20",
        "batch_displacement_raw_component",
        "batch_displacement_normalized_component",
        "composite_influence_score",
    } <= set(summary.columns)
    assert json.loads(summary.iloc[0]["pareto_removed_sample_ids"]) == [1]
    changes = result.prediction_changes_frame()
    assert changes.shape[0] == 18
    assert set(changes["objective_name"]) == set(OBJECTIVES)


def test_study_is_deterministic_under_omission_input_reordering() -> None:
    full, omissions = _study_inputs()
    kwargs = {
        "expected_common_pool_sha256": POOL_HASH,
        "objective_names": OBJECTIVES,
        "objective_scales": SCALES,
        "row_roles": {1: "control", 2: "r0_lhs", 3: "r0_lhs"},
        "primary_include_policy": {1: True, 2: True, 3: True},
        "top_k": 2,
    }
    first = run_observation_influence_study(full, omissions, **kwargs)
    reordered = run_observation_influence_study(
        full, tuple(reversed(omissions)), **kwargs
    )

    pd.testing.assert_frame_equal(first.summary_frame(), reordered.summary_frame())
    pd.testing.assert_frame_equal(
        first.prediction_changes_frame(), reordered.prediction_changes_frame()
    )


def test_study_rejects_pool_hash_shape_pareto_and_coverage_errors() -> None:
    full, omissions = _study_inputs()
    bad_hash_run = _run(
        "omit-1",
        1,
        selected=omissions[0].selected_X_norm,
        scores=omissions[0].pool_acquisition_scores,
        means=omissions[0].prediction_mean_at_full_candidates,
        stds=omissions[0].prediction_std_at_full_candidates,
        pareto=(2,),
        hyperparameters=dict(omissions[0].hyperparameters),
        pool_hash="b" * 64,
    )
    base_kwargs = {
        "expected_common_pool_sha256": POOL_HASH,
        "objective_names": OBJECTIVES,
        "objective_scales": SCALES,
        "row_roles": {1: "control", 2: "r0_lhs", 3: "r0_lhs"},
        "primary_include_policy": {1: True, 2: True, 3: True},
    }
    with pytest.raises(ValueError, match="verified common pool"):
        run_observation_influence_study(
            full, (bad_hash_run, omissions[1], omissions[2]), **base_kwargs
        )
    with pytest.raises(ValueError, match="cover every"):
        run_observation_influence_study(full, omissions[:2], **base_kwargs)

    invalid_pareto = _run(
        "omit-1",
        1,
        selected=omissions[0].selected_X_norm,
        scores=omissions[0].pool_acquisition_scores,
        means=omissions[0].prediction_mean_at_full_candidates,
        stds=omissions[0].prediction_std_at_full_candidates,
        pareto=(1, 2),
        hyperparameters=dict(omissions[0].hyperparameters),
    )
    with pytest.raises(ValueError, match="still lists"):
        run_observation_influence_study(
            full, (invalid_pareto, omissions[1], omissions[2]), **base_kwargs
        )
