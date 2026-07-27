import numpy as np
import pytest

from mobo_kit.batch_selection import (
    BaseScoreResult,
    LocalPenalizationConfig,
    UndersizedBatchError,
    select_local_penalized_batch,
    soft_local_penalty,
)
from mobo_kit.candidate_pool import CandidatePool


def _pool(values):
    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    grid_indices = np.zeros(X.shape, dtype=int)
    grid_indices[:, 0] = np.arange(X.shape[0])
    return CandidatePool(
        grid_indices=grid_indices,
        X_phys=X.copy(),
        X_norm=X.copy(),
        seed=1,
        draws=X.shape[0],
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )


def _static_callback(scores, calls=None):
    scores = np.asarray(scores, dtype=float)

    def callback(remaining, selected):
        if calls is not None:
            calls.append((remaining.copy(), selected.copy()))
        return BaseScoreResult(
            base_log_score=scores[remaining],
            base_score=np.exp(scores[remaining]),
            diagnostics={"selected_before": selected.copy()},
        )

    return callback


def test_soft_penalty_zero_and_far_distance_limits():
    factors, logs = soft_local_penalty(np.array([0.0, 10.0]), radius=0.2, epsilon=1e-9)
    assert factors[0] == pytest.approx(0.0)
    assert logs[0] == pytest.approx(np.log(1e-9))
    assert factors[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="radius"):
        soft_local_penalty(np.array([1.0]), radius=0)
    with pytest.raises(ValueError, match="non-boolean"):
        LocalPenalizationConfig(radius=True, min_batch_distance=0.0)


def test_stable_tie_break_and_callback_receives_selected_indices():
    calls = []
    result = select_local_penalized_batch(
        _pool([0.0, 0.5, 1.0]),
        2,
        _static_callback([0.0, 0.0, 0.0], calls),
        LocalPenalizationConfig(radius=0.1, min_batch_distance=0.0),
    )
    assert result.selected_pool_indices.tolist() == [0, 2]
    assert calls[0][1].tolist() == []
    assert calls[1][1].tolist() == [0]
    assert [step.order for step in result.steps] == [1, 2]


def test_local_penalty_increases_diversity_over_unpenalized_top_q():
    pool = _pool([0.0, 0.01, 0.02, 0.6, 1.0])
    base_logs = np.log([1.0, 0.99, 0.98, 0.8, 0.7])
    unpenalized_top = pool.X_norm[np.argsort(-base_logs, kind="stable")[:3], 0]
    result = select_local_penalized_batch(
        pool,
        3,
        _static_callback(base_logs),
        LocalPenalizationConfig(radius=0.2, min_batch_distance=0.0),
    )
    selected = np.sort(result.X_norm[:, 0])
    assert np.min(np.diff(selected)) > np.min(np.diff(np.sort(unpenalized_top)))
    assert result.steps[1].penalty_factor < 1.0


def test_none_radius_disables_only_the_soft_penalty():
    pool = _pool([0.0, 0.01, 0.5, 1.0])
    result = select_local_penalized_batch(
        pool,
        3,
        _static_callback([0.0, -0.1, -0.2, -0.3]),
        LocalPenalizationConfig(radius=None, min_batch_distance=0.0),
    )
    assert result.selected_pool_indices.tolist() == [0, 1, 2]
    assert result.distance_diagnostics["radius"] is None
    assert all(step.penalty_factor == pytest.approx(1.0) for step in result.steps)
    assert all(step.log_penalty == pytest.approx(0.0) for step in result.steps)
    assert all(
        step.penalized_log_score == pytest.approx(step.base_log_score)
        for step in result.steps
    )


def test_none_radius_preserves_hard_batch_and_observed_rules():
    result = select_local_penalized_batch(
        _pool([0.0, 0.1, 0.5, 0.9]),
        2,
        _static_callback([0.0, -0.01, -0.2, -0.3]),
        LocalPenalizationConfig(
            radius=None,
            min_batch_distance=0.4,
            min_observed_distance=0.15,
        ),
        observed_pending_norm=np.array([[0.9]]),
    )
    # The second-highest score is too close to the first selection, and the
    # final point is an exact observed recipe. The hard rules therefore choose
    # the third-ranked point even though no soft penalty is active.
    assert result.selected_pool_indices.tolist() == [0, 2]
    assert result.distance_diagnostics["minimum_within_batch_distance"] >= 0.4
    assert all(step.penalty_factor == pytest.approx(1.0) for step in result.steps)


def test_hard_batch_and_observed_distances_are_enforced():
    pool = _pool([0.0, 0.1, 0.3, 0.55, 0.9])
    result = select_local_penalized_batch(
        pool,
        3,
        _static_callback([0.0, -0.1, -0.2, -0.3, -0.4]),
        LocalPenalizationConfig(
            radius=0.1, min_batch_distance=0.25, min_observed_distance=0.15
        ),
        observed_pending_norm=np.array([[0.3]]),
    )
    matrix = result.distance_diagnostics["pairwise_distance_matrix"]
    triangle = matrix[np.triu_indices(3, k=1)]
    assert np.all(triangle >= 0.25 - 1e-12)
    assert np.all(np.abs(result.X_norm[:, 0] - 0.3) >= 0.15 - 1e-12)


def test_exact_observed_duplicate_is_excluded_even_with_zero_threshold():
    result = select_local_penalized_batch(
        _pool([0.0, 0.5, 1.0]),
        1,
        _static_callback([1.0, 0.0, -1.0]),
        LocalPenalizationConfig(radius=0.1, min_batch_distance=0.0),
        observed_pending_norm=np.array([[0.0]]),
    )
    assert result.selected_pool_indices.tolist() == [1]


def test_impossible_spacing_and_all_ineligible_fail_structurally():
    pool = _pool([0.0, 0.1, 0.2])
    with pytest.raises(UndersizedBatchError) as captured:
        select_local_penalized_batch(
            pool,
            2,
            _static_callback([0.0, -0.1, -0.2]),
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0.5),
        )
    assert captured.value.selected_size == 1
    assert captured.value.requested_size == 2

    with pytest.raises(UndersizedBatchError) as all_zero:
        select_local_penalized_batch(
            pool,
            1,
            _static_callback([-np.inf, -np.inf, -np.inf]),
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0.0),
        )
    assert all_zero.value.selected_size == 0
    assert all_zero.value.remaining_candidate_count == 0
    assert all_zero.value.hard_valid_candidate_count == 3


def test_log_epsilon_never_relaxes_hard_distance():
    pool = _pool([0.0, 0.4999999999995])
    with pytest.raises(UndersizedBatchError):
        select_local_penalized_batch(
            pool,
            2,
            _static_callback([0.0, -0.1]),
            LocalPenalizationConfig(
                radius=0.1,
                min_batch_distance=0.5,
                epsilon=0.5,
            ),
        )


def test_dimension_weight_validation_and_effect():
    pool = _pool([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
    config = LocalPenalizationConfig(
        radius=0.1,
        min_batch_distance=0,
        dimension_weights=np.array([4.0, 1.0]),
    )
    result = select_local_penalized_batch(
        pool, 2, _static_callback([0.0, 0.0, 0.0]), config
    )
    assert result.selected_pool_indices.tolist() == [0, 1]
    with pytest.raises(ValueError, match="shape"):
        select_local_penalized_batch(
            pool,
            1,
            _static_callback([0.0, 0.0, 0.0]),
            LocalPenalizationConfig(
                radius=0.1,
                min_batch_distance=0,
                dimension_weights=np.ones(3),
            ),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        LocalPenalizationConfig(
            radius=0.1,
            min_batch_distance=0,
            dimension_weights=np.array([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="non-boolean"):
        LocalPenalizationConfig(
            radius=0.1,
            min_batch_distance=0,
            dimension_weights=np.array([True, True]),
        )


def test_duplicate_pool_and_invalid_scores_are_rejected():
    duplicate_pool = CandidatePool(
        grid_indices=np.array([[0], [0]]),
        X_phys=np.array([[0.0], [0.0]]),
        X_norm=np.array([[0.0], [0.0]]),
        seed=1,
        draws=2,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )
    with pytest.raises(ValueError, match="duplicate"):
        select_local_penalized_batch(
            duplicate_pool,
            1,
            _static_callback([0.0, 0.0]),
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0),
        )

    duplicate_coordinates = CandidatePool(
        grid_indices=np.array([[0], [1]]),
        X_phys=np.array([[0.0], [0.0]]),
        X_norm=np.array([[0.0], [0.0]]),
        seed=1,
        draws=2,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )
    with pytest.raises(ValueError, match="duplicate normalized"):
        select_local_penalized_batch(
            duplicate_coordinates,
            1,
            _static_callback([0.0, 0.0]),
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0),
        )

    def bad_callback(remaining, selected):
        del selected
        return BaseScoreResult(np.full(remaining.size, np.nan))

    with pytest.raises(ValueError, match="NaN"):
        select_local_penalized_batch(
            _pool([0.0, 1.0]),
            1,
            bad_callback,
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0),
        )


def test_selector_rejects_out_of_range_references_and_nonfinite_physical_rows():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        select_local_penalized_batch(
            _pool([0.0, 1.0]),
            1,
            _static_callback([0.0, -1.0]),
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0),
            observed_pending_norm=np.array([[2.0]]),
        )
    invalid = _pool([0.0, 1.0])
    object.__setattr__(invalid, "X_phys", np.array([[np.nan], [1.0]]))
    with pytest.raises(ValueError, match="physical rows"):
        select_local_penalized_batch(
            invalid,
            1,
            _static_callback([0.0, -1.0]),
            LocalPenalizationConfig(radius=0.1, min_batch_distance=0),
        )
