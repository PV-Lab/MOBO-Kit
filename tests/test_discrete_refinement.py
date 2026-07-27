import numpy as np

from mobo_kit.candidate_pool import CandidatePool
from mobo_kit.design import InputSpec, build_design
from mobo_kit.discrete_refinement import (
    CachedGridScorer,
    RefinementConfig,
    grid_indices_to_physical_and_normalized,
    propose_refined_discrete_batch,
    refine_discrete_acquisition_anchors,
)


def _design():
    return build_design(
        [
            InputSpec("x", 0, 4, 1),
            InputSpec("y", 0, 3, 1),
        ]
    )


def _pool(design) -> CandidatePool:
    indices = np.asarray([[x, y] for x in range(5) for y in range(4)], dtype=np.int64)
    physical, normalized = grid_indices_to_physical_and_normalized(indices, design)
    return CandidatePool(
        grid_indices=indices,
        X_phys=physical,
        X_norm=normalized,
        seed=73,
        draws=len(indices),
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )


def test_refinement_reaches_coordinate_local_optimum_and_is_monotone() -> None:
    design = _design()

    def score(rows: np.ndarray) -> np.ndarray:
        return 100.0 - (rows[:, 0] - 3) ** 2 - 2.0 * (rows[:, 1] - 2) ** 2

    refined, trace = refine_discrete_acquisition_anchors(
        design,
        np.asarray([[0, 0], [4, 3]], dtype=np.int64),
        score,
        config=RefinementConfig(
            anchors_per_selection_step=2,
            max_sweeps=10,
            improvement_tolerance=1e-12,
            radius=None,
            min_batch_distance=0.0,
        ),
        selection_step=1,
    )

    assert {item.refined_grid_index for item in refined} == {(3, 2)}
    assert all(item.termination_reason == "no_improvement" for item in refined)
    assert all(row.score_after >= row.score_before for row in trace)
    assert any(row.accepted_move for row in trace)
    assert all(np.isfinite(row.base_score_before) for row in trace)
    assert all(np.isfinite(row.base_score_after) for row in trace)
    assert all(np.isfinite(row.penalized_log_score_before) for row in trace)
    assert all(np.isfinite(row.penalized_log_score_after) for row in trace)
    assert all(row.termination_reason in {None, "no_improvement"} for row in trace)


def test_refinement_tie_break_and_trace_are_deterministic() -> None:
    design = _design()
    config = RefinementConfig(
        anchors_per_selection_step=1,
        max_sweeps=3,
        radius=None,
        min_batch_distance=0.0,
    )

    def flat(rows: np.ndarray) -> np.ndarray:
        return np.ones(rows.shape[0])

    first = refine_discrete_acquisition_anchors(
        design,
        np.asarray([[2, 2]], dtype=np.int64),
        flat,
        config=config,
        selection_step=1,
    )
    second = refine_discrete_acquisition_anchors(
        design,
        np.asarray([[2, 2]], dtype=np.int64),
        flat,
        config=config,
        selection_step=1,
    )

    assert first == second
    assert first[0][0].refined_grid_index == (2, 2)
    assert first[0][0].termination_reason == "no_improvement"


def test_grid_normalization_matches_physical_bound_canonicalization_bitwise() -> None:
    design = build_design([InputSpec("decimal_axis", 1.0, 2.0, 0.05, decimals=2)])
    indices = np.arange(design.var_array[0].size, dtype=np.int64)[:, None]

    physical, normalized = grid_indices_to_physical_and_normalized(indices, design)
    expected = (physical - design.lowers) / (design.uppers - design.lowers)
    index_fraction = indices / float(design.var_array[0].size - 1)

    np.testing.assert_array_equal(normalized, expected)
    assert np.any(normalized != index_fraction)


def test_refined_batch_enforces_observed_exclusion_and_hard_distance() -> None:
    design = _design()
    pool = _pool(design)
    observed_grid = np.asarray([[4, 3]], dtype=np.int64)
    _, observed_norm = grid_indices_to_physical_and_normalized(observed_grid, design)

    def score(rows: np.ndarray) -> np.ndarray:
        # The forbidden observed point is the nominal maximum.
        return 100.0 + 10.0 * rows[:, 0] + rows[:, 1]

    result = propose_refined_discrete_batch(
        pool,
        design,
        score,
        q=2,
        config=RefinementConfig(
            anchors_per_selection_step=6,
            max_sweeps=5,
            radius=None,
            min_batch_distance=0.75,
        ),
        observed_grid_indices=observed_grid,
        observed_norm=observed_norm,
    )

    assert result.grid_indices.shape == (2, 2)
    assert not np.any(np.all(result.grid_indices == observed_grid[0], axis=1))
    assert np.unique(result.grid_indices, axis=0).shape[0] == 2
    assert np.linalg.norm(result.X_norm[0] - result.X_norm[1]) >= 0.75
    assert all(value > 0 for value in result.base_scores)
    assert result.distinct_converged_optima >= 2


def test_refined_batch_forwards_pending_rows_through_avoid_grid_indices() -> None:
    design = _design()
    pool = _pool(design)
    pending_grid = np.asarray([[4, 3]], dtype=np.int64)

    def score(rows: np.ndarray) -> np.ndarray:
        return 100.0 + 10.0 * rows[:, 0] + rows[:, 1]

    result = propose_refined_discrete_batch(
        pool,
        design,
        score,
        q=1,
        config=RefinementConfig(
            anchors_per_selection_step=6,
            max_sweeps=5,
            radius=None,
            min_batch_distance=0.0,
        ),
        avoid_grid_indices=pending_grid,
    )

    assert result.grid_indices.shape == (1, 2)
    assert not np.array_equal(result.grid_indices[0], pending_grid[0])


def test_refinement_reports_max_sweeps_without_off_grid_moves() -> None:
    design = _design()

    def score(rows: np.ndarray) -> np.ndarray:
        return 10.0 + rows[:, 0] + rows[:, 0] * rows[:, 1]

    refined, trace = refine_discrete_acquisition_anchors(
        design,
        np.asarray([[0, 0]], dtype=np.int64),
        score,
        config=RefinementConfig(
            anchors_per_selection_step=1,
            max_sweeps=1,
            radius=None,
            min_batch_distance=0.0,
        ),
        selection_step=1,
    )

    assert refined[0].termination_reason == "max_sweeps"
    assert trace[-1].termination_reason == "max_sweeps"
    assert np.all(np.asarray(refined[0].refined_grid_index) >= 0)
    assert refined[0].refined_grid_index[0] < 5
    assert refined[0].refined_grid_index[1] < 4


def test_cached_grid_scorer_deduplicates_vectorized_requests() -> None:
    calls: list[np.ndarray] = []

    def score(rows: np.ndarray) -> np.ndarray:
        calls.append(rows.copy())
        return rows.sum(axis=1).astype(float)

    cached = CachedGridScorer(score, dimension=2)
    requested = np.asarray([[1, 2], [1, 2], [2, 3]], dtype=np.int64)
    np.testing.assert_array_equal(cached(requested), [3.0, 3.0, 5.0])
    np.testing.assert_array_equal(cached(requested[::-1]), [5.0, 3.0, 3.0])

    assert cached.cache_size == 2
    assert len(calls) == 1
    assert calls[0].shape == (2, 2)


def test_later_steps_readd_eligible_previously_refined_optima() -> None:
    design = build_design([InputSpec("x", 0, 2, 1), InputSpec("y", 0, 2, 1)])
    pool_indices = np.asarray([[1, 0], [1, 2]], dtype=np.int64)
    physical, normalized = grid_indices_to_physical_and_normalized(pool_indices, design)
    pool = CandidatePool(
        grid_indices=pool_indices,
        X_phys=physical,
        X_norm=normalized,
        seed=73,
        draws=2,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )
    table = np.ones((3, 3), dtype=float)
    table[0, 0] = 9.0
    table[1, 0] = 5.0
    table[1, 2] = 8.0
    table[2, 2] = 10.0

    def score(rows: np.ndarray) -> np.ndarray:
        return table[rows[:, 0], rows[:, 1]]

    result = propose_refined_discrete_batch(
        pool,
        design,
        score,
        q=2,
        config=RefinementConfig(
            anchors_per_selection_step=2,
            max_sweeps=5,
            radius=None,
            min_batch_distance=0.3,
        ),
    )

    assert any(
        anchor.selection_step == 2 and anchor.anchor_pool_index is None
        for anchor in result.anchors
    )
    assert all(anchor.accepted_move_count >= 0 for anchor in result.anchors)
    assert all(
        isinstance(anchor.changed_dimensions, tuple) for anchor in result.anchors
    )
