import numpy as np
import pytest

from mobo_kit.candidate_pool import (
    CandidatePoolSamplingError,
    physical_rows_to_grid_indices,
    sample_discrete_candidate_pool,
)
from mobo_kit.design import InputSpec, build_design


def _design():
    return build_design(
        [
            InputSpec("a", 0.0, 4.0, 1.0),
            InputSpec("b", 10.0, 20.0, 5.0),
            InputSpec("c", -1.0, 1.0, 1.0),
        ]
    )


def test_exact_size_seeded_order_grid_membership_and_normalization():
    design = _design()
    first = sample_discrete_candidate_pool(design, 20, seed=12)
    repeat = sample_discrete_candidate_pool(design, 20, seed=12)
    different = sample_discrete_candidate_pool(design, 20, seed=13)
    assert first.size == 20
    assert np.array_equal(first.grid_indices, repeat.grid_indices)
    assert np.array_equal(first.X_phys, repeat.X_phys)
    assert np.array_equal(first.X_norm, repeat.X_norm)
    assert not np.array_equal(first.grid_indices, different.grid_indices)
    assert np.unique(first.grid_indices, axis=0).shape[0] == 20
    assert np.all(first.grid_indices >= 0)
    assert np.all(first.grid_indices < np.array([5, 3, 3]))
    assert np.all((first.X_norm >= 0) & (first.X_norm <= 1))
    assert np.array_equal(
        physical_rows_to_grid_indices(first.X_phys, design), first.grid_indices
    )


def test_observed_pending_and_explicit_avoid_are_excluded():
    design = _design()
    exclusions = np.array([[0.0, 10.0, -1.0], [1.0, 15.0, 0.0], [2.0, 20.0, 1.0]])
    pool = sample_discrete_candidate_pool(
        design,
        25,
        seed=4,
        observed_phys=exclusions[:1],
        pending_phys=exclusions[1:2],
        avoid_phys=exclusions[2:],
    )
    excluded_indices = physical_rows_to_grid_indices(exclusions, design)
    pool_set = {tuple(row) for row in pool.grid_indices}
    assert not any(tuple(row) in pool_set for row in excluded_indices)


def test_constraints_run_in_physical_space_and_are_fail_closed():
    design = _design()
    calls = []

    def require_even_first_input(X_phys, supplied_design):
        calls.append(X_phys.copy())
        assert supplied_design is design
        return (X_phys[:, 0] % 2) == 0

    pool = sample_discrete_candidate_pool(
        design,
        15,
        seed=3,
        row_constraints=[require_even_first_input],
        max_draws=500,
    )
    assert calls
    assert np.all(pool.X_phys[:, 0] % 2 == 0)
    assert pool.rejected_constraint > 0


def test_impossible_request_raises_structured_error():
    design = build_design([InputSpec("x", 0, 1, 1)])
    with pytest.raises(CandidatePoolSamplingError) as captured:
        sample_discrete_candidate_pool(
            design, 2, seed=1, observed_phys=np.array([[0.0]])
        )
    error = captured.value
    assert error.requested == 2
    assert error.accepted == 0
    assert "exceeds" in error.reason


def test_max_draws_failure_reports_rejections():
    design = build_design([InputSpec("x", 0, 4, 1)])

    def reject_all(X_phys, supplied_design):
        del supplied_design
        return np.zeros(X_phys.shape[0], dtype=bool)

    with pytest.raises(CandidatePoolSamplingError) as captured:
        sample_discrete_candidate_pool(
            design,
            1,
            seed=2,
            row_constraints=[reject_all],
            max_draws=5,
        )
    assert captured.value.draws == 5
    assert captured.value.rejected_constraint > 0


def test_draw_statistics_on_known_two_point_grid():
    design = build_design([InputSpec("x", 0, 1, 1)])
    pool = sample_discrete_candidate_pool(design, 2, seed=0, max_draws=10)
    # NumPy Generator seed 0 draws 1, 1, 1, 0 for the first four integers.
    assert pool.grid_indices[:, 0].tolist() == [1, 0]
    assert pool.draws == 4
    assert pool.rejected_duplicate == 2
    assert pool.rejected_avoid == 0
    assert pool.rejected_constraint == 0

    excluded = sample_discrete_candidate_pool(
        design,
        1,
        seed=0,
        observed_phys=np.array([[1.0]]),
        max_draws=10,
    )
    assert excluded.grid_indices[:, 0].tolist() == [0]
    assert excluded.draws == 4
    assert excluded.rejected_duplicate == 2
    assert excluded.rejected_avoid == 1
    assert excluded.rejected_constraint == 0


def test_off_grid_exclusions_fail_instead_of_using_fuzzy_matching():
    with pytest.raises(ValueError, match="off-grid"):
        sample_discrete_candidate_pool(
            _design(), 2, seed=1, avoid_phys=np.array([[0.25, 10.0, 0.0]])
        )
    with pytest.raises(ValueError, match="off-grid"):
        physical_rows_to_grid_indices(np.array([[1e-9, 10.0, 0.0]]), _design())


def test_sampler_never_calls_cartesian_product_allocators(monkeypatch):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("full Cartesian allocation was attempted")

    monkeypatch.setattr(np, "meshgrid", forbidden)
    monkeypatch.setattr(np, "indices", forbidden)
    pool = sample_discrete_candidate_pool(_design(), 10, seed=9)
    assert pool.size == 10
