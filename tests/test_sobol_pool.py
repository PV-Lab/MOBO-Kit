from __future__ import annotations

import itertools

import numpy as np
import pytest

from mobo_kit.candidate_pool import (
    CandidatePoolSamplingError,
    physical_rows_to_grid_indices,
)
from mobo_kit.design import InputSpec, build_design
from mobo_kit.sobol_pool import (
    build_nested_sobol_discrete_pool,
    hash_grid_index_prefix,
    map_unit_points_to_grid_indices,
)


def _design():
    return build_design(
        [
            InputSpec("a", 0.0, 8.0, 1.0),
            InputSpec("b", 10.0, 22.0, 2.0),
            InputSpec("c", -2.0, 2.0, 1.0),
        ]
    )


def test_unit_points_map_by_floor_to_exact_grid_indices():
    design = build_design(
        [InputSpec("x", 0.0, 3.0, 1.0), InputSpec("y", 10.0, 20.0, 10.0)]
    )
    points = np.array(
        [
            [0.0, 0.0],
            [0.249999, 0.499999],
            [0.25, 0.5],
            [np.nextafter(1.0, 0.0), np.nextafter(1.0, 0.0)],
        ]
    )
    np.testing.assert_array_equal(
        map_unit_points_to_grid_indices(points, design),
        np.array([[0, 0], [0, 0], [1, 1], [3, 1]]),
    )


@pytest.mark.parametrize(
    "points, message",
    [
        (np.array([[1.0, 0.0]]), "half-open"),
        (np.array([[-1e-12, 0.0]]), "half-open"),
        (np.array([[np.nan, 0.0]]), "finite"),
        (np.array([0.2, 0.3]), "shape"),
    ],
)
def test_unit_point_mapping_fails_closed(points, message):
    design = build_design(
        [InputSpec("x", 0.0, 3.0, 1.0), InputSpec("y", 0.0, 1.0, 1.0)]
    )
    with pytest.raises(ValueError, match=message):
        map_unit_points_to_grid_indices(points, design)


def test_same_scramble_is_deterministic_nested_and_has_locked_prefix_hashes():
    sizes = (8, 16, 32, 64)
    first = build_nested_sobol_discrete_pool(_design(), sizes, scramble_seed=73)
    repeat = build_nested_sobol_discrete_pool(
        _design(), tuple(reversed(sizes)), scramble_seed=73
    )
    expected_hashes = {
        8: "E36D0663991DFBF5A53C278C3F78DD66AB7D5245B8B7306D7422960F506F0361",
        16: "9E9AC7931E14CB3E4BE992D4B17D0670BEF0E6DB6AF467E059CE39A89D8A55F6",
        32: "79DBFBEEC3B6C7D1AA2E2D685E6DF19D5CA546BAD84F575A376FC83305B6A65B",
        64: "22AFBA550FA4A2F39104A9B565932225E374C0B675ABB4685FB19A38D6A748B7",
    }

    assert first.accepted_sizes == sizes
    assert first.accepted_count == 64
    assert dict(first.prefix_hashes) == expected_hashes
    assert dict(repeat.prefix_hashes) == expected_hashes
    assert first.scipy_version
    for smaller, larger in zip(sizes, sizes[1:]):
        np.testing.assert_array_equal(
            first.pools[smaller].grid_indices,
            first.pools[larger].grid_indices[:smaller],
        )
    for size in sizes:
        pool = first.pools[size]
        assert pool.size == size
        assert np.unique(pool.grid_indices, axis=0).shape[0] == size
        np.testing.assert_array_equal(
            physical_rows_to_grid_indices(pool.X_phys, _design()), pool.grid_indices
        )
        assert np.all((pool.X_norm >= 0.0) & (pool.X_norm <= 1.0))
        assert first.prefix_hashes[size] == hash_grid_index_prefix(pool.grid_indices)
        np.testing.assert_array_equal(
            pool.grid_indices, repeat.pools[size].grid_indices
        )


def test_different_scramble_changes_the_accepted_order_and_hash():
    primary = build_nested_sobol_discrete_pool(_design(), [64], scramble_seed=73)
    secondary = build_nested_sobol_discrete_pool(_design(), [64], scramble_seed=137)
    assert not np.array_equal(
        primary.largest_pool.grid_indices, secondary.largest_pool.grid_indices
    )
    assert primary.prefix_hashes[64] != secondary.prefix_hashes[64]


def test_exclusions_constraints_and_off_grid_observed_partition_are_stable():
    design = build_design(
        [
            InputSpec("a", 0.0, 4.0, 1.0),
            InputSpec("b", 0.0, 2.0, 1.0),
            InputSpec("c", 0.0, 1.0, 1.0),
        ]
    )
    observed = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 1.0]])
    pending = np.array([[2.0, 1.0, 1.0]])
    avoid = np.array([[4.0, 2.0, 1.0]])

    def require_even_a(X_phys, supplied_design):
        assert supplied_design is design
        return (X_phys[:, 0] % 2.0) == 0.0

    result = build_nested_sobol_discrete_pool(
        design,
        [5, 15],
        scramble_seed=11,
        observed_phys=observed,
        pending_phys=pending,
        avoid_phys=avoid,
        row_constraints=[require_even_a],
        max_raw_draws=4096,
    )
    excluded = {
        tuple(row)
        for row in physical_rows_to_grid_indices(
            np.vstack([observed[:1], pending, avoid]), design
        )
    }
    final = result.largest_pool
    assert result.ignored_off_grid_observed == 1
    assert final.size == 15
    assert np.all(final.X_phys[:, 0] % 2.0 == 0.0)
    assert not ({tuple(row) for row in final.grid_indices} & excluded)
    assert final.rejected_avoid > 0
    assert final.rejected_constraint > 0
    np.testing.assert_array_equal(result.pools[5].grid_indices, final.grid_indices[:5])


def test_pending_and_explicit_avoid_rows_must_be_exactly_on_grid():
    off_grid = np.array([[0.25, 10.0, 0.0]])
    with pytest.raises(ValueError, match="off-grid"):
        build_nested_sobol_discrete_pool(
            _design(), [4], scramble_seed=1, pending_phys=off_grid
        )
    with pytest.raises(ValueError, match="off-grid"):
        build_nested_sobol_discrete_pool(
            _design(), [4], scramble_seed=1, avoid_phys=off_grid
        )

    with pytest.raises(ValueError, match="within the design bounds"):
        build_nested_sobol_discrete_pool(
            _design(),
            [4],
            scramble_seed=1,
            observed_phys=np.array([[9.0, 10.0, 0.0]]),
        )


def test_sampler_never_materializes_the_cartesian_product(monkeypatch):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("full Cartesian materialization was attempted")

    monkeypatch.setattr(np, "meshgrid", forbidden)
    monkeypatch.setattr(np, "indices", forbidden)
    monkeypatch.setattr(itertools, "product", forbidden)
    result = build_nested_sobol_discrete_pool(_design(), [16, 32], scramble_seed=9)
    assert result.largest_pool.size == 32


def test_impossible_capacity_and_draw_limit_fail_with_structured_errors():
    tiny = build_design([InputSpec("x", 0.0, 1.0, 1.0)])
    with pytest.raises(CandidatePoolSamplingError) as capacity:
        build_nested_sobol_discrete_pool(
            tiny,
            [2],
            scramble_seed=1,
            observed_phys=np.array([[0.0]]),
        )
    assert capacity.value.draws == 0
    assert "exceeds" in capacity.value.reason

    larger = build_design([InputSpec("x", 0.0, 7.0, 1.0)])

    def reject_all(X_phys, supplied_design):
        del supplied_design
        return np.zeros(X_phys.shape[0], dtype=bool)

    with pytest.raises(CandidatePoolSamplingError) as limited:
        build_nested_sobol_discrete_pool(
            larger,
            [1],
            scramble_seed=2,
            row_constraints=[reject_all],
            max_raw_draws=8,
        )
    assert limited.value.draws == 8
    assert limited.value.accepted == 0
    assert limited.value.rejected_constraint > 0


@pytest.mark.parametrize(
    "sizes, seed, max_draws, message",
    [
        ([], 1, None, "must not be empty"),
        ([0], 1, None, "positive integers"),
        ([2, 2], 1, None, "duplicates"),
        ([2], True, None, "scramble_seed"),
        ([2], 1, 0, "max_raw_draws"),
    ],
)
def test_sampler_configuration_validation(sizes, seed, max_draws, message):
    kwargs = {} if max_draws is None else {"max_raw_draws": max_draws}
    with pytest.raises(ValueError, match=message):
        build_nested_sobol_discrete_pool(_design(), sizes, scramble_seed=seed, **kwargs)


def test_prefix_hash_requires_an_integer_matrix():
    with pytest.raises(TypeError, match="integer dtype"):
        hash_grid_index_prefix(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        hash_grid_index_prefix(np.array([0, 1], dtype=np.int64))
