from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mobo_kit.candidate_diagnostics import (
    boundary_flags,
    grid_membership_mask,
    pairwise_normalized_distances,
    plot_candidate_pca,
    plot_distance_heatmap,
    plot_parallel_coordinates,
    plot_selection_scores,
    summarize_candidate_batch,
)
from mobo_kit.design import InputSpec, build_design


def _design():
    return build_design(
        [
            InputSpec("a", 0.0, 2.0, 1.0),
            InputSpec("b", 10.0, 20.0, 5.0),
        ]
    )


def test_pairwise_normalized_distances_and_summary():
    selected = np.array([[0.0, 0.0], [0.3, 0.4], [1.0, 0.0]])
    matrix = pairwise_normalized_distances(selected)
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)

    summary = summarize_candidate_batch(
        selected,
        observed_pending_norm=np.array([[0.0, 0.1]]),
        X_phys=np.array([[0.0, 10.0], [1.0, 15.0], [2.0, 10.0]]),
        design=_design(),
        metadata={"method": "TEST_ONLY"},
    )
    triangle = matrix[np.triu_indices(3, k=1)]
    assert summary.minimum_within_batch_distance == pytest.approx(triangle.min())
    assert summary.mean_within_batch_distance == pytest.approx(triangle.mean())
    assert summary.maximum_within_batch_distance == pytest.approx(triangle.max())
    assert summary.nearest_observed_pending_distance[0] == pytest.approx(0.1)
    assert summary.duplicate_row_pairs == ()
    assert summary.grid_valid_rows.tolist() == [True, True, True]
    assert summary.metadata == {"method": "TEST_ONLY"}


def test_duplicate_grid_and_boundary_checks():
    selected = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 1.0]])
    summary = summarize_candidate_batch(selected)
    assert summary.duplicate_row_pairs == ((0, 1),)
    assert boundary_flags(selected).tolist() == [
        [True, True],
        [True, True],
        [False, True],
    ]
    valid = grid_membership_mask(
        np.array([[0.0, 10.0], [1.5, 15.0], [2.0, 19.0]]), _design()
    )
    assert valid.tolist() == [True, False, False]


def test_weight_validation_and_weighted_distance():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    matrix = pairwise_normalized_distances(X, dimension_weights=np.array([1.0, 4.0]))
    assert matrix[0, 1] == pytest.approx(np.sqrt(5.0))
    with pytest.raises(ValueError, match="strictly positive"):
        pairwise_normalized_distances(X, dimension_weights=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        pairwise_normalized_distances(X, dimension_weights=np.ones(3))


def test_singleton_summary_has_explicit_empty_within_batch_statistics():
    summary = summarize_candidate_batch(np.array([[0.2, 0.8]]))
    assert summary.minimum_within_batch_distance is None
    assert summary.mean_within_batch_distance is None
    assert summary.maximum_within_batch_distance is None
    assert np.isnan(summary.nearest_observed_pending_distance[0])


def test_plotting_helpers_write_headless_pngs(tmp_path: Path):
    observed = np.array([[0.0, 0.0], [0.5, 0.3], [0.9, 1.0]])
    pool = np.linspace(0.0, 1.0, 40).reshape(20, 2)
    selected = np.array([[0.1, 0.8], [0.8, 0.2], [0.5, 0.5]])
    watermark = "DEBUG ONLY - TEST PLOT"
    paths = [
        plot_candidate_pca(
            observed,
            selected,
            tmp_path / "pca.png",
            pool_norm=pool,
            watermark=watermark,
        ),
        plot_parallel_coordinates(
            selected,
            ["a", "b"],
            tmp_path / "parallel.png",
            watermark=watermark,
        ),
        plot_distance_heatmap(selected, tmp_path / "distance.png", watermark=watermark),
        plot_selection_scores(
            [1, 2, 3],
            [-1.0, -1.2, -1.4],
            [-1.0, -1.8, -2.1],
            tmp_path / "scores.png",
            watermark=watermark,
        ),
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1000
        with Image.open(path) as image:
            assert image.info["Description"] == watermark


def test_diagnostics_reject_shape_and_partial_grid_context():
    with pytest.raises(ValueError, match="shape"):
        pairwise_normalized_distances(np.ones(3))
    with pytest.raises(ValueError, match="supplied together"):
        summarize_candidate_batch(np.ones((2, 2)), X_phys=np.ones((2, 2)), design=None)
    with pytest.raises(ValueError, match="same row count"):
        summarize_candidate_batch(
            np.ones((2, 2)), X_phys=np.ones((1, 2)), design=_design()
        )
    with pytest.raises(ValueError, match="duplicate_atol"):
        summarize_candidate_batch(np.ones((2, 2)), duplicate_atol=-1)
