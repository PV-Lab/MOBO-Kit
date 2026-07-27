from __future__ import annotations

import numpy as np
import pytest

from mobo_kit.batch_comparison import (
    DEFAULT_REGIONAL_THRESHOLDS,
    compare_candidate_batches,
    regional_match_batches,
)


def test_known_exact_regional_chamfer_and_hausdorff_metrics():
    reference = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    comparison = np.array([[0.0, 0.0], [0.5, 0.6], [0.8, 1.0]])
    result = regional_match_batches(reference, comparison)

    assert result.exact_overlap_count == 1
    assert result.exact_overlap == 1
    assert result.jaccard_overlap == pytest.approx(1.0 / 5.0)
    np.testing.assert_allclose(result.matched_pair_distances, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(result.matched_distances, [0.0, 0.1, 0.2])
    assert result.mean_matched_distance == pytest.approx(0.1)
    assert result.maximum_matched_distance == pytest.approx(0.2)
    assert result.max_matched_distance == pytest.approx(0.2)
    assert dict(result.regional_match_counts) == {0.10: 2, 0.15: 2, 0.20: 3}
    assert result.regional_match_count(0.15) == 2
    assert result.symmetric_chamfer_distance == pytest.approx(0.1)
    assert result.chamfer_distance == pytest.approx(0.1)
    assert result.hausdorff_distance == pytest.approx(0.2)
    assert result.thresholds == DEFAULT_REGIONAL_THRESHOLDS


def test_hungarian_assignment_enforces_optimal_one_to_one_matching():
    reference = np.array([[0.0], [0.1]])
    comparison = np.array([[0.09], [1.0]])
    result = compare_candidate_batches(reference, comparison)

    np.testing.assert_allclose(result.matched_pair_distances, [0.09, 0.9])
    np.testing.assert_allclose(result.matched_reference_rows, [[0.0], [0.1]])
    np.testing.assert_allclose(result.matched_comparison_rows, [[0.09], [1.0]])
    assert result.mean_matched_distance == pytest.approx(0.495)
    assert result.maximum_matched_distance == pytest.approx(0.9)
    assert dict(result.regional_match_counts) == {0.10: 1, 0.15: 1, 0.20: 1}


def test_all_metrics_and_canonical_pairs_are_invariant_to_row_reordering():
    reference = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.4], [0.6, 0.7], [0.0, 0.0]])
    comparison = np.array(
        [[0.31, 0.39], [0.62, 0.72], [0.82, 0.18], [0.0, 0.0], [0.2, 0.8]]
    )
    baseline = regional_match_batches(reference, comparison)
    reordered = regional_match_batches(
        reference[[3, 0, 4, 1, 2]], comparison[[2, 4, 1, 0, 3]]
    )

    assert reordered.exact_overlap_count == baseline.exact_overlap_count
    assert reordered.jaccard_overlap == baseline.jaccard_overlap
    assert dict(reordered.regional_match_counts) == dict(baseline.regional_match_counts)
    assert reordered.mean_matched_distance == baseline.mean_matched_distance
    assert reordered.maximum_matched_distance == baseline.maximum_matched_distance
    assert reordered.symmetric_chamfer_distance == baseline.symmetric_chamfer_distance
    assert reordered.hausdorff_distance == baseline.hausdorff_distance
    np.testing.assert_array_equal(
        reordered.matched_reference_rows, baseline.matched_reference_rows
    )
    np.testing.assert_array_equal(
        reordered.matched_comparison_rows, baseline.matched_comparison_rows
    )
    np.testing.assert_array_equal(
        reordered.matched_pair_distances, baseline.matched_pair_distances
    )


def test_rectangular_batches_use_minimum_cardinality_assignment_and_full_chamfer():
    reference = np.array([[0.0], [0.5], [1.0]])
    comparison = np.array([[0.1], [0.9]])
    result = regional_match_batches(reference, comparison)

    assert result.matched_pair_distances.shape == (2,)
    np.testing.assert_allclose(result.matched_pair_distances, [0.1, 0.1])
    assert result.mean_matched_distance == pytest.approx(0.1)
    assert result.symmetric_chamfer_distance == pytest.approx(
        0.5 * ((0.1 + 0.4 + 0.1) / 3.0 + (0.1 + 0.1) / 2.0)
    )
    assert result.hausdorff_distance == pytest.approx(0.4)


def test_exact_overlap_uses_set_jaccard_even_with_repeated_rows():
    reference = np.array([[0.0], [0.0], [0.5]])
    comparison = np.array([[0.0], [1.0], [1.0]])
    result = regional_match_batches(reference, comparison)
    assert result.exact_overlap_count == 1
    assert result.jaccard_overlap == pytest.approx(1.0 / 3.0)


def test_custom_thresholds_are_sorted_and_require_exact_lookup():
    result = regional_match_batches(
        np.array([[0.0], [1.0]]),
        np.array([[0.05], [0.8]]),
        thresholds=(0.2, 0.05, 0.1),
    )
    assert result.thresholds == (0.05, 0.1, 0.2)
    assert dict(result.regional_match_counts) == {0.05: 1, 0.1: 1, 0.2: 2}
    with pytest.raises(KeyError, match="not evaluated"):
        result.regional_match_count(0.15)


@pytest.mark.parametrize(
    "reference, comparison, thresholds, message",
    [
        (np.empty((0, 2)), np.ones((1, 2)), (0.1,), "non-empty"),
        (np.ones((1, 2)), np.ones((1, 3)), (0.1,), "same input dimension"),
        (np.array([[np.nan]]), np.ones((1, 1)), (0.1,), "finite"),
        (np.array([[1.01]]), np.ones((1, 1)), (0.1,), "normalized"),
        (np.ones((1, 1)), np.ones((1, 1)), (), "must not be empty"),
        (np.ones((1, 1)), np.ones((1, 1)), (0.1, 0.1), "duplicates"),
        (np.ones((1, 1)), np.ones((1, 1)), (-0.1,), "non-negative"),
    ],
)
def test_batch_comparison_validation(reference, comparison, thresholds, message):
    with pytest.raises(ValueError, match=message):
        regional_match_batches(reference, comparison, thresholds=thresholds)
