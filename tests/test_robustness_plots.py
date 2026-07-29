from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from mobo_kit.robustness_plots import (  # noqa: E402
    DEBUG_WATERMARK,
    plot_acquisition_quality_vs_persistence,
    plot_ard_lengthscale_comparison,
    plot_boundary_enrichment,
    plot_bounded_utility_comparison,
    plot_candidate_predictions_vs_observed_ranges,
    plot_control_omission_candidate_region_comparison,
    plot_local_penalty_tradeoff,
    plot_loocv_diagnostics,
    plot_model_policy_region_correspondence,
    plot_nested_search_convergence,
    plot_observation_influence_ranking,
    plot_robust_region_overview,
    plot_run_region_persistence_heatmap,
    plot_shortlist_medoid_parallel_coordinates,
    plot_shortlist_region_influence_sensitivity,
)


def _loocv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "objective": ["Uniformity", "Uniformity", "Thickness", "Thickness"],
            "observed": [0.2, 0.8, 0.4, 0.9],
            "predicted_mean": [0.25, 0.72, 0.48, 0.82],
            "predicted_std": [0.1, 0.12, 0.15, 0.1],
        }
    )


def _convergence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool_size": [16_384, 32_768, 65_536, 131_072],
            "mean_matched_distance": [0.16, 0.11, 0.07, 0.0],
            "maximum_matched_distance": [0.29, 0.21, 0.12, 0.0],
            "regional_matches_0_10": [1, 2, 4, 5],
            "regional_matches_0_15": [2, 4, 5, 5],
            "regional_matches_0_20": [3, 5, 5, 5],
        }
    )


def _hyperparameter_frame() -> pd.DataFrame:
    rows = []
    for variant_index, variant in enumerate(("dim_scaled_prior", "conservative")):
        for objective_index, objective in enumerate(("Uniformity", "Thickness")):
            for fold in range(2):
                rows.append(
                    {
                        "variant_name": variant,
                        "objective_name": objective,
                        "fit_key": f"{variant}-{objective}-{fold}",
                        "ard_lengthscale_speed": 0.08
                        + 0.03 * variant_index
                        + 0.01 * objective_index
                        + 0.005 * fold,
                        "ard_lengthscale_time": 0.2
                        + 0.04 * variant_index
                        + 0.01 * objective_index
                        + 0.005 * fold,
                        "ard_lengthscale_speed_near_floor": False,
                        "ard_lengthscale_speed_very_small_normalized_domain": False,
                        "ard_lengthscale_speed_extremely_large_flat": False,
                    }
                )
    return pd.DataFrame(rows)


def _prediction_range_frame() -> pd.DataFrame:
    rows = []
    ranges = {"Uniformity": (0.1, 0.9), "Thickness": (-1.5, 1.2)}
    for objective_index, (objective, observed_range) in enumerate(ranges.items()):
        for candidate_index, candidate in enumerate(("C1", "C2")):
            for model_index, model in enumerate(("dim_scaled_prior", "conservative")):
                rows.append(
                    {
                        "candidate_id": candidate,
                        "model_variant": model,
                        "objective_name": objective,
                        "predicted_mean": 0.25
                        + 0.12 * candidate_index
                        + 0.04 * model_index
                        - 0.3 * objective_index,
                        "predicted_std": 0.08 + 0.01 * model_index,
                        "observed_minimum": observed_range[0],
                        "observed_maximum": observed_range[1],
                    }
                )
    return pd.DataFrame(rows)


def _shortlist_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": ["REGION-001", "REGION-002", "REGION-010"],
            "medoid_norm_0": [0.1, 0.45, 0.82],
            "medoid_norm_1": [0.25, 0.7, 0.55],
            "medoid_norm_2": [0.35, 0.2, 0.9],
            "family_weighted_persistence": [0.9, 0.65, 0.4],
        }
    )


def _membership_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["pool-1", "pool-1", "pool-2", "model-default", "model-alt"],
            "run_family": ["nested", "nested", "nested", "model", "model"],
            "region_id": [
                "REGION-001",
                "REGION-002",
                "REGION-001",
                "REGION-002",
                "REGION-010",
            ],
            "model_variant": [
                "dim_scaled_prior",
                "dim_scaled_prior",
                "dim_scaled_prior",
                "dim_scaled_prior",
                "conservative",
            ],
            "bound_policy": ["clip_ucb", "clip_ucb", "none", "clip_ucb", "clip_ucb"],
        }
    )


def _region_quality_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": ["REGION-001", "REGION-002", "REGION-010"],
            "family_weighted_persistence": [0.9, 0.65, 0.4],
            "median_acquisition_score": [4.1, 3.7, 2.8],
            "distinct_family_count": [5, 4, 2],
        }
    )


def _region_sensitivity_frame() -> pd.DataFrame:
    rows = []
    for region_index, region in enumerate(("REGION-001", "REGION-002", "REGION-010")):
        for sample in (1, 2, 10):
            rows.append(
                {
                    "region_id": region,
                    "omitted_sample_id": sample,
                    "prediction_change": 0.01 * (region_index + 1) * sample,
                }
            )
    return pd.DataFrame(rows)


def _assert_watermarked_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1000
    assert DEBUG_WATERMARK.encode() in path.read_bytes()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.info["Description"] == DEBUG_WATERMARK
        assert "Title" not in image.info
        assert all(
            "SECRET_RECIPE_VALUE" not in str(value) for value in image.info.values()
        )


def test_all_required_plot_topics_write_watermarked_pngs_and_close_figures(tmp_path):
    paths = [
        plot_loocv_diagnostics(_loocv_frame(), tmp_path / "loocv.png"),
        plot_nested_search_convergence(_convergence_frame(), tmp_path / "nested.png"),
        plot_bounded_utility_comparison(
            pd.DataFrame(
                {
                    "policy": ["none", "none", "clip_ucb", "clip_ucb"],
                    "raw_value": [1.2, 0.8, 1.2, 0.8],
                    "bounded_value": [1.2, 0.8, 1.0, 0.8],
                }
            ),
            tmp_path / "bounded.png",
        ),
        plot_local_penalty_tradeoff(
            pd.DataFrame(
                {
                    "variant": ["none", "radius_0.15", "radius_0.25"],
                    "sum_base_hvi": [4.2, 4.0, 3.8],
                    "minimum_within_batch_distance": [0.11, 0.16, 0.21],
                }
            ),
            tmp_path / "penalty.png",
        ),
        plot_observation_influence_ranking(
            pd.DataFrame(
                {"sample_id": [1, 2, 3, 4], "influence_score": [0.9, 0.2, 0.5, 0.1]}
            ),
            tmp_path / "influence.png",
        ),
        plot_boundary_enrichment(
            pd.DataFrame(
                {
                    "input_name": ["speed", "time", "volume"],
                    "pool_lower_boundary_frequency": [0.1, 0.08, 0.12],
                    "top_lower_boundary_frequency": [0.2, 0.15, 0.25],
                    "selected_lower_boundary_frequency": [0.4, 0.2, 0.6],
                    "pool_upper_boundary_frequency": [0.1, 0.08, 0.12],
                    "top_upper_boundary_frequency": [0.2, 0.15, 0.25],
                    "selected_upper_boundary_frequency": [0.3, 0.2, 0.5],
                }
            ),
            tmp_path / "boundary.png",
        ),
        plot_robust_region_overview(
            np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.12, 0.18, 0.34],
                    [0.8, 0.7, 0.6],
                    [0.82, 0.73, 0.58],
                ]
            ),
            ["A", "A", "B", "B"],
            tmp_path / "regions.png",
            persistence=[3.0, 3.0, 5.0, 5.0],
            input_names=["speed", "time", "volume"],
        ),
    ]
    assert len(set(paths)) == 7
    for path in paths:
        _assert_watermarked_png(path)
    assert plt.get_fignums() == []


def test_extended_model_region_plot_topics_are_watermarked_and_headless(tmp_path):
    membership = _membership_frame()
    paths = [
        plot_ard_lengthscale_comparison(
            _hyperparameter_frame(), tmp_path / "ard_lengthscales.png"
        ),
        plot_control_omission_candidate_region_comparison(
            np.array([[0.1, 0.2, 0.3], [0.75, 0.7, 0.65]]),
            np.array([[0.12, 0.22, 0.29], [0.68, 0.72, 0.61]]),
            tmp_path / "control_omission_regions.png",
            full_region_labels=["REGION-001", "REGION-002"],
            omit_region_labels=["REGION-001", "REGION-002"],
        ),
        plot_candidate_predictions_vs_observed_ranges(
            _prediction_range_frame(), tmp_path / "predicted_ranges.png"
        ),
        plot_shortlist_medoid_parallel_coordinates(
            _shortlist_frame(),
            tmp_path / "shortlist_parallel.png",
            input_names=["speed", "time", "volume"],
        ),
        plot_run_region_persistence_heatmap(
            membership, tmp_path / "run_region_heatmap.png"
        ),
        plot_model_policy_region_correspondence(
            membership, tmp_path / "model_policy_regions.png"
        ),
        plot_acquisition_quality_vs_persistence(
            _region_quality_frame(), tmp_path / "quality_persistence.png"
        ),
        plot_shortlist_region_influence_sensitivity(
            _region_sensitivity_frame(), tmp_path / "region_sensitivity.png"
        ),
    ]
    assert len(set(paths)) == 8
    for path in paths:
        _assert_watermarked_png(path)
    assert plt.get_fignums() == []


def test_nested_plot_is_deterministic_and_row_order_invariant(tmp_path):
    frame = _convergence_frame()
    first = plot_nested_search_convergence(frame, tmp_path / "first.png")
    second = plot_nested_search_convergence(
        frame.iloc[::-1].reset_index(drop=True), tmp_path / "second.png"
    )
    assert (
        hashlib.sha256(first.read_bytes()).digest()
        == hashlib.sha256(second.read_bytes()).digest()
    )


def test_extended_heatmap_is_deterministic_and_row_order_invariant(tmp_path):
    frame = _membership_frame()
    first = plot_run_region_persistence_heatmap(frame, tmp_path / "heatmap-first.png")
    second = plot_run_region_persistence_heatmap(
        frame.sample(frac=1.0, random_state=73).reset_index(drop=True),
        tmp_path / "heatmap-second.png",
    )
    assert (
        hashlib.sha256(first.read_bytes()).digest()
        == hashlib.sha256(second.read_bytes()).digest()
    )


def test_control_omission_region_comparison_is_row_order_invariant(tmp_path):
    full = np.array([[0.1, 0.2, 0.3], [0.75, 0.7, 0.65], [0.4, 0.5, 0.6]])
    omitted = np.array([[0.12, 0.22, 0.29], [0.68, 0.72, 0.61], [0.39, 0.53, 0.57]])
    full_labels = np.array(["REGION-001", "REGION-002", "REGION-003"])
    omitted_labels = np.array(["REGION-001", "REGION-002", "REGION-003"])
    first = plot_control_omission_candidate_region_comparison(
        full,
        omitted,
        tmp_path / "control-omission-first.png",
        full_region_labels=full_labels,
        omit_region_labels=omitted_labels,
    )
    full_order = np.array([2, 0, 1])
    omitted_order = np.array([1, 2, 0])
    second = plot_control_omission_candidate_region_comparison(
        full[full_order],
        omitted[omitted_order],
        tmp_path / "control-omission-second.png",
        full_region_labels=full_labels[full_order],
        omit_region_labels=omitted_labels[omitted_order],
    )
    assert (
        hashlib.sha256(first.read_bytes()).digest()
        == hashlib.sha256(second.read_bytes()).digest()
    )


def test_loocv_accepts_explicit_column_names(tmp_path):
    renamed = _loocv_frame().rename(
        columns={
            "objective": "target",
            "observed": "truth",
            "predicted_mean": "estimate",
            "predicted_std": "uncertainty",
        }
    )
    path = plot_loocv_diagnostics(
        renamed,
        tmp_path / "custom.png",
        objective_column="target",
        observed_column="truth",
        predicted_column="estimate",
        std_column="uncertainty",
    )
    _assert_watermarked_png(path)


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda path: plot_loocv_diagnostics(pd.DataFrame(), path),
            "must not be empty",
        ),
        (
            lambda path: plot_nested_search_convergence(
                pd.DataFrame({"pool_size": [10]}), path
            ),
            "missing required columns",
        ),
        (
            lambda path: plot_bounded_utility_comparison(
                pd.DataFrame(
                    {
                        "policy": ["none"],
                        "raw_value": [np.nan],
                        "bounded_value": [1.0],
                    }
                ),
                path,
            ),
            "complete and finite",
        ),
        (
            lambda path: plot_local_penalty_tradeoff(
                pd.DataFrame(
                    {
                        "variant": ["none"],
                        "sum_base_hvi": [1.0],
                        "minimum_within_batch_distance": [-0.1],
                    }
                ),
                path,
            ),
            "non-negative",
        ),
        (
            lambda path: plot_observation_influence_ranking(
                pd.DataFrame({"sample_id": [1, 1], "influence_score": [0.2, 0.3]}),
                path,
            ),
            "unique",
        ),
        (
            lambda path: plot_boundary_enrichment(
                pd.DataFrame(
                    {
                        "input_name": ["x"],
                        "pool_lower_boundary_frequency": [0.1],
                        "top_lower_boundary_frequency": [0.2],
                        "selected_lower_boundary_frequency": [1.1],
                        "pool_upper_boundary_frequency": [0.1],
                        "top_upper_boundary_frequency": [0.2],
                        "selected_upper_boundary_frequency": [0.3],
                    }
                ),
                path,
            ),
            r"remain in \[0, 1\]",
        ),
        (
            lambda path: plot_robust_region_overview(
                np.array([[0.1, 0.2], [0.8, 0.9]]), ["A"], path
            ),
            "one label per",
        ),
    ],
)
def test_plot_helpers_reject_empty_missing_or_invalid_data(tmp_path, call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call(tmp_path / "invalid.png")
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda path: plot_ard_lengthscale_comparison(
                _hyperparameter_frame().assign(ard_lengthscale_speed=0.0), path
            ),
            "strictly positive",
        ),
        (
            lambda path: plot_control_omission_candidate_region_comparison(
                np.array([[0.1, 0.2]]), np.array([[0.1, 0.2, 0.3]]), path
            ),
            "share dimensions",
        ),
        (
            lambda path: plot_candidate_predictions_vs_observed_ranges(
                _prediction_range_frame().assign(
                    observed_minimum=2.0, observed_maximum=1.0
                ),
                path,
            ),
            "must not exceed",
        ),
        (
            lambda path: plot_shortlist_medoid_parallel_coordinates(
                _shortlist_frame().drop(columns="medoid_norm_1"), path
            ),
            "contiguous",
        ),
        (
            lambda path: plot_run_region_persistence_heatmap(
                pd.concat(
                    [
                        _membership_frame(),
                        _membership_frame().iloc[[0]].assign(run_family="conflicting"),
                    ],
                    ignore_index=True,
                ),
                path,
            ),
            "exactly one run family",
        ),
        (
            lambda path: plot_model_policy_region_correspondence(
                _membership_frame().assign(weight=-1.0),
                path,
                value_column="weight",
            ),
            "non-negative",
        ),
        (
            lambda path: plot_acquisition_quality_vs_persistence(
                _region_quality_frame().assign(family_weighted_persistence=1.2),
                path,
            ),
            r"remain in \[0, 1\]",
        ),
        (
            lambda path: plot_shortlist_region_influence_sensitivity(
                _region_sensitivity_frame().iloc[:-1], path
            ),
            "cover every",
        ),
    ],
)
def test_extended_plot_helpers_fail_closed_on_invalid_inputs(tmp_path, call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call(tmp_path / "invalid-extended.png")
    assert plt.get_fignums() == []


def test_loocv_uncertainty_and_region_inputs_fail_closed(tmp_path):
    invalid_std = _loocv_frame()
    invalid_std.loc[0, "predicted_std"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        plot_loocv_diagnostics(invalid_std, tmp_path / "std.png")

    with pytest.raises(ValueError, match="at least two rows"):
        plot_robust_region_overview(np.array([[0.2, 0.3]]), ["A"], tmp_path / "one.png")
    with pytest.raises(ValueError, match="strictly positive"):
        plot_robust_region_overview(
            np.array([[0.2, 0.3], [0.7, 0.8]]),
            ["A", "B"],
            tmp_path / "persist.png",
            persistence=[1.0, 0.0],
        )
    with pytest.raises(ValueError, match="one nonblank name"):
        plot_robust_region_overview(
            np.array([[0.2, 0.3], [0.7, 0.8]]),
            ["A", "B"],
            tmp_path / "names.png",
            input_names=["only_one"],
        )
    assert plt.get_fignums() == []


def test_output_must_be_png_and_no_figure_is_left_open(tmp_path):
    with pytest.raises(ValueError, match=r"\.png suffix"):
        plot_observation_influence_ranking(
            pd.DataFrame({"sample_id": [1], "influence_score": [0.5]}),
            tmp_path / "not_png.pdf",
        )
    assert plt.get_fignums() == []
