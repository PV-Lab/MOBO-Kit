import numpy as np
import pandas as pd

from mobo_kit.robust_regions import (
    DEBUG_WATERMARK,
    cluster_candidate_regions,
    evaluate_consensus_criteria,
    select_robust_shortlist,
)


def _records() -> pd.DataFrame:
    rows = [
        ("pool_large", "pool", 0, 0, 0.00, 0.00, 10.0),
        ("model_default", "model", 0, 1, 0.00, 0.05, 9.0),
        ("beta_four", "beta", 1, 0, 0.05, 0.00, 8.0),
        ("pool_small", "pool", 8, 8, 0.80, 0.80, 7.0),
        ("model_default", "model", 8, 9, 0.80, 0.90, 6.0),
        ("beta_four", "beta", 9, 8, 0.90, 0.80, 5.0),
        ("pool_large", "pool", 4, 9, 0.40, 0.90, 4.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "run_family",
            "grid_0",
            "grid_1",
            "norm_0",
            "norm_1",
            "acquisition_score",
        ],
    ).assign(
        phys_0=lambda frame: frame["grid_0"] * 10.0,
        phys_1=lambda frame: frame["grid_1"] * 5.0,
        grid_valid=True,
        hard_distance_valid=True,
        boundary_coordinate_count=0,
        nearest_control_distance=0.5,
        pred_mean_0=0.6,
        pred_std_0=0.1,
    )


def _registry() -> dict[str, str]:
    return {
        "pool_large": "pool",
        "pool_small": "pool",
        "model_default": "model",
        "beta_four": "beta",
    }


def test_complete_link_regions_are_deterministic_and_family_weighted() -> None:
    records = _records()
    first = cluster_candidate_regions(
        records, distance_threshold=0.15, core_run_registry=_registry()
    )
    second = cluster_candidate_regions(
        records.sample(frac=1.0, random_state=91).reset_index(drop=True),
        distance_threshold=0.15,
        core_run_registry=_registry(),
    )

    assert first.region_count == 3
    comparable = [
        "region_id",
        "member_count",
        "distinct_run_count",
        "distinct_family_count",
        "family_weighted_persistence",
        "cluster_diameter",
        "medoid_grid_0",
        "medoid_grid_1",
    ]
    pd.testing.assert_frame_equal(
        first.regions[comparable], second.regions[comparable], check_exact=True
    )
    assert (first.regions["cluster_diameter"] <= 0.15 + 1e-12).all()
    assert first.regions.iloc[0]["distinct_family_count"] == 3
    assert np.isclose(
        first.regions.iloc[0]["family_weighted_persistence"],
        np.mean([0.5, 1.0, 1.0]),
    )
    assert first.regions["candidate_status"].eq(DEBUG_WATERMARK).all()
    assert (~first.regions["approved_for_experiment"]).all()


def test_threshold_sensitivity_and_diverse_shortlist() -> None:
    records = _records()
    strict = cluster_candidate_regions(
        records, distance_threshold=0.04, core_run_registry=_registry()
    )
    primary = cluster_candidate_regions(
        records, distance_threshold=0.15, core_run_registry=_registry()
    )
    loose = cluster_candidate_regions(
        records, distance_threshold=0.25, core_run_registry=_registry()
    )

    assert strict.region_count > primary.region_count
    assert loose.region_count <= primary.region_count
    shortlist = select_robust_shortlist(
        primary, minimum_count=2, maximum_count=3, minimum_normalized_distance=0.15
    )
    assert 2 <= len(shortlist) <= 3
    coords = shortlist[["medoid_norm_0", "medoid_norm_1"]].to_numpy()
    if len(coords) > 1:
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        assert distances[np.triu_indices(len(coords), k=1)].min() >= 0.15
    assert shortlist["shortlist_id"].is_unique
    assert shortlist["candidate_status"].eq(DEBUG_WATERMARK).all()


def _consensus_inputs(pass_all: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = pd.DataFrame(
        {
            "distinct_family_count": [3, 4, 3, 3, 5],
            "distinct_nonbaseline_family_count": [3, 4, 3, 3, 5],
            "all_grid_valid": [True] * 5,
            "all_hard_distance_valid": [True] * 5,
        }
    )
    shortlist = pd.DataFrame(
        {
            "distinct_nonbaseline_family_count": [3, 4, 3, 3, 5],
            "medoid_grid_0": [0, 0, 1, 2, 4],
            "medoid_grid_1": [0, 4, 0, 2, 4],
            "medoid_norm_0": [0.0, 0.0, 0.25, 0.5, 1.0],
            "medoid_norm_1": [0.0, 1.0, 0.0, 0.5, 1.0],
            "all_grid_valid": [True] * 5,
            "all_hard_distance_valid": [True] * 5,
            "debug_only": [True] * 5,
            "approved_for_experiment": [False] * 5,
            "approved_for_production": [False] * 5,
        }
    )
    if not pass_all:
        shortlist.loc[0, "all_hard_distance_valid"] = False
    return regions, shortlist


def test_consensus_is_created_only_when_every_declared_gate_passes() -> None:
    regions, shortlist = _consensus_inputs(pass_all=True)
    passed = evaluate_consensus_criteria(
        regions,
        shortlist,
        largest_two_regional_matches_within_0_15=4,
        largest_two_mean_matched_distance=0.08,
    )
    assert passed.passed is True
    assert passed.reasons == ()

    regions, shortlist = _consensus_inputs(pass_all=False)
    failed = evaluate_consensus_criteria(
        regions,
        shortlist,
        largest_two_regional_matches_within_0_15=3,
        largest_two_mean_matched_distance=0.11,
    )
    assert failed.passed is False
    assert set(failed.reasons) >= {
        "largest_two_nested_regional_matches",
        "largest_two_nested_mean_matched_distance",
        "chosen_pairwise_hard_distance_valid",
    }


def test_family_count_uses_distinct_categories_not_run_frequency() -> None:
    records = pd.concat([_records()] * 4, ignore_index=True)
    result = cluster_candidate_regions(
        records, distance_threshold=0.15, core_run_registry=_registry()
    )
    assert result.regions["distinct_family_count"].max() == 3
    assert result.regions["distinct_core_run_count"].max() <= 4


def test_region_table_exposes_relative_hvi_prediction_distance_and_boundary_metrics() -> (
    None
):
    records = _records()
    records["selection_order"] = records.groupby("run_id").cumcount() + 1
    records["nearest_observed_distance"] = [0.2, 0.3, 0.4, 0.8, 0.9, 1.0, 0.7]
    records["pred_mean_0"] = [0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4]
    records["pred_std_0"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    records["boundary_dimensions"] = [
        "input_a|input_b",
        "input_a",
        "input_b",
        "",
        "",
        "",
        "",
    ]

    result = cluster_candidate_regions(
        records, distance_threshold=0.15, core_run_registry=_registry()
    )
    region = result.regions.loc[
        (result.regions["medoid_grid_0"] == 0)
        & (result.regions["medoid_grid_1"].isin([0, 1]))
    ].iloc[0]

    assert region["median_selection_order"] == 1
    assert region["median_run_normalized_base_hvi"] == 1.0
    assert region["iqr_run_normalized_base_hvi"] == 0.0
    assert np.isclose(region["median_pred_mean_0"], 0.6)
    assert np.isclose(region["median_pred_std_0"], 0.2)
    assert np.isclose(region["median_nearest_observed_distance"], 0.3)
    assert np.isclose(region["boundary_dimension_0_frequency"], 2 / 3)
    assert np.isclose(region["boundary_dimension_1_frequency"], 2 / 3)
    assert region["boundary_dimension_name_frequency"] == (
        "input_a:0.666667|input_b:0.666667"
    )
    assert np.isclose(
        region["maximum_within_region_distance"], region["cluster_diameter"]
    )
    assert np.isclose(region["nested_coverage"], 0.5)
    assert region["model_coverage"] == 1.0
    assert region["beta_coverage"] == 1.0
    assert pd.isna(region["scramble_coverage"])
    assert not region["control_omission_correspondence_available"]
    assert pd.isna(region["control_included_vs_control_omitted_correspondence"])
    assert "base_hvi_normalized_within_run" in result.membership


def test_step2c_baseline_is_shared_but_six_family_coverage_is_equal_weighted() -> None:
    rows = [
        ("baseline", "baseline", 0, 0, 0.0, 0.0, 10.0),
        ("nested_1", "nested_pool", 8, 8, 0.8, 0.8, 9.0),
        ("scramble_1", "sobol_scramble", 8, 8, 0.8, 0.8, 8.0),
        ("model_1", "model_variant", 8, 8, 0.8, 0.8, 7.0),
        ("bound_1", "bounded_utility", 8, 8, 0.8, 0.8, 6.0),
        ("beta_1", "beta", 8, 8, 0.8, 0.8, 5.0),
        ("penalty_1", "local_penalty", 8, 8, 0.8, 0.8, 4.0),
    ]
    records = pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "run_family",
            "grid_0",
            "grid_1",
            "norm_0",
            "norm_1",
            "acquisition_score",
        ],
    )
    registry = dict(
        records[["run_id", "run_family"]].itertuples(index=False, name=None)
    )

    result = cluster_candidate_regions(
        records, distance_threshold=0.15, core_run_registry=registry
    )
    baseline_region = result.regions.loc[result.regions["medoid_grid_0"] == 0].iloc[0]

    for column in (
        "model_coverage",
        "nested_pool_coverage",
        "sobol_scramble_coverage",
        "bound_policy_coverage",
        "beta_coverage",
        "local_penalty_coverage",
    ):
        assert baseline_region[column] == 0.5
    assert baseline_region["family_weighted_persistence"] == 0.5
    assert baseline_region["study_family_weighted_persistence"] == 0.5
    assert baseline_region["registry_family_weighted_persistence"] == 1 / 7
    assert baseline_region["persistence_basis"] == (
        "six_equal_weight_step2c_study_families"
    )
    assert baseline_region["distinct_family_count"] == 1


def test_control_omission_correspondence_is_regional_and_configured() -> None:
    records = pd.DataFrame(
        [
            ("full_a", "model", 0, 0, 0.00, 0.00, 3.0, np.nan),
            ("omit_control", "model", 0, 1, 0.00, 0.05, 2.0, 1001),
            ("full_b", "model", 8, 8, 0.80, 0.80, 1.0, 1002),
        ],
        columns=[
            "run_id",
            "run_family",
            "grid_0",
            "grid_1",
            "norm_0",
            "norm_1",
            "acquisition_score",
            "omitted_sample_id",
        ],
    )
    registry = dict(
        records[["run_id", "run_family"]].itertuples(index=False, name=None)
    )

    result = cluster_candidate_regions(
        records,
        distance_threshold=0.15,
        core_run_registry=registry,
        control_sample_id=1001,
    )
    paired = result.regions.loc[result.regions["member_count"] == 2].iloc[0]
    unpaired = result.regions.loc[result.regions["member_count"] == 1].iloc[0]

    assert paired["control_omission_correspondence_available"]
    assert paired["control_included_represented_run_count"] == 1
    assert paired["control_included_total_run_count"] == 2
    assert paired["control_omitted_represented_run_count"] == 1
    assert paired["control_omitted_total_run_count"] == 1
    assert paired["control_included_vs_control_omitted_correspondence"] == 0.5
    assert unpaired["control_included_vs_control_omitted_correspondence"] == 0.0
