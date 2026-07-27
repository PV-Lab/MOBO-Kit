from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mobo_kit.candidate_pool import physical_rows_to_grid_indices
from mobo_kit.d2d_campaign import (
    D2D_DEBUG_WATERMARK,
    D2D_INPUT_COLUMNS,
    D2D_OBJECTIVE_COLUMNS,
    D2D_OBJECTIVE_NAMES,
    D2D_REFERENCE_POINT_UTILITY,
    D2DDebugConfigError,
    ReplicateAggregationResult,
    aggregate_replicate_objectives,
    build_d2d_objective_transform,
    combine_r0_and_aggregated_r1,
    expand_candidates_to_replicates,
    load_d2d_debug_config,
    prepare_d2d_training_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG_PATH = REPO_ROOT / "configs" / "d2d_step2b_debug.yaml"
WORKBOOK_INPUT_COLUMNS = (
    "speed_1",
    "time_1",
    "speed_2",
    "time_2",
    "precur_conc",
    "precur_vol (uL)",
    "anneal_temp",
    "anneal_time",
    "anti_vol",
    "anti_time",
)


def _load_raw_config(*, materialize: bool = True) -> dict[str, Any]:
    raw = yaml.safe_load(DEBUG_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    if materialize:
        raw["template_only"] = False
        raw["workbook"]["expected_sha256"] = hashlib.sha256(
            b"public synthetic workbook identity"
        ).hexdigest()
    return raw


def _write_config(
    tmp_path: Path, raw: dict[str, Any], *, name: str = "sanitized_debug.yaml"
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path):
    return load_d2d_debug_config(_write_config(tmp_path, _load_raw_config()))


def _synthetic_r0_frame(config) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    workbook_columns = dict(zip(D2D_INPUT_COLUMNS, WORKBOOK_INPUT_COLUMNS))
    exceptions = {
        (exception.sample_id, exception.input_name): exception.observed_value
        for exception in config.off_grid_exceptions
    }
    for sample_index, sample_id in enumerate(config.expected_sample_ids):
        row: dict[str, float | int] = {"Sample number": sample_id}
        for dimension, (column, grid) in enumerate(
            zip(WORKBOOK_INPUT_COLUMNS, config.design.var_array)
        ):
            grid_index = (sample_index * (dimension + 1) + dimension) % len(grid)
            row[column] = float(grid[grid_index])
        for input_name in D2D_INPUT_COLUMNS:
            key = (sample_id, input_name)
            if key in exceptions:
                row[workbook_columns[input_name]] = exceptions[key]
        row["Uniformity score"] = 0.25 + 0.02 * sample_index
        row["Optoelectronic score"] = -2.0 + 0.05 * sample_index
        row["Thickness score"] = 0.35 + 0.015 * sample_index
        rows.append(row)
    return pd.DataFrame(rows)


def _five_synthetic_candidates(config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_index in range(5):
        row: dict[str, Any] = {"campaign_id": config.raw["campaign_id"]}
        for dimension, (name, grid) in enumerate(
            zip(D2D_INPUT_COLUMNS, config.design.var_array)
        ):
            if dimension == 0:
                grid_index = candidate_index
            else:
                grid_index = len(grid) - 1
            row[name] = float(grid[grid_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _measured_replicates(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _five_synthetic_candidates(config)
    records = expand_candidates_to_replicates(candidates)
    for candidate_index, candidate_id in enumerate(
        records["candidate_id"].drop_duplicates(), start=1
    ):
        mask = records["candidate_id"] == candidate_id
        records.loc[mask, "Uniformity score"] = (
            np.asarray([0.10, 0.15, 0.20]) + candidate_index / 10.0
        )
        records.loc[mask, "Optoelectronic score"] = (
            np.asarray([-2.0, -1.0, 0.0]) + candidate_index
        )
        records.loc[mask, "Thickness score"] = (
            np.asarray([0.30, 0.40, 0.50]) + candidate_index / 20.0
        )
    return candidates, records


def test_debug_config_resolves_materialized_public_contract(config) -> None:
    assert config.raw["run_mode"] == "debug"
    assert config.raw["debug_run_authorized"] is True
    assert config.raw["approved_for_production"] is False
    assert config.raw["approved_for_experiment"] is False
    assert config.workbook_profile == "d2d_summary_v3_scores"
    assert config.workbook_sheet == "Sheet1"
    assert config.expected_content_range == "A1:AI20"
    assert config.expected_sample_ids == tuple(range(1001, 1016))
    assert tuple(config.design.names) == D2D_INPUT_COLUMNS
    assert config.control_sample_ids == (1001,)
    assert config.include_control_in_debug_model is True
    assert config.r1_batch_size == 5
    assert config.replicates_per_condition == 3
    assert config.debug_watermark == D2D_DEBUG_WATERMARK
    assert config.output_root == "local_outputs/d2d_step2b_debug"
    np.testing.assert_array_equal(
        config.reference_point_utility, D2D_REFERENCE_POINT_UTILITY
    )


def test_tracked_debug_config_is_a_nonrunnable_public_template(
    tmp_path: Path,
) -> None:
    raw = _load_raw_config(materialize=False)
    with pytest.raises(D2DDebugConfigError, match="public template"):
        load_d2d_debug_config(_write_config(tmp_path, raw))


def test_public_template_requires_explicit_synthetic_resolution() -> None:
    config = load_d2d_debug_config(DEBUG_CONFIG_PATH, allow_public_template=True)

    assert len(config.expected_workbook_sha256) == 64
    assert config.expected_workbook_sha256 != ("REQUIRED_IN_IGNORED_PRIVATE_CONFIG")
    exception = config.off_grid_exceptions[0]
    dimension = config.design.names.index(exception.input_name)
    grid = config.design.var_array[dimension]
    assert exception.observed_value == pytest.approx((grid[0] + grid[1]) / 2.0)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("run_mode", "production"),
        ("debug_run_authorized", False),
        ("debug_run_authorized", 1),
        ("approved_for_production", True),
        ("approved_for_production", 0),
        ("approved_for_experiment", True),
        ("approved_for_experiment", 0),
    ],
)
def test_debug_config_rejects_weakened_or_nonboolean_safety_flags(
    tmp_path: Path, field: str, invalid_value: Any
) -> None:
    raw = _load_raw_config()
    raw[field] = invalid_value

    with pytest.raises(D2DDebugConfigError, match=field):
        load_d2d_debug_config(_write_config(tmp_path, raw))


def test_debug_config_rejects_contract_drift(tmp_path: Path) -> None:
    mutations = [
        ("schema version", lambda raw: raw.update(schema_version="production-v1")),
        ("campaign id", lambda raw: raw.update(campaign_id="")),
        (
            "content range",
            lambda raw: raw["workbook"].update(expected_content_range="A1:AB16"),
        ),
        (
            "objective direction",
            lambda raw: raw["objectives"][0].update(direction="minimize"),
        ),
        (
            "objective transform",
            lambda raw: raw["objectives"][1].update(utility_transform="affine"),
        ),
        (
            "objective source",
            lambda raw: raw["objectives"][2].update(excel_column="AC"),
        ),
        (
            "approved input grid",
            lambda raw: raw["inputs"][0].update(stop=6500),
        ),
        (
            "support formula",
            lambda raw: raw["objectives"][0].update(support_formula="L * N"),
        ),
        (
            "thickness target",
            lambda raw: raw["objectives"][2].update(target_nm=700.0),
        ),
        (
            "score validation policy",
            lambda raw: raw["qc_policy"].update(thickness_mismatch="warn"),
        ),
        (
            "safe output root",
            lambda raw: raw["outputs"].update(root="docs/private_candidates"),
        ),
        ("baseline beta", lambda raw: raw["r1"].update(beta=9.0)),
        (
            "local baseline",
            lambda raw: raw["local_penalization"].update(radius=0.35),
        ),
        ("reproducibility seed", lambda raw: raw["reproducibility"].update(seed=137)),
        (
            "reference point",
            lambda raw: raw.update(reference_point_utility=[0.0, -10.0, -0.01]),
        ),
        ("constraints", lambda raw: raw.update(constraints=["synthetic constraint"])),
    ]
    for index, (label, mutate) in enumerate(mutations):
        raw = deepcopy(_load_raw_config())
        mutate(raw)
        with pytest.raises((D2DDebugConfigError, ValueError)):
            load_d2d_debug_config(
                _write_config(tmp_path, raw, name=f"invalid-{index}-{label}.yaml")
            )


def test_objective_contract_is_named_ordered_identity_and_maximize(config) -> None:
    transform = build_d2d_objective_transform()

    assert transform.names == D2D_OBJECTIVE_NAMES
    assert (
        tuple(spec.source_column for spec in transform.specs) == D2D_OBJECTIVE_COLUMNS
    )
    assert tuple(spec.goal for spec in transform.specs) == ("maximize",) * 3
    assert tuple(spec.transform for spec in transform.specs) == ("identity",) * 3
    values = torch.tensor([[0.7, -1.25, 0.9]], dtype=torch.float64)
    transformed = transform(values)
    assert torch.equal(transformed, values)
    np.testing.assert_array_equal(
        config.reference_point_utility, np.asarray([-0.01, -10.0, -0.01])
    )


def test_training_data_includes_control_and_partitions_approved_exception(
    config,
) -> None:
    frame = _synthetic_r0_frame(config)
    training = prepare_d2d_training_data(frame, config)

    assert training.X_phys_all.shape == (15, 10)
    assert training.X_norm_all.shape == (15, 10)
    assert training.Y_objectives.shape == (15, 3)
    assert training.include_in_model.tolist() == [True] * 15
    assert training.row_roles == ("control",) + ("r0_lhs",) * 14
    exception = config.off_grid_exceptions[0]
    assert (
        training.X_phys_all[0, D2D_INPUT_COLUMNS.index(exception.input_name)]
        == exception.observed_value
    )
    assert training.on_grid_mask.tolist() == [False] + [True] * 14
    assert training.on_grid_sample_ids.tolist() == list(config.expected_sample_ids[1:])
    assert training.on_grid_grid_indices.shape == (14, 10)
    assert training.off_grid_exceptions == config.off_grid_exceptions
    assert all(np.isfinite(training.X_phys_all.ravel()))
    assert np.all((training.X_norm_all >= 0.0) & (training.X_norm_all <= 1.0))
    assert np.all(training.Y_objectives > config.reference_point_utility)

    expected_indices = physical_rows_to_grid_indices(
        training.X_phys_all[1:], config.design
    )
    np.testing.assert_array_equal(training.on_grid_grid_indices, expected_indices)
    with pytest.raises(ValueError, match="off-grid"):
        physical_rows_to_grid_indices(training.X_phys_all, config.design)

    ablated = prepare_d2d_training_data(frame, config, include_control=False)
    assert ablated.include_in_model.tolist() == [False] + [True] * 14


def test_training_data_rejects_unapproved_off_grid_observation(config) -> None:
    frame = _synthetic_r0_frame(config)
    exception = config.off_grid_exceptions[0]
    workbook_column = WORKBOOK_INPUT_COLUMNS[
        D2D_INPUT_COLUMNS.index(exception.input_name)
    ]
    frame.loc[1, workbook_column] = exception.observed_value

    with pytest.raises(ValueError, match="has an unapproved off-grid value"):
        prepare_d2d_training_data(frame, config)


def test_training_data_rejects_duplicate_observed_recipes(config) -> None:
    frame = _synthetic_r0_frame(config)
    frame.loc[2, list(WORKBOOK_INPUT_COLUMNS)] = frame.loc[
        1, list(WORKBOOK_INPUT_COLUMNS)
    ].to_numpy()

    with pytest.raises(ValueError, match="training recipes must be unique"):
        prepare_d2d_training_data(frame, config)


@pytest.mark.parametrize(
    ("objective", "invalid_score"),
    [
        ("Uniformity score", -0.01),
        ("Uniformity score", 1.01),
        ("Thickness score", 1.01),
    ],
)
def test_training_data_rejects_uniformity_outside_unit_interval(
    config, objective: str, invalid_score: float
) -> None:
    frame = _synthetic_r0_frame(config)
    frame.loc[4, objective] = invalid_score

    with pytest.raises(ValueError, match=rf"{objective}.*\[0, 1\]"):
        prepare_d2d_training_data(frame, config)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("speed_1", 999.0, "within configured bounds"),
        ("time_1", np.inf, "input values must be complete and finite"),
        ("Uniformity score", np.nan, "objective values must be complete and finite"),
        (
            "Optoelectronic score",
            -10.0,
            "strictly dominate the fixed reference point",
        ),
    ],
)
def test_training_data_rejects_bounds_finite_and_dominance_failures(
    config, column: str, value: float, message: str
) -> None:
    frame = _synthetic_r0_frame(config)
    frame.loc[7, column] = value

    with pytest.raises(ValueError, match=message):
        prepare_d2d_training_data(frame, config)


def test_training_data_preserves_named_objective_order(config) -> None:
    frame = _synthetic_r0_frame(config)
    frame.loc[0, list(D2D_OBJECTIVE_COLUMNS)] = [0.61, -3.25, 0.87]

    training = prepare_d2d_training_data(frame, config)

    np.testing.assert_array_equal(training.Y_objectives[0], [0.61, -3.25, 0.87])
    manifest = training.manifest_frame()
    required_metadata = {
        "campaign_id",
        "round",
        "sample_id",
        "candidate_id",
        "row_role",
        "replicate_group",
        "replicate_number",
        "candidate_status",
        "include_in_model",
        "measurement_provenance",
        "off_grid_exception",
        "exclusion_reason",
    }
    assert required_metadata <= set(manifest.columns)
    assert manifest["campaign_id"].eq(config.raw["campaign_id"]).all()
    assert manifest["round"].eq("R0").all()
    assert manifest["candidate_id"].is_unique
    assert manifest.loc[0, "off_grid_exception"]
    assert manifest.loc[0, "off_grid_exception_reason"]
    assert list(manifest.columns[-3:]) == list(D2D_OBJECTIVE_COLUMNS)
    np.testing.assert_array_equal(
        manifest.loc[0, list(D2D_OBJECTIVE_COLUMNS)].to_numpy(dtype=float),
        [0.61, -3.25, 0.87],
    )


def test_five_candidates_expand_to_three_replicates_each(config) -> None:
    candidates = _five_synthetic_candidates(config)

    expanded = expand_candidates_to_replicates(candidates)

    assert expanded.shape[0] == 15
    assert expanded["candidate_id"].drop_duplicates().tolist() == [
        "R1-C01",
        "R1-C02",
        "R1-C03",
        "R1-C04",
        "R1-C05",
    ]
    assert expanded["execution_id"].is_unique
    assert expanded["measurement_provenance"].eq("pending_measurement").all()
    assert not expanded["off_grid_exception"].any()
    assert expanded.groupby("candidate_id", sort=False).size().tolist() == [3] * 5
    assert (
        expanded.groupby("candidate_id", sort=False)["replicate_number"]
        .agg(list)
        .tolist()
        == [[1, 2, 3]] * 5
    )
    assert expanded["replicate_group"].equals(expanded["candidate_id"])
    assert expanded["candidate_status"].eq(D2D_DEBUG_WATERMARK).all()
    assert expanded["campaign_id"].eq(config.raw["campaign_id"]).all()
    assert expanded["debug_only"].eq(True).all()  # noqa: E712
    assert expanded["approved_for_experiment"].eq(False).all()  # noqa: E712
    assert expanded.loc[:, list(D2D_OBJECTIVE_COLUMNS)].isna().all().all()
    for candidate_id, group in expanded.groupby("candidate_id", sort=False):
        source = candidates.iloc[int(candidate_id[-2:]) - 1]
        expected = np.repeat(
            source.loc[list(D2D_INPUT_COLUMNS)].to_numpy(dtype=float)[None, :],
            3,
            axis=0,
        )
        np.testing.assert_array_equal(
            group.loc[:, list(D2D_INPUT_COLUMNS)].to_numpy(dtype=float), expected
        )


def test_candidate_expansion_rejects_duplicate_candidate_ids(config) -> None:
    candidates = _five_synthetic_candidates(config)
    candidates["candidate_id"] = [
        "R1-C01",
        "R1-C01",
        "R1-C03",
        "R1-C04",
        "R1-C05",
    ]

    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        expand_candidates_to_replicates(candidates)


@pytest.mark.parametrize("invalid_count", [True, 3.5, 0, -1])
def test_replicate_helpers_reject_nonpositive_or_noninteger_counts(
    config, invalid_count: Any
) -> None:
    candidates, records = _measured_replicates(config)

    with pytest.raises(ValueError, match="replicates_per_condition"):
        expand_candidates_to_replicates(
            candidates, replicates_per_condition=invalid_count
        )
    with pytest.raises(ValueError, match="expected_replicates"):
        aggregate_replicate_objectives(records, expected_replicates=invalid_count)


def test_replicate_aggregation_reports_statistics_counts_and_sources(config) -> None:
    _, records = _measured_replicates(config)

    result = aggregate_replicate_objectives(records)

    assert result.warnings == ()
    assert result.frame.shape[0] == 5
    assert result.frame["replicate_group"].tolist() == [
        "R1-C01",
        "R1-C02",
        "R1-C03",
        "R1-C04",
        "R1-C05",
    ]
    assert result.frame["complete_replicate_set"].eq(True).all()  # noqa: E712
    assert result.frame["include_in_next_model"].eq(True).all()  # noqa: E712
    for candidate_index, row in result.frame.iterrows():
        source_index = candidate_index + 1
        assert row["source_sample_ids"] == "|".join(
            f"R1-C{source_index:02d}-R{replicate}" for replicate in (1, 2, 3)
        )
        expected_values = {
            "Uniformity score": np.asarray([0.10, 0.15, 0.20]) + source_index / 10.0,
            "Optoelectronic score": np.asarray([-2.0, -1.0, 0.0]) + source_index,
            "Thickness score": np.asarray([0.30, 0.40, 0.50]) + source_index / 20.0,
        }
        for objective, values in expected_values.items():
            assert row[f"{objective}_mean"] == pytest.approx(np.mean(values))
            assert row[f"{objective}_sample_std"] == pytest.approx(
                np.std(values, ddof=1)
            )
            assert row[f"{objective}_standard_error"] == pytest.approx(
                np.std(values, ddof=1) / np.sqrt(3)
            )
            assert row[f"{objective}_count"] == 3


def test_replicate_aggregation_rejects_input_mismatch(config) -> None:
    _, records = _measured_replicates(config)
    records.loc[1, "speed_1"] = float(config.design.var_array[0][6])

    with pytest.raises(ValueError, match="contains different requested input"):
        aggregate_replicate_objectives(records)


@pytest.mark.parametrize("objective", ["Uniformity score", "Thickness score"])
def test_replicate_aggregation_rejects_bounded_score_out_of_range(
    config, objective: str
) -> None:
    _, records = _measured_replicates(config)
    records.loc[0, objective] = 1.01

    with pytest.raises(ValueError, match=rf"{objective!r} must remain in \[0, 1\]"):
        aggregate_replicate_objectives(records)


@pytest.mark.parametrize("duplicate_field", ["execution_id", "replicate_number"])
def test_replicate_aggregation_rejects_duplicate_execution_metadata(
    config, duplicate_field: str
) -> None:
    _, records = _measured_replicates(config)
    records.loc[1, duplicate_field] = records.loc[0, duplicate_field]

    with pytest.raises(
        ValueError,
        match=f"{duplicate_field} values must be unique|duplicate {duplicate_field}",
    ):
        aggregate_replicate_objectives(records)


def test_incomplete_single_replicate_has_nan_sample_std_and_sem(config) -> None:
    _, records = _measured_replicates(config)
    single = records.iloc[[0]].copy()

    result = aggregate_replicate_objectives(single)

    assert len(result.warnings) == 1
    assert "incomplete" in result.warnings[0]
    row = result.frame.iloc[0]
    assert bool(row["complete_replicate_set"]) is False
    assert bool(row["include_in_next_model"]) is False
    assert row["source_sample_ids"] == "R1-C01-R1"
    for objective in D2D_OBJECTIVE_COLUMNS:
        assert row[f"{objective}_count"] == 1
        assert row[f"{objective}_mean"] == pytest.approx(single.iloc[0][objective])
        assert np.isnan(row[f"{objective}_sample_std"])
        assert np.isnan(row[f"{objective}_standard_error"])


def test_r0_and_aggregated_r1_combine_as_twenty_conditions(config) -> None:
    training = prepare_d2d_training_data(_synthetic_r0_frame(config), config)
    candidates, records = _measured_replicates(config)
    aggregated = aggregate_replicate_objectives(records)

    X, Y = combine_r0_and_aggregated_r1(training, aggregated)

    assert X.shape == (20, 10)
    assert Y.shape == (20, 3)
    assert np.unique(X, axis=0).shape[0] == 20
    np.testing.assert_array_equal(X[:15], training.X_phys_all)
    np.testing.assert_array_equal(
        X[15:], candidates.loc[:, list(D2D_INPUT_COLUMNS)].to_numpy(dtype=float)
    )
    mean_columns = [f"{objective}_mean" for objective in D2D_OBJECTIVE_COLUMNS]
    np.testing.assert_allclose(
        Y[15:], aggregated.frame.loc[:, mean_columns].to_numpy(dtype=float)
    )


def test_combine_rejects_duplicate_condition_between_rounds(config) -> None:
    training = prepare_d2d_training_data(_synthetic_r0_frame(config), config)
    duplicate = training.X_phys_all[1]
    row: dict[str, Any] = {
        name: float(value) for name, value in zip(D2D_INPUT_COLUMNS, duplicate)
    }
    for objective, value in zip(D2D_OBJECTIVE_COLUMNS, [0.7, -1.0, 0.8]):
        row[f"{objective}_mean"] = value
    row["include_in_next_model"] = True
    aggregated = ReplicateAggregationResult(pd.DataFrame([row]), ())

    with pytest.raises(ValueError, match="duplicate recipes"):
        combine_r0_and_aggregated_r1(training, aggregated)
