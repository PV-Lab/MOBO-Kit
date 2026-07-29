from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest
import yaml

from mobo_kit.step2c_artifacts import (
    CONSENSUS_BATCH_FILE,
    DEBUG_WATERMARK,
    NO_STABLE_BATCH_REASON_FILE,
    REQUIRED_PLOT_DIRECTORIES,
    REQUIRED_PLOT_FILES,
    REQUIRED_CSV_COLUMNS,
    REQUIRED_TOP_LEVEL_FILES,
    Step2CArtifactContractError,
    validate_step2c_artifact_bundle,
)


STAMP_COLUMNS = [
    "debug_only",
    "approved_for_experiment",
    "approved_for_production",
    "candidate_status",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _canonical_mapping_sha256(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _stamped_payload(**values):
    return {
        **values,
        "debug_only": True,
        "approved_for_experiment": False,
        "approved_for_production": False,
        "candidate_status": DEBUG_WATERMARK,
    }


def _write_json(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_stamped_payload(**values), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(
    path: Path, *, row_count: int = 1, overrides: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_columns = REQUIRED_CSV_COLUMNS.get(path.name, ("value",))
    fieldnames = [*semantic_columns, *STAMP_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            row = {column: index for column in semantic_columns}
            for column, value in (overrides or {}).items():
                if column in row:
                    row[column] = value(index) if callable(value) else value
            row.update(
                {
                    "debug_only": True,
                    "approved_for_experiment": False,
                    "approved_for_production": False,
                    "candidate_status": DEBUG_WATERMARK,
                }
            )
            writer.writerow(row)


def _write_consensus_csv(path: Path, *, row_count: int = 5) -> None:
    semantic_columns = [
        "consensus_candidate_id",
        "selection_order",
        "distinct_nonbaseline_family_count",
        "all_grid_valid",
        "all_hard_distance_valid",
        *[
            value
            for dimension, name in enumerate(
                (
                    "speed_1",
                    "time_1",
                    "speed_2",
                    "time_2",
                    "precur_conc",
                    "precur_vol",
                    "anneal_temp",
                    "anneal_time",
                    "anti_vol",
                    "anti_time",
                )
            )
            for value in (name, f"medoid_grid_{dimension}", f"medoid_norm_{dimension}")
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*semantic_columns, *STAMP_COLUMNS])
        writer.writeheader()
        for index in range(row_count):
            row = {
                "consensus_candidate_id": f"DEBUG-{index + 1}",
                "selection_order": index + 1,
                "distinct_nonbaseline_family_count": 3,
                "all_grid_valid": True,
                "all_hard_distance_valid": True,
                "debug_only": True,
                "approved_for_experiment": False,
                "approved_for_production": False,
                "candidate_status": DEBUG_WATERMARK,
            }
            for dimension, name in enumerate(
                (
                    "speed_1",
                    "time_1",
                    "speed_2",
                    "time_2",
                    "precur_conc",
                    "precur_vol",
                    "anneal_temp",
                    "anneal_time",
                    "anti_vol",
                    "anti_time",
                )
            ):
                row[name] = index
                row[f"medoid_grid_{dimension}"] = index
                row[f"medoid_norm_{dimension}"] = index / 4.0
            writer.writerow(row)


def _write_png(path: Path, *, description: str | None = DEBUG_WATERMARK) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngImagePlugin.PngInfo()
    if description is not None:
        metadata.add_text("Description", description)
    Image.new("RGB", (12, 8), color="white").save(path, pnginfo=metadata)


def _rewrite_manifest(output_dir: Path, update) -> None:
    path = output_dir / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_csv_rows(path: Path, update) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    update(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_bundle(
    tmp_path: Path,
    *,
    consensus: bool = False,
    inside_configured_root: bool = True,
) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    configured_relative = Path("local_outputs/d2d_step2c_robustness")
    parent = (
        repository / configured_relative
        if inside_configured_root
        else repository / "some_other_output_root"
    )
    output_dir = parent / "run-001"
    output_dir.mkdir(parents=True)

    workbook = repository / "local_inputs" / "private_campaign_input.xlsx"
    workbook.parent.mkdir()
    workbook.write_bytes(b"immutable workbook fixture")
    fixed_mtime = 1_700_000_000_123_456_700
    os.utime(workbook, ns=(fixed_mtime, fixed_mtime))
    workbook_hash = _sha256(workbook)
    workbook_mtime = workbook.stat().st_mtime_ns

    config_directory = repository / "configs"
    config_directory.mkdir()
    input_names = (
        "speed_1",
        "time_1",
        "speed_2",
        "time_2",
        "precur_conc",
        "precur_vol",
        "anneal_temp",
        "anneal_time",
        "anti_vol",
        "anti_time",
    )
    base_config_path = config_directory / "synthetic_step2b.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "inputs": [
                    {"name": name, "start": 0, "stop": 4, "step": 1, "decimals": 0}
                    for name in input_names
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = {
        "run_mode": "debug",
        "approved_for_experiment": False,
        "approved_for_production": False,
        "debug_watermark": DEBUG_WATERMARK,
        "base_step2b_config": "configs/synthetic_step2b.yaml",
        "candidate_search": {"nested_unique_sizes": [8, 16, 32, 64]},
        "observation_influence": {"omitted_sample_ids": [1, 2, 3]},
        "robust_regions": {
            "clustering": "agglomerative_complete_link",
            "primary_distance_threshold": 0.15,
            "consensus_batch_size": 5,
            "consensus_required_criteria": {
                "largest_two_nested_regional_matches_within_0_15": 4,
                "largest_two_nested_mean_matched_distance_max": 0.10,
                "minimum_region_family_coverage": 3,
                "require_grid_valid": True,
                "require_hard_distance_valid": True,
            },
        },
        "execution_modes": {
            "fast": {
                "nested_unique_sizes": [2, 4, 8, 16],
                "omitted_sample_ids": [1],
            },
            "full": {
                "nested_unique_sizes": [8, 16, 32, 64],
                "omitted_sample_ids": [1, 2, 3],
            },
        },
        "outputs": {"root": configured_relative.as_posix()},
    }
    source_config_path = config_directory / "d2d_step2c_debug.yaml"
    source_config_path.write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "resolved_debug_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    for name in REQUIRED_TOP_LEVEL_FILES:
        path = output_dir / name
        if name in {"resolved_debug_config.yaml", "run_manifest.json"}:
            continue
        if name == "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt":
            path.write_text(f"{DEBUG_WATERMARK}\n", encoding="utf-8")
            continue
        if path.suffix == ".json":
            _write_json(path, artifact=name)
        elif path.suffix == ".csv":
            _write_csv(path, row_count=0 if name == "model_fit_warnings.csv" else 1)

    evidence_count = 5 if consensus else 4
    _write_csv(
        output_dir / "nested_pool_convergence_summary.csv",
        overrides={
            "comparison_type": "prefix_vs_largest",
            "reference_pool_size": 64,
            "comparison_pool_size": 32,
            "mean_matched_distance": 0.0,
            "regional_matches_within_0.15": 5,
        },
    )
    common_region_overrides = {
        "region_id": lambda index: f"REGION-{index + 1:03d}",
        "member_count": 1,
        "distinct_nonbaseline_family_count": 3,
        "all_grid_valid": True,
        "all_hard_distance_valid": True,
        "family_weighted_persistence": 1.0,
        **{f"medoid_grid_{dimension}": lambda index: index for dimension in range(10)},
        **{
            f"medoid_norm_{dimension}": lambda index: index / 4.0
            for dimension in range(10)
        },
    }
    _write_csv(
        output_dir / "robust_regions.csv",
        row_count=evidence_count,
        overrides=common_region_overrides,
    )
    _write_csv(
        output_dir / "r1_robust_shortlist_debug.csv",
        row_count=evidence_count,
        overrides={
            **common_region_overrides,
            "shortlist_id": lambda index: f"R1-RS{index + 1:02d}",
            "lower_boundary_dimensions": "",
            "upper_boundary_dimensions": "",
            **{name: lambda index: index for name in input_names},
        },
    )

    consensus_checks = {
        "full_mode_eligible_for_consensus": True,
        "largest_two_nested_regional_matches": True,
        "largest_two_nested_mean_matched_distance": True,
        "five_regions_cover_three_core_families": consensus,
        "exact_five_candidates": consensus,
        "chosen_regions_cover_three_core_families": consensus,
        "finite_and_bounded": consensus,
        "unique_and_on_grid": consensus,
        "chosen_pairwise_hard_distance_valid": consensus,
        "debug_only": consensus,
        "experimental_approval_false": consensus,
        "production_approval_false": consensus,
    }
    consensus_observed = {
        "largest_two_regional_matches_within_0_15": 5,
        "largest_two_mean_matched_distance": 0.0,
        "regions_with_minimum_family_coverage": evidence_count,
        "consensus_candidate_count": evidence_count,
        "chosen_regions_family_qualified": consensus,
        "chosen_candidates_finite_and_bounded": consensus,
        "chosen_candidates_unique_and_on_grid": consensus,
        "chosen_pairwise_minimum_distance": (
            (10 * (1 / 4) ** 2) ** 0.5 if consensus else 0.0
        ),
        "required_pairwise_minimum_distance": 0.15,
        "chosen_hard_distance_valid": consensus,
        "full_mode_eligible_for_consensus": True,
    }
    _write_json(
        output_dir / "run_manifest.json",
        schema_version="d2d-step2c-robustness-run-v1",
        method_version="synthetic-test-v1",
        git_commit="1" * 40,
        git_dirty=True,
        git_status_at_start=["synthetic fixture"],
        step2b_checkpoint="2" * 40,
        config_path=str(source_config_path),
        config_sha256=_sha256(source_config_path),
        resolved_config_hash=_canonical_mapping_sha256(config),
        workbook_path=workbook.relative_to(repository).as_posix(),
        workbook_sha256_before=workbook_hash,
        workbook_sha256_after=workbook_hash,
        workbook_mtime_ns_before=workbook_mtime,
        workbook_mtime_ns_after=workbook_mtime,
        source_workbook_modified=False,
        mode="full",
        consensus_passed=consensus,
        objective_order=[
            "uniformity_score",
            "optoelectronic_score",
            "thickness_score",
        ],
        objective_source_columns=[
            "Uniformity score",
            "Optoelectronic score",
            "Thickness score",
        ],
        reference_point=[-0.01, -10.0, -0.01],
        objective_bounds=[[0.0, 1.0], [None, None], [0.0, 1.0]],
        moment_method="analytic_identity",
        analytic_mc_comparison={"analytic_mc_debug_check_passed": True},
        sobol_seeds=[73, 137, 911],
        nested_pool_sizes=[8, 16, 32, 64],
        pool_prefix_hashes={
            "73": {str(size): "A" * 64 for size in (8, 16, 32, 64)},
            "137": {"64": "B" * 64},
            "911": {"64": "C" * 64},
        },
        local_refinement={"anchors_per_selection_step": 2, "max_sweeps": 2},
        beta_values=[1.0, 4.0, 9.0],
        bound_policies=["none", "clip_ucb"],
        local_penalty_variants=[
            {"label": label}
            for label in (
                "no_soft_no_hard",
                "no_soft_hard_0_15",
                "radius_0_15",
                "radius_0_25",
                "radius_0_35",
            )
        ],
        model_variants=[
            {"name": "dim_scaled_prior"},
            {"name": "conservative"},
        ],
        influence_common_pool_hash="D" * 64,
        influence_common_pool_size=32,
        influence_omitted_sample_ids=[1, 2, 3],
        robust_region_clustering="agglomerative_complete_link",
        robust_region_threshold=0.15,
        robust_region_sensitivity_counts={
            "threshold_0.10": 6,
            "threshold_0.20": 4,
        },
        robust_region_count=evidence_count,
        shortlist_count=evidence_count,
        stability_criteria={
            "consensus_batch_size": 5,
            "regional_match_threshold": 0.15,
            "largest_two_nested_regional_match_minimum": 4,
            "largest_two_nested_mean_matched_distance_maximum": 0.10,
            "minimum_nonbaseline_core_family_coverage": 3,
            "required_pairwise_minimum_distance": 0.15,
            "require_finite_and_bounded": True,
            "require_unique_and_on_grid": True,
            "require_debug_only_and_approval_false": True,
        },
        consensus_checks=consensus_checks,
        consensus_observed=consensus_observed,
        runtime_versions={
            name: "test"
            for name in (
                "python",
                "numpy",
                "pandas",
                "scipy",
                "scikit_learn",
                "matplotlib",
                "torch",
                "botorch",
                "gpytorch",
            )
        },
        hardware={"logical_cpu_count": 1},
        phase_runtime_seconds={"synthetic": 0.1},
        runtime_seconds_total=0.2,
        known_uniformity_score_mismatch=True,
        uniformity_warning_count=1,
        control_assumption="measured_in_current_campaign",
        off_grid_control_exception=[{"sample_id": 1, "value": 12.0}],
        real_r2_proposal_generated=False,
        workbook_writeback_performed=False,
        output_directory=str(output_dir),
        public_summary_archive_requested=False,
        private_evidence_archive_requested=False,
    )
    if consensus:
        _write_consensus_csv(output_dir / CONSENSUS_BATCH_FILE, row_count=5)
    else:
        failed_checks = [
            name for name, passed in consensus_checks.items() if not passed
        ]
        _write_json(
            output_dir / NO_STABLE_BATCH_REASON_FILE,
            message="No stable batch in synthetic fixture.",
            checks=consensus_checks,
            failed_checks=failed_checks,
            observed=consensus_observed,
            consensus_passed=False,
        )
    for relative in REQUIRED_PLOT_FILES:
        _write_png(output_dir / relative)
    return repository, output_dir, workbook


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int, int]]:
    return {
        path: (_sha256(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    }


def test_valid_no_stable_bundle_returns_hash_map_and_is_read_only(tmp_path):
    repository, output_dir, workbook = _build_bundle(tmp_path)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    before = _snapshot([*files, workbook])

    result = validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    assert result.output_dir == output_dir.resolve()
    assert (
        result.ignored_output_root
        == (repository / "local_outputs/d2d_step2c_robustness").resolve()
    )
    assert result.conditional_artifact == NO_STABLE_BATCH_REASON_FILE
    assert result.workbook_path == workbook.resolve()
    assert result.workbook_sha256 == _sha256(workbook)
    assert result.workbook_mtime_ns == workbook.stat().st_mtime_ns
    assert "model_fit_warnings.csv" in result.csv_files_checked
    assert set(result.png_files_checked) == set(REQUIRED_PLOT_FILES)
    assert set(result.artifact_sha256) == {
        path.relative_to(output_dir).as_posix() for path in files
    }
    assert result.artifact_sha256["run_manifest.json"] == _sha256(
        output_dir / "run_manifest.json"
    )
    assert _snapshot([*files, workbook]) == before


def test_consensus_allows_only_explicit_debug_preview_replicate(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, consensus=True)
    preview = output_dir / "r1_replicate_debug_preview.csv"
    _write_csv(preview)

    result = validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    assert result.conditional_artifact == CONSENSUS_BATCH_FILE
    assert preview.name in result.csv_files_checked


def test_consensus_requires_full_mode_and_exactly_five_rows(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, consensus=True)
    _rewrite_manifest(output_dir, lambda payload: payload.__setitem__("mode", "fast"))
    with pytest.raises(Step2CArtifactContractError, match="forbidden outside a full"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    _rewrite_manifest(output_dir, lambda payload: payload.__setitem__("mode", "full"))
    _write_consensus_csv(output_dir / CONSENSUS_BATCH_FILE, row_count=4)
    with pytest.raises(Step2CArtifactContractError, match="exactly five"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_all_exact_required_top_level_names_are_enforced(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    (output_dir / "sobol_scramble_comparison.csv").unlink()

    with pytest.raises(
        Step2CArtifactContractError,
        match="Missing required top-level.*sobol_scramble_comparison.csv",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


@pytest.mark.parametrize("conditional_state", ["both", "neither"])
def test_exactly_one_conditional_artifact_is_required(tmp_path, conditional_state):
    repository, output_dir, _ = _build_bundle(tmp_path)
    if conditional_state == "both":
        _write_csv(output_dir / CONSENSUS_BATCH_FILE)
    else:
        (output_dir / NO_STABLE_BATCH_REASON_FILE).unlink()

    with pytest.raises(Step2CArtifactContractError, match="Exactly one conditional"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_required_plot_topic_directories_and_files_are_enforced(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    missing = output_dir / "plots/search_convergence/boundary_enrichment.png"
    missing.unlink()

    with pytest.raises(
        Step2CArtifactContractError, match="Missing required plot-topic artifact"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    assert set(REQUIRED_PLOT_DIRECTORIES) == {
        "plots/model_validation",
        "plots/search_convergence",
        "plots/bounded_utility",
        "plots/local_penalty",
        "plots/influence",
        "plots/robust_regions",
    }


def test_zero_row_csv_requires_stamp_headers_but_not_stamp_rows(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    empty_path = output_dir / "model_fit_warnings.csv"
    empty_path.write_text("warning,debug_only\n", encoding="utf-8")

    with pytest.raises(
        Step2CArtifactContractError, match="missing debug stamp columns"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


@pytest.mark.parametrize(
    "relative, field, bad_value, message",
    [
        ("score_validation.csv", "debug_only", False, "debug_only=true"),
        (
            "training_row_manifest.csv",
            "approved_for_experiment",
            True,
            "approved_for_experiment=false",
        ),
        (
            "robust_regions.csv",
            "candidate_status",
            "NOT A DEBUG WATERMARK",
            "exact debug watermark",
        ),
    ],
)
def test_every_csv_row_must_carry_exact_debug_approval_stamps(
    tmp_path, relative, field, bad_value, message
):
    repository, output_dir, _ = _build_bundle(tmp_path)
    path = output_dir / relative
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["value", *STAMP_COLUMNS])
        writer.writeheader()
        row = {
            "value": 1,
            "debug_only": True,
            "approved_for_experiment": False,
            "approved_for_production": False,
            "candidate_status": DEBUG_WATERMARK,
        }
        row[field] = bad_value
        writer.writerow(row)

    with pytest.raises(Step2CArtifactContractError, match=message):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_additional_nested_csv_and_every_json_are_also_stamped(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    additional = output_dir / "plots" / "review" / "extra.csv"
    additional.parent.mkdir()
    additional.write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(
        Step2CArtifactContractError, match="extra.csv.*missing debug stamp columns"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    additional.unlink()
    audit = output_dir / "workbook_audit.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["approved_for_production"] = "false"
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        Step2CArtifactContractError, match="approved_for_production=false"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


@pytest.mark.parametrize("description", [None, "DEBUG ONLY"])
def test_every_png_requires_exact_description_watermark(tmp_path, description):
    repository, output_dir, _ = _build_bundle(tmp_path)
    path = output_dir / REQUIRED_PLOT_FILES[0]
    _write_png(path, description=description)

    with pytest.raises(
        Step2CArtifactContractError, match="PNG Description.*exact debug watermark"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_replicate_worklist_is_forbidden_without_consensus(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    _write_csv(output_dir / "r1_replicate_debug_preview.csv")

    with pytest.raises(
        Step2CArtifactContractError, match="forbidden without.*consensus"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_consensus_replicate_file_must_be_named_debug_preview(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, consensus=True)
    _write_csv(output_dir / "r1_replicate_worklist.csv")

    with pytest.raises(
        Step2CArtifactContractError, match="explicitly named as a debug preview"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_output_must_be_below_resolved_config_ignored_root(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, inside_configured_root=False)

    with pytest.raises(
        Step2CArtifactContractError,
        match="strict descendant of configured ignored root",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_configured_output_root_cannot_escape_repository(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    config_path = output_dir / "resolved_debug_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["outputs"]["root"] = "../outside-repository"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(
        Step2CArtifactContractError, match="strict descendant of repository_root"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


@pytest.mark.parametrize(
    "update, message",
    [
        (
            lambda payload: payload.__setitem__("workbook_sha256_after", "0" * 64),
            "SHA-256 changed",
        ),
        (
            lambda payload: payload.__setitem__(
                "workbook_mtime_ns_after", payload["workbook_mtime_ns_before"] + 1
            ),
            "mtime changed",
        ),
        (
            lambda payload: payload.__setitem__("source_workbook_modified", True),
            "source_workbook_modified.*false",
        ),
    ],
)
def test_manifest_requires_unchanged_workbook_before_after_proof(
    tmp_path, update, message
):
    repository, output_dir, _ = _build_bundle(tmp_path)
    _rewrite_manifest(output_dir, update)

    with pytest.raises(Step2CArtifactContractError, match=message):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_live_workbook_hash_must_match_manifest_proof(tmp_path):
    repository, output_dir, workbook = _build_bundle(tmp_path)
    workbook.write_bytes(b"changed after completed run")

    with pytest.raises(
        Step2CArtifactContractError, match="live source workbook SHA-256"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_live_workbook_mtime_must_match_manifest_proof(tmp_path):
    repository, output_dir, workbook = _build_bundle(tmp_path)
    recorded = workbook.stat().st_mtime_ns
    os.utime(workbook, ns=(recorded + 100, recorded + 100))

    with pytest.raises(Step2CArtifactContractError, match="live source workbook mtime"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_manifest_requires_complete_step2c_provenance(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    _rewrite_manifest(output_dir, lambda payload: payload.pop("sobol_seeds"))

    with pytest.raises(
        Step2CArtifactContractError, match="missing required provenance.*sobol_seeds"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_resolved_config_hash_is_recomputed_from_bundled_yaml(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    path = output_dir / "resolved_debug_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["robust_regions"]["consensus_batch_size"] = 4
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        Step2CArtifactContractError, match="canonical resolved_debug_config.*SHA-256"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_nested_convergence_claim_is_cross_checked_against_csv(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    path = output_dir / "nested_pool_convergence_summary.csv"
    _rewrite_csv_rows(
        path,
        lambda rows: rows[0].__setitem__("regional_matches_within_0.15", "3"),
    )

    with pytest.raises(
        Step2CArtifactContractError,
        match="consensus check largest_two_nested_regional_matches disagrees",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_manifest_cannot_weaken_resolved_consensus_criteria(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    _rewrite_manifest(
        output_dir,
        lambda payload: payload["stability_criteria"].__setitem__(
            "largest_two_nested_regional_match_minimum", 1
        ),
    )

    with pytest.raises(
        Step2CArtifactContractError,
        match="stability criterion.*does not match the resolved debug config",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_qualifying_region_count_is_cross_checked_against_csv(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    path = output_dir / "robust_regions.csv"
    _rewrite_csv_rows(path, lambda rows: rows.pop())

    with pytest.raises(
        Step2CArtifactContractError,
        match="robust_region_count disagrees",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_required_csvs_require_semantic_columns_and_nonempty_rows(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    _write_csv(output_dir / "model_validation_summary.csv", row_count=0)

    with pytest.raises(
        Step2CArtifactContractError,
        match="model_validation_summary.csv must contain at least one",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_consensus_requires_recipe_and_independently_validates_uniqueness(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, consensus=True)
    _write_csv(output_dir / CONSENSUS_BATCH_FILE, row_count=5)
    with pytest.raises(
        Step2CArtifactContractError, match="missing safety/recipe columns"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    _write_consensus_csv(output_dir / CONSENSUS_BATCH_FILE, row_count=5)
    path = output_dir / CONSENSUS_BATCH_FILE
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for key in rows[0]:
        if key in STAMP_COLUMNS or key in {
            "consensus_candidate_id",
            "selection_order",
        }:
            continue
        rows[1][key] = rows[0][key]
    rows[1]["consensus_candidate_id"] = "DEBUG-2"
    rows[1]["selection_order"] = "2"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(Step2CArtifactContractError, match="must be unique exact grid"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_consensus_recipes_must_match_eligible_shortlist_head(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, consensus=True)
    path = output_dir / CONSENSUS_BATCH_FILE

    def change_to_different_valid_recipe(rows):
        rows[0]["speed_1"] = "1"
        rows[0]["medoid_grid_0"] = "1"
        rows[0]["medoid_norm_0"] = "0.25"

    _rewrite_csv_rows(path, change_to_different_valid_recipe)

    with pytest.raises(
        Step2CArtifactContractError,
        match="do not match the eligible shortlist head",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_no_stable_reason_must_name_failed_manifest_checks(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    reason = output_dir / NO_STABLE_BATCH_REASON_FILE
    payload = json.loads(reason.read_text(encoding="utf-8"))
    original_failed = list(payload["failed_checks"])
    payload["failed_checks"] = original_failed[:-1]
    reason.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Step2CArtifactContractError, match="must name every failed"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)

    payload["failed_checks"] = original_failed
    payload["observed"] = {"fabricated": 999}
    reason.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Step2CArtifactContractError, match="observed values must match"):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_shortlist_regions_must_match_robust_region_evidence(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path, consensus=True)
    path = output_dir / "r1_robust_shortlist_debug.csv"
    _rewrite_csv_rows(
        path, lambda rows: rows[0].__setitem__("region_id", "NOT-IN-ROBUST-REGIONS")
    )

    with pytest.raises(
        Step2CArtifactContractError,
        match="shortlist region_id must reference robust_regions.csv",
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)


def test_unstamped_recipe_capable_formats_are_forbidden(tmp_path):
    repository, output_dir, _ = _build_bundle(tmp_path)
    (output_dir / "candidate_preview.xlsx").write_bytes(b"not permitted")

    with pytest.raises(
        Step2CArtifactContractError, match="Unsupported artifact format"
    ):
        validate_step2c_artifact_bundle(output_dir, repository_root=repository)
