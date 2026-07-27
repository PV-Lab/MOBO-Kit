from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from mobo_kit.candidate_pool import CandidatePool
from mobo_kit.d2d_campaign import D2D_INPUT_COLUMNS
from mobo_kit.d2d_step2c_robustness import (
    StudyBatch,
    _augment_model_summary_with_candidate_diagnostics,
    _boundary_enrichment_table,
    _build_public_summary_zip_staging,
    _declared_bound_flags,
    _penalty_tradeoff_table,
    _publish_validated_bundle_transaction,
)
from mobo_kit.step2c_artifacts import (
    DEBUG_WATERMARK,
    PUBLIC_SUMMARY_ARCHIVE_ROOT,
    PUBLIC_SUMMARY_CSV_FILES,
    PUBLIC_SUMMARY_FILES,
    PUBLIC_SUMMARY_MANIFEST_FILE,
    Step2CArtifactContractError,
    validate_step2c_public_summary_archive,
)


def _study_batch(
    penalty_label: str,
    *,
    score_scale: float = 1.0,
    penalty_factor: float = 1.0,
    changed_rows: int = 0,
) -> StudyBatch:
    dimension = len(D2D_INPUT_COLUMNS)
    normalized = np.full((5, dimension), 0.5, dtype=float)
    normalized[:, 0] = np.linspace(0.0, 1.0, 5)
    normalized[:changed_rows, 1] = 0.8
    grid = np.zeros((5, dimension), dtype=np.int64)
    grid[:, 0] = np.arange(5)
    scores = score_scale * np.linspace(5.0, 1.0, 5)
    return StudyBatch(
        run_id=penalty_label,
        run_family="penalty",
        core_run=True,
        grid_indices=grid,
        X_phys=normalized.copy(),
        X_norm=normalized,
        base_scores=scores,
        penalized_scores=scores * penalty_factor,
        model_variant="default_current",
        pool_seed=73,
        pool_size=32,
        pool_hash="A" * 64,
        beta=4.0,
        bound_policy="clip_ucb",
        penalty_label=penalty_label,
        refinement_enabled=True,
        refinement_runtime_seconds=0.1,
        proposal_runtime_seconds=0.2,
    )


def test_declared_bound_flags_keep_unbounded_objective_unflagged() -> None:
    values = np.asarray([-0.1, 0.5, 1.1])

    below, above, outside = _declared_bound_flags(values, (0.0, 1.0))
    np.testing.assert_array_equal(below, [True, False, False])
    np.testing.assert_array_equal(above, [False, False, True])
    np.testing.assert_array_equal(outside, [True, False, True])

    below, above, outside = _declared_bound_flags(values, (None, None))
    assert not below.any()
    assert not above.any()
    assert not outside.any()


def test_boundary_enrichment_separates_lower_and_upper_endpoints() -> None:
    dimension = len(D2D_INPUT_COLUMNS)
    pool_norm = np.full((4, dimension), 0.5, dtype=float)
    pool_norm[:, 0] = [0.0, 1.0, 0.5, 0.0]
    pool = CandidatePool(
        grid_indices=np.zeros((4, dimension), dtype=np.int64),
        X_phys=pool_norm.copy(),
        X_norm=pool_norm,
        seed=73,
        draws=4,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )
    selected = _study_batch("no_soft_no_hard")

    table = _boundary_enrichment_table(
        pool,
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        selected,
        comparison_batches=(),
    )
    pool_row = table[
        (table["input_name"] == D2D_INPUT_COLUMNS[0]) & (table["group"] == "pool")
    ].iloc[0]
    selected_row = table[
        (table["input_name"] == D2D_INPUT_COLUMNS[0])
        & (table["group"] == "selected_batch")
    ].iloc[0]

    assert pool_row["lower_boundary_count"] == 2
    assert pool_row["upper_boundary_count"] == 1
    assert pool_row["boundary_count"] == 3
    assert pool_row["lower_boundary_rate"] == pytest.approx(0.5)
    assert pool_row["upper_boundary_rate"] == pytest.approx(0.25)
    assert selected_row["lower_boundary_count"] == 1
    assert selected_row["upper_boundary_count"] == 1
    assert selected_row["lower_enrichment_ratio_vs_pool"] == pytest.approx(0.4)
    assert selected_row["upper_enrichment_ratio_vs_pool"] == pytest.approx(0.8)


def test_candidate_model_summary_counts_declared_support_extrapolation() -> None:
    summary = pd.DataFrame(
        {
            "variant_name": ["default_current"] * 3,
            "objective_index": [0, 1, 2],
            "objective_name": ["uniformity", "optoelectronic", "thickness"],
        }
    )
    candidate_rows = pd.DataFrame(
        {
            "run_id": ["baseline"] * 5,
            "model_variant": ["default_current"] * 5,
            "pred_mean_0": [-0.1, 0.2, 0.4, 0.8, 1.1],
            "pred_mean_1": [-20.0, -5.0, 0.0, 5.0, 20.0],
            "pred_mean_2": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )
    training_y = np.asarray(
        [
            [0.1, -10.0, 0.2],
            [0.3, -8.0, 0.4],
            [0.5, -6.0, 0.6],
            [0.7, -4.0, 0.8],
            [0.9, -2.0, 1.0],
        ]
    )

    result = _augment_model_summary_with_candidate_diagnostics(
        summary,
        candidate_rows,
        training_y,
        ((0.0, 1.0), (None, None), (0.0, 1.0)),
    )

    counts = result.set_index("objective_index")[
        "selected_prediction_outside_declared_bounds_count"
    ]
    assert counts.to_dict() == {0: 2.0, 1: 0.0, 2: 0.0}


def test_penalty_tradeoff_covers_all_variants_without_hard_relaxation() -> None:
    batches = (
        _study_batch("no_soft_no_hard"),
        _study_batch("no_soft_hard_0_15", score_scale=0.99),
        _study_batch("radius_0_15", score_scale=0.98, penalty_factor=0.9995),
        _study_batch(
            "radius_0_25",
            score_scale=0.96,
            penalty_factor=0.95,
            changed_rows=1,
        ),
        _study_batch(
            "radius_0_35",
            score_scale=0.90,
            penalty_factor=0.80,
            changed_rows=2,
        ),
    )
    observed = np.full((2, len(D2D_INPUT_COLUMNS)), 0.5)

    table = _penalty_tradeoff_table(batches, observed_norm=observed)

    assert table["penalty_label"].tolist() == [
        "no_soft_no_hard",
        "no_soft_hard_0_15",
        "radius_0_15",
        "radius_0_25",
        "radius_0_35",
    ]
    assert not table["hard_distance_relaxed"].any()
    reference_labels = table.set_index("penalty_label")[
        "comparison_reference_penalty_label"
    ]
    assert reference_labels["no_soft_hard_0_15"] == "no_soft_no_hard"
    assert reference_labels["radius_0_25"] == "no_soft_hard_0_15"
    classifications = table.set_index("penalty_label")[
        "penalty_activity_classification"
    ]
    assert classifications["radius_0_15"].startswith("implemented but effectively")
    assert classifications["radius_0_25"] == (
        "active but only modestly changes diversity"
    )
    assert classifications["radius_0_35"] == ("active and materially changes diversity")


def _publication_paths(tmp_path):
    destination = tmp_path / "published"
    staging = tmp_path / ".published.staging"
    backup = tmp_path / ".published.backup"
    zip_path = tmp_path / "published.zip"
    zip_staging = tmp_path / ".published.zip.staging"
    zip_backup = tmp_path / ".published.zip.backup"
    return destination, staging, backup, zip_path, zip_staging, zip_backup


def test_atomic_publish_restores_previous_bundle_when_staging_rename_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, staging, backup, zip_path, zip_staging, zip_backup = (
        _publication_paths(tmp_path)
    )
    destination.mkdir()
    staging.mkdir()
    (destination / "original.txt").write_text("original", encoding="utf-8")
    (staging / "replacement.txt").write_text("replacement", encoding="utf-8")
    zip_path.write_text("original zip", encoding="utf-8")
    zip_staging.write_text("replacement zip", encoding="utf-8")
    real_rename = type(staging).rename

    def fail_staging_rename(self, target):
        if self == staging:
            raise OSError("injected staging rename failure")
        return real_rename(self, target)

    monkeypatch.setattr(type(staging), "rename", fail_staging_rename)

    with pytest.raises(OSError, match="injected"):
        _publish_validated_bundle_transaction(
            staging,
            destination,
            backup,
            zip_staging=zip_staging,
            zip_path=zip_path,
            zip_backup=zip_backup,
            overwrite=True,
            publish_zip=True,
            validate_published=lambda: None,
        )

    assert (destination / "original.txt").read_text(encoding="utf-8") == "original"
    assert zip_path.read_text(encoding="utf-8") == "original zip"
    assert not backup.exists()
    assert not zip_backup.exists()
    assert (staging / "replacement.txt").is_file()
    assert not zip_staging.exists()


def test_atomic_publish_restores_directory_and_zip_when_zip_replace_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination, staging, backup, zip_path, zip_staging, zip_backup = (
        _publication_paths(tmp_path)
    )
    destination.mkdir()
    staging.mkdir()
    (destination / "original.txt").write_text("original", encoding="utf-8")
    (staging / "replacement.txt").write_text("replacement", encoding="utf-8")
    zip_path.write_text("original zip", encoding="utf-8")
    zip_staging.write_text("replacement zip", encoding="utf-8")

    def fail_zip_replace(source, target):
        if source == zip_staging and target == zip_path:
            raise OSError("injected ZIP publication failure")
        raise AssertionError("Unexpected os.replace call")

    monkeypatch.setattr("mobo_kit.d2d_step2c_robustness.os.replace", fail_zip_replace)

    with pytest.raises(OSError, match="ZIP publication"):
        _publish_validated_bundle_transaction(
            staging,
            destination,
            backup,
            zip_staging=zip_staging,
            zip_path=zip_path,
            zip_backup=zip_backup,
            overwrite=True,
            publish_zip=True,
            validate_published=lambda: None,
        )

    assert (destination / "original.txt").read_text(encoding="utf-8") == "original"
    assert zip_path.read_text(encoding="utf-8") == "original zip"
    assert not backup.exists()
    assert not zip_backup.exists()
    assert not staging.exists()
    assert not zip_staging.exists()


def test_atomic_publish_restores_both_originals_when_final_validation_fails(
    tmp_path,
) -> None:
    destination, staging, backup, zip_path, zip_staging, zip_backup = (
        _publication_paths(tmp_path)
    )
    destination.mkdir()
    staging.mkdir()
    (destination / "original.txt").write_text("original", encoding="utf-8")
    (staging / "replacement.txt").write_text("replacement", encoding="utf-8")
    zip_path.write_text("original zip", encoding="utf-8")
    zip_staging.write_text("replacement zip", encoding="utf-8")

    def fail_final_validation():
        assert (destination / "replacement.txt").is_file()
        assert zip_path.read_text(encoding="utf-8") == "replacement zip"
        raise RuntimeError("injected final validation failure")

    with pytest.raises(RuntimeError, match="final validation"):
        _publish_validated_bundle_transaction(
            staging,
            destination,
            backup,
            zip_staging=zip_staging,
            zip_path=zip_path,
            zip_backup=zip_backup,
            overwrite=True,
            publish_zip=True,
            validate_published=fail_final_validation,
        )

    assert (destination / "original.txt").read_text(encoding="utf-8") == "original"
    assert zip_path.read_text(encoding="utf-8") == "original zip"
    assert not backup.exists()
    assert not zip_backup.exists()
    assert not staging.exists()
    assert not zip_staging.exists()


def test_atomic_publish_without_zip_removes_stale_zip_only_after_validation(
    tmp_path,
) -> None:
    destination, staging, backup, zip_path, zip_staging, zip_backup = (
        _publication_paths(tmp_path)
    )
    destination.mkdir()
    staging.mkdir()
    (destination / "original.txt").write_text("original", encoding="utf-8")
    (staging / "replacement.txt").write_text("replacement", encoding="utf-8")
    zip_path.write_text("stale zip", encoding="utf-8")

    def validate():
        assert (destination / "replacement.txt").is_file()
        assert not zip_path.exists()
        assert zip_backup.read_text(encoding="utf-8") == "stale zip"
        return "validated"

    result = _publish_validated_bundle_transaction(
        staging,
        destination,
        backup,
        zip_staging=zip_staging,
        zip_path=zip_path,
        zip_backup=zip_backup,
        overwrite=True,
        publish_zip=False,
        validate_published=validate,
    )

    assert result == "validated"
    assert (destination / "replacement.txt").is_file()
    assert not zip_path.exists()
    assert not backup.exists()
    assert not zip_backup.exists()


def _public_summary_source(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "full_private_bundle"
    source.mkdir()
    for relative in PUBLIC_SUMMARY_CSV_FILES:
        pd.DataFrame(
            {
                "metric_name": ["aggregate"],
                "metric_value": [1.0],
                "debug_only": [True],
                "approved_for_experiment": [False],
                "approved_for_production": [False],
                "candidate_status": [DEBUG_WATERMARK],
            }
        ).to_csv(source / relative, index=False)
    private_manifest = {
        "method_version": "synthetic-public-export-test-v1",
        "mode": "full",
        "input_data_kind": "private_pinned_workbook",
        "git_commit": "a" * 40,
        "objective_order": [
            "uniformity_score",
            "optoelectronic_score",
            "thickness_score",
        ],
        "reference_point": [-0.01, -10.0, -0.01],
        "objective_bounds": [[0.0, 1.0], [None, None], [0.0, 1.0]],
        "moment_method": "analytic_identity",
        "robust_region_count": 7,
        "shortlist_count": 5,
        "consensus_passed": False,
        "consensus_checks": {"stable": False},
        "consensus_observed": {"region_count": 7},
        "runtime_versions": {"python": "test"},
        # The public builder must ignore all of these private-only values.
        "workbook_path": r"C:\Users\ExampleUser\private_campaign_input.xlsx",
        "config_path": r"C:\Users\ExampleUser\repo\private.yaml",
        "git_status_at_start": ["?? local_inputs/private_campaign_input.xlsx"],
        "hardware": {"profile": "ExampleUser"},
    }
    return source, private_manifest


def _rewrite_public_archive(
    archive_path: Path,
    update,
) -> None:
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        payloads = {info.filename: archive.read(info) for info in archive.infolist()}
    update(payloads)
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, payload in sorted(payloads.items()):
            archive.writestr(name, payload)


def _replace_public_csv_and_hash(
    payloads: dict[str, bytes], relative: str, replacement: bytes
) -> None:
    csv_entry = f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{relative}"
    payloads[csv_entry] = replacement
    manifest_entry = f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{PUBLIC_SUMMARY_MANIFEST_FILE}"
    manifest = json.loads(payloads[manifest_entry].decode("utf-8"))
    manifest["included_files"][relative] = (
        hashlib.sha256(replacement).hexdigest().upper()
    )
    payloads[manifest_entry] = json.dumps(manifest, indent=2, sort_keys=True).encode(
        "utf-8"
    )


def test_public_summary_builder_strips_private_provenance_and_recipes(
    tmp_path: Path,
) -> None:
    source, private_manifest = _public_summary_source(tmp_path)
    archive_path = tmp_path / "public_summary.zip"

    _build_public_summary_zip_staging(
        source,
        archive_path,
        private_manifest=private_manifest,
        overwrite=False,
    )
    validated = validate_step2c_public_summary_archive(
        archive_path, forbidden_profile_strings=("ExampleUser",)
    )

    assert set(validated.entry_sha256) == set(PUBLIC_SUMMARY_FILES)
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        names = archive.namelist()
        combined_text = "\n".join(
            archive.read(name).decode("utf-8") for name in names
        ).casefold()
    assert set(names) == {
        f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{relative}" for relative in PUBLIC_SUMMARY_FILES
    }
    for forbidden in (
        "c:\\users\\",
        "/users/",
        "/home/",
        "exampleuser",
        "private_campaign_input.xlsx",
        "workbook_path",
        "config_path",
        "git_status_at_start",
        "hardware",
        "r1_robust_shortlist_debug.csv",
        "study_candidates_long.csv",
    ):
        assert forbidden not in combined_text


@pytest.mark.parametrize(
    ("injected", "forbidden_profiles", "message"),
    [
        (r"C:\Users\ExampleUser\private.csv", (), "Windows absolute path"),
        ("/Users/example-user/private.csv", (), "profile path"),
        ("/home/example-user/private.csv", (), "profile path"),
        ("/private/location/data.csv", (), "POSIX absolute path"),
        ("private_campaign_input.xlsx", (), "workbook filename"),
        ("ExampleUser", ("ExampleUser",), "profile identifier"),
    ],
)
def test_public_summary_validator_rejects_paths_profiles_and_workbook_names(
    tmp_path: Path,
    injected: str,
    forbidden_profiles: tuple[str, ...],
    message: str,
) -> None:
    source, private_manifest = _public_summary_source(tmp_path)
    archive_path = tmp_path / "public_summary.zip"
    _build_public_summary_zip_staging(
        source,
        archive_path,
        private_manifest=private_manifest,
        overwrite=False,
    )
    relative = PUBLIC_SUMMARY_CSV_FILES[0]

    def inject(payloads: dict[str, bytes]) -> None:
        csv_entry = f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{relative}"
        replacement = (
            payloads[csv_entry]
            .decode("utf-8")
            .replace("aggregate", injected)
            .encode("utf-8")
        )
        _replace_public_csv_and_hash(payloads, relative, replacement)

    _rewrite_public_archive(archive_path, inject)

    with pytest.raises(Step2CArtifactContractError, match=message):
        validate_step2c_public_summary_archive(
            archive_path, forbidden_profile_strings=forbidden_profiles
        )


def test_public_summary_validator_rejects_recipe_header_and_private_file(
    tmp_path: Path,
) -> None:
    source, private_manifest = _public_summary_source(tmp_path)
    archive_path = tmp_path / "public_summary.zip"
    _build_public_summary_zip_staging(
        source,
        archive_path,
        private_manifest=private_manifest,
        overwrite=False,
    )
    relative = PUBLIC_SUMMARY_CSV_FILES[0]

    def add_recipe_header(payloads: dict[str, bytes]) -> None:
        csv_entry = f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{relative}"
        lines = payloads[csv_entry].decode("utf-8").splitlines()
        replacement = "\n".join(
            [f"speed_1,{lines[0]}", *[f"12,{line}" for line in lines[1:]]]
        ).encode("utf-8")
        _replace_public_csv_and_hash(payloads, relative, replacement)

    _rewrite_public_archive(archive_path, add_recipe_header)
    with pytest.raises(Step2CArtifactContractError, match="recipe/sample-level"):
        validate_step2c_public_summary_archive(archive_path)

    _build_public_summary_zip_staging(
        source,
        archive_path,
        private_manifest=private_manifest,
        overwrite=True,
    )

    def add_private_file(payloads: dict[str, bytes]) -> None:
        payloads[f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/r1_consensus_debug_batch.csv"] = (
            b"speed_1\n12\n"
        )

    _rewrite_public_archive(archive_path, add_private_file)
    with pytest.raises(Step2CArtifactContractError, match="private artifact"):
        validate_step2c_public_summary_archive(archive_path)


def test_atomic_publish_restores_public_and_private_zips_when_public_scan_fails(
    tmp_path: Path,
) -> None:
    destination, staging, backup, zip_path, zip_staging, zip_backup = (
        _publication_paths(tmp_path)
    )
    private_zip_path = tmp_path / "published_PRIVATE_EVIDENCE_DO_NOT_SHARE.zip"
    private_zip_staging = tmp_path / ".published_PRIVATE.zip.staging"
    private_zip_backup = tmp_path / ".published_PRIVATE.zip.backup"
    destination.mkdir()
    staging.mkdir()
    (destination / "original.txt").write_text("original", encoding="utf-8")
    (staging / "replacement.txt").write_text("replacement", encoding="utf-8")
    zip_path.write_text("original public", encoding="utf-8")
    zip_staging.write_text("replacement public", encoding="utf-8")
    private_zip_path.write_text("original private", encoding="utf-8")
    private_zip_staging.write_text("replacement private", encoding="utf-8")

    def fail_public_scan() -> None:
        assert zip_path.read_text(encoding="utf-8") == "replacement public"
        assert private_zip_path.read_text(encoding="utf-8") == "replacement private"
        raise RuntimeError("injected public privacy scan failure")

    with pytest.raises(RuntimeError, match="privacy scan"):
        _publish_validated_bundle_transaction(
            staging,
            destination,
            backup,
            zip_staging=zip_staging,
            zip_path=zip_path,
            zip_backup=zip_backup,
            overwrite=True,
            publish_zip=True,
            validate_published=lambda: None,
            private_zip_staging=private_zip_staging,
            private_zip_path=private_zip_path,
            private_zip_backup=private_zip_backup,
            publish_private_zip=True,
            validate_public_zip=fail_public_scan,
        )

    assert (destination / "original.txt").read_text(encoding="utf-8") == "original"
    assert zip_path.read_text(encoding="utf-8") == "original public"
    assert private_zip_path.read_text(encoding="utf-8") == "original private"
    assert not backup.exists()
    assert not zip_backup.exists()
    assert not private_zip_backup.exists()
