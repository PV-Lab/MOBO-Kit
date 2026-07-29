"""Read-only validation for completed D2D Step 2C artifact bundles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import getpass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence
import zipfile

from PIL import Image, UnidentifiedImageError
import yaml


DEBUG_WATERMARK = "DEBUG ONLY - NOT APPROVED FOR EXPERIMENT"

PUBLIC_SUMMARY_ARCHIVE_ROOT = "MOBO_Kit_Step2C_Public_Summary"
PUBLIC_SUMMARY_SCHEMA_VERSION = "d2d-step2c-public-summary-v1"
PUBLIC_SUMMARY_README_FILE = "PUBLIC_SUMMARY_README.txt"
PUBLIC_SUMMARY_MANIFEST_FILE = "PUBLIC_SUMMARY_MANIFEST.json"
PUBLIC_SUMMARY_DEBUG_MARKER_FILE = "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt"

# Public export is deliberately allowlisted. These tables contain only aggregate
# model/search diagnostics; row-level measurements, candidate coordinates,
# recipes, private-workbook proof, and plots of recipe coordinates stay in the
# ignored local evidence bundle.
PUBLIC_SUMMARY_CSV_FILES = (
    "model_validation_summary.csv",
    "model_hyperparameter_stability_summary.csv",
    "nested_pool_convergence_summary.csv",
    "sobol_scramble_comparison.csv",
    "bounded_utility_comparison.csv",
    "local_penalty_tradeoff.csv",
    "beta_robustness.csv",
    "boundary_enrichment.csv",
    "study_run_summary.csv",
)

PUBLIC_SUMMARY_FILES = (
    PUBLIC_SUMMARY_DEBUG_MARKER_FILE,
    PUBLIC_SUMMARY_README_FILE,
    PUBLIC_SUMMARY_MANIFEST_FILE,
    *PUBLIC_SUMMARY_CSV_FILES,
)

REQUIRED_TOP_LEVEL_FILES = (
    "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt",
    "workbook_audit.json",
    "score_validation.csv",
    "training_row_manifest.csv",
    "resolved_debug_config.yaml",
    "run_manifest.json",
    "model_validation_summary.csv",
    "loocv_predictions_long.csv",
    "model_hyperparameters.csv",
    "model_fit_warnings.csv",
    "nested_pool_manifest.csv",
    "nested_pool_convergence_summary.csv",
    "nested_pool_candidates_long.csv",
    "sobol_scramble_comparison.csv",
    "local_refinement_trace.csv",
    "bounded_utility_comparison.csv",
    "local_penalty_tradeoff.csv",
    "beta_robustness.csv",
    "boundary_enrichment.csv",
    "observation_influence_summary.csv",
    "observation_influence_candidates_long.csv",
    "observation_influence_predictions.csv",
    "robust_regions.csv",
    "r1_robust_shortlist_debug.csv",
)

CONSENSUS_BATCH_FILE = "r1_consensus_debug_batch.csv"
NO_STABLE_BATCH_REASON_FILE = "r1_no_stable_batch_reason.json"

# One PNG may cover multiple closely related topics. For example, the LOOCV
# diagnostic combines parity, standardized residual, and interval-coverage
# panels, while the robust-region overview combines PCA and medoid panels.
REQUIRED_PLOT_FILES = (
    "plots/model_validation/loocv_diagnostics.png",
    "plots/model_validation/ard_lengthscale_comparison.png",
    "plots/model_validation/candidate_predictions_vs_observed_ranges.png",
    "plots/search_convergence/nested_pool_convergence.png",
    "plots/search_convergence/boundary_enrichment.png",
    "plots/bounded_utility/bounded_utility_comparison.png",
    "plots/local_penalty/local_penalty_tradeoff.png",
    "plots/influence/observation_influence_ranking.png",
    "plots/influence/full_vs_omit_control_candidates.png",
    "plots/influence/shortlist_region_influence_sensitivity.png",
    "plots/robust_regions/robust_region_overview.png",
    "plots/robust_regions/shortlist_parallel_coordinates.png",
    "plots/robust_regions/run_region_persistence_heatmap.png",
    "plots/robust_regions/model_policy_region_correspondence.png",
    "plots/robust_regions/acquisition_quality_vs_persistence.png",
)

REQUIRED_PLOT_DIRECTORIES = tuple(
    dict.fromkeys(
        str(Path(name).parent).replace("\\", "/") for name in REQUIRED_PLOT_FILES
    )
)

_CSV_STAMP_COLUMNS = (
    "debug_only",
    "approved_for_experiment",
    "approved_for_production",
    "candidate_status",
)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_D2D_INPUT_COLUMNS = (
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
_D2D_OBJECTIVE_NAMES = (
    "uniformity_score",
    "optoelectronic_score",
    "thickness_score",
)
_GRID_COLUMNS = tuple(f"grid_{index}" for index in range(10))
_NORM_COLUMNS = tuple(f"norm_{index}" for index in range(10))
_MEDOID_GRID_COLUMNS = tuple(f"medoid_grid_{index}" for index in range(10))
_MEDOID_NORM_COLUMNS = tuple(f"medoid_norm_{index}" for index in range(10))

_PUBLIC_PRIVATE_FILE_NAMES = frozenset(
    {
        "workbook_audit.json",
        "score_validation.csv",
        "training_row_manifest.csv",
        "resolved_debug_config.yaml",
        "run_manifest.json",
        "loocv_predictions_long.csv",
        "model_hyperparameters.csv",
        "model_fit_warnings.csv",
        "nested_pool_manifest.csv",
        "nested_pool_candidates_long.csv",
        "local_refinement_trace.csv",
        "bounded_utility_candidates_long.csv",
        "observation_influence_summary.csv",
        "observation_influence_candidates_long.csv",
        "observation_influence_predictions.csv",
        "robust_regions.csv",
        "robust_region_membership.csv",
        "r1_robust_shortlist_debug.csv",
        "r1_consensus_debug_batch.csv",
        "study_candidates_long.csv",
    }
)
_PUBLIC_BANNED_HEADER_NAMES = frozenset(
    {
        *_D2D_INPUT_COLUMNS,
        "sample_id",
        "omitted_sample_id",
        "row_position",
        "row_label",
        "row_role",
        "measurement_provenance",
        "candidate_id",
        "consensus_candidate_id",
        "shortlist_id",
        "selection_order",
        "location_id",
    }
)
_PUBLIC_BANNED_HEADER_PREFIXES = (
    "grid_",
    "norm_",
    "phys_",
    "medoid_",
    "pred_mean_",
    "pred_std_",
    "ucb_raw_",
    "ucb_effective_",
)
_PUBLIC_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]+")
_PUBLIC_UNC_PATH = re.compile(r"(?<![\\])\\\\[^\\\s]+[\\/]")
_PUBLIC_POSIX_PROFILE_PATH = re.compile(r"(?i)(?:^|[\s\"'(=])/(?:users|home)/")
_PUBLIC_COMMON_POSIX_PATH = re.compile(
    r"(?i)(?:^|[\s\"'(=])/(?:tmp|var|opt|mnt|etc|usr|private)(?:/|\b)"
)
_PUBLIC_GENERIC_POSIX_PATH = re.compile(
    r"(?:^|[\s\"'(=])/(?:[A-Za-z0-9._~-]+/)+[A-Za-z0-9._~ -]+"
)
_PUBLIC_WORKBOOK_NAME = re.compile(r"(?i)\.(?:xlsx?|xlsm|xlsb)\b")
_IGNORED_PROFILE_IDENTITIES = frozenset({"root", "user", "runner", "admin"})
_PUBLIC_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "method_version",
        "mode",
        "input_data_kind",
        "source_dataset",
        "public_export",
        "debug_only",
        "approved_for_experiment",
        "approved_for_production",
        "candidate_status",
        "contains_candidate_recipes",
        "contains_sample_level_data",
        "contains_local_paths",
        "full_private_evidence_included",
        "git_commit",
        "objective_order",
        "reference_point",
        "objective_bounds",
        "moment_method",
        "robust_region_count",
        "shortlist_count",
        "consensus_passed",
        "consensus_checks",
        "consensus_observed",
        "runtime_versions",
        "archive_root",
        "included_files",
    }
)

REQUIRED_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "score_validation.csv": (
        "sample_id",
        "uniformity_score_authoritative",
        "uniformity_matches",
        "optoelectronic_objective_matches",
        "thickness_objective_matches",
        "warning_codes",
        "error_codes",
    ),
    "training_row_manifest.csv": (
        "sample_id",
        "row_role",
        "include_in_model",
        "measurement_provenance",
        "on_grid",
    ),
    "model_validation_summary.csv": (
        "variant_name",
        "objective_index",
        "objective_name",
        "prediction_count",
        "mae",
        "rmse",
        "coverage_68_percent",
        "coverage_95_percent",
        "mean_gaussian_nlpd",
        "selected_prediction_outside_observed_count",
        "selected_prediction_outside_declared_bounds_count",
    ),
    "loocv_predictions_long.csv": (
        "variant_name",
        "omitted_sample_id",
        "objective_index",
        "objective_name",
        "observed",
        "predicted_mean",
        "predictive_std",
        "standardized_residual",
    ),
    "model_hyperparameters.csv": (
        "variant_name",
        "fit_key",
        "objective_name",
        "likelihood_noise",
        "outputscale",
        "noise_near_floor",
        "any_lengthscale_very_small_normalized_domain",
        "any_lengthscale_extremely_large_flat",
    ),
    "model_fit_warnings.csv": (
        "variant_name",
        "fit_key",
        "stage",
        "warning_category",
        "message",
    ),
    "nested_pool_manifest.csv": (
        "pool_role",
        "scramble_seed",
        "accepted_size",
        "prefix_sha256",
        "exact_accepted_prefix",
        "full_cartesian_grid_materialized",
    ),
    "nested_pool_convergence_summary.csv": (
        "comparison_type",
        "reference_pool_size",
        "comparison_pool_size",
        "mean_matched_distance",
        "hausdorff_distance",
        "regional_matches_within_0.15",
        "refinement_acquisition_gain",
    ),
    "nested_pool_candidates_long.csv": (
        "run_id",
        "selection_order",
        "acquisition_score",
        "grid_valid",
        "bounds_valid",
        *_GRID_COLUMNS,
        *_NORM_COLUMNS,
    ),
    "sobol_scramble_comparison.csv": (
        "reference_run_id",
        "comparison_run_id",
        "mean_matched_distance",
        "regional_matches_within_0.15",
    ),
    "local_refinement_trace.csv": (
        "run_id",
        "record_type",
        "selection_step",
        "base_score_before",
        "base_score_after",
        "accepted_move",
    ),
    "bounded_utility_comparison.csv": (
        "bounded_reference_run",
        "unbounded_comparison_run",
        "clipped_pool_coordinate_count",
        "maximum_pool_clip_amount",
        "training_targets_mutated",
    ),
    "local_penalty_tradeoff.csv": (
        "penalty_label",
        "comparison_reference_penalty_label",
        "minimum_within_batch_distance",
        "acquisition_sacrifice",
        "candidate_region_changed_vs_reference",
        "penalty_activity_classification",
        "hard_distance_relaxed",
    ),
    "beta_robustness.csv": (
        "beta",
        "mean_matched_distance",
        "regional_matches_within_0.15",
    ),
    "boundary_enrichment.csv": (
        "dimension_index",
        "input_name",
        "group",
        "lower_boundary_count",
        "lower_boundary_rate",
        "upper_boundary_count",
        "upper_boundary_rate",
    ),
    "observation_influence_summary.csv": (
        "omitted_sample_id",
        "common_pool_sha256",
        "mean_matched_distance",
        "composite_influence_score",
        "influence_rank",
        "influence_percentile",
        "full_lower_boundary_rate_0",
        "omitted_lower_boundary_rate_0",
        "full_upper_boundary_rate_0",
        "omitted_upper_boundary_rate_0",
    ),
    "observation_influence_candidates_long.csv": (
        "run_id",
        "omitted_sample_id",
        "selection_order",
        "nearest_observed_distance",
        "boundary_coordinate_count",
        "lower_boundary_dimensions",
        "upper_boundary_dimensions",
        "ucb_raw_0",
        "ucb_effective_0",
        "ucb_clip_amount_0",
        *_GRID_COLUMNS,
        *_NORM_COLUMNS,
    ),
    "observation_influence_predictions.csv": (
        "omitted_sample_id",
        "objective_name",
        "full_mean",
        "omitted_mean",
        "normalized_absolute_mean_delta",
        "prediction_location_kind",
        "location_id",
    ),
    "robust_regions.csv": (
        "region_id",
        "member_count",
        "distinct_nonbaseline_family_count",
        "all_grid_valid",
        "all_hard_distance_valid",
        "family_weighted_persistence",
        *_MEDOID_GRID_COLUMNS,
        *_MEDOID_NORM_COLUMNS,
    ),
    "r1_robust_shortlist_debug.csv": (
        "shortlist_id",
        "region_id",
        "distinct_nonbaseline_family_count",
        "all_grid_valid",
        "all_hard_distance_valid",
        "lower_boundary_dimensions",
        "upper_boundary_dimensions",
        *_D2D_INPUT_COLUMNS,
        *_MEDOID_GRID_COLUMNS,
        *_MEDOID_NORM_COLUMNS,
    ),
}

_EMPTY_CSV_ALLOWED = {"model_fit_warnings.csv"}


class Step2CArtifactContractError(ValueError):
    """Raised when a completed output bundle violates the Step 2C contract."""


@dataclass(frozen=True)
class Step2CArtifactValidation:
    """Auditable result returned after a bundle satisfies the full contract."""

    output_dir: Path
    ignored_output_root: Path
    conditional_artifact: str
    workbook_path: Path
    workbook_sha256: str
    workbook_mtime_ns: int
    csv_files_checked: tuple[str, ...]
    json_files_checked: tuple[str, ...]
    png_files_checked: tuple[str, ...]
    artifact_sha256: dict[str, str]
    debug_watermark: str = DEBUG_WATERMARK


@dataclass(frozen=True)
class Step2CPublicSummaryValidation:
    """Validation result for the sanitized, aggregate-only public ZIP."""

    archive_path: Path
    archive_root: str
    entries_checked: tuple[str, ...]
    entry_sha256: dict[str, str]
    debug_watermark: str = DEBUG_WATERMARK


def _error(message: str) -> Step2CArtifactContractError:
    return Step2CArtifactContractError(message)


def _relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _read_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise _error(f"{label} must be readable valid YAML: {path}.") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} must contain a top-level mapping: {path}.")
    return value


def _configured_output_root(config: Mapping[str, Any], repository_root: Path) -> Path:
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping):
        raise _error("resolved_debug_config.yaml must contain an outputs mapping.")
    raw_root = outputs.get("root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise _error("resolved_debug_config.yaml outputs.root must be nonblank.")
    configured = Path(raw_root.strip())
    if configured.is_absolute():
        resolved = configured.resolve()
    else:
        resolved = (repository_root / configured).resolve()
    if resolved == repository_root or repository_root not in resolved.parents:
        raise _error(
            "resolved_debug_config.yaml outputs.root must be a strict descendant "
            "of repository_root."
        )
    return resolved


def _validate_config_stamps(config: Mapping[str, Any]) -> None:
    if config.get("run_mode") != "debug":
        raise _error("resolved_debug_config.yaml run_mode must be exactly 'debug'.")
    if config.get("approved_for_experiment") is not False:
        raise _error(
            "resolved_debug_config.yaml approved_for_experiment must be false."
        )
    if config.get("approved_for_production") is not False:
        raise _error(
            "resolved_debug_config.yaml approved_for_production must be false."
        )
    if config.get("debug_watermark") != DEBUG_WATERMARK:
        raise _error(
            "resolved_debug_config.yaml debug_watermark must equal the exact "
            "Step 2C watermark."
        )


def _validate_csv_stamp(path: Path, output_dir: Path) -> tuple[tuple[str, ...], int]:
    relative = _relative(path, output_dir)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise _error(
                    f"CSV artifact {relative} is empty and has no header."
                ) from exc
            if not header or any(not name.strip() for name in header):
                raise _error(f"CSV artifact {relative} has a blank column name.")
            if len(set(header)) != len(header):
                raise _error(f"CSV artifact {relative} has duplicate column names.")
            missing = [name for name in _CSV_STAMP_COLUMNS if name not in header]
            if missing:
                raise _error(
                    f"CSV artifact {relative} is missing debug stamp columns: {missing}."
                )
            positions = {name: header.index(name) for name in _CSV_STAMP_COLUMNS}
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                if len(row) != len(header):
                    raise _error(
                        f"CSV artifact {relative} row {row_number} has "
                        f"{len(row)} fields; expected {len(header)}."
                    )
                if row[positions["debug_only"]].strip().casefold() != "true":
                    raise _error(
                        f"CSV artifact {relative} row {row_number} must set "
                        "debug_only=true."
                    )
                for approval in (
                    "approved_for_experiment",
                    "approved_for_production",
                ):
                    if row[positions[approval]].strip().casefold() != "false":
                        raise _error(
                            f"CSV artifact {relative} row {row_number} must set "
                            f"{approval}=false."
                        )
                if row[positions["candidate_status"]] != DEBUG_WATERMARK:
                    raise _error(
                        f"CSV artifact {relative} row {row_number} must carry "
                        "the exact debug watermark."
                    )
            return tuple(header), row_count
    except Step2CArtifactContractError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(f"CSV artifact {relative} must be readable valid CSV.") from exc


def _validate_required_csv_schema(
    relative: str,
    header: tuple[str, ...],
    row_count: int,
) -> None:
    required = REQUIRED_CSV_COLUMNS.get(relative)
    if required is None:
        return
    missing = sorted(set(required) - set(header))
    if missing:
        raise _error(
            f"CSV artifact {relative} is missing required semantic columns: {missing}."
        )
    if row_count == 0 and relative not in _EMPTY_CSV_ALLOWED:
        raise _error(f"CSV artifact {relative} must contain at least one data row.")


def _read_and_validate_json_stamp(path: Path, output_dir: Path) -> dict[str, Any]:
    relative = _relative(path, output_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error(f"JSON artifact {relative} must be readable valid JSON.") from exc
    if not isinstance(value, dict):
        raise _error(f"JSON artifact {relative} must contain a top-level object.")
    if value.get("debug_only") is not True:
        raise _error(f"JSON artifact {relative} must set debug_only=true.")
    if value.get("approved_for_experiment") is not False:
        raise _error(
            f"JSON artifact {relative} must set approved_for_experiment=false."
        )
    if value.get("approved_for_production") is not False:
        raise _error(
            f"JSON artifact {relative} must set approved_for_production=false."
        )
    if value.get("candidate_status") != DEBUG_WATERMARK:
        raise _error(f"JSON artifact {relative} must carry the exact debug watermark.")
    return value


def _validate_png_watermark(path: Path, output_dir: Path) -> None:
    relative = _relative(path, output_dir)
    try:
        with Image.open(path) as image:
            image_format = image.format
            description = image.info.get("Description")
            dimensions = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise _error(f"Plot artifact {relative} must be a readable PNG.") from exc
    if image_format != "PNG" or dimensions[0] <= 0 or dimensions[1] <= 0:
        raise _error(f"Plot artifact {relative} must be a non-empty PNG image.")
    if description != DEBUG_WATERMARK:
        raise _error(
            f"Plot artifact {relative} PNG Description must equal the exact "
            "debug watermark."
        )


def _validated_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise _error(f"run_manifest.json {field} must be a SHA-256 hex digest.")
    return value.upper()


def _validated_mtime(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(
            f"run_manifest.json {field} must be a non-negative integer timestamp."
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _public_profile_identities(additional: Sequence[str]) -> tuple[str, ...]:
    identities = {
        value.strip()
        for value in (
            getpass.getuser(),
            Path.home().name,
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
            os.environ.get("LOGNAME", ""),
            *additional,
        )
        if isinstance(value, str)
        and len(value.strip()) >= 4
        and value.strip().casefold() not in _IGNORED_PROFILE_IDENTITIES
    }
    return tuple(sorted(identities, key=str.casefold))


def _scan_public_text(
    relative: str,
    text: str,
    *,
    forbidden_profile_strings: Sequence[str],
) -> None:
    patterns = (
        (_PUBLIC_WINDOWS_ABSOLUTE_PATH, "a Windows absolute path"),
        (_PUBLIC_UNC_PATH, "a UNC path"),
        (_PUBLIC_POSIX_PROFILE_PATH, "a macOS/Linux profile path"),
        (_PUBLIC_COMMON_POSIX_PATH, "a common POSIX absolute path"),
        (_PUBLIC_GENERIC_POSIX_PATH, "a POSIX absolute path"),
        (_PUBLIC_WORKBOOK_NAME, "a workbook filename"),
    )
    for pattern, description in patterns:
        if pattern.search(text) is not None:
            raise _error(
                f"Public summary entry {relative} contains {description}; "
                "the export must be path- and workbook-free."
            )
    lowered = text.casefold()
    for private_name in _PUBLIC_PRIVATE_FILE_NAMES:
        if private_name in lowered:
            raise _error(
                f"Public summary entry {relative} names private artifact "
                f"{private_name}."
            )
    for identity in _public_profile_identities(forbidden_profile_strings):
        identity_pattern = re.compile(
            rf"(?i)(?<![A-Za-z0-9]){re.escape(identity)}(?![A-Za-z0-9])"
        )
        if identity_pattern.search(text) is not None:
            raise _error(
                f"Public summary entry {relative} contains a local profile "
                "identifier."
            )


def _validate_public_csv_payload(relative: str, text: str) -> None:
    try:
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise _error(f"Public summary CSV {relative} has no valid header.") from exc
    if not header or any(not name.strip() for name in header):
        raise _error(f"Public summary CSV {relative} has a blank column name.")
    if len(header) != len(set(header)):
        raise _error(f"Public summary CSV {relative} has duplicate columns.")
    normalized_header = tuple(name.strip().casefold() for name in header)
    banned = sorted(
        name
        for name in normalized_header
        if name in _PUBLIC_BANNED_HEADER_NAMES
        or any(name.startswith(prefix) for prefix in _PUBLIC_BANNED_HEADER_PREFIXES)
    )
    if banned:
        raise _error(
            f"Public summary CSV {relative} contains recipe/sample-level "
            f"column(s): {banned}."
        )
    missing_stamps = sorted(set(_CSV_STAMP_COLUMNS) - set(header))
    if missing_stamps:
        raise _error(
            f"Public summary CSV {relative} is missing debug stamp columns: "
            f"{missing_stamps}."
        )
    positions = {name: header.index(name) for name in _CSV_STAMP_COLUMNS}
    row_count = 0
    try:
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != len(header):
                raise _error(
                    f"Public summary CSV {relative} row {row_number} has "
                    f"{len(row)} fields; expected {len(header)}."
                )
            if row[positions["debug_only"]].strip().casefold() != "true":
                raise _error(
                    f"Public summary CSV {relative} row {row_number} must set "
                    "debug_only=true."
                )
            for approval in (
                "approved_for_experiment",
                "approved_for_production",
            ):
                if row[positions[approval]].strip().casefold() != "false":
                    raise _error(
                        f"Public summary CSV {relative} row {row_number} must set "
                        f"{approval}=false."
                    )
            if row[positions["candidate_status"]] != DEBUG_WATERMARK:
                raise _error(
                    f"Public summary CSV {relative} row {row_number} must carry "
                    "the exact debug watermark."
                )
    except csv.Error as exc:
        raise _error(f"Public summary CSV {relative} is malformed.") from exc
    if row_count == 0:
        raise _error(f"Public summary CSV {relative} must contain aggregate rows.")


def _validate_public_manifest(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> None:
    if set(manifest) != _PUBLIC_MANIFEST_FIELDS:
        missing = sorted(_PUBLIC_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - _PUBLIC_MANIFEST_FIELDS)
        raise _error(
            "PUBLIC_SUMMARY_MANIFEST.json must use the strict public schema; "
            f"missing={missing}, extra={extra}."
        )
    if manifest.get("schema_version") != PUBLIC_SUMMARY_SCHEMA_VERSION:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json has the wrong schema version.")
    if manifest.get("archive_root") != PUBLIC_SUMMARY_ARCHIVE_ROOT:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json has the wrong archive root.")
    if manifest.get("mode") not in {"fast", "full"}:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json mode must be fast or full.")
    if manifest.get("input_data_kind") not in {
        "private_campaign_dataset_redacted",
        "generated_synthetic_dataset",
    } or manifest.get("source_dataset") != manifest.get("input_data_kind"):
        raise _error(
            "PUBLIC_SUMMARY_MANIFEST.json must use a redacted/generic data label."
        )
    required_true = ("public_export", "debug_only")
    required_false = (
        "approved_for_experiment",
        "approved_for_production",
        "contains_candidate_recipes",
        "contains_sample_level_data",
        "contains_local_paths",
        "full_private_evidence_included",
    )
    for field in required_true:
        if manifest.get(field) is not True:
            raise _error(f"PUBLIC_SUMMARY_MANIFEST.json {field} must be true.")
    for field in required_false:
        if manifest.get(field) is not False:
            raise _error(f"PUBLIC_SUMMARY_MANIFEST.json {field} must be false.")
    if manifest.get("candidate_status") != DEBUG_WATERMARK:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json has the wrong debug watermark.")
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json git_commit is invalid.")
    if tuple(manifest.get("objective_order", ())) != _D2D_OBJECTIVE_NAMES:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json objective order is invalid.")
    if manifest.get("moment_method") != "analytic_identity":
        raise _error("PUBLIC_SUMMARY_MANIFEST.json moment method is invalid.")
    if not isinstance(manifest.get("consensus_passed"), bool) or not all(
        isinstance(manifest.get(field), Mapping)
        for field in ("consensus_checks", "consensus_observed", "runtime_versions")
    ):
        raise _error("PUBLIC_SUMMARY_MANIFEST.json aggregate evidence is invalid.")
    for field in ("robust_region_count", "shortlist_count"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _error(f"PUBLIC_SUMMARY_MANIFEST.json {field} is invalid.")

    hashes = manifest.get("included_files")
    expected_hash_names = set(PUBLIC_SUMMARY_FILES) - {PUBLIC_SUMMARY_MANIFEST_FILE}
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hash_names:
        raise _error(
            "PUBLIC_SUMMARY_MANIFEST.json included_files must exactly match the "
            "allowlisted non-manifest entries."
        )
    for relative in sorted(expected_hash_names):
        expected_hash = hashes.get(relative)
        if (
            not isinstance(expected_hash, str)
            or _SHA256_PATTERN.fullmatch(expected_hash) is None
            or expected_hash.upper() != _sha256_bytes(payloads[relative])
        ):
            raise _error(f"PUBLIC_SUMMARY_MANIFEST.json hash mismatch for {relative}.")


def validate_step2c_public_summary_archive(
    archive_path: str | Path,
    *,
    forbidden_profile_strings: Sequence[str] = (),
) -> Step2CPublicSummaryValidation:
    """Validate the strict aggregate-only public Step 2C ZIP without mutation."""

    archive_file = Path(archive_path).resolve()
    if not archive_file.is_file():
        raise _error(f"Public summary archive does not exist: {archive_file}.")
    expected_entries = {
        f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{relative}" for relative in PUBLIC_SUMMARY_FILES
    }
    payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(archive_file, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise _error("Public summary archive contains duplicate entry names.")
            for info in infos:
                pure = PurePosixPath(info.filename)
                if (
                    info.is_dir()
                    or pure.is_absolute()
                    or "\\" in info.filename
                    or ".." in pure.parts
                    or len(pure.parts) != 2
                ):
                    raise _error(
                        f"Unsafe public summary archive entry: {info.filename!r}."
                    )
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise _error(
                        f"Public summary archive entry is a symbolic link: "
                        f"{info.filename!r}."
                    )
                if pure.name.casefold() in _PUBLIC_PRIVATE_FILE_NAMES:
                    raise _error(
                        f"Public summary archive includes private artifact "
                        f"{pure.name}."
                    )
                if info.file_size > 32 * 1024 * 1024:
                    raise _error(
                        f"Public summary entry {info.filename} exceeds 32 MiB."
                    )
            actual_entries = set(names)
            if actual_entries != expected_entries:
                missing = sorted(expected_entries - actual_entries)
                extra = sorted(actual_entries - expected_entries)
                raise _error(
                    "Public summary archive must contain exactly the allowlisted "
                    f"surface; missing={missing}, extra={extra}."
                )
            for info in infos:
                relative = PurePosixPath(info.filename).name
                payloads[relative] = archive.read(info)
    except Step2CArtifactContractError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise _error("Public summary archive must be a readable valid ZIP.") from exc

    texts: dict[str, str] = {}
    for relative, payload in payloads.items():
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise _error(
                f"Public summary entry {relative} must be UTF-8 text."
            ) from exc
        texts[relative] = text
        _scan_public_text(
            relative,
            text,
            forbidden_profile_strings=forbidden_profile_strings,
        )
    if texts[PUBLIC_SUMMARY_DEBUG_MARKER_FILE].strip() != DEBUG_WATERMARK:
        raise _error("Public summary archive has the wrong debug marker.")
    for relative in PUBLIC_SUMMARY_CSV_FILES:
        _validate_public_csv_payload(relative, texts[relative])
    try:
        manifest = json.loads(texts[PUBLIC_SUMMARY_MANIFEST_FILE])
    except json.JSONDecodeError as exc:
        raise _error("PUBLIC_SUMMARY_MANIFEST.json must be valid JSON.") from exc
    if not isinstance(manifest, Mapping):
        raise _error("PUBLIC_SUMMARY_MANIFEST.json must contain an object.")
    _validate_public_manifest(manifest, payloads)

    return Step2CPublicSummaryValidation(
        archive_path=archive_file,
        archive_root=PUBLIC_SUMMARY_ARCHIVE_ROOT,
        entries_checked=tuple(sorted(expected_entries)),
        entry_sha256={
            relative: _sha256_bytes(payload)
            for relative, payload in sorted(payloads.items())
        },
    )


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_workbook_proof(
    manifest: Mapping[str, Any], repository_root: Path, output_dir: Path
) -> tuple[Path, str, int]:
    raw_path = manifest.get("workbook_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise _error("run_manifest.json workbook_path must be a nonblank path.")
    workbook_path = Path(raw_path.strip())
    if not workbook_path.is_absolute():
        workbook_path = repository_root / workbook_path
    workbook_path = workbook_path.resolve()
    if not workbook_path.is_file():
        raise _error(
            f"run_manifest.json source workbook does not exist: {workbook_path}."
        )
    if workbook_path == output_dir or output_dir in workbook_path.parents:
        raise _error("The source workbook must remain outside the output bundle.")

    before_hash = _validated_sha256(
        manifest.get("workbook_sha256_before"), field="workbook_sha256_before"
    )
    after_hash = _validated_sha256(
        manifest.get("workbook_sha256_after"), field="workbook_sha256_after"
    )
    before_mtime = _validated_mtime(
        manifest.get("workbook_mtime_ns_before"), field="workbook_mtime_ns_before"
    )
    after_mtime = _validated_mtime(
        manifest.get("workbook_mtime_ns_after"), field="workbook_mtime_ns_after"
    )
    if manifest.get("source_workbook_modified") is not False:
        raise _error(
            "run_manifest.json source_workbook_modified must be exactly false."
        )
    if before_hash != after_hash:
        raise _error("Source workbook SHA-256 changed between before and after proof.")
    if before_mtime != after_mtime:
        raise _error("Source workbook mtime changed between before and after proof.")

    stat_before = workbook_path.stat()
    current_hash = _sha256_file(workbook_path)
    stat_after = workbook_path.stat()
    if (
        stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_size != stat_after.st_size
    ):
        raise _error("Source workbook changed while its proof was being validated.")
    if current_hash != after_hash:
        raise _error(
            "The live source workbook SHA-256 does not match run_manifest.json."
        )
    if stat_after.st_mtime_ns != after_mtime:
        raise _error("The live source workbook mtime does not match run_manifest.json.")
    return workbook_path, current_hash, stat_after.st_mtime_ns


def _require_mapping_fields(
    value: Mapping[str, Any], fields: tuple[str, ...], *, label: str
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise _error(f"{label} is missing required provenance fields: {missing}.")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_manifest_provenance(
    manifest: Mapping[str, Any],
    repository_root: Path,
    ignored_output_root: Path,
    resolved_config: Mapping[str, Any],
) -> None:
    required = (
        "schema_version",
        "method_version",
        "mode",
        "debug_only",
        "approved_for_experiment",
        "approved_for_production",
        "candidate_status",
        "real_r2_proposal_generated",
        "workbook_writeback_performed",
        "git_commit",
        "git_dirty",
        "git_status_at_start",
        "step2b_checkpoint",
        "config_path",
        "config_sha256",
        "resolved_config_hash",
        "objective_order",
        "objective_source_columns",
        "reference_point",
        "objective_bounds",
        "moment_method",
        "analytic_mc_comparison",
        "sobol_seeds",
        "nested_pool_sizes",
        "pool_prefix_hashes",
        "local_refinement",
        "beta_values",
        "bound_policies",
        "local_penalty_variants",
        "model_variants",
        "influence_common_pool_hash",
        "influence_common_pool_size",
        "influence_omitted_sample_ids",
        "robust_region_clustering",
        "robust_region_threshold",
        "robust_region_sensitivity_counts",
        "robust_region_count",
        "shortlist_count",
        "stability_criteria",
        "consensus_passed",
        "consensus_checks",
        "consensus_observed",
        "runtime_versions",
        "hardware",
        "phase_runtime_seconds",
        "runtime_seconds_total",
        "known_uniformity_score_mismatch",
        "uniformity_warning_count",
        "control_assumption",
        "off_grid_control_exception",
        "output_directory",
        "public_summary_archive_requested",
        "private_evidence_archive_requested",
    )
    _require_mapping_fields(manifest, required, label="run_manifest.json")
    if manifest.get("schema_version") != "d2d-step2c-robustness-run-v1":
        raise _error("run_manifest.json schema_version is not the Step 2C schema.")
    if manifest.get("mode") not in {"fast", "full"}:
        raise _error("run_manifest.json mode must be 'fast' or 'full'.")
    if manifest.get("debug_only") is not True:
        raise _error("run_manifest.json debug_only must be exactly true.")
    for field in (
        "approved_for_experiment",
        "approved_for_production",
        "real_r2_proposal_generated",
        "workbook_writeback_performed",
    ):
        if manifest.get(field) is not False:
            raise _error(f"run_manifest.json {field} must be exactly false.")
    if manifest.get("candidate_status") != DEBUG_WATERMARK:
        raise _error("run_manifest.json must carry the exact debug watermark.")
    if not isinstance(manifest.get("git_dirty"), bool) or not isinstance(
        manifest.get("git_status_at_start"), list
    ):
        raise _error("run_manifest.json Git dirty/status provenance is invalid.")
    for field in ("git_commit", "step2b_checkpoint"):
        value = manifest.get(field)
        if not isinstance(value, str) or _GIT_COMMIT_PATTERN.fullmatch(value) is None:
            raise _error(f"run_manifest.json {field} must be a 40-character commit.")

    config_path = Path(str(manifest["config_path"]))
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file() or (
        config_path != repository_root and repository_root not in config_path.parents
    ):
        raise _error("run_manifest.json config_path must be a repository file.")
    config_hash = _validated_sha256(manifest["config_sha256"], field="config_sha256")
    if _sha256_file(config_path) != config_hash:
        raise _error("The live Step 2C config SHA-256 does not match the manifest.")
    resolved_hash = _validated_sha256(
        manifest["resolved_config_hash"], field="resolved_config_hash"
    )
    if resolved_hash != _canonical_mapping_sha256(resolved_config):
        raise _error(
            "The canonical resolved_debug_config.yaml SHA-256 does not match "
            "run_manifest.json."
        )
    source_config = _read_yaml_mapping(config_path, label="live Step 2C config")
    if resolved_hash != _canonical_mapping_sha256(source_config):
        raise _error(
            "resolved_debug_config.yaml does not canonically match the live Step 2C "
            "source config."
        )

    if tuple(manifest["objective_order"]) != _D2D_OBJECTIVE_NAMES:
        raise _error("run_manifest.json objective_order changed from Z/AA/AB order.")
    reference = manifest["reference_point"]
    if (
        not isinstance(reference, list)
        or len(reference) != 3
        or not all(_is_finite_number(value) for value in reference)
    ):
        raise _error(
            "run_manifest.json reference_point must contain three finite values."
        )
    bounds = manifest["objective_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 3:
        raise _error("run_manifest.json objective_bounds must contain three pairs.")
    if manifest.get("moment_method") != "analytic_identity":
        raise _error("run_manifest.json moment_method must be analytic_identity.")
    if not isinstance(manifest["analytic_mc_comparison"], Mapping):
        raise _error("run_manifest.json analytic_mc_comparison must be a mapping.")

    seeds = manifest["sobol_seeds"]
    sizes = manifest["nested_pool_sizes"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
    ):
        raise _error("run_manifest.json sobol_seeds must contain three integers.")
    if (
        not isinstance(sizes, list)
        or len(sizes) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sizes
        )
        or sizes != sorted(set(sizes))
    ):
        raise _error(
            "run_manifest.json nested_pool_sizes must be four increasing sizes."
        )
    prefix_hashes = manifest["pool_prefix_hashes"]
    if not isinstance(prefix_hashes, Mapping):
        raise _error("run_manifest.json pool_prefix_hashes must be a mapping.")
    for seed_index, seed in enumerate(seeds):
        by_size = prefix_hashes.get(str(seed))
        if not isinstance(by_size, Mapping):
            raise _error(
                f"run_manifest.json has no prefix hashes for Sobol seed {seed}."
            )
        expected_sizes = sizes if seed_index == 0 else [sizes[-1]]
        for size in expected_sizes:
            _validated_sha256(
                by_size.get(str(size), by_size.get(size)),
                field=f"pool_prefix_hashes[{seed}][{size}]",
            )

    for field in ("local_refinement", "hardware", "phase_runtime_seconds"):
        if not isinstance(manifest[field], Mapping) or not manifest[field]:
            raise _error(f"run_manifest.json {field} must be a non-empty mapping.")
    if any(
        not _is_finite_number(value) or float(value) < 0.0
        for value in manifest["phase_runtime_seconds"].values()
    ):
        raise _error(
            "run_manifest.json phase runtimes must be finite and non-negative."
        )
    if (
        not _is_finite_number(manifest["runtime_seconds_total"])
        or float(manifest["runtime_seconds_total"]) < 0.0
    ):
        raise _error("run_manifest.json runtime_seconds_total must be non-negative.")
    if tuple(float(value) for value in manifest["beta_values"]) != (1.0, 4.0, 9.0):
        raise _error("run_manifest.json beta_values must preserve the 1/4/9 study.")
    if set(manifest["bound_policies"]) != {"none", "clip_ucb"}:
        raise _error("run_manifest.json bound_policies are incomplete.")
    if (
        not isinstance(manifest["local_penalty_variants"], list)
        or len(manifest["local_penalty_variants"]) != 5
    ):
        raise _error("run_manifest.json must record all five penalty variants.")
    model_variants = manifest["model_variants"]
    if not isinstance(model_variants, list) or {
        item.get("name") for item in model_variants if isinstance(item, Mapping)
    } != {"dim_scaled_prior", "conservative"}:
        raise _error("run_manifest.json must record both GP model variants.")
    _validated_sha256(
        manifest["influence_common_pool_hash"], field="influence_common_pool_hash"
    )
    if (
        isinstance(manifest["influence_common_pool_size"], bool)
        or not isinstance(manifest["influence_common_pool_size"], int)
        or manifest["influence_common_pool_size"] <= 0
        or not isinstance(manifest["influence_omitted_sample_ids"], list)
        or not manifest["influence_omitted_sample_ids"]
    ):
        raise _error("run_manifest.json influence-study provenance is invalid.")
    if manifest["robust_region_clustering"] != "agglomerative_complete_link" or not (
        _is_finite_number(manifest["robust_region_threshold"])
        and float(manifest["robust_region_threshold"]) > 0.0
    ):
        raise _error("run_manifest.json robust-region method/threshold is invalid.")
    for field in ("robust_region_count", "shortlist_count"):
        if (
            isinstance(manifest[field], bool)
            or not isinstance(manifest[field], int)
            or manifest[field] <= 0
        ):
            raise _error(f"run_manifest.json {field} must be a positive integer.")

    criteria = manifest["stability_criteria"]
    checks = manifest["consensus_checks"]
    observed = manifest["consensus_observed"]
    if not all(isinstance(value, Mapping) for value in (criteria, checks, observed)):
        raise _error(
            "run_manifest.json consensus criteria/checks/observed must be mappings."
        )
    _require_mapping_fields(
        criteria,
        (
            "consensus_batch_size",
            "regional_match_threshold",
            "largest_two_nested_regional_match_minimum",
            "largest_two_nested_mean_matched_distance_maximum",
            "minimum_nonbaseline_core_family_coverage",
            "required_pairwise_minimum_distance",
        ),
        label="run_manifest.json stability_criteria",
    )
    if criteria["consensus_batch_size"] != 5:
        raise _error(
            "run_manifest.json stability criteria must require five candidates."
        )
    for field in (
        "require_finite_and_bounded",
        "require_unique_and_on_grid",
        "require_debug_only_and_approval_false",
    ):
        if criteria.get(field) is not True:
            raise _error(
                f"run_manifest.json stability criteria must keep {field}=true."
            )
    if not isinstance(manifest["consensus_passed"], bool):
        raise _error("run_manifest.json consensus_passed must be boolean.")

    versions = manifest["runtime_versions"]
    if not isinstance(versions, Mapping) or not {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "matplotlib",
        "torch",
        "botorch",
        "gpytorch",
    } <= set(versions):
        raise _error("run_manifest.json runtime_versions are incomplete.")
    if manifest["known_uniformity_score_mismatch"] is not True:
        raise _error("run_manifest.json must retain the known Uniformity mismatch.")
    if (
        isinstance(manifest["uniformity_warning_count"], bool)
        or not isinstance(manifest["uniformity_warning_count"], int)
        or manifest["uniformity_warning_count"] <= 0
    ):
        raise _error("run_manifest.json must record Uniformity warnings.")
    if (
        not isinstance(manifest["control_assumption"], str)
        or not manifest["control_assumption"].strip()
    ):
        raise _error("run_manifest.json control_assumption must be nonblank.")
    if (
        not isinstance(manifest["off_grid_control_exception"], list)
        or not manifest["off_grid_control_exception"]
    ):
        raise _error("run_manifest.json must retain the off-grid control exception.")
    for field in (
        "public_summary_archive_requested",
        "private_evidence_archive_requested",
    ):
        if not isinstance(manifest[field], bool):
            raise _error(f"run_manifest.json {field} must be boolean.")
    output_directory = Path(str(manifest["output_directory"]))
    if not output_directory.is_absolute():
        output_directory = repository_root / output_directory
    output_directory = output_directory.resolve()
    if (
        output_directory == ignored_output_root
        or ignored_output_root not in output_directory.parents
    ):
        raise _error("run_manifest.json output_directory escapes the configured root.")


def _resolved_input_specs(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[dict[str, Any], ...]:
    raw_base = config.get("base_step2b_config")
    if not isinstance(raw_base, str) or not raw_base.strip():
        raise _error("resolved_debug_config.yaml must name base_step2b_config.")
    base_path = Path(raw_base.strip())
    if not base_path.is_absolute():
        base_path = repository_root / base_path
    base_path = base_path.resolve()
    if not base_path.is_file() or repository_root not in base_path.parents:
        raise _error("base_step2b_config must resolve to a repository file.")
    base = _read_yaml_mapping(base_path, label="base_step2b_config")
    inputs = base.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(_D2D_INPUT_COLUMNS):
        raise _error("base_step2b_config must contain the ten D2D input grids.")
    if (
        tuple(item.get("name") for item in inputs if isinstance(item, Mapping))
        != _D2D_INPUT_COLUMNS
    ):
        raise _error("base_step2b_config D2D input order changed.")
    return tuple(dict(item) for item in inputs)


def _parse_finite_csv_number(value: str, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"Consensus field {field} must be numeric.") from exc
    if not math.isfinite(number):
        raise _error(f"Consensus field {field} must be finite.")
    return number


def _parse_csv_boolean(value: str, *, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise _error(f"Consensus field {field} must be true or false.")


def _read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(f"{label} must be readable valid CSV.") from exc
    if any(None in row for row in rows):
        raise _error(f"{label} contains a malformed CSV row.")
    return rows


def _strict_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(f"{field} must be an integer >= {minimum}.")
    return value


def _strict_number(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if not _is_finite_number(value) or float(value) < minimum:
        raise _error(f"{field} must be finite and >= {minimum}.")
    return float(value)


def _strict_integer_list(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise _error(f"{field} must be a non-empty integer list.")
    return tuple(
        _strict_integer(item, field=f"{field}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )


def _config_mapping(
    value: Mapping[str, Any], key: str, *, field: str
) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise _error(f"resolved_debug_config.yaml {field} must be a mapping.")
    return nested


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    return _is_finite_number(actual) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12
    )


def _validate_consensus_evidence(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    """Recompute every consensus gate from resolved config and bundled tables."""

    regions_config = _config_mapping(config, "robust_regions", field="robust_regions")
    required_config = _config_mapping(
        regions_config,
        "consensus_required_criteria",
        field="robust_regions.consensus_required_criteria",
    )
    search_config = _config_mapping(
        config, "candidate_search", field="candidate_search"
    )
    influence_config = _config_mapping(
        config, "observation_influence", field="observation_influence"
    )
    modes_config = _config_mapping(config, "execution_modes", field="execution_modes")
    mode = str(manifest["mode"])
    mode_config = _config_mapping(modes_config, mode, field=f"execution_modes.{mode}")

    full_sizes = _strict_integer_list(
        search_config.get("nested_unique_sizes"),
        field="resolved_debug_config.yaml candidate_search.nested_unique_sizes",
    )
    full_omissions = _strict_integer_list(
        influence_config.get("omitted_sample_ids"),
        field="resolved_debug_config.yaml observation_influence.omitted_sample_ids",
    )
    mode_sizes = _strict_integer_list(
        mode_config.get("nested_unique_sizes"),
        field=f"resolved_debug_config.yaml execution_modes.{mode}.nested_unique_sizes",
    )
    mode_omissions = _strict_integer_list(
        mode_config.get("omitted_sample_ids"),
        field=f"resolved_debug_config.yaml execution_modes.{mode}.omitted_sample_ids",
    )
    if tuple(manifest["nested_pool_sizes"]) != mode_sizes:
        raise _error(
            "run_manifest.json nested_pool_sizes do not match the resolved execution "
            "mode."
        )
    if tuple(manifest["influence_omitted_sample_ids"]) != mode_omissions:
        raise _error(
            "run_manifest.json influence omissions do not match the resolved "
            "execution mode."
        )

    batch_size = _strict_integer(
        regions_config.get("consensus_batch_size"),
        field="resolved_debug_config.yaml robust_regions.consensus_batch_size",
        minimum=1,
    )
    region_threshold = _strict_number(
        regions_config.get("primary_distance_threshold"),
        field="resolved_debug_config.yaml robust_regions.primary_distance_threshold",
    )
    if regions_config.get("clustering") != manifest["robust_region_clustering"]:
        raise _error(
            "run_manifest.json robust-region clustering does not match the resolved "
            "debug config."
        )
    if not _values_match(manifest["robust_region_threshold"], region_threshold):
        raise _error(
            "run_manifest.json robust-region threshold does not match the resolved "
            "debug config."
        )
    nested_match_minimum = _strict_integer(
        required_config.get("largest_two_nested_regional_matches_within_0_15"),
        field=(
            "resolved_debug_config.yaml consensus largest-two regional-match minimum"
        ),
        minimum=1,
    )
    mean_distance_maximum = _strict_number(
        required_config.get("largest_two_nested_mean_matched_distance_max"),
        field=(
            "resolved_debug_config.yaml consensus largest-two mean-distance maximum"
        ),
    )
    family_minimum = _strict_integer(
        required_config.get("minimum_region_family_coverage"),
        field="resolved_debug_config.yaml consensus family-coverage minimum",
        minimum=1,
    )
    for field in ("require_grid_valid", "require_hard_distance_valid"):
        if required_config.get(field) is not True:
            raise _error(
                f"resolved_debug_config.yaml consensus criterion {field} must be true."
            )

    criteria = manifest["stability_criteria"]
    expected_criteria: dict[str, int | float | bool] = {
        "consensus_batch_size": batch_size,
        "regional_match_threshold": region_threshold,
        "largest_two_nested_regional_match_minimum": nested_match_minimum,
        "largest_two_nested_mean_matched_distance_maximum": (mean_distance_maximum),
        "minimum_nonbaseline_core_family_coverage": family_minimum,
        "required_pairwise_minimum_distance": region_threshold,
        "require_finite_and_bounded": True,
        "require_unique_and_on_grid": True,
        "require_debug_only_and_approval_false": True,
    }
    for field, expected in expected_criteria.items():
        if not _values_match(criteria.get(field), expected):
            raise _error(
                f"run_manifest.json stability criterion {field} does not match the "
                "resolved debug config."
            )

    full_mode_eligible = bool(
        mode == "full" and mode_sizes == full_sizes and mode_omissions == full_omissions
    )
    largest_size, second_largest_size = mode_sizes[-1], mode_sizes[-2]
    convergence_rows = _read_csv_rows(
        output_dir / "nested_pool_convergence_summary.csv",
        label="nested_pool_convergence_summary.csv",
    )
    largest_two_rows = []
    for row_number, row in enumerate(convergence_rows, start=2):
        reference = _parse_finite_csv_number(
            row["reference_pool_size"],
            field=f"nested convergence row {row_number} reference_pool_size",
        )
        comparison = _parse_finite_csv_number(
            row["comparison_pool_size"],
            field=f"nested convergence row {row_number} comparison_pool_size",
        )
        if (
            row.get("comparison_type") == "prefix_vs_largest"
            and reference == largest_size
            and comparison == second_largest_size
        ):
            largest_two_rows.append((row_number, row))
    if len(largest_two_rows) != 1:
        raise _error(
            "nested_pool_convergence_summary.csv must contain exactly one largest-two "
            "pool comparison row."
        )
    row_number, largest_two = largest_two_rows[0]
    largest_two_matches = _parse_finite_csv_number(
        largest_two["regional_matches_within_0.15"],
        field=f"nested convergence row {row_number} regional matches",
    )
    if not largest_two_matches.is_integer():
        raise _error("Largest-two regional-match count must be an integer.")
    regional_match_count = int(largest_two_matches)
    mean_matched_distance = _parse_finite_csv_number(
        largest_two["mean_matched_distance"],
        field=f"nested convergence row {row_number} mean_matched_distance",
    )

    region_rows = _read_csv_rows(
        output_dir / "robust_regions.csv", label="robust_regions.csv"
    )
    if manifest["robust_region_count"] != len(region_rows):
        raise _error(
            "run_manifest.json robust_region_count disagrees with robust_regions.csv."
        )
    qualifying_region_count = 0
    region_evidence: dict[
        str,
        tuple[
            int,
            bool,
            bool,
            tuple[int, ...],
            tuple[float, ...],
        ],
    ] = {}
    for row_number, row in enumerate(region_rows, start=2):
        region_id = row["region_id"].strip()
        if not region_id or region_id in region_evidence:
            raise _error(
                "robust_regions.csv region_id values must be unique and nonblank."
            )
        family_count = _parse_finite_csv_number(
            row["distinct_nonbaseline_family_count"],
            field=f"robust region row {row_number} family count",
        )
        if not family_count.is_integer():
            raise _error("Robust-region family counts must be integers.")
        qualifying_region_count += int(family_count) >= family_minimum
        grid_valid = _parse_csv_boolean(
            row["all_grid_valid"],
            field=f"robust region row {row_number} all_grid_valid",
        )
        hard_valid = _parse_csv_boolean(
            row["all_hard_distance_valid"],
            field=f"robust region row {row_number} all_hard_distance_valid",
        )
        grid_values = []
        for column in _MEDOID_GRID_COLUMNS:
            value = _parse_finite_csv_number(
                row[column], field=f"robust region row {row_number} {column}"
            )
            if not value.is_integer():
                raise _error("Robust-region medoid grid indices must be integers.")
            grid_values.append(int(value))
        normalized = tuple(
            _parse_finite_csv_number(
                row[column], field=f"robust region row {row_number} {column}"
            )
            for column in _MEDOID_NORM_COLUMNS
        )
        if any(value < 0.0 or value > 1.0 for value in normalized):
            raise _error("Robust-region medoid coordinates must lie in [0, 1].")
        region_evidence[region_id] = (
            int(family_count),
            grid_valid,
            hard_valid,
            tuple(grid_values),
            normalized,
        )

    shortlist_rows = _read_csv_rows(
        output_dir / "r1_robust_shortlist_debug.csv",
        label="r1_robust_shortlist_debug.csv",
    )
    if manifest["shortlist_count"] != len(shortlist_rows):
        raise _error(
            "run_manifest.json shortlist_count disagrees with "
            "r1_robust_shortlist_debug.csv."
        )
    eligible_rows: list[tuple[int, dict[str, str]]] = []
    seen_shortlist_regions: set[str] = set()
    for row_number, row in enumerate(shortlist_rows, start=2):
        region_id = row["region_id"].strip()
        if not region_id or region_id in seen_shortlist_regions:
            raise _error("Shortlist region_id values must be unique and nonblank.")
        seen_shortlist_regions.add(region_id)
        source_evidence = region_evidence.get(region_id)
        if source_evidence is None:
            raise _error("Every shortlist region_id must reference robust_regions.csv.")
        family_count = _parse_finite_csv_number(
            row["distinct_nonbaseline_family_count"],
            field=f"shortlist row {row_number} family count",
        )
        if not family_count.is_integer():
            raise _error("Shortlist family counts must be integers.")
        grid_valid = _parse_csv_boolean(
            row["all_grid_valid"],
            field=f"shortlist row {row_number} all_grid_valid",
        )
        hard_valid = _parse_csv_boolean(
            row["all_hard_distance_valid"],
            field=f"shortlist row {row_number} all_hard_distance_valid",
        )
        shortlist_grid = []
        for column in _MEDOID_GRID_COLUMNS:
            value = _parse_finite_csv_number(
                row[column], field=f"shortlist row {row_number} {column}"
            )
            if not value.is_integer():
                raise _error("Shortlist medoid grid indices must be integers.")
            shortlist_grid.append(int(value))
        shortlist_norm = tuple(
            _parse_finite_csv_number(
                row[column], field=f"shortlist row {row_number} {column}"
            )
            for column in _MEDOID_NORM_COLUMNS
        )
        (
            source_family_count,
            source_grid_valid,
            source_hard_valid,
            source_grid,
            source_norm,
        ) = source_evidence
        if (
            int(family_count) != source_family_count
            or grid_valid is not source_grid_valid
            or hard_valid is not source_hard_valid
            or tuple(shortlist_grid) != source_grid
            or any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
                for left, right in zip(shortlist_norm, source_norm)
            )
        ):
            raise _error("Shortlist region evidence does not match robust_regions.csv.")
        if int(family_count) >= family_minimum and grid_valid and hard_valid:
            eligible_rows.append((row_number, row))
    chosen_rows = eligible_rows[:batch_size]
    exact_count = len(chosen_rows) == batch_size
    coordinates: list[tuple[float, ...]] = []
    grids: list[tuple[int, ...]] = []
    chosen_grid_valid = True
    chosen_hard_valid = True
    chosen_debug = True
    chosen_experiment_false = True
    chosen_production_false = True
    for row_number, row in chosen_rows:
        normalized = tuple(
            _parse_finite_csv_number(
                row[column], field=f"shortlist row {row_number} {column}"
            )
            for column in _MEDOID_NORM_COLUMNS
        )
        grid_values = []
        for column in _MEDOID_GRID_COLUMNS:
            value = _parse_finite_csv_number(
                row[column], field=f"shortlist row {row_number} {column}"
            )
            if not value.is_integer():
                raise _error("Shortlist grid indices must be integers.")
            grid_values.append(int(value))
        coordinates.append(normalized)
        grids.append(tuple(grid_values))
        chosen_grid_valid = chosen_grid_valid and _parse_csv_boolean(
            row["all_grid_valid"],
            field=f"shortlist row {row_number} all_grid_valid",
        )
        chosen_hard_valid = chosen_hard_valid and _parse_csv_boolean(
            row["all_hard_distance_valid"],
            field=f"shortlist row {row_number} all_hard_distance_valid",
        )
        chosen_debug = chosen_debug and _parse_csv_boolean(
            row["debug_only"], field=f"shortlist row {row_number} debug_only"
        )
        chosen_experiment_false = chosen_experiment_false and not _parse_csv_boolean(
            row["approved_for_experiment"],
            field=f"shortlist row {row_number} approved_for_experiment",
        )
        chosen_production_false = chosen_production_false and not _parse_csv_boolean(
            row["approved_for_production"],
            field=f"shortlist row {row_number} approved_for_production",
        )

    finite_and_bounded = bool(
        exact_count
        and all(
            len(row) == len(_MEDOID_NORM_COLUMNS)
            and all(0.0 <= value <= 1.0 for value in row)
            for row in coordinates
        )
    )
    unique_and_on_grid = bool(
        exact_count and chosen_grid_valid and len(set(grids)) == batch_size
    )
    pairwise_minimum = 0.0
    if exact_count and len(coordinates) > 1:
        pairwise_minimum = min(
            math.dist(coordinates[left], coordinates[right])
            for left in range(len(coordinates))
            for right in range(left + 1, len(coordinates))
        )
    hard_distance_valid = bool(
        exact_count
        and chosen_hard_valid
        and pairwise_minimum + 1.0e-12 >= region_threshold
    )
    family_qualified = exact_count

    expected_checks = {
        "full_mode_eligible_for_consensus": full_mode_eligible,
        "largest_two_nested_regional_matches": (
            regional_match_count >= nested_match_minimum
        ),
        "largest_two_nested_mean_matched_distance": (
            mean_matched_distance <= mean_distance_maximum
        ),
        "five_regions_cover_three_core_families": (
            qualifying_region_count >= batch_size
        ),
        "exact_five_candidates": exact_count,
        "chosen_regions_cover_three_core_families": family_qualified,
        "finite_and_bounded": finite_and_bounded,
        "unique_and_on_grid": unique_and_on_grid,
        "chosen_pairwise_hard_distance_valid": hard_distance_valid,
        "debug_only": bool(exact_count and chosen_debug),
        "experimental_approval_false": bool(exact_count and chosen_experiment_false),
        "production_approval_false": bool(exact_count and chosen_production_false),
    }
    checks = manifest["consensus_checks"]
    if set(checks) != set(expected_checks):
        raise _error(
            "run_manifest.json consensus_checks do not contain the exact supported "
            "gate set."
        )
    for field, expected in expected_checks.items():
        if checks[field] is not expected:
            raise _error(
                f"run_manifest.json consensus check {field} disagrees with bundled "
                "evidence."
            )

    expected_observed: dict[str, int | float | bool] = {
        "largest_two_regional_matches_within_0_15": regional_match_count,
        "largest_two_mean_matched_distance": mean_matched_distance,
        "regions_with_minimum_family_coverage": qualifying_region_count,
        "consensus_candidate_count": len(chosen_rows),
        "chosen_regions_family_qualified": family_qualified,
        "chosen_candidates_finite_and_bounded": finite_and_bounded,
        "chosen_candidates_unique_and_on_grid": unique_and_on_grid,
        "chosen_pairwise_minimum_distance": pairwise_minimum,
        "required_pairwise_minimum_distance": region_threshold,
        "chosen_hard_distance_valid": hard_distance_valid,
        "full_mode_eligible_for_consensus": full_mode_eligible,
    }
    observed = manifest["consensus_observed"]
    if set(observed) != set(expected_observed):
        raise _error(
            "run_manifest.json consensus_observed does not contain the exact "
            "supported evidence set."
        )
    for field, expected in expected_observed.items():
        if not _values_match(observed[field], expected):
            raise _error(
                f"run_manifest.json consensus observed value {field} disagrees with "
                "bundled evidence."
            )
    if manifest["consensus_passed"] is not all(expected_checks.values()):
        raise _error(
            "run_manifest.json consensus_passed disagrees with recomputed gates."
        )
    return tuple(grids)


def _validate_consensus_batch(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    repository_root: Path,
    expected_grid_tuples: tuple[tuple[int, ...], ...],
) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            header = tuple(reader.fieldnames or ())
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(
            "r1_consensus_debug_batch.csv must be readable valid CSV."
        ) from exc
    required = {
        "consensus_candidate_id",
        "selection_order",
        "distinct_nonbaseline_family_count",
        "all_grid_valid",
        "all_hard_distance_valid",
        *_D2D_INPUT_COLUMNS,
        *_MEDOID_GRID_COLUMNS,
        *_MEDOID_NORM_COLUMNS,
        *_CSV_STAMP_COLUMNS,
    }
    missing = sorted(required - set(header))
    if missing:
        raise _error(
            "r1_consensus_debug_batch.csv is missing safety/recipe columns: "
            f"{missing}."
        )
    if len(rows) != 5:
        raise _error("r1_consensus_debug_batch.csv must contain exactly five rows.")
    specs = _resolved_input_specs(config, repository_root)
    grid_tuples: list[tuple[int, ...]] = []
    normalized_rows: list[tuple[float, ...]] = []
    orders: list[int] = []
    family_minimum = int(
        manifest["stability_criteria"]["minimum_nonbaseline_core_family_coverage"]
    )
    for row_number, row in enumerate(rows, start=2):
        order_value = _parse_finite_csv_number(
            row["selection_order"], field=f"row {row_number} selection_order"
        )
        if not order_value.is_integer():
            raise _error("Consensus selection_order values must be integers.")
        orders.append(int(order_value))
        if (
            row["all_grid_valid"].strip().casefold() != "true"
            or row["all_hard_distance_valid"].strip().casefold() != "true"
        ):
            raise _error("Every consensus row must pass grid and hard-distance flags.")
        family_count = _parse_finite_csv_number(
            row["distinct_nonbaseline_family_count"],
            field=f"row {row_number} family coverage",
        )
        if family_count < family_minimum:
            raise _error("Every consensus row must meet the family-coverage criterion.")
        grid_values: list[int] = []
        norm_values: list[float] = []
        for dimension, (name, spec) in enumerate(zip(_D2D_INPUT_COLUMNS, specs)):
            grid_value = _parse_finite_csv_number(
                row[f"medoid_grid_{dimension}"],
                field=f"row {row_number} medoid_grid_{dimension}",
            )
            if not grid_value.is_integer():
                raise _error("Consensus grid indices must be integers.")
            grid_index = int(grid_value)
            start = float(spec["start"])
            stop = float(spec["stop"])
            step = float(spec["step"])
            maximum_index = int(round((stop - start) / step))
            if grid_index < 0 or grid_index > maximum_index:
                raise _error("Consensus grid index lies outside an allowed D2D grid.")
            physical = _parse_finite_csv_number(
                row[name], field=f"row {row_number} {name}"
            )
            expected_physical = start + grid_index * step
            if not math.isclose(
                physical, expected_physical, rel_tol=0.0, abs_tol=1.0e-10
            ):
                raise _error(
                    "Consensus physical values do not match their grid indices."
                )
            normalized = _parse_finite_csv_number(
                row[f"medoid_norm_{dimension}"],
                field=f"row {row_number} medoid_norm_{dimension}",
            )
            expected_normalized = (
                0.0 if stop == start else (physical - start) / (stop - start)
            )
            if not 0.0 <= normalized <= 1.0 or not math.isclose(
                normalized, expected_normalized, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise _error(
                    "Consensus normalized values are invalid for the D2D grid."
                )
            grid_values.append(grid_index)
            norm_values.append(normalized)
        grid_tuples.append(tuple(grid_values))
        normalized_rows.append(tuple(norm_values))
    if orders != [1, 2, 3, 4, 5]:
        raise _error(
            "Consensus rows and selection_order must be exactly ordered 1 through 5."
        )
    if len(set(grid_tuples)) != 5:
        raise _error("Consensus candidates must be unique exact grid recipes.")
    if tuple(grid_tuples) != expected_grid_tuples:
        raise _error(
            "Consensus grid recipes do not match the eligible shortlist head in "
            "selection order."
        )
    minimum_required = float(
        manifest["stability_criteria"]["required_pairwise_minimum_distance"]
    )
    pairwise_minimum = math.inf
    for left in range(5):
        for right in range(left + 1, 5):
            distance = math.sqrt(
                sum(
                    (
                        normalized_rows[left][dimension]
                        - normalized_rows[right][dimension]
                    )
                    ** 2
                    for dimension in range(len(_D2D_INPUT_COLUMNS))
                )
            )
            pairwise_minimum = min(pairwise_minimum, distance)
    if pairwise_minimum + 1.0e-12 < minimum_required:
        raise _error(
            "Consensus candidates violate the required hard pairwise distance."
        )
    checks = manifest["consensus_checks"]
    if not checks or any(value is not True for value in checks.values()):
        raise _error(
            "A consensus artifact requires every manifest consensus check to pass."
        )
    if manifest["consensus_observed"].get("consensus_candidate_count") != 5:
        raise _error("Manifest consensus_candidate_count must equal five.")


def _validate_no_stable_reason(
    payload: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    _require_mapping_fields(
        payload,
        ("message", "checks", "failed_checks", "observed", "consensus_passed"),
        label=NO_STABLE_BATCH_REASON_FILE,
    )
    failed = payload["failed_checks"]
    checks = payload["checks"]
    expected_failed = [
        name for name, passed in manifest["consensus_checks"].items() if passed is False
    ]
    if (
        payload["consensus_passed"] is not False
        or not isinstance(failed, list)
        or not failed
        or not isinstance(checks, Mapping)
        or any(not isinstance(item, str) for item in failed)
        or len(failed) != len(expected_failed)
        or set(failed) != set(expected_failed)
    ):
        raise _error("The no-stable-batch reason must name every failed false check.")
    if dict(checks) != dict(manifest["consensus_checks"]):
        raise _error("No-stable-batch checks must match run_manifest.json.")
    if (
        not isinstance(payload["observed"], Mapping)
        or not isinstance(payload["message"], str)
        or not payload["message"].strip()
    ):
        raise _error("The no-stable-batch reason needs observed values and a message.")
    if dict(payload["observed"]) != dict(manifest["consensus_observed"]):
        raise _error("No-stable-batch observed values must match run_manifest.json.")


def _validate_file_surface(files: tuple[Path, ...], output_dir: Path) -> None:
    allowed_suffixes = {".csv", ".json", ".png", ".txt", ".yaml"}
    for path in files:
        relative = _relative(path, output_dir)
        if path.suffix.casefold() not in allowed_suffixes:
            raise _error(
                f"Unsupported artifact format {relative}; recipe-bearing formats "
                "outside stamped CSV/JSON/PNG are forbidden."
            )
        if path.suffix.casefold() == ".txt":
            if (
                relative != "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt"
                or path.read_text(encoding="utf-8").strip() != DEBUG_WATERMARK
            ):
                raise _error(
                    "The only permitted text artifact is the exact debug marker."
                )
        if (
            path.suffix.casefold() == ".yaml"
            and relative != "resolved_debug_config.yaml"
        ):
            raise _error("Only resolved_debug_config.yaml is permitted in the bundle.")


def _all_regular_files(output_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise _error(
                f"Artifact bundles may not contain symbolic links: "
                f"{_relative(path, output_dir)}."
            )
        if path.is_file():
            resolved = path.resolve()
            if output_dir not in resolved.parents:
                raise _error(
                    f"Artifact resolves outside the output bundle: "
                    f"{_relative(path, output_dir)}."
                )
            paths.append(path)
    return tuple(sorted(paths, key=lambda item: _relative(item, output_dir)))


def _validate_replicate_worklists(
    files: tuple[Path, ...], output_dir: Path, *, consensus_exists: bool
) -> None:
    for path in files:
        lowered = path.name.casefold()
        if "replicate" not in lowered and "worklist" not in lowered:
            continue
        relative = _relative(path, output_dir)
        if not consensus_exists:
            raise _error(
                f"Replicate worklist artifact {relative} is forbidden without "
                "r1_consensus_debug_batch.csv."
            )
        if "debug_preview" not in lowered and "debug-preview" not in lowered:
            raise _error(
                f"Replicate worklist artifact {relative} must be explicitly named "
                "as a debug preview."
            )


def validate_step2c_artifact_bundle(
    output_dir: str | Path,
    *,
    repository_root: str | Path,
) -> Step2CArtifactValidation:
    """Validate one completed local Step 2C output directory without mutation.

    The trusted ``repository_root`` anchors the relative ``outputs.root`` in the
    resolved configuration and any relative source-workbook path in the run
    manifest. The function performs only reads and returns a content-hash map for
    every artifact after all contract checks pass.
    """

    repository = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    if not repository.is_dir():
        raise _error(f"repository_root must be an existing directory: {repository}.")
    if not destination.is_dir():
        raise _error(f"output_dir must be an existing directory: {destination}.")

    config_path = destination / "resolved_debug_config.yaml"
    if not config_path.is_file():
        raise _error("Missing required top-level artifact: resolved_debug_config.yaml.")
    config = _read_yaml_mapping(config_path, label="resolved_debug_config.yaml")
    _validate_config_stamps(config)
    ignored_root = _configured_output_root(config, repository)
    if destination == ignored_root or ignored_root not in destination.parents:
        raise _error(
            f"output_dir must be a strict descendant of configured ignored root "
            f"{ignored_root}."
        )
    if (repository / ".git").exists():
        try:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", str(destination)],
                cwd=repository,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise _error(
                "Unable to verify that the Step 2C output is Git-ignored."
            ) from exc
        if ignored.returncode != 0:
            raise _error("The configured Step 2C output directory must be Git-ignored.")

    missing_top_level = [
        name for name in REQUIRED_TOP_LEVEL_FILES if not (destination / name).is_file()
    ]
    if missing_top_level:
        raise _error(
            "Missing required top-level Step 2C artifact(s): " f"{missing_top_level}."
        )

    consensus_path = destination / CONSENSUS_BATCH_FILE
    no_stable_path = destination / NO_STABLE_BATCH_REASON_FILE
    consensus_exists = consensus_path.is_file()
    no_stable_exists = no_stable_path.is_file()
    if consensus_exists == no_stable_exists:
        raise _error(
            "Exactly one conditional artifact is required: "
            f"{CONSENSUS_BATCH_FILE} or {NO_STABLE_BATCH_REASON_FILE}."
        )
    conditional_artifact = (
        CONSENSUS_BATCH_FILE if consensus_exists else NO_STABLE_BATCH_REASON_FILE
    )

    for directory in REQUIRED_PLOT_DIRECTORIES:
        path = destination / directory
        if not path.is_dir() or path.is_symlink():
            raise _error(f"Missing required plot-topic directory: {directory}.")
    missing_plots = [
        name for name in REQUIRED_PLOT_FILES if not (destination / name).is_file()
    ]
    if missing_plots:
        raise _error(f"Missing required plot-topic artifact(s): {missing_plots}.")

    files = _all_regular_files(destination)
    _validate_file_surface(files, destination)
    _validate_replicate_worklists(files, destination, consensus_exists=consensus_exists)
    csv_files = tuple(
        _relative(path, destination)
        for path in files
        if path.suffix.casefold() == ".csv"
    )
    json_files = tuple(
        _relative(path, destination)
        for path in files
        if path.suffix.casefold() == ".json"
    )
    png_files = tuple(
        _relative(path, destination)
        for path in files
        if path.suffix.casefold() == ".png"
    )
    for relative in csv_files:
        header, row_count = _validate_csv_stamp(destination / relative, destination)
        _validate_required_csv_schema(relative, header, row_count)
    json_payloads: dict[str, dict[str, Any]] = {}
    for relative in json_files:
        json_payloads[relative] = _read_and_validate_json_stamp(
            destination / relative, destination
        )
    for relative in png_files:
        _validate_png_watermark(destination / relative, destination)

    manifest = json_payloads.get("run_manifest.json")
    if manifest is None:
        raise _error("run_manifest.json was not validated as a top-level JSON object.")
    _validate_manifest_provenance(manifest, repository, ignored_root, config)
    if consensus_exists:
        if manifest.get("consensus_passed") is not True:
            raise _error(
                "r1_consensus_debug_batch.csv requires consensus_passed=true in "
                "run_manifest.json."
            )
        if manifest.get("mode") != "full":
            raise _error(
                "r1_consensus_debug_batch.csv is forbidden outside a full run."
            )
    else:
        if manifest.get("consensus_passed") is not False:
            raise _error(
                "r1_no_stable_batch_reason.json requires consensus_passed=false in "
                "run_manifest.json."
            )
    expected_consensus_grids = _validate_consensus_evidence(
        destination, manifest=manifest, config=config
    )
    if consensus_exists:
        _validate_consensus_batch(
            consensus_path,
            manifest=manifest,
            config=config,
            repository_root=repository,
            expected_grid_tuples=expected_consensus_grids,
        )
    else:
        reason_payload = json_payloads.get(NO_STABLE_BATCH_REASON_FILE)
        if reason_payload is None:
            raise _error("The no-stable-batch reason was not validated as JSON.")
        _validate_no_stable_reason(reason_payload, manifest)
    workbook_path, workbook_hash, workbook_mtime = _validate_workbook_proof(
        manifest, repository, destination
    )

    artifact_sha256 = {
        _relative(path, destination): _sha256_file(path) for path in files
    }
    return Step2CArtifactValidation(
        output_dir=destination,
        ignored_output_root=ignored_root,
        conditional_artifact=conditional_artifact,
        workbook_path=workbook_path,
        workbook_sha256=workbook_hash,
        workbook_mtime_ns=workbook_mtime,
        csv_files_checked=csv_files,
        json_files_checked=json_files,
        png_files_checked=png_files,
        artifact_sha256=artifact_sha256,
    )


__all__ = [
    "CONSENSUS_BATCH_FILE",
    "DEBUG_WATERMARK",
    "NO_STABLE_BATCH_REASON_FILE",
    "PUBLIC_SUMMARY_ARCHIVE_ROOT",
    "PUBLIC_SUMMARY_CSV_FILES",
    "PUBLIC_SUMMARY_DEBUG_MARKER_FILE",
    "PUBLIC_SUMMARY_FILES",
    "PUBLIC_SUMMARY_MANIFEST_FILE",
    "PUBLIC_SUMMARY_README_FILE",
    "PUBLIC_SUMMARY_SCHEMA_VERSION",
    "REQUIRED_CSV_COLUMNS",
    "REQUIRED_PLOT_DIRECTORIES",
    "REQUIRED_PLOT_FILES",
    "REQUIRED_TOP_LEVEL_FILES",
    "Step2CArtifactContractError",
    "Step2CArtifactValidation",
    "Step2CPublicSummaryValidation",
    "validate_step2c_artifact_bundle",
    "validate_step2c_public_summary_archive",
]
