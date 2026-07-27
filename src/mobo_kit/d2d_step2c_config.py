"""Fail-closed configuration for the D2D Step 2C robustness study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .d2d_campaign import (
    D2D_DEBUG_WATERMARK,
    D2D_INPUT_COLUMNS,
    D2D_OBJECTIVE_COLUMNS,
    D2D_OBJECTIVE_NAMES,
    D2D_REFERENCE_POINT_UTILITY,
    OffGridObservedException,
    ResolvedD2DDebugConfig,
    load_d2d_debug_config,
)


STEP2C_SCHEMA_VERSION = "d2d-step2c-robustness-debug-1"
STEP2C_OUTPUT_ROOT = "local_outputs/d2d_step2c_robustness"
STEP2C_NESTED_POOL_SIZES = (16384, 32768, 65536, 131072)
STEP2C_SECONDARY_SOBOL_SEEDS = (137, 911)
STEP2C_BETA_VALUES = (1.0, 4.0, 9.0)
STEP2C_BOUND_POLICIES = ("none", "clip_ucb")
STEP2C_PRIVATE_SOURCE_KIND = "private_pinned_workbook"
STEP2C_SYNTHETIC_SOURCE_KIND = "sanitized_synthetic_ci"
STEP2C_RUNTIME_GENERATED = "runtime_generated"


class Step2CConfigError(ValueError):
    """Raised when a Step 2C config weakens the debug-only contract."""


@dataclass(frozen=True)
class PenaltyVariant:
    label: str
    radius: float | None
    min_batch_distance: float


@dataclass(frozen=True)
class ExecutionModeSettings:
    nested_unique_sizes: tuple[int, ...]
    anchors_per_selection_step: int
    omitted_sample_ids: tuple[int, ...]
    mc_comparison_samples: int


@dataclass(frozen=True)
class ResolvedStep2CConfig:
    """Validated Step 2C settings plus the audited Step 2B ingestion contract."""

    raw: dict[str, Any]
    config_path: Path
    config_sha256: str
    resolved_config_hash: str
    base_config_path: Path
    base: ResolvedD2DDebugConfig
    workbook_source_kind: str
    workbook_relative_path: str
    workbook_expected_sha256: str
    resolved_off_grid_exceptions: tuple[OffGridObservedException, ...]
    objective_bounds: tuple[tuple[float | None, float | None], ...]
    beta_values: tuple[float, ...]
    primary_beta: float
    moment_method: str
    score_chunk_size: int
    positive_hvi_threshold: float
    numeric_tolerance: float
    bound_policies: tuple[str, ...]
    primary_bound_policy: str
    mc_comparison_samples: int
    mc_comparison_seed: int
    primary_sobol_seed: int
    secondary_sobol_seeds: tuple[int, ...]
    nested_pool_sizes: tuple[int, ...]
    refinement_anchors: int
    refinement_max_sweeps: int
    refinement_tolerance: float
    penalty_variants: tuple[PenaltyVariant, ...]
    primary_penalty_variant: str
    model_variant_names: tuple[str, ...]
    primary_model_variant: str
    influence_pool_size: int
    influence_pool_seed: int
    influence_sample_ids: tuple[int, ...]
    influence_top_k: int
    regional_thresholds: tuple[float, ...]
    region_threshold: float
    region_sensitivity_thresholds: tuple[float, ...]
    shortlist_min: int
    shortlist_max: int
    consensus_batch_size: int
    consensus_nested_match_min: int
    consensus_mean_distance_max: float
    consensus_family_coverage_min: int
    execution_modes: dict[str, ExecutionModeSettings]
    model_seed: int
    output_root: str
    create_portable_zip: bool

    # These properties make the object safe to pass to the already-audited
    # Step 2B workbook/training adapter without duplicating that boundary.
    @property
    def design(self):
        return self.base.design

    @property
    def expected_sample_ids(self) -> tuple[int, ...]:
        return self.base.expected_sample_ids

    @property
    def reference_point_utility(self) -> np.ndarray:
        return self.base.reference_point_utility.copy()

    @property
    def control_sample_ids(self) -> tuple[int, ...]:
        return self.base.control_sample_ids

    @property
    def control_measurement_provenance_assumption(self) -> str:
        return self.base.control_measurement_provenance_assumption

    @property
    def off_grid_exceptions(self):
        return self.resolved_off_grid_exceptions

    def mode(self, name: str) -> ExecutionModeSettings:
        try:
            return self.execution_modes[name]
        except KeyError as exc:
            raise Step2CConfigError("execution mode must be 'fast' or 'full'.") from exc


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Step2CConfigError(f"{field} must be a mapping.")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise Step2CConfigError(f"{field} must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise Step2CConfigError(f"{field} must be an integer >= {minimum}.")
    return result


def _number(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise Step2CConfigError(f"{field} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise Step2CConfigError(f"{field} must be finite and >= {minimum}.")
    return result


def _int_tuple(value: Any, *, field: str, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise Step2CConfigError(f"{field} must be a non-empty integer list.")
    return tuple(
        _integer(item, field=f"{field}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _number_tuple(value: Any, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise Step2CConfigError(f"{field} must be a non-empty numeric list.")
    return tuple(
        _number(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_repository_path(raw_path: Any, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise Step2CConfigError(f"{field} must be a nonblank repository path.")
    repository_root = Path(__file__).resolve().parents[2]
    candidate = Path(raw_path)
    resolved = (
        (repository_root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if resolved != repository_root and repository_root not in resolved.parents:
        raise Step2CConfigError(f"{field} must resolve inside the repository.")
    return resolved


def _validate_debug_boundary(raw: dict[str, Any]) -> None:
    exact = {
        "schema_version": STEP2C_SCHEMA_VERSION,
        "run_mode": "debug",
        "approved_for_experiment": False,
        "approved_for_production": False,
        "debug_watermark": D2D_DEBUG_WATERMARK,
    }
    for field, expected in exact.items():
        actual = raw.get(field)
        if isinstance(expected, bool):
            valid = isinstance(actual, bool) and actual is expected
        else:
            valid = actual == expected
        if not valid:
            raise Step2CConfigError(f"{field} must be exactly {expected!r}.")
    campaign_id = raw.get("campaign_id")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id.strip()
        or len(campaign_id) > 128
        or not campaign_id.endswith("-debug")
        or "/" in campaign_id
        or "\\" in campaign_id
    ):
        raise Step2CConfigError(
            "campaign_id must be a nonblank, path-free identifier ending in '-debug'."
        )


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Step2CConfigError(f"{field} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Step2CConfigError(
            f"{field} must be a 64-character SHA-256 digest."
        ) from exc
    return value.upper()


def _relative_path_under(raw_path: Any, root: str, *, field: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise Step2CConfigError(f"{field} must be a nonblank relative path.")
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise Step2CConfigError(f"{field} must be a safe repository-relative path.")
    normalized = candidate.as_posix()
    if not normalized.startswith(f"{root}/"):
        raise Step2CConfigError(f"{field} must be below {root}/.")
    _resolve_repository_path(normalized, field=field)
    return normalized


def _validate_workbook_and_objectives(
    raw: dict[str, Any], base: ResolvedD2DDebugConfig
) -> tuple[str, str, str, tuple[tuple[float | None, float | None], ...]]:
    workbook = _mapping(raw.get("workbook"), field="workbook")
    source_kind = workbook.get("source_kind")
    if source_kind == STEP2C_SYNTHETIC_SOURCE_KIND:
        allowed_keys = {"source_kind", "path", "expected_sha256"}
        if set(workbook) not in ({"source_kind"}, allowed_keys):
            raise Step2CConfigError(
                "Synthetic workbook settings must be runtime-generated or contain "
                "only source_kind, path, and expected_sha256."
            )
        if set(workbook) == {"source_kind"}:
            workbook_path = STEP2C_RUNTIME_GENERATED
            workbook_hash = STEP2C_RUNTIME_GENERATED.upper()
        else:
            workbook_path = _relative_path_under(
                workbook["path"], STEP2C_OUTPUT_ROOT, field="workbook.path"
            )
            workbook_hash = _sha256(
                workbook["expected_sha256"], field="workbook.expected_sha256"
            )
    else:
        source_kind = STEP2C_PRIVATE_SOURCE_KIND
        expected_keys = {
            "profile",
            "path",
            "sheet",
            "expected_range",
            "expected_sha256",
            "objective_columns",
        }
        if set(workbook) != expected_keys:
            raise Step2CConfigError(
                "Private workbook settings must preserve the complete read-only "
                "ingestion contract."
            )
        if (
            workbook["profile"] != base.workbook_profile
            or workbook["sheet"] != base.workbook_sheet
            or workbook["expected_range"] != base.expected_content_range
            or workbook["objective_columns"]
            != [objective["excel_column"] for objective in base.raw["objectives"]]
        ):
            raise Step2CConfigError(
                "Private workbook settings differ from the audited base contract."
            )
        workbook_path = _relative_path_under(
            workbook["path"], "local_inputs", field="workbook.path"
        )
        workbook_hash = _sha256(
            workbook["expected_sha256"], field="workbook.expected_sha256"
        )
        if workbook_hash != base.expected_workbook_sha256:
            raise Step2CConfigError(
                "Private workbook hash differs from the audited base contract."
            )

    objectives = _mapping(raw.get("objectives"), field="objectives")
    if objectives.get("order") != list(D2D_OBJECTIVE_NAMES):
        raise Step2CConfigError("objectives.order must preserve the three D2D scores.")
    if objectives.get("source_columns") != list(D2D_OBJECTIVE_COLUMNS):
        raise Step2CConfigError(
            "objectives.source_columns must preserve Z/AA/AB roles."
        )
    reference = np.asarray(objectives.get("reference_point"), dtype=float)
    if reference.shape != (3,) or not np.array_equal(
        reference, D2D_REFERENCE_POINT_UTILITY
    ):
        raise Step2CConfigError(
            "objectives.reference_point must be [-0.01, -10.0, -0.01]."
        )
    if objectives.get("transforms") != ["identity"] * 3:
        raise Step2CConfigError("All Step 2C objective transforms must be identity.")
    if objectives.get("directions") != ["maximize"] * 3:
        raise Step2CConfigError("All Step 2C objectives must be maximized.")
    expected_bounds = {
        "uniformity_score": {"lower": 0.0, "upper": 1.0},
        "optoelectronic_score": {"lower": None, "upper": None},
        "thickness_score": {"lower": 0.0, "upper": 1.0},
    }
    if objectives.get("bounds") != expected_bounds:
        raise Step2CConfigError(
            "objectives.bounds must preserve the declared score support."
        )
    mismatch = objectives.get("known_uniformity_mismatch")
    if mismatch != {
        "allowed_in_debug": True,
        "blocks_experimental_approval": True,
    }:
        raise Step2CConfigError(
            "The known Uniformity mismatch must remain visible and approval-blocking."
        )
    return (
        source_kind,
        workbook_path,
        workbook_hash,
        ((0.0, 1.0), (None, None), (0.0, 1.0)),
    )


def _validate_r0(
    raw: dict[str, Any],
    base: ResolvedD2DDebugConfig,
    *,
    workbook_source_kind: str,
) -> tuple[OffGridObservedException, ...]:
    section = _mapping(raw.get("r0"), field="r0")
    if section.get("inherit_from_base") is True:
        allowed_keys = {"inherit_from_base", "synthetic_off_grid_override"}
        if set(section) - allowed_keys:
            raise Step2CConfigError("r0 contains unsupported inherited settings.")
        override = section.get("synthetic_off_grid_override")
        if override is None:
            resolved = base.off_grid_exceptions
        else:
            if workbook_source_kind != STEP2C_SYNTHETIC_SOURCE_KIND:
                raise Step2CConfigError(
                    "An r0 synthetic override requires a synthetic workbook source."
                )
            value = _mapping(override, field="r0.synthetic_off_grid_override")
            if set(value) != {"sample_id", "field", "value"}:
                raise Step2CConfigError(
                    "r0.synthetic_off_grid_override has unsupported fields."
                )
            if len(base.off_grid_exceptions) != 1:
                raise Step2CConfigError(
                    "A synthetic override requires one inherited off-grid exception."
                )
            inherited = base.off_grid_exceptions[0]
            sample_id = _integer(
                value["sample_id"], field="r0.synthetic_off_grid_override.sample_id"
            )
            field = value["field"]
            observed = _number(
                value["value"], field="r0.synthetic_off_grid_override.value"
            )
            if sample_id != inherited.sample_id or field != inherited.input_name:
                raise Step2CConfigError(
                    "The synthetic off-grid override must target the inherited row "
                    "and input."
                )
            dimension = tuple(base.design.names).index(field)
            grid = np.asarray(base.design.var_array[dimension], dtype=float)
            if np.any(np.isclose(observed, grid, rtol=0.0, atol=1.0e-12)):
                raise Step2CConfigError(
                    "The synthetic off-grid override must remain outside the grid."
                )
            if observed < float(grid.min()) or observed > float(grid.max()):
                raise Step2CConfigError(
                    "The synthetic off-grid override must remain within input bounds."
                )
            resolved = (
                OffGridObservedException(
                    sample_id=sample_id,
                    input_name=field,
                    observed_value=observed,
                    reason="sanitized synthetic off-grid fixture",
                ),
            )
    else:
        expected_exceptions = [
            {
                "sample_id": exception.sample_id,
                "field": exception.input_name,
                "value": exception.observed_value,
                "include_in_gp": True,
                "include_in_distance_reference": True,
                "include_in_grid_index_exclusion": False,
            }
            for exception in base.off_grid_exceptions
        ]
        expected = {
            "sample_count": len(base.expected_sample_ids),
            "control_sample_ids": list(base.control_sample_ids),
            "include_control_in_primary_model": True,
            "control_measurement_provenance_assumption": (
                base.control_measurement_provenance_assumption
            ),
            "off_grid_observed_exceptions": expected_exceptions,
        }
        if section != expected:
            raise Step2CConfigError(
                "r0 must preserve the audited control/off-grid policy."
            )
        resolved = base.off_grid_exceptions
    if raw.get("constraints") != []:
        raise Step2CConfigError("constraints must remain an explicit empty list.")
    return tuple(resolved)


def _validate_penalties(raw: dict[str, Any]) -> tuple[tuple[PenaltyVariant, ...], str]:
    section = _mapping(raw.get("local_penalty_study"), field="local_penalty_study")
    if section.get("hard_distance_relaxation") is not False:
        raise Step2CConfigError("Hard-distance relaxation must remain false.")
    expected = (
        PenaltyVariant("no_soft_no_hard", None, 0.0),
        PenaltyVariant("no_soft_hard_0_15", None, 0.15),
        PenaltyVariant("radius_0_15", 0.15, 0.15),
        PenaltyVariant("radius_0_25", 0.25, 0.15),
        PenaltyVariant("radius_0_35", 0.35, 0.15),
    )
    variants = section.get("variants")
    if not isinstance(variants, list) or len(variants) != len(expected):
        raise Step2CConfigError(
            "local_penalty_study.variants must contain five variants."
        )
    parsed: list[PenaltyVariant] = []
    for index, item in enumerate(variants):
        value = _mapping(item, field=f"local_penalty_study.variants[{index}]")
        radius_raw = value.get("radius")
        radius = (
            None
            if radius_raw is None
            else _number(
                radius_raw,
                field=f"local_penalty_study.variants[{index}].radius",
                minimum=0.0,
            )
        )
        parsed.append(
            PenaltyVariant(
                label=str(value.get("label", "")),
                radius=radius,
                min_batch_distance=_number(
                    value.get("min_batch_distance"),
                    field=f"local_penalty_study.variants[{index}].min_batch_distance",
                    minimum=0.0,
                ),
            )
        )
    if tuple(parsed) != expected:
        raise Step2CConfigError("local_penalty_study variants changed from the pack.")
    if section.get("primary_variant") != "radius_0_25":
        raise Step2CConfigError("primary local-penalty variant must be radius_0_25.")
    if (
        section.get("min_observed_distance") != 0.0
        or section.get("dimension_weights") is not None
    ):
        raise Step2CConfigError(
            "Observed-distance and dimension-weight settings changed."
        )
    return tuple(parsed), "radius_0_25"


def _validate_execution_modes(
    raw: dict[str, Any], *, expected_sample_ids: tuple[int, ...]
) -> dict[str, ExecutionModeSettings]:
    section = _mapping(raw.get("execution_modes"), field="execution_modes")
    if set(section) != {"fast", "full"}:
        raise Step2CConfigError("execution_modes must contain exactly fast and full.")
    result: dict[str, ExecutionModeSettings] = {}
    for name in ("fast", "full"):
        item = _mapping(section[name], field=f"execution_modes.{name}")
        result[name] = ExecutionModeSettings(
            nested_unique_sizes=_int_tuple(
                item.get("nested_unique_sizes"),
                field=f"execution_modes.{name}.nested_unique_sizes",
                minimum=1,
            ),
            anchors_per_selection_step=_integer(
                item.get("anchors_per_selection_step"),
                field=f"execution_modes.{name}.anchors_per_selection_step",
                minimum=1,
            ),
            omitted_sample_ids=_int_tuple(
                item.get("omitted_sample_ids"),
                field=f"execution_modes.{name}.omitted_sample_ids",
                minimum=1,
            ),
            mc_comparison_samples=_integer(
                item.get("mc_comparison_samples"),
                field=f"execution_modes.{name}.mc_comparison_samples",
                minimum=2,
            ),
        )
    expected = {
        "fast": ExecutionModeSettings(
            nested_unique_sizes=(512, 1024, 2048, 4096),
            anchors_per_selection_step=8,
            omitted_sample_ids=expected_sample_ids[:3],
            mc_comparison_samples=256,
        ),
        "full": ExecutionModeSettings(
            nested_unique_sizes=STEP2C_NESTED_POOL_SIZES,
            anchors_per_selection_step=64,
            omitted_sample_ids=expected_sample_ids,
            mc_comparison_samples=2048,
        ),
    }
    if result != expected:
        raise Step2CConfigError(
            "execution_modes must preserve the declared fast/full audit settings."
        )
    return result


def load_step2c_config(path: str | Path) -> ResolvedStep2CConfig:
    """Load and validate the full debug-only Step 2C robustness contract."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Step 2C config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise Step2CConfigError("Step 2C config must contain a mapping.")
    _validate_debug_boundary(raw)

    base_path = _resolve_repository_path(
        raw.get("base_step2b_config"), field="base_step2b_config"
    )
    workbook_section = _mapping(raw.get("workbook"), field="workbook")
    allow_public_template = (
        workbook_section.get("source_kind") == STEP2C_SYNTHETIC_SOURCE_KIND
    )
    base = load_d2d_debug_config(base_path, allow_public_template=allow_public_template)
    (
        workbook_source_kind,
        workbook_path,
        workbook_hash,
        objective_bounds,
    ) = _validate_workbook_and_objectives(raw, base)
    off_grid_exceptions = _validate_r0(
        raw, base, workbook_source_kind=workbook_source_kind
    )

    ucb = _mapping(raw.get("ucb_hvi"), field="ucb_hvi")
    beta_values = _number_tuple(ucb.get("beta_values"), field="ucb_hvi.beta_values")
    if beta_values != STEP2C_BETA_VALUES:
        raise Step2CConfigError("ucb_hvi.beta_values must be [1.0, 4.0, 9.0].")
    bound_policies_raw = ucb.get("bound_policies")
    if (
        not isinstance(bound_policies_raw, list)
        or tuple(bound_policies_raw) != STEP2C_BOUND_POLICIES
    ):
        raise Step2CConfigError("ucb_hvi.bound_policies must be [none, clip_ucb].")
    if ucb.get("moment_method") != "analytic_identity":
        raise Step2CConfigError("ucb_hvi.moment_method must be analytic_identity.")
    primary_beta = _number(ucb.get("primary_beta"), field="ucb_hvi.primary_beta")
    score_chunk_size = _integer(
        ucb.get("score_chunk_size"), field="ucb_hvi.score_chunk_size", minimum=1
    )
    numeric_tolerance = _number(
        ucb.get("numeric_tolerance"),
        field="ucb_hvi.numeric_tolerance",
        minimum=0.0,
    )
    primary_bound_policy = str(ucb.get("primary_bound_policy"))
    mc_comparison_samples = _integer(
        ucb.get("mc_comparison_samples"),
        field="ucb_hvi.mc_comparison_samples",
        minimum=2,
    )
    mc_comparison_seed = _integer(
        ucb.get("mc_comparison_seed"), field="ucb_hvi.mc_comparison_seed"
    )
    if (
        primary_beta,
        score_chunk_size,
        numeric_tolerance,
        primary_bound_policy,
        mc_comparison_samples,
        mc_comparison_seed,
    ) != (4.0, 2048, 1.0e-12, "clip_ucb", 2048, 73):
        raise Step2CConfigError(
            "The primary UCB-HVI, clipping, numeric, and MC-audit settings changed."
        )
    positive_threshold = _number(
        ucb.get("positive_hvi_threshold"),
        field="ucb_hvi.positive_hvi_threshold",
        minimum=0.0,
    )
    if positive_threshold != 0.0:
        raise Step2CConfigError("positive_hvi_threshold must express raw HVI > 0.")

    search = _mapping(raw.get("candidate_search"), field="candidate_search")
    if (
        search.get("method") != "nested_sobol_grid_indices"
        or search.get("scramble") is not True
    ):
        raise Step2CConfigError(
            "candidate_search must use scrambled nested Sobol grid indices."
        )
    nested = _int_tuple(
        search.get("nested_unique_sizes"),
        field="candidate_search.nested_unique_sizes",
        minimum=1,
    )
    if (
        nested != STEP2C_NESTED_POOL_SIZES
        or search.get("primary_full_size") != nested[-1]
    ):
        raise Step2CConfigError(
            "candidate_search must preserve the four declared full sizes."
        )
    if search.get("preserve_accepted_prefix_nesting") is not True:
        raise Step2CConfigError("Accepted-prefix nesting must remain mandatory.")
    primary_seed = _integer(
        search.get("primary_seed"), field="candidate_search.primary_seed"
    )
    secondary_seeds = _int_tuple(
        search.get("secondary_seeds"), field="candidate_search.secondary_seeds"
    )
    if primary_seed != 73 or secondary_seeds != STEP2C_SECONDARY_SOBOL_SEEDS:
        raise Step2CConfigError("Sobol seeds must remain 73 / [137, 911].")

    refinement = _mapping(raw.get("local_refinement"), field="local_refinement")
    expected_refinement_literals = {
        "enabled": True,
        "coordinate_values": "all_allowed_grid_values",
        "stable_tie_break": "lower_grid_index",
    }
    for field, expected in expected_refinement_literals.items():
        if refinement.get(field) != expected:
            raise Step2CConfigError(f"local_refinement.{field} must be {expected!r}.")
    anchors = _integer(
        refinement.get("anchors_per_selection_step"),
        field="local_refinement.anchors_per_selection_step",
        minimum=1,
    )
    sweeps = _integer(
        refinement.get("max_sweeps"), field="local_refinement.max_sweeps", minimum=1
    )
    tolerance = _number(
        refinement.get("improvement_tolerance"),
        field="local_refinement.improvement_tolerance",
        minimum=0.0,
    )
    if (anchors, sweeps, tolerance) != (64, 10, 1.0e-10):
        raise Step2CConfigError("local_refinement settings changed from the pack.")

    penalties, primary_penalty = _validate_penalties(raw)

    models = _mapping(raw.get("models"), field="models")
    expected_models = [
        {"name": "default_current", "type": "existing_default"},
        {
            "name": "conservative",
            "type": "explicit_conservative",
            "min_noise": 0.01,
            "min_lengthscale": 0.05,
        },
    ]
    if models.get("variants") != expected_models:
        raise Step2CConfigError(
            "models.variants must preserve default and conservative settings."
        )
    if (
        models.get("primary_for_debug") != "default_current"
        or models.get("exact_leave_one_out") is not True
        or models.get("report_training_posterior_only_as_diagnostic") is not True
    ):
        raise Step2CConfigError("models must preserve the strict validation policy.")

    influence = _mapping(
        raw.get("observation_influence"), field="observation_influence"
    )
    influence_ids = _int_tuple(
        influence.get("omitted_sample_ids"),
        field="observation_influence.omitted_sample_ids",
        minimum=1,
    )
    regional_thresholds = _number_tuple(
        influence.get("regional_thresholds"),
        field="observation_influence.regional_thresholds",
    )
    if (
        influence.get("enabled") is not True
        or influence_ids != base.expected_sample_ids
        or regional_thresholds != (0.10, 0.15, 0.20)
    ):
        raise Step2CConfigError(
            "observation_influence must cover every configured sample and three "
            "thresholds."
        )
    influence_pool_size = _integer(
        influence.get("common_pool_size"),
        field="observation_influence.common_pool_size",
        minimum=1,
    )
    influence_pool_seed = _integer(
        influence.get("common_pool_seed"),
        field="observation_influence.common_pool_seed",
    )
    influence_top_k = _integer(
        influence.get("top_k"), field="observation_influence.top_k", minimum=1
    )
    if (influence_pool_size, influence_pool_seed, influence_top_k) != (
        32768,
        73,
        100,
    ):
        raise Step2CConfigError(
            "The observation-influence common pool, seed, or top-K changed."
        )

    regions = _mapping(raw.get("robust_regions"), field="robust_regions")
    if regions.get("clustering") != "agglomerative_complete_link":
        raise Step2CConfigError("robust_regions.clustering must use complete link.")
    criteria = _mapping(
        regions.get("consensus_required_criteria"),
        field="robust_regions.consensus_required_criteria",
    )
    expected_boolean_criteria = {
        "require_grid_valid": True,
        "require_hard_distance_valid": True,
    }
    for field, expected in expected_boolean_criteria.items():
        if criteria.get(field) is not expected:
            raise Step2CConfigError(f"consensus criterion {field} must remain true.")
    region_threshold = _number(
        regions.get("primary_distance_threshold"),
        field="robust_regions.primary_distance_threshold",
        minimum=0.0,
    )
    region_sensitivity_thresholds = _number_tuple(
        regions.get("sensitivity_thresholds"),
        field="robust_regions.sensitivity_thresholds",
    )
    shortlist_min = _integer(
        regions.get("shortlist_min"), field="robust_regions.shortlist_min", minimum=1
    )
    shortlist_max = _integer(
        regions.get("shortlist_max"), field="robust_regions.shortlist_max", minimum=1
    )
    consensus_batch_size = _integer(
        regions.get("consensus_batch_size"),
        field="robust_regions.consensus_batch_size",
        minimum=1,
    )
    consensus_nested_match_min = _integer(
        criteria.get("largest_two_nested_regional_matches_within_0_15"),
        field="consensus.nested_match_min",
        minimum=1,
    )
    consensus_mean_distance_max = _number(
        criteria.get("largest_two_nested_mean_matched_distance_max"),
        field="consensus.mean_distance_max",
        minimum=0.0,
    )
    consensus_family_coverage_min = _integer(
        criteria.get("minimum_region_family_coverage"),
        field="consensus.family_coverage_min",
        minimum=1,
    )
    if (
        region_threshold,
        region_sensitivity_thresholds,
        shortlist_min,
        shortlist_max,
        consensus_batch_size,
        consensus_nested_match_min,
        consensus_mean_distance_max,
        consensus_family_coverage_min,
    ) != (0.15, (0.10, 0.20), 8, 12, 5, 4, 0.10, 3):
        raise Step2CConfigError(
            "Robust-region thresholds, shortlist limits, or consensus gates changed."
        )

    execution_modes = _validate_execution_modes(
        raw, expected_sample_ids=base.expected_sample_ids
    )

    reproducibility = _mapping(raw.get("reproducibility"), field="reproducibility")
    expected_reproducibility = {
        "model_seed": 73,
        "record_git_commit": True,
        "record_dirty_state": True,
        "record_environment_versions": True,
        "record_config_hash": True,
        "record_workbook_hash_and_mtime_before_after": True,
    }
    if reproducibility != expected_reproducibility:
        raise Step2CConfigError("reproducibility settings must remain fully enabled.")
    outputs = _mapping(raw.get("outputs"), field="outputs")
    if outputs != {
        "root": STEP2C_OUTPUT_ROOT,
        "tracked_private_recipes": False,
        "create_portable_ignored_zip": True,
    }:
        raise Step2CConfigError("outputs must remain ignored, private, and debug-only.")

    return ResolvedStep2CConfig(
        raw=raw,
        config_path=config_path,
        config_sha256=_file_sha256(config_path),
        resolved_config_hash=_canonical_hash(raw),
        base_config_path=base_path,
        base=base,
        workbook_source_kind=workbook_source_kind,
        workbook_relative_path=workbook_path,
        workbook_expected_sha256=workbook_hash,
        resolved_off_grid_exceptions=off_grid_exceptions,
        objective_bounds=objective_bounds,
        beta_values=beta_values,
        primary_beta=primary_beta,
        moment_method="analytic_identity",
        score_chunk_size=score_chunk_size,
        positive_hvi_threshold=positive_threshold,
        numeric_tolerance=numeric_tolerance,
        bound_policies=tuple(bound_policies_raw),
        primary_bound_policy=primary_bound_policy,
        mc_comparison_samples=mc_comparison_samples,
        mc_comparison_seed=mc_comparison_seed,
        primary_sobol_seed=primary_seed,
        secondary_sobol_seeds=secondary_seeds,
        nested_pool_sizes=nested,
        refinement_anchors=anchors,
        refinement_max_sweeps=sweeps,
        refinement_tolerance=tolerance,
        penalty_variants=penalties,
        primary_penalty_variant=primary_penalty,
        model_variant_names=("default_current", "conservative"),
        primary_model_variant="default_current",
        influence_pool_size=influence_pool_size,
        influence_pool_seed=influence_pool_seed,
        influence_sample_ids=influence_ids,
        influence_top_k=influence_top_k,
        regional_thresholds=regional_thresholds,
        region_threshold=region_threshold,
        region_sensitivity_thresholds=region_sensitivity_thresholds,
        shortlist_min=shortlist_min,
        shortlist_max=shortlist_max,
        consensus_batch_size=consensus_batch_size,
        consensus_nested_match_min=consensus_nested_match_min,
        consensus_mean_distance_max=consensus_mean_distance_max,
        consensus_family_coverage_min=consensus_family_coverage_min,
        execution_modes=execution_modes,
        model_seed=_integer(
            reproducibility.get("model_seed"), field="reproducibility.model_seed"
        ),
        output_root=STEP2C_OUTPUT_ROOT,
        create_portable_zip=True,
    )


__all__ = [
    "ExecutionModeSettings",
    "PenaltyVariant",
    "ResolvedStep2CConfig",
    "STEP2C_BOUND_POLICIES",
    "STEP2C_NESTED_POOL_SIZES",
    "STEP2C_OUTPUT_ROOT",
    "STEP2C_PRIVATE_SOURCE_KIND",
    "STEP2C_RUNTIME_GENERATED",
    "STEP2C_SCHEMA_VERSION",
    "STEP2C_SYNTHETIC_SOURCE_KIND",
    "Step2CConfigError",
    "load_step2c_config",
]
