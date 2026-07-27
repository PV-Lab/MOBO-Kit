"""Fail-closed approval validation for campaign-facing candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


_PLACEHOLDER_MARKERS = ("pending", "provisional", "tbd", "todo", "replace_me")
_ALLOWED_NULL_PATHS = {"local_penalization.dimension_weights"}


class ProductionApprovalError(ValueError):
    """Raised with every discovered reason a campaign is not production-ready."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            "Production candidate generation is blocked:\n- " + "\n- ".join(self.errors)
        )


class CampaignProposalDisabledError(RuntimeError):
    """Raised when approved configuration reaches an unwired legacy proposal path."""


@dataclass(frozen=True)
class ProductionApprovalReceipt:
    """Validated approval provenance to record with a future campaign run."""

    schema_version: str
    campaign_id: str
    workbook_profile: str
    input_contract_version: str
    objective_count: int
    approved_by: str
    approved_at: str
    decision_record: str
    resolved_config_sha256: str


def _value_at(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


class _Missing:
    pass


_MISSING = _Missing()


def _is_placeholder(value: Any) -> bool:
    if value is None or value is _MISSING:
        return True
    if isinstance(value, str):
        normalized = value.strip().casefold()
        return not normalized or any(
            marker in normalized for marker in _PLACEHOLDER_MARKERS
        )
    return False


def _contains_unresolved(value: Any) -> bool:
    if _is_placeholder(value):
        return True
    if isinstance(value, Mapping):
        return not value or any(_contains_unresolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return not value or any(_contains_unresolved(item) for item in value)
    return False


def _find_unresolved(
    value: Any, *, path: str = "", allow_empty_constraints: bool = True
) -> list[str]:
    findings: list[str] = []
    if value is None:
        if path not in _ALLOWED_NULL_PATHS:
            findings.append(path or "<root>")
        return findings
    if isinstance(value, str):
        if _is_placeholder(value):
            findings.append(path or "<root>")
        return findings
    if isinstance(value, Mapping):
        if not value:
            findings.append(path or "<root>")
            return findings
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            findings.extend(
                _find_unresolved(
                    item,
                    path=child,
                    allow_empty_constraints=allow_empty_constraints,
                )
            )
        return findings
    if isinstance(value, (list, tuple)):
        if not value:
            if not (allow_empty_constraints and path == "constraints"):
                findings.append(path or "<root>")
            return findings
        for index, item in enumerate(value):
            findings.extend(
                _find_unresolved(
                    item,
                    path=f"{path}[{index}]",
                    allow_empty_constraints=allow_empty_constraints,
                )
            )
    return findings


def _require_resolved(config: Mapping[str, Any], path: str, errors: list[str]) -> Any:
    value = _value_at(config, path)
    if value is _MISSING:
        errors.append(f"Missing required field {path!r}.")
        return None
    if _contains_unresolved(value):
        errors.append(f"Field {path!r} contains an unresolved placeholder.")
        return None
    return value


def _require_string(
    config: Mapping[str, Any], path: str, errors: list[str]
) -> str | None:
    value = _require_resolved(config, path, errors)
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"Field {path!r} must be a resolved string.")
        return None
    return value.strip()


def _finite_number(
    value: Any, *, path: str, errors: list[str], positive: bool
) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        errors.append(f"Field {path!r} must be a real non-boolean number.")
        return None
    number = float(value)
    if not np.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "strictly positive" if positive else "non-negative"
        errors.append(f"Field {path!r} must be finite and {qualifier}.")
        return None
    return number


def _finite_real(value: Any, *, path: str, errors: list[str]) -> float | None:
    """Validate a finite real without imposing a sign restriction."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        errors.append(f"Field {path!r} must be a finite real non-boolean number.")
        return None
    number = float(value)
    if not np.isfinite(number):
        errors.append(f"Field {path!r} must be a finite real number.")
        return None
    return number


def _positive_integer(value: Any, *, path: str, errors: list[str]) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        errors.append(f"Field {path!r} must be a positive integer.")
        return None
    number = int(value)
    if number <= 0:
        errors.append(f"Field {path!r} must be a positive integer.")
        return None
    return number


def _canonical_hash(config: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionApprovalError(
            [f"Resolved configuration is not canonically JSON-serializable: {exc}."]
        ) from exc
    return sha256(payload).hexdigest()


def validate_production_config(
    config: Mapping[str, Any],
) -> ProductionApprovalReceipt:
    """Validate a fully resolved campaign contract or block candidate generation.

    This gate validates authorization and provenance, not scientific correctness.
    Scientific approvers remain responsible for the supplied formulas, scales,
    reference point, QC rules, and constraints.
    """
    if not isinstance(config, Mapping):
        raise ProductionApprovalError(["Configuration must be a mapping."])
    errors: list[str] = []
    unresolved_paths = _find_unresolved(config)
    if unresolved_paths:
        errors.append(
            "Resolved configuration still contains null/blank/placeholder values "
            f"at: {', '.join(unresolved_paths)}."
        )
    schema_version = _require_string(config, "schema_version", errors)
    campaign_id = _require_string(config, "campaign_id", errors)
    approved = _value_at(config, "approved_for_production")
    if approved is not True:
        errors.append("'approved_for_production' must be explicitly true.")
    approved_by = _require_string(config, "approval.approved_by", errors)
    approved_at = _require_string(config, "approval.approved_at", errors)
    decision_record = _require_string(config, "approval.decision_record", errors)
    input_contract_version = _require_string(config, "input_contract_version", errors)
    if approved_at is not None:
        try:
            timestamp = datetime.fromisoformat(approved_at)
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
        except ValueError:
            errors.append(
                "approval.approved_at must be an ISO-8601 timestamp with timezone."
            )

    if _value_at(config, "objective_mapping_status") != "approved":
        errors.append("objective_mapping_status must be explicitly 'approved'.")
    if _value_at(config, "constraints_status") != "approved":
        errors.append("constraints_status must be explicitly 'approved'.")

    workbook_profile = _value_at(config, "workbook_profile")
    if workbook_profile is _MISSING:
        workbook_profile = _value_at(config, "workbook.profile")
    if _contains_unresolved(workbook_profile):
        errors.append("A resolved workbook_profile (or workbook.profile) is required.")
    elif not isinstance(workbook_profile, str):
        errors.append("workbook_profile must be a resolved string.")
    elif workbook_profile != "d2d_summary_v2":
        errors.append("D2D production requires workbook profile 'd2d_summary_v2'.")

    objectives = _value_at(config, "objectives")
    if not isinstance(objectives, list) or len(objectives) != 3:
        errors.append("'objectives' must be a list containing exactly three entries.")
        objectives = []
    objective_names: set[str] = set()
    objective_fields = (
        "name",
        "model_source_column",
        "utility_transform",
        "transform_version",
        "formula_version",
        "direction",
        "scaling",
    )
    for index, objective in enumerate(objectives):
        prefix = f"objectives[{index}]"
        if not isinstance(objective, Mapping):
            errors.append(f"{prefix} must be a mapping.")
            continue
        for field in objective_fields:
            value = objective.get(field, _MISSING)
            if _contains_unresolved(value):
                errors.append(f"{prefix}.{field} must be present and resolved.")
        for field in objective_fields[:-1]:
            value = objective.get(field)
            if value is not None and not isinstance(value, str):
                errors.append(f"{prefix}.{field} must be a string.")
        name = objective.get("name")
        if isinstance(name, str) and name.strip():
            if name in objective_names:
                errors.append(f"Objective name {name!r} is duplicated.")
            objective_names.add(name)
        if objective.get("direction") not in {"maximize", "minimize", "target"}:
            errors.append(f"{prefix}.direction must be maximize, minimize, or target.")
        scaling = objective.get("scaling")
        if not isinstance(scaling, Mapping):
            errors.append(f"{prefix}.scaling must be a resolved mapping.")
        else:
            mode = scaling.get("mode")
            version = scaling.get("version")
            if not isinstance(version, str) or _is_placeholder(version):
                errors.append(f"{prefix}.scaling.version must be resolved.")
            if mode == "already_normalized":
                allowed_keys = {"mode", "version"}
                unexpected = sorted(
                    (key for key in scaling if key not in allowed_keys), key=str
                )
                if unexpected:
                    errors.append(
                        f"{prefix}.scaling already_normalized has unsupported "
                        f"field(s): {unexpected}."
                    )
            elif mode == "fixed_affine":
                allowed_keys = {
                    "mode",
                    "version",
                    "lower_anchor",
                    "upper_anchor",
                }
                unexpected = sorted(
                    (key for key in scaling if key not in allowed_keys), key=str
                )
                if unexpected:
                    errors.append(
                        f"{prefix}.scaling fixed_affine has unsupported field(s): "
                        f"{unexpected}."
                    )
                lower = scaling.get("lower_anchor")
                upper = scaling.get("upper_anchor")
                lower_value = _finite_real(
                    lower,
                    path=f"{prefix}.scaling.lower_anchor",
                    errors=errors,
                )
                upper_value = _finite_real(
                    upper,
                    path=f"{prefix}.scaling.upper_anchor",
                    errors=errors,
                )
                if (
                    lower_value is not None
                    and upper_value is not None
                    and lower_value >= upper_value
                ):
                    errors.append(
                        f"{prefix}.scaling requires lower_anchor < upper_anchor."
                    )
            else:
                errors.append(
                    f"{prefix}.scaling.mode must be 'already_normalized' or "
                    "'fixed_affine'; observed/data-derived scaling is forbidden."
                )
    reference = _value_at(config, "reference_point_utility")
    if not isinstance(reference, list) or len(reference) != len(objectives):
        errors.append(
            "'reference_point_utility' must be a fixed list matching the objective count."
        )
    elif any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        for value in reference
    ):
        errors.append("'reference_point_utility' must contain only finite numbers.")

    for field in (
        "complete_case_rule",
        "failed_measurement_rule",
        "outlier_rule",
        "control_rule",
        "replicate_rule",
    ):
        _require_string(config, f"qc_policy.{field}", errors)

    constraints = _value_at(config, "constraints")
    if not isinstance(constraints, list):
        errors.append("'constraints' must be an explicit list; [] is acceptable.")
    elif any(
        not isinstance(item, Mapping) or _contains_unresolved(item)
        for item in constraints
    ):
        errors.append("Every configured constraint must be a resolved mapping.")

    if _value_at(config, "r1.method") != "ucb_hvi":
        errors.append("r1.method must be 'ucb_hvi'.")
    r1_batch_size = _value_at(config, "r1.batch_size")
    if (
        isinstance(r1_batch_size, (bool, np.bool_))
        or not isinstance(r1_batch_size, (int, np.integer))
        or int(r1_batch_size) != 5
    ):
        errors.append("r1.batch_size must be explicitly set to 5.")
    beta = _require_resolved(config, "r1.beta", errors)
    if beta is not None:
        _finite_number(beta, path="r1.beta", errors=errors, positive=False)
    for field in ("posterior_samples", "candidate_pool_size"):
        value = _require_resolved(config, f"r1.{field}", errors)
        if value is not None:
            _positive_integer(value, path=f"r1.{field}", errors=errors)

    if _value_at(config, "r2.method") != "qlognehvi":
        errors.append("r2.method must be 'qlognehvi'.")
    r2_batch_size = _value_at(config, "r2.batch_size")
    if (
        isinstance(r2_batch_size, (bool, np.bool_))
        or not isinstance(r2_batch_size, (int, np.integer))
        or int(r2_batch_size) != 3
    ):
        errors.append("r2.batch_size must be explicitly set to 3.")
    if _value_at(config, "r2.sequential_pending") is not True:
        errors.append("r2.sequential_pending must be explicitly true.")
    for field in ("mc_samples", "candidate_pool_size"):
        value = _require_resolved(config, f"r2.{field}", errors)
        if value is not None:
            _positive_integer(value, path=f"r2.{field}", errors=errors)

    metric = _require_resolved(config, "local_penalization.distance_metric", errors)
    if metric is not None and metric not in {
        "normalized_euclidean",
        "weighted_normalized_euclidean",
    }:
        errors.append("local_penalization.distance_metric is unsupported.")
    dimension_weights = _value_at(config, "local_penalization.dimension_weights")
    if metric == "weighted_normalized_euclidean":
        configured_inputs = config.get("inputs")
        expected_dimension = (
            len(configured_inputs) if isinstance(configured_inputs, list) else 0
        )
        if expected_dimension <= 0:
            errors.append(
                "Weighted distance requires a non-empty inputs list to validate weights."
            )
        if (
            not isinstance(dimension_weights, list)
            or len(dimension_weights) != expected_dimension
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or float(value) <= 0
                for value in dimension_weights
            )
        ):
            errors.append(
                "Weighted distance requires one finite positive dimension weight "
                "per configured input."
            )
    elif (
        metric == "normalized_euclidean"
        and dimension_weights is not _MISSING
        and dimension_weights is not None
    ):
        errors.append(
            "Ordinary normalized_euclidean distance requires dimension_weights: null."
        )
    for field, positive in (
        ("radius", True),
        ("min_batch_distance", False),
        ("min_observed_distance", False),
    ):
        value = _require_resolved(config, f"local_penalization.{field}", errors)
        if value is not None:
            _finite_number(
                value,
                path=f"local_penalization.{field}",
                errors=errors,
                positive=positive,
            )
    if (
        _value_at(config, "local_penalization.allow_hard_distance_relaxation")
        is not False
    ):
        errors.append(
            "local_penalization.allow_hard_distance_relaxation must be false."
        )

    seed = _require_resolved(config, "reproducibility.seed", errors)
    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        errors.append("reproducibility.seed must be a non-negative integer.")
    for field in (
        "record_git_commit",
        "record_environment_versions",
        "record_resolved_config_hash",
    ):
        if _value_at(config, f"reproducibility.{field}") is not True:
            errors.append(f"reproducibility.{field} must be explicitly true.")

    if errors:
        raise ProductionApprovalError(errors)
    return ProductionApprovalReceipt(
        schema_version=str(schema_version),
        campaign_id=str(campaign_id),
        workbook_profile=str(workbook_profile),
        input_contract_version=str(input_contract_version),
        objective_count=len(objectives),
        approved_by=str(approved_by),
        approved_at=str(approved_at),
        decision_record=str(decision_record),
        resolved_config_sha256=_canonical_hash(config),
    )


assert_production_approved = validate_production_config


def block_legacy_campaign_proposal(config: Mapping[str, Any]) -> None:
    """Validate approval, then block the legacy runner until the Step 2B adapter.

    Validation is intentionally performed first so an unresolved campaign
    receives the complete approval errors. Even an approved configuration must
    not enter the old raw-objective qNEHVI path, which does not implement the
    Step 2A objective contract, discrete pool, or shared batch selector.
    """
    validate_production_config(config)
    raise CampaignProposalDisabledError(
        "Campaign candidate proposal is disabled in Step 2A: the legacy qNEHVI "
        "runner is not wired to the versioned objective contract, discrete "
        "candidate pool, and shared local-penalized selector. Implement and "
        "review the Step 2B campaign adapter before generating a real batch."
    )


__all__ = [
    "CampaignProposalDisabledError",
    "ProductionApprovalError",
    "ProductionApprovalReceipt",
    "assert_production_approved",
    "block_legacy_campaign_proposal",
    "validate_production_config",
]
