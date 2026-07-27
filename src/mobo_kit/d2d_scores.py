"""Campaign-specific D2D score calculations and read-only validation.

The final scores supplied in workbook columns Z, AA, and AB are authoritative.
The helpers in this module calculate independent support values for validation;
they never replace or mutate those supplied objective values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real
from typing import Iterable, Self, Sequence

import numpy as np
import pandas as pd


SAMPLE_ID_COLUMN = "Sample number"
UNIFORMITY_COMPONENT_COLUMNS = ("Coverage", "1 - Uniformity", "Phase purity")
OPTOELECTRONIC_COMPONENT_COLUMNS = (
    "PL - Implied Voc (Max)",
    "Photoconductance (Max)",
)
OPTOELECTRONIC_CHECK_COLUMN = "Log10 (Photoconductance (Max) x PL - Implied Voc (Max))"
THICKNESS_MEASUREMENT_COLUMNS = ("T1", "T2", "T3", "T4")
THICKNESS_CHECK_COLUMN = "Normalized thickness (sigma = 250)"
OBJECTIVE_COLUMNS = (
    "Uniformity score",
    "Optoelectronic score",
    "Thickness score",
)

_REQUIRED_VALIDATION_COLUMNS = (
    *UNIFORMITY_COMPONENT_COLUMNS,
    *OPTOELECTRONIC_COMPONENT_COLUMNS,
    *THICKNESS_MEASUREMENT_COLUMNS,
    *OBJECTIVE_COLUMNS,
)


class ValidationSeverity(str, Enum):
    """Severity attached to a score-validation finding."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class D2DScoreTolerances:
    """Absolute comparison tolerances for two-decimal workbook score fields.

    A two-decimal rounded value can differ from its unrounded calculation by
    exactly 0.005. ``rounding_slack`` absorbs only floating-point noise at that
    boundary; it does not materially relax the documented rounding tolerance.
    """

    uniformity_atol: float = 0.005
    optoelectronic_atol: float = 0.005
    thickness_atol: float = 0.005
    rounding_slack: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "uniformity_atol",
            "optoelectronic_atol",
            "thickness_atol",
            "rounding_slack",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real non-boolean number.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class D2DScoreFinding:
    """One warning or error tied to a source DataFrame row."""

    severity: ValidationSeverity
    code: str
    row_position: int
    row_label: object
    sample_id: object | None
    column: str | None
    message: str


@dataclass(frozen=True)
class D2DScoreRowValidation:
    """Calculated and supplied score comparisons for one campaign row."""

    row_position: int
    row_label: object
    sample_id: object | None
    uniformity_score_authoritative: float | None
    uniformity_score_calculated: float | None
    uniformity_absolute_difference: float | None
    uniformity_matches: bool | None
    optoelectronic_check_supplied: float | None
    optoelectronic_score_authoritative: float | None
    optoelectronic_score_calculated: float | None
    optoelectronic_check_absolute_difference: float | None
    optoelectronic_check_matches: bool | None
    optoelectronic_objective_absolute_difference: float | None
    optoelectronic_objective_matches: bool | None
    thickness_average_calculated: float | None
    thickness_check_supplied: float | None
    thickness_score_authoritative: float | None
    thickness_score_calculated: float | None
    thickness_check_absolute_difference: float | None
    thickness_check_matches: bool | None
    thickness_objective_absolute_difference: float | None
    thickness_objective_matches: bool | None
    warning_codes: tuple[str, ...]
    error_codes: tuple[str, ...]


class D2DScoreValidationError(ValueError):
    """Raised when a structured score-validation result contains errors."""

    def __init__(self, findings: Sequence[D2DScoreFinding]) -> None:
        self.findings = tuple(findings)
        details = "\n- ".join(finding.message for finding in self.findings)
        super().__init__(f"D2D score validation failed:\n- {details}")


@dataclass(frozen=True)
class D2DScoreValidationResult:
    """Structured, exportable result of validating supplied D2D final scores."""

    rows: tuple[D2DScoreRowValidation, ...]
    findings: tuple[D2DScoreFinding, ...]
    tolerances: D2DScoreTolerances

    @property
    def warnings(self) -> tuple[D2DScoreFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.severity is ValidationSeverity.WARNING
        )

    @property
    def errors(self) -> tuple[D2DScoreFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.severity is ValidationSeverity.ERROR
        )

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def known_uniformity_score_mismatch(self) -> bool:
        return any(finding.code == "uniformity_mismatch" for finding in self.findings)

    @property
    def uniformity_warning_count(self) -> int:
        """Return the number of rows with a calculated-vs-supplied mismatch."""
        return sum(finding.code == "uniformity_mismatch" for finding in self.findings)

    @property
    def frame(self) -> pd.DataFrame:
        """Return a new one-row-per-input-row validation DataFrame."""
        return self.to_frame()

    def to_frame(self) -> pd.DataFrame:
        """Export per-row comparisons without sharing mutable source state."""
        return pd.DataFrame(asdict(row) for row in self.rows)

    def findings_frame(self) -> pd.DataFrame:
        """Export warnings and errors as a separate tidy DataFrame."""
        records = []
        for finding in self.findings:
            record = asdict(finding)
            record["severity"] = finding.severity.value
            records.append(record)
        return pd.DataFrame(records)

    def raise_for_errors(self) -> Self:
        """Raise :class:`D2DScoreValidationError` or return this result."""
        if self.errors:
            raise D2DScoreValidationError(self.errors)
        return self


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.replace("\u00a0", " ").strip()
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _optional_finite_real(value: object, *, name: str) -> float | None:
    if _is_missing(value):
        return None
    return _finite_real(value, name=name)


def compute_uniformity_score(
    coverage: Real,
    one_minus_uniformity: Real,
    phase_purity: Real,
) -> float:
    """Calculate the D2D uniformity support score ``L * N * O``."""
    components = (
        _finite_real(coverage, name="coverage"),
        _finite_real(one_minus_uniformity, name="one_minus_uniformity"),
        _finite_real(phase_purity, name="phase_purity"),
    )
    score = components[0] * components[1] * components[2]
    if not math.isfinite(score):
        raise ValueError("The calculated uniformity score is non-finite.")
    return score


def compute_optoelectronic_score(
    implied_voc_max: Real,
    photoconductance_max: Real,
) -> float:
    """Calculate ``log10(P * Q)`` for finite, strictly positive P and Q.

    Summing the two base-10 logarithms is algebraically identical to logging
    the product and avoids intermediate floating-point overflow or underflow.
    """
    implied_voc = _finite_real(implied_voc_max, name="implied_voc_max")
    photoconductance = _finite_real(photoconductance_max, name="photoconductance_max")
    if implied_voc <= 0:
        raise ValueError("implied_voc_max must be strictly positive.")
    if photoconductance <= 0:
        raise ValueError("photoconductance_max must be strictly positive.")
    return math.log10(implied_voc) + math.log10(photoconductance)


def compute_thickness_average(
    t1: object,
    t2: object,
    t3: object,
    t4: object,
) -> float:
    """Average numeric T1:T4 values, ignoring normalized blank cells.

    Only the four named campaign measurements are accepted by the API, so the
    adjacent ``T anom`` workbook field cannot enter this calculation.
    """
    measurements: list[float] = []
    for position, value in enumerate((t1, t2, t3, t4), start=1):
        if _is_missing(value):
            continue
        measurements.append(_finite_real(value, name=f"T{position}"))
    if not measurements:
        raise ValueError("At least one valid T1:T4 thickness value is required.")
    try:
        average = math.fsum(measurements) / len(measurements)
    except OverflowError as exc:
        raise ValueError("The calculated thickness average is non-finite.") from exc
    if not math.isfinite(average):
        raise ValueError("The calculated thickness average is non-finite.")
    return average


def compute_thickness_score(
    thickness_values: Iterable[object],
    target: Real = 650.0,
    scale: Real = 250.0,
) -> float:
    """Calculate the D2D no-half-factor target score from exactly T1:T4."""
    if isinstance(thickness_values, (str, bytes)):
        raise TypeError("thickness_values must contain exactly T1:T4 values.")
    values = tuple(thickness_values)
    if len(values) != 4:
        raise ValueError("thickness_values must contain exactly T1:T4 values.")
    center = _finite_real(target, name="target")
    width = _finite_real(scale, name="scale")
    if width <= 0:
        raise ValueError("scale must be strictly positive.")
    average = compute_thickness_average(*values)
    standardized = (average - center) / width
    return math.exp(-(standardized * standardized))


def _comparison(
    calculated: float,
    supplied: float,
    *,
    absolute_tolerance: float,
    rounding_slack: float,
) -> tuple[float, bool]:
    difference = abs(calculated - supplied)
    return difference, difference <= absolute_tolerance + rounding_slack


def _validate_required_columns(frame: pd.DataFrame) -> None:
    duplicate_required = sorted(
        {
            str(column)
            for column in frame.columns[frame.columns.duplicated(keep=False)]
            if column in _REQUIRED_VALIDATION_COLUMNS
            or column in (OPTOELECTRONIC_CHECK_COLUMN, THICKNESS_CHECK_COLUMN)
        }
    )
    if duplicate_required:
        raise ValueError(
            "Ambiguous duplicate D2D score-validation column(s): "
            + ", ".join(repr(column) for column in duplicate_required)
            + "."
        )
    missing = [column for column in _REQUIRED_VALIDATION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            "Missing required D2D score-validation column(s): "
            + ", ".join(repr(column) for column in missing)
            + "."
        )


def validate_supplied_d2d_scores(
    frame: pd.DataFrame,
    tolerances: D2DScoreTolerances | None = None,
) -> D2DScoreValidationResult:
    """Validate support equations while preserving Z/AA/AB as authoritative.

    Uniformity discrepancies are warnings. Unexpected optoelectronic or
    thickness discrepancies, invalid support inputs, and missing/non-finite
    final scores are errors. The authoritative Uniformity score must also stay
    in its resolved [0, 1] range. Optional calculated check columns R and Y are
    compared whenever their row value is present.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if tolerances is None:
        tolerances = D2DScoreTolerances()
    elif not isinstance(tolerances, D2DScoreTolerances):
        raise TypeError("tolerances must be a D2DScoreTolerances instance.")
    _validate_required_columns(frame)

    findings: list[D2DScoreFinding] = []
    rows: list[D2DScoreRowValidation] = []

    for row_position in range(len(frame)):
        source = frame.iloc[row_position]
        row_label = frame.index[row_position]
        sample_id = source.get(SAMPLE_ID_COLUMN)
        if _is_missing(sample_id):
            sample_id = None
        row_findings: list[D2DScoreFinding] = []

        def add_finding(
            severity: ValidationSeverity,
            code: str,
            message: str,
            *,
            column: str | None = None,
        ) -> None:
            finding = D2DScoreFinding(
                severity=severity,
                code=code,
                row_position=row_position,
                row_label=row_label,
                sample_id=sample_id,
                column=column,
                message=f"Row {row_label!r}: {message}",
            )
            findings.append(finding)
            row_findings.append(finding)

        authoritative: dict[str, float | None] = {}
        for column in OBJECTIVE_COLUMNS:
            try:
                value = _optional_finite_real(source[column], name=column)
            except (TypeError, ValueError) as exc:
                value = None
                add_finding(
                    ValidationSeverity.ERROR,
                    "authoritative_score_invalid",
                    f"authoritative {column!r} is invalid: {exc}",
                    column=column,
                )
            else:
                if value is None:
                    add_finding(
                        ValidationSeverity.ERROR,
                        "authoritative_score_missing",
                        f"authoritative {column!r} is missing.",
                        column=column,
                    )
            authoritative[column] = value

        uniformity_authoritative = authoritative[OBJECTIVE_COLUMNS[0]]
        if uniformity_authoritative is not None and not (
            0.0 <= uniformity_authoritative <= 1.0
        ):
            add_finding(
                ValidationSeverity.ERROR,
                "uniformity_score_out_of_range",
                "authoritative 'Uniformity score' must remain in [0, 1]; "
                f"found {uniformity_authoritative}.",
                column=OBJECTIVE_COLUMNS[0],
            )

        uniformity_calculated: float | None = None
        uniformity_difference: float | None = None
        uniformity_matches: bool | None = None
        try:
            uniformity_calculated = compute_uniformity_score(
                source[UNIFORMITY_COMPONENT_COLUMNS[0]],
                source[UNIFORMITY_COMPONENT_COLUMNS[1]],
                source[UNIFORMITY_COMPONENT_COLUMNS[2]],
            )
        except (TypeError, ValueError) as exc:
            add_finding(
                ValidationSeverity.WARNING,
                "uniformity_support_invalid",
                f"uniformity support value could not be calculated: {exc}",
            )
        if uniformity_calculated is not None and uniformity_authoritative is not None:
            uniformity_difference, uniformity_matches = _comparison(
                uniformity_calculated,
                uniformity_authoritative,
                absolute_tolerance=tolerances.uniformity_atol,
                rounding_slack=tolerances.rounding_slack,
            )
            if not uniformity_matches:
                add_finding(
                    ValidationSeverity.WARNING,
                    "uniformity_mismatch",
                    "calculated L*N*O differs from authoritative 'Uniformity "
                    f"score' by {uniformity_difference:.12g}; the supplied score "
                    "remains unchanged.",
                    column=OBJECTIVE_COLUMNS[0],
                )

        optoelectronic_calculated: float | None = None
        optoelectronic_check: float | None = None
        optoelectronic_check_difference: float | None = None
        optoelectronic_check_matches: bool | None = None
        optoelectronic_objective_difference: float | None = None
        optoelectronic_objective_matches: bool | None = None
        try:
            optoelectronic_calculated = compute_optoelectronic_score(
                source[OPTOELECTRONIC_COMPONENT_COLUMNS[0]],
                source[OPTOELECTRONIC_COMPONENT_COLUMNS[1]],
            )
        except (TypeError, ValueError) as exc:
            add_finding(
                ValidationSeverity.ERROR,
                "optoelectronic_support_invalid",
                f"optoelectronic support value could not be calculated: {exc}",
            )
        if OPTOELECTRONIC_CHECK_COLUMN in frame:
            try:
                optoelectronic_check = _optional_finite_real(
                    source[OPTOELECTRONIC_CHECK_COLUMN],
                    name=OPTOELECTRONIC_CHECK_COLUMN,
                )
            except (TypeError, ValueError) as exc:
                add_finding(
                    ValidationSeverity.ERROR,
                    "optoelectronic_check_invalid",
                    f"calculated-check column R is invalid: {exc}",
                    column=OPTOELECTRONIC_CHECK_COLUMN,
                )
        if optoelectronic_calculated is not None and optoelectronic_check is not None:
            (
                optoelectronic_check_difference,
                optoelectronic_check_matches,
            ) = _comparison(
                optoelectronic_calculated,
                optoelectronic_check,
                absolute_tolerance=tolerances.optoelectronic_atol,
                rounding_slack=tolerances.rounding_slack,
            )
            if not optoelectronic_check_matches:
                add_finding(
                    ValidationSeverity.ERROR,
                    "optoelectronic_check_mismatch",
                    "calculated log10(P*Q) differs from check column R by "
                    f"{optoelectronic_check_difference:.12g}.",
                    column=OPTOELECTRONIC_CHECK_COLUMN,
                )
        optoelectronic_authoritative = authoritative[OBJECTIVE_COLUMNS[1]]
        if (
            optoelectronic_calculated is not None
            and optoelectronic_authoritative is not None
        ):
            (
                optoelectronic_objective_difference,
                optoelectronic_objective_matches,
            ) = _comparison(
                optoelectronic_calculated,
                optoelectronic_authoritative,
                absolute_tolerance=tolerances.optoelectronic_atol,
                rounding_slack=tolerances.rounding_slack,
            )
            if not optoelectronic_objective_matches:
                add_finding(
                    ValidationSeverity.ERROR,
                    "optoelectronic_objective_mismatch",
                    "calculated log10(P*Q) differs from authoritative "
                    f"'Optoelectronic score' by "
                    f"{optoelectronic_objective_difference:.12g}.",
                    column=OBJECTIVE_COLUMNS[1],
                )

        thickness_average: float | None = None
        thickness_calculated: float | None = None
        thickness_check: float | None = None
        thickness_check_difference: float | None = None
        thickness_check_matches: bool | None = None
        thickness_objective_difference: float | None = None
        thickness_objective_matches: bool | None = None
        thickness_values = tuple(
            source[column] for column in THICKNESS_MEASUREMENT_COLUMNS
        )
        try:
            thickness_average = compute_thickness_average(*thickness_values)
            thickness_calculated = compute_thickness_score(thickness_values)
        except (TypeError, ValueError) as exc:
            add_finding(
                ValidationSeverity.ERROR,
                "thickness_support_invalid",
                f"thickness support value could not be calculated: {exc}",
            )
        if THICKNESS_CHECK_COLUMN in frame:
            try:
                thickness_check = _optional_finite_real(
                    source[THICKNESS_CHECK_COLUMN], name=THICKNESS_CHECK_COLUMN
                )
            except (TypeError, ValueError) as exc:
                add_finding(
                    ValidationSeverity.ERROR,
                    "thickness_check_invalid",
                    f"calculated-check column Y is invalid: {exc}",
                    column=THICKNESS_CHECK_COLUMN,
                )
        if thickness_calculated is not None and thickness_check is not None:
            thickness_check_difference, thickness_check_matches = _comparison(
                thickness_calculated,
                thickness_check,
                absolute_tolerance=tolerances.thickness_atol,
                rounding_slack=tolerances.rounding_slack,
            )
            if not thickness_check_matches:
                add_finding(
                    ValidationSeverity.ERROR,
                    "thickness_check_mismatch",
                    "calculated no-half-factor thickness score differs from check "
                    f"column Y by {thickness_check_difference:.12g}.",
                    column=THICKNESS_CHECK_COLUMN,
                )
        thickness_authoritative = authoritative[OBJECTIVE_COLUMNS[2]]
        if thickness_calculated is not None and thickness_authoritative is not None:
            (
                thickness_objective_difference,
                thickness_objective_matches,
            ) = _comparison(
                thickness_calculated,
                thickness_authoritative,
                absolute_tolerance=tolerances.thickness_atol,
                rounding_slack=tolerances.rounding_slack,
            )
            if not thickness_objective_matches:
                add_finding(
                    ValidationSeverity.ERROR,
                    "thickness_objective_mismatch",
                    "calculated no-half-factor thickness score differs from "
                    f"authoritative 'Thickness score' by "
                    f"{thickness_objective_difference:.12g}.",
                    column=OBJECTIVE_COLUMNS[2],
                )

        rows.append(
            D2DScoreRowValidation(
                row_position=row_position,
                row_label=row_label,
                sample_id=sample_id,
                uniformity_score_authoritative=uniformity_authoritative,
                uniformity_score_calculated=uniformity_calculated,
                uniformity_absolute_difference=uniformity_difference,
                uniformity_matches=uniformity_matches,
                optoelectronic_check_supplied=optoelectronic_check,
                optoelectronic_score_authoritative=optoelectronic_authoritative,
                optoelectronic_score_calculated=optoelectronic_calculated,
                optoelectronic_check_absolute_difference=(
                    optoelectronic_check_difference
                ),
                optoelectronic_check_matches=optoelectronic_check_matches,
                optoelectronic_objective_absolute_difference=(
                    optoelectronic_objective_difference
                ),
                optoelectronic_objective_matches=optoelectronic_objective_matches,
                thickness_average_calculated=thickness_average,
                thickness_check_supplied=thickness_check,
                thickness_score_authoritative=thickness_authoritative,
                thickness_score_calculated=thickness_calculated,
                thickness_check_absolute_difference=thickness_check_difference,
                thickness_check_matches=thickness_check_matches,
                thickness_objective_absolute_difference=(
                    thickness_objective_difference
                ),
                thickness_objective_matches=thickness_objective_matches,
                warning_codes=tuple(
                    finding.code
                    for finding in row_findings
                    if finding.severity is ValidationSeverity.WARNING
                ),
                error_codes=tuple(
                    finding.code
                    for finding in row_findings
                    if finding.severity is ValidationSeverity.ERROR
                ),
            )
        )

    return D2DScoreValidationResult(
        rows=tuple(rows), findings=tuple(findings), tolerances=tolerances
    )


def raise_for_errors(result: D2DScoreValidationResult) -> D2DScoreValidationResult:
    """Raise for a failed result; convenient for campaign-adapter pipelines."""
    if not isinstance(result, D2DScoreValidationResult):
        raise TypeError("result must be a D2DScoreValidationResult.")
    return result.raise_for_errors()


__all__ = [
    "D2DScoreFinding",
    "D2DScoreRowValidation",
    "D2DScoreTolerances",
    "D2DScoreValidationError",
    "D2DScoreValidationResult",
    "OBJECTIVE_COLUMNS",
    "OPTOELECTRONIC_CHECK_COLUMN",
    "OPTOELECTRONIC_COMPONENT_COLUMNS",
    "SAMPLE_ID_COLUMN",
    "THICKNESS_CHECK_COLUMN",
    "THICKNESS_MEASUREMENT_COLUMNS",
    "UNIFORMITY_COMPONENT_COLUMNS",
    "ValidationSeverity",
    "compute_optoelectronic_score",
    "compute_thickness_average",
    "compute_thickness_score",
    "compute_uniformity_score",
    "raise_for_errors",
    "validate_supplied_d2d_scores",
]
