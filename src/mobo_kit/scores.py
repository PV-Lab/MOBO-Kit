"""Compute each objective's model input from the raw measurement columns.

The workbook also stores the derived values the model used to train on --
``Uniformity score`` (Z), ``Optoelectronic score`` (AA), ``Thickness (avg)``
(X) -- and three of those cells are **pasted literals, not formulas**, so they do
not update when the measurements behind them are edited.  That is the failure
that produced the original uniformity discrepancy, and an audit of all 15 R0 rows
on 2026-07-29 found the same divergence already present between ``AB`` and ``Y``:
``Y`` evaluates the thickness Gaussian on the rounded ``X`` while ``AB`` was
pasted from the same Gaussian on the unrounded T mean, and the two disagree by up
to 1.7e-3.

So the polarity is inverted here.  **Python computes; the workbook is checked.**
Each objective declares a ``measurement`` block naming a recipe and its input
columns.  The stored column becomes a cross-check that warns on disagreement and
never reaches the model.

The recipes are in code because they are the campaign's physics, not its
configuration; the column names and tolerances are in config because those are
what change when the dataset changes:

===================  =========================================================
``product``          ``Coverage * (1 - Uniformity) * Phase purity``      v2 only
``log10_product``    ``log10(Implied Voc) + log10(Photoconductance)``    v2 only
``mean``             the mean of every input, all of them required       v3
``mean_of_present``  mean of whichever of ``T1..T4`` were measured       both
===================  =========================================================

Two objective contracts are live at once and the recipes serve both.  The first
campaign (``d2d-objectives-v2-nm-thickness``) multiplied its uniformity terms and
took a log10 product for optoelectronic; the second
(``d2d-objectives-v3-test``) averages instead, on a workbook whose columns moved.
Recipes are never edited in place for a new dataset -- a redefined objective with
an unchanged name makes every cross-round hypervolume incomparable while every
plot still renders -- so a new shape arrives as a new recipe plus a new
``contract_version``.

``log10_product`` sums two logarithms rather than logging the product, which is
algebraically identical and cannot overflow on the way there.

``mean_of_present`` needs at least one reading; every other recipe needs all of
theirs.  **Blank means not measured, never zero.**  Thickness rows carry three or
four readings depending on the film, so a recipe that demanded all four would
reject the campaign; a blank ``Coverage``, by contrast, is a missing measurement
and ``mean`` refuses it.

Two input transforms carry a threshold, and both come from the v3 workbook:

* ``clamped_complement`` reproduces its ``Uniformity (clamped to 0.99)`` column,
  which pins a reading above 1 to 0.99 before taking the complement.  Two of the
  fifteen rows are clamped (uniformity 1.659 and 1.277).  The clamp is strictly
  above the threshold, so an exact 1.0 keeps its own value and yields a
  complement of exactly 0.
* ``capped_ratio`` reproduces ``Normalized Voc (to 1.4V)``.  **The cap is a
  deliberate divergence from the sheet**, which divides by 1.4 with no ceiling.
  It is dormant on the current data -- the largest observed reading is 1.135 --
  so the cross-check below agrees exactly today and would start warning the day a
  reading exceeds 1.4.  That is the intended behaviour: the divergence announces
  itself rather than being discovered later.

Cross-check tolerances differ by what the stored cell is, and the audited numbers
are the reason:

* a live formula (``Z``, ``R``) should agree to floating-point noise;
* a full-precision paste (``AA``) agrees to 1.8e-15 today, and if it ever stops
  agreeing that is exactly the staleness worth hearing about;
* a deliberately rounded cell (``X`` is ``ROUND(mean(T1..T4))``) can differ by
  half a unit and still be correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AgreementCheck",
    "FormulaFingerprint",
    "CrossCheck",
    "MeasurementInput",
    "MeasurementResult",
    "MeasurementSpec",
    "RECIPES",
    "ScoreFinding",
    "ScoreSeverity",
    "ScoreValidationError",
    "compute_measurements",
    "entry_columns",
    "measurement_spec_from_config",
    "row_completeness",
]


#: A cell holding this is empty as far as the campaign is concerned.  The
#: workbook's ``T anom`` column is full of non-breaking spaces, which are not
#: ``None`` and are not whitespace to ``str.strip`` unless normalised first.
_BLANK_TEXT = frozenset({"", "-", "--", "n/a", "na"})

#: Excel leaves non-breaking spaces in cells that look empty -- the
#: workbook's ``T anom`` column is full of them -- and ``str.strip`` does not
#: remove one, so it has to be normalised before any blank test.
_NBSP = "\u00a0"


class _NotNumeric(ValueError):
    """A cell holds something that is present but not a number."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.replace(_NBSP, " ").strip().lower() in _BLANK_TEXT
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _number(value: Any, *, column: str) -> float | None:
    """Return ``value`` as a float, ``None`` if the cell is empty."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise _NotNumeric(f"{column!r} holds a boolean, not a measurement.")
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.replace(_NBSP, " ").strip())
        except ValueError as exc:
            raise _NotNumeric(f"{column!r} holds {value!r}, which is not a number.") from exc
    else:
        raise _NotNumeric(f"{column!r} holds {type(value).__name__}, not a number.")
    if not math.isfinite(number):
        raise _NotNumeric(f"{column!r} holds {value!r}, which is not finite.")
    return number


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Recipe:
    """One way of turning measured columns into a single model input."""

    name: str
    combine: Callable[[Sequence[float]], float]
    #: ``True`` when every declared input must be present, ``False`` when the
    #: recipe averages over whichever ones were measured.
    requires_all: bool
    description: str

    def apply(self, values: Sequence[float]) -> float:
        result = float(self.combine(values))
        if not math.isfinite(result):
            raise _NotNumeric(f"recipe {self.name!r} produced a non-finite result.")
        return result


def _product(values: Sequence[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _log10_product(values: Sequence[float]) -> float:
    for value in values:
        if value <= 0:
            raise _NotNumeric(
                f"log10 needs strictly positive inputs; got {value!r}. A failed "
                "film must be recorded as blank, not as zero."
            )
    return math.fsum(math.log10(value) for value in values)


def _mean_of_present(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _single(values: Sequence[float]) -> float:
    """The one value, unchanged. Guarded because 'stored' means exactly one column."""
    if len(values) != 1:
        raise _NotNumeric(
            f"the 'stored' recipe takes exactly one column; got {len(values)}."
        )
    return float(values[0])


RECIPES: Mapping[str, _Recipe] = {
    recipe.name: recipe
    for recipe in (
        _Recipe("product", _product, True, "the product of every input"),
        _Recipe(
            "log10_product",
            _log10_product,
            True,
            "the sum of the base-10 logarithms, i.e. log10 of the product",
        ),
        # Same arithmetic as `mean_of_present`, opposite policy on a blank cell.
        # `mean` is for terms that were all supposed to be measured -- a missing
        # Coverage is a hole in the row, and averaging the other two would quietly
        # answer a different question. `mean_of_present` is for repeated readings
        # of one quantity, where three instead of four is a normal film.
        _Recipe("mean", _mean_of_present, True, "the mean of every input"),
        # The FROZEN objective. `stored` takes the workbook's own score column as
        # the objective value and computes nothing, which is a deliberate reversal
        # of this module's usual polarity -- normally Python computes and the
        # stored cell is demoted to a cross-check.
        #
        # It exists because the group is still revising how uniformity and
        # optoelectronic are defined. Reimplementing a formula that is about to
        # change means the code and the sheet disagree at exactly the moment
        # someone edits the sheet, and the disagreement would look like a bug in
        # whichever one was checked second. Reading the value instead makes the
        # workbook the single source of truth while the definition moves.
        #
        # What is LOST by freezing, and is worth saying out loud: there is no
        # independent recomputation of these two objectives under this contract,
        # so a stale pasted literal in the score column cannot be detected by
        # comparing it against anything. `formula_fingerprint` is the partial
        # replacement -- it notices when the DEFINITION moves, not when a value
        # goes stale.
        _Recipe(
            "stored",
            _single,
            True,
            "the workbook's own score column, taken as computed and not recomputed",
        ),
        _Recipe(
            "mean_of_present",
            _mean_of_present,
            False,
            "the mean of whichever inputs were measured",
        ),
    )
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MeasurementInput:
    """One measured column feeding a recipe.

    ``complement`` exists because the workbook records ``Uniformity`` and the
    score wants ``1 - Uniformity``.  The workbook also stores that complement in
    its own column, but as a pasted literal -- so it is computed here and the
    stored column is only ever a cross-check.  The same is true of the v3
    workbook's clamped copy of the reading.

    The two parameterised transforms take their thresholds from config rather
    than hard-coding them, because a clamp at 0.99 and a cap at 1.4 V are
    campaign decisions the group can revise, not physics.
    """

    column: str
    transform: str = "identity"
    #: ``clamped_complement``: readings STRICTLY above this are replaced.
    clamp_above: float | None = None
    #: ``clamped_complement``: what they are replaced with.
    clamp_to: float | None = None
    #: ``capped_ratio``: the ceiling, which is also the divisor.
    cap: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise ValueError("A measurement input needs a non-empty column name.")
        object.__setattr__(self, "column", self.column.strip())
        if self.transform not in _TRANSFORM_PARAMETERS:
            raise ValueError(
                f"Unsupported measurement transform {self.transform!r}; "
                f"expected one of {sorted(_TRANSFORM_PARAMETERS)}."
            )

        def _required(field: str, *, positive: bool = False) -> float:
            raw = getattr(self, field)
            if raw is None:
                raise ValueError(
                    f"Transform {self.transform!r} on column {self.column!r} needs "
                    f"{field!r}."
                )
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
                raise ValueError(
                    f"{field!r} on column {self.column!r} must be a number; "
                    f"got {raw!r}."
                )
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"{field!r} on column {self.column!r} must be finite.")
            if positive and number <= 0:
                raise ValueError(
                    f"{field!r} on column {self.column!r} must be positive; "
                    f"got {number!r}."
                )
            object.__setattr__(self, field, number)
            return number

        unused = [
            field
            for field in ("clamp_above", "clamp_to", "cap")
            if getattr(self, field) is not None
            and field not in _TRANSFORM_PARAMETERS[self.transform]
        ]
        if unused:
            # A threshold that silently does nothing is how a clamp gets believed
            # to be active when it is not.
            raise ValueError(
                f"Transform {self.transform!r} on column {self.column!r} ignores "
                f"{unused}; remove them or change the transform."
            )
        for field in _TRANSFORM_PARAMETERS[self.transform]:
            _required(field, positive=field == "cap")

    def evaluate(self, value: float) -> float:
        if self.transform == "identity":
            return value
        if self.transform == "complement":
            return 1.0 - value
        if self.transform == "clamped_complement":
            # strictly above, so an exact clamp_above keeps its own value
            clamped = self.clamp_to if value > self.clamp_above else value
            return 1.0 - float(clamped)
        return min(value, self.cap) / self.cap


#: Which thresholds each transform consumes. Declared once so an unused threshold
#: is an error rather than a silent no-op.
_TRANSFORM_PARAMETERS: Mapping[str, tuple[str, ...]] = {
    "identity": (),
    "complement": (),
    "clamped_complement": ("clamp_above", "clamp_to"),
    "capped_ratio": ("cap",),
}


@dataclass(frozen=True)
class CrossCheck:
    """A stored column to compare the computed value against.

    ``atol`` is per column on purpose: a live formula and a deliberately rounded
    literal do not deserve the same tolerance.
    """

    column: str
    atol: float = 0.005

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise ValueError("A cross-check needs a non-empty column name.")
        object.__setattr__(self, "column", self.column.strip())
        if isinstance(self.atol, (bool, np.bool_)) or not isinstance(self.atol, Real):
            raise ValueError(f"Cross-check {self.column!r} atol must be a number.")
        atol = float(self.atol)
        if not math.isfinite(atol) or atol < 0:
            raise ValueError(
                f"Cross-check {self.column!r} atol must be finite and non-negative."
            )
        object.__setattr__(self, "atol", atol)


@dataclass(frozen=True)
class AgreementCheck:
    """Does a supplied normalised column still rank like the raw one it summarises?

    Some columns arrive already normalised, with the derivation living outside the
    workbook -- ``Normalized photoconductance`` is one, and nothing in the sheet
    computes it.  A recipe can only take such a column on trust, which means a
    normalisation that has come loose from its raw measurement is invisible: every
    value is in range, every row computes, and the objective is simply about
    something else than it says.

    Rank agreement is the check that needs no formula.  Whatever the mapping is,
    a normalisation of a raw quantity must at least preserve its order, so
    Spearman between the two is expected to be strongly positive.  This reports it
    and warns below ``min_spearman``.

    It is a FINDING, never a gate: which column the model trains on is a decision
    for the group, and a diagnostic that blocks a round would make that decision
    by refusing to run.
    """

    raw: str
    normalized: str
    min_spearman: float = 0.0

    def __post_init__(self) -> None:
        for field in ("raw", "normalized"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"An agreement check needs a non-empty {field!r}.")
            object.__setattr__(self, field, value.strip())
        if self.raw == self.normalized:
            raise ValueError(
                f"An agreement check compares two different columns; got "
                f"{self.raw!r} twice."
            )
        threshold = self.min_spearman
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise ValueError("min_spearman must be a number.")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("min_spearman must be finite and within [-1, 1].")
        object.__setattr__(self, "min_spearman", threshold)


@dataclass(frozen=True)
class FormulaFingerprint:
    """The formula text a frozen score column is expected to carry.

    A ``stored`` objective is read rather than recomputed, so nothing in Python
    knows what it means.  That is the point -- the group is still revising the
    definition -- but it removes the cross-check that would otherwise catch a
    redefinition.  This is the partial replacement: record the formula as it
    stands, and say so when it changes.

    **It notices a changed definition, not a stale value.**  A pasted literal that
    has stopped tracking its inputs looks identical to a correct one from here.
    That gap is inherent to freezing and is recorded rather than papered over; it
    closes when the group settles the formulas and the recipes are unfrozen.

    Compared after collapsing whitespace and upper-casing, because Excel rewrites
    those freely, and with the row number stripped so one fingerprint covers every
    row rather than fifteen near-copies.
    """

    column: str
    formula: str

    def __post_init__(self) -> None:
        for field in ("column", "formula"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"A formula fingerprint needs a non-empty {field!r}.")
            object.__setattr__(self, field, value.strip())

    @staticmethod
    def canonical(formula: Any) -> str:
        """Row-independent, whitespace-independent form of a formula string."""
        import re

        if not isinstance(formula, str):
            return ""
        text = formula.strip().lstrip("=").upper()
        text = re.sub(r"\s+", "", text)
        # A2 -> A, AJ17 -> AJ: the same formula copied down a column differs only
        # in the row, and fingerprinting per row would report fifteen changes for
        # one edit.
        return re.sub(r"(\$?[A-Z]{1,3})\$?\d+", r"\1", text)

    def matches(self, formula: Any) -> bool:
        return self.canonical(formula) == self.canonical(self.formula)


@dataclass(frozen=True)
class MeasurementSpec:
    """How one objective's model input is computed and checked."""

    name: str
    recipe: str
    inputs: tuple[MeasurementInput, ...]
    cross_checks: tuple[CrossCheck, ...] = ()
    #: Rank agreement between a supplied normalised column and its raw source.
    agreement_check: "AgreementCheck | None" = None
    #: Expected formula text for a frozen score column; checked, never evaluated.
    formula_fingerprint: "FormulaFingerprint | None" = None
    #: Columns holding readings the operator judged anomalous.  They never enter
    #: the recipe; their presence is recorded so an exclusion is visible rather
    #: than silent.
    excluded: tuple[str, ...] = ()
    #: Warn when ``(max - min) / mean`` over the used inputs exceeds this.  Set
    #: for thickness, where three R0 rows hold readings that split into two
    #: clusters rather than scattering around one value.
    spread_warning_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A measurement spec needs a non-empty name.")
        object.__setattr__(self, "name", self.name.strip())
        if self.recipe not in RECIPES:
            raise ValueError(
                f"Objective {self.name!r} requests unknown recipe {self.recipe!r}; "
                f"known recipes are {sorted(RECIPES)}."
            )
        if not self.inputs:
            raise ValueError(f"Objective {self.name!r} declares no measurement inputs.")
        if self.recipe == "stored" and len(self.inputs) != 1:
            raise ValueError(
                f"Objective {self.name!r} uses the 'stored' recipe, which reads one "
                f"score column; it declares {len(self.inputs)} inputs."
            )
        columns = [item.column for item in self.inputs]
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"Objective {self.name!r} repeats measurement input(s) {duplicates}."
            )
        if self.spread_warning_ratio is not None:
            ratio = float(self.spread_warning_ratio)
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(
                    f"Objective {self.name!r} spread_warning_ratio must be a "
                    "positive finite number."
                )
            object.__setattr__(self, "spread_warning_ratio", ratio)

    @property
    def recipe_impl(self) -> _Recipe:
        return RECIPES[self.recipe]

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns without which this objective cannot be computed at all."""
        if self.recipe_impl.requires_all:
            return tuple(item.column for item in self.inputs)
        return ()

    @property
    def optional_columns(self) -> tuple[str, ...]:
        # BOTH of the agreement check's columns are offered but never required.
        # It used to list only `raw`, on the assumption that `normalized` was an
        # input to the recipe -- true while optoelectronic was computed, false the
        # moment it was frozen and its only input became the score column. The
        # check then silently reported "column absent" on a sheet that had it.
        agreement = (
            (self.agreement_check.raw, self.agreement_check.normalized)
            if self.agreement_check
            else ()
        )
        if self.recipe_impl.requires_all:
            return tuple(self.excluded) + agreement
        return (
            tuple(item.column for item in self.inputs)
            + tuple(self.excluded)
            + agreement
        )


def measurement_spec_from_config(entry: Mapping[str, Any]) -> MeasurementSpec | None:
    """Build a spec from one objective's ``measurement`` block, if present.

    Returning ``None`` for an objective without the block is deliberate: an
    objective that still reads its stored column keeps working unchanged.
    """
    block = entry.get("measurement")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError("measurement must be a mapping.")

    raw_inputs = block.get("inputs")
    if (
        not isinstance(raw_inputs, Sequence)
        or isinstance(raw_inputs, (str, bytes))
        or not raw_inputs
    ):
        raise ValueError("measurement.inputs must be a non-empty list.")
    inputs = []
    _INPUT_KEYS = {"column", "transform", "clamp_above", "clamp_to", "cap"}
    for item in raw_inputs:
        if isinstance(item, str):
            inputs.append(MeasurementInput(item))
        elif isinstance(item, Mapping):
            # A misspelt threshold would otherwise be dropped in silence, leaving
            # a clamp everyone believes is configured and nothing applying it.
            unknown = sorted(set(item) - _INPUT_KEYS)
            if unknown:
                raise ValueError(
                    f"Measurement input {item.get('column')!r} has unknown key(s) "
                    f"{unknown}; expected {sorted(_INPUT_KEYS)}."
                )
            inputs.append(
                MeasurementInput(
                    str(item["column"]),
                    str(item.get("transform", "identity")),
                    clamp_above=item.get("clamp_above"),
                    clamp_to=item.get("clamp_to"),
                    cap=item.get("cap"),
                )
            )
        else:
            raise ValueError("Each measurement input must be a string or a mapping.")

    raw_checks = block.get("cross_check") or ()
    if isinstance(raw_checks, Mapping):
        raw_checks = [raw_checks]
    elif isinstance(raw_checks, (str, bytes)):
        raw_checks = [{"column": raw_checks}]
    checks = []
    for item in raw_checks:
        if isinstance(item, str):
            checks.append(CrossCheck(item))
        elif isinstance(item, Mapping):
            atol = item.get("atol")
            checks.append(
                CrossCheck(str(item["column"]), 0.005 if atol is None else float(atol))
            )
        else:
            raise ValueError("Each cross_check must be a string or a mapping.")

    raw_excluded = block.get("excluded") or ()
    if isinstance(raw_excluded, (str, bytes)):
        raw_excluded = [raw_excluded]
    excluded = tuple(
        str(item["column"]) if isinstance(item, Mapping) else str(item)
        for item in raw_excluded
    )

    raw_fingerprint = block.get("formula_fingerprint")
    if raw_fingerprint is None:
        fingerprint = None
    elif isinstance(raw_fingerprint, Mapping):
        fingerprint = FormulaFingerprint(
            column=str(raw_fingerprint["column"]),
            formula=str(raw_fingerprint["formula"]),
        )
    else:
        raise ValueError("measurement.formula_fingerprint must be a mapping.")

    raw_agreement = block.get("agreement_check")
    if raw_agreement is None:
        agreement = None
    elif isinstance(raw_agreement, Mapping):
        threshold = raw_agreement.get("min_spearman")
        agreement = AgreementCheck(
            raw=str(raw_agreement["raw"]),
            normalized=str(raw_agreement["normalized"]),
            min_spearman=0.0 if threshold is None else float(threshold),
        )
    else:
        raise ValueError("measurement.agreement_check must be a mapping.")

    ratio = block.get("spread_warning_ratio")
    return MeasurementSpec(
        name=str(entry.get("name", block.get("name", "<unnamed>"))),
        recipe=str(block["recipe"]),
        inputs=tuple(inputs),
        cross_checks=tuple(checks),
        agreement_check=agreement,
        formula_fingerprint=fingerprint,
        excluded=excluded,
        spread_warning_ratio=None if ratio is None else float(ratio),
    )


def entry_columns(
    specs: Sequence[MeasurementSpec],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The columns a worklist sheet must offer, split required / optional.

    Order is declaration order, de-duplicated.  A column that is required by one
    objective and optional for another counts as required.
    """
    required: list[str] = []
    optional: list[str] = []
    for spec in specs:
        for column in spec.required_columns:
            if column not in required:
                required.append(column)
        for column in spec.optional_columns:
            if column not in optional:
                optional.append(column)
    optional = [column for column in optional if column not in required]
    return tuple(required), tuple(optional)


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #


class ScoreSeverity(str, Enum):
    """How much a finding should stop you."""

    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ScoreFinding:
    """One thing worth saying about one row."""

    severity: ScoreSeverity
    code: str
    objective: str
    row_position: int
    sample_id: Any
    message: str
    column: str | None = None

    @property
    def is_column_level(self) -> bool:
        """True when the finding is about a COLUMN rather than about one row.

        ``row_position = -1`` is the marker. Rank agreement over fifteen rows and
        an absent column are both statements about the sheet, not about a film.
        """
        return self.row_position < 0

    def __str__(self) -> str:
        # A column-level finding used to render as "sample ?", which reads as a
        # row whose identity got lost rather than as a finding that has no row.
        where = (
            f"all rows, {self.objective}"
            if self.is_column_level
            else f"sample {self.sample_id}, {self.objective}"
        )
        return f"[{self.severity.value}] {where}: {self.message}"


class ScoreValidationError(ValueError):
    """At least one row could not be turned into a model input."""

    def __init__(self, findings: Sequence[ScoreFinding]) -> None:
        self.findings = tuple(findings)
        detail = "\n- ".join(str(finding) for finding in self.findings)
        super().__init__(f"Objective values could not be computed:\n- {detail}")


@dataclass(frozen=True)
class MeasurementResult:
    """Computed model inputs, how many readings each used, and what to say."""

    values: pd.DataFrame
    """One column per objective, in declaration order. NaN where unusable."""
    inputs_used: pd.DataFrame
    """How many measured inputs each value was computed from."""
    findings: tuple[ScoreFinding, ...]

    def _by(self, severity: ScoreSeverity) -> tuple[ScoreFinding, ...]:
        return tuple(f for f in self.findings if f.severity is severity)

    @property
    def notes(self) -> tuple[ScoreFinding, ...]:
        return self._by(ScoreSeverity.NOTE)

    @property
    def warnings(self) -> tuple[ScoreFinding, ...]:
        return self._by(ScoreSeverity.WARNING)

    @property
    def errors(self) -> tuple[ScoreFinding, ...]:
        return self._by(ScoreSeverity.ERROR)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def findings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "severity": [f.severity.value for f in self.findings],
                "code": [f.code for f in self.findings],
                "objective": [f.objective for f in self.findings],
                "sample_id": [f.sample_id for f in self.findings],
                "column": [f.column for f in self.findings],
                "message": [f.message for f in self.findings],
            }
        )

    def raise_for_errors(self) -> "MeasurementResult":
        if self.errors:
            raise ScoreValidationError(self.errors)
        return self


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #


def _cell(frame: pd.DataFrame, column: str, position: int) -> Any:
    return frame[column].to_numpy(dtype=object)[position]


def _with_stripped_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare column names with surrounding whitespace removed, on both sides.

    :class:`MeasurementInput` and :class:`CrossCheck` strip the names they are
    given, so a config may quote a header verbatim -- and the v3 sheet has two
    that end in a space, ``'PL - Implied Voc (Max) Raw '`` and ``'Normalized
    photoconductance '``.  ``workbook_io`` strips the sheet side when it builds
    its header index, but a caller who reads the sheet with pandas directly does
    not, and would then be told the column is missing.  It fails closed rather
    than silently, but "missing" is the wrong answer to give about a column that
    is right there.

    Renaming is skipped entirely when it would merge two distinct labels, because
    quietly dropping one of them would be worse than the confusion this avoids.
    """
    labels = list(frame.columns)
    stripped = [name.strip() if isinstance(name, str) else name for name in labels]
    if stripped == labels:
        return frame
    if len(set(map(str, stripped))) != len(stripped):
        return frame
    renamed = frame.copy(deep=False)
    renamed.columns = stripped
    return renamed


def _absent_required_columns(
    frame: pd.DataFrame, specs: Sequence[MeasurementSpec]
) -> tuple[str, ...]:
    """Input columns whose absence makes an objective impossible, not merely thin.

    A recipe that needs all its inputs cannot proceed without any one of them. A
    recipe that averages over whatever was measured can: a sheet with no ``T4``
    column is a sheet where nobody measured a fourth point, which is the same
    situation as an empty ``T4`` cell and is handled the same way.
    """
    absent: list[str] = []
    for spec in specs:
        for column in spec.required_columns:
            if column not in frame.columns and column not in absent:
                absent.append(column)
    return tuple(absent)


def _agreement_findings(
    frame: pd.DataFrame,
    spec: MeasurementSpec,
    sample_ids: Sequence[Any],
) -> list[ScoreFinding]:
    """Rank-compare a supplied normalised column against the raw one it summarises.

    One finding for the whole column, not one per row -- the question is about the
    mapping, so ``row_position`` is -1 and ``sample_id`` is None.  The message
    names the highest-raw film explicitly, because "Spearman is negative" is a
    statistic and "the strongest film scores lowest" is the thing a reviewer can
    act on.
    """
    check = spec.agreement_check
    assert check is not None  # caller checks; keeps the type narrow

    def finding(severity: ScoreSeverity, code: str, message: str) -> ScoreFinding:
        return ScoreFinding(
            severity=severity,
            code=code,
            objective=spec.name,
            row_position=-1,
            sample_id=None,
            message=message,
            column=check.normalized,
        )

    missing = [c for c in (check.raw, check.normalized) if c not in frame.columns]
    if missing:
        return [
            finding(
                ScoreSeverity.NOTE,
                "agreement_check_absent",
                f"{missing} not in this sheet, so {check.normalized!r} cannot be "
                f"checked against {check.raw!r}.",
            )
        ]

    pairs: list[tuple[float, float, Any]] = []
    for position in range(len(frame)):
        try:
            raw = _number(_cell(frame, check.raw, position), column=check.raw)
            normalized = _number(
                _cell(frame, check.normalized, position), column=check.normalized
            )
        except _NotNumeric:
            continue
        if raw is None or normalized is None:
            continue
        pairs.append((raw, normalized, sample_ids[position]))

    if len(pairs) < 3:
        return [
            finding(
                ScoreSeverity.NOTE,
                "agreement_check_too_few_rows",
                f"only {len(pairs)} row(s) have both {check.raw!r} and "
                f"{check.normalized!r}; a rank comparison needs at least 3.",
            )
        ]

    raw_values = [item[0] for item in pairs]
    normalized_values = [item[1] for item in pairs]
    # Checked before calling scipy rather than by testing the result for NaN: a
    # constant column makes `spearmanr` emit a ConstantInputWarning, and this
    # suite keeps its warning tail fixed so that a NEW warning means something.
    constant = [
        name
        for name, series in (
            (check.raw, raw_values),
            (check.normalized, normalized_values),
        )
        if len(set(series)) == 1
    ]
    if constant:
        return [
            finding(
                ScoreSeverity.NOTE,
                "agreement_check_undefined",
                f"rank correlation is undefined over {len(pairs)} rows because "
                f"{constant} is constant.",
            )
        ]

    from scipy.stats import spearmanr

    result = spearmanr(raw_values, normalized_values)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(rho):  # pragma: no cover - constant input is caught above
        return [
            finding(
                ScoreSeverity.NOTE,
                "agreement_check_undefined",
                f"rank correlation over {len(pairs)} rows is not a finite number.",
            )
        ]

    strongest = max(pairs, key=lambda item: item[0])
    weakest = min(pairs, key=lambda item: item[0])
    detail = (
        f"Spearman({check.raw!r}, {check.normalized!r}) = {rho:+.4f} "
        f"(p = {p_value:.4f}) over {len(pairs)} rows. The highest raw reading "
        f"({strongest[0]:.4g}, sample {strongest[2]}) normalises to "
        f"{strongest[1]:.4g}; the lowest ({weakest[0]:.4g}, sample {weakest[2]}) "
        f"normalises to {weakest[1]:.4g}."
    )
    if rho < check.min_spearman:
        return [
            finding(
                ScoreSeverity.WARNING,
                "agreement_not_monotonic",
                f"{check.normalized!r} does not rank like {check.raw!r}, so this "
                f"objective is provisional until the group supplies the "
                f"normalisation. {detail} Expected at least "
                f"{check.min_spearman:+.4f}. The computed value still stands -- "
                f"this is a finding, not a gate.",
            )
        ]
    return [
        finding(
            ScoreSeverity.NOTE,
            "agreement_monotonic",
            f"{check.normalized!r} ranks like {check.raw!r}. {detail}",
        )
    ]


def compute_measurements(
    frame: pd.DataFrame,
    specs: Sequence[MeasurementSpec],
    *,
    sample_ids: Sequence[Any] | None = None,
) -> MeasurementResult:
    """Compute one model input per objective per row, and check the workbook.

    ``frame`` holds the raw cells as read -- blanks, non-breaking spaces and
    strings are all handled here rather than by the caller, because the point of
    this module is that nothing downstream has to know how the workbook spells
    "not measured".

    A row that cannot be computed gets ``NaN`` and an ``error`` finding rather
    than an exception, so one bad row does not hide the state of the other
    fourteen.  Call :meth:`MeasurementResult.raise_for_errors` to fail closed.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    specs = tuple(specs)
    if not specs:
        raise ValueError("At least one MeasurementSpec is required.")
    frame = _with_stripped_columns(frame)
    absent = _absent_required_columns(frame, specs)
    if absent:
        raise ValueError(
            f"Measurement column(s) missing from the sheet: {list(absent)}. The "
            "objectives are computed from these, so they cannot be skipped."
        )

    n_rows = len(frame)
    ids = list(sample_ids) if sample_ids is not None else [None] * n_rows
    if len(ids) != n_rows:
        raise ValueError("sample_ids must have one entry per row.")

    findings: list[ScoreFinding] = []
    for spec in specs:
        for item in spec.inputs:
            if item.column not in frame.columns:
                findings.append(
                    ScoreFinding(
                        severity=ScoreSeverity.NOTE,
                        code="input_column_absent",
                        objective=spec.name,
                        row_position=-1,
                        sample_id=None,
                        message=(
                            f"there is no {item.column!r} column in this sheet, so "
                            "it counts as unmeasured for every row."
                        ),
                        column=item.column,
                    )
                )
    values: dict[str, list[float]] = {spec.name: [] for spec in specs}
    counts: dict[str, list[int]] = {spec.name: [] for spec in specs}

    for spec in specs:
        recipe = spec.recipe_impl
        for position in range(n_rows):
            sample_id = ids[position]

            def record(
                severity: ScoreSeverity,
                code: str,
                message: str,
                *,
                column: str | None = None,
                _position: int = position,
                _sample_id: Any = sample_id,
                _objective: str = spec.name,
            ) -> None:
                findings.append(
                    ScoreFinding(
                        severity=severity,
                        code=code,
                        objective=_objective,
                        row_position=_position,
                        sample_id=_sample_id,
                        message=message,
                        column=column,
                    )
                )

            used: list[float] = []
            raw_used: list[float] = []
            failed = False
            for item in spec.inputs:
                if item.column not in frame.columns:
                    continue  # already reported once, above
                try:
                    number = _number(_cell(frame, item.column, position), column=item.column)
                except _NotNumeric as exc:
                    record(
                        ScoreSeverity.ERROR,
                        "input_not_numeric",
                        str(exc),
                        column=item.column,
                    )
                    failed = True
                    continue
                if number is None:
                    if recipe.requires_all:
                        record(
                            ScoreSeverity.ERROR,
                            "input_missing",
                            f"{item.column!r} is empty, and {spec.recipe!r} needs "
                            "every input. Blank means not measured, not zero.",
                            column=item.column,
                        )
                        failed = True
                    continue
                raw_used.append(number)
                used.append(item.evaluate(number))

            for column in spec.excluded:
                if column not in frame.columns:
                    continue
                try:
                    excluded_value = _number(_cell(frame, column, position), column=column)
                except _NotNumeric:
                    excluded_value = None
                if excluded_value is not None:
                    record(
                        ScoreSeverity.NOTE,
                        "reading_excluded",
                        f"{column!r} holds {excluded_value:g}, a reading judged "
                        "anomalous by the operator. It is recorded, not averaged.",
                        column=column,
                    )

            if failed or not used:
                if not failed:
                    record(
                        ScoreSeverity.ERROR,
                        "no_inputs_measured",
                        f"none of {[i.column for i in spec.inputs]} was measured, so "
                        f"{spec.name!r} cannot be computed for this row.",
                    )
                values[spec.name].append(float("nan"))
                counts[spec.name].append(len(used))
                continue

            try:
                value = recipe.apply(used)
            except _NotNumeric as exc:
                record(ScoreSeverity.ERROR, "recipe_failed", str(exc))
                values[spec.name].append(float("nan"))
                counts[spec.name].append(len(used))
                continue

            if spec.spread_warning_ratio is not None and len(raw_used) > 1:
                spread = max(raw_used) - min(raw_used)
                centre = math.fsum(raw_used) / len(raw_used)
                if centre != 0 and spread / abs(centre) > spec.spread_warning_ratio:
                    record(
                        ScoreSeverity.WARNING,
                        "readings_disagree",
                        f"{len(raw_used)} readings span {spread:g} around a mean of "
                        f"{centre:g} ({100 * spread / abs(centre):.0f}% of it): "
                        f"{[f'{v:g}' for v in raw_used]}. The mean may not describe "
                        "this film.",
                    )

            for check in spec.cross_checks:
                if check.column not in frame.columns:
                    record(
                        ScoreSeverity.NOTE,
                        "cross_check_absent",
                        f"cross-check column {check.column!r} is not in the sheet; "
                        "the computed value stands unchecked.",
                        column=check.column,
                    )
                    continue
                try:
                    stored = _number(_cell(frame, check.column, position), column=check.column)
                except _NotNumeric as exc:
                    record(
                        ScoreSeverity.WARNING,
                        "cross_check_not_numeric",
                        str(exc),
                        column=check.column,
                    )
                    continue
                if stored is None:
                    record(
                        ScoreSeverity.WARNING,
                        "cross_check_empty",
                        f"{check.column!r} is empty. If it is a formula column, a "
                        "non-Excel tool has saved this file and dropped the cached "
                        "value.",
                        column=check.column,
                    )
                    continue
                difference = abs(value - stored)
                if difference > check.atol + 1e-12:
                    record(
                        ScoreSeverity.WARNING,
                        "cross_check_mismatch",
                        f"computed {value:.10g} against stored {stored:.10g} in "
                        f"{check.column!r}, a difference of {difference:.3g} above "
                        f"the {check.atol:g} tolerance. The computed value is what "
                        "the model uses.",
                        column=check.column,
                    )

            values[spec.name].append(value)
            counts[spec.name].append(len(used))

        if spec.agreement_check is not None:
            findings.extend(_agreement_findings(frame, spec, ids))

    index = frame.index
    return MeasurementResult(
        values=pd.DataFrame(
            {spec.name: values[spec.name] for spec in specs}, index=index, dtype=float
        ),
        inputs_used=pd.DataFrame(
            {spec.name: counts[spec.name] for spec in specs}, index=index, dtype=int
        ),
        findings=tuple(findings),
    )


def row_completeness(
    frame: pd.DataFrame, specs: Sequence[MeasurementSpec]
) -> pd.Series:
    """Which rows have enough measurements for every objective.

    This is what "has this row been measured yet" means once objectives are
    computed rather than read: ``product`` and ``log10_product`` need all their
    inputs, ``mean_of_present`` needs one.  Round detection uses it, so getting
    it wrong either blocks a finished round or advances on a half-filled sheet.
    """
    specs = tuple(specs)
    frame = _with_stripped_columns(frame)
    complete = pd.Series(True, index=frame.index)
    for spec in specs:
        requires_all = spec.recipe_impl.requires_all
        for position in range(len(frame)):
            present = 0
            for item in spec.inputs:
                if item.column not in frame.columns:
                    continue
                try:
                    number = _number(_cell(frame, item.column, position), column=item.column)
                except _NotNumeric:
                    number = None
                if number is not None:
                    present += 1
            enough = (
                present == len(spec.inputs) if requires_all else present >= 1
            )
            if not enough:
                complete.iloc[position] = False
    return complete


def describe_findings(findings: Iterable[ScoreFinding]) -> str:
    """A short human-readable block, worst first.  Empty string when clean."""
    ordered = sorted(
        findings,
        key=lambda f: (
            {ScoreSeverity.ERROR: 0, ScoreSeverity.WARNING: 1, ScoreSeverity.NOTE: 2}[
                f.severity
            ],
            f.row_position,
        ),
    )
    return "\n".join(str(finding) for finding in ordered)
