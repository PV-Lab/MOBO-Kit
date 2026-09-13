"""Opt-in physical-space constraints for campaign designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .design import Design


RowConstraint = Callable[[np.ndarray, "Design"], np.ndarray]


@dataclass(frozen=True)
class NamedConstraint:
    """A row constraint that can say what it is when it rejects something.

    Callable, so it is a ``RowConstraint`` everywhere one is expected and the
    candidate pool needs no knowledge of it.  The name and description exist for
    the review artifact: a reviewer being told a condition was rejected needs the
    rule, not an index.
    """

    name: str
    description: str
    check: RowConstraint

    def __call__(self, X: np.ndarray, design: "Design") -> np.ndarray:
        return self.check(X, design)


def check_clausius_clapeyron_np(ah_vals, temp_c_vals) -> np.ndarray:
    """Return a validity mask for the Clausius-Clapeyron constraint.

    Absolute humidity is expressed in g/m^3 and temperature in degrees C.
    The calculation converts temperature to kelvin internally.
    """

    absolute_humidity = np.asarray(ah_vals, dtype=float)
    temperature_k = np.asarray(temp_c_vals, dtype=float) + 273.15

    saturation_pressure = 0.6113 * np.exp(
        (17.27 * (temperature_k - 273.15)) / (temperature_k - 35.86)
    )
    maximum_absolute_humidity = saturation_pressure / (4.61e-4 * temperature_k)
    return (absolute_humidity <= maximum_absolute_humidity) & np.isfinite(
        maximum_absolute_humidity
    )


def _idx_for(name: str, design: "Design") -> int:
    try:
        return design.names.index(name)
    except ValueError as exc:
        raise KeyError(
            f"Constraint refers to column '{name}', but it is not in "
            f"design.names={design.names}."
        ) from exc


def _build_clausius_clapeyron(specification: Dict, design: "Design") -> RowConstraint:
    absolute_humidity_column = specification.get("ah_col")
    temperature_column = specification.get("temp_c_col")
    if absolute_humidity_column is None or temperature_column is None:
        raise KeyError(
            "Clausius-Clapeyron constraint requires 'ah_col' and 'temp_c_col'."
        )

    absolute_humidity_index = _idx_for(absolute_humidity_column, design)
    temperature_index = _idx_for(temperature_column, design)

    def row_constraint_fn(X: np.ndarray, _design: "Design") -> np.ndarray:
        return check_clausius_clapeyron_np(
            X[:, absolute_humidity_index], X[:, temperature_index]
        )

    return row_constraint_fn


def _column_names(names, *, field: str) -> List[str]:
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names:
        raise KeyError(
            f"Constraint field {field!r} must name one column or a non-empty list "
            f"of columns; got {names!r}."
        )
    return [str(name) for name in names]


def _build_zero_coupled(specification, design: "Design") -> RowConstraint:
    """Two settings that describe one optional process step: both on, or both off.

    A second spin stage that runs for 0 s at 3500 rpm is not a slower stage, it is
    a contradiction -- and so is one that runs for 30 s at 0 rpm. Exactly one of
    the pair being zero is the invalid case; both zero means the step was skipped,
    which is a real recipe (sample 2 of the v3 workbook is a one-step film).

    Stated as an iff rather than as two separate bounds because that is what makes
    the skipped-step recipe reachable at all: a plain lower bound on either column
    would delete it.
    """
    names = _column_names(specification, field="zero_coupled")
    if len(names) < 2:
        raise KeyError("zero_coupled needs at least two columns to couple.")
    indices = [_idx_for(name, design) for name in names]

    def row_constraint_fn(X: np.ndarray, _design: "Design") -> np.ndarray:
        block = np.asarray(X, dtype=float)[:, indices]
        zeros = np.isclose(block, 0.0, rtol=0.0, atol=1e-12)
        return zeros.all(axis=1) | (~zeros).all(axis=1)

    return row_constraint_fn


def _build_sum_upper_strict(specification, design: "Design") -> RowConstraint:
    """``lhs < sum(rhs)``, strictly.

    Written for ``anti_time < time_1 + time_2``: the antisolvent has to be dropped
    while the substrate is still spinning, so equality is already too late rather
    than just in time. Strictness is the whole point of the constraint and is why
    this is not the existing bounds check with a different argument.
    """
    if not isinstance(specification, dict):
        raise KeyError(
            "sum_upper_strict takes a mapping with 'lhs' and 'rhs'; "
            f"got {specification!r}."
        )
    left = specification.get("lhs")
    if not isinstance(left, str) or not left.strip():
        raise KeyError("sum_upper_strict needs 'lhs' to name one column.")
    left_index = _idx_for(left.strip(), design)
    right_indices = [
        _idx_for(name, design)
        for name in _column_names(specification.get("rhs"), field="rhs")
    ]

    def row_constraint_fn(X: np.ndarray, _design: "Design") -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return values[:, left_index] < values[:, right_indices].sum(axis=1)

    return row_constraint_fn


def _build_nonzero_minimum(specification, design: "Design") -> RowConstraint:
    """A column is either exactly zero or at least ``minimum``.

    This exists because the design grid is arithmetic -- ``start``/``stop``/``step``
    with uniform spacing, which ``lhs`` asserts -- so a grid of ``{0} U {10, 15,
    ... 60}`` cannot be declared directly. Reaching 0 with ``step: 5`` also reaches
    5, and a 5 s second spin stage was not in the first campaign's design and has
    never been run.

    Declaring the hole here keeps it visible in config and enforced everywhere the
    other constraints are, rather than widening the design space in silence.
    """
    if not isinstance(specification, dict):
        raise KeyError(
            "nonzero_minimum takes a mapping with 'column' and 'minimum'; "
            f"got {specification!r}."
        )
    column = specification.get("column")
    if not isinstance(column, str) or not column.strip():
        raise KeyError("nonzero_minimum needs 'column' to name one column.")
    index = _idx_for(column.strip(), design)
    minimum = specification.get("minimum")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise KeyError("nonzero_minimum needs a numeric 'minimum'.")
    threshold = float(minimum)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("nonzero_minimum 'minimum' must be finite and positive.")

    def row_constraint_fn(X: np.ndarray, _design: "Design") -> np.ndarray:
        values = np.asarray(X, dtype=float)[:, index]
        return np.isclose(values, 0.0, rtol=0.0, atol=1e-12) | (values >= threshold)

    return row_constraint_fn


_SUPPORTED_BOOL_KEYS = {
    "clausius_clapeyron": _build_clausius_clapeyron,
}

#: Types whose entry key carries the constraint's parameters rather than a bool.
#: ``clausius_clapeyron: true`` predates these and keeps its flag spelling.
_SUPPORTED_VALUE_KEYS = {
    "zero_coupled": _build_zero_coupled,
    "sum_upper_strict": _build_sum_upper_strict,
    "nonzero_minimum": _build_nonzero_minimum,
}


def constraints_from_config(cfg: Dict, design: "Design") -> List[RowConstraint]:
    """Build only explicitly configured campaign constraints.

    An entry names exactly one type. ``clausius_clapeyron`` takes a boolean flag
    and its parameters as siblings; the rest carry their parameters on the type
    key itself::

        constraints:
          - clausius_clapeyron: true
            ah_col: absolute_humidity
            temp_c_col: temperature_c

          - zero_coupled: [speed_2, time_2]
          - sum_upper_strict: {lhs: anti_time, rhs: [time_1, time_2]}
          - nonzero_minimum: {column: time_2, minimum: 10}

    An optional ``name:`` overrides the label a violation is reported under.

    Missing ``constraints`` and an empty list both mean no constraints. Invalid
    explicit entries fail instead of being silently ignored.
    """

    if not isinstance(cfg, dict):
        raise TypeError("Constraint configuration must be a mapping.")

    items = cfg.get("constraints", [])
    if not items:
        return []
    if not isinstance(items, list):
        raise TypeError("Config 'constraints' must be a list of mappings.")

    known_types = sorted({*_SUPPORTED_BOOL_KEYS, *_SUPPORTED_VALUE_KEYS})
    constraints: List[RowConstraint] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise TypeError(
                f"Constraint entry at index {index} must be a mapping; "
                f"got {type(raw).__name__}."
            )

        known_keys = [key for key in known_types if key in raw]
        if not known_keys:
            raise KeyError(
                f"Constraint entry at index {index} has no supported type; "
                f"expected one of {known_types}."
            )
        if len(known_keys) > 1:
            raise ValueError(
                f"Constraint entry at index {index} enables multiple types: "
                f"{known_keys}. Use one constraint type per entry."
            )

        chosen_key = known_keys[0]
        if chosen_key in _SUPPORTED_BOOL_KEYS:
            enabled = raw[chosen_key]
            if not isinstance(enabled, bool):
                raise TypeError(
                    f"Constraint flag '{chosen_key}' must be true or false; "
                    f"got {enabled!r}."
                )
            if not enabled:
                continue
            builder = _SUPPORTED_BOOL_KEYS[chosen_key]
            # the flag spelling keeps its parameters as siblings of the flag
            parameters = raw
        else:
            builder = _SUPPORTED_VALUE_KEYS[chosen_key]
            # the value spelling carries its parameters on the type key itself
            parameters = raw[chosen_key]

        label = raw.get("name")
        constraints.append(
            NamedConstraint(
                name=str(label) if label else chosen_key,
                description=_describe(chosen_key, parameters),
                check=builder(parameters, design),
            )
        )

    return constraints


def _describe(kind: str, parameters) -> str:
    """A one-line statement of the rule, for a reviewer rather than a log."""
    if kind == "zero_coupled":
        names = [parameters] if isinstance(parameters, str) else list(parameters or ())
        return f"{' and '.join(str(c) for c in names)} are all zero or all nonzero"
    if kind == "sum_upper_strict" and isinstance(parameters, dict):
        rhs = parameters.get("rhs")
        names = [rhs] if isinstance(rhs, str) else list(rhs or ())
        return f"{parameters.get('lhs')} < {' + '.join(str(c) for c in names)}"
    if kind == "nonzero_minimum" and isinstance(parameters, dict):
        return (
            f"{parameters.get('column')} is 0 or at least "
            f"{parameters.get('minimum')}"
        )
    return kind


def constraint_violations(
    X_phys: np.ndarray,
    design: "Design",
    constraints: Optional[Sequence[RowConstraint]],
) -> List[List[str]]:
    """Per row, the names of the constraints it breaks.

    :func:`apply_row_constraints` answers "may this row be used"; this answers
    "and if not, which rule". A batch review needs the second, because "condition
    3 is invalid" is not something anyone can act on.
    """
    X_phys = np.asarray(X_phys, dtype=float)
    if X_phys.ndim != 2:
        raise ValueError("Physical input array must be two-dimensional.")
    per_row: List[List[str]] = [[] for _ in range(X_phys.shape[0])]
    if not constraints or X_phys.shape[0] == 0:
        return per_row
    for index, constraint in enumerate(constraints):
        mask = apply_row_constraints(X_phys, design, [constraint])
        label = getattr(constraint, "name", f"constraint #{index}")
        for row in np.flatnonzero(~mask):
            per_row[int(row)].append(label)
    return per_row


def apply_row_constraints(
    X_phys: np.ndarray,
    design: "Design",
    constraints: Optional[Sequence[RowConstraint]],
) -> np.ndarray:
    """AND zero or more row constraints over physical-space input rows."""

    X_phys = np.asarray(X_phys, dtype=float)
    if X_phys.ndim != 2 or X_phys.shape[1] != len(design.names):
        raise ValueError(
            "Physical input array must have shape (n_rows, n_design_inputs); "
            f"got {X_phys.shape} for {len(design.names)} design inputs."
        )

    row_count = X_phys.shape[0]
    if not constraints:
        return np.ones(row_count, dtype=bool)

    mask = np.ones(row_count, dtype=bool)
    for index, constraint in enumerate(constraints):
        result = constraint(X_phys, design)
        if (
            result is None
            or not isinstance(result, np.ndarray)
            or result.dtype != bool
            or result.shape != (row_count,)
        ):
            actual = (
                None
                if result is None
                else (getattr(result, "dtype", None), getattr(result, "shape", None))
            )
            raise ValueError(
                f"Constraint #{index} must return a boolean mask of shape "
                f"({row_count},); got {actual}."
            )
        mask &= result
    return mask


__all__ = [
    "NamedConstraint",
    "RowConstraint",
    "apply_row_constraints",
    "check_clausius_clapeyron_np",
    "constraint_violations",
    "constraints_from_config",
]
