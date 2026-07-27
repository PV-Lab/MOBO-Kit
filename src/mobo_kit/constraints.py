"""Opt-in physical-space constraints for campaign designs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .design import Design


RowConstraint = Callable[[np.ndarray, "Design"], np.ndarray]


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


_SUPPORTED_BOOL_KEYS = {
    "clausius_clapeyron": _build_clausius_clapeyron,
}


def constraints_from_config(cfg: Dict, design: "Design") -> List[RowConstraint]:
    """Build only explicitly configured campaign constraints.

    A supported entry has one boolean type flag and the parameters required by
    that type, for example::

        constraints:
          - clausius_clapeyron: true
            ah_col: absolute_humidity
            temp_c_col: temperature_c

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

    constraints: List[RowConstraint] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise TypeError(
                f"Constraint entry at index {index} must be a mapping; "
                f"got {type(raw).__name__}."
            )

        known_keys = [key for key in _SUPPORTED_BOOL_KEYS if key in raw]
        if not known_keys:
            raise KeyError(
                f"Constraint entry at index {index} has no supported type; "
                f"expected one of {sorted(_SUPPORTED_BOOL_KEYS)}."
            )
        if len(known_keys) > 1:
            raise ValueError(
                f"Constraint entry at index {index} enables multiple types: "
                f"{known_keys}. Use one constraint type per entry."
            )

        chosen_key = known_keys[0]
        enabled = raw[chosen_key]
        if not isinstance(enabled, bool):
            raise TypeError(
                f"Constraint flag '{chosen_key}' must be true or false; "
                f"got {enabled!r}."
            )
        if not enabled:
            continue

        constraints.append(_SUPPORTED_BOOL_KEYS[chosen_key](raw, design))

    return constraints


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
    "RowConstraint",
    "apply_row_constraints",
    "check_clausius_clapeyron_np",
    "constraints_from_config",
]
