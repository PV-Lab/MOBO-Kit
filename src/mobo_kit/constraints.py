# src/constraints.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

# Public type: row-wise constraint (PHYSICAL units in, boolean mask out)
RowConstraint = Callable[[np.ndarray, "Design"], np.ndarray]


# =========================
# Clausius–Clapeyron (C → K)
# =========================

def check_clausius_clapeyron_np(ah_vals, temp_c_vals) -> np.ndarray:
    """
    Return boolean mask for valid points satisfying Clausius–Clapeyron constraint.
    Ensures absolute humidity (g/m^3) does not exceed saturation at given temperature.

    Parameters
    ----------
    ah_vals     : array-like, absolute humidity [g/m^3]
    temp_c_vals : array-like, temperature [°C]  (converted to K internally)

    Returns
    -------
    valid_mask : np.ndarray[bool]
    """
    AH = np.asarray(ah_vals, dtype=float)
    T_c = np.asarray(temp_c_vals, dtype=float)   # Celsius in
    T   = T_c + 273.15                           # Kelvin

    # Saturation vapor pressure (kPa)
    es = 0.6113 * np.exp((17.27 * (T - 273.15)) / (T - 35.86))

    # Max absolute humidity (g/m^3)
    AH_max = es / (4.61e-4 * T)

    return (AH <= AH_max) & np.isfinite(AH_max)


# =========================
# Builders (one per supported constraint key)
# =========================

def _idx_for(name: str, design: "Design") -> int:
    try:
        return design.names.index(name)
    except ValueError as e:
        raise KeyError(
            f"Constraint refers to column '{name}', but it is not in design.names={design.names}"
        ) from e

def _build_cc(spec: Dict, design: "Design") -> RowConstraint:

    ah_col = spec.get("ah_col")
    t_col  = spec.get("temp_c_col")
    if ah_col is None or t_col is None:
        raise KeyError(
            "Clausius–Clapeyron constraint needs 'absolute_humidity_col' and 'temperature_col'."
        )

    i_ah = _idx_for(ah_col, design)
    i_tC = _idx_for(t_col,  design)

    def row_constraint_fn(X: np.ndarray, _design: "Design") -> np.ndarray:
        return check_clausius_clapeyron_np(X[:, i_ah], X[:, i_tC])

    return row_constraint_fn


# Map of **boolean keys** in YAML → builder
_SUPPORTED_BOOL_KEYS = {
    "clausius_clapeyron": _build_cc,
    # add more: "my_constraint_key": _build_my_constraint
}


# =========================
# Config parsing & application
# =========================

def constraints_from_config(cfg: Dict, design: "Design") -> List[RowConstraint]:
    """
    Parse a YAML layout like:

    constraints:
      - clausius_clapeyron: true
        absolute_humidity_col: "absolute_humidity"
        temperature_col: "temperature_c"

      - clausius_clapeyron: false   # ignored
        ...

    Returns a list of row-constraint callables (possibly empty).
    """
    items = cfg.get("constraints", [])
    if not items:
        return []

    fns: List[RowConstraint] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue

        # find which supported boolean key (if any) is enabled
        chosen_key = None
        for key, builder in _SUPPORTED_BOOL_KEYS.items():
            val = raw.get(key, None)
            if isinstance(val, bool) and val:
                chosen_key = key
                break

        if chosen_key is None:
            # no supported boolean flag set to true -> skip this entry
            continue

        # build constraint from the same dict (which also holds column names, etc.)
        builder = _SUPPORTED_BOOL_KEYS[chosen_key]
        fns.append(builder(raw, design))

    return fns


def apply_row_constraints(
    X_phys: np.ndarray,
    design: "Design",
    constraints: Optional[Sequence[RowConstraint]],
) -> np.ndarray:
    """
    Apply zero or more row-wise constraints; returns a boolean mask AND-ing all.
    If constraints is None or empty, returns all True.
    """
    n = X_phys.shape[0]
    if not constraints:
        return np.ones(n, dtype=bool)

    mask = np.ones(n, dtype=bool)
    for k, fn in enumerate(constraints):
        m = fn(X_phys, design)
        if m is None or m.dtype != bool or m.shape != (n,):
            raise ValueError(
                f"Constraint #{k} must return a boolean mask of shape ({n},); "
                f"got {None if m is None else (m.dtype, m.shape)}"
            )
        mask &= m
    return mask
