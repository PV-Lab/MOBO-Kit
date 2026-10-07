# src/design.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


def _finite_float(value: Any, *, field: str, input_name: str) -> float:
    """Return ``value`` as a finite float with a campaign-friendly error."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"Input '{input_name}' field '{field}' must be a finite number, not bool."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Input '{input_name}' field '{field}' must be a finite number; "
            f"got {value!r}."
        ) from exc
    if not np.isfinite(number):
        raise ValueError(
            f"Input '{input_name}' field '{field}' must be finite; got {value!r}."
        )
    return number


def _validate_decimals(value: Any, *, input_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"Input '{input_name}' field 'decimals' must be a non-negative integer; "
            f"got {value!r}."
        )
    decimals = int(value)
    if decimals < 0:
        raise ValueError(
            f"Input '{input_name}' field 'decimals' must be a non-negative integer; "
            f"got {decimals}."
        )
    return decimals


def make_linspace(
    start: float,
    stop: float,
    step: float,
    decimals: int = 6,
) -> np.ndarray:
    """Create an endpoint-aligned grid with the requested step.

    ``numpy.linspace`` can silently change the requested spacing when the endpoint
    is not aligned. Campaign inputs must instead satisfy
    ``stop == start + k * step`` (within floating-point tolerance).
    """
    input_name = "<grid>"
    start_f = _finite_float(start, field="start", input_name=input_name)
    stop_f = _finite_float(stop, field="stop", input_name=input_name)
    step_f = _finite_float(step, field="step", input_name=input_name)
    decimals_i = _validate_decimals(decimals, input_name=input_name)

    if step_f <= 0:
        raise ValueError(f"Grid step must be > 0; got {step_f}.")
    if stop_f < start_f:
        raise ValueError(
            f"Grid stop must be greater than or equal to start; "
            f"got start={start_f}, stop={stop_f}."
        )

    span = stop_f - start_f
    if span == 0:
        return np.asarray([round(start_f, decimals_i)], dtype=float)

    interval_count = int(round(span / step_f))
    if interval_count < 1:
        raise ValueError(
            "Grid endpoint is not aligned with step: "
            f"start={start_f}, stop={stop_f}, step={step_f}. "
            "Require stop = start + k * step for an integer k."
        )
    aligned_span = interval_count * step_f
    alignment_atol = max(1e-12, abs(step_f) * 1e-9)
    if not np.isclose(span, aligned_span, rtol=0.0, atol=alignment_atol):
        raise ValueError(
            "Grid endpoint is not aligned with step: "
            f"start={start_f}, stop={stop_f}, step={step_f}. "
            "Require stop = start + k * step for an integer k."
        )

    grid = np.round(
        start_f + step_f * np.arange(interval_count + 1, dtype=float),
        decimals_i,
    )
    grid[0] = round(start_f, decimals_i)
    grid[-1] = round(stop_f, decimals_i)

    if np.unique(grid).size != grid.size:
        raise ValueError(
            f"Rounding the grid to {decimals_i} decimals creates duplicate values. "
            "Increase 'decimals' or use a larger step."
        )
    return grid


@dataclass
class InputSpec:
    """Specification for an input parameter with grid-based discretization."""

    name: str
    start: float
    stop: float
    step: float
    unit: Optional[str] = None
    decimals: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Input name must be a non-empty string.")
        self.name = self.name.strip()
        self.start = _finite_float(self.start, field="start", input_name=self.name)
        self.stop = _finite_float(self.stop, field="stop", input_name=self.name)
        self.step = _finite_float(self.step, field="step", input_name=self.name)
        self.decimals = _validate_decimals(self.decimals, input_name=self.name)
        if self.step <= 0:
            raise ValueError(
                f"Input '{self.name}' field 'step' must be > 0; got {self.step}."
            )
        if self.stop < self.start:
            raise ValueError(
                f"Input '{self.name}' requires stop >= start; "
                f"got start={self.start}, stop={self.stop}."
            )
        # Constructing the grid here validates endpoint alignment and precision.
        try:
            make_linspace(self.start, self.stop, self.step, self.decimals)
        except ValueError as exc:
            raise ValueError(f"Invalid grid for input '{self.name}': {exc}") from exc


def build_input_spec_list(cfg_inputs: List[Dict[str, Any]]) -> List[InputSpec]:
    """Build InputSpec objects from config dictionary."""
    if not isinstance(cfg_inputs, list) or not cfg_inputs:
        raise ValueError("Config 'inputs' must be a non-empty list.")

    specs = []
    seen_names = set()
    for index, item in enumerate(cfg_inputs):
        if not isinstance(item, dict):
            raise ValueError(
                f"Config input at index {index} must be a mapping; "
                f"got {type(item).__name__}."
            )
        missing = [key for key in ("name", "start", "stop", "step") if key not in item]
        if missing:
            display_name = item.get("name", f"index {index}")
            raise ValueError(
                f"Input '{display_name}' is missing required field(s): "
                f"{', '.join(missing)}."
            )

        spec = InputSpec(
            name=item["name"],
            unit=item.get("unit"),
            start=item["start"],
            stop=item["stop"],
            step=item["step"],
            decimals=item.get("decimals", 6),
        )
        if spec.name in seen_names:
            raise ValueError(
                f"Input names must be unique; duplicate name '{spec.name}'."
            )
        seen_names.add(spec.name)
        specs.append(spec)

    return specs


@dataclass
class Design:
    """Design space specification with grid-based discretization."""

    names: List[str]
    units: List[Optional[str]]
    lowers: np.ndarray  # (D,) minimum values
    uppers: np.ndarray  # (D,) maximum values
    steps: np.ndarray  # (D,) step sizes
    var_array: List[np.ndarray]  # grid values for each feature
    # Grid values for each feature (same as var_array for compatibility).
    var_list: List[np.ndarray]


def build_design(specs: List[InputSpec]) -> Design:
    """Build a Design object from InputSpec objects."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("At least one InputSpec is required to build a design.")

    names, units = [], []
    lowers, uppers, steps = [], [], []
    var_array: List[np.ndarray] = []
    seen_names = set()

    for index, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, InputSpec):
            raise TypeError(
                f"Design item at index {index} must be an InputSpec; "
                f"got {type(raw_spec).__name__}."
            )
        # InputSpec is mutable, so revalidate a fresh copy at the boundary.
        spec = InputSpec(
            name=raw_spec.name,
            start=raw_spec.start,
            stop=raw_spec.stop,
            step=raw_spec.step,
            unit=raw_spec.unit,
            decimals=raw_spec.decimals,
        )
        if spec.name in seen_names:
            raise ValueError(
                f"Input names must be unique; duplicate name '{spec.name}'."
            )
        seen_names.add(spec.name)
        names.append(spec.name)
        units.append(spec.unit)

        # Create grid for this parameter
        grid = make_linspace(spec.start, spec.stop, spec.step, spec.decimals)
        var_array.append(grid)

        # Store bounds and step
        lowers.append(float(grid.min()))
        uppers.append(float(grid.max()))
        steps.append(float(spec.step))

    return Design(
        names=names,
        units=units,
        lowers=np.asarray(lowers, dtype=float),
        uppers=np.asarray(uppers, dtype=float),
        steps=np.asarray(steps, dtype=float),
        var_array=var_array,
        var_list=var_array,  # For backward compatibility
    )


def build_design_from_config(config: Dict[str, Any]) -> Design:
    """Build a Design object directly from a config dictionary."""
    if not isinstance(config, dict):
        raise ValueError(
            "Config must be a mapping containing a non-empty 'inputs' list."
        )
    if "inputs" not in config:
        raise ValueError("Config must contain 'inputs' key.")

    specs = build_input_spec_list(config["inputs"])
    return build_design(specs)


def get_variable_space() -> List[np.ndarray]:
    """Get the variable space as a list of arrays (for backward compatibility)."""
    # This function would need a config to work with the new system
    # For now, return empty list - users should use build_design_from_config instead
    return []


def get_parameter_space():
    """Get the parameter space (for backward compatibility)."""
    # This function would need a config to work with the new system
    # For now, return None - users should use build_design_from_config instead
    return None


def generate_initial_design(
    n_samples: int,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Generate initial design using Latin Hypercube Sampling.

    Args:
        n_samples: Number of samples to generate
        config: Optional config dictionary. If provided, uses the new design system.

    Returns:
        Array of shape (n_samples, n_features)
    """
    if config is not None:
        # Use new design system
        design = build_design_from_config(config)
        # This would integrate with the LHS module
        # For now, return random samples in the bounds
        samples = np.random.uniform(
            low=design.lowers, high=design.uppers, size=(n_samples, len(design.names))
        )
        return samples
    else:
        # Fallback to old system (deprecated)
        # This maintains backward compatibility but should be avoided
        return np.random.uniform(0, 1, size=(n_samples, 8))
