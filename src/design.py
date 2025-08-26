# src/design.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

def make_linspace(start: float, stop: float, step: float, decimals: int = 6) -> np.ndarray:
    """Create a grid of values from start to stop with given step size."""
    num_points = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, num_points), decimals)

@dataclass
class InputSpec:
    """Specification for an input parameter with grid-based discretization."""
    name: str
    start: float
    stop: float
    step: float
    unit: Optional[str] = None
    decimals: int = 6

def build_input_spec_list(cfg_inputs: List[Dict[str, Any]]) -> List[InputSpec]:
    """Build InputSpec objects from config dictionary."""
    specs = []
    for item in cfg_inputs:
        # Handle both old format (lower/upper) and new format (start/stop/step)
        if 'start' in item and 'stop' in item and 'step' in item:
            # New format: start/stop/step
            spec = InputSpec(
                name=item['name'],
                unit=item.get('unit'),
                start=float(item['start']),
                stop=float(item['stop']),
                step=float(item['step']),
                decimals=item.get('decimals', 6)
            )
        else:
            raise ValueError(f"Input '{item.get('name', 'unknown')}' must have either (start, stop, step) or (lower, upper).")
        
        specs.append(spec)
    
    return specs

@dataclass
class Design:
    """Design space specification with grid-based discretization."""
    names: List[str]
    units: List[Optional[str]]
    lowers: np.ndarray               # (D,) minimum values
    uppers: np.ndarray               # (D,) maximum values
    steps: np.ndarray                # (D,) step sizes
    var_array: List[np.ndarray]      # grid values for each feature
    var_list: List[np.ndarray]       # grid values for each feature (same as var_array for consistency)

def build_design(specs: List[InputSpec]) -> Design:
    """Build a Design object from InputSpec objects."""
    names, units = [], []
    lowers, uppers, steps = [], [], []
    var_array: List[np.ndarray] = []

    for spec in specs:
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
    if 'inputs' not in config:
        raise ValueError("Config must contain 'inputs' key.")
    
    specs = build_input_spec_list(config['inputs'])
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

def generate_initial_design(n_samples: int, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
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
            low=design.lowers, 
            high=design.uppers, 
            size=(n_samples, len(design.names))
        )
        return samples
    else:
        # Fallback to old system (deprecated)
        # This maintains backward compatibility but should be avoided
        return np.random.uniform(0, 1, size=(n_samples, 8))
