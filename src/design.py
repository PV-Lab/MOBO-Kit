# src/design.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

def make_linspace(start: float, stop: float, step: float, decimals: int = 6) -> np.ndarray:
    num_points = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, num_points), decimals)

@dataclass
class InputSpec:
    name: str
    label: Optional[str] = None
    unit: Optional[str] = None
    kind: str = "range"            # "range" or "grid"
    # range
    lower: Optional[float] = None
    upper: Optional[float] = None
    # grid
    values: Optional[List[float]] = None
    start: Optional[float] = None
    stop: Optional[float] = None
    step: Optional[float] = None
    decimals: int = 6

def build_input_spec_list(cfg_inputs: List[Dict[str, Any]]) -> List[InputSpec]:
    return [InputSpec(**item) for item in cfg_inputs]

@dataclass
class Design:
    names: List[str]
    labels: List[str]
    units: List[Optional[str]]
    lowers: np.ndarray               # (D,)
    uppers: np.ndarray               # (D,)
    var_array: List[np.ndarray]      # for min/max per feature
    var_list: List[Optional[np.ndarray]]  # grid per feature or None

def build_design(specs: List[InputSpec]) -> Design:
    names, labels, units = [], [], []
    lowers, uppers = [], []
    var_array: List[np.ndarray] = []
    var_list: List[Optional[np.ndarray]] = []

    for s in specs:
        names.append(s.name)
        labels.append(s.label if s.label else s.name)
        units.append(s.unit)

        if s.kind == "grid":
            if s.values is None:
                if s.start is None or s.stop is None or s.step is None:
                    raise ValueError(f"Grid input '{s.name}' requires 'values' or (start, stop, step).")
                grid = make_linspace(s.start, s.stop, s.step, s.decimals)
            else:
                grid = np.array(sorted(set(s.values)), dtype=float)
            var_list.append(grid)
            lowers.append(float(grid.min()))
            uppers.append(float(grid.max()))
            var_array.append(grid)
        elif s.kind == "range":
            if s.lower is None or s.upper is None:
                raise ValueError(f"Range input '{s.name}' requires 'lower' and 'upper'.")
            lowers.append(float(s.lower))
            uppers.append(float(s.upper))
            var_list.append(None)
            var_array.append(np.array([s.lower, s.upper], dtype=float))
        else:
            raise ValueError(f"Unknown kind '{s.kind}' for input '{s.name}'.")

    return Design(
        names=names,
        labels=labels,
        units=units,
        lowers=np.asarray(lowers, dtype=float),
        uppers=np.asarray(uppers, dtype=float),
        var_array=var_array,
        var_list=var_list,
    )
