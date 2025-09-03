# src/utils.py
from __future__ import annotations
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import torch
import yaml

from .design import Design

def get_objective_names(cfg: dict) -> List[str]:
    names = cfg.get("objectives", {}).get("names", [])
    if not names:
        raise ValueError("Config must have objectives.names as a non-empty list.")
    return list(names)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_XY(df: pd.DataFrame, design: Design, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a DataFrame into input features (X) and objectives (Y) using the new config/design structure.
    
    This function works with CSV files like configCSV_example.csv that contain:
    - Rows 0-4: Configuration metadata (column names, units, start, stop, step)
    - Row 5: Empty row
    - Rows 6+: Experimental data
    
    Args:
        df: DataFrame containing the data (including metadata rows)
        design: Design object with input parameter names
        config: Configuration dictionary with objectives.names
        
    Returns:
        Tuple of (X, Y) arrays where:
        - X: (N, D) array of input features
        - Y: (N, M) array of objectives
        
    Raises:
        KeyError: If required columns are missing from the DataFrame
    """
    # Get input column names from design
    x_cols = list(design.names)
    
    # Get objective column names from config
    y_cols = get_objective_names(config)
    
    # Check for missing columns
    miss_x = [c for c in x_cols if c not in df.columns]
    miss_y = [c for c in y_cols if c not in df.columns]
    
    if miss_x or miss_y:
        parts = []
        if miss_x: 
            parts.append(f"missing inputs: {miss_x}")
        if miss_y: 
            parts.append(f"missing objectives: {miss_y}")
        raise KeyError("CSV column check failed: " + "; ".join(parts))
    
    # Extract data rows (skip the first 6 rows: metadata + empty row)
    data_df = df.iloc[6:].copy()
    
    # Remove rows with all NaN values (empty rows)
    data_df = data_df.dropna(how='all')
    
    # Extract X and Y arrays
    X = data_df[x_cols].astype(float)
    Y = data_df[y_cols].astype(float)
    
    return X, Y

def select_device(prefer: str = "cuda") -> torch.device:
    return torch.device("cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu")

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def np_to_torch(
    *arrays: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_device: bool = False,
):
    """
    Convert one or more NumPy arrays to torch tensors on the chosen device.

    Usage:
        X_t = np_to_torch(X)                                       # one array -> one tensor
        X_t, Y_t = np_to_torch(X, Y)                               # many arrays -> many tensors
        (X_t, Y_t), dev = np_to_torch(X, Y, return_device=True)    # also get the device used
    """
    if device is None:
        device = select_device("cuda")  # uses your existing helper
    tensors = tuple(torch.as_tensor(a, dtype=dtype, device=device) for a in arrays)
    out = tensors[0] if len(tensors) == 1 else tensors
    return (out, device) if return_device else out


def torch_to_np(*tensors: torch.Tensor):
    """
    Convert one or more torch tensors to NumPy arrays (detached, moved to CPU).

    Usage:
        X_np = torch_to_np(X_t)
        X_np, Y_np = torch_to_np(X_t, Y_t)
    """
    arrays = tuple(t.detach().cpu().numpy() for t in tensors)
    return arrays[0] if len(arrays) == 1 else arrays


def csv_to_config(csv_path: str, output_path: str = None) -> str:
    """
    Convert a CSV configuration file to a YAML config file.
    
    Expected CSV format:
    - Row 0: Column names (input parameters + objectives)
    - Row 1: Units for each column
    - Row 2: Start values for input parameters
    - Row 3: Stop values for input parameters
    - Row 4: Step values for input parameters
    - Row 5: Empty row
    - Row 6+: Experimental data
    
    Args:
        csv_path: Path to the CSV configuration file
        output_path: Path for the output YAML file. If None, generates a default name.
        
    Returns:
        Generated config dictionary
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract metadata from first few rows
    column_names = df.columns.tolist()
    units = df.iloc[0].tolist()
    starts = df.iloc[1].tolist()
    stops = df.iloc[2].tolist()
    steps = df.iloc[3].tolist()
    
    # Identify input parameters (first 8 columns, skipping the first unnamed column) and objectives (last 3 columns)
    input_params = column_names[1:9]  # Skip first column, take next 8 columns
    objective_params = column_names[-3:]  # Last 3 columns are objectives
    
    # Build the config dictionary
    config = {
        "inputs": [],
        "objectives": {"names": objective_params},
        "constraints": [
            {
                "clausius_clapeyron": True,
                "ah_col": "absolute_humidity",
                "temp_c_col": "temperature_c"
            }
        ]
    }
    
    # Add input parameters
    for i, param in enumerate(input_params):
        # Skip empty column names
        if pd.isna(param) or str(param).strip() == "":
            continue
            
        # Get corresponding metadata (adjust index to account for skipping first column)
        metadata_idx = i + 1  # +1 because we skipped the first column
        unit = units[metadata_idx] if metadata_idx < len(units) else ""
        start = starts[metadata_idx] if metadata_idx < len(starts) else 0.0
        stop = stops[metadata_idx] if metadata_idx < len(stops) else 1.0
        step = steps[metadata_idx] if metadata_idx < len(steps) else 0.01
        
        # Convert to appropriate types
        try:
            start = float(start) if pd.notna(start) else 0.0
            stop = float(stop) if pd.notna(stop) else 1.0
            step = float(step) if pd.notna(step) else 0.01
        except (ValueError, TypeError):
            # Use defaults if conversion fails
            start, stop, step = 0.0, 1.0, 0.01
                
        input_spec = {
            "name": str(param).strip(),
            "unit": unit,
            "start": start,
            "stop": stop,
            "step": step
        }
        
        config["inputs"].append(input_spec)
    
    # Generate output path if not provided
    if output_path is None:
        import os
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"configs/{csv_basename}_config.yaml"
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    return config