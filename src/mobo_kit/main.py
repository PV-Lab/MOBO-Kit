"""
MOBO-Kit: Multi-objective Bayesian Optimization Toolkit

Main entry point for the MOBO-Kit package.
Provides a simple Python API for running MOBO experiments.
"""

import os
import sys
from typing import Optional, Dict, Any
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

from .utils import (
    load_csv, split_XY, csv_to_config, set_seeds, select_device,
    get_objective_names
)
from .design import build_input_spec_list, build_design
from .data import x_normalizer_np
from .models import fit_gp_models, posterior_report
from .plotting import plot_parity_np
from .acquisition import propose_batch
from .metrics import compute_ref_pareto_hv
from .constraints import constraints_from_config
from .lhs import lhs_dataframe_optimized


def generate_initial_experiments(
    config_path: str,
    n_samples: int,
    save_path: str,
    seed: int = 42,
    verbose: bool = True,
    max_abs_corr: Optional[float] = None,
    max_attempts: int = 100
) -> Dict[str, Any]:
    """
    Generate initial experiments using Latin Hypercube Sampling.
    
    This function is useful when you have a design space but no existing data.
    It generates a CSV file with initial experiments to run.
    
    Args:
        config_path: Path to YAML configuration file
        n_samples: Number of initial experiments to generate
        save_path: Path to save the generated CSV file
        seed: Random seed for reproducibility
        verbose: Whether to print progress information
        max_abs_corr: Maximum absolute correlation between variables (optional)
        max_attempts: Maximum attempts for LHS generation
        
    Returns:
        Dictionary with generation results and metadata
    """
    if verbose:
        print("MOBO-Kit: Initial Experiment Generation")
        print("=" * 50)
        print(f"Generating {n_samples} initial experiments...")
        print(f"Config: {config_path}")
        print(f"Output: {save_path}")
    
    # Set random seed
    set_seeds(seed)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build design space
    space = build_design(config)
    
    if verbose:
        print(f"Design space: {len(space.names)} variables")
        print(f"Variables: {', '.join(space.names)}")
    
    # Get constraints
    row_constraints = constraints_from_config(config, space)
    
    # Generate LHS samples
    if verbose:
        print("Generating Latin Hypercube samples...")
    
    lhs_df = lhs_dataframe_optimized(
        design=space,
        n=n_samples,
        seed=seed,
        snap_to_grids=True,
        row_constraints=row_constraints,
        max_abs_corr=max_abs_corr,
        max_attempts=max_attempts,
        verbose=verbose
    )
    
    # Save to CSV
    lhs_df.to_csv(save_path, index=False)
    
    if verbose:
        print(f"Generated {len(lhs_df)} initial experiments")
        print(f"Saved to: {save_path}")
        print("=" * 50)
    
    return {
        "status": "success",
        "n_samples": len(lhs_df),
        "config_path": config_path,
        "save_path": save_path,
        "seed": seed,
        "variables": space.names,
        "constraints_applied": len(row_constraints) > 0 if row_constraints else False
    }


def run_mobo_experiment(
    csv_path: str,
    save_dir: str = "results/experiment",
    config_path: Optional[str] = None,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
    batch_size: int = 5,
    propose_candidates: bool = True
) -> Dict[str, Any]:
    """
    Run a complete MOBO experiment from CSV data.
    
    Args:
        csv_path: Path to CSV file with experimental data
        save_dir: Directory to save results
        config_path: Optional path to YAML config file (if None, auto-generates from CSV)
        seed: Random seed for reproducibility
        device: Device to use ("auto", "cpu", "cuda")
        verbose: Whether to print progress information
        batch_size: Number of candidates to propose for next batch
        propose_candidates: Whether to propose new candidates for next experiments
        
    Returns:
        Dictionary with experiment results and metadata
    """
    if verbose:
        print("MOBO-Kit: Multi-objective Bayesian Optimization Toolkit")
        print("=" * 60)
    
    # Set random seeds
    set_seeds(seed)
    
    # Select device
    if device == "auto":
        device_obj = select_device("cuda")
    else:
        device_obj = select_device(device)
    
    if verbose:
        print(f"Using device: {device_obj}")
        print(f"Loading data from: {csv_path}")
    
    # Load and process data
    df = load_csv(csv_path)
    
    # Load or generate configuration
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if verbose:
            print(f"Loaded config from: {config_path}")
    else:
        config = csv_to_config(csv_path)
        if verbose:
            print("Auto-generated config from CSV metadata")
    
    # Build design space
    specs = build_input_spec_list(config["inputs"])
    space = build_design(specs)
    
    # Split data into inputs and objectives
    X, Y = split_XY(df, space, config)
    
    if verbose:
        print(f"Loaded {len(X)} samples with {X.shape[1]} inputs and {Y.shape[1]} objectives")
        print(f"Objective names: {get_objective_names(config)}")
    
    # Normalize inputs to [0,1] range
    X_norm = x_normalizer_np(X, space)
    
    # Convert to torch tensors (Y is used directly, not standardized)
    X_t = torch.tensor(X_norm, dtype=torch.float64, device=device_obj)
    Y_t = torch.tensor(Y.values, dtype=torch.float64, device=device_obj)
    
    if verbose:
        print("Fitting Gaussian Process models...")
    
    # Fit GP models
    model = fit_gp_models(X_t, Y_t)
    
    if verbose:
        print("Generating predictions...")
    
    # Generate predictions (already in original units due to internal standardization)
    pred_mean, pred_std = posterior_report(model, X_t)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Predictions are generated but not saved to separate CSV
    
    # Generate plots
    try:
        if verbose:
            print("Generating plots...")
        
        # Create parity plots
        parity_path = os.path.join(save_dir, "parity_plots.png")
        fig, metrics_df = plot_parity_np(
            true_Y=Y.values,
            pred_mean=pred_mean,
            pred_std=pred_std,
            objective_names=get_objective_names(config),
            save=parity_path,
            show_plot=False
        )
        plt.close(fig)  # Close the figure to free memory
        
        if verbose:
            print(f"Parity plots saved to: {parity_path}")
            
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate plots: {e}")
    
    # Propose new candidates for next batch
    candidates = None
    if propose_candidates:
        try:
            if verbose:
                print("Proposing new candidates for next batch...")
            
            # Compute reference point and hypervolume
            _, pareto_Y_t, hv_val = compute_ref_pareto_hv(Y_t)
            ref_point_t = torch.tensor([-0.01] * Y.shape[1], dtype=X_t.dtype, device=device_obj)
            
            if verbose:
                print(f"Current hypervolume: {hv_val:.4f}")
                print(f"Pareto points: {pareto_Y_t.shape[0]}")
            
            # Get constraints
            row_constraints = constraints_from_config(config, space)
            
            # Propose batch
            batch_result = propose_batch(
                design=space,
                model=model,
                train_X=X_t,
                ref_point_t=ref_point_t,
                batch_size=batch_size,
                row_constraints=row_constraints,
                verbose=verbose
            )
            
            candidates = batch_result
            candidates_path = os.path.join(save_dir, "next_batch.csv")
            
            # Create next_batch.csv in the same format as input CSV with metadata
            # Load the original CSV to get metadata structure
            original_df = load_csv(csv_path)
            
            # Create new candidates DataFrame with same structure
            new_candidates_df = pd.DataFrame(
                batch_result['X_phys'], 
                columns=space.names
            )
            
            # Add empty objective columns to match original format
            objective_names = get_objective_names(config)
            for obj_name in objective_names:
                new_candidates_df[obj_name] = ""
            
            # Combine original data with new candidates
            combined_df = pd.concat([original_df, new_candidates_df], ignore_index=True)
            
            # Save the combined CSV
            combined_df.to_csv(candidates_path, index=False)
            
            if verbose:
                print(f"Proposed {batch_size} candidates for next batch")
                print(f"Candidates saved to: {candidates_path}")
                
        except Exception as e:
            if verbose:
                print(f"Warning: Could not propose candidates: {e}")
    
    # Prepare results summary
    results = {
        "status": "success",
        "n_samples": len(X),
        "n_inputs": X.shape[1],
        "n_objectives": Y.shape[1],
        "objective_names": get_objective_names(config),
        "device": str(device_obj),
        "seed": seed,
        "save_dir": save_dir,
        "config": config,
        "candidates": candidates
    }
    
    if verbose:
        print(f"Experiment completed successfully!")
        print(f"Results saved to: {save_dir}")
    
    return results


def main():
    """
    Main entry point for running MOBO-Kit with default settings.
    
    This function runs a complete MOBO experiment using the default CSV file
    and saves results to the default output directory.
    """
    # Default paths
    csv_path = "data/processed/configCSV_example.csv"
    save_dir = "results/demo"
    
    # Check if default CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Default data file not found: {csv_path}")
        print("Please provide a valid CSV file path or ensure the default file exists.")
        sys.exit(1)
    
    try:
        results = run_mobo_experiment(
            csv_path=csv_path,
            save_dir=save_dir,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY:")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Inputs: {results['n_inputs']}")
        print(f"  Objectives: {results['n_objectives']}")
        print(f"  Objectives: {', '.join(results['objective_names'])}")
        print(f"  Results: {results['save_dir']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running MOBO experiment: {e}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
