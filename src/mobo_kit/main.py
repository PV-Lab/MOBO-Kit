"""
MOBO-Kit: Multi-objective Bayesian Optimization Toolkit

Main entry point for the MOBO-Kit package.
Provides a simple Python API for running MOBO experiments.
"""

import os
import sys
from typing import Optional, Dict, Any, Sequence
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

from .utils import (
    parse_campaign_csv,
    split_XY,
    set_seeds,
    select_device,
    get_objective_names,
)
from .design import Design, build_design_from_config
from .data import x_normalizer_np
from .models import fit_gp_models, posterior_report
from .plotting import plot_parity_np
from .acquisition import propose_batch
from .metrics import compute_ref_pareto_hv
from .constraints import constraints_from_config
from .lhs import lhs_dataframe_optimized
from .production_gate import (
    ProductionApprovalError,
    block_legacy_campaign_proposal,
)


def generate_initial_experiments(
    config_path: str,
    n_samples: int,
    save_path: str,
    seed: int = 42,
    verbose: bool = True,
    max_abs_corr: Optional[float] = None,
    max_attempts: int = 100,
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
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build design space through the validated config-to-design path.
    space = build_design_from_config(config)

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
        verbose=verbose,
    )

    # Save to CSV
    output_parent = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(output_parent, exist_ok=True)
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
        "constraints_applied": len(row_constraints) > 0 if row_constraints else False,
    }


def _validate_candidate_batch(
    batch_result: Dict[str, Any],
    design: Design,
    observed_inputs: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Fail closed on incomplete, duplicate, observed, or off-grid batches."""

    if not isinstance(batch_result, dict) or "X_phys" not in batch_result:
        raise ValueError("Candidate proposal did not return an 'X_phys' array.")
    try:
        candidates = np.asarray(batch_result["X_phys"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate physical inputs must be numeric.") from exc

    expected_shape = (batch_size, len(design.names))
    if candidates.shape != expected_shape:
        raise ValueError(
            "Candidate proposal returned an incomplete or malformed batch: "
            f"expected shape {expected_shape}, got {candidates.shape}."
        )
    if not np.isfinite(candidates).all():
        raise ValueError("Candidate proposal contains non-finite physical inputs.")

    tolerances = np.maximum(np.abs(design.steps) * 1e-9, 1e-10)
    grid_indices = np.empty(expected_shape, dtype=np.int64)
    for column_index, (name, grid) in enumerate(zip(design.names, design.var_list)):
        distances = np.abs(candidates[:, column_index, None] - grid[None, :])
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[
            np.arange(batch_size, dtype=np.int64), nearest_indices
        ]
        if np.any(nearest_distances > tolerances[column_index]):
            bad_rows = np.flatnonzero(
                nearest_distances > tolerances[column_index]
            ).tolist()
            raise ValueError(
                f"Candidate input '{name}' is off-grid at batch rows {bad_rows}."
            )
        grid_indices[:, column_index] = nearest_indices

    if np.unique(grid_indices, axis=0).shape[0] != batch_size:
        raise ValueError("Candidate proposal contains duplicate snapped recipes.")

    observed = observed_inputs.loc[:, design.names].to_numpy(dtype=float)
    if observed.size:
        matches_observed = np.all(
            np.isclose(
                candidates[:, None, :],
                observed[None, :, :],
                rtol=0.0,
                atol=tolerances,
            ),
            axis=2,
        )
        repeated_rows = np.flatnonzero(matches_observed.any(axis=1)).tolist()
        if repeated_rows:
            raise ValueError(
                "Candidate proposal repeats observed recipes at batch rows "
                f"{repeated_rows}."
            )

    return candidates


def run_mobo_experiment(
    csv_path: str,
    save_dir: str = "local_outputs/experiment",
    config_path: Optional[str] = None,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
    batch_size: int = 5,
    propose_candidates: bool = False,
    reference_point: Optional[Sequence[float]] = None,
    num_restarts: int = 20,
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
        propose_candidates: Request campaign candidates. Step 2A always blocks
            this legacy path; a reviewed Step 2B campaign adapter is required.
        reference_point: Explicit hypervolume reference point in the exact same
            transformed objective space as ``Y``. Required when proposing.
        num_restarts: Number of acquisition-optimization restarts.

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

    # Load the explicit configuration first when supplied, then parse the CSV
    # exactly once against the expected objective names.
    if config_path is not None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if propose_candidates:
            # This executes before campaign CSV parsing, model fitting, or any
            # output-directory creation. Unapproved configs receive complete
            # gate errors; approved configs still cannot enter the legacy
            # raw-objective proposal implementation.
            block_legacy_campaign_proposal(config)
        parsed_csv = parse_campaign_csv(
            csv_path, expected_objectives=get_objective_names(config)
        )
        if verbose:
            print(f"Loaded config from: {config_path}")
    else:
        if propose_candidates:
            raise ProductionApprovalError(
                [
                    "Campaign candidate proposal requires an explicit resolved "
                    "production configuration; CSV auto-configuration is not "
                    "scientific approval."
                ]
            )
        parsed_csv = parse_campaign_csv(csv_path)
        config = parsed_csv.config
        if verbose:
            print("Auto-generated config from CSV metadata")

    df = parsed_csv.data

    # Build the design through the same validated path used by LHS generation.
    space = build_design_from_config(config)

    # Split data into inputs and objectives
    X, Y = split_XY(df, space, config)

    if propose_candidates and reference_point is None:
        raise ValueError(
            "Candidate proposal requires an explicit reference_point in the "
            "same transformed objective space as the model outputs. MOBO-Kit "
            "will not invent a campaign reference point."
        )

    if verbose:
        print(
            f"Loaded {len(X)} samples with {X.shape[1]} inputs and {Y.shape[1]} objectives"
        )
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

        # The runner writes diagnostics to disk and must also work on lab PCs,
        # CI workers, and managed Python installs without a Tcl/Tk GUI runtime.
        plt.switch_backend("Agg")

        # Create parity plots
        parity_path = os.path.join(save_dir, "parity_plots.png")
        fig, metrics_df = plot_parity_np(
            true_Y=Y.values,
            pred_mean=pred_mean,
            pred_std=pred_std,
            objective_names=get_objective_names(config),
            save=parity_path,
            show_plot=False,
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

            ref_point_np = np.asarray(reference_point, dtype=float)
            if ref_point_np.shape != (Y.shape[1],):
                raise ValueError(
                    "reference_point must contain exactly one value per objective "
                    f"({Y.shape[1]} expected, got shape {ref_point_np.shape})."
                )

            # Compute reference point and hypervolume
            ref_point_t, pareto_Y_t, hv_val = compute_ref_pareto_hv(
                Y_t, ref_point_np=ref_point_np
            )

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
                num_restarts=num_restarts,
                verbose=verbose,
            )

            candidate_array = _validate_candidate_batch(
                batch_result=batch_result,
                design=space,
                observed_inputs=X,
                batch_size=batch_size,
            )
            batch_result = dict(batch_result)
            batch_result["X_phys"] = candidate_array

            candidates = batch_result
            candidates_path = os.path.join(save_dir, "next_batch.csv")

            # Retain the legacy flat-table export for the explicit qNEHVI path.
            # It is not a metadata-style campaign CSV and therefore is not a
            # supported round-trip input to parse_campaign_csv.
            new_candidates_df = pd.DataFrame(candidate_array, columns=space.names)

            # Add empty objective columns to match original format
            objective_names = get_objective_names(config)
            for obj_name in objective_names:
                new_candidates_df[obj_name] = ""

            # Combine original data with new candidates
            combined_df = pd.concat([df, new_candidates_df], ignore_index=True)

            # Save the combined CSV
            combined_df.to_csv(candidates_path, index=False)

            if verbose:
                print(f"Proposed {batch_size} candidates for next batch")
                print(f"Candidates saved to: {candidates_path}")

        except Exception as e:
            if verbose:
                print(f"Candidate proposal failed: {e}")
            raise RuntimeError(
                "Candidate proposal failed; no batch was accepted."
            ) from e

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
        "candidates": candidates,
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
    save_dir = "local_outputs/demo"

    # Check if default CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Default data file not found: {csv_path}")
        print("Please provide a valid CSV file path or ensure the default file exists.")
        sys.exit(1)

    try:
        results = run_mobo_experiment(
            csv_path=csv_path, save_dir=save_dir, verbose=True
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
