#!/usr/bin/env python3
"""
Synthetic multi-objective test problem for debugging MOBO pipeline
"""

import numpy as np
import torch
from scipy.stats import qmc
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.priors import LogNormalPrior

from src.models import fit_gp_models, posterior_report, loocv_select_models
from src.data import y_standardize_np
from src.utils import np_to_torch
from src.design import Design, InputSpec, build_design
from src.plotting import plot_parity_np, plot_shap
from src.acquisition import propose_batch
from src.metrics import compute_ref_pareto_hv

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def synthetic_objectives(X):
    """
    Two synthetic objective functions with known properties (MAXIMIZATION):
    - Obj1: Inverted quadratic with global maximum
    - Obj2: Sinusoidal with multiple local maxima
    
    Args:
        X: array of shape (N, 2) with X in [0, 1]^2
    
    Returns:
        Y: array of shape (N, 2) with objectives (higher is better)
    """
    x1, x2 = X[:, 0], X[:, 1]
    
    # Objective 1: Inverted quadratic hill (maximize)
    # Maximum at (0.3, 0.7) with value ~1.0
    obj1 = 1.0 - ((x1 - 0.3)**2 + (x2 - 0.7)**2)
    
    # Objective 2: Scaled sinusoidal (maximize)
    # Oscillates between 0 and 1, with multiple local maxima
    obj2 = 0.5 * (1 + np.sin(4 * np.pi * x1) * np.cos(3 * np.pi * x2))
    
    return np.column_stack([obj1, obj2])

def generate_training_data(
    n_points: int = 20,
    noise_std: float = 0.05,
    seed: int = 42,
    dim: int = 2,
    bounds: np.ndarray | None = None,
):
    """
    Generate training data using Latin Hypercube Sampling with controlled noise.

    Args:
        n_points: number of samples
        noise_std: std dev of Gaussian noise added to objectives
        seed: RNG seed for reproducibility
        dim: dimensionality of X
        bounds: optional (dim, 2) array of [low, high] for each dim; defaults to [0,1]^dim

    Returns:
        X_train: (n_points, dim) sampled inputs
        Y_noisy: (n_points, M) noisy objective values (same shape as Y_true)
        Y_true:  (n_points, M) true objective values
    """
    if bounds is None:
        bounds = np.tile([0.0, 1.0], (dim, 1))  # [[0,1],[0,1],...]

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X_unit = sampler.random(n=n_points)  # (n_points, dim) in [0,1]

    # Scale to bounds
    X_train = qmc.scale(X_unit, bounds[:, 0], bounds[:, 1])

    # Evaluate true objectives
    Y_true = synthetic_objectives(X_train)

    # Add Gaussian noise
    rng = np.random.default_rng(seed + 1)  # separate seed for noise
    noise = rng.normal(0.0, noise_std, size=Y_true.shape)
    Y_noisy = Y_true + noise

    return X_train, Y_noisy, Y_true

def plot_true_objectives():
    """Plot the true objective functions for reference"""
    # Create a fine grid for visualization
    n_grid = 100
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Evaluate true objectives
    Y_true = synthetic_objectives(X_grid)
    obj1_grid = Y_true[:, 0].reshape(n_grid, n_grid)
    obj2_grid = Y_true[:, 1].reshape(n_grid, n_grid)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Objective 1: Inverted Quadratic
    im1 = axes[0].contourf(X1, X2, obj1_grid, levels=20, cmap='viridis')
    axes[0].set_title('Objective 1: Inverted Quadratic Hill\n1 - ((x₁-0.3)² + (x₂-0.7)²)')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].plot(0.3, 0.7, 'r*', markersize=15, label='Global maximum')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0])
    
    # Objective 2: Sinusoidal
    im2 = axes[1].contourf(X1, X2, obj2_grid, levels=20, cmap='plasma')
    axes[1].set_title('Objective 2: Sinusoidal\n0.5×(1 + sin(4πx₁)×cos(3πx₂))')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('synthetic_true_objectives.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_gp_predictions(model, X_train, Y_train, title_suffix=""):
    """Plot GP predictions vs true functions"""
    # Create prediction grid
    n_grid = 50
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # True values
    Y_true_grid = synthetic_objectives(X_grid)
    
    # GP predictions - ensure tensor is on same device as model
    device = next(model.parameters()).device
    X_grid_t = torch.tensor(X_grid, dtype=torch.float64, device=device)
    pred_mean, pred_std = posterior_report(model, X_grid_t)
    
    # Reshape for plotting
    obj1_true = Y_true_grid[:, 0].reshape(n_grid, n_grid)
    obj1_pred = pred_mean[:, 0].reshape(n_grid, n_grid)
    obj1_std = pred_std[:, 0].reshape(n_grid, n_grid)
    
    obj2_true = Y_true_grid[:, 1].reshape(n_grid, n_grid)
    obj2_pred = pred_mean[:, 1].reshape(n_grid, n_grid)
    obj2_std = pred_std[:, 1].reshape(n_grid, n_grid)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Objective 1 row
    # True
    im1 = axes[0,0].contourf(X1, X2, obj1_true, levels=20, cmap='viridis')
    axes[0,0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], 
                     s=60, cmap='viridis', edgecolors='white', linewidth=1)
    axes[0,0].set_title('Obj1: True')
    axes[0,0].set_ylabel('x₂')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Predicted
    im2 = axes[0,1].contourf(X1, X2, obj1_pred, levels=20, cmap='viridis')
    axes[0,1].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], 
                     s=60, cmap='viridis', edgecolors='white', linewidth=1)
    axes[0,1].set_title('Obj1: GP Prediction')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Uncertainty
    im3 = axes[0,2].contourf(X1, X2, obj1_std, levels=20, cmap='Reds')
    axes[0,2].scatter(X_train[:, 0], X_train[:, 1], c='white', 
                     s=60, edgecolors='black', linewidth=1)
    axes[0,2].set_title('Obj1: GP Uncertainty')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Objective 2 row
    # True
    im4 = axes[1,0].contourf(X1, X2, obj2_true, levels=20, cmap='plasma')
    axes[1,0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 1], 
                     s=60, cmap='plasma', edgecolors='white', linewidth=1)
    axes[1,0].set_title('Obj2: True')
    axes[1,0].set_xlabel('x₁')
    axes[1,0].set_ylabel('x₂')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Predicted
    im5 = axes[1,1].contourf(X1, X2, obj2_pred, levels=20, cmap='plasma')
    axes[1,1].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 1], 
                     s=60, cmap='plasma', edgecolors='white', linewidth=1)
    axes[1,1].set_title('Obj2: GP Prediction')
    axes[1,1].set_xlabel('x₁')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Uncertainty
    im6 = axes[1,2].contourf(X1, X2, obj2_std, levels=20, cmap='Reds')
    axes[1,2].scatter(X_train[:, 0], X_train[:, 1], c='white', 
                     s=60, edgecolors='black', linewidth=1)
    axes[1,2].set_title('Obj2: GP Uncertainty')
    axes[1,2].set_xlabel('x₁')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.suptitle(f'GP Model Performance {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'synthetic_gp_predictions{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_objective_space(Y_data, labels, title="Objective Space"):
    """Plot the objective space and Pareto front"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(Y_data)))
    
    for i, (Y, label) in enumerate(zip(Y_data, labels)):
        ax.scatter(Y[:, 0], Y[:, 1], alpha=0.7, label=label, c=[colors[i]], s=60)
    
    ax.set_xlabel('Objective 1 (Inverted Quadratic)')
    ax.set_ylabel('Objective 2 (Sinusoidal)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_mobo_progression(batch_info, hypervolumes):
    """Plot the MOBO progression over iterations"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Hypervolume progression
    batches = [info['batch'] for info in batch_info]
    n_points = [info['n_points'] for info in batch_info]
    
    axes[0].plot(batches, hypervolumes, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Hypervolume')
    axes[0].set_title('Hypervolume Progression')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Number of Pareto points
    n_pareto = [info['n_pareto'] for info in batch_info]
    axes[1].plot(batches, n_pareto, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Number of Pareto Points')
    axes[1].set_title('Pareto Front Growth')
    axes[1].grid(True, alpha=0.3)
    
    # # Plot 3: Objective space evolution
    # colors = plt.cm.viridis(np.linspace(0, 1, len(batch_info)))
    # for i, (info, color) in enumerate(zip(batch_info, colors)):
    #     Y_batch = info['Y_batch']
    #     alpha = 0.3 if i < len(batch_info) - 1 else 1.0
    #     size = 20 if i < len(batch_info) - 1 else 60
    #     label = f'Batch {i} (N={info["n_points"]})'
    #     axes[2].scatter(Y_batch[:, 0], Y_batch[:, 1], 
    #                    c=[color], alpha=alpha, s=size, label=label)
    
    # axes[2].set_xlabel('Objective 1 (Inverted Quadratic)')
    # axes[2].set_ylabel('Objective 2 (Sinusoidal)')
    # axes[2].set_title('Objective Space Evolution')
    # axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mobo_progression.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def test_gp_pipeline():
    """Test the complete GP modeling pipeline on synthetic data"""
    
    print("=== SYNTHETIC MULTI-OBJECTIVE TEST ===")
    
    # 0. Show the true objective functions
    print("Plotting true objective functions...")
    plot_true_objectives()
    
    # 1. Generate training data
    X_train, Y_train, Y_true = generate_training_data(n_points=20, noise_std=0.03)
    print(f"Training data: {X_train.shape[0]} points, {X_train.shape[1]} dimensions")
    print(f"Objectives: {Y_train.shape[1]} (Inverted Quadratic, Sinusoidal)")
    print(f"Y_train range: {Y_train.min(axis=0)} to {Y_train.max(axis=0)}")
    
    # Plot objective space
    plot_objective_space([Y_train], ['Training Data'], 'Training Data in Objective Space')
    
    # 2. Standardize Y and prepare tensors
    Y_std, Y_mean, Y_scale = y_standardize_np(Y_train)
    (X_t, Y_t), device = np_to_torch(X_train, Y_train, device='cpu', return_device=True)
    print(f"Y Mean: mean={Y_std.mean(axis=0)}, std={Y_std.std(axis=0)}")
    
    # 3. Fit GP models with different configurations
    print("\n--- Testing Different GP Configurations ---")
    
    # Simple default model
    model_default = fit_gp_models(X_t, Y_t)
    print("✓ Default model fitted")
    
    # Model with noise priors
    noise_priors = [LogNormalPrior(-4.0, 0.5), LogNormalPrior(-3.5, 0.5)]
    
    # Model with different kernels per objective
    kernels = [
        lambda d: RBFKernel(ard_num_dims=d),      # Smooth for inverted quadratic
        lambda d: MaternKernel(nu=1.5, ard_num_dims=d)  # More flexible for sinusoidal
    ]
    model_mixed = fit_gp_models(X_t, Y_t, kernel_fn=kernels, noise_priors=noise_priors)
    print("✓ Model with mixed kernels and noise priors fitted")
    
    # 4. Test predictions on training data
    print("\n--- Testing Predictions ---")
    pred_mean, pred_std = posterior_report(model_mixed, X_t)
    
    print(f"Prediction shapes: mean={pred_mean.shape}, std={pred_std.shape}")
    print(f"Prediction ranges: mean={pred_mean.min(axis=0)} to {pred_mean.max(axis=0)}")
    print(f"Uncertainty ranges: std={pred_std.min(axis=0)} to {pred_std.max(axis=0)}")
    
    # Plot GP predictions vs truth
    print("Plotting GP model predictions...")
    plot_gp_predictions(model_mixed, X_train, Y_train, f"(N={len(X_train)})")
    
    # 5. Compute training fit metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2_scores = [r2_score(Y_train[:, j], pred_mean[:, j]) for j in range(2)]
    rmse_scores = [np.sqrt(mean_squared_error(Y_train[:, j], pred_mean[:, j])) for j in range(2)]
    
    print(f"\nTraining fit quality:")
    print(f"  Objective 1 (Inverted Quadratic): R²={r2_scores[0]:.3f}, RMSE={rmse_scores[0]:.4f}")
    print(f"  Objective 2 (Sinusoidal): R²={r2_scores[1]:.3f}, RMSE={rmse_scores[1]:.4f}")
    
    # 6. Test on dense grid for visualization
    print("\n--- Testing on Dense Grid ---")
    n_grid = 50
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # True objectives on grid
    Y_grid_true = synthetic_objectives(X_grid)
    
    # GP predictions on grid - ensure tensor is on same device as model
    device = next(model_mixed.parameters()).device
    X_grid_t = torch.tensor(X_grid, dtype=torch.float64, device=device)
    pred_grid_mean, pred_grid_std = posterior_report(model_mixed, X_grid_t)
    
    # Compute prediction errors
    grid_errors = np.abs(pred_grid_mean - Y_grid_true)
    mean_abs_errors = grid_errors.mean(axis=0)
    max_abs_errors = grid_errors.max(axis=0)
    
    print(f"Grid prediction errors:")
    print(f"  Objective 1: MAE={mean_abs_errors[0]:.4f}, Max={max_abs_errors[0]:.4f}")
    print(f"  Objective 2: MAE={mean_abs_errors[1]:.4f}, Max={max_abs_errors[1]:.4f}")
    
    # 7. Test LOOCV if dataset isn't too small
    # if X_train.shape[0] >= 10:
    #     print("\n--- Testing LOOCV ---")
    #     try:
    #         kernel_options = [
    #             lambda d: RBFKernel(ard_num_dims=d),
    #             lambda d: MaternKernel(nu=1.5, ard_num_dims=d)
    #         ]
    #         noise_options = [None, LogNormalPrior(-4.0, 0.5)]
            
    #         model_cv, cv_results = loocv_select_models(
    #             X_t, Y_t, 
    #             objective_names=['Inverted Quadratic', 'Sinusoidal'],
    #             matern_options=kernel_options,
    #             noise_options=noise_options
    #         )
            
    #         print("LOOCV Results:")
    #         print(cv_results)
            
    #     except Exception as e:
    #         print(f"LOOCV failed: {e}")
    
    # 8. Create parity plots
    print("\n--- Parity Plots ---")
    try:
        fig_parity, metrics_parity = plot_parity_np(
            Y_train, pred_mean, pred_std, 
            objective_names=['Inverted Quadratic', 'Sinusoidal'],
            save='synthetic_parity_plot.png'
        )
        print("Parity plot metrics:")
        print(metrics_parity)
    except Exception as e:
        print(f"Parity plot failed: {e}")
    
    print("\n=== GP MODELING TEST COMPLETED ===")
    return {
        'X_train': X_train, 'Y_train': Y_train, 'Y_true': Y_true,
        'model': model_mixed, 'Y_mean': Y_mean, 'Y_scale': Y_scale,
        'pred_mean': pred_mean, 'pred_std': pred_std,
        'r2_scores': r2_scores, 'rmse_scores': rmse_scores
    }


def test_mobo_loop(n_initial=20, n_batches=5, batch_size=10, seed=42):
    """
    Test the complete MOBO loop with maximization objectives
    
    Args:
        n_initial: Number of initial training points
        n_batches: Number of MOBO iterations
        batch_size: Number of points to propose per batch
        seed: Random seed for reproducibility
    """
    print("\n=== MOBO LOOP TEST (MAXIMIZATION) ===")
    
    # Set up design space
    input_specs = [
        InputSpec(name='x1', unit=None, start=0.0, stop=1.0, step=0.01, decimals=2),
        InputSpec(name='x2', unit=None, start=0.0, stop=1.0, step=0.01, decimals=2)
    ]
    design = build_design(input_specs)
    print(f"Design space: {len(input_specs)} dimensions")
    
    # Generate initial training data
    np.random.seed(seed)
    X_all, Y_all, _ = generate_training_data(n_points=n_initial, noise_std=0.02, seed=seed)
    
    print(f"Initial training data: {n_initial} points")
    print(f"Y_initial range: {Y_all.min(axis=0)} to {Y_all.max(axis=0)}")
    
    # Track MOBO progression
    batch_info = []
    hypervolumes = []
    
    # Reference point will be computed automatically by compute_ref_pareto_hv
    ref_point_np = np.array([0.02-1e-2, -1e-2])  # Will be set after first hypervolume computation
    ref_point_t = None  # Will be set during first hypervolume computation
    
    for batch in range(n_batches + 1):  # +1 to include initial evaluation
        print(f"\n--- Batch {batch} (N={len(X_all)}) ---")
        
        # Standardize and prepare data
        #Y_std, Y_mean, Y_scale = y_standardize_np(Y_all)
        (X_t, Y_t), device = np_to_torch(X_all, Y_all, device='cuda', return_device=True)
        
        # Fit GP models
        try:
            # Use different kernels for different objectives
            kernels = [
                lambda d: RBFKernel(ard_num_dims=d),       # Smooth for inverted quadratic
                lambda d: PeriodicKernel(ard_num_dims=d)  # Flexible for sinusoidal
            ]
            noise_priors = None #[LogNormalPrior(-4.0, 0.5), LogNormalPrior(-3.5, 0.5)]
            
            model = fit_gp_models(X_t, Y_t)#, kernel_fn=kernels, noise_priors=noise_priors)
            plot_gp_predictions(model, X_all, Y_all, f"(N={len(X_all)})")
            #plot_shap(design, X_all, model)
            print("✓ GP models fitted a")
            for i, gp in enumerate(model.models):    
                print("Lengthscales:", gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten())
                print("Outputscale:", gp.covar_module.outputscale.item())
                print("Noise:", gp.likelihood.noise.item())
            
        except Exception as e:
            print(f"⚠️ GP fitting failed: {e}")
            # Fall back to default model
            model = fit_gp_models(X_t, Y_t)
            plot_gp_predictions(model, X_all, Y_all, f"(N={len(X_all)})")
            print("✓ Fallback to default GP model")
        
        # Compute hypervolume
        try:
            # Convert to tensor for compute_ref_pareto_hv - use same device as model
            Y_tensor = torch.tensor(Y_all, dtype=torch.float64, device=device)
            ref_point_t, pareto_Y, hv = compute_ref_pareto_hv(Y_tensor, ref_point_np)
            hypervolumes.append(hv)
            print(f"Hypervolume: {hv:.6f}")
            
            # Pareto front info from the function
            n_pareto = len(pareto_Y)
            Y_pareto_np = pareto_Y.detach().cpu().numpy()
            
            print(f"Pareto points: {n_pareto}/{len(Y_all)}")
            print(f"Pareto front range: {Y_pareto_np.min(axis=0)} to {Y_pareto_np.max(axis=0)}")
            print(f"Reference point: {ref_point_t.detach().cpu().numpy()}")
            
        except Exception as e:
            print(f"⚠️ Hypervolume computation failed: {e}")
            hv = 0.0
            hypervolumes.append(hv)
            n_pareto = 0
            # Set a fallback reference point if hypervolume computation fails
            if ref_point_t is None:
                ref_point_t = torch.tensor([0.1-1e-2, -1e-2], dtype=torch.float64, device=device)
        
        # Store batch information
        batch_info.append({
            'batch': batch,
            'n_points': len(X_all),
            'n_pareto': n_pareto,
            'Y_batch': Y_all.copy(),
            'hypervolume': hv
        })
        
        # Stop if this is the last evaluation
        if batch >= n_batches:
            break
            
        # Propose next batch
        print(f"Proposing {batch_size} new points...")
        try:
            # Use the reference point computed from hypervolume calculation
            #ref_point_device = ref_point_t.to(device=device, dtype=torch.float64)
            
            # Call propose_batch with correct signature
            result = propose_batch(
                design=design,
                model=model,
                train_X=X_t,  # Normalized training inputs
                ref_point_t=ref_point_t,
                batch_size=batch_size,
                sample_shape=64,
                verbose=True
            )
            
            # Extract physical coordinates (already snapped)
            X_next = result['X_phys']
            print(f"✓ Proposed points: {X_next.shape}")
            
            # Evaluate new points
            Y_next = synthetic_objectives(X_next)
            print(f"New objectives range: {Y_next.min(axis=0)} to {Y_next.max(axis=0)}")
            
            # Add to dataset
            X_all = np.vstack([X_all, X_next])
            Y_all = np.vstack([Y_all, Y_next])
            
        except Exception as e:
            print(f"⚠️ Batch proposal failed: {e}")
            break
    
    print(f"\n=== MOBO LOOP COMPLETED ===")
    print(f"Total points collected: {len(X_all)}")
    print(f"Final hypervolume: {hypervolumes[-1]:.6f}")
    print(f"Hypervolume improvement: {hypervolumes[-1] - hypervolumes[0]:.6f}")
    
    # Create progression plots
    print("\nCreating MOBO progression plots...")
    try:
        plot_mobo_progression(batch_info, hypervolumes)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
    
    # Plot final objective space
    try:
        Y_batches = [info['Y_batch'] for info in batch_info]  # Every batch
        labels = [f"Batch {info['batch']}" for info in batch_info]
        plot_objective_space(Y_batches, labels, "MOBO Objective Space Evolution")
    except Exception as e:
        print(f"⚠️ Objective space plot failed: {e}")
    
    return {
        'X_final': X_all,
        'Y_final': Y_all, 
        'batch_info': batch_info,
        'hypervolumes': hypervolumes,
        'model': model,
    }


def compute_pareto_front(Y, minimize=False):
    """
    Compute Pareto front for maximization (default) or minimization
    
    Args:
        Y: Objective values (N, M)
        minimize: If True, find Pareto front for minimization
        
    Returns:
        pareto_mask: Boolean mask of Pareto optimal points
    """
    Y_work = -Y if not minimize else Y  # Convert to minimization
    
    n_points, n_obj = Y_work.shape
    pareto_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not pareto_mask[i]:
            continue
            
        # Check if point i is dominated by any other point
        for j in range(n_points):
            if i == j or not pareto_mask[j]:
                continue
                
            # j dominates i if j is better in all objectives
            if np.all(Y_work[j] <= Y_work[i]) and np.any(Y_work[j] < Y_work[i]):
                pareto_mask[i] = False
                break
    
    return pareto_mask


if __name__ == "__main__":
    # Run GP modeling test
    print("Starting synthetic test with maximization objectives...")
    gp_results = test_gp_pipeline()
    
    # Run MOBO loop test
    print("\n" + "="*60)
    mobo_results = test_mobo_loop(n_initial=20, n_batches=5, batch_size=5)
    
    print(f"\n🎯 SYNTHETIC TEST SUMMARY:")
    print(f"GP modeling: ✓ Completed")
    print(f"MOBO loop: ✓ Completed ({len(mobo_results['X_final'])} total points)")
    print(f"Hypervolume improvement: {mobo_results['hypervolumes'][-1] - mobo_results['hypervolumes'][0]:.6f}")
    print(f"Final Pareto points: {mobo_results['batch_info'][-1]['n_pareto']}")
    print("All plots saved to current directory ✓")
