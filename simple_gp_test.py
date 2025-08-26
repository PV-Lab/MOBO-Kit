#!/usr/bin/env python3
"""
Simple GP test with visualizations - no LOOCV, just basic functionality
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel

from src.models import fit_gp_models, posterior_report
from src.data import y_standardize_np
from src.utils import np_to_torch

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def synthetic_objectives(X):
    """
    Two synthetic objective functions with known properties:
    - Obj1: Quadratic with global minimum
    - Obj2: Sinusoidal with multiple local optima
    """
    x1, x2 = X[:, 0], X[:, 1]
    
    # Objective 1: Quadratic bowl (minimize)
    obj1 = (x1 - 0.3)**2 + (x2 - 0.7)**2 + 0.1
    
    # Objective 2: Sinusoidal (minimize) 
    obj2 = 0.5 * np.sin(6 * np.pi * x1) * np.cos(4 * np.pi * x2) + 0.5
    
    return np.column_stack([obj1, obj2])

def generate_training_data(n_points=20, noise_std=0.05, seed=42):
    """Generate training data with controlled noise"""
    np.random.seed(seed)
    
    # Random sampling in [0,1]^2
    X_train = np.random.uniform(0, 1, size=(n_points, 2))
    
    # Evaluate true objectives
    Y_true = synthetic_objectives(X_train)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, Y_true.shape)
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
    
    # Objective 1: Quadratic
    im1 = axes[0].contourf(X1, X2, obj1_grid, levels=20, cmap='viridis')
    axes[0].set_title('Objective 1: Quadratic Bowl\n(x₁-0.3)² + (x₂-0.7)² + 0.1')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].plot(0.3, 0.7, 'r*', markersize=15, label='Global minimum')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0])
    
    # Objective 2: Sinusoidal
    im2 = axes[1].contourf(X1, X2, obj2_grid, levels=20, cmap='plasma')
    axes[1].set_title('Objective 2: Sinusoidal\n0.5×sin(6πx₁)×cos(4πx₂) + 0.5')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('simple_true_objectives.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_gp_predictions(model, X_train, Y_train, Y_mean, Y_std, title_suffix=""):
    """Plot GP predictions vs true functions"""
    # Create prediction grid
    n_grid = 50
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # True values
    Y_true_grid = synthetic_objectives(X_grid)
    
    # GP predictions
    X_grid_t = torch.tensor(X_grid, dtype=torch.float64)
    pred_mean, pred_std = posterior_report(model, X_grid_t, Y_mean, Y_std)
    
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
                     s=80, cmap='viridis', edgecolors='white', linewidth=2)
    axes[0,0].set_title('Obj1: True')
    axes[0,0].set_ylabel('x₂')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Predicted
    im2 = axes[0,1].contourf(X1, X2, obj1_pred, levels=20, cmap='viridis')
    axes[0,1].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], 
                     s=80, cmap='viridis', edgecolors='white', linewidth=2)
    axes[0,1].set_title('Obj1: GP Prediction')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Uncertainty
    im3 = axes[0,2].contourf(X1, X2, obj1_std, levels=20, cmap='Reds')
    axes[0,2].scatter(X_train[:, 0], X_train[:, 1], c='white', 
                     s=80, edgecolors='black', linewidth=2)
    axes[0,2].set_title('Obj1: GP Uncertainty')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Objective 2 row
    # True
    im4 = axes[1,0].contourf(X1, X2, obj2_true, levels=20, cmap='plasma')
    axes[1,0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 1], 
                     s=80, cmap='plasma', edgecolors='white', linewidth=2)
    axes[1,0].set_title('Obj2: True')
    axes[1,0].set_xlabel('x₁')
    axes[1,0].set_ylabel('x₂')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Predicted
    im5 = axes[1,1].contourf(X1, X2, obj2_pred, levels=20, cmap='plasma')
    axes[1,1].scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 1], 
                     s=80, cmap='plasma', edgecolors='white', linewidth=2)
    axes[1,1].set_title('Obj2: GP Prediction')
    axes[1,1].set_xlabel('x₁')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Uncertainty
    im6 = axes[1,2].contourf(X1, X2, obj2_std, levels=20, cmap='Reds')
    axes[1,2].scatter(X_train[:, 0], X_train[:, 1], c='white', 
                     s=80, edgecolors='black', linewidth=2)
    axes[1,2].set_title('Obj2: GP Uncertainty')
    axes[1,2].set_xlabel('x₁')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.suptitle(f'Default GP Model Performance {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'simple_gp_predictions{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_training_fit(Y_true, Y_pred, Y_std, objective_names=['Obj1', 'Obj2']):
    """Simple parity plot for training fit"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, obj_name in enumerate(objective_names):
        ax = axes[i]
        
        # Plot parity line
        y_min, y_max = min(Y_true[:, i].min(), Y_pred[:, i].min()), max(Y_true[:, i].max(), Y_pred[:, i].max())
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', alpha=0.5, label='Perfect fit')
        
        # Plot predictions with error bars
        ax.errorbar(Y_true[:, i], Y_pred[:, i], yerr=Y_std[:, i], 
                   fmt='o', alpha=0.7, capsize=3, label='GP predictions')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(Y_true[:, i], Y_pred[:, i])
        
        ax.set_xlabel(f'True {obj_name}')
        ax.set_ylabel(f'Predicted {obj_name}')
        ax.set_title(f'{obj_name}: R² = {r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Make axes equal
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('simple_parity_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def simple_gp_test():
    """Simple test of default GP models with visualizations"""
    
    print("=== SIMPLE GP TEST ===")
    
    # Show the true objective functions
    print("1. Plotting true objective functions...")
    plot_true_objectives()
    
    # Generate training data
    print("\n2. Generating training data...")
    X_train, Y_train, Y_true = generate_training_data(n_points=100, noise_std=0.02, seed=42)
    print(f"   Training data: {X_train.shape[0]} points, {X_train.shape[1]} dimensions")
    print(f"   Objectives: {Y_train.shape[1]} (Quadratic, Sinusoidal)")
    print(f"   Y_train range: {Y_train.min(axis=0)} to {Y_train.max(axis=0)}")
    
    # Standardize Y and prepare tensors
    print("\n3. Standardizing data...")
    Y_std, Y_mean, Y_scale = y_standardize_np(Y_train)
    (X_t, Y_t), device = np_to_torch(X_train, Y_std, device='cpu', return_device=True)
    print(f"   Y standardized: mean={Y_std.mean(axis=0)}, std={Y_std.std(axis=0)}")
    print(f"   Using device: {device}")
    
    # Fit default GP model
    print("\n4. Fitting default GP model...")
    try:
        model = fit_gp_models(X_t, Y_t, kernel_fn=[lambda d: RBFKernel(ard_num_dims=d), lambda d: MaternKernel(nu=0.5,ard_num_dims=d)])  # No additional arguments - just defaults
        print("   ✓ Default GP model fitted successfully!")
        
        # Check model details
        print(f"   Model type: {type(model).__name__}")
        print(f"   Number of sub-models: {len(model.models)}")
        for i, sub_model in enumerate(model.models):
            print(f"   Sub-model {i}: {type(sub_model).__name__}")
            
    except Exception as e:
        print(f"   ✗ GP fitting failed: {e}")
        return None
    
    # Test predictions on training data
    print("\n5. Testing predictions...")
    try:
        pred_mean, pred_std = posterior_report(model, X_t, Y_mean, Y_scale)
        print(f"   Prediction shapes: mean={pred_mean.shape}, std={pred_std.shape}")
        print(f"   Prediction ranges: mean={pred_mean.min(axis=0)} to {pred_mean.max(axis=0)}")
        print(f"   Uncertainty ranges: std={pred_std.min(axis=0)} to {pred_std.max(axis=0)}")
        
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        return None
    
    # Compute and show training fit quality
    print("\n6. Computing training fit metrics...")
    from sklearn.metrics import r2_score, mean_squared_error
    
    r2_scores = []
    rmse_scores = []
    for j in range(2):
        r2 = r2_score(Y_train[:, j], pred_mean[:, j])
        rmse = np.sqrt(mean_squared_error(Y_train[:, j], pred_mean[:, j]))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        
        obj_name = "Quadratic" if j == 0 else "Sinusoidal"
        print(f"   {obj_name}: R² = {r2:.3f}, RMSE = {rmse:.4f}")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    
    # Plot GP predictions
    plot_gp_predictions(model, X_train, Y_train, Y_mean, Y_scale, f"(N={len(X_train)})")
    
    # Plot training fit
    plot_training_fit(Y_train, pred_mean, pred_std, ['Quadratic', 'Sinusoidal'])
    
    print("\n=== SIMPLE GP TEST COMPLETED ===")
    print("Check the generated PNG files for visualizations!")
    
    return {
        'X_train': X_train, 'Y_train': Y_train,
        'model': model, 'Y_mean': Y_mean, 'Y_scale': Y_scale,
        'pred_mean': pred_mean, 'pred_std': pred_std,
        'r2_scores': r2_scores, 'rmse_scores': rmse_scores
    }

if __name__ == "__main__":
    results = simple_gp_test()
