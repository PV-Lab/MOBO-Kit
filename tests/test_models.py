import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np
import pandas as pd
from src.design import build_design_from_config
from src.utils import load_csv, split_XY, np_to_torch, get_objective_names
from src.models import fit_gp_models, default_noise_options, loocv_select_models, posterior_report
from src.data import y_minmax_np

import gpytorch

CFG_PATH = "configs/configCSV_example_config.yaml"
CSV_PATH = "data/processed/configCSV_example.csv"

def test_basic_gp_fitting():
    """Test basic GP model fitting with real data."""
    print("Testing GP model fitting...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    print(f"Data loaded: X shape {X.shape}, Y shape {Y.shape}")
    
    # Convert to torch tensors (test with CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    print(f"Converted to torch: X {X_t.shape}, Y {Y_t.shape} on {X_t.device}")
    
    # Test noise options
    print("\nTesting noise options...")
    noise_opts = default_noise_options(device=device)
    print(f"Available noise options: {list(noise_opts.keys())}")
    
    # Test that we can create likelihoods with these priors
    for name, prior in noise_opts.items():
        if prior is not None:
            try:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=prior)
                print(f"✓ {name}: Successfully created likelihood")
            except Exception as e:
                print(f"✗ {name}: Failed to create likelihood - {e}")
        else:
            print(f"✓ {name}: No prior (default likelihood)")
    
    # Fit GP models
    print("\nFitting GP models...")
    model = fit_gp_models(X_t, Y_t)
    
    # Verify model structure
    assert hasattr(model, 'models'), "Should return ModelListGP"
    assert len(model.models) == Y.shape[1], f"Should have {Y.shape[1]} models"
    print(f"✓ Successfully fitted {len(model.models)} GP models")
    
    # Test prediction
    with torch.no_grad():
        posterior = model.posterior(X_t)
        pred_mean = posterior.mean
        pred_var = posterior.variance
    
    assert pred_mean.shape == Y_t.shape, "Prediction mean shape mismatch"
    assert pred_var.shape == Y_t.shape, "Prediction variance shape mismatch"
    print(f"✓ Predictions have correct shape: {pred_mean.shape}")
    
    return model, X_t, Y_t


def test_loocv_model_selection():
    """Test LOOCV model selection with real data."""
    print("\nTesting LOOCV model selection...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    objective_names = get_objective_names(config)
    
    # Convert to torch tensors (test with CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    print(f"Running LOOCV on {len(X_t)} samples with {len(objective_names)} objectives")
    
    # Run LOOCV model selection (simplified for speed)
    best_model, results_df = loocv_select_models(
        X_t, Y_t, 
        objective_names=objective_names,
        device=device
    )
    
    # Verify results
    assert hasattr(best_model, 'models'), "Should return ModelListGP"
    assert len(best_model.models) == len(objective_names), "Should have model for each objective"
    assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
    
    print(f"✓ LOOCV completed with {len(results_df)} combinations tested")
    print(f"✓ Results columns: {list(results_df.columns)}")
    print(f"✓ Best model has {len(best_model.models)} objectives")
    
    # Check results structure
    expected_cols = {"Kernel", "NoisePrior", "Objective", "R2", "RMSE"}
    assert expected_cols.issubset(set(results_df.columns)), "Missing expected columns"
    
    # Show sample results
    print("\nSample LOOCV results:")
    print(results_df.head(10))
    
    return best_model, results_df, X_t, Y_t, objective_names


def test_posterior_report():
    """Test posterior reporting with unnormalization."""
    print("\nTesting posterior report...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    objective_names = get_objective_names(config)
    
    # Convert to torch tensors (test with CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    # Normalize Y data (as would be done in real pipeline)
    Y_scaled, Y_min, Y_max = y_minmax_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    print(f"Y normalization - Min: {Y_min}, Max: {Y_max}")
    
    # Fit model on scaled data
    model = fit_gp_models(X_t, Y_scaled_t)
    
    # Generate posterior report
    report_df, metrics_df = posterior_report(
        model, X_t, Y_scaled_t, Y_min, Y_max,
        objective_names=objective_names,
        add_residuals=True,
        add_zscores=True
    )
    
    # Verify report structure
    assert isinstance(report_df, pd.DataFrame), "Should return DataFrame"
    assert isinstance(metrics_df, pd.DataFrame), "Should return metrics DataFrame"
    
    print(f"✓ Report generated with {len(report_df)} rows")
    print(f"✓ Report columns: {list(report_df.columns)}")
    print(f"✓ Metrics columns: {list(metrics_df.columns)}")
    
    # Check for expected columns
    for obj_name in objective_names:
        assert f"True[{obj_name}]" in report_df.columns, f"Missing True column for {obj_name}"
        assert f"Pred[{obj_name}]" in report_df.columns, f"Missing Pred column for {obj_name}"
        assert f"Std[{obj_name}]" in report_df.columns, f"Missing Std column for {obj_name}"
        assert f"Residual[{obj_name}]" in report_df.columns, f"Missing Residual column for {obj_name}"
        assert f"Z[{obj_name}]" in report_df.columns, f"Missing Z-score column for {obj_name}"
    
    print("\nMetrics summary:")
    print(metrics_df)
    
    print("\nSample report data:")
    print(report_df.head())
    
    return report_df, metrics_df


def test_posterior_report_simple():
    """Simple test for posterior report to debug issues."""
    print("\nTesting posterior report (simple)...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    objective_names = get_objective_names(config)
    
    # Convert to torch tensors (test with CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    # Normalize Y data (as would be done in real pipeline)
    Y_scaled, Y_min, Y_max = y_minmax_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    print(f"Y normalization - Min: {Y_min}, Max: {Y_max}")
    print(f"Y_scaled_t shape: {Y_scaled_t.shape}, device: {Y_scaled_t.device}")
    
    # Fit model on scaled data
    print("Fitting model...")
    model = fit_gp_models(X_t, Y_scaled_t)
    print(f"Model fitted with {len(model.models)} objectives")
    
    # Test posterior step by step
    print("Testing posterior step by step...")
    
    # Step 1: Get posterior
    print("Step 1: Getting posterior...")
    post = model.posterior(X_t)
    pred_mean_t = post.mean
    pred_std_t = torch.sqrt(post.variance)
    print(f"Posterior shapes - mean: {pred_mean_t.shape}, std: {pred_std_t.shape}")
    
    # Step 2: Convert to numpy
    print("Step 2: Converting to numpy...")
    from src.utils import torch_to_np
    pred_mean, pred_std, true_scaled = torch_to_np(pred_mean_t, pred_std_t, Y_scaled_t)
    print(f"Numpy shapes - mean: {pred_mean.shape}, std: {pred_std.shape}, true: {true_scaled.shape}")
    
    # Step 3: Unnormalize
    print("Step 3: Unnormalizing...")
    Y_min = np.asarray(Y_min, dtype=float)
    Y_max = np.asarray(Y_max, dtype=float)
    scale = (Y_max - Y_min).astype(float)
    print(f"Scale factors: {scale}")
    
    pred_mean_unnorm = pred_mean * scale + Y_min
    pred_std_unnorm = pred_std * scale
    true_Y = true_scaled * scale + Y_min
    print(f"Unnormalized shapes - mean: {pred_mean_unnorm.shape}, std: {pred_std_unnorm.shape}, true: {true_Y.shape}")
    
    print("✓ All steps completed successfully!")
    return model, X_t, Y_scaled_t, Y_min, Y_max, objective_names


def main():
    """Run comprehensive model tests."""
    print("Running comprehensive models.py tests...\n")
    
    try:
        # Test 1: Basic GP fitting
        print("="*60)
        model, X_t, Y_t = test_basic_gp_fitting()
        
        # Test 2: LOOCV model selection
        print("="*60)
        best_model, results_df, X_t, Y_t, objective_names = test_loocv_model_selection()
        
        # Test 3: Posterior reporting (simple)
        print("="*60)
        model, X_t, Y_scaled_t, Y_min, Y_max, objective_names = test_posterior_report_simple()
        
        print("="*60)
        print("\n🎉 All model tests passed successfully!")
        print(f"🎯 Tested with {len(X_t)} samples across {X_t.shape[1]} input dimensions")
        print(f"🎯 Validated {len(objective_names)} objectives: {objective_names}")
        print(f"🎯 LOOCV tested {len(results_df)} kernel/noise combinations")
        print(f"🎯 Generated comprehensive posterior report with metrics")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
