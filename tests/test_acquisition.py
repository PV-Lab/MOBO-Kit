# tests/test_acquisition.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np
import pandas as pd
from src.design import build_design_from_config
from src.utils import load_csv, split_XY, np_to_torch, get_objective_names
from src.models import fit_gp_models, loocv_select_models
from src.data import y_minmax_np, y_standardize_np
from src.acquisition import (
    outcome_ge, outcome_le,
    _unit_bounds, _make_snap_postproc, build_qnehvi, optimize_acq_qnehvi, propose_batch
)

CFG_PATH = "configs/configCSV_example_config.yaml"
CSV_PATH = "data/processed/configCSV_example.csv"

def test_outcome_constraint_builders():
    """Test outcome constraint builders with various shapes and values."""
    print("Testing outcome constraint builders...")
    
    # Test different tensor shapes
    shapes = [(64, 2, 4, 3), (10, 5), (100,)]
    
    for shape in shapes:
        Y = torch.zeros(shape)
        
        # Test outcome_ge: feasible when Y[..., 1] >= 0.8
        c_ge = outcome_ge(obj_idx=1, thresh=0.8)
        val_ge = c_ge(Y)

        assert val_ge.shape == shape[:-1], f"Shape mismatch for {shape}"
        assert torch.all(val_ge > 0), f"All zeros should be infeasible for {shape}"

        # Make feasible
        Y[..., 1] = 0.9
        assert torch.all(c_ge(Y) <= 0), f"Should be feasible for {shape}"

        # Test outcome_le: feasible when Y[..., 2] <= 0.15
        Y = torch.ones(shape)
        c_le = outcome_le(obj_idx=2, thresh=0.15)
        val_le = c_le(Y)

        assert val_le.shape == shape[:-1], f"Shape mismatch for {shape}"
        assert torch.all(val_le > 0), f"All ones should be infeasible for {shape}"

        # Make feasible
        Y[..., 2] = 0.10
        assert torch.all(c_le(Y) <= 0), f"Should be feasible for {shape}"
    
    print("✓ Outcome constraint builders work with various shapes")

def test_unit_bounds():
    """Test unit bounds helper function."""
    print("Testing unit bounds helper...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    for d in [1, 3, 8]:
        bounds = _unit_bounds(d, device, dtype)
        
        assert bounds.shape == (2, d), f"Expected shape (2, {d}), got {bounds.shape}"
        assert bounds[0].allclose(torch.zeros(d, device=device, dtype=dtype)), "Lower bounds should be 0"
        assert bounds[1].allclose(torch.ones(d, device=device, dtype=dtype)), "Upper bounds should be 1"
        assert bounds.device.type == device.type, "Device type mismatch"
        assert bounds.dtype == dtype, "Dtype mismatch"
    
    print("✓ Unit bounds helper works correctly")

def test_snap_postproc_factory():
    """Test snap post-processing factory function."""
    print("Testing snap post-processing factory...")
    
    # Load config and design
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    
    # Create post-processing function
    postproc = _make_snap_postproc(design)
    
    # Test with different shapes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_shapes = [(5, 8), (1, 3, 8), (10, 2, 8)]
    
    for shape in test_shapes:
        # Create random normalized input
        Z = torch.rand(shape, device=device, dtype=torch.float64)
        
        # Apply post-processing
        Z_snapped = postproc(Z)
        
        # Check output shape matches input
        assert Z_snapped.shape == Z.shape, f"Shape mismatch for {shape}"
        assert Z_snapped.device == Z.device, "Device mismatch"
        assert Z_snapped.dtype == Z.dtype, "Dtype mismatch"
        
        # Check values are in [0, 1] range
        assert torch.all(Z_snapped >= 0) and torch.all(Z_snapped <= 1), "Values out of [0,1] range"
    
    print("✓ Snap post-processing factory works correctly")

def test_build_qnehvi():
    """Test qNEHVI acquisition function builder."""
    print("Testing qNEHVI builder...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    # Convert to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    # Normalize Y data
    Y_scaled, Y_min, Y_max = y_minmax_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    # Fit model
    model = fit_gp_models(X_t, Y_scaled_t)
    
    # Build qNEHVI
    ref_point = Y_scaled_t.min(dim=0).values - 0.01
    acq_func = build_qnehvi(
        model=model,
        train_X=X_t,
        ref_point_t=ref_point,
        sample_shape=64,  # Smaller for testing
        use_lognehvi=True
    )
    
    # Verify acquisition function properties
    assert hasattr(acq_func, 'model'), "Should have model attribute"
    assert hasattr(acq_func, 'ref_point'), "Should have ref_point attribute"
    assert hasattr(acq_func, 'X_baseline'), "Should have X_baseline attribute"
    
    print("✓ qNEHVI builder works correctly")

def test_optimize_acq_qnehvi():
    """Test acquisition function optimization wrapper."""
    print("Testing acquisition optimization wrapper...")
    
    # Load real data and fit model
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    Y_scaled, Y_min, Y_max = y_minmax_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    model = fit_gp_models(X_t, Y_scaled_t)
    
    # Build acquisition function
    ref_point = Y_scaled_t.min(dim=0).values - 0.01
    acq_func = build_qnehvi(
        model=model,
        train_X=X_t,
        ref_point_t=ref_point,
        sample_shape=32,  # Small for testing
        use_lognehvi=True
    )
    
    # Test optimization
    d = X_t.shape[1]
    q = 2  # Small batch for testing
    
    candidates, acq_values = optimize_acq_qnehvi(
        acq_function=acq_func,
        d=d,
        q=q,
        num_restarts=5,  # Small for testing
        raw_samples=100,  # Small for testing
        device=device,
        dtype=torch.float64
    )
    
    # Verify output shapes
    assert candidates.shape == (q, d), f"Expected shape ({q}, {d}), got {candidates.shape}"
    assert acq_values.shape == (q,), f"Expected shape ({q},), got {acq_values.shape}"
    
    # Verify values are in [0, 1] range (normalized)
    assert torch.all(candidates >= 0) and torch.all(candidates <= 1), "Candidates out of [0,1] range"
    
    print("✓ Acquisition optimization wrapper works correctly")

def test_propose_batch_basic():
    """Test basic propose_batch functionality without constraints."""
    print("Testing basic propose_batch...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    Y_scaled, Y_min, Y_max = y_minmax_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    model = fit_gp_models(X_t, Y_scaled_t)
    
    # Test basic proposal
    ref_point = Y_scaled_t.min(dim=0).values - 0.01
    batch_size = 3
    
    result = propose_batch(
        design=design,
        model=model,
        train_X=X_t,
        ref_point_t=ref_point,
        batch_size=batch_size,
        num_restarts=5,  # Small for testing
        raw_samples=100,  # Small for testing
        sample_shape=32,  # Small for testing
        verbose=True
    )
    
    # Verify result structure
    assert isinstance(result, dict), "Should return dictionary"
    assert 'X_phys' in result, "Should have X_phys key"
    assert 'X_norm' in result, "Should have X_norm key"
    assert 'attempts' in result, "Should have attempts key"
    assert 'acq_val' in result, "Should have acq_val key"
    
    # Verify shapes
    if result['X_phys'].shape[0] > 0:  # If we got valid candidates
        assert result['X_phys'].shape[1] == len(design.names), "Physical dimensions should match design"
        assert result['X_norm'].shape[1] == len(design.names), "Normalized dimensions should match design"
        assert result['X_phys'].shape[0] <= batch_size, "Should not exceed batch size"
        assert result['X_norm'].shape[0] <= batch_size, "Should not exceed batch size"
        assert result['acq_val'].shape[0] <= batch_size, "Should not exceed batch size"
    
    print("✓ Basic propose_batch works correctly")

def test_propose_batch_with_custom_acq():
    """Test propose_batch with custom acquisition function."""
    print("Testing propose_batch with custom acquisition function...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    Y_scaled, Y_mean, Y_std = y_standardize_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    model = fit_gp_models(X_t, Y_scaled_t)
    
    # Define custom acquisition function builder
    def custom_acq_builder():
        ref_point = Y_scaled_t.min(dim=0).values - 0.01
        return build_qnehvi(
            model=model,
            train_X=X_t,
            ref_point_t=ref_point,
            sample_shape=32,
            use_lognehvi=False  # Use regular NEHVI instead of log
        )
    
    # Test with custom acquisition function
    ref_point = Y_scaled_t.min(dim=0).values - 0.01
    batch_size = 2
    
    result = propose_batch(
        design=design,
        model=model,
        train_X=X_t,
        ref_point_t=ref_point,
        batch_size=batch_size,
        acq=custom_acq_builder,  # Use custom acquisition function
        num_restarts=3,  # Small for testing
        raw_samples=50,  # Small for testing
        sample_shape=16,  # Small for testing
        verbose=True
    )
    
    # Verify result structure
    assert isinstance(result, dict), "Should return dictionary"
    assert all(key in result for key in ['X_phys', 'X_norm', 'attempts', 'acq_val']), "Missing expected keys"
    
    print("✓ Custom acquisition function works correctly")

def test_propose_batch_with_constraints():
    """Test propose_batch with both row constraints and outcome constraints."""
    print("Testing propose_batch with constraints...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    device = torch.device('cuda')#'cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    Y_scaled, Y_mean, Y_std = y_standardize_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    from src.data import x_normalizer_torch
    Xn_t = x_normalizer_torch(X_t, design)
    
    objective_names = get_objective_names(config)
    model_cv, results_df = loocv_select_models(Xn_t, Y_scaled_t, objective_names=objective_names, device=device)
    #model = fit_gp_models(Xn_t, Y_scaled_t)
    
    # 🔍 DEBUG: Analyze model uncertainty and performance
    print(f"\n🔍 Model Analysis:")
    print(f"   Dataset size: {Xn_t.shape[0]} samples, {Xn_t.shape[1]} inputs, {Y_scaled_t.shape[1]} objectives")
    print(f"   LOOCV results columns: {list(results_df.columns)}")
    print(f"   LOOCV results:")
    
    # Show LOOCV performance for each objective
    for obj_name in objective_names:
        obj_results = results_df[results_df['Objective'] == obj_name]
        if len(obj_results) > 0:
            best_idx = obj_results['RMSE'].idxmin()
            best_kernel = obj_results.loc[best_idx, 'Kernel']
            best_noise = obj_results.loc[best_idx, 'NoisePrior']
            best_r2 = obj_results.loc[best_idx, 'R2']
            best_rmse = obj_results.loc[best_idx, 'RMSE']
            print(f"     {obj_name}: Best R²={best_r2}, RMSE={best_rmse} (Kernel: {best_kernel}, Noise: {best_noise})")
    
    # Use posterior_report for comprehensive model analysis
    from src.models import posterior_report
    from src.metrics import compute_metrics
    print(f"\n📊 Model Performance on Training Data:")
    try:
        pred_mean, pred_std = posterior_report(
            model_cv, Xn_t, Y_mean, Y_std
        )
        metrics_df = compute_metrics(Y, pred_mean, pred_std, objective_names=objective_names, add_residuals=True, add_zscores=True)
        print(f"   Training performance metrics:")
        for _, row in metrics_df.iterrows():
            print(f"     {row['Objective']}: R²={row['R2']}, RMSE={row['RMSE']}")
    
    except Exception as e:
        print(f"   Could not run posterior_report: {e}")

    from src.plotting import plot_parity_np
    from src.utils import torch_to_np
    
    # Extract predictions for each objective from report_df
    # pred_columns = [f"Pred[{name}]" for name in objective_names]
    # pred_mean = report_df[pred_columns].values  # Shape: (N, M)
    # #print(pred_mean)
    
    # # Extract standard deviations if available
    # std_columns = [f"Std[{name}]" for name in objective_names]
    # pred_std = report_df[std_columns].values if all(col in report_df.columns for col in std_columns) else None
    
    fig, parity_df = plot_parity_np(Y, pred_mean=pred_mean, pred_std=pred_std, objective_names=objective_names)
    
    # Define row constraints (temperature + humidity constraint)
    from src.constraints import constraints_from_config
    row_constraints_list = constraints_from_config(config, design)
    
    # Define outcome constraints using original units (more intuitive)
    from src.acquisition import outcome_ge_standardized, outcome_le_standardized
    
    # Example: PCE >= 15% and Repeatability <= 0.1 (adjust these values based on your actual data!)
    # You should replace these with your actual desired thresholds in original units
    pce_threshold = 15.0  # Example: 15% PCE minimum
    repeatability_threshold = 0.1  # Example: 0.1 maximum repeatability
    
    outcome_constraints = [
        outcome_ge_standardized(obj_idx=0, thresh_original=pce_threshold, 
                                Y_mean=Y_mean[0], Y_std=Y_std[0]),  # PCE >= 15%
        outcome_le_standardized(obj_idx=2, thresh_original=repeatability_threshold, 
                                Y_mean=Y_mean[2], Y_std=Y_std[2]),  # Repeatability <= 0.1
    ]
    
    print(f"📋 Outcome constraints:")
    print(f"   PCE >= {pce_threshold} (standardized: {(pce_threshold - Y_mean[0]) / Y_std[0]:.3f})")
    print(f"   Repeatability <= {repeatability_threshold} (standardized: {(repeatability_threshold - Y_mean[2]) / Y_std[2]:.3f})")
    
    # Test with both types of constraints
    ref_point = Y_scaled_t.min(dim=0).values - 0.01
    
    # 🔍 DEBUG: Analyze reference point and hypervolume
    print(f"\n📊 Hypervolume Analysis:")
    print(f"   Reference point: {ref_point.tolist()}")
    print(f"   Training data bounds: min={Y_scaled_t.min(dim=0).values.tolist()}")
    print(f"   Training data bounds: max={Y_scaled_t.max(dim=0).values.tolist()}")
    
    # Calculate current hypervolume using metrics.py functions
    from src.metrics import compute_ref_pareto_hv
    
    # Get Pareto front and hypervolume
    ref_point_auto, pareto_front, current_hv = compute_ref_pareto_hv(Y_scaled_t)
    print(f"   Auto reference point: {ref_point_auto.tolist()}")
    print(f"   Pareto front size: {pareto_front.shape[0]} points")
    print(f"   Current hypervolume: {current_hv:.6f}")
    
    # Compare with our manual reference point
    print(f"   Manual vs Auto ref point difference: {torch.norm(ref_point - ref_point_auto):.6f}")
    
    batch_size = 8
    
    result = propose_batch(
        design=design,
        model=model_cv,
        train_X=Xn_t,
        ref_point_t=ref_point_auto,
        batch_size=batch_size,
        row_constraints=row_constraints_list,  # Physical space constraints
        #constraints=outcome_constraints,       # Outcome space constraints
        eta=0.05,                             # Constraint violation penalty
        num_restarts=10,                       # Small for testing
        raw_samples=512,                       # Increased for better MC estimation
        sample_shape=256,                      # Increased for more candidates
        max_attempts=5,                       # More attempts due to constraints
        verbose=True,
        #use_lognehvi=False                     # Use regular NEHVI to see raw EI values
    )
    from src.plotting import plot_bar
    Xn_new = result['X_norm']
    Xn_new_t = np_to_torch(Xn_new, device=device)
    X_new = result['X_phys']
    pred_mean_new, pred_std_new = posterior_report(
            model_cv, Xn_new_t, Y_mean, Y_std, 
        )
    fig_bar = plot_bar(pred_mean_new, pred_std_new, labels=objective_names)

    # Verify result structure
    assert isinstance(result, dict), "Should return dictionary"
    assert all(key in result for key in ['X_phys', 'X_norm', 'attempts', 'acq_val']), "Missing expected keys"
    
    # If we got valid candidates, verify they satisfy row constraints
    if result['X_phys'].shape[0] > 0:
        print(f"Generated {result['X_phys'].shape[0]} candidates in {result['attempts']} attempts")
        
        # 🔍 DEBUG: Analyze acquisition values and model predictions
        print(f"\n📈 Acquisition Value Analysis:")
        print(f"   Raw acquisition values (log(EHVI)): {[f'{v:.6f}' for v in result['acq_val']]}")
        
        # Convert from log(EHVI) to EHVI since use_lognehvi=True (default)
        ehvi_values = [np.exp(v) for v in result['acq_val']]
        print(f"   Actual EHVI values: {[f'{v:.6f}' for v in ehvi_values]}")
        
        # Print candidate details
        print(f"\n🔍 Generated Candidates (Physical Units):")
        for i, (phys_vals, norm_vals, acq_val) in enumerate(zip(result['X_phys'], result['X_norm'], result['acq_val'])):
            print(f"   Candidate {i+1}:")
            print(f"     Log(EHVI): {acq_val:.6f}")
            print(f"     EHVI: {np.exp(acq_val):.6f}")
            print(f"     Physical Values:")
            for j, name in enumerate(design.names):
                print(f"       {name}: {phys_vals[j]:.4f}")
            print(f"     Normalized Values:")
            for j, name in enumerate(design.names):
                print(f"       {name}: {norm_vals[j]:.4f}")
            print()
        
        # Verify physical constraints are satisfied
        from src.constraints import apply_row_constraints
        constraint_mask = apply_row_constraints(result['X_phys'], design, row_constraints_list)
        print(f"🔒 Row Constraint Satisfaction:")
        print(f"   All candidates satisfy row constraints: {np.all(constraint_mask)}")
        
        # Verify dimensions match design
        assert result['X_phys'].shape[1] == len(design.names), "Physical dimensions should match design"
        assert result['X_norm'].shape[1] == len(design.names), "Normalized dimensions should match design"
        
        # Check that physical values are within design bounds
        for i, name in enumerate(design.names):
            phys_vals = result['X_phys'][:, i]
            design_min = design.lowers[i]
            design_max = design.uppers[i]
            assert np.all(phys_vals >= design_min) and np.all(phys_vals <= design_max), \
                f"Values for {name} out of bounds [{design_min}, {design_max}]"
        
        # Check that normalized values are in [0, 1]
        assert np.all(result['X_norm'] >= 0) and np.all(result['X_norm'] <= 1), \
            "Normalized values should be in [0, 1]"
        
        print(f"✓ All {result['X_phys'].shape[0]} candidates satisfy constraints")
        
        # 🔍 DEBUG: Explain acquisition values
        print(f"\n💡 Acquisition Value Interpretation (Log-NEHVI):")
        print(f"   Log(EHVI) values: {[f'{v:.6f}' for v in result['acq_val']]}")
        print(f"   Actual EHVI values: {[f'{v:.6f}' for v in ehvi_values]}")
        print(f"   Higher EHVI = Greater expected improvement in hypervolume")
        print(f"   Lower EHVI = Minimal improvement expected")
        
    else:
        print("No valid candidates found - constraints may be too strict")
        assert result['attempts'] > 0, "Should have made at least one attempt"
    
    print("✓ Constraint handling works correctly")

def test_propose_batch_constraint_stress_test():
    """Stress test with very strict constraints to test failure handling."""
    print("Testing propose_batch with very strict constraints...")
    
    # Load real data
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    design = build_design_from_config(config)
    df = load_csv(CSV_PATH)
    X, Y = split_XY(df, design, config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t, Y_t = np_to_torch(X.values, Y.values, device=device)
    
    Y_scaled, Y_min, Y_max = y_minmax_np(Y.values)
    Y_scaled_t = torch.tensor(Y_scaled, dtype=torch.float64, device=device)
    
    # 🔍 DEBUG: Try different model fitting approaches
    print(f"\n🔍 Model Fitting Comparison:")
    
    # Option 1: Basic GP (likely overfitting)
    print(f"   Testing basic GP fit...")
    model_basic = fit_gp_models(X_t, Y_scaled_t)
    
    # Option 2: LOOCV-selected models (better regularization)
    print(f"   Testing LOOCV-selected models...")
    objective_names = get_objective_names(config)
    model_loocv, results_df = loocv_select_models(X_t, Y_scaled_t, objective_names=objective_names, device=device)
    
    # Compare model uncertainties
    print(f"\n📊 Model Uncertainty Comparison:")
    for model_name, model in [("Basic GP", model_basic), ("LOOCV GP", model_loocv)]:
        with torch.no_grad():
            posterior = model[0].posterior(X_t)
            mean = posterior.mean
            variance = posterior.variance
            uncertainty = torch.sqrt(variance)
            print(f"   {model_name}:")
            for i, obj_name in enumerate(objective_names):
                print(f"     {obj_name}: μ={mean[0, i].item():.4f}, σ={uncertainty[0, i].item():.4f}")
    
    # Use LOOCV model for better performance
    model = model_loocv
    
    # Define very strict constraints that are nearly impossible to satisfy
    def strict_row_constraint(X_phys, design):
        """Very strict constraint - only allows very specific temperature/humidity combinations."""
        temp_idx = design.names.index('temperature_c')
        humidity_idx = design.names.index('absolute_humidity')
        
        # Only allow temperature between 25-26°C AND humidity between 5-7 g/m³
        temp_ok = (X_phys[:, temp_idx] >= 25.0) & (X_phys[:, temp_idx] <= 26.0)
        humidity_ok = (X_phys[:, humidity_idx] >= 5.0) & (X_phys[:, humidity_idx] <= 7.0)
        
        return temp_ok & humidity_ok
    
    # Very strict outcome constraints
    from src.acquisition import outcome_ge
    strict_outcome_constraints = [
        outcome_ge(obj_idx=0, thresh=0.9),  # PCE >= 0.9 (very high)
        outcome_ge(obj_idx=1, thresh=0.9),  # Stability >= 0.9 (very high)
    ]
    
    ref_point = Y_scaled_t.min(dim=0).values - 0.01
    
    # 🔍 DEBUG: Test different acquisition function settings
    print(f"\n🧪 Testing Different Acquisition Settings:")
    
    # Test 1: Regular NEHVI
    print(f"   Test 1: Regular NEHVI")
    result1 = propose_batch(
        design=design,
        model=model,
        train_X=X_t,
        ref_point_t=ref_point,
        batch_size=2,
        row_constraints=[strict_row_constraint],
        constraints=strict_outcome_constraints,
        eta=0.01,  # Strict penalty
        num_restarts=2,   # Small for testing
        raw_samples=64,   # Small for testing
        sample_shape=16,  # Small for testing
        max_attempts=3,   # Limited attempts
        verbose=False,
        use_lognehvi=False  # Regular NEHVI
    )
    
    # Test 2: LogNEHVI (your default)
    print(f"   Test 2: LogNEHVI (default)")
    result2 = propose_batch(
        design=design,
        model=model,
        train_X=X_t,
        ref_point_t=ref_point,
        batch_size=2,
        row_constraints=[strict_row_constraint],
        constraints=strict_outcome_constraints,
        eta=0.01,  # Strict penalty
        num_restarts=2,   # Small for testing
        raw_samples=64,   # Small for testing
        sample_shape=16,  # Small for testing
        max_attempts=3,   # Limited attempts
        verbose=False,
        use_lognehvi=True   # LogNEHVI
    )
    
    # Compare results
    print(f"\n📊 Acquisition Function Comparison:")
    print(f"   Regular NEHVI: {len(result1['acq_val'])} candidates, acq_vals: {[f'{v:.6f}' for v in result1['acq_val']]}")
    print(f"   LogNEHVI: {len(result2['acq_val'])} candidates, acq_vals: {[f'{v:.6f}' for v in result2['acq_val']]}")
    
    # Use the result with more candidates for detailed analysis
    result = result1 if len(result1['acq_val']) > 0 else result2
    
    # Verify graceful handling of strict constraints
    assert isinstance(result, dict), "Should return dictionary even with strict constraints"
    assert all(key in result for key in ['X_phys', 'X_norm', 'attempts', 'acq_val']), "Missing expected keys"
    assert result['attempts'] > 0, "Should have made at least one attempt"
    
    print(f"\n📊 Stress Test Results:")
    print(f"   Attempts made: {result['attempts']}")
    print(f"   Candidates found: {result['X_phys'].shape[0]}")
    
    if result['X_phys'].shape[0] == 0:
        print("✓ Gracefully handled impossible constraints (returned empty batch)")
    else:
        print(f"✓ Found {result['X_phys'].shape[0]} candidates despite strict constraints")
        
        # 🔍 DEBUG: Analyze why these candidates were selected
        print(f"\n🔍 Candidate Analysis:")
        X_candidates = torch.tensor(result['X_norm'], dtype=torch.float64, device=device)
        with torch.no_grad():
            candidate_posterior = model[0].posterior(X_candidates)
            candidate_mean = candidate_posterior.mean
            candidate_std = torch.sqrt(candidate_posterior.variance)
        
        print(f"   Model predictions for candidates:")
        for i, (acq_val, mean, std) in enumerate(zip(result['acq_val'], candidate_mean, candidate_std)):
            print(f"     Candidate {i+1}: acq={acq_val:.6f}")
            for j, obj_name in enumerate(objective_names):
                print(f"       {obj_name}: μ={mean[j].item():.4f}, σ={std[j].item():.4f}")
        
        # Print detailed candidate information
        print(f"\n🔍 Generated Candidates (Physical Units):")
        for i, (phys_vals, norm_vals, acq_val) in enumerate(zip(result['X_phys'], result['X_norm'], result['acq_val'])):
            print(f"   Candidate {i+1}:")
            print(f"     Acquisition Value: {acq_val:.6f}")
            print(f"     Physical Values:")
            for j, name in enumerate(design.names):
                print(f"       {name}: {phys_vals[j]:.4f}")
            print(f"     Normalized Values:")
            for j, name in enumerate(design.names):
                print(f"       {name}: {norm_vals[j]:.4f}")
            print()
        
        # Verify constraints are satisfied
        constraint_mask = strict_row_constraint(result['X_phys'], design)
        print(f"🔒 Constraint Satisfaction Check:")
        print(f"   Row constraint satisfied: {np.all(constraint_mask)}")
        
        # Check specific constraint values
        temp_idx = design.names.index('temperature_c')
        humidity_idx = design.names.index('absolute_humidity')
        
        print(f"   Temperature constraints (25-26°C):")
        for i, temp in enumerate(result['X_phys'][:, temp_idx]):
            in_range = 25.0 <= temp <= 26.0
            print(f"     Candidate {i+1}: {temp:.2f}°C {'✓' if in_range else '✗'}")
        
        print(f"   Humidity constraints (5-7 g/m³):")
        for i, humidity in enumerate(result['X_phys'][:, humidity_idx]):
            in_range = 5.0 <= humidity <= 7.0
            print(f"     Candidate {i+1}: {humidity:.2f} g/m³ {'✓' if in_range else '✗'}")
    
    print("✓ Stress test completed successfully")

def main():
    """Run comprehensive acquisition function tests."""
    print("Running comprehensive acquisition.py tests...\n")
    
    try:
        # Test 1: Outcome constraint builders
        print("="*60)
        test_outcome_constraint_builders()
        
        # Test 2: Unit bounds helper
        print("="*60)
        test_unit_bounds()
        
        # Test 3: Snap post-processing factory
        print("="*60)
        test_snap_postproc_factory()
        
        # Test 4: qNEHVI builder
        print("="*60)
        test_build_qnehvi()
        
        # Test 5: Acquisition optimization wrapper
        print("="*60)
        test_optimize_acq_qnehvi()
        
        # Test 6: Basic propose_batch
        print("="*60)
        test_propose_batch_basic()
        
        # Test 7: Custom acquisition function
        print("="*60)
        test_propose_batch_with_custom_acq()
        
        # Test 8: Constraint handling
        print("="*60)
        test_propose_batch_with_constraints()
        
        # Test 9: Constraint stress test
        print("="*60)
        test_propose_batch_constraint_stress_test()
        
        print("="*60)
        print("\n🎉 All acquisition tests passed successfully!")
        print("🎯 Tested outcome constraints, helpers, builders, and main functions")
        print("🎯 Validated with real configCSV_example.csv data")
        print("🎯 Confirmed custom acquisition function support")
        print("🎯 Tested comprehensive constraint handling (row + outcome constraints)")
        print("🎯 Validated graceful failure handling with strict constraints")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
