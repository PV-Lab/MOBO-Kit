# tests/smoke_test.py
import numpy as np
import torch
import pandas as pd

from src.utils import load_config, get_objective_names, load_csv, split_XY_from_cfg, np_to_torch, set_seeds
from src.design import build_input_spec_list, build_design
from src.data import y_minmax_np, x_normalizer_torch, x_denormalizer_np
from src.metrics import compute_ref_pareto_hv
from src.models import fit_gp_models
from src.acquisition import build_qnehvi, _make_snap_postproc, optimize_acq_qnehvi
# If you want row constraints, also:
# from src.constraints import constraints_from_config
# and later pass row_constraints into propose_qnehvi_batch instead of the lower-level calls.

CFG_PATH = "configs/example_inputs.yaml"
CSV_PATH = "data/processed/R0+R1 full results-1.csv"  # ← update if your file is named differently

def main():
    set_seeds(123)

    # 1) Config → Design
    cfg = load_config(CFG_PATH)
    specs = build_input_spec_list(cfg["inputs"])
    design = build_design(specs)
    obj_names = get_objective_names(cfg)
    print(f"[OK] D={len(design.names)} inputs, M={len(obj_names)} objectives")

    # 2) Load CSV → split X,Y (physical units)
    df = load_csv(CSV_PATH)
    X_np, Y_np = split_XY_from_cfg(df, design, cfg)
    print(f"[OK] CSV N={len(X_np)} rows")

    # 3) Scale Y to [0,1]; normalize X to [0,1]^D
    Y_scaled, Y_min, Y_max = y_minmax_np(Y_np, eps=1e-12)
    X_t = np_to_torch(X_np)
    Xn_t = x_normalizer_torch(X_t, design)
    Y_t = np_to_torch(Y_scaled)
    device, dtype = Xn_t.device, Xn_t.dtype
    print(f"[OK] Normalized X, scaled Y (device={device}, dtype={dtype})")

    # 4) Fit vanilla GP per objective
    model = fit_gp_models(Xn_t, Y_t)
    model = model.to(device=device, dtype=dtype)
    print("[OK] Fitted ModelListGP")

    # 5) Ref point + hypervolume on scaled space
    ref_point_t, pareto_Y_t, hv_val = compute_ref_pareto_hv(Y_t)
    print(f"[OK] HV={hv_val:.4f} with {pareto_Y_t.shape[0]} Pareto points")

    # 6) Build qNEHVI and propose q=5 (snapped in optimizer)
    acq = build_qnehvi(model=model, train_X=Xn_t, ref_point_t=ref_point_t, sample_shape=128)
    postproc = _make_snap_postproc(design)
    cand_norm_t, acq_val_t = optimize_acq_qnehvi(
        acq_function=acq, d=Xn_t.shape[1], q=5, num_restarts=5, raw_samples=256,
        device=device, dtype=dtype, options={"retry_on_optimization_warning": True},
        sequential=True, post_processing_func=postproc,
    )
    cand_norm = cand_norm_t.detach().cpu().numpy()
    X_phys = x_denormalizer_np(cand_norm, design)
    print("[OK] Proposed 5 candidates (physical). First row:", np.round(X_phys[0], 4))

if __name__ == "__main__":
    main()
