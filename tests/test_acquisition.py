# tests/test_acquisition.py
import torch
from src.acquisition import outcome_ge, outcome_le

def test_outcome_constraint_builders_shape_and_sign():
    S, B, q, m = 64, 2, 4, 3
    Y = torch.zeros(S, B, q, m)  # all zeros

    # ge: feasible when Y[..., 1] >= 0.8
    c_ge = outcome_ge(obj_idx=1, thresh=0.8)
    val_ge = c_ge(Y)
    assert val_ge.shape == (S, B, q)
    assert torch.all(val_ge > 0)  # 0 < 0.8 -> infeasible -> positive

    Y[..., 1] = 0.9
    assert torch.all(c_ge(Y) <= 0)  # now feasible -> <= 0

    # le: feasible when Y[..., 2] <= 0.15
    Y = torch.ones(S, B, q, m)  # all ones
    c_le = outcome_le(obj_idx=2, thresh=0.15)
    val_le = c_le(Y)
    assert val_le.shape == (S, B, q)
    assert torch.all(val_le > 0)  # 1.0 > 0.15 -> infeasible -> positive

    Y[..., 2] = 0.10
    assert torch.all(c_le(Y) <= 0)  # now feasible
