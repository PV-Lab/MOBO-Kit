"""CPU-fast regression tests for the active GP model helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior

import mobo_kit.models as models_module
from mobo_kit.models import (
    default_kernel_options,
    default_noise_options,
    fit_gp_models,
    posterior_report,
)


def _training_data():
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.3],
            [0.6, 0.9],
            [0.8, 0.2],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    train_y = torch.stack(
        (
            train_x[:, 0] + 0.5 * train_x[:, 1],
            1.0 - train_x[:, 0].square() + train_x[:, 1],
        ),
        dim=1,
    )
    return train_x, train_y


def test_default_model_options_build_expected_active_types():
    kernel_factories = default_kernel_options()
    kernels = [factory(2) for factory in kernel_factories]
    noise_options = default_noise_options(torch.device("cpu"))

    assert len(kernels) == 4
    assert all(isinstance(kernel, Kernel) for kernel in kernels)
    assert all(kernel.ard_num_dims == 2 for kernel in kernels)
    assert noise_options[0] is None
    assert all(option is None or isinstance(option, Prior) for option in noise_options)


def test_fit_gp_models_and_posterior_report_have_multioutput_shapes(monkeypatch):
    fit_calls = []

    def skip_hyperparameter_optimization(mll):
        fit_calls.append(mll)
        return mll

    monkeypatch.setattr(
        models_module, "fit_gpytorch_mll", skip_hyperparameter_optimization
    )
    train_x, train_y = _training_data()

    model = fit_gp_models(train_x, train_y)
    pred_mean, pred_std = posterior_report(model, train_x[:3])

    assert isinstance(model, ModelListGP)
    assert len(model.models) == train_y.shape[1]
    assert len(fit_calls) == train_y.shape[1]
    assert pred_mean.shape == (3, 2)
    assert pred_std.shape == (3, 2)
    assert np.isfinite(pred_mean).all()
    assert np.isfinite(pred_std).all()
    assert (pred_std >= 0.0).all()
    assert all(next(gp.parameters()).device.type == "cpu" for gp in model.models)


def test_fit_gp_models_rejects_mismatched_per_objective_options(monkeypatch):
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", lambda mll: mll)
    train_x, train_y = _training_data()

    with pytest.raises(ValueError, match="kernel_fn list length"):
        fit_gp_models(train_x, train_y, kernel_fn=[default_kernel_options()[0]])

    with pytest.raises(ValueError, match="noise_priors list length"):
        fit_gp_models(train_x, train_y, noise_priors=[None])
