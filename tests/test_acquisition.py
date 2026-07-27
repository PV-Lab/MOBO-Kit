"""Fast smoke tests for the active acquisition API."""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize

import mobo_kit.acquisition as acquisition
from mobo_kit.acquisition import (
    _make_snap_postproc,
    _unit_bounds,
    build_qnehvi,
    optimize_acq_qnehvi,
    outcome_ge,
    outcome_ge_standardized,
    outcome_le,
    outcome_le_standardized,
    propose_batch,
)
from mobo_kit.design import InputSpec, build_design


def _design():
    return build_design(
        [
            InputSpec("temperature", 10.0, 20.0, 5.0, unit="C"),
            InputSpec("ratio", 0.0, 1.0, 0.25),
        ]
    )


def _unfitted_two_objective_model(train_x: torch.Tensor) -> ModelListGP:
    y_1 = (train_x[:, :1] + 0.5 * train_x[:, 1:2]).square()
    y_2 = 1.0 - 0.25 * train_x[:, :1] + train_x[:, 1:2]
    models = [
        SingleTaskGP(train_x, y, outcome_transform=Standardize(m=1)) for y in (y_1, y_2)
    ]
    return ModelListGP(*models)


def test_unit_bounds_and_outcome_constraint_signs():
    bounds = _unit_bounds(3, torch.device("cpu"), torch.float64)

    assert bounds.shape == (2, 3)
    assert bounds.device.type == "cpu"
    assert bounds.dtype == torch.float64
    torch.testing.assert_close(bounds[0], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(bounds[1], torch.ones(3, dtype=torch.float64))

    outcomes = torch.tensor([[0.4, 1.2], [0.8, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(
        outcome_ge(0, 0.5)(outcomes),
        torch.tensor([0.1, -0.3], dtype=torch.float64),
    )
    torch.testing.assert_close(
        outcome_le(1, 1.0)(outcomes),
        torch.tensor([0.2, -0.5], dtype=torch.float64),
    )

    standardized = torch.tensor([[1.0], [2.5]], dtype=torch.float64)
    torch.testing.assert_close(
        outcome_ge_standardized(0, 14.0, 10.0, 2.0)(standardized),
        torch.tensor([1.0, -0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        outcome_le_standardized(0, 14.0, 10.0, 2.0)(standardized),
        torch.tensor([-1.0, 0.5], dtype=torch.float64),
    )


def test_snap_postprocessor_preserves_shape_dtype_and_grid():
    postprocess = _make_snap_postproc(_design())
    candidates = torch.tensor([[[0.15, 0.15], [0.84, 0.90]]], dtype=torch.float64)

    snapped = postprocess(candidates)

    assert snapped.shape == candidates.shape
    assert snapped.dtype == candidates.dtype
    assert snapped.device == candidates.device
    torch.testing.assert_close(
        snapped,
        torch.tensor([[[0.0, 0.25], [1.0, 1.0]]], dtype=torch.float64),
    )


def test_build_qnehvi_constructs_active_botorch_acquisition():
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.3],
            [0.7, 1.0],
            [1.0, 0.5],
        ],
        dtype=torch.float64,
    )
    model = _unfitted_two_objective_model(train_x)

    acq = build_qnehvi(
        model=model,
        train_X=train_x,
        ref_point_t=torch.tensor([-0.5, -0.5], dtype=torch.float64),
        sample_shape=8,
        prune_baseline=False,
    )

    assert isinstance(acq, qLogNoisyExpectedHypervolumeImprovement)
    assert acq.sampler.sample_shape == torch.Size([8])


def test_optimize_wrapper_passes_cpu_bounds_and_options(monkeypatch):
    received = {}

    def fake_optimize_acqf(**kwargs):
        received.update(kwargs)
        bounds = kwargs["bounds"]
        q = kwargs["q"]
        candidates = bounds.mean(dim=0).repeat(q, 1)
        values = torch.arange(q, device=bounds.device, dtype=bounds.dtype)
        return candidates, values

    monkeypatch.setattr(acquisition, "optimize_acqf", fake_optimize_acqf)
    options = {"maxiter": 2}

    candidates, values = optimize_acq_qnehvi(
        acq_function=object(),
        d=2,
        q=3,
        num_restarts=4,
        raw_samples=16,
        device=torch.device("cpu"),
        dtype=torch.float64,
        options=options,
        sequential=False,
    )

    assert candidates.shape == (3, 2)
    assert values.shape == (3,)
    assert received["num_restarts"] == 4
    assert received["raw_samples"] == 16
    assert received["options"] is options
    assert received["sequential"] is False
    torch.testing.assert_close(
        received["bounds"],
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
    )


def test_propose_batch_returns_snapped_physical_and_normalized_arrays(monkeypatch):
    class DummyAcquisition:
        def __init__(self):
            self.pending_calls = []

        def set_X_pending(self, value):
            self.pending_calls.append(value)

    dummy_acquisition = DummyAcquisition()

    def fake_optimize(**kwargs):
        raw = torch.tensor(
            [[0.15, 0.15], [0.84, 0.90]],
            dtype=kwargs["dtype"],
            device=kwargs["device"],
        )
        candidates = kwargs["post_processing_func"](raw)
        values = torch.tensor(
            [1.5, 1.0], dtype=kwargs["dtype"], device=kwargs["device"]
        )
        return candidates, values

    monkeypatch.setattr(acquisition, "optimize_acq_qnehvi", fake_optimize)
    train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    result = propose_batch(
        design=_design(),
        model=None,
        train_X=train_x,
        ref_point_t=torch.tensor([-1.0, -1.0], dtype=torch.float64),
        batch_size=2,
        acq=lambda: dummy_acquisition,
        max_attempts=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert set(result) == {"X_phys", "X_norm", "acq_val", "attempts"}
    np.testing.assert_allclose(result["X_norm"], [[0.0, 0.25], [1.0, 1.0]])
    np.testing.assert_allclose(result["X_phys"], [[10.0, 0.25], [20.0, 1.0]])
    np.testing.assert_allclose(result["acq_val"], [1.5, 1.0])
    assert result["attempts"] == 1
    assert dummy_acquisition.pending_calls == [None]
