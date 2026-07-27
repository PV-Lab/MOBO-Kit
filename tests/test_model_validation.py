from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import mobo_kit.model_validation as validation_module
import mobo_kit.models as models_module
from mobo_kit.model_validation import (
    CONSERVATIVE,
    DEFAULT_CURRENT,
    ModelFitCache,
    ModelFitError,
    ModelVariantSpec,
    compute_prediction_metrics,
    extract_model_hyperparameters,
    fit_model_variant,
    fit_warnings_frame,
    model_variant_spec,
    run_exact_loocv,
    validate_model_variant,
)
from mobo_kit.models import fit_gp_models


def _training_data() -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.5, 0.3],
            [0.8, 0.9],
            [1.0, 0.1],
        ],
        dtype=torch.double,
    )
    Y = torch.stack(
        (
            0.2 + 0.7 * X[:, 0] + 0.1 * X[:, 1],
            -1.0 + 0.3 * X[:, 0] - 0.5 * X[:, 1],
        ),
        dim=1,
    )
    return X, Y, (1, 2, 3, 4, 5)


def _skip_optimization(mll: object) -> object:
    return mll


def test_model_variant_contracts_are_fixed() -> None:
    assert model_variant_spec("default_current") is DEFAULT_CURRENT
    assert DEFAULT_CURRENT.min_noise == pytest.approx(1.0e-3)
    assert DEFAULT_CURRENT.min_lengthscale is None
    assert model_variant_spec("conservative") is CONSERVATIVE
    assert CONSERVATIVE.min_noise == pytest.approx(0.01)
    assert CONSERVATIVE.min_lengthscale == pytest.approx(0.05)

    with pytest.raises(ValueError, match="default_current"):
        ModelVariantSpec("default_current", min_noise=0.01, min_lengthscale=None)
    with pytest.raises(ValueError, match="conservative"):
        ModelVariantSpec("conservative", min_noise=0.01, min_lengthscale=0.01)
    with pytest.raises(ValueError, match="Unsupported"):
        model_variant_spec("mystery")


def test_default_current_matches_step2b_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", _skip_optimization)
    monkeypatch.setattr(models_module, "fit_gpytorch_mll", _skip_optimization)
    X, Y, sample_ids = _training_data()

    torch.manual_seed(73)
    strict = fit_model_variant(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=DEFAULT_CURRENT,
        seed=73,
    )
    torch.manual_seed(73)
    historical = fit_gp_models(X, Y)
    query = X[:3]
    with torch.no_grad():
        strict_posterior = strict.model.posterior(query)
        historical_posterior = historical.posterior(query)

    assert all(
        type(strict_gp.covar_module.base_kernel)
        is type(historical_gp.covar_module.base_kernel)
        for strict_gp, historical_gp in zip(strict.model.models, historical.models)
    )
    torch.testing.assert_close(strict_posterior.mean, historical_posterior.mean)
    torch.testing.assert_close(strict_posterior.variance, historical_posterior.variance)


def test_conservative_constraints_and_hyperparameter_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", _skip_optimization)
    X, Y, sample_ids = _training_data()
    record = fit_model_variant(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=CONSERVATIVE,
    )

    for gp in record.model.models:
        noise_floor = gp.likelihood.noise_covar.raw_noise_constraint.lower_bound
        lengthscale_floor = (
            gp.covar_module.base_kernel.raw_lengthscale_constraint.lower_bound
        )
        assert float(noise_floor) == pytest.approx(0.01)
        assert float(lengthscale_floor) == pytest.approx(0.05)
        assert float(gp.likelihood.noise.detach()) > 0.01
        assert torch.all(gp.covar_module.base_kernel.lengthscale > 0.05)

    parameters = extract_model_hyperparameters(
        record, input_names=("input_a", "input_b")
    )
    assert len(parameters) == 2
    assert all(row.configured_min_noise == 0.01 for row in parameters)
    assert all(row.configured_min_lengthscale == 0.05 for row in parameters)
    assert all(len(row.ard_lengthscales) == 2 for row in parameters)
    assert all(row.input_parameter_space == "normalized_0_1" for row in parameters)
    assert all(
        row.outcome_parameter_space == "standardized_internal" for row in parameters
    )
    flattened = pd.DataFrame(row.as_flat_dict() for row in parameters)
    assert {
        "ard_lengthscale_input_a",
        "ard_lengthscale_input_b",
        "likelihood_noise",
        "outputscale",
    } <= set(flattened.columns)


def test_lengthscale_flags_are_relative_to_normalized_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", _skip_optimization)
    X, Y, sample_ids = _training_data()
    record = fit_model_variant(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=DEFAULT_CURRENT,
    )
    for gp in record.model.models:
        gp.covar_module.base_kernel.lengthscale = torch.tensor(
            [[0.01, 20.0]], dtype=torch.double
        )

    parameters = extract_model_hyperparameters(
        record, input_names=("input_a", "input_b")
    )

    assert all(
        row.lengthscales_very_small_normalized_domain == (True, False)
        for row in parameters
    )
    assert all(
        row.lengthscales_extremely_large_flat == (False, True) for row in parameters
    )
    flattened = pd.DataFrame(row.as_flat_dict() for row in parameters)
    assert flattened["any_lengthscale_very_small_normalized_domain"].all()
    assert flattened["any_lengthscale_extremely_large_flat"].all()
    assert flattened["ard_lengthscale_input_a_very_small_normalized_domain"].all()
    assert flattened["ard_lengthscale_input_b_extremely_large_flat"].all()


def test_fit_failure_is_structured_and_never_printed_or_retried(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def fail_strictly(mll: object) -> object:
        nonlocal calls
        calls += 1
        warnings.warn("optimizer diagnostic", RuntimeWarning, stacklevel=2)
        raise RuntimeError("optimizer stopped")

    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", fail_strictly)
    X, Y, sample_ids = _training_data()

    with pytest.raises(ModelFitError) as captured:
        fit_model_variant(
            X,
            Y,
            sample_ids=sample_ids,
            objective_names=("one", "two"),
            variant=DEFAULT_CURRENT,
        )

    error = captured.value
    assert calls == 1
    assert error.stage == "optimize"
    assert error.objective_index == 0
    assert isinstance(error.cause, RuntimeError)
    assert any(
        row.warning_category == "RuntimeWarning"
        and row.message == "optimizer diagnostic"
        for row in error.fit_warnings
    )
    captured_output = capsys.readouterr()
    assert captured_output.out == ""
    assert captured_output.err == ""


def test_constructor_failure_retains_structured_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_during_construction(*args: object, **kwargs: object) -> object:
        del args, kwargs
        warnings.warn("constructor diagnostic", UserWarning, stacklevel=2)
        raise RuntimeError("constructor stopped")

    monkeypatch.setattr(
        validation_module, "_build_single_task_gp", fail_during_construction
    )
    X, Y, sample_ids = _training_data()
    with pytest.raises(ModelFitError) as captured:
        fit_model_variant(
            X,
            Y,
            sample_ids=sample_ids,
            objective_names=("one", "two"),
            variant=DEFAULT_CURRENT,
        )

    assert captured.value.stage == "construct"
    assert any(
        row.message == "constructor diagnostic"
        and row.warning_category == "UserWarning"
        for row in captured.value.fit_warnings
    )


def test_successful_fit_warnings_are_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warn_and_succeed(mll: object) -> object:
        warnings.warn("fit reached a bound", UserWarning, stacklevel=2)
        return mll

    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", warn_and_succeed)
    X, Y, sample_ids = _training_data()
    record = fit_model_variant(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=DEFAULT_CURRENT,
        fit_key="warning-test",
    )

    optimizer_warnings = [row for row in record.warnings if row.stage == "optimize"]
    assert len(optimizer_warnings) == 2
    assert all(row.warning_category == "UserWarning" for row in optimizer_warnings)
    assert all(row.fit_key == "warning-test" for row in optimizer_warnings)
    frame = fit_warnings_frame([record])
    assert frame.shape[0] >= 2
    assert {"stage", "warning_category", "message"} <= set(frame.columns)


def test_exact_loocv_retains_folds_uncertainty_roles_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_calls = 0

    def count_fit(mll: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        return mll

    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", count_fit)
    X, Y, sample_ids = _training_data()
    cache = ModelFitCache()
    first = run_exact_loocv(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=DEFAULT_CURRENT,
        seed=19,
        row_roles=("control", "r0", "r0", "r0", "r0"),
        control_sample_ids=(1,),
        cache=cache,
    )
    first_fit_calls = fit_calls
    repeat = run_exact_loocv(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=DEFAULT_CURRENT,
        seed=19,
        row_roles=("control", "r0", "r0", "r0", "r0"),
        control_sample_ids=(1,),
        cache=cache,
    )

    assert first_fit_calls == len(sample_ids) * Y.shape[1]
    assert fit_calls == first_fit_calls
    assert cache.hits == len(sample_ids)
    assert len(first.predictions) == len(sample_ids) * Y.shape[1]
    assert len(first.fold_records) == len(sample_ids)
    assert [row.omitted_sample_id for row in first.predictions] == [
        sample_id for sample_id in sample_ids for _ in range(Y.shape[1])
    ]
    assert sum(row.is_control for row in first.predictions) == Y.shape[1]
    assert {row.row_role for row in first.predictions if row.is_control} == {"control"}
    assert all(row.predictive_std >= row.latent_std for row in first.predictions)
    assert any(row.predictive_std > row.latent_std for row in first.predictions)
    assert all(np.isfinite(row.gaussian_nlpd) for row in first.predictions)
    for omitted_id, record in first.fold_records.items():
        assert omitted_id not in record.sample_ids
        assert len(record.sample_ids) == len(sample_ids) - 1
    pd.testing.assert_frame_equal(
        first.predictions_frame(), repeat.predictions_frame(), check_exact=False
    )
    assert len(first.metrics) == Y.shape[1]
    assert all(metric.prediction_count == len(sample_ids) for metric in first.metrics)

    run_exact_loocv(
        X,
        Y,
        sample_ids=sample_ids,
        objective_names=("one", "two"),
        variant=DEFAULT_CURRENT,
        seed=20,
        cache=cache,
    )
    assert fit_calls == first_fit_calls * 2


def test_validate_model_variant_extracts_full_and_fold_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation_module, "fit_gpytorch_mll", _skip_optimization)
    X, Y, sample_ids = _training_data()
    result = validate_model_variant(
        X,
        Y,
        sample_ids=sample_ids,
        input_names=("input_a", "input_b"),
        objective_names=("one", "two"),
        variant=CONSERVATIVE,
        seed=73,
        control_sample_ids=(1,),
    )

    assert result.full_fit.omitted_sample_id is None
    assert len(result.loocv.fold_records) == len(sample_ids)
    assert len(result.hyperparameters) == (len(sample_ids) + 1) * Y.shape[1]
    frame = result.hyperparameters_frame()
    assert set(frame["fit_key"]) == {
        "full",
        *{f"omit:int:{sample_id!r}" for sample_id in sample_ids},
    }
    assert np.allclose(frame["noise_constraint_lower_bound"], 0.01)
    assert np.allclose(frame["lengthscale_constraint_lower_bound"], 0.05)
    assert set(result.loocv.fold_records) == set(sample_ids)
    assert result.loocv.cache is not None


def test_prediction_metrics_match_hand_calculation() -> None:
    observed = np.array([0.0, 1.0])
    predicted = np.array([0.1, 0.9])
    uncertainty = np.array([0.2, 0.2])

    result = compute_prediction_metrics(
        observed,
        predicted,
        uncertainty,
        variant_name="hand",
        objective_index=2,
        objective_name="score",
    )

    expected_nlpd = 0.5 * np.log(2.0 * np.pi * 0.2**2) + 0.5 * 0.5**2
    assert result.prediction_count == 2
    assert result.mae == pytest.approx(0.1)
    assert result.rmse == pytest.approx(0.1)
    assert result.r_squared == pytest.approx(0.96)
    assert result.spearman_rank_correlation == pytest.approx(1.0)
    assert result.mean_signed_error == pytest.approx(0.0, abs=1e-15)
    assert result.median_absolute_error == pytest.approx(0.1)
    assert result.coverage_68_percent == 1.0
    assert result.coverage_95_percent == 1.0
    assert result.mean_standardized_residual == pytest.approx(0.0, abs=1e-15)
    assert result.maximum_absolute_standardized_residual == pytest.approx(0.5)
    assert result.mean_gaussian_nlpd == pytest.approx(expected_nlpd)
    assert "small N=2" in result.r_squared_warning


def test_prediction_metrics_report_undefined_r_squared_and_validate_uncertainty() -> (
    None
):
    constant = compute_prediction_metrics(
        [1.0, 1.0, 1.0],
        [0.9, 1.0, 1.1],
        [0.2, 0.2, 0.2],
    )
    assert np.isnan(constant.r_squared)
    assert np.isnan(constant.spearman_rank_correlation)
    assert "undefined" in constant.r_squared_warning

    with pytest.raises(ValueError, match="strictly positive"):
        compute_prediction_metrics([0.0], [0.0], [0.0])
