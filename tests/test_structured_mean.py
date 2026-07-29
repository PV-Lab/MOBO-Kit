from __future__ import annotations

import numpy as np
import pytest

from mobo_kit.campaign import load_campaign_config
from mobo_kit.structured_mean import (
    MeanFeature,
    StructuredMeanSpec,
    apply_structured_mean,
    fit_structured_mean,
    mean_spec_from_config,
)

NAMES = ["speed_1", "time_1", "precur_conc", "anneal_temp"]


def _X(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack(
        [
            rng.uniform(1000, 6000, n),
            rng.uniform(5, 50, n),
            rng.uniform(1.0, 2.0, n),
            rng.uniform(100, 185, n),
        ]
    )


def test_log_response_recovers_a_power_law_exactly() -> None:
    """T = c * speed^a * conc^b is linear in log-log, so the mean should absorb
    it completely and leave zero residual."""
    X = _X()
    T = 5.0e5 * X[:, 0] ** -0.4 * X[:, 2] ** 1.3
    spec = StructuredMeanSpec(
        response="log",
        features=(MeanFeature("speed_1", "log"), MeanFeature("precur_conc", "log")),
    )
    coefficients, residual = fit_structured_mean(X, T, spec, NAMES)
    assert np.abs(residual).max() < 1e-9
    assert coefficients[1] == pytest.approx(-0.4, abs=1e-6)
    assert coefficients[2] == pytest.approx(1.3, abs=1e-6)


def test_identity_response_recovers_a_linear_trend() -> None:
    X = _X()
    y = 3.0 - 0.05 * X[:, 3]
    spec = StructuredMeanSpec("identity", (MeanFeature("anneal_temp"),))
    coefficients, residual = fit_structured_mean(X, y, spec, NAMES)
    assert np.abs(residual).max() < 1e-9
    assert coefficients[1] == pytest.approx(-0.05, abs=1e-9)


def test_trend_round_trips_through_apply() -> None:
    X = _X()
    y = 2.0 + 0.01 * X[:, 3]
    spec = StructuredMeanSpec("identity", (MeanFeature("anneal_temp"),))
    coefficients, residual = fit_structured_mean(X, y, spec, NAMES)
    post = apply_structured_mean(
        coefficients, X, spec, NAMES, residual, np.zeros(len(X))
    )
    np.testing.assert_allclose(post.mean, y, atol=1e-9)
    assert post.link == "identity"


def test_log_response_reports_its_link() -> None:
    """The link is what tells the utility layer to integrate a lognormal instead
    of using the Gaussian closed form."""
    X = _X()
    spec = StructuredMeanSpec("log", (MeanFeature("speed_1", "log"),))
    coefficients, residual = fit_structured_mean(X, np.full(len(X), 700.0), spec, NAMES)
    post = apply_structured_mean(
        coefficients, X, spec, NAMES, residual, np.zeros(len(X))
    )
    assert post.link == "log"
    np.testing.assert_allclose(np.exp(post.mean), 700.0, atol=1e-8)


def test_coefficients_come_only_from_the_rows_passed_in() -> None:
    """Guards the leakage property: fitting on a subset must not see the rest."""
    X = _X(14)
    y = 2.0 + 0.01 * X[:, 3]
    spec = StructuredMeanSpec("identity", (MeanFeature("anneal_temp"),))
    keep = list(range(13))
    a, _ = fit_structured_mean(X[keep], y[keep], spec, NAMES)
    b, _ = fit_structured_mean(X[keep], y[keep] * 1.0, spec, NAMES)
    np.testing.assert_allclose(a, b)
    # perturbing the held-out row must not change the fitted trend
    y_perturbed = y.copy()
    y_perturbed[13] += 1000.0
    c, _ = fit_structured_mean(X[keep], y_perturbed[keep], spec, NAMES)
    np.testing.assert_allclose(a, c)


def test_log_response_rejects_non_positive_observations() -> None:
    X = _X()
    spec = StructuredMeanSpec("log", (MeanFeature("speed_1", "log"),))
    y = np.full(len(X), 1.0)
    y[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_structured_mean(X, y, spec, NAMES)


def test_unknown_feature_column_is_rejected() -> None:
    spec = StructuredMeanSpec("identity", (MeanFeature("not_an_input"),))
    with pytest.raises(ValueError, match="not a declared input"):
        fit_structured_mean(_X(), np.ones(12), spec, NAMES)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"response": "sqrt", "features": (MeanFeature("speed_1"),)}, "response link"),
        ({"response": "identity", "features": ()}, "at least one feature"),
        (
            {
                "response": "identity",
                "features": (MeanFeature("speed_1"), MeanFeature("speed_1")),
            },
            "unique",
        ),
    ],
)
def test_invalid_specs_are_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        StructuredMeanSpec(**kwargs)


# --------------------------------------------------------------------------- #
# the campaign's declared shapes
# --------------------------------------------------------------------------- #


def test_campaign_declares_the_two_measured_mean_functions() -> None:
    """Opposite shapes by design: thickness needs a pair of log terms,
    optoelectronic needs exactly one linear term. Neither generalises."""
    config = load_campaign_config("configs/FA0.9CS0.1PbI3_260407_Config.yaml")
    by_name = {s["name"]: s for s in config["objectives"]["specs"]}

    assert mean_spec_from_config(by_name["uniformity"]) is None

    opto = mean_spec_from_config(by_name["optoelectronic"])
    assert opto.response == "identity"
    assert [f.column for f in opto.features] == ["anneal_temp"]
    assert [f.transform for f in opto.features] == ["identity"]

    thickness = mean_spec_from_config(by_name["thickness"])
    assert thickness.response == "log"
    assert [f.column for f in thickness.features] == ["speed_1", "precur_conc"]
    assert [f.transform for f in thickness.features] == ["log", "log"]
