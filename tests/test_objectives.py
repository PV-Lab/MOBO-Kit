import math

import pytest
import torch

from mobo_kit.objectives import (
    BoundedMCMultiOutputObjective,
    BoundedPosteriorSampleTransform,
    ConfiguredMCMultiOutputObjective,
    ObjectiveSpec,
    ObjectiveTransform,
)


def _mixed_transform():
    return ObjectiveTransform(
        [
            ObjectiveSpec("already_utility", "maximize", "identity"),
            ObjectiveSpec(
                "maximize_raw",
                "maximize",
                "affine",
                lower_anchor=10.0,
                upper_anchor=20.0,
            ),
            ObjectiveSpec(
                "minimize_raw",
                "minimize",
                "affine",
                lower_anchor=0.0,
                upper_anchor=4.0,
            ),
            ObjectiveSpec(
                "target_raw",
                "target",
                "gaussian_target",
                target=650.0,
                sigma=100.0,
            ),
        ],
        version="TEST_ONLY-v1",
    )


def _identity_transform():
    return ObjectiveTransform(
        [
            ObjectiveSpec("uniformity", "maximize", "identity"),
            ObjectiveSpec("optoelectronic", "maximize", "identity"),
            ObjectiveSpec("thickness", "maximize", "identity"),
        ],
        version="TEST_IDENTITY-v1",
    )


def test_identity_affine_and_arbitrary_leading_dimensions():
    Y = torch.tensor(
        [
            [[[0.2, 10.0, 0.0, 650.0], [0.8, 20.0, 4.0, 750.0]]],
            [[[0.4, 15.0, 2.0, 550.0], [0.1, 25.0, -2.0, 650.0]]],
        ],
        dtype=torch.double,
    )
    result = _mixed_transform()(Y)
    assert result.shape == Y.shape
    assert result.dtype == Y.dtype
    assert result.device == Y.device
    assert torch.allclose(result[..., 0], Y[..., 0])
    assert result[0, 0, 0, 1].item() == pytest.approx(0.0)
    assert result[0, 0, 1, 1].item() == pytest.approx(1.0)
    assert result[0, 0, 0, 2].item() == pytest.approx(1.0)
    assert result[0, 0, 1, 2].item() == pytest.approx(0.0)


def test_affine_clip_is_explicit():
    transform = ObjectiveTransform(
        [
            ObjectiveSpec(
                "clipped",
                "maximize",
                "affine",
                lower_anchor=0,
                upper_anchor=1,
                clip=True,
            )
        ],
        version="TEST_ONLY-v1",
    )
    result = transform(torch.tensor([[-1.0], [0.4], [2.0]]))
    assert result[:, 0].tolist() == pytest.approx([0.0, 0.4, 1.0])
    with pytest.raises(ValueError, match="cannot enable clip"):
        ObjectiveSpec("identity", "maximize", "identity", clip=True)


def test_gaussian_target_value_symmetry_and_monotonicity():
    transform = ObjectiveTransform(
        [
            ObjectiveSpec(
                "thickness",
                "target",
                "gaussian_target",
                target=650,
                sigma=100,
            )
        ],
        version="TEST_ONLY-v1",
    )
    result = transform(
        torch.tensor([[650.0], [600.0], [700.0], [450.0]], dtype=torch.double)
    )[:, 0]
    assert result[0].item() == pytest.approx(1.0)
    assert result[1].item() == pytest.approx(result[2].item())
    assert result[1] < result[0]
    assert result[3] < result[1]


def test_negative_absolute_target_hand_calculation():
    transform = ObjectiveTransform(
        [
            ObjectiveSpec(
                "target",
                "target",
                "negative_absolute_target",
                target=10,
                scale=2,
            )
        ],
        version="TEST_ONLY-v1",
    )
    assert transform(torch.tensor([[8.0], [10.0], [13.0]]))[:, 0].tolist() == [
        -1.0,
        -0.0,
        -1.5,
    ]


def test_nonlinear_transform_is_applied_before_sample_mean():
    transform = ObjectiveTransform(
        [
            ObjectiveSpec(
                "target",
                "target",
                "gaussian_target",
                target=0,
                sigma=1,
            )
        ],
        version="TEST_ONLY-v1",
    )
    posterior_samples = torch.tensor([[[-1.0]], [[1.0]]])
    mean_after_transform = transform(posterior_samples).mean(dim=0)
    transform_of_mean = transform(posterior_samples.mean(dim=0))
    assert mean_after_transform.item() == pytest.approx(
        torch.exp(torch.tensor(-0.5)).item()
    )
    assert transform_of_mean.item() == pytest.approx(1.0)
    assert not torch.allclose(mean_after_transform, transform_of_mean)


def test_botorch_objective_matches_direct_transform():
    transform = _mixed_transform()
    objective = ConfiguredMCMultiOutputObjective(transform)
    samples = torch.tensor(
        [[[[0.2, 15.0, 2.0, 650.0], [0.8, 20.0, 0.0, 750.0]]]],
        dtype=torch.double,
    )
    assert torch.equal(objective(samples), transform(samples))


def test_bounded_posterior_sample_transform_is_explicit_and_non_mutating():
    transform = _identity_transform()
    bounded = BoundedPosteriorSampleTransform(
        transform,
        [(0.0, 1.0), (None, None), (0.0, 1.0)],
    )
    samples = torch.tensor(
        [
            [[[-0.2, -3.5, 1.2], [0.4, 2.1, 0.7]]],
            [[[1.4, 8.0, -0.1], [0.9, -1.2, 2.0]]],
        ],
        dtype=torch.double,
    )
    samples_before = samples.clone()
    utilities = bounded(samples)

    assert utilities.shape == samples.shape
    assert utilities.dtype == samples.dtype
    assert utilities.device == samples.device
    assert torch.equal(samples, samples_before)
    assert torch.all((utilities[..., 0] >= 0.0) & (utilities[..., 0] <= 1.0))
    assert torch.equal(utilities[..., 1], samples[..., 1])
    assert torch.all((utilities[..., 2] >= 0.0) & (utilities[..., 2] <= 1.0))
    assert bounded.bounds == ((0.0, 1.0), (None, None), (0.0, 1.0))
    assert bounded.version == "TEST_IDENTITY-v1+posterior-sample-bounds-v1"

    # The base contract remains unchanged for observed/training targets. Bounds
    # apply only when the acquisition-specific wrapper is explicitly invoked.
    training_targets = samples_before[0, 0].clone()
    training_targets_before = training_targets.clone()
    assert torch.equal(transform(training_targets), training_targets_before)
    assert torch.equal(training_targets, training_targets_before)


def test_bounded_botorch_objective_matches_wrapper_without_touching_reference():
    transform = _identity_transform()
    bounds = [(0.0, 1.0), (None, None), (0.0, 1.0)]
    objective = BoundedMCMultiOutputObjective(transform, bounds)
    samples = torch.tensor([[[-1.0, 2.5, 3.0]]], dtype=torch.float32)
    reference_point = torch.tensor([-0.1, -4.0, -0.2], dtype=torch.float32)
    reference_before = reference_point.clone()

    expected = BoundedPosteriorSampleTransform(transform, bounds)(samples)
    assert torch.equal(objective(samples), expected)
    assert torch.equal(reference_point, reference_before)
    assert objective.bounds == tuple(bounds)


@pytest.mark.parametrize(
    "bounds, match",
    [
        ([(0.0, 1.0)], "one .* pair per objective"),
        ([(1.0, 0.0), (None, None), (0.0, 1.0)], "must not exceed"),
        ([(None, None), (None, None), (None, None)], "At least one"),
        ([(False, 1.0), (None, None), (0.0, 1.0)], "non-boolean"),
        ([(0.0, float("inf")), (None, None), (0.0, 1.0)], "finite"),
    ],
)
def test_bounded_posterior_sample_contract_validation(bounds, match):
    with pytest.raises(ValueError, match=match):
        BoundedPosteriorSampleTransform(_identity_transform(), bounds)


def test_posterior_sample_bounds_reject_nonidentity_objectives():
    transform = ObjectiveTransform(
        [
            ObjectiveSpec(
                "scaled",
                "maximize",
                "affine",
                lower_anchor=0.0,
                upper_anchor=1.0,
            )
        ],
        version="TEST_AFFINE-v1",
    )
    with pytest.raises(ValueError, match="identity/maximize"):
        BoundedPosteriorSampleTransform(transform, [(0.0, 1.0)])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "", "goal": "maximize", "transform": "identity"}, "name"),
        ({"name": "x", "goal": "minimize", "transform": "identity"}, "only"),
        ({"name": "x", "goal": "target", "transform": "affine"}, "requires"),
        (
            {
                "name": "x",
                "goal": "maximize",
                "transform": "affine",
                "lower_anchor": 1,
                "upper_anchor": 1,
            },
            "lower_anchor < upper_anchor",
        ),
        (
            {
                "name": "x",
                "goal": "target",
                "transform": "gaussian_target",
                "target": 0,
                "sigma": 0,
            },
            "strictly positive",
        ),
        (
            {
                "name": "x",
                "goal": "maximize",
                "transform": "affine",
                "lower_anchor": False,
                "upper_anchor": 1,
            },
            "non-boolean",
        ),
        (
            {
                "name": "x",
                "goal": "maximize",
                "transform": "affine",
                "lower_anchor": "0",
                "upper_anchor": 1,
            },
            "non-boolean",
        ),
        (
            {
                "name": "x",
                "goal": "target",
                "transform": "negative_absolute_target",
                "target": 0,
                "scale": -1,
            },
            "strictly positive",
        ),
    ],
)
def test_invalid_objective_specs_fail(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ObjectiveSpec(**kwargs)


def test_wrong_dimension_integer_nonfinite_and_duplicate_names_fail():
    transform = ObjectiveTransform(
        [ObjectiveSpec("x", "maximize", "identity")], version="TEST_ONLY-v1"
    )
    with pytest.raises(ValueError, match="final dimension"):
        transform(torch.ones((2, 2)))
    with pytest.raises(TypeError, match="floating"):
        transform(torch.ones((2, 1), dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        transform(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="unique"):
        ObjectiveTransform(
            [
                ObjectiveSpec("x", "maximize", "identity"),
                ObjectiveSpec("x", "maximize", "identity"),
            ],
            version="TEST_ONLY-v1",
        )


# --------------------------------------------------------------------------- #
# expected utility under a Gaussian posterior
# --------------------------------------------------------------------------- #


def _mc_expected(spec, mu, var, *, draws=2_000_000, seed=0):
    """Monte-Carlo reference for E[transform(Y)], Y ~ N(mu, var)."""
    transform = ObjectiveTransform([spec], version="TEST_ONLY-v1")
    g = torch.Generator().manual_seed(seed)
    y = mu + math.sqrt(var) * torch.randn(draws, 1, generator=g, dtype=torch.double)
    return float(transform(y).mean())


@pytest.mark.parametrize("mu", [400.0, 650.0, 900.0, 1300.0])
@pytest.mark.parametrize("var", [0.0, 20.0**2, 100.0**2, 300.0**2])
def test_gaussian_target_expected_matches_monte_carlo(mu, var):
    """The closed form must agree with sampling, including at variance zero."""
    # workbook uses exp(-((T-650)/250)^2), i.e. sigma = 250/sqrt(2) here
    spec = ObjectiveSpec(
        "thickness",
        "target",
        "gaussian_target",
        target=650.0,
        sigma=250.0 / math.sqrt(2.0),
    )
    transform = ObjectiveTransform([spec], version="TEST_ONLY-v1")
    got = float(
        transform.expected_transform(
            torch.tensor([[mu]], dtype=torch.double),
            torch.tensor([[var]], dtype=torch.double),
        )
    )
    if var == 0.0:
        assert got == pytest.approx(
            float(transform(torch.tensor([[mu]], dtype=torch.double))), rel=1e-12
        )
    else:
        assert got == pytest.approx(_mc_expected(spec, mu, var), abs=1e-3)


def test_gaussian_target_expected_penalises_uncertainty_at_the_target():
    """Identical predicted mean, wider posterior, strictly lower expected utility."""
    spec = ObjectiveSpec(
        "thickness",
        "target",
        "gaussian_target",
        target=650.0,
        sigma=250.0 / math.sqrt(2.0),
    )
    transform = ObjectiveTransform([spec], version="TEST_ONLY-v1")
    mean = torch.full((3, 1), 650.0, dtype=torch.double)
    var = torch.tensor([[20.0], [100.0], [300.0]], dtype=torch.double) ** 2
    scores = transform.expected_transform(mean, var).flatten().tolist()
    assert scores[0] > scores[1] > scores[2]
    assert scores == pytest.approx([0.993661, 0.870388, 0.507673], abs=1e-5)


def test_negative_absolute_target_expected_matches_monte_carlo():
    spec = ObjectiveSpec(
        "t", "target", "negative_absolute_target", target=650.0, scale=250.0
    )
    transform = ObjectiveTransform([spec], version="TEST_ONLY-v1")
    for mu, var in ((650.0, 100.0**2), (400.0, 50.0**2), (900.0, 300.0**2)):
        got = float(
            transform.expected_transform(
                torch.tensor([[mu]], dtype=torch.double),
                torch.tensor([[var]], dtype=torch.double),
            )
        )
        assert got == pytest.approx(_mc_expected(spec, mu, var), abs=2e-3)


def test_linear_transforms_expectation_equals_transform_of_mean():
    transform = ObjectiveTransform(
        [
            ObjectiveSpec("a", "maximize", "identity"),
            ObjectiveSpec(
                "b", "maximize", "affine", lower_anchor=0.0, upper_anchor=2.0
            ),
        ],
        version="TEST_ONLY-v1",
    )
    mean = torch.tensor([[0.3, 1.1]], dtype=torch.double)
    var = torch.tensor([[4.0, 9.0]], dtype=torch.double)
    torch.testing.assert_close(transform.expected_transform(mean, var), transform(mean))


def test_expected_transform_rejects_bad_input():
    transform = ObjectiveTransform(
        [ObjectiveSpec("x", "maximize", "identity")], version="TEST_ONLY-v1"
    )
    ok = torch.ones((2, 1), dtype=torch.double)
    with pytest.raises(ValueError, match="share a shape"):
        transform.expected_transform(ok, torch.ones((3, 1), dtype=torch.double))
    with pytest.raises(ValueError, match="non-negative"):
        transform.expected_transform(ok, -ok)
    with pytest.raises(ValueError, match="final dimension"):
        transform.expected_transform(
            torch.ones((2, 2), dtype=torch.double),
            torch.ones((2, 2), dtype=torch.double),
        )


# --------------------------------------------------------------------------- #
# lognormal expectation (GP fitted in log space)
# --------------------------------------------------------------------------- #


def _thickness_utility():
    return ObjectiveTransform(
        [
            ObjectiveSpec(
                "thickness",
                "target",
                "gaussian_target",
                target=650.0,
                sigma=250.0 / math.sqrt(2.0),
            )
        ],
        version="TEST_ONLY-v1",
    )


@pytest.mark.parametrize("median_nm", [500.0, 700.0, 900.0])
@pytest.mark.parametrize("s_log", [0.10, 0.20, 0.40])
def test_lognormal_expectation_matches_monte_carlo(median_nm, s_log):
    transform = _thickness_utility()
    m = math.log(median_nm)
    got = float(
        transform.expected_transform_lognormal(
            torch.tensor([[m]], dtype=torch.double),
            torch.tensor([[s_log**2]], dtype=torch.double),
        )
    )
    g = torch.Generator().manual_seed(0)
    z = m + s_log * torch.randn(2_000_000, 1, generator=g, dtype=torch.double)
    mc = float(transform(torch.exp(z)).mean())
    assert got == pytest.approx(mc, abs=1e-3)


def test_lognormal_expectation_beats_moment_matching_by_orders_of_magnitude():
    """Moment-matching a lognormal to a Gaussian and reusing the closed form is
    an approximation whose error is large enough to reorder candidates."""
    transform = _thickness_utility()
    m, s = math.log(700.0), 0.40
    gh = float(
        transform.expected_transform_lognormal(
            torch.tensor([[m]], dtype=torch.double),
            torch.tensor([[s**2]], dtype=torch.double),
        )
    )
    mu = math.exp(m + s * s / 2.0)
    var = (math.exp(s * s) - 1.0) * math.exp(2 * m + s * s)
    mm = float(
        transform.expected_transform(
            torch.tensor([[mu]], dtype=torch.double),
            torch.tensor([[var]], dtype=torch.double),
        )
    )
    g = torch.Generator().manual_seed(0)
    z = m + s * torch.randn(2_000_000, 1, generator=g, dtype=torch.double)
    mc = float(transform(torch.exp(z)).mean())
    assert abs(gh - mc) < 1e-3
    assert abs(mm - mc) > 50 * abs(gh - mc)


def test_moment_matching_error_changes_sign_across_the_range():
    """The reason moment-matching is not merely a constant offset: the bias
    flips sign, so it permutes the candidate ordering."""
    transform = _thickness_utility()
    s = 0.22  # the campaign's measured posterior width in log space
    signed = []
    for median_nm in (550.0, 850.0):
        m = math.log(median_nm)
        gh = float(
            transform.expected_transform_lognormal(
                torch.tensor([[m]], dtype=torch.double),
                torch.tensor([[s**2]], dtype=torch.double),
            )
        )
        mu = math.exp(m + s * s / 2.0)
        var = (math.exp(s * s) - 1.0) * math.exp(2 * m + s * s)
        mm = float(
            transform.expected_transform(
                torch.tensor([[mu]], dtype=torch.double),
                torch.tensor([[var]], dtype=torch.double),
            )
        )
        signed.append(mm - gh)
    assert signed[0] > 0 > signed[1], f"expected a sign change, got {signed}"


def test_lognormal_expectation_degenerates_to_the_plain_transform():
    transform = _thickness_utility()
    m = torch.tensor([[math.log(650.0)]], dtype=torch.double)
    got = float(transform.expected_transform_lognormal(m, torch.zeros_like(m)))
    assert got == pytest.approx(1.0, abs=1e-9)


def test_lognormal_expectation_rejects_bad_input():
    transform = _thickness_utility()
    ok = torch.zeros((2, 1), dtype=torch.double)
    with pytest.raises(ValueError, match="share a shape"):
        transform.expected_transform_lognormal(
            ok, torch.zeros((3, 1), dtype=torch.double)
        )
    with pytest.raises(ValueError, match="non-negative"):
        transform.expected_transform_lognormal(ok, ok - 1.0)
    with pytest.raises(ValueError, match="nodes"):
        transform.expected_transform_lognormal(ok, ok, nodes=1)


# --------------------------------------------------------------------------- #
# model_link: the two acquisition paths must agree
# --------------------------------------------------------------------------- #


def _log_link_contract():
    """The campaign's shape: two identity-link objectives and one log-link."""
    return ObjectiveTransform(
        [
            ObjectiveSpec(
                "uniformity", "maximize", "affine", lower_anchor=0.0, upper_anchor=1.0
            ),
            ObjectiveSpec(
                "optoelectronic",
                "maximize",
                "affine",
                lower_anchor=-10.0,
                upper_anchor=-6.0,
            ),
            ObjectiveSpec(
                "thickness",
                "target",
                "gaussian_target",
                model_link="log",
                target=650.0,
                sigma=250.0 / math.sqrt(2.0),
            ),
        ],
        version="TEST_ONLY-v1",
    )


def test_log_link_decodes_exactly_once():
    """The trap: quadrature and MC both route through one link decode. If either
    exponentiates separately the utility is computed on exp(exp(x))."""
    transform = _log_link_contract()
    nm = 687.0
    model_output = torch.tensor([[0.5, -8.0, math.log(nm)]], dtype=torch.double)
    got = transform(model_output)[0, 2].item()
    expected = math.exp(-(((nm - 650.0) / 250.0) ** 2))
    assert got == pytest.approx(expected, abs=1e-12)


def test_identity_link_objectives_are_untouched_by_the_link_machinery():
    transform = _log_link_contract()
    model_output = torch.tensor([[0.5, -8.0, math.log(650.0)]], dtype=torch.double)
    utilities = transform(model_output)
    assert utilities[0, 0].item() == pytest.approx(0.5)
    assert utilities[0, 1].item() == pytest.approx((-8.0 + 10.0) / 4.0)


def test_ucb_and_qlognehvi_paths_agree_on_the_same_posterior():
    """The guard that matters operationally.

    UCB reads analytic utility moments; qLogNEHVI transforms posterior samples.
    Both are correct, and nothing else forces them to match. If they drift, the
    two rounds silently optimise different objectives and it surfaces only as R1
    and R2 disagreeing for reasons nobody can trace.
    """
    transform = _log_link_contract()
    mean = torch.tensor(
        [[0.4, -8.2, math.log(700.0)], [0.6, -7.5, math.log(480.0)]],
        dtype=torch.double,
    )
    variance = torch.tensor(
        [[0.01, 0.04, 0.22**2], [0.02, 0.09, 0.30**2]], dtype=torch.double
    )

    analytic = transform.expected_transform(mean, variance)

    # the sampling path: draw in MODEL space, transform, average
    g = torch.Generator().manual_seed(0)
    draws = 400_000
    samples = mean.unsqueeze(0) + variance.sqrt().unsqueeze(0) * torch.randn(
        (draws, *mean.shape), generator=g, dtype=torch.double
    )
    sampled = transform(samples).mean(dim=0)

    # MC standard error over this many draws is ~1e-3
    torch.testing.assert_close(analytic, sampled, atol=4e-3, rtol=0.0)


def test_expected_transform_dispatches_per_objective_not_globally():
    """A contract mixing links must not apply one rule to every column."""
    transform = _log_link_contract()
    mean = torch.tensor([[0.4, -8.0, math.log(650.0)]], dtype=torch.double)
    zero = torch.zeros_like(mean)

    # at zero variance every path collapses to the plain transform
    torch.testing.assert_close(
        transform.expected_transform(mean, zero), transform(mean)
    )

    # widening only the log-link column must move only that utility
    widened = zero.clone()
    widened[0, 2] = 0.30**2
    expected = transform.expected_transform(mean, widened)
    baseline = transform(mean)
    assert expected[0, 0].item() == pytest.approx(baseline[0, 0].item())
    assert expected[0, 1].item() == pytest.approx(baseline[0, 1].item())
    assert expected[0, 2].item() < baseline[0, 2].item()
