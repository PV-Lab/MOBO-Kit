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
