import numpy as np
import pytest
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize

from mobo_kit.ucb_hvi import (
    UCBHVIScoreResult,
    apply_ucb_bound_policy,
    hypervolume_improvement_scores,
    posterior_identity_moments,
    posterior_utility_moments,
    propose_ucb_hvi_batch,
    score_ucb_hvi_from_moments,
    score_ucb_hvi_pool,
)
from mobo_kit.batch_selection import LocalPenalizationConfig, UndersizedBatchError
from mobo_kit.candidate_pool import CandidatePool
from mobo_kit.objectives import ObjectiveSpec, ObjectiveTransform
import mobo_kit.ucb_hvi as module


class IdentityTransform:
    def transform(self, Y: torch.Tensor) -> torch.Tensor:
        return Y


def _identity_contract(count=1):
    return ObjectiveTransform(
        [
            ObjectiveSpec(f"objective_{index}", "maximize", "identity")
            for index in range(count)
        ],
        version="TEST-IDENTITY-v1",
    )


class _ExactPosteriorModel:
    def posterior(self, X, *, observation_noise):
        mean = torch.stack((X[..., 0] + 0.25, 1.5 - X[..., 0]), dim=-1)
        variance = torch.full_like(mean, 0.09 if not observation_noise else 0.16)
        return type("Posterior", (), {"mean": mean, "variance": variance})()


def _observed_2d():
    return np.array([[0.4, 0.8], [0.8, 0.4], [0.2, 0.2]])


def test_beta_zero_and_uncertainty_definition():
    means = np.array([[0.7, 0.7], [0.6, 0.6]])
    std = np.array([[0.2, 0.1], [0.0, 0.0]])
    zero = score_ucb_hvi_from_moments(
        means, std, _observed_2d(), np.array([0.0, 0.0]), beta=0.0
    )
    assert np.allclose(zero.utility_ucb, means)
    positive = score_ucb_hvi_from_moments(
        means, std, _observed_2d(), np.array([0.0, 0.0]), beta=4.0
    )
    assert positive.kappa == pytest.approx(2.0)
    assert np.allclose(positive.utility_ucb, means + 2.0 * std)
    assert positive.base_score[0] > zero.base_score[0]
    with pytest.raises(ValueError, match="beta"):
        score_ucb_hvi_from_moments(
            means, std, _observed_2d(), np.array([0.0, 0.0]), beta=-0.1
        )
    with pytest.raises(ValueError, match="non-boolean"):
        score_ucb_hvi_from_moments(
            means, std, _observed_2d(), np.array([0.0, 0.0]), beta=True
        )


def test_hvi_matches_hand_calculation_and_dominated_is_true_zero():
    scores, baseline, pareto, _ = hypervolume_improvement_scores(
        np.array([[0.7, 0.7], [0.3, 0.3]]),
        _observed_2d(),
        np.array([0.0, 0.0]),
    )
    assert baseline == pytest.approx(0.48)
    # Added area: (0.7-0.4)*(0.7-0.4) = 0.09.
    assert scores[0] == pytest.approx(0.09)
    assert scores[1] == 0.0
    assert pareto.shape == (2, 2)


def test_three_objective_hvi_and_chunk_invariance():
    observed = np.array([[0.5, 0.5, 0.5]])
    candidates = np.array([[0.6, 0.6, 0.6], [0.4, 0.4, 0.4]])
    chunked = hypervolume_improvement_scores(
        candidates, observed, np.zeros(3), chunk_size=1
    )
    whole = hypervolume_improvement_scores(
        candidates, observed, np.zeros(3), chunk_size=20
    )
    assert np.allclose(chunked[0], whole[0])
    assert chunked[0][0] == pytest.approx(0.6**3 - 0.5**3)
    assert chunked[0][1] == 0.0


def test_dominated_observed_rows_do_not_change_scores():
    candidates = np.array([[0.7, 0.7]])
    with_dominated = hypervolume_improvement_scores(
        candidates, _observed_2d(), np.zeros(2)
    )[0]
    without_dominated = hypervolume_improvement_scores(
        candidates, _observed_2d()[:2], np.zeros(2)
    )[0]
    assert np.allclose(with_dominated, without_dominated)


@pytest.mark.parametrize(
    "reference, message",
    [(None, "required"), (np.zeros(3), "shape"), (np.array([0.0, np.nan]), "finite")],
)
def test_reference_point_validation(reference, message):
    with pytest.raises(ValueError, match=message):
        hypervolume_improvement_scores(
            np.array([[0.7, 0.7]]), _observed_2d(), reference
        )


def test_posterior_utility_moments_are_seeded_and_chunk_invariant():
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    model.eval()
    pool = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    one = posterior_utility_moments(
        model,
        pool,
        IdentityTransform(),
        mc_samples=16,
        seed=13,
        chunk_size=1,
    )
    all_at_once = posterior_utility_moments(
        model,
        pool,
        IdentityTransform(),
        mc_samples=16,
        seed=13,
        chunk_size=100,
    )
    repeat = posterior_utility_moments(
        model,
        pool,
        IdentityTransform(),
        mc_samples=16,
        seed=13,
        chunk_size=2,
    )
    assert np.allclose(one.utility_mean, all_at_once.utility_mean)
    assert np.allclose(one.utility_std, all_at_once.utility_std)
    assert np.allclose(one.utility_mean, repeat.utility_mean)
    assert np.allclose(one.utility_std, repeat.utility_std)
    assert np.all(one.utility_std > 0)


def test_analytic_identity_moments_are_exact_chunk_dtype_and_device_invariant():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = torch.linspace(0.1, 0.9, 7, dtype=torch.float32, device=device).unsqueeze(-1)
    model = _ExactPosteriorModel()
    contract = _identity_contract(2)

    one = posterior_identity_moments(
        model, pool, contract, chunk_size=1, observation_noise=False
    )
    whole = posterior_identity_moments(
        model, pool, contract, chunk_size=100, observation_noise=False
    )
    expected = model.posterior(pool, observation_noise=False)

    assert one.moment_method == "analytic_identity"
    assert one.objective_contract_version == "TEST-IDENTITY-v1"
    assert one.utility_mean.dtype == pool.dtype
    assert one.utility_mean.device == pool.device
    assert one.utility_std.dtype == pool.dtype
    assert one.utility_std.device == pool.device
    assert torch.equal(one.utility_mean, expected.mean)
    assert torch.equal(one.utility_std, expected.variance.sqrt())
    assert torch.equal(one.utility_mean, whole.utility_mean)
    assert torch.equal(one.utility_std, whole.utility_std)


def test_analytic_identity_moments_reject_nonidentity_or_implicit_contracts():
    pool = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    nonidentity = ObjectiveTransform(
        [
            ObjectiveSpec(
                "scaled",
                "maximize",
                "affine",
                lower_anchor=0.0,
                upper_anchor=1.0,
            )
        ],
        version="TEST-NONLINEAR-v1",
    )
    with pytest.raises(ValueError, match="every objective.*identity"):
        posterior_identity_moments(
            _ExactPosteriorModel(), pool, nonidentity, chunk_size=2
        )
    with pytest.raises(ValueError, match="explicit versioned objective contract"):
        posterior_identity_moments(
            _ExactPosteriorModel(), pool, IdentityTransform(), chunk_size=2
        )


def test_analytic_moments_agree_with_high_sample_mc_and_stable_hvi_selection():
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    first_Y = torch.tensor([[0.1], [0.9], [0.4]], dtype=torch.double)
    second_Y = torch.tensor([[0.8], [0.2], [0.7]], dtype=torch.double)
    model = ModelListGP(
        SingleTaskGP(train_X, first_Y, outcome_transform=Standardize(m=1)),
        SingleTaskGP(train_X, second_Y, outcome_transform=Standardize(m=1)),
    )
    model.eval()
    pool = torch.linspace(0.05, 0.95, 9, dtype=torch.double).unsqueeze(-1)
    contract = _identity_contract(2)

    analytic_moments = posterior_identity_moments(model, pool, contract, chunk_size=4)
    mc_moments = posterior_utility_moments(
        model,
        pool,
        contract,
        mc_samples=4096,
        seed=17,
        chunk_size=3,
    )
    np.testing.assert_allclose(
        mc_moments.utility_mean,
        analytic_moments.utility_mean.cpu().numpy(),
        atol=0.02,
        rtol=0.02,
    )
    np.testing.assert_allclose(
        mc_moments.utility_std,
        analytic_moments.utility_std.cpu().numpy(),
        atol=0.02,
        rtol=0.05,
    )

    observed = torch.cat((first_Y, second_Y), dim=1)
    reference = np.array([-0.1, -0.1])
    analytic_scores = score_ucb_hvi_pool(
        model,
        pool,
        observed,
        contract,
        reference,
        beta=4.0,
        moment_method="analytic_identity",
        posterior_chunk_size=4,
    )
    mc_scores = score_ucb_hvi_pool(
        model,
        pool,
        observed,
        contract,
        reference,
        beta=4.0,
        moment_method="monte_carlo",
        mc_samples=4096,
        seed=17,
        posterior_chunk_size=3,
    )
    assert analytic_scores.moment_method == "analytic_identity"
    assert analytic_scores.mc_samples is None
    assert analytic_scores.seed is None
    assert int(np.argmax(analytic_scores.base_score)) == int(
        np.argmax(mc_scores.base_score)
    )


def test_ucb_bound_policy_retains_raw_effective_and_clip_amounts():
    raw = np.array([[1.2, -3.0, -0.2], [0.8, -2.0, 1.1]])
    original = raw.copy()
    bounds = [(0.0, 1.0), (None, None), (0.0, 1.0)]

    unchanged = apply_ucb_bound_policy(raw, bounds, "none")
    clipped = apply_ucb_bound_policy(raw, bounds, "clip_ucb")

    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(unchanged.utility_ucb_raw, raw)
    np.testing.assert_array_equal(unchanged.utility_ucb_effective, raw)
    np.testing.assert_array_equal(unchanged.utility_ucb_clip_amount, 0.0)
    np.testing.assert_allclose(
        clipped.utility_ucb_effective,
        np.array([[1.0, -3.0, 0.0], [0.8, -2.0, 1.0]]),
    )
    np.testing.assert_allclose(
        clipped.utility_ucb_clip_amount,
        np.array([[0.2, 0.0, 0.2], [0.0, 0.0, 0.1]]),
    )


def test_score_uses_effective_bounded_ucb_for_hvi_and_keeps_compatibility_alias():
    means = np.array([[1.2, 0.8, 1.2]])
    observed = np.array([[0.5, 0.5, 0.5]])
    bounds = [(0.0, 1.0), (None, None), (0.0, 1.0)]
    result = score_ucb_hvi_from_moments(
        means,
        np.zeros_like(means),
        observed,
        np.zeros(3),
        beta=0.0,
        bound_policy="clip_ucb",
        utility_bounds=bounds,
        moment_method="analytic_identity",
    )
    expected_scores = hypervolume_improvement_scores(
        np.array([[1.0, 0.8, 1.0]]), observed, np.zeros(3)
    )[0]
    np.testing.assert_allclose(result.base_score, expected_scores)
    np.testing.assert_allclose(result.utility_ucb_raw, means)
    np.testing.assert_allclose(result.utility_ucb_effective, [[1.0, 0.8, 1.0]])
    np.testing.assert_array_equal(result.utility_ucb, result.utility_ucb_effective)
    assert result.bound_policy == "clip_ucb"


@pytest.mark.parametrize(
    "bounds, policy, message",
    [
        (None, "clip_ucb", "requires explicit"),
        ([(0.0, 1.0)], "clip_ucb", "one.*per objective"),
        ([(1.0, 0.0), (None, None)], "clip_ucb", "must not exceed"),
        ([(None, None), (None, None)], "clip_ucb", "at least one"),
        (None, "invalid", "none.*clip_ucb"),
    ],
)
def test_ucb_bound_policy_rejects_invalid_contract(bounds, policy, message):
    with pytest.raises(ValueError, match=message):
        apply_ucb_bound_policy(np.ones((2, 2)), bounds, policy)


def test_zero_scores_remain_zero_while_log_scores_are_stabilized():
    result = score_ucb_hvi_from_moments(
        np.array([[0.1, 0.1]]),
        np.zeros((1, 2)),
        _observed_2d(),
        np.zeros(2),
        beta=0.0,
        log_epsilon=1e-9,
    )
    assert result.base_score[0] == 0.0
    assert result.base_log_score[0] == pytest.approx(np.log(1e-9))


def test_invalid_standard_deviation_and_shapes_fail():
    with pytest.raises(ValueError, match="negative"):
        score_ucb_hvi_from_moments(
            np.ones((1, 2)),
            np.array([[-1.0, 0.0]]),
            _observed_2d(),
            np.zeros(2),
            beta=1.0,
        )
    with pytest.raises(ValueError, match="same shape"):
        score_ucb_hvi_from_moments(
            np.ones((1, 2)),
            np.ones((2, 2)),
            _observed_2d(),
            np.zeros(2),
            beta=1.0,
        )


def _proposal_pool():
    X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
    return CandidatePool(
        grid_indices=np.arange(5)[:, None],
        X_phys=X.copy(),
        X_norm=X.copy(),
        seed=9,
        draws=5,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )


def _static_ucb_result(scores):
    scores = np.asarray(scores, dtype=float)
    return UCBHVIScoreResult(
        base_score=scores,
        base_log_score=np.log(np.maximum(scores, 1e-12)),
        utility_mean=np.column_stack([scores, scores]),
        utility_std=np.zeros((scores.size, 2)),
        utility_ucb=np.column_stack([scores, scores]),
        baseline_hypervolume=0.1,
        pareto_utility=np.array([[0.5, 0.5]]),
        reference_point_utility=np.zeros(2),
        beta=1.0,
        kappa=1.0,
        mc_samples=8,
        seed=4,
        observation_noise=False,
        objective_contract_version="TEST_ONLY-v1",
    )


def test_ucb_proposal_selects_only_positive_hvi_and_exact_q(monkeypatch):
    monkeypatch.setattr(
        module,
        "score_ucb_hvi_pool",
        lambda *args, **kwargs: _static_ucb_result([0.0, 0.8, 0.7, 0.0, 0.6]),
    )
    proposal = propose_ucb_hvi_batch(
        _proposal_pool(),
        torch.nn.Linear(1, 1).double(),
        np.ones((2, 2)),
        object(),
        np.zeros(2),
        q=2,
        beta=1.0,
        local_penalization_config=LocalPenalizationConfig(
            radius=0.15, min_batch_distance=0.1
        ),
        positive_score_tolerance=1e-6,
    )
    assert proposal.selection.selected_pool_indices.size == 2
    assert np.all(
        proposal.scoring.base_score[proposal.selection.selected_pool_indices] > 0
    )
    assert proposal.metadata["pool_seed"] == 9
    assert proposal.metadata["objective_contract_version"] == "TEST_ONLY-v1"
    assert proposal.metadata["beta"] == 1.0


def test_ucb_proposal_refuses_to_fill_with_zero_hvi(monkeypatch):
    monkeypatch.setattr(
        module,
        "score_ucb_hvi_pool",
        lambda *args, **kwargs: _static_ucb_result([0.0, 0.8, 0.0, 0.0, 0.0]),
    )
    with pytest.raises(UndersizedBatchError):
        propose_ucb_hvi_batch(
            _proposal_pool(),
            torch.nn.Linear(1, 1).double(),
            np.ones((2, 2)),
            object(),
            np.zeros(2),
            q=2,
            beta=1.0,
            local_penalization_config=LocalPenalizationConfig(
                radius=0.15, min_batch_distance=0
            ),
        )
