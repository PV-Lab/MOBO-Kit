import numpy as np
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize

import mobo_kit.qlognehvi_batch as module
from mobo_kit.batch_selection import LocalPenalizationConfig, UndersizedBatchError
from mobo_kit.candidate_pool import CandidatePool
from mobo_kit.objectives import (
    BoundedMCMultiOutputObjective,
    ConfiguredMCMultiOutputObjective,
    ObjectiveSpec,
    ObjectiveTransform,
)


def _objective(count=2):
    transform = ObjectiveTransform(
        [
            ObjectiveSpec(f"utility_{index}", "maximize", "identity")
            for index in range(count)
        ],
        version="TEST_ONLY-qlog-v1",
    )
    return ConfiguredMCMultiOutputObjective(transform)


class FakeAcquisition:
    def __init__(self, pending_count: int):
        self.pending_count = pending_count
        self.shapes = []

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(X.shape))
        # A deterministic log score with a visible pending-point effect.
        return X[..., 0, :].sum(dim=-1) - self.pending_count


def test_singleton_shape_chunking_and_pending_metadata(monkeypatch):
    acquisitions = []

    def fake_builder(**kwargs):
        pending = kwargs["X_pending"]
        acquisition = FakeAcquisition(0 if pending is None else pending.shape[0])
        acquisitions.append(acquisition)
        return acquisition

    monkeypatch.setattr(module, "_build_qlognehvi", fake_builder)
    train = torch.zeros((3, 2), dtype=torch.double)
    pool = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.double)
    pending = torch.tensor([[0.0, 1.0]], dtype=torch.double)
    result = module.score_qlognehvi_singletons(
        object(),
        train,
        pool,
        _objective(),
        np.array([-1.0, -1.0]),
        mc_samples=16,
        seed=8,
        chunk_size=2,
        X_pending_norm=pending,
    )
    assert result.evaluated_shape == (3, 1, 2)
    assert result.pending_count == 1
    assert result.mc_samples == 16
    assert result.seed == 8
    assert np.allclose(result.base_log_score, [-0.7, -0.3, 0.1])
    assert acquisitions[0].shapes == [(2, 1, 2), (1, 1, 2)]


def test_chunked_and_unchunked_fake_scores_are_equal(monkeypatch):
    monkeypatch.setattr(
        module,
        "_build_qlognehvi",
        lambda **kwargs: FakeAcquisition(0),
    )
    train = torch.zeros((2, 2), dtype=torch.double)
    pool = torch.rand((7, 2), generator=torch.Generator().manual_seed(2))
    kwargs = (object(), train, pool, _objective(), np.array([-1.0, -1.0]))
    chunked = module.score_qlognehvi_singletons(*kwargs, chunk_size=2)
    whole = module.score_qlognehvi_singletons(*kwargs, chunk_size=100)
    assert np.array_equal(chunked.base_log_score, whole.base_log_score)


@pytest.mark.parametrize(
    "reference, match",
    [(None, "required"), (np.array([]), "shape"), (np.array([np.nan]), "finite")],
)
def test_reference_validation(monkeypatch, reference, match):
    monkeypatch.setattr(
        module,
        "_build_qlognehvi",
        lambda **kwargs: FakeAcquisition(0),
    )
    with pytest.raises(ValueError, match=match):
        module.score_qlognehvi_singletons(
            object(),
            torch.zeros((2, 1)),
            torch.zeros((1, 1)),
            _objective(1),
            reference,
        )


def test_seed_is_passed_to_qmc_builder(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return FakeAcquisition(0)

    monkeypatch.setattr(module, "_build_qlognehvi", fake_builder)
    module.score_qlognehvi_singletons(
        object(),
        torch.zeros((2, 1)),
        torch.zeros((1, 1)),
        _objective(1),
        np.array([-1.0]),
        mc_samples=32,
        seed=41,
    )
    assert captured["mc_samples"] == 32
    assert captured["seed"] == 41


def test_configured_objective_and_reference_dimensions_are_required(monkeypatch):
    monkeypatch.setattr(
        module,
        "_build_qlognehvi",
        lambda **kwargs: FakeAcquisition(0),
    )
    train = torch.zeros((2, 1))
    pool = torch.zeros((1, 1))
    with pytest.raises(TypeError, match="configured or bounded configured"):
        module.score_qlognehvi_singletons(object(), train, pool, None, np.array([-1.0]))
    with pytest.raises(ValueError, match="configured objective count"):
        module.score_qlognehvi_singletons(
            object(), train, pool, _objective(2), np.array([-1.0])
        )


def test_real_qlognehvi_accepts_bounded_objective_and_returns_reproducible_scores():
    train_X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [
            [0.15, -3.0, 0.20],
            [0.55, -2.2, 0.45],
            [0.90, -1.5, 0.85],
            [0.65, -1.9, 0.60],
            [0.25, -2.8, 0.30],
        ],
        dtype=torch.double,
    )
    model = ModelListGP(
        *[
            SingleTaskGP(
                train_X,
                train_Y[:, index : index + 1],
                outcome_transform=Standardize(m=1),
            )
            for index in range(train_Y.shape[1])
        ]
    )
    transform = ObjectiveTransform(
        [
            ObjectiveSpec("uniformity", "maximize", "identity"),
            ObjectiveSpec("optoelectronic", "maximize", "identity"),
            ObjectiveSpec("thickness", "maximize", "identity"),
        ],
        version="TEST_ONLY-bounded-qlog-v1",
    )
    objective = BoundedMCMultiOutputObjective(
        transform,
        bounds=((0.0, 1.0), (None, None), (0.0, 1.0)),
    )
    pool = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    reference = np.array([-0.01, -4.0, -0.01])

    first = module.score_qlognehvi_singletons(
        model,
        train_X,
        pool,
        objective,
        reference,
        mc_samples=16,
        seed=73,
        chunk_size=2,
        prune_baseline=False,
    )
    second = module.score_qlognehvi_singletons(
        model,
        train_X,
        pool,
        objective,
        reference,
        mc_samples=16,
        seed=73,
        chunk_size=2,
        prune_baseline=False,
    )

    assert first.evaluated_shape == (4, 1, 1)
    assert first.objective_contract_version == objective.version
    assert np.array_equal(first.base_log_score, second.base_log_score)
    assert not np.any(np.isnan(first.base_log_score))
    assert not np.any(np.isposinf(first.base_log_score))
    assert np.any(np.isfinite(first.base_log_score))
    assert np.array_equal(first.reference_point_utility, reference)


def _proposal_pool():
    X = np.array([[0.3], [0.45], [0.6], [0.75], [0.9]])
    return CandidatePool(
        grid_indices=np.arange(5)[:, None],
        X_phys=X.copy(),
        X_norm=X.copy(),
        seed=3,
        draws=5,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )


def test_sequential_proposal_updates_pending_and_returns_exact_three(monkeypatch):
    pending_counts = []
    seen_objectives = []

    def fake_score(model, train_X, X_pool, objective, reference, **kwargs):
        del model, train_X
        pending = kwargs.get("X_pending_norm")
        pending_count = 0 if pending is None else pending.shape[0]
        pending_counts.append(pending_count)
        seen_objectives.append(objective)
        return module.QLogNEHVIPoolScoreResult(
            base_log_score=X_pool[:, 0].detach().cpu().numpy(),
            evaluated_shape=(X_pool.shape[0], 1, X_pool.shape[1]),
            pending_count=pending_count,
            mc_samples=kwargs["mc_samples"],
            seed=kwargs["seed"],
            reference_point_utility=np.asarray(reference),
            objective_contract_version=objective.objective_transform.version,
        )

    monkeypatch.setattr(module, "score_qlognehvi_singletons", fake_score)
    objective = _objective(1)
    proposal = module.propose_qlognehvi_penalized_batch(
        _proposal_pool(),
        object(),
        torch.tensor([[0.05], [0.15]], dtype=torch.double),
        objective,
        np.array([-1.0]),
        q=3,
        local_penalization_config=LocalPenalizationConfig(
            radius=0.12, min_batch_distance=0.1
        ),
        X_pending_norm=torch.tensor([[0.2]], dtype=torch.double),
        mc_samples=16,
        seed=7,
    )
    assert proposal.selection.selected_pool_indices.size == 3
    assert np.unique(proposal.selection.selected_pool_indices).size == 3
    assert pending_counts == [1, 2, 3]
    assert seen_objectives == [objective, objective, objective]
    assert [history.pending_count for history in proposal.score_history] == [1, 2, 3]
    assert all(step.base_score is not None for step in proposal.selection.steps)
    assert proposal.metadata["pool_seed"] == 3
    assert proposal.metadata["mc_seed"] == 7
    assert proposal.metadata["objective_contract_version"] == "TEST_ONLY-qlog-v1"


def test_proposal_rejects_observed_or_pending_pool_overlap():
    with pytest.raises(ValueError, match="overlaps observed"):
        module.propose_qlognehvi_penalized_batch(
            _proposal_pool(),
            object(),
            torch.tensor([[0.3]], dtype=torch.double),
            _objective(1),
            np.array([-1.0]),
            q=1,
            local_penalization_config=LocalPenalizationConfig(
                radius=0.1, min_batch_distance=0
            ),
        )


def test_qlog_proposal_hard_distance_failure_is_explicit(monkeypatch):
    def fake_score(model, train_X, X_pool, objective, reference, **kwargs):
        del model, train_X
        return module.QLogNEHVIPoolScoreResult(
            base_log_score=np.zeros(X_pool.shape[0]),
            evaluated_shape=(X_pool.shape[0], 1, X_pool.shape[1]),
            pending_count=0,
            mc_samples=kwargs["mc_samples"],
            seed=kwargs["seed"],
            reference_point_utility=np.asarray(reference),
            objective_contract_version=objective.objective_transform.version,
        )

    monkeypatch.setattr(module, "score_qlognehvi_singletons", fake_score)
    with pytest.raises(UndersizedBatchError):
        module.propose_qlognehvi_penalized_batch(
            _proposal_pool(),
            object(),
            torch.tensor([[0.05]], dtype=torch.double),
            _objective(1),
            np.array([-1.0]),
            q=3,
            local_penalization_config=LocalPenalizationConfig(
                radius=0.1, min_batch_distance=0.7
            ),
        )
