"""Hypervolume against a fixed reference.

No test covered `compute_ref_pareto_hv` before 2026-07-30, which is how its
degenerate auto-reference survived: it returned a number, and nobody compared that
number with the right one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.utils.multi_objective.hypervolume import infer_reference_point

from mobo_kit.metrics import compute_diversity_score, compute_ref_pareto_hv


def _front() -> torch.Tensor:
    """Three mutually non-dominated points plus one dominated one."""
    return torch.tensor(
        [
            [1.0, 0.2, 0.5],
            [0.2, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [0.1, 0.1, 0.1],
        ],
        dtype=torch.double,
    )


def test_a_missing_reference_is_refused_with_the_config_key_named() -> None:
    """The old default was `Y.min(dim=0) - 1e-8`, which made every slab 1e-8 thick
    and re-derived itself from the data on every call."""
    with pytest.raises(ValueError, match="reference_point_utility"):
        compute_ref_pareto_hv(_front())


def test_the_error_says_why_an_inferred_reference_is_wrong() -> None:
    with pytest.raises(ValueError, match="incomparable across them"):
        compute_ref_pareto_hv(_front(), None)


def test_an_explicit_reference_gives_the_dominated_volume() -> None:
    Y = _front()
    reference = np.array([-0.01, -0.01, -0.01])
    ref_point_t, pareto_Y, volume = compute_ref_pareto_hv(Y, reference)
    assert ref_point_t.dtype == Y.dtype
    assert pareto_Y.shape[0] == 3  # the dominated point is dropped
    assert volume > 0.0


def test_the_reference_is_not_re_derived_from_the_data() -> None:
    """The same reference on a growing dataset must give a monotone,
    comparable series. With the old auto-reference it did not."""
    Y = _front()
    reference = np.array([-0.01, -0.01, -0.01])
    _, _, first = compute_ref_pareto_hv(Y[:3], reference)
    _, _, second = compute_ref_pareto_hv(Y, reference)
    assert second >= first
    extra = torch.cat([Y, torch.tensor([[1.2, 1.2, 1.2]], dtype=Y.dtype)])
    _, _, third = compute_ref_pareto_hv(extra, reference)
    assert third > second


def test_the_old_auto_reference_collapses_on_a_real_trade_off_front() -> None:
    """Pins the reason this changed, and the condition for it.

    `Y.min(dim=0) - 1e-8` is only harmless while some *dominated* point sets the
    per-objective minima. As soon as the Pareto set itself sets them -- which is
    what a genuine trade-off front looks like, each point best in one objective and
    worst in another -- every slab is 1e-8 thick in at least one dimension and the
    volume collapses. That is the 6e-8-against-1.448 in the campaign notes.
    """
    trade_off = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.double
    )
    degenerate = (trade_off.min(dim=0).values - 1e-8).numpy()
    _, _, degenerate_volume = compute_ref_pareto_hv(trade_off, degenerate)
    inferred = infer_reference_point(trade_off).numpy()
    _, _, inferred_volume = compute_ref_pareto_hv(trade_off, inferred)

    assert degenerate_volume < 1e-10
    # not pinned tightly: infer_reference_point's margin below the nadir is a
    # BoTorch heuristic, and the claim here is the ratio, not its exact value
    assert inferred_volume > 1e-3
    assert degenerate_volume < inferred_volume / 1e6


def test_a_reference_nothing_dominates_is_refused_not_reported_as_zero() -> None:
    """BoTorch silently drops points that do not dominate the reference, so an
    unreachable reference reads as 0.0 -- indistinguishable from a sign error."""
    with pytest.raises(ValueError, match="No observation dominates"):
        compute_ref_pareto_hv(_front(), np.array([10.0, 10.0, 10.0]))


def test_a_flipped_sign_convention_is_caught_by_the_same_check() -> None:
    minimising = -_front()
    with pytest.raises(ValueError, match="every objective must be maximised"):
        compute_ref_pareto_hv(minimising, np.array([-0.01, -0.01, -0.01]))


@pytest.mark.parametrize(
    "reference, match",
    [
        (np.zeros((2, 3)), "must be 1D"),
        (np.zeros(2), "does not match number of objectives"),
        (np.array([0.0, np.inf, 0.0]), "must be finite"),
        ("not an array", "must be a numpy.ndarray"),
    ],
)
def test_a_malformed_reference_is_refused(reference, match) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        compute_ref_pareto_hv(_front(), reference)


def test_a_torch_reference_is_accepted() -> None:
    """Callers hold the reference as a tensor as often as an array."""
    _, _, volume = compute_ref_pareto_hv(
        _front(), torch.tensor([-0.01, -0.01, -0.01], dtype=torch.double)
    )
    assert volume > 0.0


def test_diversity_score_is_the_mean_pairwise_distance() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert compute_diversity_score(X) == pytest.approx(5.0)
    assert compute_diversity_score(np.array([[1.0, 1.0]])) == 0.0
