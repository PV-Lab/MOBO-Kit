"""End-to-end acceptance test on a synthetic problem with a known Pareto front.

DTLZ2 with 3 objectives and 10 inputs, run through the real campaign path:
``run_r0_lhs`` -> ``run_r1_ucb(5)`` -> ``run_r2_qlognehvi(3)``.  Nothing here
touches the experimental data, so it answers "does the algorithm work" separately
from "are the measurements right".

Two conventions had to be got right, and both fail silently if you don't.

**DTLZ2 minimises by default.** ``negate=True`` is mandatory. Without it the
objectives are positive, no point dominates the reference, and the test would
measure the opposite of optimisation.

**BoTorch's Hypervolume assumes maximisation and silently drops points that do
not dominate the reference** -- there is no warning and no exception, you simply
get a smaller number, or 0.0. So :func:`_hypervolume` asserts that at least one
point dominates before trusting the result.

The optimisation claim is deliberately weak, because the honest one is:
cumulative hypervolume rises monotonically *by construction* (adding points can
only grow the dominated region), so "HV increased" would pass for random
sampling too. The meaningful comparison is against a random baseline at the same
budget, and BO wins **on average, not on every seed** -- measured 5/8 seeds with
a mean gain ratio of 1.35x. Asserting a per-seed win would be a flaky test
asserting something untrue.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from mobo_kit.campaign import (
    build_objective_transform,
    run_r0_lhs,
    run_r1_ucb,
    run_r2_qlognehvi,
)
from mobo_kit.design import build_design_from_config
from mobo_kit.ucb_hvi import pareto_utility_above_reference

INPUT_DIM = 10
OBJECTIVES = 3
R0_SIZE = 15
R1_SIZE = 5
R2_SIZE = 3


def _problem() -> DTLZ2:
    # negate=True -> maximisation, which is what the campaign and BoTorch's
    # hypervolume both assume
    return DTLZ2(dim=INPUT_DIM, num_objectives=OBJECTIVES, negate=True).to(
        dtype=torch.double
    )


def _config(pool: int = 1024, mc_samples: int = 32) -> dict:
    """A campaign config for DTLZ2. Pool sizes are shrunk for test runtime.

    The production config uses 32768, which costs ~124 s for one R0->R1->R2 pass.
    R2 is the bottleneck and ``mc_samples`` is the cheapest lever, so that is what
    is reduced most.
    """
    return {
        "inputs": [
            {"name": f"x{i}", "start": 0.0, "stop": 1.0, "step": 0.05}
            for i in range(INPUT_DIM)
        ],
        "objectives": {
            "contract_version": "TEST_ONLY-dtlz2-v1",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": f"f{i}",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": f"f{i}",
                    # negated DTLZ2 lands in roughly [-1.9, 0]
                    "lower_anchor": -2.0,
                    "upper_anchor": 0.0,
                }
                for i in range(OBJECTIVES)
            ],
        },
        "reference_point_utility": [-0.01] * OBJECTIVES,
        "rounds": {
            "r1": {
                "method": "ucb_hvi",
                "batch_size": R1_SIZE,
                "replicates_per_condition": 3,
                "beta": 4.0,
                "candidate_pool_size": pool,
                "posterior_samples": 256,
                "moment_method": "monte_carlo",
            },
            "r2": {
                "method": "qlognehvi",
                "batch_size": R2_SIZE,
                "replicates_per_condition": 3,
                "candidate_pool_size": pool,
                "mc_samples": mc_samples,
            },
        },
        "local_penalization": {
            "radius": 0.25,
            "min_batch_distance": 0.15,
            "min_observed_distance": 0.0,
            "dimension_weights": None,
        },
        "model": {"variant": "dim_scaled_prior"},
        "reproducibility": {"seed": 73},
        "constraints": [],
    }


def _evaluate(problem: DTLZ2, X_phys: np.ndarray) -> np.ndarray:
    """DTLZ2 lives on [0,1]^10, which is exactly the declared design domain."""
    return problem(torch.tensor(np.asarray(X_phys, float), dtype=torch.double)).numpy()


def _hypervolume(config: dict, Y_raw: np.ndarray) -> float:
    """Hypervolume in UTILITY space against the fixed campaign reference.

    Computing it in utility space rather than raw space is what makes values
    comparable across rounds: the campaign's scales are fixed, so the reference
    does not drift as data arrives.
    """
    transform = build_objective_transform(config)
    reference = torch.tensor(config["reference_point_utility"], dtype=torch.double)
    # transform_measurements, not transform: Y_raw holds MEASUREMENT-space values,
    # and transform decodes the link itself. The two are the same call while every
    # objective is affine, which is exactly why this file could not see the R1
    # baseline bug -- see test_measurement_space_encoding.py.
    utility = transform.transform_measurements(
        torch.tensor(np.asarray(Y_raw, float), dtype=torch.double)
    )
    if not bool((utility >= reference).all(dim=-1).any()):
        raise AssertionError(
            "No point dominates the reference point. BoTorch would silently drop "
            "every point and return 0.0 rather than raising, so this is checked "
            "explicitly."
        )
    return Hypervolume(ref_point=reference).compute(utility[is_non_dominated(utility)])


def _run_campaign(config: dict, seed: int, evaluate=None) -> dict:
    """One full R0 -> R1 -> R2 pass, evaluating DTLZ2 at each proposed batch."""
    problem = _problem()
    evaluate = _evaluate if evaluate is None else evaluate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = run_r0_lhs(config, n=R0_SIZE, seed=seed)
        X0 = r0.conditions.to_numpy(float)
        Y0 = evaluate(problem, X0)

        r1 = run_r1_ucb(config, X0, Y0, seed=seed)
        X1 = r1.conditions.to_numpy(float)
        Y1 = evaluate(problem, X1)

        X01, Y01 = np.vstack([X0, X1]), np.vstack([Y0, Y1])
        r2 = run_r2_qlognehvi(config, X01, Y01, seed=seed)
        X2 = r2.conditions.to_numpy(float)
        Y2 = evaluate(problem, X2)

    return {
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "hv": [
            _hypervolume(config, Y0),
            _hypervolume(config, Y01),
            _hypervolume(config, np.vstack([Y01, Y2])),
        ],
        "X": [X0, X1, X2],
    }


@pytest.fixture(scope="module")
def campaign() -> dict:
    return _run_campaign(_config(), seed=73)


# --------------------------------------------------------------------------- #
# the algorithm produces well-formed batches
# --------------------------------------------------------------------------- #


def test_each_round_proposes_exactly_the_requested_count(campaign: dict) -> None:
    assert len(campaign["r0"].conditions) == R0_SIZE
    assert len(campaign["r1"].conditions) == R1_SIZE
    assert len(campaign["r2"].conditions) == R2_SIZE
    # 23 distinct conditions, 3 films each
    total = R0_SIZE + R1_SIZE + R2_SIZE
    assert total == 23
    assert len(campaign["r2"].replicates) == R2_SIZE * 3


@pytest.mark.parametrize("round_key", ["r0", "r1", "r2"])
def test_every_batch_is_unique_on_grid_and_in_bounds(
    campaign: dict, round_key: str
) -> None:
    report = campaign[round_key].diagnostics["validity"]
    assert report["unique"]
    assert report["on_grid"]
    assert report["in_bounds"]
    assert report["finite"]


@pytest.mark.parametrize("round_key", ["r1", "r2"])
def test_local_penalization_spreads_the_batch(campaign: dict, round_key: str) -> None:
    """Without penalization a batch collapses onto the single best pool point.

    The floor is the configured 0.15; the measured values are far above it, which
    is the signal that penalization is doing work rather than merely not failing.
    """
    minimum = campaign[round_key].diagnostics["validity"]["min_pairwise_distance"]
    assert minimum >= 0.15
    assert minimum > 0.5, f"batch is unexpectedly clustered: {minimum:.4f}"


def test_proposed_points_are_distinct_from_the_observed_set(campaign: dict) -> None:
    """Re-proposing an already-measured recipe would waste a film."""
    X0, X1, X2 = campaign["X"]
    seen = {tuple(np.round(row, 9)) for row in X0}
    for row in np.vstack([X1, X2]):
        assert tuple(np.round(row, 9)) not in seen


# --------------------------------------------------------------------------- #
# the algorithm optimises
# --------------------------------------------------------------------------- #


def test_cumulative_hypervolume_never_decreases(campaign: dict) -> None:
    """True by construction -- adding points cannot shrink the dominated region.

    It is asserted anyway because a violation would mean something is broken in
    the transform, the reference point, or the sign convention. It is NOT
    evidence of optimisation; see the random-baseline test for that.
    """
    hv0, hv1, hv2 = campaign["hv"]
    assert hv0 <= hv1 <= hv2
    assert hv0 > 0.0


def test_hypervolume_actually_improves_over_the_initial_design(
    campaign: dict,
) -> None:
    hv0, _, hv2 = campaign["hv"]
    assert hv2 > hv0
    # DTLZ2's optimum against its own reference is 0.807; we are in the right
    # order of magnitude rather than chasing a specific value
    assert 0.0 < hv2 < 1.0


def test_the_batches_are_deterministic_for_a_fixed_seed(campaign: dict) -> None:
    """A reproducible campaign is a precondition for auditing one."""
    repeat = _run_campaign(_config(), seed=73)
    np.testing.assert_allclose(
        repeat["r1"].conditions.to_numpy(float), campaign["X"][1]
    )
    np.testing.assert_allclose(
        repeat["r2"].conditions.to_numpy(float), campaign["X"][2]
    )


# --------------------------------------------------------------------------- #
# every link type the campaign uses, exercised end to end
# --------------------------------------------------------------------------- #
#
# Added 2026-07-31, after `run_r1_ucb` was found to have been handing the objective
# transform measurement-space values for the life of the campaign. This file could
# not have caught it: every objective above is affine, and for an affine objective
# measurement space and model space are the same numbers, so a link-encoding
# mistake is invisible BY CONSTRUCTION.
#
# The live campaign has a log-link objective (thickness trains on log(nm)), so the
# synthetic acceptance test must have one too, or "the loop passes end to end"
# keeps meaning "the loop passes end to end for half of the link types in use".


def _config_with_log_link(pool: int = 1024, mc_samples: int = 32) -> dict:
    """The same DTLZ2 problem with its third objective reached through a log link.

    ``f2`` is reported as ``exp(f2)`` -- a strictly positive measurement -- and the
    objective declares ``response: log``. The GP therefore trains on
    ``log(exp(f2)) = f2``: the SAME latent quantity the affine config models,
    reached by a different route. Any mis-encoding shows up as a difference in
    something that ought to be identical.
    """
    config = _config(pool=pool, mc_samples=mc_samples)
    config["objectives"]["contract_version"] = "TEST_ONLY-dtlz2-loglink-v1"
    config["objectives"]["specs"][2] = {
        "name": "f2",
        "goal": "maximize",
        "transform": "affine",
        "model_source_column": "f2",
        # negated DTLZ2 lands in roughly [-1.9, 0], so exp() lands in [0.15, 1]
        "lower_anchor": float(np.exp(-2.0)),
        "upper_anchor": 1.0,
        "mean_function": {
            "response": "log",
            "features": [{"column": "x0", "transform": "identity"}],
        },
    }
    return config


def _evaluate_log_linked(problem: DTLZ2, X_phys: np.ndarray) -> np.ndarray:
    Y = _evaluate(problem, X_phys)
    return np.column_stack([Y[:, 0], Y[:, 1], np.exp(Y[:, 2])])


@pytest.fixture(scope="module")
def log_linked_campaign() -> dict:
    return _run_campaign(
        _config_with_log_link(), seed=73, evaluate=_evaluate_log_linked
    )


def test_the_log_link_config_really_is_log_linked() -> None:
    """Guards the guard: if this reverts to identity the tests below go quiet."""
    specs = build_objective_transform(_config_with_log_link()).specs
    assert [spec.model_link for spec in specs] == ["identity", "identity", "log"]
    # and the plain config remains the affine-only case, so both are covered
    assert [s.model_link for s in build_objective_transform(_config()).specs] == [
        "identity"
    ] * OBJECTIVES


def test_a_log_linked_campaign_runs_end_to_end(log_linked_campaign: dict) -> None:
    assert len(log_linked_campaign["r0"].conditions) == R0_SIZE
    assert len(log_linked_campaign["r1"].conditions) == R1_SIZE
    assert len(log_linked_campaign["r2"].conditions) == R2_SIZE
    for key in ("r0", "r1", "r2"):
        report = log_linked_campaign[key].diagnostics["validity"]
        assert report["unique"] and report["on_grid"] and report["in_bounds"]
    hv0, hv1, hv2 = log_linked_campaign["hv"]
    assert 0.0 < hv0 <= hv1 <= hv2


def test_no_observed_utility_collapses_to_zero_under_a_log_link(
    log_linked_campaign: dict,
) -> None:
    """The invariant the R1 baseline bug violated.

    Under the mis-encoding every observation scored exactly 0.0 on the log-linked
    axis -- a finite, unremarkable number that no check rejected. A measured point
    with a finite value inside its anchors has non-zero utility; a hard zero means
    an encoding step was skipped.

    Only points whose raw value lies strictly INSIDE the objective's anchors are
    checked. An affine objective legitimately clips to 0.0 when a measurement falls
    at or below its lower anchor, and DTLZ2 does produce such points; asserting on
    those would be asserting that clipping is a bug.
    """
    config = _config_with_log_link()
    transform = build_objective_transform(config)
    spec = transform.specs[2]
    problem = _problem()
    checked = 0
    for X in log_linked_campaign["X"]:
        Y = _evaluate_log_linked(problem, X)
        utility = transform.transform_measurements(
            torch.tensor(Y, dtype=torch.double)
        ).numpy()
        assert np.isfinite(utility).all()
        raw = Y[:, 2]
        inside = (raw > spec.lower_anchor) & (raw < spec.upper_anchor)
        assert not np.any(utility[inside, 2] == 0.0), (
            "a finite measurement strictly inside its anchors scored exactly zero"
        )
        checked += int(inside.sum())
    # the assertion above is vacuous if nothing was inside the anchors
    assert checked >= 15, f"only {checked} points were in range; test has no teeth"


def test_the_r1_baseline_is_right_when_a_link_has_to_be_decoded(
    log_linked_campaign: dict,
) -> None:
    """End-to-end version of the comparator that did not exist.

    ``run_r1_ucb`` reports the baseline hypervolume it actually used; this
    recomputes it by an independent route. On the pre-fix code the reported value
    is the collapsed one and this fails.
    """
    config = _config_with_log_link()
    transform = build_objective_transform(config)
    reference = np.asarray(config["reference_point_utility"], dtype=float)

    X0 = log_linked_campaign["r0"].conditions.to_numpy(float)
    Y0 = _evaluate_log_linked(_problem(), X0)
    utility = transform.transform_measurements(
        torch.tensor(Y0, dtype=torch.double)
    ).numpy()
    pareto = pareto_utility_above_reference(utility, reference)
    expected = float(
        Hypervolume(ref_point=torch.tensor(reference, dtype=torch.double)).compute(
            torch.tensor(pareto, dtype=torch.double)
        )
    )
    reported = log_linked_campaign["r1"].diagnostics["observed_baseline_hypervolume"]
    assert reported == pytest.approx(expected, rel=1e-9)
    assert log_linked_campaign["r1"].diagnostics[
        "observed_baseline_pareto_size"
    ] == len(pareto)


@pytest.mark.slow
def test_bayesian_optimisation_beats_random_search_on_average() -> None:
    """The test that makes the hypervolume numbers mean something.

    Cumulative HV rises for random sampling too, so the only informative
    comparison is against a random baseline at the same budget (8 extra points
    from the same 15-point start).

    BO wins on the MEAN, not on every seed: measured 5 of 8 seeds with a mean
    gain ratio of 1.35x. With 8 added points in 10 dimensions that is the
    honest expectation, and asserting a per-seed win would be a flaky test
    asserting something false.
    """
    config = _config()
    design = build_design_from_config(config)
    problem = _problem()
    grids = [np.asarray(g, float) for g in design.var_array]

    bo_gains, random_gains, wins = [], [], 0
    for seed in (1, 2, 3, 4, 5):
        result = _run_campaign(config, seed=seed)
        hv0, _, hv_bo = result["hv"]

        rng = np.random.default_rng(seed)
        X_random = np.column_stack(
            [rng.choice(g, size=R1_SIZE + R2_SIZE) for g in grids]
        )
        Y_random = np.vstack(
            [_evaluate(problem, result["X"][0]), _evaluate(problem, X_random)]
        )
        hv_random = _hypervolume(config, Y_random)

        bo_gains.append(hv_bo - hv0)
        random_gains.append(hv_random - hv0)
        wins += (hv_bo - hv0) > (hv_random - hv0)

    mean_bo, mean_random = float(np.mean(bo_gains)), float(np.mean(random_gains))
    assert mean_bo > 0, "BO made no hypervolume progress at all"
    assert mean_random > 0, "the random baseline is broken, not a fair comparison"
    assert mean_bo > mean_random, (
        f"BO mean gain {mean_bo:.4f} did not beat random {mean_random:.4f}; "
        f"won {wins}/5 seeds"
    )
