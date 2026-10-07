"""Measurement space against model space, and the bug that lived in the gap.

``ObjectiveTransform.transform`` is a MODEL-OUTPUT decoder: it undoes the link
(``exp`` for a log objective) before computing utility. Handing it a raw
measurement exponentiates a number that was never a logarithm.

``run_r1_ucb`` did exactly that with its observed HVI baseline until 2026-07-31.
``exp(360…1303)`` saturates the 650 nm Gaussian to exactly ``0.0`` -- finite, so
neither the transform's own finiteness check nor the caller's fired. Every
observation's thickness utility was zero and the baseline hypervolume came out
0.004659 where the truth is 0.436442.

Two things about how it survived, both encoded as tests here.

**It was already documented.** ``test_transform_reproduces_the_workbook_thickness_score``
in ``test_campaign.py`` says in as many words that "feeding it raw nm would
silently score exp(687) instead of 687" -- and then only ever tests the correct
usage. Knowing a trap exists is not the same as testing that no caller falls in
it.

**Nothing compared the baseline to anything.** It was a plausible finite number
that no test reproduced independently -- the same shape as the hypervolume
auto-reference and the swallowed ``train_Yvar``. So ``run_r1_ucb`` now reports
its baseline in diagnostics, and the test below recomputes it by a different
route.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

from mobo_kit.campaign import (
    build_objective_transform,
    load_campaign_config,
    run_r0_lhs,
    run_r1_ucb,
)
from mobo_kit.ucb_hvi import pareto_utility_above_reference

CONFIG_PATH = "configs/campaign_d2d_perovskite.yaml"
SEED = 73

#: The two numbers this bug produced on the real 15 R0 rows. Pinned so the size of
#: the defect stays on record even if the workbook is later replaced.
MISENCODED_BASELINE_HV = 0.004659
CORRECT_BASELINE_HV = 0.436442


def _synthetic_config(pool: int = 256) -> dict:
    """A campaign with one log-link objective, so the encoding is exercised.

    Inputs start at 1.0 so the log-response mean function has positive features.
    """
    return {
        "inputs": [
            {"name": f"x{i}", "start": 1.0, "stop": 2.0, "step": 0.05} for i in range(10)
        ],
        "objectives": {
            "contract_version": "TEST_ONLY-encoding-v1",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": "affine_a",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": "affine_a",
                    "lower_anchor": 0.0,
                    "upper_anchor": 3.0,
                },
                {
                    "name": "affine_b",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": "affine_b",
                    "lower_anchor": -4.0,
                    "upper_anchor": 0.0,
                },
                {
                    "name": "log_linked",
                    "goal": "target",
                    "transform": "gaussian_target",
                    "model_source_column": "log_linked",
                    "target": 650.0,
                    "sigma": 176.7766952966369,
                    "mean_function": {
                        "response": "log",
                        "features": [{"column": "x0", "transform": "log"}],
                    },
                },
            ],
        },
        "reference_point_utility": [-0.01, -0.01, -0.01],
        "rounds": {
            "r1": {
                "method": "ucb_hvi",
                "batch_size": 5,
                "replicates_per_condition": 3,
                "beta": 4.0,
                "candidate_pool_size": pool,
                "posterior_samples": 16,
                "moment_method": "monte_carlo",
            },
            "r2": {
                "method": "qlognehvi",
                "batch_size": 3,
                "replicates_per_condition": 3,
                "candidate_pool_size": pool,
                "mc_samples": 8,
            },
        },
        "local_penalization": {
            "radius": 0.25,
            "min_batch_distance": 0.15,
            "min_observed_distance": 0.0,
            "dimension_weights": None,
        },
        "model": {"variant": "dim_scaled_prior"},
        "reproducibility": {"seed": SEED},
        "constraints": [],
    }


def _measurements(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([
        X.mean(axis=1),
        -np.linalg.norm(X - 1.5, axis=1),
        650.0 * X[:, 0] ** -0.5 * X[:, 1] ** 0.3,  # straddles the 650 nm target
    ])


# --------------------------------------------------------------------------- #
# the encoder itself
# --------------------------------------------------------------------------- #


def test_encode_measurements_logs_only_the_log_link_axes() -> None:
    transform = build_objective_transform(_synthetic_config())
    measured = torch.tensor([[1.5, -2.0, 700.0], [2.0, -1.0, 500.0]], dtype=torch.double)
    encoded = transform.encode_measurements(measured)
    torch.testing.assert_close(encoded[:, :2], measured[:, :2])
    torch.testing.assert_close(encoded[:, 2], torch.log(measured[:, 2]))


def test_transform_measurements_is_encode_then_transform() -> None:
    transform = build_objective_transform(_synthetic_config())
    measured = torch.tensor([[1.5, -2.0, 700.0]], dtype=torch.double)
    torch.testing.assert_close(
        transform.transform_measurements(measured),
        transform.transform(transform.encode_measurements(measured)),
    )


def test_transform_measurements_matches_the_gaussian_computed_by_hand() -> None:
    """An independent comparator: the closed form, not another code path."""
    transform = build_objective_transform(_synthetic_config())
    nm = np.array([360.0, 650.0, 700.0, 1303.0])
    measured = torch.tensor(
        np.column_stack([np.full(nm.size, 1.5), np.full(nm.size, -2.0), nm]),
        dtype=torch.double,
    )
    got = transform.transform_measurements(measured)[:, 2].numpy()
    sigma = 176.7766952966369
    expected = np.exp(-0.5 * ((nm - 650.0) / sigma) ** 2)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_encode_measurements_rejects_non_positive_on_a_log_link() -> None:
    transform = build_objective_transform(_synthetic_config())
    with pytest.raises(ValueError, match="strictly positive"):
        transform.encode_measurements(
            torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.double)
        )


def test_encode_measurements_rejects_non_finite_input() -> None:
    transform = build_objective_transform(_synthetic_config())
    with pytest.raises(ValueError, match="finite"):
        transform.encode_measurements(
            torch.tensor([[1.0, -1.0, float("inf")]], dtype=torch.double)
        )


# --------------------------------------------------------------------------- #
# the failure mode, pinned so it stays recognisable
# --------------------------------------------------------------------------- #


def test_unencoded_nanometres_collapse_to_exactly_zero_utility() -> None:
    """Why the bug was silent: the wrong answer is a finite, ordinary-looking 0.0.

    This asserts the BROKEN behaviour of the raw call deliberately. It is the
    fingerprint to recognise if it ever reappears somewhere else.
    """
    transform = build_objective_transform(_synthetic_config())
    measured = torch.tensor([[1.5, -2.0, 360.0], [1.5, -2.0, 1303.0]], dtype=torch.double)

    unencoded = transform.transform(measured)  # the mistake
    assert torch.isfinite(unencoded).all(), "no guard fires -- that is the problem"
    assert bool((unencoded[:, 2] == 0.0).all())

    encoded = transform.transform_measurements(measured)  # the fix
    assert bool((encoded[:, 2] > 0.0).all())


@pytest.mark.parametrize("nm", [360.0, 500.0, 650.0, 900.0, 1303.0])
def test_a_finite_in_range_measurement_never_scores_exactly_zero(nm: float) -> None:
    """The invariant the bug violated, stated directly.

    A real film that was measured at all has some merit on every axis. A utility of
    exactly 0.0 for a finite measurement means an encoding was skipped, not that
    the film was worthless.
    """
    transform = build_objective_transform(_synthetic_config())
    measured = torch.tensor([[1.5, -2.0, nm]], dtype=torch.double)
    utility = transform.transform_measurements(measured)
    assert torch.isfinite(utility).all()
    assert not bool((utility == 0.0).any())


# --------------------------------------------------------------------------- #
# the regression test: it fails on the pre-fix run_r1_ucb
# --------------------------------------------------------------------------- #


def test_run_r1_ucb_baseline_matches_an_independent_computation() -> None:
    """The comparator that did not exist.

    ``run_r1_ucb`` now reports the baseline hypervolume the acquisition actually
    used. Here it is recomputed by a separate route -- explicit encode, explicit
    Pareto filter, explicit Hypervolume -- and the two must agree.

    On the pre-fix code the reported value is the collapsed one and this fails.
    """
    config = _synthetic_config()
    transform = build_objective_transform(config)
    reference = np.asarray(config["reference_point_utility"], dtype=float)

    X = run_r0_lhs(config, n=15, seed=SEED).conditions.to_numpy(float)
    Y = _measurements(X)

    result = run_r1_ucb(config, X, Y, seed=SEED)
    reported = result.diagnostics["observed_baseline_hypervolume"]

    utility = transform.transform_measurements(
        torch.tensor(Y, dtype=torch.double)
    ).numpy()
    pareto = pareto_utility_above_reference(utility, reference)
    expected = float(
        Hypervolume(ref_point=torch.tensor(reference, dtype=torch.double)).compute(
            torch.tensor(pareto, dtype=torch.double)
        )
    )

    assert reported == pytest.approx(expected, rel=1e-9)
    assert result.diagnostics["observed_baseline_pareto_size"] == len(pareto)

    # and the mis-encoded route gives a materially different answer, so the
    # assertion above has teeth rather than passing on a coincidence
    collapsed = transform.transform(torch.tensor(Y, dtype=torch.double)).numpy()
    assert not np.allclose(collapsed[:, 2], utility[:, 2])


def test_the_baseline_is_reported_at_all() -> None:
    """A number nobody can see is a number nobody can check."""
    config = _synthetic_config()
    X = run_r0_lhs(config, n=15, seed=SEED).conditions.to_numpy(float)
    result = run_r1_ucb(config, X, _measurements(X), seed=SEED)
    assert "observed_baseline_hypervolume" in result.diagnostics
    assert "observed_baseline_pareto_size" in result.diagnostics
    assert result.diagnostics["observed_baseline_hypervolume"] > 0.0


# --------------------------------------------------------------------------- #
# the live campaign's own numbers
# --------------------------------------------------------------------------- #


@pytest.mark.local_input
@pytest.mark.skipif(
    not __import__("pathlib").Path("local_inputs/Summary Table.xlsx").is_file(),
    reason="local_inputs/Summary Table.xlsx is not present in this checkout",
)
def test_the_real_workbook_reproduces_the_two_recorded_baselines() -> None:
    """The 94x, on the actual data, so the recorded numbers stay falsifiable."""
    from mobo_kit.workbook_io import read_campaign_workbook

    config = load_campaign_config(CONFIG_PATH)
    transform = build_objective_transform(config)
    reference = np.asarray(config["reference_point_utility"], dtype=float)

    contents = read_campaign_workbook("local_inputs/Summary Table.xlsx", config)
    assert contents.errors == ()
    Y = contents.model_values.to_numpy(float)

    def hv(utility: np.ndarray) -> float:
        pareto = pareto_utility_above_reference(utility, reference)
        if pareto.shape[0] == 0:
            return 0.0
        return float(
            Hypervolume(ref_point=torch.tensor(reference, dtype=torch.double)).compute(
                torch.tensor(pareto, dtype=torch.double)
            )
        )

    correct = hv(transform.transform_measurements(torch.tensor(Y, dtype=torch.double)).numpy())
    misencoded = hv(transform.transform(torch.tensor(Y, dtype=torch.double)).numpy())

    assert correct == pytest.approx(CORRECT_BASELINE_HV, abs=5e-6)
    assert misencoded == pytest.approx(MISENCODED_BASELINE_HV, abs=5e-6)
    # every thickness utility was zero under the mis-encoding
    collapsed = transform.transform(torch.tensor(Y, dtype=torch.double)).numpy()
    assert bool((collapsed[:, 2] == 0.0).all())
