"""SHAP attribution over the campaign's own models.

What is pinned here is not "the numbers look plausible" -- that is how the last
three silent-failure bugs survived -- but properties that fail loudly if the
attribution stops meaning what the figures claim:

* **additivity.** Shapley values must reconstruct the model output exactly:
  ``base_value + sum(shap) == f(x)``. At 10 features ``KernelExplainer``
  enumerates all ``2**10`` coalitions, so this holds to machine precision and is a
  genuine comparator rather than a plausibility check.
* **determinism.** The figures and the summary CSV are compared across model
  states, which is meaningless if two runs of the same input disagree.
* **the mean function shows up where it must.** An objective carrying a declared
  physics trend on a feature had better attribute to that feature; if it does not,
  either the mean module is not reaching the posterior or the explained function
  is the wrong one.

The synthetic campaign needs no workbook. One test does and skips without it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobo_kit.campaign import (
    build_objective_transform,
    fit_campaign_models,
    run_r0_lhs,
    run_r2_qlognehvi,
)
from mobo_kit.candidate_pool import sample_discrete_candidate_pool
from mobo_kit.design import build_design_from_config
from mobo_kit.research_qnehvi import R2_ACQUISITIONS, run_r2_qnehvi_research

SOURCE = "local_inputs/Summary Table.xlsx"
SEED = 73
INPUT_DIM = 10


def _load():
    path = Path("scripts") / "plot_shap_attribution.py"
    spec = importlib.util.spec_from_file_location("_script_plot_shap", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


psa = _load()


def _config(pool: int = 256) -> dict:
    """Synthetic campaign with a log-link objective whose trend is on x0."""
    return {
        "inputs": [
            {"name": f"x{i}", "unit": "u", "start": 1.0, "stop": 2.0, "step": 0.05}
            for i in range(INPUT_DIM)
        ],
        "objectives": {
            "contract_version": "TEST_ONLY-shap-v1",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": "plain",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": "plain",
                    "lower_anchor": 0.0,
                    "upper_anchor": 3.0,
                },
                {
                    "name": "peaked",
                    "goal": "target",
                    "transform": "gaussian_target",
                    "model_source_column": "peaked",
                    "target": 650.0,
                    "sigma": 176.7766952966369,
                    "mean_function": {
                        "response": "log",
                        "features": [{"column": "x0", "transform": "log"}],
                    },
                },
            ],
        },
        "reference_point_utility": [-0.01, -0.01],
        "rounds": {
            "r1": {
                "method": "ucb_hvi", "batch_size": 5, "replicates_per_condition": 3,
                "beta": 4.0, "candidate_pool_size": pool, "posterior_samples": 16,
                "moment_method": "monte_carlo",
            },
            "r2": {
                "method": "qlognehvi", "batch_size": 3,
                "replicates_per_condition": 3,
                "candidate_pool_size": pool, "mc_samples": 8,
            },
        },
        "local_penalization": {
            "radius": 0.25, "min_batch_distance": 0.15,
            "min_observed_distance": 0.0, "dimension_weights": None,
        },
        "model": {"variant": "dim_scaled_prior"},
        "reproducibility": {"seed": SEED},
        "constraints": [],
    }


def _measurements(X: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for measured columns.

    ``peaked`` deliberately carries structure the declared mean function CANNOT
    absorb (the ``x1`` term), so the residual GP has non-zero posterior variance.
    Without it the trend fits exactly, the variance collapses, and
    ``expected_transform`` becomes indistinguishable from transforming the mean --
    which would make the quadrature test vacuous rather than passing.
    """
    X = np.asarray(X, dtype=float)
    return np.column_stack([
        X.mean(axis=1),
        650.0 * X[:, 0] ** -0.6 * X[:, 1] ** 0.3,
    ])


@pytest.fixture(scope="module")
def fitted():
    config = _config()
    transform = build_objective_transform(config)
    X = run_r0_lhs(config, n=15, seed=SEED).conditions.to_numpy(float)
    Y = _measurements(X)
    model, warnings = fit_campaign_models(config, X, Y, seed=SEED)
    assert not warnings
    design = build_design_from_config(dict(config))
    instances = np.asarray(
        sample_discrete_candidate_pool(design, 12, seed=SEED).X_phys, dtype=float
    )
    return {
        "config": config, "transform": transform, "X": X, "Y": Y,
        "model": model, "instances": instances, "design": design,
    }


# --------------------------------------------------------------------------- #
# the attribution itself
# --------------------------------------------------------------------------- #


def test_shap_values_have_one_column_per_campaign_input(fitted) -> None:
    values = psa.shap_values_for(
        fitted["model"], fitted["config"], fitted["transform"], 1,
        background=fitted["X"], instances=fitted["instances"], seed=SEED,
    )
    assert values.shape == (len(fitted["instances"]), INPUT_DIM)
    assert np.isfinite(values).all()


def test_attributions_reconstruct_the_model_output_exactly(fitted) -> None:
    """Additivity. The comparator that makes the rest of this meaningful.

    Shapley values are defined by summing to the difference between the model
    output and its expectation over the background. If that fails, the beeswarm is
    a picture of something other than the model.
    """
    import shap

    f = psa.expected_utility_fn(
        fitted["model"], fitted["config"], fitted["transform"], 1
    )
    explainer = shap.KernelExplainer(f, fitted["X"])
    values = np.asarray(explainer.shap_values(fitted["instances"], silent=True))
    reconstructed = float(explainer.expected_value) + values.sum(axis=1)
    np.testing.assert_allclose(
        reconstructed, f(fitted["instances"]), rtol=0, atol=1e-9
    )


def test_attributions_are_deterministic(fitted) -> None:
    kwargs = dict(
        background=fitted["X"], instances=fitted["instances"], seed=SEED
    )
    first = psa.shap_values_for(
        fitted["model"], fitted["config"], fitted["transform"], 1, **kwargs
    )
    second = psa.shap_values_for(
        fitted["model"], fitted["config"], fitted["transform"], 1, **kwargs
    )
    assert np.array_equal(first, second)
    # and the ranking a figure would draw is stable, not just the raw array
    assert np.array_equal(
        np.argsort(-np.abs(first).mean(axis=0)),
        np.argsort(-np.abs(second).mean(axis=0)),
    )


def test_the_declared_mean_function_feature_dominates(fitted) -> None:
    """`peaked` carries a log trend on x0 and nothing else; x0 must lead.

    This is the property the figures' construction caveat warns about, asserted
    rather than assumed -- and it doubles as a check that the structured mean
    reaches ``posterior()`` at all.
    """
    values = psa.shap_values_for(
        fitted["model"], fitted["config"], fitted["transform"], 1,
        background=fitted["X"], instances=fitted["instances"], seed=SEED,
    )
    mean_abs = np.abs(values).mean(axis=0)
    assert int(np.argmax(mean_abs)) == 0, "x0 carries the declared trend"
    assert mean_abs[0] > 2.0 * np.median(mean_abs)


def test_expected_utility_uses_the_lognormal_quadrature_not_the_mean(fitted) -> None:
    """The explained function must be E[utility], not utility(E[.]).

    For a peaked target on a lognormal posterior the two differ, and the second is
    biased by Jensen's inequality and blind to variance.
    """
    import torch

    from mobo_kit.campaign import normalise_inputs

    config, transform = fitted["config"], fitted["transform"]
    f = psa.expected_utility_fn(fitted["model"], config, transform, 1)
    expected = f(fitted["instances"])

    model = fitted["model"]
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(
            torch.tensor(
                normalise_inputs(config, fitted["instances"]), dtype=torch.double
            ),
            observation_noise=False,
        )
        naive = transform.transform(posterior.mean)[:, 1].numpy()
    assert np.isfinite(expected).all()
    assert not np.allclose(expected, naive, atol=1e-6), (
        "expected utility must differ from the transformed mean, or the "
        "quadrature is not being used"
    )


# --------------------------------------------------------------------------- #
# the qNEHVI research variant
# --------------------------------------------------------------------------- #


def test_qnehvi_proposes_a_valid_batch(fitted) -> None:
    config = fitted["config"]
    X = fitted["X"]
    Y = fitted["Y"]
    X1 = run_r0_lhs(config, n=5, seed=SEED + 1).conditions.to_numpy(float)
    X01 = np.vstack([X, X1])
    Y01 = np.vstack([Y, _measurements(X1)])

    result = run_r2_qnehvi_research(config, X01, Y01, seed=SEED)
    assert len(result.conditions) == 3
    report = result.diagnostics["validity"]
    assert report["unique"] and report["on_grid"] and report["in_bounds"]
    assert report["min_pairwise_distance"] >= 0.15
    assert result.diagnostics["method"] == "qnehvi"
    assert result.diagnostics["research_only"] is True


def test_both_acquisitions_are_selectable_and_named() -> None:
    assert R2_ACQUISITIONS == ("qlognehvi", "qnehvi")


def test_identical_batch_detection(fitted) -> None:
    """The detection logic, exercised on both outcomes.

    Whether the two acquisitions actually agree is a property of the data, not
    something a test should assert; what must work is noticing either way.
    """
    frame = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"]
    )
    same = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    different = frame.copy()
    different.iloc[0, 0] = 9.0
    assert psa.batch_hash(frame) == psa.batch_hash(same)
    assert psa.batch_hash(frame) != psa.batch_hash(different)


def test_the_two_acquisitions_run_on_the_same_inputs(fitted) -> None:
    """Both runners accept the same contract, so a comparison is apples to apples."""
    config, X, Y = fitted["config"], fitted["X"], fitted["Y"]
    X1 = run_r0_lhs(config, n=5, seed=SEED + 1).conditions.to_numpy(float)
    X01, Y01 = np.vstack([X, X1]), np.vstack([Y, _measurements(X1)])

    a = run_r2_qlognehvi(config, X01, Y01, seed=SEED)
    b = run_r2_qnehvi_research(config, X01, Y01, seed=SEED)
    assert list(a.conditions.columns) == list(b.conditions.columns)
    assert len(a.conditions) == len(b.conditions) == 3
    # both are valid batches whether or not they agree
    for result in (a, b):
        assert result.diagnostics["validity"]["on_grid"]


# --------------------------------------------------------------------------- #
# figures and captions
# --------------------------------------------------------------------------- #


def test_beeswarm_renders_headlessly(fitted, tmp_path) -> None:
    values = psa.shap_values_for(
        fitted["model"], fitted["config"], fitted["transform"], 1,
        background=fitted["X"], instances=fitted["instances"], seed=SEED,
    )
    path = tmp_path / "beeswarm.png"
    psa.plot_beeswarm(
        path, values, fitted["instances"], fitted["config"], "peaked",
        "test state", seed=SEED,
        caveats=psa.caveats_for("peaked", "final", True),
    )
    assert path.is_file() and path.stat().st_size > 0


def test_feature_labels_carry_the_physical_range(fitted) -> None:
    """The colorbar is per-feature normalised, so the units live in the labels."""
    labels = psa._feature_labels(fitted["config"], fitted["instances"])
    assert len(labels) == INPUT_DIM
    assert all("\n" in label for label in labels)
    assert labels[0].startswith("x0")
    assert "u" in labels[0], "unit must appear in the label"


def test_captions_state_only_what_is_true_of_that_figure() -> None:
    r0 = psa.caveats_for("thickness", "r0_only", identical=False)
    assert any("15 real" in line for line in r0)
    assert not any("Oracle:" in line for line in r0)

    final = psa.caveats_for("thickness", "final", identical=False)
    assert any("Oracle:" in line for line in final)
    assert any("mean function" in line for line in final)

    uniformity = psa.caveats_for("uniformity", "final", identical=False)
    assert any("fitted noise" in line for line in uniformity)

    with_note = psa.caveats_for("thickness", "final", identical=True)
    assert any("IDENTICAL" in line for line in with_note)
    # the R0 anchor never carries the acquisition note: it predates R2
    assert not any(
        "IDENTICAL" in line
        for line in psa.caveats_for("thickness", "r0_only", identical=True)
    )


# --------------------------------------------------------------------------- #
# end to end on the real workbook
# --------------------------------------------------------------------------- #


@pytest.mark.local_input
@pytest.mark.skipif(
    not Path(SOURCE).is_file(), reason=f"{SOURCE} is not present in this checkout"
)
def test_headless_smoke_on_the_real_workbook(tmp_path) -> None:
    import yaml

    from mobo_kit.campaign import load_campaign_config

    config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")
    config["rounds"]["r1"]["candidate_pool_size"] = 128
    config["rounds"]["r1"]["posterior_samples"] = 8
    config["rounds"]["r2"]["candidate_pool_size"] = 128
    config["rounds"]["r2"]["mc_samples"] = 4
    scratch = tmp_path / "campaign.yaml"
    scratch.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = tmp_path / "out"
    code = psa.main([
        "--workbook", SOURCE,
        "--config", str(scratch),
        "--output-dir", str(output),
        "--instances", "6",
        "--objectives", "thickness",
    ])
    assert code == 0

    summary = pd.read_csv(output / "shap_summary.csv")
    assert set(summary["objective"]) == {"thickness"}
    assert summary["rank"].min() == 1
    # one row per feature per model state
    assert len(summary) % INPUT_DIM == 0
    assert (output / "figures").is_dir()
    assert list((output / "figures").glob("*.png"))
