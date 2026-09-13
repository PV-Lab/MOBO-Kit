"""Replicate variance into train_Yvar.

Wired and tested against synthetic replicates now, so that the arrival of the R1
triplicates is a data event rather than a code event.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from mobo_kit.model_validation import DIM_SCALED_PRIOR, fit_model_variant
from mobo_kit.replicate_variance import (
    WITHIN_FILM_LOG_THICKNESS_VARIANCE,
    PooledVariance,
    pool_between_film_variance,
    sanity_floor_findings,
    train_yvar_for_rows,
    variance_config,
)

NAMES = ["uniformity", "optoelectronic", "thickness"]


def _spread(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, columns=NAMES)


def _films(counts: dict[str, list[int]]) -> pd.DataFrame:
    return pd.DataFrame(counts, columns=NAMES)


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #


def test_pooling_is_dof_weighted() -> None:
    """sum((n-1) s^2) / sum(n-1): a condition with more films counts for more."""
    spread = _spread({"uniformity": [0.2, 0.4], "optoelectronic": [0.1, 0.1], "thickness": [0.3, 0.5]})
    films = _films({"uniformity": [3, 3], "optoelectronic": [3, 3], "thickness": [2, 4]})
    pooled = pool_between_film_variance(spread, films)

    assert pooled["uniformity"].variance == pytest.approx((2 * 0.04 + 2 * 0.16) / 4)
    assert pooled["uniformity"].dof == 4
    # thickness: one dof at 0.09, three at 0.25
    assert pooled["thickness"].variance == pytest.approx((1 * 0.09 + 3 * 0.25) / 4)
    assert pooled["thickness"].dof == 4


def test_a_single_film_contributes_no_dof_rather_than_zero_variance() -> None:
    """One film measures no reproducibility. Counting it as zero variance is how a
    model ends up certain about a process nobody measured twice."""
    spread = _spread(
        {"uniformity": [0.2, float("nan")], "optoelectronic": [0.2, float("nan")], "thickness": [0.2, float("nan")]}
    )
    films = _films({"uniformity": [3, 1], "optoelectronic": [3, 1], "thickness": [3, 1]})
    pooled = pool_between_film_variance(spread, films)
    assert pooled["thickness"].variance == pytest.approx(0.04)
    assert pooled["thickness"].dof == 2
    assert pooled["thickness"].n_conditions == 1


def test_no_replicated_condition_at_all_is_refused() -> None:
    spread = _spread({name: [float("nan")] for name in NAMES})
    films = _films({name: [1] for name in NAMES})
    with pytest.raises(ValueError, match="no condition with two or more usable films"):
        pool_between_film_variance(spread, films)


def test_the_pooled_space_follows_the_aggregation_rule() -> None:
    """Thickness aggregates in log space, so its variance is of log(T). Recording
    the space is what stops an nm^2 variance reaching a model that trains on logs."""
    spread = _spread({name: [0.2, 0.2] for name in NAMES})
    films = _films({name: [3, 3] for name in NAMES})
    pooled = pool_between_film_variance(
        spread, films, aggregates={"thickness": "mean_of_log", "uniformity": "mean"}
    )
    assert pooled["thickness"].space == "log"
    assert pooled["uniformity"].space == "value"


def test_sd_and_variance_of_the_mean() -> None:
    pooled = PooledVariance("thickness", 0.09, dof=10, n_conditions=5, space="log")
    assert pooled.sd == pytest.approx(0.3)
    # three films average to a third of the variance
    assert pooled.variance_of_mean(3) == pytest.approx(0.03)
    with pytest.raises(ValueError):
        pooled.variance_of_mean(0)


# --------------------------------------------------------------------------- #
# the sanity floor
# --------------------------------------------------------------------------- #


def test_between_film_variance_below_the_within_film_floor_is_reported() -> None:
    """Films cannot be more reproducible than points on one film."""
    pooled = {"thickness": PooledVariance("thickness", 0.01, 10, 5, "log")}
    messages = sanity_floor_findings(
        pooled, {"thickness": WITHIN_FILM_LOG_THICKNESS_VARIANCE}
    )
    assert len(messages) == 1
    assert "BELOW the within-film floor" in messages[0]
    assert "0.0593" in messages[0]


def test_a_healthy_between_film_variance_says_nothing() -> None:
    pooled = {"thickness": PooledVariance("thickness", 0.2, 10, 5, "log")}
    assert sanity_floor_findings(pooled, {"thickness": WITHIN_FILM_LOG_THICKNESS_VARIANCE}) == ()


def test_the_floor_is_a_floor_not_the_estimate() -> None:
    """Guards the substitution this whole module exists to prevent: the within-film
    number is not an answer, it is a lower bound on one."""
    assert WITHIN_FILM_LOG_THICKNESS_VARIANCE == pytest.approx(0.0593)
    pooled = {"thickness": PooledVariance("thickness", 0.0593, 10, 5, "log")}
    # equal to the floor is not below it
    assert sanity_floor_findings(pooled, {"thickness": WITHIN_FILM_LOG_THICKNESS_VARIANCE}) == ()


def test_the_archived_v2_config_declares_its_log_floor_and_the_r0_policy() -> None:
    """The FIRST campaign's config, which trained thickness on log T. The live
    contract's floor is pinned in test_final_campaign.py, in nm^2."""
    from mobo_kit.campaign import load_campaign_config

    config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")
    settings = variance_config(config)
    assert settings["sanity_floor"]["thickness"] == pytest.approx(0.0593)
    assert settings["rows_without_replicates"] == 1


# --------------------------------------------------------------------------- #
# per-row variance
# --------------------------------------------------------------------------- #


def test_the_variance_handed_to_the_model_is_of_the_mean() -> None:
    """The observation is an average of n films, so its variance is pooled / n.
    Passing the single-film variance would be three times too large on a triplicate
    -- overstating the uncertainty of exactly the conditions that were replicated
    most carefully -- and nothing errors."""
    pooled = {name: PooledVariance(name, 0.09, 10, 5, "value") for name in NAMES}
    films = _films({name: [3, 3, 1] for name in NAMES})
    yvar = train_yvar_for_rows(pooled, films, NAMES)
    assert yvar.shape == (3, 3)
    assert yvar[0, 0] == pytest.approx(0.03)
    assert yvar[2, 0] == pytest.approx(0.09)  # one film carries the full variance


def test_rows_without_replicates_take_the_declared_film_count() -> None:
    pooled = {name: PooledVariance(name, 0.09, 10, 5, "value") for name in NAMES}
    counts = np.zeros((2, 3))
    yvar = train_yvar_for_rows(pooled, counts, NAMES, rows_without_replicates=1)
    assert np.allclose(yvar, 0.09)


def test_identical_replicates_are_refused_rather_than_called_exact() -> None:
    """Zero variance tells the model the observation is exact. Films that agree to
    the last digit are a transcription, not a measurement."""
    pooled = {name: PooledVariance(name, 0.0, 10, 5, "value") for name in NAMES}
    with pytest.raises(ValueError, match="transcription, not a measurement"):
        train_yvar_for_rows(pooled, np.full((2, 3), 3.0), NAMES)


def test_a_missing_objective_is_refused() -> None:
    pooled = {"uniformity": PooledVariance("uniformity", 0.09, 10, 5, "value")}
    with pytest.raises(ValueError, match="No pooled variance"):
        train_yvar_for_rows(pooled, np.ones((2, 3)), NAMES)


# --------------------------------------------------------------------------- #
# reaching the model
# --------------------------------------------------------------------------- #


def test_measured_variance_replaces_the_fitted_noise() -> None:
    """With train_Yvar the noise is given, not inferred: the likelihood becomes a
    fixed-noise one carrying a value per observation."""
    X = torch.rand(10, 2, dtype=torch.double)
    Y = (3.0 * X[:, :1] + 0.05 * torch.randn(10, 1, dtype=torch.double)).double()
    Yvar = torch.full_like(Y, 0.04)

    record = fit_model_variant(
        X,
        Y,
        sample_ids=tuple(range(10)),
        objective_names=("y",),
        variant=DIM_SCALED_PRIOR,
        train_Yvar=Yvar,
    )
    gp = record.model.models[0]
    assert type(gp.likelihood).__name__ == "FixedNoiseGaussianLikelihood"
    assert gp.likelihood.noise.detach().reshape(-1).numel() == 10


def test_the_variance_is_taken_in_original_units_not_standardized() -> None:
    """`Standardize` rescales train_Yvar along with the targets, so it must arrive
    in the target's own units. A pre-standardized variance would be wrong by
    var(Y) and would fail silently."""
    X = torch.rand(10, 2, dtype=torch.double)
    Y = (100.0 * X[:, :1]).double()
    Yvar = torch.full_like(Y, 25.0)

    record = fit_model_variant(
        X,
        Y,
        sample_ids=tuple(range(10)),
        objective_names=("y",),
        variant=DIM_SCALED_PRIOR,
        train_Yvar=Yvar,
    )
    gp = record.model.models[0]
    observed = float(gp.likelihood.noise.detach().reshape(-1)[0])
    assert observed == pytest.approx(25.0 / float(Y.var()), rel=1e-6)


def test_heteroskedastic_variance_survives_to_the_model() -> None:
    """Rows with fewer films are noisier, and the model has to see that rather than
    one averaged number."""
    X = torch.rand(8, 2, dtype=torch.double)
    Y = (2.0 * X[:, :1]).double()
    Yvar = torch.tensor([[0.01]] * 4 + [[0.09]] * 4, dtype=torch.double)

    record = fit_model_variant(
        X,
        Y,
        sample_ids=tuple(range(8)),
        objective_names=("y",),
        variant=DIM_SCALED_PRIOR,
        train_Yvar=Yvar,
    )
    noise = record.model.models[0].likelihood.noise.detach().reshape(-1)
    assert noise[0] < noise[-1]
    assert float(noise[-1] / noise[0]) == pytest.approx(9.0, rel=1e-6)


def test_a_round_accepts_measured_variance_end_to_end() -> None:
    """The whole point: when the triplicates land, this is a data change."""
    from mobo_kit.campaign import load_campaign_config, run_r1_ucb
    from mobo_kit.design import build_design_from_config
    from mobo_kit.lhs import lhs_dataframe_optimized

    config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")
    design = build_design_from_config(dict(config))
    X = lhs_dataframe_optimized(design, 12, seed=5, snap_to_grids=True).to_numpy(float)
    names = list(design.names)
    speed = X[:, names.index("speed_1")]
    concentration = X[:, names.index("precur_conc")]
    temperature = X[:, names.index("anneal_temp")]
    rng = np.random.default_rng(0)

    thickness = np.exp(
        9.6 - 0.38 * np.log(speed) + 0.5 * np.log(concentration) + rng.normal(0, 0.08, 12)
    )
    optoelectronic = -6.2 - 0.012 * temperature + rng.normal(0, 0.05, 12)
    uniformity = np.clip(0.5 + 0.4 * np.sin(concentration * 3.0), 0.02, 0.98)
    Y = np.column_stack([uniformity, optoelectronic, thickness])

    # The declared variance must be smaller than the observed spread, or the model
    # is being told its signal is noise -- which the collapse guard rightly refuses.
    # The first version of this test declared sd 0.1 against a uniformity spread of
    # 0.039 and was correctly rejected.
    pooled = {
        "uniformity": PooledVariance("uniformity", 0.002, 10, 5, "value"),
        "optoelectronic": PooledVariance("optoelectronic", 0.02, 10, 5, "value"),
        # log space, matching `response: log` and the mean_of_log aggregation
        "thickness": PooledVariance("thickness", 0.0593, 10, 5, "log"),
    }
    yvar = train_yvar_for_rows(pooled, np.full((12, 3), 3.0), list(pooled))

    result = run_r1_ucb(config, X, Y, n=3, observed_Yvar=yvar)
    assert result.n_conditions == 3
    assert math.isfinite(result.diagnostics["validity"]["min_pairwise_distance"])


# --------------------------------------------------------------------------- #
# the live contract, after the R1 triplicates (2026-09-10)
# --------------------------------------------------------------------------- #

LIVE = "configs/campaign_d2d_perovskite_final.yaml"


def _results(spread: float):
    from types import SimpleNamespace

    return SimpleNamespace(
        replicate_spread=_spread({name: [spread, spread] for name in NAMES}),
        films_used=_films({name: [3, 3] for name in NAMES}),
    )


def _pooled_config(path: str, **thickness: str) -> dict:
    import copy

    from mobo_kit.campaign import load_campaign_config

    config = copy.deepcopy(load_campaign_config(path))
    config["model"]["observation_noise"] = "replicate_pooled"
    for entry in config["objectives"]["specs"]:
        if entry["name"] == "thickness":
            entry.update(thickness)
    return config


def test_the_live_config_pools_by_film_count_without_a_floor_alarm() -> None:
    """The live config's thickness rule is accepted; a single-film row takes the
    whole pooled variance and a triplicate a third; and an nm-scale variance clears
    the 91.2 nm^2 floor. That the variance really arrives in nm is exercised on the
    real sheets in test_final_campaign.py."""
    from mobo_kit.campaign import replicate_aggregates
    from mobo_kit.replicate_variance import yvar_for_campaign

    config = _pooled_config(LIVE)
    yvar, findings = yvar_for_campaign(
        config,
        _results(20.0),
        n_rows_without_replicates=2,
        objective_names=NAMES,
        aggregates=replicate_aggregates(config),
    )
    column = NAMES.index("thickness")
    assert yvar[0, column] == pytest.approx(400.0)
    assert yvar[-1, column] == pytest.approx(400.0 / 3)
    # 400 nm^2 sits well above the 91.2 nm^2 floor of a film mean
    assert findings == ()


def test_a_log_aggregation_on_a_nanometre_model_is_refused() -> None:
    """The failure that would have shipped silently: the model learns thickness in
    nm, the aggregation pooled log T, and 0.0019 went in as nm^2."""
    from mobo_kit.campaign import CampaignConfigError, replicate_aggregates

    config = _pooled_config(LIVE, replicate_aggregate="mean_of_log")
    with pytest.raises(CampaignConfigError, match="'identity' link"):
        replicate_aggregates(config)


def test_a_log_model_keeps_its_log_aggregation_and_refuses_the_reverse() -> None:
    """The archived v2 contract trains thickness on log T under a log mean
    function; the guard must leave that pairing alone."""
    from mobo_kit.campaign import CampaignConfigError, replicate_aggregates

    config = _pooled_config("configs/campaign_d2d_perovskite.yaml")
    assert replicate_aggregates(config)[NAMES.index("thickness")] == "mean_of_log"
    config = _pooled_config(
        "configs/campaign_d2d_perovskite.yaml", replicate_aggregate="mean"
    )
    with pytest.raises(CampaignConfigError, match="'log' link"):
        replicate_aggregates(config)


def test_a_variance_the_model_cannot_hold_is_refused_not_floored() -> None:
    """BoTorch standardizes train_Yvar and gpytorch floors the result at 1e-6, so a
    variance in the wrong units used to land on that floor with only a filtered
    library warning. A round now checks the noise it asked for is the noise the
    model holds."""
    from mobo_kit.campaign import (
        CampaignConfigError,
        fit_campaign_models,
        load_campaign_config,
    )
    from mobo_kit.design import build_design_from_config
    from mobo_kit.lhs import lhs_dataframe_optimized

    config = load_campaign_config(LIVE)
    design = build_design_from_config(dict(config))
    X = lhs_dataframe_optimized(design, 12, seed=5, snap_to_grids=True).to_numpy(float)
    rng = np.random.default_rng(0)
    Y = np.column_stack(
        [
            rng.uniform(0.6, 0.9, 12),
            rng.uniform(0.45, 0.75, 12),
            rng.uniform(380.0, 1300.0, 12),
        ]
    )
    measured = np.column_stack(
        [np.full(12, 0.002), np.full(12, 0.001), np.full(12, 660.0)]
    )
    model, _ = fit_campaign_models(config, X, Y, seed=73, Yvar=measured)
    assert model is not None

    wrong_units = measured.copy()
    wrong_units[:, 2] = 0.0019 / 3.0  # a variance of log T, handed to an nm model
    with pytest.raises(CampaignConfigError, match="thickness"):
        fit_campaign_models(config, X, Y, seed=73, Yvar=wrong_units)
