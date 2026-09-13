"""Exact leave-one-out for one objective, with its declared structured mean.

There is one fold loop in this project and this is it.  ``scripts/intake_new_data.py``
is canonical for LOO numbers, the round report plots them, and
``scripts/permutation_rank_test.py`` builds a null out of them -- so they must be
*the same* numbers rather than three implementations that agree today.  This module
exists because they briefly were three implementations.

Two rules the fold loop keeps, both of which flatter the result if broken:

* **the trend is refit inside every fold**, on the training rows only.  Fitting it
  once on everything and holding it fixed leaks the held-out value into the mean
  function.
* **the model is refit from scratch per fold** under the campaign's own variant and
  seeding, so this reproduces the round's model rather than a similar one.

:func:`model_validation.run_exact_loocv` is the multi-output validation harness and
is a different tool: it fits one N-1 model for all objectives at once and does not
take a per-objective mean module.  Objectives here carry different structured means,
so they are fitted one at a time.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .model_validation import DIM_SCALED_PRIOR, SIGNAL_COLLAPSE_STAGE, fit_model_variant
from .structured_mean import build_structured_mean, mean_spec_from_config

__all__ = ["LooResult", "loo_predictions", "null_loo_r2", "resolution_sd"]

#: Bootstrap sd of LOO R2 measured at N=15 on the first campaign, 4000 resamples.
#: Scaled by ``sqrt(15/N)`` elsewhere, which is an approximation -- re-run the
#: bootstrap if a decision turns on the third decimal.
RESOLUTION_SD_AT_15 = 0.236
RESOLUTION_REFERENCE_N = 15


def null_loo_r2(n: int) -> float:
    """What predicting the leave-one-out mean scores, independent of the data.

    ``1 - (N/(N-1))^2``: -0.148 at 15, -0.105 at 21, -0.069 at 31.  It moves with
    N, so recompute rather than reuse.

    **THIS IS NOT A SIGNIFICANCE THRESHOLD, and this project read it as one for a
    year.**  It is the score of ONE SPECIFIC PREDICTOR -- predict every held-out
    row with the mean of the others -- and a fitted GP does not behave like that
    predictor.  Measured on the v4 campaign, 2026-09-04, 300 permutations of a
    real objective with the campaign's own model variant:

        fitted GP under permuted y   median -0.4210   95th percentile +0.2890
        fraction of pure-noise draws scoring above -0.1480:   28.7%

    So "beats the null" happens better than one time in four when there is no
    signal at all.  The GP makes real predictions with about six times the spread
    of the constant predictor; they are noise, and they land FURTHER from y, which
    is why the empirical null sits well below this number while its upper tail
    sits well above it.

    Below this value a model has certainly learned nothing.  ABOVE it means
    nothing on its own.  The honest single-candidate bar is the 95th percentile of
    that candidate's own permutation null -- ``scripts/raw_component_screen.py
    --calibrate`` measures it -- and the project's standing adjudicator for a
    real verdict is the rank permutation test in
    ``scripts/permutation_rank_test.py``, which was always the right instrument
    and is now the only one.

    A declared mean function LOWERS this empirical null rather than raising it:
    an OLS trend fitted on N-1 rows of shuffled y is a noise fit, and
    extrapolating it to the held-out row adds error.  Median goes -0.4075 with no
    mean, -0.4368 with one feature, -0.5384 with two.
    """
    if n < 2:
        raise ValueError("The leave-one-out null needs at least two rows.")
    return 1.0 - (n / (n - 1)) ** 2


def resolution_sd(n: int) -> float:
    """Sampling sd of LOO R2 at this N, scaled from the N=15 bootstrap."""
    if n < 1:
        raise ValueError("resolution_sd needs at least one row.")
    return RESOLUTION_SD_AT_15 * math.sqrt(RESOLUTION_REFERENCE_N / n)


@contextlib.contextmanager
def _single_threaded_torch():
    """Run the fold loop on one thread, and put the setting back afterwards.

    Every fold here fits a GP on an N-1 x D design -- 14 x 10 on this campaign.
    At that size intra-op threading costs more than it buys. Measured on an IDLE
    machine, 45 fits: **9.8 s at 1, 2 or 4 threads against 15.1 s at this box's
    default of 12**, for bit-identical results (LOO R2 -0.6447 / -0.5842 / +0.6630
    at every setting).

    The idleness matters and is not a footnote. The first version of this comment
    claimed 51 s against 117 s, which was measured while sixteen permutation
    workers were saturating the CPU -- a real effect, but of the load and not of
    the thread count. A timing taken under contention is a wrong number in the
    same way any other unreproduced number is, so it was re-measured before being
    written down.

    Scoped rather than set globally, because the same setting would SLOW the
    round itself down: scoring a 32768-point candidate pool is exactly the large
    matrix work threads are for. Not safe to call while another thread in this
    process is computing with torch.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True)
class LooResult:
    """One objective's exact leave-one-out predictions, in both spaces."""

    objective: str
    #: What the GP emits: ``log(y)`` when the objective declares a log-response
    #: mean function, ``y`` otherwise. This is what a utility transform consumes.
    mean_model_space: np.ndarray
    variance_model_space: np.ndarray
    #: The measurement's own units -- nanometres for thickness. This is what a
    #: parity plot must show, because nobody reads log(nm).
    predicted: np.ndarray
    predictive_sd: np.ndarray
    observed: np.ndarray
    r2: float
    spearman: float
    has_mean_function: bool
    model_link: str
    collapse_warnings: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.observed)


def loo_predictions(
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    X_phys: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 73,
    use_mean_function: bool = True,
    y_var: np.ndarray | None = None,
) -> LooResult:
    """Exact leave-one-out for one objective under the campaign's own contract.

    ``y`` is in MEASUREMENT space, as :func:`workbook_io.read_campaign_workbook`
    returns it.  Set ``use_mean_function=False`` for the plain-GP comparison the
    intake verdict rests on.

    ``y_var`` is the measured observation variance per row, as a round under
    ``replicate_pooled`` hands it to the model; each fold keeps the variances of
    the rows it keeps.  Rows must be RECIPES (condition means), never single
    replicate films: holding out one of three films leaves its two siblings at the
    same inputs in the training set and flatters the score.
    """
    from .campaign import normalise_inputs
    from scipy.stats import spearmanr

    X_phys = np.asarray(X_phys, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError(f"Leave-one-out needs at least three rows; got {n}.")

    mean_spec = mean_spec_from_config(entry) if use_mean_function else None
    log_response = mean_spec is not None and mean_spec.response == "log"
    design_names = [item["name"] for item in config["inputs"]]
    lowers = np.array([float(item["start"]) for item in config["inputs"]])
    uppers = np.array([float(item["stop"]) for item in config["inputs"]])
    X_norm = normalise_inputs(config, X_phys)

    variances = None
    if y_var is not None:
        variances = np.asarray(y_var, dtype=float).reshape(-1)
        if variances.shape != y.shape:
            raise ValueError(
                f"y_var has {variances.size} rows for {n} observations; it must be "
                "one measured variance per row of y."
            )
    mu = np.empty(n)
    var = np.empty(n)
    collapse: list[str] = []
    with _single_threaded_torch():
        for held in range(n):
            keep = [i for i in range(n) if i != held]
            target = y[keep]
            module = None
            if mean_spec is not None:
                module, target = build_structured_mean(
                    X_phys[keep], y[keep], mean_spec, design_names, lowers, uppers
                )
            torch.manual_seed(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = fit_model_variant(
                    torch.tensor(X_norm[keep], dtype=torch.double),
                    torch.tensor(target, dtype=torch.double).unsqueeze(-1),
                    sample_ids=tuple(range(len(keep))),
                    objective_names=("y",),
                    variant=DIM_SCALED_PRIOR,
                    seed=seed,
                    mean_module=module,
                    train_Yvar=(
                        None
                        if variances is None
                        else torch.tensor(
                            variances[keep], dtype=torch.double
                        ).unsqueeze(-1)
                    ),
                )
            collapse.extend(
                w.message for w in record.warnings if w.stage == SIGNAL_COLLAPSE_STAGE
            )
            gp = record.model.models[0]
            gp.eval()
            with torch.no_grad():
                posterior = gp.posterior(
                    torch.tensor(X_norm[held : held + 1], dtype=torch.double)
                )
                mu[held] = float(posterior.mean.reshape(-1)[0])
                var[held] = float(max(0.0, float(posterior.variance.reshape(-1)[0])))

    if log_response:
        # The GP emits log(y), so the measurement-space POINT prediction is the
        # median exp(mu), matching the convention the round simulation and the
        # SHAP oracle both use. The sd is the lognormal one; it is asymmetric in
        # nanometres, which is why the parity plot labels it as an interval rather
        # than pretending to a symmetric error bar.
        predicted = np.exp(mu)
        predictive_sd = np.sqrt(np.clip(np.exp(var) - 1.0, 0.0, None)) * np.exp(
            mu + var / 2.0
        )
    else:
        predicted = mu
        predictive_sd = np.sqrt(var)

    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rank = (
        float(spearmanr(y, predicted).statistic)
        if len(np.unique(predicted)) > 1
        else float("nan")
    )

    return LooResult(
        objective=str(entry.get("name", "<unnamed>")),
        mean_model_space=mu,
        variance_model_space=var,
        predicted=predicted,
        predictive_sd=predictive_sd,
        observed=y,
        r2=r2,
        spearman=rank,
        has_mean_function=mean_spec is not None,
        model_link="log" if log_response else "identity",
        collapse_warnings=tuple(dict.fromkeys(collapse)),
    )


def loo_for_objectives(
    config: Mapping[str, Any],
    X_phys: np.ndarray,
    Y_measured: np.ndarray,
    *,
    names: Sequence[str],
    seed: int = 73,
) -> dict[str, LooResult]:
    """One :class:`LooResult` per objective, each with its own declared mean."""
    entries = config["objectives"]["specs"]
    out: dict[str, LooResult] = {}
    for index, name in enumerate(names):
        out[name] = loo_predictions(
            config, entries[index], X_phys, np.asarray(Y_measured)[:, index], seed=seed
        )
    return out
