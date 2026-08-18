"""Shapley attribution over the campaign's own fitted models.

**What is explained is ``E[utility]``, not the posterior mean.**  For thickness the
posterior is lognormal and the utility is a peaked Gaussian on a 650 nm target, so
transforming the mean is biased by Jensen's inequality and blind to the variance
that a target-seeking utility depends on.  ``expected_transform`` is the correct
route and is what the acquisition consumes.

**Attributions explain the MODEL, not the world.**  Where a feature appears in an
objective's declared ``mean_function``, the model was *told* that relationship by
the config; SHAP recovering it is a consistency check, not a discovery.  And on an
objective with no learnable signal the attributions are structure fitted to noise:
they have real magnitude and orderly ranking and mean nothing.  Both campaigns have
produced exactly that picture for uniformity, and it is the most persuasive figure
in the set.

This module owns the explainer.  ``scripts/plot_shap_attribution.py`` owns the
beeswarm figures and the extreme-cell comparison, and imports from here; nothing is
duplicated between them.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .campaign import normalise_inputs
from .objectives import ObjectiveTransform

__all__ = [
    "EXACT_ENUMERATION_FEATURE_LIMIT",
    "expected_utility_fn",
    "mean_absolute_shap",
    "shap_values_for",
]

#: ``KernelExplainer`` enumerates every one of ``2**d`` coalitions at or below this
#: many features, which makes the result the EXACT Shapley decomposition rather
#: than a sampled approximation -- and therefore independent of the seed.  Above
#: it, the values become a sample and the seed starts to matter.
EXACT_ENUMERATION_FEATURE_LIMIT = 10


def expected_utility_fn(
    model: Any,
    config: Mapping[str, Any],
    transform: ObjectiveTransform,
    objective_index: int,
    *,
    batch_rows: int = 65536,
):
    """``X_phys -> E[utility]`` for one objective, batched and deterministic.

    ``batch_rows`` must stay ABOVE one KernelExplainer block, which is
    ``coalitions x background rows`` -- 1022 x 23 = 23506 for a 23-point model.
    Splitting a block is not merely twice the work: measured on this stack, the
    same 23506 rows cost 0.03 s in one call and 0.59 s in two, a 20x penalty that
    turned a 37 s attribution run into 470 s. Raise it if the background grows.
    """

    def f(X_phys: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(X_phys, dtype=float))
        out = np.empty(values.shape[0], dtype=float)
        model.eval()
        with torch.no_grad():
            for start in range(0, values.shape[0], batch_rows):
                block = values[start : start + batch_rows]
                X_norm = normalise_inputs(config, block)
                posterior = model.posterior(
                    torch.tensor(X_norm, dtype=torch.double),
                    observation_noise=False,
                )
                utility = transform.expected_transform(
                    posterior.mean, posterior.variance.clamp_min(0.0)
                )
                out[start : start + block.shape[0]] = (
                    utility[..., objective_index].detach().cpu().double().numpy()
                )
        return out

    return f


def shap_values_for(
    model: Any,
    config: Mapping[str, Any],
    transform: ObjectiveTransform,
    objective_index: int,
    background: np.ndarray,
    instances: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Exact Shapley values over the campaign inputs.

    ``KernelExplainer`` with the default sample budget enumerates **every** one of
    the ``2**10 = 1024`` coalitions at this feature count, so the result is the
    exact Shapley decomposition rather than a sampled approximation -- and is
    therefore reproducible without depending on the seed. The seed is set anyway,
    because that stops being true the moment anyone adds an eleventh input.
    """
    import shap  # imported here: a heavy dependency only this path needs

    np.random.seed(int(seed))
    f = expected_utility_fn(model, config, transform, objective_index)
    explainer = shap.KernelExplainer(f, np.asarray(background, dtype=float))
    values = explainer.shap_values(np.asarray(instances, dtype=float), silent=True)
    return np.asarray(values, dtype=float)


def mean_absolute_shap(values: np.ndarray) -> np.ndarray:
    """Mean ``|SHAP|`` per feature -- the magnitude a bar chart ranks by.

    Separate from the signed mean on purpose.  A feature with a large mean
    ``|SHAP|`` and a near-zero signed mean matters in both directions: that is a
    non-monotone effect, not a weak one, and averaging the signed values would
    report it as nothing.
    """
    return np.abs(np.asarray(values, dtype=float)).mean(axis=0)
