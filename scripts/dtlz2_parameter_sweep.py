"""Is beta = 4.0 with radius = 0.25 a defensible default, or just the first guess?

Sweeps UCB ``beta`` against the local-penalization ``radius`` on DTLZ2 -- a
synthetic problem with a known Pareto front, so the answer does not depend on
whether the campaign's measurements are right.

**The decision rule is pre-committed, and it is written here before the numbers
exist so that reading them cannot move it.**  Keep 4.0 / 0.25 unless a cell beats
the current mean hypervolume gain by MORE than the per-seed standard deviation of
gains, AND does not reduce batch spacing.  A sweep that finds everything flat
within noise is a pass, not a failure: it says the default is not a lucky pick and
the knob does not need attention.

Each cell runs the real campaign path per seed -- ``run_r0_lhs`` -> ``run_r1_ucb``
-> ``run_r2_qlognehvi`` -- and is scored on hypervolume gain over the R0 start,
against a random on-grid baseline at the same budget, exactly as the acceptance
test does.

Boundary-coordinate counts are reported as a secondary readout because of what the
review artifact found on the live campaign: every proposed condition pinned
``anneal_temp`` to its range edge. That was traced to a monotone mean function
rather than to the acquisition, but a beta or radius that pushes batches onto
range edges by itself is worth seeing.

    python scripts/dtlz2_parameter_sweep.py            # 3 x 3 cells, 8 seeds
    python scripts/dtlz2_parameter_sweep.py --seeds 3  # a quicker look
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
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
from mobo_kit.candidate_pool import sample_discrete_candidate_pool
from mobo_kit.design import build_design_from_config

INPUT_DIM = 10
OBJECTIVES = 3
R0_SIZE = 15
R1_SIZE = 5
R2_SIZE = 3
ADDED = R1_SIZE + R2_SIZE

CURRENT_BETA = 4.0
CURRENT_RADIUS = 0.25
BETAS = (2.0, 4.0, 8.0)
RADII = (0.15, 0.25, 0.35)
#: Fixed across the sweep on purpose: it is a hard floor on batch spacing, not a
#: tuning knob, and moving it would change what "spacing" even means per cell.
MIN_BATCH_DISTANCE = 0.15


def _problem() -> DTLZ2:
    return DTLZ2(dim=INPUT_DIM, num_objectives=OBJECTIVES, negate=True).to(
        dtype=torch.double
    )


def _config(beta: float, radius: float, *, pool: int = 1024, mc_samples: int = 32) -> dict:
    return {
        "inputs": [
            {"name": f"x{i}", "start": 0.0, "stop": 1.0, "step": 0.05}
            for i in range(INPUT_DIM)
        ],
        "objectives": {
            "contract_version": "TEST_ONLY-dtlz2-sweep-v1",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": f"f{i}",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": f"f{i}",
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
                "beta": beta,
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
            "radius": radius,
            "min_batch_distance": MIN_BATCH_DISTANCE,
            "min_observed_distance": 0.0,
            "dimension_weights": None,
        },
        "model": {"variant": "dim_scaled_prior"},
        "reproducibility": {"seed": 73},
        "constraints": [],
    }


def _evaluate(problem: DTLZ2, X: np.ndarray) -> np.ndarray:
    return problem(torch.tensor(np.asarray(X, float), dtype=torch.double)).numpy()


def _hypervolume(config: dict, Y: np.ndarray) -> float:
    transform = build_objective_transform(config)
    reference = torch.tensor(config["reference_point_utility"], dtype=torch.double)
    utility = transform(torch.tensor(np.asarray(Y, float), dtype=torch.double))
    if not bool((utility >= reference).all(dim=-1).any()):
        # BoTorch would silently drop every point and return 0.0
        raise AssertionError("no point dominates the reference")
    return float(Hypervolume(ref_point=reference).compute(utility[is_non_dominated(utility)]))


def _boundary_counts(X: np.ndarray) -> int:
    """How many coordinates across the batch sit at 0 or 1, the grid's edges."""
    values = np.asarray(X, float)
    return int((np.isclose(values, 0.0) | np.isclose(values, 1.0)).sum())


@dataclass
class CellResult:
    beta: float
    radius: float
    bo_gain: list[float]
    random_gain: list[float]
    spacing: list[float]
    boundary: list[int]

    @property
    def mean_gain(self) -> float:
        return float(np.mean(self.bo_gain))

    @property
    def sd_gain(self) -> float:
        return float(np.std(self.bo_gain, ddof=1))

    @property
    def mean_random(self) -> float:
        return float(np.mean(self.random_gain))

    @property
    def mean_spacing(self) -> float:
        return float(np.mean(self.spacing))

    @property
    def mean_boundary(self) -> float:
        return float(np.mean(self.boundary))


def _one_seed(config: dict, seed: int) -> tuple[float, float, float, int]:
    """Returns (bo gain, random gain, min batch spacing, boundary coords)."""
    problem = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = run_r0_lhs(config, n=R0_SIZE, seed=seed)
        X0 = r0.conditions.to_numpy(float)
        Y0 = _evaluate(problem, X0)
        start = _hypervolume(config, Y0)

        r1 = run_r1_ucb(config, X0, Y0, seed=seed)
        X1 = r1.conditions.to_numpy(float)
        Y1 = _evaluate(problem, X1)
        X01, Y01 = np.vstack([X0, X1]), np.vstack([Y0, Y1])

        r2 = run_r2_qlognehvi(config, X01, Y01, seed=seed)
        X2 = r2.conditions.to_numpy(float)
        Y2 = _evaluate(problem, X2)
        bo = _hypervolume(config, np.vstack([Y01, Y2])) - start

        # random on-grid baseline at the same budget
        design = build_design_from_config(dict(config))
        pool = sample_discrete_candidate_pool(design, ADDED, seed=seed + 9999)
        Yr = _evaluate(problem, np.asarray(pool.X_phys, float)[:ADDED])
        random_gain = _hypervolume(config, np.vstack([Y0, Yr])) - start

    spacing = min(
        float(r1.diagnostics["validity"]["min_pairwise_distance"]),
        float(r2.diagnostics["validity"]["min_pairwise_distance"]),
    )
    return bo, random_gain, spacing, _boundary_counts(X1) + _boundary_counts(X2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--pool", type=int, default=1024)
    args = parser.parse_args()
    seeds = [73 + 11 * i for i in range(args.seeds)]

    print(f"DTLZ2 sweep: beta x radius, {len(seeds)} seeds per cell, pool {args.pool}")
    print(f"min_batch_distance fixed at {MIN_BATCH_DISTANCE} (a floor, not a knob)\n")
    print("PRE-COMMITTED RULE: keep beta=4.0 / radius=0.25 unless a cell beats its")
    print("mean HV gain by more than the per-seed sd of gains, without reducing")
    print("spacing. Flat within noise is a PASS.\n")

    cells: list[CellResult] = []
    for beta in BETAS:
        for radius in RADII:
            config = _config(beta, radius, pool=args.pool)
            rows = [_one_seed(config, seed) for seed in seeds]
            cell = CellResult(
                beta=beta,
                radius=radius,
                bo_gain=[r[0] for r in rows],
                random_gain=[r[1] for r in rows],
                spacing=[r[2] for r in rows],
                boundary=[r[3] for r in rows],
            )
            cells.append(cell)
            print(
                f"  beta={beta:<4g} radius={radius:<5g} "
                f"gain {cell.mean_gain:+.4f} (sd {cell.sd_gain:.4f})  "
                f"random {cell.mean_random:+.4f}  "
                f"spacing {cell.mean_spacing:.3f}  "
                f"edge coords {cell.mean_boundary:.1f}"
            )

    baseline = next(
        c for c in cells if c.beta == CURRENT_BETA and c.radius == CURRENT_RADIUS
    )
    threshold = baseline.mean_gain + baseline.sd_gain

    print("\n" + "=" * 78)
    print(
        f"current default beta={CURRENT_BETA} radius={CURRENT_RADIUS}: "
        f"mean gain {baseline.mean_gain:+.4f}, per-seed sd {baseline.sd_gain:.4f}"
    )
    print(f"a challenger must exceed {threshold:+.4f} AND not reduce spacing below "
          f"{baseline.mean_spacing:.3f}")

    challengers = [
        c
        for c in cells
        if c.mean_gain > threshold and c.mean_spacing >= baseline.mean_spacing
    ]
    if challengers:
        best = max(challengers, key=lambda c: c.mean_gain)
        print(
            f"\nRULE TRIGGERED: beta={best.beta} radius={best.radius} gives "
            f"{best.mean_gain:+.4f} at spacing {best.mean_spacing:.3f}."
        )
    else:
        near = [c for c in cells if c.mean_gain > baseline.mean_gain]
        print(
            f"\nNO CHANGE. {len(near)} of {len(cells)} cells have a higher mean gain, "
            "none by more than one per-seed sd while holding spacing. The default is "
            "flat within noise, which is the outcome that says it was not a lucky pick."
        )

    print(
        f"\nBO beats random in {sum(c.mean_gain > c.mean_random for c in cells)} "
        f"of {len(cells)} cells on the mean."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
