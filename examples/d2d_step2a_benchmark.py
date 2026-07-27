"""Optional TEST_ONLY 10,000-point CPU scoring benchmark for Step 2A."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source in (REPOSITORY_ROOT / "src", Path(__file__).resolve().parent):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from d2d_step2a_synthetic import build_test_only_d2d_design  # noqa: E402
from mobo_kit.candidate_pool import sample_discrete_candidate_pool  # noqa: E402
from mobo_kit.ucb_hvi import score_ucb_hvi_from_moments  # noqa: E402


def main() -> int:
    design = build_test_only_d2d_design()
    pool_started = perf_counter()
    pool = sample_discrete_candidate_pool(design, 10_000, seed=99173)
    pool_seconds = perf_counter() - pool_started

    X = pool.X_norm
    utility_mean = np.column_stack(
        [
            0.15 + 0.65 * X[:, 0],
            0.10 + 0.70 * X[:, 4],
            np.exp(-0.5 * ((X[:, 7] - 0.55) / 0.22) ** 2),
        ]
    )
    utility_std = 0.02 + 0.04 * np.column_stack([X[:, 1], 1.0 - X[:, 5], X[:, 9]])
    observed = np.array(
        [
            [0.35, 0.75, 0.60],
            [0.55, 0.55, 0.80],
            [0.75, 0.35, 0.65],
        ]
    )
    score_started = perf_counter()
    result = score_ucb_hvi_from_moments(
        utility_mean,
        utility_std,
        observed,
        np.array([-0.05, -0.05, -0.05]),
        beta=1.0,
        chunk_size=512,
        objective_contract_version="TEST_ONLY-benchmark-utilities-v1",
    )
    score_seconds = perf_counter() - score_started
    report = {
        "status": "TEST_ONLY_BENCHMARK",
        "production_candidate_generation": "NOT_RUN",
        "pool_size": pool.size,
        "pool_sampling_seconds": round(pool_seconds, 4),
        "ucb_hvi_scoring_seconds": round(score_seconds, 4),
        "positive_hvi_count": int(np.count_nonzero(result.base_score > 0)),
        "full_cartesian_grid_materialized": False,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
