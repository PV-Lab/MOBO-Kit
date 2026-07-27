"""CPU-only TEST_ONLY exercise of the Step 2A computational core.

This script never reads a campaign workbook and never authorizes or emits real
D2D R1/R2 recipes. All observations, objective settings, reference values, and
acquisition parameters below are synthetic fixtures for software verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mobo_kit.batch_selection import LocalPenalizationConfig  # noqa: E402
from mobo_kit.candidate_diagnostics import (  # noqa: E402
    plot_candidate_pca,
    plot_distance_heatmap,
    plot_parallel_coordinates,
    plot_selection_scores,
    summarize_candidate_batch,
)
from mobo_kit.candidate_pool import (  # noqa: E402
    CandidatePool,
    sample_discrete_candidate_pool,
)
from mobo_kit.design import InputSpec, build_design  # noqa: E402
from mobo_kit.objectives import (  # noqa: E402
    ConfiguredMCMultiOutputObjective,
    ObjectiveSpec,
    ObjectiveTransform,
)
from mobo_kit.qlognehvi_batch import (  # noqa: E402
    propose_qlognehvi_penalized_batch,
)
from mobo_kit.ucb_hvi import propose_ucb_hvi_batch  # noqa: E402


TEST_ONLY_REFERENCE_POINT_UTILITY = np.array([-0.05, -0.05, -0.05])
TEST_ONLY_SEED = 20260724


@dataclass(frozen=True)
class SyntheticRunSummary:
    elapsed_seconds: float
    observed_count: int
    ucb_pool_size: int
    qlognehvi_pool_size: int
    ucb_selected_pool_indices: np.ndarray
    qlognehvi_selected_pool_indices: np.ndarray
    ucb_minimum_distance: float
    qlognehvi_minimum_distance: float
    ucb_metadata: dict[str, Any]
    qlognehvi_metadata: dict[str, Any]
    plot_paths: tuple[Path, ...]


def _json_safe(value: Any) -> Any:
    """Convert nested NumPy/PyTorch metadata to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _selection_table(selection: Any) -> list[dict[str, Any]]:
    """Return all score, penalty, distance, and order fields for a batch."""
    return [
        {
            "order": step.order,
            "pool_index": step.pool_index,
            "base_score": step.base_score,
            "base_log_score": step.base_log_score,
            "penalty_factor": step.penalty_factor,
            "log_penalty": step.log_penalty,
            "penalized_log_score": step.penalized_log_score,
            "nearest_selected_distance_before": (step.nearest_selected_distance_before),
            "nearest_observed_distance": step.nearest_observed_distance,
        }
        for step in selection.steps
    ]


def build_test_only_d2d_design():
    """Build the exact ten-dimensional D2D grid without campaign semantics."""
    return build_design(
        [
            InputSpec("speed_1", 1000, 6000, 500, unit="rpm"),
            InputSpec("time_1", 5, 50, 5, unit="s"),
            InputSpec("speed_2", 0, 5000, 500, unit="rpm"),
            InputSpec("time_2", 10, 60, 5, unit="s"),
            InputSpec("precur_conc", 1.0, 2.0, 0.05, unit="M"),
            InputSpec("precur_vol", 40, 200, 10, unit="uL"),
            InputSpec("anneal_temp", 100, 185, 5, unit="C"),
            InputSpec("anneal_time", 10, 60, 5, unit="min"),
            InputSpec("anti_vol", 100, 200, 5, unit="uL"),
            InputSpec("anti_time", 9, 25, 2, unit="s"),
        ]
    )


def _synthetic_raw_outcomes(X_norm: np.ndarray) -> np.ndarray:
    """Create two bounded maximize outcomes and one raw target outcome."""
    first = 0.15 + 0.35 * X_norm[:, 0] + 0.25 * X_norm[:, 4] + 0.15 * X_norm[:, 7]
    second = (
        0.10 + 0.30 * X_norm[:, 1] + 0.20 * (1.0 - X_norm[:, 2]) + 0.25 * X_norm[:, 8]
    )
    raw_target = 430.0 + 420.0 * (0.55 * X_norm[:, 3] + 0.45 * X_norm[:, 6])
    return np.column_stack([first, second, raw_target])


def _fit_synthetic_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> ModelListGP:
    models = []
    for objective_index in range(train_Y.shape[1]):
        output = train_Y[:, objective_index : objective_index + 1]
        output_variance = output.var(correction=0).clamp_min(1e-8)
        # Standardize(m=1) scales this to a stable TEST_ONLY variance of 1e-4.
        known_noise = torch.full_like(output, float(output_variance * 1e-4))
        model = SingleTaskGP(
            train_X,
            output,
            train_Yvar=known_noise,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": 25, "ftol": 1e-7}},
        )
        model.eval()
        models.append(model)
    return ModelListGP(*models)


def _objective_contract() -> ObjectiveTransform:
    return ObjectiveTransform(
        [
            ObjectiveSpec("synthetic_maximize_1", "maximize", "identity"),
            ObjectiveSpec("synthetic_maximize_2", "maximize", "identity"),
            ObjectiveSpec(
                "synthetic_target",
                "target",
                "gaussian_target",
                target=650.0,
                sigma=120.0,
            ),
        ],
        version="TEST_ONLY-synthetic-objectives-v1",
    )


def _plots_for_method(
    method: str,
    output_dir: Path,
    design,
    observed_norm: np.ndarray,
    pool: CandidatePool,
    selection,
) -> tuple[Path, ...]:
    steps = selection.steps
    return (
        plot_candidate_pca(
            observed_norm,
            selection.X_norm,
            output_dir / f"{method}_pca.png",
            pool_norm=pool.X_norm,
            seed=TEST_ONLY_SEED,
        ),
        plot_parallel_coordinates(
            selection.X_norm,
            design.names,
            output_dir / f"{method}_parallel_coordinates.png",
        ),
        plot_distance_heatmap(
            selection.X_norm, output_dir / f"{method}_distance_heatmap.png"
        ),
        plot_selection_scores(
            [step.order for step in steps],
            [step.base_log_score for step in steps],
            [step.penalized_log_score for step in steps],
            output_dir / f"{method}_selection_scores.png",
        ),
    )


def run_synthetic_step2a(
    output_dir: str | Path,
    *,
    ucb_pool_size: int = 128,
    qlognehvi_pool_size: int = 96,
    posterior_samples: int = 32,
    qlognehvi_samples: int = 16,
) -> SyntheticRunSummary:
    """Fit synthetic GPs and deterministically propose TEST_ONLY 5/3 batches."""
    started = perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(TEST_ONLY_SEED)
    np.random.seed(TEST_ONLY_SEED)

    design = build_test_only_d2d_design()
    observed_pool = sample_discrete_candidate_pool(design, 15, seed=TEST_ONLY_SEED)
    train_X = torch.as_tensor(observed_pool.X_norm, dtype=torch.double)
    train_Y = torch.as_tensor(
        _synthetic_raw_outcomes(observed_pool.X_norm), dtype=torch.double
    )
    model = _fit_synthetic_model(train_X, train_Y)
    transform = _objective_contract()
    mc_objective = ConfiguredMCMultiOutputObjective(transform)

    ucb_pool = sample_discrete_candidate_pool(
        design,
        ucb_pool_size,
        seed=TEST_ONLY_SEED + 1,
        observed_phys=observed_pool.X_phys,
    )
    local_config = LocalPenalizationConfig(
        radius=0.65,
        min_batch_distance=0.20,
        min_observed_distance=0.08,
    )
    ucb_kwargs = dict(
        q=5,
        beta=1.0,
        local_penalization_config=local_config,
        observed_pending_norm=observed_pool.X_norm,
        positive_score_tolerance=1e-12,
        mc_samples=posterior_samples,
        seed=TEST_ONLY_SEED + 2,
        posterior_chunk_size=32,
        hvi_chunk_size=64,
    )
    ucb = propose_ucb_hvi_batch(
        ucb_pool,
        model,
        train_Y,
        transform,
        TEST_ONLY_REFERENCE_POINT_UTILITY,
        **ucb_kwargs,
    )
    ucb_repeat = propose_ucb_hvi_batch(
        ucb_pool,
        model,
        train_Y,
        transform,
        TEST_ONLY_REFERENCE_POINT_UTILITY,
        **ucb_kwargs,
    )
    if not np.array_equal(
        ucb.selection.selected_pool_indices,
        ucb_repeat.selection.selected_pool_indices,
    ):
        raise RuntimeError("TEST_ONLY UCB-HVI rerun was not deterministic.")

    qlog_pool = sample_discrete_candidate_pool(
        design,
        qlognehvi_pool_size,
        seed=TEST_ONLY_SEED + 3,
        observed_phys=observed_pool.X_phys,
        pending_phys=ucb.selection.X_phys,
    )
    ucb_pending = torch.as_tensor(ucb.selection.X_norm, dtype=torch.double)
    qlog_kwargs = dict(
        q=3,
        local_penalization_config=local_config,
        X_pending_norm=ucb_pending,
        mc_samples=qlognehvi_samples,
        seed=TEST_ONLY_SEED + 4,
        chunk_size=32,
    )
    qlog = propose_qlognehvi_penalized_batch(
        qlog_pool,
        model,
        train_X,
        mc_objective,
        TEST_ONLY_REFERENCE_POINT_UTILITY,
        **qlog_kwargs,
    )
    qlog_repeat = propose_qlognehvi_penalized_batch(
        qlog_pool,
        model,
        train_X,
        mc_objective,
        TEST_ONLY_REFERENCE_POINT_UTILITY,
        **qlog_kwargs,
    )
    if not np.array_equal(
        qlog.selection.selected_pool_indices,
        qlog_repeat.selection.selected_pool_indices,
    ):
        raise RuntimeError("TEST_ONLY qLogNEHVI rerun was not deterministic.")

    ucb_diagnostics = summarize_candidate_batch(
        ucb.selection.X_norm,
        observed_pending_norm=observed_pool.X_norm,
        X_phys=ucb.selection.X_phys,
        design=design,
        metadata=ucb.metadata,
    )
    qlog_diagnostics = summarize_candidate_batch(
        qlog.selection.X_norm,
        observed_pending_norm=np.vstack([observed_pool.X_norm, ucb.selection.X_norm]),
        X_phys=qlog.selection.X_phys,
        design=design,
        metadata=qlog.metadata,
    )
    if not np.all(ucb_diagnostics.grid_valid_rows) or not np.all(
        qlog_diagnostics.grid_valid_rows
    ):
        raise RuntimeError("Synthetic proposal contained an off-grid row.")
    if ucb_diagnostics.duplicate_row_pairs or qlog_diagnostics.duplicate_row_pairs:
        raise RuntimeError("Synthetic proposal contained a duplicate row.")

    plot_paths = _plots_for_method(
        "ucb_hvi",
        output,
        design,
        observed_pool.X_norm,
        ucb_pool,
        ucb.selection,
    ) + _plots_for_method(
        "qlognehvi",
        output,
        design,
        observed_pool.X_norm,
        qlog_pool,
        qlog.selection,
    )
    elapsed = perf_counter() - started
    summary = SyntheticRunSummary(
        elapsed_seconds=elapsed,
        observed_count=observed_pool.size,
        ucb_pool_size=ucb_pool.size,
        qlognehvi_pool_size=qlog_pool.size,
        ucb_selected_pool_indices=ucb.selection.selected_pool_indices,
        qlognehvi_selected_pool_indices=qlog.selection.selected_pool_indices,
        ucb_minimum_distance=float(ucb_diagnostics.minimum_within_batch_distance),
        qlognehvi_minimum_distance=float(
            qlog_diagnostics.minimum_within_batch_distance
        ),
        ucb_metadata=dict(ucb_diagnostics.metadata),
        qlognehvi_metadata=dict(qlog_diagnostics.metadata),
        plot_paths=plot_paths,
    )
    ucb_selected = ucb.selection.selected_pool_indices
    ucb_utility_diagnostics = [
        {
            "pool_index": int(pool_index),
            "utility_mean": ucb.scoring.utility_mean[pool_index],
            "utility_std": ucb.scoring.utility_std[pool_index],
            "utility_ucb": ucb.scoring.utility_ucb[pool_index],
            "raw_hvi": ucb.scoring.base_score[pool_index],
        }
        for pool_index in ucb_selected
    ]
    report = {
        "status": "TEST_ONLY_SYNTHETIC_PASS",
        "production_candidate_generation": "NOT_RUN",
        "elapsed_seconds": round(elapsed, 3),
        "observed_count": summary.observed_count,
        "seeds": {
            "global": TEST_ONLY_SEED,
            "observed_pool": TEST_ONLY_SEED,
            "ucb_pool": TEST_ONLY_SEED + 1,
            "ucb_posterior": TEST_ONLY_SEED + 2,
            "qlognehvi_pool": TEST_ONLY_SEED + 3,
            "qlognehvi_mc": TEST_ONLY_SEED + 4,
        },
        "ucb_hvi": {
            "pool_size": summary.ucb_pool_size,
            "batch_size": len(summary.ucb_selected_pool_indices),
            "selected_pool_indices": summary.ucb_selected_pool_indices.tolist(),
            "minimum_normalized_distance": summary.ucb_minimum_distance,
            "metadata": summary.ucb_metadata,
            "selection_steps": _selection_table(ucb.selection),
            "selected_utility_diagnostics": ucb_utility_diagnostics,
            "nearest_observed_pending_distance": (
                ucb_diagnostics.nearest_observed_pending_distance
            ),
            "boundary_flags": ucb_diagnostics.boundary_flags,
            "grid_valid_rows": ucb_diagnostics.grid_valid_rows,
        },
        "qlognehvi": {
            "pool_size": summary.qlognehvi_pool_size,
            "batch_size": len(summary.qlognehvi_selected_pool_indices),
            "selected_pool_indices": summary.qlognehvi_selected_pool_indices.tolist(),
            "minimum_normalized_distance": summary.qlognehvi_minimum_distance,
            "metadata": summary.qlognehvi_metadata,
            "selection_steps": _selection_table(qlog.selection),
            "pending_counts_by_selection_step": [
                item.pending_count for item in qlog.score_history
            ],
            "nearest_observed_pending_distance": (
                qlog_diagnostics.nearest_observed_pending_distance
            ),
            "boundary_flags": qlog_diagnostics.boundary_flags,
            "grid_valid_rows": qlog_diagnostics.grid_valid_rows,
        },
        "objective_contract_version": transform.version,
        "reference_point": {
            "space": "TEST_ONLY transformed utility",
            "value": TEST_ONLY_REFERENCE_POINT_UTILITY.tolist(),
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "botorch": package_version("botorch"),
            "gpytorch": package_version("gpytorch"),
            "device": "cpu",
        },
        "plots": [path.name for path in plot_paths],
    }
    (output / "synthetic_summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    output = REPOSITORY_ROOT / "local_outputs" / "step2a_synthetic"
    summary = run_synthetic_step2a(output)
    print(
        "TEST_ONLY synthetic Step 2A PASS: "
        f"R1={len(summary.ucb_selected_pool_indices)}, "
        f"R2={len(summary.qlognehvi_selected_pool_indices)}, "
        f"elapsed={summary.elapsed_seconds:.2f}s, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
