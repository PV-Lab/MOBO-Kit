"""Synthetic-only future R2 boundary for the resolved D2D objective contract."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .batch_selection import LocalPenalizationConfig
from .candidate_pool import CandidatePool, physical_rows_to_grid_indices
from .data import x_normalizer_np
from .d2d_campaign import ResolvedD2DDebugConfig, build_d2d_objective_transform
from .objectives import ConfiguredMCMultiOutputObjective
from .qlognehvi_batch import QLogNEHVIBatchProposal, propose_qlognehvi_penalized_batch


D2D_R2_TEST_WATERMARK = "SYNTHETIC R2 TEST ONLY - NOT APPROVED FOR EXPERIMENT"


def _validate_resolved_candidate_pool(
    candidate_pool: CandidatePool,
    config: ResolvedD2DDebugConfig,
) -> None:
    """Require exact agreement among physical, grid-index, and normalized rows."""
    if not isinstance(candidate_pool, CandidatePool):
        raise TypeError("candidate_pool must be a CandidatePool.")
    expected_indices = physical_rows_to_grid_indices(
        candidate_pool.X_phys, config.design
    )
    actual_indices = np.asarray(candidate_pool.grid_indices)
    if actual_indices.shape != expected_indices.shape or not np.array_equal(
        actual_indices, expected_indices
    ):
        raise ValueError(
            "Candidate pool grid_indices do not match its physical rows on the "
            "approved D2D grid."
        )
    expected_norm = x_normalizer_np(candidate_pool.X_phys, config.design)
    actual_norm = np.asarray(candidate_pool.X_norm, dtype=float)
    if actual_norm.shape != expected_norm.shape or not np.allclose(
        actual_norm, expected_norm, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Candidate pool normalized rows do not match its physical/grid rows."
        )


def propose_d2d_qlognehvi_test_batch(
    candidate_pool: CandidatePool,
    model: Any,
    train_X_norm: torch.Tensor,
    config: ResolvedD2DDebugConfig,
    *,
    test_only: bool,
    mc_samples: int | None = None,
    seed: int | None = None,
    chunk_size: int = 512,
) -> QLogNEHVIBatchProposal:
    """Exercise the future three-condition R2 path on sanitized data only.

    Real R2 generation is intentionally impossible through this wrapper unless
    the caller supplies the explicit test-only acknowledgement. No workbook or
    measured R1 candidate artifact is accepted by this API.
    """
    if test_only is not True:
        raise ValueError("The Step 2B qLogNEHVI boundary is synthetic test-only.")
    _validate_resolved_candidate_pool(candidate_pool, config)
    r2 = config.raw.get("r2_test_only", {})
    if r2.get("method") != "qlognehvi" or r2.get("sequential_pending") is not True:
        raise ValueError(
            "The resolved config must retain sequential test-only qLogNEHVI."
        )
    if r2.get("batch_size_unique_conditions") != 3:
        raise ValueError(
            "The synthetic R2 test batch must contain exactly three conditions."
        )
    sample_value = r2.get("mc_samples") if mc_samples is None else mc_samples
    if isinstance(sample_value, (bool, np.bool_)) or not isinstance(
        sample_value, (int, np.integer)
    ):
        raise ValueError("mc_samples must be a positive non-boolean integer.")
    samples = int(sample_value)
    if samples <= 0:
        raise ValueError("mc_samples must be a positive non-boolean integer.")
    seed_value = config.seed if seed is None else seed
    if isinstance(seed_value, (bool, np.bool_)) or not isinstance(
        seed_value, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative non-boolean integer.")
    selected_seed = int(seed_value)
    if selected_seed < 0:
        raise ValueError("seed must be a non-negative non-boolean integer.")
    local = LocalPenalizationConfig(
        radius=config.local_radius,
        min_batch_distance=config.min_batch_distance,
        min_observed_distance=config.min_observed_distance,
        dimension_weights=config.dimension_weights,
    )
    objective = ConfiguredMCMultiOutputObjective(build_d2d_objective_transform())
    result = propose_qlognehvi_penalized_batch(
        candidate_pool,
        model,
        train_X_norm,
        objective,
        config.reference_point_utility,
        q=3,
        local_penalization_config=local,
        mc_samples=samples,
        seed=selected_seed,
        chunk_size=chunk_size,
        prune_baseline=False,
    )
    if result.selection.X_phys.shape != (3, len(config.design.names)):
        raise RuntimeError(
            "The synthetic R2 path did not return exactly three conditions."
        )
    if np.unique(result.selection.X_phys, axis=0).shape[0] != 3:
        raise RuntimeError("The synthetic R2 path returned duplicate conditions.")
    selected_indices = physical_rows_to_grid_indices(
        result.selection.X_phys, config.design
    )
    selected_pool_indices = result.selection.selected_pool_indices
    if not np.array_equal(
        selected_indices, candidate_pool.grid_indices[selected_pool_indices]
    ):
        raise RuntimeError(
            "The synthetic R2 selection is inconsistent with the approved grid."
        )
    expected_selected_norm = x_normalizer_np(result.selection.X_phys, config.design)
    if not np.allclose(
        result.selection.X_norm, expected_selected_norm, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "The synthetic R2 selection has inconsistent normalized coordinates."
        )
    selection = replace(
        result.selection,
        method_diagnostics={
            **result.selection.method_diagnostics,
            "test_only": True,
            "synthetic_only": True,
            "approved_for_experiment": False,
            "candidate_status": D2D_R2_TEST_WATERMARK,
        },
    )
    return replace(
        result,
        selection=selection,
        metadata={
            **result.metadata,
            "test_only": True,
            "synthetic_only": True,
            "debug_only": True,
            "approved_for_experiment": False,
            "candidate_status": D2D_R2_TEST_WATERMARK,
        },
    )


__all__ = ["D2D_R2_TEST_WATERMARK", "propose_d2d_qlognehvi_test_batch"]
