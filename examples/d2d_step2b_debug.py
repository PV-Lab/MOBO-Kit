"""Run the read-only D2D Step 2B algorithm-debug adapter from the repo root."""

from __future__ import annotations

import argparse
from pathlib import Path

from mobo_kit.d2d_step2b_debug import run_d2d_step2b_debug


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate a watermarked D2D R1 debug bundle. This never approves or "
            "writes an experimental worklist to Excel."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="Explicit path to the ignored private campaign workbook.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit path to the matching ignored private debug configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root / "local_outputs" / "d2d_step2b_debug" / "baseline_seed73"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Run only the configured baseline (useful for a quick local smoke test).",
    )
    args = parser.parse_args()
    result = run_d2d_step2b_debug(
        args.workbook,
        args.config,
        args.output,
        overwrite=args.overwrite,
        run_sensitivity=not args.skip_sensitivity,
    )
    print("DEBUG ONLY - NOT APPROVED FOR EXPERIMENT")
    print(f"Output directory: {result.output_dir}")
    print(f"Unique R1 conditions: {len(result.candidates_unique)}")
    print(f"Replicate execution rows: {len(result.replicate_worklist)}")
    print(f"Sensitivity rows: {len(result.sensitivity_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
