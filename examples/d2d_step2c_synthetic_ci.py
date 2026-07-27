"""Run the sanitized synthetic Step 2C fast end-to-end fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from mobo_kit.d2d_step2c_synthetic import run_synthetic_step2c_fast_ci


def build_parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 2C fast orchestration on generated sanitized data. "
            "This is CI/debug coverage only and cannot approve an experiment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root / "configs" / "d2d_step2c_debug.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root / "local_outputs" / "d2d_step2c_robustness" / "synthetic_ci"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-portable-zip", action="store_true")
    return parser


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    args = build_parser(repository_root).parse_args()
    result = run_synthetic_step2c_fast_ci(
        args.config,
        args.output,
        overwrite=args.overwrite,
        create_portable_zip=not args.no_portable_zip,
    )
    print("DEBUG ONLY - NOT APPROVED FOR EXPERIMENT")
    print("Input data: sanitized synthetic CI fixture")
    print(f"Output directory: {result.output_dir}")
    print(f"Validated artifacts: {len(result.artifact_hashes)}")
    print(f"Consensus criteria passed: {result.consensus.passed}")
    print("Private campaign workbook read: false")
    print("Experimental approval enabled: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
