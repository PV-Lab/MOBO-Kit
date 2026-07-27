"""Run the read-only D2D Step 2C robustness study from the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path

from mobo_kit.d2d_step2c_robustness import run_d2d_step2c_robustness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a watermarked, read-only D2D R1 robustness audit. "
            "This command never writes the workbook or approves fabrication."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="Path to the Git-ignored campaign workbook.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the matching Git-ignored private Step 2C configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Run directory below local_outputs/d2d_step2c_robustness. "
            "Defaults to <mode>_seed73."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "full"),
        default="full",
        help="Use fast for a smoke audit or full for the declared Step 2C study.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-portable-zip",
        action="store_true",
        help="Skip the ignored portable ZIP while retaining the validated directory.",
    )
    return parser


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args()
    output = args.output or (
        repository_root
        / "local_outputs"
        / "d2d_step2c_robustness"
        / f"{args.mode}_seed73"
    )
    result = run_d2d_step2c_robustness(
        args.workbook,
        args.config,
        output,
        mode=args.mode,
        overwrite=args.overwrite,
        create_portable_zip=not args.no_portable_zip,
    )
    print("DEBUG ONLY - NOT APPROVED FOR EXPERIMENT")
    print(f"Mode: {result.mode}")
    print(f"Output directory: {result.output_dir}")
    print(f"Robust regions: {len(result.robust_regions)}")
    print(f"Debug shortlist rows: {len(result.shortlist)}")
    print(f"Consensus criteria passed: {result.consensus.passed}")
    print(
        "Sample 1 influence rank: "
        f"{result.run_manifest['sample_1_influence_rank']} "
        f"({result.run_manifest['sample_1_influence_percentile']:.1f} percentile)"
    )
    print(f"Validated artifacts: {len(result.artifact_hashes)}")
    print("Workbook writeback performed: false")
    print("Real R2 proposal generated: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
