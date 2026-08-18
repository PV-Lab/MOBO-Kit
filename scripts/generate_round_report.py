"""Render a round's figures from a terminal, exactly as the launcher does.

    python scripts/generate_round_report.py --workbook "local_inputs/Summary Table Test.xlsx"
    python scripts/generate_round_report.py --workbook <path> --data-only

Two modes, matching the two buttons:

* **default** re-reads the round that was last proposed and renders the full set,
  including the two batch figures. It re-derives the proposal from the config and
  the measured rows rather than reading it back from the worklist, so what the
  figures describe is the model's answer at this seed -- if that has drifted from
  the sheet on disk, these figures say so and that is worth knowing.
* **--data-only** renders everything that depends on measurements alone. This is
  the mode to use the moment a round's results are entered.

Nothing here writes to the source workbook, and nothing here approves anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mobo_kit.campaign import load_campaign_config, run_r1_ucb, run_r2_qlognehvi
from mobo_kit.batch_review import build_batch_review
from mobo_kit.launcher import gather_observations, inspect_campaign
from mobo_kit.round_report import generate_round_report
from mobo_kit.workbook_io import read_campaign_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--config", default="configs/campaign_d2d_perovskite_test.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="figures from the measurements alone; no batch is proposed",
    )
    parser.add_argument(
        "--shap-instances",
        type=int,
        default=15,
        help="rows to attribute; the runtime knob, recorded in the manifest",
    )
    args = parser.parse_args(argv)

    config = load_campaign_config(args.config)
    workbook = Path(args.workbook)

    proposal = None
    review = None
    if not args.data_only:
        status = inspect_campaign(workbook, config)
        if not status.can_generate:
            print(f"No round is due: {status.reason}")
            print("Rendering the data-only report instead.")
        else:
            round_name = str(status.next_round)
            print(f"Proposing {round_name} to describe it...")
            X, Y, Yvar, _ = gather_observations(
                workbook, config, for_round=round_name
            )
            runner = run_r1_ucb if round_name == "R1" else run_r2_qlognehvi
            proposal = runner(config, X, Y, seed=args.seed, observed_Yvar=Yvar)
            contents = read_campaign_workbook(workbook, config)
            review = build_batch_review(
                config,
                X,
                Y,
                proposal.conditions,
                round_name=round_name,
                seed=proposal.diagnostics.get("seed"),
                findings=contents.findings,
            )

    manifest = generate_round_report(
        workbook,
        config,
        proposal=proposal,
        review=review,
        outdir=args.outdir,
        seed=args.seed,
        shap_max_instances=args.shap_instances,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    print()
    print(manifest.summary())
    print()
    print(f"{manifest.runtime_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
