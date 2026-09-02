"""The one-button loop for the experimentalist.

Double-click ``launch_mobo_kit.bat`` (Windows) or ``launch_mobo_kit.command``
(macOS), point it at the campaign workbook, press the button.  It works out which
round is due, reads what has been measured, proposes the next batch and writes it
to a sheet beside the workbook.

Everything above the UI lives in plain functions -- :func:`inspect_campaign`,
:func:`gather_observations`, :func:`generate_next_round` -- so the decisions can
be tested without a display, and so the same steps are available from a script
when someone would rather not click.

Three rules the UI keeps, all of them inherited rather than invented:

* **The source workbook is never opened for writing.**  Candidates go to a
  sibling file, because openpyxl discards cached formula values on save.
* **Fail closed.**  A half-filled sheet, a film with no usable measurement, a
  workbook missing a column: each stops the round with a plain sentence rather
  than being guessed at.
* **Nothing here approves a batch.**  Fifteen films is a real cost; the window
  shows what was proposed and why, and a human decides.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .batch_review import BatchReview, build_batch_review, write_review_sheet
from .campaign import (
    RoundResult,
    load_campaign_config,
    objective_names,
    replicate_aggregates,
    run_r1_ucb,
    run_r2_qlognehvi,
)
from .replicate_variance import yvar_for_campaign
from .scores import ScoreFinding, ScoreSeverity, describe_findings
from .workbook_io import (
    CandidateSheetError,
    candidate_workbook_path,
    detect_round,
    read_campaign_workbook,
    read_candidate_results,
    source_sheet,
    write_candidate_sheet,
)

__all__ = [
    "CampaignStatus",
    "DEFAULT_CONFIG",
    "Generated",
    "LauncherError",
    "gather_observations",
    "generate_next_round",
    "inspect_campaign",
    "main",
]

#: The ACTIVE campaign. This is the one path an experimentalist reaches by
#: double-clicking, so it must never point at an archived contract: the launcher
#: would then ask the new workbook for the previous campaign's columns and report
#: it as a missing column, which reads as a broken workbook rather than as the
#: config mismatch it is. That happened once, on 2026-08-18, between archiving
#: campaign_d2d_perovskite.yaml and updating this line.
DEFAULT_CONFIG = "configs/campaign_d2d_perovskite_final.yaml"

#: Remembered between runs so the experimentalist browses to the workbook once.
#: Kept in the user's home rather than the repo, so moving the checkout does not
#: lose it.  Every read and write here is best-effort: a launcher that cannot
#: start because of its own preferences file would be worse than one that forgets.
SETTINGS_PATH = Path.home() / ".mobo_kit" / "launcher.json"


class LauncherError(RuntimeError):
    """Something the user needs to fix, phrased for the user."""


# --------------------------------------------------------------------------- #
# remembering the workbook
# --------------------------------------------------------------------------- #


def load_settings() -> dict[str, Any]:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as handle:
            settings = json.load(handle)
        return settings if isinstance(settings, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: Mapping[str, Any]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(dict(settings), handle, indent=2)
    except OSError:
        pass


def remembered_workbook() -> Path | None:
    raw = load_settings().get("workbook")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def remember_workbook(path: str | Path) -> None:
    settings = load_settings()
    settings["workbook"] = str(Path(path).resolve())
    save_settings(settings)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CampaignStatus:
    """What the campaign looks like right now, in the user's terms."""

    workbook: Path
    next_round: str | None
    reason: str
    scored_rows: int
    total_rows: int
    observed_conditions: int
    findings: tuple[ScoreFinding, ...] = ()
    #: Which sheet the rows came from; "Sheet1" on the older contracts, "R0"
    #: on v4. Shown rather than assumed, because a user looking at the wrong
    #: sheet is exactly the confusion this line exists to prevent.
    source_sheet: str = "Sheet1"

    @property
    def can_generate(self) -> bool:
        return self.next_round is not None

    @property
    def headline(self) -> str:
        if self.next_round:
            return f"Ready to propose {self.next_round}."
        return "Nothing to propose yet."

    @property
    def errors(self) -> tuple[ScoreFinding, ...]:
        return tuple(f for f in self.findings if f.severity is ScoreSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ScoreFinding, ...]:
        return tuple(f for f in self.findings if f.severity is ScoreSeverity.WARNING)

    def detail(self) -> str:
        """The body text of the window: what is known, then what was noticed."""
        lines = [
            f"Workbook:   {self.workbook}",
            f"Measured:   {self.observed_conditions} conditions on {self.source_sheet}",
            f"Status:     {self.reason}",
        ]
        if self.total_rows:
            lines.append(
                f"Candidates: {self.scored_rows} of {self.total_rows} rows measured"
            )
        if self.errors:
            lines += ["", "These must be fixed before a round can run:"]
            lines += [f"  {finding}" for finding in self.errors]
        if self.warnings:
            lines += ["", "Worth a look, but not blocking:"]
            lines += [f"  {finding}" for finding in self.warnings]
        notes = [f for f in self.findings if f.severity is ScoreSeverity.NOTE]
        if notes:
            lines += ["", "For the record:"]
            lines += [f"  {finding}" for finding in notes]
        return "\n".join(lines)


def inspect_campaign(
    workbook: str | Path, config: Mapping[str, Any]
) -> CampaignStatus:
    """Read the workbook and decide what is due, without proposing anything."""
    path = Path(workbook)
    if not path.exists():
        raise LauncherError(f"{path} does not exist.")
    contents = read_campaign_workbook(path, config)
    state = detect_round(path, config)
    return CampaignStatus(
        workbook=path.resolve(),
        next_round=state.next_round,
        reason=state.reason,
        scored_rows=state.scored_rows,
        total_rows=state.total_rows,
        observed_conditions=contents.n_rows,
        findings=contents.findings,
        source_sheet=source_sheet(config),
    )


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #


def gather_observations(
    workbook: str | Path, config: Mapping[str, Any], *, for_round: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    """Every measured design point the next round should learn from.

    R1 trains on Sheet1 alone.  R2 trains on Sheet1 plus the aggregated R1
    conditions -- three films become one observation, which is why
    :func:`read_candidate_results` exists.

    Returns ``(X, Y, Yvar, provenance)``.  ``Yvar`` is ``None`` unless the config
    asks for measured replicate variance *and* replicated conditions exist to pool
    from; see :mod:`replicate_variance`.

    Raises rather than dropping rows: a NaN objective reaching the GP is how a
    round gets proposed from data nobody checked.
    """
    path = Path(workbook)
    names = list(objective_names(config))
    input_names = [item["name"] for item in config["inputs"]]

    contents = read_campaign_workbook(path, config)
    if contents.errors:
        raise LauncherError(
            "Sheet1 has rows that cannot be turned into objective values:\n"
            + describe_findings(contents.errors)
        )
    X = [contents.inputs.to_numpy(dtype=float)]
    Y = [contents.model_values.to_numpy(dtype=float)]
    provenance = [f"{source_sheet(config)}: {contents.n_rows} conditions"]
    Yvar: np.ndarray | None = None

    if for_round.upper() == "R2":
        results = read_candidate_results(path, config, "R1")
        if results.errors:
            raise LauncherError(
                "The R1 sheet has conditions that cannot be turned into objective "
                "values:\n" + describe_findings(results.errors)
            )
        X.append(results.conditions[input_names].to_numpy(dtype=float))
        Y.append(results.model_values[names].to_numpy(dtype=float))
        provenance.append(
            f"R1 sheet: {results.n_conditions} conditions from "
            f"{len(results.replicates)} films"
        )
        Yvar, floor_findings = yvar_for_campaign(
            config,
            results,
            n_rows_without_replicates=contents.n_rows,
            objective_names=names,
            aggregates=replicate_aggregates(config),
        )
        if Yvar is not None:
            provenance.append(
                "observation noise: pooled between-film variance from the R1 "
                "triplicates, not fitted"
            )
        for message in floor_findings:
            provenance.append(f"WARNING {message}")

    X_all = np.vstack(X)
    Y_all = np.vstack(Y)
    if not np.all(np.isfinite(Y_all)):
        bad = int((~np.isfinite(Y_all)).any(axis=1).sum())
        raise LauncherError(
            f"{bad} observation(s) still hold a non-finite objective value after "
            "aggregation. Fix the measurements before proposing a round."
        )
    return X_all, Y_all, Yvar, provenance


# --------------------------------------------------------------------------- #
# generating
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Generated:
    """What a successful press of the button produced."""

    round_name: str
    sheet_path: Path
    result: RoundResult
    provenance: list[str] = field(default_factory=list)
    review: BatchReview | None = None
    report: Any = None
    """The round report's manifest, or ``None`` if it was skipped or failed."""
    report_error: str | None = None
    """Why the report is missing. A batch is never rolled back over a figure."""

    @property
    def n_films(self) -> int:
        return len(self.result.replicates)

    def summary(self) -> str:
        """The body text after a successful run: what, from what, and how spread."""
        diagnostics = self.result.diagnostics
        validity = diagnostics.get("validity", {})
        distance = validity.get("min_pairwise_distance")
        lines = [
            f"Wrote {self.result.n_conditions} {self.round_name} conditions "
            f"({self.n_films} films) to:",
            f"  {self.sheet_path}",
            "",
            "Trained on:",
            *(f"  {item}" for item in self.provenance),
            "",
            f"Method:                {diagnostics.get('method')}",
            f"Seed:                  {diagnostics.get('seed')}",
            f"Candidate pool:        {diagnostics.get('pool_size')}",
            f"Objective contract:    {diagnostics.get('objective_contract')}",
            "Min pairwise distance: "
            + (f"{distance:.4f}" if isinstance(distance, float) else str(distance)),
            f"Boundary coords/row:   {validity.get('boundary_coords_per_condition')}",
            "",
            "Proposed conditions, physical units:",
            self.result.conditions.to_string(
                index=False, float_format=lambda value: f"{value:g}"
            ),
        ]
        if self.review is not None:
            # the review is the artifact; the lines above are its provenance
            lines += ["", self.review.to_text()]
        else:
            lines += [
                "",
                "Nothing here is approved. Read the conditions, then run each in "
                "triplicate and fill in the highlighted columns.",
            ]
        if self.report is not None:
            lines += ["", self.report.summary()]
        elif self.report_error is not None:
            lines += [
                "",
                "FIGURES NOT PRODUCED",
                "-" * 78,
                self.report_error,
            ]
        return "\n".join(lines)


def generate_next_round(
    workbook: str | Path,
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    progress: Callable[[str], None] | None = None,
    with_report: bool = True,
) -> Generated:
    """Propose and write whichever round is due.  Refuses if none is.

    ``with_report`` renders the round's figures beside the workbook afterwards.
    **A failure there never costs the batch.** The worklist and the Review sheet
    are already written and correct at that point; discarding them because a
    figure could not be drawn would throw away the expensive, careful part of the
    run over the cheap, decorative one. The failure is reported loudly instead.
    """

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    path = Path(workbook)
    say("Reading the workbook...")
    status = inspect_campaign(path, config)
    if not status.can_generate:
        raise LauncherError(status.reason)

    round_name = str(status.next_round)
    destination = candidate_workbook_path(path, round_name)
    if destination.exists():
        raise LauncherError(
            f"{destination.name} already exists. Rename or delete it first; this "
            "tool never overwrites a file that may hold measurements."
        )

    say(f"Collecting observations for {round_name}...")
    X, Y, Yvar, provenance = gather_observations(path, config, for_round=round_name)

    say(f"Fitting the model and scoring candidates for {round_name}. About 10 seconds.")
    runner = run_r1_ucb if round_name == "R1" else run_r2_qlognehvi
    result = runner(config, X, Y, seed=seed, observed_Yvar=Yvar)

    say(f"Writing {destination.name}...")
    replicates = int(
        (config.get("rounds", {}).get(round_name.lower(), {}) or {}).get(
            "replicates_per_condition", 3
        )
    )
    sheet_path = write_candidate_sheet(
        path,
        config,
        result.conditions,
        round_name=round_name,
        replicates=replicates,
    )
    say("Building the review...")
    contents = read_campaign_workbook(path, config)
    # The report directory is decided BEFORE the review is written, so the Review
    # sheet can point at the figures. The report is rendered into exactly this
    # directory afterwards; if it fails, the sheet points at a directory holding
    # the error trace, which is more useful than pointing at nothing.
    from .round_report import report_directory

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figures_dir = report_directory(path, round_name, when=stamp)
    review = build_batch_review(
        config,
        X,
        Y,
        result.conditions,
        round_name=round_name,
        seed=result.diagnostics.get("seed"),
        findings=contents.findings,
        context={
            "Round": round_name,
            "Worklist": sheet_path.name,
            "Figures": str(figures_dir),
            "Trained on": "; ".join(provenance),
            "Observations": len(X),
            "Method": result.diagnostics.get("method"),
            "Seed": result.diagnostics.get("seed"),
            "Candidate pool": result.diagnostics.get("pool_size"),
            "Objective contract": result.diagnostics.get("objective_contract"),
            "Films to run": len(result.replicates),
        },
    )
    write_review_sheet(sheet_path, review)

    report = None
    report_error = None
    if with_report:
        from .round_report import generate_round_report

        try:
            report = generate_round_report(
                path,
                config,
                proposal=result,
                review=review,
                outdir=figures_dir,
                when=stamp,
                seed=result.diagnostics.get("seed"),
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001 - a figure must never cost a batch
            report_error = (
                f"{type(exc).__name__}: {exc}. The worklist and the Review sheet "
                "were written and are unaffected."
            )

    say("Done.")
    return Generated(
        round_name=round_name,
        sheet_path=sheet_path,
        result=result,
        provenance=provenance,
        review=review,
        report=report,
        report_error=report_error,
    )


def generate_data_report(
    workbook: str | Path,
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Figures from the measurements alone, with no batch proposed.

    What the "Figures from current data" button runs. Useful the moment a round's
    measurements are entered and before anyone decides whether to propose: the
    parity, attribution, hypervolume and objective-space figures are all about
    what has been measured, and none of them needs a candidate batch.
    """
    from .round_report import generate_round_report

    return generate_round_report(workbook, config, seed=seed, progress=progress)


def reveal(path: str | Path) -> None:
    """Show a file in the platform's file manager.  Never raises."""
    target = Path(path)
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", str(target)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(target)], check=False)
        else:
            subprocess.run(["xdg-open", str(target.parent)], check=False)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #


class LauncherWindow:
    """A small tkinter window over the functions above.

    tkinter is imported here rather than at module scope so that the logic can be
    imported and tested on a machine with no display.

    **Results are matched to the request that asked for them.**  Work runs off the
    main thread and reports back through a queue, so without that matching two
    things can paint the pane with an answer to a question the user has moved on
    from: the auto-check scheduled 200 ms after startup, and any second press while
    the first is still running.  Each dispatch takes a request id; a reply carrying
    a stale id is dropped.  A status reply also names the workbook it examined and
    is dropped if the selection has changed since -- reporting "Ready to propose R1"
    over a workbook the user has navigated away from is worse than reporting
    nothing.  A dropped reply still clears the busy state, or the window would
    disable its own buttons forever.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        import queue
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self._queue: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self._config_path = Path(config_path)
        self._config: dict[str, Any] | None = None
        self._status: CampaignStatus | None = None
        self._generated: Generated | None = None
        self._report: Any = None
        self._busy = False
        self._request_id = 0
        self._auto_check_id: Any = None

        self.root = tk.Tk()
        self.root.title("MOBO-Kit - propose the next round")
        self.root.minsize(760, 520)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        chooser = ttk.Frame(outer)
        chooser.pack(fill="x")
        ttk.Label(chooser, text="Campaign workbook:").pack(side="left")
        self.path_var = tk.StringVar()
        remembered = remembered_workbook()
        if remembered is not None:
            self.path_var.set(str(remembered))
        ttk.Entry(chooser, textvariable=self.path_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(chooser, text="Browse...", command=self.browse).pack(side="left")

        self.headline = ttk.Label(outer, text="Choose a workbook, then check it.")
        self.headline.pack(anchor="w", pady=(12, 4))

        # a text pane with both scrollbars, laid out on a grid so neither one
        # overlaps the text -- proposed conditions are ten columns wide
        pane = ttk.Frame(outer)
        pane.pack(fill="both", expand=True)
        pane.rowconfigure(0, weight=1)
        pane.columnconfigure(0, weight=1)
        self.text = tk.Text(pane, wrap="none", height=20, state="disabled")
        scroll_y = ttk.Scrollbar(pane, orient="vertical", command=self.text.yview)
        scroll_x = ttk.Scrollbar(pane, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")

        self.progress = ttk.Progressbar(outer, mode="indeterminate")

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(10, 0))
        self.check_button = ttk.Button(buttons, text="Check workbook", command=self.check)
        self.check_button.pack(side="left")
        self.generate_button = ttk.Button(
            buttons, text="Propose next round", command=self.generate, state="disabled"
        )
        self.generate_button.pack(side="left", padx=6)
        self.reveal_button = ttk.Button(
            buttons, text="Show the new sheet", command=self.reveal, state="disabled"
        )
        self.reveal_button.pack(side="left")
        # Enabled from the start: it needs measurements, not a proposal, and the
        # question "is the model learning anything yet" is worth asking before
        # deciding whether to spend fifteen films on a batch.
        self.figures_button = ttk.Button(
            buttons, text="Figures from current data", command=self.figures
        )
        self.figures_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Close", command=self.root.destroy).pack(side="right")

        self.root.after(120, self._drain)
        if remembered is not None:
            self._auto_check_id = self.root.after(200, self.check)

    # -- helpers ----------------------------------------------------------- #

    def _write(self, body: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", body)
        self.text.configure(state="disabled")

    def _config_or_load(self) -> dict[str, Any]:
        if self._config is None:
            if not self._config_path.exists():
                raise LauncherError(
                    f"Cannot find the campaign configuration at {self._config_path}. "
                    "Run the launcher from the MOBO-Kit folder, or pass the path as "
                    "an argument."
                )
            self._config = load_campaign_config(self._config_path)
        return self._config

    def _start(self, message: str) -> None:
        self._busy = True
        self.check_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        self.headline.configure(text=message)
        self.progress.pack(fill="x", pady=(8, 0))
        self.progress.start(12)

    def _finish(self) -> None:
        self._busy = False
        self.progress.stop()
        self.progress.pack_forget()
        self.check_button.configure(state="normal")
        can = self._status is not None and self._status.can_generate
        self.generate_button.configure(state="normal" if can else "disabled")

    def _cancel_auto_check(self) -> None:
        """Drop the startup auto-check the moment the user does anything.

        Without this it fires 200 ms in and answers a question about whichever
        workbook was remembered, which may no longer be the one on screen.
        """
        if self._auto_check_id is not None:
            try:
                self.root.after_cancel(self._auto_check_id)
            except Exception:
                pass
            self._auto_check_id = None

    def _selection(self) -> str:
        raw = self.path_var.get().strip()
        try:
            return str(Path(raw).resolve()) if raw else ""
        except OSError:
            return raw

    def _in_thread(
        self, work: Callable[[Callable[[str, Any], None]], tuple[str, Any]]
    ) -> None:
        import threading

        self._request_id += 1
        request = self._request_id

        def post(kind: str, payload: Any) -> None:
            self._queue.put((request, kind, payload))

        def target() -> None:
            try:
                kind, payload = work(post)
                post(kind, payload)
            except Exception as exc:  # surfaced in the window, never a traceback box
                post("error", exc)

        threading.Thread(target=target, daemon=True).start()

    def drain_once(self) -> None:
        """Apply whatever the worker threads have reported, dropping stale replies.

        Separate from the polling loop so the drop rules can be tested by putting a
        message on the queue, rather than by racing two real threads and hoping the
        timing lands -- which is a flaky test of a race-condition fix.
        """
        import queue

        try:
            while True:
                request, kind, payload = self._queue.get_nowait()
                if request != self._request_id:
                    # superseded by a newer press; that request's own reply follows
                    self._finish()
                    continue
                if kind == "status" and str(payload.workbook) != self._selection():
                    # answers a workbook the user has navigated away from
                    self._finish()
                    continue
                self._handle(kind, payload)
        except queue.Empty:
            pass

    def _drain(self) -> None:
        self.drain_once()
        self.root.after(120, self._drain)

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "progress":
            self.headline.configure(text=str(payload))
            return
        if kind == "status":
            self._status = payload
            self.headline.configure(text=payload.headline)
            self._write(payload.detail())
            if payload.can_generate:
                self.generate_button.configure(
                    text=f"Propose {payload.next_round}"
                )
            self._finish()
            return
        if kind == "generated":
            self._generated = payload
            self.headline.configure(
                text=f"{payload.round_name} written. Nothing is approved -- read it first."
            )
            self._write(payload.summary())
            self.reveal_button.configure(state="normal")
            self._status = None
            self._finish()
            return
        if kind == "report":
            self._report = payload
            self.headline.configure(
                text=f"{len(payload.figures)} figures written. Nothing is approved."
            )
            self._write(payload.summary())
            self._finish()
            return
        if kind == "error":
            error = payload
            if isinstance(error, (LauncherError, CandidateSheetError, ValueError)):
                body = str(error)
                self.headline.configure(text="Cannot continue.")
            else:
                body = (
                    "Something unexpected went wrong. The details below are for a "
                    "developer; the workbook was not modified.\n\n"
                    + "".join(
                        traceback.format_exception(
                            type(error), error, error.__traceback__
                        )
                    )
                )
                self.headline.configure(text="Unexpected error.")
            self._write(body)
            self._finish()

    # -- actions ----------------------------------------------------------- #

    def browse(self) -> None:
        from tkinter import filedialog

        self._cancel_auto_check()
        chosen = filedialog.askopenfilename(
            title="Choose the campaign workbook",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
        )
        if chosen:
            self.path_var.set(chosen)
            self.check()

    def check(self) -> None:
        self._auto_check_id = None  # this call IS the auto-check when scheduled
        if self._busy:
            return
        workbook = self.path_var.get().strip()
        if not workbook:
            self.headline.configure(text="Choose a workbook first.")
            return
        self._status = None
        self._start("Reading the workbook...")

        def work(post: Callable[[str, Any], None]) -> tuple[str, Any]:
            config = self._config_or_load()
            status = inspect_campaign(workbook, config)
            remember_workbook(workbook)
            return "status", status

        self._in_thread(work)

    def generate(self) -> None:
        self._cancel_auto_check()
        if self._busy or self._status is None or not self._status.can_generate:
            return
        workbook = self.path_var.get().strip()
        self._start(f"Proposing {self._status.next_round}...")

        def work(post: Callable[[str, Any], None]) -> tuple[str, Any]:
            config = self._config_or_load()
            generated = generate_next_round(
                workbook,
                config,
                progress=lambda message: post("progress", message),
            )
            return "generated", generated

        self._in_thread(work)

    def figures(self) -> None:
        """Render the data-only report. Needs measurements, not a proposal."""
        self._cancel_auto_check()
        if self._busy:
            return
        workbook = self.path_var.get().strip()
        if not workbook:
            self.headline.configure(text="Choose a workbook first.")
            return
        self._start("Rendering figures from the measured data...")

        def work(post: Callable[[str, Any], None]) -> tuple[str, Any]:
            config = self._config_or_load()
            manifest = generate_data_report(
                workbook,
                config,
                progress=lambda message: post("progress", message),
            )
            return "report", manifest

        self._in_thread(work)

    def reveal(self) -> None:
        if self._generated is not None:
            reveal(self._generated.sheet_path)

    def run(self) -> None:
        self.root.mainloop()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the double-click launchers and ``python -m``."""
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = args[0] if args else DEFAULT_CONFIG
    try:
        LauncherWindow(config_path).run()
    except ImportError as exc:  # tkinter absent from a stripped Python
        print(
            "This launcher needs tkinter, which this Python does not have "
            f"({exc}). Install a python.org build, or use the API directly:\n"
            "    from mobo_kit.launcher import generate_next_round",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
