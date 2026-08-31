"""
Unified long-running task progress for Howso ``Trainee`` methods.

Howso surfaces two distinct progress signals:

* **Engine-side** — methods that accept ``task_id`` cooperate with
  ``client.get_progress(trainee_id, task_id)``, which can be polled from
  another thread for step/total updates emitted by the Amalgam engine. This
  requires an engine that tolerates being polled while it works: a
  single-threaded engine does not — polling one terminates the process — so
  for those Trainees the engine source is silently skipped and only the
  Python-side signal is used. See :func:`engine_polling_supported`.
* **Python-side** — methods that accept ``progress_callback`` chunk work in
  Python and invoke the callback with a :class:`ProgressTimer` between
  batches.

A few methods (``train``, ``react``, ``react_series``,
``react_series_stationary``) expose both. This module wires either or both
into a single reporter so a caller does not need to know which is available.

Typical use::

    from howso.utilities import with_progress

    with_progress("Train", trainee.train, data, ...)
    with_progress("React", trainee.analyze, ...)
"""
from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from functools import wraps
import importlib.util
import inspect
import io
import os
import sys
import threading
from time import monotonic
from typing import Any, Literal, overload, Protocol, TypeVar
from uuid import uuid4

from rich.console import Console
from rich.measure import Measurement
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from howso.utilities.monitors import ProgressTimer

__all__ = [
    "ProgressEvent",
    "ProgressReporter",
    "RichDisplayProgressReporter",
    "RichNotebookProgressReporter",
    "RichProgressReporter",
    "SimpleProgressReporter",
    "auto_progress",
    "auto_progress_enabled",
    "auto_progress_forced",
    "auto_progress_scope",
    "auto_reporter",
    "disable_auto_progress",
    "enable_auto_progress",
    "engine_polling_supported",
    "reset_auto_progress",
    "with_progress",
]


# Used by auto_progress()
_M = TypeVar("_M", bound=Callable[..., Any])


ProgressSource = Literal["engine", "batch", "activity"]
"""
Which mechanism a progress event came from.

``activity`` is not a measurement — it is declared when a method opts into
:func:`auto_progress`'s ``indeterminate`` and offers no real progress hook, so
that a long call still reads as alive rather than silent.
"""


def _env_number(name: str, default: float) -> float:
    """Read a numeric environment variable, falling back on anything unparsable."""
    with suppress(TypeError, ValueError):
        return float(os.environ[name]) if name in os.environ else default
    return default


# Databricks notebook cells can disconnect if no output is emitted for ~30s.
# A heartbeat well under that window keeps the cell alive during long batches.
HEARTBEAT_INTERVAL = _env_number("HOWSO_HEARTBEAT_INTERVAL", 15.0)

# Width of a notebook progress bar. Pinned rather than left to rich, whose
# ``Console.size`` probes ``os.get_terminal_size()`` on the std file descriptors
# before falling back to 80 — so a kernel launched from a terminal would
# silently inherit *that* terminal's width.
NOTEBOOK_COLUMNS = int(_env_number("HOWSO_PROGRESS_COLUMNS", 120.0))

# Refresh rate for notebook bars. Every frame is a full-width write over the
# kernel's IOPub channel, so this stays well below rich's default of 10 and
# under Jupyter's ``iopub_data_rate_limit``, while still emitting often enough
# to satisfy the Databricks cell keepalive noted above.
NOTEBOOK_REFRESH_HZ = _env_number("HOWSO_PROGRESS_FPS", 4.0)

# Ceiling on the details column. Long text is truncated rather than wrapped:
# a second rendered line would reintroduce the cursor-up codes that
# :class:`RichNotebookProgressReporter` exists to avoid.
NOTEBOOK_DETAIL_LIMIT = 48

# Bar width in columns. Sized from the worst case that must not clip: the
# longest label, a comma-grouped 100,000,000-row counter (23 characters), the
# details, elapsed and an estimate, all inside NOTEBOOK_COLUMNS.
BAR_WIDTH = 24

@dataclass
class ProgressEvent:
    """A single progress update from one of the two progress sources."""

    source: ProgressSource
    """Which mechanism produced this event: ``engine`` polling or ``batch`` callback."""

    step: int
    """Current step within ``total`` (1-indexed conceptually, but the source decides)."""

    total: int
    """Total steps. May be ``0`` while the engine has not yet reported a bound."""

    details: str = ""
    """Human-readable description, when available."""

    eta: timedelta | None = None
    """Estimated time until this source completes, when it can be estimated."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Source-specific extras (e.g. batch response payload). Reserved for callers."""


class ProgressReporter(Protocol):
    """
    Sink for :class:`ProgressEvent` updates produced by :func:`with_progress`.

    Lifecycle is ``start`` -> zero or more ``update`` calls -> ``finish``.
    ``start`` declares the set of progress ``sources`` up front, establishing
    one logical track per source. Each subsequent ``update`` carries a
    ``source`` discriminator that selects the track it applies to:

    * ``event.source`` must be one of the ``sources`` passed to ``start``;
      updates for an undeclared source (or any update before ``start`` /
      after ``finish``) are ignored.
    * A source maps to exactly one track. How a track renders is
      implementation-defined: :class:`RichProgressReporter` updates a single
      live bar in place, while :class:`SimpleProgressReporter` emits an
      append-only stream of lines for the source.
    * ``sources`` may be empty — for example an engine-only method on a
      Trainee whose engine cannot be polled. No track is created, every
      ``update`` is ignored, and the session degrades to a label plus a
      completion line.
    """

    def start(self, label: str, *, sources: Sequence[ProgressSource]) -> None:
        """Begin a reporting session, declaring the tracks ``update`` may target."""
        ...

    def update(self, event: ProgressEvent) -> None:
        """Apply a single progress event to the track named by ``event.source``."""
        ...

    def finish(self, *, success: bool, duration: timedelta) -> None:
        """End the reporting session."""
        ...


class BaseProgressReporter:
    """Base of all implementations of a Progress Reporter."""

    _console: Console
    """Console the concrete reporter renders into. Set by each subclass."""

    _label: str
    """Current session label. Set by each subclass in ``start``."""

    def _mark_done(self, *, success: bool) -> None:
        """
        Settle every track before the final frame is rendered.

        Marks the session ended either way — which stops the spinner — and
        records whether it succeeded, which is what decides the bar's color: a
        failed session stays in the pending color, reading correctly beside the
        red mark on the completion line.

        The estimate is always cleared — a finished bar should not advertise a
        time remaining. The details are cleared only on success, where
        ``batch 7`` is merely the last chunk index and the counter already says
        ``120/120``. On a failure the same text says *where* it stopped, which
        is worth keeping.

        Parameters
        ----------
        success : bool
            Whether the wrapped call completed without raising.

        Returns
        -------
        None
        """
        progress = getattr(self, "_progress", None)
        if progress is None:
            return
        totals = {task.id: task.total for task in progress.tasks}
        for task_id in set(getattr(self, "_tasks", {}).values()):
            with suppress(Exception):
                fields: dict[str, Any] = {"done": True, "ok": success, "eta": ""}
                if success:
                    fields["details"] = ""
                    # The engine stops reporting the moment the call returns, so
                    # its last step is usually short of the total — leaving a
                    # part-filled bar that reads as still running next to its own
                    # "complete" line. The call did finish; say so.
                    total = totals.get(task_id)
                    if total is not None:
                        fields["completed"] = total
                progress.update(task_id, **fields)

    def _eta_text(self, eta: timedelta | None) -> str:
        """
        Render an estimate, labeled to fit this reporter's console.

        Parameters
        ----------
        eta : timedelta or None
            The estimate to render.

        Returns
        -------
        str
            The rendered estimate, or ``""`` when there is none.
        """
        return _format_eta(eta, long=self._console.width >= ETA_LABEL_MIN_WIDTH)

    def _completion_markup(self, *, success: bool, duration: timedelta) -> str:
        """
        Build the final status line shared by every reporter.

        Parameters
        ----------
        success : bool
            Whether the wrapped call completed without raising.
        duration : timedelta
            Total elapsed time.

        Returns
        -------
        str
            Console markup for the line, or ``""`` when there is no label to
            name and so nothing worth printing.
        """
        if not self._label:
            return ""
        marker = "[green]✓[/green]" if success else "[red]✗[/red]"
        status = "complete" if success else "failed"
        # Bold cyan to match the bar row above it, where the task name is
        # already styled that way.
        return (
            f"{marker} [bold cyan]{self._label}[/bold cyan] "
            f"{status} in {_format_duration(duration)}"
        )

    def _flush_all(self) -> None:
        """
        Drain every stream this reporter could interleave with.

        Progress is rendered either as a live region redrawn in place
        (:class:`RichProgressReporter`) or as an append-only run of lines
        (:class:`SimpleProgressReporter`). Either way, bytes still sitting in
        some *other* stream's buffer — a warning written to ``stderr``, a
        ``print`` from the wrapped call — surface whenever that buffer
        happens to drain, which is frequently mid-redraw. Flushing at the
        session boundaries forces that output out first, so it lands above
        the progress region instead of through it.

        Streams are deduplicated by identity (the console's file is normally
        ``sys.stdout`` itself) and flushed newest-wrapper-first, so a stack
        such as ipykernel's ``OutStream`` over the real ``stdout`` drains in
        order. Every flush is individually guarded: a closed, detached, or
        replaced stream — pytest's captured ``stdout``, a torn-down notebook
        kernel — must not take down the reporting session.

        Note that this reaches only Python-level buffers.

        Returns
        -------
        None
        """
        streams = (
            self._console.file,
            sys.stdout,
            sys.stderr,
            sys.__stdout__,
            sys.__stderr__,
        )
        seen: set[int] = set()
        for stream in streams:
            flush = getattr(stream, "flush", None)
            if flush is None or id(stream) in seen:
                continue
            seen.add(id(stream))
            with suppress(Exception):
                flush()


class RichProgressReporter(BaseProgressReporter):
    r"""
    Rich-based reporter that renders every progress source as one merged bar.

    A session declares its sources upstream, in :func:`with_progress`, from the
    wrapped method's ``progress_callback`` / ``task_id`` hooks. However many
    there are, they share a single bar: ``batch`` owns it whenever it is live,
    because it is the meaningful outer measure, and ``engine`` is folded into
    the details column beside it::

        React ---------------->  2/4  engine 1/3 - reacting        0:00:09

    One bar rather than a nested pair is what lets this render identically in a
    terminal and in a notebook. A notebook front-end renders SGR color codes
    and treats ``\r`` as a real line rewind, but *discards* cursor-motion codes
    rather than acting on them, so any layout taller than one line would append
    a fresh copy of itself on every refresh instead of redrawing in place.
    Rather than keep two layouts in step, there is one.

    Subclasses vary only in **delivery** -- how a rendered frame reaches the
    reader. The lifecycle around it (build the model, apply events, settle the
    tracks, print the completion line) lives here and is not overridden;
    :meth:`_make_console`, :meth:`_open_delivery`, :meth:`_deliver` and
    :meth:`_close_delivery` are the seams.

    With no sources at all, nothing is rendered until the final completion line.

    Parameters
    ----------
    console : Console, optional
        Console to render into. Defaults to :meth:`_make_console`.
    transient : bool, optional
        Override :attr:`_transient` for this instance.
    """

    _refresh_hz: float = 10.0
    """Frames per second for the live region. rich's own default."""

    _transient: bool = True
    """
    Which route a *successful* session takes to hand its line to the summary.

    Cleared, and the summary is printed where the region was; kept, and the
    summary is repainted *into* the region as its last frame. The reader ends
    up with the same single line either way — this only picks how. Clearing is
    the natural fit for a terminal and is the default, but it costs a
    cursor-up, so a front-end that discards those turns it off.

    A failed session ignores this and keeps its bar, whichever route the
    reporter would normally take. See :meth:`_close_delivery`.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        console: Console | None = None,
        transient: bool | None = None,
    ) -> None:
        """Initialize the reporter."""
        self._console = console if console is not None else self._make_console()
        if transient is not None:
            self._transient = transient
        self._progress: _SummarizingProgress | None = None
        self._tasks: dict[ProgressSource, TaskID] = {}
        self._label: str = ""
        self._primary: ProgressSource | None = None
        self._secondary: ProgressSource | None = None
        self._detail: str = ""
        self._engine: tuple[int, int, str] | None = None

    def _make_console(self) -> Console:
        """
        Build the console to render into when the caller supplies none.

        Returns
        -------
        Console
            A stock console, which measures the attached terminal.
        """
        return Console()

    def _bar_column(self) -> BarColumn:
        """
        Build the bar itself, so subclasses can substitute one.

        Returns
        -------
        BarColumn
            A bar that colors itself from the session's state.
        """
        return _StateBarColumn(bar_width=BAR_WIDTH)

    def _make_columns(self, *, activity: bool = False) -> tuple[ProgressColumn, ...]:
        """
        Build the column layout shared by every bar this reporter renders.

        The label column sizes itself from the one label it holds. Nothing has
        to line up across sessions: only one bar is ever on screen, since a
        successful session hands its line to the summary on the way out.

        Parameters
        ----------
        activity : bool, default False
            Drop the counter, details and estimate columns. An activity track
            measures nothing, so they would render a meaningless ``0/?`` and
            two permanently empty columns — each still costing its padding.

        Returns
        -------
        tuple of ProgressColumn
            The columns to hand to :class:`rich.progress.Progress`.
        """
        label = TextColumn("[bold cyan]{task.description}")
        if activity:
            # No details column either: an activity track carries none, and an
            # empty column still takes its padding on both sides, leaving a
            # visible extra space between the bar and the elapsed time.
            return (
                _StateSpinnerColumn(),
                label,
                self._bar_column(),
                _ElapsedColumn(),
            )
        return (
            _StateSpinnerColumn(),
            label,
            self._bar_column(),
            _CountColumn(),
            _ElapsedColumn(),
            # Green to tie it to the duration on the completion line, which is
            # the figure this is predicting.
            TextColumn("[green]{task.fields[eta]}"),
        )

    def _compose_details(self) -> str:
        """
        Render the details column from whichever sources have reported.

        Returns
        -------
        str
            The primary source's own details, with the secondary source's
            progress prepended when it has reported any.
        """
        if self._engine is None:
            return self._detail
        step, total, detail = self._engine
        note = f"engine {step}/{total or '?'}"
        return _one_line(f"{note} · {detail}" if detail else note)

    def _prepare_merged(self, label: str, sources: Sequence[ProgressSource]) -> bool:
        """
        Build the single-bar model, without delivering anything.

        Splitting this from delivery is what lets every reporter share one
        rendering: whether frames are repainted over stdout or pushed into a
        display slot, they are frames of the same model.

        Parameters
        ----------
        label : str
            Session label.
        sources : sequence of ProgressSource
            Declared sources, all of which share one bar.

        Returns
        -------
        bool
            False when there is nothing to track, so the caller can bail out
            rather than open a live region or claim a display slot for nothing.
        """
        self._label = label
        self._detail = ""
        self._engine = None
        # ``batch`` is the outer measure when it is live, so it owns the bar and
        # ``engine`` is demoted to the details text. Otherwise the first declared
        # source owns it, which matters for ``activity``: it is neither, and
        # would otherwise leave the session blank.
        if "batch" in sources:
            self._primary = "batch"
        elif "engine" in sources:
            self._primary = "engine"
        else:
            self._primary = sources[0] if sources else None
        self._secondary = (
            "engine" if (self._primary == "batch" and "engine" in sources) else None
        )
        if self._primary is None:
            return False
        self._progress = _SummarizingProgress(
            *self._make_columns(activity=tuple(sources) == ("activity",)),
            console=self._console,
            transient=self._transient,
            # Each column takes exactly the width its content needs. Letting
            # rich expand to fill the console redistributes the slack between
            # them, which moves the bar every time the details text changes
            # length.
            expand=False,
            refresh_per_second=self._refresh_hz,
        )
        return True

    def _add_merged_task(self, label: str, sources: Sequence[ProgressSource]) -> None:
        """
        Create the one shared track and point every declared source at it.

        Parameters
        ----------
        label : str
            Session label, shown on the bar.
        sources : sequence of ProgressSource
            Sources to register against the single track.

        Returns
        -------
        None
        """
        if self._progress is None:
            return
        task = self._progress.add_task(
            label or "Working", total=None, details="", eta="", done=False, ok=False
        )
        # Registering every declared source against the one track is what makes
        # the "an undeclared source is ignored" guard in update() work.
        for source in sources:
            self._tasks[source] = task

    def _open_delivery(self) -> None:
        """
        Begin delivering frames to the reader.

        Called once per session, after the model and its track exist, so an
        implementation is free to render a first frame immediately.

        Returns
        -------
        None
        """
        if self._progress is not None:
            self._progress.start()

    def _deliver(self) -> None:
        """
        Push the current frame, if this delivery is not self-driving.

        A no-op here: rich's ``Live`` runs its own refresh thread.

        Returns
        -------
        None
        """

    def _close_delivery(self, line: str, *, success: bool) -> None:
        """
        Stop delivering frames and emit the completion line.

        The summary takes the bar's line over, by whichever route this
        reporter's front-end supports — see :attr:`_transient`. Two cases keep
        the bar instead and put the summary beneath it:

        * **A failed session.** The settled bar is the only record of how far
          the call got — ``batch 7``, ``engine 2/5`` — and the summary has room
          for none of that. Losing it on the one run where it matters most
          would be a poor trade for a saved line.
        * **An unlabeled session**, which has no summary to take the line over
          with. Clearing there would leave nothing at all.

        Both are decided here rather than per reporter, so the two routes agree
        on what a reader ends up with.

        Parameters
        ----------
        line : str
            Console markup for the completion line, or ``""`` when there is
            nothing worth printing.
        success : bool
            Whether the wrapped call completed without raising.

        Returns
        -------
        None
        """
        # Clearing only makes sense when a summary is taking the line over.
        clearing = success and self._transient and bool(line)
        if self._progress is not None:
            if success and line and not clearing:
                # Repaint the region with the summary so it lands *on* the
                # bar's line. Printing it instead would put it underneath, and
                # reclaiming the bar's line needs a cursor-up that a notebook
                # discards — which is what would split the two paths again.
                self._progress.summary = Text.from_markup(line)
            # rich reads this at stop time, so a failed session can withdraw
            # the clearing its reporter would normally do.
            self._progress.live.transient = clearing
            self._progress.stop()
        if line and not (success and not clearing):
            # Either the region was cleared on the way out, taking the summary
            # with it, or the bar is being kept and the summary goes below it.
            self._console.print(line)
        self._unwind_rich_proxies()

    @staticmethod
    def _unwind_rich_proxies() -> None:
        """
        Put ``sys.stdout``/``sys.stderr`` back if rich's ``Live`` left a proxy.

        ``Live`` swaps both streams for a ``FileProxy`` so stray prints render
        above the bar, and restores them in a ``finally``. A ``KeyboardInterrupt``
        landing inside that teardown can still leak one, which in a notebook
        kernel is sticky: every later cell would write into a dead console until
        the user restarts it.

        Returns
        -------
        None
        """
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            proxied = getattr(stream, "rich_proxied_file", None)
            if proxied is not None and proxied is not stream:
                setattr(sys, name, proxied)

    def start(self, label: str, *, sources: Sequence[ProgressSource]) -> None:
        """
        Begin a reporting session, mapping every source onto a single bar.

        Parameters
        ----------
        label : str
            Short description shown on the bar.
        sources : sequence of ProgressSource
            Which progress sources will emit events. All of them share one
            bar. May be empty, in which case nothing is rendered and no
            delivery is opened.

        Returns
        -------
        None
        """
        if not self._prepare_merged(label, sources):
            return
        self._add_merged_task(label, sources)
        # Drain both stderr and stdout, so anything already buffered lands
        # above the region about to be painted rather than through it.
        self._flush_all()
        self._open_delivery()

    def update(self, event: ProgressEvent) -> None:
        """
        Apply an event to the shared bar, routing by source.

        The primary source drives the bar's position; the secondary source
        only contributes text, so it can never move a bar it does not own.
        Events for an undeclared source, or events arriving before
        :meth:`start`, are ignored.

        Parameters
        ----------
        event : ProgressEvent
            The progress update to render.

        Returns
        -------
        None
        """
        if self._progress is None or event.source not in self._tasks:
            return
        task = self._tasks[event.source]
        if event.source == self._secondary:
            self._engine = (event.step, event.total, _one_line(event.details))
            self._progress.update(task, details=self._compose_details())
        else:
            self._detail = _one_line(event.details)
            self._progress.update(
                task,
                completed=event.step,
                total=event.total or None,
                details=self._compose_details(),
                eta=self._eta_text(event.eta),
            )
        self._deliver()

    def finish(self, *, success: bool, duration: timedelta) -> None:
        """
        Settle the bar, close delivery and emit a final completion line.

        Parameters
        ----------
        success : bool
            Whether the wrapped call completed without raising.
        duration : timedelta
            Total elapsed time, shown in the completion line.

        Returns
        -------
        None
        """
        line = self._completion_markup(success=success, duration=duration)
        if self._progress is None:
            # No session was ever rendered, so there is nothing to close.
            if line:
                self._console.print(line)
        else:
            self._mark_done(success=success)
            self._close_delivery(line, success=success)
            self._progress = None
            self._tasks.clear()
        self._primary = None
        self._secondary = None
        self._engine = None
        self._detail = ""
        self._flush_all()


def _notebook_console(*, width: int | None = None) -> Console:
    """
    Build a console that reaches rich's plain-ANSI path from inside a kernel.

    Every flag here is load-bearing:

    * ``force_jupyter=False`` is the essential one. ``Live.refresh()`` tests
      ``console.is_jupyter`` *before* its terminal branch, and that branch
      hard-requires ``ipywidgets`` — which is not a dependency. Without this
      flag a notebook session emits one ``UserWarning`` and renders nothing.
    * ``force_terminal=True`` because a kernel's stdout is not a tty, and
      ``is_terminal`` otherwise gates the live repaint off.
    * ``legacy_windows=False`` because rich computes
      ``detect_legacy_windows() and not self.is_jupyter`` — forcing
      ``is_jupyter`` off *removes* that guard, so a Windows kernel could
      otherwise fall into the Win32 render path.
    * ``color_system`` because with ``is_jupyter`` off rich detects color from
      ``TERM``/``COLORTERM``, which a kernel leaves unset — that would drop us
      to 8 colors, where rich itself uses truecolor for Jupyter.

    Parameters
    ----------
    width : int, optional
        Bar width in columns. Defaults to :data:`NOTEBOOK_COLUMNS`.

    Returns
    -------
    Console
        A console that renders ANSI progress into the kernel's stdout stream.
    """
    return Console(
        force_jupyter=False,
        force_terminal=True,
        legacy_windows=False,
        color_system="truecolor",
        width=width or NOTEBOOK_COLUMNS,
    )


def _one_line(text: str, limit: int = NOTEBOOK_DETAIL_LIMIT) -> str:
    """
    Flatten text to a single bounded line.

    Sanitizing here is mandatory rather than defensive: ``details`` arrives
    straight off an arbitrary engine payload, and a single embedded newline
    makes the bar render two lines high — which immediately reintroduces the
    cursor-up codes notebooks cannot honor.

    Parameters
    ----------
    text : str
        Raw text to flatten.
    limit : int, default NOTEBOOK_DETAIL_LIMIT
        Maximum length; longer text is truncated with an ellipsis.

    Returns
    -------
    str
        A single line of at most ``limit`` characters.
    """
    flattened = " ".join(str(text).split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1] + "\u2026"


class _SolidBar:
    """
    A single-run bar drawn in one color, for work with no known total.

    The reason we don't use the default "pulsing" bar is because that pulsing
    bar requires many times more symbols across the wire to accomplish.

    Parameters
    ----------
    style : str
        Style to draw the whole bar in.
    width : int, optional
        Bar width in columns, mirroring ``BarColumn.bar_width``. ``None`` fills
        the space available, which is rich's flexible mode. Honoring this is
        what keeps an indeterminate bar the same length as a determinate one:
        the activity layout drops two columns, and a bar that simply filled
        would absorb the freed space and render visibly longer.
    """

    def __init__(self, style: str, width: int | None = None) -> None:
        """Initialize the bar."""
        self.style = style
        self.width = width

    def __rich_console__(self, console: Console, options: Any) -> Any:
        """
        Emit the bar as one styled segment spanning the available width.

        Parameters
        ----------
        console : Console
            Console being rendered into.
        options : ConsoleOptions
            Render options, which carry the width to fill.

        Yields
        ------
        Segment
            A single run of bar glyphs.
        """
        yield Segment("\u2501" * self._width(options), console.get_style(self.style))

    def _width(self, options: Any) -> int:
        """Resolve the drawn width against the space available."""
        width = options.max_width if self.width is None else min(self.width, options.max_width)
        return max(width, 1)

    def __rich_measure__(self, console: Console, options: Any) -> Measurement:
        """
        Report the exact width this bar wants.

        Without a measurement rich hands the column all the space left over and
        the bar simply draws short inside it, so the blank remainder pushes
        every following column to the right. That only shows on tracks with an
        unknown total, since a determinate bar is rich's own ``ProgressBar``,
        which measures itself.

        Parameters
        ----------
        console : Console
            Console being measured against.
        options : ConsoleOptions
            Render options carrying the available width.

        Returns
        -------
        Measurement
            The same width for minimum and maximum, so the column is exact.
        """
        width = self._width(options)
        return Measurement(width, width)


class _StateSpinnerColumn(SpinnerColumn):
    """
    A spinner that stops when the session ends, even with no known total.

    rich blanks its spinner on ``task.finished``, which is ``completed >=
    total`` — never true while the total is ``None``. A completed
    indeterminate session would otherwise keep a spinner frame on screen and
    read as though it were still running.
    """

    def render(self, task: Any) -> Any:
        """
        Render the spinner, or the finished marker once the session has ended.

        Parameters
        ----------
        task : Task
            The task to render.

        Returns
        -------
        RenderableType
            The spinner frame, or ``finished_text``.
        """
        if task.fields.get("done"):
            return self.finished_text
        # Not ``super().render``: rich blanks its spinner on ``task.finished``,
        # which is ``completed >= total``. The engine reporting its last step is
        # not the call returning, so that stops the spinner while the work
        # carries on — the same mistake that froze the elapsed clock and turned
        # the bar green early. Only the session ending stops it.
        return self.spinner.render(task.get_time())


class _StateBarColumn(BarColumn):
    r"""
    A bar that shows *state* rather than motion when the total is unknown.

    rich pulses whenever the total is unknown (``should_pulse = self.pulse or
    self.total is None`` in ``rich/progress_bar.py``), coloring every cell of
    a 20-step gradient individually: ~980 characters per frame against ~165 for
    a determinate one. On a stream repainted with a carriage return that whole
    gulf has to be padded over, producing a very wide blank line. Lowering the
    color depth does not help — the cost is the per-cell coloring, not the
    palette (measured 982 truecolor, 742 at 256, 502 at standard).

    Motion is already covered: :class:`~rich.progress.SpinnerColumn` leads every
    layout and animates from ``Live`` refreshes at constant cost. So this draws
    a single solid run instead — red while pending, green once done — which
    holds frame length exactly constant and uses one named-ANSI color that the
    front-end maps onto the user's theme.

    A task with no total is never ``finished`` in rich's terms, so completion is
    read from a ``done`` field the reporter sets. A failed session simply never
    turns green, which reads correctly beside the red mark on the completion
    line.
    """

    PENDING_STYLE = "red"
    DONE_STYLE = "green"
    # The unfilled track. rich's default is ``grey23``, a 256-cube index that no
    # theme remaps — roughly #3a3a3a, which reads as *filled* on a light
    # background. A named ANSI color lands in the palette the front-end themes.
    TRACK_STYLE = "bright_black"

    def render(self, task: Any) -> Any:
        """
        Render the bar, substituting a solid run when the total is unknown.

        Parameters
        ----------
        task : Task
            The task to render.

        Returns
        -------
        ProgressBar or _SolidBar
            The bar to draw.
        """
        succeeded = bool(task.fields.get("done") and task.fields.get("ok"))
        style = self.DONE_STYLE if succeeded else self.PENDING_STYLE
        if task.total is None:
            return _SolidBar(style, self.bar_width)
        bar = super().render(task)
        # Override rich's own styles rather than take them: it switches to
        # ``finished_style`` as soon as ``completed >= total``, but the engine
        # reporting its last step is not the call returning. A bar that goes
        # green while the work carries on says the opposite of the elapsed time
        # ticking beside it.
        bar.complete_style = style
        bar.finished_style = style
        bar.style = self.TRACK_STYLE
        return bar


class _OverwriteSafeWriter(io.TextIOBase):
    r"""
    Pad each repaint so it fully covers the frame it replaces.

    Notebook front-ends implement ``\r`` as a raw-index overwrite and strip the
    erase-line code rich pairs with it. A frame that is shorter *in characters*
    than its predecessor therefore leaves that predecessor's tail on screen,
    and because the tail usually starts mid-escape-sequence it renders as
    literal text — ``;112m`` and the like.

    Constant visible width does not help: what matters is raw length, and the
    two diverge wildly. rich's indeterminate pulse spends ~980 characters on a
    20-step color gradient occupying the same ~97 columns that a determinate
    frame draws in ~165.

    Padding uses **spaces**, deliberately. Escape sequences would occupy raw
    length while rendering as nothing, which looks ideal until the incoming
    frame ends part-way through one: the orphaned remainder then renders as
    literal text (``[0m``). A cut run of spaces is still spaces, so residue
    stays invisible no matter where the boundary falls. The pad target only
    grows, since the screen holds the longest frame written so far.

    Parameters
    ----------
    wrapped : Any
        The file object to write through to.
    """

    _PAD = " "

    # Cursor show/hide, which rich emits because we force ``is_terminal``.
    # There is no cursor to hide in a notebook, so these are pure noise: most
    # front-ends drop them, but nbconvert's HTML export handles only SGR codes
    # and renders the rest as literal text — a stray ``[?25l`` above the bars.
    _CURSOR_CODES = ("\x1b[?25l", "\x1b[?25h")

    def __init__(self, wrapped: Any) -> None:
        """Initialize the writer."""
        # ``io.TextIOBase`` so this satisfies the ``IO[str]`` that
        # ``Console.file`` is typed as. rich wraps stdout the same way for the
        # same reason — see ``rich.file_proxy.FileProxy``.
        super().__init__()
        self._wrapped = wrapped
        self._written = 0

    def writable(self) -> bool:
        """Report that this stream accepts writes."""
        return True

    def write(self, text: str) -> int:
        r"""
        Write ``text``, padding a repaint that would under-cover the last one.

        Parameters
        ----------
        text : str
            The text rich is emitting. A repaint arrives as a single write
            beginning with ``\r``.

        Returns
        -------
        int
            Characters written by the underlying file.
        """
        for code in self._CURSOR_CODES:
            text = text.replace(code, "")
        if not text:
            return 0
        if "\r" in text:
            # Pad the current line only: what is on screen is the text after the
            # last carriage return, up to the newline that ends the region. A
            # frame's trailing newline is not always last in the write — rich
            # appends a show-cursor code after it — so padding the whole write
            # would spill blanks onto the following line.
            head, _, tail = text.rpartition("\r")
            line, newline, rest = tail.partition("\n")
            shortfall = self._written - len(line)
            if shortfall > 0:
                line += self._PAD * shortfall
            text = f"{head}\r{line}{newline}{rest}"
            # A carriage return puts the cursor back at column zero, so only
            # what follows the last one is on the line now.
            self._written = self._line_length(f"{line}{newline}{rest}", 0)
        else:
            # No carriage return: this appends to whatever is already there.
            # Tracking it matters — rich emits its *first* frame with no cursor
            # positioning at all, since it has no previous frame to rewind over.
            # Leaving that one untracked is what let a session ending before the
            # second frame (a call fast enough to finish inside one refresh
            # interval) leave the whole bar standing behind its own summary.
            self._written = self._line_length(text, self._written)
        return self._wrapped.write(text)

    @staticmethod
    def _line_length(text: str, start: int) -> int:
        """
        Return how much is on the line once ``text`` is written from ``start``.

        Parameters
        ----------
        text : str
            The text being written, with no carriage returns before it matters.
        start : int
            Raw characters already on the line.

        Returns
        -------
        int
            Raw characters on the line the cursor ends up on. A newline starts
            a fresh line, so only what follows the last one counts.
        """
        if "\n" in text:
            return len(text.rpartition("\n")[2])
        return start + len(text)

    def flush(self) -> None:
        """Flush the underlying file."""
        self._wrapped.flush()

    def isatty(self) -> bool:
        """Report whether the underlying file is a terminal."""
        return bool(getattr(self._wrapped, "isatty", bool)())


class _SummarizingProgress(Progress):
    """
    A ``Progress`` whose live region can be handed a closing line to render.

    Progress reporting ends with two things to say — the bar's final state and
    a one-line summary — and only one line to say them on. Printing the summary
    separately puts it *beneath* the bar, and reclaiming the bar's line then
    needs a cursor-up, which a notebook front-end discards. Rendering the
    summary as the region's own last frame repaints it exactly where the bar
    was, using nothing but the carriage return every front-end honors.

    ``get_renderable`` is rich's documented hook for exactly this.
    """

    summary: Text | None = None
    """Renders in place of the bars once set. ``None`` while the session runs."""

    def get_renderable(self) -> Any:
        """
        Return the summary once there is one, else the bars.

        Returns
        -------
        Any
            A rich renderable for the current frame.
        """
        if self.summary is not None:
            return self.summary
        return super().get_renderable()


class RichNotebookProgressReporter(RichProgressReporter):
    r"""
    Rich reporter delivering frames to a notebook front-end's stdout.

    The layout is the parent's, unchanged — that is the point. What differs is
    the two things a kernel demands of the stream underneath it:

    * The console must be built to reach rich's plain-ANSI path from inside a
      kernel, which :func:`_notebook_console` does; a stock ``Console()`` there
      silently renders nothing.
    * The repaint must survive a front-end that implements ``\r`` as a raw-index
      overwrite and drops the erase-line code, so a frame shorter than its
      predecessor would leave that predecessor's tail on screen.
      :class:`_OverwriteSafeWriter` pads over it.

    ``transient`` is not exposed. Clearing the finished bar needs the cursor
    motion a notebook discards, so a caller who wants that should construct
    :class:`RichProgressReporter` directly.

    Parameters
    ----------
    console : Console, optional
        Console to render into. Defaults to :func:`_notebook_console`. A
        console supplied here is used as-is, so it must already be built with
        ``force_jupyter=False`` and ``force_terminal=True`` — rich exposes no
        way to retrofit those.
    """

    _refresh_hz: float = NOTEBOOK_REFRESH_HZ
    """Slower than a terminal: every frame is a message over the kernel's IOPub."""

    _transient: bool = False
    """
    Never clear: rich clears a one-line region with ``\r ESC[1A ESC[2K``, and a
    notebook honors only the first of those three. The bar would survive and the
    summary would land beneath it, indented by the overwrite padding. Repainting
    the region with the summary reaches the same place using only ``\r``.
    """

    def __init__(self, *, console: Console | None = None) -> None:
        """Initialize the reporter."""
        super().__init__(console=console)
        self._unwrapped_file: Any = None

    def _make_console(self) -> Console:
        """
        Build a console that renders ANSI from inside a kernel.

        Returns
        -------
        Console
            A console on rich's plain-ANSI path, pinned to a fixed width.
        """
        return _notebook_console()

    def _open_delivery(self) -> None:
        """
        Wrap the stream against short frames, then start the live region.

        Returns
        -------
        None
        """
        self._unwrapped_file = self._console.file
        self._console.file = _OverwriteSafeWriter(self._unwrapped_file)  # pyright: ignore[reportAttributeAccessIssue]
        super()._open_delivery()

    def _close_delivery(self, line: str, *, success: bool) -> None:
        """
        Stop the live region, then put the plain stream back.

        The wrapper must stay in place across the ``super()`` call, because
        that is what calls ``Progress.stop()`` — which emits one final frame.
        Restoring first left that frame un-padded against a much longer
        predecessor, which was the whole visible bug.

        Parameters
        ----------
        line : str
            Console markup for the completion line.
        success : bool
            Whether the wrapped call completed without raising.

        Returns
        -------
        None
        """
        try:
            super()._close_delivery(line, success=success)
        finally:
            if self._unwrapped_file is not None:
                self._console.file = self._unwrapped_file
                self._unwrapped_file = None


def _interactive_frontend() -> bool:
    """
    Report whether a live front-end is driving this kernel.

    A notebook executed headlessly — ``nbconvert``, ``papermill``, anything
    built on ``nbclient`` — runs a real kernel with a real IPython shell, so
    every other notebook check passes, but nothing is rendering the output
    interactively. Its HTML export handles only SGR escape codes and does not
    implement carriage-return overwrite at all, so an in-place repaint is
    committed to the document one frame per line.

    The signal is the execute request's ``allow_stdin`` flag, which the kernel
    records per request. ``nbclient`` hard-codes ``self.kc.allow_stdin = False``
    while JupyterLab's client defaults it to ``true``. Anything that omits the
    field reads as ``False`` (``kernelbase`` does ``content.get("allow_stdin",
    False)``), so an unrecognized front-end degrades to plain lines rather than
    to corrupted output — the safe direction.

    Returns
    -------
    bool
        True when the current execution can prompt for input, and therefore has
        someone watching it.
    """
    ipython_mod = sys.modules.get("IPython")
    if ipython_mod is None:
        return False
    get_ipython = getattr(ipython_mod, "get_ipython", None)
    shell = get_ipython() if callable(get_ipython) else None
    return bool(getattr(getattr(shell, "kernel", None), "_allow_stdin", False))


def _display_handle_available() -> bool:
    """
    Report whether an updatable IPython display slot can be created.

    This checks only for a live IPython shell and an importable
    ``IPython.display``. It deliberately does **not** try to establish whether
    the front-end honors ``update_display_data`` — like widget support, that is
    unknowable from the kernel, which has no back-channel and may be serving
    several front-ends at once. The difference is the failure mode: a
    front-end that ignores the message appends frames instead of updating,
    which is visible and recoverable, rather than rendering nothing.

    Returns
    -------
    bool
        True when :class:`RichDisplayProgressReporter` can obtain a handle.
    """
    ipython_mod = sys.modules.get("IPython")
    if ipython_mod is None:
        return False
    get_ipython = getattr(ipython_mod, "get_ipython", None)
    if not callable(get_ipython) or get_ipython() is None:
        return False
    return importlib.util.find_spec("IPython.display") is not None


class _TightRenderable:
    """
    Wrap a rich renderable so its notebook HTML carries no outer margin.

    rich emits its Jupyter HTML as a bare ``<pre>`` with no margin reset
    (``rich/jupyter.py`` ``JUPYTER_HTML_FORMAT``), which browsers give a
    default ``margin: 1em 0``. Every progress group is its own notebook output
    block, so those margins stack between groups and read as a large gap.

    Parameters
    ----------
    renderable : Any
        The rich renderable to delegate to. Must itself be a ``JupyterMixin``.
    """

    def __init__(self, renderable: Any) -> None:
        """Initialize the wrapper."""
        self._renderable = renderable

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs: Any
    ) -> dict[str, str]:
        """
        Return the delegate's mimebundle with the ``<pre>`` margin stripped.

        Parameters
        ----------
        include : sequence of str
            Mime types to keep, per the IPython protocol.
        exclude : sequence of str
            Mime types to drop, per the IPython protocol.
        **kwargs : Any
            Forwarded to the delegate.

        Returns
        -------
        dict
            Mime type to rendered content.
        """
        data = self._renderable._repr_mimebundle_(include, exclude, **kwargs)
        html = data.get("text/html")
        if html is not None:
            data["text/html"] = html.replace('<pre style="', '<pre style="margin:0;', 1)
        return data


class RichDisplayProgressReporter(RichNotebookProgressReporter):
    """
    Rich reporter that repaints via IPython's display-update protocol.

    This exists for notebooks rendered *headlessly* — by ``nbconvert``,
    ``papermill``, anything built on ``nbclient``. Those exports handle SGR
    color codes but do not implement carriage-return overwrite at all, so the
    parent's in-place repaint would be committed to the saved document one
    frame per line. Here the reporter claims one display slot with
    ``display(..., display_id=True)`` and replaces its contents wholesale, so
    only the last frame survives into the notebook.

    This rides ``update_display_data``, part of the core Jupyter messaging
    spec since 5.1 — notably *not* ``ipywidgets``, which cannot work here: a
    kernel cannot learn whether its front-end renders widgets, and a front-end
    that does not leaves the cell showing the string ``Output()``.

    When something *is* watching, it stays on the parent's stdout delivery
    instead. A notebook merges consecutive stdout writes into one block but
    never merges display blocks, so a display slot is fenced off from the lines
    around it — including the caller's own prints — by the notebook's padding.
    The frame is identical either way; only the block it lands in differs.

    Two things to know about the slot path. The frame is turned into HTML by
    rich's ``JupyterMixin``, which renders through the *global* console rather
    than this reporter's; in a kernel that is a Jupyter console and the output
    matches the stdout path exactly, while a colorless or narrower global
    console would render the bar's unfilled track as blanks instead. And colors
    are baked to literal hex, since the frame is delivered as HTML rather than
    ANSI. This is the one route whose palette is *not* theme-mapped: the others
    emit ``red``/``green``/``bright_black`` as ANSI codes for the front-end to
    resolve against its own theme, while here they freeze at render time to
    that palette's values (``#800000``, ``#008000``, ``#808080``). There is no
    back-channel from HTML to a theme, and this route only ever serves headless
    renders, where there is no live theme to honor in the first place.

    Parameters
    ----------
    console : Console, optional
        Console used for the completion line, and for the whole render when a
        front-end is watching. Defaults to :func:`_notebook_console`.
    """

    _slot_when_headless: bool = False
    """
    Whether a headless render claims an updatable display slot.

    Off, the default: nothing is rendered until the session ends, and the final
    state is written to stdout in one go. Nobody is watching a headless run and
    nbconvert cannot animate the frames it would receive, so the intermediate
    ones buy nothing — and a notebook merges consecutive stdout writes into a
    single output block, where every display slot is its own block carrying its
    own vertical padding. A cell of several calls reads as adjacent lines
    instead of a stack of padded panels.

    On: each session claims a slot and repaints it as events arrive. Worth
    turning on when the headless verdict may be wrong — it rests on the
    execute request's ``allow_stdin`` flag, and a front-end that leaves the
    field out reads as headless while some person watches it repaint nothing.
    """

    def __init__(self, *, console: Console | None = None) -> None:
        """Initialize the reporter."""
        super().__init__(console=console)
        self._handle: Any = None
        self._last_push: float = 0.0
        self._inline: bool = False

    @staticmethod
    def _frame(progress: Progress) -> Any:
        """
        Render the current state of every track as one renderable.

        Parameters
        ----------
        progress : Progress
            The progress model to snapshot.

        Returns
        -------
        Any
            A rich renderable carrying one row per track.
        """
        return progress.make_tasks_table(progress.tasks)

    def _push(self, *, force: bool = False) -> None:
        """
        Replace the display slot's contents, throttled.

        Unlike the ``Live`` path there is no refresh thread here — repaints are
        driven by incoming events, which arrive far faster than a reader can
        follow and each cost a full HTML payload over IOPub. ``force`` bypasses
        the throttle so the final frame always lands on 100%.

        Parameters
        ----------
        force : bool, default False
            Push regardless of how recently the last frame was sent.

        Returns
        -------
        None
        """
        if self._handle is None or self._progress is None:
            return
        now = monotonic()
        if not force and now - self._last_push < 1.0 / NOTEBOOK_REFRESH_HZ:
            return
        self._last_push = now
        with suppress(Exception):
            self._handle.update(_TightRenderable(self._frame(self._progress)))

    def _open_delivery(self) -> None:
        """
        Claim a display slot, or fall back to the parent's stdout repaint.

        Returns
        -------
        None
        """
        if self._progress is None:
            return
        if _interactive_frontend():
            # Someone is watching: stay in the caller's own output block.
            self._inline = True
            super()._open_delivery()
            return
        if not self._slot_when_headless:
            # Render nothing until the close, which writes the final state.
            return
        # Deliberately no ``Progress.start()``: that would install rich's Live
        # renderer and with it the repaint this path exists to avoid. The
        # Progress here is only a model that knows how to render itself.
        #
        # Imported here rather than at module scope: IPython is not a
        # dependency, and this class is only ever selected once
        # ``_display_handle_available()`` has confirmed a live shell.
        with suppress(ImportError):
            from IPython.display import display  # noqa: PLC0415

            # Returns None when there is no active shell to render into.
            self._handle = display(
                _TightRenderable(self._frame(self._progress)), display_id=True
            )
        self._last_push = monotonic()

    def _deliver(self) -> None:
        """
        Repaint the slot, unless the parent's live region is driving this.

        Returns
        -------
        None
        """
        if self._inline:
            super()._deliver()
            return
        self._push()

    def _close_delivery(self, line: str, *, success: bool) -> None:
        """
        Leave the reader with what the other routes leave on the bar's line.

        On success that is the summary alone, in place of the bar; on failure
        the bar is kept with the summary under it. When a slot is in play both
        go into the *same* renderable, because a notebook shows a stream block
        and a display block as two outputs, each with its own vertical padding,
        and printing the summary would open a conspicuous gap under the bar.
        Writing to stdout has no such constraint — consecutive writes land in
        one block — so that path simply prints them.

        Parameters
        ----------
        line : str
            Console markup for the completion line.
        success : bool
            Whether the wrapped call completed without raising.

        Returns
        -------
        None
        """
        if self._inline:
            # The parent is delivering; it owns the close too.
            self._inline = False
            super()._close_delivery(line, success=success)
            return
        if self._handle is None:
            # No slot: write the final state to stdout in one go. Two prints
            # rather than one stacked renderable — consecutive stdout writes
            # merge into a single output block, which is the point of this
            # path, so nothing has to hold them together.
            if self._progress is not None and not (success and line):
                self._console.print(self._frame(self._progress))
            if line:
                self._console.print(line)
            return
        frame: Any
        if success and line:
            frame = Text.from_markup(line)
        elif self._progress is not None:
            # A failed or unlabeled session keeps its bar.
            frame = self._frame(self._progress)
            if line:
                # ``Table.grid`` rather than ``Group``: only a JupyterMixin
                # carries the ``_repr_mimebundle_`` that makes IPython render
                # HTML, and Group is not one — it would reach the notebook as
                # a bare repr.
                #
                # ``expand`` is load-bearing. A grid sizes its column to the
                # child's *maximum* measurement, and a bar measures narrower
                # than it lays out — so the default sizing squeezed the frame
                # and wrapped the details onto a second row, which no other
                # route does. Expanding hands the frame the same width it gets
                # when pushed on its own.
                stacked = Table.grid(expand=True)
                stacked.add_row(frame)
                stacked.add_row(Text.from_markup(line))
                frame = stacked
        else:
            frame = None
        if frame is not None:
            with suppress(Exception):
                self._handle.update(_TightRenderable(frame))
        self._handle = None


class SimpleProgressReporter(BaseProgressReporter):
    """
    Line-printing reporter for terminals where rich's live renderer is unreliable.

    Prints a new line whenever a step changes, with a periodic heartbeat to
    keep notebook cells from timing out during a long-running step.

    Parameters
    ----------
    console : Console, optional
        Console to print to. Defaults to a fresh :class:`rich.console.Console`.
    """

    def __init__(self, *, console: Console | None = None) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialize the reporter."""
        # ``force_jupyter=False`` keeps writes on stdout instead of the
        # ``IPython.display`` path. In Jupyter / VS Code notebooks each
        # ``display(html)`` call produces a separate output block with its
        # own vertical padding; routing through stdout lets IPython collate
        # consecutive lines into a single stream output. ANSI styles still
        # render correctly in the notebook viewer.
        self._console = console or Console(force_jupyter=False)
        self._label: str = ""
        self._last_step: dict[ProgressSource, int] = {}
        self._last_output: dict[ProgressSource, float] = {}
        self._start_time: float = 0.0
        self._prefixes: dict[ProgressSource, str] = {}
        self._finished: bool = False

    def _eta_text(self, eta: timedelta | None) -> str:
        """
        Render an estimate, always spelled out.

        Unlike a bar, these lines have no columns competing for width — the
        estimate cannot crowd anything out, so the label never needs shortening.

        Parameters
        ----------
        eta : timedelta or None
            The estimate to render.

        Returns
        -------
        str
            The rendered estimate, or ``""`` when there is none.
        """
        return _format_eta(eta)

    def start(self, label: str, *, sources: Sequence[ProgressSource]) -> None:
        """
        Begin a reporting session and print the session header.

        Parameters
        ----------
        label : str
            Short description printed as the header line.
        sources : sequence of ProgressSource
            Which progress sources will emit events; per-source step and
            heartbeat tracking is initialized for each. May be empty, in which
            case only the label and completion lines are printed.

        Returns
        -------
        None
        """
        self._label = label
        self._start_time = monotonic()
        self._finished = False
        self._last_step = dict.fromkeys(sources, -1)
        self._last_output = dict.fromkeys(sources, 0.0)
        self._flush_all()
        # Every progress line is indented under the label header for a
        # uniform "section + body" look. When both sources are present,
        # the engine bar gets an additional indent so it visually nests
        # under the batch line it belongs to.
        both = "batch" in sources and "engine" in sources
        self._prefixes = {
            s: ("    " if both and s == "engine" else "  ") for s in sources
        }
        if label:
            self._console.print(f"[bold cyan]{label}[/bold cyan]")

    def update(self, event: ProgressEvent) -> None:
        """
        Print a line for a changed step, or a heartbeat for a stalled one.

        A new line is printed whenever the step advances. When the step is
        unchanged but ``HEARTBEAT_INTERVAL`` seconds have elapsed since the
        last output, a heartbeat line is printed to keep notebook cells alive.

        Parameters
        ----------
        event : ProgressEvent
            The progress update to render.

        Returns
        -------
        None
        """
        if self._finished:
            # A late update from the poll thread (e.g. after join() timed out)
            # must not print past the completion line.
            return
        if event.source not in self._last_step:
            # An undeclared source has no track; ignore it so behavior matches
            # RichProgressReporter (see the ProgressReporter contract).
            return
        now = monotonic()
        prefix = self._prefixes.get(event.source, "")
        if event.source == "activity":
            # Nothing is being counted, so a "[0/?]" line would say nothing.
            # Report liveness on the heartbeat cadence instead: the ticker
            # fires far more often than a reader needs to see a line.
            if now - self._last_output.get(event.source, 0.0) < HEARTBEAT_INTERVAL:
                return
            elapsed = timedelta(seconds=int(now - self._start_time))
            detail = f" {event.details}" if event.details else ""
            self._console.print(f"{prefix}[dim]\u00b7{detail} {elapsed} elapsed[/dim]")
            self._last_output[event.source] = now
            return
        total = event.total or "?"
        width = len(str(total))
        eta = self._eta_text(event.eta)
        if event.step != self._last_step.get(event.source, -1):
            suffix = f" [dim]\u00b7 {eta}[/dim]" if eta else ""
            self._console.print(
                f"{prefix}[dim]\\[{event.step:>{width}}/{total}][/dim] {event.details}{suffix}"
            )
            self._last_step[event.source] = event.step
            self._last_output[event.source] = now
        elif now - self._last_output.get(event.source, 0.0) >= HEARTBEAT_INTERVAL:
            elapsed = timedelta(seconds=int(now - self._start_time))
            # A stalled step is exactly when a reader wants the estimate.
            trailer = f" \u00b7 {eta}" if eta else ""
            self._console.print(
                f"{prefix}[dim]\\[{event.step:>{width}}/{total}] · {elapsed} elapsed{trailer}[/dim]"
            )
            self._last_output[event.source] = now

    def finish(self, *, success: bool, duration: timedelta) -> None:
        """
        Print a final completion line.

        Parameters
        ----------
        success : bool
            Whether the wrapped call completed without raising.
        duration : timedelta
            Total elapsed time, shown in the completion line.

        Returns
        -------
        None
        """
        self._finished = True
        line = self._completion_markup(success=success, duration=duration)
        if line:
            self._console.print(line)
        self._flush_all()


def auto_reporter(*, console: Console | None = None) -> ProgressReporter:
    """
    Choose the reporter that best fits the current environment.

    Selection runs in this order:

    1. ``HOWSO_SIMPLE_PROGRESS`` set to a truthy value
       (``1``/``on``/``true``/``yes``) forces the line-printing reporter. This
       is the escape hatch from everything below.
    2. A tty gets :class:`RichProgressReporter`, repainting a live region.
    3. A notebook kernel gets a rich reporter when it passes the checks each
       one needs. With a live IPython shell,
       :class:`RichDisplayProgressReporter`, which repaints stdout while a
       front-end is watching and falls back to a display slot for a headless
       render; with an interactive front-end but no importable
       ``IPython.display``, :class:`RichNotebookProgressReporter`, which
       repaints stdout using only the control codes those front-ends honor.
    4. Anything else — a notebook that passes neither check, a pipe, a
       redirect, a CI log — gets :class:`SimpleProgressReporter`.

    All three rich reporters render the same merged bar and differ only in how
    a frame reaches the reader. The tty check deliberately precedes the
    notebook check: ``_in_notebook()`` is also true for *terminal* IPython,
    which has a real terminal underneath it.

    Parameters
    ----------
    console : Console, optional
        Console the chosen reporter renders into. Defaults to a fresh
        :class:`rich.console.Console` created by the reporter. Note that
        :class:`RichNotebookProgressReporter` needs specific console flags, so
        prefer leaving this unset when a notebook is in play.

    Returns
    -------
    ProgressReporter
        The reporter matching the environment, per the order above.
    """
    if _parse_tristate(os.environ.get("HOWSO_SIMPLE_PROGRESS")) is True:
        return SimpleProgressReporter(console=console)
    if sys.stdout.isatty():
        return RichProgressReporter(console=console)
    if _in_notebook():
        # The display slot survives headless execution: nbclient replaces the
        # output's data in place for each update, so an nbconvert/papermill run
        # stores the final frame as ordinary HTML and exports it faithfully.
        if _display_handle_available():
            return RichDisplayProgressReporter(console=console)
        # An in-place carriage-return repaint does not survive: nothing applies
        # it, so every frame is committed to the document as its own line.
        if _interactive_frontend():
            return RichNotebookProgressReporter(console=console)
    return SimpleProgressReporter(console=console)


def _supports_param(bound_func: Callable[..., Any], name: str) -> bool:
    """
    Report whether ``bound_func`` accepts a parameter called ``name``.

    Parameters
    ----------
    bound_func : callable
        The function to inspect. Only its signature is read, never called, so
        this is deliberately as wide as a callable annotation goes.
    name : str
        The parameter to look for.

    Returns
    -------
    bool
        True when the parameter is present. A callable whose signature cannot
        be read — a builtin, or an object with a hostile ``__signature__`` —
        reports False rather than raising, since the caller only uses this to
        decide whether it is worth passing a progress hook.
    """
    try:
        sig = inspect.signature(bound_func)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters


def engine_polling_supported(client: Any, trainee_id: str | None) -> bool:
    """
    Report whether a Trainee's ``get_progress`` is safe to poll from another thread.

    Polling a **single-threaded** engine while it is executing terminates the
    process: the engine may die before Python can raise, so no ``try``/``except``
    around the poll can contain it. This check is therefore preventative and
    *fail-closed* — polling is reported as supported only when it can be
    positively confirmed safe.

    Two sources answer this, in order:

    * A client running an engine in-process exposes the library as ``amlg``,
      which reports its own concurrency type authoritatively.
    * Any other client is asked via ``client.get_trainee_runtime``, which every
      client implements, and only the ``mt`` library type is treated as safe.

    A Trainee on a single-threaded library — and any Trainee whose engine
    cannot be identified — is reported as unsupported. Note that thread
    *counts* are not a usable signal here: a single-threaded build with OpenMP
    reports several threads, and a multi-threaded build can be pinned to one.

    ``HOWSO_ENGINE_PROGRESS=off`` (or ``0``/``no``/``false``) forces the same
    result. No value of that variable can force polling *on*, because doing so
    would re-enable a hard process crash.

    Parameters
    ----------
    client : Any
        The client whose ``get_progress`` would be polled.
    trainee_id : str or None
        The Trainee that would be polled. Without one there is nothing to ask
        about, and polling is reported as unsupported.

    Returns
    -------
    bool
        True only when engine progress polling is confirmed safe.
    """
    if client is None or not trainee_id:
        return False
    if _parse_tristate(os.environ.get("HOWSO_ENGINE_PROGRESS")) is False:
        return False
    try:
        amlg = getattr(client, "amlg", None)
        if amlg is not None:
            # An in-process library identifies its own build exactly, which
            # ``library_type`` cannot always do: it is derived from the library
            # filename, so it reports "mt" for a path carrying no postfix.
            concurrency = amlg.get_concurrency_type_string()
            if isinstance(concurrency, bytes):
                concurrency = concurrency.decode("utf-8", errors="replace")
            supported = concurrency.strip().lstrip().startswith("MultiThreaded")
        else:
            # A remote client reports the library type it provisioned. Only
            # "mt" is safe; "st" and the OpenMP builds' "st-omp" are not.
            runtime = client.get_trainee_runtime(trainee_id)
            library_type = (
                runtime.get("library_type")
                if isinstance(runtime, Mapping)
                else getattr(runtime, "library_type", None)
            )
            supported = "mt" in str(library_type).strip().split("-")
    except Exception:  # noqa: BLE001
        # A client that cannot answer, a Trainee the service no longer knows
        # about, or an unexpected runtime shape all mean the same thing here:
        # we cannot prove the engine is multi-threaded, so we must assume it
        # is not.
        return False
    return supported


def _cached_polling_support(trainee: Any, client: Any, trainee_id: str | None) -> bool:
    """
    Read a Trainee's cached polling support, falling back to asking directly.

    A ``Trainee`` resolves :func:`engine_polling_supported` once and caches it
    for its lifetime, since the library type is fixed when the Trainee is
    created. Other bound-method hosts have no such cache and are asked on
    every call.
    """
    supported = getattr(trainee, "supports_engine_progress", None)
    if supported is None:
        return engine_polling_supported(client, trainee_id)
    return bool(supported)


def _resolve_owner(
    bound_func: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Any, Any, str | None]:
    """
    Work out who owns a decorated call, and which Trainee it concerns.

    The decorators sit on client methods, but may also be applied to a facade
    such as ``Trainee`` that delegates to one. The two differ in where the
    information lives:

    * a ``Trainee`` reaches its client through ``.client`` and knows its own
      ``.id``;
    * a client *is* the client, and takes ``trainee_id`` as a call argument.

    Reading only the ``Trainee`` shape leaves both values ``None`` for a
    client-bound method, which silently disables engine polling for every
    session -- the gate cannot confirm a multi-threaded engine without them.

    Parameters
    ----------
    bound_func : callable
        The bound method being wrapped.
    args : tuple
        Positional arguments of the call.
    kwargs : Mapping
        Keyword arguments of the call.

    Returns
    -------
    tuple
        The owner, its client (or itself), and the Trainee id if known.
    """
    owner = getattr(bound_func, "__self__", None)
    if owner is None:
        return None, None, None
    # ``get_trainee_runtime`` is the same duck-type ``engine_polling_supported``
    # uses to interrogate a client, so anything it accepts is accepted here.
    client = getattr(owner, "client", None)
    if client is None and hasattr(owner, "get_trainee_runtime"):
        client = owner
    trainee_id = getattr(owner, "id", None)
    if trainee_id is None:
        # Read it from the call rather than guessing a position: these
        # signatures take ``trainee_id`` first, but that is not ours to assume.
        with suppress(TypeError, ValueError):
            bound = inspect.signature(bound_func).bind_partial(*args, **kwargs)
            trainee_id = bound.arguments.get("trainee_id")
    return owner, client, trainee_id


def with_progress(  # noqa: PLR0915
    label: str,
    bound_func: Callable[..., Any],
    /,
    *args: Any,
    reporter: ProgressReporter | None = None,
    polling_interval: float = 1.0,
    indeterminate: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Invoke ``bound_func`` with unified progress reporting.

    The function inspects ``bound_func`` for the two progress hooks Howso
    methods may expose:

    * ``task_id`` — a fresh UUID is supplied and a background thread polls
      ``trainee.client.get_progress`` while the call runs. Skipped when
      :func:`engine_polling_supported` cannot confirm the Trainee is safe to
      poll — notably Trainees on a single-threaded engine, where polling would
      kill the process.
    * ``progress_callback`` — a wrapper translates each
      :class:`ProgressTimer` tick into a :class:`ProgressEvent`.

    Whichever hooks are present and usable are wired into ``reporter``. If
    neither is — including the case where the only hook is ``task_id`` on a
    Trainee that cannot be polled — ``bound_func`` is still invoked and a
    completion line is printed.

    Parameters
    ----------
    label : str
        Short description shown by the reporter (e.g. ``"Train"``).
    bound_func : Callable
        A bound method on a ``Trainee`` instance. ``__self__`` is used to
        reach ``trainee.client`` for engine polling.
    *args, **kwargs :
        Forwarded to ``bound_func``. Any caller-supplied ``task_id`` or
        ``progress_callback`` is honored and progress wiring for that source
        is skipped to avoid stomping on the caller's choice.
    reporter : ProgressReporter, optional
        Custom reporter. Defaults to :func:`auto_reporter`.
    polling_interval : float, default 1.0
        Seconds between engine progress polls when ``task_id`` is wired.

    Returns
    -------
    Any
        Whatever ``bound_func`` returns.
    """
    reporter = reporter or auto_reporter()
    start_time = monotonic()

    has_batch_cb = (
        _supports_param(bound_func, "progress_callback")
        and kwargs.get("progress_callback") is None
    )
    has_task_id = (
        _supports_param(bound_func, "task_id")
        and kwargs.get("task_id") is None
    )

    trainee, client, trainee_id = _resolve_owner(bound_func, args, kwargs)

    # Engine polling is only useful when we can actually reach get_progress,
    # and only safe when the engine behind it tolerates a concurrent poll. A
    # single-threaded engine does not: polling it terminates the process, so
    # this gate must be preventative — the ``except`` clauses in ``_poll``
    # below are powerless against it.
    has_task_id = has_task_id and _cached_polling_support(trainee, client, trainee_id)

    sources: list[ProgressSource] = []
    if has_batch_cb:
        sources.append("batch")
    if has_task_id:
        sources.append("engine")

    if has_batch_cb:
        def _batch_cb(progress: ProgressTimer, *_extra: Any, **__: Any) -> None:
            # ``time_remaining`` divides by ``max(current_tick, 1)``, so before
            # the first tick lands it reports roughly the whole run as still
            # remaining. Withhold it until there is a real measurement, and
            # tolerate the timer not having been started at all.
            eta: timedelta | None = None
            if progress.current_tick > 0:
                with suppress(ValueError):
                    eta = progress.time_remaining
            reporter.update(ProgressEvent(
                source="batch",
                step=progress.current_tick,
                total=progress.total_ticks,
                details=f"batch {progress.update_count}",
                eta=eta,
            ))
        kwargs["progress_callback"] = _batch_cb

    stop_event = threading.Event()
    poll_thread: threading.Thread | None = None

    if has_task_id:
        # Deferred import: howso.client pulls in howso.utilities at import time,
        # so we can't import it at module scope without a circular dependency.
        from howso.client.exceptions import NoOngoingTaskError  # noqa: PLC0415

        task_id = str(uuid4())
        kwargs["task_id"] = task_id

        def _poll() -> None:
            while not stop_event.is_set():
                try:
                    p = client.get_progress(trainee_id, task_id)  # pyright: ignore[reportOptionalMemberAccess]
                except NoOngoingTaskError:
                    # Between batches (or before the engine registers the task)
                    # there is no live task to report on; skip this tick and
                    # keep polling. Falls through to the wait below rather than
                    # ``continue`` so we don't busy-loop the engine.
                    pass
                except Exception:  # noqa: BLE001
                    # Any other engine error means progress can't be reported —
                    # stop quietly rather than killing this daemon thread with a
                    # traceback on stderr (the wrapped call itself is
                    # unaffected). Progress is best-effort.
                    return
                else:
                    # get_progress returns None between/after tasks; only
                    # render when the engine actually reported a mapping.
                    # (Skipping via ``continue`` would bypass the wait below
                    # and busy-loop the engine.)
                    if isinstance(p, Mapping):
                        reporter.update(ProgressEvent(
                            source="engine",
                            step=int(p.get("step", 0) or 0),
                            total=int(p.get("total", 0) or 0),
                            details=p.get("details") or "",
                        ))
                stop_event.wait(polling_interval)

        poll_thread = threading.Thread(target=_poll, daemon=True)

    tick_thread: threading.Thread | None = None
    if indeterminate and not sources:
        # Nothing measurable to report, but the caller asked for the session to
        # read as alive. Two of the four reporters only emit on ``update`` (the
        # display slot pushes a frame, the line printer prints a line), so a
        # ticker is what makes them show anything at all before completion.
        sources.append("activity")

        def _tick() -> None:
            while not stop_event.is_set():
                reporter.update(ProgressEvent(source="activity", step=0, total=0))
                stop_event.wait(polling_interval)

        tick_thread = threading.Thread(target=_tick, daemon=True)

    reporter.start(label, sources=tuple(sources))
    success = False
    try:
        if poll_thread is not None:
            poll_thread.start()
        if tick_thread is not None:
            tick_thread.start()
        result = bound_func(*args, **kwargs)
        success = True
        return result
    finally:
        duration = timedelta(seconds=monotonic() - start_time)
        stop_event.set()
        for thread in (poll_thread, tick_thread):
            if thread is not None:
                with suppress(RuntimeError):
                    thread.join(timeout=max(polling_interval * 2, 2.0))
        reporter.finish(success=success, duration=duration)


_state = threading.local()


_TRUTHY = {"on", "true", "yes", "y", "1"}
_FALSY = {"off", "false", "no", "n", "0"}


def _parse_tristate(value: Any) -> bool | None:
    """
    Parse a permissive tri-state value.

    Returns ``True`` / ``False`` for recognized on/off-style strings (and
    Python bools / ints), or ``None`` for ``"auto"`` / empty / unrecognized
    values so the caller can fall through to the next precedence layer.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int — check first
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return None  # "auto", "", or anything else → fallthrough


# Below this console width the spelled-out estimate label crowds the bar hard
# enough to truncate real data. Measured against the longest label this module
# generates ("React series stationary"): at 88 columns the counter and elapsed
# time render as "12000/…" and "0:00:…", while at 96 nothing is cut and the bar
# still holds 17 columns. 100 keeps a margin, and is what a notebook uses.
ETA_LABEL_MIN_WIDTH = 100


def _format_duration(delta: timedelta) -> str:
    """
    Render a duration as ``H:MM:SS``, letting the hours run past a day.

    ``str(timedelta)`` switches to ``"1 day, 1:00:00"`` past 24 hours and
    ``"41 days, 16:00:00"`` past a month — 14 to 17 characters where the same
    value was 7. In a pinned layout that silently shifts or clips a column, and
    multi-hour runs are ordinary here. Accumulating into the hours field keeps
    the width growing by one character per decade instead.

    Parameters
    ----------
    delta : timedelta
        The duration to render. Negative values clamp to zero.

    Returns
    -------
    str
        For example ``"0:00:09"``, ``"10:00:00"`` or ``"100:00:00"``.
    """
    total = max(int(delta.total_seconds()), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _format_count(value: int) -> str:
    """
    Render a step count with thousands separators.

    Batches routinely reach tens of millions of rows, where a bare
    ``100000000`` is unreadable. Grouping costs width — ``100,000,000`` is 11
    characters against 9 — which is why the bar is sized from this worst case
    rather than the other way round.

    Parameters
    ----------
    value : int
        The count to render.

    Returns
    -------
    str
        For example ``"9,999"`` or ``"100,000,000"``.
    """
    return f"{value:,}"


class _CountColumn(MofNCompleteColumn):
    """
    Step counter, carrying the details text alongside it.

    The details share this column rather than occupying their own. An empty
    column still costs its padding on both sides, so once the details are
    cleared at completion a separate column would leave two spaces before the
    elapsed time where every other row has one.
    """

    def render(self, task: Any) -> Any:
        """
        Render ``completed/total``, followed by any details.

        Parameters
        ----------
        task : Task
            The task to render.

        Returns
        -------
        Text
            The counter, and the details when there are any.
        """
        total = _format_count(int(task.total)) if task.total is not None else "?"
        counter = f"{_format_count(int(task.completed))}{self.separator}{total}"
        details = task.fields.get("details") or ""
        text = Text(counter, style="progress.download")
        if details:
            # The label's hue, un-bolded: the details describe the same task, so
            # they read as subordinate to it rather than as a separate signal.
            # Not rich's ``progress.data.speed``, which resolves to plain red —
            # the color a pending bar uses, meaning something else entirely.
            text.append(f" {details}", style="cyan")
        return text


class _ElapsedColumn(TimeElapsedColumn):
    """Elapsed time in the same ``H:MM:SS`` form as every other duration."""

    def render(self, task: Any) -> Any:
        """
        Render elapsed time.

        Parameters
        ----------
        task : Task
            The task to render.

        Returns
        -------
        Text
            The elapsed time, or a placeholder before the task starts.
        """
        # Deliberately not rich's ``finished_time if task.finished``: rich
        # freezes that the instant ``completed >= total``, but the engine
        # reporting its last step is not the call returning. An ``analyze``
        # whose engine reported 1/1 after a second went on working for another
        # seventeen, and the bar sat at 0:00:01 beside a completion line
        # reading 0:00:18. ``elapsed`` keeps running until the Progress stops,
        # which is the moment the session actually ends.
        elapsed = task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        return Text(_format_duration(timedelta(seconds=elapsed)), style="progress.elapsed")


def _format_eta(eta: timedelta | None, *, long: bool = True) -> str:
    """
    Render an estimate, labeled to fit the space available.

    Sub-second precision is dropped: a ``timedelta`` stringifies with
    microseconds (``0:01:23.456789``), which is false precision on a figure
    that is an estimate to begin with.

    Parameters
    ----------
    eta : timedelta or None
        The estimate to render.
    long : bool, default True
        Whether there is room to spell the label out. Callers competing for
        horizontal space pass ``False`` when the console is narrow, so the bar
        and counter keep their columns; see :data:`ETA_LABEL_MIN_WIDTH`.

    Returns
    -------
    str
        ``"est. rem.: 0:01:23"``, ``"ETA 0:01:23"``, or ``""`` when there
        is no usable estimate.
    """
    if eta is None or int(eta.total_seconds()) <= 0:
        # Below a second there is nothing useful left to say, and "0:00:00"
        # on a bar that has just filled reads as a stuck clock.
        return ""
    label = "est. rem.:" if long else "ETA"
    return f"{label} {_format_duration(eta)}"


def _default_label(name: str) -> str:
    """Derive a human label from a method name (``react_series`` → ``React series``)."""
    return name.replace("_", " ").capitalize()


def _in_notebook() -> bool:
    """Return True if running inside Jupyter / IPython / Databricks."""
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return True
    ipython_mod = sys.modules.get("IPython")
    if ipython_mod is None:
        return False
    get_ipython = getattr(ipython_mod, "get_ipython", None)
    return callable(get_ipython) and get_ipython() is not None


def _config_auto_progress(owner: Any) -> bool | None:
    """
    Resolve the ``auto_progress`` setting from the client configuration.

    ``owner`` is whatever the decorated method is bound to, which may be either
    a ``Trainee`` — reaching its client through ``.client`` — or a client
    itself, which carries ``.configuration`` directly. Handling both matters:
    when the decorators moved onto the client methods, reading only through
    ``.client`` silently stopped consulting the configuration at all.

    Parameters
    ----------
    owner : Any
        The object the decorated method is bound to.

    Returns
    -------
    bool or None
        The configured value, or None to defer to the next precedence layer.
    """
    cfg = getattr(getattr(owner, "client", None), "configuration", None)
    if cfg is None:
        cfg = getattr(owner, "configuration", None)
    value = getattr(cfg, "auto_progress", None) if cfg is not None else None
    return _parse_tristate(value)


def _auto_progress_enabled(trainee: Any) -> bool:
    """
    Decide whether the next decorated call on ``trainee`` should be wrapped.

    Precedence (first match wins):

    1. Re-entrancy guard — nested wrapped calls never stack bars,
       regardless of the force flag below.
    2. Thread-local force flag (``enable_auto_progress`` / ``auto_progress_scope``).
    3. ``HOWSO_PROGRESS`` env var.
    4. Client config ``auto_progress`` value.
    5. Default heuristic: TTY *or* notebook kernel.

    Both the env var and the config value accept any of
    ``on``/``true``/``yes``/``y``/``1`` for True,
    ``off``/``false``/``no``/``n``/``0`` for False, and ``auto`` (or
    anything unrecognized) to defer to the next precedence layer.
    """
    if getattr(_state, "depth", 0) > 0:
        return False

    forced = getattr(_state, "forced", None)
    if forced is not None:
        return bool(forced)

    env = _parse_tristate(os.environ.get("HOWSO_PROGRESS"))
    if env is not None:
        return env

    cfg = _config_auto_progress(trainee)
    if cfg is not None:
        return cfg

    return sys.stdout.isatty() or _in_notebook()


@overload
def auto_progress(label_or_method: _M, /) -> _M: ...
@overload
def auto_progress(
    label_or_method: str | None = ..., /, *, indeterminate: bool = ...
) -> Callable[[_M], _M]: ...
def auto_progress(label_or_method: Any = None, /, *, indeterminate: bool = False) -> Any:
    """
    Decorate a ``Trainee`` method to opt into unified progress reporting.

    Usable bare (label inferred from the method name) or as a factory with
    an explicit label::

        @auto_progress
        def train(self, ...): ...

        @auto_progress("React (series)")
        def react_series(self, ...): ...

    At call time the decorator consults :func:`_auto_progress_enabled`.
    When disabled (env=off, non-TTY, etc.) the wrapped method behaves
    identically to the original — no extra thread, no callback, no reporter.
    When enabled it delegates to :func:`with_progress`, which honors any
    caller-supplied ``task_id`` or ``progress_callback``.

    Nested wrapped calls do not stack bars: an outer call increments a
    thread-local depth counter that inner calls see and short-circuit on.
    """
    def _decorate(method: Callable[..., Any], label: str) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _auto_progress_enabled(self):
                return method(self, *args, **kwargs)
            depth = getattr(_state, "depth", 0)
            _state.depth = depth + 1
            try:
                return with_progress(
                    label,
                    method.__get__(self, type(self)),
                    *args,
                    indeterminate=indeterminate,
                    **kwargs,
                )
            finally:
                _state.depth = depth
        wrapper._auto_progress_label = label  # type: ignore[attr-defined]
        return wrapper

    if callable(label_or_method):
        method = label_or_method
        return _decorate(method, _default_label(method.__name__))

    label = label_or_method
    def factory(method: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate(method, label or _default_label(method.__name__))
    return factory


def auto_progress_enabled(trainee: Any = None) -> bool:
    """
    Report whether an :func:`auto_progress`-decorated call would be wrapped now.

    Public read of the same decision the decorator makes for ``trainee``
    (see :func:`_auto_progress_enabled` for the precedence stack). Intended
    for downstream packages that compose their own progress sessions and
    need to honor the same force-flag / env / config / heuristic gating.

    Parameters
    ----------
    trainee : Any, optional
        The trainee whose client configuration participates in the decision.
        When omitted, the config layer is skipped and the remaining layers
        (force flag, env var, TTY/notebook heuristic) decide.

    Returns
    -------
    bool
        True when auto-progress would be enabled for the next decorated call.
    """
    return _auto_progress_enabled(trainee)


def auto_progress_forced() -> bool | None:
    """
    Return the thread-local auto-progress force flag.

    ``True``/``False`` when :func:`enable_auto_progress` /
    :func:`disable_auto_progress` / :func:`auto_progress_scope` has forced a
    state for the current thread, or ``None`` when unset. Lets downstream
    packages slot their own overrides between the force flags and the
    env-var/config layers of the precedence stack.

    Returns
    -------
    bool or None
        The forced state, or ``None`` when no force is in effect.
    """
    return getattr(_state, "forced", None)


def enable_auto_progress() -> None:
    """Force auto-progress on for the current thread until reset."""
    _state.forced = True


def disable_auto_progress() -> None:
    """Force auto-progress off for the current thread until reset."""
    _state.forced = False


def reset_auto_progress() -> None:
    """Clear any thread-local force flag and return to env/config behavior."""
    _state.forced = None


@contextmanager
def auto_progress_scope(enabled: bool = True) -> Generator[None]:
    """
    Temporarily force auto-progress on or off for the current thread.

    Restores the prior state on exit, so nested scopes behave correctly.
    """
    prev = getattr(_state, "forced", None)
    _state.forced = enabled
    try:
        yield
    finally:
        _state.forced = prev
