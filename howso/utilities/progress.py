"""
Unified long-running task progress for Howso ``Trainee`` methods.

Howso surfaces two distinct progress signals:

* **Engine-side** — methods that accept ``task_id`` cooperate with
  ``client.get_progress(trainee_id, task_id)``, which can be polled from
  another thread for step/total updates emitted by the Amalgam engine.
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

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from functools import wraps
import inspect
import os
import sys
import threading
from time import monotonic
from typing import Any, Literal, overload, Protocol
from uuid import uuid4

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from howso.utilities.monitors import ProgressTimer

__all__ = [
    "ProgressEvent",
    "ProgressReporter",
    "RichProgressReporter",
    "SimpleProgressReporter",
    "auto_progress",
    "auto_progress_scope",
    "auto_reporter",
    "disable_auto_progress",
    "enable_auto_progress",
    "reset_auto_progress",
    "with_progress",
]

ProgressSource = Literal["engine", "batch"]

_NO_ONGOING_TASK = "There is no currently ongoing task matching the specified task_id."

# Databricks notebook cells can disconnect if no output is emitted for ~30s.
# A heartbeat well under that window keeps the cell alive during long batches.
HEARTBEAT_INTERVAL = float(os.environ.get("HOWSO_HEARTBEAT_INTERVAL", "15.0"))


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

    extras: dict[str, Any] = field(default_factory=dict)
    """Source-specific extras (e.g. batch response payload). Reserved for callers."""


class ProgressReporter(Protocol):
    """Sink for :class:`ProgressEvent` updates produced by :func:`with_progress`."""

    def start(self, label: str, *, sources: tuple[ProgressSource, ...]) -> None:
        """Begin a reporting session with the given label and known sources."""
        ...

    def update(self, event: ProgressEvent) -> None:
        """Apply a single progress event."""
        ...

    def finish(self, *, success: bool, duration: timedelta) -> None:
        """End the reporting session."""
        ...


class RichProgressReporter:
    """
    Rich-based reporter that renders one bar per progress source.

    When a method exposes both ``progress_callback`` and ``task_id``, two
    bars are shown — the outer one tracks Python batches, the inner one
    tracks engine-reported steps within the current batch.

    Parameters
    ----------
    console : Console, optional
        Console to render into. Defaults to a fresh :class:`rich.console.Console`.
    transient : bool, default True
        When ``True``, the progress bars are cleared once the session
        finishes, leaving only the final completion line.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        console: Console | None = None,
        transient: bool = True,
    ) -> None:
        """Initialize the reporter."""
        self._console = console or Console()
        self._transient = transient
        self._progress: Progress | None = None
        self._tasks: dict[ProgressSource, TaskID] = {}
        self._label: str = ""

    def start(self, label: str, *, sources: tuple[ProgressSource, ...]) -> None:
        """
        Begin a reporting session and add one bar per progress source.

        Parameters
        ----------
        label : str
            Short description shown on the batch (outer) bar.
        sources : tuple of ProgressSource
            Which progress sources will emit events; one bar is created for
            each, in the given order.

        Returns
        -------
        None
        """
        self._label = label
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[details]}"),
            TimeElapsedColumn(),
            console=self._console,
            transient=self._transient,
        )
        self._progress.start()
        # When both sources are present, label the outer (batch) bar with the
        # method name and the inner (engine) bar with an indented hint so the
        # nesting reads visually. When engine is the only source (e.g.
        # ``analyze``), use the method label directly — no orphan indent.
        both = "batch" in sources and "engine" in sources
        fallback = label or "Working"
        descriptions: dict[ProgressSource, str] = {
            "batch": fallback,
            "engine": "  engine" if both else fallback,
        }
        for source in sources:
            self._tasks[source] = self._progress.add_task(
                descriptions.get(source, fallback),
                total=None,
                details="",
            )

    def update(self, event: ProgressEvent) -> None:
        """
        Apply a single progress event to its corresponding bar.

        Events for an unknown source, or events arriving before
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
        self._progress.update(
            self._tasks[event.source],
            completed=event.step,
            total=event.total or None,
            details=event.details,
        )

    def finish(self, *, success: bool, duration: timedelta) -> None:
        """
        Tear down the live renderer and print a final completion line.

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
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._tasks.clear()
        marker = "[green]✓[/green]" if success else "[red]✗[/red]"
        status = "complete" if success else "failed"
        if self._label:
            self._console.print(f"{marker} {self._label} {status} in {duration}")


class SimpleProgressReporter:
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

    def start(self, label: str, *, sources: tuple[ProgressSource, ...]) -> None:
        """
        Begin a reporting session and print the session header.

        Parameters
        ----------
        label : str
            Short description printed as the header line.
        sources : tuple of ProgressSource
            Which progress sources will emit events; per-source step and
            heartbeat tracking is initialized for each.

        Returns
        -------
        None
        """
        self._label = label
        self._start_time = monotonic()
        self._finished = False
        self._last_step = dict.fromkeys(sources, -1)
        self._last_output = dict.fromkeys(sources, 0.0)
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
        now = monotonic()
        prefix = self._prefixes.get(event.source, "")
        total = event.total or "?"
        width = len(str(total))
        if event.step != self._last_step.get(event.source, -1):
            self._console.print(
                f"{prefix}[dim]\\[{event.step:>{width}}/{total}][/dim] {event.details}"
            )
            self._last_step[event.source] = event.step
            self._last_output[event.source] = now
        elif now - self._last_output.get(event.source, 0.0) >= HEARTBEAT_INTERVAL:
            elapsed = timedelta(seconds=int(now - self._start_time))
            self._console.print(
                f"{prefix}[dim]\\[{event.step:>{width}}/{total}] · {elapsed} elapsed[/dim]"
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
        marker = "[green]✓[/green]" if success else "[red]✗[/red]"
        status = "complete" if success else "failed"
        if self._label:
            self._console.print(f"{marker} {self._label} {status} in {duration}")


def auto_reporter(*, console: Console | None = None) -> ProgressReporter:
    """
    Choose the reporter that best fits the current environment.

    A simple line-printing reporter is selected when running under Databricks
    (which can drop live-rendered output), when ``HOWSO_SIMPLE_PROGRESS`` is
    set to a truthy value (``1``/``on``/``true``/``yes``), or when stdout is
    not a tty. Otherwise rich's live renderer is used.
    """
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return SimpleProgressReporter(console=console)
    if _parse_tristate(os.environ.get("HOWSO_SIMPLE_PROGRESS")) is True:
        return SimpleProgressReporter(console=console)
    if not sys.stdout.isatty():
        return SimpleProgressReporter(console=console)
    return RichProgressReporter(console=console)


def _supports_param(bound_func: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(bound_func)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters


def with_progress(
    label: str,
    bound_func: Callable[..., Any],
    /,
    *args: Any,
    reporter: ProgressReporter | None = None,
    polling_interval: float = 1.0,
    **kwargs: Any,
) -> Any:
    """
    Invoke ``bound_func`` with unified progress reporting.

    The function inspects ``bound_func`` for the two progress hooks Howso
    methods may expose:

    * ``task_id`` — a fresh UUID is supplied and a background thread polls
      ``trainee.client.get_progress`` while the call runs.
    * ``progress_callback`` — a wrapper translates each
      :class:`ProgressTimer` tick into a :class:`ProgressEvent`.

    Whichever hooks are present are wired into ``reporter``. If neither is
    present, ``bound_func`` is still invoked and a completion line is printed.

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

    trainee = getattr(bound_func, "__self__", None)
    client = getattr(trainee, "client", None) if trainee is not None else None
    trainee_id = getattr(trainee, "id", None) if trainee is not None else None

    # Engine polling is only useful when we can actually reach get_progress.
    has_task_id = has_task_id and client is not None and trainee_id is not None

    sources: list[ProgressSource] = []
    if has_batch_cb:
        sources.append("batch")
    if has_task_id:
        sources.append("engine")

    if has_batch_cb:
        def _batch_cb(progress: ProgressTimer, *_extra: Any, **__: Any) -> None:
            reporter.update(ProgressEvent(
                source="batch",
                step=progress.current_tick,
                total=progress.total_ticks,
                details=f"batch {progress.update_count}",
            ))
        kwargs["progress_callback"] = _batch_cb

    stop_event = threading.Event()
    poll_thread: threading.Thread | None = None

    if has_task_id:
        # Deferred import: howso.client pulls in howso.utilities at import time,
        # so we can't import it at module scope without a circular dependency.
        from howso.client.exceptions import HowsoError

        task_id = str(uuid4())
        kwargs["task_id"] = task_id

        def _poll() -> None:
            while not stop_event.is_set():
                try:
                    p = client.get_progress(trainee_id, task_id)  # pyright: ignore[reportOptionalMemberAccess]
                except HowsoError as exc:
                    # Between batches there may be no live task; ignore that
                    # specific error and keep polling. Any other engine error
                    # means progress can't be reported — stop quietly rather
                    # than killing this daemon thread with a traceback on
                    # stderr (the wrapped call itself is unaffected).
                    if _NO_ONGOING_TASK not in str(exc):
                        return
                except Exception:
                    # Progress is best-effort; never let the poller thread die
                    # noisily and clutter an otherwise-successful call.
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

    reporter.start(label, sources=tuple(sources))
    success = False
    try:
        if poll_thread is not None:
            poll_thread.start()
        result = bound_func(*args, **kwargs)
        success = True
        return result
    finally:
        stop_event.set()
        if poll_thread is not None:
            with suppress(RuntimeError):
                poll_thread.join(timeout=max(polling_interval * 2, 2.0))
        duration = timedelta(seconds=monotonic() - start_time)
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


def _config_auto_progress(trainee: Any) -> bool | None:
    """Resolve the ``auto_progress`` setting from the trainee's client configuration."""
    cfg = getattr(getattr(trainee, "client", None), "configuration", None)
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
def auto_progress(label_or_method: Callable[..., Any], /) -> Callable[..., Any]: ...
@overload
def auto_progress(label_or_method: str | None = ..., /) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def auto_progress(label_or_method=None, /):
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
def auto_progress_scope(enabled: bool = True) -> Iterator[None]:
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
