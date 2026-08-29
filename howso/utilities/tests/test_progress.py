# pyright: reportMissingParameterType=false
from __future__ import annotations

import builtins
from datetime import timedelta
import inspect
import io
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
import warnings

import pytest
from rich.console import Console
from rich.progress import BarColumn, Progress

from howso.client.configuration import ClientOptions, HowsoConfiguration
from howso.utilities import (
    auto_progress,
    auto_progress_enabled,
    auto_progress_forced,
    auto_progress_scope,
    auto_reporter,
    disable_auto_progress,
    enable_auto_progress,
    engine_polling_supported,
    ProgressEvent,
    reset_auto_progress,
    RichDisplayProgressReporter,
    RichNotebookProgressReporter,
    RichProgressReporter,
    SimpleProgressReporter,
    with_progress,
)
from howso.utilities.monitors import ProgressTimer
from howso.utilities.progress import (
    _auto_progress_enabled,  # pyright: ignore[reportPrivateUsage]
    _display_handle_available,  # pyright: ignore[reportPrivateUsage]
    _experimental_notebook_reporter,  # pyright: ignore[reportPrivateUsage]
    _format_duration,  # pyright: ignore[reportPrivateUsage]
    _format_eta,  # pyright: ignore[reportPrivateUsage]
    _in_notebook,  # pyright: ignore[reportPrivateUsage]
    _interactive_frontend,  # pyright: ignore[reportPrivateUsage]
    _notebook_console,  # pyright: ignore[reportPrivateUsage]
    _one_line,  # pyright: ignore[reportPrivateUsage]
    _OverwriteSafeWriter,  # pyright: ignore[reportPrivateUsage]
    _parse_tristate,  # pyright: ignore[reportPrivateUsage]
    _SolidBar,  # pyright: ignore[reportPrivateUsage]
    BAR_WIDTH,
    ETA_LABEL_MIN_WIDTH,
    NOTEBOOK_COLUMNS,
)


class _FakeClient:
    """Minimal client double exposing the subset ``with_progress`` touches."""

    def __init__(self, progress_payloads=None, library_type="mt") -> None:  # pyright: ignore[reportMissingSuperCall]
        self._payloads = list(progress_payloads or [
            {"step": 1, "total": 3, "details": "step 1"},
        ])
        # Defaults to the multi-threaded library so that engine polling is
        # permitted; single-threaded cases opt in explicitly. Pass an
        # ``Exception`` instance to make the lookup raise.
        self._library_type = library_type
        self.poll_count = 0

    def get_progress(self, trainee_id, task_id):  # noqa: ARG002
        self.poll_count += 1
        idx = min(self.poll_count - 1, len(self._payloads) - 1)
        return self._payloads[idx]

    def get_trainee_runtime(self, trainee_id):  # noqa: ARG002
        if isinstance(self._library_type, Exception):
            raise self._library_type
        return {
            "library_type": self._library_type,
            "tracing_enabled": False,
            "versions": {"trainee": "1.0.0", "amalgam": "1.0.0"},
        }


class _FakeAmalgam:
    """Amalgam double reporting a configurable concurrency type."""

    def __init__(self, concurrency: Any = b"MultiThreaded") -> None:  # pyright: ignore[reportMissingSuperCall]
        self._concurrency = concurrency

    def get_concurrency_type_string(self):
        if isinstance(self._concurrency, Exception):
            raise self._concurrency
        return self._concurrency


class _FakeLocalClient(_FakeClient):
    """Client double running an engine in-process, as the direct client does."""

    def __init__(self, concurrency: Any = b"MultiThreaded", **kwargs: Any) -> None:  # pyright: ignore[reportMissingSuperCall]
        super().__init__(**kwargs)
        self.amlg = _FakeAmalgam(concurrency)


class _FakeTrainee:
    """Bound-method host with both progress hooks available."""

    id = "fake-trainee"

    def __init__(self, client=None) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.client = client or _FakeClient()
        self.received_task_id = None
        self.received_progress_callback = None

    def cb_only(self, *, progress_callback=None):
        """Support ``progress_callback`` only (mirrors ``train``)."""
        self.received_progress_callback = progress_callback
        with ProgressTimer(2) as p:
            for _ in range(2):
                p.update(1)
                if progress_callback:
                    progress_callback(p)
        return "cb_only-done"

    def task_only(self, *, task_id=None):
        """Support ``task_id`` only (mirrors ``analyze``)."""
        self.received_task_id = task_id
        # Sleep briefly so the polling thread has time to fire.
        time.sleep(0.05)
        return "task_only-done"

    def both(self, *, task_id=None, progress_callback=None):
        """Support both hooks (mirrors ``react``)."""
        self.received_task_id = task_id
        self.received_progress_callback = progress_callback
        with ProgressTimer(2) as p:
            for _ in range(2):
                p.update(1)
                if progress_callback:
                    progress_callback(p, None)
                time.sleep(0.02)
        return "both-done"

    def neither(self):
        """Expose no progress hooks."""
        return "neither-done"


@pytest.mark.parametrize("value", ["on", "ON", "true", "True", "yes", "y", "1", 1, True])
def test_parse_tristate_truthy(value):
    assert _parse_tristate(value) is True


@pytest.mark.parametrize("value", ["off", "OFF", "false", "False", "no", "n", "0", 0, False])
def test_parse_tristate_falsy(value):
    assert _parse_tristate(value) is False


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "maybe", "??", "yes please"])
def test_parse_tristate_fallthrough(value):
    assert _parse_tristate(value) is None


def test_simple_reporter_single_source_no_indent(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=1, total=6, details="Analyzing"))
    reporter.update(ProgressEvent(source="engine", step=2, total=6, details="Computing"))
    reporter.finish(success=True, duration=timedelta(seconds=1.5))
    out = capsys.readouterr().out
    assert "Analyze" in out
    assert "  [1/6] Analyzing" in out
    assert "  [2/6] Computing" in out
    assert "    [" not in out  # no double-indent in single-source mode
    assert "Analyze complete in 0:00:01" in out


def test_simple_reporter_both_sources_engine_indented(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("React", sources=("batch", "engine"))
    reporter.update(ProgressEvent(source="batch", step=10, total=100, details="batch 1"))
    reporter.update(ProgressEvent(source="engine", step=1, total=3, details="step 1"))
    reporter.finish(success=True, duration=timedelta(seconds=2.0))
    out = capsys.readouterr().out
    assert "  [ 10/100] batch 1" in out      # batch: 2-space prefix
    assert "    [1/3] step 1" in out          # engine: 4-space prefix (nested)


def test_simple_reporter_numerator_padded_to_denominator_width(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=0, total=1999, details="batch 0"))
    reporter.update(ProgressEvent(source="batch", step=100, total=1999, details="batch 1"))
    reporter.update(ProgressEvent(source="batch", step=1999, total=1999, details="batch 6"))
    out = capsys.readouterr().out
    assert "  [   0/1999] batch 0" in out
    assert "  [ 100/1999] batch 1" in out
    assert "  [1999/1999] batch 6" in out


def test_simple_reporter_failure_marker(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("Train", sources=("batch",))
    reporter.finish(success=False, duration=timedelta(seconds=0.1))
    out = capsys.readouterr().out
    assert "Train failed in" in out


def test_simple_reporter_heartbeat_fires_when_step_stalls(capsys, monkeypatch):
    """Verify a heartbeat prints when the step is unchanged but HEARTBEAT_INTERVAL has elapsed."""
    # Shorten the heartbeat window for a fast test.
    monkeypatch.setattr("howso.utilities.progress.HEARTBEAT_INTERVAL", 0.05)
    reporter = SimpleProgressReporter()
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=3, total=6, details="Computing"))
    time.sleep(0.07)
    # Same step a second time → should print a heartbeat line containing 'elapsed'.
    reporter.update(ProgressEvent(source="engine", step=3, total=6, details="Computing"))
    reporter.finish(success=True, duration=timedelta(seconds=1.0))
    out = capsys.readouterr().out
    assert "elapsed" in out


def test_simple_reporter_unknown_total_renders_question_mark(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=0, total=0, details=""))
    out = capsys.readouterr().out
    assert "[0/?]" in out


class _RecordingStream:
    """Minimal file-like double that counts ``flush`` calls."""

    def __init__(self, *, raises: bool = False) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.flushes = 0
        self._raises = raises

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        self.flushes += 1
        if self._raises:
            raise ValueError("underlying buffer has been detached")


def _reporter_writing_to(monkeypatch, console_file) -> SimpleProgressReporter:
    """Build a reporter whose console renders into ``console_file``."""
    reporter = SimpleProgressReporter()
    monkeypatch.setattr(reporter, "_console", SimpleNamespace(file=console_file))
    return reporter


def test_flush_all_drains_console_stdout_and_stderr(monkeypatch):
    console_file, out, err = _RecordingStream(), _RecordingStream(), _RecordingStream()
    reporter = _reporter_writing_to(monkeypatch, console_file)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    reporter._flush_all()  # pyright: ignore[reportPrivateUsage]
    assert (console_file.flushes, out.flushes, err.flushes) == (1, 1, 1)


def test_flush_all_flushes_a_shared_stream_only_once(monkeypatch):
    """The console's file is normally ``sys.stdout`` itself — flush it once, not thrice."""
    shared = _RecordingStream()
    reporter = _reporter_writing_to(monkeypatch, shared)
    monkeypatch.setattr(sys, "stdout", shared)
    monkeypatch.setattr(sys, "stderr", shared)
    reporter._flush_all()  # pyright: ignore[reportPrivateUsage]
    assert shared.flushes == 1


def test_flush_all_survives_a_detached_stream(monkeypatch):
    """A stream that raises on flush must not abort the session, nor the remaining flushes."""
    broken, out, err = _RecordingStream(raises=True), _RecordingStream(), _RecordingStream()
    reporter = _reporter_writing_to(monkeypatch, broken)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    reporter._flush_all()  # pyright: ignore[reportPrivateUsage]
    assert broken.flushes == 1
    assert (out.flushes, err.flushes) == (1, 1)


@pytest.mark.parametrize(
    "reporter_cls",
    [SimpleProgressReporter, RichProgressReporter, RichNotebookProgressReporter],
)
def test_reporter_flushes_at_both_session_boundaries(reporter_cls, monkeypatch, capsys):  # noqa: ARG001
    """Pending output is drained before the first render and again after the last."""
    reporter = reporter_cls()
    flushes: list[str] = []
    monkeypatch.setattr(reporter, "_flush_all", lambda: flushes.append("flush"))
    reporter.start("Train", sources=("batch",))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert len(flushes) == 2


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Every duration renders as H:MM:SS, hours running past a day.
_DURATION = re.compile(r"\d+:\d\d:\d\d")


def _require_progress(reporter: RichProgressReporter) -> Progress:
    """
    Return a reporter's live ``Progress``, asserting it exists.

    Typed against ``RichProgressReporter`` because that is the class declaring
    ``_progress``; the notebook and display reporters inherit it, while
    ``SimpleProgressReporter`` prints lines and has none. The attribute is
    ``Progress | None`` because a session with no sources never starts one.
    Every caller here has just started a session that does, so the assertion
    narrows the type and states that precondition rather than letting the test
    fail later with an opaque ``NoneType`` error.
    """
    progress = reporter._progress
    assert progress is not None
    return progress


def _render(reporter, sources, events) -> str:
    """
    Drive a full session and return the raw bytes the console received.

    Each update is followed by an explicit ``refresh()``. Rich's live display
    is driven by a background thread at ``refresh_per_second``, so without
    this a short session can finish before the frame under test is ever
    painted — which would make every control-code assertion below vacuous.
    """
    buf = io.StringIO()
    reporter._console.file = buf  # is_terminal is unaffected by a file swap
    reporter.start("Train", sources=sources)
    for event in events:
        reporter.update(event)
        if reporter._progress is not None:
            _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=4))
    return buf.getvalue()


def _fake_kernel(monkeypatch):
    """Make rich's own ``_is_jupyter()`` believe it is inside a kernel."""
    class ZMQInteractiveShell:
        pass

    monkeypatch.setattr(builtins, "get_ipython", ZMQInteractiveShell, raising=False)


def _bar_color(frame: str) -> list[str]:
    """
    Return the first SGR color code applied to a bar glyph in ``frame``.

    A list rather than a scalar so an unstyled frame compares as ``[]`` instead
    of raising, which keeps a failure message readable.
    """
    return re.findall(r"\x1b\[([0-9;]+)m\u2501", frame)[:1]


def _overwrite_residue(raw: str) -> str:
    r"""
    Replay a repaint region and return whatever outlives the final frame.

    Front-ends implement ``\r`` as a raw-index overwrite and strip the
    erase-line code, so a frame shorter than its predecessor leaves that
    predecessor's tail on screen.

    The assertion to build on this is that the residue is *blank*, not that
    there is none — residue is unavoidable whenever frames shrink. Comparing
    ANSI-stripped text instead would pass or fail on where the byte boundary
    happens to land, which is exactly how this bug once reached a user.
    """
    region = raw.split("\n", maxsplit=1)[0]
    line = ""
    for chunk in region.split("\r"):
        line = chunk + line[len(chunk):]
    return line[len(region.split("\r")[-1]):]


def test_notebook_reporter_leaves_no_residue_when_frames_shrink():
    """
    Verify a short frame still covers the long one before it.

    Visible width is constant, but raw length is not: rich's indeterminate
    pulse spends ~980 characters on a color gradient occupying the same
    columns a determinate frame draws in ~165. Since the overwrite is by raw
    index, the shortfall would otherwise surface as literal escape-sequence
    fragments such as ``;112m``.
    """
    reporter = RichNotebookProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=("batch",))
    _require_progress(reporter).refresh()          # the long pulse frame
    for step in (1, 60, 120):             # then much shorter determinate frames
        reporter.update(ProgressEvent(source="batch", step=step, total=120, details="batch 2"))
        _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=1))
    residue = _overwrite_residue(buf.getvalue())
    assert set(residue) <= {" "}, f"visible residue: {residue!r}"


def test_notebook_reporter_padding_is_invisible():
    """The padding must add raw length only — never color, never extra columns."""
    reporter = RichNotebookProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=("batch",))
    _require_progress(reporter).refresh()
    reporter.update(ProgressEvent(source="batch", step=120, total=120, details="b"))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=1))
    for line in buf.getvalue().split("\n"):
        for chunk in line.split("\r"):
            if "\u2501" in chunk:
                # padding may extend the line, but only ever with blanks
                assert _ANSI.sub("", chunk)[NOTEBOOK_COLUMNS:].strip() == ""


@pytest.mark.parametrize("sources", [("batch",), ("engine",), ("batch", "engine")])
def test_notebook_reporter_emits_no_cursor_up(sources):
    """The whole point of the class: notebooks discard cursor motion, so never emit it."""
    events = [
        ProgressEvent(source=s, step=i, total=10, details=f"{s} {i}")
        for i, s in enumerate(sources, 1)
    ]
    out = _render(RichNotebookProgressReporter(), sources, events)
    assert "\x1b[1A" not in out
    assert "\r" in out  # it still repaints in place


def test_notebook_reporter_survives_multiline_engine_details():
    """``details`` is arbitrary engine payload; one newline would make the bar 2 lines high."""
    out = _render(RichNotebookProgressReporter(), ("batch", "engine"), [
        ProgressEvent(source="batch", step=1, total=10, details="b"),
        ProgressEvent(source="engine", step=1, total=3, details="line1\nline2\nline3 " * 20),
    ])
    assert "\x1b[1A" not in out


def test_notebook_reporter_frames_fit_the_console():
    """Front-ends overwrite by length and ignore erase-line, so a short frame leaves residue."""
    out = _render(RichNotebookProgressReporter(), ("batch", "engine"), [
        ProgressEvent(source="batch", step=1, total=10, details="b"),
        ProgressEvent(source="engine", step=9, total=10, details="a much longer detail string"),
        ProgressEvent(source="batch", step=9, total=10, details="b"),
    ])
    # A frame is a \r-delimited chunk *within* one physical line. Frames fill
    # the console width give or take a column: rich distributes flexible column
    # widths with integer rounding, and with seven columns the remainder can
    # leave one unallocated. Exact coverage is not this property's job anyway —
    # _OverwriteSafeWriter guarantees that by raw length. What matters here is
    # that no frame is *substantially* short and that any excess is blank.
    # Frames no longer fill the console: ``expand`` is off, because letting rich
    # redistribute slack is what made labels and bars land in different columns
    # from one session to the next. What still has to hold is that a frame never
    # exceeds the console width and never carries visible trailing padding.
    for line in out.split("\n"):
        for chunk in line.split("\r"):
            if "\u2501" not in chunk:
                continue
            visible = _ANSI.sub("", chunk)
            # The writer pads by *raw* length, and escapes occupy raw length
            # without occupying columns, so trailing blanks can run a little
            # past the console width. What must hold is that the content fits
            # and everything past it is blank.
            assert len(visible.rstrip()) <= NOTEBOOK_COLUMNS
            assert visible[NOTEBOOK_COLUMNS:].strip() == ""


def test_terminal_reporter_still_uses_cursor_up():
    """Characterization: the terminal reporter keeps its nested layout, cursor codes and all."""
    reporter = RichProgressReporter(console=_notebook_console())
    out = _render(reporter, ("batch", "engine"), [
        ProgressEvent(source="batch", step=1, total=10, details="b"),
        ProgressEvent(source="engine", step=1, total=3, details="e"),
    ])
    assert "\x1b[1A" in out


def test_notebook_console_flags():
    console = RichNotebookProgressReporter()._console
    assert console.is_jupyter is False
    assert console.is_terminal is True
    assert console.legacy_windows is False
    assert console.width == NOTEBOOK_COLUMNS


def test_notebook_console_overrides_rich_jupyter_detection(monkeypatch):
    """A stock Console would go Jupyter here; ours must not."""
    _fake_kernel(monkeypatch)
    from rich.console import Console  # noqa: PLC0415

    assert Console().is_jupyter is True  # the fake is doing its job
    assert RichNotebookProgressReporter()._console.is_jupyter is False


def test_notebook_reporter_emits_no_ipywidgets_warning(monkeypatch):
    """End-to-end guard: rich's Jupyter path warns and renders nothing without ipywidgets."""
    _fake_kernel(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _render(RichNotebookProgressReporter(), ("batch",), [
            ProgressEvent(source="batch", step=1, total=2, details="b"),
        ])
    assert "Train" in _ANSI.sub("", out)


def test_notebook_reporter_uses_one_task_for_both_sources():
    reporter = RichNotebookProgressReporter()
    reporter.start("Train", sources=("batch", "engine"))
    try:
        assert len(set(reporter._tasks.values())) == 1
        assert len(_require_progress(reporter).tasks) == 1
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))


def test_notebook_reporter_engine_does_not_move_the_bar():
    """``batch`` owns the bar; ``engine`` may only contribute text."""
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch", "engine"))
    try:
        reporter.update(ProgressEvent(source="batch", step=5, total=10, details="b"))
        reporter.update(ProgressEvent(source="engine", step=1, total=3, details="e"))
        task = _require_progress(reporter).tasks[0]
        assert (task.completed, task.total) == (5, 10)
        assert "engine 1/3" in task.fields["details"]
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))


def test_notebook_reporter_engine_only_drives_the_bar():
    """With no batch source, engine is promoted to owning the bar."""
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Analyze", sources=("engine",))
    try:
        reporter.update(ProgressEvent(source="engine", step=3, total=6, details="computing"))
        task = _require_progress(reporter).tasks[0]
        assert (task.completed, task.total) == (3, 6)
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))


def test_notebook_reporter_ignores_undeclared_source():
    """Same Protocol contract as the other reporters."""
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Analyze", sources=("engine",))
    try:
        reporter.update(ProgressEvent(source="batch", step=9, total=9, details="nope"))
        assert _require_progress(reporter).tasks[0].completed == 0
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))


def test_notebook_reporter_empty_sources_starts_no_live_region():
    """No sources: no bar, no FileProxy swap — but still a completion line."""
    reporter = RichNotebookProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=())
    assert reporter._progress is None
    assert not hasattr(sys.stdout, "rich_proxied_file")  # no FileProxy installed
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert "Train complete in" in _ANSI.sub("", buf.getvalue())


def test_notebook_reporter_restores_stdio():
    """A leaked FileProxy in a kernel is sticky, so finish() must always unwind it."""
    original_out, original_err = sys.stdout, sys.stderr
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch",))
    assert hasattr(sys.stdout, "rich_proxied_file")  # rich swapped it in
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert sys.stdout is original_out
    assert sys.stderr is original_err


@pytest.mark.parametrize(("raw", "expected"), [
    ("plain", "plain"),
    ("two\nlines", "two lines"),
    ("tabs\tand\r\nnewlines", "tabs and newlines"),
    ("  collapses   runs  ", "collapses runs"),
])
def test_one_line_flattens_whitespace(raw, expected):
    assert _one_line(raw) == expected


def test_one_line_truncates_with_ellipsis():
    assert _one_line("x" * 100, limit=10) == "x" * 9 + "\u2026"


def test_auto_reporter_notebook_picks_rich_notebook(monkeypatch):
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    assert type(auto_reporter()) is RichNotebookProgressReporter


def test_auto_reporter_tty_wins_over_notebook(monkeypatch):
    """Terminal IPython is a notebook by our heuristic but should keep the full layout."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    assert type(auto_reporter()) is RichProgressReporter


def test_auto_reporter_simple_env_wins_in_notebook(monkeypatch):
    """HOWSO_SIMPLE_PROGRESS remains the escape hatch back to the old behavior."""
    monkeypatch.setenv("HOWSO_SIMPLE_PROGRESS", "1")
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    assert type(auto_reporter()) is SimpleProgressReporter


class _FakeDisplayHandle:
    """Stands in for an IPython ``DisplayHandle``, recording every frame."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.frames: list[Any] = []

    def update(self, renderable: Any) -> None:
        self.frames.append(renderable)


def _fake_display(monkeypatch) -> _FakeDisplayHandle:
    """
    Install a stub ``IPython.display.display`` and return its handle.

    A stub rather than a real kernel: IPython is not a declared dependency, so
    CI has no ipykernel to render into. The handle records what would have been
    pushed, which is exactly what these tests assert on.
    """
    handle = _FakeDisplayHandle()

    def display(renderable, display_id=None):  # noqa: ARG001
        handle.frames.append(renderable)
        return handle

    module = SimpleNamespace(display=display)
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(
        display=module, get_ipython=object))
    return handle


def _rows(renderable) -> list[str]:
    """Return the visible lines of one pushed frame."""
    plain = renderable._repr_mimebundle_([], [])["text/plain"]
    return [line for line in plain.splitlines() if line.strip()]


@pytest.mark.parametrize("sources", [(), ("batch",), ("engine",), ("batch", "engine")])
def test_display_reporter_stays_inline_when_watched(monkeypatch, sources):
    """
    Verify a watched notebook never claims a display slot, even for two sources.

    A notebook merges consecutive stdout writes into one block but never merges
    display blocks, so a display group is fenced off from the lines around it by
    the notebook's padding. The stdout reporter folds the engine into the outer
    bar's details rather than paying that for a second bar.
    """
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: True)
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=sources)
    for source in sources:
        reporter.update(ProgressEvent(source=source, step=5, total=5, details="b"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert handle.frames == []
    assert "Train complete in" in _ANSI.sub("", buf.getvalue())


def test_overwrite_writer_drops_cursor_visibility_codes():
    """
    Verify the cursor show/hide codes never reach the notebook stream.

    rich emits them because this console forces ``is_terminal``, but a notebook
    has no cursor to hide. Most front-ends drop them silently; nbconvert's HTML
    export handles only SGR codes and renders the rest literally, so they show
    up as a stray ``[?25l`` above the bars.
    """
    sink = io.StringIO()
    writer = _OverwriteSafeWriter(sink)
    writer.write("\x1b[?25l")
    writer.write("\rframe")
    writer.write("\x1b[?25h")
    assert sink.getvalue() == "\rframe"


def test_notebook_reporter_emits_no_cursor_codes():
    """End-to-end: a full session leaks neither cursor code."""
    reporter = RichNotebookProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=("batch",))
    _require_progress(reporter).refresh()
    reporter.update(ProgressEvent(source="batch", step=150, total=150, details="b"))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert "\x1b[?25" not in buf.getvalue()


def test_overwrite_writer_pads_before_an_embedded_newline():
    """
    Verify padding lands on the line being padded, not the one after it.

    A frame's trailing newline is not the last thing in its write — rich
    appends a show-cursor code after it — so treating the write as one unit put
    the blanks after the newline, visibly indenting whatever came next.
    """
    sink = io.StringIO()
    writer = _OverwriteSafeWriter(sink)
    writer.write("\rTHIS-FIRST-FRAME-IS-LONG")     # sets the width to cover
    writer.write("\rshort\n\x1b[?25h")             # newline is NOT last
    first_line, _, rest = sink.getvalue().partition("\n")
    assert first_line.endswith(" ")                # padded, on its own line
    assert not rest.startswith(" ")                # and nothing spilled over


def test_overwrite_writer_pads_only_when_the_frame_shrinks():
    """A frame at least as long as its predecessor needs no padding at all."""
    sink = io.StringIO()
    writer = _OverwriteSafeWriter(sink)
    writer.write("\rshort")
    writer.write("\rmuch-longer-frame")
    assert sink.getvalue() == "\rshort\rmuch-longer-frame"


def test_notebook_reporter_keeps_the_writer_through_stop(monkeypatch):
    """
    Verify the overwrite-safe writer is still installed when Progress stops.

    ``Progress.stop()`` runs inside ``super().finish()`` and emits one last
    frame. Restoring the plain file before that call left the final frame
    un-padded against a much longer predecessor, which is the residue that
    reached a user. This asserts the ordering directly: with the quiet bar the
    last frame now happens to be the longest, so the defect no longer shows up
    in the output it produces, and a test on residue alone would not catch a
    regression here.
    """
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch",))
    seen: dict[str, str] = {}
    original = RichProgressReporter.finish

    def spy(self, **kwargs):  # noqa: ANN003
        seen["file"] = type(self._console.file).__name__
        return original(self, **kwargs)

    monkeypatch.setattr(RichProgressReporter, "finish", spy)
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert seen["file"] == "_OverwriteSafeWriter"


def test_notebook_reporter_bar_never_pulses():
    """
    Verify an unknown total renders a static track, not a pulse.

    A pulse frame spends ~980 characters on a color gradient where the
    determinate frame replacing it needs ~165, and on a carriage-return
    repaint that entire disparity has to be padded over as blanks.
    """
    reporter = RichNotebookProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=("batch",))
    _require_progress(reporter).refresh()
    reporter.update(ProgressEvent(source="batch", step=1, total=120, details="b"))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=1))
    # Measure the pulse itself, not its knock-on size: the writer pads every
    # frame up to the running maximum, so a length comparison would be masked
    # by the very padding the pulse makes necessary.
    first = next(c for c in buf.getvalue().split("\r") if "\u2501" in c)
    colors = set(re.findall(r"\x1b\[([0-9;]+)m", first))
    assert len(colors) <= 6, f"gradient detected, {len(colors)} colors"


@pytest.mark.parametrize("sources", [("batch", "engine")])
def test_display_reporter_uses_a_slot_for_every_bar(monkeypatch, sources):
    """
    Verify even a lone bar is rendered through a display slot.

    Sending a single bar to stdout instead would close the vertical gap between
    groups, but stdout repaints in place with a carriage return, and rich's
    frames shrink drastically — a ~980 character pulse frame is replaced by a
    ~165 character determinate one. The survivors begin mid-escape-sequence and
    render as literal text. A display slot replaces its content outright, so
    that cannot happen.
    """
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=sources)
    for source in sources:
        reporter.update(ProgressEvent(source=source, step=5, total=5, details="b"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert handle.frames                       # a slot was claimed
    assert len(_rows(handle.frames[-1])) == len(sources) + 1   # bars + completion


def test_two_sources_merge_onto_one_bar_when_inline(monkeypatch):
    """The engine rides in the details text rather than getting its own line."""
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: True)
    reporter = RichNotebookProgressReporter()
    sink = io.StringIO()          # keep our own handle: start() wraps the file
    reporter._console.file = sink
    reporter.start("React", sources=("batch", "engine"))
    try:
        reporter.update(ProgressEvent(source="batch", step=12, total=30, details="batch 3"))
        reporter.update(ProgressEvent(source="engine", step=1, total=3, details="reacting"))
        _require_progress(reporter).refresh()
        assert len(_require_progress(reporter).tasks) == 1
        line = _ANSI.sub("", [f for f in sink.getvalue().split("\r") if "\u2501" in f][-1])
        assert "12/30" in line          # the batch owns the bar
        assert "engine 1/3" in line     # the engine rides along
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))


def test_display_reporter_renders_both_bars(monkeypatch):
    """The headline feature: the nested layout the ANSI path had to give up."""
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch", "engine"))
    reporter.update(ProgressEvent(source="batch", step=1, total=5, details="batch 1"))
    reporter.update(ProgressEvent(source="engine", step=2, total=5, details="step 2"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    rows = _rows(handle.frames[-1])
    assert len(rows) == 3               # two bars plus the completion line
    assert "Train" in rows[0]
    assert "engine" in rows[1]          # the indented inner track
    assert "engine" not in rows[0]


def test_display_reporter_puts_completion_line_in_the_same_block(monkeypatch):
    """
    Verify the summary shares one output block with the bars.

    A notebook renders a stdout stream and an HTML display as two separate
    outputs, each with its own vertical padding, so printing the line instead
    of folding it in leaves a conspicuous gap under the bars.
    """
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=("batch", "engine"))
    reporter.update(ProgressEvent(source="batch", step=5, total=5, details="b"))
    reporter.finish(success=True, duration=timedelta(seconds=3))
    rows = _rows(handle.frames[-1])
    assert "Train complete in" in rows[-1]      # folded into the display slot
    assert "complete in" not in buf.getvalue()  # and NOT printed to stdout


def test_display_reporter_emits_no_control_codes(monkeypatch):
    """Nothing is repainted in place, so no cursor motion should appear at all."""
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=("batch", "engine"))
    reporter.update(ProgressEvent(source="batch", step=1, total=5, details="b"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    payload = "".join(f._repr_mimebundle_([], [])["text/html"] for f in handle.frames)
    assert "\x1b[" not in payload
    assert "\x1b[1A" not in buf.getvalue()   # nor on the completion line


def test_display_reporter_strips_the_pre_margin(monkeypatch):
    """
    Verify the notebook HTML carries no outer margin.

    rich emits a bare ``<pre>``, which browsers default to ``margin: 1em 0``.
    Each progress group is its own output block, so those margins stack
    between groups and show up as a large gap.
    """
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch", "engine"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert handle.frames                        # guard: not vacuously true
    for frame in handle.frames:
        bundle = frame._repr_mimebundle_([], [])
        assert bundle["text/html"].startswith('<pre style="margin:0;')
        assert "text/plain" in bundle      # the fallback must survive wrapping


def test_display_reporter_throttles_pushes(monkeypatch):
    """Events arrive far faster than a reader can follow, and each push costs HTML."""
    handle = _fake_display(monkeypatch)
    clock = iter([0.0] + [0.01 * i for i in range(1, 60)])
    monkeypatch.setattr("howso.utilities.progress.monotonic", lambda: next(clock))
    reporter = RichDisplayProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch", "engine"))
    for i in range(1, 21):   # 20 events across ~0.2s, well under 1/4s apart
        reporter.update(ProgressEvent(source="batch", step=i, total=20, details="b"))
    pushed_during_updates = len(handle.frames)
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert 0 < pushed_during_updates < 20   # throttled, but not silent


def test_display_reporter_always_pushes_the_final_frame(monkeypatch):
    """A last event inside the throttle window must not leave the bar short of 100%."""
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch", "engine"))
    reporter.update(ProgressEvent(source="batch", step=20, total=20, details="done"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert "20/20" in _rows(handle.frames[-1])[0]


def test_display_reporter_empty_sources_claims_no_slot(monkeypatch):
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Train", sources=())
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert handle.frames == []
    assert "Train complete in" in _ANSI.sub("", buf.getvalue())


def test_display_reporter_keeps_rich_stock_bar_palette():
    """
    Verify the bar keeps rich's stock styles.

    The HTML path bakes literal hex, and rich's stock styles bake to exactly
    the RGB a truecolor terminal shows. Restyling the bar would break that
    match.
    """
    bar = next(c for c in RichDisplayProgressReporter()._make_columns()
               if isinstance(c, BarColumn))
    assert (bar.style, bar.complete_style, bar.finished_style, bar.pulse_style) == (
        "bar.back", "bar.complete", "bar.finished", "bar.pulse",
    )


def test_display_handle_available_requires_a_live_shell(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: None))
    assert _display_handle_available() is False
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert _display_handle_available() is False


def test_auto_reporter_notebook_with_display_picks_display_reporter(monkeypatch):
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._display_handle_available", lambda: True)
    assert type(auto_reporter()) is RichDisplayProgressReporter


def test_auto_reporter_notebook_without_display_falls_back_to_ansi(monkeypatch):
    """A Databricks runtime that never imported IPython still gets a working bar."""
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._display_handle_available", lambda: False)
    assert type(auto_reporter()) is RichNotebookProgressReporter


def test_format_eta_drops_sub_second_precision():
    """A timedelta stringifies with microseconds, which is false precision here."""
    assert _format_eta(timedelta(seconds=83, microseconds=456789)) == (
        "est. rem.: 0:01:23"
    )


@pytest.mark.parametrize("eta", [None, timedelta(seconds=-5)])
def test_format_eta_renders_nothing_without_a_usable_estimate(eta):
    assert _format_eta(eta) == ""


def test_format_eta_shortens_the_label_on_a_narrow_console():
    """
    Verify the label degrades rather than crowding out real data.

    Measured against the longest label this module generates: at 88 columns the
    spelled-out form pushes the counter and elapsed time into ellipses, while
    the short form leaves them intact.
    """
    assert _format_eta(timedelta(seconds=83), long=True) == "est. rem.: 0:01:23"
    assert _format_eta(timedelta(seconds=83), long=False) == "ETA 0:01:23"
    # and a bar reporter picks between them from its console width
    assert RichProgressReporter(
        console=Console(width=ETA_LABEL_MIN_WIDTH)
    )._eta_text(timedelta(seconds=83)) == _format_eta(timedelta(seconds=83), long=True)
    assert RichProgressReporter(
        console=Console(width=ETA_LABEL_MIN_WIDTH - 1)
    )._eta_text(timedelta(seconds=83)) == _format_eta(timedelta(seconds=83), long=False)


def test_batch_callback_carries_an_estimate():
    """The estimate already exists on ProgressTimer; the event must carry it."""
    reporter = _RecordingReporter()
    trainee = _FakeTrainee()
    with_progress("Train", trainee.cb_only, reporter=reporter)
    batch_events = [e for e in reporter.events if e.source == "batch"]
    assert batch_events
    assert any(e.eta is not None for e in batch_events)


def test_batch_callback_withholds_the_estimate_at_tick_zero():
    """
    Verify no estimate is reported before the first tick lands.

    ``ProgressTimer.time_remaining`` divides by ``max(current_tick, 1)``, so at
    tick zero it reports roughly the entire run as still remaining — a number
    worse than showing nothing. ``_FakeTrainee.cb_only`` ticks before it fires
    the callback and so can never reach this state; this fake fires first.
    """
    class _FiresAtTickZero:
        id = "fake-trainee"

        def __init__(self) -> None:
            self.client = _FakeClient()

        def cb_only(self, *, progress_callback=None):
            # ``with_progress`` always supplies the callback; assert rather
            # than guard with ``if``, so a harness that stopped wiring it fails
            # here instead of quietly reporting no events.
            assert progress_callback is not None
            with ProgressTimer(10) as timer:
                progress_callback(timer)  # current_tick is still 0
                timer.update(1)
                progress_callback(timer)  # now there is a measurement
            return "done"

    reporter = _RecordingReporter()
    with_progress("Train", _FiresAtTickZero().cb_only, reporter=reporter)
    events = [e for e in reporter.events if e.source == "batch"]
    assert len(events) == 2
    assert events[0].step == 0
    assert events[0].eta is None      # withheld
    assert events[1].eta is not None  # reported once measurable


def test_rich_reporter_renders_the_estimate():
    reporter = RichProgressReporter(console=_notebook_console())
    out = _render(reporter, ("batch",), [
        ProgressEvent(source="batch", step=5, total=10, details="batch 2",
                      eta=timedelta(seconds=83)),
    ])
    assert _format_eta(timedelta(seconds=83)) in _ANSI.sub("", out)


def test_simple_reporter_renders_the_estimate(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=1200, total=10000,
                                  details="batch 3", eta=timedelta(seconds=83)))
    reporter.update(ProgressEvent(source="batch", step=2400, total=10000, details="batch 6"))
    out = capsys.readouterr().out
    assert f"batch 3 · {_format_eta(timedelta(seconds=83))}" in out
    assert "batch 6\n" in out          # no stray separator when there is no estimate


def test_experimental_notebook_reporter_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", raising=False)
    assert _experimental_notebook_reporter() is False
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    assert _experimental_notebook_reporter() is True
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "0")
    assert _experimental_notebook_reporter() is False


def test_interactive_frontend_false_without_ipython(monkeypatch):
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert _interactive_frontend() is False


@pytest.mark.parametrize(("shell", "expected"), [
    (SimpleNamespace(kernel=SimpleNamespace(_allow_stdin=True)), True),
    (SimpleNamespace(kernel=SimpleNamespace(_allow_stdin=False)), False),
    (SimpleNamespace(kernel=SimpleNamespace()), False),   # field omitted
    (SimpleNamespace(), False),                           # no kernel at all
    (None, False),                                        # no shell
])
def test_interactive_frontend_reads_allow_stdin(monkeypatch, shell, expected):
    """
    Verify headless execution is distinguished from a live front-end.

    ``nbclient`` — and so nbconvert and papermill — hard-codes
    ``allow_stdin = False``, while JupyterLab's client defaults it to true.
    A front-end that omits the field reads as False, which degrades to plain
    lines rather than to corrupted output.
    """
    monkeypatch.setitem(sys.modules, "IPython",
                        SimpleNamespace(get_ipython=lambda: shell))  # pyright: ignore[reportUnknownLambdaType]
    assert _interactive_frontend() is expected


def test_auto_reporter_headless_notebook_still_gets_the_display_slot(monkeypatch):
    """
    Verify an nbconvert/papermill run keeps the rich bars.

    nbclient's ``_update_display_id`` replaces an output's data in place, so
    every repaint overwrites the same output and the saved notebook keeps the
    final frame as ordinary HTML — which exports faithfully.
    """
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._display_handle_available", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: False)
    assert type(auto_reporter()) is RichDisplayProgressReporter


def test_auto_reporter_headless_without_display_uses_simple(monkeypatch):
    """
    Verify the carriage-return repaint is never chosen headlessly.

    Nothing applies it there, so each frame is committed to the document as its
    own line.
    """
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._display_handle_available", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: False)
    assert type(auto_reporter()) is SimpleProgressReporter


def test_display_reporter_uses_a_slot_headlessly_even_for_one_bar(monkeypatch):
    """A lone bar must not take the repaint path when nothing will apply it."""
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: False)
    handle = _fake_display(monkeypatch)
    reporter = RichDisplayProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=5, total=5, details="b"))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    assert reporter._inline is None
    assert handle.frames


def test_auto_reporter_notebook_with_display_but_flag_off_uses_simple(monkeypatch):
    """
    Verify the whole rich notebook branch is opt-in, not just the ANSI half.

    Rich progress in a notebook rests on undocumented front-end behavior either
    way — an in-place repaint, or the display-update protocol — so with the flag
    unset a notebook takes the path it always did.
    """
    monkeypatch.delenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._display_handle_available", lambda: True)
    assert type(auto_reporter()) is SimpleProgressReporter


def test_auto_reporter_tty_is_unaffected_by_the_flag(monkeypatch):
    """A real terminal keeps the full reporter whether or not the flag is set."""
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    for value in ("1", None):
        if value is None:
            monkeypatch.delenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", raising=False)
        else:
            monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", value)
        assert type(auto_reporter()) is RichProgressReporter


def test_auto_reporter_notebook_without_display_and_flag_off_uses_simple(monkeypatch):
    """With no display slot and no opt-in, fall back to plain lines."""
    monkeypatch.delenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._display_handle_available", lambda: False)
    assert type(auto_reporter()) is SimpleProgressReporter


class _NoHooks:
    """A trainee whose only progress hook disappears when engine polling is off."""

    id = "fake-trainee"

    def __init__(self, sleep: float = 0.0) -> None:
        self.client = _FakeClient()
        self._sleep = sleep

    def neither(self):
        if self._sleep:
            time.sleep(self._sleep)
        return "neither-done"


def test_engine_events_never_carry_an_estimate():
    """
    Verify engine work never claims a time to completion.

    Engine step durations vary enough that an estimate derived from them would
    mislead, so the field must stay empty at every step and total.
    """
    trainee = _FakeTrainee()
    reporter = _RecordingReporter()
    with_progress("React", trainee.both, reporter=reporter, polling_interval=0.01)
    engine = [e for e in reporter.events if e.source == "engine"]
    assert engine
    assert all(e.eta is None for e in engine)


def test_stepped_engine_still_fills_proportionally():
    """A known total is real information about steps, even without a time claim."""
    reporter = RichProgressReporter(console=_notebook_console())
    reporter._console.file = io.StringIO()
    reporter.start("Analyze", sources=("engine",))
    try:
        reporter.update(ProgressEvent(source="engine", step=12, total=30, details="k-fold"))
        task = _require_progress(reporter).tasks[0]
        assert (task.completed, task.total) == (12, 30)
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))


def test_indeterminate_declares_an_activity_source():
    reporter = _RecordingReporter()
    with_progress("Analyze", _NoHooks().neither, reporter=reporter, indeterminate=True)
    assert reporter.started_sources == ("activity",)


def test_without_the_flag_a_hookless_method_declares_no_sources():
    reporter = _RecordingReporter()
    with_progress("Analyze", _NoHooks().neither, reporter=reporter)
    assert reporter.started_sources == ()


def test_activity_ticker_reports_before_completion():
    """
    Verify something reaches the reporter while the call is still running.

    Two of the four reporters emit only on ``update``, so without the ticker a
    long call would stay silent until it finished — the whole point here.
    """
    reporter = _RecordingReporter()
    with_progress("Analyze", _NoHooks(sleep=0.25).neither, reporter=reporter,
                  indeterminate=True, polling_interval=0.02)
    assert [e for e in reporter.events if e.source == "activity"]


def test_activity_ticker_does_not_outlive_the_call():
    before = threading.active_count()
    with_progress("Analyze", _NoHooks(sleep=0.05).neither, reporter=_RecordingReporter(),
                  indeterminate=True, polling_interval=0.01)
    assert threading.active_count() <= before


def _indeterminate_frames(reporter, *, success: bool = True) -> list[str]:
    """Render a source-less session and return its bar frames."""
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Analyze", sources=("activity",))
    for _ in range(3):
        time.sleep(0.11)
        _require_progress(reporter).refresh()
    reporter.finish(success=success, duration=timedelta(seconds=1))
    return [f for f in buf.getvalue().split("\r") if "\u2501" in f]


@pytest.mark.parametrize(("sources", "event"), [
    (("batch",), ProgressEvent(source="batch", step=2, total=2, details="batch 1")),
    (("engine",), ProgressEvent(source="engine", step=3, total=0, details="k-fold")),
    (("activity",), None),
])
def test_every_bar_is_the_same_width(sources, event):
    """
    Verify a bar is the same length whatever the session type.

    The activity layout drops the counter and estimate columns, so a bar that
    filled the space available would absorb what they freed and render visibly
    longer than a batch bar in the same notebook. ``BarColumn.bar_width``
    governs the determinate case; the solid bar has to honor it too.
    """
    reporter = RichProgressReporter(console=_notebook_console())
    reporter._console.file = io.StringIO()
    reporter.start("React aggregate", sources=sources)
    try:
        if event is not None:
            reporter.update(event)
        _require_progress(reporter).refresh()
        frame = [f for f in reporter._console.file.getvalue().split("\r") if "\u2501" in f][-1]
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))
    assert len(re.findall(r"[\u2501\u257a\u2578]", _ANSI.sub("", frame))) == BAR_WIDTH


def test_completed_indeterminate_session_stops_spinning():
    """
    Verify the spinner stops once the session ends.

    rich blanks its spinner on ``task.finished``, which is ``completed >=
    total`` and so never true while the total is ``None``. Left alone, a
    finished ``analyze`` keeps a spinner frame beside its own completion line
    and reads as though it were still running.
    """
    frames = _indeterminate_frames(RichProgressReporter(console=_notebook_console()))
    assert _ANSI.sub("", frames[0]).strip()[:1] not in {"", "A"}   # spinning during
    assert _ANSI.sub("", frames[-1]).lstrip().startswith("Analyze")  # blank after


@pytest.mark.parametrize("eta", [
    timedelta(seconds=0),
    timedelta(microseconds=400),
    timedelta(milliseconds=999),
])
def test_spent_estimate_renders_as_nothing(eta):
    """A bar that has just filled must not advertise "est. remaining: 0:00:00"."""
    assert _format_eta(eta) == ""


def test_every_color_is_theme_mappable():
    """
    Verify nothing is painted in a colour the notebook theme cannot remap.

    Only ANSI palette indices 0-15 are themed. rich's stock track is
    ``grey23`` — a 256-cube index around #3a3a3a, which no theme touches and
    which reads as *filled* against a light background.
    """
    reporter = RichNotebookProgressReporter()
    sink = io.StringIO()
    reporter._console.file = sink
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=40, total=120,
                                  details="batch 2", eta=timedelta(seconds=83)))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=20))
    fixed = {
        code for code in re.findall(r"\x1b\[([0-9;]+)m", sink.getvalue())
        if "38" in code.split(";") and {"2", "5"} & set(code.split(";"))
    }
    assert fixed == set(), f"not theme-mappable: {sorted(fixed)}"


@pytest.mark.parametrize(("success", "final_color"), [(True, "32"), (False, "31")])
def test_stepped_bar_stays_red_until_the_session_ends(success, final_color):
    """
    Verify a full bar does not turn green while the call is still running.

    rich switches to its finished style as soon as ``completed >= total``, but
    the engine reporting its last step is not the call returning — the same
    mistake that froze the elapsed clock. A bar going green beside a still
    ticking timer says the opposite of what the timer says.
    """
    reporter = RichNotebookProgressReporter()
    sink = io.StringIO()
    reporter._console.file = sink
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=1, total=1, details=""))
    _require_progress(reporter).refresh()
    during = [f for f in sink.getvalue().split("\r") if "\u2501" in f][-1]
    assert _bar_color(during) == ["31"], "a running session must not look finished"
    reporter.finish(success=success, duration=timedelta(seconds=20))
    final = [f for f in sink.getvalue().split("\r") if "\u2501" in f][-1]
    assert _bar_color(final) == [final_color]


def test_nothing_looks_finished_while_the_session_runs():
    """
    Verify no column treats the engine's last step as the session ending.

    rich keys three separate behaviours off ``completed >= total``: it freezes
    the elapsed clock, switches the bar to its finished style, and blanks the
    spinner. All three are wrong here — the engine reporting its final step
    says nothing about when the call returns. Each was found and fixed
    separately; this covers the property they share.
    """
    reporter = RichNotebookProgressReporter()
    sink = io.StringIO()
    reporter._console.file = sink
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=1, total=1, details=""))
    progress = _require_progress(reporter)
    task = progress.tasks[0]
    assert task.finished, "rich should consider the task done — that is the trap"
    assert task.start_time is not None
    task.start_time -= 20
    progress.refresh()
    row = _ANSI.sub("", [f for f in sink.getvalue().split("\r")
                         if "\u2501" in f][-1]).split("\n")[0]
    raw = [f for f in sink.getvalue().split("\r") if "\u2501" in f][-1]
    assert row.strip()[:1] != "A", "spinner must still be turning"   # not the label
    assert _bar_color(raw) == ["31"], "bar must still read as running"
    assert "0:00:20" in row, "clock must still be counting"
    reporter.finish(success=True, duration=timedelta(seconds=20))


def test_elapsed_keeps_running_after_the_engine_reports_its_last_step():
    """
    Verify the clock tracks the session, not the engine's last step.

    rich sets ``finished_time`` the instant ``completed >= total`` and
    ``TimeElapsedColumn`` freezes on it. But the engine reporting its final
    step is not the call returning: an ``analyze`` whose engine reported 1/1
    after a second went on working for another seventeen, and the bar sat at
    ``0:00:01`` beside a completion line reading ``0:00:18``.
    """
    reporter = RichNotebookProgressReporter()
    sink = io.StringIO()
    reporter._console.file = sink
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=1, total=1, details=""))
    # Wind the start back rather than sleep: the task is now 20 seconds old
    # while the engine has already reported everything it is going to.
    task = _require_progress(reporter).tasks[0]
    assert task.start_time is not None
    task.start_time -= 20
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=20))
    # Only the bar row: the last chunk also carries the completion line, whose
    # duration comes from finish() and would satisfy this assertion on its own.
    bar_row = _ANSI.sub("", [f for f in sink.getvalue().split("\r")
                             if "\u2501" in f][-1]).split("\n")[0]
    assert "0:00:20" in bar_row, bar_row


def test_engine_bar_completes_when_the_call_succeeds():
    """
    Verify a short-reporting engine still ends on a full bar.

    The poll thread stops the moment the wrapped call returns, so the engine's
    last reported step is usually below its total. Left alone the bar sits
    part-filled — reading as still running directly above its own "complete"
    line.
    """
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=2, total=3, details=""))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=1))
    final = _ANSI.sub("", [f for f in reporter._console.file.getvalue().split("\r")
                           if "\u2501" in f][-1])
    assert "3/3" in final
    assert "2/3" not in final


def test_failed_engine_bar_stays_where_it_stopped():
    """A failure must not be dressed up as a completed run."""
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Analyze", sources=("engine",))
    reporter.update(ProgressEvent(source="engine", step=2, total=3, details=""))
    _require_progress(reporter).refresh()
    reporter.finish(success=False, duration=timedelta(seconds=1))
    final = _ANSI.sub("", [f for f in reporter._console.file.getvalue().split("\r")
                           if "\u2501" in f][-1])
    assert "2/3" in final


@pytest.mark.parametrize(("seconds", "expected"), [
    (0.987, "0:00:00"),
    (9.5, "0:00:09"),
    (83, "0:01:23"),
    (3600, "1:00:00"),
    (360000, "100:00:00"),
])
def test_durations_render_as_a_clock(seconds, expected):
    """
    Verify one duration format throughout, hours accumulating past a day.

    ``str(timedelta)`` switches to ``"4 days, 4:00:00"`` past 24 hours, which
    is 15 characters where the same value was 7 — enough to shift or clip a
    pinned column on a run of the length these routinely reach.
    """
    assert _format_duration(timedelta(seconds=seconds)) == expected


def test_completed_batch_drops_its_details_with_no_gap():
    """
    Verify the chunk index goes on completion, leaving exactly one space.

    ``batch 7`` is only the last chunk index; once the counter reads 120/120 it
    says nothing. Clearing it must not leave a hole: the details share the
    counter's column precisely because an empty column of their own would cost
    padding on both sides and show as two spaces before the elapsed time.
    """
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=120, total=120, details="batch 7"))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=8))
    final = _ANSI.sub("", [f for f in reporter._console.file.getvalue().split("\r")
                           if "\u2501" in f][-1])
    assert "batch 7" not in final
    elapsed = _DURATION.search(final)
    assert elapsed
    before = final[:elapsed.start()]
    assert len(before) - len(before.rstrip()) == 1


def test_failed_batch_keeps_its_details():
    """On a failure the last detail says where it stopped, which is worth keeping."""
    reporter = RichNotebookProgressReporter()
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=60, total=120, details="batch 4"))
    _require_progress(reporter).refresh()
    reporter.finish(success=False, duration=timedelta(seconds=8))
    final = _ANSI.sub("", [f for f in reporter._console.file.getvalue().split("\r")
                           if "\u2501" in f][-1])
    assert "batch 4" in final


def test_completed_batch_bar_drops_its_estimate():
    """The estimate is cleared at session end, not left reading zero."""
    reporter = RichProgressReporter(console=_notebook_console())
    reporter._console.file = io.StringIO()
    reporter.start("Train", sources=("batch",))
    reporter.update(ProgressEvent(source="batch", step=1000, total=2000,
                                  details="batch 3", eta=timedelta(seconds=45)))
    _require_progress(reporter).refresh()
    during = _ANSI.sub("", reporter._console.file.getvalue())
    reporter.finish(success=True, duration=timedelta(seconds=1))
    final = _ANSI.sub("", reporter._console.file.getvalue().split("\r")[-2])
    assert "est. rem.: 0:00:45" in during
    assert "remaining" not in final


def test_solid_bar_measures_itself_exactly():
    """
    Verify the indeterminate bar reports its own width.

    Without a measurement rich hands the column all the space left over; the
    bar draws short inside it and the blank remainder shoves every following
    column to the right. Only tracks with an unknown total were affected, since
    a determinate bar is rich's own ``ProgressBar``, which measures itself.
    """
    console = _notebook_console()
    bar = _SolidBar("red", BAR_WIDTH)
    options = console.options.update(width=NOTEBOOK_COLUMNS)
    measurement = bar.__rich_measure__(console, options)
    assert (measurement.minimum, measurement.maximum) == (BAR_WIDTH, BAR_WIDTH)


def test_indeterminate_row_left_aligns_its_trailing_columns():
    """A row with no counter must not leave a gulf between bar and elapsed."""
    reporter = RichProgressReporter(console=_notebook_console())
    reporter._console.file = io.StringIO()
    reporter.start("Analyze", sources=("activity",))
    try:
        reporter.update(ProgressEvent(source="activity", step=0, total=0))
        _require_progress(reporter).refresh()
        line = _ANSI.sub("", [f for f in reporter._console.file.getvalue().split("\r")
                              if "\u2501" in f][-1])
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))
    bar_end = line.rindex("\u2501") + 1
    elapsed = _DURATION.search(line)
    assert elapsed is not None
    # Exactly one space, the same as a batch row. An empty column still costs
    # its padding on both sides, so a details column that never has content
    # shows up as a second space between the bar and the elapsed time.
    assert elapsed.start() - bar_end == 1, f"gap of {elapsed.start() - bar_end} columns"


def test_indeterminate_bar_is_no_costlier_than_a_normal_one():
    """
    Verify the indeterminate bar stays cheap.

    rich's pulse colors every cell of a 20-step gradient — ~980 characters a
    frame against ~160 — and on the carriage-return path that entire gulf has
    to be padded over as blanks. A solid single-run bar holds the length flat.
    """
    reporter = RichProgressReporter(console=_notebook_console())
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Analyze", sources=("engine",))
    try:
        _require_progress(reporter).refresh()                      # total unknown
        indeterminate = [f for f in buf.getvalue().split("\r") if "\u2501" in f][-1]
        reporter.update(ProgressEvent(source="engine", step=5, total=10, details=""))
        _require_progress(reporter).refresh()                      # total known
        determinate = [f for f in buf.getvalue().split("\r") if "\u2501" in f][-1]
    finally:
        reporter.finish(success=True, duration=timedelta(seconds=1))
    # Constant length is not the property to assert — rich's pulse frames are
    # constant too, just uniformly enormous. What matters is that the
    # indeterminate frame costs no more than an ordinary one, since the gulf
    # between them is what _OverwriteSafeWriter has to pad over.
    assert len(indeterminate) < len(determinate) * 1.5, (
        f"indeterminate={len(indeterminate)} determinate={len(determinate)}"
    )


@pytest.mark.parametrize(("success", "final"), [(True, "32"), (False, "31")])
def test_indeterminate_bar_shows_state_by_color(success, final):
    """Red while pending, green once done — and a failure never turns green."""
    frames = _indeterminate_frames(
        RichProgressReporter(console=_notebook_console()), success=success)
    assert _bar_color(frames[0]) == ["31"]
    assert _bar_color(frames[-1]) == [final]


def test_indeterminate_spinner_animates():
    """
    Verify the spinner is what conveys liveness.

    The bar is deliberately static, so a frozen spinner would leave the session
    looking hung while every length and color assertion still passed.
    """
    frames = _indeterminate_frames(RichProgressReporter(console=_notebook_console()))
    glyphs = {_ANSI.sub("", f).strip()[:1] for f in frames}
    assert len(glyphs) > 1


def test_activity_track_drops_the_counter_and_estimate():
    """An activity track measures nothing, so "0/?" would be noise."""
    frames = _indeterminate_frames(RichProgressReporter(console=_notebook_console()))
    rendered = _ANSI.sub("", "".join(frames))
    assert "0/?" not in rendered
    assert _DURATION.search(rendered)      # elapsed is still shown


def test_notebook_reporter_renders_an_activity_session():
    """
    Verify the stdout reporter does not ignore an activity-only session.

    Its primary-track selection knows ``batch`` and ``engine``; without a
    fallback, ``activity`` matched neither and the reporter started no Progress
    at all, leaving the very sessions this exists for completely blank.
    """
    reporter = RichNotebookProgressReporter()
    buf = io.StringIO()
    reporter._console.file = buf
    reporter.start("Analyze", sources=("activity",))
    assert reporter._progress is not None
    reporter.update(ProgressEvent(source="activity", step=0, total=0))
    _require_progress(reporter).refresh()
    reporter.finish(success=True, duration=timedelta(seconds=1))
    frames = [f for f in buf.getvalue().split("\r") if "\u2501" in f]
    assert frames
    # The property that matters is the same one the other repaint tests assert:
    # whatever the writer leaves behind must be blank. Equal frame lengths would
    # be stricter than necessary — the closing frame is longer, which needs no
    # padding at all.
    residue = _overwrite_residue(buf.getvalue())
    assert set(residue) <= {" "}, f"visible residue: {residue!r}"


def test_simple_reporter_activity_line(capsys, monkeypatch):
    monkeypatch.setattr("howso.utilities.progress.HEARTBEAT_INTERVAL", 0.0)
    reporter = SimpleProgressReporter()
    reporter.start("Analyze", sources=("activity",))
    reporter.update(ProgressEvent(source="activity", step=0, total=0))
    reporter.finish(success=True, duration=timedelta(seconds=1))
    out = capsys.readouterr().out
    assert "elapsed" in out
    assert "0/?" not in out


def test_auto_reporter_databricks_picks_rich_notebook(monkeypatch):
    """Databricks lost its carve-out and is now treated as an ordinary notebook."""
    monkeypatch.setenv("HOWSO_EXPERIMENTAL_NOTEBOOK_REPORTER", "1")
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "13.3.x-scala2.12")
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._interactive_frontend", lambda: True)
    assert type(auto_reporter()) is RichNotebookProgressReporter


def test_auto_reporter_simple_env_set(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setenv("HOWSO_SIMPLE_PROGRESS", "1")
    assert isinstance(auto_reporter(), SimpleProgressReporter)


def test_auto_reporter_simple_env_zero_is_falsy(monkeypatch):
    """Verify ``HOWSO_SIMPLE_PROGRESS=0`` does NOT force the simple reporter."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setenv("HOWSO_SIMPLE_PROGRESS", "0")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert type(auto_reporter()) is RichProgressReporter


def test_auto_reporter_non_tty_falls_back_to_simple(monkeypatch):
    """Neither a tty nor a notebook — a pipe or CI log — still gets plain lines."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    assert type(auto_reporter()) is SimpleProgressReporter


def test_auto_reporter_tty_picks_rich(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert type(auto_reporter()) is RichProgressReporter


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Ensure each test starts with no thread-local force and no env var."""
    reset_auto_progress()
    monkeypatch.delenv("HOWSO_PROGRESS", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_ENGINE_PROGRESS", raising=False)
    yield
    reset_auto_progress()


def _trainee_with_cfg(cfg_value):
    """Build a fake trainee whose client.configuration.auto_progress returns cfg_value."""
    class Cfg:
        auto_progress = cfg_value
    class Client:
        configuration = Cfg()
    class T:
        id = "x"
        client = Client()
    return T()


def test_gating_re_entrancy_short_circuits_even_when_forced(monkeypatch):  # noqa: ARG001
    """Verify the re-entrancy guard runs before the forced flag — nested calls never stack."""
    t = _trainee_with_cfg(None)
    enable_auto_progress()
    # Simulate "we're already inside one wrapped call".
    from howso.utilities.progress import _state  # pyright: ignore[reportPrivateUsage]  # noqa: PLC0415
    _state.depth = 1
    try:
        assert _auto_progress_enabled(t) is False
    finally:
        _state.depth = 0


def test_gating_forced_on(monkeypatch):
    t = _trainee_with_cfg(None)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    enable_auto_progress()
    assert _auto_progress_enabled(t) is True


def test_gating_forced_off_overrides_env_on(monkeypatch):
    t = _trainee_with_cfg(None)
    monkeypatch.setenv("HOWSO_PROGRESS", "on")
    disable_auto_progress()
    assert _auto_progress_enabled(t) is False


def test_gating_env_overrides_config(monkeypatch):
    t = _trainee_with_cfg("off")
    monkeypatch.setenv("HOWSO_PROGRESS", "on")
    assert _auto_progress_enabled(t) is True


def test_gating_config_when_env_unset(monkeypatch):
    t = _trainee_with_cfg("on")
    monkeypatch.delenv("HOWSO_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert _auto_progress_enabled(t) is True


def test_gating_tty_heuristic_when_nothing_set(monkeypatch):
    t = _trainee_with_cfg(None)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    assert _auto_progress_enabled(t) is True


def test_gating_notebook_heuristic_when_nothing_set(monkeypatch):
    t = _trainee_with_cfg(None)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: True)
    assert _auto_progress_enabled(t) is True


def test_gating_off_when_no_signals(monkeypatch):
    t = _trainee_with_cfg(None)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    assert _auto_progress_enabled(t) is False


@pytest.mark.parametrize("env_value", ["1", "yes", "y", "TRUE"])
def test_gating_env_var_truthy_vocabulary(monkeypatch, env_value):
    t = _trainee_with_cfg(None)
    monkeypatch.setenv("HOWSO_PROGRESS", env_value)
    assert _auto_progress_enabled(t) is True


@pytest.mark.parametrize("env_value", ["0", "no", "n", "FALSE"])
def test_gating_env_var_falsy_vocabulary(monkeypatch, env_value):
    t = _trainee_with_cfg(None)
    monkeypatch.setenv("HOWSO_PROGRESS", env_value)
    assert _auto_progress_enabled(t) is False


def test_gating_env_var_garbage_falls_through(monkeypatch):
    t = _trainee_with_cfg("on")  # config says on
    monkeypatch.setenv("HOWSO_PROGRESS", "maybe")  # env unrecognized
    assert _auto_progress_enabled(t) is True  # config wins


def test_auto_progress_forced_reflects_force_flags():
    assert auto_progress_forced() is None
    enable_auto_progress()
    assert auto_progress_forced() is True
    disable_auto_progress()
    assert auto_progress_forced() is False
    reset_auto_progress()
    assert auto_progress_forced() is None


def test_auto_progress_forced_tracks_scope():
    with auto_progress_scope(False):
        assert auto_progress_forced() is False
        with auto_progress_scope(True):
            assert auto_progress_forced() is True
        assert auto_progress_forced() is False
    assert auto_progress_forced() is None


def test_auto_progress_enabled_matches_private_gate(monkeypatch):
    """Verify the public accessor mirrors the decorator's gating decision."""
    t = _trainee_with_cfg("on")
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    assert auto_progress_enabled(t) is _auto_progress_enabled(t) is True
    monkeypatch.setenv("HOWSO_PROGRESS", "off")
    assert auto_progress_enabled(t) is _auto_progress_enabled(t) is False


def test_auto_progress_enabled_without_trainee(monkeypatch):
    """Verify the trainee argument is optional — config layer is skipped."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert auto_progress_enabled() is True
    disable_auto_progress()
    assert auto_progress_enabled() is False


def test_auto_progress_scope_restores_prior_state(monkeypatch):
    t = _trainee_with_cfg(None)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    enable_auto_progress()  # explicitly force on
    with auto_progress_scope(False):
        assert _auto_progress_enabled(t) is False
    # Forced-on state is restored after the scope.
    assert _auto_progress_enabled(t) is True


class _RecordingReporter:
    """Capture every event sent to the reporter for inspection in assertions."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.events = []
        self.started_sources = None
        self.finished_success = None
        self.label = None
        self.finished_duration = None

    def start(self, label, *, sources):
        self.label = label
        self.started_sources = sources

    def update(self, event):
        self.events.append(event)

    def finish(self, *, success, duration):
        self.finished_success = success
        self.finished_duration = duration


def test_with_progress_cb_only_method_wires_batch_source():
    t = _FakeTrainee()
    r = _RecordingReporter()
    result = with_progress("CB", t.cb_only, reporter=r)
    assert result == "cb_only-done"
    assert r.started_sources == ("batch",)
    assert {e.source for e in r.events} == {"batch"}
    assert any(e.step == 2 and e.total == 2 for e in r.events)
    assert r.finished_success is True


def test_with_progress_task_only_method_wires_engine_source():
    t = _FakeTrainee()
    r = _RecordingReporter()
    result = with_progress("Task", t.task_only, reporter=r, polling_interval=0.01)
    assert result == "task_only-done"
    assert r.started_sources == ("engine",)
    assert t.received_task_id is not None  # decorator injected a UUID
    assert {e.source for e in r.events} == {"engine"}
    assert r.finished_success is True


def test_with_progress_both_method_wires_both_sources():
    t = _FakeTrainee()
    r = _RecordingReporter()
    result = with_progress("Both", t.both, reporter=r, polling_interval=0.01)
    assert result == "both-done"
    assert set(r.started_sources or ()) == {"batch", "engine"}
    sources_seen = {e.source for e in r.events}
    assert "batch" in sources_seen


@pytest.mark.parametrize(("library_type", "expected"), [
    ("mt", True),
    (" mt ", True),
    # A multi-threaded variant stays supported, mirroring how "st-omp" exists.
    ("mt-omp", True),
    # The value is sometimes carried in Amalgam library postfix form.
    ("-mt", True),
    (" -mt ", True),
    ("st", False),
    # The OpenMP build is single-threaded despite advertising several threads.
    ("st-omp", False),
    ("-st", False),
    ("-st-omp", False),
    (None, False),
    # An unrecognized value is not multi-threaded just because it starts "mt".
    ("mtx", False),
    ("mt_single", False),
    ("", False),
])
def test_engine_polling_supported_library_type(library_type, expected):
    client = _FakeClient(library_type=library_type)
    assert engine_polling_supported(client, "fake-trainee") is expected


@pytest.mark.parametrize(("concurrency", "expected"), [
    (b"MultiThreaded", True),
    (b"SingleThreaded", False),
    (b"SingleThreaded+OpenMP", False),
    ("MultiThreaded", True),          # a str return is understood too
    (b"  MultiThreaded  ", True),
])
def test_engine_polling_supported_uses_in_process_library(concurrency, expected):
    """A client with an in-process library is asked directly, not via the runtime."""
    client = _FakeLocalClient(concurrency)
    assert engine_polling_supported(client, "fake-trainee") is expected


def test_engine_polling_supported_prefers_library_over_reported_type():
    """``library_type`` guesses "mt" for a postfix-less path; the library knows better."""
    client = _FakeLocalClient(b"SingleThreaded", library_type="mt")
    assert engine_polling_supported(client, "fake-trainee") is False


def test_engine_polling_supported_library_call_raises_is_fail_closed():
    client = _FakeLocalClient(OSError("no symbol"))
    assert engine_polling_supported(client, "fake-trainee") is False


def test_with_progress_in_process_single_threaded_skips_engine_source():
    t = _FakeTrainee(_FakeLocalClient(b"SingleThreaded"))
    r = _RecordingReporter()
    assert with_progress("Task", t.task_only, reporter=r, polling_interval=0.01) == "task_only-done"
    assert r.started_sources == ()
    assert t.client.poll_count == 0


def test_engine_polling_supported_runtime_object_attribute():
    """A client returning an object rather than a mapping is still understood."""
    class Runtime:
        library_type = "mt"

    class Client:
        def get_trainee_runtime(self, trainee_id):  # noqa: ARG002
            return Runtime()

    assert engine_polling_supported(Client(), "fake-trainee") is True


def test_engine_polling_supported_runtime_missing_library_type():
    class Client:
        def get_trainee_runtime(self, trainee_id):  # noqa: ARG002
            return {"tracing_enabled": False}

    assert engine_polling_supported(Client(), "fake-trainee") is False


def test_engine_polling_supported_runtime_raises_is_fail_closed():
    client = _FakeClient(library_type=RuntimeError("boom"))  # pyright: ignore[reportArgumentType]
    assert engine_polling_supported(client, "fake-trainee") is False


def test_engine_polling_supported_client_without_runtime_method():
    """A client that cannot answer must not be assumed safe."""
    assert engine_polling_supported(object(), "fake-trainee") is False


@pytest.mark.parametrize("trainee_id", [None, ""])
def test_engine_polling_supported_requires_trainee_id(trainee_id):
    assert engine_polling_supported(_FakeClient(), trainee_id) is False


def test_engine_polling_supported_requires_client():
    assert engine_polling_supported(None, "fake-trainee") is False


@pytest.mark.parametrize("env_value", ["off", "0", "no", "n", "false", "FALSE"])
def test_engine_polling_supported_env_off_overrides_multithreaded(monkeypatch, env_value):
    monkeypatch.setenv("HOWSO_ENGINE_PROGRESS", env_value)
    assert engine_polling_supported(_FakeClient(library_type="mt"), "fake-trainee") is False


@pytest.mark.parametrize("env_value", ["on", "1", "yes", "y", "true", "TRUE"])
def test_engine_polling_supported_env_on_cannot_force_single_threaded(monkeypatch, env_value):
    """The override is one-way: nothing may re-enable a poll that kills the process."""
    monkeypatch.setenv("HOWSO_ENGINE_PROGRESS", env_value)
    assert engine_polling_supported(_FakeClient(library_type="st"), "fake-trainee") is False


def test_engine_polling_supported_env_on_leaves_multithreaded_enabled(monkeypatch):
    monkeypatch.setenv("HOWSO_ENGINE_PROGRESS", "on")
    assert engine_polling_supported(_FakeClient(library_type="mt"), "fake-trainee") is True


@pytest.mark.parametrize(("library_type", "expected"), [("mt", True), ("st", False)])
def test_engine_polling_supported_env_garbage_falls_through(monkeypatch, library_type, expected):
    monkeypatch.setenv("HOWSO_ENGINE_PROGRESS", "maybe")
    assert engine_polling_supported(_FakeClient(library_type=library_type), "fake-trainee") is expected


@pytest.mark.parametrize("library_type", ["st", "st-omp"])
def test_with_progress_single_threaded_skips_engine_source(library_type):
    """An engine-only method degrades to a bare session rather than polling."""
    t = _FakeTrainee(_FakeClient(library_type=library_type))
    r = _RecordingReporter()
    result = with_progress("Task", t.task_only, reporter=r, polling_interval=0.01)
    assert result == "task_only-done"
    assert r.started_sources == ()
    assert t.received_task_id is None  # no task_id injected
    assert r.events == []
    assert t.client.poll_count == 0  # the poll thread never started
    assert r.finished_success is True


def test_with_progress_single_threaded_keeps_batch_source():
    """``train``/``react`` keep their Python-side bar on a single-threaded engine."""
    t = _FakeTrainee(_FakeClient(library_type="st"))
    r = _RecordingReporter()
    result = with_progress("Both", t.both, reporter=r, polling_interval=0.01)
    assert result == "both-done"
    assert r.started_sources == ("batch",)
    assert t.received_task_id is None
    assert {e.source for e in r.events} == {"batch"}
    assert t.client.poll_count == 0


def test_with_progress_runtime_lookup_failure_skips_engine_source():
    t = _FakeTrainee(_FakeClient(library_type=OSError("unreachable")))  # pyright: ignore[reportArgumentType]
    r = _RecordingReporter()
    result = with_progress("Task", t.task_only, reporter=r, polling_interval=0.01)
    assert result == "task_only-done"
    assert r.started_sources == ()
    assert t.client.poll_count == 0


def test_with_progress_engine_env_off_leaves_batch_source(monkeypatch):
    monkeypatch.setenv("HOWSO_ENGINE_PROGRESS", "off")
    t = _FakeTrainee(_FakeClient(library_type="mt"))
    r = _RecordingReporter()
    result = with_progress("Both", t.both, reporter=r, polling_interval=0.01)
    assert result == "both-done"
    assert r.started_sources == ("batch",)
    assert t.client.poll_count == 0


def test_simple_reporter_empty_sources_prints_label_and_completion(capsys):
    reporter = SimpleProgressReporter()
    reporter.start("Analyze", sources=())
    reporter.update(ProgressEvent(source="engine", step=1, total=6, details="Analyzing"))
    reporter.finish(success=True, duration=timedelta(seconds=1.5))
    out = capsys.readouterr().out
    assert "Analyze" in out
    assert "Analyze complete in 0:00:01" in out
    assert "[" not in out  # the undeclared source produced no track line


def test_rich_reporter_empty_sources_completes_lifecycle(capsys):
    reporter = RichProgressReporter()
    reporter.start("Analyze", sources=())
    reporter.update(ProgressEvent(source="engine", step=1, total=6, details="Analyzing"))
    reporter.finish(success=True, duration=timedelta(seconds=1.5))
    assert "Analyze complete in" in capsys.readouterr().out


def test_with_progress_neither_method_still_runs():
    t = _FakeTrainee()
    r = _RecordingReporter()
    result = with_progress("Neither", t.neither, reporter=r)
    assert result == "neither-done"
    assert r.started_sources == ()
    assert r.events == []
    assert r.finished_success is True


def test_with_progress_honors_caller_supplied_task_id():
    t = _FakeTrainee()
    r = _RecordingReporter()
    with_progress("Task", t.task_only, reporter=r, polling_interval=0.01, task_id="user")
    assert t.received_task_id == "user"  # not overwritten by a fresh UUID


def test_with_progress_honors_caller_supplied_callback():
    t = _FakeTrainee()
    r = _RecordingReporter()
    captured = []
    def my_cb(p, *a, **k):  # noqa: ANN002, ANN003
        captured.append(p.current_tick)
    with_progress("CB", t.cb_only, reporter=r, progress_callback=my_cb)
    assert captured == [1, 2]
    # Reporter saw no batch events because its callback wasn't wired.
    assert all(e.source != "batch" for e in r.events)


def test_with_progress_propagates_exceptions_and_marks_failure():
    class T2(_FakeTrainee):
        def boom(self, *, task_id=None):  # noqa: ARG002
            raise RuntimeError("kapow")
    t = T2()
    r = _RecordingReporter()
    with pytest.raises(RuntimeError, match="kapow"):
        with_progress("Boom", t.boom, reporter=r, polling_interval=0.01)
    assert r.finished_success is False


def test_decorator_preserves_metadata():
    @auto_progress
    def my_method(self: Any, x: int) -> int:  # noqa: ARG001
        """Add five."""
        return x + 5

    assert my_method.__name__ == "my_method"
    assert "Add five" in (my_method.__doc__ or "")
    assert my_method.__wrapped__.__name__ == "my_method"  # pyright: ignore[reportFunctionMemberAccess]
    sig = inspect.signature(my_method)
    assert list(sig.parameters) == ["self", "x"]


def test_decorator_factory_form_uses_explicit_label():
    @auto_progress("Custom Label")
    def m(self):  # noqa: ARG001
        return 1
    assert getattr(m, "_auto_progress_label", None) == "Custom Label"


def test_decorator_bare_form_derives_label_from_method_name():
    @auto_progress
    def react_series_stationary(self):  # noqa: ARG001
        return 1
    assert getattr(react_series_stationary, "_auto_progress_label", None) == (
        "React series stationary"
    )


def test_decorator_passes_through_when_disabled(monkeypatch):
    """Verify the original method is called directly when gating returns False."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    calls = []
    class T(_FakeTrainee):
        @auto_progress("Cb")
        def cb_only(self, *, progress_callback=None):  # pyright: ignore[reportIncompatibleMethodOverride]
            calls.append(progress_callback)
            return "ok"
    t = T()
    assert t.cb_only() == "ok"
    # No instrumentation wired: callback the method saw is None.
    assert calls == [None]


def test_decorator_wires_progress_when_enabled(monkeypatch):  # noqa: ARG001
    """Verify the method's callback is the with_progress wrapper when gating returns True."""
    enable_auto_progress()
    class T(_FakeTrainee):
        @auto_progress("Cb")
        def cb_only(self, *, progress_callback=None):
            # The wrapper we received should be callable, not None.
            assert callable(progress_callback)
            return super().cb_only(progress_callback=progress_callback)
    t = T()
    assert t.cb_only() == "cb_only-done"


def test_decorator_nested_calls_do_not_stack(capsys, monkeypatch):  # noqa: ARG001
    """Verify an inner wrapped call does NOT spawn its own bar inside an outer wrapped call."""
    enable_auto_progress()
    class T(_FakeTrainee):
        @auto_progress("Inner")
        def inner(self, *, progress_callback=None):
            if progress_callback:
                with ProgressTimer(1) as p:
                    p.update(1)
                    progress_callback(p)
            return "inner"
        @auto_progress("Outer")
        def outer(self, *, progress_callback=None):  # noqa: ARG002
            return self.inner()  # nested decorated call
    t = T()
    assert t.outer() == "inner"
    out = capsys.readouterr().out
    # Outer label appears; Inner label must NOT (re-entrancy guard).
    assert "Outer" in out
    assert "Inner" not in out


@pytest.mark.parametrize(("name", "label"), [
    ("train", "Train"),
    ("analyze", "Analyze"),
    ("react", "React"),
    ("react_series", "React Series"),
    ("react_series_stationary", "React Series (stationary)"),
    ("react_aggregate", "React aggregate"),
    ("react_group", "React group"),
    ("react_into_features", "React into features"),
    ("impute", "Impute"),
])
def test_trainee_methods_decorated_with_expected_labels(name, label):
    from howso.engine import Trainee  # noqa: PLC0415
    method = getattr(Trainee, name)
    assert getattr(method, "_auto_progress_label", None) == label
    # functools.wraps preserves original signature for with_progress's
    # signature introspection to still work.
    assert "self" in inspect.signature(method).parameters


def test_predict_is_not_decorated():
    """Verify ``predict`` is not wrapped — it has no progress hooks."""
    from howso.engine import Trainee  # noqa: PLC0415
    assert not hasattr(Trainee.predict, "_auto_progress_label")


def test_client_options_auto_progress_default():
    assert ClientOptions({}).auto_progress == "auto"


def test_client_options_auto_progress_explicit_on():
    assert ClientOptions({"auto_progress": "on"}).auto_progress == "on"


def test_client_options_auto_progress_case_insensitive():
    assert ClientOptions({"auto_progress": "OFF"}).auto_progress == "off"


def test_client_options_auto_progress_none_safe():
    # YAML can deserialize a bare key with no value to None — must not crash.
    assert ClientOptions({"auto_progress": None}).auto_progress == "auto"


def test_howso_configuration_passthrough_quoted(tmp_path):
    """Verify quoted ``"on"`` survives the YAML round-trip and reads back as ``"on"``."""
    yaml_path = tmp_path / "howso.yml"
    yaml_path.write_text('Howso:\n  auto_progress: "on"\n')
    cfg = HowsoConfiguration(config_path=yaml_path)
    assert cfg.auto_progress == "on"


def test_howso_configuration_yaml_bool_resolves_to_enabled(tmp_path):
    """Verify bare ``on`` parses as a YAML bool and still resolves to "enabled"."""
    yaml_path = tmp_path / "howso.yml"
    yaml_path.write_text("Howso:\n  auto_progress: on\n")
    cfg = HowsoConfiguration(config_path=yaml_path)
    # YAML 1.1 coerces bare ``on`` to True → property stringifies to ``"true"``.
    # What matters is that _parse_tristate ultimately resolves it to True.
    assert _parse_tristate(cfg.auto_progress) is True


def test_howso_configuration_yaml_bool_off_resolves_to_disabled(tmp_path):
    yaml_path = tmp_path / "howso.yml"
    yaml_path.write_text("Howso:\n  auto_progress: off\n")
    cfg = HowsoConfiguration(config_path=yaml_path)
    assert _parse_tristate(cfg.auto_progress) is False


def test_in_notebook_true_for_databricks(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "13.3.x")
    assert _in_notebook() is True


def test_in_notebook_false_without_ipython(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "IPython", None)
    # When sys.modules["IPython"] is None, get_ipython lookup short-circuits.
    assert _in_notebook() is False
