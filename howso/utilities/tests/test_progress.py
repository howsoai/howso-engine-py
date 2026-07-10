"""Tests for ``howso.utilities.progress``."""
from __future__ import annotations

from datetime import timedelta
import inspect
import time
from typing import Any

import pytest

from howso.client.configuration import ClientOptions, HowsoConfiguration
from howso.utilities import (
    auto_progress,
    auto_progress_enabled,
    auto_progress_forced,
    auto_progress_scope,
    auto_reporter,
    disable_auto_progress,
    enable_auto_progress,
    ProgressEvent,
    reset_auto_progress,
    RichProgressReporter,
    SimpleProgressReporter,
    with_progress,
)
from howso.utilities.monitors import ProgressTimer
from howso.utilities.progress import (
    _auto_progress_enabled,  # pyright: ignore[reportPrivateUsage]
    _in_notebook,  # pyright: ignore[reportPrivateUsage]
    _parse_tristate,  # pyright: ignore[reportPrivateUsage]
)


class _FakeClient:
    """Minimal client double exposing the subset ``with_progress`` touches."""

    def __init__(self, progress_payloads=None):  # pyright: ignore[reportMissingSuperCall]
        self._payloads = list(progress_payloads or [
            {"step": 1, "total": 3, "details": "step 1"},
        ])
        self.poll_count = 0

    def get_progress(self, trainee_id, task_id):  # noqa: ARG002
        self.poll_count += 1
        idx = min(self.poll_count - 1, len(self._payloads) - 1)
        return self._payloads[idx]


class _FakeTrainee:
    """Bound-method host with both progress hooks available."""

    id = "fake-trainee"

    def __init__(self, client=None):  # pyright: ignore[reportMissingSuperCall]
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
    assert "Analyze complete in 0:00:01.500000" in out


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


def test_auto_reporter_databricks_prefers_simple(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "13.3.x-scala2.12")
    assert isinstance(auto_reporter(), SimpleProgressReporter)


def test_auto_reporter_simple_env_set(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setenv("HOWSO_SIMPLE_PROGRESS", "1")
    assert isinstance(auto_reporter(), SimpleProgressReporter)


def test_auto_reporter_simple_env_zero_is_falsy(monkeypatch):
    """Verify ``HOWSO_SIMPLE_PROGRESS=0`` does NOT force the simple reporter."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setenv("HOWSO_SIMPLE_PROGRESS", "0")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(auto_reporter(), RichProgressReporter)


def test_auto_reporter_non_tty_falls_back_to_simple(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert isinstance(auto_reporter(), SimpleProgressReporter)


def test_auto_reporter_tty_picks_rich(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(auto_reporter(), RichProgressReporter)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Ensure each test starts with no thread-local force and no env var."""
    reset_auto_progress()
    monkeypatch.delenv("HOWSO_PROGRESS", raising=False)
    monkeypatch.delenv("HOWSO_SIMPLE_PROGRESS", raising=False)
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
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
    from howso.utilities.progress import _state  # pyright: ignore[reportPrivateUsage]
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

    def __init__(self):  # pyright: ignore[reportMissingSuperCall]
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
    def my_cb(p, *a, **k):
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
    assert my_method.__wrapped__.__name__ == "my_method"
    sig = inspect.signature(my_method)
    assert list(sig.parameters) == ["self", "x"]


def test_decorator_factory_form_uses_explicit_label():
    @auto_progress("Custom Label")
    def m(self):  # noqa: ARG001
        return 1
    assert m._auto_progress_label == "Custom Label"


def test_decorator_bare_form_derives_label_from_method_name():
    @auto_progress
    def react_series_stationary(self):  # noqa: ARG001
        return 1
    assert react_series_stationary._auto_progress_label == "React series stationary"


def test_decorator_passes_through_when_disabled(monkeypatch):
    """Verify the original method is called directly when gating returns False."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("howso.utilities.progress._in_notebook", lambda: False)
    calls = []
    class T(_FakeTrainee):
        @auto_progress("Cb")
        def cb_only(self, *, progress_callback=None):
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


@pytest.mark.parametrize("name,label", [
    ("train", "Train"),
    ("analyze", "Analyze"),
    ("react", "React"),
    ("react_series", "React (series)"),
    ("react_series_stationary", "React (stationary)"),
    ("react_aggregate", "React aggregate"),
    ("react_group", "React group"),
    ("react_into_features", "React into features"),
    ("impute", "Impute"),
])
def test_trainee_methods_decorated_with_expected_labels(name, label):
    from howso.engine import Trainee
    method = getattr(Trainee, name)
    assert getattr(method, "_auto_progress_label", None) == label
    # functools.wraps preserves original signature for with_progress's
    # signature introspection to still work.
    assert "self" in inspect.signature(method).parameters


def test_predict_is_not_decorated():
    """Verify ``predict`` is not wrapped — it has no progress hooks."""
    from howso.engine import Trainee
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
