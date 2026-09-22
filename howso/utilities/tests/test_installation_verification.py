"""Tests for the Windows CPU availability check.

The check only does real work on Windows, so these tests simulate that platform
rather than skipping everywhere else. The behavior that matters is that the
check reports *lost compute*, not merely a reduced process priority.
"""
import types

import pytest

from howso.utilities import installation_verification as iv

# Windows priority-class flag values, as psutil exposes them there. Their
# numeric values deliberately do not follow their ordering.
NORMAL_PRIORITY = 32
IDLE_PRIORITY = 64
BELOW_NORMAL_PRIORITY = 16384

PRIORITY_NOTE = "priority, which is the usual trigger"
PARKING_HINT = "core parking"


def _fake_psutil(*, physical=16, nice=NORMAL_PRIORITY, nice_error=None):
    """Stand in for psutil as it behaves on Windows."""
    class _Process:
        def nice(self):
            if nice_error is not None:
                raise nice_error
            return nice

    return types.SimpleNamespace(
        IDLE_PRIORITY_CLASS=IDLE_PRIORITY,
        BELOW_NORMAL_PRIORITY_CLASS=BELOW_NORMAL_PRIORITY,
        cpu_count=lambda logical=True: physical * 2 if logical else physical,
        Process=_Process,
    )


def _simulate_windows(monkeypatch, *, effective=1.0, physical=16,
                      nice=NORMAL_PRIORITY, nice_error=None,
                      measure_error=None):
    """Patch the module so the check runs its Windows path deterministically.

    `effective` is the parallel speedup the probe should appear to measure.
    """
    monkeypatch.setattr(iv.sys, "platform", "win32")
    monkeypatch.setattr(
        iv, "psutil",
        _fake_psutil(physical=physical, nice=nice, nice_error=nice_error))

    baseline = 1_000
    results = iter([baseline, round(baseline * effective)])

    def fake_measure(*_args: object, **_kwargs: object) -> int:
        if measure_error is not None:
            raise measure_error
        return next(results)

    monkeypatch.setattr(iv, "_measure_throughput", fake_measure)
    return iv.InstallationCheckRegistry()


def test_passes_off_windows(monkeypatch):
    """The check is a no-op on every other platform."""
    monkeypatch.setattr(iv.sys, "platform", "linux")
    registry = iv.InstallationCheckRegistry()
    assert iv.check_cpu_availability(registry=registry) == (iv.Status.OK, "")


def test_healthy_machine_passes(monkeypatch):
    """Near-ideal parallel throughput is not reported."""
    registry = _simulate_windows(monkeypatch, effective=15.2, physical=16)
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.OK
    assert msg == ""


def test_limited_compute_warns(monkeypatch):
    """Throughput far below the core count is reported, and names the cause."""
    registry = _simulate_windows(monkeypatch, effective=2.0, physical=16)
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.WARNING
    assert PARKING_HINT in msg
    assert "2.0x" in msg
    assert "16" in msg


def test_reduced_priority_alone_does_not_warn(monkeypatch):
    """A low-priority process that still gets its compute is fine.

    This is the whole point of the check: priority is only a problem when it
    actually costs us cores.
    """
    registry = _simulate_windows(monkeypatch, effective=15.2, physical=16,
                                 nice=BELOW_NORMAL_PRIORITY)
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.OK
    assert msg == ""


def test_lost_compute_warns_even_at_normal_priority(monkeypatch):
    """Compute loss is reported whatever the priority, and without the note."""
    registry = _simulate_windows(monkeypatch, effective=2.0, physical=16,
                                 nice=NORMAL_PRIORITY)
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.WARNING
    assert PRIORITY_NOTE not in msg


@pytest.mark.parametrize("priority", [BELOW_NORMAL_PRIORITY, IDLE_PRIORITY])
def test_reduced_priority_is_reported_as_context(monkeypatch, priority):
    """When compute is lost, a reduced priority is called out as the likely cause."""
    registry = _simulate_windows(monkeypatch, effective=2.0, physical=16,
                                 nice=priority)
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.WARNING
    assert PRIORITY_NOTE in msg


@pytest.mark.parametrize(("delta", "expected"), [
    (0.1, iv.Status.OK),
    (-0.1, iv.Status.WARNING),
])
def test_threshold_boundary(monkeypatch, delta, expected):
    """The verdict flips either side of CPU_PROBE_MIN_EFFICIENCY."""
    workers = 16
    effective = workers * iv.CPU_PROBE_MIN_EFFICIENCY + delta
    registry = _simulate_windows(monkeypatch, effective=effective,
                                 physical=workers)
    status, _ = iv.check_cpu_availability(registry=registry)
    assert status is expected


def test_worker_count_is_capped(monkeypatch):
    """A very wide machine is probed at the cap, not at its full core count.

    Windows confines threads to one processor group on machines with more than
    64 logical processors, so probing every core would misreport there.
    """
    huge = iv.CPU_PROBE_MAX_WORKERS * 8
    registry = _simulate_windows(
        monkeypatch, effective=iv.CPU_PROBE_MAX_WORKERS, physical=huge)
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.OK
    assert msg == ""


def test_single_core_machine_passes(monkeypatch):
    """There is no parallelism to measure on one core."""
    registry = _simulate_windows(monkeypatch, effective=1.0, physical=1)
    assert iv.check_cpu_availability(registry=registry) == (iv.Status.OK, "")


def test_measurement_failure_does_not_raise(monkeypatch):
    """A failed probe degrades to a warning.

    `run_checks` does not guard against a check raising, so a check that
    raises would abort the whole verification run.
    """
    registry = _simulate_windows(monkeypatch, measure_error=OSError("nope"))
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.WARNING
    assert "Unable to measure" in msg


def test_priority_lookup_failure_is_tolerated(monkeypatch):
    """The throughput verdict still stands if the priority cannot be read."""
    registry = _simulate_windows(monkeypatch, effective=2.0, physical=16,
                                 nice_error=OSError("access denied"))
    status, msg = iv.check_cpu_availability(registry=registry)
    assert status is iv.Status.WARNING
    assert PRIORITY_NOTE not in msg


def test_warning_message_is_console_safe(monkeypatch):
    """The message must survive a non-UTF-8 Windows console and rich markup."""
    registry = _simulate_windows(monkeypatch, effective=2.0, physical=16,
                                 nice=BELOW_NORMAL_PRIORITY)
    _, msg = iv.check_cpu_availability(registry=registry)
    assert msg.isascii()
    assert "[" not in msg
    assert "]" not in msg


def test_priority_note_is_empty_off_windows(monkeypatch):
    """The helper is safe to call anywhere, despite Windows-only constants."""
    monkeypatch.setattr(iv.sys, "platform", "linux")
    registry = iv.InstallationCheckRegistry()
    assert iv._priority_note(registry) == ""


def test_measure_throughput_runs_real_threads():
    """Exercise the real probe: threads, barrier and hashing all cooperate."""
    buffer = b"\xa5" * (1 << 16)
    assert iv._measure_throughput(1, 0.05, buffer) > 0
    assert iv._measure_throughput(2, 0.05, buffer) > 0


def test_check_is_registered():
    """The check runs as part of a default verification."""
    registry = iv.InstallationCheckRegistry()
    assert "Python: CPU availability" in [c.name for c in registry._checks]


# --- check_visible_cores -------------------------------------------------


def _registry_with_engine(threads=None, *, remote=False, error=None):
    """Build a registry whose cached client reports `threads` Engine threads.

    A Howso Platform client has no in-process Engine, and so no `amlg`.
    """
    class _Amlg:
        def get_max_num_threads(self):
            if error is not None:
                raise error
            return threads

    client = types.SimpleNamespace()
    if not remote:
        client.amlg = _Amlg()
    registry = iv.InstallationCheckRegistry()
    registry._client = client
    return registry


def _patch_cpu_counts(monkeypatch, n_logical, n_physical=None, error=None):
    """Make psutil report a chosen CPU topology."""
    def cpu_count(logical=True):
        if error is not None:
            raise error
        if logical:
            return n_logical
        return n_logical if n_physical is None else n_physical

    monkeypatch.setattr(iv, "psutil", types.SimpleNamespace(cpu_count=cpu_count))


def test_visible_cores_agreement_is_informational(monkeypatch):
    """Matching counts report the number without flagging a problem."""
    _patch_cpu_counts(monkeypatch, 16)
    status, msg = iv.check_visible_cores(registry=_registry_with_engine(16))
    assert status is iv.Status.OK
    assert "16" in msg
    assert "agrees" in msg


def test_visible_cores_reports_physical_when_it_differs(monkeypatch):
    """A hyperthreaded host shows both numbers."""
    _patch_cpu_counts(monkeypatch, 32, 16)
    status, msg = iv.check_visible_cores(registry=_registry_with_engine(32))
    assert status is iv.Status.OK
    assert "32 logical" in msg
    assert "16 physical" in msg


def test_visible_cores_disagreement_warns(monkeypatch):
    """A mismatch between Python and the Engine is worth a warning."""
    _patch_cpu_counts(monkeypatch, 16)
    status, msg = iv.check_visible_cores(registry=_registry_with_engine(8))
    assert status is iv.Status.WARNING
    assert "16" in msg
    assert "8" in msg
    assert "virtual machines" in msg


def test_visible_cores_skips_comparison_for_remote_engine(monkeypatch):
    """A Platform client runs the Engine elsewhere, so counts are not compared."""
    _patch_cpu_counts(monkeypatch, 16)
    registry = _registry_with_engine(remote=True)
    status, msg = iv.check_visible_cores(registry=registry)
    assert status is iv.Status.OK
    assert "not compared" in msg


def test_visible_cores_tolerates_automatic_thread_count(monkeypatch):
    """Amalgam reports zero when it sizes its own pool; that is not a mismatch."""
    _patch_cpu_counts(monkeypatch, 16)
    status, msg = iv.check_visible_cores(registry=_registry_with_engine(0))
    assert status is iv.Status.OK
    assert "automatically" in msg


def test_visible_cores_tolerates_engine_error(monkeypatch):
    """An unreadable Engine count still reports what Python sees."""
    _patch_cpu_counts(monkeypatch, 16)
    registry = _registry_with_engine(error=RuntimeError("boom"))
    status, msg = iv.check_visible_cores(registry=registry)
    assert status is iv.Status.OK
    assert "16" in msg


@pytest.mark.parametrize(("n_logical", "error"), [
    (None, None),
    (0, None),
    (None, OSError("nope")),
])
def test_visible_cores_warns_when_undeterminable(monkeypatch, n_logical, error):
    """An unknown CPU count is itself worth reporting."""
    _patch_cpu_counts(monkeypatch, n_logical, error=error)
    status, msg = iv.check_visible_cores(registry=_registry_with_engine(16))
    assert status is iv.Status.WARNING
    assert "Unable to determine" in msg


def test_visible_cores_message_is_console_safe(monkeypatch):
    """The message must survive a non-UTF-8 console and rich markup."""
    _patch_cpu_counts(monkeypatch, 16)
    _, msg = iv.check_visible_cores(registry=_registry_with_engine(8))
    assert msg.isascii()
    assert "[" not in msg
    assert "]" not in msg


def test_visible_cores_is_registered():
    """The check runs for any client type."""
    registry = iv.InstallationCheckRegistry()
    iv.configure(registry)
    checks = {c.name: c for c in registry._checks}
    assert "Howso Client: Visible CPUs" in checks
    assert checks["Howso Client: Visible CPUs"].client_required == "AbstractHowsoClient"
