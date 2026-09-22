"""Tests for the Windows core parking and CPU visibility checks.

The core parking check only does real work on Windows, so these tests simulate
that platform rather than skipping everywhere else.
"""
import types

import pytest

from howso.utilities import installation_verification as iv

# Windows priority-class flag values, as psutil exposes them there. Their
# numeric values deliberately do not follow their ordering.
NORMAL_PRIORITY = 32
IDLE_PRIORITY = 64
HIGH_PRIORITY = 128
ABOVE_NORMAL_PRIORITY = 32768
BELOW_NORMAL_PRIORITY = 16384

CANNOT_TEST = "cannot be tested from here"
PARKING_FOUND = "Core parking appears to be enabled"
AFFINITY_SHRANK = "was allowed only"


def _fake_psutil(*, physical=16, nice=NORMAL_PRIORITY, nice_error=None):
    """Stand in for psutil as it behaves on Windows."""
    class _Process:
        def nice(self, *_args: object):
            if nice_error is not None:
                raise nice_error
            return nice

    return types.SimpleNamespace(
        IDLE_PRIORITY_CLASS=IDLE_PRIORITY,
        BELOW_NORMAL_PRIORITY_CLASS=BELOW_NORMAL_PRIORITY,
        cpu_count=lambda logical=True: physical * 2 if logical else physical,
        Process=_Process,
    )


def _simulate_windows(monkeypatch, *, normal=16.0, lowered=16.0,
                      nice=NORMAL_PRIORITY, nice_error=None,
                      measure_error=None, physical=16,
                      normal_affinity=None, low_affinity=None,
                      probe_failed=False):
    """Patch the module so the check runs its Windows path deterministically.

    `normal` and `lowered` are the effective parallelism each arm of the probe
    should appear to measure; the affinity arguments are the CPU lists each
    arm reports.
    """
    monkeypatch.setattr(iv.sys, "platform", "win32")
    monkeypatch.setattr(
        iv, "psutil",
        _fake_psutil(physical=physical, nice=nice, nice_error=nice_error))

    def fake_normal(_workers):
        if measure_error is not None:
            raise measure_error
        return normal

    monkeypatch.setattr(iv, "_measure_effective_parallelism", fake_normal)
    monkeypatch.setattr(iv, "_cpu_affinity", lambda: normal_affinity)
    monkeypatch.setattr(
        iv, "_run_low_priority_probe",
        lambda *_a, **_k: None if probe_failed else (lowered, low_affinity))
    return iv.InstallationCheckRegistry()


# --- check_core_parking --------------------------------------------------


def test_passes_off_windows(monkeypatch):
    """The check is a no-op on every other platform."""
    monkeypatch.setattr(iv.sys, "platform", "linux")
    registry = iv.InstallationCheckRegistry()
    assert iv.check_core_parking(registry=registry) == (iv.Status.OK, "")


@pytest.mark.parametrize("priority", [BELOW_NORMAL_PRIORITY, IDLE_PRIORITY])
def test_warns_when_invoked_below_normal(monkeypatch, priority):
    """Started below Normal, the comparison is meaningless, so say so.

    This is the case where the probe cannot distinguish parked cores from the
    priority it was handed, so it must not report a result either way.
    """
    registry = _simulate_windows(monkeypatch, nice=priority)
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert CANNOT_TEST in msg
    assert "Normal priority or higher" in msg
    assert PARKING_FOUND not in msg


@pytest.mark.parametrize("priority",
                         [NORMAL_PRIORITY, ABOVE_NORMAL_PRIORITY, HIGH_PRIORITY])
def test_proceeds_at_normal_priority_or_better(monkeypatch, priority):
    """Normal and above are all eligible to run the comparison."""
    registry = _simulate_windows(monkeypatch, nice=priority,
                                 normal=16.0, lowered=15.5)
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.OK
    assert CANNOT_TEST not in msg


def test_equal_throughput_passes(monkeypatch):
    """A lower-priority process that keeps its compute is healthy."""
    registry = _simulate_windows(monkeypatch, normal=15.8, lowered=15.6)
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.OK
    assert "15.8x" in msg
    assert "15.6x" in msg


def test_lower_priority_losing_compute_warns(monkeypatch):
    """The customer's failure: the low-priority arm collapses."""
    registry = _simulate_windows(monkeypatch, normal=16.0, lowered=2.0)
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert PARKING_FOUND in msg
    assert "16.0x" in msg
    assert "2.0x" in msg
    assert "scheduled task" in msg


def test_core_parking_is_a_warning_not_an_error(monkeypatch):
    """Detecting parking must not fail the run."""
    registry = _simulate_windows(monkeypatch, normal=16.0, lowered=1.0)
    status, _ = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert status not in (iv.Status.ERROR, iv.Status.CRITICAL)


@pytest.mark.parametrize(("delta", "expected"), [
    (0.05, iv.Status.OK),
    (-0.05, iv.Status.WARNING),
])
def test_threshold_boundary(monkeypatch, delta, expected):
    """The verdict flips either side of CORE_PARKING_MIN_RATIO."""
    normal = 16.0
    lowered = normal * (iv.CORE_PARKING_MIN_RATIO + delta)
    registry = _simulate_windows(monkeypatch, normal=normal, lowered=lowered)
    status, _ = iv.check_core_parking(registry=registry)
    assert status is expected


def test_single_core_machine_passes(monkeypatch):
    """There is no parallelism to measure on one core."""
    registry = _simulate_windows(monkeypatch, physical=1)
    assert iv.check_core_parking(registry=registry) == (iv.Status.OK, "")


def test_priority_lookup_failure_warns(monkeypatch):
    """An unreadable priority means the check cannot vouch for a result."""
    registry = _simulate_windows(monkeypatch, nice_error=OSError("denied"))
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert "Unable to determine" in msg


def test_normal_measurement_failure_warns(monkeypatch):
    """A failed probe degrades to a warning rather than raising.

    `run_checks` does not guard against a check raising, so a check that
    raises would abort the whole verification run.
    """
    registry = _simulate_windows(monkeypatch, measure_error=OSError("nope"))
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert "Unable to measure available CPU throughput" in msg


def test_low_priority_probe_failure_warns(monkeypatch):
    """If the child process cannot report, say the check did not run."""
    registry = _simulate_windows(monkeypatch, probe_failed=True)
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert "lower-priority process" in msg
    assert PARKING_FOUND not in msg


def test_shrunken_affinity_warns(monkeypatch):
    """Fewer CPUs offered to the lower-priority process is a warning.

    Throughput alone is healthy here, so only the affinity signal fires.
    """
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=list(range(16)), low_affinity=list(range(4)))
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert AFFINITY_SHRANK in msg
    assert "4 of the 16 CPUs" in msg
    assert PARKING_FOUND not in msg


def test_identical_affinity_passes(monkeypatch):
    """Matching CPU sets are not reported."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=list(range(16)), low_affinity=list(range(16)))
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.OK
    assert AFFINITY_SHRANK not in msg


def test_reordered_affinity_is_not_a_mismatch(monkeypatch):
    """The same CPUs in a different order are the same CPUs."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=[3, 1, 2, 0], low_affinity=[0, 1, 2, 3])
    status, _ = iv.check_core_parking(registry=registry)
    assert status is iv.Status.OK


@pytest.mark.parametrize(("normal_affinity", "low_affinity"), [
    (None, [0, 1]),
    ([0, 1], None),
    (None, None),
])
def test_unavailable_affinity_is_not_a_mismatch(monkeypatch, normal_affinity,
                                                low_affinity):
    """Where affinity cannot be read there is nothing to compare."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=normal_affinity, low_affinity=low_affinity)
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.OK
    assert AFFINITY_SHRANK not in msg


def test_both_signals_are_reported_together(monkeypatch):
    """A machine that both restricts CPUs and parks cores says so once."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=2.0,
        normal_affinity=list(range(16)), low_affinity=list(range(2)))
    status, msg = iv.check_core_parking(registry=registry)
    assert status is iv.Status.WARNING
    assert AFFINITY_SHRANK in msg
    assert PARKING_FOUND in msg


def test_affinity_is_not_read_off_windows(monkeypatch):
    """`cpu_affinity` does not exist on macOS, so the helper must not call it."""
    monkeypatch.setattr(iv.sys, "platform", "darwin")
    assert iv._cpu_affinity() is None


@pytest.mark.parametrize("lowered", [2.0, 16.0])
def test_messages_are_console_safe(monkeypatch, lowered):
    """Messages must survive a non-UTF-8 console and rich markup."""
    registry = _simulate_windows(monkeypatch, normal=16.0, lowered=lowered)
    _, msg = iv.check_core_parking(registry=registry)
    assert msg.isascii()
    assert "[" not in msg
    assert "]" not in msg


def test_below_normal_detection_is_empty_off_windows(monkeypatch):
    """The helper is safe to call anywhere, despite Windows-only constants."""
    monkeypatch.setattr(iv.sys, "platform", "linux")
    assert iv._is_below_normal_priority() is None


def test_measure_throughput_runs_real_threads():
    """Exercise the real probe: threads, barrier and hashing all cooperate."""
    buffer = b"\xa5" * (1 << 16)
    assert iv._measure_throughput(1, 0.05, buffer) > 0
    assert iv._measure_throughput(2, 0.05, buffer) > 0


def test_check_is_registered():
    """The check runs as part of a default verification."""
    registry = iv.InstallationCheckRegistry()
    assert "Python: Core parking" in [c.name for c in registry._checks]


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
