"""Tests for the low-priority CPU access and CPU visibility checks.

The low-priority comparison only does real work on Windows, so these tests
simulate that platform rather than skipping everywhere else.
"""
import importlib
import pkgutil
import subprocess
import sys
import types

import pytest

from howso.utilities import installation_verification as iv
from howso.utilities.installation_verification import (
    _constants,
    _output,
    _probes,
    _types,
    checks_cpu,
    registry as _registry,
)
from howso.utilities.installation_verification.main import configure


def _iv_modules():
    """Every module in the package, so a patch lands where a name is used."""
    return [iv, *(importlib.import_module(f"{iv.__name__}.{info.name}")
                  for info in pkgutil.iter_modules(iv.__path__)
                  if info.name != "__main__")]


_IV_MODULES = _iv_modules()


def _patch_iv(monkeypatch, name, value):
    """Patch `name` in every package module that binds it.

    The package is split across modules, so each looks a name up in its own
    globals. Patching only the package would leave the real one in place
    wherever it is actually used, and the test would quietly prove nothing.
    """
    targets = [m for m in _IV_MODULES if hasattr(m, name)]
    assert targets, f"{name!r} is not bound anywhere in the package"
    for mod in targets:
        monkeypatch.setattr(mod, name, value)

# Windows priority-class flag values, as psutil exposes them there. Their
# numeric values deliberately do not follow their ordering.
NORMAL_PRIORITY = 32
IDLE_PRIORITY = 64
HIGH_PRIORITY = 128
ABOVE_NORMAL_PRIORITY = 32768
BELOW_NORMAL_PRIORITY = 16384

CANNOT_TEST = "cannot be tested from here"
STARVED_FOUND = "is given less compute on this machine"
AFFINITY_SHRANK = "was allowed only"

#: The product name carries a trademark sign, which the rest of this module
#: already prints and which `run_checks` reconfigures stdout to handle on
#: Windows. Nothing else non-ASCII belongs in a message.
TRADEMARK = "™"


def _assert_console_safe(msg):
    """Assert a message survives a Windows console and rich's markup parser."""
    assert msg.replace(TRADEMARK, "").isascii(), msg
    assert "[" not in msg, msg
    assert "]" not in msg, msg


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
    monkeypatch.setattr(sys, "platform", "win32")
    _patch_iv(monkeypatch, "psutil",
        _fake_psutil(physical=physical, nice=nice, nice_error=nice_error))

    def fake_normal(_workers):
        if measure_error is not None:
            raise measure_error
        return normal

    _patch_iv(monkeypatch, "_measure_effective_parallelism", fake_normal)
    _patch_iv(monkeypatch, "_cpu_affinity", lambda: normal_affinity)
    _patch_iv(monkeypatch, "_run_low_priority_probe",
        lambda *_a, **_k: None if probe_failed else (lowered, low_affinity))
    return _registry.InstallationCheckRegistry()


# --- _console_safe -------------------------------------------------------


def _patch_console(monkeypatch, encoding, *, legacy=False):
    """Pretend rich is writing to a console with the given capabilities."""
    _patch_iv(monkeypatch, "get_console",
        lambda: types.SimpleNamespace(encoding=encoding, legacy_windows=legacy))


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_trademark_kept_where_console_supports_it(monkeypatch, encoding):
    """A capable console shows the real sign."""
    _patch_console(monkeypatch, encoding)
    assert _output._console_safe(f"Howso Engine{TRADEMARK}") == f"Howso Engine{TRADEMARK}"


@pytest.mark.parametrize("encoding", ["cp437", "ascii", "cp932"])
def test_trademark_downgraded_where_encoding_cannot_hold_it(monkeypatch, encoding):
    """cp437 is a common Windows console code page and cannot encode it."""
    _patch_console(monkeypatch, encoding)
    assert _output._console_safe(f"Howso Engine{TRADEMARK}") == "Howso Engine(tm)"


def test_trademark_downgraded_on_legacy_windows_console(monkeypatch):
    """The old console renders it badly even when the encoding accepts it."""
    _patch_console(monkeypatch, "utf-8", legacy=True)
    assert _output._console_safe(f"Howso{TRADEMARK}") == "Howso(tm)"


def test_trademark_downgraded_when_console_cannot_be_read(monkeypatch):
    """An unreadable console falls back rather than raising."""
    def boom():
        raise RuntimeError("no console")

    _patch_iv(monkeypatch, "get_console", boom)
    assert _output._console_safe(f"Howso{TRADEMARK}") == "Howso(tm)"


def test_console_safe_leaves_other_text_alone(monkeypatch):
    """Text without the sign is returned untouched, console never consulted."""
    def boom():
        raise AssertionError("should not be consulted")

    _patch_iv(monkeypatch, "get_console", boom)
    assert _output._console_safe("nothing to do here") == "nothing to do here"


def test_iv_print_downgrades_the_trademark(monkeypatch, capsys):
    """The downgrade happens at the output funnel, covering every message."""
    _patch_console(monkeypatch, "cp437")
    _output.iv_print(f"You are ready to use Howso{TRADEMARK}!")
    assert "Howso(tm)!" in capsys.readouterr().out


# --- check_low_priority_compute --------------------------------------------------


def test_passes_off_windows(monkeypatch):
    """The check is a no-op on every other platform."""
    monkeypatch.setattr(sys, "platform", "linux")
    registry = _registry.InstallationCheckRegistry()
    assert checks_cpu.check_low_priority_compute(registry=registry) == (_types.Status.OK, "")


@pytest.mark.parametrize("priority", [BELOW_NORMAL_PRIORITY, IDLE_PRIORITY])
def test_warns_when_invoked_below_normal(monkeypatch, priority):
    """Started below Normal, the comparison is meaningless, so say so.

    This is the case where the probe cannot distinguish parked cores from the
    priority it was handed, so it must not report a result either way.
    """
    registry = _simulate_windows(monkeypatch, nice=priority)
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert CANNOT_TEST in msg
    assert "Normal priority or higher" in msg
    assert STARVED_FOUND not in msg


@pytest.mark.parametrize("priority",
                         [NORMAL_PRIORITY, ABOVE_NORMAL_PRIORITY, HIGH_PRIORITY])
def test_proceeds_at_normal_priority_or_better(monkeypatch, priority):
    """Normal and above are all eligible to run the comparison."""
    registry = _simulate_windows(monkeypatch, nice=priority,
                                 normal=16.0, lowered=15.5)
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.OK
    assert CANNOT_TEST not in msg


def test_equal_throughput_passes(monkeypatch):
    """A lower-priority process that keeps its compute is healthy."""
    registry = _simulate_windows(monkeypatch, normal=15.8, lowered=15.6)
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.OK
    assert "15.8x" in msg
    assert "15.6x" in msg


def test_lower_priority_losing_compute_warns(monkeypatch):
    """The customer's failure: the low-priority arm collapses."""
    registry = _simulate_windows(monkeypatch, normal=16.0, lowered=2.0)
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert STARVED_FOUND in msg
    assert "16.0x" in msg
    assert "2.0x" in msg
    assert "scheduled task" in msg


def test_lost_compute_is_a_warning_not_an_error(monkeypatch):
    """Detecting the shortfall must not fail the run."""
    registry = _simulate_windows(monkeypatch, normal=16.0, lowered=1.0)
    status, _ = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert status not in (_types.Status.ERROR, _types.Status.CRITICAL)


@pytest.mark.parametrize(("delta", "expected"), [
    (0.05, _types.Status.OK),
    (-0.05, _types.Status.WARNING),
])
def test_threshold_boundary(monkeypatch, delta, expected):
    """The verdict flips either side of LOW_PRIORITY_MIN_RATIO."""
    normal = 16.0
    lowered = normal * (_constants.LOW_PRIORITY_MIN_RATIO + delta)
    registry = _simulate_windows(monkeypatch, normal=normal, lowered=lowered)
    status, _ = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is expected


def test_single_core_machine_passes(monkeypatch):
    """There is no parallelism to measure on one core."""
    registry = _simulate_windows(monkeypatch, physical=1)
    assert checks_cpu.check_low_priority_compute(registry=registry) == (_types.Status.OK, "")


def test_priority_lookup_failure_warns(monkeypatch):
    """An unreadable priority means the check cannot vouch for a result."""
    registry = _simulate_windows(monkeypatch, nice_error=OSError("denied"))
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert "Unable to determine" in msg


def test_normal_measurement_failure_warns(monkeypatch):
    """A failed probe degrades to a warning rather than raising.

    `run_checks` does not guard against a check raising, so a check that
    raises would abort the whole verification run.
    """
    registry = _simulate_windows(monkeypatch, measure_error=OSError("nope"))
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert "Unable to measure available CPU throughput" in msg


def test_low_priority_probe_failure_warns(monkeypatch):
    """If the child process cannot report, say the check did not run."""
    registry = _simulate_windows(monkeypatch, probe_failed=True)
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert "lower-priority process" in msg
    assert STARVED_FOUND not in msg


def test_shrunken_affinity_warns(monkeypatch):
    """Fewer CPUs offered to the lower-priority process is a warning.

    Throughput alone is healthy here, so only the affinity signal fires.
    """
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=list(range(16)), low_affinity=list(range(4)))
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert AFFINITY_SHRANK in msg
    assert "4 of the 16 CPUs" in msg
    assert STARVED_FOUND not in msg


def test_identical_affinity_passes(monkeypatch):
    """Matching CPU sets are not reported."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=list(range(16)), low_affinity=list(range(16)))
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.OK
    assert AFFINITY_SHRANK not in msg


def test_reordered_affinity_is_not_a_mismatch(monkeypatch):
    """The same CPUs in a different order are the same CPUs."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=15.8,
        normal_affinity=[3, 1, 2, 0], low_affinity=[0, 1, 2, 3])
    status, _ = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.OK


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
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.OK
    assert AFFINITY_SHRANK not in msg


def test_both_signals_are_reported_together(monkeypatch):
    """A machine that both restricts CPUs and parks cores says so once."""
    registry = _simulate_windows(
        monkeypatch, normal=16.0, lowered=2.0,
        normal_affinity=list(range(16)), low_affinity=list(range(2)))
    status, msg = checks_cpu.check_low_priority_compute(registry=registry)
    assert status is _types.Status.WARNING
    assert AFFINITY_SHRANK in msg
    assert STARVED_FOUND in msg


def test_affinity_is_not_read_off_windows(monkeypatch):
    """`cpu_affinity` does not exist on macOS, so the helper must not call it."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _probes._cpu_affinity() is None


@pytest.mark.parametrize("lowered", [2.0, 16.0])
def test_messages_are_console_safe(monkeypatch, lowered):
    """Messages must survive a non-UTF-8 console and rich markup."""
    registry = _simulate_windows(monkeypatch, normal=16.0, lowered=lowered)
    _, msg = checks_cpu.check_low_priority_compute(registry=registry)
    _assert_console_safe(msg)


def test_below_normal_detection_is_empty_off_windows(monkeypatch):
    """The helper is safe to call anywhere, despite Windows-only constants."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert _probes._is_below_normal_priority() is None


def test_measure_throughput_runs_real_threads():
    """Exercise the real probe: threads, barrier and hashing all cooperate."""
    buffer = b"\xa5" * (1 << 16)
    assert _probes._measure_throughput(1, 0.05, buffer) > 0
    assert _probes._measure_throughput(2, 0.05, buffer) > 0


def test_check_is_registered():
    """The check runs as part of a default verification."""
    registry = _registry.InstallationCheckRegistry()
    assert "Python: Low-priority CPU access" in [c.name for c in registry._checks]


# --- the lower-priority probe subprocess ---------------------------------


def _probe_stdout(effective, affinity, noise=""):
    """Build stdout as the child would emit it, optionally with other output."""
    import json as _json
    return (f"{noise}{_constants.PROBE_SENTINEL}"
            f"{_json.dumps([effective, affinity])}\n")


def _patch_subprocess(monkeypatch, *, stdout="", error=None):
    """Capture the arguments the probe would launch with."""
    captured = {}

    def fake_run(cmd, **kwargs: object):
        captured["cmd"] = cmd
        captured.update(kwargs)
        if error is not None:
            raise error
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_probe_command_guards_against_module_shadowing():
    """`-P` keeps the child's own directory off sys.path.

    Without it, running from `howso/utilities` lets `random.py` shadow the
    standard library and the child dies on import.
    """
    cmd = _probes._probe_command(4)
    assert cmd[0] == sys.executable
    assert "-P" in cmd
    assert _constants.PROBE_SENTINEL in cmd[-1]


def test_probe_is_created_at_low_priority_not_demoted(monkeypatch):
    """The child must be *created* below Normal.

    A process demoted after it is already running has already inherited its
    affinity, so a policy that assigns one at creation would never apply.
    """
    captured = _patch_subprocess(monkeypatch, stdout=_probe_stdout(8.0, [0, 1]))
    registry = _registry.InstallationCheckRegistry()
    assert _probes._run_low_priority_probe(registry, 8) == (8.0, [0, 1])
    assert captured["creationflags"] == _constants.BELOW_NORMAL_PRIORITY_CLASS
    assert _constants.BELOW_NORMAL_PRIORITY_CLASS == 0x00004000


def test_probe_result_is_found_among_other_output(monkeypatch):
    """Importing the package may print; the sentinel still locates the result."""
    noisy = "Warning: something chatty at import time\n"
    _patch_subprocess(monkeypatch, stdout=_probe_stdout(4.5, [0], noise=noisy))
    registry = _registry.InstallationCheckRegistry()
    assert _probes._run_low_priority_probe(registry, 8) == (4.5, [0])


def test_probe_returns_none_without_a_sentinel(monkeypatch):
    """Output with no result line reports unknown rather than guessing."""
    _patch_subprocess(monkeypatch, stdout="nothing useful here\n")
    registry = _registry.InstallationCheckRegistry()
    assert _probes._run_low_priority_probe(registry, 8) is None


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired(cmd="python", timeout=1),
    subprocess.CalledProcessError(returncode=1, cmd="python"),
    OSError("cannot spawn"),
])
def test_probe_returns_none_when_the_child_fails(monkeypatch, error):
    """A failed child never propagates out of the check."""
    _patch_subprocess(monkeypatch, error=error)
    registry = _registry.InstallationCheckRegistry()
    assert _probes._run_low_priority_probe(registry, 8) is None


def test_probe_is_skipped_off_windows(monkeypatch):
    """`creationflags` is Windows-only, so no child is launched elsewhere.

    Asserting on the return value alone would not catch a missing guard: the
    call would raise and be swallowed, returning None either way.
    """
    launched = []

    def record(*_args: object, **_kwargs: object):
        # Recorded rather than raised: the probe catches Exception, so a
        # raising stub would be swallowed and prove nothing.
        launched.append(1)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", record)
    registry = _registry.InstallationCheckRegistry()
    assert _probes._run_low_priority_probe(registry, 8) is None
    assert launched == [], "no child should be launched off Windows"


# --- check_cpu_steal -----------------------------------------------------


def _patch_steal(monkeypatch, *, fraction=0.0, cores=10, elapsed=1.0,
                 available=True, measure_error=None):
    """Simulate a host reporting `fraction` of its CPU time as stolen."""
    if available:
        stolen = fraction * elapsed * cores
        # Called three times: availability probe, before, after.
        readings = iter([0.0, 0.0, stolen])
        _patch_iv(monkeypatch, "_steal_seconds", lambda: next(readings))
    else:
        _patch_iv(monkeypatch, "_steal_seconds", lambda: None)

    clock = iter([0.0, elapsed])
    _patch_iv(monkeypatch, "time",
                        types.SimpleNamespace(monotonic=lambda: next(clock)))
    _patch_iv(monkeypatch, "psutil", types.SimpleNamespace(cpu_count=lambda **_kwargs: cores))

    def fake_measure(*_args: object, **_kwargs: object) -> int:
        if measure_error is not None:
            raise measure_error
        return 1

    _patch_iv(monkeypatch, "_measure_throughput", fake_measure)
    return _registry.InstallationCheckRegistry()


def test_steal_skipped_where_unreported(monkeypatch):
    """Windows and macOS do not report stolen time, so there is nothing to do."""
    registry = _patch_steal(monkeypatch, available=False)
    assert checks_cpu.check_cpu_steal(registry=registry) == (_types.Status.OK, "")


def test_steal_does_not_burn_cpu_when_unreported(monkeypatch):
    """The availability probe must come before the load, not after.

    Otherwise every macOS and Windows run pays for a measurement whose result
    can never be used.
    """
    burned = []
    _patch_iv(monkeypatch, "_steal_seconds", lambda: None)
    _patch_iv(monkeypatch, "_measure_throughput",
                        lambda *_a, **_k: burned.append(1))
    checks_cpu.check_cpu_steal(registry=_registry.InstallationCheckRegistry())
    assert burned == []


def test_negligible_steal_is_silent(monkeypatch):
    """A trace of stolen time is normal on any shared host."""
    registry = _patch_steal(monkeypatch, fraction=0.002)
    assert checks_cpu.check_cpu_steal(registry=registry) == (_types.Status.OK, "")


def test_moderate_steal_is_reported_without_warning(monkeypatch):
    """Worth seeing when chasing slowness, not worth failing over."""
    registry = _patch_steal(monkeypatch, fraction=0.05)
    status, msg = checks_cpu.check_cpu_steal(registry=registry)
    assert status is _types.Status.OK
    assert "5%" in msg


def test_heavy_steal_warns(monkeypatch):
    """The host is oversubscribed; say so, and point away from the install."""
    registry = _patch_steal(monkeypatch, fraction=0.35)
    status, msg = checks_cpu.check_cpu_steal(registry=registry)
    assert status is _types.Status.WARNING
    assert "35%" in msg
    assert "oversubscribed" in msg


@pytest.mark.parametrize(("fraction", "expected"), [
    (0.0099, _types.Status.OK),   # below the reporting floor: silent
    (0.0999, _types.Status.OK),   # reported, but not a warning
    (0.1000, _types.Status.WARNING),
])
def test_steal_thresholds(monkeypatch, fraction, expected):
    """The verdict turns at STEAL_REPORT_FRACTION and STEAL_WARN_FRACTION."""
    registry = _patch_steal(monkeypatch, fraction=fraction)
    status, _ = checks_cpu.check_cpu_steal(registry=registry)
    assert status is expected


def test_steal_measurement_failure_is_not_reported(monkeypatch):
    """A failed measurement says nothing rather than raising or guessing."""
    registry = _patch_steal(monkeypatch, measure_error=OSError("nope"))
    assert checks_cpu.check_cpu_steal(registry=registry) == (_types.Status.OK, "")


def test_steal_message_is_console_safe(monkeypatch):
    """The message must survive a Windows console and rich markup."""
    registry = _patch_steal(monkeypatch, fraction=0.35)
    _, msg = checks_cpu.check_cpu_steal(registry=registry)
    _assert_console_safe(msg)


def test_steal_check_is_registered():
    """The check runs as part of a default verification."""
    registry = _registry.InstallationCheckRegistry()
    assert "Python: CPU steal time" in [c.name for c in registry._checks]


# --- check_usable_cpus ---------------------------------------------------


def _patch_usable(monkeypatch, usable, active):
    """Pin what the process may use and what the system reports."""
    _patch_iv(monkeypatch, "_usable_cpu_count", lambda: usable)
    _patch_iv(monkeypatch, "_active_cpu_count", lambda: active)
    return _registry.InstallationCheckRegistry()


def test_usable_cpus_all_available_passes(monkeypatch):
    """Nothing is hidden, so report the count and move on."""
    registry = _patch_usable(monkeypatch, 16, 16)
    status, msg = checks_cpu.check_usable_cpus(registry=registry)
    assert status is _types.Status.OK
    assert "all 16 CPUs" in msg


def test_usable_cpus_restricted_warns(monkeypatch):
    """An affinity mask hiding CPUs is worth a warning."""
    registry = _patch_usable(monkeypatch, 4, 16)
    status, msg = checks_cpu.check_usable_cpus(registry=registry)
    assert status is _types.Status.WARNING
    assert "only 4 of the 16 CPUs" in msg
    assert "affinity" in msg


def test_usable_cpus_flags_windows_processor_group_cap(monkeypatch):
    """The >64 CPU case: psutil's affinity mask covers one group only."""
    registry = _patch_usable(monkeypatch, 64, 128)
    status, msg = checks_cpu.check_usable_cpus(registry=registry)
    assert status is _types.Status.WARNING
    assert "only 64 of the 128 CPUs" in msg
    assert "processor group" in msg


@pytest.mark.parametrize(("usable", "active"), [
    (None, 16),   # macOS: no affinity API at all
    (16, None),   # active count unavailable
    (16, 0),      # nonsense reading
    (None, None),
])
def test_usable_cpus_passes_when_undeterminable(monkeypatch, usable, active):
    """Where there is nothing to compare, stay quiet."""
    registry = _patch_usable(monkeypatch, usable, active)
    assert checks_cpu.check_usable_cpus(registry=registry) == (_types.Status.OK, "")


def test_usable_cpus_tolerates_more_usable_than_active(monkeypatch):
    """A usable count above the active one is not a restriction."""
    registry = _patch_usable(monkeypatch, 17, 16)
    status, _ = checks_cpu.check_usable_cpus(registry=registry)
    assert status is _types.Status.OK


def test_usable_cpus_message_is_console_safe(monkeypatch):
    """The message must survive a non-UTF-8 console and rich markup."""
    registry = _patch_usable(monkeypatch, 4, 16)
    _, msg = checks_cpu.check_usable_cpus(registry=registry)
    _assert_console_safe(msg)


def test_usable_count_uses_sched_getaffinity_off_windows(monkeypatch):
    """On Linux the count comes from sched_getaffinity, not psutil."""
    monkeypatch.setattr(sys, "platform", "linux")
    fake_os = types.SimpleNamespace(sched_getaffinity=lambda _pid: {0, 1, 2})
    _patch_iv(monkeypatch, "os", fake_os)
    assert _probes._usable_cpu_count() == 3


def test_usable_count_is_none_without_an_affinity_api(monkeypatch):
    """There is no affinity API on macOS, so there is nothing to report."""
    monkeypatch.setattr(sys, "platform", "darwin")
    _patch_iv(monkeypatch, "os", types.SimpleNamespace())
    assert _probes._usable_cpu_count() is None


def test_usable_count_tolerates_affinity_errors(monkeypatch):
    """A raising affinity call is reported as unknown, not propagated."""
    monkeypatch.setattr(sys, "platform", "linux")

    def boom(_pid):
        raise OSError("nope")

    _patch_iv(monkeypatch, "os",
                        types.SimpleNamespace(sched_getaffinity=boom))
    assert _probes._usable_cpu_count() is None


class _FakeWin32Fn:
    """Stand in for a ctypes-bound kernel32 function."""

    def __init__(self) -> None:
        self.argtypes = None
        self.restype = None
        self.called_with = None

    def __call__(self, group) -> int:
        self.called_with = group
        return 128


def _patch_win32_kernel32(monkeypatch, fn, *, psutil_says=64):
    """Route ctypes at a fake kernel32 and give psutil a different answer."""
    monkeypatch.setattr(sys, "platform", "win32")
    _patch_iv(monkeypatch, "ctypes", types.SimpleNamespace(
        WinDLL=lambda *_a, **_k: types.SimpleNamespace(
            GetActiveProcessorCount=fn),
        c_ushort=int, c_ulong=int))
    _patch_iv(monkeypatch, "psutil", types.SimpleNamespace(
        cpu_count=lambda **_kwargs: psutil_says))


def test_active_count_uses_win32_all_processor_groups(monkeypatch):
    """Windows must ask the OS, not psutil.

    psutil's affinity mask covers a single processor group and cannot see past
    64 CPUs, which is exactly the case this branch exists to catch.
    """
    fn = _FakeWin32Fn()
    _patch_win32_kernel32(monkeypatch, fn, psutil_says=64)
    assert _probes._active_cpu_count() == 128
    assert fn.called_with == 0xFFFF


def test_active_count_tolerates_win32_failure(monkeypatch):
    """A failed Win32 call reports unknown rather than raising."""
    def boom(*_args: object, **_kwargs: object):
        raise OSError("no kernel32")

    monkeypatch.setattr(sys, "platform", "win32")
    _patch_iv(monkeypatch, "ctypes", types.SimpleNamespace(WinDLL=boom))
    assert _probes._active_cpu_count() is None


def test_active_count_falls_back_to_psutil_off_windows(monkeypatch):
    """Off Windows the active count is psutil's logical CPU count."""
    monkeypatch.setattr(sys, "platform", "linux")
    _patch_iv(monkeypatch, "psutil",
        types.SimpleNamespace(cpu_count=lambda **_kwargs: 12))
    assert _probes._active_cpu_count() == 12


def test_usable_cpus_is_registered():
    """The check runs as part of a default verification."""
    registry = _registry.InstallationCheckRegistry()
    assert "Python: Usable CPUs" in [c.name for c in registry._checks]


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
    registry = _registry.InstallationCheckRegistry()
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

    _patch_iv(monkeypatch, "psutil", types.SimpleNamespace(cpu_count=cpu_count))


def test_visible_cores_agreement_is_informational(monkeypatch):
    """Matching counts report the number without flagging a problem."""
    _patch_cpu_counts(monkeypatch, 16)
    status, msg = checks_cpu.check_visible_cores(registry=_registry_with_engine(16))
    assert status is _types.Status.OK
    assert "16" in msg
    assert "agrees" in msg


def test_visible_cores_reports_physical_when_it_differs(monkeypatch):
    """A hyperthreaded host shows both numbers."""
    _patch_cpu_counts(monkeypatch, 32, 16)
    status, msg = checks_cpu.check_visible_cores(registry=_registry_with_engine(32))
    assert status is _types.Status.OK
    assert "32 logical" in msg
    assert "16 physical" in msg


def test_visible_cores_disagreement_warns(monkeypatch):
    """A mismatch between Python and the Engine is worth a warning."""
    _patch_cpu_counts(monkeypatch, 16)
    status, msg = checks_cpu.check_visible_cores(registry=_registry_with_engine(8))
    assert status is _types.Status.WARNING
    assert "16" in msg
    assert "8" in msg
    # The cause is a configured thread cap, not an over-provisioned host:
    # over-provisioning changes neither count.
    assert "max_num_threads" in msg
    assert "virtual machine" not in msg


def test_visible_cores_skips_comparison_for_remote_engine(monkeypatch):
    """A Platform client runs the Engine elsewhere, so counts are not compared."""
    _patch_cpu_counts(monkeypatch, 16)
    registry = _registry_with_engine(remote=True)
    status, msg = checks_cpu.check_visible_cores(registry=registry)
    assert status is _types.Status.OK
    assert "not compared" in msg


def test_visible_cores_tolerates_automatic_thread_count(monkeypatch):
    """Amalgam reports zero when it sizes its own pool; that is not a mismatch."""
    _patch_cpu_counts(monkeypatch, 16)
    status, msg = checks_cpu.check_visible_cores(registry=_registry_with_engine(0))
    assert status is _types.Status.OK
    assert "automatically" in msg


def test_visible_cores_tolerates_engine_error(monkeypatch):
    """An unreadable Engine count still reports what Python sees."""
    _patch_cpu_counts(monkeypatch, 16)
    registry = _registry_with_engine(error=RuntimeError("boom"))
    status, msg = checks_cpu.check_visible_cores(registry=registry)
    assert status is _types.Status.OK
    assert "16" in msg


@pytest.mark.parametrize(("n_logical", "error"), [
    (None, None),
    (0, None),
    (None, OSError("nope")),
])
def test_visible_cores_warns_when_undeterminable(monkeypatch, n_logical, error):
    """An unknown CPU count is itself worth reporting."""
    _patch_cpu_counts(monkeypatch, n_logical, error=error)
    status, msg = checks_cpu.check_visible_cores(registry=_registry_with_engine(16))
    assert status is _types.Status.WARNING
    assert "Unable to determine" in msg


def test_visible_cores_message_is_console_safe(monkeypatch):
    """The message must survive a non-UTF-8 console and rich markup."""
    _patch_cpu_counts(monkeypatch, 16)
    _, msg = checks_cpu.check_visible_cores(registry=_registry_with_engine(8))
    _assert_console_safe(msg)


def test_visible_cores_is_registered():
    """The check runs for any client type."""
    registry = _registry.InstallationCheckRegistry()
    configure(registry)
    checks = {c.name: c for c in registry._checks}
    assert "Howso Client: Visible CPUs" in checks
    assert checks["Howso Client: Visible CPUs"].client_required == "AbstractHowsoClient"
