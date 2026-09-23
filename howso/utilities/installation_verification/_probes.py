from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING

import psutil

from howso.client.exceptions import HowsoError

from ._constants import (
    BELOW_NORMAL_PRIORITY_CLASS,
    CPU_PROBE_BATCH,
    CPU_PROBE_SECONDS,
    LOW_PRIORITY_TIMEOUT,
    PROBE_SENTINEL,
)

if TYPE_CHECKING:
    from .registry import InstallationCheckRegistry


__all__ = [
    "_active_cpu_count",
    "_affinity_note",
    "_cpu_affinity",
    "_cpu_burn",
    "_is_below_normal_priority",
    "_measure_effective_parallelism",
    "_measure_throughput",
    "_probe_command",
    "_run_low_priority_probe",
    "_steal_seconds",
    "_usable_cpu_count",
]

def _cpu_burn(duration: float, buffer: bytes, barrier: threading.Barrier,
              counts: list[int], index: int) -> None:
    """Hash `buffer` repeatedly for `duration` seconds, recording the count."""
    try:
        barrier.wait(timeout=duration + 5.0)
    except threading.BrokenBarrierError:
        return
    end = time.monotonic() + duration
    count = 0
    while time.monotonic() < end:
        for _ in range(CPU_PROBE_BATCH):
            hashlib.sha256(buffer).digest()
        count += CPU_PROBE_BATCH
    counts[index] = count


def _measure_throughput(workers: int, duration: float, buffer: bytes) -> int:
    """Run `workers` burners simultaneously and return their total iterations."""
    counts = [0] * workers
    barrier = threading.Barrier(workers)
    threads = [
        threading.Thread(target=_cpu_burn,
                         args=(duration, buffer, barrier, counts, i),
                         daemon=True)
        for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=duration + 10.0)
    return sum(counts)


def _measure_effective_parallelism(workers: int) -> float:
    """Return how many workers' worth of throughput `workers` actually achieve."""
    buffer = b"\xa5" * (1 << 20)
    baseline = _measure_throughput(1, CPU_PROBE_SECONDS, buffer)
    parallel = _measure_throughput(workers, CPU_PROBE_SECONDS, buffer)
    if not baseline:
        raise HowsoError("Unable to measure CPU throughput.")
    return parallel / baseline


def _cpu_affinity() -> list[int] | None:
    """Return the CPUs this process may be scheduled on, or None if unknown."""
    if sys.platform != "win32":
        # `cpu_affinity` is not implemented on every platform.
        return None
    try:
        return sorted(psutil.Process().cpu_affinity())
    except Exception:  # noqa: BLE001
        return None


def _probe_command(workers: int) -> list[str]:
    """Build the command that measures compute from a lower-priority process.

    `-P` keeps the child's own directory off `sys.path`, so a module in this
    package named for a standard library one cannot shadow it. The child
    imports this module by its full path rather than through the package, so
    the package root does not have to re-export anything private.
    """
    code = (
        "import json;"
        "from howso.utilities.installation_verification._probes import ("
        "_cpu_affinity, _measure_effective_parallelism);"
        f"print('{PROBE_SENTINEL}' + json.dumps("
        f"[_measure_effective_parallelism({workers}), _cpu_affinity()]))"
    )
    return [sys.executable, "-P", "-c", code]


def _is_below_normal_priority() -> str | None:
    """Name this process's priority class when it is below Normal, else None."""
    if sys.platform != "win32":
        # The priority classes used below are defined only on Windows.
        return None
    # These priority classes are flags whose numeric values do not follow their
    # ordering, so match the named constants rather than compare magnitudes.
    reduced = {
        psutil.IDLE_PRIORITY_CLASS: "Low/Idle",
        psutil.BELOW_NORMAL_PRIORITY_CLASS: "Below Normal",
    }
    return reduced.get(psutil.Process().nice())


def _affinity_note(normal_affinity: list[int] | None,
                   low_affinity: list[int] | None) -> str:
    """Describe a CPU affinity that shrank with priority, else return ''."""
    if not normal_affinity or not low_affinity:
        # Unavailable on this platform, so there is nothing to compare.
        return ""
    if set(normal_affinity) == set(low_affinity):
        return ""
    return (f" A lower-priority process was allowed only "
            f"{len(low_affinity):,d} of the {len(normal_affinity):,d} CPUs "
            "this one may use, so Windows is restricting it directly.")


def _run_low_priority_probe(
    registry: InstallationCheckRegistry, workers: int
) -> tuple[float, list[int] | None] | None:
    """Measure parallelism and CPU affinity in a lower-priority child, or None.

    The child is *created* below Normal priority rather than demoted once it
    is already running, so a configuration that assigns an affinity mask when
    a process starts applies to it just as it would to a real workload.
    """
    if sys.platform != "win32":
        # `creationflags` is a Windows-only argument.
        return None
    try:
        completed = subprocess.run(  # noqa: S603 -- our own interpreter, no external input
            _probe_command(workers),
            capture_output=True, text=True, check=True,
            timeout=LOW_PRIORITY_TIMEOUT,
            creationflags=BELOW_NORMAL_PRIORITY_CLASS,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return None

    # Importing this package may print, so find the result rather than
    # assuming it is the only thing on stdout.
    for line in completed.stdout.splitlines():
        if line.startswith(PROBE_SENTINEL):
            effective, affinity = json.loads(line[len(PROBE_SENTINEL):])
            return (float(effective), affinity)
    return None


def _usable_cpu_count() -> int | None:
    """Count the CPUs this process may actually be scheduled on.

    Returns None where the platform exposes no affinity API at all, such as
    macOS, in which case there is nothing to compare.
    """
    if sys.platform == "win32":
        try:
            return len(psutil.Process().cpu_affinity())
        except Exception:  # noqa: BLE001
            return None
    # Linux and FreeBSD. Reflects taskset and cpuset restrictions. Fetched by
    # name because it is absent on other platforms.
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is None:
        return None
    try:
        return len(sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        return None


def _active_cpu_count() -> int | None:
    """Count the CPUs the system reports as active, across all groups.

    On Windows this deliberately avoids psutil: its affinity mask is a single
    64-bit value covering only the calling process's processor group, so a
    machine with more than 64 logical processors needs the Win32 call to learn
    its real size.
    """
    if sys.platform == "win32":
        try:
            all_processor_groups = 0xFFFF
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_count = kernel32.GetActiveProcessorCount
            get_count.argtypes = [ctypes.c_ushort]
            get_count.restype = ctypes.c_ulong
            return int(get_count(all_processor_groups)) or None
        except Exception:  # noqa: BLE001
            return None
    try:
        return psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        return None


def _steal_seconds() -> float | None:
    """Seconds of CPU time the hypervisor has taken from this host, or None.

    Only Linux reports this, and only when the hypervisor supplies it. Read by
    name because the field is absent from the named tuple elsewhere.
    """
    try:
        return getattr(psutil.cpu_times(), "steal", None)
    except Exception:  # noqa: BLE001
        return None
