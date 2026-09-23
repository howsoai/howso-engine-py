from __future__ import annotations

import sys
import time
import traceback
from typing import TYPE_CHECKING

import psutil

from ._constants import (
    CPU_PROBE_MAX_WORKERS,
    CPU_PROBE_SECONDS,
    LOW_PRIORITY_MIN_RATIO,
    STEAL_REPORT_FRACTION,
    STEAL_WARN_FRACTION,
)
from ._probes import (
    _active_cpu_count,
    _affinity_note,
    _cpu_affinity,
    _is_below_normal_priority,
    _measure_effective_parallelism,
    _measure_throughput,
    _run_low_priority_probe,
    _steal_seconds,
    _usable_cpu_count,
)
from ._types import Status

if TYPE_CHECKING:
    from .registry import InstallationCheckRegistry


def check_low_priority_compute(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Check whether a lower-priority process is given less CPU than this one.

    Some Windows configurations hand a process running below Normal priority a
    narrower set of CPUs, or otherwise less compute, than a Normal priority
    one. The cores still show up in the CPU count, so the only symptom is that
    Howso Engine™ runs several times slower than the hardware suggests
    whenever it is started that way, such as from a scheduled task.

    The probe measures achieved parallelism here, then again from a child
    process created below Normal priority, and compares the two. The cause is
    deliberately not assumed: an assigned affinity mask, a job object and a
    power configuration all show up the same way, as less compute for the
    lower-priority process. Core parking on its own generally does not, since
    it yields cores back as demand rises, and this probe makes that demand.

    The comparison requires this process to be at Normal priority or better;
    started any lower there is nothing to compare against, and the check says
    so instead of reporting a result it cannot stand behind. This simply passes
    on other operating systems.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    if (early := _low_priority_precheck(registry)) is not None:
        return early

    try:
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        workers = min(cores, CPU_PROBE_MAX_WORKERS)
        if workers < 2:
            # Nothing to measure on a single-core machine.
            return (Status.OK, "")
        normal = _measure_effective_parallelism(workers)
        normal_affinity = _cpu_affinity()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.WARNING, "Unable to measure available CPU throughput.")

    result = _run_low_priority_probe(registry, workers)
    if result is None:
        return (
            Status.WARNING,
            ("Unable to measure CPU throughput from a lower-priority process, "
             "so low-priority CPU access was not checked.")
        )
    lowered, low_affinity = result

    measured = (f"A Normal priority process reached {normal:,.1f}x the "
                f"throughput of a single worker; a lower-priority one reached "
                f"{lowered:,.1f}x")
    affinity_note = _affinity_note(normal_affinity, low_affinity)
    starved = lowered < normal * LOW_PRIORITY_MIN_RATIO

    if not starved and not affinity_note:
        return (Status.OK, f"{measured}.")

    starved_note = ""
    if starved:
        starved_note = (
            " A lower-priority process is given less compute on this machine, "
            "so Howso Engine™ will run slower whenever it is started that "
            "way, such as from a scheduled task."
        )
    return (
        Status.WARNING,
        (
            f"{measured}.{affinity_note}{starved_note} Check for an "
            "affinity mask or job object applied to the process or its "
            "parent, and failing that review the processor power settings "
            "with `powercfg /q SCHEME_CURRENT SUB_PROCESSOR`."
        )
    )


def _low_priority_precheck(
    registry: InstallationCheckRegistry
) -> tuple[Status, str] | None:
    """Return an early result when the comparison cannot be made from here."""
    if sys.platform != "win32":
        return (Status.OK, "")

    try:
        reduced_name = _is_below_normal_priority()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (
            Status.WARNING,
            ("Unable to determine this process's priority, so low-priority "
             "CPU access was not checked.")
        )

    if reduced_name is not None:
        return (
            Status.WARNING,
            (
                f'This process was started at "{reduced_name}" priority, so '
                "low-priority CPU access cannot be tested from here. Re-run "
                "the verification at Normal priority or higher to check it."
            )
        )
    return None


def check_cpu_steal(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Check whether the hypervisor is taking CPU time from this machine.

    A guest sold more virtual CPUs than its host can deliver runs slower than
    its CPU count suggests, and nothing about the count reveals it: the guest
    still sees every vCPU it was sold. Stolen time is the signal, and it only
    accrues while a vCPU actually wants to run, so this makes the machine busy
    for a moment and measures what it is charged for that.

    Reported only where the platform offers the figure, which in practice
    means a Linux guest.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    if _steal_seconds() is None:
        return (Status.OK, "")

    try:
        cores = psutil.cpu_count(logical=True) or 1
        workers = min(cores, CPU_PROBE_MAX_WORKERS)
        buffer = b"\xa5" * (1 << 20)
        before = _steal_seconds()
        started = time.monotonic()
        # Generate demand: with nothing wanting to run, nothing is stolen.
        _measure_throughput(workers, CPU_PROBE_SECONDS, buffer)
        elapsed = time.monotonic() - started
        stolen = (_steal_seconds() or 0.0) - (before or 0.0)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.OK, "")

    available = elapsed * cores
    fraction = stolen / available if available > 0 else 0.0
    if fraction < STEAL_REPORT_FRACTION:
        return (Status.OK, "")

    measured = (f"The hypervisor took {fraction:.0%} of this machine's CPU "
                "time while the check was busy")
    if fraction < STEAL_WARN_FRACTION:
        return (Status.OK, f"{measured}.")

    return (
        Status.WARNING,
        (
            f"{measured}, so Howso Engine™ gets less compute than the CPU "
            "count suggests. That points at the host being oversubscribed "
            "rather than at anything in this installation, and is worth "
            "raising with whoever provides it."
        )
    )


def check_usable_cpus(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:  # noqa: ARG001
    """
    Check that this process may use every CPU the system reports as active.

    A process can be confined to fewer CPUs than the machine has, by an
    affinity mask, by `taskset` or a cpuset on Linux, or on Windows by being
    held to a single processor group. The Engine sizes its work from the CPU
    count it is told, so a process quietly restricted this way runs slower
    than the hardware suggests with nothing to show for it.

    This passes where the platform exposes no affinity API, such as macOS.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    usable = _usable_cpu_count()
    active = _active_cpu_count()
    if usable is None or not active:
        return (Status.OK, "")

    if usable >= active:
        return (Status.OK, f"This process may use all {active:,d} CPUs.")

    return (
        Status.WARNING,
        (
            f"This process may use only {usable:,d} of the {active:,d} CPUs "
            "the system reports as active, so Howso Engine™ has less compute "
            "than the machine suggests. Check for an affinity mask on the "
            "process or its parent; on Windows a process is also held to one "
            "processor group, which caps it at 64 CPUs on larger machines."
        )
    )


def check_visible_cores(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Report how many CPUs Python and the Howso Engine each believe they have.

    The two counts are read independently, so a disagreement points at
    something between them misreporting the host, which is worth knowing
    before trusting any of the performance numbers this tool reports. The
    counts are only comparable when the Engine runs in-process; a Howso
    Platform client runs it on another machine entirely.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        logical = physical = None

    if not logical:
        return (Status.WARNING, "Unable to determine the number of visible CPUs.")

    seen = f"Python sees {logical:,d} logical CPUs"
    if physical and physical != logical:
        seen += f" on {physical:,d} physical cores"

    # Only an in-process Engine shares this machine's CPUs with us.
    try:
        amlg = getattr(registry.client, "amlg", None)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        amlg = None
    if amlg is None:
        return (
            Status.OK,
            f"{seen}. Howso Engine™ runs remotely, so its thread count is not compared."
        )

    try:
        engine_threads = amlg.get_max_num_threads()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.OK, f"{seen}. Unable to read the Howso Engine™ thread count.")

    if not engine_threads:
        # Amalgam treats zero as automatic, so there is nothing to compare.
        return (
            Status.OK,
            f"{seen}. Howso Engine™ selects its thread count automatically."
        )

    if engine_threads != logical:
        return (
            Status.WARNING,
            (
                f"{seen}, but Howso Engine™ reports {engine_threads:,d} "
                "threads. Howso Engine™ will size its work from its own count, "
                "so it will use less of this machine than it could. The usual "
                "cause is a max_num_threads value set explicitly in the Howso "
                "configuration."
            )
        )

    return (Status.OK, f"{seen}, and Howso Engine™ agrees.")
