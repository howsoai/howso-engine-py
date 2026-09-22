from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from functools import cached_property, partial
import hashlib
import importlib.metadata
import inspect
from io import StringIO
import logging
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
import traceback
from typing import Any, IO, Protocol, TypeAlias
import warnings

from faker.config import AVAILABLE_LOCALES
import pandas as pd
import psutil
from requests.exceptions import ConnectionError as RequestsConnectionError
from rich import print as rich_print
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

try:
    from howso import engine
except ImportError:
    engine = None
from howso.client import AbstractHowsoClient, HowsoClient
from howso.client.client import get_howso_client_class
from howso.client.exceptions import HowsoConfigurationError, HowsoError
from howso.client.schemas import Trainee
from howso.direct.client import HowsoDirectClient

try:
    from howso.platform import HowsoPlatformClient  # noqa: might not be available # type: ignore[reportMissingImports]
except ImportError:
    HowsoPlatformClient = None
try:
    from howso.validator import Validator  # noqa: might not be available # type: ignore[reportMissingImports]
except OSError as e:
    Validator = e
except ImportError:
    Validator = None
from howso.utilities import infer_feature_attributes

try:
    from howso.synthesizer import Synthesizer  # noqa: might not be available # type: ignore[reportMissingImports]
except ImportError:
    Synthesizer = None
from howso.utilities import StopExecution, Timer
from howso.utilities.locale import get_default_locale
from howso.utilities.posix import PlatformError, sysctl_by_name

logger = logging.getLogger(__name__)


def is_databricks() -> bool:
    """Check environment is on Databricks."""
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION", None))


def iv_print(*args: Any, **kwargs: Any) -> None:
    """Print wrapper for handling prints in different environments."""
    if is_databricks():
        # strip out rich formatting before printing
        alist = list(args)
        alist[0] = re.sub(r"\[.*\]", "", alist[0])
        args = tuple(alist)
        print(*args, **kwargs)
    else:
        rich_print(*args, **kwargs)


LOG_FILE = "howso_stacktrace.txt"

#: Seconds to wait on the isolated date/time support check before giving up.
DATE_FEATURE_TIMEOUT = 60

#: Seconds each phase of the CPU availability probe runs for.
CPU_PROBE_SECONDS = 0.5
#: Most workers the CPU availability probe will run at once.
CPU_PROBE_MAX_WORKERS = 16
#: Hashes per clock check. `hashlib` releases the GIL but the loop around it
#: does not, so checking the clock every iteration would serialize the workers
#: and understate the parallelism actually available.
CPU_PROBE_BATCH = 8
#: Fraction of ideal parallel throughput below which compute is judged limited.
CPU_PROBE_MIN_EFFICIENCY = 0.6


class Status(IntEnum):
    """Status Enum."""

    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    NOTICE = 3
    OK = 4


#: Classes or objects whose truthiness indicates an optional dependency is available.
Requirements: TypeAlias = Iterable[object]


class CheckFunction(Protocol):
    """The call signature that every registered check implements."""

    def __call__(self, *, registry: InstallationCheckRegistry) -> tuple[Status, str]:
        """Run the check and return its status and an optional message."""
        ...


@dataclass
class Check:
    """Store the specification of a single check."""

    name: str
    fn: CheckFunction
    client_required: str | None = None
    other_requirements: Requirements | None = None


class InstallationCheckRegistry:
    """Simple registry and executor of verification tests."""

    def __init__(self) -> None:
        """Initialize CheckRegistry."""
        self._checks: list[Check] = []

        # Storage for property caches
        self._client: AbstractHowsoClient | None = None
        self._client_classes: list[str] = []

        # This is where we'll write any stack traces.
        self.logger: StringIO | None = StringIO()

        # Adds the first check for Python
        self.add_check(
            name="Python: Running correctly (not under emulation)",
            fn=check_not_emulated
        )
        # Windows may park CPU cores rather than waking them for this process
        self.add_check(
            name="Python: CPU availability",
            fn=check_cpu_availability
        )
        # And the next check which builds a howso client
        self.add_check(
            name="Howso Client: Configuration",
            fn=self._check_client_configuration
        )

    def add_check(self, name: str,
                  fn: CheckFunction,
                  client_required: str | None = None,
                  other_requirements: Requirements | object | None = None
                  ) -> None:
        """
        Add a check for this installation.

        Parameters
        ----------
        name : str
            The name to display for the check.
        fn : Callable
            The callable to run to perform the check.
        client_required : str, default None
            Optional. If set should be the class name of the client required.
        other_requirements : Iterable of classes or objects, default None
            Optional. Other required classes, E.g., `Synthesizer`. Note, these
            are not strings. These are the classes or instances of things
            required. They should have been imported in a try/catch and sent
            to something falsy if not imported.
        """
        requirements: Requirements | None
        if other_requirements is None:
            requirements = None
        elif isinstance(other_requirements, Iterable):
            requirements = other_requirements
        else:
            requirements = [other_requirements]
        self._checks.append(
            Check(name=name, fn=fn, client_required=client_required,
                  other_requirements=requirements))

    @cached_property
    def _name_length(self) -> int:
        """
        Compute the max *rendered* length among all check names.

        Returns
        -------
        int
            The maximum length of names of checks.
        """
        if len(self._checks):
            return max(len(c.name) for c in self._checks)
        return 1

    @property
    def client(self) -> AbstractHowsoClient:
        """
        Return a lazily-instantiated and cached client to use.

        Returns
        -------
        AbstractHowsoClient
            A instance of AbstractHowsoClient determined by the
            user's configuration.
        """
        if self._client is None:
            self._client = HowsoClient(debug=0)
        return self._client

    @property
    def client_classes(self) -> list[str]:
        """
        Return list of super class names for the current cached client.

        Returns
        -------
        list of class names
        """
        if self._client is None:
            return []
        if self._client_classes == []:
            self._client_classes = [
                c.__name__ for c in inspect.getmro(type(self._client))]

        return self._client_classes

    @staticmethod
    def _check_client_configuration(registry: InstallationCheckRegistry) -> tuple[Status, str]:  # noqa: PLR0911
        """
        Check that the Howso client can be instantiated.

        This is intended to be among the first checks, so it is a built-in.

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
                A message to display about the WARNING, ERROR or CRITICAL.
        """
        try:
            registry._client = HowsoClient(debug=0)
        except HowsoConfigurationError:
            traceback.print_exc(file=registry.logger)
            return (
                Status.CRITICAL,
                (
                    "The howso configuration file was not found in the "
                    "location that was specified in the `HOWSO_CONFIG` environment "
                    "variable. Please see the Howso Client installation "
                    "documentation for further details."
                )
            )
        except PermissionError:
            traceback.print_exc(file=registry.logger)
            return (
                Status.CRITICAL,
                (
                    "Howso Client could not be started due to file "
                    "permissions. Please see the Howso Client installation "
                    "documentation for further details."
                )
            )
        except (ModuleNotFoundError, StopExecution):
            traceback.print_exc(file=registry.logger)
            return (
                Status.CRITICAL, (
                    "Unable to connect to a Howso Platform. Please ensure "
                    "that you have a valid `howso.yml` file in the correct "
                    "location. Please see the Howso Client installation "
                    "documentation for further details."
                )
            )
        except ValueError:
            traceback.print_exc(file=registry.logger)
            return (
                Status.CRITICAL, (
                    "The client was unable to find Howso core binaries. "
                    "Please see the Howso Client installation "
                    "documentation for further details."
                )
            )
        except RequestsConnectionError:
            traceback.print_exc(file=registry.logger)
            return (
                Status.CRITICAL,
                (
                    "Unable to connect to the Howso Platform "
                    "configured in your `howso.yml` file. Please check for "
                    "configuration errors and/or network connectivity to the "
                    "platform host."
                )
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc(file=registry.logger)
            return (
                Status.CRITICAL,
                (
                    "There was a problem instantiating the Howso client. "
                    "Please see the Howso Client installation documentation "
                    "for further details."
                )
            )

        return (Status.OK, "")

    def _print_versions(self, versions: Mapping[str, str], *, file: IO[str] | None = None) -> None:
        """Output version information."""
        if not versions:
            return
        if "python" in versions:
            iv_print(f"Python version: {versions['python']}", file=file)
        if "client_type" in versions:
            iv_print(f"Client type: {versions['client_type']}", file=file)
        if "client" in versions:
            iv_print(f"Client version: {versions['client']}", file=file)
        if "client_base" in versions:
            iv_print(f"API client version: {versions['client_base']}", file=file)
        if "platform" in versions:
            iv_print(f"Platform version: {versions['platform']}", file=file)

    def run_checks(self) -> int:  # noqa: PLR0912, PLR0915
        """
        Run each of the registered checks and output their status.

        Returns
        -------
        int
            The appropriate exit code to use at program end.
        """
        all_issues = 0
        critical_issues = 0

        disable = False
        if is_databricks():
            disable = True
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
            disable=disable)

        if not self.logger:
            self.logger = StringIO()
        start_time = datetime.now().astimezone()

        versions = {
            "python": "Could not get Python version.",
            "client_type": "Could not get client type.",
            "client": "Could not get client version.",
        }

        try:
            versions = get_versions()
            self._print_versions(versions)
            with progress:
                for check in progress.track(self._checks):
                    if check.client_required and (
                        self._client is None or
                        check.client_required not in self.client_classes
                    ):
                        continue
                    if check.other_requirements and not all(check.other_requirements):
                        continue

                    progress.tasks[0].description = (
                        f"{check.name:{self._name_length}s}")

                    status, msg = check.fn(registry=self)
                    if status == Status.CRITICAL:
                        emoji = ":boom:"
                        color = "magenta"
                    elif status == Status.ERROR:
                        emoji = ":heavy_exclamation_mark:"
                        color = "red"
                    elif status == Status.WARNING:
                        emoji = ":warning:"
                        color = "yellow"
                    elif status == Status.NOTICE:
                        emoji = ":interrobang:"
                        color = "medium_turquoise"
                    else:  # status == Status.OK:
                        emoji = ":heavy_check_mark:"
                        color = "green"

                    if status < Status.NOTICE:
                        # This includes warnings
                        all_issues += 1

                    if status in [Status.CRITICAL, Status.ERROR]:
                        # This does not include warnings.
                        critical_issues += 1

                    # Force UTF-8 encoding for stdout on Windows
                    if sys.platform == "win32":
                        sys.stdout.reconfigure(encoding="utf-8")

                    if msg:
                        progress.console.print(
                            f"[bold]{check.name:{self._name_length}s} - "
                            f"[{color}]{status.name} {emoji} - {msg}")
                    else:
                        progress.console.print(
                            f"[bold]{check.name:{self._name_length}s} - "
                            f"[{color}]{status.name} {emoji}")

                progress.tasks[0].description = (
                    f"{'All checks complete':{self._name_length}s}")
        finally:
            # Write the contents of `logger`, if any, to a disk file.
            if self.logger:
                logs = self.logger.getvalue()
                self.logger.close()
                self.logger = None
                end_time = datetime.now().astimezone()
                log_file = Path(".", LOG_FILE)
                if len(logs):
                    all_issues += 1
                    with log_file.open(mode="w+") as log:
                        iv_print(f"Installation verification run: "
                                 f"{start_time.isoformat()}\n",
                                 file=log)
                        self._print_versions(versions, file=log)
                        iv_print("=" * 80 + "\n", file=log)
                        iv_print(logs, file=log)
                        iv_print(f"Verification complete: {end_time.isoformat()} "
                                 f"(elapsed time: {end_time - start_time})\n",
                                 file=log)

        if not all_issues:
            iv_print("[bold green]You are ready to use Howso™!")
        else:
            iv_print("[bold yellow]There were one or more issues. Please review "
                     "the messages emitted during the installation verification "
                     "process to identify next steps. If you cannot resolve "
                     "these issues please do not hesitate to contact your "
                     "Howso™ representative.")
            iv_print(f'[bold yellow]Any CRITICAL issues are logged in the file '
                     f'"{LOG_FILE}" in the current directory.')

        # This is largely for automated systems.
        if critical_issues:
            return 255
        return 0


def get_versions() -> dict[str, str]:
    """
    Get the Python, client, and platform versions of the environment.

    Returns
    -------
    dict
        A mapping containing keys 'python', 'client', 'client_type', and possibly
        'platform'. These all are mapped to strings indicating their version.
    """
    # python version
    try:
        py_version = sys.version_info
        py_version_string = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    except Exception:  # noqa: BLE001
        py_version_string = "Could not get Python version."

    versions = {
        "python": py_version_string,
        "client_type": "Could not get client type.",
        "client": "Could not get client version.",
    }

    # client type and version
    try:
        # Instantiating the client is often the point of failure, this won't trigger that
        client_class, _ = get_howso_client_class()
        versions["client_type"] = client_class.__name__
        engine_version = importlib.metadata.version("howso-engine")
        if issubclass(client_class, HowsoDirectClient):
            versions["client"] = engine_version
        else:
            versions["client_base"] = engine_version
            if _is_platform_client(client_class):
                versions["client"] = importlib.metadata.version("howso-platform-client")
    except Exception:  # noqa: BLE001, S110
        # Failed to get version, leave default message
        pass

    # platform version
    try:
        client = HowsoClient(debug=0)
        client_version_info = client.get_version()
        if "platform" in client_version_info:
            versions["platform"] = client_version_info["platform"]
    except Exception:  # noqa: BLE001, S110
        pass

    return versions


def get_nonce(length: int = 8) -> str:
    """
    Return a string of `length` random hexadecimal digits.

    Parameters
    ----------
    length : int, default: 8
        The length of the returned string.

    Returns
    -------
    str
        A string representing a hexadecimal number of length `length`.
    """
    return f"{random.randint(0, 16 ** length):0{length}x}"  # noqa: S311


def generate_dataframe(*, client: AbstractHowsoClient,
                       num_samples: int = 150,
                       timeout: int | None = None
                       ) -> tuple[pd.DataFrame, float | int]:
    """
    Use HowsoClient to create a dataframe of random data.

    Parameters
    ----------
    client : AbstractHowsoClient
        The Howso client instance to use.
    num_samples : int, default 150
        The number of samples to synthesize.
    timeout : int or None, default None
        Optional. If provided, `num_samples` is ignored and synthesis happens
        1 record at a time until `timeout` seconds have elapsed.

    Returns
    -------
    pd.DataFrame
        A dataframe of the synthesized records.

    """
    continuous_feature = {
        "type": "continuous",
        "decimal_places": 2,
        "bounds": {
            "min": 0.0,
            "max": 100.0,
            "allow_null": False,
        }
    }
    features = {
        "alpha": continuous_feature,
        "beta": continuous_feature,
        "gamma": continuous_feature,
        "class": {
            "type": "nominal",
            "bounds": {
                "allowed": ["apple", "banana", "cherry"],
                "allow_null": False},
        }
    }
    feature_names = list(features.keys())

    trainee = client.create_trainee(
        name=f"installation_verification generated dataframe ({get_nonce()})",
        features=features,
        persistence="allow" if _is_platform_client(client) else "never"
    )
    if not isinstance(trainee, Trainee):
        raise HowsoError("Unable to create trainee.")
    try:
        client.set_feature_attributes(trainee.id, features)
        client.acquire_trainee_resources(trainee.id, max_wait_time=0)
        action: pd.DataFrame | list[list[Any]]
        if timeout:
            # Generate 1 case at a time until `timeout` has passed.
            deadline = time.monotonic() + timeout
            action = []
            while time.monotonic() < deadline:
                if reaction := client.react(
                    trainee.id, action_features=feature_names,
                    num_cases_to_generate=1, desired_conviction=1.0,
                    generate_new_cases="no", suppress_warning=True
                ):
                    action.append(reaction["action"].iloc[0].tolist())
            elapsed_time = timeout
        else:
            with Timer() as timer:
                reaction = client.react(
                    trainee.id, action_features=feature_names,
                    num_cases_to_generate=num_samples, desired_conviction=1.0,
                    generate_new_cases="no", suppress_warning=True
                )
            action = reaction["action"] if reaction else []
            elapsed_time = timer.seconds or math.nan
    finally:
        client.delete_trainee(trainee.id)
    df = pd.DataFrame(action, columns=feature_names)
    return df, elapsed_time


def check_not_emulated(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:  # noqa: ARG001
    """
    Check that the installation is not running under emulation on MacOS.

    This simply passed under other operating systems.

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
    if sys.platform == "darwin":
        try:
            proc_translated = sysctl_by_name("sysctl.proc_translated", "int")
        except PlatformError:
            return (Status.OK, "")
        except Exception:  # noqa: BLE001
            return (Status.WARNING, "Unable to check if running under emulation.")
        if proc_translated == 1:
            # Python is running under Rosetta. Advise the user install the
            # correct Python.
            return (
                Status.WARNING,
                (
                    "Python is running under emulation on this system. This might "
                    "happen if the wrong installer was used to install Python. It "
                    "is **strongly** advised that Python is reinstalled using a "
                    '"Universal Installer" before proceeding.'
                )
            )

    return (Status.OK, "")


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


def check_cpu_availability(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Check that this process can actually use the CPU cores it appears to have.

    Some Windows power configurations park CPU cores rather than waking them
    for a process, notably when that process runs below Normal priority. The
    cores still show up in the CPU count, so the only outward symptom is that
    the Engine runs several times slower than the hardware suggests it should.

    Rather than inspect the configuration, this measures the effect: it
    compares the throughput of one busy worker against that of many running at
    once. The result is a ratio, so competing load on the machine slows both
    phases and largely cancels out. This simply passes on other operating
    systems.

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
    if sys.platform != "win32":
        return (Status.OK, "")

    try:
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        workers = min(cores, CPU_PROBE_MAX_WORKERS)
        if workers < 2:
            # Nothing to measure on a single-core machine.
            return (Status.OK, "")

        buffer = b"\xa5" * (1 << 20)
        baseline = _measure_throughput(1, CPU_PROBE_SECONDS, buffer)
        parallel = _measure_throughput(workers, CPU_PROBE_SECONDS, buffer)
        if not baseline:
            return (Status.WARNING, "Unable to measure available CPU throughput.")
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.WARNING, "Unable to measure available CPU throughput.")

    effective = parallel / baseline
    if effective >= workers * CPU_PROBE_MIN_EFFICIENCY:
        return (Status.OK, "")

    return (
        Status.WARNING,
        (
            f"Running {workers} busy workers produced only {effective:,.1f}x "
            f"the throughput of a single worker, on a machine reporting "
            f"{cores:,d} physical cores. The Engine will run correspondingly "
            "slower than this hardware suggests. Windows power plans can park "
            "CPU cores instead of waking them for a process, especially one "
            "running below Normal priority."
            + _priority_note(registry)
            + ' Review "Processor performance core parking" in the active '
            "power plan with `powercfg /q SCHEME_CURRENT SUB_PROCESSOR`."
        )
    )


def _priority_note(registry: InstallationCheckRegistry) -> str:
    """Describe the process priority when it is below Normal, else return ''."""
    if sys.platform != "win32":
        # The priority classes used below are defined only on Windows.
        return ""
    try:
        priority = psutil.Process().nice()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return ""
    # These priority classes are flags whose numeric values do not follow their
    # ordering, so match the named constants rather than compare magnitudes.
    reduced = {
        psutil.IDLE_PRIORITY_CLASS: "Low/Idle",
        psutil.BELOW_NORMAL_PRIORITY_CLASS: "Below Normal",
    }
    if (name := reduced.get(priority)) is not None:
        return (f' This process is running at "{name}" priority, which is the '
                "usual trigger.")
    return ""


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
            f"{seen}. The Engine runs remotely, so its thread count is not compared."
        )

    try:
        engine_threads = amlg.get_max_num_threads()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.OK, f"{seen}. Unable to read the Engine thread count.")

    if not engine_threads:
        # Amalgam treats zero as automatic, so there is nothing to compare.
        return (
            Status.OK,
            f"{seen}. The Engine selects its thread count automatically."
        )

    if engine_threads != logical:
        return (
            Status.WARNING,
            (
                f"{seen}, but Howso Engine reports {engine_threads:,d} "
                "threads. The Engine will size its work from its own count, "
                "so the two disagreeing may mean the host is misreporting "
                "its CPUs, as over-provisioned virtual machines often do, or "
                "that a thread count was set explicitly in configuration."
            )
        )

    return (Status.OK, f"{seen}, and Howso Engine agrees.")


def check_generate_dataframe(*, registry: InstallationCheckRegistry,
                             threshold: float | None = None) -> tuple[Status, str]:
    """
    Rate the speed in which a dataframe was able to be generated.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    threshold : float or None, default None
        Optional. If provided determines how long the process can run before
        considering it to return a status of WARNING.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        _, duration = generate_dataframe(client=registry.client,
                                         num_samples=150)
    except ValueError:
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "The client was unable to find Howso core binaries. "
                "Please see the Howso Client installation documentation "
                "for further details."
            )
        )
    if threshold is not None and duration > threshold:
        return (
            Status.WARNING,
            (
                f"The client required a duration of {duration:,.1f} to "
                f"synthesize a DataFrame, this should require no more than "
                f"{threshold:,.1f} seconds. This warning may be expected in "
                f"auto-scaling installations."
            )
        )
    return (Status.OK, "")


def check_basic_synthesis(*, registry: InstallationCheckRegistry,
                          source_df: pd.DataFrame | None = None) -> tuple[Status, str]:
    """
    Validate that Synthesizer can perform a basic synthesis.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        # Using a made-up dataframe, synthesize a new one from it.
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client)

        features = infer_feature_attributes(source_df)
        if not Synthesizer:
            raise AssertionError("Howso Synthesizer™ is not installed.")  # noqa: TRY301
        with Synthesizer(client=registry.client, privacy_override=True) as s:
            s.train(source_df, features)
            synthesized_df = s.synthesize_cases(n_samples=100)
        if synthesized_df.shape != (100, 4):
            return (Status.CRITICAL, "Synthetic dataframe is the wrong shape.")
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete check. Check installation.")
    else:
        return (Status.OK, "")


def check_locales_available(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Check that default locale is available in faker.

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
        if (default_locale := get_default_locale()[0]) not in AVAILABLE_LOCALES:
            return (
                Status.WARNING,
                (
                    f"Current locale, {default_locale} is not available in Faker "
                    f"(https://faker.readthedocs.io/en/master/locales.html). "
                    f"The locale for Faker will be set to 'en_US'."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete check. Check installation.")
    else:
        return (Status.OK, "")


def check_save(*, registry: InstallationCheckRegistry,
               source_df: pd.DataFrame | None = None) -> tuple[Status, str]:
    """
    Ensure that a Trainee can can be saved.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    client = trainee = None
    try:
        client = registry.client
        if source_df is None:
            source_df, _ = generate_dataframe(client=client)
        features = infer_feature_attributes(source_df)
        feature_names = list(features.keys())
        if trainee := client.create_trainee(
            name=f"installation_verification check save ({get_nonce()})",
            features=features
        ):
            client.train(trainee.id, source_df, features=feature_names)
            client.persist_trainee(trainee.id)
        else:
            raise HowsoError("Could not create a trainee.")  # noqa: TRY301
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not save Trainee. Please check file permissions.")
    else:
        return (Status.OK, "")
    finally:
        try:
            if client and trainee:
                client.delete_trainee(trainee.id)
        except Exception:  # noqa: BLE001, S110
            pass


def check_synthesizer_create_delete(*, registry: InstallationCheckRegistry,
                                    source_df: pd.DataFrame | None = None
                                    ) -> tuple[Status, str]:
    """
    Ensure that a Trainee can can be created and deleted.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
    """
    s = None
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client)

        features = infer_feature_attributes(source_df)
        if not Synthesizer:
            raise AssertionError("Howso Synthesizer™ is not installed.")  # noqa: TRY301
        s = Synthesizer(client=registry.client, privacy_override=True)

        s.train(source_df[:50], features)
        n = s.cl.get_num_training_cases(s.trainee.id)
        if n != 50:
            return (
                Status.ERROR,
                (
                    f"Training did not produce the correct number of "
                    f"training cases ({n}). Howso Synthesizer might not be "
                    "installed correctly. "
                    "Try: `pip install --upgrade howso-synthesizer`."
                )
            )

        s.cl.delete_trainee(s.trainee.id)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "Could not complete check. Check installation. "
                "Try: `pip install --upgrade howso-synthesizer`."
            )
        )
    else:
        return (Status.OK, "")
    finally:
        try:
            if s:
                s.cl.delete_trainee(s.trainee.id)
        except Exception:  # noqa: BLE001, S110
            pass


def check_latency(*, registry: InstallationCheckRegistry,
                  source_df: pd.DataFrame | None = None,
                  notice_threshold: int = 25, warning_threshold: int = 20,
                  timeout: int = 10) -> tuple[Status, str]:
    """
    Ensure creation of `sample_threshold` requests within `timeout` seconds.

    # This test uses a deliberately inefficient manner to synthesize records
    # and is # done this way to expose network latency issues.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.
    notice_threshold : int, default 25
        The number of samples that should be generated within `timeout`
        seconds. If it cannot generate this number within the timeout, then the
        resulting Status will be NOTICE.
    warning_threshold : int, default 20
        The number of samples that should be generated within `timeout`
        seconds. If it cannot generate this number within the timeout, then the
        resulting Status will be WARNING.
    timeout : int, default 10
        The number of seconds to run synthesis, one sample at a time.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client,
                                              timeout=timeout)
        num_rows = source_df.shape[0]
        if num_rows < warning_threshold:
            return (
                Status.WARNING,
                (
                    f"{num_rows} records synthesized in {timeout:,d} seconds. "
                    f"A minimum of {warning_threshold:,d} samples expected. "
                    "Ensure a good network connection and that Howso "
                    "Platform is installed on sufficient cluster hardware. "
                    "In auto-scaling installations this may be due to slow node "
                    "start-ups."
                )
            )
        if num_rows < notice_threshold:
            return (
                Status.NOTICE,
                (
                    f"{num_rows} records synthesized in {timeout:,d} seconds. "
                    f"Less than {notice_threshold:,d} may indicate "
                    "a poor network connection or slow/oversubscribed "
                    "cluster hardware. This notice is expected in auto-scaling "
                    "installations."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete check. Check installation.")
    else:
        return (Status.OK, "")


def check_performance(*, registry: InstallationCheckRegistry,
                      num_samples: int = 5_000, notice_threshold: float = 10.0,
                      warning_threshold: float = 20.0) -> tuple[Status, str]:
    """
    Ensure can generate `num_samples` records with `time_threshold` seconds.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    num_samples : int, default 5,000
        The number of samples to generate.
    notice_threshold : float, default 10.0
        The notice time-threshold in seconds. If the generation of `num_samples`
        requires more than `threshold` seconds, the returned Status will be
        NOTICE.
    warning_threshold : float, default 20.0
        The warning time-threshold in seconds. If the generation of `num_samples`
        requires more than `threshold` seconds, the returned Status will be
        WARNING.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        _, num_seconds = generate_dataframe(client=registry.client,
                                            num_samples=num_samples)
        msg = (
            f"{num_samples:,d} records were synthesized in "
            f"{num_seconds:,.1f} seconds. "
        )
        if num_seconds > warning_threshold:
            return (
                Status.WARNING,
                (
                    msg + f" This should require fewer than {warning_threshold:,.1f} "
                    "seconds. Ensure the installation is on equipment that meets "
                    "Howso's recommended hardware specifications. "
                    "In auto-scaling installations this may be due to slow node "
                    "start-ups."
                )
            )
        if num_seconds > notice_threshold:
            return (
                Status.NOTICE,
                (
                    msg + f" Greater than {notice_threshold:,.1f} seconds may indicate "
                    "slow or underpowered hardware. Ensure the installation is on "
                    "equipment that meets Howso's recommended specifications. "
                    "This notice is expected in auto-scaling installations."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            "Could not complete operation. Check installation."
        )
    else:
        return (Status.OK, "")


def overridable_resources(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Ensure that resources are overridable for Platform workers.

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
    # Create a trainee and acquire its resources.
    try:
        if engine:
            with engine.Trainee(
                name=f"installation_verification overridable resources ({get_nonce()})",
                persistence="never",
                runtime={"scaling": {"resources": {"cpu": {"minimum": 1_500}}}},
            ) as trainee:
                runtime = trainee.get_runtime()
                try:
                    cpu_limits = runtime["scaling"]["resources"]["cpu"]  # type: ignore[reportOptionalSubscript]
                    if cpu_limits["minimum"] != 1_500:  # type: ignore[reportOptionalSubscript]
                        raise AssertionError("Incorrect value returned")  # noqa: TRY301
                except (KeyError, TypeError, ValueError):
                    traceback.print_exc(file=registry.logger)
                    return (Status.CRITICAL, "get_runtime() returned an unexpected response.")
                except AssertionError:
                    traceback.print_exc(file=registry.logger)
                    return (Status.CRITICAL, "get_runtime() returned an unexpected value for minimum CPU cores.")
                else:
                    return (Status.OK, "")
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL, "Could not create a trainee.")

    # Reached only if `engine` could not be imported.
    return (Status.CRITICAL, "Howso Engine™ is not installed.")


def check_engine_operation(
    *,
    registry: InstallationCheckRegistry,
    source_df: pd.DataFrame | None = None
) -> tuple[Status, str]:
    """
    Ensure that Howso Engine operates as it should.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    trainee = None
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client,
                                              num_samples=150)

        features = infer_feature_attributes(source_df)

        train_idx = source_df.sample(frac=0.8).index
        df_train = source_df.loc[source_df.index.isin(train_idx)]
        df_test = source_df.loc[~source_df.index.isin(train_idx)]
        x_train = df_train.drop("class", axis=1)
        y_train = df_train["class"]
        x_test = df_test.drop("class", axis=1)

        action_features = ["class"]
        context_features = x_train.columns.tolist()
        if not engine:
            raise AssertionError("Howso Engine™ is not installed.")  # noqa: TRY301
        trainee = engine.Trainee(
            name=(f"installation_verification "
                  f"check engine operations ({get_nonce()})"),
            features=features, overwrite_existing=True
        )
        trainee.train(x_train.join(y_train))
        trainee.analyze()
        response = trainee.react(x_test, context_features=context_features,
                                 action_features=action_features)
        results = response["action"][action_features]
        if results.shape[0] != x_test.shape[0]:
            return (
                Status.ERROR,
                (
                    "Results do not have the same number of samples as the "
                    "input data."
                )
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete operation. Check installation.")
    else:
        return (Status.OK, "")
    finally:
        try:
            if engine and trainee:
                engine.delete_trainee(trainee.id)
        except Exception:  # noqa: BLE001, S110
            pass


def check_validator_operation(
    *, registry: InstallationCheckRegistry,
    source_df: pd.DataFrame | None = None,
) -> tuple[Status, str]:
    """
    Ensure that Validator-Enterprise operates as it should.

    Parameters
    ----------
    registry : The InstallationCheckRegistry
        The registry used to run this check.
    source_df : pd.DataFrame or None, default None
        Optional. If not provided a new dataframe will be synthesized.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    if isinstance(Validator, Exception):
        iv_print(Validator, file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "Howso Validator™ was not installed correctly. "
                "Please check installation."
            )
        )
    try:
        if source_df is None:
            source_df, _ = generate_dataframe(client=registry.client, num_samples=150)

        orig_df = source_df.sample(frac=0.5)
        gen_df = source_df[~source_df.index.isin(orig_df.index)]
        features = infer_feature_attributes(orig_df)
        if not Validator:
            raise AssertionError("Howso Validator™ is not installed.")  # noqa: TRY301

        with Validator(orig_df, gen_df, features=features, verbose=-1) as v:
            result = v.run_metric("DescriptiveStatistics")

        if result.desirability == 0 or len(result.errors):
            return (Status.CRITICAL, "Validator encountered one or more errors.")

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (Status.CRITICAL,
                "Could not complete operation. Check installation.")
    else:
        return (Status.OK, "")


def _attempt_train_date_feature(result_queue: multiprocessing.Queue[int]) -> None:
    """
    Attempt to train a date feature to check for proper time zone support.

    Parameters
    ----------
    result_queue : A multiprocessing queue instance
        A queue to put the results.
    """
    client = HowsoClient()
    features = {"date": {"type": "continuous", "date_time_format": "%Y-%m-%d"}}
    trainee = client.create_trainee(
        name=f"installation_verification check_tzdata_installed ({get_nonce()})",
        features=features,
        persistence="allow" if _is_platform_client(client) else "never"
    )
    try:
        client.train(trainee_id=trainee.id, cases=[["2001-01-01"]], features=["date"])
        result_queue.put(client.get_num_training_cases(trainee.id))
    finally:
        client.delete_trainee(trainee.id)


def _is_platform_client(client: type[AbstractHowsoClient] | AbstractHowsoClient) -> bool:
    """Check if a client is a platform client type or instance."""
    if HowsoPlatformClient is None:
        return False
    if isinstance(client, type) and issubclass(client, HowsoPlatformClient):
        return True
    return isinstance(client, HowsoPlatformClient)


def check_tzdata_installed(*, registry: InstallationCheckRegistry) -> tuple[Status, str]:
    """
    Check for timezone support in host OS.

    The installation_verification module has already checked normal operations
    without dates. This check is to ensure that the host OS can support time-
    zone aware date-time handling. This is accomplished by merely training a
    Trainee with a date feature.

    In some configurations, this may result in a SegFault, so, we need to
    isolate the critical part of this test into another process.

    Parameters
    ----------
    registry : InstallationCheckRegistry
        The InstallationCheckRegistry instance.

    Returns
    -------
    tuple
        Status
            The status of the check as OK, WARNING, ERROR or CRITICAL.
        str
            A message to display about the WARNING, ERROR or CRITICAL result.
    """
    try:
        # If the host OS does not have timezone support, simply creating and
        # training on a date/time feature will SegFault. So, we do this in a
        # spawned process so we can detect this in the main thread.
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue(maxsize=-1)
        proc = ctx.Process(target=_attempt_train_date_feature,
                           args=(result_queue, ))
        proc.start()
        proc.join(timeout=DATE_FEATURE_TIMEOUT)
        if proc.is_alive():
            # The child neither finished nor crashed, so don't let it block the
            # rest of the verification run.
            proc.kill()
            proc.join()
            return (
                Status.CRITICAL,
                (
                    f"Timed out after {DATE_FEATURE_TIMEOUT} seconds checking "
                    "date/time support. Please ensure that the host OS has "
                    "timezone support."
                )
            )
        # The child enqueues its result only once training succeeds. If it died
        # instead (e.g. SegFault), the queue is empty and this raises
        # `queue.Empty`, which the handler below reports.
        result_queue.get(block=False)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=registry.logger)
        return (
            Status.CRITICAL,
            (
                "Unable to work with date/times. Please ensure that the host "
                "OS has timezone support."
            )
        )
    else:
        return (Status.OK, "")


def configure(registry: InstallationCheckRegistry) -> None:
    """
    Register the correct checks for the install environment.

    Parameters
    ----------
    registry : InstallationCheckRegistry
        The InstallationCheckRegistry instance.
    """
    registry.add_check(
        name="Howso Client: Visible CPUs",
        fn=check_visible_cores,
        client_required="AbstractHowsoClient",
    )

    registry.add_check(
        name="Howso Local: Timezone support",
        fn=check_tzdata_installed,
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Client: Basic react",
        fn=partial(check_generate_dataframe, threshold=10.0),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Local: Basic react",
        fn=partial(check_generate_dataframe, threshold=1.0),
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Client: Network latency",
        fn=partial(check_latency, notice_threshold=25,
                   warning_threshold=20),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Client: System performance",
        fn=partial(check_performance, num_samples=2_000,
                   notice_threshold=15.0, warning_threshold=20.0),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Client: Overridable resources",
        fn=partial(overridable_resources),
        client_required="HowsoPlatformClient",
    )

    registry.add_check(
        name="Howso Local: System performance",
        fn=partial(check_performance, num_samples=5_000,
                   notice_threshold=10.0, warning_threshold=20.0),
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Local: Save Trainee",
        fn=check_save,
        client_required="HowsoDirectClient",
    )

    registry.add_check(
        name="Howso Engine™: Basic operations",
        fn=check_engine_operation,
        client_required="AbstractHowsoClient",
        other_requirements=[engine],
    )

    registry.add_check(
        name="Howso Synthesizer™: Supported system locales",
        fn=check_locales_available,
        client_required="AbstractHowsoClient",
        other_requirements=[Synthesizer],
    )

    registry.add_check(
        name="Howso Synthesizer: Basic operations",
        fn=partial(check_synthesizer_create_delete),
        client_required="AbstractHowsoClient",
        other_requirements=[Synthesizer],
    )

    registry.add_check(
        name="Howso Synthesizer: Basic synthesis",
        fn=check_basic_synthesis,
        client_required="AbstractHowsoClient",
        other_requirements=[Synthesizer],
    )

    registry.add_check(
        name="Howso Validator: Basic operations",
        fn=check_validator_operation,
        client_required="AbstractHowsoClient",
        other_requirements=[Validator],
    )


def main() -> None:
    """Primary entry point."""
    iv_print("[bold]Validating Howso™ Installation")
    registry = InstallationCheckRegistry()

    with warnings.catch_warnings():
        configure(registry)
        warnings.simplefilter("ignore")
        result = registry.run_checks()
        if is_databricks():
            return
        sys.exit(result)


if __name__ == "__main__":
    main()
