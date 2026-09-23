from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import cached_property
import inspect
from io import StringIO
from pathlib import Path
import sys
import traceback
from typing import IO, TYPE_CHECKING

from requests.exceptions import ConnectionError as RequestsConnectionError
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from howso.client import AbstractHowsoClient, HowsoClient
from howso.client.exceptions import HowsoConfigurationError
from howso.utilities import auto_progress_scope, StopExecution

from ._constants import LOG_FILE
from ._helpers import get_versions
from ._output import _console_safe, is_databricks, iv_print
from ._types import Check, Status
from .checks_cpu import check_cpu_steal, check_low_priority_compute, check_usable_cpus
from .checks_environment import check_not_emulated

if TYPE_CHECKING:
    from ._types import CheckFunction, Requirements


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
        # Windows may give lower-than-normal priority processes less CPU
        self.add_check(
            name="Python: Low-priority CPU access",
            fn=check_low_priority_compute
        )
        # An affinity mask or processor group may hide CPUs from this process
        self.add_check(
            name="Python: Usable CPUs",
            fn=check_usable_cpus
        )
        # An oversubscribed host takes CPU time from its guests
        self.add_check(
            name="Python: CPU steal time",
            fn=check_cpu_steal
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
            # The checks drive client operations that render progress of their
            # own; nesting that inside this one's live display garbles both.
            with auto_progress_scope(enabled=False), progress:
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
                        progress.console.print(_console_safe(
                            f"[bold]{check.name:{self._name_length}s} - "
                            f"[{color}]{status.name} {emoji} - {msg}"))
                    else:
                        progress.console.print(_console_safe(
                            f"[bold]{check.name:{self._name_length}s} - "
                            f"[{color}]{status.name} {emoji}"))

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
            iv_print("[bold green]You are ready to use Howso®!")
        else:
            iv_print("[bold yellow]There were one or more issues. Please review "
                     "the messages emitted during the installation verification "
                     "process to identify next steps. If you cannot resolve "
                     "these issues please do not hesitate to contact your "
                     "Howso® representative.")
            iv_print(f'[bold yellow]Any CRITICAL issues are logged in the file '
                     f'"{LOG_FILE}" in the current directory.')

        # This is largely for automated systems.
        if critical_issues:
            return 255
        return 0
