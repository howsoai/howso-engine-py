from __future__ import annotations

import multiprocessing
import sys
import traceback
from typing import TYPE_CHECKING

from faker.config import AVAILABLE_LOCALES

from howso.client import HowsoClient
from howso.utilities import auto_progress_scope
from howso.utilities.locale import get_default_locale
from howso.utilities.posix import PlatformError, sysctl_by_name

from ._constants import DATE_FEATURE_TIMEOUT
from ._helpers import _is_platform_client, get_nonce
from ._types import Status

if TYPE_CHECKING:
    from .registry import InstallationCheckRegistry


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


def _attempt_train_date_feature(result_queue: multiprocessing.Queue[int]) -> None:
    """
    Attempt to train a date feature to check for proper time zone support.

    Parameters
    ----------
    result_queue : A multiprocessing queue instance
        A queue to put the results.
    """
    # A spawned process does not inherit the caller's thread-local progress
    # setting, so disable it here too. This trains a Trainee, and its progress
    # would otherwise render into the parent's live display.
    with auto_progress_scope(enabled=False):
        client = HowsoClient()
        features = {"date": {"type": "continuous", "date_time_format": "%Y-%m-%d"}}
        trainee = client.create_trainee(
            name=f"installation_verification check_tzdata_installed ({get_nonce()})",
            features=features,
            persistence="allow" if _is_platform_client(client) else "never"
        )
        try:
            client.train(trainee_id=trainee.id, cases=[["2001-01-01"]],
                         features=["date"])
            result_queue.put(client.get_num_training_cases(trainee.id))
        finally:
            client.delete_trainee(trainee.id)


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
