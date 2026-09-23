from __future__ import annotations

import importlib.metadata
import random
import sys

from howso.client import AbstractHowsoClient, HowsoClient
from howso.client.client import get_howso_client_class
from howso.direct.client import HowsoDirectClient

try:
    from howso.platform import HowsoPlatformClient  # noqa: might not be available # type: ignore[reportMissingImports]
except ImportError:
    HowsoPlatformClient = None


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


def _is_platform_client(client: type[AbstractHowsoClient] | AbstractHowsoClient) -> bool:
    """Check if a client is a platform client type or instance."""
    if HowsoPlatformClient is None:
        return False
    if isinstance(client, type) and issubclass(client, HowsoPlatformClient):
        return True
    return isinstance(client, HowsoPlatformClient)
