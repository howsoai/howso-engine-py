"""
The Python API for the Howso Client.

The Howso Python Client API has two major components,

- client module:
    A basic client that implements the Howso REST API.
- scikit module:
    Implements a scikit-learn Estimator which uses the Howso
    cloud service to make predictions off of fit data.


Additional submodules are included in the package but are for internal client/scikit operations and thus are omitted
from the documentation.

Examples implementations are included in the howso/examples directory.
"""

from howso.client import typing  # noqa: F401
from howso.client.api import get_api  # noqa: F401
from howso.client.base import AbstractHowsoClient  # noqa: F401
from howso.client.client import (  # noqa: F401
    CONFIG_FILE_ENV_VAR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_CONFIG_FILE_ALT,
    get_configuration_path,
    get_howso_client,
    HowsoClient,
)
from howso.client.pandas.client import HowsoPandasClient  # noqa: F401

__all__ = [
    "AbstractHowsoClient",
    "CONFIG_FILE_ENV_VAR",
    "DEFAULT_CONFIG_FILE_ALT",
    "DEFAULT_CONFIG_FILE",
    "get_api",
    "get_configuration_path",
    "get_howso_client",
    "HowsoClient",
    "HowsoPandasClient",
    "typing",
]
