"""Utilities to aide in testing `howso-engine`."""
from __future__ import annotations

from collections.abc import Callable
import os
from unittest.mock import patch

from howso.client.base import AbstractHowsoClient
from howso.direct import HowsoDirectClient


def get_test_options():
    """
    Simply parses the ENV variable 'TEST_OPTIONS' into a list, if possible
    and returns it. This will be used with `pytest.skipif` to conditionally
    test some additional tests.

    Example:
        >>> from . import get_test_options
        >>> ...
        >>> @pytest.mark.skipif('FOO' not in get_test_options, reason='FOO not in ENV')  # noqa
        >>> def test_bar(...):
        >>>     ...

    Returns
    -------
    list[str]
    """
    try:
        options = os.getenv('TEST_OPTIONS').split(',')
    except (AttributeError, ValueError):
        options = []
    return options


def get_configurationless_test_client(
    client_class: type[AbstractHowsoClient] | Callable = HowsoDirectClient,
    **kwargs
) -> AbstractHowsoClient:
    """
    Return a client for use within testing.

    By default, this will instantiate a default HowsoDirectClient with no
    passed configuration nor any of the special environment variables. If the
    TEST_OPTIONS environment variable contains "USE_HOWSO_CONFIG", then the
    client will resume normal behavior and use a configuration found in the
    normal places or as defined by the `HOWSO_CONFIG` environment variable.

    Parameters
    ----------
    client_class : AbstractHowsoClient or Callable
        A subclass of AbstractHowsoClient or the HowsoClient callable.

    Returns
    -------
    AbstractHowsoClient
        An instance of a subclass of AbstractHowsoClient.
    """
    if "USE_HOWSO_CONFIG" in get_test_options():
        return client_class(**kwargs)
    else:
        # Ignore any locally defined config (howso.yml) files.
        names_to_remove = ('XDG_CONFIG_HOME', 'HOWSO_CONFIG')
        # And the HOWSO_CONFIG and
        modified_environ = {
            k: v for k, v in os.environ.items()
            if k not in names_to_remove
        }

        if kwargs.get("trace"):
            amlg_options = kwargs.get("amalgam", {})
            amlg_options.setdefault("execution_trace_dir", "./traces")
            kwargs["amalgam"] = amlg_options

        with (patch('howso.client.client.get_configuration_path') as mocked_fn,
                patch.dict('os.environ', modified_environ, clear=True)):
            mocked_fn.return_value = None
            return client_class(**kwargs)
