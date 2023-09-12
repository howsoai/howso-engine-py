"""
A convenient wrapper for pysimdjson.

json_wrapper module provides an interface either, the standard module `json`
built-into Python or the simdjson "drop-in replacement API", depending one what
is installed.

The `pysimdjson` package must be installed separately.
"""
import json
from typing import Any

try:
    import simdjson  # noqa
except ImportError:
    simdjson = None


def detect_encoding(b):
    """
    Detect encoring.

    Always pass-thru to built-in `json` module for detect_encoding().
    """
    return json.detect_encoding(b)


def dump(*args, **kwargs):
    """
    Dump object to JSON.

    Always pass-thru to built-in `json` module for dump().
    """
    kwargs.setdefault('allow_nan', False)
    return json.dump(*args, **kwargs)


def dumps(*args, **kwargs):
    """
    Dumpy object to JSON string.

    Always pass-thru to built-in `json` module for dumps().
    """
    kwargs.setdefault('allow_nan', False)
    return json.dumps(*args, **kwargs)


def load(fp, *, object_hook=None, **kwargs) -> Any:
    """
    Use the fastest available `load` for JSON based on kwargs given.

    For simple JSON loading, implement via simdjson, which offers ~2X
    performance vs. Python's native `json.load()`. If any keyword parameters
    are supplied except `object_hook` or simdjson is not available then default
    to the native implementation.
    """
    if simdjson and len(kwargs) == 0:
        return simdjson.load(fp, object_hook=object_hook)
    else:
        return json.load(fp, object_hook=object_hook, **kwargs)


def loads(s, *, object_hook=None, **kwargs) -> Any:
    """
    Use the fastest available `loads` for JSON based on kwargs given.

    For simple JSON loading, implement via simdjson, which offers ~2X
    performance vs. Python's native `json.loads()`. If any keyword parameters
    are supplied except `object_hook` or simdjson is not available then default
    to the native implementation.
    """
    if simdjson and len(kwargs) == 0:
        return simdjson.loads(s, object_hook=object_hook)
    else:
        return json.loads(s, object_hook=object_hook, **kwargs)
