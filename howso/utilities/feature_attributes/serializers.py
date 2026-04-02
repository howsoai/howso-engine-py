"""JSON serialization utilities for FeatureAttributesBase."""

from __future__ import annotations

from ast import literal_eval
import json
import typing as t

import numpy as np

_TUPLE_KEY_PREFIX = "__tuple_key__:"


__all__ = [
    "FeatureAttributesEncoder",
    "feature_attributes_pairs_hook",
]


class FeatureAttributesEncoder(json.JSONEncoder):
    """
    JSON encoder that preserves tuples and handles numpy types.

    Tuples are encoded with special markers so they can be distinguished
    from lists and restored on decode. Dict keys that are tuples are
    prefixed with a sentinel string; tuple values are wrapped in a
    ``{"__tuple__": [...]}`` envelope.

    Numpy scalar types (``np.integer``, ``np.floating``, ``np.bool_``)
    are converted to their native Python equivalents. Numpy arrays are
    converted to lists.
    """

    def encode(self, o: t.Any) -> str:
        """
        Encode the given object to a JSON-formatted string.

        Parameters
        ----------
        o : Any
            The object to serialize.

        Returns
        -------
        str
            A JSON-formatted string.
        """
        return super().encode(self._pre_process(o))

    def _pre_process(self, obj: t.Any) -> t.Any:
        """
        Recursively transform an object tree for JSON compatibility.

        Parameters
        ----------
        obj : Any
            The object (or sub-object) to transform.

        Returns
        -------
        Any
            A JSON-compatible equivalent of *obj*.
        """
        if isinstance(obj, dict):
            return {
                f"{_TUPLE_KEY_PREFIX}{k!r}" if isinstance(k, tuple) else k:
                    self._pre_process(v)
                for k, v in obj.items()
            }
        if isinstance(obj, tuple):
            return {"__tuple__": [self._pre_process(i) for i in obj]}
        if isinstance(obj, list):
            return [self._pre_process(i) for i in obj]
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


def feature_attributes_pairs_hook(pairs: list[tuple[str, t.Any]]) -> dict[t.Any, t.Any] | tuple:
    """
    Decode hook that restores tuples from their JSON representation.

    Intended for use as the ``object_pairs_hook`` parameter of
    ``json.loads`` / ``json.load``.

    Parameters
    ----------
    pairs : list of tuple of (str, Any)
        Key-value pairs produced by the JSON parser for a single object.

    Returns
    -------
    dict or tuple
        A ``tuple`` if the object is a ``{"__tuple__": [...]}`` envelope,
        otherwise a ``dict`` with any tuple-key sentinels decoded back to
        real tuples.
    """
    d: dict[t.Any, t.Any] = {}
    for k, v in pairs:
        if k == "__tuple__" and len(pairs) == 1:
            return tuple(v)
        if isinstance(k, str) and k.startswith(_TUPLE_KEY_PREFIX):
            k = literal_eval(k[len(_TUPLE_KEY_PREFIX):])
        d[k] = v
    return d
