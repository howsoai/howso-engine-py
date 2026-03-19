from __future__ import annotations

from ast import literal_eval
import json

_TUPLE_KEY_PREFIX = "__tuple_key__:"


__all__ = [
    "TupleAwareEncoder",
    "tuple_aware_object_pairs_hook",
]


class TupleAwareEncoder(json.JSONEncoder):
    """JSON encoder that preserves tuples (both as dict keys and values)."""

    def encode(self, o):
        return super().encode(self._pre_process(o))

    def _pre_process(self, obj):
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
        return obj


def tuple_aware_object_pairs_hook(pairs: list[tuple]) -> dict | tuple:
    """Decode hook that restores tuples from their JSON representation."""
    d = {}
    for k, v in pairs:
        if k == "__tuple__" and len(pairs) == 1:
            return tuple(v)
        if isinstance(k, str) and k.startswith(_TUPLE_KEY_PREFIX):
            k = literal_eval(k[len(_TUPLE_KEY_PREFIX):])
        d[k] = v
    return d
