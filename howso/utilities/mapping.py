from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
import typing as t

if t.TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

_VT = t.TypeVar("_VT")


class CaseInsensitiveMap(MutableMapping[str, _VT]):
    """A case insensitive mutable mapping."""

    def __init__(
        self,
        map: t.Optional[SupportsKeysAndGetItem[str, _VT] | Iterable[tuple[str, _VT]]] = None,
        **kwargs: _VT
    ):
        super().__init__()
        self._store = dict[str | None, tuple[str, _VT]]()
        if map is None:
            map = {}
        self.update(map, **kwargs)

    def __setitem__(self, key: str, value: _VT):
        """Set item in mapping."""
        # Use the parsed key for lookups, but store the actual
        # key alongside the value.
        self._store[self.parse_key(key)] = (key, value)

    def __getitem__(self, key: str):
        """Get item from mapping."""
        return self._store[self.parse_key(key)][1]

    def __delitem__(self, key: str):
        """Delete key from mapping."""
        del self._store[self.parse_key(key)]

    def __iter__(self):
        """Iterate over mapping keys."""
        return (orig_key for orig_key, _ in self._store.values())

    def __len__(self):
        """Get mapping length."""
        return len(self._store)

    def __eq__(self, other: object):
        """Compare against another object."""
        if isinstance(other, Mapping):
            other = CaseInsensitiveMap(other)
        else:
            return False
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    def __repr__(self):
        """Implement repr magic method."""
        return str(dict(self.items()))

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return ((key, val[1]) for (key, val) in self._store.items())

    def copy(self):
        """Copy mapping."""
        return CaseInsensitiveMap(self._store.values())

    @classmethod
    def parse_key(cls, key: str | None):
        """Parse the key."""
        if key is None:
            return None
        return key.lower()
