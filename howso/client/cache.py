from __future__ import annotations

from collections.abc import Collection, Iterator
import threading
import typing as t

from semantic_version import Version
from typing_extensions import NotRequired

if t.TYPE_CHECKING:
    from howso.client.schemas import Trainee


class TraineeCacheItem(t.TypedDict):
    """Type definition for trainee cache items."""

    trainee: Trainee
    """Trainee object."""

    feature_attributes: dict[str, dict] | None
    """Trainee's feature attributes."""

    version: NotRequired[Version]
    """Version of the Trainee."""


class TraineeCache(Collection):
    """
    Thread-safe cache of trainee related information, keyed by trainee id.

    A single cache may be shared by clients running on several threads. All
    access to the stored entries is serialized by a lock owned by the cache.
    The enumeration methods (``ids``, ``items``, ``trainees`` and iteration
    over the cache) return snapshots taken under that lock, so concurrent
    ``set``, ``discard`` and ``clear`` calls never disturb an enumeration in
    progress.

    The ``TraineeCacheItem`` dictionaries returned by ``get_item`` and
    ``items`` are the cache's own entries, not copies; changes made to them
    are visible to every holder of the cache.
    """

    __slots__ = ("_items", "_lock")

    __marker = object()

    def __init__(self) -> None:
        """Initialize an empty cache."""
        # Entries hold the TraineeCacheItem keys plus any extra keys callers
        # pass to ``set`` (for example a platform ``revision``).
        self._items: dict[str, dict[str, t.Any]] = {}
        self._lock = threading.Lock()

    def set(self, trainee: Trainee, **kwargs) -> None:
        """
        Set trainee in cache.

        A trainee id not yet in the cache gets a new entry whose
        ``feature_attributes`` is ``None``. For an id already in the cache,
        the entry's ``trainee`` is replaced and ``kwargs`` are merged into
        the existing entry, keeping any keys not named in ``kwargs``.

        Parameters
        ----------
        trainee : Trainee
            The trainee to cache. Trainees without an id are ignored.
        **kwargs
            Additional entry keys to store alongside the trainee.
        """
        if trainee.id:
            with self._lock:
                item = self._items.setdefault(trainee.id, {"feature_attributes": None})
                item.update({"trainee": trainee, **kwargs})

    def get(self, trainee_id: str, default=__marker) -> Trainee:
        """Get trainee instance by id."""
        with self._lock:
            try:
                return self._items[trainee_id]["trainee"]
            except KeyError:
                if default is self.__marker:
                    raise
                return default

    def get_item(self, trainee_id: str, default=__marker) -> TraineeCacheItem:
        """Get trainee cache item by id."""
        with self._lock:
            try:
                # Entries always carry the TraineeCacheItem keys because ``set``
                # creates them; extra caller-supplied keys are why storage is
                # typed as a plain dict.
                return t.cast(TraineeCacheItem, self._items[trainee_id])
            except KeyError:
                if default is self.__marker:
                    raise
                return default

    def discard(self, trainee_id: str) -> None:
        """Remove trainee from cache if exists."""
        with self._lock:
            self._items.pop(trainee_id, None)

    def ids(self) -> list[str]:
        """Return a snapshot list of the ids in the cache."""
        with self._lock:
            return list(self._items)

    def items(self) -> list[tuple[str, TraineeCacheItem]]:
        """Return a snapshot list of the ``(id, item)`` pairs in the cache."""
        with self._lock:
            # See ``get_item`` for why entries are cast to TraineeCacheItem.
            return [(key, t.cast(TraineeCacheItem, item)) for key, item in self._items.items()]

    def trainees(self) -> Iterator[tuple[str, Trainee]]:
        """Return an iterator over a snapshot of the ``(id, trainee)`` pairs in the cache."""
        with self._lock:
            snapshot = [(key, item["trainee"]) for key, item in self._items.items()]
        return iter(snapshot)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._items.clear()

    def __contains__(self, key: object) -> bool:
        """Return if trainee id is in cache."""
        with self._lock:
            return key in self._items

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over a snapshot of the cached trainee ids."""
        with self._lock:
            snapshot = list(self._items)
        return iter(snapshot)

    def __len__(self) -> int:
        """Return length of the cache."""
        with self._lock:
            return len(self._items)

    def __str__(self) -> str:
        """Return string representation of the cache."""
        with self._lock:
            snapshot = dict(self._items)
        return str(snapshot)
