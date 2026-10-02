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
    access to the cache's set of entries is serialized by a lock owned by the
    cache.

    The enumeration methods (``ids``, ``items``, ``trainees`` and iteration
    over the cache) snapshot *which* entries exist, under that lock.
    Concurrent ``set``, ``discard`` and ``clear`` calls never disturb an
    enumeration in progress, and an enumeration does not reflect entries
    added or removed after it was taken.

    The entries themselves are not snapshotted. Each ``TraineeCacheItem``
    returned by ``get_item`` or ``items`` is the cache's own live entry:
    assigning to one of its keys updates the cache for every holder, and a
    later ``set`` for the same trainee id updates a dictionary already handed
    out. The lock does not cover reads or writes made through an entry.
    """

    __slots__ = ("_items", "_lock")

    __marker = object()

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._items: dict[str, TraineeCacheItem] = {}
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
                item = self._items.setdefault(trainee.id, {"trainee": trainee, "feature_attributes": None})
                item["trainee"] = trainee
                # kwargs may carry keys TraineeCacheItem does not declare
                # (for example a platform ``revision``), so they are assigned
                # one at a time.
                for key, value in kwargs.items():
                    item[key] = value

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
        """
        Get the cache entry for a trainee, by id.

        The returned dictionary is the cache's own live entry, not a copy.
        Assigning to its keys (for example ``feature_attributes``) changes
        what the cache holds for this trainee, and a later ``set`` call for
        the same id changes this dictionary. Once the trainee is discarded,
        the dictionary is detached from the cache: it keeps its contents, and
        a subsequent ``set`` for that id creates a new entry.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        default : optional
            The value to return if the trainee is not in the cache. If not
            provided, a missing trainee raises ``KeyError``.

        Returns
        -------
        TraineeCacheItem
            The trainee's live cache entry, or ``default``.

        Raises
        ------
        KeyError
            If the trainee is not in the cache and no ``default`` is given.
        """
        with self._lock:
            try:
                return self._items[trainee_id]
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
        """
        Return a list of the ``(id, item)`` pairs in the cache.

        The list is a snapshot of which trainees are cached, but each
        ``item`` is the cache's own live entry, as returned by ``get_item``.
        """
        with self._lock:
            return list(self._items.items())

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
