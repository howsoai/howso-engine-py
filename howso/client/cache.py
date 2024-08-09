from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
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
    """Cache of trainee related information."""

    __slots__ = ('__dict__', )

    __marker = object()

    def set(self, trainee: Trainee, **kwargs) -> None:
        """Set trainee in cache."""
        if trainee.id:
            self.__dict__.setdefault(trainee.id, {
                'feature_attributes': None
            })
            self.__dict__[trainee.id].update({
                'trainee': trainee,
                **kwargs
            })

    def get(self, trainee_id: str, default=__marker) -> Trainee:
        """Get trainee instance by id."""
        try:
            return self.__dict__[trainee_id]['trainee']
        except KeyError:
            if default is self.__marker:
                raise
            return default

    def get_item(self, trainee_id: str, default=__marker) -> TraineeCacheItem:
        """Get trainee cache item by id."""
        try:
            return self.__dict__[trainee_id]
        except KeyError:
            if default is self.__marker:
                raise
            return default

    def discard(self, trainee_id: str) -> None:
        """Remove trainee from cache if exists."""
        try:
            del self.__dict__[trainee_id]
        except KeyError:
            pass

    def ids(self) -> Iterable[str]:
        """Return view of ids in cache."""
        return self.__dict__.keys()

    def items(self) -> Iterable[tuple[str, TraineeCacheItem]]:
        """Return view items in cache."""
        return self.__dict__.items()

    def trainees(self) -> Iterator[tuple[str, Trainee]]:
        """Return iterator to all trainee instances in cache."""
        for key, item in self.__dict__.items():
            yield (key, item['trainee'])

    def clear(self) -> None:
        """Clear the cache."""
        self.__dict__.clear()

    def __contains__(self, key: str) -> bool:
        """Return if trainee id is in cache."""
        try:
            self.__dict__[key]
        except KeyError:
            return False
        else:
            return True

    def __iter__(self) -> Iterator[str]:
        """Return iterator of the cached trainee ids."""
        return iter(self.__dict__)

    def __len__(self) -> int:
        """Return length of the cache."""
        return len(self.__dict__)

    def __str__(self) -> str:
        """Return string representation of the cache."""
        return str(self.__dict__)
