from collections.abc import Collection
import typing as t

if t.TYPE_CHECKING:
    from howso.openapi.models import Trainee


class TraineeCacheItem(t.TypedDict):
    """Type definition for trainee cache items."""

    trainee: "Trainee"
    user_defaults: t.Dict[str, t.Dict]


class TraineeCache(Collection):
    """Cache of trainee related information."""

    __slots__ = ('__dict__', )

    __marker = object()

    def set(self, trainee: "Trainee", **kwargs) -> None:
        """Set trainee in cache."""
        if trainee.id:
            self.__dict__.setdefault(trainee.id, {
                'user_defaults': {}
            })
            self.__dict__[trainee.id].update({
                'trainee': trainee,
                **kwargs
            })

    def get(self, trainee_id: str, default=__marker) -> "Trainee":
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

    def get_user_defaults(self, trainee_id: str, user_id: str
                          ) -> t.Dict[str, t.Dict]:
        """Get user defaults for trainee."""
        user_defaults = self.__dict__[trainee_id]['user_defaults']
        try:
            return user_defaults[user_id]
        except KeyError:
            return user_defaults.setdefault(user_id, {})

    def discard(self, trainee_id: str) -> None:
        """Remove trainee from cache if exists."""
        try:
            del self.__dict__[trainee_id]
        except KeyError:
            pass

    def ids(self) -> t.Iterable[str]:
        """Return view of ids in cache."""
        return self.__dict__.keys()

    def items(self) -> t.Iterable[t.Tuple[str, TraineeCacheItem]]:
        """Return view items in cache."""
        return self.__dict__.items()

    def trainees(self) -> t.Iterator[t.Tuple[str, "Trainee"]]:
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

    def __iter__(self) -> t.Iterator[str]:
        """Return iterator of the cached trainee ids."""
        return iter(self.__dict__)

    def __len__(self) -> int:
        """Return length of the cache."""
        return len(self.__dict__)

    def __str__(self) -> str:
        """Return string representation of the cache."""
        return str(self.__dict__)
