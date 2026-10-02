from collections.abc import Callable, Iterator

import pytest

from howso.client.cache import TraineeCache
from howso.client.schemas import Trainee

ID_A = "00000000-0000-0000-0000-000000000001"
ID_B = "00000000-0000-0000-0000-000000000002"
ID_C = "00000000-0000-0000-0000-000000000003"


def iter_trainee_ids(cache: TraineeCache) -> Iterator[str]:
    """Return an iterator of ids drawn from ``TraineeCache.trainees``."""
    return (key for key, _ in cache.trainees())


def iter_ids(cache: TraineeCache) -> Iterator[str]:
    """Return an iterator over ``TraineeCache.ids``."""
    return iter(cache.ids())


def iter_item_ids(cache: TraineeCache) -> Iterator[str]:
    """Return an iterator of ids drawn from ``TraineeCache.items``."""
    return (key for key, _ in cache.items())


def iter_cache(cache: TraineeCache) -> Iterator[str]:
    """Return ``iter(cache)``."""
    return iter(cache)


ENUMERATORS: tuple[Callable[[TraineeCache], Iterator[str]], ...] = (
    iter_trainee_ids,
    iter_ids,
    iter_item_ids,
    iter_cache,
)


@pytest.fixture
def cache() -> TraineeCache:
    """Return a cache holding trainees ``a`` and ``b``."""
    trainee_cache = TraineeCache()
    trainee_cache.set(Trainee(name="a", id=ID_A))
    trainee_cache.set(Trainee(name="b", id=ID_B))
    return trainee_cache


def test_resolve_iteration_survives_insert() -> None:
    """Advancing a ``trainees()`` iterator after an insert does not raise."""
    trainee_cache = TraineeCache()
    trainee_cache.set(Trainee(name="a", id=ID_A))
    it = trainee_cache.trainees()
    next(it)
    trainee_cache.set(Trainee(name="b", id=ID_B))
    assert next(it, None) is None


@pytest.mark.parametrize("enumerate_cache", ENUMERATORS)
def test_enumeration_survives_insert(
    cache: TraineeCache, enumerate_cache: Callable[[TraineeCache], Iterator[str]]
) -> None:
    """An in-progress enumeration is a snapshot unaffected by ``set``."""
    it = enumerate_cache(cache)
    first = next(it)
    cache.set(Trainee(name="c", id=ID_C))
    assert sorted([first, *it]) == [ID_A, ID_B]


@pytest.mark.parametrize("enumerate_cache", ENUMERATORS)
def test_enumeration_survives_discard(
    cache: TraineeCache, enumerate_cache: Callable[[TraineeCache], Iterator[str]]
) -> None:
    """An in-progress enumeration is a snapshot unaffected by ``discard``."""
    it = enumerate_cache(cache)
    first = next(it)
    cache.discard(ID_A)
    cache.discard(ID_B)
    assert sorted([first, *it]) == [ID_A, ID_B]


@pytest.mark.parametrize("enumerate_cache", ENUMERATORS)
def test_enumeration_survives_clear(
    cache: TraineeCache, enumerate_cache: Callable[[TraineeCache], Iterator[str]]
) -> None:
    """An in-progress enumeration is a snapshot unaffected by ``clear``."""
    it = enumerate_cache(cache)
    first = next(it)
    cache.clear()
    assert sorted([first, *it]) == [ID_A, ID_B]
    assert len(cache) == 0


def test_trainees_snapshot_taken_at_call(cache: TraineeCache) -> None:
    """``trainees()`` captures the cache contents when it is called."""
    it = cache.trainees()
    cache.set(Trainee(name="c", id=ID_C))
    assert sorted(str(trainee.name) for _, trainee in it) == ["a", "b"]


def test_snapshots_are_lists(cache: TraineeCache) -> None:
    """``ids()`` and ``items()`` return independent lists."""
    ids = cache.ids()
    items = cache.items()
    assert isinstance(ids, list)
    assert isinstance(items, list)
    cache.clear()
    assert sorted(ids) == [ID_A, ID_B]
    assert sorted(key for key, _ in items) == [ID_A, ID_B]


def test_new_cache_is_empty() -> None:
    """A new cache has no entries, including no internal attributes."""
    trainee_cache = TraineeCache()
    assert len(trainee_cache) == 0
    assert list(trainee_cache) == []
    assert trainee_cache.ids() == []
    assert str(trainee_cache) == "{}"


def test_set_creates_entry(cache: TraineeCache) -> None:
    """A newly set trainee has unset feature attributes."""
    item = cache.get_item(ID_A)
    assert item["trainee"].name == "a"
    assert item["feature_attributes"] is None
    assert cache.get(ID_A).name == "a"
    assert ID_A in cache
    assert len(cache) == 2


def test_set_merges_into_existing_entry(cache: TraineeCache) -> None:
    """Setting an existing id replaces the trainee and keeps other keys."""
    features = {"x": {"type": "continuous"}}
    cache.set(Trainee(name="a", id=ID_A), feature_attributes=features, revision=1)
    cache.set(Trainee(name="a2", id=ID_A))
    item = cache.get_item(ID_A)
    assert item["trainee"].name == "a2"
    assert item["feature_attributes"] == features
    assert dict(item)["revision"] == 1
    assert len(cache) == 2


def test_get_item_returns_shared_entry(cache: TraineeCache) -> None:
    """Changes to the entry returned by ``get_item`` are visible in the cache."""
    features = {"x": {"type": "nominal"}}
    cache.get_item(ID_A)["feature_attributes"] = features
    assert cache.get_item(ID_A)["feature_attributes"] == features


def test_missing_key_behavior(cache: TraineeCache) -> None:
    """Missing ids raise ``KeyError`` unless a default is supplied."""
    with pytest.raises(KeyError):
        cache.get(ID_C)
    with pytest.raises(KeyError):
        cache.get_item(ID_C)
    assert cache.get(ID_C, None) is None
    assert cache.get_item(ID_C, None) is None
    assert ID_C not in cache
    cache.discard(ID_C)
    assert len(cache) == 2


def test_str_lists_entries(cache: TraineeCache) -> None:
    """``str()`` renders the entries keyed by trainee id."""
    rendered = str(cache)
    assert ID_A in rendered
    assert ID_B in rendered
