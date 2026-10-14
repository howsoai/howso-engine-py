"""Tests for identifying a client's multi-threaded Amalgam build."""
from __future__ import annotations

from collections.abc import Generator

import pytest

from howso.client.base import (
    _is_multithreaded_library_type,  # pyright: ignore[reportPrivateUsage]
    AbstractHowsoClient,
)
from howso.direct.client import (
    _is_multithreaded_concurrency,  # pyright: ignore[reportPrivateUsage]
    HowsoDirectClient,
)
from howso.engine import Trainee


@pytest.fixture(name="trainee")
def trainee_fixture() -> Generator[Trainee, None, None]:
    """Provide a minimal Trainee on the default direct client, deleted afterward."""
    t = Trainee(features={"x": {"type": "continuous"}})
    try:
        yield t
    finally:
        t.delete()


@pytest.mark.parametrize(("library_type", "expected"), [
    ("mt", True),
    (" mt ", True),
    # A multi-threaded variant is multi-threaded, mirroring how "st-omp" exists.
    ("mt-omp", True),
    # The value is sometimes carried in Amalgam library postfix form.
    ("-mt", True),
    (" -mt ", True),
    ("st", False),
    # The OpenMP build is single-threaded despite advertising several threads.
    ("st-omp", False),
    ("-st", False),
    ("-st-omp", False),
    (None, False),
    # An unrecognized value is not multi-threaded just because it starts "mt".
    ("mtx", False),
    ("mt_single", False),
    ("", False),
])
def test_is_multithreaded_library_type(library_type: str | None, expected: bool) -> None:
    assert _is_multithreaded_library_type(library_type) is expected


@pytest.mark.parametrize(("concurrency", "expected"), [
    (b"MultiThreaded", True),
    (b"SingleThreaded", False),
    (b"SingleThreaded+OpenMP", False),
    ("MultiThreaded", True),  # a str return is understood too
    (b"  MultiThreaded  ", True),
    (b"", False),
])
def test_is_multithreaded_concurrency(concurrency: bytes | str, expected: bool) -> None:
    assert _is_multithreaded_concurrency(concurrency) is expected


def test_direct_client_reports_its_library_build(trainee: Trainee) -> None:
    """The direct client identifies the build of the library it loaded."""
    client = trainee.client
    assert isinstance(client, HowsoDirectClient)
    expected = trainee.get_runtime()["library_type"] == "mt"
    assert client.is_multithreaded_build(trainee.id) is expected


def test_direct_client_answer_does_not_depend_on_trainee(trainee: Trainee) -> None:
    """A direct client's build is a property of the process, not of a Trainee."""
    client = trainee.client
    assert isinstance(client, HowsoDirectClient)
    assert client.is_multithreaded_build("no-such-trainee") is client.is_multithreaded_build(trainee.id)


def test_base_implementation_reads_runtime_library_type(trainee: Trainee) -> None:
    """The inherited implementation answers from ``get_trainee_runtime``."""
    client = trainee.client
    assert isinstance(client, AbstractHowsoClient)
    expected = trainee.get_runtime()["library_type"] == "mt"
    assert AbstractHowsoClient.is_multithreaded_build(client, trainee.id) is expected
