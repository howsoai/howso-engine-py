from pathlib import Path

import pytest

from amalgam.api import Amalgam
from howso.client.exceptions import HowsoError
from howso.direct import HowsoDirectClient
from howso.utilities.testing import get_configurationless_test_client


@pytest.fixture
def client(tmp_path: Path):
    """Direct client instance using latest binaries."""
    return get_configurationless_test_client(client_class=HowsoDirectClient,
                                             verbose=True, trace=True, default_persist_path=tmp_path)


def test_direct_client(client: HowsoDirectClient):
    """Sanity check client instantiation."""
    assert isinstance(client.amlg, Amalgam)
    version = client.get_version()
    assert version.get('client') is not None


@pytest.mark.parametrize(('filename', 'truthiness'), (
    ('./banana.txt', True),
    ('./ba\nana.txt', True),
    ('./bañän🤣a.txt', True),
    ('./banana\0.txt', False),
))
def test_check_name_valid_for_save(client, filename, truthiness):
    """Ensure that the internal function `check_name_valid_for_save` works."""
    assert client.check_name_valid_for_save(filename, clobber=True)[0] == truthiness


@pytest.mark.parametrize(
    ("trainee_id", "trainee_name", "truthiness"),
    (
        (None, "banana", True),
        (None, "ba\nana", True),
        (None, "bañän🤣a", True),
        (None, "abc/test", False),
        (None, "abc:test", False),
        (None, "abc\0test", False),
        ("foo", "abc/test", True),
        ("foo", "abc:test", True),
        ("foo", "abc", True),
        ("abc:test", "foo", False),
        ("abc/test", "foo", False),
    ),
)
def test_trainee_valid_for_save(client, trainee_id, trainee_name, truthiness):
    """Ensure trainee filename validation applies to id or name as expected."""
    if truthiness:
        try:
            trainee = client.create_trainee(id=trainee_id, name=trainee_name)
            assert trainee is not None
        finally:
            client.delete_trainee(trainee_id or trainee_name)
    else:
        with pytest.raises(HowsoError, match="Trainee file name"):
            trainee = client.create_trainee(id=trainee_id, name=trainee_name)


def test_persistence_always(client: HowsoDirectClient, tmp_path: Path):
    """Test that persist-always mode creates a file on disk."""
    trainee = client.create_trainee(persistence='always')
    trainee_path = tmp_path / f"{trainee.id}.caml"
    client.set_feature_attributes(trainee.id, {"x": {"type": "continuous"}})
    assert trainee_path.exists()


def test_persistence_always_file_size(client: HowsoDirectClient, tmp_path: Path):
    """Test the file size in persist-always mode."""
    # More specifically: since this now runs in transactional mode, we expect
    # each action to make the file incrementally larger, until the library
    # chooses to compact.  But it should never reach 2x the original file
    # size.
    trainee = client.create_trainee(persistence='always')
    trainee_path = tmp_path / f"{trainee.id}.caml"
    client.set_feature_attributes(trainee.id, {"x": {"type": "continuous"}})
    assert trainee_path.exists()
    base_size = trainee_path.stat().st_size
    old_size = base_size
    while True:
        client.train(trainee.id, [[0.0]], ['x'])
        new_size = trainee_path.stat().st_size
        assert new_size < 2 * base_size
        if new_size < old_size:
            break  # the library has compacted
        old_size = new_size


def test_load_subtrainee_from_memory(client: HowsoDirectClient, tmp_path: Path) -> None:
    """Test loading a parent and child trainee both from memory."""
    # Regression test for #24706
    features = { "x": { "type": "continuous" }}

    t1 = client.create_trainee(features=features)
    client.train(t1.id, cases=[[1]], features=["x"])
    client.persist_trainee(t1.id)
    caml1 = (tmp_path / f"{t1.id}.caml").read_bytes()
    client.delete_trainee(t1.id)

    t2 = client.create_trainee(features=features)
    client.train(t2.id, cases=[[2]], features=["x"], start_index=1)
    client.persist_trainee(t2.id)
    caml2 = (tmp_path / f"{t2.id}.caml").read_bytes()
    client.delete_trainee(t2.id)

    client.create_trainee_from_memory("test", caml1, file_type="caml")
    try:
        client.create_trainee_from_memory("test", caml2, file_type="caml", path=[])
        client.combine_trainee_with_subtrainees("test")

        df = client.get_cases("test")
        assert df["features"] == ["x"]
        assert df["cases"] == [[1], [2]]
    finally:
        client.delete_trainee("test")
