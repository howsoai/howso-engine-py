from pathlib import Path

import pytest

from amalgam.api import Amalgam
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
))
def test_check_name_valid_for_save(client, filename, truthiness):
    """Ensure that the internal function `check_name_valid_for_save` works."""
    assert client.check_name_valid_for_save(filename, clobber=True)[0] == truthiness


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
