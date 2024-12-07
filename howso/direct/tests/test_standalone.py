from pathlib import Path

import pytest

from amalgam.api import Amalgam
from howso.direct import HowsoDirectClient
from howso.direct.schemas.trainee import TraineeDirectRuntimeOptions
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
    client.set_feature_attributes(trainee.id, {"f": {"type": "nominal"}})
    assert trainee_path.exists()


@pytest.mark.skip(reason="compressed file sizes are not so predictable")
def test_persistence_always_shrinks(client: HowsoDirectClient, tmp_path: Path):
    """Test that persist-always mode rewrites a file to maybe be smaller."""
    trainee = client.create_trainee(persistence='always')
    trainee_path = tmp_path / f"{trainee.id}.caml"
    client.set_feature_attributes(trainee.id, {"feature_1": {"type": "nominal"},
                                               "other_unrelated_feature": {"type": "continuous"},
                                               "yet_another_feature": {"type": "continuous"}})
    old_size = trainee_path.stat().st_size
    client.set_feature_attributes(trainee.id, {"feature_1": {"type": "nominal"}})
    new_size = trainee_path.stat().st_size
    # We've deleted two features so the file should be smaller
    # (Note that this is not 100% guaranteed due to the compression)
    assert new_size < old_size


def test_persistence_always_transactional_grows(client: HowsoDirectClient, tmp_path: Path):
    """Test that transactional mode makes a file larger."""
    trainee = client.create_trainee(persistence='always', runtime=TraineeDirectRuntimeOptions(transactional=True))
    trainee_path = tmp_path / f"{trainee.id}.caml"
    client.set_feature_attributes(trainee.id, {"feature_1": {"type": "nominal"},
                                               "other_unrelated_feature": {"type": "continuous"}})
    old_size = trainee_path.stat().st_size
    client.set_feature_attributes(trainee.id, {"feature_1": {"type": "nominal"}})
    # Transactional mode always makes the file larger
    new_size = trainee_path.stat().st_size
    assert new_size > old_size
    client.persist_trainee(trainee.id)
    # But now saving should compact the file
    new_new_size = trainee_path.stat().st_size
    assert new_new_size < old_size


def test_still_transactional_after_persist(client: HowsoDirectClient, tmp_path: Path):
    """Test that the file is still transactional after persisting."""
    trainee = client.create_trainee(
        features={"one": {"type": "continuous"}},
        persistence='always',
        runtime=TraineeDirectRuntimeOptions(transactional=True)
    )
    trainee_path = tmp_path / f"{trainee.id}.caml"
    size1 = trainee_path.stat().st_size
    client.train(trainee.id, [[1]], features=['one'])
    size2 = trainee_path.stat().st_size
    assert size2 > size1
    client.persist_trainee(trainee.id)
    size3 = trainee_path.stat().st_size
    assert size3 < size2
    client.train(trainee.id, [[2]], features=['one'])
    size4 = trainee_path.stat().st_size
    assert size4 > size3
