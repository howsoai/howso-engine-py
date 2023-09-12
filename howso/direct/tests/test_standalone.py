import pytest

from amalgam.api import Amalgam
from howso.direct import HowsoDirectClient, HowsoCore
from howso.utilities.testing import get_configurationless_test_client


@pytest.fixture
def client():
    """Direct client instance using latest binaries."""
    return get_configurationless_test_client(client_class=HowsoDirectClient,
                                             verbose=True, trace=True)


def test_direct_client(client):
    """Sanity check client instantiation."""
    assert isinstance(client.howso, HowsoCore)
    assert isinstance(client.howso.amlg, Amalgam)
    version = client.get_version()
    assert version.api is not None
    assert version.client is not None


@pytest.mark.parametrize(('filename', 'truthiness'), (
    ('./banana.txt', True),
    ('./ba\nana.txt', True),
    ('./bañän🤣a.txt', True),
))
def test_check_name_valid_for_save(client, filename, truthiness):
    """Ensure that the internal function `check_name_valid_for_save` works."""
    assert client.check_name_valid_for_save(filename, clobber=True)[0] == truthiness
