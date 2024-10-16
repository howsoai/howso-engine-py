import pytest

from howso.client.api import DEFAULT_ENGINE_PATH, get_api
from howso.client.exceptions import HowsoError


def test_get_api():
    """Test get_api response."""
    doc = get_api()
    assert isinstance(doc, dict)
    assert isinstance(doc["description"], str)
    assert isinstance(doc["schemas"], dict)
    assert len(doc["schemas"]) > 0
    assert isinstance(doc["labels"], dict)
    assert "get_api" in doc["labels"]


def test_get_api_raises():
    """Test get_api raises when Engine files fail to load."""
    with pytest.raises(HowsoError, match="The Howso Engine file path does not exist"):
        # Should raise if passed a non-existent file
        get_api("dne")
    with pytest.raises(HowsoError, match="Failed to retrieve the Howso Engine API"):
        # Should raise if file is invalid to be loaded in Amalgam
        get_api(DEFAULT_ENGINE_PATH.joinpath("version.json"))
