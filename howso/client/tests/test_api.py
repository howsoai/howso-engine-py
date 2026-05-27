import pytest

from howso.client.api import get_api
from howso.client.exceptions import HowsoError
from howso.engine import Trainee


def test_get_api():
    """Test get_api response."""
    # New Trainees will default to using the Amalgam type specified by the user
    amalgam_postfix = "-" + Trainee().get_runtime()["library_type"]
    doc = get_api(amalgam_postfix=amalgam_postfix)
    assert isinstance(doc, dict)
    assert isinstance(doc["description"], str)
    assert isinstance(doc["schemas"], dict)
    assert len(doc["schemas"]) > 0
    assert isinstance(doc["labels"], dict)
    assert "get_api" in doc["labels"]


def test_get_api_raises():
    """Test get_api raises when Engine files fail to load."""
    # New Trainees will default to using the Amalgam type specified by the user
    amalgam_postfix = "-" + Trainee().get_runtime()["library_type"]
    with pytest.raises(HowsoError, match="The Howso Engine file path does not exist"):
        # Should raise if passed a non-existent file
        get_api("dne", amalgam_postfix=amalgam_postfix)
