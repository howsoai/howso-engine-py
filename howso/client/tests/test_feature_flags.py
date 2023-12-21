import pytest
import warnings

from ..feature_flags import FeatureFlags


@pytest.mark.parametrize("flags, target, expected", [
    (None, "test", False),
    ({}, "test", False),
    ({"test": False}, "test", False),
    ({"test": True}, "test", True),
    ({"TEST_ABC": True}, "test_ABc", True),  # Case should not matter
    ({"TEST-TEST": True}, "Test_Test", True),  # _ and - are interchangeable
])
def test_feature_flag_state(flags, target, expected):
    """Test flag state is expected."""
    ff = FeatureFlags(flags)
    assert ff.is_enabled(target) == expected


def test_feature_fags_warn():
    """Test obsolete feature flag warns."""

    class TestFlags(FeatureFlags):
        _obsolete_flags = {'test_bad'}

    # Should warn
    with pytest.warns(UserWarning, match="The following Howso feature flags"):
        TestFlags({"TEST_BAD": False})

    # Should not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TestFlags({"test_ok": True})


def test_feature_flags_magic():
    """Test magic methods of FeatureFlags class."""

    ff = FeatureFlags({"TEST_ABC": True})

    # Test __repr__
    assert repr(ff) == "{'test_abc': True}"

    # Test __iter__
    it = ff.__iter__()
    assert next(it) == ("test_abc", True)
    with pytest.raises(StopIteration):
        next(it)
