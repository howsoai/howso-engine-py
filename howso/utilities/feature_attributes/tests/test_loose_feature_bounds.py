import pytest

from howso.utilities.feature_attributes.base import InferFeatureAttributesBase


@pytest.mark.parametrize("observed_min, observed_max, inferred_min, inferred_max", [
    (100, 105, 96.756, 108.2436),
    (-100, 105, -232.988, 237.988),
    (-105, -100, -108.244, -96.756),
    (41, 105, 0, 146.518),
    (0, 200, 0, 329.744),
    (-200, 0, -329.744, 0),
    (100, 100, 100, 100),
    (0, 0, 0, 0),
])
def test_infer_loose_bounds(observed_min, observed_max, inferred_min, inferred_max):
    """Test that our calculations produce expected values."""
    func = InferFeatureAttributesBase.infer_loose_feature_bounds

    loose_min, loose_max = func(observed_min, observed_max)

    assert pytest.approx(loose_min, abs=1e-3) == inferred_min
    assert pytest.approx(loose_max, abs=1e-3) == inferred_max
