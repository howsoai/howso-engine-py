import pmlb
import pytest

from howso.utilities import infer_feature_attributes


@pytest.fixture(scope="module", name="data")
def load_data():
    return pmlb.fetch_data("iris")


@pytest.fixture(scope="module", name="features")
def guess_features(data):
    return infer_feature_attributes(data)
