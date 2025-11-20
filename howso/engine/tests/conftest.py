import pandas as pd
from pathlib import Path
import pytest

from howso.utilities import infer_feature_attributes


@pytest.fixture(scope="module", name="data")
def load_data():
    filename = Path(Path(__file__).parent, "data/iris.csv")
    return pd.read_csv(filename)


@pytest.fixture(scope="module", name="features")
def guess_features(data):
    return infer_feature_attributes(data)
