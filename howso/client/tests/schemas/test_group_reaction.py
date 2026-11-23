import numpy as np
import pandas as pd

from howso.client.schemas import GroupReaction
from howso.utilities import infer_feature_attributes


def test_influential_cases_have_consistent_dtypes():
    """Tests that per-group influential cases are formatted with consistent dtypes across all groups."""
    df = pd.DataFrame({
        "num": [1.5, 2.5, 3.5],
        "int": [1, 2, 3],
        "nom": ["a", "b", "c"],
        "datetime": ["2020-09-12T09:09:09", "2020-10-12T10:10:10", "2020-12-12T12:12:12"],
    })
    features = ["nom", "num", "int", "datetime", ".influence_weight"]
    # Influential cases arrive from the engine as serialized values (datetimes as strings)
    rows = df[features[:-1]].values.tolist()
    df["datetime"] = pd.to_datetime(df["datetime"])
    attributes = infer_feature_attributes(df, default_time_zone="UTC")
    data = {
        "action_features": ["num"],
        "action_values": [[2.0], [3.5]],
        "influential_cases": {
            "features": features,
            "values": [
                [[*rows[0], 0.5], [*rows[1], 0.5]],
                # Whole-number influence weight, which per-group inference would type as int64
                [[*rows[2], 1]],
            ],
        },
    }
    influential_cases = GroupReaction(data, attributes)["details"].get("influential_cases")

    assert isinstance(influential_cases, list)
    assert len(influential_cases) == 2
    for group in influential_cases:
        assert isinstance(group, pd.DataFrame)
        assert group.columns.tolist() == features
        assert group.index.tolist() == list(range(len(group)))
        assert group["num"].dtype == np.float64
        assert group["int"].dtype == np.int64
        assert pd.api.types.is_datetime64_any_dtype(group["datetime"])
        assert group[".influence_weight"].dtype == np.float64
    assert influential_cases[0]["nom"].tolist() == ["a", "b"]
    assert influential_cases[1]["int"].tolist() == [3]


def test_influential_cases_with_empty_and_null_groups():
    """Tests that groups without influential cases are formatted as empty DataFrames, never None."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    data = {
        "influential_cases": {"features": ["num", "int"], "values": [None, [[2.5, 2]], []]},
    }
    influential_cases = GroupReaction(data, attributes)["details"].get("influential_cases")

    assert isinstance(influential_cases, list)
    assert len(influential_cases) == 3
    assert all(isinstance(group, pd.DataFrame) for group in influential_cases)
    assert all(group.columns.tolist() == ["num", "int"] for group in influential_cases)
    assert influential_cases[0].empty
    assert influential_cases[2].empty
    assert influential_cases[1]["int"].tolist() == [2]


def test_influential_cases_with_null_features():
    """Tests that influential cases without features are formatted as one empty DataFrame per group."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    data = {"influential_cases": {"features": None, "values": [None, None]}}
    influential_cases = GroupReaction(data, attributes)["details"].get("influential_cases")

    assert isinstance(influential_cases, list)
    assert len(influential_cases) == 2
    for group in influential_cases:
        assert isinstance(group, pd.DataFrame)
        assert group.empty


def test_influential_cases_to_dict():
    """Tests that `to_dict` converts per-group influential cases back into lists of records."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    data = {
        "influential_cases": {"features": ["num", "int"], "values": [[[1.5, 1], [2.5, 2]], []]},
    }
    as_dict = GroupReaction(data, attributes).to_dict()

    assert as_dict["details"]["influential_cases"] == [
        [{"num": 1.5, "int": 1}, {"num": 2.5, "int": 2}],
        [],
    ]
