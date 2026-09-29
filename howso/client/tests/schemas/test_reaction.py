import numpy as np
import pandas as pd

from howso.client.schemas.reaction import Reaction
from howso.utilities import infer_feature_attributes


def test_cases_with_details_add_reaction():
    """Tests that `Reaction` `add_reaction` works with different data types."""
    df = pd.DataFrame({
        "nom": ["a", "b", "c", "d"],
        "datetime": ["2020-09-12T09:09:09", "2020-10-12T10:10:10", "2020-12-12T12:12:12", "2020-10-11T11:11:11"],
        "num": [1, 2, 3, 4]
    })
    df["datetime"] = pd.to_datetime(df["datetime"])

    react_response = {
        "details": {"action_features": df.columns.tolist()},
        "action": df,
    }
    attributes = infer_feature_attributes(df, default_time_zone="UTC")

    cwd = Reaction(react_response['action'], react_response['details'], attributes)
    cwd.accumulate(Reaction(react_response['action'].to_dict(), react_response['details'], attributes))
    # List of dicts
    cwd.accumulate(Reaction(react_response['action'].to_dict(orient='records'), react_response['details'], attributes))
    cwd.accumulate(Reaction(react_response['action'], react_response['details'], attributes))

    assert cwd["details"].get("action_features") == df.columns.tolist()
    assert cwd['action'].shape[0] == 16


def test_action_and_context_features():
    """Tests that `Reaction` maintains correct data type and order for action/context features."""
    df = pd.DataFrame({
        "a": ["a", "b", "c"],
        "b": ["x", "y", "z"],
        "c": [1, 2, 3],
        "d": [9, 8, 7],
    })
    attributes = infer_feature_attributes(df)

    # Test empty action
    react_response = {
        "details": {"action_features": [], "context_features": []},
        "action": pd.DataFrame(),
    }
    cwd = Reaction(react_response['action'], react_response['details'], attributes)
    assert cwd["action"].columns.tolist() == []
    assert cwd["details"].get("action_features") == []
    assert cwd["details"].get("context_features") == []
    cwd.accumulate(Reaction(react_response['action'], react_response['details'], attributes))
    assert cwd["action"].columns.tolist() == []
    assert cwd["details"].get("action_features") == []
    assert cwd["details"].get("context_features") == []

    # Test populated features list
    react_response = {
        "details": {"action_features": ["b", "c"], "context_features": ["c", "a"]},
        "action": df.loc[:, ["b", "c"]],
    }
    cwd = Reaction(react_response["action"], react_response["details"], attributes)
    assert cwd["action"].columns.tolist() == ["b", "c"]
    assert cwd["details"].get("action_features") == ["b", "c"]
    assert cwd["details"].get("context_features") == ["c", "a"]
    cwd.accumulate(Reaction(react_response["action"], react_response["details"], attributes))
    assert cwd["action"].columns.tolist() == ["b", "c"]
    assert cwd["details"].get("action_features") == ["b", "c"]
    assert cwd["details"].get("context_features") == ["c", "a"]


def test_cases_with_details_instantiate():
    """Tests that `Reaction` can be instantiated with different data types."""
    df = pd.DataFrame({
        "nom": ["a", "b", "c", "d"],
        "datetime": ["2020-09-12T09:09:09", "2020-10-12T10:10:10", "2020-12-12T12:12:12", "2020-10-11T11:11:11"]
    })
    react_response = {
        'details': {'action_features': ['datetime']},
        'action': df
    }
    attributes = infer_feature_attributes(df, default_time_zone="UTC")

    cwd = Reaction(react_response['action'], react_response['details'], attributes)
    assert cwd['action'].shape[0] == 4

    cwd = Reaction(react_response['action'].to_dict(), react_response['details'], attributes)
    assert cwd['action'].shape[0] == 4

    cwd = Reaction(react_response['action'].to_dict(orient='records'), react_response['details'], attributes)
    assert cwd['action'].shape[0] == 4


def test_case_list_details_have_consistent_dtypes():
    """Tests that per-case DataFrame details are formatted with consistent dtypes across all cases."""
    df = pd.DataFrame({
        "num": [1.5, 2.5, 3.5, 4.5],
        "int": [1, 2, 3, 4],
        "nom": ["a", "b", "c", "d"],
        "datetime": ["2020-09-12T09:09:09", "2020-10-12T10:10:10", "2020-12-12T12:12:12", "2020-10-11T11:11:11"],
    })
    # Feature attributes are inferred from typed data, but case details arrive from the engine as serialized
    # values (datetimes as strings), so keep `rows` as the raw string form.
    rows = df.to_dict(orient="records")
    df["datetime"] = pd.to_datetime(df["datetime"])
    attributes = infer_feature_attributes(df, default_time_zone="UTC")
    # Second case has whole-number influence weights, which per-case inference would type as int64.
    details = {
        "action_features": ["num"],
        "influential_cases": [
            [{**rows[0], ".influence_weight": 0.5}, {**rows[1], ".influence_weight": 0.5}],
            [{**rows[2], ".influence_weight": 1}],
            [],
            [{**rows[3], ".influence_weight": 0.25}, {**rows[0], ".influence_weight": 0.75}],
        ],
    }
    reaction = Reaction(df[["num"]], details, attributes)
    cases = reaction["details"]["influential_cases"]

    assert len(cases) == 4
    assert cases[2].empty
    assert all(isinstance(case, pd.DataFrame) for case in cases)
    for case in (cases[0], cases[1], cases[3]):
        assert case.index.tolist() == list(range(len(case)))
        assert case["num"].dtype == np.float64
        assert case["int"].dtype == np.int64
        assert pd.api.types.is_datetime64_any_dtype(case["datetime"])
        assert case[".influence_weight"].dtype == np.float64
    assert cases[1][".influence_weight"].tolist() == [1.0]
    assert cases[3]["nom"].tolist() == ["d", "a"]


def test_case_list_details_with_non_uniform_keys():
    """Tests that per-case details whose rows have differing keys are still formatted per case."""
    df = pd.DataFrame({"num": [1.5, 2.5], "nom": ["a", "b"]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "most_similar_cases": [
            [{"num": 1.5, "nom": "a"}],
            [{"num": 2.5}],
        ],
    }
    cases = Reaction(df[["num"]], details, attributes)["details"]["most_similar_cases"]
    assert cases[0].columns.tolist() == ["num", "nom"]
    assert cases[1].columns.tolist() == ["num"]


def test_case_list_details_nested_per_series():
    """Tests that per-case details nested one level deeper (as in `react_series`) are still formatted per case."""
    df = pd.DataFrame({"num": [1.5, 2.5, 3.5], "int": [1, 2, 3]})
    attributes = infer_feature_attributes(df)
    rows = df.to_dict(orient="records")
    # Two series, each with two time steps, each step holding a list of influential cases.
    details = {
        "action_features": ["num"],
        "influential_cases": [
            [
                [{**rows[0], ".influence_weight": 1}],
                [{**rows[1], ".influence_weight": 0.5}, {**rows[2], ".influence_weight": 0.5}],
            ],
            [[{**rows[2], ".influence_weight": 0.25}], []],
        ],
    }
    series = Reaction(df[["num"]], details, attributes)["details"]["influential_cases"]
    assert len(series) == 2
    assert all(len(steps) == 2 for steps in series)
    assert series[1][1].empty
    for steps in series:
        for step in steps:
            assert isinstance(step, pd.DataFrame)
            if not step.empty:
                assert step["int"].dtype == np.int64
                assert step[".influence_weight"].dtype == np.float64
    assert series[0][1]["num"].tolist() == [2.5, 3.5]


def test_case_list_details_with_ragged_rows_within_a_case():
    """Tests that a case whose rows have differing keys falls back to per-case formatting with a column union."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2], "nom": ["a", "b"]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "influential_cases": [
            [{"num": 1.5, "int": 1, "nom": "a"}, {"num": 2.5, "nom": "b"}],
            [{"num": 1.5, "int": 1, "nom": "a"}],
        ],
    }
    cases = Reaction(df[["num"]], details, attributes)["details"]["influential_cases"]
    assert [list(case.columns) for case in cases] == [["num", "int", "nom"], ["num", "int", "nom"]]
    # The missing value is filled with a null, so the integer column becomes nullable for that case only.
    assert str(cases[0]["int"].dtype) == "Int64"
    assert cases[0]["int"].isna().tolist() == [False, True]
    assert cases[0]["int"].iloc[0] == 1
    assert str(cases[1]["int"].dtype) == "int64"
    assert cases[0]["nom"].tolist() == ["a", "b"]


def test_case_list_details_with_non_dict_row():
    """Tests that a case containing a non-dict row is left untouched rather than forced into a DataFrame."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "influential_cases": [
            [{"num": 1.5, "int": 1}, None],
            [{"num": 2.5, "int": 2}],
        ],
    }
    cases = Reaction(df[["num"]], details, attributes)["details"]["influential_cases"]
    assert cases[0] == [{"num": 1.5, "int": 1}, None]
    assert isinstance(cases[1], pd.DataFrame)
    assert cases[1]["int"].dtype == np.int64
