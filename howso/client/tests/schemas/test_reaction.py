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
