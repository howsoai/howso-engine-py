from typing import Any

import numpy as np
import pandas as pd
import pytest

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


def test_accumulate_does_not_mutate_source_reactions():
    """Tests that `accumulate` does not extend list details owned by the accumulated reactions."""
    df = pd.DataFrame({"num": [1.5, 2.5]})
    attributes = infer_feature_attributes(df)
    target = Reaction(df, {"action_features": ["num"]}, attributes)
    first = Reaction(df, {"action_features": ["num"], "distance_contribution": [0.1, 0.2]}, attributes)
    second = Reaction(df, {"action_features": ["num"], "distance_contribution": [0.3, 0.4]}, attributes)

    target.accumulate([first, second])

    assert target["details"]["distance_contribution"] == [0.1, 0.2, 0.3, 0.4]
    assert first["details"]["distance_contribution"] == [0.1, 0.2]
    assert second["details"]["distance_contribution"] == [0.3, 0.4]


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


def test_multi_table_details_have_consistent_dtypes():
    """Tests that per-case DataFrame details are formatted with consistent dtypes across all cases."""
    df = pd.DataFrame({
        "num": [1.5, 2.5, 3.5, 4.5],
        "int": [1, 2, 3, 4],
        "nom": ["a", "b", "c", "d"],
        "datetime": ["2020-09-12T09:09:09", "2020-10-12T10:10:10", "2020-12-12T12:12:12", "2020-10-11T11:11:11"],
    })
    # Feature attributes are inferred from typed data, but case details arrive from the engine as serialized
    # values (datetimes as strings), so keep `rows` as the raw string form.
    rows = df.values.tolist()
    df["datetime"] = pd.to_datetime(df["datetime"])
    attributes = infer_feature_attributes(df, default_time_zone="UTC")
    features = ["num", "int", "nom", "datetime", ".influence_weight"]
    # Second case has whole-number influence weights, which per-case inference would type as int64.
    details = {
        "action_features": ["num"],
        "influential_cases": {
            "features": features,
            "values": [
                [[*rows[0], 0.5], [*rows[1], 0.5]],
                [[*rows[2], 1]],
                [],
                [[*rows[3], 0.25], [*rows[0], 0.75]],
            ],
        },
    }
    reaction = Reaction(df[["num"]], details, attributes)
    cases = reaction["details"].get("influential_cases")

    assert isinstance(cases, list)
    assert len(cases) == 4
    assert all(isinstance(case, pd.DataFrame) for case in cases)
    assert all(case.columns.tolist() == features for case in cases)
    assert cases[2].empty
    for case in (cases[0], cases[1], cases[3]):
        assert case.index.tolist() == list(range(len(case)))
        assert case["num"].dtype == np.float64
        assert case["int"].dtype == np.int64
        assert pd.api.types.is_datetime64_any_dtype(case["datetime"])
        assert case[".influence_weight"].dtype == np.float64
    assert cases[1][".influence_weight"].tolist() == [1.0]
    assert cases[3]["nom"].tolist() == ["d", "a"]


def test_multi_table_details_column_order_follows_features():
    """Tests that per-case DataFrame columns follow the order of the detail's `features` list."""
    df = pd.DataFrame({"num": [1.5, 2.5], "nom": ["a", "b"]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "most_similar_cases": {
            "features": ["nom", "num"],
            "values": [[["a", 0.5]], [["b", 2.5], ["a", 1.5]]],
        },
    }
    cases = Reaction(df[["num"]], details, attributes)["details"].get("most_similar_cases")
    assert isinstance(cases, list)
    assert [case.columns.tolist() for case in cases] == [["nom", "num"], ["nom", "num"]]
    assert cases[1]["nom"].tolist() == ["b", "a"]
    assert cases[1]["num"].tolist() == [2.5, 1.5]


def test_multi_table_details_without_any_rows():
    """Tests that per-case details with no rows produce one empty DataFrame per case, keeping the columns."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "boundary_cases": {"features": ["num", "int"], "values": [[], []]},
        "influential_cases": {"features": ["num", "int"], "values": []},
    }
    formatted = Reaction(df[["num"]], details, attributes)["details"]

    boundary_cases = formatted.get("boundary_cases")
    assert isinstance(boundary_cases, list)
    assert len(boundary_cases) == 2
    for case in boundary_cases:
        assert isinstance(case, pd.DataFrame)
        assert case.empty
        assert case.columns.tolist() == ["num", "int"]
    assert formatted.get("influential_cases") == []


def test_multi_table_details_with_null_tables():
    """Tests that null per-case tables (details not computed for a case) are formatted as empty DataFrames."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        # As sent by the engine when the detail was not computed for any case
        "boundary_cases": {"features": None, "values": [None, None]},
        "influential_cases": {"features": ["num", "int"], "values": [None, [[1.5, 1]], []]},
    }
    formatted = Reaction(df[["num"]], details, attributes)["details"]

    boundary_cases = formatted.get("boundary_cases")
    assert isinstance(boundary_cases, list)
    assert len(boundary_cases) == 2
    for case in boundary_cases:
        assert isinstance(case, pd.DataFrame)
        assert case.empty
        assert case.columns.tolist() == []

    influential_cases = formatted.get("influential_cases")
    assert isinstance(influential_cases, list)
    assert len(influential_cases) == 3
    assert all(isinstance(case, pd.DataFrame) for case in influential_cases)
    assert all(case.columns.tolist() == ["num", "int"] for case in influential_cases)
    assert influential_cases[0].empty
    assert influential_cases[2].empty
    assert influential_cases[1]["num"].tolist() == [1.5]
    assert influential_cases[1]["int"].dtype == np.int64


def test_multi_table_details_nested_per_series_with_null_tables():
    """Tests that null per-step tables within a `react_series` detail are formatted as empty DataFrames."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "influential_cases": [
            {"features": ["num", "int"], "values": [None, [[2.5, 2]]]},
            {"features": None, "values": [None]},
        ],
    }
    series = Reaction(df[["num"]], details, attributes)["details"].get("influential_cases")
    assert isinstance(series, list)
    assert [len(steps) for steps in series] == [2, 1]
    assert all(isinstance(step, pd.DataFrame) for steps in series for step in steps)
    assert series[0][0].empty
    assert series[0][0].columns.tolist() == ["num", "int"]
    assert series[0][1]["int"].tolist() == [2]
    assert series[1][0].empty


def test_multi_table_details_nested_per_series():
    """Tests that per-case details nested one level deeper (as in `react_series`) are still formatted per case."""
    df = pd.DataFrame({"num": [1.5, 2.5, 3.5], "int": [1, 2, 3]})
    attributes = infer_feature_attributes(df)
    rows = df.values.tolist()
    features = ["num", "int", ".influence_weight"]
    # Two series, each with two time steps, each step holding a list of influential cases.
    details = {
        "action_features": ["num"],
        "influential_cases": [
            {
                "features": features,
                "values": [
                    [[*rows[0], 1]],
                    [[*rows[1], 0.5], [*rows[2], 0.5]],
                ],
            },
            {"features": features, "values": [[[*rows[2], 0.25]], []]},
        ],
    }
    series = Reaction(df[["num"]], details, attributes)["details"].get("influential_cases")
    assert isinstance(series, list)
    assert len(series) == 2
    assert all(isinstance(steps, list) and len(steps) == 2 for steps in series)
    assert series[1][1].empty
    for steps in series:
        for step in steps:
            assert isinstance(step, pd.DataFrame)
            assert step.columns.tolist() == features
            if not step.empty:
                assert step["int"].dtype == np.int64
                assert step[".influence_weight"].dtype == np.float64
    # The whole-number weight in the first step is batched with the rest of its series, so it is a float too.
    assert series[0][0][".influence_weight"].tolist() == [1.0]
    assert series[0][1]["num"].tolist() == [2.5, 3.5]


def test_single_table_case_details():
    """Tests that one-to-one case details are formatted into a single DataFrame with one row per reaction case."""
    df = pd.DataFrame({
        "num": [1.5, 2.5],
        "int": [1, 2],
        "nom": ["a", "b"],
        "datetime": ["2020-09-12T09:09:09", "2020-10-12T10:10:10"],
    })
    rows = df.values.tolist()
    df["datetime"] = pd.to_datetime(df["datetime"])
    attributes = infer_feature_attributes(df, default_time_zone="UTC")
    details = {
        "action_features": ["num"],
        "predicted_values_for_case": {"features": ["num", "int", "nom", "datetime"], "values": rows},
        "hypothetical_values": {"features": ["int", "nom"], "values": [[2, "b"], [1, "a"]]},
    }
    formatted = Reaction(df[["num"]], details, attributes)["details"]

    predicted = formatted.get("predicted_values_for_case")
    assert isinstance(predicted, pd.DataFrame)
    assert predicted.columns.tolist() == ["num", "int", "nom", "datetime"]
    assert predicted.index.tolist() == [0, 1]
    assert predicted["num"].dtype == np.float64
    assert predicted["int"].dtype == np.int64
    assert pd.api.types.is_datetime64_any_dtype(predicted["datetime"])
    assert predicted["nom"].tolist() == ["a", "b"]

    hypothetical = formatted.get("hypothetical_values")
    assert isinstance(hypothetical, pd.DataFrame)
    assert hypothetical.columns.tolist() == ["int", "nom"]
    assert hypothetical["int"].tolist() == [2, 1]
    assert hypothetical["nom"].tolist() == ["b", "a"]


def test_single_table_case_details_without_values():
    """Tests that one-to-one case details with no values produce an empty DataFrame that keeps its columns."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "predicted_values_for_case": {"features": ["num", "int"], "values": []},
    }
    predicted = Reaction(df[["num"]], details, attributes)["details"].get("predicted_values_for_case")
    assert isinstance(predicted, pd.DataFrame)
    assert predicted.empty
    assert predicted.columns.tolist() == ["num", "int"]


def test_single_table_case_details_with_null_rows():
    """Tests that null rows (details not computed for a case) are kept as null rows aligned to their case."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2], "nom": ["a", "b"]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "predicted_values_for_case": {"features": ["num", "int", "nom"], "values": [None, [2.5, 2, "b"]]},
        "hypothetical_values": {"features": ["num", "int"], "values": [None, None]},
    }
    formatted = Reaction(df[["num"]], details, attributes)["details"]

    predicted = formatted.get("predicted_values_for_case")
    assert isinstance(predicted, pd.DataFrame)
    assert predicted.columns.tolist() == ["num", "int", "nom"]
    assert predicted.index.tolist() == [0, 1]
    assert predicted.iloc[0].isna().all()
    assert predicted.iloc[1].tolist() == [2.5, 2, "b"]
    assert predicted["num"].dtype == np.float64
    # The null makes the integer column nullable
    assert str(predicted["int"].dtype) == "Int64"

    hypothetical = formatted.get("hypothetical_values")
    assert isinstance(hypothetical, pd.DataFrame)
    assert hypothetical.columns.tolist() == ["num", "int"]
    assert hypothetical.index.tolist() == [0, 1]
    assert hypothetical.isna().all().all()


@pytest.mark.parametrize(("detail", "expected_index"), [
    pytest.param({"features": None, "values": None}, [], id="null-features-and-values"),
    pytest.param({"features": None, "values": [None, None]}, [0, 1], id="null-features-null-rows"),
])
def test_single_table_case_details_with_null_features(detail: dict[str, Any], expected_index: list[int]):
    """Tests that a one-to-one case detail without features keeps one columnless row per case."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    details = {"action_features": ["num"], "predicted_values_for_case": detail}
    predicted = Reaction(df[["num"]], details, attributes)["details"].get("predicted_values_for_case")
    assert isinstance(predicted, pd.DataFrame)
    assert predicted.empty
    assert predicted.columns.tolist() == []
    assert predicted.index.tolist() == expected_index


def test_accumulate_single_table_case_details_with_null_features():
    """Tests that per-case placeholder rows without features stay aligned to action rows after accumulation."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2], "nom": ["a", "b"]})
    attributes = infer_feature_attributes(df)
    reaction = Reaction(
        df[["num"]],
        {"action_features": ["num"], "predicted_values_for_case": {"features": None, "values": [None, None]}},
        attributes,
    )
    reaction.accumulate(Reaction(
        df[["num"]].iloc[1:],
        {
            "action_features": ["num"],
            "predicted_values_for_case": {"features": ["num", "int", "nom"], "values": [[2.5, 2, "b"]]},
        },
        attributes,
    ))

    predicted = reaction["details"].get("predicted_values_for_case")
    assert isinstance(predicted, pd.DataFrame)
    assert len(reaction["action"]) == 3
    assert len(predicted) == len(reaction["action"])
    assert predicted.columns.tolist() == ["num", "int", "nom"]
    # The prediction must land on the third action row, after the two placeholder rows
    assert predicted.iloc[:2].isna().all().all()
    assert predicted.iloc[2].tolist() == [2.5, 2, "b"]


def test_accumulate_case_details():
    """Tests that accumulating reactions extends per-case DataFrame lists and concatenates single-table details."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    rows = df.values.tolist()

    def _make_reaction(case_rows: list[list[Any]]) -> Reaction:
        return Reaction(
            df[["num"]],
            {
                "action_features": ["num"],
                "influential_cases": {"features": ["num", "int"], "values": [[row] for row in case_rows]},
                "predicted_values_for_case": {"features": ["num", "int"], "values": case_rows},
            },
            attributes,
        )

    reaction = _make_reaction(rows)
    reaction.accumulate(_make_reaction(rows[::-1]))

    influential_cases = reaction["details"].get("influential_cases")
    assert isinstance(influential_cases, list)
    assert len(influential_cases) == 4
    assert [case["int"].tolist() for case in influential_cases] == [[1], [2], [2], [1]]

    predicted = reaction["details"].get("predicted_values_for_case")
    assert isinstance(predicted, pd.DataFrame)
    assert predicted["int"].tolist() == [1, 2, 2, 1]
    assert predicted["int"].dtype == np.int64


def test_relevant_values():
    """Tests that `relevant_values` is formatted into a dict of feature name to Series for each reaction case."""
    df = pd.DataFrame({"num": [1.5, 2.5, 3.5], "int": [1, 2, 3], "nom": ["a", "b", "c"]})
    attributes = infer_feature_attributes(df)
    details = {
        "action_features": ["num"],
        "relevant_values": {
            "features": ["int", "nom"],
            "values": [
                [[1, 2], ["a", "b"]],
                [[3], ["c", "a", "b"]],
            ],
        },
    }
    relevant_values = Reaction(df[["num"]], details, attributes)["details"].get("relevant_values")
    assert isinstance(relevant_values, list)
    assert len(relevant_values) == 2
    for case in relevant_values:
        assert list(case.keys()) == ["int", "nom"]
        assert all(isinstance(series, pd.Series) for series in case.values())
        assert case["int"].name == "int"
        assert case["int"].dtype == np.int64
    assert relevant_values[0]["int"].tolist() == [1, 2]
    assert relevant_values[1]["nom"].tolist() == ["c", "a", "b"]


def test_relevant_values_non_mapping_is_unchanged():
    """Tests that a non-mapping `relevant_values` (as in `react_series`) is returned as-is."""
    df = pd.DataFrame({"num": [1.5, 2.5], "int": [1, 2]})
    attributes = infer_feature_attributes(df)
    relevant_values = [{"features": ["int"], "values": [[[1, 2]]]}]
    details = {"action_features": ["num"], "relevant_values": relevant_values}
    formatted = Reaction(df[["num"]], details, attributes)["details"].get("relevant_values")
    assert formatted == relevant_values
