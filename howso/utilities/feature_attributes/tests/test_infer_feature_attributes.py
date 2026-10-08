"""Tests the `infer_feature_attributes` package."""
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import copy
import datetime
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from howso.client.typing import ProtectedValueMultiplier
from howso.utilities.feature_attributes import infer_feature_attributes
from howso.utilities.feature_attributes.base import FeatureAttributesBase, FLOAT_MAX, FLOAT_MIN, INTEGER_MAX
from howso.utilities.feature_attributes.pandas import InferFeatureAttributesDataFrame
from howso.utilities.feature_attributes.suggestions import IFASuggestionCollector
from howso.utilities.features import FeatureType
from howso.utilities.utilities import get_optimized_partition_size

if platform.system().lower() == "windows":
    DT_MAX = "6053-01-24"
    ALMOST_DT_MAX = "6053-01-23"
else:
    DT_MAX = "2262-04-11"
    ALMOST_DT_MAX = "2262-04-10"

cwd = Path(__file__).parent.parent.parent.parent
iris_path = Path(cwd, "utilities", "tests", "data", "iris.csv")
int_path = Path(cwd, "utilities", "tests", "data", "integers.csv")
joined_olist_df = pd.read_parquet(Path(cwd, "utilities", "tests", "data", "joined_olist.parquet"))[:10000]
try:
    nypd_arrest_df = pd.read_parquet(Path(cwd, "utilities", "tests", "data", "NYPD_arrest_data_25K.parquet"))
except ImportError:
    nypd_arrest_df = None
stock_path = Path(cwd, "utilities", "tests", "data", "mini_stock_data.csv")
ts_path = Path(cwd, "utilities", "tests", "data", "example_timeseries.csv")

# Frame with two independent nominal keys, a continuous column, and a
# globally-constant column. A constant column is functionally determined by
# every key, so before the fan-out fix it attached to every key's fan-out set,
# producing sibling levels that tripped the "strict-tree" warning and wrongly
# listed the constant as a fan-out feature. Used by the constant-column
# regression tests in both the Pandas and ADC suites.
_FANOUT_CONSTANT_ROWS = 240
fanout_constant_df = pd.DataFrame({
    "key_a": [f"a{i % 5}" for i in range(_FANOUT_CONSTANT_ROWS)],
    "key_b": [f"b{i % 8}" for i in range(_FANOUT_CONSTANT_ROWS)],
    "measure": [i * 0.1 for i in range(_FANOUT_CONSTANT_ROWS)],
    "const_col": ["CONSTANT"] * _FANOUT_CONSTANT_ROWS,
})

# Partially defined dictionary-1
features_1 = {
    "sepal_length": {
        "type": "continuous",
        "bounds": {
            "min": 2.72,
            "max": 3,
            "allow_null": True
        },
    },
    "sepal_width": {
        "type": "continuous"
    }
}

# Partially defined dictionary-2
features_2 = {
    "sepal_length": {
        "type": "continuous"
    },
    "sepal_width": {
        "type": "continuous"
    }
}

# Partially defined dictionary-3
features_3 = {
    "sepal_length": {
        "type": "nominal"
    },
    "sepal_width": {
        "type": "continuous"
    }
}

# Partially defined "ordered" dict
features_4 = OrderedDict(
    (f_name, features_3[f_name]) for f_name in features_3
)


def test_infer_features_attributes():
    """Litmus test for infer feature types for iris dataset."""
    df = pd.read_csv(iris_path)

    expected_types = {
        "sepal_length": "continuous",
        "sepal_width": "continuous",
        "petal_length": "continuous",
        "petal_width": "continuous",
        "class": "nominal"
    }

    features = infer_feature_attributes(df)

    for feature, attributes in features.items():
        assert expected_types[feature] == attributes["type"]


@pytest.mark.parametrize(
    "feature, nominality", [
        # "id_no" _would_ be inferred "continuous", but we specifically tell
        # `_process()` that it is indeed an ID feature, so it
        # will be set to "nominal".
        ("id_no", "nominal"),

        # The "badge_no" feature contains ALL unique values so it exceeds the
        # sqrt(total num. rows) test, but they are all the same length
        # integers, so it passes "all the same length" check.
        ("badge_no", "nominal"),

        # The "salary" feature has mostly uniques (some duplicates) but too
        # many that it readily exceeds the threshold of sqrt(total num. rows)
        # and they are not all the same length, so, "continuous".
        ("salary", "continuous"),

        # The "dept_no" feature has a number of uniques that exceed the
        # sqrt(total num. rows) but all the integers are the same length,
        # so "nominal".
        ("dept_no", "nominal"),

        # This column is all None, will be returned as "continuous".
        ("hat_size", "continuous"),
    ]
)
def test_integer_nominality(feature, nominality):
    """Exercise infer_feature_attributes for integers and their nominality."""
    df = pd.read_csv(int_path)
    inferred_features = infer_feature_attributes(df, id_feature_name=["id_no"])
    assert inferred_features[feature]["type"] == nominality


@pytest.mark.parametrize("data, expected_type", [
    # Integer
    (pd.DataFrame([[1], [None]], dtype="Int8", columns=["a"]),
     {"data_type": str(FeatureType.INTEGER), "size": 1}),
    (pd.DataFrame([[1], [16]], dtype="int", columns=["a"]),
     {"data_type": str(FeatureType.INTEGER), "size": 8}),
    # Float
    (pd.DataFrame([[1.0], [4.4]], dtype="float", columns=["a"]),
     {"data_type": str(FeatureType.NUMERIC), "size": 8}),
    (pd.DataFrame([[None], [4.4]], dtype="float32", columns=["a"]),
     {"data_type": str(FeatureType.NUMERIC), "size": 4}),
    # Boolean
    (pd.DataFrame([[True], [False], [None]], dtype="bool", columns=["a"]),
     {"data_type": str(FeatureType.BOOLEAN)}),
    # String
    (pd.DataFrame([["test"], [None]], columns=["a"]),
     {"data_type": str(FeatureType.STRING)}),
    (pd.DataFrame([["test"], [None]], dtype="string", columns=["a"]),
     {"data_type": str(FeatureType.STRING)}),
    (pd.DataFrame([["test"], [None]], dtype=np.bytes_, columns=["a"]),
     {"data_type": str(FeatureType.STRING)}),
    (pd.DataFrame([["test"]], dtype="S", columns=["a"]),
     {"data_type": str(FeatureType.STRING)}),
    (pd.DataFrame([["test"]], dtype="U", columns=["a"]),
     {"data_type": str(FeatureType.STRING)}),
    # Datetime
    (pd.DataFrame([["2020-01-01T10:00:00"]], dtype="datetime64[ns]", columns=["a"]),
     {"data_type": str(FeatureType.DATETIME)}),
    (pd.DataFrame([[datetime.datetime.now()]], columns=["a"]),
     {"data_type": str(FeatureType.DATETIME)}),
    (pd.DataFrame([[datetime.datetime.now(ZoneInfo("US/Eastern"))]], columns=["a"]),
     {"data_type": str(FeatureType.DATETIME), "timezone": "US/Eastern"}),
    (pd.DataFrame([[datetime.datetime.now(datetime.timezone(datetime.timedelta(minutes=300)))]], columns=["a"]),
     {"data_type": str(FeatureType.DATETIME)}),
    # Date
    (pd.DataFrame([[datetime.date(2020, 1, 1)]], columns=["a"]),
     {"data_type": str(FeatureType.DATE)}),
    (pd.DataFrame([[pd.Timestamp(datetime.date(2020, 1, 1))]], columns=["a"]),
     {"data_type": str(FeatureType.DATE)}),
    (pd.DataFrame([["2020-01-01"]], dtype="datetime64[ns]", columns=["a"]),
     {"data_type": str(FeatureType.DATE)}),
    # Timedelta
    (pd.DataFrame([[datetime.timedelta(days=1)]], columns=["a"]),
     {"data_type": str(FeatureType.TIMEDELTA), "unit": "seconds"}),
    (pd.DataFrame([[np.timedelta64(5, "D")]], columns=["a"]),
     {"data_type": str(FeatureType.TIMEDELTA), "unit": "seconds"}),
    (pd.DataFrame([[np.timedelta64(5, "Y")]], columns=["a"]),
     {"data_type": str(FeatureType.TIMEDELTA), "unit": "seconds"}),
    (pd.DataFrame([[np.timedelta64(5000, "ns")]], columns=["a"]),
     {"data_type": str(FeatureType.TIMEDELTA), "unit": "seconds"}),
    (pd.DataFrame([[np.timedelta64(5000, "s")]], columns=["a"]),
     {"data_type": str(FeatureType.TIMEDELTA), "unit": "seconds"}),
    (pd.DataFrame([[np.timedelta64(60, "m")]], columns=["a"]),
     {"data_type": str(FeatureType.TIMEDELTA), "unit": "seconds"}),
])
def test_get_feature_type(data, expected_type):
    """Test get_feature_type returns expected data types."""
    infer = InferFeatureAttributesDataFrame(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_type, original_type = infer._get_feature_type("a")
    expected_feature_type = expected_type.pop("data_type")
    assert str(feature_type) == expected_feature_type
    assert original_type == expected_type


@pytest.mark.parametrize("data, is_time, expected_format, provided_format", [
    (pd.DataFrame(["08:08:08"], columns=["a"]), True, "%H:%M:%S", None),
    (pd.DataFrame(["8:8:8"], columns=["a"]), True, "%H:%M:%S", None),
    (pd.DataFrame(["8:59:59am"], columns=["a"]), True, "%I:%M:%S%p", None),
    (pd.DataFrame(["01:00:00"], columns=["a"]), True, "%H:%M:%S", None),
    (pd.DataFrame(["23:59:59"], columns=["a"]), True, "%H:%M:%S", None),
    (pd.DataFrame(["23:59:59.59"], columns=["a"]), True, "%H:%M:%S.%f", None),
    (pd.DataFrame(["2:30 am"], columns=["a"]), True, "%I:%M %p", None),
    (pd.DataFrame(["2:01 pm"], columns=["a"]), True, "%I:%M %p", None),
    (pd.DataFrame(["4:25"], columns=["a"]), True, "%H:%M", None),
    (pd.DataFrame(["20:00"], columns=["a"]), True, "%H:%M", None),
    (pd.DataFrame(["1am"], columns=["a"]), True, "%I%p", None),
    (pd.DataFrame(["12 pm"], columns=["a"]), True, "%I %p", None),
    (pd.DataFrame([datetime.time(15)], columns=["a"]), True, "%H:%M:%S", None),
    (pd.DataFrame(["-01:01:01"], columns=["a"]), False, None, None),
    (pd.DataFrame(["24:0:0"], columns=["a"]), False, None, None),
    (pd.DataFrame(["59:0:0"], columns=["a"]), False, None, None),
    (pd.DataFrame(["3:60"], columns=["a"]), False, None, None),
    (pd.DataFrame(["12 o'clock"], columns=["a"]), False, None, None),
    (pd.DataFrame(["10pm is the time"], columns=["a"]), False, None, None),
    (pd.DataFrame(["8.5:32:32"], columns=["a"]), False, None, None),
    (pd.DataFrame([["2020-01-01T10:00:00"]], columns=["a"]), False, None, None),
    (pd.DataFrame([["2020-01-01"]], columns=["a"]), False, None, None),
    (pd.DataFrame(["08/03/1999 23:59:59"], columns=["a"]), False, None, "%M/%D/%Y %H:%M:%S"),
    (pd.DataFrame(["5"], columns=["a"]), False, None, "%C"),
    (pd.DataFrame(["1999 23"], columns=["a"]), False, None, "%Y %H"),
])
def test_infer_time_features(data, is_time, expected_format, provided_format):
    """Test IFA against many possible valid and invalid time-only features."""
    ifa = InferFeatureAttributesDataFrame(data)
    feature_type, _ = ifa._get_feature_type("a")
    if is_time:
        assert feature_type == FeatureType.TIME
        features = infer_feature_attributes(data, tight_bounds=["a"],
                                            datetime_feature_formats={"a": provided_format})
        assert features["a"]["type"] == "continuous"
        assert features["a"]["date_time_format"] == expected_format
    else:
        assert feature_type != FeatureType.TIME


@pytest.mark.parametrize("data, tight_bounds, provided_format, expected_bounds, cycle_length", [
    (
        pd.DataFrame(["00:00:00", "23:59:59"], columns=["a"]), ["a"], None,
        {"min": 0, "max": 86399, "observed_min": 0, "observed_max": 86399, "allow_null": True}, 86400
    ),
    (
        pd.DataFrame(["03:00:00.0", "12:00:01.5"], columns=["a"]), ["a"], None,
        {"min": 10800, "max": 43201.5, "observed_min": 10800, "observed_max": 43201.5, "allow_null": True}, 86400
    ),
    (
        pd.DataFrame(["03:00:00.0", "12:00:01.5"], columns=["a"]), None, None,
        {"min": 0, "max": 86400, "observed_min": 10800.0, "observed_max": 43201.5, "allow_null": True}, 86400
    ),
    (
        pd.DataFrame(["25:0", "30:0"], columns=["a"]), ["a"], "%M:%S",
        {"min": 1500, "max": 1800, "observed_min": 1500, "observed_max": 1800, "allow_null": True}, 3600
    ),
    (
        pd.DataFrame(["25.0", "30.5"], columns=["a"]), None, "%S.%f",
        {"min": 0, "max": 60, "observed_min": 25.0, "observed_max": 30.5, "allow_null": True}, 60
    ),
    (
        pd.DataFrame(["5", "7"], columns=["a"]), None, "%f",
        {"min": 0, "max": 1, "observed_min": 0.5, "observed_max": 0.7, "allow_null": True}, 1
    ),
])
def test_infer_time_feature_bounds(data, tight_bounds, provided_format, expected_bounds, cycle_length):
    """Test that IFA correctly calculates the bounds and cycle length of time-only features."""
    features = infer_feature_attributes(data, tight_bounds=tight_bounds,
                                        datetime_feature_formats={"a": provided_format})
    assert features["a"]["type"] == "continuous"
    assert "cycle_length" in features["a"]
    assert features["a"]["cycle_length"] == cycle_length
    bounds = features["a"]["bounds"]
    # `nulls_observed` is always present under `bounds`; its value is covered by test_feature_contains_nulls
    assert "nulls_observed" in bounds
    del bounds["nulls_observed"]
    assert bounds == expected_bounds
    assert features["a"]["date_time_format"] is not None
    assert features["a"]["data_type"] == "formatted_time"


@pytest.mark.parametrize("data, data_type", [
    (123, "float128"),
])
def test_get_feature_type_raises(data, data_type):
    """Test get_feature_type raises exception."""
    # Place this here to avoid circular import
    from howso.client.exceptions import HowsoError
    if not hasattr(np, data_type):
        pytest.skip("Unsupported platform")

    with pytest.raises(HowsoError):
        df = pd.DataFrame([[getattr(np, data_type)(data)]], columns=["a"])
        infer_feature_attributes(df)


def test_excessive_float_precision_warning():
    """Test that features exceeding 64-bit float precision are reported in a single warning."""
    # Place this here to avoid circular import
    from howso.utilities.feature_attributes.pandas import InferFeatureAttributesDataFrame
    if not hasattr(np, "float128"):
        pytest.skip("Unsupported platform")

    df = pd.DataFrame({
        "a": np.arange(20, dtype=np.float128) + 0.5,
        "b": np.arange(20, dtype=np.float128) * 1.5,
        "c": np.arange(20, dtype="float64") + 0.25,
    })
    ifa = InferFeatureAttributesDataFrame(df)
    ifa.attributes = {}
    attributes = {feature: ifa._infer_floating_point_attributes(feature) for feature in df.columns}

    # Features beyond the supported precision get no `decimal_places`
    assert "decimal_places" not in attributes["a"]
    assert "decimal_places" not in attributes["b"]
    assert attributes["c"]["decimal_places"] == 2

    with pytest.warns(UserWarning, match="exceed the maximum supported precision") as record:
        ifa.warnings_collector.emit_all()

    assert len(record) == 1
    message = str(record[0].message)
    assert "- a" in message and "- b" in message
    assert "- c" not in message


@pytest.mark.parametrize("should_fail, data", [
    (True, [[1]]),
    (True, {3: [1]}),
    (False, {"col1": [1]}),
])
def test_column_names(should_fail, data):
    """Test invalid column names raises."""
    df = pd.DataFrame(data)
    if should_fail:
        expected_msg = r"Unexpected DataFrame column name format"
        with pytest.raises(ValueError, match=expected_msg):
            infer_feature_attributes(df)
    else:
        features = infer_feature_attributes(df)
        assert features is not None


@pytest.mark.parametrize("should_include, dependent_features", [
    (False, None),
    (True, {"sepal_length": ["sepal_width", "class"]}),
    (True, {"sepal_width": ["sepal_length"]}),
    (True, {"sepal_length": ["class"]}),
    (False, None),
    (True, None),
])
def test_dependent_features(should_include, dependent_features):
    """Test depdendent features are added to feature attributes dict."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df, dependent_features=dependent_features)

    if should_include:
        # Should include dependent features
        if dependent_features:
            for feat, dep_feats in dependent_features.items():
                assert "dependent_features" in features[feat]
                for dep_feat in dep_feats:
                    assert dep_feat in features[feat]["dependent_features"]
    else:
        # Should not include dependent features
        for attributes in features.values():
            assert "dependent_features" not in attributes


@pytest.mark.parametrize("tight_bounds, data, expected_bounds", [
    (None, [2, 3, 4, 5, 6, 7], {"min": 0, "max": 10, "observed_min": 2, "observed_max": 7, "allow_null": True}),
    (None, [2, 3, 4, 4, 5, 6, 6, 6, 6], {"min": 0, "max": 6, "observed_min": 2, "observed_max": 6, "allow_null": True}),  # noqa: E501
    (None, [2, 3, 4, 4, 4, 4, 6, 6, 6, 6], {"min": 0, "max": 6.0, "observed_min": 2.0, "observed_max": 6.0, "allow_null": True}),  # noqa: E501
    (None, [2, 2, 2, 2, 4, 5, 6, 6, 6, 6], {"min": 2.0, "max": 6.0, "observed_min": 2.0, "observed_max": 6.0, "allow_null": True}),  # noqa: E501
    (None, [2, 2, 2, 2, 4, 5, 6, 6, 6, 6, 6], {"min": 0, "max": 6.0, "observed_min": 2.0, "observed_max": 6.0, "allow_null": True}),  # noqa: E501
    (None, [2, 2, 2, 2, 4, 5, 6, 7], {"min": 2.0, "max": 10.0, "observed_min": 2.0, "observed_max": 7.0, "allow_null": True}),  # noqa: E501
    (None, [float("nan"), float("nan")], {"allow_null": True}),
    (["a"], [2, 3, 4, 5, 6, 7], {"min": 2, "max": 7, "observed_min": 2, "observed_max": 7, "allow_null": True}),
    (["a"], [2, 3, 4, None, 6, 7], {"min": 2, "max": 7, "observed_min": 2, "observed_max": 7, "allow_null": True}),
    (
        ["a"],
        ["1905-01-01", "1904-05-03", "2020-01-15", "2000-04-26", "2000-04-24"],
        {"min": "1904-05-03", "max": "2020-01-15", "observed_min": "1904-05-03", "observed_max": "2020-01-15"}
    ),
    (
        None,
        ["1905-01-01", "1904-05-03", "2020-01-15", "2000-04-26", "2000-04-24"],
        {"min": "1829-04-11", "max": "2095-02-04", "observed_min": "1904-05-03", "observed_max": "2020-01-15"}
    ),
    (
        None,
        ["1905-01-01", "1904-05-03", "2020-01-15", "2020-01-15", "2020-01-15",
         "2020-01-15", "2000-04-26", "2000-04-24"],
        {"min": "1829-04-11", "max": "2020-01-15", "observed_min": "1904-05-03", "observed_max": "2020-01-15"}
    ),
    (
        None,
        ["1905-01-01", "1904-05-03", "1904-05-03", "1904-05-03", "1904-05-03",
         "2020-01-15", "2020-01-15", "2020-01-15", "2020-01-15", "2000-04-26",
         "2000-04-24"],
        {"min": "1904-05-03", "max": "2020-01-15", "observed_min": "1904-05-03", "observed_max": "2020-01-15"}
    ),
    (
        None,
        ["1905-01-01T00:00:00+0100", "2022-03-26T00:00:00+0500",
         "1904-05-03T00:00:00+0500", "1904-05-03T00:00:00+0500",
         "1904-05-03T00:00:00+0500", "1904-05-03T00:00:00-0200",
         "1904-05-03T00:00:00+0500", "2022-01-15T00:00:00+0500"],
        {"min": "1904-05-03T00:00:00+0500", "max": "2098-09-17T14:04:45+0500",
         "observed_min": "1904-05-03T00:00:00+0500", "observed_max": "2022-03-26T00:00:00+0500"}
    ),
    (
        ["a"],
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {"min": "1904-05-03", "max": "2022-03-26",
         "observed_min": "1904-05-03", "observed_max": "2022-03-26"}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {"min": "1827-11-08", "max": "2098-09-17",
         "observed_min": "1904-05-03", "observed_max": "2022-03-26"}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(1904, 5, 3), datetime.datetime(1904, 5, 3),
         datetime.datetime(1904, 5, 3), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {"min": "1904-05-03", "max": "2098-09-17",
         "observed_min": "1904-05-03", "observed_max": "2022-03-26"}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(2020, 1, 15, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(2022, 3, 26, tzinfo=datetime.timezone(datetime.timedelta(minutes=300)))],
        {"min": "1904-05-03T00:00:00+0500", "max": "2098-09-17T14:04:45+0500",
         "observed_min": "1904-05-03T00:00:00+0500", "observed_max": "2022-03-26T00:00:00+0500"}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(minutes=100))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=-400))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(1904, 5, 3, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(2020, 1, 15, tzinfo=datetime.timezone(datetime.timedelta(minutes=300))),
         datetime.datetime(2022, 3, 26, tzinfo=datetime.timezone(datetime.timedelta(minutes=300)))],
        {"min": "1904-05-03T00:00:00+0500", "max": "2098-09-17T14:04:45+0500",
         "observed_min": "1904-05-03T00:00:00+0500", "observed_max": "2022-03-26T00:00:00+0500"}
    ),
    (
        ["a"],
        [datetime.timedelta(days=1), datetime.timedelta(days=1),
         datetime.timedelta(seconds=5), datetime.timedelta(days=1, seconds=30),
         datetime.timedelta(minutes=50), datetime.timedelta(days=5)],
        {"min": 5, "max": 5 * 24 * 60 * 60,
         "observed_min": 5, "observed_max": 5 * 24 * 60 * 60,
         "allow_null": True, "allow_null": True}
    ),
    (
        None,
        [datetime.timedelta(days=1), datetime.timedelta(days=1),
         datetime.timedelta(seconds=5), datetime.timedelta(days=1, seconds=30),
         datetime.timedelta(minutes=50), datetime.timedelta(days=5),
         datetime.timedelta(days=5), datetime.timedelta(days=5),
         datetime.timedelta(days=5)],
        {"min": 0, "max": 5 * 24 * 60 * 60.0,
         "observed_min": 5.0, "observed_max": 5 * 24 * 60 * 60.0,
         "allow_null": True, "allow_null": True}
    ),
])
def test_infer_feature_bounds(data, tight_bounds, expected_bounds):
    """Test the infer_feature_bounds() method."""
    df = pd.DataFrame(pd.Series(data), columns=["a"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = infer_feature_attributes(df, tight_bounds=tight_bounds)
    assert features["a"]["type"] == "continuous"
    assert "bounds" in features["a"]
    bounds = features["a"]["bounds"]
    # `nulls_observed` is always present under `bounds`; its value is covered by test_feature_contains_nulls
    assert "nulls_observed" in bounds
    del bounds["nulls_observed"]
    assert bounds == expected_bounds


def test_to_json() -> None:
    """Test that to_json() method returns a JSON representation of the object."""
    df = pd.read_csv(iris_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred_features = infer_feature_attributes(df)

    to_json = inferred_features.to_json()
    assert isinstance(to_json, str)
    features = json.loads(to_json)

    # Make sure the json representation has expected data
    for k, v in inferred_features.items():
        assert v["type"] == features[k]["type"]
        if "bounds" in v:
            assert v["bounds"] == features[k]["bounds"]


@pytest.mark.parametrize("dependent_features", [
    {"sepal_length": ["class"],
     "sepal_width": ["petal_width"]}
])
def test_get_parameters(dependent_features):
    """Test the get_parameters() method."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df, dependent_features=dependent_features)

    # Verify dependent_features
    assert "dependent_features" in features.get_parameters()
    for key, value in dependent_features.items():
        assert features.get_parameters()["dependent_features"][key] == value


def test_get_names_without():
    """Test the get_names() method."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)

    all_features = list(df)

    # Test get all feature names
    assert features.get_names() == all_features

    # Test get feature names without
    without = ["sepal_length", "petal_length", "petal_width"]
    assert features.get_names(without=without) == [f for f in all_features if f not in without]

    # Test a feature in 'without' that is not in the features list
    with pytest.raises(ValueError):
        without = ["sepal_length", "petal_length", "personality"]
        features.get_names(without=without)


@pytest.mark.parametrize("types, data_types, num", [
    ("continuous", None, 4),
    ({"continuous"}, None, 4),
    (("nominal"), None, 1),
    (["continuous", "nominal"], None, 5),
    ("continuous", ["number"], 4),
    (("nominal"), ["string"], 1),
    (["continuous", "nominal"], ["string", "number"], 5),
    (("nominal"), ["boolean"], 0),
])
def test_get_names_types(types, data_types, num):
    """Test the get_names() method with the types and/or data_types parameter(s)."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)
    names = features.get_names(types=types, data_types=data_types)
    assert len(names) == num


def test_copy():
    """Test that copy works as expected."""
    df = pd.read_csv(iris_path)
    f_orig = infer_feature_attributes(df)
    f_copy = copy(f_orig)

    assert f_copy.params == f_orig.params
    orig = f_orig["sepal_width"]["bounds"]["min"]
    assert f_copy["sepal_width"]["bounds"]["min"] == orig

    # Now, change the orig, so we can ensure that f_copy is independent.
    f_orig["sepal_width"]["bounds"]["min"] = -2
    # Assert that f_copy was unaffected
    assert f_copy["sepal_width"]["bounds"]["min"] == orig


@pytest.mark.parametrize("tight_bounds", [
    (["DATE", "TURNOVER", "%DELIVERABLE"]),
    (["DATE", "%DELIVERABLE"]),
    (["DATE", "TURNOVER"]),
    (["%DELIVERABLE", "TURNOVER"]),
    (["DATE"]),
    (["TURNOVER"]),
    (["%DELIVERABLE"]),
    ([""])
])
def test_tight_bounds(tight_bounds):
    """Test the tight_bounds argument with a features list."""
    df = pd.read_csv(stock_path)
    features = infer_feature_attributes(df, tight_bounds=tight_bounds)

    all_tight_bounds = infer_feature_attributes(df, tight_bounds=features.get_names())
    no_tight_bounds = infer_feature_attributes(df)

    for feature in features.keys():
        if "bounds" not in features[feature]:
            continue
        if feature in tight_bounds:
            assert features[feature]["bounds"] == all_tight_bounds[feature]["bounds"]
        else:
            assert features[feature]["bounds"] == no_tight_bounds[feature]["bounds"]


def test_validate_dataframe():
    """Test the validate method with a DataFrame."""
    # Test valid feature attributes against their original datasets
    # (should not raise any exceptions!)
    # Iris dataset
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)
    assert features.validate(df, raise_errors=True) is None
    # Integers dataset
    df = pd.read_csv(int_path)
    features = infer_feature_attributes(df)
    assert features.validate(df, raise_errors=True) is None
    # Example timeseries dataset
    df = pd.read_csv(ts_path)
    features = infer_feature_attributes(df, time_feature_name="date")
    assert features.validate(df, raise_errors=True) is None
    # Mini stock data dataset
    df = pd.read_csv(stock_path)
    features = infer_feature_attributes(df, time_feature_name="DATE")
    assert features.validate(df, raise_errors=True) is None
    # Also try this one with a non-ts infer
    df = pd.read_csv(stock_path)
    features = infer_feature_attributes(df)
    assert features.validate(df, raise_errors=True) is None
    # Should not raise any exceptions and return a "coerced" dataframe
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)
    df["sepal_length"] = df["sepal_length"].astype("int64")
    df = features.validate(df, coerce=True, raise_errors=True)
    assert df is not None
    assert pd.api.types.is_float_dtype(df["sepal_length"])
    # Try validating a categorical feature
    df = pd.read_csv(iris_path)
    features["class"]["type"] = "ordinal"
    features["class"]["bounds"] = {}
    unique = list(df["class"].unique())
    features["class"]["bounds"]["allowed"] = unique
    df["class"] = df["class"].astype(pd.CategoricalDtype(categories=unique))
    df = features.validate(df, coerce=True, raise_errors=True)
    assert df is not None
    assert isinstance(df["class"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("ftype, data_type, decimal_places, bounds, date_time_format, expected_dtype", [
    ("continuous", "number", 0, {"allow_null": True}, None, "int64"),
    ("continuous", "number", 1, {"allow_null": True}, None, "float64"),
    ("continuous", "number", 0, {"allow_null": True}, "%Y-%m-%d", "datetime64"),
    ("ordinal", "number", 0, {"allow_null": True}, None, "int64"),
    ("ordinal", "number", 2, {"allow_null": True}, None, "float64"),
    ("ordinal", "string", None, {"allowed": ["SBIN"], "allow_null": True}, None, "object"),
    ("nominal", "number", 0, {"allow_null": True}, None, "int64"),
    ("nominal", "number", 9, {"allow_null": True}, None, "float64"),
    ("nominal", "boolean", 0, {"allow_null": True}, None, "bool"),
])
def test_validate_df_multiple_dtypes(ftype, data_type, decimal_places, bounds, date_time_format,
                                     expected_dtype):
    """Test the validate() method with all possible inferred dtypes."""
    # First, read in the mini_stock_series dataset as it has a variety of data types
    df = pd.read_csv(stock_path)
    # Based on the expected_dtype, choose the feature in the dataset that is loosely described by the given parameters
    if expected_dtype == "int64":
        feature = "VOLUME"
    elif expected_dtype == "float64":
        feature = "PREV CLOSE"
    elif expected_dtype == "datetime64":
        feature = "DATE"
    elif expected_dtype == "bool":
        # Make a new column of a bool dtype since there are none in the dataset
        df["NEW"] = True
        feature = "NEW"
    else:
        feature = "SYMBOL"
    # Infer the feature attributes like normal, but replace the attributes for the chosen feature
    # with our parameter attributes, which should also be considered valid.
    attrs = infer_feature_attributes(df, time_feature_name="DATE")
    attrs[feature] = {
        "type": ftype,
        "data_type": data_type,
        "decimal_places": decimal_places,
        "bounds": bounds,
        "date_time_format": date_time_format,
    }
    if not date_time_format:
        del attrs[feature]["date_time_format"]
    # validate() should not raise any errors
    coerced_df = attrs.validate(df, raise_errors=True, coerce=True)
    assert coerced_df is not None
    # coerced_df should also contain a coerced DATE column, as it is originally detected as a string
    assert pd.api.types.is_datetime64_any_dtype(coerced_df["DATE"].dtype)


@pytest.mark.parametrize(
    ("data", "expected_data_type", "expected_orig_type"),
    (
        ({"a": 1}, "json", "container"),
        ([1, 2, 3], "json", "container"),
        ('{"a": 1}', "json", "string"),
        ('["a", "b", "c"]', "json", "string"),
        ("doc:\n  abc: 1", "yaml", "string"),
        ("(list 1 2 3)", "amalgam", "string"),
        ('(assoc "a" 1 "b" 2)', "amalgam", "string"),
    ),
)
def test_validate_df_semi_structured(data, expected_data_type: str, expected_orig_type: str):
    """Test validate_df handles semi-structured features correctly."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "doc": [data, None, data],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feature_attributes = infer_feature_attributes(df, enable_suggestions=False)

    attrs = feature_attributes["doc"]

    assert "original_type" in attrs
    assert attrs["original_type"]["data_type"] == expected_orig_type

    if expected_data_type == "amalgam":
        # Amalgam is not automatically inferred set it manually
        attrs["type"] = "continuous"
        attrs["data_type"] = "amalgam"
    else:
        assert attrs["type"] == "continuous"
        assert attrs.get("data_type") == expected_data_type

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feature_attributes.validate(df, raise_errors=True)


@pytest.mark.parametrize("values", (
    [True, False, pd.NA, True],
    [True, False, False, True],
))
@pytest.mark.parametrize("dtype", ("boolean", "bool[pyarrow]", object))
@pytest.mark.parametrize("coerce", (False, True))
def test_validate_df_nullable_boolean(values, dtype, coerce):
    """Test validate_df accepts boolean columns whose nulls are `pd.NA`, keeping their nulls."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "flag": pd.array(values, dtype=dtype),
    })
    feature_attributes = infer_feature_attributes(df, enable_suggestions=False)
    assert feature_attributes["flag"]["data_type"] == "boolean"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = feature_attributes.validate(df, coerce=coerce, raise_errors=True)

    if coerce:
        has_nulls = any(value is pd.NA for value in values)
        if not has_nulls:
            assert result["flag"].dtype.name == "bool"
        assert result["flag"].isna().sum() == int(has_nulls)


@pytest.mark.parametrize("dtype", ("datetime64[ns]", "timestamp[ns][pyarrow]"))
def test_validate_df_coerce_localizes_datetimes(dtype):
    """
    Test validate_df coercion localizes naive numpy and pyarrow datetimes to UTC.

    The result is a numpy datetime, whose UTC localization does not depend on pyarrow finding a
    timezone database (which it looks for in the user's Downloads folder on Windows).
    """
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "date": pd.array(pd.to_datetime(["2020-01-01", None, "2020-03-08", "2021-11-07"]), dtype=dtype),
    })
    feature_attributes = infer_feature_attributes(df, enable_suggestions=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = feature_attributes.validate(df, coerce=True, raise_errors=True)

    assert isinstance(result["date"].dtype, pd.DatetimeTZDtype)
    assert str(result["date"].dt.tz) == "UTC"
    assert result["date"].isna().sum() == 1


def test_validate_df_timedelta():
    """Test validate_df accepts timedelta columns and checks them against bounds in seconds."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "duration": pd.to_timedelta(["1D", "2D", None, "1D"]),
    })
    feature_attributes = infer_feature_attributes(df, enable_suggestions=False)
    bounds = feature_attributes["duration"]["bounds"]
    assert (bounds["observed_min"], bounds["observed_max"]) == (86400.0, 172800.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = feature_attributes.validate(df, coerce=True, raise_errors=True)
        assert pd.api.types.is_timedelta64_dtype(result["duration"].dtype)

        del feature_attributes["duration"]["original_type"]
        feature_attributes.validate(df, raise_errors=True)

    feature_attributes["duration"]["bounds"]["max"] = 90000.0
    with pytest.raises(ValueError, match="outside of bounds"):
        feature_attributes.validate(df, raise_errors=True)


@pytest.mark.parametrize("extra_attrs, success", (
    ({}, False),
    ({"auto_derive_on_train": False}, False),
    ({"auto_derive_on_train": True}, False),
    ({"derived_feature_code": "{* #VOLUMNE 2.2}"}, False),
    ({"auto_derive_on_train": False,
      "derived_feature_code": "{* #VOLUMNE 2.2}"}, False),
    ({"auto_derive_on_train": True,
      "derived_feature_code": "{* #VOLUMNE 2.2}"}, True),
))
def test_validate_df_missing_features(extra_attrs, success):
    """
    Test that missing features raise warnings in `_validate_df`.

    Specifically, if a feature is to be derived during train, it should be
    exempt from raising warnings that the feature is missing.
    """
    df = pd.read_csv(stock_path)
    attrs = infer_feature_attributes(df, time_feature_name="DATE")
    # Add a would-be derived/computed feature
    attrs["to_be_computed"] = {"type": "continuous"}
    attrs["to_be_computed"].update(extra_attrs)

    if success:
        # We expect this to run without raising an error (due to the warning)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            attrs.validate(df)
    else:
        # We expect this to raise an exception when run.
        with pytest.raises(Exception), warnings.catch_warnings():
            warnings.simplefilter("error")
            attrs.validate(df)


@pytest.mark.parametrize("datetime_min_max, float_min_max, int_min_max", [
    (
        ("DATE", None, None, False),
        ("CLOSE", None, None, False),
        ("VOLUME", None, None, False),
    ),
    (
        ("DATE", None, DT_MAX, True),
        ("CLOSE", FLOAT_MIN - .00001, FLOAT_MAX + .00001, True),
        ("VOLUME", int(INTEGER_MAX / 10) * -1, INTEGER_MAX, True),
    ),
    (
        ("DATE", ALMOST_DT_MAX, None, False),
        ("CLOSE", -1 * (FLOAT_MAX / 10.0), None, False),
        ("VOLUME", -1 * int(INTEGER_MAX / 10), None, False),
    ),
    (
        ("DATE", DT_MAX, ALMOST_DT_MAX, True),
        ("CLOSE", FLOAT_MIN * -0.1, FLOAT_MIN * -0.01, True),
        ("VOLUME", INTEGER_MAX * -1, (INTEGER_MAX * -1) - 1, True),
    ),
    (
        ("DATE", None, None, False),
        ("CLOSE", FLOAT_MAX * -1.0, (FLOAT_MAX * -1.0) - 1, True),
        ("VOLUME", None, None, False),
    ),
])
def test_unsupported_data(datetime_min_max, float_min_max, int_min_max):
    """Test that infer_feature_attributes correctly identifies features that contain unsupported data."""
    df = pd.read_csv(stock_path)

    expected_unsupported = {}

    for feature, val_1, val_2, unsupported in [datetime_min_max, float_min_max, int_min_max]:
        if val_1 is not None:
            df.at[0, feature] = val_1
        if val_2 is not None:
            df.at[1, feature] = val_2
        expected_unsupported[feature] = unsupported

    features = infer_feature_attributes(df, tight_bounds=["DATE", "CLOSE", "VOLUME"])

    for feature in features.keys():
        if expected_unsupported.get(feature, False):
            assert features.has_unsupported_data(feature)
        else:
            assert not features.has_unsupported_data(feature)


@pytest.mark.parametrize("value, is_json, is_yaml", [
    ('{"key": "value", "_key": "_value"}', True, False),
    ('{"key":\n    {"key2": [1, 2, 3, 4]}\n}', True, False),
    ("[]", True, False),
    ("{}", True, False),
    ("a: 1\nb:\nc: 3\n\nd: 4", False, True),
    ("---\nname: The Howso Engine.\ndescription: >\n  The Howso Engine™ is a "
     "natively and fully explainable ML engine and toolbox.", False, True),
    ("not:valid:\nyaml\norjson", False, False),
    (12345, False, False),
    ("abcdefg", False, False),
    ("abcd\nefg", False, False),
    (None, False, False),
    ([1, 2, 3, 4], True, False),
    ({"a": "b", "c": "d"}, True, False),
])
def test_json_yaml_features(value, is_json, is_yaml):
    """Test that infer_feature_attributes correctly identifies JSON and YAML features."""
    df = pd.DataFrame({"a": [value]})

    features = infer_feature_attributes(df)

    if is_json:
        assert features["a"]["type"] == "continuous"
        assert features["a"]["data_type"] == "json"
    elif is_yaml:
        assert features["a"]["type"] == "continuous"
        assert features["a"]["data_type"] == "yaml"
    else:
        assert features["a"].get("data_type") != "json"
        assert features["a"].get("data_type") != "yaml"


@pytest.mark.parametrize("max_workers", [0, 2])
@pytest.mark.parametrize("data, types, expected_types, is_valid", [
    (pd.DataFrame({"a": [0, 1, 2, 0, 1, 2]}), dict(a="continuous"), dict(a="continuous"), True),
    (pd.DataFrame({"a": [0, 1, 2], "b": ["1", "2", "3"]}, columns=["a", "b"]), dict(continuous=["a", "b"]),
     dict(a="continuous", b="continuous"), True),
    (pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7]}), dict(a="nominal"), dict(a="nominal"), True),
    (pd.DataFrame({"a": [True, False, False, True]}), dict(a="continuous"), dict(a="nominal"), True),
    (pd.DataFrame({"nominal": [True, False, False, True]}), dict(nominal="nominal"), dict(nominal="nominal"), True),
    (pd.DataFrame({"nominal": [True, False, False, True]}), dict(nominal=["nominal"]), dict(nominal="nominal"), True),
    (pd.DataFrame({"ordinal": [True, False, False, True]}), dict(ordinal="nominal"), dict(ordinal="nominal"), True),
    (pd.DataFrame({"continuous": [True, False, False]}), dict(continuous="nominal"), dict(continuous="nominal"), True),
    (pd.DataFrame({"a": [True, False, False, True]}), dict(a="boolean"), {}, False),
    (pd.DataFrame({"a": ["one", "two", "three", "four"]}), dict(ordinal=["a"]), {}, False),
    (pd.DataFrame({"a": ["one", "two", "three", "four"]}), dict(a="ordinal"), {}, False),
])
def test_preset_feature_types(data, types, expected_types, is_valid, max_workers):
    """Test that infer_feature_attributes correctly presets feature types with the `types` parameter."""
    features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_valid:
            features = infer_feature_attributes(data, types=types, max_workers=max_workers)
            for feature_name, expected_type in expected_types.items():
                # Make sure it is the correct type
                assert features[feature_name]["type"] == expected_type
                # All features in this test, including nominals, should have bounds (at the very least: `allow_null`)
                assert "allow_null" in features[feature_name].get("bounds", {}).keys()
        else:
            with pytest.raises(ValueError):
                infer_feature_attributes(data, types=types, max_workers=max_workers)


def test_preset_feature_types_with_multiprocessing():
    """Test that the `types` parameter behaves well with multiprocessing enabled."""
    df = pd.read_csv(stock_path)
    # Identified continuous features
    continuous = ["CLOSE"]
    # Everything else is nominal, in this case.
    nominals = [f for f in df.columns if f not in continuous]
    features = infer_feature_attributes(df, types={"nominal": nominals, "continuous": continuous}, max_workers=2)
    assert features is not None


def test_feature_order():
    """Test that `infer_feature_attributes` returns features in same order as the DataFrame columns."""
    def same_order(one: Iterable, two: Iterable) -> bool:
        for idx, k in enumerate(one):
            if idx >= len(two) or k != two[idx]:
                return False
        return True
    df = pd.read_csv(stock_path)
    continuous = ["CLOSE"]
    nominals = [f for f in df.columns if f not in continuous]
    features = infer_feature_attributes(df, types={"nominal": nominals, "continuous": continuous}, max_workers=10)
    assert same_order(features.keys(), df.columns)
    # Try it without multiprocessing as well
    features = infer_feature_attributes(df, types={"nominal": nominals, "continuous": continuous}, max_workers=0)
    assert same_order(features.keys(), df.columns)


def test_archival():
    """
    Test that archival of the FeatureAttributes instance works as expected.

    This also tests that tuples as keys or values of IFA parameters are
    preserved through the archival process (to_json / from_json).
    """
    data = pd.DataFrame({
        "a": [0, 1, 2, 3, 4, 5, 6, 7],
        "b": ["apple", "banana", "banana", "cherry", "apple", "apple", "cherry", "banana"],
        "c": [2, 3, 4, 2, 3, 4, 2, 3],
        "d": [1, 1, 1, 1, 2, 2, 2, 2],
        # NOTE: That the second red is not a typo.
        "e": ["red", "orange", "yellow", "red", "green", "blue", "indigo", "violet"],
    })
    features = infer_feature_attributes(
        data,
        tight_bounds=["a"],
        fanout_feature_map={
            ("c", "d"): ("e", ),
        }
    )
    assert features["a"]["type"] == "continuous"
    assert features["b"]["type"] == "nominal"

    archive = features.to_json(archive=True)
    new_features = FeatureAttributesBase.from_json(archive)

    assert new_features["a"]["type"] == "continuous"
    assert new_features["b"]["type"] == "nominal"
    assert new_features.params["tight_bounds"] == ["a"]
    fanout_feature_key = list(new_features.params["fanout_feature_map"].keys())[0]
    assert isinstance(fanout_feature_key, tuple)
    assert isinstance(new_features.params["fanout_feature_map"][fanout_feature_key], tuple)


def test_disk_archival():
    """Test that archival of the FeatureAttributes instance to disk works as expected."""
    data = pd.DataFrame({
        "a": [0, 1, 2, 3, 4, 5, 6, 7],
        "b": ["apple", "banana", "banana", "cherry", "apple", "apple", "cherry", "banana"]
    })
    features = infer_feature_attributes(data, tight_bounds=["a"])
    assert features["a"]["type"] == "continuous"
    assert features["b"]["type"] == "nominal"

    with TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir, "fa_archive.json")

        features.to_json(archive=True, json_path=json_path)
        new_features = FeatureAttributesBase.from_json(json_path=json_path)

    assert new_features["a"]["type"] == "continuous"
    assert new_features["b"]["type"] == "nominal"
    assert new_features.params["tight_bounds"] == ["a"]


@pytest.mark.parametrize(
    "series, ordinals, min_value, max_value", [
        (  # ordinal strings
            ["grape", "apple", "banana", "banana", "cherry", "apple", "apple", "fig", "cherry", "banana"],
            ["apple", "banana", "cherry"],
            "apple", "cherry"
        ),
        (  # ordinal strings, includes an empty string
            ["**", "*", "***", "*", "****", "*****", "**", "", "****", "***"],
            ["", "*", "**", "***", "****", "*****"],
            "", "*****"
        ),
        (  # ordinal numerals
            [4, 2, 1, 7, 3, 3, 8, 2, 1, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            0, 8
        ),
        (  # ordinal numerals crossing zero
            [-4, 2, 4, 7, -2, -6, 8, 2, 1, 0],
            [-8, -6, -4, -2, 0, 2, 4, 6, 8],
            -6, 8
        ),
        (  # ordinal numerals as strings.
            ["4", "2", "1", "7", "3", "3", "8", "2", "1", "0"],
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "1", "8"
        ),
        (  # ordinal numerals as string ordinals with unusual ordering
            ["4", "2", "1", "7", "3", "3", "8", "2", "1", "0"],
            ["5", "7", "3", "2", "4", "9", "10", "8", "6", "1"],
            "7", "1"
        ),
        (  # floats as ordinals
            [0.4, 0.2, 0.1, 0.7, 0.3, 0.3, 0.8, 0.2, 0.1, 0.0],
            [0.1, 0.2, 0.3, 0.4, 5, 6, 7, 8, 9, 10],
            0.0, 0.8
        ),
        (  # Dates as ordinals
            ["1-Mar-2020", "1-Mar-2020", "1-Apr-2020", "1-Mar-2020", "1-Feb-2020", "1-Dec-2020", "1-Jul-2020"],
            ["1-Jan-2020", "1-Feb-2020", "1-Mar-2020", "1-Apr-2020", "1-May-2020", "1-Jun-2020"],
            "1-Feb-2020", "1-Apr-2020"
        )
    ]
)
def test_observed_ordinal_values(series, ordinals, min_value, max_value):
    """Test that observed_min/max in ordinal features works as expected."""
    data = pd.DataFrame({"a": series})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = infer_feature_attributes(data, ordinal_feature_values={"a": ordinals})
    assert features["a"]["bounds"]["observed_min"] == min_value
    assert features["a"]["bounds"]["observed_max"] == max_value


def test_formatted_date_time():
    """Test formatted_date_time is set when a datetime, and raises when no date_time_format is specified."""
    data = pd.DataFrame({
        "a": [0, 1, 2, 3],
        "time": ["10-10", "04-25", "10-30", "12-01"],
        "custom": ["2010/10/10", "2010/10/11", "2010/10/12", "2010/10/14"],
        "iso": ["2010-10-10", "2010-10-11", "2010-10-12", "2010-10-14"]
    })

    # Verify formatted_date_time is set when a date_time_format is configured
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features = infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d"},
                                            default_time_zone="UTC", enable_suggestions=False)
        assert features["a"]["data_type"] != "formatted_date_time"
        # custom feature dates should be formatted_date_time
        assert features["custom"]["data_type"] == "formatted_date_time"
        # auto detected iso dates should be formatted_date_time
        assert features["iso"]["data_type"] == "formatted_date_time"


def test_default_time_zone():
    """Test that ``infer_feature_attributes`` correctly handles default time zones."""
    data = pd.DataFrame({
        "custom": ["2010/10/10 7:30", "2010/10/11 8:45", "2010/10/12 9:00", "2010/10/14 12:00"],
        "custom2": ["2002/10/10 3:30", "2000/10/11 10:45", "2013/10/12 5:00", "2014/10/14 11:00"],
        "custom3": ["2010/10/10 07:30 -0500", "2010/10/11 08:45 -0500", "2010/10/12 09:00 -0500",
                    "2012/12/12 06:00 -0500"],
    })

    # No default time zone or time zone identifier in format string; warning should be raised
    with pytest.warns(match="features do not include a time zone and will default to UTC"):
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M",
                                                                 "custom2": "%Y/%m/%d %H:%M"})
        # Also try with multiprocessing
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M",
                                                                 "custom2": "%Y/%m/%d %H:%M"}, max_workers=2)

    # Using UTC offsets should also result in a warning
    with pytest.warns(match="The following features are using UTC offsets"):
        infer_feature_attributes(data, datetime_feature_formats={"custom3": "%Y/%m/%d %H:%M %z"})
        # Also try with multiprocessing
        infer_feature_attributes(data, datetime_feature_formats={"custom3": "%Y/%m/%d %H:%M %z"}, max_workers=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Providing a default_time_zone should prevent the warning
        data.drop("custom3", axis=1, inplace=True)  # Will raise an unrelated warning that we already tested for
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M",
                                                                 "custom2": "%Y/%m/%d %H:%M"}, default_time_zone="EST",
                                 enable_suggestions=False)
        data = pd.DataFrame({
            "custom": ["2010/10/10 07:30 UTC", "2010/10/11 08:45 UTC", "2010/10/12 09:00 UTC"],
            "custom2": ["2010/10/10 07:30 GMT", "2010/10/11 08:45 GMT", "2010/10/12 09:00 GMT"],
        })
        # Providing data with a time zone and corresponding format string identifier should prevent the error
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M %Z",
                                                                 "custom2": "%Y/%m/%d %H:%M %Z"},
                                 enable_suggestions=False)


def test_constrained_date_bounds():
    """Constrained datetime formats may make bounds undeterminable."""
    df = pd.DataFrame([["01"], ["02"]], columns=["date"])
    with pytest.warns(match="bounds could not be computed. This is likely due to a constrained date time format"):
        # Loose bounds may cause min bound to be > max bound if the date format is constrained
        infer_feature_attributes(df, datetime_feature_formats={"date": "%m"})


def test_nullable_integer_validation():
    """Test that IFA correctly validates data with nullable integers."""
    df = pd.DataFrame({"a": ["1", "2", "3", pd.NA, "4"]}, dtype="Int64")
    attrs = infer_feature_attributes(df)
    df = df.astype("float64")  # Force a coersion back to Int64
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        attrs.validate(df, coerce=True)


def test_memory_usage_warning():
    """Test that IFA will warn if a feature has columns that are too large."""
    df = pd.DataFrame([["a" * 1024], ["b" * 512], ["c" * 256]], columns=["big"])
    # Test that a single violating feature raises a warning
    with pytest.warns(match="feature 'big' exceeds the configured threshold"):
        infer_feature_attributes(df)
    # Test that the warning disappears when the threshold is configured to be larger
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer_feature_attributes(df, memory_warning_threshold=1024, enable_suggestions=False)
    # Test that two violating features raise a warning
    df = pd.DataFrame({"big": ["a" * 1024], "bigger": ["b" * 2048]})
    with pytest.warns(match="2 features have an average memory size exceeding the "
                      "configured threshold of 512 bytes. The feature with the largest "
                      "memory footprint is 'bigger'"):
        infer_feature_attributes(df)


@pytest.mark.skipif(nypd_arrest_df is None, reason="Cannot load Parquet files")
def test_ambiguous_datetime_format():
    """Test that a non-ISO8601 datetime feature results in a warning."""
    assert nypd_arrest_df is not None
    with pytest.warns(UserWarning, match="these features will be treated as nominal strings"):
        infer_feature_attributes(nypd_arrest_df)  # NYPD arrest data includes a non-ISO8601 date string


def test_no_warnings_datetime_feature_formats():
    """Test that providing non-ISO8601 datetime features with corresponding formats do not trigger any warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = pd.DataFrame([["01-01-2015", "2025-01"]], columns=["date", "month"])
        infer_feature_attributes(df, datetime_feature_formats={"date": "%d-%m-%Y", "month": "%Y-%m"},
                                 default_time_zone="UTC", enable_suggestions=False)


def test_datetime_empty_time_values():
    """Test that datetimes with an empty time value still are determined datetime features with the correct format."""
    df = pd.DataFrame({"a": ["2025-08-22T00:00:00"], "b": ["2025-08-22 00:00:00"]})
    features = infer_feature_attributes(df, default_time_zone="UTC")
    assert features["a"]["date_time_format"] == "%Y-%m-%dT%H:%M:%S"
    assert features["b"]["date_time_format"] == "%Y-%m-%d %H:%M:%S"


def test_empty_string_first_non_nulls():
    """Test that IFA correctly handles first non-null values that are empty strings."""
    df = pd.DataFrame({"a": ["", "ahoy", "howdy"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer_feature_attributes(df, enable_suggestions=False)
    df = pd.DataFrame({"a": ["", "ahoy", "howdy"], "b": ["\n", "8/26/2025", "8/3/1999"]})
    with pytest.warns(UserWarning, match="these features will be treated as nominal strings"):
        infer_feature_attributes(df)


def test_infer_tokenizable_string():
    """Test that IFA correctly detects and sets feature attributes for short tokenizable text."""
    data = {
        "product": [
            "turbo-encabulator",
            "banana-phone",
            "boneless-pizza",
        ],
        "rating": [
            5,
            3,
            1,
        ],
        "review": [
            "Not only provides inverse reactive current for use in unilateral phase detractors, but is also capable "
            "of automatically synchronizing cardinal gram-meters.",
            "it's ok. works well enough. the connection isn't very clear but what else can you expect from a banana.",
            "they forgot to take the bones out!!!!11"
        ],
    }
    df = pd.DataFrame(data)
    feature_attributes = infer_feature_attributes(df, types={"continuous": ["review"]})
    assert feature_attributes["review"]["data_type"] == "json"
    assert feature_attributes["review"]["type"] == "continuous"
    assert feature_attributes["review"]["original_type"]["data_type"] == "tokenizable_string"
    # Product should still be a nominal string
    assert feature_attributes["product"]["data_type"] == "string"
    assert feature_attributes["product"]["type"] == "nominal"


def test_boolean_detection():
    """Test that IFA correctly detects Python bool objects and string booleans."""
    df = pd.DataFrame()
    # Python bool
    df["boolean"] = [True, False] * 100
    feature_attributes = infer_feature_attributes(df)
    assert feature_attributes["boolean"]["data_type"] == "boolean"

    # True/False string
    df["boolean"] = ["True", "False"] * 100
    feature_attributes = infer_feature_attributes(df)
    assert feature_attributes["boolean"]["data_type"] == "boolean"

    # Possible boolean but should be inferred as string
    df["boolean"] = ["t", "f"] * 100
    feature_attributes = infer_feature_attributes(df)
    assert feature_attributes["boolean"]["data_type"] == "string"

    # Mix of booleans and non-booleans
    df["boolean"] = ["true", "false", "maybe", "another_thing"] * 50
    feature_attributes = infer_feature_attributes(df)
    assert feature_attributes["boolean"]["data_type"] == "string"


def test_dependent_features_uniques_warning():
    """Test that IFA correctly warns if a feature in a depnedent relationship has too many unique values."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 4, 4, 4, 4, 4],  # Too many
            "b": [1, 3, 3, 3, 3, 3, 3],  # Fine
            "c": [7, 7, 7, 7, 7, 7, 7],  # Fine
            "d": [8, 8, 8, 8, 8, 8, 9],  # Fine
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer_feature_attributes(df, dependent_features={"d": ["b", "c"]}, enable_suggestions=False)
    with pytest.warns(UserWarning, match="- a\n"):
        infer_feature_attributes(df, dependent_features={"d": ["b", "c", "a"]})
    with pytest.warns(UserWarning, match="- a\n"):
        infer_feature_attributes(df, dependent_features={"a": ["b", "c", "d"]})


def test_set_data():
    """Test that IFA recognized Python sets and correctly updates the original_type."""
    df = pd.DataFrame({
        "a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "b": [{1, 2, 3}, {4, 5, 6}, {7, 8, 9}],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features = infer_feature_attributes(df, enable_suggestions=False)
        assert features["a"]["data_type"] == "json"
        assert features["a"]["original_type"]["data_type"] == "container"
        assert "coercion" not in features["a"]["original_type"]
        assert features["b"]["data_type"] == "json"
        assert features["b"]["original_type"]["data_type"] == "container"
        assert features["b"]["original_type"]["coercion"] == "set"


def _percentile_df(n: int = 100_000) -> pd.DataFrame:
    """
    Build cases whose nominal values depend on their position within each block of 100.

    Feature `a` is "1" except for the last percentile, which is null; feature `b` is "x" for the
    first 95 percentiles, "y" for the next 4 and "z" for the last.
    """
    percentile = np.arange(n) % 100
    return pd.DataFrame({
        "a": np.where(percentile < 99, "1", None),
        "b": np.select([percentile < 95, percentile < 99], ["x", "y"], default="z"),
        "i": np.arange(1, n + 1),
        "mass": 1,
    })


def test_preserve_rare_values(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that IFA correctly infers and suggests `preserve_rare_values` configurations."""
    df = _percentile_df()

    # Test auto-apply with all values
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=1563, preserve_rare_values="all",
                                            significance_threshold=30, max_workers=2)
    assert "value_weight_multipliers" in features["a"]
    assert "value_weight_multipliers" in features["b"]
    # The protected value we're looking here is actually "none"
    assert _prv(features["a"])[0]["value"] is None
    # The rare value keeps the significance threshold exactly: 30 * 100,000 / (1,563 * 1,000)
    assert round(_multiplier(features["a"], None), 2) == 1.92
    # The common value funds it, and is the only other value listed
    assert round(_multiplier(features["a"], "1"), 2) == 0.99
    assert set(_multipliers(features["a"])) == {None, "1"}

    # Without `max_distilled_cases` the values are weighted for the default target, and the user is told so
    # even when, as here, the value keeps the threshold on its own at that target and needs nothing
    with pytest.warns(UserWarning, match="rare values of `a` were weighted for an assumed distillation target"):
        features = infer_feature_attributes(df, preserve_rare_values={"a": [None]}, max_workers=2)
    assert "value_weight_multipliers" not in features["a"]
    assert "preserve_rare_values" not in features["a"]

    # Test that a suggestion is issued, and summarized on the console rather than as a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=1563, significance_threshold=30)
    assert "Feature Attributes Summary" in capsys.readouterr().out
    assert not any("value_weight_multipliers" in attrs for attrs in features.values())
    # Test a suggestion application
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features.apply_suggestion("all")
    assert "value_weight_multipliers" in features["a"]
    assert "value_weight_multipliers" in features["b"]

    values_map = features.suggestions.preserve_rare_values.get_values_map()
    config = features.suggestions.preserve_rare_values.get_config()

    # The machine-readable parameters match the getters and survive a JSON round trip
    payload = json.loads(features.suggestions.to_json())
    prv = next(sug for sug in payload["suggestions"] if sug["name"] == "preserve_rare_values")
    assert prv["can_apply"] is True
    assert prv["caveats"] == []
    assert prv["parameters"]["preserve_rare_values"] == values_map
    assert set(prv["details"]["value_weight_multipliers"]) == set(config)
    assert prv["details"]["num_features"] == len(config)
    assert 0 < len(prv["details"]["top_values"]) <= 5

    # Supplying the suggested config, or the values map with the same target, reproduces the suggestion silently
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features = infer_feature_attributes(df, preserve_rare_values=config, enable_suggestions=False)
        assert _prv(features["a"]) == config["a"]["value_weight_multipliers"]
        assert _prv(features["b"]) == config["b"]["value_weight_multipliers"]
        features = infer_feature_attributes(df, preserve_rare_values=values_map, max_distilled_cases=1563,
                                            significance_threshold=30, enable_suggestions=False)
        assert _multipliers(features["a"]) == pytest.approx(
            {e["value"]: e["multiplier"] for e in config["a"]["value_weight_multipliers"]})
        assert _multipliers(features["b"]) == pytest.approx(
            {e["value"]: e["multiplier"] for e in config["b"]["value_weight_multipliers"]})

    # Test data with unhashable values
    df["unhashable"] = [[1, 2]] * len(df)  # lists are unhashable; value_counts will raise TypeError

    with pytest.warns(UserWarning, match="Could not process some value counts"):
        infer_feature_attributes(df, types={"unhashable": "nominal"})

def _rare_values_df() -> pd.DataFrame:
    """
    100,000 cases where feature `a` has one common value, five rare values and two small values.

    The rare values have 200 cases each, and the small values have 10 cases each, fewer than the
    default significance threshold of 30.
    """
    values = (["common"] * 98_980 + [f"rare{i}" for i in range(5) for _ in range(200)]
              + ["small0"] * 10 + ["small1"] * 10)
    return pd.DataFrame({"a": values, "i": range(len(values))})


def _prv(attributes: Mapping[str, Any]) -> list[ProtectedValueMultiplier]:
    """Get a feature's computed value weight multipliers."""
    return attributes["value_weight_multipliers"]


def _multipliers(attributes: Mapping[str, Any]) -> dict[Any, float]:
    """Get a feature's value weight multipliers, by value."""
    return {cfg["value"]: cfg["multiplier"] for cfg in _prv(attributes)}


def _multiplier(attributes: Mapping[str, Any], value: Any) -> float:
    """Get the listed multiplier of one value of a feature."""
    return _multipliers(attributes)[value]


def _total_weight(df: pd.DataFrame, attributes: Mapping[str, Any], feature: str = "a") -> float:
    """Compute the total case weight of a feature under its value weight multipliers."""
    multipliers = _multipliers(attributes)
    return sum(count * multipliers.get(value, 1.0) for value, count in df[feature].value_counts().items())


@pytest.mark.parametrize("max_workers", [0, 2])
@pytest.mark.parametrize("caps", [None, ["a"], {"a": 0.2}])
@pytest.mark.parametrize("max_distilled_cases", [100, 300, 1_000])
def test_preserve_rare_values_reweighted(max_workers: int, caps: list[str] | dict[str, float] | None,
                                         max_distilled_cases: int) -> None:
    """Test that rare values are lifted to the threshold, largest-first within what the common value can give."""
    df = _rare_values_df()
    target, _ = get_optimized_partition_size(row_count=len(df), max_partition_size=max_distilled_cases)
    floor = 30 * len(df) / target
    # Uncapped, the common value can go down to the floor; capped, it keeps the capped share of its weight
    cap = None if caps is None else (0.5 if isinstance(caps, list) else caps["a"])
    budget = 98_980 - max(floor, (1 - cap) * 98_980 if cap is not None else 0)
    expected_kept = min(5, int(budget // (floor - 200)))
    if expected_kept < 5:
        context = pytest.warns(UserWarning, match=f"Preserved {expected_kept} of the 5 rare values of feature `a`")
    else:
        context = warnings.catch_warnings()
    with context:
        if expected_kept == 5:
            warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=max_distilled_cases, significance_threshold=30,
                                            preserve_rare_values=["a"], preserve_rare_values_caps=caps,
                                            max_workers=max_workers)
    if expected_kept == 0:
        assert "value_weight_multipliers" not in features["a"]
        return
    multipliers = _multipliers(features["a"])
    kept = {value for value, multiplier in multipliers.items() if multiplier > 1}
    assert len(kept) == expected_kept
    assert kept <= {f"rare{i}" for i in range(5)}
    # Only the kept rare values and the common value that funds them are listed; the small values and
    # any rare value left at a multiplier of 1 are not
    assert set(multipliers) == kept | {"common"}
    for value in kept:
        # Each kept rare value keeps exactly the threshold after distillation
        assert multipliers[value] == pytest.approx(floor / 200)
    used = expected_kept * (floor - 200)
    assert multipliers["common"] == pytest.approx((98_980 - used) / 98_980)
    if cap is not None:
        assert multipliers["common"] >= 1 - cap
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))


def test_preserve_rare_values_caps_validation():
    """Test the checks on `preserve_rare_values_caps`."""
    df = _rare_values_df()
    with pytest.raises(ValueError,
                       match="`preserve_rare_values_caps` names features that are not in the data: `missing`"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values_caps=["missing"])
    with pytest.raises(ValueError, match="must be greater than 0 and at most 1"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values_caps={"a": 0})
    with pytest.raises(ValueError, match="must be greater than 0 and at most 1"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values_caps={"a": 1.5})
    with pytest.raises(TypeError, match="list of feature names or a mapping"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values_caps="a")


def _two_rare_features_df() -> pd.DataFrame:
    """Rare values in features `a`, `b` and the non-nominal `n`, with a column named `off`."""
    df = _rare_values_df()
    df["b"] = ["common"] * (len(df) - 200) + ["rare"] * 200
    df["off"] = df["b"]
    df["n"] = np.arange(len(df)) % 7 * 0.5
    return df


@pytest.mark.parametrize("max_workers", [0, 2])
def test_preserve_rare_values_feature_name(max_workers: int) -> None:
    """Test that naming a feature preserves all of its rare value candidates and no other feature's."""
    df = _two_rare_features_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values=["b"],
                                            max_workers=max_workers, types={"n": "continuous"})
    assert "value_weight_multipliers" not in features["a"]
    assert "value_weight_multipliers" not in features["off"]
    assert set(_multipliers(features["b"])) == {"rare", "common"}
    assert _multiplier(features["b"], "rare") > 1 > _multiplier(features["b"], "common")


def test_preserve_rare_values_feature_names():
    """Test that listing several features preserves the candidates of each, and no other feature's."""
    df = _two_rare_features_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values=["a", "b"],
                                            types={"n": "continuous"})
    assert {f for f, attrs in features.items() if "value_weight_multipliers" in attrs} == {"a", "b"}


def test_preserve_rare_values_feature_name_errors():
    """Test the errors for a `preserve_rare_values` naming an unusable feature, or given as a bare string."""
    df = _two_rare_features_df()
    with pytest.raises(ValueError, match="not in the data"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values=["missing"])
    with pytest.raises(ValueError, match="inferred to be continuous; rare values can only be set for nominal"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values=["n"],
                                 types={"n": "continuous"})
    with pytest.raises(ValueError, match="must also provide `max_distilled_cases`"):
        infer_feature_attributes(df, preserve_rare_values=["a"])
    # A feature name is passed in a list, never as a bare string, so a string is only ever "all" or "off"
    with pytest.raises(ValueError, match='got the string "a"'):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values="a")
    with pytest.raises(TypeError, match="got int"):
        infer_feature_attributes(df, max_distilled_cases=1_000,
                                 preserve_rare_values=3)  # pyright: ignore[reportArgumentType]


def test_preserve_rare_values_off():
    """Test that "off" disables rare value preservation, including its suggestion."""
    df = _two_rare_features_df()
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values="off")
    assert not any("value_weight_multipliers" in attrs for attrs in features.values())
    assert "preserve_rare_values" not in features.suggestions.suggestions


def test_preserve_rare_values_all():
    """Test that "all" configures every feature with rare value candidates, without warnings."""
    df = _two_rare_features_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values="all",
                                            types={"n": "continuous"})
    assert {f for f, attrs in features.items() if "value_weight_multipliers" in attrs} == {"a", "b", "off"}
    assert 'preserve_rare_values="all"' in repr(infer_feature_attributes(
        df, max_distilled_cases=1_000, types={"n": "continuous"}).suggestions.preserve_rare_values)


def test_preserve_rare_values_suggestion_config_format():
    """Test that the values map names only the rare values, while the config also lists the small values at 1."""
    df = _rare_values_df()
    features = infer_feature_attributes(df, max_distilled_cases=1_000)
    suggestion = features.suggestions.preserve_rare_values
    assert suggestion.get_values_map() == {"a": [f"rare{i}" for i in range(5)]}
    assert suggestion.details["num_values"] == 5
    assert suggestion.details["num_preserved"] == 5
    assert suggestion.details["limits"] == []
    assert suggestion.caveats == []
    config = suggestion.get_config()
    multipliers = {cfg["value"]: cfg["multiplier"] for cfg in config["a"]["value_weight_multipliers"]}
    assert set(multipliers) == {f"rare{i}" for i in range(5)} | {"common"}
    assert 0 < multipliers["common"] < 1


def test_preserve_rare_values_suggestion_reports_limit(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that a suggestion whose rare values do not all fit reports the limit, and still applies."""
    df = _rare_values_df()
    # Uncapped, the common value can fund two of the five rare values at this target
    features = infer_feature_attributes(df, max_distilled_cases=100, significance_threshold=30)
    capsys.readouterr()
    suggestion = features.suggestions.preserve_rare_values
    assert suggestion.details["num_values"] == 5
    assert suggestion.details["num_preserved"] == 2
    (limit,) = suggestion.details["limits"]
    assert limit["feature"] == "a"
    assert (limit["preserved"], limit["candidates"]) == (2, 5)
    assert 100 < limit["min_max_distilled_cases"] < len(df)
    assert [c["code"] for c in suggestion.caveats] == ["partial_rare_value_preservation"]
    assert "Preserved 2 of the 5 rare values" in suggestion.caveats[0]["message"]
    # The headline counts what applying writes, and the rest separately
    assert suggestion.summary == ("Found 2 rare values across 1 column that can be preserved during data "
                                  "distillation, and 3 more across 1 column that cannot be at this "
                                  "`max_distilled_cases`")
    description = " ".join(repr(suggestion).split())
    assert "we identified 5 values across 1 column" in description
    assert "of which 2 values across 1 column can be preserved" in description
    assert suggestion.get_values_map() == {"a": ["rare0", "rare1"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features.apply_suggestion("preserve_rare_values")
    assert len({v for v, m in _multipliers(features["a"]).items() if m > 1}) == 2
    # The reported target does fit every rare value
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        refit = infer_feature_attributes(df, max_distilled_cases=limit["min_max_distilled_cases"],
                                         significance_threshold=30, preserve_rare_values=["a"])
    assert len({v for v, m in _multipliers(refit["a"]).items() if m > 1}) == 5


def test_preserve_rare_values_given_multipliers():
    """Test that user-provided multipliers are funded by the other values, scaled down if they do not fit."""
    df = _rare_values_df()
    # These fit: 5 values gaining 9 * 200 cases each is well within half of the common value's weight
    fitting = [{"value": f"rare{i}", "multiplier": 10.0} for i in range(5)]
    # Without `max_distilled_cases`, the floor comes from the default target, and the user is told so
    with pytest.warns(UserWarning, match="assumed distillation target of 50,000 cases"):
        features = infer_feature_attributes(df, preserve_rare_values={"a": fitting})
    multipliers = _multipliers(features["a"])
    assert all(multipliers[f"rare{i}"] == 10.0 for i in range(5))
    # The small values keep a multiplier of 1 and are not listed
    assert set(multipliers) == {f"rare{i}" for i in range(5)} | {"common"}
    assert multipliers["common"] == pytest.approx((98_980 - 5 * 9 * 200) / 98_980)
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))

    # These do not: without `max_distilled_cases` the floor comes from the default target of 50,000,
    # so the common value keeps 30 * 100,000 / 50,000 = 60 cases and the increases are scaled down
    # together to use the rest of its weight
    too_high = [{"value": f"rare{i}", "multiplier": 200.0} for i in range(5)]
    with pytest.warns(UserWarning, match="each multiplier's increase over 1 was scaled by"):
        features = infer_feature_attributes(df, preserve_rare_values={"a": too_high})
    multipliers = _multipliers(features["a"])
    scale = (98_980 - 60) / (5 * 199 * 200)
    assert all(multipliers[f"rare{i}"] == pytest.approx(1 + 199 * scale) for i in range(5))
    assert multipliers["common"] == pytest.approx(60 / 98_980)
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))
    # A cap limits what the common value gives up
    with pytest.warns(UserWarning, match="scaled by"):
        features = infer_feature_attributes(df, preserve_rare_values={"a": too_high},
                                            preserve_rare_values_caps={"a": 0.8})
    assert _multiplier(features["a"], "common") == pytest.approx(0.2)

    # With `max_distilled_cases`, nothing is assumed and nothing is reported
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=50_000,
                                            preserve_rare_values={"a": fitting})
    assert _multipliers(features["a"])["rare0"] == 10.0

    # A full config is used as-is, without computing anything, so nothing is assumed either
    full = {"value_weight_multipliers": [*too_high, {"value": "common", "multiplier": 0.25}]}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, preserve_rare_values={"a": full})
    assert _prv(features["a"]) == full["value_weight_multipliers"]
    # A dict is only accepted in that form
    with pytest.raises(ValueError, match="got a dict with the keys \\['protected_values_multipliers'\\]"):
        infer_feature_attributes(df, preserve_rare_values={"a": {"protected_values_multipliers": fitting}})

    # Multipliers must be at least 1, and values must exist
    with pytest.raises(ValueError, match="must be a finite number of at least 1"):
        infer_feature_attributes(df, preserve_rare_values={"a": [{"value": "rare0", "multiplier": 0.5}]})
    with pytest.raises(ValueError, match="not found in column"):
        infer_feature_attributes(df, preserve_rare_values={"a": [{"value": "missing", "multiplier": 2}]})


def test_preserve_rare_values_deficit_exactly_exhausts_budget():
    """Test that a rare value whose deficit equals the available weight exactly is preserved."""
    # At 252 -> 63 the floor is 120: the common value can give up 146 - 120 = 26 cases and the rare value
    # needs 120 - 94 = 26, computed as 94 * (120 / 94 - 1), which floating point puts a few ulps above 26
    df = pd.DataFrame({"a": ["common"] * 146 + ["rare"] * 94 + ["small"] * 12})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=63, significance_threshold=30,
                                            preserve_rare_values={"a": ["common", "rare"]},
                                            enable_suggestions=False)
    multipliers = _multipliers(features["a"])
    assert set(multipliers) == {"common", "rare"}
    assert multipliers["rare"] == pytest.approx(120 / 94)
    assert multipliers["common"] == pytest.approx(120 / 146)
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))


def test_preserve_rare_values_deficit_just_exceeds_budget():
    """Test that a deficit a hair over the available weight, within or beyond the slack, conserves total weight."""
    # Same counts as above: the common value can give up 26 cases at a floor of 120. A given multiplier
    # asks for 26 cases plus a fraction of the slack (1e-9 * 120), or plus several times it
    df = pd.DataFrame({"a": ["common"] * 146 + ["rare"] * 94 + ["small"] * 12})
    for excess, admitted in ((0.5e-9 * 120, True), (5e-9 * 120, False)):
        multiplier = 1 + (26 + excess) / 94
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            features = infer_feature_attributes(df, max_distilled_cases=63, significance_threshold=30,
                                                preserve_rare_values={"a": [{"value": "rare",
                                                                             "multiplier": multiplier}]},
                                                enable_suggestions=False)
        multipliers = _multipliers(features["a"])
        # Within the slack the target is admitted silently; beyond it, it is scaled with a warning. Either
        # way the rare value is funded from the 26 cases alone, and the total weight is exactly conserved
        assert any("scaled by" in str(w.message) for w in record) is not admitted
        assert multipliers["rare"] == pytest.approx(120 / 94, rel=1e-9)
        assert multipliers["common"] == pytest.approx(120 / 146, rel=1e-9)
        assert _total_weight(df, features["a"]) == pytest.approx(len(df), abs=1e-9)


def test_preserve_rare_values_significant_values_exactly_at_floor():
    """Test the multipliers when funding the targets puts every significant value exactly at the floor."""
    # With a target equal to the data size, the floor is the threshold itself: 29 cases. v1 and v2
    # can give up 71 each, and `rare` asks for 5 * 29 = 145, so its multiplier is scaled to fit and
    # both v1 and v2 end with exactly 29 cases, a factor of 0.29 that floating point cannot represent
    df = pd.DataFrame({"a": ["v1"] * 100 + ["v2"] * 100 + ["rare"] * 5})
    with pytest.warns(UserWarning, match="scaled by 0.979"):
        features = infer_feature_attributes(df, max_distilled_cases=205, significance_threshold=29,
                                            preserve_rare_values={
                                                "a": [{"value": "rare", "multiplier": 30.0}]},
                                            enable_suggestions=False)
    multipliers = _multipliers(features["a"])
    assert multipliers["rare"] == pytest.approx(1 + 29 * 142 / 145)
    assert multipliers["v1"] == multipliers["v2"] == pytest.approx(0.29)
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))

    # Significant values of different sizes: the smaller one is held at the floor, the larger one is
    # scaled, and together they give up exactly what the target needs
    df = pd.DataFrame({"a": ["v1"] * 100 + ["v2"] * 50 + ["rare"] * 5})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=155, significance_threshold=29,
                                            preserve_rare_values={
                                                "a": [{"value": "rare", "multiplier": 19.0}]},
                                            enable_suggestions=False)
    multipliers = _multipliers(features["a"])
    # v1 and v2 must end with 150 - 5 * 18 = 60 cases: v2 at the floor of 29, v1 with the other 31
    assert multipliers["v2"] == pytest.approx(29 / 50)
    assert multipliers["v1"] == pytest.approx(0.31)
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))


@pytest.mark.parametrize("max_workers", [0, 2])
def test_preserve_rare_values_unknown_feature(max_workers: int) -> None:
    """Test that a misspelled feature in `preserve_rare_values` raises before anything is processed."""
    df = _rare_values_df()
    with pytest.raises(ValueError, match="`preserve_rare_values` names features that are not in the data: `typo`"):
        infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values={"typo": ["rare0"]},
                                 max_workers=max_workers)
    with pytest.raises(ValueError, match="`preserve_rare_values` names features that are not in the data: `typo`"):
        infer_feature_attributes(df, preserve_rare_values={"typo": [{"value": "rare0", "multiplier": 2}]},
                                 max_workers=max_workers)


@pytest.mark.parametrize("max_workers", [0, 2])
def test_preserve_rare_values_mixed_forms(max_workers: int) -> None:
    """Test that one mapping can specify each feature in a different form, and that the forms are checked."""
    df = _two_rare_features_df()
    df["c"] = df["b"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=1_000, max_workers=max_workers,
                                            preserve_rare_values={
                                                "a": ["rare0", "rare1"],
                                                "b": [{"value": "rare", "multiplier": 3.0}],
                                                "c": {"value_weight_multipliers": [
                                                    {"value": "rare", "multiplier": 2.5}]},
                                            })
    # `a`: multipliers computed for the named values; `b`: the given multiplier, funded; `c`: as given
    assert {v for v, m in _multipliers(features["a"]).items() if m > 1} == {"rare0", "rare1"}
    assert _multiplier(features["b"], "rare") == 3.0
    assert _multiplier(features["b"], "common") < 1
    assert _prv(features["c"]) == [{"value": "rare", "multiplier": 2.5}]
    assert "value_weight_multipliers" not in features["off"]

    # Each check happens once, before any feature is processed
    with pytest.raises(ValueError, match="mixes plain values with dicts"):
        infer_feature_attributes(df, max_distilled_cases=1_000, max_workers=max_workers,
                                 preserve_rare_values={"a": ["rare0", {"value": "rare1", "multiplier": 2}]})
    with pytest.raises(ValueError, match="pairs value `rare0` with no multiplier"):
        infer_feature_attributes(df, max_distilled_cases=1_000, max_workers=max_workers,
                                 preserve_rare_values={"a": [{"value": "rare0"}]})
    with pytest.raises(TypeError, match="got str"):
        infer_feature_attributes(df, max_distilled_cases=1_000, max_workers=max_workers,
                                 preserve_rare_values={"a": "rare0"})  # pyright: ignore[reportArgumentType]


def test_preserve_rare_values_nullable_numeric():
    """Test that a null can be protected in a numeric feature, through a map and through a config."""
    values = [1.0] * 9_800 + [2.0] * 100 + [None] * 50 + [np.nan] * 50
    df = pd.DataFrame({"x": values, "i": range(len(values))})
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values={"x": [None, 2.0]},
                                        types={"x": "nominal"})
    multipliers = _multipliers(features["x"])
    assert set(multipliers) == {None, 2.0, 1.0}
    target, _ = get_optimized_partition_size(row_count=len(df), max_partition_size=1_000)
    assert multipliers[None] == pytest.approx(30 * len(df) / target / 100)
    with pytest.warns(UserWarning, match="assumed distillation target"):
        features = infer_feature_attributes(df, preserve_rare_values={"x": [{"value": None, "multiplier": 2}]},
                                            types={"x": "nominal"})
    multipliers = _multipliers(features["x"])
    assert multipliers[None] == 2.0
    # The value 2.0 keeps the threshold on its own at the default target, so it funds the null alongside 1.0
    assert set(multipliers) == {None, 1.0, 2.0}
    assert multipliers[1.0] == multipliers[2.0] < 1


def test_preserve_rare_values_keeps_integer_values():
    """Test that protected values of an integer nominal feature are stored as the integers the user gave."""
    values = [1] * 9_900 + [7] * 100
    df = pd.DataFrame({"k": values, "i": range(len(values))})
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values={"k": [7]},
                                        types={"k": "nominal"})
    assert set(_multipliers(features["k"])) == {7, 1}
    assert all(type(entry["value"]) is int for entry in _prv(features["k"]))
    # Candidates found in the data come back as plain Python integers too
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values=["k"],
                                        types={"k": "nominal"})
    assert set(_multipliers(features["k"])) == {7, 1}
    assert all(type(entry["value"]) is int for entry in _prv(features["k"]))


def test_preserve_rare_values_mixed_nulls_are_one_value():
    """Test that None, NaN and NA in one feature are preserved as a single null value with conserved weight."""
    values = ["common"] * 9_880 + [None] * 40 + [np.nan] * 40 + [pd.NA] * 40
    df = pd.DataFrame({"a": pd.Series(values, dtype=object), "i": range(len(values))})
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values="all")
    multipliers = _multipliers(features["a"])
    assert set(multipliers) == {None, "common"}
    target, _ = get_optimized_partition_size(row_count=len(df), max_partition_size=1_000)
    assert multipliers[None] == pytest.approx(30 * len(df) / target / 120)
    null_count = int(df["a"].isna().sum())
    total = null_count * multipliers[None] + (len(df) - null_count) * multipliers["common"]
    assert total == pytest.approx(len(df))


def test_preserve_rare_values_limit_target_counts_future_donors():
    """Test that the reported target for fitting every rare value counts values that can donate only there."""
    # 250 common, 100 intermediate and seven rare values of 30: at 560 -> 70 the floor is 240, so only the
    # common value can donate, and its 10 spare cases fund none of the 210-case deficits. At 280 the floor is
    # 60, the intermediate value becomes a donor (190 + 40 = 230 against 7 * 30 = 210), and every rare value fits.
    values = ["common"] * 250 + ["mid"] * 100 + [f"rare{i}" for i in range(7) for _ in range(30)]
    df = pd.DataFrame({"a": values})
    rare_map = {"a": [f"rare{i}" for i in range(7)]}
    with pytest.warns(UserWarning, match="Preserved 0 of the 7 rare values of feature `a`.*at least 280,"):
        features = infer_feature_attributes(df, max_distilled_cases=100, significance_threshold=30,
                                            preserve_rare_values=rare_map)
    assert "value_weight_multipliers" not in features["a"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        refit = infer_feature_attributes(df, max_distilled_cases=280, significance_threshold=30,
                                         preserve_rare_values=rare_map)
    assert len({v for v, m in _multipliers(refit["a"]).items() if m > 1}) == 7


def test_preserve_rare_values_no_donor_is_reported():
    """Test that a request nothing can fund is reported, for a map and for a simple config."""
    # 400 common and ten rare values of 60: at 1,000 -> 63 the floor is 476, above the common value's
    # count, so no value can donate; at 500 the floor is 60 and the rare values need nothing
    values = ["common"] * 400 + [f"rare{i}" for i in range(10) for _ in range(60)]
    df = pd.DataFrame({"a": values})
    rare_values = [f"rare{i}" for i in range(10)]
    with pytest.warns(UserWarning, match="Preserved 0 of the 10 rare values of feature `a`.*at least 500,"):
        features = infer_feature_attributes(df, max_distilled_cases=100, significance_threshold=30,
                                            preserve_rare_values={"a": rare_values})
    assert "value_weight_multipliers" not in features["a"]
    config = {"a": [{"value": value, "multiplier": 2.0} for value in rare_values]}
    with pytest.warns(UserWarning, match="Preserved 0 of the 10 rare values of feature `a`"):
        features = infer_feature_attributes(df, max_distilled_cases=100, significance_threshold=30,
                                            preserve_rare_values=config)
    assert "value_weight_multipliers" not in features["a"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        refit = infer_feature_attributes(df, max_distilled_cases=500, significance_threshold=30,
                                         preserve_rare_values={"a": rare_values})
    assert "value_weight_multipliers" not in refit["a"]


def test_preserve_rare_values_suggestion_with_nothing_funded(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the automatic suggestion still reports candidates none of which can be funded."""
    # As above, no value can donate at 1,000 -> 63; the common value is itself a candidate there
    values = ["common"] * 400 + [f"rare{i}" for i in range(10) for _ in range(60)]
    df = pd.DataFrame({"a": values})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=100, significance_threshold=30)
    capsys.readouterr()
    suggestion = features.suggestions.preserve_rare_values
    assert suggestion.can_apply is False
    assert suggestion.details["num_preserved"] == 0
    assert suggestion.details["num_values"] == suggestion.details["limits"][0]["candidates"] == 11
    assert suggestion.details["num_candidate_features"] == 1
    assert suggestion.details["num_features"] == 0
    assert suggestion.summary == ("Found 11 rare values across 1 column whose signal may be lost during data "
                                  "distillation, none of which can be preserved at this `max_distilled_cases`")
    assert suggestion.details["value_weight_multipliers"] == {}
    (caveat,) = suggestion.caveats
    assert caveat["code"] == "partial_rare_value_preservation"
    assert "Preserved 0 of the 11 rare values of feature `a`" in caveat["message"]
    assert "at least 500" in caveat["message"]
    assert suggestion.get_values_map() == {}
    with pytest.warns(UserWarning, match="^This suggestion was not applied: none of the rare values found"):
        features.apply_suggestion("preserve_rare_values")
    assert "value_weight_multipliers" not in features["a"]


def test_preserve_rare_values_duplicate_values():
    """Test that a value listed twice is weighted once, in both the plain and the paired form."""
    df = pd.DataFrame({"a": ["common"] * 900 + ["rare"] * 60 + ["small"] * 10})
    for spec in (["rare", "rare"], [{"value": "rare", "multiplier": 3.0}, {"value": "rare", "multiplier": 3.0}]):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            features = infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                                preserve_rare_values={"a": spec}, enable_suggestions=False)
        entries = _prv(features["a"])
        assert [entry["value"] for entry in entries] == ["rare", "common"]
        assert _total_weight(df, features["a"]) == pytest.approx(len(df))
    # Null forms are one value too
    nullable = pd.DataFrame({"a": ["common"] * 900 + [None] * 60 + [np.nan] * 30 + ["small"] * 10})
    features = infer_feature_attributes(nullable, max_distilled_cases=200, significance_threshold=30,
                                        preserve_rare_values={"a": [None, np.nan]}, enable_suggestions=False)
    assert [entry["value"] for entry in _prv(features["a"])] == [None, "common"]
    # The same value with two different multipliers is a contradiction
    with pytest.raises(ValueError, match="gives value `rare` two different multipliers, 3 and 4"):
        infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                 preserve_rare_values={"a": [{"value": "rare", "multiplier": 3.0},
                                                             {"value": "rare", "multiplier": 4.0}]})


def test_preserve_rare_values_multiplier_of_one_is_omitted():
    """Test that a value paired with a multiplier of 1 is left out of the output, like any unchanged value."""
    df = pd.DataFrame({"a": ["common"] * 900 + ["rare"] * 60 + ["small"] * 10})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                            preserve_rare_values={"a": [{"value": "rare", "multiplier": 2},
                                                                        {"value": "small", "multiplier": 1}]},
                                            enable_suggestions=False)
    assert set(_multipliers(features["a"])) == {"rare", "common"}
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))
    # A value pinned at 1 is a target, not a donor: `fixed` has enough cases to fund `rare` but is left alone
    pinned = pd.DataFrame({"a": ["common"] * 900 + ["rare"] * 60 + ["fixed"] * 300})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(pinned, max_distilled_cases=400, significance_threshold=30,
                                            preserve_rare_values={"a": [{"value": "rare", "multiplier": 2},
                                                                        {"value": "fixed", "multiplier": 1}]},
                                            enable_suggestions=False)
    multipliers = _multipliers(features["a"])
    assert set(multipliers) == {"rare", "common"}
    assert multipliers["rare"] == 2.0
    assert multipliers["common"] == pytest.approx((900 - 60) / 900)
    assert _total_weight(pinned, features["a"]) == pytest.approx(len(pinned))
    # A pinned value is not a rare value to fund: when the pin is the only value that could have donated,
    # the report counts and the suggested target consider the lifted value alone
    pinned_donor = pd.DataFrame({"a": ["fixed"] * 900 + ["rare"] * 60})
    with pytest.warns(UserWarning, match="Preserved 0 of the 1 rare values of feature `a`.*at least 480,"):
        features = infer_feature_attributes(pinned_donor, max_distilled_cases=400, significance_threshold=30,
                                            preserve_rare_values={"a": [{"value": "rare", "multiplier": 2},
                                                                        {"value": "fixed", "multiplier": 1}]},
                                            enable_suggestions=False)
    assert "value_weight_multipliers" not in features["a"]
    # A pinned value is validated like any other
    with pytest.raises(ValueError, match="not found in column"):
        infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                 preserve_rare_values={"a": [{"value": "typo", "multiplier": 1}]})
    # Asking for no change at all writes nothing and says nothing, with or without a target, even when
    # the feature has no value that could donate
    no_donor = pd.DataFrame({"a": ["x"] * 50 + ["y"] * 50})
    for frame in (df, no_donor):
        for kwargs in ({"max_distilled_cases": 200}, {}):
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                features = infer_feature_attributes(frame, significance_threshold=30, enable_suggestions=False,
                                                    preserve_rare_values={"a": [{"value": "x" if frame is no_donor
                                                                                 else "rare", "multiplier": 1}]},
                                                    **kwargs)
            assert "value_weight_multipliers" not in features["a"]
    # A multiplier of 1 still counts as a conflicting duplicate of another multiplier for the same value
    with pytest.raises(ValueError, match="two different multipliers, 1 and 2"):
        infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                 preserve_rare_values={"a": [{"value": "rare", "multiplier": 1},
                                                             {"value": "rare", "multiplier": 2}]})


@pytest.mark.parametrize("max_workers", [0, 2])
def test_preserve_rare_values_huge_multiplier_conserves_weight(max_workers: int) -> None:
    """Test that a multiplier near the float limit is scaled down to the budget without overflowing."""
    # 900 common and 100 rare cases at 1,000 -> 63: the floor is 476.19, so the common value can give up
    # 423.81 cases, and that is what the rare value receives whatever multiplier was asked for
    df = pd.DataFrame({"a": ["common"] * 900 + ["rare"] * 100, "i": range(1_000)})
    target, _ = get_optimized_partition_size(row_count=len(df), max_partition_size=100)
    floor = 30 * len(df) / target
    with pytest.warns(UserWarning, match="scaled by"):
        features = infer_feature_attributes(df, max_distilled_cases=100, significance_threshold=30,
                                            preserve_rare_values={"a": [{"value": "rare", "multiplier": 1e308}]},
                                            max_workers=max_workers)
    multipliers = _multipliers(features["a"])
    assert multipliers["rare"] == pytest.approx(1 + (900 - floor) / 100)
    assert multipliers["common"] == pytest.approx(floor / 900)
    assert _total_weight(df, features["a"]) == pytest.approx(len(df))


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf"), -float("inf"), "2", None])
def test_preserve_rare_values_multiplier_must_be_finite(multiplier) -> None:
    """Test that a given multiplier must be a finite number, both when funded and in a complete configuration."""
    df = pd.DataFrame({"a": ["common"] * 900 + ["rare"] * 60 + ["small"] * 10})
    with pytest.raises(ValueError, match="must be a finite number of at least 1"):
        infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                 preserve_rare_values={"a": [{"value": "rare", "multiplier": multiplier}]})
    with pytest.raises(ValueError, match="must be a finite number of at least 0"):
        infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                 preserve_rare_values={"a": {"value_weight_multipliers": [
                                     {"value": "rare", "multiplier": multiplier}]}})
    with pytest.raises(ValueError, match="must be positive; got 0"):
        infer_feature_attributes(df, max_distilled_cases=200, significance_threshold=30,
                                 preserve_rare_values={"a": {"value_weight_multipliers": [
                                     {"value": "rare", "multiplier": 0}]}})


def test_preserve_rare_values_all_cases_protected():
    """Test that a feature is skipped, with a warning, when every one of its cases holds a protected value."""
    df = pd.DataFrame({"a": ["x"] * 50 + ["y"] * 50, "i": range(100)})
    with pytest.warns(UserWarning, match="Preserved 0 of the 2 rare values of feature `a`"):
        features = infer_feature_attributes(df, max_distilled_cases=10, preserve_rare_values={"a": ["x", "y"]})
    assert "value_weight_multipliers" not in features["a"]
    all_values = [{"value": "x", "multiplier": 2}, {"value": "y", "multiplier": 2}]
    with pytest.warns(UserWarning, match="Preserved 0 of the 2 rare values of feature `a`"):
        features = infer_feature_attributes(df, preserve_rare_values={"a": all_values})
    assert "value_weight_multipliers" not in features["a"]


def test_preserve_rare_values_null_values():
    """Test that all null values count as one value, listed once as None when it is protected and never otherwise."""
    values = ["common"] * 9_880 + ["rare"] * 100 + [None] * 10 + [np.nan] * 10
    df = pd.DataFrame({"a": values, "i": range(len(values))})
    # Too few nulls to be significant: they keep a multiplier of 1 and are not listed
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values={"a": ["rare"]})
    multipliers = _multipliers(features["a"])
    assert set(multipliers) == {"rare", "common"}
    # Protected: listed once, as None, with the count of every null form; `rare` sits below the floor
    # without being named, so it keeps its weight rather than being lifted, and is not listed
    features = infer_feature_attributes(df, max_distilled_cases=1_000, preserve_rare_values={"a": [None]})
    multipliers = _multipliers(features["a"])
    assert set(multipliers) == {None, "common"}
    target, _ = get_optimized_partition_size(row_count=len(df), max_partition_size=1_000)
    assert multipliers[None] == pytest.approx(30 * len(df) / target / 20)


def test_infer_fanout_features(capsys):
    """Test that `infer_feature_attributes` correctly infers and issues suggestions about fan-out features."""
    # Test that a suggestion is issued, and summarized on the console rather than as a warning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        features = infer_feature_attributes(joined_olist_df, default_time_zone="UTC")
    # Suggestions are summarized on the console, not shouted about as a warning.
    assert not [w for w in record if "suggestions" in str(w.message).lower()]
    summary = capsys.readouterr().out
    assert "Feature Attributes Summary" in summary
    assert "fan-out feature" in summary
    for feat in features:
        assert "fanout_on" not in feat
    # Test a suggestion application
    features.apply_suggestion("fanout_features")
    assert "customer_id" in features["customer_city"].get("fanout_on", [])
    assert "product_id" in features["product_height_cm"].get("fanout_on", [])

    fof_map = features.suggestions.fanout_features.get_fanout_feature_map()
    suggestions_json = features.suggestions.to_json()

    # Supplying the suggested fanout_feature_map to IFA
    features = infer_feature_attributes(joined_olist_df, fanout_feature_map=fof_map, max_workers=2, default_time_zone="UTC")
    assert "customer_id" in features["customer_state"].get("fanout_on", [])
    assert "product_id" in features["product_length_cm"].get("fanout_on", [])

    # The JSON form of the suggested map is accepted by IFA and yields the same configuration
    fanout = next(sug for sug in json.loads(suggestions_json)["suggestions"] if sug["name"] == "fanout_features")
    list_form = fanout["parameters"]["fanout_feature_map"]
    list_features = infer_feature_attributes(joined_olist_df, fanout_feature_map=list_form, max_workers=2,
                                             default_time_zone="UTC")
    for feature, attributes in features.items():
        assert list_features[feature].get("fanout_on") == attributes.get("fanout_on")


def test_infer_fanout_features_ignores_constant_columns(capsys):
    """Globally-constant columns must not produce fan-out suggestions or trip the strict-tree warning.

    A constant column is functionally determined by every key, so prior to the
    fix it attached to every key's fan-out set -- manufacturing sibling levels
    that violate the strict-subset-chain assumption (emitting a misleading
    "strict-tree" warning) and listing the constant itself as a fan-out feature.
    Once the constant column is correctly excluded, this data has no
    fan-out structure, so no fan-out suggestion (and thus no printed summary)
    should be produced.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        features = infer_feature_attributes(fanout_constant_df)
    summary = capsys.readouterr().out

    strict_tree = [w for w in record if "strict" in str(w.message).lower()]
    assert not strict_tree, (
        f"unexpected strict-tree warning(s): {[str(w.message) for w in strict_tree]}"
    )

    # The constant column is the only fan-out candidate and is correctly excluded, so
    # no fan-out suggestion should be produced, therefore no summary is printed.
    assert "fanout_features" not in features.suggestions.suggestions
    assert "Feature Attributes Summary" not in summary

    # No feature should be configured as a fan-out feature, least of all the constant column.
    for feat in fanout_constant_df.columns:
        assert "fanout_on" not in features[feat]


def test_enable_suggestions_false(capsys):
    """enable_suggestions=False must skip fanout and PRV inference with no summary and empty suggestions."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        features = infer_feature_attributes(joined_olist_df, enable_suggestions=False, default_time_zone="UTC")

    assert isinstance(features.suggestions, IFASuggestionCollector)
    assert len(features.suggestions.suggestions) == 0
    assert "Feature Attributes Summary" not in capsys.readouterr().out

    # Explicit fanout_feature_map must still be applied even when suggestions are disabled
    fof_map = {"customer_id": ["customer_city", "customer_state"]}
    features = infer_feature_attributes(
        joined_olist_df, enable_suggestions=False, fanout_feature_map=fof_map, default_time_zone="UTC"
    )
    assert "customer_id" in features["customer_city"].get("fanout_on", [])

def test_feature_contains_nulls():
    """Ensure that the `nulls_observed` attribute is correctly set."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df, default_time_zone="UTC", enable_suggestions=False)
    assert not features["class"].get("bounds", {}).get("nulls_observed")

    df.loc[len(df) - 1, "class"] = None
    features = infer_feature_attributes(df, default_time_zone="UTC", enable_suggestions=False)
    assert features["class"].get("bounds", {}).get("nulls_observed")
