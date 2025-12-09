"""Tests infer_feature_attributes with time series data (InferFeatureAttributesTimeSeries)."""
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
import random
import warnings

from howso import client
from howso.engine import Trainee
from howso.utilities.feature_attributes import infer_feature_attributes
from howso.utilities.feature_attributes.base import SingleTableFeatureAttributes
import numpy as np
import pandas as pd
import pytest


root_path = (
    Path(client.__file__).parent.parent
)
data_path = root_path.joinpath("utilities/tests/data")
example_timeseries_path = data_path.joinpath("example_timeseries.csv")

# Partially defined dictionary-1
features_1 = {
    'f1': {
        'bounds': {
            'allow_null': False,
            'max': 5103.08,
            'min': 20.08
        },
        'type': 'continuous'
    },
    'f2': {
        'bounds': {
            'allow_null': False,
            'max': 6103,
            'min': 0
        },
        'time_series': True,
        'type': 'continuous'
    },
}


# Partially defined "ordered" dict
features_2 = OrderedDict(
    (f_name, features_1[f_name]) for f_name in features_1
)


def test_infer_features_attributes_single_ID():
    """Litmus test for infer feature types for iris dataset."""
    df = pd.read_csv(example_timeseries_path)

    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    expected_types = {
        "ID": "nominal",
        "f1": "continuous",
        "f2": "continuous",
        "f3": "continuous",
        "date": "continuous"
    }
    features = infer_feature_attributes(
        df,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        include_sample=True
    )

    for feature, attributes in features.items():
        print(feature)
        assert expected_types[feature] == attributes['type']
        assert 'sample' in attributes and attributes['sample'] is not None


def test_infer_features_attributes_multiple_ID():
    """Litmus test for infer feature types for iris dataset with multiple ID features."""
    df = pd.read_csv(example_timeseries_path)
    df["ID2"] = df["ID"].copy()

    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = ["ID", "ID2"]
    time_feature_name = "date"

    expected_types = {
        "ID": "nominal",
        "ID2": "nominal",
        "f1": "continuous",
        "f2": "continuous",
        "f3": "continuous",
        "date": "continuous"
    }
    features = infer_feature_attributes(
        df,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format}
    )

    for feature, attributes in features.items():
        print(feature)
        assert expected_types[feature] == attributes['type']

    assert features["ID"]["id_feature"] is True
    assert features["ID2"]["id_feature"] is True


def test_set_rate_delta_boundaries():
    """Test infer_feature_attributes for time series with rate/delta boundaries set."""
    df = pd.read_csv(example_timeseries_path)

    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    # Test overwrite order 0
    rate_boundaries = {"f3": {"min": {'0': 1234.5678, '1': 12345}, "max": {'0': 5678.1234}}}
    delta_boundaries = {"date": {"min": {'0': 8765.4321}, "max": {'0': 4321.8765, '1': 54321}}}

    features = infer_feature_attributes(
        df,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        rate_boundaries=rate_boundaries,
        delta_boundaries=delta_boundaries
    )

    # Make sure that order 0 was overwritten for rate/delta min & max
    assert features['f3']['time_series']['rate_min'][0] == rate_boundaries['f3']['min']['0']
    assert features['f3']['time_series']['rate_max'][0] == rate_boundaries['f3']['max']['0']
    assert features['date']['time_series']['delta_min'][0] == delta_boundaries['date']['min']['0']
    assert features['date']['time_series']['delta_max'][0] == delta_boundaries['date']['max']['0']

    # Make sure that order 1 was ignored
    assert len(features['f3']['time_series']['rate_min']) == 1
    assert len(features['f3']['time_series']['rate_max']) == 1
    assert len(features['date']['time_series']['delta_min']) == 1
    assert len(features['date']['time_series']['delta_max']) == 1

    # If a boundary type is mismatched, a warning should be raised and the value should be ignored
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rate_boundaries = {"date": {"min": {'0': 1234.5678, '1': 12345}, "max": {'0': 5678.1234}}}
        delta_boundaries = {"f3": {"min": {'0': 8765.4321}, "max": {'0': 4321.8765, '1': 54321}}}
        features = infer_feature_attributes(
            df,
            time_feature_name=time_feature_name,
            id_feature_name=id_feature_name,
            datetime_feature_formats={time_feature_name: time_format},
            rate_boundaries=rate_boundaries,
            delta_boundaries=delta_boundaries,
        )
        assert 'rate_min' not in features['date']['time_series']
        assert 'rate_max' not in features['date']['time_series']
        assert 'delta_min' not in features['f3']['time_series']
        assert 'delta_max' not in features['f3']['time_series']


@pytest.mark.parametrize(
    ("universal_value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
    ]
)
def test_time_feature_is_universal(universal_value, expected):
    """Validates that time_feature_is_universal is working as expected."""
    df = pd.read_csv(example_timeseries_path)

    # Define time format
    time_format = "%Y%m%d"
    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    features = infer_feature_attributes(
        df,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        time_feature_is_universal=universal_value,
    )

    assert features[time_feature_name]['time_series'].get("universal") == expected


def test_infer_features_attributes_tight_bounds_dependent_functionality():
    """Test tight bounds and dependent features functionality for time series IFA."""
    df = pd.read_csv(example_timeseries_path)

    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    features = infer_feature_attributes(
        df,
        dependent_features={"f1": ["f2"]},
        tight_bounds=["f3"],
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        include_sample=True
    )

    f3_max = df["f3"].max()
    f3_min = df["f3"].min()

    assert features["f1"]["dependent_features"] == ["f2"]
    assert features["f3"]["bounds"]["max"] == f3_max
    assert features["f3"]["bounds"]["min"] == f3_min


def test_invalid_time_feature_format():
    """Validates that an invalid time feature date_time_format raises."""
    df = pd.read_csv(example_timeseries_path)
    time_feature_name = "date"

    with pytest.raises(ValueError, match="does not match the data"):
        with warnings.catch_warnings():
            # warnings are expected
            warnings.filterwarnings("ignore")
            infer_feature_attributes(
                df,
                time_feature_name=time_feature_name,
                id_feature_name="ID",
                datetime_feature_formats={time_feature_name: "%Y-%m-%dT%H"}  # invalid format
            )


@pytest.mark.parametrize('data, types, expected_types, is_valid', [
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5], 'b': [3, 4, 5, 6, 7, 8]}), dict(b='continuous'), dict(b='continuous'),
     True),
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5], 'b': [3, 4, 5, 6, 7, 8]}), dict(a='nominal'), dict(a='continuous'), True),
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7], 'b': [3, 4, 5, 6, 7, 8, 9, 1]}), dict(b='nominal'),
     dict(b='nominal'), True),
    (pd.DataFrame({'a': [True, False, False, True], 'b': [False, True, False, True]}), dict(b='continuous'),
     dict(b='nominal'), True),
])
def test_preset_feature_types(data, types, expected_types, is_valid):
    """Test that infer_feature_attributes correctly presets feature types with the `types` parameter."""
    features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_valid:
            features = infer_feature_attributes(data, types=types, time_feature_name='a')
            for feature_name, expected_type in expected_types.items():
                # Make sure it is the correct type
                assert features[feature_name]['type'] == expected_type
                # All features in this test, including nominals, should have bounds (at the very least: `allow_null`)
                assert 'allow_null' in features[feature_name].get('bounds', {}).keys()
        else:
            with pytest.raises(ValueError):
                infer_feature_attributes(data, types=types)


def test_lags():
    """Validates that `lags` can be set as an argument for time series IFA."""
    df = pd.read_csv(example_timeseries_path)

    # The below should not raise any exceptions
    lags = {"date": 0}
    fa = infer_feature_attributes(df, time_feature_name="date", id_feature_name="ID", lags=lags)
    Trainee(features=fa)

    # Ensure that an invalid lags value raises a helpful TypeError
    with pytest.raises(TypeError):
        fa = infer_feature_attributes(df, time_feature_name="date", id_feature_name="ID", lags={"date": '0'})


def test_nominal_vs_continuous_detection():
    """Test that IFA correctly determines nominal vs. continuous features in time series data."""
    # Time-series data with 4756 cases, 41 series; avg. 116 cases/series; sqrt(116) ~= 10.77
    df = pd.read_csv(example_timeseries_path)
    # Add a new integer column with n_uniques > sqrt(avg_n_cases_per_series) -- should be continuous
    series = np.resize(np.arange(11), len(df))
    np.random.shuffle(series)
    df["f4"] = series
    # Add a new integer column with n_uniques < sqrt(avg_n_cases_per_series) -- should be nominal
    series = np.resize(np.arange(10), len(df))
    np.random.shuffle(series)
    df["f5"] = series
    # Ensure that feature types were properly inferred
    features = infer_feature_attributes(df, id_feature_name="ID", time_feature_name="date")
    assert features["f4"]["type"] == "continuous"
    assert features["f5"]["type"] == "nominal"
    # Ensure that pre-setting feature types overrides this logic
    features = infer_feature_attributes(df, id_feature_name="ID", time_feature_name="date",
                                        types={"f4": "nominal", "f5": "continuous"})
    assert features["f5"]["type"] == "continuous"
    assert features["f4"]["type"] == "nominal"


@pytest.mark.parametrize(
    "data_type, value",
    [
        ("json", ['{"potion": 5, "gauntlet": 2}', '{"staff": 1, "potion": 2}']),
        ("yaml", ["potion: 5\ngauntlet: 2", "staff: 1\npotion: 2"]),
    ],
)
def test_semi_structured_features(data_type: str, value: list[str]):
    """Test that IFA detects semi structured features."""
    df = pd.DataFrame([
        {"class": "Fighter", "turn": 1, "hp": 100, "magic": 0, "inventory": value[0]},
        {"class": "Fighter", "turn": 2, "hp": 70, "magic": 0, "inventory": value[0]},
        {"class": "Mage", "turn": 1, "hp": 100, "magic": 100, "inventory": value[1]},
        {"class": "Mage", "turn": 2, "hp": 100, "magic": 85, "inventory": value[1]},
    ])

    features = infer_feature_attributes(
        df,
        time_feature_name="turn",
        id_feature_name="class",
        num_lags=3
    )
    assert features["inventory"]["type"] == "continuous"
    assert features["inventory"]["data_type"] == data_type
    assert features["inventory"]["original_type"] == {"data_type": "string"}
    assert features["inventory"]["time_series"]["type"] == "rate"
    assert features["inventory"]["time_series"]["num_lags"] == 3


def test_time_series_features_pandas():
    valid = SingleTableFeatureAttributes.from_json(json_path=data_path.joinpath("example_timeseries.features.json"))

    df = pd.read_csv(data_path.joinpath("example_timeseries.csv"))
    features = infer_feature_attributes(
        df,
        id_feature_name = "ID",
        time_feature_name="date",
        datetime_feature_formats={"date": "%Y%m%d"},
    )
    pprint(pprint(features))

    for feature, attrs in features.items():
        if "time_series" in attrs:
            assert "time_series" in valid[feature]
            if valid[feature]["time_series"]["type"] == "rate":
                assert attrs["time_series"]["type"] == "rate"
                assert valid[feature]["time_series"]["rate_min"] == attrs["time_series"]["rate_min"]
                assert valid[feature]["time_series"]["rate_max"] == attrs["time_series"]["rate_max"]
            elif valid[feature]["time_series"]["type"] == "delta":
                assert attrs["time_series"]["type"] == "delta"
                assert valid[feature]["time_series"]["delta_min"] == attrs["time_series"]["delta_min"]
                assert valid[feature]["time_series"]["delta_max"] == attrs["time_series"]["delta_max"]
            else:
                raise ValueError(f"Invalid time-series type: {valid[feature]['time_series']['type']} for {feature=}.")


def test_time_series_features_pandas_native_date():
    valid = SingleTableFeatureAttributes.from_json(json_path=data_path.joinpath("example_timeseries.features.json"))

    df = pd.read_csv(data_path.joinpath("example_timeseries.csv"))
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    features = infer_feature_attributes(
        df,
        id_feature_name = "ID",
        time_feature_name="date",
    )
    pprint(pprint(features))

    for feature, attrs in features.items():
        if "time_series" in attrs:
            assert "time_series" in valid[feature]
            if valid[feature]["time_series"]["type"] == "rate":
                assert attrs["time_series"]["type"] == "rate"
                assert valid[feature]["time_series"]["rate_min"] == attrs["time_series"]["rate_min"]
                assert valid[feature]["time_series"]["rate_max"] == attrs["time_series"]["rate_max"]
            elif valid[feature]["time_series"]["type"] == "delta":
                assert attrs["time_series"]["type"] == "delta"
                assert valid[feature]["time_series"]["delta_min"] == attrs["time_series"]["delta_min"]
                assert valid[feature]["time_series"]["delta_max"] == attrs["time_series"]["delta_max"]
            else:
                raise ValueError(f"Invalid time-series type: {valid[feature]['time_series']['type']} for {feature=}.")

def test_nominals_are_ignored_in_ifa_for_ts():
    """Ensure that nominals are never marked for a TS type in IFA."""
    df = pd.read_csv(data_path.joinpath("mini_stock_data.csv"))

    # Add a nominal column that varies (this dataset's existing nominals are
    # all identical, and one is the ID feature.)
    action_map = {
        "BUY": 0.2,
        "HOLD": 0.6,
        "SELL": 0.2
    }
    df["ACTION"] = random.choices(list(action_map.keys()), weights=list(action_map.values()), k=len(df))

    # Also, add a semi-structured feature such as JSON for good measure. A
    # semi-structured data feature should also be ignored, even though it is
    # continuous.
    config_map = {
        '{"alpha": 0.1}': 0.1,
        '{"beta": 0.2}': 0.2,
        '{"gamma": 0.3}': 0.3,
        '{"delta": 0.4}': 0.4
    }
    df["CONFIG"] = random.choices(list(config_map.keys()), weights=list(config_map.values()), k=len(df))

    features = infer_feature_attributes(
        df,
        id_feature_name = "SERIES",
        time_feature_name="DATE",
        default_time_zone="UTC",
    )

    # Verify that the semi-structured JSON continuous was set to continuous.
    assert features["CONFIG"]["type"] == "continuous" and features["CONFIG"]["data_type"] == "json"

    # Make sure continuous still work as expected...
    for feature in features.get_names(types=["continuous"]):
        # We do not wish to assert that semi-structured string-continuous features have time_series:type set.
        if features[feature]["data_type"] in {"json", "yaml", "amalgam", "string_mixable"}:
            continue
        assert "time_series" in features[feature]
        assert features[feature]["time_series"]["type"] in {"rate", "delta", "invariant"}

    # Are there any bad nominals?
    bad_nominals = [
        feature for feature in features.get_names(types=["nominal"])
        if "time_series" in features[feature]
    ]
    print(", ".join(bad_nominals))
    assert bool(bad_nominals) is False
