"""Tests infer_feature_attributes with time series data and AbstractData classes."""
from collections import OrderedDict
from pathlib import Path
import warnings

import pandas as pd
import pytest

from howso.connectors.abstract_data import convert_data, DataFrameData
from howso.engine import Trainee
from howso.utilities.feature_attributes import infer_feature_attributes

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

cwd = Path(__file__).parent.parent.parent.parent
iris_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'iris.csv'))
int_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'integers.csv'))
stock_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'mini_stock_data.csv'))
ts_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'example_timeseries.csv'))


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
def test_infer_features_attributes_single_ID(adc):
    """Litmus test for infer feature types for iris dataset."""
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
        adc,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        include_sample=True
    )

    for feature, attributes in features.items():
        print(feature)
        assert expected_types[feature] == attributes['type']
        assert 'sample' in attributes and attributes['sample'] is not None


@pytest.mark.parametrize('adc', [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
def test_infer_features_attributes_multiple_ID(adc):
    """Litmus test for infer feature types for iris dataset with multiple ID features."""
    df = ts_df.copy()
    df["ID2"] = df["ID"].copy()
    convert_data(DataFrameData(df), adc)

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
        adc,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format}
    )

    for feature, attributes in features.items():
        print(feature)
        assert expected_types[feature] == attributes['type']

    assert features["ID"]["id_feature"] is True
    assert features["ID2"]["id_feature"] is True


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
@pytest.mark.parametrize(
    "features",
    [features_1, features_2]
)
def test_partially_filled_feature_types(features: dict, adc) -> None:
    """
    Make sure the partially filled feature types remain intact.

    Parameters
    ----------
    df: pandas.DataFrame
    features
    """
    pre_inferred_features = features.copy()

    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    inferred_features = infer_feature_attributes(
        adc,
        time_feature_name=time_feature_name,
        features=features,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format}
    )

    for k, v in pre_inferred_features.items():
        assert v['type'] == inferred_features[k]['type']

        if 'bounds' in v:
            # Make sure the bounds are not altered
            # by `infer_feature_attributes` function
            assert v['bounds'] == inferred_features[k]['bounds']


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
def test_set_rate_delta_boundaries(adc):
    """Test infer_feature_attributes for time series with rate/delta boundaries set."""
    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    # Test overwrite order 0
    rate_boundaries = {"f3": {"min": {'0': 1234.5678, '1': 12345}, "max": {'0': 5678.1234}}}
    delta_boundaries = {"date": {"min": {'0': 8765.4321}, "max": {'0': 4321.8765, '1': 54321}}}

    features = infer_feature_attributes(
        adc,
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
            adc,
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


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
@pytest.mark.parametrize(
    ("universal_value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
    ]
)
def test_time_feature_is_universal(universal_value, expected, adc):
    """Validates that time_feature_is_universal is working as expected."""
    # Define time format
    time_format = "%Y%m%d"
    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    features = infer_feature_attributes(
        adc,
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        time_feature_is_universal=universal_value,
    )

    assert features[time_feature_name]['time_series'].get("universal") == expected


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
def test_infer_features_attributes_tight_bounds_dependent_functionality(adc):
    """Test tight bounds and dependent features functionality for time series IFA."""
    # Define time format
    time_format = "%Y%m%d"

    # Identify id-feature and time-feature
    id_feature_name = "ID"
    time_feature_name = "date"

    features = infer_feature_attributes(
        adc,
        dependent_features={"f1": ["f2"]},
        tight_bounds=["f3"],
        time_feature_name=time_feature_name,
        id_feature_name=id_feature_name,
        datetime_feature_formats={time_feature_name: time_format},
        include_sample=True
    )

    f3_max = ts_df["f3"].max()
    f3_min = ts_df["f3"].min()

    assert features["f1"]["dependent_features"] == ["f2"]
    assert features["f3"]["bounds"]["max"] == f3_max
    assert features["f3"]["bounds"]["min"] == f3_min


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
def test_invalid_time_feature_format(adc):
    """Validates that an invalid time feature date_time_format raises."""
    time_feature_name = "date"

    with pytest.raises(ValueError, match="does not match the data"):
        with warnings.catch_warnings():
            # warnings are expected
            warnings.filterwarnings("ignore")
            infer_feature_attributes(
                adc,
                time_feature_name=time_feature_name,
                id_feature_name="ID",
                datetime_feature_formats={time_feature_name: "%Y-%m-%dT%H"}  # invalid format
            )


@pytest.mark.parametrize('adc', [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
@pytest.mark.parametrize('data, types, expected_types, is_valid', [
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5], 'b': [3, 4, 5, 6, 7, 8]}), dict(b='continuous'),
     dict(b='continuous'), True),
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5], 'b': [3, 4, 5, 6, 7, 8]}), dict(a='nominal'),
     dict(a='continuous'), True),
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7], 'b': [3, 4, 5, 6, 7, 8, 9, 1]}), dict(b='nominal'),
     dict(b='nominal'), True),
    (pd.DataFrame({'a': [True, False, False, True], 'b': [False, True, False, True]}), dict(b='continuous'),
     dict(b='nominal'), True),
])
def test_preset_feature_types(data, types, expected_types, is_valid, adc):
    """Test that infer_feature_attributes correctly presets feature types with the `types` parameter."""
    convert_data(DataFrameData(data), adc)
    features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_valid:
            features = infer_feature_attributes(adc, types=types, time_feature_name='a')
            for feature_name, expected_type in expected_types.items():
                # Make sure it is the correct type
                assert features[feature_name]['type'] == expected_type
                # All features in this test, including nominals, should have bounds (at the very least: `allow_null`)
                assert 'allow_null' in features[feature_name].get('bounds', {}).keys()
        else:
            with pytest.raises(ValueError):
                infer_feature_attributes(adc, types=types)


@pytest.mark.parametrize('adc', [
    ("MongoDBData", ts_df),
    ("SQLTableData", ts_df),
    ("ParquetDataFile", ts_df),
    ("ParquetDataset", ts_df),
    ("TabularFile", ts_df),
    ("DaskDataFrameData", ts_df),
    ("DataFrameData", ts_df),
], indirect=True)
def test_lags(adc):
    """Validates that `lags` can be set as an argument for time series IFA."""
    # The below should not raise any exceptions
    lags = {"date": 0}
    fa = infer_feature_attributes(adc, time_feature_name="date", id_feature_name="ID", lags=lags)
    Trainee(features=fa)

    # Ensure that an invalid lags value raises a helpful TypeError
    with pytest.raises(TypeError):
        fa = infer_feature_attributes(adc, time_feature_name="date", id_feature_name="ID", lags={"date": '0'})
