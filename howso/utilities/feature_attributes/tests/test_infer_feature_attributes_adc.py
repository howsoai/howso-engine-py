"""Tests the `infer_feature_attributes` package with AbstractData classes."""
from pathlib import Path
import warnings

import pandas as pd
import pytest

try:
    from howso.connectors.abstract_data import (
        convert_data,
        DataFrameData,
        make_data_source,
    )
except (ModuleNotFoundError, ImportError):
    pytest.skip('howso-engine-connectors not installed', allow_module_level=True)

from howso.utilities.feature_attributes import infer_feature_attributes
from howso.utilities.feature_attributes.abstract_data import InferFeatureAttributesAbstractData
from howso.utilities.features import FeatureType

cwd = Path(__file__).parent.parent.parent.parent
iris_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'iris.csv'))
int_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'integers.csv'))
nypd_arrest_pq_path = Path(cwd, 'utilities', 'tests', 'data', 'NYPD_arrest_data_25K.parquet')
stock_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'mini_stock_data.csv'))
ts_df = pd.read_csv(Path(cwd, 'utilities', 'tests', 'data', 'example_timeseries.csv'))

# Partially defined dictionary-1
features_1 = {
    "sepal_length": {
        "type": "continuous",
        'bounds': {
            'min': 2.72,
            'max': 3,
            'allow_null': True
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


@pytest.mark.parametrize('adc', [
    ("MongoDBData", iris_df),
    ("SQLTableData", iris_df),
    ("ParquetDataFile", iris_df),
    ("ParquetDataset", iris_df),
    ("TabularFile", iris_df),
    ("DaskDataFrameData", iris_df),
    ("DataFrameData", iris_df),
], indirect=True)
def test_infer_feature_attributes(adc):
    """Litmus test for infer feature types for iris dataset."""
    expected_types = {
        "sepal_length": "continuous",
        "sepal_width": "continuous",
        "petal_length": "continuous",
        "petal_width": "continuous",
        "class": "nominal"
    }

    features = infer_feature_attributes(adc)

    for feature, attributes in features.items():
        assert expected_types[feature] == attributes['type']


@pytest.mark.parametrize('adc', [
    ("MongoDBData", int_df),
    ("SQLTableData", int_df),
    ("ParquetDataFile", int_df),
    ("ParquetDataset", int_df),
    ("TabularFile", int_df),
    ("DaskDataFrameData", int_df),
    ("DataFrameData", int_df),
], indirect=True)
@pytest.mark.parametrize(
    'feature, nominality', [
        # "id_no" _would_ be inferred "continuous", but we specifically tell
        # `_process()` that it is indeed an ID feature, so it
        # will be set to "nominal".
        ('id_no', 'nominal'),

        # The "badge_no" feature contains ALL unique values so it exceeds the
        # sqrt(total num. rows) test, but they are all the same length
        # integers, so it passes "all the same length" check.
        ('badge_no', 'nominal'),

        # The "salary" feature has mostly uniques (some duplicates) but too
        # many that it readily exceeds the threshold of sqrt(total num. rows)
        # and they are not all the same length, so, "continuous".
        ('salary', 'continuous'),

        # The "dept_no" feature has a number of uniques that exceed the
        # sqrt(total num. rows) but all the integers are the same length,
        # so "nominal".
        ('dept_no', 'nominal'),

        # This column is all None, will be returned as "continuous".
        ('hat_size', 'continuous'),
    ]
)
def test_integer_nominality(feature, nominality, adc):
    """Exercise infer_feature_attributes for integers and their nominality."""
    inferred_features = infer_feature_attributes(adc, id_feature_name=['id_no'])
    assert inferred_features[feature]['type'] == nominality


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
@pytest.mark.parametrize('data, is_time, expected_format, provided_format', [
    (pd.DataFrame(["08:08:08"], columns=['a']), True, '%H:%M:%S', None),
    (pd.DataFrame(["8:8:8"], columns=['a']), True, '%H:%M:%S', None),
    (pd.DataFrame(["8:59:59am"], columns=['a']), True, '%I:%M:%S%p', None),
    (pd.DataFrame(["01:00:00"], columns=['a']), True, '%H:%M:%S', None),
    (pd.DataFrame(["23:59:59"], columns=['a']), True, '%H:%M:%S', None),
    (pd.DataFrame(["23:59:59.59"], columns=['a']), True, '%H:%M:%S.%f', None),
    (pd.DataFrame(["2:30 am"], columns=['a']), True, '%I:%M %p', None),
    (pd.DataFrame(["2:01 pm"], columns=['a']), True, '%I:%M %p', None),
    (pd.DataFrame(["4:25"], columns=['a']), True, '%H:%M', None),
    (pd.DataFrame(["20:00"], columns=['a']), True, '%H:%M', None),
    (pd.DataFrame(["1am"], columns=['a']), True, '%I%p', None),
    (pd.DataFrame(["12 pm"], columns=['a']), True, '%I %p', None),
    (pd.DataFrame(["-01:01:01"], columns=['a']), False, None, None),
    (pd.DataFrame(["24:0:0"], columns=['a']), False, None, None),
    (pd.DataFrame(["59:0:0"], columns=['a']), False, None, None),
    (pd.DataFrame(["3:60"], columns=['a']), False, None, None),
    (pd.DataFrame(["12 o'clock"], columns=['a']), False, None, None),
    (pd.DataFrame(["10pm is the time"], columns=['a']), False, None, None),
    (pd.DataFrame(["8.5:32:32"], columns=['a']), False, None, None),
    (pd.DataFrame([["2020-01-01T10:00:00"]], columns=['a']), False, None, None),
    (pd.DataFrame([["2020-01-01"]], columns=['a']), False, None, None),
    (pd.DataFrame(["08/03/1999 23:59:59"], columns=['a']), False, None, '%M/%D/%Y %H:%M:%S'),
    (pd.DataFrame(["5"], columns=['a']), False, None, '%C'),
    (pd.DataFrame(["1999 23"], columns=['a']), False, None, '%Y %H'),
])
def test_infer_time_features(adc, data, is_time, expected_format, provided_format):
    """Test IFA against many possible valid and invalid time-only features."""
    # First, transfer data to empty ADC
    convert_data(DataFrameData(data), adc)
    # Test
    ifa = InferFeatureAttributesAbstractData(adc)
    feature_type, _ = ifa._get_feature_type('a')
    if is_time:
        assert feature_type == FeatureType.TIME
        features = infer_feature_attributes(adc, tight_bounds=['a'],
                                            datetime_feature_formats={'a': provided_format})
        assert features['a']['type'] == 'continuous'
        assert features['a']['date_time_format'] == expected_format
    else:
        assert feature_type != FeatureType.TIME


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
@pytest.mark.parametrize('data, tight_bounds, provided_format, expected_bounds, cycle_length', [
    (
        pd.DataFrame(["00:00:00", "23:59:59"], columns=['a']), ['a'], None,
        {'min': 0, 'max': 86399, 'observed_min': 0, 'observed_max': 86399, 'allow_null': True}, 86400
    ),
    (
        pd.DataFrame(["03:00:00.0", "12:00:01.5"], columns=['a']), ['a'], None,
        {'min': 10800, 'max': 43201.5, 'observed_min': 10800, 'observed_max': 43201.5, 'allow_null': True}, 86400
    ),
    (
        pd.DataFrame(["03:00:00.0", "12:00:01.5"], columns=['a']), None, None,
        {'min': 0, 'max': 86400, 'observed_min': 10800.0, 'observed_max': 43201.5, 'allow_null': True}, 86400
    ),
    (
        pd.DataFrame(["25:0", "30:0"], columns=['a']), ['a'], '%M:%S',
        {'min': 1500, 'max': 1800, 'observed_min': 1500, 'observed_max': 1800, 'allow_null': True}, 3600
    ),
    (
        pd.DataFrame(["25.0", "30.5"], columns=['a']), None, '%S.%f',
        {'min': 0, 'max': 60, 'observed_min': 25.0, 'observed_max': 30.5, 'allow_null': True}, 60
    ),
    (
        pd.DataFrame(["5", "7"], columns=['a']), None, '%f',
        {'min': 0, 'max': 1, 'observed_min': 0.5, 'observed_max': 0.7, 'allow_null': True}, 1
    ),
])
def test_infer_time_feature_bounds(adc, data, tight_bounds, provided_format, expected_bounds, cycle_length):
    """Test that IFA correctly calculates the bounds and cycle length of time-only features."""
    # First, transfer data to empty ADC
    convert_data(DataFrameData(data), adc)
    # Test
    features = infer_feature_attributes(adc, tight_bounds=tight_bounds,
                                        datetime_feature_formats={'a': provided_format})
    assert features['a']['type'] == 'continuous'
    assert 'cycle_length' in features['a']
    assert features['a']['cycle_length'] == cycle_length
    assert features['a']['bounds'] == expected_bounds
    assert features['a']['date_time_format'] is not None
    assert features['a']['data_type'] == "formatted_time"


@pytest.mark.parametrize("adc", [
    ("MongoDBData", iris_df),
    ("SQLTableData", iris_df),
    ("ParquetDataFile", iris_df),
    ("ParquetDataset", iris_df),
    ("TabularFile", iris_df),
    ("DaskDataFrameData", iris_df),
    ("DataFrameData", iris_df),
], indirect=True)
@pytest.mark.parametrize('should_include, dependent_features', [
    (False, None),
    (True, {'sepal_length': ['sepal_width', 'class']}),
    (True, {'sepal_width': ['sepal_length']}),
    (True, {'sepal_length': ['class']}),
    (False, None),
    (True, None),
])
def test_dependent_features(adc, should_include, dependent_features):
    """Test depdendent features are added to feature attributes dict."""
    features = infer_feature_attributes(adc, dependent_features=dependent_features)

    if should_include:
        # Should include dependent features
        if dependent_features:
            for feat, dep_feats in dependent_features.items():
                assert 'dependent_features' in features[feat]
                for dep_feat in dep_feats:
                    assert dep_feat in features[feat]['dependent_features']
    else:
        # Should not include dependent features
        for attributes in features.values():
            assert 'dependent_features' not in attributes


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
@pytest.mark.parametrize('tight_bounds, data, expected_bounds', [
    (None, [2, 3, 4, 5, 6, 7], {'min': 0, 'max': 10, 'observed_min': 2, 'observed_max': 7, 'allow_null': False}),
    (None, [2, 3, 4, 4, 5, 6, 6, 6, 6], {'min': 0, 'max': 6, 'observed_min': 2, 'observed_max': 6, 'allow_null': False}),  # noqa: E501
    (None, [2, 3, 4, 4, 4, 4, 6, 6, 6, 6], {'min': 0, 'max': 6.0, 'observed_min': 2.0, 'observed_max': 6.0, 'allow_null': False}),  # noqa: E501
    (None, [2, 2, 2, 2, 4, 5, 6, 6, 6, 6], {'min': 2.0, 'max': 6.0, 'observed_min': 2.0, 'observed_max': 6.0, 'allow_null': False}),  # noqa: E501
    (None, [2, 2, 2, 2, 4, 5, 6, 6, 6, 6, 6], {'min': 0, 'max': 6.0, 'observed_min': 2.0, 'observed_max': 6.0, 'allow_null': False}),  # noqa: E501
    (None, [2, 2, 2, 2, 4, 5, 6, 7], {'min': 2.0, 'max': 10.0, 'observed_min': 2.0, 'observed_max': 7.0, 'allow_null': False}),  # noqa: E501
    (['a'], [2, 3, 4, 5, 6, 7], {'min': 2, 'max': 7, 'observed_min': 2, 'observed_max': 7, 'allow_null': False}),
    (['a'], [2, 3, 4, None, 6, 7], {'min': 2, 'max': 7, 'observed_min': 2, 'observed_max': 7, 'allow_null': True}),
    (
        ['a'],
        ['1905-01-01', '1904-05-03', '2020-01-15', '2000-04-26', '2000-04-24'],
        {'min': '1904-05-03', 'max': '2020-01-15', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15', 'allow_null': True},  # noqa: E501
    ),
    (
        None,
        ['1905-01-01', '1904-05-03', '2020-01-15', '2000-04-26', '2000-04-24'],
        {'min': '1829-04-11', 'max': '2095-02-04', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15', 'allow_null': True},  # noqa: E501
    ),
    (
        None,
        ['1905-01-01', '1904-05-03', '2020-01-15', '2020-01-15', '2020-01-15',
         '2020-01-15', '2000-04-26', '2000-04-24'],
        {'min': '1829-04-11', 'max': '2020-01-15', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15', 'allow_null': True},  # noqa: E501
    ),
    (
        None,
        ['1905-01-01', '1904-05-03', '1904-05-03', '1904-05-03', '1904-05-03',
         '2020-01-15', '2020-01-15', '2020-01-15', '2020-01-15', '2000-04-26',
         '2000-04-24'],
        {'min': '1904-05-03', 'max': '2020-01-15', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15', 'allow_null': True},  # noqa: E501
    ),
    (
        None,
        ["1905-01-01T00:00:00+0100", "2022-03-26T00:00:00+0500",
         "1904-05-03T00:00:00+0500", "1904-05-03T00:00:00+0500",
         "1904-05-03T00:00:00+0500", "1904-05-03T00:00:00-0200",
         "1904-05-03T00:00:00+0500", "2022-01-15T00:00:00+0500"],
        {'min': '1904-05-03T00:00:00+0500', 'max': '2098-09-17T14:04:45+0500',
         'observed_min': '1904-05-03T00:00:00+0500', 'observed_max': '2022-03-26T00:00:00+0500', 'allow_null': True},  # noqa: E501
    ),
])
def test_infer_feature_bounds(adc, data, tight_bounds, expected_bounds):
    """Test the infer_feature_bounds() method."""
    df = pd.DataFrame(pd.Series(data), columns=['a'])
    convert_data(DataFrameData(df), adc)
    features = infer_feature_attributes(adc, tight_bounds=tight_bounds)
    assert features['a']['type'] == 'continuous'
    assert 'bounds' in features['a']
    assert features['a']['bounds'] == expected_bounds


@pytest.mark.parametrize("adc", [
    ("MongoDBData", stock_df),
    ("SQLTableData", stock_df),
    ("ParquetDataFile", stock_df),
    ("ParquetDataset", stock_df),
    ("TabularFile", stock_df),
    ("DaskDataFrameData", stock_df),
    ("DataFrameData", stock_df),
], indirect=True)
@pytest.mark.parametrize("tight_bounds", [
    (['DATE', 'TURNOVER', '%DELIVERABLE']),
    (['DATE', '%DELIVERABLE']),
    (['DATE', 'TURNOVER']),
    (['%DELIVERABLE', 'TURNOVER']),
    (['DATE']),
    (['TURNOVER']),
    (['%DELIVERABLE']),
    ([''])
])
def test_tight_bounds(adc, tight_bounds):
    """Test the tight_bounds argument with a features list."""
    features = infer_feature_attributes(adc, tight_bounds=tight_bounds)

    all_tight_bounds = infer_feature_attributes(adc, tight_bounds=features.get_names())
    no_tight_bounds = infer_feature_attributes(adc)

    for feature in features.keys():
        if 'bounds' not in features[feature]:
            continue
        if feature in tight_bounds:
            assert features[feature]['bounds'] == all_tight_bounds[feature]['bounds']
        else:
            assert features[feature]['bounds'] == no_tight_bounds[feature]['bounds']


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
@pytest.mark.parametrize('data, types, expected_types, is_valid', [
    (pd.DataFrame({'a': [0, 1, 2, 0, 1, 2]}), dict(a='continuous'), dict(a='continuous'), True),
    (pd.DataFrame({'a': [0, 1, 2], 'b': ['1', '2', '3']}, columns=['a', 'b']), dict(continuous=['a', 'b']),
     dict(a='continuous', b='continuous'), True),
    (pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7]}), dict(a='nominal'), dict(a='nominal'), True),
    (pd.DataFrame({'a': [True, False, False, True]}), dict(a='continuous'), dict(a='nominal'), True),
    (pd.DataFrame({'nominal': [True, False, False, True]}), dict(nominal='nominal'), dict(nominal='nominal'), True),
    (pd.DataFrame({'nominal': [True, False, False, True]}), dict(nominal=['nominal']), dict(nominal='nominal'), True),
    (pd.DataFrame({'ordinal': [True, False, False, True]}), dict(ordinal='nominal'), dict(ordinal='nominal'), True),
    (pd.DataFrame({'continuous': [True, False, False]}), dict(continuous='nominal'), dict(continuous='nominal'), True),
    (pd.DataFrame({'a': [True, False, False, True]}), dict(a='boolean'), {}, False),
    (pd.DataFrame({'a': ['one', 'two', 'three', 'four']}), dict(ordinal=['a']), {}, False),
    (pd.DataFrame({'a': ['one', 'two', 'three', 'four']}), dict(a='ordinal'), {}, False),
])
def test_preset_feature_types(adc, data, types, expected_types, is_valid):
    """Test that infer_feature_attributes correctly presets feature types with the `types` parameter."""
    convert_data(DataFrameData(data), adc)
    features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_valid:
            features = infer_feature_attributes(adc, types=types, )
            for feature_name, expected_type in expected_types.items():
                # Make sure it is the correct type
                assert features[feature_name]['type'] == expected_type
                # All features in this test, including nominals, should have bounds (at the very least: `allow_null`)
                assert 'allow_null' in features[feature_name].get('bounds', {}).keys()
        else:
            with pytest.raises(ValueError):
                infer_feature_attributes(adc, types=types)


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    # ("TabularFile", pd.DataFrame()), -->    When using convert_data, the 'object' dtype is correctly carried over to
    ("DaskDataFrameData", pd.DataFrame()),  # the destination chunk during `write_chunk`; however, in the case of
    ("DataFrameData", pd.DataFrame()),      # tabular ADCs, it is lost on conversion to .csv, breaking this test.
], indirect=True)
@pytest.mark.parametrize(
    'series, ordinals, min_value, max_value', [
        (  # ordinal strings
            ['grape', 'apple', 'banana', 'banana', 'cherry', 'apple', 'apple', 'fig', 'cherry', 'banana'],
            ['apple', 'banana', 'cherry'],
            'apple', 'cherry'
        ),
        (  # ordinal strings, includes an empty string
            ['**', '*', '***', '*', '****', '*****', '**', '', '****', '***'],
            ['', '*', '**', '***', '****', '*****'],
            '', '*****'
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
            ['4', '2', '1', '7', '3', '3', '8', '2', '1', '0'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            '1', '8'
        ),
        (  # ordinal numerals as string ordinals with unusual ordering
            ['4', '2', '1', '7', '3', '3', '8', '2', '1', '0'],
            ['5', '7', '3', '2', '4', '9', '10', '8', '6', '1'],
            '7', '1'
        ),
        (  # floats as ordinals
            [0.4, 0.2, 0.1, 0.7, 0.3, 0.3, 0.8, 0.2, 0.1, 0.0],
            [0.1, 0.2, 0.3, 0.4, 5, 6, 7, 8, 9, 10],
            0.0, 0.8
        ),
        (  # Dates as ordinals
            ["1-Mar-2020", "1-Mar-2020", "1-Apr-2020", "1-Mar-2020", "1-Feb-2020", '1-Dec-2020', '1-Jul-2020'],
            ["1-Jan-2020", "1-Feb-2020", "1-Mar-2020", "1-Apr-2020", "1-May-2020", "1-Jun-2020"],
            '1-Feb-2020', '1-Apr-2020'
        )
    ]
)
def test_observed_ordinal_values(adc, series, ordinals, min_value, max_value):
    """Test that observed_min/max in ordinal features works as expected."""
    data = pd.DataFrame({'a': series})
    convert_data(DataFrameData(data), adc)
    features = infer_feature_attributes(adc, ordinal_feature_values={'a': ordinals})
    assert features['a']['bounds']['observed_min'] == min_value
    assert features['a']['bounds']['observed_max'] == max_value


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
def test_formatted_date_time(adc):
    """Test formatted_date_time is set when a datetime, and raises when no date_time_format is specified."""
    data = pd.DataFrame({
        'a': [0, 1, 2, 3],
        'time': ['10-10', '04-25', '10-30', '12-01'],
        'custom': ['2010/10/10', '2010/10/11', '2010/10/12', '2010/10/14'],
        'iso': ['2010-10-10', '2010-10-11', '2010-10-12', '2010-10-14']
    })

    convert_data(DataFrameData(data), adc)

    # Verify formatted_date_time is set when a date_time_format is configured
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        features = infer_feature_attributes(adc, datetime_feature_formats={"custom": "%Y/%m/%d"},
                                            default_time_zone="UTC")
        assert features['a']['data_type'] != "formatted_date_time"
        # custom feature dates should be formatted_date_time
        assert features['custom']['data_type'] == "formatted_date_time"
        # auto detected iso dates should be formatted_date_time
        assert features['iso']['data_type'] == "formatted_date_time"


@pytest.mark.parametrize("adc", [
    ("MongoDBData", pd.DataFrame()),
    ("SQLTableData", pd.DataFrame()),
    ("ParquetDataFile", pd.DataFrame()),
    ("ParquetDataset", pd.DataFrame()),
    ("TabularFile", pd.DataFrame()),
    ("DaskDataFrameData", pd.DataFrame()),
    ("DataFrameData", pd.DataFrame()),
], indirect=True)
def test_constrained_date_bounds(adc):
    """Constrained datetime formats may make bounds undeterminable."""
    df = pd.DataFrame([["01"], ["02"]], columns=["date"])
    convert_data(DataFrameData(df), adc)
    with pytest.warns(match="bounds could not be computed. This is likely due to a constrained date time format"):
        # Loose bounds may cause min bound to be > max bound if the date format is constrained
        infer_feature_attributes(adc, datetime_feature_formats={"date": "%m"})


def test_parquet_dataset_with_s3():
    """Test ParquetDataset with s3 integration."""
    anon_options = {"anon": True}
    data_sources = [
        "s3://howso-ci-test-anon/parquet_files/bank/",
        "s3://howso-ci-test-anon/parquet_files/iris/",
    ]
    for src in data_sources:
        adc = make_data_source(src, storage_options=anon_options)
        infer_feature_attributes(adc)


def test_ambiguous_datetime_format():
    """Test the NYPD arrest dataset."""
    adc = make_data_source(nypd_arrest_pq_path)  # Contains a non-ISO8601 date column
    with pytest.warns(UserWarning, match="these features will be treated as nominal strings"):
        infer_feature_attributes(adc)


def test_datetime_empty_time_values():
    """Test that datetimes with an empty time value still are determined datetime features with the correct format."""
    df = pd.DataFrame({'a': ['2025-08-22T00:00:00'], 'b': ['2025-08-22 00:00:00']})
    adc = make_data_source(df)
    features = infer_feature_attributes(adc, default_time_zone='UTC')
    assert features['a']['date_time_format'] == '%Y-%m-%dT%H:%M:%S'
    assert features['b']['date_time_format'] == '%Y-%m-%d %H:%M:%S'


def test_empty_string_first_non_nulls():
    """Test that IFA correctly handles first non-null values that are empty strings."""
    df = pd.DataFrame({'a': ['', 'ahoy', 'howdy']})
    adc = make_data_source(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer_feature_attributes(adc)
    df = pd.DataFrame({'a': ['', 'ahoy', 'howdy'], 'b': ['\n', '8/26/2025', '8/3/1999']})
    adc = make_data_source(df)
    with pytest.warns(UserWarning, match="these features will be treated as nominal strings"):
        infer_feature_attributes(adc)


def test_dataframe_datetime64_dtype_column():
    """Test that IFA correctly handles datetime64 dtype columns."""
    df = pd.DataFrame({'a': ['2025-11-24T14:30.000']})
    # Convert to datetime64 dtype
    df['a'] = df['a'].astype('datetime64[ns]')
    adc = make_data_source(df)
    infer_feature_attributes(adc)
