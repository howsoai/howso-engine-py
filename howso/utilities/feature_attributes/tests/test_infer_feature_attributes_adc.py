"""Tests the `infer_feature_attributes` package with AbstractData classes."""
from collections import OrderedDict
import datetime
from pathlib import Path
from unittest.mock import patch
import warnings

import mongomock
import pandas as pd
import pytest

try:
    from howso.connectors.abstract_data import MongoDBData
except (ModuleNotFoundError, ImportError):
    pytest.skip("howso-engine-connectors not installed", allow_module_level=True)
from howso.utilities.feature_attributes import infer_feature_attributes
from howso.utilities.feature_attributes.abstract_data import InferFeatureAttributesAbstractData
from howso.utilities.features import FeatureType

cwd = Path(__file__).parent.parent.parent.parent
iris_path = Path(cwd, 'utilities', 'tests', 'data', 'iris.csv')
int_path = Path(cwd, 'utilities', 'tests', 'data', 'integers.csv')
stock_path = Path(cwd, 'utilities', 'tests', 'data', 'mini_stock_data.csv')
ts_path = Path(cwd, 'utilities', 'tests', 'data', 'example_timeseries.csv')

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

# Partially defined "ordered" dict
features_4 = OrderedDict(
    (f_name, features_3[f_name]) for f_name in features_3
)


@patch("howso.connectors.abstract_data.mongodb_data.MongoClient", new=mongomock.MongoClient)
def df_to_mongo_adc(df: pd.DataFrame) -> MongoDBData:
    """Helper function to convert a Pandas DataFrame to a MongoDB ADC w/ mocked MongoDB instance."""
    adc = MongoDBData(
        uri="mongodb://localhost",
        database_name="test_db",
        collection_name="test_collection"
    )
    # Populate the mocked MongoDB with the provided DataFrame
    adc._collection.insert_many(df.to_dict(orient="records"))
    return adc


def test_infer_features_attributes():
    """Litmus test for infer feature types for iris dataset."""
    df = pd.read_csv(iris_path)
    adc = df_to_mongo_adc(df)

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
def test_integer_nominality(feature, nominality):
    """Exercise infer_feature_attributes for integers and their nominality."""
    df = pd.read_csv(int_path)
    adc = df_to_mongo_adc(df)
    inferred_features = infer_feature_attributes(adc, id_feature_name=['id_no'])
    assert inferred_features[feature]['type'] == nominality


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
def test_infer_time_features(data, is_time, expected_format, provided_format):
    """Test IFA against many possible valid and invalid time-only features."""
    adc = df_to_mongo_adc(data)
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
def test_infer_time_feature_bounds(data, tight_bounds, provided_format, expected_bounds, cycle_length):
    """Test that IFA correctly calculates the bounds and cycle length of time-only features."""
    adc = df_to_mongo_adc(data)
    features = infer_feature_attributes(adc, tight_bounds=tight_bounds,
                                        datetime_feature_formats={'a': provided_format})
    assert features['a']['type'] == 'continuous'
    assert 'cycle_length' in features['a']
    assert features['a']['cycle_length'] == cycle_length
    assert features['a']['bounds'] == expected_bounds
    assert features['a']['date_time_format'] is not None
    assert features['a']['data_type'] == "formatted_time"


@pytest.mark.parametrize('should_include, base_features, dependent_features', [
    (False, None, None),
    (True, None, {'sepal_length': ['sepal_width', 'class']}),
    (True, {"sepal_length": {"type": "continuous"}},
     {'sepal_width': ['sepal_length']}),
    (True, {"sepal_length": {"type": "continuous"}},
     {'sepal_length': ['class']}),
    (False, {"sepal_length": {"type": "continuous"}},
     None),
    (True, {"sepal_length": {"dependent_features": ["class"]}},
     None),
])
def test_dependent_features(should_include, base_features, dependent_features):
    """Test depdendent features are added to feature attributes dict."""
    df = pd.read_csv(iris_path)
    adc = df_to_mongo_adc(df)
    features = infer_feature_attributes(adc, features=base_features, dependent_features=dependent_features)

    if should_include:
        # Should include dependent features
        if dependent_features:
            for feat, dep_feats in dependent_features.items():
                assert 'dependent_features' in features[feat]
                for dep_feat in dep_feats:
                    assert dep_feat in features[feat]['dependent_features']
        # Make sure dependent features provided in the base dict are also included
        if base_features:
            for feat in base_features.keys():
                if 'dependent_features' in base_features[feat]:
                    assert 'dependent_features' in features[feat]
                    for dep_feat in base_features[feat]['dependent_features']:
                        assert dep_feat in features[feat]['dependent_features']
    else:
        # Should not include dependent features
        for attributes in features.values():
            assert 'dependent_features' not in attributes


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
    (
        ['a'],
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {'min': '1904-05-03T00:00:00', 'max': '2022-03-26T00:00:00',
         'observed_min': '1904-05-03T00:00:00', 'observed_max': '2022-03-26T00:00:00', 'allow_null': True},  # noqa: E501
    ),
])
def test_infer_feature_bounds(data, tight_bounds, expected_bounds):
    """Test the infer_feature_bounds() method."""
    df = pd.DataFrame(pd.Series(data), columns=['a'])
    adc = df_to_mongo_adc(df)
    features = infer_feature_attributes(adc, tight_bounds=tight_bounds)
    assert features['a']['type'] == 'continuous'
    assert 'bounds' in features['a']
    assert features['a']['bounds'] == expected_bounds


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
def test_tight_bounds(tight_bounds):
    """Test the tight_bounds argument with a features list."""
    df = pd.read_csv(stock_path)
    adc = df_to_mongo_adc(df)
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
def test_preset_feature_types(data, types, expected_types, is_valid):
    """Test that infer_feature_attributes correctly presets feature types with the `types` parameter."""
    adc = df_to_mongo_adc(data)
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
def test_observed_ordinal_values(series, ordinals, min_value, max_value):
    """Test that observed_min/max in ordinal features works as expected."""
    data = pd.DataFrame({'a': series})
    adc = df_to_mongo_adc(data)
    features = infer_feature_attributes(adc, ordinal_feature_values={'a': ordinals})
    assert features['a']['bounds']['observed_min'] == min_value
    assert features['a']['bounds']['observed_max'] == max_value


def test_formatted_date_time():
    """Test formatted_date_time is set when a datetime, and raises when no date_time_format is specified."""
    data = pd.DataFrame({
        'a': [0, 1, 2, 3],
        'time': ['10-10', '04-25', '10-30', '12-01'],
        'custom': ['2010/10/10', '2010/10/11', '2010/10/12', '2010/10/14'],
        'iso': ['2010-10-10', '2010-10-11', '2010-10-12', '2010-10-14']
    })

    adc = df_to_mongo_adc(data)

    # When data_type is formatted_date_time, a date_time_format must be set
    with pytest.raises(
        ValueError,
        match='must have a `date_time_format` defined when its `data_type` is "formatted_date_time"'
    ):
        infer_feature_attributes(adc, features={
            'custom': {
                "data_type": "formatted_date_time"
            }
        })

    # When data_type is formatted_time, a date_time_format must be set
    with pytest.raises(
        ValueError,
        match='must have a `date_time_format` defined when its `data_type` is "formatted_time"'
    ):
        infer_feature_attributes(adc, features={
            'time': {
                "data_type": "formatted_time"
            }
        })

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


def test_constrained_date_bounds():
    """Constrained datetime formats may make bounds undeterminable."""
    df = pd.DataFrame([["01"], ["02"]], columns=["date"])
    adc = df_to_mongo_adc(df)
    with pytest.warns(match="bounds could not be computed. This is likely due to a constrained date time format"):
        # Loose bounds may cause min bound to be > max bound if the date format is constrained
        infer_feature_attributes(adc, datetime_feature_formats={"date": "%m"})
