"""Tests the `infer_feature_attributes` package."""
from collections import OrderedDict
from copy import copy
import datetime
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Iterable
import warnings

from howso.utilities.feature_attributes import infer_feature_attributes
from howso.utilities.feature_attributes.base import FeatureAttributesBase, FLOAT_MAX, FLOAT_MIN, INTEGER_MAX
from howso.utilities.feature_attributes.pandas import InferFeatureAttributesDataFrame
from howso.utilities.features import FeatureType
import numpy as np
import pandas as pd
import pytest
import pytz

if platform.system().lower() == 'windows':
    DT_MAX = '6053-01-24'
    ALMOST_DT_MAX = '6053-01-23'
else:
    DT_MAX = '2262-04-11'
    ALMOST_DT_MAX = '2262-04-10'

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
        assert expected_types[feature] == attributes['type']


@pytest.mark.parametrize(
    "features, max_workers",
    [
        (features_1, 0),
        (features_2, 0),
        (features_3, 0),
        (features_4, 0),
        (features_1, 2),
        (features_2, 2),
        (features_3, 2),
        (features_4, 2),
    ]
)
def test_partially_filled_feature_types(features: dict, max_workers: int) -> None:
    """
    Make sure the partially filled feature types remain intact.

    Note:
        max_workers: 0 - Forces the non-multi-processing path which would
                         be normal for this dataset anyway.
        max_workers: 2 - Forces the multi-processing path which would otherwise
                         be unnatural for this dataset.

    Parameters
    ----------
    df: pandas.DataFrame
    features
    """
    df = pd.read_csv(iris_path)
    pre_inferred_features = features.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred_features = infer_feature_attributes(
            df, features=features, max_workers=max_workers)

    for k, v in pre_inferred_features.items():
        assert v['type'] == inferred_features[k]['type']

        if 'bounds' in v:
            # Make sure the bounds are not altered
            # by _process()
            assert v['bounds'] == inferred_features[k]['bounds']


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
    inferred_features = infer_feature_attributes(df, id_feature_name=['id_no'])
    assert inferred_features[feature]['type'] == nominality


@pytest.mark.parametrize("data, expected_type", [
    # Integer
    (pd.DataFrame([[1], [None]], dtype='Int8', columns=['a']),
     {'data_type': str(FeatureType.INTEGER), 'size': 1}),
    (pd.DataFrame([[1], [16]], dtype='int', columns=['a']),
     {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    # Float
    (pd.DataFrame([[1.0], [4.4]], dtype='float', columns=['a']),
     {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    (pd.DataFrame([[None], [4.4]], dtype='float32', columns=['a']),
     {'data_type': str(FeatureType.NUMERIC), 'size': 4}),
    # Boolean
    (pd.DataFrame([[True], [False], [None]], dtype='bool', columns=['a']),
     {'data_type': str(FeatureType.BOOLEAN)}),
    # String
    (pd.DataFrame([["test"], [None]], columns=['a']),
     {'data_type': str(FeatureType.STRING)}),
    (pd.DataFrame([["test"], [None]], dtype='string', columns=['a']),
     {'data_type': str(FeatureType.STRING)}),
    (pd.DataFrame([["test"], [None]], dtype=np.bytes_, columns=['a']),
     {'data_type': str(FeatureType.STRING)}),
    (pd.DataFrame([["test"]], dtype='S', columns=['a']),
     {'data_type': str(FeatureType.STRING)}),
    (pd.DataFrame([["test"]], dtype='U', columns=['a']),
     {'data_type': str(FeatureType.STRING)}),
    # Datetime
    (pd.DataFrame([["2020-01-01T10:00:00"]], dtype='datetime64[ns]', columns=['a']),
     {'data_type': str(FeatureType.DATETIME)}),
    (pd.DataFrame([["2020-01-01"]], dtype='datetime64[ns]', columns=['a']),
     {'data_type': str(FeatureType.DATETIME)}),
    (pd.DataFrame([[datetime.datetime.now()]], columns=['a']),
     {'data_type': str(FeatureType.DATETIME)}),
    (pd.DataFrame([[datetime.datetime.now(pytz.timezone('US/Eastern'))]], columns=['a']),
     {'data_type': str(FeatureType.DATETIME), 'timezone': 'US/Eastern'}),
    (pd.DataFrame([[datetime.datetime.now(pytz.FixedOffset(300))]], columns=['a']),
     {'data_type': str(FeatureType.DATETIME)}),
    # Date
    (pd.DataFrame([[datetime.date(2020, 1, 1)]], columns=['a']),
     {'data_type': str(FeatureType.DATE)}),
    # Timedelta
    (pd.DataFrame([[datetime.timedelta(days=1)]], columns=['a']),
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    (pd.DataFrame([[np.timedelta64(5, 'D')]], columns=['a']),
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    (pd.DataFrame([[np.timedelta64(5, 'Y')]], columns=['a']),
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    (pd.DataFrame([[np.timedelta64(5000, 'ns')]], columns=['a']),
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    (pd.DataFrame([[np.timedelta64(5000, 's')]], columns=['a']),
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    (pd.DataFrame([[np.timedelta64(60, 'm')]], columns=['a']),
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
])
def test_get_feature_type(data, expected_type):
    """Test get_feature_type returns expected data types."""
    infer = InferFeatureAttributesDataFrame(data)
    feature_type, original_type = infer._get_feature_type('a')
    expected_feature_type = expected_type.pop('data_type')
    assert str(feature_type) == expected_feature_type
    assert original_type == expected_type


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
    (pd.DataFrame([datetime.time(15)], columns=['a']), True, '%H:%M:%S', None),
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
    ifa = InferFeatureAttributesDataFrame(data)
    feature_type, _ = ifa._get_feature_type('a')
    if is_time:
        assert feature_type == FeatureType.TIME
        features = infer_feature_attributes(data, tight_bounds=['a'],
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
    features = infer_feature_attributes(data, tight_bounds=tight_bounds,
                                        datetime_feature_formats={'a': provided_format})
    assert features['a']['type'] == 'continuous'
    assert 'cycle_length' in features['a']
    assert features['a']['cycle_length'] == cycle_length
    assert features['a']['bounds'] == expected_bounds
    assert features['a']['date_time_format'] is not None
    assert features['a']['data_type'] == "formatted_time"


@pytest.mark.parametrize('data, data_type', [
    (123, 'float128'),
])
def test_get_feature_type_raises(data, data_type):
    """Test get_feature_type raises exception."""
    # Place this here to avoid circular import
    from howso.client.exceptions import HowsoError
    if not hasattr(np, data_type):
        pytest.skip('Unsupported platform')

    with pytest.raises(HowsoError):
        df = pd.DataFrame([[getattr(np, data_type)(data)]], columns=['a'])
        infer_feature_attributes(df)


@pytest.mark.parametrize('should_fail, data', [
    (True, [[1]]),
    (True, {3: [1]}),
    (False, {'col1': [1]}),
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
    features = infer_feature_attributes(df, features=base_features, dependent_features=dependent_features)

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
    (None, [float('nan'), float('nan')], {'allow_null': True}),
    (['a'], [2, 3, 4, 5, 6, 7], {'min': 2, 'max': 7, 'observed_min': 2, 'observed_max': 7, 'allow_null': False}),
    (['a'], [2, 3, 4, None, 6, 7], {'min': 2, 'max': 7, 'observed_min': 2, 'observed_max': 7, 'allow_null': True}),
    (
        ['a'],
        ['1905-01-01', '1904-05-03', '2020-01-15', '2000-04-26', '2000-04-24'],
        {'min': '1904-05-03', 'max': '2020-01-15', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15'}
    ),
    (
        None,
        ['1905-01-01', '1904-05-03', '2020-01-15', '2000-04-26', '2000-04-24'],
        {'min': '1829-04-11', 'max': '2095-02-04', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15'}
    ),
    (
        None,
        ['1905-01-01', '1904-05-03', '2020-01-15', '2020-01-15', '2020-01-15',
         '2020-01-15', '2000-04-26', '2000-04-24'],
        {'min': '1829-04-11', 'max': '2020-01-15', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15'}
    ),
    (
        None,
        ['1905-01-01', '1904-05-03', '1904-05-03', '1904-05-03', '1904-05-03',
         '2020-01-15', '2020-01-15', '2020-01-15', '2020-01-15', '2000-04-26',
         '2000-04-24'],
        {'min': '1904-05-03', 'max': '2020-01-15', 'observed_min': '1904-05-03', 'observed_max': '2020-01-15'}
    ),
    (
        None,
        ["1905-01-01T00:00:00+0100", "2022-03-26T00:00:00+0500",
         "1904-05-03T00:00:00+0500", "1904-05-03T00:00:00+0500",
         "1904-05-03T00:00:00+0500", "1904-05-03T00:00:00-0200",
         "1904-05-03T00:00:00+0500", "2022-01-15T00:00:00+0500"],
        {'min': '1904-05-03T00:00:00+0500', 'max': '2098-09-17T14:04:45+0500',
         'observed_min': '1904-05-03T00:00:00+0500', 'observed_max': '2022-03-26T00:00:00+0500'}
    ),
    (
        ['a'],
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {'min': '1904-05-03T00:00:00', 'max': '2022-03-26T00:00:00',
         'observed_min': '1904-05-03T00:00:00', 'observed_max': '2022-03-26T00:00:00'}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {'min': '1827-11-08T09:55:14', 'max': '2098-09-17T14:04:45',
         'observed_min': '1904-05-03T00:00:00', 'observed_max': '2022-03-26T00:00:00'}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1), datetime.datetime(1904, 5, 3),
         datetime.datetime(1904, 5, 3), datetime.datetime(1904, 5, 3),
         datetime.datetime(1904, 5, 3), datetime.datetime(1904, 5, 3),
         datetime.datetime(2020, 1, 15), datetime.datetime(2022, 3, 26)],
        {'min': '1904-05-03T00:00:00', 'max': '2098-09-17T14:04:45',
         'observed_min': '1904-05-03T00:00:00', 'observed_max': '2022-03-26T00:00:00'}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(2020, 1, 15, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(2022, 3, 26, tzinfo=pytz.FixedOffset(300))],
        {'min': '1904-05-03T00:00:00+0500', 'max': '2098-09-17T14:04:45+0500',
         'observed_min': '1904-05-03T00:00:00+0500', 'observed_max': '2022-03-26T00:00:00+0500'}
    ),
    (
        None,
        [datetime.datetime(1905, 1, 1, tzinfo=pytz.FixedOffset(100)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(-400)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(1904, 5, 3, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(2020, 1, 15, tzinfo=pytz.FixedOffset(300)),
         datetime.datetime(2022, 3, 26, tzinfo=pytz.FixedOffset(300))],
        {'min': '1904-05-03T00:00:00+0500', 'max': '2098-09-17T14:04:45+0500',
         'observed_min': '1904-05-03T00:00:00+0500', 'observed_max': '2022-03-26T00:00:00+0500'}
    ),
    (
        ['a'],
        [datetime.timedelta(days=1), datetime.timedelta(days=1),
         datetime.timedelta(seconds=5), datetime.timedelta(days=1, seconds=30),
         datetime.timedelta(minutes=50), datetime.timedelta(days=5)],
        {'min': 5, 'max': 5 * 24 * 60 * 60,
         'observed_min': 5, 'observed_max': 5 * 24 * 60 * 60,
         'allow_null': True, 'allow_null': True}
    ),
    (
        None,
        [datetime.timedelta(days=1), datetime.timedelta(days=1),
         datetime.timedelta(seconds=5), datetime.timedelta(days=1, seconds=30),
         datetime.timedelta(minutes=50), datetime.timedelta(days=5),
         datetime.timedelta(days=5), datetime.timedelta(days=5),
         datetime.timedelta(days=5)],
        {'min': 0, 'max': 5 * 24 * 60 * 60.0,
         'observed_min': 5.0, 'observed_max': 5 * 24 * 60 * 60.0,
         'allow_null': True, 'allow_null': True}
    ),
])
def test_infer_feature_bounds(data, tight_bounds, expected_bounds):
    """Test the infer_feature_bounds() method."""
    df = pd.DataFrame(pd.Series(data), columns=['a'])
    features = infer_feature_attributes(df, tight_bounds=tight_bounds)
    assert features['a']['type'] == 'continuous'
    assert 'bounds' in features['a']
    assert features['a']['bounds'] == expected_bounds


@pytest.mark.parametrize(
    "features",
    [features_1, features_2, features_3, features_4]
)
def test_to_json(features: dict) -> None:
    """Test that to_json() method returns a JSON representation of the object."""
    df = pd.read_csv(iris_path)
    pre_inferred_features = features.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred_features = infer_feature_attributes(df, features=features)

    to_json = inferred_features.to_json()
    assert isinstance(to_json, str)
    features = json.loads(to_json)

    # Make sure the json representation has expected data
    for k, v in pre_inferred_features.items():
        assert v['type'] == features[k]['type']
        if 'bounds' in v:
            assert v['bounds'] == features[k]['bounds']


@pytest.mark.parametrize('partial_features', [
    {"sepal_length": {"type": "continuous"},
     'sepal_width': {"dependent_features": ['sepal_length']}}
])
def test_get_parameters(partial_features):
    """Test the get_parameters() method."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df, features=partial_features)

    # Verify dependent_features
    assert 'features' in features.get_parameters()
    for key, value in partial_features.items():
        assert features.get_parameters()['features'][key] == value


def test_get_names_without():
    """Test the get_names() method."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)

    all_features = list(df)

    # Test get all feature names
    assert features.get_names() == all_features

    # Test get feature names without
    without = ['sepal_length', 'petal_length', 'petal_width']
    assert features.get_names(without=without) == [f for f in all_features if f not in without]

    # Test a feature in 'without' that is not in the features list
    with pytest.raises(ValueError):
        without = ['sepal_length', 'petal_length', 'personality']
        features.get_names(without=without)


@pytest.mark.parametrize("types, num", [
    ("continuous", 4),
    ({"continuous"}, 4),
    (('nominal'), 1),
    (['continuous', 'nominal'], 5),
])
def test_get_names_types(types, num):
    """Test the get_names() method with the types parameter."""
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)
    names = features.get_names(types=types)
    assert len(names) == num


def test_copy():
    """Test that copy works as expected."""
    df = pd.read_csv(iris_path)
    f_orig = infer_feature_attributes(df)
    f_copy = copy(f_orig)

    assert f_copy.params == f_orig.params
    orig = f_orig['sepal_width']['bounds']['min']
    assert f_copy['sepal_width']['bounds']['min'] == orig

    # Now, change the orig, so we can ensure that f_copy is independent.
    f_orig['sepal_width']['bounds']['min'] = -2
    # Assert that f_copy was unaffected
    assert f_copy['sepal_width']['bounds']['min'] == orig


@pytest.mark.parametrize("features", [
    ({'OPEN': {'type': 'continuous', 'decimal_places': 3}}),
    ({'VWAP': {'decimal_places': 5}}),
    ({'DATE': {'type': 'continuous', 'date_time_format': '%Y-%m-%f'}}),
    ({'DATE': {'date_time_format': '%Y-%m-%f'}}),
    ({'CLOSE': {'type': 'continuous', 'dependent_features': ['OPEN', 'DATE']}}),
])
def test_partial_features(features):
    """Test that filling a partial features dict works as expected."""
    feature_attributes = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    df = pd.read_csv(stock_path)
    feature_attributes = infer_feature_attributes(df, time_feature_name='DATE', features=features)

    # Ensure that the partial features remain
    for feature in features.keys():
        for k, v in features[feature].items():
            assert k in feature_attributes[feature]
            assert feature_attributes[feature][k] == v


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
    features = infer_feature_attributes(df, tight_bounds=tight_bounds)

    all_tight_bounds = infer_feature_attributes(df, tight_bounds=features.get_names())
    no_tight_bounds = infer_feature_attributes(df)

    for feature in features.keys():
        if 'bounds' not in features[feature]:
            continue
        if feature in tight_bounds:
            assert features[feature]['bounds'] == all_tight_bounds[feature]['bounds']
        else:
            assert features[feature]['bounds'] == no_tight_bounds[feature]['bounds']


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
    features = infer_feature_attributes(df, time_feature_name='date')
    assert features.validate(df, raise_errors=True) is None
    # Mini stock data dataset
    df = pd.read_csv(stock_path)
    features = infer_feature_attributes(df, time_feature_name='DATE')
    assert features.validate(df, raise_errors=True) is None
    # Also try this one with a non-ts infer
    df = pd.read_csv(stock_path)
    features = infer_feature_attributes(df)
    assert features.validate(df, raise_errors=True) is None
    # Should not raise any exceptions and return a "coerced" dataframe
    df = pd.read_csv(iris_path)
    features = infer_feature_attributes(df)
    df['sepal_length'] = df['sepal_length'].astype('int64')
    df = features.validate(df, coerce=True, raise_errors=True)
    assert df is not None
    assert pd.api.types.is_float_dtype(df['sepal_length'])
    # Try validating a categorical feature
    df = pd.read_csv(iris_path)
    features['class']['type'] = 'ordinal'
    features['class']['bounds'] = {}
    unique = list(df['class'].unique())
    features['class']['bounds']['allowed'] = unique
    df['class'] = df['class'].astype(pd.CategoricalDtype(categories=unique))
    df = features.validate(df, coerce=True, raise_errors=True)
    assert df is not None
    assert isinstance(df['class'].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("ftype, data_type, decimal_places, bounds, date_time_format, expected_dtype", [
    ("continuous", "number", 0, {'allow_null': False}, None, "int64"),
    ("continuous", "number", 1, {'allow_null': False}, None, "float64"),
    ("continuous", "number", 0, {'allow_null': False}, "%Y-%m-%d", "datetime64"),
    ("ordinal", "number", 0, {'allow_null': False}, None, "int64"),
    ("ordinal", "number", 2, {'allow_null': True}, None, "float64"),
    ("ordinal", "string", None, {'allowed': ['SBIN'], 'allow_null': False}, None, "object"),
    ("nominal", "number", 0, {'allow_null': False}, None, "int64"),
    ("nominal", "number", 9, {'allow_null': False}, None, "float64"),
    ("nominal", "boolean", 0, {'allow_null': False}, None, "bool"),
])
def test_validate_df_multiple_dtypes(ftype, data_type, decimal_places, bounds, date_time_format,
                                     expected_dtype):
    """Test the validate() method with all possible inferred dtypes."""
    # First, read in the mini_stock_series dataset as it has a variety of data types
    df = pd.read_csv(stock_path)
    # Based on the expected_dtype, choose the feature in the dataset that is loosely described by the given parameters
    if expected_dtype == 'int64':
        feature = 'VOLUME'
    elif expected_dtype == 'float64':
        feature = 'PREV CLOSE'
    elif expected_dtype == 'datetime64':
        feature = 'DATE'
    elif expected_dtype == 'bool':
        # Make a new column of a bool dtype since there are none in the dataset
        df['NEW'] = True
        feature = 'NEW'
    else:
        feature = 'SYMBOL'
    # Infer the feature attributes like normal, but replace the attributes for the chosen feature
    # with our parameter attributes, which should also be considered valid.
    attrs = infer_feature_attributes(df, time_feature_name='DATE')
    attrs[feature] = {
        'type': ftype,
        'data_type': data_type,
        'decimal_places': decimal_places,
        'bounds': bounds,
        'date_time_format': date_time_format,
    }
    if not date_time_format:
        del attrs[feature]['date_time_format']
    # validate() should not raise any errors
    coerced_df = attrs.validate(df, raise_errors=True, coerce=True)
    assert coerced_df is not None
    # coerced_df should also contain a coerced DATE column, as it is originally detected as a string
    assert pd.api.types.is_datetime64_any_dtype(coerced_df['DATE'].dtype)


@pytest.mark.parametrize("extra_attrs, success", (
    ({}, False),
    ({'auto_derive_on_train': False}, False),
    ({'auto_derive_on_train': True}, False),
    ({'derived_feature_code': '{* #VOLUMNE 2.2}'}, False),
    ({'auto_derive_on_train': False,
      'derived_feature_code': '{* #VOLUMNE 2.2}'}, False),
    ({'auto_derive_on_train': True,
      'derived_feature_code': '{* #VOLUMNE 2.2}'}, True),
))
def test_validate_df_missing_features(extra_attrs, success):
    """
    Test that missing features raise warnings in `_validate_df`.

    Specifically, if a feature is to be derived during train, it should be
    exempt from raising warnings that the feature is missing.
    """
    df = pd.read_csv(stock_path)
    attrs = infer_feature_attributes(df, time_feature_name='DATE')
    # Add a would-be derived/computed feature
    attrs['to_be_computed'] = {"type": "continuous"}
    attrs['to_be_computed'].update(extra_attrs)

    if success:
        # We expect this to run without raising an error (due to the warning)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            attrs.validate(df)
    else:
        # We expect this to raise an exception when run.
        with pytest.raises(Exception):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                attrs.validate(df)


@pytest.mark.parametrize("datetime_min_max, float_min_max, int_min_max", [
    (
        ('DATE', None, None, False),
        ('CLOSE', None, None, False),
        ('VOLUME', None, None, False),
    ),
    (
        ('DATE', None, DT_MAX, True),
        ('CLOSE', FLOAT_MIN - .00001, FLOAT_MAX + .00001, True),
        ('VOLUME', int(INTEGER_MAX / 10) * -1, INTEGER_MAX, True),
    ),
    (
        ('DATE', ALMOST_DT_MAX, None, False),
        ('CLOSE', -1 * (FLOAT_MAX / 10.0), None, False),
        ('VOLUME', -1 * int(INTEGER_MAX / 10), None, False),
    ),
    (
        ('DATE', DT_MAX, ALMOST_DT_MAX, True),
        ('CLOSE', FLOAT_MIN * -0.1, FLOAT_MIN * -0.01, True),
        ('VOLUME', INTEGER_MAX * -1, (INTEGER_MAX * -1) - 1, True),
    ),
    (
        ('DATE', None, None, False),
        ('CLOSE', FLOAT_MAX * -1.0, (FLOAT_MAX * -1.0) - 1, True),
        ('VOLUME', None, None, False),
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

    features = infer_feature_attributes(df, tight_bounds=['DATE', 'CLOSE', 'VOLUME'])

    for feature in features.keys():
        if expected_unsupported.get(feature, False):
            assert features.has_unsupported_data(feature)
        else:
            assert not features.has_unsupported_data(feature)


@pytest.mark.parametrize("value, is_json, is_yaml", [
    ('{"key": "value", "_key": "_value"}', True, False),
    ('{"key":\n    {"key2": [1, 2, 3, 4]}\n}', True, False),
    ('[]', True, False),
    ('{}', True, False),
    ('a: 1\nb:\nc: 3\n\nd: 4', False, True),
    ("---\nname: The Howso Engine.\ndescription: >\n  The Howso Engine™ is a "
     "natively and fully explainable ML engine and toolbox.", False, True),
    ('not:valid:\nyaml\norjson', False, False),
    (12345, False, False),
    ('abcdefg', False, False),
    ('abcd\nefg', False, False),
    (None, False, False),
])
def test_json_yaml_features(value, is_json, is_yaml):
    """Test that infer_feature_attributes correctly identifies JSON and YAML features."""
    df = pd.read_csv(iris_path)
    df['sepal_width'] = value

    features = infer_feature_attributes(df)

    if is_json:
        assert features['sepal_width']['type'] == 'continuous'
        assert features['sepal_width']['data_type'] == 'json'
    elif is_yaml:
        assert features['sepal_width']['type'] == 'continuous'
        assert features['sepal_width']['data_type'] == 'yaml'
    else:
        assert features['sepal_width'].get('data_type') != 'json'
        assert features['sepal_width'].get('data_type') != 'yaml'


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
    features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_valid:
            features = infer_feature_attributes(data, types=types, )
            for feature_name, expected_type in expected_types.items():
                # Make sure it is the correct type
                assert features[feature_name]['type'] == expected_type
                # All features in this test, including nominals, should have bounds (at the very least: `allow_null`)
                assert 'allow_null' in features[feature_name].get('bounds', {}).keys()
        else:
            with pytest.raises(ValueError):
                infer_feature_attributes(data, types=types)


def test_preset_feature_types_with_multiprocessing():
    """Test that the `types` parameter behaves well with multiprocessing enabled."""
    df = pd.read_csv(stock_path)
    # Identified continuous features
    continuous = ['CLOSE']
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
    continuous = ['CLOSE']
    nominals = [f for f in df.columns if f not in continuous]
    features = infer_feature_attributes(df, types={"nominal": nominals, "continuous": continuous}, max_workers=10)
    assert same_order(features.keys(), df.columns)
    # Try it without multiprocessing as well
    features = infer_feature_attributes(df, types={"nominal": nominals, "continuous": continuous}, max_workers=0)
    assert same_order(features.keys(), df.columns)


def test_archival():
    """Test that archival of the FeatureAttributes instance works as expected."""
    data = pd.DataFrame({
        'a': [0, 1, 2, 3, 4, 5, 6, 7],
        'b': ['apple', 'banana', 'banana', 'cherry', 'apple', 'apple', 'cherry', 'banana']
    })
    features = infer_feature_attributes(data, tight_bounds=['a'])
    assert features['a']['type'] == 'continuous'
    assert features['b']['type'] == 'nominal'

    archive = features.to_json(archive=True)
    new_features = FeatureAttributesBase.from_json(archive)

    assert new_features['a']['type'] == 'continuous'
    assert new_features['b']['type'] == 'nominal'
    assert new_features.params['tight_bounds'] == ['a']


def test_disk_archival():
    """Test that archival of the FeatureAttributes instance to disk works as expected."""
    data = pd.DataFrame({
        'a': [0, 1, 2, 3, 4, 5, 6, 7],
        'b': ['apple', 'banana', 'banana', 'cherry', 'apple', 'apple', 'cherry', 'banana']
    })
    features = infer_feature_attributes(data, tight_bounds=['a'])
    assert features['a']['type'] == 'continuous'
    assert features['b']['type'] == 'nominal'

    with TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir, "fa_archive.json")

        features.to_json(archive=True, json_path=json_path)
        new_features = FeatureAttributesBase.from_json(json_path=json_path)

    assert new_features['a']['type'] == 'continuous'
    assert new_features['b']['type'] == 'nominal'
    assert new_features.params['tight_bounds'] == ['a']


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
    features = infer_feature_attributes(data, ordinal_feature_values={'a': ordinals})
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

    # When data_type is formatted_date_time, a date_time_format must be set
    with pytest.raises(
        ValueError,
        match='must have a `date_time_format` defined when its `data_type` is "formatted_date_time"'
    ):
        infer_feature_attributes(data, features={
            'custom': {
                "data_type": "formatted_date_time"
            }
        })

    # When data_type is formatted_time, a date_time_format must be set
    with pytest.raises(
        ValueError,
        match='must have a `date_time_format` defined when its `data_type` is "formatted_time"'
    ):
        infer_feature_attributes(data, features={
            'time': {
                "data_type": "formatted_time"
            }
        })

    # Verify formatted_date_time is set when a date_time_format is configured
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features = infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d"},
                                            default_time_zone="UTC")
        assert features['a']['data_type'] != "formatted_date_time"
        # custom feature dates should be formatted_date_time
        assert features['custom']['data_type'] == "formatted_date_time"
        # auto detected iso dates should be formatted_date_time
        assert features['iso']['data_type'] == "formatted_date_time"


def test_default_time_zone():
    """Test that ``infer_feature_attributes`` correctly handles default time zones."""
    data = pd.DataFrame({
        'custom': ['2010/10/10 7:30', '2010/10/11 8:45', '2010/10/12 9:00', '2010/10/14 12:00'],
        'custom2': ['2002/10/10 3:30', '2000/10/11 10:45', '2013/10/12 5:00', '2014/10/14 11:00'],
        'custom3': ['2010/10/10 07:30 -0500', '2010/10/11 08:45 -0500', '2010/10/12 09:00 -0500',
                    '2012/12/12 06:00 -0500'],
    })

    # No default time zone or time zone identifier in format string; warning should be raised
    with pytest.warns(match="features do not include a time zone and will default to UTC"):
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M",
                                                                 "custom2": "%Y/%m/%d %H:%M"})
        # Also try with multiprocessing
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M",
                                                                 "custom2": "%Y/%m/%d %H:%M"}, max_workers=2)

    # Using UTC offsets should also result in an error
    with pytest.warns(match="The following features are using UTC offsets"):
        infer_feature_attributes(data, datetime_feature_formats={"custom3": "%Y/%m/%d %H:%M %z"})
        # Also try with multiprocessing
        infer_feature_attributes(data, datetime_feature_formats={"custom3": "%Y/%m/%d %H:%M %z"}, max_workers=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Providing a default_time_zone should prevent the error
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M"}, default_time_zone="EST")
        data = pd.DataFrame({
            'custom': ['2010/10/10 07:30 EST', '2010/10/11 08:45 EST', '2010/10/12 09:00 EST'],
            'custom2': ['2010/10/10 07:30 GMT', '2010/10/11 08:45 GMT', '2010/10/12 09:00 GMT'],
        })
        # Providing data with a time zone and corresponding format string identifier should prevent the error
        infer_feature_attributes(data, datetime_feature_formats={"custom": "%Y/%m/%d %H:%M %Z",
                                                                 "custom2": "%Y/%m/%d %H:%M %Z"})


def test_constrained_date_bounds():
    """Constrained datetime formats may make bounds undeterminable."""
    df = pd.DataFrame([["01"], ["02"]], columns=["date"])
    with pytest.warns(match="bounds could not be computed. This is likely due to a constrained date time format"):
        # Loose bounds may cause min bound to be > max bound if the date format is constrained
        infer_feature_attributes(df, datetime_feature_formats={"date": "%m"})
