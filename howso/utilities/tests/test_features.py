import datetime
import decimal
import json
import locale
import warnings


from howso.utilities import infer_feature_attributes
from howso.utilities.features import FeatureSerializer, FeatureType
from howso.utilities.internals import sanitize_for_json
from howso.utilities.utilities import LocaleOverride
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_string_dtype,
)
import pytest
import pytz

from . import has_locales


@pytest.mark.parametrize('data_format', ['pandas', 'numpy'])
@pytest.mark.parametrize('data, original_type', [
    ([datetime.datetime(2020, 10, 10, 10, 5,
                        tzinfo=pytz.timezone('US/Eastern'))],
     {'data_type': str(FeatureType.DATETIME), 'timezone': 'US/Eastern'}),
    ([datetime.datetime(2020, 10, 10, 10, 5, tzinfo=pytz.FixedOffset(-300))],
     {'data_type': str(FeatureType.DATETIME)}),
    ([datetime.datetime.fromisoformat("2022-01-01")],
     {'data_type': str(FeatureType.DATETIME)}),
    ([datetime.datetime.fromisoformat("2022-01-01T10:00:00-05:00")],
     {'data_type': str(FeatureType.DATETIME)}),
    ([datetime.datetime(2022, 1, 1, 10, 0, 0, tzinfo=pytz.timezone("EST"))],
     {'data_type': str(FeatureType.DATETIME), 'timezone': 'EST'}),
    ([datetime.date.fromisoformat("2022-01-01")],
     {'data_type': str(FeatureType.DATE)}),
    ([datetime.timedelta(days=1, seconds=60)],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([datetime.timedelta(microseconds=1000)],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(5, 'Y')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(3, 'M')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(50, 'W')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(10, 'D')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(91, 'D')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(10, 'h'), np.timedelta64(10, 'm')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(55, 'm')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(60, 's')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(1000, 'ms')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(1000, 'us')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([np.timedelta64(1000, 'ns')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}),
    ([100, 100000000000], {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([np.int8(125)], {'data_type': str(FeatureType.INTEGER), 'size': 1}),
    ([np.int16(1000)], {'data_type': str(FeatureType.INTEGER), 'size': 2}),
    ([np.int32(1000)], {'data_type': str(FeatureType.INTEGER), 'size': 4}),
    ([np.int64(1000)], {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([5, None], {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    ([5.5, None], {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    ([5.5, 1000000.0], {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    ([np.float16(1.5)], {'data_type': str(FeatureType.NUMERIC), 'size': 2}),
    ([np.float32(10.5)], {'data_type': str(FeatureType.NUMERIC), 'size': 4}),
    ([np.float64(100.5)], {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    ([decimal.Decimal('1.1')], {'data_type': str(FeatureType.NUMERIC),
                                'format': 'decimal'}),
    ([None, 'test'],
     {'data_type': str(FeatureType.STRING)}),
    ([True, False], {'data_type': str(FeatureType.BOOLEAN)}),
    ([None, None], {'data_type': str(FeatureType.UNKNOWN)})
])
def test_feature_deserialization(data_format, data, original_type):
    """
    Test case serialization.

    Tests serialization of data from DataFrame/Numpy to JSON then back again, results
    in original data types.
    """
    # No warnings should be raised
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter("error", append=True)
        columns = ['a']
        df = pd.DataFrame(pd.Series(data, name='a'))
        features = infer_feature_attributes(df)
        if data_format == "numpy":
            df_numpy = np.array(data)
            df_numpy = np.array([[value] for value in df_numpy])
            cases = FeatureSerializer.serialize(df_numpy, columns, features)
        else:
            cases = FeatureSerializer.serialize(df, columns, features)
        json_data = json.dumps(sanitize_for_json(cases))
        new_df = FeatureSerializer.deserialize(
            json.loads(json_data), columns, features)

        assert 'original_type' in features['a']
        assert features['a']['original_type'] == original_type
        difference = df.compare(new_df)
        assert difference.empty


@pytest.mark.parametrize('data, expected_dtype, original_type', [
    # Integers
    ([[1], [2]], 'int8',
     {'data_type': str(FeatureType.INTEGER), 'size': 1}),
    ([[1], [2]], 'uint8',
     {'data_type': str(FeatureType.INTEGER), 'size': 1, 'unsigned': True}),
    ([[1], [None], [2]], 'Int64',
     {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([[1], [float('nan')], [2]], 'Int16',
     {'data_type': str(FeatureType.INTEGER), 'size': 2}),
    ([[10.5], [None], [5.0]], 'Int64',
     {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([[10.5], [float('nan')], [5.0]], 'Int32',
     {'data_type': str(FeatureType.INTEGER), 'size': 4}),
    ([["1.2"], [None], ["3.3"]], 'Int64',
     {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([["1.2"], ["1"], ["3.3"]], 'int64',
     {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([["1.2"], [float('nan')], ["3.3"]], 'Int32',
     {'data_type': str(FeatureType.INTEGER), 'size': 4}),
    ([["1.2"], [float('nan')], ["3.3"]], 'UInt32',
     {'data_type': str(FeatureType.INTEGER), 'size': 4, 'unsigned': True}),
    ([["1"], [None], ["3"]], 'Int64',
     {'data_type': str(FeatureType.INTEGER), 'size': 8}),
    ([["1"], [float('nan')], ["3"]], 'Int32',
     {'data_type': str(FeatureType.INTEGER), 'size': 4}),
    # Floats
    ([[1], [2], [3]], 'float64',
     {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    ([[1.5], [float('nan')], [None], [float('inf')], [3.0]], 'float64',
     {'data_type': str(FeatureType.NUMERIC), 'size': 8}),
    ([["1"], ['nan'], ["3"]], 'float32',
     {'data_type': str(FeatureType.NUMERIC), 'size': 4}),
    ([["1.5"], [float('nan')], [float('inf')], ["3.3"]], 'float128',
     {'data_type': str(FeatureType.NUMERIC), 'size': 16}),
    ([["1.5"], ['NaN'], [None], ["3.3"]], 'object',
     {'data_type': str(FeatureType.NUMERIC), 'format': 'decimal'}),
])
def test_feature_deserialization_number_conversions(data, original_type,
                                                    expected_dtype):
    if expected_dtype in ['longdouble', 'float128', 'float256']:
        if not hasattr(np, expected_dtype):
            pytest.skip('Unsupported platform')

    # No warnings should be raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features = {'a': {'type': 'continuous', 'original_type': original_type}}
        df = FeatureSerializer.deserialize(data, features.keys(), features)
        assert not df.empty
        dtype = df['a'].dtype
        if 'size' in original_type:
            assert dtype.itemsize == original_type['size']
        assert dtype == expected_dtype


@pytest.mark.parametrize('data_format', ['pandas', 'numpy'])
@pytest.mark.parametrize('data, dt_format, valid_dtype, expected_data', [
    (
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=pytz.timezone('US/Eastern'))],
        '%Y-%m-%dT%H:%M:%S%z', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=pytz.timezone('US/Eastern'))],
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=pytz.timezone('US/Eastern'))],
        '%Y-%m-%dT%H:%M:%S', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=pytz.timezone('US/Eastern'))],
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=pytz.timezone('UTC'))],
        '%Y-%m-%dT%H:%M:%S', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=pytz.timezone('UTC'))],
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=pytz.FixedOffset(-300))],
        '%Y-%m-%dT%H:%M:%S%z', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=pytz.FixedOffset(-300))],
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=pytz.FixedOffset(-300))],
        '%Y-%m-%dT%I:%M:%S %p', lambda dtype: is_datetime64_dtype(dtype),
        [datetime.datetime(2020, 10, 25, 10, 5)],
    ),
    (
        [datetime.datetime.fromisoformat("2022-01-25")],
        '%d/%m/%Y %H:%M:%S', lambda dtype: is_datetime64_dtype(dtype),
        [datetime.datetime.fromisoformat("2022-01-25")],
    ),
    (
        [datetime.date.fromisoformat("2022-01-25")],
        '%Y/%m/%d', lambda dtype: is_datetime64_dtype(dtype),
        [datetime.date.fromisoformat("2022-01-25")],
    ),
])
def test_date_feature_serialization(
    data_format,
    data,
    dt_format,
    valid_dtype,
    expected_data
):
    # No warnings should be raised
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter("error", append=True)
        columns = ['a']
        df = pd.DataFrame(pd.Series(data, name='a'))
        features = infer_feature_attributes(df)
        features['a']['date_time_format'] = dt_format
        if data_format == "numpy":
            df_numpy = np.array(data)
            df_numpy = np.array([[value] for value in df_numpy])
            cases = FeatureSerializer.serialize(df_numpy, columns, features)
        else:
            cases = FeatureSerializer.serialize(df, columns, features)
        json_data = json.dumps(sanitize_for_json(cases))
        new_df = FeatureSerializer.deserialize(
            json.loads(json_data), columns, features)
        expected_df = pd.DataFrame(pd.Series(expected_data, name='a'))

        difference = expected_df.compare(new_df)
        assert valid_dtype(new_df['a'].dtype)
        assert difference.empty


@pytest.mark.skipif(
    not has_locales(['fr_FR.utf8', 'fr_FR.iso88591']),
    reason="Test locale: 'fr_FR' is not available")
@pytest.mark.parametrize('data_format', ['pandas', 'numpy'])
@pytest.mark.parametrize(
    'language_code, encoding, category, dt_format, valid_dtype, data', [
        (
            'fr_FR', 'iso8859-1', locale.LC_ALL, '%a %d %B %Y %X',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.datetime(2020, 11, 25, 10, 5,
                               tzinfo=pytz.timezone('Europe/Paris'))],
        ),
        (
            'fr_FR', 'iso8859-1', locale.LC_ALL, '%a %d %B %Y %X %z',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.datetime(2020, 11, 25, 10, 5,
                               tzinfo=pytz.timezone('Europe/Paris'))],
        ),
        (
            'fr_FR', 'utf-8', locale.LC_ALL, '%a %d %B %Y %X %z',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.datetime(2020, 11, 25, 10, 5,
                               tzinfo=pytz.timezone('Europe/Paris'))],
        ),
        (
            'fr_FR', None, locale.LC_ALL, '%a %d %B %Y %X',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.datetime(2020, 11, 25, 10, 5)],
        ),
        (
            'fr_FR', None, locale.LC_ALL, '%a %d %B %Y %X',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.date(2020, 11, 25)],
        ),
        (
            'fr_FR', 'utf-8', locale.LC_ALL, '%a %d %B %Y %X',
            lambda dtype: is_string_dtype(dtype),
            ["mer. 25 novembre 2020 01:00:00"],
        ),
    ]
)
def test_date_locale_serialization(
    data_format,
    data,
    dt_format,
    language_code,
    encoding,
    valid_dtype,
    category
):
    """Test datetimes using alternate locales."""
    # No warnings should be raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with LocaleOverride(language_code=language_code, encoding=encoding,
                            category=category):
            columns = ['a']
            df = pd.DataFrame(pd.Series(data, name='a'))
            features = infer_feature_attributes(df)
            features['a']['date_time_format'] = dt_format
            features['a']['locale'] = language_code
            if data_format == 'numpy':
                df_numpy = np.array(data)
                df_numpy = np.array([[value] for value in df_numpy])
                cases = FeatureSerializer.serialize(df_numpy, columns, features)
            else:
                cases = FeatureSerializer.serialize(df, columns, features)

        # Deserialize outside locale override to test locale handling of
        # feature deserialize
        json_data = json.dumps(sanitize_for_json(cases))
        new_df = FeatureSerializer.deserialize(
            json.loads(json_data), columns, features)

        difference = df.compare(new_df)
        assert valid_dtype(new_df['a'].dtype)
        assert difference.empty
