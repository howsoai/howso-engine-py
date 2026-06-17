import datetime
import decimal
import json
import locale
from pathlib import Path
import re
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_string_dtype,
)
import pytest

from howso.engine import Trainee
from howso.utilities import infer_feature_attributes
from howso.utilities.features import FeatureSerializer, FeatureType
from howso.utilities.internals import sanitize_for_json
from howso.utilities.utilities import LocaleOverride

from . import has_locales

cwd = Path(__file__).parent.parent.parent
monk_path = Path(cwd, 'utilities', 'tests', 'data', 'monk1.csv')


@pytest.mark.parametrize('data_format', ['pandas', 'numpy'])
@pytest.mark.parametrize('data, original_type, should_warn', [
    ([datetime.datetime(2020, 10, 10, 10, 5,
                        tzinfo=ZoneInfo('US/Eastern'))],
     {'data_type': str(FeatureType.DATETIME), 'timezone': 'US/Eastern'}, False),
    ([datetime.datetime(2020, 10, 10, 10, 5, tzinfo=datetime.timezone(datetime.timedelta(minutes=-300)))],
     {'data_type': str(FeatureType.DATETIME)}, True),
    ([datetime.datetime.fromisoformat("2022-01-01")],
     {'data_type': str(FeatureType.DATE)}, False),
    ([datetime.datetime.fromisoformat("2022-01-01T10:00:00-05:00")],
     {'data_type': str(FeatureType.DATETIME)}, True),
    ([datetime.datetime(2022, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("GMT"))],
     {'data_type': str(FeatureType.DATETIME), 'timezone': 'GMT'}, False),
    ([datetime.date.fromisoformat("2022-01-01")],
     {'data_type': str(FeatureType.DATE)}, False),
    ([datetime.timedelta(days=1, seconds=60)],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([datetime.timedelta(microseconds=1000)],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(5, 'Y')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(3, 'M')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(50, 'W')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(10, 'D')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(91, 'D')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(10, 'h'), np.timedelta64(10, 'm')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(55, 'm')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(60, 's')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(1000, 'ms')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(1000, 'us')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([np.timedelta64(1000, 'ns')],
     {'data_type': str(FeatureType.TIMEDELTA), 'unit': 'seconds'}, False),
    ([100, 100000000000], {'data_type': str(FeatureType.INTEGER), 'size': 8}, False),
    ([np.int8(125)], {'data_type': str(FeatureType.INTEGER), 'size': 1}, False),
    ([np.int16(1000)], {'data_type': str(FeatureType.INTEGER), 'size': 2}, False),
    ([np.int32(1000)], {'data_type': str(FeatureType.INTEGER), 'size': 4}, False),
    ([np.int64(1000)], {'data_type': str(FeatureType.INTEGER), 'size': 8}, False),
    ([5, None], {'data_type': str(FeatureType.NUMERIC), 'size': 8}, False),
    ([5.5, None], {'data_type': str(FeatureType.NUMERIC), 'size': 8}, False),
    ([5.5, 1000000.0], {'data_type': str(FeatureType.NUMERIC), 'size': 8}, False),
    ([np.float16(1.5)], {'data_type': str(FeatureType.NUMERIC), 'size': 2}, False),
    ([np.float32(10.5)], {'data_type': str(FeatureType.NUMERIC), 'size': 4}, False),
    ([np.float64(100.5)], {'data_type': str(FeatureType.NUMERIC), 'size': 8}, False),
    ([decimal.Decimal('1.1')], {'data_type': str(FeatureType.NUMERIC),
                                'format': 'decimal'}, False),
    ([None, 'test'],
     {'data_type': str(FeatureType.STRING)}, False),
    ([True, False], {'data_type': str(FeatureType.BOOLEAN)}, False),
    ([None, None], {'data_type': str(FeatureType.UNKNOWN)}, False),
    (["The quick brown fox jumped over the lazy dog."],
     {"data_type": FeatureType.TOKENIZABLE_STRING.value}, False),
])
def test_feature_deserialization(data_format, data, original_type, should_warn):
    """
    Test case serialization.

    Tests serialization of data from DataFrame/Numpy to JSON then back again, results
    in original data types.
    """
    action = "ignore" if should_warn else "error"
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action=action, append=True)
        columns = ['a']
        df = pd.DataFrame(pd.Series(data, name='a'))
        if original_type["data_type"] == FeatureType.TOKENIZABLE_STRING.value:
            # Tokenizable strings must have their type pre-set to "continuous"
            features = infer_feature_attributes(df, default_time_zone="UTC", types={"a": "continuous"})
        else:
            features = infer_feature_attributes(df, default_time_zone="UTC")
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
    """Test the deserialization of number conversions."""
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
@pytest.mark.parametrize('data, dt_format, valid_dtype, expected_data, default_tz, should_warn', [
    (
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=ZoneInfo('US/Eastern'))],
        '%Y-%m-%dT%H:%M:%S%z', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=ZoneInfo('US/Eastern'))], None, False,
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=ZoneInfo('US/Eastern'))],
        '%Y-%m-%dT%H:%M:%S', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5,
                           tzinfo=ZoneInfo('US/Eastern'))], None, False,
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=ZoneInfo('UTC'))],
        '%Y-%m-%dT%H:%M:%S', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=ZoneInfo('UTC'))], None, False,
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=datetime.timezone(datetime.timedelta(minutes=-300)))],
        '%Y-%m-%dT%H:%M:%S%z', lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=datetime.timezone(datetime.timedelta(minutes=-300)))], None, True,
    ),
    (
        [datetime.datetime(2020, 10, 25, 10, 5, tzinfo=datetime.timezone(datetime.timedelta(minutes=-300)))],
        '%Y-%m-%dT%I:%M:%S %p', lambda dtype: is_datetime64_dtype(dtype),
        [datetime.datetime(2020, 10, 25, 10, 5)], None, True,
    ),
    (
        [datetime.datetime.fromisoformat("2022-01-25")],
        '%d/%m/%Y %H:%M:%S', lambda dtype: is_datetime64_dtype(dtype),
        [datetime.datetime.fromisoformat("2022-01-25")], 'GMT', False,
    ),
    (
        [datetime.date.fromisoformat("2022-01-25")],
        '%Y/%m/%d', lambda dtype: is_datetime64_dtype(dtype),
        [datetime.date.fromisoformat("2022-01-25")], None, True,
    ),
])
def test_date_feature_serialization(
    data_format,
    data,
    dt_format,
    valid_dtype,
    expected_data,
    default_tz,
    should_warn,
):
    """Test the serialization of date features."""
    action = "ignore" if should_warn else "error"
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action=action, append=True)
        columns = ['a']
        df = pd.DataFrame(pd.Series(data, name='a'))
        features = infer_feature_attributes(df, default_time_zone=default_tz)
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
                               tzinfo=ZoneInfo('Europe/Paris'))],
        ),
        (
            'fr_FR', 'iso8859-1', locale.LC_ALL, '%a %d %B %Y %X %z',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.datetime(2020, 11, 25, 10, 5,
                               tzinfo=ZoneInfo('Europe/Paris'))],
        ),
        (
            'fr_FR', 'utf-8', locale.LC_ALL, '%a %d %B %Y %X %z',
            lambda dtype: isinstance(dtype, pd.DatetimeTZDtype),
            [datetime.datetime(2020, 11, 25, 10, 5,
                               tzinfo=ZoneInfo('Europe/Paris'))],
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


def test_boolean_features():
    """Test that binary nominal and boolean features result in a nearly equivalent output from HSE."""
    # Columns of all 0s and 1s will be nominal ints by default
    nominal_df = pd.read_csv(monk_path)
    boolean_df = nominal_df.copy()
    boolean_df["Has tie"] = boolean_df["Has tie"].astype("boolean")
    boolean_df["Is smiling"] = boolean_df["Is smiling"].astype("boolean")
    boolean_df["target"] = boolean_df["target"].astype("boolean")

    # Compute feature attributes
    nominal_attributes = infer_feature_attributes(nominal_df)
    boolean_attributes = infer_feature_attributes(boolean_df)

    # Ensure Pandas knows the boolean DataFrame has boolean types
    assert boolean_attributes["target"]["data_type"] == "boolean"

    # Train
    nominal_trainee = Trainee(features=nominal_attributes)
    nominal_trainee.train(nominal_df)

    boolean_trainee = Trainee(features=boolean_attributes)
    boolean_trainee.train(boolean_df)

    # Set action/context features
    action_features = ["target"]
    context_features = [col for col in nominal_df.columns if col not in action_features]

    # Set prediction stats/metrics
    prediction_stats = [
        "accuracy",
        "adjusted_smape",
        "smape",
        "mae",
        "mcc",
        "recall",
        "precision",
        "r2",
        "rmse",
        "spearman_coeff",
        "missing_value_accuracy",
    ]
    metrics = {
        "estimated_residual_lower_bound": True,
        "feature_full_residuals": True,
        "feature_robust_residuals": True,
        "feature_deviations": True,
        "feature_full_prediction_contributions": True,
        "feature_robust_prediction_contributions": True,
        "feature_full_accuracy_contributions": True,
        "feature_full_accuracy_contributions_permutation": True,
        "feature_robust_accuracy_contributions": True,
        "feature_robust_accuracy_contributions_permutation": True,
    }

    # Do a react for each data type
    nominal_react = nominal_trainee.react_aggregate(action_features=action_features,
                                                    feature_influences_action_feature="target",
                                                    context_features=context_features,
                                                    details={
                                                        "prediction_stats": True,
                                                        "selected_prediction_stats": prediction_stats,
                                                        **metrics,
                                                    })

    boolean_react = boolean_trainee.react_aggregate(action_features=action_features,
                                                    feature_influences_action_feature="target",
                                                    context_features=context_features,
                                                    details={
                                                        "prediction_stats": True,
                                                        "selected_prediction_stats": prediction_stats,
                                                        **metrics,
                                                    })

    # Compare metrics within a precision of 3 decimals
    for metric in nominal_react:
        pd.testing.assert_frame_equal(nominal_react[metric], boolean_react[metric], check_exact=False, rtol=3)


def test_feature_original_type_to_dtype():
    """Test deserialized feature dtypes given the original type including for derived time-series features."""
    df = pd.DataFrame(
        {
            # Define a time-series dataset for different combinations of feature type, data type and nullable
            "id": ["0", "0", "0", "1", "1", "1", "2", "2", "2", "3"],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # continuous int
            "b": [0, 1, 2, None, 4, 5, 6, 7, None, 9],  # continuous int (Nullable)
            "c": [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],  # continuous float
            "d": [0, 1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, float("nan"), 9],  # continuous float (NaN)
            "e": [0, 1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, None, 9],  # continuous float (Nullable)
            "f": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],  # nominal int
            "g": [None, None, None, None, 1, 1, None, 1, 1, 2],  # nominal int (Nullable)
            "h": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],  # nominal str
            "i": ["0", "1", None, "3", "4", "5", "6", None, "8", "9"],  # nominal str (Nullable)
            "j": [True, True, False, True, True, False, False, False, False, True],  # nominal bool
            "k": [True, True, False, True, None, False, False, False, False, True],  # nominal bool (Nullable)
            "l": ["A", "B", "C", "C", "A", "B", "C", "A", "B", "B"],  # ordinal str
            "m": ["A", "B", "C", None, "A", "B", "C", "A", "B", "B"],  # ordinal str (Nullable)
            "n": [1, 2, 3, 2, 3, 1, 3, 1, 2, 2],  # ordinal int
            "o": [1, 2, 3, 2, 3, 1, None, 1, 2, 2],  # ordinal int (Nullable)
        }
    )
    # Correct inferred dtypes
    df["b"] = df["b"].astype("Int64")
    df["g"] = df["g"].astype("Int64")
    df["o"] = df["o"].astype("Int64")

    # Define the expected attribute types per feature
    expected_attributes = {
        # feature type, pandas dtype, original_type.data_type
        "id": {"type": "nominal", "dtype": pd.StringDtype, "otype": "string"},
        "time": {"type": "continuous", "dtype": "int", "otype": "integer"},
        "a": {"type": "continuous", "dtype": "int", "otype": "integer"},
        "b": {"type": "continuous", "dtype": "Int64", "otype": "integer", "nullable": True},
        "c": {"type": "continuous", "dtype": "float", "otype": "numeric"},
        "d": {"type": "continuous", "dtype": "float", "otype": "numeric", "nullable": True},
        "e": {"type": "continuous", "dtype": "float", "otype": "numeric", "nullable": True},
        "f": {"type": "nominal", "dtype": "int", "otype": "integer"},
        "g": {"type": "nominal", "dtype": "Int64", "otype": "integer", "nullable": True},
        "h": {"type": "nominal", "dtype": pd.StringDtype, "otype": "string"},
        "i": {"type": "nominal", "dtype": pd.StringDtype, "otype": "string", "nullable": True},
        "j": {"type": "nominal", "dtype": "bool", "otype": "boolean"},
        "k": {"type": "nominal", "dtype": "object", "otype": "boolean", "nullable": True},  # nullable bool is object
        "l": {"type": "ordinal", "dtype": pd.StringDtype, "otype": "string"},
        "m": {"type": "ordinal", "dtype": pd.StringDtype, "otype": "string", "nullable": True},
        "n": {"type": "ordinal", "dtype": "int", "otype": "integer"},
        "o": {"type": "ordinal", "dtype": "Int64", "otype": "integer", "nullable": True},
        # Time series features
        ".reverse_series_index": {"type": "continuous", "dtype": "int", "otype": "integer"},
        ".series_index": {"type": "continuous", "dtype": "int", "otype": "integer"},
        ".synchronous_counter": {"type": "continuous", "dtype": "int", "otype": "integer"},
        ".synchronous_counter_lag_1": {"type": "continuous", "dtype": "Int64", "otype": "integer", "nullable": True},
        ".time_to_horizon": {"type": "continuous", "dtype": "float", "otype": "numeric"},
    }

    features = infer_feature_attributes(
        df,
        id_feature_name="id",
        time_feature_name="time",
        types={"nominal": ["f", "g"]},
        num_lags=2,
        ordinal_feature_values={
            "l": ["A", "B", "C"],
            "m": ["A", "B", "C"],
            "n": [1, 2, 3],
            "o": [1, 2, 3],
        },
    )
    trainee = Trainee("test", features=features, overwrite_existing=True)
    trainee.train(df)
    trainee.analyze()

    re_derived = re.compile(r"\.([^.]+)_(?:delta|rate|lag)_(\d+)")
    cases = trainee.get_cases()

    for name, attributes in trainee.features.items():
        if not name.startswith(".synchronous_counter") and (matches := re_derived.match(name)):
            # Define expected types for lag/delta/rate
            nullable = True
            if "_lag_" in name:
                # lags use parent type, except integer/bool should always be nullable
                expected = dict(expected_attributes[matches[1]])
                if expected["dtype"] == "int":
                    expected["dtype"] = "Int64"
                elif expected["dtype"] == "bool":
                    expected["dtype"] = "object"  # nullable bool is object
            else:
                # delta/rate should be numeric
                expected = {"type": "continuous", "dtype": "float", "otype": "numeric"}
        else:
            expected = dict(expected_attributes[name])
            nullable = expected.get("nullable")

        # Validate expected feature type
        assert attributes["type"] == expected["type"], f"{name} type: {attributes['type']} != {expected['type']}"

        # Validate expected original type
        assert "original_type" in attributes
        otype = attributes["original_type"]["data_type"]
        expected_otype = expected["otype"]
        assert otype == expected_otype, f"{name} original type: {otype} != {expected_otype}"

        # Validate expected dtypes given all cases or filtered cases
        expected_dtype = expected["dtype"]
        assert cases[name].dtype == expected_dtype, f"{name} dtype: {cases[name].dtype} != {expected_dtype}"

        filtered_cases = trainee.get_cases(features=[name], condition={name: None})
        if nullable:
            dtype = filtered_cases[name].dtype
            assert dtype == expected_dtype, f"{name} dtype: {dtype} != {expected_dtype}"
        else:
            assert len(filtered_cases) == 0, f"{name}: found unexpected null cases"
