from math import e

import numpy as np
import pandas as pd
import pytest

from howso.client.exceptions import DatetimeFormatWarning
from howso.utilities import infer_feature_attributes
from . import NOMINAL_SUBSTITUTION_AVAILABLE


class TestInferFeatureAttributes:
    """Encapsulate InferFeatureAttributes tests."""

    def test_datetime_iso_sunny_path(self):
        """Test that infer_feature_attributes infers dates correctly."""
        df = pd.DataFrame(
            data=np.asarray(
                [
                    ['a', 'b', 'c', 'd'],
                    [
                        None, '2020-10-12T10:10:10',
                        '2020-12-12T12:12:12',
                        '1920-10-11T11:11:11'
                    ],
                ]
            ).transpose(), columns=['nom', 'datetime'])
        feature_attribs = infer_feature_attributes(df, tight_bounds=['nom', 'datetime'])

        assert feature_attribs['nom']['type'] == 'nominal'
        assert feature_attribs['datetime']['type'] == 'continuous'
        assert feature_attribs['datetime']['date_time_format'] == '%Y-%m-%dT%H:%M:%S'

        # values pre-epoch work and bounds are set to values in the dataset
        assert feature_attribs['datetime']['bounds']['min'] == '1920-10-11T11:11:11'
        assert feature_attribs['datetime']['bounds']['max'] == '2020-12-12T12:12:12'

    def test_not_datetime_column(self):
        """Test that infer_feature_attributes infers dates correctly."""
        df = pd.DataFrame(data=np.asarray([
            ["1.2", "3.4", "5.0"],
            ['2020-10-12T10:10:10', 'okay not really a date', 'that first value is a fluke']
        ]).transpose(), columns=['nom', 'not_actually_datetime'])
        feature_attribs = infer_feature_attributes(df)

        assert feature_attribs['nom']['type'] == 'nominal'
        assert feature_attribs['not_actually_datetime']['type'] == 'nominal'

    def test_datetime_custom_format(self):
        """Test that infer_feature_attributes infers dates correctly."""
        df = pd.DataFrame(data=np.asarray([
            ['1', '2', '3', ],
            ['10.10.2020', '12.12.2020', '11.11.1920']
        ]).transpose(), columns=['nom', 'datetime'])

        custom_format = {'datetime': ('%m.%d.%Y', 'en_US')}

        feature_attribs = infer_feature_attributes(df, datetime_feature_formats=custom_format)

        assert feature_attribs['nom']['type'] == 'nominal'
        assert feature_attribs['datetime']['type'] == 'continuous'
        assert feature_attribs['datetime']['date_time_format'] == '%m.%d.%Y'
        assert feature_attribs['datetime']['locale'] == 'en_US'

        # loose bounds select dates beyond what's in the dataset
        assert feature_attribs['datetime']['bounds']['min'] == '12.07.1855'
        assert feature_attribs['datetime']['bounds']['max'] == '11.15.2085'

    def test_datetime_warns_custom_format(self):
        """Test that infer_feature_attributes infers dates correctly."""
        df = pd.DataFrame(data=np.asarray([
            ['10.10.2020', '12.12.2020', '11.11.1920']
        ]).transpose(), columns=['datetime'])
        # convert column to contain datetime objects instead of strings
        df['datetime'] = pd.to_datetime(df['datetime'])

        custom_format = {'datetime': ('%m.%d.%Y', 'en_US')}

        expected_warning = (
            'Providing a datetime feature format for the feature "datetime" '
            'is not necessary because the data is already formatted as a '
            'datetime object. This custom format will be ignored.')

        with pytest.warns(Warning, match=expected_warning):
            feature_attribs = infer_feature_attributes(df, datetime_feature_formats=custom_format)

        assert feature_attribs['datetime']['type'] == 'continuous'
        assert 'locale' not in feature_attribs['datetime']
        assert feature_attribs['datetime']['date_time_format'] == (
            '%Y-%m-%dT%H:%M:%S')

    def test_datetime_iso_zoned_sunny_path(self):
        """Test that infer_feature_attributes infers dates correctly."""
        df = pd.DataFrame(data=np.asarray([
            ['2020-10-12T10:12:00Z', '2020-10-12T10:10:10Z', '2020-12-12T12:12:12Z', '1920-10-11T11:11:11Z'],
            [None, '2020-10-12T10:10:10+0000', '2020-12-12T12:12:12+0100', '1920-10-11T11:11:11-0300']
        ]).transpose(), columns=['datetime', 'datetimez'])
        feature_attribs = infer_feature_attributes(df, tight_bounds=['datetimez', 'datetime'])

        assert feature_attribs['datetime']['type'] == 'continuous'
        assert feature_attribs['datetime']['date_time_format'] == '%Y-%m-%dT%H:%M:%SZ'
        assert feature_attribs['datetimez']['type'] == 'continuous'
        assert feature_attribs['datetimez']['date_time_format'] == '%Y-%m-%dT%H:%M:%S%z'

        # values pre-epoch work and bounds are set to values in the dataset
        assert feature_attribs['datetime']['bounds']['min'] == '1920-10-11T11:11:11Z'
        assert feature_attribs['datetime']['bounds']['max'] == '2020-12-12T12:12:12Z'
        assert feature_attribs['datetimez']['bounds']['min'] == '1920-10-11T11:11:11-0300'
        assert feature_attribs['datetimez']['bounds']['max'] == '2020-12-12T12:12:12+0100'

    def test_determine_iso_format_warn(self):
        """Test that infer_feature_attributes infers dates correctly."""
        df = pd.DataFrame(data=np.asarray([
            ['a', 'b', 'c'],
            ['2010-10', '2012-01', '2011-01']
        ]).transpose(), columns=['nom', 'datetime'])

        # dates are a subset ISO format, show different warning
        with pytest.warns(DatetimeFormatWarning) as warn:
            feature_attribs = infer_feature_attributes(df)
            assert str(warn[0].message) == (
                'Feature "datetime" is a datetime but may not work properly if '
                "user does not specify the correct format.")
            assert feature_attribs['datetime']['type'] == 'continuous'
            assert feature_attribs['datetime']['date_time_format'] == '%Y-%m-%dT%H:%M:%S'

    @pytest.mark.skipif(not NOMINAL_SUBSTITUTION_AVAILABLE, reason="Nominal Substitution not supported")
    def test_infer_extended_nominal(self):
        """
        Test that `infer_feature_attributes` infers extended nominals correcly.

        ... to make sure warnings and other behavior is as per specification.
        """
        data = {
            "time": [1, 2, 3, 4],
            "gender": ["male", "male", "female", "female"],
            "ssn": ["1", "2", "3", "4"],
            "balance": [1.0, 2.0, 3.0, 4.0]
        }
        df = pd.DataFrame(data)

        # 1. Sanity check (non-timeseries)
        features = infer_feature_attributes(df)

        # No subtype is supposed to be set by default
        assert not features["gender"].get("subtype", None)
        assert not features["ssn"].get("subtype", None)

        # 2. Sanity check (timeseries)
        features = infer_feature_attributes(
            df, time_feature_name="time"
        )

        # No subtype is supposed to be set by default
        assert not features["gender"].get("subtype", None)
        assert not features["ssn"].get("subtype", None)
        assert features["time"]["time_series"]["delta_max"] == [e]
        assert features["time"]["time_series"]["delta_min"] == [1 / e]
        assert features["balance"]["time_series"]["rate_max"] == [e]
        assert features["balance"]["time_series"]["rate_min"] == [1 / e]

        # 3. Enable `attempt_infer_extended_nominals` parameter
        features = infer_feature_attributes(df, attempt_infer_extended_nominals=True)

        # 3a. Make sure it recognizes "gender" subtype
        assert features["gender"].get("subtype", None) == "gender"
        # 3b. Make sure it defaults to "int-id" subtype
        assert features["ssn"].get("subtype", None) == "int-id"

        # 4. Enable `attempt_infer_extended_nominals` parameter
        features = infer_feature_attributes(df, time_feature_name="time", attempt_infer_extended_nominals=True)

        # 4a. Make sure it recognizes "gender" subtype
        assert features["gender"].get("subtype", None) == "gender"
        # 4b. Make sure it defaults to "int-id" subtype
        assert features["ssn"].get("subtype", None) == "int-id"
        # 4c. Make sure "time" feature is continuous
        assert features["time"].get("type", None) == "continuous"
        # 4d. Make sure "time" has no subtypes
        assert not features["time"].get("subtype", None)

    def test_infer_time_series(self):
        """
        Test test_infer_time_series.

        Test the `infer_feature_attributes` for time series to make sure
        warnings and other behavior is as per specification.
        """
        data = {
            "ID": ["a", "a", "a", "a", "a", "a"],
            "time": [1, 2, 3, 4, 5, 6],
            "gender": ["male", "male", "male", "female", "female", "female"],
            "value": [11, 12, 15, 18, 23, 31],
            "balance": [1.0, 2.0, 3.0, 4.0, 4.9, 5.8],
            "bal_scaled": [10, 20, 30, 40, 49, 58]
        }
        df = pd.DataFrame(data)

        features = infer_feature_attributes(df, time_feature_name="time", id_feature_name="ID")

        # 1. Verify time series types
        assert features["time"]["time_series"]["type"] == 'delta'
        assert features["time"]["time_series"]["time_feature"] is True
        assert "time_series" not in features["ID"]

        # 1a. Make sure time series type is 'rate'
        assert features["gender"]["time_series"]["type"] == 'rate'
        assert features["value"]["time_series"]["type"] == 'rate'
        assert features["balance"]["time_series"]["type"] == 'rate'
        assert features["bal_scaled"]["time_series"]["type"] == 'rate'

        # 1b. Make sure rate min/max are stored, but not delta min/max
        assert "rate_max" in features["bal_scaled"]["time_series"]
        assert "rate_min" in features["bal_scaled"]["time_series"]
        assert "delta_max" not in features["bal_scaled"]["time_series"]
        assert "delta_min" not in features["bal_scaled"]["time_series"]

        features = infer_feature_attributes(
            df,
            time_feature_name="time",
            id_feature_name="ID",
            time_series_types_override={
                "balance": "delta",
                "bal_scaled": "delta"
            }
        )

        # 2. Verify custom-specified delta features are correctly set
        assert features["time"]["time_series"]["type"] == 'delta'
        assert features["gender"]["time_series"]["type"] == 'rate'
        assert features["value"]["time_series"]["type"] == 'rate'
        assert features["balance"]["time_series"]["type"] == 'delta'
        assert features["bal_scaled"]["time_series"]["type"] == 'delta'

        # 2a. Make sure for delta feature, delta min/max are stored instead
        assert "rate_max" not in features["bal_scaled"]["time_series"]
        assert "rate_min" not in features["bal_scaled"]["time_series"]
        assert "delta_max" in features["bal_scaled"]["time_series"]
        assert "delta_min" in features["bal_scaled"]["time_series"]

        features = infer_feature_attributes(
            df,
            time_feature_name="time",
            id_feature_name="ID",
            time_series_type_default='delta'
        )

        # 3. Verify all features set as 'delta'
        assert features["time"]["time_series"]["type"] == 'delta'
        assert features["gender"]["time_series"]["type"] == 'delta'
        assert features["value"]["time_series"]["type"] == 'delta'
        assert features["balance"]["time_series"]["type"] == 'delta'
        assert features["bal_scaled"]["time_series"]["type"] == 'delta'

        # 3a. Make sure delta min/max are stored instead of rate min/max
        assert "rate_max" not in features["value"]["time_series"]
        assert "rate_min" not in features["value"]["time_series"]
        assert "delta_max" in features["value"]["time_series"]
        assert "delta_min" in features["value"]["time_series"]

        features = infer_feature_attributes(
            df,
            time_feature_name="time",
            id_feature_name="ID",
            time_invariant_features=["gender", "balance"]
        )

        # 4. Verify time invariant features are correct, others use defaults
        assert "time_series" not in features["gender"]
        assert "time_series" not in features["balance"]
        assert features["value"]["time_series"]["type"] == 'rate'
        assert features["bal_scaled"]["time_series"]["type"] == 'rate'

        features = infer_feature_attributes(
            df,
            time_feature_name="time",
            id_feature_name="ID",
            time_series_types_override={
                "bal_scaled": "delta"
            },
            orders_of_derivatives={
                "value": 3,
                "bal_scaled": 2
            },
            derived_orders={"value": 3}
        )

        # 5. Verify high order rates and deltas are correctly computed
        assert features["value"]["time_series"]["order"] == 3
        assert features["value"]["time_series"]["rate_min"] == [1 / e, 0.0, -2 * e]
        assert features["value"]["time_series"]["rate_max"] == [8 * e, 3 * e, 2 * e]

        assert features["bal_scaled"]["time_series"]["order"] == 2
        assert features["bal_scaled"]["time_series"]["delta_min"] == [9 / e, -e]
        assert features["bal_scaled"]["time_series"]["delta_max"] == [10 * e, 0.0]

        # 6. Verify derived_orders is clamped down to 2
        assert features["value"]["time_series"]["derived_orders"] == 2
