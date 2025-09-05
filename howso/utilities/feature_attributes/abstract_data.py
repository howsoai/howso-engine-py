from __future__ import annotations

from collections.abc import Iterable, Mapping
import datetime
from datetime import time, timedelta
import decimal
import inspect
import logging
from math import isnan
import re
import typing as t
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
)
import pytz

from .base import InferFeatureAttributesBase, SingleTableFeatureAttributes
from .protocols import IFACompatibleADCProtocol
from ..features import FeatureType
from ..utilities import (
    date_to_epoch,
    determine_iso_format,
    epoch_to_date,
    infer_time_feature_cycle_length,
    infer_time_format,
    ISO_8601_DATE_FORMAT,
    ISO_8601_FORMAT,
    SIMPLE_TIME_PATTERN,
    TIME_PATTERN,
    time_to_seconds,
)

logger = logging.getLogger(__name__)

# Attempt to import howso.connectors, but don't require it unless an IFA-ADC class is instantiated
# as it pulls in many potentially unused imports for most infer_feature_attributes use cases.
try:
    from howso.connectors import AbstractData
except ImportError:
    AbstractData = None


class InferFeatureAttributesAbstractData(InferFeatureAttributesBase):
    """Infer feature attributes from AbstractData classes."""

    def __init__(self, data: IFACompatibleADCProtocol):  # type: ignore reportMissingSuperCall
        """
        Instantiate this InferFeatureAttributesAbstractData object.

        Parameters
        ----------
        data : IFACompatibleADCProtocol
            The AbstractData class containing the features whose attributes will be inferred.
        """
        if not AbstractData:
            raise ImportError("The howso-engine-connectors package must installed to use "
                              "infer_feature_attributes with AbstractData classes.")
        self.data = data
        # Accessed in the IFA base class as 'columns' instead of 'headers'
        self.data.columns = self.data.headers
        # Keep track of features that contain unsupported data
        self.unsupported = []
        # Keep track of any features that are missing time zone information
        # If a `default_time_zone` is provided, this list should stay empty
        self.missing_tz_features = []
        # Keep track of any features that use UTC offsets, as these could lead
        # to unexpected results due to daylight savings time in some time zones
        self.utc_offset_features = []
        # Keep track of any features that we detected to be datetimes but were
        # not in ISO8601 format
        self.unknown_datetime_features = []

    def __call__(self, **kwargs) -> SingleTableFeatureAttributes:
        """Process and return feature attributes."""
        feature_attributes = self._process(**kwargs)
        self.emit_time_zone_warnings(self.missing_tz_features, self.utc_offset_features)
        self.emit_unknown_datetime_warnings(self.unknown_datetime_features)
        return SingleTableFeatureAttributes(
            feature_attributes, params=kwargs,
            unsupported=self.unsupported
        )

    def _is_primary_key(self, feature_name: str) -> bool:
        if self.data.primary_keys is not None:
            return feature_name in self.data.primary_keys
        return False

    def _is_foreign_key(self, feature_name: str) -> bool:
        if self.data.foreign_keys is not None:
            return feature_name in self.data.foreign_keys
        return False

    def _get_num_features(self) -> int:
        return len(self.data.headers)

    def _get_num_cases(self, feature_name: str) -> int:
        return self.data.get_num_cases(feature_name)

    def _get_feature_names(self) -> list[str]:
        return self.data.headers

    def _has_unique_constraint(self, feature_name: str) -> bool:
        """Return True if the given feature_name has a unique constraint."""
        if self._is_primary_key(feature_name):
            # All PKs have a natural unique constraint
            return True

        # If the ADC is SQL-based, we can attempt to see if it has a unique constraint without
        # being a primary key. However, if the ADC is not SQL-based, do not assume that a column
        # is unique unless it is explicitly labeled as a primary key. A false positive 'unique'
        # feature attribute could cause a data science disaster.
        inspector = None
        try:
            inspector = self.data.get_inspector()
            uniques = inspector.get_unique_constraints(self.data.table_name.table,
                                                       schema=self.data.table_name.schema)
        except (AttributeError, NotImplementedError):
            # Non-SQL ADCs will not have the get_inspector() method implemented
            if inspector is None:
                return False
            # MSSQL is not currently supported in sqlalchemy for get_unique_constraints;
            # however, get_indexes can also be used to identify UNIQUE constraints
            indexes = inspector.get_indexes(self.data.table_name.table,
                                            schema=self.data.table_name.schema)
            return any([c['column_names'] == [feature_name] and c['unique'] for c in indexes])

        return any([c['column_names'] == [feature_name] for c in uniques])

    def _get_feature_type(self, feature_name: str  # noqa: C901
                          ) -> tuple[FeatureType | None, dict | None]:
        # Import this here to avoid circular import
        from howso.client.exceptions import HowsoError

        dtype = self.data.get_column_dtype(feature_name)

        # Some ADCs will return the string representation of the dtype.
        # Convert to NumPy DType so that we can access attributes like 'itemsize.'
        if isinstance(dtype, str):
            try:
                dtype = np.dtype(dtype)
            except Exception:
                # If there is a problem, leave as-is
                pass
        try:
            if is_float_dtype(dtype):
                typing_info = {}
                if itemsize := getattr(dtype, 'itemsize', None):
                    if itemsize > 8:
                        raise HowsoError(
                            f'Unsupported data type "{dtype}" found for '
                            f'feature "{feature_name}", Howso does not '
                            'currently support numbers larger than 64-bit.')
                    typing_info['size'] = itemsize

                return FeatureType.NUMERIC, typing_info

            elif is_integer_dtype(dtype):
                typing_info = {}
                if itemsize := getattr(dtype, 'itemsize', None):
                    typing_info['size'] = itemsize
                if is_unsigned_integer_dtype(dtype):
                    typing_info['unsigned'] = True

                return FeatureType.INTEGER, typing_info

            elif is_datetime64_any_dtype(dtype):
                typing_info = {}
                if dtype in ['datetime64[Y]', 'datetime64[M]', 'datetime64[D]']:
                    return FeatureType.DATE, {}
                elif isinstance(dtype, pd.DatetimeTZDtype):
                    if isinstance(dtype.tz, pytz.BaseTzInfo) and dtype.tz.zone:
                        # If using a named time zone capture it, otherwise
                        # rely on the offset in the iso8601 format
                        typing_info['timezone'] = dtype.tz.zone
                return FeatureType.DATETIME, typing_info

            elif is_timedelta64_dtype(dtype):
                # All time deltas will be converted to seconds
                return FeatureType.TIMEDELTA, {'unit': 'seconds'}

            elif is_bool_dtype(dtype):
                return FeatureType.BOOLEAN, {}

            elif np.issubdtype(dtype, np.character):
                if getattr(dtype, 'kind', None) != 'U':
                    warnings.warn(
                        f'The column "{feature_name}" contained bytes, original '
                        'encoding of this column cannot be guaranteed.'
                    )
                return FeatureType.STRING, {}
        except ValueError:  # Some of the above checks may not play nice with all dtypes
            pass

        # Try to determine feature type by inspecting the data
        first_non_null = self._get_first_non_null(feature_name, strip=True)
        # DataFrames may use 'object' dtype for strings, detect
        # string columns by checking the type of the data
        if isinstance(first_non_null, str):
            # First, determine if the string resembles common time-only formats
            if re.match(TIME_PATTERN, first_non_null) or re.match(SIMPLE_TIME_PATTERN,
                                                                  first_non_null):
                return FeatureType.TIME, {}
            # explicitly declared formatted_date_time/time; don't try to guess
            if getattr(self, 'datetime_feature_formats', {}).get(feature_name) is not None:
                return FeatureType.STRING, {}  # Could be datetime or time-only; let base.py figure it out
            # Depending on the data source, datetimes/timedeltas could easily be strings.
            # See if the string can be converted to a Pandas datetime/timedelta.
            try:
                # If the feature looks like a date or datetime, but it's not in ISO8601 format,
                # handle it as a string to avoid ambiguity.
                converted_dtype = pd.to_datetime(pd.Series([first_non_null])).dtype
                converted_val = pd.to_datetime(first_non_null)
                if not self._is_iso8601_datetime_column(feature_name):
                    self.unknown_datetime_features.append(feature_name)
                    return FeatureType.STRING, {}
                # Unfortunately, Pandas does not differentiate between datetimes and "pure" dates.
                # If the below code executes, that means Pandas recognizes the value as a datetime,
                # but we now need to check if the 'time' component is zero. If so, we can cast to
                # a Numpy datetime64[D] dtype.
                #
                # However, we need to be careful with this -- if the user has a datetime feature of the format
                # '%y-%m-%d', for example, the `to_datetime()` conversion above will add an empty time component
                # as previously described. But, if the user has a datetime feature that *actually* has an empty
                # time component in the string -- for example, '%y-%m-%dT00:00:00', we must respect the original
                # format even if it is intended to be a date-only feature.
                if all([converted_val.time() == pd.Timestamp(0).time(),
                        converted_val.tz is None,
                        # Ensure there is no time component in the unconverted string
                        'T' not in first_non_null,
                        '00:00:00' not in first_non_null]):
                    converted_dtype = np.datetime64(converted_val, 'D').dtype
                typing_info = {}
                if converted_dtype in ['datetime64[Y]', 'datetime64[M]', 'datetime64[D]']:
                    return FeatureType.DATE, {}
                elif isinstance(converted_dtype, pd.DatetimeTZDtype):
                    if isinstance(converted_dtype.tz, pytz.BaseTzInfo) and converted_dtype.tz.zone:
                        # If using a named time zone capture it, otherwise
                        # rely on the offset in the iso8601 format
                        typing_info['timezone'] = converted_dtype.tz.zone
                return FeatureType.DATETIME, typing_info
            except Exception:
                return FeatureType.STRING, {}
        elif isinstance(first_non_null, bytes):
            warnings.warn(
                f'The column "{feature_name}" contained bytes, original '
                'encoding of this column cannot be guaranteed.'
            )
            return FeatureType.STRING, {}
        elif isinstance(first_non_null, datetime.datetime):
            return FeatureType.DATETIME, {}
        elif isinstance(first_non_null, datetime.date):
            return FeatureType.DATE, {}
        elif isinstance(first_non_null, datetime.time):
            return FeatureType.TIME, {}
        elif isinstance(first_non_null, decimal.Decimal):
            return FeatureType.NUMERIC, {'format': 'decimal'}
        # Feature is of generic object type
        return FeatureType.UNKNOWN, {}

    def _get_first_non_null(self, feature_name: str, strip=False) -> t.Any | None:
        """
        Get the first non-null value in the given column.

        Parameters
        ----------
        feature_name : str
            The name of the feature to get the first non-null value of.
        strip : bool, default False
            If True, considers values that are emtpy or whitespace-only as null.

        Returns
        -------
        The first non-null value for the provided feature, if it exists; else, returns None.
        """
        if not strip:
            return self.data.get_first_non_null(feature_name)
        for chunk in self.data.yield_chunk():
            if val := next((x for x in chunk[feature_name].dropna() if str(x).strip()), None):
                return val
        return None

    def _get_random_value(self, feature_name: str, no_nulls: bool = False) -> t.Any | None:
        """
        Return a random sample from the given column.

        The return type is determined by the column type.

        if `no_nulls` is set, select a random value from the set of non-null
        values, if any. If there are no such non-nulls, this will return None.
        """
        return self.data.get_random_value(feature_name, no_nulls=no_nulls)

    def _get_unique_count(self, feature_name: str | Iterable[str]) -> int:
        """Get the number of unique values in the provided column(s)."""
        return self.data.get_unique_count(feature_name)

    @classmethod
    def _value_to_number(cls, value: t.Any) -> t.Any:
        """Convert value to a number."""
        if pd.isna(value):
            return float('nan')
        elif isinstance(value, decimal.Decimal):
            return float(value)
        elif isinstance(value, timedelta):
            return value.total_seconds()
        elif isinstance(value, time):
            return time_to_seconds(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return value

    def _infer_feature_bounds(  # noqa: C901
        self,
        feature_attributes: Mapping[str, Mapping],
        feature_name: str,
        tight_bounds: t.Optional[Iterable[str]] = None,
        mode_bound_features: t.Optional[Iterable[str]] = None,
    ) -> dict | None:
        # prevent circular import
        output = dict()
        allow_null = True
        original_type = feature_attributes[feature_name]['original_type']
        decimal_places = feature_attributes[feature_name].get('decimal_places')

        # Only integers by default do no allow nulls.
        if original_type.get('data_type') == FeatureType.INTEGER.value:
            allow_null = False

        if feature_attributes[feature_name].get('type') == 'continuous':
            # Grab the natural feature_type and raw_feature_type
            format_dt = None

            # Compute time-only feature bounds
            if feature_attributes[feature_name].get('data_type') == 'formatted_time':
                time_format = feature_attributes[feature_name].get('date_time_format')
                if not time_format:
                    raise ValueError(f'Error computing bounds for {feature_name}: '
                                     f'`date_time_format` must be specified in attributes')
                if "datetime_format" in inspect.signature(self.data.get_min_max_values).parameters:
                    min_time, max_time = (
                        self.data.get_min_max_values(feature_name, datetime_format=time_format))
                else:
                    # howso-engine-connectors < 2.2.0
                    min_time, max_time = (
                        self.data.get_min_max_values(feature_name))
                # Fractional seconds must be normalized to six decimal places for use with Pandas
                # when the data is represented as a float
                if '%f' in time_format and original_type.get('data_type') == FeatureType.NUMERIC.value:
                    try:
                        min_time = f"{float(min_time):06.6f}"
                        max_time = f"{float(max_time):06.6f}"
                    except (TypeError, ValueError):
                        # Not a float, do nothing
                        pass
                # Min/max values from ADC are raw; convert to datetime.time
                if not isinstance(min_time, datetime.time):
                    min_time = pd.to_datetime(min_time, format=time_format, errors='coerce').time()
                if not isinstance(max_time, datetime.time):
                    max_time = pd.to_datetime(max_time, format=time_format, errors='coerce').time()
                if (
                    tight_bounds is None
                    or feature_name not in tight_bounds
                ):
                    # Loose bounds
                    if not feature_attributes[feature_name].get('cycle_length'):
                        raise ValueError(f'Error computing loose bounds for {feature_name}: '
                                         '`cycle_length` must be specified in attributes')
                    return {
                        'min': 0, 'max': feature_attributes[feature_name]['cycle_length'],
                        'observed_min': time_to_seconds(min_time), 'observed_max': time_to_seconds(max_time),
                        'allow_null': True
                    }
                else:
                    # Tight bounds
                    return {
                        'min': time_to_seconds(min_time), 'max': time_to_seconds(max_time),
                        'observed_min': time_to_seconds(min_time), 'observed_max': time_to_seconds(max_time),
                        'allow_null': True
                    }

            if 'date_time_format' in feature_attributes[feature_name]:
                format_dt = feature_attributes[feature_name].get('date_time_format')

                # Trust that the ADC can handle finding min/max datetimes
                if "datetime_format" in inspect.signature(self.data.get_min_max_values).parameters:
                    min_date_obj, max_date_obj = (
                        self.data.get_min_max_values(feature_name, datetime_format=format_dt))
                else:
                    # howso-engine-connectors < 2.2.0
                    min_date_obj, max_date_obj = (
                        self.data.get_min_max_values(feature_name))

                # Min/max values from ADC are raw; convert to datetime.time
                if not isinstance(min_date_obj, datetime.datetime):
                    min_date_obj = pd.to_datetime(min_date_obj, format=format_dt)
                if not isinstance(max_date_obj, datetime.datetime):
                    max_date_obj = pd.to_datetime(max_date_obj, format=format_dt)

                # Capture the timezone information, so it can be included
                # in the conversion back from epoch.
                min_date_tz = None
                max_date_tz = None
                if isinstance(min_date_obj, (datetime.datetime, datetime.time)):
                    min_date_tz = min_date_obj.tzinfo
                if isinstance(max_date_obj, (datetime.datetime, datetime.time)):
                    max_date_tz = max_date_obj.tzinfo

                # Convert the found date bounds to float seconds since Epoch
                min_v = date_to_epoch(min_date_obj, format_dt)
                max_v = date_to_epoch(max_date_obj, format_dt)

            else:
                min_v, max_v = (
                    self.data.get_min_max_values(feature_name))
                min_v = self._value_to_number(min_v)
                max_v = self._value_to_number(max_v)

            observed_min_value = min_v
            observed_max_value = max_v

            if (
                min_v is not None and max_v is not None and
                not isnan(min_v) and
                not isnan(max_v)
            ):
                if (
                    tight_bounds is None
                    or feature_name not in tight_bounds
                ):
                    min_v, max_v = (
                        self.infer_loose_feature_bounds(observed_min_value,
                                                        observed_max_value))

                    if (
                        mode_bound_features is None or
                        feature_name in mode_bound_features
                    ):
                        # If the mode for the feature is same as an original
                        # bound, set that appropriate bound to the mode value
                        # since in this case, it probably represents an
                        # application-specific min/max.
                        col_modes = self.data.get_mode(feature_name)

                        for mode_value, mode_count in col_modes:
                            if mode_count < 4:
                                # Only apply when the value has more than 3
                                # instances in the dataset
                                continue
                            if format_dt:
                                mode_f = date_to_epoch(mode_value, format_dt)
                            else:
                                mode_f = self._value_to_number(mode_value)
                            if observed_min_value == mode_f:
                                min_v = observed_min_value
                            if observed_max_value == mode_f:
                                max_v = observed_max_value
                # If this is a datetime feature, convert back from epoch time
                if format_dt is not None:
                    min_v = epoch_to_date(min_v, format_dt, min_date_tz)
                    max_v = epoch_to_date(max_v, format_dt, max_date_tz)
                    observed_min_value = epoch_to_date(observed_min_value, format_dt, min_date_tz)
                    observed_max_value = epoch_to_date(observed_max_value, format_dt, max_date_tz)
                    if date_to_epoch(min_v, format_dt) > date_to_epoch(max_v, format_dt):
                        warnings.warn(
                            f'Feature "{feature_name}" bounds could not be computed. '
                            'This is likely due to a constrained date time format.'
                        )
                        min_v, max_v = None, None

                if is_float_dtype(min_v):
                    min_v = float(min_v)
                if is_float_dtype(max_v):
                    max_v = float(max_v)
                if is_float_dtype(observed_min_value):
                    observed_min_value = float(observed_min_value)
                if is_float_dtype(observed_max_value):
                    observed_max_value = float(observed_max_value)

                if min_v and isinstance(min_v, str):
                    output.update(min=min_v)
                elif min_v is not None and not isnan(min_v):
                    output.update(min=min_v)

                if max_v and isinstance(max_v, str):
                    output.update(max=max_v)
                elif max_v is not None and not isnan(max_v):
                    output.update(max=max_v)

                if observed_min_value and isinstance(observed_min_value, str):
                    output.update(observed_min=observed_min_value)
                elif not isnan(observed_min_value):
                    output.update(observed_min=observed_min_value)

                if observed_max_value and isinstance(observed_max_value, str):
                    output.update(observed_max=observed_max_value)
                elif not isnan(observed_max_value):
                    output.update(observed_max=observed_max_value)
            else:
                # If no min/max were found from the data, use min/max size of
                # the data type.
                min_v, max_v = self._get_min_max_number_size_bounds(
                    feature_attributes, feature_name)
                if min_v is not None and max_v is not None:
                    output = {'min': min_v, 'max': max_v}

            output.update(allow_null=allow_null)

        else:  # Non-continuous
            output: dict = {'allow_null': allow_null}

            if (
                original_type.get('data_type') == FeatureType.INTEGER.value or
                original_type.get('data_type') == FeatureType.NUMERIC.value
            ):
                # Numeric types are assumed to be ranked in natural order.
                min_v, max_v = self.data.get_min_max_values(feature_name)
                output.update({"observed_min": min_v, "observed_max": max_v})

            else:  # Objects/strings
                # For string ordinals, we can only rank them if they are given
                # a rank via the `allowed` key in `bounds`.
                if allowed := feature_attributes[feature_name].get('bounds', {}).get('allowed'):
                    unique_values: set = self._get_unique_values(feature_name)
                    # Find the first value in allowed_values present in unique_values
                    observed_min = next((value for value in allowed if value in unique_values), None)
                    # Find the last value in allowed_values present in unique_values
                    observed_max = next((value for value in reversed(allowed) if value in unique_values), None)
                    output.update({'observed_min': observed_min, 'observed_max': observed_max})

        if decimal_places is not None:
            if 'max' in output:
                output['max'] = round(output['max'], decimal_places)
            if 'min' in output:
                output['min'] = round(output['min'], decimal_places)

        return output

    def _infer_floating_point_attributes(self, feature_name: str) -> dict:
        preset_feature_type = self.attributes.get(feature_name, {}).get('type')
        if preset_feature_type == 'nominal':
            return {
                'type': 'nominal',
                'data_type': 'number',
            }
        else:
            attributes: dict[str, t.Any] = {'type': 'continuous', 'data_type': 'number'}

        n_cases = self.data.get_num_cases(feature_name)
        n_nulls = self.data.get_null_count(feature_name)

        # Ensure we have at least one valid value before attempting to
        # introspect further.
        if n_nulls < n_cases:

            # determine if nominal by checking if number of uniques <= 2
            if (self.data.get_unique_count(feature_name) <= 2 and n_cases > 10
                    and preset_feature_type not in ('continuous', 'ordinal')):
                return {
                    'type': 'nominal',
                    'data_type': 'number'
                }

            decimals = self.data.get_decimal_places(feature_name)
            if decimals is None:
                warnings.warn(f'Cannot compute decimal places for feature "{feature_name}')
            else:
                attributes['decimal_places'] = decimals

        return attributes

    def _infer_datetime_attributes(self, feature_name: str) -> dict:
        # Although rare, it is plausible that a datetime field could be a
        # primary- or foreign-key.
        if (
                self._is_primary_key(feature_name) or
                self._is_foreign_key(feature_name)
        ):
            return {
                'type': 'nominal',
            }

        dtype = self.data.get_column_dtype(feature_name)
        dt_format = ISO_8601_FORMAT
        if not self._is_iso8601_datetime_column(feature_name):
            raise ValueError(f'Feature {feature_name} recognized as a datetime with non-ISO8601 format. Please '
                             'specify the format via `datetime_feature_formats`.')
        if isinstance(dtype, pd.DatetimeTZDtype):
            # Include timezone offset in format
            dt_format += '%z'
        elif dtype == 'object':
            first_non_null = self._get_first_non_null(feature_name)
            if isinstance(first_non_null, datetime.datetime):
                # In the event of mixed timezone values the column dtype will
                # be 'object'. In this case we check if the first non-null
                # value has tzinfo and include the timezone in the format
                if first_non_null.tzinfo is not None:
                    dt_format += '%z'
            elif self._is_iso8601_datetime_column(feature_name):
                # if datetime, determine the iso8601 format it's using
                if first_non_null := self._get_first_non_null(feature_name):
                    fmt = determine_iso_format(first_non_null, feature_name)
                    return {
                        'type': 'continuous',
                        'data_type': 'formatted_date_time',
                        'date_time_format': fmt
                    }
            # Try converting the string to datetime using Pandas to determine
            # if tz info is present.
            elif pd.to_datetime(first_non_null).tz is not None:
                dt_format += '%z'
        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': dt_format,
        }

    def _infer_date_attributes(self, feature_name: str) -> dict:
        if not self._is_iso8601_datetime_column(feature_name):
            raise ValueError(f'Feature {feature_name} recognized as a date with non-ISO8601 format. Please '
                             'specify the format via `datetime_feature_formats`.')
        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': ISO_8601_DATE_FORMAT,
        }

    def _infer_time_attributes(self, feature_name: str, user_time_format: str = None) -> dict:
        # Import this here to avoid circular import
        from howso.client.exceptions import HowsoError
        first_non_null = self._get_first_non_null(feature_name)
        # If the type is datetime.time
        if isinstance(first_non_null, datetime.time):
            if user_time_format:
                warnings.warn(
                    f'Feature "{feature_name}" is an instance of `datetime.time`, '
                    'the user-provided time format string will be ignored.'
                )
            # Format string representation of datetime.time types
            time_format = '%H:%M:%S'
        # If the type is a string
        elif user_time_format is not None:
            time_format = user_time_format
        else:
            try:
                time_format = infer_time_format(first_non_null)
            except ValueError as e:
                raise HowsoError(f'Please specify the format of feature "{feature_name}" in '
                                 '"datetime_feature_formats"') from e
        return {
            'type': 'continuous',
            'cycle_length': infer_time_feature_cycle_length(time_format),
            'data_type': 'formatted_time',
            'date_time_format': time_format,
        }

    def _infer_timedelta_attributes(self, feature_name: str) -> dict:
        return {
            'type': 'continuous',
            'data_type': 'number',
        }

    def _infer_boolean_attributes(self, feature_name: str) -> dict:
        return {
            'type': 'nominal',
            'data_type': 'boolean',
        }

    def _infer_integer_attributes(self, feature_name: str) -> dict:
        preset_feature_type = self.attributes.get(feature_name, {}).get('type')
        # Decide if categorical by checking number of uniques is fewer
        # than the square root of the total samples or if every value
        # has exactly the same length.
        num_uniques = self._get_unique_count(feature_name)
        if num_uniques < self._get_cont_threshold(feature_name) or preset_feature_type == 'nominal':
            guess_nominals = True
        else:
            # Find the largest and smallest non-null values in column.
            try:
                col_min, col_max = self.data.get_min_max_values(feature_name)
            except TypeError:
                # Column is all None?
                guess_nominals = False
            else:
                # Guess nominals if ALL of:
                #   - `col_min` and `col_max` are both greater than zero
                #   - Their length is at least 5
                #   - They have the same length
                guess_nominals = (
                    col_min > 0 and col_max > 0 and
                    len(str(col_min)) >= 5 and
                    len(str(col_min)) == len(str(col_max))
                )

        if guess_nominals:
            attributes = {
                'type': 'nominal',
                'data_type': 'number',
                'decimal_places': 0,
            }
        else:
            attributes = {
                'type': 'continuous',
                'data_type': 'number',
                'decimal_places': 0,
            }

        return attributes

    def _infer_string_attributes(self, feature_name: str) -> dict:
        # Column has arbitrary string values, first check if they
        # are ISO8601 datetimes.
        if self._is_iso8601_datetime_column(feature_name):
            # if datetime, determine the iso8601 format it's using
            if first_non_null := self._get_first_non_null(feature_name):
                fmt = determine_iso_format(first_non_null, feature_name)
                return {
                    'type': 'continuous',
                    'data_type': 'formatted_date_time',
                    'date_time_format': fmt
                }
            else:
                # It isn't clear how this method would be called on a feature
                # if it has no data, but just in case...
                return {
                    'type': 'continuous',
                }
        elif self._is_json_feature(feature_name):
            return {
                'type': 'continuous',
                'data_type': 'json'
            }
        elif self._is_yaml_feature(feature_name):
            return {
                'type': 'continuous',
                'data_type': 'yaml'
            }
        else:
            return self._infer_unknown_attributes(feature_name)

    def _infer_unknown_attributes(self, feature_name: str) -> dict:
        return {
            'type': 'nominal',
            'data_type': 'string',
        }

    def _get_unique_values(self, feature_name: str) -> set[t.Any]:
        """Return the set of unique values for the given feature."""
        return self.data.get_unique_values(feature_name)
