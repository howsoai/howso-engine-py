from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import as_completed, Future, ProcessPoolExecutor
import datetime
import decimal
import logging
from math import isnan, prod
import multiprocessing as mp
import typing as t
import warnings

from dateutil.parser import parse as dt_parse
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
from ..features import FeatureType
from ..utilities import (
    date_to_epoch,
    determine_iso_format,
    epoch_to_date,
    ISO_8601_DATE_FORMAT,
    ISO_8601_FORMAT,
    time_to_seconds,
)

logger = logging.getLogger(__name__)

SMALLEST_TIME_DELTA = 0.001


def _shard(data: pd.DataFrame, *, kwargs: dict[str, t.Any]):
    """Internal function to aid multiprocessing of feature attributes."""
    ifr_inst = InferFeatureAttributesDataFrame(data)
    # Filter out features that are not related to this shard.
    _kwargs = kwargs.copy()
    if "features" in _kwargs:
        _kwargs['features'] = {
            k: v for k, v in _kwargs["features"].items()
            if k in data.columns
        }

    feature_attributes = ifr_inst._process(**_kwargs)  # type: ignore reportPrivateUsage
    return feature_attributes, ifr_inst.unsupported


class InferFeatureAttributesDataFrame(InferFeatureAttributesBase):
    """Support inferring feature attributes for Pandas DataFrames."""

    def __init__(self, data: pd.DataFrame):  # type: ignore reportMissingSuperCall
        """
        Instantiate this InferFeatureAttributesDataFrame object.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the features whose attributes will be inferred.
        """
        self.data = data
        # Keep track of features that contain unsupported data
        self.unsupported = []

    def __call__(self, **kwargs) -> SingleTableFeatureAttributes:
        """Process and return feature attributes."""
        max_workers = kwargs.pop("max_workers", None)
        # The default with be to not use multiprocessing if the product of rows
        # and columns is less than 25M.
        if prod(self.data.shape) < 25_000_000 and max_workers is None:
            max_workers = 0

        if max_workers is None or max_workers >= 1:
            mp_context = mp.get_context("spawn")
            futures: dict[Future, str] = dict()

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
                for f in self.data.columns:
                    future = pool.submit(_shard, self.data[[f]], kwargs=kwargs)
                    futures[future] = f

                feature_attributes: dict[str, t.Any] = dict()
                unsupported: list[str] = list()
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        feature_attributes.update(response[0])
                        unsupported.extend(response[1])
                    except Exception as e:
                        warnings.warn(
                            f"Infer_feature_attributes raised an exception "
                            f"while processing '{futures[future]}' ({str(e)})."
                        )

            # Re-order the keys like the original dataframe
            feature_attributes = {
                k: feature_attributes[k] for k in self.data.columns
            }

            return SingleTableFeatureAttributes(
                feature_attributes=feature_attributes, params=kwargs,
                unsupported=unsupported
            )

        else:
            return SingleTableFeatureAttributes(
                self._process(**kwargs), params=kwargs,
                unsupported=self.unsupported
            )

    def _get_num_features(self) -> int:
        return self.data.shape[1]

    def _get_num_cases(self) -> int:
        return self.data.shape[0]

    def _get_feature_names(self) -> list[str]:
        feature_names = self.data.columns.tolist()
        for name in feature_names:
            if not isinstance(name, str):
                raise ValueError(
                    'Unexpected DataFrame column name format, please update '
                    'your DataFrame to use string column names.')
        return feature_names

    def _has_unique_constraint(self, feature_name: str) -> bool:
        # This always returns False for DataFrames, which don't support such
        # constraints.
        return False

    def _get_feature_type(self, feature_name: str  # noqa: C901
                          ) -> tuple[t.Optional[FeatureType], t.Optional[dict]]:
        # Import this here to avoid circular import
        from howso.client.exceptions import HowsoError
        feature = self.data[feature_name]
        if feature is not None:
            if isinstance(feature, pd.DataFrame):
                raise ValueError(
                    'The provided DataFrame contains duplicate column names. '
                    'All column names must be unique.')
            dtype = feature.dtype
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

            elif self._is_character_dtype(dtype):
                if getattr(dtype, 'kind', None) != 'U':
                    warnings.warn(f'The column "{feature_name}" contained '
                                  'bytes, original encoding of this column '
                                  'is not be guaranteed.')
                return FeatureType.STRING, {}

            else:
                first_non_null = self._get_first_non_null(feature_name)
                if isinstance(first_non_null, str):
                    # DataFrames may use 'object' dtype for strings, detect
                    # string columns by checking the type of the data
                    return FeatureType.STRING, {}
                elif isinstance(first_non_null, bytes):
                    warnings.warn(f'The column "{feature_name}" contained '
                                  'bytes, original encoding of this column '
                                  'is not be guaranteed.')
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
        else:
            return None, None

    @staticmethod
    def _is_character_dtype(dtype) -> bool:
        """Check if dtype is a subdtype of numpy.character."""
        try:
            # If numpy doesn't recognize the dtype, it will raise
            return np.issubdtype(dtype, np.character)
        except Exception:  # noqa: Deliberately broad
            return False

    def _get_first_non_null(self, feature_name: str) -> t.Any | None:
        index = self.data[feature_name].first_valid_index()
        if index is None:
            return None
        return self.data[feature_name][index]

    def _get_random_value(self, feature_name: str, no_nulls: bool = False) -> t.Any | None:
        """
        Return a random sample from the given DataFrame column.

        The return type is determined by the column type.

        if `no_nulls` is set, select a random value from the set of non-null
        values, if any. If there are no such non-nulls, this will return None.
        """
        cases = self.data[feature_name]
        if no_nulls:
            cases = cases.loc[~self.data[feature_name].isnull()]
        if len(cases) < 1:
            return None
        elif len(cases) == 1:
            return cases.iloc[0]
        else:
            return cases.iloc[1 + np.random.randint(len(cases) - 1)]

    def _infer_feature_bounds(  # noqa: C901
        self,
        feature_attributes: Mapping[str, Mapping],
        feature_name: str,
        tight_bounds: t.Optional[Iterable[str]] = None,
        mode_bound_features: t.Optional[Iterable[str]] = None,
    ) -> dict | None:
        output: dict[str, t.Any] = dict()
        allow_null = True
        column = self.data[feature_name]
        decimal_places = feature_attributes[feature_name].get('decimal_places')
        # only integers by default do not allow nulls
        if is_integer_dtype(column.dtype):
            allow_null = False

        def _as_float(value, dtype):
            # Convert datatype to float
            if is_timedelta64_dtype(dtype):
                if pd.isna(value):
                    return float('nan')
                return value.to_pytimedelta().total_seconds()
            elif isinstance(value, datetime.time):
                if pd.isna(value):  # type: ignore
                    return float('nan')
                return time_to_seconds(value)
            else:
                return float(value)

        if feature_attributes[feature_name].get('type') == 'continuous':
            column_filtered = self.data[feature_name].dropna()

            if (format_dt := feature_attributes[feature_name].get('date_time_format')) is not None:
                min_date, max_date = None, None
                min_date_tz, max_date_tz = None, None

                if is_datetime64_any_dtype(self.data[feature_name].dtype):
                    min_date = column_filtered.min()
                    max_date = column_filtered.max()
                else:
                    try:
                        max_date = pd.to_datetime(column_filtered, format=format_dt).max()
                        min_date = pd.to_datetime(column_filtered, format=format_dt).min()
                    except Exception as e:  # noqa: Deliberately broad
                        # This was likely due to Pandas not being able to handle mixed tz-offsets
                        # or possibly a non-matching datetime format, fall back to the (much)
                        # slower, but more thorough method.
                        warnings.warn(
                            f"Falling back to a more robust (albeit slower) check for feature "
                            f"bounds due to Pandas raising the following exception:\n\n{str(e)}."
                        )
                        try:
                            min_date_sec = np.inf
                            max_date_sec = -np.inf
                            # loop over all datetimes to determine the min and max
                            for datetime_val in column_filtered.values:
                                seconds = date_to_epoch(datetime_val, format_dt)
                                if seconds < min_date_sec:
                                    min_date_sec = seconds
                                    min_date = datetime_val
                                if seconds > max_date_sec:
                                    max_date_sec = seconds
                                    max_date = datetime_val

                            # If value is a string, parse it into datetime object
                            if isinstance(min_date, str):
                                min_date = dt_parse(min_date)
                            if isinstance(max_date, str):
                                max_date = dt_parse(max_date)
                        except Exception:  # noqa: Intentionally broad
                            warnings.warn(
                                f'Feature {feature_name} does not match the '
                                f'provided date time format, unable to guess '
                                f'bounds.'
                            )
                            return None

                if pd.isnull(min_date) or pd.isnull(max_date):
                    return None

                # Capture the timezone information, so it can be included
                # in the conversion back from epoch.
                if isinstance(min_date, (datetime.datetime, datetime.time)):
                    min_date_tz = min_date.tzinfo
                if isinstance(max_date, (datetime.datetime, datetime.time)):
                    max_date_tz = max_date.tzinfo

                try:
                    actual_min_f = min_f = date_to_epoch(min_date, format_dt)
                    actual_max_f = max_f = date_to_epoch(max_date, format_dt)
                    if (
                        tight_bounds is None
                        or feature_name not in tight_bounds
                    ):
                        # Calculate loose bounds
                        min_f, max_f = self.infer_loose_feature_bounds(min_f, max_f)
                        # Check for mode bounds
                        if (
                            mode_bound_features is None or
                            feature_name in mode_bound_features
                        ):
                            col_modes = column_filtered.mode()
                            if len(col_modes) == len(column_filtered):
                                # All values are unique
                                col_modes = pd.Series(dtype=column.dtype)
                        else:
                            col_modes = pd.Series(dtype=column.dtype)

                        if (
                            not col_modes.empty and
                            is_datetime64_any_dtype(column) and
                            is_datetime64_any_dtype(col_modes) and
                            col_modes.dt.tz is None and
                            column.dt.tz is not None
                        ):
                            # Due to a limitation of pandas in python 3.7:
                            # If the mode is returned as timezone unaware but
                            # the original column is timezone aware it has been
                            # returned as the UTC value instead. Convert the
                            # mode to timezone aware so we can correctly detect
                            # the number of instances of it in the original
                            # column.
                            col_modes = col_modes.dt.tz_localize(pytz.utc)
                            col_modes = col_modes.dt.tz_convert(column.dt.tz)

                        # If the mode for the feature is same as an original
                        # bound, set that appropriate bound to the mode value
                        # since in this case, it probably represents an
                        # application-specific min/max. Only applies if there
                        # are more than 3 instances of the value.
                        for mode_value in col_modes:
                            if self.data[column == mode_value].shape[0] > 3:
                                mode_f = date_to_epoch(mode_value, format_dt)
                                if actual_min_f == mode_f:
                                    min_f = actual_min_f
                                elif actual_max_f == mode_f:
                                    max_f = actual_max_f

                    min_date = epoch_to_date(min_f, format_dt, min_date_tz)
                    max_date = epoch_to_date(max_f, format_dt, max_date_tz)
                    return {'min': min_date, 'max': max_date}
                except Exception:  # noqa: Intentionally broad
                    w_str = (f'Feature {feature_name} does not match the '
                             'provided date time format, unable to guess '
                             'bounds.')
                    warnings.warn(w_str)
                    max_f = np.nan
                    min_f = np.nan

            else:
                min_f = _as_float(column_filtered.min(), column.dtype)
                max_f = _as_float(column_filtered.max(), column.dtype)

            if not (isnan(min_f) or isnan(max_f)):  # type: ignore
                actual_min_f = min_f
                actual_max_f = max_f
                # set loose bounds if no tight bounds for all and this
                # feature isn't on the tight bounds list
                if (
                    tight_bounds is None
                    or feature_name not in tight_bounds
                ):
                    min_f, max_f = self.infer_loose_feature_bounds(
                        actual_min_f, actual_max_f  # type: ignore
                    )
                    # Check for mode bounds
                    if (
                        mode_bound_features is None or
                        feature_name in mode_bound_features
                    ):
                        col_modes = column_filtered.mode()
                        if len(col_modes) == len(column_filtered):
                            # All values are unique
                            col_modes = []

                        # If the mode for the feature is same as an original
                        # bound, set that appropriate bound to the mode value
                        # since in this case, it probably represents an
                        # application-specific min/max. Only applies if there
                        # are more than 3 instances of the value.
                        for mode_value in col_modes:
                            if self.data[column == mode_value].shape[0] > 3:
                                mode_f = _as_float(mode_value, column.dtype)
                                if actual_min_f == mode_f:
                                    min_f = actual_min_f
                                elif actual_max_f == mode_f:
                                    max_f = actual_max_f

                output = {'min': min_f, 'max': max_f, 'allow_null': allow_null}
            else:
                # If no min/max were found from the data, use min/max size of
                # the data type.
                min_value, max_value = self._get_min_max_number_size_bounds(
                    feature_attributes, feature_name)
                if min_value is not None and max_value is not None:
                    output = {'min': min_value, 'max': max_value}

        else:  # Ordinals
            output = {'allow_null': allow_null}

        if decimal_places is not None:
            if 'max' in output:
                output['max'] = round(output['max'], decimal_places)
            if 'min' in output:
                output['min'] = round(output['min'], decimal_places)

        return output or None

    def _infer_floating_point_attributes(self, feature_name: str) -> dict:
        attributes: dict[str, t.Any] = {'type': 'continuous', 'data_type': 'number'}

        n_cases = self.data[feature_name].shape[0]

        # Ensure we have at least one valid value before attempting to
        # introspect further.
        if self.data[feature_name].isna().sum() < n_cases:

            # determine if nominal by checking if number of uniques <= 2
            if self.data[feature_name].nunique() <= 2 and n_cases > 10:
                return {
                    'type': 'nominal',
                    'data_type': 'number'
                }

            # Determine number of decimal places
            # If series > 1000 subset
            if len(self.data[feature_name]) > 1000:
                ind = np.random.randint(0, len(self.data[feature_name]), size=1000)
                col = self.data[feature_name].iloc[ind]

                # Remove Inf, -Inf values
                col = col.replace([np.inf, -np.inf], np.nan)

                # Check to see if / how many are NaNs
                nan_ratio = col.isna().sum() / len(col)

                # If NaN ratio > 0.8 then drop the NaN first and re-sample
                if nan_ratio > 0.8:
                    infinities = [np.inf, -np.inf]
                    col = (self.data[feature_name].replace(infinities, np.nan)
                           .dropna().reset_index(drop=True))
                    ind = np.random.randint(0, col.shape[0], size=1000)
                    col = col.iloc[ind]
                else:
                    col = col.dropna().reset_index(drop=True)

            # If series is less than 1000
            else:
                col = self.data[feature_name]
                col = (col.replace([np.inf, -np.inf], np.nan)
                          .dropna().reset_index(drop=True))

            # Determine number of decimal places using
            # np.format_float_positional to handle scientific notation.
            decimals = max([
                len((str(np.format_float_positional(r))).split('.')[1])
                for r in col
            ])

            # specify decimal place. Proceed with training but issue a warning.
            if pd.api.types.is_float_dtype(col.dtype):
                try:
                    if getattr(col.dtype, 'itemsize') <= 8:
                        attributes['decimal_places'] = decimals
                    else:
                        warnings.warn(
                            f'Feature {feature_name} contains floating point '
                            'values that exceed the maximum supported precision '
                            'of 64 bits.'
                        )
                except AttributeError:
                    warnings.warn(
                        f'Feature {feature_name} may contain floating point '
                        'values that exceed the maximum supported precision '
                        'of 64 bits.'
                    )

        return attributes

    def _infer_datetime_attributes(self, feature_name: str) -> dict:
        column = self.data[feature_name]
        dt_format = ISO_8601_FORMAT
        if hasattr(column, 'dt') and getattr(column.dt, 'tz', None):
            # Include timezone offset in format
            dt_format += '%z'
        elif column.dtype == object:
            first_non_null = self._get_first_non_null(feature_name)
            if isinstance(first_non_null, datetime.datetime):
                # In the event of mixed timezone values the column dtype will
                # be 'object'. In this case we check if the first non-null
                # value has tzinfo and include the timezone in the format
                if first_non_null.tzinfo is not None:
                    dt_format += '%z'
        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': dt_format,
        }

    def _infer_date_attributes(self, feature_name: str) -> dict:
        return {
            'type': 'continuous',
            'data_type': 'formatted_date_time',
            'date_time_format': ISO_8601_DATE_FORMAT,
        }

    def _infer_time_attributes(self, feature_name: str) -> dict:
        return {
            'type': 'continuous',
            'data_type': 'number',
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
        # Decide if categorical by checking number of uniques is fewer
        # than the square root of the total samples or if every value
        # has exactly the same length.
        num_uniques = self.data[feature_name].nunique()
        n_cases = int(self.data[feature_name].count())
        if num_uniques < pow(n_cases, 0.5):
            guess_nominals = True
        else:
            # Find the largest and smallest non-null values in column.
            try:
                col_min = int(self.data[feature_name].nsmallest(1).iloc[0])
                col_max = int(self.data[feature_name].nlargest(1).iloc[0])
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
        }
