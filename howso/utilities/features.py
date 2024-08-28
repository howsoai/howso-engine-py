from __future__ import annotations

from collections.abc import Iterable, Mapping
import decimal
from enum import Enum
from functools import partial
import locale
import typing as t
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
import pytz

from .internals import (
    deserialize_to_dataframe,
    IgnoreWarnings,
    to_pandas_datetime_format
)
from .utilities import (
    DATETIME_TIMEZONE_PATTERN,
    LocaleOverride,
    seconds_to_time,
    serialize_datetimes,
)


__all__ = [
    'FeatureSerializer',
    'FeatureType',
    'deserialize_cases',
    'format_dataframe',
    'serialize_cases',
]


class FeatureType(Enum):
    """Feature type enum."""

    UNKNOWN = 'object'
    STRING = 'string'
    NUMERIC = 'numeric'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    DATETIME = 'datetime'
    DATE = 'date'
    TIME = 'time'
    TIMEDELTA = 'timedelta'

    def __str__(self):
        """Return a string representation."""
        return str(self.value)


class FeatureSerializer:
    """Adapter for serialization and deserialization of feature data."""

    @classmethod
    def serialize(  # noqa: C901
        cls,
        data: t.Optional[pd.DataFrame | np.ndarray | Iterable[t.Any]],
        columns: Iterable[str] | None,
        features: Mapping,
        *,
        warn: bool = False
    ) -> list[list[t.Any]] | None:
        """
        Serialize case data into list of lists.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray or Iterable, default None
            The data to serialize, typically in a Pandas DataFrame, Numpy ndarray or Python Iterable such as a list.
        columns : Iterable of str, default None
            The case column mapping. The order corresponds to the order of cases in output.
            `columns` must be provided for non-DataFrame Iterables.
        features : Mapping
            The dictionary of feature name to feature attributes.
        warn : bool, default False
            If warnings should be raised by serializer.

        Returns
        -------
        list of list or Any or None
            The serialized data from DataFrame.

        ...

        Raises
        ------
        HowsoError
            An `pd.ndarray` or `Iterable` is provided, `columns` was left undefined 
            or the given columns does not match the columns defined within a given `pd.DataFrame`.
        ValueError
            The provided `pd.DataFrame` contains non-unique columns or, an unexpected datatype 
            was received (should be either pd.DataFrame, np.ndarray or Python Iterable (non-str)).
        """
        # Import locally to prevent a circular import
        from howso.client.exceptions import HowsoError

        if data is None:
            return None
        if isinstance(data, np.ndarray):
            if columns is None:
                raise HowsoError(
                    "Columns must be provided with numpy arrays."
                )
            data = pd.DataFrame.from_records(data, columns=columns)

        if isinstance(data, pd.DataFrame):
            if columns is not None:
                columns = list(columns)
                try:
                    filtered_data = data[columns]
                except KeyError as key_error:
                    raise HowsoError(
                        "The provided DataFrame is missing one or more of the "
                        f"expected named columns: {columns}."
                    ) from key_error
            else:
                columns = data.columns.tolist()
                filtered_data = data

            data_columns = []
            for col in columns:
                col_data = filtered_data[col]
                if isinstance(col_data, pd.DataFrame):
                    raise ValueError(
                        "The provided DataFrame contains duplicate column "
                        "names. All column names must be unique.")
                dtype = col_data.dtype
                if is_datetime64_any_dtype(dtype):
                    # `to_pydatetime` emits a FutureWarning even when the issue
                    # is fixed.
                    with IgnoreWarnings(FutureWarning):
                        # Make sure datetimes are returned as python datetimes
                        data_columns.append(np.array(col_data.dt.to_pydatetime()))
                elif is_timedelta64_dtype(dtype):
                    # Make sure timedeltas are returned as python timedeltas
                    data_columns.append(col_data.dt.to_pytimedelta())
                else:
                    data_columns.append(col_data.to_numpy())
            result = np.array(data_columns).T.tolist()

        elif isinstance(data, Iterable) and not isinstance(data, str):
            if columns is None:
                raise HowsoError(
                    "Columns must be provided with python Iterables."
                )
            result = list(data)
        else:
            raise ValueError(
                "Received unexpected data type for case data. Cases "
                "should be either a DataFrame or a list.")

        # Convert 1d list (single case) to 2d list for serialization
        if len(result) > 0 and len(np.array(result).shape) == 1:
            result = [result]

        # Serialize datetime objects
        serialize_datetimes(result, columns, features, warn=warn)

        return result

    @classmethod
    def deserialize(
        cls,
        data: Iterable[Iterable[t.Any] | Mapping[str, t.Any]],
        columns: Iterable[str],
        features: t.Optional[Mapping] = None
    ) -> pd.DataFrame:
        """
        Deserialize case data into a DataFrame.

        If feature attributes contain original typing information, columns
        will be converted to the same data type as original training cases.

        Parameters
        ----------
        data : list of list or list of dict
            The context data.
        columns : Iterable of str
            The case column mapping. The order corresponds to the order of cases in output.
            `columns` must be provided for non-DataFrame Iterables.

            The order corresponds to how the data will be mapped to columns in
            the output. Ignored for list of dict where the dict key is the column
            name.
        features : Mapping, default None
            (Optional) The dictionary of feature name to feature attributes.

            If not specified, no column typing will be attempted.

        Returns
        -------
        pandas.DataFrame
            The deserialized data.
        """
        df = deserialize_to_dataframe(data, columns)
        if features is not None:
            cls.format_dataframe(df, features)
        return df

    @classmethod
    def format_dataframe(cls, df: pd.DataFrame, features: Mapping
                         ) -> pd.DataFrame:
        """
        Format DataFrame columns to original type using feature attributes.

        .. note::

            Modifies DataFrame in place.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to format columns of.
        features : Mapping
            The dictionary of feature name to feature attributes.

        Returns
        -------
        pandas.DataFrame
            The formatted data.
        """
        for col in df.columns.tolist():
            try:
                attributes = features[col]
            except (TypeError, KeyError):
                # Column not in feature attributes, skip column
                continue
            df[col] = cls.format_column(df[col], attributes)
        return df

    @classmethod
    def format_column(cls, column: pd.Series,  # noqa: C901
                      feature: Mapping) -> pd.Series:
        """
        Format column based on feature typing information.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        feature = {} if feature is None else feature
        typing_info = cls._get_typing_info(feature)
        data_type = typing_info.get('data_type')

        if data_type == FeatureType.NUMERIC.value:
            return cls.format_numeric_column(column, feature)
        elif data_type == FeatureType.INTEGER.value:
            return cls.format_integer_column(column, feature)
        elif data_type == FeatureType.STRING.value:
            return cls.format_string_column(column, feature)
        elif data_type == FeatureType.DATETIME.value:
            return cls.format_datetime_column(column, feature)
        elif data_type == FeatureType.DATE.value:
            return cls.format_date_column(column, feature)
        elif data_type == FeatureType.TIME.value:
            return cls.format_time_column(column, feature)
        elif data_type == FeatureType.TIMEDELTA.value:
            return cls.format_timedelta_column(column, feature)
        elif data_type == FeatureType.BOOLEAN.value:
            return cls.format_boolean_column(column, feature)
        else:
            return cls.format_unknown_column(column, feature)

    @classmethod
    def format_timedelta_column(cls, column: pd.Series, feature: Mapping
                                ) -> pd.Series:
        """
        Format timedelta column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)
        try:
            unit = typing_info['unit']
            if unit not in ['days', 'seconds', 'nanoseconds']:
                raise ValueError('invalid unit type')
        except (TypeError, KeyError, ValueError):
            warnings.warn(
                f'Unknown timedelta unit for column "{column.name}", '
                'column will not be translated to a timedelta64.')
            return column

        try:
            return pd.to_timedelta(column, unit=unit)
        except Exception:  # noqa: Deliberately broad
            warnings.warn(
                f'Timedelta column "{column.name}" failed to be parsed as '
                'timedelta64 values, column will retain current dtype.'
            )
        return column

    @classmethod
    def format_datetime_column(cls, column: pd.Series, feature: Mapping  # noqa: C901
                               ) -> pd.Series:
        """
        Format datetime column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)
        date_time_format = feature.get('date_time_format') or None
        date_locale = feature.get('locale') or None
        format_includes_timezone = False
        if (
            date_time_format and
            DATETIME_TIMEZONE_PATTERN.findall(date_time_format)
        ):
            format_includes_timezone = True

        # Load timezone information if provided
        tz = typing_info.get('timezone')
        if tz is not None:
            try:
                tz = pytz.timezone(tz)
            except pytz.UnknownTimeZoneError:
                warnings.warn(
                    f'Unknown timezone "{tz}" for feature "{column.name}", '
                    'datetime column may not contain original timezone '
                    'information.')
                tz = None

        def _to_datetime():
            # Format column to datetime
            target_format = to_pandas_datetime_format(date_time_format)
            if tz is not None:
                if format_includes_timezone:
                    # When format string includes timezone we want to load
                    # the datetime as UTC, then we can convert it to the named
                    # timezone
                    col = pd.to_datetime(column, format=target_format, utc=True)
                    return col.dt.tz_convert(tz)
                else:
                    # Since format string does not include timezone
                    # we just have to localize the datetime object
                    col = pd.to_datetime(column, format=target_format)
                    return col.dt.tz_localize(tz)

            return pd.to_datetime(column, format=target_format)

        try:
            if date_locale:
                with LocaleOverride(date_locale, category=locale.LC_TIME):
                    return _to_datetime()
            else:
                return _to_datetime()
        except locale.Error:
            warnings.warn(
                f'Unable to parse column "{column.name}" as datetime64, the '
                f'locale "{date_locale}" of the datetime does not appear to be '
                'available on this system. Column will retain current dtype. '
                'Note: This feature may not work properly until this locale '
                'is installed.')
        except Exception:  # noqa: Deliberately broad
            warnings.warn(f'Unable to parse column "{column.name}" as '
                          f'datetime64 using format "{date_time_format}". '
                          'Column will retain current dtype.')
        return column

    @classmethod
    def format_date_column(cls, column: pd.Series, feature: Mapping
                           ) -> pd.Series:
        """
        Format date only column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        date_time_format = feature.get('date_time_format') or None
        date_locale = feature.get('locale') or None
        target_format = to_pandas_datetime_format(date_time_format)
        try:
            if date_locale:
                with LocaleOverride(date_locale, category=locale.LC_TIME):
                    return pd.to_datetime(column, format=target_format)
            else:
                return pd.to_datetime(column, format=target_format)
        except locale.Error:
            warnings.warn(
                f'Unable to parse column "{column.name}" as datetime64, the '
                f'locale "{date_locale}" of the date does not appear to be '
                'available on this system. Column will retain current dtype. '
                'Note: This feature may not work properly until this locale '
                'is installed.')
        except Exception:  # noqa: Deliberately broad
            warnings.warn(f'Unable to parse column "{column.name}" as '
                          f'datetime64 using format {date_time_format}. '
                          'Column will retain current dtype.')
        return column

    @classmethod
    def format_time_column(cls, column: pd.Series, feature: Mapping  # noqa: C901
                           ) -> pd.Series:
        """
        Format time only column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)
        date_time_format = feature.get('date_time_format') or None
        if date_time_format:
            return cls.format_datetime_column(column, feature)
        else:
            # Time feature does not use a date_time_format, treat as seconds
            try:
                tz = typing_info['timezone']
                if tz is not None:
                    tz = pytz.timezone(tz)
            except pytz.UnknownTimeZoneError:
                tz = None
                warnings.warn(
                    f'Unknown timezone defined for column "{column.name}", '
                    'column will not be timezone aware.')
            except (TypeError, KeyError):
                tz = None

            return column.apply(partial(seconds_to_time, tzinfo=tz))

    @classmethod
    def format_boolean_column(cls, column: pd.Series, feature: Mapping
                              ) -> pd.Series:
        """
        Format boolean column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        try:
            def _apply_bool(value):
                if isinstance(value, str):
                    return value.lower() in ["true", "yes", "on", "(true)"]
                return bool(value)

            return column.apply(_apply_bool)
        except Exception:  # noqa: Deliberately broad
            warnings.warn(f'Unable to parse column "{column.name}" as '
                          f'boolean. Column will retain current dtype.')
            return column

    @classmethod
    def format_integer_column(cls, column: pd.Series, feature: Mapping
                              ) -> pd.Series:
        """
        Format integer column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)

        has_nulls = column.isnull().any()
        if has_nulls:
            # Use pandas extension type for nullable integer
            type_name = 'Int'
        else:
            type_name = 'int'

        size = typing_info.get('size')
        if size in [1, 2, 4, 8]:
            type_name = f'{type_name}{size * 8}'
        unsigned = typing_info.get('unsigned')
        if unsigned:
            type_name = f'U{type_name}' if has_nulls else f'u{type_name}'

        # unique or int-id features are output as 64bit ints, leave them
        # as-is if original datatype is less than 64 bits
        if size < 8 and (getattr(feature, 'unique', False) or
                         getattr(feature, 'subtype', None) == 'int-id'):
            warnings.warn(f'Unable to restore column "{column.name}" as '
                          f'{type_name}. Column will retain current dtype.')
            return column

        try:
            # We use floor and to_numeric here to handle cases where the value
            # may be a string of float like values, so we get the column as a
            # generic number first, remove the decimal places and then set the
            # type to the expected format
            return np.floor(pd.to_numeric(column)).astype(type_name)
        except Exception:  # noqa: Deliberately broad
            warnings.warn(f'Unable to parse column "{column.name}" as '
                          f'{type_name}. Column will retain current dtype.')
            return column

    @classmethod
    def format_numeric_column(cls, column: pd.Series, feature: Mapping
                              ) -> pd.Series:
        """
        Format numeric column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)
        number_format = typing_info.get('format')
        type_name = 'float'

        try:
            if number_format == 'decimal':
                type_name = 'Decimal'
                # Note: We cast to string first here in attempt to prevent
                # floating point errors.
                return column.apply(lambda x: decimal.Decimal(
                    str(x)) if x is not None else None)
            else:
                type_name = 'float'
                size = typing_info.get('size')
                if size in [1, 2, 4, 8, 16]:
                    type_name = f'{type_name}{size * 8}'

                # unique or int-id features are output as 64bit ints, leave them
                # as-is if original datatype is less than 64 bits
                if size < 8 and (getattr(feature, 'unique', False) or
                                 getattr(feature, 'subtype', None) == 'int-id'):
                    warnings.warn(
                        f'Unable to restore column "{column.name}" as '
                        f'{type_name}. Column will retain current dtype.'
                    )
                    return column

                return column.astype(type_name)
        except Exception:  # noqa: Deliberately broad
            warnings.warn(f'Unable to parse column "{column.name}" as '
                          f'{type_name}. Column will retain current dtype.')
            return column

    @classmethod
    def format_string_column(cls, column: pd.Series, feature: Mapping
                             ) -> pd.Series:
        """
        Format string column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        # Nothing to do for string columns
        return column

    @classmethod
    def format_unknown_column(cls, column: pd.Series, feature: Mapping
                              ) -> pd.Series:
        """
        Format unknown typed column.

        Parameters
        ----------
        column : pandas.Series
            The column to format.
        feature : Mapping
            The feature attributes for the column.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        # Unknown original type, don't modify column
        return column

    @staticmethod
    def _get_typing_info(feature: t.Optional[Mapping]) -> dict:
        """
        Get typing info from feature attributes.

        Parameters
        ----------
        feature : Mapping or None
            The feature attributes.

        Returns
        -------
        dict
            The typing info for the feature. Or empty dict if none found.
        """
        try:
            typing_info = feature['original_type']
        except (TypeError, KeyError):
            return dict()
        return typing_info or dict()


serialize_cases = FeatureSerializer.serialize
deserialize_cases = FeatureSerializer.deserialize
format_dataframe = FeatureSerializer.format_dataframe
