from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
import decimal
from enum import Enum
from functools import partial
import json
import locale
from typing import Any, cast, TYPE_CHECKING, TypeAlias
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

from howso.utilities.internals import IgnoreWarnings, to_pandas_datetime_format
from howso.utilities.tokenizing import HowsoTokenizer, TokenizerProtocol
from howso.utilities.utilities import (
    DATETIME_TIMEZONE_PATTERN,
    is_null_value,
    LocaleOverride,
    seconds_to_time,
    serialize_datetimes,
    tokenize_strings,
)

if TYPE_CHECKING:
    from pandas._typing import AstypeArg

__all__ = [
    "ColumnValues",
    "FeatureSerializer",
    "FeatureType",
    "deserialize_cases",
    "format_dataframe",
    "serialize_cases",
]

ColumnValues: TypeAlias = list[Any] | tuple[Any] | np.ndarray | ExtensionArray | pd.Series
"""The values of one column: a list of case values or an existing array/Series."""

_TRUE = frozenset({"true", "y", "yes", "on", ".true"})
"""String values that translate to True for boolean data."""


class FeatureType(Enum):
    """Feature type enum."""

    UNKNOWN = "object"
    STRING = "string"
    TOKENIZABLE_STRING = "tokenizable_string"
    NUMERIC = "numeric"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    TIMEDELTA = "timedelta"
    CONTAINER = "container"

    def __str__(self) -> str:
        """Return a string representation."""
        return str(self.value)


class FeatureSerializer:
    """Adapter for serialization and deserialization of feature data."""

    @classmethod
    def serialize(  # noqa: PLR0912
        cls,
        data: pd.DataFrame | np.ndarray | Iterable[Any] | None,
        columns: Sequence[str] | None,
        features: Mapping[str, Any],
        *,
        tokenizer: TokenizerProtocol | None = None,
        warn: bool = False,
    ) -> list[list[Any]] | None:
        """
        Serialize case data into list of lists.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray or Iterable or None
            The data to serialize, typically in a Pandas DataFrame, Numpy ndarray or Python Iterable such as a list.
        columns : Sequence of str or None
            The case column mapping. The order corresponds to the order of cases in output.
            `columns` must be provided for non-DataFrame Iterables.
        features : Mapping
            The dictionary of feature name to feature attributes.
        tokenizer : TokenizerProtocol, default None
            An object that satisfies :class:`howso.client.protocols.TokenizerProtocol`. Provides a tokenizer and
            `detokenize` method for processing tokenizable strings. If not specified, defaults to using
            :class:`howso.utilities.HowsoTokenizer`.
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
        from howso.client.exceptions import HowsoError  # noqa: PLC0415

        if data is None:
            return None
        if isinstance(data, np.ndarray):
            if columns is None:
                raise HowsoError("Columns must be provided with numpy arrays.")
            data = pd.DataFrame.from_records(data, columns=list(columns))

        if isinstance(data, pd.DataFrame):
            if columns is not None:
                columns = list(columns)
                try:
                    filtered_data = data[columns]
                except KeyError as key_error:
                    raise HowsoError(
                        f"The provided DataFrame is missing one or more of the expected named columns: {columns}."
                    ) from key_error
            else:
                columns = data.columns.tolist()
                filtered_data = data

            data_columns = []
            for col in columns:
                col_data = filtered_data[col]
                if isinstance(col_data, pd.DataFrame):
                    raise ValueError(  # noqa: TRY004
                        "The provided DataFrame contains duplicate column names. All column names must be unique."
                    )
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
            result = np.array(data_columns, dtype=object).T.tolist()

        elif isinstance(data, Iterable) and not isinstance(data, str):
            if columns is None:
                raise HowsoError("Columns must be provided with python Iterables.")
            result = list(data)
        else:
            raise TypeError(
                "Received unexpected data type for case data. Cases should be either a DataFrame or a list."
            )

        # Convert 1d list (single case) to 2d list for serialization
        # We check if the first item in the serialized list is a list
        # rather than converting to a numpy array and checking the shape to
        # avoid the conversion of a (potentially quite large) list.
        if len(result) > 0 and not isinstance(result[0], list):
            result = [result]

        # Serialize datetime objects
        serialize_datetimes(result, columns, features, warn=warn)

        # Tokenize any tokenizable strings
        tokenize_strings(result, columns, features, tokenizer=tokenizer or HowsoTokenizer())

        # Stringify any raw JSONs
        cls._stringify_json(result, columns, features)

        return result

    @staticmethod
    def _stringify_json(
        cases: list[list[Any]],
        columns: Sequence[str],
        features: Mapping[str, Any],
    ) -> None:
        """
        Ensure that any JSON features have their cases stringified.

        Null values are left as-is so that they are sent to the engine as nulls rather than as the
        JSON string "null".

        Parameters
        ----------
        cases : list of list of Any
            A 2d list of case values corresponding to the provided feature names.
        columns : Sequence of str
            The list of feature names, corresponding to the columns in the 2d case list.
        features : Mapping
            The dictionary of feature name to feature attributes.
        """
        for idx, feature_name in enumerate(columns):
            if features.get(feature_name, {}).get("data_type") == "json":
                orig_type_info = features[feature_name].get("original_type", {})
                # Applicable if original type is an object (Python list/dict) or string, tokenized into a list
                if orig_type_info.get("data_type", {}) in ["container", "tokenizable_string"]:
                    coerce_set = orig_type_info.get("coercion") == "set"
                    for row in cases:
                        value = row[idx]
                        if is_null_value(value):
                            row[idx] = None
                            continue
                        if coerce_set:
                            # If an original type indicates a set, convert to list for serialization
                            value = list(value)
                        row[idx] = json.dumps(value)

    @classmethod
    def deserialize(
        cls,
        data: list[list[Any]],
        columns: Sequence[str],
        index: pd.Index | None = None,
        *,
        features: Mapping[str, Any] | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> pd.DataFrame:
        """
        Deserialize case data into a DataFrame.

        If feature attributes contain original typing information, columns
        will be converted to the same data type as original training cases.

        Parameters
        ----------
        data : list of list
            The context data.
        columns : Sequence of str, optional
            The case column mapping. The order corresponds to the order of cases in output.
            `columns` must be provided for non-DataFrame Iterables.

            The order corresponds to how the data will be mapped to columns in
            the output. Ignored for list of dict where the dict key is the column
            name.
        index : pandas.Index, optional
            The index of the resulting DataFrame. Defaults to a RangeIndex.
        features : Mapping, default None
            (Optional) The dictionary of feature name to feature attributes.

            If not specified, no column typing will be attempted.
        tokenizer : TokenizerProtocol, default None
            An object that satisfies :class:`howso.client.protocols.TokenizerProtocol`. Provides a tokenizer and
            `detokenize` method for processing tokenizable strings. If not specified, defaults to using
            :class:`howso.utilities.HowsoTokenizer`.

        Returns
        -------
        pandas.DataFrame
            The deserialized data.
        """
        data = [] if data is None else data  # pyright: ignore[reportUnnecessaryComparison]

        # One shared index so the constructor doesn't need to align the Series
        index = pd.RangeIndex(len(data)) if index is None else index
        values = cls._convert_columns(data, columns, index, features=features, tokenizer=tokenizer)

        # Key columns by position to allow duplicate column names, constructor will consolidate same-dtype columns
        series = dict(enumerate(values))
        del values
        df = pd.DataFrame(series, index=index)
        del series
        df.columns = pd.Index(columns)
        return df

    @classmethod
    def _convert_columns(
        cls,
        rows: list[list[Any]],
        columns: Sequence[str],
        index: pd.Index,
        *,
        features: Mapping[str, Any] | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> list[pd.Series]:
        """Convert and consume each column of `rows` into an array of its final dtype, in column order."""
        n_cols = len(columns)
        for row in rows:
            if len(row) != n_cols:
                raise ValueError(
                    f"Unable to deserialize data. Every row must have {n_cols} values, "
                    f"encountered row with {len(row)}."
                )

        formatted_columns: list[pd.Series | None] = [None] * n_cols

        # Work from the last column back: `list.pop` takes each value out of its row, so the
        # column list holds the only reference to each value and they are freed once converted.
        for position in reversed(range(n_cols)):
            attributes = features.get(columns[position]) if features is not None else None
            values = [row.pop() for row in rows]
            formatted_columns[position] = cls.format_column(
                column=values,
                name=columns[position],
                index=index,
                feature=attributes,
                tokenizer=tokenizer,
            )
            del values
        rows.clear()
        return cast(list[pd.Series], formatted_columns)

    @classmethod
    def format_dataframe(
        cls,
        df: pd.DataFrame,
        features: Mapping[str, Any],
        tokenizer: TokenizerProtocol | None = None,
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
        tokenizer : TokenizerProtocol, default None
            An object that satisfies :class:`howso.client.protocols.TokenizerProtocol`. Provides a tokenizer and
            `detokenize` method for processing tokenizable strings. If not specified, defaults to using
            :class:`howso.utilities.HowsoTokenizer`.

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
            df[col] = cls.format_column(df[col], attributes, tokenizer=tokenizer)

        return df

    @classmethod
    def format_column(  # noqa: PLR0911, PLR0912
        cls,
        column: ColumnValues,
        feature: Mapping[str, Any] | None,
        *,
        index: pd.Index | None = None,
        name: Hashable | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> pd.Series:
        """
        Format column based on feature typing information.

        Parameters
        ----------
        column : ColumnValues
            The column to format; a Series, or the column's values.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series. Defaults to the index of `column` when it is a Series, otherwise
            a RangeIndex. Values are matched to the index by position.
        name : Hashable, optional
            The name of the resulting Series. Defaults to the name of `column` when it is a Series.
        tokenizer : TokenizerProtocol, optional
            An object that satisfies :class:`howso.client.protocols.TokenizerProtocol`. Provides a tokenizer and
            `detokenize` method for processing tokenizable strings. If not specified, defaults to using
            :class:`howso.utilities.HowsoTokenizer`.

        Returns
        -------
        pandas.Series
            The formatted column. When `column` is a Series that needs no formatting, `column` itself.
        """
        feature = {} if feature is None else feature
        data_type = cls._get_typing_info(feature).get("data_type")

        if isinstance(column, pd.Series):
            if name is None:
                name = column.name
            if index is None:
                index = column.index

        try:
            match data_type:
                case FeatureType.NUMERIC.value:
                    return cls._format_numeric_column(column, feature, index, name)
                case FeatureType.INTEGER.value:
                    return cls._format_integer_column(column, feature, index, name)
                case FeatureType.STRING.value:
                    return cls._format_string_column(column, feature, index, name)
                case FeatureType.TOKENIZABLE_STRING.value:
                    return cls._format_tokenizable_string_column(column, feature, index, name, tokenizer=tokenizer)
                case FeatureType.DATETIME.value:
                    return cls._format_datetime_column(column, feature, index, name)
                case FeatureType.DATE.value:
                    return cls._format_date_column(column, feature, index, name)
                case FeatureType.TIME.value:
                    return cls._format_time_column(column, feature, index, name)
                case FeatureType.TIMEDELTA.value:
                    return cls._format_timedelta_column(column, feature, index, name)
                case FeatureType.BOOLEAN.value:
                    return cls._format_boolean_column(column, feature, index, name)
                case FeatureType.CONTAINER.value:
                    return cls._format_container_column(column, feature, index, name)
                case _:
                    return cls._format_unknown_column(column, feature, index, name)
        except locale.Error:
            feature_locale = feature.get("locale")
            warnings.warn(
                f'Unable to format {data_type} column "{name}", the required '
                f'locale "{feature_locale}" does not appear to be available on this system. '
                "Column will retain inferred dtype. Note: This feature may not work "
                "properly until this locale is installed.",
                stacklevel=2,
            )
        except Exception:  # noqa: BLE001
            warnings.warn(
                f'Unable to format {data_type} column "{name}". Column will retain inferred dtype.',
                stacklevel=2,
            )

        return cls._to_series(column, index=index, name=name)

    @classmethod
    def _format_timedelta_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format timedelta column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        if isinstance(values, pd.Series) and pd.api.types.is_timedelta64_dtype(values.dtype):
            # Nothing to format
            return values

        typing_info = cls._get_typing_info(feature)
        try:
            unit = typing_info["unit"]
            if unit not in ["days", "seconds", "nanoseconds"]:
                raise ValueError("invalid unit type")  # noqa: TRY301
        except (TypeError, KeyError, ValueError):
            warnings.warn(
                f'Unknown timedelta unit for feature "{name}", column will not be translated to a timedelta64.',
                stacklevel=3,
            )
            return cls._to_series(values, index=index, name=name)

        timedeltas = pd.to_timedelta(values, unit=unit)
        return cls._to_series(timedeltas, index=index, name=name)

    @classmethod
    def _format_datetime_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format datetime column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)
        date_time_format = feature.get("date_time_format") or None
        date_locale = feature.get("locale") or None
        format_includes_timezone = False
        if date_time_format and DATETIME_TIMEZONE_PATTERN.findall(date_time_format):
            format_includes_timezone = True

        # Load timezone information if provided
        tz = typing_info.get("timezone")
        if tz is not None:
            try:
                tz = ZoneInfo(tz)
            except KeyError:
                warnings.warn(
                    f'Unknown timezone "{tz}" for feature "{name}", '
                    "datetime column may not contain original timezone information.",
                    stacklevel=3,
                )
                tz = None

        def _to_datetime() -> pd.Series:
            # Format column to datetime
            target_format = None
            if date_time_format is not None:
                target_format = to_pandas_datetime_format(date_time_format)
            if tz is not None:
                if format_includes_timezone:
                    # When format string includes timezone we want to load
                    # the datetime as UTC, then we can convert it to the named
                    # timezone
                    col = pd.to_datetime(column, format=target_format, utc=True)
                    return col.dt.tz_convert(tz)
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
                "available on this system. Column will retain current dtype. "
                "Note: This feature may not work properly until this locale "
                "is installed."
            )
        except Exception:  # noqa: BLE001
            warnings.warn(
                f'Unable to parse column "{column.name}" as '
                f'datetime64 using format "{date_time_format}". '
                "Column will retain current dtype."
            )
        return column

    @classmethod
    def _format_date_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format date only column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        date_time_format = feature.get("date_time_format") or None
        date_locale = feature.get("locale") or None
        target_format = None
        if date_time_format is not None:
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
                "available on this system. Column will retain current dtype. "
                "Note: This feature may not work properly until this locale "
                "is installed."
            )
        except Exception:  # noqa: BLE001
            warnings.warn(
                f'Unable to parse column "{column.name}" as '
                f"datetime64 using format {date_time_format}. "
                "Column will retain current dtype."
            )
        return column

    @classmethod
    def _format_time_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format time only column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)
        date_time_format = feature.get("date_time_format")
        if date_time_format:
            return cls._format_datetime_column(values, feature, index, name)

        # Time feature does not use a date_time_format, treat as seconds
        tz = None
        try:
            tz_name = typing_info["timezone"]
            if tz_name is not None:
                try:
                    tz = ZoneInfo(tz_name)
                except KeyError:
                    warnings.warn(
                        f'Unknown timezone defined for feature "{name}", time column will not be timezone aware.',
                        stacklevel=3,
                    )
        except (TypeError, KeyError):
            pass

        array = np.fromiter(map(partial(seconds_to_time, tzinfo=tz), values), dtype=object, count=len(values))
        return cls._to_series(array, index=index, name=name)

    @classmethod
    def _format_boolean_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format boolean column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        nullable = feature.get("bounds", {}).get("null_observed", False)

        def _to_code(value: Any) -> int:
            # 0 = False, 1 = True, 2 = null
            if value is True or value is False:
                return value
            if value is None:
                return 2
            if isinstance(value, int):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in _TRUE
            if is_null_value(value):
                return 2
            return bool(value)

        # Encode all values in one pass into a compact array
        codes = np.fromiter(map(_to_code, values), dtype=np.uint8, count=len(values))
        mask = codes == 2  # null values
        if nullable or mask.any():
            # Nullable boolean array, masked positions become <NA>
            array = pd.arrays.BooleanArray(codes == 1, mask)
        else:
            # Plain bool array when no nulls present
            array = codes == 1
        return cls._to_series(array, index=index, name=name)


    @classmethod
    def _format_integer_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format integer column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)

        has_nulls = column.isna().any()
        if has_nulls:
            # Use pandas extension type for nullable integer
            type_name = "Int"
        else:
            type_name = "int"

        size = typing_info.get("size")
        if size in [1, 2, 4, 8]:
            type_name = f"{type_name}{size * 8}"
        unsigned = typing_info.get("unsigned")
        if unsigned:
            type_name = f"U{type_name}" if has_nulls else f"u{type_name}"

        # unique or int-id features are output as 64bit ints, leave them
        # as-is if original datatype is less than 64 bits
        if (
            size is not None
            and size < 8
            and (getattr(feature, "unique", False) or getattr(feature, "subtype", None) == "int-id")
        ):
            warnings.warn(
                f'Unable to restore column "{name}" as {type_name}. Column will retain current dtype.',
                stacklevel=3,
            )
            return column

        try:
            # We use floor and to_numeric here to handle cases where the value
            # may be a string of float like values, so we get the column as a
            # generic number first, remove the decimal places and then set the
            # type to the expected format (np.floor will actually return a
            # pandas series when given one, but the type hint only shows ndarray)
            return cast(pd.Series, np.floor(pd.to_numeric(column)).astype(cast(np.dtype, type_name)))
        except Exception:  # noqa: BLE001
            warnings.warn(
                f'Unable to parse column "{name}" as {type_name}. Column will retain current dtype.',
                stacklevel=3,
            )
            return column

    @classmethod
    def _format_numeric_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format numeric column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        typing_info = cls._get_typing_info(feature)

        if typing_info.get("format") == "decimal":
            # Note: We cast to string first here in attempt to prevent floating point errors.
            decimals = [decimal.Decimal(str(x)) if x is not None else None for x in values]
            return cls._to_series(decimals, index, name)

        if feature.get("subtype") == "int-id" or feature.get("unique"):
            # unique or int-id features are output as 64bit ints, leave as-is ignore original type size
            dtype = "float"
        else:
            size = typing_info.get("size")
            dtype = f"float{size * 8}" if size in {1, 2, 4, 8, 16} else "float"

        return cls._to_series(values, index, name, dtype=np.dtype(dtype))


    @classmethod
    def _format_string_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],  # noqa: ARG003
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format string column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        return cls._to_series(values, index, name)

    @classmethod
    def _format_tokenizable_string_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
        *,
        tokenizer: TokenizerProtocol | None = None,
    ) -> pd.Series:
        """
        Format tokenizable string column.

        Null values are preserved as None, as there is nothing to detokenize.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.
        tokenizer : TokenizerProtocol, default None
            An object that satisfies :class:`howso.client.protocols.TokenizerProtocol`. Provides a tokenizer and
            `detokenize` method for processing tokenizable strings. If not specified, defaults to using
            :class:`howso.utilities.HowsoTokenizer`.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        tokenizer = tokenizer or HowsoTokenizer()
        values = cls._destringify_json(values, feature, tokenizer=tokenizer)
        return cls._to_series(values, index, name)

    @classmethod
    def _format_container_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format a container column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        values = cls._destringify_json(values, feature)
        return cls._to_series(values, index, name, dtype="object")

    @classmethod
    def _format_unknown_column(
        cls,
        values: ColumnValues,
        feature: Mapping[str, Any],  # noqa: ARG003
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series:
        """
        Format unknown typed column.

        Parameters
        ----------
        values : ColumnValues
            The column values to format.
        feature : Mapping
            The feature attributes for the column.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.

        Returns
        -------
        pandas.Series
            The formatted column.
        """
        return cls._to_series(values, index, name)

    @staticmethod
    def _destringify_json(
        values: ColumnValues,
        features: Mapping[str, Any],
        *,
        tokenizer: TokenizerProtocol | None = None,
    ) -> np.ndarray:
        """
        De-stringify JSON feature values.

        Null values are returned as None. A null may arrive either directly as a null value or JSON
        encoded as the literal ``null`` (which `json.loads` decodes to None).

        Parameters
        ----------
        values : ColumnValues
            The column values to destringify.
        features : Mapping
            The feature attributes for the column.
        tokenizer : TokenizerProtocol, optional
            A tokenizer to allow detokenizing strings.
            Only applies to features whose original data type is "tokenizable_string".
        """
        typing_info = features.get("original_type", {}) or {}
        is_set = typing_info.get("coercion") == "set"
        is_tokenized = typing_info.get("data_type") == "tokenizable_string"

        def _process(raw_value: Any) -> Any:
            if isinstance(raw_value, (str, bytes, bytearray)):
                # The JSON literal ``null`` decodes to None, which cannot be coerced any further
                value = json.loads(raw_value)
            elif is_null_value(raw_value):
                return None
            else:
                value = raw_value

            if value is not None:
                if is_tokenized and tokenizer is not None:
                    value = tokenizer.detokenize(value)
                elif is_set:
                    value = set(value)
            return value

        return np.fromiter(map(_process, values), dtype=object, count=len(values))

    @staticmethod
    def _to_series(
        values: ColumnValues | pd.Index,
        index: pd.Index | None = None,
        name: Hashable | None = None,
        *,
        copy: bool = False,
        dtype: AstypeArg | None = None,
    ) -> pd.Series:
        """
        Make a Series of `values`, with `index` and `name` applied by position (never aligned).

        A Series is only changed through its own methods, so it is returned as is when nothing differs, and is
        never copied unless its dtype changes. Arrays are used without copying by default, so formatters should only
        pass arrays they created or specify `copy=True`. A list is converted in one pass, straight to `dtype`
        when specified.

        Parameters
        ----------
        values : ColumnValues or pandas.Index
            The values of the Series.
        index : pandas.Index, optional
            The index of the resulting Series.
        name : Hashable, optional
            The name of the resulting Series.
        copy : bool,  default False
            If values should be copied when constructing the Series object.
            (Only applies to numpy or extension arrays)
        dtype : AstypeArg, optional
            The dtype to convert to. Defaults to the dtype of `values` when it is an array, otherwise pandas'
            default inference.

        Returns
        -------
        pandas.Series
            The Series.
        """
        if isinstance(values, pd.Series):
            if dtype is not None and values.dtype != dtype:
                values = values.astype(dtype)
            if index is not None and values.index is not index:
                values = values.set_axis(index)
            if name is not None and values.name != name:
                values = values.copy(deep=False)
                values.name = name
            return values
        if dtype is None and isinstance(values, (np.ndarray, ExtensionArray)):
            # Keep the dtype, otherwise pandas 3 re-infers object arrays of strings as `str` (a copy)
            # If the input is pandas.Index, leave `dtype` None
            dtype = values.dtype
        return pd.Series(values, index=index, name=name, dtype=dtype, copy=copy)

    @staticmethod
    def _get_typing_info(feature: Mapping[str, Any] | None) -> dict[str, Any]:
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
        if feature is None:
            return {}
        try:
            typing_info = feature["original_type"]
        except (TypeError, KeyError):
            return {}
        return typing_info or {}


serialize_cases = FeatureSerializer.serialize
deserialize_cases = FeatureSerializer.deserialize
format_column = FeatureSerializer.format_column
format_dataframe = FeatureSerializer.format_dataframe
