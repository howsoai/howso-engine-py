from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Container, Iterable, Mapping, MutableSequence, Sequence, Set
from copy import deepcopy
import datetime
from functools import singledispatchmethod
import json
import logging
import math
from pathlib import Path
import platform
import typing as t
import warnings
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse, parse as dt_parse
import numpy as np
import pandas as pd
from typing_extensions import Self
import yaml

from howso.utilities.feature_attributes.serializers import feature_attributes_pairs_hook, FeatureAttributesEncoder
from howso.utilities.feature_attributes.suggestions import (
    FullPreserveRareValuesConfig,
    IFASuggestion,
    IFASuggestionCollector,
    PreserveRareValuesConfig,
    PreserveRareValuesMap,
    PRVSuggestion,
)
from howso.utilities.feature_attributes.warnings import IFAWarningCollector, IFAWarningEmitterType
from howso.utilities.features import FeatureType
from howso.utilities.utilities import (
    determine_iso_format,
    get_optimized_max_chunk_size,
    is_valid_datetime_format,
    time_to_seconds,
)

if t.TYPE_CHECKING:
    from howso.client.typing import FeatureAttributes

logger = logging.getLogger(__name__)

# Format string tokens for datetime and time-only features
DATE_TOKENS = {"%m", "%d", "%y", "%z", "%D", "%F", "%Y", "%G", "%C"}
TIME_TOKENS = {"%R", "%T", "%I", "%X", "%r", "%H", "%M", "%S", "%f", "%p"}
# Maximum/minimum data sizes for integers, floats, datetimes supported by the core
FLOAT_MAX = 1.7976931348623157 * math.pow(10, 308)
FLOAT_MIN = 2.2250738585072014 * math.pow(10, -308)
INTEGER_MAX = int(math.pow(2, 53))
LINUX_DT_MAX = "2262-04-11"
WIN_DT_MAX = "6053-01-24"

# Define a TypeVar which is FeatureAttributesBase or any subclass.
FeatureAttributesBaseType = t.TypeVar("FeatureAttributesBaseType", bound="FeatureAttributesBase")

SIGNIFICANT_THRESHOLD_DEFAULT: int = 30


class FeatureAttributesBase(dict[str, "FeatureAttributes"]):
    """Provides accessor methods for and dict-like access to inferred feature attributes."""

    def __init__(self, feature_attributes: Mapping, params: dict | None = None, unsupported: list[str] | None = None,
                 suggestions_collector: IFASuggestionCollector | None = None) -> None:
        """
        Instantiate this FeatureAttributesBase object.

        Parameters
        ----------
        feature_attributes : dict
            The feature attributes dictionary to be wrapped by this object.
        params : dict
            (Optional) The parameters used in the call to infer_feature_attributes.
        unsupported : list of str
            (Optional) A list of features that contain data that is unsupported by the engine.
        suggestions_collector : IFASuggestionCollector
            (Optional) Collector of suggestions for this FeatureAttributesBase object.
        """
        if not isinstance(feature_attributes, Mapping):
            raise TypeError("Provided feature attributes must be a Mapping.")
        self.params = params or {}
        self.update(feature_attributes)
        self.unsupported = unsupported or []
        self.warnings_collector = IFAWarningCollector()
        self.suggestions_collector = suggestions_collector or "You have no suggestions."

    def __copy__(self) -> FeatureAttributesBase:
        """Return a (deep)copy of this instance of FeatureAttributesBase."""
        cls = self.__class__
        obj_copy = cls.__new__(cls)
        obj_copy.update(deepcopy(self))
        obj_copy.params = self.params
        return obj_copy

    def apply_suggestion(self, key: str) -> None:
        """
        Apply the suggestion under the provided key.

        Parameters
        ----------
        key : str
            The key of the suggestion to apply. Use "all" to apply all suggestions.
        """
        if not isinstance(self.suggestions_collector, str):
            if key == "all":
                for suggestion in self.suggestions_collector.suggestions.values():
                    suggestion.apply(self)
            else:
                suggestion: IFASuggestion = getattr(self.suggestions_collector, key)
                if not suggestion:
                    raise KeyError(f"No suggestion found under key `{key}`")
                suggestion.apply(self)
        else:
            raise ValueError(self.suggestions_collector)  # noqa: TRY004

    @property
    def suggestions(self) -> IFASuggestionCollector:
        """Get the suggestions for this FeatureAttributesBase object."""
        return self.suggestions_collector

    def get_parameters(self) -> dict:
        """
        Get the keyword arguments used with the initial call to infer_feature_attributes.

        Returns
        -------
        dict
            A dictionary containing the kwargs used in the call to `infer_feature_attributes`.

        """
        return self.params

    def to_json(self, archive: bool = False, json_path: Path | None = None) -> str:
        """
        Get a JSON string representation of this FeatureAttributes object.

        Parameters
        ----------
        archive : bool, default False
            If True, the returned JSON includes 3 top-level keys:

            - feature_attributes - A nested map of the inferred feature attributes.
            - params - A map of parameters and their values used to infer the feature attributes.
            - unsupported - A list of features not supported by the Howso Engine.

            If False, only the nested map of the feature attributes is returned.

        json_path : Path, optional
            If provided, the JSON will be written to this path in addition to being returned.

        Returns
        -------
        String
            A JSON representation of the inferred feature attributes.
        """
        if archive:
            json_str = json.dumps({
                "feature_attributes": self,
                "params": self.params,
                "unsupported": self.unsupported,
            }, cls=FeatureAttributesEncoder)
        else:
            json_str = json.dumps(self, cls=FeatureAttributesEncoder)

        if json_path:
            with Path.open(json_path, mode="w") as fp:
                fp.write(json_str)

        return json_str

    @classmethod
    def from_json(
        cls: Self,
        json_str: str | None = None,
        *,
        json_path: str | None = None
    ) -> Self:
        """
        Reconstruct a FeatureAttributesBase from JSON.

        Parameters
        ----------
        json_str : str, optional
            A JSON object serialized to a string.
        json_path : Path, optional
            A path to a JSON file.

        Returns
        -------
        FeatureAttributesBaseType
            An instance of FeatureAttributesBase or any of its subclasses.
        """
        if json_path and json_str:
            warnings.warn(
                "The `json_str` parameter of `from_json` is ignored if the "
                "`json_path` parameter is also provided.", UserWarning
            )

        if not json_path and not json_str:
            warnings.warn(
                "Either the `json_str` or `json_path` parameter of "
                "`from_json` is required.", UserWarning
            )

        if json_path:
            with open(json_path, mode="r") as fp:
                obj_dict = json.load(fp, object_pairs_hook=feature_attributes_pairs_hook)
        else:
            obj_dict = json.loads(json_str or "", object_pairs_hook=feature_attributes_pairs_hook)

        # If there are no top-level keys other than the archival_keys, it's an archive.
        archival_keys = {"feature_attributes", "params", "unsupported"}
        if not (set(obj_dict.keys()) - archival_keys):
            return cls(**obj_dict)

        # Else, it's just the feature_attributes.
        return cls(feature_attributes=obj_dict)

    def to_dataframe(self, *, include_all: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame of the feature attributes.

        Among other reasons, this is useful for presenting feature attributes
        in a Jupyter notebook or other medium.

        Returns
        -------
        pandas.DataFrame
            A DataFrame representation of the inferred feature attributes.
        """
        raise NotImplementedError('Function not yet implemented for all subclasses of `FeatureAttributesBase`')

    def get_names(self, *, types: t.Optional[str | Container] = None,
                  data_types: t.Optional[str | Container] = None,
                  without: t.Optional[str | Iterable[str]] = None,
                  ) -> list[str]:
        """
        Get feature names associated with this FeatureAttributes object.

        Parameters
        ----------
        types : String, Container (of String), default None
            (Optional) A feature type as a string (E.g., 'continuous') or a
            list of feature types to limit the output feature names.
        data_types : String, Container (of String), default None
            (Optional) A ``data_type`` as a string (E.g., 'datetime') or a list
            of ``data_type`` to limit the output of feature names.
        without : String or Iterable of String
            (Optional) A feature name or an Iterable of feature names to exclude from the return object.

        Returns
        -------
        list of str
            A list of feature names.
        """
        if isinstance(without, str):
            without = [without]
        if without:
            for feature in without:
                if feature not in self.keys():
                    raise ValueError(f'Feature {feature} does not exist in this FeatureAttributes '
                                     'object')
        names = self.keys()

        if types:
            if isinstance(types, str):
                types = [types, ]
        else:
            types = []
        if data_types:
            if isinstance(data_types, str):
                data_types = [data_types, ]
        else:
            data_types = []
        names = [
            name for name in names
            if (self[name].get('type') in types or not types)
            and (self[name].get('data_type') in data_types or not data_types)
        ]

        return [
            key for key in names
            if without is None or key not in without
        ]

    def _validate_bounds(self, data: pd.DataFrame, feature: str,  # noqa: C901
                         attributes: FeatureAttributes) -> list[str]:
        """Validate the feature bounds of the provided DataFrame."""
        # Import here to avoid circular import
        from howso.utilities import date_to_epoch

        errors = []

        # Ensure that there are bounds to validate
        if not isinstance(attributes.get("bounds"), Mapping) or attributes.get("data_type") in ["json", "yaml"]:
            return errors

        # Gather some data to use for validation
        series = data[feature]
        bounds = attributes['bounds']  # pyright: ignore[reportTypedDictNotRequiredAccess]
        min_bound = bounds.get('min')
        max_bound = bounds.get('max')
        # Get unique values but exclude NoneTypes
        unique_values = series.dropna().unique()
        additional_errors = 0

        if bounds.get('allowed'):
            # Check nominal bounds
            allowed_values = attributes['bounds']['allowed']  # pyright: ignore[reportTypedDictNotRequiredAccess]
            out_of_band_values = set(unique_values) - set(allowed_values)
            if pd.isna(list(out_of_band_values)).all():
                # Placeholder for behavior when columns contain nans
                pass
            elif out_of_band_values:
                errors.append(f"'{feature}' contains out-of-band values: {out_of_band_values}")
        elif attributes.get('date_time_format'):
            # Time-only attributes have bounds represented in seconds
            if attributes.get('original_type', {}).get('data_type') == 'time':
                unique_time_values = pd.to_datetime(
                    series,
                    format=attributes['date_time_format'],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    errors='coerce'
                ).dropna().unique()
                for value in unique_time_values:
                    value_in_seconds = time_to_seconds(value.time())
                    if (max_bound and value_in_seconds > max_bound) or (min_bound and value_in_seconds < min_bound):
                        if len(errors) < 5:
                            errors.append(
                                f'"{feature}" has a value outside of bounds '
                                f'(min: {min_bound}, max: {max_bound}): {value}'
                            )
                        else:
                            additional_errors += 1
            # If this is a datetime feature, convert dates to epoch time for bounds comparison
            else:
                try:
                    if min_bound:
                        min_bound_epoch = date_to_epoch(min_bound, time_format=attributes['date_time_format'])
                    if max_bound:
                        max_bound_epoch = date_to_epoch(max_bound, time_format=attributes['date_time_format'])
                    for value in unique_values:
                        epoch = date_to_epoch(value, time_format=attributes['date_time_format'])
                        if (max_bound and epoch > max_bound_epoch) or (min_bound and epoch < min_bound_epoch):
                            if len(errors) < 5:
                                errors.append(
                                    f'"{feature}" has a value outside of bounds '
                                    f'(min: {min_bound}, max: {max_bound}): {value}'
                                )
                            else:
                                additional_errors += 1
                except ValueError as err:
                    errors.append(f'Could not validate datetime bounds due to the following error: {err}')
        elif min_bound or max_bound:
            # Check int/float bounds
            for value in unique_values:
                if (max_bound and float(value) > float(max_bound)) or (min_bound and float(value) < float(min_bound)):
                    if len(errors) < 5:
                        errors.append(
                            f'"{feature}" has a value outside of bounds '
                            f'(min: {min_bound}, max: {max_bound}): {value}'
                        )
                    else:
                        additional_errors += 1
        if additional_errors > 0:
            errors.append(
                f'"{feature}" had {additional_errors} additional values outside of bounds that were not displayed.')
        return errors

    def _validate_dtype(self, data: pd.DataFrame, feature: str,  # noqa: C901
                        expected_dtype: str | pd.CategoricalDtype, coerced_df: pd.DataFrame,
                        coerce: bool = False, localize_datetimes: bool = True) -> list[str]:
        """Validate the data type of a feature and optionally attempt to coerce."""
        errors = []
        series = coerced_df[feature]
        is_valid = False
        coerce_err = ""

        if isinstance(expected_dtype, pd.CategoricalDtype):
            # If the feature is a Categorical dtype, try to coerce
            try:
                series = series.astype(expected_dtype)
                if coerce:
                    coerced_df[feature] = series
                is_valid = True
            except Exception: # noqa: Intentionally broad
                pass
        elif expected_dtype == 'datetime64':
            try:
                format = self[feature]['date_time_format']  # pyright: ignore[reportTypedDictNotRequiredAccess]
                if ".%f" in format:
                    format = "ISO8601"
                series = pd.to_datetime(coerced_df[feature], format=format)
                if coerce:
                    if localize_datetimes and not isinstance(series, pd.DatetimeTZDtype):
                        coerced_df[feature] = series.dt.tz_localize(
                            'UTC', ambiguous='infer', nonexistent='NaT'
                        )
                    else:
                        coerced_df[feature] = series
                is_valid = True
            except Exception: # noqa: Intentionally broad
                pass
        else:
            # Else, compare the dtype directly
            if data[feature].dtype.name == expected_dtype:
                is_valid = True
            # If the feature can be converted, consider it valid (slightly differing numeric types, etc.)
            else:
                try:
                    series = series.astype(expected_dtype)
                    if coerce:
                        coerced_df[feature] = series
                    is_valid = True
                except pd.errors.IntCastingNaNError:
                    # If this happens, there is a null value, thus a float dtype is OK
                    if pd.api.types.is_float_dtype(series):
                        is_valid = True
                except Exception as err: # noqa: Intentionally broad
                    coerce_err = str(err)
                    pass

        # Raise warnings if the types do not match
        if not is_valid:
            if coerce:
                errors.append(f"Expected dtype '{expected_dtype}' for feature '{feature}' "
                              f"but could not coerce:\nActual dtype: {data[feature].dtype}"
                              f"\nError raised from Pandas.astype():\n\n{coerce_err}")
            else:
                errors.append(f"Feature '{feature}' should be '{expected_dtype}' dtype, but found "
                              f"'{data[feature].dtype}'")

        return errors

    @staticmethod
    def _allows_null(attributes: FeatureAttributes) -> bool:
        """Return whether the given attributes indicates the allowance of null values."""
        return 'bounds' in attributes and attributes['bounds'].get('allow_null', False)

    def _validate_df(self, data: pd.DataFrame, coerce: bool = False,  # noqa: C901
                     raise_errors: bool = False, table_name: t.Optional[str] = None, validate_bounds=True,
                     allow_missing_features: bool = False, localize_datetimes=True, nullable_int_dtype='Int64'):
        errors = []
        coerced_df = data.copy(deep=True)
        features = t.cast(dict[str, "FeatureAttributes"], self[table_name] if table_name else self)

        for feature, attributes in features.items():
            if feature not in data.columns:
                # Check if column is missing (and not supposed to be)
                if not (
                    feature.startswith('.')
                    or (
                        attributes.get('auto_derive_on_train', False)
                        and 'derived_feature_code' in attributes
                    )
                    or allow_missing_features
                ):
                    errors.append(f'{feature} is missing from the dataframe')
                # OK if it's an internal feature or is being processed by Validator
                continue

            # Check nominal types
            if attributes['type'] == 'nominal':
                if attributes.get('data_type') == 'number':
                    # Check type (float)
                    if attributes.get('decimal_places', 0) > 0:
                        errors.extend(self._validate_dtype(data, feature, 'float64',
                                                           coerced_df, coerce=coerce))
                    # Check type (nullable Int)
                    elif self._allows_null(attributes):
                        errors.extend(self._validate_dtype(data, feature, nullable_int_dtype,
                                                           coerced_df, coerce=coerce))
                    # Check type (int)
                    else:
                        errors.extend(self._validate_dtype(data, feature, 'int64',
                                                           coerced_df, coerce=coerce))
                elif attributes.get('data_type') == 'boolean':
                    # Check type (boolean)
                    errors.extend(self._validate_dtype(data, feature, 'bool',
                                                       coerced_df, coerce=coerce))
                elif attributes.get('bounds') and attributes['bounds'].get('allowed'):  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    # Check type (categorical)
                    schema_dtype = pd.CategoricalDtype(attributes['bounds']['allowed'],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                                                       ordered=True)
                    errors.extend(self._validate_dtype(data, feature, schema_dtype,
                                                       coerced_df, coerce=coerce))
                else:
                    # Else, should be an object
                    errors.extend(self._validate_dtype(data, feature, 'object',
                                                       coerced_df, coerce=coerce))

            # Check ordinal types
            elif attributes['type'] == 'ordinal':
                if attributes.get('bounds') and attributes['bounds'].get('allowed'):  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    # Check type (categorical)
                    schema_dtype = pd.CategoricalDtype(attributes['bounds']['allowed'],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                                                       ordered=True)
                    errors.extend(self._validate_dtype(data, feature, schema_dtype,
                                                       coerced_df, coerce=coerce))
                # Check type (float)
                elif attributes.get('decimal_places', 0) > 0:
                    errors.extend(self._validate_dtype(data, feature, 'float64',
                                                       coerced_df, coerce=coerce))
                # Check type (nullable Int)
                elif self._allows_null(attributes):
                    errors.extend(self._validate_dtype(data, feature, nullable_int_dtype,
                                                       coerced_df, coerce=coerce))
                # Check type (int)
                else:
                    errors.extend(self._validate_dtype(data, feature, 'int64',
                                                       coerced_df, coerce=coerce))

            # Check continuous types
            else:
                if 'date_time_format' in attributes:
                    # Check type (datetime)
                    errors.extend(self._validate_dtype(data, feature, 'datetime64',
                                                       coerced_df, coerce=coerce,
                                                       localize_datetimes=localize_datetimes))

                # Check semi-structured type (object)
                elif attributes.get("data_type") in {"json", "yaml", "amalgam", "string", "string_mixable"}:
                    errors.extend(self._validate_dtype(data, feature, "object", coerced_df, coerce=coerce))

                # Check type (float)
                elif attributes.get('decimal_places', -1) > 0:
                    errors.extend(self._validate_dtype(data, feature, 'float64',
                                                       coerced_df, coerce=coerce))
                # Check type (nullable Int)
                elif self._allows_null(attributes):
                    errors.extend(self._validate_dtype(data, feature, nullable_int_dtype,
                                                       coerced_df, coerce=coerce))
                # Check type (int)
                elif attributes.get('decimal_places', -1) == 0:
                    errors.extend(self._validate_dtype(data, feature, 'int64',
                                                       coerced_df, coerce=coerce))
                elif attributes.get('data_type') == 'number':
                    # If feature is continuous and not a datetime, it should have a numeric data_type.
                    # If it cannot be casted to a float, then add an error.
                    if len(self._validate_dtype(data, feature, 'float64',
                                                coerced_df, coerce=True)):
                        errors.extend([f"Feature '{feature}' should be numeric"
                                       " when 'type' is 'continuous' and "
                                       "'data_type' is 'number'."])

            # Check feature bounds
            if validate_bounds:
                errors.extend(self._validate_bounds(data, feature, attributes))

        if errors:
            msg = ('Failed to validate DataFrame against feature attributes due to the '
                   'following errors:\n')
            for error in errors:
                msg = msg + f'{error}\n'
            if raise_errors:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        if coerce:
            return coerced_df

        return None

    def validate(self, data: t.Any, coerce: bool = False, raise_errors: bool = False, validate_bounds: bool = True,
                 allow_missing_features: bool = False, localize_datetimes: bool = True) -> None | pd.DataFrame:
        """
        Validate the given data against this FeatureAttributes object.

        Check that feature bounds and data types loosely describe the data. Optionally
        attempt to coerce the data into conformity.

        Parameters
        ----------
        data : Any
            The data to validate
        coerce : bool (default False)
            Whether to attempt to coerce DataFrame columns into correct data types. Coerced
            datetimes will be localized to UTC.
        raise_errors : bool (default False)
            If True, raises a ValueError if nonconforming columns are found; else issue a warning
        validate_bounds : bool (default True)
            Whether to validate the data against the attributes' inferred bounds
        allow_missing_features : bool (default False)
            Allows features that are missing from the DataFrame to be ignored
        localize_datetimes : bool (default True)
            Whether to localize datetime features to UTC.

        Returns
        -------
        None | DataFrame
            None or the coerced DataFrame if 'coerce' is True and there were no errors.
        """
        raise NotImplementedError()

    @staticmethod
    def merge(attributes: dict[str, dict], entries: dict[str, dict]) -> FeatureAttributesBase:
        """
        Update the given attributes with one or more new entries such that types are preserved.

        Do not overwrite preexisting feature types if they exist. Other attributes will be merged
        regardless of their current values. Performs basic validation of incoming feature types.

        Parameters
        ----------
        attributes: dict of str to dict
            A feature attributes dictionary to accept new entries.
        entries: dict of str to dict
            The new feature attributes entries to validate and set, where keys are feature
            names and values are feature attributes.

        Returns
        -------
        FeatureAttributesBase
            A dict-like FeatureAttributesBase instance that is the merged result of the inputs.

        Raises
        ------
        ValueError
            If any provided feature types are invalid.
        """
        # Avoid circular import
        from howso.utilities import validate_features
        # Make copies
        attributes = deepcopy(attributes)
        entries = deepcopy(entries)
        # Do basic type validation
        validate_features(entries)
        # Compare to existing attributes
        for feature_name in entries.keys():
            orig_type = attributes.get(feature_name, {}).get('type')
            new_type = entries[feature_name].get('type')
            # TODO 22059: Allow ordinals here when we can attempt to infer values
            if new_type == 'ordinal' and not (
                attributes.get(feature_name, {}).get('bounds', {}).get('allowed') or
                entries.get(feature_name, {}).get('bounds', {}).get('allowed')
            ):
                raise ValueError('Inference of ordinal values is not yet supported. Please '
                                 'preset ordinal features with their ordered values using '
                                 '`ordinal_feature_values`.')
            # Sanity check: booleans must be nominal
            elif entries[feature_name].get('data_type') == 'boolean' and orig_type and orig_type != 'nominal':
                warnings.warn(
                    f'Feature "{feature_name}" was preset as {orig_type} '
                    'but was detected to be a boolean. Booleans '
                    'must be "nominal", thus the type override will be ignored.'
                )
            # In otherwise valid cases, ensure that existing types are not overwritten
            elif orig_type and new_type:
                del entries[feature_name]['type']
            # Finally, update the dict with all remaining attributes
            if feature_name not in attributes.keys():
                attributes[feature_name] = entries[feature_name]
            else:
                attributes[feature_name].update(entries[feature_name])

        return attributes


class MultiTableFeatureAttributes(FeatureAttributesBase):
    """A dict-like object containing feature attributes for multiple tables."""

    pass


class SingleTableFeatureAttributes(FeatureAttributesBase):
    """A dict-like object containing feature attributes for a single table or DataFrame."""

    @singledispatchmethod
    def validate(data: t.Any, **kwargs):
        """
        Validate the given single table data against this FeatureAttributes object.

        Check that feature bounds and data types loosely describe the data. Optionally
        attempt to coerce the data into conformity.

        Parameters
        ----------
        data : Any
            The data to validate (single table only).
        coerce : bool, default False
            Whether to attempt to coerce DataFrame columns into correct data types.
        raise_errors : bool, default False
            If True, raises a ValueError if nonconforming columns are found; else, issue a warning.
        validate_bounds : bool, default True
            Whether to validate the data against the attributes' inferred bounds.
        allow_missing_features : bool, default False
            Allows features that are missing from the DataFrame to be ignored.
        localize_datetimes : bool, default True
            Whether to localize datetime features to UTC.
        nullable_int_dtype : str or dtype or ExtensionDtype, default 'Int64'
            A NumPy Dtype, Pandas Dtype extension object, or string representation thereof to
            attempt to use when a feature is detected to be an integer and `allow_null=True`
            in its feature attributes.

        Returns
        -------
        None | DataFrame
            None or the coerced DataFrame if 'coerce' is True and there were no errors.
        """
        raise NotImplementedError("'data' is an unsupported type")

    @validate.register
    def _(self, data: pd.DataFrame, coerce=False, raise_errors=False, validate_bounds=True,
          allow_missing_features=False, localize_datetimes=True,
          nullable_int_dtype: str | np.dtype | pd.api.extensions.ExtensionDtype = 'Int64'):
        return self._validate_df(data, coerce=coerce, raise_errors=raise_errors,
                                 validate_bounds=validate_bounds,
                                 allow_missing_features=allow_missing_features,
                                 localize_datetimes=localize_datetimes,
                                 nullable_int_dtype=nullable_int_dtype)

    def has_unsupported_data(self, feature_name: str) -> bool:
        """
        Return whether the given feature has data that is unsupported by Howso Engine.

        Parameters
        ----------
        feature_name: str
            The feature to check.

        Returns
        -------
        bool
            Whether feature_name was determined to have unsupported data.
        """
        return feature_name in self.unsupported

    def to_dataframe(self, *, include_all: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame of the feature attributes.

        Among other reasons, this is useful for presenting feature attributes
        in a Jupyter notebook or other medium.

        Returns
        -------
        pandas.DataFrame
            A DataFrame representation of the inferred feature attributes.
        """
        sep = '|'
        key_order = [
            "sample",
            "type",
            "date_time_format",
            "decimal_places",
            "significant_digits",
            "bounds",
            "data_type",
            "non_sensitive",
        ]

        # Ensure that these keys are available and reduced to an iterable of
        # only the unique values.
        all_keys = {k: None for a in self.values() for k in a.keys()}.keys()
        key_order = [k for k in key_order if k in all_keys]

        # Ensure we include extra keys not in the above list, also maintained as
        # only the unique values.
        extra_keys = {
            k: None for a in self.values() for k in a.keys()
            if k not in key_order
        }.keys()
        key_order.extend(sorted(extra_keys))

        frames = []
        for feature, attributes in self.items():
            # Create a DataFrame from the nested dictionary
            df = pd.json_normalize(attributes, sep=sep)
            # Update the column names to create a MultiIndex
            df.columns = pd.MultiIndex.from_tuples([
                tuple(c.split(sep)) if sep in c else (c, '')
                for c in df.columns
            ])
            # Set the outer key (e.g., 'f0') as the index
            df.index = [feature]
            frames.append(df)

        # Concatenate all the DataFrames along the index
        df = pd.concat(frames)

        # Create tuples for the desired order and include sub-keys
        desired_order_tuples = []
        for col in key_order:
            # Get all sub-keys for this column
            sub_keys = df.columns.get_level_values(1)[
                df.columns.get_level_values(0) == col].unique()
            # Create a tuple for each potential sub-key
            if not len(sub_keys):
                # Just the main column key if no sub-keys
                desired_order_tuples.append((col, ""))
            else:
                for sub_key in sub_keys:
                    desired_order_tuples.append((col, sub_key))

        # Reorder the columns based on the desired order tuples
        return df.loc[:, desired_order_tuples]


class InferFeatureAttributesBase(ABC):
    """
    This is an abstract Feature Attributes inferrer base class.

    It is agnostic to the type of data being inspected.
    """

    warnings_collector: IFAWarningCollector = IFAWarningCollector()
    suggestions_collector: IFASuggestionCollector = IFASuggestionCollector()

    def _process(self,
                 attempt_infer_extended_nominals: bool = False,
                 max_distilled_cases: int | None = None,
                 datetime_feature_formats: dict | None = None,
                 default_time_zone: str | None = None,
                 dependent_features: dict[str, list[str]] | None = None,
                 fanout_feature_map: dict[tuple[str] | str, list[str]] | None = None,
                 id_feature_name: str | Iterable[str] | None = None,
                 include_extended_nominal_probabilities: bool = False,
                 include_sample: bool = False,
                 infer_bounds: bool = True,
                 max_rows_to_eval: int = 10_000_000,
                 max_workers: int | None = None,
                 memory_warning_threshold: int | None = 512,
                 mode_bound_features: Iterable[str] | None = None,
                 num_series: int = 1,
                 nominal_substitution_config: dict[str, dict] | None = None,
                 ordinal_feature_values: dict[str, list[str]] | None = None,
                 preserve_rare_values_map: PreserveRareValuesMap | t.Literal["all", "off"] | None = None,
                 preserve_rare_values_config: PreserveRareValuesConfig | FullPreserveRareValuesConfig | None = None,
                 significance_threshold: int = SIGNIFICANT_THRESHOLD_DEFAULT,
                 tight_bounds: Iterable[str] | None = None,
                 types: dict[str, str] | dict[str, MutableSequence[str]] | None = None,
                 ) -> dict:
        """
        Get inferred feature attributes for the parameters.

        See ``infer_feature_attributes`` for full docstring.
        """
        self.attributes: FeatureAttributesBase = FeatureAttributesBase({})

        self.max_rows_to_eval = max_rows_to_eval

        if datetime_feature_formats is None:
            datetime_feature_formats = dict()

        self.datetime_feature_formats = datetime_feature_formats

        # If not set by an external caller (e.g., InferFeatureAttributesTimeSeries), set a default
        if not hasattr(self, "_time_invariant_features"):
            self._time_invariant_features = []

        if ordinal_feature_values is None:
            ordinal_feature_values = dict()

        if dependent_features is None:
            dependent_features = dict()

        self.default_time_zone = default_time_zone

        self.num_series = num_series

        if isinstance(id_feature_name, str):
            self.id_feature_names = [id_feature_name]
        elif isinstance(id_feature_name, Iterable):
            self.id_feature_names = id_feature_name
        elif id_feature_name is not None:
            raise ValueError("ID feature must be of type `str` or `list[str], "
                             f"not {type(id_feature_name)}.")
        else:
            self.id_feature_names = []

        # Preprocess user-defined feature types
        preset_types = {}
        # Check the `types` argument
        if types:
            # Can be either str -> str or str -> Iterable[str]
            for k, v in types.items():
                if isinstance(v, MutableSequence):
                    for feat_name in v:
                        # The feature might not be present if this is executed under multiprocessing
                        if feat_name in self.data.columns:
                            preset_types[feat_name] = {"type": k}
                # The feature might not be present if this is executed under multiprocessing
                elif k in self.data.columns:
                    preset_types[k] = {"type": v}

        # Make updates with the `merge` function
        merge = FeatureAttributesBase.merge

        # If any ordinals were specified in *both* `types` and `ordinal_feature_values`,
        # set the bounds from `ordinal_feature_values` first else `merge` will raise an
        # error about missing bounds.
        pre_processed_ordinals = []
        for feat_name in preset_types:
            if feat_name in ordinal_feature_values:
                self.attributes = merge(self.attributes, {feat_name: {
                    "type": "ordinal",
                    "bounds": {"allowed": ordinal_feature_values[feat_name]}
                }})
                pre_processed_ordinals.append(feat_name)

        # Update the feature attributes dictionary with the user-defined base types
        self.attributes = merge(self.attributes, preset_types)

        feature_names_list = self._get_feature_names()
        for feature_name in feature_names_list:
            # What type is this feature?
            feature_type, typing_info = self._get_feature_type(feature_name)

            typing_info = typing_info or dict()

            # EXPLICITLY DECLARED ORDINALS
            if feature_name in ordinal_feature_values:
                if feature_name not in pre_processed_ordinals:
                    self.attributes = merge(self.attributes, {feature_name: {
                        "type": "ordinal",
                        "bounds": {"allowed": ordinal_feature_values[feature_name]}
                    }})

            # EXPLICITLY DECLARED DATETIME & TIME FEATURES
            elif self.datetime_feature_formats.get(feature_name, None):
                # datetime_feature_formats is expected to either be only a
                # single string (format) or a tuple of strings (format, locale)
                user_dt_format = self.datetime_feature_formats[feature_name]
                # If a datetime format is defined, first ensure values can be parsed with it
                test_value = self._get_random_value(feature_name, no_nulls=True)
                if test_value is not None and not is_valid_datetime_format(
                    test_value, user_dt_format[0] if isinstance(user_dt_format, tuple) else user_dt_format
                ):
                    raise ValueError(
                        f'The date time format "{user_dt_format}" does not match the data of feature '
                        f'"{feature_name}". Data sample: "{test_value}"')

                if feature_type == FeatureType.DATETIME:
                    # When feature is a datetime instance, we won't need to
                    # parse the datetime from a string using a custom format.
                    self.attributes = merge(self.attributes, {
                        feature_name: self._infer_datetime_attributes(feature_name)})
                    warnings.warn(
                        'Providing a datetime feature format for the feature '
                        f'"{feature_name}" is not necessary because the data '
                        'is already formatted as a datetime object. This '
                        'custom format will be ignored.')
                elif feature_type == FeatureType.DATE:
                    # When feature is a date instance, we won't need to
                    # parse the datetime from a string using a custom format.
                    self.attributes = merge(self.attributes, {
                        feature_name: self._infer_date_attributes(feature_name)})
                    warnings.warn(
                        'Providing a datetime feature format for the feature '
                        f'"{feature_name}" is not necessary because the data '
                        'is already formatted as a date object. This custom '
                        'format will be ignored.')
                elif feature_type == FeatureType.TIME:
                    self.attributes = merge(self.attributes, {
                        feature_name: self._infer_time_attributes(feature_name, user_dt_format)})
                elif isinstance(user_dt_format, str):
                    # User passed only the format string
                    # First see if it is likely a time-only feature
                    if (not any(date_id in user_dt_format
                        for date_id in DATE_TOKENS)
                            and any(time_id in user_dt_format
                                    for time_id in TIME_TOKENS)):
                        self.attributes = merge(self.attributes, {
                            feature_name: self._infer_time_attributes(feature_name, user_dt_format)})
                    else:
                        self.attributes = merge(self.attributes, {feature_name: {
                            "type": "continuous",
                            "data_type": "formatted_date_time",
                            "date_time_format": user_dt_format,
                        }})
                elif (
                    isinstance(user_dt_format, Collection) and
                    len(user_dt_format) == 2
                ):
                    # User passed format string and a locale string
                    dt_format, dt_locale = user_dt_format
                    self.attributes = merge(self.attributes, {feature_name: {
                        "type": "continuous",
                        "data_type": "formatted_date_time",
                        "date_time_format": dt_format,
                        "locale": dt_locale,
                    }})
                else:
                    # Not really sure what they passed.
                    raise TypeError(
                        f'The value passed (`{user_dt_format}`) to '
                        f'`datetime_feature_formats` for feature "{feature_name}"'
                        f'is invalid. It should be either a single string '
                        f'(format), or a tuple of 2 strings (format, locale).')

            # FLOATING POINT FEATURES
            elif feature_type == FeatureType.NUMERIC:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_floating_point_attributes(feature_name)})

            # IMPLICITLY DEFINED DATETIME FEATURES
            elif feature_type == FeatureType.DATETIME:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_datetime_attributes(feature_name)})

            # DATE ONLY FEATURES
            elif feature_type == FeatureType.DATE:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_date_attributes(feature_name)})

            # TIME ONLY FEATURES
            elif feature_type == FeatureType.TIME:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_time_attributes(feature_name)})

            # TIMEDELTA FEATURES
            elif feature_type == FeatureType.TIMEDELTA:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_timedelta_attributes(feature_name)})

            # INTEGER FEATURES
            elif feature_type == FeatureType.INTEGER:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_integer_attributes(feature_name)})

            # BOOLEAN FEATURES
            elif feature_type == FeatureType.BOOLEAN:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_boolean_attributes(feature_name)})

            # ALL OTHER FEATURES
            else:
                self.attributes = merge(self.attributes, {
                    feature_name: self._infer_string_attributes(feature_name)})

            # Is column constrained to be unique?
            if self._has_unique_constraint(feature_name):
                self.attributes[feature_name]["unique"] = True

            # Add original type to feature if not already set
            if not self.attributes[feature_name].get("original_type"):
                if original_type := typing_info.pop("original_type", None):
                    self.attributes[feature_name]["original_type"] = {
                        "data_type": str(original_type),
                        **typing_info
                    }
                elif feature_type is not None:
                    self.attributes[feature_name]["original_type"] = {
                        "data_type": str(feature_type),
                        **typing_info
                    }

            # DECLARED DEPENDENTS
            # First determine if there are any dependent features in the partial features dict
            # Set dependent features: `dependent_features` + partial features dict, if provided
            if feature_name in dependent_features:
                self.attributes[feature_name]["dependent_features"] = dependent_features[feature_name]

            # Set default time if provided
            if self.default_time_zone is not None:
                self.attributes[feature_name]["default_time_zone"] = self.default_time_zone

        # Edit ID feature attributes in-place
        for id_feature in self.id_feature_names:
            self._add_id_attribute(self.attributes, id_feature)

        if infer_bounds:
            for feature_name, _attributes in self.attributes.items():
                # If multiprocessing is enabled, this InferFeatureAttributes instance may not have
                # access to all columns in the data, though they could still be present in the
                # attributes dictionary in some circumstances.
                if feature_name not in self.data.columns:
                    continue
                # Don't infer bounds for JSON/YAML features
                if _attributes.get("data_type") in ["json", "yaml"]:
                    continue
                try:
                    bounds = self._infer_feature_bounds(
                        self.attributes, feature_name,
                        tight_bounds=tight_bounds,
                        mode_bound_features=mode_bound_features,
                    )
                except ValueError as err:
                    if "could not convert" in str(err):
                        # Try to catch any errors on data conversion and suggest something relevant.
                        if feature_name in preset_types:
                            suggestion = (f"Please verify that the provided type for '{feature_name}' "
                                          f"({preset_types[feature_name]['type']}) is reflected by the data.")
                        else:
                            suggestion = f"Please verify that cases in '{feature_name}' are of a consistent data type."
                        raise ValueError(f"The following error was raised while trying to compute bounds for feature "
                                         f"'{feature_name}':\n\n {err}\n\n{suggestion}") from err
                    else:
                        raise
                if bounds:
                    # Use `update` on the bounds dictionary in case `allowed` ordinal values have already been set
                    bounds.update(self.attributes[feature_name].get("bounds", {}))
                    _attributes["bounds"] = bounds

        # Do any features contain data unsupported by the core?
        self._check_unsupported_data(self.attributes)

        # Do any features in dependent relationships have many (> ~N/2) uniques?
        # If so, warn the user about result quality implications.
        self._check_dependent_features_uniqueness()

        # Check if there are any features that consume an unusually large amount of memory
        if isinstance(self.data, pd.DataFrame):
            self._check_feature_memory_use(max_size=memory_warning_threshold)

        # If requested, infer extended nominals.
        if attempt_infer_extended_nominals:
            # Attempt to import the NominalDetectionEngine.
            try:
                from howso.nominal_substitution import (
                    NominalDetectionEngine,
                )
                # Grab whether the user wants the probabilities saved in the feature
                # metadata.
                include_meta = include_extended_nominal_probabilities

                # Get the assigned extended nominal probabilities (aenp) and all
                # probabilities.
                nde = NominalDetectionEngine(nominal_substitution_config)
                aenp, all_probs = nde.detect(self.data)

                nominal_default_subtype = "int-id"
                # Apply them if they are above the threshold value.
                for feature_name in feature_names_list:
                    if feature_name in aenp:
                        if len(aenp[feature_name]) > 0:
                            self.attributes[feature_name]["subtype"] = (max(
                                aenp[feature_name], key=aenp[feature_name].get))

                        if include_meta:
                            self.attributes[feature_name].update({
                                "extended_nominal_probabilities":
                                    all_probs[feature_name]
                            })

                    # If `subtype` is a nominal feature, assign it to 'int-id'
                    if (
                        self.attributes[feature_name]["type"] == "nominal" and
                        not self.attributes[feature_name].get("subtype", None)
                    ):
                        self.attributes[feature_name]["subtype"] = (
                            nominal_default_subtype)
            except ImportError:
                warnings.warn("Cannot infer extended nominals: not supported")

        # Insert a ``sample`` value (as string) for each feature, if possible.
        if include_sample:
            for feature_name in self.attributes:
                sample = self._get_random_value(feature_name, no_nulls=True)
                if sample is not None:
                    sample = str(sample)
                self.attributes[feature_name]["sample"] = sample

        # Validate datetimes after any user-defined features have been re-implemented
        self._validate_date_times()

        # Configure the fanout feature attributes according to the input if given.
        if fanout_feature_map:
            for key_features, fanout_features in fanout_feature_map.items():
                if isinstance(key_features, str):
                    key_features = [key_features]
                for f in fanout_features:
                    if f in self.attributes:
                        self.attributes[f]["fanout_on"] = list(key_features)

        self._process_rare_values(preserve_rare_values_map, preserve_rare_values_config, max_distilled_cases,
                                  significance_threshold)

        # Re-order the keys like the original dataframe
        ordered_attributes = {}
        for fname in self.data.columns:
            # Check to see if the key is a sqlalchemy Column
            if hasattr(fname, "name"):
                fname = fname.name
            if fname not in self.attributes:
                warnings.warn(f"Feature {fname} exists in provided data but was not computed in feature attributes.")
                continue
            ordered_attributes[fname] = self.attributes[fname]

        return ordered_attributes

    @abstractmethod
    def __call__(self) -> FeatureAttributesBase:
        """Process and return the feature attributes."""

    @abstractmethod
    def _infer_floating_point_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given floating-point column."""

    @abstractmethod
    def _infer_datetime_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given date-time column."""

    @abstractmethod
    def _infer_date_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given date only column."""

    @abstractmethod
    def _infer_time_attributes(self, feature_name: str, user_time_format: str | None = None) -> dict:
        """Get inferred attributes for the given time column."""

    @abstractmethod
    def _infer_timedelta_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given timedelta column."""

    @abstractmethod
    def _infer_boolean_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given boolean column."""

    @abstractmethod
    def _infer_integer_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given integer column."""

    def _infer_string_attributes(self, feature_name: str) -> dict:
        """Get inferred attributes for the given string column."""
        # Column has arbitrary string values, first check if they
        # are ISO8601 datetimes.
        if self._is_iso8601_datetime_column(feature_name):
            # if datetime, determine the iso8601 format it's using
            if first_non_null := self._get_first_non_null(feature_name):
                fmt = determine_iso_format(first_non_null, feature_name)
                return {
                    "type": "continuous",
                    "data_type": "formatted_date_time",
                    "date_time_format": fmt
                }
            else:
                # It isn't clear how this method would be called on a feature
                # if it has no data, but just in case...
                return {
                    "type": "continuous",
                    "data_type": "number",
                }
        elif self._is_json_feature(feature_name):
            typing_attrs = {
                "type": "continuous",
                "data_type": "json",
            }
            first_non_null = self._get_first_non_null(feature_name)
            if isinstance(first_non_null, (Set, Sequence, Mapping)) and not isinstance(first_non_null, (str, bytes)):
                typing_attrs["original_type"] = {"data_type": FeatureType.CONTAINER.value}
                if isinstance(first_non_null, Set):
                    typing_attrs["original_type"]["coercion"] = "set"
            return typing_attrs
        elif self._is_yaml_feature(feature_name):
            return {
                "type": "continuous",
                "data_type": "yaml"
            }
        else:
            # The user may have pre-set the type as "continuous" to force it to be considered a tokenizable string;
            # but that may also be the case for string ints or floats. Check that first.
            is_tokenizable_string = False
            if self.attributes.get(feature_name, {}).get("type") == "continuous":
                try:
                    # If the column can be converted to float, and was set to be "continuous",
                    # it is probably not a tokenizable string.
                    col = self.data[feature_name]
                    col.astype("float")
                except Exception:  # noqa: Intentionally broad
                    # If it cannot be converted to float, but it was set to be "continuous",
                    # it is probably a tokenizable string.
                    is_tokenizable_string = True
            if is_tokenizable_string:
                return {
                    "type": "continuous",
                    "data_type": "json",
                    # Also set the original_type here so that we do not need to re-check _is_tokenizable_string
                    "original_type": {"data_type": FeatureType.TOKENIZABLE_STRING.value},
                }
            else:
                return self._infer_unknown_attributes(feature_name)

    def _infer_unknown_attributes(self, *args: t.Any) -> dict:
        """Get inferred attributes for the given unknown-type column."""
        return {
            "type": "nominal",
            "data_type": "string",
        }

    @abstractmethod
    def _infer_feature_bounds(
        self,
        feature_attributes: Mapping[str, Mapping],
        feature_name: str,
        tight_bounds: Iterable[str] | None = None,
        mode_bound_features: Iterable[str] | None = None,
    ) -> dict | None:
        """
        Return inferred bounds for the given column.

        Features with datetimes are converted to seconds since epoch and their
        bounds are calculated accordingly. Features with timedeltas are
        converted to total seconds.

        Parameters
        ----------
        feature_attributes : dict
            A dictionary of feature names to a dictionary of parameters.
        feature_name : str
            The name of feature to infer bounds for.
        tight_bounds: Iterable of str, default None
            Set tight min and max bounds for the features specified in
            the Iterable.
        mode_bound_features : list of str, optional
            Explicit list of feature names that should use mode bounds. When
            None, uses all features.

        Returns
        -------
        dict or None
            Dictionary of bounds for the specified feature, or None if no
            bounds.
        """

    @staticmethod
    def infer_loose_feature_bounds(min_bound: float,
                                   max_bound: float
                                   ) -> tuple[float, float]:
        """
        Infer the loose bound values given a tight min and max bound value.

        Parameters
        ----------
        min_bound : int or float
            The minimum value in a dataset for a feature, must be equal to or less
            than the max value
        max_bound : int or float
            The maximum value in a dataset for a feature, must be equal to or more
            than the min value

        Returns
        -------
        tuple
            Tuple (min_bound, max_bound) of loose bounds around the provided tight
            min and max_bound bounds
        """
        if min_bound > max_bound:
            raise AssertionError(
                "Feature min_bound cannot be larger than max_bound."
            )
        scale_factor = 0.5
        value_range = max_bound - min_bound
        if value_range == 0.0:
            new_range = np.exp(scale_factor)
        else:
            new_range = np.exp(np.log(value_range) + scale_factor)

        base_min_bound = max_bound - new_range
        base_max_bound = min_bound + new_range

        new_min_bound = max(0, base_min_bound) if min_bound >= 0 else base_min_bound
        new_max_bound = min(0, base_max_bound) if max_bound <= 0 else base_max_bound

        return float(new_min_bound), float(new_max_bound)

    def _get_cont_threshold(self, feature_name: str) -> int:
        """Get the minimum number of unique values a feature must have to be considered continuous."""
        n_cases = self._get_num_cases(feature_name)
        # If the provided feature is stationary, we should simply evaluate the number of series
        if getattr(self, "id_feature_names", None) and feature_name in self._time_invariant_features:
            return math.ceil(pow(self.num_series, 0.5))
        # Return the sqrt of max(avg. cases per series, num. series)
        return math.ceil(pow(max(self.num_series, (n_cases / self.num_series)), 0.5))

    @staticmethod
    def _get_datetime_max() -> str:
        # Avoid circular import
        from howso.client.client import get_howso_client_class
        from howso.direct import HowsoDirectClient
        # If on Direct, check the user's platform. Else, default to Unix.
        klass, _ = get_howso_client_class()
        if issubclass(klass, HowsoDirectClient):
            plat = platform.system().lower()
            if plat == "windows":
                return WIN_DT_MAX
        return LINUX_DT_MAX

    def _check_dependent_features_uniqueness(self) -> None:
        """
        Validate that all features that are part of a dependent relationship are not unique or near-unique.

        If any features in a dependent relationship in either direction have sufficient (~N/2) uniqueness,
        warn the user about potential result quality implications.
        """
        dependent_features = set()
        for feature in self.attributes:
            if features_to_add := self.attributes[feature].get("dependent_features"):
                dependent_features |= set(features_to_add + [feature])

        for feature in dependent_features:
            unique_count = self._get_unique_count(feature)
            case_count = self._get_num_cases(feature)
            if unique_count >= math.floor(case_count / 2):
                self.warnings_collector.triage(IFAWarningEmitterType.NEAR_UNIQUE_DEPENDENT_FEATURES, feature)

    def _check_unsupported_data(self, feature_attributes: dict) -> None:
        """
        Determine whether any features contain data that is unsupported by the core.

        Unsupported data could be a number or datetime that exceeds the min/max of the core or
        user operating system. If unsupported data is found, add the feature to an internal list
        that indicates which features should be removed before training.

        Parameters
        ----------
        feature_attributes : Dict
            A feature attributes dictionary.
        """
        # Avoid circular import
        from howso.utilities import date_to_epoch
        feature_names = list(feature_attributes.keys())
        for feature_name in feature_names:
            # Cyclic time features won't have unsupported data as they cannot exceed 24hour bounds
            if feature_attributes[feature_name].get("data_type") == "formatted_time":
                continue
            # Check original data type for ints, floats, datetimes
            orig_type = feature_attributes[feature_name].get("original_type", {}).get("data_type")
            if (orig_type in ["integer", "numeric"] or "date_time_format" in
                    feature_attributes[feature_name]):
                # Get feature bounds
                with warnings.catch_warnings():
                    # Prevent duplication of raised warnings, since we do not nee to raise them here
                    # and infer bounds was likely already called previously
                    warnings.simplefilter("ignore")
                    bounds = self._infer_feature_bounds(
                        feature_attributes,
                        feature_name=feature_name,
                        tight_bounds=feature_names
                    )
                if not bounds or bounds.get("min") is None or bounds.get("max") is None:
                    continue
                omit = False
                # Datetimes
                dt_fmt = feature_attributes[feature_name].get("date_time_format")
                if dt_fmt is not None:
                    # Get maximum compatible datetime (depends on platform)
                    allowed_max = date_to_epoch(self._get_datetime_max(), time_format="%Y-%m-%d")
                    actual_max = date_to_epoch(bounds["max"], time_format=dt_fmt)
                    # Verify
                    if actual_max >= allowed_max:
                        omit = True
                else:
                    # Determine the largest absolute value from the feature bounds
                    largest_value = max(abs(bounds["min"]), bounds["max"])
                    # Verify integer min/max
                    if orig_type == "integer":
                        if largest_value >= INTEGER_MAX:
                            omit = True
                    # Verify float min/max
                    elif orig_type == "numeric":
                        # Determine the smallest absolute value from the feature bounds
                        smallest_value = min(abs(bounds["min"]), abs(bounds["max"]))
                        if largest_value >= FLOAT_MAX or smallest_value <= FLOAT_MIN:
                            omit = True
                # Keep track of unsupported data internally
                if omit:
                    self.unsupported.append(feature_name)

    def _validate_date_times(self) -> None:
        """Validate date time features are configured correctly."""
        for feature_name, attributes in self.attributes.items():
            dt_format = attributes.get("date_time_format")
            data_type = attributes.get("data_type")
            if not dt_format and data_type in {"formatted_date_time", "formatted_time"}:
                raise ValueError(
                    f'The feature "{feature_name}" must have a `date_time_format` defined '
                    f'when its `data_type` is "{data_type}".'
                )
            elif dt_format and data_type in {"formatted_date_time", "formatted_time"}:
                # If the date/time format does not include a time zone, warn the user that
                # the default of UTC will be used. However, due to potential multiprocessing,
                # and to avoid an excess of warnings if done per-feature, stash the offending
                # features and do a single warning later on.
                if not any(["%z" in dt_format,
                            "%Z" in dt_format,
                            dt_format[-1] == "Z",  # Last char of 'Z' is ISO8601 identifier for UTC
                            self.default_time_zone is not None]):
                    self.warnings_collector.triage(IFAWarningEmitterType.MISSING_TZ_FEATURES, feature_name)
                elif "%z" in dt_format:
                    rand_val = self._get_random_value(feature_name)
                    if isinstance(rand_val, datetime.datetime):
                        # Some datetime objects might have a time zone attribute not visible as a string
                        if getattr(rand_val, 'tzinfo', None) is not None and isinstance(rand_val.tzinfo, ZoneInfo):
                            continue
                    # Warn in case of UTC offset -- could lead to unexpected results due to time zone
                    # differences
                    self.warnings_collector.triage(IFAWarningEmitterType.UTC_OFFSET, feature_name)

    @staticmethod
    def _is_datetime(string: str) -> bool:
        """
        Return True if string can be interpreted as a date.

        Parameters
        ----------
        string : str
            The string to check.

        Returns
        -------
        True if string is a date, False if not.
        """
        # Return False if string contains only letters.
        if isinstance(string, str) and string.isalpha():
            return False
        try:
            # if the string is a number, it's not a datetime
            float(string)
            return False
        except (TypeError, ValueError):
            pass

        try:
            dt_parse(string)
            return True
        except Exception:  # noqa: Intentionally broad
            return False

    def _is_iso8601_datetime_column(self, feature: str) -> bool:
        """
        Return whether the given feature contains ISO 8601 datetimes.

        Parameters
        ----------
        feature : string
            The feature to check the values of.

        Returns
        -------
        True if the column values can be parsed into an ISO 8601 datetime
        """
        first_non_none = self._get_first_non_null(feature)
        if first_non_none is None:
            return False

        # Pick another value and test if it's also a date to make sure
        # that the first one wasn't a date by accident, for example
        # if the column is 'miscellaneous notes' or 'comment', it's
        # possible for it to have a datetime as the sole value, but the
        # column itself is not actually a datetime type
        rand_val = self._get_random_value(feature, no_nulls=True)

        # Try to parse one or both values as strictly iso8601
        try:
            if not self._is_datetime(first_non_none):
                return False
            isoparse(first_non_none)
            if rand_val is not None:
                if not self._is_datetime(rand_val):
                    return False
                isoparse(rand_val)
        except Exception:  # noqa: Intentionally broad
            return False

        # No issues; it's valid
        return True

    def _is_boolean_feature(self, feature: str) -> bool:
        """
        Return whether the given feature is a bool object or "true"/"false" string.

        Parameters
        ----------
        feature: string
            The feature to check the values of.

        Returns
        -------
        True if the column values can be parsed into a boolean.
        """
        # Sample 30 random values
        random_values = self._get_random_value(feature, no_nulls=True, count=30)
        if not random_values:
            return False
        for random_value in random_values:
            # Check for a Python bool object or a string representation thereof
            if not isinstance(random_value, bool) and not (isinstance(random_value, str) and
                                                           random_value.strip().lower() in ("true", "false")):
                return False
        return True

    def _is_json_feature(self, feature: str) -> bool:
        """
        Return whether the given feature contains valid JSON.

        Parameters
        ----------
        feature: string
            The feature to check the values of.

        Returns
        -------
        True if the column values can be parsed into JSON.
        """
        first_non_none = self._get_first_non_null(feature)
        if first_non_none is None:
            return False

        # Sample 30 random values
        for _ in range(30):
            rand_val = self._get_random_value(feature, no_nulls=True)
            if rand_val is None:
                return False

            # Try to parse rand_val as JSON
            try:
                # We can handle sets by converting them to lists and letting the Engine know
                if isinstance(rand_val, Set):
                    json.dumps(list(rand_val))
                # Python objects and lists are valid JSON
                elif not isinstance(rand_val, str):
                    json.dumps(rand_val)
                else:
                    if not any(c in rand_val for c in "{}[]"):
                        return False
                    json.loads(rand_val)
            except (TypeError, json.JSONDecodeError):
                return False

        # No exception: valid JSON
        return True

    def _is_yaml_feature(self, feature: str) -> bool:
        """
        Return whether the given feature contains valid YAML.

        Parameters
        ----------
        feature: string
            The feature to check the values of.

        Returns
        -------
        True if the column values can be parsed into YAML.
        """
        # If there is no data, return False
        first_non_none = self._get_first_non_null(feature)
        if first_non_none is None:
            return False

        # Sample up-to 30 random values
        for _ in range(30):
            sample = self._get_random_value(feature, no_nulls=True)

            # Non-string types are not valid YAML documents on their own for
            # the sake of infer_feature_attributes.
            if not isinstance(sample, str):
                return False

            # Try to parse rand_val as YAML
            try:
                yaml.safe_load(sample)
                if len(sample.split(":")) <= 1 or "\n" not in sample:
                    return False
            except yaml.YAMLError:
                return False

        return True

    @staticmethod
    def _add_id_attribute(feature_attributes: Mapping, id_feature_name: str) -> None:
        """Update the given feature_attributes in-place for id_features."""
        if id_feature_name in feature_attributes:
            feature_attributes[id_feature_name]["id_feature"] = True
            # If id feature was inferred to be continuous, change it to nominal
            # with 'data_type':number attribute to prevent string conversion.
            if feature_attributes[id_feature_name]["type"] == "continuous":
                feature_attributes[id_feature_name]["type"] = "nominal"
                if "decimal_places" in feature_attributes[id_feature_name]:
                    del feature_attributes[id_feature_name]["decimal_places"]

    @classmethod
    def _get_min_max_number_size_bounds(
        cls, feature_attributes: Mapping,
        feature_name: str
    ) -> tuple[float | int | None, float | int | None]:
        """
        Get the minimum and maximum size bounds for a numeric feature.

        The minimum and maximum value is based on the storage size of the
        number obtained from the "original_type" feature attribute, i.e. for a
        8bit integer: min=-128, max=127.

        .. NOTE::
            Bounds will not be returned for 64bit floats since this is the
            maximum supported numeric size, so no bounds are necessary.

        Parameters
        ----------
        feature_attributes : dict
            A dictionary of feature names to a dictionary of parameters.
        feature_name : str
            The name of feature.

        Returns
        -------
        Number or None
            The minimum size.
        Number or None
            The maximum size.
        """
        try:
            original_type = feature_attributes[feature_name]["original_type"]
        except (TypeError, KeyError):
            # Feature not found or original typing info not defined
            return None, None

        min_value = None
        max_value = None
        if original_type and original_type.get("size"):
            size = original_type.get("size")
            data_type = original_type.get("data_type")

            if size in [1, 2, 4, 8]:
                if data_type == FeatureType.INTEGER.value:
                    if original_type.get("unsigned"):
                        dtype_info = np.iinfo(f"uint{size * 8}")
                    else:
                        dtype_info = np.iinfo(f"int{size * 8}")
                elif data_type == FeatureType.NUMERIC.value and size < 8:
                    dtype_info = np.finfo(f"float{size * 8}")
                else:
                    # Not a numeric feature or is 64bit float
                    return None, None

                min_value = float(dtype_info.min)
                max_value = float(dtype_info.max)
            elif size == 3 and data_type == FeatureType.INTEGER.value:
                # Some database dialects support 24bit integers
                if original_type.get("unsigned"):
                    min_value = 0
                    max_value = 16777215
                else:
                    min_value = -8388608
                    max_value = 8388607

        return min_value, max_value

    @abstractmethod
    def _get_feature_type(self, feature_name: str
                          ) -> tuple[FeatureType | None, dict | None]:
        """
        Return the type information for a given feature.

        Parameters
        ----------
        feature_name : str
            The name of the feature to get the type of

        Returns
        -------
        FeatureType or None
            The feature type or None if the column could not be found.
        Dict or None
            Additional typing information about the feature or None if the
            column could not be found.
        """

    @abstractmethod
    def _get_n_random_rows(self, samples: int = 5000, seed: int | None = None) -> pd.DataFrame:
        """Get random samples from the given data as a DataFrame."""

    @abstractmethod
    def _get_random_value(self, feature_name: str, no_nulls: bool = False) -> t.Any:
        """Retrieve a random value from the data."""

    @abstractmethod
    def _has_unique_constraint(self, feature_name: str) -> bool:
        """Return whether this feature has a unique constraint."""

    @abstractmethod
    def _get_first_non_null(self, feature_name: str) -> t.Any:
        """
        Get the first non-null value in the given column.

        NOTE: "first" means arbitrarily the first one that the DataFrame or database
              returned; there is no implication of ordering.
        """

    @abstractmethod
    def _get_num_features(self) -> int:
        """Get the number of features/columns in the data."""

    @abstractmethod
    def _get_num_cases(self, feature_name: str) -> int:
        """Get the number of non-null cases of the provided feature."""

    @abstractmethod
    def _get_feature_names(self) -> list[str]:
        """Get the names of the features/columns of the data."""

    @abstractmethod
    def _get_unique_count(self, feature_name: str | Iterable[str]) -> int:
        """Get the number of unique values in the provided feature(s)."""

    @abstractmethod
    def _get_unique_values(self, feature_name: str) -> Collection[t.Any]:
        """Get a set of the unique values for the given feature_name."""

    @abstractmethod
    def _get_row_count(self) -> int:
        """Get the total number of rows in the data."""

    @abstractmethod
    def _get_value_count(self, feature_name: str, value: t.Any) -> int:
        """Get the number of occurrences of the provided value of the provided feature."""

    def _find_protected_value_candidates(self, max_distilled_cases: int,
                                         significance_threshold: int) -> tuple[PreserveRareValuesMap, list[dict]]:
        """
        Analyze the data to determine if any values might be good candidates for signal preservation techniques.

        Parameters
        ----------
        max_distilled_cases : int
            The maximum number of cases in the resultant data following distillation.
        significance_threshold : int
            The number of cases that are expected to result in a maintained signal for a particular
            value post-distillation.

        Returns
        -------
        PreserveRareValuesMap
            A Mapping of feature name to list of rare value candidates.
        List of dict
            An ordered list of dict of the top 5 most significant rare values in the orignal data.
        """
        pvm: PreserveRareValuesMap = {}
        value_counts = []
        for feature, attributes in self.attributes.items():
            if attributes["type"] != "nominal":
                continue
            total_cases = self._get_row_count()
            if self._get_unique_count(feature) == total_cases:
                # Don't make a suggestion for a completely unique feature
                continue
            uniques = self._get_unique_values(feature)
            for unique_value in uniques:
                try:
                    count = self._get_value_count(feature, unique_value)
                except TypeError:
                    self.warnings_collector.triage(IFAWarningEmitterType.VALUE_COUNTS_PROCESSING, feature)
                    continue
                # Don't include values that aren't significant to begin with
                if count < significance_threshold:
                    continue
                expected_freq_at_target_size = (max_distilled_cases / total_cases) * count
                if expected_freq_at_target_size < significance_threshold:
                    value_counts.append({"feature": feature, "value": unique_value, "count": count})
                    if feature not in pvm:
                        pvm[feature] = [unique_value]
                    else:
                        pvm[feature].append(unique_value)
        top_five = sorted(value_counts, key=lambda d: d["count"], reverse=True)[:5]
        return pvm, top_five

    def _compute_unprotected_multiplier(self, feature: str, protected_values_multipliers: Sequence[dict[str, t.Any]],
                                        *, row_count: int | None = None) -> float:
        """
        Compute the unprotected multiplier for the provided feature given a list of rare values with multipliers.

        Parameters
        ----------
        feature : str
            The name of the feature to compute the unprotected multiplier for.
        protected_values : Sequence of dict of str to Any
            A list of dicts with information about the protected values and their multipliers.
            This is of the same type as the list under the `protected_values_multipliers` key
            in the final feature attributes object.

            Example::

                [
                    {"value": "X", "multiplier": 2},
                    {"value": "Y", "multiplier": 3}
                ]
        row_count : int, default none
            (Optional) The row count of the feature. If not provided, will compute.

        Returns
        -------
        float
            The unprotected multiplier.
        """
        total_cases = row_count or self._get_row_count()
        orig_unprotected_mass = 0
        new_protected_mass = 0
        for value_cfg in protected_values_multipliers:
            count = self._get_value_count(feature, value_cfg["value"])
            orig_unprotected_mass += count
            new_protected_mass += count * value_cfg["multiplier"]
        orig_unprotected_mass = total_cases - orig_unprotected_mass
        return min(float((total_cases - new_protected_mass) / orig_unprotected_mass), 1)

    def _compute_preserve_rare_values_config(
        self,
        max_distilled_cases: int,
        preserve_rare_values_map: PreserveRareValuesMap | t.Literal["all"],
        significance_threshold: int
    ) -> FullPreserveRareValuesConfig:
        """
        Determine the case weight multipliers for the provided protected values and the unprotected values.

        Parameters
        ----------
        max_distilled_cases : int
            The maximum number of cases in the resultant data following distillation.
        preserve_rare_values_map : PreserveRareValuesmap or "all"
            A mapping of feature name to list of rare values to compute multipliers for.
            Use "all" to find rare value candidates and compute multipliers for them all.
        significance_threshold : int
            The number of cases that are expected to result in a maintained signal for a
            particular value post-distillation.

        Returns
        -------
        FullPreserveRareValuesConfig
            A full `preserve_rare_values` configuration with all multipliers ready for
            application to the feature attributes.
        """
        prvc: FullPreserveRareValuesConfig = {}
        if preserve_rare_values_map == "all":
            preserve_rare_values_map, _ = self._find_protected_value_candidates(max_distilled_cases,
                                                                                significance_threshold)
        for feature, values in preserve_rare_values_map.items():
            prvc[feature] = {"protected_values_multipliers": []}
            total_cases = self._get_row_count()
            data_type = self.attributes[feature]["data_type"] # pyright: ignore[reportTypedDictNotRequiredAccess]
            for value in values:
                count = self._get_value_count(feature, value)
                if count == 0:
                    raise ValueError(f"Specified protected value `{value}` not found in column `{feature}`. "
                                     "Please verify the value and type.")
                expected_freq_at_target_size = (max_distilled_cases / total_cases) * count
                multiplier = significance_threshold / expected_freq_at_target_size
                # If a value has a computed multiplier of < 1, it does not need signal preservation
                if multiplier < 1:
                    continue
                prvc[feature]["protected_values_multipliers"].append(
                    {"value": float(value) if data_type == "number" else value,
                    "multiplier": max(float(multiplier), 1)}
                )
            # Now that all protected value multipliers have been computed, determine the unprotected value multiplier
            prvc[feature]["unprotected_multiplier"] = self._compute_unprotected_multiplier(
                feature, prvc[feature]["protected_values_multipliers"], row_count=total_cases
            )
        return prvc

    def _process_rare_values(self, preserve_rare_values_map: PreserveRareValuesMap,  # noqa: PLR0912
                             preserve_rare_values_config: PreserveRareValuesConfig, max_distilled_cases: int,
                             significance_threshold: int) -> None:
        """Procesess `preserve_rare_values` configuration or make recommendation."""
        _prvc: FullPreserveRareValuesConfig = {}
        # Did the user specify max_distilled_cases? Save this information for later.
        user_set_mdc = False
        # User wants to do nothing; exit silently
        if preserve_rare_values_map and preserve_rare_values_map == "off":
            return
        # If available, pre-cache value counts to enhance performance
        if hasattr(self.data, "_cache_value_counts") and callable(self.data._cache_value_counts):
            feature_names = []
            total_cases = self._get_row_count()
            for feature, attributes in self.attributes.items():
                # Only cache features that are eligible for rare values
                if attributes["type"] == "nominal" and self._get_unique_count(feature) < total_cases:
                    feature_names.append(feature)
            exceptions = self.data._cache_value_counts(feature_names, max_rows_to_eval=self.max_rows_to_eval,
                                                       chunk_size=50_000)
            if exceptions:
                unprocessed_msg = "Could not evaluate rare values candidates for some columns due to the following:\n"
                for feat, err in exceptions.items():
                    unprocessed_msg += f"\n\t- Column name: {feat}, Error: {err}"
                self.warnings_collector.triage(IFAWarningEmitterType.SIMPLE, unprocessed_msg)
        if max_distilled_cases is not None:
            user_set_mdc = True
            # Compute the optimized max_distilled_cases value if available
            max_distilled_cases, _ = get_optimized_max_chunk_size(row_count=self._get_row_count(),
                                                                  max_chunk_size=max_distilled_cases)
        else:
            # Set a small default
            max_distilled_cases = 25_000

        # Workflow 1: User provided a config with protected multipliers; may need to compute unprotected multipliers
        if preserve_rare_values_config is not None:
            if preserve_rare_values_map is not None:
                self.warnings_collector.triage(IFAWarningEmitterType.SIMPLE, "A `preserve_rare_values_map` was "
                                               "provided with a full `preserve_rare_values_config`; the former "
                                               "will be ignored.")
            # Config provided; check if unprotected multipliers need computation
            for feature, cfg in preserve_rare_values_config.items():
                feature_full_config = {}
                if not isinstance(cfg, Mapping):
                    # Workflow 1A: User provided a "simple" config (Mapping of feature names to list of rare values)
                    feature_full_config["protected_values_multipliers"] = deepcopy(cfg)
                else:
                    # Workflow 1B: User provided a "full" config with sub-keys (likely through the suggestion loop)
                    feature_full_config = deepcopy(cfg)
                # In any case, ensure the unprotected multipliers are present
                if "unprotected_multiplier" not in feature_full_config:
                    feature_full_config["unprotected_multiplier"] = self._compute_unprotected_multiplier(feature,
                                                                                                         feature_full_config["protected_values_multipliers"])
                _prvc[feature] = feature_full_config
        # Workflow 2: User provided a map of rare values to protect, but no multipliers
        elif preserve_rare_values_map is not None:
            # Workflow 2A: User set the max_distilled_cases, so we can compute multipliers here
            if user_set_mdc:
                if preserve_rare_values_map == "all":
                    preserve_rare_values_map, _ = self._find_protected_value_candidates(max_distilled_cases,
                                                                                        significance_threshold)
                _prvc = self._compute_preserve_rare_values_config(max_distilled_cases,
                                                                  preserve_rare_values_map,
                                                                  significance_threshold)
            # Workflow 2B: User did not set max_distilled_cases, so we cannot guarantee accurate multipliers.
            # Let another part of the stack figure it out; set only the protected values in the attributes.
            else:
                if preserve_rare_values_map == "all":
                    raise ValueError('If `preserve_rare_values_map` is set to "all," you must also provide '
                                     '`max_distilled_cases` to accurately determine rare value candidates.')
                for feature, values in preserve_rare_values_map.items():
                    if feature not in self.attributes:
                        # Multiprocessing is enabled, and this feature will be handled in another process
                        continue
                    self.attributes[feature]["preserve_rare_values"] = {"protected_values": values}  # pyright: ignore[reportGeneralTypeIssues]

        # Workflow 3: User provided no value specifications; determine candidates and make a suggestion
        else:
            # Skip this if data is smaller than the default (true for many test cases);
            # probably indicates that data distillation happening is unlikely
            if self._get_row_count() < max_distilled_cases:
                return
            # Compute but don't automatically apply
            preserve_rare_values_map, values_ranking = self._find_protected_value_candidates(max_distilled_cases,
                                                                                             significance_threshold)
            if preserve_rare_values_map:
                candidate_prvc = self._compute_preserve_rare_values_config(max_distilled_cases,
                                                                           preserve_rare_values_map,
                                                                           significance_threshold)
                prvc_suggestion = PRVSuggestion(candidate_prvc, values_ranking, user_set_mdc)
                self.suggestions_collector.append(prvc_suggestion)

        # Apply rare values multipliers to feature attributes if applicable (workflows 1, 2A)
        if _prvc:
            for feature, config in _prvc.items():
                if feature not in self.attributes:
                    # Multiprocessing is enabled, and this feature will be handled in another process
                    continue
                self.attributes[feature]["preserve_rare_values"] = config  # pyright: ignore[reportGeneralTypeIssues]
