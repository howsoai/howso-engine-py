from abc import ABC, abstractmethod
from collections.abc import Container
from copy import deepcopy
from functools import singledispatchmethod
import json
import logging
import math
import numbers
import platform
from typing import (
    Any, Collection, Dict, Iterable, List, Mapping, Optional, Tuple, Union
)
import warnings

from dateutil.parser import isoparse
from dateutil.parser import parse as dt_parse
import numpy as np
import pandas as pd
import yaml

from howso.utilities.features import FeatureType
from howso.utilities.internals import serialize_models

logger = logging.getLogger(__name__)

# Maximum/minimum data sizes for integers, floats, datetimes supported by the core
INTEGER_MAX = int(math.pow(2, 53))
FLOAT_MAX = 1.7976931348623157 * math.pow(10, 308)
FLOAT_MIN = 2.2250738585072014 * math.pow(10, -308)
WIN_DT_MAX = '6053-01-24'
LINUX_DT_MAX = '2262-04-11'


class FeatureAttributesBase(dict):
    """Provides accessor methods for and dict-like access to inferred feature attributes."""

    def __init__(self, feature_attributes: Mapping, params: Dict = {}, unsupported: List[str] = []):
        """
        Instantiate this FeatureAttributesBase object.

        Parameters
        ----------
        feature_attributes : Dict
            The feature attributes dictionary to be wrapped by this object.
        params : Dict
            (Optional) The parameters used in the call to infer_feature_attributes.
        unsupported : List of str
            (Optional) A list of features that contain data that is unsupported by the engine.

        """
        if not isinstance(feature_attributes, Mapping):
            raise TypeError('Provided feature attributes must be a Dict.')
        self.params = params
        self.update(feature_attributes)
        self.unsupported = unsupported

    def __copy__(self) -> "FeatureAttributesBase":
        """Return a (deep)copy of this instance of FeatureAttributesBase."""
        cls = self.__class__
        obj_copy = cls.__new__(cls)
        obj_copy.update(deepcopy(self))
        obj_copy.params = self.params
        return obj_copy

    def get_parameters(self) -> dict:
        """
        Get the keyword arguments used with the initial call to infer_feature_attributes.

        Returns
        -------
        Dict
            A dictionary containing the kwargs used in the call to `infer_feature_attributes`.

        """
        return self.params

    def to_json(self) -> str:
        """
        Get a JSON string representation of this FeatureAttributes object.

        Returns
        -------
        String
            A JSON representation of the inferred feature attributes.
        """
        return json.dumps(self)

    def get_names(self, *, types: Union[str, Container, None] = None,
                  without: Union[Iterable[str], None] = None,
                  ) -> List[str]:
        """
        Get feature names associated with this FeatureAttributes object.

        Parameters
        ----------
        types : String, Container (of String), default None
            (Optional) A feature type as a string (E.g., 'continuous') or a
            list of feature types to limit the output feature names.
        without : Iterable of String
            (Optional) An Iterable of feature names to exclude from the return object.

        Returns
        -------
        List of String
            A list of feature names.
        """
        if without:
            for feature in without:
                if feature not in self.keys():
                    raise ValueError(f'Feature {feature} does not exist in this FeatureAttributes '
                                     'object')
        names = self.keys()

        if types:
            if isinstance(types, str):
                types = [types, ]
            names = [
                name for name in names
                if self[name].get('type') in types
            ]

        return [
            key for key in names
            if without is None or key not in without
        ]

    def _validate_bounds(self, data: pd.DataFrame, feature: str,
                         attributes: Dict) -> List[str]:  # noqa: C901
        """Validate the feature bounds of the provided DataFrame."""
        # Import here to avoid circular import
        from howso.utilities import date_to_epoch

        errors = []

        # Ensure that there are bounds to validate
        if not isinstance(attributes.get('bounds'), Mapping):
            return errors

        # Gather some data to use for validation
        series = data[feature]
        bounds = attributes['bounds']
        min_bound = bounds.get('min')
        max_bound = bounds.get('max')
        # Get unique values but exclude NoneTypes
        unique_values = series.dropna().unique()

        if bounds.get('allowed'):
            # Check nominal bounds
            allowed_values = attributes['bounds']['allowed']
            out_of_band_values = set(unique_values) - set(allowed_values)
            if pd.isna(list(out_of_band_values)).all():
                # Placeholder for behavior when columns contain nans
                pass
            elif out_of_band_values:
                errors.append(f"'{feature}' contains out-of-band values: {out_of_band_values}")
        elif attributes.get('date_time_format'):
            # Since this is a datetime feature, convert dates to epoch time for bounds comparison
            try:
                if min_bound:
                    min_bound_epoch = date_to_epoch(min_bound, time_format=attributes['date_time_format'])
                if max_bound:
                    max_bound_epoch = date_to_epoch(max_bound, time_format=attributes['date_time_format'])
                for value in unique_values:
                    epoch = date_to_epoch(value, time_format=attributes['date_time_format'])
                    if (max_bound and epoch > max_bound_epoch) or (min_bound and epoch < min_bound_epoch):
                        errors.append(f"'{feature}' has a value outside of bounds (min: {min_bound}, "
                                      f"max: {max_bound}): {value}")
            except ValueError as err:
                errors.append(f'Could not validate datetime bounds due to the following error: {err}')
        elif min_bound or max_bound:
            # Check int/float bounds
            for value in unique_values:
                if (max_bound and float(value) > float(max_bound)) or (min_bound and float(value) < float(min_bound)):
                    errors.append(f"'{feature}' has a value outside of bounds (min: {min_bound}, "
                                  f"max: {max_bound}): {value}")

        return errors

    def _validate_dtype(self, data: pd.DataFrame, feature: str,  # noqa: C901
                        expected_dtype: Union[str, pd.CategoricalDtype], coerced_df: pd.DataFrame,
                        coerce: bool = False, localize_datetimes=True) -> List[str]:
        """Validate the data type of a feature and optionally attempt to coerce."""
        errors = []
        series = coerced_df[feature]
        is_valid = False

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
                format = self[feature]['date_time_format']
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
                except Exception: # noqa: Intentionally broad
                    pass

        # Raise warnings if the types do not match
        if not is_valid:
            if coerce:
                errors.append(f"Expected dtype '{expected_dtype}' for feature '{feature}' "
                              "but could not coerce.")
            else:
                errors.append(f"Feature '{feature}' should be '{expected_dtype}' dtype, but found "
                              f"'{data[feature].dtype}'")

        return errors

    @staticmethod
    def _allows_null(attributes: Dict) -> bool:
        """Return whether the given attributes indicates the allowance of null values."""
        return 'bounds' in attributes and attributes['bounds'].get('allow_null', False)

    def _validate_df(self, data: pd.DataFrame, coerce: bool = False,  # noqa: C901
                     raise_errors: bool = False, table_name: str = None, validate_bounds=True,
                     allow_missing_features: bool = False, localize_datetimes=True):
        errors = []
        coerced_df = data.copy(deep=True)
        features = self[table_name] if table_name else self

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
                    if attributes.get('decimal_places', 0) > 0 or self._allows_null(attributes):
                        errors.extend(self._validate_dtype(data, feature, 'float64',
                                                           coerced_df, coerce=coerce))
                    # Check type (int)
                    else:
                        errors.extend(self._validate_dtype(data, feature, 'int64',
                                                           coerced_df, coerce=coerce))
                elif attributes.get('data_type') == 'boolean':
                    # Check type (boolean)
                    errors.extend(self._validate_dtype(data, feature, 'bool',
                                                       coerced_df, coerce=coerce))
                elif attributes.get('bounds') and attributes['bounds'].get('allowed'):
                    # Check type (categorical)
                    schema_dtype = pd.CategoricalDtype(attributes['bounds']['allowed'],
                                                       ordered=True)
                    errors.extend(self._validate_dtype(data, feature, schema_dtype,
                                                       coerced_df, coerce=coerce))
                else:
                    # Else, should be an object
                    errors.extend(self._validate_dtype(data, feature, 'object',
                                                       coerced_df, coerce=coerce))

            # Check ordinal types
            elif attributes['type'] == 'ordinal':
                if attributes.get('bounds') and attributes['bounds'].get('allowed'):
                    # Check type (categorical)
                    schema_dtype = pd.CategoricalDtype(attributes['bounds']['allowed'],
                                                       ordered=True)
                    errors.extend(self._validate_dtype(data, feature, schema_dtype,
                                                       coerced_df, coerce=coerce))
                # Check type (float)
                elif attributes.get('decimal_places', 0) > 0 or self._allows_null(attributes):
                    errors.extend(self._validate_dtype(data, feature, 'float64',
                                                       coerced_df, coerce=coerce))
                # Check type (int)
                else:
                    errors.extend(self._validate_dtype(data, feature, 'int64',
                                                       coerced_df, coerce=coerce))

            # Check continuous types
            elif attributes['type'] == 'continuous':
                if 'date_time_format' in attributes:
                    # Check type (datetime)
                    errors.extend(self._validate_dtype(data, feature, 'datetime64',
                                                       coerced_df, coerce=coerce,
                                                       localize_datetimes=localize_datetimes))
                elif attributes.get('decimal_places', -1) > 0 or self._allows_null(attributes):
                    # Check type (float)
                    errors.extend(self._validate_dtype(data, feature, 'float64',
                                                       coerced_df, coerce=coerce))
                elif attributes.get('decimal_places', -1) == 0:
                    # Check type (int)
                    errors.extend(self._validate_dtype(data, feature, 'int64',
                                                       coerced_df, coerce=coerce))
                elif attributes.get('data_type') == 'number':
                    # If feature is continuous and not a datetime, it should have a numeric data_type.
                    # If it cannot be casted to a float, then add an error.
                    if len(self._validate_dtype(data, feature, 'float64',
                                                coerced_df, coerce=True)):
                        errors.extend([f"Feature '{feature}' should be numeric"
                                       " when 'type' is 'continuous' and "
                                       "'data_type' is undefined."])

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

    @abstractmethod
    def validate(self, data: Any, coerce=False, raise_errors=False, validate_bounds=True,
                 allow_missing_features=False, localize_datetimes=True):
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


class MultiTableFeatureAttributes(FeatureAttributesBase):
    """A dict-like object containing feature attributes for multiple tables."""

    pass


class SingleTableFeatureAttributes(FeatureAttributesBase):
    """A dict-like object containing feature attributes for a single table or DataFrame."""

    @singledispatchmethod
    def validate(data: Any, **kwargs):
        """
        Validate the given single table data against this FeatureAttributes object.

        Check that feature bounds and data types loosely describe the data. Optionally
        attempt to coerce the data into conformity.
        Parameters
        ----------
        data : Any
            The data to validate (single table only).
        coerce : bool (default False)
            Whether to attempt to coerce DataFrame columns into correct data types.
        raise_errors : bool (default False)
            If True, raises a ValueError if nonconforming columns are found; else, issue a warning.
        validate_bounds : bool (default True)
            Whether to validate the data against the attributes' inferred bounds.
        allow_missing_features : bool (default False)
            Allows features that are missing from the DataFrame to be ignored.
        localize_datetimes : bool (default True)
            Whether to localize datetime features to UTC.

        Returns
        -------
        None | DataFrame
            None or the coerced DataFrame if 'coerce' is True and there were no errors.
        """
        raise NotImplementedError("'data' is an unsupported type")

    @validate.register
    def _(self, data: pd.DataFrame, coerce=False, raise_errors=False, validate_bounds=True,
          allow_missing_features=False, localize_datetimes=True):
        return self._validate_df(data, coerce=coerce, raise_errors=raise_errors,
                                 validate_bounds=validate_bounds,
                                 allow_missing_features=allow_missing_features,
                                 localize_datetimes=localize_datetimes)

    def has_unsupported_data(self, feature_name: str) -> bool:
        """
        Returns whether the given feature has data that is unsupported by Howso Engine.

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

    def _process(self,  # noqa: C901
                 features: Optional[Dict[str, Dict]] = None,
                 infer_bounds: bool = True,
                 tight_bounds: Optional[Iterable[str]] = None,
                 mode_bound_features: Optional[Iterable[str]] = None,
                 id_feature_name: Optional[Union[str, Iterable[str]]] = None,
                 attempt_infer_extended_nominals: bool = False,
                 nominal_substitution_config: Optional[Dict[str, Dict]] = None,
                 include_extended_nominal_probabilities: Optional[bool] = False,
                 datetime_feature_formats: Optional[Dict] = None,
                 ordinal_feature_values: Optional[Dict[str, List[str]]] = None,
                 dependent_features: Optional[Dict[str, List[str]]] = None,
                 include_sample: bool = False,
                 max_workers: Optional[int] = None,
                 ) -> Dict:
        """
        Get inferred feature attributes for the parameters.

        See `infer_feature_attributes` for full docstring.
        """
        if isinstance(tight_bounds, bool):
            warnings.warn("Using booleans with 'tight_bounds' is deprecated. Please "
                          "specify the features you would like tight bounds for as an "
                          "iterable of strings.", DeprecationWarning)
            if tight_bounds:
                tight_bounds = self._get_feature_names()
            else:
                tight_bounds = None

        if features and not isinstance(features, dict):
            raise ValueError(
                f"The parameter `features` needs to be a `dict` and not of "
                f"type {type(features)}."
            )

        if features:
            feature_attributes: Dict = serialize_models(features)
        else:
            feature_attributes = dict()

        if datetime_feature_formats is None:
            datetime_feature_formats = dict()

        if ordinal_feature_values is None:
            ordinal_feature_values = dict()

        if dependent_features is None:
            dependent_features = dict()

        feature_names_list = self._get_feature_names()
        for feature_name in feature_names_list:
            # What type is this feature?
            feature_type, typing_info = self._get_feature_type(feature_name)

            typing_info = typing_info or dict()

            # EXPLICITLY DECLARED ORDINALS
            if feature_name in ordinal_feature_values:
                feature_attributes[feature_name] = {
                    'type': 'ordinal',
                    'bounds': {'allowed': ordinal_feature_values[feature_name]}
                }

            # EXPLICITLY DECLARED DATETIME FEATURES
            elif datetime_feature_formats.get(feature_name, None):
                # datetime_feature_formats is expected to either be only a
                # single string (format) or a tuple of strings (format, locale)
                user_dt_format = datetime_feature_formats[feature_name]
                if features and 'date_time_format' in features.get(feature_name, {}):
                    warnings.warn(f'Warning: date_time_format for {feature_name} '
                                  'provided in both `features` (ignored) and '
                                  '`datetime_feature_formats`.')
                    del features[feature_name]['date_time_format']

                if feature_type == FeatureType.DATETIME:
                    # When feature is a datetime instance, we won't need to
                    # parse the datetime from a string using a custom format.
                    feature_attributes[feature_name] = (
                        self._infer_datetime_attributes(feature_name))
                    warnings.warn(
                        'Providing a datetime feature format for the feature '
                        f'"{feature_name}" is not necessary because the data '
                        'is already formatted as a datetime object. This '
                        'custom format will be ignored.')
                elif feature_type == FeatureType.DATE:
                    # When feature is a date instance, we won't need to
                    # parse the datetime from a string using a custom format.
                    feature_attributes[feature_name] = (
                        self._infer_date_attributes(feature_name))
                    warnings.warn(
                        'Providing a datetime feature format for the feature '
                        f'"{feature_name}" is not necessary because the data '
                        'is already formatted as a date object. This custom '
                        'format will be ignored.')
                elif feature_type == FeatureType.TIME:
                    # When feature is a time instance, we won't need to
                    # parse the datetime from a string using a custom format.
                    feature_attributes[feature_name] = (
                        self._infer_datetime_attributes(feature_name))
                    warnings.warn(
                        'Time only features with a datetime feature format '
                        'will be treated as a datetime using the date '
                        '1970-1-1.', UserWarning)
                elif isinstance(user_dt_format, str):
                    # User passed only the format string
                    feature_attributes[feature_name] = {
                        'type': 'continuous',
                        'data_type': 'formatted_date_time',
                        'date_time_format': user_dt_format,
                    }
                elif (
                    isinstance(user_dt_format, Collection) and
                    len(user_dt_format) == 2
                ):
                    # User passed format string and a locale string
                    dt_format, dt_locale = user_dt_format
                    feature_attributes[feature_name] = {
                        'type': 'continuous',
                        'data_type': 'formatted_date_time',
                        'date_time_format': dt_format,
                        'locale': dt_locale,
                    }
                else:
                    # Not really sure what they passed.
                    raise TypeError(
                        f'The value passed (`{user_dt_format}`) to '
                        f'`datetime_feature_formats` for feature "{feature_name}"'
                        f'is invalid. It should be either a single string '
                        f'(format), or a tuple of 2 strings (format, locale).')

            # FLOATING POINT FEATURES
            elif feature_type == FeatureType.NUMERIC:
                feature_attributes[feature_name] = (
                    self._infer_floating_point_attributes(feature_name))

            # IMPLICITLY DEFINED DATETIME FEATURES
            elif feature_type == FeatureType.DATETIME:
                feature_attributes[feature_name] = (
                    self._infer_datetime_attributes(feature_name))

            # DATE ONLY FEATURES
            elif feature_type == FeatureType.DATE:
                feature_attributes[feature_name] = (
                    self._infer_date_attributes(feature_name))

            # TIME ONLY FEATURES
            elif feature_type == FeatureType.TIME:
                feature_attributes[feature_name] = (
                    self._infer_time_attributes(feature_name))

            # TIMEDELTA FEATURES
            elif feature_type == FeatureType.TIMEDELTA:
                feature_attributes[feature_name] = (
                    self._infer_timedelta_attributes(feature_name))

            # INTEGER FEATURES
            elif feature_type == FeatureType.INTEGER:
                feature_attributes[feature_name] = (
                    self._infer_integer_attributes(feature_name))

            # BOOLEAN FEATURES
            elif feature_type == FeatureType.BOOLEAN:
                feature_attributes[feature_name] = (
                    self._infer_boolean_attributes(feature_name))

            # ALL OTHER FEATURES
            else:
                feature_attributes[feature_name] = (
                    self._infer_string_attributes(feature_name))

            # Is column constrained to be unique?
            if self._has_unique_constraint(feature_name):
                feature_attributes[feature_name]['unique'] = True

            # Add original type to feature
            if feature_type is not None:
                feature_attributes[feature_name]['original_type'] = {
                    'data_type': str(feature_type),
                    **typing_info
                }

            # DECLARED DEPENDENTS
            # First determine if there are any dependent features in the partial features dict
            partial_dependent_features = []
            if features and 'dependent_features' in features.get(feature_name, {}):
                partial_dependent_features = features[feature_name]['dependent_features']
            # Set dependent features: `dependent_features` + partial features dict, if provided
            if feature_name in dependent_features:
                feature_attributes[feature_name]['dependent_features'] = list(
                    set(partial_dependent_features + dependent_features[feature_name])
                )

        if isinstance(id_feature_name, str):
            self._add_id_attribute(feature_attributes, id_feature_name)
        elif isinstance(id_feature_name, Iterable):
            for id_feature in id_feature_name:
                self._add_id_attribute(feature_attributes, id_feature)
        elif id_feature_name is not None:
            raise ValueError('ID feature must be of type `str` or `list[str], '
                             f'not {type(id_feature_name)}.')

        if infer_bounds:
            for feature_name in feature_attributes:
                # Don't infer bounds for manually-specified nominal features
                if features and features.get(feature_name, {}).get('type') == 'nominal':
                    continue
                # Likewise, don't infer bounds for JSON/YAML features
                elif any([
                    feature_attributes[feature_name].get('data_type') in ['json', 'yaml'],
                    features and features.get(feature_name, {}).get('data_type') in ['json', 'yaml']
                ]):
                    continue
                bounds = self._infer_feature_bounds(
                    feature_attributes, feature_name,
                    tight_bounds=tight_bounds,
                    mode_bound_features=mode_bound_features,
                )
                if bounds:
                    feature_attributes[feature_name]['bounds'] = bounds  # noqa

        # Do any features contain data unsupported by the core?
        self._check_unsupported_data(feature_attributes)

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

                nominal_default_subtype = 'int-id'
                # Apply them if they are above the threshold value.
                for feature_name in feature_names_list:
                    if feature_name in aenp:
                        if len(aenp[feature_name]) > 0:
                            feature_attributes[feature_name]['subtype'] = (max(
                                aenp[feature_name], key=aenp[feature_name].get))

                        if include_meta:
                            feature_attributes[feature_name].update({
                                'extended_nominal_probabilities':
                                    all_probs[feature_name]
                            })

                    # If `subtype` is a nominal feature, assign it to 'int-id'
                    if (
                        feature_attributes[feature_name]['type'] == 'nominal' and
                        not feature_attributes[feature_name].get('subtype', None)
                    ):
                        feature_attributes[feature_name]['subtype'] = (
                            nominal_default_subtype)
            except ImportError:
                warnings.warn('Cannot infer extended nominals: not supported')

        # Insert a ``sample`` value (as string) for each feature, if possible.
        if include_sample:
            for feature_name in feature_attributes.keys():
                sample = self._get_random_value(feature_name, no_nulls=True)
                if sample is not None:
                    sample = str(sample)
                feature_attributes[feature_name]['sample'] = sample

        # Re-insert any partial features provided as an argument
        if features:
            for feature in features.keys():
                for attribute, value in features[feature].items():
                    feature_attributes[feature][attribute] = value

        return feature_attributes

    @abstractmethod
    def __call__(self) -> FeatureAttributesBase:
        """Process and return the feature attributes."""

    @abstractmethod
    def _infer_floating_point_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given floating-point column."""

    @abstractmethod
    def _infer_datetime_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given date-time column."""

    @abstractmethod
    def _infer_date_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given date only column."""

    @abstractmethod
    def _infer_time_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given time column."""

    @abstractmethod
    def _infer_timedelta_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given timedelta column."""

    @abstractmethod
    def _infer_boolean_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given boolean column."""

    @abstractmethod
    def _infer_integer_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given integer column."""

    @abstractmethod
    def _infer_string_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given string column."""

    @abstractmethod
    def _infer_unknown_attributes(self, feature_name: str) -> Dict:
        """Get inferred attributes for the given unknown-type column."""

    @abstractmethod
    def _infer_feature_bounds(
        self,
        feature_attributes: Mapping[str, Mapping],
        feature_name: str,
        tight_bounds: Optional[Iterable[str]] = None,
        mode_bound_features: Optional[Iterable[str]] = None,
    ) -> Optional[Dict]:
        """
        Return inferred bounds for the given column.

        Features with datetimes are converted to seconds since epoch and their
        bounds are calculated accordingly. Features with timedeltas are
        converted to total seconds.

        Parameters
        ----------
        data : Any
            Input data
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
    def infer_loose_feature_bounds(min_bound: Union[int, float],
                                   max_bound: Union[int, float]
                                   ) -> Tuple[float, float]:
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
        Tuple of (min_bound, max_bound) of loose bounds around the provided tight
        min and max_bound bounds
        """
        # NOTE: It was considered to use a smoother bounds-expansion function that
        #       looked like max_loose_bounds = exp(ln(max_bounds) + 0.5), but this
        #       could leak privacy since the actual bounds would be
        #       reverse-engineer-able.
        assert min_bound <= max_bound, \
            "Feature min_bound cannot be larger than max_bound"

        if min_bound < 0:
            # for negative min_bound boundary values:  e^ceil(ln(num))
            min_bound = -np.exp(np.ceil(np.log(-min_bound)))
        elif min_bound > 0:
            # for positive min_bound boundary values:  e^floor(ln(num))
            min_bound = np.exp(np.floor(np.log(min_bound)))

        if max_bound < 0:
            # for negative max_bound boundary values: e^floor(ln(num))
            max_bound = -np.exp(np.floor(np.log(-max_bound)))
        elif max_bound > 0:
            # for positive max_bound boundary values: e^ceil(ln(num))
            max_bound = np.exp(np.ceil(np.log(max_bound)))

        return min_bound, max_bound

    @staticmethod
    def _get_datetime_max():
        # Avoid circular import
        from howso.client.client import get_howso_client_class
        from howso.direct import HowsoDirectClient
        # If on Direct, check the user's platform. Else, default to Unix.
        klass, _ = get_howso_client_class()
        if issubclass(klass, HowsoDirectClient):
            plat = platform.system().lower()
            if plat == 'windows':
                return WIN_DT_MAX
        return LINUX_DT_MAX

    def _check_unsupported_data(self, feature_attributes: Dict) -> None:
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
            # Check original data type for ints, floats, datetimes
            orig_type = feature_attributes[feature_name]['original_type']['data_type']
            if (orig_type in ['integer', 'numeric'] or 'date_time_format' in
                    feature_attributes[feature_name]):
                # Get feature bounds
                bounds = self._infer_feature_bounds(feature_attributes, feature_name,
                                                    tight_bounds=feature_names)
                if not bounds or bounds.get('min') is None or bounds.get('max') is None:
                    continue
                omit = False
                # Datetimes
                dt_fmt = feature_attributes[feature_name].get('date_time_format')
                if dt_fmt is not None:
                    # Get maximum compatible datetime (depends on platform)
                    allowed_max = date_to_epoch(self._get_datetime_max(), time_format='%Y-%m-%d')
                    actual_max = date_to_epoch(bounds['max'], time_format=dt_fmt)
                    # Verify
                    if actual_max >= allowed_max:
                        omit = True
                else:
                    # Determine the largest absolute value from the feature bounds
                    largest_value = max(abs(bounds['min']), bounds['max'])
                    # Verify integer min/max
                    if orig_type == 'integer':
                        if largest_value >= INTEGER_MAX:
                            omit = True
                    # Verify float min/max
                    elif orig_type == 'numeric':
                        # Determine the smallest absolute value from the feature bounds
                        smallest_value = min(abs(bounds['min']), abs(bounds['max']))
                        if largest_value >= FLOAT_MAX or smallest_value <= FLOAT_MIN:
                            omit = True
                # Keep track of unsupported data internally
                if omit:
                    self.unsupported.append(feature_name)

    @staticmethod
    def _is_datetime(string):
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
                if all([
                    '{' not in rand_val and '}' not in rand_val,
                    '[' not in rand_val and ']' not in rand_val,
                ]):
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
                if len(sample.split(':')) <= 1 or '\n' not in sample:
                    return False
            except yaml.YAMLError:
                return False

        return True

    @staticmethod
    def _add_id_attribute(feature_attributes: Mapping,
                          id_feature_name: str
                          ) -> None:
        """Update the given feature_attributes in-place for id_features."""
        if id_feature_name in feature_attributes:
            feature_attributes[id_feature_name]['id_feature'] = True
            # If id feature was inferred to be continuous, change it to nominal
            # with 'data_type':number attribute to prevent string conversion.
            if feature_attributes[id_feature_name]['type'] == 'continuous':
                feature_attributes[id_feature_name]['type'] = 'nominal'
                feature_attributes[id_feature_name]['data_type'] = 'number'
                if 'decimal_places' in feature_attributes[id_feature_name]:
                    del feature_attributes[id_feature_name]['decimal_places']

    @classmethod
    def _get_min_max_number_size_bounds(
        cls, feature_attributes: Mapping,
        feature_name: str
    ) -> Tuple[Optional[numbers.Number], Optional[numbers.Number]]:
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
            original_type = feature_attributes[feature_name]['original_type']
        except (TypeError, KeyError):
            # Feature not found or original typing info not defined
            return None, None

        min_value = None
        max_value = None
        if original_type and original_type.get('size'):
            size = original_type.get('size')
            data_type = original_type.get('data_type')

            if size in [1, 2, 4, 8]:
                if data_type == FeatureType.INTEGER.value:
                    if original_type.get('unsigned'):
                        dtype_info = np.iinfo(f'uint{size * 8}')
                    else:
                        dtype_info = np.iinfo(f'int{size * 8}')
                elif data_type == FeatureType.NUMERIC.value and size < 8:
                    dtype_info = np.finfo(f'float{size * 8}')
                else:
                    # Not a numeric feature or is 64bit float
                    return None, None

                min_value = dtype_info.min
                max_value = dtype_info.max
            elif size == 3 and data_type == FeatureType.INTEGER.value:
                # Some database dialects support 24bit integers
                if original_type.get('unsigned'):
                    min_value = 0
                    max_value = 16777215
                else:
                    min_value = -8388608
                    max_value = 8388607

        return min_value, max_value

    @abstractmethod
    def _get_feature_type(self, feature_name: str
                          ) -> Tuple[Optional[FeatureType], Optional[Dict]]:
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
    def _get_random_value(self, feature_name: str, no_nulls: bool = False) -> Any:
        """Retrieve a random value from the data."""

    @abstractmethod
    def _has_unique_constraint(self, feature_name: str) -> bool:
        """Return whether this feature has a unique constraint."""

    @abstractmethod
    def _get_first_non_null(self, feature_name: str) -> Optional[Any]:
        """
        Get the first non-null value in the given column.

        NOTE: "first" means arbitrarily the first one that the DataFrame or database
              returned; there is no implication of ordering.
        """

    @abstractmethod
    def _get_num_features(self) -> int:
        """Get the number of features/columns in the data."""

    @abstractmethod
    def _get_num_cases(self) -> int:
        """Get the number of cases/rows in the data."""

    @abstractmethod
    def _get_feature_names(self) -> List[str]:
        """Get the names of the features/columns of the data."""
