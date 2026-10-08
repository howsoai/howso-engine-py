from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Container, Iterable, Mapping, MutableSequence, Sequence, Set
from copy import deepcopy
import datetime
from functools import singledispatchmethod
import json
import logging
import math
from pathlib import Path
import platform
from numbers import Real
from typing import Any, cast, Literal, Self, TYPE_CHECKING
import warnings
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse, parse as dt_parse
import numpy as np
import pandas as pd
import yaml

from howso.utilities.fanout_features import infer_fanout_feature_config
from howso.utilities.feature_attributes.serializers import feature_attributes_pairs_hook, FeatureAttributesEncoder
from howso.utilities.feature_attributes.suggestions import (
    FanoutFeaturesInput,
    FanoutFeaturesSuggestion,
    IFASuggestion,
    IFASuggestionCollector,
    normalize_fanout_feature_map,
    partial_rare_value_preservation_message,
    PRVSuggestion,
    RareValuePreservationLimit,
)
from howso.utilities.feature_attributes.warnings import IFAWarningCollector, IFAWarningEmitterType
from howso.utilities.features import FeatureType
from howso.utilities.utilities import (
    determine_iso_format,
    get_optimized_max_chunk_size,
    is_null_value,
    is_valid_datetime_format,
    time_to_seconds,
)

if TYPE_CHECKING:
    from howso.client.typing import (
        FeatureAttributes,
        FeatureRareValueConfig,
        FullPreserveRareValuesConfig,
        FeatureRareValues,
        PreserveRareValues,
        PreserveRareValuesCaps,
        PreserveRareValuesConfig,
        PreserveRareValuesMap,
        ProtectedValueMultiplier,
    )

    RareValueReweighting = tuple[
        FeatureRareValueConfig | None, list[ProtectedValueMultiplier], RareValuePreservationLimit | None
    ]
    """
    A feature's rare value configuration, the kept rare values alone, and how preservation was limited.

    The configuration is None when nothing could be preserved.
    """


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

SIGNIFICANT_THRESHOLD_DEFAULT: int = 30
"""The ceiling of the significance threshold computed for a feature when `significance_threshold` is not given."""


def _dynamic_significance_threshold(total_cases: int, max_distilled_cases: int, distinct_values: int) -> int:
    """
    Compute how many cases a value of a feature needs to keep a signal through distillation.

    The threshold is the compression ratio, the number of cases per distilled case, below which a
    value is not expected to keep even one case; a feature whose average number of cases per value
    is higher uses that average instead, up to :data:`SIGNIFICANT_THRESHOLD_DEFAULT`.

    Parameters
    ----------
    total_cases : int
        The number of cases in the data.
    max_distilled_cases : int
        The distillation target, as :func:`get_optimized_max_chunk_size` rounds it.
    distinct_values : int
        The number of distinct values of the feature, counting every null form as one value.

    Returns
    -------
    int
        The threshold, rounded down and at least 1.
    """
    compression_ratio = total_cases / max_distilled_cases
    average_cases_per_value = total_cases / max(distinct_values, 1)
    return max(1, int(max(compression_ratio, min(average_cases_per_value, SIGNIFICANT_THRESHOLD_DEFAULT))))

DEFAULT_MAX_DISTILLED_CASES: int = 50_000
"""The distillation target assumed for rare value weighting when `max_distilled_cases` is not given."""

DEFAULT_RARE_VALUE_CAP: float = 0.5
"""
The largest share of its case weight a significant value gives up to fund rare value preservation.

Applies to a feature listed in ``preserve_rare_values_caps`` without a cap of its own. A feature
that is not listed has no cap: its significant values give up as much as preserving every rare
value requires.
"""


def _rare_value_features(preserve_rare_values: PreserveRareValues | None) -> list[str]:
    """
    Get the feature names that a ``preserve_rare_values`` given as a sequence of names selects.

    Parameters
    ----------
    preserve_rare_values : PreserveRareValues, optional
        Any accepted form of ``preserve_rare_values``.

    Returns
    -------
    list of str
        The named features, or an empty list for a mapping, "all", "off" or None.

    Raises
    ------
    ValueError
        If `preserve_rare_values` is a string other than "all" or "off".
    TypeError
        If `preserve_rare_values` is neither a mapping, a sequence of names, nor one of those strings.
    """
    if preserve_rare_values is None or isinstance(preserve_rare_values, Mapping):
        return []
    if isinstance(preserve_rare_values, str):
        if preserve_rare_values in ("all", "off"):
            return []
        raise ValueError('`preserve_rare_values` must be a mapping of feature name to rare values, a list of '
                         f'feature names, "all" or "off"; got the string "{preserve_rare_values}". To '
                         "preserve every rare value of one feature, pass its name in a list.")
    if not isinstance(preserve_rare_values, Sequence):
        raise TypeError("`preserve_rare_values` must be a mapping of feature name to rare values, a list of "
                        f'feature names, "all" or "off"; got {type(preserve_rare_values).__name__}.')
    return [str(feature) for feature in preserve_rare_values]


def _split_rare_values(
    preserve_rare_values: Mapping[str, FeatureRareValues],
) -> tuple[PreserveRareValuesMap, PreserveRareValuesConfig, FullPreserveRareValuesConfig]:
    """
    Sort the features of a ``preserve_rare_values`` mapping by the form of their specification.

    Parameters
    ----------
    preserve_rare_values : Mapping of str to FeatureRareValues
        The rare value specification of each feature.

    Returns
    -------
    PreserveRareValuesMap
        The features given as values to protect, whose multipliers are to be computed.
    PreserveRareValuesConfig
        The features given as values paired with the multipliers they should receive.
    FullPreserveRareValuesConfig
        The features given as a complete configuration.

    Repeated values are listed once, every null form counting as the one value None. Multipliers
    must be finite numbers: at least 1 when paired with a value to fund, and positive in a complete
    configuration.

    Raises
    ------
    ValueError
        If a feature's specification is a mapping without ``value_weight_multipliers``, a sequence
        that mixes plain values with value and multiplier pairs, a pair without a multiplier, a
        multiplier that is not a finite number in range, or one value paired with two different
        multipliers.
    TypeError
        If a feature's specification is neither a mapping nor a sequence.
    """
    values_map: PreserveRareValuesMap = {}
    config: PreserveRareValuesConfig = {}
    full: FullPreserveRareValuesConfig = {}
    forms = ('a list of values, a list of dicts of "value" and "multiplier", or a dict with a '
             '"value_weight_multipliers" list')
    for feature, spec in preserve_rare_values.items():
        if isinstance(spec, Mapping):
            if "value_weight_multipliers" not in spec:
                raise ValueError(f"The `preserve_rare_values` entry for feature `{feature}` must be {forms}; got a "
                                 f"dict with the keys {sorted(spec)}.")
            for entry in spec["value_weight_multipliers"]:
                _validate_multiplier(feature, entry["value"], entry["multiplier"], minimum=0.0)
                if entry["multiplier"] == 0:
                    raise ValueError(f"The multiplier for value `{entry['value']}` of feature `{feature}` must be "
                                     "positive; got 0.")
            full[feature] = spec  # pyright: ignore[reportArgumentType]
            continue
        if isinstance(spec, (str, bytes)) or not isinstance(spec, Iterable):
            raise TypeError(f"The `preserve_rare_values` entry for feature `{feature}` must be {forms}; got "
                            f"{type(spec).__name__}.")
        entries = list(spec)
        paired = [isinstance(entry, Mapping) and "value" in entry for entry in entries]
        if entries and all(paired):
            config[feature] = _dedupe_pairs(feature, entries)
        elif any(paired):
            raise ValueError(f"The `preserve_rare_values` entry for feature `{feature}` mixes plain values with "
                             'dicts of "value" and "multiplier". List the values alone to have every multiplier '
                             "computed, or give every value a multiplier.")
        else:
            values_map[feature] = _dedupe_values(entries)
    return values_map, config, full


def _dedupe_pairs(feature: str, entries: Sequence[Mapping[str, Any]]) -> list[ProtectedValueMultiplier]:
    """
    Validate value and multiplier pairs and list each distinct value once.

    Parameters
    ----------
    feature : str
        The name of the feature, for error messages.
    entries : Sequence of Mapping
        Dicts with a "value" key and, for each to be valid, a "multiplier" key.

    Returns
    -------
    list of ProtectedValueMultiplier
        One entry per distinct value whose multiplier is above 1, in order of first appearance, with
        every null form as None. A value paired with a multiplier of exactly 1 asks for no change,
        which is what an unlisted value gets, so it is left out.

    Raises
    ------
    ValueError
        If an entry has no multiplier, a multiplier is not a finite number of at least 1, or one
        value is given two different multipliers.
    """
    multipliers: dict[int, float] = {}
    values = [None if is_null_value(entry["value"]) else entry["value"] for entry in entries]
    distinct = _dedupe_values(values)
    for value, entry in zip(values, entries, strict=True):
        if "multiplier" not in entry:
            raise ValueError(f"The `preserve_rare_values` entry for feature `{feature}` pairs value `{value}` with "
                             "no multiplier. Give each value a multiplier, or list the values alone to have the "
                             "multipliers computed.")
        multiplier = _validate_multiplier(feature, value, entry["multiplier"], minimum=1.0)
        index = next(i for i, seen in enumerate(distinct) if seen is value or seen == value)
        if index in multipliers and multipliers[index] != multiplier:
            raise ValueError(f"The `preserve_rare_values` entry for feature `{feature}` gives value `{value}` two "
                             f"different multipliers, {multipliers[index]:g} and {multiplier:g}.")
        multipliers[index] = multiplier
    return [{"value": value, "multiplier": multipliers[i]} for i, value in enumerate(distinct) if multipliers[i] != 1]


def _normalize_rare_value_caps(caps: PreserveRareValuesCaps | None) -> dict[str, float]:
    """
    Turn either accepted form of ``preserve_rare_values_caps`` into a mapping of feature name to cap.

    Parameters
    ----------
    caps : Sequence of str or Mapping of str to float, optional
        Feature names, each given :data:`DEFAULT_RARE_VALUE_CAP`, or a mapping of feature name
        to the largest share of its weight a significant value of the feature may give up.

    Returns
    -------
    dict of str to float
        The cap of each listed feature.

    Raises
    ------
    TypeError
        If `caps` is neither a sequence of names nor a mapping.
    ValueError
        If a cap is not between 0 (exclusive) and 1 (inclusive).
    """
    if caps is None:
        return {}
    if isinstance(caps, Mapping):
        normalized = {str(feature): float(cap) for feature, cap in caps.items()}
    elif isinstance(caps, (str, bytes)) or not isinstance(caps, Iterable):
        raise TypeError("`preserve_rare_values_caps` must be a list of feature names or a mapping of feature "
                        f"name to cap; got {type(caps).__name__}.")
    else:
        normalized = {str(feature): DEFAULT_RARE_VALUE_CAP for feature in caps}
    for feature, cap in normalized.items():
        if not 0 < cap <= 1:
            raise ValueError(f"The `preserve_rare_values_caps` value for feature `{feature}` must be greater than 0 "
                             f"and at most 1; got {cap}.")
    return normalized


def _as_feature_value(value: Any) -> Any:
    """
    Convert a feature value to the form stored in the feature attributes.

    Nulls are stored as None whatever the feature's type, numpy scalars as the equivalent Python
    type, and every other value as given, so an integer stays an integer.
    """
    if is_null_value(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dedupe_values(values: Iterable[Any]) -> list[Any]:
    """
    Return the distinct values of `values`, in order of first appearance.

    Every null form counts as the single value None. Unhashable values are compared by equality.
    """
    result: list[Any] = []
    null_seen = False
    seen_hashable: set[Any] = set()
    seen_unhashable: list[Any] = []
    for value in values:
        if is_null_value(value):
            if null_seen:
                continue
            null_seen = True
            result.append(None)
            continue
        try:
            if value in seen_hashable:
                continue
            seen_hashable.add(value)
        except TypeError:
            if any(value == seen for seen in seen_unhashable):
                continue
            seen_unhashable.append(value)
        result.append(value)
    return result


def _validate_multiplier(feature: str, value: Any, multiplier: Any, minimum: float) -> float:
    """
    Check that a user-given case weight multiplier is a finite number no smaller than `minimum`.

    Returns
    -------
    float
        The multiplier as a float.

    Raises
    ------
    ValueError
        If the multiplier is not a number, is NaN or infinite, or is below `minimum`.
    """
    if not isinstance(multiplier, Real) or not math.isfinite(multiplier) or multiplier < minimum:
        raise ValueError(f"The multiplier for value `{value}` of feature `{feature}` must be a finite number of at "
                         f"least {minimum:g}; got {multiplier!r}.")
    return float(multiplier)


def _bucket_protected_values(protected_values: Iterable[Any]) -> tuple[bool, set[Any], list[Any]]:
    """
    Split protected values into whether any is null, the hashable values, and the unhashable values.

    Hashable values can be looked up by hash; the unhashable ones are compared one by one.
    """
    null_protected = False
    hashable: set[Any] = set()
    unhashable: list[Any] = []
    for value in protected_values:
        if is_null_value(value):
            null_protected = True
            continue
        try:
            hashable.add(value)
        except TypeError:
            unhashable.append(value)
    return null_protected, hashable, unhashable


class FeatureAttributesBase(dict[str, "FeatureAttributes"]):
    """Provides accessor methods for and dict-like access to inferred feature attributes."""

    def __init__(
        self,
        feature_attributes: Mapping[str, FeatureAttributes],
        params: dict[str, Any] | None = None,
        unsupported: list[str] | None = None,
        suggestions_collector: IFASuggestionCollector | None = None,
    ) -> None:
        """
        Instantiate this FeatureAttributesBase object.

        Parameters
        ----------
        feature_attributes : Mapping
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
        self.suggestions_collector = suggestions_collector or IFASuggestionCollector()

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

    def get_parameters(self) -> dict[str, Any]:
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
        cls,
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
        FeatureAttributesBase
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
            with open(json_path) as fp:
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
        raise NotImplementedError("Function not yet implemented for all subclasses of `FeatureAttributesBase`")

    def get_names(self, *, types: str | Container[str] | None = None,
                  data_types: str | Container[str] | None = None,
                  without: str | Iterable[str] | None = None,
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
                    raise ValueError(f"Feature {feature} does not exist in this FeatureAttributes "
                                     "object")
        names = self.keys()

        if types:
            if isinstance(types, str):
                types = [types ]
        else:
            types = []
        if data_types:
            if isinstance(data_types, str):
                data_types = [data_types ]
        else:
            data_types = []
        names = [
            name for name in names
            if (self[name].get("type") in types or not types)
            and (self[name].get("data_type") in data_types or not data_types)
        ]

        return [
            key for key in names
            if without is None or key not in without
        ]

    def _validate_bounds(self, data: pd.DataFrame, feature: str,
                         attributes: FeatureAttributes) -> list[str]:
        """Validate the feature bounds of the provided DataFrame."""
        # Import here to avoid circular import
        from howso.utilities import date_to_epoch  # noqa: PLC0415

        errors = []

        # Ensure that there are bounds to validate
        if not isinstance(attributes.get("bounds"), Mapping) or attributes.get("data_type") in ["json", "yaml"]:
            return errors

        # Gather some data to use for validation
        series = data[feature]
        if pd.api.types.is_timedelta64_dtype(series.dtype):
            # Timedelta bounds are inferred, and timedeltas serialized, as total seconds
            series = series.dt.total_seconds()
        bounds = attributes["bounds"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        min_bound = bounds.get("min")
        max_bound = bounds.get("max")
        # Get unique values but exclude NoneTypes
        unique_values = series.dropna().unique()
        additional_errors = 0

        if bounds.get("allowed"):
            # Check nominal bounds
            allowed_values = attributes["bounds"]["allowed"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            out_of_band_values = set(unique_values) - set(allowed_values)
            if pd.isna(list(out_of_band_values)).all():
                # Placeholder for behavior when columns contain nans
                pass
            elif out_of_band_values:
                errors.append(f"'{feature}' contains out-of-band values: {out_of_band_values}")
        elif attributes.get("date_time_format"):
            # Time-only attributes have bounds represented in seconds
            if attributes.get("original_type", {}).get("data_type") == "time":
                unique_time_values = pd.to_datetime(
                    series,
                    format=attributes["date_time_format"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    errors="coerce"
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
                        min_bound_epoch = date_to_epoch(min_bound, time_format=attributes["date_time_format"])
                    if max_bound:
                        max_bound_epoch = date_to_epoch(max_bound, time_format=attributes["date_time_format"])
                    for value in unique_values:
                        epoch = date_to_epoch(value, time_format=attributes["date_time_format"])
                        if (max_bound and epoch > max_bound_epoch) or (min_bound and epoch < min_bound_epoch):
                            if len(errors) < 5:
                                errors.append(
                                    f'"{feature}" has a value outside of bounds '
                                    f'(min: {min_bound}, max: {max_bound}): {value}'
                                )
                            else:
                                additional_errors += 1
                except ValueError as err:
                    errors.append(f"Could not validate datetime bounds due to the following error: {err}")
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

    @staticmethod
    def _is_numeric_dtype(dtype: str | np.dtype | pd.api.extensions.ExtensionDtype | pd.CategoricalDtype) -> bool:
        """Return whether `dtype` holds numbers, i.e. an integer, nullable integer, or float dtype."""
        try:
            dtype = pd.api.types.pandas_dtype(dtype)
        except TypeError:
            return False
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)

    def _validate_dtype(self, data: pd.DataFrame, feature: str,
                        expected_dtype: str | pd.CategoricalDtype, coerced_df: pd.DataFrame,
                        coerce: bool = False, localize_datetimes: bool = True) -> list[str]:
        """Validate the data type of a feature and optionally attempt to coerce."""
        errors = []
        series = coerced_df[feature]
        actual_dtype = data[feature].dtype
        is_valid = False
        coerce_err = ""

        if isinstance(expected_dtype, pd.CategoricalDtype):
            # If the feature is a Categorical dtype, try to coerce
            try:
                series = series.astype(expected_dtype)
                if coerce:
                    coerced_df[feature] = series
                is_valid = True
            except Exception as err: # noqa: Intentionally broad
                coerce_err = str(err)
        elif expected_dtype == "datetime64":
            try:
                format = self[feature]["date_time_format"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
                if ".%f" in format:
                    format = "ISO8601"
                series = pd.to_datetime(coerced_df[feature], format=format)
                if coerce:
                    # `series.dt.tz` covers both numpy and pyarrow timestamps. Naive pyarrow
                    # timestamps become numpy datetimes before localizing, because pyarrow's time
                    # zone support needs a separate timezone database on Windows. UTC has no DST
                    # transitions, so localizing to it is never ambiguous or nonexistent.
                    if localize_datetimes and series.dt.tz is None:
                        if isinstance(series.dtype, pd.ArrowDtype):
                            series = series.astype(series.dtype.numpy_dtype)
                        coerced_df[feature] = series.dt.tz_localize("UTC")
                    else:
                        coerced_df[feature] = series
                is_valid = True
            except Exception as err: # noqa: Intentionally broad
                coerce_err = str(err)
        # Else, compare the dtype directly
        elif actual_dtype.name == expected_dtype:
            is_valid = True
        # If the feature can be converted, consider it valid (slightly differing numeric types, etc.)
        else:
            try:
                series = series.astype(expected_dtype)
                if coerce:
                    coerced_df[feature] = series
                is_valid = True
            except Exception as err: # noqa: Intentionally broad
                # Numeric dtypes differ only in representation here: validation does not alter the
                # data unless `coerce` is set, so a numeric column is trained as it stands whichever
                # dtype the attributes imply. A column that cannot be cast keeps its own dtype.
                is_valid = self._is_numeric_dtype(expected_dtype) and self._is_numeric_dtype(actual_dtype)
                coerce_err = str(err)

        # Raise warnings if the types do not match
        if not is_valid:
            if coerce:
                errors.append(f"Expected dtype '{expected_dtype}' for feature '{feature}' "
                              f"but could not coerce:\nActual dtype: {actual_dtype}"
                              f"\nError raised from Pandas.astype():\n\n{coerce_err}")
            else:
                errors.append(f"Feature '{feature}' should be '{expected_dtype}' dtype, but found "
                              f"'{actual_dtype}'")

        return errors

    @staticmethod
    def _allows_null(attributes: FeatureAttributes) -> bool:
        """Return whether the given attributes indicates the allowance of null values."""
        return "bounds" in attributes and attributes["bounds"].get("allow_null", False)

    def _validate_df(self, data: pd.DataFrame, coerce: bool = False,
                     raise_errors: bool = False, table_name: str | None = None, validate_bounds=True,
                     allow_missing_features: bool = False, localize_datetimes=True, nullable_int_dtype="Int64"):
        errors = []
        coerced_df = data.copy(deep=True)
        features = cast(dict[str, "FeatureAttributes"], self[table_name] if table_name else self)

        for feature, attributes in features.items():
            if feature not in data.columns:
                # Check if column is missing (and not supposed to be)
                if not (
                    feature.startswith(".")
                    or (
                        attributes.get("auto_derive_on_train", False)
                        and "derived_feature_code" in attributes
                    )
                    or allow_missing_features
                ):
                    errors.append(f"{feature} is missing from the dataframe")
                # OK if it's an internal feature or is being processed by Validator
                continue

            # Check nominal types
            if attributes["type"] == "nominal":
                if attributes.get("data_type") == "number":
                    # Check type (float)
                    if attributes.get("decimal_places", 0) > 0:
                        errors.extend(self._validate_dtype(data, feature, "float64",
                                                           coerced_df, coerce=coerce))
                    # Check type (nullable Int)
                    elif self._allows_null(attributes):
                        errors.extend(self._validate_dtype(data, feature, nullable_int_dtype,
                                                           coerced_df, coerce=coerce))
                    # Check type (int)
                    else:
                        errors.extend(self._validate_dtype(data, feature, "int64",
                                                           coerced_df, coerce=coerce))
                elif attributes.get("data_type") == "boolean":
                    # Check type (boolean). A boolean column that also holds nulls keeps its dtype,
                    # since casting it to `bool` would turn every null into `False`. Only its
                    # non-null values are checked, so null markers `bool()` rejects (`pd.NA` in the
                    # nullable `boolean`, `bool[pyarrow]`, or object dtypes) do not fail the check.
                    if data[feature].isna().any():
                        non_null = data[[feature]].dropna()
                        errors.extend(self._validate_dtype(non_null, feature, "bool", non_null.copy()))
                    else:
                        errors.extend(self._validate_dtype(data, feature, "bool", coerced_df,
                                                           coerce=coerce))
                elif attributes.get("bounds") and attributes["bounds"].get("allowed"):  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    # Check type (categorical)
                    schema_dtype = pd.CategoricalDtype(attributes["bounds"]["allowed"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                                                       ordered=True)
                    errors.extend(self._validate_dtype(data, feature, schema_dtype,
                                                       coerced_df, coerce=coerce))
                else:
                    # Else, should be an object
                    errors.extend(self._validate_dtype(data, feature, "object",
                                                       coerced_df, coerce=coerce))

            # Check ordinal types
            elif attributes["type"] == "ordinal":
                if attributes.get("bounds") and attributes["bounds"].get("allowed"):  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    # Check type (categorical)
                    schema_dtype = pd.CategoricalDtype(attributes["bounds"]["allowed"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                                                       ordered=True)
                    errors.extend(self._validate_dtype(data, feature, schema_dtype,
                                                       coerced_df, coerce=coerce))
                # Check type (float)
                elif attributes.get("decimal_places", 0) > 0:
                    errors.extend(self._validate_dtype(data, feature, "float64",
                                                       coerced_df, coerce=coerce))
                # Check type (nullable Int)
                elif self._allows_null(attributes):
                    errors.extend(self._validate_dtype(data, feature, nullable_int_dtype,
                                                       coerced_df, coerce=coerce))
                # Check type (int)
                else:
                    errors.extend(self._validate_dtype(data, feature, "int64",
                                                       coerced_df, coerce=coerce))

            # A timedelta column is already in its native form. It is serialized as total seconds,
            # so the numeric dtype checks below do not apply.
            elif pd.api.types.is_timedelta64_dtype(data[feature].dtype):
                pass

            # Check continuous types
            elif "date_time_format" in attributes:
                # Check type (datetime)
                errors.extend(self._validate_dtype(data, feature, "datetime64",
                                                   coerced_df, coerce=coerce,
                                                   localize_datetimes=localize_datetimes))

            # Check semi-structured type (object)
            elif attributes.get("data_type") in {"json", "yaml", "amalgam", "string", "string_mixable"}:
                errors.extend(self._validate_dtype(data, feature, "object", coerced_df, coerce=coerce))

            # Check type (float)
            elif attributes.get("decimal_places", -1) > 0:
                errors.extend(self._validate_dtype(data, feature, "float64",
                                                   coerced_df, coerce=coerce))
            # Check type (nullable Int)
            elif self._allows_null(attributes):
                errors.extend(self._validate_dtype(data, feature, nullable_int_dtype,
                                                   coerced_df, coerce=coerce))
            # Check type (int)
            elif attributes.get("decimal_places", -1) == 0:
                errors.extend(self._validate_dtype(data, feature, "int64",
                                                   coerced_df, coerce=coerce))
            elif attributes.get("data_type") == "number":
                # If feature is continuous and not a datetime, it should have a numeric data_type.
                # If it cannot be casted to a float, then add an error.
                if len(self._validate_dtype(data, feature, "float64",
                                            coerced_df, coerce=True)):
                    errors.extend([f"Feature '{feature}' should be numeric"
                                   " when 'type' is 'continuous' and "
                                   "'data_type' is 'number'."])

            # Check feature bounds
            if validate_bounds:
                errors.extend(self._validate_bounds(data, feature, attributes))

        if errors:
            msg = ("Failed to validate DataFrame against feature attributes due to the "
                   "following errors:\n")
            for error in errors:
                msg = msg + f"{error}\n"
            if raise_errors:
                raise ValueError(msg)
            warnings.warn(msg)

        if coerce:
            return coerced_df

        return None

    def validate(self, data: Any, coerce: bool = False, raise_errors: bool = False, validate_bounds: bool = True,
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
        raise NotImplementedError

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
            orig_type = attributes.get(feature_name, {}).get("type")
            new_type = entries[feature_name].get("type")
            # TODO 22059: Allow ordinals here when we can attempt to infer values
            if new_type == "ordinal" and not (
                attributes.get(feature_name, {}).get("bounds", {}).get("allowed") or
                entries.get(feature_name, {}).get("bounds", {}).get("allowed")
            ):
                raise ValueError("Inference of ordinal values is not yet supported. Please "
                                 "preset ordinal features with their ordered values using "
                                 "`ordinal_feature_values`.")
            # Sanity check: booleans must be nominal
            if entries[feature_name].get("data_type") == "boolean" and orig_type and orig_type != "nominal":
                warnings.warn(
                    f'Feature "{feature_name}" was preset as {orig_type} '
                    'but was detected to be a boolean. Booleans '
                    'must be "nominal", thus the type override will be ignored.'
                )
            # In otherwise valid cases, ensure that existing types are not overwritten
            elif orig_type and new_type:
                del entries[feature_name]["type"]
            # Finally, update the dict with all remaining attributes
            if feature_name not in attributes.keys():
                attributes[feature_name] = entries[feature_name]
            else:
                attributes[feature_name].update(entries[feature_name])

        return attributes


class MultiTableFeatureAttributes(FeatureAttributesBase):
    """A dict-like object containing feature attributes for multiple tables."""


class SingleTableFeatureAttributes(FeatureAttributesBase):
    """A dict-like object containing feature attributes for a single table or DataFrame."""

    @singledispatchmethod
    def validate(data: Any, **kwargs: Any):
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
          nullable_int_dtype: str | np.dtype | pd.api.extensions.ExtensionDtype = "Int64"):
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
        sep = "|"
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
                tuple(c.split(sep)) if sep in c else (c, "")
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

    _summarize: bool = True
    """
    Whether :meth:`_emit_summary` prints anything.

    An inferrer that delegates to another — time series wraps the DataFrame or
    AbstractData inferrer — turns this off on the inner instance so the summary
    is printed once, by the outermost call, after every suggestion has been
    collected.
    """

    def _emit_summary(self) -> None:
        """
        Print the feature attributes summary, once inference has completed.

        Nothing is printed when there are no suggestions, so a clean run stays
        silent, and nothing is printed when :attr:`_summarize` is off.
        """
        if self._summarize and self.suggestions_collector.suggestions:
            self.suggestions_collector.print_summary()

    def _validate_rare_value_parameters(
        self,
        preserve_rare_values: PreserveRareValues | None,
        preserve_rare_values_caps: PreserveRareValuesCaps | None = None,
    ) -> None:
        """
        Check the rare value parameters against the features in the data.

        Called once, before features are processed in separate processes, so an error is raised once
        and a feature missing from a process's share of the columns is never mistaken for a typo.

        Raises
        ------
        ValueError
            If any parameter names a feature that is not in the data, a feature's rare value
            specification is of an unrecognized form, or a cap is out of range.
        """
        feature_names = self._get_feature_names()
        if isinstance(preserve_rare_values, Mapping):
            _split_rare_values(preserve_rare_values)
            rare_value_features: Iterable[str] = preserve_rare_values
        else:
            rare_value_features = _rare_value_features(preserve_rare_values)
        named_features: dict[str, Iterable[str]] = {
            "preserve_rare_values_caps": _normalize_rare_value_caps(preserve_rare_values_caps),
            "preserve_rare_values": rare_value_features,
        }
        for parameter, features in named_features.items():
            unknown = [feature for feature in features if feature not in feature_names]
            if unknown:
                names = ", ".join(f"`{feature}`" for feature in unknown)
                raise ValueError(f"`{parameter}` names features that are not in the data: {names}.")

    def _process(self,
                 attempt_infer_extended_nominals: bool = False,
                 max_distilled_cases: int | None = None,
                 datetime_feature_formats: dict | None = None,
                 default_time_zone: str | None = None,
                 dependent_features: dict[str, list[str]] | None = None,
                 enable_suggestions: bool = True,
                 fanout_feature_map: FanoutFeaturesInput | None = None,
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
                 ordinal_feature_values: dict[str, list[Any]] | None = None,
                 preserve_rare_values: PreserveRareValues | None = None,
                 preserve_rare_values_caps: PreserveRareValuesCaps | None = None,
                 significance_threshold: int | None = None,
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
                    raise
                if bounds:
                    # Use `update` on the bounds dictionary in case `allowed` ordinal values have already been set
                    bounds.update(self.attributes[feature_name].get("bounds", {}))
                    _attributes["bounds"] = bounds
                # Record whether we have observed any nulls in this column
                if "bounds" not in _attributes:
                    _attributes["bounds"] = {}
                _attributes["bounds"]["nulls_observed"] = self._contains_nulls(feature_name)

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
            for key_features, fanout_features in normalize_fanout_feature_map(fanout_feature_map).items():
                if isinstance(key_features, str):
                    key_features = [key_features]
                for f in fanout_features:
                    if f in self.attributes:
                        self.attributes[f]["fanout_on"] = list(key_features)
        # If not provided, infer them and issue a suggestion (unless suggestions are disabled)
        elif enable_suggestions:
            candidate_fanout = infer_fanout_feature_config(self.attributes, self.data, max_rows=self.max_rows_to_eval)
            if candidate_fanout:
                self.suggestions_collector.append(FanoutFeaturesSuggestion(candidate_fanout))

        # Compute or suggest `preserve_rare_values` configuration
        self._process_rare_values(preserve_rare_values, max_distilled_cases, significance_threshold,
                                  enable_suggestions, caps=_normalize_rare_value_caps(preserve_rare_values_caps))

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
    def _contains_nulls(self, feature_name: str) -> bool:
        """Get whether the provided feature has any nulls."""

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
            # It isn't clear how this method would be called on a feature
            # if it has no data, but just in case...
            return {
                "type": "continuous",
                "data_type": "number",
            }
        if self._is_json_feature(feature_name):
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
        if self._is_yaml_feature(feature_name):
            return {
                "type": "continuous",
                "data_type": "yaml"
            }
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
        return self._infer_unknown_attributes(feature_name)

    def _infer_unknown_attributes(self, *args: Any) -> dict:
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
            if dt_format and data_type in {"formatted_date_time", "formatted_time"}:
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
                        if getattr(rand_val, "tzinfo", None) is not None and isinstance(rand_val.tzinfo, ZoneInfo):
                            continue
                    # Warn in case of UTC offset -- could lead to unexpected results due to time zone
                    # differences
                    self.warnings_collector.triage(IFAWarningEmitterType.UTC_OFFSET, feature_name)

    @staticmethod
    def _is_datetime(string: str) -> bool:
        """
        Return True if string can be interpreted as a date(time).

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

        # Casting `datetime` objects to string will result in valid ISO-8601
        if isinstance(rand_val, (datetime.date, datetime.datetime)):
            return True

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
    def _get_random_value(self, feature_name: str, no_nulls: bool = False) -> Any:
        """Retrieve a random value from the data."""

    @abstractmethod
    def _has_unique_constraint(self, feature_name: str) -> bool:
        """Return whether this feature has a unique constraint."""

    @abstractmethod
    def _get_first_non_null(self, feature_name: str) -> Any:
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
    def _get_unique_values(self, feature_name: str) -> Collection[Any]:
        """Get a set of the unique values for the given feature_name."""

    @abstractmethod
    def _get_row_count(self) -> int:
        """Get the total number of rows in the data."""

    @abstractmethod
    def _get_value_count(self, feature_name: str, value: Any) -> int:
        """Get the number of occurrences of the provided value of the provided feature."""

    def _get_distinct_value_count(self, feature_name: str) -> int:
        """Get the number of distinct values of a feature, counting every null form as one value."""
        return self._get_unique_count(feature_name) + (1 if self._contains_nulls(feature_name) else 0)

    def _significance_threshold(self, feature: str, max_distilled_cases: int,
                                significance_threshold: int | None) -> tuple[int, Callable[[int], int]]:
        """
        Resolve the significance threshold of a feature for a distillation target.

        Parameters
        ----------
        feature : str
            The name of the feature.
        max_distilled_cases : int
            The distillation target, as :func:`get_optimized_max_chunk_size` rounds it.
        significance_threshold : int, optional
            A threshold given by the user, which applies to every feature at every target.

        Returns
        -------
        int
            The threshold for `max_distilled_cases`.
        Callable of int to int
            The threshold for any other distillation target, for the search for a target at which
            every rare value fits.
        """
        if significance_threshold is not None:
            return significance_threshold, lambda _: significance_threshold
        total_cases = self._get_row_count()
        distinct_values = self._get_distinct_value_count(feature)
        return (_dynamic_significance_threshold(total_cases, max_distilled_cases, distinct_values),
                lambda target: _dynamic_significance_threshold(total_cases, target, distinct_values))

    def _find_protected_value_candidates(
        self,
        max_distilled_cases: int,
        significance_threshold: int | None,
        *,
        features: Container[str] | None = None,
    ) -> tuple[PreserveRareValuesMap, list[dict]]:
        """
        Analyze the data to determine if any values might be good candidates for signal preservation techniques.

        Parameters
        ----------
        max_distilled_cases : int
            The maximum number of cases in the resultant data following distillation.
        significance_threshold : int, optional
            The number of cases that are expected to result in a maintained signal for a particular
            value post-distillation. Computed per feature when not given.
        features : Container of str, optional
            The features to search. Defaults to every nominal feature.

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
            if attributes["type"] != "nominal" or (features is not None and feature not in features):
                continue
            total_cases = self._get_row_count()
            if self._get_unique_count(feature) == total_cases:
                # Don't make a suggestion for a completely unique feature
                continue
            threshold, _ = self._significance_threshold(feature, max_distilled_cases, significance_threshold)
            # Every null form (None, NaN, NA) is counted together, so it is one candidate, None
            null_seen = False
            for unique_value in self._get_unique_values(feature):
                value = unique_value
                if is_null_value(unique_value):
                    if null_seen:
                        continue
                    null_seen = True
                    value = None
                try:
                    count = self._get_value_count(feature, value)
                except TypeError:
                    self.warnings_collector.triage(IFAWarningEmitterType.VALUE_COUNTS_PROCESSING, feature)
                    continue
                # Don't include values that aren't significant to begin with
                if count < threshold:
                    continue
                expected_freq_at_target_size = (max_distilled_cases / total_cases) * count
                if expected_freq_at_target_size < threshold:
                    value_counts.append({"feature": feature, "value": value, "count": count})
                    if feature not in pvm:
                        pvm[feature] = [value]
                    else:
                        pvm[feature].append(value)
        top_five = sorted(value_counts, key=lambda d: d["count"], reverse=True)[:5]
        return pvm, top_five

    def _reweight_rare_values(  # noqa: PLR0912, PLR0915
        self,
        feature: str,
        targets: Sequence[ProtectedValueMultiplier],
        significance_threshold: int,
        floor: float,
        *,
        fit: Literal["largest_first", "scale"],
        cap: float | None = None,
        threshold_at: Callable[[int], int] | None = None,
    ) -> RareValueReweighting:
        """
        Compute the case weight multipliers of a feature that fund the preservation of its rare values.

        The `targets` are the rare values to weight up, each to its given multiplier. Values with
        at least `floor` cases are significant: these are all scaled by one common factor, except
        that none is scaled below `floor` cases, so a value that is significant before
        distillation stays significant after it. With a `cap`, the factor is no smaller than
        ``1 - cap``. The factor is chosen so the total case weight of the feature is unchanged.
        Every other value, whether it has fewer than `significance_threshold` cases or falls
        between that and the floor, keeps a multiplier of 1.

        When the significant values cannot fund every target within those limits, `fit` decides
        what happens: ``"largest_first"`` keeps the targets with the most cases and leaves the
        rest at a multiplier of 1, while ``"scale"`` keeps every target and shrinks each one's
        increase over 1 by a common factor.

        Parameters
        ----------
        feature : str
            The name of the feature.
        targets : Sequence of ProtectedValueMultiplier
            The rare values and the multipliers they should receive, each at least 1. Values must
            already be in stored form, as :func:`_as_feature_value` gives them: nulls as None and
            numpy scalars as Python types.
        significance_threshold : int
            The number of cases below which a value is small.
        floor : float
            The number of cases a significant value must keep, ``significance_threshold`` divided
            by the reduction ratio of distillation, or 0 when no distillation target is known.
        fit : {"largest_first", "scale"}
            How to reconcile the targets with the available case weight when they do not fit.
        cap : float, optional
            The largest share of its weight a significant value may give up. Without a cap, the
            significant values give up as much as the targets require, down to the floor.
        threshold_at : Callable of int to int, optional
            The significance threshold at any distillation target, for reporting the target at
            which every rare value would fit. Defaults to `significance_threshold` at every target.

        Returns
        -------
        FeatureRareValueConfig or None
            The feature's configuration: under ``value_weight_multipliers``, the kept targets and
            every significant value, whether scaled by the common factor or held at the floor. Values
            that keep a multiplier of 1 are not listed. Every listed value keeps at least `floor`
            cases of weight, so a configuration lists at most ``total_cases / floor`` values, which
            is the distillation target divided by the significance threshold. None when the
            feature has no significant values to take case weight from, or no target could be
            funded.
        list of ProtectedValueMultiplier
            The kept targets alone.
        RareValuePreservationLimit or None
            How the targets were limited, or None when every target received its multiplier.
        """
        total_cases = self._get_row_count()
        target_values = [target["value"] for target in targets]
        null_target, hashable_targets, unhashable_targets = _bucket_protected_values(target_values)
        target_counts = [self._get_value_count(feature, value) for value in target_values]
        # The smallest factor a significant value may be scaled by; without a cap, only the floor limits it
        min_multiplier = 1 - cap if cap is not None else 0.0

        # The algorithm, in four steps:
        #   1. Sort every non-target value into "unchanged" (keeps weight 1) or "significant" (can give
        #      weight up). A significant value has at least `floor` cases, so it can lose some and still
        #      keep `significance_threshold` after distillation.
        #   2. Compare what the targets need ("deficits") with what the significant values can give
        #      ("budgets"), and decide which targets are funded.
        #   3. Find the one factor that scales the significant values so their total loss equals the
        #      funded deficits, holding any value that the factor would push under the floor at the floor.
        #   4. List every value whose multiplier differs from 1: the funded targets and the significant
        #      values, each at the factor or held at the floor.

        # Step 1: classify the other values. A value is unchanged when it is small (fewer than
        # `significance_threshold` cases) or already below the floor, since it has no weight to spare and
        # is not a target to lift. Nulls are counted together as the single value None.
        null_seen = False
        unchanged_counts: list[int] = []
        significant: list[tuple[Any, int]] = []
        for unique_value in self._get_unique_values(feature):
            try:
                if is_null_value(unique_value):
                    if null_target or null_seen:
                        continue
                    null_seen = True
                    value = None
                else:
                    value = _as_feature_value(unique_value)
                    try:
                        is_target = value in hashable_targets
                    except TypeError:
                        is_target = False
                    if is_target or any(value == target for target in unhashable_targets):
                        continue
                count = self._get_value_count(feature, value)
            except (TypeError, ValueError):
                self.warnings_collector.triage(IFAWarningEmitterType.VALUE_COUNTS_PROCESSING, feature)
                continue
            if count < significance_threshold or count < floor:
                unchanged_counts.append(count)
            else:
                significant.append((value, count))
        # A value below the floor at this target can donate at a larger one, so the search for a target
        # that fits every rare value considers all non-target values
        all_other_counts = np.array([count for _, count in significant] + unchanged_counts, dtype=float)
        if not significant:
            # Nothing can give up weight at this target, so nothing is preserved; report that rather
            # than dropping the request silently
            limit: RareValuePreservationLimit = {
                "feature": feature,
                "preserved": 0,
                "candidates": len(targets),
                "min_max_distilled_cases": self._min_distilled_cases_to_fit(
                    target_counts=np.array(target_counts, dtype=float),
                    other_counts=all_other_counts,
                    total_cases=total_cases,
                    threshold_at=threshold_at or (lambda _: significance_threshold),
                    min_multiplier=min_multiplier,
                ),
                "multiplier_scale": None,
            }
            return None, [], limit
        # Ascending by count: step 3 relies on the values held at the floor being the smallest ones
        significant.sort(key=lambda item: item[1])
        counts = np.array([count for _, count in significant], dtype=float)

        # Step 2: budgets and deficits, both in cases of weight. A significant value can drop to the
        # floor, or to `min_multiplier` times its count, whichever is higher. A target's deficit is the
        # weight its multiplier adds on top of its own count.
        budgets = counts - np.maximum(floor, min_multiplier * counts)
        budget = float(budgets.sum())
        # Every comparison against a budget or the floor below allows `tolerance` amount of slack. The
        # quantities compared are computed along different paths that are equal in exact arithmetic but
        # not in floating point: a target's deficit is `count * (floor / count - 1)`, while the budget it
        # is compared with is `count - floor` of some other value, and a value scaled by the factor is
        # `(remaining / mass) * count`, while the floor it must reach is `floor` itself. At an exact
        # boundary, such as a deficit that uses the whole budget, the two sides can differ by a few ulps
        # in either direction, and a strict comparison would then drop a target that fits or fail to
        # find a factor. The slack is relative to the floor, the natural unit of weight here, and is far
        # below one case. A target admitted within the slack is funded from the budget alone: its
        # increase over 1 is scaled by the share of the deficit the budget covers, which is at most the
        # slack relative, so the total weight stays exactly conserved.
        tolerance = 1e-9 * max(floor, 1.0)
        deficits = [count * (target["multiplier"] - 1) for count, target in zip(target_counts, targets, strict=True)]
        needed = sum(deficits)
        kept: list[ProtectedValueMultiplier] = []
        kept_indices: set[int] = set()
        limit: RareValuePreservationLimit | None = None
        if needed <= budget + tolerance:
            # Everything fits: every target gets its multiplier
            kept = [{"value": target["value"], "multiplier": target["multiplier"]} for target in targets]
            kept_indices = set(range(len(targets)))
            used = needed
        elif fit == "largest_first":
            # Fund targets in order of their counts until the next one would exceed the budget; the
            # rest stay at weight 1. The limit records what was dropped and the distillation target
            # at which everything would have fit.
            order = sorted(range(len(targets)), key=lambda i: target_counts[i], reverse=True)
            used = 0.0
            for i in order:
                if used + deficits[i] <= budget + tolerance:
                    kept.append({"value": targets[i]["value"], "multiplier": targets[i]["multiplier"]})
                    kept_indices.add(i)
                    used += deficits[i]
            limit = {
                "feature": feature,
                "preserved": len(kept),
                "candidates": len(targets),
                "min_max_distilled_cases": self._min_distilled_cases_to_fit(
                    target_counts=np.array(target_counts, dtype=float),
                    other_counts=all_other_counts,
                    total_cases=total_cases,
                    threshold_at=threshold_at or (lambda _: significance_threshold),
                    min_multiplier=min_multiplier,
                ),
                "multiplier_scale": None,
            }
        else:
            # Keep every target but shrink each multiplier's increase over 1 by the same factor, so the
            # targets together use exactly the budget and keep their proportions to one another. The
            # factor is `budget / needed`, but a multiplier near the float limit makes `needed` overflow
            # to infinity, so the increases are measured relative to the largest one: those ratios and
            # their weighted sum are finite, and each target's share of the budget follows from them
            # without forming `needed` itself.
            increases = [target["multiplier"] - 1 for target in targets]
            largest_increase = max(increases)
            relative_increases = [increase / largest_increase for increase in increases]
            relative_needed = sum(count * relative for count, relative in zip(target_counts, relative_increases,
                                                                               strict=True))
            kept = [{"value": target["value"], "multiplier": 1 + relative * budget / relative_needed}
                    for target, relative in zip(targets, relative_increases, strict=True)]
            kept_indices = set(range(len(targets)))
            used = budget
            limit = {"feature": feature, "preserved": len(kept), "candidates": len(targets),
                     "min_max_distilled_cases": None,
                     "multiplier_scale": budget / relative_needed / largest_increase}
        if used <= 0:
            # No target could be funded; the limit, if any, still tells the caller what was dropped
            return None, [], limit
        if used > budget:
            # The kept targets were admitted within the slack: fund them from the budget alone, so the
            # significant values are left exactly their floors in step 3 and the total weight is conserved
            squeeze = budget / used
            for entry in kept:
                entry["multiplier"] = 1 + (entry["multiplier"] - 1) * squeeze
            used = budget

        # Step 3: the significant values must end with `remaining` cases of weight in total. Scaling
        # all of them by one factor `s` would give sum(s * count), but any value with s * count below
        # the floor is held at the floor instead, so the total is
        #     k * floor + s * (sum of the counts above the floor)
        # where k is the number of values held at the floor. Because the counts are sorted, those k
        # values are the first k. Walk k upward: solve the equation for s assuming the first k values
        # are held at the floor, and stop at the first k whose next value keeps at least the floor
        # under that s. If it falls short, it is held at the floor too, and the walk continues. The
        # budget guarantees `remaining` covers the floors, so the walk ends with a consistent s unless
        # every value is held at the floor, in which case s is applied to no case.
        remaining = float(counts.sum()) - used
        # suffix_mass[k] is the summed count of the values from index k onward
        suffix_mass = np.cumsum(counts[::-1])[::-1]
        held = len(counts)
        factor = 1.0
        for k in range(len(counts)):
            candidate = (remaining - k * floor) / suffix_mass[k]
            # When the budget is used up exactly, this value lands on the floor, and the quotient times
            # its count can come out a few ulps under it; without the slack the walk would hold it at
            # the floor too, and so on for every value, ending with no factor applied to any case
            if candidate * counts[k] >= floor - tolerance:
                held = k
                factor = float(candidate)
                break

        # Step 4: assemble the configuration. The Engine applies a multiplier of 1 to every case whose
        # value is not listed, so every value whose multiplier is 1 is left out: the targets that were
        # not funded and the unchanged values. Listed are the funded targets, the significant values
        # held at the floor (at floor / count, which is above the factor) and the significant values
        # scaled by the factor.
        donors: list[ProtectedValueMultiplier] = [
            {"value": value, "multiplier": float(floor / count)} for (value, count) in significant[:held]
        ]
        donors.extend({"value": value, "multiplier": float(factor)} for (value, _) in significant[held:])
        config: FeatureRareValueConfig = {
            "value_weight_multipliers": [entry for entry in kept + donors if entry["multiplier"] != 1]
        }
        return config, kept, limit

    @staticmethod
    def _min_distilled_cases_to_fit(target_counts: np.ndarray, other_counts: np.ndarray, total_cases: int,
                                    threshold_at: Callable[[int], int], min_multiplier: float) -> int | None:
        """
        Find the smallest `max_distilled_cases` at which every rare value of a feature can be preserved.

        Parameters
        ----------
        target_counts : np.ndarray
            The case counts of the rare values to preserve.
        other_counts : np.ndarray
            The case counts of every other value of the feature. Which of them can donate is decided
            at each searched target, since the floor falls as the target grows.
        total_cases : int
            The number of cases in the data.
        threshold_at : Callable of int to int
            The number of cases each value should keep after distillation, for a distillation target
            as :func:`get_optimized_max_chunk_size` rounds it.
        min_multiplier : float
            The smallest multiplier the feature's significant values may receive.

        Returns
        -------
        int or None
            The smallest distillation target whose floor lets the other values fund every rare
            value, at most `total_cases`, or None when no target does. The floor is computed from
            the target as :func:`get_optimized_max_chunk_size` rounds it, the same way
            `infer_feature_attributes` treats a `max_distilled_cases` argument.
        """
        counts = np.concatenate([target_counts, other_counts])

        def fits(max_distilled_cases: int) -> bool:
            max_distilled_cases, _ = get_optimized_max_chunk_size(row_count=total_cases,
                                                                  max_chunk_size=max_distilled_cases)
            floor = threshold_at(max_distilled_cases) * total_cases / max_distilled_cases
            # Targets below the floor need funding; any value at or above it can give some up
            needed = float((floor - target_counts[target_counts < floor]).sum())
            significant = counts[counts >= floor]
            budget = float((significant - np.maximum(floor, min_multiplier * significant)).sum())
            return needed <= budget

        if not fits(total_cases):
            return None
        low, high = 1, total_cases
        while low < high:
            mid = (low + high) // 2
            if fits(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def _compute_preserve_rare_values_config(
        self,
        max_distilled_cases: int,
        values_map: PreserveRareValuesMap,
        significance_threshold: int | None,
        caps: Mapping[str, float] | None = None,
    ) -> tuple[FullPreserveRareValuesConfig, PreserveRareValuesMap, list[RareValuePreservationLimit]]:
        """
        Determine the case weight multipliers for the provided rare values and the other values of their features.

        Each rare value is weighted so that it keeps `significance_threshold` cases after
        distillation, funded by the feature's significant values as described in
        :meth:`_reweight_rare_values`. When not every rare value fits, those with the most cases
        are preserved.

        Parameters
        ----------
        max_distilled_cases : int
            The maximum number of cases in the resultant data following distillation.
        values_map : PreserveRareValuesMap
            A mapping of feature name to list of rare values to compute multipliers for.
        significance_threshold : int, optional
            The number of cases that are expected to result in a maintained signal for a
            particular value post-distillation. Computed per feature when not given.
        caps : Mapping of str to float, optional
            The largest share of its weight a significant value of each listed feature may give up.

        Returns
        -------
        FullPreserveRareValuesConfig
            A `preserve_rare_values` configuration per feature, ready for application to the
            feature attributes.
        PreserveRareValuesMap
            The rare values of each configured feature that were weighted up.
        list of RareValuePreservationLimit
            The features whose rare values could not all be preserved.
        """
        prvc: FullPreserveRareValuesConfig = {}
        protected_map: PreserveRareValuesMap = {}
        limits: list[RareValuePreservationLimit] = []
        caps = caps or {}
        total_cases = self._get_row_count()
        for feature, values in values_map.items():
            if feature not in self.attributes:
                # Multiprocessing is enabled, and this feature will be handled in another process
                continue
            threshold, threshold_at = self._significance_threshold(feature, max_distilled_cases,
                                                                   significance_threshold)
            floor = threshold * total_cases / max_distilled_cases
            targets: list[ProtectedValueMultiplier] = []
            for value in _dedupe_values(values):
                count = self._get_value_count(feature, value)
                if count == 0:
                    raise ValueError(f"Specified protected value `{value}` not found in column `{feature}`. "
                                     "Please verify the value and type.")
                multiplier = floor / count
                # A value that keeps the threshold on its own does not need signal preservation
                if multiplier <= 1:
                    continue
                targets.append({"value": _as_feature_value(value), "multiplier": float(multiplier)})
            if not targets:
                continue
            config, kept, limit = self._reweight_rare_values(
                feature=feature,
                targets=targets,
                significance_threshold=threshold,
                floor=floor,
                fit="largest_first",
                cap=caps.get(feature),
                threshold_at=threshold_at,
            )
            if limit is not None:
                limits.append(limit)
            if config is not None:
                prvc[feature] = config
                protected_map[feature] = [entry["value"] for entry in kept]
        return prvc, protected_map, limits

    def _process_rare_values(  # noqa: PLR0912, PLR0915
        self,
        preserve_rare_values: PreserveRareValues | None,
        max_distilled_cases: int | None,
        significance_threshold: int | None,
        enable_suggestions: bool = True,
        *,
        caps: Mapping[str, float] | None = None,
    ) -> None:
        """
        Compute the value weight multipliers that `preserve_rare_values` asks for, or suggest some.

        Each feature's specification is handled by its form: a complete configuration is written as
        given; values paired with multipliers are funded by the feature's other values; plain values
        get their multipliers computed. Without `max_distilled_cases`, the multipliers are computed
        for :data:`DEFAULT_MAX_DISTILLED_CASES` and a warning says so. "all" and a list of feature
        names select the rare value candidates of the features and treat them as plain values; both
        require `max_distilled_cases`. Without a specification, candidates are found and offered as a
        suggestion. "off" does nothing at all.
        """
        caps = caps or {}
        # User wants to do nothing; exit silently
        if isinstance(preserve_rare_values, str) and preserve_rare_values == "off":
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
        user_set_mdc = max_distilled_cases is not None
        if max_distilled_cases is None:
            # Consistent with Enterprise; the suggestion reports that it was assumed
            max_distilled_cases = DEFAULT_MAX_DISTILLED_CASES
        requested_max_distilled_cases = max_distilled_cases
        # The target as distillation will apply it
        max_distilled_cases, _ = get_optimized_max_chunk_size(row_count=self._get_row_count(),
                                                              max_chunk_size=max_distilled_cases)

        if preserve_rare_values is None:
            # Nothing was asked for: find candidates and offer them as a suggestion, applying nothing.
            # Data smaller than the distillation target (true for many test cases) would not be reduced
            if not enable_suggestions or self._get_row_count() < requested_max_distilled_cases:
                return
            candidates, values_ranking = self._find_protected_value_candidates(
                max_distilled_cases=max_distilled_cases, significance_threshold=significance_threshold)
            if candidates:
                candidate_prvc, protected_map, limits = self._compute_preserve_rare_values_config(
                    max_distilled_cases=max_distilled_cases,
                    values_map=candidates,
                    significance_threshold=significance_threshold,
                    caps=caps,
                )
                if candidate_prvc or limits:
                    # A suggestion is made even when nothing could be funded, so that its caveats report why
                    self.suggestions_collector.append(PRVSuggestion(candidate_prvc, values_ranking, user_set_mdc,
                                                                    protected_values=protected_map, limits=limits))
            return

        given: PreserveRareValuesConfig = {}
        full: FullPreserveRareValuesConfig = {}
        if isinstance(preserve_rare_values, Mapping):
            values_map, given, full = _split_rare_values(preserve_rare_values)
        else:
            # "all" or a list of feature names selects the rare value candidates of those features
            if not user_set_mdc:
                raise ValueError("If not explicitly providing rare values to preserve, you must also provide "
                                 "`max_distilled_cases` to accurately determine rare value candidates.")
            search_features = None if preserve_rare_values == "all" else _rare_value_features(preserve_rare_values)
            for search_feature in search_features or []:
                if search_feature in self.attributes and self.attributes[search_feature]["type"] != "nominal":
                    actual_type = self.attributes[search_feature]["type"]
                    raise ValueError(f"`preserve_rare_values` names the feature `{search_feature}`, which was "
                                     f"inferred to be {actual_type}; rare values can only be set for nominal "
                                     "features. If this feature is actually nominal, please override the "
                                     "inference with the `types` parameter.")
            values_map, _ = self._find_protected_value_candidates(
                max_distilled_cases=max_distilled_cases,
                significance_threshold=significance_threshold,
                features=search_features,
            )

        prvc: FullPreserveRareValuesConfig = {}
        # A complete configuration is used as-is
        for feature, cfg in full.items():
            if feature in self.attributes:
                prvc[feature] = deepcopy(cfg)
        # Values paired with multipliers are funded by the other values of their features
        if given:
            prvc.update(self._fund_given_multipliers(
                given=given,
                max_distilled_cases=max_distilled_cases,
                significance_threshold=significance_threshold,
                caps=caps,
            ))
        # Plain values get their multipliers computed
        computed, _, limits = self._compute_preserve_rare_values_config(
            max_distilled_cases=max_distilled_cases,
            values_map=values_map,
            significance_threshold=significance_threshold,
            caps=caps,
        )
        prvc.update(computed)
        for limit in limits:
            self.warnings_collector.triage(IFAWarningEmitterType.SIMPLE,
                                           partial_rare_value_preservation_message(limit))
        if not user_set_mdc:
            # Whether a value needs preservation, and by how much, depends on the target, so every feature
            # weighted against the assumed one is named, including those that needed nothing at it
            assumed = [feature for feature, spec in (*values_map.items(), *given.items())
                       if spec and feature in self.attributes]
            if assumed:
                names = ", ".join(f"`{feature}`" for feature in assumed)
                self.warnings_collector.triage(
                    IFAWarningEmitterType.SIMPLE,
                    f"`max_distilled_cases` was not provided, so the rare values of {names} were weighted for an "
                    f"assumed distillation target of {DEFAULT_MAX_DISTILLED_CASES:,} cases. Provide "
                    "`max_distilled_cases` if you will distill your data to a different size."
                )

        for feature, config in prvc.items():
            # A feature missing here is handled in another process when multiprocessing is enabled
            if feature in self.attributes:
                self.attributes[feature]["value_weight_multipliers"] = config["value_weight_multipliers"]

    def _fund_given_multipliers(
        self,
        given: PreserveRareValuesConfig,
        max_distilled_cases: int,
        significance_threshold: int | None,
        caps: Mapping[str, float],
    ) -> FullPreserveRareValuesConfig:
        """
        Fund the multipliers a user gave for rare values with the other values of their features.

        Parameters
        ----------
        given : PreserveRareValuesConfig
            The rare values of each feature with the multipliers they should receive, as
            :func:`_split_rare_values` validates and deduplicates them.
        max_distilled_cases : int
            The distillation target, as :func:`get_optimized_max_chunk_size` rounds it.
        significance_threshold : int, optional
            The threshold given by the user, or None to compute one per feature.
        caps : Mapping of str to float
            The largest share of its weight a significant value of each listed feature may give up.

        Returns
        -------
        FullPreserveRareValuesConfig
            The configuration of each feature whose multipliers could be funded.

        Raises
        ------
        ValueError
            If a value is not in the data.
        """
        prvc: FullPreserveRareValuesConfig = {}
        total_cases = self._get_row_count()
        for feature, value_cfgs in given.items():
            if feature not in self.attributes:
                # Multiprocessing is enabled, and this feature will be handled in another process
                continue
            targets: list[ProtectedValueMultiplier] = []
            for value_cfg in value_cfgs:
                if self._get_value_count(feature, value_cfg["value"]) == 0:
                    raise ValueError(f"Specified protected value `{value_cfg['value']}` not found in column "
                                     f"`{feature}`. Please verify the value and type.")
                targets.append({"value": _as_feature_value(value_cfg["value"]),
                                "multiplier": float(value_cfg["multiplier"])})
            if not targets:
                # Every pair asked for a multiplier of 1, which is what an unlisted value gets
                continue
            threshold, threshold_at = self._significance_threshold(feature, max_distilled_cases,
                                                                   significance_threshold)
            floor = threshold * total_cases / max_distilled_cases
            config, _, limit = self._reweight_rare_values(
                feature=feature,
                targets=targets,
                significance_threshold=threshold,
                floor=floor,
                fit="scale",
                cap=caps.get(feature),
                threshold_at=threshold_at,
            )
            if limit is not None and limit["multiplier_scale"] is None:
                # No value could donate, so the request was not applied at all
                self.warnings_collector.triage(IFAWarningEmitterType.SIMPLE,
                                               partial_rare_value_preservation_message(limit))
            elif limit is not None:
                self.warnings_collector.triage(
                    IFAWarningEmitterType.SIMPLE,
                    f"The multipliers provided for feature `{feature}` need more case weight than its other "
                    f"values can give up, so each multiplier's increase over 1 was scaled by "
                    f"{limit['multiplier_scale']:.3g}. Reduce the multipliers or protect fewer values."
                )
            if config is None:
                continue
            prvc[feature] = config
        return prvc
