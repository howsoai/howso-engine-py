from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import os
from typing import Any, Literal, Union

from pandas import DataFrame
from typing_extensions import NotRequired, Sequence, TypeAlias, TypedDict


class Cases(TypedDict):
    """Representation of a table of cases."""

    cases: list[list[Any]]
    """Matrix of row and column values."""

    features: list[str]
    """The feature column names."""


class Distances(TypedDict):
    """Representation of a case distances result."""

    case_indices: CaseIndices
    """The corresponding distances case indices."""

    distances: DataFrame
    """The matrix of computed distances."""


class Evaluation(TypedDict):
    """Representation of an Evaluate result."""

    aggregated: Any
    """The aggregated evaluation output."""

    evaluated: dict[str, list[Any]]
    """A mapping of feature names to lists of values."""


class TrainStatus(TypedDict):
    """Representation of a status output from AbstractHowsoClient.train."""

    needs_analyze: NotRequired[bool]
    """Indicates whether the Trainee needs an analyze."""

    needs_data_reduction: NotRequired[bool]
    """Indicates whether the Trainee recommends a call to `reduce_data`."""


class SortByFeature(TypedDict):
    """Represents a single feature sorting directive to apply."""

    feature: str
    """The name of the feature to sort on."""

    order: Literal["asc", "desc"]
    """The direction of the sort."""

class ValueMasses(TypedDict):
    """Represents the computed value masses of a single feature."""

    values: DataFrame
    """A dataframe containing each feature value and its corresponding mass."""

    remaining: float
    """The combined mass of all omitted feature values."""

class ConfusionMatrix(TypedDict):
    """Represents a confusion matrix of a reaction."""

    matrix: DataFrame
    """Index of actual classes to columns of predicted classes to frequencies."""

    leftover_correct: int | float
    """Total number of correct predictions for classes that were not statistically significant."""

    leftover_incorrect: int | float
    """Total number of incorrect predictions for classes with any correct but statistically insignificant predictions."""

    other_counts: dict[str, int | float] | int | float
    """Total number of all other statistically insignificant predictions."""


class FeatureAutoDeriveOnTrain(TypedDict, total=False):
    """Configuration for auto deriving feature values based on the other values of the case or series."""

    code: str
    """The Amalgam code used to derive the feature value."""

    code_features: list[str]
    """A list of features needed to derive code."""

    derive_type: Literal["custom", "start", "end"]
    """The train derive operation type."""

    ordered_by_features: list[str]
    """Feature name(s) that define the order of the series."""

    series_id_features: list[str]
    """Feature name(s) whose values are used to identify cases within the same series."""


class FeatureBounds(TypedDict, total=False):
    """Feature bounds, allowed values, and constraints."""

    allow_null: bool
    """Allow nulls to be output, per their distribution in the data. Defaults to True."""

    allowed: list[Any]
    """Explicitly allowed values to be output."""

    constraint: str
    """
    Amalgam constraint code.

    This code logic must evaluate to true for value to be considered valid when this feature is being generated.
    Same format as 'derived_feature_code'.

    Examples:
        - ``(> #f1 0 #f2 0)``: Feature 'f1' value from current (offset 0) data must be bigger than
          feature 'f2' value from current (offset 0) data.
    """

    max: float | str
    """The maximum value to be output. May be a number or date string."""

    min: float | str
    """The minimum value to be output. May be a number or date string."""

    observed_max: float | str
    """The observed maximum value in the data. May be a number, string, or date string."""

    observed_min: float | str
    """The observed minimum value in the data. May be a number, string, or date string."""


class FeatureTimeSeries(TypedDict, total=False):
    """
    Time series options for a feature.

    Configures how a feature behaves in time series, including lag generation,
    derivative orders, and rate/delta boundaries.
    """

    delta_max: list[float | None]
    """
    Maximum difference between feature values.

    If specified, ensures that the largest difference between feature values is not larger than this specified value.
    A null value means no max boundary. The length of the list must match the number of derivatives as specified by
    `order`. Only applicable when time series type is set to `delta`.
    """

    delta_min: list[float | None]
    """
    Minimum difference between feature values.

    If specified, ensures that the smallest difference between features values is not smaller than this specified
    value. A null value means no min boundary. The length of the list must match the number of derivatives as
    specified by `order`. Only applicable when time series type is set to `delta`.
    """

    derived_orders: int
    """
    The number of orders of derivatives that should be derived instead of synthesized.

    Ignored if order is not specified.
    """

    lags: list[int]
    """
    Lag feature offsets to generate.

    If specified, generates lag features containing previous values using the enumerated lag offsets.
    Takes precedence over `num_lags`. If neither `num_lags` nor `lags` is specified for a feature, then a
    single lag feature is generated.
    """

    num_lags: int
    """
    Number of lag features to generate.

    If specified, generates the specified amount of lag features containing previous values. If `lags` is specified,
    then this parameter will be ignored. If neither `num_lags` nor `lags` is specified for a feature, then a single
    lag feature is generated.
    """

    order: int
    """If specified, generates the specified number of derivatives and boundary values."""

    rate_max: list[float | None]
    """
    Maximum rate for the feature.

    If specified, ensures that the rate (the difference quotient, the discrete version of derivative) for this feature
    won't be more than the value specified. A null value means no max boundary. The value must be in epoch format for
    the time feature. The length of the list must match the number of derivatives as specified by `order`.

    Only applicable when time series type is set to `rate`.
    """

    rate_min: list[float | None]
    """
    Minimum rate for the feature.

    If specified, ensures that the rate (the difference quotient, the discrete version of derivative) for this feature
    won't be less than the value specified. A null value means no min boundary. The value must be in epoch format for
    the time feature. The length of the list must match the number of derivatives as specified by `order`.

    Only applicable when time series type is set to `rate`.
    """

    time_feature: bool
    """
    Whether this feature will be treated as the time feature for time series modeling.

    Note: Time features must use type `delta`.
    """

    type: Literal["rate", "delta", "covariate"]
    """
    Time series type.

    - `rate`: Uses the difference of the current value from its previous value divided by the change in time
      since the previous value.
    - `delta`: Uses the difference of the current value from its previous value regardless of the elapsed time.
      (required if `time_feature` is true)
    - `covariate`: Temporal changes are not modeled and feature values are directly predicted with interpolation in
      series generation rather than derived using a rate or delta.
    """

    universal: bool
    """
    Controls whether future values of independent time series are considered.

    Applicable only to the time feature. When false, the time feature is not universal and allows using future data
    from other series in decisions; this is applicable when the time is not globally relevant and is independent for
    each time series. When true, universally excludes using any data with from the future from all series; this is
    applicable when time is globally relevant and there are events that may affect all time series. If there is
    any possibility of global relevancy of time, it is generally recommended to set this value to true, which is
    the default.
    """

class FanoutFeatureGroup(TypedDict):
    """
    Configuration for a single collection of fan-out features.

    Configuration describing both the list of fan-out features and the list
    of "key" features whose values can be used to find the groups of cases that
    all use the same duplicated values.
    """

    key_features: list[str]
    """List of the features whose values can be used to find groups of cases with the same duplicated values."""

    fanout_features: list[str]
    """List of features whose values are fanned out across multiple cases from a single observation."""

class FeatureAttributes(TypedDict):
    """
    Attributes for a single feature.

    Comprehensive configuration for a machine learning feature, including type
    information, bounds, time series settings, and data processing options.
    """

    type: Literal["continuous", "ordinal", "nominal"]
    """
    The type of the feature.

    - continuous: A continuous numeric value (e.g., temperature or humidity)
    - nominal: A numeric or string value with no ordering (e.g., fruit names)
    - ordinal: A nominal numeric value with ordering (e.g., rating scale, 1-5 stars)
    """

    auto_derive_on_train: NotRequired[FeatureAutoDeriveOnTrain]
    """Configuration for auto deriving feature values based on the other values of the case or series."""

    bounds: NotRequired[FeatureBounds]
    """A map defining any feature bounds, allowed values, and constraints."""

    code_features: NotRequired[list[str]]
    """A list of features needed to derive code."""

    cycle_length: NotRequired[int]
    """
    Cyclic feature configuration.

    Sets the upper bound of the difference for the cycle range. For example,
    if `cycle_length` is 360, then values 1 and 359 will have a difference of 2.

    Cyclic features have no input restrictions but output on a scale from 0 to
    `cycle_length`. To constrain output to a different range, modify `min` and
    `max` in the `bounds` attribute.

    Examples:
        - degrees: values 0-359, cycle_length = 360
        - days: values 0-6, cycle_length = 7
        - hours: values 0-23, cycle_length = 24
    """

    data_type: NotRequired[
        Literal[
            "string",
            "number",
            "boolean",
            "formatted_date_time",
            "formatted_time",
            "string_mixable",
            "json",
            "yaml",
            "amalgam",
        ]
    ]
    """
    The data type of a feature.

    Default is `string` for nominals and `number` for continuous.

    Valid values:
        - string: nominal or continuous
        - number: nominal or continuous
        - formatted_date_time: nominal or continuous
        - formatted_time: nominal or continuous
        - json: nominal or continuous
        - yaml: nominal or continuous
        - amalgam: nominal or continuous
        - boolean: nominal only
        - string_mixable: continuous only (predicted values may result in interpolated
          strings containing character combinations from multiple original values)
    """

    date_time_format: NotRequired[str]
    """
    Date format specification.

    Feature values should match the date format specified by this string.
    Only applicable to continuous features.
    """

    decimal_places: NotRequired[int]
    """
    Number of decimal places to round to.

    Default is no rounding. If `significant_digits` is also specified, the number
    will be rounded to significant digits first, then to decimal points.
    """

    default_time_zone: NotRequired[str]
    """The default time zone for datetimes. Defaults to 'UTC' if unspecified."""

    dependent_features: NotRequired[list[str]]
    """
    Features that this feature depends on or that depend on this feature.

    Should be used when there are multi-type value features that tightly depend
    on values based on other multi-type value features.
    """

    fanout_on: NotRequired[list[str]]
    """
    Features whose values can be used to select other cases that have the same
    duplicated value for this fan-out feature.

    Should be used when this is a fan-out feature.
    """

    derived_feature_code: NotRequired[str]
    """
    Code defining how to derive this feature's value.

    Used when this feature is specified as a `derived_context_feature` or
    `derived_action_feature` during react flows. For `react_series`, the data
    referenced is accumulated series data (as a list of rows); for non-series
    reacts, data is a single row. Each row comprises all combined context and
    action features.

    Referencing data uses 0-based indexing where the current row index is 0,
    previous row is 1, etc. The code may perform simple logic and numeric
    operations on feature values referenced via feature name and row offset.

    Examples:
        - ``(call value {feature \"x\" lag 1})``:
          Use value for feature 'x' from the previously processed row (offset 1, one lag value).
        - ``(- (call value {feature \"y\" lag 0}) (call value {feature \"x\" lag 1}))``:
          Feature 'y' value from current row minus feature 'x' from previous row.
    """

    id_feature: NotRequired[bool]
    """
    Whether this is an ID feature.

    Set to true for nominal features containing nominal IDs, specifying that this
    feature should be used to compute case weights for id-based privacy. For time
    series, this feature will be used as the id for each time series generation.
    """

    locale: NotRequired[str]
    """The date time format locale. If unspecified, uses platform default locale."""

    max_row_lag: NotRequired[int]
    """The number of time steps traced back by the maximum lag feature created for this feature."""

    nominal_numbers: NotRequired[bool]
    """
    Controls how numbers are compared in semi-structured features.
    Only applicable to code features (when `data_type` is json/yaml/amalgam).

    Defaults to false, compares similarity of values.
    When true, assumes that all numbers will match only if identical.
    """

    nominal_strings: NotRequired[bool]
    """
    Controls how strings are compared in semi-structured features.
    Only applicable to code features (when `data_type` is json/yaml/amalgam).

    Defaults to true, assumes that all strings will match only if identical.
    When false, uses string edit distance to compare similarity.
    """

    non_sensitive: NotRequired[bool]
    """
    Flag a categorical nominal feature as non-sensitive.

    It is recommended that all nominal features be represented with either an `int-id`
    subtype or another available nominal subtype using the `subtype` attribute. However,
    if the nominal feature is non-sensitive, setting this parameter to true will bypass
    the `subtype` requirement.

    Only applicable to nominal features.
    """

    null_is_dependent: NotRequired[bool]
    """
    How dependent features with nulls are treated during react.

    Specifically affects when they use null as a context value. Only applicable
    to dependent features.

    - When false (default): Feature is treated as a non-dependent context feature.
    - When true for nominal types: Treats null as an individual dependent class
      value; only cases with nulls for this feature's value will be considered.
    - When true for continuous types: Only the cases with same dependent feature values
      as the cases that also have nulls as this feature's value will be considered.
    """

    observational_error: NotRequired[float]
    """Specifies the observational mean absolute error for this feature. Use when the error value is already known."""

    original_format: NotRequired[dict[str, Any]]
    """
    Original data format details.

    Automatically populated by clients to store client language-specific
    context about features.
    """

    original_type: NotRequired[dict[str, Any]]
    """
    Original data type details.

    Used by clients to determine how to serialize and deserialize feature data.
    """

    parent: NotRequired[str]
    """The feature whose values this time-series feature's values are derived from."""

    parent_type: NotRequired[Literal["delta", "rate", "covariate"]]
    """The type of time-series processing used by the parent feature."""

    post_process: NotRequired[str]
    """Custom Amalgam code that is called on resulting values of this feature during react operations."""

    recursive_matching: NotRequired[bool]
    """
    Whether operations work recursively on feature values.

    Only applicable to code features (when `data_type` is json/yaml/amalgam).
    Defaults to false for json and yaml features, true for amalgam features.

    When true, operations work recursively on feature values. When false, operates
    on positional matches without considering recursion, yielding better and faster
    results if the schema of semi-structured data is not recursive.
    """

    sample: NotRequired[Any]
    """A sample of a value for the feature."""

    shared_deviations: NotRequired[list[str] | bool]
    """
    Feature names that will share deviations with this feature.

    In analysis, predictions computed for this feature and the specified features are
    combined to create deviations that are used for all of the involved features. If
    this is a time series feature, child lag features automatically share deviations.
    Specifying false, will not share deviations for the automatically created lag features.
    """

    significant_digits: NotRequired[int]
    """Number of significant digits to round to. Default is no rounding."""

    subtype: NotRequired[str]
    """The type used in novel nominal substitution."""

    time_series: NotRequired[FeatureTimeSeries]
    """Time series options for a feature."""

    ts_order: NotRequired[int]
    """The order of rate/delta being described by this time-series feature. Must be >= 0."""

    ts_type: NotRequired[Literal["lag", "delta", "rate"]]
    """The type of value being captured by this time-series feature."""

    types_must_match: NotRequired[bool]
    """
    Defaults to true, when true considers nodes common if their types match.
    Only applicable to code features (when `data_type` is json/yaml/amalgam).
    """

    unique: NotRequired[bool]
    """Flag feature as only having unique values. Only applicable to nominal features."""


FeatureAttributesIndex: TypeAlias = MutableMapping[str, FeatureAttributes]
"""Feature name to feature attribute configuration."""

CaseIndices: TypeAlias = Sequence[tuple[str, int]]
"""Sequence of ``case_indices`` tuples."""

GenerateNewCases: TypeAlias = Literal["always", "attempt", "no"]
"""Valid values for ``generate_new_cases`` parameters."""

LibraryType: TypeAlias = Literal["st", "mt"]
"""Valid values for ``library_type`` parameters."""

Mode: TypeAlias = Literal["robust", "full"]
"""Valid values for ``mode`` parameters."""

NewCaseThreshold: TypeAlias = Literal["max", "min", "most_similar"]
"""Valid values for ``new_case_threshold`` parameters."""

NormalizeMethod: TypeAlias = Literal["fractional_absolute", "fractional", "relative"]
"""Valid values for ``normalize_method`` parameters."""

PathLike: TypeAlias = Union[str, os.PathLike]
"""Objects which can be interpreted as paths."""

Persistence: TypeAlias = Literal["allow", "always", "never"]
"""Valid values for ``persistence`` parameters."""

Precision: TypeAlias = Literal["exact", "similar"]
"""Valid values for ``precision`` parameters."""

SeriesIDTracking: TypeAlias = Literal["fixed", "dynamic", "no"]
"""Valid values for ``series_id_tracking`` parameters."""

SeriesStopMap: TypeAlias = Mapping[str, Mapping[str, Any]]
"""Valid values for ``series_stop_maps`` parameters."""

TabularData2D: TypeAlias = Union[DataFrame, list[list[Any]]]
"""2-dimensional tabular data."""

TabularData3D: TypeAlias = Union[list[DataFrame], list[list[list[Any]]]]
"""3-dimensional tabular (i.e., time-series) data."""

TargetedModel: TypeAlias = Literal["single_targeted", "omni_targeted", "targetless"]
"""Valid values for ``targeted_model`` parameters."""

_ThresholdMeasureKey: TypeAlias = Literal[
    "accuracy",
    "adjusted_smape",
    "mcc",
    "missing_value_accuracy",
    "precision",
    "r2",
    "recall",
    "rmse",
    "smape",
    "spearman_coeff",
]
"""Valid values for ``prediction_stats`` and related parameters."""

AblationThresholdMap: TypeAlias = dict[_ThresholdMeasureKey, dict[str, float]]
"""Threshold map(s) for auto-ablation and data reduction."""
